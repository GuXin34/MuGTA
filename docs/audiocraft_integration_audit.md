# PTC-OPD 接入 AudioCraft 审计

## 0. 审计范围与结论

- 只读源项目：`/Users/suorangu/Desktop/audio_music/ICASSP2027_OPD_Music`
- 重点子仓库：`third_party/audiocraft`
- 子仓库状态：clean
- 子仓库版本：`896ec7c47f5e5d1e5aa1e4b260c4405328bf009d`
- AudioCraft 版本字符：`1.4.0a2`
- 本文中源码行号均指上述 commit；以后换 commit 必须重做对齐测试。

**最终实施结论：PTC-OPD 在独立 workpack 的 `torchrun` runner 中接入，
并在 `LMModel.compute_predictions()` 返回的原始 codec 坐标系
`[B,Q,T,V]` 上计算。** 不新增 AudioCraft solver、不走 Dora/Slurm，也不改
`codebooks_patterns.py` 或 Transformer `forward()`。上游 AudioCraft 唯一代码
overlay 是经过 parity 测试的显式 `use_cfg=False` 生成路径。

早期源码审计曾把 `MusicGenSolver` 子类视为候选接入点。实现阶段改用独立
runner，是因为本项目需要 prompt-only batch、精确的四微批全局比率归一化、
optimizer-step checkpoint 和严格拒绝跨机 process group。最终实现入口见
`scripts/train_stage1.py`，核心逻辑见 `src/ptc_opd/`。

主方法在本审计中固定为：

1. 学生以当前无 CFG 条件分布生成 on-policy codec trajectory。
2. 学生和冻结 CFG 教师在同一 trajectory 上做完整序列因果前向。
3. 损失核心布局为 `[B,Q,T,V]`；有效 mask 为 `[B,Q,T]`。
4. 在每个 `(sample, codebook)` 内，仅根据 detached JS 选 top-`rho`。
5. codebook prior `a_q` **不参与 gate 排序**，只作为已选 KL 的归一化权重。

---

## 1. 上游训练路径和张量布局

### 1.1 Solver 训练主路径

| 源码位置 | 类/函数 | 上游行为 | 对 PTC-OPD 的含义 |
|---|---|---|---|
| `audiocraft/solvers/musicgen.py:32` | `MusicGenSolver` | MusicGen 训练 solver | 只作上游语义参考；最终 runner 不继承它，也不改 solver registry |
| `audiocraft/solvers/musicgen.py:140-169` | `build_model()` | 装入 EnCodec，校验 `n_q/card`，创建 LM 并初始化 optimizer | 冻结学生 conditioner 时要注意 optimizer 已在这里创建 |
| `audiocraft/solvers/musicgen.py:259-361` | `_prepare_tokens_and_attributes()` | 把真实音频编成 `[B,Q,T]`，同时对条件做 dropout | OPD 只用 caption 生成 trajectory，不应调此函数去编码原音频 |
| `audiocraft/solvers/musicgen.py:315-318` | 条件预处理 | 依次调 `cfg_dropout` 和 `att_dropout` | OPD 不能沿用这一随机路径，否则 rollout 和 KL 重计算的条件可能不同 |
| `audiocraft/solvers/musicgen.py:363-442` | `run_step()` | `compute_predictions` → CE → backward/sync/step | 新 solver 的主要替换点 |
| `audiocraft/solvers/musicgen.py:219-251` | `_compute_cross_entropy()` | 每个 codebook 先求 CE，再以 `1/Q` 等权平均 | 可作为 uniform token objective 的对照，但 OPD 需要 token-level KL/JS |
| `audiocraft/solvers/base.py:517-557` | `common_train_valid()` | 一个 dataloader batch 就调一次 `run_step()` | 原生没有本项目需要的 4 次微批累积语义 |

`_prepare_tokens_and_attributes()` 输出：

- `condition_tensors`: conditioner 输出字典；文本条件主体为
  `(embedding [B,L_text,D], mask [B,L_text])`。
- `audio_tokens`: `[B,Q,T]`。
- `padding_mask`: `[B,Q,T]`。

PTC-OPD 的 batch 中只需从 `infos` 构造
`ConditioningAttributes`，不使用真实音频也不调 EnCodec encode。这既避免无用计算，
也避免把 off-policy 真实 codec token 误当成 rollout。

### 1.2 最关键的密集坐标接口

`audiocraft/models/lm.py:112-117` 定义 `LMOutput`：

```text
logits: [B,Q,T,V]
mask:   [B,Q,T]
```

`audiocraft/models/lm.py:270-321`, `LMModel.compute_predictions()` 完成：

```text
codec codes [B,Q,T]
  -> Pattern.build_pattern_sequence(...)
interleaved codes [B,Q,S]
  -> LMModel.forward(...)
sequence logits [B,Q,S,V]
  -> Pattern.revert_pattern_logits(...)
dense aligned logits [B,Q,T,V] + mask [B,Q,T]
```

上游在 `LMOutput` 注释中明确保证 logits 已和原 codec code 对齐，**不要再做
one-token shift**。`Q` 也已经恢复为原始 EnCodec codebook 索引，因此离线估计的
`a_q` 可直接以 `q=0..Q-1` 应用，不需要根据 delay 再排列。

**这是 PTC-OPD 损失核的唯一规范布局。**

---

## 2. MusicGen codebook pattern 和 mask 语义

### 2.1 官方 MusicGen 配置

`config/model/lm/musicgen_lm.yaml:10-25` 固定：

```yaml
codebooks_pattern:
  modeling: delay
  delay:
    delays: [0, 1, 2, 3]
transformer_lm:
  n_q: 4
  card: 2048
```

`config/solver/musicgen/musicgen_base_32khz.yaml:15-19` 说明 EnCodec 为 32 kHz、
4 个 RVQ codebook、50 codec frame/s。因此 10 s rollout 的密集 codec 时间长度
`T=500`，完整 delay pattern 长度为 `S=T+max(delay)+1=504`。

### 2.2 Pattern 源码

- `audiocraft/modules/codebooks_patterns.py:21-53`: `Pattern`。
- `.../codebooks_patterns.py:120-179`: 密集 `[B,Q,T]` 构造 pattern `[B,Q,S]`。
- `.../codebooks_patterns.py:181-223`: 构造恢复索引和 mask。
- `.../codebooks_patterns.py:250-269`: logits 恢复到 `[B,V,Q,T]`，之后
  `compute_predictions()` 转成 `[B,Q,T,V]`。
- `.../codebooks_patterns.py:305-356`: `DelayedPatternProvider`。

`keep_only_valid_steps=True` 是 `compute_predictions()` 默认值，也是现有 MusicGen CE
训练使用的语义。对 `T=500` 和 delays `[0,1,2,3]`，pattern mask 的
理论有效数为：

```text
q0: 500, q1: 499, q2: 498, q3: 497
```

最后 `q` 个 delay-tail 位置不参与训练损失。PTC-OPD 必须延续这一语义，
而不是为了补齐 codebook 数量而强行打开 tail。

### 2.3 一个会静默污染损失的陷阱

`compute_predictions()` 用 `NaN` 填充无效 logits（`lm.py:315-318`）。所以不能：

```python
loss = (token_kl * valid_mask).sum()  # 错：0 * NaN 仍是 NaN
```

应先以 bool mask gather 有效 vocabulary logits，然后才做 float32
`log_softmax`/KL/JS：

```text
valid = student_mask & teacher_cond_mask & teacher_uncond_mask & rollout_mask
student_valid = student_logits[valid]       # [N_valid,V]
cond_valid    = teacher_cond_logits[valid]  # [N_valid,V]
uncond_valid  = teacher_uncond_logits[valid]
```

不要先对全 `[B,Q,T,V]` 做 softmax 再乘 mask。

---

## 3. CFG logits 的正确复用方式

### 3.1 上游 CFG 定义

`audiocraft/models/lm.py:323-418`, `LMModel._sample_next_token()` 实现普通 CFG。
当 conditional/unconditional logits 为 `l_c,l_u` 时：

```text
l_CFG = l_u + w * (l_c - l_u)
```

默认的单次批处理路径会把 sequence 在 batch 维复制两份，一次前向后
split conditional/unconditional（`lm.py:389-401`）。

`LMModel.generate()` 在 `lm.py:420-587`：

- 有 `@torch.no_grad()`。
- 要求 `model.training == False`。
- 在 `479-509` 构造 conditional + null conditions。
- 在 `521-576` 依据 delay pattern 流式生成，最后恢复为 `[B,Q,T]`。

还有一个需要避开的上游细节：`two_step_cfg` 分支在
`lm.py:378-387` 组合 logits 时使用 `self.cfg_coef`，而不是函数局部的
`cfg_coef`。因此在 CFG scale 开发 sweep 中固定 `two_step_cfg=false`，除非先修复
并做 parity test。

### 3.2 教师不要逐 token 重跑

教师不参与 rollout 采样，只在学生已生成的 trajectory 上评分。由于 LM 是
causal，对整个 `[B,Q,T]` 做一次 teacher-forced 前向，就同时得到每个学生
prefix 的 next-token logits。所以教师评分应为：

```text
rollout.repeat(2 on batch) [2B,Q,T]
  + concatenated conditional/null condition tensors
  -> frozen teacher.compute_predictions(...)
  -> split l_c,l_u, each [B,Q,T,V]
  -> l_T = l_u.float() + w * (l_c.float() - l_u.float())
```

这和在 500 个 codec frame 上再跑 500 次教师完全不同；后者不仅慢，还容易
错位 delay pattern。

### 3.3 学生无 CFG rollout 需要一个小补丁

直接调上游 `generate(..., cfg_coef=1.0)` 在数学上得到 conditional logits，可用于
最早的 small 管线 smoke test；但它仍然计算 conditional + unconditional 两份 batch，
且 BF16 中的 `l_u + (l_c-l_u)` 不是真正的 conditional 直通。

在任何正式 pilot 前，应在新工作包的 AudioCraft 副本中给
`LMModel.generate()` 和 `_sample_next_token()` 增加显式 `use_cfg=False`路径：

```text
conditions -> condition_provider -> conditional tensors only
sequence   -> model(..., condition_tensors=conditional tensors) -> logits
```

该补丁只绕过 CFG 批复制，不复制、不重写 pattern 生成循环。在 fixed seed 下必须
通过与 `cfg_coef=1` 的 greedy/logit parity 测试后才进入 medium。

---

## 4. PTC-OPD 损失核的精确实现规格

### 4.1 输入

```text
student_logits:      [B,Q,T,V], requires_grad=True
teacher_cond_logits: [B,Q,T,V], no grad
teacher_null_logits: [B,Q,T,V], no grad
student_mask:        [B,Q,T], bool
teacher masks:       [B,Q,T], bool
rollout_mask:        [B,Q,T], bool
prior:               [Q], float32, positive
rho:                 scalar in (0,1]
teacher_cfg_scale:   scalar
```

MusicGen 主设置中 `Q=4,V=2048,T=500`。

### 4.2 有效 token 上的分布和分数

先 gather 有效 logits，然后以 float32 计算：

```text
l_T    = l_u + w(l_c-l_u)
log pT = log_softmax(l_T, dim=-1)                 # detached
log pS = log_softmax(student_valid.float(), -1)   # keeps gradient

KL_tq = sum_v pT_v (log pT_v - log pS_v)

log m = logaddexp(log pT, stopgrad(log pS)) - log(2)
JS_tq = 0.5 KL(pT || m) + 0.5 KL(stopgrad(pS) || m)
```

JS 整段可放在 `torch.no_grad()` 中，避免构建无用的 gate 反向图。

### 4.3 Gate 是每个 `(b,q)` 独立 top-`rho`

对每个 sample `b` 和 codebook `q`：

```text
n_bq = number of valid t
k_bq = max(1, ceil(rho * n_bq))
g_bqt = 1 for the k_bq largest detached JS values, else 0
```

不是全 batch top-k，不是每个 sample 跨 codebook top-k，也不是
`a_q * JS` top-k。这保证每个 codebook 都保留同比例的时间位置，同时不让
prior 同时改变“选谁”和“权重多大”两个因素。

对完全相等的 JS 预先固定 tie rule（建议 stable descending sort，相等时时间索引
较小者优先），否则 CUDA `topk` 的同分选择可以破坏 bitwise 复现。

### 4.4 Prior 只作已选 KL 的归一化权重

```text
L = sum_bqt valid_bqt * g_bqt * a_q * KL_bqt
    / sum_bqt valid_bqt * g_bqt * a_q
```

- `a_q` 必须先归一化且设正数 floor。
- 分母不能省；否则不同 `rho` 会暗中改变有效学习率。
- uniform OPD：`g=valid`, `a_q=1/Q`。
- codebook-only：`g=valid`, `a_q=perceptual prior`。
- disagreement-only：每 `(b,q)` 的 JS top-`rho`, `a_q=1/Q`。
- PTC：同样的 JS top-`rho` gate，`a_q=perceptual prior`。
- random-50：每 `(b,q)` 保留与 PTC 完全相同的 `k_bq`，`a_q=1/Q`。

---

## 5. 最小可靠代码面（最终实现）

所有新逻辑位于 workpack；旧项目只读。训练机上仅按明确流程把 no-CFG patch
应用到一份可校验的 AudioCraft 工作副本。

### 5.1 已实现

最终没有新增 `PTCOPDSolver`、没有修改 solver registry，也没有新增 Dora/Hydra
配置。规范实现为：

1. `scripts/train_stage1.py`：单机八卡 standalone runner；
2. `src/ptc_opd/{losses,distributed,sampling,audiocraft_adapter,train_utils}.py`：
   可独立测试的核心；
3. `patches/audiocraft/0001-explicit-no-cfg-generation.patch`：唯一上游源码
   overlay；
4. `scripts/check_audiocraft_overlay.py`：补丁状态和源码门禁；
5. `docs/remote_training_runbook.md`：唯一规范远端命令。

下面的 solver-subclass 清单保留为**已否决的早期候选设计**，不得据此创建
另一套训练路径。

<details>
<summary>已否决的候选：AudioCraft solver subclass</summary>

### 5.2 早期候选（不实施）

1. **新增 `audiocraft/solvers/ptc_opd.py`**

   新建 `PTCOPDSolver(MusicGenSolver)`，负责：

   - 从 batch metadata 取 caption attributes，不编码真实音频。
   - 冻结学生 `condition_provider`，创建并冻结教师 LM。
   - 切换学生 `eval()` 生成 no-CFG rollout，然后恢复 `train()`。
   - 在同一 optimizer step 中重计算学生 logits 和教师 CFG logits。
   - 调用独立损失核，记录每 codebook KL/JS/entropy/selection rate。
   - 实现逻辑 optimizer-step 计数、梯度累积和 step checkpoint。
   - 把 `best_metric_name` 改为实际存在的 validation OPD 指标；不能保留上游的
     `ce`，否则 best-state 更新会找不到 metric。

2. **修改 `audiocraft/solvers/builders.py:44-65`**

   在 solver registry 增加 `'ptc_opd': PTCOPDSolver`。

3. **修改 `audiocraft/solvers/__init__.py`**

   导出 `PTCOPDSolver`，保持 solver package 接口完整。

4. **新增 `config/solver/ptc_opd/musicgen_base_32khz.yaml`**

   可复制 `config/solver/musicgen/musicgen_base_32khz.yaml` 的组成逻辑，把最终
   `solver` 改为 `ptc_opd`，并增加结构化 `ptc` 配置：

   ```yaml
   ptc:
     mode: ptc
     retention: 0.5
     teacher_cfg_scale: 3.0
     teacher_checkpoint: ???
     prior_path: ???
     divergence: forward_kl
     selector: js_per_sample_codebook
     accum_steps: 4
     rollout:
       duration_seconds: 10
       use_sampling: true
       temperature: 1.0
       top_k: 250
       top_p: 0.0
       use_cfg: false
     checkpoint_steps: [0, 250, 500, 1000, 2000, 3000]
   ```

5. **修改 `audiocraft/models/lm.py` 的小块 CFG 生成路径**

   仅增加 parity-tested `use_cfg=False` conditional-only generation；不改 pattern，不改
   `compute_predictions()` 输出语义。

6. **新增独立损失核和测试**

   损失核不应埋在 `run_step()` 的大段代码中。它要能用纯合成
   `[B,Q,T,V]` 张量做 CPU/CUDA 单测，不依赖下载 MusicGen。

</details>

### 5.3 明确不改

- `audiocraft/modules/codebooks_patterns.py`
- `LMModel.forward()`
- `LMModel.compute_predictions()` 的对齐语义
- EnCodec 权重和解码逻辑
- 旧项目中的任何文件

---

## 6. 冻结教师和条件的实现边界

### 6.1 T5 配置并不等于整个 conditioner 已冻结

`config/conditioner/text2music.yaml:17-24` 设置 `t5-base`, `finetune:false`,
`word_dropout:0.3`。`audiocraft/modules/conditioners.py:450-515` 表明：

- `finetune:false` 使底层 T5 encoder 不反向。
- conditioner 的 `output_proj` 仍是注册 module，默认会进 optimizer。
- `word_dropout` 在 conditioner 处于 training mode 时仍生效。

因此要满足“文本 conditioner 冻结”的协议，需显式：

```python
student.condition_provider.requires_grad_(False)
```

并在 rollout/条件嵌入计算时处于 eval mode。损失前向应复用已计算且 detached
的 condition tensors，避免 KL 路径再触发 word/CFG/attribute dropout。

### 6.2 教师的加载时机

`StandardSolver.__init__()` 先创建模型和 optimizer，而 `StandardSolver.run()` 在后面才
`restore()` 学生 checkpoint（`audiocraft/solvers/base.py:489-499`）。因此不能在
`build_model()` 中假设学生已经是预训练权重并直接校验学生/教师一致。

建议：

- 教师从独立、不可变的本地 checkpoint 加载。
- 学生 `restore()` 后，在第一次 rollout 前校验 architecture，并对起始权重做一次
  exact/full hash 或严格 state comparison。
- 教师 `eval()` + `requires_grad_(False)`，不注册进 optimizer/EMA/best-state。
- 校验 optimizer 的所有 parameter id 与教师 parameter id 不相交。
- 只在 rank 0 的起点和终点计算完整教师 hash；不要每 step 把 1.5B 权重
  搬回 CPU。

### 6.3 公开 checkpoint 的 revision 问题

`audiocraft/models/loaders.py:40-71` 调用 `hf_hub_download()` 时没有 `revision`参数。
直接使用 `facebook/musicgen-medium` 是“仓库名 pin”，不是“权重 revision pin”。
正式实验应先下载到不可变本地 snapshot，记录 revision 和
`state_dict.bin` SHA-256，然后让学生和教师从同一本地文件读取。

同样需冻结 EnCodec 和 T5 缓存对象。正式运行使用 offline cache，避免 8 个 rank
同时写 Hugging Face cache。

---

## 7. On-policy rollout 的最大技术风险

**最大风险不是生成 API 会立即报错，而是“表面能训，实际不再 on-policy”的静默错误。**

最容易发生的路径是：

1. rollout 时模型在 eval 且条件未 dropout。
2. 重计算 `p_S` 时回到 train，`word_dropout=0.3` 或
   `classifier_free_guidance.training_dropout=0.3` 改变了条件。
3. 或者 rollout 仍走 CFG 两路径，而损失中的学生是 conditional-only。
4. 或者在 rollout 和重计算之间更新了权重/切换了 EMA。

这些情况下，trajectory 并非由 KL 中的那个 `p_S` 生成，但损失仍可以下降，
所以比 OOM 更危险。

必须的防护：

- 在一个逻辑 optimizer step 内完成 rollout 和重计算，中间不 step。
- 冻结 conditioner，在 eval/no-grad 中计算条件，损失前向复用同一条件张量。
- 正式运行只使用显式 conditional-only rollout 路径。
- 记录 rollout 前的 student parameter/global-step id、condition hash、sampling seed 和 token hash。
- 用固定短 prefix 比较流式生成中的 logits 与
  `compute_predictions()` 密集恢复后对应 `(q,t)` logits。

第二大风险是 delay mask/NaN 处理错误；它会导致错位 codebook 或 NaN，已在
第 2 节给出强制规则。

---

## 8. 梯度累积、8 GPU 和 standalone torchrun 接入

本节的 AudioCraft/Dora 分析解释了为什么最终没有复用上游 solver。正式
命令只以 `docs/remote_training_runbook.md` 为准：一台机器一个
`torchrun --standalone --nnodes=1 --nproc_per_node=8` 作业，禁止 Slurm、Dora、
submitit 和 32 卡跨机 process group。

### 8.1 原生 batch size 是全局值

`audiocraft/train.py:38-49` 在创建 solver 前执行：

```python
cfg.dataset.batch_size //= flashy.distrib.world_size()
```

所以上游 Dora 若被采用，单机 8 卡、每 GPU physical batch 2 时会要求：

```text
dataset.batch_size=16
```

不是 2，也不是 64。要得到 effective global batch 64，需 `accum_steps=4`。

### 8.2 上游没有现成的本项目累积语义

`MusicGenSolver.run_step():394-429` 每个 batch 都 backward、sync、optimizer step、
scheduler step 和 zero-grad。所以只在 config 中写 `accum_steps:4` 不会生效。

最终独立 runner 以“逻辑 optimizer step”为单位：

1. 连续处理 4 个 physical microbatch，每次 loss 除以 4。
2. 前 3 次只 backward，不全局通信、不 step。
3. 第 4 次后显式对累积 gradient 做一次 distributed sync，然后 clip/step/schedule/zero。
4. 仅在此处增加 `optimizer_step`，并用它触发 250/500/... checkpoint。
5. `optim.updates_per_epoch`、scheduler total updates、EMA 频率和日志都不能继续把 microbatch
   当 optimizer step。

为减少累积歧义，首批实验建议：

```text
fsdp.use=false
optim.eager_sync=false
optim.ema.use=false
autocast=true
autocast_dtype=bfloat16
```

EMA 如果以后开启，必须按逻辑 optimizer step 更新，并在所有方法中一致。

### 8.3 为什么不走 Dora

`audiocraft/train.py:130-160` 是 Dora 入口，在 `main()` 中调用
`flashy.distrib.init()`。`docs/TRAINING.md:270-273` 说明 `dora run -d` 使用当前可见
GPU 做本地 distributed run。`AUDIOCRAFT_DORA_DIR` 由
`audiocraft/environment.py:103-111` 解析。

上游入口会接管 batch-size 解释、solver 生命周期、scheduler/checkpoint 时机，
而且没有本研究声明的四微批全局比率语义。维护两套入口会让 resume 和方法
对照不可审计。因此 Dora 只作为被审计过的上游背景，不是备用 launcher。
本项目中一个 run 只在一台机器的 8 卡上运行，四台机器并行不同
condition/seed；规范 torchrun 命令见远端 runbook。

---

## 9. 正式运行前必须通过的测试

### 9.1 纯损失核

- 构造含 NaN 的 invalid logits，输出和 gradient 仍必须 finite。
- `rho=1`, `a_q=1/Q` 与所有有效 token 的 uniform KL 数值和 gradient 一致。
- 每 `(b,q)` 实际选中数等于 `ceil(rho*n_bq)`。
- 更改 `a_q` 不改变 gate index，只改变已选 KL 加权。
- random-50 与 PTC 对每 `(b,q)` 的选中数完全相同。
- 分母归一化使 `rho=0.25/0.5/0.75/1` 不自动改变 loss scale。

### 9.2 Pattern/logit 对齐

- `DelayedPatternProvider(4,[0,1,2,3])` 上，`compute_predictions()` 形状为
  `[B,4,T,2048]`。
- `T=500` 时 mask 计数为 `[500,499,498,497]`。
- 对固定短 prefix，流式生成 logits 与密集 `compute_predictions()` 在相同
  `(q,t)` 的 logits 一致。
- 日志中 q0..q3 指原始 EnCodec codebook，而不是 pattern step。

### 9.3 CFG/on-policy

- 教师和学生权重相同、`w=1`、dropout 全关时，float32 KL 近似 0。
- 手动完整序列 CFG 与上游单步 CFG 在短 prefix 上对齐。
- `use_cfg=False` 无 CFG 快速路径与 conditional logits 直接前向对齐。
- rollout 和 student KL 重计算之间 student step/hash 不变，condition hash 不变。
- rollout token 全部在 `[0,V-1]`，有效位不出现 special/unknown token。

### 9.4 冻结和分布式

- 教师全部 `requires_grad=False`，不在 optimizer 中，起点/终点 hash 一致。
- 学生 conditioner 全部无 gradient。
- debug model 上 4 个 microbatch 累积与一个拼接 batch 的 gradient/update 对齐。
- 8 rank 只在 accumulation boundary 做一次 sync，所有 rank 更新后学生 hash 一致。
- checkpoint 名义的 step 是 optimizer step，不是 microbatch index。

---

## 10. 先 small 后 medium 的实施顺序

### Phase A：上游副本和回归基线

1. 在新工作包中准备 commit `896ec7c...` 的 AudioCraft 副本/补丁层。
2. 不带 OPD 先跑官方 debug 和 MusicGen-small 固定 prompt 生成，保存 token/audio hash。
3. 冻结所有公开 checkpoint revision 和 SHA-256，切到 offline cache。

### Phase B：纯张量损失核

1. 实现 valid gather 、float32 CFG/KL/JS。
2. 实现每 `(b,q)` stable top-`rho`。
3. 实现 prior-only normalized weighting 和 random matched gate。
4. 先通过第 9.1 节全部 CPU/CUDA 测试，再接 MusicGen。

### Phase C：MusicGen-small，单 GPU

1. 用独立 runner 的 dry-run 和假模型测试先验证配置/恢复语义。
2. 加入 `use_cfg=False` 直接条件 rollout 补丁并做 parity。
3. 验证 pattern mask 、CFG=1 零 KL、teacher hash 和所有损失消融模式。
4. 跑 2 step 、20 step 和 100 step，不允许 NaN/Inf。

### Phase D：MusicGen-small，单机 8 GPU

1. 实现并验证 4 次 microbatch 累积。
2. 检查 rank seed、梯度 sync、optimizer-step 日志和 step checkpoint。
3. 跑 500 step correctness gate，完整保存 uniform/PTC 的 per-codebook 统计。

### Phase E：MusicGen-medium 上车门

1. 仅替换 small 为 medium，其他配置不改；先跑 1 GPU 单 step 形状/内存门。
2. 单机 8 GPU，physical batch 从 2/GPU 开始，2 update 后再做 50 update 预热。
3. 预热后计时 200 个完整 optimizer step，分别记录 rollout、teacher forward、
   student forward/backward、communication 和 checkpoint 时间。
4. 保留 5--10% peak VRAM 余量，通过后才启动 1,000-step scientific pilot。
5. 先跑 uniform 和 PTC，证明主信号存在后，再扩展 codebook-only/
   disagreement-only/random-50。

### Phase F：三种子正式实验

只使用通过上述门禁的同一代码、配置和 checkpoint snapshot。四台机器每台
运一个独立 8-GPU condition/seed，方法之间轮换物理机器。

---

## 11. 最终建议

1. **损失一定放在 `compute_predictions()` 的 `[B,Q,T,V]` 输出上。**
2. **gate 一定是每 `(sample,codebook)` 内的 detached JS top-`rho`。**
3. **prior 只做已选 KL 的归一化权重，不进 gate score。**
4. **教师用完整序列因果前向一次评估所有 prefix，不逐 token 查询。**
5. **正式 rollout 必须有 explicit no-CFG direct path，并冻结/复用条件张量。**
6. **把梯度累积和 optimizer-step checkpoint 当成 runner 的强制语义，而不是 YAML 参数。**
7. **在 small 通过数值、pattern、CFG、冻结、累积五类门禁前，不开 medium 长跑。**
