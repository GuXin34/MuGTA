# CFG teacher scale 冻结门禁（MusicGen-small / MusicGen-medium）

**当前状态：生成器、两个跨环境 evaluator wrapper、严格 score consumer、决策器
和模型无关的 fixture 测试均已就绪。前一 workpack 已在真实 GPU 上完成 small 和
medium 的 sealed 门禁，本 workpack 依据
`docs/CFG_PRECISION_PROVENANCE_WAIVER_20260814.md` 保留两个 `scale=5` 决策，不重跑。**
两个 wrapper 都完整重哈希 generation artifact，并且只有逐样本推理全部成功后才
原子发布目录。决策器仍然 fail closed，不能用手填分数或总体均值代替。

这个门禁只使用 `dev.full.jsonl`，在 A2 disagreement probe 和任何 OPD 训练
之前冻结一个 teacher CFG scale。`test.full.jsonl` 不得进入该流程。

## 1. 冻结的科学配置

- 模型：`facebook/musicgen-small` 或 `facebook/musicgen-medium`。二者是两套独立实验，
  必须各自生成、评测、决策并写入互不重叠的 artifact 目录，不能共享 decision；
- AudioCraft base：`896ec7c47f5e5d1e5aa1e4b260c4405328bf009d`，并已应用
  `patches/audiocraft/0001-explicit-no-cfg-generation.patch`；
- prompts：`dev.full.jsonl` 的全部且恰好 300 条；
- anchors：显式 `use_cfg=False` 的 `no_cfg`，以及 `CFG={2.0,3.0,5.0}`；
- 每个 prompt 只派生一个稳定 seed，同一个 seed 在四个 anchor 前分别重置；
- sampling：10 秒、`temperature=1.0`、`top_k=250`、`top_p=0`、
  `two_step_cfg=False`、500 codec frames；
- 音频：32 kHz 单声道 IEEE-float WAV，直接写 decoder tensor，不做 loudness
  normalization、clipping 或 rescale；
- bootstrap：按 prompt 成对重采样，10,000 次，seed `4703`。

一轮会生成 `300 × 4 = 1200` 个 WAV，约需 1.54 GB（不含文件系统开销）。

checkpoint 必须是完全解引用后的本地不可变目录；目录及
`state_dict.bin`/`compression_state_dict.bin` 都不能是 symlink。若从 HF cache
复制，使用 `cp -aL`（或等价的 dereference 方式）复制到实验快照目录后再运行。
`model.safetensors` 不在 AudioCraft MusicGen 的必需文件契约内：small snapshot
存在而 medium snapshot 不存在是正常的公开仓库布局差异。验证器要求并实际加载
前述两个 `.bin` 文件，同时分别封存每个 snapshot 当前存在的完整文件树；它不要求
small/medium 具有相同成员集合。

AudioCraft 的非 finetune T5 通过 `__dict__["t5"]` 保存，不是注册子模块。
脚本会单独将它移动到目标 GPU、设为 frozen/eval，并记录实际加载的 T5
state、config 和 tokenizer vocabulary SHA-256。脚本在 import AudioCraft 前强制
`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 和
`HF_DATASETS_OFFLINE=1`；本地缓存缺少 `t5-base` 时直接失败，不允许临时联网换
revision。

所有权重先在 CPU 以 FP32 加载并通过 dtype 检查，再显式移动到 GPU。每个 prompt
在 autocast 外用 FP32 做且只做一次 2B conditioner forward（conditional + null）；
之后 no-CFG 与 CFG 都复用这些张量。LM 自回归生成位于显式 CUDA BF16 autocast
中；EnCodec decode 位于独立的 `torch.autocast(device_type="cuda", enabled=False)`
区域，以 FP32 运行。冻结的精度身份因此是：LM 参数 FP32、LM 生成 BF16、
conditioner/T5 参数与计算 FP32、EnCodec 参数与 decode 计算 FP32。脚本不使用
AudioCraft 默认的 CUDA/FP16 `model.autocast`。不得把 decode 移回 BF16：冻结的
torch 2.1/CUDA 12.1 环境会因 SEANet stacked LSTM 缺少 BF16 fused kernel 而失败。

真实加载后还会硬校验 `Q=4`、vocabulary cardinality `2048`；small 架构为
`dim/layers/heads=1024/24/16`，medium 为 `1536/48/24`，并校验标准 delay pattern
`delays=[0,1,2,3]`。对 `T=500` 构造的 mask 必须是 `[4,504]`，每码本恰有
500 个有效 cell，首末有效 sequence index 分别为 `[1,2,3,4]` 和
`[500,501,502,503]`；mask 本身也写入 SHA-256。

## 2. 生成前检查（不 import AudioCraft）

```bash
export PTC_WORKPACK_ROOT=<local>/ICASSP2027_PTC_OPD_A1R2_Workpack_20260814
export PTC_TRAIN_PY=<local>/envs/ptc-opd-train-py39-cu121/bin/python
export PTC_AUDIOCRAFT_ROOT="${PTC_WORKPACK_ROOT}/vendor/audiocraft"
export PTC_MUSICGEN_SMALL=/absolute/local/snapshot/facebook--musicgen-small
export PTC_MUSICGEN_MEDIUM=/absolute/local/snapshot/facebook--musicgen-medium
export PTC_DEV_MANIFEST="${PTC_WORKPACK_ROOT}/manifests/musiccaps-v1/dev.full.jsonl"
export PTC_MODEL_ID=facebook/musicgen-small
export PTC_MODEL_TAG=small
export PTC_MUSICGEN_CHECKPOINT="${PTC_MUSICGEN_SMALL}"
export PTC_CFG_ROOT="${PTC_WORKPACK_ROOT}/artifacts/cfg_scale/${PTC_MODEL_TAG}"
export PTC_CFG_AUDIO_DIR="${PTC_CFG_ROOT}/generation"

"${PTC_TRAIN_PY}" "${PTC_WORKPACK_ROOT}/scripts/cfg_scale_gate.py" generate \
  --manifest "${PTC_DEV_MANIFEST}" \
  --model-id "${PTC_MODEL_ID}" \
  --checkpoint "${PTC_MUSICGEN_CHECKPOINT}" \
  --audiocraft-root "${PTC_AUDIOCRAFT_ROOT}" \
  --output-dir "${PTC_CFG_AUDIO_DIR}" \
  --device cuda:0 \
  --check-only
```

这里的 AudioCraft 必须是已按 overlay workflow 准备好的 workpack 内副本。
旧项目的 `third_party/audiocraft` 只允许作为固定基线的复制源，不得直接打补丁，
也不得作为 CFG 生成命令的 `--audiocraft-root`。

`--check-only` 校验：manifest basename/count、snapshot 内
`source_row_sha256`、无 symlink 的
`state_dict.bin`/`compression_state_dict.bin`、完整 snapshot hash、AudioCraft
Git base commit、patched `lm.py`、Python source hash，以及排除 Git internals 和
工具 cache 后的完整 patched checkout tree hash。它不创建输出目录，
也不 import AudioCraft。输出里必须是 `"audiocraft_imported": false`。

## 3. 单 GPU 生成并封存

```bash
CUDA_VISIBLE_DEVICES=0 "${PTC_TRAIN_PY}" \
  "${PTC_WORKPACK_ROOT}/scripts/cfg_scale_gate.py" generate \
  --manifest "${PTC_DEV_MANIFEST}" \
  --model-id "${PTC_MODEL_ID}" \
  --checkpoint "${PTC_MUSICGEN_CHECKPOINT}" \
  --audiocraft-root "${PTC_AUDIOCRAFT_ROOT}" \
  --output-dir "${PTC_CFG_AUDIO_DIR}" \
  --device cuda:0
```

`--output-dir` 必须尚不存在。所有内容先在同一父目录的隐藏临时目录生成；只有
1200 条记录、1200 个 WAV、逐文件 hash 和四 anchor 集合全部完整才原子发布。
失败不会留下看似完成的目标目录，重跑也不会覆盖已经发布的结果。

最终目录：

```text
generation/
├── audio/{no_cfg,cfg_2.0,cfg_3.0,cfg_5.0}/*.wav
├── samples.jsonl
├── generation_run.json
└── artifact_seal.json
```

`samples.jsonl` 每行至少封存：`sample_id`、精确 prompt 及其 SHA-256、
`condition`、`cfg_scale`、`condition_id`、配对 seed、相对 WAV path、WAV
SHA-256、checkpoint SHA-256、patched AudioCraft source SHA-256 和最终 scientific
config SHA-256。`generation_run.json` 还记录未注册 T5 的三个运行时 hash。

可在任一后续环境重新核对所有 1200 个文件：

```bash
"${PTC_TRAIN_PY}" "${PTC_WORKPACK_ROOT}/scripts/cfg_scale_gate.py" \
  verify-generation --generation-dir "${PTC_CFG_AUDIO_DIR}"
```

## 4. 两阶段外部评测

质量环境逐 WAV 产生：

- `muq_mi`：固定 MuQ-Eval A1 的 MI score；
- `audiobox_ce` 和 `audiobox_pq`：固定 Audiobox Aesthetics 的 CE/PQ；

FAD/CLAP 环境逐 WAV+prompt 产生：

- `music_clap`：固定 music-CLAP 的 cosine relevance，数值越高越好。

不要响度归一化或另存音频；evaluator 必须直接读取 generation artifact 中由
hash 锁定的 WAV。FAD 不参加这个 scale selection。

外评必须跨两个已经验收的 Python 3.11 环境完成。中间结果是一个不可变、可复制的
quality artifact；第二阶段重新验证 generation 和 quality 两套 seal，再以
`(sample_id, condition_id)` 严格 join，输出恰好 1200 行统一 score JSONL。少一行、
重复一行、prompt/audio/config hash 不一致或非有限分数都会被拒绝。

### 4.1 eval-quality：MuQ-Eval A1 + Audiobox

只传**本地、非 symlink** 的不可变文件/目录；wrapper 不调用下载 API。若文件仍位于
HF cache 的 symlink snapshot，先用 `cp -aL`（或等价方式）复制为解引用快照：

```bash
export PTC_PROJECT_ROOT=<local>/ICASSP2027
export PTC_QUALITY_PY=<local>/envs/ptc-opd-eval-quality/bin/python
export PTC_MUQ_EVAL_ROOT="${PTC_PROJECT_ROOT}/third_party/MuQ-Eval"
export PTC_MUQ_A1_DIR=/absolute/local/snapshot/MuQ-Eval-A1
export PTC_MUQ_A1_CONFIG="${PTC_MUQ_A1_DIR}/config.yaml"
export PTC_MUQ_BACKBONE=/absolute/local/snapshot/MuQ-large-msd-iter
export PTC_AUDIOBOX_CKPT=/absolute/local/audiobox-aesthetics/checkpoint.pt
export PTC_ENV_ACCEPTANCE_REPORT="${PTC_WORKPACK_ROOT}/docs/environment_acceptance_20260812.md"
export PTC_CFG_QUALITY_DIR="${PTC_CFG_ROOT}/quality"

test "$(git -C "${PTC_MUQ_EVAL_ROOT}" rev-parse HEAD)" = \
  60a88f8ac0909ca1fd1a78af3660f1fc376977a1

CUDA_VISIBLE_DEVICES=0 "${PTC_QUALITY_PY}" \
  "${PTC_WORKPACK_ROOT}/scripts/eval_cfg_quality.py" run \
  --generation-dir "${PTC_CFG_AUDIO_DIR}" \
  --muq-eval-root "${PTC_MUQ_EVAL_ROOT}" \
  --muq-config "${PTC_MUQ_A1_CONFIG}" \
  --muq-state-dict "${PTC_MUQ_A1_DIR}/model_state_dict.pt" \
  --muq-backbone "${PTC_MUQ_BACKBONE}" \
  --audiobox-checkpoint "${PTC_AUDIOBOX_CKPT}" \
  --environment-report "${PTC_ENV_ACCEPTANCE_REPORT}" \
  --output-dir "${PTC_CFG_QUALITY_DIR}" \
  --device cuda:0 --batch-size 8
```

MuQ-Eval Git HEAD 必须为 `60a88...`，tracked files 必须 clean，且不得有非 cache 的
untracked files；实际 tracked-source hash 会
进入 provenance。README 的旧 `AudioProcessor(sample_rate=..., max_duration=...)`
示例不适用于该 commit；wrapper 按源码真实 API 使用
`AudioProcessor(target_sr=24000, clip_samples=240000).process(..., mode="center")`。
`MusicQualityModel` 的浮点参数/buffer 均强制 FP32，不使用 autocast；config、
`model_state_dict.pt`、本地 MuQ backbone snapshot 都实际重哈希；provenance 的
`config_sha256` 锁定把本地 backbone 路径注入后、完全 resolve 的有效模型配置以及
center-crop/FP32 推理设置，两个 released YAML 的逐文件 hash 和集合 hash也同时保留
在 details 中。`source_sha256` 是 MuQ-Eval pinned Git tracked tree 与**实际 import
执行的 `muq` Python 包源码**的复合身份；details 还保存 `muq==0.1.0`、逐 Python
文件 hash，以及实际 import 的 `torch==2.2.2+cu121`、
`torchaudio==2.2.2+cu121`、`numpy==1.26.4`、`soundfile==0.12.1`。任一版本漂移都会
在加载 checkpoint 前 fail closed。

环境验收报告是用户提供文件的 byte-for-byte 副本（SHA-256
`7bd3d913bd210cf1b94165986e8368d8560702871dafac1d1e45beff1fb4d73d`）。它位于
managed `docs/`，因此进入 `WORKPACK_MANIFEST.sha256`；本次 evaluator 还把该文件
重新 hash 到 `runtime_environment`，使每条质量分数经 provenance hash 间接绑定
验收报告。不得以另一个同名内容不同的报告运行。

Audiobox 只允许本地 `checkpoint.pt`，调用
`initialize_predictor(local_ckpt).forward`，消费 CE/PQ；checkpoint 内嵌
`model_cfg`/`target_transform` 形成 config hash，实际安装包 Python 源码形成
source hash。推理前强制 HF/Transformers/Datasets offline，并封禁 Python socket。

成功目录（路径已存在则拒绝覆盖）：

```text
quality/
├── quality_scores.jsonl
├── quality_provenance.json
└── artifact_seal.json
```

复制到第二环境后可先验证：

```bash
"${PTC_QUALITY_PY}" "${PTC_WORKPACK_ROOT}/scripts/eval_cfg_quality.py" verify \
  --generation-dir "${PTC_CFG_AUDIO_DIR}" \
  --quality-dir "${PTC_CFG_QUALITY_DIR}"
```

### 4.2 eval-fad：仅做 music-CLAP，不做 FAD

环境名称虽为 `eval-fad`，本门禁**明确不计算 FAD**。固定显式加载本地
`music_audioset_epoch_15_esc_90.14.pt`，构造
`CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=...)` 后调用
`load_ckpt(local_path)`。每个 batch 分别取得 audio/text embedding，在 FP32 中各自
L2 normalize，再做同一行 prompt/audio 的 dot product。

该 checkpoint 冻结为已用于 small/medium sealed 评分的实际字节身份
`fae3e9c087f2909c28a09dc31c8dfcdacbc42ba44c70e972b58c1bd1caf6dedd`；脚本在构造
模型和反序列化前先重哈希并 fail closed。`laion-clap==1.1.6` 的 `load_ckpt` 未显式
传入 `weights_only`，而 PyTorch 2.6+ 默认改为 `True`，会拒绝 checkpoint 中的
NumPy scalar。因此 wrapper 仅在这一次、且仅在 hash 已通过的 `load_ckpt` 调用期间
为缺省参数注入 `weights_only=False`，并用 `finally` 保证成功或异常后都恢复原始
`torch.load`。该兼容逻辑不允许扩展为进程级 monkey patch。

```bash
export PTC_FAD_PY=<local>/envs/ptc-opd-eval-fad/bin/python
export PTC_MUSIC_CLAP_CKPT=/absolute/local/laion_clap/music_audioset_epoch_15_esc_90.14.pt
export PTC_CFG_EXTERNAL_DIR="${PTC_CFG_ROOT}/external"

CUDA_VISIBLE_DEVICES=0 "${PTC_FAD_PY}" \
  "${PTC_WORKPACK_ROOT}/scripts/eval_cfg_music_clap.py" run \
  --generation-dir "${PTC_CFG_AUDIO_DIR}" \
  --quality-dir "${PTC_CFG_QUALITY_DIR}" \
  --clap-checkpoint "${PTC_MUSIC_CLAP_CKPT}" \
  --output-dir "${PTC_CFG_EXTERNAL_DIR}" \
  --device cuda:0 --batch-size 8

"${PTC_FAD_PY}" "${PTC_WORKPACK_ROOT}/scripts/eval_cfg_music_clap.py" verify \
  --generation-dir "${PTC_CFG_AUDIO_DIR}" \
  --quality-dir "${PTC_CFG_QUALITY_DIR}" \
  --output-dir "${PTC_CFG_EXTERNAL_DIR}"
```

成功目录含 `scores.jsonl`、`evaluator_provenance.json`、`artifact_seal.json`。
其中 `scores.jsonl` 每行 schema：

```json
{
  "schema_version": "ptc-opd-cfg-score-v1",
  "sample_id": "musiccaps:...",
  "prompt_sha256": "64-hex",
  "condition": "cfg",
  "cfg_scale": 3.0,
  "condition_id": "cfg_3.0",
  "audio_sha256": "64-hex",
  "scientific_config_sha256": "generation record中的64-hex",
  "evaluator_provenance_sha256": "下述provenance JSON的文件SHA-256",
  "metrics": {
    "muq_mi": 0.0,
    "audiobox_ce": 0.0,
    "audiobox_pq": 0.0,
    "music_clap": 0.0
  }
}
```

同时写一个 evaluator provenance JSON；`metrics` 的顺序和值必须精确如下，
三个 evaluator 的每个 hash 都是实际 artifact/source/config 的 SHA-256，不是
模型名、Git 40 位 commit 或占位符：

```json
{
  "schema_version": "ptc-opd-cfg-evaluator-provenance-v1",
  "status": "accepted_external_evaluation",
  "metrics": ["muq_mi", "audiobox_ce", "audiobox_pq", "music_clap"],
  "evaluators": {
    "muq_eval": {
      "checkpoint_sha256": "64-hex",
      "source_sha256": "64-hex",
      "config_sha256": "64-hex"
    },
    "audiobox_aesthetics": {
      "checkpoint_sha256": "64-hex",
      "source_sha256": "64-hex",
      "config_sha256": "64-hex"
    },
    "music_clap": {
      "checkpoint_sha256": "64-hex",
      "source_sha256": "64-hex",
      "config_sha256": "64-hex"
    }
  }
}
```

`status=accepted_external_evaluation` 只由第二 wrapper 在 generation、quality、CLAP
三者全部验证成功后写入；不得手工创建。provenance 还绑定 quality seal、generation
seal、实际 evaluator source/config/checkpoint hash，并显式记录
`fad_computed=false`。决策器会重新验证 quality 目录的三个成员和 seal、
external 目录的三个成员和 seal，并且要求两级 provenance 中的 evaluator
identity 完全一致。冻结的 protocol 字段集和五个 offline 环境值也必须精确
相等，不是只检查它们是否存在。

## 5. 决策规则与不可变 artifact

对每个原始 metric，先由 300 个 no-CFG base 样本计算 sample SD；每个 prompt
的 paired delta 除以该 SD。原协议中的定义保持不变：

```text
Aesthetic = 0.5 * Delta-z(Audiobox CE) + 0.5 * Delta-z(Audiobox PQ)
Q_dev     = 0.5 * Delta-z(MuQ MI) + 0.5 * Aesthetic
          = 0.5 * Delta-z(MuQ MI)
          + 0.25 * Delta-z(Audiobox CE)
          + 0.25 * Delta-z(Audiobox PQ)
```

因此 `0.50/0.25/0.25` 是把既有 `0.5 MuQ + 0.5 Aesthetic` 展开，并不是修改
质量门禁。music-CLAP 只作 guardrail，不进入 `Q_dev`。

候选 scale 合格当且仅当：

1. point-estimate `Q_dev > 0`；
2. paired prompt bootstrap 的 `Pr(Q_dev>0) >= 0.90`；
3. music-CLAP paired change `>= -0.10` 个 no-CFG base SD。

从合格候选中选 `Q_dev` 最高者；只有浮点结果**数值精确相等**才选更低 scale，
不使用人为 tolerance。若没有候选合格，输出 stop decision，后续 runner 必须拒绝。

```bash
export PTC_CFG_DECISION_DIR=${PTC_CFG_ROOT}/decision

"${PTC_TRAIN_PY}" "${PTC_WORKPACK_ROOT}/scripts/cfg_scale_gate.py" decide \
  --generation-dir "${PTC_CFG_AUDIO_DIR}" \
  --quality-dir "${PTC_CFG_QUALITY_DIR}" \
  --external-evaluation-dir "${PTC_CFG_EXTERNAL_DIR}" \
  --output-dir "${PTC_CFG_DECISION_DIR}"
```

`decide` 不再提供 `--scores`/`--evaluator-provenance` 的 loose-file 兼容通道。
这是有意的 fail-closed schema 升级：已经用旧命令生成的**真实评测目录**仍然
可用，因为它本来就含 `scores.jsonl`、`evaluator_provenance.json` 和
`artifact_seal.json`；只需改用上面的目录参数。Python 中的 loose score join
函数只保留给 evaluator wrapper 在原子发布前做内部完整性检查，不是科学
决策接口。

成功目录包含：

```text
decision/
├── cfg_scale_decision.json
└── cfg_scale_decision.sha256.json
```

成功 JSON 固定包含 `status="selected"`、数值型 `selected_cfg_scale`、全部输入
hash（包括 external seal、quality seal、generation seal、两级 score/provenance、
protocol 和 offline contract）、每候选统计、scientific config hash 和
`decision_payload_sha256`；sidecar
再锁定最终文件 SHA-256。验证命令：

```bash
"${PTC_TRAIN_PY}" "${PTC_WORKPACK_ROOT}/scripts/cfg_scale_gate.py" \
  verify-decision --decision-dir "${PTC_CFG_DECISION_DIR}"
```

A2 probe 和 Stage-1 training 只能消费这个 decision artifact，并把 decision file
SHA、payload SHA 和 scale 写入各自 immutable config；不得继续接受未绑定 artifact
的手填 `--teacher-cfg-scale`。
