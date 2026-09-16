# Phase A1-R2 运行手册：MusicGen 渐进码本感知边际

**协议：`A1-R2`。状态：实现和离线测试就绪，真实 FMA + MusicGen codec
结果尚未运行。** 在正式产物出现前，不能声称 R2 现象或 P1 已通过。

A1-R1 已因 FMA `107535` 的反相立体声下混归零而失败。R1 必须永久只读并标记：

```text
A1-R1: failed_scientific_gate_exact_antiphase_downmix
```

不得覆盖、重命名或删改旧目录 `artifacts/phase_a1_codec_prior/`。R2 使用新 workpack、
新 manifest 目录和新 artifact 目录。

## 1. 冻结契约

### 1.1 FMA 与选择

- FMA-small archive SHA-1 必须为
  `ade154f733639d52e35e32f5593efe5be76c6d70`。
- 正式运行必须找到恰好 7,994 个原始候选：唯一数值 track ID、可完整解码且
  至少 10 秒。
- 对每个候选先用原域、原 seed 计算确定性 10 秒段，而不是先选 512 首：

```text
selection hash = SHA256("ptc-opd-codec-cal-v1|2701|" + track_id)
segment hash   = SHA256("ptc-opd-codec-segment-v1|2701|" + track_id)
```

- segment hash 前 8 字节按 big-endian uint64 解读，对含末端起点的合法起点数取模。
- 片段 PCM 哈希是 frame-major、channel-interleaved、little-endian float32 C-order。

### 1.2 pre-codec eligibility

每个 10 秒候选片段都必须调用与 estimator 相同、来自固定 vendored AudioCraft
源码树的 `audiocraft.data.audio_utils.convert_audio`，产生精确
float32 `[1,1,320000]` 的 32 kHz mono codec input。不得用 soundfile、
torchaudio 或自行实现的 downmix/resample 近似替代。

```text
pre_downmix_rms = sqrt(mean(original_segment[channel,time]^2))
mono_rms        = sqrt(mean(exact_32k_mono_codec_input^2))
ratio           = mono_rms / max(pre_downmix_rms, 1e-12)

eligible iff:
  all original-segment PCM samples finite
  mono_rms >= 1e-5
  ratio    >= 1e-3
```

L/R Pearson correlation只记录诊断，绝不参与筛选；代码中不得出现 track-ID
黑名单。完成全部 7,994 条 eligibility 后，才按原 selection hash 取最低的
512 条。`107535` 被一般规则排除，下一条 eligible track 自动补位。

### 1.3 codec 与 estimator

- codec 必须满足 `channels=1`、`sample_rate=32000`、`frame_rate=50`、
  `cardinality=2048`、`num_codebooks=4`。
- 输入必须为 float32 `[1,1,320000]`；单次 encode 输出必须为 int64
  `[1,4,500]`，token 严格位于 `[0,2048)`。
- 每首只 encode 一次；`q=0` 是 `zeros_like(input)`，`q>=1` 解码同一 code
  tensor 的 `codes[:,:q,:]`。不递归重编码，不做重建后响度归一化。
- MR-STFT 固定 FFT `{512,1024,2048}`、hop=`FFT/4`、Hann window；主公式不变：

```text
Delta_iq = d_i(q-1) - d_i(q)
Delta_q  = mean_i(Delta_iq)
a_q      = max(Delta_q, 1e-8) / sum_j max(Delta_j, 1e-8)
```

- 任一 reference STFT Frobenius norm `<=1e-7` 立即报错。
  denominator 不再 `clamp_min`；正式结果必须记录
  `denominator_floor_activation=0`。
- bootstrap 固定 10,000 次、PCG64 seed 4701；split-half 继续使用
  `ptc-opd-codec-split-v1|4701`。
- 主估计器始终是 mean。median 和 1% trimmed mean 仅作敏感性分析；N=512 时
  每个 codebook 独立排序，两端各删除 `floor(.01*512)=5` 条，保留 502 条。
- 对 512 条逐一重算 leave-one-out prior 并报告
  `TV(a, a_without_i)`，不得把 LOO prior 替换成主 prior。

### 1.4 schema、路径与源码身份

```text
manifest schema : ptc-opd-fma-calibration-v2
result schema   : ptc-opd-codec-prior-v3
seal schema     : ptc-opd-codec-artifact-seal-v3
identity schema : ptc-opd-codec-prior-artifact-identity-v3

manifests/a1-r2/codec_calibration.train.jsonl
manifests/a1-r2/codec_calibration.train.report.json
artifacts/phase_a1_codec_prior_r2/
```

CLI 会拒绝其他 basename/父目录，防止误覆盖 R1。report 必须审计全部 7,994
候选，并绑定 manifest SHA、builder SHA、AudioCraft base commit 和完整源码树身份。
estimator 会重新验证 report、在 codec encode 前重算每条 eligibility，并将
builder、estimator、`perceptual_prior.py` 三个源码 SHA 同时写入 prior 和 seal。

## 2. 环境变量与依赖检查

变量名只是示例；不要复用系统 `HOME`：

```bash
export PTC_WORKPACK=/path/to/ICASSP2027_PTC_OPD_A1R2_Workpack_20260814
export PTC_TRAIN_PY=<local>/envs/ptc-opd-train-py39-cu121/bin/python
export PTC_FMA_ROOT=/path/to/extracted/fma_small
export PTC_FMA_ARCHIVE=/path/to/fma_small.zip
export PTC_AUDIOCRAFT_ROOT="$PTC_WORKPACK/vendor/audiocraft"
export PTC_CODEC_CKPT=/path/to/local/musicgen-small-compression-checkpoint
export PTC_A1_MANIFEST_DIR="$PTC_WORKPACK/manifests/a1-r2"
export PTC_A1_OUTPUT="$PTC_WORKPACK/artifacts/phase_a1_codec_prior_r2"
```

确认解码依赖和 AudioCraft revision：

```bash
"$PTC_TRAIN_PY" "$PTC_WORKPACK/scripts/build_fma_calibration_manifest.py" --check-deps
git -C "$PTC_AUDIOCRAFT_ROOT" rev-parse HEAD
```

AudioCraft base commit 必须为
`896ec7c47f5e5d1e5aa1e4b260c4405328bf009d`。builder 和 estimator 都会重验
commit 及源码树身份。

## 3. 构建 A1-R2 manifest + 全候选 report

不要预先创建两个输出文件；已有任一文件时 builder 会拒绝覆盖。

```bash
mkdir -p "$PTC_A1_MANIFEST_DIR"

"$PTC_TRAIN_PY" "$PTC_WORKPACK/scripts/build_fma_calibration_manifest.py" \
  --fma-root "$PTC_FMA_ROOT" \
  --archive "$PTC_FMA_ARCHIVE" \
  --audiocraft-root "$PTC_AUDIOCRAFT_ROOT" \
  --output-manifest "$PTC_A1_MANIFEST_DIR/codec_calibration.train.jsonl" \
  --output-report "$PTC_A1_MANIFEST_DIR/codec_calibration.train.report.json"
```

必须确认 report 至少满足：

```text
schema_version                 = ptc-opd-fma-calibration-v2
protocol_label                 = A1-R2
original_candidate_tracks      = 7994
eligibility.audited_tracks     = 7994
selected_tracks                = 512
archive_sha1_verified          = true
publication_eligible           = true
eligibility.track_id_filtering = false
```

report 中应能看到 `107535` 因一般 eligibility 条件失败；但这不是允许写死
该 ID 的要求。

## 4. 固定哈希与离线模型

```bash
export PTC_MANIFEST_SHA256=$(sha256sum "$PTC_A1_MANIFEST_DIR/codec_calibration.train.jsonl" | awk '{print $1}')
export PTC_REPORT_SHA256=$(sha256sum "$PTC_A1_MANIFEST_DIR/codec_calibration.train.report.json" | awk '{print $1}')
export PTC_CODEC_SHA256=$(sha256sum "$PTC_CODEC_CKPT/compression_state_dict.bin" | awk '{print $1}')

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

若 codec checkpoint 本身就是文件，对该文件计算 SHA。只允许 self-contained
AudioCraft export，或精确指向 `facebook/encodec_32khz` 的官方 pretrained
indirection；离线缓存缺失必须失败。

## 5. 单 GPU preflight 与正式运行

```bash
unset MASTER_ADDR MASTER_PORT RANK WORLD_SIZE LOCAL_RANK NODE_RANK GROUP_RANK
unset SLURM_NTASKS SLURM_PROCID SLURM_LOCALID
export CUDA_VISIBLE_DEVICES=0
```

Preflight 不产生科学结果：

```bash
"$PTC_TRAIN_PY" "$PTC_WORKPACK/scripts/estimate_codec_prior.py" \
  --manifest "$PTC_A1_MANIFEST_DIR/codec_calibration.train.jsonl" \
  --manifest-sha256 "$PTC_MANIFEST_SHA256" \
  --manifest-report "$PTC_A1_MANIFEST_DIR/codec_calibration.train.report.json" \
  --manifest-report-sha256 "$PTC_REPORT_SHA256" \
  --source-root "$PTC_FMA_ROOT" \
  --audiocraft-root "$PTC_AUDIOCRAFT_ROOT" \
  --codec-checkpoint "$PTC_CODEC_CKPT" \
  --codec-checkpoint-sha256 "$PTC_CODEC_SHA256" \
  --device cuda:0 \
  --check-only
```

只有 stdout 为 `preflight_passed_not_scientific_result` 才继续。正式输出目录必须
完全不存在：

```bash
test ! -e "$PTC_A1_OUTPUT"

"$PTC_TRAIN_PY" "$PTC_WORKPACK/scripts/estimate_codec_prior.py" \
  --manifest "$PTC_A1_MANIFEST_DIR/codec_calibration.train.jsonl" \
  --manifest-sha256 "$PTC_MANIFEST_SHA256" \
  --manifest-report "$PTC_A1_MANIFEST_DIR/codec_calibration.train.report.json" \
  --manifest-report-sha256 "$PTC_REPORT_SHA256" \
  --source-root "$PTC_FMA_ROOT" \
  --audiocraft-root "$PTC_AUDIOCRAFT_ROOT" \
  --codec-checkpoint "$PTC_CODEC_CKPT" \
  --codec-checkpoint-sha256 "$PTC_CODEC_SHA256" \
  --output-dir "$PTC_A1_OUTPUT" \
  --device cuda:0
```

成功时以一次目录级 rename 发布以下闭集：

| 文件 | 内容 |
|---|---|
| `codec_prior.per_clip.jsonl.gz` | 距离、边际、3 个 reference norm、telescoping residual、逐条 LOO influence 与输入身份 |
| `codec_prior.summary.csv` | mean、median、1% trimmed mean、prior、CI 与 split-half |
| `codec_prior.json` | 主 prior、bootstrap、LOO、敏感性、P1 gate、运行与源码身份 |
| `ARTIFACT_SEAL.json` | v3 seal、三个 payload hash/size、manifest/report/codec/AudioCraft/实现身份 |

任何异常都清理 staging，最终目录保持不存在。重跑必须整体归档旧 R2 目录并使用
一个全新的同 basename 目标，不能删单个成员再拼接。

## 6. R2 验收与停止线

工程门槛全部满足才可解释科学结果：

- 512 个不同 track，且 report 对全部 7,994 候选完成 eligibility；
- 三个 reference norm 对每条均严格 `>1e-7`；
- distance/marginal 全有限，`denominator_floor_activation=0`；
- 每条满足 `sum_q Delta_iq ~= d_i(0)-d_i(4)`；
- manifest/report/codec/源码身份与完整 v3 seal 全部通过。

预注册 P1 科学门槛：

```text
bootstrap 95% lower bound of TV(a, uniform) > 0.05
split-half Kendall tau-b                     >= 2/3
max_i TV(a, a_without_i)                     <= 0.05
Kendall(median ranking, primary ranking)     >= 2/3
Kendall(1%-trimmed ranking, primary ranking) >= 2/3
```

代码中的阈值是精确 `2/3`，不是四舍五入的 `0.67`。任一 engineering gate
失败即该运行无效；任一 scientific/sensitivity gate 失败即
`fail_stop_review`。在 A1-R2 通过且 node-3 训练门禁通过前，不启动 A2/A3、
500-step pilot、学习率 sweep 或正式 Stage-1。

## 7. 定向测试

```bash
cd "$PTC_WORKPACK"
PYTHONPATH=src "$PTC_TRAIN_PY" -m unittest discover \
  -s "${PTC_WORKPACK}/tests" -p test_perceptual_prior.py -v
PYTHONPATH=src "$PTC_TRAIN_PY" -m unittest discover \
  -s "${PTC_WORKPACK}/tests" -p test_codec_phenomenon_scripts.py -v
```

测试覆盖反相/正常输入分支、确定性 selection/offset/hash、阈值与 manifest
校验、无 denominator clamp 的静音 fail-closed、512 条 trimmed/LOO、源码与
report hash、协议 tamper、原子发布和拒绝覆盖。它们不替代真实 512-clip GPU
运行。
