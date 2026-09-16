# Stage-1 MERT diversity 与 FAD pipeline-check

本子系统消费已经封存的 Stage-1 generation artifact
（`ptc-opd-stage1-generation-v1`，128 prompts × seeds `{31001,31002}`），输出：

1. 每个 prompt 的 MERT 两-seed cosine distance；
2. 该 condition 的 128-prompt mean cosine distance；
3. `clap-laion-music` 与 `MERT-v1-95M layer12` 两个有限 FAD 数值。

MERT diversity 是预声明的小模型 pilot point-estimate gate。FAD 只验证完整
pipeline 能得到有限数值，禁止用于模型、方法、checkpoint、LR 选择，也禁止把本轮
有限性检查写成论文效果结论。

## 1. 冻结契约

- FADtk：official tag `1.1.0`，commit
  `6f815282007288a7165d037a95dacd6d783b6f32`；10 个 Python 文件的聚合 source
  SHA-256 必须为
  `7cffe82dc3e13d1508cdad06bf845aec7fce269e871ed64de5c6210e3d142fe9`。
- MERT：`m-a-p/MERT-v1-95M`，revision
  `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`，hidden layer `12`。正式
  consumer 逐字节重验以下且仅以下 5 个科学文件：

| relative file | bytes | SHA-256 |
|---|---:|---|
| `config.json` | 1817 | `ea2627c4c7825cd66f3c944b6b966331604c35928174e0100cd4a82829424e32` |
| `configuration_MERT.py` | 5340 | `ae0ec2bab8f59c724ba9878a7c20b67210189536ea62d34a56775968e9decb03` |
| `modeling_MERT.py` | 18033 | `6c3ee73cef6f0c30ef494f88d96f891fa6925ffe663fa391b512f4b57abecc6c` |
| `preprocessor_config.json` | 211 | `cc5a5e4a5d3b1a758a5ed984b2eaa15bb0522d811d44a9eed82bfca4baa0dc8f` |
| `pytorch_model.bin` | 377552987 | `a2b8b747f72c06e0595aeae41ae5473f4364938c6b39b2c58be38c48e6bd3fcd` |

  `pytorch_model.bin` 必须是唯一可加载权重。任何位置出现
  `model.safetensors`、`*.index.json`、`pytorch_model-*.bin`、
  `model-*.safetensors`、`tf_model.h5` 或 `flax_model.msgpack` 都立即失败。
- prompt diversity：每个 WAV 的 layer-12 frame embeddings 做 arithmetic mean，
  再 L2 normalization；同一 prompt 的 seed `31001` 与 `31002` 做
  `1 - cosine_similarity`。
- music-CLAP：`clap-laion-music`，checkpoint
  `music_audioset_epoch_15_esc_90.14.pt`，接受的 SHA-256 为
  `fae3e9c087f2909c28a09dc31c8dfcdacbc42ba44c70e972b58c1bd1caf6dedd`。
- FAD reference：A1-R2 封存的 512 个 eligible 10-second segments 中，按
  `sha256("ptc-opd-stage1-fad-reference-v1|2701|<fma_track_id>")` 升序取前
  256 个；hash 相同时按整数 track ID 排序。

旧文档 checkpoint hash `6004...` 不属于本环境已验收的 republished checkpoint，
不能混用。模型 pin artifact 科学上绑定上述 MERT 5 文件、CLAP checkpoint 与 FADtk
1.1.0 Python source。MERT 完整 snapshot tree 只作为 operational observation 记录；
`README`/model card 等非科学文件在不同 cache materialization 之间可以不同。

## 2. 不可变输入与 staging 规则

正式 runner 不直接把 generation/reference WAV 交给 FADtk。它先逐文件校验
SHA-256，再复制到一个独立临时目录：

```text
<output-parent>/.<output-name>.compute.*/
  generated_audio/
  reference_audio/
  resources/music_audioset_epoch_15_esc_90.14.pt
  hf_home/
  torch_home/
  tmp/
```

FADtk 1.1.0 会在输入旁创建 `convert/`、`embeddings/` 和 `stats/`。因此只有上述
临时副本能产生缓存。runner 在计算前后重新计算 generation/reference tree hash、
MERT 完整 snapshot tree hash 和原始 CLAP checkpoint hash，并在真实 MERT load 前及
计算后再次重验 5 个科学文件。一次运行内部任何变化都会失败且不发布 artifact；这不
把 README 等非科学字节误当作跨机器科学 pin。临时计算目录不进入最终 seal，成功或
失败均清理。

运行时强制：

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
TOKENIZERS_PARALLELISM=false
```

同时 monkey-patch Python socket connect API 为 fail-closed。MERT 必须从显式本地
snapshot 加载；CLAP 必须从 staging checkpoint copy 加载。

## 3. 第一次运行：封存 evaluator resources

本节在 `STAGE1_AUTONOMY_RUNBOOK.md` §0 与 §4 的正式 shell 中执行；路径如下，不再另找
“最新” snapshot：

```bash
export MERT_SNAPSHOT="${HF_HUB_CACHE}/models--m-a-p--MERT-v1-95M/snapshots/12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"
export CLAP_CKPT="<local>/models/ICASSP2027/laion_clap/music_audioset_epoch_15_esc_90.14.pt"
export MODEL_PINS="${PTC_WORKPACK}/artifacts/stage1/evaluator_model_pins"
```

在已验收的 `ptc-opd-eval-fad` 环境运行：

```bash
"${PTC_FAD_PY}" "${PTC_WORKPACK}/scripts/build_stage1_eval_model_pins.py" build \
  --mert-snapshot "$MERT_SNAPSHOT" \
  --clap-checkpoint "$CLAP_CKPT" \
  --output-dir "$MODEL_PINS"

"${PTC_FAD_PY}" "${PTC_WORKPACK}/scripts/build_stage1_eval_model_pins.py" verify \
  --mert-snapshot "$MERT_SNAPSHOT" \
  --clap-checkpoint "$CLAP_CKPT" \
  --artifact-dir "$MODEL_PINS"
```

不得用 revision 不同但文件名相同的目录，也不得通过参数跳过任一 hash。`build`、
`verify` 与每次 live evaluation 都会独立重验同一 5-file/alternative-weight contract。

## 4. 一次性构建 FAD reference

以下输入必须是正式 A1-R2 的两个文件，且目录 basename 为 `a1-r2`：

```bash
export A1_MANIFEST="${PTC_RETAINED}/manifests/a1-r2/codec_calibration.train.jsonl"
export A1_REPORT="${PTC_RETAINED}/manifests/a1-r2/codec_calibration.train.report.json"
export FMA_ROOT="<local>/data/ICASSP2027/fma_small/extracted/fma_small"
export FAD_REFERENCE="${PTC_WORKPACK}/artifacts/stage1/fad_reference"

export A1_MANIFEST_SHA256=$(sha256sum "$A1_MANIFEST" | awk '{print $1}')
export A1_REPORT_SHA256=$(sha256sum "$A1_REPORT" | awk '{print $1}')

"${PTC_FAD_PY}" "${PTC_WORKPACK}/scripts/build_stage1_fad_reference.py" build \
  --a1-manifest "$A1_MANIFEST" \
  --a1-manifest-sha256 "$A1_MANIFEST_SHA256" \
  --a1-report "$A1_REPORT" \
  --a1-report-sha256 "$A1_REPORT_SHA256" \
  --fma-root "$FMA_ROOT" \
  --output-dir "$FAD_REFERENCE"

"${PTC_FAD_PY}" "${PTC_WORKPACK}/scripts/build_stage1_fad_reference.py" verify \
  --a1-manifest "$A1_MANIFEST" \
  --a1-report "$A1_REPORT" \
  --artifact-dir "$FAD_REFERENCE"
```

Builder 会调用 workpack 内 A1-R2 manifest/report 全量 verifier，并从每条原始 FMA
文件重解码冻结的 start frame 和 10 秒 PCM。输出 WAV 是 source-rate/source-channel
float32 wrapper，不做 loudness normalization；两个 FAD backend 再从这些原始片段
各自独立 mono/resample。

## 5. 运行一个 condition

每个 generation artifact 单独运行一次：

```bash
"${PTC_FAD_PY}" "${PTC_WORKPACK}/scripts/eval_stage1_diversity_fad.py" \
  --generation-dir "${PTC_SMALL_GENERATIONS[5]}" \
  --eval-manifest-dir "${PTC_PILOT_MANIFEST}" \
  --reference-dir "$FAD_REFERENCE" \
  --a1-manifest "$A1_MANIFEST" \
  --a1-report "$A1_REPORT" \
  --model-pins-dir "$MODEL_PINS" \
  --mert-snapshot "$MERT_SNAPSHOT" \
  --clap-checkpoint "$CLAP_CKPT" \
  --output-dir "${PTC_SMALL_DIVERSITIES[5]}" \
  --device cuda:0
```

建议单 condition/单 GPU 串行运行；不要让多个进程共享同一个 output 或 staging
目录。正式命令不支持 resume 或覆盖已有 artifact。

## 6. CPU 独立复核

Verifier 不 import torch、transformers、laion-clap 或 fadtk：

```bash
"${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/verify_stage1_diversity_fad.py" \
  --artifact-dir "${PTC_SMALL_DIVERSITIES[5]}" \
  --generation-dir "${PTC_SMALL_GENERATIONS[5]}" \
  --eval-manifest-dir "${PTC_PILOT_MANIFEST}" \
  --reference-dir "$FAD_REFERENCE" \
  --a1-manifest "$A1_MANIFEST" \
  --a1-report "$A1_REPORT" \
  --model-pins-dir "$MODEL_PINS"
```

成功输出的 consumer 字段：

```json
{
  "status": "verified",
  "condition_id": "...",
  "mert_diversity_mean_cosine_distance": 0.0,
  "fad_pipeline_check_passed": true,
  "fad_scores": {
    "MERT-v1-95M-layer12": 0.0,
    "clap-laion-music": 0.0
  },
  "artifact_seal_sha256": "..."
}
```

`fad_scores` 数值只归档，controller 只能消费 `fad_pipeline_check_passed`；任何代码
把 FAD score 放入排序、retention、checkpoint selection 或论文统计都违反本契约。

## 7. Artifact schemas

- model pins seal：`ptc-opd-stage1-diversity-fad-model-pins-seal-v2`
- reference seal：`ptc-opd-stage1-fad-reference-seal-v1`
- evaluation seal：`ptc-opd-stage1-diversity-fad-seal-v1`
- evaluation status：`complete_diversity_fad_pipeline_check`
- summary：`evaluation_summary.json`
- 128 prompt rows：`mert_diversity.jsonl`
- provenance：`evaluator_provenance.json`

Python consumer：

```python
from ptc_opd.stage1_diversity_fad import verify_diversity_fad_artifact

verified = verify_diversity_fad_artifact(
    artifact_dir,
    generation_dir=generation_dir,
    eval_manifest_dir=eval_manifest_dir,
    reference_dir=reference_dir,
    a1_manifest=a1_manifest,
    a1_report=a1_report,
    model_pins_dir=model_pins_dir,
)
diversity = verified["mert_diversity_mean_cosine_distance"]
fad_ok = verified["fad_pipeline_check_passed"]
```
