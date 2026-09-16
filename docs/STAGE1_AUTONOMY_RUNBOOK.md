# Stage-1 自主执行总 runbook

版本：`ptc-opd-stage1-autonomy-runbook-v1`，2026-08-19。本文是远端下属的唯一逐步
执行指南；科学与修复权限以 `../AUTONOMY_POLICY.md` 为准。

## 0. 固定路径与公共 shell

node-0 先执行并保存为自己的正式 shell 环境；node-1～3 使用相同路径：

```bash
set -euo pipefail
umask 077

export PTC_WORKPACK=<local>/ICASSP2027_PTC_OPD_Stage1Autonomy_20260819
export PTC_RETAINED=<local>/ICASSP2027_PTC_OPD_A1R2_Workpack_20260814
export PTC_PROJECT_ROOT=<local>/ICASSP2027
export PTC_T5_OUTER=<local>/ptc_local/deliverables/t5closure/20260819T064624Z
export PTC_TRAIN_ENV=<local>/envs/ptc-opd-train-py39-cu121
export PTC_QUALITY_ENV=<local>/envs/ptc-opd-eval-quality
export PTC_FAD_ENV=<local>/envs/ptc-opd-eval-fad
export PTC_TRAIN_PY="${PTC_TRAIN_ENV}/bin/python"
export PTC_TORCHRUN="${PTC_TRAIN_ENV}/bin/torchrun"
export PTC_QUALITY_PY="${PTC_QUALITY_ENV}/bin/python"
export PTC_FAD_PY="${PTC_FAD_ENV}/bin/python"
export HF_HOME="${PTC_PROJECT_ROOT}/storage/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${PTC_PROJECT_ROOT}/storage/cache/torch"

export PTC_TRAIN_MANIFEST="${PTC_RETAINED}/manifests/musiccaps-v1/train.full.jsonl"
export PTC_DEV_MANIFEST="${PTC_RETAINED}/manifests/musiccaps-v1/dev.full.jsonl"
export PTC_PROBE_MANIFEST="${PTC_RETAINED}/manifests/musiccaps-v1/phenomenon_probe.dev.jsonl"
export PTC_A1_MANIFEST="${PTC_RETAINED}/manifests/a1-r2/codec_calibration.train.jsonl"
export PTC_A1_REPORT="${PTC_RETAINED}/manifests/a1-r2/codec_calibration.train.report.json"
export PTC_A1_PRIOR="${PTC_RETAINED}/artifacts/phase_a1_codec_prior_r2"
export PTC_A2_PROBE="${PTC_RETAINED}/artifacts/phase_a2_a3/probe"
export PTC_A3_SUMMARY="${PTC_RETAINED}/artifacts/phase_a2_a3/summary"
export PTC_NODE3="${PTC_RETAINED}/console_logs/node3_gate_20260818T080050Z"
export PTC_AUDIOCRAFT="${PTC_RETAINED}/vendor/audiocraft"
export PTC_CFG_SMALL="${PTC_RETAINED}/artifacts/cfg_scale/small/decision"

mapfile -t PTC_SMALL_CANDIDATES < <(find "${PTC_RETAINED}/checkpoints" \
  -mindepth 1 -maxdepth 1 -type d -name 'musicgen-small-*' -print | sort)
test "${#PTC_SMALL_CANDIDATES[@]}" -eq 1
export PTC_MUSICGEN_SMALL="${PTC_SMALL_CANDIDATES[0]}"

for required in \
  "${PTC_WORKPACK}" "${PTC_RETAINED}" "${PTC_T5_OUTER}" \
  "${PTC_TRAIN_PY}" "${PTC_TORCHRUN}" "${PTC_QUALITY_PY}" "${PTC_FAD_PY}" \
  "${PTC_TRAIN_MANIFEST}" "${PTC_DEV_MANIFEST}" "${PTC_PROBE_MANIFEST}" \
  "${PTC_A1_MANIFEST}" "${PTC_A1_REPORT}" "${PTC_A1_PRIOR}" \
  "${PTC_A2_PROBE}" "${PTC_A3_SUMMARY}" "${PTC_NODE3}" \
  "${PTC_AUDIOCRAFT}" "${PTC_CFG_SMALL}" "${PTC_MUSICGEN_SMALL}"; do
  test -e "${required}"
done

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
```

若 checkpoint 候选不是恰好一个，不要手选“最新目录”；先按冻结 small CFG decision 的
checkpoint/tree identity 找到唯一匹配项。

每次启动单机八卡前执行：

```bash
unset MASTER_ADDR MASTER_PORT RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE
unset GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
unset SLURM_JOB_ID SLURM_PROCID SLURM_LOCALID SLURM_NTASKS
unset PMI_RANK PMI_SIZE PMI_FD PMIX_RANK PMIX_NAMESPACE
unset OMPI_COMM_WORLD_RANK OMPI_COMM_WORLD_SIZE
unset OMPI_COMM_WORLD_LOCAL_RANK OMPI_COMM_WORLD_LOCAL_SIZE
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_ASYNC_ERROR_HANDLING=1
```

长任务在每台机器各开一个命名 tmux session；先进入 session、粘贴本节公共 shell，再
执行该机器唯一的训练命令。外层 stdout/stderr 用 `tee` 写到
`${PTC_WORKPACK}/console_logs/stage1/${PTC_STAGE1_UTC}/`，不要写进 closed artifact 目录。tmux 只是
进程托管，不改变“一台机器同一时刻最多一个八卡训练 job”。

## 1. 源码 seal、controller 与上游 readiness

```bash
"${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/seal_workpack.py" verify \
  --root "${PTC_WORKPACK}"
"${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/check_workpack.py"
"${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/run_stage1_controller.py" preflight
"${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/run_stage1_controller.py" plan

source "${PTC_WORKPACK}/scripts/stage1_controller_shell.sh"
stage1_controller_init

export PTC_T5_EVIDENCE_JSON="${PTC_CONTROLLER_MAP_ROOT}/t5_closure.attempt-1.evidence.json"
export PTC_T5_DEPS_JSON="${PTC_CONTROLLER_MAP_ROOT}/t5_closure.attempt-1.dependencies.json"
stage1_write_path_map "${PTC_T5_EVIDENCE_JSON}" closure_dir "${PTC_T5_OUTER}"
stage1_write_path_map "${PTC_T5_DEPS_JSON}"
stage1_controller_commit t5_closure 1 initial \
  "${PTC_T5_EVIDENCE_JSON}" "${PTC_T5_DEPS_JSON}"

export PTC_STAGE1_UTC=$(date -u +%Y%m%dT%H%M%SZ)
export PTC_UPSTREAM="${PTC_WORKPACK}/artifacts/stage1/upstream_readiness/${PTC_STAGE1_UTC}"
mkdir -p "$(dirname "${PTC_UPSTREAM}")"
test ! -e "${PTC_UPSTREAM}"

"${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/verify_stage1_upstreams.py" \
  --workpack-root "${PTC_WORKPACK}" \
  --a1-dir "${PTC_A1_PRIOR}" \
  --small-cfg-decision-dir "${PTC_CFG_SMALL}" \
  --a2-probe-dir "${PTC_A2_PROBE}" \
  --a3-summary-dir "${PTC_A3_SUMMARY}" \
  --node3-evidence-dir "${PTC_NODE3}" \
  --t5-closure-dir "${PTC_T5_OUTER}" \
  --output-dir "${PTC_UPSTREAM}"

export PTC_RETAINED_EVIDENCE_JSON="${PTC_CONTROLLER_MAP_ROOT}/retained_seals_audit.attempt-1.evidence.json"
export PTC_RETAINED_DEPS_JSON="${PTC_CONTROLLER_MAP_ROOT}/retained_seals_audit.attempt-1.dependencies.json"
stage1_write_path_map "${PTC_RETAINED_EVIDENCE_JSON}" \
  upstream_dir "${PTC_UPSTREAM}" \
  a1_dir "${PTC_A1_PRIOR}" small_cfg_dir "${PTC_CFG_SMALL}" \
  a2_dir "${PTC_A2_PROBE}" a3_dir "${PTC_A3_SUMMARY}" \
  node3_dir "${PTC_NODE3}" t5_closure_dir "${PTC_T5_OUTER}" \
  t5_archive "${PTC_T5_OUTER}.tar.gz" \
  train_manifest "${PTC_TRAIN_MANIFEST}" dev_manifest "${PTC_DEV_MANIFEST}" \
  probe_manifest "${PTC_PROBE_MANIFEST}" a1_manifest "${PTC_A1_MANIFEST}" \
  a1_report "${PTC_A1_REPORT}" audiocraft_dir "${PTC_AUDIOCRAFT}" \
  musicgen_small_dir "${PTC_MUSICGEN_SMALL}"
stage1_write_path_map "${PTC_RETAINED_DEPS_JSON}" \
  t5_closure "$(stage1_dependency_receipt t5_closure)"
stage1_controller_commit retained_seals_audit 1 initial \
  "${PTC_RETAINED_EVIDENCE_JSON}" "${PTC_RETAINED_DEPS_JSON}"
```

controller 是严格的 **node-0 单写者**。四机 producer、GPU verifier 和离线评估可以并行，
但所有 `stage1_write_path_map`、`stage1_controller_commit` 与 ledger/receipt CLI 只能由
node-0 串行执行；其他节点绝不调用 commit。并行完成的 `pilot_eval_manifest` 与
`b1_prestability` 也固定先登记 pilot、再登记 B1，避免两个 receipt 竞争同一 ledger
revision。

它必须生成 `ready_for_b1_prestability`，并证明所有上游 before/after tree hash 相同。
这里的 T5 参数是 attempt 外层，因为 readiness 同时核对 `DELIVERY_STATUS.json` 和相邻
tar.gz；B1 wrapper 的 T5 参数则是 `${PTC_T5_OUTER}/output_dir`。

## 2. 并行建立 pilot manifest 与执行 B1 pre-stability

pilot manifest 是纯 CPU，可在 node-1 与 node-0 的 B1 同时进行。node-1 **只执行**
下面的 producer block；完成 `verify` 后停下，不写 path map、不调用 controller：

```bash
export PTC_PILOT_MANIFEST="${PTC_WORKPACK}/artifacts/stage1/pilot_eval_manifest"
test ! -e "${PTC_PILOT_MANIFEST}"
"${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/build_pilot_eval_manifest.py" build \
  --source-dev-manifest "${PTC_DEV_MANIFEST}" \
  --output-dir "${PTC_PILOT_MANIFEST}"
"${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/build_pilot_eval_manifest.py" verify \
  --source-dev-manifest "${PTC_DEV_MANIFEST}" \
  --artifact-dir "${PTC_PILOT_MANIFEST}"
```

共享目录出现后，下面的登记 block **只由 node-0** 执行。node-0 先独立重验，再按固定
顺序登记 pilot receipt；它可在 B1 GPU producer 仍运行时完成，但 B1 receipt 必须随后由
同一个 node-0 单写者串行登记：

```bash
export PTC_PILOT_MANIFEST="${PTC_WORKPACK}/artifacts/stage1/pilot_eval_manifest"
test -d "${PTC_PILOT_MANIFEST}"
"${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/build_pilot_eval_manifest.py" verify \
  --source-dev-manifest "${PTC_DEV_MANIFEST}" \
  --artifact-dir "${PTC_PILOT_MANIFEST}"

export PTC_PILOT_EVIDENCE_JSON="${PTC_CONTROLLER_MAP_ROOT}/pilot_eval_manifest.attempt-1.evidence.json"
export PTC_PILOT_DEPS_JSON="${PTC_CONTROLLER_MAP_ROOT}/pilot_eval_manifest.attempt-1.dependencies.json"
stage1_write_path_map "${PTC_PILOT_EVIDENCE_JSON}" \
  artifact_dir "${PTC_PILOT_MANIFEST}" source_dev_manifest "${PTC_DEV_MANIFEST}"
stage1_write_path_map "${PTC_PILOT_DEPS_JSON}" \
  t5_closure "$(stage1_dependency_receipt t5_closure)" \
  retained_seals_audit "$(stage1_dependency_receipt retained_seals_audit)"
stage1_controller_commit pilot_eval_manifest 1 initial \
  "${PTC_PILOT_EVIDENCE_JSON}" "${PTC_PILOT_DEPS_JSON}"
```

node-0 的 B1：

```bash
export PTC_PYTHON="${PTC_TRAIN_PY}"
export B1_CHECKPOINT="${PTC_MUSICGEN_SMALL}"
export AUDIOCRAFT_ROOT="${PTC_AUDIOCRAFT}"
export PROBE_MANIFEST="${PTC_PROBE_MANIFEST}"
export DEV_MANIFEST="${PTC_DEV_MANIFEST}"
export CFG_DECISION_DIR="${PTC_CFG_SMALL}"
export CODEBOOK_PRIOR_DIR="${PTC_A1_PRIOR}"
export A2_PROBE_DIR="${PTC_A2_PROBE}"
export NODE3_EVIDENCE_DIR="${PTC_NODE3}"
export T5_CLOSURE_ARTIFACT_DIR="${PTC_T5_OUTER}/output_dir"
export B1_OUTPUT_ROOT="${PTC_WORKPACK}/artifacts/stage1/b1_prestability"
mkdir -p "${B1_OUTPUT_ROOT}"

export PTC_B1_UTC=$(date -u +%Y%m%dT%H%M%SZ)
export PTC_B1_SESSION="ptc_b1_${PTC_B1_UTC}"
export PTC_B1_START_MARKER="${B1_OUTPUT_ROOT}/.${PTC_B1_SESSION}.start"
test ! -e "${PTC_B1_START_MARKER}"
: > "${PTC_B1_START_MARKER}"
if tmux has-session -t "${PTC_B1_SESSION}" 2>/dev/null; then
  echo "refusing to reuse tmux session ${PTC_B1_SESSION}" >&2
  false
fi
for variable_name in \
  PTC_PYTHON B1_CHECKPOINT AUDIOCRAFT_ROOT PROBE_MANIFEST DEV_MANIFEST \
  CFG_DECISION_DIR CODEBOOK_PRIOR_DIR A2_PROBE_DIR NODE3_EVIDENCE_DIR \
  T5_CLOSURE_ARTIFACT_DIR B1_OUTPUT_ROOT; do
  tmux set-environment -g "${variable_name}" "${!variable_name}"
done
tmux new-session -d -s "${PTC_B1_SESSION}" \
  "exec bash '${PTC_WORKPACK}/scripts/run_remote_b1_prestability.sh'"
tmux attach -t "${PTC_B1_SESSION}"
```

wrapper 完成后按 start marker 自动发现唯一的新 `b1_prestability` 目录。独立复核必须显示
`prestability_gate_passed=true` 且 `full_b1_passed=false`：

```bash
mapfile -t PTC_B1_FINAL_CANDIDATES < <(find "${B1_OUTPUT_ROOT}" \
  -mindepth 2 -maxdepth 2 -type d -name b1_prestability \
  -newer "${PTC_B1_START_MARKER}" \
  -print | sort)
test "${#PTC_B1_FINAL_CANDIDATES[@]}" -eq 1
export PTC_B1_FINAL="${PTC_B1_FINAL_CANDIDATES[0]}"
"${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/verify_b1_prestability.py" \
  final "${PTC_B1_FINAL}"

export PTC_B1_EVIDENCE_JSON="${PTC_CONTROLLER_MAP_ROOT}/b1_prestability.attempt-1.evidence.json"
export PTC_B1_DEPS_JSON="${PTC_CONTROLLER_MAP_ROOT}/b1_prestability.attempt-1.dependencies.json"
stage1_write_path_map "${PTC_B1_EVIDENCE_JSON}" \
  artifact_dir "${PTC_B1_FINAL}" probe_manifest "${PTC_PROBE_MANIFEST}" \
  dev_manifest "${PTC_DEV_MANIFEST}" small_cfg_dir "${PTC_CFG_SMALL}" \
  a1_dir "${PTC_A1_PRIOR}" a2_dir "${PTC_A2_PROBE}" \
  node3_dir "${PTC_NODE3}" t5_closure_dir "${PTC_T5_OUTER}" \
  audiocraft_dir "${PTC_AUDIOCRAFT}" musicgen_small_dir "${PTC_MUSICGEN_SMALL}"
stage1_write_path_map "${PTC_B1_DEPS_JSON}" \
  t5_closure "$(stage1_dependency_receipt t5_closure)" \
  retained_seals_audit "$(stage1_dependency_receipt retained_seals_audit)"
stage1_controller_commit b1_prestability 1 initial \
  "${PTC_B1_EVIDENCE_JSON}" "${PTC_B1_DEPS_JSON}"
```

任何 KL、gradient、pattern round-trip 或八卡 reference 断言失败均为红灯。

## 3. 四机 paired performance assay

B1 通过后，四机分别运行 A/B/C/D 平衡顺序。不得据结果修改 production
`find_unused_parameters=True` policy。完整命令只执行
`PERF_BENCHMARK_RUNBOOK.md` 第 5～8 节；输入 manifest、AudioCraft、CFG 使用本文件的
`PTC_RETAINED` 路径，输出使用：

```bash
export PTC_PERF_UTC=$(date -u +%Y%m%dT%H%M%SZ)  # 仅 node-0 执行一次
export PTC_PERF_ROOT="${PTC_WORKPACK}/artifacts/stage1/performance/${PTC_PERF_UTC}"
```

把 `PTC_PERF_UTC` 的同一字面值复制到 node-1～3；其他节点不得再次执行 `date`。
共享 parent 可以已存在，但每台机器对应的 `node-0`～`node-3` leaf 必须不存在。

只有四个 node verifier 和 aggregate verifier 全通过才继续。测量 CV 超阈值时按同一
四臂计划新 UTC 完整延长，禁止删除慢 block。

aggregate 完成后 node-0 立即登记，不得先启动 LR：

```bash
export PTC_PERF_EVIDENCE_JSON="${PTC_CONTROLLER_MAP_ROOT}/performance_benchmark.attempt-1.evidence.json"
export PTC_PERF_DEPS_JSON="${PTC_CONTROLLER_MAP_ROOT}/performance_benchmark.attempt-1.dependencies.json"
stage1_write_path_map "${PTC_PERF_EVIDENCE_JSON}" \
  aggregate_dir "${PTC_PERF_ROOT}/aggregate" \
  node0_dir "${PTC_PERF_ROOT}/node-0" node1_dir "${PTC_PERF_ROOT}/node-1" \
  node2_dir "${PTC_PERF_ROOT}/node-2" node3_dir "${PTC_PERF_ROOT}/node-3" \
  train_manifest "${PTC_TRAIN_MANIFEST}" small_cfg_dir "${PTC_CFG_SMALL}" \
  audiocraft_dir "${PTC_AUDIOCRAFT}" \
  musicgen_small_dir "${PTC_MUSICGEN_SMALL}"
stage1_write_path_map "${PTC_PERF_DEPS_JSON}" \
  b1_prestability "$(stage1_dependency_receipt b1_prestability)"
stage1_controller_commit performance_benchmark 1 initial \
  "${PTC_PERF_EVIDENCE_JSON}" "${PTC_PERF_DEPS_JSON}"
```

这四个科学输入 evidence 不是说明性路径：controller 将它们逐字节绑定到
`retained_seals_audit` receipt，并让四个 node verifier 逐一复核 16 个 arm 的
`benchmark_manifest.json` 与 `formal_run/run_manifest.json`。即使替换品本身是另一个
合法的 MusicGen-small、CFG decision、AudioCraft tree 或 train manifest，只要不是
retained receipt 授权的同一路径和 byte identity，performance receipt 就失败；16 个
formal run 还必须给出同一 training-lineage anchor，并与 `b1_prestability` receipt
逐字段相等。

## 4. 一次性准备 evaluator 资源

下列定位器只接受已验收目录中的**唯一 hash 匹配项**，不下载、不访问 mutable
`main`，也不接受“最新目录”。MuQ-Eval-A1、MuQ backbone、Audiobox 必须已按上一阶段
规则解引用到 `${PTC_RETAINED}/checkpoints/eval/`；MERT 使用冻结 revision 的 HF
snapshot，并逐字节冻结 5 个真正影响 load 的文件；music-CLAP 与 FMA 使用已经验收的
固定路径。MERT 完整 cache tree 仅记录单次运行 before/after，不作为跨机器科学 pin。

```bash
export PTC_MUQ_EVAL_ROOT="${PTC_PROJECT_ROOT}/third_party/MuQ-Eval"
export PTC_EVAL_ASSET_ROOT="${PTC_RETAINED}/checkpoints/eval"

resolve_unique_file_sha256 () {
  local search_root=$1
  local basename=$2
  local expected_sha256=$3
  local candidates=()
  mapfile -t candidates < <(
    find "${search_root}" -type f -name "${basename}" -print0 |
      while IFS= read -r -d '' candidate; do
        if [[ "$(sha256sum "${candidate}" | awk '{print $1}')" == "${expected_sha256}" ]]; then
          readlink -f "${candidate}"
        fi
      done | sort -u
  )
  if [[ "${#candidates[@]}" -ne 1 ]]; then
    echo "expected one ${basename} with SHA-256 ${expected_sha256}, found ${#candidates[@]}" >&2
    return 1
  fi
  printf '%s\n' "${candidates[0]}"
}

resolve_unique_tree_sha256 () {
  local search_root=$1
  local marker_basename=$2
  local expected_sha256=$3
  PYTHONPATH="${PTC_WORKPACK}/src" "${PTC_TRAIN_PY}" - \
    "${search_root}" "${marker_basename}" "${expected_sha256}" <<'PY'
import sys
from pathlib import Path
from ptc_opd.stage1_artifact import sha256_tree

root = Path(sys.argv[1]).resolve()
marker_basename = sys.argv[2]
expected = sys.argv[3]
matches = []
seen = set()
for marker in sorted(root.rglob(marker_basename)):
    candidate = marker.parent.resolve()
    if candidate in seen:
        continue
    seen.add(candidate)
    try:
        if sha256_tree(candidate) == expected:
            matches.append(candidate)
    except (OSError, ValueError):
        pass
if len(matches) != 1:
    raise SystemExit(
        "expected one tree marked by {} with SHA-256 {}, found {}".format(
            marker_basename, expected, len(matches)
        )
    )
print(matches[0])
PY
}

export PTC_MUQ_CONFIG="${PTC_MUQ_EVAL_ROOT}/configs/A1_frozen_mlp.yaml"
export PTC_MUQ_STATE_DICT="$(resolve_unique_file_sha256 \
  "${PTC_EVAL_ASSET_ROOT}" model_state_dict.pt \
  4163ec9ba81bc0f7616611804414215220ae46fe1c8ae6fdff3f7919e7d21455)"
export PTC_MUQ_A1_DIR="$(dirname "${PTC_MUQ_STATE_DICT}")"
export PTC_MUQ_BACKBONE="$(resolve_unique_tree_sha256 \
  "${PTC_EVAL_ASSET_ROOT}" config.json \
  e505d08d56ac94204db81da4ce36c3ab7e54c2815db275d0fd53a7f5738171a4)"
export PTC_AUDIOBOX_CKPT="$(resolve_unique_file_sha256 \
  "${PTC_EVAL_ASSET_ROOT}" checkpoint.pt \
  a4931a7a01c3e6733352e9d85371835f03bf9135f8b31e1583c23538811d4a32)"
export PTC_CLAP_CKPT="<local>/models/ICASSP2027/laion_clap/music_audioset_epoch_15_esc_90.14.pt"
export PTC_MERT_SNAPSHOT="${HF_HUB_CACHE}/models--m-a-p--MERT-v1-95M/snapshots/12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"
export PTC_FMA_ROOT="<local>/data/ICASSP2027/fma_small/extracted/fma_small"

for required in \
  "${PTC_MUQ_EVAL_ROOT}" "${PTC_MUQ_CONFIG}" "${PTC_MUQ_STATE_DICT}" \
  "${PTC_MUQ_BACKBONE}" "${PTC_AUDIOBOX_CKPT}" "${PTC_CLAP_CKPT}" \
  "${PTC_MERT_SNAPSHOT}" "${PTC_FMA_ROOT}"; do
  test -e "${required}"
done

test "$(sha256sum "${PTC_MUQ_EVAL_ROOT}/configs/A1_frozen_mlp.yaml" | awk '{print $1}')" = \
  b605f0987844d813562967974085cabf65566844237062f6949b949b9cb717a8
test "$(sha256sum "${PTC_MUQ_EVAL_ROOT}/configs/base.yaml" | awk '{print $1}')" = \
  edb87805653cffaf625cab0a71deeac65987eb64e7949f36dff860ecc2fbb44b
test "$(sha256sum "${PTC_CLAP_CKPT}" | awk '{print $1}')" = \
  fae3e9c087f2909c28a09dc31c8dfcdacbc42ba44c70e972b58c1bd1caf6dedd

while read -r relative size digest; do
  test -f "${PTC_MERT_SNAPSHOT}/${relative}"
  test "$(stat -Lc '%s' "${PTC_MERT_SNAPSHOT}/${relative}")" = "${size}"
  test "$(sha256sum "${PTC_MERT_SNAPSHOT}/${relative}" | awk '{print $1}')" = \
    "${digest}"
done <<'PTC_MERT_SCIENTIFIC_PINS'
config.json 1817 ea2627c4c7825cd66f3c944b6b966331604c35928174e0100cd4a82829424e32
configuration_MERT.py 5340 ae0ec2bab8f59c724ba9878a7c20b67210189536ea62d34a56775968e9decb03
modeling_MERT.py 18033 6c3ee73cef6f0c30ef494f88d96f891fa6925ffe663fa391b512f4b57abecc6c
preprocessor_config.json 211 cc5a5e4a5d3b1a758a5ed984b2eaa15bb0522d811d44a9eed82bfca4baa0dc8f
pytorch_model.bin 377552987 a2b8b747f72c06e0595aeae41ae5473f4364938c6b39b2c58be38c48e6bd3fcd
PTC_MERT_SCIENTIFIC_PINS

mapfile -t PTC_MERT_FORBIDDEN < <(
  find -L "${PTC_MERT_SNAPSHOT}" -type f \( \
    -name model.safetensors -o -name '*.index.json' -o \
    -name 'pytorch_model-*.bin' -o -name 'model-*.safetensors' -o \
    -name tf_model.h5 -o -name flax_model.msgpack \
  \) -print | sort
)
test "${#PTC_MERT_FORBIDDEN[@]}" -eq 0
test "$(find -L "${PTC_MERT_SNAPSHOT}" -type f -name pytorch_model.bin -print | wc -l)" -eq 1

export PTC_MODEL_PINS="${PTC_WORKPACK}/artifacts/stage1/evaluator_model_pins"
test ! -e "${PTC_MODEL_PINS}"
"${PTC_FAD_PY}" "${PTC_WORKPACK}/scripts/build_stage1_eval_model_pins.py" build \
  --mert-snapshot "${PTC_MERT_SNAPSHOT}" \
  --clap-checkpoint "${PTC_CLAP_CKPT}" \
  --output-dir "${PTC_MODEL_PINS}"
"${PTC_FAD_PY}" "${PTC_WORKPACK}/scripts/build_stage1_eval_model_pins.py" verify \
  --mert-snapshot "${PTC_MERT_SNAPSHOT}" \
  --clap-checkpoint "${PTC_CLAP_CKPT}" \
  --artifact-dir "${PTC_MODEL_PINS}"

export PTC_FAD_REFERENCE="${PTC_WORKPACK}/artifacts/stage1/fad_reference"
test ! -e "${PTC_FAD_REFERENCE}"
export PTC_A1_MANIFEST_SHA256=$(sha256sum "${PTC_A1_MANIFEST}" | awk '{print $1}')
export PTC_A1_REPORT_SHA256=$(sha256sum "${PTC_A1_REPORT}" | awk '{print $1}')
"${PTC_FAD_PY}" "${PTC_WORKPACK}/scripts/build_stage1_fad_reference.py" build \
  --a1-manifest "${PTC_A1_MANIFEST}" \
  --a1-manifest-sha256 "${PTC_A1_MANIFEST_SHA256}" \
  --a1-report "${PTC_A1_REPORT}" \
  --a1-report-sha256 "${PTC_A1_REPORT_SHA256}" \
  --fma-root "${PTC_FMA_ROOT}" \
  --output-dir "${PTC_FAD_REFERENCE}"
"${PTC_FAD_PY}" "${PTC_WORKPACK}/scripts/build_stage1_fad_reference.py" verify \
  --a1-manifest "${PTC_A1_MANIFEST}" \
  --a1-report "${PTC_A1_REPORT}" \
  --artifact-dir "${PTC_FAD_REFERENCE}"
```

FADtk 只允许正式 pin 的 1.1.0 source identity；MERT revision 必须精确为上面的 40 位
commit，5 个科学文件 size/hash 必须全等且只能加载 `pytorch_model.bin`；CLAP
checkpoint 必须是已验收的 republished hash `fae3e9...6dedd`。

## 5. 统一的生成与质量评估模板

下面函数用于 base、teacher、LR 候选和六个 pilot condition。所有 output 都必须不存在：

```bash
stage1_generate_base_or_teacher () {
  local source_kind=$1
  local output_dir=$2
  "${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/generate_stage1_audio.py" \
    --source-kind "${source_kind}" \
    --base-checkpoint "${PTC_MUSICGEN_SMALL}" \
    --audiocraft-root "${PTC_AUDIOCRAFT}" \
    --cfg-scale-decision-dir "${PTC_CFG_SMALL}" \
    --eval-manifest-dir "${PTC_PILOT_MANIFEST}" \
    --output-dir "${output_dir}" --device cuda:0
  "${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/verify_stage1_generation.py" \
    --generation-dir "${output_dir}" \
    --eval-manifest-dir "${PTC_PILOT_MANIFEST}"
}

stage1_generate_trained () {
  local run_dir=$1
  local checkpoint_step=$2
  local output_dir=$3
  "${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/generate_stage1_audio.py" \
    --source-kind trained_no_cfg \
    --base-checkpoint "${PTC_MUSICGEN_SMALL}" \
    --audiocraft-root "${PTC_AUDIOCRAFT}" \
    --cfg-scale-decision-dir "${PTC_CFG_SMALL}" \
    --eval-manifest-dir "${PTC_PILOT_MANIFEST}" \
    --stage1-run "${run_dir}" --checkpoint-step "${checkpoint_step}" \
    --output-dir "${output_dir}" --device cuda:0
  "${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/verify_stage1_generation.py" \
    --generation-dir "${output_dir}" \
    --eval-manifest-dir "${PTC_PILOT_MANIFEST}"
}

stage1_quality_and_clap () {
  local generation_dir=$1
  local quality_dir=$2
  local metric_dir=$3
  "${PTC_QUALITY_PY}" "${PTC_WORKPACK}/scripts/eval_stage1_quality.py" run \
    --generation-dir "${generation_dir}" \
    --eval-manifest-dir "${PTC_PILOT_MANIFEST}" \
    --muq-eval-root "${PTC_MUQ_EVAL_ROOT}" \
    --muq-config "${PTC_MUQ_CONFIG}" \
    --muq-state-dict "${PTC_MUQ_STATE_DICT}" \
    --muq-backbone "${PTC_MUQ_BACKBONE}" \
    --audiobox-checkpoint "${PTC_AUDIOBOX_CKPT}" \
    --output-dir "${quality_dir}" --device cuda:0
  "${PTC_FAD_PY}" "${PTC_WORKPACK}/scripts/eval_stage1_clap.py" run \
    --generation-dir "${generation_dir}" \
    --eval-manifest-dir "${PTC_PILOT_MANIFEST}" \
    --quality-dir "${quality_dir}" \
    --clap-checkpoint "${PTC_CLAP_CKPT}" \
    --output-dir "${metric_dir}" --device cuda:0
  "${PTC_QUALITY_PY}" "${PTC_WORKPACK}/scripts/eval_stage1_quality.py" verify \
    --generation-dir "${generation_dir}" \
    --eval-manifest-dir "${PTC_PILOT_MANIFEST}" \
    --quality-dir "${quality_dir}"
  "${PTC_FAD_PY}" "${PTC_WORKPACK}/scripts/eval_stage1_clap.py" verify \
    --generation-dir "${generation_dir}" \
    --eval-manifest-dir "${PTC_PILOT_MANIFEST}" \
    --quality-dir "${quality_dir}" \
    --metric-dir "${metric_dir}"
}

stage1_diversity_fad () {
  local generation_dir=$1
  local output_dir=$2
  "${PTC_FAD_PY}" "${PTC_WORKPACK}/scripts/eval_stage1_diversity_fad.py" \
    --generation-dir "${generation_dir}" \
    --eval-manifest-dir "${PTC_PILOT_MANIFEST}" \
    --reference-dir "${PTC_FAD_REFERENCE}" \
    --a1-manifest "${PTC_A1_MANIFEST}" \
    --a1-report "${PTC_A1_REPORT}" \
    --model-pins-dir "${PTC_MODEL_PINS}" \
    --mert-snapshot "${PTC_MERT_SNAPSHOT}" \
    --clap-checkpoint "${PTC_CLAP_CKPT}" \
    --output-dir "${output_dir}" --device cuda:0
  "${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/verify_stage1_diversity_fad.py" \
    --artifact-dir "${output_dir}" \
    --generation-dir "${generation_dir}" \
    --eval-manifest-dir "${PTC_PILOT_MANIFEST}" \
    --reference-dir "${PTC_FAD_REFERENCE}" \
    --a1-manifest "${PTC_A1_MANIFEST}" \
    --a1-report "${PTC_A1_REPORT}" \
    --model-pins-dir "${PTC_MODEL_PINS}"
}
```

先对 `base_no_cfg` 与 `frozen_cfg_teacher` 各运行一次生成和 quality+CLAP，作为评估
pipeline qualification。两套输出均需 verifier 通过；base 输出之后必须原字节复用为
LR 和 small summary 的共同 anchor，不能重新生成一个“等价 base”。

```bash
export PTC_EVAL_QUAL="${PTC_WORKPACK}/artifacts/stage1/evaluator_qualification"
export PTC_BASE_GENERATION="${PTC_EVAL_QUAL}/base/generation"
export PTC_BASE_QUALITY="${PTC_EVAL_QUAL}/base/quality"
export PTC_BASE_METRIC="${PTC_EVAL_QUAL}/base/metric"
export PTC_TEACHER_GENERATION="${PTC_EVAL_QUAL}/teacher/generation"
export PTC_TEACHER_QUALITY="${PTC_EVAL_QUAL}/teacher/quality"
export PTC_TEACHER_METRIC="${PTC_EVAL_QUAL}/teacher/metric"
test ! -e "${PTC_EVAL_QUAL}"

stage1_generate_base_or_teacher base_no_cfg "${PTC_BASE_GENERATION}"
stage1_quality_and_clap \
  "${PTC_BASE_GENERATION}" "${PTC_BASE_QUALITY}" "${PTC_BASE_METRIC}"
stage1_generate_base_or_teacher frozen_cfg_teacher "${PTC_TEACHER_GENERATION}"
stage1_quality_and_clap \
  "${PTC_TEACHER_GENERATION}" "${PTC_TEACHER_QUALITY}" "${PTC_TEACHER_METRIC}"

export PTC_EVALQ_EVIDENCE_JSON="${PTC_CONTROLLER_MAP_ROOT}/evaluation_pipeline_qualification.attempt-1.evidence.json"
export PTC_EVALQ_DEPS_JSON="${PTC_CONTROLLER_MAP_ROOT}/evaluation_pipeline_qualification.attempt-1.dependencies.json"
stage1_write_path_map "${PTC_EVALQ_EVIDENCE_JSON}" \
  eval_manifest_dir "${PTC_PILOT_MANIFEST}" \
  musicgen_small_dir "${PTC_MUSICGEN_SMALL}" \
  audiocraft_dir "${PTC_AUDIOCRAFT}" small_cfg_dir "${PTC_CFG_SMALL}" \
  base_generation_dir "${PTC_BASE_GENERATION}" \
  base_quality_dir "${PTC_BASE_QUALITY}" base_metric_dir "${PTC_BASE_METRIC}" \
  teacher_generation_dir "${PTC_TEACHER_GENERATION}" \
  teacher_quality_dir "${PTC_TEACHER_QUALITY}" \
  teacher_metric_dir "${PTC_TEACHER_METRIC}"
stage1_write_path_map "${PTC_EVALQ_DEPS_JSON}" \
  pilot_eval_manifest "$(stage1_dependency_receipt pilot_eval_manifest)" \
  b1_prestability "$(stage1_dependency_receipt b1_prestability)"
stage1_controller_commit evaluation_pipeline_qualification 1 initial \
  "${PTC_EVALQ_EVIDENCE_JSON}" "${PTC_EVALQ_DEPS_JSON}"
```

该 receipt 不只要求 base/teacher 彼此一致：controller 会重算上述三个 retained
目录的实体身份，并要求两份 generation lineage 与 B1 对真实 checkpoint 审计得到的
canonical lineage 完全相等（包括加载后的 LM state 与 T5 identity）。

## 6. 通用八卡训练模板

```bash
stage1_train () {
  local mode=$1
  local learning_rate=$2
  local optimizer_steps=$3
  local output_dir=$4
  local prior_args=()
  if [[ "${mode}" == codebook100 || "${mode}" == ptc50 ]]; then
    prior_args=(--codebook-prior-artifact-dir "${PTC_A1_PRIOR}")
  fi
  unset MASTER_ADDR MASTER_PORT RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE
  unset GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
  unset SLURM_JOB_ID SLURM_PROCID SLURM_LOCALID SLURM_NTASKS
  unset PMI_RANK PMI_SIZE PMI_FD PMIX_RANK PMIX_NAMESPACE
  unset OMPI_COMM_WORLD_RANK OMPI_COMM_WORLD_SIZE
  unset OMPI_COMM_WORLD_LOCAL_RANK OMPI_COMM_WORLD_LOCAL_SIZE
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  "${PTC_TORCHRUN}" --standalone --nnodes=1 --nproc_per_node=8 \
    "${PTC_WORKPACK}/scripts/train_stage1.py" \
    --manifest "${PTC_TRAIN_MANIFEST}" \
    --student-checkpoint "${PTC_MUSICGEN_SMALL}" \
    --teacher-checkpoint "${PTC_MUSICGEN_SMALL}" \
    --audiocraft-root "${PTC_AUDIOCRAFT}" \
    --cfg-scale-decision-dir "${PTC_CFG_SMALL}" \
    --output-dir "${output_dir}" \
    --mode "${mode}" "${prior_args[@]}" \
    --seed 2027 --learning-rate "${learning_rate}" \
    --max-optimizer-steps "${optimizer_steps}" \
    --save-every 250 --log-every 1 \
    --rank-batch-size 2 --expected-world-size 8 \
    --grad-accum-steps 4 --effective-global-batch 64 \
    --weight-decay 0 \
    --duration-seconds 10 --codec-frame-rate 50 --token-frames 500 \
    --rollout-temperature 1 --rollout-top-k 250 --rollout-top-p 0 \
    --distillation-temperature 1 --grad-clip-norm 1 \
    --teacher-forward-mode batched --random-mask-namespace 5701 \
    --denominator-rtol 1e-6
  "${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/verify_stage1_run.py" \
    --run-dir "${output_dir}"
}
```

不要加 `--no-check-finite`，不要改 batch/accumulation/teacher-forward/temperature、
denominator tolerance 或 DDP policy。

没有 CLI flag 的字段同样由 receipt 中的 full-config verifier 冻结：AdamW
`betas=(0.9,0.95), eps=1e-8`；`linear_warmup_then_constant`、warmup 50 updates；forward KL；
teacher CFG 从 sealed small decision 读取且必须为 5；`find_unused_parameters=True`；
codebook prior 只准用于 `codebook100/ptc50` 并绑定 retained A1-R2，其他四种 mode 必须没有
prior。任何 `teacher_forward_mode=separate`、sampling/schedule/optimizer drift 或 partial
config 记录都会被 registered verifier 拒绝。

## 7. LR sweep、自动汇总与选择

node-0 只生成一次 UTC，并把该**字面值**发给 node-1～3；四机随后都执行变量与数组
定义。数组顺序永久固定为 `1e-6, 3e-6, 1e-5`：

```bash
export PTC_LR_UTC=$(date -u +%Y%m%dT%H%M%SZ)  # 仅 node-0 执行
export PTC_LR_RUN_ROOT="${PTC_WORKPACK}/runs/stage1/lr/${PTC_LR_UTC}"
export PTC_LR_EVAL_ROOT="${PTC_WORKPACK}/artifacts/stage1/lr_evaluation/${PTC_LR_UTC}"
export PTC_LR_SUMMARY="${PTC_WORKPACK}/artifacts/stage1/lr_summary/${PTC_LR_UTC}"
export PTC_LR_DECISION="${PTC_WORKPACK}/artifacts/stage1/lr_decision/${PTC_LR_UTC}"
PTC_LR_VALUES=(1e-6 3e-6 1e-5)
PTC_LR_RUNS=(
  "${PTC_LR_RUN_ROOT}/uniform100.lr-1e-6.seed-2027"
  "${PTC_LR_RUN_ROOT}/uniform100.lr-3e-6.seed-2027"
  "${PTC_LR_RUN_ROOT}/uniform100.lr-1e-5.seed-2027"
)
PTC_LR_GENERATIONS=(
  "${PTC_LR_EVAL_ROOT}/lr-1e-6/generation"
  "${PTC_LR_EVAL_ROOT}/lr-3e-6/generation"
  "${PTC_LR_EVAL_ROOT}/lr-1e-5/generation"
)
PTC_LR_QUALITIES=(
  "${PTC_LR_EVAL_ROOT}/lr-1e-6/quality"
  "${PTC_LR_EVAL_ROOT}/lr-3e-6/quality"
  "${PTC_LR_EVAL_ROOT}/lr-1e-5/quality"
)
PTC_LR_METRICS=(
  "${PTC_LR_EVAL_ROOT}/lr-1e-6/metric"
  "${PTC_LR_EVAL_ROOT}/lr-3e-6/metric"
  "${PTC_LR_EVAL_ROOT}/lr-1e-5/metric"
)
```

node-0、node-1、node-2 分别执行 index 0、1、2。三个训练 run 先作为
`lr_uniform_sweep` 整体登记；receipt 通过后，三节点才分别生成并独立验证 step 500，
再运行 quality+CLAP。node-3 不重新生成 base/teacher；§5 的共同 anchor 只读复用。

```bash
stage1_lr_candidate () {
  local index=$1
  test ! -e "${PTC_LR_RUNS[index]}"
  stage1_train uniform100 "${PTC_LR_VALUES[index]}" 500 "${PTC_LR_RUNS[index]}"
}

# node-0: stage1_lr_candidate 0
# node-1: stage1_lr_candidate 1
# node-2: stage1_lr_candidate 2

# 三个 official run 都已通过后，仅 node-0 登记 sweep：
export PTC_LRSWEEP_EVIDENCE_JSON="${PTC_CONTROLLER_MAP_ROOT}/lr_uniform_sweep.attempt-1.evidence.json"
export PTC_LRSWEEP_DEPS_JSON="${PTC_CONTROLLER_MAP_ROOT}/lr_uniform_sweep.attempt-1.dependencies.json"
stage1_write_path_map "${PTC_LRSWEEP_EVIDENCE_JSON}" \
  run_1e6 "${PTC_LR_RUNS[0]}" run_3e6 "${PTC_LR_RUNS[1]}" \
  run_1e5 "${PTC_LR_RUNS[2]}"
stage1_write_path_map "${PTC_LRSWEEP_DEPS_JSON}" \
  performance_benchmark "$(stage1_dependency_receipt performance_benchmark)" \
  evaluation_pipeline_qualification \
    "$(stage1_dependency_receipt evaluation_pipeline_qualification)"
stage1_controller_commit lr_uniform_sweep 1 initial \
  "${PTC_LRSWEEP_EVIDENCE_JSON}" "${PTC_LRSWEEP_DEPS_JSON}"

# sweep receipt 通过后，node-0/1/2 再分别运行 index 0/1/2 的固定 step-500 评估：
stage1_lr_evaluate () {
  local index=$1
  test ! -e "${PTC_LR_GENERATIONS[index]}"
  test ! -e "${PTC_LR_QUALITIES[index]}"
  test ! -e "${PTC_LR_METRICS[index]}"
  stage1_generate_trained "${PTC_LR_RUNS[index]}" 500 "${PTC_LR_GENERATIONS[index]}"
  stage1_quality_and_clap \
    "${PTC_LR_GENERATIONS[index]}" \
    "${PTC_LR_QUALITIES[index]}" \
    "${PTC_LR_METRICS[index]}"
}
# node-0: stage1_lr_evaluate 0
# node-1: stage1_lr_evaluate 1
# node-2: stage1_lr_evaluate 2
```

三节点全部成功后，node-0 以相同数组顺序运行完整聚合、summary verifier、decision
publisher 与 decision verifier：

```bash
test ! -e "${PTC_LR_SUMMARY}"
test ! -e "${PTC_LR_DECISION}"
"${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/build_stage1_lr_summary.py" \
  --eval-manifest-dir "${PTC_PILOT_MANIFEST}" \
  --base-generation-dir "${PTC_BASE_GENERATION}" \
  --base-quality-dir "${PTC_BASE_QUALITY}" \
  --base-metric-dir "${PTC_BASE_METRIC}" \
  --candidate-run-dir "${PTC_LR_RUNS[0]}" \
  --candidate-run-dir "${PTC_LR_RUNS[1]}" \
  --candidate-run-dir "${PTC_LR_RUNS[2]}" \
  --candidate-generation-dir "${PTC_LR_GENERATIONS[0]}" \
  --candidate-generation-dir "${PTC_LR_GENERATIONS[1]}" \
  --candidate-generation-dir "${PTC_LR_GENERATIONS[2]}" \
  --candidate-quality-dir "${PTC_LR_QUALITIES[0]}" \
  --candidate-quality-dir "${PTC_LR_QUALITIES[1]}" \
  --candidate-quality-dir "${PTC_LR_QUALITIES[2]}" \
  --candidate-metric-dir "${PTC_LR_METRICS[0]}" \
  --candidate-metric-dir "${PTC_LR_METRICS[1]}" \
  --candidate-metric-dir "${PTC_LR_METRICS[2]}" \
  --output-dir "${PTC_LR_SUMMARY}"

"${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/decide_stage1_lr.py" verify-summary \
  --summary-dir "${PTC_LR_SUMMARY}"

export PTC_LRSUM_EVIDENCE_JSON="${PTC_CONTROLLER_MAP_ROOT}/lr_evaluation_summary.attempt-1.evidence.json"
export PTC_LRSUM_DEPS_JSON="${PTC_CONTROLLER_MAP_ROOT}/lr_evaluation_summary.attempt-1.dependencies.json"
stage1_write_path_map "${PTC_LRSUM_EVIDENCE_JSON}" \
  eval_manifest_dir "${PTC_PILOT_MANIFEST}" \
  base_generation_dir "${PTC_BASE_GENERATION}" base_quality_dir "${PTC_BASE_QUALITY}" \
  base_metric_dir "${PTC_BASE_METRIC}" summary_dir "${PTC_LR_SUMMARY}" \
  candidate_1e6_run_dir "${PTC_LR_RUNS[0]}" \
  candidate_1e6_generation_dir "${PTC_LR_GENERATIONS[0]}" \
  candidate_1e6_quality_dir "${PTC_LR_QUALITIES[0]}" \
  candidate_1e6_metric_dir "${PTC_LR_METRICS[0]}" \
  candidate_3e6_run_dir "${PTC_LR_RUNS[1]}" \
  candidate_3e6_generation_dir "${PTC_LR_GENERATIONS[1]}" \
  candidate_3e6_quality_dir "${PTC_LR_QUALITIES[1]}" \
  candidate_3e6_metric_dir "${PTC_LR_METRICS[1]}" \
  candidate_1e5_run_dir "${PTC_LR_RUNS[2]}" \
  candidate_1e5_generation_dir "${PTC_LR_GENERATIONS[2]}" \
  candidate_1e5_quality_dir "${PTC_LR_QUALITIES[2]}" \
  candidate_1e5_metric_dir "${PTC_LR_METRICS[2]}"
stage1_write_path_map "${PTC_LRSUM_DEPS_JSON}" \
  lr_uniform_sweep "$(stage1_dependency_receipt lr_uniform_sweep)"
stage1_controller_commit lr_evaluation_summary 1 initial \
  "${PTC_LRSUM_EVIDENCE_JSON}" "${PTC_LRSUM_DEPS_JSON}"

"${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/decide_stage1_lr.py" seal \
  --summary-dir "${PTC_LR_SUMMARY}" \
  --output-dir "${PTC_LR_DECISION}"
"${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/decide_stage1_lr.py" verify \
  --summary-dir "${PTC_LR_SUMMARY}" \
  --decision-dir "${PTC_LR_DECISION}"

export PTC_LRDEC_EVIDENCE_JSON="${PTC_CONTROLLER_MAP_ROOT}/lr_decision.attempt-1.evidence.json"
export PTC_LRDEC_DEPS_JSON="${PTC_CONTROLLER_MAP_ROOT}/lr_decision.attempt-1.dependencies.json"
stage1_write_path_map "${PTC_LRDEC_EVIDENCE_JSON}" \
  decision_dir "${PTC_LR_DECISION}" summary_dir "${PTC_LR_SUMMARY}"
stage1_write_path_map "${PTC_LRDEC_DEPS_JSON}" \
  lr_evaluation_summary "$(stage1_dependency_receipt lr_evaluation_summary)"
stage1_controller_commit lr_decision 1 initial \
  "${PTC_LRDEC_EVIDENCE_JSON}" "${PTC_LRDEC_DEPS_JSON}"

export PTC_SELECTED_LR=$(PYTHONPATH="${PTC_WORKPACK}/src" "${PTC_TRAIN_PY}" - \
  "${PTC_LR_DECISION}" "${PTC_LR_SUMMARY}" <<'PY'
import sys
from pathlib import Path
from ptc_opd.stage1_control import LR_GRID, verify_lr_decision

decision = verify_lr_decision(Path(sys.argv[1]), Path(sys.argv[2]))
value = decision.get("selected_learning_rate")
if decision.get("status") != "selected" or value not in LR_GRID:
    raise SystemExit("no frozen eligible learning rate")
print(format(value, ".17g"))
PY
)
case "${PTC_SELECTED_LR}" in
  9.9999999999999995e-07|3.0000000000000001e-06|1.0000000000000001e-05) ;;
  *) echo "selected LR is outside the frozen grid: ${PTC_SELECTED_LR}" >&2; false ;;
esac
```

node-0 把 `PTC_LR_UTC` 与 `PTC_SELECTED_LR` 的字面值保存到 milestone capsule，并在
small pilot 前原样提供给 node-1～3；其他节点不得自行从未验证 JSON 或目录名猜 LR。

数字只允许由 raw 256-row metric artifacts 自动计算。三组 LR 都不 eligible 是红灯，
不得扩大 grid。恰好相同 Q_dev 才用较小 LR；不能用肉眼、FAD 或 checkpoint 挑选。

## 8. PTC500、B1.11 与 full B1 closure

node-0 为 PTC500 创建新 UTC，并使用上节 verifier 返回的唯一 LR：

```bash
export PTC_PTC500_UTC=$(date -u +%Y%m%dT%H%M%SZ)
export PTC_PTC500_RUN="${PTC_WORKPACK}/runs/stage1/ptc500/${PTC_PTC500_UTC}/ptc50.seed-2027"
export PTC_PTC500_DECISION="${PTC_WORKPACK}/artifacts/stage1/ptc500_stability/${PTC_PTC500_UTC}"
export PTC_B1_FULL="${PTC_WORKPACK}/artifacts/stage1/b1_full_closure/${PTC_PTC500_UTC}"
test ! -e "${PTC_PTC500_RUN}"
test ! -e "${PTC_PTC500_DECISION}"
test ! -e "${PTC_B1_FULL}"
stage1_train ptc50 "${PTC_SELECTED_LR}" 500 "${PTC_PTC500_RUN}"

export PTC_PTCTRAIN_EVIDENCE_JSON="${PTC_CONTROLLER_MAP_ROOT}/ptc500_training.attempt-1.evidence.json"
export PTC_PTCTRAIN_DEPS_JSON="${PTC_CONTROLLER_MAP_ROOT}/ptc500_training.attempt-1.dependencies.json"
stage1_write_path_map "${PTC_PTCTRAIN_EVIDENCE_JSON}" \
  run_dir "${PTC_PTC500_RUN}" lr_summary_dir "${PTC_LR_SUMMARY}" \
  lr_decision_dir "${PTC_LR_DECISION}"
stage1_write_path_map "${PTC_PTCTRAIN_DEPS_JSON}" \
  lr_decision "$(stage1_dependency_receipt lr_decision)"
stage1_controller_commit ptc500_training 1 initial \
  "${PTC_PTCTRAIN_EVIDENCE_JSON}" "${PTC_PTCTRAIN_DEPS_JSON}"
```

随后用 `decide_ptc500_stability.py consume/verify` 绑定该 run、LR summary/decision 与
`PTC_B1_FINAL`，再用 `finalize_b1_full_closure.py publish/verify` 合并 B1.1–B1.9 与
B1.11：

```bash
"${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/decide_ptc500_stability.py" consume \
  --run-dir "${PTC_PTC500_RUN}" \
  --lr-summary-dir "${PTC_LR_SUMMARY}" \
  --lr-decision-dir "${PTC_LR_DECISION}" \
  --b1-prestability-dir "${PTC_B1_FINAL}" \
  --output-dir "${PTC_PTC500_DECISION}"
"${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/decide_ptc500_stability.py" verify \
  --run-dir "${PTC_PTC500_RUN}" \
  --lr-summary-dir "${PTC_LR_SUMMARY}" \
  --lr-decision-dir "${PTC_LR_DECISION}" \
  --b1-prestability-dir "${PTC_B1_FINAL}" \
  --artifact-dir "${PTC_PTC500_DECISION}"

export PTC_PTCSTAB_EVIDENCE_JSON="${PTC_CONTROLLER_MAP_ROOT}/ptc500_stability.attempt-1.evidence.json"
export PTC_PTCSTAB_DEPS_JSON="${PTC_CONTROLLER_MAP_ROOT}/ptc500_stability.attempt-1.dependencies.json"
stage1_write_path_map "${PTC_PTCSTAB_EVIDENCE_JSON}" \
  artifact_dir "${PTC_PTC500_DECISION}" run_dir "${PTC_PTC500_RUN}" \
  lr_summary_dir "${PTC_LR_SUMMARY}" lr_decision_dir "${PTC_LR_DECISION}" \
  b1_dir "${PTC_B1_FINAL}"
stage1_write_path_map "${PTC_PTCSTAB_DEPS_JSON}" \
  ptc500_training "$(stage1_dependency_receipt ptc500_training)"
stage1_controller_commit ptc500_stability 1 initial \
  "${PTC_PTCSTAB_EVIDENCE_JSON}" "${PTC_PTCSTAB_DEPS_JSON}"

"${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/finalize_b1_full_closure.py" publish \
  --ptc500-artifact-dir "${PTC_PTC500_DECISION}" \
  --b1-prestability-dir "${PTC_B1_FINAL}" \
  --output-dir "${PTC_B1_FULL}"
"${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/finalize_b1_full_closure.py" verify \
  --ptc500-artifact-dir "${PTC_PTC500_DECISION}" \
  --b1-prestability-dir "${PTC_B1_FINAL}" \
  --artifact-dir "${PTC_B1_FULL}"

PYTHONPATH="${PTC_WORKPACK}/src" "${PTC_TRAIN_PY}" - "${PTC_B1_FULL}" <<'PY'
import sys
from pathlib import Path
from ptc_opd.stage1_artifact import load_json_strict

value = load_json_strict(Path(sys.argv[1]) / "b1_full_closure.json")
if value.get("full_b1_passed") is not True:
    raise SystemExit("combined B1 closure did not pass")
PY

export PTC_B1FULL_EVIDENCE_JSON="${PTC_CONTROLLER_MAP_ROOT}/b1_full_closure.attempt-1.evidence.json"
export PTC_B1FULL_DEPS_JSON="${PTC_CONTROLLER_MAP_ROOT}/b1_full_closure.attempt-1.dependencies.json"
stage1_write_path_map "${PTC_B1FULL_EVIDENCE_JSON}" \
  artifact_dir "${PTC_B1_FULL}" ptc500_dir "${PTC_PTC500_DECISION}" \
  b1_dir "${PTC_B1_FINAL}"
stage1_write_path_map "${PTC_B1FULL_DEPS_JSON}" \
  b1_prestability "$(stage1_dependency_receipt b1_prestability)" \
  ptc500_stability "$(stage1_dependency_receipt ptc500_stability)"
stage1_controller_commit b1_full_closure 1 initial \
  "${PTC_B1FULL_EVIDENCE_JSON}" "${PTC_B1FULL_DEPS_JSON}"
```

只有 combined artifact 明确写出 `full_b1_passed=true`，才启动 small pilot；不得手工
编辑 closure JSON。

## 9. 六个 small 1000-update jobs

node-0 只生成一次 small UTC，并把该字面值连同 `PTC_SELECTED_LR`、`PTC_B1_FULL` 的
绝对路径发给其他三机。六个方法数组顺序必须与 summary
consumer 相同：`uniform100, codebook100, random50, prefix50, disagreement50, ptc50`。

```bash
export PTC_SMALL_UTC=$(date -u +%Y%m%dT%H%M%SZ)  # 仅 node-0 执行
export PTC_SMALL_RUN_ROOT="${PTC_WORKPACK}/runs/stage1/small_pilot/${PTC_SMALL_UTC}"
export PTC_SMALL_EVAL_ROOT="${PTC_WORKPACK}/artifacts/stage1/small_pilot_evaluation/${PTC_SMALL_UTC}"
export PTC_SMALL_SUMMARY="${PTC_WORKPACK}/artifacts/stage1/small_pilot_summary/${PTC_SMALL_UTC}"
export PTC_SMALL_DECISION="${PTC_WORKPACK}/artifacts/stage1/small_pilot_decision/${PTC_SMALL_UTC}"
PTC_SMALL_METHODS=(uniform100 codebook100 random50 prefix50 disagreement50 ptc50)
PTC_SMALL_RUNS=()
PTC_SMALL_GENERATIONS=()
PTC_SMALL_QUALITIES=()
PTC_SMALL_METRICS=()
PTC_SMALL_DIVERSITIES=()
for method in "${PTC_SMALL_METHODS[@]}"; do
  PTC_SMALL_RUNS+=("${PTC_SMALL_RUN_ROOT}/${method}.seed-2027")
  PTC_SMALL_GENERATIONS+=("${PTC_SMALL_EVAL_ROOT}/${method}/generation")
  PTC_SMALL_QUALITIES+=("${PTC_SMALL_EVAL_ROOT}/${method}/quality")
  PTC_SMALL_METRICS+=("${PTC_SMALL_EVAL_ROOT}/${method}/metric")
  PTC_SMALL_DIVERSITIES+=("${PTC_SMALL_EVAL_ROOT}/${method}/diversity_fad")
done

stage1_small_train () {
  local index=$1
  test ! -e "${PTC_SMALL_RUNS[index]}"
  stage1_train "${PTC_SMALL_METHODS[index]}" "${PTC_SELECTED_LR}" 1000 \
    "${PTC_SMALL_RUNS[index]}"
}

stage1_small_evaluate () {
  local index=$1
  test ! -e "${PTC_SMALL_GENERATIONS[index]}"
  test ! -e "${PTC_SMALL_QUALITIES[index]}"
  test ! -e "${PTC_SMALL_METRICS[index]}"
  test ! -e "${PTC_SMALL_DIVERSITIES[index]}"
  stage1_generate_trained "${PTC_SMALL_RUNS[index]}" 1000 \
    "${PTC_SMALL_GENERATIONS[index]}"
  stage1_quality_and_clap \
    "${PTC_SMALL_GENERATIONS[index]}" \
    "${PTC_SMALL_QUALITIES[index]}" \
    "${PTC_SMALL_METRICS[index]}"
  stage1_diversity_fad \
    "${PTC_SMALL_GENERATIONS[index]}" \
    "${PTC_SMALL_DIVERSITIES[index]}"
}
```

在所有机器重新验证 `PTC_B1_FULL` 后执行两波；index 来自上面的冻结数组，不按完成速度
重排。训练 wave 1：node-0 `stage1_small_train 0`，node-1 `1`，node-2 `2`，node-3 `4`；
训练 wave 2：node-0 `stage1_small_train 5`，node-1 `3`。六个 official run 都完成后，
node-0 先执行以下唯一 training commit：

```bash
export PTC_SMALLTRAIN_EVIDENCE_JSON="${PTC_CONTROLLER_MAP_ROOT}/small_pilot_training.attempt-1.evidence.json"
export PTC_SMALLTRAIN_DEPS_JSON="${PTC_CONTROLLER_MAP_ROOT}/small_pilot_training.attempt-1.dependencies.json"
stage1_write_path_map "${PTC_SMALLTRAIN_EVIDENCE_JSON}" \
  uniform100_run_dir "${PTC_SMALL_RUNS[0]}" \
  codebook100_run_dir "${PTC_SMALL_RUNS[1]}" \
  random50_run_dir "${PTC_SMALL_RUNS[2]}" \
  prefix50_run_dir "${PTC_SMALL_RUNS[3]}" \
  disagreement50_run_dir "${PTC_SMALL_RUNS[4]}" \
  ptc50_run_dir "${PTC_SMALL_RUNS[5]}" \
  lr_summary_dir "${PTC_LR_SUMMARY}" lr_decision_dir "${PTC_LR_DECISION}" \
  b1_full_dir "${PTC_B1_FULL}"
stage1_write_path_map "${PTC_SMALLTRAIN_DEPS_JSON}" \
  b1_full_closure "$(stage1_dependency_receipt b1_full_closure)"
stage1_controller_commit small_pilot_training 1 initial \
  "${PTC_SMALLTRAIN_EVIDENCE_JSON}" "${PTC_SMALLTRAIN_DEPS_JSON}"
```

training receipt 通过后再并行执行六个 `stage1_small_evaluate`：沿用训练节点运行 0/1/2/4，
第二波 node-0 运行 5、node-1 运行 3；可调整机器以利用空闲 GPU，但每个 index 只能生成
一套 artifact。六套评估都通过后，node-0 登记 evaluation group：

```bash
export PTC_SMALLEVAL_EVIDENCE_JSON="${PTC_CONTROLLER_MAP_ROOT}/small_pilot_evaluation.attempt-1.evidence.json"
export PTC_SMALLEVAL_DEPS_JSON="${PTC_CONTROLLER_MAP_ROOT}/small_pilot_evaluation.attempt-1.dependencies.json"
PTC_SMALLEVAL_ARGS=(
  eval_manifest_dir "${PTC_PILOT_MANIFEST}"
  base_generation_dir "${PTC_BASE_GENERATION}"
  base_quality_dir "${PTC_BASE_QUALITY}" base_metric_dir "${PTC_BASE_METRIC}"
  reference_dir "${PTC_FAD_REFERENCE}" a1_manifest "${PTC_A1_MANIFEST}"
  a1_report "${PTC_A1_REPORT}" model_pins_dir "${PTC_MODEL_PINS}"
)
for index in 0 1 2 3 4 5; do
  method=${PTC_SMALL_METHODS[index]}
  PTC_SMALLEVAL_ARGS+=(
    "${method}_run_dir" "${PTC_SMALL_RUNS[index]}"
    "${method}_generation_dir" "${PTC_SMALL_GENERATIONS[index]}"
    "${method}_quality_dir" "${PTC_SMALL_QUALITIES[index]}"
    "${method}_metric_dir" "${PTC_SMALL_METRICS[index]}"
    "${method}_diversity_dir" "${PTC_SMALL_DIVERSITIES[index]}"
  )
done
stage1_write_path_map "${PTC_SMALLEVAL_EVIDENCE_JSON}" "${PTC_SMALLEVAL_ARGS[@]}"
stage1_write_path_map "${PTC_SMALLEVAL_DEPS_JSON}" \
  small_pilot_training "$(stage1_dependency_receipt small_pilot_training)"
stage1_controller_commit small_pilot_evaluation 1 initial \
  "${PTC_SMALLEVAL_EVIDENCE_JSON}" "${PTC_SMALLEVAL_DEPS_JSON}"
```

FAD 只检查六条 pipeline 都得到有限数，不参与排序；MERT 双 seed diversity 是预声明
科学 gate。

## 10. small summary 与最终决定

六方法都通过后 node-0 执行；所有重复参数均按冻结数组顺序出现恰好六次：

```bash
test ! -e "${PTC_SMALL_SUMMARY}"
test ! -e "${PTC_SMALL_DECISION}"
"${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/build_stage1_pilot_summary.py" \
  --eval-manifest-dir "${PTC_PILOT_MANIFEST}" \
  --lr-summary-dir "${PTC_LR_SUMMARY}" \
  --lr-decision-dir "${PTC_LR_DECISION}" \
  --base-generation-dir "${PTC_BASE_GENERATION}" \
  --base-quality-dir "${PTC_BASE_QUALITY}" \
  --base-metric-dir "${PTC_BASE_METRIC}" \
  --method-run-dir "${PTC_SMALL_RUNS[0]}" \
  --method-run-dir "${PTC_SMALL_RUNS[1]}" \
  --method-run-dir "${PTC_SMALL_RUNS[2]}" \
  --method-run-dir "${PTC_SMALL_RUNS[3]}" \
  --method-run-dir "${PTC_SMALL_RUNS[4]}" \
  --method-run-dir "${PTC_SMALL_RUNS[5]}" \
  --method-generation-dir "${PTC_SMALL_GENERATIONS[0]}" \
  --method-generation-dir "${PTC_SMALL_GENERATIONS[1]}" \
  --method-generation-dir "${PTC_SMALL_GENERATIONS[2]}" \
  --method-generation-dir "${PTC_SMALL_GENERATIONS[3]}" \
  --method-generation-dir "${PTC_SMALL_GENERATIONS[4]}" \
  --method-generation-dir "${PTC_SMALL_GENERATIONS[5]}" \
  --method-quality-dir "${PTC_SMALL_QUALITIES[0]}" \
  --method-quality-dir "${PTC_SMALL_QUALITIES[1]}" \
  --method-quality-dir "${PTC_SMALL_QUALITIES[2]}" \
  --method-quality-dir "${PTC_SMALL_QUALITIES[3]}" \
  --method-quality-dir "${PTC_SMALL_QUALITIES[4]}" \
  --method-quality-dir "${PTC_SMALL_QUALITIES[5]}" \
  --method-metric-dir "${PTC_SMALL_METRICS[0]}" \
  --method-metric-dir "${PTC_SMALL_METRICS[1]}" \
  --method-metric-dir "${PTC_SMALL_METRICS[2]}" \
  --method-metric-dir "${PTC_SMALL_METRICS[3]}" \
  --method-metric-dir "${PTC_SMALL_METRICS[4]}" \
  --method-metric-dir "${PTC_SMALL_METRICS[5]}" \
  --method-diversity-fad-dir "${PTC_SMALL_DIVERSITIES[0]}" \
  --method-diversity-fad-dir "${PTC_SMALL_DIVERSITIES[1]}" \
  --method-diversity-fad-dir "${PTC_SMALL_DIVERSITIES[2]}" \
  --method-diversity-fad-dir "${PTC_SMALL_DIVERSITIES[3]}" \
  --method-diversity-fad-dir "${PTC_SMALL_DIVERSITIES[4]}" \
  --method-diversity-fad-dir "${PTC_SMALL_DIVERSITIES[5]}" \
  --reference-dir "${PTC_FAD_REFERENCE}" \
  --a1-manifest "${PTC_A1_MANIFEST}" \
  --a1-report "${PTC_A1_REPORT}" \
  --model-pins-dir "${PTC_MODEL_PINS}" \
  --output-dir "${PTC_SMALL_SUMMARY}"

"${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/decide_stage1_pilot.py" verify-summary \
  --summary-dir "${PTC_SMALL_SUMMARY}"

export PTC_SMALLSUM_EVIDENCE_JSON="${PTC_CONTROLLER_MAP_ROOT}/small_pilot_summary.attempt-1.evidence.json"
export PTC_SMALLSUM_DEPS_JSON="${PTC_CONTROLLER_MAP_ROOT}/small_pilot_summary.attempt-1.dependencies.json"
PTC_SMALLSUM_ARGS=(
  eval_manifest_dir "${PTC_PILOT_MANIFEST}"
  lr_summary_dir "${PTC_LR_SUMMARY}" lr_decision_dir "${PTC_LR_DECISION}"
  base_generation_dir "${PTC_BASE_GENERATION}"
  base_quality_dir "${PTC_BASE_QUALITY}" base_metric_dir "${PTC_BASE_METRIC}"
  reference_dir "${PTC_FAD_REFERENCE}" a1_manifest "${PTC_A1_MANIFEST}"
  a1_report "${PTC_A1_REPORT}" model_pins_dir "${PTC_MODEL_PINS}"
  summary_dir "${PTC_SMALL_SUMMARY}"
)
for index in 0 1 2 3 4 5; do
  method=${PTC_SMALL_METHODS[index]}
  PTC_SMALLSUM_ARGS+=(
    "${method}_run_dir" "${PTC_SMALL_RUNS[index]}"
    "${method}_generation_dir" "${PTC_SMALL_GENERATIONS[index]}"
    "${method}_quality_dir" "${PTC_SMALL_QUALITIES[index]}"
    "${method}_metric_dir" "${PTC_SMALL_METRICS[index]}"
    "${method}_diversity_dir" "${PTC_SMALL_DIVERSITIES[index]}"
  )
done
stage1_write_path_map "${PTC_SMALLSUM_EVIDENCE_JSON}" "${PTC_SMALLSUM_ARGS[@]}"
stage1_write_path_map "${PTC_SMALLSUM_DEPS_JSON}" \
  small_pilot_evaluation "$(stage1_dependency_receipt small_pilot_evaluation)"
stage1_controller_commit small_pilot_summary 1 initial \
  "${PTC_SMALLSUM_EVIDENCE_JSON}" "${PTC_SMALLSUM_DEPS_JSON}"

"${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/decide_stage1_pilot.py" seal \
  --summary-dir "${PTC_SMALL_SUMMARY}" \
  --output-dir "${PTC_SMALL_DECISION}"
"${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/decide_stage1_pilot.py" verify \
  --summary-dir "${PTC_SMALL_SUMMARY}" \
  --decision-dir "${PTC_SMALL_DECISION}"

export PTC_SMALLDEC_EVIDENCE_JSON="${PTC_CONTROLLER_MAP_ROOT}/small_pilot_decision.attempt-1.evidence.json"
export PTC_SMALLDEC_DEPS_JSON="${PTC_CONTROLLER_MAP_ROOT}/small_pilot_decision.attempt-1.dependencies.json"
stage1_write_path_map "${PTC_SMALLDEC_EVIDENCE_JSON}" \
  decision_dir "${PTC_SMALL_DECISION}" summary_dir "${PTC_SMALL_SUMMARY}"
stage1_write_path_map "${PTC_SMALLDEC_DEPS_JSON}" \
  small_pilot_summary "$(stage1_dependency_receipt small_pilot_summary)"
stage1_controller_commit small_pilot_decision 1 initial \
  "${PTC_SMALLDEC_EVIDENCE_JSON}" "${PTC_SMALLDEC_DEPS_JSON}"
```

任一科学 gate 不通过即红灯，是论文结果，不是 retry 原因。若全部通过，唯一后续动作是
`prepare_medium_scale_gate_workpack_do_not_launch_medium`。

## 11. Ledger、receipt 与最小回报

§1 已 source `scripts/stage1_controller_shell.sh` 并初始化 revision 0；§1～10 对 16 个
节点逐一给出了 evidence/dependency map 与 `stage1_controller_commit`。每次 commit 固定
执行 registered verifier issue、live receipt verify、追加一个 immutable ledger revision、
完整 ledger-chain verify、current-only `next-action`。receipt 绑定当前 ledger、完整 evidence
字节 identity、当前 dependency terminal receipts、canonical contract 与 workpack manifest；
旧 ledger、手写 passed JSON、另一个合法但未绑定的 run 都不能授权。

node-0 随时可做只读总复核：

```bash
"${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/record_stage1_controller_receipt.py" \
  list-registry --workpack-root "${PTC_WORKPACK}" \
  --contract-path "${PTC_STAGE1_CONTRACT}"
export PTC_CURRENT_LEDGER=$(stage1_current_ledger)
"${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/record_stage1_controller_receipt.py" \
  verify-ledger --workpack-root "${PTC_WORKPACK}" \
  --contract-path "${PTC_STAGE1_CONTRACT}" --ledger-dir "${PTC_CURRENT_LEDGER}"
"${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/run_stage1_controller.py" \
  --workpack-root "${PTC_WORKPACK}" --contract "${PTC_STAGE1_CONTRACT}" \
  next-action --ledger-dir "${PTC_CURRENT_LEDGER}"
```

若一个已进入 registered verifier 的基础设施失败或 performance measurement
inconclusive 被 receipt 记为 `pending`，helper 在完成 ledger 登记后返回 rc 20 并锁住
下游。保留旧 evidence，修复基础设施，在新 UTC 产生完整新 evidence；复制该 stage 的
map block，把两个 map 文件名改为 `attempt-2`、evidence 路径改为新目录，dependency
仍用 `stage1_dependency_receipt` 读取。设置实际 stage ID 后执行：

```bash
export PTC_YELLOW_STAGE=performance_benchmark
stage1_controller_commit "${PTC_YELLOW_STAGE}" 2 yellow \
  "${PTC_CONTROLLER_MAP_ROOT}/${PTC_YELLOW_STAGE}.attempt-2.evidence.json" \
  "${PTC_CONTROLLER_MAP_ROOT}/${PTC_YELLOW_STAGE}.attempt-2.dependencies.json"
```

若 attempt 2 仍 pending，同样使用 attempt 3 / `yellow`；这已消耗两次 yellow 修复，
attempt 4 会在 verifier 前被拒绝。其他 stage 只把 `PTC_YELLOW_STAGE` 换成 registry 中
该 stage 的精确 ID，并复用其原 block 的完整 key set；不得删 key 或改 dependency。

这里的 ledger 是 verifier-attempt ledger，不是 GPU launch reservation system。只有所需
evidence 全部存在、正式进入注册 verifier 的 initial/yellow attempt 才由 controller 机器
计数。若 producer 在此之前中断或只产生半套输出，保留其不可变 attempt/`FAILED` logs，写入
capsule 并人工累计；同一 DAG 节点仍最多两次 yellow 修复。每次必须使用新 UTC/new output，
不得以“没有 receipt”为由无限重启。若根因是
managed runner/consumer/controller/verifier 源码 bug，停止并等待最小 additive patch 与新
workpack seal；旧 ledger 不得在 manifest 改变后冒充可原地继续。

绿灯只发 `../AUTONOMY_POLICY.md` §6 的 milestone capsule。红灯只回传 decision/seal、
verifier 输出和最小失败复现；完整 run tree、WAV、checkpoint 和 cache 默认留在远端。
