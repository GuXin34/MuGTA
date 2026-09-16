#!/usr/bin/env bash
# Execute the formal B1 pre-stability chain on one H20 node.  No Slurm.
set -euo pipefail

WORKPACK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

required_variables=(
  PTC_PYTHON
  B1_CHECKPOINT
  AUDIOCRAFT_ROOT
  PROBE_MANIFEST
  DEV_MANIFEST
  CFG_DECISION_DIR
  CODEBOOK_PRIOR_DIR
  A2_PROBE_DIR
  NODE3_EVIDENCE_DIR
  T5_CLOSURE_ARTIFACT_DIR
  B1_OUTPUT_ROOT
)
for variable_name in "${required_variables[@]}"; do
  if [[ -z "${!variable_name:-}" ]]; then
    echo "missing required environment variable: ${variable_name}" >&2
    exit 2
  fi
done
if [[ ! -x "${PTC_PYTHON}" ]]; then
  echo "PTC_PYTHON is not an executable file: ${PTC_PYTHON}" >&2
  exit 2
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1

"${PTC_PYTHON}" "${WORKPACK_ROOT}/scripts/seal_workpack.py" verify \
  --root "${WORKPACK_ROOT}"
"${PTC_PYTHON}" "${WORKPACK_ROOT}/scripts/check_workpack.py"

attempt_utc="$(date -u +%Y%m%dT%H%M%SZ)"
attempt_root="${B1_OUTPUT_ROOT%/}/${attempt_utc}"
log_root="${attempt_root}.console_logs"
mkdir -p "${log_root}"

single_artifact="${attempt_root}/single_gpu"
distributed_artifact="${attempt_root}/distributed_8gpu"
final_artifact="${attempt_root}/b1_prestability"

# A single-card audit must not inherit any previous torchrun rendezvous state.
unset MASTER_ADDR MASTER_PORT RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
unset SLURM_JOB_ID SLURM_PROCID SLURM_LOCALID SLURM_NTASKS
unset PMI_RANK PMI_SIZE PMI_FD PMIX_RANK PMIX_NAMESPACE
unset OMPI_COMM_WORLD_RANK OMPI_COMM_WORLD_SIZE OMPI_COMM_WORLD_LOCAL_RANK OMPI_COMM_WORLD_LOCAL_SIZE
CUDA_VISIBLE_DEVICES=0 "${PTC_PYTHON}" \
  "${WORKPACK_ROOT}/scripts/run_b1_single_gpu_audit.py" \
  --checkpoint "${B1_CHECKPOINT}" \
  --audiocraft-root "${AUDIOCRAFT_ROOT}" \
  --probe-manifest "${PROBE_MANIFEST}" \
  --dev-manifest "${DEV_MANIFEST}" \
  --cfg-scale-decision-dir "${CFG_DECISION_DIR}" \
  --codebook-prior-artifact-dir "${CODEBOOK_PRIOR_DIR}" \
  --a2-probe-dir "${A2_PROBE_DIR}" \
  --node3-evidence-dir "${NODE3_EVIDENCE_DIR}" \
  --t5-closure-artifact-dir "${T5_CLOSURE_ARTIFACT_DIR}" \
  --output-dir "${single_artifact}" \
  2>&1 | tee "${log_root}/single_gpu.log"
"${PTC_PYTHON}" "${WORKPACK_ROOT}/scripts/verify_b1_prestability.py" \
  single "${single_artifact}" | tee "${log_root}/single_gpu.verify.log"

# torchrun owns the rank variables below.  The formal runner rejects Slurm and
# anything other than a standalone, one-node, eight-H20 process group.
unset MASTER_ADDR MASTER_PORT RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
unset SLURM_JOB_ID SLURM_PROCID SLURM_LOCALID SLURM_NTASKS
unset PMI_RANK PMI_SIZE PMI_FD PMIX_RANK PMIX_NAMESPACE
unset OMPI_COMM_WORLD_RANK OMPI_COMM_WORLD_SIZE OMPI_COMM_WORLD_LOCAL_RANK OMPI_COMM_WORLD_LOCAL_SIZE
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_ASYNC_ERROR_HANDLING=1
"${PTC_PYTHON}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  "${WORKPACK_ROOT}/scripts/run_b1_distributed_audit.py" \
  --single-artifact-dir "${single_artifact}" \
  --checkpoint "${B1_CHECKPOINT}" \
  --audiocraft-root "${AUDIOCRAFT_ROOT}" \
  --output-dir "${distributed_artifact}" \
  2>&1 | tee "${log_root}/distributed_8gpu.log"
"${PTC_PYTHON}" "${WORKPACK_ROOT}/scripts/verify_b1_prestability.py" \
  distributed "${distributed_artifact}" | tee "${log_root}/distributed_8gpu.verify.log"

"${PTC_PYTHON}" "${WORKPACK_ROOT}/scripts/finalize_b1_prestability.py" \
  --single-artifact-dir "${single_artifact}" \
  --distributed-artifact-dir "${distributed_artifact}" \
  --output-dir "${final_artifact}" \
  2>&1 | tee "${log_root}/finalize.log"
"${PTC_PYTHON}" "${WORKPACK_ROOT}/scripts/verify_b1_prestability.py" \
  final "${final_artifact}" | tee "${log_root}/final.verify.log"

echo "B1 pre-stability evidence: ${final_artifact}"
echo "Console logs (outside closed artifacts): ${log_root}"
echo "B1.11 remains pending; full_b1_passed is intentionally false."
