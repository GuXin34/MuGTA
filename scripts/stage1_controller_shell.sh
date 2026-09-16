#!/usr/bin/env bash
# Shell helpers for the immutable Stage-1 verifier-receipt/ledger chain.
# Source this file; do not execute it as a standalone orchestrator.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source this file from the frozen Stage-1 runbook shell" >&2
  exit 64
fi

: "${PTC_WORKPACK:?PTC_WORKPACK must be exported}"
: "${PTC_TRAIN_PY:?PTC_TRAIN_PY must be exported}"

export PTC_STAGE1_CONTRACT="${PTC_WORKPACK}/configs/stage1_autonomy_contract.json"
export PTC_CONTROLLER_ROOT="${PTC_WORKPACK}/artifacts/stage1/controller"
export PTC_CONTROLLER_MAP_ROOT="${PTC_WORKPACK}/console_logs/stage1/controller_maps"
export PTC_RECEIPT_ROOT="${PTC_CONTROLLER_ROOT}/receipts"
mkdir -p "${PTC_CONTROLLER_ROOT}" "${PTC_CONTROLLER_MAP_ROOT}" "${PTC_RECEIPT_ROOT}"

stage1_current_ledger () {
  PYTHONPATH="${PTC_WORKPACK}/src" "${PTC_TRAIN_PY}" - \
    "${PTC_CONTROLLER_ROOT}/CONTROLLER_AUTHORITY.json" <<'PY'
import sys
from pathlib import Path
from ptc_opd.stage1_artifact import load_json_strict

value = load_json_strict(Path(sys.argv[1]))
reference = value.get("current_ledger")
if not isinstance(reference, dict) or not isinstance(reference.get("artifact_dir"), str):
    raise SystemExit("controller authority has no current ledger")
print(reference["artifact_dir"])
PY
}

stage1_controller_init () {
  local initial="${PTC_CONTROLLER_ROOT}/ledger.r0000"
  test ! -e "${PTC_CONTROLLER_ROOT}/CONTROLLER_AUTHORITY.json"
  test ! -e "${initial}"
  "${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/record_stage1_controller_receipt.py" \
    init-ledger --workpack-root "${PTC_WORKPACK}" \
    --contract-path "${PTC_STAGE1_CONTRACT}" --output-dir "${initial}"
  "${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/record_stage1_controller_receipt.py" \
    verify-ledger --workpack-root "${PTC_WORKPACK}" \
    --contract-path "${PTC_STAGE1_CONTRACT}" --ledger-dir "${initial}"
  "${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/run_stage1_controller.py" \
    --workpack-root "${PTC_WORKPACK}" --contract "${PTC_STAGE1_CONTRACT}" \
    next-action --ledger-dir "${initial}"
}

stage1_write_path_map () {
  local output=$1
  shift
  test ! -e "${output}"
  "${PTC_TRAIN_PY}" - "${output}" "$@" <<'PY'
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[1])
items = sys.argv[2:]
if len(items) % 2:
    raise SystemExit("path map requires KEY PATH pairs")
value = {}
for index in range(0, len(items), 2):
    key, raw_path = items[index], items[index + 1]
    if not key or key in value or not raw_path:
        raise SystemExit("path-map key/path is empty or duplicated")
    value[key] = str(Path(raw_path).expanduser().absolute())
output.parent.mkdir(parents=True, exist_ok=True)
descriptor = os.open(str(output), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
    json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
PY
}

stage1_dependency_receipt () {
  local stage_id=$1
  local ledger
  ledger=$(stage1_current_ledger)
  PYTHONPATH="${PTC_WORKPACK}/src" "${PTC_TRAIN_PY}" - \
    "${ledger}/controller_ledger.json" "${stage_id}" <<'PY'
import sys
from pathlib import Path
from ptc_opd.stage1_artifact import load_json_strict

value = load_json_strict(Path(sys.argv[1]))
entry = value.get("stages", {}).get(sys.argv[2])
reference = entry.get("terminal_receipt") if isinstance(entry, dict) else None
if (not isinstance(reference, dict) or reference.get("stage_status") != "passed"
        or not isinstance(reference.get("artifact_dir"), str)):
    raise SystemExit("dependency has no passed terminal receipt: " + sys.argv[2])
print(reference["artifact_dir"])
PY
}

stage1_controller_commit () {
  local stage_id=$1
  local attempt_number=$2
  local attempt_kind=$3
  local evidence_json=$4
  local dependencies_json=$5
  local ledger revision next_revision receipt next_ledger stage_status

  ledger=$(stage1_current_ledger)
  revision=$(PYTHONPATH="${PTC_WORKPACK}/src" "${PTC_TRAIN_PY}" - \
    "${ledger}/controller_ledger.json" <<'PY'
import sys
from pathlib import Path
from ptc_opd.stage1_artifact import load_json_strict
value = load_json_strict(Path(sys.argv[1]))
revision = value.get("revision")
if type(revision) is not int or revision < 0:
    raise SystemExit("invalid ledger revision")
print(revision)
PY
  )
  next_revision=$((revision + 1))
  receipt="${PTC_RECEIPT_ROOT}/${stage_id}.attempt-${attempt_number}"
  next_ledger="${PTC_CONTROLLER_ROOT}/ledger.r$(printf '%04d' "${next_revision}")"
  test ! -e "${receipt}"
  test ! -e "${next_ledger}"

  "${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/record_stage1_controller_receipt.py" \
    issue --workpack-root "${PTC_WORKPACK}" \
    --contract-path "${PTC_STAGE1_CONTRACT}" \
    --stage-id "${stage_id}" --evidence-json "${evidence_json}" \
    --dependency-receipts-json "${dependencies_json}" \
    --attempt-number "${attempt_number}" --attempt-kind "${attempt_kind}" \
    --ledger-dir "${ledger}" --output-dir "${receipt}"
  "${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/record_stage1_controller_receipt.py" \
    verify --workpack-root "${PTC_WORKPACK}" \
    --contract-path "${PTC_STAGE1_CONTRACT}" --receipt-dir "${receipt}"
  "${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/record_stage1_controller_receipt.py" \
    record --workpack-root "${PTC_WORKPACK}" \
    --contract-path "${PTC_STAGE1_CONTRACT}" --ledger-dir "${ledger}" \
    --receipt-dir "${receipt}" --output-dir "${next_ledger}"
  "${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/record_stage1_controller_receipt.py" \
    verify-ledger --workpack-root "${PTC_WORKPACK}" \
    --contract-path "${PTC_STAGE1_CONTRACT}" --ledger-dir "${next_ledger}"
  "${PTC_TRAIN_PY}" "${PTC_WORKPACK}/scripts/run_stage1_controller.py" \
    --workpack-root "${PTC_WORKPACK}" --contract "${PTC_STAGE1_CONTRACT}" \
    next-action --ledger-dir "${next_ledger}"

  stage_status=$(PYTHONPATH="${PTC_WORKPACK}/src" "${PTC_TRAIN_PY}" - \
    "${receipt}/verifier_receipt.json" <<'PY'
import sys
from pathlib import Path
from ptc_opd.stage1_artifact import load_json_strict
value = load_json_strict(Path(sys.argv[1]))
status = value.get("stage_status")
if status not in {"passed", "pending", "failed", "inconclusive"}:
    raise SystemExit("unclassified receipt stage status")
print(status)
PY
  )
  if [[ "${stage_status}" == passed ]]; then
    return 0
  fi
  echo "stage ${stage_id} recorded as ${stage_status}; downstream remains locked" >&2
  if [[ "${stage_status}" == pending ]]; then
    return 20
  fi
  return 21
}
