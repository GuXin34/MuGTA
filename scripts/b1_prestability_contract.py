#!/usr/bin/env python3
"""Pure-stdlib contract helpers for the B1 pre-stability gate.

This module deliberately does not import torch, AudioCraft, or NumPy.  The
producer and the independent consumer share only byte-level artifact framing
and immutable schema constants; numerical evidence is checked independently by
``verify_b1_prestability.py``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Dict, Iterator, Mapping, Sequence, Tuple


SINGLE_SCHEMA = "ptc-opd-b1-real-checkpoint-single-v1"
DISTRIBUTED_SCHEMA = "ptc-opd-b1-real-checkpoint-ddp-v1"
SUMMARY_SCHEMA = "ptc-opd-b1-prestability-summary-v1"
SEAL_SCHEMA = "ptc-opd-b1-artifact-seal-v1"

SINGLE_STATUS = "b1_single_gpu_1_6_passed"
DISTRIBUTED_STATUS = "b1_real_checkpoint_ddp_9_passed"
SUMMARY_STATUS = "b1_prestability_passed"

SINGLE_CASES: Tuple[str, ...] = (
    "B1.1_cfg1_identical_weights_zero_kl",
    "B1.2_all_selected_uniform_equivalence",
    "B1.3_ptc_equals_disagreement_equal_weights",
    "B1.4_random_js_counts_and_stable_ties",
    "B1.5_padding_nan_and_zero_invalid_gradient",
    "B1.6_pattern_roundtrip_frame_codebook_identity",
)

RETAINED_NODE3_ITEMS: Tuple[str, ...] = (
    "B1.7_frozen_teacher_hash",
    "B1.8_resume_first_batch_and_gate",
    "B1.10_resume_state_lifecycle",
)

PENDING_B1_ITEM = "B1.11_ptc_stability_500_updates"

# Scientific thresholds are copied verbatim from configs/stage1_matrix.yaml.
# They are constants, not CLI arguments.
CFG1_MEAN_KL_LT = 1.0e-6
CFG1_MAX_VALID_KL_LT = 1.0e-5
RELATIVE_LOSS_ERROR_LT = 1.0e-6
RELATIVE_GRADIENT_ERROR_LT = 1.0e-5

# This is an audit-identifiability guard rather than a scientific threshold.
# A real-checkpoint DDP audit is inconclusive if the intentionally wrong mean of
# unequal local ratios happens to equal the frozen global ratio this closely.
MEAN_OF_MEANS_SENSITIVITY_GT = 1.0e-8

PRIMARY_CODEBOOKS = 4
PRIMARY_FRAMES = 500
PRIMARY_CARD = 2048
SINGLE_BATCH_SIZE = 2
DISTRIBUTED_WORLD_SIZE = 8
ROLLOUT_SEED = 31001
ROLLOUT_TEMPERATURE = 1.0
ROLLOUT_TOP_K = 250
ROLLOUT_TOP_P = 0.0
SELECTOR_RHO = 0.5
RANDOM_NAMESPACE = 5701
RANDOM_RUN_SEED = 2027
RANDOM_OPTIMIZER_STEP = 0
PADDING_TOKEN_MUTATION_OFFSET = 17
SINGLE_PADDING_LENGTHS: Tuple[int, ...] = (487, 461)
DISTRIBUTED_PADDING_LENGTHS: Tuple[int, ...] = (
    500,
    487,
    474,
    461,
    448,
    435,
    422,
    409,
)

# Frozen upstream facts already adjudicated in this project.
EXPECTED_A2_PROBE_SEAL_SHA256 = (
    "d417dbbdd3519b54e6f46f3141e3bb2787de3b8f70f66b1344492edb8a159c42"
)
EXPECTED_A1_R2_ARTIFACT_SEAL_SHA256 = (
    "6bff7e518562e597962dbfdd2eb61679923dfe9559423db2a6e37e4602c96ba6"
)
EXPECTED_T5_CLOSURE_SEAL_SHA256 = (
    "850f0fab6f8e757ca3d82a01e27c1f3361338d2b75f98db825a48de5853ec7f4"
)
EXPECTED_SMALL_CFG_DECISION_SHA256 = (
    "731d0a00ac519079b40889e4fd79ec5ee450b1da8507e32ec815b042f0bb96d8"
)

SINGLE_MEMBERS: Tuple[str, ...] = (
    "audit_inputs.npz",
    "single_gpu_results.json",
)
DISTRIBUTED_MEMBERS: Tuple[str, ...] = ("distributed_results.json",)
FINAL_MEMBERS: Tuple[str, ...] = (
    "audit_inputs.npz",
    "single_gpu_results.json",
    "single_gpu_artifact_seal.json",
    "distributed_results.json",
    "distributed_artifact_seal.json",
    "b1_prestability_summary.json",
)


class ContractError(ValueError):
    """Raised when a frozen B1 artifact contract is not satisfied."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _reject_constant(value: str) -> None:
    raise ContractError("JSON contains forbidden non-finite constant {}".format(value))


def _unique_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("JSON contains duplicate key {!r}".format(key))
        result[key] = value
    return result


def load_json(path: Path, label: str = "JSON") -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ContractError("{} must be a regular non-symlink file".format(label))
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(
                stream,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("{} is not strict UTF-8 JSON".format(label)) from exc
    if not isinstance(value, dict):
        raise ContractError("{} must contain one JSON object".format(label))
    return value


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = stable_json_bytes(dict(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def regular_file_identity(path: Path) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ContractError("artifact member is not a regular file: {}".format(path))
    return {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


@contextlib.contextmanager
def staged_output_directory(target: Path) -> Iterator[Path]:
    """Create beside *target* and publish the complete directory by one rename."""

    target = target.expanduser().resolve()
    if target.exists():
        raise FileExistsError("refusing to overwrite output {}".format(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(
        tempfile.mkdtemp(prefix=".{}.partial.".format(target.name), dir=str(target.parent))
    )
    try:
        yield partial
        if target.exists():
            raise FileExistsError("output target appeared during staging")
        os.replace(str(partial), str(target))
    except BaseException:
        if partial.exists():
            shutil.rmtree(str(partial))
        raise


def write_artifact_seal(
    directory: Path,
    *,
    status: str,
    member_names: Sequence[str],
) -> Path:
    names = tuple(member_names)
    if len(set(names)) != len(names):
        raise ContractError("artifact member names are not unique")
    members = {
        name: regular_file_identity(directory / name)
        for name in names
    }
    path = directory / "artifact_seal.json"
    write_json_exclusive(
        path,
        {
            "schema_version": SEAL_SCHEMA,
            "status": status,
            "members": members,
        },
    )
    return path


def verify_artifact_directory(
    directory: Path,
    *,
    status: str,
    member_names: Sequence[str],
) -> Dict[str, Any]:
    supplied = directory.expanduser()
    if supplied.is_symlink():
        raise ContractError("artifact directory must not be a symlink")
    resolved = supplied.resolve(strict=True)
    if not resolved.is_dir():
        raise ContractError("artifact path must be a directory")
    expected = set(member_names) | {"artifact_seal.json"}
    entries = list(resolved.iterdir())
    observed = {entry.name for entry in entries}
    if observed != expected:
        raise ContractError(
            "artifact members differ; missing={}, unexpected={}".format(
                sorted(expected - observed), sorted(observed - expected)
            )
        )
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ContractError("artifact directory contains a non-regular member")
    seal = load_json(resolved / "artifact_seal.json", "artifact seal")
    if set(seal) != {"schema_version", "status", "members"}:
        raise ContractError("artifact seal field set differs")
    if seal.get("schema_version") != SEAL_SCHEMA or seal.get("status") != status:
        raise ContractError("artifact seal schema/status differs")
    members = seal.get("members")
    if not isinstance(members, dict) or set(members) != set(member_names):
        raise ContractError("artifact seal member set differs")
    for name in member_names:
        item = members.get(name)
        if not isinstance(item, dict) or set(item) != {"size_bytes", "sha256"}:
            raise ContractError("seal identity malformed for {}".format(name))
        if item != regular_file_identity(resolved / name):
            raise ContractError("artifact member identity differs for {}".format(name))
    return {
        "directory": str(resolved),
        "seal_sha256": sha256_file(resolved / "artifact_seal.json"),
        "members": members,
    }


def require_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError("{} must be a real number".format(label))
    number = float(value)
    if not math.isfinite(number):
        raise ContractError("{} must be finite".format(label))
    return number


def require_pass_case(case: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(case, dict) or case.get("status") != "passed":
        raise ContractError("{} did not pass".format(label))
    return case


def validate_node3_evidence(directory: Path) -> Dict[str, Any]:
    """Validate the retained 15/15 node-3 evidence without changing it."""

    root = directory.expanduser().resolve(strict=True)
    status_path = root / "STATUS.json"
    sums_path = root / "SHA256SUMS.txt"
    sums_sidecar_path = root / "SHA256SUMS.txt.sha256"
    verify_path = root / "SHA256SUMS.verify.log"
    status = load_json(status_path, "node-3 STATUS")
    if (
        status.get("status") != "passed"
        or status.get("gate_count") != 15
        or status.get("passed_count") != 15
        or status.get("failure") is not None
    ):
        raise ContractError("node-3 STATUS is not a clean 15/15 pass")
    gates = status.get("gates")
    if not isinstance(gates, dict) or len(gates) != 15:
        raise ContractError("node-3 gate map is incomplete")
    if any(not isinstance(item, dict) or item.get("status") != "passed" for item in gates.values()):
        raise ContractError("node-3 gate map contains a non-pass")
    if sums_path.is_symlink() or not sums_path.is_file():
        raise ContractError("node-3 SHA256SUMS.txt is missing")
    checked = 0
    covered_names = set()
    for raw_line in sums_path.read_text(encoding="utf-8").splitlines():
        if not raw_line:
            continue
        try:
            digest, relative = raw_line.split("  ", 1)
        except ValueError as exc:
            raise ContractError("malformed node-3 checksum line") from exc
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ContractError("malformed node-3 SHA-256")
        relative_path = PurePosixPath(relative)
        if (
            not relative
            or "\\" in relative
            or relative_path.is_absolute()
            or relative_path.as_posix() != relative
            or any(part in ("", ".", "..") for part in relative_path.parts)
        ):
            raise ContractError("node-3 checksum path is non-canonical")
        member = root / relative
        if member.is_symlink() or not member.is_file() or sha256_file(member) != digest:
            raise ContractError("node-3 checksum mismatch for {}".format(relative))
        if relative in covered_names:
            raise ContractError("node-3 checksum inventory contains a duplicate path")
        covered_names.add(relative)
        checked += 1
    if checked == 0:
        raise ContractError("node-3 checksum inventory is empty")
    if sums_sidecar_path.is_symlink() or not sums_sidecar_path.is_file():
        raise ContractError("node-3 SHA256SUMS sidecar is missing")
    expected_sidecar = "{}  SHA256SUMS.txt\n".format(sha256_file(sums_path))
    if sums_sidecar_path.read_text(encoding="utf-8", errors="strict") != expected_sidecar:
        raise ContractError("node-3 SHA256SUMS sidecar differs")
    if verify_path.is_symlink() or not verify_path.is_file():
        raise ContractError("node-3 checksum verification log is missing")
    verify_text = verify_path.read_text(encoding="utf-8", errors="strict")
    verify_lines = dict(
        line.split("=", 1)
        for line in verify_text.splitlines()
        if "=" in line
    )
    if (
        verify_lines.get("status") != "passed"
        or verify_lines.get("verified_files") != str(checked)
    ):
        raise ContractError("node-3 checksum verification log did not pass")
    observed_names = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ContractError("node-3 evidence contains a symbolic link")
        if path.is_file():
            observed_names.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise ContractError("node-3 evidence contains a non-regular entry")
    expected_names = covered_names | {
        "SHA256SUMS.txt",
        "SHA256SUMS.txt.sha256",
        "SHA256SUMS.verify.log",
    }
    if observed_names != expected_names:
        raise ContractError(
            "node-3 evidence closed set differs; missing={}, unexpected={}".format(
                sorted(expected_names - observed_names),
                sorted(observed_names - expected_names),
            )
        )
    return {
        "status_sha256": sha256_file(status_path),
        "sha256sums_sha256": sha256_file(sums_path),
        "sha256sums_sidecar_sha256": sha256_file(sums_sidecar_path),
        "verify_log_sha256": sha256_file(verify_path),
        "verified_member_count": checked,
        "schema_version": status.get("schema_version"),
        "b05_policy_schema_version": status.get("b05_policy_schema_version"),
        "gate_count": 15,
        "passed_count": 15,
    }


def validate_t5_closure_artifact(directory: Path) -> Dict[str, Any]:
    """Bind the adjudicated attempt-3 closure artifact by its frozen seal hash."""

    root = directory.expanduser().resolve(strict=True)
    expected_files = {
        "STATUS.json",
        "t5_tokenization_equivalence.json",
        "per_case.jsonl.gz",
        "environment.json",
        "artifact_seal.json",
        "SHA256SUMS.txt",
        "SHA256SUMS.txt.sha256",
    }
    entries = list(root.iterdir())
    if {entry.name for entry in entries} != expected_files:
        raise ContractError("T5 closure artifact has a non-exact member set")
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ContractError("T5 closure artifact contains a non-regular member")
    status_path = root / "STATUS.json"
    seal_path = root / "artifact_seal.json"
    result_path = root / "t5_tokenization_equivalence.json"
    status = load_json(status_path, "T5 closure STATUS")
    if sha256_file(seal_path) != EXPECTED_T5_CLOSURE_SEAL_SHA256:
        raise ContractError("T5 closure artifact seal differs from attempt-3 milestone")
    seal = load_json(seal_path, "T5 closure artifact seal")
    if set(seal) != {"schema_version", "status", "scientific_config_sha256", "members"}:
        raise ContractError("T5 closure seal field set differs")
    if (
        seal.get("schema_version") != "ptc-opd-t5-tokenization-equivalence-seal-v3"
        or seal.get("status") != "equivalent_all_inputs"
    ):
        raise ContractError("T5 closure seal schema/status differs")
    members = seal.get("members")
    expected_sealed = {
        "STATUS.json",
        "t5_tokenization_equivalence.json",
        "per_case.jsonl.gz",
        "environment.json",
    }
    if not isinstance(members, dict) or set(members) != expected_sealed:
        raise ContractError("T5 closure sealed member set differs")
    for name in expected_sealed:
        identity = members.get(name)
        if (
            not isinstance(identity, dict)
            or set(identity) != {"sha256", "size_bytes"}
            or identity != regular_file_identity(root / name)
        ):
            raise ContractError("T5 closure seal identity differs for {}".format(name))

    checksum_names = sorted(expected_files - {"SHA256SUMS.txt", "SHA256SUMS.txt.sha256"})
    checksum_lines = (root / "SHA256SUMS.txt").read_text(
        encoding="utf-8", errors="strict"
    ).splitlines()
    if len(checksum_lines) != len(checksum_names):
        raise ContractError("T5 closure checksum inventory length differs")
    observed_checksum_names = []
    for line in checksum_lines:
        try:
            digest, name = line.split("  ", 1)
        except ValueError as exc:
            raise ContractError("malformed T5 closure checksum line") from exc
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ContractError("malformed T5 closure checksum digest")
        observed_checksum_names.append(name)
        if name not in expected_files or sha256_file(root / name) != digest:
            raise ContractError("T5 closure checksum mismatch for {}".format(name))
    if observed_checksum_names != checksum_names:
        raise ContractError("T5 closure checksum member order/set differs")
    expected_self_pin = "{}  SHA256SUMS.txt\n".format(
        sha256_file(root / "SHA256SUMS.txt")
    )
    if (root / "SHA256SUMS.txt.sha256").read_text(
        encoding="utf-8", errors="strict"
    ) != expected_self_pin:
        raise ContractError("T5 closure checksum self-pin differs")

    if (
        set(status)
        != {
            "schema_version",
            "status",
            "waiver_closed",
            "pilot_blocked",
            "required_action",
            "result_sha256",
        }
        or status.get("schema_version")
        != "ptc-opd-t5-tokenization-equivalence-status-v3"
        or status.get("status") != "equivalent_all_inputs"
        or status.get("waiver_closed") is not True
        or status.get("pilot_blocked") is not False
        or status.get("required_action") != "none"
        or status.get("result_sha256") != sha256_file(result_path)
    ):
        raise ContractError("T5 closure is not equivalent/closed/unblocked")
    result = load_json(result_path, "T5 closure result")
    config = result.get("scientific_config")
    decision = result.get("decision")
    if (
        result.get("schema_version") != "ptc-opd-t5-tokenization-equivalence-v3"
        or result.get("scientific_status") != "equivalent_all_inputs"
        or result.get("primary_contract_passed") is not True
        or not isinstance(config, dict)
        or result.get("scientific_config_sha256") != canonical_json_sha256(config)
        or seal.get("scientific_config_sha256") != result.get("scientific_config_sha256")
        or not isinstance(decision, dict)
        or decision.get("waiver_closed") is not True
        or decision.get("pilot_blocked") is not False
        or decision.get("required_action") != "none"
    ):
        raise ContractError("T5 closure result/config identity differs")
    return {
        "status_sha256": sha256_file(status_path),
        "artifact_seal_sha256": sha256_file(seal_path),
        "scientific_status": status.get("status"),
        "waiver_closed": True,
        "pilot_blocked": False,
    }


__all__ = [name for name in globals() if name.isupper()] + [
    "ContractError",
    "canonical_json_sha256",
    "load_json",
    "regular_file_identity",
    "require_finite_number",
    "require_pass_case",
    "sha256_file",
    "staged_output_directory",
    "stable_json_bytes",
    "validate_node3_evidence",
    "validate_t5_closure_artifact",
    "verify_artifact_directory",
    "write_artifact_seal",
    "write_json_exclusive",
]
