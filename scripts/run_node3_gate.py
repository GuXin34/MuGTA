#!/usr/bin/env python3
"""Run and seal the fail-closed node-3 single-machine/eight-GPU gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


WORKPACK_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "ptc-opd-node3-gate-v1"
B05_LIFECYCLE_SCHEMA_VERSION = "ptc-opd-node3-resume-lifecycle-v1"
UTC_PATTERN = re.compile(r"\d{8}T\d{6}Z\Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")

POLLUTION_VARIABLES = (
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "LOCAL_WORLD_SIZE",
    "ROLE_WORLD_SIZE",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "SLURM_JOB_ID",
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "SLURM_NTASKS",
    "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_SIZE",
    "PMI_RANK",
    "PMI_SIZE",
    "PTC_RUN_MULTIPROCESS_TESTS",
)

GATE_IDS = (
    "A01_workpack_seal",
    "A02_workpack_check",
    "A03_overlay_exact",
    "A04_overlay_test",
    "A05_full_tests",
    "A06_eight_process_ratio",
    "B01_uninterrupted_two_updates",
    "B02_controlled_step1_failure",
    "B03_attempt0_immutable",
    "B04_stale_rejected_latest_accepted",
    "B05_resume_equivalence",
    "B06_copied_run_verification",
    "B07_cpu_fp32_conditioners_nocfg",
    "B08_bf16_finite_denominator_teacher",
    "B09_all_rank_memory",
)

CHECKSUM_EXCLUSIONS = {
    "SHA256SUMS.txt",
    "SHA256SUMS.txt.sha256",
    "SHA256SUMS.verify.log",
}
STEP2_LOSS_RTOL = 1.0e-6
STEP2_LOSS_ATOL = 1.0e-8
STEP2_GRAD_NORM_RTOL = 1.0e-6
STEP2_GRAD_NORM_ATOL = 1.0e-8
FROZEN_NODE3_SEED = 2027
FROZEN_NODE3_LEARNING_RATE = 3.0e-6
RETAINED_SMALL_DECISION_SHA256 = (
    "731d0a00ac519079b40889e4fd79ec5ee450b1da8507e32ec815b042f0bb96d8"
)
RETAINED_SMALL_DECISION_SIDECAR_SHA256 = (
    "e60320bf2a052fd72b2caec9edbd788e4e39999122fc8f02f0ce766843d29899"
)
NODE3_DETERMINISM_CONTRACT = {
    "gate_mode": True,
    "deterministic_algorithms": True,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "cublas_workspace_config": ":4096:8",
}
NODE3_DDP_REDUCER_SCHEMA_VERSION = "ptc-opd-ddp-reducer-contract-v1"
NODE3_DDP_REDUCER_POLICY_ID = "torch-2.1-fixed-initial-buckets-v1"
NODE3_DDP_REDUCER_STATIC_CONTRACT = {
    "schema_version": NODE3_DDP_REDUCER_SCHEMA_VERSION,
    "policy_id": NODE3_DDP_REDUCER_POLICY_ID,
    "torch_version": "2.1.0+cu121",
    "torch_cuda_runtime": "12.1",
    "find_unused_parameters": True,
    "static_graph": False,
    "gradient_as_bucket_view": False,
    "bucket_cap_bytes": 25 * 1024 * 1024,
    "has_rebuilt_buckets": False,
}
DISTRIBUTED_TARGET_TESTS = (
    "test_ddp_scaling_algebra_matches_concatenated_reference "
    "(test_distributed.DistributedRatioTest)",
    "test_single_process_matches_local_loss "
    "(test_distributed.DistributedRatioTest)",
    "test_two_rank_unequal_counts_matches_concatenated_reference "
    "(test_distributed.DistributedRatioTest)",
)


class Node3GateError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise Node3GateError("cannot hash non-regular file: {}".format(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_text_exclusive(path: Path, value: str) -> None:
    payload = value.encode("utf-8")
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_json(path: Path) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Node3GateError("required JSON is not a regular file: {}".format(path))
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise Node3GateError("JSON must contain an object: {}".format(path))
    return value


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise Node3GateError("required JSONL is not a regular file: {}".format(path))
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise Node3GateError("blank JSONL line {} in {}".format(line_number, path))
            value = json.loads(line)
            if not isinstance(value, dict):
                raise Node3GateError("JSONL record must be an object")
            records.append(value)
    return records


def validate_prospective_output_root(path: Path, root: Path, label: str) -> Path:
    """Validate an output root without creating any filesystem entry."""

    if root.is_symlink():
        raise Node3GateError("workpack root may not be a symlink")
    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise Node3GateError("workpack root must be a directory")
    lexical = Path(os.path.abspath(str(path.expanduser())))
    try:
        relative = lexical.relative_to(resolved_root)
    except ValueError as exc:
        raise Node3GateError("{} must stay inside {}".format(label, resolved_root)) from exc
    if not relative.parts:
        raise Node3GateError("{} may not equal the workpack root".format(label))
    current = lexical
    while current != resolved_root:
        if current.is_symlink():
            raise Node3GateError("{} has a symlink path component: {}".format(label, current))
        if current.exists() and not current.is_dir():
            raise Node3GateError("{} has a non-directory path component: {}".format(label, current))
        current = current.parent
    resolved = lexical.resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise Node3GateError(
            "{} resolves outside {}".format(label, resolved_root)
        ) from exc
    return lexical


def prepare_output_roots(
    console_root: Path, runs_root: Path, root: Path = WORKPACK_ROOT
) -> Tuple[Path, Path]:
    """Validate both roots before creating either, then revalidate them."""

    console = validate_prospective_output_root(console_root, root, "console root")
    runs = validate_prospective_output_root(runs_root, root, "runs root")
    # No mkdir is permitted until both prospective paths have passed.
    console.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    console = validate_prospective_output_root(console, root, "console root")
    runs = validate_prospective_output_root(runs, root, "runs root")
    return console, runs


def validate_node3_protocol_hyperparameters(seed: int, learning_rate: float) -> None:
    """Reject a syntactically valid launch that is not the frozen smoke."""

    if type(seed) is not int or seed != FROZEN_NODE3_SEED:
        raise Node3GateError(
            "node-3 seed must be exactly {}".format(FROZEN_NODE3_SEED)
        )
    if (
        type(learning_rate) is not float
        or not math.isfinite(learning_rate)
        or learning_rate != FROZEN_NODE3_LEARNING_RATE
    ):
        raise Node3GateError(
            "node-3 learning rate must be exactly {:.1e}".format(
                FROZEN_NODE3_LEARNING_RATE
            )
        )


def _verify_retained_small_decision(directory: Path) -> Dict[str, Any]:
    expected = {"cfg_scale_decision.json", "cfg_scale_decision.sha256.json"}
    observed = {item.name for item in directory.iterdir()}
    if observed != expected:
        raise Node3GateError(
            "retained small decision must be an exact two-member directory"
        )
    decision = directory / "cfg_scale_decision.json"
    sidecar = directory / "cfg_scale_decision.sha256.json"
    if any(path.is_symlink() or not path.is_file() for path in (decision, sidecar)):
        raise Node3GateError("retained small decision members must be regular files")
    actual_decision = sha256_file(decision)
    actual_sidecar = sha256_file(sidecar)
    if actual_decision != RETAINED_SMALL_DECISION_SHA256:
        raise Node3GateError(
            "retained small cfg_scale_decision.json differs from the Gate-1 pin"
        )
    if actual_sidecar != RETAINED_SMALL_DECISION_SIDECAR_SHA256:
        raise Node3GateError(
            "retained small cfg_scale_decision.sha256.json sidecar differs from the Gate-1 pin"
        )
    payload = load_json(decision)
    if payload.get("status") != "selected" or payload.get("selected_cfg_scale") != 5.0:
        raise Node3GateError("retained small decision is not selected scale 5")
    generation = payload.get("generation_identity")
    if not isinstance(generation, dict) or generation.get("model_id") != "facebook/musicgen-small":
        raise Node3GateError("retained CFG decision is not MusicGen-small")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "directory": str(directory.resolve()),
        "decision_sha256": actual_decision,
        "sidecar_sha256": actual_sidecar,
        "selected_cfg_scale": 5.0,
        "model_id": "facebook/musicgen-small",
    }


def clean_child_environment() -> Tuple[Dict[str, str], List[str]]:
    environment = dict(os.environ)
    removed = sorted(name for name in POLLUTION_VARIABLES if name in environment)
    for name in POLLUTION_VARIABLES:
        environment.pop(name, None)
    environment.update(
        {
            "PTC_NODE3_GATE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(WORKPACK_ROOT / "src"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
    return environment, removed


class GateRecorder:
    def __init__(self, evidence_root: Path, environment: Mapping[str, str]) -> None:
        self.evidence_root = evidence_root
        self.environment = dict(environment)
        self.results: Dict[str, Dict[str, Any]] = {
            gate_id: {"status": "not_run", "evidence": []} for gate_id in GATE_IDS
        }
        self.command_index = 0

    def pass_gate(self, gate_id: str, evidence: Iterable[str]) -> None:
        if gate_id not in self.results:
            raise Node3GateError("unknown gate id {}".format(gate_id))
        self.results[gate_id] = {"status": "passed", "evidence": list(evidence)}

    def fail_gate(self, gate_id: str, message: str) -> None:
        if gate_id in self.results and self.results[gate_id]["status"] == "not_run":
            self.results[gate_id] = {
                "status": "failed",
                "evidence": [],
                "error": message,
            }

    def run(
        self,
        gate_id: str,
        label: str,
        command: Sequence[str],
        *,
        cwd: Path,
        expected_success: bool = True,
        required_marker: Optional[str] = None,
        environment_overrides: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, Any]:
        self.command_index += 1
        prefix = "{:02d}_{}".format(self.command_index, label)
        stdout_path = self.evidence_root / (prefix + ".stdout.log")
        stderr_path = self.evidence_root / (prefix + ".stderr.log")
        metadata_path = self.evidence_root / (prefix + ".command.json")
        started = time.time()
        child_environment = dict(self.environment)
        if environment_overrides is not None:
            child_environment.update(
                {str(key): str(value) for key, value in environment_overrides.items()}
            )
        with stdout_path.open("xb") as stdout_stream, stderr_path.open("xb") as stderr_stream:
            completed = subprocess.run(
                [str(item) for item in command],
                cwd=str(cwd),
                env=child_environment,
                stdout=stdout_stream,
                stderr=stderr_stream,
                check=False,
            )
            stdout_stream.flush()
            stderr_stream.flush()
            os.fsync(stdout_stream.fileno())
            os.fsync(stderr_stream.fileno())
        stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
        combined = stdout_text + "\n" + stderr_text
        rc_ok = completed.returncode == 0 if expected_success else completed.returncode != 0
        marker_ok = required_marker is None or required_marker in combined
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "gate_id": gate_id,
            "label": label,
            "argv": [str(item) for item in command],
            "cwd": str(cwd.resolve()),
            "started_unix_seconds": started,
            "finished_unix_seconds": time.time(),
            "return_code": completed.returncode,
            "expected_success": expected_success,
            "required_marker": required_marker,
            "marker_found": marker_ok,
            "environment_overrides": dict(environment_overrides or {}),
            "stdout": stdout_path.name,
            "stderr": stderr_path.name,
            "stdout_sha256": sha256_file(stdout_path),
            "stderr_sha256": sha256_file(stderr_path),
            "status": "passed" if rc_ok and marker_ok else "failed",
        }
        write_json_exclusive(metadata_path, metadata)
        if not rc_ok or not marker_ok:
            self.fail_gate(
                gate_id,
                "command {} returned rc={} (expected_success={}, marker_ok={})".format(
                    label, completed.returncode, expected_success, marker_ok
                ),
            )
            raise Node3GateError(self.results[gate_id]["error"])
        return metadata


def _training_command(
    args: argparse.Namespace, output_dir: Path, *, resume: Optional[Path] = None
) -> List[str]:
    command = [
        str(args.torchrun),
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=8",
        str(WORKPACK_ROOT / "scripts" / "train_stage1.py"),
        "--manifest",
        str(args.manifest),
        "--student-checkpoint",
        str(args.student_checkpoint),
        "--teacher-checkpoint",
        str(args.teacher_checkpoint),
        "--audiocraft-root",
        str(args.audiocraft_root),
        "--cfg-scale-decision-dir",
        str(args.cfg_scale_decision_dir),
        "--output-dir",
        str(output_dir),
        "--mode",
        "uniform100",
        "--seed",
        str(args.seed),
        "--learning-rate",
        str(args.learning_rate),
        "--max-optimizer-steps",
        "2",
        "--save-every",
        "1",
        "--log-every",
        "1",
    ]
    if resume is not None:
        command.extend(["--resume", str(resume)])
    return command


def _dry_run_command(args: argparse.Namespace, output_dir: Path) -> List[str]:
    return [
        str(args.python),
        str(WORKPACK_ROOT / "scripts" / "train_stage1.py"),
        "--dry-run",
        "--manifest",
        str(args.manifest),
        "--student-checkpoint",
        str(args.student_checkpoint),
        "--teacher-checkpoint",
        str(args.teacher_checkpoint),
        "--audiocraft-root",
        str(args.audiocraft_root),
        "--cfg-scale-decision-dir",
        str(args.cfg_scale_decision_dir),
        "--output-dir",
        str(output_dir),
        "--mode",
        "uniform100",
        "--seed",
        str(args.seed),
        "--learning-rate",
        str(args.learning_rate),
        "--max-optimizer-steps",
        "2",
        "--save-every",
        "1",
        "--log-every",
        "1",
    ]


def _distributed_target_test_command(
    python: Path, tests_directory: Path
) -> List[str]:
    """Build the A06 target-test command without importing a top-level tests package."""

    return [
        str(python),
        "-m",
        "unittest",
        "discover",
        "-s",
        str(tests_directory),
        "-p",
        "test_distributed.py",
        "-v",
    ]


def validate_distributed_target_test_output(
    stdout_text: str, stderr_text: str
) -> Dict[str, Any]:
    """Require all three A06 target tests to execute and pass without skips."""

    combined = stdout_text + "\n" + stderr_text
    for test_name in DISTRIBUTED_TARGET_TESTS:
        pattern = re.compile(
            r"^{} \.\.\. ok\r?$".format(re.escape(test_name)), re.MULTILINE
        )
        if pattern.search(combined) is None:
            raise Node3GateError(
                "A06 target test did not execute and pass: {}".format(test_name)
            )
    run_counts = re.findall(r"^Ran ([0-9]+) tests? in [^\r\n]+\r?$", combined, re.MULTILINE)
    if run_counts != [str(len(DISTRIBUTED_TARGET_TESTS))]:
        raise Node3GateError(
            "A06 target-test summary must report exactly {} tests; got {}".format(
                len(DISTRIBUTED_TARGET_TESTS), run_counts
            )
        )
    if len(re.findall(r"^OK\r?$", combined, re.MULTILINE)) != 1:
        raise Node3GateError("A06 target-test summary must be exactly OK with no skips")
    if re.search(r"\bskip(?:ped|s)?\b", combined, re.IGNORECASE) is not None:
        raise Node3GateError("A06 target test output contains a skip")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "expected_tests": list(DISTRIBUTED_TARGET_TESTS),
        "ran_test_count": len(DISTRIBUTED_TARGET_TESTS),
        "skip_count": 0,
    }


def _attempt0_identities(run_dir: Path) -> Dict[str, Any]:
    paths = (
        run_dir / "logs" / "attempt-0000.json",
        run_dir / "logs" / "metrics.attempt-0000.jsonl",
    )
    return {
        path.relative_to(run_dir).as_posix(): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    }


def _preresume_seed_identities(run_dir: Path) -> Dict[str, Any]:
    """Fingerprint every immutable member needed to repeat resume from step 1."""

    relative_paths = (
        Path("run_manifest.json"),
        Path("status.json"),
        Path("FAILED.json"),
        Path("logs/attempt-0000.json"),
        Path("logs/metrics.attempt-0000.jsonl"),
        Path("checkpoints/step-00000/checkpoint.pt"),
        Path("checkpoints/step-00000/SHA256.json"),
        Path("checkpoints/step-00001/checkpoint.pt"),
        Path("checkpoints/step-00001/SHA256.json"),
    )
    identities: Dict[str, Any] = {}
    for relative in relative_paths:
        path = run_dir / relative
        if path.is_symlink() or not path.is_file():
            raise Node3GateError(
                "pre-resume seed member is missing/non-regular: {}".format(relative)
            )
        identities[relative.as_posix()] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return identities


def _validate_preresume_seed_closed_world(run_dir: Path) -> Dict[str, Any]:
    """Reject any extra/missing member in a controlled-failure step-1 seed."""

    expected: Dict[Path, set] = {
        Path("."): {
            "run_manifest.json",
            "status.json",
            "FAILED.json",
            "logs",
            "checkpoints",
        },
        Path("logs"): {
            "attempt-0000.json",
            "metrics.attempt-0000.jsonl",
        },
        Path("checkpoints"): {"step-00000", "step-00001"},
        Path("checkpoints/step-00000"): {"checkpoint.pt", "SHA256.json"},
        Path("checkpoints/step-00001"): {"checkpoint.pt", "SHA256.json"},
    }
    root = run_dir
    if root.is_symlink() or not root.is_dir():
        raise Node3GateError("pre-resume seed root is not a regular directory")
    layouts: Dict[str, List[str]] = {}
    for relative, expected_names in expected.items():
        directory = root if relative == Path(".") else root / relative
        if directory.is_symlink() or not directory.is_dir():
            raise Node3GateError(
                "pre-resume seed directory is missing/non-regular: {}".format(
                    relative.as_posix()
                )
            )
        entries = list(directory.iterdir())
        if any(entry.is_symlink() for entry in entries):
            raise Node3GateError(
                "pre-resume seed contains a symlink under {}".format(
                    relative.as_posix()
                )
            )
        observed_names = {entry.name for entry in entries}
        if observed_names != expected_names:
            raise Node3GateError(
                "pre-resume seed member set differs under {}: expected={} observed={}".format(
                    relative.as_posix(),
                    sorted(expected_names),
                    sorted(observed_names),
                )
            )
        layouts[relative.as_posix()] = sorted(observed_names)
    for relative in (
        Path("run_manifest.json"),
        Path("status.json"),
        Path("FAILED.json"),
        Path("logs/attempt-0000.json"),
        Path("logs/metrics.attempt-0000.jsonl"),
        Path("checkpoints/step-00000/checkpoint.pt"),
        Path("checkpoints/step-00000/SHA256.json"),
        Path("checkpoints/step-00001/checkpoint.pt"),
        Path("checkpoints/step-00001/SHA256.json"),
    ):
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise Node3GateError(
                "pre-resume seed member is not a regular file: {}".format(
                    relative.as_posix()
                )
            )
    return {
        "schema_version": B05_LIFECYCLE_SCHEMA_VERSION,
        "status": "closed_world",
        "layouts": layouts,
    }


def rotate_completed_resume_with_seed(
    *, completed: Path, seed: Path, retained: Path, runs_root: Path
) -> Dict[str, Any]:
    """Atomically retain cold-resume A and restore its untouched step-1 seed.

    The three paths must be distinct direct children of the already validated
    runs root.  Nothing is deleted or overwritten: two same-filesystem renames
    turn the completed first resume into a retained run and put the byte-copy
    of its pre-resume tree back at the original path for cold-resume B.
    """

    root = runs_root.resolve(strict=True)
    resolved_parents = {
        path.parent.resolve(strict=True) for path in (completed, seed, retained)
    }
    if resolved_parents != {root} or len({completed, seed, retained}) != 3:
        raise Node3GateError("cold-resume rotation paths must be distinct run-root children")
    if completed.is_symlink() or not completed.is_dir():
        raise Node3GateError("completed cold-resume A is not a regular directory")
    if seed.is_symlink() or not seed.is_dir():
        raise Node3GateError("cold-resume seed is not a regular directory")
    if retained.exists() or retained.is_symlink():
        raise Node3GateError("cold-resume A retention path already exists")
    seed_layout = _validate_preresume_seed_closed_world(seed)
    completed_step1 = _preresume_seed_identities(completed)
    seed_step1 = _preresume_seed_identities(seed)
    if completed_step1 != seed_step1:
        raise Node3GateError("cold-resume pre-resume seed bytes differ before rotation")
    completed.rename(retained)
    try:
        seed.rename(completed)
    except BaseException:
        # Recover the original name if the second rename fails.  The retained
        # target was proven absent, so this rollback cannot overwrite a tree.
        retained.rename(completed)
        raise
    return {
        "schema_version": B05_LIFECYCLE_SCHEMA_VERSION,
        "status": "rotated",
        "operation": "two_same_filesystem_renames_no_delete_no_overwrite",
        "retained_cold_a": str(retained),
        "restored_seed_as_cold_b": str(completed),
        "restored_seed_closed_world": seed_layout,
        "preresume_seed_identity": completed_step1,
    }


def _record_at_step(path: Path, step: int) -> Dict[str, Any]:
    matches = [
        record for record in load_jsonl(path) if record.get("optimizer_step") == step
    ]
    if len(matches) != 1:
        raise Node3GateError(
            "expected exactly one optimizer_step={} record in {}".format(step, path)
        )
    return matches[0]


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_ddp_reducer_contract(value: Any) -> Dict[str, Any]:
    expected_fields = set(NODE3_DDP_REDUCER_STATIC_CONTRACT) | {
        "bucket_sizes",
        "bucket_count",
        "trainable_parameter_tensor_count",
        "trainable_parameter_numel",
        "trainable_gradient_bytes",
        "parameter_layout_sha256",
        "identity_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise Node3GateError("DDP reducer contract field set differs")
    for key, expected in NODE3_DDP_REDUCER_STATIC_CONTRACT.items():
        observed = value.get(key)
        if type(observed) is not type(expected) or observed != expected:
            raise Node3GateError("DDP reducer contract differs at {}".format(key))
    bucket_sizes = value.get("bucket_sizes")
    if not isinstance(bucket_sizes, str) or re.fullmatch(
        r"[1-9][0-9]*(?:,\s*[1-9][0-9]*)*", bucket_sizes
    ) is None:
        raise Node3GateError("DDP bucket-size evidence is malformed")
    parsed = [int(item) for item in re.split(r",\s*", bucket_sizes)]
    if type(value.get("bucket_count")) is not int or value.get(
        "bucket_count"
    ) != len(parsed):
        raise Node3GateError("DDP bucket count differs from bucket-size evidence")
    for key in (
        "trainable_parameter_tensor_count",
        "trainable_parameter_numel",
        "trainable_gradient_bytes",
    ):
        if type(value.get(key)) is not int or value[key] <= 0:
            raise Node3GateError("DDP reducer {} is not a positive integer".format(key))
    if sum(parsed) != value["trainable_gradient_bytes"]:
        raise Node3GateError("DDP buckets do not cover the trainable gradient bytes")
    parameter_layout_sha256 = value.get("parameter_layout_sha256")
    identity_sha256 = value.get("identity_sha256")
    if (
        not isinstance(parameter_layout_sha256, str)
        or SHA256_PATTERN.fullmatch(parameter_layout_sha256) is None
        or not isinstance(identity_sha256, str)
        or SHA256_PATTERN.fullmatch(identity_sha256) is None
    ):
        raise Node3GateError("DDP reducer identity digest is malformed")
    identity_payload = dict(value)
    del identity_payload["identity_sha256"]
    if _canonical_json_sha256(identity_payload) != identity_sha256:
        raise Node3GateError("DDP reducer identity digest differs")
    return dict(value)


def _canonical_resume_audit(record: Mapping[str, Any]) -> List[Dict[str, Any]]:
    optimizer_step = record.get("optimizer_step")
    global_microstep = record.get("global_microstep")
    if type(optimizer_step) is not int or optimizer_step <= 0:
        raise Node3GateError("step record optimizer_step must be a positive integer")
    if global_microstep != 4 * optimizer_step:
        raise Node3GateError("step record progress must equal optimizer_step*4")
    audits = record.get("all_rank_audit")
    if not isinstance(audits, list) or len(audits) != 8:
        raise Node3GateError("step record must contain all eight rank audits")
    canonical: List[Dict[str, Any]] = []
    for expected_rank, rank_audit in enumerate(audits):
        if not isinstance(rank_audit, dict) or rank_audit.get("rank") != expected_rank:
            raise Node3GateError("rank audit is incomplete or out of order")
        microsteps = rank_audit.get("microsteps")
        if not isinstance(microsteps, list) or len(microsteps) != 4:
            raise Node3GateError("rank audit must contain four microsteps")
        canonical_microsteps: List[Dict[str, Any]] = []
        for expected_index, item in enumerate(microsteps):
            if not isinstance(item, dict):
                raise Node3GateError("rank microstep audit must be an object")
            if item.get("accumulation_index") != expected_index:
                raise Node3GateError("accumulation indices must be exactly 0..3")
            expected_global_microstep = 4 * (optimizer_step - 1) + expected_index
            if item.get("global_microstep") != expected_global_microstep:
                raise Node3GateError(
                    "rank microstep progress does not match optimizer step"
                )
            canonical_microsteps.append(
                {
                    "accumulation_index": item.get("accumulation_index"),
                    "global_microstep": item.get("global_microstep"),
                    "sample_ids": item.get("sample_ids"),
                    "selected_gate_sha256": item.get("selected_gate_sha256"),
                    "selected_cells": item.get("selected_cells"),
                    "valid_cells": item.get("valid_cells"),
                    "rollout_shape": item.get("rollout_shape"),
                    "rollout_dtype": item.get("rollout_dtype"),
                    "rollout_codes_sha256": item.get("rollout_codes_sha256"),
                    "use_cfg": item.get("use_cfg"),
                    "condition_tensors_source": item.get(
                        "condition_tensors_source"
                    ),
                }
            )
        canonical.append(
            {
                "rank": expected_rank,
                "microsteps": canonical_microsteps,
                "ddp_reducer": _validate_ddp_reducer_contract(
                    rank_audit.get("ddp_reducer")
                ),
            }
        )
    return canonical


def _validate_resume_rng_restore(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Require an immediate, pre-reseed RNG restoration proof from every rank."""

    audits = record.get("all_rank_audit")
    if not isinstance(audits, list) or len(audits) != 8:
        raise Node3GateError("resume RNG audit lacks all eight ranks")
    identities: List[Dict[str, Any]] = []
    expected_fields = {
        "schema_version",
        "rank",
        "captured_before_next_rollout_seed",
        "expected_sha256",
        "observed_sha256",
        "exact",
    }
    for expected_rank, rank_audit in enumerate(audits):
        if not isinstance(rank_audit, dict):
            raise Node3GateError("resume rank audit is malformed")
        identity = rank_audit.get("resume_rng_restore")
        if not isinstance(identity, dict) or set(identity) != expected_fields:
            raise Node3GateError("resume RNG restore identity is malformed")
        if identity.get("schema_version") != "ptc-opd-node3-rng-restore-v1":
            raise Node3GateError("resume RNG restore schema differs")
        if identity.get("rank") != expected_rank:
            raise Node3GateError("resume RNG restore ranks do not cover 0..7")
        if identity.get("captured_before_next_rollout_seed") is not True:
            raise Node3GateError("resume RNG restore was not captured pre-reseed")
        expected_digest = identity.get("expected_sha256")
        observed_digest = identity.get("observed_sha256")
        if (
            not isinstance(expected_digest, str)
            or SHA256_PATTERN.fullmatch(expected_digest) is None
            or observed_digest != expected_digest
            or identity.get("exact") is not True
        ):
            raise Node3GateError("resume RNG state was not restored exactly")
        identities.append(dict(identity))
    return {"rank_count": 8, "identities": identities}


def _validate_resume_state_restore(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Require exact checkpoint-to-live student/optimizer proof on every rank."""

    audits = record.get("all_rank_audit")
    if not isinstance(audits, list) or len(audits) != 8:
        raise Node3GateError("resume state audit lacks all eight ranks")
    expected_fields = {
        "schema_version",
        "rank",
        "captured_after_checkpoint_load",
        "captured_before_next_forward",
        "canonical_identity",
        "checkpoint",
        "live",
        "exact_fields",
        "exact",
    }
    expected_hash_fields = {
        "student_state_sha256",
        "optimizer_state_sha256",
    }
    expected_identity = (
        "finite tensor-tree SHA-256 exact over dtype/shape/layout/stride/"
        "storage_offset/logical-bytes and typed containers"
    )
    identities: List[Dict[str, Any]] = []
    for expected_rank, rank_audit in enumerate(audits):
        if not isinstance(rank_audit, dict):
            raise Node3GateError("resume state rank audit is malformed")
        identity = rank_audit.get("resume_state_restore")
        if not isinstance(identity, dict) or set(identity) != expected_fields:
            raise Node3GateError("resume state restore identity is malformed")
        if identity.get("schema_version") != "ptc-opd-node3-resume-state-restore-v1":
            raise Node3GateError("resume state restore schema differs")
        if identity.get("rank") != expected_rank:
            raise Node3GateError("resume state restore ranks do not cover 0..7")
        if identity.get("captured_after_checkpoint_load") is not True:
            raise Node3GateError("resume state was not captured after checkpoint load")
        if identity.get("captured_before_next_forward") is not True:
            raise Node3GateError("resume state was not captured before next forward")
        if identity.get("canonical_identity") != expected_identity:
            raise Node3GateError("resume state canonical identity differs")
        checkpoint = identity.get("checkpoint")
        live = identity.get("live")
        exact_fields = identity.get("exact_fields")
        if not all(isinstance(value, dict) for value in (checkpoint, live, exact_fields)):
            raise Node3GateError("resume state restore hashes are malformed")
        if any(set(value) != expected_hash_fields for value in (checkpoint, live, exact_fields)):
            raise Node3GateError("resume state restore hash fields differ")
        if any(
            not isinstance(checkpoint[field], str)
            or SHA256_PATTERN.fullmatch(checkpoint[field]) is None
            or live[field] != checkpoint[field]
            or exact_fields[field] is not True
            for field in expected_hash_fields
        ) or identity.get("exact") is not True:
            raise Node3GateError("resume student/optimizer state was not restored exactly")
        identities.append(dict(identity))
    return {"rank_count": 8, "identities": identities}


def _compare_step2_metrics(
    uninterrupted: Mapping[str, Any], resumed: Mapping[str, Any]
) -> Dict[str, Any]:
    for field, expected in (
        ("optimizer_step", 2),
        ("global_microstep", 8),
        ("mode", "uniform100"),
    ):
        if uninterrupted.get(field) != expected or resumed.get(field) != expected:
            raise Node3GateError("step-2 metric contract differs at {}".format(field))
    comparisons: Dict[str, Any] = {}
    for field, rtol, atol in (
        ("loss", STEP2_LOSS_RTOL, STEP2_LOSS_ATOL),
        ("gradient_norm", STEP2_GRAD_NORM_RTOL, STEP2_GRAD_NORM_ATOL),
    ):
        left = uninterrupted.get(field)
        right = resumed.get(field)
        if not all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in (left, right)
        ):
            raise Node3GateError("step-2 {} is missing/non-finite".format(field))
        equal = math.isclose(float(left), float(right), rel_tol=rtol, abs_tol=atol)
        comparisons[field] = {
            "uninterrupted": float(left),
            "resumed": float(right),
            "absolute_difference": abs(float(left) - float(right)),
            "rtol": rtol,
            "atol": atol,
            "status": "within_tolerance" if equal else "failed",
        }
        if not equal:
            raise Node3GateError(
                "step-2 {} differs beyond rtol={} atol={}".format(field, rtol, atol)
            )
    exact_fields = (
        "learning_rate",
        "global_denominators",
        "denominator_window_constant",
        "selected_cells_rank0",
        "valid_cells_rank0",
    )
    exact = {
        field: uninterrupted.get(field) == resumed.get(field)
        for field in exact_fields
    }
    if not all(exact.values()):
        raise Node3GateError(
            "step-2 exact metric fields differ: {}".format(
                sorted(field for field, equal in exact.items() if not equal)
            )
        )
    return {
        "status": "passed",
        "tolerance_comparisons": comparisons,
        "exact_fields": exact,
    }


def _compare_cold_step2_metrics_exact(
    cold_a: Mapping[str, Any], cold_b: Mapping[str, Any]
) -> Dict[str, Any]:
    """Apply the shared metric contract, then require cold/cold scalar identity."""

    base = _compare_step2_metrics(cold_a, cold_b)
    scalar_exact = {
        field: cold_a.get(field) == cold_b.get(field)
        for field in ("loss", "gradient_norm")
    }
    if not all(scalar_exact.values()):
        raise Node3GateError(
            "cold-resume step-2 scalar metrics differ: {}".format(
                sorted(field for field, exact in scalar_exact.items() if not exact)
            )
        )
    return {
        "status": "passed",
        "policy": "exact cold-resume step-2 scalar and exact-field identity",
        "scalar_exact": scalar_exact,
        "exact_fields": base["exact_fields"],
    }


def _validate_runtime_contract(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    contracts = manifest.get("runtime_contract_by_rank")
    if not isinstance(contracts, list) or len(contracts) != 8:
        raise Node3GateError("run manifest lacks eight runtime contracts")
    ddp_identities: List[str] = []
    for rank, contract in enumerate(contracts):
        if not isinstance(contract, dict) or contract.get("rank") != rank:
            raise Node3GateError("runtime contracts do not cover ranks 0..7")
        cpu = contract.get("cpu_load")
        if not isinstance(cpu, dict) or not all(
            cpu.get(key) is True
            for key in (
                "student_all_fp32",
                "teacher_all_fp32",
                "student_all_cpu",
                "teacher_all_cpu",
            )
        ):
            raise Node3GateError("student/teacher CPU FP32 load contract failed")
        for key in ("student_conditioner", "teacher_conditioner"):
            conditioner = contract.get(key)
            if not isinstance(conditioner, dict) or conditioner.get("status") != "frozen_eval":
                raise Node3GateError("conditioner freeze/eval contract failed")
            if conditioner.get("provider_training") is not False:
                raise Node3GateError("conditioner is in training mode")
            if conditioner.get("provider_trainable_parameter_count") != 0:
                raise Node3GateError("conditioner has trainable parameters")
            if any(conditioner.get("external_t5_training", [])):
                raise Node3GateError("external T5 is in training mode")
            if any(conditioner.get("external_t5_trainable_parameter_counts", [])):
                raise Node3GateError("external T5 has trainable parameters")
        if contract.get("generate_accepts_use_cfg") is not True:
            raise Node3GateError("AudioCraft generate lacks use_cfg")
        if contract.get("generate_accepts_condition_tensors") is not True:
            raise Node3GateError("AudioCraft generate lacks condition_tensors")
        if contract.get("bf16_supported") is not True:
            raise Node3GateError("GPU does not support BF16")
        if contract.get("bf16_autocast_probe_dtype") != "torch.bfloat16":
            raise Node3GateError("BF16 autocast probe dtype differs")
        if contract.get("bf16_autocast_probe_finite") is not True:
            raise Node3GateError("BF16 autocast probe is non-finite")
        if contract.get("gpu_name") != "NVIDIA H20":
            raise Node3GateError("node-3 gate must run on NVIDIA H20")
        if contract.get("gpu_capability") != [9, 0]:
            raise Node3GateError("node-3 H20 compute capability must be [9,0]")
        if contract.get("cuda_device_count") != 8:
            raise Node3GateError("node-3 must expose exactly eight CUDA devices")
        if contract.get("node3_determinism") != NODE3_DETERMINISM_CONTRACT:
            raise Node3GateError("node-3 deterministic runtime contract differs")
        ddp_contract = _validate_ddp_reducer_contract(
            contract.get("ddp_reducer")
        )
        ddp_identities.append(str(ddp_contract["identity_sha256"]))
    if len(set(ddp_identities)) != 1:
        raise Node3GateError("DDP reducer contract differs across ranks")
    return {
        "rank_count": len(contracts),
        "ddp_reducer_identity_sha256": ddp_identities[0],
        "contracts": contracts,
    }


def _validate_step_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise Node3GateError("training metrics are empty")
    memory: List[Dict[str, int]] = []
    ddp_reducer_identities = set()
    for record in records:
        for key in ("loss", "gradient_norm"):
            value = record.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise Node3GateError("{} is missing or non-finite".format(key))
        denominators = record.get("global_denominators")
        if not isinstance(denominators, list) or len(denominators) != 4:
            raise Node3GateError("one optimizer step must record four denominators")
        if not all(
            isinstance(item, (int, float))
            and math.isfinite(float(item))
            and float(item) > 0.0
            for item in denominators
        ):
            raise Node3GateError("denominators must be finite and positive")
        reference = float(denominators[0])
        if any(abs(float(item) - reference) > 1.0e-6 * abs(reference) for item in denominators[1:]):
            raise Node3GateError("four accumulation denominators differ beyond 1e-6")
        audits = _canonical_resume_audit(record)
        record_ddp_identities = {
            item["ddp_reducer"]["identity_sha256"] for item in audits
        }
        if len(record_ddp_identities) != 1:
            raise Node3GateError("DDP reducer contract differs across step ranks")
        ddp_reducer_identities.update(record_ddp_identities)
        source_audits = record["all_rank_audit"]
        for canonical, source in zip(audits, source_audits):
            if source.get("gradient_finite") is not True:
                raise Node3GateError("rank gradient audit is not finite")
            if source.get("all_trainable_gradients_present") is not True:
                raise Node3GateError("rank trainable-gradient coverage is incomplete")
            if source.get("student_conditioner_status") != "frozen_eval":
                raise Node3GateError("student conditioner drifted")
            if source.get("teacher_conditioner_status") != "frozen_eval":
                raise Node3GateError("teacher conditioner drifted")
            if source.get("node3_determinism") != NODE3_DETERMINISM_CONTRACT:
                raise Node3GateError("node-3 deterministic flags drifted")
            allocated = source.get("cuda_max_memory_allocated")
            reserved = source.get("cuda_max_memory_reserved")
            total = source.get("cuda_total_memory_bytes")
            if not all(type(item) is int and item > 0 for item in (allocated, reserved, total)):
                raise Node3GateError("all-rank CUDA memory evidence is incomplete")
            if not allocated <= reserved <= total:
                raise Node3GateError("CUDA allocated/reserved/total ordering is invalid")
            memory.append(
                {
                    "optimizer_step": int(record["optimizer_step"]),
                    "rank": int(canonical["rank"]),
                    "allocated": allocated,
                    "reserved": reserved,
                    "total": total,
                }
            )
            for microstep, source_microstep in zip(
                canonical["microsteps"], source["microsteps"]
            ):
                if microstep["rollout_shape"] != [2, 4, 500]:
                    raise Node3GateError("rollout shape is not [2,4,500]")
                if microstep["rollout_dtype"] != "torch.int64":
                    raise Node3GateError("rollout dtype is not torch.int64")
                rollout_digest = microstep["rollout_codes_sha256"]
                if (
                    not isinstance(rollout_digest, str)
                    or SHA256_PATTERN.fullmatch(rollout_digest) is None
                ):
                    raise Node3GateError("rollout-code digest is malformed")
                if microstep["use_cfg"] is not False:
                    raise Node3GateError("rollout used CFG")
                if microstep["condition_tensors_source"] != "conditional":
                    raise Node3GateError("rollout did not use conditional tensors")
                digest = microstep["selected_gate_sha256"]
                if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
                    raise Node3GateError("selected gate digest is malformed")
                sample_ids = microstep["sample_ids"]
                if (
                    not isinstance(sample_ids, list)
                    or len(sample_ids) != 2
                    or any(not isinstance(item, str) or not item for item in sample_ids)
                ):
                    raise Node3GateError("rank microstep must bind two sample IDs")
                if microstep["selected_cells"] != microstep["valid_cells"]:
                    raise Node3GateError("uniform100 selected gate is not all-valid")
                if source_microstep.get("global_loss_finite") is not True:
                    raise Node3GateError("microstep global loss is non-finite")
                index = int(microstep["accumulation_index"])
                if not math.isclose(
                    float(source_microstep.get("global_denominator")),
                    float(denominators[index]),
                    rel_tol=0.0,
                    abs_tol=0.0,
                ):
                    raise Node3GateError("rank microstep denominator audit differs")
    if len(ddp_reducer_identities) != 1:
        raise Node3GateError("DDP reducer contract differs across attempts or steps")
    return {
        "record_count": len(records),
        "ddp_reducer_identity_sha256": next(iter(ddp_reducer_identities)),
        "memory": memory,
    }


def _write_checksums(evidence_root: Path) -> None:
    files = []
    for path in sorted(evidence_root.rglob("*")):
        if path.is_symlink():
            raise Node3GateError("evidence directory contains a symlink")
        if path.is_file() and path.name not in CHECKSUM_EXCLUSIONS:
            files.append(path)
    manifest_path = evidence_root / "SHA256SUMS.txt"
    lines = [
        "{}  {}\n".format(sha256_file(path), path.relative_to(evidence_root).as_posix())
        for path in files
    ]
    descriptor = os.open(
        str(manifest_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
    )
    try:
        os.write(descriptor, "".join(lines).encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    sidecar = evidence_root / "SHA256SUMS.txt.sha256"
    write_text_exclusive(
        sidecar,
        "{}  SHA256SUMS.txt\n".format(sha256_file(manifest_path)),
    )
    verified = 0
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if sha256_file(evidence_root / relative) != digest:
            raise Node3GateError("evidence checksum verification failed")
        verified += 1
    write_text_exclusive(
        evidence_root / "SHA256SUMS.verify.log",
        "status=passed\nverified_files={}\nexcluded_infrastructure={}\n".format(
            verified, ",".join(sorted(CHECKSUM_EXCLUSIONS))
        ),
    )
    directory_fd = os.open(str(evidence_root), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-project-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--torchrun", type=Path, required=True)
    parser.add_argument("--audiocraft-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--cfg-scale-decision-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=WORKPACK_ROOT / "runs")
    parser.add_argument(
        "--console-root", type=Path, default=WORKPACK_ROOT / "console_logs"
    )
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--learning-rate", type=float, default=3.0e-6)
    parser.add_argument("--timestamp", help="optional UTC YYYYMMDDTHHMMSSZ run id")
    return parser.parse_args(argv)


def execute(args: argparse.Namespace) -> Tuple[Path, bool]:
    timestamp = args.timestamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if UTC_PATTERN.fullmatch(timestamp) is None:
        raise Node3GateError("--timestamp must be UTC YYYYMMDDTHHMMSSZ")
    validate_node3_protocol_hyperparameters(args.seed, args.learning_rate)
    if WORKPACK_ROOT.is_symlink():
        raise Node3GateError("workpack root may not be a symlink")
    for label, path in (("python", args.python), ("torchrun", args.torchrun)):
        try:
            executable = path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise Node3GateError("{} does not exist: {}".format(label, path)) from exc
        if not executable.is_file() or not os.access(str(executable), os.X_OK):
            raise Node3GateError("{} must resolve to an executable file".format(label))
    if args.manifest.is_symlink() or not args.manifest.is_file():
        raise Node3GateError("manifest must be a regular non-symlink file")
    for label, path in (
        ("base project", args.base_project_root),
        ("AudioCraft", args.audiocraft_root),
        ("student checkpoint", args.student_checkpoint),
        ("teacher checkpoint", args.teacher_checkpoint),
        ("CFG decision", args.cfg_scale_decision_dir),
    ):
        if path.is_symlink() or not path.is_dir():
            raise Node3GateError("{} must be a regular directory: {}".format(label, path))

    # Validate both prospective paths before the first mkdir. A typo outside
    # the workpack must therefore leave no directory behind.
    args.console_root, args.runs_root = prepare_output_roots(
        args.console_root, args.runs_root, WORKPACK_ROOT
    )
    evidence_root = args.console_root / ("node3_gate_" + timestamp)
    uninterrupted = args.runs_root / ("node3_gate_" + timestamp + ".uninterrupted")
    resumed = args.runs_root / ("node3_gate_" + timestamp + ".resume")
    resume_seed = args.runs_root / ("node3_gate_" + timestamp + ".resume.seed")
    cold_a = args.runs_root / ("node3_gate_" + timestamp + ".resume.cold-a")
    copied = args.runs_root / ("node3_gate_" + timestamp + ".resume.copy")
    for path in (
        evidence_root,
        uninterrupted,
        resumed,
        resume_seed,
        cold_a,
        copied,
    ):
        if path.exists() or path.is_symlink():
            raise Node3GateError("refusing to reuse output path {}".format(path))
    evidence_root.mkdir()
    environment, removed = clean_child_environment()
    recorder = GateRecorder(evidence_root, environment)
    write_json_exclusive(
        evidence_root / "environment.json",
        {
            "schema_version": SCHEMA_VERSION,
            "hostname": socket.gethostname(),
            "workpack": str(WORKPACK_ROOT),
            "python": str(args.python),
            "torchrun": str(args.torchrun),
            "removed_pollution_variables": removed,
            "child_contract": {
                key: environment[key]
                for key in (
                    "PTC_NODE3_GATE",
                    "PYTHONDONTWRITEBYTECODE",
                    "PYTHONPATH",
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                    "HF_DATASETS_OFFLINE",
                    "CUBLAS_WORKSPACE_CONFIG",
                )
            },
            "launcher": "manual_torchrun_standalone",
            "slurm": False,
            "cross_machine_process_group": False,
        },
    )
    active_gate = GATE_IDS[0]
    success = False
    failure_message: Optional[str] = None
    try:
        active_gate = "A01_workpack_seal"
        meta = recorder.run(
            active_gate,
            "workpack_seal_verify",
            [
                str(args.python),
                str(WORKPACK_ROOT / "scripts" / "seal_workpack.py"),
                "verify",
                "--root",
                str(WORKPACK_ROOT),
            ],
            cwd=WORKPACK_ROOT,
        )
        recorder.pass_gate(active_gate, [meta["stdout"], meta["stderr"], meta["label"]])

        active_gate = "A02_workpack_check"
        meta = recorder.run(
            active_gate,
            "workpack_check",
            [
                str(args.python),
                str(WORKPACK_ROOT / "scripts" / "check_workpack.py"),
                "--base-project-root",
                str(args.base_project_root),
            ],
            cwd=WORKPACK_ROOT,
        )
        recorder.pass_gate(active_gate, [meta["stdout"], meta["stderr"]])

        active_gate = "A03_overlay_exact"
        meta = recorder.run(
            active_gate,
            "overlay_exact",
            [
                str(args.python),
                str(WORKPACK_ROOT / "scripts" / "verify_node3_overlay.py"),
                "--audiocraft-root",
                str(args.audiocraft_root),
            ],
            cwd=WORKPACK_ROOT,
        )
        recorder.pass_gate(active_gate, [meta["stdout"], meta["stderr"]])

        active_gate = "A04_overlay_test"
        meta = recorder.run(
            active_gate,
            "overlay_test",
            [
                str(args.python),
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "tests/models/test_lm_no_cfg.py",
            ],
            cwd=args.audiocraft_root,
        )
        recorder.pass_gate(active_gate, [meta["stdout"], meta["stderr"]])

        active_gate = "A05_full_tests"
        meta = recorder.run(
            active_gate,
            "full_workpack_tests",
            [
                str(args.python),
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-q",
            ],
            cwd=WORKPACK_ROOT,
        )
        recorder.pass_gate(active_gate, [meta["stdout"], meta["stderr"]])

        active_gate = "A06_eight_process_ratio"
        legacy_ratio_test = recorder.run(
            active_gate,
            "distributed_target_test",
            _distributed_target_test_command(
                args.python, WORKPACK_ROOT / "tests"
            ),
            cwd=WORKPACK_ROOT,
            environment_overrides={"PTC_RUN_MULTIPROCESS_TESTS": "1"},
        )
        target_test_validation_path = (
            evidence_root / "distributed_target_test_validation.json"
        )
        target_test_validation = validate_distributed_target_test_output(
            (evidence_root / legacy_ratio_test["stdout"]).read_text(
                encoding="utf-8", errors="replace"
            ),
            (evidence_root / legacy_ratio_test["stderr"]).read_text(
                encoding="utf-8", errors="replace"
            ),
        )
        write_json_exclusive(target_test_validation_path, target_test_validation)
        ratio_path = evidence_root / "eight_process_ratio.json"
        meta = recorder.run(
            active_gate,
            "eight_process_ratio",
            [
                str(args.torchrun),
                "--standalone",
                "--nnodes=1",
                "--nproc_per_node=8",
                str(WORKPACK_ROOT / "scripts" / "node3_ratio_probe.py"),
                "--output",
                str(ratio_path),
            ],
            cwd=WORKPACK_ROOT,
        )
        ratio = load_json(ratio_path)
        if ratio.get("status") != "passed" or ratio.get("world_size") != 8:
            raise Node3GateError("eight-process ratio artifact is invalid")
        recorder.pass_gate(
            active_gate,
            [
                legacy_ratio_test["stdout"],
                legacy_ratio_test["stderr"],
                target_test_validation_path.name,
                meta["stdout"],
                meta["stderr"],
                ratio_path.name,
            ],
        )
        write_json_exclusive(
            evidence_root / "phase_a_status.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "passed",
                "gates": list(GATE_IDS[:6]),
            },
        )

        active_gate = "B01_uninterrupted_two_updates"
        retained_small_identity = _verify_retained_small_decision(
            args.cfg_scale_decision_dir
        )
        write_json_exclusive(
            evidence_root / "retained_small_cfg_decision_identity.json",
            retained_small_identity,
        )
        dry_meta = recorder.run(
            "B01_uninterrupted_two_updates",
            "stage1_dry_run",
            _dry_run_command(args, args.runs_root / ("node3_gate_" + timestamp + ".dry-run-not-created")),
            cwd=WORKPACK_ROOT,
        )
        train_meta = recorder.run(
            active_gate,
            "uninterrupted_two_updates",
            _training_command(args, uninterrupted),
            cwd=WORKPACK_ROOT,
        )
        uninterrupted_verify = recorder.run(
            active_gate,
            "uninterrupted_verify",
            [
                str(args.python),
                str(WORKPACK_ROOT / "scripts" / "verify_stage1_run.py"),
                "--run-dir",
                str(uninterrupted),
            ],
            cwd=WORKPACK_ROOT,
        )
        if load_json(uninterrupted / "DONE.json").get("status") != "complete":
            raise Node3GateError("uninterrupted smoke lacks DONE success commit")
        retained_small_after = _verify_retained_small_decision(
            args.cfg_scale_decision_dir
        )
        if retained_small_after != retained_small_identity:
            raise Node3GateError("retained small decision identity changed during B01")
        write_json_exclusive(
            evidence_root / "retained_small_cfg_decision_identity.after_b01.json",
            retained_small_after,
        )
        recorder.pass_gate(
            active_gate,
            [
                "retained_small_cfg_decision_identity.json",
                "retained_small_cfg_decision_identity.after_b01.json",
                dry_meta["stdout"],
                train_meta["stdout"],
                uninterrupted_verify["stdout"],
            ],
        )

        active_gate = "B02_controlled_step1_failure"
        interrupted_command = _training_command(args, resumed)
        interrupted_command.extend(["--node3-stop-after-step", "1"])
        injected = recorder.run(
            active_gate,
            "controlled_step1_failure",
            interrupted_command,
            cwd=WORKPACK_ROOT,
            expected_success=False,
            required_marker="NODE3_GATE_INJECTED_STOP_AFTER_STEP_1",
        )
        failure = load_json(resumed / "FAILED.json")
        if "NODE3_GATE_INJECTED_STOP_AFTER_STEP_1" not in str(failure.get("error")):
            raise Node3GateError("controlled failure marker is absent from FAILED.json")
        for path in (
            resumed / "checkpoints" / "step-00001" / "checkpoint.pt",
            resumed / "checkpoints" / "step-00001" / "SHA256.json",
            resumed / "logs" / "attempt-0000.json",
            resumed / "logs" / "metrics.attempt-0000.jsonl",
        ):
            if path.is_symlink() or not path.is_file():
                raise Node3GateError("controlled step-1 evidence is incomplete")
        before = _attempt0_identities(resumed)
        preresume_seed_before = _preresume_seed_identities(resumed)
        preresume_layout_before = _validate_preresume_seed_closed_world(resumed)
        write_json_exclusive(evidence_root / "attempt0_before.json", before)
        write_json_exclusive(
            evidence_root / "preresume_seed_before.json", preresume_seed_before
        )
        write_json_exclusive(
            evidence_root / "preresume_seed_layout_before.json",
            preresume_layout_before,
        )
        recorder.pass_gate(
            active_gate,
            [
                injected["stdout"],
                injected["stderr"],
                "attempt0_before.json",
                "preresume_seed_before.json",
                "preresume_seed_layout_before.json",
            ],
        )

        active_gate = "B04_stale_rejected_latest_accepted"
        stale = recorder.run(
            active_gate,
            "stale_step0_rejected",
            _training_command(
                args, resumed, resume=resumed / "checkpoints" / "step-00000"
            ),
            cwd=WORKPACK_ROOT,
            expected_success=False,
            required_marker="--resume must name the latest committed checkpoint",
        )
        if _attempt0_identities(resumed) != before:
            raise Node3GateError("stale-resume rejection changed attempt-0000")
        if _validate_preresume_seed_closed_world(resumed) != preresume_layout_before:
            raise Node3GateError("stale-resume rejection changed the seed layout")
        seed_copy = recorder.run(
            active_gate,
            "copy_preresume_seed",
            ["cp", "-a", "--reflink=auto", str(resumed), str(resume_seed)],
            cwd=WORKPACK_ROOT,
        )
        if _preresume_seed_identities(resume_seed) != preresume_seed_before:
            raise Node3GateError("copied pre-resume seed differs from its source")
        copied_seed_layout = _validate_preresume_seed_closed_world(resume_seed)
        if copied_seed_layout != preresume_layout_before:
            raise Node3GateError("copied pre-resume seed layout differs")
        write_json_exclusive(
            evidence_root / "preresume_seed_copy_layout.json", copied_seed_layout
        )
        latest = recorder.run(
            active_gate,
            "latest_step1_cold_resume_a",
            _training_command(
                args, resumed, resume=resumed / "checkpoints" / "step-00001"
            ),
            cwd=WORKPACK_ROOT,
        )
        attempt1 = load_json(resumed / "logs" / "attempt-0001.json")
        if attempt1.get("start_optimizer_step") != 1:
            raise Node3GateError("attempt-0001 did not resume from optimizer step 1")
        if attempt1.get("start_checkpoint", {}).get("directory") != "checkpoints/step-00001":
            raise Node3GateError("attempt-0001 start checkpoint differs")
        recorder.pass_gate(
            active_gate,
            [
                stale["stderr"],
                seed_copy["stdout"],
                "preresume_seed_copy_layout.json",
                latest["stdout"],
                "logs/attempt-0001.json",
            ],
        )

        active_gate = "B03_attempt0_immutable"
        after = _attempt0_identities(resumed)
        preresume_seed_after = _preresume_seed_identities(resumed)
        write_json_exclusive(evidence_root / "attempt0_after.json", after)
        write_json_exclusive(
            evidence_root / "preresume_seed_after.json", preresume_seed_after
        )
        if after != before:
            raise Node3GateError("attempt-0000 bytes changed across stale test/resume")
        if preresume_seed_after != preresume_seed_before:
            raise Node3GateError("immutable pre-resume seed changed across resume")
        recorder.pass_gate(
            active_gate,
            [
                "attempt0_before.json",
                "attempt0_after.json",
                "preresume_seed_before.json",
                "preresume_seed_after.json",
            ],
        )

        active_gate = "B05_resume_equivalence"
        first_resume_verify = recorder.run(
            active_gate,
            "cold_resume_a_verify_before_rotation",
            [
                str(args.python),
                str(WORKPACK_ROOT / "scripts" / "verify_stage1_run.py"),
                "--run-dir",
                str(resumed),
            ],
            cwd=WORKPACK_ROOT,
        )
        rotation = rotate_completed_resume_with_seed(
            completed=resumed,
            seed=resume_seed,
            retained=cold_a,
            runs_root=args.runs_root,
        )
        rotation_path = evidence_root / "cold_resume_rotation.json"
        write_json_exclusive(rotation_path, rotation)
        cold_a_relocated_verify = recorder.run(
            active_gate,
            "cold_resume_a_verify_after_rotation",
            [
                str(args.python),
                str(WORKPACK_ROOT / "scripts" / "verify_stage1_run.py"),
                "--run-dir",
                str(cold_a),
            ],
            cwd=WORKPACK_ROOT,
        )
        cold_b_meta = recorder.run(
            active_gate,
            "latest_step1_cold_resume_b",
            _training_command(
                args, resumed, resume=resumed / "checkpoints" / "step-00001"
            ),
            cwd=WORKPACK_ROOT,
        )
        if _preresume_seed_identities(resumed) != preresume_seed_before:
            raise Node3GateError("cold-resume B changed an immutable seed member")
        resumed_verify = recorder.run(
            active_gate,
            "cold_resume_b_verify",
            [
                str(args.python),
                str(WORKPACK_ROOT / "scripts" / "verify_stage1_run.py"),
                "--run-dir",
                str(resumed),
            ],
            cwd=WORKPACK_ROOT,
        )
        for label, run in (("cold-A", cold_a), ("cold-B", resumed)):
            done = load_json(run / "DONE.json")
            seal = load_json(run / "SEALED.json")
            if done.get("failed_record_superseded") is not True:
                raise Node3GateError(
                    "{} DONE does not bind the superseded controlled failure".format(
                        label
                    )
                )
            if seal.get("superseded_failure") is None:
                raise Node3GateError(
                    "{} SEALED does not bind the controlled failure".format(label)
                )
            if len(seal.get("attempt_logs", [])) != 2:
                raise Node3GateError(
                    "{} seal does not bind exactly two attempts".format(label)
                )
        resumed_done = load_json(resumed / "DONE.json")
        resumed_seal = load_json(resumed / "SEALED.json")

        uninterrupted_step2 = _record_at_step(
            uninterrupted / "logs" / "metrics.attempt-0000.jsonl", 2
        )
        cold_a_step2 = _record_at_step(
            cold_a / "logs" / "metrics.attempt-0001.jsonl", 2
        )
        resumed_step2 = _record_at_step(
            resumed / "logs" / "metrics.attempt-0001.jsonl", 2
        )
        uninterrupted_audit = _canonical_resume_audit(uninterrupted_step2)
        cold_a_audit = _canonical_resume_audit(cold_a_step2)
        resumed_audit = _canonical_resume_audit(resumed_step2)
        if uninterrupted_audit != resumed_audit:
            raise Node3GateError(
                "resumed step-2 samples/gates/rollouts/DDP topology differ "
                "from uninterrupted"
            )
        if cold_a_audit != resumed_audit:
            raise Node3GateError(
                "two independent cold resumes used different samples/gates/rollouts/DDP topology"
            )
        resume_rng_restore_a = _validate_resume_rng_restore(cold_a_step2)
        resume_rng_restore_b = _validate_resume_rng_restore(resumed_step2)
        if resume_rng_restore_a != resume_rng_restore_b:
            raise Node3GateError("two cold resumes restored different RNG identities")
        resume_state_restore_a = _validate_resume_state_restore(cold_a_step2)
        resume_state_restore_b = _validate_resume_state_restore(resumed_step2)
        if resume_state_restore_a != resume_state_restore_b:
            raise Node3GateError("two cold resumes restored different live-state identities")
        warm_cold_metrics = _compare_step2_metrics(
            uninterrupted_step2, resumed_step2
        )
        cold_cold_metrics = _compare_cold_step2_metrics_exact(
            cold_a_step2, resumed_step2
        )

        cold_exact_path = evidence_root / "cold_resume_exact_state.json"
        cold_exact_meta = recorder.run(
            active_gate,
            "cold_resume_checkpoint_exact_equivalence",
            [
                str(args.python),
                str(WORKPACK_ROOT / "scripts" / "compare_node3_checkpoints.py"),
                "--uninterrupted-run",
                str(cold_a),
                "--resumed-run",
                str(resumed),
                "--output",
                str(cold_exact_path),
            ],
            cwd=WORKPACK_ROOT,
        )
        cold_exact = load_json(cold_exact_path)
        if cold_exact.get("status") != "passed" or set(
            cold_exact.get("comparisons", {})
        ) != {"step_1", "step_2"}:
            raise Node3GateError(
                "two independent cold resumes were not checkpoint-exact"
            )

        warm_cold_path = evidence_root / "warm_cold_numerical_continuity.json"
        warm_cold_meta = recorder.run(
            active_gate,
            "warm_cold_numerical_continuity",
            [
                str(args.python),
                str(WORKPACK_ROOT / "scripts" / "analyze_node3_checkpoint_drift.py"),
                "--uninterrupted-run",
                str(uninterrupted),
                "--resumed-run",
                str(resumed),
                "--output",
                str(warm_cold_path),
            ],
            cwd=WORKPACK_ROOT,
        )
        warm_cold = load_json(warm_cold_path)
        if warm_cold.get("status") != "passed":
            raise Node3GateError("warm/cold numerical-continuity consumer did not pass")

        equivalence_path = evidence_root / "resume_equivalence.json"
        write_json_exclusive(
            equivalence_path,
            {
                "schema_version": B05_LIFECYCLE_SCHEMA_VERSION,
                "status": "passed",
                "compared_optimizer_step": 2,
                "rank_count": 8,
                "microsteps_per_rank": 4,
                "sample_and_gate_policy": "exact structured equality",
                "audit": resumed_audit,
                "resume_rng_restore": resume_rng_restore_b,
                "resume_state_restore": resume_state_restore_b,
                "warm_cold_metric_equivalence": warm_cold_metrics,
                "cold_cold_metric_equivalence": cold_cold_metrics,
                "checkpoint_state_policy": {
                    "checkpoint_to_live_after_load": "exact on every rank",
                    "cold_resume_a_vs_b": cold_exact.get("policy"),
                    "warm_vs_cold": warm_cold.get("policy"),
                },
                "cold_resume_exact_evidence": {
                    "path": cold_exact_path.name,
                    "sha256": sha256_file(cold_exact_path),
                },
                "warm_cold_numerical_evidence": {
                    "path": warm_cold_path.name,
                    "sha256": sha256_file(warm_cold_path),
                },
            },
        )
        recorder.pass_gate(
            active_gate,
            [
                first_resume_verify["stdout"],
                cold_a_relocated_verify["stdout"],
                cold_b_meta["stdout"],
                resumed_verify["stdout"],
                rotation_path.name,
                cold_exact_meta["stdout"],
                cold_exact_path.name,
                warm_cold_meta["stdout"],
                warm_cold_path.name,
                equivalence_path.name,
            ],
        )

        active_gate = "B06_copied_run_verification"
        copy_meta = recorder.run(
            active_gate,
            "copy_resumed_run",
            ["cp", "-a", "--reflink=auto", str(resumed), str(copied)],
            cwd=WORKPACK_ROOT,
        )
        copied_verify = recorder.run(
            active_gate,
            "copied_run_verify",
            [
                str(args.python),
                str(WORKPACK_ROOT / "scripts" / "verify_stage1_run.py"),
                "--run-dir",
                str(copied),
            ],
            cwd=WORKPACK_ROOT,
        )
        original_verify_payload = json.loads(
            (evidence_root / resumed_verify["stdout"]).read_text(encoding="utf-8")
        )
        copied_verify_payload = json.loads(
            (evidence_root / copied_verify["stdout"]).read_text(encoding="utf-8")
        )
        if original_verify_payload.get("status") != "verified":
            raise Node3GateError("original resumed run verifier did not pass")
        if copied_verify_payload.get("status") != "verified":
            raise Node3GateError("copied resumed run verifier did not pass")
        for key in (
            "optimizer_step",
            "attempt_count",
            "checkpoint_count",
            "run_manifest_sha256",
            "SEALED.json_sha256",
            "DONE.json_sha256",
            "final_checkpoint_sha256",
        ):
            if original_verify_payload.get(key) != copied_verify_payload.get(key):
                raise Node3GateError("copied verification identity differs at {}".format(key))
        recorder.pass_gate(
            active_gate,
            [
                resumed_verify["stdout"],
                copy_meta["stdout"],
                copied_verify["stdout"],
            ],
        )

        uninterrupted_manifest = load_json(uninterrupted / "run_manifest.json")
        cold_a_manifest = load_json(cold_a / "run_manifest.json")
        resumed_manifest = load_json(resumed / "run_manifest.json")
        active_gate = "B07_cpu_fp32_conditioners_nocfg"
        if _verify_retained_small_decision(
            args.cfg_scale_decision_dir
        ) != retained_small_identity:
            raise Node3GateError("retained small decision identity changed during node-3")
        for manifest in (uninterrupted_manifest, cold_a_manifest, resumed_manifest):
            config = manifest.get("config")
            if not isinstance(config, dict) or config.get(
                "cfg_scale_decision_file_sha256"
            ) != RETAINED_SMALL_DECISION_SHA256:
                raise Node3GateError(
                    "Stage-1 run manifest is not bound to the retained small decision pin"
                )
        runtime_uninterrupted = _validate_runtime_contract(uninterrupted_manifest)
        runtime_cold_a = _validate_runtime_contract(cold_a_manifest)
        runtime_resumed = _validate_runtime_contract(resumed_manifest)
        runtime_identities = {
            runtime["ddp_reducer_identity_sha256"]
            for runtime in (
                runtime_uninterrupted,
                runtime_cold_a,
                runtime_resumed,
            )
        }
        if len(runtime_identities) != 1:
            raise Node3GateError(
                "DDP reducer construction differs across warm/cold runs"
            )
        runtime_path = evidence_root / "runtime_contract_summary.json"
        write_json_exclusive(
            runtime_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "passed",
                "uninterrupted": runtime_uninterrupted,
                "cold_resume_a": runtime_cold_a,
                "resumed": runtime_resumed,
            },
        )
        all_records = load_jsonl(
            uninterrupted / "logs" / "metrics.attempt-0000.jsonl"
        ) + load_jsonl(cold_a / "logs" / "metrics.attempt-0000.jsonl") + load_jsonl(
            cold_a / "logs" / "metrics.attempt-0001.jsonl"
        ) + load_jsonl(resumed / "logs" / "metrics.attempt-0000.jsonl") + load_jsonl(
            resumed / "logs" / "metrics.attempt-0001.jsonl"
        )
        step_summary = _validate_step_records(all_records)
        if step_summary["ddp_reducer_identity_sha256"] != runtime_uninterrupted[
            "ddp_reducer_identity_sha256"
        ]:
            raise Node3GateError(
                "DDP reducer step identity differs from construction identity"
            )
        recorder.pass_gate(active_gate, [runtime_path.name, "resume_equivalence.json"])

        active_gate = "B08_bf16_finite_denominator_teacher"
        for name, manifest, seal in (
            ("uninterrupted", uninterrupted_manifest, load_json(uninterrupted / "SEALED.json")),
            ("cold_resume_a", cold_a_manifest, load_json(cold_a / "SEALED.json")),
            ("resumed", resumed_manifest, resumed_seal),
        ):
            initial = manifest.get("teacher_state_sha256_initial")
            final = seal.get("teacher_state_sha256_final")
            if not isinstance(initial, str) or SHA256_PATTERN.fullmatch(initial) is None:
                raise Node3GateError("{} teacher initial hash is malformed".format(name))
            if initial != final:
                raise Node3GateError("{} frozen teacher hash changed".format(name))
        numerical_path = evidence_root / "numerical_contract_summary.json"
        write_json_exclusive(
            numerical_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "passed",
                "metric_record_count": step_summary["record_count"],
                "denominator_rtol": 1.0e-6,
                "teacher_hash_unchanged": True,
                "bf16_autocast": "passed_on_all_ranks",
                "finite_loss_and_gradient": True,
            },
        )
        recorder.pass_gate(active_gate, [numerical_path.name])

        active_gate = "B09_all_rank_memory"
        memory_path = evidence_root / "all_rank_memory.json"
        write_json_exclusive(
            memory_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "passed",
                "records": step_summary["memory"],
                "rank_coverage": list(range(8)),
                "interpretation": (
                    "two-update smoke peak only; rerun inspection before a 500-update gate"
                ),
            },
        )
        recorder.pass_gate(active_gate, [memory_path.name])
        success = True
    except BaseException as exc:
        failure_message = "{}: {}".format(type(exc).__name__, exc)
        recorder.fail_gate(active_gate, failure_message)
        try:
            write_json_exclusive(
                evidence_root / "FAILURE.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed",
                    "active_gate": active_gate,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        except FileExistsError:
            pass
    status = {
        "schema_version": SCHEMA_VERSION,
        "b05_policy_schema_version": B05_LIFECYCLE_SCHEMA_VERSION,
        "status": "passed" if success else "failed",
        "timestamp_utc": timestamp,
        "hostname": socket.gethostname(),
        "workpack": str(WORKPACK_ROOT),
        "evidence_root": str(evidence_root),
        "runs": {
            "uninterrupted": str(uninterrupted),
            "cold_resume_a": str(cold_a),
            "resumed": str(resumed),
            "copied": str(copied),
        },
        "gate_count": len(GATE_IDS),
        "passed_count": sum(
            result["status"] == "passed" for result in recorder.results.values()
        ),
        "gates": recorder.results,
        "failure": failure_message,
        "authorization": {
            "slurm": False,
            "sbatch": False,
            "submitit": False,
            "dora": False,
            "cross_machine_ddp": False,
            "research_training_authorized": False,
        },
        "checksum_policy": {
            "covered": "all regular evidence files frozen before SHA256SUMS",
            "excluded": sorted(CHECKSUM_EXCLUSIONS),
        },
    }
    write_json_exclusive(evidence_root / "STATUS.json", status)
    _write_checksums(evidence_root)
    return evidence_root, success


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        evidence_root, success = execute(args)
    except (Node3GateError, OSError, ValueError) as exc:
        print("NODE3 GATE SETUP FAILED: {}".format(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "passed" if success else "failed",
                "evidence_root": str(evidence_root),
            },
            sort_keys=True,
        )
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
