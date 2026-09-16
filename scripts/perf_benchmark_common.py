#!/usr/bin/env python3
"""Pure-logic contract for the Stage-1 DDP/telemetry performance assay.

This module intentionally imports only the Python standard library.  The GPU
worker, the four-node orchestrator, the aggregator, the verifier, and CPU-only
tests all consume the same closed definitions from here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


TRAIN_STAGE1_SHA256 = "697a450da4848a1589b257816e2572dbc681ae54e4535d300d2fda39257df7c5"

BENCHMARK_SCHEMA_VERSION = "ptc-opd-stage1-perf-benchmark-v1"
RANK_SCHEMA_VERSION = "ptc-opd-stage1-perf-rank-v1"
ARM_STATUS_SCHEMA_VERSION = "ptc-opd-stage1-perf-arm-status-v1"
NODE_STATUS_SCHEMA_VERSION = "ptc-opd-stage1-perf-node-status-v1"
SUMMARY_SCHEMA_VERSION = "ptc-opd-stage1-perf-summary-v1"
SEAL_SCHEMA_VERSION = "ptc-opd-stage1-perf-seal-v1"

PERF_RUN_SCHEMA_VERSION = "ptc-opd-benchmark-only-run-v1"
PERF_CHECKPOINT_SCHEMA_VERSION = "ptc-opd-benchmark-only-checkpoint-v1"
PERF_ATTEMPT_SCHEMA_VERSION = "ptc-opd-benchmark-only-attempt-v1"
PERF_SEAL_SCHEMA_VERSION = "ptc-opd-benchmark-only-terminal-seal-v1"
PERF_DONE_SCHEMA_VERSION = "ptc-opd-benchmark-only-done-v1"
PERF_REDUCER_SCHEMA_VERSION = "ptc-opd-benchmark-only-ddp-reducer-v1"

FORMAL_RUN_SCHEMA_VERSION = "ptc-opd-stage1-run-v3"

PERF_INPUT_BINDING_SCHEMA_VERSION = "ptc-opd-stage1-perf-input-binding-v1"
CFG_DECISION_FILENAME = "cfg_scale_decision.json"
CFG_DECISION_SIDECAR_FILENAME = "cfg_scale_decision.sha256.json"
AUDIOCRAFT_SOURCE_SCHEMA_VERSION = "ptc-opd-audiocraft-source-tree-v1"
AUDIOCRAFT_SOURCE_EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "artifacts",
        "build",
        "dist",
        "logs",
        "outputs",
        "runs",
    }
)
AUDIOCRAFT_SOURCE_EXCLUDED_FILE_SUFFIXES = (".pyc", ".pyo")

WARMUP_STEPS = 10
MEASURED_STEPS = 30
BLOCK_STEPS = 5
TOTAL_STEPS = WARMUP_STEPS + MEASURED_STEPS
EXPECTED_BLOCKS = MEASURED_STEPS // BLOCK_STEPS
WORLD_SIZE = 8

ARM_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "A": {
        "name": "true_full",
        "find_unused_parameters": True,
        "audit_mode": "full",
    },
    "B": {
        "name": "false_full",
        "find_unused_parameters": False,
        "audit_mode": "full",
    },
    "C": {
        "name": "true_min",
        "find_unused_parameters": True,
        "audit_mode": "min",
    },
    "D": {
        "name": "false_min",
        "find_unused_parameters": False,
        "audit_mode": "min",
    },
}

ARM_NAME_TO_ID = {value["name"]: key for key, value in ARM_DEFINITIONS.items()}

# Balanced Williams square.  Every arm appears once in every period, and every
# ordered carry-over pair appears exactly once across the four node sequences.
WILLIAMS_SCHEDULE: Dict[str, Tuple[str, ...]] = {
    "node-0": ("A", "B", "D", "C"),
    "node-1": ("B", "C", "A", "D"),
    "node-2": ("C", "D", "B", "A"),
    "node-3": ("D", "A", "C", "B"),
}

COMPONENT_NAMES = (
    "rollout_codes_sha_seconds",
    "selected_gate_sha_seconds",
    "reducer_audit_seconds",
    "audit_all_gather_seconds",
    "json_encode_write_fsync_seconds",
    "stdout_flush_seconds",
)


class PerfContractError(ValueError):
    """Raised when performance evidence violates its closed contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path, relative: Optional[str] = None) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PerfContractError("expected regular file: {}".format(path))
    return {
        "path": relative if relative is not None else path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _canonical_input_path(path: Path, label: str, *, kind: str) -> Path:
    supplied = path.expanduser().absolute()
    if supplied.is_symlink():
        raise PerfContractError("{} root may not be a symlink".format(label))
    resolved = supplied.resolve(strict=True)
    if kind == "file":
        if resolved.is_symlink() or not resolved.is_file():
            raise PerfContractError("{} must be a regular file".format(label))
    elif kind == "tree":
        if not resolved.is_dir():
            raise PerfContractError("{} must be a directory".format(label))
    else:
        raise PerfContractError("unsupported live-input kind {!r}".format(kind))
    return resolved


def _require_regular_tree(root: Path, label: str) -> None:
    for member in sorted(root.rglob("*")):
        if member.is_symlink():
            raise PerfContractError(
                "{} contains a symlink: {}".format(label, member.relative_to(root))
            )
        if not member.is_file() and not member.is_dir():
            raise PerfContractError(
                "{} contains a non-regular member: {}".format(
                    label, member.relative_to(root)
                )
            )


def sha256_path(path: Path) -> str:
    """Match the formal Stage-1 checkpoint tree digest exactly."""

    resolved = path.resolve(strict=True)
    if resolved.is_file():
        return sha256_file(resolved)
    if not resolved.is_dir():
        raise PerfContractError("checkpoint path must be a file or directory")
    files = sorted(item for item in resolved.rglob("*") if item.is_file())
    if not files:
        raise PerfContractError("checkpoint directory contains no files")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(resolved).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(item.stat().st_size.to_bytes(8, "big"))
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def audiocraft_source_identity(root: Path) -> Dict[str, Any]:
    """Match ``ptc_opd.reproducibility.audiocraft_source_identity``."""

    required = root / "audiocraft" / "models" / "lm.py"
    if required.is_symlink() or not required.is_file():
        raise PerfContractError("AudioCraft source has no regular audiocraft/models/lm.py")
    digest = hashlib.sha256()
    count = 0
    for member in sorted(root.rglob("*")):
        relative = member.relative_to(root)
        if any(
            part in AUDIOCRAFT_SOURCE_EXCLUDED_DIRECTORY_NAMES
            for part in relative.parts[:-1]
        ):
            continue
        if member.is_symlink():
            raise PerfContractError(
                "AudioCraft source contains a symlink: {}".format(relative)
            )
        if member.is_dir():
            continue
        if not member.is_file():
            raise PerfContractError(
                "AudioCraft source contains a non-regular member: {}".format(relative)
            )
        if (
            member.name == ".DS_Store"
            or member.suffix in AUDIOCRAFT_SOURCE_EXCLUDED_FILE_SUFFIXES
        ):
            continue
        encoded = relative.as_posix().encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(member.stat().st_size.to_bytes(8, "big"))
        with member.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        count += 1
    if count == 0:
        raise PerfContractError("AudioCraft source identity contains no files")
    payload: Dict[str, Any] = {
        "schema_version": AUDIOCRAFT_SOURCE_SCHEMA_VERSION,
        "file_count": count,
        "tree_sha256": digest.hexdigest(),
    }
    payload["identity_sha256"] = canonical_json_sha256(payload)
    return payload


def _live_cfg_identity(directory: Path) -> Dict[str, Any]:
    expected_members = {CFG_DECISION_FILENAME, CFG_DECISION_SIDECAR_FILENAME}
    observed_members = {member.name for member in directory.iterdir()}
    if observed_members != expected_members:
        raise PerfContractError("live small CFG decision member set differs")
    decision_path = directory / CFG_DECISION_FILENAME
    sidecar_path = directory / CFG_DECISION_SIDECAR_FILENAME
    decision = read_json(decision_path)
    sidecar = read_json(sidecar_path)
    decision_sha256 = sha256_file(decision_path)
    payload_sha256 = decision.get("decision_payload_sha256")
    scientific_sha256 = decision.get("scientific_config_sha256")
    generation = decision.get("generation_identity")
    if (
        not isinstance(payload_sha256, str)
        or len(payload_sha256) != 64
        or not isinstance(scientific_sha256, str)
        or len(scientific_sha256) != 64
        or not isinstance(generation, dict)
    ):
        raise PerfContractError("live small CFG decision identity is malformed")
    if (
        sidecar.get("sha256") != decision_sha256
        or sidecar.get("decision_payload_sha256") != payload_sha256
    ):
        raise PerfContractError("live small CFG sidecar identity differs")
    return {
        "selected_cfg_scale": decision.get("selected_cfg_scale"),
        "decision_file_sha256": decision_sha256,
        "decision_payload_sha256": payload_sha256,
        "scientific_config_sha256": scientific_sha256,
        "generation_identity": generation,
    }


def build_live_input_identity(
    *,
    train_manifest: Path,
    small_cfg_dir: Path,
    audiocraft_dir: Path,
    musicgen_small_dir: Path,
) -> Dict[str, Any]:
    """Rehash the four scientific inputs consumed by every benchmark arm."""

    train = _canonical_input_path(train_manifest, "train manifest", kind="file")
    cfg = _canonical_input_path(small_cfg_dir, "small CFG", kind="tree")
    source = _canonical_input_path(audiocraft_dir, "AudioCraft", kind="tree")
    checkpoint = _canonical_input_path(
        musicgen_small_dir, "MusicGen-small", kind="tree"
    )
    for tree, label in (
        (cfg, "small CFG"),
        (source, "AudioCraft"),
        (checkpoint, "MusicGen-small"),
    ):
        _require_regular_tree(tree, label)
    state_dict = checkpoint / "state_dict.bin"
    compression_state_dict = checkpoint / "compression_state_dict.bin"
    if state_dict.is_symlink() or not state_dict.is_file():
        raise PerfContractError("MusicGen-small state_dict.bin is absent")
    if compression_state_dict.is_symlink() or not compression_state_dict.is_file():
        raise PerfContractError("MusicGen-small compression_state_dict.bin is absent")
    source_identity = audiocraft_source_identity(source)
    checkpoint_identity = {
        "checkpoint_sha256": sha256_path(checkpoint),
        "state_dict_sha256": sha256_file(state_dict),
        "compression_state_dict_sha256": sha256_file(compression_state_dict),
    }
    cfg_identity = _live_cfg_identity(cfg)
    generation = cfg_identity["generation_identity"]
    expected_generation = {
        "model_id": "facebook/musicgen-small",
        "checkpoint_sha256": checkpoint_identity["checkpoint_sha256"],
        "state_dict_sha256": checkpoint_identity["state_dict_sha256"],
        "compression_state_dict_sha256": checkpoint_identity[
            "compression_state_dict_sha256"
        ],
        "audiocraft_source_sha256": source_identity["tree_sha256"],
        "audiocraft_lm_sha256": sha256_file(
            source / "audiocraft" / "models" / "lm.py"
        ),
    }
    for field, expected in expected_generation.items():
        if generation.get(field) != expected:
            raise PerfContractError(
                "live CFG generation binding differs at {}".format(field)
            )
    return {
        "schema_version": PERF_INPUT_BINDING_SCHEMA_VERSION,
        "paths": {
            "train_manifest": str(train),
            "small_cfg_dir": str(cfg),
            "audiocraft_dir": str(source),
            "musicgen_small_dir": str(checkpoint),
        },
        "manifest_sha256": sha256_file(train),
        "checkpoint_identity": checkpoint_identity,
        "audiocraft_source_identity": source_identity,
        "audiocraft_lm_sha256": expected_generation["audiocraft_lm_sha256"],
        "cfg_decision_identity": cfg_identity,
    }


def read_json(path: Path) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PerfContractError("expected regular JSON file: {}".format(path))
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise PerfContractError("{} must contain a JSON object".format(path))
    return value


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(str(path), flags, 0o644)
    try:
        payload = json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def arm_definition(arm: str) -> Dict[str, Any]:
    arm_id = ARM_NAME_TO_ID.get(arm, arm)
    if arm_id not in ARM_DEFINITIONS:
        raise PerfContractError("unknown benchmark arm {!r}".format(arm))
    return {"arm_id": arm_id, **ARM_DEFINITIONS[arm_id]}


def schedule_for_node(node_label: str) -> Tuple[str, ...]:
    try:
        return WILLIAMS_SCHEDULE[node_label]
    except KeyError as exc:
        raise PerfContractError(
            "node label must be one of {}".format(sorted(WILLIAMS_SCHEDULE))
        ) from exc


def validate_williams_schedule() -> None:
    arms = set(ARM_DEFINITIONS)
    if set(WILLIAMS_SCHEDULE) != {"node-0", "node-1", "node-2", "node-3"}:
        raise PerfContractError("Williams schedule must cover node-0..3")
    for node, sequence in WILLIAMS_SCHEDULE.items():
        if len(sequence) != 4 or set(sequence) != arms:
            raise PerfContractError("{} does not contain each arm once".format(node))
    for period in range(4):
        if {sequence[period] for sequence in WILLIAMS_SCHEDULE.values()} != arms:
            raise PerfContractError("period {} is not balanced".format(period + 1))
    observed_pairs = {
        (sequence[index], sequence[index + 1])
        for sequence in WILLIAMS_SCHEDULE.values()
        for index in range(3)
    }
    expected_pairs = {(left, right) for left in arms for right in arms if left != right}
    if observed_pairs != expected_pairs:
        raise PerfContractError("Williams schedule does not balance ordered carry-over")


def _finite_positive(value: Any, label: str) -> float:
    if type(value) not in (int, float):
        raise PerfContractError("{} must be numeric".format(label))
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise PerfContractError("{} must be finite and positive".format(label))
    return result


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise PerfContractError("cannot compute a quantile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise PerfContractError("quantile probability is outside [0,1]")
    ordered = sorted(_finite_positive(value, "quantile value") for value in values)
    if len(ordered) == 1:
        return ordered[0]
    location = probability * (len(ordered) - 1)
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def robust_cv(values: Sequence[float]) -> float:
    checked = [_finite_positive(value, "timing value") for value in values]
    median = statistics.median(checked)
    mad = statistics.median(abs(value - median) for value in checked)
    return 1.4826 * mad / median


def _required_keys(value: Mapping[str, Any], required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(value))
    if missing:
        raise PerfContractError("{} lacks fields {}".format(label, missing))


def validate_rank_payload(payload: Mapping[str, Any], expected_arm: str) -> None:
    definition = arm_definition(expected_arm)
    _required_keys(
        payload,
        {
            "schema_version",
            "benchmark_only",
            "scientific_use_forbidden",
            "status",
            "arm",
            "arm_id",
            "rank",
            "world_size",
            "warmup_steps",
            "measured_steps",
            "timing_block_steps",
            "source_hashes",
            "blocks",
            "components_seconds",
            "reducer_lifecycle",
            "phase_events",
            "cuda_max_memory_allocated",
            "cuda_max_memory_reserved",
            "cuda_total_memory_bytes",
        },
        "rank payload",
    )
    if payload["schema_version"] != RANK_SCHEMA_VERSION:
        raise PerfContractError("rank payload schema differs")
    if payload["benchmark_only"] is not True or payload["scientific_use_forbidden"] is not True:
        raise PerfContractError("rank payload is not explicitly benchmark-only")
    if payload["status"] != "passed":
        raise PerfContractError("rank payload did not pass")
    if payload["arm"] != definition["name"] or payload["arm_id"] != definition["arm_id"]:
        raise PerfContractError("rank payload arm differs")
    rank = payload["rank"]
    if type(rank) is not int or not 0 <= rank < WORLD_SIZE:
        raise PerfContractError("rank payload rank is outside 0..7")
    if payload["world_size"] != WORLD_SIZE:
        raise PerfContractError("rank payload world size differs")
    if (
        payload["warmup_steps"] != WARMUP_STEPS
        or payload["measured_steps"] != MEASURED_STEPS
        or payload["timing_block_steps"] != BLOCK_STEPS
    ):
        raise PerfContractError("rank payload timing contract differs")
    source_hashes = payload["source_hashes"]
    if not isinstance(source_hashes, dict) or source_hashes.get("train_stage1.py") != TRAIN_STAGE1_SHA256:
        raise PerfContractError("rank payload is not bound to the frozen train runner")
    blocks = payload["blocks"]
    if not isinstance(blocks, list) or len(blocks) != EXPECTED_BLOCKS:
        raise PerfContractError("rank payload must contain six timing blocks")
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise PerfContractError("timing block is not an object")
        expected_start = WARMUP_STEPS + index * BLOCK_STEPS
        expected_end = expected_start + BLOCK_STEPS
        if (
            block.get("block_index") != index
            or block.get("start_completed_step") != expected_start
            or block.get("end_completed_step") != expected_end
            or block.get("steps") != BLOCK_STEPS
        ):
            raise PerfContractError("timing block progress differs at index {}".format(index))
        _finite_positive(block.get("seconds"), "timing block seconds")
    components = payload["components_seconds"]
    if not isinstance(components, dict) or set(components) != set(COMPONENT_NAMES):
        raise PerfContractError("rank component timing field set differs")
    for name, value in components.items():
        if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) < 0.0:
            raise PerfContractError("component {} is negative or non-finite".format(name))
    if definition["audit_mode"] == "min" and any(float(value) != 0.0 for value in components.values()):
        raise PerfContractError("min arm recorded production telemetry components")
    if definition["audit_mode"] == "full" and any(
        float(value) <= 0.0 for value in components.values()
    ):
        raise PerfContractError("full arm lacks a measured telemetry component")
    lifecycle = payload["reducer_lifecycle"]
    if not isinstance(lifecycle, list) or not lifecycle:
        raise PerfContractError("rank payload has no reducer lifecycle evidence")
    lifecycle_steps = {
        item.get("completed_step") for item in lifecycle if isinstance(item, dict)
    }
    if not {0, WARMUP_STEPS, TOTAL_STEPS}.issubset(lifecycle_steps):
        raise PerfContractError("rank reducer lifecycle lacks construction/warmup/final anchors")
    if definition["find_unused_parameters"]:
        if any(item.get("has_rebuilt_buckets") is not False for item in lifecycle):
            raise PerfContractError("production True policy rebuilt buckets")
    phase_events = payload["phase_events"]
    if not isinstance(phase_events, list):
        raise PerfContractError("rank phase events are not a list")
    save_steps = {
        item.get("completed_step")
        for item in phase_events
        if isinstance(item, dict) and item.get("event") == "save_checkpoint"
    }
    if save_steps != {0, TOTAL_STEPS}:
        raise PerfContractError("rank phase events do not cover initial/final checkpoints")
    allocated = int(payload["cuda_max_memory_allocated"])
    reserved = int(payload["cuda_max_memory_reserved"])
    total = int(payload["cuda_total_memory_bytes"])
    if not 0 <= allocated <= reserved <= total:
        raise PerfContractError("CUDA memory accounting is inconsistent")


def arm_rank_payloads(arm_dir: Path) -> List[Dict[str, Any]]:
    manifest = read_json(arm_dir / "benchmark_manifest.json")
    arm_name = manifest.get("arm")
    definition = arm_definition(str(arm_name))
    ranks_dir = arm_dir / "ranks"
    if ranks_dir.is_symlink() or not ranks_dir.is_dir():
        raise PerfContractError("arm ranks directory is missing")
    observed = {item.name for item in ranks_dir.iterdir()}
    expected = {"rank-{:02d}.json".format(rank) for rank in range(WORLD_SIZE)}
    if observed != expected:
        raise PerfContractError("rank evidence members differ")
    payloads = [read_json(ranks_dir / name) for name in sorted(expected)]
    for expected_rank, payload in enumerate(payloads):
        validate_rank_payload(payload, definition["name"])
        if payload["rank"] != expected_rank:
            raise PerfContractError("rank evidence filename/payload mismatch")
    return payloads


def summarize_arm_directory(arm_dir: Path) -> Dict[str, Any]:
    manifest = read_json(arm_dir / "benchmark_manifest.json")
    definition = arm_definition(str(manifest.get("arm")))
    payloads = arm_rank_payloads(arm_dir)
    global_blocks: List[float] = []
    for block_index in range(EXPECTED_BLOCKS):
        global_seconds = max(
            float(payload["blocks"][block_index]["seconds"]) for payload in payloads
        )
        global_blocks.append(global_seconds / BLOCK_STEPS)
    median_seconds = statistics.median(global_blocks)
    p90_seconds = quantile(global_blocks, 0.90)
    max_reserved = max(int(payload["cuda_max_memory_reserved"]) for payload in payloads)
    min_total = min(int(payload["cuda_total_memory_bytes"]) for payload in payloads)
    memory_margin = (min_total - max_reserved) / min_total
    component_totals = {
        name: max(float(payload["components_seconds"][name]) for payload in payloads)
        for name in COMPONENT_NAMES
    }
    return {
        "arm_id": definition["arm_id"],
        "arm": definition["name"],
        "find_unused_parameters": definition["find_unused_parameters"],
        "audit_mode": definition["audit_mode"],
        "global_block_seconds_per_step": global_blocks,
        "median_seconds_per_step": median_seconds,
        "p90_seconds_per_step": p90_seconds,
        "robust_cv": robust_cv(global_blocks),
        "steps_per_hour": 3600.0 / median_seconds,
        "effective_samples_per_second": 64.0 / median_seconds,
        "peak_reserved_memory_margin": memory_margin,
        "components_seconds_rank_max": component_totals,
    }


def _seal_payload_identities(root: Path, relative_paths: Sequence[str]) -> List[Dict[str, Any]]:
    identities = []
    for relative in sorted(relative_paths):
        path = root / relative
        identities.append(file_identity(path, relative))
    return identities


def write_seal(root: Path, *, scope: str, relative_paths: Sequence[str]) -> Path:
    seal_path = root / "ARTIFACT_SEAL.json"
    if seal_path.exists() or seal_path.is_symlink():
        raise FileExistsError("seal already exists: {}".format(seal_path))
    payload = {
        "schema_version": SEAL_SCHEMA_VERSION,
        "benchmark_only": True,
        "scientific_use_forbidden": True,
        "scope": scope,
        "payloads": _seal_payload_identities(root, relative_paths),
    }
    payload["payload_set_sha256"] = canonical_json_sha256(payload["payloads"])
    write_json_exclusive(seal_path, payload)
    return seal_path


def verify_seal(root: Path, expected_scope: str) -> Dict[str, Any]:
    seal = read_json(root / "ARTIFACT_SEAL.json")
    if (
        seal.get("schema_version") != SEAL_SCHEMA_VERSION
        or seal.get("scope") != expected_scope
        or seal.get("benchmark_only") is not True
        or seal.get("scientific_use_forbidden") is not True
    ):
        raise PerfContractError("artifact seal contract differs")
    payloads = seal.get("payloads")
    if not isinstance(payloads, list) or not payloads:
        raise PerfContractError("artifact seal payload list is empty")
    if canonical_json_sha256(payloads) != seal.get("payload_set_sha256"):
        raise PerfContractError("artifact seal payload-set digest differs")
    paths: List[str] = []
    for entry in payloads:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "size_bytes"}:
            raise PerfContractError("artifact seal payload identity is malformed")
        relative = entry["path"]
        if not isinstance(relative, str) or relative.startswith("/") or ".." in Path(relative).parts:
            raise PerfContractError("artifact seal contains unsafe relative path")
        paths.append(relative)
        observed = file_identity(root / relative, relative)
        if observed != entry:
            raise PerfContractError("sealed payload changed: {}".format(relative))
    if len(set(paths)) != len(paths):
        raise PerfContractError("artifact seal contains duplicate paths")
    return seal


def arm_seal_paths() -> List[str]:
    paths = [
        "benchmark_manifest.json",
        "STATUS.json",
        "console.log",
        "formal_run/run_manifest.json",
        "formal_run/SEALED.json",
        "formal_run/DONE.json",
    ]
    paths.extend("ranks/rank-{:02d}.json".format(rank) for rank in range(WORLD_SIZE))
    return paths


def verify_arm_directory(arm_dir: Path) -> Dict[str, Any]:
    expected_members = {
        "ARTIFACT_SEAL.json",
        "STATUS.json",
        "benchmark_manifest.json",
        "console.log",
        "formal_run",
        "ranks",
    }
    observed_members = {item.name for item in arm_dir.iterdir()}
    if observed_members != expected_members:
        raise PerfContractError("arm top-level members differ")
    manifest = read_json(arm_dir / "benchmark_manifest.json")
    definition = arm_definition(str(manifest.get("arm")))
    if (
        manifest.get("schema_version") != BENCHMARK_SCHEMA_VERSION
        or manifest.get("benchmark_only") is not True
        or manifest.get("scientific_use_forbidden") is not True
    ):
        raise PerfContractError("benchmark manifest contract differs")
    if manifest.get("arm_id") != definition["arm_id"]:
        raise PerfContractError("benchmark manifest arm ID differs")
    source_hashes = manifest.get("source_hashes")
    if not isinstance(source_hashes, dict) or source_hashes.get("train_stage1.py") != TRAIN_STAGE1_SHA256:
        raise PerfContractError("benchmark manifest source binding differs")
    status = read_json(arm_dir / "STATUS.json")
    if (
        status.get("schema_version") != ARM_STATUS_SCHEMA_VERSION
        or status.get("status") != "passed"
        or status.get("benchmark_only") is not True
        or status.get("arm") != definition["name"]
    ):
        raise PerfContractError("arm status did not pass")
    formal_manifest = read_json(arm_dir / "formal_run" / "run_manifest.json")
    formal_seal = read_json(arm_dir / "formal_run" / "SEALED.json")
    formal_done = read_json(arm_dir / "formal_run" / "DONE.json")
    if (
        formal_manifest.get("schema_version") != PERF_RUN_SCHEMA_VERSION
        or formal_manifest.get("benchmark_only") is not True
        or formal_manifest.get("scientific_use_forbidden") is not True
        or formal_seal.get("schema_version") != PERF_SEAL_SCHEMA_VERSION
        or formal_seal.get("status") != "sealed"
        or formal_done.get("schema_version") != PERF_DONE_SCHEMA_VERSION
        or formal_done.get("status") != "complete"
    ):
        raise PerfContractError("derived run is not explicitly benchmark-only")
    if formal_done.get("SEALED.json_sha256") != sha256_file(
        arm_dir / "formal_run" / "SEALED.json"
    ):
        raise PerfContractError("benchmark DONE does not bind SEALED.json")
    if formal_done.get("run_manifest_sha256") != sha256_file(
        arm_dir / "formal_run" / "run_manifest.json"
    ):
        raise PerfContractError("benchmark DONE does not bind run_manifest.json")
    if formal_manifest.get("schema_version") == FORMAL_RUN_SCHEMA_VERSION:
        raise PerfContractError("benchmark run could be mistaken for a formal Stage-1 run")
    summary = summarize_arm_directory(arm_dir)
    seal = verify_seal(arm_dir, "arm")
    if {item["path"] for item in seal["payloads"]} != set(arm_seal_paths()):
        raise PerfContractError("arm seal payload path set differs")
    return {
        "schema_version": "ptc-opd-stage1-perf-verification-v1",
        "status": "verified",
        "scope": "arm",
        "arm": definition["name"],
        "formal_stage1_consumer_must_reject": True,
        "summary": summary,
        "artifact_seal_sha256": sha256_file(arm_dir / "ARTIFACT_SEAL.json"),
    }


def _require_recorded_input_path(
    value: Any, expected: Path, label: str, *, kind: str
) -> None:
    if not isinstance(value, str) or value != str(expected):
        raise PerfContractError("{} path differs from the live input".format(label))
    observed = _canonical_input_path(Path(value), label, kind=kind)
    if observed != expected:
        raise PerfContractError("{} does not resolve to the live input".format(label))


def verify_arm_input_bindings(
    arm_dir: Path, live_input_identity: Mapping[str, Any]
) -> Dict[str, Any]:
    """Bind one sealed arm's producer and formal-run records to live inputs."""

    verify_arm_directory(arm_dir)
    paths = live_input_identity.get("paths")
    checkpoint = live_input_identity.get("checkpoint_identity")
    source = live_input_identity.get("audiocraft_source_identity")
    cfg = live_input_identity.get("cfg_decision_identity")
    if not all(isinstance(value, dict) for value in (paths, checkpoint, source, cfg)):
        raise PerfContractError("live benchmark input identity is malformed")

    benchmark = read_json(arm_dir / "benchmark_manifest.json")
    benchmark_inputs = benchmark.get("inputs")
    expected_benchmark_keys = {
        "manifest",
        "student_checkpoint",
        "teacher_checkpoint",
        "audiocraft_root",
        "cfg_scale_decision_dir",
    }
    if not isinstance(benchmark_inputs, dict) or set(benchmark_inputs) != expected_benchmark_keys:
        raise PerfContractError("benchmark manifest input field set differs")
    path_specs = (
        ("manifest", "train_manifest", "file"),
        ("student_checkpoint", "musicgen_small_dir", "tree"),
        ("teacher_checkpoint", "musicgen_small_dir", "tree"),
        ("audiocraft_root", "audiocraft_dir", "tree"),
        ("cfg_scale_decision_dir", "small_cfg_dir", "tree"),
    )
    for recorded, expected_name, kind in path_specs:
        _require_recorded_input_path(
            benchmark_inputs.get(recorded),
            Path(str(paths[expected_name])),
            "benchmark manifest {}".format(recorded),
            kind=kind,
        )

    formal = read_json(arm_dir / "formal_run" / "run_manifest.json")
    config = formal.get("config")
    if not isinstance(config, dict):
        raise PerfContractError("benchmark formal run config is absent")
    for recorded, expected_name, kind in path_specs:
        _require_recorded_input_path(
            config.get(recorded),
            Path(str(paths[expected_name])),
            "benchmark formal config {}".format(recorded),
            kind=kind,
        )

    expected_formal_scalars = {
        "manifest_sha256": live_input_identity.get("manifest_sha256"),
        "student_checkpoint_sha256": checkpoint.get("checkpoint_sha256"),
        "teacher_checkpoint_sha256": checkpoint.get("checkpoint_sha256"),
        "student_state_dict_sha256": checkpoint.get("state_dict_sha256"),
        "teacher_state_dict_sha256": checkpoint.get("state_dict_sha256"),
        "audiocraft_lm_sha256": live_input_identity.get("audiocraft_lm_sha256"),
    }
    for field, expected in expected_formal_scalars.items():
        if formal.get(field) != expected:
            raise PerfContractError(
                "benchmark formal input identity differs at {}".format(field)
            )
    if formal.get("audiocraft_source_identity") != source:
        raise PerfContractError("benchmark formal AudioCraft source identity differs")
    if formal.get("cfg_scale_decision") != cfg:
        raise PerfContractError("benchmark formal CFG decision identity differs")

    generation = cfg.get("generation_identity")
    if not isinstance(generation, dict):
        raise PerfContractError("live CFG generation identity is malformed")
    expected_cfg_binding = {
        "checkpoint_sha256": checkpoint.get("checkpoint_sha256"),
        "state_dict_sha256": checkpoint.get("state_dict_sha256"),
        "audiocraft_source_sha256": source.get("tree_sha256"),
    }
    if formal.get("cfg_generation_binding") != expected_cfg_binding:
        raise PerfContractError("benchmark formal CFG generation binding differs")
    expected_config_scalars = {
        "cfg_scale_decision_file_sha256": cfg.get("decision_file_sha256"),
        "cfg_scale_decision_payload_sha256": cfg.get("decision_payload_sha256"),
        "cfg_scale_scientific_config_sha256": cfg.get("scientific_config_sha256"),
        "cfg_generation_checkpoint_sha256": checkpoint.get("checkpoint_sha256"),
        "cfg_generation_state_dict_sha256": checkpoint.get("state_dict_sha256"),
        "cfg_generation_audiocraft_source_sha256": source.get("tree_sha256"),
        "cfg_generation_loaded_t5_identity_sha256": generation.get(
            "loaded_t5_identity_sha256"
        ),
        "teacher_cfg_scale": cfg.get("selected_cfg_scale"),
    }
    for field, expected in expected_config_scalars.items():
        if config.get(field) != expected:
            raise PerfContractError(
                "benchmark formal config identity differs at {}".format(field)
            )
    loaded_t5 = formal.get("loaded_t5_identity")
    if (
        not isinstance(loaded_t5, dict)
        or loaded_t5.get("identity_sha256")
        != generation.get("loaded_t5_identity_sha256")
    ):
        raise PerfContractError("benchmark formal loaded-T5 identity differs")
    return {
        "benchmark_manifest_sha256": sha256_file(
            arm_dir / "benchmark_manifest.json"
        ),
        "formal_run_manifest_sha256": sha256_file(
            arm_dir / "formal_run" / "run_manifest.json"
        ),
    }


def node_arm_directories(node_dir: Path) -> Dict[str, Path]:
    config = read_json(node_dir / "NODE_CONFIG.json")
    node_label = config.get("node_label")
    sequence = schedule_for_node(str(node_label))
    observed: Dict[str, Path] = {}
    for period, arm_id in enumerate(sequence, start=1):
        name = ARM_DEFINITIONS[arm_id]["name"]
        path = node_dir / "period-{:02d}.{}".format(period, name)
        if not path.is_dir() or path.is_symlink():
            raise PerfContractError("node arm directory is missing: {}".format(path))
        observed[name] = path
    if set(observed) != set(ARM_NAME_TO_ID):
        raise PerfContractError("node does not contain all four arm names")
    return observed


def node_seal_paths(node_dir: Path) -> List[str]:
    paths = ["NODE_CONFIG.json", "STATUS.json"]
    for arm_dir in node_arm_directories(node_dir).values():
        paths.append(arm_dir.relative_to(node_dir).as_posix() + "/ARTIFACT_SEAL.json")
    return paths


def verify_node_directory(node_dir: Path) -> Dict[str, Any]:
    config = read_json(node_dir / "NODE_CONFIG.json")
    node_label = str(config.get("node_label"))
    expected_sequence = list(schedule_for_node(node_label))
    if (
        config.get("schema_version") != BENCHMARK_SCHEMA_VERSION
        or config.get("benchmark_only") is not True
        or config.get("arm_sequence") != expected_sequence
    ):
        raise PerfContractError("node benchmark config differs")
    status = read_json(node_dir / "STATUS.json")
    if (
        status.get("schema_version") != NODE_STATUS_SCHEMA_VERSION
        or status.get("status") != "passed"
        or status.get("node_label") != node_label
    ):
        raise PerfContractError("node benchmark status did not pass")
    sequence = schedule_for_node(node_label)
    expected_members = {"ARTIFACT_SEAL.json", "NODE_CONFIG.json", "STATUS.json"}
    expected_members.update(
        "period-{:02d}.{}".format(period, ARM_DEFINITIONS[arm_id]["name"])
        for period, arm_id in enumerate(sequence, start=1)
    )
    if {item.name for item in node_dir.iterdir()} != expected_members:
        raise PerfContractError("node top-level members differ")
    arms = node_arm_directories(node_dir)
    summaries = {name: verify_arm_directory(path)["summary"] for name, path in arms.items()}
    seal = verify_seal(node_dir, "node")
    if {item["path"] for item in seal["payloads"]} != set(node_seal_paths(node_dir)):
        raise PerfContractError("node seal payload path set differs")
    return {
        "schema_version": "ptc-opd-stage1-perf-verification-v1",
        "status": "verified",
        "scope": "node",
        "node_label": node_label,
        "arm_summaries": summaries,
        "artifact_seal_sha256": sha256_file(node_dir / "ARTIFACT_SEAL.json"),
    }


def verify_node_input_bindings(
    node_dir: Path,
    *,
    train_manifest: Path,
    small_cfg_dir: Path,
    audiocraft_dir: Path,
    musicgen_small_dir: Path,
) -> Dict[str, Any]:
    """Verify all four arm/formal-run records against one live input quartet."""

    result = verify_node_directory(node_dir)
    live = build_live_input_identity(
        train_manifest=train_manifest,
        small_cfg_dir=small_cfg_dir,
        audiocraft_dir=audiocraft_dir,
        musicgen_small_dir=musicgen_small_dir,
    )
    arms = node_arm_directories(node_dir)
    bindings = {
        name: verify_arm_input_bindings(path, live)
        for name, path in sorted(arms.items())
    }
    if set(bindings) != set(ARM_NAME_TO_ID):
        raise PerfContractError("node input verifier did not cover all four arms")
    result["input_binding_schema_version"] = PERF_INPUT_BINDING_SCHEMA_VERSION
    result["verified_input_arm_count"] = len(bindings)
    result["live_input_identity"] = live
    result["arm_input_bindings"] = bindings
    return result


def _paired_ratio(node_summaries: Mapping[str, Mapping[str, Any]], numerator: str, denominator: str) -> List[float]:
    return [
        float(summary[numerator]["median_seconds_per_step"])
        / float(summary[denominator]["median_seconds_per_step"])
        for _, summary in sorted(node_summaries.items())
    ]


def aggregate_node_directories(node_dirs: Sequence[Path]) -> Dict[str, Any]:
    if len(node_dirs) != 4:
        raise PerfContractError("aggregation requires exactly four node directories")
    validated: Dict[str, Dict[str, Any]] = {}
    for path in node_dirs:
        result = verify_node_directory(path)
        label = result["node_label"]
        if label in validated:
            raise PerfContractError("duplicate node label in aggregation")
        validated[label] = result["arm_summaries"]
    if set(validated) != set(WILLIAMS_SCHEDULE):
        raise PerfContractError("aggregation must cover node-0..3 exactly")

    ratio_specs = {
        "ddp_cost_under_full_audit": ("true_full", "false_full"),
        "ddp_cost_under_min_audit": ("true_min", "false_min"),
        "audit_cost_under_true": ("true_full", "true_min"),
        "audit_cost_under_false": ("false_full", "false_min"),
    }
    ratios: Dict[str, Any] = {}
    for name, (numerator, denominator) in ratio_specs.items():
        values = _paired_ratio(validated, numerator, denominator)
        ratios[name] = {
            "numerator": numerator,
            "denominator": denominator,
            "per_node": {
                node: value for node, value in zip(sorted(validated), values)
            },
            "median_ratio": statistics.median(values),
            "min_ratio": min(values),
            "max_ratio": max(values),
            "median_percent_overhead": 100.0 * (statistics.median(values) - 1.0),
        }
    interaction_values = [
        ratios["ddp_cost_under_full_audit"]["per_node"][node]
        / ratios["ddp_cost_under_min_audit"]["per_node"][node]
        for node in sorted(validated)
    ]
    ratios["ddp_audit_interaction"] = {
        "per_node": {
            node: value for node, value in zip(sorted(validated), interaction_values)
        },
        "median_ratio": statistics.median(interaction_values),
        "min_ratio": min(interaction_values),
        "max_ratio": max(interaction_values),
    }

    true_full = [validated[node]["true_full"] for node in sorted(validated)]
    production_median = statistics.median(
        item["median_seconds_per_step"] for item in true_full
    )
    production_p90 = max(item["p90_seconds_per_step"] for item in true_full)
    minimum_memory_margin = min(item["peak_reserved_memory_margin"] for item in true_full)
    max_robust_cv = max(
        item["robust_cv"]
        for summary in validated.values()
        for item in summary.values()
    )
    hard_gate_passed = minimum_memory_margin >= 0.05
    measurement_quality = (
        "complete" if max_robust_cv <= 0.10 else "inconclusive_extend_measurement"
    )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "benchmark_only": True,
        "scientific_use_forbidden": True,
        "node_count": 4,
        "node_labels": sorted(validated),
        "warmup_steps": WARMUP_STEPS,
        "measured_steps_per_arm_per_node": MEASURED_STEPS,
        "timing_block_steps": BLOCK_STEPS,
        "ratios": ratios,
        "production_true_full": {
            "median_seconds_per_step_across_nodes": production_median,
            "conservative_p90_seconds_per_step": production_p90,
            "steps_per_hour": 3600.0 / production_median,
            "effective_samples_per_second": 64.0 / production_median,
            "projected_500_step_training_seconds_excluding_checkpoints": 500.0 * production_median,
            "projected_1000_step_training_seconds_excluding_checkpoints": 1000.0 * production_median,
            "minimum_peak_reserved_memory_margin": minimum_memory_margin,
        },
        "hard_resource_gate_passed": hard_gate_passed,
        "measurement_quality": measurement_quality,
        "max_arm_robust_cv": max_robust_cv,
        "performance_result_role": "informational_only",
        "production_policy_decision": "retain_true_full",
        "false_or_min_outputs_authorized_for_science": False,
        "pilot_policy_change_authorized": False,
    }


def write_summary_csv(path: Path, summary: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "comparison",
                "numerator",
                "denominator",
                "median_ratio",
                "min_ratio",
                "max_ratio",
                "median_percent_overhead",
            ]
        )
        for name, payload in summary["ratios"].items():
            writer.writerow(
                [
                    name,
                    payload.get("numerator", ""),
                    payload.get("denominator", ""),
                    payload["median_ratio"],
                    payload["min_ratio"],
                    payload["max_ratio"],
                    payload.get("median_percent_overhead", ""),
                ]
            )


validate_williams_schedule()


__all__ = [
    "ARM_DEFINITIONS",
    "ARM_NAME_TO_ID",
    "ARM_STATUS_SCHEMA_VERSION",
    "BENCHMARK_SCHEMA_VERSION",
    "BLOCK_STEPS",
    "COMPONENT_NAMES",
    "EXPECTED_BLOCKS",
    "FORMAL_RUN_SCHEMA_VERSION",
    "MEASURED_STEPS",
    "NODE_STATUS_SCHEMA_VERSION",
    "PERF_ATTEMPT_SCHEMA_VERSION",
    "PERF_CHECKPOINT_SCHEMA_VERSION",
    "PERF_DONE_SCHEMA_VERSION",
    "PERF_REDUCER_SCHEMA_VERSION",
    "PERF_RUN_SCHEMA_VERSION",
    "PERF_SEAL_SCHEMA_VERSION",
    "PerfContractError",
    "RANK_SCHEMA_VERSION",
    "SEAL_SCHEMA_VERSION",
    "SUMMARY_SCHEMA_VERSION",
    "TOTAL_STEPS",
    "TRAIN_STAGE1_SHA256",
    "WARMUP_STEPS",
    "WILLIAMS_SCHEDULE",
    "WORLD_SIZE",
    "aggregate_node_directories",
    "arm_definition",
    "arm_rank_payloads",
    "arm_seal_paths",
    "canonical_json_sha256",
    "file_identity",
    "node_arm_directories",
    "node_seal_paths",
    "quantile",
    "read_json",
    "robust_cv",
    "schedule_for_node",
    "sha256_file",
    "summarize_arm_directory",
    "validate_rank_payload",
    "validate_williams_schedule",
    "verify_arm_directory",
    "verify_node_directory",
    "verify_seal",
    "write_json_exclusive",
    "write_seal",
    "write_summary_csv",
]
