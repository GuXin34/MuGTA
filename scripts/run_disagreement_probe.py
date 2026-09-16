#!/usr/bin/env python3
"""Run and summarize the frozen Phase-A2/A3 disagreement probe.

``probe`` is the real AudioCraft/GPU path and is intentionally strict.  It
requires the pinned no-CFG overlay, a local checkpoint, an immutable CFG-scale
decision, the sealed A1 codebook-prior artifact, exactly 256 development-probe
prompts proven to be an exact subset of the 300-prompt development manifest,
and exactly the two frozen rollout seeds.  It writes one sealed directory.

``summarize`` is model-independent.  It consumes only a sealed probe directory,
streams the scalar map, bootstraps at the prompt level (both rollouts stay
nested), and atomically emits one sealed CSV/NPZ/JSON directory.

This script does not support a test manifest and refuses output overwrite.
Every final artifact is installed atomically only after its payload is closed.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import hashlib
import importlib
import inspect
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch


WORKPACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKPACK_ROOT / "src"))

from ptc_opd.audiocraft_adapter import batch_condition_tensors  # noqa: E402
from ptc_opd.cfg_decision import (  # noqa: E402
    EXPECTED_PRECISION_IDENTITY,
    verify_cfg_scale_decision,
)
from ptc_opd.codec_prior_artifact import (  # noqa: E402
    load_codec_prior_artifact,
    verify_local_codec_snapshot,
)
from ptc_opd.musicgen_contract import (  # noqa: E402
    PRIMARY_CARD,
    PRIMARY_FRAME_RATE,
    PRIMARY_FRAMES,
    strict_validate_musicgen_batch,
    strict_validate_musicgen_model,
)
from ptc_opd.phenomena import (  # noqa: E402
    CELL_SCHEMA_VERSION,
    PRIMARY_BOOTSTRAP_REPLICATES,
    PRIMARY_BOOTSTRAP_SEED,
    PRIMARY_ROLLOUT_SEEDS,
    SUMMARY_SCHEMA_VERSION,
    PhenomenonAccumulator,
    audit_topk_records,
    cell_records_from_logits,
)
from ptc_opd.reproducibility import (  # noqa: E402
    audiocraft_source_identity,
    loaded_t5_identity,
    require_offline_hf_environment,
)
from ptc_opd.train_utils import hash_module_state  # noqa: E402


PROBE_MANIFEST_BASENAME = "phenomenon_probe.dev.jsonl"
DEV_MANIFEST_BASENAME = "dev.full.jsonl"
EXPECTED_PROMPTS = 256
EXPECTED_DEV_PROMPTS = 300
EXPECTED_AUDIT_PROMPTS = 16
EXPECTED_AUDIT_TOPK = 32
EXPECTED_VALID_CELLS_PER_SEQUENCE = 500 + 499 + 498 + 497
DEFAULT_DURATION_SECONDS = 10.0
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = 250
DEFAULT_TOP_P = 0.0
JSON_GZIP_MTIME = 0
PINNED_AUDIOCRAFT_BASE_COMMIT = "896ec7c47f5e5d1e5aa1e4b260c4405328bf009d"
MUSICGEN_SMALL_ARCHITECTURE = {
    "num_codebooks": 4,
    "cardinality": 2048,
    "transformer_dim": 1024,
    "transformer_layers": 24,
    "transformer_heads": 16,
}

PROBE_CONFIG_SCHEMA_VERSION = "ptc-opd-disagreement-probe-config-v2"
PROBE_RUN_SCHEMA_VERSION = "ptc-opd-disagreement-probe-run-v2"
PROBE_SEAL_SCHEMA_VERSION = "ptc-opd-disagreement-probe-seal-v1"
SUMMARY_CONFIG_SCHEMA_VERSION = "ptc-opd-disagreement-summary-config-v2"
SUMMARY_RUN_SCHEMA_VERSION = "ptc-opd-disagreement-summary-run-v2"
SUMMARY_SEAL_SCHEMA_VERSION = "ptc-opd-disagreement-summary-seal-v1"

PROBE_CELLS_FILENAME = "disagreement_cells.jsonl.gz"
PROBE_AUDIT_FILENAME = "disagreement_top32_audit.jsonl.gz"
PROBE_METADATA_FILENAME = "disagreement_probe_metadata.json"
PROBE_SEAL_FILENAME = "artifact_seal.json"
SUMMARY_CSV_FILENAME = "disagreement_summary.csv"
SUMMARY_NPZ_FILENAME = "disagreement_bootstrap.npz"
SUMMARY_JSON_FILENAME = "disagreement_summary.json"
SUMMARY_SEAL_FILENAME = "artifact_seal.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    """Hash relative names, file sizes, and bytes of one local snapshot."""

    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError("checkpoint directory contains no files: {}".format(path))
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = item.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        return sha256_directory(path)
    raise FileNotFoundError(path)


def git_head(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(
            "AudioCraft checkout has no Git identity: {}".format(
                completed.stderr.strip() or path
            )
        )
    return completed.stdout.strip()


def verify_checkpoint_snapshot(path: Path) -> Dict[str, str]:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("formal probe checkpoint must be a local snapshot directory")
    # A Hugging Face cache snapshot commonly contains relative symlinks into
    # ``blobs/``.  Those links are not portable when the workpack is copied to
    # another machine, so the formal artifact must be fully dereferenced.
    for member in path.rglob("*"):
        if member.is_symlink():
            raise ValueError(
                "formal probe checkpoint must be fully dereferenced; symlink found: {}".format(
                    member.relative_to(path)
                )
            )
    lm_payload = path / "state_dict.bin"
    compression_payload = path / "compression_state_dict.bin"
    for payload in (lm_payload, compression_payload):
        if payload.is_symlink() or not payload.is_file():
            raise ValueError("checkpoint snapshot is missing regular {}".format(payload.name))
    return {
        "checkpoint_sha256": sha256_directory(path),
        "state_dict_sha256": sha256_file(lm_payload),
        "compression_state_dict_sha256": sha256_file(compression_payload),
    }


def verify_audiocraft_source(path: Path) -> Dict[str, str]:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("AudioCraft source root must be a regular non-symlink directory")
    lm_path = path / "audiocraft" / "models" / "lm.py"
    if lm_path.is_symlink() or not lm_path.is_file():
        raise ValueError("AudioCraft source has no regular audiocraft/models/lm.py")
    source = lm_path.read_text(encoding="utf-8")
    required_overlay_fragments = (
        "use_cfg: bool = True",
        "condition_tensors:",
        "elif condition_tensors is not None:",
        "CFG condition_tensors batch must be even",
    )
    if any(fragment not in source for fragment in required_overlay_fragments):
        raise ValueError("AudioCraft explicit no-CFG/precomputed-CFG overlay is absent")
    commit = git_head(path)
    if commit != PINNED_AUDIOCRAFT_BASE_COMMIT:
        raise ValueError(
            "AudioCraft base commit mismatch: expected {}, got {}".format(
                PINNED_AUDIOCRAFT_BASE_COMMIT, commit
            )
        )
    identity = audiocraft_source_identity(path)
    return {
        "audiocraft_base_commit": commit,
        "audiocraft_source_sha256": str(identity["tree_sha256"]),
        "audiocraft_source_identity_sha256": str(identity["identity_sha256"]),
        "audiocraft_lm_sha256": sha256_file(lm_path),
    }


def stable_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )


def sha256_json(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assert_new_outputs(paths: Sequence[Path]) -> None:
    duplicated = len(set(paths)) != len(paths)
    if duplicated:
        raise ValueError("output paths must be distinct")
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing outputs: {}".format(existing))
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


@contextlib.contextmanager
def staged_output_directory(target: Path) -> Iterator[Path]:
    """Build a complete artifact beside ``target`` and publish by one rename."""

    target = target.resolve()
    if target.exists():
        raise FileExistsError("refusing to overwrite artifact directory: {}".format(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=".{}.partial.".format(target.name), dir=str(target.parent)
        )
    )
    try:
        yield temporary
        if target.exists():
            raise FileExistsError("artifact target appeared during staging: {}".format(target))
        os.replace(str(temporary), str(target))
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def artifact_member(path: Path) -> Dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("artifact member is missing or not regular: {}".format(path))
    return {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_artifact_seal(
    path: Path, *, schema_version: str, status: str, members: Sequence[Path]
) -> None:
    names = [member.name for member in members]
    if len(set(names)) != len(names):
        raise ValueError("artifact seal member basenames must be unique")
    write_json_atomic(
        path,
        {
            "schema_version": schema_version,
            "status": status,
            "members": {member.name: artifact_member(member) for member in members},
        },
    )


def verify_artifact_seal(
    directory: Path,
    *,
    seal_filename: str,
    seal_schema: str,
    expected_status: str,
    expected_members: Sequence[str],
) -> Dict[str, Dict[str, object]]:
    supplied = directory.expanduser()
    if supplied.is_symlink():
        raise ValueError("artifact directory must not be a symlink")
    directory = supplied.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("artifact path must be a regular directory")
    expected_names = set(expected_members) | {seal_filename}
    entries = list(directory.iterdir())
    observed_names = {member.name for member in entries}
    if observed_names != expected_names:
        raise ValueError(
            "artifact directory members differ; missing={}, unexpected={}".format(
                sorted(expected_names - observed_names),
                sorted(observed_names - expected_names),
            )
        )
    for member in entries:
        if member.is_symlink() or not member.is_file():
            raise ValueError(
                "artifact directory contains a non-regular member: {}".format(
                    member.name
                )
            )
    seal_path = directory / seal_filename
    if seal_path.is_symlink() or not seal_path.is_file():
        raise ValueError("artifact seal is missing or not regular")
    with seal_path.open("r", encoding="utf-8") as stream:
        seal = json.load(stream)
    if not isinstance(seal, dict) or seal.get("schema_version") != seal_schema:
        raise ValueError("artifact seal schema mismatch")
    if seal.get("status") != expected_status:
        raise ValueError("artifact seal status mismatch")
    members = seal.get("members")
    if not isinstance(members, dict) or set(members) != set(expected_members):
        raise ValueError("artifact seal member set mismatch")
    normalized: Dict[str, Dict[str, object]] = {}
    for name in expected_members:
        item = members.get(name)
        if not isinstance(item, dict) or set(item) != {"size_bytes", "sha256"}:
            raise ValueError("artifact seal entry mismatch for {}".format(name))
        path = directory / name
        observed = artifact_member(path)
        if item != observed:
            raise ValueError("artifact member size/hash mismatch for {}".format(name))
        normalized[name] = observed
    return normalized


def _load_json_object(path: Path, label: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("{} is missing or not regular".format(label))

    def reject_constant(value: str) -> None:
        raise ValueError("{} contains forbidden non-finite {}".format(label, value))

    def unique_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("{} contains duplicate key {!r}".format(label, key))
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(
                stream,
                object_pairs_hook=unique_object,
                parse_constant=reject_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("{} is not valid UTF-8 JSON".format(label)) from exc
    if not isinstance(value, dict):
        raise ValueError("{} must contain one JSON object".format(label))
    return value


def _require_unique_strings(value: object, label: str) -> List[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("{} must be a non-empty JSON array".format(label))
    resolved: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError("{} must contain non-empty strings".format(label))
        resolved.append(item)
    if len(set(resolved)) != len(resolved):
        raise ValueError("{} must not contain duplicates".format(label))
    return resolved


def _require_exact_int(value: object, expected: int, label: str) -> int:
    if type(value) is not int or value != expected:
        raise ValueError("{} must equal {}".format(label, expected))
    return int(value)


def verify_probe_artifact(
    directory: Path, *, require_primary: bool = True
) -> Dict[str, object]:
    """Rehash and structurally validate one closed A2 probe directory."""

    expected_status = "complete_gpu_probe" if require_primary else "complete_cpu_debug"
    members = verify_artifact_seal(
        directory,
        seal_filename=PROBE_SEAL_FILENAME,
        seal_schema=PROBE_SEAL_SCHEMA_VERSION,
        expected_status=expected_status,
        expected_members=(
            PROBE_CELLS_FILENAME,
            PROBE_AUDIT_FILENAME,
            PROBE_METADATA_FILENAME,
        ),
    )
    resolved = directory.expanduser().resolve(strict=True)
    metadata = _load_json_object(
        resolved / PROBE_METADATA_FILENAME, PROBE_METADATA_FILENAME
    )
    if metadata.get("schema_version") != PROBE_RUN_SCHEMA_VERSION:
        raise ValueError("probe metadata schema mismatch")
    expected_scientific_status = (
        "gpu_probe_complete" if require_primary else "cpu_debug_only"
    )
    if metadata.get("scientific_status") != expected_scientific_status:
        raise ValueError("probe scientific status mismatch")
    if metadata.get("primary_contract_passed") is not require_primary:
        raise ValueError("probe primary-contract marker mismatch")

    config = metadata.get("scientific_config")
    if not isinstance(config, dict):
        raise ValueError("probe metadata has no scientific_config object")
    if config.get("schema_version") != PROBE_CONFIG_SCHEMA_VERSION:
        raise ValueError("probe scientific-config schema mismatch")
    if metadata.get("scientific_config_sha256") != sha256_json(config):
        raise ValueError("probe scientific-config hash mismatch")
    if config.get("primary_contract") is not require_primary:
        raise ValueError("probe scientific-config primary marker mismatch")

    sample_ids = _require_unique_strings(
        metadata.get("probe_sample_ids"), "probe_sample_ids"
    )
    audit_ids = _require_unique_strings(
        metadata.get("audit_sample_ids"), "audit_sample_ids"
    )
    if audit_ids != sample_ids[: len(audit_ids)]:
        raise ValueError("audit IDs must be an ordered prefix of probe IDs")
    seeds = config.get("rollout_seeds")
    if seeds != list(PRIMARY_ROLLOUT_SEEDS):
        raise ValueError("probe rollout seeds differ from the frozen pair")
    records_per_sequence = metadata.get("records_per_sequence")
    if type(records_per_sequence) is not int or records_per_sequence <= 0:
        raise ValueError("probe records_per_sequence must be positive")

    expected_cell_count = len(sample_ids) * len(PRIMARY_ROLLOUT_SEEDS) * records_per_sequence
    expected_audit_count = len(audit_ids) * len(PRIMARY_ROLLOUT_SEEDS) * records_per_sequence
    _require_exact_int(metadata.get("cell_count"), expected_cell_count, "cell_count")
    _require_exact_int(
        metadata.get("expected_cell_count"), expected_cell_count, "expected_cell_count"
    )
    _require_exact_int(
        metadata.get("audit_cell_count"), expected_audit_count, "audit_cell_count"
    )
    _require_exact_int(
        metadata.get("expected_audit_cell_count"),
        expected_audit_count,
        "expected_audit_cell_count",
    )

    outputs = metadata.get("outputs")
    expected_output_names = {PROBE_CELLS_FILENAME, PROBE_AUDIT_FILENAME}
    if not isinstance(outputs, dict) or set(outputs) != expected_output_names:
        raise ValueError("probe output identity set mismatch")
    for name in expected_output_names:
        if outputs.get(name) != members[name]:
            raise ValueError("probe metadata output hash mismatch for {}".format(name))

    if metadata.get("strict_raw_branch_contract_before_intersection") is not True:
        raise ValueError("probe did not attest strict raw-branch validation")
    if metadata.get("student_teacher_same_checkpoint_and_state") is not True:
        raise ValueError("probe did not attest student/teacher state identity")
    if metadata.get("condition_tensors_reused_for_rollout_and_scoring") is not True:
        raise ValueError("probe did not attest conditioner-tensor reuse")

    if require_primary:
        frozen_contract = {
            "probe_is_exact_dev_subset": True,
            "probe_prompts": EXPECTED_PROMPTS,
            "dev_prompts": EXPECTED_DEV_PROMPTS,
            "duration_seconds": DEFAULT_DURATION_SECONDS,
            "token_frames": PRIMARY_FRAMES,
            "temperature": DEFAULT_TEMPERATURE,
            "top_k": DEFAULT_TOP_K,
            "top_p": DEFAULT_TOP_P,
            "batch_size": 2,
            "parameter_dtype": "torch.float32",
            "compute_dtype": "torch.bfloat16",
            "device_type": "cuda",
            "model_id": "facebook/musicgen-small",
            "musicgen_architecture": MUSICGEN_SMALL_ARCHITECTURE,
        }
        for key, expected in frozen_contract.items():
            if config.get(key) != expected:
                raise ValueError(
                    "primary probe scientific_config.{} must equal {!r}".format(
                        key, expected
                    )
                )
        _require_exact_int(
            records_per_sequence,
            EXPECTED_VALID_CELLS_PER_SEQUENCE,
            "records_per_sequence",
        )
        if len(sample_ids) != EXPECTED_PROMPTS:
            raise ValueError("primary probe must contain exactly 256 sample IDs")
        if len(audit_ids) != EXPECTED_AUDIT_PROMPTS:
            raise ValueError("primary audit must contain exactly 16 sample IDs")
        _require_exact_int(
            metadata.get("audit_top_k"), EXPECTED_AUDIT_TOPK, "audit_top_k"
        )
    elif config.get("device_type") != "cpu":
        raise ValueError("CPU-debug artifact cannot claim a non-CPU device")

    # These streaming passes prove every expected sample/seed block is present
    # with its exact number of scalar cells, even after copying the directory.
    for _ in validated_probe_records(
        iter_jsonl_gzip(resolved / PROBE_CELLS_FILENAME),
        expected_sample_ids=sample_ids,
        expected_seeds=PRIMARY_ROLLOUT_SEEDS,
        expected_records_per_sequence=records_per_sequence,
    ):
        pass
    for _ in validated_probe_records(
        iter_jsonl_gzip(resolved / PROBE_AUDIT_FILENAME),
        expected_sample_ids=audit_ids,
        expected_seeds=PRIMARY_ROLLOUT_SEEDS,
        expected_records_per_sequence=records_per_sequence,
        audit_top_k=EXPECTED_AUDIT_TOPK,
    ):
        pass

    return {
        "directory": str(resolved),
        "metadata": metadata,
        "members": members,
        "artifact_seal_sha256": sha256_file(resolved / PROBE_SEAL_FILENAME),
    }


@contextlib.contextmanager
def atomic_binary_writer(path: Path) -> Iterator[io.BufferedWriter]:
    """Yield a same-directory temporary file and atomically install on success."""

    _assert_new_outputs([path])
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary_path), str(path))
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def write_json_atomic(path: Path, value: object) -> None:
    with atomic_binary_writer(path) as stream:
        stream.write(stable_json_bytes(value))


def write_jsonl_gzip_atomic(path: Path, records: Iterable[Mapping[str, object]]) -> int:
    count = 0
    with atomic_binary_writer(path) as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=JSON_GZIP_MTIME) as compressed:
            for record in records:
                compressed.write(
                    (json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n").encode(
                        "utf-8"
                    )
                )
                count += 1
    return count


def iter_jsonl_gzip(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError("{}:{} blank JSONL records are forbidden".format(path, line_number))
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("{}:{} invalid JSON".format(path, line_number)) from exc
            if not isinstance(value, dict):
                raise ValueError("{}:{} must be a JSON object".format(path, line_number))
            yield value


def validated_probe_records(
    records: Iterable[Mapping[str, object]],
    *,
    expected_sample_ids: Sequence[str],
    expected_seeds: Sequence[int],
    expected_records_per_sequence: Optional[int] = None,
    audit_top_k: Optional[int] = None,
) -> Iterator[Mapping[str, object]]:
    """Gate atomic installation on exact prompt x rollout completeness."""

    expected_ids = set(expected_sample_ids)
    if len(expected_ids) != len(expected_sample_ids):
        raise ValueError("expected sample IDs must be unique")
    expected_seed_set = {int(seed) for seed in expected_seeds}
    if len(expected_seed_set) != len(expected_seeds):
        raise ValueError("expected rollout seeds must be unique")
    observed: Dict[str, set] = {sample_id: set() for sample_id in expected_ids}
    record_counts: Dict[Tuple[str, int], int] = {}
    observed_cells: Dict[Tuple[str, int], set] = {
        (sample_id, seed): set()
        for sample_id in expected_ids
        for seed in expected_seed_set
    }
    prompt_hashes: Dict[str, str] = {}
    selection_records: Dict[Tuple[str, int, int], List[Tuple[float, int, bool]]] = {}
    for record in records:
        sample_id = str(record.get("sample_id", ""))
        if type(record.get("rollout_seed")) is not int:
            raise ValueError("probe rollout_seed must be an integer")
        rollout_seed = int(record["rollout_seed"])
        if sample_id not in expected_ids:
            raise ValueError("probe emitted unexpected sample ID {}".format(sample_id))
        if rollout_seed not in expected_seed_set:
            raise ValueError("probe emitted unexpected rollout seed {}".format(rollout_seed))
        observed[sample_id].add(rollout_seed)
        key = (sample_id, rollout_seed)
        record_counts[key] = record_counts.get(key, 0) + 1
        if type(record.get("q")) is not int or type(record.get("t")) is not int:
            raise ValueError("probe q/t must be integers")
        q_index = int(record["q"])
        time_index = int(record["t"])
        if q_index < 0 or time_index < 0:
            raise ValueError("probe q/t must be non-negative")
        cell = (q_index, time_index)
        if cell in observed_cells[key]:
            raise ValueError(
                "duplicate probe cell {} for sample/seed {}".format(cell, key)
            )
        observed_cells[key].add(cell)
        prompt_hash = record.get("prompt_sha256")
        if (
            not isinstance(prompt_hash, str)
            or len(prompt_hash) != 64
            or any(character not in "0123456789abcdef" for character in prompt_hash)
        ):
            raise ValueError("probe prompt_sha256 must be lowercase SHA-256")
        previous_prompt_hash = prompt_hashes.setdefault(sample_id, prompt_hash)
        if previous_prompt_hash != prompt_hash:
            raise ValueError("prompt hash changed across probe records")
        if audit_top_k is None:
            if record.get("schema_version") != CELL_SCHEMA_VERSION:
                raise ValueError("probe cell schema mismatch")
            if set(record) != {
                "schema_version",
                "sample_id",
                "prompt_sha256",
                "rollout_seed",
                "q",
                "t",
                "temporal_decile",
                "js",
                "forward_kl",
                "teacher_entropy",
                "student_entropy",
                "sampled_token_logp_teacher",
                "sampled_token_logp_student",
                "a_q",
                "top50_js",
            }:
                raise ValueError("probe cell fields differ from the frozen schema")
            if type(record.get("temporal_decile")) is not int:
                raise ValueError("probe temporal_decile must be an integer")
            expected_decile = min(9, (10 * time_index) // PRIMARY_FRAMES)
            if int(record["temporal_decile"]) != expected_decile:
                raise ValueError("probe temporal_decile is inconsistent with t")
            numeric_names = (
                "js",
                "forward_kl",
                "teacher_entropy",
                "student_entropy",
                "sampled_token_logp_teacher",
                "sampled_token_logp_student",
                "a_q",
            )
            normalized_numeric: Dict[str, float] = {}
            for name in numeric_names:
                value = record.get(name)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ValueError("probe {} must be finite numeric".format(name))
                normalized_numeric[name] = float(value)
            if normalized_numeric["js"] < 0.0:
                raise ValueError("probe JS must be non-negative")
            if normalized_numeric["forward_kl"] < -1.0e-5:
                raise ValueError("probe forward KL is materially negative")
            if normalized_numeric["a_q"] <= 0.0:
                raise ValueError("probe a_q must be strictly positive")
            if type(record.get("top50_js")) is not bool:
                raise ValueError("probe top50_js must be boolean")
            selection_records.setdefault((sample_id, rollout_seed, q_index), []).append(
                (normalized_numeric["js"], time_index, bool(record["top50_js"]))
            )
        else:
            if record.get("schema_version") != "ptc-opd-disagreement-topk-audit-v1":
                raise ValueError("probe audit schema mismatch")
            if set(record) != {
                "schema_version",
                "sample_id",
                "prompt_sha256",
                "rollout_seed",
                "q",
                "t",
                "sampled_token_id",
                "student_top_token_ids",
                "student_top_logits",
                "teacher_top_token_ids",
                "teacher_top_logits",
            }:
                raise ValueError("probe audit fields differ from the frozen schema")
            sampled_token = record.get("sampled_token_id")
            if type(sampled_token) is not int or not 0 <= sampled_token < PRIMARY_CARD:
                raise ValueError("audit sampled_token_id is outside [0,2048)")
            for name in (
                "student_top_token_ids",
                "student_top_logits",
                "teacher_top_token_ids",
                "teacher_top_logits",
            ):
                values = record.get(name)
                if not isinstance(values, list) or len(values) != audit_top_k:
                    raise ValueError(
                        "audit {} must have length {}".format(name, audit_top_k)
                    )
                if name.endswith("token_ids"):
                    if any(type(value) is not int or not 0 <= value < PRIMARY_CARD for value in values):
                        raise ValueError("audit {} contains an invalid token ID".format(name))
                    if len(set(values)) != audit_top_k:
                        raise ValueError("audit {} contains duplicate token IDs".format(name))
                else:
                    if any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        for value in values
                    ):
                        raise ValueError("audit {} contains a non-finite logit".format(name))
                    if any(float(left) < float(right) for left, right in zip(values, values[1:])):
                        raise ValueError("audit {} must be sorted descending".format(name))
        yield record
    incomplete = {
        sample_id: sorted(expected_seed_set - seeds)
        for sample_id, seeds in observed.items()
        if seeds != expected_seed_set
    }
    empty_pairs = [
        (sample_id, seed)
        for sample_id in sorted(expected_ids)
        for seed in sorted(expected_seed_set)
        if record_counts.get((sample_id, seed), 0) <= 0
    ]
    wrong_counts = []
    if expected_records_per_sequence is not None:
        wrong_counts = [
            (sample_id, seed, record_counts.get((sample_id, seed), 0))
            for sample_id in sorted(expected_ids)
            for seed in sorted(expected_seed_set)
            if record_counts.get((sample_id, seed), 0)
            != expected_records_per_sequence
        ]
    if incomplete or empty_pairs or wrong_counts:
        raise ValueError(
            "incomplete prompt x rollout probe; missing_seeds={}, empty_pairs={}, "
            "wrong_counts={}".format(
                incomplete, empty_pairs, wrong_counts
            )
        )
    if expected_records_per_sequence == EXPECTED_VALID_CELLS_PER_SEQUENCE:
        expected_lattice = {
            (q_index, time_index)
            for q_index in range(4)
            for time_index in range(PRIMARY_FRAMES - q_index)
        }
        wrong_lattices = [
            (sample_id, seed)
            for sample_id in sorted(expected_ids)
            for seed in sorted(expected_seed_set)
            if observed_cells[(sample_id, seed)] != expected_lattice
        ]
        if wrong_lattices:
            raise ValueError(
                "probe cell lattice differs from frozen MusicGen delays for {}".format(
                    wrong_lattices
                )
            )
    if audit_top_k is None:
        wrong_top50: List[Tuple[str, int, int]] = []
        for key, values in selection_records.items():
            ordered = sorted(values, key=lambda item: (-item[0], item[1]))
            selected_count = int(math.ceil(0.5 * len(ordered)))
            expected_selected = {time_index for _, time_index, _ in ordered[:selected_count]}
            observed_selected = {
                time_index for _, time_index, selected in values if selected
            }
            if observed_selected != expected_selected:
                wrong_top50.append(key)
        if wrong_top50:
            raise ValueError(
                "probe top50_js differs from stable (-JS,t) ranking for {}".format(
                    wrong_top50
                )
            )


def _reject_test_manifest(path: Path) -> None:
    basename = path.name.lower()
    if "test" in basename or basename == "test.full.jsonl":
        raise ValueError("test manifests are forbidden for Phase A2/A3")


def load_manifest_records(
    path: Path,
    *,
    expected_basename: str,
    expected_prompts: int,
    require_source_row: bool = True,
) -> List[dict]:
    _reject_test_manifest(path)
    if path.name != expected_basename:
        raise ValueError(
            "manifest basename must be {!r}, got {!r}".format(
                expected_basename, path.name
            )
        )
    records: List[dict] = []
    seen = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError("{}:{} blank manifest records are forbidden".format(path, line_number))
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("{}:{} invalid JSON".format(path, line_number)) from exc
            if not isinstance(record, dict):
                raise ValueError("manifest records must be JSON objects")
            sample_id = record.get("sample_id")
            prompt = record.get("prompt")
            source_row_sha256 = record.get("source_row_sha256")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError("manifest sample_id must be non-empty")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("manifest prompt must be non-empty")
            if require_source_row and (
                not isinstance(source_row_sha256, str)
                or len(source_row_sha256) != 64
                or any(character not in "0123456789abcdef" for character in source_row_sha256)
            ):
                raise ValueError("manifest source_row_sha256 must be lowercase SHA-256")
            if sample_id in seen:
                raise ValueError("duplicate sample_id {}".format(sample_id))
            seen.add(sample_id)
            records.append(
                {
                    "sample_id": sample_id,
                    "prompt": prompt,
                    "source_row_sha256": source_row_sha256,
                }
            )
    if len(records) != expected_prompts:
        raise ValueError(
            "manifest must contain exactly {} prompts, got {}".format(
                expected_prompts, len(records)
            )
        )
    return records


def load_probe_manifest(
    path: Path,
    expected_prompts: int = EXPECTED_PROMPTS,
    *,
    require_source_row: bool = True,
) -> List[dict]:
    return load_manifest_records(
        path,
        expected_basename=PROBE_MANIFEST_BASENAME,
        expected_prompts=expected_prompts,
        require_source_row=require_source_row,
    )


def validate_probe_is_exact_dev_subset(
    probe_path: Path, dev_path: Path
) -> Tuple[List[dict], List[dict]]:
    probe = load_manifest_records(
        probe_path,
        expected_basename=PROBE_MANIFEST_BASENAME,
        expected_prompts=EXPECTED_PROMPTS,
    )
    dev = load_manifest_records(
        dev_path,
        expected_basename=DEV_MANIFEST_BASENAME,
        expected_prompts=EXPECTED_DEV_PROMPTS,
    )
    dev_by_id = {record["sample_id"]: record for record in dev}
    for record in probe:
        sample_id = record["sample_id"]
        if sample_id not in dev_by_id:
            raise ValueError("probe sample is absent from dev manifest: {}".format(sample_id))
        dev_record = dev_by_id[sample_id]
        if record["prompt"].encode("utf-8") != dev_record["prompt"].encode("utf-8"):
            raise ValueError("probe prompt is not byte-exact dev text: {}".format(sample_id))
        if record["source_row_sha256"] != dev_record["source_row_sha256"]:
            raise ValueError("probe source-row identity differs from dev: {}".format(sample_id))
    return probe, dev


def _temporary_import_path(path: Path):
    @contextlib.contextmanager
    def manager():
        existing = sys.modules.get("audiocraft")
        if existing is not None:
            loaded_from = str(getattr(existing, "__file__", ""))
            if not loaded_from.startswith(str(path.resolve())):
                raise RuntimeError(
                    "AudioCraft was already imported from another tree: {}".format(
                        loaded_from
                    )
                )
        sys.path.insert(0, str(path))
        try:
            yield
        finally:
            try:
                sys.path.remove(str(path))
            except ValueError:
                pass

    return manager()


def _assert_audiocraft_overlay(lm: object) -> None:
    generate = getattr(lm, "generate", None)
    if not callable(generate):
        raise TypeError("loaded checkpoint does not expose LMModel.generate")
    parameters = inspect.signature(generate).parameters
    for required in ("use_cfg", "condition_tensors"):
        if required not in parameters:
            raise RuntimeError(
                "AudioCraft no-CFG overlay is missing {!r}; apply patches/audiocraft first".format(
                    required
                )
            )


def _prepare_frozen_lm(lm: object, device: torch.device) -> None:
    """Move/freeze an AudioCraft LM, including its unregistered T5 encoder.

    In the pinned AudioCraft revision a non-finetuned ``T5Conditioner`` keeps
    ``t5`` directly in ``__dict__`` so that it is not checkpointed as a child
    module.  Consequently ``lm.to(device)``, ``lm.eval()``, and
    ``lm.requires_grad_(False)`` do not reach that encoder.  The probe uses the
    student provider to build the exact condition tensors, so leaving this T5
    on CPU would make the subsequent CUDA generation fail (and leaving it in
    train mode would make the probe stochastic for the wrong reason).
    """

    if not isinstance(lm, torch.nn.Module):
        raise TypeError("loaded AudioCraft LM must be a torch.nn.Module")
    provider = getattr(lm, "condition_provider", None)
    if not isinstance(provider, torch.nn.Module):
        raise TypeError("AudioCraft LM has no nn.Module condition_provider")

    lm.to(device)
    lm.requires_grad_(False)
    lm.eval()
    provider.requires_grad_(False)
    provider.eval()
    for module in provider.modules():
        external_t5 = module.__dict__.get("t5")
        if isinstance(external_t5, torch.nn.Module):
            external_t5.to(device)
            external_t5.requires_grad_(False)
            external_t5.eval()
            # ``T5Conditioner.tokenize`` consults this attribute when moving
            # token IDs before invoking the unregistered encoder.
            if hasattr(module, "device"):
                module.device = str(device)


def _require_musicgen_small_architecture(lm: object) -> Dict[str, int]:
    cfg = getattr(lm, "cfg", None)
    transformer = getattr(cfg, "transformer_lm", None)
    observed = {
        "num_codebooks": int(getattr(lm, "num_codebooks", -1)),
        "cardinality": int(getattr(lm, "card", -1)),
        "transformer_dim": int(getattr(transformer, "dim", -1)),
        "transformer_layers": int(getattr(transformer, "num_layers", -1)),
        "transformer_heads": int(getattr(transformer, "num_heads", -1)),
    }
    if observed != MUSICGEN_SMALL_ARCHITECTURE:
        raise RuntimeError(
            "A2 checkpoint is not the frozen MusicGen-small architecture: {}".format(
                observed
            )
        )
    return observed


def _condition_tensors(lm: object, conditions: list) -> dict:
    provider = lm.condition_provider
    with torch.no_grad():
        tokenized = provider.tokenize(conditions)
        return provider(tokenized)


def _slice_condition_tensors(
    tensors: Mapping[str, Tuple[Tensor, Tensor]], start: int, stop: int
) -> Dict[str, Tuple[Tensor, Tensor]]:
    """Slice the batch axis without recomputing conditioner outputs."""

    return {
        key: (embedding[start:stop], mask[start:stop])
        for key, (embedding, mask) in tensors.items()
    }


def _build_conditions(prompts: Sequence[str], conditioning_class: object) -> list:
    return [conditioning_class(text={"description": prompt}) for prompt in prompts]


def _null_conditions(conditions: list, dropout_class: object) -> list:
    return dropout_class(p=1.0)(conditions)


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _score_probe_no_grad(
    student_lm: object,
    teacher_lm: object,
    codes: Tensor,
    student_conditional_tensors: Mapping[str, Tuple[Tensor, Tensor]],
    teacher_conditional_tensors: Mapping[str, Tuple[Tensor, Tensor]],
    null_tensors: Mapping[str, Tuple[Tensor, Tensor]],
    *,
    teacher_cfg_scale: float,
    bf16: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Whole-trajectory scoring without the training adapter's autograd graph."""

    batch, codebooks, time = codes.shape
    batched_tensors = batch_condition_tensors(
        teacher_conditional_tensors, null_tensors, expected_batch=batch
    )
    # ``no_grad`` avoids retaining model graphs but returns ordinary tensors;
    # the shared loss kernel can then attach a tiny diagnostics-only leaf graph
    # without hitting inference-tensor restrictions.
    with torch.no_grad(), _autocast_context(codes.device, bf16):
        student_output = student_lm.compute_predictions(
            codes,
            conditions=[],
            condition_tensors=dict(student_conditional_tensors),
            keep_only_valid_steps=True,
        )
        teacher_output = teacher_lm.compute_predictions(
            torch.cat((codes, codes), dim=0),
            conditions=[],
            condition_tensors=batched_tensors,
            keep_only_valid_steps=True,
        )
    expected = (batch, codebooks, time)
    if tuple(student_output.logits.shape[:3]) != expected:
        raise ValueError("student LMOutput has an unexpected BQT prefix")
    if tuple(teacher_output.logits.shape[:3]) != (2 * batch, codebooks, time):
        raise ValueError("teacher LMOutput has an unexpected 2BQT prefix")
    if student_output.mask.dtype != torch.bool or teacher_output.mask.dtype != torch.bool:
        raise TypeError("AudioCraft LMOutput masks must be boolean")
    raw_student_logits = student_output.logits
    raw_teacher_cond = teacher_output.logits[:batch]
    raw_teacher_null = teacher_output.logits[batch:]
    raw_student_mask = student_output.mask
    raw_teacher_cond_mask = teacher_output.mask[:batch]
    raw_teacher_null_mask = teacher_output.mask[batch:]
    # This gate must precede any mask intersection, teacher-CFG arithmetic, or
    # invalid-cell sanitization.  It proves that all three raw scoring branches
    # independently expose the exact frozen [B,4,500] delay mask and BQTV logits.
    valid = strict_validate_musicgen_batch(
        codes,
        int(getattr(student_lm, "card")),
        {
            "student": raw_student_mask,
            "teacher_cond": raw_teacher_cond_mask,
            "teacher_null": raw_teacher_null_mask,
        },
        {
            "student": raw_student_logits,
            "teacher_cond": raw_teacher_cond,
            "teacher_null": raw_teacher_null,
        },
    )
    teacher_cond = raw_teacher_cond.float()
    teacher_null = raw_teacher_null.float()
    teacher_cfg = teacher_null + teacher_cfg_scale * (teacher_cond - teacher_null)
    for name, tensor in (
        ("student logits", student_output.logits),
        ("teacher conditional logits", teacher_cond),
        ("teacher null logits", teacher_null),
        ("teacher CFG logits", teacher_cfg),
    ):
        if not bool(torch.isfinite(tensor[valid]).all().item()):
            raise FloatingPointError("{} contains NaN/Inf at valid cells".format(name))
    return raw_student_logits.detach(), teacher_cfg.detach(), valid.detach()


@contextlib.contextmanager
def _rollout_rng(rollout_seed: int, device: torch.device) -> Iterator[None]:
    """Seed one complete manifest pass, restoring caller RNG state afterward."""

    cuda_devices: List[int] = []
    if device.type == "cuda":
        cuda_devices = [
            torch.cuda.current_device() if device.index is None else int(device.index)
        ]
    python_state = random.getstate()
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            random.seed(int(rollout_seed))
            torch.manual_seed(int(rollout_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(rollout_seed))
            yield
    finally:
        random.setstate(python_state)


def _rollout(
    lm: object,
    condition_tensors: Mapping[str, Tuple[Tensor, Tensor]],
    *,
    max_gen_len: int,
    temperature: float,
    top_k: int,
    top_p: float,
    bf16: bool,
) -> Tensor:
    device = next(iter(lm.parameters())).device
    with _autocast_context(device, bf16):
        return lm.generate(
            conditions=[],
            condition_tensors=dict(condition_tensors),
            num_samples=next(iter(condition_tensors.values()))[0].shape[0],
            max_gen_len=max_gen_len,
            use_sampling=True,
            temp=temperature,
            top_k=top_k,
            top_p=top_p,
            use_cfg=False,
            check=True,
        )


def _scored_probe_batches(
    args: argparse.Namespace,
    manifest: Sequence[dict],
    student_lm: object,
    teacher_lm: object,
    conditioning_class: object,
    dropout_class: object,
) -> Iterator[Tuple[int, List[str], List[str], Tensor, Tensor, Tensor, Tensor]]:
    """Yield batches while seeding each complete manifest rollout once."""

    device = next(iter(student_lm.parameters())).device
    max_gen_len = PRIMARY_FRAMES
    for rollout_seed in args.rollout_seeds:
        with _rollout_rng(rollout_seed, device):
            for start in range(0, len(manifest), args.batch_size):
                batch_records = manifest[start : start + args.batch_size]
                prompts = [str(record["prompt"]) for record in batch_records]
                sample_ids = [str(record["sample_id"]) for record in batch_records]
                conditions = _build_conditions(prompts, conditioning_class)
                # One 2B conditioner call guarantees that conditional and null
                # branches share provider state.  The conditional half is then
                # reused verbatim by rollout, student scoring, and teacher scoring.
                null_conditions = _null_conditions(conditions, dropout_class)
                student_batched_tensors = _condition_tensors(
                    student_lm, conditions + null_conditions
                )
                conditional_tensors = _slice_condition_tensors(
                    student_batched_tensors, 0, len(conditions)
                )
                null_tensors = _slice_condition_tensors(
                    student_batched_tensors, len(conditions), 2 * len(conditions)
                )
                codes = _rollout(
                    student_lm,
                    conditional_tensors,
                    max_gen_len=max_gen_len,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    bf16=args.bf16,
                )
                student_logits, teacher_logits, valid = _score_probe_no_grad(
                    student_lm,
                    teacher_lm,
                    codes,
                    conditional_tensors,
                    conditional_tensors,
                    null_tensors,
                    teacher_cfg_scale=args.teacher_cfg_scale,
                    bf16=args.bf16,
                )
                yield (
                    rollout_seed,
                    sample_ids,
                    prompts,
                    codes,
                    student_logits,
                    teacher_logits,
                    valid,
                )


def _probe_records(
    args: argparse.Namespace,
    manifest: Sequence[dict],
    prior: Tensor,
    student_lm: object,
    teacher_lm: object,
    conditioning_class: object,
    dropout_class: object,
) -> Iterator[dict]:
    device = next(iter(student_lm.parameters())).device
    if int(getattr(student_lm, "num_codebooks")) != int(prior.numel()):
        raise ValueError("prior length does not match LM num_codebooks")
    for (
        rollout_seed,
        sample_ids,
        prompts,
        codes,
        student_logits,
        teacher_logits,
        valid,
    ) in _scored_probe_batches(
        args,
        manifest,
        student_lm,
        teacher_lm,
        conditioning_class,
        dropout_class,
    ):
        yield from cell_records_from_logits(
            student_logits,
            teacher_logits,
            codes,
            valid,
            sample_ids=sample_ids,
            prompts=prompts,
            rollout_seeds=[rollout_seed] * len(prompts),
            codebook_weights=prior.to(device),
        )


def _audit_records(
    args: argparse.Namespace,
    manifest: Sequence[dict],
    student_lm: object,
    teacher_lm: object,
    conditioning_class: object,
    dropout_class: object,
) -> Iterator[dict]:
    audit = manifest[:EXPECTED_AUDIT_PROMPTS]
    for (
        rollout_seed,
        sample_ids,
        prompts,
        codes,
        student_logits,
        teacher_logits,
        valid,
    ) in _scored_probe_batches(
        args,
        audit,
        student_lm,
        teacher_lm,
        conditioning_class,
        dropout_class,
    ):
        yield from audit_topk_records(
            student_logits,
            teacher_logits,
            codes,
            valid,
            sample_ids=sample_ids,
            prompts=prompts,
            rollout_seeds=[rollout_seed] * len(prompts),
            top_k=EXPECTED_AUDIT_TOPK,
        )


def run_probe(args: argparse.Namespace) -> None:
    requested_device = torch.device(args.device)
    if args.allow_cpu_debug:
        if requested_device.type != "cpu":
            raise ValueError("--allow-cpu-debug is only valid with --device cpu")
        is_primary = False
        device = requested_device
    else:
        is_primary = True
        device = requested_device
    if is_primary and args.batch_size != 2:
        raise ValueError("primary GPU probe freezes batch-size=2")
    if tuple(args.rollout_seeds) != PRIMARY_ROLLOUT_SEEDS:
        raise ValueError("primary probe rollout seeds are frozen to 31001 31002")
    if (args.temperature, args.top_k, args.top_p) != (
        DEFAULT_TEMPERATURE,
        DEFAULT_TOP_K,
        DEFAULT_TOP_P,
    ):
        raise ValueError("primary sampling is frozen to temp=1, top-k=250, top-p=0")
    if args.duration_seconds != DEFAULT_DURATION_SECONDS:
        raise ValueError("primary rollout duration is frozen to 10 seconds")
    if is_primary and device.type != "cuda":
        raise RuntimeError("formal probe requires CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("formal probe requires CUDA BF16 support")
    if is_primary and not args.bf16:
        raise RuntimeError("formal probe requires BF16 rollout/scoring")

    manifest_path = Path(args.probe_manifest).resolve()
    dev_manifest_path = Path(args.dev_manifest).resolve()
    supplied_checkpoint_path = Path(args.checkpoint).expanduser()
    checkpoint_path = supplied_checkpoint_path.resolve()
    prior_artifact_path = Path(args.codebook_prior_artifact_dir).resolve()
    cfg_decision_path = Path(args.cfg_scale_decision_dir).resolve()
    supplied_audiocraft_root = Path(args.audiocraft_root).expanduser()
    audiocraft_root = supplied_audiocraft_root.resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite probe artifact: {}".format(output_dir))

    manifest, dev_manifest = validate_probe_is_exact_dev_subset(
        manifest_path, dev_manifest_path
    )
    manifest_hash = sha256_file(manifest_path)
    dev_manifest_hash = sha256_file(dev_manifest_path)
    cfg_decision = verify_cfg_scale_decision(cfg_decision_path)
    if cfg_decision.generation_identity.get("model_id") != "facebook/musicgen-small":
        raise ValueError("primary A2 disagreement probe is frozen to MusicGen-small")
    if cfg_decision.generation_identity["manifest_sha256"] != dev_manifest_hash:
        raise ValueError("CFG decision is not bound to the supplied dev manifest")
    args.teacher_cfg_scale = float(cfg_decision.selected_cfg_scale)
    prior_artifact = load_codec_prior_artifact(prior_artifact_path)
    verified_local_codec_snapshot = verify_local_codec_snapshot(prior_artifact)
    prior = torch.as_tensor(prior_artifact.prior, dtype=torch.float32)
    checkpoint_identity = verify_checkpoint_snapshot(supplied_checkpoint_path)
    source_identity = verify_audiocraft_source(supplied_audiocraft_root)
    for key, expected in EXPECTED_PRECISION_IDENTITY.items():
        if cfg_decision.generation_identity.get(key) != expected:
            raise ValueError("CFG precision identity mismatch for {}".format(key))
    for key in ("checkpoint_sha256", "state_dict_sha256", "compression_state_dict_sha256"):
        if checkpoint_identity[key] != cfg_decision.generation_identity[key]:
            raise ValueError("checkpoint/CFG decision mismatch for {}".format(key))
    for key in ("audiocraft_base_commit", "audiocraft_source_sha256", "audiocraft_lm_sha256"):
        if source_identity[key] != cfg_decision.generation_identity[key]:
            raise ValueError("AudioCraft/CFG decision mismatch for {}".format(key))
    if prior_artifact.checkpoint_sha256 != checkpoint_identity["compression_state_dict_sha256"]:
        raise ValueError("A1 prior codec checkpoint differs from supplied MusicGen codec")
    prior_audiocraft = prior_artifact.identity["audiocraft"]
    if not isinstance(prior_audiocraft, Mapping):
        raise ValueError("A1 prior has no AudioCraft source identity")
    prior_source = prior_audiocraft.get("source_identity")
    if not isinstance(prior_source, Mapping) or prior_source.get(
        "tree_sha256"
    ) != source_identity["audiocraft_source_sha256"]:
        raise ValueError("A1 prior and A2 probe use different AudioCraft source trees")
    offline_environment = (
        {
            name: str(os.environ.get(name, ""))
            for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
        }
        if not is_primary
        else require_offline_hf_environment()
    )

    with _temporary_import_path(audiocraft_root):
        loaders = importlib.import_module("audiocraft.models.loaders")
        conditioners = importlib.import_module("audiocraft.modules.conditioners")
        student_lm = loaders.load_lm_model(str(checkpoint_path), device="cpu")
        teacher_lm = loaders.load_lm_model(str(checkpoint_path), device="cpu")
        if student_lm is teacher_lm:
            raise RuntimeError("student and teacher must be independent objects")
        architectures: Dict[str, Dict[str, int]] = {}
        for name, lm in (("student", student_lm), ("teacher", teacher_lm)):
            _assert_audiocraft_overlay(lm)
            if any(parameter.dtype != torch.float32 for parameter in lm.parameters()):
                raise RuntimeError("{} LM did not load as CPU FP32".format(name))
            strict_validate_musicgen_model(lm, frame_rate=PRIMARY_FRAME_RATE)
            architectures[name] = _require_musicgen_small_architecture(lm)
            _prepare_frozen_lm(lm, device)
        if architectures["student"] != architectures["teacher"]:
            raise RuntimeError("student and teacher MusicGen architectures differ")
        student_t5 = loaded_t5_identity(student_lm)
        teacher_t5 = loaded_t5_identity(teacher_lm)
        if student_t5["identity_sha256"] != teacher_t5["identity_sha256"]:
            raise RuntimeError("student and teacher loaded different external T5")
        if student_t5["identity_sha256"] != cfg_decision.generation_identity["loaded_t5_identity_sha256"]:
            raise ValueError("loaded T5 differs from CFG generation identity")
        student_state_hash = hash_module_state(student_lm)
        if student_state_hash != hash_module_state(teacher_lm):
            raise RuntimeError("student and teacher loaded different LM state")

        scientific_config = {
            "schema_version": PROBE_CONFIG_SCHEMA_VERSION,
            "probe_manifest_sha256": manifest_hash,
            "dev_manifest_sha256": dev_manifest_hash,
            "probe_is_exact_dev_subset": True,
            "probe_prompts": len(manifest),
            "dev_prompts": len(dev_manifest),
            "checkpoint_identity": checkpoint_identity,
            "audiocraft_identity": source_identity,
            "loaded_t5_identity_sha256": student_t5["identity_sha256"],
            "loaded_lm_state_sha256": student_state_hash,
            "codebook_prior_artifact_identity_sha256": prior_artifact.identity_sha256,
            "codebook_prior_artifact_seal_sha256": prior_artifact.artifact_seal_sha256,
            "verified_local_codec_snapshot": verified_local_codec_snapshot,
            "cfg_scale_decision_file_sha256": cfg_decision.decision_file_sha256,
            "cfg_scale_decision_payload_sha256": cfg_decision.decision_payload_sha256,
            "teacher_cfg_scale": args.teacher_cfg_scale,
            "model_id": cfg_decision.generation_identity["model_id"],
            "musicgen_architecture": architectures["student"],
            "rollout_seeds": list(args.rollout_seeds),
            "duration_seconds": args.duration_seconds,
            "token_frames": PRIMARY_FRAMES,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "batch_size": args.batch_size,
            "device_type": device.type,
            "primary_contract": is_primary,
            "parameter_dtype": "torch.float32",
            "compute_dtype": "torch.bfloat16" if args.bf16 else "torch.float32",
            "offline_environment": dict(offline_environment),
        }
        with staged_output_directory(output_dir) as staging:
            output_cells = staging / PROBE_CELLS_FILENAME
            output_audit = staging / PROBE_AUDIT_FILENAME
            output_metadata = staging / PROBE_METADATA_FILENAME
            cell_count = write_jsonl_gzip_atomic(
                output_cells,
                validated_probe_records(
                    _probe_records(
                        args, manifest, prior, student_lm, teacher_lm,
                        conditioners.ConditioningAttributes,
                        conditioners.ClassifierFreeGuidanceDropout,
                    ),
                    expected_sample_ids=[record["sample_id"] for record in manifest],
                    expected_seeds=PRIMARY_ROLLOUT_SEEDS,
                    expected_records_per_sequence=EXPECTED_VALID_CELLS_PER_SEQUENCE,
                ),
            )
            audit_count = write_jsonl_gzip_atomic(
                output_audit,
                validated_probe_records(
                    _audit_records(
                        args, manifest, student_lm, teacher_lm,
                        conditioners.ConditioningAttributes,
                        conditioners.ClassifierFreeGuidanceDropout,
                    ),
                    expected_sample_ids=[record["sample_id"] for record in manifest[:EXPECTED_AUDIT_PROMPTS]],
                    expected_seeds=PRIMARY_ROLLOUT_SEEDS,
                    expected_records_per_sequence=EXPECTED_VALID_CELLS_PER_SEQUENCE,
                    audit_top_k=EXPECTED_AUDIT_TOPK,
                ),
            )
            metadata = {
                "schema_version": PROBE_RUN_SCHEMA_VERSION,
                "scientific_status": "gpu_probe_complete" if is_primary else "cpu_debug_only",
                "primary_contract_passed": is_primary,
                "scientific_config": scientific_config,
                "scientific_config_sha256": sha256_json(scientific_config),
                "probe_manifest_path": str(manifest_path),
                "dev_manifest_path": str(dev_manifest_path),
                "codebook_prior_artifact_dir": prior_artifact.directory,
                "cfg_scale_decision_dir": cfg_decision.directory,
                "checkpoint_path": str(checkpoint_path),
                "audiocraft_root": str(audiocraft_root),
                "student_teacher_same_checkpoint_and_state": True,
                "condition_tensors_reused_for_rollout_and_scoring": True,
                "strict_raw_branch_contract_before_intersection": True,
                "probe_sample_ids": [record["sample_id"] for record in manifest],
                "audit_sample_ids": [
                    record["sample_id"] for record in manifest[:EXPECTED_AUDIT_PROMPTS]
                ],
                "records_per_sequence": EXPECTED_VALID_CELLS_PER_SEQUENCE,
                "cell_count": cell_count,
                "expected_cell_count": EXPECTED_PROMPTS * 2 * EXPECTED_VALID_CELLS_PER_SEQUENCE,
                "audit_prompt_count": EXPECTED_AUDIT_PROMPTS,
                "audit_top_k": EXPECTED_AUDIT_TOPK,
                "audit_cell_count": audit_count,
                "expected_audit_cell_count": EXPECTED_AUDIT_PROMPTS * 2 * EXPECTED_VALID_CELLS_PER_SEQUENCE,
                "loaded_t5_identity": student_t5,
                "codebook_prior_artifact_identity": prior_artifact.identity,
                "cfg_scale_decision": {
                    "status": cfg_decision.status,
                    "selected_cfg_scale": cfg_decision.selected_cfg_scale,
                    "decision_file_sha256": cfg_decision.decision_file_sha256,
                    "decision_payload_sha256": cfg_decision.decision_payload_sha256,
                    "generation_identity": dict(cfg_decision.generation_identity),
                },
                "outputs": {
                    PROBE_CELLS_FILENAME: artifact_member(output_cells),
                    PROBE_AUDIT_FILENAME: artifact_member(output_audit),
                },
            }
            write_json_atomic(output_metadata, metadata)
            write_artifact_seal(
                staging / PROBE_SEAL_FILENAME,
                schema_version=PROBE_SEAL_SCHEMA_VERSION,
                status="complete_gpu_probe" if is_primary else "complete_cpu_debug",
                members=[output_cells, output_audit, output_metadata],
            )
            verify_probe_artifact(staging, require_primary=is_primary)


def _csv_value(value: object) -> object:
    return "" if value is None else value


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty summary CSV")
    fields = list(rows[0].keys())
    with atomic_binary_writer(path) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8", newline="", write_through=True)
        writer = csv.DictWriter(text, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})
        text.flush()
        text.detach()


def write_npz_atomic(path: Path, arrays: Mapping[str, Tensor]) -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("the sealed summary artifact requires numpy") from exc
    with atomic_binary_writer(path) as stream:
        np.savez_compressed(
            stream,
            **{key: value.detach().cpu().numpy() for key, value in arrays.items()},
        )


def verify_summary_artifact(
    directory: Path, *, require_primary: bool = True
) -> Dict[str, object]:
    """Rehash and validate one closed model-independent A3 summary."""

    expected_status = (
        "complete_primary_summary" if require_primary else "complete_debug_summary"
    )
    members = verify_artifact_seal(
        directory,
        seal_filename=SUMMARY_SEAL_FILENAME,
        seal_schema=SUMMARY_SEAL_SCHEMA_VERSION,
        expected_status=expected_status,
        expected_members=(
            SUMMARY_CSV_FILENAME,
            SUMMARY_NPZ_FILENAME,
            SUMMARY_JSON_FILENAME,
        ),
    )
    resolved = directory.expanduser().resolve(strict=True)
    payload = _load_json_object(
        resolved / SUMMARY_JSON_FILENAME, SUMMARY_JSON_FILENAME
    )
    if payload.get("schema_version") != SUMMARY_RUN_SCHEMA_VERSION:
        raise ValueError("summary metadata schema mismatch")
    if payload.get("phenomenon_summary_schema_version") != SUMMARY_SCHEMA_VERSION:
        raise ValueError("phenomenon summary schema mismatch")
    expected_scientific_status = (
        "primary_summary" if require_primary else "nonprimary_debug"
    )
    if payload.get("scientific_status") != expected_scientific_status:
        raise ValueError("summary scientific status mismatch")
    if payload.get("primary_contract_passed") is not require_primary:
        raise ValueError("summary primary-contract marker mismatch")
    config = payload.get("scientific_config")
    if not isinstance(config, dict):
        raise ValueError("summary has no scientific_config object")
    if config.get("schema_version") != SUMMARY_CONFIG_SCHEMA_VERSION:
        raise ValueError("summary scientific-config schema mismatch")
    if payload.get("scientific_config_sha256") != sha256_json(config):
        raise ValueError("summary scientific-config hash mismatch")
    if config.get("primary_contract") is not require_primary:
        raise ValueError("summary config primary marker mismatch")

    outputs = payload.get("outputs")
    expected_outputs = {SUMMARY_CSV_FILENAME, SUMMARY_NPZ_FILENAME}
    if not isinstance(outputs, dict) or set(outputs) != expected_outputs:
        raise ValueError("summary output identity set mismatch")
    for name in expected_outputs:
        if outputs.get(name) != members[name]:
            raise ValueError("summary output hash mismatch for {}".format(name))

    for key in ("prompt_count", "sequence_count", "cell_count", "codebook_count"):
        if type(payload.get(key)) is not int or int(payload[key]) <= 0:
            raise ValueError("summary {} must be a positive integer".format(key))
    if payload.get("rollout_seeds") != list(PRIMARY_ROLLOUT_SEEDS):
        raise ValueError("summary rollout seeds differ from the frozen pair")
    bootstrap = payload.get("bootstrap")
    if not isinstance(bootstrap, dict):
        raise ValueError("summary bootstrap metadata is absent")
    if bootstrap.get("unit") != "prompt" or bootstrap.get(
        "rollouts_nested_within_prompt"
    ) is not True:
        raise ValueError("summary did not use prompt-outer nested bootstrap")
    if type(bootstrap.get("seed")) is not int or type(bootstrap.get("replicates")) is not int:
        raise ValueError("summary bootstrap seed/replicates must be integers")
    if require_primary:
        _require_exact_int(
            payload.get("prompt_count"), EXPECTED_PROMPTS, "prompt_count"
        )
        _require_exact_int(
            payload.get("sequence_count"),
            EXPECTED_PROMPTS * len(PRIMARY_ROLLOUT_SEEDS),
            "sequence_count",
        )
        _require_exact_int(
            payload.get("cell_count"),
            EXPECTED_PROMPTS
            * len(PRIMARY_ROLLOUT_SEEDS)
            * EXPECTED_VALID_CELLS_PER_SEQUENCE,
            "cell_count",
        )
        _require_exact_int(payload.get("codebook_count"), 4, "codebook_count")
        _require_exact_int(
            bootstrap.get("seed"), PRIMARY_BOOTSTRAP_SEED, "bootstrap.seed"
        )
        _require_exact_int(
            bootstrap.get("replicates"),
            PRIMARY_BOOTSTRAP_REPLICATES,
            "bootstrap.replicates",
        )

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("verifying a summary artifact requires numpy") from exc
    with np.load(resolved / SUMMARY_NPZ_FILENAME, allow_pickle=False) as archive:
        if "tv_u_vs_w" not in archive.files:
            raise ValueError("summary bootstrap NPZ lacks tv_u_vs_w")
        replicates = int(bootstrap["replicates"])
        for name in archive.files:
            array = archive[name]
            if array.dtype.hasobject:
                raise ValueError("summary bootstrap NPZ contains object arrays")
            if array.ndim < 1 or int(array.shape[0]) != replicates:
                raise ValueError(
                    "summary bootstrap array {} has the wrong replicate axis".format(name)
                )
    return {
        "directory": str(resolved),
        "metadata": payload,
        "members": members,
        "artifact_seal_sha256": sha256_file(resolved / SUMMARY_SEAL_FILENAME),
    }


def run_summarize(args: argparse.Namespace) -> None:
    if not args.allow_nonprimary_input and (
        args.bootstrap_replicates != PRIMARY_BOOTSTRAP_REPLICATES
        or args.bootstrap_seed != PRIMARY_BOOTSTRAP_SEED
    ):
        raise ValueError(
            "primary summary freezes prompt bootstrap to 10000 replicates, seed 4702"
        )
    probe_dir = Path(args.probe_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    verified_probe = verify_probe_artifact(
        probe_dir, require_primary=not args.allow_nonprimary_input
    )
    probe_metadata = verified_probe["metadata"]
    assert isinstance(probe_metadata, dict)
    input_cells = probe_dir / PROBE_CELLS_FILENAME

    accumulator = PhenomenonAccumulator()
    accumulator.consume_many(iter_jsonl_gzip(input_cells))
    summary = accumulator.finalize(
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        retain_bootstrap_arrays=True,
    )
    if not args.allow_nonprimary_input:
        if summary.prompt_count != EXPECTED_PROMPTS:
            raise ValueError(
                "primary summary requires exactly {} prompts, got {}".format(
                    EXPECTED_PROMPTS, summary.prompt_count
                )
            )
        if summary.sequence_count != EXPECTED_PROMPTS * len(PRIMARY_ROLLOUT_SEEDS):
            raise ValueError(
                "primary summary requires exactly {} prompt-rollout sequences, got {}".format(
                    EXPECTED_PROMPTS * len(PRIMARY_ROLLOUT_SEEDS), summary.sequence_count
                )
            )
        if summary.rollout_seeds != PRIMARY_ROLLOUT_SEEDS:
            raise ValueError(
                "primary summary requires rollout seeds {}, got {}".format(
                    PRIMARY_ROLLOUT_SEEDS, summary.rollout_seeds
                )
            )
        if any(
            seeds != PRIMARY_ROLLOUT_SEEDS
            for seeds in summary.rollout_seeds_by_prompt.values()
        ):
            raise ValueError(
                "every primary prompt must contain exactly rollout seeds 31001 and 31002"
            )
    assert summary.bootstrap_arrays is not None
    scientific_config = {
        "schema_version": SUMMARY_CONFIG_SCHEMA_VERSION,
        "primary_contract": not args.allow_nonprimary_input,
        "input_probe_artifact_seal_sha256": verified_probe[
            "artifact_seal_sha256"
        ],
        "input_probe_metadata_sha256": sha256_file(
            probe_dir / PROBE_METADATA_FILENAME
        ),
        "input_cells_sha256": sha256_file(input_cells),
        "bootstrap_seed": summary.bootstrap_seed,
        "bootstrap_replicates": summary.bootstrap_replicates,
        "bootstrap_unit": "prompt",
        "rollouts_nested_within_prompt": True,
    }
    with staged_output_directory(output_dir) as staging:
        output_csv = staging / SUMMARY_CSV_FILENAME
        output_npz = staging / SUMMARY_NPZ_FILENAME
        output_json = staging / SUMMARY_JSON_FILENAME
        write_csv_atomic(output_csv, summary.rows)
        write_npz_atomic(output_npz, summary.bootstrap_arrays)
        payload = {
            "schema_version": SUMMARY_RUN_SCHEMA_VERSION,
            "phenomenon_summary_schema_version": SUMMARY_SCHEMA_VERSION,
            "scientific_status": (
                "nonprimary_debug"
                if args.allow_nonprimary_input
                else "primary_summary"
            ),
            "primary_contract_passed": not args.allow_nonprimary_input,
            "scientific_config": scientific_config,
            "scientific_config_sha256": sha256_json(scientific_config),
            "input_probe_dir": str(probe_dir),
            "input_probe_artifact_seal_sha256": verified_probe[
                "artifact_seal_sha256"
            ],
            "input_probe_scientific_config_sha256": probe_metadata[
                "scientific_config_sha256"
            ],
            "prompt_count": summary.prompt_count,
            "sequence_count": summary.sequence_count,
            "cell_count": summary.cell_count,
            "codebook_count": summary.codebook_count,
            "rollout_seeds": list(summary.rollout_seeds),
            "bootstrap": {
                "unit": "prompt",
                "rollouts_nested_within_prompt": True,
                "seed": summary.bootstrap_seed,
                "replicates": summary.bootstrap_replicates,
            },
            "tv_u_vs_w": summary.tv,
            "summary_rows": summary.rows,
            "outputs": {
                SUMMARY_CSV_FILENAME: artifact_member(output_csv),
                SUMMARY_NPZ_FILENAME: artifact_member(output_npz),
            },
        }
        write_json_atomic(output_json, payload)
        write_artifact_seal(
            staging / SUMMARY_SEAL_FILENAME,
            schema_version=SUMMARY_SEAL_SCHEMA_VERSION,
            status=(
                "complete_debug_summary"
                if args.allow_nonprimary_input
                else "complete_primary_summary"
            ),
            members=[output_csv, output_npz, output_json],
        )
        verify_summary_artifact(
            staging, require_primary=not args.allow_nonprimary_input
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser("probe", help="run the real AudioCraft/GPU probe")
    probe.add_argument("--probe-manifest", required=True)
    probe.add_argument("--dev-manifest", required=True)
    probe.add_argument("--checkpoint", required=True, help="local checkpoint file/directory")
    probe.add_argument("--audiocraft-root", required=True, help="patched AudioCraft repository")
    probe.add_argument("--codebook-prior-artifact-dir", required=True)
    probe.add_argument("--cfg-scale-decision-dir", required=True)
    probe.add_argument("--output-dir", required=True)
    probe.add_argument("--rollout-seeds", type=int, nargs=2, default=list(PRIMARY_ROLLOUT_SEEDS))
    probe.add_argument("--duration-seconds", type=float, default=DEFAULT_DURATION_SECONDS)
    probe.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    probe.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    probe.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    probe.add_argument("--batch-size", type=int, default=2)
    probe.add_argument("--device", default="cuda:0")
    probe.add_argument("--bf16", action="store_true")
    probe.add_argument("--allow-cpu-debug", action="store_true", help="never treat as science")
    probe.set_defaults(func=run_probe)

    summarize = subparsers.add_parser("summarize", help="aggregate an existing scalar map")
    summarize.add_argument("--probe-dir", required=True)
    summarize.add_argument("--output-dir", required=True)
    summarize.add_argument(
        "--bootstrap-replicates", type=int, default=PRIMARY_BOOTSTRAP_REPLICATES
    )
    summarize.add_argument("--bootstrap-seed", type=int, default=PRIMARY_BOOTSTRAP_SEED)
    summarize.add_argument(
        "--allow-nonprimary-input",
        action="store_true",
        help="unit/debug summaries only; skips the 256 x 2 completeness gate",
    )
    summarize.set_defaults(func=run_summarize)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "batch_size", 1) <= 0:
        raise ValueError("batch-size must be positive")
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
