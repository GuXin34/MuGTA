#!/usr/bin/env python3
"""Shared immutable-artifact code for the two CFG evaluator environments.

This module deliberately contains no evaluator imports.  The quality and
music-CLAP wrappers can therefore verify, join, and seal artifacts in a
lightweight Python 3.11 process, and their unit tests never need the real
models.  The generation verifier is the exact verifier used by
``cfg_scale_gate.py`` rather than a second, weaker implementation.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import cfg_scale_gate as gate

# The pinned evaluator checkout must not acquire untracked Python bytecode just
# because this wrapper imported it.  This also makes small/medium evaluation
# repeatable against the same read-only source checkout.
sys.dont_write_bytecode = True


QUALITY_SCORE_SCHEMA_VERSION = "ptc-opd-cfg-quality-score-v1"
QUALITY_PROVENANCE_SCHEMA_VERSION = gate.QUALITY_PROVENANCE_SCHEMA_VERSION
QUALITY_SEAL_SCHEMA_VERSION = gate.QUALITY_SEAL_SCHEMA_VERSION
EXTERNAL_SEAL_SCHEMA_VERSION = gate.EXTERNAL_SEAL_SCHEMA_VERSION
QUALITY_METRICS: Tuple[str, ...] = ("muq_mi", "audiobox_ce", "audiobox_pq")
QUALITY_EVALUATORS: Tuple[str, ...] = ("muq_eval", "audiobox_aesthetics")
PINNED_MUQ_EVAL_COMMIT = "60a88f8ac0909ca1fd1a78af3660f1fc376977a1"
HEX_DIGITS = frozenset("0123456789abcdef")

OFFLINE_ENVIRONMENT = dict(gate.OFFLINE_ENVIRONMENT)


def enforce_offline_environment() -> None:
    """Set offline flags before any evaluator package is imported."""

    for name, value in OFFLINE_ENVIRONMENT.items():
        os.environ[name] = value


@contextlib.contextmanager
def deny_network_connections() -> Iterator[None]:
    """Fail closed if a Python evaluator tries to open a network socket."""

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection

    def blocked(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("network access is forbidden during CFG evaluation")

    socket.socket.connect = blocked  # type: ignore[assignment]
    socket.socket.connect_ex = blocked  # type: ignore[assignment]
    socket.create_connection = blocked  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[assignment]
        socket.socket.connect_ex = original_connect_ex  # type: ignore[assignment]
        socket.create_connection = original_create_connection  # type: ignore[assignment]


def canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    return gate.canonical_json_bytes(value, pretty=pretty)


def sha256_file(path: Path) -> str:
    return gate.sha256_file(path)


def sha256_json(value: object) -> str:
    return gate.sha256_json(value)


def require_sha256(value: object, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(character not in HEX_DIGITS for character in normalized):
        raise ValueError("{} must be 64 lowercase hexadecimal characters".format(label))
    return normalized


def require_local_file(path: Path, label: str, *, basename: Optional[str] = None) -> Path:
    path = path.expanduser()
    if path.is_symlink() or not path.is_file():
        raise ValueError("{} must be an existing non-symlink regular file: {}".format(label, path))
    path = path.resolve()
    if basename is not None and path.name != basename:
        raise ValueError("{} must be named {!r}, got {}".format(label, basename, path.name))
    return path


def require_local_directory(path: Path, label: str) -> Path:
    path = path.expanduser()
    if path.is_symlink() or not path.is_dir():
        raise ValueError("{} must be an existing non-symlink directory: {}".format(label, path))
    return path.resolve()


def _hash_named_paths(root: Path, paths: Sequence[Path]) -> str:
    if not paths:
        raise ValueError("cannot hash an empty source/configuration set")
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink() and not path.exists():
            raise ValueError("broken symlink in hashed tree: {}".format(path))
        if not path.is_file():
            raise ValueError("hashed tree contains a non-file: {}".format(path))
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def sha256_local_tree(root: Path) -> str:
    """Hash a local snapshot by relative names and dereferenced file bytes."""

    root = require_local_directory(root, "local snapshot")
    paths: List[Path] = []
    for item in root.rglob("*"):
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in item.relative_to(root).parts):
            continue
        if item.name == ".DS_Store" or item.suffix == ".pyc":
            continue
        if item.is_symlink() and not item.exists():
            raise ValueError("broken symlink in local snapshot: {}".format(item))
        if item.is_file():
            paths.append(item)
    return _hash_named_paths(root, paths)


def sha256_python_package(package_root: Path) -> str:
    package_root = require_local_directory(package_root, "imported evaluator package")
    paths = sorted(package_root.rglob("*.py"))
    return _hash_named_paths(package_root, paths)


def imported_package_identity(module: object, label: str) -> Dict[str, object]:
    source = getattr(module, "__file__", None)
    if not isinstance(source, str) or not source:
        raise RuntimeError("cannot locate imported {} package".format(label))
    root = Path(source).resolve().parent
    return {
        "package_root": str(root),
        "source_sha256": sha256_python_package(root),
    }


def package_file_hashes(package_root: Path) -> Dict[str, object]:
    """Return a transparent per-file identity for an imported Python package."""

    package_root = require_local_directory(package_root, "imported evaluator package")
    files = sorted(package_root.rglob("*.py"))
    if not files:
        raise ValueError("imported evaluator package contains no Python files")
    return {
        "python_file_count": len(files),
        "python_files": [
            {
                "relative_path": path.relative_to(package_root).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in files
        ],
    }


def pinned_git_source_identity(root: Path) -> Dict[str, object]:
    """Require the exact clean MuQ-Eval commit and hash all tracked bytes."""

    root = require_local_directory(root, "MuQ-Eval source root")
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if head.returncode != 0:
        raise ValueError("MuQ-Eval root has no Git provenance: {}".format(head.stderr.strip()))
    commit = head.stdout.strip()
    if commit != PINNED_MUQ_EVAL_COMMIT:
        raise ValueError(
            "MuQ-Eval commit mismatch: expected {}, observed {}".format(
                PINNED_MUQ_EVAL_COMMIT, commit
            )
        )
    tracked_status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if tracked_status.returncode != 0:
        raise ValueError(
            "cannot inspect MuQ-Eval worktree: {}".format(tracked_status.stderr.strip())
        )
    if tracked_status.stdout.strip():
        raise ValueError("MuQ-Eval tracked files must be clean at the pinned commit")
    untracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if untracked.returncode != 0:
        raise ValueError("cannot enumerate MuQ-Eval untracked files")
    forbidden_untracked = []
    for raw in untracked.stdout.split(b"\0"):
        if not raw:
            continue
        relative = Path(raw.decode("utf-8"))
        if (
            "__pycache__" in relative.parts
            or ".pytest_cache" in relative.parts
            or relative.name == ".DS_Store"
            or relative.suffix == ".pyc"
        ):
            continue
        forbidden_untracked.append(relative.as_posix())
    if forbidden_untracked:
        raise ValueError(
            "MuQ-Eval checkout has untracked non-cache files: {}".format(
                forbidden_untracked[:20]
            )
        )
    listed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if listed.returncode != 0:
        raise ValueError("cannot enumerate MuQ-Eval tracked files")
    paths = [root / raw.decode("utf-8") for raw in listed.stdout.split(b"\0") if raw]
    return {
        "git_commit": commit,
        "source_sha256": _hash_named_paths(root, paths),
        "source_root": str(root),
    }


def hash_config_files(paths: Sequence[Path]) -> str:
    resolved = [require_local_file(path, "configuration file") for path in paths]
    try:
        common = Path(os.path.commonpath([str(path.parent) for path in resolved]))
    except ValueError as exc:
        raise ValueError("configuration files do not share a local root") from exc
    return _hash_named_paths(common, resolved)


def finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("{} must be numeric".format(label))
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("{} must be finite".format(label))
    return result


def read_json(path: Path) -> Dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(
            "{} contains forbidden non-finite JSON constant {}".format(path, value)
        )

    def unique_object(pairs: Sequence[Tuple[str, object]]) -> Dict[str, object]:
        result: Dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("{} contains duplicate key {!r}".format(path, key))
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as stream:
        value = json.load(
            stream,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    if not isinstance(value, dict):
        raise ValueError("JSON artifact must contain an object: {}".format(path))
    return value


def iter_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.endswith("\n"):
                raise ValueError("JSONL line {} is not newline terminated".format(line_number))
            if not line.strip():
                raise ValueError("blank JSONL line {} is forbidden".format(line_number))
            try:
                value = json.loads(
                    line,
                    object_pairs_hook=lambda pairs: _unique_json_object(
                        pairs, "JSONL line {}".format(line_number)
                    ),
                    parse_constant=lambda value: _reject_json_constant(
                        value, "JSONL line {}".format(line_number)
                    ),
                )
            except json.JSONDecodeError as exc:
                raise ValueError("invalid JSONL line {}: {}".format(line_number, exc)) from exc
            if not isinstance(value, dict):
                raise ValueError("JSONL line {} must be an object".format(line_number))
            yield value


def _unique_json_object(
    pairs: Sequence[Tuple[str, object]], label: str
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("{} contains duplicate key {!r}".format(label, key))
        result[key] = value
    return result


def _reject_json_constant(value: str, label: str) -> None:
    raise ValueError(
        "{} contains forbidden non-finite JSON constant {}".format(label, value)
    )


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def write_json(path: Path, value: object) -> None:
    _write_bytes(path, canonical_json_bytes(value, pretty=True))


def write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
            count += 1
        stream.flush()
        os.fsync(stream.fileno())
    return count


@contextlib.contextmanager
def staged_directory(target: Path) -> Iterator[Path]:
    target = target.expanduser().resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError("refusing to overwrite artifact directory: {}".format(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".{}.partial.".format(target.name), dir=str(target.parent)))
    try:
        yield temporary
        if target.exists() or target.is_symlink():
            raise FileExistsError("output appeared while staging: {}".format(target))
        os.replace(str(temporary), str(target))
        directory_fd = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def load_verified_generation(generation_dir: Path) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    generation_dir = require_local_directory(generation_dir, "generation artifact")
    identity = gate.verify_generation_directory(generation_dir, rehash_audio=True)
    records = list(iter_jsonl(generation_dir / "samples.jsonl"))
    if len(records) != int(identity["sample_records"]):
        raise ValueError("generation record count changed after verification")
    seen = set()
    for record in records:
        key = (str(record.get("sample_id", "")), str(record.get("condition_id", "")))
        if not all(key) or key in seen:
            raise ValueError("generation record key is empty or duplicated: {}".format(key))
        seen.add(key)
        relative = Path(str(record.get("path", "")))
        audio_path = generation_dir / relative
        if not audio_path.is_file() or sha256_file(audio_path) != record.get("audio_sha256"):
            raise ValueError("generation audio changed after verification: {}".format(audio_path))
    return identity, records


def generation_binding(identity: Mapping[str, object]) -> Dict[str, object]:
    return {
        field: identity[field]
        for field in (
            "generation_run_sha256",
            "samples_jsonl_sha256",
            "artifact_seal_sha256",
            "scientific_config_sha256",
            "prompt_count",
            "sample_records",
        )
    }


def validate_evaluator_identity(value: object, label: str) -> Dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "checkpoint_sha256",
        "source_sha256",
        "config_sha256",
        "details",
    }:
        raise ValueError("{} evaluator identity must be an object".format(label))
    normalized = dict(value)
    for field in ("checkpoint_sha256", "source_sha256", "config_sha256"):
        normalized[field] = require_sha256(value.get(field), "{}.{}".format(label, field))
    details = value.get("details")
    if not isinstance(details, dict) or not details:
        raise ValueError("{}.details must be a nonempty object".format(label))
    return normalized


def quality_row(
    generated: Mapping[str, object],
    *,
    provenance_sha256: str,
    muq_mi: object,
    audiobox_ce: object,
    audiobox_pq: object,
) -> Dict[str, object]:
    return {
        "schema_version": QUALITY_SCORE_SCHEMA_VERSION,
        "sample_id": generated["sample_id"],
        "prompt_sha256": generated["prompt_sha256"],
        "condition": generated["condition"],
        "cfg_scale": generated["cfg_scale"],
        "condition_id": generated["condition_id"],
        "audio_sha256": generated["audio_sha256"],
        "scientific_config_sha256": generated["scientific_config_sha256"],
        "quality_provenance_sha256": provenance_sha256,
        "metrics": {
            "muq_mi": finite_float(muq_mi, "muq_mi"),
            "audiobox_ce": finite_float(audiobox_ce, "audiobox_ce"),
            "audiobox_pq": finite_float(audiobox_pq, "audiobox_pq"),
        },
    }


def verify_quality_artifact(
    quality_dir: Path,
    generation_dir: Path,
) -> Tuple[Dict[str, object], List[Dict[str, object]], Dict[str, object]]:
    quality_dir = require_local_directory(quality_dir, "quality artifact")
    expected_names = {"quality_scores.jsonl", "quality_provenance.json", "artifact_seal.json"}
    observed_names = {item.name for item in quality_dir.iterdir()}
    if observed_names != expected_names:
        raise ValueError(
            "quality artifact members mismatch: expected {}, observed {}".format(
                sorted(expected_names), sorted(observed_names)
            )
        )
    for name in expected_names:
        path = quality_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("quality artifact member must be a regular non-symlink file: {}".format(path))

    generation_identity, generation_records = load_verified_generation(generation_dir)
    expected = {
        (str(record["sample_id"]), str(record["condition_id"])): record
        for record in generation_records
    }
    provenance_path = quality_dir / "quality_provenance.json"
    scores_path = quality_dir / "quality_scores.jsonl"
    seal_path = quality_dir / "artifact_seal.json"
    provenance = read_json(provenance_path)
    provenance_hash = sha256_file(provenance_path)
    if set(provenance) != {
        "schema_version",
        "status",
        "metrics",
        "generation",
        "evaluators",
        "offline_environment",
    }:
        raise ValueError("quality provenance field set mismatch")
    if provenance.get("schema_version") != QUALITY_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("quality provenance schema mismatch")
    if provenance.get("status") != "accepted_quality_evaluation":
        raise ValueError("quality provenance is not accepted")
    if provenance.get("metrics") != list(QUALITY_METRICS):
        raise ValueError("quality metric contract mismatch")
    if provenance.get("generation") != generation_binding(generation_identity):
        raise ValueError("quality provenance/generation binding mismatch")
    if provenance.get("offline_environment") != OFFLINE_ENVIRONMENT:
        raise ValueError("quality evaluator offline environment mismatch")
    evaluators = provenance.get("evaluators")
    if not isinstance(evaluators, dict) or set(evaluators) != set(QUALITY_EVALUATORS):
        raise ValueError("quality evaluator set mismatch")
    for name in QUALITY_EVALUATORS:
        validate_evaluator_identity(evaluators[name], name)

    rows: List[Dict[str, object]] = []
    observed = set()
    for row in iter_jsonl(scores_path):
        if set(row) != {
            "schema_version",
            "sample_id",
            "prompt_sha256",
            "condition",
            "cfg_scale",
            "condition_id",
            "audio_sha256",
            "scientific_config_sha256",
            "quality_provenance_sha256",
            "metrics",
        }:
            raise ValueError("quality score field set mismatch")
        if row.get("schema_version") != QUALITY_SCORE_SCHEMA_VERSION:
            raise ValueError("quality score schema mismatch")
        key = (str(row.get("sample_id", "")), str(row.get("condition_id", "")))
        if key in observed:
            raise ValueError("duplicate quality score record {}".format(key))
        generated = expected.get(key)
        if generated is None:
            raise ValueError("quality score is outside generation set: {}".format(key))
        for field in (
            "prompt_sha256",
            "condition",
            "cfg_scale",
            "condition_id",
            "audio_sha256",
            "scientific_config_sha256",
        ):
            if row.get(field) != generated.get(field):
                raise ValueError("quality/generation {} mismatch for {}".format(field, key))
        if row.get("quality_provenance_sha256") != provenance_hash:
            raise ValueError("quality provenance hash mismatch for {}".format(key))
        metrics = row.get("metrics")
        if not isinstance(metrics, dict) or set(metrics) != set(QUALITY_METRICS):
            raise ValueError("quality metric fields mismatch for {}".format(key))
        for metric in QUALITY_METRICS:
            finite_float(metrics[metric], "{} for {}".format(metric, key))
        rows.append(row)
        observed.add(key)
    if observed != set(expected) or len(rows) != len(expected):
        missing = sorted(set(expected) - observed)
        raise ValueError("quality score set is incomplete; missing {}".format(missing[:20]))

    seal = read_json(seal_path)
    if seal.get("schema_version") != QUALITY_SEAL_SCHEMA_VERSION:
        raise ValueError("quality seal schema mismatch")
    expected_seal = {
        "schema_version": QUALITY_SEAL_SCHEMA_VERSION,
        "quality_scores_sha256": sha256_file(scores_path),
        "quality_provenance_sha256": provenance_hash,
        "score_records": len(expected),
        "generation": generation_binding(generation_identity),
    }
    if seal != expected_seal:
        raise ValueError("quality seal payload mismatch")
    return provenance, rows, generation_identity


def build_quality_seal(
    *,
    scores_path: Path,
    provenance_path: Path,
    record_count: int,
    generation_identity: Mapping[str, object],
) -> Dict[str, object]:
    return {
        "schema_version": QUALITY_SEAL_SCHEMA_VERSION,
        "quality_scores_sha256": sha256_file(scores_path),
        "quality_provenance_sha256": sha256_file(provenance_path),
        "score_records": record_count,
        "generation": generation_binding(generation_identity),
    }


def build_external_seal(
    *,
    scores_path: Path,
    provenance_path: Path,
    quality_dir: Path,
    record_count: int,
    generation_identity: Mapping[str, object],
) -> Dict[str, object]:
    return {
        "schema_version": EXTERNAL_SEAL_SCHEMA_VERSION,
        "scores_jsonl_sha256": sha256_file(scores_path),
        "evaluator_provenance_sha256": sha256_file(provenance_path),
        "quality_artifact_seal_sha256": sha256_file(quality_dir / "artifact_seal.json"),
        "score_records": record_count,
        "generation": generation_binding(generation_identity),
    }


def verify_external_artifact(output_dir: Path, generation_dir: Path, quality_dir: Path) -> Dict[str, object]:
    output_dir = require_local_directory(output_dir, "external evaluation artifact")
    expected_names = {"scores.jsonl", "evaluator_provenance.json", "artifact_seal.json"}
    observed_names = {item.name for item in output_dir.iterdir()}
    if observed_names != expected_names:
        raise ValueError("external evaluation artifact members mismatch")
    for name in expected_names:
        path = output_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("external evaluation member is not a regular file: {}".format(path))
    quality_provenance, quality_rows, quality_generation = verify_quality_artifact(
        quality_dir, generation_dir
    )
    matched, inputs = gate.load_and_match_scores(
        generation_dir,
        output_dir / "scores.jsonl",
        output_dir / "evaluator_provenance.json",
    )
    if len(matched) != len(quality_rows):
        raise ValueError("external evaluation/quality record count mismatch")
    external_provenance = inputs["evaluator_provenance"]
    if not isinstance(external_provenance, dict):
        raise ValueError("external evaluator provenance is malformed")
    external_evaluators = external_provenance.get("evaluators")
    quality_evaluators = quality_provenance.get("evaluators")
    if not isinstance(external_evaluators, dict) or not isinstance(quality_evaluators, dict):
        raise ValueError("external/quality evaluator identity set is malformed")
    for evaluator in QUALITY_EVALUATORS:
        if external_evaluators.get(evaluator) != quality_evaluators.get(evaluator):
            raise ValueError(
                "external evaluation changed the sealed {} identity".format(evaluator)
            )
    actual_quality_binding = {
        "artifact_seal_sha256": sha256_file(quality_dir / "artifact_seal.json"),
        "quality_scores_sha256": sha256_file(quality_dir / "quality_scores.jsonl"),
        "quality_provenance_sha256": sha256_file(
            quality_dir / "quality_provenance.json"
        ),
    }
    if external_provenance.get("quality_artifact") != actual_quality_binding:
        raise ValueError("external evaluator provenance/quality artifact binding mismatch")
    if external_provenance.get("generation") != generation_binding(quality_generation):
        raise ValueError("external evaluator provenance/generation binding mismatch")
    if external_provenance.get("protocol") != gate.EXTERNAL_EVALUATION_PROTOCOL:
        raise ValueError("external evaluator protocol mismatch")
    if external_provenance.get("offline_environment") != OFFLINE_ENVIRONMENT:
        raise ValueError("external evaluator offline environment mismatch")
    seal = read_json(output_dir / "artifact_seal.json")
    expected_seal = build_external_seal(
        scores_path=output_dir / "scores.jsonl",
        provenance_path=output_dir / "evaluator_provenance.json",
        quality_dir=quality_dir,
        record_count=len(matched),
        generation_identity=quality_generation,
    )
    if seal != expected_seal:
        raise ValueError("external evaluation seal payload mismatch")
    return {
        "score_records": len(matched),
        "scores_jsonl_sha256": inputs["scores_jsonl_sha256"],
        "evaluator_provenance_sha256": inputs["evaluator_provenance_sha256"],
        "artifact_seal_sha256": sha256_file(output_dir / "artifact_seal.json"),
        "quality_artifact_seal_sha256": actual_quality_binding[
            "artifact_seal_sha256"
        ],
        "quality_scores_sha256": actual_quality_binding["quality_scores_sha256"],
        "quality_provenance_sha256": actual_quality_binding[
            "quality_provenance_sha256"
        ],
        "generation_artifact_seal_sha256": quality_generation[
            "artifact_seal_sha256"
        ],
        "protocol_sha256": sha256_json(gate.EXTERNAL_EVALUATION_PROTOCOL),
        "offline_environment_sha256": sha256_json(OFFLINE_ENVIRONMENT),
    }


enforce_offline_environment()
