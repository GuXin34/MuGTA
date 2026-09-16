"""Closed contracts for Stage-1 MERT diversity and FAD pipeline checks.

The scientific producer lives in ``scripts/eval_stage1_diversity_fad.py``.
This module deliberately contains no torch, transformers, or fadtk import so
that every artifact can be verified on a CPU login node.
"""

from __future__ import annotations

import hashlib
import fnmatch
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .stage1_artifact import (
    Stage1ArtifactError,
    artifact_member,
    canonical_json_sha256,
    load_json_strict,
    regular_tree_files,
    require_finite_number,
    require_sha256,
    sha256_file,
    sha256_tree,
    verify_simple_seal,
)
from .stage1_generation import GENERATION_SEEDS, verify_generation_artifact


FADTK_VERSION = "1.1.0"
FADTK_UPSTREAM_TAG = "1.1.0"
FADTK_UPSTREAM_COMMIT = "6f815282007288a7165d037a95dacd6d783b6f32"
FADTK_SOURCE_SHA256 = (
    "7cffe82dc3e13d1508cdad06bf845aec7fce269e871ed64de5c6210e3d142fe9"
)
FADTK_PYTHON_FILE_COUNT = 10
MERT_MODEL_ID = "m-a-p/MERT-v1-95M"
MERT_REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"
MERT_LAYER = 12
MERT_SAMPLE_RATE = 24_000
MERT_FEATURE_DIM = 768
MERT_SCIENTIFIC_FILE_PINS = {
    "config.json": {
        "size_bytes": 1817,
        "sha256": "ea2627c4c7825cd66f3c944b6b966331604c35928174e0100cd4a82829424e32",
    },
    "configuration_MERT.py": {
        "size_bytes": 5340,
        "sha256": "ae0ec2bab8f59c724ba9878a7c20b67210189536ea62d34a56775968e9decb03",
    },
    "modeling_MERT.py": {
        "size_bytes": 18033,
        "sha256": "6c3ee73cef6f0c30ef494f88d96f891fa6925ffe663fa391b512f4b57abecc6c",
    },
    "preprocessor_config.json": {
        "size_bytes": 211,
        "sha256": "cc5a5e4a5d3b1a758a5ed984b2eaa15bb0522d811d44a9eed82bfca4baa0dc8f",
    },
    "pytorch_model.bin": {
        "size_bytes": 377552987,
        "sha256": "a2b8b747f72c06e0595aeae41ae5473f4364938c6b39b2c58be38c48e6bd3fcd",
    },
}
MERT_FORBIDDEN_WEIGHT_PATTERNS = (
    "model.safetensors",
    "*.index.json",
    "pytorch_model-*.bin",
    "model-*.safetensors",
    "tf_model.h5",
    "flax_model.msgpack",
)
CLAP_BACKEND = "clap-laion-music"
CLAP_CHECKPOINT_BASENAME = "music_audioset_epoch_15_esc_90.14.pt"
CLAP_CHECKPOINT_SHA256 = (
    "fae3e9c087f2909c28a09dc31c8dfcdacbc42ba44c70e972b58c1bd1caf6dedd"
)

MODEL_PINS_REPORT = "model_pins.json"
MODEL_PINS_SCHEMA = "ptc-opd-stage1-diversity-fad-model-pins-v2"
MODEL_PINS_SEAL_SCHEMA = "ptc-opd-stage1-diversity-fad-model-pins-seal-v2"

REFERENCE_SELECTION_DOMAIN = "ptc-opd-stage1-fad-reference-v1"
REFERENCE_SELECTION_SEED = 2701
REFERENCE_COUNT = 256
REFERENCE_SECONDS = 10
REFERENCE_MANIFEST = "reference_manifest.jsonl"
REFERENCE_REPORT = "reference_report.json"
REFERENCE_SCHEMA = "ptc-opd-stage1-fad-reference-v1"
REFERENCE_RECORD_SCHEMA = "ptc-opd-stage1-fad-reference-record-v1"
REFERENCE_SEAL_SCHEMA = "ptc-opd-stage1-fad-reference-seal-v1"

EVALUATION_SUMMARY = "evaluation_summary.json"
EVALUATOR_PROVENANCE = "evaluator_provenance.json"
DIVERSITY_ROWS = "mert_diversity.jsonl"
EVALUATION_SCHEMA = "ptc-opd-stage1-diversity-fad-evaluation-v1"
PROVENANCE_SCHEMA = "ptc-opd-stage1-diversity-fad-provenance-v1"
DIVERSITY_ROW_SCHEMA = "ptc-opd-stage1-mert-diversity-row-v1"
EVALUATION_SEAL_SCHEMA = "ptc-opd-stage1-diversity-fad-seal-v1"
SEAL_NAME = "artifact_seal.json"
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


def _exact_fields(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    expected_set = set(expected)
    observed = set(value)
    if observed != expected_set:
        raise Stage1ArtifactError(
            "{} fields differ; missing={}, unexpected={}".format(
                label,
                sorted(expected_set - observed),
                sorted(observed - expected_set),
            )
        )


def _strict_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise Stage1ArtifactError("JSONL is missing or not regular: {}".format(path))

    def reject_constant(value: str) -> None:
        raise Stage1ArtifactError("JSONL contains non-finite {}".format(value))

    def unique(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Stage1ArtifactError("JSONL contains duplicate key {!r}".format(key))
            result[key] = value
        return result

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.endswith("\n") or not raw.strip():
                raise Stage1ArtifactError(
                    "{}:{} is blank or not newline terminated".format(path, line_number)
                )
            try:
                value = json.loads(
                    raw, object_pairs_hook=unique, parse_constant=reject_constant
                )
            except json.JSONDecodeError as exc:
                raise Stage1ArtifactError(
                    "invalid JSONL at {}:{}".format(path, line_number)
                ) from exc
            if not isinstance(value, dict):
                raise Stage1ArtifactError("JSONL record must be an object")
            rows.append(value)
    return rows


def _dereferenced_tree_files(root: Path) -> Dict[str, Path]:
    """Enumerate a local HF snapshot while allowing its blob symlinks.

    The relative name and dereferenced bytes, never the host-specific symlink
    target, define the identity.  Directory symlinks and broken links remain
    forbidden.
    """

    root = root.expanduser().absolute()
    if root.is_symlink() or not root.is_dir():
        raise Stage1ArtifactError("snapshot root must be a regular directory")
    result: Dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if any(part in {".git", "__pycache__", ".cache"} for part in Path(relative).parts):
            continue
        if path.name == ".DS_Store" or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            if not path.exists() or not path.resolve().is_file():
                raise Stage1ArtifactError("snapshot contains a broken/non-file symlink")
            result[relative] = path.resolve()
        elif path.is_file():
            result[relative] = path
        elif not path.is_dir():
            raise Stage1ArtifactError("snapshot contains a non-regular member")
    if not result:
        raise Stage1ArtifactError("snapshot contains no files")
    return result


def dereferenced_tree_identity(root: Path) -> Dict[str, Any]:
    files = _dereferenced_tree_files(root)
    digest = hashlib.sha256()
    members: List[Dict[str, Any]] = []
    for relative, path in files.items():
        encoded = relative.encode("utf-8")
        size = path.stat().st_size
        file_hash = sha256_file(path)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(file_hash))
        members.append(
            {"relative_path": relative, "sha256": file_hash, "size_bytes": size}
        )
    return {
        "tree_sha256": digest.hexdigest(),
        "file_count": len(members),
        "files": members,
    }


def verify_mert_scientific_snapshot(root: Path) -> Dict[str, Dict[str, Any]]:
    """Rehash the only five MERT files allowed to influence model loading.

    README and other non-scientific metadata may vary across equivalent local
    HF cache materializations.  Alternative Transformers weight names and
    shard indexes are fail-closed so ``pytorch_model.bin`` is the sole loadable
    weight candidate.
    """

    snapshot = root.expanduser().absolute()
    if snapshot.name != MERT_REVISION:
        raise Stage1ArtifactError(
            "MERT snapshot basename must equal frozen revision {}".format(
                MERT_REVISION
            )
        )
    files = _dereferenced_tree_files(snapshot)
    alternatives = []
    for relative in files:
        basename = Path(relative).name
        if (
            (basename == "pytorch_model.bin" and relative != "pytorch_model.bin")
            or any(
                fnmatch.fnmatchcase(basename, pattern)
                for pattern in MERT_FORBIDDEN_WEIGHT_PATTERNS
            )
        ):
            alternatives.append(relative)
    if alternatives:
        raise Stage1ArtifactError(
            "MERT snapshot contains forbidden alternative weight/index files: {}".format(
                sorted(alternatives)
            )
        )

    if set(MERT_SCIENTIFIC_FILE_PINS) != {
        "config.json",
        "configuration_MERT.py",
        "modeling_MERT.py",
        "preprocessor_config.json",
        "pytorch_model.bin",
    }:
        raise Stage1ArtifactError("internal MERT scientific-file pin set differs")
    observed: Dict[str, Dict[str, Any]] = {}
    for relative, expected in MERT_SCIENTIFIC_FILE_PINS.items():
        path = files.get(relative)
        if path is None:
            raise Stage1ArtifactError(
                "MERT snapshot is missing required scientific file {}".format(relative)
            )
        identity = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if identity != expected:
            raise Stage1ArtifactError(
                "MERT scientific file identity differs for {}".format(relative)
            )
        observed[relative] = identity
    return observed


def python_source_identity(package_root: Path) -> Dict[str, Any]:
    package_root = package_root.expanduser().absolute()
    if package_root.is_symlink() or not package_root.is_dir():
        raise Stage1ArtifactError("fadtk package root must be a regular directory")
    files = sorted(package_root.rglob("*.py"))
    if not files:
        raise Stage1ArtifactError("fadtk package contains no Python source")
    digest = hashlib.sha256()
    members = []
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise Stage1ArtifactError("fadtk source must be regular files")
        relative = path.relative_to(package_root).as_posix()
        encoded = relative.encode("utf-8")
        size = path.stat().st_size
        file_hash = sha256_file(path)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(file_hash))
        members.append(
            {"relative_path": relative, "sha256": file_hash, "size_bytes": size}
        )
    return {
        "source_sha256": digest.hexdigest(),
        "python_file_count": len(members),
        "python_files": members,
    }


def _verify_listed_tree_identity(
    identity: Mapping[str, Any],
    *,
    list_field: str,
    count_field: str,
    hash_field: str,
    label: str,
) -> None:
    _exact_fields(identity, {list_field, count_field, hash_field}, label)
    rows = identity.get(list_field)
    if not isinstance(rows, list) or identity.get(count_field) != len(rows) or not rows:
        raise Stage1ArtifactError("{} member count differs".format(label))
    digest = hashlib.sha256()
    previous = None
    for row in rows:
        if not isinstance(row, dict):
            raise Stage1ArtifactError("{} member is malformed".format(label))
        _exact_fields(row, {"relative_path", "sha256", "size_bytes"}, label + " member")
        relative = row.get("relative_path")
        size = row.get("size_bytes")
        file_hash = require_sha256(row.get("sha256"), label + " member SHA-256")
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or "\\" in relative
            or any(part in {"", ".", ".."} for part in relative.split("/"))
            or type(size) is not int
            or size < 0
            or (previous is not None and relative <= previous)
        ):
            raise Stage1ArtifactError("{} member path/order/size differs".format(label))
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(file_hash))
        previous = relative
    if identity.get(hash_field) != digest.hexdigest():
        raise Stage1ArtifactError("{} aggregate identity differs".format(label))


def build_model_pins_report(
    *,
    mert_snapshot: Path,
    clap_checkpoint: Path,
    fadtk_package_root: Path,
    fadtk_version: str,
) -> Dict[str, Any]:
    snapshot = mert_snapshot.expanduser().absolute()
    scientific_files = verify_mert_scientific_snapshot(snapshot)
    snapshot_observation = dereferenced_tree_identity(snapshot)

    checkpoint = clap_checkpoint.expanduser().absolute()
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise Stage1ArtifactError("CLAP checkpoint must be a regular local file")
    if checkpoint.name != CLAP_CHECKPOINT_BASENAME:
        raise Stage1ArtifactError("CLAP checkpoint basename differs")
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != CLAP_CHECKPOINT_SHA256:
        raise Stage1ArtifactError(
            "CLAP checkpoint SHA-256 differs from the accepted republished checkpoint"
        )
    if fadtk_version != FADTK_VERSION:
        raise Stage1ArtifactError("fadtk distribution must be exactly 1.1.0")
    fadtk_source = python_source_identity(fadtk_package_root)
    if (
        fadtk_source["source_sha256"] != FADTK_SOURCE_SHA256
        or fadtk_source["python_file_count"] != FADTK_PYTHON_FILE_COUNT
    ):
        raise Stage1ArtifactError(
            "fadtk source differs from official tag 1.1.0 commit {}".format(
                FADTK_UPSTREAM_COMMIT
            )
        )
    return {
        "schema_version": MODEL_PINS_SCHEMA,
        "status": "complete_local_model_pins",
        "offline_required": True,
        "fadtk": {
            "distribution_version": FADTK_VERSION,
            "upstream_tag": FADTK_UPSTREAM_TAG,
            "upstream_commit": FADTK_UPSTREAM_COMMIT,
            **fadtk_source,
        },
        "mert": {
            "model_id": MERT_MODEL_ID,
            "revision": MERT_REVISION,
            "layer": MERT_LAYER,
            "sample_rate": MERT_SAMPLE_RATE,
            "feature_dim": MERT_FEATURE_DIM,
            "scientific_files": scientific_files,
            "snapshot_observation": snapshot_observation,
        },
        "clap_laion_music": {
            "backend_name": CLAP_BACKEND,
            "checkpoint_basename": CLAP_CHECKPOINT_BASENAME,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_size_bytes": checkpoint.stat().st_size,
        },
    }


def verify_model_pins_artifact(
    directory: Path,
    *,
    mert_snapshot: Optional[Path] = None,
    clap_checkpoint: Optional[Path] = None,
    fadtk_package_root: Optional[Path] = None,
    fadtk_version: Optional[str] = None,
) -> Dict[str, Any]:
    supplied_directory = directory.expanduser()
    if supplied_directory.is_symlink():
        raise Stage1ArtifactError("model-pins artifact root may not be a symlink")
    directory = supplied_directory.resolve(strict=True)
    if not directory.is_dir():
        raise Stage1ArtifactError("model-pins artifact root must be a directory")
    verify_simple_seal(
        directory,
        seal_name=SEAL_NAME,
        schema_version=MODEL_PINS_SEAL_SCHEMA,
        status="complete_local_model_pins",
        payload_names=[MODEL_PINS_REPORT],
    )
    report = load_json_strict(directory / MODEL_PINS_REPORT)
    _exact_fields(
        report,
        {"schema_version", "status", "offline_required", "fadtk", "mert", "clap_laion_music"},
        "model pins report",
    )
    if (
        report.get("schema_version") != MODEL_PINS_SCHEMA
        or report.get("status") != "complete_local_model_pins"
        or report.get("offline_required") is not True
    ):
        raise Stage1ArtifactError("model pins schema/status differs")
    mert = report.get("mert")
    clap = report.get("clap_laion_music")
    fadtk = report.get("fadtk")
    if not isinstance(mert, dict) or not isinstance(clap, dict) or not isinstance(fadtk, dict):
        raise Stage1ArtifactError("model pin components are malformed")
    _exact_fields(
        mert,
        {
            "model_id", "revision", "layer", "sample_rate", "feature_dim",
            "scientific_files", "snapshot_observation",
        },
        "MERT pin",
    )
    _exact_fields(
        clap,
        {
            "backend_name", "checkpoint_basename", "checkpoint_sha256",
            "checkpoint_size_bytes",
        },
        "CLAP pin",
    )
    _exact_fields(
        fadtk,
        {
            "distribution_version", "upstream_tag", "upstream_commit",
            "source_sha256", "python_file_count", "python_files",
        },
        "fadtk pin",
    )
    if (
        mert.get("model_id") != MERT_MODEL_ID
        or mert.get("revision") != MERT_REVISION
        or mert.get("layer") != MERT_LAYER
        or mert.get("sample_rate") != MERT_SAMPLE_RATE
        or mert.get("feature_dim") != MERT_FEATURE_DIM
    ):
        raise Stage1ArtifactError("MERT model contract differs")
    if (
        clap.get("backend_name") != CLAP_BACKEND
        or clap.get("checkpoint_basename") != CLAP_CHECKPOINT_BASENAME
        or clap.get("checkpoint_sha256") != CLAP_CHECKPOINT_SHA256
    ):
        raise Stage1ArtifactError("CLAP model contract differs")
    if (
        fadtk.get("distribution_version") != FADTK_VERSION
        or fadtk.get("upstream_tag") != FADTK_UPSTREAM_TAG
        or fadtk.get("upstream_commit") != FADTK_UPSTREAM_COMMIT
        or fadtk.get("source_sha256") != FADTK_SOURCE_SHA256
        or fadtk.get("python_file_count") != FADTK_PYTHON_FILE_COUNT
    ):
        raise Stage1ArtifactError("fadtk official tag/commit/source pin differs")
    require_sha256(fadtk.get("source_sha256"), "fadtk source SHA-256")
    scientific_files = mert.get("scientific_files")
    if scientific_files != MERT_SCIENTIFIC_FILE_PINS:
        raise Stage1ArtifactError("MERT scientific-file byte pins differ")
    snapshot = mert.get("snapshot_observation")
    if not isinstance(snapshot, dict):
        raise Stage1ArtifactError("MERT snapshot observation is missing")
    _verify_listed_tree_identity(
        snapshot,
        list_field="files",
        count_field="file_count",
        hash_field="tree_sha256",
        label="MERT snapshot",
    )
    _verify_listed_tree_identity(
        {
            "python_files": fadtk["python_files"],
            "python_file_count": fadtk["python_file_count"],
            "source_sha256": fadtk["source_sha256"],
        },
        list_field="python_files",
        count_field="python_file_count",
        hash_field="source_sha256",
        label="fadtk source",
    )
    if type(clap.get("checkpoint_size_bytes")) is not int or clap["checkpoint_size_bytes"] <= 0:
        raise Stage1ArtifactError("CLAP checkpoint size differs")

    if any(
        value is not None
        for value in (mert_snapshot, clap_checkpoint, fadtk_package_root, fadtk_version)
    ):
        if None in (mert_snapshot, clap_checkpoint, fadtk_package_root, fadtk_version):
            raise Stage1ArtifactError("resource rehash requires all four resource arguments")
        observed = build_model_pins_report(
            mert_snapshot=mert_snapshot,  # type: ignore[arg-type]
            clap_checkpoint=clap_checkpoint,  # type: ignore[arg-type]
            fadtk_package_root=fadtk_package_root,  # type: ignore[arg-type]
            fadtk_version=str(fadtk_version),
        )
        observed_mert = dict(observed["mert"])
        sealed_mert = dict(mert)
        # The full cache tree is an operational observation, not a scientific
        # identity.  Only the exact five files above are byte-frozen, so README
        # or model-card changes do not create a false scientific mismatch.
        observed_mert.pop("snapshot_observation")
        sealed_mert.pop("snapshot_observation")
        if (
            observed["fadtk"] != fadtk
            or observed["clap_laion_music"] != clap
            or observed_mert != sealed_mert
        ):
            raise Stage1ArtifactError("live evaluator resources differ from sealed pins")
    return {
        "artifact_seal_sha256": sha256_file(directory / SEAL_NAME),
        "report_sha256": sha256_file(directory / MODEL_PINS_REPORT),
        "report": report,
    }


def reference_selection_hash(track_id: int) -> str:
    payload = "{}|{}|{}".format(
        REFERENCE_SELECTION_DOMAIN, REFERENCE_SELECTION_SEED, int(track_id)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_reference_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if len(rows) != 512:
        raise Stage1ArtifactError("A1-R2 source must contain exactly 512 rows")
    if len({int(row["fma_track_id"]) for row in rows}) != 512:
        raise Stage1ArtifactError("A1-R2 source contains duplicate track IDs")
    for row in rows:
        if (
            row.get("schema_version") != "ptc-opd-fma-calibration-v2"
            or row.get("protocol_label") != "A1-R2"
            or row.get("eligible") is not True
            or row.get("publication_eligible") is not True
            or row.get("segment_num_frames") != 10 * int(row.get("decoded_sample_rate", 0))
        ):
            raise Stage1ArtifactError("A1-R2 reference candidate contract differs")
    ranked = sorted(
        rows,
        key=lambda row: (
            reference_selection_hash(int(row["fma_track_id"])),
            int(row["fma_track_id"]),
        ),
    )
    return [dict(row) for row in ranked[:REFERENCE_COUNT]]


def verify_reference_artifact(
    directory: Path,
    *,
    a1_manifest: Path,
    a1_report: Path,
) -> Dict[str, Any]:
    supplied_directory = directory.expanduser()
    if supplied_directory.is_symlink():
        raise Stage1ArtifactError("FAD reference artifact root may not be a symlink")
    directory = supplied_directory.resolve(strict=True)
    if not directory.is_dir():
        raise Stage1ArtifactError("FAD reference artifact root must be a directory")
    observed_root = {path.name for path in directory.iterdir()}
    expected_root = {REFERENCE_MANIFEST, REFERENCE_REPORT, SEAL_NAME, "audio"}
    if observed_root != expected_root:
        raise Stage1ArtifactError("FAD reference root member set differs")
    audio_root = directory / "audio"
    if audio_root.is_symlink() or not audio_root.is_dir():
        raise Stage1ArtifactError("FAD reference audio root must be regular")
    seal = load_json_strict(directory / SEAL_NAME)
    if (
        seal.get("schema_version") != REFERENCE_SEAL_SCHEMA
        or seal.get("status") != "complete_fad_reference"
        or seal.get("record_count") != REFERENCE_COUNT
        or seal.get("members")
        != {
            REFERENCE_MANIFEST: artifact_member(directory / REFERENCE_MANIFEST),
            REFERENCE_REPORT: artifact_member(directory / REFERENCE_REPORT),
        }
        or seal.get("audio_tree_sha256") != sha256_tree(audio_root)
    ):
        raise Stage1ArtifactError("FAD reference seal differs")

    if a1_manifest.is_symlink() or a1_report.is_symlink():
        raise Stage1ArtifactError("A1-R2 manifest/report may not be symlinks")
    a1_manifest = a1_manifest.resolve(strict=True)
    a1_report = a1_report.resolve(strict=True)
    a1_manifest_hash = sha256_file(a1_manifest)
    a1_report_hash = sha256_file(a1_report)
    a1_rows = _strict_jsonl(a1_manifest)
    selected = select_reference_rows(a1_rows)
    report = load_json_strict(directory / REFERENCE_REPORT)
    _exact_fields(
        report,
        {
            "schema_version", "status", "source", "selection", "record_count",
            "duration_seconds", "manifest", "audio_tree_sha256",
        },
        "FAD reference report",
    )
    if (
        report.get("schema_version") != REFERENCE_SCHEMA
        or report.get("status") != "complete_fad_reference"
        or report.get("record_count") != REFERENCE_COUNT
        or report.get("duration_seconds") != REFERENCE_SECONDS
        or report.get("manifest") != artifact_member(directory / REFERENCE_MANIFEST)
        or report.get("audio_tree_sha256") != sha256_tree(audio_root)
    ):
        raise Stage1ArtifactError("FAD reference report differs")
    source = report.get("source")
    if not isinstance(source, dict) or source != {
        "a1_r2_manifest_basename": a1_manifest.name,
        "a1_r2_manifest_sha256": a1_manifest_hash,
        "a1_r2_report_basename": a1_report.name,
        "a1_r2_report_sha256": a1_report_hash,
        "candidate_count": 512,
    }:
        raise Stage1ArtifactError("FAD reference A1-R2 binding differs")
    selection = report.get("selection")
    if selection != {
        "algorithm": "lowest_sha256_then_track_id",
        "domain": REFERENCE_SELECTION_DOMAIN,
        "seed": REFERENCE_SELECTION_SEED,
        "count": REFERENCE_COUNT,
        "source_population": "A1-R2 512 eligible 10-second segments",
    }:
        raise Stage1ArtifactError("FAD reference selection contract differs")
    source_report = load_json_strict(a1_report)
    if (
        source_report.get("schema_version") != "ptc-opd-fma-calibration-v2"
        or source_report.get("protocol_label") != "A1-R2"
        or source_report.get("manifest_sha256") != a1_manifest_hash
        or source_report.get("selected_tracks") != 512
        or source_report.get("publication_eligible") is not True
    ):
        raise Stage1ArtifactError("A1-R2 report does not bind the 512-row source")

    rows = _strict_jsonl(directory / REFERENCE_MANIFEST)
    if len(rows) != REFERENCE_COUNT:
        raise Stage1ArtifactError("FAD reference must contain 256 records")
    expected_audio = set()
    for index, (row, source_row) in enumerate(zip(rows, selected)):
        _exact_fields(
            row,
            {
                "schema_version", "selection_rank", "selection_hash", "fma_track_id",
                "relative_audio_path", "source_audio_sha256", "source_pcm_sha256",
                "segment_start_frame", "segment_num_frames", "sample_rate", "channels",
                "duration_seconds", "path", "audio_sha256", "audio_frames",
                "audio_channels", "audio_sample_rate", "audio_subtype",
            },
            "FAD reference row",
        )
        track_id = int(source_row["fma_track_id"])
        expected_path = "audio/{:06d}.wav".format(track_id)
        if (
            row.get("schema_version") != REFERENCE_RECORD_SCHEMA
            or row.get("selection_rank") != index
            or row.get("selection_hash") != reference_selection_hash(track_id)
            or row.get("fma_track_id") != track_id
            or row.get("relative_audio_path") != source_row["relative_audio_path"]
            or row.get("source_audio_sha256") != source_row["source_audio_sha256"]
            or row.get("source_pcm_sha256") != source_row["extracted_pcm_sha256"]
            or row.get("segment_start_frame") != source_row["segment_start_frame"]
            or row.get("segment_num_frames") != source_row["segment_num_frames"]
            or row.get("sample_rate") != source_row["decoded_sample_rate"]
            or row.get("channels") != source_row["decoded_channels"]
            or row.get("duration_seconds") != REFERENCE_SECONDS
            or row.get("path") != expected_path
            or row.get("audio_frames") != source_row["segment_num_frames"]
            or row.get("audio_channels") != source_row["decoded_channels"]
            or row.get("audio_sample_rate") != source_row["decoded_sample_rate"]
            or row.get("audio_subtype") != "FLOAT"
        ):
            raise Stage1ArtifactError("FAD reference row/source binding differs")
        audio_path = directory / expected_path
        if audio_path.is_symlink() or not audio_path.is_file():
            raise Stage1ArtifactError("FAD reference WAV is missing")
        if row.get("audio_sha256") != sha256_file(audio_path):
            raise Stage1ArtifactError("FAD reference WAV hash differs")
        expected_audio.add(expected_path)
    actual_audio = {
        path.relative_to(directory).as_posix()
        for path in audio_root.rglob("*")
        if path.is_file()
    }
    if any(path.is_dir() for path in audio_root.rglob("*")):
        raise Stage1ArtifactError("FAD reference audio tree must be flat")
    if actual_audio != expected_audio:
        raise Stage1ArtifactError("FAD reference WAV set differs")
    return {
        "artifact_seal_sha256": sha256_file(directory / SEAL_NAME),
        "report_sha256": sha256_file(directory / REFERENCE_REPORT),
        "manifest_sha256": sha256_file(directory / REFERENCE_MANIFEST),
        "rows": rows,
        "report": report,
    }


def _sample_key(row: Mapping[str, Any]) -> Tuple[str, int]:
    sample_id = row.get("sample_id")
    seed = row.get("generation_seed")
    if not isinstance(sample_id, str) or type(seed) is not int:
        raise Stage1ArtifactError("generated sample key is malformed")
    return sample_id, seed


def verify_diversity_fad_artifact(
    directory: Path,
    *,
    generation_dir: Path,
    eval_manifest_dir: Path,
    reference_dir: Path,
    a1_manifest: Path,
    a1_report: Path,
    model_pins_dir: Path,
) -> Dict[str, Any]:
    supplied_directory = directory.expanduser()
    if supplied_directory.is_symlink():
        raise Stage1ArtifactError("diversity/FAD artifact root may not be a symlink")
    directory = supplied_directory.resolve(strict=True)
    if not directory.is_dir():
        raise Stage1ArtifactError("diversity/FAD artifact root must be a directory")
    for label, root in (
        ("generation", generation_dir),
        ("evaluation manifest", eval_manifest_dir),
        ("reference", reference_dir),
        ("model pins", model_pins_dir),
    ):
        if root.expanduser().is_symlink():
            raise Stage1ArtifactError("{} root may not be a symlink".format(label))
    payload_names = [DIVERSITY_ROWS, EVALUATION_SUMMARY, EVALUATOR_PROVENANCE]
    verify_simple_seal(
        directory,
        seal_name=SEAL_NAME,
        schema_version=EVALUATION_SEAL_SCHEMA,
        status="complete_diversity_fad_pipeline_check",
        payload_names=payload_names,
    )
    generation = verify_generation_artifact(
        generation_dir, eval_manifest_dir=eval_manifest_dir, rehash_pcm=False
    )
    reference = verify_reference_artifact(
        reference_dir, a1_manifest=a1_manifest, a1_report=a1_report
    )
    pins = verify_model_pins_artifact(model_pins_dir)
    provenance = load_json_strict(directory / EVALUATOR_PROVENANCE)
    _exact_fields(
        provenance,
        {
            "schema_version", "status", "generation_artifact_seal_sha256",
            "reference_artifact_seal_sha256", "model_pins_artifact_seal_sha256",
            "offline_environment", "network_access_forbidden", "evaluator",
            "generation_integrity", "reference_integrity", "resource_integrity",
            "staging_policy",
        },
        "diversity/FAD provenance",
    )
    if (
        provenance.get("schema_version") != PROVENANCE_SCHEMA
        or provenance.get("status") != "accepted_offline_evaluation"
        or provenance.get("generation_artifact_seal_sha256")
        != generation["artifact_seal_sha256"]
        or provenance.get("reference_artifact_seal_sha256")
        != reference["artifact_seal_sha256"]
        or provenance.get("model_pins_artifact_seal_sha256")
        != pins["artifact_seal_sha256"]
        or provenance.get("network_access_forbidden") is not True
    ):
        raise Stage1ArtifactError("diversity/FAD provenance binding differs")
    staging_policy = provenance.get("staging_policy")
    if staging_policy != {
        "generation_audio_copied_to_independent_staging": True,
        "reference_audio_copied_to_independent_staging": True,
        "clap_checkpoint_copied_to_independent_staging": True,
        "fadtk_cache_roots_inside_staging_only": True,
        "generation_artifact_mutation_forbidden": True,
    }:
        raise Stage1ArtifactError("FAD staging policy differs")
    generation_integrity = provenance.get("generation_integrity")
    if not isinstance(generation_integrity, dict):
        raise Stage1ArtifactError("generation integrity record is missing")
    canonical_generation_dir = Path(generation["directory"])
    current_generation_tree = sha256_tree(canonical_generation_dir)
    if (
        generation_integrity.get("before_tree_sha256") != current_generation_tree
        or generation_integrity.get("after_tree_sha256") != current_generation_tree
        or generation_integrity.get("unchanged") is not True
    ):
        raise Stage1ArtifactError("generation artifact changed during/after evaluation")
    if provenance.get("offline_environment") != OFFLINE_ENVIRONMENT:
        raise Stage1ArtifactError("offline environment contract differs")
    evaluator = provenance.get("evaluator")
    if not isinstance(evaluator, dict) or set(evaluator) != {
        "fadtk", "mert", "clap_laion_music", "runtime_device"
    }:
        raise Stage1ArtifactError("evaluator provenance component set differs")
    if evaluator.get("fadtk") != pins["report"]["fadtk"]:
        raise Stage1ArtifactError("runtime fadtk identity differs from model pins")
    mert_evaluator = evaluator.get("mert")
    clap_evaluator = evaluator.get("clap_laion_music")
    if (
        not isinstance(mert_evaluator, dict)
        or mert_evaluator.get("model_id") != MERT_MODEL_ID
        or mert_evaluator.get("revision") != MERT_REVISION
        or mert_evaluator.get("layer") != MERT_LAYER
        or not isinstance(clap_evaluator, dict)
        or clap_evaluator.get("backend") != CLAP_BACKEND
        or clap_evaluator.get("checkpoint_sha256") != CLAP_CHECKPOINT_SHA256
        or clap_evaluator.get("checkpoint_loaded_from_staging_copy") is not True
    ):
        raise Stage1ArtifactError("runtime evaluator identity differs")
    reference_integrity = provenance.get("reference_integrity")
    current_reference_tree = sha256_tree(Path(reference_dir))
    if not isinstance(reference_integrity, dict) or reference_integrity != {
        "before_tree_sha256": current_reference_tree,
        "after_tree_sha256": current_reference_tree,
        "unchanged": True,
    }:
        raise Stage1ArtifactError("reference artifact changed during evaluation")
    resource_integrity = provenance.get("resource_integrity")
    if not isinstance(resource_integrity, dict) or set(resource_integrity) != {
        "mert_snapshot_before_tree_sha256",
        "mert_snapshot_after_tree_sha256",
        "clap_checkpoint_before_sha256",
        "clap_checkpoint_after_sha256",
        "unchanged",
    }:
        raise Stage1ArtifactError("model resources changed during evaluation")
    mert_before = require_sha256(
        resource_integrity.get("mert_snapshot_before_tree_sha256"),
        "runtime MERT snapshot before tree",
    )
    mert_after = require_sha256(
        resource_integrity.get("mert_snapshot_after_tree_sha256"),
        "runtime MERT snapshot after tree",
    )
    if (
        mert_before != mert_after
        or resource_integrity.get("clap_checkpoint_before_sha256")
        != CLAP_CHECKPOINT_SHA256
        or resource_integrity.get("clap_checkpoint_after_sha256")
        != CLAP_CHECKPOINT_SHA256
        or resource_integrity.get("unchanged") is not True
    ):
        raise Stage1ArtifactError("model resources changed during evaluation")

    provenance_hash = sha256_file(directory / EVALUATOR_PROVENANCE)
    rows = _strict_jsonl(directory / DIVERSITY_ROWS)
    # Ruling #9 §6 self-heal (post-Milestone-7): 128-prompt pilot vs
    # dynamic test-set count.  Dispatch by eval_manifest_dir basename.
    _emd = eval_manifest_dir.expanduser().resolve()
    if _emd.name == "test_prompts" or _emd.name.startswith("test_prompts"):
        _tp_manifest = _emd / "test_prompts_manifest.json"
        with _tp_manifest.open("r", encoding="utf-8") as _h:
            _tp_meta = json.load(_h)
        _expected_prompts = len(
            _tp_meta.get("selection", {}).get("sample_ids", [])
        )
        if _expected_prompts < 1:
            raise Stage1ArtifactError(
                "test_prompts manifest has no sample_ids"
            )
    else:
        _expected_prompts = 128
    _expected_gen_samples = _expected_prompts * 2  # 2 generation seeds per prompt
    if len(rows) != _expected_prompts:
        raise Stage1ArtifactError(
            "MERT diversity must contain {} prompt pairs (got {})".format(
                _expected_prompts, len(rows)
            )
        )
    generated_by_key = {_sample_key(row): row for row in generation["samples"]}
    expected_ids = sorted({sample_id for sample_id, _ in generated_by_key})
    if len(expected_ids) != _expected_prompts or len(generated_by_key) != _expected_gen_samples:
        raise Stage1ArtifactError(
            "generation is not an exact {}x2 plan".format(_expected_prompts)
        )
    distances: List[float] = []
    for row, sample_id in zip(rows, expected_ids):
        _exact_fields(
            row,
            {
                "schema_version", "sample_id", "condition_id", "generation_seeds",
                "audio_sha256_by_seed", "embedding_sha256_by_seed",
                "embedding_frame_count_by_seed", "model_id", "layer", "pooling",
                "normalization", "cosine_similarity", "cosine_distance",
                "evaluator_provenance_sha256",
            },
            "MERT diversity row",
        )
        first = generated_by_key[(sample_id, GENERATION_SEEDS[0])]
        second = generated_by_key[(sample_id, GENERATION_SEEDS[1])]
        if (
            row.get("schema_version") != DIVERSITY_ROW_SCHEMA
            or row.get("sample_id") != sample_id
            or row.get("condition_id") != first.get("condition_id")
            or first.get("condition_id") != second.get("condition_id")
            or row.get("generation_seeds") != list(GENERATION_SEEDS)
            or row.get("audio_sha256_by_seed")
            != {
                str(GENERATION_SEEDS[0]): first.get("audio_sha256"),
                str(GENERATION_SEEDS[1]): second.get("audio_sha256"),
            }
            or row.get("model_id") != MERT_MODEL_ID
            or row.get("layer") != MERT_LAYER
            or row.get("pooling") != "arithmetic_mean_over_frame_axis"
            or row.get("normalization") != "L2_after_frame_mean"
            or row.get("evaluator_provenance_sha256") != provenance_hash
        ):
            raise Stage1ArtifactError("MERT diversity row source/metric contract differs")
        similarity = require_finite_number(row.get("cosine_similarity"), "cosine similarity")
        distance = require_finite_number(row.get("cosine_distance"), "cosine distance")
        if not -1.000001 <= similarity <= 1.000001 or not -0.000001 <= distance <= 2.000001:
            raise Stage1ArtifactError("cosine result lies outside numerical bounds")
        if not math.isclose(distance, 1.0 - similarity, rel_tol=0.0, abs_tol=1.0e-10):
            raise Stage1ArtifactError("cosine distance is not 1-similarity")
        frame_counts = row.get("embedding_frame_count_by_seed")
        embedding_hashes = row.get("embedding_sha256_by_seed")
        if (
            not isinstance(frame_counts, dict)
            or set(frame_counts) != {str(seed) for seed in GENERATION_SEEDS}
            or any(type(value) is not int or value <= 0 for value in frame_counts.values())
            or not isinstance(embedding_hashes, dict)
            or set(embedding_hashes) != {str(seed) for seed in GENERATION_SEEDS}
        ):
            raise Stage1ArtifactError("MERT embedding identity/count map differs")
        for value in embedding_hashes.values():
            require_sha256(value, "MERT embedding SHA-256")
        distances.append(distance)

    summary = load_json_strict(directory / EVALUATION_SUMMARY)
    _exact_fields(
        summary,
        {
            "schema_version", "status", "condition_id", "generation_artifact_seal_sha256",
            "reference_artifact_seal_sha256", "model_pins_artifact_seal_sha256",
            "evaluator_provenance_sha256", "mert_diversity", "fad_pipeline_checks",
            "selection_contract",
        },
        "diversity/FAD summary",
    )
    condition_ids = {str(row.get("condition_id")) for row in generation["samples"]}
    if len(condition_ids) != 1:
        raise Stage1ArtifactError("one generation artifact must contain one condition")
    if (
        summary.get("schema_version") != EVALUATION_SCHEMA
        or summary.get("status") != "complete_diversity_fad_pipeline_check"
        or summary.get("condition_id") != next(iter(condition_ids))
        or summary.get("generation_artifact_seal_sha256")
        != generation["artifact_seal_sha256"]
        or summary.get("reference_artifact_seal_sha256")
        != reference["artifact_seal_sha256"]
        or summary.get("model_pins_artifact_seal_sha256") != pins["artifact_seal_sha256"]
        or summary.get("evaluator_provenance_sha256") != provenance_hash
    ):
        raise Stage1ArtifactError("diversity/FAD summary source binding differs")
    diversity = summary.get("mert_diversity")
    # Ruling #9 §6 self-heal: use _expected_prompts computed above (128 for
    # pilot, dynamic for test_prompts).
    if not isinstance(diversity, dict) or diversity.get("prompt_pair_count") != _expected_prompts:
        raise Stage1ArtifactError(
            "MERT diversity summary differs (expected prompt_pair_count={})".format(
                _expected_prompts
            )
        )
    _exact_fields(
        diversity,
        {
            "model_id", "layer", "prompt_pair_count", "pairing", "pooling",
            "mean_cosine_distance", "min_cosine_distance", "max_cosine_distance",
            "selection_role",
        },
        "MERT diversity summary",
    )
    observed_mean = require_finite_number(
        diversity.get("mean_cosine_distance"), "mean MERT diversity"
    )
    expected_mean = math.fsum(distances) / len(distances)
    if not math.isclose(observed_mean, expected_mean, rel_tol=0.0, abs_tol=1.0e-12):
        raise Stage1ArtifactError("MERT diversity mean differs from rows")
    if (
        diversity.get("model_id") != MERT_MODEL_ID
        or diversity.get("layer") != MERT_LAYER
        or diversity.get("pairing") != "same_prompt_seed31001_vs_seed31002"
        or diversity.get("pooling") != "frame_mean_then_L2"
        or diversity.get("selection_role") != "pilot_point_estimate_gate"
    ):
        raise Stage1ArtifactError("MERT diversity scientific contract differs")
    fad = summary.get("fad_pipeline_checks")
    if not isinstance(fad, dict) or set(fad) != {CLAP_BACKEND, "MERT-v1-95M-layer12"}:
        raise Stage1ArtifactError("FAD backend set differs")
    for backend, value in fad.items():
        if not isinstance(value, dict):
            raise Stage1ArtifactError("FAD backend result is malformed")
        _exact_fields(
            value,
            {
                "score", "finite", "pipeline_check_passed",
                "selection_use_forbidden", "paper_claim_use_forbidden",
            },
            "FAD backend result",
        )
        require_finite_number(value.get("score"), "{} FAD".format(backend))
        if (
            value.get("finite") is not True
            or value.get("pipeline_check_passed") is not True
            or value.get("selection_use_forbidden") is not True
            or value.get("paper_claim_use_forbidden") is not True
        ):
            raise Stage1ArtifactError("FAD is not a finite non-selection pipeline check")
    if summary.get("selection_contract") != {
        "mert_diversity_used_only_as_predeclared_pilot_point_estimate": True,
        "fad_used_for_model_or_checkpoint_selection": False,
        "fad_used_for_paper_claim": False,
    }:
        raise Stage1ArtifactError("FAD selection redline differs")
    return {
        "artifact_seal_sha256": sha256_file(directory / SEAL_NAME),
        "summary_sha256": sha256_file(directory / EVALUATION_SUMMARY),
        "summary": summary,
        "rows": rows,
        "provenance": provenance,
        "mert_diversity_mean_cosine_distance": observed_mean,
        "fad_pipeline_check_passed": True,
        "fad_scores": {key: float(value["score"]) for key, value in fad.items()},
        "generation": generation,
    }
