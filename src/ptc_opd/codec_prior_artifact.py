"""Strict consumer for one sealed Phase-A1 codec-prior artifact directory.

The producer publishes three payloads and ``ARTIFACT_SEAL.json`` with one
directory-level rename.  This module is the corresponding fail-closed reader:
it accepts only that closed four-member set, rehashes every payload, validates
the frozen scientific contract, and returns a portable identity together with
the normalized four-codebook prior.

No AudioCraft, CUDA, checkpoint, or source-audio import is needed here.  Paths
inside the seal are provenance strings only; callers may therefore copy the
sealed directory to another machine without retaining the producer's original
absolute paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from pathlib import PurePosixPath
import re
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple


ARTIFACT_SEAL_SCHEMA = "ptc-opd-codec-artifact-seal-v3"
RESULT_SCHEMA = "ptc-opd-codec-prior-v3"
MANIFEST_SCHEMA = "ptc-opd-fma-calibration-v2"
IDENTITY_SCHEMA = "ptc-opd-codec-prior-artifact-identity-v3"
IMPLEMENTATION_IDENTITY_SCHEMA = "ptc-opd-a1-r2-implementation-v1"
PROTOCOL_LABEL = "A1-R2"

ARTIFACT_SEAL_NAME = "ARTIFACT_SEAL.json"
PER_CLIP_NAME = "codec_prior.per_clip.jsonl.gz"
SUMMARY_NAME = "codec_prior.summary.csv"
PRIOR_NAME = "codec_prior.json"
PAYLOAD_NAMES: Tuple[str, ...] = (PER_CLIP_NAME, SUMMARY_NAME, PRIOR_NAME)
DIRECTORY_MEMBERS = frozenset(PAYLOAD_NAMES + (ARTIFACT_SEAL_NAME,))

EXPECTED_CLIPS = 512
EXPECTED_CODEBOOKS = 4
EXPECTED_ARTIFACT_DIRECTORY_BASENAME = "phase_a1_codec_prior_r2"
EXPECTED_MANIFEST_BASENAME = "codec_calibration.train.jsonl"
EXPECTED_MANIFEST_REPORT_BASENAME = "codec_calibration.train.report.json"
EXPECTED_MANIFEST_PARENT = "a1-r2"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 4701
REFERENCE_NORM_MINIMUM = 1.0e-7
SCIENTIFIC_KENDALL_THRESHOLD = 2.0 / 3.0
SCIENTIFIC_TV_LOWER_THRESHOLD = 0.05
MAX_LEAVE_ONE_OUT_TV = 0.05
TRIM_EACH_TAIL_AT_N512 = 5
TRIMMED_RETAINED_AT_N512 = 502
REQUIRED_ARCHIVE_SHA1 = "ade154f733639d52e35e32f5593efe5be76c6d70"
PINNED_AUDIOCRAFT_BASE_COMMIT = "896ec7c47f5e5d1e5aa1e4b260c4405328bf009d"
AUDIOCRAFT_SOURCE_SCHEMA = "ptc-opd-audiocraft-source-tree-v1"
ALLOWED_PRETRAINED_CODEC_ID = "facebook/encodec_32khz"
EXPECTED_IMPLEMENTATION_PATHS = (
    "scripts/build_fma_calibration_manifest.py",
    "scripts/estimate_codec_prior.py",
    "src/ptc_opd/perceptual_prior.py",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class CodecPriorArtifactError(ValueError):
    """The supplied directory is not one complete formal A1 artifact."""


@dataclass(frozen=True)
class CodecPriorArtifact:
    """Verified A1 prior and the immutable identities that produced it.

    ``prior`` must already be the canonical normalized float64 payload and is
    cross-checked against the frozen marginal arithmetic.  It is returned as a
    tuple so this common consumer remains independent of NumPy and PyTorch;
    model code can use
    ``torch.as_tensor(result.prior)`` at its own device/dtype boundary.
    """

    directory: str
    prior: Tuple[float, ...]
    manifest_sha256: str
    checkpoint_sha256: str
    ordered_source_records_sha256: str
    codec_prior_sha256: str
    artifact_seal_sha256: str
    codec_load: Mapping[str, Any]
    identity: Mapping[str, Any]
    identity_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
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


def _reject_json_constant(value: str) -> None:
    raise CodecPriorArtifactError("JSON contains forbidden non-finite value {}".format(value))


def _unique_json_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CodecPriorArtifactError("JSON contains duplicate key {!r}".format(key))
        value[key] = item
    return value


def _load_json_object(path: Path, label: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CodecPriorArtifactError("{} is missing or not a regular file".format(label))
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(
                stream,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
    except CodecPriorArtifactError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CodecPriorArtifactError("{} is not valid UTF-8 JSON".format(label)) from exc
    if not isinstance(value, dict):
        raise CodecPriorArtifactError("{} must contain one JSON object".format(label))
    return value


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CodecPriorArtifactError("{} must be a JSON object".format(label))
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: Sequence[str], label: str
) -> None:
    expected_set = set(expected)
    observed = set(value)
    if observed != expected_set:
        raise CodecPriorArtifactError(
            "{} keys differ; missing={}, unexpected={}".format(
                label,
                sorted(expected_set - observed),
                sorted(observed - expected_set, key=str),
            )
        )


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CodecPriorArtifactError(
            "{} must be a lowercase SHA-256 hex string".format(label)
        )
    return value


def _require_int(value: Any, expected: int, label: str) -> int:
    if type(value) is not int or value != expected:
        raise CodecPriorArtifactError("{} must equal {}".format(label, expected))
    return value


def _require_number(value: Any, expected: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CodecPriorArtifactError("{} must be numeric".format(label))
    resolved = float(value)
    if not math.isfinite(resolved) or resolved != float(expected):
        raise CodecPriorArtifactError("{} must equal {}".format(label, expected))
    return resolved


def _require_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CodecPriorArtifactError("{} must be numeric".format(label))
    resolved = float(value)
    if not math.isfinite(resolved):
        raise CodecPriorArtifactError("{} must be finite".format(label))
    return resolved


def _require_bool(value: Any, expected: bool, label: str) -> bool:
    if type(value) is not bool or value is not expected:
        raise CodecPriorArtifactError("{} must be {}".format(label, str(expected).lower()))
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CodecPriorArtifactError("{} must be a non-empty string".format(label))
    return value


def _validate_manifest(value: Any) -> Dict[str, Any]:
    manifest = dict(_require_mapping(value, "seal.manifest"))
    _require_exact_keys(
        manifest, ("schema_version", "path", "sha256", "items"), "seal.manifest"
    )
    if manifest["schema_version"] != MANIFEST_SCHEMA:
        raise CodecPriorArtifactError("seal manifest schema mismatch")
    manifest_path = _require_nonempty_string(
        manifest["path"], "seal.manifest.path"
    )
    pure_manifest = PurePosixPath(manifest_path)
    if (
        pure_manifest.name != EXPECTED_MANIFEST_BASENAME
        or pure_manifest.parent.name != EXPECTED_MANIFEST_PARENT
    ):
        raise CodecPriorArtifactError(
            "seal manifest path must end in a1-r2/codec_calibration.train.jsonl"
        )
    manifest["sha256"] = _require_sha256(
        manifest["sha256"], "seal.manifest.sha256"
    )
    _require_int(manifest["items"], EXPECTED_CLIPS, "seal.manifest.items")
    return manifest


def _validate_checkpoint(value: Any) -> Dict[str, Any]:
    checkpoint = dict(_require_mapping(value, "seal.checkpoint"))
    _require_exact_keys(
        checkpoint,
        (
            "path",
            "payload_path",
            "sha256",
            "load_mode",
            "pretrained_model_id",
            "resolved_snapshot_revision",
        ),
        "seal.checkpoint",
    )
    _require_nonempty_string(checkpoint["path"], "seal.checkpoint.path")
    _require_nonempty_string(
        checkpoint["payload_path"], "seal.checkpoint.payload_path"
    )
    checkpoint["sha256"] = _require_sha256(
        checkpoint["sha256"], "seal.checkpoint.sha256"
    )
    if checkpoint["load_mode"] not in {"self_contained", "pretrained_indirection"}:
        raise CodecPriorArtifactError("seal checkpoint load_mode is not recognized")
    for key in ("pretrained_model_id", "resolved_snapshot_revision"):
        if checkpoint[key] is not None and not isinstance(checkpoint[key], str):
            raise CodecPriorArtifactError("seal.checkpoint.{} must be string/null".format(key))
    if checkpoint["load_mode"] == "self_contained":
        if (
            checkpoint["pretrained_model_id"] is not None
            or checkpoint["resolved_snapshot_revision"] is not None
        ):
            raise CodecPriorArtifactError(
                "self-contained checkpoint must not name a pretrained snapshot"
            )
    else:
        if checkpoint["pretrained_model_id"] != ALLOWED_PRETRAINED_CODEC_ID:
            raise CodecPriorArtifactError("pretrained codec model ID mismatch")
        revision = checkpoint["resolved_snapshot_revision"]
        if not isinstance(revision, str) or _COMMIT_RE.fullmatch(revision) is None:
            raise CodecPriorArtifactError(
                "pretrained codec snapshot revision must be a 40-hex commit"
            )
    return checkpoint


def _validate_snapshot_relative_path(value: Any, label: str) -> str:
    path = _require_nonempty_string(value, label)
    pure = PurePosixPath(path)
    if pure.is_absolute() or path != pure.as_posix() or any(
        part in {"", ".", ".."} for part in pure.parts
    ):
        raise CodecPriorArtifactError("{} must be a safe POSIX relative path".format(label))
    return path


def _validate_portable_codec_load(value: Any, label: str) -> Dict[str, Any]:
    codec_load = dict(_require_mapping(value, label))
    _require_exact_keys(
        codec_load,
        (
            "load_mode",
            "pretrained_model_id",
            "resolved_snapshot_revision",
            "resolved_snapshot_files",
        ),
        label,
    )
    mode = codec_load["load_mode"]
    if mode not in {"self_contained", "pretrained_indirection"}:
        raise CodecPriorArtifactError("{}.load_mode is not recognized".format(label))
    model_id = codec_load["pretrained_model_id"]
    revision = codec_load["resolved_snapshot_revision"]
    raw_files = codec_load["resolved_snapshot_files"]
    if not isinstance(raw_files, list):
        raise CodecPriorArtifactError("{}.resolved_snapshot_files must be a list".format(label))
    files = []
    for index, raw in enumerate(raw_files):
        file_label = "{}.resolved_snapshot_files[{}]".format(label, index)
        item = dict(_require_mapping(raw, file_label))
        _require_exact_keys(item, ("relative_path", "size_bytes", "sha256"), file_label)
        relative_path = _validate_snapshot_relative_path(
            item["relative_path"], "{}.relative_path".format(file_label)
        )
        size = item["size_bytes"]
        if type(size) is not int or size < 0:
            raise CodecPriorArtifactError(
                "{}.size_bytes must be a non-negative integer".format(file_label)
            )
        files.append(
            {
                "relative_path": relative_path,
                "size_bytes": size,
                "sha256": _require_sha256(
                    item["sha256"], "{}.sha256".format(file_label)
                ),
            }
        )
    if files != sorted(files, key=lambda item: item["relative_path"]):
        raise CodecPriorArtifactError("{} snapshot files are not canonically ordered".format(label))
    if len({item["relative_path"] for item in files}) != len(files):
        raise CodecPriorArtifactError("{} snapshot contains duplicate paths".format(label))
    if mode == "self_contained":
        if model_id is not None or revision is not None or files:
            raise CodecPriorArtifactError(
                "{} self-contained codec must not contain snapshot provenance".format(label)
            )
    else:
        if model_id != ALLOWED_PRETRAINED_CODEC_ID:
            raise CodecPriorArtifactError("{} pretrained codec model ID mismatch".format(label))
        if not isinstance(revision, str) or _COMMIT_RE.fullmatch(revision) is None:
            raise CodecPriorArtifactError(
                "{} snapshot revision must be a 40-hex commit".format(label)
            )
        if not files:
            raise CodecPriorArtifactError("{} pretrained snapshot file list is empty".format(label))
    return {
        "load_mode": mode,
        "pretrained_model_id": model_id,
        "resolved_snapshot_revision": revision,
        "resolved_snapshot_files": files,
    }


def _portable_codec_load_from_prior(value: Any) -> Dict[str, Any]:
    label = "codec_prior.codec_load"
    raw = dict(_require_mapping(value, label))
    _require_exact_keys(
        raw,
        (
            "load_mode",
            "pretrained_model_id",
            "resolved_snapshot_root",
            "resolved_snapshot_revision",
            "resolved_snapshot_files",
        ),
        label,
    )
    root = raw["resolved_snapshot_root"]
    if root is not None and not isinstance(root, str):
        raise CodecPriorArtifactError("codec_prior.codec_load snapshot root must be string/null")
    raw_files = raw["resolved_snapshot_files"]
    if not isinstance(raw_files, list):
        raise CodecPriorArtifactError("codec_prior.codec_load snapshot files must be a list")
    portable_files = []
    for index, raw_file in enumerate(raw_files):
        file_label = "codec_prior.codec_load.resolved_snapshot_files[{}]".format(index)
        item = dict(_require_mapping(raw_file, file_label))
        _require_exact_keys(
            item,
            (
                "relative_path",
                "visible_absolute_path",
                "resolved_absolute_path",
                "is_symlink",
                "size_bytes",
                "sha256",
            ),
            file_label,
        )
        for key in ("visible_absolute_path", "resolved_absolute_path"):
            _require_nonempty_string(item[key], "{}.{}".format(file_label, key))
        if type(item["is_symlink"]) is not bool:
            raise CodecPriorArtifactError("{}.is_symlink must be boolean".format(file_label))
        portable_files.append(
            {
                "relative_path": item["relative_path"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
        )
    return _validate_portable_codec_load(
        {
            "load_mode": raw["load_mode"],
            "pretrained_model_id": raw["pretrained_model_id"],
            "resolved_snapshot_revision": raw["resolved_snapshot_revision"],
            "resolved_snapshot_files": portable_files,
        },
        label,
    )


def _validate_source_identity(value: Any) -> Dict[str, Any]:
    source = dict(_require_mapping(value, "seal.source_identity"))
    _require_exact_keys(
        source,
        (
            "source_root",
            "archive_sha1",
            "items",
            "ordered_source_records_sha256",
            "manifest_report_sha256",
        ),
        "seal.source_identity",
    )
    _require_nonempty_string(source["source_root"], "seal.source_identity.source_root")
    if source["archive_sha1"] != REQUIRED_ARCHIVE_SHA1:
        raise CodecPriorArtifactError("seal source archive SHA-1 mismatch")
    _require_int(source["items"], EXPECTED_CLIPS, "seal.source_identity.items")
    source["ordered_source_records_sha256"] = _require_sha256(
        source["ordered_source_records_sha256"],
        "seal.source_identity.ordered_source_records_sha256",
    )
    source["manifest_report_sha256"] = _require_sha256(
        source["manifest_report_sha256"],
        "seal.source_identity.manifest_report_sha256",
    )
    return source


def _validate_protocol(value: Any) -> Dict[str, Any]:
    protocol = dict(_require_mapping(value, "seal.protocol"))
    _require_exact_keys(
        protocol,
        (
            "identity_schema",
            "label",
            "manifest_schema",
            "result_schema",
            "artifact_seal_schema",
            "manifest_report",
            "implementation",
        ),
        "seal.protocol",
    )
    expected_scalars = {
        "identity_schema": IDENTITY_SCHEMA,
        "label": PROTOCOL_LABEL,
        "manifest_schema": MANIFEST_SCHEMA,
        "result_schema": RESULT_SCHEMA,
        "artifact_seal_schema": ARTIFACT_SEAL_SCHEMA,
    }
    for key, expected in expected_scalars.items():
        if protocol[key] != expected:
            raise CodecPriorArtifactError(
                "seal protocol {} mismatch".format(key)
            )

    report = dict(
        _require_mapping(protocol["manifest_report"], "seal.protocol.manifest_report")
    )
    _require_exact_keys(
        report, ("schema_version", "sha256"), "seal.protocol.manifest_report"
    )
    if report["schema_version"] != MANIFEST_SCHEMA:
        raise CodecPriorArtifactError("seal protocol manifest-report schema mismatch")
    report["sha256"] = _require_sha256(
        report["sha256"], "seal.protocol.manifest_report.sha256"
    )

    implementation = dict(
        _require_mapping(protocol["implementation"], "seal.protocol.implementation")
    )
    _require_exact_keys(
        implementation,
        ("schema_version", "files"),
        "seal.protocol.implementation",
    )
    if implementation["schema_version"] != IMPLEMENTATION_IDENTITY_SCHEMA:
        raise CodecPriorArtifactError("seal protocol implementation schema mismatch")
    raw_files = implementation["files"]
    if not isinstance(raw_files, list) or len(raw_files) != len(
        EXPECTED_IMPLEMENTATION_PATHS
    ):
        raise CodecPriorArtifactError(
            "seal protocol implementation must bind exactly three files"
        )
    files = []
    for index, (raw, expected_path) in enumerate(
        zip(raw_files, EXPECTED_IMPLEMENTATION_PATHS)
    ):
        label = "seal.protocol.implementation.files[{}]".format(index)
        item = dict(_require_mapping(raw, label))
        _require_exact_keys(item, ("relative_path", "sha256"), label)
        if item["relative_path"] != expected_path:
            raise CodecPriorArtifactError(
                "seal protocol implementation path mismatch at index {}".format(index)
            )
        files.append(
            {
                "relative_path": expected_path,
                "sha256": _require_sha256(item["sha256"], label + ".sha256"),
            }
        )
    protocol["manifest_report"] = report
    protocol["implementation"] = {
        "schema_version": IMPLEMENTATION_IDENTITY_SCHEMA,
        "files": files,
    }
    return protocol


def _validate_audiocraft(value: Any) -> Dict[str, Any]:
    audiocraft = dict(_require_mapping(value, "seal.audiocraft"))
    _require_exact_keys(
        audiocraft, ("base_commit", "source_identity"), "seal.audiocraft"
    )
    if audiocraft["base_commit"] != PINNED_AUDIOCRAFT_BASE_COMMIT:
        raise CodecPriorArtifactError("seal AudioCraft base commit mismatch")
    source = dict(
        _require_mapping(
            audiocraft["source_identity"],
            "seal.audiocraft.source_identity",
        )
    )
    _require_exact_keys(
        source,
        ("schema_version", "file_count", "tree_sha256", "identity_sha256"),
        "seal.audiocraft.source_identity",
    )
    if source["schema_version"] != AUDIOCRAFT_SOURCE_SCHEMA:
        raise CodecPriorArtifactError("seal AudioCraft source schema mismatch")
    if type(source["file_count"]) is not int or source["file_count"] <= 0:
        raise CodecPriorArtifactError(
            "seal AudioCraft source file_count must be positive"
        )
    source["tree_sha256"] = _require_sha256(
        source["tree_sha256"], "seal.audiocraft.source_identity.tree_sha256"
    )
    identity_hash = _require_sha256(
        source["identity_sha256"],
        "seal.audiocraft.source_identity.identity_sha256",
    )
    without_identity = dict(source)
    without_identity.pop("identity_sha256")
    if canonical_json_sha256(without_identity) != identity_hash:
        raise CodecPriorArtifactError("seal AudioCraft source identity hash mismatch")
    return {"base_commit": audiocraft["base_commit"], "source_identity": source}


def _validate_codec_contract(value: Any) -> None:
    contract = _require_mapping(value, "codec_prior.codec_contract")
    _require_int(contract.get("channels"), 1, "codec_contract.channels")
    _require_int(contract.get("sample_rate"), 32_000, "codec_contract.sample_rate")
    _require_number(contract.get("frame_rate"), 50.0, "codec_contract.frame_rate")
    _require_int(contract.get("cardinality"), 2_048, "codec_contract.cardinality")
    _require_int(contract.get("num_codebooks"), 4, "codec_contract.num_codebooks")
    _require_int(contract.get("input_frames"), 320_000, "codec_contract.input_frames")
    if contract.get("canonical_code_shape") != [1, 4, 500]:
        raise CodecPriorArtifactError(
            "codec_contract.canonical_code_shape must equal [1,4,500]"
        )
    if contract.get("validated_for_every_clip") is not True:
        raise CodecPriorArtifactError(
            "codec_contract.validated_for_every_clip must be true"
        )


def _normalized_prior(value: Any) -> Tuple[float, ...]:
    if not isinstance(value, list) or len(value) != EXPECTED_CODEBOOKS:
        raise CodecPriorArtifactError("codec_prior.prior must contain exactly four values")
    resolved = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise CodecPriorArtifactError("codec_prior.prior[{}] must be numeric".format(index))
        number = float(item)
        if not math.isfinite(number) or number <= 0.0:
            raise CodecPriorArtifactError(
                "codec_prior.prior[{}] must be finite and strictly positive".format(index)
            )
        resolved.append(number)
    total = math.fsum(resolved)
    if not math.isfinite(total) or total <= 0.0:
        raise CodecPriorArtifactError("codec_prior.prior has an invalid sum")
    return tuple(number / total for number in resolved)


def _validate_four_finite_numbers(value: Any, label: str) -> Tuple[float, ...]:
    if not isinstance(value, list) or len(value) != EXPECTED_CODEBOOKS:
        raise CodecPriorArtifactError("{} must contain exactly four values".format(label))
    return tuple(
        _require_finite_number(item, "{}[{}]".format(label, index))
        for index, item in enumerate(value)
    )


def _validate_recorded_prior(value: Any, label: str) -> Tuple[float, ...]:
    normalized = _normalized_prior(value)
    observed = tuple(float(item) for item in value)
    if not math.isclose(math.fsum(observed), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise CodecPriorArtifactError("{} must already sum to one".format(label))
    if any(
        not math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-12)
        for left, right in zip(observed, normalized)
    ):
        raise CodecPriorArtifactError("{} must already be normalized".format(label))
    return observed


def _expected_prior_from_marginal(
    marginal: Sequence[float], *, epsilon: float = 1.0e-8
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    clipped = tuple(max(float(value), epsilon) for value in marginal)
    total = math.fsum(clipped)
    return clipped, tuple(value / total for value in clipped)


def _require_vectors_close(
    observed: Sequence[float], expected: Sequence[float], label: str
) -> None:
    if len(observed) != len(expected) or any(
        not math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-14)
        for left, right in zip(observed, expected)
    ):
        raise CodecPriorArtifactError("{} is inconsistent with frozen arithmetic".format(label))


def _validate_a1_r2_scientific_payload(prior_payload: Mapping[str, Any]) -> None:
    """Reject a formally sealed result unless every frozen A1-R2 gate passed."""

    primary_prior = _validate_recorded_prior(
        prior_payload.get("prior"), "codec_prior.prior"
    )
    raw_mean = _validate_four_finite_numbers(
        prior_payload.get("raw_mean_marginal"), "codec_prior.raw_mean_marginal"
    )
    stored_clipped = _validate_four_finite_numbers(
        prior_payload.get("clipped_mean_marginal"),
        "codec_prior.clipped_mean_marginal",
    )
    expected_clipped, expected_primary = _expected_prior_from_marginal(raw_mean)
    _require_vectors_close(
        stored_clipped, expected_clipped, "codec_prior.clipped_mean_marginal"
    )
    _require_vectors_close(primary_prior, expected_primary, "codec_prior.prior")

    metric = _require_mapping(prior_payload.get("metric"), "codec_prior.metric")
    _require_exact_keys(
        metric,
        (
            "id",
            "fft_sizes",
            "hop_ratio",
            "window",
            "distance_epsilon",
            "reference_norm_guard",
            "denominator_flooring",
            "denominator_floor_activation",
            "marginal_epsilon",
        ),
        "codec_prior.metric",
    )
    if metric["id"] != "mrstft_spectral_convergence_plus_log_magnitude_l1":
        raise CodecPriorArtifactError("codec_prior.metric.id mismatch")
    if metric["fft_sizes"] != [512, 1024, 2048]:
        raise CodecPriorArtifactError("codec_prior.metric.fft_sizes mismatch")
    _require_number(metric["hop_ratio"], 0.25, "codec_prior.metric.hop_ratio")
    if metric["window"] != "Hann, win_length equals FFT size":
        raise CodecPriorArtifactError("codec_prior.metric.window mismatch")
    _require_number(
        metric["distance_epsilon"],
        REFERENCE_NORM_MINIMUM,
        "codec_prior.metric.distance_epsilon",
    )
    if metric["reference_norm_guard"] != (
        "raise if any per-channel FFT norm <= distance_epsilon"
    ):
        raise CodecPriorArtifactError("codec_prior.metric.reference_norm_guard mismatch")
    _require_bool(
        metric["denominator_flooring"],
        False,
        "codec_prior.metric.denominator_flooring",
    )
    _require_int(
        metric["denominator_floor_activation"],
        0,
        "codec_prior.metric.denominator_floor_activation",
    )
    _require_number(metric["marginal_epsilon"], 1.0e-8, "codec_prior.metric.marginal_epsilon")

    diagnostics = _require_mapping(
        prior_payload.get("diagnostics"), "codec_prior.diagnostics"
    )
    _require_int(
        diagnostics.get("denominator_floor_activation"),
        0,
        "codec_prior.diagnostics.denominator_floor_activation",
    )
    diagnostic_tv = _require_finite_number(
        diagnostics.get("total_variation_from_uniform"),
        "codec_prior.diagnostics.total_variation_from_uniform",
    )
    expected_tv = 0.5 * math.fsum(
        abs(value - 1.0 / EXPECTED_CODEBOOKS) for value in primary_prior
    )
    if not 0.0 <= diagnostic_tv <= 1.0 or not math.isclose(
        diagnostic_tv, expected_tv, rel_tol=1.0e-12, abs_tol=1.0e-14
    ):
        raise CodecPriorArtifactError(
            "codec_prior diagnostic total variation is inconsistent with the prior"
        )

    bootstrap = _require_mapping(
        prior_payload.get("bootstrap"), "codec_prior.bootstrap"
    )
    bootstrap_tv_low = _require_finite_number(
        bootstrap.get("total_variation_from_uniform_ci95_low"),
        "codec_prior.bootstrap.total_variation_from_uniform_ci95_low",
    )
    if not SCIENTIFIC_TV_LOWER_THRESHOLD < bootstrap_tv_low <= 1.0:
        raise CodecPriorArtifactError(
            "codec_prior bootstrap TV CI95 lower bound must be strictly greater than 0.05"
        )

    split = _require_mapping(prior_payload.get("split_half"), "codec_prior.split_half")
    if split.get("domain") != "ptc-opd-codec-split-v1":
        raise CodecPriorArtifactError("codec_prior.split_half.domain mismatch")
    _require_int(split.get("seed"), BOOTSTRAP_SEED, "codec_prior.split_half.seed")
    _require_int(split.get("half_a_n"), 256, "codec_prior.split_half.half_a_n")
    _require_int(split.get("half_b_n"), 256, "codec_prior.split_half.half_b_n")
    split_tau = _require_finite_number(
        split.get("kendall_tau_b"), "codec_prior.split_half.kendall_tau_b"
    )
    if not SCIENTIFIC_KENDALL_THRESHOLD <= split_tau <= 1.0:
        raise CodecPriorArtifactError(
            "codec_prior split-half Kendall tau-b must be at least 2/3"
        )

    influence = _require_mapping(
        prior_payload.get("leave_one_out_influence"),
        "codec_prior.leave_one_out_influence",
    )
    _require_exact_keys(
        influence,
        (
            "definition",
            "max_total_variation",
            "max_manifest_index",
            "max_fma_track_id",
            "per_clip",
        ),
        "codec_prior.leave_one_out_influence",
    )
    if influence["definition"] != (
        "TV(primary_prior, prior_recomputed_without_one_clip)"
    ):
        raise CodecPriorArtifactError("codec_prior leave-one-out definition mismatch")
    maximum_tv = _require_finite_number(
        influence["max_total_variation"],
        "codec_prior.leave_one_out_influence.max_total_variation",
    )
    if maximum_tv < 0.0 or maximum_tv > MAX_LEAVE_ONE_OUT_TV:
        raise CodecPriorArtifactError(
            "codec_prior maximum leave-one-out TV must be in [0, 0.05]"
        )
    maximum_index = influence["max_manifest_index"]
    maximum_track_id = influence["max_fma_track_id"]
    if type(maximum_index) is not int or not 0 <= maximum_index < EXPECTED_CLIPS:
        raise CodecPriorArtifactError("codec_prior leave-one-out max manifest index is invalid")
    if type(maximum_track_id) is not int or maximum_track_id < 0:
        raise CodecPriorArtifactError("codec_prior leave-one-out max track ID is invalid")
    raw_per_clip = influence["per_clip"]
    if not isinstance(raw_per_clip, list) or len(raw_per_clip) != EXPECTED_CLIPS:
        raise CodecPriorArtifactError(
            "codec_prior leave-one-out must contain exactly 512 per-clip records"
        )
    observed_records = []
    observed_track_ids = set()
    for expected_index, raw_record in enumerate(raw_per_clip):
        label = "codec_prior.leave_one_out_influence.per_clip[{}]".format(
            expected_index
        )
        record = _require_mapping(raw_record, label)
        _require_exact_keys(
            record,
            (
                "manifest_index",
                "fma_track_id",
                "raw_mean_marginal_without_clip",
                "prior_without_clip",
                "total_variation_from_primary_prior",
            ),
            label,
        )
        _require_int(record["manifest_index"], expected_index, label + ".manifest_index")
        track_id = record["fma_track_id"]
        if type(track_id) is not int or track_id < 0 or track_id in observed_track_ids:
            raise CodecPriorArtifactError(label + ".fma_track_id must be unique and non-negative")
        observed_track_ids.add(track_id)
        raw_without = _validate_four_finite_numbers(
            record["raw_mean_marginal_without_clip"],
            label + ".raw_mean_marginal_without_clip",
        )
        prior_without = _validate_recorded_prior(
            record["prior_without_clip"], label + ".prior_without_clip"
        )
        _, expected_without = _expected_prior_from_marginal(raw_without)
        _require_vectors_close(
            prior_without, expected_without, label + ".prior_without_clip"
        )
        tv = _require_finite_number(
            record["total_variation_from_primary_prior"],
            label + ".total_variation_from_primary_prior",
        )
        if tv < 0.0 or tv > MAX_LEAVE_ONE_OUT_TV:
            raise CodecPriorArtifactError(label + " total variation must be in [0, 0.05]")
        expected_loo_tv = 0.5 * math.fsum(
            abs(left - right) for left, right in zip(primary_prior, prior_without)
        )
        if not math.isclose(tv, expected_loo_tv, rel_tol=1.0e-12, abs_tol=1.0e-14):
            raise CodecPriorArtifactError(label + " total variation is inconsistent")
        observed_records.append((tv, expected_index, track_id))
    observed_max = max(observed_records, key=lambda item: (item[0], -item[1]))
    if (
        observed_max[0] != maximum_tv
        or observed_max[1] != maximum_index
        or observed_max[2] != maximum_track_id
    ):
        raise CodecPriorArtifactError("codec_prior leave-one-out maximum is inconsistent")

    sensitivity = _require_mapping(
        prior_payload.get("sensitivity"), "codec_prior.sensitivity"
    )
    _require_exact_keys(
        sensitivity,
        ("primary_estimator", "median", "trimmed_mean_1pct", "use_for_primary_prior"),
        "codec_prior.sensitivity",
    )
    if sensitivity["primary_estimator"] != "arithmetic_mean":
        raise CodecPriorArtifactError("codec_prior primary estimator must be arithmetic_mean")
    _require_bool(
        sensitivity["use_for_primary_prior"],
        False,
        "codec_prior.sensitivity.use_for_primary_prior",
    )
    median = _require_mapping(sensitivity["median"], "codec_prior.sensitivity.median")
    _require_exact_keys(
        median,
        ("marginal", "prior", "kendall_tau_b_vs_primary"),
        "codec_prior.sensitivity.median",
    )
    median_marginal = _validate_four_finite_numbers(
        median["marginal"], "codec_prior.sensitivity.median.marginal"
    )
    median_prior = _validate_recorded_prior(
        median["prior"], "codec_prior.sensitivity.median.prior"
    )
    _, expected_median_prior = _expected_prior_from_marginal(median_marginal)
    _require_vectors_close(
        median_prior, expected_median_prior, "codec_prior.sensitivity.median.prior"
    )
    median_tau = _require_finite_number(
        median["kendall_tau_b_vs_primary"],
        "codec_prior.sensitivity.median.kendall_tau_b_vs_primary",
    )
    if not SCIENTIFIC_KENDALL_THRESHOLD <= median_tau <= 1.0:
        raise CodecPriorArtifactError("codec_prior median Kendall tau-b must be at least 2/3")

    trimmed = _require_mapping(
        sensitivity["trimmed_mean_1pct"],
        "codec_prior.sensitivity.trimmed_mean_1pct",
    )
    _require_exact_keys(
        trimmed,
        (
            "definition",
            "trim_each_tail",
            "retained_clips_per_codebook",
            "marginal",
            "prior",
            "kendall_tau_b_vs_primary",
        ),
        "codec_prior.sensitivity.trimmed_mean_1pct",
    )
    if trimmed["definition"] != (
        "sort each codebook independently; remove floor(0.01*N) from each tail"
    ):
        raise CodecPriorArtifactError("codec_prior 1% trimmed-mean definition mismatch")
    _require_int(
        trimmed["trim_each_tail"],
        TRIM_EACH_TAIL_AT_N512,
        "codec_prior.sensitivity.trimmed_mean_1pct.trim_each_tail",
    )
    _require_int(
        trimmed["retained_clips_per_codebook"],
        TRIMMED_RETAINED_AT_N512,
        "codec_prior.sensitivity.trimmed_mean_1pct.retained_clips_per_codebook",
    )
    trimmed_marginal = _validate_four_finite_numbers(
        trimmed["marginal"], "codec_prior.sensitivity.trimmed_mean_1pct.marginal"
    )
    trimmed_prior = _validate_recorded_prior(
        trimmed["prior"], "codec_prior.sensitivity.trimmed_mean_1pct.prior"
    )
    _, expected_trimmed_prior = _expected_prior_from_marginal(trimmed_marginal)
    _require_vectors_close(
        trimmed_prior,
        expected_trimmed_prior,
        "codec_prior.sensitivity.trimmed_mean_1pct.prior",
    )
    trimmed_tau = _require_finite_number(
        trimmed["kendall_tau_b_vs_primary"],
        "codec_prior.sensitivity.trimmed_mean_1pct.kendall_tau_b_vs_primary",
    )
    if not SCIENTIFIC_KENDALL_THRESHOLD <= trimmed_tau <= 1.0:
        raise CodecPriorArtifactError(
            "codec_prior trimmed-mean Kendall tau-b must be at least 2/3"
        )

    gates = _require_mapping(
        prior_payload.get("acceptance_gates"), "codec_prior.acceptance_gates"
    )
    _require_exact_keys(
        gates, ("thresholds", "scientific", "scientific_status"), "codec_prior.acceptance_gates"
    )
    thresholds = _require_mapping(
        gates["thresholds"], "codec_prior.acceptance_gates.thresholds"
    )
    _require_exact_keys(
        thresholds,
        (
            "bootstrap_tv_ci95_low_strictly_greater_than",
            "kendall_tau_b_minimum",
            "kendall_tau_b_minimum_exact",
            "max_leave_one_out_total_variation",
        ),
        "codec_prior.acceptance_gates.thresholds",
    )
    _require_number(
        thresholds["bootstrap_tv_ci95_low_strictly_greater_than"],
        SCIENTIFIC_TV_LOWER_THRESHOLD,
        "codec_prior.acceptance_gates.thresholds.bootstrap_tv",
    )
    _require_number(
        thresholds["kendall_tau_b_minimum"],
        SCIENTIFIC_KENDALL_THRESHOLD,
        "codec_prior.acceptance_gates.thresholds.kendall_tau_b",
    )
    if thresholds["kendall_tau_b_minimum_exact"] != "2/3":
        raise CodecPriorArtifactError("codec_prior exact Kendall threshold mismatch")
    _require_number(
        thresholds["max_leave_one_out_total_variation"],
        MAX_LEAVE_ONE_OUT_TV,
        "codec_prior.acceptance_gates.thresholds.max_leave_one_out_tv",
    )
    science = _require_mapping(
        gates["scientific"], "codec_prior.acceptance_gates.scientific"
    )
    expected_science_keys = (
        "bootstrap_tv_ci95_low_gt_0_05",
        "split_half_kendall_tau_b_ge_two_thirds",
        "max_leave_one_out_tv_le_0_05",
        "median_rank_kendall_tau_b_ge_two_thirds",
        "trimmed_rank_kendall_tau_b_ge_two_thirds",
    )
    _require_exact_keys(
        science, expected_science_keys, "codec_prior.acceptance_gates.scientific"
    )
    for key in expected_science_keys:
        _require_bool(science[key], True, "codec_prior.acceptance_gates.scientific." + key)
    if gates["scientific_status"] != "pass":
        raise CodecPriorArtifactError("codec_prior scientific status must be pass")


def load_codec_prior_artifact(path: Path) -> CodecPriorArtifact:
    """Verify and load one formal A1 artifact directory.

    The function always rehashes all three payload files.  It does not trust a
    caller-supplied prior path and it rejects extra files, symlinks, partial
    generations, debug-sized assays, or internally inconsistent provenance.
    """

    supplied = Path(path).expanduser()
    if supplied.is_symlink():
        raise CodecPriorArtifactError("A1 artifact directory must not be a symlink")
    try:
        directory = supplied.resolve(strict=True)
    except FileNotFoundError as exc:
        raise CodecPriorArtifactError("A1 artifact directory does not exist") from exc
    if not directory.is_dir():
        raise CodecPriorArtifactError("A1 artifact path must be a directory")
    if directory.name != EXPECTED_ARTIFACT_DIRECTORY_BASENAME:
        raise CodecPriorArtifactError(
            "formal A1-R2 artifact directory basename must be {}".format(
                EXPECTED_ARTIFACT_DIRECTORY_BASENAME
            )
        )

    entries = list(directory.iterdir())
    observed_names = {entry.name for entry in entries}
    if len(entries) != len(observed_names) or observed_names != DIRECTORY_MEMBERS:
        raise CodecPriorArtifactError(
            "A1 artifact directory members differ; missing={}, unexpected={}".format(
                sorted(DIRECTORY_MEMBERS - observed_names),
                sorted(observed_names - DIRECTORY_MEMBERS),
            )
        )
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise CodecPriorArtifactError(
                "A1 artifact member is not a regular file: {}".format(entry.name)
            )

    seal_path = directory / ARTIFACT_SEAL_NAME
    seal = _load_json_object(seal_path, ARTIFACT_SEAL_NAME)
    _require_exact_keys(
        seal,
        (
            "schema_version",
            "status",
            "manifest",
            "checkpoint",
            "source_identity",
            "audiocraft",
            "codec_load",
            "protocol",
            "artifacts",
        ),
        "artifact seal",
    )
    if seal["schema_version"] != ARTIFACT_SEAL_SCHEMA:
        raise CodecPriorArtifactError("artifact seal schema mismatch")
    if seal["status"] != "complete":
        raise CodecPriorArtifactError("artifact seal status must be complete")

    manifest = _validate_manifest(seal["manifest"])
    checkpoint = _validate_checkpoint(seal["checkpoint"])
    source_identity = _validate_source_identity(seal["source_identity"])
    audiocraft = _validate_audiocraft(seal["audiocraft"])
    codec_load = _validate_portable_codec_load(seal["codec_load"], "seal.codec_load")
    protocol = _validate_protocol(seal["protocol"])
    if (
        protocol["manifest_report"]["sha256"]
        != source_identity["manifest_report_sha256"]
    ):
        raise CodecPriorArtifactError(
            "seal protocol/source manifest-report hash mismatch"
        )
    if (
        checkpoint["load_mode"] != codec_load["load_mode"]
        or checkpoint["pretrained_model_id"] != codec_load["pretrained_model_id"]
        or checkpoint["resolved_snapshot_revision"]
        != codec_load["resolved_snapshot_revision"]
    ):
        raise CodecPriorArtifactError("seal checkpoint/codec-load provenance mismatch")
    artifacts = _require_mapping(seal["artifacts"], "seal.artifacts")
    _require_exact_keys(artifacts, PAYLOAD_NAMES, "seal.artifacts")

    verified_artifacts: Dict[str, Dict[str, Any]] = {}
    for name in PAYLOAD_NAMES:
        expected = _require_mapping(artifacts[name], "seal.artifacts.{}".format(name))
        _require_exact_keys(expected, ("sha256", "size_bytes"), "seal.artifacts.{}".format(name))
        expected_sha256 = _require_sha256(
            expected["sha256"], "seal.artifacts.{}.sha256".format(name)
        )
        expected_size = expected["size_bytes"]
        if type(expected_size) is not int or expected_size < 0:
            raise CodecPriorArtifactError(
                "seal.artifacts.{}.size_bytes must be a non-negative integer".format(name)
            )
        member = directory / name
        observed = {
            "sha256": sha256_file(member),
            "size_bytes": int(member.stat().st_size),
        }
        if observed != {"sha256": expected_sha256, "size_bytes": expected_size}:
            raise CodecPriorArtifactError(
                "artifact hash/size mismatch for {}".format(name)
            )
        verified_artifacts[name] = observed

    prior_payload = _load_json_object(directory / PRIOR_NAME, PRIOR_NAME)
    if prior_payload.get("schema_version") != RESULT_SCHEMA:
        raise CodecPriorArtifactError("codec prior payload schema mismatch")
    _require_int(prior_payload.get("n_clips"), EXPECTED_CLIPS, "codec_prior.n_clips")
    _require_int(
        prior_payload.get("num_codebooks"), EXPECTED_CODEBOOKS, "codec_prior.num_codebooks"
    )
    bootstrap = _require_mapping(prior_payload.get("bootstrap"), "codec_prior.bootstrap")
    _require_int(
        bootstrap.get("replicates"), BOOTSTRAP_REPLICATES, "codec_prior.bootstrap.replicates"
    )
    _require_int(bootstrap.get("seed"), BOOTSTRAP_SEED, "codec_prior.bootstrap.seed")
    _validate_codec_contract(prior_payload.get("codec_contract"))
    prior = _validate_recorded_prior(prior_payload.get("prior"), "codec_prior.prior")
    if prior_payload.get("protocol") != protocol:
        raise CodecPriorArtifactError("prior payload/seal A1-R2 protocol mismatch")
    report_path = _require_nonempty_string(
        prior_payload.get("manifest_report_path"),
        "codec_prior.manifest_report_path",
    )
    pure_report = PurePosixPath(report_path)
    if (
        pure_report.name != EXPECTED_MANIFEST_REPORT_BASENAME
        or pure_report.parent.name != EXPECTED_MANIFEST_PARENT
    ):
        raise CodecPriorArtifactError(
            "codec prior manifest-report path must end in "
            "a1-r2/codec_calibration.train.report.json"
        )
    payload_report_hash = _require_sha256(
        prior_payload.get("manifest_report_sha256"),
        "codec_prior.manifest_report_sha256",
    )
    if payload_report_hash != protocol["manifest_report"]["sha256"]:
        raise CodecPriorArtifactError(
            "prior payload/seal manifest-report hash mismatch"
        )
    _validate_a1_r2_scientific_payload(prior_payload)

    payload_codec_load = _portable_codec_load_from_prior(
        prior_payload.get("codec_load")
    )
    if payload_codec_load != codec_load:
        raise CodecPriorArtifactError("prior payload/seal codec-load provenance mismatch")
    if prior_payload.get("audiocraft_commit") != audiocraft["base_commit"]:
        raise CodecPriorArtifactError("prior payload/seal AudioCraft commit mismatch")
    payload_audiocraft_source = prior_payload.get("audiocraft_source_identity")
    if payload_audiocraft_source != audiocraft["source_identity"]:
        raise CodecPriorArtifactError(
            "prior payload/seal AudioCraft source identity mismatch"
        )

    payload_manifest_hash = _require_sha256(
        prior_payload.get("manifest_sha256"), "codec_prior.manifest_sha256"
    )
    if payload_manifest_hash != manifest["sha256"]:
        raise CodecPriorArtifactError("prior payload/ seal manifest hash mismatch")
    payload_checkpoint_hash = _require_sha256(
        prior_payload.get("codec_checkpoint_sha256"),
        "codec_prior.codec_checkpoint_sha256",
    )
    if payload_checkpoint_hash != checkpoint["sha256"]:
        raise CodecPriorArtifactError("prior payload/seal checkpoint hash mismatch")
    payload_hashes = _require_mapping(
        prior_payload.get("artifact_payload_sha256"),
        "codec_prior.artifact_payload_sha256",
    )
    _require_exact_keys(
        payload_hashes,
        (PER_CLIP_NAME, SUMMARY_NAME),
        "codec_prior.artifact_payload_sha256",
    )
    for name in (PER_CLIP_NAME, SUMMARY_NAME):
        if _require_sha256(
            payload_hashes[name], "codec_prior.artifact_payload_sha256.{}".format(name)
        ) != verified_artifacts[name]["sha256"]:
            raise CodecPriorArtifactError(
                "prior payload internal artifact hash mismatch for {}".format(name)
            )

    artifact_seal_sha256 = sha256_file(seal_path)
    identity: Dict[str, Any] = {
        "schema_version": IDENTITY_SCHEMA,
        "artifact_seal_sha256": artifact_seal_sha256,
        "artifacts": verified_artifacts,
        "manifest": manifest,
        "checkpoint": checkpoint,
        "source_identity": source_identity,
        "audiocraft": audiocraft,
        "codec_load": codec_load,
        "protocol": protocol,
        "normalized_prior": list(prior),
    }
    identity_sha256 = canonical_json_sha256(identity)
    return CodecPriorArtifact(
        directory=str(directory),
        prior=prior,
        manifest_sha256=manifest["sha256"],
        checkpoint_sha256=checkpoint["sha256"],
        ordered_source_records_sha256=source_identity[
            "ordered_source_records_sha256"
        ],
        codec_prior_sha256=verified_artifacts[PRIOR_NAME]["sha256"],
        artifact_seal_sha256=artifact_seal_sha256,
        codec_load=codec_load,
        identity=identity,
        identity_sha256=identity_sha256,
    )


def verify_local_codec_snapshot(
    artifact: CodecPriorArtifact,
    *,
    cache_resolver: Optional[Callable[..., Any]] = None,
) -> Optional[Mapping[str, Any]]:
    """Rehash the locally resolved EnCodec snapshot bound by a sealed A1 run.

    Official MusicGen exports can store only ``{"pretrained":
    "facebook/encodec_32khz"}`` in ``compression_state_dict.bin``.  In that
    case the small wrapper hash is insufficient: every downstream machine must
    prove that its offline Hugging Face cache resolves the exact revision and
    file bytes sealed by A1.  A self-contained compression export needs no
    external snapshot and returns ``None``.
    """

    codec_load = artifact.codec_load
    if codec_load.get("load_mode") == "self_contained":
        return None
    model_id = codec_load.get("pretrained_model_id")
    revision = codec_load.get("resolved_snapshot_revision")
    if model_id != ALLOWED_PRETRAINED_CODEC_ID or not isinstance(revision, str):
        raise CodecPriorArtifactError("sealed pretrained codec identity is malformed")
    if cache_resolver is None:
        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError as exc:
            raise CodecPriorArtifactError(
                "huggingface_hub is required to verify the sealed codec snapshot"
            ) from exc
        cache_resolver = try_to_load_from_cache
    cached = cache_resolver(
        repo_id=model_id,
        filename="config.json",
        revision=revision,
    )
    if not isinstance(cached, str):
        raise CodecPriorArtifactError(
            "offline cache lacks config.json for sealed codec revision {}".format(
                revision
            )
        )
    visible_config = Path(cached).absolute()
    parts = visible_config.parts
    if "snapshots" not in parts:
        raise CodecPriorArtifactError(
            "cached codec config path has no snapshots/<revision> identity"
        )
    marker = parts.index("snapshots")
    if marker + 1 >= len(parts) or parts[marker + 1] != revision:
        raise CodecPriorArtifactError("cached codec snapshot revision mismatch")
    snapshot_root = Path(*parts[: marker + 2])
    if snapshot_root.is_symlink() or not snapshot_root.is_dir():
        raise CodecPriorArtifactError("cached codec snapshot root is not a directory")
    observed_files = []
    for visible_path in sorted(snapshot_root.rglob("*")):
        if visible_path.is_symlink() and not visible_path.exists():
            raise CodecPriorArtifactError(
                "cached codec snapshot contains a broken symlink: {}".format(
                    visible_path
                )
            )
        if not visible_path.is_file():
            continue
        try:
            resolved_path = visible_path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise CodecPriorArtifactError(
                "cannot resolve cached codec snapshot member: {}".format(
                    visible_path
                )
            ) from exc
        if not resolved_path.is_file():
            raise CodecPriorArtifactError(
                "cached codec snapshot target is not a regular file"
            )
        observed_files.append(
            {
                "relative_path": visible_path.relative_to(snapshot_root).as_posix(),
                "size_bytes": int(resolved_path.stat().st_size),
                "sha256": sha256_file(resolved_path),
            }
        )
    expected_files = list(codec_load.get("resolved_snapshot_files", ()))
    if observed_files != expected_files:
        raise CodecPriorArtifactError(
            "local codec snapshot files differ from the sealed A1 byte identity"
        )
    return {
        "pretrained_model_id": model_id,
        "resolved_snapshot_revision": revision,
        "resolved_snapshot_files": observed_files,
    }


__all__ = [
    "ARTIFACT_SEAL_NAME",
    "ARTIFACT_SEAL_SCHEMA",
    "CodecPriorArtifact",
    "CodecPriorArtifactError",
    "IDENTITY_SCHEMA",
    "IMPLEMENTATION_IDENTITY_SCHEMA",
    "MANIFEST_SCHEMA",
    "PAYLOAD_NAMES",
    "PER_CLIP_NAME",
    "PRIOR_NAME",
    "RESULT_SCHEMA",
    "PROTOCOL_LABEL",
    "SUMMARY_NAME",
    "canonical_json_sha256",
    "load_codec_prior_artifact",
    "sha256_file",
    "verify_local_codec_snapshot",
]
