#!/usr/bin/env python3
"""Estimate the frozen progressive MusicGen codec prior on FMA calibration clips.

The command is deliberately rank-zero/single-GPU.  It loads a *local*
AudioCraft compression checkpoint, verifies every calibration source and PCM
hash, encodes each clip once, and decodes cumulative canonical RVQ prefixes.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import numbers
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


WORKPACK_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKPACK_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 4701
SPLIT_DOMAIN = "ptc-opd-codec-split-v1"
SPLIT_SEED = 4701
EXPECTED_ITEMS = 512
REQUIRED_ARCHIVE_SHA1 = "ade154f733639d52e35e32f5593efe5be76c6d70"
PINNED_AUDIOCRAFT_BASE_COMMIT = "896ec7c47f5e5d1e5aa1e4b260c4405328bf009d"
REQUIRED_MANIFEST_SCHEMA = "ptc-opd-fma-calibration-v2"
RESULT_SCHEMA = "ptc-opd-codec-prior-v3"
ARTIFACT_SEAL_SCHEMA = "ptc-opd-codec-artifact-seal-v3"
ARTIFACT_IDENTITY_SCHEMA = "ptc-opd-codec-prior-artifact-identity-v3"
IMPLEMENTATION_IDENTITY_SCHEMA = "ptc-opd-a1-r2-implementation-v1"
PROTOCOL_LABEL = "A1-R2"
ARTIFACT_SEAL_NAME = "ARTIFACT_SEAL.json"
FROZEN_CODEC_CHANNELS = 1
FROZEN_CODEC_SAMPLE_RATE = 32_000
FROZEN_CODEC_FRAME_RATE = 50.0
FROZEN_CODEC_CARDINALITY = 2_048
FROZEN_CODEC_CODEBOOKS = 4
FROZEN_CODEC_INPUT_FRAMES = 320_000
FROZEN_CODEC_CODE_FRAMES = 500
FORBIDDEN_INPUT_BASENAME = re.compile(
    r"(^|[._-])(test|dev|development|probe|phenomenon_probe)([._-]|$)", re.IGNORECASE
)
REQUIRED_INPUT_BASENAME = "codec_calibration.train.jsonl"
REQUIRED_REPORT_BASENAME = "codec_calibration.train.report.json"
REQUIRED_OUTPUT_DIR_BASENAME = "phase_a1_codec_prior_r2"
EXPECTED_ORIGINAL_CANDIDATES = 7_994
SELECTION_DOMAIN = "ptc-opd-codec-cal-v1"
SEGMENT_DOMAIN = "ptc-opd-codec-segment-v1"
SELECTION_SEED = 2701
PRE_DOWNMIX_RMS_DENOMINATOR = 1.0e-12
MIN_MONO_RMS = 1.0e-5
MIN_MONO_COMPATIBILITY_RATIO = 1.0e-3
REFERENCE_NORM_MINIMUM = 1.0e-7
SCIENTIFIC_KENDALL_THRESHOLD = 2.0 / 3.0
SCIENTIFIC_TV_LOWER_THRESHOLD = 0.05
MAX_LEAVE_ONE_OUT_TV = 0.05
ALLOWED_PRETRAINED_CODEC_ID = "facebook/encodec_32khz"
REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "fma_track_id",
        "relative_audio_path",
        "source_audio_sha256",
        "decoded_duration_frames",
        "decoded_sample_rate",
        "decoded_channels",
        "segment_start_frame",
        "segment_num_frames",
        "extracted_pcm_sha256",
        "archive_sha1",
        "archive_sha1_verified",
        "publication_eligible",
        "protocol_label",
        "selection_rank",
        "selection_rank_sha256",
        "segment_offset_sha256",
        "pcm_hash_encoding",
        "all_pcm_samples_finite",
        "pre_downmix_rms",
        "mono_rms",
        "mono_compatibility_ratio",
        "lr_pearson_correlation",
        "eligibility_codec_input_pcm_sha256",
        "eligible",
        "failed_conditions",
        "eligibility_thresholds",
    }
)
OUTPUT_NAMES = (
    "codec_prior.per_clip.jsonl.gz",
    "codec_prior.summary.csv",
    "codec_prior.json",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def pcm_f32le_sha256(samples: Any) -> str:
    import numpy as np

    array = np.asarray(samples)
    if array.ndim != 2:
        raise ValueError("PCM must have shape [frames, channels]")
    canonical = np.ascontiguousarray(array, dtype=np.dtype("<f4"))
    return sha256_bytes(canonical.tobytes(order="C"))


def codes_i64le_sha256(codes: Any) -> str:
    import numpy as np

    array = np.asarray(codes)
    if array.ndim != 3:
        raise ValueError("codes must have shape [B,Q,T]")
    canonical = np.ascontiguousarray(array, dtype=np.dtype("<i8"))
    return sha256_bytes(canonical.tobytes(order="C"))


def _domain_hash(domain: str, seed: int, track_id: int) -> str:
    return hashlib.sha256(
        f"{domain}|{int(seed)}|{int(track_id)}".encode("utf-8")
    ).hexdigest()


def _selection_hash(track_id: int) -> str:
    return _domain_hash(SELECTION_DOMAIN, SELECTION_SEED, track_id)


def _segment_hash(track_id: int) -> str:
    return _domain_hash(SEGMENT_DOMAIN, SELECTION_SEED, track_id)


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def implementation_identity() -> Dict[str, Any]:
    """Hash the three exact A1-R2 implementation files used by a run."""

    paths = (
        WORKPACK_ROOT / "scripts" / "build_fma_calibration_manifest.py",
        Path(__file__).resolve(),
        WORKPACK_ROOT / "src" / "ptc_opd" / "perceptual_prior.py",
    )
    files = [
        {
            "relative_path": path.relative_to(WORKPACK_ROOT).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    return {
        "schema_version": IMPLEMENTATION_IDENTITY_SCHEMA,
        "files": files,
    }


def validate_codec_model_contract(model: Any) -> Dict[str, Any]:
    """Fail closed unless the loaded codec is exactly the frozen MusicGen codec."""

    required = (
        ("channels", FROZEN_CODEC_CHANNELS),
        ("sample_rate", FROZEN_CODEC_SAMPLE_RATE),
        ("frame_rate", FROZEN_CODEC_FRAME_RATE),
        ("cardinality", FROZEN_CODEC_CARDINALITY),
        ("num_codebooks", FROZEN_CODEC_CODEBOOKS),
    )
    observed: Dict[str, Any] = {}
    for name, expected in required:
        if not hasattr(model, name):
            raise RuntimeError(f"codec model does not expose required metadata {name!r}")
        value = getattr(model, name)
        # AudioCraft metadata can be exposed as Python or NumPy numeric scalars.
        # This is the non-scientific compatibility fix used by the successful
        # A1-R1 remote execution; bool remains explicitly forbidden.
        if isinstance(value, bool):
            valid_type = False
        elif isinstance(expected, int):
            valid_type = isinstance(value, numbers.Integral)
        else:
            valid_type = isinstance(value, numbers.Real)
        if not valid_type:
            raise RuntimeError(
                f"codec metadata {name} has invalid type {type(value).__name__}"
            )
        if not math.isfinite(float(value)):
            raise RuntimeError(f"codec metadata {name} is NaN/Inf")
        if float(value) != float(expected):
            raise RuntimeError(
                f"frozen codec requires {name}={expected}, observed {value}"
            )
        observed[name] = int(value) if isinstance(expected, int) else float(value)
    if "InterleaveStereo" in type(model).__name__:
        raise RuntimeError(
            "InterleaveStereoCompressionModel is forbidden for cumulative prefix decoding"
        )
    return {
        **observed,
        "input_duration_seconds": 10,
        "input_frames": FROZEN_CODEC_INPUT_FRAMES,
        "codec_input_shape": [1, FROZEN_CODEC_CHANNELS, FROZEN_CODEC_INPUT_FRAMES],
        "codec_input_dtype": "torch.float32",
        "canonical_code_shape": [1, FROZEN_CODEC_CODEBOOKS, FROZEN_CODEC_CODE_FRAMES],
        "canonical_code_dtype": "torch.int64",
        "canonical_code_range": f"0 <= token < {FROZEN_CODEC_CARDINALITY}",
    }


def validate_codec_input(waveform: Any, torch_module: Any) -> None:
    """Validate one exact float32 ``[1,1,320000]`` codec input."""

    if not isinstance(waveform, torch_module.Tensor):
        raise TypeError("codec input waveform must be a torch.Tensor")
    expected_waveform_shape = (1, FROZEN_CODEC_CHANNELS, FROZEN_CODEC_INPUT_FRAMES)
    if tuple(waveform.shape) != expected_waveform_shape:
        raise RuntimeError(
            f"codec input must have exact shape {expected_waveform_shape}, "
            f"observed {tuple(waveform.shape)}"
        )
    if waveform.dtype != torch_module.float32:
        raise RuntimeError(
            f"codec input must have dtype torch.float32, observed {waveform.dtype}"
        )
    if not bool(torch_module.isfinite(waveform).all().item()):
        raise RuntimeError("codec input contains NaN/Inf")


def validate_codec_codes(codes: Any, torch_module: Any) -> None:
    """Validate exact long ``[1,4,500]`` canonical codes and token range."""

    if not isinstance(codes, torch_module.Tensor):
        raise TypeError("codec encode output must be a torch.Tensor")
    expected_codes_shape = (1, FROZEN_CODEC_CODEBOOKS, FROZEN_CODEC_CODE_FRAMES)
    if tuple(codes.shape) != expected_codes_shape:
        raise RuntimeError(
            f"canonical codes must have exact shape {expected_codes_shape}, "
            f"observed {tuple(codes.shape)}"
        )
    if codes.dtype != torch_module.long:
        raise RuntimeError(
            f"canonical codes must have dtype torch.long/int64, observed {codes.dtype}"
        )
    minimum = int(codes.min().item())
    maximum = int(codes.max().item())
    if minimum < 0 or maximum >= FROZEN_CODEC_CARDINALITY:
        raise RuntimeError(
            "canonical codec token is outside the frozen range "
            f"[0,{FROZEN_CODEC_CARDINALITY}): min={minimum}, max={maximum}"
        )


def validate_codec_tensors(waveform: Any, codes: Any, torch_module: Any) -> None:
    """Validate one exact 10-second codec input and its canonical encoding."""

    validate_codec_input(waveform, torch_module)
    validate_codec_codes(codes, torch_module)


def _json_line(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def manifest_sha256_and_rows(path: Path, expected_sha256: str) -> Tuple[str, List[Dict[str, Any]]]:
    if path.name != REQUIRED_INPUT_BASENAME:
        detail = "contains a forbidden dev/test/probe marker" if FORBIDDEN_INPUT_BASENAME.search(path.name) else "does not match the frozen calibration basename"
        raise ValueError(
            f"manifest basename must be {REQUIRED_INPUT_BASENAME!r}; got {path.name!r} ({detail})"
        )
    if path.parent.name != "a1-r2":
        raise ValueError("formal A1-R2 manifest must live in a directory named 'a1-r2'")
    payload = path.read_bytes()
    observed = sha256_bytes(payload)
    expected = expected_sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("--manifest-sha256 must be 64 lowercase/uppercase hexadecimal characters")
    if observed != expected:
        raise RuntimeError(
            f"manifest SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    rows: List[Dict[str, Any]] = []
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line.strip():
            raise ValueError(f"manifest contains a blank line at {line_number}")
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at manifest line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"manifest line {line_number} is not a JSON object")
        missing = REQUIRED_FIELDS.difference(value)
        if missing:
            raise ValueError(
                f"manifest line {line_number} is missing fields: {sorted(missing)}"
            )
        rows.append(value)
    validate_manifest_rows(rows)
    return observed, rows


def validate_manifest_rows(rows: Sequence[Mapping[str, Any]], expected_items: int = EXPECTED_ITEMS) -> None:
    if len(rows) != expected_items:
        raise ValueError(f"codec calibration must contain exactly {expected_items} items, got {len(rows)}")
    track_ids = set()
    relative_paths = set()
    ranking_keys = []
    for index, row in enumerate(rows):
        if row["schema_version"] != REQUIRED_MANIFEST_SCHEMA:
            raise ValueError(f"row {index} has unsupported schema_version {row['schema_version']!r}")
        if row["protocol_label"] != PROTOCOL_LABEL:
            raise ValueError(f"row {index} has unsupported protocol_label")
        if row["archive_sha1"] != REQUIRED_ARCHIVE_SHA1:
            raise ValueError(f"row {index} does not carry the frozen FMA-small archive SHA-1")
        if row["archive_sha1_verified"] is not True or row["publication_eligible"] is not True:
            raise ValueError(f"row {index} came from an unverified/test-only archive waiver")
        track_id = int(row["fma_track_id"])
        relative_path = str(row["relative_audio_path"])
        if track_id in track_ids or relative_path in relative_paths:
            raise ValueError(f"duplicate track ID or path at manifest row {index}")
        track_ids.add(track_id)
        relative_paths.add(relative_path)
        source_sha = str(row["source_audio_sha256"]).lower()
        pcm_sha = str(row["extracted_pcm_sha256"]).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", source_sha) or not re.fullmatch(r"[0-9a-f]{64}", pcm_sha):
            raise ValueError(f"row {index} contains a malformed SHA-256")
        decoded_frames = int(row["decoded_duration_frames"])
        sample_rate = int(row["decoded_sample_rate"])
        channels = int(row["decoded_channels"])
        start = int(row["segment_start_frame"])
        length = int(row["segment_num_frames"])
        if min(decoded_frames, sample_rate, channels, length) <= 0 or start < 0:
            raise ValueError(f"row {index} contains non-positive audio metadata")
        if start + length > decoded_frames:
            raise ValueError(f"row {index} segment lies outside decoded audio")
        if length != 10 * sample_rate:
            raise ValueError(f"row {index} is not an exact 10-second source-rate segment")
        if int(row["selection_rank"]) != index:
            raise ValueError(f"row {index} has inconsistent selection_rank")
        if row["selection_rank_sha256"] != _selection_hash(track_id):
            raise ValueError(f"row {index} has inconsistent selection hash")
        ranking_keys.append((str(row["selection_rank_sha256"]), track_id))
        if row["segment_offset_sha256"] != _segment_hash(track_id):
            raise ValueError(f"row {index} has inconsistent segment-offset hash")
        if row["pcm_hash_encoding"] != "frames_x_channels_float32_little_endian_c_order":
            raise ValueError(f"row {index} has unsupported PCM hash encoding")
        if row["all_pcm_samples_finite"] is not True or row["eligible"] is not True:
            raise ValueError(f"row {index} is not A1-R2 input-eligible")
        if row["failed_conditions"] != []:
            raise ValueError(f"row {index} carries failed eligibility conditions")
        thresholds = row["eligibility_thresholds"]
        if thresholds != {
            "pre_downmix_rms_denominator": PRE_DOWNMIX_RMS_DENOMINATOR,
            "minimum_mono_rms": MIN_MONO_RMS,
            "minimum_mono_compatibility_ratio": MIN_MONO_COMPATIBILITY_RATIO,
        }:
            raise ValueError(f"row {index} eligibility thresholds drifted")
        pre_rms = _finite_float(row["pre_downmix_rms"], f"row {index} pre_downmix_rms")
        mono_rms = _finite_float(row["mono_rms"], f"row {index} mono_rms")
        ratio = _finite_float(
            row["mono_compatibility_ratio"],
            f"row {index} mono_compatibility_ratio",
        )
        if pre_rms < 0.0 or mono_rms < MIN_MONO_RMS or ratio < MIN_MONO_COMPATIBILITY_RATIO:
            raise ValueError(f"row {index} violates frozen eligibility bounds")
        correlation = row["lr_pearson_correlation"]
        if correlation is not None:
            correlation_value = _finite_float(correlation, f"row {index} L/R correlation")
            if not -1.0 <= correlation_value <= 1.0:
                raise ValueError(f"row {index} L/R correlation lies outside [-1,1]")
        codec_pcm_sha = str(row["eligibility_codec_input_pcm_sha256"]).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", codec_pcm_sha):
            raise ValueError(f"row {index} contains malformed eligibility codec-input SHA-256")
    if ranking_keys != sorted(ranking_keys):
        raise ValueError("manifest rows are not in increasing selection-hash order")


def manifest_report_sha256_and_payload(
    path: Path,
    expected_sha256: str,
    *,
    manifest_sha256: str,
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    """Load and fail-closed validate the all-candidate A1-R2 audit report."""

    if path.name != REQUIRED_REPORT_BASENAME:
        raise ValueError(
            f"manifest report basename must be {REQUIRED_REPORT_BASENAME!r}; got {path.name!r}"
        )
    if path.parent.name != "a1-r2":
        raise ValueError("formal A1-R2 manifest report must live in a directory named 'a1-r2'")
    payload = path.read_bytes()
    observed = sha256_bytes(payload)
    expected = expected_sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("--manifest-report-sha256 must be 64 hexadecimal characters")
    if observed != expected:
        raise RuntimeError(
            f"manifest report SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    try:
        report = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest report is invalid JSON: {exc}") from exc
    if not isinstance(report, dict):
        raise ValueError("manifest report must contain one JSON object")
    if report.get("schema_version") != REQUIRED_MANIFEST_SCHEMA:
        raise ValueError("manifest report schema mismatch")
    if report.get("protocol_label") != PROTOCOL_LABEL:
        raise ValueError("manifest report protocol label mismatch")
    if report.get("manifest_sha256") != manifest_sha256:
        raise ValueError("manifest report does not bind the supplied manifest")
    frozen_scalars = {
        "original_candidate_tracks": EXPECTED_ORIGINAL_CANDIDATES,
        "expected_original_candidate_tracks": EXPECTED_ORIGINAL_CANDIDATES,
        "selected_tracks": EXPECTED_ITEMS,
        "selection_count_contract": EXPECTED_ITEMS,
        "selection_count_used": EXPECTED_ITEMS,
        "split_seed": SELECTION_SEED,
        "selection_domain": SELECTION_DOMAIN,
        "segment_domain": SEGMENT_DOMAIN,
        "segment_seconds": 10,
        "archive_sha1": REQUIRED_ARCHIVE_SHA1,
        "archive_sha1_verified": True,
        "publication_eligible": True,
    }
    for key, expected_value in frozen_scalars.items():
        if report.get(key) != expected_value:
            raise ValueError(
                f"manifest report {key} mismatch: expected {expected_value!r}, "
                f"observed {report.get(key)!r}"
            )
    eligibility = report.get("eligibility")
    expected_eligibility = {
        "stage": "pre_codec_pre_marginal",
        "all_pcm_samples_finite_required": True,
        "pre_downmix_rms_denominator": PRE_DOWNMIX_RMS_DENOMINATOR,
        "minimum_mono_rms": MIN_MONO_RMS,
        "minimum_mono_compatibility_ratio": MIN_MONO_COMPATIBILITY_RATIO,
        "codec_sample_rate": FROZEN_CODEC_SAMPLE_RATE,
        "codec_channels": FROZEN_CODEC_CHANNELS,
        "codec_input_frames": FROZEN_CODEC_INPUT_FRAMES,
        "conversion": "audiocraft.data.audio_utils.convert_audio",
        "lr_pearson_is_diagnostic_only": True,
        "track_id_filtering": False,
        "audited_tracks": EXPECTED_ORIGINAL_CANDIDATES,
    }
    if eligibility != expected_eligibility:
        raise ValueError("manifest report eligibility contract mismatch")
    if not isinstance(report.get("input_eligible_tracks"), int) or report["input_eligible_tracks"] < EXPECTED_ITEMS:
        raise ValueError("manifest report has too few input-eligible tracks")
    audit = report.get("eligibility_audit")
    if not isinstance(audit, list) or len(audit) != EXPECTED_ORIGINAL_CANDIDATES:
        raise ValueError("manifest report does not audit exactly 7994 candidates")
    audit_by_id: Dict[int, Mapping[str, Any]] = {}
    eligible_audit_rows: List[Mapping[str, Any]] = []
    previous_track_id: Optional[int] = None
    for audit_index, entry in enumerate(audit):
        if not isinstance(entry, dict):
            raise ValueError(f"eligibility audit row {audit_index} is not an object")
        track_id = int(entry.get("fma_track_id"))
        if previous_track_id is not None and track_id <= previous_track_id:
            raise ValueError("manifest report eligibility audit is not in increasing track-ID order")
        previous_track_id = track_id
        if track_id in audit_by_id:
            raise ValueError("manifest report eligibility audit contains duplicate track IDs")
        for field in (
            "relative_audio_path",
            "source_audio_sha256",
            "decoded_duration_frames",
            "decoded_sample_rate",
            "decoded_channels",
            "segment_start_frame",
            "segment_num_frames",
            "segment_offset_sha256",
            "extracted_pcm_sha256",
            "all_pcm_samples_finite",
            "pre_downmix_rms",
            "mono_rms",
            "mono_compatibility_ratio",
            "lr_pearson_correlation",
            "eligibility_codec_input_pcm_sha256",
            "eligible",
            "failed_conditions",
        ):
            if field not in entry:
                raise ValueError(f"eligibility audit row {audit_index} lacks {field}")
        if entry["segment_offset_sha256"] != _segment_hash(track_id):
            raise ValueError(f"eligibility audit row {audit_index} segment hash drifted")
        if type(entry["all_pcm_samples_finite"]) is not bool or type(entry["eligible"]) is not bool:
            raise ValueError(f"eligibility audit row {audit_index} has non-boolean flags")
        if not isinstance(entry["failed_conditions"], list):
            raise ValueError(f"eligibility audit row {audit_index} failed_conditions is not a list")
        if entry["all_pcm_samples_finite"]:
            pre_rms = _finite_float(entry["pre_downmix_rms"], f"audit {audit_index} pre RMS")
            mono_rms = _finite_float(entry["mono_rms"], f"audit {audit_index} mono RMS")
            ratio = _finite_float(entry["mono_compatibility_ratio"], f"audit {audit_index} ratio")
            if pre_rms < 0.0 or mono_rms < 0.0 or ratio < 0.0:
                raise ValueError(f"eligibility audit row {audit_index} contains negative energy")
            expected_failed = []
            if mono_rms < MIN_MONO_RMS:
                expected_failed.append("mono_rms_below_1e-5")
            if ratio < MIN_MONO_COMPATIBILITY_RATIO:
                expected_failed.append("mono_compatibility_ratio_below_1e-3")
            if entry["failed_conditions"] != expected_failed or entry["eligible"] != (not expected_failed):
                raise ValueError(f"eligibility audit row {audit_index} rule outcome mismatch")
            if not re.fullmatch(
                r"[0-9a-f]{64}",
                str(entry["eligibility_codec_input_pcm_sha256"]),
            ):
                raise ValueError(f"eligibility audit row {audit_index} codec-input hash malformed")
        else:
            if entry["eligible"] or entry["failed_conditions"] != ["non_finite_pcm"]:
                raise ValueError(f"eligibility audit row {audit_index} non-finite rule mismatch")
            for field in (
                "pre_downmix_rms",
                "mono_rms",
                "mono_compatibility_ratio",
                "lr_pearson_correlation",
                "eligibility_codec_input_pcm_sha256",
            ):
                if entry[field] is not None:
                    raise ValueError(f"eligibility audit row {audit_index} non-finite diagnostics must be null")
        audit_by_id[track_id] = entry
        if entry["eligible"]:
            eligible_audit_rows.append(entry)
    if len(eligible_audit_rows) != report["input_eligible_tracks"]:
        raise ValueError("manifest report input-eligible count does not match its audit")
    expected_selected_ids = [
        int(entry["fma_track_id"])
        for entry in sorted(
            eligible_audit_rows,
            key=lambda item: (
                _selection_hash(int(item["fma_track_id"])),
                int(item["fma_track_id"]),
            ),
        )[:EXPECTED_ITEMS]
    ]
    if [int(row["fma_track_id"]) for row in rows] != expected_selected_ids:
        raise ValueError("manifest does not contain the 512 lowest-hash eligible audit tracks")
    compared_fields = (
        "relative_audio_path",
        "source_audio_sha256",
        "segment_start_frame",
        "segment_num_frames",
        "extracted_pcm_sha256",
        "all_pcm_samples_finite",
        "pre_downmix_rms",
        "mono_rms",
        "mono_compatibility_ratio",
        "lr_pearson_correlation",
        "eligibility_codec_input_pcm_sha256",
        "eligible",
        "failed_conditions",
    )
    for row in rows:
        audit_row = audit_by_id.get(int(row["fma_track_id"]))
        if audit_row is None:
            raise ValueError("selected manifest track is absent from eligibility audit")
        for field in compared_fields:
            if audit_row.get(field) != row[field]:
                raise ValueError(
                    f"manifest/report eligibility mismatch for track {row['fma_track_id']} field {field}"
                )
    implementation = report.get("implementation")
    builder_path = WORKPACK_ROOT / "scripts" / "build_fma_calibration_manifest.py"
    if not isinstance(implementation, dict) or implementation.get("builder_sha256") != sha256_file(builder_path):
        raise ValueError("manifest report builder SHA-256 does not match this workpack")
    audiocraft = report.get("audiocraft")
    if not isinstance(audiocraft, dict) or audiocraft.get("base_commit") != PINNED_AUDIOCRAFT_BASE_COMMIT:
        raise ValueError("manifest report AudioCraft base commit mismatch")
    if not isinstance(audiocraft.get("source_identity"), dict):
        raise ValueError("manifest report lacks AudioCraft source identity")
    return observed, report


def safe_source_path(source_root: Path, relative_audio_path: str) -> Path:
    root = source_root.resolve()
    candidate = (root / relative_audio_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"relative audio path escapes source root: {relative_audio_path}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"calibration source is missing: {candidate}")
    return candidate


def split_key(track_id: int) -> str:
    material = f"{SPLIT_DOMAIN}|{SPLIT_SEED}|{int(track_id)}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def kendall_tau_b(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    """Small dependency-free Kendall tau-b over paired codebook statistics."""

    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Kendall inputs must have equal length >= 2")
    concordant = discordant = ties_left = ties_right = 0
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            dx = float(left[i]) - float(left[j])
            dy = float(right[i]) - float(right[j])
            if dx == 0.0 and dy == 0.0:
                continue
            if dx == 0.0:
                ties_left += 1
            elif dy == 0.0:
                ties_right += 1
            elif dx * dy > 0.0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_left)
        * (concordant + discordant + ties_right)
    )
    if denominator == 0.0:
        return None
    return (concordant - discordant) / denominator


def aggregate_marginals(
    marginals: Any,
    track_ids: Sequence[int],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    marginal_epsilon: float = 1.0e-8,
) -> Dict[str, Any]:
    """Aggregate clip-by-codebook marginals with frozen bootstrap/split rules."""

    import numpy as np

    values = np.asarray(marginals, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("marginals must have shape [N,Q] with N,Q >= 2")
    if len(track_ids) != values.shape[0] or len(set(int(value) for value in track_ids)) != len(track_ids):
        raise ValueError("track_ids must be unique and align with marginals")
    if not np.isfinite(values).all():
        raise ValueError("marginals contain NaN/Inf")
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    if not math.isfinite(float(marginal_epsilon)) or marginal_epsilon <= 0.0:
        raise ValueError("marginal_epsilon must be finite and positive")

    def normalized(vector: Any) -> Any:
        local_clipped = np.maximum(np.asarray(vector, dtype=np.float64), float(marginal_epsilon))
        return local_clipped / local_clipped.sum()

    raw_mean = values.mean(axis=0)
    clipped = np.maximum(raw_mean, float(marginal_epsilon))
    prior = normalized(raw_mean)
    clip_std = values.std(axis=0, ddof=1)
    clip_median = np.median(values, axis=0)
    clip_positive_fraction = (values > 0.0).mean(axis=0)
    trim_each_tail = int(math.floor(0.01 * values.shape[0]))
    if 2 * trim_each_tail >= values.shape[0]:
        raise ValueError("1% trimmed mean leaves no retained samples")
    sorted_values = np.sort(values, axis=0)
    retained = (
        sorted_values
        if trim_each_tail == 0
        else sorted_values[trim_each_tail : values.shape[0] - trim_each_tail]
    )
    trimmed_mean = retained.mean(axis=0)
    median_prior = normalized(clip_median)
    trimmed_prior = normalized(trimmed_mean)
    median_tau = kendall_tau_b(raw_mean.tolist(), clip_median.tolist())
    trimmed_tau = kendall_tau_b(raw_mean.tolist(), trimmed_mean.tolist())

    total = values.sum(axis=0)
    leave_one_out = []
    for index, track_id in enumerate(track_ids):
        without_mean = (total - values[index]) / float(values.shape[0] - 1)
        without_prior = normalized(without_mean)
        tv = float(0.5 * np.abs(prior - without_prior).sum())
        leave_one_out.append(
            {
                "manifest_index": index,
                "fma_track_id": int(track_id),
                "raw_mean_marginal_without_clip": without_mean.tolist(),
                "prior_without_clip": without_prior.tolist(),
                "total_variation_from_primary_prior": tv,
            }
        )
    maximum_influence = max(
        leave_one_out,
        key=lambda item: (item["total_variation_from_primary_prior"], -item["manifest_index"]),
    )

    generator = np.random.Generator(np.random.PCG64(int(bootstrap_seed)))
    bootstrap_raw = np.empty((bootstrap_replicates, values.shape[1]), dtype=np.float64)
    bootstrap_prior = np.empty_like(bootstrap_raw)
    cursor = 0
    while cursor < bootstrap_replicates:
        count = min(256, bootstrap_replicates - cursor)
        indices = generator.integers(0, values.shape[0], size=(count, values.shape[0]))
        means = values[indices].mean(axis=1)
        boot_clipped = np.maximum(means, float(marginal_epsilon))
        bootstrap_raw[cursor : cursor + count] = means
        bootstrap_prior[cursor : cursor + count] = boot_clipped / boot_clipped.sum(axis=1, keepdims=True)
        cursor += count
    raw_ci = np.percentile(bootstrap_raw, [2.5, 97.5], axis=0)
    prior_ci = np.percentile(bootstrap_prior, [2.5, 97.5], axis=0)
    bootstrap_uniform = np.full(values.shape[1], 1.0 / values.shape[1], dtype=np.float64)
    bootstrap_tv = 0.5 * np.abs(bootstrap_prior - bootstrap_uniform).sum(axis=1)
    bootstrap_tv_ci = np.percentile(bootstrap_tv, [2.5, 97.5])

    ordered = sorted(range(values.shape[0]), key=lambda index: (split_key(int(track_ids[index])), int(track_ids[index])))
    half_a_indices = ordered[0::2]
    half_b_indices = ordered[1::2]
    half_a_mean = values[half_a_indices].mean(axis=0)
    half_b_mean = values[half_b_indices].mean(axis=0)
    tau = kendall_tau_b(half_a_mean.tolist(), half_b_mean.tolist())

    uniform = np.full(values.shape[1], 1.0 / values.shape[1], dtype=np.float64)
    positive = raw_mean[raw_mean > 0.0]
    positive_ratio = None if positive.size == 0 else float(positive.max() / positive.min())
    return {
        "n_clips": int(values.shape[0]),
        "num_codebooks": int(values.shape[1]),
        "raw_mean_marginal": raw_mean.tolist(),
        "clip_marginal_std": clip_std.tolist(),
        "clip_marginal_median": clip_median.tolist(),
        "clip_positive_marginal_fraction": clip_positive_fraction.tolist(),
        "clipped_mean_marginal": clipped.tolist(),
        "prior": prior.tolist(),
        "bootstrap": {
            "replicates": int(bootstrap_replicates),
            "seed": int(bootstrap_seed),
            "rng": "NumPy-PCG64",
            "interval": "clip-bootstrap percentile 95%",
            "raw_mean_marginal_ci95_low": raw_ci[0].tolist(),
            "raw_mean_marginal_ci95_high": raw_ci[1].tolist(),
            "prior_ci95_low": prior_ci[0].tolist(),
            "prior_ci95_high": prior_ci[1].tolist(),
            "total_variation_from_uniform_ci95_low": float(bootstrap_tv_ci[0]),
            "total_variation_from_uniform_ci95_high": float(bootstrap_tv_ci[1]),
        },
        "split_half": {
            "domain": SPLIT_DOMAIN,
            "seed": SPLIT_SEED,
            "assignment": "sort by domain-separated SHA-256, then alternate even/odd ranks",
            "half_a_n": len(half_a_indices),
            "half_b_n": len(half_b_indices),
            "half_a_mean_marginal": half_a_mean.tolist(),
            "half_b_mean_marginal": half_b_mean.tolist(),
            "kendall_tau_b": tau,
        },
        "leave_one_out_influence": {
            "definition": "TV(primary_prior, prior_recomputed_without_one_clip)",
            "max_total_variation": maximum_influence[
                "total_variation_from_primary_prior"
            ],
            "max_manifest_index": maximum_influence["manifest_index"],
            "max_fma_track_id": maximum_influence["fma_track_id"],
            "per_clip": leave_one_out,
        },
        "sensitivity": {
            "primary_estimator": "arithmetic_mean",
            "median": {
                "marginal": clip_median.tolist(),
                "prior": median_prior.tolist(),
                "kendall_tau_b_vs_primary": median_tau,
            },
            "trimmed_mean_1pct": {
                "definition": "sort each codebook independently; remove floor(0.01*N) from each tail",
                "trim_each_tail": trim_each_tail,
                "retained_clips_per_codebook": int(values.shape[0] - 2 * trim_each_tail),
                "marginal": trimmed_mean.tolist(),
                "prior": trimmed_prior.tolist(),
                "kendall_tau_b_vs_primary": trimmed_tau,
            },
            "use_for_primary_prior": False,
        },
        "diagnostics": {
            "total_variation_from_uniform": float(0.5 * np.abs(prior - uniform).sum()),
            "largest_to_smallest_positive_raw_mean_marginal_ratio": positive_ratio,
            "clips_with_any_negative_marginal_fraction": float((values < 0.0).any(axis=1).mean()),
            "denominator_floor_activation": 0,
        },
    }


def summary_csv_bytes(aggregate: Mapping[str, Any]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "codebook_index",
            "raw_mean_marginal",
            "clip_marginal_std",
            "clip_marginal_median",
            "trimmed_mean_1pct_marginal",
            "clip_positive_marginal_fraction",
            "clipped_mean_marginal",
            "prior",
            "raw_ci95_low",
            "raw_ci95_high",
            "prior_ci95_low",
            "prior_ci95_high",
            "split_half_a_mean",
            "split_half_b_mean",
        ]
    )
    bootstrap = aggregate["bootstrap"]
    split = aggregate["split_half"]
    for q in range(int(aggregate["num_codebooks"])):
        writer.writerow(
            [
                q,
                format(float(aggregate["raw_mean_marginal"][q]), ".17g"),
                format(float(aggregate["clip_marginal_std"][q]), ".17g"),
                format(float(aggregate["clip_marginal_median"][q]), ".17g"),
                format(float(aggregate["sensitivity"]["trimmed_mean_1pct"]["marginal"][q]), ".17g"),
                format(float(aggregate["clip_positive_marginal_fraction"][q]), ".17g"),
                format(float(aggregate["clipped_mean_marginal"][q]), ".17g"),
                format(float(aggregate["prior"][q]), ".17g"),
                format(float(bootstrap["raw_mean_marginal_ci95_low"][q]), ".17g"),
                format(float(bootstrap["raw_mean_marginal_ci95_high"][q]), ".17g"),
                format(float(bootstrap["prior_ci95_low"][q]), ".17g"),
                format(float(bootstrap["prior_ci95_high"][q]), ".17g"),
                format(float(split["half_a_mean_marginal"][q]), ".17g"),
                format(float(split["half_b_mean_marginal"][q]), ".17g"),
            ]
        )
    return buffer.getvalue().encode("utf-8")


def deterministic_gzip_jsonl(records: Iterable[Mapping[str, Any]]) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as handle:
        for record in records:
            handle.write(_json_line(record))
    return buffer.getvalue()


def _checkpoint_payload_path(checkpoint: Path) -> Path:
    checkpoint = checkpoint.resolve()
    if checkpoint.is_file():
        return checkpoint
    candidate = checkpoint / "compression_state_dict.bin"
    if checkpoint.is_dir() and candidate.is_file():
        return candidate
    raise FileNotFoundError(
        "codec checkpoint must be a local file or a directory containing compression_state_dict.bin"
    )


def _inspect_local_checkpoint_package(torch_module: Any, payload: Path) -> Dict[str, Any]:
    """Inspect AudioCraft's two official compression export shapes."""

    package = torch_module.load(str(payload), map_location="cpu")
    if not isinstance(package, Mapping):
        raise RuntimeError("AudioCraft compression checkpoint payload is not a mapping")
    if "pretrained" in package:
        reference = package["pretrained"]
        if reference != ALLOWED_PRETRAINED_CODEC_ID:
            raise RuntimeError(
                "AudioCraft pretrained compression indirection is not on the frozen whitelist: "
                f"expected {ALLOWED_PRETRAINED_CODEC_ID!r}, observed {reference!r}"
            )
        result = {
            "load_mode": "pretrained_indirection",
            "pretrained_model_id": reference,
        }
        del package
        return result
    missing = {"xp.cfg", "best_state"}.difference(package)
    if missing:
        raise RuntimeError(
            f"local compression checkpoint is missing AudioCraft fields: {sorted(missing)}"
        )
    result = {"load_mode": "self_contained", "pretrained_model_id": None}
    del package
    return result


def _require_offline_hf_environment() -> None:
    required = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    missing = [name for name in required if os.environ.get(name) != "1"]
    if missing:
        raise RuntimeError(
            "pretrained-indirection loading is allowed only in strict offline mode; "
            "set these variables to exactly 1 before process start: " + ", ".join(missing)
        )


def _resolved_snapshot_root(path: Path) -> Tuple[Path, Optional[str]]:
    # Do not resolve a config/weight symlink before finding ``snapshots``: HF
    # cache files commonly point into ``blobs/``, which would erase the commit
    # identity from the visible path.
    visible = path.absolute()
    parts = visible.parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            root = Path(*parts[: index + 2])
            return root, parts[index + 1]
    return (visible if visible.is_dir() else visible.parent), None


def resolved_hf_snapshot_from_cache(model_id: str) -> Tuple[Path, str]:
    """Resolve the exact cached snapshot selected for the default revision."""

    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError as exc:  # pragma: no cover - required by AudioCraft.
        raise RuntimeError("huggingface_hub is required to audit the offline codec cache") from exc
    cached_config = try_to_load_from_cache(repo_id=model_id, filename="config.json")
    if not isinstance(cached_config, str):
        raise RuntimeError(
            f"offline HF cache has no resolved config.json for {model_id!r}; cache miss is fatal"
        )
    config_path = Path(cached_config)
    if not config_path.is_file():
        raise RuntimeError(f"cached HF config path is not a file: {config_path}")
    snapshot_root, revision = _resolved_snapshot_root(config_path)
    if revision is None:
        raise RuntimeError(
            "cached HF config path does not expose a snapshots/<revision> identity: "
            f"{config_path}"
        )
    return snapshot_root, revision


def hash_resolved_model_snapshot(model: Any, model_id: str = ALLOWED_PRETRAINED_CODEC_ID) -> Dict[str, Any]:
    """Hash every regular file in the resolved local HF snapshot."""

    config = getattr(getattr(model, "model", None), "config", None)
    config_source = getattr(config, "_name_or_path", None)
    if isinstance(config_source, str) and config_source and Path(config_source).exists():
        snapshot_root, revision = _resolved_snapshot_root(Path(config_source))
        if revision is None:
            snapshot_root, revision = resolved_hf_snapshot_from_cache(model_id)
    else:
        # Transformers commonly retains the repo ID rather than the resolved
        # path in config._name_or_path.  Resolve config.json through the same
        # offline HF cache and recover its snapshots/<commit> parent.
        snapshot_root, revision = resolved_hf_snapshot_from_cache(model_id)
    if not snapshot_root.is_dir():
        raise RuntimeError(f"resolved HF snapshot root is not a directory: {snapshot_root}")
    files: List[Dict[str, Any]] = []
    for visible_path in sorted(snapshot_root.rglob("*")):
        if visible_path.is_symlink() and not visible_path.exists():
            raise RuntimeError(f"resolved HF snapshot contains a broken symlink: {visible_path}")
        if not visible_path.is_file():
            continue
        resolved_path = visible_path.resolve(strict=True)
        if not resolved_path.is_file():
            raise RuntimeError(f"resolved HF snapshot entry is not a regular file: {resolved_path}")
        files.append(
            {
                "relative_path": visible_path.relative_to(snapshot_root).as_posix(),
                "visible_absolute_path": str(visible_path.absolute()),
                "resolved_absolute_path": str(resolved_path),
                "is_symlink": visible_path.is_symlink(),
                "size_bytes": int(resolved_path.stat().st_size),
                "sha256": sha256_file(resolved_path),
            }
        )
    if not files:
        raise RuntimeError(f"resolved HF snapshot contains no regular files: {snapshot_root}")
    return {
        "resolved_snapshot_root": str(snapshot_root.resolve()),
        "resolved_snapshot_revision": revision,
        "resolved_snapshot_files": files,
    }


def _git_commit(repository: Path) -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _path_exists_even_if_broken_symlink(path: Path) -> bool:
    return os.path.lexists(str(path))


def require_output_dir_absent(output_dir: Path) -> Path:
    """Resolve the publication target without creating it or its parent."""

    target = Path(os.path.abspath(str(output_dir)))
    if target.name in {"", ".", ".."}:
        raise ValueError(f"invalid --output-dir publication target: {output_dir}")
    if _path_exists_even_if_broken_symlink(target):
        raise FileExistsError(
            "formal A1 output directory must not exist at all; archive the whole "
            f"previous directory before retrying: {target}"
        )
    return target


def guard_single_process_environment(environment: Optional[Mapping[str, str]] = None) -> None:
    """Fail closed if this rank-zero assay is launched through torchrun/Slurm."""

    values = os.environ if environment is None else environment
    allowed = {
        "WORLD_SIZE": {None, "", "1"},
        "RANK": {None, "", "0"},
        "LOCAL_RANK": {None, "", "0"},
        "SLURM_NTASKS": {None, "", "1"},
        "SLURM_PROCID": {None, "", "0"},
    }
    polluted = {
        name: values.get(name)
        for name, accepted in allowed.items()
        if values.get(name) not in accepted
    }
    if polluted:
        raise RuntimeError(
            "codec-prior estimation is rank-zero/single-GPU only; unset distributed "
            f"launcher variables (observed {polluted})"
        )


def _write_fsynced_file(path: Path, payload: bytes) -> None:
    if path.name in {"", ".", ".."} or path.parent == path:
        raise ValueError(f"invalid staged output path: {path}")
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _artifact_seal_bytes(
    payloads: Mapping[str, bytes], identity: Mapping[str, Any]
) -> bytes:
    if set(payloads) != set(OUTPUT_NAMES):
        raise ValueError(
            f"transaction payload names must be exactly {list(OUTPUT_NAMES)}"
        )
    seal = {
        "schema_version": ARTIFACT_SEAL_SCHEMA,
        "status": "complete",
        "manifest": identity["manifest"],
        "checkpoint": identity["checkpoint"],
        "source_identity": identity["source_identity"],
        "audiocraft": identity["audiocraft"],
        "codec_load": identity["codec_load"],
        "protocol": identity["protocol"],
        "artifacts": {
            name: {
                "sha256": sha256_bytes(payloads[name]),
                "size_bytes": len(payloads[name]),
            }
            for name in sorted(payloads)
        },
    }
    return json.dumps(
        seal,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def _validate_staged_artifact_directory(
    staging_dir: Path,
    payloads: Mapping[str, bytes],
    identity: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    expected_names = set(OUTPUT_NAMES) | {ARTIFACT_SEAL_NAME}
    observed_names = {path.name for path in staging_dir.iterdir()}
    if observed_names != expected_names:
        raise RuntimeError(
            "staged A1 directory is incomplete or has unexpected files: "
            f"expected={sorted(expected_names)}, observed={sorted(observed_names)}"
        )
    seal_path = staging_dir / ARTIFACT_SEAL_NAME
    if seal_path.is_symlink() or not seal_path.is_file():
        raise RuntimeError("staged ARTIFACT_SEAL.json is not a regular file")
    try:
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("staged ARTIFACT_SEAL.json is unreadable") from exc
    if seal.get("schema_version") != ARTIFACT_SEAL_SCHEMA or seal.get("status") != "complete":
        raise RuntimeError("staged artifact seal is not complete or has the wrong schema")
    for key in (
        "manifest",
        "checkpoint",
        "source_identity",
        "audiocraft",
        "codec_load",
        "protocol",
    ):
        if seal.get(key) != identity.get(key):
            raise RuntimeError(f"staged artifact seal identity mismatch for {key}")
    verified: Dict[str, Dict[str, Any]] = {}
    for name in OUTPUT_NAMES:
        path = staging_dir / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"staged artifact is not a regular file: {path}")
        expected = seal.get("artifacts", {}).get(name)
        observed = {"sha256": sha256_file(path), "size_bytes": int(path.stat().st_size)}
        if expected != observed:
            raise RuntimeError(
                f"staged artifact hash/size mismatch for {name}: "
                f"expected={expected}, observed={observed}"
            )
        if path.read_bytes() != payloads[name]:
            raise RuntimeError(f"staged artifact bytes changed for {name}")
        verified[name] = observed
    try:
        prior_payload = json.loads(payloads["codec_prior.json"])
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("staged codec_prior.json is invalid") from exc
    if prior_payload.get("protocol") != identity.get("protocol"):
        raise RuntimeError("staged prior/seal protocol identity mismatch")
    if prior_payload.get("manifest_sha256") != identity["manifest"]["sha256"]:
        raise RuntimeError("staged prior/seal manifest identity mismatch")
    report_sha = identity["protocol"]["manifest_report"]["sha256"]
    if (
        prior_payload.get("manifest_report_sha256") != report_sha
        or identity["source_identity"].get("manifest_report_sha256") != report_sha
    ):
        raise RuntimeError("staged prior/seal manifest-report identity mismatch")
    return verified


def publish_artifact_directory(
    output_dir: Path,
    payloads: Mapping[str, bytes],
    identity: Mapping[str, Any],
) -> Dict[str, Path]:
    """Atomically publish all A1 artifacts as one sealed directory generation."""

    target = require_output_dir_absent(output_dir)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    target = require_output_dir_absent(target)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging.", dir=str(parent))
    )
    published = False
    try:
        for name in OUTPUT_NAMES:
            _write_fsynced_file(staging_dir / name, payloads[name])
        _write_fsynced_file(
            staging_dir / ARTIFACT_SEAL_NAME,
            _artifact_seal_bytes(payloads, identity),
        )
        _validate_staged_artifact_directory(staging_dir, payloads, identity)
        staging_fd = os.open(str(staging_dir), os.O_RDONLY)
        try:
            os.fsync(staging_fd)
        finally:
            os.close(staging_fd)
        require_output_dir_absent(target)
        os.replace(str(staging_dir), str(target))
        published = True
        parent_fd = os.open(str(parent), os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return {name: target / name for name in OUTPUT_NAMES}
    finally:
        if not published and _path_exists_even_if_broken_symlink(staging_dir):
            shutil.rmtree(staging_dir)


def _load_and_verify_segment(row: Mapping[str, Any], source_root: Path) -> Tuple[Any, int]:
    import numpy as np
    import soundfile as sf

    path = safe_source_path(source_root, str(row["relative_audio_path"]))
    observed_source_hash = sha256_file(path)
    if observed_source_hash != str(row["source_audio_sha256"]).lower():
        raise RuntimeError(
            f"source SHA-256 mismatch for track {row['fma_track_id']}: "
            f"expected {row['source_audio_sha256']}, observed {observed_source_hash}"
        )
    with sf.SoundFile(str(path), mode="r") as handle:
        if int(handle.frames) != int(row["decoded_duration_frames"]):
            raise RuntimeError(f"decoded frame count changed for track {row['fma_track_id']}")
        if int(handle.samplerate) != int(row["decoded_sample_rate"]):
            raise RuntimeError(f"decoded sample rate changed for track {row['fma_track_id']}")
        if int(handle.channels) != int(row["decoded_channels"]):
            raise RuntimeError(f"decoded channel count changed for track {row['fma_track_id']}")
        handle.seek(int(row["segment_start_frame"]))
        pcm = handle.read(
            frames=int(row["segment_num_frames"]),
            dtype="float32",
            always_2d=True,
            fill_value=None,
        )
    expected_shape = (int(row["segment_num_frames"]), int(row["decoded_channels"]))
    if pcm.shape != expected_shape or not bool(np.isfinite(pcm).all()):
        raise RuntimeError(f"invalid deterministic PCM for track {row['fma_track_id']}")
    observed_pcm_hash = pcm_f32le_sha256(pcm)
    if observed_pcm_hash != str(row["extracted_pcm_sha256"]).lower():
        raise RuntimeError(
            f"PCM SHA-256 mismatch for track {row['fma_track_id']}: "
            f"expected {row['extracted_pcm_sha256']}, observed {observed_pcm_hash}"
        )
    return pcm, int(row["decoded_sample_rate"])


def _rms_float64(np: Any, samples: Any) -> float:
    values = np.asarray(samples, dtype=np.float64)
    return math.sqrt(float(np.mean(np.square(values, dtype=np.float64), dtype=np.float64)))


def _lr_pearson(np: Any, pcm: Any) -> Optional[float]:
    if pcm.ndim != 2 or pcm.shape[1] != 2:
        return None
    left = np.asarray(pcm[:, 0], dtype=np.float64)
    right = np.asarray(pcm[:, 1], dtype=np.float64)
    left = left - left.mean(dtype=np.float64)
    right = right - right.mean(dtype=np.float64)
    denominator = math.sqrt(float(np.dot(left, left)) * float(np.dot(right, right)))
    if denominator == 0.0:
        return None
    return float(max(-1.0, min(1.0, float(np.dot(left, right)) / denominator)))


def recompute_and_validate_eligibility(
    row: Mapping[str, Any],
    pcm: Any,
    codec_input_cpu: Any,
    *,
    np: Any,
    torch_module: Any,
) -> Dict[str, Any]:
    """Recompute a selected row's v2 eligibility before codec encode."""

    pcm_array = np.ascontiguousarray(pcm, dtype=np.float32)
    if not bool(np.isfinite(pcm_array).all()):
        raise RuntimeError(f"non-finite source PCM for track {row['fma_track_id']}")
    expected_shape = (1, FROZEN_CODEC_CHANNELS, FROZEN_CODEC_INPUT_FRAMES)
    if tuple(codec_input_cpu.shape) != expected_shape:
        raise RuntimeError(
            f"eligibility codec input must have shape {expected_shape}, got {tuple(codec_input_cpu.shape)}"
        )
    if codec_input_cpu.device.type != "cpu" or codec_input_cpu.dtype != torch_module.float32:
        raise RuntimeError("eligibility codec input must be CPU float32")
    if not bool(torch_module.isfinite(codec_input_cpu).all().item()):
        raise RuntimeError("eligibility codec input contains NaN/Inf")
    mono_frame_major = codec_input_cpu[0].transpose(0, 1).detach().numpy()
    pre_rms = _rms_float64(np, pcm_array)
    mono_rms = _rms_float64(np, mono_frame_major)
    ratio = mono_rms / max(pre_rms, PRE_DOWNMIX_RMS_DENOMINATOR)
    observed = {
        "all_pcm_samples_finite": True,
        "pre_downmix_rms": pre_rms,
        "mono_rms": mono_rms,
        "mono_compatibility_ratio": ratio,
        "lr_pearson_correlation": _lr_pearson(np, pcm_array),
        "eligibility_codec_input_pcm_sha256": pcm_f32le_sha256(mono_frame_major),
    }
    for key in ("pre_downmix_rms", "mono_rms", "mono_compatibility_ratio"):
        if not math.isclose(
            float(observed[key]),
            float(row[key]),
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ):
            raise RuntimeError(
                f"eligibility {key} mismatch for track {row['fma_track_id']}: "
                f"manifest={row[key]}, recomputed={observed[key]}"
            )
    if observed["lr_pearson_correlation"] != row["lr_pearson_correlation"]:
        raise RuntimeError(
            f"eligibility L/R diagnostic mismatch for track {row['fma_track_id']}"
        )
    if observed["eligibility_codec_input_pcm_sha256"] != row["eligibility_codec_input_pcm_sha256"]:
        raise RuntimeError(
            f"eligibility codec-input hash mismatch for track {row['fma_track_id']}"
        )
    if mono_rms < MIN_MONO_RMS or ratio < MIN_MONO_COMPATIBILITY_RATIO:
        raise RuntimeError(
            f"track {row['fma_track_id']} fails A1-R2 eligibility on recomputation"
        )
    return observed


def _load_runtime(audiocraft_root: Path, device: str) -> Tuple[Any, Any, Any, Any, Any]:
    try:
        import numpy as np
        import soundfile as sf
        import torch
    except ImportError as exc:
        raise RuntimeError("NumPy, soundfile, and PyTorch are required") from exc
    if not device.startswith("cuda"):
        raise ValueError("real codec prior estimation is GPU-gated; --device must be cuda:<index>")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; do not report the real-codec A1 assay as run")
    audiocraft_root = audiocraft_root.resolve()
    if not (audiocraft_root / "audiocraft").is_dir():
        raise ValueError(f"--audiocraft-root does not contain the audiocraft package: {audiocraft_root}")
    if str(audiocraft_root) not in sys.path:
        sys.path.insert(0, str(audiocraft_root))
    from audiocraft.data.audio_utils import convert_audio
    from audiocraft.models.loaders import load_compression_model
    import audiocraft

    imported = Path(audiocraft.__file__).resolve()
    try:
        imported.relative_to(audiocraft_root)
    except ValueError as exc:
        raise RuntimeError(f"imported AudioCraft from the wrong tree: {imported}") from exc
    return np, sf, torch, convert_audio, load_compression_model


def run_codec_assay(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from ptc_opd.perceptual_prior import (
        DEFAULT_FFT_SIZES,
        mrstft_reference_norms,
        progressive_prior,
    )
    from ptc_opd.reproducibility import audiocraft_source_identity

    manifest_hash, rows = manifest_sha256_and_rows(args.manifest.resolve(), args.manifest_sha256)
    if args.manifest.resolve().parent != args.manifest_report.resolve().parent:
        raise ValueError("A1-R2 manifest and report must share the same a1-r2 directory")
    manifest_report_hash, manifest_report = manifest_report_sha256_and_payload(
        args.manifest_report.resolve(),
        args.manifest_report_sha256,
        manifest_sha256=manifest_hash,
        rows=rows,
    )
    code_identity = implementation_identity()
    audiocraft_root = args.audiocraft_root.resolve()
    audiocraft_commit = _git_commit(audiocraft_root)
    if audiocraft_commit != PINNED_AUDIOCRAFT_BASE_COMMIT:
        raise RuntimeError(
            "A1 requires pinned AudioCraft base commit {}; observed {}".format(
                PINNED_AUDIOCRAFT_BASE_COMMIT, audiocraft_commit
            )
        )
    audiocraft_identity = audiocraft_source_identity(audiocraft_root)
    if manifest_report["audiocraft"]["source_identity"] != audiocraft_identity:
        raise RuntimeError(
            "manifest report AudioCraft source identity differs from the estimator runtime"
        )
    np, sf, torch, convert_audio, load_compression_model = _load_runtime(
        args.audiocraft_root, args.device
    )
    checkpoint_payload = _checkpoint_payload_path(args.codec_checkpoint)
    observed_checkpoint_hash = sha256_file(checkpoint_payload)
    expected_checkpoint_hash = args.codec_checkpoint_sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_checkpoint_hash):
        raise ValueError("--codec-checkpoint-sha256 must be 64 hexadecimal characters")
    if observed_checkpoint_hash != expected_checkpoint_hash:
        raise RuntimeError(
            f"codec checkpoint SHA-256 mismatch: expected {expected_checkpoint_hash}, "
            f"observed {observed_checkpoint_hash}"
        )

    checkpoint_package = _inspect_local_checkpoint_package(torch, checkpoint_payload)
    if checkpoint_package["load_mode"] == "pretrained_indirection":
        _require_offline_hf_environment()

    torch.manual_seed(BOOTSTRAP_SEED)
    torch.cuda.manual_seed_all(BOOTSTRAP_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_compression_model(str(args.codec_checkpoint.resolve()), device=args.device)
    model.float()
    model.requires_grad_(False)
    model.eval()
    codec_contract = validate_codec_model_contract(model)
    resolved_snapshot = (
        hash_resolved_model_snapshot(model)
        if checkpoint_package["load_mode"] == "pretrained_indirection"
        else {
            "resolved_snapshot_root": None,
            "resolved_snapshot_revision": None,
            "resolved_snapshot_files": [],
        }
    )
    q_model = FROZEN_CODEC_CODEBOOKS

    per_clip: List[Dict[str, Any]] = []
    all_marginals: List[List[float]] = []
    prefix_smoke_passed = False
    device = torch.device(args.device)
    with torch.inference_mode():
        for index, row in enumerate(rows):
            pcm, source_rate = _load_and_verify_segment(row, args.source_root)
            source_waveform = torch.from_numpy(
                np.asarray(pcm, dtype=np.float32).T.copy()
            ).unsqueeze(0)
            codec_input_cpu = convert_audio(
                source_waveform,
                from_rate=source_rate,
                to_rate=FROZEN_CODEC_SAMPLE_RATE,
                to_channels=FROZEN_CODEC_CHANNELS,
            ).contiguous().to(device="cpu", dtype=torch.float32)
            eligibility = recompute_and_validate_eligibility(
                row,
                pcm,
                codec_input_cpu,
                np=np,
                torch_module=torch,
            )
            waveform = codec_input_cpu.to(device=device, dtype=torch.float32)
            try:
                validate_codec_input(waveform, torch)
            except (TypeError, RuntimeError) as exc:
                raise RuntimeError(
                    f"codec input contract failed for track {row['fma_track_id']}: {exc}"
                ) from exc
            codec_input_pcm_hash = eligibility[
                "eligibility_codec_input_pcm_sha256"
            ]

            # Exactly one canonical encode.  Every q reconstruction below
            # reuses this immutable code tensor and the corresponding scale.
            codes, scale = model.encode(waveform)
            try:
                validate_codec_codes(codes, torch)
            except (TypeError, RuntimeError) as exc:
                raise RuntimeError(
                    f"codec code contract failed for track {row['fma_track_id']}: {exc}"
                ) from exc
            reference_length = int(waveform.shape[-1])
            reconstructions = [torch.zeros_like(waveform)]
            for q in range(1, q_model + 1):
                try:
                    reconstruction = model.decode(codes[:, :q, :], scale)
                except Exception as exc:
                    if q == 1:
                        raise RuntimeError(
                            "q=1 cumulative-prefix smoke failed; this codec/wrapper is not accepted for A1"
                        ) from exc
                    raise
                if reconstruction.ndim != 3 or reconstruction.shape[:2] != waveform.shape[:2]:
                    raise RuntimeError(f"invalid q={q} decode shape for track {row['fma_track_id']}")
                if reconstruction.shape[-1] < reference_length:
                    raise RuntimeError(f"q={q} reconstruction is shorter than the aligned input")
                reconstruction = reconstruction[..., :reference_length].float()
                if not bool(torch.isfinite(reconstruction).all().item()):
                    raise RuntimeError(f"q={q} reconstruction contains NaN/Inf")
                reconstructions.append(reconstruction)
                if q == 1:
                    prefix_smoke_passed = True
            stacked = torch.stack(reconstructions, dim=1)
            reference_norm_tensor = mrstft_reference_norms(
                waveform,
                fft_sizes=DEFAULT_FFT_SIZES,
                hop_ratio=0.25,
            )
            reference_norms = [
                float(value)
                for value in reference_norm_tensor[0, 0].detach().cpu().tolist()
            ]
            if any(value <= REFERENCE_NORM_MINIMUM for value in reference_norms):
                raise RuntimeError(
                    f"illegal MR-STFT reference norm for track {row['fma_track_id']}: "
                    f"{reference_norms}"
                )
            result = progressive_prior(
                waveform,
                stacked,
                fft_sizes=DEFAULT_FFT_SIZES,
                hop_ratio=0.25,
                distance_epsilon=1.0e-7,
                marginal_epsilon=1.0e-8,
            )
            distances = [float(value) for value in result.distances[0].detach().cpu().tolist()]
            marginals = [float(value) for value in result.marginals[0].detach().cpu().tolist()]
            if len(marginals) != q_model or not all(math.isfinite(value) for value in distances + marginals):
                raise RuntimeError(f"non-finite or malformed A1 statistics for track {row['fma_track_id']}")
            telescoping_residual = sum(marginals) - (distances[0] - distances[-1])
            tolerance = 1.0e-6 * max(1.0, abs(distances[0] - distances[-1]))
            if abs(telescoping_residual) > tolerance:
                raise RuntimeError(
                    f"marginals do not telescope for track {row['fma_track_id']}: "
                    f"residual={telescoping_residual}, tolerance={tolerance}"
                )
            codes_cpu = codes.detach().cpu().numpy()
            all_marginals.append(marginals)
            per_clip.append(
                {
                    "schema_version": RESULT_SCHEMA,
                    "manifest_index": index,
                    "fma_track_id": int(row["fma_track_id"]),
                    "relative_audio_path": str(row["relative_audio_path"]),
                    "source_audio_sha256": str(row["source_audio_sha256"]),
                    "extracted_pcm_sha256": str(row["extracted_pcm_sha256"]),
                    "codec_sample_rate": int(model.sample_rate),
                    "codec_channels": int(model.channels),
                    "codec_input_frames": reference_length,
                    "codec_input_pcm_sha256": codec_input_pcm_hash,
                    "codec_input_pcm_hash_encoding": "frames_x_channels_float32_little_endian_c_order",
                    "a1_r2_eligibility": eligibility,
                    "num_codebooks": q_model,
                    "codec_code_frames": int(codes.shape[-1]),
                    "canonical_codes_i64le_sha256": codes_i64le_sha256(codes_cpu),
                    "distance_q0_through_qQ": distances,
                    "marginal_delta_q1_through_qQ": marginals,
                    "reference_stft_frobenius_norm_by_fft": {
                        str(fft): value
                        for fft, value in zip(DEFAULT_FFT_SIZES, reference_norms)
                    },
                    "reference_norm_minimum_exclusive": REFERENCE_NORM_MINIMUM,
                    "denominator_floor_activation": 0,
                    "telescoping_residual": telescoping_residual,
                    "telescoping_tolerance": tolerance,
                    "q0_definition": "waveform_zeros_like_aligned_input",
                    "q_ge_1_definition": "decode_first_q_codebooks_from_one_canonical_encoding",
                }
            )
    if not prefix_smoke_passed:
        raise RuntimeError("q=1 prefix smoke was not exercised")

    aggregate = aggregate_marginals(
        all_marginals,
        [int(row["fma_track_id"]) for row in rows],
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    for record, influence in zip(
        per_clip, aggregate["leave_one_out_influence"]["per_clip"]
    ):
        if record["manifest_index"] != influence["manifest_index"]:
            raise RuntimeError("leave-one-out records lost manifest alignment")
        record["leave_one_out_influence"] = influence
    median_tau = aggregate["sensitivity"]["median"]["kendall_tau_b_vs_primary"]
    trimmed_tau = aggregate["sensitivity"]["trimmed_mean_1pct"][
        "kendall_tau_b_vs_primary"
    ]
    science_gates = {
        "bootstrap_tv_ci95_low_gt_0_05": bool(
            aggregate["bootstrap"]["total_variation_from_uniform_ci95_low"]
            > SCIENTIFIC_TV_LOWER_THRESHOLD
        ),
        "split_half_kendall_tau_b_ge_two_thirds": bool(
            aggregate["split_half"]["kendall_tau_b"] is not None
            and aggregate["split_half"]["kendall_tau_b"]
            >= SCIENTIFIC_KENDALL_THRESHOLD
        ),
        "max_leave_one_out_tv_le_0_05": bool(
            aggregate["leave_one_out_influence"]["max_total_variation"]
            <= MAX_LEAVE_ONE_OUT_TV
        ),
        "median_rank_kendall_tau_b_ge_two_thirds": bool(
            median_tau is not None and median_tau >= SCIENTIFIC_KENDALL_THRESHOLD
        ),
        "trimmed_rank_kendall_tau_b_ge_two_thirds": bool(
            trimmed_tau is not None and trimmed_tau >= SCIENTIFIC_KENDALL_THRESHOLD
        ),
    }
    protocol_identity = {
        "identity_schema": ARTIFACT_IDENTITY_SCHEMA,
        "label": PROTOCOL_LABEL,
        "manifest_schema": REQUIRED_MANIFEST_SCHEMA,
        "result_schema": RESULT_SCHEMA,
        "artifact_seal_schema": ARTIFACT_SEAL_SCHEMA,
        "manifest_report": {
            "schema_version": REQUIRED_MANIFEST_SCHEMA,
            "sha256": manifest_report_hash,
        },
        "implementation": code_identity,
    }
    aggregate.update(
        {
            "schema_version": RESULT_SCHEMA,
            "protocol": protocol_identity,
            "manifest_path": str(args.manifest.resolve()),
            "manifest_sha256": manifest_hash,
            "manifest_report_path": str(args.manifest_report.resolve()),
            "manifest_report_sha256": manifest_report_hash,
            "source_root": str(args.source_root.resolve()),
            "codec_checkpoint": str(args.codec_checkpoint.resolve()),
            "codec_checkpoint_payload": str(checkpoint_payload),
            "codec_checkpoint_sha256": observed_checkpoint_hash,
            "codec_load": {**checkpoint_package, **resolved_snapshot},
            "audiocraft_root": str(audiocraft_root),
            "audiocraft_commit": audiocraft_commit,
            "audiocraft_source_identity": audiocraft_identity,
            "device": args.device,
            "runtime": {
                "numpy": np.__version__,
                "soundfile": sf.__version__,
                "libsndfile": sf.__libsndfile_version__,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_device_name": torch.cuda.get_device_name(device),
                "deterministic_algorithms": True,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "allow_tf32": False,
                "codec_parameter_dtype": "float32",
            },
            "metric": {
                "id": "mrstft_spectral_convergence_plus_log_magnitude_l1",
                "fft_sizes": [512, 1024, 2048],
                "hop_ratio": 0.25,
                "window": "Hann, win_length equals FFT size",
                "distance_epsilon": 1.0e-7,
                "reference_norm_guard": "raise if any per-channel FFT norm <= distance_epsilon",
                "denominator_flooring": False,
                "denominator_floor_activation": 0,
                "marginal_epsilon": 1.0e-8,
            },
            "acceptance_gates": {
                "thresholds": {
                    "bootstrap_tv_ci95_low_strictly_greater_than": SCIENTIFIC_TV_LOWER_THRESHOLD,
                    "kendall_tau_b_minimum": SCIENTIFIC_KENDALL_THRESHOLD,
                    "kendall_tau_b_minimum_exact": "2/3",
                    "max_leave_one_out_total_variation": MAX_LEAVE_ONE_OUT_TV,
                },
                "scientific": science_gates,
                "scientific_status": (
                    "pass" if all(science_gates.values()) else "fail_stop_review"
                ),
            },
            "codec_contract": {
                **codec_contract,
                "validated_for_every_clip": True,
                "encode_once_per_clip": True,
                "q0": "waveform silence via zeros_like aligned codec input",
                "q_ge_1": "decode codes[:, :q] from the single canonical encoding",
                "q1_prefix_smoke_passed": prefix_smoke_passed,
                "recursive_reencoding": False,
                "post_reconstruction_loudness_normalization": False,
            },
        }
    )
    return per_clip, aggregate


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument("--manifest-report-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True, help="Same extracted FMA-small root used by the manifest builder")
    parser.add_argument("--audiocraft-root", type=Path, required=True, help="Pinned local AudioCraft checkout")
    parser.add_argument("--codec-checkpoint", type=Path, required=True, help="Local exported compression checkpoint file/directory")
    parser.add_argument("--codec-checkpoint-sha256", required=True, help="Expected SHA-256 of checkpoint file or compression_state_dict.bin")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate paths, manifest hash/schema/count, dependencies, CUDA, and checkpoint hash without running the codec",
    )
    return parser.parse_args(argv)


def _preflight(args: argparse.Namespace) -> Dict[str, Any]:
    from ptc_opd.reproducibility import audiocraft_source_identity

    manifest_hash, rows = manifest_sha256_and_rows(args.manifest.resolve(), args.manifest_sha256)
    if args.manifest.resolve().parent != args.manifest_report.resolve().parent:
        raise ValueError("A1-R2 manifest and report must share the same a1-r2 directory")
    report_hash, report = manifest_report_sha256_and_payload(
        args.manifest_report.resolve(),
        args.manifest_report_sha256,
        manifest_sha256=manifest_hash,
        rows=rows,
    )
    _, _, torch, _, _ = _load_runtime(args.audiocraft_root, args.device)
    payload = _checkpoint_payload_path(args.codec_checkpoint)
    observed = sha256_file(payload)
    if observed != args.codec_checkpoint_sha256.strip().lower():
        raise RuntimeError(
            f"codec checkpoint SHA-256 mismatch: expected {args.codec_checkpoint_sha256}, observed {observed}"
        )
    checkpoint_package = _inspect_local_checkpoint_package(torch, payload)
    if checkpoint_package["load_mode"] == "pretrained_indirection":
        _require_offline_hf_environment()
    if not args.source_root.resolve().is_dir():
        raise FileNotFoundError(f"source root does not exist: {args.source_root.resolve()}")
    audiocraft_root = args.audiocraft_root.resolve()
    audiocraft_commit = _git_commit(audiocraft_root)
    if audiocraft_commit != PINNED_AUDIOCRAFT_BASE_COMMIT:
        raise RuntimeError(
            "A1 requires pinned AudioCraft base commit {}; observed {}".format(
                PINNED_AUDIOCRAFT_BASE_COMMIT, audiocraft_commit
            )
        )
    source_tree_identity = audiocraft_source_identity(audiocraft_root)
    if report["audiocraft"]["source_identity"] != source_tree_identity:
        raise RuntimeError(
            "manifest report AudioCraft source identity differs from the preflight runtime"
        )
    # Preflight checks source containment/existence but deliberately avoids the
    # expensive 512-file hashing/decoding that belongs to the actual assay.
    for row in rows:
        safe_source_path(args.source_root, str(row["relative_audio_path"]))
    return {
        "status": "preflight_passed_not_scientific_result",
        "manifest_sha256": manifest_hash,
        "manifest_report_sha256": report_hash,
        "protocol": PROTOCOL_LABEL,
        "implementation": implementation_identity(),
        "items": len(rows),
        "checkpoint_payload": str(payload),
        "checkpoint_sha256": observed,
        "checkpoint_load_mode": checkpoint_package["load_mode"],
        "audiocraft_base_commit": audiocraft_commit,
        "audiocraft_source_identity": source_tree_identity,
        "device": args.device,
    }


def build_artifact_identity(
    args: argparse.Namespace,
    aggregate: Mapping[str, Any],
    per_clip: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build immutable scientific input identity embedded in the directory seal."""

    expected_items = int(aggregate["n_clips"])
    if expected_items != EXPECTED_ITEMS or len(per_clip) != expected_items:
        raise RuntimeError(
            "artifact identity requires exactly 512 aligned per-clip records"
        )
    if aggregate.get("protocol", {}).get("implementation") != implementation_identity():
        raise RuntimeError("aggregate implementation identity changed before sealing")
    ordered_source_records = [
        {
            "manifest_index": int(record["manifest_index"]),
            "fma_track_id": int(record["fma_track_id"]),
            "relative_audio_path": str(record["relative_audio_path"]),
            "source_audio_sha256": str(record["source_audio_sha256"]),
            "extracted_pcm_sha256": str(record["extracted_pcm_sha256"]),
            "eligibility_codec_input_pcm_sha256": str(
                record["a1_r2_eligibility"][
                    "eligibility_codec_input_pcm_sha256"
                ]
            ),
        }
        for record in per_clip
    ]
    if [record["manifest_index"] for record in ordered_source_records] != list(
        range(expected_items)
    ):
        raise RuntimeError("per-clip records are not in canonical manifest order")
    source_payload = b"".join(_json_line(record) for record in ordered_source_records)
    codec_load = aggregate["codec_load"]
    portable_snapshot_files = [
        {
            "relative_path": str(record["relative_path"]),
            "size_bytes": int(record["size_bytes"]),
            "sha256": str(record["sha256"]),
        }
        for record in codec_load["resolved_snapshot_files"]
    ]
    if portable_snapshot_files != sorted(
        portable_snapshot_files, key=lambda record: record["relative_path"]
    ):
        raise RuntimeError("resolved codec snapshot files are not canonically ordered")
    if len({record["relative_path"] for record in portable_snapshot_files}) != len(
        portable_snapshot_files
    ):
        raise RuntimeError("resolved codec snapshot contains duplicate relative paths")
    if codec_load["load_mode"] == "pretrained_indirection" and not portable_snapshot_files:
        raise RuntimeError("pretrained codec indirection has no sealed snapshot files")
    if codec_load["load_mode"] == "self_contained" and portable_snapshot_files:
        raise RuntimeError("self-contained codec unexpectedly has snapshot files")
    return {
        "manifest": {
            "schema_version": REQUIRED_MANIFEST_SCHEMA,
            "path": str(args.manifest.resolve()),
            "sha256": str(aggregate["manifest_sha256"]),
            "items": int(aggregate["n_clips"]),
        },
        "checkpoint": {
            "path": str(args.codec_checkpoint.resolve()),
            "payload_path": str(aggregate["codec_checkpoint_payload"]),
            "sha256": str(aggregate["codec_checkpoint_sha256"]),
            "load_mode": str(codec_load["load_mode"]),
            "pretrained_model_id": codec_load.get("pretrained_model_id"),
            "resolved_snapshot_revision": codec_load.get("resolved_snapshot_revision"),
        },
        "source_identity": {
            "source_root": str(args.source_root.resolve()),
            "archive_sha1": REQUIRED_ARCHIVE_SHA1,
            "items": len(ordered_source_records),
            "ordered_source_records_sha256": sha256_bytes(source_payload),
            "manifest_report_sha256": str(aggregate["manifest_report_sha256"]),
        },
        "audiocraft": {
            "base_commit": str(aggregate["audiocraft_commit"]),
            "source_identity": dict(aggregate["audiocraft_source_identity"]),
        },
        # Absolute cache paths and symlink layout are intentionally excluded:
        # this is the portable byte identity that survives copying the sealed
        # A1 directory to another machine.
        "codec_load": {
            "load_mode": str(codec_load["load_mode"]),
            "pretrained_model_id": codec_load.get("pretrained_model_id"),
            "resolved_snapshot_revision": codec_load.get(
                "resolved_snapshot_revision"
            ),
            "resolved_snapshot_files": portable_snapshot_files,
        },
        "protocol": dict(aggregate["protocol"]),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    guard_single_process_environment()
    if args.check_only:
        print(json.dumps(_preflight(args), sort_keys=True))
        return 0
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --check-only is used")
    if args.output_dir.name != REQUIRED_OUTPUT_DIR_BASENAME:
        raise ValueError(
            f"formal A1-R2 --output-dir basename must be {REQUIRED_OUTPUT_DIR_BASENAME!r}"
        )
    # This gate runs before model loading or the 512-clip assay.  The final
    # directory is never created here; publication happens only after a sealed
    # staging generation validates successfully.
    output_dir = require_output_dir_absent(args.output_dir)
    per_clip, aggregate = run_codec_assay(args)
    per_clip_payload = deterministic_gzip_jsonl(per_clip)
    summary_payload = summary_csv_bytes(aggregate)
    aggregate["artifact_payload_sha256"] = {
        OUTPUT_NAMES[0]: sha256_bytes(per_clip_payload),
        OUTPUT_NAMES[1]: sha256_bytes(summary_payload),
    }
    prior_payload = _json_line(aggregate)
    payloads = {
        OUTPUT_NAMES[0]: per_clip_payload,
        OUTPUT_NAMES[1]: summary_payload,
        OUTPUT_NAMES[2]: prior_payload,
    }
    identity = build_artifact_identity(args, aggregate, per_clip)
    paths = publish_artifact_directory(output_dir, payloads, identity)
    seal_path = output_dir / ARTIFACT_SEAL_NAME
    print(
        json.dumps(
            {
                "status": "complete",
                "n_clips": aggregate["n_clips"],
                "num_codebooks": aggregate["num_codebooks"],
                "prior": aggregate["prior"],
                "split_half_kendall_tau_b": aggregate["split_half"]["kendall_tau_b"],
                "scientific_status": aggregate["acceptance_gates"]["scientific_status"],
                "max_leave_one_out_tv": aggregate["leave_one_out_influence"][
                    "max_total_variation"
                ],
                "outputs": {
                    **{
                        path.name: {"path": str(path), "sha256": sha256_file(path)}
                        for path in paths.values()
                    },
                    ARTIFACT_SEAL_NAME: {
                        "path": str(seal_path),
                        "sha256": sha256_file(seal_path),
                    },
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileExistsError,
        FileNotFoundError,
        FloatingPointError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
