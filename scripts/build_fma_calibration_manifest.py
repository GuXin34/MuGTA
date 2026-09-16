#!/usr/bin/env python3
"""Build the frozen A1-R2 512-clip FMA-small calibration manifest.

This is intentionally a standalone, single-process preprocessing command.  It
does not download FMA.  Audio decoding is gated on the optional ``soundfile``
dependency so that importing the deterministic/hash helpers does not require a
working audio stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ARCHIVE_SHA1 = "ade154f733639d52e35e32f5593efe5be76c6d70"
SELECTION_DOMAIN = "ptc-opd-codec-cal-v1"
SEGMENT_DOMAIN = "ptc-opd-codec-segment-v1"
SPLIT_SEED = 2701
SELECTION_COUNT = 512
SEGMENT_SECONDS = 10
AUDIO_EXTENSIONS = frozenset({".aac", ".aif", ".aiff", ".flac", ".m4a", ".mp3", ".ogg", ".wav"})
MANIFEST_SCHEMA = "ptc-opd-fma-calibration-v2"
PROTOCOL_LABEL = "A1-R2"
PINNED_AUDIOCRAFT_BASE_COMMIT = "896ec7c47f5e5d1e5aa1e4b260c4405328bf009d"
EXPECTED_ORIGINAL_CANDIDATES = 7_994
CODEC_SAMPLE_RATE = 32_000
CODEC_CHANNELS = 1
CODEC_INPUT_FRAMES = SEGMENT_SECONDS * CODEC_SAMPLE_RATE
PRE_DOWNMIX_RMS_DENOMINATOR = 1.0e-12
MIN_MONO_RMS = 1.0e-5
MIN_MONO_COMPATIBILITY_RATIO = 1.0e-3

WORKPACK_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKPACK_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _hash_file(path: Path, algorithm: str, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    return _hash_file(path, "sha256")


def sha1_file(path: Path) -> str:
    return _hash_file(path, "sha1")


def canonical_track_id(path: Path) -> int:
    """Return the canonical integer FMA ID encoded in an audio basename."""

    if not path.stem.isdigit():
        raise ValueError("audio basename is not a numeric FMA track ID")
    return int(path.stem, 10)


def domain_hash(domain: str, seed: int, track_id: int) -> str:
    material = f"{domain}|{int(seed)}|{int(track_id)}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def selection_hash(track_id: int, seed: int = SPLIT_SEED) -> str:
    return domain_hash(SELECTION_DOMAIN, seed, track_id)


def segment_hash(track_id: int, seed: int = SPLIT_SEED) -> str:
    return domain_hash(SEGMENT_DOMAIN, seed, track_id)


def deterministic_start_frame(
    track_id: int,
    decoded_frames: int,
    segment_frames: int,
    seed: int = SPLIT_SEED,
) -> Tuple[int, str]:
    """Choose an inclusive valid start with the first SHA-256 uint64.

    The digest is interpreted as an unsigned, big-endian 64-bit integer.  A
    clip with exactly the requested length has one valid start (zero).
    """

    decoded_frames = int(decoded_frames)
    segment_frames = int(segment_frames)
    if segment_frames <= 0:
        raise ValueError("segment_frames must be positive")
    valid_start_count = decoded_frames - segment_frames + 1
    if valid_start_count <= 0:
        raise ValueError("decoded audio is shorter than the requested segment")
    digest_hex = segment_hash(track_id, seed)
    value = int.from_bytes(bytes.fromhex(digest_hex)[:8], byteorder="big", signed=False)
    return value % valid_start_count, digest_hex


def pcm_f32le_sha256(samples: Any) -> str:
    """Hash frame-major/channel-interleaved little-endian float32 PCM bytes."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - torch environments include NumPy.
        raise RuntimeError("NumPy is required to hash decoded PCM") from exc
    array = np.asarray(samples)
    if array.ndim != 2:
        raise ValueError("PCM must have shape [frames, channels]")
    canonical = np.ascontiguousarray(array, dtype=np.dtype("<f4"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def choose_lowest_hashes(
    eligible: Sequence[Dict[str, Any]], count: int = SELECTION_COUNT
) -> List[Dict[str, Any]]:
    if count <= 0:
        raise ValueError("count must be positive")
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)} eligible tracks; need {count}")
    ranked = sorted(
        eligible,
        key=lambda row: (selection_hash(int(row["fma_track_id"])), int(row["fma_track_id"])),
    )
    return [dict(row) for row in ranked[:count]]


def discover_audio(root: Path) -> List[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def _audio_dependencies() -> Tuple[Any, Any]:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "FMA decoding requires NumPy and soundfile. Install the calibration "
            "extra (for example: pip install 'numpy<2' 'soundfile>=0.12,<0.14') "
            "and confirm that libsndfile can decode MP3 on this machine."
        ) from exc
    if "MP3" not in sf.available_formats():
        raise RuntimeError(
            "the active libsndfile build does not advertise MP3 decoding, which "
            "is required for the official FMA-small archive"
        )
    return np, sf


def _safe_reason(exc: BaseException) -> str:
    message = " ".join(str(exc).replace("\n", " ").split())
    return f"{type(exc).__name__}: {message}"[:500]


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


def _audio_conversion_dependency(audiocraft_root: Path) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load the exact vendored AudioCraft conversion used by the estimator."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for A1-R2 input eligibility") from exc
    root = audiocraft_root.resolve()
    if not (root / "audiocraft").is_dir():
        raise ValueError(
            f"--audiocraft-root does not contain the audiocraft package: {root}"
        )
    commit = _git_commit(root)
    if commit != PINNED_AUDIOCRAFT_BASE_COMMIT:
        raise RuntimeError(
            "A1-R2 requires pinned AudioCraft base commit {}; observed {}".format(
                PINNED_AUDIOCRAFT_BASE_COMMIT, commit
            )
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from audiocraft.data.audio_utils import convert_audio
    import audiocraft
    from ptc_opd.reproducibility import audiocraft_source_identity

    imported = Path(audiocraft.__file__).resolve()
    try:
        imported.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"imported AudioCraft from the wrong tree: {imported}") from exc
    identity = {
        "base_commit": commit,
        "source_identity": audiocraft_source_identity(root),
        "imported_package": str(imported),
        "convert_audio_qualname": "audiocraft.data.audio_utils.convert_audio",
    }
    return torch, convert_audio, identity


def _rms_float64(np: Any, samples: Any) -> float:
    values = np.asarray(samples, dtype=np.float64)
    return math.sqrt(float(np.mean(np.square(values, dtype=np.float64), dtype=np.float64)))


def _lr_pearson(np: Any, pcm: Any) -> Optional[float]:
    """Return stereo L/R Pearson correlation for diagnostics, never filtering."""

    if pcm.ndim != 2 or pcm.shape[1] != 2:
        return None
    left = np.asarray(pcm[:, 0], dtype=np.float64)
    right = np.asarray(pcm[:, 1], dtype=np.float64)
    left = left - left.mean(dtype=np.float64)
    right = right - right.mean(dtype=np.float64)
    denominator = math.sqrt(
        float(np.dot(left, left)) * float(np.dot(right, right))
    )
    if denominator == 0.0:
        return None
    return float(max(-1.0, min(1.0, float(np.dot(left, right)) / denominator)))


def analyze_input_eligibility(
    pcm: Any,
    sample_rate: int,
    *,
    np: Any,
    torch_module: Any,
    convert_audio: Any,
) -> Dict[str, Any]:
    """Measure the frozen pre-codec A1-R2 eligibility rule on one segment."""

    values = np.asarray(pcm)
    if values.ndim != 2 or min(values.shape) <= 0:
        raise ValueError("deterministic PCM must have shape [frames,channels]")
    pcm_finite = bool(np.isfinite(values).all())
    if not pcm_finite:
        return {
            "all_pcm_samples_finite": False,
            "pre_downmix_rms": None,
            "mono_rms": None,
            "mono_compatibility_ratio": None,
            "lr_pearson_correlation": None,
            "eligibility_codec_input_pcm_sha256": None,
            "eligible": False,
            "failed_conditions": ["non_finite_pcm"],
        }
    canonical = np.ascontiguousarray(values, dtype=np.float32)
    pre_downmix_rms = _rms_float64(np, canonical)
    waveform = torch_module.from_numpy(canonical.T.copy()).unsqueeze(0)
    with torch_module.inference_mode():
        mono = convert_audio(
            waveform,
            from_rate=int(sample_rate),
            to_rate=CODEC_SAMPLE_RATE,
            to_channels=CODEC_CHANNELS,
        ).contiguous().to(device="cpu", dtype=torch_module.float32)
    expected_shape = (1, CODEC_CHANNELS, CODEC_INPUT_FRAMES)
    if tuple(mono.shape) != expected_shape:
        raise RuntimeError(
            f"exact AudioCraft conversion must produce {expected_shape}, got {tuple(mono.shape)}"
        )
    if not bool(torch_module.isfinite(mono).all().item()):
        raise FloatingPointError("exact 32 kHz mono codec input contains NaN/Inf")
    mono_frame_major = mono[0].transpose(0, 1).detach().cpu().numpy()
    mono_rms = _rms_float64(np, mono_frame_major)
    ratio = mono_rms / max(pre_downmix_rms, PRE_DOWNMIX_RMS_DENOMINATOR)
    failed = []
    if mono_rms < MIN_MONO_RMS:
        failed.append("mono_rms_below_1e-5")
    if ratio < MIN_MONO_COMPATIBILITY_RATIO:
        failed.append("mono_compatibility_ratio_below_1e-3")
    return {
        "all_pcm_samples_finite": True,
        "pre_downmix_rms": pre_downmix_rms,
        "mono_rms": mono_rms,
        "mono_compatibility_ratio": ratio,
        "lr_pearson_correlation": _lr_pearson(np, canonical),
        "eligibility_codec_input_pcm_sha256": pcm_f32le_sha256(mono_frame_major),
        "eligible": not failed,
        "failed_conditions": failed,
    }


def _scan_tracks(
    root: Path,
    paths: Sequence[Path],
    *,
    torch_module: Any,
    convert_audio: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """Analyze the deterministic segment of every structurally valid track.

    Hash ranking happens only after this full pass.  Consequently a rejected
    input cannot be replaced by an ad-hoc track-ID rule or a codec outcome.
    """

    np, sf = _audio_dependencies()
    by_id: Dict[int, List[Path]] = {}
    rejected: List[Dict[str, Any]] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        try:
            track_id = canonical_track_id(path)
        except ValueError as exc:
            rejected.append({"relative_audio_path": relative, "reason": str(exc)})
            continue
        by_id.setdefault(track_id, []).append(path)

    eligible: List[Dict[str, Any]] = []
    eligibility_audit: List[Dict[str, Any]] = []
    original_candidates = 0
    for track_id in sorted(by_id):
        candidates = by_id[track_id]
        if len(candidates) != 1:
            for path in candidates:
                rejected.append(
                    {
                        "fma_track_id": track_id,
                        "relative_audio_path": path.relative_to(root).as_posix(),
                        "reason": "duplicate numeric FMA track ID",
                    }
                )
            continue
        path = candidates[0]
        relative = path.relative_to(root).as_posix()
        try:
            with sf.SoundFile(str(path), mode="r") as handle:
                sample_rate = int(handle.samplerate)
                decoded_frames = int(handle.frames)
                channels = int(handle.channels)
                if sample_rate <= 0 or decoded_frames <= 0 or channels <= 0:
                    raise ValueError("decoder returned invalid sample rate or channel metadata")
                segment_frames = int(SEGMENT_SECONDS * sample_rate)
                if decoded_frames < segment_frames:
                    raise ValueError(
                        f"decoded duration is shorter than {SEGMENT_SECONDS} seconds "
                        f"({decoded_frames} < {segment_frames} frames)"
                    )
                start_frame, offset_digest = deterministic_start_frame(
                    track_id, decoded_frames, segment_frames
                )
                handle.seek(start_frame)
                pcm = handle.read(
                    frames=segment_frames,
                    dtype="float32",
                    always_2d=True,
                    fill_value=None,
                )
            if pcm.shape != (segment_frames, channels):
                raise ValueError(f"short deterministic segment: decoded shape {tuple(pcm.shape)}")
            if decoded_frames < segment_frames:
                raise ValueError(
                    f"decoded duration is shorter than {SEGMENT_SECONDS} seconds "
                    f"({decoded_frames} < {segment_frames} frames)"
                )
            source_sha256 = sha256_file(path)
            original_candidates += 1
            analysis = analyze_input_eligibility(
                pcm,
                sample_rate,
                np=np,
                torch_module=torch_module,
                convert_audio=convert_audio,
            )
        except Exception as exc:
            rejected.append(
                {
                    "fma_track_id": track_id,
                    "relative_audio_path": relative,
                    "reason": _safe_reason(exc),
                }
            )
            continue
        metadata = {
            "fma_track_id": track_id,
            "relative_audio_path": relative,
            "source_audio_sha256": source_sha256,
            "decoded_duration_frames": decoded_frames,
            "decoded_sample_rate": sample_rate,
            "decoded_channels": channels,
            "segment_start_frame": start_frame,
            "segment_num_frames": segment_frames,
            "segment_offset_sha256": offset_digest,
            "extracted_pcm_sha256": pcm_f32le_sha256(pcm),
            "pcm_hash_encoding": "frames_x_channels_float32_little_endian_c_order",
            **analysis,
        }
        eligibility_audit.append(dict(metadata))
        if analysis["eligible"]:
            eligible.append(metadata)
        else:
            rejected.append(
                {
                    "fma_track_id": track_id,
                    "relative_audio_path": relative,
                    "reason": "A1-R2 input eligibility failed: "
                    + ",".join(analysis["failed_conditions"]),
                    "eligibility": analysis,
                }
            )
    rejected.sort(key=lambda row: (str(row.get("relative_audio_path", "")), str(row.get("reason", ""))))
    return eligible, rejected, eligibility_audit, original_candidates


def _materialize_selected(
    root: Path,
    selected: Sequence[Dict[str, Any]],
    *,
    torch_module: Any,
    convert_audio: Any,
) -> List[Dict[str, Any]]:
    np, sf = _audio_dependencies()
    records: List[Dict[str, Any]] = []
    for selection_rank, metadata in enumerate(selected):
        path = root / str(metadata["relative_audio_path"])
        sample_rate = int(metadata["decoded_sample_rate"])
        segment_frames = int(SEGMENT_SECONDS * sample_rate)
        start_frame = int(metadata["segment_start_frame"])
        try:
            with sf.SoundFile(str(path), mode="r") as handle:
                if int(handle.samplerate) != sample_rate or int(handle.channels) != int(metadata["decoded_channels"]):
                    raise ValueError("decoder metadata changed between eligibility scan and extraction")
                handle.seek(start_frame)
                pcm = handle.read(
                    frames=segment_frames,
                    dtype="float32",
                    always_2d=True,
                    fill_value=None,
                )
            if pcm.shape != (segment_frames, int(metadata["decoded_channels"])):
                raise ValueError(f"short deterministic segment: decoded shape {tuple(pcm.shape)}")
            analysis = analyze_input_eligibility(
                pcm,
                sample_rate,
                np=np,
                torch_module=torch_module,
                convert_audio=convert_audio,
            )
            if not analysis["eligible"]:
                raise ValueError("selected segment no longer passes A1-R2 eligibility")
            if pcm_f32le_sha256(pcm) != metadata["extracted_pcm_sha256"]:
                raise ValueError("deterministic segment bytes changed after eligibility scan")
            for key in (
                "all_pcm_samples_finite",
                "pre_downmix_rms",
                "mono_rms",
                "mono_compatibility_ratio",
                "lr_pearson_correlation",
                "eligibility_codec_input_pcm_sha256",
                "eligible",
                "failed_conditions",
            ):
                if analysis[key] != metadata[key]:
                    raise ValueError(f"eligibility metric {key} changed during materialization")
        except Exception as exc:
            raise RuntimeError(
                f"selected track {metadata['fma_track_id']} failed deterministic extraction: "
                f"{_safe_reason(exc)}"
            ) from exc
        record = dict(metadata)
        record.update(
            {
                "schema_version": MANIFEST_SCHEMA,
                "protocol_label": PROTOCOL_LABEL,
                "selection_rank": selection_rank,
                "selection_rank_sha256": selection_hash(int(metadata["fma_track_id"])),
                "eligibility_thresholds": {
                    "pre_downmix_rms_denominator": PRE_DOWNMIX_RMS_DENOMINATOR,
                    "minimum_mono_rms": MIN_MONO_RMS,
                    "minimum_mono_compatibility_ratio": MIN_MONO_COMPATIBILITY_RATIO,
                },
            }
        )
        records.append(record)
    return records


def build_manifest(
    root: Path,
    audiocraft_root: Path,
    selection_count: int = SELECTION_COUNT,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    np, sf = _audio_dependencies()
    torch_module, convert_audio, audiocraft_identity = _audio_conversion_dependency(
        audiocraft_root
    )
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"FMA root is not a directory: {root}")
    paths = discover_audio(root)
    if not paths:
        raise ValueError(f"no supported audio files found under {root}")
    eligible, rejected, eligibility_audit, original_candidates = _scan_tracks(
        root,
        paths,
        torch_module=torch_module,
        convert_audio=convert_audio,
    )
    selected = choose_lowest_hashes(eligible, selection_count)
    records = _materialize_selected(
        root,
        selected,
        torch_module=torch_module,
        convert_audio=convert_audio,
    )
    report: Dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "protocol_label": PROTOCOL_LABEL,
        "fma_root": str(root),
        "discovered_audio_files": len(paths),
        "original_candidate_tracks": original_candidates,
        "expected_original_candidate_tracks": EXPECTED_ORIGINAL_CANDIDATES,
        "input_eligible_tracks": len(eligible),
        "rejected_tracks": len(rejected),
        "selected_tracks": len(records),
        "selection_count_contract": SELECTION_COUNT,
        "selection_count_used": int(selection_count),
        "split_seed": SPLIT_SEED,
        "selection_domain": SELECTION_DOMAIN,
        "segment_domain": SEGMENT_DOMAIN,
        "segment_seconds": SEGMENT_SECONDS,
        "eligibility": {
            "stage": "pre_codec_pre_marginal",
            "all_pcm_samples_finite_required": True,
            "pre_downmix_rms_denominator": PRE_DOWNMIX_RMS_DENOMINATOR,
            "minimum_mono_rms": MIN_MONO_RMS,
            "minimum_mono_compatibility_ratio": MIN_MONO_COMPATIBILITY_RATIO,
            "codec_sample_rate": CODEC_SAMPLE_RATE,
            "codec_channels": CODEC_CHANNELS,
            "codec_input_frames": CODEC_INPUT_FRAMES,
            "conversion": "audiocraft.data.audio_utils.convert_audio",
            "lr_pearson_is_diagnostic_only": True,
            "track_id_filtering": False,
            "audited_tracks": len(eligibility_audit),
        },
        "audiocraft": audiocraft_identity,
        "implementation": {
            "builder_path": str(Path(__file__).resolve()),
            "builder_sha256": sha256_file(Path(__file__).resolve()),
        },
        "decoder": {
            "numpy": np.__version__,
            "soundfile": sf.__version__,
            "libsndfile": sf.__libsndfile_version__,
            "pcm_hash_encoding": "frames_x_channels_float32_little_endian_c_order",
        },
        "rejections": rejected,
        "eligibility_audit": eligibility_audit,
    }
    return records, report


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _manifest_bytes(records: Iterable[Dict[str, Any]]) -> bytes:
    return b"".join(_json_bytes(record) for record in records)


def _refuse_existing(outputs: Sequence[Path]) -> None:
    if len({str(path.resolve()) for path in outputs}) != len(outputs):
        raise ValueError("output manifest and report must be different paths")
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing output(s): " + ", ".join(str(path) for path in existing)
        )


def _atomic_publish(payloads: Mapping[Path, bytes]) -> None:
    """Write every temp first, then publish the complete output generation."""

    temporaries: Dict[Path, Path] = {}
    try:
        for path, payload in payloads.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
            if temporary.exists():
                raise FileExistsError(f"stale temporary output exists: {temporary}")
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporaries[path] = temporary
        for path in sorted(payloads, key=lambda value: value.name):
            os.replace(str(temporaries[path]), str(path))
            temporaries.pop(path, None)
    finally:
        for temporary in temporaries.values():
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fma-root", type=Path, help="Root of the extracted fma_small archive")
    parser.add_argument("--archive", type=Path, help="Original fma_small.zip; required for publication-eligible construction and SHA-1 gated")
    parser.add_argument(
        "--audiocraft-root",
        type=Path,
        help="Pinned vendored AudioCraft checkout providing the exact convert_audio implementation",
    )
    parser.add_argument(
        "--allow-unverified-archive-for-test",
        action="store_true",
        help="TEST-ONLY waiver when the original archive is unavailable; makes the report publication-ineligible",
    )
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--output-report", type=Path)
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Only verify that NumPy/soundfile import and that libsndfile reports a version",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.check_deps:
        np, sf = _audio_dependencies()
        print(json.dumps({"numpy": np.__version__, "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__}, sort_keys=True))
        return 0

    missing = [
        flag
        for flag, value in (
            ("--fma-root", args.fma_root),
            ("--audiocraft-root", args.audiocraft_root),
            ("--output-manifest", args.output_manifest),
            ("--output-report", args.output_report),
        )
        if value is None
    ]
    if missing:
        raise ValueError("missing required arguments for manifest construction: " + ", ".join(missing))

    outputs = [args.output_manifest.resolve(), args.output_report.resolve()]
    if args.output_manifest.name != "codec_calibration.train.jsonl":
        raise ValueError("formal A1-R2 manifest basename must be 'codec_calibration.train.jsonl'")
    if args.output_report.name != "codec_calibration.train.report.json":
        raise ValueError(
            "formal A1-R2 report basename must be 'codec_calibration.train.report.json'"
        )
    if outputs[0].parent != outputs[1].parent or outputs[0].parent.name != "a1-r2":
        raise ValueError("formal A1-R2 manifest/report must share a directory named 'a1-r2'")
    _refuse_existing(outputs)
    archive_report: Dict[str, Any]
    if args.archive is not None:
        archive = args.archive.resolve()
        if not archive.is_file():
            raise FileNotFoundError(f"archive does not exist: {archive}")
        observed_sha1 = sha1_file(archive)
        if observed_sha1 != ARCHIVE_SHA1:
            raise RuntimeError(
                f"FMA-small archive SHA-1 mismatch: expected {ARCHIVE_SHA1}, observed {observed_sha1}"
            )
        archive_report = {
            "archive_path": str(archive),
            "archive_sha1": observed_sha1,
            "archive_sha1_verified": True,
            "publication_eligible": True,
        }
    else:
        if not args.allow_unverified_archive_for_test:
            raise ValueError(
                "--archive is required by the frozen A1 protocol; the only bypass is the "
                "explicit --allow-unverified-archive-for-test waiver"
            )
        archive_report = {
            "archive_path": None,
            "archive_sha1": None,
            "archive_sha1_verified": False,
            "publication_eligible": False,
            "warning": "TEST-ONLY WAIVER: archive omitted; this manifest is forbidden for paper results",
        }

    records, report = build_manifest(
        args.fma_root,
        args.audiocraft_root,
        SELECTION_COUNT,
    )
    if report["original_candidate_tracks"] != EXPECTED_ORIGINAL_CANDIDATES:
        raise RuntimeError(
            "formal A1-R2 requires exactly {} originally eligible FMA tracks; observed {}".format(
                EXPECTED_ORIGINAL_CANDIDATES, report["original_candidate_tracks"]
            )
        )
    if report["eligibility"]["audited_tracks"] != EXPECTED_ORIGINAL_CANDIDATES:
        raise RuntimeError(
            "A1-R2 eligibility was not completed for every original candidate: "
            "audited={}, expected={}".format(
                report["eligibility"]["audited_tracks"], EXPECTED_ORIGINAL_CANDIDATES
            )
        )
    for record in records:
        record.update(
            {
                "archive_sha1": archive_report["archive_sha1"],
                "archive_sha1_verified": archive_report["archive_sha1_verified"],
                "publication_eligible": archive_report["publication_eligible"],
            }
        )
    manifest_payload = _manifest_bytes(records)
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    report.update(archive_report)
    report["manifest_path"] = str(args.output_manifest.resolve())
    report["manifest_sha256"] = manifest_sha256
    report_payload = _json_bytes(report)

    # Both paths were checked before preprocessing, and both temp files are
    # fully synced before the short publish window begins.
    _atomic_publish(
        {
            args.output_manifest.resolve(): manifest_payload,
            args.output_report.resolve(): report_payload,
        }
    )
    print(
        json.dumps(
            {
                "manifest": str(args.output_manifest.resolve()),
                "manifest_sha256": manifest_sha256,
                "report": str(args.output_report.resolve()),
                "report_sha256": hashlib.sha256(report_payload).hexdigest(),
                "selected": len(records),
                "rejected": report["rejected_tracks"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
