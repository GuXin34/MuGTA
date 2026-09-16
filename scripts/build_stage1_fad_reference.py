#!/usr/bin/env python3
"""Build/verify the sealed 256-clip Stage-1 FAD reference from A1-R2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence


WORKPACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKPACK_ROOT / "src"))
sys.path.insert(0, str(WORKPACK_ROOT / "scripts"))

from ptc_opd.stage1_artifact import (  # noqa: E402
    Stage1ArtifactError,
    artifact_member,
    canonical_json_bytes,
    sha256_file,
    sha256_tree,
)
from ptc_opd.stage1_diversity_fad import (  # noqa: E402
    REFERENCE_COUNT,
    REFERENCE_MANIFEST,
    REFERENCE_RECORD_SCHEMA,
    REFERENCE_REPORT,
    REFERENCE_SCHEMA,
    REFERENCE_SEAL_SCHEMA,
    REFERENCE_SELECTION_DOMAIN,
    REFERENCE_SELECTION_SEED,
    REFERENCE_SECONDS,
    SEAL_NAME,
    reference_selection_hash,
    select_reference_rows,
    verify_reference_artifact,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--a1-manifest", type=Path, required=True)
    build.add_argument("--a1-manifest-sha256", required=True)
    build.add_argument("--a1-report", type=Path, required=True)
    build.add_argument("--a1-report-sha256", required=True)
    build.add_argument("--fma-root", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--a1-manifest", type=Path, required=True)
    verify.add_argument("--a1-report", type=Path, required=True)
    verify.add_argument("--artifact-dir", type=Path, required=True)
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(dict(value)))
        stream.flush()
        os.fsync(stream.fileno())


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())


def _source_path(root: Path, relative: str) -> Path:
    if root.expanduser().is_symlink():
        raise Stage1ArtifactError("FMA root may not be a symlink")
    root = root.resolve(strict=True)
    supplied_candidate = root / relative
    if supplied_candidate.is_symlink():
        raise Stage1ArtifactError("A1-R2 source audio may not be a symlink")
    candidate = supplied_candidate.resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise Stage1ArtifactError("A1-R2 audio path escapes FMA root") from exc
    if candidate.is_symlink() or not candidate.is_file():
        raise Stage1ArtifactError("A1-R2 source audio must be a regular file")
    return candidate


def _canonical_existing(path: Path, label: str, *, kind: str) -> Path:
    supplied = path.expanduser().absolute()
    if supplied.is_symlink():
        raise Stage1ArtifactError("{} root may not be a symlink".format(label))
    resolved = supplied.resolve(strict=True)
    if kind == "tree" and not resolved.is_dir():
        raise Stage1ArtifactError("{} must be a directory".format(label))
    if kind == "file" and (not resolved.is_file() or resolved.is_symlink()):
        raise Stage1ArtifactError("{} must be a regular file".format(label))
    return resolved


def _canonical_new_output(output: Path, protected: Sequence[Path]) -> Path:
    supplied = output.expanduser().absolute()
    current = Path(supplied.anchor)
    for part in supplied.parts[1:-1]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise Stage1ArtifactError("FAD reference output parent contains a symlink")
        if not current.exists():
            break
    supplied.parent.mkdir(parents=True, exist_ok=True)
    current = Path(supplied.anchor)
    for part in supplied.parts[1:-1]:
        current = current / part
        if current.is_symlink():
            raise Stage1ArtifactError("FAD reference output parent contains a symlink")
    resolved = supplied.parent.resolve(strict=True) / supplied.name
    if resolved.exists() or resolved.is_symlink():
        raise FileExistsError("refusing to overwrite FAD reference: {}".format(resolved))
    for root in protected:
        protected_root = root.resolve(strict=True)
        try:
            resolved.relative_to(protected_root)
        except ValueError:
            pass
        else:
            raise Stage1ArtifactError("FAD reference output overlaps a protected input")
        try:
            protected_root.relative_to(resolved)
        except ValueError:
            pass
        else:
            raise Stage1ArtifactError("FAD reference output contains a protected input")
    return resolved


def _verify_full_a1_contract(
    manifest: Path,
    report: Path,
    *,
    manifest_sha256: str,
    report_sha256: str,
) -> tuple[str, str, list[Dict[str, Any]]]:
    import estimate_codec_prior as a1

    if manifest.expanduser().is_symlink() or report.expanduser().is_symlink():
        raise Stage1ArtifactError("A1-R2 manifest/report may not be symlinks")
    manifest = manifest.resolve(strict=True)
    report = report.resolve(strict=True)
    if manifest.parent != report.parent:
        raise Stage1ArtifactError("A1-R2 manifest/report must share one directory")
    observed_manifest, rows = a1.manifest_sha256_and_rows(
        manifest, manifest_sha256
    )
    observed_report, _ = a1.manifest_report_sha256_and_payload(
        report,
        report_sha256,
        manifest_sha256=observed_manifest,
        rows=rows,
    )
    return observed_manifest, observed_report, rows


def build(args: argparse.Namespace) -> Path:
    import numpy as np
    import soundfile as sf
    from build_fma_calibration_manifest import pcm_f32le_sha256

    manifest_path = _canonical_existing(args.a1_manifest, "A1-R2 manifest", kind="file")
    report_path = _canonical_existing(args.a1_report, "A1-R2 report", kind="file")
    fma_root = _canonical_existing(args.fma_root, "FMA", kind="tree")
    manifest_hash, report_hash, source_rows = _verify_full_a1_contract(
        manifest_path,
        report_path,
        manifest_sha256=args.a1_manifest_sha256,
        report_sha256=args.a1_report_sha256,
    )
    selected = select_reference_rows(source_rows)
    output = _canonical_new_output(
        args.output_dir, (manifest_path, report_path, fma_root)
    )
    staging = Path(
        tempfile.mkdtemp(prefix=".{}.partial.".format(output.name), dir=str(output.parent))
    )
    try:
        audio_root = staging / "audio"
        audio_root.mkdir()
        rows: list[Dict[str, Any]] = []
        for selection_rank, source_row in enumerate(selected):
            track_id = int(source_row["fma_track_id"])
            source = _source_path(fma_root, str(source_row["relative_audio_path"]))
            if sha256_file(source) != source_row["source_audio_sha256"]:
                raise Stage1ArtifactError("FMA source SHA-256 differs")
            sample_rate = int(source_row["decoded_sample_rate"])
            channels = int(source_row["decoded_channels"])
            frames = int(source_row["segment_num_frames"])
            with sf.SoundFile(str(source), mode="r") as handle:
                if (
                    int(handle.samplerate) != sample_rate
                    or int(handle.channels) != channels
                    or int(handle.frames) != int(source_row["decoded_duration_frames"])
                ):
                    raise Stage1ArtifactError("FMA decoder metadata differs from A1-R2")
                handle.seek(int(source_row["segment_start_frame"]))
                pcm = handle.read(frames=frames, dtype="float32", always_2d=True)
            if tuple(pcm.shape) != (frames, channels) or not bool(np.isfinite(pcm).all()):
                raise Stage1ArtifactError("A1-R2 segment decode is short/non-finite")
            if pcm_f32le_sha256(pcm) != source_row["extracted_pcm_sha256"]:
                raise Stage1ArtifactError("A1-R2 deterministic PCM SHA-256 differs")
            relative = Path("audio") / "{:06d}.wav".format(track_id)
            target = staging / relative
            sf.write(str(target), pcm, sample_rate, format="WAV", subtype="FLOAT")
            roundtrip, roundtrip_rate = sf.read(
                str(target), dtype="float32", always_2d=True
            )
            info = sf.info(str(target))
            if (
                roundtrip_rate != sample_rate
                or tuple(roundtrip.shape) != tuple(pcm.shape)
                or not bool(np.array_equal(roundtrip, pcm))
                or info.format != "WAV"
                or info.subtype != "FLOAT"
            ):
                raise Stage1ArtifactError("reference WAV is not an exact float32 PCM wrapper")
            rows.append(
                {
                    "schema_version": REFERENCE_RECORD_SCHEMA,
                    "selection_rank": selection_rank,
                    "selection_hash": reference_selection_hash(track_id),
                    "fma_track_id": track_id,
                    "relative_audio_path": source_row["relative_audio_path"],
                    "source_audio_sha256": source_row["source_audio_sha256"],
                    "source_pcm_sha256": source_row["extracted_pcm_sha256"],
                    "segment_start_frame": source_row["segment_start_frame"],
                    "segment_num_frames": frames,
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "duration_seconds": REFERENCE_SECONDS,
                    "path": relative.as_posix(),
                    "audio_sha256": sha256_file(target),
                    "audio_frames": int(info.frames),
                    "audio_channels": int(info.channels),
                    "audio_sample_rate": int(info.samplerate),
                    "audio_subtype": str(info.subtype),
                }
            )
        _write_jsonl(staging / REFERENCE_MANIFEST, rows)
        audio_tree_hash = sha256_tree(audio_root)
        report = {
            "schema_version": REFERENCE_SCHEMA,
            "status": "complete_fad_reference",
            "source": {
                "a1_r2_manifest_basename": manifest_path.name,
                "a1_r2_manifest_sha256": manifest_hash,
                "a1_r2_report_basename": report_path.name,
                "a1_r2_report_sha256": report_hash,
                "candidate_count": 512,
            },
            "selection": {
                "algorithm": "lowest_sha256_then_track_id",
                "domain": REFERENCE_SELECTION_DOMAIN,
                "seed": REFERENCE_SELECTION_SEED,
                "count": REFERENCE_COUNT,
                "source_population": "A1-R2 512 eligible 10-second segments",
            },
            "record_count": REFERENCE_COUNT,
            "duration_seconds": REFERENCE_SECONDS,
            "manifest": artifact_member(staging / REFERENCE_MANIFEST),
            "audio_tree_sha256": audio_tree_hash,
        }
        _write_json(staging / REFERENCE_REPORT, report)
        _write_json(
            staging / SEAL_NAME,
            {
                "schema_version": REFERENCE_SEAL_SCHEMA,
                "status": "complete_fad_reference",
                "record_count": REFERENCE_COUNT,
                "members": {
                    REFERENCE_MANIFEST: artifact_member(staging / REFERENCE_MANIFEST),
                    REFERENCE_REPORT: artifact_member(staging / REFERENCE_REPORT),
                },
                "audio_tree_sha256": audio_tree_hash,
            },
        )
        verify_reference_artifact(
            staging, a1_manifest=manifest_path, a1_report=report_path
        )
        os.replace(str(staging), str(output))
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "build":
            output = build(args)
        else:
            _verify_full_a1_contract(
                args.a1_manifest,
                args.a1_report,
                manifest_sha256=sha256_file(args.a1_manifest),
                report_sha256=sha256_file(args.a1_report),
            )
            output = args.artifact_dir
        verified = verify_reference_artifact(
            output, a1_manifest=args.a1_manifest, a1_report=args.a1_report
        )
    except (OSError, RuntimeError, ValueError, Stage1ArtifactError) as exc:
        print(
            json.dumps(
                {"status": "invalid", "error_type": type(exc).__name__, "error": str(exc)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 3
    print(
        json.dumps(
            {
                "status": "verified",
                "artifact_dir": str(Path(output).absolute()),
                "record_count": len(verified["rows"]),
                "artifact_seal_sha256": verified["artifact_seal_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
