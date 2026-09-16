#!/usr/bin/env python3
"""Add pinned music-CLAP to a Stage-1 quality artifact and seal four metrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence


WORKPACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKPACK_ROOT / "src"))
sys.path.insert(0, str(WORKPACK_ROOT / "scripts"))

from ptc_opd.stage1_artifact import (  # noqa: E402
    artifact_member,
    canonical_json_bytes,
    sha256_file,
)
from ptc_opd.stage1_generation import load_eval_manifest_artifact  # noqa: E402
from ptc_opd.stage1_metrics import (  # noqa: E402
    FINAL_METRICS,
    METRIC_PROVENANCE_NAME,
    METRIC_PROVENANCE_SCHEMA_VERSION,
    METRIC_SCHEMA_VERSION,
    METRIC_SCORES_NAME,
    METRIC_SEAL_SCHEMA_VERSION,
    SEAL_NAME,
    verify_accepted_evaluator_identity,
    verify_metric_artifact,
    verify_quality_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--generation-dir", type=Path, required=True)
    run.add_argument("--eval-manifest-dir", type=Path, required=True)
    run.add_argument("--quality-dir", type=Path, required=True)
    run.add_argument("--clap-checkpoint", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--batch-size", type=int, default=8)
    verify = sub.add_parser("verify")
    verify.add_argument("--generation-dir", type=Path, required=True)
    verify.add_argument("--eval-manifest-dir", type=Path, required=True)
    verify.add_argument("--quality-dir", type=Path, required=True)
    verify.add_argument("--metric-dir", type=Path, required=True)
    return parser


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(dict(value)))
        stream.flush()
        os.fsync(stream.fileno())


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())


def run(args: argparse.Namespace) -> Path:
    import cfg_eval_common as common
    # Set every offline switch before laion-clap/transformers can be imported.
    common.enforce_offline_environment()
    from eval_cfg_music_clap import MusicClapBackend

    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    quality = verify_quality_artifact(
        args.quality_dir,
        generation_dir=args.generation_dir,
        eval_manifest_dir=args.eval_manifest_dir,
    )
    generation = quality["generation"]
    prompts, _ = load_eval_manifest_artifact(args.eval_manifest_dir)
    prompt_by_id = {row["sample_id"]: row["prompt"] for row in prompts}
    output = args.output_dir.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    generation_root = Path(generation["directory"])
    paths = [generation_root / str(row["path"]) for row in generation["samples"]]
    prompt_values = [prompt_by_id[str(row["sample_id"])] for row in generation["samples"]]
    with common.staged_directory(output) as staging:
        with common.deny_network_connections():
            backend = MusicClapBackend(args)
            clap_identity = common.validate_evaluator_identity(backend.identity, "music_clap")
            verify_accepted_evaluator_identity(clap_identity, "music_clap")
            clap_scores: List[float] = []
            for start in range(0, len(paths), args.batch_size):
                values = backend.score_batch(
                    paths[start : start + args.batch_size],
                    prompt_values[start : start + args.batch_size],
                )
                clap_scores.extend(
                    common.finite_float(value, "music_clap") for value in values
                )
            del backend
        if len(clap_scores) != len(generation["samples"]):
            raise RuntimeError("music-CLAP output count mismatch")
        quality_by_key = {
            (str(row["sample_id"]), int(row["generation_seed"])): row
            for row in quality["rows"]
        }
        provenance = {
            "schema_version": METRIC_PROVENANCE_SCHEMA_VERSION,
            "status": "accepted_stage1_evaluation",
            "metrics": list(FINAL_METRICS),
            "generation_artifact_seal_sha256": generation["artifact_seal_sha256"],
            "quality_artifact_seal_sha256": quality["artifact_seal_sha256"],
            "evaluators": {
                **quality["provenance"]["evaluators"],
                "music_clap": clap_identity,
            },
            "offline_environment": dict(common.OFFLINE_ENVIRONMENT),
            "fad_computed": False,
            "fad_role": "separate_nonselection_pipeline_check",
        }
        provenance_path = staging / METRIC_PROVENANCE_NAME
        _write_json(provenance_path, provenance)
        provenance_hash = sha256_file(provenance_path)
        rows: List[Dict[str, Any]] = []
        for source, clap in zip(generation["samples"], clap_scores):
            key = (str(source["sample_id"]), int(source["generation_seed"]))
            quality_row = quality_by_key[key]
            metrics = dict(quality_row["metrics"])
            metrics["music_clap"] = common.finite_float(clap, "music_clap")
            rows.append(
                {
                    "schema_version": METRIC_SCHEMA_VERSION,
                    "sample_id": source["sample_id"],
                    "generation_seed": source["generation_seed"],
                    "prompt_sha256": source["prompt_sha256"],
                    "condition_id": source["condition_id"],
                    "audio_sha256": source["audio_sha256"],
                    "scientific_config_sha256": source["scientific_config_sha256"],
                    "evaluator_provenance_sha256": provenance_hash,
                    "metrics": metrics,
                }
            )
        scores_path = staging / METRIC_SCORES_NAME
        _write_jsonl(scores_path, rows)
        _write_json(
            staging / SEAL_NAME,
            {
                "schema_version": METRIC_SEAL_SCHEMA_VERSION,
                "status": "complete_stage1_evaluation",
                "record_count": len(rows),
                "generation_artifact_seal_sha256": generation[
                    "artifact_seal_sha256"
                ],
                "quality_artifact_seal_sha256": quality["artifact_seal_sha256"],
                "members": {
                    METRIC_SCORES_NAME: artifact_member(scores_path),
                    METRIC_PROVENANCE_NAME: artifact_member(provenance_path),
                },
            },
        )
        verify_metric_artifact(
            staging,
            generation_dir=generation_root,
            quality_dir=args.quality_dir,
            eval_manifest_dir=args.eval_manifest_dir,
        )
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            output = run(args)
            result = {"status": "complete", "output_dir": str(output)}
        else:
            verified = verify_metric_artifact(
                args.metric_dir,
                generation_dir=args.generation_dir,
                quality_dir=args.quality_dir,
                eval_manifest_dir=args.eval_manifest_dir,
            )
            result = {
                "status": "verified",
                "record_count": len(verified["rows"]),
                "artifact_seal_sha256": verified["artifact_seal_sha256"],
            }
    except (OSError, ValueError, RuntimeError) as exc:
        print("STAGE1 CLAP FAILED: {}".format(exc), file=sys.stderr)
        return 3
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
