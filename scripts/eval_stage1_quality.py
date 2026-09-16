#!/usr/bin/env python3
"""Evaluate Stage-1 generated audio with pinned MuQ and Audiobox backends."""

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
from ptc_opd.stage1_generation import verify_generation_artifact  # noqa: E402
from ptc_opd.stage1_metrics import (  # noqa: E402
    QUALITY_METRICS,
    QUALITY_PROVENANCE_NAME,
    QUALITY_PROVENANCE_SCHEMA_VERSION,
    QUALITY_SCHEMA_VERSION,
    QUALITY_SCORES_NAME,
    QUALITY_SEAL_SCHEMA_VERSION,
    SEAL_NAME,
    verify_accepted_evaluator_identity,
    verify_quality_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--generation-dir", type=Path, required=True)
    run.add_argument("--eval-manifest-dir", type=Path, required=True)
    run.add_argument("--muq-eval-root", type=Path, required=True)
    run.add_argument("--muq-config", type=Path, required=True)
    run.add_argument("--muq-state-dict", type=Path, required=True)
    run.add_argument("--muq-backbone", type=Path, required=True)
    run.add_argument("--audiobox-checkpoint", type=Path, required=True)
    run.add_argument(
        "--environment-report",
        type=Path,
        default=WORKPACK_ROOT / "docs" / "environment_acceptance_20260812.md",
    )
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--batch-size", type=int, default=8)
    verify = sub.add_parser("verify")
    verify.add_argument("--generation-dir", type=Path, required=True)
    verify.add_argument("--eval-manifest-dir", type=Path, required=True)
    verify.add_argument("--quality-dir", type=Path, required=True)
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
    # Set the live offline contract before evaluator modules can import
    # transformers or any checkpoint loader.
    common.enforce_offline_environment()
    from eval_cfg_quality import (
        AudioboxBackend,
        MuQBackend,
        _score_audiobox,
        _score_muq,
    )

    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    generation = verify_generation_artifact(
        args.generation_dir,
        eval_manifest_dir=args.eval_manifest_dir,
        rehash_pcm=False,
    )
    output = args.output_dir.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    generation_root = Path(generation["directory"])
    paths = [generation_root / str(row["path"]) for row in generation["samples"]]
    with common.staged_directory(output) as staging:
        with common.deny_network_connections():
            muq = MuQBackend(args)
            muq_identity = common.validate_evaluator_identity(muq.identity, "muq_eval")
            verify_accepted_evaluator_identity(muq_identity, "muq_eval")
            muq_scores = _score_muq(muq, paths, args.batch_size)
            del muq
            audiobox = AudioboxBackend(args)
            audiobox_identity = common.validate_evaluator_identity(
                audiobox.identity, "audiobox_aesthetics"
            )
            verify_accepted_evaluator_identity(
                audiobox_identity, "audiobox_aesthetics"
            )
            audiobox_scores = _score_audiobox(audiobox, paths, args.batch_size)
            del audiobox
        if not (len(paths) == len(muq_scores) == len(audiobox_scores)):
            raise RuntimeError("quality backend output count mismatch")
        provenance = {
            "schema_version": QUALITY_PROVENANCE_SCHEMA_VERSION,
            "status": "accepted_quality_evaluation",
            "metrics": list(QUALITY_METRICS),
            "generation_artifact_seal_sha256": generation["artifact_seal_sha256"],
            "evaluators": {
                "muq_eval": muq_identity,
                "audiobox_aesthetics": audiobox_identity,
            },
            "offline_environment": dict(common.OFFLINE_ENVIRONMENT),
        }
        provenance_path = staging / QUALITY_PROVENANCE_NAME
        _write_json(provenance_path, provenance)
        provenance_hash = sha256_file(provenance_path)
        rows: List[Dict[str, Any]] = []
        for source, muq_value, aesthetics in zip(
            generation["samples"], muq_scores, audiobox_scores
        ):
            rows.append(
                {
                    "schema_version": QUALITY_SCHEMA_VERSION,
                    "sample_id": source["sample_id"],
                    "generation_seed": source["generation_seed"],
                    "prompt_sha256": source["prompt_sha256"],
                    "condition_id": source["condition_id"],
                    "audio_sha256": source["audio_sha256"],
                    "scientific_config_sha256": source["scientific_config_sha256"],
                    "evaluator_provenance_sha256": provenance_hash,
                    "metrics": {
                        "muq_mi": common.finite_float(muq_value, "muq_mi"),
                        "audiobox_ce": common.finite_float(
                            aesthetics["audiobox_ce"], "audiobox_ce"
                        ),
                        "audiobox_pq": common.finite_float(
                            aesthetics["audiobox_pq"], "audiobox_pq"
                        ),
                    },
                }
            )
        scores_path = staging / QUALITY_SCORES_NAME
        _write_jsonl(scores_path, rows)
        _write_json(
            staging / SEAL_NAME,
            {
                "schema_version": QUALITY_SEAL_SCHEMA_VERSION,
                "status": "complete_quality_evaluation",
                "record_count": len(rows),
                "generation_artifact_seal_sha256": generation[
                    "artifact_seal_sha256"
                ],
                "members": {
                    QUALITY_SCORES_NAME: artifact_member(scores_path),
                    QUALITY_PROVENANCE_NAME: artifact_member(provenance_path),
                },
            },
        )
        verify_quality_artifact(
            staging,
            generation_dir=generation_root,
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
            verified = verify_quality_artifact(
                args.quality_dir,
                generation_dir=args.generation_dir,
                eval_manifest_dir=args.eval_manifest_dir,
            )
            result = {
                "status": "verified",
                "record_count": len(verified["rows"]),
                "artifact_seal_sha256": verified["artifact_seal_sha256"],
            }
    except (OSError, ValueError, RuntimeError) as exc:
        print("STAGE1 QUALITY FAILED: {}".format(exc), file=sys.stderr)
        return 3
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
