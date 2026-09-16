#!/usr/bin/env python3
"""Build the LR summary only from sealed runs and metric artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional, Sequence


WORKPACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKPACK_ROOT / "src"))
sys.path.insert(0, str(WORKPACK_ROOT / "scripts"))

from decide_ptc500_stability import run_official_stage1_verifier  # noqa: E402
from ptc_opd.stage1_artifact import Stage1ArtifactError, sha256_file  # noqa: E402
from ptc_opd.stage1_control import (  # noqa: E402
    LR_SUMMARY_BASENAME,
    publish_lr_evaluation_summary,
    verify_lr_evaluation_summary,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-manifest-dir", type=Path, required=True)
    parser.add_argument("--base-generation-dir", type=Path, required=True)
    parser.add_argument("--base-quality-dir", type=Path, required=True)
    parser.add_argument("--base-metric-dir", type=Path, required=True)
    parser.add_argument("--candidate-run-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--candidate-generation-dir", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--candidate-quality-dir", type=Path, action="append", required=True
    )
    parser.add_argument("--candidate-metric-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workpack-root", type=Path, default=WORKPACK_ROOT)
    return parser.parse_args(argv)


def _print(value: Mapping[str, Any], *, stream: Any = sys.stdout) -> None:
    print(
        json.dumps(
            dict(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        ),
        file=stream,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    groups = (
        args.candidate_run_dir,
        args.candidate_generation_dir,
        args.candidate_quality_dir,
        args.candidate_metric_dir,
    )
    try:
        if any(len(group) != 3 for group in groups):
            raise Stage1ArtifactError(
                "each candidate directory option must be supplied exactly three times"
            )
        candidates: list[Dict[str, Any]] = []
        for run_dir, generation_dir, quality_dir, metric_dir in zip(*groups):
            candidates.append(
                {
                    "run_dir": run_dir,
                    "run_verification": run_official_stage1_verifier(
                        args.workpack_root, run_dir
                    ),
                    "generation_dir": generation_dir,
                    "quality_dir": quality_dir,
                    "metric_dir": metric_dir,
                }
            )
        publish_lr_evaluation_summary(
            eval_manifest_dir=args.eval_manifest_dir,
            base_generation_dir=args.base_generation_dir,
            base_quality_dir=args.base_quality_dir,
            base_metric_dir=args.base_metric_dir,
            candidates=candidates,
            output_dir=args.output_dir,
        )
        summary = verify_lr_evaluation_summary(args.output_dir)
        result = {
            "schema_version": "ptc-opd-lr-summary-automatic-producer-v1",
            "status": "verified",
            "numeric_input_from_operator": False,
            "candidate_count": len(summary["candidates"]),
            "summary_sha256": sha256_file(args.output_dir / LR_SUMMARY_BASENAME),
            "artifact_seal_sha256": sha256_file(
                args.output_dir / "artifact_seal.json"
            ),
        }
    except (OSError, Stage1ArtifactError, TypeError, ValueError) as exc:
        _print(
            {
                "schema_version": "ptc-opd-lr-summary-automatic-producer-v1",
                "status": "invalid",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            stream=sys.stderr,
        )
        return 1
    _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
