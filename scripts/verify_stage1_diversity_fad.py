#!/usr/bin/env python3
"""Verify a closed Stage-1 MERT-diversity/FAD pipeline-check artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence


WORKPACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKPACK_ROOT / "src"))

from ptc_opd.stage1_artifact import Stage1ArtifactError  # noqa: E402
from ptc_opd.stage1_diversity_fad import verify_diversity_fad_artifact  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--artifact-dir", type=Path, required=True)
    result.add_argument("--generation-dir", type=Path, required=True)
    result.add_argument("--eval-manifest-dir", type=Path, required=True)
    result.add_argument("--reference-dir", type=Path, required=True)
    result.add_argument("--a1-manifest", type=Path, required=True)
    result.add_argument("--a1-report", type=Path, required=True)
    result.add_argument("--model-pins-dir", type=Path, required=True)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    try:
        verified = verify_diversity_fad_artifact(
            args.artifact_dir,
            generation_dir=args.generation_dir,
            eval_manifest_dir=args.eval_manifest_dir,
            reference_dir=args.reference_dir,
            a1_manifest=args.a1_manifest,
            a1_report=args.a1_report,
            model_pins_dir=args.model_pins_dir,
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
                "artifact_seal_sha256": verified["artifact_seal_sha256"],
                "condition_id": verified["summary"]["condition_id"],
                "mert_diversity_mean_cosine_distance": verified[
                    "mert_diversity_mean_cosine_distance"
                ],
                "fad_pipeline_check_passed": verified["fad_pipeline_check_passed"],
                "fad_scores": verified["fad_scores"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
