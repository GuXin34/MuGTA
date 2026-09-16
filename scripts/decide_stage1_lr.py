#!/usr/bin/env python3
"""Verify a frozen LR evaluation summary and seal its deterministic decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


WORKPACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKPACK_ROOT / "src"))

from ptc_opd.stage1_artifact import Stage1ArtifactError, sha256_file  # noqa: E402
from ptc_opd.stage1_control import (  # noqa: E402
    LR_DECISION_BASENAME,
    publish_lr_decision,
    verify_lr_decision,
    verify_lr_evaluation_summary,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser(
        "verify-summary", help="validate the sealed three-candidate input"
    )
    summary.add_argument("--summary-dir", type=Path, required=True)

    seal = subparsers.add_parser("seal", help="publish a new LR decision artifact")
    seal.add_argument("--summary-dir", type=Path, required=True)
    seal.add_argument("--output-dir", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="recompute and verify a decision")
    verify.add_argument("--summary-dir", type=Path, required=True)
    verify.add_argument("--decision-dir", type=Path, required=True)
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
    try:
        if args.command == "verify-summary":
            summary = verify_lr_evaluation_summary(args.summary_dir)
            result = {
                "schema_version": "ptc-opd-lr-summary-verification-v1",
                "status": "verified",
                "candidate_count": len(summary["candidates"]),
                "artifact_seal_sha256": sha256_file(
                    args.summary_dir / "artifact_seal.json"
                ),
            }
        elif args.command == "seal":
            publish_lr_decision(args.summary_dir, args.output_dir)
            decision = verify_lr_decision(args.output_dir, args.summary_dir)
            result = {
                "schema_version": "ptc-opd-lr-decision-verification-v1",
                "status": "verified",
                "scientific_status": decision["status"],
                "selected_learning_rate": decision["selected_learning_rate"],
                "decision_sha256": sha256_file(
                    args.output_dir / LR_DECISION_BASENAME
                ),
                "artifact_seal_sha256": sha256_file(
                    args.output_dir / "artifact_seal.json"
                ),
            }
        else:
            decision = verify_lr_decision(args.decision_dir, args.summary_dir)
            result = {
                "schema_version": "ptc-opd-lr-decision-verification-v1",
                "status": "verified",
                "scientific_status": decision["status"],
                "selected_learning_rate": decision["selected_learning_rate"],
                "decision_sha256": sha256_file(
                    args.decision_dir / LR_DECISION_BASENAME
                ),
                "artifact_seal_sha256": sha256_file(
                    args.decision_dir / "artifact_seal.json"
                ),
            }
    except (OSError, Stage1ArtifactError, TypeError, ValueError) as exc:
        _print(
            {
                "schema_version": "ptc-opd-lr-decision-cli-v1",
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
