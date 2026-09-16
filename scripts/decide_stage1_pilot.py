#!/usr/bin/env python3
"""Seal or verify the frozen six-method small-pilot go/no-go decision."""

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
    SMALL_PILOT_DECISION_BASENAME,
    publish_small_pilot_decision,
    verify_small_pilot_decision,
    verify_small_pilot_summary,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    summary = sub.add_parser("verify-summary")
    summary.add_argument("--summary-dir", type=Path, required=True)
    seal = sub.add_parser("seal")
    seal.add_argument("--summary-dir", type=Path, required=True)
    seal.add_argument("--output-dir", type=Path, required=True)
    verify = sub.add_parser("verify")
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
            summary = verify_small_pilot_summary(args.summary_dir)
            result = {
                "schema_version": "ptc-opd-small-pilot-summary-verification-v1",
                "status": "verified",
                "method_count": len(summary["methods"]),
                "artifact_seal_sha256": sha256_file(
                    args.summary_dir / "artifact_seal.json"
                ),
            }
        else:
            if args.command == "seal":
                publish_small_pilot_decision(args.summary_dir, args.output_dir)
                decision_dir = args.output_dir
            else:
                decision_dir = args.decision_dir
            decision = verify_small_pilot_decision(decision_dir, args.summary_dir)
            result = {
                "schema_version": "ptc-opd-small-pilot-decision-verification-v1",
                "status": "verified",
                "scientific_status": decision["scientific_status"],
                "gate_passed": decision["gate_passed"],
                "required_action": decision["required_action"],
                "decision_sha256": sha256_file(
                    decision_dir / SMALL_PILOT_DECISION_BASENAME
                ),
                "artifact_seal_sha256": sha256_file(
                    decision_dir / "artifact_seal.json"
                ),
            }
    except (OSError, Stage1ArtifactError, TypeError, ValueError) as exc:
        _print(
            {
                "schema_version": "ptc-opd-small-pilot-decision-cli-v1",
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
