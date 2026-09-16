#!/usr/bin/env python3
"""Publish or verify the combined B1.1--B1.11 closure artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


WORKPACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKPACK_ROOT / "src"))
sys.path.insert(0, str(WORKPACK_ROOT / "scripts"))

from decide_ptc500_stability import run_official_b1_verifier  # noqa: E402
from ptc_opd.stage1_artifact import Stage1ArtifactError, sha256_file  # noqa: E402
from ptc_opd.stage1_control import (  # noqa: E402
    B1_FULL_CLOSURE_BASENAME,
    publish_b1_full_closure,
    verify_b1_full_closure,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("publish", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("--ptc500-artifact-dir", type=Path, required=True)
        child.add_argument("--b1-prestability-dir", type=Path, required=True)
        child.add_argument("--workpack-root", type=Path, default=WORKPACK_ROOT)
        if command == "publish":
            child.add_argument("--output-dir", type=Path, required=True)
        else:
            child.add_argument("--artifact-dir", type=Path, required=True)
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
        verification = run_official_b1_verifier(
            args.workpack_root, args.b1_prestability_dir
        )
        if args.command == "publish":
            publish_b1_full_closure(
                args.ptc500_artifact_dir,
                args.b1_prestability_dir,
                verification,
                args.output_dir,
            )
            artifact_dir = args.output_dir
        else:
            artifact_dir = args.artifact_dir
        closure = verify_b1_full_closure(
            artifact_dir,
            args.ptc500_artifact_dir,
            args.b1_prestability_dir,
            verification,
        )
        result = {
            "schema_version": "ptc-opd-b1-full-closure-verification-v1",
            "status": "verified",
            "scientific_status": closure["scientific_status"],
            "gate_passed": closure["gate_passed"],
            "full_b1_passed": closure["full_b1_passed"],
            "required_action": closure["required_action"],
            "report_sha256": sha256_file(
                artifact_dir / B1_FULL_CLOSURE_BASENAME
            ),
            "artifact_seal_sha256": sha256_file(
                artifact_dir / "artifact_seal.json"
            ),
        }
    except (OSError, Stage1ArtifactError, TypeError, ValueError) as exc:
        _print(
            {
                "schema_version": "ptc-opd-b1-full-closure-cli-v1",
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
