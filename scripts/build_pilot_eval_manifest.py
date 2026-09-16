#!/usr/bin/env python3
"""Build or verify the frozen 128-prompt Stage-1 pilot evaluation manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


WORKPACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKPACK_ROOT / "src"))

from ptc_opd.stage1_artifact import Stage1ArtifactError  # noqa: E402
from ptc_opd.stage1_control import (  # noqa: E402
    publish_pilot_eval_manifest,
    verify_pilot_eval_manifest,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="publish a new sealed manifest")
    build.add_argument("--source-dev-manifest", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="bit-rebuild and verify an artifact")
    verify.add_argument("--source-dev-manifest", type=Path, required=True)
    verify.add_argument("--artifact-dir", type=Path, required=True)
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
        if args.command == "build":
            publish_pilot_eval_manifest(args.source_dev_manifest, args.output_dir)
            result = verify_pilot_eval_manifest(
                args.output_dir, args.source_dev_manifest
            )
        else:
            result = verify_pilot_eval_manifest(
                args.artifact_dir, args.source_dev_manifest
            )
    except (OSError, Stage1ArtifactError, TypeError, ValueError) as exc:
        _print(
            {
                "schema_version": "ptc-opd-pilot-eval-manifest-cli-v1",
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
