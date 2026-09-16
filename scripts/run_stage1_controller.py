#!/usr/bin/env python3
"""Inspect the fail-closed Stage-1 DAG and choose its next authorized action."""

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
    controller_next_action,
    controller_plan,
    controller_preflight,
    load_autonomy_contract,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workpack-root", type=Path, default=WORKPACK_ROOT)
    parser.add_argument(
        "--contract",
        type=Path,
        default=WORKPACK_ROOT / "configs" / "stage1_autonomy_contract.json",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    subparsers.add_parser("preflight")
    action = subparsers.add_parser("next-action")
    action.add_argument(
        "--ledger-dir",
        type=Path,
        required=True,
        help="sealed controller-ledger revision directory, never a bare JSON file",
    )
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
        contract = load_autonomy_contract(args.contract)
        if args.command == "plan":
            result = controller_plan(contract)
        else:
            preflight = controller_preflight(args.workpack_root, args.contract)
            if args.command == "preflight":
                result = preflight
            else:
                result = controller_next_action(
                    contract,
                    preflight,
                    args.ledger_dir,
                    workpack_root=args.workpack_root,
                    contract_path=args.contract,
                )
    except (OSError, Stage1ArtifactError, TypeError, ValueError) as exc:
        _print(
            {
                "schema_version": "ptc-opd-stage1-controller-cli-v1",
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
