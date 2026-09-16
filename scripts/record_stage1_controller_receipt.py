#!/usr/bin/env python3
"""Issue/verify Stage-1 receipts and immutable controller-ledger revisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional, Sequence


WORKPACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKPACK_ROOT / "src"))

from ptc_opd.stage1_artifact import Stage1ArtifactError, load_json_strict  # noqa: E402
from ptc_opd.stage1_controller_ledger import (  # noqa: E402
    CONTRACT_RELATIVE_PATH,
    initialize_controller_ledger,
    issue_stage_verifier_receipt,
    record_controller_receipt,
    stage_verifier_catalog,
    validate_current_controller_ledger_authorizations,
    validate_controller_ledger_authorizations,
    verify_stage_verifier_receipt,
)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workpack-root", type=Path, default=WORKPACK_ROOT)
    parser.add_argument("--contract-path", type=Path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    listed = sub.add_parser("list-registry")
    _common(listed)

    issue = sub.add_parser("issue")
    _common(issue)
    issue.add_argument("--stage-id", required=True)
    issue.add_argument("--evidence-json", type=Path, required=True)
    issue.add_argument("--dependency-receipts-json", type=Path, required=True)
    issue.add_argument("--attempt-number", type=int, required=True)
    issue.add_argument("--attempt-kind", choices=("initial", "yellow"), required=True)
    issue.add_argument(
        "--ledger-dir",
        type=Path,
        required=True,
        help="current sealed ledger revision; every initial/yellow attempt is parent-bound",
    )
    issue.add_argument("--output-dir", type=Path, required=True)

    verify = sub.add_parser("verify")
    _common(verify)
    verify.add_argument("--receipt-dir", type=Path, required=True)

    init = sub.add_parser("init-ledger")
    _common(init)
    init.add_argument("--output-dir", type=Path, required=True)

    record = sub.add_parser("record")
    _common(record)
    record.add_argument("--ledger-dir", type=Path, required=True)
    record.add_argument("--receipt-dir", type=Path, required=True)
    record.add_argument("--output-dir", type=Path, required=True)

    verify_ledger = sub.add_parser("verify-ledger")
    _common(verify_ledger)
    verify_ledger.add_argument("--ledger-dir", type=Path, required=True)
    return result


def _paths(path: Path, label: str) -> Dict[str, Path]:
    value = load_json_strict(path)
    result: Dict[str, Path] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or not isinstance(item, str) or not item:
            raise Stage1ArtifactError("{} must map nonempty names to paths".format(label))
        result[key] = Path(item)
    return result


def _context(args: argparse.Namespace) -> tuple[Path, Path]:
    root = args.workpack_root
    contract = args.contract_path or (root / CONTRACT_RELATIVE_PATH)
    return root, contract


def _print(value: Mapping[str, Any]) -> None:
    print(json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    root, contract = _context(args)
    try:
        if args.command == "list-registry":
            _print({"status": "verified", "registry": stage_verifier_catalog()})
        elif args.command == "issue":
            output = issue_stage_verifier_receipt(
                stage_id=args.stage_id,
                evidence_paths=_paths(args.evidence_json, "evidence JSON"),
                dependency_receipt_dirs=_paths(
                    args.dependency_receipts_json, "dependency receipt JSON"
                ),
                workpack_root=root,
                contract_path=contract,
                output_dir=args.output_dir,
                attempt_number=args.attempt_number,
                attempt_kind=args.attempt_kind,
                ledger_dir=args.ledger_dir,
            )
            verified = verify_stage_verifier_receipt(
                output, workpack_root=root, contract_path=contract
            )
            _print({"status": "verified", "receipt": verified["reference"]})
        elif args.command == "verify":
            verified = verify_stage_verifier_receipt(
                args.receipt_dir, workpack_root=root, contract_path=contract
            )
            _print({"status": "verified", "receipt": verified["reference"]})
        elif args.command == "init-ledger":
            output = initialize_controller_ledger(
                workpack_root=root, contract_path=contract, output_dir=args.output_dir
            )
            verified = validate_controller_ledger_authorizations(
                output, workpack_root=root, contract_path=contract
            )
            _print(
                {
                    "status": "verified",
                    "ledger_dir": str(output.resolve()),
                    "revision": verified["ledger"]["revision"],
                }
            )
        elif args.command == "record":
            output = record_controller_receipt(
                args.ledger_dir,
                args.receipt_dir,
                workpack_root=root,
                contract_path=contract,
                output_dir=args.output_dir,
            )
            verified = validate_controller_ledger_authorizations(
                output, workpack_root=root, contract_path=contract
            )
            _print(
                {
                    "status": "verified",
                    "ledger_dir": str(output.resolve()),
                    "revision": verified["ledger"]["revision"],
                }
            )
        else:
            verified = validate_current_controller_ledger_authorizations(
                args.ledger_dir, workpack_root=root, contract_path=contract
            )
            _print(
                {
                    "status": "verified",
                    "revision": verified["ledger"]["revision"],
                    "stages": verified["legacy_stages"],
                }
            )
    except (OSError, RuntimeError, TypeError, ValueError, Stage1ArtifactError) as exc:
        print(
            json.dumps(
                {"status": "invalid", "error_type": type(exc).__name__, "error": str(exc)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
