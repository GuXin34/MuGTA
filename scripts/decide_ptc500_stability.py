#!/usr/bin/env python3
"""Consume a sealed PTC500 run and publish or verify its stability decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Mapping, Optional, Sequence


WORKPACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKPACK_ROOT / "src"))

from ptc_opd.stage1_artifact import Stage1ArtifactError, sha256_file  # noqa: E402
from ptc_opd.stage1_control import (  # noqa: E402
    PTC500_REPORT_BASENAME,
    publish_ptc500_stability_report,
    verify_ptc500_stability_report,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("consume", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("--run-dir", type=Path, required=True)
        child.add_argument("--lr-summary-dir", type=Path, required=True)
        child.add_argument("--lr-decision-dir", type=Path, required=True)
        child.add_argument("--b1-prestability-dir", type=Path, required=True)
        child.add_argument("--workpack-root", type=Path, default=WORKPACK_ROOT)
        if command == "consume":
            child.add_argument("--output-dir", type=Path, required=True)
        else:
            child.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Stage1ArtifactError(
                "official Stage-1 verifier emitted duplicate key {!r}".format(key)
            )
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise Stage1ArtifactError(
        "official Stage-1 verifier emitted non-finite {}".format(value)
    )


def run_official_stage1_verifier(
    workpack_root: Path, run_dir: Path
) -> Dict[str, Any]:
    supplied_root = workpack_root.expanduser()
    if supplied_root.is_symlink():
        raise Stage1ArtifactError("workpack root may not be a symlink")
    verifier = supplied_root.resolve(strict=True) / "scripts" / "verify_stage1_run.py"
    if verifier.is_symlink() or not verifier.is_file():
        raise Stage1ArtifactError("official Stage-1 verifier is missing or a symlink")
    completed = subprocess.run(
        [sys.executable, str(verifier), "--run-dir", str(run_dir)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise Stage1ArtifactError(
            "official Stage-1 verifier failed rc={}: {}".format(
                completed.returncode, detail
            )
        )
    try:
        value = json.loads(
            completed.stdout,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise Stage1ArtifactError(
            "official Stage-1 verifier did not emit one JSON object"
        ) from exc
    if not isinstance(value, dict) or value.get("status") != "verified":
        raise Stage1ArtifactError("official Stage-1 verifier result is not verified")
    return value


def run_official_b1_verifier(
    workpack_root: Path, artifact_dir: Path
) -> Dict[str, Any]:
    supplied_root = workpack_root.expanduser()
    supplied_artifact = artifact_dir.expanduser()
    if supplied_root.is_symlink() or supplied_artifact.is_symlink():
        raise Stage1ArtifactError("workpack/B1 roots may not be symlinks")
    verifier = (
        supplied_root.resolve(strict=True)
        / "scripts"
        / "verify_b1_prestability.py"
    )
    if verifier.is_symlink() or not verifier.is_file():
        raise Stage1ArtifactError("official B1 verifier is missing or a symlink")
    completed = subprocess.run(
        [sys.executable, str(verifier), "final", str(supplied_artifact)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise Stage1ArtifactError(
            "official B1 verifier failed rc={}: {}".format(
                completed.returncode, detail
            )
        )
    try:
        value = json.loads(
            completed.stdout,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise Stage1ArtifactError(
            "official B1 verifier did not emit one JSON object"
        ) from exc
    if (
        not isinstance(value, dict)
        or value.get("kind") != "final"
        or value.get("prestability_gate_passed") is not True
        or value.get("full_b1_passed") is not False
    ):
        raise Stage1ArtifactError("official B1 verifier result is not prestability-pass")
    return value


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
        run_verification = run_official_stage1_verifier(
            args.workpack_root, args.run_dir
        )
        b1_verification = run_official_b1_verifier(
            args.workpack_root, args.b1_prestability_dir
        )
        if args.command == "consume":
            publish_ptc500_stability_report(
                args.run_dir,
                args.lr_decision_dir,
                args.lr_summary_dir,
                run_verification,
                args.b1_prestability_dir,
                b1_verification,
                args.output_dir,
            )
            artifact_dir = args.output_dir
        else:
            artifact_dir = args.artifact_dir
        decision = verify_ptc500_stability_report(
            artifact_dir,
            args.run_dir,
            args.lr_decision_dir,
            args.lr_summary_dir,
            run_verification,
            args.b1_prestability_dir,
            b1_verification,
        )
        result = {
            "schema_version": "ptc-opd-ptc500-stability-verification-v1",
            "status": "verified",
            "scientific_status": decision["scientific_status"],
            "gate_passed": decision["gate_passed"],
            "required_action": decision["required_action"],
            "report_sha256": sha256_file(artifact_dir / PTC500_REPORT_BASENAME),
            "artifact_seal_sha256": sha256_file(
                artifact_dir / "artifact_seal.json"
            ),
        }
    except (OSError, Stage1ArtifactError, TypeError, ValueError) as exc:
        _print(
            {
                "schema_version": "ptc-opd-ptc500-stability-cli-v1",
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
