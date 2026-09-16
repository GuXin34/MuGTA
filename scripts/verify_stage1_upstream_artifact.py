#!/usr/bin/env python3
"""Verify a copied Stage-1 upstream-readiness artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence


WORKPACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKPACK_ROOT / "src"))

from ptc_opd.stage1_artifact import (  # noqa: E402
    Stage1ArtifactError,
    canonical_json_sha256,
    load_json_strict,
    verify_simple_seal,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    return parser.parse_args(argv)


def verify(directory: Path) -> dict:
    verify_simple_seal(
        directory,
        seal_name="artifact_seal.json",
        schema_version="ptc-opd-stage1-upstream-verification-seal-v1",
        status="ready_for_b1_prestability",
        payload_names=("upstream_verification.json",),
    )
    report = load_json_strict(directory / "upstream_verification.json")
    if report.get("schema_version") != "ptc-opd-stage1-upstream-verification-v1":
        raise Stage1ArtifactError("upstream report schema mismatch")
    if (
        report.get("scientific_status") != "ready_for_b1_prestability"
        or report.get("gate_passed") is not True
        or report.get("operationally_accepted") is not True
        or report.get("redline_touched") is not False
    ):
        raise Stage1ArtifactError("upstream report is not accepted")
    if report.get("full_b1_passed") is not False:
        raise Stage1ArtifactError("upstream report may not claim full B1")
    config = report.get("scientific_config")
    if not isinstance(config, dict) or report.get(
        "scientific_config_sha256"
    ) != canonical_json_sha256(config):
        raise Stage1ArtifactError("upstream scientific-config hash mismatch")
    if report.get("upstream_tree_sha256_before") != report.get(
        "upstream_tree_sha256_after"
    ):
        raise Stage1ArtifactError("upstream before/after identities differ")
    if report.get("t5_archive_sha256_before") != report.get(
        "t5_archive_sha256_after"
    ):
        raise Stage1ArtifactError("T5 archive before/after identity differs")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = verify(args.artifact_dir)
    except (OSError, ValueError, Stage1ArtifactError) as exc:
        print("UPSTREAM ARTIFACT VERIFY FAILED: {}".format(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "passed",
                "scientific_status": report["scientific_status"],
                "required_action": report["required_action"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
