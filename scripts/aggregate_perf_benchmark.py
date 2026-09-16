#!/usr/bin/env python3
"""Aggregate four sealed node-local Williams performance sequences."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from perf_benchmark_common import (  # noqa: E402
    SUMMARY_SCHEMA_VERSION,
    aggregate_node_directories,
    file_identity,
    sha256_file,
    verify_node_directory,
    verify_seal,
    write_json_exclusive,
    write_seal,
    write_summary_csv,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if len(args.node_dir) != 4:
        raise ValueError("pass --node-dir exactly four times")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError("refusing to overwrite {}".format(args.output_dir))
    node_dirs = [path.resolve() for path in args.node_dir]
    node_verifications = [verify_node_directory(path) for path in node_dirs]
    summary = aggregate_node_directories(node_dirs)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary["created_utc"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary["input_nodes"] = {
        item["node_label"]: {
            "directory": str(path),
            "artifact_seal_sha256": item["artifact_seal_sha256"],
        }
        for path, item in zip(node_dirs, node_verifications)
    }
    write_json_exclusive(args.output_dir / "paired_summary.json", summary)
    write_summary_csv(args.output_dir / "paired_summary.csv", summary)
    status = {
        "schema_version": "ptc-opd-stage1-perf-aggregate-status-v1",
        "benchmark_only": True,
        "scientific_use_forbidden": True,
        "status": "passed" if summary["hard_resource_gate_passed"] else "resource_gate_failed",
        "redline_touched": False,
        "measurement_quality": summary["measurement_quality"],
        "performance_result_role": "informational_only",
        "production_policy_decision": "retain_true_full",
        "pilot_policy_change_authorized": False,
        "required_action": (
            "auto_continue_to_next_frozen_stage"
            if summary["hard_resource_gate_passed"] and summary["measurement_quality"] == "complete"
            else "extend_measurement_or_escalate_resource_failure_without_changing_ddp_policy"
        ),
    }
    write_json_exclusive(args.output_dir / "STATUS.json", status)
    write_seal(
        args.output_dir,
        scope="aggregate",
        relative_paths=["STATUS.json", "paired_summary.csv", "paired_summary.json"],
    )
    verify_seal(args.output_dir, "aggregate")
    result = {
        "schema_version": "ptc-opd-stage1-perf-verification-v1",
        "status": "verified",
        "scope": "aggregate",
        "scientific_status": "informational_performance_quantified",
        "hard_resource_gate_passed": summary["hard_resource_gate_passed"],
        "measurement_quality": summary["measurement_quality"],
        "production_policy_decision": "retain_true_full",
        "artifact_seal_sha256": sha256_file(args.output_dir / "ARTIFACT_SEAL.json"),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
