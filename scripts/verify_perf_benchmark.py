#!/usr/bin/env python3
"""Fail-closed verifier for arm, node, or aggregate performance evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from perf_benchmark_common import (  # noqa: E402
    SUMMARY_SCHEMA_VERSION,
    PerfContractError,
    build_live_input_identity,
    read_json,
    sha256_file,
    verify_arm_directory,
    verify_arm_input_bindings,
    verify_node_input_bindings,
    verify_seal,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--arm-dir", type=Path)
    group.add_argument("--node-dir", type=Path)
    group.add_argument("--aggregate-dir", type=Path)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--small-cfg-dir", type=Path)
    parser.add_argument("--audiocraft-dir", type=Path)
    parser.add_argument("--musicgen-small-dir", type=Path)
    args = parser.parse_args(argv)
    scientific = (
        args.train_manifest,
        args.small_cfg_dir,
        args.audiocraft_dir,
        args.musicgen_small_dir,
    )
    if args.aggregate_dir is not None:
        if any(item is not None for item in scientific):
            parser.error("live scientific inputs are not accepted with --aggregate-dir")
    elif any(item is None for item in scientific):
        parser.error(
            "--arm-dir/--node-dir require --train-manifest, --small-cfg-dir, "
            "--audiocraft-dir, and --musicgen-small-dir"
        )
    return args


def verify_aggregate(path: Path) -> Dict[str, Any]:
    expected_members = {
        "ARTIFACT_SEAL.json",
        "STATUS.json",
        "paired_summary.csv",
        "paired_summary.json",
    }
    if {item.name for item in path.iterdir()} != expected_members:
        raise PerfContractError("aggregate top-level members differ")
    seal = verify_seal(path, "aggregate")
    if {item["path"] for item in seal["payloads"]} != {
        "STATUS.json", "paired_summary.csv", "paired_summary.json"
    }:
        raise PerfContractError("aggregate seal payload path set differs")
    summary = read_json(path / "paired_summary.json")
    status = read_json(path / "STATUS.json")
    if (
        summary.get("schema_version") != SUMMARY_SCHEMA_VERSION
        or summary.get("benchmark_only") is not True
        or summary.get("scientific_use_forbidden") is not True
        or summary.get("production_policy_decision") != "retain_true_full"
        or summary.get("false_or_min_outputs_authorized_for_science") is not False
        or summary.get("pilot_policy_change_authorized") is not False
    ):
        raise PerfContractError("aggregate summary policy contract differs")
    production = summary.get("production_true_full")
    if not isinstance(production, dict):
        raise PerfContractError("aggregate summary lacks production timing")
    for name in (
        "median_seconds_per_step_across_nodes",
        "conservative_p90_seconds_per_step",
        "steps_per_hour",
        "effective_samples_per_second",
    ):
        value = production.get(name)
        if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) <= 0.0:
            raise PerfContractError("aggregate production metric {} is invalid".format(name))
    expected_status = "passed" if summary.get("hard_resource_gate_passed") is True else "resource_gate_failed"
    if status.get("status") != expected_status:
        raise PerfContractError("aggregate STATUS and summary hard gate differ")
    return {
        "schema_version": "ptc-opd-stage1-perf-verification-v1",
        "status": "verified",
        "scope": "aggregate",
        "hard_resource_gate_passed": summary["hard_resource_gate_passed"],
        "measurement_quality": summary["measurement_quality"],
        "production_policy_decision": "retain_true_full",
        "artifact_seal_sha256": sha256_file(path / "ARTIFACT_SEAL.json"),
    }


def _canonical_artifact_root(path: Path, label: str) -> Path:
    supplied = path.expanduser().absolute()
    if supplied.is_symlink():
        raise PerfContractError("{} root may not be a symlink".format(label))
    resolved = supplied.resolve(strict=True)
    if not resolved.is_dir():
        raise PerfContractError("{} root must be a directory".format(label))
    return resolved


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.arm_dir is not None:
        arm_dir = _canonical_artifact_root(args.arm_dir, "arm")
        live = build_live_input_identity(
            train_manifest=args.train_manifest,
            small_cfg_dir=args.small_cfg_dir,
            audiocraft_dir=args.audiocraft_dir,
            musicgen_small_dir=args.musicgen_small_dir,
        )
        result = verify_arm_directory(arm_dir)
        result["live_input_identity"] = live
        result["arm_input_binding"] = verify_arm_input_bindings(arm_dir, live)
    elif args.node_dir is not None:
        result = verify_node_input_bindings(
            _canonical_artifact_root(args.node_dir, "node"),
            train_manifest=args.train_manifest,
            small_cfg_dir=args.small_cfg_dir,
            audiocraft_dir=args.audiocraft_dir,
            musicgen_small_dir=args.musicgen_small_dir,
        )
    else:
        result = verify_aggregate(
            _canonical_artifact_root(args.aggregate_dir, "aggregate")
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
