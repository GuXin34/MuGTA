#!/usr/bin/env python3
"""Combine independently verified B1 single/DDP evidence into one atomic packet."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from typing import Optional, Sequence


SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import b1_prestability_contract as contract
import verify_b1_prestability as verifier


_B19_ITEM = "B1.9_real_checkpoint_eight_rank_concatenated_reference"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-artifact-dir", type=Path, required=True)
    parser.add_argument("--distributed-artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def execute(args: argparse.Namespace) -> None:
    supplied_single = args.single_artifact_dir.expanduser().absolute()
    supplied_distributed = args.distributed_artifact_dir.expanduser().absolute()
    if supplied_single.is_symlink() or supplied_distributed.is_symlink():
        raise contract.ContractError("B1 input artifact roots must not be symlinks")
    single_root = supplied_single.resolve(strict=True)
    distributed_root = supplied_distributed.resolve(strict=True)
    single_identity = verifier.verify_single(single_root)
    distributed_identity = verifier.verify_distributed(distributed_root)
    if distributed_identity.get("single_artifact_seal_sha256") != single_identity.get("seal_sha256"):
        raise contract.ContractError("distributed evidence was not derived from the supplied single artifact")
    if distributed_identity.get("sample_ids") != single_identity.get("sample_ids"):
        raise contract.ContractError("single/distributed audit sample order differs")

    item_status = {}
    for name in contract.SINGLE_CASES:
        item_status[name] = {
            "status": "passed",
            "source": "single_gpu_real_checkpoint",
        }
    for name in contract.RETAINED_NODE3_ITEMS:
        item_status[name] = {
            "status": "passed_retained",
            "source": "node3_15_of_15",
        }
    item_status[_B19_ITEM] = {
        "status": "passed",
        "source": "eight_gpu_real_checkpoint",
    }
    item_status[contract.PENDING_B1_ITEM] = {
        "status": "pending",
        "source": None,
    }
    completed = list(contract.SINGLE_CASES) + list(contract.RETAINED_NODE3_ITEMS) + [_B19_ITEM]
    summary = {
        "schema_version": contract.SUMMARY_SCHEMA,
        "scientific_status": contract.SUMMARY_STATUS,
        "gate_passed": True,
        "operationally_accepted": True,
        "redline_touched": False,
        "prestability_gate_passed": True,
        "full_b1_passed": False,
        "item_status": item_status,
        "completed_b1_items": completed,
        "pending_b1_items": [contract.PENDING_B1_ITEM],
        "evidence": {
            "single_artifact_seal_sha256": single_identity["seal_sha256"],
            "single_scientific_config_sha256": single_identity[
                "scientific_config_sha256"
            ],
            "distributed_artifact_seal_sha256": distributed_identity["seal_sha256"],
            "distributed_scientific_config_sha256": distributed_identity[
                "scientific_config_sha256"
            ],
            "node3_status_sha256": single_identity["node3_identity"][
                "status_sha256"
            ],
            "t5_closure_artifact_seal_sha256": (
                contract.EXPECTED_T5_CLOSURE_SEAL_SHA256
            ),
        },
        "next_authorized_stage": "performance_benchmark_then_uniform_lr_sweep",
        "authorization_scope": "pre_stability_only_b1_11_remains_mandatory",
    }

    with contract.staged_output_directory(args.output_dir) as staging:
        copies = {
            single_root / "audit_inputs.npz": staging / "audit_inputs.npz",
            single_root / "single_gpu_results.json": staging / "single_gpu_results.json",
            single_root / "artifact_seal.json": staging / "single_gpu_artifact_seal.json",
            distributed_root / "distributed_results.json": staging / "distributed_results.json",
            distributed_root / "artifact_seal.json": staging / "distributed_artifact_seal.json",
        }
        for source, target in copies.items():
            if target.exists():
                raise FileExistsError("staging target unexpectedly exists")
            shutil.copyfile(str(source), str(target))
        contract.write_json_exclusive(staging / "b1_prestability_summary.json", summary)
        contract.write_artifact_seal(
            staging,
            status=contract.SUMMARY_STATUS,
            member_names=contract.FINAL_MEMBERS,
        )
        verifier.verify_final(staging)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    execute(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
