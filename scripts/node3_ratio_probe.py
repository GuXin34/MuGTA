#!/usr/bin/env python3
"""Eight-process proof of the Stage-1 global numerator/denominator algebra."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence


WORKPACK_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKPACK_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
import torch.distributed as dist

from ptc_opd.distributed import globally_normalized_loss


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def write_json_exclusive(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if os.environ.get("PTC_NODE3_GATE") != "1":
        raise RuntimeError("node3 ratio probe requires PTC_NODE3_GATE=1")
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
    if any(name not in os.environ for name in required):
        raise RuntimeError("launch node3 ratio probe with torchrun")
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm is forbidden for the node-3 gate")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    group_rank = int(os.environ.get("GROUP_RANK", "-1"))
    if (world_size, local_world_size, group_rank) != (8, 8, 0):
        raise RuntimeError(
            "node3 ratio probe requires one standalone node with exactly 8 ranks"
        )
    if not 0 <= local_rank < 8:
        raise RuntimeError("LOCAL_RANK must be in [0,7]")

    dist.init_process_group(backend="gloo", init_method="env://")
    try:
        coefficient = float(rank + 1)
        parameter = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        output = SimpleNamespace(
            numerator=coefficient * parameter,
            effective_weight_sum=torch.tensor(coefficient, dtype=torch.float64),
        )
        ratio = globally_normalized_loss(output, require_distributed=True)
        ratio.backward_loss.backward()
        if parameter.grad is None:
            raise RuntimeError("ratio probe produced no local gradient")
        ddp_averaged_gradient = parameter.grad.detach().clone()
        dist.all_reduce(ddp_averaged_gradient, op=dist.ReduceOp.SUM)
        ddp_averaged_gradient /= float(world_size)

        expected_denominator = float(sum(range(1, 9)))
        expected_global_loss = 0.5
        expected_gradient = 1.0
        tolerance = 1.0e-12
        local = {
            "rank": rank,
            "local_rank": local_rank,
            "local_numerator": float(output.numerator.detach().item()),
            "local_denominator": coefficient,
            "local_backward_gradient": float(parameter.grad.item()),
            "global_numerator": float(ratio.global_numerator.item()),
            "global_denominator": float(ratio.global_denominator.item()),
            "global_loss": float(ratio.global_loss.item()),
            "ddp_averaged_gradient": float(ddp_averaged_gradient.item()),
        }
        gathered: List[Any] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local)
        passed = all(
            abs(item["global_denominator"] - expected_denominator) <= tolerance
            and abs(item["global_loss"] - expected_global_loss) <= tolerance
            and abs(item["ddp_averaged_gradient"] - expected_gradient) <= tolerance
            for item in gathered
        )
        if [item["rank"] for item in gathered] != list(range(8)):
            raise RuntimeError("ratio evidence does not cover ranks 0..7")
        if not passed:
            raise RuntimeError("eight-process global-ratio proof failed")
        if rank == 0:
            write_json_exclusive(
                args.output,
                {
                    "schema_version": "ptc-opd-node3-ratio-v1",
                    "status": "passed",
                    "backend": "gloo",
                    "world_size": world_size,
                    "expected_global_denominator": expected_denominator,
                    "expected_global_loss": expected_global_loss,
                    "expected_ddp_averaged_gradient": expected_gradient,
                    "ranks": gathered,
                },
            )
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
