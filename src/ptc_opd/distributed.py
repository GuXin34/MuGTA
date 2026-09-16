"""Distributed helpers for globally normalized PTC-OPD losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor

from .losses import OPDLossOutput


@dataclass(frozen=True)
class GlobalRatioOutput:
    """DDP-correct training loss and detached global logging statistics."""

    backward_loss: Tensor
    global_loss: Tensor
    global_numerator: Tensor
    global_denominator: Tensor
    world_size: int


def globally_normalized_loss(
    output: OPDLossOutput,
    *,
    process_group: Optional[dist.ProcessGroup] = None,
    require_distributed: bool = False,
) -> GlobalRatioOutput:
    """Construct the true global numerator/global denominator objective.

    Ordinary ``DistributedDataParallel`` averages gradients across ranks. If
    rank ``r`` owns differentiable numerator ``N_r`` and detached denominator
    ``D_r``, each rank must backpropagate

    ``world_size * N_r / sum_j D_j``.

    DDP's subsequent gradient average then equals the gradient of
    ``sum_j N_j / sum_j D_j``. Averaging local ratios is wrong whenever ranks
    contain different valid/selected effective weight.
    """

    numerator = output.numerator
    denominator = output.effective_weight_sum.detach()
    if numerator.ndim != 0 or denominator.ndim != 0:
        raise ValueError("PTC numerator and denominator must be scalar tensors")
    if numerator.device != denominator.device:
        raise ValueError("PTC numerator and denominator must share a device")
    if not numerator.requires_grad:
        raise ValueError("output.numerator must retain its autograd graph")

    initialized = dist.is_available() and dist.is_initialized()
    if require_distributed and not initialized:
        raise RuntimeError("torch.distributed is not initialized")
    world_size = dist.get_world_size(process_group) if initialized else 1

    global_denominator = denominator.clone()
    global_numerator = numerator.detach().clone()
    if initialized:
        dist.all_reduce(
            global_denominator, op=dist.ReduceOp.SUM, group=process_group
        )
        dist.all_reduce(
            global_numerator, op=dist.ReduceOp.SUM, group=process_group
        )

    if not bool(torch.isfinite(global_denominator).item()):
        raise FloatingPointError("global effective-weight denominator is NaN/Inf")
    if not bool((global_denominator > 0).item()):
        raise ValueError("global effective-weight denominator must be positive")
    if not bool(torch.isfinite(global_numerator).item()):
        raise FloatingPointError("global numerator is NaN/Inf")

    backward_loss = float(world_size) * numerator / global_denominator
    global_loss = global_numerator / global_denominator
    return GlobalRatioOutput(
        backward_loss=backward_loss,
        global_loss=global_loss,
        global_numerator=global_numerator,
        global_denominator=global_denominator,
        world_size=world_size,
    )


__all__ = ["GlobalRatioOutput", "globally_normalized_loss"]

