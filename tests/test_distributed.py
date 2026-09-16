from __future__ import annotations

from datetime import timedelta
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ptc_opd.distributed import globally_normalized_loss
from ptc_opd.losses import ptc_opd_loss


class _SharedStudentLogits(nn.Module):
    """One shared parameter whose rank-local slices emulate different batches."""

    def __init__(self, student: torch.Tensor) -> None:
        super().__init__()
        self.student = nn.Parameter(student)

    def forward(self, rank: int) -> torch.Tensor:
        return self.student[rank : rank + 1]


def _worker(rank: int, world_size: int, rendezvous_file: str, output_dir: str) -> None:
    torch.set_num_threads(1)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        base_student = torch.tensor(
            [
                [[[0.4, -0.1], [0.2, 0.0], [0.1, -0.2]]],
                [[[-0.3, 0.5], [0.8, -0.4], [0.0, 0.0]]],
            ],
            dtype=torch.float64,
        )
        teacher = torch.tensor(
            [
                [[[-0.2, 0.6], [0.7, -0.1], [0.3, -0.3]]],
                [[[0.4, -0.2], [-0.2, 0.6], [0.1, -0.1]]],
            ],
            dtype=torch.float64,
        )
        valid = torch.tensor(
            [
                [[True, False, False]],
                [[True, True, True]],
            ]
        )
        ddp_student = DDP(
            _SharedStudentLogits(base_student),
            device_ids=None,
            output_device=None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        output = ptc_opd_loss(
            ddp_student(rank),
            teacher[rank : rank + 1],
            valid_mask=valid[rank : rank + 1],
            mode="uniform",
        )
        global_ratio = globally_normalized_loss(output, require_distributed=True)
        global_ratio.backward_loss.backward()
        # This is the real CPU DDP reducer result for one shared parameter, not
        # a manual all-reduce of semantically unrelated rank-local inputs.
        gradient = ddp_student.module.student.grad
        if gradient is None:
            raise RuntimeError("DDP shared student parameter produced no gradient")
        torch.save(
            {
                "rank": rank,
                "gradient": gradient.detach().clone(),
                "global_loss": global_ratio.global_loss,
                "global_denominator": global_ratio.global_denominator,
            },
            Path(output_dir) / f"rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available(), "torch.distributed unavailable")
class DistributedRatioTest(unittest.TestCase):
    def test_single_process_matches_local_loss(self) -> None:
        student = torch.randn(1, 2, 3, 4, requires_grad=True)
        teacher = torch.randn_like(student)
        output = ptc_opd_loss(student, teacher, mode="uniform")
        result = globally_normalized_loss(output)
        torch.testing.assert_close(result.backward_loss, output.loss)
        torch.testing.assert_close(result.global_loss, output.loss.detach())
        self.assertEqual(result.world_size, 1)

    def test_ddp_scaling_algebra_matches_concatenated_reference(self) -> None:
        student = torch.tensor(
            [
                [[[0.4, -0.1], [0.2, 0.0], [0.1, -0.2]]],
                [[[-0.3, 0.5], [0.8, -0.4], [0.0, 0.0]]],
            ],
            dtype=torch.float64,
        )
        teacher = torch.tensor(
            [
                [[[-0.2, 0.6], [0.7, -0.1], [0.3, -0.3]]],
                [[[0.4, -0.2], [-0.2, 0.6], [0.1, -0.1]]],
            ],
            dtype=torch.float64,
        )
        valid = torch.tensor(
            [
                [[True, False, False]],
                [[True, True, True]],
            ]
        )
        local_students = [
            student[index : index + 1].clone().requires_grad_() for index in range(2)
        ]
        local_outputs = [
            ptc_opd_loss(
                local_students[index],
                teacher[index : index + 1],
                valid_mask=valid[index : index + 1],
                mode="uniform",
            )
            for index in range(2)
        ]
        global_denominator = sum(
            (output.effective_weight_sum for output in local_outputs),
            torch.zeros((), dtype=torch.float32),
        )
        # DDP will average these two independently computed gradients.
        for output in local_outputs:
            (2.0 * output.numerator / global_denominator).backward()

        reference_student = student.clone().requires_grad_()
        reference = ptc_opd_loss(
            reference_student, teacher, valid_mask=valid, mode="uniform"
        )
        reference.loss.backward()
        for index, local_student in enumerate(local_students):
            torch.testing.assert_close(
                local_student.grad / 2.0,
                reference_student.grad[index : index + 1],
            )

    @unittest.skipUnless(
        os.environ.get("PTC_RUN_MULTIPROCESS_TESTS") == "1",
        "set PTC_RUN_MULTIPROCESS_TESTS=1 on a host that permits local process groups",
    )
    def test_two_rank_unequal_counts_matches_concatenated_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rendezvous = str(Path(temporary) / "gloo-rendezvous")
            # PTC_RUN_MULTIPROCESS_TESTS=1 is an explicit qualification opt-in.
            # Any rendezvous or process-group failure must therefore fail A06,
            # never degrade the required integration test to a successful skip.
            mp.spawn(_worker, args=(2, rendezvous, temporary), nprocs=2, join=True)
            rank_outputs = [
                torch.load(Path(temporary) / f"rank-{rank}.pt") for rank in range(2)
            ]

            student = torch.tensor(
                [
                    [[[0.4, -0.1], [0.2, 0.0], [0.1, -0.2]]],
                    [[[-0.3, 0.5], [0.8, -0.4], [0.0, 0.0]]],
                ],
                dtype=torch.float64,
                requires_grad=True,
            )
            teacher = torch.tensor(
                [
                    [[[-0.2, 0.6], [0.7, -0.1], [0.3, -0.3]]],
                    [[[0.4, -0.2], [-0.2, 0.6], [0.1, -0.1]]],
                ],
                dtype=torch.float64,
            )
            valid = torch.tensor(
                [
                    [[True, False, False]],
                    [[True, True, True]],
                ]
            )
            reference = ptc_opd_loss(
                student, teacher, valid_mask=valid, mode="uniform"
            )
            reference.loss.backward()

            for rank, saved in enumerate(rank_outputs):
                self.assertEqual(saved["rank"], rank)
                torch.testing.assert_close(saved["global_loss"], reference.loss.detach())
                torch.testing.assert_close(
                    saved["gradient"],
                    student.grad,
                )
                torch.testing.assert_close(
                    saved["global_denominator"],
                    reference.effective_weight_sum,
                )


if __name__ == "__main__":
    unittest.main()
