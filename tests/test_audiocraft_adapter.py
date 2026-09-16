"""Unit tests for the import-free AudioCraft scoring adapter."""

from __future__ import annotations

from dataclasses import dataclass
import sys
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import Tensor, nn

from ptc_opd.audiocraft_adapter import (
    batch_condition_tensors,
    score_audiocraft_trajectory,
)


@dataclass
class _FakeLMOutput:
    logits: Tensor
    mask: Tensor


class _FakeLM(nn.Module):
    """LMOutput-compatible fake whose condition marker controls its logits."""

    def __init__(
        self,
        num_codebooks: int,
        vocabulary: int,
        masks_by_marker: Dict[int, Tensor],
        *,
        bad_position: Optional[Tuple[int, int]] = None,
        truncate_time: bool = False,
    ) -> None:
        super().__init__()
        self.num_codebooks = num_codebooks
        self.vocabulary = vocabulary
        self.card = vocabulary
        self.masks_by_marker = masks_by_marker
        self.bad_position = bad_position
        self.truncate_time = truncate_time
        self.weight = nn.Parameter(torch.tensor(0.75))
        self.calls: List[dict] = []

    def compute_predictions(
        self,
        codes: Tensor,
        conditions: list,
        condition_tensors: dict,
        keep_only_valid_steps: bool,
    ) -> _FakeLMOutput:
        self.assert_no_raw_conditions(conditions)
        embedding, condition_mask = condition_tensors["description"]
        markers = embedding[:, 0, 0]
        batch, codebooks, time = codes.shape
        vocabulary_axis = torch.arange(
            1,
            self.vocabulary + 1,
            dtype=self.weight.dtype,
            device=codes.device,
        ).view(1, 1, 1, self.vocabulary)
        raw = codes.to(self.weight.dtype).unsqueeze(-1) + vocabulary_axis
        logits = self.weight * raw + markers.to(self.weight.dtype).view(
            batch, 1, 1, 1
        ) * vocabulary_axis
        mask = torch.stack(
            [self.masks_by_marker[int(value.item())] for value in markers],
            dim=0,
        ).to(device=codes.device)
        logits = torch.where(
            mask.unsqueeze(-1),
            logits,
            torch.full_like(logits, float("nan")),
        )
        if self.bad_position is not None:
            q_index, t_index = self.bad_position
            bad = torch.zeros_like(mask)
            bad[:, q_index, t_index] = True
            logits = logits.masked_fill(bad.unsqueeze(-1), float("inf"))

        self.calls.append(
            {
                "batch": batch,
                "codes": codes.detach().clone(),
                "markers": markers.detach().clone(),
                "condition_embedding": embedding.detach().clone(),
                "condition_mask": condition_mask.detach().clone(),
                "grad_enabled": torch.is_grad_enabled(),
                "training": self.training,
                "keep_only_valid_steps": keep_only_valid_steps,
            }
        )
        if self.truncate_time:
            logits = logits[:, :, :-1]
            mask = mask[:, :, :-1]
        return _FakeLMOutput(logits=logits, mask=mask)

    @staticmethod
    def assert_no_raw_conditions(conditions: list) -> None:
        if conditions:
            raise AssertionError("adapter must use precomputed condition tensors")


def _conditions(marker: int, batch: int, length: int) -> dict:
    embedding = torch.full((batch, length, 3), float(marker))
    mask = torch.ones(batch, length, dtype=torch.int64)
    if marker < 0:
        mask.zero_()
    return {"description": (embedding, mask)}


class AudioCraftAdapterTest(unittest.TestCase):
    def test_primary_adapter_rejects_one_raw_teacher_mask_mismatch(self) -> None:
        """The formal Q4/T500 path must never hide drift by intersection."""

        batch, codebooks, time, vocabulary = 1, 4, 500, 2048
        expected = torch.ones(codebooks, time, dtype=torch.bool)
        for codebook in range(codebooks):
            if codebook:
                expected[codebook, time - codebook :] = False
        wrong_conditional = expected.clone()
        wrong_conditional[0, 123] = False
        student = _FakeLM(codebooks, vocabulary, {2: expected})
        teacher = _FakeLM(
            codebooks,
            vocabulary,
            {2: wrong_conditional, -1: expected},
        )
        codes = torch.zeros(batch, codebooks, time, dtype=torch.long)
        with self.assertRaisesRegex(
            ValueError, "LM masks are not elementwise identical"
        ):
            score_audiocraft_trajectory(
                student,
                teacher,
                codes,
                _conditions(2, batch, length=2),
                _conditions(-1, batch, length=1),
                rollout_mask=torch.ones_like(codes, dtype=torch.bool),
                teacher_cfg_scale=3.0,
            )

    def test_batched_cfg_mask_intersection_and_grad_boundaries(self) -> None:
        batch, codebooks, time, vocabulary = 2, 2, 4, 3
        student_mask = torch.tensor(
            [[True, True, False, True], [True, True, True, True]]
        )
        teacher_cond_mask = torch.tensor(
            [[True, False, True, True], [True, True, True, False]]
        )
        teacher_null_mask = torch.tensor(
            [[True, True, True, False], [False, True, True, True]]
        )
        student = _FakeLM(
            codebooks, vocabulary, {2: student_mask}
        )
        teacher = _FakeLM(
            codebooks,
            vocabulary,
            {2: teacher_cond_mask, -1: teacher_null_mask},
        )
        # Deliberately start the teacher in training mode: the adapter must use
        # eval behavior for its forward and then restore the caller's state.
        teacher.train()
        codes = torch.arange(batch * codebooks * time).reshape(
            batch, codebooks, time
        )
        rollout_mask = torch.tensor(
            [
                [[True, True, True, True], [True, True, False, True]],
                [[True, True, True, False], [True, True, True, True]],
            ]
        )
        padding_mask = torch.tensor(
            [
                [[True, True, True, False], [True, True, True, True]],
                [[False, True, True, True], [True, False, True, True]],
            ]
        )
        conditional = _conditions(2, batch, length=3)
        null = _conditions(-1, batch, length=1)

        # The student graph must survive even if a caller accidentally wraps
        # the scoring call in no_grad.
        with torch.no_grad():
            output = score_audiocraft_trajectory(
                student,
                teacher,
                codes,
                conditional,
                null,
                rollout_mask=rollout_mask,
                padding_mask=padding_mask,
                teacher_cfg_scale=2.5,
            )

        expected_mask = (
            student_mask.unsqueeze(0).expand(batch, -1, -1)
            & teacher_cond_mask.unsqueeze(0).expand(batch, -1, -1)
            & teacher_null_mask.unsqueeze(0).expand(batch, -1, -1)
            & rollout_mask
            & padding_mask
        )
        self.assertTrue(torch.equal(output.valid_mask, expected_mask))
        self.assertEqual(output.teacher_cfg_logits.dtype, torch.float32)
        self.assertFalse(output.teacher_cfg_logits.requires_grad)
        self.assertTrue(output.student_logits.requires_grad)
        self.assertFalse(hasattr(output, "teacher_cond_logits"))
        self.assertFalse(hasattr(output, "teacher_null_logits"))

        vocabulary_axis = torch.arange(1, vocabulary + 1).view(1, 1, 1, -1)
        raw = codes.float().unsqueeze(-1) + vocabulary_axis
        conditional_logits = teacher.weight.detach() * raw + 2 * vocabulary_axis
        null_logits = teacher.weight.detach() * raw - vocabulary_axis
        expected_cfg = null_logits + 2.5 * (conditional_logits - null_logits)
        torch.testing.assert_close(
            output.teacher_cfg_logits[expected_mask], expected_cfg[expected_mask]
        )

        self.assertEqual(len(student.calls), 1)
        self.assertEqual(len(teacher.calls), 1)
        self.assertEqual(student.calls[0]["batch"], batch)
        self.assertEqual(teacher.calls[0]["batch"], 2 * batch)
        torch.testing.assert_close(
            teacher.calls[0]["markers"],
            torch.tensor([2.0, 2.0, -1.0, -1.0]),
        )
        self.assertTrue(student.calls[0]["grad_enabled"])
        self.assertFalse(teacher.calls[0]["grad_enabled"])
        self.assertFalse(teacher.calls[0]["training"])
        self.assertTrue(teacher.training)
        self.assertTrue(teacher.calls[0]["keep_only_valid_steps"])
        self.assertEqual(
            tuple(teacher.calls[0]["condition_embedding"].shape),
            (2 * batch, 3, 3),
        )
        # The separately computed null condition is zero-padded to the
        # conditional length before the one-pass CFG batch.
        torch.testing.assert_close(
            teacher.calls[0]["condition_embedding"][batch:, 1:],
            torch.zeros(batch, 2, 3),
        )
        self.assertTrue(
            torch.equal(
                teacher.calls[0]["condition_mask"][batch:],
                torch.zeros(batch, 3, dtype=torch.int64),
            )
        )

        output.student_logits[output.valid_mask].sum().backward()
        self.assertIsNotNone(student.weight.grad)
        self.assertIsNone(teacher.weight.grad)

    def test_separate_teacher_calls_match_batched_cfg(self) -> None:
        batch, codebooks, time, vocabulary = 1, 2, 3, 4
        all_valid = torch.ones(codebooks, time, dtype=torch.bool)
        codes = torch.arange(batch * codebooks * time).reshape(
            batch, codebooks, time
        )
        conditional = _conditions(2, batch, length=2)
        null = _conditions(-1, batch, length=1)
        rollout = torch.ones(batch, codebooks, time, dtype=torch.bool)

        student_batched = _FakeLM(codebooks, vocabulary, {2: all_valid})
        teacher_batched = _FakeLM(
            codebooks, vocabulary, {2: all_valid, -1: all_valid}
        )
        batched = score_audiocraft_trajectory(
            student_batched,
            teacher_batched,
            codes,
            conditional,
            null,
            rollout_mask=rollout,
            teacher_cfg_scale=3.0,
            teacher_forward_mode="batched",
        )

        student_separate = _FakeLM(codebooks, vocabulary, {2: all_valid})
        teacher_separate = _FakeLM(
            codebooks, vocabulary, {2: all_valid, -1: all_valid}
        )
        separate = score_audiocraft_trajectory(
            student_separate,
            teacher_separate,
            codes,
            conditional,
            null,
            rollout_mask=rollout,
            teacher_cfg_scale=3.0,
            teacher_forward_mode="separate",
        )

        torch.testing.assert_close(
            batched.teacher_cfg_logits, separate.teacher_cfg_logits
        )
        self.assertTrue(torch.equal(batched.valid_mask, separate.valid_mask))
        self.assertEqual(len(teacher_batched.calls), 1)
        self.assertEqual(len(teacher_separate.calls), 2)
        self.assertEqual(teacher_separate.calls[0]["batch"], batch)
        self.assertEqual(teacher_separate.calls[1]["batch"], batch)
        torch.testing.assert_close(
            teacher_separate.calls[0]["markers"], torch.tensor([2.0])
        )
        torch.testing.assert_close(
            teacher_separate.calls[1]["markers"], torch.tensor([-1.0])
        )

    def test_condition_batching_validates_keys_and_pads_time(self) -> None:
        conditional = _conditions(2, batch=1, length=3)
        null = _conditions(-1, batch=1, length=1)
        batched = batch_condition_tensors(
            conditional, null, expected_batch=1
        )
        embedding, mask = batched["description"]
        self.assertEqual(tuple(embedding.shape), (2, 3, 3))
        self.assertEqual(tuple(mask.shape), (2, 3))
        torch.testing.assert_close(embedding[1, 1:], torch.zeros(2, 3))

        with self.assertRaisesRegex(ValueError, "condition keys differ"):
            batch_condition_tensors(
                conditional, {"other": null["description"]}, expected_batch=1
            )

    def test_nonfinite_values_are_checked_only_after_mask_intersection(self) -> None:
        batch, codebooks, time = 1, 1, 2
        all_valid = torch.ones(codebooks, time, dtype=torch.bool)
        codes = torch.zeros(batch, codebooks, time, dtype=torch.long)
        conditional = _conditions(2, batch, length=1)
        null = _conditions(-1, batch, length=1)
        rollout = torch.ones(batch, codebooks, time, dtype=torch.bool)
        padding_excludes_bad = torch.tensor([[[False, True]]])

        student = _FakeLM(codebooks, 3, {2: all_valid})
        teacher = _FakeLM(
            codebooks,
            3,
            {2: all_valid, -1: all_valid},
            bad_position=(0, 0),
        )
        output = score_audiocraft_trajectory(
            student,
            teacher,
            codes,
            conditional,
            null,
            rollout_mask=rollout,
            padding_mask=padding_excludes_bad,
        )
        self.assertFalse(output.valid_mask[0, 0, 0].item())

        student = _FakeLM(codebooks, 3, {2: all_valid})
        teacher = _FakeLM(
            codebooks,
            3,
            {2: all_valid, -1: all_valid},
            bad_position=(0, 0),
        )
        with self.assertRaisesRegex(FloatingPointError, "valid positions"):
            score_audiocraft_trajectory(
                student,
                teacher,
                codes,
                conditional,
                null,
                rollout_mask=rollout,
                padding_mask=torch.ones_like(rollout),
            )

    def test_rejects_declared_q_and_output_time_mismatches(self) -> None:
        codes = torch.zeros(1, 2, 3, dtype=torch.long)
        all_valid = torch.ones(2, 3, dtype=torch.bool)
        conditional = _conditions(2, batch=1, length=1)
        null = _conditions(-1, batch=1, length=1)
        rollout = torch.ones_like(codes, dtype=torch.bool)

        wrong_q_student = _FakeLM(3, 4, {2: all_valid})
        teacher = _FakeLM(2, 4, {2: all_valid, -1: all_valid})
        with self.assertRaisesRegex(ValueError, "declares Q=3"):
            score_audiocraft_trajectory(
                wrong_q_student,
                teacher,
                codes,
                conditional,
                null,
                rollout_mask=rollout,
            )

        truncated_student = _FakeLM(
            2, 4, {2: all_valid}, truncate_time=True
        )
        with self.assertRaisesRegex(ValueError, "student_output.logits"):
            score_audiocraft_trajectory(
                truncated_student,
                teacher,
                codes,
                conditional,
                null,
                rollout_mask=rollout,
            )


if __name__ == "__main__":
    unittest.main()
