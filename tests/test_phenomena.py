from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from ptc_opd.phenomena import (
    CELL_SCHEMA_VERSION,
    PhenomenonAccumulator,
    audit_topk_records,
    cell_records_from_logits,
    prompt_sha256,
    summarize_cell_records,
)


def _cell(
    sample_id: str,
    seed: int,
    q_index: int,
    time_index: int,
    *,
    js: float,
    kl: float,
    selected: bool,
    a_q: float,
) -> dict:
    return {
        "schema_version": CELL_SCHEMA_VERSION,
        "sample_id": sample_id,
        "prompt_sha256": prompt_sha256("prompt " + sample_id),
        "rollout_seed": seed,
        "q": q_index,
        "t": time_index,
        "temporal_decile": time_index,
        "js": js,
        "forward_kl": kl,
        "teacher_entropy": 2.0 + q_index,
        "student_entropy": 1.5 + q_index,
        "sampled_token_logp_teacher": -0.4 - time_index,
        "sampled_token_logp_student": -0.5 - time_index,
        "a_q": a_q,
        "top50_js": selected,
    }


class PhenomenonCellTest(unittest.TestCase):
    def setUp(self) -> None:
        self.student = torch.tensor(
            [
                [
                    [[2.0, 0.0, -1.0], [0.5, 1.0, -0.5], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    [[0.0, 0.5, 1.0], [1.5, 0.0, 0.0], [0.0, -1.0, 2.0], [0.2, 0.1, 0.0]],
                ],
                [
                    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, -1.0]],
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [2.0, -1.0, 0.0], [0.0, 0.0, 0.0]],
                ],
            ],
            requires_grad=True,
        )
        offset = torch.tensor([0.4, -0.2, 0.1]).view(1, 1, 1, 3)
        self.teacher = self.student.detach() + offset
        self.codes = torch.tensor(
            [
                [[0, 1, 2, 1], [2, 0, 2, 1]],
                [[1, 2, 0, 0], [0, 1, 0, 2]],
            ],
            dtype=torch.long,
        )
        self.mask = torch.tensor(
            [
                [[True, True, True, False], [True, True, False, True]],
                [[True, True, True, True], [True, False, True, True]],
            ]
        )

    def test_scalar_records_use_shared_loss_gate(self) -> None:
        records = list(
            cell_records_from_logits(
                self.student,
                self.teacher,
                self.codes,
                self.mask,
                sample_ids=["a", "b"],
                prompts=["piano", "drums"],
                rollout_seeds=[31001, 31001],
                codebook_weights=torch.tensor([3.0, 1.0]),
            )
        )
        self.assertEqual(len(records), int(self.mask.sum().item()))
        self.assertEqual(records[0]["prompt_sha256"], prompt_sha256("piano"))
        self.assertNotIn("student_logits", records[0])
        self.assertNotIn("teacher_logits", records[0])
        self.assertAlmostEqual(records[0]["a_q"], 0.75)
        self.assertTrue(all(record["schema_version"] == CELL_SCHEMA_VERSION for record in records))

        grouped = {}
        for record in records:
            grouped.setdefault((record["sample_id"], record["q"]), []).append(record)
        for group in grouped.values():
            self.assertEqual(
                sum(record["top50_js"] for record in group),
                math.ceil(len(group) / 2),
            )

        expected_logprob = torch.log_softmax(self.student[0, 0, 0], dim=-1)[0]
        self.assertAlmostEqual(
            records[0]["sampled_token_logp_student"], float(expected_logprob.item()), places=6
        )
        expected_teacher_logp = torch.log_softmax(self.teacher[0, 0, 0], dim=-1)[0]
        self.assertAlmostEqual(
            records[0]["sampled_token_logp_teacher"],
            float(expected_teacher_logp.item()),
            places=6,
        )
        self.assertGreaterEqual(records[0]["js"], 0.0)

    def test_ties_are_stable_by_time(self) -> None:
        logits = torch.zeros(1, 1, 5, 3, requires_grad=True)
        records = list(
            cell_records_from_logits(
                logits,
                logits.detach().clone(),
                torch.zeros(1, 1, 5, dtype=torch.long),
                torch.ones(1, 1, 5, dtype=torch.bool),
                sample_ids=["tie"],
                prompts=["tie prompt"],
                rollout_seeds=[31001],
                codebook_weights=torch.ones(1),
            )
        )
        self.assertEqual(
            [record["t"] for record in records if record["top50_js"]],
            [0, 1, 2],
        )

    def test_optional_topk_audit_is_bounded(self) -> None:
        records = list(
            audit_topk_records(
                self.student[:1],
                self.teacher[:1],
                self.codes[:1],
                self.mask[:1],
                sample_ids=["a"],
                prompts=["piano"],
                rollout_seeds=[31001],
                top_k=2,
            )
        )
        self.assertEqual(len(records), int(self.mask[:1].sum().item()))
        self.assertEqual(len(records[0]["student_top_token_ids"]), 2)
        self.assertEqual(len(records[0]["teacher_top_logits"]), 2)
        self.assertNotIn("prompt", records[0])

    def test_invalid_sampled_token_is_rejected_only_when_valid(self) -> None:
        codes = self.codes.clone()
        codes[0, 0, 0] = 99
        with self.assertRaises(ValueError):
            list(
                cell_records_from_logits(
                    self.student,
                    self.teacher,
                    codes,
                    self.mask,
                    sample_ids=["a", "b"],
                    prompts=["piano", "drums"],
                    rollout_seeds=[31001, 31001],
                    codebook_weights=torch.tensor([3.0, 1.0]),
                )
            )
        codes = self.codes.clone()
        codes[0, 0, 3] = 99
        list(
            cell_records_from_logits(
                self.student,
                self.teacher,
                codes,
                self.mask,
                sample_ids=["a", "b"],
                prompts=["piano", "drums"],
                rollout_seeds=[31001, 31001],
                codebook_weights=torch.tensor([3.0, 1.0]),
            )
        )


class PhenomenonSummaryTest(unittest.TestCase):
    def _records(self) -> list:
        records = []
        # Sequence blocks are deliberately prompt/seed ordered.  Each q has two
        # cells and exactly one stable top-50 cell.
        for sample_id, base in (("a", 1.0), ("b", 9.0)):
            for seed in (31001, 31002):
                records.extend(
                    [
                        _cell(sample_id, seed, 0, 0, js=2.0, kl=base, selected=True, a_q=0.75),
                        _cell(sample_id, seed, 0, 5, js=1.0, kl=base, selected=False, a_q=0.75),
                        _cell(sample_id, seed, 1, 0, js=4.0, kl=3 * base, selected=True, a_q=0.25),
                        _cell(sample_id, seed, 1, 5, js=2.0, kl=3 * base, selected=False, a_q=0.25),
                    ]
                )
        return records

    def test_exact_allocation_mass_and_tv(self) -> None:
        summary = summarize_cell_records(self._records(), bootstrap_replicates=0)
        self.assertEqual(summary.prompt_count, 2)
        self.assertEqual(summary.sequence_count, 4)
        self.assertEqual(summary.cell_count, 16)
        self.assertEqual(summary.rollout_seeds, (31001, 31002))
        codebook_rows = {
            row["q"]: row for row in summary.rows if row["group_type"] == "codebook"
        }
        self.assertAlmostEqual(codebook_rows[0]["nominal_u"], 0.5)
        self.assertAlmostEqual(codebook_rows[0]["weighted_w_a"], 0.75)
        self.assertAlmostEqual(codebook_rows[0]["kl_mass_k"], 0.25)
        self.assertAlmostEqual(codebook_rows[1]["kl_mass_k"], 0.75)
        self.assertAlmostEqual(codebook_rows[0]["c50_js"], 2.0 / 3.0)
        self.assertAlmostEqual(codebook_rows[0]["c50_kl"], 0.5)
        self.assertAlmostEqual(summary.tv["value"], 0.25)
        joint_rows = [row for row in summary.rows if row["group_type"] == "joint"]
        self.assertEqual(len(joint_rows), 4)
        js_decile_rows = [
            row for row in summary.rows if row["group_type"] == "within_q_js_decile"
        ]
        self.assertEqual(
            {(row["q"], row["within_q_js_decile"]) for row in js_decile_rows},
            {(0, 0), (0, 5), (1, 0), (1, 5)},
        )

    def test_bootstrap_is_deterministic_and_prompt_nested(self) -> None:
        first = summarize_cell_records(
            self._records(),
            bootstrap_replicates=128,
            bootstrap_seed=4702,
            retain_bootstrap_arrays=True,
        )
        second = summarize_cell_records(
            self._records(),
            bootstrap_replicates=128,
            bootstrap_seed=4702,
            retain_bootstrap_arrays=True,
        )
        assert first.bootstrap_arrays is not None
        assert second.bootstrap_arrays is not None
        torch.testing.assert_close(
            first.bootstrap_arrays["mean_forward_kl"],
            second.bootstrap_arrays["mean_forward_kl"],
        )
        # Overall is the first row.  Prompt-level resampling of prompt means 1
        # and 9 can only produce 1, 5, or 9 when drawing two prompts.
        observed = set(first.bootstrap_arrays["mean_forward_kl"][:, 0].tolist())
        self.assertTrue(observed.issubset({2.0, 10.0, 18.0}))
        # Overall cell mean also includes q1=3*q0, hence values are 2, 10, 18.

    def test_bad_top50_and_noncontiguous_blocks_are_rejected(self) -> None:
        records = self._records()
        records[0] = dict(records[0], top50_js=False)
        with self.assertRaises(ValueError):
            summarize_cell_records(records, bootstrap_replicates=0)

        accumulator = PhenomenonAccumulator()
        one = _cell("a", 31001, 0, 0, js=1.0, kl=1.0, selected=True, a_q=1.0)
        two = _cell("b", 31001, 0, 0, js=1.0, kl=1.0, selected=True, a_q=1.0)
        accumulator.consume(one)
        accumulator.consume(two)
        with self.assertRaises(ValueError):
            accumulator.consume(one)

    def test_mismatched_rollout_sets_are_rejected(self) -> None:
        records = self._records()
        records = [
            record
            for record in records
            if not (record["sample_id"] == "b" and record["rollout_seed"] == 31002)
        ]
        with self.assertRaises(ValueError):
            summarize_cell_records(records, bootstrap_replicates=0)


if __name__ == "__main__":
    unittest.main()
