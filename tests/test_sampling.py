"""Tests for deterministic distributed data and selector sampling."""

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from ptc_opd.sampling import (
    DeterministicDistributedBatchSampler,
    keyed_random_scores,
)


class TestDistributedBatchSampler(unittest.TestCase):
    def _samplers(self, sample_ids=None):
        return [
            DeterministicDistributedBatchSampler(
                manifest_length=19,
                seed=2701,
                rank=rank,
                world_size=4,
                per_rank_batch=2,
                sample_ids=sample_ids,
            )
            for rank in range(4)
        ]

    def test_rank_batches_are_fixed_size_and_disjoint(self):
        samplers = self._samplers()
        self.assertTrue(all(s.microsteps_per_epoch == 2 for s in samplers))
        self.assertTrue(all(s.dropped_per_epoch == 3 for s in samplers))

        epoch_indices = []
        for global_microstep in range(2):
            assignments = [s.batch(global_microstep) for s in samplers]
            gathered = [index for batch in assignments for index in batch.indices]
            self.assertEqual(len(gathered), 8)
            self.assertEqual(len(set(gathered)), 8)
            self.assertTrue(all(len(batch.indices) == 2 for batch in assignments))
            epoch_indices.extend(gathered)

        self.assertEqual(len(epoch_indices), 16)
        self.assertEqual(len(set(epoch_indices)), 16)
        dropped_indices = set(range(19)) - set(epoch_indices)
        self.assertEqual(len(dropped_indices), 3)

    def test_global_microstep_derives_epoch_and_drops_tail(self):
        sampler = self._samplers()[0]
        last_epoch_zero = sampler.batch(1)
        first_epoch_one = sampler.batch(2)
        self.assertEqual(
            (last_epoch_zero.epoch, last_epoch_zero.microstep_in_epoch),
            (0, 1),
        )
        self.assertEqual(
            (first_epoch_one.epoch, first_epoch_one.microstep_in_epoch),
            (1, 0),
        )

        recreated = self._samplers()[0]
        self.assertEqual(first_epoch_one, recreated.batch(2))
        epoch_one = list(sampler.iter_epoch(1))
        self.assertEqual(epoch_one, [sampler.batch(2), sampler.batch(3)])

    def test_seed_changes_assignment_but_recreation_is_exact(self):
        baseline = self._samplers()[0].batch(0)
        recreated = self._samplers()[0].batch(0)
        changed_seed = DeterministicDistributedBatchSampler(
            manifest_length=19,
            seed=2702,
            rank=0,
            world_size=4,
            per_rank_batch=2,
        ).batch(0)
        self.assertEqual(baseline, recreated)
        self.assertNotEqual(baseline.indices, changed_seed.indices)

    def test_stable_sample_ids_follow_manifest_indices(self):
        ids = ["yt-{:02d}@10".format(index) for index in range(19)]
        for sampler in self._samplers(sample_ids=ids):
            assignment = sampler.batch(0)
            expected_ids = tuple(ids[index] for index in assignment.indices)
            self.assertEqual(assignment.sample_ids, expected_ids)

    def test_default_ids_are_manifest_indices(self):
        assignment = self._samplers()[2].batch(0)
        self.assertEqual(assignment.sample_ids, assignment.indices)

    def test_rejects_invalid_or_ambiguous_configuration(self):
        with self.assertRaisesRegex(ValueError, "full global batch"):
            DeterministicDistributedBatchSampler(7, 1, 0, 4, 2)
        with self.assertRaisesRegex(ValueError, "rank must be smaller"):
            DeterministicDistributedBatchSampler(8, 1, 4, 4, 2)
        with self.assertRaisesRegex(ValueError, "length must equal"):
            DeterministicDistributedBatchSampler(8, 1, 0, 4, 2, ["x"])
        with self.assertRaisesRegex(ValueError, "must be unique"):
            DeterministicDistributedBatchSampler(
                8, 1, 0, 4, 2, ["same"] * 8
            )
        with self.assertRaises(TypeError):
            DeterministicDistributedBatchSampler(True, 1, 0, 1, 1)


class TestKeyedRandomScores(unittest.TestCase):
    def test_repeatability_and_batch_order_invariance(self):
        ids = ["sample-c", "sample-a", "sample-b"]
        first = keyed_random_scores(ids, 4, 11, 17, 5)
        repeated = keyed_random_scores(ids, 4, 11, 17, 5)
        reordered_ids = ["sample-a", "sample-b", "sample-c"]
        reordered = keyed_random_scores(reordered_ids, 4, 11, 17, 5)

        torch.testing.assert_close(first, repeated, rtol=0.0, atol=0.0)
        by_id = {sample_id: first[index] for index, sample_id in enumerate(ids)}
        for index, sample_id in enumerate(reordered_ids):
            torch.testing.assert_close(
                reordered[index], by_id[sample_id], rtol=0.0, atol=0.0
            )

    def test_rank_partition_matches_combined_batch(self):
        rank_zero_ids = ["a", "b"]
        rank_one_ids = ["c", "d"]
        combined = keyed_random_scores(
            rank_zero_ids + rank_one_ids, 3, 7, 100, 9
        )
        rank_zero = keyed_random_scores(rank_zero_ids, 3, 7, 100, 9)
        rank_one = keyed_random_scores(rank_one_ids, 3, 7, 100, 9)
        torch.testing.assert_close(combined[:2], rank_zero, rtol=0.0, atol=0.0)
        torch.testing.assert_close(combined[2:], rank_one, rtol=0.0, atol=0.0)

    def test_seed_and_step_change_scores(self):
        base = keyed_random_scores(["a"], 4, 32, 10, 7)
        changed_seed = keyed_random_scores(["a"], 4, 32, 11, 7)
        changed_step = keyed_random_scores(["a"], 4, 32, 10, 8)
        self.assertFalse(torch.equal(base, changed_seed))
        self.assertFalse(torch.equal(base, changed_step))

    def test_recorded_random_namespace_changes_scores(self):
        base = keyed_random_scores(["a"], 4, 32, 10, 7)
        changed = keyed_random_scores(
            ["a"], 4, 32, 10, 7, random_namespace=5702
        )
        self.assertFalse(torch.equal(base, changed))

    def test_codebooks_use_domain_separated_streams(self):
        scores = keyed_random_scores(["a"], 4, 32, 10, 7)
        for left in range(4):
            for right in range(left + 1, 4):
                self.assertFalse(torch.equal(scores[0, left], scores[0, right]))

    def test_does_not_mutate_global_rng_state(self):
        torch.manual_seed(123)
        state_before = torch.random.get_rng_state().clone()
        keyed_random_scores(["a", "b"], 2, 5, 10, 7)
        state_after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(state_before, state_after))

    def test_shape_dtype_empty_batch_and_id_type_separation(self):
        scores = keyed_random_scores(
            [1, "1"], 2, 3, 0, 0, device="cpu", dtype=torch.float64
        )
        self.assertEqual(scores.shape, (2, 2, 3))
        self.assertEqual(scores.dtype, torch.float64)
        self.assertFalse(torch.equal(scores[0], scores[1]))

        empty = keyed_random_scores(
            [], 2, 3, 0, 0, device=torch.device("cpu"), dtype=torch.float32
        )
        self.assertEqual(empty.shape, (0, 2, 3))
        self.assertEqual(empty.dtype, torch.float32)

    def test_rejects_invalid_arguments(self):
        with self.assertRaises(ValueError):
            keyed_random_scores(["a"], 0, 3, 0, 0)
        with self.assertRaises(TypeError):
            keyed_random_scores(["a"], 2, 3, 0, 0, dtype=torch.int64)
        with self.assertRaises(TypeError):
            keyed_random_scores([object()], 2, 3, 0, 0)
        with self.assertRaises(ValueError):
            keyed_random_scores([""], 2, 3, 0, 0)


if __name__ == "__main__":
    unittest.main()
