"""CPU/fake tests for the strict primary MusicGen tensor contract."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from ptc_opd.musicgen_contract import (
    PRIMARY_FRAMES,
    REQUIRED_SCORE_NAMES,
    expected_musicgen_delay_mask,
    strict_validate_musicgen_batch,
    strict_validate_musicgen_metadata,
    strict_validate_musicgen_model,
)


def _valid_batch(batch: int = 1, card: int = 2048):
    codes = torch.arange(batch * 4 * PRIMARY_FRAMES, dtype=torch.long)
    codes = codes.reshape(batch, 4, PRIMARY_FRAMES) % card
    expected = expected_musicgen_delay_mask(batch)
    masks = {name: expected.clone() for name in REQUIRED_SCORE_NAMES}
    logits = {
        name: torch.zeros(batch, 4, PRIMARY_FRAMES, card)
        for name in REQUIRED_SCORE_NAMES
    }
    return codes, masks, logits


class StrictMusicGenBatchTest(unittest.TestCase):
    def test_expected_delay_mask_has_exact_prefixes_and_counts(self) -> None:
        mask = expected_musicgen_delay_mask(2)
        self.assertEqual(tuple(mask.shape), (2, 4, 500))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask[0].sum(dim=-1).tolist(), [500, 499, 498, 497])
        for codebook, tail in enumerate((0, 1, 2, 3)):
            valid = 500 - tail
            self.assertTrue(mask[:, codebook, :valid].all())
            if tail:
                self.assertFalse(mask[:, codebook, valid:].any())

    def test_valid_codes_masks_and_bqtv_logits_return_shared_mask(self) -> None:
        codes, masks, logits = _valid_batch()
        result = strict_validate_musicgen_batch(codes, 2048, masks, logits)
        self.assertTrue(torch.equal(result, masks["student"]))
        self.assertEqual(result.dtype, torch.bool)
        self.assertFalse(result.data_ptr() == masks["student"].data_ptr())

    def test_rejects_lm_special_token_card_and_negative_code(self) -> None:
        codes, masks, logits = _valid_batch(batch=1)
        codes[0, 0, 0] = 2048
        with self.assertRaisesRegex(ValueError, "special token ID card"):
            strict_validate_musicgen_batch(codes, 2048, masks, logits)
        codes[0, 0, 0] = -1
        with self.assertRaisesRegex(ValueError, "below zero"):
            strict_validate_musicgen_batch(codes, 2048, masks, logits)

    def test_rejects_code_dtype_rank_q_t_and_empty_batch(self) -> None:
        _, masks, _ = _valid_batch(batch=1)
        cases = (
            (torch.zeros(1, 4, 500), TypeError, "torch.long"),
            (torch.zeros(4, 500, dtype=torch.long), ValueError, "rank 2"),
            (torch.zeros(1, 3, 500, dtype=torch.long), ValueError, r"\[B,4,500\]"),
            (torch.zeros(1, 4, 499, dtype=torch.long), ValueError, r"\[B,4,500\]"),
            (torch.zeros(0, 4, 500, dtype=torch.long), ValueError, "batch"),
        )
        for codes, error, message in cases:
            with self.subTest(shape=tuple(codes.shape)), self.assertRaisesRegex(error, message):
                strict_validate_musicgen_batch(codes, 2048, masks)

    def test_requires_exact_student_teacher_names(self) -> None:
        codes, masks, _ = _valid_batch(batch=1)
        missing = dict(masks)
        missing.pop("teacher_null")
        with self.assertRaisesRegex(ValueError, "missing=.*teacher_null"):
            strict_validate_musicgen_batch(codes, 2048, missing)
        extra = dict(masks)
        extra["other"] = masks["student"]
        with self.assertRaisesRegex(ValueError, "unexpected=.*other"):
            strict_validate_musicgen_batch(codes, 2048, extra)

    def test_rejects_mask_dtype_shape_disagreement_and_wrong_delay(self) -> None:
        codes, masks, _ = _valid_batch(batch=1)
        bad_dtype = dict(masks)
        bad_dtype["student"] = bad_dtype["student"].long()
        with self.assertRaisesRegex(TypeError, "student.*torch.bool"):
            strict_validate_musicgen_batch(codes, 2048, bad_dtype)

        bad_shape = dict(masks)
        bad_shape["student"] = bad_shape["student"][:, :, :-1]
        with self.assertRaisesRegex(ValueError, "student.*shape"):
            strict_validate_musicgen_batch(codes, 2048, bad_shape)

        disagreement = {name: value.clone() for name, value in masks.items()}
        disagreement["teacher_null"][0, 0, 0] = False
        with self.assertRaisesRegex(ValueError, "not elementwise identical"):
            strict_validate_musicgen_batch(codes, 2048, disagreement)

        wrong_all = {name: torch.ones_like(value) for name, value in masks.items()}
        with self.assertRaisesRegex(ValueError, "violates frozen delay"):
            strict_validate_musicgen_batch(codes, 2048, wrong_all)

    def test_rejects_logits_name_shape_vocab_and_dtype(self) -> None:
        codes, masks, logits = _valid_batch(batch=1)
        missing = dict(logits)
        missing.pop("teacher_cond")
        with self.assertRaisesRegex(ValueError, "named_logits.*missing"):
            strict_validate_musicgen_batch(codes, 2048, masks, missing)

        wrong_vocabulary = dict(logits)
        wrong_vocabulary["teacher_cond"] = torch.zeros(1, 4, 500, 2047)
        with self.assertRaisesRegex(ValueError, "BQTV shape"):
            strict_validate_musicgen_batch(codes, 2048, masks, wrong_vocabulary)

        wrong_layout = dict(logits)
        wrong_layout["student"] = torch.zeros(1, 500, 4, 2048)
        with self.assertRaisesRegex(ValueError, "BQTV shape"):
            strict_validate_musicgen_batch(codes, 2048, masks, wrong_layout)

        wrong_dtype = dict(logits)
        wrong_dtype["teacher_null"] = torch.zeros(1, 4, 500, 2048, dtype=torch.long)
        with self.assertRaisesRegex(TypeError, "floating-point"):
            strict_validate_musicgen_batch(codes, 2048, masks, wrong_dtype)

    def test_rejects_bad_card(self) -> None:
        codes, masks, _ = _valid_batch(batch=1)
        for card in (True, 0, -1, 1.5, 1024):
            with self.subTest(card=card), self.assertRaises((TypeError, ValueError)):
                strict_validate_musicgen_batch(codes, card, masks)


class _FakePattern:
    def __init__(self, mask: torch.Tensor) -> None:
        self.mask = mask
        self.build_calls = 0
        self.revert_calls = 0

    def build_pattern_sequence(
        self, codes, special_token, keep_only_valid_steps
    ):
        self.build_calls += 1
        if special_token <= 0 or not keep_only_valid_steps:
            raise AssertionError("unexpected pattern build arguments")
        sequence = torch.zeros(codes.shape[0], codes.shape[1], 504, dtype=torch.long)
        return sequence, torch.zeros(4, 504, dtype=torch.long), torch.ones(4, 504, dtype=torch.bool)

    def revert_pattern_logits(
        self, logits, special_token, keep_only_valid_steps
    ):
        self.revert_calls += 1
        if logits.shape != (1, 1, 4, 504) or not keep_only_valid_steps:
            raise AssertionError("unexpected pattern revert arguments")
        return torch.zeros(1, 1, 4, 500), torch.zeros(4, 500, dtype=torch.long), self.mask


class _FakeProvider:
    def __init__(self, mask: torch.Tensor) -> None:
        self.pattern = _FakePattern(mask)
        self.timesteps = []

    def get_pattern(self, timesteps: int):
        self.timesteps.append(timesteps)
        return self.pattern


class StrictMusicGenMetadataTest(unittest.TestCase):
    def test_valid_metadata_without_pattern(self) -> None:
        output = strict_validate_musicgen_metadata(
            num_codebooks=4, frame_rate=50, card=2048
        )
        self.assertEqual(output.num_codebooks, 4)
        self.assertEqual(output.frame_rate, 50.0)
        self.assertEqual(output.card, 2048)
        self.assertFalse(output.pattern_checked)

    def test_public_pattern_methods_are_checked_against_frozen_mask(self) -> None:
        provider = _FakeProvider(expected_musicgen_delay_mask(1)[0])
        output = strict_validate_musicgen_metadata(
            num_codebooks=4,
            frame_rate=50.0,
            card=2048,
            pattern_provider=provider,
        )
        self.assertTrue(output.pattern_checked)
        self.assertEqual(provider.timesteps, [500])
        self.assertEqual(provider.pattern.build_calls, 1)
        self.assertEqual(provider.pattern.revert_calls, 1)

    def test_wrong_pattern_counts_fail(self) -> None:
        wrong = expected_musicgen_delay_mask(1)[0]
        wrong[3, -1] = True
        provider = _FakeProvider(wrong)
        with self.assertRaisesRegex(ValueError, "pattern_provider is not frozen delay"):
            strict_validate_musicgen_metadata(
                num_codebooks=4,
                frame_rate=50,
                card=2048,
                pattern_provider=provider,
            )

    def test_bad_metadata_and_pattern_interface_fail(self) -> None:
        cases = (
            ({"num_codebooks": 3, "frame_rate": 50, "card": 2048}, ValueError, "num_codebooks=4"),
            ({"num_codebooks": 4, "frame_rate": 25, "card": 2048}, ValueError, "frame_rate=50"),
            ({"num_codebooks": 4, "frame_rate": float("nan"), "card": 2048}, ValueError, "finite"),
            ({"num_codebooks": 4, "frame_rate": 50, "card": 0}, ValueError, "positive"),
            ({"num_codebooks": True, "frame_rate": 50, "card": 2048}, TypeError, "integer"),
            ({"num_codebooks": 4, "frame_rate": 50, "card": 1024}, ValueError, "card=2048"),
        )
        for kwargs, error, message in cases:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(error, message):
                strict_validate_musicgen_metadata(**kwargs)
        with self.assertRaisesRegex(TypeError, "get_pattern"):
            strict_validate_musicgen_metadata(
                num_codebooks=4, frame_rate=50, card=2048, pattern_provider=object()
            )

    def test_model_convenience_gate_reads_public_attributes(self) -> None:
        class Model:
            num_codebooks = 4
            card = 2048
            pattern_provider = _FakeProvider(expected_musicgen_delay_mask(1)[0])

        output = strict_validate_musicgen_model(Model(), frame_rate=50)
        self.assertEqual(output.card, 2048)
        self.assertTrue(output.pattern_checked)

        with self.assertRaisesRegex(TypeError, "num_codebooks and card"):
            strict_validate_musicgen_model(object(), frame_rate=50)


if __name__ == "__main__":
    unittest.main()
