from __future__ import annotations

import unittest

import torch

from ptc_opd.perceptual_prior import (
    mrstft_distance,
    mrstft_reference_norms,
    normalize_marginals,
    progressive_prior,
)


class PerceptualPriorTest(unittest.TestCase):
    def test_identity_is_zero(self) -> None:
        generator = torch.Generator().manual_seed(7)
        audio = torch.randn(2, 1, 256, generator=generator)
        distance = mrstft_distance(audio, audio, fft_sizes=(32, 64))
        torch.testing.assert_close(distance, torch.zeros_like(distance), atol=1e-7, rtol=0)

    def test_progressive_improvement_and_prior(self) -> None:
        time = torch.linspace(0, 1, 512)
        reference = torch.sin(2 * torch.pi * 7 * time).view(1, 1, -1)
        silence = torch.zeros_like(reference)
        coarse = 0.5 * reference
        exact = reference
        levels = torch.stack([silence, coarse, exact], dim=1)
        output = progressive_prior(reference, levels, fft_sizes=(32, 64))
        self.assertEqual(tuple(output.distances.shape), (1, 3))
        self.assertEqual(tuple(output.marginals.shape), (1, 2))
        self.assertTrue(bool((output.marginals > 0).all().item()))
        torch.testing.assert_close(output.prior.sum(), torch.tensor(1.0))

    def test_negative_marginal_is_reported_then_floored(self) -> None:
        mean = torch.tensor([2.0, -1.0, 0.0])
        clipped, prior = normalize_marginals(mean, epsilon=0.1)
        torch.testing.assert_close(clipped, torch.tensor([2.0, 0.1, 0.1]))
        torch.testing.assert_close(prior, clipped / clipped.sum())

    def test_silent_or_near_floor_reference_fails_closed_without_clamp(self) -> None:
        silence = torch.zeros(1, 1, 256)
        reconstruction = torch.ones_like(silence).mul_(1.0e-4)
        norms = mrstft_reference_norms(silence, fft_sizes=(32, 64))
        torch.testing.assert_close(norms, torch.zeros_like(norms))
        with self.assertRaisesRegex(FloatingPointError, "strictly greater"):
            mrstft_distance(
                silence,
                reconstruction,
                fft_sizes=(32, 64),
                epsilon=1.0e-7,
            )

    def test_validation(self) -> None:
        audio = torch.zeros(1, 128)
        with self.assertRaises(ValueError):
            mrstft_distance(audio, torch.zeros(2, 128), fft_sizes=(32,))
        with self.assertRaises(FloatingPointError):
            mrstft_distance(audio.fill_(float("nan")), audio, fft_sizes=(32,))
        with self.assertRaises(ValueError):
            normalize_marginals(torch.empty(0))


if __name__ == "__main__":
    unittest.main()
