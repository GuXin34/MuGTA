"""Perceptual codebook-prior primitives used by the Stage-1 assay."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
from torch import Tensor


DEFAULT_FFT_SIZES: Tuple[int, ...] = (512, 1024, 2048)


@dataclass(frozen=True)
class ProgressivePriorOutput:
    """Per-example progressive distances and marginal improvements.

    ``distances`` has shape ``[B, Q+1]`` and includes the silence baseline at
    index zero. ``marginals`` has shape ``[B, Q]`` with
    ``distance[q] - distance[q+1]``. ``prior`` is the positive, sum-one vector
    derived from the batch-mean marginals.
    """

    distances: Tensor
    marginals: Tensor
    mean_marginals: Tensor
    clipped_marginals: Tensor
    prior: Tensor


def _as_bcs(waveform: Tensor, name: str) -> Tensor:
    if not waveform.is_floating_point():
        raise TypeError(f"{name} must be a floating tensor")
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(1)
    if waveform.ndim != 3:
        raise ValueError(f"{name} must have shape [B,S] or [B,C,S]")
    if min(waveform.shape) <= 0:
        raise ValueError(f"{name} dimensions must be non-zero")
    return waveform


def _validate_fft_sizes(fft_sizes: Sequence[int]) -> Tuple[int, ...]:
    values = tuple(int(value) for value in fft_sizes)
    if not values or any(value < 2 for value in values):
        raise ValueError("fft_sizes must contain positive integers >= 2")
    if len(set(values)) != len(values):
        raise ValueError("fft_sizes must not contain duplicates")
    return values


def mrstft_distance(
    reference: Tensor,
    reconstruction: Tensor,
    *,
    fft_sizes: Sequence[int] = DEFAULT_FFT_SIZES,
    hop_ratio: float = 0.25,
    epsilon: float = 1.0e-7,
) -> Tensor:
    """Return fixed multi-resolution STFT distance for every example.

    The output has shape ``[B]``. At each resolution, spectral convergence and
    log-magnitude L1 are computed per channel, then averaged over channels and
    resolutions. Calculations are float32, including when input audio is BF16.
    """

    reference_bcs = _as_bcs(reference, "reference")
    reconstruction_bcs = _as_bcs(reconstruction, "reconstruction")
    if reference_bcs.shape != reconstruction_bcs.shape:
        raise ValueError("reference and reconstruction must have equal shapes")
    if reference_bcs.device != reconstruction_bcs.device:
        raise ValueError("reference and reconstruction must share a device")
    if not math.isfinite(float(hop_ratio)) or not 0.0 < hop_ratio <= 1.0:
        raise ValueError("hop_ratio must be finite and lie in (0,1]")
    if not math.isfinite(float(epsilon)) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    sizes = _validate_fft_sizes(fft_sizes)

    reference_flat = reference_bcs.float().reshape(-1, reference_bcs.shape[-1])
    reconstruction_flat = reconstruction_bcs.float().reshape(
        -1, reconstruction_bcs.shape[-1]
    )
    if not bool(torch.isfinite(reference_flat).all().item()):
        raise FloatingPointError("reference contains NaN/Inf")
    if not bool(torch.isfinite(reconstruction_flat).all().item()):
        raise FloatingPointError("reconstruction contains NaN/Inf")

    per_resolution = []
    for n_fft in sizes:
        hop_length = max(1, int(round(n_fft * hop_ratio)))
        window = torch.hann_window(
            n_fft, device=reference_bcs.device, dtype=torch.float32
        )
        reference_stft = torch.stft(
            reference_flat,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            pad_mode="constant",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        reconstruction_stft = torch.stft(
            reconstruction_flat,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            pad_mode="constant",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        reference_magnitude = reference_stft.abs()
        reconstruction_magnitude = reconstruction_stft.abs()
        difference = reference_magnitude - reconstruction_magnitude
        reference_norm = torch.linalg.vector_norm(
            reference_magnitude, dim=(-2, -1)
        )
        if bool((reference_norm <= float(epsilon)).any().item()):
            minimum = float(reference_norm.min().item())
            raise FloatingPointError(
                "MR-STFT reference norm must be strictly greater than "
                f"{float(epsilon):.17g}; observed minimum {minimum:.17g} "
                f"at n_fft={n_fft}"
            )
        spectral_convergence = torch.linalg.vector_norm(
            difference, dim=(-2, -1)
        ) / reference_norm
        log_magnitude_l1 = (
            torch.log(reference_magnitude + float(epsilon))
            - torch.log(reconstruction_magnitude + float(epsilon))
        ).abs().mean(dim=(-2, -1))
        channel_distance = spectral_convergence + log_magnitude_l1
        per_example = channel_distance.reshape(
            reference_bcs.shape[0], reference_bcs.shape[1]
        ).mean(dim=1)
        per_resolution.append(per_example)

    distance = torch.stack(per_resolution, dim=0).mean(dim=0)
    if not bool(torch.isfinite(distance).all().item()):
        raise FloatingPointError("MR-STFT distance produced NaN/Inf")
    return distance


def mrstft_reference_norms(
    reference: Tensor,
    *,
    fft_sizes: Sequence[int] = DEFAULT_FFT_SIZES,
    hop_ratio: float = 0.25,
) -> Tensor:
    """Return per-example/channel Frobenius norms for every FFT resolution.

    The returned tensor has shape ``[B, C, R]``.  This audit helper uses the
    exact window, hop, padding, and float32 arithmetic of :func:`mrstft_distance`.
    It does not apply a floor or decide eligibility.
    """

    reference_bcs = _as_bcs(reference, "reference")
    if not math.isfinite(float(hop_ratio)) or not 0.0 < hop_ratio <= 1.0:
        raise ValueError("hop_ratio must be finite and lie in (0,1]")
    sizes = _validate_fft_sizes(fft_sizes)
    reference_flat = reference_bcs.float().reshape(-1, reference_bcs.shape[-1])
    if not bool(torch.isfinite(reference_flat).all().item()):
        raise FloatingPointError("reference contains NaN/Inf")
    norms = []
    for n_fft in sizes:
        hop_length = max(1, int(round(n_fft * hop_ratio)))
        window = torch.hann_window(
            n_fft, device=reference_bcs.device, dtype=torch.float32
        )
        reference_stft = torch.stft(
            reference_flat,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            pad_mode="constant",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        norms.append(
            torch.linalg.vector_norm(reference_stft.abs(), dim=(-2, -1)).reshape(
                reference_bcs.shape[0], reference_bcs.shape[1]
            )
        )
    result = torch.stack(norms, dim=-1)
    if not bool(torch.isfinite(result).all().item()):
        raise FloatingPointError("MR-STFT reference norm produced NaN/Inf")
    return result


def normalize_marginals(mean_marginals: Tensor, epsilon: float = 1.0e-8) -> Tuple[Tensor, Tensor]:
    """Floor marginal improvements and return ``(clipped, sum-one prior)``."""

    if not mean_marginals.is_floating_point() or mean_marginals.ndim != 1:
        raise ValueError("mean_marginals must be a rank-one floating tensor")
    if mean_marginals.numel() == 0:
        raise ValueError("mean_marginals must be non-empty")
    if not math.isfinite(float(epsilon)) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    values = mean_marginals.float()
    if not bool(torch.isfinite(values).all().item()):
        raise FloatingPointError("mean_marginals contains NaN/Inf")
    clipped = values.clamp_min(float(epsilon))
    prior = clipped / clipped.sum()
    return clipped, prior


def progressive_prior(
    reference: Tensor,
    reconstructions: Tensor,
    *,
    fft_sizes: Sequence[int] = DEFAULT_FFT_SIZES,
    hop_ratio: float = 0.25,
    distance_epsilon: float = 1.0e-7,
    marginal_epsilon: float = 1.0e-8,
) -> ProgressivePriorOutput:
    """Compute the frozen prior from cumulative reconstructions.

    Args:
        reference: ``[B,C,S]`` or ``[B,S]`` target waveform.
        reconstructions: ``[B,Q+1,C,S]`` (or ``[B,Q+1,S]`` for mono), where
            index zero is waveform silence and index ``q`` uses codec
            codebooks ``0..q-1`` from one fixed encoding.
    """

    reference_bcs = _as_bcs(reference, "reference")
    if not reconstructions.is_floating_point():
        raise TypeError("reconstructions must be floating")
    if reconstructions.ndim == 3:
        reconstructions = reconstructions.unsqueeze(2)
    if reconstructions.ndim != 4:
        raise ValueError("reconstructions must have shape [B,Q+1,S] or [B,Q+1,C,S]")
    if reconstructions.shape[0] != reference_bcs.shape[0]:
        raise ValueError("batch dimensions do not match")
    if reconstructions.shape[2:] != reference_bcs.shape[1:]:
        raise ValueError("reconstruction channel/sample dimensions do not match reference")
    if reconstructions.shape[1] < 2:
        raise ValueError("at least silence and one codebook reconstruction are required")
    if reconstructions.device != reference_bcs.device:
        raise ValueError("reference and reconstructions must share a device")

    batch, levels, channels, samples = reconstructions.shape
    tiled_reference = reference_bcs[:, None].expand(-1, levels, -1, -1)
    distances = mrstft_distance(
        tiled_reference.reshape(batch * levels, channels, samples),
        reconstructions.reshape(batch * levels, channels, samples),
        fft_sizes=fft_sizes,
        hop_ratio=hop_ratio,
        epsilon=distance_epsilon,
    ).reshape(batch, levels)
    marginals = distances[:, :-1] - distances[:, 1:]
    mean_marginals = marginals.mean(dim=0)
    clipped, prior = normalize_marginals(mean_marginals, marginal_epsilon)
    return ProgressivePriorOutput(
        distances=distances,
        marginals=marginals,
        mean_marginals=mean_marginals,
        clipped_marginals=clipped,
        prior=prior,
    )


__all__ = [
    "DEFAULT_FFT_SIZES",
    "ProgressivePriorOutput",
    "mrstft_distance",
    "mrstft_reference_norms",
    "normalize_marginals",
    "progressive_prior",
]
