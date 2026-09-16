"""Strict frozen tensor and model-metadata contract for primary MusicGen runs.

This module intentionally depends only on PyTorch.  It does not import
AudioCraft, and it treats AudioCraft-like models and pattern providers through
their public attributes/methods.  The primary Stage-1 contract is deliberately
narrow: four codebooks, 500 codec frames at 50 Hz, and delays ``[0, 1, 2, 3]``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Tuple

import torch
from torch import Tensor


PRIMARY_CODEBOOKS = 4
PRIMARY_FRAMES = 500
PRIMARY_FRAME_RATE = 50.0
PRIMARY_CARD = 2048
PRIMARY_DELAYS: Tuple[int, ...] = (0, 1, 2, 3)
REQUIRED_SCORE_NAMES: Tuple[str, ...] = (
    "student",
    "teacher_cond",
    "teacher_null",
)


@dataclass(frozen=True)
class MusicGenContractMetadata:
    """Canonical model metadata after all strict gates pass."""

    num_codebooks: int
    frame_rate: float
    card: int
    pattern_checked: bool


def _strict_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("{} must be an integer, not {!r}".format(name, type(value).__name__))
    if value <= 0:
        raise ValueError("{} must be positive, got {}".format(name, value))
    return value


def _mapping_with_exact_names(name: str, value: Any) -> Mapping[str, Tensor]:
    if not isinstance(value, Mapping):
        raise TypeError("{} must be a mapping from score name to tensor".format(name))
    keys = set(value.keys())
    required = set(REQUIRED_SCORE_NAMES)
    if keys != required:
        missing = sorted(required - keys)
        unexpected = sorted(keys - required, key=str)
        raise ValueError(
            "{} must contain exactly {}; missing={}, unexpected={}".format(
                name, list(REQUIRED_SCORE_NAMES), missing, unexpected
            )
        )
    return value


def expected_musicgen_delay_mask(
    batch_size: int,
    *,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Return frozen ``[B, 4, 500]`` validity for delays ``[0,1,2,3]``.

    Codebook ``q`` has its first ``500-q`` cells valid and exactly ``q`` false
    tail cells.  Per-codebook valid counts are therefore
    ``[500, 499, 498, 497]`` for every example.
    """

    batch_size = _strict_positive_int("batch_size", batch_size)
    time = torch.arange(PRIMARY_FRAMES, device=device).view(1, PRIMARY_FRAMES)
    valid_until = (
        PRIMARY_FRAMES
        - torch.tensor(PRIMARY_DELAYS, device=device).view(PRIMARY_CODEBOOKS, 1)
    )
    per_codebook = time < valid_until
    return per_codebook.unsqueeze(0).expand(batch_size, -1, -1).clone()


def strict_validate_musicgen_batch(
    codes: Tensor,
    card: int,
    named_masks: Mapping[str, Tensor],
    named_logits: Optional[Mapping[str, Tensor]] = None,
) -> Tensor:
    """Validate one complete student/teacher scoring batch.

    Args:
        codes: Canonical codec tokens, exactly ``torch.long [B,4,500]``.
            Every value must satisfy ``0 <= token < card``.  In particular,
            AudioCraft's LM special-token ID ``card`` is rejected even though
            upstream generation has historically asserted ``<= card``.
        card: Positive LM vocabulary cardinality read from ``lm.card``.
        named_masks: Exactly ``student``, ``teacher_cond``, and
            ``teacher_null`` boolean masks.  Each must be ``[B,4,500]``, share
            the code device, equal the other masks elementwise, and equal the
            frozen delay expectation exactly.
        named_logits: If provided, exactly the same three names, with each
            floating tensor shaped ``[B,4,500,card]`` on the code device.

    Returns:
        A detached canonical boolean ``[B,4,500]`` valid mask.  It is exactly
        the validated shared LM mask, not a union or a permissive intersection.
    """

    card = _strict_positive_int("card", card)
    if card != PRIMARY_CARD:
        raise ValueError(
            "primary MusicGen requires card=2048, got {}".format(card)
        )
    if not isinstance(codes, Tensor):
        raise TypeError("codes must be a torch.Tensor")
    if codes.dtype != torch.long:
        raise TypeError("codes must have dtype torch.long")
    if codes.ndim != 3:
        raise ValueError("codes must have shape [B,4,500], got rank {}".format(codes.ndim))
    batch, codebooks, frames = codes.shape
    if batch <= 0:
        raise ValueError("codes batch dimension must be positive")
    if codebooks != PRIMARY_CODEBOOKS or frames != PRIMARY_FRAMES:
        raise ValueError(
            "codes must have shape [B,4,500], got {}".format(tuple(codes.shape))
        )
    minimum = int(codes.min().item())
    maximum = int(codes.max().item())
    if minimum < 0:
        raise ValueError("codes contain token {} below zero".format(minimum))
    if maximum >= card:
        raise ValueError(
            "codes contain token {} outside [0, card); card={} (the LM special "
            "token ID card is not a codec token)".format(maximum, card)
        )

    masks = _mapping_with_exact_names("named_masks", named_masks)
    expected_shape = (batch, PRIMARY_CODEBOOKS, PRIMARY_FRAMES)
    expected = expected_musicgen_delay_mask(batch, device=codes.device)
    reference: Optional[Tensor] = None
    reference_name: Optional[str] = None
    for score_name in REQUIRED_SCORE_NAMES:
        mask = masks[score_name]
        if not isinstance(mask, Tensor):
            raise TypeError("named_masks[{!r}] must be a tensor".format(score_name))
        if mask.dtype != torch.bool:
            raise TypeError("named_masks[{!r}] must have dtype torch.bool".format(score_name))
        if tuple(mask.shape) != expected_shape:
            raise ValueError(
                "named_masks[{!r}] must have shape {}, got {}".format(
                    score_name, expected_shape, tuple(mask.shape)
                )
            )
        if mask.device != codes.device:
            raise ValueError(
                "named_masks[{!r}] must be on {}, got {}".format(
                    score_name, codes.device, mask.device
                )
            )
        if reference is not None and not torch.equal(mask, reference):
            mismatch = int((mask != reference).sum().item())
            raise ValueError(
                "LM masks are not elementwise identical: {!r} differs from "
                "{!r} at {} cells".format(score_name, reference_name, mismatch)
            )
        if not torch.equal(mask, expected):
            mismatch = int((mask != expected).sum().item())
            counts = mask.sum(dim=-1)[0].tolist()
            raise ValueError(
                "named_masks[{!r}] violates frozen delay [0,1,2,3]: {} "
                "mismatched cells; first-example counts={} expected="
                "[500,499,498,497]".format(score_name, mismatch, counts)
            )
        reference = mask
        reference_name = score_name

    if named_logits is not None:
        logits = _mapping_with_exact_names("named_logits", named_logits)
        expected_logits_shape = expected_shape + (card,)
        reference_shape = None
        for score_name in REQUIRED_SCORE_NAMES:
            value = logits[score_name]
            if not isinstance(value, Tensor):
                raise TypeError("named_logits[{!r}] must be a tensor".format(score_name))
            if not value.is_floating_point():
                raise TypeError(
                    "named_logits[{!r}] must have a floating-point dtype".format(
                        score_name
                    )
                )
            if tuple(value.shape) != expected_logits_shape:
                raise ValueError(
                    "named_logits[{!r}] must have BQTV shape {}, got {}".format(
                        score_name, expected_logits_shape, tuple(value.shape)
                    )
                )
            if value.device != codes.device:
                raise ValueError(
                    "named_logits[{!r}] must be on {}, got {}".format(
                        score_name, codes.device, value.device
                    )
                )
            if reference_shape is not None and value.shape != reference_shape:
                raise ValueError("student/teacher logits are not shape-isomorphic")
            reference_shape = value.shape

    assert reference is not None
    return reference.detach().clone()


def _validate_pattern_provider(pattern_provider: Any, card: int) -> None:
    get_pattern = getattr(pattern_provider, "get_pattern", None)
    if not callable(get_pattern):
        raise TypeError("pattern_provider must expose callable get_pattern(timesteps)")
    try:
        pattern = get_pattern(PRIMARY_FRAMES)
        build = getattr(pattern, "build_pattern_sequence", None)
        revert = getattr(pattern, "revert_pattern_logits", None)
        if not callable(build) or not callable(revert):
            raise TypeError(
                "pattern must expose build_pattern_sequence() and "
                "revert_pattern_logits()"
            )
        dummy_codes = torch.zeros(
            (1, PRIMARY_CODEBOOKS, PRIMARY_FRAMES), dtype=torch.long
        )
        sequence, _, _ = build(
            dummy_codes,
            special_token=card,
            keep_only_valid_steps=True,
        )
        if not isinstance(sequence, Tensor) or sequence.ndim != 3:
            raise TypeError("pattern build returned an invalid sequence tensor")
        dummy_logits = torch.zeros(
            (1, 1, PRIMARY_CODEBOOKS, sequence.shape[-1]), dtype=torch.float32
        )
        _, _, actual_mask = revert(
            dummy_logits,
            special_token=float("nan"),
            keep_only_valid_steps=True,
        )
    except (AssertionError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            "could not validate pattern_provider at 500 frames: {}".format(exc)
        ) from exc

    expected = expected_musicgen_delay_mask(1)[0]
    if not isinstance(actual_mask, Tensor):
        raise TypeError("pattern revert did not return a tensor mask")
    if actual_mask.dtype != torch.bool:
        raise TypeError("pattern validity mask must have dtype torch.bool")
    if tuple(actual_mask.shape) != tuple(expected.shape):
        raise ValueError(
            "pattern validity mask must have shape [4,500], got {}".format(
                tuple(actual_mask.shape)
            )
        )
    actual_mask = actual_mask.cpu()
    if not torch.equal(actual_mask, expected):
        mismatch = int((actual_mask != expected).sum().item())
        counts = actual_mask.sum(dim=-1).tolist()
        raise ValueError(
            "pattern_provider is not frozen delay [0,1,2,3]: {} mismatched "
            "cells; counts={} expected=[500,499,498,497]".format(
                mismatch, counts
            )
        )


def strict_validate_musicgen_metadata(
    *,
    num_codebooks: int,
    frame_rate: float,
    card: int,
    pattern_provider: Optional[Any] = None,
) -> MusicGenContractMetadata:
    """Gate primary MusicGen model metadata and optionally its real pattern."""

    num_codebooks = _strict_positive_int("num_codebooks", num_codebooks)
    card = _strict_positive_int("card", card)
    if isinstance(frame_rate, bool) or not isinstance(frame_rate, (int, float)):
        raise TypeError("frame_rate must be a real scalar")
    frame_rate_value = float(frame_rate)
    if not math.isfinite(frame_rate_value) or frame_rate_value <= 0.0:
        raise ValueError("frame_rate must be finite and positive")
    if num_codebooks != PRIMARY_CODEBOOKS:
        raise ValueError(
            "primary MusicGen requires num_codebooks=4, got {}".format(
                num_codebooks
            )
        )
    if card != PRIMARY_CARD:
        raise ValueError(
            "primary MusicGen requires card=2048, got {}".format(card)
        )
    if frame_rate_value != PRIMARY_FRAME_RATE:
        raise ValueError(
            "primary MusicGen requires frame_rate=50, got {}".format(
                frame_rate_value
            )
        )
    pattern_checked = pattern_provider is not None
    if pattern_checked:
        _validate_pattern_provider(pattern_provider, card)
    return MusicGenContractMetadata(
        num_codebooks=num_codebooks,
        frame_rate=frame_rate_value,
        card=card,
        pattern_checked=pattern_checked,
    )


def strict_validate_musicgen_model(
    model: Any,
    *,
    frame_rate: float,
) -> MusicGenContractMetadata:
    """Convenience metadata gate for an AudioCraft-like LM object."""

    if model is None:
        raise TypeError("model must not be None")
    try:
        num_codebooks = getattr(model, "num_codebooks")
        card = getattr(model, "card")
    except AttributeError as exc:
        raise TypeError("model must expose num_codebooks and card") from exc
    return strict_validate_musicgen_metadata(
        num_codebooks=num_codebooks,
        frame_rate=frame_rate,
        card=card,
        pattern_provider=getattr(model, "pattern_provider", None),
    )


__all__ = [
    "MusicGenContractMetadata",
    "PRIMARY_CARD",
    "PRIMARY_CODEBOOKS",
    "PRIMARY_DELAYS",
    "PRIMARY_FRAME_RATE",
    "PRIMARY_FRAMES",
    "REQUIRED_SCORE_NAMES",
    "expected_musicgen_delay_mask",
    "strict_validate_musicgen_batch",
    "strict_validate_musicgen_metadata",
    "strict_validate_musicgen_model",
]
