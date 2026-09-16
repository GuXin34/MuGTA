"""Small, import-free adapter between AudioCraft-like LMs and PTC-OPD.

The adapter deliberately relies only on the public ``compute_predictions``
shape contract instead of importing AudioCraft.  An LM-like object must expose
the following method::

    compute_predictions(
        codes, conditions=[], condition_tensors=...,
        keep_only_valid_steps=True,
    ) -> output

where ``output.logits`` is ``[B, Q, T, V]`` and ``output.mask`` is a boolean
``[B, Q, T]`` tensor.  This is the native dense-coordinate ``LMOutput``
contract in the pinned AudioCraft revision.

Only the student logits, the already-combined float32 CFG teacher logits, and
their strict common validity mask escape this module.  Conditional and
unconditional teacher logits are temporary values and are never retained in
the returned object.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Any, Dict, Iterator, Literal, Mapping, Optional, Tuple

import torch
from torch import Tensor

from .musicgen_contract import strict_validate_musicgen_batch


ConditionValue = Tuple[Tensor, Tensor]
ConditionTensors = Mapping[str, ConditionValue]
TeacherForwardMode = Literal["batched", "separate"]


@dataclass(frozen=True)
class AudioCraftTrajectoryScores:
    """Logits and validity mask for one student-sampled trajectory.

    Attributes:
        student_logits: Conditional student logits in ``[B, Q, T, V]``.  The
            original training graph is retained.
        teacher_cfg_logits: Detached float32 teacher logits constructed as
            ``uncond + scale * (cond - uncond)`` in ``[B, Q, T, V]``.
        valid_mask: Boolean ``[B, Q, T]`` intersection of the student,
            conditional-teacher, unconditional-teacher, rollout, and optional
            padding masks.
        teacher_cfg_scale: Finite CFG scale used for the combination.
        teacher_forward_mode: ``"batched"`` for one ``2B`` teacher call or
            ``"separate"`` for two ``B`` calls.
    """

    student_logits: Tensor
    teacher_cfg_logits: Tensor
    valid_mask: Tensor
    teacher_cfg_scale: float
    teacher_forward_mode: str


def _validate_condition_value(
    value: Any,
    *,
    name: str,
    key: str,
    expected_batch: int,
) -> ConditionValue:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(
            "{}['{}'] must be an (embedding, mask) tensor tuple".format(
                name, key
            )
        )
    embedding, mask = value
    if not isinstance(embedding, Tensor) or not isinstance(mask, Tensor):
        raise TypeError(
            "{}['{}'] must contain two tensors".format(name, key)
        )
    if embedding.ndim < 2 or mask.ndim < 2:
        raise ValueError(
            "{}['{}'] embedding and mask must include batch and time axes".format(
                name, key
            )
        )
    if embedding.shape[0] != expected_batch or mask.shape[0] != expected_batch:
        raise ValueError(
            "{}['{}'] has batch sizes embedding={} and mask={}, expected {}".format(
                name,
                key,
                embedding.shape[0],
                mask.shape[0],
                expected_batch,
            )
        )
    if embedding.shape[1] != mask.shape[1]:
        raise ValueError(
            "{}['{}'] embedding/mask time axes differ: {} versus {}".format(
                name, key, embedding.shape[1], mask.shape[1]
            )
        )
    return embedding, mask


def _validate_condition_tensors(
    condition_tensors: ConditionTensors,
    *,
    name: str,
    expected_batch: int,
) -> None:
    if not isinstance(condition_tensors, Mapping):
        raise TypeError("{} must be a mapping".format(name))
    for key, value in condition_tensors.items():
        if not isinstance(key, str):
            raise TypeError("{} keys must be strings".format(name))
        _validate_condition_value(
            value, name=name, key=key, expected_batch=expected_batch
        )


def _pad_time_axis(value: Tensor, target_length: int) -> Tensor:
    if value.shape[1] == target_length:
        return value
    shape = list(value.shape)
    shape[1] = target_length
    padded = value.new_zeros(shape)
    index = [slice(None)] * value.ndim
    index[1] = slice(0, value.shape[1])
    padded[tuple(index)] = value
    return padded


def batch_condition_tensors(
    conditional: ConditionTensors,
    unconditional: ConditionTensors,
    *,
    expected_batch: int,
) -> Dict[str, ConditionValue]:
    """Batch conditional then unconditional tensors for one ``2B`` forward.

    AudioCraft condition values are ``(embedding, mask)`` pairs.  Text lengths
    can differ if conditional and null conditions were precomputed separately,
    so this helper zero-pads their time axes before concatenating on batch.
    Non-time feature shapes, dtypes, devices, and key sets must match exactly.

    The returned order is always ``[conditional batch, unconditional batch]``;
    :func:`score_audiocraft_trajectory` relies on that order when splitting the
    teacher ``LMOutput``.
    """

    _validate_condition_tensors(
        conditional, name="conditional_condition_tensors", expected_batch=expected_batch
    )
    _validate_condition_tensors(
        unconditional,
        name="null_condition_tensors",
        expected_batch=expected_batch,
    )
    conditional_keys = set(conditional.keys())
    unconditional_keys = set(unconditional.keys())
    if conditional_keys != unconditional_keys:
        missing_null = sorted(conditional_keys - unconditional_keys)
        missing_conditional = sorted(unconditional_keys - conditional_keys)
        raise ValueError(
            "conditional/null condition keys differ; missing from null={}, "
            "missing from conditional={}".format(
                missing_null, missing_conditional
            )
        )

    batched: Dict[str, ConditionValue] = {}
    for key in conditional:
        cond_embedding, cond_mask = conditional[key]
        null_embedding, null_mask = unconditional[key]
        if cond_embedding.ndim != null_embedding.ndim:
            raise ValueError(
                "condition '{}' embedding ranks differ".format(key)
            )
        if cond_mask.ndim != null_mask.ndim:
            raise ValueError("condition '{}' mask ranks differ".format(key))
        if cond_embedding.shape[2:] != null_embedding.shape[2:]:
            raise ValueError(
                "condition '{}' embedding feature shapes differ: {} versus {}".format(
                    key, cond_embedding.shape[2:], null_embedding.shape[2:]
                )
            )
        if cond_mask.shape[2:] != null_mask.shape[2:]:
            raise ValueError(
                "condition '{}' mask feature shapes differ: {} versus {}".format(
                    key, cond_mask.shape[2:], null_mask.shape[2:]
                )
            )
        if cond_embedding.dtype != null_embedding.dtype:
            raise TypeError("condition '{}' embedding dtypes differ".format(key))
        if cond_mask.dtype != null_mask.dtype:
            raise TypeError("condition '{}' mask dtypes differ".format(key))
        if cond_embedding.device != null_embedding.device:
            raise ValueError("condition '{}' embedding devices differ".format(key))
        if cond_mask.device != null_mask.device:
            raise ValueError("condition '{}' mask devices differ".format(key))

        target_length = max(cond_embedding.shape[1], null_embedding.shape[1])
        cond_embedding = _pad_time_axis(cond_embedding, target_length)
        null_embedding = _pad_time_axis(null_embedding, target_length)
        cond_mask = _pad_time_axis(cond_mask, target_length)
        null_mask = _pad_time_axis(null_mask, target_length)
        batched[key] = (
            torch.cat((cond_embedding, null_embedding), dim=0),
            torch.cat((cond_mask, null_mask), dim=0),
        )
    return batched


def _validate_codes(codes: Tensor) -> Tuple[int, int, int]:
    if not isinstance(codes, Tensor):
        raise TypeError("student_codes must be a tensor")
    if codes.ndim != 3:
        raise ValueError("student_codes must have shape [B, Q, T]")
    if codes.dtype != torch.long:
        raise TypeError("student_codes must have dtype torch.long")
    batch, codebooks, time = codes.shape
    if batch <= 0 or codebooks <= 0 or time <= 0:
        raise ValueError("student_codes B, Q, and T dimensions must be positive")
    return batch, codebooks, time


def _validate_declared_codebooks(lm: Any, expected: int, name: str) -> None:
    for attribute in ("num_codebooks", "n_q"):
        if not hasattr(lm, attribute):
            continue
        value = getattr(lm, attribute)
        if callable(value):
            value = value()
        try:
            declared = int(value)
        except (TypeError, ValueError):
            continue
        if declared != expected:
            raise ValueError(
                "{} declares Q={}, but student_codes has Q={}".format(
                    name, declared, expected
                )
            )
        return


def _validate_mask(
    mask: Tensor,
    *,
    name: str,
    expected_shape: Tuple[int, int, int],
    expected_device: torch.device,
) -> Tensor:
    if not isinstance(mask, Tensor):
        raise TypeError("{} must be a tensor".format(name))
    if tuple(mask.shape) != expected_shape:
        raise ValueError(
            "{} must have shape {}, got {}".format(
                name, expected_shape, tuple(mask.shape)
            )
        )
    if mask.dtype != torch.bool:
        raise TypeError("{} must have dtype torch.bool".format(name))
    if mask.device != expected_device:
        raise ValueError(
            "{} must be on {}, got {}".format(
                name, expected_device, mask.device
            )
        )
    return mask


def _compute_predictions(
    lm: Any, codes: Tensor, condition_tensors: ConditionTensors
) -> Any:
    method = getattr(lm, "compute_predictions", None)
    if not callable(method):
        raise TypeError("LM-like objects must expose compute_predictions()")
    return method(
        codes,
        conditions=[],
        condition_tensors=dict(condition_tensors),
        keep_only_valid_steps=True,
    )


def _unpack_lm_output(
    output: Any,
    *,
    name: str,
    expected_batch: int,
    expected_codebooks: int,
    expected_time: int,
) -> Tuple[Tensor, Tensor]:
    logits = getattr(output, "logits", None)
    mask = getattr(output, "mask", None)
    if not isinstance(logits, Tensor) or not isinstance(mask, Tensor):
        raise TypeError("{} must expose tensor .logits and .mask".format(name))
    expected_prefix = (expected_batch, expected_codebooks, expected_time)
    if logits.ndim != 4 or tuple(logits.shape[:3]) != expected_prefix:
        raise ValueError(
            "{}.logits must have shape [B, Q, T, V] with prefix {}, got {}".format(
                name, expected_prefix, tuple(logits.shape)
            )
        )
    if logits.shape[-1] <= 0:
        raise ValueError("{}.logits vocabulary dimension must be positive".format(name))
    if not logits.is_floating_point():
        raise TypeError("{}.logits must have a floating-point dtype".format(name))
    if tuple(mask.shape) != expected_prefix:
        raise ValueError(
            "{}.mask must have shape {}, got {}".format(
                name, expected_prefix, tuple(mask.shape)
            )
        )
    if mask.dtype != torch.bool:
        raise TypeError("{}.mask must have dtype torch.bool".format(name))
    if logits.device != mask.device:
        raise ValueError("{}.logits and .mask must share a device".format(name))
    return logits, mask


def _check_finite_at_valid(name: str, logits: Tensor, valid_mask: Tensor) -> None:
    values = logits[valid_mask]
    if values.numel() == 0:
        return
    finite = torch.isfinite(values)
    if not bool(finite.all().item()):
        count = int((~finite).sum().item())
        raise FloatingPointError(
            "{} contains {} NaN/Inf values at valid positions".format(
                name, count
            )
        )


@contextmanager
def _temporary_teacher_eval(teacher_lm: Any) -> Iterator[None]:
    """Use eval behavior for scoring without permanently changing mode."""

    previous_training = getattr(teacher_lm, "training", None)
    eval_method = getattr(teacher_lm, "eval", None)
    train_method = getattr(teacher_lm, "train", None)
    if callable(eval_method):
        eval_method()
    try:
        yield
    finally:
        if previous_training is not None and callable(train_method):
            train_method(bool(previous_training))


def score_audiocraft_trajectory(
    student_lm: Any,
    teacher_lm: Any,
    student_codes: Tensor,
    conditional_condition_tensors: ConditionTensors,
    null_condition_tensors: ConditionTensors,
    *,
    rollout_mask: Tensor,
    padding_mask: Optional[Tensor] = None,
    teacher_cfg_scale: float = 3.0,
    teacher_forward_mode: TeacherForwardMode = "batched",
    check_finite: bool = True,
) -> AudioCraftTrajectoryScores:
    """Score one student trajectory with a conditional student and CFG teacher.

    The student forward is explicitly grad-enabled.  Teacher forwards and CFG
    construction run under ``torch.no_grad()``, and the returned teacher tensor
    is detached.  ``teacher_forward_mode="batched"`` duplicates codes and
    batches conditional/null condition tensors for one teacher call; use
    ``"separate"`` only when a conditioner cannot be safely batch-collated.

    Invalid AudioCraft pattern locations may contain NaNs.  Finiteness is
    therefore checked *only* after intersecting every LMOutput mask with the
    rollout and optional padding masks.
    """

    batch, codebooks, time = _validate_codes(student_codes)
    _validate_declared_codebooks(student_lm, codebooks, "student_lm")
    _validate_declared_codebooks(teacher_lm, codebooks, "teacher_lm")
    _validate_condition_tensors(
        conditional_condition_tensors,
        name="conditional_condition_tensors",
        expected_batch=batch,
    )
    _validate_condition_tensors(
        null_condition_tensors,
        name="null_condition_tensors",
        expected_batch=batch,
    )
    conditional_keys = set(conditional_condition_tensors.keys())
    null_keys = set(null_condition_tensors.keys())
    if conditional_keys != null_keys:
        raise ValueError(
            "conditional/null condition keys differ; missing from null={}, "
            "missing from conditional={}".format(
                sorted(conditional_keys - null_keys),
                sorted(null_keys - conditional_keys),
            )
        )

    expected_mask_shape = (batch, codebooks, time)
    rollout_mask = _validate_mask(
        rollout_mask,
        name="rollout_mask",
        expected_shape=expected_mask_shape,
        expected_device=student_codes.device,
    )
    if padding_mask is None:
        padding_mask = torch.ones_like(rollout_mask)
    else:
        padding_mask = _validate_mask(
            padding_mask,
            name="padding_mask",
            expected_shape=expected_mask_shape,
            expected_device=student_codes.device,
        )

    try:
        cfg_scale = float(teacher_cfg_scale)
    except (TypeError, ValueError) as exc:
        raise TypeError("teacher_cfg_scale must be a real scalar") from exc
    if not math.isfinite(cfg_scale):
        raise ValueError("teacher_cfg_scale must be finite")
    if teacher_forward_mode not in ("batched", "separate"):
        raise ValueError("teacher_forward_mode must be 'batched' or 'separate'")

    # Explicitly restore gradient recording if an outer validation/no-grad
    # context was accidentally left active.  A trainable student must still
    # produce graph-connected logits here.
    with torch.enable_grad():
        student_output = _compute_predictions(
            student_lm, student_codes, conditional_condition_tensors
        )
    student_logits, student_lm_mask = _unpack_lm_output(
        student_output,
        name="student_output",
        expected_batch=batch,
        expected_codebooks=codebooks,
        expected_time=time,
    )
    if not student_logits.requires_grad:
        raise RuntimeError(
            "student logits do not require gradients; check that the student "
            "parameters are trainable"
        )

    with _temporary_teacher_eval(teacher_lm), torch.no_grad():
        if teacher_forward_mode == "batched":
            batched_conditions = batch_condition_tensors(
                conditional_condition_tensors,
                null_condition_tensors,
                expected_batch=batch,
            )
            teacher_output = _compute_predictions(
                teacher_lm,
                torch.cat((student_codes, student_codes), dim=0),
                batched_conditions,
            )
            teacher_logits, teacher_mask = _unpack_lm_output(
                teacher_output,
                name="teacher_output",
                expected_batch=2 * batch,
                expected_codebooks=codebooks,
                expected_time=time,
            )
            teacher_cond_logits = teacher_logits[:batch]
            teacher_null_logits = teacher_logits[batch:]
            teacher_cond_mask = teacher_mask[:batch]
            teacher_null_mask = teacher_mask[batch:]
        else:
            teacher_cond_output = _compute_predictions(
                teacher_lm, student_codes, conditional_condition_tensors
            )
            teacher_null_output = _compute_predictions(
                teacher_lm, student_codes, null_condition_tensors
            )
            teacher_cond_logits, teacher_cond_mask = _unpack_lm_output(
                teacher_cond_output,
                name="teacher_cond_output",
                expected_batch=batch,
                expected_codebooks=codebooks,
                expected_time=time,
            )
            teacher_null_logits, teacher_null_mask = _unpack_lm_output(
                teacher_null_output,
                name="teacher_null_output",
                expected_batch=batch,
                expected_codebooks=codebooks,
                expected_time=time,
            )

        if student_logits.shape[-1] != teacher_cond_logits.shape[-1]:
            raise ValueError(
                "student and teacher vocabulary sizes differ: {} versus {}".format(
                    student_logits.shape[-1], teacher_cond_logits.shape[-1]
                )
            )
        if teacher_cond_logits.shape[-1] != teacher_null_logits.shape[-1]:
            raise ValueError("conditional and null teacher vocabulary sizes differ")
        if not (
            student_logits.device
            == teacher_cond_logits.device
            == teacher_null_logits.device
            == student_codes.device
        ):
            raise ValueError(
                "student codes and all logits must be on the same device"
            )

        # Do the extrapolation in float32 even when both LMs run under BF16.
        teacher_cond_fp32 = teacher_cond_logits.float()
        teacher_null_fp32 = teacher_null_logits.float()
        teacher_cfg_logits = (
            teacher_null_fp32
            + cfg_scale * (teacher_cond_fp32 - teacher_null_fp32)
        ).detach()

    # Primary MusicGen uses one exact delayed prediction lattice.  An
    # intersection would silently hide an entire missing branch/codebook, so
    # validate each raw LM mask and the vocabulary/card contract before any
    # optional rollout/padding restriction is applied.
    if codebooks == 4 and time == 500:
        card = getattr(student_lm, "card", student_logits.shape[-1])
        contract_mask = strict_validate_musicgen_batch(
            student_codes,
            int(card),
            {
                "student": student_lm_mask,
                "teacher_cond": teacher_cond_mask,
                "teacher_null": teacher_null_mask,
            },
            {
                "student": student_logits,
                "teacher_cond": teacher_cond_logits,
                "teacher_null": teacher_null_logits,
            },
        )
    else:
        # Small synthetic shapes remain useful for unit tests. Formal runners
        # separately gate Q=4/T=500 before invoking this adapter.
        contract_mask = student_lm_mask & teacher_cond_mask & teacher_null_mask

    valid_mask = (contract_mask & rollout_mask & padding_mask).detach()

    if check_finite:
        _check_finite_at_valid("student_logits", student_logits, valid_mask)
        _check_finite_at_valid(
            "teacher_cond_logits", teacher_cond_logits, valid_mask
        )
        _check_finite_at_valid(
            "teacher_null_logits", teacher_null_logits, valid_mask
        )
        _check_finite_at_valid(
            "teacher_cfg_logits", teacher_cfg_logits, valid_mask
        )

    return AudioCraftTrajectoryScores(
        student_logits=student_logits,
        teacher_cfg_logits=teacher_cfg_logits,
        valid_mask=valid_mask,
        teacher_cfg_scale=cfg_scale,
        teacher_forward_mode=teacher_forward_mode,
    )


__all__ = [
    "AudioCraftTrajectoryScores",
    "ConditionTensors",
    "ConditionValue",
    "TeacherForwardMode",
    "batch_condition_tensors",
    "score_audiocraft_trajectory",
]
