"""Framework-independent PyTorch loss kernel for PTC-OPD.

The canonical tensor layout follows AudioCraft's ``LMOutput``:

* logits: ``[batch, codebook, time, vocabulary]`` (``BQTV``)
* valid mask: ``[batch, codebook, time]`` (``BQT``)

``BTQV``/``BTQ`` is supported only when ``layout="BTQV"`` is passed.  Axes are
never inferred from their sizes.  All diagnostic tensors are returned in the
canonical ``BQT`` layout regardless of the input layout.

The frozen primary design retains ``ceil(rho * n_valid)`` time positions within
every ``(batch, codebook)`` group.  Thus matched-random, disagreement, and PTC
have identical per-codebook budgets.  PTC and disagreement rank by the same
detached JS score; they differ only in whether ``a_q`` weights the normalized
KL.  This makes the component ablation identifiable.  A sequence-global top-rho
scope remains available only as a diagnostic, never as the primary default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


_VALID_MODES = {
    "uniform",
    "codebook_only",
    "random_stratified",
    "prefix",
    "disagreement",
    "ptc",
}
_VALID_DIRECTIONS = {"forward", "reverse"}
_VALID_SCOPES = {"protocol", "sequence", "codebook"}
_VALID_LAYOUTS = {"BQTV", "BTQV"}


@dataclass(frozen=True)
class OPDLossOutput:
    """Loss plus detached diagnostics in canonical ``BQT`` layout.

    ``loss`` and ``numerator`` retain an autograd graph.  ``numerator`` is
    exposed specifically so a framework adapter can construct the required DDP
    global numerator/global denominator objective; it must not average local
    rank means.  With ordinary DDP gradient averaging, all-reduce the detached
    denominator and optimize ``world_size * local_numerator / global_denominator``.
    All-reduce a *detached* numerator separately for logging.  All other tensors
    are detached and safe to log.  Counts have shape ``[B, Q]`` and are reduced
    only over time.
    """

    loss: Tensor
    token_kl: Tensor
    js_divergence: Tensor
    selection_score: Tensor
    selected_mask: Tensor
    effective_weights: Tensor
    codebook_weights: Tensor
    numerator: Tensor
    effective_weight_sum: Tensor
    valid_counts_per_codebook: Tensor
    selected_counts_per_codebook: Tensor
    selection_rates_per_codebook: Tensor
    realized_retention: Tensor
    mode: str
    kl_direction: str
    selection_scope: str
    input_layout: str
    requested_rho: Optional[float]
    resolved_rho: float
    temperature: float


def _canonical_mode(mode: str) -> str:
    canonical = mode.strip().lower().replace("-", "_")
    aliases = {
        "codebook": "codebook_only",
        "random": "random_stratified",
        "random_per_codebook": "random_stratified",
        "prefix_stratified": "prefix",
        "early": "prefix",
        "js": "disagreement",
    }
    canonical = aliases.get(canonical, canonical)
    if canonical not in _VALID_MODES:
        raise ValueError(
            "mode must be one of " + ", ".join(sorted(_VALID_MODES))
        )
    return canonical


def _canonical_direction(direction: str) -> str:
    canonical = direction.strip().lower().replace("_kl", "")
    if canonical not in _VALID_DIRECTIONS:
        raise ValueError("kl_direction must be 'forward' or 'reverse'")
    return canonical


def _canonical_scope(scope: str) -> str:
    canonical = scope.strip().lower().replace("per_", "")
    aliases = {
        "global": "sequence",
        "global_sequence": "sequence",
        "global_joint_selector": "sequence",
        "stratified": "codebook",
    }
    canonical = aliases.get(canonical, canonical)
    if canonical not in _VALID_SCOPES:
        raise ValueError(
            "selection_scope must be 'protocol', 'sequence', or 'codebook'"
        )
    return canonical


def _canonical_layout(layout: str) -> str:
    canonical = layout.strip().upper().replace("[", "").replace("]", "")
    canonical = canonical.replace(",", "").replace(" ", "")
    if canonical not in _VALID_LAYOUTS:
        raise ValueError("layout must be exactly 'BQTV' or 'BTQV'")
    return canonical


def _to_bqtv(logits: Tensor, layout: str) -> Tensor:
    if logits.ndim != 4:
        raise ValueError(
            "student_logits and teacher_logits must be rank-4 tensors; "
            "expected BQTV or explicitly declared BTQV"
        )
    if layout == "BQTV":
        return logits
    return logits.permute(0, 2, 1, 3)


def _to_bqt(mask: Tensor, layout: str) -> Tensor:
    if mask.ndim != 3:
        raise ValueError("valid_mask must be rank 3 (BQT, or BTQ for BTQV input)")
    if layout == "BQTV":
        return mask
    return mask.permute(0, 2, 1)


def _resolve_rho(mode: str, rho: Optional[float]) -> float:
    if rho is None:
        return 1.0 if mode in {"uniform", "codebook_only"} else 0.5
    try:
        value = float(rho)
    except (TypeError, ValueError) as exc:
        raise TypeError("rho must be a real scalar") from exc
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("rho must be finite and lie in [0, 1]")
    if mode in {"uniform", "codebook_only"} and value != 1.0:
        raise ValueError(
            "uniform and codebook_only are all-position conditions and require "
            "rho=1.0 (or rho=None); use a fractional selector mode otherwise"
        )
    return value


def _retained_count(valid_count: int, rho: float) -> int:
    """Return a deterministic count, using ceil for fractional small groups."""

    if valid_count <= 0 or rho <= 0.0:
        return 0
    if rho >= 1.0:
        return valid_count
    return min(valid_count, max(1, int(math.ceil(rho * valid_count))))


def _top_fraction(
    score: Tensor,
    valid_mask: Tensor,
    rho: float,
    scope: str,
) -> Tensor:
    """Select stable top-rho positions within sequence or codebook groups."""

    selected = torch.zeros_like(valid_mask, dtype=torch.bool)
    if rho <= 0.0:
        return selected
    if rho >= 1.0:
        return valid_mask.clone()

    batch, codebooks, _ = valid_mask.shape
    # Selection is discrete by design; no gradient may flow through JS/ranking.
    with torch.no_grad():
        for batch_index in range(batch):
            if scope == "sequence":
                valid_flat = valid_mask[batch_index].reshape(-1)
                valid_indices = torch.nonzero(valid_flat, as_tuple=False).flatten()
                keep = _retained_count(valid_indices.numel(), rho)
                if keep:
                    candidate_score = score[batch_index].reshape(-1)[valid_indices]
                    order = torch.argsort(
                        candidate_score, descending=True, stable=True
                    )
                    selected[batch_index].view(-1)[valid_indices[order[:keep]]] = True
            else:
                for codebook_index in range(codebooks):
                    valid_line = valid_mask[batch_index, codebook_index]
                    valid_indices = torch.nonzero(
                        valid_line, as_tuple=False
                    ).flatten()
                    keep = _retained_count(valid_indices.numel(), rho)
                    if keep:
                        candidate_score = score[
                            batch_index, codebook_index, valid_indices
                        ]
                        order = torch.argsort(
                            candidate_score, descending=True, stable=True
                        )
                        selected[
                            batch_index,
                            codebook_index,
                            valid_indices[order[:keep]],
                        ] = True
    return selected


def _prefix_score(valid_mask: Tensor) -> Tensor:
    """Score canonical time positions so the earliest valid cells rank first."""

    time = valid_mask.shape[-1]
    score = torch.arange(
        time, 0, -1, device=valid_mask.device, dtype=torch.float32
    ).view(1, 1, time)
    return score.expand_as(valid_mask).detach()


def _random_score(
    shape: torch.Size,
    device: torch.device,
    generator: Optional[torch.Generator],
    random_seed: Optional[int],
) -> Tensor:
    if generator is not None and random_seed is not None:
        raise ValueError("pass generator or random_seed, not both")

    if random_seed is not None:
        # Generate on CPU so one declared seed is reproducible across CPU/CUDA
        # integration tests as well as across repeated runs on the same device.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(random_seed))

    if generator is None:
        return torch.rand(shape, device=device, dtype=torch.float32)

    source_device = getattr(generator, "device", torch.device("cpu"))
    try:
        score = torch.rand(
            shape,
            device=source_device,
            dtype=torch.float32,
            generator=generator,
        )
    except RuntimeError as exc:
        raise ValueError(
            "generator is incompatible with its declared device; pass a seeded "
            "torch.Generator or random_seed"
        ) from exc
    return score.to(device=device)


def _resolve_scope(mode: str, requested_scope: str) -> str:
    if mode in {"uniform", "codebook_only"}:
        return "all"
    if requested_scope != "protocol":
        return requested_scope
    # Frozen primary protocol: every fractional selector uses the same retained
    # count in each (batch, codebook) group.  Sequence-global is diagnostic only.
    return "codebook"


def _normalized_codebook_weights(
    weights: Optional[Tensor],
    codebooks: int,
    device: torch.device,
) -> Tensor:
    if weights is None:
        return torch.full(
            (codebooks,), 1.0 / codebooks, device=device, dtype=torch.float32
        )
    value = torch.as_tensor(weights, device=device, dtype=torch.float32).detach()
    if value.ndim != 1 or value.shape[0] != codebooks:
        raise ValueError(
            "codebook_weights must be a rank-1 tensor/list with length Q"
        )
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError("codebook_weights must be finite")
    if not bool((value > 0).all().item()):
        raise ValueError(
            "codebook_weights must be strictly positive; the protocol prior "
            "uses max(delta, eps) before normalization"
        )
    return value / value.sum()


def _check_valid_logits_finite(name: str, logits: Tensor, valid_mask: Tensor) -> None:
    values = logits[valid_mask]
    if values.numel() == 0:
        return
    finite = torch.isfinite(values)
    if not bool(finite.all().item()):
        bad = int((~finite).sum().item())
        raise FloatingPointError(
            "{} contains {} NaN/Inf values at valid token positions".format(
                name, bad
            )
        )


def ptc_opd_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    valid_mask: Optional[Tensor] = None,
    mode: str = "ptc",
    rho: Optional[float] = None,
    codebook_weights: Optional[Tensor] = None,
    kl_direction: str = "forward",
    selection_scope: str = "protocol",
    layout: str = "BQTV",
    generator: Optional[torch.Generator] = None,
    random_seed: Optional[int] = None,
    random_scores: Optional[Tensor] = None,
    temperature: float = 1.0,
    normalization_eps: float = 1.0e-12,
    check_finite: bool = True,
    allow_empty: bool = False,
) -> OPDLossOutput:
    """Compute a frozen primary OPD loss or the small-only prefix diagnostic.

    Args:
        student_logits: Trainable logits in explicit ``layout``.
        teacher_logits: Frozen-teacher logits, shape-identical to student logits.
            They are detached inside this function even if ``requires_grad`` is
            accidentally enabled upstream.
        valid_mask: Boolean ``BQT`` mask (``BTQ`` for ``layout="BTQV"``).
            If omitted, every time-codebook position is valid.  Invalid logits
            may contain NaN/Inf and are sanitized before softmax; non-finite
            logits at a valid position raise ``FloatingPointError`` by default.
        mode: ``uniform``, ``codebook_only``, ``random_stratified``,
            ``prefix``, ``disagreement``, or ``ptc``. ``prefix`` selects the
            earliest matched half as a pre-pilot temporal-skew control.
        rho: Retained fraction in ``[0, 1]``.  ``None`` resolves to 1.0 for the
            all-position uniform/codebook-only modes and 0.5 for fractional
            modes.  Explicit uniform/codebook-only rho must equal 1.0 so a
            configuration mismatch cannot be silently ignored.  Fractional
            group sizes use ``ceil(rho * n_valid)``; rho=0 selects none and
            rho=1 selects all.
        codebook_weights: Positive offline perceptual prior ``a_q``.  It is
            normalized to sum to one.  It weights PTC/codebook-only KL but never
            the within-codebook ranking score (multiplying a codebook by its
            constant ``a_q`` cannot alter that ranking).  Other modes use a
            uniform codebook prior, matching the frozen experiment matrix.
        kl_direction: ``forward`` computes ``KL(p_teacher || p_student)``;
            ``reverse`` computes ``KL(p_student || p_teacher)``.
        selection_scope: ``protocol`` (primary default), ``sequence`` (global
            over all valid q,t in each example), or ``codebook`` (top-rho
            separately for every example/codebook).  Under ``protocol``, all
            fractional modes use codebook scope, so random/disagreement/PTC
            retain exactly matched ``(b,q)`` counts.  ``global_sequence`` and
            ``global_joint_selector`` are accepted as explicit diagnostic
            aliases for ``sequence``.  In that PTC-only diagnostic, ranking is
            joint ``a_q * JS``; primary protocol/codebook ranking is plain JS.
        layout: Exactly ``BQTV`` (AudioCraft canonical) or explicit ``BTQV``.
            Axis sizes are never inspected to guess a layout.
        generator: Optional seeded ``torch.Generator`` for random selection.
        random_seed: Convenience deterministic seed; mutually exclusive with
            ``generator``.  In distributed runs derive it from experiment seed,
            rank, and optimizer step to avoid repeated masks.
        random_scores: Optional caller-supplied random ranking scores.  Its
            shape follows the declared layout (``BQT`` for ``BQTV`` or ``BTQ``
            for ``BTQV``).  This is the preferred distributed-training path:
            derive each sample's scores from a stable sample ID so its random
            control mask does not depend on rank or batch order.  Values are
            detached internally and must be finite at valid positions.  It is
            mutually exclusive with ``generator`` and ``random_seed``.
        temperature: Positive finite distillation temperature ``tau``.  Both
            distributions and the detached JS selector use ``logits / tau``;
            token KL is multiplied by ``tau ** 2`` to retain the standard
            distillation gradient scale.  JS itself is not multiplied by
            ``tau ** 2`` because it is used only for ranking.
        normalization_eps: Positive clamp for the effective-weight denominator.
        check_finite: Validate all logits at valid positions and all resulting
            divergences.  Keep enabled for tests/pilots; disabling it avoids a
            synchronization-heavy diagnostic in a fully audited training run.
        allow_empty: If false (training-safe default), raise when the batch has
            no valid tokens or selection yields no effective token (e.g.
            ``rho=0``).  If true, return a differentiable exact zero for an
            explicit boundary test/diagnostic.

    Returns:
        ``OPDLossOutput``.  The scalar loss is

        ``sum(g_tq * a_q * KL_tq) / sum(g_tq * a_q)``.

        ``numerator`` retains its autograd graph for a DDP adapter.  With normal
        DDP gradient averaging, all-reduce a detached denominator and optimize
        ``world_size * local_numerator / global_denominator`` on every rank.
        This yields the gradient of the true global ratio; never average local
        means and do not use a non-autograd ``dist.all_reduce`` directly on the
        live numerator.  A detached numerator may be all-reduced for logging.
        If ``allow_empty=True`` and no effective token is selected, the local
        loss and numerator are differentiable exact zeros.
    """

    canonical_mode = _canonical_mode(mode)
    canonical_direction = _canonical_direction(kl_direction)
    canonical_scope = _canonical_scope(selection_scope)
    canonical_layout = _canonical_layout(layout)
    rho_value = _resolve_rho(canonical_mode, rho)

    random_inputs = sum(
        value is not None for value in (generator, random_seed, random_scores)
    )
    if random_inputs > 1:
        raise ValueError(
            "pass exactly one of generator, random_seed, or random_scores"
        )
    if canonical_mode != "random_stratified" and random_inputs:
        raise ValueError(
            "generator, random_seed, and random_scores are valid only for "
            "mode='random_stratified'"
        )

    try:
        temperature_value = float(temperature)
    except (TypeError, ValueError) as exc:
        raise TypeError("temperature must be a real scalar") from exc
    if not math.isfinite(temperature_value) or temperature_value <= 0.0:
        raise ValueError("temperature must be finite and positive")

    if not student_logits.is_floating_point() or not teacher_logits.is_floating_point():
        raise TypeError("student_logits and teacher_logits must be floating tensors")
    if student_logits.device != teacher_logits.device:
        raise ValueError("student_logits and teacher_logits must share a device")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student_logits and teacher_logits must have equal shapes")
    if not math.isfinite(float(normalization_eps)) or normalization_eps <= 0.0:
        raise ValueError("normalization_eps must be finite and positive")

    student = _to_bqtv(student_logits, canonical_layout)
    teacher = _to_bqtv(teacher_logits, canonical_layout)
    batch, codebooks, time, vocabulary = student.shape
    if min(batch, codebooks, time, vocabulary) <= 0:
        raise ValueError("B, Q, T, and V dimensions must all be non-zero")

    if valid_mask is None:
        valid = torch.ones(
            (batch, codebooks, time), device=student.device, dtype=torch.bool
        )
    else:
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must have dtype torch.bool")
        if valid_mask.device != student.device:
            raise ValueError("valid_mask and logits must share a device")
        valid = _to_bqt(valid_mask, canonical_layout)
        if valid.shape != (batch, codebooks, time):
            raise ValueError(
                "valid_mask shape does not match logits/layout; expected "
                "({}, {}, {}) in canonical BQT".format(batch, codebooks, time)
            )

    supplied_random_score: Optional[Tensor] = None
    if random_scores is not None:
        if not isinstance(random_scores, Tensor):
            raise TypeError("random_scores must be a tensor")
        if not random_scores.is_floating_point():
            raise TypeError("random_scores must have a floating-point dtype")
        if random_scores.device != student.device:
            raise ValueError("random_scores and logits must share a device")
        supplied_random_score = _to_bqt(random_scores, canonical_layout)
        if supplied_random_score.shape != (batch, codebooks, time):
            raise ValueError(
                "random_scores shape does not match logits/layout; expected "
                "({}, {}, {}) in canonical BQT".format(batch, codebooks, time)
            )
        supplied_random_score = supplied_random_score.detach().float()
        if not bool(torch.isfinite(supplied_random_score[valid]).all().item()):
            raise FloatingPointError(
                "random_scores contains NaN/Inf at a valid position"
            )

    if not allow_empty and not bool(valid.any().item()):
        raise ValueError(
            "valid_mask contains no valid tokens; pass allow_empty=True only "
            "for an explicit diagnostic"
        )

    prior = _normalized_codebook_weights(
        codebook_weights, codebooks, student.device
    )
    uniform_prior = torch.full_like(prior, 1.0 / codebooks)

    if check_finite:
        _check_valid_logits_finite("student_logits", student, valid)
        _check_valid_logits_finite("teacher_logits", teacher, valid)

    # Invalid positions are replaced before softmax, so padding garbage can
    # neither poison the loss nor receive a gradient.  Divergences use fp32 even
    # when the model emits fp16/bf16 logits.
    valid_vocab = valid.unsqueeze(-1)
    student_fp32 = torch.where(
        valid_vocab, student.float(), torch.zeros((), device=student.device)
    )
    teacher_fp32 = torch.where(
        valid_vocab,
        teacher.detach().float(),
        torch.zeros((), device=teacher.device),
    )
    student_logp = F.log_softmax(student_fp32 / temperature_value, dim=-1)
    teacher_logp = F.log_softmax(teacher_fp32 / temperature_value, dim=-1)
    student_prob = student_logp.exp()
    teacher_prob = teacher_logp.exp()

    if canonical_direction == "forward":
        token_kl = temperature_value**2 * (
            teacher_prob * (teacher_logp - student_logp)
        ).sum(dim=-1)
    else:
        token_kl = temperature_value**2 * (
            student_prob * (student_logp - teacher_logp)
        ).sum(dim=-1)
    token_kl = torch.where(valid, token_kl, torch.zeros_like(token_kl))

    with torch.no_grad():
        student_logp_detached = student_logp.detach()
        student_prob_detached = student_prob.detach()
        log_mixture = torch.logaddexp(
            teacher_logp, student_logp_detached
        ) - math.log(2.0)
        js_divergence = 0.5 * (
            (
                teacher_prob * (teacher_logp - log_mixture)
            ).sum(dim=-1)
            + (
                student_prob_detached
                * (student_logp_detached - log_mixture)
            ).sum(dim=-1)
        )
        # Tiny negative values are roundoff, not meaningful ranking signals.
        js_divergence = js_divergence.clamp_min(0.0)
        js_divergence = torch.where(
            valid, js_divergence, torch.zeros_like(js_divergence)
        )

    if check_finite:
        if not bool(torch.isfinite(token_kl[valid]).all().item()):
            raise FloatingPointError("KL produced NaN/Inf at a valid position")
        if not bool(torch.isfinite(js_divergence[valid]).all().item()):
            raise FloatingPointError("JS produced NaN/Inf at a valid position")

    used_scope = _resolve_scope(canonical_mode, canonical_scope)
    if canonical_mode in {"uniform", "codebook_only"}:
        selection_score = torch.zeros_like(js_divergence)
        selected = valid.clone()
    elif canonical_mode == "random_stratified":
        selection_score = (
            supplied_random_score
            if supplied_random_score is not None
            else _random_score(
                valid.shape, student.device, generator, random_seed
            )
        )
        selection_score = torch.where(
            valid, selection_score, torch.zeros_like(selection_score)
        )
        selected = _top_fraction(
            selection_score, valid, rho_value, used_scope
        )
    elif canonical_mode == "prefix":
        selection_score = torch.where(
            valid, _prefix_score(valid), torch.zeros_like(js_divergence)
        )
        selected = _top_fraction(
            selection_score, valid, rho_value, used_scope
        )
    elif canonical_mode == "disagreement":
        selection_score = js_divergence
        selected = _top_fraction(
            selection_score, valid, rho_value, used_scope
        )
    else:
        # The primary PTC selector is trajectory-aware through JS.  Codebook
        # awareness enters through loss_prior below, keeping PTC and
        # disagreement masks exactly matched for a clean component ablation.
        # The explicit sequence-global diagnostic instead restores joint
        # a_q*JS ranking to study cross-codebook budget reallocation.
        selection_score = (
            js_divergence * prior.view(1, codebooks, 1)
            if used_scope == "sequence"
            else js_divergence
        )
        selected = _top_fraction(
            selection_score, valid, rho_value, used_scope
        )

    loss_prior = (
        prior
        if canonical_mode in {"codebook_only", "ptc"}
        else uniform_prior
    )
    effective_weights = (
        selected.to(dtype=torch.float32) * loss_prior.view(1, codebooks, 1)
    )
    numerator = (effective_weights * token_kl).sum()
    denominator = effective_weights.sum()
    if not allow_empty and not bool((denominator > 0).item()):
        raise ValueError(
            "selection contains no effective token; increase rho or pass "
            "allow_empty=True only for an explicit diagnostic"
        )
    loss = numerator / denominator.clamp_min(float(normalization_eps))

    valid_counts = valid.sum(dim=2)
    selected_counts = selected.sum(dim=2)
    selection_rates = torch.where(
        valid_counts > 0,
        selected_counts.float() / valid_counts.clamp_min(1).float(),
        torch.zeros_like(selected_counts, dtype=torch.float32),
    )
    total_valid = valid_counts.sum()
    total_selected = selected_counts.sum()
    realized_retention = torch.where(
        total_valid > 0,
        total_selected.float() / total_valid.clamp_min(1).float(),
        torch.zeros((), device=student.device, dtype=torch.float32),
    )

    return OPDLossOutput(
        loss=loss,
        token_kl=token_kl.detach(),
        js_divergence=js_divergence.detach(),
        selection_score=selection_score.detach(),
        selected_mask=selected.detach(),
        effective_weights=effective_weights.detach(),
        codebook_weights=loss_prior.detach(),
        numerator=numerator,
        effective_weight_sum=denominator.detach(),
        valid_counts_per_codebook=valid_counts.detach(),
        selected_counts_per_codebook=selected_counts.detach(),
        selection_rates_per_codebook=selection_rates.detach(),
        realized_retention=realized_retention.detach(),
        mode=canonical_mode,
        kl_direction=canonical_direction,
        selection_scope=used_scope,
        input_layout=canonical_layout,
        requested_rho=None if rho is None else float(rho),
        resolved_rho=rho_value,
        temperature=temperature_value,
    )


__all__ = ["OPDLossOutput", "ptc_opd_loss"]
