# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import pytest
import torch

from audiocraft.models.builders import get_debug_lm_model
from audiocraft.models.lm import CFGConditions, LMModel
from audiocraft.modules.conditioners import (
    ClassifierFreeGuidanceDropout,
    ConditioningAttributes,
)


def _conditions() -> tp.List[ConditioningAttributes]:
    return [
        ConditioningAttributes(text={"description": "bright piano"}),
        ConditioningAttributes(text={"description": "slow distorted guitar and drums"}),
    ]


def _condition_tensors(
    model: LMModel,
    conditions: tp.List[ConditioningAttributes],
    *,
    include_null: bool,
) -> CFGConditions:
    if include_null:
        null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
        conditions = conditions + null_conditions
    tokenized = model.condition_provider.tokenize(conditions)
    return model.condition_provider(tokenized)


def _sample_and_capture_forward_logits(
    model: LMModel,
    sequence: torch.Tensor,
    condition_tensors: CFGConditions,
    *,
    use_cfg: bool,
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    forwards = []

    def _capture(
        module: torch.nn.Module,
        inputs: tp.Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        forwards.append(output.detach().clone())

    handle = model.register_forward_hook(_capture)
    try:
        with model.streaming():
            next_token = model._sample_next_token(
                sequence,
                condition_tensors,
                model.get_streaming_state(),
                use_sampling=False,
                cfg_coef=1.0,
                use_cfg=use_cfg,
            )
    finally:
        handle.remove()

    assert len(forwards) == 1
    return forwards[0], next_token


def test_no_cfg_logits_and_greedy_tokens_match_cfg_coef_one() -> None:
    """Conditional-only decoding is the cfg_coef=1 mathematical endpoint."""
    torch.manual_seed(0)
    model = get_debug_lm_model(device="cpu").float().eval()
    conditions = _conditions()
    batch_size = len(conditions)
    sequence = torch.randint(
        low=0,
        high=model.card,
        size=(batch_size, model.num_codebooks, 3),
        dtype=torch.long,
    )

    conditional = _condition_tensors(model, conditions, include_null=False)
    cfg_batched = _condition_tensors(model, conditions, include_null=True)

    no_cfg_logits, no_cfg_token = _sample_and_capture_forward_logits(
        model, sequence, conditional, use_cfg=False
    )
    cfg_logits, cfg_token = _sample_and_capture_forward_logits(
        model, sequence, cfg_batched, use_cfg=True
    )

    # No-CFG performs one B-sized conditional forward. Standard one-pass CFG
    # performs one 2B-sized conditional+unconditional forward.
    assert no_cfg_logits.shape[0] == batch_size
    assert cfg_logits.shape[0] == 2 * batch_size
    cond_logits, uncond_logits = cfg_logits.split(batch_size, dim=0)
    cfg_coef_one_logits = uncond_logits + (cond_logits - uncond_logits)

    torch.testing.assert_close(
        no_cfg_logits.float(), cfg_coef_one_logits.float(), rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(no_cfg_token, cfg_token, rtol=0, atol=0)


def test_no_cfg_generation_preserves_pattern_shape_and_mask_checks() -> None:
    torch.manual_seed(1)
    model = get_debug_lm_model(device="cpu").float().eval()
    conditions = _conditions()
    prompt = torch.randint(
        low=0,
        high=model.card,
        size=(len(conditions), model.num_codebooks, 2),
        dtype=torch.long,
    )

    generated = model.generate(
        prompt=prompt,
        conditions=conditions,
        max_gen_len=6,
        use_sampling=False,
        use_cfg=False,
        # These settings must not construct tuple/triple CFG conditions when
        # explicit no-CFG mode is selected.
        cfg_coef=7.0,
        cfg_coef_beta=0.5,
        two_step_cfg=True,
        remove_prompts=True,
        check=True,
    )

    assert generated.shape == (len(conditions), model.num_codebooks, 4)
    assert generated.dtype == torch.long
    assert (generated >= 0).all()
    assert (generated <= model.card).all()


def test_use_cfg_default_matches_explicit_true() -> None:
    """The new flag must leave every legacy call on the old CFG path."""
    torch.manual_seed(2)
    model = get_debug_lm_model(device="cpu").float().eval()
    conditions = _conditions()
    kwargs = dict(
        conditions=conditions,
        max_gen_len=5,
        use_sampling=False,
        cfg_coef=2.5,
        check=True,
    )

    legacy = model.generate(**kwargs)
    explicit = model.generate(**kwargs, use_cfg=True)

    torch.testing.assert_close(legacy, explicit, rtol=0, atol=0)


def test_no_cfg_reuses_precomputed_condition_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(3)
    model = get_debug_lm_model(device="cpu").float().eval()
    conditions = _conditions()
    conditional = _condition_tensors(model, conditions, include_null=False)
    kwargs = dict(max_gen_len=5, use_sampling=False, use_cfg=False, check=True)

    recomputed = model.generate(conditions=conditions, **kwargs)

    def _unexpected_tokenize(*args: tp.Any, **kwargs: tp.Any) -> tp.NoReturn:
        raise AssertionError("precomputed condition tensors must bypass tokenization")

    monkeypatch.setattr(model.condition_provider, "tokenize", _unexpected_tokenize)
    reused = model.generate(condition_tensors=conditional, **kwargs)

    torch.testing.assert_close(recomputed, reused, rtol=0, atol=0)


def test_precomputed_cfg_tensors_match_recomputed_and_validate_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(4)
    model = get_debug_lm_model(device="cpu").float().eval()
    conditions = _conditions()
    cfg_batched = _condition_tensors(model, conditions, include_null=True)
    kwargs = dict(
        max_gen_len=5,
        use_sampling=False,
        use_cfg=True,
        cfg_coef=2.5,
        check=True,
    )

    recomputed = model.generate(conditions=conditions, **kwargs)

    def _unexpected_tokenize(*args: tp.Any, **kwargs: tp.Any) -> tp.NoReturn:
        raise AssertionError("precomputed CFG tensors must bypass tokenization")

    monkeypatch.setattr(model.condition_provider, "tokenize", _unexpected_tokenize)
    reused = model.generate(condition_tensors=cfg_batched, **kwargs)

    torch.testing.assert_close(recomputed, reused, rtol=0, atol=0)

    odd_batched = {
        name: (embedding[:-1], mask[:-1])
        for name, (embedding, mask) in cfg_batched.items()
    }
    with pytest.raises(ValueError, match="batch must be even"):
        model.generate(
            condition_tensors=odd_batched,
            use_cfg=True,
            max_gen_len=2,
        )

    with pytest.raises(ValueError, match="mutually exclusive"):
        model.generate(
            conditions=conditions,
            condition_tensors=cfg_batched,
            use_cfg=True,
            max_gen_len=2,
        )
    with pytest.raises(ValueError, match="inconsistent batch sizes"):
        model.generate(
            num_samples=1,
            condition_tensors=cfg_batched,
            use_cfg=True,
            max_gen_len=2,
        )


def test_no_cfg_rejects_two_step_condition_tuple() -> None:
    model = get_debug_lm_model(device="cpu").float().eval()
    conditions = _conditions()
    conditional = _condition_tensors(model, conditions, include_null=False)
    sequence = torch.zeros(
        (len(conditions), model.num_codebooks, 1), dtype=torch.long
    )

    with pytest.raises(AssertionError):
        model._sample_next_token(
            sequence,
            (conditional, conditional),
            {},
            use_sampling=False,
            use_cfg=False,
        )
