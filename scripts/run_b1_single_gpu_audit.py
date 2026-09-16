#!/usr/bin/env python3
"""Formal single-H20, real-MusicGen audit for B1 correctness items 1--6.

The output is intentionally a *partial* B1 artifact.  It can never claim that
B1 is complete because the literal eight-rank/reference check (B1.9) and the
500-update PTC stability run (B1.11) are separate gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


WORKPACK_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKPACK_ROOT / "src"
SCRIPTS_ROOT = WORKPACK_ROOT / "scripts"
for path in (str(SRC_ROOT), str(SCRIPTS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch
import torch.nn.functional as F

import b1_prestability_contract as contract
import run_disagreement_probe as probe
from ptc_opd import ptc_opd_loss
from ptc_opd.cfg_decision import verify_cfg_scale_decision
from ptc_opd.codec_prior_artifact import (
    load_codec_prior_artifact,
    verify_local_codec_snapshot,
)
from ptc_opd.losses import _top_fraction
from ptc_opd.sampling import keyed_random_scores
from ptc_opd.train_utils import hash_module_state


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audiocraft-root", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--cfg-scale-decision-dir", type=Path, required=True)
    parser.add_argument("--codebook-prior-artifact-dir", type=Path, required=True)
    parser.add_argument("--a2-probe-dir", type=Path, required=True)
    parser.add_argument("--node3-evidence-dir", type=Path, required=True)
    parser.add_argument("--t5-closure-artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    header = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    array = tensor.numpy()
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _relative_scalar_error(observed: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = abs(float(observed.detach().item()) - float(reference.detach().item()))
    denominator = max(abs(float(reference.detach().item())), 1.0e-12)
    return numerator / denominator


def _relative_l2_error(observed: torch.Tensor, reference: torch.Tensor) -> float:
    left = observed.detach().double()
    right = reference.detach().double()
    numerator = torch.linalg.vector_norm(left - right)
    denominator = torch.linalg.vector_norm(right).clamp_min(1.0e-12)
    return float((numerator / denominator).item())


def _direct_forward_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Direct forward-KL oracle over ONLY the valid cells.

    Design (pilot ruling 2026-08-21T08:30Z, R1'-Gather, classification
    ``yellow / audit-reference-harness-only``):

    * AudioCraft's ``lm.py`` intentionally fills delayed-pattern invalid
      logit positions with ``NaN`` so that any downstream consumer that
      forgets to mask them will observe a hard failure rather than silently
      correct results.  See ``vendor/audiocraft/audiocraft/models/lm.py``.
    * The production loss ``ptc_opd_loss`` sanitises invalid rows *before*
      softmax (see ``src/ptc_opd/losses.py``); its reference oracle used to
      call ``F.log_softmax`` on the full lattice and then boolean-index the
      valid rows out of ``token_kl`` -- which lets ``NaN`` leak into the
      backward pass, because the ``.mean()`` reduction distributes gradient
      through every element that participated in the forward compute graph.
      A ``torch.where`` cleanup *after* the full-lattice softmax cannot
      remove that leak (``0 * NaN == NaN`` in IEEE 754).
    * The correct oracle therefore gathers the valid rows FIRST, softmaxes
      only those rows, refuses any non-finite value at a valid cell, and
      scatters the resulting per-cell KL values back into a zero-filled
      lattice.  Invalid positions never enter the graph, so their gradient
      contribution is exactly zero.

    The change is confined to this audit-reference harness.  It does NOT
    modify B1.2 thresholds, valid-lattice construction, selection scope,
    ``rho``, or the KL direction; and it does NOT modify the production
    ``ptc_opd_loss`` under ``src/ptc_opd/losses.py``.
    """

    student_valid = student_logits.float()[valid_mask]
    teacher_valid = teacher_logits.detach().float()[valid_mask]

    if student_valid.numel() == 0:
        raise ValueError("direct KL received no valid cells")
    if not bool(torch.isfinite(student_valid).all().item()):
        raise FloatingPointError("student logits are non-finite at a valid cell")
    if not bool(torch.isfinite(teacher_valid).all().item()):
        raise FloatingPointError("teacher logits are non-finite at a valid cell")

    student_logp = F.log_softmax(student_valid, dim=-1)
    teacher_logp = F.log_softmax(teacher_valid, dim=-1)
    valid_kl = (
        teacher_logp.exp() * (teacher_logp - student_logp)
    ).sum(dim=-1)

    token_kl = torch.zeros(
        valid_mask.shape,
        dtype=valid_kl.dtype,
        device=valid_kl.device,
    ).masked_scatter(valid_mask, valid_kl)

    return valid_kl.mean(), token_kl


def _loss_and_logit_gradient(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    mode: str,
    rho: Optional[float] = None,
    codebook_weights: Optional[torch.Tensor] = None,
    random_scores: Optional[torch.Tensor] = None,
) -> Tuple[Any, torch.Tensor]:
    leaf = student_logits.detach().float().clone().requires_grad_(True)
    output = ptc_opd_loss(
        leaf,
        teacher_logits.detach().float(),
        valid_mask=valid_mask,
        mode=mode,
        rho=rho,
        codebook_weights=codebook_weights,
        random_scores=random_scores,
        kl_direction="forward",
        selection_scope="protocol",
        layout="BQTV",
        temperature=1.0,
        check_finite=True,
    )
    gradient = torch.autograd.grad(output.loss, leaf, retain_graph=False)[0]
    return output, gradient


def _direct_loss_and_gradient(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    leaf = student_logits.detach().float().clone().requires_grad_(True)
    loss, token_kl = _direct_forward_kl_loss(leaf, teacher_logits, valid_mask)
    gradient = torch.autograd.grad(loss, leaf, retain_graph=False)[0]
    return loss.detach(), gradient, token_kl.detach()


def _require_runtime() -> Dict[str, Any]:
    contaminated = sorted(
        name
        for name in (
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "SLURM_JOB_ID",
        )
        if os.environ.get(name)
    )
    if contaminated:
        raise RuntimeError(
            "single-GPU audit inherited distributed/scheduler variables: {}".format(
                contaminated
            )
        )
    if torch.__version__ != "2.1.0+cu121" or torch.version.cuda != "12.1":
        raise RuntimeError("formal B1 audit requires torch 2.1.0+cu121 / CUDA 12.1")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "set CUDA_VISIBLE_DEVICES to one H20; formal single audit requires exactly one visible GPU"
        )
    device = torch.device("cuda", 0)
    name = torch.cuda.get_device_name(device)
    capability = list(torch.cuda.get_device_capability(device))
    if name != "NVIDIA H20" or capability != [9, 0]:
        raise RuntimeError("formal single audit requires NVIDIA H20 capability [9,0]")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("formal single audit requires native CUDA BF16 support")
    if torch.is_autocast_enabled():
        raise RuntimeError("outer CUDA autocast must be disabled")
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "device": "cuda:0",
        "gpu_name": name,
        "gpu_capability": capability,
        "visible_cuda_devices": 1,
        "bf16_supported": True,
        "scoring_autocast": False,
        "divergence_dtype": "torch.float32",
    }


def _prepare_inputs_and_upstreams(args: argparse.Namespace) -> Dict[str, Any]:
    if args.output_dir.expanduser().resolve().exists():
        raise FileExistsError("refusing to overwrite --output-dir")
    manifest, dev_manifest = probe.validate_probe_is_exact_dev_subset(
        args.probe_manifest.resolve(), args.dev_manifest.resolve()
    )
    if len(manifest) != 256 or len(dev_manifest) != 300:
        raise ValueError("formal B1 audit requires the sealed 256/300 probe/dev manifests")
    checkpoint_identity = probe.verify_checkpoint_snapshot(args.checkpoint)
    source_identity = probe.verify_audiocraft_source(args.audiocraft_root)
    cfg = verify_cfg_scale_decision(args.cfg_scale_decision_dir)
    if cfg.decision_file_sha256 != contract.EXPECTED_SMALL_CFG_DECISION_SHA256:
        raise ValueError("small CFG decision file differs from the retained pin")
    if cfg.selected_cfg_scale != 5.0:
        raise ValueError("retained MusicGen-small CFG scale must equal 5.0")
    if cfg.generation_identity.get("model_id") != "facebook/musicgen-small":
        raise ValueError("CFG decision is not MusicGen-small")
    dev_hash = probe.sha256_file(args.dev_manifest.resolve())
    probe_hash = probe.sha256_file(args.probe_manifest.resolve())
    if cfg.generation_identity.get("manifest_sha256") != dev_hash:
        raise ValueError("CFG decision is not bound to the supplied dev manifest")
    for key in ("checkpoint_sha256", "state_dict_sha256", "compression_state_dict_sha256"):
        if cfg.generation_identity.get(key) != checkpoint_identity.get(key):
            raise ValueError("checkpoint/CFG identity differs at {}".format(key))
    for key in ("audiocraft_base_commit", "audiocraft_source_sha256", "audiocraft_lm_sha256"):
        if cfg.generation_identity.get(key) != source_identity.get(key):
            raise ValueError("AudioCraft/CFG identity differs at {}".format(key))

    prior = load_codec_prior_artifact(args.codebook_prior_artifact_dir)
    local_codec = verify_local_codec_snapshot(prior)
    if prior.artifact_seal_sha256 != contract.EXPECTED_A1_R2_ARTIFACT_SEAL_SHA256:
        raise ValueError("A1-R2 artifact seal differs from the retained milestone")
    if prior.checkpoint_sha256 != checkpoint_identity["compression_state_dict_sha256"]:
        raise ValueError("A1 codec differs from the supplied MusicGen checkpoint")

    a2 = probe.verify_probe_artifact(args.a2_probe_dir, require_primary=True)
    if a2["artifact_seal_sha256"] != contract.EXPECTED_A2_PROBE_SEAL_SHA256:
        raise ValueError("A2 probe seal differs from the retained milestone")
    a2_config = a2["metadata"]["scientific_config"]
    if a2_config.get("probe_manifest_sha256") != probe_hash:
        raise ValueError("A2/probe manifest identity differs")
    if a2_config.get("dev_manifest_sha256") != dev_hash:
        raise ValueError("A2/dev manifest identity differs")
    if a2_config.get("checkpoint_identity") != checkpoint_identity:
        raise ValueError("A2/checkpoint identity differs")
    if a2_config.get("audiocraft_identity") != source_identity:
        raise ValueError("A2/AudioCraft identity differs")
    if (
        a2_config.get("codebook_prior_artifact_identity_sha256")
        != prior.identity_sha256
        or a2_config.get("codebook_prior_artifact_seal_sha256")
        != prior.artifact_seal_sha256
    ):
        raise ValueError("A2/A1-R2 prior identity differs")
    if a2_config.get("verified_local_codec_snapshot") != local_codec:
        raise ValueError("A2/A1-R2 local codec verification differs")
    if (
        a2_config.get("cfg_scale_decision_file_sha256")
        != cfg.decision_file_sha256
        or a2_config.get("cfg_scale_decision_payload_sha256")
        != cfg.decision_payload_sha256
        or a2_config.get("teacher_cfg_scale") != 5.0
        or a2_config.get("loaded_t5_identity_sha256")
        != cfg.generation_identity["loaded_t5_identity_sha256"]
    ):
        raise ValueError("A2/retained CFG or T5 identity differs")
    node3 = contract.validate_node3_evidence(args.node3_evidence_dir)
    t5 = contract.validate_t5_closure_artifact(args.t5_closure_artifact_dir)
    audit_records = manifest[: contract.DISTRIBUTED_WORLD_SIZE]
    return {
        "manifest": manifest,
        "audit_records": audit_records,
        "checkpoint_identity": checkpoint_identity,
        "audiocraft_identity": source_identity,
        "cfg_decision": {
            "selected_cfg_scale": cfg.selected_cfg_scale,
            "decision_file_sha256": cfg.decision_file_sha256,
            "decision_payload_sha256": cfg.decision_payload_sha256,
            "scientific_config_sha256": cfg.scientific_config_sha256,
            "loaded_t5_identity_sha256": cfg.generation_identity[
                "loaded_t5_identity_sha256"
            ],
        },
        "probe_manifest_sha256": probe_hash,
        "dev_manifest_sha256": dev_hash,
        "a1_identity_sha256": prior.identity_sha256,
        "a1_artifact_seal_sha256": prior.artifact_seal_sha256,
        "verified_local_codec_snapshot": local_codec,
        "a2_artifact_seal_sha256": a2["artifact_seal_sha256"],
        "a2_scientific_config_sha256": a2["metadata"][
            "scientific_config_sha256"
        ],
        "node3": node3,
        "t5_closure": t5,
    }


def _move_external_t5(lm: torch.nn.Module, device: torch.device) -> None:
    provider = getattr(lm, "condition_provider", None)
    if not isinstance(provider, torch.nn.Module):
        raise TypeError("MusicGen LM has no nn.Module condition_provider")
    for module in provider.modules():
        external = module.__dict__.get("t5")
        if isinstance(external, torch.nn.Module):
            external.to(device)
            external.requires_grad_(False)
            external.eval()
            if hasattr(module, "device"):
                module.device = str(device)


def _prepare_model(lm: torch.nn.Module, device: torch.device) -> None:
    parameters = list(lm.parameters())
    if not parameters or any(item.dtype != torch.float32 for item in parameters):
        raise RuntimeError("MusicGen LM did not load entirely as CPU FP32")
    if any(item.device.type != "cpu" for item in parameters):
        raise RuntimeError("MusicGen LM did not load on CPU")
    lm.to(device)
    lm.requires_grad_(False)
    lm.eval()
    provider = getattr(lm, "condition_provider", None)
    if not isinstance(provider, torch.nn.Module):
        raise TypeError("MusicGen LM has no condition provider")
    provider.requires_grad_(False)
    provider.eval()
    _move_external_t5(lm, device)


def _condition_tensors(
    lm: torch.nn.Module,
    prompts: Sequence[str],
    conditioning_class: Any,
    dropout_class: Any,
) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
    conditions = probe._build_conditions(prompts, conditioning_class)
    null_conditions = probe._null_conditions(conditions, dropout_class)
    batched = probe._condition_tensors(lm, conditions + null_conditions)
    conditional = probe._slice_condition_tensors(batched, 0, len(conditions))
    null = probe._slice_condition_tensors(batched, len(conditions), 2 * len(conditions))
    return conditional, null


def _generate_fixed_codes(
    lm: torch.nn.Module,
    records: Sequence[Mapping[str, Any]],
    conditioning_class: Any,
    dropout_class: Any,
) -> torch.Tensor:
    batches: List[torch.Tensor] = []
    device = next(lm.parameters()).device
    with probe._rollout_rng(contract.ROLLOUT_SEED, device):
        for start in range(0, len(records), contract.SINGLE_BATCH_SIZE):
            block = records[start : start + contract.SINGLE_BATCH_SIZE]
            prompts = [str(item["prompt"]) for item in block]
            conditional, _ = _condition_tensors(
                lm, prompts, conditioning_class, dropout_class
            )
            codes = probe._rollout(
                lm,
                conditional,
                max_gen_len=contract.PRIMARY_FRAMES,
                temperature=contract.ROLLOUT_TEMPERATURE,
                top_k=contract.ROLLOUT_TOP_K,
                top_p=contract.ROLLOUT_TOP_P,
                bf16=True,
            )
            if (
                codes.dtype != torch.long
                or tuple(codes.shape) != (len(block), 4, 500)
                or not bool(((codes >= 0) & (codes < 2048)).all().item())
            ):
                raise RuntimeError("fixed audit rollout violates [B,4,500] codec contract")
            batches.append(codes.detach().cpu())
    output = torch.cat(batches, dim=0)
    if tuple(output.shape) != (contract.DISTRIBUTED_WORLD_SIZE, 4, 500):
        raise AssertionError("fixed audit input count differs")
    return output


def _score(
    student_lm: torch.nn.Module,
    teacher_lm: torch.nn.Module,
    codes: torch.Tensor,
    prompts: Sequence[str],
    conditioning_class: Any,
    dropout_class: Any,
    cfg_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    conditional, null = _condition_tensors(
        student_lm, prompts, conditioning_class, dropout_class
    )
    return probe._score_probe_no_grad(
        student_lm,
        teacher_lm,
        codes,
        conditional,
        conditional,
        null,
        teacher_cfg_scale=cfg_scale,
        bf16=False,
    )


def _expected_counts(valid: torch.Tensor, rho: float) -> torch.Tensor:
    counts = valid.sum(dim=2)
    return torch.ceil(counts.float() * rho).long()


def _run_cases(
    student_lm: torch.nn.Module,
    teacher_lm: torch.nn.Module,
    all_codes_cpu: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    conditioning_class: Any,
    dropout_class: Any,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    device = next(student_lm.parameters()).device
    codes = all_codes_cpu[:2].to(device)
    prompts = [str(item["prompt"]) for item in records[:2]]
    sample_ids = [str(item["sample_id"]) for item in records[:2]]

    student_cfg1, teacher_cfg1, valid_cfg1 = _score(
        student_lm,
        teacher_lm,
        codes,
        prompts,
        conditioning_class,
        dropout_class,
        1.0,
    )
    if student_cfg1.dtype != torch.float32 or teacher_cfg1.dtype != torch.float32:
        raise RuntimeError("B1.1 scoring did not remain FP32")
    _, cfg1_token_kl = _direct_forward_kl_loss(
        student_cfg1.float(), teacher_cfg1.float(), valid_cfg1
    )
    cfg1_values = cfg1_token_kl[valid_cfg1]
    cfg1_mean = float(cfg1_values.mean().item())
    cfg1_max = float(cfg1_values.max().item())
    if not (
        math.isfinite(cfg1_mean)
        and math.isfinite(cfg1_max)
        and cfg1_mean < contract.CFG1_MEAN_KL_LT
        and cfg1_max < contract.CFG1_MAX_VALID_KL_LT
    ):
        raise AssertionError("B1.1 CFG=1 real-checkpoint KL threshold failed")

    student, teacher, valid = _score(
        student_lm,
        teacher_lm,
        codes,
        prompts,
        conditioning_class,
        dropout_class,
        5.0,
    )
    expected_valid_counts = torch.tensor(
        [[500, 499, 498, 497], [500, 499, 498, 497]],
        device=device,
    )
    if not torch.equal(valid.sum(dim=2), expected_valid_counts):
        raise AssertionError("real MusicGen valid lattice differs")

    direct_loss, direct_grad, _ = _direct_loss_and_gradient(student, teacher, valid)
    uniform, uniform_grad = _loss_and_logit_gradient(
        student, teacher, valid, mode="uniform"
    )
    ptc_all, ptc_all_grad = _loss_and_logit_gradient(
        student,
        teacher,
        valid,
        mode="ptc",
        rho=1.0,
        codebook_weights=torch.ones(4, device=device),
    )
    b12_loss_error = max(
        _relative_scalar_error(uniform.loss, direct_loss),
        _relative_scalar_error(ptc_all.loss, direct_loss),
    )
    b12_grad_error = max(
        _relative_l2_error(uniform_grad, direct_grad),
        _relative_l2_error(ptc_all_grad, direct_grad),
    )
    if not torch.equal(uniform.selected_mask, valid) or not torch.equal(
        ptc_all.selected_mask, valid
    ):
        raise AssertionError("B1.2 all-selected gate differs from valid lattice")
    if not (
        b12_loss_error < contract.RELATIVE_LOSS_ERROR_LT
        and b12_grad_error < contract.RELATIVE_GRADIENT_ERROR_LT
    ):
        raise AssertionError("B1.2 uniform equivalence tolerance failed")

    disagreement, disagreement_grad = _loss_and_logit_gradient(
        student, teacher, valid, mode="disagreement", rho=0.5
    )
    ptc_equal, ptc_equal_grad = _loss_and_logit_gradient(
        student,
        teacher,
        valid,
        mode="ptc",
        rho=0.5,
        codebook_weights=torch.ones(4, device=device),
    )
    b13_gate_exact = torch.equal(disagreement.selected_mask, ptc_equal.selected_mask)
    b13_loss_error = _relative_scalar_error(ptc_equal.loss, disagreement.loss)
    b13_grad_error = _relative_l2_error(ptc_equal_grad, disagreement_grad)
    if not (
        b13_gate_exact
        and b13_loss_error < contract.RELATIVE_LOSS_ERROR_LT
        and b13_grad_error < contract.RELATIVE_GRADIENT_ERROR_LT
    ):
        raise AssertionError("B1.3 PTC/disagreement equal-weight equivalence failed")

    random_scores = keyed_random_scores(
        sample_ids,
        4,
        500,
        contract.RANDOM_RUN_SEED,
        contract.RANDOM_OPTIMIZER_STEP,
        random_namespace=contract.RANDOM_NAMESPACE,
        device=device,
    )
    random_output, _ = _loss_and_logit_gradient(
        student,
        teacher,
        valid,
        mode="random_stratified",
        rho=0.5,
        random_scores=random_scores,
    )
    repeated, _ = _loss_and_logit_gradient(
        student, teacher, valid, mode="disagreement", rho=0.5
    )
    expected_counts = _expected_counts(valid, 0.5)
    if not (
        torch.equal(random_output.selected_counts_per_codebook, expected_counts)
        and torch.equal(disagreement.selected_counts_per_codebook, expected_counts)
        and torch.equal(disagreement.selected_mask, repeated.selected_mask)
    ):
        raise AssertionError("B1.4 selected counts/repeatability failed")
    # ------------------------------------------------------------------
    # B1.4 EXACT-ZERO-SCORE STABLE-TIE fixture (pilot ruling 2026-08-24
    # CST β = B1.4 exact-tie audit fixture defect; classification
    # ``yellow / audit-reference-harness-only``).
    #
    # Design rationale.  The frozen ``disagreement`` selector ranks by the
    # detached JS score via ``torch.argsort(descending=True, stable=True)``
    # inside ``_top_fraction``.  Only when scores are BITWISE-EQUAL does
    # the stable sort fall back to the ascending original-index order,
    # which for a valid-indices vector coincides with earliest-time order.
    #
    # A previous version of this fixture used
    # ``_loss_and_logit_gradient(student, student.detach(), valid, ...)``
    # to try to construct a full tie by setting the teacher equal to the
    # student.  Mathematically ``JS(P, P) == 0``; however the production
    # loss computes the JS through fp32 ``logaddexp(x, x) - log(2.0)`` on
    # normalized logits, which produces position-dependent tiny positive
    # residuals in fp32.  The old fixture therefore never realised
    # numerical ties, so the stable-sort tie-breaker was never actually
    # exercised.  The read-only sidecar
    # ``ptc_local/bin/ptc_stage1_b1_tie_diagnostic.py`` documents this
    # empirically: ``self_js_unique_count > 1`` and
    # ``self_js_exact_zero_count < n_valid``.
    #
    # The corrected fixture feeds a true bitwise-zero score of shape
    # ``valid_mask.shape`` directly through the frozen ``_top_fraction``
    # selector at rho=0.5 with scope=``codebook``.  This is the only
    # score for which the stable-tie tie-breaker is guaranteed to be
    # reached at EVERY valid cell.  The expected mask keeps the earliest
    # ``ceil(0.5 * n_valid)`` positions per (batch, codebook) valid line,
    # which is exactly what the stable sort must produce.
    #
    # Confinement: this change is confined to this reference-harness
    # fixture.  It does NOT modify ``src/ptc_opd/losses.py``, does NOT
    # modify ``configs/stage1_matrix.yaml`` (whose L494
    # ``earliest_valid_time_within_each_sample_codebook`` selector
    # belongs to ``prefix50``, not ``disagreement50``), does NOT change
    # ``disagreement50`` from the frozen ``detached_js_topk``, does NOT
    # add any epsilon-tie quantisation to the JS score, and does NOT
    # alter the stable-sort logic in ``_top_fraction``.
    zero_js_score = torch.zeros_like(valid, dtype=torch.float32)
    tie_selected = _top_fraction(
        zero_js_score,
        valid,
        0.5,
        "codebook",
    )
    expected_tie = torch.zeros_like(valid)
    for batch_index in range(valid.shape[0]):
        for q_index in range(valid.shape[1]):
            valid_indices = torch.nonzero(
                valid[batch_index, q_index], as_tuple=False
            ).flatten()
            keep = int(math.ceil(0.5 * int(valid_indices.numel())))
            expected_tie[batch_index, q_index, valid_indices[:keep]] = True
    if not torch.equal(tie_selected, expected_tie):
        raise AssertionError(
            "B1.4 exact-zero-score stable tie did not select earliest valid times"
        )

    # ------------------------------------------------------------------
    # B1.5 delayed-pattern-aware padding/NaN/zero-gradient exact contract
    # (pilot RULING #3 2026-08-24 CST ε = B1.5 padding-mutation mask
    # defect; classification ``yellow / audit-reference-harness-only``).
    #
    # Design rationale.  Producer must probe two contracts:
    #
    #   (P1) mutating the raw padding-suffix codec tokens at raw time
    #        positions ``t >= L`` (per batch length ``L``, with the same
    #        ``L`` for all four codebooks) MUST NOT change student /
    #        teacher logits, uniform loss, or logit gradient at cells
    #        that MusicGen's frozen delayed-pattern actually predicts
    #        from those raw positions;
    #   (P2) poisoning invalid cells with NaN / Inf MUST NOT influence
    #        the loss, the selected mask, effective weights, or the
    #        gradient sanitisation contract in ``ptc_opd_loss``.
    #
    # A previous version of this block truncated all four codebooks at
    # the same length ``L`` and used ``padded_valid = valid & (time < L)``
    # as both the invariance mask AND the poison-injection mask.  Under
    # the frozen MusicGen ``DelayedPatternProvider`` with delays
    # ``[0, 1, 2, 3]``, the prediction actually corresponding to raw
    # codes of length ``L`` per (batch, codebook q) is
    # ``t + delays[q] < L``, i.e. per-codebook lengths
    # ``[L, L-1, L-2, L-3]``.  The delay-blind mask therefore included
    # cells ``(q1, t=L-1)``, ``(q2, t in {L-2, L-1})``,
    # ``(q3, t in {L-3, L-2, L-1})`` where causal AR flow LEGITIMATELY
    # propagates mutated raw suffix tokens.  This is ordinary AR
    # causality, not a padding leak, and pilot ruled it out of scope
    # for AudioCraft/``ptc_opd_loss``.
    #
    # The corrected fixture:
    #   * keeps the raw suffix mutation unchanged (same ``codes.clone``
    #     followed by ``+ PADDING_TOKEN_MUTATION_OFFSET mod 2048`` at
    #     ``t >= L``), and
    #   * verifies logit / loss / gradient invariance ONLY on cells
    #     inside the causal-prediction domain
    #     ``padded_valid = valid & ((time + delays) < lengths)``.
    #
    # The frozen ``DelayedPatternProvider`` delays are checked against
    # ``[0, 1, 2, 3]`` and the per-(batch, codebook) causal-valid counts
    # are asserted against the pilot-mandated target
    # ``[[487, 486, 485, 484], [461, 460, 459, 458]]`` (total 3780) so
    # any future upstream change is caught here.  Diagnostic evidence
    # is enriched with ``padding_semantics`` and ``causal_valid_counts``
    # (see the sealed cases dictionary below).
    #
    # Confinement (pilot RULING #3): this change is confined to this
    # reference-harness fixture.  It does NOT modify
    # ``src/ptc_opd/losses.py``, does NOT modify ``audiocraft/``, does
    # NOT modify ``configs/stage1_matrix.yaml``, does NOT alter numeric
    # thresholds, and does NOT change ``SINGLE_PADDING_LENGTHS`` or
    # ``PADDING_TOKEN_MUTATION_OFFSET``.
    time = torch.arange(500, device=device).view(1, 1, 500)
    lengths = torch.tensor(
        contract.SINGLE_PADDING_LENGTHS, device=device
    ).view(2, 1, 1)
    provider_delays_list = list(student_lm.pattern_provider.delays)
    if provider_delays_list != [0, 1, 2, 3]:
        raise AssertionError(
            "unexpected MusicGen delay pattern: {} (expected [0, 1, 2, 3])".format(
                provider_delays_list
            )
        )
    delays = torch.tensor(
        provider_delays_list, device=device
    ).view(1, 4, 1)
    padded_valid = valid & ((time + delays) < lengths)
    causal_valid_counts = padded_valid.sum(dim=2).tolist()
    if causal_valid_counts != [[487, 486, 485, 484], [461, 460, 459, 458]]:
        raise AssertionError(
            "B1.5 delayed-pattern causal_valid_counts differ from pilot target"
        )
    mutated_codes = codes.clone()
    for batch_index, length in enumerate(contract.SINGLE_PADDING_LENGTHS):
        mutated_codes[batch_index, :, length:] = (
            mutated_codes[batch_index, :, length:]
            + contract.PADDING_TOKEN_MUTATION_OFFSET
        ) % 2048
    mutated_student, mutated_teacher, mutated_delay_valid = _score(
        student_lm,
        teacher_lm,
        mutated_codes,
        prompts,
        conditioning_class,
        dropout_class,
        5.0,
    )
    if not torch.equal(mutated_delay_valid, valid):
        raise AssertionError("padding-token mutation changed the delay mask")
    valid_student_logits_exact = torch.equal(
        student[padded_valid], mutated_student[padded_valid]
    )
    valid_teacher_logits_exact = torch.equal(
        teacher[padded_valid], mutated_teacher[padded_valid]
    )
    baseline_padded, baseline_padded_grad = _loss_and_logit_gradient(
        student, teacher, padded_valid, mode="uniform"
    )
    mutated_padded, mutated_padded_grad = _loss_and_logit_gradient(
        mutated_student, mutated_teacher, padded_valid, mode="uniform"
    )
    padding_loss_exact = torch.equal(
        baseline_padded.loss.detach(), mutated_padded.loss.detach()
    )
    padding_valid_gradient_exact = torch.equal(
        baseline_padded_grad[padded_valid], mutated_padded_grad[padded_valid]
    )
    poisoned_student = student.detach().float().clone()
    poisoned_teacher = teacher.detach().float().clone()
    poisoned_student[~padded_valid] = float("nan")
    poisoned_teacher[~padded_valid] = float("inf")
    poisoned, poisoned_grad = _loss_and_logit_gradient(
        poisoned_student,
        poisoned_teacher,
        padded_valid,
        mode="uniform",
    )
    poisoned_loss_exact = torch.equal(
        baseline_padded.loss.detach(), poisoned.loss.detach()
    )
    invalid_gradient_zero = torch.equal(
        poisoned_grad[~padded_valid],
        torch.zeros_like(poisoned_grad[~padded_valid]),
    )
    invalid_selected_zero = not bool(poisoned.selected_mask[~padded_valid].any().item())
    invalid_effective_weight_zero = torch.equal(
        poisoned.effective_weights[~padded_valid],
        torch.zeros_like(poisoned.effective_weights[~padded_valid]),
    )
    if not all(
        (
            valid_student_logits_exact,
            valid_teacher_logits_exact,
            padding_loss_exact,
            padding_valid_gradient_exact,
            poisoned_loss_exact,
            invalid_gradient_zero,
            invalid_selected_zero,
            invalid_effective_weight_zero,
        )
    ):
        raise AssertionError("B1.5 padding/NaN/zero-gradient exact contract failed")

    pattern = student_lm.pattern_provider.get_pattern(500)
    sentinel = (
        torch.arange(4).view(1, 4, 1) * 500
        + torch.arange(500).view(1, 1, 500)
    ).long()
    sequence, _, _ = pattern.build_pattern_sequence(
        sentinel, special_token=2048, keep_only_valid_steps=True
    )
    reverted, revert_indices, revert_mask = pattern.revert_pattern_sequence(
        sequence, special_token=2048, keep_only_valid_steps=True
    )
    expected_mask_cpu = valid[0].detach().cpu()
    sentinel_exact = torch.equal(
        reverted[0][expected_mask_cpu], sentinel[0][expected_mask_cpu]
    )
    actual_sequence, _, _ = pattern.build_pattern_sequence(
        all_codes_cpu[:1], special_token=2048, keep_only_valid_steps=True
    )
    actual_reverted, _, actual_mask = pattern.revert_pattern_sequence(
        actual_sequence, special_token=2048, keep_only_valid_steps=True
    )
    actual_exact = torch.equal(
        actual_reverted[0][expected_mask_cpu],
        all_codes_cpu[0][expected_mask_cpu],
    )
    if not (
        torch.equal(revert_mask.cpu(), expected_mask_cpu)
        and torch.equal(actual_mask.cpu(), expected_mask_cpu)
        and sentinel_exact
        and actual_exact
        and int(revert_mask.sum().item()) == 1994
    ):
        raise AssertionError("B1.6 real pattern roundtrip identity failed")

    cases = {
        contract.SINGLE_CASES[0]: {
            "status": "passed",
            "compute_dtype": "torch.float32",
            "valid_cell_count": int(valid_cfg1.sum().item()),
            "mean_kl": cfg1_mean,
            "max_valid_cell_kl": cfg1_max,
            "mean_kl_threshold_exclusive": contract.CFG1_MEAN_KL_LT,
            "max_kl_threshold_exclusive": contract.CFG1_MAX_VALID_KL_LT,
        },
        contract.SINGLE_CASES[1]: {
            "status": "passed",
            "all_selected_gate_exact": True,
            "relative_loss_error": b12_loss_error,
            "relative_gradient_l2_error": b12_grad_error,
            "relative_loss_threshold_exclusive": contract.RELATIVE_LOSS_ERROR_LT,
            "relative_gradient_threshold_exclusive": contract.RELATIVE_GRADIENT_ERROR_LT,
        },
        contract.SINGLE_CASES[2]: {
            "status": "passed",
            "gate_exact": b13_gate_exact,
            "relative_loss_error": b13_loss_error,
            "relative_gradient_l2_error": b13_grad_error,
            "relative_loss_threshold_exclusive": contract.RELATIVE_LOSS_ERROR_LT,
            "relative_gradient_threshold_exclusive": contract.RELATIVE_GRADIENT_ERROR_LT,
        },
        contract.SINGLE_CASES[3]: {
            "status": "passed",
            "expected_counts_per_sample_codebook": expected_counts.cpu().tolist(),
            "random_counts_exact": True,
            "js_counts_exact": True,
            "repeated_js_gate_exact": True,
            "stable_zero_js_tie_gate_exact": True,
            "js_gate_sha256": _tensor_sha256(disagreement.selected_mask),
            "random_gate_sha256": _tensor_sha256(random_output.selected_mask),
        },
        contract.SINGLE_CASES[4]: {
            "status": "passed",
            "padding_lengths": list(contract.SINGLE_PADDING_LENGTHS),
            "padding_semantics": "raw_suffix_mutation__delayed_pattern_valid_scoring",
            "provider_delays": provider_delays_list,
            "causal_valid_counts": causal_valid_counts,
            "valid_student_logits_exact_after_input_mutation": True,
            "valid_teacher_logits_exact_after_input_mutation": True,
            "loss_exact_after_input_mutation": True,
            "valid_gradient_exact_after_input_mutation": True,
            "loss_exact_with_invalid_nan_inf": True,
            "invalid_selected_exact_zero": True,
            "invalid_effective_weight_exact_zero": True,
            "invalid_logit_gradient_exact_zero": True,
        },
        contract.SINGLE_CASES[5]: {
            "status": "passed",
            "pattern_provider_type": type(student_lm.pattern_provider).__name__,
            "sequence_steps": int(sequence.shape[-1]),
            "valid_coordinate_count": int(revert_mask.sum().item()),
            "sentinel_qt_identity_exact": sentinel_exact,
            "real_rollout_identity_exact": actual_exact,
            "revert_indices_sha256": _tensor_sha256(revert_indices),
            "revert_mask_sha256": _tensor_sha256(revert_mask),
        },
    }
    diagnostics = {
        "sample_ids": sample_ids,
        "rollout_codes_sha256": _tensor_sha256(codes),
        "student_cfg5_logits_sha256": _tensor_sha256(student),
        "teacher_cfg5_logits_sha256": _tensor_sha256(teacher),
        "valid_mask_sha256": _tensor_sha256(valid),
    }
    return cases, diagnostics


def execute(args: argparse.Namespace) -> None:
    runtime = _require_runtime()
    offline = probe.require_offline_hf_environment()
    upstream = _prepare_inputs_and_upstreams(args)
    preflight_upstream_sha = contract.canonical_json_sha256(
        {key: value for key, value in upstream.items() if key not in ("manifest", "audit_records")}
    )
    device = torch.device("cuda", 0)

    supplied_audiocraft = args.audiocraft_root.expanduser().resolve()
    supplied_checkpoint = args.checkpoint.expanduser().resolve()
    with probe._temporary_import_path(supplied_audiocraft):
        from audiocraft.models.loaders import load_lm_model
        from audiocraft.modules.conditioners import (
            ClassifierFreeGuidanceDropout,
            ConditioningAttributes,
        )

        student_lm = load_lm_model(str(supplied_checkpoint), device="cpu")
        teacher_lm = load_lm_model(str(supplied_checkpoint), device="cpu")
        if student_lm is teacher_lm:
            raise RuntimeError("student and teacher must be independent objects")
        for lm in (student_lm, teacher_lm):
            probe._assert_audiocraft_overlay(lm)
            probe._require_musicgen_small_architecture(lm)
            probe.strict_validate_musicgen_model(lm, frame_rate=50.0)
            _prepare_model(lm, device)
        student_t5 = probe.loaded_t5_identity(student_lm)
        teacher_t5 = probe.loaded_t5_identity(teacher_lm)
        if student_t5 != teacher_t5:
            raise RuntimeError("student and teacher T5 identities differ")
        if student_t5["identity_sha256"] != upstream["cfg_decision"][
            "loaded_t5_identity_sha256"
        ]:
            raise RuntimeError("loaded T5 differs from retained CFG identity")
        initial_student_hash = hash_module_state(student_lm)
        initial_teacher_hash = hash_module_state(teacher_lm)
        if initial_student_hash != initial_teacher_hash:
            raise RuntimeError("real student/teacher states differ at load")

        all_codes = _generate_fixed_codes(
            student_lm,
            upstream["audit_records"],
            ConditioningAttributes,
            ClassifierFreeGuidanceDropout,
        )
        cases, diagnostics = _run_cases(
            student_lm,
            teacher_lm,
            all_codes,
            upstream["audit_records"],
            ConditioningAttributes,
            ClassifierFreeGuidanceDropout,
        )
        final_student_hash = hash_module_state(student_lm)
        final_teacher_hash = hash_module_state(teacher_lm)
        if (
            final_student_hash != initial_student_hash
            or final_teacher_hash != initial_teacher_hash
        ):
            raise RuntimeError("single-GPU audit mutated student or teacher state")

    postflight = _prepare_inputs_and_upstreams(args)
    postflight_upstream_sha = contract.canonical_json_sha256(
        {key: value for key, value in postflight.items() if key not in ("manifest", "audit_records")}
    )
    if preflight_upstream_sha != postflight_upstream_sha:
        raise RuntimeError("immutable upstream identities changed during B1 single audit")

    audit_rows = [
        {
            "sample_id": str(item["sample_id"]),
            "prompt": str(item["prompt"]),
            "source_row_sha256": str(item["source_row_sha256"]),
        }
        for item in upstream["audit_records"]
    ]
    scientific_config = {
        "schema_version": "ptc-opd-b1-single-scientific-config-v1",
        "model_id": "facebook/musicgen-small",
        "audit_sample_selection": "ordered first 8 records of sealed phenomenon probe",
        "single_gpu_sample_count": 2,
        "distributed_reserved_sample_count": 8,
        "rollout_seed": contract.ROLLOUT_SEED,
        "rollout_without_cfg": True,
        "rollout_compute_dtype": "torch.bfloat16",
        "scoring_compute_dtype": "torch.float32",
        "divergence_dtype": "torch.float32",
        "duration_frames": contract.PRIMARY_FRAMES,
        "temperature": contract.ROLLOUT_TEMPERATURE,
        "top_k": contract.ROLLOUT_TOP_K,
        "top_p": contract.ROLLOUT_TOP_P,
        "selector_rho": contract.SELECTOR_RHO,
        "random_namespace": contract.RANDOM_NAMESPACE,
        "random_run_seed": contract.RANDOM_RUN_SEED,
        "random_optimizer_step": contract.RANDOM_OPTIMIZER_STEP,
        "single_padding_lengths": list(contract.SINGLE_PADDING_LENGTHS),
        "padding_token_mutation_offset": contract.PADDING_TOKEN_MUTATION_OFFSET,
        "thresholds": {
            "cfg1_mean_kl_lt": contract.CFG1_MEAN_KL_LT,
            "cfg1_max_valid_kl_lt": contract.CFG1_MAX_VALID_KL_LT,
            "relative_loss_error_lt": contract.RELATIVE_LOSS_ERROR_LT,
            "relative_gradient_error_lt": contract.RELATIVE_GRADIENT_ERROR_LT,
        },
        "checkpoint_identity": upstream["checkpoint_identity"],
        "audiocraft_identity": upstream["audiocraft_identity"],
        "probe_manifest_sha256": upstream["probe_manifest_sha256"],
        "dev_manifest_sha256": upstream["dev_manifest_sha256"],
        "cfg_decision": upstream["cfg_decision"],
        "a1_identity_sha256": upstream["a1_identity_sha256"],
        "a1_artifact_seal_sha256": upstream["a1_artifact_seal_sha256"],
        "a2_artifact_seal_sha256": upstream["a2_artifact_seal_sha256"],
        "a2_scientific_config_sha256": upstream[
            "a2_scientific_config_sha256"
        ],
        "node3_identity": upstream["node3"],
        "t5_closure_identity": upstream["t5_closure"],
    }
    result = {
        "schema_version": contract.SINGLE_SCHEMA,
        "scientific_status": contract.SINGLE_STATUS,
        "gate_passed": True,
        "operationally_accepted": True,
        "redline_touched": False,
        "full_b1_passed": False,
        "pending_b1_items": [
            "B1.9_real_checkpoint_eight_rank_concatenated_reference",
            contract.PENDING_B1_ITEM,
        ],
        "scientific_config": scientific_config,
        "scientific_config_sha256": contract.canonical_json_sha256(scientific_config),
        "runtime": runtime,
        "offline_environment": offline,
        "audit_batch": audit_rows,
        "audit_inputs": {
            "format": "numpy-npz-allow_pickle-false",
            "codes_key": "codes",
            "codes_shape": [8, 4, 500],
            "codes_dtype": "int64",
            "sample_ids_key": "sample_ids",
            "rollout_codes_sha256": _tensor_sha256(all_codes),
        },
        "loaded_t5_identity": student_t5,
        "student_state_sha256_before": initial_student_hash,
        "student_state_sha256_after": final_student_hash,
        "teacher_state_sha256_before": initial_teacher_hash,
        "teacher_state_sha256_after": final_teacher_hash,
        "cases": cases,
        "diagnostics": diagnostics,
        "upstream_seals_before_after_equal": True,
        "upstream_identity_sha256_before": preflight_upstream_sha,
        "upstream_identity_sha256_after": postflight_upstream_sha,
    }

    with contract.staged_output_directory(args.output_dir) as staging:
        np.savez(
            str(staging / "audit_inputs.npz"),
            codes=all_codes.numpy().astype(np.int64, copy=False),
            sample_ids=np.asarray(
                [str(item["sample_id"]) for item in upstream["audit_records"]]
            ),
        )
        contract.write_json_exclusive(staging / "single_gpu_results.json", result)
        contract.write_artifact_seal(
            staging,
            status=contract.SINGLE_STATUS,
            member_names=contract.SINGLE_MEMBERS,
        )
        contract.verify_artifact_directory(
            staging,
            status=contract.SINGLE_STATUS,
            member_names=contract.SINGLE_MEMBERS,
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    execute(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
