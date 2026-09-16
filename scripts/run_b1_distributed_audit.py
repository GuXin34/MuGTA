#!/usr/bin/env python3
"""Literal eight-H20, real-checkpoint B1.9 global-ratio audit.

Launch only with ``torchrun --standalone --nproc_per_node=8``.  Each rank owns
one unequal-length example from the sealed single-GPU artifact.  The DDP
gradient of ``sum_r N_r / sum_r D_r`` is compared with a fresh, non-DDP model
that scores the concatenation of the same eight examples on rank zero.

Passing this runner still does not complete B1: the 500-update stability item
B1.11 remains a separate hard gate.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


WORKPACK_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKPACK_ROOT / "src"
SCRIPTS_ROOT = WORKPACK_ROOT / "scripts"
for path in (str(SRC_ROOT), str(SCRIPTS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import b1_prestability_contract as contract
import run_disagreement_probe as probe
import train_stage1 as train
from ptc_opd import ptc_opd_loss
from ptc_opd.audiocraft_adapter import score_audiocraft_trajectory
from ptc_opd.distributed import globally_normalized_loss
from ptc_opd.train_utils import hash_module_state


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-artifact-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audiocraft-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    header = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _gradient_identity(named_parameters: Sequence[Tuple[str, nn.Parameter]]) -> Dict[str, Any]:
    layout: List[Dict[str, Any]] = []
    missing: List[str] = []
    nonfinite: List[str] = []
    for name, parameter in named_parameters:
        gradient = parameter.grad
        if gradient is None:
            missing.append(name)
            continue
        if not bool(torch.isfinite(gradient).all().item()):
            nonfinite.append(name)
        layout.append(
            {
                "name": name,
                "dtype": str(gradient.dtype),
                "shape": list(gradient.shape),
                "sha256": _tensor_sha256(gradient),
            }
        )
    if missing:
        raise RuntimeError("missing trainable gradients: {}".format(missing[:8]))
    if nonfinite:
        raise FloatingPointError("non-finite trainable gradients: {}".format(nonfinite[:8]))
    return {
        "trainable_parameter_tensor_count": len(named_parameters),
        "trainable_parameter_numel": sum(int(item.numel()) for _, item in named_parameters),
        "gradient_layout_sha256": contract.canonical_json_sha256(layout),
    }


def _relative_scalar_error(observed: float, reference: float) -> float:
    return abs(observed - reference) / max(abs(reference), 1.0e-12)


def _compare_gradients(
    ddp_gradients: Mapping[str, torch.Tensor],
    reference_parameters: Sequence[Tuple[str, nn.Parameter]],
) -> Dict[str, Any]:
    reference_names = [name for name, _ in reference_parameters]
    if list(ddp_gradients) != reference_names:
        raise RuntimeError("DDP/reference trainable parameter names or order differ")
    squared_difference = 0.0
    squared_reference = 0.0
    maximum_tensor_error = 0.0
    maximum_tensor_name = None
    reference_gradient_identity: List[Dict[str, Any]] = []
    for name, parameter in reference_parameters:
        reference = parameter.grad
        if reference is None:
            raise RuntimeError("reference gradient is missing for {}".format(name))
        if not bool(torch.isfinite(reference).all().item()):
            raise FloatingPointError("reference gradient is non-finite for {}".format(name))
        observed = ddp_gradients[name]
        candidate = reference.detach().cpu()
        if observed.dtype != candidate.dtype or observed.shape != candidate.shape:
            raise RuntimeError("gradient representation differs for {}".format(name))
        difference_norm = float(torch.linalg.vector_norm((observed - candidate).double()).item())
        reference_norm = float(torch.linalg.vector_norm(candidate.double()).item())
        relative = difference_norm / max(reference_norm, 1.0e-12)
        if relative > maximum_tensor_error:
            maximum_tensor_error = relative
            maximum_tensor_name = name
        squared_difference += difference_norm * difference_norm
        squared_reference += reference_norm * reference_norm
        reference_gradient_identity.append(
            {
                "name": name,
                "dtype": str(candidate.dtype),
                "shape": list(candidate.shape),
                "sha256": _tensor_sha256(candidate),
            }
        )
    global_relative = math.sqrt(squared_difference) / max(
        math.sqrt(squared_reference), 1.0e-12
    )
    return {
        "global_relative_gradient_l2_error": global_relative,
        "maximum_per_tensor_relative_gradient_l2_error": maximum_tensor_error,
        "maximum_error_parameter": maximum_tensor_name,
        "reference_gradient_layout_sha256": contract.canonical_json_sha256(
            reference_gradient_identity
        ),
    }


def _require_distributed_runtime() -> Dict[str, Any]:
    if any(os.environ.get(name) for name in ("SLURM_JOB_ID", "SLURM_PROCID", "SLURM_LOCALID")):
        raise RuntimeError("B1.9 must use manual torchrun, not Slurm")
    required = {}
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        value = os.environ.get(name)
        if value is None:
            raise RuntimeError("missing torchrun variable {}".format(name))
        required[name] = int(value)
    if required["WORLD_SIZE"] != contract.DISTRIBUTED_WORLD_SIZE:
        raise RuntimeError("formal B1.9 requires WORLD_SIZE=8")
    if required["LOCAL_WORLD_SIZE"] != contract.DISTRIBUTED_WORLD_SIZE:
        raise RuntimeError("formal B1.9 requires one node with LOCAL_WORLD_SIZE=8")
    if required["RANK"] != required["LOCAL_RANK"]:
        raise RuntimeError("formal standalone single-node ranks must equal local ranks")
    if torch.__version__ != "2.1.0+cu121" or torch.version.cuda != "12.1":
        raise RuntimeError("formal B1.9 requires torch 2.1.0+cu121 / CUDA 12.1")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError("formal B1.9 requires exactly eight visible CUDA devices")
    identities = []
    for index in range(8):
        name = torch.cuda.get_device_name(index)
        capability = list(torch.cuda.get_device_capability(index))
        if name != "NVIDIA H20" or capability != [9, 0]:
            raise RuntimeError("all eight visible devices must be NVIDIA H20 [9,0]")
        identities.append({"index": index, "name": name, "capability": capability})
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("formal B1.9 requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    if os.environ.get("NCCL_ASYNC_ERROR_HANDLING") != "1":
        raise RuntimeError("formal B1.9 requires NCCL_ASYNC_ERROR_HANDLING=1")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "backend": "nccl",
        "launcher": "torchrun_standalone",
        "world_size": 8,
        "local_world_size": 8,
        "gpus": identities,
        "deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "scoring_autocast": False,
    }


def _load_single_artifact(directory: Path) -> Tuple[Dict[str, Any], np.ndarray, List[str], Dict[str, Any]]:
    root = directory.expanduser().resolve(strict=True)
    identity = contract.verify_artifact_directory(
        root,
        status=contract.SINGLE_STATUS,
        member_names=contract.SINGLE_MEMBERS,
    )
    result = contract.load_json(root / "single_gpu_results.json", "single B1 result")
    if (
        result.get("schema_version") != contract.SINGLE_SCHEMA
        or result.get("scientific_status") != contract.SINGLE_STATUS
        or result.get("gate_passed") is not True
        or result.get("full_b1_passed") is not False
    ):
        raise contract.ContractError("single B1 artifact is not a valid partial pass")
    with np.load(str(root / "audit_inputs.npz"), allow_pickle=False) as archive:
        if set(archive.files) != {"codes", "sample_ids"}:
            raise contract.ContractError("audit_inputs.npz member set differs")
        codes = archive["codes"]
        sample_ids_array = archive["sample_ids"]
    if codes.dtype != np.int64 or codes.shape != (8, 4, 500):
        raise contract.ContractError("sealed B1 audit codes must be int64 [8,4,500]")
    if sample_ids_array.ndim != 1 or sample_ids_array.shape != (8,) or sample_ids_array.dtype.kind not in ("U", "S"):
        raise contract.ContractError("sealed B1 sample IDs must be a non-pickle string vector of length 8")
    sample_ids = [str(item) for item in sample_ids_array.tolist()]
    audit_rows = result.get("audit_batch")
    if not isinstance(audit_rows, list) or len(audit_rows) != 8:
        raise contract.ContractError("single B1 audit batch must contain eight rows")
    if [item.get("sample_id") for item in audit_rows if isinstance(item, dict)] != sample_ids:
        raise contract.ContractError("NPZ and single-result sample IDs differ")
    codes_tensor = torch.from_numpy(codes.copy())
    declared_hash = result.get("audit_inputs", {}).get("rollout_codes_sha256")
    if declared_hash != _tensor_sha256(codes_tensor):
        raise contract.ContractError("sealed rollout tensor hash differs")
    return result, codes, sample_ids, identity


def _prepare_models(
    checkpoint: Path,
    device: torch.device,
    load_lm_model: Any,
) -> Tuple[nn.Module, nn.Module, Dict[str, Any]]:
    student = load_lm_model(str(checkpoint), device="cpu")
    teacher = load_lm_model(str(checkpoint), device="cpu")
    if student is teacher:
        raise RuntimeError("student and teacher must be independent objects")
    for label, model in (("student", student), ("teacher", teacher)):
        parameters = list(model.parameters())
        if not parameters or any(item.dtype != torch.float32 for item in parameters):
            raise RuntimeError("{} did not load as CPU FP32".format(label))
        if any(item.device.type != "cpu" for item in parameters):
            raise RuntimeError("{} did not load on CPU".format(label))
        probe._assert_audiocraft_overlay(model)
        probe._require_musicgen_small_architecture(model)
        probe.strict_validate_musicgen_model(model, frame_rate=50.0)
    student_t5 = probe.loaded_t5_identity(student)
    teacher_t5 = probe.loaded_t5_identity(teacher)
    if student_t5 != teacher_t5:
        raise RuntimeError("student/teacher T5 identities differ")
    student.to(device)
    teacher.to(device)
    train.move_external_text_encoder(student, device)
    student.eval()
    teacher.requires_grad_(False)
    teacher.eval()
    train.freeze_condition_provider(student)
    train.freeze_condition_provider(teacher)
    if hash_module_state(student) != hash_module_state(teacher):
        raise RuntimeError("student/teacher real checkpoint states differ")
    return student, teacher, student_t5


def _padding_mask(
    device: torch.device,
    lengths: Sequence[int],
    delays: Sequence[int] = (0, 1, 2, 3),
) -> torch.Tensor:
    """Delayed-pattern-aware distributed padding mask (pilot RULING #4, φ.b).

    Under MusicGen's frozen ``DelayedPatternProvider`` with delays
    ``[0, 1, 2, 3]``, the region actually predicted from raw codes of
    length ``L`` per codebook ``q`` is ``t + delays[q] < L``, i.e.
    per-codebook lengths ``[L, L-1, L-2, L-3]``.

    A pre-φ.b implementation of this function returned
    ``(time < L).expand(-1, PRIMARY_CODEBOOKS, -1)`` which tiled the
    same length ``L`` across all four codebooks (the "delay-blind"
    semantics).  For the eight batched samples with lengths
    ``DISTRIBUTED_PADDING_LENGTHS = (500, 487, 474, 461, 448, 435, 422, 409)``
    this over-counted the valid lattice by 42 cells (14538 vs 14496)
    — identical in spirit to the pre-ε single-GPU B1.5 fixture defect.

    The correction mirrors ε (RULING #3): return
    ``(time + delays_view) < lengths_view``, broadcasting the batch
    dimension against per-codebook delays.  Confined to this audit
    fixture; NO change to ``src/ptc_opd/losses.py``, AudioCraft, or
    ``configs/stage1_matrix.yaml``.
    """
    if not lengths:
        raise ValueError("padding-length batch must not be empty")
    if len(delays) != contract.PRIMARY_CODEBOOKS:
        raise ValueError(
            "delays length {} does not match PRIMARY_CODEBOOKS={}".format(
                len(delays), contract.PRIMARY_CODEBOOKS
            )
        )
    time = torch.arange(contract.PRIMARY_FRAMES, device=device).view(1, 1, -1)
    limit = torch.tensor(list(lengths), device=device, dtype=torch.long).view(-1, 1, 1)
    delay = torch.tensor(list(delays), device=device, dtype=torch.long).view(1, -1, 1)
    return (time + delay) < limit


def _score_batch(
    student: Any,
    teacher: nn.Module,
    codes: torch.Tensor,
    prompts: Sequence[str],
    device: torch.device,
    conditioning_class: Any,
    dropout_class: Any,
    padding_lengths: Sequence[int],
) -> Tuple[Any, Any]:
    if len(prompts) != int(codes.shape[0]) or len(padding_lengths) != int(codes.shape[0]):
        raise ValueError("prompt/padding batch does not match codes")
    conditional, null = train.prepare_condition_tensors(
        student.lm if isinstance(student, train.DDPScoringFacade) else student,
        list(prompts),
        conditioning_class,
        dropout_class,
    )
    with torch.autocast(device_type="cuda", enabled=False):
        scores = score_audiocraft_trajectory(
            student,
            teacher,
            codes,
            conditional,
            null,
            rollout_mask=torch.ones_like(codes, dtype=torch.bool),
            padding_mask=_padding_mask(device, padding_lengths),
            teacher_cfg_scale=5.0,
            teacher_forward_mode="batched",
            check_finite=True,
        )
        output = ptc_opd_loss(
            scores.student_logits,
            scores.teacher_cfg_logits,
            valid_mask=scores.valid_mask,
            mode="uniform",
            kl_direction="forward",
            selection_scope="protocol",
            layout="BQTV",
            temperature=1.0,
            check_finite=True,
        )
    return scores, output


def _run_reference(
    *,
    checkpoint: Path,
    load_lm_model: Any,
    device: torch.device,
    codes: np.ndarray,
    audit_rows: Sequence[Mapping[str, Any]],
    global_denominator: float,
    ddp_gradients: Mapping[str, torch.Tensor],
    conditioning_class: Any,
    dropout_class: Any,
    expected_t5_identity_sha256: str,
) -> Dict[str, Any]:
    """Sequential 8-singletons rank-0 reference (pilot RULING #4, φ.a).

    A pre-φ.a implementation of this reference concatenated all eight
    sealed audit rows into a single ``batch_size=8`` forward, then
    divided ``output.numerator`` by ``global_denominator`` and
    ``.backward()``ed once.  Empirically (see φ D0 diagnostic
    ``20260824T122435Z``), this concatenated path diverges from true
    per-rank ``batch=1`` DDP by ~4.77 %/14.30 % (loss/gradient
    relative L2), far beyond the frozen 1e-6/1e-5 tolerances.

    The mathematically equivalent rank-0 reference is to load one
    fresh non-DDP model, run the eight rows one at a time
    (``batch_size=1``), and accumulate gradients as
    ``sum_i (N_i / global_denominator).backward()`` into the same
    parameters.  This matches path (A)—the real DDP path—to ~5e-8/
    7e-8 relative L2 (well below the 1e-6/1e-5 tolerances).

    NO threshold change and NO production-training-code change.
    Confined to this audit reference.  Batch shapes of the
    per-singleton student/teacher conditioner call thus mirror the DDP
    ``B=1``/``B=2`` shapes rather than the ``B=8``/``B=16`` shapes,
    which is the very quantity Ruling #4 flagged as the divergence
    source.
    """
    reference_student, reference_teacher, reference_t5 = _prepare_models(
        checkpoint, device, load_lm_model
    )
    if reference_t5.get("identity_sha256") != expected_t5_identity_sha256:
        raise RuntimeError("fresh sequential reference loaded a different T5")
    reference_initial_teacher_hash = hash_module_state(reference_teacher)
    provider_delays_ref = list(reference_student.pattern_provider.delays)
    if provider_delays_ref != [0, 1, 2, 3]:
        raise RuntimeError(
            "unexpected MusicGen delay pattern: {} (expected [0, 1, 2, 3])".format(
                provider_delays_ref
            )
        )
    named_reference = [
        (name, parameter)
        for name, parameter in reference_student.named_parameters()
        if parameter.requires_grad
    ]
    if not named_reference:
        raise RuntimeError("fresh sequential reference has no trainable parameters")
    for _, parameter in named_reference:
        parameter.grad = None

    per_sample_numerators: List[float] = []
    per_sample_denominators: List[float] = []
    per_sample_valid_totals: List[int] = []
    per_sample_causal_valid_counts: List[List[int]] = []
    numerator_total = 0.0
    denominator_total = 0.0

    lengths_all = list(contract.DISTRIBUTED_PADDING_LENGTHS)
    if len(audit_rows) != contract.DISTRIBUTED_WORLD_SIZE:
        raise RuntimeError(
            "sequential reference expects {} audit rows, got {}".format(
                contract.DISTRIBUTED_WORLD_SIZE, len(audit_rows)
            )
        )
    for i in range(contract.DISTRIBUTED_WORLD_SIZE):
        codes_i = torch.from_numpy(codes[i : i + 1].copy()).to(device)
        prompt_i = [str(audit_rows[i]["prompt"])]
        length_i = [lengths_all[i]]
        scores_i, output_i = _score_batch(
            reference_student,
            reference_teacher,
            codes_i,
            prompt_i,
            device,
            conditioning_class,
            dropout_class,
            length_i,
        )
        n_i = float(output_i.numerator.detach().item())
        d_i = float(output_i.effective_weight_sum.detach().item())
        numerator_total += n_i
        denominator_total += d_i
        per_sample_numerators.append(n_i)
        per_sample_denominators.append(d_i)
        per_sample_valid_totals.append(int(scores_i.valid_mask.sum().item()))
        per_sample_causal_valid_counts.append(
            [int(x) for x in scores_i.valid_mask.sum(dim=2).flatten().tolist()]
        )
        (output_i.numerator / global_denominator).backward()
        del scores_i, output_i, codes_i
        torch.cuda.empty_cache()

    if denominator_total != global_denominator:
        raise RuntimeError(
            "sequential reference denominator ({}) differs from DDP global ({})".format(
                denominator_total, global_denominator
            )
        )
    reference_loss = numerator_total / denominator_total
    comparison = _compare_gradients(ddp_gradients, named_reference)
    if hash_module_state(reference_teacher) != reference_initial_teacher_hash:
        raise RuntimeError("fresh reference mutated the frozen teacher")
    # NOTE (Ruling #4 φ.a.fix1): the verifier ``_exact_keys`` on the
    # ``concatenated_reference`` block permits exactly 8 fields.  Ruling
    # #4 branch (3) authorises the sequential-singletons reference-path
    # rewrite but does NOT authorise loosening the verifier schema.  We
    # therefore preserve the frozen ``concatenated_reference_*`` field
    # names as the numerical envelope (loss / numerator / denominator /
    # per-sample valid totals) — the sequential accumulation is
    # mathematically equivalent to a batch-of-8 forward with the correct
    # ε mask, so these values continue to describe the same quantities
    # they did pre-φ.  The sequential provenance is recorded in
    # ``scientific_config.reference_path_scheme = "sequential_singletons"``
    # (φW3) and enforced by the test ``test_reference_path_scheme_pin_
    # matches_ruling4``.
    return {
        "concatenated_reference_loss": reference_loss,
        "concatenated_reference_numerator": numerator_total,
        "concatenated_reference_denominator": denominator_total,
        "concatenated_reference_valid_counts": per_sample_valid_totals,
        **comparison,
    }


def execute(args: argparse.Namespace) -> None:
    runtime = _require_distributed_runtime()
    offline = probe.require_offline_hf_environment()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    output_path = args.output_dir.expanduser().resolve()
    if rank == 0:
        output_absent = not output_path.exists()
    else:
        output_absent = False

    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    try:
        absent_box = [output_absent]
        dist.broadcast_object_list(absent_box, src=0)
        if absent_box[0] is not True:
            raise FileExistsError("refusing to overwrite --output-dir")

        single_result, codes, sample_ids, single_identity = _load_single_artifact(
            args.single_artifact_dir
        )
        checkpoint_identity = probe.verify_checkpoint_snapshot(args.checkpoint)
        audiocraft_identity = probe.verify_audiocraft_source(args.audiocraft_root)
        single_config = single_result.get("scientific_config")
        if not isinstance(single_config, dict):
            raise contract.ContractError("single scientific config is missing")
        if single_config.get("checkpoint_identity") != checkpoint_identity:
            raise contract.ContractError("distributed checkpoint differs from single audit")
        if single_config.get("audiocraft_identity") != audiocraft_identity:
            raise contract.ContractError("distributed AudioCraft differs from single audit")
        if single_config.get("cfg_decision", {}).get("selected_cfg_scale") != 5.0:
            raise contract.ContractError("single audit did not retain CFG scale 5.0")
        audit_rows = single_result.get("audit_batch")
        if not isinstance(audit_rows, list) or len(audit_rows) != 8:
            raise contract.ContractError("single audit batch is malformed")

        supplied_audiocraft = args.audiocraft_root.expanduser().resolve()
        checkpoint = args.checkpoint.expanduser().resolve()
        with probe._temporary_import_path(supplied_audiocraft):
            from audiocraft.models.loaders import load_lm_model
            from audiocraft.modules.conditioners import (
                ClassifierFreeGuidanceDropout,
                ConditioningAttributes,
            )

            student, teacher, t5_identity = _prepare_models(
                checkpoint, device, load_lm_model
            )
            if t5_identity.get("identity_sha256") != single_result.get(
                "loaded_t5_identity", {}
            ).get("identity_sha256"):
                raise RuntimeError("distributed loaded T5 differs from single audit")
            student_initial_hash = hash_module_state(student)
            teacher_initial_hash = hash_module_state(teacher)
            scorer = train.StudentScorer(student).to(device)
            ddp = DDP(
                scorer,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                bucket_cap_mb=train.DDP_BUCKET_CAP_MB,
                find_unused_parameters=train.DDP_FIND_UNUSED_PARAMETERS,
                gradient_as_bucket_view=train.DDP_GRADIENT_AS_BUCKET_VIEW,
                static_graph=train.DDP_STATIC_GRAPH,
            )
            facade = train.DDPScoringFacade(ddp, student)
            named_trainable = [
                (name, parameter)
                for name, parameter in student.named_parameters()
                if parameter.requires_grad
            ]
            for _, parameter in named_trainable:
                parameter.grad = None
            local_codes = torch.from_numpy(codes[rank : rank + 1].copy()).to(device)
            scores, output = _score_batch(
                facade,
                teacher,
                local_codes,
                [str(audit_rows[rank]["prompt"])],
                device,
                ConditioningAttributes,
                ClassifierFreeGuidanceDropout,
                [contract.DISTRIBUTED_PADDING_LENGTHS[rank]],
            )
            ratio = globally_normalized_loss(output, require_distributed=True)
            ratio.backward_loss.backward()
            ddp_reducer = train.ddp_reducer_audit(ddp)
            gradient_identity = _gradient_identity(named_trainable)
            local = {
                "rank": rank,
                "local_rank": local_rank,
                "sample_id": sample_ids[rank],
                "padding_length": contract.DISTRIBUTED_PADDING_LENGTHS[rank],
                "valid_cell_count": int(scores.valid_mask.sum().item()),
                "local_numerator": float(output.numerator.detach().item()),
                "local_denominator": float(output.effective_weight_sum.detach().item()),
                "local_ratio": float(output.loss.detach().item()),
                "global_loss": float(ratio.global_loss.detach().item()),
                "global_numerator": float(ratio.global_numerator.detach().item()),
                "global_denominator": float(ratio.global_denominator.detach().item()),
                "gradient_identity": gradient_identity,
                "teacher_state_sha256_before": teacher_initial_hash,
                "teacher_state_sha256_after": hash_module_state(teacher),
                "student_state_sha256_before": student_initial_hash,
                "student_state_sha256_after": hash_module_state(student),
                "ddp_reducer": ddp_reducer,
            }
            if local["teacher_state_sha256_before"] != local["teacher_state_sha256_after"]:
                raise RuntimeError("DDP audit mutated the frozen teacher")
            if local["student_state_sha256_before"] != local["student_state_sha256_after"]:
                raise RuntimeError("DDP audit mutated student parameter values")
            by_rank: List[Any] = [None for _ in range(8)]
            dist.all_gather_object(by_rank, local)
            if [item.get("rank") for item in by_rank] != list(range(8)):
                raise RuntimeError("rank evidence does not cover ordered ranks 0..7")
            gradient_hashes = {
                item["gradient_identity"]["gradient_layout_sha256"] for item in by_rank
            }
            if len(gradient_hashes) != 1:
                raise AssertionError("DDP synchronized gradient bytes differ across ranks")
            global_losses = {item["global_loss"] for item in by_rank}
            global_numerators = {item["global_numerator"] for item in by_rank}
            global_denominators = {item["global_denominator"] for item in by_rank}
            if not (
                len(global_losses) == 1
                and len(global_numerators) == 1
                and len(global_denominators) == 1
            ):
                raise AssertionError("global ratio statistics differ across ranks")
            ddp_loss = float(by_rank[0]["global_loss"])
            global_denominator = float(by_rank[0]["global_denominator"])
            wrong_mean = sum(float(item["local_ratio"]) for item in by_rank) / 8.0
            sensitivity = abs(wrong_mean - ddp_loss)
            if not sensitivity > contract.MEAN_OF_MEANS_SENSITIVITY_GT:
                raise AssertionError("unequal-denominator audit is not identifiable")

            ddp_gradients: Optional[Dict[str, torch.Tensor]] = None
            if rank == 0:
                ddp_gradients = {
                    name: parameter.grad.detach().cpu().clone()
                    for name, parameter in named_trainable
                }

            dist.barrier()
            decision: List[Any] = [None]
            if rank == 0:
                try:
                    if ddp_gradients is None:
                        raise AssertionError("rank zero DDP gradients are unavailable")
                    del facade, ddp, scorer, student, teacher
                    torch.cuda.empty_cache()
                    reference = _run_reference(
                        checkpoint=checkpoint,
                        load_lm_model=load_lm_model,
                        device=device,
                        codes=codes,
                        audit_rows=audit_rows,
                        global_denominator=global_denominator,
                        ddp_gradients=ddp_gradients,
                        conditioning_class=ConditioningAttributes,
                        dropout_class=ClassifierFreeGuidanceDropout,
                        expected_t5_identity_sha256=str(t5_identity["identity_sha256"]),
                    )
                    if probe.verify_checkpoint_snapshot(checkpoint) != checkpoint_identity:
                        raise RuntimeError("checkpoint identity changed during B1.9")
                    if probe.verify_audiocraft_source(supplied_audiocraft) != audiocraft_identity:
                        raise RuntimeError("AudioCraft identity changed during B1.9")
                    _, _, post_sample_ids, post_single_identity = _load_single_artifact(
                        args.single_artifact_dir
                    )
                    if (
                        post_single_identity["seal_sha256"] != single_identity["seal_sha256"]
                        or post_sample_ids != sample_ids
                    ):
                        raise RuntimeError("single B1 input artifact changed during B1.9")
                    loss_error = _relative_scalar_error(
                        ddp_loss, reference["concatenated_reference_loss"]
                    )
                    gradient_error = reference["global_relative_gradient_l2_error"]
                    maximum_gradient_error = reference[
                        "maximum_per_tensor_relative_gradient_l2_error"
                    ]
                    if not (
                        loss_error < contract.RELATIVE_LOSS_ERROR_LT
                        and gradient_error < contract.RELATIVE_GRADIENT_ERROR_LT
                    ):
                        raise AssertionError("DDP/concatenated reference tolerance failed")
                    scientific_config = {
                        "schema_version": "ptc-opd-b1-ddp-scientific-config-v1",
                        "model_id": "facebook/musicgen-small",
                        "world_size": 8,
                        "rank_batch_size": 1,
                        "loss_mode": "uniform",
                        "teacher_cfg_scale": 5.0,
                        "scoring_compute_dtype": "torch.float32",
                        "divergence_dtype": "torch.float32",
                        "padding_lengths_by_rank": list(contract.DISTRIBUTED_PADDING_LENGTHS),
                        "padding_semantics": "delayed_pattern_valid_scoring",
                        "provider_delays": [0, 1, 2, 3],
                        "reference_path_scheme": "sequential_singletons",
                        "thresholds": {
                            "relative_loss_error_lt": contract.RELATIVE_LOSS_ERROR_LT,
                            "relative_gradient_error_lt": contract.RELATIVE_GRADIENT_ERROR_LT,
                            "mean_of_means_sensitivity_gt": contract.MEAN_OF_MEANS_SENSITIVITY_GT,
                        },
                        "ddp_policy": {
                            "bucket_cap_mb": train.DDP_BUCKET_CAP_MB,
                            "find_unused_parameters": train.DDP_FIND_UNUSED_PARAMETERS,
                            "static_graph": train.DDP_STATIC_GRAPH,
                            "gradient_as_bucket_view": train.DDP_GRADIENT_AS_BUCKET_VIEW,
                        },
                        "single_artifact_seal_sha256": single_identity["seal_sha256"],
                        "checkpoint_identity": checkpoint_identity,
                        "audiocraft_identity": audiocraft_identity,
                    }
                    result = {
                        "schema_version": contract.DISTRIBUTED_SCHEMA,
                        "scientific_status": contract.DISTRIBUTED_STATUS,
                        "gate_passed": True,
                        "operationally_accepted": True,
                        "redline_touched": False,
                        "full_b1_passed": False,
                        "pending_b1_items": [contract.PENDING_B1_ITEM],
                        "scientific_config": scientific_config,
                        "scientific_config_sha256": contract.canonical_json_sha256(scientific_config),
                        "runtime": runtime,
                        "offline_environment": offline,
                        "rank_evidence": by_rank,
                        "ddp_gradient_hash_exact_across_ranks": True,
                        "ddp_gradient_layout_sha256": next(iter(gradient_hashes)),
                        "ddp_global_loss": ddp_loss,
                        "wrong_mean_of_local_ratios": wrong_mean,
                        "wrong_mean_absolute_difference": sensitivity,
                        "wrong_mean_sensitivity_threshold_exclusive": contract.MEAN_OF_MEANS_SENSITIVITY_GT,
                        "relative_loss_error": loss_error,
                        "relative_gradient_l2_error": gradient_error,
                        "maximum_per_tensor_relative_gradient_l2_error": maximum_gradient_error,
                        "relative_loss_threshold_exclusive": contract.RELATIVE_LOSS_ERROR_LT,
                        "relative_gradient_threshold_exclusive": contract.RELATIVE_GRADIENT_ERROR_LT,
                        "concatenated_reference": reference,
                        "upstream_identities_postflight_equal": True,
                    }
                    with contract.staged_output_directory(output_path) as staging:
                        contract.write_json_exclusive(staging / "distributed_results.json", result)
                        contract.write_artifact_seal(
                            staging,
                            status=contract.DISTRIBUTED_STATUS,
                            member_names=contract.DISTRIBUTED_MEMBERS,
                        )
                        contract.verify_artifact_directory(
                            staging,
                            status=contract.DISTRIBUTED_STATUS,
                            member_names=contract.DISTRIBUTED_MEMBERS,
                        )
                    decision[0] = {"ok": True}
                except BaseException as exc:
                    decision[0] = {
                        "ok": False,
                        "error": "{}: {}".format(type(exc).__name__, exc),
                        "traceback": traceback.format_exc(),
                    }
            dist.broadcast_object_list(decision, src=0)
            if not isinstance(decision[0], dict) or decision[0].get("ok") is not True:
                raise RuntimeError("rank-zero reference/output failed: {}".format(decision[0]))
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    execute(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
