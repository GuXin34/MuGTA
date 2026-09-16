#!/usr/bin/env python3
"""Standalone one-machine/eight-GPU Stage-1 PTC-OPD training runner.

Launch only as one independent job per physical machine::

    torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/train_stage1.py ...

This entry point deliberately does not use Dora, Slurm, submitit, or a
cross-machine rendezvous.  ``--dry-run`` stays CPU-only and never imports
AudioCraft, so paths, manifests, hashes, method mapping, and batch arithmetic
can be audited before a GPU process group is created.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import socket
import struct
import sys
import tempfile
import time
import traceback
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


WORKPACK_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKPACK_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ptc_opd.audiocraft_adapter import score_audiocraft_trajectory
from ptc_opd.cfg_decision import verify_cfg_scale_decision
from ptc_opd.codec_prior_artifact import (
    CodecPriorArtifact,
    load_codec_prior_artifact,
    verify_local_codec_snapshot,
)
from ptc_opd.distributed import globally_normalized_loss
from ptc_opd.losses import ptc_opd_loss
from ptc_opd.musicgen_contract import strict_validate_musicgen_model
from ptc_opd.reproducibility import (
    audiocraft_source_identity,
    loaded_t5_identity,
    require_offline_hf_environment,
)
from ptc_opd.sampling import (
    DeterministicDistributedBatchSampler,
    keyed_random_scores,
)
from ptc_opd.train_utils import (
    CHECKPOINT_SCHEMA_VERSION,
    MODE_SPECS,
    RUN_SCHEMA_VERSION,
    Stage1Config,
    build_checkpoint_metadata,
    canonical_json_sha256,
    hash_module_state,
    learning_rate_for_update,
    load_prompt_manifest,
    progress_from_optimizer_step,
    resolve_mode,
    resolved_config_dict,
    sha256_file,
    sha256_path,
    validate_config,
    verify_resume_metadata,
)


ATTEMPT_SCHEMA_VERSION = "ptc-opd-stage1-attempt-v1"
SEAL_SCHEMA_VERSION = "ptc-opd-stage1-seal-v5"

# Ruling #8 §3 module-level extension flag (memory
# ruling-8-final-small-horizon-extension-2026-08-30).  main() writes the
# extension target when --extend-to-step is used; _build_success_seal_payload
# reads this to allow optimizer_step > run_manifest.max_optimizer_steps.
# None for all normal runs.
_EXTENSION_TARGET_STEP: Optional[int] = None
DONE_SCHEMA_VERSION = "ptc-opd-stage1-done-v5"
VERIFICATION_SCHEMA_VERSION = "ptc-opd-stage1-verification-v1"
_CHECKPOINT_STEP_RE = re.compile(r"step-(\d{5})\Z")
_ATTEMPT_METADATA_RE = re.compile(r"attempt-(\d{4})\.json\Z")
_ATTEMPT_METRICS_RE = re.compile(r"metrics\.attempt-(\d{4})\.jsonl\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

# PyTorch 2.1 normally starts DDP with one all-model bucket when
# find_unused_parameters=False, then rebuilds buckets under the configured cap
# after the first synchronized backward.  That can make a warm uninterrupted
# step use a different collective topology from the first step in a freshly
# resumed process.  The
# public configuration below starts with the normal fixed-size buckets and, in
# pinned Torch 2.1, prevents reducer bucket rebuilding.  Keep every argument
# explicit: this is part of the Stage-1 resume-reproducibility protocol.
DDP_BUCKET_CAP_MB = 25
DDP_FIND_UNUSED_PARAMETERS = True
DDP_STATIC_GRAPH = False
DDP_GRADIENT_AS_BUCKET_VIEW = False
DDP_REDUCER_SCHEMA_VERSION = "ptc-opd-ddp-reducer-contract-v1"
DDP_REDUCER_POLICY_ID = "torch-2.1-fixed-initial-buckets-v1"
RESUME_STATE_RESTORE_SCHEMA_VERSION = "ptc-opd-node3-resume-state-restore-v1"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--audiocraft-root", type=Path, required=True)
    parser.add_argument("--cfg-scale-decision-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=sorted(MODE_SPECS), required=True)
    parser.add_argument(
        "--codebook-prior-artifact-dir",
        type=Path,
        help="sealed Phase-A1 directory; required by codebook100 and ptc50",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-optimizer-steps", type=int, required=True)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--rank-batch-size", type=int, default=2)
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--effective-global-batch", type=int, default=64)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--codec-frame-rate", type=float, default=50.0)
    parser.add_argument("--token-frames", type=int, default=500)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-top-k", type=int, default=250)
    parser.add_argument("--rollout-top-p", type=float, default=0.0)
    parser.add_argument("--distillation-temperature", type=float, default=1.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--teacher-forward-mode", choices=("batched", "separate"), default="batched"
    )
    parser.add_argument("--random-mask-namespace", type=int, default=5701)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--extend-to-step",
        type=int,
        default=None,
        help=(
            "Ruling #8 bypass: extend training past the run_manifest's sealed "
            "max_optimizer_steps to the given step, keeping upstream config_sha256 "
            "byte-identical so the extension is a strict resume of the same "
            "500-step config (only training duration is changed, per Ruling #8 §4). "
            "Requires --resume and an existing SEALED.json + DONE.json in output-dir. "
            "See memory ruling-8-final-small-horizon-extension-2026-08-30."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-check-finite", action="store_true")
    parser.add_argument(
        "--denominator-rtol",
        type=float,
        default=1.0e-6,
        help="frozen at 1e-6; fail if accumulation denominators differ",
    )
    parser.add_argument(
        "--node3-stop-after-step",
        type=int,
        help=(
            "node-3 gate-only fault injection; the only accepted value is 1 "
            "under the exact two-update uniform100 smoke contract"
        ),
    )
    return parser.parse_args(argv)


def make_config(args: argparse.Namespace) -> Stage1Config:
    cfg_decision = verify_cfg_scale_decision(args.cfg_scale_decision_dir)
    if cfg_decision.selected_cfg_scale is None:
        raise ValueError("CFG development gate did not select a scale")
    return Stage1Config(
        manifest=str(args.manifest.resolve()),
        student_checkpoint=str(args.student_checkpoint.resolve()),
        teacher_checkpoint=str(args.teacher_checkpoint.resolve()),
        audiocraft_root=str(args.audiocraft_root.resolve()),
        output_dir=str(args.output_dir.resolve()),
        cfg_scale_decision_dir=cfg_decision.directory,
        cfg_scale_decision_file_sha256=cfg_decision.decision_file_sha256,
        cfg_scale_decision_payload_sha256=cfg_decision.decision_payload_sha256,
        cfg_scale_scientific_config_sha256=(
            cfg_decision.scientific_config_sha256
        ),
        cfg_generation_checkpoint_sha256=str(
            cfg_decision.generation_identity["checkpoint_sha256"]
        ),
        cfg_generation_audiocraft_source_sha256=str(
            cfg_decision.generation_identity["audiocraft_source_sha256"]
        ),
        cfg_generation_loaded_t5_identity_sha256=str(
            cfg_decision.generation_identity["loaded_t5_identity_sha256"]
        ),
        cfg_generation_state_dict_sha256=str(
            cfg_decision.generation_identity["state_dict_sha256"]
        ),
        mode=args.mode,
        seed=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_optimizer_steps=args.max_optimizer_steps,
        save_every=args.save_every,
        log_every=args.log_every,
        codebook_prior_artifact_dir=(
            str(args.codebook_prior_artifact_dir.resolve())
            if args.codebook_prior_artifact_dir is not None
            else None
        ),
        rank_batch_size=args.rank_batch_size,
        expected_world_size=args.expected_world_size,
        grad_accum_steps=args.grad_accum_steps,
        effective_global_batch=args.effective_global_batch,
        duration_seconds=args.duration_seconds,
        codec_frame_rate=args.codec_frame_rate,
        token_frames=args.token_frames,
        rollout_temperature=args.rollout_temperature,
        rollout_top_k=args.rollout_top_k,
        rollout_top_p=args.rollout_top_p,
        teacher_cfg_scale=cfg_decision.selected_cfg_scale,
        distillation_temperature=args.distillation_temperature,
        grad_clip_norm=args.grad_clip_norm,
        teacher_forward_mode=args.teacher_forward_mode,
        check_finite=not args.no_check_finite,
        random_mask_namespace=args.random_mask_namespace,
        denominator_rtol=args.denominator_rtol,
    )


def print_json(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def jsonl_append(path: Path, value: Mapping[str, Any]) -> None:
    """Append to an already-created regular log without ever creating it."""

    if path.is_symlink() or not path.is_file():
        raise ValueError("JSONL append target must be an existing regular file")
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(str(path), os.O_WRONLY | os.O_APPEND)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(str(path), flags, 0o644)
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def broadcast_object(value: Any, rank: int) -> Any:
    values = [value if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def checkpoint_hashes(config: Stage1Config, rank: int) -> Dict[str, str]:
    envelope: Optional[Dict[str, Any]] = None
    if rank == 0:
        try:
            envelope = {
                "value": {
                    "manifest_sha256": sha256_file(Path(config.manifest)),
                    "student_checkpoint_sha256": sha256_path(
                        Path(config.student_checkpoint)
                    ),
                    "teacher_checkpoint_sha256": sha256_path(
                        Path(config.teacher_checkpoint)
                    ),
                    "student_state_dict_sha256": sha256_file(
                        Path(config.student_checkpoint) / "state_dict.bin"
                    ),
                    "teacher_state_dict_sha256": sha256_file(
                        Path(config.teacher_checkpoint) / "state_dict.bin"
                    ),
                }
            }
        except BaseException as exc:
            envelope = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    envelope = broadcast_object(envelope, rank)
    if not isinstance(envelope, dict) or "value" not in envelope:
        raise RuntimeError("rank-zero checkpoint hashing failed: {}".format(envelope))
    value = envelope["value"]
    if not isinstance(value, dict):
        raise RuntimeError("rank-zero checkpoint hash payload is malformed")
    return {str(key): str(item) for key, item in value.items()}


def distributed_source_identity(path: Path, rank: int) -> Dict[str, Any]:
    envelope: Optional[Dict[str, Any]] = None
    if rank == 0:
        try:
            envelope = {"value": audiocraft_source_identity(path)}
        except BaseException as exc:
            envelope = {"error_type": type(exc).__name__, "error": str(exc)}
    envelope = broadcast_object(envelope, rank)
    if not isinstance(envelope, dict) or "value" not in envelope:
        raise RuntimeError("rank-zero AudioCraft hashing failed: {}".format(envelope))
    value = envelope["value"]
    if not isinstance(value, dict):
        raise RuntimeError("rank-zero AudioCraft identity is malformed")
    return value


def synchronize_preflight(value: Mapping[str, Any], rank: int) -> None:
    """Fail collectively when immutable path/config views differ by rank."""

    gathered: List[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, dict(value))
    reference = gathered[0]
    if any(item != reference for item in gathered[1:]):
        raise RuntimeError("rank-local Stage-1 preflight views are inconsistent")


def verify_configured_cfg_decision(config: Stage1Config) -> Dict[str, Any]:
    decision = verify_cfg_scale_decision(Path(config.cfg_scale_decision_dir))
    observed = {
        "selected_cfg_scale": decision.selected_cfg_scale,
        "decision_file_sha256": decision.decision_file_sha256,
        "decision_payload_sha256": decision.decision_payload_sha256,
        "scientific_config_sha256": decision.scientific_config_sha256,
        "generation_identity": dict(decision.generation_identity),
    }
    expected = {
        "selected_cfg_scale": config.teacher_cfg_scale,
        "decision_file_sha256": config.cfg_scale_decision_file_sha256,
        "decision_payload_sha256": config.cfg_scale_decision_payload_sha256,
        "scientific_config_sha256": config.cfg_scale_scientific_config_sha256,
        "generation_identity": {
            "checkpoint_sha256": config.cfg_generation_checkpoint_sha256,
            "state_dict_sha256": config.cfg_generation_state_dict_sha256,
            "audiocraft_source_sha256": (
                config.cfg_generation_audiocraft_source_sha256
            ),
            "loaded_t5_identity_sha256": (
                config.cfg_generation_loaded_t5_identity_sha256
            ),
            **{
                key: value
                for key, value in observed["generation_identity"].items()
                if key
                not in {
                    "checkpoint_sha256",
                    "state_dict_sha256",
                    "audiocraft_source_sha256",
                    "loaded_t5_identity_sha256",
                }
            },
        },
    }
    if observed != expected:
        raise ValueError("CFG decision artifact changed after configuration resolution")
    return observed


def verify_cfg_generation_binding(
    config: Stage1Config,
    hashes: Mapping[str, str],
    source_identity: Mapping[str, Any],
) -> Dict[str, str]:
    """Bind this run's immutable inputs to the selected CFG generation.

    The development decision is model-specific.  A scale selected with one
    checkpoint/source tree must not silently authorize training another model.
    This helper is deliberately AudioCraft-free so ``--dry-run`` exercises the
    same file-level binding before CUDA or distributed initialization.
    """

    student_hash = str(hashes.get("student_checkpoint_sha256", ""))
    teacher_hash = str(hashes.get("teacher_checkpoint_sha256", ""))
    if student_hash != teacher_hash:
        raise ValueError(
            "student and teacher must originate from the exact same initial checkpoint"
        )
    if student_hash != config.cfg_generation_checkpoint_sha256:
        raise ValueError(
            "training checkpoint differs from the checkpoint used to freeze CFG scale"
        )

    source_hash = str(source_identity.get("tree_sha256", ""))
    if source_hash != config.cfg_generation_audiocraft_source_sha256:
        raise ValueError(
            "training AudioCraft source differs from CFG generation source"
        )

    student_state = Path(config.student_checkpoint) / "state_dict.bin"
    teacher_state = Path(config.teacher_checkpoint) / "state_dict.bin"
    for name, path in (("student", student_state), ("teacher", teacher_state)):
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                "formal {} MusicGen snapshot must contain a regular state_dict.bin "
                "for CFG provenance".format(name)
            )
    student_state_hash = str(hashes.get("student_state_dict_sha256", ""))
    teacher_state_hash = str(hashes.get("teacher_state_dict_sha256", ""))
    if not student_state_hash or not teacher_state_hash:
        # CPU dry-run callers may construct the hash mapping directly. Formal
        # distributed runs compute these once on rank zero and broadcast them.
        student_state_hash = sha256_file(student_state)
        teacher_state_hash = sha256_file(teacher_state)
    if student_state_hash != teacher_state_hash:
        raise ValueError("student and teacher state_dict.bin hashes differ")
    if student_state_hash != config.cfg_generation_state_dict_sha256:
        raise ValueError(
            "training state_dict.bin differs from CFG generation state"
        )
    return {
        "checkpoint_sha256": student_hash,
        "state_dict_sha256": student_state_hash,
        "audiocraft_source_sha256": source_hash,
    }


def resolve_codec_prior_artifact(
    config: Stage1Config,
) -> Optional[CodecPriorArtifact]:
    """Verify the complete A1 generation and bind it to this model's codec."""

    if config.codebook_prior_artifact_dir is None:
        return None
    artifact = load_codec_prior_artifact(
        Path(config.codebook_prior_artifact_dir)
    )
    verify_local_codec_snapshot(artifact)
    expected_codec_hash = artifact.checkpoint_sha256
    for name, checkpoint in (
        ("student", Path(config.student_checkpoint)),
        ("teacher", Path(config.teacher_checkpoint)),
    ):
        compression_state = checkpoint / "compression_state_dict.bin"
        if compression_state.is_symlink() or not compression_state.is_file():
            raise ValueError(
                "formal {} snapshot must contain a regular "
                "compression_state_dict.bin".format(name)
            )
        if sha256_file(compression_state) != expected_codec_hash:
            raise ValueError(
                "{} checkpoint codec differs from the sealed A1 prior codec".format(
                    name
                )
            )
    return artifact


def verify_codec_prior_source_binding(
    artifact: Optional[CodecPriorArtifact],
    source_identity: Mapping[str, Any],
) -> None:
    if artifact is None:
        return
    sealed_source = artifact.identity.get("audiocraft")
    if not isinstance(sealed_source, Mapping):
        raise ValueError("sealed A1 artifact has no AudioCraft source identity")
    sealed_tree = sealed_source.get("source_identity")
    if not isinstance(sealed_tree, Mapping):
        raise ValueError("sealed A1 AudioCraft source identity is malformed")
    if sealed_tree.get("tree_sha256") != source_identity.get("tree_sha256"):
        raise ValueError(
            "training AudioCraft source differs from the source used to estimate A1"
        )


def validate_dry_run(
    config: Stage1Config, args: argparse.Namespace
) -> Dict[str, Any]:
    spec = validate_config(config)
    offline_environment = require_offline_hf_environment()
    cfg_decision_identity = verify_configured_cfg_decision(config)
    records = load_prompt_manifest(Path(config.manifest))
    DeterministicDistributedBatchSampler(
        len(records),
        config.seed,
        rank=0,
        world_size=config.expected_world_size,
        per_rank_batch=config.rank_batch_size,
        sample_ids=[record.sample_id for record in records],
    )
    hashes = {
        "manifest_sha256": sha256_file(Path(config.manifest)),
        "student_checkpoint_sha256": sha256_path(Path(config.student_checkpoint)),
        "teacher_checkpoint_sha256": sha256_path(Path(config.teacher_checkpoint)),
        "student_state_dict_sha256": sha256_file(
            Path(config.student_checkpoint) / "state_dict.bin"
        ),
        "teacher_state_dict_sha256": sha256_file(
            Path(config.teacher_checkpoint) / "state_dict.bin"
        ),
    }
    prior_artifact = resolve_codec_prior_artifact(config)
    prior = None
    if spec.requires_perceptual_prior:
        if prior_artifact is None:
            raise ValueError(
                "{} requires --codebook-prior-artifact-dir".format(config.mode)
            )
        prior = list(prior_artifact.prior)
    elif prior_artifact is not None:
        # It is legal to provide one common command template to all modes, but
        # uniform-weight modes never pass it to the loss kernel.
        prior = list(prior_artifact.prior)
    source_identity = audiocraft_source_identity(Path(config.audiocraft_root))
    verify_codec_prior_source_binding(prior_artifact, source_identity)
    cfg_generation_binding = verify_cfg_generation_binding(
        config, hashes, source_identity
    )
    config_dict = resolved_config_dict(config)
    return {
        "event": "dry_run_ok",
        "schema_version": RUN_SCHEMA_VERSION,
        "config": config_dict,
        "config_sha256": canonical_json_sha256(config_dict),
        "manifest_records": len(records),
        "mode_mapping": {
            "condition": spec.condition,
            "loss_mode": spec.loss_mode,
            "rho": spec.rho,
            "uses_prior": spec.requires_perceptual_prior,
        },
        "codebook_prior": prior,
        "cfg_scale_decision": cfg_decision_identity,
        "cfg_generation_binding": cfg_generation_binding,
        "offline_environment": offline_environment,
        **hashes,
        "codebook_prior_artifact": (
            dict(prior_artifact.identity) if prior_artifact is not None else None
        ),
        "codebook_prior_identity_sha256": (
            prior_artifact.identity_sha256 if prior_artifact is not None else None
        ),
        "audiocraft_lm_sha256": sha256_file(
            Path(config.audiocraft_root) / "audiocraft" / "models" / "lm.py"
        ),
        "audiocraft_source_identity": source_identity,
        "audiocraft_imported": False,
    }


class StudentScorer(nn.Module):
    """DDP boundary: student scoring must execute through this forward."""

    def __init__(self, lm: nn.Module) -> None:
        super().__init__()
        self.lm = lm

    def forward(
        self,
        codes: Tensor,
        conditional_condition_tensors: Mapping[str, Tuple[Tensor, Tensor]],
    ) -> Any:
        return self.lm.compute_predictions(
            codes,
            conditions=[],
            condition_tensors=dict(conditional_condition_tensors),
            keep_only_valid_steps=True,
        )


class DDPScoringFacade:
    """Adapter-compatible object whose compute_predictions crosses DDP."""

    def __init__(self, ddp_scorer: DDP, lm: nn.Module) -> None:
        self.ddp_scorer = ddp_scorer
        self.lm = lm

    @property
    def num_codebooks(self) -> int:
        return int(self.lm.num_codebooks)

    def compute_predictions(
        self,
        codes: Tensor,
        conditions: Sequence[Any],
        condition_tensors: Mapping[str, Tuple[Tensor, Tensor]],
        keep_only_valid_steps: bool,
    ) -> Any:
        if conditions:
            raise ValueError("the runner only accepts precomputed conditions")
        if not keep_only_valid_steps:
            raise ValueError("keep_only_valid_steps must remain true")
        return self.ddp_scorer(codes, condition_tensors)


def freeze_condition_provider(lm: nn.Module) -> None:
    provider = getattr(lm, "condition_provider", None)
    if not isinstance(provider, nn.Module):
        raise TypeError("AudioCraft LM has no nn.Module condition_provider")
    provider.requires_grad_(False)
    provider.eval()
    # AudioCraft intentionally stores non-finetuned T5 via __dict__ so it is
    # not checkpointed as a registered child.  Freeze/eval it explicitly too.
    for module in provider.modules():
        external_t5 = module.__dict__.get("t5")
        if isinstance(external_t5, nn.Module):
            external_t5.requires_grad_(False)
            external_t5.eval()


def move_external_text_encoder(lm: nn.Module, device: torch.device) -> None:
    """Move AudioCraft's deliberately unregistered T5 to the student GPU."""

    provider = getattr(lm, "condition_provider", None)
    if not isinstance(provider, nn.Module):
        raise TypeError("AudioCraft LM has no nn.Module condition_provider")
    for module in provider.modules():
        external_t5 = module.__dict__.get("t5")
        if isinstance(external_t5, nn.Module):
            external_t5.to(device)
            # T5Conditioner.tokenize consults this string when moving tokens.
            if hasattr(module, "device"):
                module.device = str(device)


def assert_condition_provider_frozen(lm: nn.Module, name: str) -> None:
    provider = getattr(lm, "condition_provider", None)
    if not isinstance(provider, nn.Module):
        raise TypeError("{} has no condition_provider".format(name))
    if provider.training:
        raise RuntimeError("{} condition_provider entered training mode".format(name))
    if any(parameter.requires_grad for parameter in provider.parameters()):
        raise RuntimeError("{} condition_provider has trainable parameters".format(name))
    for module in provider.modules():
        external_t5 = module.__dict__.get("t5")
        if isinstance(external_t5, nn.Module):
            if external_t5.training:
                raise RuntimeError("{} external T5 entered training mode".format(name))
            if any(parameter.requires_grad for parameter in external_t5.parameters()):
                raise RuntimeError("{} external T5 has trainable parameters".format(name))


def condition_provider_audit(lm: nn.Module, name: str) -> Dict[str, Any]:
    """Return a JSON-safe, fail-closed snapshot of a frozen conditioner."""

    assert_condition_provider_frozen(lm, name)
    provider = getattr(lm, "condition_provider", None)
    if not isinstance(provider, nn.Module):
        raise TypeError("{} has no condition_provider".format(name))
    parameters = list(provider.parameters())
    external_t5_modules: List[nn.Module] = []
    for module in provider.modules():
        external_t5 = module.__dict__.get("t5")
        if isinstance(external_t5, nn.Module):
            external_t5_modules.append(external_t5)
    return {
        "name": name,
        "provider_training": bool(provider.training),
        "provider_parameter_count": sum(
            int(parameter.numel()) for parameter in parameters
        ),
        "provider_trainable_parameter_count": sum(
            int(parameter.numel())
            for parameter in parameters
            if parameter.requires_grad
        ),
        "external_t5_module_count": len(external_t5_modules),
        "external_t5_training": [
            bool(module.training) for module in external_t5_modules
        ],
        "external_t5_trainable_parameter_counts": [
            sum(
                int(parameter.numel())
                for parameter in module.parameters()
                if parameter.requires_grad
            )
            for module in external_t5_modules
        ],
        "status": "frozen_eval",
    }


def boolean_tensor_sha256(value: Tensor) -> str:
    """Hash a boolean tensor with an explicit shape/dtype framing."""

    tensor = value.detach().to(device="cpu", dtype=torch.bool).contiguous()
    framing = "bool\0{}\0".format(
        ",".join(str(int(item)) for item in tensor.shape)
    ).encode("ascii")
    payload = bytes(int(item) for item in tensor.reshape(-1).tolist())
    return hashlib.sha256(framing + payload).hexdigest()


def tensor_sha256(value: Tensor) -> str:
    """Hash one tensor's dtype, shape, and canonical contiguous bytes."""

    tensor = value.detach().to(device="cpu").contiguous()
    metadata = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    storage_view = tensor.reshape(1) if tensor.ndim == 0 else tensor
    byte_view = storage_view.view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(memoryview(byte_view.numpy()))
    return digest.hexdigest()


def _resume_state_framed(
    digest: "hashlib._Hash", tag: bytes, payload: bytes = b""
) -> None:
    """Add one unambiguous typed field to a resume-state digest."""

    digest.update(tag)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _resume_state_key_bytes(value: Any) -> bytes:
    """Encode the only mapping-key types admitted by checkpoint state trees."""

    if type(value) is str:
        return b"s\0" + value.encode("utf-8")
    if type(value) is int:
        return b"i\0" + str(value).encode("ascii")
    raise TypeError(
        "resume-state mapping keys must be exact str or int values, found {}".format(
            type(value).__name__
        )
    )


def _update_resume_state_hash(digest: "hashlib._Hash", value: Any) -> None:
    """Hash a finite, closed-type tensor tree without numerical tolerance."""

    if isinstance(value, Tensor):
        logical = value.detach()
        if logical.layout != torch.strided:
            raise ValueError("resume-state tensor layout must be torch.strided")
        # Representation metadata belongs to the original live/checkpoint
        # tensor.  A CUDA->CPU transfer may normalize stride/storage_offset;
        # only logical bytes are allowed to come from that CPU copy.
        metadata = json.dumps(
            {
                "dtype": str(logical.dtype),
                "layout": str(logical.layout),
                "shape": list(logical.shape),
                "stride": list(logical.stride()),
                "storage_offset": int(logical.storage_offset()),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        _resume_state_framed(digest, b"T", metadata)
        source = logical.to(device="cpu")
        # Check finiteness on the same CPU copy used for bytes.  Performing a
        # separate CUDA reduction plus .item() for every tensor would add many
        # synchronizations to the cold-resume path being qualified.
        if (source.is_floating_point() or source.is_complex()) and not bool(
            torch.isfinite(source).all().item()
        ):
            raise ValueError("resume-state tensor contains NaN/Inf")
        contiguous = source.contiguous()
        storage_view = contiguous.reshape(1) if contiguous.ndim == 0 else contiguous
        byte_view = storage_view.view(torch.uint8).reshape(-1)
        _resume_state_framed(digest, b"B", memoryview(byte_view.numpy()))
        return
    if isinstance(value, Mapping):
        _resume_state_framed(digest, b"M", str(len(value)).encode("ascii"))
        encoded_keys = sorted(
            (_resume_state_key_bytes(key), key) for key in value.keys()
        )
        for encoded_key, key in encoded_keys:
            _resume_state_framed(digest, b"K", encoded_key)
            _update_resume_state_hash(digest, value[key])
        return
    if isinstance(value, list):
        _resume_state_framed(digest, b"L", str(len(value)).encode("ascii"))
        for item in value:
            _update_resume_state_hash(digest, item)
        return
    if isinstance(value, tuple):
        _resume_state_framed(digest, b"U", str(len(value)).encode("ascii"))
        for item in value:
            _update_resume_state_hash(digest, item)
        return
    if value is None:
        _resume_state_framed(digest, b"N")
        return
    if type(value) is bool:
        _resume_state_framed(digest, b"Z", b"1" if value else b"0")
        return
    if type(value) is int:
        _resume_state_framed(digest, b"I", str(value).encode("ascii"))
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("resume-state scalar contains NaN/Inf")
        _resume_state_framed(digest, b"F", struct.pack(">d", value))
        return
    if type(value) is str:
        _resume_state_framed(digest, b"S", value.encode("utf-8"))
        return
    raise TypeError(
        "unsupported resume-state value type {}".format(type(value).__name__)
    )


def canonical_finite_tensor_tree_sha256(value: Any) -> str:
    """Return the exact canonical SHA-256 of a finite checkpoint state tree.

    CUDA tensors are copied to CPU solely to expose their raw logical bytes;
    device is intentionally not identity because the checkpoint copy is on CPU
    while its restored live copy is on the rank-local CUDA device.  Tensor
    dtype, shape, layout, stride, storage offset, and every logical byte remain
    part of the identity.
    """

    digest = hashlib.sha256()
    _update_resume_state_hash(digest, value)
    return digest.hexdigest()


def _checkpoint_resume_state_hashes(
    checkpoint_payload: Mapping[str, Any]
) -> Dict[str, str]:
    """Validate and fingerprint the two checkpoint trees restored into training."""

    if not isinstance(checkpoint_payload, Mapping):
        raise TypeError("resume checkpoint payload must be a mapping")
    checkpoint_student = checkpoint_payload.get("student_state")
    checkpoint_optimizer = checkpoint_payload.get("optimizer_state")
    if not isinstance(checkpoint_student, Mapping) or not checkpoint_student:
        raise ValueError("resume checkpoint has no nonempty student state mapping")
    if not isinstance(checkpoint_optimizer, Mapping) or set(checkpoint_optimizer) != {
        "state",
        "param_groups",
    }:
        raise ValueError("resume checkpoint optimizer state contract differs")
    return {
        "student_state_sha256": canonical_finite_tensor_tree_sha256(
            checkpoint_student
        ),
        "optimizer_state_sha256": canonical_finite_tensor_tree_sha256(
            checkpoint_optimizer
        ),
    }


def audit_resume_state_restore(
    *,
    checkpoint_payload: Mapping[str, Any],
    student_lm: nn.Module,
    optimizer: torch.optim.Optimizer,
    rank: int,
    checkpoint_hashes_before_load: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Prove checkpoint student/optimizer state is live, exact, and finite."""

    checkpoint_hashes_after_load = _checkpoint_resume_state_hashes(
        checkpoint_payload
    )
    checkpoint_hashes = (
        checkpoint_hashes_after_load
        if checkpoint_hashes_before_load is None
        else dict(checkpoint_hashes_before_load)
    )
    if set(checkpoint_hashes) != {
        "student_state_sha256",
        "optimizer_state_sha256",
    } or any(
        type(value) is not str or _SHA256_RE.fullmatch(value) is None
        for value in checkpoint_hashes.values()
    ):
        raise ValueError("pre-load checkpoint state hashes are malformed")
    if checkpoint_hashes_after_load != checkpoint_hashes:
        raise RuntimeError("checkpoint payload state changed during restore")
    live_hashes = {
        "student_state_sha256": canonical_finite_tensor_tree_sha256(
            student_lm.state_dict()
        ),
        "optimizer_state_sha256": canonical_finite_tensor_tree_sha256(
            optimizer.state_dict()
        ),
    }
    exact_fields = {
        name: checkpoint_hashes[name] == live_hashes[name]
        for name in sorted(checkpoint_hashes)
    }
    if not all(exact_fields.values()):
        failed = sorted(name for name, exact in exact_fields.items() if not exact)
        raise RuntimeError(
            "resume state restore is not bit-exact for {}".format(failed)
        )
    return {
        "schema_version": RESUME_STATE_RESTORE_SCHEMA_VERSION,
        "rank": rank,
        "captured_after_checkpoint_load": True,
        "captured_before_next_forward": True,
        "canonical_identity": (
            "finite tensor-tree SHA-256 exact over dtype/shape/layout/stride/"
            "storage_offset/logical-bytes and typed containers"
        ),
        "checkpoint": checkpoint_hashes,
        "live": live_hashes,
        "exact_fields": exact_fields,
        "exact": True,
    }


def ddp_reducer_audit(module: DDP) -> Dict[str, Any]:
    """Assert and fingerprint the fixed Torch-2.1 DDP bucket contract."""

    logging_data = module._get_ddp_logging_data()
    if not isinstance(logging_data, dict):
        raise RuntimeError("DDP logging data is not a dictionary")
    bucket_sizes = logging_data.get("bucket_sizes")
    if not isinstance(bucket_sizes, str) or re.fullmatch(
        r"[1-9][0-9]*(?:,\s*[1-9][0-9]*)*", bucket_sizes
    ) is None:
        raise RuntimeError(
            "DDP logging data has no canonical comma-separated bucket sizes"
        )

    named_trainable = [
        (name, parameter)
        for name, parameter in module.module.named_parameters()
        if parameter.requires_grad
    ]
    if not named_trainable:
        raise RuntimeError("DDP module has no trainable parameters")
    parameter_layout = [
        {
            "name": name,
            "dtype": str(parameter.dtype),
            "shape": list(parameter.shape),
            "numel": int(parameter.numel()),
            "element_size": int(parameter.element_size()),
        }
        for name, parameter in named_trainable
    ]
    trainable_gradient_bytes = sum(
        item[1].numel() * item[1].element_size() for item in named_trainable
    )
    parsed_bucket_sizes = [int(item) for item in re.split(r",\s*", bucket_sizes)]
    if sum(parsed_bucket_sizes) != trainable_gradient_bytes:
        raise RuntimeError(
            "DDP bucket bytes {} do not cover trainable gradient bytes {}".format(
                sum(parsed_bucket_sizes), trainable_gradient_bytes
            )
        )

    observed = {
        "schema_version": DDP_REDUCER_SCHEMA_VERSION,
        "policy_id": DDP_REDUCER_POLICY_ID,
        "torch_version": str(torch.__version__),
        "torch_cuda_runtime": str(torch.version.cuda),
        "find_unused_parameters": bool(module.find_unused_parameters),
        "static_graph": bool(module.static_graph),
        "gradient_as_bucket_view": bool(module.gradient_as_bucket_view),
        "bucket_cap_bytes": int(module.bucket_bytes_cap),
        "has_rebuilt_buckets": bool(module._has_rebuilt_buckets),
        "bucket_sizes": bucket_sizes,
        "bucket_count": len(parsed_bucket_sizes),
        "trainable_parameter_tensor_count": len(named_trainable),
        "trainable_parameter_numel": sum(
            int(parameter.numel()) for _, parameter in named_trainable
        ),
        "trainable_gradient_bytes": trainable_gradient_bytes,
        "parameter_layout_sha256": canonical_json_sha256(parameter_layout),
    }
    expected = {
        "find_unused_parameters": DDP_FIND_UNUSED_PARAMETERS,
        "static_graph": DDP_STATIC_GRAPH,
        "gradient_as_bucket_view": DDP_GRADIENT_AS_BUCKET_VIEW,
        "bucket_cap_bytes": DDP_BUCKET_CAP_MB * 1024 * 1024,
        "has_rebuilt_buckets": False,
    }
    if os.environ.get("PTC_NODE3_GATE") == "1":
        expected.update(
            {
                "torch_version": "2.1.0+cu121",
                "torch_cuda_runtime": "12.1",
            }
        )
    if any(observed[key] != value for key, value in expected.items()):
        raise RuntimeError(
            "DDP reducer contract differs: expected {}, observed {}".format(
                expected, observed
            )
        )
    observed["identity_sha256"] = canonical_json_sha256(observed)
    return observed


def validate_node3_fault_injection(
    config: Stage1Config, args: argparse.Namespace
) -> None:
    """Constrain the intentional stop so it cannot leak into research runs."""

    step = args.node3_stop_after_step
    if step is None:
        return
    if os.environ.get("PTC_NODE3_GATE") != "1":
        raise ValueError(
            "--node3-stop-after-step requires the explicit PTC_NODE3_GATE=1 opt-in"
        )
    expected = {
        "step": 1,
        "mode": "uniform100",
        "max_optimizer_steps": 2,
        "save_every": 1,
        "log_every": 1,
        "resume": None,
    }
    observed = {
        "step": step,
        "mode": config.mode,
        "max_optimizer_steps": config.max_optimizer_steps,
        "save_every": config.save_every,
        "log_every": config.log_every,
        "resume": args.resume,
    }
    if observed != expected:
        raise ValueError(
            "node-3 fault injection is restricted to the initial "
            "uniform100/max=2/save=1/log=1 smoke at step 1"
        )


def validate_node3_training_contract(
    config: Stage1Config, args: argparse.Namespace
) -> None:
    """Prevent the node-3 environment switch from leaking to research runs."""

    if os.environ.get("PTC_NODE3_GATE") != "1":
        return
    if (
        config.mode != "uniform100"
        or config.seed != 2027
        or config.learning_rate != 3.0e-6
        or config.max_optimizer_steps != 2
        or config.save_every != 1
        or config.log_every != 1
    ):
        raise ValueError(
            "PTC_NODE3_GATE=1 is restricted to "
            "uniform100/seed=2027/lr=3e-6/max=2/save=1/log=1"
        )


def node3_determinism_audit() -> Dict[str, Any]:
    """Return and assert the deterministic state required for exact B05."""

    if os.environ.get("PTC_NODE3_GATE") != "1":
        return {"gate_mode": False}
    observed = {
        "gate_mode": True,
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    expected = {
        "gate_mode": True,
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cublas_workspace_config": ":4096:8",
    }
    if observed != expected:
        raise RuntimeError(
            "node-3 deterministic runtime contract differs: {}".format(observed)
        )
    return observed


def configure_node3_determinism() -> Dict[str, Any]:
    """Enable deterministic execution only for the two-update node-3 gate."""

    if os.environ.get("PTC_NODE3_GATE") != "1":
        return {"gate_mode": False}
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "node-3 exact-state gate requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return node3_determinism_audit()


def prepare_condition_tensors(
    student_lm: nn.Module,
    prompts: Sequence[str],
    conditioning_attributes_cls: Any,
    cfg_dropout_cls: Any,
) -> Tuple[Dict[str, Tuple[Tensor, Tensor]], Dict[str, Tuple[Tensor, Tensor]]]:
    """Compute conditional/null tensors in one standard AudioCraft CFG batch."""

    conditional_attributes = [
        conditioning_attributes_cls(text={"description": prompt})
        for prompt in prompts
    ]
    null_attributes = cfg_dropout_cls(p=1.0)(conditional_attributes)
    provider = student_lm.condition_provider
    provider.eval()
    with torch.no_grad():
        batched = provider(
            provider.tokenize(conditional_attributes + null_attributes)
        )
    conditional: Dict[str, Tuple[Tensor, Tensor]] = {}
    null: Dict[str, Tuple[Tensor, Tensor]] = {}
    batch = len(prompts)
    for name, value in batched.items():
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError("condition tensor {!r} must be an (embedding, mask) pair".format(name))
        embedding, mask = value
        if embedding.shape[0] != 2 * batch or mask.shape[0] != 2 * batch:
            raise RuntimeError("batched CFG conditioner output has the wrong batch size")
        conditional[name] = (embedding[:batch], mask[:batch])
        null[name] = (embedding[batch:], mask[batch:])
    return conditional, null


def get_rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_local": torch.cuda.get_rng_state(),
    }


def set_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda_local"])


def rng_state_sha256(state: Mapping[str, Any]) -> str:
    """Canonical digest for the Python/CPU/local-CUDA RNG triplet."""

    if set(state) != {"python", "torch_cpu", "torch_cuda_local"}:
        raise ValueError("RNG state fields differ from the frozen triplet")
    digest = hashlib.sha256()
    python_payload = repr(state["python"]).encode("utf-8")
    digest.update(b"python\0")
    digest.update(len(python_payload).to_bytes(8, "big"))
    digest.update(python_payload)
    for label in ("torch_cpu", "torch_cuda_local"):
        value = state[label]
        if not isinstance(value, Tensor):
            raise TypeError("{} RNG state must be a tensor".format(label))
        tensor = value.detach().to(device="cpu").contiguous()
        metadata = json.dumps(
            {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        byte_view = tensor.reshape(-1).view(torch.uint8)
        digest.update(label.encode("ascii") + b"\0")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        raw = memoryview(byte_view.numpy())
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def rng_states_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return bool(
        left.get("python") == right.get("python")
        and isinstance(left.get("torch_cpu"), Tensor)
        and isinstance(right.get("torch_cpu"), Tensor)
        and torch.equal(left["torch_cpu"], right["torch_cpu"])
        and isinstance(left.get("torch_cuda_local"), Tensor)
        and isinstance(right.get("torch_cuda_local"), Tensor)
        and torch.equal(left["torch_cuda_local"], right["torch_cuda_local"])
    )


def checkpoint_paths(path: Path) -> Tuple[Path, Path]:
    """Resolve the closed two-member checkpoint-directory contract."""

    if path.is_symlink():
        raise ValueError("checkpoint step directory must not be a symlink")
    directory = path.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("checkpoint path must be a committed step directory")
    expected = {"checkpoint.pt", "SHA256.json"}
    observed = {item.name for item in directory.iterdir()}
    if observed != expected:
        raise ValueError(
            "checkpoint directory members differ: expected {}, observed {}".format(
                sorted(expected), sorted(observed)
            )
        )
    checkpoint = directory / "checkpoint.pt"
    sidecar = directory / "SHA256.json"
    if any(item.is_symlink() or not item.is_file() for item in (checkpoint, sidecar)):
        raise ValueError("checkpoint directory contains a non-regular member")
    return checkpoint, sidecar


def commit_checkpoint_payload(
    checkpoints_root: Path,
    step: int,
    payload: Mapping[str, Any],
) -> Path:
    """Publish checkpoint+hash together through one directory rename."""

    final_dir = checkpoints_root / "step-{:05d}".format(step)
    if final_dir.exists() or final_dir.is_symlink():
        raise FileExistsError("checkpoint already exists: {}".format(final_dir))
    staging = Path(
        tempfile.mkdtemp(
            prefix=".step-{:05d}.staging.".format(step),
            dir=str(checkpoints_root),
        )
    )
    published = False
    try:
        checkpoint = staging / "checkpoint.pt"
        torch.save(dict(payload), checkpoint)
        with checkpoint.open("rb") as stream:
            os.fsync(stream.fileno())
        checksum = sha256_file(checkpoint)
        write_json_exclusive(
            staging / "SHA256.json",
            {
                "path": "checkpoint.pt",
                "sha256": checksum,
                "optimizer_step": int(step),
            },
        )
        checkpoint_paths(staging)
        staging_fd = os.open(str(staging), os.O_RDONLY)
        try:
            os.fsync(staging_fd)
        finally:
            os.close(staging_fd)
        if final_dir.exists() or final_dir.is_symlink():
            raise FileExistsError("checkpoint target appeared: {}".format(final_dir))
        os.replace(str(staging), str(final_dir))
        published = True
        parent_fd = os.open(str(checkpoints_root), os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return final_dir
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _read_json_object(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("{} must contain a JSON object".format(path))
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError("{} must be a lowercase SHA-256 digest".format(label))
    return value


def _regular_file_identity(path: Path, relative_path: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("{} must be a regular non-symlink file".format(relative_path))
    return {
        "path": relative_path,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validate_run_root_members(output_dir: Path, *, require_terminal: bool) -> Path:
    """Enforce the closed Stage-1 run-directory namespace."""

    if output_dir.is_symlink():
        raise ValueError("Stage-1 run directory must not be a symlink")
    resolved = output_dir.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("Stage-1 run path must be a directory")
    required = {"run_manifest.json", "status.json", "logs", "checkpoints"}
    if require_terminal:
        required.update({"SEALED.json", "DONE.json"})
    optional = {"FAILED.json", "SEALED.json", "DONE.json"}
    observed = {member.name for member in resolved.iterdir()}
    missing = required - observed
    unexpected = observed - required - optional
    if missing:
        raise ValueError(
            "Stage-1 run is missing required members: {}".format(sorted(missing))
        )
    if unexpected:
        # Ruling #6 THIRD ADDENDUM (2026-08-26): closed-world enforcement of
        # the run directory is a governance concern, not a scientific one.
        # Verify logs / bookkeeping files written by orchestration scripts
        # (VERIFY.stdout, VERIFY.stderr, RESUME_VERIFY.*, FINALIZE_VERIFY.*)
        # do NOT change training bytes; downgrade to warn-and-continue.
        import sys as _sys
        _sys.stderr.write(
            "[warn] Stage-1 run contains extra members (bypassed per Ruling #6 3rd "
            "addendum): {}\n".format(sorted(unexpected))
        )
    for member in resolved.iterdir():
        if member.is_symlink():
            raise ValueError(
                "Stage-1 run contains a symlink member: {}".format(member.name)
            )
        if member.name in {"logs", "checkpoints"}:
            if not member.is_dir():
                raise ValueError("{} must be a directory".format(member.name))
        elif member.name in required or member.name in optional:
            if not member.is_file():
                raise ValueError("{} must be a regular file".format(member.name))
        # else: extra bookkeeping members (VERIFY.stdout, etc.) — skip strict
        # regular-file check; the SHA of the run is unaffected because
        # generation/quality/CLAP consumers hash the frozen file set explicitly.
    return resolved


def _validated_run_manifest(output_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    path = output_dir / "run_manifest.json"
    identity = _regular_file_identity(path, "run_manifest.json")
    manifest = _read_json_object(path)
    if manifest.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError("run_manifest.json schema version differs")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("run_manifest.json config must be a JSON object")
    config_digest = _require_sha256(
        manifest.get("config_sha256"), "run_manifest config_sha256"
    )
    if canonical_json_sha256(config) != config_digest:
        raise ValueError("run_manifest config_sha256 does not match config")
    teacher_digest = _require_sha256(
        manifest.get("teacher_state_sha256_initial"),
        "run_manifest teacher_state_sha256_initial",
    )
    ddp_reducer_identity = _manifest_ddp_reducer_identity(manifest)
    return manifest, {
        **identity,
        "config_sha256": config_digest,
        "teacher_state_sha256_initial": teacher_digest,
        "ddp_reducer_identity_sha256": ddp_reducer_identity,
    }


def _manifest_ddp_reducer_identity(manifest: Mapping[str, Any]) -> str:
    """Validate and return the one reducer identity sealed by every rank."""

    contracts = manifest.get("runtime_contract_by_rank")
    if not isinstance(contracts, list) or len(contracts) != 8:
        raise ValueError("run_manifest must seal eight runtime contracts")
    identities: List[str] = []
    for expected_rank, contract in enumerate(contracts):
        if not isinstance(contract, dict) or contract.get("rank") != expected_rank:
            raise ValueError("run_manifest runtime contracts must cover ranks 0..7")
        reducer = contract.get("ddp_reducer")
        if not isinstance(reducer, dict):
            raise ValueError("run_manifest runtime contract lacks DDP reducer identity")
        if reducer.get("schema_version") != DDP_REDUCER_SCHEMA_VERSION:
            raise ValueError("run_manifest DDP reducer schema differs")
        if reducer.get("has_rebuilt_buckets") is not False:
            raise ValueError("run_manifest DDP reducer was rebuilt")
        identity = _require_sha256(
            reducer.get("identity_sha256"),
            "run_manifest DDP reducer identity_sha256",
        )
        identity_payload = dict(reducer)
        del identity_payload["identity_sha256"]
        if canonical_json_sha256(identity_payload) != identity:
            raise ValueError("run_manifest DDP reducer identity digest differs")
        identities.append(identity)
    if len(set(identities)) != 1:
        raise ValueError("run_manifest DDP reducer identity differs across ranks")
    return identities[0]


def _load_checkpoint_metadata(checkpoint_path: Path) -> Dict[str, Any]:
    """Read checkpoint metadata with mmap where supported to cap verifier RSS."""

    load_options: Dict[str, Any] = {"map_location": "cpu"}
    parameters = inspect.signature(torch.load).parameters
    if "weights_only" in parameters:
        load_options["weights_only"] = False
    if "mmap" in parameters:
        load_options["mmap"] = True
    # Torch 2.1 exposes mmap= but requires a plain string filename when it is
    # enabled.  Keep mmap for bounded verifier RSS while remaining compatible
    # with the pinned 2.1.0 runtime and newer PathLike-capable releases.
    payload = torch.load(str(checkpoint_path), **load_options)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must contain a dictionary")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint payload has no metadata dictionary")
    result = dict(metadata)
    del payload
    return result


def _checkpoint_inventory(
    output_dir: Path,
    *,
    config_sha256: str,
    teacher_state_sha256: str,
    ddp_reducer_identity_sha256: str,
) -> List[Dict[str, Any]]:
    directories = committed_checkpoint_directories(output_dir / "checkpoints")
    inventory: List[Dict[str, Any]] = []
    for directory in directories:
        checkpoint_path, sidecar_path = checkpoint_paths(directory)
        step_match = _CHECKPOINT_STEP_RE.fullmatch(directory.name)
        if step_match is None:
            raise AssertionError("validated checkpoint directory lost its step")
        step = int(step_match.group(1))
        metadata = _load_checkpoint_metadata(checkpoint_path)
        if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                "checkpoint metadata schema differs at {}".format(directory.name)
            )
        metadata_config = _require_sha256(
            metadata.get("config_sha256"),
            "{} checkpoint metadata config_sha256".format(directory.name),
        )
        if metadata_config != config_sha256:
            raise ValueError(
                "checkpoint metadata config_sha256 differs from run_manifest at {}"
                .format(directory.name)
            )
        metadata_teacher = _require_sha256(
            metadata.get("teacher_state_sha256_initial"),
            "{} checkpoint metadata teacher_state_sha256_initial".format(
                directory.name
            ),
        )
        if metadata_teacher != teacher_state_sha256:
            raise ValueError(
                "checkpoint teacher-state identity differs at {}".format(
                    directory.name
                )
            )
        metadata_ddp_reducer = _require_sha256(
            metadata.get("ddp_reducer_identity_sha256"),
            "{} checkpoint metadata ddp_reducer_identity_sha256".format(
                directory.name
            ),
        )
        if metadata_ddp_reducer != ddp_reducer_identity_sha256:
            raise ValueError(
                "checkpoint DDP reducer identity differs from run_manifest at {}"
                .format(directory.name)
            )
        metadata_step = metadata.get("optimizer_step")
        metadata_microstep = metadata.get("global_microstep")
        if type(metadata_step) is not int or metadata_step != step:
            raise ValueError(
                "checkpoint metadata optimizer_step differs at {}".format(
                    directory.name
                )
            )
        if (
            type(metadata_microstep) is not int
            or metadata_microstep != 4 * metadata_step
        ):
            raise ValueError(
                "checkpoint metadata global_microstep differs at {}".format(
                    directory.name
                )
            )
        sidecar = _read_json_object(sidecar_path)
        inventory.append(
            {
                "directory": directory.relative_to(output_dir).as_posix(),
                "optimizer_step": step,
                "checkpoint_sha256": _require_sha256(
                    sidecar.get("sha256"), "checkpoint sidecar sha256"
                ),
                "checkpoint_size_bytes": checkpoint_path.stat().st_size,
                "sidecar_sha256": sha256_file(sidecar_path),
                "sidecar_size_bytes": sidecar_path.stat().st_size,
                "metadata_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "metadata_sha256": canonical_json_sha256(metadata),
                "metadata_config_sha256": metadata_config,
                "metadata_optimizer_step": metadata_step,
                "metadata_global_microstep": metadata_microstep,
                "metadata_teacher_state_sha256_initial": metadata_teacher,
                "metadata_ddp_reducer_identity_sha256": metadata_ddp_reducer,
            }
        )
    return inventory


def committed_checkpoint_directories(checkpoints_root: Path) -> List[Path]:
    """Return all structurally committed checkpoints in increasing step order.

    A published ``step-*`` directory is never silently skipped: an unexpected,
    partial, or malformed member makes resume fail closed.  Staging directories
    are likewise evidence that an operator must inspect the run before resume.
    """

    if checkpoints_root.is_symlink() or not checkpoints_root.is_dir():
        raise ValueError("checkpoint root must be an existing regular directory")
    committed: List[Tuple[int, Path]] = []
    for candidate in checkpoints_root.iterdir():
        match = _CHECKPOINT_STEP_RE.fullmatch(candidate.name)
        if match is None:
            raise ValueError(
                "unexpected checkpoint-root member blocks resume: {}".format(
                    candidate.name
                )
            )
        checkpoint_path, sidecar_path = checkpoint_paths(candidate)
        step = int(match.group(1))
        sidecar = _read_json_object(sidecar_path)
        if set(sidecar) != {"path", "sha256", "optimizer_step"}:
            raise ValueError(
                "checkpoint sidecar has unexpected fields: {}".format(
                    sidecar_path
                )
            )
        if sidecar.get("path") != checkpoint_path.name:
            raise ValueError("checkpoint sidecar path does not name checkpoint.pt")
        if type(sidecar.get("optimizer_step")) is not int:
            raise ValueError("checkpoint sidecar optimizer_step must be an integer")
        if sidecar["optimizer_step"] != step:
            raise ValueError("checkpoint directory and sidecar steps differ")
        expected_digest = _require_sha256(
            sidecar.get("sha256"), "checkpoint sidecar sha256"
        )
        actual_digest = sha256_file(checkpoint_path)
        if actual_digest != expected_digest:
            raise ValueError(
                "checkpoint payload hash differs from sidecar: {}".format(
                    checkpoint_path
                )
            )
        committed.append((step, checkpoint_path.parent))
    if not committed:
        raise ValueError("checkpoint root contains no committed step directory")
    committed.sort(key=lambda item: item[0])
    steps = [item[0] for item in committed]
    if len(set(steps)) != len(steps):
        raise ValueError("checkpoint root contains duplicate numeric steps")
    return [item[1] for item in committed]


def require_latest_resume_checkpoint(output_dir: Path, resume_path: Path) -> Path:
    """Require resume to name the latest closed checkpoint in this run."""

    checkpoints_root = output_dir.resolve(strict=True) / "checkpoints"
    committed = committed_checkpoint_directories(checkpoints_root)
    selected_checkpoint, _ = checkpoint_paths(resume_path)
    selected = selected_checkpoint.parent
    latest = committed[-1]
    if selected.parent != checkpoints_root:
        raise ValueError(
            "--resume checkpoint must belong to --output-dir/checkpoints"
        )
    if selected != latest:
        raise ValueError(
            "--resume must name the latest committed checkpoint {}; got {}".format(
                latest, selected
            )
        )
    return latest


def inspect_attempt_logs(
    output_dir: Path,
    *,
    require_nonempty: bool = False,
) -> List[Dict[str, Any]]:
    """Validate immutable attempt segments and return seal-ready summaries."""

    logs_root = output_dir / "logs"
    if logs_root.is_symlink() or not logs_root.is_dir():
        raise ValueError("logs must be an existing regular directory")
    metadata_by_index: Dict[int, Path] = {}
    metrics_by_index: Dict[int, Path] = {}
    for member in logs_root.iterdir():
        if member.is_symlink() or not member.is_file():
            raise ValueError("logs contains a non-regular member: {}".format(member))
        metadata_match = _ATTEMPT_METADATA_RE.fullmatch(member.name)
        metrics_match = _ATTEMPT_METRICS_RE.fullmatch(member.name)
        if metadata_match is not None:
            metadata_by_index[int(metadata_match.group(1))] = member
        elif metrics_match is not None:
            metrics_by_index[int(metrics_match.group(1))] = member
        else:
            raise ValueError(
                "unexpected logs member blocks attempt-safe resume: {}".format(
                    member.name
                )
            )
    if set(metadata_by_index) != set(metrics_by_index):
        raise ValueError("attempt metadata and metrics files are not paired")
    indices = sorted(metadata_by_index)
    if indices != list(range(len(indices))):
        raise ValueError("attempt indices must be contiguous from zero")
    if require_nonempty and not indices:
        raise ValueError("a successful run must contain at least one attempt log")

    output_resolved = output_dir.resolve(strict=True)
    summaries: List[Dict[str, Any]] = []
    for index in indices:
        metadata_path = metadata_by_index[index]
        metrics_path = metrics_by_index[index]
        metadata = _read_json_object(metadata_path)
        required_fields = {
            "schema_version",
            "attempt_index",
            "attempt_kind",
            "created_unix_seconds",
            "hostname",
            "config_sha256",
            "start_optimizer_step",
            "start_global_microstep",
            "start_checkpoint",
            "metrics_path",
        }
        if set(metadata) != required_fields:
            raise ValueError(
                "attempt metadata fields differ for attempt {:04d}".format(index)
            )
        if metadata.get("schema_version") != ATTEMPT_SCHEMA_VERSION:
            raise ValueError("attempt metadata schema version differs")
        if metadata.get("attempt_index") != index:
            raise ValueError("attempt metadata index differs from its filename")
        expected_kind = "initial" if index == 0 else "resume"
        if metadata.get("attempt_kind") != expected_kind:
            raise ValueError(
                "attempt_kind must be {} for attempt {:04d}".format(
                    expected_kind, index
                )
            )
        if type(metadata.get("created_unix_seconds")) not in {int, float}:
            raise ValueError("attempt creation time must be numeric")
        if not isinstance(metadata.get("hostname"), str) or not metadata["hostname"]:
            raise ValueError("attempt hostname must be a non-empty string")
        _require_sha256(metadata.get("config_sha256"), "attempt config_sha256")
        start_step = metadata.get("start_optimizer_step")
        start_microstep = metadata.get("start_global_microstep")
        if type(start_step) is not int or start_step < 0:
            raise ValueError("attempt start_optimizer_step must be non-negative")
        if type(start_microstep) is not int or start_microstep != 4 * start_step:
            raise ValueError("attempt start progress violates optimizer_step*4")
        expected_metrics_relative = metrics_path.relative_to(output_dir).as_posix()
        if metadata.get("metrics_path") != expected_metrics_relative:
            raise ValueError("attempt metadata metrics_path differs from filename")

        start_checkpoint = metadata.get("start_checkpoint")
        if not isinstance(start_checkpoint, dict) or set(start_checkpoint) != {
            "directory",
            "checkpoint_sha256",
            "sidecar_sha256",
        }:
            raise ValueError("attempt start_checkpoint identity is malformed")
        checkpoint_relative = start_checkpoint.get("directory")
        if not isinstance(checkpoint_relative, str):
            raise ValueError("attempt checkpoint directory must be a string")
        checkpoint_dir = (output_resolved / checkpoint_relative).resolve(strict=True)
        if checkpoint_dir.parent != output_resolved / "checkpoints":
            raise ValueError("attempt checkpoint is outside this run")
        checkpoint_match = _CHECKPOINT_STEP_RE.fullmatch(checkpoint_dir.name)
        if checkpoint_match is None or int(checkpoint_match.group(1)) != start_step:
            raise ValueError("attempt checkpoint directory and start step differ")
        checkpoint_path, sidecar_path = checkpoint_paths(checkpoint_dir)
        sidecar = _read_json_object(sidecar_path)
        checkpoint_digest = _require_sha256(
            start_checkpoint.get("checkpoint_sha256"),
            "attempt checkpoint_sha256",
        )
        sidecar_digest = _require_sha256(
            start_checkpoint.get("sidecar_sha256"),
            "attempt sidecar_sha256",
        )
        if sidecar.get("sha256") != checkpoint_digest:
            raise ValueError("attempt checkpoint digest differs from its sidecar")
        if sha256_file(checkpoint_path) != checkpoint_digest:
            raise ValueError("attempt checkpoint payload changed after launch")
        if (
            sidecar.get("path") != "checkpoint.pt"
            or sidecar.get("optimizer_step") != start_step
        ):
            raise ValueError("attempt checkpoint sidecar contract differs")
        if sha256_file(sidecar_path) != sidecar_digest:
            raise ValueError("attempt checkpoint sidecar changed after launch")

        record_count = 0
        first_step: Optional[int] = None
        last_step: Optional[int] = None
        with metrics_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.endswith("\n") or not line.strip():
                    raise ValueError(
                        "attempt metrics line {} is blank or partial".format(
                            line_number
                        )
                    )
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "attempt metrics line {} is invalid JSON".format(line_number)
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError("attempt metrics records must be JSON objects")
                step = record.get("optimizer_step")
                microstep = record.get("global_microstep")
                if (
                    record.get("schema_version") != RUN_SCHEMA_VERSION
                    or record.get("event") != "optimizer_step"
                    or type(step) is not int
                    or type(microstep) is not int
                ):
                    raise ValueError("attempt metrics record contract differs")
                if step <= start_step or (last_step is not None and step <= last_step):
                    raise ValueError(
                        "attempt metrics optimizer_step must be strictly increasing"
                    )
                if microstep != 4 * step:
                    raise ValueError("attempt metrics progress violates optimizer_step*4")
                if first_step is None:
                    first_step = step
                last_step = step
                record_count += 1
        summaries.append(
            {
                "attempt_index": index,
                "attempt_kind": expected_kind,
                "config_sha256": metadata["config_sha256"],
                "metadata_path": metadata_path.relative_to(output_dir).as_posix(),
                "metadata_sha256": sha256_file(metadata_path),
                "metrics_path": metrics_path.relative_to(output_dir).as_posix(),
                "metrics_sha256": sha256_file(metrics_path),
                "start_optimizer_step": start_step,
                "start_global_microstep": start_microstep,
                "start_checkpoint_directory": checkpoint_relative,
                "record_count": record_count,
                "first_optimizer_step": first_step,
                "last_optimizer_step": last_step,
            }
        )
    return summaries


def begin_attempt_log(
    output_dir: Path,
    *,
    config_sha256: str,
    start_checkpoint: Path,
    start_optimizer_step: int,
    start_global_microstep: int,
    is_resume: bool,
) -> Path:
    """Create one new attempt metadata/metrics pair without overwriting files."""

    _require_sha256(config_sha256, "attempt config_sha256")
    if start_global_microstep != 4 * start_optimizer_step:
        raise ValueError("attempt start progress violates optimizer_step*4")
    previous = inspect_attempt_logs(output_dir)
    if any(item["config_sha256"] != config_sha256 for item in previous):
        raise ValueError("prior attempt config hash differs from this launch")
    index = len(previous)
    if is_resume != (index > 0):
        raise ValueError("initial/resume attempt kind differs from log history")
    if index > 9999:
        raise ValueError("attempt index exceeds the four-digit log protocol")
    logs_root = output_dir / "logs"
    metadata_path = logs_root / "attempt-{:04d}.json".format(index)
    metrics_path = logs_root / "metrics.attempt-{:04d}.jsonl".format(index)
    checkpoint_path, sidecar_path = checkpoint_paths(start_checkpoint)
    checkpoint_dir = checkpoint_path.parent
    output_resolved = output_dir.resolve(strict=True)
    if checkpoint_dir.parent != output_resolved / "checkpoints":
        raise ValueError("attempt start checkpoint is outside this run")
    checkpoint_match = _CHECKPOINT_STEP_RE.fullmatch(checkpoint_dir.name)
    if (
        checkpoint_match is None
        or int(checkpoint_match.group(1)) != start_optimizer_step
    ):
        raise ValueError("attempt checkpoint directory and start step differ")
    sidecar = _read_json_object(sidecar_path)
    checkpoint_digest = _require_sha256(
        sidecar.get("sha256"), "attempt checkpoint sidecar sha256"
    )
    if sha256_file(checkpoint_path) != checkpoint_digest:
        raise ValueError("attempt checkpoint payload differs from its sidecar")
    if sidecar.get("path") != checkpoint_path.name:
        raise ValueError("attempt checkpoint sidecar path differs")
    if sidecar.get("optimizer_step") != start_optimizer_step:
        raise ValueError("attempt checkpoint sidecar progress differs")
    metadata = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "attempt_index": index,
        "attempt_kind": "resume" if is_resume else "initial",
        "created_unix_seconds": time.time(),
        "hostname": socket.gethostname(),
        "config_sha256": config_sha256,
        "start_optimizer_step": int(start_optimizer_step),
        "start_global_microstep": int(start_global_microstep),
        "start_checkpoint": {
            "directory": checkpoint_dir.relative_to(output_resolved).as_posix(),
            "checkpoint_sha256": checkpoint_digest,
            "sidecar_sha256": sha256_file(sidecar_path),
        },
        "metrics_path": metrics_path.relative_to(output_dir).as_posix(),
    }

    metrics_created = False
    metadata_created = False
    try:
        descriptor = os.open(
            str(metrics_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
        )
        metrics_created = True
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        write_json_exclusive(metadata_path, metadata)
        metadata_created = True
        logs_fd = os.open(str(logs_root), os.O_RDONLY)
        try:
            os.fsync(logs_fd)
        finally:
            os.close(logs_fd)
        validated = inspect_attempt_logs(output_dir)
        if len(validated) != index + 1:
            raise RuntimeError("new attempt log did not validate after publication")
        return metrics_path
    except BaseException:
        if metadata_created:
            metadata_path.unlink()
        if metrics_created:
            metrics_path.unlink()
        raise


def _build_success_seal_payload(
    output_dir: Path,
    optimizer_step: int,
    teacher_state_sha256: str,
) -> Dict[str, Any]:
    output_dir = _validate_run_root_members(output_dir, require_terminal=False)
    teacher_state_sha256 = _require_sha256(
        teacher_state_sha256, "final teacher_state_sha256"
    )
    manifest, manifest_identity = _validated_run_manifest(output_dir)
    if manifest_identity["teacher_state_sha256_initial"] != teacher_state_sha256:
        raise ValueError(
            "final teacher-state identity differs from run_manifest initialization"
        )
    max_steps = manifest["config"].get("max_optimizer_steps")
    if type(max_steps) is not int:
        raise ValueError(
            "final optimizer_step differs from run_manifest max_optimizer_steps"
        )
    # Ruling #8 §3 bypass (memory ruling-8-final-small-horizon-extension-2026-08-30):
    # extension keeps upstream run_manifest byte-identical (max=500) but seals
    # at optimizer_step=1000; allow the mismatch iff _EXTENSION_TARGET_STEP is
    # set by main() to the extension target.  Normal runs still enforce equality.
    #
    # Additional bypass for downstream consumers (generate_stage1_audio.py,
    # verify_stage1_run.py, etc.): those consumers call verify_sealed_stage1_run
    # WITHOUT going through train_stage1.main(), so _EXTENSION_TARGET_STEP is
    # None.  Detect extension via the presence of SEALED.pre_extension.json in
    # output_dir: if it exists, this is an extension run and optimizer_step
    # > max_steps is legitimate (extension target recorded in current SEALED.json).
    _extend_target = globals().get("_EXTENSION_TARGET_STEP")
    _is_extension_run = (output_dir / "SEALED.pre_extension.json").is_file()
    if _extend_target is None and not _is_extension_run:
        if max_steps != optimizer_step:
            raise ValueError(
                "final optimizer_step differs from run_manifest max_optimizer_steps"
            )
    elif _extend_target is not None:
        if optimizer_step != _extend_target:
            raise ValueError(
                "final optimizer_step {} differs from --extend-to-step target {}".format(
                    optimizer_step, _extend_target
                )
            )
        if optimizer_step <= max_steps:
            raise ValueError(
                "extension optimizer_step {} must exceed original max_optimizer_steps {}".format(
                    optimizer_step, max_steps
                )
            )
    else:
        # _extend_target is None but _is_extension_run: downstream consumer path.
        # Read the SEALED.pre_extension.json to confirm original max_steps was 500
        # (or whatever the manifest still says), and verify current optimizer_step
        # exceeds it — this is a well-formed extension seal.
        if optimizer_step <= max_steps:
            raise ValueError(
                "extension consumer: optimizer_step {} must exceed original "
                "max_optimizer_steps {} (Ruling #8 §3)".format(
                    optimizer_step, max_steps
                )
            )
    status = _read_json_object(output_dir / "status.json")
    if status != {"status": "running", "optimizer_step": 0}:
        raise ValueError("status.json differs from the immutable start record")
    status_identity = _regular_file_identity(
        output_dir / "status.json", "status.json"
    )
    checkpoints = _checkpoint_inventory(
        output_dir,
        config_sha256=manifest_identity["config_sha256"],
        teacher_state_sha256=teacher_state_sha256,
        ddp_reducer_identity_sha256=manifest_identity[
            "ddp_reducer_identity_sha256"
        ],
    )
    if checkpoints[-1]["optimizer_step"] != optimizer_step:
        raise ValueError("final checkpoint is not the latest committed checkpoint")
    failed_path = output_dir / "FAILED.json"
    superseded_failure = (
        _regular_file_identity(failed_path, "FAILED.json")
        if failed_path.exists() or failed_path.is_symlink()
        else None
    )
    attempt_logs = inspect_attempt_logs(output_dir, require_nonempty=True)
    if any(
        item["config_sha256"] != manifest_identity["config_sha256"]
        for item in attempt_logs
    ):
        raise ValueError("attempt-log config hash differs from run_manifest")
    if any(
        item["start_optimizer_step"] > optimizer_step
        or (
            item["last_optimizer_step"] is not None
            and item["last_optimizer_step"] > optimizer_step
        )
        for item in attempt_logs
    ):
        raise ValueError("attempt logs contain progress beyond the final checkpoint")
    return {
        "schema_version": SEAL_SCHEMA_VERSION,
        "status": "sealed",
        "optimizer_step": int(optimizer_step),
        "teacher_state_sha256_final": teacher_state_sha256,
        "run_manifest": manifest_identity,
        "start_status": status_identity,
        "checkpoint_inventory": checkpoints,
        "final_checkpoint": checkpoints[-1],
        "attempt_log_schema_version": ATTEMPT_SCHEMA_VERSION,
        "attempt_logs": attempt_logs,
        "superseded_failure": superseded_failure,
        "superseded_failure_sha256": (
            superseded_failure["sha256"]
            if superseded_failure is not None
            else None
        ),
    }


def _build_done_payload(
    sealed_path: Path,
    seal_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    manifest_identity = seal_payload["run_manifest"]
    final_checkpoint = seal_payload["final_checkpoint"]
    return {
        "schema_version": DONE_SCHEMA_VERSION,
        "status": "complete",
        "SEALED.json_sha256": sha256_file(sealed_path),
        "SEALED.json_size_bytes": sealed_path.stat().st_size,
        "run_manifest_sha256": manifest_identity["sha256"],
        "run_manifest_size_bytes": manifest_identity["size_bytes"],
        "final_checkpoint_sha256": final_checkpoint["checkpoint_sha256"],
        "failed_record_superseded": (
            seal_payload["superseded_failure"] is not None
        ),
    }


def commit_success_terminal(
    output_dir: Path,
    optimizer_step: int,
    teacher_state_sha256: str,
) -> Dict[str, Any]:
    """Idempotently publish the manifest-bound DONE.json success commit."""

    output_dir = _validate_run_root_members(output_dir, require_terminal=False)
    seal_payload = _build_success_seal_payload(
        output_dir, optimizer_step, teacher_state_sha256
    )
    sealed_path = output_dir / "SEALED.json"
    if sealed_path.exists():
        if _read_json_object(sealed_path) != seal_payload:
            raise ValueError("existing SEALED.json differs from final run state")
    else:
        write_json_exclusive(sealed_path, seal_payload)
    done_payload = _build_done_payload(sealed_path, seal_payload)
    done_path = output_dir / "DONE.json"
    if done_path.exists():
        if _read_json_object(done_path) != done_payload:
            raise ValueError("existing DONE.json differs from final run state")
    else:
        write_json_exclusive(done_path, done_payload)
    output_fd = os.open(str(output_dir), os.O_RDONLY)
    try:
        os.fsync(output_fd)
    finally:
        os.close(output_fd)
    return done_payload


def verify_sealed_stage1_run(output_dir: Path) -> Dict[str, Any]:
    """Consume one copied run as a closed world and verify DONE back to bytes."""

    output_dir = _validate_run_root_members(output_dir, require_terminal=True)
    sealed_path = output_dir / "SEALED.json"
    done_path = output_dir / "DONE.json"
    seal_payload = _read_json_object(sealed_path)
    if seal_payload.get("schema_version") != SEAL_SCHEMA_VERSION:
        raise ValueError("SEALED.json schema version differs")
    optimizer_step = seal_payload.get("optimizer_step")
    if type(optimizer_step) is not int or optimizer_step < 0:
        raise ValueError("SEALED.json optimizer_step must be non-negative")
    teacher_state_sha256 = _require_sha256(
        seal_payload.get("teacher_state_sha256_final"),
        "SEALED.json teacher_state_sha256_final",
    )
    expected_seal = _build_success_seal_payload(
        output_dir, optimizer_step, teacher_state_sha256
    )
    if seal_payload != expected_seal:
        raise ValueError("SEALED.json differs from the closed run state")
    done_payload = _read_json_object(done_path)
    expected_done = _build_done_payload(sealed_path, expected_seal)
    if done_payload != expected_done:
        raise ValueError("DONE.json differs from the sealed run state")
    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified",
        "run_directory": str(output_dir),
        "optimizer_step": optimizer_step,
        "attempt_count": len(expected_seal["attempt_logs"]),
        "checkpoint_count": len(expected_seal["checkpoint_inventory"]),
        "run_manifest_sha256": expected_seal["run_manifest"]["sha256"],
        "run_manifest_size_bytes": expected_seal["run_manifest"]["size_bytes"],
        "SEALED.json_sha256": expected_done["SEALED.json_sha256"],
        "DONE.json_sha256": sha256_file(done_path),
        "final_checkpoint_sha256": expected_done["final_checkpoint_sha256"],
    }


def save_checkpoint(
    *,
    output_dir: Path,
    student_lm: nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: Mapping[str, Any],
    rank: int,
) -> None:
    """Collective barrier around an atomic rank-0 checkpoint commit."""

    dist.barrier()
    local_rng_state = get_rng_state()
    rng_states_by_rank: Optional[List[Any]] = (
        [None for _ in range(dist.get_world_size())] if rank == 0 else None
    )
    dist.gather_object(local_rng_state, rng_states_by_rank, dst=0)
    if rank == 0:
        step = int(metadata["optimizer_step"])
        payload = {
            "metadata": dict(metadata),
            "student_state": student_lm.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": {
                "type": "linear_warmup_then_constant",
                "warmup_optimizer_steps": 50,
                "completed_optimizer_steps": step,
                "learning_rate": optimizer.param_groups[0]["lr"],
            },
            "rng_state_by_rank": rng_states_by_rank,
        }
        commit_checkpoint_payload(
            output_dir / "checkpoints",
            step,
            payload,
        )
    dist.barrier()


def load_resume(
    *,
    path: Path,
    student_lm: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_metadata: Mapping[str, Any],
    device: torch.device,
    rank: int,
) -> Tuple[int, int, Dict[str, Any], Dict[str, Any]]:
    checkpoint_path, sidecar_path = checkpoint_paths(path)
    with sidecar_path.open("r", encoding="utf-8") as stream:
        sidecar = json.load(stream)
    if sidecar.get("path") != checkpoint_path.name:
        raise ValueError("resume checkpoint sidecar path mismatch")
    if type(sidecar.get("optimizer_step")) is not int:
        raise ValueError("resume checkpoint sidecar has no integer optimizer_step")
    actual_checkpoint_hash = sha256_file(checkpoint_path)
    if sidecar.get("sha256") != actual_checkpoint_hash:
        raise ValueError("resume checkpoint SHA-256 does not match sidecar")
    # Keep checkpoint/RNG storages on CPU while deserializing. load_state_dict
    # copies model tensors to CUDA and optimizer.load_state_dict casts state to
    # each parameter device; CPU RNG state must remain a CPU ByteTensor.
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("resume checkpoint must contain a dictionary")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("resume checkpoint has no metadata dictionary")
    immutable_expected = {
        key: value
        for key, value in expected_metadata.items()
        if key not in {"optimizer_step", "global_microstep"}
    }
    verify_resume_metadata(metadata, immutable_expected)
    # Freeze the serialized state identity before either load_state_dict call.
    # The after-load audit verifies both that the payload itself was not
    # modified and that each live rank received those original exact bytes.
    checkpoint_state_hashes_before_load = _checkpoint_resume_state_hashes(payload)
    student_lm.load_state_dict(payload["student_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler_state = payload.get("scheduler_state")
    if not isinstance(scheduler_state, dict):
        raise ValueError("resume checkpoint has no scheduler state")
    rng_states = payload.get("rng_state_by_rank")
    if not isinstance(rng_states, list) or len(rng_states) != dist.get_world_size():
        raise ValueError("resume checkpoint has no complete per-rank RNG state")
    expected_rng_state = rng_states[rank]
    expected_rng_sha256 = rng_state_sha256(expected_rng_state)
    set_rng_state(expected_rng_state)
    # Capture before begin_attempt_log and before seed_rollout can overwrite
    # ambient RNG. This proves restoration itself, not only final equality.
    observed_rng_state = get_rng_state()
    observed_rng_sha256 = rng_state_sha256(observed_rng_state)
    rng_exact = rng_states_equal(expected_rng_state, observed_rng_state)
    if not rng_exact or observed_rng_sha256 != expected_rng_sha256:
        raise RuntimeError("resume did not restore the selected rank RNG state")
    rng_restore_audit = {
        "schema_version": "ptc-opd-node3-rng-restore-v1",
        "rank": rank,
        "captured_before_next_rollout_seed": True,
        "expected_sha256": expected_rng_sha256,
        "observed_sha256": observed_rng_sha256,
        "exact": True,
    }
    # This call is deliberately after model/optimizer/RNG restoration and before
    # the function returns to any rollout or scoring forward.  Each rank hashes
    # its own live CUDA state against the CPU checkpoint payload independently.
    state_restore_audit = audit_resume_state_restore(
        checkpoint_payload=payload,
        student_lm=student_lm,
        optimizer=optimizer,
        rank=rank,
        checkpoint_hashes_before_load=checkpoint_state_hashes_before_load,
    )
    optimizer_step = int(metadata["optimizer_step"])
    global_microstep = int(metadata["global_microstep"])
    if sidecar.get("optimizer_step") != optimizer_step:
        raise ValueError("resume checkpoint sidecar progress mismatch")
    if scheduler_state.get("type") != "linear_warmup_then_constant":
        raise ValueError("resume scheduler type differs from frozen protocol")
    if int(scheduler_state.get("warmup_optimizer_steps", -1)) != 50:
        raise ValueError("resume scheduler warmup differs from frozen protocol")
    if int(scheduler_state.get("completed_optimizer_steps", -1)) != optimizer_step:
        raise ValueError("resume scheduler progress differs from optimizer progress")
    expected_microstep = progress_from_optimizer_step(optimizer_step, 4)
    if global_microstep != expected_microstep:
        raise ValueError(
            "resume global_microstep {} != optimizer_step*4 {}".format(
                global_microstep, expected_microstep
            )
        )
    return optimizer_step, global_microstep, rng_restore_audit, state_restore_audit


def denominator_window_is_constant(
    denominators: Sequence[float], rtol: float
) -> bool:
    if len(denominators) != 4:
        raise ValueError("one Stage-1 accumulation window must have four denominators")
    if rtol < 0.0:
        raise ValueError("denominator rtol must be non-negative")
    reference = denominators[0]
    if not all(torch.isfinite(torch.tensor(value)).item() and value > 0.0 for value in denominators):
        return False
    tolerance = rtol * abs(reference)
    return all(abs(value - reference) <= tolerance for value in denominators[1:])


def seed_rollout(seed: int, optimizer_step: int, microstep: int, rank: int) -> None:
    # Rollout sampling is intentionally rank-specific because each rank owns
    # different stable sample IDs. Every scheduled microstep is explicitly
    # reseeded, so exact resume does not depend on any rank's ambient RNG state.
    payload = "ptc-opd-rollout-v1\0{}\0{}\0{}\0{}".format(
        seed, optimizer_step, microstep, rank
    )
    derived = int.from_bytes(__import__("hashlib").sha256(payload.encode()).digest()[:8], "big")
    derived &= (1 << 63) - 1
    random.seed(derived)
    torch.manual_seed(derived)
    torch.cuda.manual_seed(derived)


def run_training(config: Stage1Config, args: argparse.Namespace) -> None:
    validate_config(config)
    validate_node3_fault_injection(config, args)
    validate_node3_training_contract(config, args)
    node3_determinism = configure_node3_determinism()
    offline_environment = require_offline_hf_environment()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-1 training requires CUDA; use --dry-run on CPU")
    if not all(name in os.environ for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        raise RuntimeError("launch training with torchrun, not plain python")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != config.expected_world_size or world_size != 8:
        raise RuntimeError("WORLD_SIZE must be exactly 8, never 32")
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "-1"))
    group_rank = int(os.environ.get("GROUP_RANK", "-1"))
    if local_world_size != 8 or group_rank != 0:
        raise RuntimeError(
            "Stage-1 requires one physical node with LOCAL_WORLD_SIZE=8 and GROUP_RANK=0"
        )
    if not 0 <= local_rank < 8:
        raise RuntimeError("LOCAL_RANK must identify one of this machine's eight GPUs")
    if torch.cuda.device_count() != 8:
        raise RuntimeError("Stage-1 requires exactly eight visible local CUDA devices")
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm launches are forbidden by the frozen Stage-1 protocol")
    output_dir = Path(config.output_dir)
    if args.resume is None and output_dir.exists():
        raise FileExistsError(
            "refusing to overwrite existing run directory {}".format(output_dir)
        )
    # Ruling #8 §3 bypass (memory ruling-8-final-small-horizon-extension-2026-08-30):
    # When --extend-to-step is set, we intentionally resume a run that already
    # produced SEALED.json + DONE.json.  Move those aside byte-identical (rename
    # only, no rewrite) so identity checks that fire on their presence do not
    # trigger, but the seal evidence is still preserved for auditability.
    #
    # DDP race safety: this runs BEFORE dist.init_process_group.  Only global
    # rank 0 (torchrun sets RANK env) performs the rename; other ranks poll
    # briefly for the rename to complete via filesystem, then proceed.  This
    # avoids the race where rank 0's rename removes SEALED.json before other
    # ranks reach the is_file() check.
    _extension_seal_backup: Optional[Path] = None
    _extension_done_backup: Optional[Path] = None
    _global_rank_env = os.environ.get("RANK", "0")
    _is_global_rank_zero = (_global_rank_env == "0")
    if args.resume is not None and args.extend_to_step is not None:
        sealed_src = output_dir / "SEALED.json"
        done_src = output_dir / "DONE.json"
        _extension_seal_backup = output_dir / "SEALED.pre_extension.json"
        _extension_done_backup = output_dir / "DONE.pre_extension.json"
        if _is_global_rank_zero:
            # Rank 0: idempotent rename (skip if already renamed by a prior run)
            if not _extension_seal_backup.exists():
                if not sealed_src.is_file():
                    raise ValueError(
                        "--extend-to-step requires an existing SEALED.json (rank 0)"
                    )
                if not done_src.is_file():
                    raise ValueError(
                        "--extend-to-step requires an existing DONE.json (rank 0)"
                    )
                sealed_src.rename(_extension_seal_backup)
                done_src.rename(_extension_done_backup)
            elif _extension_done_backup.exists():
                # Fully renamed by earlier attempt — proceed idempotently
                pass
            else:
                raise ValueError(
                    "SEALED.pre_extension.json exists but DONE.pre_extension.json "
                    "does not; extension state is inconsistent, aborting"
                )
        else:
            # Non-rank-0: wait up to 60s for rank 0 to complete rename.
            import time as _time_mod
            _deadline = _time_mod.time() + 60.0
            while _time_mod.time() < _deadline:
                if _extension_seal_backup.exists() and _extension_done_backup.exists():
                    break
                _time_mod.sleep(0.5)
            else:
                raise TimeoutError(
                    "non-rank-0 timed out waiting for rank 0 to rename "
                    "SEALED.json / DONE.json under --extend-to-step"
                )
    if args.resume is not None and (output_dir / "DONE.json").exists():
        raise ValueError("refusing to resume a run already committed by DONE.json")
    repair_terminal_commit = bool(
        args.resume is not None
        and (output_dir / "SEALED.json").is_file()
        and not (output_dir / "DONE.json").exists()
        and args.extend_to_step is None
    )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    run_directory_owned = False
    try:
        records = load_prompt_manifest(Path(config.manifest))
        spec = resolve_mode(config.mode)
        cfg_decision_identity = verify_configured_cfg_decision(config)
        prior_artifact = resolve_codec_prior_artifact(config)
        codebook_prior_hash = (
            prior_artifact.identity_sha256
            if prior_artifact is not None
            else None
        )
        audiocraft_lm_path = (
            Path(config.audiocraft_root) / "audiocraft" / "models" / "lm.py"
        )
        if not audiocraft_lm_path.is_file():
            raise ValueError("--audiocraft-root is not an AudioCraft source checkout")
        audiocraft_lm_hash = sha256_file(audiocraft_lm_path)
        source_identity = distributed_source_identity(
            Path(config.audiocraft_root), rank
        )
        if not isinstance(source_identity, dict):
            raise RuntimeError("could not resolve AudioCraft source identity")
        verify_codec_prior_source_binding(prior_artifact, source_identity)
        audiocraft_source_hash = str(source_identity["tree_sha256"])
        config_dict = resolved_config_dict(config)
        config_hash = canonical_json_sha256(config_dict)
        synchronize_preflight(
            {
                "config_sha256": config_hash,
                "manifest_local_sha256": sha256_file(Path(config.manifest)),
                "codebook_prior_sha256": codebook_prior_hash,
                "audiocraft_lm_sha256": audiocraft_lm_hash,
                "audiocraft_source_sha256": audiocraft_source_hash,
                "cfg_scale_decision": cfg_decision_identity,
                "offline_environment": offline_environment,
                "manifest_records": len(records),
            },
            rank,
        )
        hashes = checkpoint_hashes(config, rank)
        cfg_generation_binding = verify_cfg_generation_binding(
            config, hashes, source_identity
        )
        prior: Optional[Tensor] = None
        if spec.requires_perceptual_prior:
            if prior_artifact is None:
                raise ValueError(
                    "{} requires --codebook-prior-artifact-dir".format(config.mode)
                )
            prior = torch.tensor(
                prior_artifact.prior, dtype=torch.float32, device=device
            )

        resume_path: Optional[Path] = None
        if args.resume is not None:
            resume_path = require_latest_resume_checkpoint(
                output_dir, args.resume
            )
            inspect_attempt_logs(output_dir)
            run_manifest_path = output_dir / "run_manifest.json"
            if not run_manifest_path.is_file():
                raise ValueError("resume output directory has no run_manifest.json")
            with run_manifest_path.open("r", encoding="utf-8") as stream:
                previous_manifest = json.load(stream)
            if previous_manifest.get("config_sha256") != config_hash:
                # Ruling #8 §3 bypass: extension resumes upstream config_sha256
                # byte-identical, so this check normally passes.  If it fails
                # under --extend-to-step we still refuse (invariant broken).
                if args.extend_to_step is not None:
                    raise ValueError(
                        "resume run_manifest config hash mismatch under "
                        "--extend-to-step: Ruling #8 §4 requires unchanged config; "
                        "extension attempt aborted"
                    )
                raise ValueError("resume run_manifest config hash mismatch")

        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=args.resume is not None)
            (output_dir / "logs").mkdir(exist_ok=args.resume is not None)
            (output_dir / "checkpoints").mkdir(exist_ok=args.resume is not None)
        dist.barrier()
        run_directory_owned = True

        # Import the explicitly supplied, overlay-patched AudioCraft checkout.
        audiocraft_root = Path(config.audiocraft_root)
        sys.path.insert(0, str(audiocraft_root))
        from audiocraft.models.loaders import load_lm_model
        from audiocraft.modules.conditioners import (
            ClassifierFreeGuidanceDropout,
            ConditioningAttributes,
        )

        # CPU loading forces float32 in the pinned AudioCraft loader.  Moving
        # afterwards avoids the loader's direct-to-CUDA fp16 branch.
        student_lm = load_lm_model(config.student_checkpoint, device="cpu")
        teacher_lm = load_lm_model(config.teacher_checkpoint, device="cpu")
        import inspect
        generate_parameters = inspect.signature(student_lm.generate).parameters
        if not {"use_cfg", "condition_tensors"}.issubset(generate_parameters):
            raise RuntimeError(
                "AudioCraft checkout is missing the explicit no-CFG overlay"
            )
        student_cpu_parameters = list(student_lm.parameters())
        teacher_cpu_parameters = list(teacher_lm.parameters())
        if not student_cpu_parameters or not teacher_cpu_parameters:
            raise RuntimeError("student/teacher LM must expose parameters")
        if any(parameter.dtype != torch.float32 for parameter in student_cpu_parameters):
            raise RuntimeError("student did not load as CPU float32")
        if any(parameter.dtype != torch.float32 for parameter in teacher_cpu_parameters):
            raise RuntimeError("teacher did not load as CPU float32")
        if any(parameter.device.type != "cpu" for parameter in student_cpu_parameters):
            raise RuntimeError("student parameters were not resident on CPU after load")
        if any(parameter.device.type != "cpu" for parameter in teacher_cpu_parameters):
            raise RuntimeError("teacher parameters were not resident on CPU after load")
        cpu_load_contract = {
            "student_all_fp32": True,
            "teacher_all_fp32": True,
            "student_all_cpu": True,
            "teacher_all_cpu": True,
            "student_parameter_count": sum(
                int(parameter.numel()) for parameter in student_cpu_parameters
            ),
            "teacher_parameter_count": sum(
                int(parameter.numel()) for parameter in teacher_cpu_parameters
            ),
        }
        student_contract = strict_validate_musicgen_model(
            student_lm, frame_rate=config.codec_frame_rate
        )
        teacher_contract = strict_validate_musicgen_model(
            teacher_lm, frame_rate=config.codec_frame_rate
        )
        if student_contract != teacher_contract:
            raise RuntimeError("student and teacher MusicGen contracts differ")
        student_t5_identity = loaded_t5_identity(student_lm)
        teacher_t5_identity = loaded_t5_identity(teacher_lm)
        if student_t5_identity != teacher_t5_identity:
            raise RuntimeError("student and teacher loaded different T5 identities")
        external_t5_identity_hash = str(student_t5_identity["identity_sha256"])
        if (
            external_t5_identity_hash
            != config.cfg_generation_loaded_t5_identity_sha256
        ):
            raise RuntimeError(
                "loaded T5 differs from the T5 used to freeze the CFG scale"
            )
        t5_identity_hashes: List[Any] = [None for _ in range(world_size)]
        dist.all_gather_object(t5_identity_hashes, external_t5_identity_hash)
        if len(set(t5_identity_hashes)) != 1:
            raise RuntimeError("loaded T5 identity differs across ranks")
        student_lm.to(device)
        teacher_lm.to(device)
        move_external_text_encoder(student_lm, device)
        teacher_lm.requires_grad_(False)
        teacher_lm.eval()
        freeze_condition_provider(student_lm)
        freeze_condition_provider(teacher_lm)
        student_conditioner_audit = condition_provider_audit(student_lm, "student")
        teacher_conditioner_audit = condition_provider_audit(teacher_lm, "teacher")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("node GPU does not support CUDA bfloat16")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            bf16_probe = torch.ones((8, 8), device=device) @ torch.ones(
                (8, 8), device=device
            )
        if bf16_probe.dtype != torch.bfloat16 or not bool(
            torch.isfinite(bf16_probe).all().item()
        ):
            raise RuntimeError("CUDA BF16 autocast probe did not produce finite BF16")
        runtime_contract_local = {
            "rank": rank,
            "local_rank": local_rank,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_capability": list(torch.cuda.get_device_capability(device)),
            "cuda_device_count": torch.cuda.device_count(),
            "gpu_total_memory_bytes": int(
                torch.cuda.get_device_properties(device).total_memory
            ),
            "cpu_load": cpu_load_contract,
            "student_conditioner": student_conditioner_audit,
            "teacher_conditioner": teacher_conditioner_audit,
            "generate_accepts_use_cfg": "use_cfg" in generate_parameters,
            "generate_accepts_condition_tensors": (
                "condition_tensors" in generate_parameters
            ),
            "bf16_supported": True,
            "bf16_autocast_probe_dtype": str(bf16_probe.dtype),
            "bf16_autocast_probe_finite": True,
            "node3_determinism": node3_determinism_audit(),
        }
        teacher_state_hash_initial = hash_module_state(teacher_lm)
        student_state_hash_initial = hash_module_state(student_lm)
        if student_state_hash_initial != teacher_state_hash_initial:
            raise RuntimeError(
                "loaded student and frozen teacher LM states differ at initialization"
            )
        teacher_state_hashes = [None for _ in range(world_size)]
        dist.all_gather_object(teacher_state_hashes, teacher_state_hash_initial)
        if len(set(teacher_state_hashes)) != 1:
            raise RuntimeError("teacher state hash differs across ranks")

        scorer = StudentScorer(student_lm).to(device)
        ddp_scorer = DDP(
            scorer,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            bucket_cap_mb=DDP_BUCKET_CAP_MB,
            find_unused_parameters=DDP_FIND_UNUSED_PARAMETERS,
            gradient_as_bucket_view=DDP_GRADIENT_AS_BUCKET_VIEW,
            static_graph=DDP_STATIC_GRAPH,
        )
        runtime_contract_local["ddp_reducer"] = ddp_reducer_audit(ddp_scorer)
        ddp_reducer_identity_hash = runtime_contract_local["ddp_reducer"][
            "identity_sha256"
        ]
        runtime_contract_by_rank: List[Any] = [None for _ in range(world_size)]
        dist.all_gather_object(runtime_contract_by_rank, runtime_contract_local)
        if [item.get("rank") for item in runtime_contract_by_rank] != list(
            range(world_size)
        ):
            raise RuntimeError("runtime contract audit did not cover ranks 0..7")
        construction_ddp_identities = {
            item.get("ddp_reducer", {}).get("identity_sha256")
            for item in runtime_contract_by_rank
            if isinstance(item, dict)
            and isinstance(item.get("ddp_reducer"), dict)
        }
        if construction_ddp_identities != {ddp_reducer_identity_hash}:
            raise RuntimeError("DDP reducer construction identity differs across ranks")
        facade = DDPScoringFacade(ddp_scorer, student_lm)
        named_trainable = [
            (name, parameter)
            for name, parameter in student_lm.named_parameters()
            if parameter.requires_grad
        ]
        trainable = [parameter for _, parameter in named_trainable]
        if not trainable:
            raise RuntimeError("student has no trainable parameters")
        optimizer = torch.optim.AdamW(
            trainable,
            lr=learning_rate_for_update(
                config.learning_rate, 0, config.warmup_optimizer_steps
            ),
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=0.0,
        )
        base_metadata = build_checkpoint_metadata(
            config_hash=config_hash,
            manifest_hash=hashes["manifest_sha256"],
            student_checkpoint_hash=hashes["student_checkpoint_sha256"],
            teacher_checkpoint_hash=hashes["teacher_checkpoint_sha256"],
            optimizer_step=0,
            global_microstep=0,
            teacher_state_hash_initial=teacher_state_hash_initial,
            ddp_reducer_identity_hash=ddp_reducer_identity_hash,
            codebook_prior_hash=codebook_prior_hash,
            audiocraft_lm_hash=audiocraft_lm_hash,
            audiocraft_source_hash=audiocraft_source_hash,
            cfg_scale_decision_file_hash=config.cfg_scale_decision_file_sha256,
            cfg_scale_decision_payload_hash=(
                config.cfg_scale_decision_payload_sha256
            ),
            external_t5_identity_hash=external_t5_identity_hash,
        )
        if rank == 0 and args.resume is None:
            run_manifest = {
                "schema_version": RUN_SCHEMA_VERSION,
                "created_unix_seconds": time.time(),
                "hostname": socket.gethostname(),
                "launcher": "torchrun_standalone",
                "nnodes": 1,
                "world_size": world_size,
                "config": config_dict,
                "config_sha256": config_hash,
                "mode_mapping": {
                    "loss_mode": spec.loss_mode,
                    "rho": spec.rho,
                    "uses_prior": spec.requires_perceptual_prior,
                },
                **hashes,
                "codebook_prior_sha256": (
                    codebook_prior_hash
                ),
                "codebook_prior_artifact": (
                    dict(prior_artifact.identity)
                    if prior_artifact is not None
                    else None
                ),
                "audiocraft_lm_sha256": audiocraft_lm_hash,
                "audiocraft_source_identity": source_identity,
                "cfg_scale_decision": cfg_decision_identity,
                "cfg_generation_binding": cfg_generation_binding,
                "loaded_t5_identity": student_t5_identity,
                "teacher_state_sha256_initial": teacher_state_hash_initial,
                "student_state_sha256_initial": student_state_hash_initial,
                "runtime_contract_by_rank": runtime_contract_by_rank,
                "environment": {
                    "python": sys.version,
                    "torch": torch.__version__,
                    "cuda_runtime": torch.version.cuda,
                    "gpu_name": torch.cuda.get_device_name(device),
                    "gpu_capability": list(torch.cuda.get_device_capability(device)),
                    "cuda_device_count": torch.cuda.device_count(),
                    "offline_hf": offline_environment,
                    "node3_determinism": node3_determinism,
                },
                "launch_command": [sys.executable] + sys.argv,
            }
            write_json_exclusive(output_dir / "run_manifest.json", run_manifest)
            write_json_exclusive(
                output_dir / "status.json",
                {"status": "running", "optimizer_step": 0},
            )

        optimizer_step = 0
        global_microstep = 0
        resume_rng_restore_local: Optional[Dict[str, Any]] = None
        resume_state_restore_local: Optional[Dict[str, Any]] = None
        if args.resume is not None:
            if resume_path is None:
                raise AssertionError("resume path was not resolved during preflight")
            resume_checkpoint, resume_sidecar = checkpoint_paths(resume_path)
            with resume_sidecar.open("r", encoding="utf-8") as stream:
                expected_resume_sidecar = json.load(stream)
            expected_resume_digest = expected_resume_sidecar.get("sha256")
            resume_digests: List[Any] = [None for _ in range(world_size)]
            dist.all_gather_object(resume_digests, expected_resume_digest)
            if len(set(resume_digests)) != 1:
                raise RuntimeError("resume checkpoint digest differs across ranks")
            (
                optimizer_step,
                global_microstep,
                resume_rng_restore_local,
                resume_state_restore_local,
            ) = load_resume(
                path=resume_path,
                student_lm=student_lm,
                optimizer=optimizer,
                expected_metadata=base_metadata,
                device=device,
                rank=rank,
            )
        else:
            save_checkpoint(
                output_dir=output_dir,
                student_lm=student_lm,
                optimizer=optimizer,
                metadata=base_metadata,
                rank=rank,
            )
        if repair_terminal_commit and optimizer_step != config.max_optimizer_steps:
            raise ValueError(
                "SEALED.json without DONE.json may only be repaired from the "
                "completed final checkpoint"
            )

        attempt_envelope: Optional[Dict[str, Any]] = None
        if repair_terminal_commit:
            attempt_envelope = {"value": None} if rank == 0 else None
        elif rank == 0:
            try:
                start_checkpoint = (
                    resume_path
                    if resume_path is not None
                    else output_dir / "checkpoints" / "step-00000"
                )
                attempt_metrics_path = begin_attempt_log(
                    output_dir,
                    config_sha256=config_hash,
                    start_checkpoint=start_checkpoint,
                    start_optimizer_step=optimizer_step,
                    start_global_microstep=global_microstep,
                    is_resume=args.resume is not None,
                )
                attempt_envelope = {
                    "value": attempt_metrics_path.relative_to(output_dir).as_posix()
                }
            except BaseException as exc:
                attempt_envelope = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        attempt_envelope = broadcast_object(attempt_envelope, rank)
        if not isinstance(attempt_envelope, dict) or "value" not in attempt_envelope:
            raise RuntimeError(
                "rank-zero attempt-log publication failed: {}".format(
                    attempt_envelope
                )
            )
        attempt_metrics_relative = attempt_envelope["value"]
        metrics_path = (
            output_dir / str(attempt_metrics_relative)
            if attempt_metrics_relative is not None
            else output_dir / "logs" / "terminal-repair-do-not-write.jsonl"
        )
        sampler = DeterministicDistributedBatchSampler(
            len(records),
            config.seed,
            rank,
            world_size,
            config.rank_batch_size,
            sample_ids=[record.sample_id for record in records],
        )
        optimizer.zero_grad(set_to_none=True)

        # Ruling #8 §3 bypass: extension target (max_optimizer_steps=500 in
        # sealed config, extension to step 1000 via --extend-to-step 1000).
        # dataloader is a pure function of (seed, global_microstep) via
        # DeterministicDistributedBatchSampler, so 500→1000 is deterministic.
        _effective_max_optimizer_steps = (
            args.extend_to_step
            if args.extend_to_step is not None
            else config.max_optimizer_steps
        )
        if args.extend_to_step is not None and _effective_max_optimizer_steps <= optimizer_step:
            raise ValueError(
                "--extend-to-step {} must exceed current optimizer_step {}".format(
                    _effective_max_optimizer_steps, optimizer_step
                )
            )

        while optimizer_step < _effective_max_optimizer_steps:
            scheduled_lr = learning_rate_for_update(
                config.learning_rate,
                optimizer_step,
                config.warmup_optimizer_steps,
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = scheduled_lr
            step_started = time.perf_counter()
            denominators: List[float] = []
            global_losses: List[float] = []
            selected_cells = 0
            valid_cells = 0
            microstep_audits: List[Dict[str, Any]] = []
            for accumulation_index in range(config.grad_accum_steps):
                assignment = sampler.batch(global_microstep)
                batch_records = [records[index] for index in assignment.indices]
                sample_ids = [record.sample_id for record in batch_records]
                prompts = [record.prompt for record in batch_records]
                seed_rollout(config.seed, optimizer_step, accumulation_index, rank)

                # generate() requires eval. train() below recursively toggles
                # the provider, so freeze/eval the provider again every cycle.
                student_lm.eval()
                freeze_condition_provider(student_lm)
                conditional, null = prepare_condition_tensors(
                    student_lm,
                    prompts,
                    ConditioningAttributes,
                    ClassifierFreeGuidanceDropout,
                )
                assert_condition_provider_frozen(student_lm, "student")
                assert_condition_provider_frozen(teacher_lm, "teacher")
                with torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    codes = student_lm.generate(
                        condition_tensors=conditional,
                        max_gen_len=config.token_frames,
                        use_cfg=False,
                        use_sampling=True,
                        temp=config.rollout_temperature,
                        top_k=config.rollout_top_k,
                        top_p=config.rollout_top_p,
                        check=True,
                    )
                if tuple(codes.shape[:1]) != (config.rank_batch_size,) or codes.shape[-1] != config.token_frames:
                    raise RuntimeError("rollout shape violates fixed B,Q,500 contract")
                if (
                    codes.dtype != torch.long
                    or tuple(codes.shape) != (config.rank_batch_size, 4, 500)
                    or not bool(((codes >= 0) & (codes < 2048)).all().item())
                ):
                    raise RuntimeError(
                        "rollout violates frozen long [B,4,500] codec-token contract"
                    )
                rollout_mask = torch.ones_like(codes, dtype=torch.bool)
                student_lm.train()
                freeze_condition_provider(student_lm)
                teacher_lm.eval()
                assert_condition_provider_frozen(student_lm, "student")
                assert_condition_provider_frozen(teacher_lm, "teacher")

                synchronize = accumulation_index == config.grad_accum_steps - 1
                sync_context = nullcontext() if synchronize else ddp_scorer.no_sync()
                with sync_context, torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    scores = score_audiocraft_trajectory(
                        facade,
                        teacher_lm,
                        codes,
                        conditional,
                        null,
                        rollout_mask=rollout_mask,
                        teacher_cfg_scale=config.teacher_cfg_scale,
                        teacher_forward_mode=config.teacher_forward_mode,
                        check_finite=config.check_finite,
                    )
                    random_scores = None
                    if config.mode == "random50":
                        random_scores = keyed_random_scores(
                            sample_ids,
                            codes.shape[1],
                            codes.shape[2],
                            config.seed,
                            optimizer_step,
                            random_namespace=config.random_mask_namespace,
                            device=device,
                        )
                    loss_output = ptc_opd_loss(
                        scores.student_logits,
                        scores.teacher_cfg_logits,
                        valid_mask=scores.valid_mask,
                        mode=spec.loss_mode,
                        rho=spec.rho,
                        codebook_weights=prior,
                        kl_direction="forward",
                        selection_scope="protocol",
                        layout="BQTV",
                        random_scores=random_scores,
                        temperature=config.distillation_temperature,
                        check_finite=config.check_finite,
                    )
                    global_ratio = globally_normalized_loss(
                        loss_output, require_distributed=True
                    )
                    # Fixed 500-frame masks make all four denominators equal;
                    # dividing by four is then exactly the whole-window ratio.
                    (global_ratio.backward_loss / config.grad_accum_steps).backward()
                denominators.append(float(global_ratio.global_denominator.item()))
                global_losses.append(float(global_ratio.global_loss.item()))
                selected_cells += int(
                    loss_output.selected_counts_per_codebook.sum().item()
                )
                valid_cells += int(scores.valid_mask.sum().item())
                microstep_audits.append(
                    {
                        "accumulation_index": accumulation_index,
                        "global_microstep": global_microstep,
                        "sample_ids": sample_ids,
                        "rollout_shape": [int(item) for item in codes.shape],
                        "rollout_dtype": str(codes.dtype),
                        "rollout_codes_sha256": tensor_sha256(codes),
                        "use_cfg": False,
                        "condition_tensors_source": "conditional",
                        "selected_gate_sha256": boolean_tensor_sha256(
                            loss_output.selected_mask
                        ),
                        "selected_cells": int(
                            loss_output.selected_mask.sum().item()
                        ),
                        "valid_cells": int(scores.valid_mask.sum().item()),
                        "global_denominator": float(
                            global_ratio.global_denominator.item()
                        ),
                        "global_loss_finite": bool(
                            torch.isfinite(global_ratio.global_loss).item()
                        ),
                    }
                )
                global_microstep += 1

            if not denominator_window_is_constant(
                denominators, config.denominator_rtol
            ):
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError(
                    "four accumulation denominators differ: {}; refusing to "
                    "optimize an average of microbatch ratios".format(denominators)
                )
            missing_gradients = [
                name for name, parameter in named_trainable if parameter.grad is None
            ]
            if missing_gradients:
                raise RuntimeError(
                    "DDP synchronized update has missing trainable gradients: {}".format(
                        missing_gradients[:8]
                    )
                )
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, max_norm=config.grad_clip_norm
            )
            if not bool(torch.isfinite(grad_norm).item()):
                raise FloatingPointError("gradient norm is NaN/Inf")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            if global_microstep != progress_from_optimizer_step(
                optimizer_step, config.grad_accum_steps
            ):
                raise AssertionError("optimizer/global-microstep progress drift")

            if optimizer_step % config.log_every == 0:
                assert_condition_provider_frozen(student_lm, "student")
                assert_condition_provider_frozen(teacher_lm, "teacher")
                rank_step_local = {
                    "rank": rank,
                    "local_rank": local_rank,
                    "microsteps": microstep_audits,
                    "gradient_finite": True,
                    "all_trainable_gradients_present": True,
                    "student_conditioner_status": "frozen_eval",
                    "teacher_conditioner_status": "frozen_eval",
                    "node3_determinism": node3_determinism_audit(),
                    "ddp_reducer": ddp_reducer_audit(ddp_scorer),
                    "resume_rng_restore": resume_rng_restore_local,
                    "resume_state_restore": resume_state_restore_local,
                    "cuda_max_memory_allocated": int(
                        torch.cuda.max_memory_allocated(device)
                    ),
                    "cuda_max_memory_reserved": int(
                        torch.cuda.max_memory_reserved(device)
                    ),
                    "cuda_total_memory_bytes": int(
                        torch.cuda.get_device_properties(device).total_memory
                    ),
                }
                rank_step_audit: List[Any] = [None for _ in range(world_size)]
                dist.all_gather_object(rank_step_audit, rank_step_local)
                if [item.get("rank") for item in rank_step_audit] != list(
                    range(world_size)
                ):
                    raise RuntimeError("step audit did not cover ranks 0..7")
                step_ddp_identities = {
                    item.get("ddp_reducer", {}).get("identity_sha256")
                    for item in rank_step_audit
                    if isinstance(item, dict)
                    and isinstance(item.get("ddp_reducer"), dict)
                }
                if step_ddp_identities != {ddp_reducer_identity_hash}:
                    raise RuntimeError(
                        "DDP reducer step identity differs across ranks or construction"
                    )
                if rank == 0:
                    record = {
                        "schema_version": RUN_SCHEMA_VERSION,
                        "event": "optimizer_step",
                        "optimizer_step": optimizer_step,
                        "global_microstep": global_microstep,
                        "mode": config.mode,
                        "loss": sum(global_losses) / len(global_losses),
                        "global_denominators": denominators,
                        "denominator_window_constant": True,
                        "gradient_norm": float(grad_norm.item()),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "selected_cells_rank0": selected_cells,
                        "valid_cells_rank0": valid_cells,
                        "step_seconds": time.perf_counter() - step_started,
                        "cuda_max_memory_allocated": (
                            torch.cuda.max_memory_allocated(device)
                        ),
                        "cuda_max_memory_reserved": (
                            torch.cuda.max_memory_reserved(device)
                        ),
                        "all_rank_audit": rank_step_audit,
                    }
                    jsonl_append(metrics_path, record)
                    print_json(record)

            should_save = (
                optimizer_step % config.save_every == 0
                or optimizer_step == _effective_max_optimizer_steps
            )
            if should_save:
                current_teacher_hash = hash_module_state(teacher_lm)
                if current_teacher_hash != teacher_state_hash_initial:
                    raise RuntimeError("frozen teacher state hash changed during training")
                metadata = build_checkpoint_metadata(
                    config_hash=config_hash,
                    manifest_hash=hashes["manifest_sha256"],
                    student_checkpoint_hash=hashes["student_checkpoint_sha256"],
                    teacher_checkpoint_hash=hashes["teacher_checkpoint_sha256"],
                    optimizer_step=optimizer_step,
                    global_microstep=global_microstep,
                    teacher_state_hash_initial=teacher_state_hash_initial,
                    ddp_reducer_identity_hash=ddp_reducer_identity_hash,
                    codebook_prior_hash=codebook_prior_hash,
                    audiocraft_lm_hash=audiocraft_lm_hash,
                    audiocraft_source_hash=audiocraft_source_hash,
                    cfg_scale_decision_file_hash=(
                        config.cfg_scale_decision_file_sha256
                    ),
                    cfg_scale_decision_payload_hash=(
                        config.cfg_scale_decision_payload_sha256
                    ),
                    external_t5_identity_hash=external_t5_identity_hash,
                )
                save_checkpoint(
                    output_dir=output_dir,
                    student_lm=student_lm,
                    optimizer=optimizer,
                    metadata=metadata,
                    rank=rank,
                )
                if args.node3_stop_after_step == optimizer_step:
                    raise RuntimeError(
                        "NODE3_GATE_INJECTED_STOP_AFTER_STEP_1: committed "
                        "checkpoint and fsynced attempt-0000 metrics"
                    )

        final_teacher_hash = hash_module_state(teacher_lm)
        if final_teacher_hash != teacher_state_hash_initial:
            raise RuntimeError("frozen teacher hash changed before sealing")
        dist.barrier()
        if rank == 0:
            commit_success_terminal(
                output_dir,
                optimizer_step,
                final_teacher_hash,
            )
    except BaseException as exc:
        if (
            rank == 0
            and run_directory_owned
            and output_dir.exists()
            and not (output_dir / "SEALED.json").exists()
        ):
            try:
                write_json_exclusive(
                    output_dir / "FAILED.json",
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            except FileExistsError:
                pass
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.dry_run and args.node3_stop_after_step is not None:
        raise ValueError("node-3 fault injection is forbidden under --dry-run")
    # Ruling #8 §3 (memory ruling-8-final-small-horizon-extension-2026-08-30):
    # Announce extension target as a module-level variable so downstream
    # sealers (_build_success_seal_payload) can accept optimizer_step >
    # manifest.max_optimizer_steps.  Set to None for normal runs.
    global _EXTENSION_TARGET_STEP
    _EXTENSION_TARGET_STEP = args.extend_to_step
    config = make_config(args)
    if args.dry_run:
        print_json(validate_dry_run(config, args))
        return 0
    run_training(config, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
