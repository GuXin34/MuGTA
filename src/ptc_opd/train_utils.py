"""Pure, testable utilities for the standalone Stage-1 training runner.

The functions in this module do not import AudioCraft and do not initialize a
distributed process group.  That separation is intentional: ``--dry-run`` can
validate the scientific contract, prompt manifest, hashes, and progress math
on a CPU login node before any large model is imported.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


TRAIN_MANIFEST_BASENAME = "train.full.jsonl"
CHECKPOINT_SCHEMA_VERSION = "ptc-opd-stage1-checkpoint-v3"
RUN_SCHEMA_VERSION = "ptc-opd-stage1-run-v3"


@dataclass(frozen=True)
class ModeSpec:
    """Exact mapping from paper condition names to the loss-kernel API."""

    condition: str
    loss_mode: str
    rho: float
    requires_perceptual_prior: bool


MODE_SPECS: Mapping[str, ModeSpec] = {
    "uniform100": ModeSpec("uniform100", "uniform", 1.0, False),
    "codebook100": ModeSpec("codebook100", "codebook_only", 1.0, True),
    "random50": ModeSpec("random50", "random_stratified", 0.5, False),
    "prefix50": ModeSpec("prefix50", "prefix", 0.5, False),
    "disagreement50": ModeSpec(
        "disagreement50", "disagreement", 0.5, False
    ),
    "ptc50": ModeSpec("ptc50", "ptc", 0.5, True),
}


@dataclass(frozen=True)
class PromptRecord:
    sample_id: str
    prompt: str
    line_number: int


@dataclass(frozen=True)
class Stage1Config:
    """Resolved scientific/runtime configuration stored in every checkpoint."""

    manifest: str
    student_checkpoint: str
    teacher_checkpoint: str
    audiocraft_root: str
    output_dir: str
    cfg_scale_decision_dir: str
    cfg_scale_decision_file_sha256: str
    cfg_scale_decision_payload_sha256: str
    cfg_scale_scientific_config_sha256: str
    cfg_generation_checkpoint_sha256: str
    cfg_generation_audiocraft_source_sha256: str
    cfg_generation_loaded_t5_identity_sha256: str
    cfg_generation_state_dict_sha256: str
    mode: str
    seed: int
    learning_rate: float
    weight_decay: float
    max_optimizer_steps: int
    save_every: int
    log_every: int
    codebook_prior_artifact_dir: Optional[str] = None
    rank_batch_size: int = 2
    expected_world_size: int = 8
    grad_accum_steps: int = 4
    effective_global_batch: int = 64
    duration_seconds: float = 10.0
    codec_frame_rate: float = 50.0
    token_frames: int = 500
    rollout_temperature: float = 1.0
    rollout_top_k: int = 250
    rollout_top_p: float = 0.0
    teacher_cfg_scale: float = 3.0
    distillation_temperature: float = 1.0
    grad_clip_norm: float = 1.0
    teacher_forward_mode: str = "batched"
    check_finite: bool = True
    random_mask_namespace: int = 5701
    optimizer_schedule: str = "linear_warmup_then_constant"
    warmup_optimizer_steps: int = 50
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1.0e-8
    kl_direction: str = "forward"
    denominator_rtol: float = 1.0e-6


def resolve_mode(condition: str) -> ModeSpec:
    try:
        return MODE_SPECS[condition]
    except KeyError as exc:
        raise ValueError(
            "unknown Stage-1 mode {!r}; expected one of {}".format(
                condition, sorted(MODE_SPECS)
            )
        ) from exc


def validate_config(config: Stage1Config) -> ModeSpec:
    """Validate frozen Stage-1 invariants and return the resolved mode."""

    spec = resolve_mode(config.mode)
    if config.rank_batch_size != 2:
        raise ValueError("Stage-1 requires physical batch size 2 per GPU")
    if config.expected_world_size != 8:
        raise ValueError("Stage-1 requires exactly eight ranks per independent job")
    if config.grad_accum_steps != 4:
        raise ValueError("Stage-1 requires exactly four accumulation microsteps")
    computed_batch = (
        config.rank_batch_size
        * config.expected_world_size
        * config.grad_accum_steps
    )
    if computed_batch != config.effective_global_batch or computed_batch != 64:
        raise ValueError(
            "effective global batch must be 2 * 8 * 4 = 64, got {}".format(
                computed_batch
            )
        )
    if config.max_optimizer_steps <= 0:
        raise ValueError("max_optimizer_steps must be positive")
    if config.save_every <= 0 or config.log_every <= 0:
        raise ValueError("save_every and log_every must be positive")
    if config.seed < 0 or config.random_mask_namespace < 0:
        raise ValueError("seed and random_mask_namespace must be non-negative")
    for name, value in (
        ("learning_rate", config.learning_rate),
        ("duration_seconds", config.duration_seconds),
        ("codec_frame_rate", config.codec_frame_rate),
        ("rollout_temperature", config.rollout_temperature),
        ("teacher_cfg_scale", config.teacher_cfg_scale),
        ("distillation_temperature", config.distillation_temperature),
        ("grad_clip_norm", config.grad_clip_norm),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError("{} must be finite and positive".format(name))
    if config.weight_decay != 0.0:
        raise ValueError("Stage-1 freezes AdamW weight_decay=0 for pure OPD")
    if config.rollout_top_k < 0:
        raise ValueError("rollout_top_k must be non-negative")
    if not math.isfinite(float(config.rollout_top_p)) or not (
        0.0 <= config.rollout_top_p <= 1.0
    ):
        raise ValueError("rollout_top_p must lie in [0, 1]")
    expected_frames = int(round(config.duration_seconds * config.codec_frame_rate))
    if config.token_frames != expected_frames or config.token_frames != 500:
        raise ValueError(
            "the frozen 10 s / 50 Hz Stage-1 rollout must contain 500 frames"
        )
    if config.teacher_forward_mode not in {"batched", "separate"}:
        raise ValueError("teacher_forward_mode must be 'batched' or 'separate'")
    if config.optimizer_schedule != "linear_warmup_then_constant":
        raise ValueError("Stage-1 requires linear warmup then constant LR")
    if config.warmup_optimizer_steps != 50:
        raise ValueError("Stage-1 requires exactly 50 optimizer warmup updates")
    if (config.adam_beta1, config.adam_beta2, config.adam_eps) != (
        0.9,
        0.95,
        1.0e-8,
    ):
        raise ValueError("Stage-1 AdamW requires betas=(0.9,0.95), eps=1e-8")
    if config.kl_direction != "forward":
        raise ValueError("Stage-1 primary runs require forward KL")
    if config.denominator_rtol != 1.0e-6:
        raise ValueError("Stage-1 freezes accumulation denominator rtol=1e-6")
    if config.teacher_cfg_scale not in {2.0, 3.0, 5.0}:
        raise ValueError(
            "teacher CFG scale must come from the frozen development candidates"
        )
    for name, value in (
        (
            "cfg_scale_decision_file_sha256",
            config.cfg_scale_decision_file_sha256,
        ),
        (
            "cfg_scale_decision_payload_sha256",
            config.cfg_scale_decision_payload_sha256,
        ),
        (
            "cfg_scale_scientific_config_sha256",
            config.cfg_scale_scientific_config_sha256,
        ),
        (
            "cfg_generation_checkpoint_sha256",
            config.cfg_generation_checkpoint_sha256,
        ),
        (
            "cfg_generation_audiocraft_source_sha256",
            config.cfg_generation_audiocraft_source_sha256,
        ),
        (
            "cfg_generation_loaded_t5_identity_sha256",
            config.cfg_generation_loaded_t5_identity_sha256,
        ),
        (
            "cfg_generation_state_dict_sha256",
            config.cfg_generation_state_dict_sha256,
        ),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("{} must be a SHA-256 hex string".format(name))
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError("{} must be hexadecimal".format(name)) from exc
    if not config.cfg_scale_decision_dir:
        raise ValueError("Stage-1 requires a CFG decision artifact directory")
    if Path(config.manifest).name != TRAIN_MANIFEST_BASENAME:
        raise ValueError(
            "training input must be named {}".format(TRAIN_MANIFEST_BASENAME)
        )
    if (
        spec.requires_perceptual_prior
        and config.codebook_prior_artifact_dir is None
    ):
        raise ValueError("{} requires a frozen codebook prior".format(config.mode))
    return spec


def load_prompt_manifest(path: Path) -> List[PromptRecord]:
    """Read and strictly validate the shared unsharded MusicCaps manifest."""

    path = path.resolve()
    if path.name != TRAIN_MANIFEST_BASENAME:
        raise ValueError(
            "training input must be named {}".format(TRAIN_MANIFEST_BASENAME)
        )
    records: List[PromptRecord] = []
    seen = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                raise ValueError("blank manifest line {}".format(line_number))
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "invalid JSON on manifest line {}".format(line_number)
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    "manifest line {} must be a JSON object".format(line_number)
                )
            sample_id = value.get("sample_id")
            prompt = value.get("prompt")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(
                    "manifest line {} has no nonempty sample_id".format(line_number)
                )
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    "manifest line {} has no nonempty prompt".format(line_number)
                )
            if sample_id in seen:
                raise ValueError("duplicate sample_id {!r}".format(sample_id))
            seen.add(sample_id)
            records.append(
                PromptRecord(
                    sample_id=sample_id,
                    prompt=prompt,
                    line_number=line_number,
                )
            )
    if not records:
        raise ValueError("training manifest is empty")
    return records


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    """Hash one local checkpoint file or logical directory tree deterministically.

    File symlinks are followed and hashed under their logical relative names.
    This supports standard Hugging Face snapshot directories without making a
    machine-specific cache target path part of the scientific identity.
    """

    path = path.resolve(strict=True)
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise ValueError("checkpoint path must be a regular file or directory")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError("checkpoint directory contains no files")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = item.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolved_config_dict(config: Stage1Config) -> Dict[str, Any]:
    return asdict(config)


def derive_seed(base_seed: int, namespace: str, *parts: object) -> int:
    """Derive a stable positive 63-bit seed without Python's salted hash()."""

    fields = [str(base_seed), namespace] + [str(part) for part in parts]
    payload = "\0".join(fields).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value & ((1 << 63) - 1)


def microsteps_per_epoch(
    dataset_size: int, world_size: int, rank_batch_size: int
) -> int:
    if min(dataset_size, world_size, rank_batch_size) <= 0:
        raise ValueError("dataset size, world size, and rank batch must be positive")
    steps = dataset_size // (world_size * rank_batch_size)
    if steps <= 0:
        raise ValueError("manifest is smaller than one global physical batch")
    return steps


def progress_from_optimizer_step(
    optimizer_step: int, grad_accum_steps: int
) -> int:
    if optimizer_step < 0 or grad_accum_steps <= 0:
        raise ValueError("invalid optimizer/accumulation step")
    return optimizer_step * grad_accum_steps


def learning_rate_for_update(
    base_learning_rate: float, optimizer_step: int, warmup_steps: int = 50
) -> float:
    """LR used by the update that advances ``optimizer_step`` to ``step+1``.

    The first update uses ``base_lr / 50``, update 50 reaches ``base_lr``, and
    every later update remains constant.  It is independent of run horizon.
    """

    if not math.isfinite(float(base_learning_rate)) or base_learning_rate <= 0.0:
        raise ValueError("base_learning_rate must be finite and positive")
    if optimizer_step < 0 or warmup_steps <= 0:
        raise ValueError("optimizer_step must be non-negative and warmup positive")
    multiplier = min(float(optimizer_step + 1) / float(warmup_steps), 1.0)
    return float(base_learning_rate) * multiplier


def indices_for_microstep(
    *,
    dataset_size: int,
    world_size: int,
    rank: int,
    rank_batch_size: int,
    global_microstep: int,
    seed: int,
) -> Tuple[int, List[int]]:
    """Return ``(epoch, rank-local indices)`` for a deterministic drop-last plan."""

    if not 0 <= rank < world_size:
        raise ValueError("rank is outside world size")
    if global_microstep < 0:
        raise ValueError("global_microstep must be non-negative")
    per_epoch = microsteps_per_epoch(dataset_size, world_size, rank_batch_size)
    epoch, within_epoch = divmod(global_microstep, per_epoch)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(derive_seed(seed, "prompt-order", epoch))
    order = torch.randperm(dataset_size, generator=generator)
    global_batch = world_size * rank_batch_size
    start = within_epoch * global_batch + rank * rank_batch_size
    indices = order[start : start + rank_batch_size].tolist()
    if len(indices) != rank_batch_size:
        raise AssertionError("drop-last planner returned an incomplete rank batch")
    return epoch, [int(index) for index in indices]


def build_checkpoint_metadata(
    *,
    config_hash: str,
    manifest_hash: str,
    student_checkpoint_hash: str,
    teacher_checkpoint_hash: str,
    optimizer_step: int,
    global_microstep: int,
    teacher_state_hash_initial: str,
    ddp_reducer_identity_hash: str,
    codebook_prior_hash: Optional[str] = None,
    audiocraft_lm_hash: Optional[str] = None,
    audiocraft_source_hash: Optional[str] = None,
    cfg_scale_decision_file_hash: Optional[str] = None,
    cfg_scale_decision_payload_hash: Optional[str] = None,
    external_t5_identity_hash: Optional[str] = None,
) -> Dict[str, Any]:
    if global_microstep < 0 or optimizer_step < 0:
        raise ValueError("checkpoint progress cannot be negative")
    values = {
        "config_sha256": config_hash,
        "manifest_sha256": manifest_hash,
        "student_checkpoint_sha256": student_checkpoint_hash,
        "teacher_checkpoint_sha256": teacher_checkpoint_hash,
        "teacher_state_sha256_initial": teacher_state_hash_initial,
        "ddp_reducer_identity_sha256": ddp_reducer_identity_hash,
    }
    if codebook_prior_hash is not None:
        values["codebook_prior_sha256"] = codebook_prior_hash
    if audiocraft_lm_hash is not None:
        values["audiocraft_lm_sha256"] = audiocraft_lm_hash
    if audiocraft_source_hash is not None:
        values["audiocraft_source_sha256"] = audiocraft_source_hash
    if cfg_scale_decision_file_hash is not None:
        values["cfg_scale_decision_file_sha256"] = cfg_scale_decision_file_hash
    if cfg_scale_decision_payload_hash is not None:
        values["cfg_scale_decision_payload_sha256"] = cfg_scale_decision_payload_hash
    if external_t5_identity_hash is not None:
        values["external_t5_identity_sha256"] = external_t5_identity_hash
    for name, value in values.items():
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("{} must be a SHA-256 hex string".format(name))
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError("{} must be hexadecimal".format(name)) from exc
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        **values,
        "optimizer_step": int(optimizer_step),
        "global_microstep": int(global_microstep),
    }


def verify_resume_metadata(
    metadata: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    for key in (
        "schema_version",
        "config_sha256",
        "manifest_sha256",
        "student_checkpoint_sha256",
        "teacher_checkpoint_sha256",
        "teacher_state_sha256_initial",
        "ddp_reducer_identity_sha256",
        "codebook_prior_sha256",
        "audiocraft_lm_sha256",
        "audiocraft_source_sha256",
        "cfg_scale_decision_file_sha256",
        "cfg_scale_decision_payload_sha256",
        "external_t5_identity_sha256",
    ):
        if key in expected and metadata.get(key) != expected.get(key):
            raise ValueError(
                "resume metadata mismatch for {}: {!r} != {!r}".format(
                    key, metadata.get(key), expected.get(key)
                )
            )
    for key in ("optimizer_step", "global_microstep"):
        if key in expected and metadata.get(key) != expected.get(key):
            raise ValueError(
                "resume metadata mismatch for {}: {!r} != {!r}".format(
                    key, metadata.get(key), expected.get(key)
                )
            )


def hash_module_state(module: nn.Module) -> str:
    """Hash every named parameter/buffer, including dtype, shape, and bytes."""

    digest = hashlib.sha256()
    state = module.state_dict()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, Tensor):
            raise TypeError("module state {!r} is not a tensor".format(name))
        contiguous = tensor.detach().cpu().contiguous()
        metadata = json.dumps(
            {
                "name": name,
                "dtype": str(contiguous.dtype),
                "shape": list(contiguous.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        # Viewing as bytes also supports dtypes (notably bfloat16) that NumPy
        # cannot represent directly in older target environments.
        byte_view = contiguous.view(torch.uint8).reshape(-1)
        digest.update(memoryview(byte_view.numpy()))
    return digest.hexdigest()


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "MODE_SPECS",
    "ModeSpec",
    "PromptRecord",
    "RUN_SCHEMA_VERSION",
    "Stage1Config",
    "TRAIN_MANIFEST_BASENAME",
    "build_checkpoint_metadata",
    "canonical_json_sha256",
    "derive_seed",
    "hash_module_state",
    "indices_for_microstep",
    "load_prompt_manifest",
    "learning_rate_for_update",
    "microsteps_per_epoch",
    "progress_from_optimizer_step",
    "resolve_mode",
    "resolved_config_dict",
    "sha256_file",
    "sha256_path",
    "validate_config",
    "verify_resume_metadata",
]
