#!/usr/bin/env python3
"""Generate and select the frozen MusicGen small/medium CFG teacher anchor.

The ``generate`` command is the only model-dependent part.  It produces one
raw float WAV for each ``dev.full.jsonl`` prompt and each of four anchors:
explicit no-CFG and CFG scales 2, 3, and 5.  Sampling is paired by resetting
all RNGs to the same sample-ID-derived seed before every anchor.

The ``decide`` command is model-independent.  It consumes the sealed
generation, quality, and final external-evaluation artifact directories.  It
never accepts loose score/provenance files: every transitive seal and the
frozen external protocol are rehashed before a decision is published.

Both commands refuse overwrite.  Artifact directories are assembled in a
same-filesystem temporary directory and renamed into place only after all
records pass completeness and hash checks.  ``generate --check-only`` never
imports AudioCraft (or any of its dependencies).
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


WORKPACK_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKPACK_ROOT / "src"
SCRIPTS_ROOT = WORKPACK_ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from ptc_opd.cfg_decision import (
    canonical_json_sha256 as canonical_decision_json_sha256,
    verify_cfg_scale_decision,
)
from ptc_opd.musicgen_contract import strict_validate_musicgen_model
from ptc_opd.reproducibility import audiocraft_source_identity, loaded_t5_identity


GENERATION_SCHEMA_VERSION = "ptc-opd-cfg-generation-v2"
SAMPLE_SCHEMA_VERSION = "ptc-opd-cfg-generated-sample-v2"
SEAL_SCHEMA_VERSION = "ptc-opd-cfg-generation-seal-v2"
SCORE_SCHEMA_VERSION = "ptc-opd-cfg-score-v1"
EVALUATOR_PROVENANCE_SCHEMA_VERSION = "ptc-opd-cfg-evaluator-provenance-v1"
QUALITY_PROVENANCE_SCHEMA_VERSION = "ptc-opd-cfg-quality-provenance-v1"
QUALITY_SEAL_SCHEMA_VERSION = "ptc-opd-cfg-quality-seal-v1"
EXTERNAL_SEAL_SCHEMA_VERSION = "ptc-opd-cfg-external-evaluation-seal-v1"
DECISION_SCHEMA_VERSION = "ptc-opd-cfg-scale-decision-v3"
DECISION_SIDECAR_SCHEMA_VERSION = "ptc-opd-cfg-scale-decision-sidecar-v3"

PINNED_AUDIOCRAFT_BASE_COMMIT = "896ec7c47f5e5d1e5aa1e4b260c4405328bf009d"
DEV_MANIFEST_BASENAME = "dev.full.jsonl"
EXPECTED_DEV_PROMPTS = 300
CONDITIONS: Tuple[Tuple[str, Optional[float]], ...] = (
    ("no_cfg", None),
    ("cfg", 2.0),
    ("cfg", 3.0),
    ("cfg", 5.0),
)
GENERATION_BASE_SEED = 29001
GENERATION_SEED_NAMESPACE = "ptc-opd-cfg-anchor-v1"
DURATION_SECONDS = 10.0
SAMPLE_RATE = 32000
CODEC_FRAME_RATE = 50.0
TOKEN_FRAMES = 500
TEMPERATURE = 1.0
TOP_K = 250
TOP_P = 0.0
BOOTSTRAP_SEED = 4703
BOOTSTRAP_REPLICATES = 10_000
CLAP_GUARDRAIL_BASE_SD = -0.10
BOOTSTRAP_PROBABILITY_THRESHOLD = 0.90
ALLOWED_MODEL_IDS = ("facebook/musicgen-small", "facebook/musicgen-medium")
MODEL_ARCHITECTURES = {
    "facebook/musicgen-small": {
        "transformer_dim": 1024,
        "transformer_layers": 24,
        "transformer_heads": 16,
    },
    "facebook/musicgen-medium": {
        "transformer_dim": 1536,
        "transformer_layers": 48,
        "transformer_heads": 24,
    },
}
LM_PARAMETER_DTYPE = "torch.float32"
COMPRESSION_PARAMETER_DTYPE = "torch.float32"
CONDITIONER_PARAMETER_DTYPE = "torch.float32"
CONDITIONER_COMPUTE_DTYPE = "torch.float32"
LM_GENERATION_COMPUTE_DTYPE = "torch.bfloat16"
# EnCodec 32 kHz's SEANet decoder ends in a stacked LSTM.  With the frozen
# torch 2.1 / CUDA 12.1 runtime, that fused CUDA LSTM has no BF16 kernel and
# raises ``_thnn_fused_lstm_cell_cuda not implemented for 'BFloat16'`` when
# decode inherits the LM autocast context.  Generation therefore has two
# explicit compute regions: BF16 for LM sampling and FP32 (autocast disabled)
# for deterministic codec decode.  This identity is sealed in every
# generation artifact and enforced by the independent decision consumer.
COMPRESSION_DECODE_COMPUTE_DTYPE = "torch.float32"

REQUIRED_METRICS = (
    "muq_mi",
    "audiobox_ce",
    "audiobox_pq",
    "music_clap",
)
REQUIRED_EVALUATORS = (
    "muq_eval",
    "audiobox_aesthetics",
    "music_clap",
)
QUALITY_EVALUATORS = (
    "muq_eval",
    "audiobox_aesthetics",
)
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
EXTERNAL_EVALUATION_PROTOCOL = {
    "music_clap_checkpoint_basename": "music_audioset_epoch_15_esc_90.14.pt",
    "amodel": "HTSAT-base",
    "enable_fusion": False,
    "paired_prompt_cosine": True,
    "embedding_l2_normalization": True,
    "fad_computed": False,
}
HEX_DIGITS = frozenset("0123456789abcdef")


def canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    else:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    return (text + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: object) -> str:
    # Scientific JSON identities deliberately exclude the presentation
    # newline used by on-disk pretty JSON writers.  Keep this byte contract
    # identical to the independent downstream decision consumer.
    return canonical_decision_json_sha256(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_named_files(root: Path, files: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    if not files:
        raise ValueError("hash input contains no files: {}".format(root))
    for visible_path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        if visible_path.is_symlink() and not visible_path.exists():
            raise ValueError("broken symlink in hash input: {}".format(visible_path))
        if not visible_path.is_file():
            raise ValueError("hash input is not a regular file: {}".format(visible_path))
        relative = visible_path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = visible_path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with visible_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    if not root.is_dir():
        raise FileNotFoundError(root)
    files: List[Path] = []
    for item in root.rglob("*"):
        if item.is_symlink() and not item.exists():
            raise ValueError("broken symlink in checkpoint tree: {}".format(item))
        if item.is_file():
            files.append(item)
    return _hash_named_files(root, files)


def sha256_python_source_tree(audiocraft_root: Path) -> str:
    package = audiocraft_root / "audiocraft"
    files = list(package.rglob("*.py")) if package.is_dir() else []
    return _hash_named_files(audiocraft_root, files)


def sha256_audiocraft_tree(audiocraft_root: Path) -> str:
    """Hash the complete reproducible checkout, including configs and patch tests.

    Git internals and tool-created caches are excluded because neither is
    imported or executed and both can change after a read-only test.  All
    source, configuration, license, test, and documentation files visible in
    the patched checkout are included.
    """

    excluded_parts = {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
    files: List[Path] = []
    for item in audiocraft_root.rglob("*"):
        relative = item.relative_to(audiocraft_root)
        if any(part in excluded_parts for part in relative.parts):
            continue
        if item.name == ".DS_Store" or item.suffix == ".pyc":
            continue
        if item.is_symlink() and not item.exists():
            raise ValueError("broken symlink in AudioCraft tree: {}".format(item))
        if item.is_file():
            files.append(item)
    return _hash_named_files(audiocraft_root, files)


def _require_sha256(value: object, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(character not in HEX_DIGITS for character in normalized):
        raise ValueError("{} must be 64 lowercase hexadecimal characters".format(label))
    return normalized


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(
            "AudioCraft root must retain Git provenance: {}".format(
                completed.stderr.strip() or root
            )
        )
    return completed.stdout.strip()


def prompt_sha256(prompt: str) -> str:
    return sha256_bytes(prompt.encode("utf-8"))


def condition_id(condition: str, scale: Optional[float]) -> str:
    if condition == "no_cfg" and scale is None:
        return "no_cfg"
    if condition == "cfg" and scale in {2.0, 3.0, 5.0}:
        return "cfg_{}".format(format(scale, ".1f"))
    raise ValueError("invalid frozen anchor condition/scale: {!r}, {!r}".format(condition, scale))


def paired_prompt_seed(sample_id: str) -> int:
    payload = "{}\0{}\0{}".format(
        GENERATION_SEED_NAMESPACE, GENERATION_BASE_SEED, sample_id
    ).encode("utf-8")
    # torch.manual_seed accepts signed 64-bit values.  Zero is legal, but add
    # one so the published seed domain is 1..2^63-1.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1) + 1


def load_dev_manifest(
    path: Path,
    *,
    expected_prompts: int = EXPECTED_DEV_PROMPTS,
    require_basename: bool = True,
) -> List[Dict[str, str]]:
    if require_basename and path.name != DEV_MANIFEST_BASENAME:
        raise ValueError(
            "CFG scale selection requires exact manifest basename {!r}".format(
                DEV_MANIFEST_BASENAME
            )
        )
    if "test" in path.name.casefold():
        raise ValueError("test manifests are forbidden for CFG-scale selection")
    records: List[Dict[str, str]] = []
    seen = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError("{}:{} blank manifest records are forbidden".format(path, line_number))
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("{}:{} invalid JSON".format(path, line_number)) from exc
            if not isinstance(record, dict):
                raise ValueError("{}:{} must be an object".format(path, line_number))
            sample_id = record.get("sample_id")
            prompt = record.get("prompt")
            source_row_sha256 = record.get("source_row_sha256")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError("{}:{} sample_id must be non-empty".format(path, line_number))
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("{}:{} prompt must be non-empty".format(path, line_number))
            _require_sha256(
                source_row_sha256,
                "{}:{} source_row_sha256".format(path, line_number),
            )
            if sample_id in seen:
                raise ValueError("duplicate dev sample_id {!r}".format(sample_id))
            seen.add(sample_id)
            records.append(
                {
                    "sample_id": sample_id,
                    "prompt": prompt,
                    "source_row_sha256": str(source_row_sha256),
                }
            )
    if len(records) != expected_prompts:
        raise ValueError(
            "dev manifest must contain exactly {} prompts, got {}".format(
                expected_prompts, len(records)
            )
        )
    return records


def _validate_local_snapshot(path: Path) -> Dict[str, object]:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("MusicGen checkpoint must be a regular local snapshot directory")
    # AudioCraft loads these two native exports.  Extra upstream members are
    # included in the tree hash but are not mandatory: notably, the public
    # small snapshot may contain model.safetensors while medium does not.
    required = (path / "state_dict.bin", path / "compression_state_dict.bin")
    missing = [str(item) for item in required if item.is_symlink() or not item.is_file()]
    if missing:
        raise ValueError("local MusicGen snapshot is incomplete: {}".format(missing))
    symlinks = [str(item) for item in path.rglob("*") if item.is_symlink()]
    if symlinks:
        raise ValueError(
            "formal MusicGen snapshot must be fully dereferenced; symlinks={}".format(
                symlinks[:10]
            )
        )
    return {
        "checkpoint_sha256": sha256_tree(path),
        "state_dict_sha256": sha256_file(required[0]),
        "compression_state_dict_sha256": sha256_file(required[1]),
    }


def _validate_audiocraft_source(path: Path) -> Dict[str, object]:
    lm_source = path / "audiocraft" / "models" / "lm.py"
    if not lm_source.is_file():
        raise ValueError("AudioCraft source root has no audiocraft/models/lm.py")
    source = lm_source.read_text(encoding="utf-8")
    required_overlay_fragments = (
        "use_cfg: bool = True",
        "condition_tensors:",
        "elif condition_tensors is not None:",
        "CFG condition_tensors batch must be even",
    )
    if any(fragment not in source for fragment in required_overlay_fragments):
        raise ValueError(
            "AudioCraft explicit no-CFG/precomputed-CFG overlay is not present in lm.py"
        )
    head = _git_head(path)
    if head != PINNED_AUDIOCRAFT_BASE_COMMIT:
        raise ValueError(
            "AudioCraft base commit mismatch: expected {}, observed {}".format(
                PINNED_AUDIOCRAFT_BASE_COMMIT, head
            )
        )
    common_source_identity = audiocraft_source_identity(path)
    return {
        "audiocraft_base_commit": head,
        "audiocraft_lm_sha256": sha256_file(lm_source),
        "audiocraft_python_source_sha256": sha256_python_source_tree(path),
        "audiocraft_source_sha256": common_source_identity["tree_sha256"],
        "audiocraft_source_identity": common_source_identity,
    }


def preflight_generation(args: argparse.Namespace) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
    model_id = str(args.model_id)
    if model_id not in ALLOWED_MODEL_IDS:
        raise ValueError("model_id must be one of {}".format(ALLOWED_MODEL_IDS))
    manifest = Path(args.manifest).expanduser().resolve()
    supplied_checkpoint = Path(args.checkpoint).expanduser()
    supplied_source = Path(args.audiocraft_root).expanduser()
    if supplied_source.is_symlink():
        raise ValueError("AudioCraft source root must not be a symlink")
    checkpoint_identity = _validate_local_snapshot(supplied_checkpoint)
    checkpoint = supplied_checkpoint.resolve()
    source = supplied_source.resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite generation output: {}".format(output))
    records = load_dev_manifest(manifest)
    source_identity = _validate_audiocraft_source(source)
    planned_config = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "model_id": model_id,
        "manifest_sha256": sha256_file(manifest),
        **checkpoint_identity,
        **source_identity,
        "conditions": [
            {"condition": condition, "cfg_scale": scale}
            for condition, scale in CONDITIONS
        ],
        "prompt_count": EXPECTED_DEV_PROMPTS,
        "samples_per_prompt": len(CONDITIONS),
        "generation_seed_base": GENERATION_BASE_SEED,
        "generation_seed_namespace": GENERATION_SEED_NAMESPACE,
        "duration_seconds": DURATION_SECONDS,
        "sample_rate": SAMPLE_RATE,
        "codec_frame_rate": CODEC_FRAME_RATE,
        "token_frames": TOKEN_FRAMES,
        "sampling": {
            "use_sampling": True,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "top_p": TOP_P,
            "two_step_cfg": False,
        },
        "audio_serialization": {
            "container": "WAV",
            "subtype": "FLOAT",
            "loudness_normalization": False,
            "clipping_or_rescale": False,
        },
        "offline_resolution": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        },
        "precision_contract": {
            "load_device": "cpu",
            "lm_parameter_dtype": LM_PARAMETER_DTYPE,
            "compression_parameter_dtype": COMPRESSION_PARAMETER_DTYPE,
            "conditioner_parameter_dtype": CONDITIONER_PARAMETER_DTYPE,
            "conditioner_compute_dtype": CONDITIONER_COMPUTE_DTYPE,
            "lm_generation_compute_dtype": LM_GENERATION_COMPUTE_DTYPE,
            "compression_decode_compute_dtype": COMPRESSION_DECODE_COMPUTE_DTYPE,
        },
    }
    return records, {
        "event": "cfg_generation_preflight_ok",
        "audio_output_dir": str(output),
        "planned_config": planned_config,
        "planned_config_sha256": sha256_json(planned_config),
        "audiocraft_imported": "audiocraft" in sys.modules,
    }


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, value: object) -> None:
    _write_bytes(path, canonical_json_bytes(value, pretty=True))


def _write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
            count += 1
        stream.flush()
        os.fsync(stream.fileno())
    return count


@contextlib.contextmanager
def _staged_directory(target: Path) -> Iterator[Path]:
    if target.exists():
        raise FileExistsError("refusing to overwrite artifact directory: {}".format(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".{}.partial.".format(target.name), dir=str(target.parent))
    )
    try:
        yield temporary
        if target.exists():
            raise FileExistsError("output appeared while staging: {}".format(target))
        os.replace(str(temporary), str(target))
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


@contextlib.contextmanager
def _audiocraft_import_path(root: Path) -> Iterator[None]:
    existing = sys.modules.get("audiocraft")
    if existing is not None:
        loaded_from = str(getattr(existing, "__file__", ""))
        if not loaded_from.startswith(str(root)):
            raise RuntimeError("AudioCraft was already imported from another source: {}".format(loaded_from))
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass


def _external_t5_modules(lm: object) -> List[Tuple[object, object]]:
    import torch

    provider = getattr(lm, "condition_provider", None)
    if not isinstance(provider, torch.nn.Module):
        raise TypeError("MusicGen LM has no nn.Module condition_provider")
    found: List[Tuple[object, object]] = []
    for conditioner in provider.modules():
        external = conditioner.__dict__.get("t5")
        if isinstance(external, torch.nn.Module):
            found.append((conditioner, external))
    if len(found) != 1:
        raise RuntimeError("expected exactly one unregistered T5 encoder, got {}".format(len(found)))
    return found


def _hash_module_state(module: object) -> str:
    import torch

    if not isinstance(module, torch.nn.Module):
        raise TypeError("module state hash requires nn.Module")
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        header = canonical_json_bytes(
            {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)},
            pretty=False,
        )
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _prepare_and_identify_frozen_model(
    model: object, device: object, *, model_id: str
) -> Dict[str, object]:
    import torch

    lm = getattr(model, "lm", None)
    compression = getattr(model, "compression_model", None)
    if not isinstance(lm, torch.nn.Module) or not isinstance(compression, torch.nn.Module):
        raise TypeError("MusicGen object must expose LM and compression nn.Modules")
    if model_id not in MODEL_ARCHITECTURES:
        raise ValueError("unsupported frozen MusicGen model_id {!r}".format(model_id))
    for name, module in (("lm", lm), ("compression", compression)):
        dtypes = {parameter.dtype for parameter in module.parameters()}
        if dtypes != {torch.float32}:
            raise RuntimeError(
                "{} must load on CPU with exclusively FP32 parameters, got {}".format(
                    name, sorted(map(str, dtypes))
                )
            )
        devices = {parameter.device.type for parameter in module.parameters()}
        if devices != {"cpu"}:
            raise RuntimeError(
                "{} must be instantiated on CPU before CUDA transfer, got {}".format(
                    name, sorted(devices)
                )
            )
    lm.to(device).requires_grad_(False).eval()
    compression.to(device).requires_grad_(False).eval()
    provider = lm.condition_provider
    provider.requires_grad_(False).eval()
    conditioner, external_t5 = _external_t5_modules(lm)[0]
    if {parameter.dtype for parameter in external_t5.parameters()} != {torch.float32}:
        raise RuntimeError("external T5 must load with exclusively FP32 parameters")
    if {parameter.device.type for parameter in external_t5.parameters()} != {"cpu"}:
        raise RuntimeError("external T5 must load on CPU before CUDA transfer")
    external_t5.to(device).requires_grad_(False).eval()
    if hasattr(conditioner, "device"):
        conditioner.device = str(device)

    if str(getattr(conditioner, "name", "")) != "t5-base":
        raise RuntimeError("pinned MusicGen-small must use t5-base text conditioning")
    tokenizer = getattr(conditioner, "t5_tokenizer", None)
    if tokenizer is None or not callable(getattr(tokenizer, "get_vocab", None)):
        raise RuntimeError("cannot fingerprint the loaded T5 tokenizer")
    t5_config = getattr(external_t5, "config", None)
    if t5_config is None or not callable(getattr(t5_config, "to_dict", None)):
        raise RuntimeError("cannot fingerprint the loaded T5 configuration")

    for name, module in (("lm", lm), ("compression", compression), ("provider", provider), ("t5", external_t5)):
        if module.training:
            raise RuntimeError("{} is not in eval mode".format(name))
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise RuntimeError("{} retains trainable parameters".format(name))

    cfg = getattr(lm, "cfg", None)
    transformer_cfg = getattr(cfg, "transformer_lm", None)
    observed = {
        "num_codebooks": int(getattr(lm, "num_codebooks")),
        "audio_channels": int(getattr(model, "audio_channels")),
        "sample_rate": int(getattr(model, "sample_rate")),
        "frame_rate": float(getattr(model, "frame_rate")),
        "cardinality": int(getattr(lm, "card", -1)),
        "transformer_dim": int(getattr(transformer_cfg, "dim", -1)),
        "transformer_layers": int(getattr(transformer_cfg, "num_layers", -1)),
        "transformer_heads": int(getattr(transformer_cfg, "num_heads", -1)),
    }
    expected = {
        "num_codebooks": 4,
        "cardinality": 2048,
        "audio_channels": 1,
        "sample_rate": SAMPLE_RATE,
        "frame_rate": CODEC_FRAME_RATE,
        **MODEL_ARCHITECTURES[model_id],
    }
    if observed != expected:
        raise RuntimeError(
            "checkpoint architecture does not match {}: {}".format(model_id, observed)
        )

    strict_model_contract = strict_validate_musicgen_model(
        lm, frame_rate=float(getattr(model, "frame_rate"))
    )

    pattern_provider = getattr(lm, "pattern_provider", None)
    if (
        pattern_provider.__class__.__name__ != "DelayedPatternProvider"
        or int(getattr(pattern_provider, "n_q", -1)) != 4
        or list(getattr(pattern_provider, "delays", [])) != [0, 1, 2, 3]
        or int(getattr(pattern_provider, "flatten_first", -1)) != 0
        or int(getattr(pattern_provider, "empty_initial", -1)) != 0
    ):
        raise RuntimeError("MusicGen does not use the frozen standard delay pattern")
    pattern = pattern_provider.get_pattern(TOKEN_FRAMES)
    dummy = torch.zeros(1, 4, TOKEN_FRAMES, dtype=torch.long)
    _, _, mask = pattern.build_pattern_sequence(
        dummy, special_token=2048, keep_only_valid_steps=False
    )
    expected_first = [1, 2, 3, 4]
    expected_last = [500, 501, 502, 503]
    first = []
    last = []
    for q in range(4):
        positions = torch.nonzero(mask[q], as_tuple=False).flatten()
        first.append(int(positions[0].item()))
        last.append(int(positions[-1].item()))
    if (
        tuple(mask.shape) != (4, 504)
        or [int(mask[q].sum().item()) for q in range(4)] != [500, 500, 500, 500]
        or first != expected_first
        or last != expected_last
    ):
        raise RuntimeError(
            "MusicGen delay-pattern mask violates the frozen Q4/T500 contract"
        )
    pattern_mask_sha256 = sha256_bytes(
        mask.to(dtype=torch.uint8).contiguous().numpy().tobytes(order="C")
    )

    t5_identity = loaded_t5_identity(lm)
    return {
        "model_id": model_id,
        "musicgen_architecture": observed,
        "strict_musicgen_contract": {
            "num_codebooks": strict_model_contract.num_codebooks,
            "frame_rate": strict_model_contract.frame_rate,
            "card": strict_model_contract.card,
            "prediction_pattern_checked": strict_model_contract.pattern_checked,
        },
        "musicgen_pattern": {
            "provider": "DelayedPatternProvider",
            "delays": [0, 1, 2, 3],
            "flatten_first": 0,
            "empty_initial": 0,
            "token_frames": TOKEN_FRAMES,
            "pattern_mask_shape": [4, 504],
            "valid_cells_per_codebook": [500, 500, 500, 500],
            "pattern_mask_sha256": pattern_mask_sha256,
        },
        "external_t5_name": "t5-base",
        "external_t5_registered_in_lm_state": False,
        "external_t5_state_sha256": _hash_module_state(external_t5),
        "external_t5_config_sha256": sha256_json(t5_config.to_dict()),
        "external_t5_tokenizer_vocab_sha256": sha256_json(tokenizer.get_vocab()),
        "loaded_t5_identity": t5_identity,
        "precision_contract": {
            "load_device": "cpu",
            "runtime_device_type": str(device.type),
            "lm_parameter_dtype": LM_PARAMETER_DTYPE,
            "compression_parameter_dtype": COMPRESSION_PARAMETER_DTYPE,
            "conditioner_parameter_dtype": CONDITIONER_PARAMETER_DTYPE,
            "conditioner_compute_dtype": CONDITIONER_COMPUTE_DTYPE,
            "lm_generation_compute_dtype": LM_GENERATION_COMPUTE_DTYPE,
            "compression_decode_compute_dtype": COMPRESSION_DECODE_COMPUTE_DTYPE,
            "condition_tensors_precomputed_once_per_prompt": True,
            "conditional_and_null_share_provider_call": True,
        },
    }


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_raw_float_wav(path: Path, waveform: object, sample_rate: int) -> Dict[str, object]:
    import numpy as np
    import soundfile as sf
    import torch

    if not isinstance(waveform, torch.Tensor):
        raise TypeError("decoded audio must be a torch.Tensor")
    value = waveform.detach().float().cpu()
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2 or value.shape[0] != 1:
        raise RuntimeError("expected one mono waveform [1,T], got {}".format(tuple(value.shape)))
    if value.shape[1] != int(DURATION_SECONDS * sample_rate):
        raise RuntimeError("decoded waveform is not exactly 10 seconds: {} samples".format(value.shape[1]))
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError("decoded waveform contains NaN or Inf")
    samples = np.ascontiguousarray(value.transpose(0, 1).numpy(), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    sf.write(str(path), samples, sample_rate, format="WAV", subtype="FLOAT")
    info = sf.info(str(path))
    roundtrip, observed_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if (
        info.format != "WAV"
        or info.subtype != "FLOAT"
        or observed_rate != sample_rate
        or info.frames != samples.shape[0]
        or info.channels != 1
        or not np.array_equal(roundtrip, samples)
    ):
        raise RuntimeError("raw float WAV round-trip integrity check failed: {}".format(path))
    return {
        "audio_sha256": sha256_file(path),
        "audio_frames": int(info.frames),
        "audio_channels": int(info.channels),
        "audio_sample_rate": int(observed_rate),
        "audio_subtype": info.subtype,
        "audio_peak_abs": float(np.max(np.abs(samples))),
    }


def _slice_condition_tensors(
    tensors: Mapping[str, Tuple[object, object]], start: int, stop: int
) -> Dict[str, Tuple[object, object]]:
    return {
        key: (embedding[start:stop], mask[start:stop])
        for key, (embedding, mask) in tensors.items()
    }


def _precompute_prompt_condition_tensors(
    model: object, prompt: str
) -> Tuple[Dict[str, Tuple[object, object]], Dict[str, Tuple[object, object]]]:
    """Run one FP32 provider call for conditional+null and return B/2B views."""

    import torch
    from audiocraft.modules.conditioners import (
        ClassifierFreeGuidanceDropout,
        ConditioningAttributes,
    )

    lm = model.lm
    attributes = [ConditioningAttributes(text={"description": prompt})]
    null_attributes = ClassifierFreeGuidanceDropout(p=1.0)(attributes)
    provider = lm.condition_provider
    device = next(iter(lm.parameters())).device
    # Explicitly disable autocast: Stage-1 and A2 likewise freeze the T5 and
    # condition projection path in FP32 before entering the BF16 LM region.
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        tokenized = provider.tokenize(attributes + null_attributes)
        batched = provider(tokenized)
    if not isinstance(batched, Mapping) or not batched:
        raise RuntimeError("condition provider returned no condition tensors")
    for name, value in batched.items():
        if not isinstance(value, tuple) or len(value) != 2:
            raise RuntimeError("condition tensor {!r} is not an embedding/mask pair".format(name))
        embedding, mask = value
        if embedding.shape[0] != 2 or mask.shape[0] != 2:
            raise RuntimeError("condition tensor {!r} does not have the frozen 2B batch".format(name))
        if embedding.dtype != torch.float32:
            raise RuntimeError("condition embedding {!r} was not computed in FP32".format(name))
        if embedding.device != device or mask.device != device:
            raise RuntimeError("condition tensor {!r} is on the wrong device".format(name))
    return _slice_condition_tensors(batched, 0, 1), dict(batched)


def _generate_one(
    model: object,
    conditional_tensors: Mapping[str, Tuple[object, object]],
    cfg_batched_tensors: Mapping[str, Tuple[object, object]],
    condition: str,
    scale: Optional[float],
    seed: int,
) -> object:
    import inspect
    import torch

    lm = model.lm
    parameters = inspect.signature(lm.generate).parameters
    if "use_cfg" not in parameters or "condition_tensors" not in parameters:
        raise RuntimeError("loaded AudioCraft lacks the explicit no-CFG/CFG tensor overlay")
    _seed_everything(seed)
    common = dict(
        prompt=None,
        num_samples=1,
        max_gen_len=TOKEN_FRAMES,
        use_sampling=True,
        temp=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        two_step_cfg=False,
        check=True,
    )
    device = next(iter(lm.parameters())).device
    if device.type != "cuda":
        raise RuntimeError("formal CFG generation requires CUDA")
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if condition == "no_cfg":
                tokens = lm.generate(
                    conditions=[],
                    condition_tensors=dict(conditional_tensors),
                    use_cfg=False,
                    **common,
                )
            else:
                tokens = lm.generate(
                    conditions=[],
                    condition_tensors=dict(cfg_batched_tensors),
                    cfg_coef=float(scale),
                    use_cfg=True,
                    **common,
                )
            if tuple(tokens.shape) != (1, 4, TOKEN_FRAMES):
                raise RuntimeError(
                    "unexpected generated-token shape {}".format(tuple(tokens.shape))
                )
            if not bool(((tokens >= 0) & (tokens < 2048)).all().item()):
                raise RuntimeError(
                    "generated trajectory contains a special/out-of-cardinality token"
                )
        with torch.autocast(device_type="cuda", enabled=False):
            audio = model.compression_model.decode(tokens, None)
    return audio


def run_generation(args: argparse.Namespace) -> None:
    manifest_records, preflight = preflight_generation(args)
    if args.check_only:
        print(canonical_json_bytes(preflight, pretty=True).decode("utf-8"), end="")
        return

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("real CFG anchor generation requires CUDA")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("real CFG anchor generation requires a CUDA --device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("real CFG anchor generation requires CUDA BF16 support")

    # The local MusicGen snapshot still refers to t5-base by model name.  Force
    # Transformers into offline mode before AudioCraft is imported so the
    # deliberately unregistered T5 can only resolve from the pinned local cache.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    checkpoint = Path(args.checkpoint).resolve()
    source = Path(args.audiocraft_root).resolve()
    output = Path(args.output_dir).resolve()
    with _audiocraft_import_path(source):
        from audiocraft.models import MusicGen
        from audiocraft.models.loaders import load_compression_model, load_lm_model

        # Do not use MusicGen.get_pretrained(device=cuda): in the pinned
        # release that constructor installs a CUDA/FP16 autocast policy.  Load
        # both checkpoint modules on CPU, prove FP32, then move explicitly.
        lm = load_lm_model(str(checkpoint), device="cpu")
        compression_model = load_compression_model(str(checkpoint), device="cpu")
        model = MusicGen(str(checkpoint), compression_model, lm)
        runtime_identity = _prepare_and_identify_frozen_model(
            model, device, model_id=args.model_id
        )
        scientific_config = dict(preflight["planned_config"])
        scientific_config["runtime_identity"] = runtime_identity
        scientific_config_sha256 = sha256_json(scientific_config)

        generated_records: List[Dict[str, object]] = []
        with _staged_directory(output) as staging:
            for prompt_index, manifest_record in enumerate(manifest_records):
                sample_id = manifest_record["sample_id"]
                prompt = manifest_record["prompt"]
                seed = paired_prompt_seed(sample_id)
                conditional_tensors, cfg_batched_tensors = (
                    _precompute_prompt_condition_tensors(model, prompt)
                )
                sample_stem = "{:04d}-{}".format(
                    prompt_index, sha256_bytes(sample_id.encode("utf-8"))[:16]
                )
                for condition, scale in CONDITIONS:
                    anchor_id = condition_id(condition, scale)
                    relative_path = Path("audio") / anchor_id / (sample_stem + ".wav")
                    audio = _generate_one(
                        model,
                        conditional_tensors,
                        cfg_batched_tensors,
                        condition,
                        scale,
                        seed,
                    )
                    audio_identity = _write_raw_float_wav(
                        staging / relative_path, audio, SAMPLE_RATE
                    )
                    generated_records.append(
                        {
                            "schema_version": SAMPLE_SCHEMA_VERSION,
                            "sample_id": sample_id,
                            "prompt": prompt,
                            "prompt_sha256": prompt_sha256(prompt),
                            "condition": condition,
                            "cfg_scale": scale,
                            "condition_id": anchor_id,
                            "seed": seed,
                            "path": relative_path.as_posix(),
                            **audio_identity,
                            "checkpoint_sha256": scientific_config["checkpoint_sha256"],
                            "audiocraft_source_sha256": scientific_config[
                                "audiocraft_source_sha256"
                            ],
                            "scientific_config_sha256": scientific_config_sha256,
                        }
                    )

            expected_count = EXPECTED_DEV_PROMPTS * len(CONDITIONS)
            if len(generated_records) != expected_count:
                raise RuntimeError(
                    "incomplete CFG anchor generation: expected {}, got {}".format(
                        expected_count, len(generated_records)
                    )
                )
            samples_path = staging / "samples.jsonl"
            _write_jsonl(samples_path, generated_records)
            run_metadata = {
                "schema_version": GENERATION_SCHEMA_VERSION,
                "scientific_status": "gpu_generation_completed",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "manifest_path": str(Path(args.manifest).resolve()),
                "checkpoint_path": str(checkpoint),
                "audiocraft_root": str(source),
                "scientific_config": scientific_config,
                "scientific_config_sha256": scientific_config_sha256,
                "sample_records": len(generated_records),
                "raw_float_wav": True,
                "loudness_normalized": False,
            }
            run_path = staging / "generation_run.json"
            _write_json(run_path, run_metadata)
            seal = {
                "schema_version": SEAL_SCHEMA_VERSION,
                "generation_run_sha256": sha256_file(run_path),
                "samples_jsonl_sha256": sha256_file(samples_path),
                "sample_records": len(generated_records),
                "audio_files": len(generated_records),
                "scientific_config_sha256": scientific_config_sha256,
            }
            _write_json(staging / "artifact_seal.json", seal)
            # Gate publication while the directory still has its temporary
            # name.  A target directory is never exposed before every WAV is
            # rehashed and the exact prompt x anchor set is complete.
            verify_generation_directory(staging, rehash_audio=True)

    # The same-filesystem directory rename cannot change file bytes; avoid a
    # second 1.5-GB audio read immediately after the pre-publication gate.
    verified = verify_generation_directory(output, rehash_audio=False)
    print(canonical_json_bytes({"event": "cfg_generation_complete", **verified}, pretty=True).decode("utf-8"), end="")


def _load_json(path: Path) -> Dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(
            "{} contains forbidden non-finite JSON constant {}".format(path, value)
        )

    def unique_object(pairs: Sequence[Tuple[str, object]]) -> Dict[str, object]:
        result: Dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("{} contains duplicate key {!r}".format(path, key))
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as stream:
        value = json.load(
            stream,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    if not isinstance(value, dict):
        raise ValueError("{} must contain a JSON object".format(path))
    return value


def _iter_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError("{}:{} blank JSONL records are forbidden".format(path, line_number))
            try:
                value = json.loads(
                    line,
                    object_pairs_hook=lambda pairs: _unique_json_object(
                        pairs, "{}:{}".format(path, line_number)
                    ),
                    parse_constant=lambda value: _reject_json_constant(
                        value, "{}:{}".format(path, line_number)
                    ),
                )
            except json.JSONDecodeError as exc:
                raise ValueError("{}:{} invalid JSON".format(path, line_number)) from exc
            if not isinstance(value, dict):
                raise ValueError("{}:{} must be an object".format(path, line_number))
            yield value


def _unique_json_object(
    pairs: Sequence[Tuple[str, object]], label: str
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("{} contains duplicate key {!r}".format(label, key))
        result[key] = value
    return result


def _reject_json_constant(value: str, label: str) -> None:
    raise ValueError(
        "{} contains forbidden non-finite JSON constant {}".format(label, value)
    )


def verify_generation_directory(path: Path, *, rehash_audio: bool) -> Dict[str, object]:
    supplied_path = Path(path)
    if supplied_path.is_symlink():
        raise ValueError("generation artifact directory must not be a symlink")
    path = supplied_path.resolve()
    if not path.is_dir():
        raise ValueError("generation artifact directory does not exist: {}".format(path))
    run_path = path / "generation_run.json"
    samples_path = path / "samples.jsonl"
    seal_path = path / "artifact_seal.json"
    for required_path in (run_path, samples_path, seal_path):
        if required_path.is_symlink() or not required_path.is_file():
            raise ValueError(
                "generation artifact member must be a regular non-symlink file: {}".format(
                    required_path.name
                )
            )
    run = _load_json(run_path)
    seal = _load_json(seal_path)
    if run.get("schema_version") != GENERATION_SCHEMA_VERSION:
        raise ValueError("generation_run.json schema mismatch")
    if run.get("scientific_status") != "gpu_generation_completed":
        raise ValueError("generation artifact is not a completed real GPU run")
    if seal.get("schema_version") != SEAL_SCHEMA_VERSION:
        raise ValueError("generation seal schema mismatch")
    if seal.get("generation_run_sha256") != sha256_file(run_path):
        raise ValueError("generation_run.json hash mismatch")
    if seal.get("samples_jsonl_sha256") != sha256_file(samples_path):
        raise ValueError("samples.jsonl hash mismatch")

    records = list(_iter_jsonl(samples_path))
    expected_count = EXPECTED_DEV_PROMPTS * len(CONDITIONS)
    if (
        len(records) != expected_count
        or seal.get("sample_records") != expected_count
        or seal.get("audio_files") != expected_count
        or run.get("sample_records") != expected_count
    ):
        raise ValueError("generation artifact does not contain exactly {} records".format(expected_count))
    if run.get("raw_float_wav") is not True or run.get("loudness_normalized") is not False:
        raise ValueError("generation run metadata violates raw non-normalized WAV contract")
    config_hash = str(run.get("scientific_config_sha256", ""))
    _require_sha256(config_hash, "generation scientific_config_sha256")
    scientific_config = run.get("scientific_config")
    if not isinstance(scientific_config, dict) or sha256_json(scientific_config) != config_hash:
        raise ValueError("generation scientific configuration hash mismatch")
    if seal.get("scientific_config_sha256") != config_hash:
        raise ValueError("generation seal configuration hash mismatch")
    if scientific_config.get("prompt_count") != EXPECTED_DEV_PROMPTS:
        raise ValueError("generation configuration prompt count mismatch")
    frozen_scalar_config = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "samples_per_prompt": len(CONDITIONS),
        "generation_seed_base": GENERATION_BASE_SEED,
        "generation_seed_namespace": GENERATION_SEED_NAMESPACE,
        "duration_seconds": DURATION_SECONDS,
        "sample_rate": SAMPLE_RATE,
        "codec_frame_rate": CODEC_FRAME_RATE,
        "token_frames": TOKEN_FRAMES,
    }
    for name, expected in frozen_scalar_config.items():
        if scientific_config.get(name) != expected:
            raise ValueError("generation configuration {} mismatch".format(name))
    if scientific_config.get("sampling") != {
        "use_sampling": True,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "two_step_cfg": False,
    }:
        raise ValueError("generation sampling configuration mismatch")
    if scientific_config.get("audio_serialization") != {
        "container": "WAV",
        "subtype": "FLOAT",
        "loudness_normalization": False,
        "clipping_or_rescale": False,
    }:
        raise ValueError("generation audio serialization configuration mismatch")
    if scientific_config.get("offline_resolution") != {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }:
        raise ValueError("generation offline-resolution configuration mismatch")
    model_id = scientific_config.get("model_id")
    if model_id not in ALLOWED_MODEL_IDS:
        raise ValueError("generation model_id is outside MusicGen small/medium")
    precision_contract = scientific_config.get("precision_contract")
    expected_precision_contract = {
        "load_device": "cpu",
        "lm_parameter_dtype": LM_PARAMETER_DTYPE,
        "compression_parameter_dtype": COMPRESSION_PARAMETER_DTYPE,
        "conditioner_parameter_dtype": CONDITIONER_PARAMETER_DTYPE,
        "conditioner_compute_dtype": CONDITIONER_COMPUTE_DTYPE,
        "lm_generation_compute_dtype": LM_GENERATION_COMPUTE_DTYPE,
        "compression_decode_compute_dtype": COMPRESSION_DECODE_COMPUTE_DTYPE,
    }
    if precision_contract != expected_precision_contract:
        raise ValueError("generation precision contract mismatch")
    runtime_identity = scientific_config.get("runtime_identity")
    if not isinstance(runtime_identity, dict):
        raise ValueError("generation runtime identity is absent")
    if runtime_identity.get("model_id") != model_id:
        raise ValueError("generation runtime model identity mismatch")
    expected_architecture = {
        "num_codebooks": 4,
        "cardinality": 2048,
        "audio_channels": 1,
        "sample_rate": SAMPLE_RATE,
        "frame_rate": CODEC_FRAME_RATE,
        **MODEL_ARCHITECTURES[str(model_id)],
    }
    if runtime_identity.get("musicgen_architecture") != expected_architecture:
        raise ValueError("generation runtime architecture mismatch")
    runtime_precision = runtime_identity.get("precision_contract")
    if not isinstance(runtime_precision, dict):
        raise ValueError("generation runtime precision identity is absent")
    expected_runtime_precision = {
        **expected_precision_contract,
        "runtime_device_type": "cuda",
        "condition_tensors_precomputed_once_per_prompt": True,
        "conditional_and_null_share_provider_call": True,
    }
    if runtime_precision != expected_runtime_precision:
        differing = {
            name: runtime_precision.get(name)
            for name in sorted(
                set(runtime_precision) | set(expected_runtime_precision)
            )
            if runtime_precision.get(name) != expected_runtime_precision.get(name)
        }
        raise ValueError(
            "generation runtime precision identity mismatch: {}".format(differing)
        )
    expected_condition_config = [
        {"condition": condition, "cfg_scale": scale}
        for condition, scale in CONDITIONS
    ]
    if scientific_config.get("conditions") != expected_condition_config:
        raise ValueError("generation configuration anchor set mismatch")
    checkpoint_hash = _require_sha256(
        scientific_config.get("checkpoint_sha256"), "generation checkpoint_sha256"
    )
    source_hash = _require_sha256(
        scientific_config.get("audiocraft_source_sha256"),
        "generation audiocraft_source_sha256",
    )
    if scientific_config.get("audiocraft_base_commit") != PINNED_AUDIOCRAFT_BASE_COMMIT:
        raise ValueError("generation AudioCraft base commit mismatch")
    by_sample: Dict[str, set] = {}
    prompt_identity_by_sample: Dict[str, Tuple[str, str]] = {}
    seen_paths = set()
    for record in records:
        if record.get("schema_version") != SAMPLE_SCHEMA_VERSION:
            raise ValueError("generated sample schema mismatch")
        sample_id = str(record.get("sample_id", ""))
        prompt = record.get("prompt")
        if not sample_id or not isinstance(prompt, str) or not prompt:
            raise ValueError("generated record has invalid sample_id/prompt")
        observed_prompt_sha256 = record.get("prompt_sha256")
        if observed_prompt_sha256 != prompt_sha256(prompt):
            raise ValueError("generated prompt hash mismatch for {}".format(sample_id))
        prompt_identity = (prompt, str(observed_prompt_sha256))
        previous_prompt_identity = prompt_identity_by_sample.setdefault(
            sample_id, prompt_identity
        )
        if previous_prompt_identity != prompt_identity:
            raise ValueError(
                "paired anchors use different prompts for {}".format(sample_id)
            )
        anchor_id = condition_id(str(record.get("condition")), record.get("cfg_scale"))
        if record.get("condition_id") != anchor_id:
            raise ValueError("generated condition_id mismatch")
        expected_seed = paired_prompt_seed(sample_id)
        if record.get("seed") != expected_seed:
            raise ValueError("generated pairing seed mismatch for {}".format(sample_id))
        if record.get("scientific_config_sha256") != config_hash:
            raise ValueError("generated record config hash mismatch")
        if record.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError("generated record checkpoint hash mismatch")
        if record.get("audiocraft_source_sha256") != source_hash:
            raise ValueError("generated record AudioCraft source hash mismatch")
        if (
            record.get("audio_frames") != int(DURATION_SECONDS * SAMPLE_RATE)
            or record.get("audio_channels") != 1
            or record.get("audio_sample_rate") != SAMPLE_RATE
            or record.get("audio_subtype") != "FLOAT"
        ):
            raise ValueError("generated record violates raw 10-second float-WAV contract")
        peak = record.get("audio_peak_abs")
        if (
            isinstance(peak, bool)
            or not isinstance(peak, (int, float))
            or not math.isfinite(float(peak))
            or float(peak) < 0.0
        ):
            raise ValueError("generated record has invalid audio_peak_abs")
        raw_relative = record.get("path")
        if not isinstance(raw_relative, str) or not raw_relative:
            raise ValueError("generated audio path must be a nonempty string")
        relative = Path(raw_relative)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != raw_relative
            or len(relative.parts) != 3
            or relative.parts[0] != "audio"
            or relative.parts[1] != anchor_id
            or relative.suffix != ".wav"
        ):
            raise ValueError("generated audio path must remain relative")
        if relative.as_posix() in seen_paths:
            raise ValueError("duplicate generated audio path")
        seen_paths.add(relative.as_posix())
        audio_path = path / relative
        if audio_path.is_symlink() or not audio_path.is_file():
            raise ValueError(
                "generated audio must be a regular non-symlink file: {}".format(
                    audio_path
                )
            )
        if rehash_audio and record.get("audio_sha256") != sha256_file(audio_path):
            raise ValueError("generated audio hash mismatch: {}".format(audio_path))
        by_sample.setdefault(sample_id, set()).add(anchor_id)
    expected_anchors = {condition_id(*item) for item in CONDITIONS}
    if len(by_sample) != EXPECTED_DEV_PROMPTS:
        raise ValueError("generated prompt set is incomplete")
    incomplete = {key: sorted(expected_anchors - value) for key, value in by_sample.items() if value != expected_anchors}
    if incomplete:
        raise ValueError("generated anchor sets are incomplete: {}".format(incomplete))

    # Close the transitive seal over the filesystem.  samples.jsonl already
    # binds every expected WAV path and digest; this exact-member check makes
    # that binding complete by rejecting unlisted files, symlinked members,
    # and unexpected directory levels (including stray metadata files).
    expected_files = {
        "generation_run.json",
        "samples.jsonl",
        "artifact_seal.json",
        *seen_paths,
    }
    expected_directories = {
        "audio",
        *("audio/{}".format(anchor) for anchor in expected_anchors),
    }
    observed_files = set()
    observed_directories = set()
    for member in path.rglob("*"):
        relative_member = member.relative_to(path).as_posix()
        if member.is_symlink():
            raise ValueError(
                "generation artifact contains a forbidden symlink: {}".format(
                    relative_member
                )
            )
        if member.is_file():
            observed_files.add(relative_member)
        elif member.is_dir():
            observed_directories.add(relative_member)
        else:
            raise ValueError(
                "generation artifact contains a non-regular member: {}".format(
                    relative_member
                )
            )
    if observed_files != expected_files:
        raise ValueError(
            "generation artifact file set mismatch: missing={}, extra={}".format(
                sorted(expected_files - observed_files),
                sorted(observed_files - expected_files),
            )
        )
    if observed_directories != expected_directories:
        raise ValueError(
            "generation artifact directory set mismatch: missing={}, extra={}".format(
                sorted(expected_directories - observed_directories),
                sorted(observed_directories - expected_directories),
            )
        )
    return {
        "generation_dir": str(path),
        "generation_run_sha256": sha256_file(run_path),
        "samples_jsonl_sha256": sha256_file(samples_path),
        "artifact_seal_sha256": sha256_file(seal_path),
        "scientific_config_sha256": config_hash,
        "generation_identity": {
            "model_id": str(model_id),
            "lm_parameter_dtype": LM_PARAMETER_DTYPE,
            "compression_parameter_dtype": COMPRESSION_PARAMETER_DTYPE,
            "conditioner_parameter_dtype": CONDITIONER_PARAMETER_DTYPE,
            "conditioner_compute_dtype": CONDITIONER_COMPUTE_DTYPE,
            "lm_generation_compute_dtype": LM_GENERATION_COMPUTE_DTYPE,
            "compression_decode_compute_dtype": COMPRESSION_DECODE_COMPUTE_DTYPE,
            "manifest_sha256": _require_sha256(
                scientific_config.get("manifest_sha256"),
                "generation manifest_sha256",
            ),
            "checkpoint_sha256": checkpoint_hash,
            "state_dict_sha256": _require_sha256(
                scientific_config.get("state_dict_sha256"),
                "generation state_dict_sha256",
            ),
            "compression_state_dict_sha256": _require_sha256(
                scientific_config.get("compression_state_dict_sha256"),
                "generation compression_state_dict_sha256",
            ),
            "audiocraft_base_commit": scientific_config.get(
                "audiocraft_base_commit"
            ),
            "audiocraft_source_sha256": source_hash,
            "audiocraft_lm_sha256": _require_sha256(
                scientific_config.get("audiocraft_lm_sha256"),
                "generation audiocraft_lm_sha256",
            ),
            "loaded_t5_identity_sha256": _require_sha256(
                scientific_config.get("runtime_identity", {})
                .get("loaded_t5_identity", {})
                .get("identity_sha256"),
                "generation loaded_t5_identity_sha256",
            ),
        },
        "prompt_count": len(by_sample),
        "sample_records": len(records),
    }


def generation_binding(identity: Mapping[str, object]) -> Dict[str, object]:
    """Return the exact portable generation identity used by evaluator seals."""

    return {
        field: identity[field]
        for field in (
            "generation_run_sha256",
            "samples_jsonl_sha256",
            "artifact_seal_sha256",
            "scientific_config_sha256",
            "prompt_count",
            "sample_records",
        )
    }


def validate_evaluator_provenance(
    path: Path,
    *,
    expected_generation: Optional[Mapping[str, object]] = None,
) -> Tuple[Dict[str, object], str]:
    value = _load_json(path)
    expected_fields = {
        "schema_version",
        "status",
        "metrics",
        "evaluators",
        "generation",
        "quality_artifact",
        "protocol",
        "offline_environment",
    }
    if set(value) != expected_fields:
        raise ValueError("evaluator provenance field set mismatch")
    if value.get("schema_version") != EVALUATOR_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("evaluator provenance schema mismatch")
    if value.get("status") != "accepted_external_evaluation":
        raise ValueError("external evaluator provenance is not explicitly accepted")
    evaluators = value.get("evaluators")
    if not isinstance(evaluators, dict) or set(evaluators) != set(REQUIRED_EVALUATORS):
        raise ValueError("evaluator provenance evaluator set mismatch")
    for evaluator in REQUIRED_EVALUATORS:
        item = evaluators.get(evaluator)
        if not isinstance(item, dict) or set(item) != {
            "checkpoint_sha256",
            "source_sha256",
            "config_sha256",
            "details",
        }:
            raise ValueError("missing evaluator provenance for {}".format(evaluator))
        for field in ("checkpoint_sha256", "source_sha256", "config_sha256"):
            _require_sha256(item.get(field), "{}.{}".format(evaluator, field))
        details = item.get("details")
        if not isinstance(details, dict) or not details:
            raise ValueError("{}.details must be a nonempty object".format(evaluator))
    metrics = value.get("metrics")
    if metrics != list(REQUIRED_METRICS):
        raise ValueError("evaluator provenance metric contract must equal {}".format(REQUIRED_METRICS))
    generation = value.get("generation")
    if not isinstance(generation, dict) or set(generation) != {
        "generation_run_sha256",
        "samples_jsonl_sha256",
        "artifact_seal_sha256",
        "scientific_config_sha256",
        "prompt_count",
        "sample_records",
    }:
        raise ValueError("evaluator provenance generation binding is malformed")
    for field in (
        "generation_run_sha256",
        "samples_jsonl_sha256",
        "artifact_seal_sha256",
        "scientific_config_sha256",
    ):
        _require_sha256(generation.get(field), "generation.{}".format(field))
    for field in ("prompt_count", "sample_records"):
        count = generation.get(field)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(
                "evaluator provenance generation {} must be positive".format(field)
            )
    if expected_generation is not None and generation != generation_binding(expected_generation):
        raise ValueError("evaluator provenance/generation binding mismatch")
    quality_artifact = value.get("quality_artifact")
    if not isinstance(quality_artifact, dict) or set(quality_artifact) != {
        "artifact_seal_sha256",
        "quality_scores_sha256",
        "quality_provenance_sha256",
    }:
        raise ValueError("evaluator provenance quality-artifact binding is malformed")
    for field in sorted(quality_artifact):
        _require_sha256(
            quality_artifact.get(field), "quality_artifact.{}".format(field)
        )
    if value.get("protocol") != EXTERNAL_EVALUATION_PROTOCOL:
        raise ValueError("external evaluator protocol mismatch")
    if value.get("offline_environment") != OFFLINE_ENVIRONMENT:
        raise ValueError("external evaluator offline environment mismatch")
    return value, sha256_file(path)


def load_and_match_scores(
    generation_dir: Path,
    scores_path: Path,
    evaluator_provenance_path: Path,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    generation_identity = verify_generation_directory(generation_dir, rehash_audio=True)
    generation_records = list(_iter_jsonl(generation_dir / "samples.jsonl"))
    expected = {
        (str(record["sample_id"]), str(record["condition_id"])): record
        for record in generation_records
    }
    provenance, provenance_sha256 = validate_evaluator_provenance(
        evaluator_provenance_path,
        expected_generation=generation_identity,
    )
    scores: List[Dict[str, object]] = []
    observed = set()
    for record in _iter_jsonl(scores_path):
        if set(record) != {
            "schema_version",
            "sample_id",
            "prompt_sha256",
            "condition",
            "cfg_scale",
            "condition_id",
            "audio_sha256",
            "scientific_config_sha256",
            "evaluator_provenance_sha256",
            "metrics",
        }:
            raise ValueError("score record field set mismatch")
        if record.get("schema_version") != SCORE_SCHEMA_VERSION:
            raise ValueError("score record schema mismatch")
        sample_id = str(record.get("sample_id", ""))
        anchor_id = condition_id(str(record.get("condition")), record.get("cfg_scale"))
        key = (sample_id, anchor_id)
        if key in observed:
            raise ValueError("duplicate score record {}".format(key))
        if key not in expected:
            raise ValueError("score record is outside generated anchor set: {}".format(key))
        generated = expected[key]
        for field in (
            "prompt_sha256",
            "condition",
            "cfg_scale",
            "condition_id",
            "audio_sha256",
            "scientific_config_sha256",
        ):
            if record.get(field) != generated.get(field):
                raise ValueError("score/generation {} mismatch for {}".format(field, key))
        if record.get("condition_id") != anchor_id:
            raise ValueError("score condition_id mismatch for {}".format(key))
        if record.get("evaluator_provenance_sha256") != provenance_sha256:
            raise ValueError("score evaluator provenance hash mismatch for {}".format(key))
        metrics = record.get("metrics")
        if not isinstance(metrics, dict) or set(metrics) != set(REQUIRED_METRICS):
            raise ValueError("score record metric field set mismatch")
        normalized_metrics: Dict[str, float] = {}
        for metric in REQUIRED_METRICS:
            value = metrics.get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("{} must be numeric for {}".format(metric, key))
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError("{} must be finite for {}".format(metric, key))
            normalized_metrics[metric] = numeric
        scores.append(
            {
                "sample_id": sample_id,
                "condition_id": anchor_id,
                "metrics": normalized_metrics,
            }
        )
        observed.add(key)
    missing = sorted(set(expected) - observed)
    if missing or len(scores) != len(expected):
        raise ValueError("score anchor set is incomplete; missing {}".format(missing[:20]))
    return scores, {
        "generation": generation_identity,
        "scores_jsonl_sha256": sha256_file(scores_path),
        "evaluator_provenance_sha256": provenance_sha256,
        "evaluator_provenance": provenance,
    }


def _sample_sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError("base standardization requires at least two prompts")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    sd = math.sqrt(variance)
    if not math.isfinite(sd) or sd <= 0.0:
        raise ValueError("base score standard deviation must be finite and positive")
    return sd


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile input is empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def decide_from_scores(
    scores: Sequence[Mapping[str, object]],
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> Dict[str, object]:
    by_key = {
        (str(record["sample_id"]), str(record["condition_id"])): record["metrics"]
        for record in scores
    }
    sample_ids = sorted(
        sample_id for sample_id, anchor_id in by_key if anchor_id == "no_cfg"
    )
    if len(sample_ids) != EXPECTED_DEV_PROMPTS:
        raise ValueError("decision requires exactly {} base prompts".format(EXPECTED_DEV_PROMPTS))
    base_stats: Dict[str, Dict[str, float]] = {}
    for metric in REQUIRED_METRICS:
        values = [float(by_key[(sample_id, "no_cfg")][metric]) for sample_id in sample_ids]
        base_stats[metric] = {
            "mean": sum(values) / len(values),
            "sample_sd": _sample_sd(values),
        }

    candidate_deltas: Dict[float, Dict[str, List[float]]] = {}
    for scale in (2.0, 3.0, 5.0):
        anchor_id = condition_id("cfg", scale)
        q_values: List[float] = []
        clap_values: List[float] = []
        component_values: Dict[str, List[float]] = {metric: [] for metric in REQUIRED_METRICS}
        for sample_id in sample_ids:
            base = by_key[(sample_id, "no_cfg")]
            candidate = by_key[(sample_id, anchor_id)]
            standardized = {
                metric: (float(candidate[metric]) - float(base[metric]))
                / base_stats[metric]["sample_sd"]
                for metric in REQUIRED_METRICS
            }
            for metric, value in standardized.items():
                component_values[metric].append(value)
            aesthetic = 0.5 * standardized["audiobox_ce"] + 0.5 * standardized["audiobox_pq"]
            q_values.append(0.5 * standardized["muq_mi"] + 0.5 * aesthetic)
            clap_values.append(standardized["music_clap"])
        candidate_deltas[scale] = {
            "q_dev": q_values,
            "clap": clap_values,
            **component_values,
        }

    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    rng = random.Random(bootstrap_seed)
    bootstrap_by_scale: Dict[float, List[float]] = {
        scale: [] for scale in (2.0, 3.0, 5.0)
    }
    # Reuse the same prompt indices across candidates without retaining a
    # 10,000 x 300 Python-int matrix in memory.
    for _ in range(bootstrap_replicates):
        indices = [rng.randrange(len(sample_ids)) for _ in sample_ids]
        for scale in (2.0, 3.0, 5.0):
            q_values = candidate_deltas[scale]["q_dev"]
            bootstrap_by_scale[scale].append(
                sum(q_values[index] for index in indices) / len(indices)
            )
    candidate_results: List[Dict[str, object]] = []
    for scale in (2.0, 3.0, 5.0):
        values = candidate_deltas[scale]
        q_values = values["q_dev"]
        bootstrap = bootstrap_by_scale[scale]
        q_dev = sum(q_values) / len(q_values)
        clap_delta = sum(values["clap"]) / len(values["clap"])
        probability_positive = sum(value > 0.0 for value in bootstrap) / len(bootstrap)
        eligible = (
            q_dev > 0.0
            and probability_positive >= BOOTSTRAP_PROBABILITY_THRESHOLD
            and clap_delta >= CLAP_GUARDRAIL_BASE_SD
        )
        candidate_results.append(
            {
                "cfg_scale": scale,
                "q_dev": q_dev,
                "q_dev_ci95": [_quantile(bootstrap, 0.025), _quantile(bootstrap, 0.975)],
                "paired_bootstrap_probability_positive": probability_positive,
                "music_clap_delta_base_sd": clap_delta,
                "component_delta_base_sd": {
                    metric: sum(values[metric]) / len(values[metric])
                    for metric in REQUIRED_METRICS
                },
                "eligible": eligible,
            }
        )

    eligible_results = [result for result in candidate_results if result["eligible"]]
    selected: Optional[Dict[str, object]] = None
    if eligible_results:
        # There is deliberately no tolerance-based tie.  Lower scale wins only
        # when the serialized Python-float Q_dev values are numerically equal.
        selected = sorted(
            eligible_results,
            key=lambda result: (-float(result["q_dev"]), float(result["cfg_scale"])),
        )[0]
    return {
        "status": "selected" if selected is not None else "stopped_no_eligible_scale",
        "selected_cfg_scale": (
            float(selected["cfg_scale"]) if selected is not None else None
        ),
        "base_standardization": base_stats,
        "candidates": candidate_results,
    }


def run_decision(args: argparse.Namespace) -> int:
    # Preserve the user-supplied final path components until each verifier has
    # had a chance to reject a symlink; resolving here would hide that fact.
    generation_dir = Path(args.generation_dir).expanduser()
    external_evaluation_dir = Path(args.external_evaluation_dir).expanduser()
    quality_dir = Path(args.quality_dir).expanduser()
    scores_path = external_evaluation_dir / "scores.jsonl"
    provenance_path = external_evaluation_dir / "evaluator_provenance.json"
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite CFG decision output: {}".format(output_dir))
    # Import lazily so generation/check-only never acquires any evaluator-side
    # dependency.  cfg_eval_common contains no model imports; this call rehashes
    # the quality artifact, final external artifact, and their transitive
    # generation bindings before loose files are even opened below.
    import cfg_eval_common as evaluation_artifacts

    external_inputs = evaluation_artifacts.verify_external_artifact(
        external_evaluation_dir,
        generation_dir,
        quality_dir,
    )
    scores, inputs = load_and_match_scores(generation_dir, scores_path, provenance_path)
    result = decide_from_scores(scores)
    scientific_config = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "evaluation_manifest": DEV_MANIFEST_BASENAME,
        "candidates": [2.0, 3.0, 5.0],
        "base_anchor": "no_cfg",
        "quality_formula": {
            "q_dev": "0.5*delta_z_muq_mi + 0.5*delta_z_aesthetic",
            "aesthetic": "0.5*delta_z_audiobox_ce + 0.5*delta_z_audiobox_pq",
            "expanded": "0.5*delta_z_muq_mi + 0.25*delta_z_audiobox_ce + 0.25*delta_z_audiobox_pq",
            "standardization": "paired raw delta divided by no_cfg base sample SD",
        },
        "guardrails": {
            "q_dev_strictly_positive": True,
            "paired_bootstrap_probability_positive_gte": BOOTSTRAP_PROBABILITY_THRESHOLD,
            "music_clap_delta_base_sd_gte": CLAP_GUARDRAIL_BASE_SD,
        },
        "paired_prompt_bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "confidence_interval": 0.95,
        },
        "selection": "highest eligible q_dev",
        "tie_break": "lower cfg scale only on exact numerical q_dev equality",
    }
    decision: Dict[str, object] = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "status": result["status"],
        "selected_cfg_scale": result["selected_cfg_scale"],
        "scientific_config": scientific_config,
        "scientific_config_sha256": sha256_json(scientific_config),
        "input_hashes": {
            "generation_run_sha256": inputs["generation"]["generation_run_sha256"],
            "generation_samples_jsonl_sha256": inputs["generation"]["samples_jsonl_sha256"],
            "generation_artifact_seal_sha256": inputs["generation"]["artifact_seal_sha256"],
            "generation_scientific_config_sha256": inputs["generation"]["scientific_config_sha256"],
            "scores_jsonl_sha256": inputs["scores_jsonl_sha256"],
            "evaluator_provenance_sha256": inputs["evaluator_provenance_sha256"],
            "external_evaluation_artifact_seal_sha256": external_inputs[
                "artifact_seal_sha256"
            ],
            "quality_artifact_seal_sha256": external_inputs[
                "quality_artifact_seal_sha256"
            ],
            "quality_scores_jsonl_sha256": external_inputs[
                "quality_scores_sha256"
            ],
            "quality_evaluator_provenance_sha256": external_inputs[
                "quality_provenance_sha256"
            ],
            "external_evaluation_protocol_sha256": external_inputs[
                "protocol_sha256"
            ],
            "external_evaluation_offline_environment_sha256": external_inputs[
                "offline_environment_sha256"
            ],
        },
        "generation_identity": inputs["generation"]["generation_identity"],
        "base_standardization": result["base_standardization"],
        "candidates": result["candidates"],
    }
    decision["decision_payload_sha256"] = sha256_json(decision)

    with _staged_directory(output_dir) as staging:
        decision_path = staging / "cfg_scale_decision.json"
        _write_json(decision_path, decision)
        sidecar = {
            "schema_version": DECISION_SIDECAR_SCHEMA_VERSION,
            "path": "cfg_scale_decision.json",
            "sha256": sha256_file(decision_path),
            "decision_payload_sha256": decision["decision_payload_sha256"],
        }
        _write_json(staging / "cfg_scale_decision.sha256.json", sidecar)
        verify_decision(staging)

    print(canonical_json_bytes({"event": "cfg_scale_decision_written", **decision}, pretty=True).decode("utf-8"), end="")
    return 0 if result["status"] == "selected" else 2


def verify_decision(path: Path) -> Dict[str, object]:
    verified = verify_cfg_scale_decision(Path(path), require_selected=False)
    return {
        "status": verified.status,
        "selected_cfg_scale": verified.selected_cfg_scale,
        "decision_file_sha256": verified.decision_file_sha256,
        "decision_payload_sha256": verified.decision_payload_sha256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate", help="generate four matched MusicGen small/medium anchors"
    )
    generate.add_argument("--manifest", type=Path, required=True)
    generate.add_argument(
        "--model-id", choices=ALLOWED_MODEL_IDS, required=True
    )
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--audiocraft-root", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--device", default="cuda:0")
    generate.add_argument(
        "--check-only",
        action="store_true",
        help="validate paths, manifests, hashes, and overlay without importing AudioCraft",
    )

    decide = subparsers.add_parser("decide", help="select and seal the CFG scale")
    decide.add_argument("--generation-dir", type=Path, required=True)
    decide.add_argument(
        "--external-evaluation-dir",
        type=Path,
        required=True,
        help="sealed directory containing scores, provenance, and external seal",
    )
    decide.add_argument(
        "--quality-dir",
        type=Path,
        required=True,
        help="sealed quality directory transitively referenced by external evaluation",
    )
    decide.add_argument("--output-dir", type=Path, required=True)

    verify_generation = subparsers.add_parser("verify-generation", help="rehash a generation artifact")
    verify_generation.add_argument("--generation-dir", type=Path, required=True)

    verify_decision_parser = subparsers.add_parser("verify-decision", help="verify a decision artifact")
    verify_decision_parser.add_argument("--decision-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        run_generation(args)
        return 0
    if args.command == "decide":
        return run_decision(args)
    if args.command == "verify-generation":
        value = verify_generation_directory(args.generation_dir, rehash_audio=True)
        print(canonical_json_bytes({"event": "cfg_generation_verified", **value}, pretty=True).decode("utf-8"), end="")
        return 0
    if args.command == "verify-decision":
        value = verify_decision(args.decision_dir)
        print(canonical_json_bytes({"event": "cfg_decision_verified", **value}, pretty=True).decode("utf-8"), end="")
        return 0 if value["status"] == "selected" else 2
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        raise SystemExit(1)
