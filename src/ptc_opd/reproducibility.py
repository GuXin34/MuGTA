"""Reproducibility identities for AudioCraft source and external T5 state.

The pinned AudioCraft implementation deliberately stores a non-finetuned T5
encoder outside ``nn.Module`` registration.  A normal ``state_dict`` hash
therefore does not identify the text-conditioning function.  This module
fingerprints the *loaded* encoder and SentencePiece tokenizer without relying
on a machine-specific Hugging Face cache path.

The helpers import neither AudioCraft nor Transformers.  They operate on the
small public object contracts exposed by an already-loaded model, which also
makes them usable in CPU/fake tests and in every generation/training entry
point.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn


T5_IDENTITY_SCHEMA_VERSION = "ptc-opd-loaded-t5-identity-v1"
AUDIOCRAFT_SOURCE_SCHEMA_VERSION = "ptc-opd-audiocraft-source-tree-v1"
REQUIRED_OFFLINE_ENVIRONMENT = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
)
_SOURCE_EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "artifacts",
        "build",
        "dist",
        "logs",
        "outputs",
        "runs",
    }
)
_SOURCE_EXCLUDED_FILE_SUFFIXES = (".pyc", ".pyo")


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_state_sha256(module: nn.Module) -> str:
    """Hash registered parameters/buffers without NumPy dtype assumptions."""

    digest = hashlib.sha256()
    state = module.state_dict()
    if not state:
        raise ValueError("external T5 encoder has an empty state_dict")
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, Tensor):
            raise TypeError("external T5 state {!r} is not a tensor".format(name))
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
        byte_view = contiguous.view(torch.uint8).reshape(-1)
        digest.update(memoryview(byte_view.numpy()))
    return digest.hexdigest()


def require_offline_hf_environment(
    environment: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Fail closed unless every Hugging Face offline switch is exactly ``1``."""

    source = os.environ if environment is None else environment
    values = {name: str(source.get(name, "")) for name in REQUIRED_OFFLINE_ENVIRONMENT}
    wrong = {name: value for name, value in values.items() if value != "1"}
    if wrong:
        rendered = ", ".join("{}={!r}".format(name, value) for name, value in wrong.items())
        raise RuntimeError(
            "formal runs require immutable local caches; set every offline "
            "switch to 1 (wrong/missing: {})".format(rendered)
        )
    return values


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not (float("-inf") < value < float("inf")):
            raise ValueError("tokenizer metadata contains NaN/Inf")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _sentencepiece_bytes(tokenizer: Any) -> bytes:
    processor = getattr(tokenizer, "sp_model", None)
    serializer = getattr(processor, "serialized_model_proto", None)
    if not callable(serializer):
        raise TypeError(
            "T5 tokenizer must expose sp_model.serialized_model_proto()"
        )
    payload = serializer()
    if not isinstance(payload, (bytes, bytearray)) or not payload:
        raise ValueError("T5 SentencePiece serialization is empty")
    return bytes(payload)


def _sanitized_encoder_config(encoder: nn.Module) -> Dict[str, Any]:
    config = getattr(encoder, "config", None)
    to_dict = getattr(config, "to_dict", None)
    if not callable(to_dict):
        return {}
    value = dict(to_dict())
    # These fields identify the cache/tooling, not the numerical function.
    for key in ("_name_or_path", "_commit_hash", "transformers_version"):
        value.pop(key, None)
    return _json_safe(value)


def loaded_t5_identity(lm: nn.Module) -> Dict[str, Any]:
    """Return a path-independent identity of the one loaded AudioCraft T5.

    The primary MusicGen contract has exactly one non-finetuned ``t5-base``
    text conditioner.  Requiring a unique match prevents a partial identity if
    a checkpoint silently changes its conditioner graph.
    """

    provider = getattr(lm, "condition_provider", None)
    if not isinstance(provider, nn.Module):
        raise TypeError("AudioCraft LM has no nn.Module condition_provider")
    matches = []
    for module_name, module in provider.named_modules():
        encoder = module.__dict__.get("t5")
        tokenizer = getattr(module, "t5_tokenizer", None)
        if encoder is None and tokenizer is None:
            continue
        if not isinstance(encoder, nn.Module) or tokenizer is None:
            raise TypeError(
                "T5 conditioner {!r} is missing its external encoder/tokenizer".format(
                    module_name
                )
            )
        matches.append((module_name, module, encoder, tokenizer))
    if len(matches) != 1:
        raise ValueError(
            "primary MusicGen requires exactly one external T5 conditioner, found {}".format(
                len(matches)
            )
        )

    module_name, conditioner, encoder, tokenizer = matches[0]
    model_name = getattr(conditioner, "name", None)
    if model_name != "t5-base":
        raise ValueError(
            "primary MusicGen text conditioner must be t5-base, got {!r}".format(
                model_name
            )
        )
    if bool(getattr(conditioner, "finetune", True)):
        raise ValueError("primary MusicGen T5 must be frozen (finetune=false)")

    vocabulary_method = getattr(tokenizer, "get_vocab", None)
    if not callable(vocabulary_method):
        raise TypeError("T5 tokenizer must expose get_vocab()")
    vocabulary = vocabulary_method()
    if not isinstance(vocabulary, Mapping) or not vocabulary:
        raise ValueError("T5 tokenizer vocabulary is empty")
    normalized_vocabulary = {str(token): int(index) for token, index in vocabulary.items()}
    sentencepiece = _sentencepiece_bytes(tokenizer)
    special_tokens = _json_safe(getattr(tokenizer, "special_tokens_map", {}))
    encoder_config = _sanitized_encoder_config(encoder)
    payload: Dict[str, Any] = {
        "schema_version": T5_IDENTITY_SCHEMA_VERSION,
        "conditioner_module": module_name,
        "conditioner_class": type(conditioner).__qualname__,
        "model_name": model_name,
        "finetune": False,
        "encoder_class": type(encoder).__qualname__,
        "encoder_state_sha256": _tensor_state_sha256(encoder),
        "encoder_config_sha256": canonical_json_sha256(encoder_config),
        "tokenizer_class": type(tokenizer).__qualname__,
        "tokenizer_sentencepiece_sha256": _sha256_bytes(sentencepiece),
        "tokenizer_vocabulary_sha256": canonical_json_sha256(normalized_vocabulary),
        "tokenizer_vocabulary_size": len(normalized_vocabulary),
        "tokenizer_special_tokens_sha256": canonical_json_sha256(special_tokens),
    }
    payload["identity_sha256"] = canonical_json_sha256(payload)
    return payload


def _iter_source_files(root: Path) -> Iterable[Tuple[str, Path]]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("AudioCraft source root must be a directory")
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in _SOURCE_EXCLUDED_DIRECTORY_NAMES for part in relative.parts[:-1]):
            continue
        if path.is_symlink():
            raise ValueError("AudioCraft source tree may not contain symlinks: {}".format(relative))
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("AudioCraft source entry is not regular: {}".format(relative))
        if path.name in {".DS_Store"} or path.suffix in _SOURCE_EXCLUDED_FILE_SUFFIXES:
            continue
        yield relative.as_posix(), path


def audiocraft_source_identity(root: Path) -> Dict[str, Any]:
    """Hash the deployable tree while excluding Git/cache/runtime noise."""

    resolved = root.resolve(strict=True)
    required = resolved / "audiocraft" / "models" / "lm.py"
    if not required.is_file():
        raise ValueError("source root is not an AudioCraft checkout")
    digest = hashlib.sha256()
    count = 0
    for relative, path in _iter_source_files(resolved):
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        count += 1
    if count == 0:
        raise ValueError("AudioCraft source identity contains no files")
    payload = {
        "schema_version": AUDIOCRAFT_SOURCE_SCHEMA_VERSION,
        "file_count": count,
        "tree_sha256": digest.hexdigest(),
    }
    payload["identity_sha256"] = canonical_json_sha256(payload)
    return payload


__all__ = [
    "AUDIOCRAFT_SOURCE_SCHEMA_VERSION",
    "REQUIRED_OFFLINE_ENVIRONMENT",
    "T5_IDENTITY_SCHEMA_VERSION",
    "audiocraft_source_identity",
    "canonical_json_sha256",
    "loaded_t5_identity",
    "require_offline_hf_environment",
]
