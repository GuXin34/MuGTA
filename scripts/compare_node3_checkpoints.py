#!/usr/bin/env python3
"""Compare uninterrupted and resumed node-3 checkpoints at state level."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import struct
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor


SCHEMA_VERSION = "ptc-opd-node3-state-equivalence-v2"
CHECKPOINT_SCHEMA_VERSION = "ptc-opd-stage1-checkpoint-v3"


class StateEquivalenceError(RuntimeError):
    pass


def _framed(digest: "hashlib._Hash", tag: bytes, payload: bytes = b"") -> None:
    digest.update(tag)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _key_bytes(value: Any) -> bytes:
    if isinstance(value, str):
        return b"s\0" + value.encode("utf-8")
    if type(value) is int:
        return b"i\0" + str(value).encode("ascii")
    raise StateEquivalenceError(
        "checkpoint mapping keys must be strings or integers, found {}".format(
            type(value).__name__
        )
    )


def _update_state_hash(digest: "hashlib._Hash", value: Any) -> None:
    if isinstance(value, Tensor):
        source = value.detach().to(device="cpu")
        if source.layout != torch.strided:
            raise StateEquivalenceError(
                "checkpoint tensor layout must be torch.strided"
            )
        if (source.is_floating_point() or source.is_complex()) and not bool(
            torch.isfinite(source).all().item()
        ):
            raise StateEquivalenceError("checkpoint tensor contains NaN/Inf")
        metadata = json.dumps(
            {
                "dtype": str(source.dtype),
                "layout": str(source.layout),
                "shape": list(source.shape),
                "stride": list(source.stride()),
                "storage_offset": int(source.storage_offset()),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        _framed(digest, b"T", metadata)
        tensor = source.contiguous()
        # dtype reinterpretation requires at least one dimension for scalar
        # optimizer counters on the pinned Torch release.
        storage_view = tensor.reshape(1) if tensor.ndim == 0 else tensor
        byte_view = storage_view.view(torch.uint8).reshape(-1)
        _framed(digest, b"B", memoryview(byte_view.numpy()))
        return
    if isinstance(value, Mapping):
        _framed(digest, b"M", str(len(value)).encode("ascii"))
        encoded = sorted((_key_bytes(key), key) for key in value)
        for key_payload, key in encoded:
            _framed(digest, b"K", key_payload)
            _update_state_hash(digest, value[key])
        return
    if isinstance(value, list):
        _framed(digest, b"L", str(len(value)).encode("ascii"))
        for item in value:
            _update_state_hash(digest, item)
        return
    if isinstance(value, tuple):
        _framed(digest, b"U", str(len(value)).encode("ascii"))
        for item in value:
            _update_state_hash(digest, item)
        return
    if value is None:
        _framed(digest, b"N")
        return
    if type(value) is bool:
        _framed(digest, b"Z", b"1" if value else b"0")
        return
    if type(value) is int:
        _framed(digest, b"I", str(value).encode("ascii"))
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise StateEquivalenceError("checkpoint state contains NaN/Inf scalar")
        _framed(digest, b"F", struct.pack(">d", value))
        return
    if isinstance(value, str):
        _framed(digest, b"S", value.encode("utf-8"))
        return
    raise StateEquivalenceError(
        "unsupported checkpoint state type {}".format(type(value).__name__)
    )


def canonical_state_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_state_hash(digest, value)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise StateEquivalenceError("required JSON is not a regular file")
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise StateEquivalenceError("JSON must contain an object")
    return value


def _checkpoint_paths(run: Path, step: int) -> Tuple[Path, Path]:
    directory = run / "checkpoints" / "step-{:05d}".format(step)
    if directory.is_symlink() or not directory.is_dir():
        raise StateEquivalenceError("missing committed step {}".format(step))
    if {item.name for item in directory.iterdir()} != {"checkpoint.pt", "SHA256.json"}:
        raise StateEquivalenceError("checkpoint directory is not a closed two-member commit")
    checkpoint = directory / "checkpoint.pt"
    sidecar = directory / "SHA256.json"
    if any(path.is_symlink() or not path.is_file() for path in (checkpoint, sidecar)):
        raise StateEquivalenceError("checkpoint members must be regular files")
    sidecar_payload = _load_json(sidecar)
    if sidecar_payload.get("optimizer_step") != step:
        raise StateEquivalenceError("checkpoint sidecar step differs")
    if sidecar_payload.get("path") != "checkpoint.pt":
        raise StateEquivalenceError("checkpoint sidecar path differs")
    if sidecar_payload.get("sha256") != sha256_file(checkpoint):
        raise StateEquivalenceError("checkpoint payload differs from its sidecar")
    return checkpoint, sidecar


def fingerprint_checkpoint(run: Path, step: int) -> Dict[str, Any]:
    checkpoint, sidecar = _checkpoint_paths(run.resolve(strict=True), step)
    options: Dict[str, Any] = {"map_location": "cpu"}
    parameters = inspect.signature(torch.load).parameters
    if "weights_only" in parameters:
        options["weights_only"] = False
    if "mmap" in parameters:
        options["mmap"] = True
    # Torch 2.1 exposes mmap= but requires a plain string filename when it is
    # enabled.  Keep mmap for bounded verifier RSS while remaining compatible
    # with the pinned 2.1.0 runtime and newer PathLike-capable releases.
    payload = torch.load(str(checkpoint), **options)
    required = {
        "metadata",
        "student_state",
        "optimizer_state",
        "scheduler_state",
        "rng_state_by_rank",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise StateEquivalenceError("checkpoint payload field set differs")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise StateEquivalenceError("checkpoint metadata is malformed")
    if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise StateEquivalenceError("checkpoint metadata schema differs")
    ddp_reducer_identity = metadata.get("ddp_reducer_identity_sha256")
    if (
        not isinstance(ddp_reducer_identity, str)
        or len(ddp_reducer_identity) != 64
        or any(character not in "0123456789abcdef" for character in ddp_reducer_identity)
    ):
        raise StateEquivalenceError("checkpoint metadata lacks DDP reducer identity")
    if metadata.get("optimizer_step") != step or metadata.get("global_microstep") != 4 * step:
        raise StateEquivalenceError("checkpoint metadata progress differs")
    student_state = payload["student_state"]
    if not isinstance(student_state, Mapping) or not student_state:
        raise StateEquivalenceError("student state must be a nonempty mapping")
    if any(not isinstance(name, str) or not isinstance(value, Tensor) for name, value in student_state.items()):
        raise StateEquivalenceError("student state must map names to tensors")
    optimizer_state = payload["optimizer_state"]
    if not isinstance(optimizer_state, Mapping) or set(optimizer_state) != {
        "state",
        "param_groups",
    }:
        raise StateEquivalenceError("optimizer state contract differs")
    rng = payload["rng_state_by_rank"]
    if not isinstance(rng, list) or len(rng) != 8:
        raise StateEquivalenceError("checkpoint lacks eight-rank RNG state")
    for rank_state in rng:
        if not isinstance(rank_state, dict) or set(rank_state) != {
            "python",
            "torch_cpu",
            "torch_cuda_local",
        }:
            raise StateEquivalenceError("per-rank RNG state contract differs")
    scheduler = payload["scheduler_state"]
    if not isinstance(scheduler, dict) or set(scheduler) != {
        "type",
        "warmup_optimizer_steps",
        "completed_optimizer_steps",
        "learning_rate",
    }:
        raise StateEquivalenceError("scheduler state contract differs")
    if (
        scheduler.get("type") != "linear_warmup_then_constant"
        or scheduler.get("warmup_optimizer_steps") != 50
        or scheduler.get("completed_optimizer_steps") != step
    ):
        raise StateEquivalenceError("scheduler state contract differs")
    learning_rate = scheduler.get("learning_rate")
    if not isinstance(learning_rate, (int, float)) or not math.isfinite(float(learning_rate)):
        raise StateEquivalenceError("scheduler learning rate is non-finite")
    immutable_metadata = {
        key: value
        for key, value in metadata.items()
        if key not in {"config_sha256", "optimizer_step", "global_microstep"}
    }
    return {
        "step": step,
        "checkpoint_sha256": sha256_file(checkpoint),
        "sidecar_sha256": sha256_file(sidecar),
        "student_state_sha256": canonical_state_sha256(student_state),
        "optimizer_state_sha256": canonical_state_sha256(optimizer_state),
        "scheduler_state_sha256": canonical_state_sha256(scheduler),
        "rng_state_by_rank_sha256": canonical_state_sha256(rng),
        "rng_state_per_rank_sha256": [
            canonical_state_sha256(rank_state) for rank_state in rng
        ],
        "scheduler_state": scheduler,
        "immutable_metadata": immutable_metadata,
    }


def compare_runs(uninterrupted: Path, resumed: Path) -> Dict[str, Any]:
    comparisons: Dict[str, Any] = {}
    for step in (1, 2):
        left = fingerprint_checkpoint(uninterrupted, step)
        right = fingerprint_checkpoint(resumed, step)
        exact_fields = (
            "student_state_sha256",
            "optimizer_state_sha256",
            "scheduler_state_sha256",
            "rng_state_by_rank_sha256",
            "rng_state_per_rank_sha256",
            "scheduler_state",
            "immutable_metadata",
        )
        equality = {field: left[field] == right[field] for field in exact_fields}
        if not all(equality.values()):
            failed = sorted(field for field, equal in equality.items() if not equal)
            raise StateEquivalenceError(
                "step {} state equivalence failed for {}".format(step, failed)
            )
        comparisons["step_{}".format(step)] = {
            "status": "exact",
            "exact_fields": equality,
            "uninterrupted": left,
            "resumed": right,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "policy": {
            "steps": [1, 2],
            "student_state": (
                "finite canonical tensor-tree SHA-256 exact, including "
                "dtype/shape/layout/stride/storage_offset"
            ),
            "optimizer_state": (
                "finite canonical tensor-tree SHA-256 exact, including "
                "dtype/shape/layout/stride/storage_offset"
            ),
            "scheduler_state": "canonical value-tree SHA-256 and payload exact",
            "rng_state": "canonical eight-rank and per-rank SHA-256 exact",
            "metadata": "all fields except run-specific config hash and progress exact",
        },
        "comparisons": comparisons,
    }


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uninterrupted-run", type=Path, required=True)
    parser.add_argument("--resumed-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = compare_runs(args.uninterrupted_run, args.resumed_run)
        write_json_exclusive(args.output, result)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {"schema_version": SCHEMA_VERSION, "status": "failed", "error": str(exc)},
                sort_keys=True,
            ),
            file=__import__("sys").stderr,
        )
        return 1
    print(json.dumps({"status": "passed", "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
