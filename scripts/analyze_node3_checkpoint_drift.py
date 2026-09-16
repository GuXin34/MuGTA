#!/usr/bin/env python3
"""Fail-closed warm-process versus cold-resume checkpoint continuity audit.

This consumer deliberately does not claim bitwise identity between a process
that has executed two optimizer steps and a fresh process restored at step 1.
It instead requires:

* every step-1 state group to be bit-exact, with only each run's separately
  validated ``metadata.config_sha256`` excluded from the comparison;
* every step-2 control-plane value to be exact (with the sole, documented
  exception of the run-local ``metadata.config_sha256`` value);
* exact tensor structure and non-floating state; and
* tightly capped, explicitly reported floating drift for student parameters
  and Adam's first/second moments.

The policy contains no generic approximate-equality predicate.  Each cap is a
frozen part of this file, every floating input must be finite, and the CLI
writes a JSON decision and exits non-zero on any violation.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import struct
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor


SCHEMA_VERSION = "ptc-opd-node3-warm-cold-continuity-v1"
CHECKPOINT_SCHEMA_VERSION = "ptc-opd-stage1-checkpoint-v3"

# These are acceptance limits, not display tolerances.  A candidate must pass
# every enabled per-tensor and aggregate cap.  The absolute limits sit less
# than one order of magnitude above the reproducible H20 drift envelope, while
# ULP, support-size, relative-L2, and direction checks prevent a small absolute
# threshold from hiding meaningful corruption.  ULP is gated only above a
# group-specific stability floor: close to zero, a tiny absolute displacement
# can cross millions of representable values while remaining less meaningful
# than its absolute/relative/directional effect.  Below that floor the same
# values remain governed by all non-ULP caps, including
# max_below_ulp_floor_abs.
FROZEN_CAPS: Dict[str, Dict[str, float]] = {
    "student": {
        "max_abs": 2.0e-9,
        "max_ulp": 4,
        "ulp_activation_floor": 2.0 ** -7,
        "max_below_ulp_floor_abs": 2.0e-9,
        "max_per_tensor_changed_fraction": 1.0e-2,
        "max_global_changed_fraction": 1.0e-5,
        "max_per_tensor_symmetric_rel_l2": 1.0e-7,
        "max_global_symmetric_rel_l2": 1.0e-9,
    },
    "exp_avg": {
        "max_abs": 2.0e-10,
        "max_ulp": 8,
        "ulp_activation_floor": 2.0 ** -12,
        "max_below_ulp_floor_abs": 2.0e-10,
        "max_per_tensor_changed_fraction": 7.5e-1,
        "max_global_changed_fraction": 5.0e-1,
        "max_per_tensor_symmetric_rel_l2": 1.0e-4,
        "max_global_symmetric_rel_l2": 5.0e-6,
    },
    "exp_avg_sq": {
        "max_abs": 2.0e-12,
        "max_ulp": 16,
        "ulp_activation_floor": 2.0 ** -19,
        "max_below_ulp_floor_abs": 2.0e-12,
        "max_per_tensor_changed_fraction": 7.5e-1,
        "max_global_changed_fraction": 5.0e-1,
        "max_per_tensor_symmetric_rel_l2": 2.0e-3,
        "max_global_symmetric_rel_l2": 2.0e-4,
    },
    "inferred_clipped_gradient": {
        "max_symmetric_rel_l2": 1.0e-5,
        "max_cosine_gap": 1.0e-9,
    },
    "adam_moment_update_direction": {
        "max_symmetric_rel_l2": 5.0e-4,
        "max_cosine_gap": 1.0e-7,
    },
}

SUPPORTED_FLOAT_DTYPES = {
    torch.float16: (np.uint16, 16),
    torch.bfloat16: (np.uint16, 16),
    torch.float32: (np.uint32, 32),
    torch.float64: (np.uint64, 64),
}


class ContinuityError(RuntimeError):
    """A malformed or non-finite checkpoint cannot receive a decision."""


def _framed(digest: "hashlib._Hash", tag: bytes, payload: bytes = b"") -> None:
    digest.update(tag)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _key_bytes(value: Any) -> bytes:
    if isinstance(value, str):
        return b"s\0" + value.encode("utf-8")
    if type(value) is int:
        return b"i\0" + str(value).encode("ascii")
    raise ContinuityError(
        "checkpoint mapping keys must be strings or integers, found {}".format(
            type(value).__name__
        )
    )


def _update_state_hash(digest: "hashlib._Hash", value: Any) -> None:
    """Canonical value-tree hash including tensor representation metadata."""

    if isinstance(value, Tensor):
        source = value.detach().to(device="cpu")
        if source.layout != torch.strided:
            raise ContinuityError("checkpoint tensor layout must be torch.strided")
        if (source.is_floating_point() or source.is_complex()) and not bool(
            torch.isfinite(source).all().item()
        ):
            raise ContinuityError("checkpoint tensor contains NaN/Inf")
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
        contiguous = source.contiguous()
        storage_view = contiguous.reshape(1) if contiguous.ndim == 0 else contiguous
        byte_view = storage_view.view(torch.uint8).reshape(-1)
        _framed(digest, b"B", memoryview(byte_view.numpy()))
        return
    if isinstance(value, Mapping):
        _framed(digest, b"M", str(len(value)).encode("ascii"))
        for key_payload, key in sorted((_key_bytes(key), key) for key in value):
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
            raise ContinuityError("checkpoint scalar contains NaN/Inf")
        _framed(digest, b"F", struct.pack(">d", value))
        return
    if isinstance(value, str):
        _framed(digest, b"S", value.encode("utf-8"))
        return
    raise ContinuityError(
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


def _read_json(path: Path) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ContinuityError("required JSON is not a regular file: {}".format(path))
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ContinuityError("JSON root must be an object: {}".format(path))
    return payload


def _load_checkpoint(run: Path, step: int) -> Dict[str, Any]:
    run = run.resolve(strict=True)
    directory = run / "checkpoints" / "step-{:05d}".format(step)
    if directory.is_symlink() or not directory.is_dir():
        raise ContinuityError("missing committed checkpoint step {}".format(step))
    if {entry.name for entry in directory.iterdir()} != {
        "checkpoint.pt",
        "SHA256.json",
    }:
        raise ContinuityError(
            "checkpoint step {} is not a closed two-member commit".format(step)
        )
    checkpoint = directory / "checkpoint.pt"
    sidecar = directory / "SHA256.json"
    if any(path.is_symlink() or not path.is_file() for path in (checkpoint, sidecar)):
        raise ContinuityError("checkpoint members must be regular files")
    sidecar_payload = _read_json(sidecar)
    if sidecar_payload.get("path") != "checkpoint.pt":
        raise ContinuityError("checkpoint sidecar path differs")
    if sidecar_payload.get("optimizer_step") != step:
        raise ContinuityError("checkpoint sidecar progress differs")
    if sidecar_payload.get("sha256") != sha256_file(checkpoint):
        raise ContinuityError("checkpoint bytes differ from the sidecar digest")

    options: Dict[str, Any] = {"map_location": "cpu"}
    parameters = inspect.signature(torch.load).parameters
    if "weights_only" in parameters:
        options["weights_only"] = False
    if "mmap" in parameters:
        options["mmap"] = True
    payload = torch.load(str(checkpoint), **options)
    required = {
        "metadata",
        "student_state",
        "optimizer_state",
        "scheduler_state",
        "rng_state_by_rank",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ContinuityError("checkpoint payload field set differs")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise ContinuityError("checkpoint metadata is malformed")
    if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ContinuityError("checkpoint metadata schema differs")
    if metadata.get("optimizer_step") != step:
        raise ContinuityError("checkpoint metadata optimizer step differs")
    if metadata.get("global_microstep") != 4 * step:
        raise ContinuityError("checkpoint metadata microstep differs")
    return payload


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContinuityError("{} must be lowercase SHA-256 hex".format(label))
    return value


def _tensor_structure(tensor: Tensor) -> Dict[str, Any]:
    return {
        "dtype": str(tensor.dtype),
        "layout": str(tensor.layout),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "storage_offset": int(tensor.storage_offset()),
    }


def _require_same_tensor_structure(left: Tensor, right: Tensor, path: str) -> None:
    left_structure = _tensor_structure(left)
    right_structure = _tensor_structure(right)
    if left_structure != right_structure:
        raise ContinuityError("tensor structure differs at {}".format(path))
    if left.layout != torch.strided:
        raise ContinuityError("tensor layout is not strided at {}".format(path))


def _require_finite_real(tensor: Tensor, path: str) -> None:
    if tensor.is_complex() or not tensor.is_floating_point():
        raise ContinuityError("numeric tensor must be real floating point at {}".format(path))
    if tensor.dtype not in SUPPORTED_FLOAT_DTYPES:
        raise ContinuityError("unsupported floating dtype at {}".format(path))
    if not bool(torch.isfinite(tensor).all().item()):
        raise ContinuityError("NaN/Inf at {}".format(path))


def _raw_unsigned_numpy(tensor: Tensor) -> np.ndarray:
    contiguous = tensor.detach().cpu().contiguous()
    if contiguous.dtype in (torch.float16, torch.bfloat16):
        signed = contiguous.view(torch.int16).numpy()
    elif contiguous.dtype == torch.float32:
        signed = contiguous.view(torch.int32).numpy()
    elif contiguous.dtype == torch.float64:
        signed = contiguous.view(torch.int64).numpy()
    else:
        raise ContinuityError("unsupported dtype for ULP analysis")
    unsigned_dtype, _ = SUPPORTED_FLOAT_DTYPES[contiguous.dtype]
    return signed.view(unsigned_dtype).reshape(-1)


def _ordered_float_bits(bits: np.ndarray, width: int) -> np.ndarray:
    sign = np.array(1 << (width - 1), dtype=bits.dtype)
    magnitude_mask = np.array((1 << (width - 1)) - 1, dtype=bits.dtype)
    # Canonicalize signed zero so numerically equal zeros have ULP distance 0.
    bits = bits.copy()
    bits[(bits & magnitude_mask) == 0] = 0
    negative = (bits & sign) != 0
    one = np.array(1, dtype=bits.dtype)
    # Two's-complement the negative encodings so -minsubnormal is exactly one
    # representable step below canonical zero (plain bitwise-not leaves a gap).
    return np.where(negative, np.bitwise_not(bits) + one, bits | sign)


def max_ulp_distance(
    left: Tensor, right: Tensor, ulp_activation_floor: float = 0.0
) -> int:
    """Maximum ULP distance outside an explicitly bounded stability region."""

    if left.numel() == 0:
        return 0
    left_bits = _raw_unsigned_numpy(left)
    right_bits = _raw_unsigned_numpy(right)
    active = (
        torch.maximum(
            torch.abs(left.detach().cpu().to(dtype=torch.float64)),
            torch.abs(right.detach().cpu().to(dtype=torch.float64)),
        )
        > ulp_activation_floor
    ).reshape(-1).numpy()
    left_bits = left_bits[active]
    right_bits = right_bits[active]
    if left_bits.size == 0:
        return 0
    _, width = SUPPORTED_FLOAT_DTYPES[left.dtype]
    maximum = 0
    # Chunking bounds peak memory for the four 2049 x 1024 embeddings.
    for start in range(0, left_bits.size, 1024 * 1024):
        end = min(start + 1024 * 1024, left_bits.size)
        left_ordered = _ordered_float_bits(left_bits[start:end], width)
        right_ordered = _ordered_float_bits(right_bits[start:end], width)
        high = np.maximum(left_ordered, right_ordered)
        low = np.minimum(left_ordered, right_ordered)
        observed = int(np.max(high - low, initial=0))
        maximum = max(maximum, observed)
    return maximum


def _symmetric_rel_l2(diff_sq: float, left_sq: float, right_sq: float) -> float:
    numerator = 2.0 * math.sqrt(max(diff_sq, 0.0))
    denominator = math.sqrt(max(left_sq, 0.0)) + math.sqrt(max(right_sq, 0.0))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def _tensor_metrics(
    left: Tensor, right: Tensor, path: str, group: str
) -> Dict[str, Any]:
    _require_same_tensor_structure(left, right, path)
    _require_finite_real(left, path + ".left")
    _require_finite_real(right, path + ".right")
    left64 = left.detach().cpu().to(dtype=torch.float64)
    right64 = right.detach().cpu().to(dtype=torch.float64)
    difference = left64 - right64
    diff_sq = float(torch.sum(difference * difference).item())
    left_sq = float(torch.sum(left64 * left64).item())
    right_sq = float(torch.sum(right64 * right64).item())
    count = int(left.numel())
    raw_left = _raw_unsigned_numpy(left)
    raw_right = _raw_unsigned_numpy(right)
    changed_count = int(np.count_nonzero(raw_left != raw_right))
    max_abs = float(torch.max(torch.abs(difference)).item()) if count else 0.0
    ulp_activation_floor = float(FROZEN_CAPS[group]["ulp_activation_floor"])
    below_ulp_floor = (
        torch.maximum(torch.abs(left64), torch.abs(right64))
        <= ulp_activation_floor
    )
    below_ulp_floor_count = int(torch.count_nonzero(below_ulp_floor).item())
    below_ulp_floor_changed_count = int(
        np.count_nonzero(
            (raw_left != raw_right) & below_ulp_floor.reshape(-1).numpy()
        )
    )
    below_ulp_floor_max_abs = (
        float(torch.max(torch.abs(difference[below_ulp_floor])).item())
        if below_ulp_floor_count
        else 0.0
    )
    return {
        "path": path,
        "structure_exact": True,
        "structure": _tensor_structure(left),
        "element_count": count,
        "changed_count": changed_count,
        "changed_fraction": changed_count / count if count else 0.0,
        "max_abs": max_abs,
        "max_ulp": max_ulp_distance(left, right, ulp_activation_floor),
        "ulp_activation_floor": ulp_activation_floor,
        "below_ulp_floor_count": below_ulp_floor_count,
        "below_ulp_floor_changed_count": below_ulp_floor_changed_count,
        "below_ulp_floor_max_abs": below_ulp_floor_max_abs,
        "symmetric_rel_l2": _symmetric_rel_l2(diff_sq, left_sq, right_sq),
        "_diff_sq": diff_sq,
        "_left_sq": left_sq,
        "_right_sq": right_sq,
    }


def _public_tensor_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in metrics.items() if not key.startswith("_")}


def _aggregate_tensor_metrics(
    metrics: Sequence[Mapping[str, Any]], group: str
) -> Dict[str, Any]:
    if not metrics:
        raise ContinuityError("numeric group {} is empty".format(group))
    element_count = sum(int(item["element_count"]) for item in metrics)
    changed_count = sum(int(item["changed_count"]) for item in metrics)
    diff_sq = math.fsum(float(item["_diff_sq"]) for item in metrics)
    left_sq = math.fsum(float(item["_left_sq"]) for item in metrics)
    right_sq = math.fsum(float(item["_right_sq"]) for item in metrics)
    return {
        "tensor_count": len(metrics),
        "element_count": element_count,
        "changed_tensor_count": sum(int(item["changed_count"] > 0) for item in metrics),
        "changed_count": changed_count,
        "changed_fraction": changed_count / element_count if element_count else 0.0,
        "max_abs": max(float(item["max_abs"]) for item in metrics),
        "max_ulp": max(int(item["max_ulp"]) for item in metrics),
        "below_ulp_floor_count": sum(
            int(item["below_ulp_floor_count"]) for item in metrics
        ),
        "below_ulp_floor_changed_count": sum(
            int(item["below_ulp_floor_changed_count"]) for item in metrics
        ),
        "below_ulp_floor_max_abs": max(
            float(item["below_ulp_floor_max_abs"]) for item in metrics
        ),
        "symmetric_rel_l2": _symmetric_rel_l2(diff_sq, left_sq, right_sq),
    }


def _cap_violations(
    group: str,
    tensors: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
) -> List[str]:
    caps = FROZEN_CAPS[group]
    violations: List[str] = []
    for metric in tensors:
        path = str(metric["path"])
        for field, cap_name in (
            ("max_abs", "max_abs"),
            ("max_ulp", "max_ulp"),
            ("below_ulp_floor_max_abs", "max_below_ulp_floor_abs"),
            ("changed_fraction", "max_per_tensor_changed_fraction"),
            ("symmetric_rel_l2", "max_per_tensor_symmetric_rel_l2"),
        ):
            if float(metric[field]) > float(caps[cap_name]):
                violations.append(
                    "{} {}={} exceeds {}={}".format(
                        path, field, metric[field], cap_name, caps[cap_name]
                    )
                )
    for field, cap_name in (
        ("max_abs", "max_abs"),
        ("max_ulp", "max_ulp"),
        ("below_ulp_floor_max_abs", "max_below_ulp_floor_abs"),
        ("changed_fraction", "max_global_changed_fraction"),
        ("symmetric_rel_l2", "max_global_symmetric_rel_l2"),
    ):
        if float(aggregate[field]) > float(caps[cap_name]):
            violations.append(
                "{} aggregate {}={} exceeds {}={}".format(
                    group, field, aggregate[field], cap_name, caps[cap_name]
                )
            )
    return violations


def _exact_tree(left: Any, right: Any, label: str) -> Dict[str, Any]:
    left_hash = canonical_state_sha256(left)
    right_hash = canonical_state_sha256(right)
    if left_hash != right_hash:
        raise ContinuityError("{} differs".format(label))
    return {"status": "exact", "canonical_sha256": left_hash}


def _metadata_without_config(metadata: Any, label: str) -> Dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ContinuityError("{} must be a mapping".format(label))
    _require_sha256(metadata.get("config_sha256"), label + ".config_sha256")
    return {key: value for key, value in metadata.items() if key != "config_sha256"}


def _compare_step1_exact(left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    """Exact restore seed, excluding only the run-local configuration digest."""

    left_metadata = _metadata_without_config(left["metadata"], "step-1 uninterrupted metadata")
    right_metadata = _metadata_without_config(right["metadata"], "step-1 resumed metadata")
    exact = {
        "metadata_except_run_local_config_sha256": _exact_tree(
            left_metadata, right_metadata, "step-1 metadata"
        ),
        "student_state": _exact_tree(
            left["student_state"], right["student_state"], "step-1 student_state"
        ),
        "optimizer_state": _exact_tree(
            left["optimizer_state"], right["optimizer_state"], "step-1 optimizer_state"
        ),
        "scheduler_state": _exact_tree(
            left["scheduler_state"], right["scheduler_state"], "step-1 scheduler_state"
        ),
        "rng_state_by_rank": _exact_tree(
            left["rng_state_by_rank"], right["rng_state_by_rank"], "step-1 RNG"
        ),
    }
    return {"status": "exact", "exact_fields": exact}


def _student_metrics(
    left: Any, right: Any
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise ContinuityError("student_state must be a mapping")
    if set(left) != set(right) or not left:
        raise ContinuityError("student_state key set differs or is empty")
    numeric: List[Dict[str, Any]] = []
    for name in sorted(left):
        left_value = left[name]
        right_value = right[name]
        if not isinstance(left_value, Tensor) or not isinstance(right_value, Tensor):
            raise ContinuityError("student_state values must be tensors")
        path = "student_state[{}]".format(json.dumps(name))
        _require_same_tensor_structure(left_value, right_value, path)
        if left_value.is_floating_point():
            numeric.append(_tensor_metrics(left_value, right_value, path, "student"))
        else:
            _exact_tree(left_value, right_value, path)
    aggregate = _aggregate_tensor_metrics(numeric, "student")
    violations = _cap_violations("student", numeric, aggregate)
    return numeric, aggregate, violations


def _optimizer_groups(state: Any, label: str) -> Tuple[Mapping[Any, Any], List[Any]]:
    if not isinstance(state, Mapping) or set(state) != {"state", "param_groups"}:
        raise ContinuityError("{} optimizer_state contract differs".format(label))
    entries = state["state"]
    groups = state["param_groups"]
    if not isinstance(entries, Mapping) or not entries:
        raise ContinuityError("{} optimizer state entries are malformed".format(label))
    if not isinstance(groups, list) or not groups:
        raise ContinuityError("{} optimizer param_groups are malformed".format(label))
    return entries, groups


def _optimizer_non_moment_tree(state: Any, label: str) -> Dict[str, Any]:
    entries, groups = _optimizer_groups(state, label)
    for parameter_id, parameter_state in entries.items():
        if type(parameter_id) is not int or not isinstance(parameter_state, Mapping):
            raise ContinuityError(
                "{} optimizer parameter state is malformed".format(label)
            )
    return {
        "state": {
            parameter_id: {
                key: value
                for key, value in parameter_state.items()
                if key not in {"exp_avg", "exp_avg_sq"}
            }
            for parameter_id, parameter_state in entries.items()
        },
        "param_groups": groups,
    }


def _student_non_floating_tree(state: Any, label: str) -> Dict[str, Tensor]:
    if not isinstance(state, Mapping):
        raise ContinuityError("{} must be a mapping".format(label))
    result: Dict[str, Tensor] = {}
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, Tensor):
            raise ContinuityError("{} must map names to tensors".format(label))
        if not tensor.is_floating_point():
            result[name] = tensor
    return result


def _optimizer_step_value(value: Any, parameter_id: int) -> int:
    if isinstance(value, Tensor):
        if value.numel() != 1 or not bool(torch.isfinite(value).all().item()):
            raise ContinuityError(
                "optimizer step is not a finite scalar for parameter {}".format(
                    parameter_id
                )
            )
        numeric = float(value.item())
    elif isinstance(value, (int, float)):
        numeric = float(value)
    else:
        raise ContinuityError(
            "optimizer step has unsupported type for parameter {}".format(parameter_id)
        )
    if not math.isfinite(numeric) or numeric != math.floor(numeric) or numeric < 1:
        raise ContinuityError(
            "optimizer step is not a positive integer for parameter {}".format(
                parameter_id
            )
        )
    return int(numeric)


def _param_group_by_id(groups: Sequence[Any]) -> Dict[int, Mapping[str, Any]]:
    result: Dict[int, Mapping[str, Any]] = {}
    for group_index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            raise ContinuityError("optimizer param group is not a mapping")
        params = group.get("params")
        if not isinstance(params, list):
            raise ContinuityError("optimizer param group params is not a list")
        betas = group.get("betas")
        if (
            not isinstance(betas, (list, tuple))
            or len(betas) != 2
            or not all(isinstance(value, (int, float)) for value in betas)
        ):
            raise ContinuityError("optimizer betas are malformed")
        eps = group.get("eps")
        if not isinstance(eps, (int, float)) or not math.isfinite(float(eps)) or eps <= 0:
            raise ContinuityError("optimizer epsilon is malformed")
        for parameter_id in params:
            if type(parameter_id) is not int or parameter_id in result:
                raise ContinuityError(
                    "optimizer parameter id is malformed or duplicated in group {}".format(
                        group_index
                    )
                )
            result[parameter_id] = group
    return result


def _optimizer_metrics(
    left_step1: Any,
    right_step1: Any,
    left_step2: Any,
    right_step2: Any,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]], List[str]]:
    left1_entries, left1_groups = _optimizer_groups(left_step1, "left step1")
    right1_entries, right1_groups = _optimizer_groups(right_step1, "right step1")
    left2_entries, left2_groups = _optimizer_groups(left_step2, "left step2")
    right2_entries, right2_groups = _optimizer_groups(right_step2, "right step2")
    _exact_tree(left1_groups, right1_groups, "step1 optimizer param_groups")
    _exact_tree(left2_groups, right2_groups, "step2 optimizer param_groups")
    if set(left1_entries) != set(right1_entries) or set(left2_entries) != set(right2_entries):
        raise ContinuityError("optimizer state parameter-id set differs")
    if set(left1_entries) != set(left2_entries):
        raise ContinuityError("optimizer state parameter-id set changed across steps")
    group_by_id = _param_group_by_id(left2_groups)
    if set(group_by_id) != set(left2_entries):
        raise ContinuityError("optimizer state and param_groups parameter ids differ")

    metrics: Dict[str, List[Dict[str, Any]]] = {"exp_avg": [], "exp_avg_sq": []}
    derived_accumulators: Dict[str, Dict[str, List[float]]] = {
        "inferred_clipped_gradient": _new_vector_accumulator(),
        "adam_moment_update_direction": _new_vector_accumulator(),
    }
    for parameter_id in sorted(left2_entries):
        states = (
            left1_entries[parameter_id],
            right1_entries[parameter_id],
            left2_entries[parameter_id],
            right2_entries[parameter_id],
        )
        if any(not isinstance(state, Mapping) for state in states):
            raise ContinuityError("optimizer parameter state must be a mapping")
        key_set = set(states[0])
        if any(set(state) != key_set for state in states[1:]):
            raise ContinuityError(
                "optimizer field set differs for parameter {}".format(parameter_id)
            )
        if not {"exp_avg", "exp_avg_sq"}.issubset(key_set):
            raise ContinuityError(
                "optimizer moments missing for parameter {}".format(parameter_id)
            )
        for field in sorted(key_set - {"exp_avg", "exp_avg_sq"}):
            _exact_tree(
                states[2][field],
                states[3][field],
                "optimizer_state[{}][{}]".format(parameter_id, field),
            )
        for field in ("exp_avg", "exp_avg_sq"):
            if any(not isinstance(state[field], Tensor) for state in states):
                raise ContinuityError("optimizer moment must be a tensor")
            path = "optimizer_state[{}][{}]".format(parameter_id, field)
            # Step 1 was already globally exact; checking all four structures
            # here protects the derived-gradient calculation from schema drift.
            for candidate in states[1:]:
                _require_same_tensor_structure(states[0][field], candidate[field], path)
            metrics[field].append(
                _tensor_metrics(states[2][field], states[3][field], path, field)
            )

        group = group_by_id[parameter_id]
        beta1 = float(group["betas"][0])
        beta2 = float(group["betas"][1])
        eps = float(group["eps"])
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ContinuityError("optimizer beta lies outside [0,1)")
        if "step" not in states[2]:
            raise ContinuityError(
                "optimizer step missing for parameter {}".format(parameter_id)
            )
        optimizer_step = _optimizer_step_value(states[2]["step"], parameter_id)
        if optimizer_step != 2:
            raise ContinuityError(
                "optimizer state step is not 2 for parameter {}".format(parameter_id)
            )
        left_m1 = states[0]["exp_avg"].to(dtype=torch.float64)
        right_m1 = states[1]["exp_avg"].to(dtype=torch.float64)
        left_m2 = states[2]["exp_avg"].to(dtype=torch.float64)
        right_m2 = states[3]["exp_avg"].to(dtype=torch.float64)
        left_v2 = states[2]["exp_avg_sq"].to(dtype=torch.float64)
        right_v2 = states[3]["exp_avg_sq"].to(dtype=torch.float64)
        left_gradient = (left_m2 - beta1 * left_m1) / (1.0 - beta1)
        right_gradient = (right_m2 - beta1 * right_m1) / (1.0 - beta1)
        # The checkpoint is written after optimizer step 2.  This is the
        # bias-corrected Adam moment direction before LR/weight decay.
        bias1 = 1.0 - beta1 ** optimizer_step
        bias2 = 1.0 - beta2 ** optimizer_step
        left_direction = (left_m2 / bias1) / (torch.sqrt(left_v2 / bias2) + eps)
        right_direction = (right_m2 / bias1) / (torch.sqrt(right_v2 / bias2) + eps)
        for tensor, label in (
            (left_gradient, "left inferred gradient"),
            (right_gradient, "right inferred gradient"),
            (left_direction, "left Adam direction"),
            (right_direction, "right Adam direction"),
        ):
            if not bool(torch.isfinite(tensor).all().item()):
                raise ContinuityError(
                    "{} contains NaN/Inf for parameter {}".format(label, parameter_id)
                )
        _accumulate_vector_pair(
            derived_accumulators["inferred_clipped_gradient"],
            left_gradient,
            right_gradient,
        )
        _accumulate_vector_pair(
            derived_accumulators["adam_moment_update_direction"],
            left_direction,
            right_direction,
        )

    result: Dict[str, Any] = {}
    violations: List[str] = []
    for field in ("exp_avg", "exp_avg_sq"):
        aggregate = _aggregate_tensor_metrics(metrics[field], field)
        violations.extend(_cap_violations(field, metrics[field], aggregate))
        result[field] = {
            "aggregate": aggregate,
            "tensors": [_public_tensor_metrics(item) for item in metrics[field]],
        }
    derived = {
        name: _finish_vector_accumulator(accumulator)
        for name, accumulator in derived_accumulators.items()
    }
    return result, derived, violations


def _new_vector_accumulator() -> Dict[str, List[float]]:
    return {
        "diff_sq": [],
        "left_sq": [],
        "right_sq": [],
        "dot": [],
        "element_count": [],
    }


def _accumulate_vector_pair(
    accumulator: Dict[str, List[float]], left: Tensor, right: Tensor
) -> None:
    if left.shape != right.shape:
        raise ContinuityError("derived vector tensor shape differs")
    difference = left - right
    accumulator["diff_sq"].append(float(torch.sum(difference * difference).item()))
    accumulator["left_sq"].append(float(torch.sum(left * left).item()))
    accumulator["right_sq"].append(float(torch.sum(right * right).item()))
    accumulator["dot"].append(float(torch.sum(left * right).item()))
    accumulator["element_count"].append(float(left.numel()))


def _finish_vector_accumulator(
    accumulator: Mapping[str, Sequence[float]]
) -> Dict[str, float]:
    if not accumulator["diff_sq"]:
        raise ContinuityError("derived vector collection is empty")
    diff_sq = math.fsum(accumulator["diff_sq"])
    left_sq = math.fsum(accumulator["left_sq"])
    right_sq = math.fsum(accumulator["right_sq"])
    dot = math.fsum(accumulator["dot"])
    denominator = math.sqrt(left_sq) * math.sqrt(right_sq)
    if denominator == 0.0:
        cosine = 1.0 if left_sq == 0.0 and right_sq == 0.0 else 0.0
    else:
        cosine = max(-1.0, min(1.0, dot / denominator))
    return {
        "tensor_count": len(accumulator["diff_sq"]),
        "element_count": int(math.fsum(accumulator["element_count"])),
        "symmetric_rel_l2": _symmetric_rel_l2(diff_sq, left_sq, right_sq),
        "cosine": cosine,
        "cosine_gap": 1.0 - cosine,
    }


def _derived_violations(group: str, metrics: Mapping[str, float]) -> List[str]:
    caps = FROZEN_CAPS[group]
    violations: List[str] = []
    if metrics["symmetric_rel_l2"] > caps["max_symmetric_rel_l2"]:
        violations.append(
            "{} symmetric_rel_l2={} exceeds {}".format(
                group, metrics["symmetric_rel_l2"], caps["max_symmetric_rel_l2"]
            )
        )
    if metrics["cosine_gap"] > caps["max_cosine_gap"]:
        violations.append(
            "{} cosine_gap={} exceeds {}".format(
                group, metrics["cosine_gap"], caps["max_cosine_gap"]
            )
        )
    return violations


def analyze_runs(uninterrupted: Path, resumed: Path) -> Dict[str, Any]:
    left1 = _load_checkpoint(uninterrupted, 1)
    right1 = _load_checkpoint(resumed, 1)
    left2 = _load_checkpoint(uninterrupted, 2)
    right2 = _load_checkpoint(resumed, 2)

    step1 = _compare_step1_exact(left1, right1)

    left_metadata = _metadata_without_config(left2["metadata"], "uninterrupted metadata")
    right_metadata = _metadata_without_config(right2["metadata"], "resumed metadata")
    control_plane = {
        "metadata_except_run_local_config_sha256": _exact_tree(
            left_metadata, right_metadata, "step2 metadata"
        ),
        "scheduler_state": _exact_tree(
            left2["scheduler_state"], right2["scheduler_state"], "step2 scheduler"
        ),
        "rng_state_by_rank": _exact_tree(
            left2["rng_state_by_rank"], right2["rng_state_by_rank"], "step2 RNG"
        ),
        "optimizer_param_groups_and_non_moment_state": _exact_tree(
            _optimizer_non_moment_tree(left2["optimizer_state"], "left step2"),
            _optimizer_non_moment_tree(right2["optimizer_state"], "right step2"),
            "step2 optimizer param_groups/non-moment state",
        ),
        "student_non_floating_state": _exact_tree(
            _student_non_floating_tree(left2["student_state"], "left step2 student"),
            _student_non_floating_tree(right2["student_state"], "right step2 student"),
            "step2 student non-floating state",
        ),
        "run_local_config_sha256": {
            "status": "individually_validated_not_compared",
            "uninterrupted": left2["metadata"]["config_sha256"],
            "resumed": right2["metadata"]["config_sha256"],
        },
    }

    student_tensors, student_aggregate, violations = _student_metrics(
        left2["student_state"], right2["student_state"]
    )
    optimizer, derived, optimizer_violations = _optimizer_metrics(
        left1["optimizer_state"],
        right1["optimizer_state"],
        left2["optimizer_state"],
        right2["optimizer_state"],
    )
    violations.extend(optimizer_violations)
    gradient = derived["inferred_clipped_gradient"]
    direction = derived["adam_moment_update_direction"]
    violations.extend(_derived_violations("inferred_clipped_gradient", gradient))
    violations.extend(_derived_violations("adam_moment_update_direction", direction))

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not violations else "failed",
        "decision": {
            "violation_count": len(violations),
            "violations": violations,
        },
        "policy": {
            "caps": FROZEN_CAPS,
            "step1": (
                "metadata (except separately validated run-local config_sha256), "
                "student, optimizer, scheduler, and RNG canonical state exact"
            ),
            "step2_control_plane": (
                "exact metadata (except separately validated run-local config_sha256), "
                "scheduler, RNG, optimizer param_groups/non-moment state, and all "
                "tensor dtype/shape/layout/stride/storage_offset"
            ),
            "floating": "finite, explicit per-tensor and aggregate caps; no generic approximate predicate",
        },
        "step_1": step1,
        "step_2": {
            "control_plane": control_plane,
            "student": {
                "aggregate": student_aggregate,
                "tensors": [_public_tensor_metrics(item) for item in student_tensors],
            },
            "optimizer": optimizer,
            "derived": {
                "inferred_clipped_gradient": gradient,
                "adam_moment_update_direction": direction,
            },
        },
    }
    return result


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
        result = analyze_runs(args.uninterrupted_run, args.resumed_run)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "decision": {"violation_count": 1, "violations": [str(exc)]},
        }
    try:
        write_json_exclusive(args.output, result)
    except (OSError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {"schema_version": SCHEMA_VERSION, "status": "failed", "error": str(exc)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": result["status"],
        "output": str(args.output),
    }
    stream = sys.stdout if result["status"] == "passed" else sys.stderr
    print(json.dumps(summary, sort_keys=True), file=stream)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
