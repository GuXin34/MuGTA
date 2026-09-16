#!/usr/bin/env python3
"""Independent, fail-closed verifier for B1 pre-stability artifacts.

This consumer is deliberately pure standard library.  It does not import
Torch, NumPy, AudioCraft, or the producer runners.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import zipfile


SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import b1_prestability_contract as contract


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_B19_ITEM = "B1.9_real_checkpoint_eight_rank_concatenated_reference"


def _exact_keys(value: Any, expected: Iterable[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise contract.ContractError("{} must be an object".format(label))
    expected_set = set(expected)
    if set(value) != expected_set:
        raise contract.ContractError(
            "{} fields differ; missing={}, unexpected={}".format(
                label, sorted(expected_set - set(value)), sorted(set(value) - expected_set)
            )
        )
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise contract.ContractError("{} must be a lowercase SHA-256".format(label))
    return value


def _true(value: Any, label: str) -> None:
    if value is not True:
        raise contract.ContractError("{} must be exact true".format(label))


def _false(value: Any, label: str) -> None:
    if value is not False:
        raise contract.ContractError("{} must be exact false".format(label))


def _threshold(value: Any, expected: float, label: str) -> None:
    observed = contract.require_finite_number(value, label)
    if observed != expected:
        raise contract.ContractError("{} differs from frozen threshold".format(label))


def _strict_lt(value: Any, threshold: float, label: str) -> float:
    observed = contract.require_finite_number(value, label)
    if not observed < threshold:
        raise contract.ContractError("{} did not satisfy exclusive upper bound".format(label))
    return observed


def _strict_gt(value: Any, threshold: float, label: str) -> float:
    observed = contract.require_finite_number(value, label)
    if not observed > threshold:
        raise contract.ContractError("{} did not satisfy exclusive lower bound".format(label))
    return observed


def _validate_offline(value: Any, label: str) -> None:
    expected = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    if value != expected:
        raise contract.ContractError("{} differs from frozen offline environment".format(label))


def _read_npy(blob: bytes, label: str) -> Tuple[Mapping[str, Any], bytes]:
    if not blob.startswith(b"\x93NUMPY") or len(blob) < 10:
        raise contract.ContractError("{} is not a canonical NPY member".format(label))
    major, minor = blob[6], blob[7]
    if (major, minor) == (1, 0):
        header_length = struct.unpack("<H", blob[8:10])[0]
        offset = 10
    elif (major, minor) in ((2, 0), (3, 0)):
        if len(blob) < 12:
            raise contract.ContractError("{} has a truncated NPY header".format(label))
        header_length = struct.unpack("<I", blob[8:12])[0]
        offset = 12
    else:
        raise contract.ContractError("{} uses an unsupported NPY version".format(label))
    header_end = offset + header_length
    if header_end > len(blob) or header_length > 65536:
        raise contract.ContractError("{} has an invalid NPY header length".format(label))
    encoding = "utf-8" if major == 3 else "latin1"
    try:
        header = ast.literal_eval(blob[offset:header_end].decode(encoding).strip())
    except (SyntaxError, ValueError, UnicodeError) as exc:
        raise contract.ContractError("{} has an invalid NPY header".format(label)) from exc
    _exact_keys(header, ("descr", "fortran_order", "shape"), label + " header")
    if header.get("fortran_order") is not False:
        raise contract.ContractError("{} must be C-contiguous".format(label))
    shape = header.get("shape")
    if not isinstance(shape, tuple) or any(type(item) is not int or item < 0 for item in shape):
        raise contract.ContractError("{} NPY shape is malformed".format(label))
    return header, blob[header_end:]


def _torch_int64_hash(raw: bytes) -> str:
    metadata = json.dumps(
        {"dtype": "torch.int64", "shape": [8, 4, 500]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(raw)
    return digest.hexdigest()


def _verify_npz(path: Path, declared_hash: str) -> List[str]:
    if path.is_symlink() or not path.is_file():
        raise contract.ContractError("audit_inputs.npz must be a regular file")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            names = [item.filename for item in infos]
            if names != ["codes.npy", "sample_ids.npy"] or len(set(names)) != 2:
                raise contract.ContractError("NPZ closed member order/set differs")
            if any(item.flag_bits & 0x1 or item.file_size > 2_000_000 for item in infos):
                raise contract.ContractError("NPZ contains encrypted or oversized members")
            codes_blob = archive.read("codes.npy")
            ids_blob = archive.read("sample_ids.npy")
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise contract.ContractError("audit_inputs.npz is unreadable") from exc

    codes_header, codes_raw = _read_npy(codes_blob, "codes.npy")
    if codes_header.get("descr") != "<i8" or codes_header.get("shape") != (8, 4, 500):
        raise contract.ContractError("codes.npy must be little-endian int64 [8,4,500]")
    if len(codes_raw) != 8 * 4 * 500 * 8:
        raise contract.ContractError("codes.npy byte length differs")
    if any(value < 0 or value >= 2048 for (value,) in struct.iter_unpack("<q", codes_raw)):
        raise contract.ContractError("codes.npy contains an out-of-card token")
    if _torch_int64_hash(codes_raw) != declared_hash:
        raise contract.ContractError("codes.npy tensor SHA-256 differs")

    ids_header, ids_raw = _read_npy(ids_blob, "sample_ids.npy")
    descriptor = ids_header.get("descr")
    match = re.fullmatch(r"<U([1-9][0-9]*)", descriptor) if isinstance(descriptor, str) else None
    if match is None or ids_header.get("shape") != (8,):
        raise contract.ContractError("sample_ids.npy must be a Unicode vector of length 8")
    width = int(match.group(1))
    if width > 512 or len(ids_raw) != 8 * width * 4:
        raise contract.ContractError("sample_ids.npy byte length differs")
    sample_ids = []
    for index in range(8):
        block = ids_raw[index * width * 4 : (index + 1) * width * 4]
        try:
            value = block.decode("utf-32-le").rstrip("\x00")
        except UnicodeDecodeError as exc:
            raise contract.ContractError("sample_ids.npy contains invalid Unicode") from exc
        if not value:
            raise contract.ContractError("sample_ids.npy contains an empty ID")
        sample_ids.append(value)
    if len(set(sample_ids)) != 8:
        raise contract.ContractError("sample IDs are not unique")
    return sample_ids


def _verify_payload_seal(
    seal_path: Path,
    *,
    status: str,
    payload_paths: Mapping[str, Path],
) -> str:
    seal = contract.load_json(seal_path, "renamed source artifact seal")
    _exact_keys(seal, ("schema_version", "status", "members"), "source seal")
    if seal.get("schema_version") != contract.SEAL_SCHEMA or seal.get("status") != status:
        raise contract.ContractError("source seal schema/status differs")
    members = seal.get("members")
    if not isinstance(members, dict) or set(members) != set(payload_paths):
        raise contract.ContractError("source seal member set differs")
    for name, path in payload_paths.items():
        identity = members.get(name)
        _exact_keys(identity, ("size_bytes", "sha256"), "source member identity")
        if identity != contract.regular_file_identity(path):
            raise contract.ContractError("source payload identity differs for {}".format(name))
    return contract.sha256_file(seal_path)


def _validate_single_result(result: Mapping[str, Any], npz_path: Path) -> Dict[str, Any]:
    _exact_keys(
        result,
        (
            "schema_version", "scientific_status", "gate_passed",
            "operationally_accepted", "redline_touched", "full_b1_passed",
            "pending_b1_items", "scientific_config", "scientific_config_sha256",
            "runtime", "offline_environment", "audit_batch", "audit_inputs",
            "loaded_t5_identity", "student_state_sha256_before",
            "student_state_sha256_after", "teacher_state_sha256_before",
            "teacher_state_sha256_after", "cases", "diagnostics",
            "upstream_seals_before_after_equal", "upstream_identity_sha256_before",
            "upstream_identity_sha256_after",
        ),
        "single result",
    )
    if result.get("schema_version") != contract.SINGLE_SCHEMA or result.get("scientific_status") != contract.SINGLE_STATUS:
        raise contract.ContractError("single result schema/status differs")
    _true(result.get("gate_passed"), "single gate_passed")
    _true(result.get("operationally_accepted"), "single operationally_accepted")
    _false(result.get("redline_touched"), "single redline_touched")
    _false(result.get("full_b1_passed"), "single full_b1_passed")
    if result.get("pending_b1_items") != [_B19_ITEM, contract.PENDING_B1_ITEM]:
        raise contract.ContractError("single pending B1 items differ")
    config = result.get("scientific_config")
    if not isinstance(config, dict) or config.get("schema_version") != "ptc-opd-b1-single-scientific-config-v1":
        raise contract.ContractError("single scientific config schema differs")
    if contract.canonical_json_sha256(config) != _sha(result.get("scientific_config_sha256"), "single config hash"):
        raise contract.ContractError("single scientific config hash differs")
    fixed_config = {
        "model_id": "facebook/musicgen-small",
        "single_gpu_sample_count": 2,
        "distributed_reserved_sample_count": 8,
        "rollout_seed": contract.ROLLOUT_SEED,
        "rollout_without_cfg": True,
        "rollout_compute_dtype": "torch.bfloat16",
        "scoring_compute_dtype": "torch.float32",
        "divergence_dtype": "torch.float32",
        "duration_frames": 500,
        "temperature": 1.0,
        "top_k": 250,
        "top_p": 0.0,
        "selector_rho": 0.5,
        "random_namespace": contract.RANDOM_NAMESPACE,
        "random_run_seed": contract.RANDOM_RUN_SEED,
        "random_optimizer_step": contract.RANDOM_OPTIMIZER_STEP,
        "single_padding_lengths": list(contract.SINGLE_PADDING_LENGTHS),
        "padding_token_mutation_offset": contract.PADDING_TOKEN_MUTATION_OFFSET,
    }
    for name, expected in fixed_config.items():
        if config.get(name) != expected:
            raise contract.ContractError("single config differs at {}".format(name))
    thresholds = config.get("thresholds")
    _exact_keys(
        thresholds,
        ("cfg1_mean_kl_lt", "cfg1_max_valid_kl_lt", "relative_loss_error_lt", "relative_gradient_error_lt"),
        "single thresholds",
    )
    _threshold(thresholds["cfg1_mean_kl_lt"], contract.CFG1_MEAN_KL_LT, "single cfg1 mean threshold")
    _threshold(thresholds["cfg1_max_valid_kl_lt"], contract.CFG1_MAX_VALID_KL_LT, "single cfg1 max threshold")
    _threshold(thresholds["relative_loss_error_lt"], contract.RELATIVE_LOSS_ERROR_LT, "single loss threshold")
    _threshold(thresholds["relative_gradient_error_lt"], contract.RELATIVE_GRADIENT_ERROR_LT, "single gradient threshold")
    cfg = config.get("cfg_decision")
    if not isinstance(cfg, dict) or cfg.get("selected_cfg_scale") != 5.0 or cfg.get("decision_file_sha256") != contract.EXPECTED_SMALL_CFG_DECISION_SHA256:
        raise contract.ContractError("single retained CFG identity differs")
    node3 = config.get("node3_identity")
    if not isinstance(node3, dict) or node3.get("gate_count") != 15 or node3.get("passed_count") != 15:
        raise contract.ContractError("single retained node-3 identity differs")
    t5 = config.get("t5_closure_identity")
    if not isinstance(t5, dict) or t5.get("artifact_seal_sha256") != contract.EXPECTED_T5_CLOSURE_SEAL_SHA256 or t5.get("waiver_closed") is not True or t5.get("pilot_blocked") is not False:
        raise contract.ContractError("single retained T5 closure identity differs")
    if config.get("a1_artifact_seal_sha256") != contract.EXPECTED_A1_R2_ARTIFACT_SEAL_SHA256:
        raise contract.ContractError("single retained A1-R2 seal differs")
    if config.get("a2_artifact_seal_sha256") != contract.EXPECTED_A2_PROBE_SEAL_SHA256:
        raise contract.ContractError("single retained A2 seal differs")
    runtime = result.get("runtime")
    _exact_keys(
        runtime,
        (
            "python", "torch", "torch_cuda_runtime", "device", "gpu_name",
            "gpu_capability", "visible_cuda_devices", "scoring_autocast",
            "bf16_supported", "divergence_dtype",
        ),
        "single runtime",
    )
    if (
        not isinstance(runtime.get("python"), str)
        or not runtime.get("python")
        or runtime.get("torch") != "2.1.0+cu121"
        or runtime.get("torch_cuda_runtime") != "12.1"
        or runtime.get("device") != "cuda:0"
        or runtime.get("gpu_name") != "NVIDIA H20"
        or runtime.get("gpu_capability") != [9, 0]
        or runtime.get("visible_cuda_devices") != 1
        or runtime.get("bf16_supported") is not True
        or runtime.get("scoring_autocast") is not False
        or runtime.get("divergence_dtype") != "torch.float32"
    ):
        raise contract.ContractError("single runtime is not the frozen one-H20 contract")
    _validate_offline(result.get("offline_environment"), "single offline environment")
    loaded_t5 = result.get("loaded_t5_identity")
    if (
        not isinstance(loaded_t5, dict)
        or loaded_t5.get("identity_sha256")
        != cfg.get("loaded_t5_identity_sha256")
    ):
        raise contract.ContractError("single loaded T5 identity differs from CFG binding")

    audit_inputs = result.get("audit_inputs")
    _exact_keys(
        audit_inputs,
        ("format", "codes_key", "codes_shape", "codes_dtype", "sample_ids_key", "rollout_codes_sha256"),
        "single audit_inputs",
    )
    if audit_inputs != {
        "format": "numpy-npz-allow_pickle-false",
        "codes_key": "codes",
        "codes_shape": [8, 4, 500],
        "codes_dtype": "int64",
        "sample_ids_key": "sample_ids",
        "rollout_codes_sha256": audit_inputs.get("rollout_codes_sha256"),
    }:
        raise contract.ContractError("single audit input declaration differs")
    rollout_hash = _sha(audit_inputs.get("rollout_codes_sha256"), "rollout codes hash")
    sample_ids = _verify_npz(npz_path, rollout_hash)
    rows = result.get("audit_batch")
    if not isinstance(rows, list) or len(rows) != 8:
        raise contract.ContractError("single audit batch must contain eight rows")
    row_ids: List[str] = []
    for index, row in enumerate(rows):
        _exact_keys(row, ("sample_id", "prompt", "source_row_sha256"), "audit row {}".format(index))
        if not isinstance(row.get("sample_id"), str) or not isinstance(row.get("prompt"), str) or not row.get("prompt"):
            raise contract.ContractError("single audit row text fields are malformed")
        _sha(row.get("source_row_sha256"), "source row hash")
        row_ids.append(row["sample_id"])
    if row_ids != sample_ids:
        raise contract.ContractError("single audit JSON/NPZ sample IDs differ")

    cases = result.get("cases")
    if not isinstance(cases, dict) or tuple(cases) != contract.SINGLE_CASES:
        raise contract.ContractError("single case set/order differs")
    for name in contract.SINGLE_CASES:
        contract.require_pass_case(cases[name], name)
    b11 = cases[contract.SINGLE_CASES[0]]
    _threshold(b11.get("mean_kl_threshold_exclusive"), contract.CFG1_MEAN_KL_LT, "B1.1 mean threshold")
    _threshold(b11.get("max_kl_threshold_exclusive"), contract.CFG1_MAX_VALID_KL_LT, "B1.1 max threshold")
    _strict_lt(b11.get("mean_kl"), contract.CFG1_MEAN_KL_LT, "B1.1 mean KL")
    _strict_lt(b11.get("max_valid_cell_kl"), contract.CFG1_MAX_VALID_KL_LT, "B1.1 max KL")
    if b11.get("compute_dtype") != "torch.float32" or b11.get("valid_cell_count") != 3988:
        raise contract.ContractError("B1.1 real lattice/dtype differs")
    for case_name, gate_name in ((contract.SINGLE_CASES[1], "all_selected_gate_exact"), (contract.SINGLE_CASES[2], "gate_exact")):
        case = cases[case_name]
        _true(case.get(gate_name), case_name + " gate")
        _threshold(case.get("relative_loss_threshold_exclusive"), contract.RELATIVE_LOSS_ERROR_LT, case_name + " loss threshold")
        _threshold(case.get("relative_gradient_threshold_exclusive"), contract.RELATIVE_GRADIENT_ERROR_LT, case_name + " gradient threshold")
        _strict_lt(case.get("relative_loss_error"), contract.RELATIVE_LOSS_ERROR_LT, case_name + " loss error")
        _strict_lt(case.get("relative_gradient_l2_error"), contract.RELATIVE_GRADIENT_ERROR_LT, case_name + " gradient error")
    b14 = cases[contract.SINGLE_CASES[3]]
    if b14.get("expected_counts_per_sample_codebook") != [[250, 250, 249, 249], [250, 250, 249, 249]]:
        raise contract.ContractError("B1.4 selected counts differ")
    for name in ("random_counts_exact", "js_counts_exact", "repeated_js_gate_exact", "stable_zero_js_tie_gate_exact"):
        _true(b14.get(name), "B1.4 " + name)
    _sha(b14.get("js_gate_sha256"), "B1.4 JS gate hash")
    _sha(b14.get("random_gate_sha256"), "B1.4 random gate hash")
    b15 = cases[contract.SINGLE_CASES[4]]
    if b15.get("padding_lengths") != list(contract.SINGLE_PADDING_LENGTHS):
        raise contract.ContractError("B1.5 padding lengths differ")
    for name in (
        "valid_student_logits_exact_after_input_mutation", "valid_teacher_logits_exact_after_input_mutation",
        "loss_exact_after_input_mutation", "valid_gradient_exact_after_input_mutation",
        "loss_exact_with_invalid_nan_inf", "invalid_selected_exact_zero",
        "invalid_effective_weight_exact_zero", "invalid_logit_gradient_exact_zero",
    ):
        _true(b15.get(name), "B1.5 " + name)
    b16 = cases[contract.SINGLE_CASES[5]]
    if b16.get("valid_coordinate_count") != 1994 or not isinstance(b16.get("sequence_steps"), int) or b16.get("sequence_steps") <= 0:
        raise contract.ContractError("B1.6 real lattice differs")
    _true(b16.get("sentinel_qt_identity_exact"), "B1.6 sentinel identity")
    _true(b16.get("real_rollout_identity_exact"), "B1.6 rollout identity")
    _sha(b16.get("revert_indices_sha256"), "B1.6 revert index hash")
    _sha(b16.get("revert_mask_sha256"), "B1.6 revert mask hash")

    state_hashes = [
        _sha(result.get(name), name)
        for name in (
            "student_state_sha256_before", "student_state_sha256_after",
            "teacher_state_sha256_before", "teacher_state_sha256_after",
        )
    ]
    if len(set(state_hashes)) != 1:
        raise contract.ContractError("single real-checkpoint states changed or differed")
    _true(result.get("upstream_seals_before_after_equal"), "single upstream equality")
    before = _sha(result.get("upstream_identity_sha256_before"), "single upstream before hash")
    after = _sha(result.get("upstream_identity_sha256_after"), "single upstream after hash")
    if before != after:
        raise contract.ContractError("single upstream before/after hashes differ")
    return {
        "scientific_config_sha256": result["scientific_config_sha256"],
        "sample_ids": sample_ids,
        "rollout_codes_sha256": rollout_hash,
        "node3_identity": node3,
        "t5_closure_identity": t5,
    }


def verify_single(directory: Path) -> Dict[str, Any]:
    supplied = directory.expanduser().absolute()
    if supplied.is_symlink():
        raise contract.ContractError("single B1 artifact root must not be a symlink")
    root = supplied.resolve(strict=True)
    identity = contract.verify_artifact_directory(
        root, status=contract.SINGLE_STATUS, member_names=contract.SINGLE_MEMBERS
    )
    result = contract.load_json(root / "single_gpu_results.json", "single B1 result")
    payload = _validate_single_result(result, root / "audit_inputs.npz")
    return {"kind": "single", **identity, **payload}


def _validate_distributed_result(
    result: Mapping[str, Any], expected_single: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    _exact_keys(
        result,
        (
            "schema_version", "scientific_status", "gate_passed", "operationally_accepted",
            "redline_touched", "full_b1_passed", "pending_b1_items", "scientific_config",
            "scientific_config_sha256", "runtime", "offline_environment", "rank_evidence",
            "ddp_gradient_hash_exact_across_ranks", "ddp_gradient_layout_sha256",
            "ddp_global_loss", "wrong_mean_of_local_ratios", "wrong_mean_absolute_difference",
            "wrong_mean_sensitivity_threshold_exclusive", "relative_loss_error",
            "relative_gradient_l2_error", "maximum_per_tensor_relative_gradient_l2_error",
            "relative_loss_threshold_exclusive", "relative_gradient_threshold_exclusive",
            "concatenated_reference", "upstream_identities_postflight_equal",
        ),
        "distributed result",
    )
    if result.get("schema_version") != contract.DISTRIBUTED_SCHEMA or result.get("scientific_status") != contract.DISTRIBUTED_STATUS:
        raise contract.ContractError("distributed schema/status differs")
    _true(result.get("gate_passed"), "distributed gate_passed")
    _true(result.get("operationally_accepted"), "distributed operationally_accepted")
    _false(result.get("redline_touched"), "distributed redline_touched")
    _false(result.get("full_b1_passed"), "distributed full_b1_passed")
    _true(
        result.get("upstream_identities_postflight_equal"),
        "distributed upstream postflight equality",
    )
    if result.get("pending_b1_items") != [contract.PENDING_B1_ITEM]:
        raise contract.ContractError("distributed pending B1 items differ")
    config = result.get("scientific_config")
    if not isinstance(config, dict) or config.get("schema_version") != "ptc-opd-b1-ddp-scientific-config-v1":
        raise contract.ContractError("distributed scientific config schema differs")
    config_hash = _sha(result.get("scientific_config_sha256"), "distributed config hash")
    if contract.canonical_json_sha256(config) != config_hash:
        raise contract.ContractError("distributed scientific config hash differs")
    fixed = {
        "model_id": "facebook/musicgen-small", "world_size": 8, "rank_batch_size": 1,
        "loss_mode": "uniform", "teacher_cfg_scale": 5.0,
        "scoring_compute_dtype": "torch.float32", "divergence_dtype": "torch.float32",
        "padding_lengths_by_rank": list(contract.DISTRIBUTED_PADDING_LENGTHS),
        "ddp_policy": {
            "bucket_cap_mb": 25, "find_unused_parameters": True,
            "static_graph": False, "gradient_as_bucket_view": False,
        },
    }
    for name, expected in fixed.items():
        if config.get(name) != expected:
            raise contract.ContractError("distributed config differs at {}".format(name))
    thresholds = config.get("thresholds")
    _exact_keys(thresholds, ("relative_loss_error_lt", "relative_gradient_error_lt", "mean_of_means_sensitivity_gt"), "distributed thresholds")
    _threshold(thresholds["relative_loss_error_lt"], contract.RELATIVE_LOSS_ERROR_LT, "distributed loss threshold")
    _threshold(thresholds["relative_gradient_error_lt"], contract.RELATIVE_GRADIENT_ERROR_LT, "distributed gradient threshold")
    _threshold(thresholds["mean_of_means_sensitivity_gt"], contract.MEAN_OF_MEANS_SENSITIVITY_GT, "distributed sensitivity threshold")
    if expected_single is not None and config.get("single_artifact_seal_sha256") != expected_single.get("seal_sha256"):
        raise contract.ContractError("distributed input is not the supplied single artifact")
    runtime = result.get("runtime")
    _exact_keys(
        runtime,
        (
            "torch", "torch_cuda_runtime", "backend", "launcher", "world_size",
            "local_world_size", "gpus", "deterministic_algorithms",
            "cublas_workspace_config", "matmul_tf32", "cudnn_tf32",
            "scoring_autocast",
        ),
        "distributed runtime",
    )
    expected_gpus = [
        {"index": index, "name": "NVIDIA H20", "capability": [9, 0]}
        for index in range(8)
    ]
    if (
        runtime.get("torch") != "2.1.0+cu121"
        or runtime.get("torch_cuda_runtime") != "12.1"
        or runtime.get("backend") != "nccl"
        or runtime.get("launcher") != "torchrun_standalone"
        or runtime.get("world_size") != 8
        or runtime.get("local_world_size") != 8
        or runtime.get("gpus") != expected_gpus
        or runtime.get("deterministic_algorithms") is not True
        or runtime.get("cublas_workspace_config") != ":4096:8"
        or runtime.get("matmul_tf32") is not False
        or runtime.get("cudnn_tf32") is not False
        or runtime.get("scoring_autocast") is not False
    ):
        raise contract.ContractError("distributed runtime is not the frozen eight-H20 contract")
    _validate_offline(result.get("offline_environment"), "distributed offline environment")

    _true(result.get("ddp_gradient_hash_exact_across_ranks"), "distributed exact gradient hashes")
    gradient_hash = _sha(result.get("ddp_gradient_layout_sha256"), "distributed gradient hash")
    _threshold(result.get("relative_loss_threshold_exclusive"), contract.RELATIVE_LOSS_ERROR_LT, "distributed result loss threshold")
    _threshold(result.get("relative_gradient_threshold_exclusive"), contract.RELATIVE_GRADIENT_ERROR_LT, "distributed result gradient threshold")
    loss_error = _strict_lt(result.get("relative_loss_error"), contract.RELATIVE_LOSS_ERROR_LT, "distributed loss error")
    gradient_error = _strict_lt(result.get("relative_gradient_l2_error"), contract.RELATIVE_GRADIENT_ERROR_LT, "distributed gradient error")
    maximum_error = contract.require_finite_number(
        result.get("maximum_per_tensor_relative_gradient_l2_error"),
        "distributed maximum tensor gradient diagnostic",
    )
    _threshold(result.get("wrong_mean_sensitivity_threshold_exclusive"), contract.MEAN_OF_MEANS_SENSITIVITY_GT, "distributed result sensitivity threshold")
    sensitivity = _strict_gt(result.get("wrong_mean_absolute_difference"), contract.MEAN_OF_MEANS_SENSITIVITY_GT, "distributed wrong-mean sensitivity")

    rows = result.get("rank_evidence")
    if not isinstance(rows, list) or len(rows) != 8:
        raise contract.ContractError("distributed rank evidence must contain eight rows")
    local_ratios: List[float] = []
    numerators: List[float] = []
    denominators: List[float] = []
    sample_ids: List[str] = []
    global_losses: List[float] = []
    global_numerators: List[float] = []
    global_denominators: List[float] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("rank") != index or row.get("local_rank") != index:
            raise contract.ContractError("distributed rank order/identity differs")
        if row.get("padding_length") != contract.DISTRIBUTED_PADDING_LENGTHS[index]:
            raise contract.ContractError("distributed padding length differs")
        if not isinstance(row.get("valid_cell_count"), int) or row["valid_cell_count"] <= 0:
            raise contract.ContractError("distributed valid cell count is invalid")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise contract.ContractError("distributed sample ID is invalid")
        sample_ids.append(sample_id)
        numerator = contract.require_finite_number(row.get("local_numerator"), "local numerator")
        denominator = _strict_gt(row.get("local_denominator"), 0.0, "local denominator")
        ratio = contract.require_finite_number(row.get("local_ratio"), "local ratio")
        if not math.isclose(ratio, numerator / denominator, rel_tol=1.0e-7, abs_tol=1.0e-9):
            raise contract.ContractError("local ratio is not numerator/denominator")
        numerators.append(numerator)
        denominators.append(denominator)
        local_ratios.append(ratio)
        global_losses.append(contract.require_finite_number(row.get("global_loss"), "rank global loss"))
        global_numerators.append(contract.require_finite_number(row.get("global_numerator"), "rank global numerator"))
        global_denominators.append(_strict_gt(row.get("global_denominator"), 0.0, "rank global denominator"))
        gradient = row.get("gradient_identity")
        if not isinstance(gradient, dict) or gradient.get("gradient_layout_sha256") != gradient_hash:
            raise contract.ContractError("rank synchronized gradient identity differs")
        if not isinstance(gradient.get("trainable_parameter_tensor_count"), int) or gradient["trainable_parameter_tensor_count"] <= 0:
            raise contract.ContractError("rank trainable parameter count is invalid")
        if not isinstance(gradient.get("trainable_parameter_numel"), int) or gradient["trainable_parameter_numel"] <= 0:
            raise contract.ContractError("rank trainable parameter numel is invalid")
        if row.get("teacher_state_sha256_before") != row.get("teacher_state_sha256_after"):
            raise contract.ContractError("rank teacher state changed")
        if row.get("student_state_sha256_before") != row.get("student_state_sha256_after"):
            raise contract.ContractError("rank student state changed")
        _sha(row.get("teacher_state_sha256_before"), "rank teacher hash")
        _sha(row.get("student_state_sha256_before"), "rank student hash")
        reducer = row.get("ddp_reducer")
        if not isinstance(reducer, dict):
            raise contract.ContractError("rank DDP reducer audit is missing")
        reducer_identity = _sha(reducer.get("identity_sha256"), "DDP reducer identity")
        if contract.canonical_json_sha256({key: value for key, value in reducer.items() if key != "identity_sha256"}) != reducer_identity:
            raise contract.ContractError("DDP reducer identity hash differs")
    if len(set(sample_ids)) != 8:
        raise contract.ContractError("distributed sample IDs are not unique")
    if expected_single is not None and sample_ids != expected_single.get("sample_ids"):
        raise contract.ContractError("distributed/single sample IDs differ")
    if len(set(global_losses)) != 1 or len(set(global_numerators)) != 1 or len(set(global_denominators)) != 1:
        raise contract.ContractError("global ratio values differ across ranks")
    global_loss = contract.require_finite_number(result.get("ddp_global_loss"), "DDP global loss")
    if global_loss != global_losses[0]:
        raise contract.ContractError("outer/rank DDP global losses differ")
    if not math.isclose(sum(denominators), global_denominators[0], rel_tol=0.0, abs_tol=1.0e-7):
        raise contract.ContractError("global denominator is not the rank sum")
    if not math.isclose(sum(numerators), global_numerators[0], rel_tol=1.0e-6, abs_tol=1.0e-7):
        raise contract.ContractError("global numerator is not the rank sum")
    if not math.isclose(global_loss, global_numerators[0] / global_denominators[0], rel_tol=1.0e-7, abs_tol=1.0e-9):
        raise contract.ContractError("DDP global loss is not the global ratio")
    wrong_mean = sum(local_ratios) / 8.0
    if not math.isclose(wrong_mean, contract.require_finite_number(result.get("wrong_mean_of_local_ratios"), "wrong mean"), rel_tol=0.0, abs_tol=1.0e-15):
        raise contract.ContractError("recorded wrong mean differs")
    if not math.isclose(abs(wrong_mean - global_loss), sensitivity, rel_tol=0.0, abs_tol=1.0e-15):
        raise contract.ContractError("recorded wrong-mean sensitivity differs")

    reference = result.get("concatenated_reference")
    _exact_keys(
        reference,
        (
            "concatenated_reference_loss", "concatenated_reference_numerator",
            "concatenated_reference_denominator", "concatenated_reference_valid_counts",
            "global_relative_gradient_l2_error", "maximum_per_tensor_relative_gradient_l2_error",
            "maximum_error_parameter", "reference_gradient_layout_sha256",
        ),
        "concatenated reference",
    )
    reference_loss = contract.require_finite_number(reference.get("concatenated_reference_loss"), "reference loss")
    reference_numerator = contract.require_finite_number(reference.get("concatenated_reference_numerator"), "reference numerator")
    reference_denominator = _strict_gt(reference.get("concatenated_reference_denominator"), 0.0, "reference denominator")
    if reference_denominator != global_denominators[0] or not math.isclose(reference_loss, reference_numerator / reference_denominator, rel_tol=1.0e-7, abs_tol=1.0e-9):
        raise contract.ContractError("concatenated reference ratio differs")
    if reference.get("concatenated_reference_valid_counts") != [row["valid_cell_count"] for row in rows]:
        raise contract.ContractError("concatenated reference valid counts differ")
    if reference.get("global_relative_gradient_l2_error") != gradient_error or reference.get("maximum_per_tensor_relative_gradient_l2_error") != maximum_error:
        raise contract.ContractError("outer/reference gradient errors differ")
    _sha(reference.get("reference_gradient_layout_sha256"), "reference gradient hash")
    recomputed_loss_error = abs(global_loss - reference_loss) / max(abs(reference_loss), 1.0e-12)
    if not math.isclose(loss_error, recomputed_loss_error, rel_tol=0.0, abs_tol=1.0e-18):
        raise contract.ContractError("recorded DDP/reference loss error differs")
    return {
        "scientific_config_sha256": config_hash,
        "sample_ids": sample_ids,
        "single_artifact_seal_sha256": config.get("single_artifact_seal_sha256"),
    }


def verify_distributed(directory: Path) -> Dict[str, Any]:
    supplied = directory.expanduser().absolute()
    if supplied.is_symlink():
        raise contract.ContractError("distributed B1 artifact root must not be a symlink")
    root = supplied.resolve(strict=True)
    identity = contract.verify_artifact_directory(
        root, status=contract.DISTRIBUTED_STATUS, member_names=contract.DISTRIBUTED_MEMBERS
    )
    result = contract.load_json(root / "distributed_results.json", "distributed B1 result")
    payload = _validate_distributed_result(result)
    return {"kind": "distributed", **identity, **payload}


def verify_final(directory: Path) -> Dict[str, Any]:
    supplied = directory.expanduser().absolute()
    if supplied.is_symlink():
        raise contract.ContractError("final B1 artifact root must not be a symlink")
    root = supplied.resolve(strict=True)
    identity = contract.verify_artifact_directory(
        root, status=contract.SUMMARY_STATUS, member_names=contract.FINAL_MEMBERS
    )
    single_seal_sha = _verify_payload_seal(
        root / "single_gpu_artifact_seal.json",
        status=contract.SINGLE_STATUS,
        payload_paths={
            "audit_inputs.npz": root / "audit_inputs.npz",
            "single_gpu_results.json": root / "single_gpu_results.json",
        },
    )
    single_result = contract.load_json(root / "single_gpu_results.json", "final single result")
    single_payload = _validate_single_result(single_result, root / "audit_inputs.npz")
    single_payload["seal_sha256"] = single_seal_sha
    distributed_seal_sha = _verify_payload_seal(
        root / "distributed_artifact_seal.json",
        status=contract.DISTRIBUTED_STATUS,
        payload_paths={"distributed_results.json": root / "distributed_results.json"},
    )
    distributed_result = contract.load_json(root / "distributed_results.json", "final distributed result")
    distributed_payload = _validate_distributed_result(distributed_result, single_payload)

    summary = contract.load_json(root / "b1_prestability_summary.json", "B1 pre-stability summary")
    _exact_keys(
        summary,
        (
            "schema_version", "scientific_status", "gate_passed", "operationally_accepted",
            "redline_touched", "prestability_gate_passed", "full_b1_passed",
            "item_status", "completed_b1_items", "pending_b1_items", "evidence",
            "next_authorized_stage", "authorization_scope",
        ),
        "B1 pre-stability summary",
    )
    if summary.get("schema_version") != contract.SUMMARY_SCHEMA or summary.get("scientific_status") != contract.SUMMARY_STATUS:
        raise contract.ContractError("B1 summary schema/status differs")
    _true(summary.get("gate_passed"), "summary gate_passed")
    _true(summary.get("operationally_accepted"), "summary operationally_accepted")
    _false(summary.get("redline_touched"), "summary redline_touched")
    _true(summary.get("prestability_gate_passed"), "summary prestability gate")
    _false(summary.get("full_b1_passed"), "summary full_b1_passed")
    expected_completed = list(contract.SINGLE_CASES) + list(contract.RETAINED_NODE3_ITEMS) + [_B19_ITEM]
    if summary.get("completed_b1_items") != expected_completed or summary.get("pending_b1_items") != [contract.PENDING_B1_ITEM]:
        raise contract.ContractError("B1 summary completed/pending item partition differs")
    item_status = summary.get("item_status")
    if not isinstance(item_status, dict) or set(item_status) != set(expected_completed + [contract.PENDING_B1_ITEM]):
        raise contract.ContractError("B1 summary item-status set differs")
    for name in contract.SINGLE_CASES:
        if item_status.get(name) != {"status": "passed", "source": "single_gpu_real_checkpoint"}:
            raise contract.ContractError("B1 single item summary differs")
    for name in contract.RETAINED_NODE3_ITEMS:
        if item_status.get(name) != {"status": "passed_retained", "source": "node3_15_of_15"}:
            raise contract.ContractError("B1 retained node-3 item summary differs")
    if item_status.get(_B19_ITEM) != {"status": "passed", "source": "eight_gpu_real_checkpoint"}:
        raise contract.ContractError("B1.9 summary differs")
    if item_status.get(contract.PENDING_B1_ITEM) != {"status": "pending", "source": None}:
        raise contract.ContractError("B1.11 summary must remain pending")
    evidence = summary.get("evidence")
    _exact_keys(
        evidence,
        (
            "single_artifact_seal_sha256", "single_scientific_config_sha256",
            "distributed_artifact_seal_sha256", "distributed_scientific_config_sha256",
            "node3_status_sha256", "t5_closure_artifact_seal_sha256",
        ),
        "B1 summary evidence",
    )
    expected_evidence = {
        "single_artifact_seal_sha256": single_seal_sha,
        "single_scientific_config_sha256": single_payload["scientific_config_sha256"],
        "distributed_artifact_seal_sha256": distributed_seal_sha,
        "distributed_scientific_config_sha256": distributed_payload["scientific_config_sha256"],
        "node3_status_sha256": single_payload["node3_identity"].get("status_sha256"),
        "t5_closure_artifact_seal_sha256": contract.EXPECTED_T5_CLOSURE_SEAL_SHA256,
    }
    if evidence != expected_evidence:
        raise contract.ContractError("B1 summary evidence identity differs")
    if summary.get("next_authorized_stage") != "performance_benchmark_then_uniform_lr_sweep":
        raise contract.ContractError("B1 summary next-stage authorization differs")
    if summary.get("authorization_scope") != "pre_stability_only_b1_11_remains_mandatory":
        raise contract.ContractError("B1 summary authorization scope differs")
    return {
        "kind": "final",
        **identity,
        "summary_sha256": contract.sha256_file(root / "b1_prestability_summary.json"),
        "single_artifact_seal_sha256": single_seal_sha,
        "distributed_artifact_seal_sha256": distributed_seal_sha,
        "prestability_gate_passed": True,
        "full_b1_passed": False,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("single", "distributed", "final"))
    parser.add_argument("artifact_dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    verifier = {
        "single": verify_single,
        "distributed": verify_distributed,
        "final": verify_final,
    }[args.kind]
    result = verifier(args.artifact_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
