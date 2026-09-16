"""Pure-stdlib producer/consumer tests for the B1 pre-stability contract."""

from __future__ import annotations

import hashlib
import ast
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import b1_prestability_contract as contract
import finalize_b1_prestability as finalizer
import verify_b1_prestability as verifier


H = "1" * 64


def _npy_blob(descriptor, shape, raw):
    shape_text = repr(tuple(shape))
    header_text = "{'descr': %r, 'fortran_order': False, 'shape': %s, }" % (
        descriptor,
        shape_text,
    )
    prefix_length = 10
    padding = (16 - ((prefix_length + len(header_text) + 1) % 16)) % 16
    header = (header_text + " " * padding + "\n").encode("latin1")
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + raw


def _tensor_hash(raw):
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


def _write_npz(path):
    values = [index % 2048 for index in range(8 * 4 * 500)]
    codes_raw = b"".join(struct.pack("<q", value) for value in values)
    sample_ids = ["s{}".format(index) for index in range(8)]
    ids_raw = b"".join(value.ljust(2, "\x00").encode("utf-32-le") for value in sample_ids)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("codes.npy", _npy_blob("<i8", (8, 4, 500), codes_raw))
        archive.writestr("sample_ids.npy", _npy_blob("<U2", (8,), ids_raw))
    return sample_ids, _tensor_hash(codes_raw)


def _single_result(sample_ids, rollout_hash):
    thresholds = {
        "cfg1_mean_kl_lt": contract.CFG1_MEAN_KL_LT,
        "cfg1_max_valid_kl_lt": contract.CFG1_MAX_VALID_KL_LT,
        "relative_loss_error_lt": contract.RELATIVE_LOSS_ERROR_LT,
        "relative_gradient_error_lt": contract.RELATIVE_GRADIENT_ERROR_LT,
    }
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
        "thresholds": thresholds,
        "checkpoint_identity": {"checkpoint_sha256": H},
        "audiocraft_identity": {"audiocraft_source_sha256": H},
        "probe_manifest_sha256": H,
        "dev_manifest_sha256": H,
        "cfg_decision": {
            "selected_cfg_scale": 5.0,
            "decision_file_sha256": contract.EXPECTED_SMALL_CFG_DECISION_SHA256,
            "decision_payload_sha256": H,
            "scientific_config_sha256": H,
            "loaded_t5_identity_sha256": H,
        },
        "a1_identity_sha256": H,
        "a1_artifact_seal_sha256": contract.EXPECTED_A1_R2_ARTIFACT_SEAL_SHA256,
        "a2_artifact_seal_sha256": contract.EXPECTED_A2_PROBE_SEAL_SHA256,
        "a2_scientific_config_sha256": H,
        "node3_identity": {
            "gate_count": 15,
            "passed_count": 15,
            "status_sha256": H,
        },
        "t5_closure_identity": {
            "artifact_seal_sha256": contract.EXPECTED_T5_CLOSURE_SEAL_SHA256,
            "waiver_closed": True,
            "pilot_blocked": False,
        },
    }
    cases = {
        contract.SINGLE_CASES[0]: {
            "status": "passed", "compute_dtype": "torch.float32",
            "valid_cell_count": 3988, "mean_kl": 0.0, "max_valid_cell_kl": 0.0,
            "mean_kl_threshold_exclusive": contract.CFG1_MEAN_KL_LT,
            "max_kl_threshold_exclusive": contract.CFG1_MAX_VALID_KL_LT,
        },
        contract.SINGLE_CASES[1]: {
            "status": "passed", "all_selected_gate_exact": True,
            "relative_loss_error": 0.0, "relative_gradient_l2_error": 0.0,
            "relative_loss_threshold_exclusive": contract.RELATIVE_LOSS_ERROR_LT,
            "relative_gradient_threshold_exclusive": contract.RELATIVE_GRADIENT_ERROR_LT,
        },
        contract.SINGLE_CASES[2]: {
            "status": "passed", "gate_exact": True,
            "relative_loss_error": 0.0, "relative_gradient_l2_error": 0.0,
            "relative_loss_threshold_exclusive": contract.RELATIVE_LOSS_ERROR_LT,
            "relative_gradient_threshold_exclusive": contract.RELATIVE_GRADIENT_ERROR_LT,
        },
        contract.SINGLE_CASES[3]: {
            "status": "passed",
            "expected_counts_per_sample_codebook": [[250, 250, 249, 249], [250, 250, 249, 249]],
            "random_counts_exact": True, "js_counts_exact": True,
            "repeated_js_gate_exact": True, "stable_zero_js_tie_gate_exact": True,
            "js_gate_sha256": H, "random_gate_sha256": H,
        },
        contract.SINGLE_CASES[4]: {
            "status": "passed", "padding_lengths": list(contract.SINGLE_PADDING_LENGTHS),
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
            "status": "passed", "pattern_provider_type": "DelayedPatternProvider",
            "sequence_steps": 503, "valid_coordinate_count": 1994,
            "sentinel_qt_identity_exact": True, "real_rollout_identity_exact": True,
            "revert_indices_sha256": H, "revert_mask_sha256": H,
        },
    }
    return {
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
        "runtime": {
            "python": "3.9.test", "torch": "2.1.0+cu121",
            "torch_cuda_runtime": "12.1", "device": "cuda:0",
            "gpu_name": "NVIDIA H20", "gpu_capability": [9, 0],
            "visible_cuda_devices": 1, "bf16_supported": True,
            "scoring_autocast": False,
            "divergence_dtype": "torch.float32",
        },
        "offline_environment": {
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        },
        "audit_batch": [
            {"sample_id": value, "prompt": "prompt " + value, "source_row_sha256": H}
            for value in sample_ids
        ],
        "audit_inputs": {
            "format": "numpy-npz-allow_pickle-false", "codes_key": "codes",
            "codes_shape": [8, 4, 500], "codes_dtype": "int64",
            "sample_ids_key": "sample_ids", "rollout_codes_sha256": rollout_hash,
        },
        "loaded_t5_identity": {"identity_sha256": H},
        "student_state_sha256_before": H,
        "student_state_sha256_after": H,
        "teacher_state_sha256_before": H,
        "teacher_state_sha256_after": H,
        "cases": cases,
        "diagnostics": {},
        "upstream_seals_before_after_equal": True,
        "upstream_identity_sha256_before": H,
        "upstream_identity_sha256_after": H,
    }


def _write_single(root):
    root.mkdir()
    sample_ids, rollout_hash = _write_npz(root / "audit_inputs.npz")
    contract.write_json_exclusive(
        root / "single_gpu_results.json", _single_result(sample_ids, rollout_hash)
    )
    contract.write_artifact_seal(
        root, status=contract.SINGLE_STATUS, member_names=contract.SINGLE_MEMBERS
    )
    return sample_ids


def _distributed_result(sample_ids, single_seal):
    denominators = [float(index + 1) for index in range(8)]
    ratios = [float(index + 1) for index in range(8)]
    numerators = [left * right for left, right in zip(denominators, ratios)]
    global_denominator = sum(denominators)
    global_numerator = sum(numerators)
    global_loss = global_numerator / global_denominator
    wrong_mean = sum(ratios) / 8.0
    reducer = {"policy": "frozen"}
    reducer["identity_sha256"] = contract.canonical_json_sha256(reducer)
    rows = []
    for index in range(8):
        rows.append(
            {
                "rank": index, "local_rank": index, "sample_id": sample_ids[index],
                "padding_length": contract.DISTRIBUTED_PADDING_LENGTHS[index],
                "valid_cell_count": 1000 + index,
                "local_numerator": numerators[index],
                "local_denominator": denominators[index],
                "local_ratio": ratios[index],
                "global_loss": global_loss,
                "global_numerator": global_numerator,
                "global_denominator": global_denominator,
                "gradient_identity": {
                    "trainable_parameter_tensor_count": 2,
                    "trainable_parameter_numel": 10,
                    "gradient_layout_sha256": H,
                },
                "teacher_state_sha256_before": H,
                "teacher_state_sha256_after": H,
                "student_state_sha256_before": H,
                "student_state_sha256_after": H,
                "ddp_reducer": reducer,
            }
        )
    scientific_config = {
        "schema_version": "ptc-opd-b1-ddp-scientific-config-v1",
        "model_id": "facebook/musicgen-small", "world_size": 8,
        "rank_batch_size": 1, "loss_mode": "uniform", "teacher_cfg_scale": 5.0,
        "scoring_compute_dtype": "torch.float32", "divergence_dtype": "torch.float32",
        "padding_lengths_by_rank": list(contract.DISTRIBUTED_PADDING_LENGTHS),
        "thresholds": {
            "relative_loss_error_lt": contract.RELATIVE_LOSS_ERROR_LT,
            "relative_gradient_error_lt": contract.RELATIVE_GRADIENT_ERROR_LT,
            "mean_of_means_sensitivity_gt": contract.MEAN_OF_MEANS_SENSITIVITY_GT,
        },
        "ddp_policy": {
            "bucket_cap_mb": 25, "find_unused_parameters": True,
            "static_graph": False, "gradient_as_bucket_view": False,
        },
        "single_artifact_seal_sha256": single_seal,
        "checkpoint_identity": {"checkpoint_sha256": H},
        "audiocraft_identity": {"audiocraft_source_sha256": H},
    }
    return {
        "schema_version": contract.DISTRIBUTED_SCHEMA,
        "scientific_status": contract.DISTRIBUTED_STATUS,
        "gate_passed": True, "operationally_accepted": True,
        "redline_touched": False, "full_b1_passed": False,
        "pending_b1_items": [contract.PENDING_B1_ITEM],
        "scientific_config": scientific_config,
        "scientific_config_sha256": contract.canonical_json_sha256(scientific_config),
        "runtime": {
            "torch": "2.1.0+cu121", "torch_cuda_runtime": "12.1",
            "backend": "nccl", "launcher": "torchrun_standalone",
            "world_size": 8, "local_world_size": 8,
            "gpus": [
                {"index": index, "name": "NVIDIA H20", "capability": [9, 0]}
                for index in range(8)
            ],
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
            "matmul_tf32": False, "cudnn_tf32": False,
            "scoring_autocast": False,
        },
        "offline_environment": {
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        },
        "rank_evidence": rows,
        "ddp_gradient_hash_exact_across_ranks": True,
        "ddp_gradient_layout_sha256": H,
        "ddp_global_loss": global_loss,
        "wrong_mean_of_local_ratios": wrong_mean,
        "wrong_mean_absolute_difference": abs(wrong_mean - global_loss),
        "wrong_mean_sensitivity_threshold_exclusive": contract.MEAN_OF_MEANS_SENSITIVITY_GT,
        "relative_loss_error": 0.0, "relative_gradient_l2_error": 0.0,
        "maximum_per_tensor_relative_gradient_l2_error": 0.0,
        "relative_loss_threshold_exclusive": contract.RELATIVE_LOSS_ERROR_LT,
        "relative_gradient_threshold_exclusive": contract.RELATIVE_GRADIENT_ERROR_LT,
        "concatenated_reference": {
            "concatenated_reference_loss": global_loss,
            "concatenated_reference_numerator": global_numerator,
            "concatenated_reference_denominator": global_denominator,
            "concatenated_reference_valid_counts": [1000 + index for index in range(8)],
            "global_relative_gradient_l2_error": 0.0,
            "maximum_per_tensor_relative_gradient_l2_error": 0.0,
            "maximum_error_parameter": None,
            "reference_gradient_layout_sha256": H,
        },
        "upstream_identities_postflight_equal": True,
    }


def _write_distributed(root, sample_ids, single_seal):
    root.mkdir()
    contract.write_json_exclusive(
        root / "distributed_results.json",
        _distributed_result(sample_ids, single_seal),
    )
    contract.write_artifact_seal(
        root,
        status=contract.DISTRIBUTED_STATUS,
        member_names=contract.DISTRIBUTED_MEMBERS,
    )


def _write_t5_closure(root):
    root.mkdir()
    config = {"schema_version": "test-config-v1"}
    config_hash = contract.canonical_json_sha256(config)
    decision = {
        "waiver_closed": True,
        "pilot_blocked": False,
        "preserve_a1": True,
        "preserve_cfg_a2_a3": True,
        "required_action": "none",
    }
    result = {
        "schema_version": "ptc-opd-t5-tokenization-equivalence-v3",
        "scientific_status": "equivalent_all_inputs",
        "primary_contract_passed": True,
        "scientific_config": config,
        "scientific_config_sha256": config_hash,
        "decision": decision,
    }
    contract.write_json_exclusive(root / "t5_tokenization_equivalence.json", result)
    status = {
        "schema_version": "ptc-opd-t5-tokenization-equivalence-status-v3",
        "status": "equivalent_all_inputs",
        "waiver_closed": True,
        "pilot_blocked": False,
        "required_action": "none",
        "result_sha256": contract.sha256_file(root / "t5_tokenization_equivalence.json"),
    }
    contract.write_json_exclusive(root / "STATUS.json", status)
    contract.write_json_exclusive(root / "environment.json", {"status": "complete"})
    (root / "per_case.jsonl.gz").write_bytes(b"test-gzip-placeholder")
    member_names = (
        "STATUS.json", "t5_tokenization_equivalence.json",
        "per_case.jsonl.gz", "environment.json",
    )
    seal = {
        "schema_version": "ptc-opd-t5-tokenization-equivalence-seal-v3",
        "status": "equivalent_all_inputs",
        "scientific_config_sha256": config_hash,
        "members": {
            name: contract.regular_file_identity(root / name)
            for name in member_names
        },
    }
    contract.write_json_exclusive(root / "artifact_seal.json", seal)
    checksum_names = sorted(member_names + ("artifact_seal.json",))
    (root / "SHA256SUMS.txt").write_text(
        "".join(
            "{}  {}\n".format(contract.sha256_file(root / name), name)
            for name in checksum_names
        ),
        encoding="utf-8",
    )
    (root / "SHA256SUMS.txt.sha256").write_text(
        "{}  SHA256SUMS.txt\n".format(contract.sha256_file(root / "SHA256SUMS.txt")),
        encoding="utf-8",
    )
    return contract.sha256_file(root / "artifact_seal.json")


def _rewrite_and_reseal(root, result_name, result, status, members):
    (root / result_name).unlink()
    (root / "artifact_seal.json").unlink()
    contract.write_json_exclusive(root / result_name, result)
    contract.write_artifact_seal(root, status=status, member_names=members)


class B1ArtifactContractTest(unittest.TestCase):
    def _fixtures(self, temporary):
        root = Path(temporary)
        single = root / "single"
        distributed = root / "distributed"
        sample_ids = _write_single(single)
        single_identity = verifier.verify_single(single)
        _write_distributed(distributed, sample_ids, single_identity["seal_sha256"])
        return single, distributed

    def test_single_distributed_and_final_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            single, distributed = self._fixtures(temporary)
            self.assertEqual(verifier.verify_single(single)["kind"], "single")
            self.assertEqual(verifier.verify_distributed(distributed)["kind"], "distributed")
            final = Path(temporary) / "final"
            finalizer.execute(
                argparse_namespace(single, distributed, final)
            )
            verified = verifier.verify_final(final)
            self.assertTrue(verified["prestability_gate_passed"])
            self.assertFalse(verified["full_b1_passed"])

    def test_closed_world_rejects_extra_member(self):
        with tempfile.TemporaryDirectory() as temporary:
            single, _ = self._fixtures(temporary)
            (single / "unexpected.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(contract.ContractError):
                verifier.verify_single(single)

    def test_single_cannot_claim_full_b1(self):
        with tempfile.TemporaryDirectory() as temporary:
            single, _ = self._fixtures(temporary)
            result = contract.load_json(single / "single_gpu_results.json")
            result["full_b1_passed"] = True
            _rewrite_and_reseal(
                single, "single_gpu_results.json", result,
                contract.SINGLE_STATUS, contract.SINGLE_MEMBERS,
            )
            with self.assertRaises(contract.ContractError):
                verifier.verify_single(single)

    def test_exclusive_threshold_rejects_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            single, _ = self._fixtures(temporary)
            result = contract.load_json(single / "single_gpu_results.json")
            result["cases"][contract.SINGLE_CASES[0]]["mean_kl"] = contract.CFG1_MEAN_KL_LT
            _rewrite_and_reseal(
                single, "single_gpu_results.json", result,
                contract.SINGLE_STATUS, contract.SINGLE_MEMBERS,
            )
            with self.assertRaises(contract.ContractError):
                verifier.verify_single(single)

    def test_b1_verifiers_reject_symlink_artifact_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            single, distributed = self._fixtures(temporary)
            single_link = Path(temporary) / "single-link"
            distributed_link = Path(temporary) / "distributed-link"
            single_link.symlink_to(single, target_is_directory=True)
            distributed_link.symlink_to(distributed, target_is_directory=True)
            with self.assertRaisesRegex(contract.ContractError, "must not be a symlink"):
                verifier.verify_single(single_link)
            with self.assertRaisesRegex(contract.ContractError, "must not be a symlink"):
                verifier.verify_distributed(distributed_link)
            with self.assertRaisesRegex(contract.ContractError, "must not be symlinks"):
                finalizer.execute(
                    argparse_namespace(
                        single_link,
                        distributed,
                        Path(temporary) / "unused-final",
                    )
                )

    def test_finalizer_rejects_unrelated_distributed_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            single, distributed = self._fixtures(temporary)
            result = contract.load_json(distributed / "distributed_results.json")
            result["scientific_config"]["single_artifact_seal_sha256"] = "2" * 64
            result["scientific_config_sha256"] = contract.canonical_json_sha256(result["scientific_config"])
            _rewrite_and_reseal(
                distributed, "distributed_results.json", result,
                contract.DISTRIBUTED_STATUS, contract.DISTRIBUTED_MEMBERS,
            )
            with self.assertRaises(contract.ContractError):
                finalizer.execute(argparse_namespace(single, distributed, Path(temporary) / "final"))

    def test_runners_expose_no_scientific_threshold_cli(self):
        for filename in ("run_b1_single_gpu_audit.py", "run_b1_distributed_audit.py"):
            source = (SCRIPTS / filename).read_text(encoding="utf-8")
            tree = ast.parse(source)
            flags = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr != "add_argument" or not node.args:
                    continue
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    flags.append(first.value)
            forbidden = {
                "--threshold", "--rho", "--seed", "--teacher-cfg-scale",
                "--padding", "--world-size", "--temperature", "--top-k", "--top-p",
            }
            self.assertFalse(any(flag in forbidden for flag in flags), (filename, flags))

    def test_producers_cannot_emit_full_b1_pass(self):
        for filename in (
            "run_b1_single_gpu_audit.py",
            "run_b1_distributed_audit.py",
            "finalize_b1_prestability.py",
        ):
            source = (SCRIPTS / filename).read_text(encoding="utf-8")
            self.assertNotIn('"full_b1_passed": True', source)
            self.assertIn('"full_b1_passed": False', source)

    def test_t5_closure_consumer_rehashes_closed_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "t5"
            seal_hash = _write_t5_closure(root)
            original = contract.EXPECTED_T5_CLOSURE_SEAL_SHA256
            contract.EXPECTED_T5_CLOSURE_SEAL_SHA256 = seal_hash
            try:
                identity = contract.validate_t5_closure_artifact(root)
                self.assertEqual(identity["artifact_seal_sha256"], seal_hash)
                (root / "environment.json").write_text("{}\n", encoding="utf-8")
                with self.assertRaises(contract.ContractError):
                    contract.validate_t5_closure_artifact(root)
            finally:
                contract.EXPECTED_T5_CLOSURE_SEAL_SHA256 = original


def argparse_namespace(single, distributed, output):
    class Namespace:
        single_artifact_dir = single
        distributed_artifact_dir = distributed
        output_dir = output
    return Namespace()


if __name__ == "__main__":
    unittest.main()
