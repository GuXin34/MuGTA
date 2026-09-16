"""CPU tests for node-3 orchestration and sealed audit consumers."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import {}".format(relative))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TRAIN = _load("node3_test_train_stage1", "scripts/train_stage1.py")
GATE = _load("node3_test_gate_runner", "scripts/run_node3_gate.py")


def _fault_config(**overrides):
    values = {
        "mode": "uniform100",
        "seed": 2027,
        "learning_rate": 3.0e-6,
        "max_optimizer_steps": 2,
        "save_every": 1,
        "log_every": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _fault_args(**overrides):
    values = {"node3_stop_after_step": 1, "resume": None}
    values.update(overrides)
    return SimpleNamespace(**values)


def _ddp_reducer_contract():
    value = dict(GATE.NODE3_DDP_REDUCER_STATIC_CONTRACT)
    value.update(
        {
            "bucket_sizes": "40",
            "bucket_count": 1,
            "trainable_parameter_tensor_count": 2,
            "trainable_parameter_numel": 10,
            "trainable_gradient_bytes": 40,
            "parameter_layout_sha256": "d" * 64,
        }
    )
    value["identity_sha256"] = GATE._canonical_json_sha256(value)
    return value


def _rank_audit(rank: int):
    microsteps = []
    for index in range(4):
        microsteps.append(
            {
                "accumulation_index": index,
                "global_microstep": 4 + index,
                "sample_ids": ["r{}-{}-a".format(rank, index), "r{}-{}-b".format(rank, index)],
                "rollout_shape": [2, 4, 500],
                "rollout_dtype": "torch.int64",
                "rollout_codes_sha256": "c" * 64,
                "use_cfg": False,
                "condition_tensors_source": "conditional",
                "selected_gate_sha256": "a" * 64,
                "selected_cells": 3992,
                "valid_cells": 3992,
                "global_denominator": 100.0,
                "global_loss_finite": True,
            }
        )
    return {
        "rank": rank,
        "local_rank": rank,
        "microsteps": microsteps,
        "gradient_finite": True,
        "all_trainable_gradients_present": True,
        "student_conditioner_status": "frozen_eval",
        "teacher_conditioner_status": "frozen_eval",
        "node3_determinism": copy.deepcopy(GATE.NODE3_DETERMINISM_CONTRACT),
        "ddp_reducer": _ddp_reducer_contract(),
        "resume_rng_restore": {
            "schema_version": "ptc-opd-node3-rng-restore-v1",
            "rank": rank,
            "captured_before_next_rollout_seed": True,
            "expected_sha256": "b" * 64,
            "observed_sha256": "b" * 64,
            "exact": True,
        },
        "resume_state_restore": {
            "schema_version": "ptc-opd-node3-resume-state-restore-v1",
            "rank": rank,
            "captured_after_checkpoint_load": True,
            "captured_before_next_forward": True,
            "canonical_identity": (
                "finite tensor-tree SHA-256 exact over dtype/shape/layout/stride/"
                "storage_offset/logical-bytes and typed containers"
            ),
            "checkpoint": {
                "student_state_sha256": "d" * 64,
                "optimizer_state_sha256": "e" * 64,
            },
            "live": {
                "student_state_sha256": "d" * 64,
                "optimizer_state_sha256": "e" * 64,
            },
            "exact_fields": {
                "student_state_sha256": True,
                "optimizer_state_sha256": True,
            },
            "exact": True,
        },
        "cuda_max_memory_allocated": 100,
        "cuda_max_memory_reserved": 120,
        "cuda_total_memory_bytes": 1000,
    }


def _step_record():
    return {
        "optimizer_step": 2,
        "global_microstep": 8,
        "loss": 0.25,
        "gradient_norm": 0.5,
        "global_denominators": [100.0, 100.0, 100.0, 100.0],
        "all_rank_audit": [_rank_audit(rank) for rank in range(8)],
    }


def _conditioner(name: str):
    return {
        "name": name,
        "status": "frozen_eval",
        "provider_training": False,
        "provider_trainable_parameter_count": 0,
        "external_t5_training": [False],
        "external_t5_trainable_parameter_counts": [0],
    }


def _runtime_contract(rank: int):
    return {
        "rank": rank,
        "cpu_load": {
            "student_all_fp32": True,
            "teacher_all_fp32": True,
            "student_all_cpu": True,
            "teacher_all_cpu": True,
        },
        "student_conditioner": _conditioner("student"),
        "teacher_conditioner": _conditioner("teacher"),
        "generate_accepts_use_cfg": True,
        "generate_accepts_condition_tensors": True,
        "bf16_supported": True,
        "bf16_autocast_probe_dtype": "torch.bfloat16",
        "bf16_autocast_probe_finite": True,
        "gpu_name": "NVIDIA H20",
        "gpu_capability": [9, 0],
        "cuda_device_count": 8,
        "node3_determinism": copy.deepcopy(GATE.NODE3_DETERMINISM_CONTRACT),
        "ddp_reducer": _ddp_reducer_contract(),
    }


class FaultInjectionContractTest(unittest.TestCase):
    def test_frozen_seed_and_learning_rate_are_exact(self) -> None:
        GATE.validate_node3_protocol_hyperparameters(2027, 3.0e-6)
        with self.assertRaisesRegex(GATE.Node3GateError, "seed must be exactly"):
            GATE.validate_node3_protocol_hyperparameters(2028, 3.0e-6)
        with self.assertRaisesRegex(GATE.Node3GateError, "learning rate"):
            GATE.validate_node3_protocol_hyperparameters(2027, 2.0e-6)

    def test_dual_key_exact_smoke_only(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "PTC_NODE3_GATE=1"):
                TRAIN.validate_node3_fault_injection(_fault_config(), _fault_args())
        with mock.patch.dict(os.environ, {"PTC_NODE3_GATE": "1"}, clear=True):
            TRAIN.validate_node3_fault_injection(_fault_config(), _fault_args())
            invalid = (
                (_fault_config(mode="ptc50"), _fault_args()),
                (_fault_config(max_optimizer_steps=500), _fault_args()),
                (_fault_config(save_every=2), _fault_args()),
                (_fault_config(log_every=2), _fault_args()),
                (_fault_config(), _fault_args(node3_stop_after_step=2)),
                (_fault_config(), _fault_args(resume=Path("step-00001"))),
            )
            for config, args in invalid:
                with self.subTest(config=config, args=args), self.assertRaises(ValueError):
                    TRAIN.validate_node3_fault_injection(config, args)

    def test_dry_run_rejects_injection_before_config_resolution(self) -> None:
        argv = [
            "--manifest", "train.full.jsonl",
            "--student-checkpoint", "student",
            "--teacher-checkpoint", "teacher",
            "--audiocraft-root", "audiocraft",
            "--cfg-scale-decision-dir", "cfg",
            "--output-dir", "run",
            "--mode", "uniform100",
            "--seed", "2027",
            "--learning-rate", "3e-6",
            "--max-optimizer-steps", "2",
            "--save-every", "1",
            "--log-every", "1",
            "--dry-run",
            "--node3-stop-after-step", "1",
        ]
        with mock.patch.object(TRAIN, "make_config") as make_config:
            with self.assertRaisesRegex(ValueError, "forbidden under --dry-run"):
                TRAIN.main(argv)
            make_config.assert_not_called()

    def test_node3_training_contract_is_gate_scoped(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            TRAIN.validate_node3_training_contract(
                _fault_config(
                    mode="ptc50",
                    seed=999,
                    learning_rate=1.0e-5,
                    max_optimizer_steps=500,
                ),
                _fault_args(),
            )
            self.assertEqual(
                TRAIN.configure_node3_determinism(), {"gate_mode": False}
            )
        with mock.patch.dict(os.environ, {"PTC_NODE3_GATE": "1"}, clear=True):
            TRAIN.validate_node3_training_contract(_fault_config(), _fault_args())
            with self.assertRaisesRegex(ValueError, "restricted"):
                TRAIN.validate_node3_training_contract(
                    _fault_config(max_optimizer_steps=500), _fault_args()
                )
            with self.assertRaisesRegex(ValueError, "seed=2027"):
                TRAIN.validate_node3_training_contract(
                    _fault_config(seed=2028), _fault_args()
                )
            with self.assertRaisesRegex(ValueError, "lr=3e-6"):
                TRAIN.validate_node3_training_contract(
                    _fault_config(learning_rate=2.0e-6), _fault_args()
                )


class AuditConsumerTest(unittest.TestCase):
    def test_full_rank_step_audit_and_tamper_rejection(self) -> None:
        record = _step_record()
        summary = GATE._validate_step_records([record])
        self.assertEqual(summary["record_count"], 1)
        self.assertEqual(len(summary["memory"]), 8)

        malformed_gate = copy.deepcopy(record)
        malformed_gate["all_rank_audit"][3]["microsteps"][1]["selected_gate_sha256"] = "bad"
        with self.assertRaisesRegex(GATE.Node3GateError, "gate digest"):
            GATE._validate_step_records([malformed_gate])

        unequal = copy.deepcopy(record)
        unequal["global_denominators"][3] = 100.01
        with self.assertRaisesRegex(GATE.Node3GateError, "denominators differ"):
            GATE._validate_step_records([unequal])

        missing_rank = copy.deepcopy(record)
        missing_rank["all_rank_audit"].pop()
        with self.assertRaisesRegex(GATE.Node3GateError, "eight rank"):
            GATE._validate_step_records([missing_rank])

        deterministic_drift = copy.deepcopy(record)
        deterministic_drift["all_rank_audit"][2]["node3_determinism"][
            "cudnn_benchmark"
        ] = True
        with self.assertRaisesRegex(GATE.Node3GateError, "deterministic flags"):
            GATE._validate_step_records([deterministic_drift])

        replayed_microstep = copy.deepcopy(record)
        replayed_microstep["all_rank_audit"][4]["microsteps"][2][
            "global_microstep"
        ] = 4
        with self.assertRaisesRegex(GATE.Node3GateError, "microstep progress"):
            GATE._validate_step_records([replayed_microstep])

        wrong_accumulation_index = copy.deepcopy(record)
        wrong_accumulation_index["all_rank_audit"][4]["microsteps"][2][
            "accumulation_index"
        ] = 1
        with self.assertRaisesRegex(GATE.Node3GateError, "indices"):
            GATE._validate_step_records([wrong_accumulation_index])

        rebuilt = copy.deepcopy(record)
        rebuilt["all_rank_audit"][0]["ddp_reducer"][
            "has_rebuilt_buckets"
        ] = True
        with self.assertRaisesRegex(GATE.Node3GateError, "DDP reducer contract"):
            GATE._validate_step_records([rebuilt])

        missing_gradient = copy.deepcopy(record)
        missing_gradient["all_rank_audit"][1][
            "all_trainable_gradients_present"
        ] = False
        with self.assertRaisesRegex(GATE.Node3GateError, "gradient coverage"):
            GATE._validate_step_records([missing_gradient])

    def test_resume_rng_restore_is_eight_rank_and_pre_reseed(self) -> None:
        record = _step_record()
        self.assertEqual(GATE._validate_resume_rng_restore(record)["rank_count"], 8)
        tampered = copy.deepcopy(record)
        tampered["all_rank_audit"][7]["resume_rng_restore"][
            "observed_sha256"
        ] = "c" * 64
        with self.assertRaisesRegex(GATE.Node3GateError, "not restored exactly"):
            GATE._validate_resume_rng_restore(tampered)
        late = copy.deepcopy(record)
        late["all_rank_audit"][1]["resume_rng_restore"][
            "captured_before_next_rollout_seed"
        ] = False
        with self.assertRaisesRegex(GATE.Node3GateError, "pre-reseed"):
            GATE._validate_resume_rng_restore(late)

    def test_resume_state_restore_is_eight_rank_exact_and_pre_forward(self) -> None:
        record = _step_record()
        result = GATE._validate_resume_state_restore(record)
        self.assertEqual(result["rank_count"], 8)
        tampered = copy.deepcopy(record)
        tampered["all_rank_audit"][4]["resume_state_restore"]["live"][
            "optimizer_state_sha256"
        ] = "f" * 64
        with self.assertRaisesRegex(GATE.Node3GateError, "not restored exactly"):
            GATE._validate_resume_state_restore(tampered)
        late = copy.deepcopy(record)
        late["all_rank_audit"][2]["resume_state_restore"][
            "captured_before_next_forward"
        ] = False
        with self.assertRaisesRegex(GATE.Node3GateError, "before next forward"):
            GATE._validate_resume_state_restore(late)

    def test_runtime_contract_is_fail_closed(self) -> None:
        manifest = {
            "runtime_contract_by_rank": [_runtime_contract(rank) for rank in range(8)]
        }
        self.assertEqual(GATE._validate_runtime_contract(manifest)["rank_count"], 8)
        tampered = copy.deepcopy(manifest)
        tampered["runtime_contract_by_rank"][5]["student_conditioner"][
            "provider_trainable_parameter_count"
        ] = 1
        with self.assertRaisesRegex(GATE.Node3GateError, "trainable"):
            GATE._validate_runtime_contract(tampered)

        deterministic_drift = copy.deepcopy(manifest)
        deterministic_drift["runtime_contract_by_rank"][6]["node3_determinism"][
            "cuda_matmul_allow_tf32"
        ] = True
        with self.assertRaisesRegex(
            GATE.Node3GateError, "deterministic runtime contract"
        ):
            GATE._validate_runtime_contract(deterministic_drift)

        rebuilt = copy.deepcopy(manifest)
        rebuilt["runtime_contract_by_rank"][2]["ddp_reducer"][
            "has_rebuilt_buckets"
        ] = True
        with self.assertRaisesRegex(GATE.Node3GateError, "DDP reducer contract"):
            GATE._validate_runtime_contract(rebuilt)

        for field, impostor in (
            ("find_unused_parameters", 1),
            ("static_graph", 0),
            ("bucket_cap_bytes", float(25 * 1024 * 1024)),
        ):
            with self.subTest(field=field):
                malformed = copy.deepcopy(manifest)
                reducer = malformed["runtime_contract_by_rank"][0]["ddp_reducer"]
                reducer[field] = impostor
                identity_payload = dict(reducer)
                del identity_payload["identity_sha256"]
                reducer["identity_sha256"] = GATE._canonical_json_sha256(
                    identity_payload
                )
                with self.assertRaisesRegex(
                    GATE.Node3GateError, "DDP reducer contract"
                ):
                    GATE._validate_runtime_contract(malformed)

    def test_resume_comparison_includes_samples_and_gate_digest(self) -> None:
        first = _step_record()
        second = copy.deepcopy(first)
        self.assertEqual(
            GATE._canonical_resume_audit(first),
            GATE._canonical_resume_audit(second),
        )
        second["all_rank_audit"][7]["microsteps"][0]["sample_ids"][0] = "changed"
        self.assertNotEqual(
            GATE._canonical_resume_audit(first),
            GATE._canonical_resume_audit(second),
        )
        second = copy.deepcopy(first)
        second["all_rank_audit"][7]["microsteps"][0][
            "rollout_codes_sha256"
        ] = "e" * 64
        self.assertNotEqual(
            GATE._canonical_resume_audit(first),
            GATE._canonical_resume_audit(second),
        )

    def test_step2_metric_tolerance_policy_is_explicit_and_fail_closed(self) -> None:
        first = _step_record()
        first.update(
            {
                "mode": "uniform100",
                "learning_rate": 1.2e-7,
                "denominator_window_constant": True,
                "selected_cells_rank0": 100,
                "valid_cells_rank0": 100,
            }
        )
        second = copy.deepcopy(first)
        second["loss"] += 0.5 * GATE.STEP2_LOSS_ATOL
        second["gradient_norm"] += 0.5 * GATE.STEP2_GRAD_NORM_ATOL
        self.assertEqual(
            GATE._compare_step2_metrics(first, second)["status"], "passed"
        )
        with self.assertRaisesRegex(GATE.Node3GateError, "cold-resume"):
            GATE._compare_cold_step2_metrics_exact(first, second)
        self.assertEqual(
            GATE._compare_cold_step2_metrics_exact(first, copy.deepcopy(first))[
                "status"
            ],
            "passed",
        )
        second["loss"] = first["loss"] + 1.0e-3
        with self.assertRaisesRegex(GATE.Node3GateError, "loss differs"):
            GATE._compare_step2_metrics(first, second)
        second = copy.deepcopy(first)
        second["learning_rate"] *= 2.0
        with self.assertRaisesRegex(GATE.Node3GateError, "exact metric fields"):
            GATE._compare_step2_metrics(first, second)


class EvidenceAndEnvironmentTest(unittest.TestCase):
    def test_outside_output_typo_creates_no_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            # macOS exposes /var as a symlink to /private/var; keep fixture
            # containment independent of that OS-level alias.
            parent = Path(temporary).resolve()
            workpack = parent / "workpack"
            workpack.mkdir()
            inside_console = workpack / "console_logs"
            outside_runs = parent / "typo-outside" / "runs"
            with self.assertRaisesRegex(GATE.Node3GateError, "must stay inside"):
                GATE.prepare_output_roots(
                    inside_console, outside_runs, workpack
                )
            self.assertFalse(inside_console.exists())
            self.assertFalse(outside_runs.exists())
            self.assertFalse(outside_runs.parent.exists())

    def test_symlink_output_ancestor_is_rejected_before_mkdir(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            # Reach the intended inner symlink check rather than the macOS
            # /var -> /private/var alias check.
            parent = Path(temporary).resolve()
            workpack = parent / "workpack"
            outside = parent / "outside"
            workpack.mkdir()
            outside.mkdir()
            link = workpack / "linked"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError as exc:
                self.skipTest("symlinks unavailable: {}".format(exc))
            target = link / "console_logs"
            with self.assertRaisesRegex(GATE.Node3GateError, "symlink"):
                GATE.validate_prospective_output_root(
                    target, workpack, "console root"
                )
            self.assertFalse((outside / "console_logs").exists())

    def test_retained_small_decision_requires_both_pinned_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            decision = directory / "cfg_scale_decision.json"
            sidecar = directory / "cfg_scale_decision.sha256.json"
            decision.write_text(
                json.dumps(
                    {
                        "status": "selected",
                        "selected_cfg_scale": 5.0,
                        "generation_identity": {
                            "model_id": "facebook/musicgen-small"
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            sidecar.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(
                GATE, "RETAINED_SMALL_DECISION_SHA256", GATE.sha256_file(decision)
            ), mock.patch.object(
                GATE,
                "RETAINED_SMALL_DECISION_SIDECAR_SHA256",
                GATE.sha256_file(sidecar),
            ):
                result = GATE._verify_retained_small_decision(directory)
                self.assertEqual(result["selected_cfg_scale"], 5.0)
                sidecar.write_text('{"tampered":true}\n', encoding="utf-8")
                with self.assertRaisesRegex(
                    GATE.Node3GateError,
                    r"cfg_scale_decision\.sha256\.json sidecar differs from the Gate-1 pin",
                ):
                    GATE._verify_retained_small_decision(directory)

    def test_pollution_is_removed_from_child_environment(self) -> None:
        additions = {name: "polluted" for name in GATE.POLLUTION_VARIABLES}
        with mock.patch.dict(os.environ, additions, clear=True):
            environment, removed = GATE.clean_child_environment()
        self.assertEqual(set(removed), set(GATE.POLLUTION_VARIABLES))
        self.assertTrue(all(name not in environment for name in GATE.POLLUTION_VARIABLES))
        self.assertEqual(environment["PTC_NODE3_GATE"], "1")

    def test_expected_failure_requires_nonzero_and_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recorder = GATE.GateRecorder(root, dict(os.environ))
            metadata = recorder.run(
                "B02_controlled_step1_failure",
                "expected_failure",
                [sys.executable, "-c", "import sys; print('MARK', file=sys.stderr); sys.exit(3)"],
                cwd=root,
                expected_success=False,
                required_marker="MARK",
            )
            self.assertEqual(metadata["status"], "passed")
            with self.assertRaisesRegex(GATE.Node3GateError, "marker_ok=False"):
                recorder.run(
                    "B02_controlled_step1_failure",
                    "missing_marker",
                    [sys.executable, "-c", "import sys; sys.exit(3)"],
                    cwd=root,
                    expected_success=False,
                    required_marker="MARK",
                )

    def test_evidence_manifest_freezes_status_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "STATUS.json").write_text('{"status":"passed"}\n', encoding="utf-8")
            (root / "command.log").write_text("ok\n", encoding="utf-8")
            GATE._write_checksums(root)
            entries = dict(
                line.split("  ", 1)
                for line in (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
            )
            self.assertIn("STATUS.json", entries.values())
            self.assertIn("command.log", entries.values())
            sidecar = (root / "SHA256SUMS.txt.sha256").read_text(encoding="utf-8")
            self.assertIn(GATE.sha256_file(root / "SHA256SUMS.txt"), sidecar)


if __name__ == "__main__":
    unittest.main()
