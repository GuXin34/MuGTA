import json
import hashlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from ptc_opd.stage1_artifact import (
    Stage1ArtifactError,
    load_json_strict,
    publish_closed_json_artifact,
)
from ptc_opd import stage1_controller_ledger as ledger


H = "a" * 64
M = "b" * 64


def _handler(paths, _workpack):
    value = paths["payload"].read_text(encoding="utf-8").strip()
    if value == "error":
        raise Stage1ArtifactError("deterministic verifier error")
    if value == "keyerror":
        raise KeyError("deterministic broad exception")
    if value == "fail":
        return {"status": "verified", "gate_passed": False}
    if value == "inconclusive":
        return {
            "status": "verified",
            "gate_passed": True,
            "measurement_quality": "inconclusive_extend_measurement",
        }
    return {"status": "verified", "gate_passed": True, "value": value}


def _graph():
    return (
        {"id": "alpha", "depends_on": []},
        {"id": "beta", "depends_on": ["alpha"]},
    )


def _registry():
    return {
        "alpha": ledger.StageVerifierSpec(
            "alpha", "alpha-verifier-v1", (), (("payload", "file"),), _handler
        ),
        "beta": ledger.StageVerifierSpec(
            "beta", "beta-verifier-v1", ("alpha",), (("payload", "file"),), _handler
        ),
    }


class ControllerLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workpack = self.root / "workpack"
        self.workpack.mkdir()
        self.contract = self.workpack / "contract.json"
        self.contract.write_text("{}\n", encoding="utf-8")
        context = {
            "workpack_root": self.workpack.resolve(),
            "contract": {},
            "contract_sha256": H,
            "workpack_manifest_sha256": M,
        }
        self.patches = (
            patch.object(ledger, "_authority_context", return_value=context),
            patch.object(ledger, "_stage_graph", side_effect=_graph),
            patch.object(ledger, "stage_verifier_registry", side_effect=_registry),
        )
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def payload(self, name, value):
        path = self.root / name
        path.write_text(value + "\n", encoding="utf-8")
        return path

    def init(self, name="ledger-v0000"):
        return ledger.initialize_controller_ledger(
            workpack_root=self.workpack,
            contract_path=self.contract,
            output_dir=self.root / name,
        )

    def issue(
        self,
        stage,
        payload,
        name,
        number=1,
        kind="initial",
        dependencies=None,
        ledger_dir=None,
    ):
        return ledger.issue_stage_verifier_receipt(
            stage_id=stage,
            evidence_paths={"payload": payload},
            dependency_receipt_dirs=dependencies or {},
            workpack_root=self.workpack,
            contract_path=self.contract,
            output_dir=self.root / name,
            attempt_number=number,
            attempt_kind=kind,
            ledger_dir=ledger_dir,
        )

    def test_receipt_and_ledger_bind_live_dependency(self):
        ledger0 = self.init()
        alpha = self.issue(
            "alpha", self.payload("a", "pass"), "receipt-alpha", ledger_dir=ledger0
        )
        ledger1 = ledger.record_controller_receipt(
            ledger0,
            alpha,
            workpack_root=self.workpack,
            contract_path=self.contract,
            output_dir=self.root / "ledger-v0001",
        )
        beta = self.issue(
            "beta",
            self.payload("b", "pass"),
            "receipt-beta",
            dependencies={"alpha": alpha},
            ledger_dir=ledger1,
        )
        ledger2 = ledger.record_controller_receipt(
            ledger1,
            beta,
            workpack_root=self.workpack,
            contract_path=self.contract,
            output_dir=self.root / "ledger-v0002",
        )
        verified = ledger.validate_controller_ledger_authorizations(
            ledger2, workpack_root=self.workpack, contract_path=self.contract
        )
        self.assertEqual(verified["legacy_stages"]["alpha"]["status"], "passed")
        self.assertEqual(verified["legacy_stages"]["beta"]["status"], "passed")
        self.assertEqual(verified["ledger"]["revision"], 2)

    def test_live_evidence_tamper_rejects_receipt(self):
        ledger0 = self.init()
        payload = self.payload("tamper", "pass")
        receipt = self.issue(
            "alpha", payload, "receipt-tamper", ledger_dir=ledger0
        )
        payload.write_text("changed\n", encoding="utf-8")
        with self.assertRaisesRegex(Stage1ArtifactError, "live evidence"):
            ledger.verify_stage_verifier_receipt(
                receipt, workpack_root=self.workpack, contract_path=self.contract
            )

    def test_plain_or_handwritten_passed_ledger_is_rejected(self):
        bare = self.root / "bare.json"
        bare.write_text("{}\n", encoding="utf-8")
        with self.assertRaises(Stage1ArtifactError):
            ledger.validate_controller_ledger_authorizations(
                bare, workpack_root=self.workpack, contract_path=self.contract
            )

        initial = self.init("initial-for-forge")
        value = load_json_strict(initial / ledger.LEDGER_BASENAME)
        value["stages"]["alpha"].update(
            {
                "status": "passed",
                "terminal_receipt": {
                    "artifact_seal_sha256": "f" * 64,
                },
            }
        )
        forged = self.root / "forged"
        publish_closed_json_artifact(
            forged,
            report_name=ledger.LEDGER_BASENAME,
            report=value,
            seal_schema=ledger.LEDGER_SEAL_SCHEMA,
            seal_status="active_controller_ledger",
        )
        with self.assertRaises(Stage1ArtifactError):
            ledger.validate_controller_ledger_authorizations(
                forged, workpack_root=self.workpack, contract_path=self.contract
            )

    def test_scientific_gate_false_is_terminal_failed(self):
        ledger0 = self.init()
        receipt = self.issue(
            "alpha", self.payload("fail", "fail"), "receipt-fail", ledger_dir=ledger0
        )
        report = load_json_strict(receipt / ledger.RECEIPT_BASENAME)
        self.assertEqual(report["stage_status"], "failed")
        self.assertEqual(report["attempt"]["failure_class"], "scientific_gate")
        ledger1 = ledger.record_controller_receipt(
            ledger0,
            receipt,
            workpack_root=self.workpack,
            contract_path=self.contract,
            output_dir=self.root / "ledger-failed",
        )
        verified = ledger.validate_controller_ledger_authorizations(
            ledger1, workpack_root=self.workpack, contract_path=self.contract
        )
        self.assertEqual(verified["legacy_stages"]["alpha"]["status"], "failed")

    def test_initial_plus_two_yellow_attempts_exhaust_budget(self):
        ledger0 = self.init()
        initial = self.issue(
            "alpha", self.payload("error-0", "error"), "receipt-error-0",
            ledger_dir=ledger0,
        )
        with self.assertRaisesRegex(Stage1ArtifactError, "pending receipt"):
            ledger.validate_current_controller_ledger_authorizations(
                ledger0, workpack_root=self.workpack, contract_path=self.contract
            )
        ledger1 = ledger.record_controller_receipt(
            ledger0,
            initial,
            workpack_root=self.workpack,
            contract_path=self.contract,
            output_dir=self.root / "ledger-error-1",
        )
        yellow1 = self.issue(
            "alpha",
            self.payload("error-1", "error"),
            "receipt-error-1",
            number=2,
            kind="yellow",
            ledger_dir=ledger1,
        )
        ledger2 = ledger.record_controller_receipt(
            ledger1,
            yellow1,
            workpack_root=self.workpack,
            contract_path=self.contract,
            output_dir=self.root / "ledger-error-2",
        )
        yellow2 = self.issue(
            "alpha",
            self.payload("error-2", "error"),
            "receipt-error-2",
            number=3,
            kind="yellow",
            ledger_dir=ledger2,
        )
        ledger3 = ledger.record_controller_receipt(
            ledger2,
            yellow2,
            workpack_root=self.workpack,
            contract_path=self.contract,
            output_dir=self.root / "ledger-error-3",
        )
        result = ledger.validate_controller_ledger_authorizations(
            ledger3, workpack_root=self.workpack, contract_path=self.contract
        )["ledger"]["stages"]["alpha"]
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["yellow_retry_count"], 2)
        self.assertEqual(result["stop_reason"], "yellow_retry_budget_exhausted")
        with self.assertRaisesRegex(Stage1ArtifactError, "budget exhausted"):
            self.issue(
                "alpha",
                self.payload("error-3", "error"),
                "receipt-error-3",
                number=4,
                kind="yellow",
                ledger_dir=ledger3,
            )

    def test_inconclusive_measurement_is_retryable_not_terminal(self):
        ledger0 = self.init()
        receipt = self.issue(
            "alpha", self.payload("inc", "inconclusive"), "receipt-inconclusive",
            ledger_dir=ledger0,
        )
        report = load_json_strict(receipt / ledger.RECEIPT_BASENAME)
        self.assertEqual(report["stage_status"], "pending")
        self.assertEqual(report["attempt"]["failure_class"], "retryable_inconclusive")

    def test_duplicate_initial_and_stale_ledger_receipt_are_rejected(self):
        ledger0 = self.init()
        receipt = self.issue(
            "alpha", self.payload("first", "pass"), "receipt-first",
            ledger_dir=ledger0,
        )
        with self.assertRaisesRegex(Stage1ArtifactError, "already pending"):
            self.issue(
                "alpha", self.payload("second", "pass"), "receipt-second",
                ledger_dir=ledger0,
            )
        ledger1 = ledger.record_controller_receipt(
            ledger0,
            receipt,
            workpack_root=self.workpack,
            contract_path=self.contract,
            output_dir=self.root / "ledger-next",
        )
        with self.assertRaisesRegex(Stage1ArtifactError, "stale"):
            ledger.record_controller_receipt(
                ledger0,
                receipt,
                workpack_root=self.workpack,
                contract_path=self.contract,
                output_dir=self.root / "ledger-branch",
            )
        with self.assertRaisesRegex(Stage1ArtifactError, "stale"):
            ledger.validate_current_controller_ledger_authorizations(
                ledger0, workpack_root=self.workpack, contract_path=self.contract
            )
        self.assertEqual(
            ledger.validate_controller_ledger_authorizations(
                ledger1, workpack_root=self.workpack, contract_path=self.contract
            )["ledger"]["revision"],
            1,
        )

    def test_broad_handler_exception_is_a_sealed_pending_receipt(self):
        ledger0 = self.init()
        receipt = self.issue(
            "alpha", self.payload("key", "keyerror"), "receipt-keyerror",
            ledger_dir=ledger0,
        )
        report = load_json_strict(receipt / ledger.RECEIPT_BASENAME)
        self.assertEqual(report["stage_status"], "pending")
        self.assertEqual(report["verifier_result"]["error_type"], "KeyError")

    def test_alternate_valid_dependency_evidence_cannot_be_swapped(self):
        ledger0 = self.init()
        alpha = self.issue(
            "alpha", self.payload("bound-a", "pass"), "receipt-bound-a",
            ledger_dir=ledger0,
        )
        ledger1 = ledger.record_controller_receipt(
            ledger0,
            alpha,
            workpack_root=self.workpack,
            contract_path=self.contract,
            output_dir=self.root / "ledger-bound-a",
        )
        with patch.dict(
            ledger.STAGE_EVIDENCE_BINDINGS,
            {"beta": {"payload": ("alpha", "payload")}},
            clear=False,
        ):
            swapped = self.issue(
                "beta",
                self.payload("bound-b", "pass"),
                "receipt-bound-b",
                dependencies={"alpha": alpha},
                ledger_dir=ledger1,
            )
        report = load_json_strict(swapped / ledger.RECEIPT_BASENAME)
        self.assertEqual(report["stage_status"], "failed")
        self.assertEqual(report["attempt"]["failure_class"], "authority_integrity")

    def test_retained_gate_rejects_wrong_checkpoint_or_audiocraft_source(self):
        checkpoint = {
            "checkpoint_sha256": H,
            "state_dict_sha256": H,
            "compression_state_dict_sha256": H,
        }
        audiocraft = {
            "audiocraft_base_commit": "commit-1",
            "audiocraft_source_sha256": H,
            "audiocraft_source_identity_sha256": H,
            "audiocraft_lm_sha256": H,
        }
        generation_identity = {
            "model_id": "facebook/musicgen-small",
            **checkpoint,
            "audiocraft_base_commit": "commit-1",
            "audiocraft_source_sha256": H,
            "audiocraft_lm_sha256": H,
        }
        accepted = ledger._validate_retained_generation_binding(
            checkpoint, audiocraft, generation_identity
        )
        self.assertEqual(accepted["checkpoint_identity"], checkpoint)
        self.assertEqual(accepted["audiocraft_identity"], audiocraft)

        wrong_checkpoint = dict(checkpoint)
        wrong_checkpoint["state_dict_sha256"] = M
        with self.assertRaisesRegex(
            Stage1ArtifactError, "MusicGen checkpoint.*state_dict_sha256"
        ):
            ledger._validate_retained_generation_binding(
                wrong_checkpoint, audiocraft, generation_identity
            )

        wrong_audiocraft = dict(audiocraft)
        wrong_audiocraft["audiocraft_source_sha256"] = M
        with self.assertRaisesRegex(
            Stage1ArtifactError, "AudioCraft source.*audiocraft_source_sha256"
        ):
            ledger._validate_retained_generation_binding(
                checkpoint, wrong_audiocraft, generation_identity
            )

    def test_self_consistent_wrong_evaluation_lineage_is_rejected_against_b1(self):
        def b1_single(digest):
            checkpoint = {
                "checkpoint_sha256": digest,
                "state_dict_sha256": digest,
                "compression_state_dict_sha256": digest,
            }
            return {
                "scientific_config": {
                    "model_id": "facebook/musicgen-small",
                    "checkpoint_identity": checkpoint,
                    "audiocraft_identity": {
                        "audiocraft_source_sha256": digest,
                    },
                    "cfg_decision": {
                        "decision_file_sha256": digest,
                        "decision_payload_sha256": digest,
                        "scientific_config_sha256": digest,
                        "selected_cfg_scale": 5.0,
                        "loaded_t5_identity_sha256": digest,
                    },
                },
                "loaded_t5_identity": {"identity_sha256": digest},
                "student_state_sha256_before": digest,
                "student_state_sha256_after": digest,
                "teacher_state_sha256_before": digest,
                "teacher_state_sha256_after": digest,
            }

        b1_lineage = ledger._training_lineage_anchor_from_b1_single_result(
            b1_single(H)
        )
        internally_consistent_wrong_lineage = (
            ledger._training_lineage_anchor_from_b1_single_result(b1_single(M))
        )
        pilot_identity = {"kind": "tree", "path": "/pilot", "sha256": H}
        model_identity = {"kind": "tree", "path": "/model", "sha256": H}
        source_identity = {"kind": "tree", "path": "/source", "sha256": H}
        cfg_identity = {"kind": "tree", "path": "/cfg", "sha256": H}
        evidence = {
            "eval_manifest_dir": pilot_identity,
            "musicgen_small_dir": model_identity,
            "audiocraft_dir": source_identity,
            "small_cfg_dir": cfg_identity,
        }
        ancestors = {
            "pilot_eval_manifest": {
                "evidence": {"artifact_dir": pilot_identity},
            },
            "retained_seals_audit": {
                "evidence": {
                    "musicgen_small_dir": model_identity,
                    "audiocraft_dir": source_identity,
                    "small_cfg_dir": cfg_identity,
                },
            },
            "b1_prestability": {
                "verifier_result": {"training_lineage_anchor": b1_lineage},
            },
        }
        self.assertEqual(
            ledger.LINEAGE_SOURCE_STAGE["evaluation_pipeline_qualification"],
            "b1_prestability",
        )
        with self.assertRaisesRegex(
            Stage1ArtifactError, "training lineage differs from b1_prestability"
        ):
            ledger._enforce_dependency_bindings(
                "evaluation_pipeline_qualification",
                evidence,
                ancestors,
                {"training_lineage_anchor": internally_consistent_wrong_lineage},
            )

    def test_evaluation_handler_rejects_alternate_retained_checkpoint(self):
        live_checkpoint = {
            "checkpoint_sha256": H,
            "state_dict_sha256": H,
            "compression_state_dict_sha256": H,
        }
        swapped_checkpoint = {
            "checkpoint_sha256": M,
            "state_dict_sha256": M,
            "compression_state_dict_sha256": M,
        }
        live_audiocraft = {"audiocraft_source_sha256": H}
        decision = types.SimpleNamespace(
            selected_cfg_scale=5.0,
            decision_file_sha256=H,
            decision_payload_sha256=H,
            scientific_config_sha256=H,
            generation_identity={
                "model_id": "facebook/musicgen-small",
                **live_checkpoint,
                "audiocraft_source_sha256": H,
                "loaded_t5_identity_sha256": H,
            },
        )
        expected_cfg = {
            "decision_file_sha256": H,
            "decision_payload_sha256": H,
            "scientific_config_sha256": H,
            "selected_cfg_scale": 5.0,
            "loaded_t5_identity_sha256": H,
        }

        def fake_metric(_metric_dir, *, generation_dir, **_kwargs):
            source_kind = (
                "base_no_cfg"
                if generation_dir.name.startswith("base")
                else "frozen_cfg_teacher"
            )
            return {
                "artifact_seal_sha256": H,
                "generation": {
                    "scientific_config": {
                        "source_kind": source_kind,
                        "model_id": "facebook/musicgen-small",
                        "base_checkpoint": swapped_checkpoint,
                        "audiocraft_source_sha256": H,
                        "cfg_decision": expected_cfg,
                        "base_lm_state_sha256": H,
                        "runtime_identity": {
                            "loaded_t5_identity": {"identity_sha256": H},
                        },
                    }
                },
            }

        fake_probe = types.ModuleType("run_disagreement_probe")
        fake_probe.verify_checkpoint_snapshot = lambda _path: live_checkpoint
        fake_probe.verify_audiocraft_source = lambda _path: live_audiocraft
        evidence = {
            name: self.root / name
            for name in (
                "eval_manifest_dir",
                "musicgen_small_dir",
                "audiocraft_dir",
                "small_cfg_dir",
                "base_generation_dir",
                "base_quality_dir",
                "base_metric_dir",
                "teacher_generation_dir",
                "teacher_quality_dir",
                "teacher_metric_dir",
            )
        }
        with patch.dict(
            sys.modules, {"run_disagreement_probe": fake_probe}
        ), patch(
            "ptc_opd.cfg_decision.verify_cfg_scale_decision",
            return_value=decision,
        ), patch(
            "ptc_opd.stage1_metrics.verify_metric_artifact",
            side_effect=fake_metric,
        ):
            with self.assertRaisesRegex(
                Stage1ArtifactError, "retained MusicGen checkpoint identity"
            ):
                ledger._handler_evaluation_qualification(evidence, self.workpack)
            swapped_checkpoint.clear()
            swapped_checkpoint.update(live_checkpoint)
            accepted = ledger._handler_evaluation_qualification(
                evidence, self.workpack
            )
            self.assertEqual(
                accepted["training_lineage_anchor"]["base_checkpoint"],
                live_checkpoint,
            )

    def test_full_frozen_run_config_rejects_teacher_and_schedule_drift(self):
        run_dir = self.root / "run-config"
        run_dir.mkdir()
        manifest_path = self.root / "train.full.jsonl"
        manifest_path.write_text('{"sample_id":"x","prompt":"p"}\n', encoding="utf-8")
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        config = {
            "manifest": str(manifest_path),
            "student_checkpoint": str(self.root / "checkpoint"),
            "teacher_checkpoint": str(self.root / "checkpoint"),
            "audiocraft_root": str(self.root / "audiocraft"),
            "output_dir": str(run_dir),
            "cfg_scale_decision_dir": str(self.root / "cfg"),
            "cfg_scale_decision_file_sha256": H,
            "cfg_scale_decision_payload_sha256": H,
            "cfg_scale_scientific_config_sha256": H,
            "cfg_generation_checkpoint_sha256": H,
            "cfg_generation_audiocraft_source_sha256": H,
            "cfg_generation_loaded_t5_identity_sha256": H,
            "cfg_generation_state_dict_sha256": H,
            "mode": "uniform100",
            "seed": 2027,
            "learning_rate": 1.0e-6,
            "weight_decay": 0.0,
            "max_optimizer_steps": 500,
            "save_every": 250,
            "log_every": 1,
            "codebook_prior_artifact_dir": None,
            "rank_batch_size": 2,
            "expected_world_size": 8,
            "grad_accum_steps": 4,
            "effective_global_batch": 64,
            "duration_seconds": 10.0,
            "codec_frame_rate": 50.0,
            "token_frames": 500,
            "rollout_temperature": 1.0,
            "rollout_top_k": 250,
            "rollout_top_p": 0.0,
            "teacher_cfg_scale": 5.0,
            "distillation_temperature": 1.0,
            "grad_clip_norm": 1.0,
            "teacher_forward_mode": "separate",
            "check_finite": True,
            "random_mask_namespace": 5701,
            "optimizer_schedule": "cosine",
            "warmup_optimizer_steps": 50,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_eps": 1.0e-8,
            "kl_direction": "forward",
            "denominator_rtol": 1.0e-6,
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "config": config,
                    "manifest_sha256": manifest_sha,
                    "codebook_prior_artifact": None,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        class FakeConfig:
            def __init__(self, **values):
                self.__dict__.update(values)

        fake_train_utils = types.ModuleType("ptc_opd.train_utils")
        fake_train_utils.Stage1Config = FakeConfig
        fake_train_utils.validate_config = lambda value: types.SimpleNamespace(
            requires_perceptual_prior=False
        )
        with patch.dict(sys.modules, {"ptc_opd.train_utils": fake_train_utils}), patch.object(
            ledger,
            "_official_run",
            return_value={"optimizer_step": 500},
        ), patch(
            "ptc_opd.stage1_control._training_lineage_anchor_from_run_manifest",
            return_value={},
        ):
            with self.assertRaisesRegex(Stage1ArtifactError, "teacher_forward_mode"):
                ledger._run_lineage_and_config(
                    self.workpack,
                    run_dir,
                    mode="uniform100",
                    learning_rate=1.0e-6,
                    optimizer_steps=500,
                )

    def test_small_generation_requires_exact_step_1000_checkpoint(self):
        live = {"final_checkpoint_sha256": H}
        ledger._require_small_checkpoint_1000(
            {
                "checkpoint_step": 1000,
                "trained_checkpoint": {"checkpoint_sha256": H},
            },
            live,
        )
        with self.assertRaisesRegex(Stage1ArtifactError, "step 1000"):
            ledger._require_small_checkpoint_1000(
                {
                    "checkpoint_step": 500,
                    "trained_checkpoint": {"checkpoint_sha256": H},
                },
                live,
            )

    def test_all_real_dag_stages_have_supported_registry_handlers(self):
        for item in self.patches:
            item.stop()
        try:
            catalog = ledger.stage_verifier_catalog()
            self.assertEqual(len(catalog), 16)
            self.assertTrue(all(item["supported"] for item in catalog.values()))
            self.assertEqual(
                catalog["performance_benchmark"]["evidence"],
                {
                    "aggregate_dir": "tree",
                    "node0_dir": "tree",
                    "node1_dir": "tree",
                    "node2_dir": "tree",
                    "node3_dir": "tree",
                    "train_manifest": "file",
                    "small_cfg_dir": "tree",
                    "audiocraft_dir": "tree",
                    "musicgen_small_dir": "tree",
                },
            )
            self.assertEqual(
                ledger.STAGE_EVIDENCE_BINDINGS["performance_benchmark"],
                {
                    name: ("retained_seals_audit", name)
                    for name in (
                        "train_manifest",
                        "small_cfg_dir",
                        "audiocraft_dir",
                        "musicgen_small_dir",
                    )
                },
            )
            self.assertEqual(
                ledger.LINEAGE_SOURCE_STAGE["performance_benchmark"],
                "b1_prestability",
            )
            self.assertTrue(
                {
                    "musicgen_small_dir",
                    "audiocraft_dir",
                    "small_cfg_dir",
                }.issubset(
                    catalog["evaluation_pipeline_qualification"]["evidence"]
                )
            )
            self.assertEqual(
                ledger.STAGE_EVIDENCE_BINDINGS[
                    "evaluation_pipeline_qualification"
                ],
                {
                    "eval_manifest_dir": (
                        "pilot_eval_manifest",
                        "artifact_dir",
                    ),
                    "musicgen_small_dir": (
                        "retained_seals_audit",
                        "musicgen_small_dir",
                    ),
                    "audiocraft_dir": (
                        "retained_seals_audit",
                        "audiocraft_dir",
                    ),
                    "small_cfg_dir": (
                        "retained_seals_audit",
                        "small_cfg_dir",
                    ),
                },
            )
        finally:
            for item in self.patches:
                item.start()


if __name__ == "__main__":
    unittest.main()
