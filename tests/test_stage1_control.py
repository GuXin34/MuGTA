import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from ptc_opd.stage1_artifact import (
    Stage1ArtifactError,
    artifact_member,
    canonical_json_bytes,
    canonical_json_sha256,
    publish_closed_json_artifact,
    sha256_file,
    sha256_tree,
)
from ptc_opd.stage1_control import (
    B1_FULL_CLOSURE_BASENAME,
    B1_FULL_CLOSURE_SEAL_SCHEMA,
    CONTROLLER_PREFLIGHT_SCHEMA,
    LR_DECISION_BASENAME,
    LR_DECISION_SEAL_SCHEMA,
    LR_GRID,
    LR_SUMMARY_BASENAME,
    LR_SUMMARY_SEAL_SCHEMA,
    PILOT_OUTPUT_BASENAME,
    PRIMARY_PILOT_METHODS,
    PTC500_REPORT_BASENAME,
    SMALL_PILOT_ARTIFACT_FIELDS,
    SMALL_PILOT_METHODS,
    SMALL_PILOT_SUMMARY_BASENAME,
    SMALL_PILOT_SUMMARY_SEAL_SCHEMA,
    TRAINING_LINEAGE_SCHEMA,
    build_small_pilot_decision,
    build_b1_full_closure,
    build_lr_decision,
    build_lr_evaluation_summary,
    build_ptc500_stability_report,
    controller_next_action,
    controller_plan,
    controller_preflight,
    load_autonomy_contract,
    publish_lr_decision,
    publish_lr_evaluation_summary,
    publish_pilot_eval_manifest,
    publish_ptc500_stability_report,
    publish_small_pilot_decision,
    publish_b1_full_closure,
    read_jsonl_strict,
    verify_lr_decision,
    verify_pilot_eval_manifest,
    verify_ptc500_stability_report,
    verify_small_pilot_decision,
    verify_b1_full_closure,
)
from ptc_opd.stage1_generation import (
    CONFIG_NAME,
    GENERATION_SCHEMA_VERSION,
    SAMPLE_SCHEMA_VERSION,
    SAMPLES_NAME,
    SEAL_NAME as GENERATION_SEAL_NAME,
    SEAL_SCHEMA_VERSION as GENERATION_SEAL_SCHEMA,
    derive_generation_seed,
)
from ptc_opd.stage1_metrics import (
    ACCEPTED_EVALUATOR_IDENTITIES,
    ACCEPTED_MUQ_BACKBONE_TREE_SHA256,
    ACCEPTED_MUQ_CONFIG_FILES,
    ACCEPTED_MUQ_CONFIG_FILE_SET_SHA256,
    FINAL_METRICS,
    METRIC_PROVENANCE_NAME,
    METRIC_PROVENANCE_SCHEMA_VERSION,
    METRIC_SCHEMA_VERSION,
    METRIC_SCORES_NAME,
    METRIC_SEAL_SCHEMA_VERSION,
    OFFLINE_ENVIRONMENT,
    QUALITY_METRICS,
    QUALITY_PROVENANCE_NAME,
    QUALITY_PROVENANCE_SCHEMA_VERSION,
    QUALITY_SCHEMA_VERSION,
    QUALITY_SCORES_NAME,
    QUALITY_SEAL_SCHEMA_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
HASH = "a" * 64


def _accepted_evaluator_identity(label):
    identity = dict(ACCEPTED_EVALUATOR_IDENTITIES[label])
    details = {}
    if label == "muq_eval":
        details = {
            "config_files": [
                {"basename": basename, "sha256": digest}
                for basename, digest in sorted(ACCEPTED_MUQ_CONFIG_FILES.items())
            ],
            "config_file_set_sha256": ACCEPTED_MUQ_CONFIG_FILE_SET_SHA256,
            "local_encoder_snapshot_sha256": ACCEPTED_MUQ_BACKBONE_TREE_SHA256,
            "declared_encoder_id": "OpenMuQ/MuQ-large-msd-iter",
        }
    identity["details"] = details
    return identity


def _lineage_anchor():
    return {
        "schema_version": TRAINING_LINEAGE_SCHEMA,
        "model_id": "facebook/musicgen-small",
        "base_checkpoint": {
            "checkpoint_sha256": "2" * 64,
            "state_dict_sha256": "3" * 64,
            "compression_state_dict_sha256": "4" * 64,
        },
        "base_lm_state_sha256": "b" * 64,
        "audiocraft_source_sha256": "7" * 64,
        "cfg_decision": {
            "decision_file_sha256": "8" * 64,
            "decision_payload_sha256": "9" * 64,
            "scientific_config_sha256": "a" * 64,
            "selected_cfg_scale": 5.0,
        },
        "loaded_t5_identity_sha256": "1" * 64,
    }


def _write_json(path, value):
    path.write_bytes(canonical_json_bytes(value))


def _write_jsonl(path, rows):
    path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))


def _source_dev(path):
    rows = [
        {
            "sample_id": "dev-{:03d}".format(index),
            "prompt": "prompt {:03d}".format(index),
            "split": "dev",
        }
        for index in range(300)
    ]
    _write_jsonl(path, rows)
    return rows


def _make_b1_prestability_stub(root):
    directory = root / "b1-prestability"
    directory.mkdir()
    summary = directory / "b1_prestability_summary.json"
    seal = directory / "artifact_seal.json"
    _write_json(summary, {"scientific_status": "b1_prestability_passed"})
    _write_json(seal, {"schema_version": "test-b1-final-seal"})
    verification = {
        "kind": "final",
        "directory": str(directory.resolve()),
        "seal_sha256": sha256_file(seal),
        "summary_sha256": sha256_file(summary),
        "prestability_gate_passed": True,
        "full_b1_passed": False,
    }
    return directory, verification


def _rank_audit(reserved=80):
    return [
        {
            "rank": rank,
            "gradient_finite": True,
            "all_trainable_gradients_present": True,
            "cuda_total_memory_bytes": 100,
            "cuda_max_memory_reserved": reserved,
            "microsteps": [
                {
                    "selected_cells": 1,
                    "valid_cells": 2,
                    "global_denominator": float(index + 1),
                    "global_loss_finite": True,
                }
                for index in range(4)
            ],
        }
        for rank in range(8)
    ]


def _metric_records(mode, base_lr, *, reserved=80, steps=500):
    return [
        {
            "schema_version": "ptc-opd-stage1-run-v1",
            "event": "optimizer_step",
            "optimizer_step": step,
            "global_microstep": 4 * step,
            "mode": mode,
            "loss": 1.0,
            "global_denominators": [1.0, 2.0, 3.0, 4.0],
            "denominator_window_constant": True,
            "gradient_norm": 0.5,
            "learning_rate": base_lr * min(float(step) / 50.0, 1.0),
            "selected_cells_rank0": 1,
            "valid_cells_rank0": 2,
            "step_seconds": 1.0,
            "all_rank_audit": _rank_audit(reserved),
        }
        for step in range(1, steps + 1)
    ]


def _make_run(root, name, mode, learning_rate, *, reserved=80, steps=500):
    run = root / name
    (run / "logs").mkdir(parents=True)
    config = {
        "mode": mode,
        "seed": 2027,
        "learning_rate": learning_rate,
        "max_optimizer_steps": steps,
        "save_every": 250,
        "log_every": 1,
        "expected_world_size": 8,
        "effective_global_batch": 64,
        "check_finite": True,
        "cfg_scale_decision_file_sha256": "8" * 64,
        "cfg_scale_decision_payload_sha256": "9" * 64,
        "cfg_scale_scientific_config_sha256": "a" * 64,
        "cfg_generation_checkpoint_sha256": "2" * 64,
        "cfg_generation_state_dict_sha256": "3" * 64,
        "cfg_generation_audiocraft_source_sha256": "7" * 64,
        "cfg_generation_loaded_t5_identity_sha256": "1" * 64,
        "teacher_cfg_scale": 5.0,
    }
    cfg_decision = {
        "selected_cfg_scale": 5.0,
        "decision_file_sha256": "8" * 64,
        "decision_payload_sha256": "9" * 64,
        "scientific_config_sha256": "a" * 64,
        "generation_identity": {
            "checkpoint_sha256": "2" * 64,
            "state_dict_sha256": "3" * 64,
            "compression_state_dict_sha256": "4" * 64,
            "audiocraft_source_sha256": "7" * 64,
            "loaded_t5_identity_sha256": "1" * 64,
        },
    }
    _write_json(
        run / "run_manifest.json",
        {
            "config": config,
            "world_size": 8,
            "nnodes": 1,
            "student_checkpoint_sha256": "2" * 64,
            "teacher_checkpoint_sha256": "2" * 64,
            "student_state_dict_sha256": "3" * 64,
            "teacher_state_dict_sha256": "3" * 64,
            "student_state_sha256_initial": "b" * 64,
            "teacher_state_sha256_initial": "b" * 64,
            "audiocraft_source_identity": {"tree_sha256": "7" * 64},
            "cfg_scale_decision": cfg_decision,
            "cfg_generation_binding": {
                "checkpoint_sha256": "2" * 64,
                "state_dict_sha256": "3" * 64,
                "audiocraft_source_sha256": "7" * 64,
            },
            "loaded_t5_identity": {"identity_sha256": "1" * 64},
        },
    )
    metrics_path = run / "logs" / "attempt-0000.metrics.jsonl"
    _write_jsonl(
        metrics_path,
        _metric_records(mode, learning_rate, reserved=reserved, steps=steps),
    )
    seal = {
        "status": "sealed",
        "optimizer_step": steps,
        "checkpoint_inventory": [
            {"optimizer_step": step} for step in (0, 250, steps)
        ],
        "attempt_logs": [
            {
                "metrics_path": "logs/attempt-0000.metrics.jsonl",
                "metrics_sha256": sha256_file(metrics_path),
                "start_optimizer_step": 0,
            }
        ],
    }
    _write_json(run / "SEALED.json", seal)
    _write_json(run / "DONE.json", {"status": "complete"})
    verification = {
        "schema_version": "ptc-opd-stage1-verification-v1",
        "status": "verified",
        "run_directory": str(run.resolve()),
        "optimizer_step": steps,
        "run_manifest_sha256": sha256_file(run / "run_manifest.json"),
        "SEALED.json_sha256": sha256_file(run / "SEALED.json"),
        "DONE.json_sha256": sha256_file(run / "DONE.json"),
        "final_checkpoint_sha256": hashlib.sha256(name.encode()).hexdigest(),
    }
    return run, verification


def _make_generation(
    root,
    name,
    eval_dir,
    prompts,
    *,
    source_kind,
    run_verification=None,
    learning_rate=None,
    method=None,
    checkpoint_step=0,
):
    directory = root / name
    audio_root = directory / "audio"
    audio_root.mkdir(parents=True)
    manifest_seal = sha256_file(eval_dir / "artifact_seal.json")
    t5_hash = "1" * 64
    condition_id = (
        source_kind
        if method is None
        else "{}.lr{:.12g}.step{}.seed{}".format(
            method, float(learning_rate), checkpoint_step, 2027
        )
    )
    config = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "model_id": "facebook/musicgen-small",
        "source_kind": source_kind,
        "condition_id": condition_id,
        "method": method,
        "train_seed": None if method is None else 2027,
        "learning_rate": learning_rate,
        "checkpoint_step": checkpoint_step,
        "base_checkpoint": {
            "checkpoint_sha256": "2" * 64,
            "state_dict_sha256": "3" * 64,
            "compression_state_dict_sha256": "4" * 64,
        },
        "trained_checkpoint": (
            None
            if run_verification is None
            else {
                "path": "checkpoints/step-00500",
                "checkpoint_sha256": run_verification["final_checkpoint_sha256"],
                "sidecar_sha256": "5" * 64,
            }
        ),
        "stage1_run": None if run_verification is None else dict(run_verification),
        "stage1_run_tree_sha256": None if run_verification is None else "6" * 64,
        "audiocraft_source_sha256": "7" * 64,
        "cfg_decision": {
            "decision_file_sha256": "8" * 64,
            "decision_payload_sha256": "9" * 64,
            "scientific_config_sha256": "a" * 64,
            "selected_cfg_scale": 5.0,
            "loaded_t5_identity_sha256": t5_hash,
        },
        "prompt_count": 128,
        "eval_manifest_artifact_seal_sha256": manifest_seal,
        "eval_manifest_sha256": sha256_file(eval_dir / PILOT_OUTPUT_BASENAME),
        "generation_seeds": [31001, 31002],
        "derived_seed_namespace": "ptc-opd-small-pilot-generation-v1",
        "duration_seconds": 10.0,
        "sample_rate": 32000,
        "codec_frame_rate": 50.0,
        "token_frames": 500,
        "sampling": {
            "use_sampling": True,
            "temperature": 1.0,
            "top_k": 250,
            "top_p": 0.0,
            "two_step_cfg": False,
        },
        "precision": {
            "load": "torch.float32",
            "conditioner_compute": "torch.float32",
            "lm_generation_compute": "torch.bfloat16",
            "compression_decode_compute": "torch.float32",
        },
        "student_inference_cfg": False,
        "teacher_cfg_scale": None,
        "no_loudness_normalization": True,
        "replace_failed_audio": False,
        "best_of_n": False,
        "runtime_identity": {
            "loaded_t5_identity": {"identity_sha256": t5_hash},
            "precision_contract": {
                "lm_parameter_dtype": "torch.float32",
                "compression_parameter_dtype": "torch.float32",
                "conditioner_parameter_dtype": "torch.float32",
                "conditioner_compute_dtype": "torch.float32",
                "lm_generation_compute_dtype": "torch.bfloat16",
                "compression_decode_compute_dtype": "torch.float32",
            },
        },
        "base_lm_state_sha256": "b" * 64,
        "loaded_trained_lm_state_sha256": (
            None if run_verification is None else "c" * 64
        ),
    }
    config_hash = canonical_json_sha256(config)
    rows = []
    for prompt in prompts:
        for seed in (31001, 31002):
            relative = "audio/{}-{}.wav".format(prompt["sample_id"], seed)
            audio = directory / relative
            audio.write_bytes(
                "{}:{}:{}".format(name, prompt["sample_id"], seed).encode()
            )
            rows.append(
                {
                    "schema_version": SAMPLE_SCHEMA_VERSION,
                    "sample_id": prompt["sample_id"],
                    "prompt_sha256": hashlib.sha256(
                        prompt["prompt"].encode()
                    ).hexdigest(),
                    "generation_seed": seed,
                    "derived_seed": derive_generation_seed(prompt["sample_id"], seed),
                    "condition_id": config["condition_id"],
                    "method": method,
                    "checkpoint_step": checkpoint_step,
                    "path": relative,
                    "audio_sha256": sha256_file(audio),
                    "audio_frames": 320000,
                    "audio_channels": 1,
                    "audio_sample_rate": 32000,
                    "audio_subtype": "FLOAT",
                    "audio_peak_abs": 0.5,
                    "audio_rms": 0.1,
                    "scientific_config_sha256": config_hash,
                }
            )
    rows.sort(key=lambda row: (row["sample_id"], row["generation_seed"]))
    _write_json(directory / CONFIG_NAME, config)
    _write_jsonl(directory / SAMPLES_NAME, rows)
    _write_json(
        directory / GENERATION_SEAL_NAME,
        {
            "schema_version": GENERATION_SEAL_SCHEMA,
            "status": "complete_gpu_generation",
            "scientific_config_sha256": config_hash,
            "sample_records": 256,
            "audio_files": 256,
            "members": {
                CONFIG_NAME: artifact_member(directory / CONFIG_NAME),
                SAMPLES_NAME: artifact_member(directory / SAMPLES_NAME),
            },
            "audio_tree_sha256": sha256_tree(audio_root),
        },
    )
    return directory, rows


def _make_metric_chain(root, name, generation_dir, generation_rows, offsets):
    quality_dir = root / (name + "-quality")
    quality_dir.mkdir()
    generation_seal = sha256_file(generation_dir / GENERATION_SEAL_NAME)
    quality_provenance = {
        "schema_version": QUALITY_PROVENANCE_SCHEMA_VERSION,
        "status": "accepted_quality_evaluation",
        "metrics": list(QUALITY_METRICS),
        "generation_artifact_seal_sha256": generation_seal,
        "evaluators": {
            "muq_eval": _accepted_evaluator_identity("muq_eval"),
            "audiobox_aesthetics": _accepted_evaluator_identity(
                "audiobox_aesthetics"
            ),
        },
        "offline_environment": dict(OFFLINE_ENVIRONMENT),
    }
    _write_json(quality_dir / QUALITY_PROVENANCE_NAME, quality_provenance)
    quality_provenance_hash = sha256_file(quality_dir / QUALITY_PROVENANCE_NAME)
    quality_rows = []
    for index, source in enumerate(generation_rows):
        base_value = float(index // 2) / 100.0
        metrics = {
            metric: base_value + float(offsets.get(metric, 0.0))
            for metric in QUALITY_METRICS
        }
        quality_rows.append(
            {
                "schema_version": QUALITY_SCHEMA_VERSION,
                "sample_id": source["sample_id"],
                "generation_seed": source["generation_seed"],
                "prompt_sha256": source["prompt_sha256"],
                "condition_id": source["condition_id"],
                "audio_sha256": source["audio_sha256"],
                "scientific_config_sha256": source["scientific_config_sha256"],
                "evaluator_provenance_sha256": quality_provenance_hash,
                "metrics": metrics,
            }
        )
    _write_jsonl(quality_dir / QUALITY_SCORES_NAME, quality_rows)
    _write_json(
        quality_dir / "artifact_seal.json",
        {
            "schema_version": QUALITY_SEAL_SCHEMA_VERSION,
            "status": "complete_quality_evaluation",
            "record_count": 256,
            "generation_artifact_seal_sha256": generation_seal,
            "members": {
                QUALITY_SCORES_NAME: artifact_member(quality_dir / QUALITY_SCORES_NAME),
                QUALITY_PROVENANCE_NAME: artifact_member(
                    quality_dir / QUALITY_PROVENANCE_NAME
                ),
            },
        },
    )

    metric_dir = root / (name + "-metric")
    metric_dir.mkdir()
    quality_seal = sha256_file(quality_dir / "artifact_seal.json")
    metric_provenance = {
        "schema_version": METRIC_PROVENANCE_SCHEMA_VERSION,
        "status": "accepted_stage1_evaluation",
        "metrics": list(FINAL_METRICS),
        "generation_artifact_seal_sha256": generation_seal,
        "quality_artifact_seal_sha256": quality_seal,
        "evaluators": {
            "muq_eval": _accepted_evaluator_identity("muq_eval"),
            "audiobox_aesthetics": _accepted_evaluator_identity(
                "audiobox_aesthetics"
            ),
            "music_clap": _accepted_evaluator_identity("music_clap"),
        },
        "offline_environment": dict(OFFLINE_ENVIRONMENT),
        "fad_computed": False,
        "fad_role": "separate_nonselection_pipeline_check",
    }
    _write_json(metric_dir / METRIC_PROVENANCE_NAME, metric_provenance)
    metric_provenance_hash = sha256_file(metric_dir / METRIC_PROVENANCE_NAME)
    metric_rows = []
    for quality_row in quality_rows:
        metrics = dict(quality_row["metrics"])
        index = int(quality_row["sample_id"].split("-")[-1])
        metrics["music_clap"] = float(index) / 100.0 + float(
            offsets.get("music_clap", 0.0)
        )
        metric_rows.append(
            {
                **{key: quality_row[key] for key in (
                    "sample_id",
                    "generation_seed",
                    "prompt_sha256",
                    "condition_id",
                    "audio_sha256",
                    "scientific_config_sha256",
                )},
                "schema_version": METRIC_SCHEMA_VERSION,
                "evaluator_provenance_sha256": metric_provenance_hash,
                "metrics": metrics,
            }
        )
    _write_jsonl(metric_dir / METRIC_SCORES_NAME, metric_rows)
    _write_json(
        metric_dir / "artifact_seal.json",
        {
            "schema_version": METRIC_SEAL_SCHEMA_VERSION,
            "status": "complete_stage1_evaluation",
            "record_count": 256,
            "generation_artifact_seal_sha256": generation_seal,
            "quality_artifact_seal_sha256": quality_seal,
            "members": {
                METRIC_SCORES_NAME: artifact_member(metric_dir / METRIC_SCORES_NAME),
                METRIC_PROVENANCE_NAME: artifact_member(
                    metric_dir / METRIC_PROVENANCE_NAME
                ),
            },
        },
    )
    return quality_dir, metric_dir


class PilotManifestTests(unittest.TestCase):
    def test_manifest_is_deterministic_sealed_and_tamper_evident(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "dev.full.jsonl"
            _source_dev(source)
            artifact = root / "pilot"
            publish_pilot_eval_manifest(source, artifact)
            verified = verify_pilot_eval_manifest(artifact, source)
            self.assertEqual(verified["record_count"], 128)
            records = read_jsonl_strict(artifact / PILOT_OUTPUT_BASENAME)
            self.assertEqual(
                [row["sample_id"] for row in records],
                sorted(row["sample_id"] for row in records),
            )
            with (artifact / PILOT_OUTPUT_BASENAME).open("ab") as stream:
                stream.write(b"{}\n")
            with self.assertRaises(Stage1ArtifactError):
                verify_pilot_eval_manifest(artifact, source)

    def test_manifest_requires_exact_frozen_source_name_and_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrong_name = root / "other.jsonl"
            _source_dev(wrong_name)
            with self.assertRaises(Stage1ArtifactError):
                publish_pilot_eval_manifest(wrong_name, root / "bad-name")
            source = root / "dev.full.jsonl"
            _write_jsonl(
                source,
                [{"sample_id": str(index), "prompt": "p"} for index in range(299)],
            )
            with self.assertRaises(Stage1ArtifactError):
                publish_pilot_eval_manifest(source, root / "bad-count")


class LearningRateControlTests(unittest.TestCase):
    def test_cross_condition_generation_and_evaluator_drift_is_rejected(self):
        import ptc_opd.stage1_control as control

        config = {
            "model_id": "facebook/musicgen-small",
            "base_checkpoint": {"checkpoint_sha256": "a" * 64},
            "base_lm_state_sha256": "b" * 64,
            "audiocraft_source_sha256": "c" * 64,
            "cfg_decision": {"decision_file_sha256": "d" * 64},
            "sampling": {"top_k": 250},
            "precision": {"lm_generation_compute": "torch.bfloat16"},
            "runtime_identity": {
                "loaded_t5_identity": {"identity_sha256": "e" * 64},
                "precision_contract": {"decode": "torch.float32"},
            },
        }
        evaluators = {
            name: {"checkpoint_sha256": hashlib.sha256(name.encode()).hexdigest()}
            for name in ("muq_eval", "audiobox_aesthetics", "music_clap")
        }
        base = {
            "generation": {"scientific_config": copy.deepcopy(config)},
            "provenance": {"evaluators": copy.deepcopy(evaluators)},
        }
        candidate = copy.deepcopy(base)
        control._require_common_evaluation_anchors(base, candidate)
        candidate["generation"]["scientific_config"]["sampling"]["top_k"] = 100
        with self.assertRaises(Stage1ArtifactError):
            control._require_common_evaluation_anchors(base, candidate)
        candidate = copy.deepcopy(base)
        candidate["provenance"]["evaluators"]["music_clap"][
            "checkpoint_sha256"
        ] = "f" * 64
        with self.assertRaises(Stage1ArtifactError):
            control._require_common_evaluation_anchors(base, candidate)

    def _automatic_inputs(self, root):
        source = root / "dev.full.jsonl"
        _source_dev(source)
        eval_dir = root / "pilot"
        publish_pilot_eval_manifest(source, eval_dir)
        prompts = read_jsonl_strict(eval_dir / PILOT_OUTPUT_BASENAME)
        base_generation, base_rows = _make_generation(
            root,
            "base-generation",
            eval_dir,
            prompts,
            source_kind="base_no_cfg",
        )
        base_quality, base_metric = _make_metric_chain(
            root, "base", base_generation, base_rows, {}
        )
        candidates = []
        for index, (learning_rate, offset) in enumerate(
            zip(LR_GRID, (0.10, 0.30, 0.20))
        ):
            run, verification = _make_run(
                root,
                "lr-run-{}".format(index),
                "uniform100",
                learning_rate,
            )
            generation, rows = _make_generation(
                root,
                "lr-generation-{}".format(index),
                eval_dir,
                prompts,
                source_kind="trained_no_cfg",
                run_verification=verification,
                learning_rate=learning_rate,
                method="uniform100",
                checkpoint_step=500,
            )
            quality, metric = _make_metric_chain(
                root,
                "candidate-{}".format(index),
                generation,
                rows,
                {name: offset for name in FINAL_METRICS},
            )
            candidates.append(
                {
                    "run_dir": run,
                    "run_verification": verification,
                    "generation_dir": generation,
                    "quality_dir": quality,
                    "metric_dir": metric,
                }
            )
        return eval_dir, base_generation, base_quality, base_metric, candidates

    def test_automatic_summary_and_decision_select_highest_q(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = self._automatic_inputs(root)
            summary_dir = root / "summary"
            publish_lr_evaluation_summary(
                eval_manifest_dir=inputs[0],
                base_generation_dir=inputs[1],
                base_quality_dir=inputs[2],
                base_metric_dir=inputs[3],
                candidates=inputs[4],
                output_dir=summary_dir,
            )
            decision_dir = root / "decision"
            publish_lr_decision(summary_dir, decision_dir)
            decision = verify_lr_decision(decision_dir, summary_dir)
            self.assertEqual(decision["status"], "selected")
            self.assertEqual(decision["selected_learning_rate"], 3.0e-6)
            self.assertFalse(decision["rule"]["q_positive_required"])
            self.assertFalse(decision["rule"]["bootstrap_gate_used"])
            self.assertEqual(decision["training_lineage_anchor"], _lineage_anchor())

    def test_lr_summary_rejects_run_from_another_base_lineage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = self._automatic_inputs(root)
            bundle = inputs[4][0]
            manifest_path = bundle["run_dir"] / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["student_checkpoint_sha256"] = "f" * 64
            _write_json(manifest_path, manifest)
            bundle["run_verification"]["run_manifest_sha256"] = sha256_file(
                manifest_path
            )
            with self.assertRaises(Stage1ArtifactError):
                build_lr_evaluation_summary(
                    eval_manifest_dir=inputs[0],
                    base_generation_dir=inputs[1],
                    base_quality_dir=inputs[2],
                    base_metric_dir=inputs[3],
                    candidates=inputs[4],
                )

    def test_exact_q_tie_prefers_lower_lr_and_bad_q_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = self._automatic_inputs(root)
            summary = build_lr_evaluation_summary(
                eval_manifest_dir=inputs[0],
                base_generation_dir=inputs[1],
                base_quality_dir=inputs[2],
                base_metric_dir=inputs[3],
                candidates=inputs[4],
            )
            summary["candidates"][1]["component_delta_base_sd"] = copy.deepcopy(
                summary["candidates"][0]["component_delta_base_sd"]
            )
            summary["candidates"][1]["q_dev"] = summary["candidates"][0]["q_dev"]
            summary["candidates"][2]["component_delta_base_sd"] = {
                metric: 0.0 for metric in FINAL_METRICS
            }
            summary["candidates"][2]["q_dev"] = 0.0
            summary_dir = root / "tie-summary"
            publish_closed_json_artifact(
                summary_dir,
                report_name=LR_SUMMARY_BASENAME,
                report=summary,
                seal_schema=LR_SUMMARY_SEAL_SCHEMA,
                seal_status="complete",
            )
            self.assertEqual(build_lr_decision(summary_dir)["selected_learning_rate"], 1.0e-6)

            bad = copy.deepcopy(summary)
            bad["candidates"][0]["q_dev"] += 1.0
            bad_dir = root / "bad-summary"
            publish_closed_json_artifact(
                bad_dir,
                report_name=LR_SUMMARY_BASENAME,
                report=bad,
                seal_schema=LR_SUMMARY_SEAL_SCHEMA,
                seal_status="complete",
            )
            with self.assertRaises(Stage1ArtifactError):
                build_lr_decision(bad_dir)


class Ptc500ControlTests(unittest.TestCase):
    def test_frozen_warmup_schedule_steps_1_50_51(self):
        records = _metric_records("ptc50", 3.0e-6, steps=51)
        self.assertEqual(records[0]["learning_rate"], 3.0e-6 / 50.0)
        self.assertEqual(records[49]["learning_rate"], 3.0e-6)
        self.assertEqual(records[50]["learning_rate"], 3.0e-6)

    def _lr_artifacts(self, root, selected_lr=3.0e-6):
        # Structurally valid automatic-summary schema, sufficient for the
        # independent PTC consumer test.
        pilot_seal = "b" * 64
        from ptc_opd.stage1_control import _lr_scientific_config

        config = _lr_scientific_config(pilot_seal, _lineage_anchor())
        candidates = []
        for learning_rate in LR_GRID:
            delta = 1.0 if learning_rate == selected_lr else 0.0
            candidates.append(
                {
                    "learning_rate": learning_rate,
                    "q_dev": delta,
                    "component_delta_base_sd": {
                        "muq_mi": delta,
                        "audiobox_ce": delta,
                        "audiobox_pq": delta,
                        "music_clap": 0.0,
                    },
                    "training_run_verified": True,
                    "generation_complete": True,
                    "evaluation_complete": True,
                    "finite_metrics": True,
                    "stable_optimization": True,
                    "artifacts": {
                        name: hashlib.sha256(
                            (name + str(learning_rate)).encode()
                        ).hexdigest()
                        for name in (
                            "training_run_manifest_sha256",
                            "training_run_seal_sha256",
                            "training_done_sha256",
                            "final_checkpoint_sha256",
                            "generation_artifact_seal_sha256",
                            "quality_artifact_seal_sha256",
                            "clap_artifact_seal_sha256",
                        )
                    },
                }
            )
        summary = {
            "schema_version": "ptc-opd-lr-evaluation-summary-v1",
            "status": "complete",
            "scientific_config": config,
            "scientific_config_sha256": canonical_json_sha256(config),
            "pilot_eval_manifest_artifact_seal_sha256": pilot_seal,
            "training_lineage_anchor": _lineage_anchor(),
            "base_artifacts": {
                name: hashlib.sha256(name.encode()).hexdigest()
                for name in (
                    "generation_artifact_seal_sha256",
                    "quality_artifact_seal_sha256",
                    "clap_artifact_seal_sha256",
                )
            },
            "model": "facebook/musicgen-small",
            "method": "uniform100",
            "train_seed": 2027,
            "optimizer_steps": 500,
            "decision_checkpoint": 500,
            "prompt_count": 128,
            "generation_seeds": [31001, 31002],
            "base_standardization": {
                metric: {"mean": 0.0, "sample_sd": 1.0}
                for metric in FINAL_METRICS
            },
            "candidates": candidates,
        }
        summary_dir = root / "summary"
        publish_closed_json_artifact(
            summary_dir,
            report_name=LR_SUMMARY_BASENAME,
            report=summary,
            seal_schema=LR_SUMMARY_SEAL_SCHEMA,
            seal_status="complete",
        )
        decision_dir = root / "decision"
        publish_lr_decision(summary_dir, decision_dir)
        return summary_dir, decision_dir

    def test_ptc500_checks_warmup_memory_and_seals_decision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary_dir, decision_dir = self._lr_artifacts(root)
            b1_dir, b1_verification = _make_b1_prestability_stub(root)
            run, verification = _make_run(root, "ptc", "ptc50", 3.0e-6)
            report = build_ptc500_stability_report(
                run,
                decision_dir,
                summary_dir,
                verification,
                b1_dir,
                b1_verification,
            )
            self.assertTrue(report["gate_passed"])
            self.assertFalse(report["full_b1_passed"])
            artifact = root / "ptc-decision"
            publish_ptc500_stability_report(
                run,
                decision_dir,
                summary_dir,
                verification,
                b1_dir,
                b1_verification,
                artifact,
            )
            verified = verify_ptc500_stability_report(
                artifact,
                run,
                decision_dir,
                summary_dir,
                verification,
                b1_dir,
                b1_verification,
            )
            self.assertEqual(
                verified["required_action"], "publish_combined_b1_closure"
            )
            self.assertTrue((artifact / PTC500_REPORT_BASENAME).is_file())
            closure_dir = root / "b1-full-closure"
            publish_b1_full_closure(
                artifact, b1_dir, b1_verification, closure_dir
            )
            closure = verify_b1_full_closure(
                closure_dir, artifact, b1_dir, b1_verification
            )
            self.assertTrue(closure["full_b1_passed"])
            self.assertEqual(closure["required_action"], "auto_continue_small_pilot")
            self.assertTrue((closure_dir / B1_FULL_CLOSURE_BASENAME).is_file())
            _write_json(
                b1_dir / "b1_prestability_summary.json",
                {"scientific_status": "tampered"},
            )
            with self.assertRaises(Stage1ArtifactError):
                verify_b1_full_closure(
                    closure_dir, artifact, b1_dir, b1_verification
                )

    def test_ptc500_rejects_wrong_warmup_and_fails_low_memory_margin(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary_dir, decision_dir = self._lr_artifacts(root)
            b1_dir, b1_verification = _make_b1_prestability_stub(root)
            run, verification = _make_run(
                root, "ptc-low-memory", "ptc50", 3.0e-6, reserved=96
            )
            report = build_ptc500_stability_report(
                run,
                decision_dir,
                summary_dir,
                verification,
                b1_dir,
                b1_verification,
            )
            self.assertFalse(report["gate_passed"])
            self.assertFalse(report["gates"]["peak_memory_margin_gte_0_05"])

            metrics = run / "logs" / "attempt-0000.metrics.jsonl"
            records = read_jsonl_strict(metrics)
            records[0]["learning_rate"] = 3.0e-6
            _write_jsonl(metrics, records)
            sealed = json.loads((run / "SEALED.json").read_text())
            sealed["attempt_logs"][0]["metrics_sha256"] = sha256_file(metrics)
            _write_json(run / "SEALED.json", sealed)
            verification["SEALED.json_sha256"] = sha256_file(run / "SEALED.json")
            with self.assertRaises(Stage1ArtifactError):
                build_ptc500_stability_report(
                    run,
                    decision_dir,
                    summary_dir,
                    verification,
                    b1_dir,
                    b1_verification,
                )

    def test_ptc500_rejects_cross_stage_base_swap_and_zero_memory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary_dir, decision_dir = self._lr_artifacts(root)
            b1_dir, b1_verification = _make_b1_prestability_stub(root)
            run, verification = _make_run(
                root, "ptc-swapped", "ptc50", 3.0e-6
            )
            manifest_path = run / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["student_checkpoint_sha256"] = "f" * 64
            _write_json(manifest_path, manifest)
            verification["run_manifest_sha256"] = sha256_file(manifest_path)
            with self.assertRaises(Stage1ArtifactError):
                build_ptc500_stability_report(
                    run,
                    decision_dir,
                    summary_dir,
                    verification,
                    b1_dir,
                    b1_verification,
                )

            zero_run, zero_verification = _make_run(
                root, "ptc-zero-memory", "ptc50", 3.0e-6, reserved=0
            )
            with self.assertRaises(Stage1ArtifactError):
                build_ptc500_stability_report(
                    zero_run,
                    decision_dir,
                    summary_dir,
                    zero_verification,
                    b1_dir,
                    b1_verification,
                )


class SmallPilotDecisionTests(unittest.TestCase):
    def _rows(self, offset):
        rows = []
        for index in range(128):
            for seed in (31001, 31002):
                base = float(index) / 100.0
                rows.append(
                    {
                        "sample_id": "pilot-{:03d}".format(index),
                        "generation_seed": seed,
                        "metrics": {
                            metric: base + offset for metric in FINAL_METRICS
                        },
                    }
                )
        return rows

    def _summary(
        self,
        disagreement_offset=0.30,
        *,
        base_lineage=None,
        method_lineage_overrides=None,
    ):
        import ptc_opd.stage1_control as control

        offsets = {
            "uniform100": 0.20,
            "codebook100": 0.22,
            "random50": 0.25,
            "prefix50": 0.15,
            "disagreement50": disagreement_offset,
            "ptc50": 0.22,
        }
        methods = {}
        for index, method in enumerate(SMALL_PILOT_METHODS):
            methods[method] = {
                "metric_rows": self._rows(offsets[method]),
                "mert_diversity_mean_cosine_distance": (
                    0.40 if method == "ptc50" else 0.30
                ),
                "fad_pipeline_check_passed": True,
                "fad_scores": {
                    "clap-laion-music": 1.0 + index,
                    "MERT-v1-95M-layer12": 2.0 + index,
                },
                "stable_optimization": True,
                "training_lineage_anchor": copy.deepcopy(
                    (method_lineage_overrides or {}).get(method, _lineage_anchor())
                ),
                "artifacts": {
                    name: hashlib.sha256((method + name).encode()).hexdigest()
                    for name in SMALL_PILOT_ARTIFACT_FIELDS
                },
            }
        return control._build_small_pilot_summary_from_verified(
            pilot_manifest_seal_sha256="d" * 64,
            lr_decision_seal_sha256="e" * 64,
            selected_learning_rate=3.0e-6,
            base_rows=self._rows(0.0),
            lr_base_artifacts={
                name: hashlib.sha256(name.encode()).hexdigest()
                for name in (
                    "generation_artifact_seal_sha256",
                    "quality_artifact_seal_sha256",
                    "clap_artifact_seal_sha256",
                )
            },
            base_artifacts={
                name: hashlib.sha256(name.encode()).hexdigest()
                for name in (
                    "generation_artifact_seal_sha256",
                    "quality_artifact_seal_sha256",
                    "clap_artifact_seal_sha256",
                )
            },
            lr_training_lineage_anchor=_lineage_anchor(),
            base_training_lineage_anchor=(
                _lineage_anchor() if base_lineage is None else base_lineage
            ),
            methods=methods,
        )

    def test_automatic_summary_decision_passes_and_prefix_stays_diagnostic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary_dir = root / "summary"
            publish_closed_json_artifact(
                summary_dir,
                report_name=SMALL_PILOT_SUMMARY_BASENAME,
                report=self._summary(),
                seal_schema=SMALL_PILOT_SUMMARY_SEAL_SCHEMA,
                seal_status="complete",
            )
            decision = build_small_pilot_decision(summary_dir)
            self.assertTrue(decision["gate_passed"])
            self.assertEqual(
                decision["required_action"],
                "prepare_medium_scale_gate_workpack_do_not_launch_medium",
            )
            self.assertEqual(decision["primary_pilot_methods"], list(PRIMARY_PILOT_METHODS))
            self.assertNotIn("prefix50", decision["primary_pilot_methods"])
            self.assertFalse(decision["prefix50"]["included_in_medium_primary_matrix"])
            decision_dir = root / "decision"
            publish_small_pilot_decision(summary_dir, decision_dir)
            self.assertTrue(
                verify_small_pilot_decision(decision_dir, summary_dir)["gate_passed"]
            )

    def test_prefix_diagnostic_gate_can_stop_but_never_becomes_primary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary_dir = root / "summary"
            publish_closed_json_artifact(
                summary_dir,
                report_name=SMALL_PILOT_SUMMARY_BASENAME,
                report=self._summary(disagreement_offset=0.10),
                seal_schema=SMALL_PILOT_SUMMARY_SEAL_SCHEMA,
                seal_status="complete",
            )
            decision = build_small_pilot_decision(summary_dir)
            self.assertFalse(decision["gate_passed"])
            self.assertFalse(decision["gates"]["disagreement_q_dev_gt_prefix"])
            self.assertNotIn("prefix50", decision["primary_pilot_methods"])

    def test_small_summary_rejects_swapped_base_or_method_lineage(self):
        swapped = _lineage_anchor()
        swapped["base_checkpoint"]["checkpoint_sha256"] = "f" * 64
        with self.assertRaises(Stage1ArtifactError):
            self._summary(base_lineage=swapped)
        with self.assertRaises(Stage1ArtifactError):
            self._summary(method_lineage_overrides={"ptc50": swapped})


class ControllerTests(unittest.TestCase):
    def test_plan_graph_is_unique_and_prefix_is_diagnostic_only(self):
        contract_path = ROOT / "configs" / "stage1_autonomy_contract.json"
        contract = load_autonomy_contract(contract_path)
        plan = controller_plan(contract)
        ids = [node["id"] for node in plan["nodes"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(plan["primary_pilot_methods"], list(PRIMARY_PILOT_METHODS))
        self.assertNotIn("prefix50", plan["primary_pilot_methods"])
        self.assertEqual(
            plan["four_machine_allocation"]["lr_uniform_sweep"]["node-3"],
            "base-generation-and-evaluation-or-spare",
        )
        self.assertIn(
            "prefix50",
            plan["four_machine_allocation"]["small_pilot_wave_2"]["node-1"],
        )
        self.assertFalse(
            plan["small_only_temporal_diagnostics"]["prefix50"][
                "included_in_primary_pilot_decision"
            ]
        )

    def test_graph_validator_rejects_duplicate_and_forward_dependency(self):
        import ptc_opd.stage1_control as control

        contract = load_autonomy_contract(
            ROOT / "configs" / "stage1_autonomy_contract.json"
        )
        duplicate = control.STAGE_NODES + (dict(control.STAGE_NODES[1]),)
        with patch.object(control, "STAGE_NODES", duplicate):
            with self.assertRaises(Stage1ArtifactError):
                controller_plan(contract)
        forward = [dict(node) for node in control.STAGE_NODES]
        forward[0] = {**forward[0], "depends_on": [forward[-1]["id"]]}
        with patch.object(control, "STAGE_NODES", tuple(forward)):
            with self.assertRaises(Stage1ArtifactError):
                controller_plan(contract)

    def test_preflight_counts_only_registered_present_mert_fad_implementation(self):
        contract_path = ROOT / "configs" / "stage1_autonomy_contract.json"
        preflight = controller_preflight(ROOT, contract_path)
        self.assertEqual(
            preflight["workpack_source_integrity"]["status"],
            "verified_closed_world",
        )
        self.assertTrue(preflight["primary_methods_exact"])
        self.assertTrue(preflight["registered_modes_exact"])
        self.assertTrue(preflight["prefix50"]["registered_in_training_runner"])
        self.assertTrue(preflight["capabilities"]["mert_evaluator"]["ready"])
        self.assertTrue(preflight["capabilities"]["fad_evaluator"]["ready"])
        self.assertFalse(preflight["cfg_only_evaluators_count_as_trained_evaluators"])

    def test_preflight_rejects_missing_transitive_source_via_closed_world_seal(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "workpack"
            shutil.copytree(ROOT, copied)
            (copied / "scripts" / "cfg_eval_common.py").unlink()
            with self.assertRaisesRegex(
                Stage1ArtifactError, "source-integrity verification failed"
            ):
                controller_preflight(
                    copied,
                    copied / "configs" / "stage1_autonomy_contract.json",
                )

    def test_next_action_obeys_dependencies_and_missing_capabilities(self):
        import ptc_opd.stage1_control as control
        from ptc_opd.stage1_controller_ledger import stage_verifier_catalog

        contract_path = ROOT / "configs" / "stage1_autonomy_contract.json"
        contract = load_autonomy_contract(contract_path)
        catalog = stage_verifier_catalog()
        preflight = {
            "schema_version": CONTROLLER_PREFLIGHT_SCHEMA,
            "contract_sha256": sha256_file(contract_path),
            "workpack_source_integrity": {
                "status": "verified_closed_world",
                "manifest_sha256": sha256_file(ROOT / "WORKPACK_MANIFEST.sha256"),
            },
            "all_stage_verifiers_supported": True,
            "stage_verifier_count": len(control.STAGE_NODES),
            "stage_verifiers": catalog,
            "capabilities": {
                name: {"ready": True} for name in contract["capabilities"]
            },
        }
        ledger = {
            "schema_version": "ptc-opd-stage1-controller-authority-ledger-v1",
            "revision": 0,
            "scientific_config_sha256": canonical_json_sha256(contract),
            "workpack_manifest_sha256": preflight["workpack_source_integrity"][
                "manifest_sha256"
            ],
            "stages": {
                node["id"]: {
                    "status": "pending",
                    "yellow_retry_count": 0,
                    "attempts": [],
                    "terminal_receipt": None,
                    "stop_reason": None,
                }
                for node in control.STAGE_NODES
            },
        }

        def normalized():
            return {
                "ledger": copy.deepcopy(ledger),
                "legacy_stages": {
                    stage_id: {
                        "status": entry["status"],
                        "artifact_sha256": (
                            None if entry["status"] == "pending" else "a" * 64
                        ),
                    }
                    for stage_id, entry in ledger["stages"].items()
                },
            }

        with patch(
            "ptc_opd.stage1_controller_ledger.validate_current_controller_ledger_authorizations",
            side_effect=lambda *args, **kwargs: normalized(),
        ):
            action = controller_next_action(
                contract,
                preflight,
                Path("/sealed-ledger"),
                workpack_root=ROOT,
                contract_path=contract_path,
            )
            self.assertEqual(action["required_action"], "await_external_evidence")

            for stage_id in ("t5_closure", "retained_seals_audit"):
                ledger["stages"][stage_id]["status"] = "passed"
            action = controller_next_action(
                contract,
                preflight,
                Path("/sealed-ledger"),
                workpack_root=ROOT,
                contract_path=contract_path,
            )
            self.assertEqual(
                set(action["runnable_stages"]),
                {"pilot_eval_manifest", "b1_prestability"},
            )

            for stage_id in ("pilot_eval_manifest", "b1_prestability"):
                ledger["stages"][stage_id]["status"] = "passed"
            preflight["capabilities"]["base_teacher_generation"]["ready"] = False
            action = controller_next_action(
                contract,
                preflight,
                Path("/sealed-ledger"),
                workpack_root=ROOT,
                contract_path=contract_path,
            )
            self.assertIn("performance_benchmark", action["runnable_stages"])
            self.assertEqual(
                action["missing_capability_stages"],
                [
                    {
                        "stage": "evaluation_pipeline_qualification",
                        "capabilities": ["base_teacher_generation"],
                    }
                ],
            )

    def test_next_action_rejects_stale_ledger_revision(self):
        import ptc_opd.stage1_control as control
        from ptc_opd.stage1_controller_ledger import stage_verifier_catalog

        contract_path = ROOT / "configs" / "stage1_autonomy_contract.json"
        contract = load_autonomy_contract(contract_path)
        preflight = {
            "schema_version": CONTROLLER_PREFLIGHT_SCHEMA,
            "contract_sha256": sha256_file(contract_path),
            "workpack_source_integrity": {
                "status": "verified_closed_world",
                "manifest_sha256": sha256_file(ROOT / "WORKPACK_MANIFEST.sha256"),
            },
            "all_stage_verifiers_supported": True,
            "stage_verifier_count": len(control.STAGE_NODES),
            "stage_verifiers": stage_verifier_catalog(),
            "capabilities": {
                name: {"ready": True} for name in contract["capabilities"]
            },
        }
        with patch(
            "ptc_opd.stage1_controller_ledger.validate_current_controller_ledger_authorizations",
            side_effect=Stage1ArtifactError("supplied ledger is stale"),
        ):
            with self.assertRaisesRegex(Stage1ArtifactError, "stale"):
                controller_next_action(
                    contract,
                    preflight,
                    Path("/old-ledger-revision"),
                    workpack_root=ROOT,
                    contract_path=contract_path,
                )


if __name__ == "__main__":
    unittest.main()
