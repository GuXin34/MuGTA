"""CPU/fake tests for the standalone Stage-1 runner and pure utilities.

These tests do not claim AudioCraft, NCCL, BF16, or H20 execution coverage.
Those are explicit remote-machine gates in the runbook.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch import nn

from ptc_opd.sampling import DeterministicDistributedBatchSampler
from ptc_opd.train_utils import (
    CHECKPOINT_SCHEMA_VERSION,
    MODE_SPECS,
    Stage1Config,
    build_checkpoint_metadata,
    canonical_json_sha256,
    hash_module_state,
    learning_rate_for_update,
    load_prompt_manifest,
    progress_from_optimizer_step,
    resolve_mode,
    sha256_path,
    validate_config,
    verify_resume_metadata,
)


def _load_runner_module():
    path = ROOT / "scripts" / "train_stage1.py"
    spec = importlib.util.spec_from_file_location("train_stage1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import train_stage1.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner_module()


def _ddp_reducer_contract():
    contract = {
        "schema_version": RUNNER.DDP_REDUCER_SCHEMA_VERSION,
        "policy_id": RUNNER.DDP_REDUCER_POLICY_ID,
        "torch_version": "2.1.0+cu121",
        "torch_cuda_runtime": "12.1",
        "find_unused_parameters": True,
        "static_graph": False,
        "gradient_as_bucket_view": False,
        "bucket_cap_bytes": 25 * 1024 * 1024,
        "has_rebuilt_buckets": False,
        "bucket_sizes": "16",
        "bucket_count": 1,
        "trainable_parameter_tensor_count": 1,
        "trainable_parameter_numel": 4,
        "trainable_gradient_bytes": 16,
        "parameter_layout_sha256": "e" * 64,
    }
    contract["identity_sha256"] = canonical_json_sha256(contract)
    return contract


def _prepare_terminal_run(
    run: Path,
    *,
    final_config_sha256: str = "",
    final_ddp_reducer_identity: str = "",
    omit_final_ddp_reducer_identity: bool = False,
):
    teacher_digest = "a" * 64
    config = {"max_optimizer_steps": 7, "mode": "uniform100"}
    config_digest = canonical_json_sha256(config)
    reducer_contract = _ddp_reducer_contract()
    reducer_identity = reducer_contract["identity_sha256"]
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    (run / "logs").mkdir()
    RUNNER.write_json_exclusive(
        run / "run_manifest.json",
        {
            "schema_version": RUNNER.RUN_SCHEMA_VERSION,
            "config": config,
            "config_sha256": config_digest,
            "teacher_state_sha256_initial": teacher_digest,
            "runtime_contract_by_rank": [
                {"rank": rank, "ddp_reducer": dict(reducer_contract)}
                for rank in range(8)
            ],
        },
    )
    RUNNER.write_json_exclusive(
        run / "status.json", {"status": "running", "optimizer_step": 0}
    )

    def metadata(step: int, digest: str, ddp_identity: str = reducer_identity):
        result = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "config_sha256": digest,
            "teacher_state_sha256_initial": teacher_digest,
            "optimizer_step": step,
            "global_microstep": 4 * step,
        }
        if not (omit_final_ddp_reducer_identity and step == 7):
            result["ddp_reducer_identity_sha256"] = ddp_identity
        return result

    initial = RUNNER.commit_checkpoint_payload(
        checkpoints,
        0,
        {"metadata": metadata(0, config_digest), "payload": torch.tensor([0])},
    )
    metrics = RUNNER.begin_attempt_log(
        run,
        config_sha256=config_digest,
        start_checkpoint=initial,
        start_optimizer_step=0,
        start_global_microstep=0,
        is_resume=False,
    )
    RUNNER.jsonl_append(
        metrics,
        {
            "schema_version": RUNNER.RUN_SCHEMA_VERSION,
            "event": "optimizer_step",
            "optimizer_step": 7,
            "global_microstep": 28,
        },
    )
    final_digest = final_config_sha256 or config_digest
    final_reducer_identity = final_ddp_reducer_identity or reducer_identity
    final = RUNNER.commit_checkpoint_payload(
        checkpoints,
        7,
        {
            "metadata": metadata(7, final_digest, final_reducer_identity),
            "payload": torch.tensor([1]),
        },
    )
    return config_digest, teacher_digest, initial, final, metrics


def _config(directory: Path, **overrides) -> Stage1Config:
    digest = "a" * 64
    values = dict(
        manifest=str(directory / "train.full.jsonl"),
        student_checkpoint=str(directory / "student.pt"),
        teacher_checkpoint=str(directory / "teacher.pt"),
        audiocraft_root=str(directory / "audiocraft"),
        output_dir=str(directory / "run"),
        cfg_scale_decision_dir=str(directory / "cfg-decision"),
        cfg_scale_decision_file_sha256=digest,
        cfg_scale_decision_payload_sha256=digest,
        cfg_scale_scientific_config_sha256=digest,
        cfg_generation_checkpoint_sha256=digest,
        cfg_generation_audiocraft_source_sha256=digest,
        cfg_generation_loaded_t5_identity_sha256=digest,
        cfg_generation_state_dict_sha256=digest,
        mode="uniform100",
        seed=2027,
        learning_rate=3.0e-6,
        weight_decay=0.0,
        max_optimizer_steps=100,
        save_every=25,
        log_every=1,
    )
    values.update(overrides)
    return Stage1Config(**values)


class Stage1ModeAndConfigTest(unittest.TestCase):
    def test_exact_five_condition_mapping(self) -> None:
        expected = {
            "uniform100": ("uniform", 1.0, False),
            "codebook100": ("codebook_only", 1.0, True),
            "random50": ("random_stratified", 0.5, False),
            "prefix50": ("prefix", 0.5, False),
            "disagreement50": ("disagreement", 0.5, False),
            "ptc50": ("ptc", 0.5, True),
        }
        self.assertEqual(set(MODE_SPECS), set(expected))
        for condition, fields in expected.items():
            spec = resolve_mode(condition)
            self.assertEqual(
                (spec.loss_mode, spec.rho, spec.requires_perceptual_prior), fields
            )
        with self.assertRaisesRegex(ValueError, "unknown Stage-1 mode"):
            resolve_mode("not-a-method")

    def test_frozen_batch_duration_and_kl_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validate_config(_config(root))
            invalid = (
                {"rank_batch_size": 1},
                {"expected_world_size": 32},
                {"grad_accum_steps": 2},
                {"effective_global_batch": 32},
                {"token_frames": 499},
                {"kl_direction": "reverse"},
                {"weight_decay": 0.01},
                {"warmup_optimizer_steps": 49},
                {"denominator_rtol": 1.0e-5},
                {"seed": -1},
                {"random_mask_namespace": -1},
            )
            for override in invalid:
                with self.subTest(override=override), self.assertRaises(ValueError):
                    validate_config(_config(root, **override))

    def test_progress_math(self) -> None:
        self.assertEqual(progress_from_optimizer_step(0, 4), 0)
        self.assertEqual(progress_from_optimizer_step(250, 4), 1000)
        with self.assertRaises(ValueError):
            progress_from_optimizer_step(-1, 4)

    def test_frozen_warmup_is_horizon_independent(self) -> None:
        base = 1.0e-5
        self.assertAlmostEqual(learning_rate_for_update(base, 0), 2.0e-7)
        self.assertAlmostEqual(learning_rate_for_update(base, 49), base)
        self.assertAlmostEqual(learning_rate_for_update(base, 50), base)
        self.assertAlmostEqual(learning_rate_for_update(base, 2999), base)


class ManifestAndSamplerTest(unittest.TestCase):
    def test_manifest_validation_and_resume_first_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.full.jsonl"
            rows = [
                {"sample_id": "sample-{:03d}".format(index), "prompt": "p {}".format(index)}
                for index in range(80)
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            records = load_prompt_manifest(path)
            sample_ids = [record.sample_id for record in records]
            sampler = DeterministicDistributedBatchSampler(
                len(records), 2027, 3, 8, 2, sample_ids=sample_ids
            )
            # Step 7 resumes at global microstep 28 for accum=4.
            uninterrupted = sampler.batch(28)
            resumed = sampler.batch(progress_from_optimizer_step(7, 4))
            self.assertEqual(uninterrupted, resumed)
            self.assertEqual(len(uninterrupted.indices), 2)

    def test_manifest_rejects_wrong_name_blank_duplicate_and_bad_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrong = root / "shard-of-4.node-0.jsonl"
            wrong.write_text('{"sample_id":"a","prompt":"p"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "train.full.jsonl"):
                load_prompt_manifest(wrong)

            path = root / "train.full.jsonl"
            path.write_text(
                '{"sample_id":"a","prompt":"p"}\n\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "blank"):
                load_prompt_manifest(path)
            path.write_text(
                '{"sample_id":"a","prompt":"p"}\n'
                '{"sample_id":"a","prompt":"q"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_prompt_manifest(path)
            path.write_text("not-json\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                load_prompt_manifest(path)


class CheckpointMetadataTest(unittest.TestCase):
    def test_checkpoint_hash_tree_is_stable_and_path_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "checkpoint"
            first.mkdir()
            (first / "a.bin").write_bytes(b"abc")
            (first / "nested").mkdir()
            (first / "nested" / "b.bin").write_bytes(b"def")
            digest = sha256_path(first)
            self.assertEqual(digest, sha256_path(first))
            (first / "nested" / "b.bin").write_bytes(b"changed")
            self.assertNotEqual(digest, sha256_path(first))

    def test_metadata_verification_catches_manifest_and_teacher_change(self) -> None:
        hex_a = "a" * 64
        metadata = build_checkpoint_metadata(
            config_hash=hex_a,
            manifest_hash="b" * 64,
            student_checkpoint_hash="c" * 64,
            teacher_checkpoint_hash="d" * 64,
            optimizer_step=9,
            global_microstep=36,
            teacher_state_hash_initial="e" * 64,
            ddp_reducer_identity_hash="f" * 64,
        )
        self.assertEqual(metadata["schema_version"], CHECKPOINT_SCHEMA_VERSION)
        verify_resume_metadata(metadata, metadata)
        wrong = dict(metadata)
        wrong["manifest_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "manifest_sha256"):
            verify_resume_metadata(metadata, wrong)
        wrong = dict(metadata)
        wrong["teacher_state_sha256_initial"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "teacher_state"):
            verify_resume_metadata(metadata, wrong)
        wrong = dict(metadata)
        wrong["ddp_reducer_identity_sha256"] = "1" * 64
        with self.assertRaisesRegex(ValueError, "ddp_reducer_identity"):
            verify_resume_metadata(metadata, wrong)
        old_schema = dict(metadata)
        old_schema["schema_version"] = "ptc-opd-stage1-checkpoint-v2"
        with self.assertRaisesRegex(ValueError, "schema_version"):
            verify_resume_metadata(old_schema, metadata)

    def test_module_hash_detects_teacher_mutation(self) -> None:
        module = nn.Linear(3, 2)
        before = hash_module_state(module)
        with torch.no_grad():
            module.weight[0, 0].add_(1.0)
        self.assertNotEqual(before, hash_module_state(module))

    def test_config_hash_is_order_independent(self) -> None:
        self.assertEqual(
            canonical_json_sha256({"a": 1, "b": 2}),
            canonical_json_sha256({"b": 2, "a": 1}),
        )


class RunnerFakeTest(unittest.TestCase):
    def test_fixed_ddp_reducer_audit_rejects_rebuild(self) -> None:
        model = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))
        gradient_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        class FakeDDP:
            module = model
            find_unused_parameters = True
            static_graph = False
            gradient_as_bucket_view = False
            bucket_bytes_cap = 25 * 1024 * 1024
            _has_rebuilt_buckets = False

            @staticmethod
            def _get_ddp_logging_data():
                return {"bucket_sizes": str(gradient_bytes)}

        audit = RUNNER.ddp_reducer_audit(FakeDDP())
        self.assertEqual(
            audit["policy_id"], "torch-2.1-fixed-initial-buckets-v1"
        )
        self.assertFalse(audit["has_rebuilt_buckets"])
        self.assertEqual(audit["trainable_gradient_bytes"], gradient_bytes)
        self.assertEqual(audit["torch_version"], str(torch.__version__))
        self.assertEqual(audit["torch_cuda_runtime"], str(torch.version.cuda))

        rebuilt = FakeDDP()
        rebuilt._has_rebuilt_buckets = True
        with self.assertRaisesRegex(RuntimeError, "DDP reducer contract differs"):
            RUNNER.ddp_reducer_audit(rebuilt)

        with mock.patch.dict(
            RUNNER.os.environ, {"PTC_NODE3_GATE": "1"}, clear=False
        ), mock.patch.object(RUNNER.torch, "__version__", "9.9.0"):
            with self.assertRaisesRegex(RuntimeError, "DDP reducer contract differs"):
                RUNNER.ddp_reducer_audit(FakeDDP())

    def test_metadata_mmap_loader_passes_plain_string_filename(self) -> None:
        observed = {}

        def torch_21_load(f, map_location=None, weights_only=None, mmap=None):
            observed.update(
                {
                    "filename": f,
                    "map_location": map_location,
                    "weights_only": weights_only,
                    "mmap": mmap,
                }
            )
            if type(f) is not str:
                raise ValueError(
                    "f must be a string filename in order to use mmap argument"
                )
            return {"metadata": {"optimizer_step": 7}}

        with mock.patch.object(RUNNER.torch, "load", new=torch_21_load):
            metadata = RUNNER._load_checkpoint_metadata(Path("checkpoint.pt"))

        self.assertIs(type(observed["filename"]), str)
        self.assertEqual(observed["map_location"], "cpu")
        self.assertIs(observed["weights_only"], False)
        self.assertIs(observed["mmap"], True)
        self.assertEqual(metadata, {"optimizer_step": 7})

    def test_checkpoint_and_terminal_commits_are_directory_atomic_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            _, teacher_digest, _, directory, metrics = _prepare_terminal_run(run)
            checkpoints = run / "checkpoints"
            self.assertEqual(
                {item.name for item in directory.iterdir()},
                {"checkpoint.pt", "SHA256.json"},
            )
            checkpoint, sidecar = RUNNER.checkpoint_paths(directory)
            self.assertEqual(
                json.loads(sidecar.read_text(encoding="utf-8"))["sha256"],
                RUNNER.sha256_file(checkpoint),
            )
            with self.assertRaises(FileExistsError):
                RUNNER.commit_checkpoint_payload(checkpoints, 7, {"other": 1})

            failed = run / "FAILED.json"
            failed.write_text('{"status":"failed"}\n', encoding="utf-8")
            done = RUNNER.commit_success_terminal(run, 7, teacher_digest)
            self.assertTrue(done["failed_record_superseded"])
            self.assertEqual(
                done, RUNNER.commit_success_terminal(run, 7, teacher_digest)
            )
            self.assertTrue((run / "DONE.json").is_file())
            sealed = json.loads((run / "SEALED.json").read_text(encoding="utf-8"))
            self.assertEqual(
                sealed["superseded_failure_sha256"], RUNNER.sha256_file(failed)
            )
            self.assertEqual(sealed["attempt_log_schema_version"], RUNNER.ATTEMPT_SCHEMA_VERSION)
            self.assertEqual(len(sealed["attempt_logs"]), 1)
            self.assertEqual(
                sealed["attempt_logs"][0]["metrics_sha256"],
                RUNNER.sha256_file(metrics),
            )
            self.assertEqual(
                sealed["run_manifest"]["sha256"],
                RUNNER.sha256_file(run / "run_manifest.json"),
            )
            self.assertEqual(
                sealed["run_manifest"]["size_bytes"],
                (run / "run_manifest.json").stat().st_size,
            )
            self.assertEqual(len(sealed["checkpoint_inventory"]), 2)
            self.assertEqual(
                sealed["final_checkpoint"]["metadata_config_sha256"],
                sealed["run_manifest"]["config_sha256"],
            )
            self.assertEqual(
                sealed["final_checkpoint"][
                    "metadata_ddp_reducer_identity_sha256"
                ],
                sealed["run_manifest"]["ddp_reducer_identity_sha256"],
            )
            verified = RUNNER.verify_sealed_stage1_run(run)
            self.assertEqual(verified["status"], "verified")
            self.assertEqual(verified["checkpoint_count"], 2)
            attempt_metadata = run / "logs" / "attempt-0000.json"
            metadata_payload = json.loads(
                attempt_metadata.read_text(encoding="utf-8")
            )
            attempt_metadata.write_text(
                json.dumps(metadata_payload, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "SEALED.json differs"):
                RUNNER.commit_success_terminal(run, 7, teacher_digest)

    def test_terminal_seal_rejects_checkpoint_config_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            _, teacher_digest, _, _, _ = _prepare_terminal_run(
                run, final_config_sha256="f" * 64
            )
            with self.assertRaisesRegex(
                ValueError, "metadata config_sha256 differs from run_manifest"
            ):
                RUNNER.commit_success_terminal(run, 7, teacher_digest)
            self.assertFalse((run / "SEALED.json").exists())
            self.assertFalse((run / "DONE.json").exists())

    def test_terminal_seal_rejects_missing_or_mismatched_ddp_identity(self) -> None:
        cases = (
            ({"omit_final_ddp_reducer_identity": True}, "ddp_reducer_identity"),
            ({"final_ddp_reducer_identity": "1" * 64}, "DDP reducer identity"),
        )
        for options, message in cases:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as temporary:
                run = Path(temporary) / "run"
                _, teacher_digest, _, _, _ = _prepare_terminal_run(run, **options)
                with self.assertRaisesRegex(ValueError, message):
                    RUNNER.commit_success_terminal(run, 7, teacher_digest)
                self.assertFalse((run / "SEALED.json").exists())
                self.assertFalse((run / "DONE.json").exists())

    def test_copied_run_cli_is_closed_world_and_manifest_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            _, teacher_digest, _, _, _ = _prepare_terminal_run(source)
            RUNNER.commit_success_terminal(source, 7, teacher_digest)
            copied = root / "copied"
            shutil.copytree(source, copied)
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "verify_stage1_run.py"),
                    "--run-dir",
                    str(copied),
                ],
                cwd=str(ROOT),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["status"], "verified")

            manifest = copied / "run_manifest.json"
            manifest.write_bytes(manifest.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "SEALED.json differs"):
                RUNNER.verify_sealed_stage1_run(copied)

            copied_again = root / "copied-again"
            shutil.copytree(source, copied_again)
            (copied_again / "unexpected.txt").write_text("extra\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected members"):
                RUNNER.verify_sealed_stage1_run(copied_again)

    def test_node3_rank_audit_metrics_are_bound_by_terminal_seal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            _, teacher_digest, _, _, metrics = _prepare_terminal_run(run)
            # Rank-level sample/gate/memory fields live inside the attempt
            # metrics byte stream. The v5 terminal chain must reject any
            # retrospective edit to those fields.
            records = [
                json.loads(line)
                for line in metrics.read_text(encoding="utf-8").splitlines()
            ]
            records[0]["all_rank_audit"] = [
                {
                    "rank": rank,
                    "sample_gate_sha256": "a" * 64,
                    "cuda_max_memory_allocated": 100 + rank,
                }
                for rank in range(8)
            ]
            metrics.write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
                encoding="utf-8",
            )
            RUNNER.commit_success_terminal(run, 7, teacher_digest)
            self.assertEqual(
                RUNNER.verify_sealed_stage1_run(run)["status"], "verified"
            )
            payload = metrics.read_text(encoding="utf-8")
            metrics.write_text(payload.replace("100", "999", 1), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SEALED.json differs"):
                RUNNER.verify_sealed_stage1_run(run)

    def test_resume_requires_latest_complete_committed_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            checkpoints = run / "checkpoints"
            checkpoints.mkdir(parents=True)
            step_zero = RUNNER.commit_checkpoint_payload(
                checkpoints, 0, {"metadata": {"optimizer_step": 0}}
            )
            step_seven = RUNNER.commit_checkpoint_payload(
                checkpoints, 7, {"metadata": {"optimizer_step": 7}}
            )
            self.assertEqual(
                RUNNER.require_latest_resume_checkpoint(run, step_seven),
                step_seven.resolve(),
            )
            with self.assertRaisesRegex(ValueError, "latest committed checkpoint"):
                RUNNER.require_latest_resume_checkpoint(run, step_zero)

            incomplete = checkpoints / "step-00008"
            incomplete.mkdir()
            (incomplete / "checkpoint.pt").write_bytes(b"partial")
            with self.assertRaisesRegex(ValueError, "members differ"):
                RUNNER.require_latest_resume_checkpoint(run, step_seven)

            (incomplete / "checkpoint.pt").unlink()
            incomplete.rmdir()
            checkpoint_zero, _ = RUNNER.checkpoint_paths(step_zero)
            checkpoint_zero.write_bytes(checkpoint_zero.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "payload hash differs"):
                RUNNER.require_latest_resume_checkpoint(run, step_seven)

    def test_resume_attempt_logs_are_segmented_and_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            checkpoints = run / "checkpoints"
            checkpoints.mkdir(parents=True)
            (run / "logs").mkdir()
            step_zero = RUNNER.commit_checkpoint_payload(
                checkpoints, 0, {"metadata": {"optimizer_step": 0}}
            )
            first_metrics = RUNNER.begin_attempt_log(
                run,
                config_sha256="a" * 64,
                start_checkpoint=step_zero,
                start_optimizer_step=0,
                start_global_microstep=0,
                is_resume=False,
            )
            RUNNER.jsonl_append(
                first_metrics,
                {
                    "schema_version": RUNNER.RUN_SCHEMA_VERSION,
                    "event": "optimizer_step",
                    "optimizer_step": 5,
                    "global_microstep": 20,
                },
            )
            first_metrics_before = first_metrics.read_bytes()
            first_metadata = run / "logs" / "attempt-0000.json"
            first_metadata_before = first_metadata.read_bytes()
            step_five = RUNNER.commit_checkpoint_payload(
                checkpoints, 5, {"metadata": {"optimizer_step": 5}}
            )

            second_metrics = RUNNER.begin_attempt_log(
                run,
                config_sha256="a" * 64,
                start_checkpoint=step_five,
                start_optimizer_step=5,
                start_global_microstep=20,
                is_resume=True,
            )
            self.assertEqual(second_metrics.name, "metrics.attempt-0001.jsonl")
            self.assertEqual(first_metrics.read_bytes(), first_metrics_before)
            self.assertEqual(first_metadata.read_bytes(), first_metadata_before)
            self.assertEqual(second_metrics.read_bytes(), b"")
            RUNNER.jsonl_append(
                second_metrics,
                {
                    "schema_version": RUNNER.RUN_SCHEMA_VERSION,
                    "event": "optimizer_step",
                    "optimizer_step": 6,
                    "global_microstep": 24,
                },
            )
            summaries = RUNNER.inspect_attempt_logs(run, require_nonempty=True)
            self.assertEqual([item["attempt_index"] for item in summaries], [0, 1])
            self.assertEqual([item["record_count"] for item in summaries], [1, 1])
            self.assertEqual(first_metrics.read_bytes(), first_metrics_before)

            RUNNER.jsonl_append(
                second_metrics,
                {
                    "schema_version": RUNNER.RUN_SCHEMA_VERSION,
                    "event": "optimizer_step",
                    "optimizer_step": 6,
                    "global_microstep": 24,
                },
            )
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                RUNNER.inspect_attempt_logs(run)

            checkpoint_zero, _ = RUNNER.checkpoint_paths(step_zero)
            checkpoint_zero.write_bytes(checkpoint_zero.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "payload changed after launch"):
                RUNNER.inspect_attempt_logs(run)

    def test_runner_consumes_sealed_prior_identity_and_binds_codec(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            student = root / "student.pt"
            teacher = root / "teacher.pt"
            for snapshot in (student, teacher):
                snapshot.mkdir()
                (snapshot / "compression_state_dict.bin").write_bytes(
                    b"frozen-codec"
                )
            codec_hash = RUNNER.sha256_file(
                student / "compression_state_dict.bin"
            )
            artifact = RUNNER.CodecPriorArtifact(
                directory=str(root / "a1"),
                prior=(0.4, 0.3, 0.2, 0.1),
                manifest_sha256="a" * 64,
                checkpoint_sha256=codec_hash,
                ordered_source_records_sha256="b" * 64,
                codec_prior_sha256="c" * 64,
                artifact_seal_sha256="d" * 64,
                codec_load={
                    "load_mode": "self_contained",
                    "pretrained_model_id": None,
                    "resolved_snapshot_revision": None,
                    "resolved_snapshot_files": [],
                },
                identity={
                    "schema_version": "test-sealed-a1",
                    "audiocraft": {
                        "source_identity": {"tree_sha256": "f" * 64}
                    },
                },
                identity_sha256="e" * 64,
            )
            config = _config(
                root, codebook_prior_artifact_dir=str(root / "a1")
            )
            with mock.patch.object(
                RUNNER, "load_codec_prior_artifact", return_value=artifact
            ):
                observed = RUNNER.resolve_codec_prior_artifact(config)
            self.assertIs(observed, artifact)
            RUNNER.verify_codec_prior_source_binding(
                artifact, {"tree_sha256": "f" * 64}
            )
            with self.assertRaisesRegex(ValueError, "used to estimate A1"):
                RUNNER.verify_codec_prior_source_binding(
                    artifact, {"tree_sha256": "0" * 64}
                )

            (teacher / "compression_state_dict.bin").write_bytes(b"drift")
            with mock.patch.object(
                RUNNER, "load_codec_prior_artifact", return_value=artifact
            ), self.assertRaisesRegex(ValueError, "teacher checkpoint codec"):
                RUNNER.resolve_codec_prior_artifact(config)

    def test_dry_run_hashes_inputs_without_importing_audiocraft(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "train.full.jsonl"
            manifest.write_text(
                "".join(
                    json.dumps(
                        {
                            "sample_id": "sample-{:02d}".format(index),
                            "prompt": "piano {}".format(index),
                        }
                    )
                    + "\n"
                    for index in range(16)
                ),
                encoding="utf-8",
            )
            student = root / "student.pt"
            teacher = root / "teacher.pt"
            student.mkdir()
            teacher.mkdir()
            (student / "state_dict.bin").write_bytes(b"same-state")
            (teacher / "state_dict.bin").write_bytes(b"same-state")
            (student / "compression_state_dict.bin").write_bytes(b"same-codec")
            (teacher / "compression_state_dict.bin").write_bytes(b"same-codec")
            lm_path = root / "audiocraft" / "audiocraft" / "models" / "lm.py"
            lm_path.parent.mkdir(parents=True)
            lm_path.write_text("# fake patched lm\n", encoding="utf-8")
            checkpoint_hash = sha256_path(student)
            source_hash = RUNNER.audiocraft_source_identity(
                root / "audiocraft"
            )["tree_sha256"]
            state_hash = RUNNER.sha256_file(student / "state_dict.bin")
            config = _config(
                root,
                max_optimizer_steps=1,
                cfg_generation_checkpoint_sha256=checkpoint_hash,
                cfg_generation_audiocraft_source_sha256=source_hash,
                cfg_generation_state_dict_sha256=state_hash,
            )
            args = type("Args", (), {"codebook_prior": None})()
            before = set(sys.modules)
            cfg_identity = {
                "selected_cfg_scale": 3.0,
                "decision_file_sha256": "a" * 64,
                "decision_payload_sha256": "a" * 64,
                "scientific_config_sha256": "a" * 64,
                "generation_identity": {},
            }
            with mock.patch.object(
                RUNNER,
                "require_offline_hf_environment",
                return_value={
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "HF_DATASETS_OFFLINE": "1",
                },
            ), mock.patch.object(
                RUNNER,
                "verify_configured_cfg_decision",
                return_value=cfg_identity,
            ):
                report = RUNNER.validate_dry_run(config, args)
            newly_imported = set(sys.modules) - before
            self.assertTrue(report["audiocraft_imported"] is False)
            self.assertEqual(report["manifest_records"], 16)
            self.assertFalse(any(name == "audiocraft" or name.startswith("audiocraft.") for name in newly_imported))

    def test_cfg_generation_binding_rejects_checkpoint_source_and_state_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            student = root / "student.pt"
            teacher = root / "teacher.pt"
            student.mkdir()
            teacher.mkdir()
            for snapshot in (student, teacher):
                (snapshot / "state_dict.bin").write_bytes(b"state")
                (snapshot / "compression_state_dict.bin").write_bytes(b"codec")
            source = root / "audiocraft" / "audiocraft" / "models"
            source.mkdir(parents=True)
            (source / "lm.py").write_text("# pinned\n", encoding="utf-8")
            hashes = {
                "student_checkpoint_sha256": sha256_path(student),
                "teacher_checkpoint_sha256": sha256_path(teacher),
            }
            source_identity = RUNNER.audiocraft_source_identity(root / "audiocraft")
            config = _config(
                root,
                cfg_generation_checkpoint_sha256=hashes["student_checkpoint_sha256"],
                cfg_generation_audiocraft_source_sha256=source_identity["tree_sha256"],
                cfg_generation_state_dict_sha256=RUNNER.sha256_file(
                    student / "state_dict.bin"
                ),
            )
            binding = RUNNER.verify_cfg_generation_binding(
                config, hashes, source_identity
            )
            self.assertEqual(
                binding["state_dict_sha256"],
                config.cfg_generation_state_dict_sha256,
            )

            with self.assertRaisesRegex(ValueError, "CFG generation source"):
                RUNNER.verify_cfg_generation_binding(
                    config, hashes, {"tree_sha256": "f" * 64}
                )
            (teacher / "state_dict.bin").write_bytes(b"changed")
            changed_hashes = dict(hashes)
            changed_hashes["teacher_checkpoint_sha256"] = sha256_path(teacher)
            with self.assertRaisesRegex(ValueError, "exact same initial checkpoint"):
                RUNNER.verify_cfg_generation_binding(
                    config, changed_hashes, source_identity
                )

    def test_ddp_scoring_facade_calls_wrapped_forward(self) -> None:
        class FakeLM(nn.Module):
            num_codebooks = 2

            def compute_predictions(self, codes, **kwargs):
                return codes + 1

        class FakeDDP:
            def __init__(self):
                self.calls = 0

            def __call__(self, codes, conditions):
                self.calls += 1
                return codes + 2

        lm = FakeLM()
        ddp = FakeDDP()
        facade = RUNNER.DDPScoringFacade(ddp, lm)
        result = facade.compute_predictions(
            torch.zeros(1, 2, 3),
            conditions=[],
            condition_tensors={},
            keep_only_valid_steps=True,
        )
        self.assertEqual(ddp.calls, 1)
        self.assertTrue(torch.equal(result, torch.full((1, 2, 3), 2.0)))

    def test_freeze_condition_provider_survives_parent_train_reset(self) -> None:
        class Conditioner(nn.Module):
            def __init__(self):
                super().__init__()
                self.__dict__["t5"] = nn.Linear(2, 2)

        class FakeLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.body = nn.Linear(2, 2)
                self.condition_provider = nn.Sequential(Conditioner(), nn.Linear(2, 2))

        model = FakeLM()
        RUNNER.freeze_condition_provider(model)
        model.train()
        self.assertTrue(model.condition_provider.training)
        RUNNER.freeze_condition_provider(model)
        RUNNER.assert_condition_provider_frozen(model, "student")
        self.assertTrue(model.body.weight.requires_grad)
        external_t5 = model.condition_provider[0].__dict__["t5"]
        self.assertFalse(external_t5.training)
        self.assertFalse(any(parameter.requires_grad for parameter in external_t5.parameters()))

    def test_null_condition_is_direct_none_and_tensors_reused(self) -> None:
        class Attributes:
            def __init__(self, text):
                self.text = text

        class Provider(nn.Module):
            def __init__(self):
                super().__init__()
                self.seen = []

            def tokenize(self, attributes):
                self.seen.append([item.text["description"] for item in attributes])
                return attributes

            def forward(self, attributes):
                values = [0.0 if item.text["description"] is None else 1.0 for item in attributes]
                embedding = torch.tensor(values).view(-1, 1, 1)
                mask = embedding[:, :, 0].to(torch.int64)
                return {"description": (embedding, mask)}

        class Dropout:
            def __init__(self, p):
                self.p = p

            def __call__(self, attributes):
                self.assert_frozen()
                return [Attributes({"description": None}) for _ in attributes]

            def assert_frozen(self):
                if self.p != 1.0:
                    raise AssertionError("CFG dropout must be p=1")

        class FakeLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.condition_provider = Provider()

        model = FakeLM()
        conditional, null = RUNNER.prepare_condition_tensors(
            model, ["piano", "drums"], Attributes, Dropout
        )
        self.assertEqual(
            model.condition_provider.seen,
            [["piano", "drums", None, None]],
        )
        self.assertTrue(conditional["description"][1].all())
        self.assertFalse(null["description"][1].any())

    def test_accumulation_denominator_guard(self) -> None:
        self.assertTrue(
            RUNNER.denominator_window_is_constant([100.0] * 4, rtol=1.0e-6)
        )
        self.assertFalse(
            RUNNER.denominator_window_is_constant(
                [100.0, 100.0, 100.0, 100.1], rtol=1.0e-6
            )
        )
        with self.assertRaises(ValueError):
            RUNNER.denominator_window_is_constant([1.0, 1.0], rtol=0.0)


if __name__ == "__main__":
    unittest.main()
