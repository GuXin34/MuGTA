from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


WORKPACK = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = WORKPACK / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_script("build_fma_calibration_manifest")
estimator = load_script("estimate_codec_prior")


class FmaManifestPrimitiveTests(unittest.TestCase):
    def test_lowest_hash_selection_is_domain_separated_and_order_independent(self) -> None:
        rows = [{"fma_track_id": value, "marker": str(value)} for value in range(1, 21)]
        independently_ranked = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"ptc-opd-codec-cal-v1|2701|{row['fma_track_id']}".encode("utf-8")
            ).hexdigest(),
        )[:7]
        expected_ids = [row["fma_track_id"] for row in independently_ranked]
        observed_forward = builder.choose_lowest_hashes(rows, 7)
        observed_reverse = builder.choose_lowest_hashes(list(reversed(rows)), 7)
        self.assertEqual([row["fma_track_id"] for row in observed_forward], expected_ids)
        self.assertEqual([row["fma_track_id"] for row in observed_reverse], expected_ids)
        self.assertEqual(rows[0], {"fma_track_id": 1, "marker": "1"})

    def test_offset_uses_first_uint64_and_inclusive_valid_range(self) -> None:
        decoded_frames = 1201
        segment_frames = 1000
        digest = hashlib.sha256(
            b"ptc-opd-codec-segment-v1|2701|42"
        ).hexdigest()
        expected = int.from_bytes(bytes.fromhex(digest)[:8], "big") % 202
        observed, observed_digest = builder.deterministic_start_frame(
            42, decoded_frames, segment_frames
        )
        self.assertEqual(observed, expected)
        self.assertEqual(observed_digest, digest)
        self.assertEqual(builder.deterministic_start_frame(42, 1000, 1000)[0], 0)
        with self.assertRaises(ValueError):
            builder.deterministic_start_frame(42, 999, 1000)

    def test_source_and_pcm_hashes_are_byte_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.bin"
            payload = b"fma-fixture\x00\xff"
            path.write_bytes(payload)
            self.assertEqual(builder.sha256_file(path), hashlib.sha256(payload).hexdigest())
            self.assertEqual(builder.sha1_file(path), hashlib.sha1(payload).hexdigest())

        pcm = np.array([[0.0, 0.5], [-0.5, 1.0]], dtype=np.float64)
        expected = hashlib.sha256(
            np.ascontiguousarray(pcm, dtype=np.dtype("<f4")).tobytes(order="C")
        ).hexdigest()
        self.assertEqual(builder.pcm_f32le_sha256(pcm), expected)
        self.assertEqual(estimator.pcm_f32le_sha256(pcm), expected)

    def test_refuses_any_existing_member_of_output_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "manifest.jsonl"
            second = Path(directory) / "report.json"
            first.write_text("partial", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                builder._refuse_existing([first, second])

    def test_a1_r2_eligibility_rejects_antiphase_without_track_id_rule(self) -> None:
        time = torch.linspace(-0.5, 0.5, 320_000, dtype=torch.float32)
        antiphase = torch.stack([time, -time], dim=1).numpy()

        def exact_convert(waveform, *, from_rate, to_rate, to_channels):
            self.assertEqual((from_rate, to_rate, to_channels), (32_000, 32_000, 1))
            return waveform.mean(dim=1, keepdim=True)

        result = builder.analyze_input_eligibility(
            antiphase,
            32_000,
            np=np,
            torch_module=torch,
            convert_audio=exact_convert,
        )
        self.assertFalse(result["eligible"])
        self.assertEqual(result["mono_rms"], 0.0)
        self.assertEqual(result["mono_compatibility_ratio"], 0.0)
        self.assertAlmostEqual(result["lr_pearson_correlation"], -1.0)
        self.assertNotIn("fma_track_id", result)

    def test_a1_r2_eligibility_accepts_ordinary_stereo_and_is_deterministic(self) -> None:
        generator = torch.Generator().manual_seed(17)
        pcm = torch.randn(320_000, 2, generator=generator).mul_(0.05).numpy()

        def exact_convert(waveform, *, from_rate, to_rate, to_channels):
            return waveform.mean(dim=1, keepdim=True)

        kwargs = dict(
            np=np,
            torch_module=torch,
            convert_audio=exact_convert,
        )
        first = builder.analyze_input_eligibility(pcm, 32_000, **kwargs)
        second = builder.analyze_input_eligibility(pcm, 32_000, **kwargs)
        self.assertEqual(first, second)
        self.assertTrue(first["eligible"])
        self.assertGreaterEqual(first["mono_rms"], builder.MIN_MONO_RMS)
        self.assertGreaterEqual(
            first["mono_compatibility_ratio"],
            builder.MIN_MONO_COMPATIBILITY_RATIO,
        )


class CodecPriorAggregationTests(unittest.TestCase):
    def test_estimator_recomputes_manifest_eligibility_before_codec(self) -> None:
        generator = torch.Generator().manual_seed(23)
        pcm_tensor = torch.randn(320_000, 2, generator=generator).mul_(0.03)
        pcm = pcm_tensor.numpy()
        codec_input = pcm_tensor.T.unsqueeze(0).mean(dim=1, keepdim=True).contiguous()
        builder_record = builder.analyze_input_eligibility(
            pcm,
            32_000,
            np=np,
            torch_module=torch,
            convert_audio=lambda waveform, **_: waveform.mean(dim=1, keepdim=True),
        )
        row = {"fma_track_id": 1, **builder_record}
        observed = estimator.recompute_and_validate_eligibility(
            row,
            pcm,
            codec_input,
            np=np,
            torch_module=torch,
        )
        self.assertEqual(
            observed["eligibility_codec_input_pcm_sha256"],
            builder_record["eligibility_codec_input_pcm_sha256"],
        )
        tampered = dict(row)
        tampered["mono_rms"] *= 1.01
        with self.assertRaisesRegex(RuntimeError, "mono_rms mismatch"):
            estimator.recompute_and_validate_eligibility(
                tampered,
                pcm,
                codec_input,
                np=np,
                torch_module=torch,
            )

    def test_frozen_codec_model_and_tensor_contract(self) -> None:
        class FrozenCodec:
            channels = 1
            sample_rate = 32_000
            frame_rate = 50.0
            cardinality = 2_048
            num_codebooks = 4

        contract = estimator.validate_codec_model_contract(FrozenCodec())
        self.assertEqual(
            {
                key: contract[key]
                for key in (
                    "channels",
                    "sample_rate",
                    "frame_rate",
                    "cardinality",
                    "num_codebooks",
                    "input_frames",
                    "canonical_code_shape",
                )
            },
            {
                "channels": 1,
                "sample_rate": 32_000,
                "frame_rate": 50.0,
                "cardinality": 2_048,
                "num_codebooks": 4,
                "input_frames": 320_000,
                "canonical_code_shape": [1, 4, 500],
            },
        )
        waveform = torch.zeros(1, 1, 320_000, dtype=torch.float32)
        codes = torch.arange(2_000, dtype=torch.long).reshape(1, 4, 500)
        estimator.validate_codec_tensors(waveform, codes, torch)

        class NumPyScalarCodec:
            channels = np.int64(1)
            sample_rate = np.int64(32_000)
            frame_rate = np.float64(50.0)
            cardinality = np.int64(2_048)
            num_codebooks = np.int64(4)

        numpy_contract = estimator.validate_codec_model_contract(NumPyScalarCodec())
        self.assertEqual(numpy_contract["frame_rate"], 50.0)

    def test_codec_contract_rejects_metadata_shape_dtype_and_range_drift(self) -> None:
        base = {
            "channels": 1,
            "sample_rate": 32_000,
            "frame_rate": 50.0,
            "cardinality": 2_048,
            "num_codebooks": 4,
        }
        for field, wrong in (
            ("channels", 2),
            ("sample_rate", 24_000),
            ("frame_rate", 25),
            ("cardinality", 1_024),
            ("num_codebooks", 8),
        ):
            model = type("Codec", (), {**base, field: wrong})()
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, field):
                estimator.validate_codec_model_contract(model)

        waveform = torch.zeros(1, 1, 320_000, dtype=torch.float32)
        codes = torch.zeros(1, 4, 500, dtype=torch.long)
        with self.assertRaisesRegex(RuntimeError, "320000"):
            estimator.validate_codec_tensors(waveform[..., :-1], codes, torch)
        with self.assertRaisesRegex(RuntimeError, r"\(1, 4, 500\)"):
            estimator.validate_codec_tensors(waveform, codes[..., :-1], torch)
        with self.assertRaisesRegex(RuntimeError, "torch.long"):
            estimator.validate_codec_tensors(waveform, codes.int(), torch)
        codes[0, 0, 0] = 2_048
        with self.assertRaisesRegex(RuntimeError, "outside the frozen range"):
            estimator.validate_codec_tensors(waveform, codes, torch)

    @staticmethod
    def _artifact_identity_fixture() -> dict:
        return {
            "manifest": {
                "schema_version": estimator.REQUIRED_MANIFEST_SCHEMA,
                "path": "/inputs/codec_calibration.train.jsonl",
                "sha256": "a" * 64,
                "items": 512,
            },
            "checkpoint": {
                "path": "/checkpoints/musicgen-small",
                "payload_path": "/checkpoints/musicgen-small/compression_state_dict.bin",
                "sha256": "b" * 64,
                "load_mode": "self_contained",
                "pretrained_model_id": None,
                "resolved_snapshot_revision": None,
            },
            "source_identity": {
                "source_root": "/data/fma_small",
                "archive_sha1": estimator.REQUIRED_ARCHIVE_SHA1,
                "items": 512,
                "ordered_source_records_sha256": "c" * 64,
                "manifest_report_sha256": "f" * 64,
            },
            "audiocraft": {
                "base_commit": estimator.PINNED_AUDIOCRAFT_BASE_COMMIT,
                "source_identity": {
                    "schema_version": "ptc-opd-audiocraft-source-tree-v1",
                    "file_count": 7,
                    "tree_sha256": "d" * 64,
                    "identity_sha256": "e" * 64,
                },
            },
            "codec_load": {
                "load_mode": "self_contained",
                "pretrained_model_id": None,
                "resolved_snapshot_revision": None,
                "resolved_snapshot_files": [],
            },
            "protocol": {
                "identity_schema": estimator.ARTIFACT_IDENTITY_SCHEMA,
                "label": estimator.PROTOCOL_LABEL,
                "manifest_schema": estimator.REQUIRED_MANIFEST_SCHEMA,
                "result_schema": estimator.RESULT_SCHEMA,
                "artifact_seal_schema": estimator.ARTIFACT_SEAL_SCHEMA,
                "manifest_report": {
                    "schema_version": estimator.REQUIRED_MANIFEST_SCHEMA,
                    "sha256": "f" * 64,
                },
                "implementation": estimator.implementation_identity(),
            },
        }

    @staticmethod
    def _artifact_payload_fixture() -> dict:
        identity = CodecPriorAggregationTests._artifact_identity_fixture()
        prior = {
            "prior": [0.4, 0.3, 0.2, 0.1],
            "manifest_sha256": identity["manifest"]["sha256"],
            "manifest_report_sha256": identity["protocol"]["manifest_report"]["sha256"],
            "protocol": identity["protocol"],
        }
        return {
            estimator.OUTPUT_NAMES[0]: b"per-clip\n",
            estimator.OUTPUT_NAMES[1]: b"summary\n",
            estimator.OUTPUT_NAMES[2]: estimator._json_line(prior),
        }

    def test_artifact_directory_publishes_once_with_complete_seal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / estimator.REQUIRED_OUTPUT_DIR_BASENAME
            payloads = self._artifact_payload_fixture()
            identity = self._artifact_identity_fixture()
            original_replace = estimator.os.replace
            with mock.patch.object(
                estimator.os, "replace", wraps=original_replace
            ) as replace:
                paths = estimator.publish_artifact_directory(output, payloads, identity)
            replace.assert_called_once()
            self.assertTrue(output.is_dir())
            self.assertEqual(
                {path.name for path in output.iterdir()},
                set(estimator.OUTPUT_NAMES) | {estimator.ARTIFACT_SEAL_NAME},
            )
            seal = json.loads(
                (output / estimator.ARTIFACT_SEAL_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(seal["status"], "complete")
            self.assertEqual(seal["schema_version"], estimator.ARTIFACT_SEAL_SCHEMA)
            self.assertEqual(seal["manifest"], identity["manifest"])
            self.assertEqual(seal["audiocraft"], identity["audiocraft"])
            self.assertEqual(seal["codec_load"], identity["codec_load"])
            self.assertEqual(seal["protocol"], identity["protocol"])
            for name, path in paths.items():
                self.assertEqual(path.read_bytes(), payloads[name])
                self.assertEqual(seal["artifacts"][name]["sha256"], estimator.sha256_file(path))
            self.assertFalse(
                list(
                    Path(directory).glob(
                        ".{}.staging.*".format(estimator.REQUIRED_OUTPUT_DIR_BASENAME)
                    )
                )
            )

    def test_artifact_transaction_cleans_staging_on_exception(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / estimator.REQUIRED_OUTPUT_DIR_BASENAME
            with mock.patch.object(
                estimator,
                "_validate_staged_artifact_directory",
                side_effect=RuntimeError("injected validation failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    estimator.publish_artifact_directory(
                        output,
                        self._artifact_payload_fixture(),
                        self._artifact_identity_fixture(),
                    )
            self.assertFalse(output.exists())
            self.assertFalse(
                list(
                    parent.glob(
                        ".{}.staging.*".format(estimator.REQUIRED_OUTPUT_DIR_BASENAME)
                    )
                )
            )

    def test_artifact_transaction_rejects_prior_protocol_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / estimator.REQUIRED_OUTPUT_DIR_BASENAME
            payloads = self._artifact_payload_fixture()
            prior = json.loads(payloads[estimator.OUTPUT_NAMES[2]])
            prior["protocol"]["label"] = "A1-R1"
            payloads[estimator.OUTPUT_NAMES[2]] = estimator._json_line(prior)
            with self.assertRaisesRegex(RuntimeError, "prior/seal protocol"):
                estimator.publish_artifact_directory(
                    output,
                    payloads,
                    self._artifact_identity_fixture(),
                )
            self.assertFalse(output.exists())
            self.assertFalse(
                list(
                    parent.glob(
                        ".{}.staging.*".format(estimator.REQUIRED_OUTPUT_DIR_BASENAME)
                    )
                )
            )

    def test_artifact_transaction_refuses_any_existing_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / estimator.REQUIRED_OUTPUT_DIR_BASENAME
            output.mkdir()
            marker = output / "do-not-touch"
            marker.write_text("old generation", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "must not exist at all"):
                estimator.publish_artifact_directory(
                    output,
                    self._artifact_payload_fixture(),
                    self._artifact_identity_fixture(),
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "old generation")

    def test_check_only_does_not_create_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-be-created"
            argv = [
                "--manifest", str(Path(directory) / "codec_calibration.train.jsonl"),
                "--manifest-sha256", "a" * 64,
                "--manifest-report", str(Path(directory) / "codec_calibration.train.report.json"),
                "--manifest-report-sha256", "f" * 64,
                "--source-root", str(Path(directory) / "fma"),
                "--audiocraft-root", str(Path(directory) / "audiocraft"),
                "--codec-checkpoint", str(Path(directory) / "codec"),
                "--codec-checkpoint-sha256", "b" * 64,
                "--output-dir", str(output),
                "--check-only",
            ]
            with mock.patch.object(estimator, "guard_single_process_environment"), mock.patch.object(
                estimator, "_preflight", return_value={"status": "preflight_passed_not_scientific_result"}
            ):
                self.assertEqual(estimator.main(argv), 0)
            self.assertFalse(output.exists())

    def test_formal_main_rejects_existing_output_before_assay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / estimator.REQUIRED_OUTPUT_DIR_BASENAME
            output.mkdir()
            argv = [
                "--manifest", str(Path(directory) / "codec_calibration.train.jsonl"),
                "--manifest-sha256", "a" * 64,
                "--manifest-report", str(Path(directory) / "codec_calibration.train.report.json"),
                "--manifest-report-sha256", "f" * 64,
                "--source-root", str(Path(directory) / "fma"),
                "--audiocraft-root", str(Path(directory) / "audiocraft"),
                "--codec-checkpoint", str(Path(directory) / "codec"),
                "--codec-checkpoint-sha256", "b" * 64,
                "--output-dir", str(output),
            ]
            with mock.patch.object(estimator, "guard_single_process_environment"), mock.patch.object(
                estimator, "run_codec_assay"
            ) as assay:
                with self.assertRaisesRegex(FileExistsError, "must not exist at all"):
                    estimator.main(argv)
            assay.assert_not_called()

    def test_aggregation_bootstrap_and_split_are_deterministic(self) -> None:
        marginals = np.array(
            [
                [4.0, 3.0, 2.0, 1.0],
                [5.0, 3.0, 1.0, 0.5],
                [3.0, 2.5, 2.0, 1.0],
                [4.5, 3.5, 1.5, 0.8],
                [5.5, 4.0, 2.5, 1.2],
                [3.5, 2.0, 1.0, 0.4],
            ],
            dtype=np.float64,
        )
        track_ids = [2, 3, 5, 7, 11, 13]
        first = estimator.aggregate_marginals(
            marginals, track_ids, bootstrap_replicates=200, bootstrap_seed=4701
        )
        second = estimator.aggregate_marginals(
            marginals, track_ids, bootstrap_replicates=200, bootstrap_seed=4701
        )
        self.assertEqual(first, second)
        np.testing.assert_allclose(first["raw_mean_marginal"], marginals.mean(axis=0))
        self.assertAlmostEqual(sum(first["prior"]), 1.0, places=14)
        self.assertEqual(first["bootstrap"]["replicates"], 200)
        self.assertEqual(first["split_half"]["half_a_n"], 3)
        self.assertEqual(first["split_half"]["half_b_n"], 3)
        self.assertAlmostEqual(first["split_half"]["kendall_tau_b"], 1.0)
        self.assertGreater(first["diagnostics"]["total_variation_from_uniform"], 0.0)
        self.assertEqual(first["diagnostics"]["denominator_floor_activation"], 0)
        self.assertEqual(len(first["leave_one_out_influence"]["per_clip"]), 6)
        self.assertEqual(
            first["sensitivity"]["trimmed_mean_1pct"]["trim_each_tail"], 0
        )

    def test_frozen_512_clip_trim_and_leave_one_out_influence(self) -> None:
        generator = np.random.Generator(np.random.PCG64(81))
        marginals = generator.normal(
            loc=np.array([4.0, 3.0, 2.0, 1.0]),
            scale=0.1,
            size=(512, 4),
        )
        result = estimator.aggregate_marginals(
            marginals,
            list(range(10_000, 10_512)),
            bootstrap_replicates=20,
        )
        trimmed = result["sensitivity"]["trimmed_mean_1pct"]
        self.assertEqual(trimmed["trim_each_tail"], 5)
        self.assertEqual(trimmed["retained_clips_per_codebook"], 502)
        self.assertEqual(len(result["leave_one_out_influence"]["per_clip"]), 512)
        self.assertLess(result["leave_one_out_influence"]["max_total_variation"], 0.05)

    def test_negative_aggregate_is_floored_before_sum_one_prior(self) -> None:
        values = np.array(
            [
                [2.0, -1.0, 0.0],
                [4.0, -3.0, 0.0],
                [3.0, -2.0, 0.0],
                [5.0, -4.0, 0.0],
            ]
        )
        result = estimator.aggregate_marginals(
            values, [1, 2, 3, 4], bootstrap_replicates=20
        )
        self.assertAlmostEqual(sum(result["prior"]), 1.0, places=14)
        self.assertEqual(result["clipped_mean_marginal"][1], 1.0e-8)
        self.assertEqual(result["clipped_mean_marginal"][2], 1.0e-8)

    def test_gzip_jsonl_has_fixed_header_and_roundtrips(self) -> None:
        records = [{"b": 2, "a": 1}, {"value": [1.25, 3.5]}]
        first = estimator.deterministic_gzip_jsonl(records)
        second = estimator.deterministic_gzip_jsonl(records)
        self.assertEqual(first, second)
        decoded = [json.loads(line) for line in gzip.GzipFile(fileobj=io.BytesIO(first))]
        self.assertEqual(decoded, records)

    def test_rank_zero_assay_rejects_distributed_launcher_environment(self) -> None:
        estimator.guard_single_process_environment({})
        estimator.guard_single_process_environment(
            {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"}
        )
        with self.assertRaises(RuntimeError):
            estimator.guard_single_process_environment(
                {"WORLD_SIZE": "8", "RANK": "0", "LOCAL_RANK": "0"}
            )

    def test_checkpoint_indirection_is_exactly_whitelisted_and_offline_gated(self) -> None:
        fake_torch = mock.Mock()
        fake_torch.load.return_value = {"pretrained": "facebook/encodec_32khz"}
        observed = estimator._inspect_local_checkpoint_package(
            fake_torch, Path("compression_state_dict.bin")
        )
        self.assertEqual(observed["load_mode"], "pretrained_indirection")
        fake_torch.load.return_value = {"pretrained": "some/other-codec"}
        with self.assertRaises(RuntimeError):
            estimator._inspect_local_checkpoint_package(
                fake_torch, Path("compression_state_dict.bin")
            )
        # Do not inherit the caller's shell policy: CI and the accepted remote
        # environment may already export all three offline flags.
        with mock.patch.dict(estimator.os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                estimator._require_offline_hf_environment()
        with mock.patch.dict(
            estimator.os.environ,
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            },
            clear=True,
        ):
            estimator._require_offline_hf_environment()

    def test_resolved_snapshot_hashes_target_bytes_not_symlink_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            blob = cache / "blobs" / "weights"
            blob.parent.mkdir()
            blob.write_bytes(b"actual-weight-bytes")
            snapshot = cache / "models--facebook--encodec_32khz" / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            visible = snapshot / "model.safetensors"
            visible.symlink_to(blob)

            class FakeConfig:
                _name_or_path = str(snapshot)

            class Wrapped:
                config = FakeConfig()

            class FakeModel:
                model = Wrapped()

            report = estimator.hash_resolved_model_snapshot(FakeModel())
            self.assertEqual(report["resolved_snapshot_revision"], "abc123")
            self.assertEqual(len(report["resolved_snapshot_files"]), 1)
            item = report["resolved_snapshot_files"][0]
            self.assertTrue(item["is_symlink"])
            self.assertEqual(item["size_bytes"], len(b"actual-weight-bytes"))
            self.assertEqual(
                item["sha256"], hashlib.sha256(b"actual-weight-bytes").hexdigest()
            )

    def test_broken_snapshot_symlink_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "models--facebook--encodec_32khz" / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            (snapshot / "model.safetensors").symlink_to(Path(directory) / "missing-blob")

            class FakeConfig:
                _name_or_path = str(snapshot)

            class Wrapped:
                config = FakeConfig()

            class FakeModel:
                model = Wrapped()

            with self.assertRaises(RuntimeError):
                estimator.hash_resolved_model_snapshot(FakeModel())

    def test_repo_id_config_uses_cached_config_snapshot_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "models--facebook--encodec_32khz" / "snapshots" / "commit987"
            snapshot.mkdir(parents=True)
            config = snapshot / "config.json"
            config.write_bytes(b"{}")

            class FakeConfig:
                _name_or_path = "facebook/encodec_32khz"

            class Wrapped:
                config = FakeConfig()

            class FakeModel:
                model = Wrapped()

            with mock.patch.object(
                estimator,
                "resolved_hf_snapshot_from_cache",
                return_value=(snapshot, "commit987"),
            ):
                report = estimator.hash_resolved_model_snapshot(FakeModel())
            self.assertEqual(report["resolved_snapshot_revision"], "commit987")
            self.assertEqual(report["resolved_snapshot_files"][0]["sha256"], hashlib.sha256(b"{}").hexdigest())

    def test_obvious_dev_or_test_manifest_basename_is_rejected_before_io(self) -> None:
        for name in ("test.full.jsonl", "codec_calibration.dev.jsonl", "phenomenon_probe.jsonl"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    estimator.manifest_sha256_and_rows(Path(name), "0" * 64)

    def test_manifest_report_hash_tamper_is_rejected_before_semantic_use(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a1-r2" / estimator.REQUIRED_REPORT_BASENAME
            path.parent.mkdir()
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "report SHA-256 mismatch"):
                estimator.manifest_report_sha256_and_payload(
                    path,
                    "0" * 64,
                    manifest_sha256="a" * 64,
                    rows=[],
                )

    def test_manifest_report_proves_all_candidate_filter_then_lowest_hash_selection(self) -> None:
        audit = []
        for track_id in range(1, estimator.EXPECTED_ORIGINAL_CANDIDATES + 1):
            audit.append(
                {
                    "fma_track_id": track_id,
                    "relative_audio_path": f"000/{track_id:06d}.mp3",
                    "source_audio_sha256": hashlib.sha256(f"s{track_id}".encode()).hexdigest(),
                    "decoded_duration_frames": 30_000,
                    "decoded_sample_rate": 1_000,
                    "decoded_channels": 2,
                    "segment_start_frame": 0,
                    "segment_num_frames": 10_000,
                    "segment_offset_sha256": estimator._segment_hash(track_id),
                    "extracted_pcm_sha256": hashlib.sha256(f"p{track_id}".encode()).hexdigest(),
                    "pcm_hash_encoding": "frames_x_channels_float32_little_endian_c_order",
                    "all_pcm_samples_finite": True,
                    "pre_downmix_rms": 0.1,
                    "mono_rms": 0.05,
                    "mono_compatibility_ratio": 0.5,
                    "lr_pearson_correlation": 0.2,
                    "eligibility_codec_input_pcm_sha256": hashlib.sha256(
                        f"c{track_id}".encode()
                    ).hexdigest(),
                    "eligible": True,
                    "failed_conditions": [],
                }
            )
        selected = sorted(
            audit,
            key=lambda row: (
                estimator._selection_hash(row["fma_track_id"]),
                row["fma_track_id"],
            ),
        )[: estimator.EXPECTED_ITEMS]
        rows = []
        for index, audit_row in enumerate(selected):
            row = dict(audit_row)
            row.update(
                {
                    "schema_version": estimator.REQUIRED_MANIFEST_SCHEMA,
                    "protocol_label": estimator.PROTOCOL_LABEL,
                    "selection_rank": index,
                    "selection_rank_sha256": estimator._selection_hash(row["fma_track_id"]),
                    "eligibility_thresholds": {
                        "pre_downmix_rms_denominator": estimator.PRE_DOWNMIX_RMS_DENOMINATOR,
                        "minimum_mono_rms": estimator.MIN_MONO_RMS,
                        "minimum_mono_compatibility_ratio": estimator.MIN_MONO_COMPATIBILITY_RATIO,
                    },
                    "archive_sha1": estimator.REQUIRED_ARCHIVE_SHA1,
                    "archive_sha1_verified": True,
                    "publication_eligible": True,
                }
            )
            rows.append(row)
        report = {
            "schema_version": estimator.REQUIRED_MANIFEST_SCHEMA,
            "protocol_label": estimator.PROTOCOL_LABEL,
            "manifest_sha256": "a" * 64,
            "original_candidate_tracks": estimator.EXPECTED_ORIGINAL_CANDIDATES,
            "expected_original_candidate_tracks": estimator.EXPECTED_ORIGINAL_CANDIDATES,
            "input_eligible_tracks": estimator.EXPECTED_ORIGINAL_CANDIDATES,
            "selected_tracks": estimator.EXPECTED_ITEMS,
            "selection_count_contract": estimator.EXPECTED_ITEMS,
            "selection_count_used": estimator.EXPECTED_ITEMS,
            "split_seed": estimator.SELECTION_SEED,
            "selection_domain": estimator.SELECTION_DOMAIN,
            "segment_domain": estimator.SEGMENT_DOMAIN,
            "segment_seconds": 10,
            "archive_sha1": estimator.REQUIRED_ARCHIVE_SHA1,
            "archive_sha1_verified": True,
            "publication_eligible": True,
            "eligibility": {
                "stage": "pre_codec_pre_marginal",
                "all_pcm_samples_finite_required": True,
                "pre_downmix_rms_denominator": estimator.PRE_DOWNMIX_RMS_DENOMINATOR,
                "minimum_mono_rms": estimator.MIN_MONO_RMS,
                "minimum_mono_compatibility_ratio": estimator.MIN_MONO_COMPATIBILITY_RATIO,
                "codec_sample_rate": estimator.FROZEN_CODEC_SAMPLE_RATE,
                "codec_channels": estimator.FROZEN_CODEC_CHANNELS,
                "codec_input_frames": estimator.FROZEN_CODEC_INPUT_FRAMES,
                "conversion": "audiocraft.data.audio_utils.convert_audio",
                "lr_pearson_is_diagnostic_only": True,
                "track_id_filtering": False,
                "audited_tracks": estimator.EXPECTED_ORIGINAL_CANDIDATES,
            },
            "eligibility_audit": audit,
            "implementation": {
                "builder_sha256": estimator.sha256_file(
                    estimator.WORKPACK_ROOT / "scripts" / "build_fma_calibration_manifest.py"
                )
            },
            "audiocraft": {
                "base_commit": estimator.PINNED_AUDIOCRAFT_BASE_COMMIT,
                "source_identity": {"schema_version": "fixture"},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a1-r2" / estimator.REQUIRED_REPORT_BASENAME
            path.parent.mkdir()
            payload = (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode()
            path.write_bytes(payload)
            observed, _ = estimator.manifest_report_sha256_and_payload(
                path,
                hashlib.sha256(payload).hexdigest(),
                manifest_sha256="a" * 64,
                rows=rows,
            )
            self.assertEqual(observed, hashlib.sha256(payload).hexdigest())

            report["eligibility"]["track_id_filtering"] = True
            tampered = (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode()
            path.write_bytes(tampered)
            with self.assertRaisesRegex(ValueError, "eligibility contract"):
                estimator.manifest_report_sha256_and_payload(
                    path,
                    hashlib.sha256(tampered).hexdigest(),
                    manifest_sha256="a" * 64,
                    rows=rows,
                )

    def test_manifest_hash_and_512_row_contract_are_checked(self) -> None:
        rows = []
        track_ids = sorted(range(1, 513), key=estimator._selection_hash)
        for index, track_id in enumerate(track_ids):
            rows.append(
                {
                    "schema_version": estimator.REQUIRED_MANIFEST_SCHEMA,
                    "protocol_label": estimator.PROTOCOL_LABEL,
                    "selection_rank": index,
                    "selection_rank_sha256": estimator._selection_hash(track_id),
                    "fma_track_id": track_id,
                    "relative_audio_path": f"000/{track_id:06d}.mp3",
                    "source_audio_sha256": hashlib.sha256(f"source-{index}".encode()).hexdigest(),
                    "decoded_duration_frames": 30_000,
                    "decoded_sample_rate": 1_000,
                    "decoded_channels": 2,
                    "segment_start_frame": index % 20_001,
                    "segment_num_frames": 10_000,
                    "segment_offset_sha256": estimator._segment_hash(track_id),
                    "extracted_pcm_sha256": hashlib.sha256(f"pcm-{index}".encode()).hexdigest(),
                    "pcm_hash_encoding": "frames_x_channels_float32_little_endian_c_order",
                    "all_pcm_samples_finite": True,
                    "pre_downmix_rms": 0.1,
                    "mono_rms": 0.05,
                    "mono_compatibility_ratio": 0.5,
                    "lr_pearson_correlation": 0.25,
                    "eligibility_codec_input_pcm_sha256": hashlib.sha256(
                        f"codec-pcm-{index}".encode()
                    ).hexdigest(),
                    "eligible": True,
                    "failed_conditions": [],
                    "eligibility_thresholds": {
                        "pre_downmix_rms_denominator": estimator.PRE_DOWNMIX_RMS_DENOMINATOR,
                        "minimum_mono_rms": estimator.MIN_MONO_RMS,
                        "minimum_mono_compatibility_ratio": estimator.MIN_MONO_COMPATIBILITY_RATIO,
                    },
                    "archive_sha1": "ade154f733639d52e35e32f5593efe5be76c6d70",
                    "archive_sha1_verified": True,
                    "publication_eligible": True,
                }
            )
        payload = b"".join(
            (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
            for row in rows
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a1-r2" / "codec_calibration.train.jsonl"
            path.parent.mkdir()
            path.write_bytes(payload)
            observed_hash, observed_rows = estimator.manifest_sha256_and_rows(
                path, hashlib.sha256(payload).hexdigest()
            )
            self.assertEqual(observed_hash, hashlib.sha256(payload).hexdigest())
            self.assertEqual(len(observed_rows), 512)
            with self.assertRaises(RuntimeError):
                estimator.manifest_sha256_and_rows(path, "0" * 64)


if __name__ == "__main__":
    unittest.main()
