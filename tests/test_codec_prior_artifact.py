from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Callable, Optional


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "codec_prior_artifact", ROOT / "src" / "ptc_opd" / "codec_prior_artifact.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

ARTIFACT_SEAL_NAME = MODULE.ARTIFACT_SEAL_NAME
ARTIFACT_SEAL_SCHEMA = MODULE.ARTIFACT_SEAL_SCHEMA
CodecPriorArtifactError = MODULE.CodecPriorArtifactError
MANIFEST_SCHEMA = MODULE.MANIFEST_SCHEMA
PER_CLIP_NAME = MODULE.PER_CLIP_NAME
PRIOR_NAME = MODULE.PRIOR_NAME
RESULT_SCHEMA = MODULE.RESULT_SCHEMA
SUMMARY_NAME = MODULE.SUMMARY_NAME
canonical_json_sha256 = MODULE.canonical_json_sha256
load_codec_prior_artifact = MODULE.load_codec_prior_artifact
verify_local_codec_snapshot = MODULE.verify_local_codec_snapshot
EXPECTED_ARTIFACT_DIRECTORY_BASENAME = MODULE.EXPECTED_ARTIFACT_DIRECTORY_BASENAME


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _valid_protocol() -> dict:
    return {
        "identity_schema": MODULE.IDENTITY_SCHEMA,
        "label": MODULE.PROTOCOL_LABEL,
        "manifest_schema": MANIFEST_SCHEMA,
        "result_schema": RESULT_SCHEMA,
        "artifact_seal_schema": ARTIFACT_SEAL_SCHEMA,
        "manifest_report": {
            "schema_version": MANIFEST_SCHEMA,
            "sha256": "e" * 64,
        },
        "implementation": {
            "schema_version": MODULE.IMPLEMENTATION_IDENTITY_SCHEMA,
            "files": [
                {"relative_path": path, "sha256": str(index + 1) * 64}
                for index, path in enumerate(MODULE.EXPECTED_IMPLEMENTATION_PATHS)
            ],
        },
    }


def _valid_prior_payload() -> dict:
    source_identity = {
        "schema_version": MODULE.AUDIOCRAFT_SOURCE_SCHEMA,
        "file_count": 7,
        "tree_sha256": "d" * 64,
    }
    source_identity["identity_sha256"] = canonical_json_sha256(source_identity)
    leave_one_out = [
        {
            "manifest_index": index,
            "fma_track_id": 100_000 + index,
            "raw_mean_marginal_without_clip": [4.0, 3.0, 2.0, 1.0],
            "prior_without_clip": [0.4, 0.3, 0.2, 0.1],
            "total_variation_from_primary_prior": 0.0,
        }
        for index in range(512)
    ]
    return {
        "schema_version": RESULT_SCHEMA,
        "protocol": _valid_protocol(),
        "n_clips": 512,
        "num_codebooks": 4,
        "raw_mean_marginal": [4.0, 3.0, 2.0, 1.0],
        "clipped_mean_marginal": [4.0, 3.0, 2.0, 1.0],
        "prior": [0.4, 0.3, 0.2, 0.1],
        "bootstrap": {
            "replicates": 10_000,
            "seed": 4701,
            "total_variation_from_uniform_ci95_low": 0.10,
        },
        "split_half": {
            "domain": "ptc-opd-codec-split-v1",
            "seed": 4701,
            "half_a_n": 256,
            "half_b_n": 256,
            "kendall_tau_b": 1.0,
        },
        "leave_one_out_influence": {
            "definition": "TV(primary_prior, prior_recomputed_without_one_clip)",
            "max_total_variation": 0.0,
            "max_manifest_index": 0,
            "max_fma_track_id": 100_000,
            "per_clip": leave_one_out,
        },
        "sensitivity": {
            "primary_estimator": "arithmetic_mean",
            "median": {
                "marginal": [4.0, 3.0, 2.0, 1.0],
                "prior": [0.4, 0.3, 0.2, 0.1],
                "kendall_tau_b_vs_primary": 1.0,
            },
            "trimmed_mean_1pct": {
                "definition": "sort each codebook independently; remove floor(0.01*N) from each tail",
                "trim_each_tail": 5,
                "retained_clips_per_codebook": 502,
                "marginal": [4.0, 3.0, 2.0, 1.0],
                "prior": [0.4, 0.3, 0.2, 0.1],
                "kendall_tau_b_vs_primary": 1.0,
            },
            "use_for_primary_prior": False,
        },
        "diagnostics": {
            "denominator_floor_activation": 0,
            "total_variation_from_uniform": 0.2,
        },
        "metric": {
            "id": "mrstft_spectral_convergence_plus_log_magnitude_l1",
            "fft_sizes": [512, 1024, 2048],
            "hop_ratio": 0.25,
            "window": "Hann, win_length equals FFT size",
            "distance_epsilon": 1.0e-7,
            "reference_norm_guard": "raise if any per-channel FFT norm <= distance_epsilon",
            "denominator_flooring": False,
            "denominator_floor_activation": 0,
            "marginal_epsilon": 1.0e-8,
        },
        "acceptance_gates": {
            "thresholds": {
                "bootstrap_tv_ci95_low_strictly_greater_than": 0.05,
                "kendall_tau_b_minimum": 2.0 / 3.0,
                "kendall_tau_b_minimum_exact": "2/3",
                "max_leave_one_out_total_variation": 0.05,
            },
            "scientific": {
                "bootstrap_tv_ci95_low_gt_0_05": True,
                "split_half_kendall_tau_b_ge_two_thirds": True,
                "max_leave_one_out_tv_le_0_05": True,
                "median_rank_kendall_tau_b_ge_two_thirds": True,
                "trimmed_rank_kendall_tau_b_ge_two_thirds": True,
            },
            "scientific_status": "pass",
        },
        "manifest_sha256": "a" * 64,
        "manifest_report_path": "/inputs/a1-r2/codec_calibration.train.report.json",
        "manifest_report_sha256": "e" * 64,
        "codec_checkpoint_sha256": "b" * 64,
        "codec_load": {
            "load_mode": "self_contained",
            "pretrained_model_id": None,
            "resolved_snapshot_root": None,
            "resolved_snapshot_revision": None,
            "resolved_snapshot_files": [],
        },
        "audiocraft_commit": MODULE.PINNED_AUDIOCRAFT_BASE_COMMIT,
        "audiocraft_source_identity": source_identity,
        "codec_contract": {
            "channels": 1,
            "sample_rate": 32_000,
            "frame_rate": 50.0,
            "cardinality": 2_048,
            "num_codebooks": 4,
            "input_frames": 320_000,
            "canonical_code_shape": [1, 4, 500],
            "validated_for_every_clip": True,
        },
    }


def _write_artifact(
    root: Path,
    *,
    mutate_prior: Optional[Callable[[dict], None]] = None,
    mutate_seal: Optional[Callable[[dict], None]] = None,
) -> Path:
    root = root.parent / EXPECTED_ARTIFACT_DIRECTORY_BASENAME
    root.mkdir()
    per_clip = b"sealed per-clip payload\n"
    summary = b"codebook_index,prior\n0,0.4\n"
    (root / PER_CLIP_NAME).write_bytes(per_clip)
    (root / SUMMARY_NAME).write_bytes(summary)

    prior = _valid_prior_payload()
    prior["artifact_payload_sha256"] = {
        PER_CLIP_NAME: _sha256(per_clip),
        SUMMARY_NAME: _sha256(summary),
    }
    if mutate_prior is not None:
        mutate_prior(prior)
    prior_bytes = (
        json.dumps(
            prior,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=True,
        )
        + "\n"
    ).encode("utf-8")
    (root / PRIOR_NAME).write_bytes(prior_bytes)

    seal = {
        "schema_version": ARTIFACT_SEAL_SCHEMA,
        "status": "complete",
        "manifest": {
            "schema_version": MANIFEST_SCHEMA,
            "path": "/inputs/a1-r2/codec_calibration.train.jsonl",
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
            "archive_sha1": "ade154f733639d52e35e32f5593efe5be76c6d70",
            "items": 512,
            "ordered_source_records_sha256": "c" * 64,
            "manifest_report_sha256": "e" * 64,
        },
        "audiocraft": {
            "base_commit": MODULE.PINNED_AUDIOCRAFT_BASE_COMMIT,
            "source_identity": dict(prior["audiocraft_source_identity"]),
        },
        "codec_load": {
            "load_mode": "self_contained",
            "pretrained_model_id": None,
            "resolved_snapshot_revision": None,
            "resolved_snapshot_files": [],
        },
        "protocol": dict(prior["protocol"]),
        "artifacts": {
            name: {
                "sha256": _sha256((root / name).read_bytes()),
                "size_bytes": (root / name).stat().st_size,
            }
            for name in (PER_CLIP_NAME, SUMMARY_NAME, PRIOR_NAME)
        },
    }
    if mutate_seal is not None:
        mutate_seal(seal)
    (root / ARTIFACT_SEAL_NAME).write_text(
        json.dumps(seal, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return root


class CodecPriorArtifactTest(unittest.TestCase):
    def test_valid_artifact_returns_normalized_prior_and_portable_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = _write_artifact(Path(temporary) / "a1")
            result = load_codec_prior_artifact(artifact)

            self.assertEqual(result.prior, (0.4, 0.3, 0.2, 0.1))
            self.assertTrue(math.isclose(sum(result.prior), 1.0))
            self.assertEqual(result.manifest_sha256, "a" * 64)
            self.assertEqual(result.checkpoint_sha256, "b" * 64)
            self.assertEqual(result.ordered_source_records_sha256, "c" * 64)
            self.assertEqual(result.codec_prior_sha256, result.identity["artifacts"][PRIOR_NAME]["sha256"])
            self.assertEqual(result.codec_load["load_mode"], "self_contained")
            self.assertEqual(result.identity_sha256, canonical_json_sha256(result.identity))
            self.assertNotIn("directory", result.identity)
            self.assertEqual(result.identity["protocol"]["label"], "A1-R2")

    def test_requires_the_formal_a1_r2_directory_and_protocol_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = _write_artifact(Path(temporary) / "a1")
            wrong_name = artifact.with_name("phase_a1_codec_prior")
            artifact.rename(wrong_name)
            with self.assertRaisesRegex(CodecPriorArtifactError, "directory basename"):
                load_codec_prior_artifact(wrong_name)

        mutations = (
            (
                lambda payload: payload["protocol"].__setitem__("label", "A1-R1"),
                "protocol label",
            ),
            (
                lambda payload: payload.__setitem__(
                    "manifest_report_path", "/inputs/codec_calibration.train.report.json"
                ),
                "manifest-report path",
            ),
            (
                lambda payload: payload.__setitem__("manifest_report_sha256", "f" * 64),
                "manifest-report hash",
            ),
        )
        for mutate, message in mutations:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                artifact = _write_artifact(
                    Path(temporary) / "a1", mutate_prior=mutate
                )
                with self.assertRaisesRegex(CodecPriorArtifactError, message):
                    load_codec_prior_artifact(artifact)

        with tempfile.TemporaryDirectory() as temporary:
            artifact = _write_artifact(
                Path(temporary) / "a1",
                mutate_seal=lambda seal: seal["protocol"].__setitem__(
                    "label", "A1-R1"
                ),
            )
            with self.assertRaisesRegex(CodecPriorArtifactError, "protocol label"):
                load_codec_prior_artifact(artifact)

    def test_rejects_any_failed_or_drifted_a1_r2_scientific_gate(self) -> None:
        mutations = (
            (
                lambda payload: payload["metric"].__setitem__("denominator_flooring", True),
                "denominator_flooring",
            ),
            (
                lambda payload: payload["metric"].__setitem__(
                    "denominator_floor_activation", 1
                ),
                "denominator_floor_activation",
            ),
            (
                lambda payload: payload["diagnostics"].__setitem__(
                    "total_variation_from_uniform", 0.1
                ),
                "diagnostic total variation",
            ),
            (
                lambda payload: payload["bootstrap"].__setitem__(
                    "total_variation_from_uniform_ci95_low", 0.05
                ),
                "strictly greater",
            ),
            (
                lambda payload: payload["split_half"].__setitem__(
                    "kendall_tau_b", 0.5
                ),
                "split-half Kendall",
            ),
            (
                lambda payload: payload["leave_one_out_influence"].__setitem__(
                    "max_total_variation", 0.051
                ),
                "maximum leave-one-out",
            ),
            (
                lambda payload: payload["leave_one_out_influence"]["per_clip"][0].__setitem__(
                    "prior_without_clip", [0.1, 0.2, 0.3, 0.4]
                ),
                "frozen arithmetic",
            ),
            (
                lambda payload: payload["leave_one_out_influence"]["per_clip"][0].__setitem__(
                    "total_variation_from_primary_prior", 0.01
                ),
                "total variation is inconsistent",
            ),
            (
                lambda payload: payload["sensitivity"]["median"].__setitem__(
                    "kendall_tau_b_vs_primary", 0.5
                ),
                "median Kendall",
            ),
            (
                lambda payload: payload["sensitivity"]["median"].__setitem__(
                    "prior", [0.1, 0.2, 0.3, 0.4]
                ),
                "frozen arithmetic",
            ),
            (
                lambda payload: payload["sensitivity"]["trimmed_mean_1pct"].__setitem__(
                    "trim_each_tail", 4
                ),
                "trim_each_tail",
            ),
            (
                lambda payload: payload["acceptance_gates"]["scientific"].__setitem__(
                    "max_leave_one_out_tv_le_0_05", False
                ),
                "must be true",
            ),
            (
                lambda payload: payload["acceptance_gates"].__setitem__(
                    "scientific_status", "fail_stop_review"
                ),
                "scientific status",
            ),
        )
        for mutate, message in mutations:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                artifact = _write_artifact(
                    Path(temporary) / "a1", mutate_prior=mutate
                )
                with self.assertRaisesRegex(CodecPriorArtifactError, message):
                    load_codec_prior_artifact(artifact)

    def test_requires_the_exact_four_regular_members(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = _write_artifact(Path(temporary) / "extra")
            (artifact / "notes.txt").write_text("not part of the seal", encoding="utf-8")
            with self.assertRaisesRegex(CodecPriorArtifactError, "members differ"):
                load_codec_prior_artifact(artifact)

        with tempfile.TemporaryDirectory() as temporary:
            artifact = _write_artifact(Path(temporary) / "missing")
            (artifact / SUMMARY_NAME).unlink()
            with self.assertRaisesRegex(CodecPriorArtifactError, "members differ"):
                load_codec_prior_artifact(artifact)

    def test_rehashes_every_payload_and_checks_size(self) -> None:
        for name in (PER_CLIP_NAME, SUMMARY_NAME, PRIOR_NAME):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                artifact = _write_artifact(Path(temporary) / "a1")
                with (artifact / name).open("ab") as stream:
                    stream.write(b"tamper")
                with self.assertRaisesRegex(CodecPriorArtifactError, "hash/size mismatch"):
                    load_codec_prior_artifact(artifact)

        with tempfile.TemporaryDirectory() as temporary:
            artifact = _write_artifact(
                Path(temporary) / "a1",
                mutate_seal=lambda seal: seal["artifacts"][SUMMARY_NAME].__setitem__(
                    "size_bytes", True
                ),
            )
            with self.assertRaisesRegex(CodecPriorArtifactError, "non-negative integer"):
                load_codec_prior_artifact(artifact)

    def test_rejects_bad_seal_schema_status_and_identity_shape(self) -> None:
        mutations = (
            (lambda seal: seal.__setitem__("schema_version", "wrong"), "schema"),
            (lambda seal: seal.__setitem__("status", "running"), "complete"),
            (lambda seal: seal["manifest"].__setitem__("items", 511), "manifest.items"),
            (lambda seal: seal["manifest"].__setitem__("schema_version", "wrong"), "manifest schema"),
            (lambda seal: seal["source_identity"].__setitem__("items", 511), "source_identity.items"),
            (lambda seal: seal["source_identity"].__setitem__("archive_sha1", "0" * 40), "archive SHA-1"),
            (lambda seal: seal["audiocraft"].__setitem__("base_commit", "0" * 40), "base commit"),
            (lambda seal: seal["audiocraft"]["source_identity"].__setitem__("tree_sha256", "e" * 64), "identity hash"),
        )
        for mutate, message in mutations:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                artifact = _write_artifact(
                    Path(temporary) / "a1", mutate_seal=mutate
                )
                with self.assertRaisesRegex(CodecPriorArtifactError, message):
                    load_codec_prior_artifact(artifact)

    def test_rejects_every_frozen_prior_and_codec_contract_drift(self) -> None:
        def set_nested(*keys_and_value: object) -> Callable[[dict], None]:
            *keys, value = keys_and_value

            def mutate(payload: dict) -> None:
                cursor = payload
                for key in keys[:-1]:
                    cursor = cursor[key]
                cursor[keys[-1]] = value

            return mutate

        mutations = (
            (set_nested("schema_version", "wrong"), "payload schema"),
            (set_nested("n_clips", 511), "n_clips"),
            (set_nested("num_codebooks", 3), "num_codebooks"),
            (set_nested("bootstrap", "replicates", 9999), "bootstrap.replicates"),
            (set_nested("bootstrap", "seed", 4702), "bootstrap.seed"),
            (set_nested("codec_contract", "channels", 2), "channels"),
            (set_nested("codec_contract", "sample_rate", 44100), "sample_rate"),
            (set_nested("codec_contract", "frame_rate", 25.0), "frame_rate"),
            (set_nested("codec_contract", "cardinality", 1024), "cardinality"),
            (set_nested("codec_contract", "num_codebooks", 8), "num_codebooks"),
            (set_nested("codec_contract", "input_frames", 319999), "input_frames"),
            (set_nested("codec_contract", "canonical_code_shape", [1, 4, 499]), "canonical_code_shape"),
            (set_nested("codec_contract", "validated_for_every_clip", False), "validated_for_every_clip"),
            (set_nested("prior", [4.0, 3.0, 2.0, 1.0]), "sum to one"),
            (set_nested("prior", [0.1, 0.2, 0.3, 0.4]), "frozen arithmetic"),
            (set_nested("clipped_mean_marginal", [5.0, 3.0, 2.0, 1.0]), "frozen arithmetic"),
            (set_nested("prior", [0.5, 0.5, 0.0, 0.0]), "strictly positive"),
            (set_nested("prior", [0.5, 0.5, 0.0]), "exactly four"),
        )
        for mutate, message in mutations:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                artifact = _write_artifact(
                    Path(temporary) / "a1", mutate_prior=mutate
                )
                with self.assertRaisesRegex(CodecPriorArtifactError, message):
                    load_codec_prior_artifact(artifact)

    def test_cross_checks_prior_manifest_checkpoint_and_internal_payload_hashes(self) -> None:
        mutations = (
            (lambda payload: payload.__setitem__("manifest_sha256", "d" * 64), "manifest hash"),
            (lambda payload: payload.__setitem__("codec_checkpoint_sha256", "d" * 64), "checkpoint hash"),
            (
                lambda payload: payload["artifact_payload_sha256"].__setitem__(
                    PER_CLIP_NAME, "d" * 64
                ),
                "internal artifact hash",
            ),
        )
        for mutate, message in mutations:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                artifact = _write_artifact(
                    Path(temporary) / "a1", mutate_prior=mutate
                )
                with self.assertRaisesRegex(CodecPriorArtifactError, message):
                    load_codec_prior_artifact(artifact)

    def test_binds_codec_snapshot_and_repeated_audiocraft_provenance(self) -> None:
        mutations = (
            (
                lambda payload: payload.__setitem__(
                    "audiocraft_commit", "0" * 40
                ),
                "AudioCraft commit",
            ),
            (
                lambda payload: payload["audiocraft_source_identity"].__setitem__(
                    "tree_sha256", "e" * 64
                ),
                "AudioCraft source identity",
            ),
            (
                lambda payload: payload["codec_load"].__setitem__(
                    "load_mode", "pretrained_indirection"
                ),
                "pretrained codec model ID",
            ),
        )
        for mutate, message in mutations:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                artifact = _write_artifact(
                    Path(temporary) / "a1", mutate_prior=mutate
                )
                with self.assertRaisesRegex(CodecPriorArtifactError, message):
                    load_codec_prior_artifact(artifact)

        def make_pretrained(payload: dict) -> None:
            payload["codec_load"] = {
                "load_mode": "pretrained_indirection",
                "pretrained_model_id": MODULE.ALLOWED_PRETRAINED_CODEC_ID,
                "resolved_snapshot_root": "/cache/snapshots/" + "1" * 40,
                "resolved_snapshot_revision": "1" * 40,
                "resolved_snapshot_files": [
                    {
                        "relative_path": "config.json",
                        "visible_absolute_path": "/cache/snapshots/config.json",
                        "resolved_absolute_path": "/cache/blobs/a",
                        "is_symlink": True,
                        "size_bytes": 17,
                        "sha256": "2" * 64,
                    }
                ],
            }

        def seal_pretrained(seal: dict) -> None:
            seal["checkpoint"].update(
                {
                    "load_mode": "pretrained_indirection",
                    "pretrained_model_id": MODULE.ALLOWED_PRETRAINED_CODEC_ID,
                    "resolved_snapshot_revision": "1" * 40,
                }
            )
            seal["codec_load"] = {
                "load_mode": "pretrained_indirection",
                "pretrained_model_id": MODULE.ALLOWED_PRETRAINED_CODEC_ID,
                "resolved_snapshot_revision": "1" * 40,
                "resolved_snapshot_files": [
                    {
                        "relative_path": "config.json",
                        "size_bytes": 17,
                        "sha256": "2" * 64,
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as temporary:
            artifact = _write_artifact(
                Path(temporary) / "a1",
                mutate_prior=make_pretrained,
                mutate_seal=seal_pretrained,
            )
            result = load_codec_prior_artifact(artifact)
            self.assertEqual(
                result.codec_load["resolved_snapshot_files"][0]["sha256"],
                "2" * 64,
            )

    def test_rehashes_the_exact_local_pretrained_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            revision = "1" * 40
            snapshot = root / "models--facebook--encodec_32khz" / "snapshots" / revision
            snapshot.mkdir(parents=True)
            config = snapshot / "config.json"
            config.write_bytes(b'{"model":"encodec"}\n')
            weights = snapshot / "model.safetensors"
            weights.write_bytes(b"sealed weights")

            def make_pretrained(payload: dict) -> None:
                payload["codec_load"] = {
                    "load_mode": "pretrained_indirection",
                    "pretrained_model_id": MODULE.ALLOWED_PRETRAINED_CODEC_ID,
                    "resolved_snapshot_root": str(snapshot),
                    "resolved_snapshot_revision": revision,
                    "resolved_snapshot_files": [
                        {
                            "relative_path": path.name,
                            "visible_absolute_path": str(path),
                            "resolved_absolute_path": str(path),
                            "is_symlink": False,
                            "size_bytes": path.stat().st_size,
                            "sha256": MODULE.sha256_file(path),
                        }
                        for path in (config, weights)
                    ],
                }

            def seal_pretrained(seal: dict) -> None:
                seal["checkpoint"].update(
                    {
                        "load_mode": "pretrained_indirection",
                        "pretrained_model_id": MODULE.ALLOWED_PRETRAINED_CODEC_ID,
                        "resolved_snapshot_revision": revision,
                    }
                )
                seal["codec_load"] = {
                    "load_mode": "pretrained_indirection",
                    "pretrained_model_id": MODULE.ALLOWED_PRETRAINED_CODEC_ID,
                    "resolved_snapshot_revision": revision,
                    "resolved_snapshot_files": [
                        {
                            "relative_path": path.name,
                            "size_bytes": path.stat().st_size,
                            "sha256": MODULE.sha256_file(path),
                        }
                        for path in (config, weights)
                    ],
                }

            artifact_path = _write_artifact(
                root / "a1",
                mutate_prior=make_pretrained,
                mutate_seal=seal_pretrained,
            )
            artifact = load_codec_prior_artifact(artifact_path)
            resolved = verify_local_codec_snapshot(
                artifact,
                cache_resolver=lambda **_: str(config),
            )
            self.assertEqual(resolved["resolved_snapshot_revision"], revision)

            weights.write_bytes(b"drift")
            with self.assertRaisesRegex(
                CodecPriorArtifactError, "differ from the sealed"
            ):
                verify_local_codec_snapshot(
                    artifact,
                    cache_resolver=lambda **_: str(config),
                )

    def test_rejects_nonfinite_json_and_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = _write_artifact(
                Path(temporary) / "nan",
                mutate_prior=lambda payload: payload.__setitem__(
                    "prior", [float("nan"), 0.3, 0.2, 0.1]
                ),
            )
            with self.assertRaisesRegex(CodecPriorArtifactError, "non-finite"):
                load_codec_prior_artifact(artifact)

        with tempfile.TemporaryDirectory() as temporary:
            artifact = _write_artifact(Path(temporary) / "duplicate")
            seal_path = artifact / ARTIFACT_SEAL_NAME
            original = seal_path.read_text(encoding="utf-8").rstrip()
            duplicate = original[:-1] + ',"status":"complete"}\n'
            seal_path.write_text(duplicate, encoding="utf-8")
            with self.assertRaisesRegex(CodecPriorArtifactError, "duplicate key"):
                load_codec_prior_artifact(artifact)


if __name__ == "__main__":
    unittest.main()
