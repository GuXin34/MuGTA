import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ptc_opd.stage1_artifact import (
    Stage1ArtifactError,
    artifact_member,
    canonical_json_bytes,
    publish_closed_files_artifact,
    publish_closed_json_artifact,
    sha256_file,
    sha256_tree,
)
from ptc_opd import stage1_diversity_fad as contract


_REFERENCE_BUILDER_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "build_stage1_fad_reference.py"
)
_REFERENCE_BUILDER_SPEC = importlib.util.spec_from_file_location(
    "stage1_fad_reference_builder_test", _REFERENCE_BUILDER_PATH
)
assert _REFERENCE_BUILDER_SPEC is not None and _REFERENCE_BUILDER_SPEC.loader is not None
reference_builder = importlib.util.module_from_spec(_REFERENCE_BUILDER_SPEC)
_REFERENCE_BUILDER_SPEC.loader.exec_module(reference_builder)


def _write_json(path, value):
    path.write_bytes(canonical_json_bytes(value))


def _write_jsonl(path, rows):
    path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))


def _make_fake_mert_snapshot(root):
    snapshot = root / contract.MERT_REVISION
    snapshot.mkdir()
    payloads = {
        "config.json": b'{"model_type":"mert"}\n',
        "configuration_MERT.py": b"class MERTConfig: pass\n",
        "modeling_MERT.py": b"class MERTModel: pass\n",
        "preprocessor_config.json": b'{"sampling_rate":24000}\n',
        "pytorch_model.bin": b"small-test-weight-payload",
    }
    pins = {}
    for name, payload in payloads.items():
        path = snapshot / name
        path.write_bytes(payload)
        pins[name] = {
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return snapshot, pins


def _a1_rows():
    return [
        {
            "schema_version": "ptc-opd-fma-calibration-v2",
            "protocol_label": "A1-R2",
            "eligible": True,
            "publication_eligible": True,
            "fma_track_id": index + 1,
            "relative_audio_path": "{:03d}/{:06d}.mp3".format(
                (index + 1) // 1000, index + 1
            ),
            "source_audio_sha256": hashlib.sha256(
                "source-{}".format(index).encode()
            ).hexdigest(),
            "extracted_pcm_sha256": hashlib.sha256(
                "pcm-{}".format(index).encode()
            ).hexdigest(),
            "segment_start_frame": index,
            "segment_num_frames": 441000,
            "decoded_sample_rate": 44100,
            "decoded_channels": 2,
        }
        for index in range(512)
    ]


def _make_a1(root):
    parent = root / "a1-r2"
    parent.mkdir()
    manifest = parent / "codec_calibration.train.jsonl"
    rows = _a1_rows()
    _write_jsonl(manifest, rows)
    report = parent / "codec_calibration.train.report.json"
    _write_json(
        report,
        {
            "schema_version": "ptc-opd-fma-calibration-v2",
            "protocol_label": "A1-R2",
            "manifest_sha256": sha256_file(manifest),
            "selected_tracks": 512,
            "publication_eligible": True,
        },
    )
    return manifest, report, rows


def _make_reference(root, a1_manifest, a1_report, source_rows):
    directory = root / "reference"
    audio = directory / "audio"
    audio.mkdir(parents=True)
    selected = contract.select_reference_rows(source_rows)
    rows = []
    for rank, source in enumerate(selected):
        track_id = source["fma_track_id"]
        relative = "audio/{:06d}.wav".format(track_id)
        target = directory / relative
        target.write_bytes("wav-{}".format(track_id).encode())
        rows.append(
            {
                "schema_version": contract.REFERENCE_RECORD_SCHEMA,
                "selection_rank": rank,
                "selection_hash": contract.reference_selection_hash(track_id),
                "fma_track_id": track_id,
                "relative_audio_path": source["relative_audio_path"],
                "source_audio_sha256": source["source_audio_sha256"],
                "source_pcm_sha256": source["extracted_pcm_sha256"],
                "segment_start_frame": source["segment_start_frame"],
                "segment_num_frames": source["segment_num_frames"],
                "sample_rate": source["decoded_sample_rate"],
                "channels": source["decoded_channels"],
                "duration_seconds": 10,
                "path": relative,
                "audio_sha256": sha256_file(target),
                "audio_frames": source["segment_num_frames"],
                "audio_channels": source["decoded_channels"],
                "audio_sample_rate": source["decoded_sample_rate"],
                "audio_subtype": "FLOAT",
            }
        )
    _write_jsonl(directory / contract.REFERENCE_MANIFEST, rows)
    tree_hash = sha256_tree(audio)
    report_payload = {
        "schema_version": contract.REFERENCE_SCHEMA,
        "status": "complete_fad_reference",
        "source": {
            "a1_r2_manifest_basename": a1_manifest.name,
            "a1_r2_manifest_sha256": sha256_file(a1_manifest),
            "a1_r2_report_basename": a1_report.name,
            "a1_r2_report_sha256": sha256_file(a1_report),
            "candidate_count": 512,
        },
        "selection": {
            "algorithm": "lowest_sha256_then_track_id",
            "domain": contract.REFERENCE_SELECTION_DOMAIN,
            "seed": contract.REFERENCE_SELECTION_SEED,
            "count": 256,
            "source_population": "A1-R2 512 eligible 10-second segments",
        },
        "record_count": 256,
        "duration_seconds": 10,
        "manifest": artifact_member(directory / contract.REFERENCE_MANIFEST),
        "audio_tree_sha256": tree_hash,
    }
    _write_json(directory / contract.REFERENCE_REPORT, report_payload)
    _write_json(
        directory / contract.SEAL_NAME,
        {
            "schema_version": contract.REFERENCE_SEAL_SCHEMA,
            "status": "complete_fad_reference",
            "record_count": 256,
            "members": {
                contract.REFERENCE_MANIFEST: artifact_member(
                    directory / contract.REFERENCE_MANIFEST
                ),
                contract.REFERENCE_REPORT: artifact_member(
                    directory / contract.REFERENCE_REPORT
                ),
            },
            "audio_tree_sha256": tree_hash,
        },
    )
    return directory


def _make_generation_stub(root):
    directory = root / "generation"
    directory.mkdir()
    (directory / "anchor").write_bytes(b"immutable-generation")
    rows = []
    for index in range(128):
        sample_id = "sample-{:03d}".format(index)
        for seed in contract.GENERATION_SEEDS:
            rows.append(
                {
                    "sample_id": sample_id,
                    "generation_seed": seed,
                    "condition_id": "ptc50.lr3e-6.step1000.seed2027",
                    "audio_sha256": hashlib.sha256(
                        "{}:{}".format(sample_id, seed).encode()
                    ).hexdigest(),
                }
            )
    return directory, rows


def _fake_fadtk_identity():
    return {
        "distribution_version": "1.1.0",
        "upstream_tag": contract.FADTK_UPSTREAM_TAG,
        "upstream_commit": contract.FADTK_UPSTREAM_COMMIT,
        "source_sha256": "f" * 64,
        "python_file_count": 1,
        "python_files": [
            {"relative_path": "__init__.py", "sha256": "e" * 64, "size_bytes": 1}
        ],
    }


def _fake_pins_report():
    return {
        "fadtk": _fake_fadtk_identity(),
        "mert": {
            "scientific_files": dict(contract.MERT_SCIENTIFIC_FILE_PINS),
            "snapshot_observation": {"tree_sha256": "d" * 64},
        },
    }


def _make_evaluation(root, generation_dir, generation_rows, reference_tree_hash):
    directory = root / "evaluation"
    provenance = {
        "schema_version": contract.PROVENANCE_SCHEMA,
        "status": "accepted_offline_evaluation",
        "generation_artifact_seal_sha256": "a" * 64,
        "reference_artifact_seal_sha256": "b" * 64,
        "model_pins_artifact_seal_sha256": "c" * 64,
        "offline_environment": dict(contract.OFFLINE_ENVIRONMENT),
        "network_access_forbidden": True,
        "evaluator": {
            "fadtk": _fake_fadtk_identity(),
            "mert": {
                "model_id": contract.MERT_MODEL_ID,
                "revision": contract.MERT_REVISION,
                "layer": 12,
            },
            "clap_laion_music": {
                "backend": contract.CLAP_BACKEND,
                "checkpoint_sha256": contract.CLAP_CHECKPOINT_SHA256,
                "checkpoint_loaded_from_staging_copy": True,
            },
            "runtime_device": "cuda:0",
        },
        "generation_integrity": {
            "before_tree_sha256": sha256_tree(generation_dir),
            "after_tree_sha256": sha256_tree(generation_dir),
            "unchanged": True,
        },
        "reference_integrity": {
            "before_tree_sha256": reference_tree_hash,
            "after_tree_sha256": reference_tree_hash,
            "unchanged": True,
        },
        "resource_integrity": {
            "mert_snapshot_before_tree_sha256": "d" * 64,
            "mert_snapshot_after_tree_sha256": "d" * 64,
            "clap_checkpoint_before_sha256": contract.CLAP_CHECKPOINT_SHA256,
            "clap_checkpoint_after_sha256": contract.CLAP_CHECKPOINT_SHA256,
            "unchanged": True,
        },
        "staging_policy": {
            "generation_audio_copied_to_independent_staging": True,
            "reference_audio_copied_to_independent_staging": True,
            "clap_checkpoint_copied_to_independent_staging": True,
            "fadtk_cache_roots_inside_staging_only": True,
            "generation_artifact_mutation_forbidden": True,
        },
    }
    provenance_bytes = canonical_json_bytes(provenance)
    provenance_hash = hashlib.sha256(provenance_bytes).hexdigest()
    source_by_key = {
        (row["sample_id"], row["generation_seed"]): row for row in generation_rows
    }
    rows = []
    distances = []
    for index in range(128):
        sample_id = "sample-{:03d}".format(index)
        similarity = 0.9 - index / 10000.0
        distance = 1.0 - similarity
        distances.append(distance)
        rows.append(
            {
                "schema_version": contract.DIVERSITY_ROW_SCHEMA,
                "sample_id": sample_id,
                "condition_id": "ptc50.lr3e-6.step1000.seed2027",
                "generation_seeds": list(contract.GENERATION_SEEDS),
                "audio_sha256_by_seed": {
                    str(seed): source_by_key[(sample_id, seed)]["audio_sha256"]
                    for seed in contract.GENERATION_SEEDS
                },
                "embedding_sha256_by_seed": {
                    str(seed): hashlib.sha256(
                        "emb:{}:{}".format(sample_id, seed).encode()
                    ).hexdigest()
                    for seed in contract.GENERATION_SEEDS
                },
                "embedding_frame_count_by_seed": {
                    str(seed): 499 for seed in contract.GENERATION_SEEDS
                },
                "model_id": contract.MERT_MODEL_ID,
                "layer": 12,
                "pooling": "arithmetic_mean_over_frame_axis",
                "normalization": "L2_after_frame_mean",
                "cosine_similarity": similarity,
                "cosine_distance": distance,
                "evaluator_provenance_sha256": provenance_hash,
            }
        )
    fad = {
        backend: {
            "score": score,
            "finite": True,
            "pipeline_check_passed": True,
            "selection_use_forbidden": True,
            "paper_claim_use_forbidden": True,
        }
        for backend, score in (
            (contract.CLAP_BACKEND, 2.5),
            ("MERT-v1-95M-layer12", 3.5),
        )
    }
    summary = {
        "schema_version": contract.EVALUATION_SCHEMA,
        "status": "complete_diversity_fad_pipeline_check",
        "condition_id": "ptc50.lr3e-6.step1000.seed2027",
        "generation_artifact_seal_sha256": "a" * 64,
        "reference_artifact_seal_sha256": "b" * 64,
        "model_pins_artifact_seal_sha256": "c" * 64,
        "evaluator_provenance_sha256": provenance_hash,
        "mert_diversity": {
            "model_id": contract.MERT_MODEL_ID,
            "layer": 12,
            "prompt_pair_count": 128,
            "pairing": "same_prompt_seed31001_vs_seed31002",
            "pooling": "frame_mean_then_L2",
            "mean_cosine_distance": math.fsum(distances) / len(distances),
            "min_cosine_distance": min(distances),
            "max_cosine_distance": max(distances),
            "selection_role": "pilot_point_estimate_gate",
        },
        "fad_pipeline_checks": fad,
        "selection_contract": {
            "mert_diversity_used_only_as_predeclared_pilot_point_estimate": True,
            "fad_used_for_model_or_checkpoint_selection": False,
            "fad_used_for_paper_claim": False,
        },
    }
    publish_closed_files_artifact(
        directory,
        payloads={
            contract.EVALUATOR_PROVENANCE: provenance_bytes,
            contract.DIVERSITY_ROWS: b"".join(canonical_json_bytes(row) for row in rows),
            contract.EVALUATION_SUMMARY: canonical_json_bytes(summary),
        },
        seal_schema=contract.EVALUATION_SEAL_SCHEMA,
        seal_status="complete_diversity_fad_pipeline_check",
    )
    return directory


class Stage1DiversityFadContractTests(unittest.TestCase):
    def test_reference_selection_is_domain_hashed_lowest_256(self):
        rows = _a1_rows()
        observed = contract.select_reference_rows(list(reversed(rows)))
        expected = sorted(
            rows,
            key=lambda row: (
                hashlib.sha256(
                    "{}|{}|{}".format(
                        contract.REFERENCE_SELECTION_DOMAIN,
                        contract.REFERENCE_SELECTION_SEED,
                        row["fma_track_id"],
                    ).encode()
                ).hexdigest(),
                row["fma_track_id"],
            ),
        )[:256]
        self.assertEqual(
            [row["fma_track_id"] for row in observed],
            [row["fma_track_id"] for row in expected],
        )

    def test_reference_artifact_is_closed_and_rejects_audio_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, report, source_rows = _make_a1(root)
            reference = _make_reference(root, manifest, report, source_rows)
            verified = contract.verify_reference_artifact(
                reference, a1_manifest=manifest, a1_report=report
            )
            self.assertEqual(len(verified["rows"]), 256)
            first = reference / verified["rows"][0]["path"]
            first.write_bytes(b"tampered")
            with self.assertRaises(Stage1ArtifactError):
                contract.verify_reference_artifact(
                    reference, a1_manifest=manifest, a1_report=report
                )

    def test_model_pin_builder_binds_snapshot_clap_and_fadtk_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot, mert_pins = _make_fake_mert_snapshot(root)
            checkpoint = root / contract.CLAP_CHECKPOINT_BASENAME
            checkpoint.write_bytes(b"checkpoint")
            package = root / "fadtk"
            package.mkdir()
            (package / "__init__.py").write_text("# source\n")
            observed_hash = sha256_file(checkpoint)
            source_identity = contract.python_source_identity(package)
            with patch.object(
                contract, "CLAP_CHECKPOINT_SHA256", observed_hash
            ), patch.object(
                contract, "MERT_SCIENTIFIC_FILE_PINS", mert_pins
            ), patch.object(
                contract, "FADTK_SOURCE_SHA256", source_identity["source_sha256"]
            ), patch.object(
                contract, "FADTK_PYTHON_FILE_COUNT", 1
            ):
                report_payload = contract.build_model_pins_report(
                    mert_snapshot=snapshot,
                    clap_checkpoint=checkpoint,
                    fadtk_package_root=package,
                    fadtk_version="1.1.0",
                )
                artifact = root / "pins"
                publish_closed_json_artifact(
                    artifact,
                    report_name=contract.MODEL_PINS_REPORT,
                    report=report_payload,
                    seal_schema=contract.MODEL_PINS_SEAL_SCHEMA,
                    seal_status="complete_local_model_pins",
                )
                # The complete cache tree is observational.  A later README
                # must not invalidate the five byte-frozen scientific files.
                (snapshot / "README.md").write_text("non-scientific metadata\n")
                verified = contract.verify_model_pins_artifact(
                    artifact,
                    mert_snapshot=snapshot,
                    clap_checkpoint=checkpoint,
                    fadtk_package_root=package,
                    fadtk_version="1.1.0",
                )
            self.assertEqual(
                verified["report"]["mert"]["scientific_files"], mert_pins
            )
            self.assertEqual(
                verified["report"]["mert"]["snapshot_observation"]["file_count"],
                5,
            )

    def test_mert_pin_rejects_altered_missing_and_alternative_weight_files(self):
        for failure in ("altered", "missing", "alternative"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                snapshot, mert_pins = _make_fake_mert_snapshot(Path(temporary))
                with patch.object(
                    contract, "MERT_SCIENTIFIC_FILE_PINS", mert_pins
                ):
                    self.assertEqual(
                        contract.verify_mert_scientific_snapshot(snapshot),
                        mert_pins,
                    )
                    if failure == "altered":
                        (snapshot / "config.json").write_bytes(b"altered")
                        pattern = "identity differs"
                    elif failure == "missing":
                        (snapshot / "configuration_MERT.py").unlink()
                        pattern = "missing required scientific file"
                    else:
                        (snapshot / "model.safetensors").write_bytes(b"alternate")
                        pattern = "forbidden alternative weight/index"
                    with self.assertRaisesRegex(Stage1ArtifactError, pattern):
                        contract.verify_mert_scientific_snapshot(snapshot)

    def test_model_pin_builder_rejects_nonofficial_fadtk_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot, mert_pins = _make_fake_mert_snapshot(root)
            checkpoint = root / contract.CLAP_CHECKPOINT_BASENAME
            checkpoint.write_bytes(b"checkpoint")
            package = root / "fadtk"
            package.mkdir()
            (package / "__init__.py").write_text("# altered source\n")
            with patch.object(
                contract, "CLAP_CHECKPOINT_SHA256", sha256_file(checkpoint)
            ), patch.object(
                contract, "MERT_SCIENTIFIC_FILE_PINS", mert_pins
            ), self.assertRaisesRegex(Stage1ArtifactError, "official tag"):
                contract.build_model_pins_report(
                    mert_snapshot=snapshot,
                    clap_checkpoint=checkpoint,
                    fadtk_package_root=package,
                    fadtk_version="1.1.0",
                )

    def test_artifact_root_symlinks_are_rejected_before_resolve(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, report, source_rows = _make_a1(root)
            reference = _make_reference(root, manifest, report, source_rows)
            alias = root / "reference-alias"
            alias.symlink_to(reference, target_is_directory=True)
            with self.assertRaisesRegex(Stage1ArtifactError, "symlink"):
                contract.verify_reference_artifact(
                    alias, a1_manifest=manifest, a1_report=report
                )

    def test_reference_builder_rejects_symlinked_output_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actual_parent = root / "actual-output-parent"
            actual_parent.mkdir()
            alias_parent = root / "output-parent-alias"
            alias_parent.symlink_to(actual_parent, target_is_directory=True)
            protected = root / "protected"
            protected.mkdir()
            with self.assertRaisesRegex(Stage1ArtifactError, "symlink"):
                reference_builder._canonical_new_output(
                    alias_parent / "reference", (protected,)
                )

    def test_evaluation_verifier_returns_consumer_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generation_dir, generation_rows = _make_generation_stub(root)
            reference_stub = root / "reference-stub"
            reference_stub.mkdir()
            (reference_stub / "anchor").write_bytes(b"reference")
            artifact = _make_evaluation(
                root, generation_dir, generation_rows, sha256_tree(reference_stub)
            )
            generation = {
                "directory": str(generation_dir.resolve()),
                "artifact_seal_sha256": "a" * 64,
                "samples": generation_rows,
            }
            with patch.object(
                contract, "verify_generation_artifact", return_value=generation
            ), patch.object(
                contract,
                "verify_reference_artifact",
                return_value={"artifact_seal_sha256": "b" * 64},
            ), patch.object(
                contract,
                "verify_model_pins_artifact",
                return_value={
                    "artifact_seal_sha256": "c" * 64,
                    "report": _fake_pins_report(),
                },
            ):
                verified = contract.verify_diversity_fad_artifact(
                    artifact,
                    generation_dir=generation_dir,
                    eval_manifest_dir=root,
                    reference_dir=reference_stub,
                    a1_manifest=root,
                    a1_report=root,
                    model_pins_dir=root,
                )
            self.assertTrue(verified["fad_pipeline_check_passed"])
            self.assertEqual(set(verified["fad_scores"]), {
                contract.CLAP_BACKEND, "MERT-v1-95M-layer12"
            })
            self.assertAlmostEqual(
                verified["mert_diversity_mean_cosine_distance"], 0.10635
            )

    def test_fad_can_never_be_authorized_for_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generation_dir, generation_rows = _make_generation_stub(root)
            reference_stub = root / "reference-stub"
            reference_stub.mkdir()
            (reference_stub / "anchor").write_bytes(b"reference")
            artifact = _make_evaluation(
                root, generation_dir, generation_rows, sha256_tree(reference_stub)
            )
            summary_path = artifact / contract.EVALUATION_SUMMARY
            summary = json.loads(summary_path.read_text())
            summary["fad_pipeline_checks"][contract.CLAP_BACKEND][
                "selection_use_forbidden"
            ] = False
            _write_json(summary_path, summary)
            # Even resealing cannot turn FAD into a selection metric.
            seal = json.loads((artifact / contract.SEAL_NAME).read_text())
            seal["members"][contract.EVALUATION_SUMMARY] = artifact_member(summary_path)
            _write_json(artifact / contract.SEAL_NAME, seal)
            generation = {
                "directory": str(generation_dir.resolve()),
                "artifact_seal_sha256": "a" * 64,
                "samples": generation_rows,
            }
            with patch.object(
                contract, "verify_generation_artifact", return_value=generation
            ), patch.object(
                contract,
                "verify_reference_artifact",
                return_value={"artifact_seal_sha256": "b" * 64},
            ), patch.object(
                contract,
                "verify_model_pins_artifact",
                return_value={
                    "artifact_seal_sha256": "c" * 64,
                    "report": _fake_pins_report(),
                },
            ), self.assertRaisesRegex(Stage1ArtifactError, "non-selection"):
                contract.verify_diversity_fad_artifact(
                    artifact,
                    generation_dir=generation_dir,
                    eval_manifest_dir=root,
                    reference_dir=reference_stub,
                    a1_manifest=root,
                    a1_report=root,
                    model_pins_dir=root,
                )

    def test_runner_declares_staging_only_fadtk_caches(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "eval_stage1_diversity_fad.py"
        ).read_text()
        self.assertIn("FADtk's ``convert/``, ``embeddings/``, and ``stats/``", source)
        self.assertIn('compute / "generated_audio"', source)
        self.assertIn('compute / "reference_audio"', source)
        self.assertNotIn("cache_embedding_file(Path(args.generation_dir", source)
        self.assertIn("torch.cuda.set_device(device)", source)
        self.assertIn('fadtk_device = torch.device("cuda")', source)
        self.assertEqual(
            source.count("verify_mert_scientific_snapshot(mert_root)"), 2
        )

    def test_finite_fad_accepts_float_convertible_scalar_and_rejects_bad_values(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "eval_stage1_diversity_fad.py"
        )
        spec = importlib.util.spec_from_file_location("stage1_eval_fad_test", script)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class Scalar:
            def __float__(self):
                return 1.25

        self.assertEqual(module._finite_fad(Scalar(), "mock"), 1.25)
        for value in (True, "1.0", float("nan"), float("inf")):
            with self.assertRaises(Stage1ArtifactError):
                module._finite_fad(value, "mock")


if __name__ == "__main__":
    unittest.main()
