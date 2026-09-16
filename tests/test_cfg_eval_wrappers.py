"""Model-free tests for the two cross-environment CFG evaluator wrappers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cfg_eval_common as common
import eval_cfg_music_clap as clap_wrapper
import eval_cfg_quality as quality_wrapper


def _identity() -> dict:
    return {
        "generation_dir": "/fixture/generation",
        "generation_run_sha256": "1" * 64,
        "samples_jsonl_sha256": "2" * 64,
        "artifact_seal_sha256": "3" * 64,
        "scientific_config_sha256": "4" * 64,
        "generation_identity": {"model_id": "facebook/musicgen-small"},
        "prompt_count": 2,
        "sample_records": 4,
    }


def _records() -> list[dict]:
    output = []
    for sample_index in range(2):
        sample_id = "sample-{}".format(sample_index)
        prompt = "fixture prompt {}".format(sample_index)
        for condition, scale, anchor in (
            ("no_cfg", None, "no_cfg"),
            ("cfg", 2.0, "cfg_2.0"),
        ):
            output.append(
                {
                    "sample_id": sample_id,
                    "prompt": prompt,
                    "prompt_sha256": common.gate.prompt_sha256(prompt),
                    "condition": condition,
                    "cfg_scale": scale,
                    "condition_id": anchor,
                    "path": "audio/{}/{}.wav".format(anchor, sample_id),
                    "audio_sha256": ("a" if anchor == "no_cfg" else "b") * 64,
                    "scientific_config_sha256": "4" * 64,
                }
            )
    return output


def _evaluator_identity(character: str) -> dict:
    digits = "0123456789abcdef"
    index = digits.index(character)
    return {
        "checkpoint_sha256": character * 64,
        "source_sha256": digits[(index + 1) % len(digits)] * 64,
        "config_sha256": digits[(index + 2) % len(digits)] * 64,
        "details": {"fixture": True},
    }


class FakeMuQ:
    identity = _evaluator_identity("5")

    def __init__(self, _args: argparse.Namespace):
        pass

    def score_batch(self, paths: list[Path]) -> list[float]:
        return [1.0 + index / 10.0 for index, _ in enumerate(paths)]


class FakeAudiobox:
    identity = _evaluator_identity("8")

    def __init__(self, _args: argparse.Namespace):
        pass

    def score_batch(self, paths: list[Path]) -> list[dict]:
        return [
            {"audiobox_ce": 2.0 + index, "audiobox_pq": 3.0 + index}
            for index, _ in enumerate(paths)
        ]


class FakeClap:
    identity = {
        "checkpoint_sha256": "c" * 64,
        "source_sha256": "d" * 64,
        "config_sha256": "e" * 64,
        "details": {"fad_computed": False},
    }

    def __init__(self, _args: argparse.Namespace):
        pass

    def score_batch(self, paths: list[Path], prompts: list[str]) -> list[float]:
        if len(paths) != len(prompts):
            raise AssertionError("fixture pairing failed")
        return [0.25 + index / 100.0 for index, _ in enumerate(paths)]


def _quality_args(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        generation_dir=root / "generation",
        output_dir=root / "quality",
        batch_size=2,
        device="cpu",
        muq_eval_root=root / "unused-muq-source",
        muq_config=root / "unused-config",
        muq_state_dict=root / "unused-state",
        muq_backbone=root / "unused-backbone",
        audiobox_checkpoint=root / "unused-audiobox",
    )


def _clap_args(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        generation_dir=root / "generation",
        quality_dir=root / "quality",
        output_dir=root / "external",
        clap_checkpoint=root / "unused-clap",
        batch_size=2,
        device="cpu",
    )


def _fake_gate_match(_generation: Path, scores: Path, _provenance: Path):
    rows = list(common.iter_jsonl(scores))
    return [
        {
            "sample_id": row["sample_id"],
            "condition_id": row["condition_id"],
            "metrics": row["metrics"],
        }
        for row in rows
    ], {
        "scores_jsonl_sha256": common.sha256_file(scores),
        "evaluator_provenance_sha256": common.sha256_file(_provenance),
        "evaluator_provenance": common.read_json(_provenance),
    }


class CFGEvalWrapperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "generation").mkdir()
        for record in _records():
            path = self.root / "generation" / record["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        self.generation_patch = mock.patch.object(
            common, "load_verified_generation", return_value=(_identity(), _records())
        )
        self.generation_patch.start()

    def tearDown(self) -> None:
        self.generation_patch.stop()
        self.temporary.cleanup()

    def _publish_quality(self) -> Path:
        args = _quality_args(self.root)
        quality_wrapper.run_quality(
            args,
            muq_backend_factory=FakeMuQ,
            audiobox_backend_factory=FakeAudiobox,
        )
        return Path(args.output_dir)

    def test_quality_artifact_binds_every_generation_field_and_hash(self) -> None:
        quality_dir = self._publish_quality()
        provenance, rows, identity = common.verify_quality_artifact(
            quality_dir, self.root / "generation"
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual(identity["scientific_config_sha256"], "4" * 64)
        self.assertEqual(provenance["status"], "accepted_quality_evaluation")
        self.assertEqual(provenance["evaluators"]["muq_eval"]["checkpoint_sha256"], "5" * 64)
        expected_by_key = {
            (record["sample_id"], record["condition_id"]): record for record in _records()
        }
        for row in rows:
            generated = expected_by_key[(row["sample_id"], row["condition_id"])]
            self.assertEqual(row["prompt_sha256"], generated["prompt_sha256"])
            self.assertEqual(row["audio_sha256"], generated["audio_sha256"])
            self.assertEqual(
                row["scientific_config_sha256"], generated["scientific_config_sha256"]
            )

    def test_runbook_uses_released_a1_config_beside_state_dict(self) -> None:
        text = (ROOT / "docs" / "cfg_scale_gate_runbook.md").read_text(encoding="utf-8")
        self.assertIn(
            'export PTC_MUQ_A1_CONFIG="${PTC_MUQ_A1_DIR}/config.yaml"', text
        )
        self.assertIn('--muq-config "${PTC_MUQ_A1_CONFIG}"', text)
        self.assertNotIn(
            'PTC_MUQ_A1_CONFIG="${PTC_MUQ_EVAL_ROOT}/configs/A1_frozen_mlp.yaml"',
            text,
        )
        self.assertIn(
            'PTC_ENV_ACCEPTANCE_REPORT="${PTC_WORKPACK_ROOT}/docs/'
            'environment_acceptance_20260812.md"',
            text,
        )
        self.assertIn('--environment-report "${PTC_ENV_ACCEPTANCE_REPORT}"', text)

    def test_environment_report_is_exact_accepted_copy(self) -> None:
        report = ROOT / "docs" / quality_wrapper.ENVIRONMENT_REPORT_BASENAME
        self.assertEqual(
            common.sha256_file(report),
            "7bd3d913bd210cf1b94165986e8368d8560702871dafac1d1e45beff1fb4d73d",
        )
        parser = quality_wrapper.build_parser()
        args = parser.parse_args(
            [
                "run",
                "--generation-dir", str(self.root / "generation"),
                "--muq-eval-root", str(self.root / "muq-eval"),
                "--muq-config", str(self.root / "config.yaml"),
                "--muq-state-dict", str(self.root / "model_state_dict.pt"),
                "--muq-backbone", str(self.root / "muq-backbone"),
                "--audiobox-checkpoint", str(self.root / "checkpoint.pt"),
                "--output-dir", str(self.root / "quality-real"),
            ]
        )
        self.assertEqual(Path(args.environment_report), report)

    def test_muq_source_identity_binds_executed_python_package(self) -> None:
        package_root = self.root / "site-packages" / "muq"
        package_root.mkdir(parents=True)
        package_file = package_root / "__init__.py"
        package_file.write_text("class MuQ: pass\n", encoding="utf-8")
        module = types.SimpleNamespace(__file__=str(package_file))
        repository = {
            "git_commit": common.PINNED_MUQ_EVAL_COMMIT,
            "source_sha256": "a" * 64,
        }
        with mock.patch.object(
            quality_wrapper, "_distribution_version", return_value="0.1.0"
        ):
            first_hash, first_details = quality_wrapper._muq_source_identity(
                muq_eval_source=repository,
                muq_module=module,
            )
            package_file.write_text("class MuQ: changed = True\n", encoding="utf-8")
            second_hash, _ = quality_wrapper._muq_source_identity(
                muq_eval_source=repository,
                muq_module=module,
            )
        self.assertNotEqual(first_hash, second_hash)
        components = first_details["components"]
        self.assertEqual(
            components["muq_eval_repository"]["git_commit"],
            common.PINNED_MUQ_EVAL_COMMIT,
        )
        self.assertEqual(
            components["muq_python_package"]["distribution_version"], "0.1.0"
        )
        self.assertEqual(first_details["muq_python_package"]["python_file_count"], 1)

    def test_quality_runtime_identity_gates_and_records_accepted_pins(self) -> None:
        report = self.root / quality_wrapper.ENVIRONMENT_REPORT_BASENAME
        report.write_text("accepted fixture\n", encoding="utf-8")
        fake_modules = {
            name: types.SimpleNamespace(__version__=version)
            for name, version in quality_wrapper.EXPECTED_QUALITY_RUNTIME_VERSIONS.items()
            if name != "muq"
        }

        def version(name: str) -> str:
            return quality_wrapper.EXPECTED_QUALITY_RUNTIME_VERSIONS[name]

        with mock.patch.object(
            quality_wrapper, "_distribution_version", side_effect=version
        ):
            identity = quality_wrapper._quality_runtime_identity(
                report, imported_modules=fake_modules
            )
        self.assertEqual(
            identity["accepted_environment_report"]["sha256"],
            common.sha256_file(report),
        )
        self.assertEqual(identity["distributions"]["muq"], "0.1.0")
        self.assertEqual(identity["imported_modules"]["torch"], "2.2.2+cu121")

        def drifted_version(name: str) -> str:
            if name == "numpy":
                return "2.0.0"
            return quality_wrapper.EXPECTED_QUALITY_RUNTIME_VERSIONS[name]

        with mock.patch.object(
            quality_wrapper, "_distribution_version", side_effect=drifted_version
        ), self.assertRaisesRegex(RuntimeError, "runtime version mismatch"):
            quality_wrapper._quality_runtime_identity(
                report, imported_modules=fake_modules
            )

    def test_released_a1_config_contract_uses_local_backbone(self) -> None:
        try:
            from omegaconf import OmegaConf  # noqa: F401
        except ImportError:
            self.skipTest("OmegaConf is absent from this lightweight test interpreter")
        config_dir = self.root / "released-a1"
        config_dir.mkdir()
        (config_dir / "base.yaml").write_text(
            """data:\n  sample_rate: 24000\n  clip_samples: 240000\nmodel:\n  encoder: muq\n  encoder_id: OpenMuQ/MuQ-large-msd-iter\n  heads:\n    - name: MI\n  tuning_mode: frozen\nloss:\n  type: mse\n""",
            encoding="utf-8",
        )
        (config_dir / "config.yaml").write_text(
            """defaults:\n  - base\nexperiment:\n  name: A1_frozen_mlp\n""",
            encoding="utf-8",
        )
        backbone = self.root / "local-muq-backbone"
        backbone.mkdir()
        (backbone / "weights.bin").write_bytes(b"local-only")
        cfg, identity = quality_wrapper._load_muq_configuration(
            config_dir / "config.yaml", backbone
        )
        self.assertEqual(str(cfg.model.encoder_id), str(backbone))
        self.assertEqual(identity["declared_encoder_id"], "OpenMuQ/MuQ-large-msd-iter")
        self.assertEqual(
            identity["runtime_config"]["local_encoder_snapshot_sha256"],
            common.sha256_local_tree(backbone),
        )

    def test_quality_verifier_rejects_duplicate_and_missing_rows(self) -> None:
        quality_dir = self._publish_quality()
        scores = quality_dir / "quality_scores.jsonl"
        rows = list(common.iter_jsonl(scores))

        scores.unlink()
        common.write_jsonl(scores, rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, "duplicate quality"):
            common.verify_quality_artifact(quality_dir, self.root / "generation")

        scores.unlink()
        common.write_jsonl(scores, rows[:-1])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            common.verify_quality_artifact(quality_dir, self.root / "generation")

    def test_quality_verifier_rejects_hash_drift_and_nan(self) -> None:
        quality_dir = self._publish_quality()
        scores = quality_dir / "quality_scores.jsonl"
        rows = list(common.iter_jsonl(scores))
        rows[0]["audio_sha256"] = "f" * 64
        scores.unlink()
        common.write_jsonl(scores, rows)
        with self.assertRaisesRegex(ValueError, "audio_sha256 mismatch"):
            common.verify_quality_artifact(quality_dir, self.root / "generation")

        rows[0]["audio_sha256"] = _records()[0]["audio_sha256"]
        rows[0]["metrics"]["muq_mi"] = float("nan")
        scores.unlink()
        with scores.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        with self.assertRaisesRegex(ValueError, "finite"):
            common.verify_quality_artifact(quality_dir, self.root / "generation")

    def test_quality_failure_is_atomic(self) -> None:
        class BrokenMuQ(FakeMuQ):
            def score_batch(self, _paths: list[Path]) -> list[float]:
                raise RuntimeError("injected evaluator failure")

        args = _quality_args(self.root)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            quality_wrapper.run_quality(
                args,
                muq_backend_factory=BrokenMuQ,
                audiobox_backend_factory=FakeAudiobox,
            )
        self.assertFalse(Path(args.output_dir).exists())
        self.assertEqual(list(self.root.glob(".quality.partial.*")), [])

    def test_music_clap_finishes_strict_join_and_gate_files(self) -> None:
        self._publish_quality()
        args = _clap_args(self.root)
        with mock.patch.object(common.gate, "load_and_match_scores", side_effect=_fake_gate_match):
            clap_wrapper.run_music_clap(args, clap_backend_factory=FakeClap)
            result = common.verify_external_artifact(
                Path(args.output_dir), Path(args.generation_dir), Path(args.quality_dir)
            )
        self.assertEqual(result["score_records"], 4)
        provenance = common.read_json(Path(args.output_dir) / "evaluator_provenance.json")
        accepted, accepted_hash = common.gate.validate_evaluator_provenance(
            Path(args.output_dir) / "evaluator_provenance.json"
        )
        self.assertEqual(accepted["status"], "accepted_external_evaluation")
        self.assertEqual(
            accepted_hash,
            common.sha256_file(Path(args.output_dir) / "evaluator_provenance.json"),
        )
        self.assertFalse(provenance["protocol"]["fad_computed"])
        self.assertEqual(provenance["evaluators"]["music_clap"]["checkpoint_sha256"], "c" * 64)
        rows = list(common.iter_jsonl(Path(args.output_dir) / "scores.jsonl"))
        self.assertEqual(set(rows[0]["metrics"]), set(common.gate.REQUIRED_METRICS))
        self.assertEqual(
            rows[0]["evaluator_provenance_sha256"],
            common.sha256_file(Path(args.output_dir) / "evaluator_provenance.json"),
        )

    def test_music_clap_torch_load_override_is_scoped_and_default_only(self) -> None:
        calls = []

        def original_load(*args, **kwargs):
            calls.append((args, dict(kwargs)))
            return object()

        fake_torch = types.SimpleNamespace(load=original_load)

        class Model:
            def load_ckpt(self, checkpoint):
                self.checkpoint = checkpoint
                fake_torch.load("implicit.pt")
                fake_torch.load("explicit.pt", weights_only=True)

        model = Model()
        clap_wrapper._load_ckpt_with_scoped_full_pickle(
            model, fake_torch, Path("trusted.pt")
        )
        self.assertEqual(model.checkpoint, "trusted.pt")
        self.assertIs(fake_torch.load, original_load)
        self.assertEqual(calls[0][1], {"weights_only": False})
        self.assertEqual(calls[1][1], {"weights_only": True})

    def test_music_clap_torch_load_is_restored_after_load_exception(self) -> None:
        def original_load(*_args, **_kwargs):
            return object()

        fake_torch = types.SimpleNamespace(load=original_load)

        class BrokenModel:
            def load_ckpt(self, _checkpoint):
                fake_torch.load("implicit.pt")
                raise RuntimeError("injected load failure")

        with self.assertRaisesRegex(RuntimeError, "injected load failure"):
            clap_wrapper._load_ckpt_with_scoped_full_pickle(
                BrokenModel(), fake_torch, Path("trusted.pt")
            )
        self.assertIs(fake_torch.load, original_load)

    def test_music_clap_checkpoint_hash_gate_precedes_model_construction(self) -> None:
        checkpoint = self.root / clap_wrapper.MUSIC_CLAP_CHECKPOINT_BASENAME
        checkpoint.write_bytes(b"untrusted checkpoint")
        constructed = []
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        fake_clap = types.ModuleType("laion_clap")

        def clap_module(**kwargs):
            constructed.append(kwargs)
            raise AssertionError("model must not be constructed before hash gate")

        fake_clap.CLAP_Module = clap_module
        args = argparse.Namespace(clap_checkpoint=checkpoint, device="cpu")
        with mock.patch.dict(
            sys.modules,
            {"torch": fake_torch, "laion_clap": fake_clap},
        ), mock.patch.object(common, "sha256_file", return_value="0" * 64):
            with self.assertRaisesRegex(RuntimeError, "checkpoint SHA-256 mismatch"):
                clap_wrapper.MusicClapBackend(args)
        self.assertEqual(constructed, [])

    def test_music_clap_nan_and_failure_leave_no_artifact(self) -> None:
        self._publish_quality()

        class NaNClap(FakeClap):
            def score_batch(self, paths: list[Path], prompts: list[str]) -> list[float]:
                return [float("nan") for _ in paths]

        args = _clap_args(self.root)
        with self.assertRaisesRegex(ValueError, "finite"):
            clap_wrapper.run_music_clap(args, clap_backend_factory=NaNClap)
        self.assertFalse(Path(args.output_dir).exists())
        self.assertEqual(list(self.root.glob(".external.partial.*")), [])

    @unittest.skipIf(torch is None, "torch is unavailable")
    def test_music_clap_backend_paired_embedding_shape_and_normalized_dot(self) -> None:
        backend = object.__new__(clap_wrapper.MusicClapBackend)

        class Model:
            def get_audio_embedding_from_filelist(self, x, use_tensor):
                self.audio_call = (x, use_tensor)
                return torch.tensor([[3.0, 4.0], [1.0, 0.0]])

            def get_text_embedding(self, x, use_tensor):
                self.text_call = (x, use_tensor)
                return torch.tensor([[0.0, 2.0], [1.0, 1.0]])

        backend.model = Model()
        backend._torch = torch
        values = backend.score_batch(
            [Path("first.wav"), Path("second.wav")], ["first prompt", "second prompt"]
        )
        self.assertAlmostEqual(values[0], 0.8, places=6)
        self.assertAlmostEqual(values[1], 2.0 ** -0.5, places=6)
        self.assertEqual(backend.model.audio_call, (["first.wav", "second.wav"], True))
        self.assertEqual(
            backend.model.text_call, (["first prompt", "second prompt"], True)
        )

        class BadTextModel(Model):
            def get_text_embedding(self, x, use_tensor):
                return torch.ones(2, 3)

        backend.model = BadTextModel()
        with self.assertRaisesRegex(RuntimeError, r"paired \[B,D\]"):
            backend.score_batch(
                [Path("first.wav"), Path("second.wav")], ["first prompt", "second prompt"]
            )


if __name__ == "__main__":
    unittest.main()
