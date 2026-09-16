"""CPU-only tests for the CFG-scale generation/decision gate.

These fixtures contain placeholder audio bytes with matching hashes.  They
exercise artifact-set and decision logic; they are not claims that MusicGen,
CUDA, soundfile, MuQ, Audiobox, or music-CLAP ran on this workstation.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

try:
    import torch
except ImportError:  # The project's real test environment includes pinned torch.
    torch = None


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "cfg_scale_gate", ROOT / "scripts" / "cfg_scale_gate.py"
)
assert spec is not None and spec.loader is not None
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)

import cfg_eval_common as common


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(gate.canonical_json_bytes(value, pretty=True))


def _fake_generation(root: Path) -> tuple[Path, list[dict]]:
    generation = root / "generation"
    generation.mkdir()
    config = {
        "schema_version": gate.GENERATION_SCHEMA_VERSION,
        "model_id": "facebook/musicgen-small",
        "prompt_count": gate.EXPECTED_DEV_PROMPTS,
        "samples_per_prompt": len(gate.CONDITIONS),
        "generation_seed_base": gate.GENERATION_BASE_SEED,
        "generation_seed_namespace": gate.GENERATION_SEED_NAMESPACE,
        "duration_seconds": gate.DURATION_SECONDS,
        "sample_rate": gate.SAMPLE_RATE,
        "codec_frame_rate": gate.CODEC_FRAME_RATE,
        "token_frames": gate.TOKEN_FRAMES,
        "sampling": {
            "use_sampling": True,
            "temperature": gate.TEMPERATURE,
            "top_k": gate.TOP_K,
            "top_p": gate.TOP_P,
            "two_step_cfg": False,
        },
        "audio_serialization": {
            "container": "WAV",
            "subtype": "FLOAT",
            "loudness_normalization": False,
            "clipping_or_rescale": False,
        },
        "offline_resolution": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        },
        "conditions": [
            {"condition": condition, "cfg_scale": scale}
            for condition, scale in gate.CONDITIONS
        ],
        "checkpoint_sha256": "a" * 64,
        "manifest_sha256": "1" * 64,
        "state_dict_sha256": "2" * 64,
        "compression_state_dict_sha256": "3" * 64,
        "audiocraft_base_commit": gate.PINNED_AUDIOCRAFT_BASE_COMMIT,
        "audiocraft_source_sha256": "b" * 64,
        "audiocraft_lm_sha256": "5" * 64,
        "runtime_identity": {
            "model_id": "facebook/musicgen-small",
            "musicgen_architecture": {
                "num_codebooks": 4,
                "cardinality": 2048,
                "audio_channels": 1,
                "sample_rate": 32000,
                "frame_rate": 50.0,
                **gate.MODEL_ARCHITECTURES["facebook/musicgen-small"],
            },
            "loaded_t5_identity": {"identity_sha256": "6" * 64},
            "precision_contract": {
                "load_device": "cpu",
                "runtime_device_type": "cuda",
                "lm_parameter_dtype": gate.LM_PARAMETER_DTYPE,
                "compression_parameter_dtype": gate.COMPRESSION_PARAMETER_DTYPE,
                "conditioner_parameter_dtype": gate.CONDITIONER_PARAMETER_DTYPE,
                "conditioner_compute_dtype": gate.CONDITIONER_COMPUTE_DTYPE,
                "lm_generation_compute_dtype": gate.LM_GENERATION_COMPUTE_DTYPE,
                "compression_decode_compute_dtype": gate.COMPRESSION_DECODE_COMPUTE_DTYPE,
                "condition_tensors_precomputed_once_per_prompt": True,
                "conditional_and_null_share_provider_call": True,
            },
        },
        "precision_contract": {
            "load_device": "cpu",
            "lm_parameter_dtype": gate.LM_PARAMETER_DTYPE,
            "compression_parameter_dtype": gate.COMPRESSION_PARAMETER_DTYPE,
            "conditioner_parameter_dtype": gate.CONDITIONER_PARAMETER_DTYPE,
            "conditioner_compute_dtype": gate.CONDITIONER_COMPUTE_DTYPE,
            "lm_generation_compute_dtype": gate.LM_GENERATION_COMPUTE_DTYPE,
            "compression_decode_compute_dtype": gate.COMPRESSION_DECODE_COMPUTE_DTYPE,
        },
    }
    config_hash = gate.sha256_json(config)
    records = []
    for prompt_index in range(gate.EXPECTED_DEV_PROMPTS):
        sample_id = "sample-{:03d}".format(prompt_index)
        prompt = "prompt {}".format(prompt_index)
        for condition, scale in gate.CONDITIONS:
            anchor = gate.condition_id(condition, scale)
            relative = Path("audio") / anchor / (sample_id + ".wav")
            path = generation / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = (sample_id + anchor).encode("utf-8")
            path.write_bytes(payload)
            records.append(
                {
                    "schema_version": gate.SAMPLE_SCHEMA_VERSION,
                    "sample_id": sample_id,
                    "prompt": prompt,
                    "prompt_sha256": gate.prompt_sha256(prompt),
                    "condition": condition,
                    "cfg_scale": scale,
                    "condition_id": anchor,
                    "seed": gate.paired_prompt_seed(sample_id),
                    "path": relative.as_posix(),
                    "audio_sha256": gate.sha256_bytes(payload),
                    "audio_frames": 320000,
                    "audio_channels": 1,
                    "audio_sample_rate": 32000,
                    "audio_subtype": "FLOAT",
                    "audio_peak_abs": 0.1,
                    "checkpoint_sha256": "a" * 64,
                    "audiocraft_source_sha256": "b" * 64,
                    "scientific_config_sha256": config_hash,
                }
            )
    samples = generation / "samples.jsonl"
    with samples.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    run = generation / "generation_run.json"
    _write_json(
        run,
        {
            "schema_version": gate.GENERATION_SCHEMA_VERSION,
            "scientific_status": "gpu_generation_completed",
            "scientific_config": config,
            "scientific_config_sha256": config_hash,
            "sample_records": len(records),
            "raw_float_wav": True,
            "loudness_normalized": False,
        },
    )
    _write_json(
        generation / "artifact_seal.json",
        {
            "schema_version": gate.SEAL_SCHEMA_VERSION,
            "generation_run_sha256": gate.sha256_file(run),
            "samples_jsonl_sha256": gate.sha256_file(samples),
            "sample_records": len(records),
            "audio_files": len(records),
            "scientific_config_sha256": config_hash,
        },
    )
    return generation, records


def _fake_provenance(root: Path, generation: Path) -> tuple[Path, str]:
    path = root / "evaluator_provenance.json"
    evaluator = {
        "checkpoint_sha256": "c" * 64,
        "source_sha256": "d" * 64,
        "config_sha256": "e" * 64,
        "details": {"fixture": True},
    }
    generation_identity = gate.verify_generation_directory(
        generation, rehash_audio=True
    )
    _write_json(
        path,
        {
            "schema_version": gate.EVALUATOR_PROVENANCE_SCHEMA_VERSION,
            "status": "accepted_external_evaluation",
            "metrics": list(gate.REQUIRED_METRICS),
            "evaluators": {
                name: dict(evaluator) for name in gate.REQUIRED_EVALUATORS
            },
            "generation": gate.generation_binding(generation_identity),
            "quality_artifact": {
                "artifact_seal_sha256": "7" * 64,
                "quality_scores_sha256": "8" * 64,
                "quality_provenance_sha256": "9" * 64,
            },
            "protocol": dict(gate.EXTERNAL_EVALUATION_PROTOCOL),
            "offline_environment": dict(gate.OFFLINE_ENVIRONMENT),
        },
    )
    return path, gate.sha256_file(path)


def _score_rows(records: list[dict], provenance_hash: str) -> list[dict]:
    output = []
    scale_gains = {"no_cfg": 0.0, "cfg_2.0": 0.4, "cfg_3.0": 0.6, "cfg_5.0": 0.5}
    for record in records:
        index = int(str(record["sample_id"]).split("-")[-1])
        # Nonconstant base values ensure every base standard deviation is
        # positive.  Equal metric gains make Q_dev ordering transparent.
        base = index / 100.0
        gain = scale_gains[str(record["condition_id"])]
        output.append(
            {
                "schema_version": gate.SCORE_SCHEMA_VERSION,
                "sample_id": record["sample_id"],
                "prompt_sha256": record["prompt_sha256"],
                "condition": record["condition"],
                "cfg_scale": record["cfg_scale"],
                "condition_id": record["condition_id"],
                "audio_sha256": record["audio_sha256"],
                "scientific_config_sha256": record["scientific_config_sha256"],
                "evaluator_provenance_sha256": provenance_hash,
                "metrics": {
                    "muq_mi": base + gain,
                    "audiobox_ce": base + gain,
                    "audiobox_pq": base + gain,
                    "music_clap": base,
                },
            }
        )
    return output


def _write_scores(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def _fake_sealed_external_artifacts(
    root: Path, generation: Path, records: list[dict]
) -> tuple[Path, Path]:
    generation_identity = gate.verify_generation_directory(
        generation, rehash_audio=True
    )
    evaluator = {
        "checkpoint_sha256": "c" * 64,
        "source_sha256": "d" * 64,
        "config_sha256": "e" * 64,
        "details": {"fixture": True},
    }
    metric_templates = _score_rows(records, "0" * 64)

    quality = root / "quality"
    quality.mkdir()
    quality_provenance = {
        "schema_version": common.QUALITY_PROVENANCE_SCHEMA_VERSION,
        "status": "accepted_quality_evaluation",
        "metrics": list(common.QUALITY_METRICS),
        "generation": common.generation_binding(generation_identity),
        "evaluators": {
            name: dict(evaluator) for name in common.QUALITY_EVALUATORS
        },
        "offline_environment": dict(common.OFFLINE_ENVIRONMENT),
    }
    quality_provenance_path = quality / "quality_provenance.json"
    common.write_json(quality_provenance_path, quality_provenance)
    quality_provenance_hash = common.sha256_file(quality_provenance_path)
    quality_rows = [
        common.quality_row(
            generated,
            provenance_sha256=quality_provenance_hash,
            muq_mi=template["metrics"]["muq_mi"],
            audiobox_ce=template["metrics"]["audiobox_ce"],
            audiobox_pq=template["metrics"]["audiobox_pq"],
        )
        for generated, template in zip(records, metric_templates)
    ]
    quality_scores_path = quality / "quality_scores.jsonl"
    common.write_jsonl(quality_scores_path, quality_rows)
    common.write_json(
        quality / "artifact_seal.json",
        common.build_quality_seal(
            scores_path=quality_scores_path,
            provenance_path=quality_provenance_path,
            record_count=len(quality_rows),
            generation_identity=generation_identity,
        ),
    )

    external = root / "external"
    external.mkdir()
    external_provenance = {
        "schema_version": gate.EVALUATOR_PROVENANCE_SCHEMA_VERSION,
        "status": "accepted_external_evaluation",
        "metrics": list(gate.REQUIRED_METRICS),
        "evaluators": {
            name: dict(evaluator) for name in gate.REQUIRED_EVALUATORS
        },
        "generation": gate.generation_binding(generation_identity),
        "quality_artifact": {
            "artifact_seal_sha256": common.sha256_file(
                quality / "artifact_seal.json"
            ),
            "quality_scores_sha256": common.sha256_file(quality_scores_path),
            "quality_provenance_sha256": quality_provenance_hash,
        },
        "protocol": dict(gate.EXTERNAL_EVALUATION_PROTOCOL),
        "offline_environment": dict(gate.OFFLINE_ENVIRONMENT),
    }
    external_provenance_path = external / "evaluator_provenance.json"
    common.write_json(external_provenance_path, external_provenance)
    external_provenance_hash = common.sha256_file(external_provenance_path)
    external_scores_path = external / "scores.jsonl"
    common.write_jsonl(
        external_scores_path, _score_rows(records, external_provenance_hash)
    )
    common.write_json(
        external / "artifact_seal.json",
        common.build_external_seal(
            scores_path=external_scores_path,
            provenance_path=external_provenance_path,
            quality_dir=quality,
            record_count=len(records),
            generation_identity=generation_identity,
        ),
    )
    return quality, external


class CFGScaleGateTest(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is unavailable in this lightweight interpreter")
    def test_runtime_identity_gates_unregistered_t5_cardinality_and_pattern(self) -> None:
        class Config:
            def to_dict(self) -> dict:
                return {"model_type": "t5"}

        class Tokenizer:
            class Processor:
                @staticmethod
                def serialized_model_proto() -> bytes:
                    return b"sentencepiece"

            sp_model = Processor()
            special_tokens_map = {"eos_token": "</s>"}

            def get_vocab(self) -> dict:
                return {"x": 0, "y": 1}

        class Conditioner(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.name = "t5-base"
                self.finetune = False
                self.device = "wrong"
                self.t5_tokenizer = Tokenizer()
                external = torch.nn.Linear(2, 2)
                external.config = Config()
                self.__dict__["t5"] = external

        class Provider(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conditioner = Conditioner()

        class Pattern:
            def build_pattern_sequence(self, dummy, special_token, keep_only_valid_steps):
                mask = torch.zeros(4, 504, dtype=torch.bool)
                for q in range(4):
                    mask[q, q + 1 : q + 501] = True
                return dummy, None, mask

            def revert_pattern_logits(self, logits, special_token, keep_only_valid_steps):
                valid = torch.ones(4, 500, dtype=torch.bool)
                valid[1, -1:] = False
                valid[2, -2:] = False
                valid[3, -3:] = False
                return torch.zeros(1, 1, 4, 500), None, valid

        class DelayedPatternProvider:
            def __init__(self) -> None:
                self.n_q = 4
                self.delays = [0, 1, 2, 3]
                self.flatten_first = 0
                self.empty_initial = 0

            def get_pattern(self, timesteps):
                if timesteps != 500:
                    raise AssertionError(timesteps)
                return Pattern()

        class LM(torch.nn.Module):
            def __init__(self, *, dim: int, layers: int, heads: int) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.condition_provider = Provider()
                self.num_codebooks = 4
                self.card = 2048
                transformer = type(
                    "TransformerConfig",
                    (),
                    {"dim": dim, "num_layers": layers, "num_heads": heads},
                )()
                self.cfg = type("LMConfig", (), {"transformer_lm": transformer})()
                self.pattern_provider = DelayedPatternProvider()

        class FakeModel:
            def __init__(self, *, dim: int, layers: int, heads: int) -> None:
                self.lm = LM(dim=dim, layers=layers, heads=heads)
                self.compression_model = torch.nn.Linear(1, 1)
                self.audio_channels = 1
                self.sample_rate = 32000
                self.frame_rate = 50.0

        for model_id, architecture in gate.MODEL_ARCHITECTURES.items():
            model = FakeModel(
                dim=architecture["transformer_dim"],
                layers=architecture["transformer_layers"],
                heads=architecture["transformer_heads"],
            )
            identity = gate._prepare_and_identify_frozen_model(
                model, torch.device("cpu"), model_id=model_id
            )
            self.assertEqual(identity["model_id"], model_id)
            self.assertEqual(identity["musicgen_architecture"]["cardinality"], 2048)
            self.assertEqual(
                identity["musicgen_pattern"]["valid_cells_per_codebook"],
                [500, 500, 500, 500],
            )
            external = model.lm.condition_provider.conditioner.__dict__["t5"]
            self.assertFalse(external.training)
            self.assertFalse(any(parameter.requires_grad for parameter in external.parameters()))

    def test_manifest_and_prompt_pairing_seed_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / gate.DEV_MANIFEST_BASENAME
            path.write_text(
                "".join(
                    json.dumps({
                        "sample_id": "s{}".format(i),
                        "prompt": "p{}".format(i),
                        "source_row_sha256": format(i + 1, "064x"),
                    })
                    + "\n"
                    for i in range(3)
                ),
                encoding="utf-8",
            )
            records = gate.load_dev_manifest(path, expected_prompts=3)
            self.assertEqual(len(records), 3)
            seeds = [gate.paired_prompt_seed(record["sample_id"]) for record in records]
            self.assertEqual(seeds, [gate.paired_prompt_seed(record["sample_id"]) for record in records])
            self.assertEqual(len(set(seeds)), 3)

            forbidden = root / "test.full.jsonl"
            forbidden.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(ValueError):
                gate.load_dev_manifest(forbidden, expected_prompts=3, require_basename=False)

    def test_check_only_preflight_does_not_import_audiocraft(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / gate.DEV_MANIFEST_BASENAME
            manifest.write_text(
                "".join(
                    json.dumps({
                        "sample_id": "s{}".format(i),
                        "prompt": "p{}".format(i),
                        "source_row_sha256": format(i + 1, "064x"),
                    })
                    + "\n"
                    for i in range(gate.EXPECTED_DEV_PROMPTS)
                ),
                encoding="utf-8",
            )
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "state_dict.bin").write_bytes(b"lm")
            (checkpoint / "compression_state_dict.bin").write_bytes(b"codec")
            source = root / "source"
            lm = source / "audiocraft" / "models" / "lm.py"
            lm.parent.mkdir(parents=True)
            lm.write_text(
                "use_cfg: bool = True\ncondition_tensors: optional\n"
                "elif condition_tensors is not None:\n"
                "CFG condition_tensors batch must be even\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                manifest=manifest,
                model_id="facebook/musicgen-medium",
                checkpoint=checkpoint,
                audiocraft_root=source,
                output_dir=root / "output",
            )
            before = sys.modules.get("audiocraft")
            with mock.patch.object(gate, "_git_head", return_value=gate.PINNED_AUDIOCRAFT_BASE_COMMIT):
                _, report = gate.preflight_generation(args)
            self.assertIs(sys.modules.get("audiocraft"), before)
            self.assertFalse(report["audiocraft_imported"])
            self.assertEqual(
                report["planned_config"]["model_id"], "facebook/musicgen-medium"
            )

            checkpoint_link = root / "checkpoint-link"
            checkpoint_link.symlink_to(checkpoint, target_is_directory=True)
            linked_args = argparse.Namespace(**vars(args))
            linked_args.checkpoint = checkpoint_link
            with self.assertRaisesRegex(ValueError, "regular local snapshot"):
                gate.preflight_generation(linked_args)

            source_link = root / "source-link"
            source_link.symlink_to(source, target_is_directory=True)
            linked_args = argparse.Namespace(**vars(args))
            linked_args.audiocraft_root = source_link
            with self.assertRaisesRegex(ValueError, "source root must not be a symlink"):
                gate.preflight_generation(linked_args)

    def test_generate_parser_requires_frozen_small_or_medium_model_id(self) -> None:
        parser = gate.build_parser()
        common = [
            "generate", "--manifest", "m", "--checkpoint", "c",
            "--audiocraft-root", "a", "--output-dir", "o",
        ]
        with self.assertRaises(SystemExit):
            parser.parse_args(common)
        for model_id in gate.ALLOWED_MODEL_IDS:
            args = parser.parse_args(common + ["--model-id", model_id])
            self.assertEqual(args.model_id, model_id)

    @unittest.skipIf(torch is None, "torch is unavailable in this lightweight interpreter")
    def test_conditioner_is_one_fp32_2b_call_outside_lm_autocast(self) -> None:
        class Provider:
            def __init__(self) -> None:
                self.tokenize_calls = 0
                self.forward_calls = 0
                self.autocast_enabled = None

            def tokenize(self, conditions):
                self.tokenize_calls += 1
                self.conditions = conditions
                return conditions

            def __call__(self, tokenized):
                self.forward_calls += 1
                self.autocast_enabled = torch.is_autocast_enabled()
                return {
                    "description": (
                        torch.zeros((len(tokenized), 2, 3), dtype=torch.float32),
                        torch.ones((len(tokenized), 2), dtype=torch.bool),
                    )
                }

        class LM(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.condition_provider = Provider()

        model = type("Model", (), {"lm": LM()})()

        class Attributes:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

        class Dropout:
            def __init__(self, p: float) -> None:
                self.p = p

            def __call__(self, conditions):
                return [object() for _ in conditions]

        fake = type(sys)("audiocraft.modules.conditioners")
        fake.ConditioningAttributes = Attributes
        fake.ClassifierFreeGuidanceDropout = Dropout
        with mock.patch.dict(sys.modules, {"audiocraft.modules.conditioners": fake}):
            conditional, cfg_batched = gate._precompute_prompt_condition_tensors(
                model, "piano"
            )
        self.assertEqual(model.lm.condition_provider.tokenize_calls, 1)
        self.assertEqual(model.lm.condition_provider.forward_calls, 1)
        self.assertFalse(model.lm.condition_provider.autocast_enabled)
        self.assertEqual(conditional["description"][0].shape[0], 1)
        self.assertEqual(cfg_batched["description"][0].shape[0], 2)
        self.assertEqual(conditional["description"][0].dtype, torch.float32)

    @unittest.skipIf(torch is None, "torch is unavailable in this lightweight interpreter")
    def test_lm_uses_bf16_but_codec_decode_explicitly_disables_autocast(self) -> None:
        active = []
        entered = []

        @contextlib.contextmanager
        def fake_autocast(*, device_type, dtype=None, enabled=True):
            state = (device_type, dtype, enabled)
            entered.append(state)
            active.append(state)
            try:
                yield
            finally:
                self.assertEqual(active.pop(), state)

        class LM:
            def parameters(self):
                return iter([type("Parameter", (), {"device": torch.device("cuda")})()])

            def generate(self, *, conditions, condition_tensors, use_cfg, **kwargs):
                self.last_kwargs = {
                    "conditions": conditions,
                    "condition_tensors": condition_tensors,
                    "use_cfg": use_cfg,
                    **kwargs,
                }
                self_outer.assertEqual(active[-1], ("cuda", torch.bfloat16, True))
                return torch.zeros((1, 4, gate.TOKEN_FRAMES), dtype=torch.long)

        class Compression:
            def decode(self, tokens, scale):
                self_outer.assertEqual(active[-1], ("cuda", None, False))
                self_outer.assertIsNone(scale)
                return tokens.to(dtype=torch.float32)

        self_outer = self
        model = type("Model", (), {"lm": LM(), "compression_model": Compression()})()
        with mock.patch.object(torch, "autocast", side_effect=fake_autocast):
            audio = gate._generate_one(
                model,
                conditional_tensors={},
                cfg_batched_tensors={},
                condition="no_cfg",
                scale=None,
                seed=123,
            )
        self.assertEqual(tuple(audio.shape), (1, 4, gate.TOKEN_FRAMES))
        self.assertEqual(
            entered,
            [("cuda", torch.bfloat16, True), ("cuda", None, False)],
        )
        self.assertEqual(gate.LM_GENERATION_COMPUTE_DTYPE, "torch.bfloat16")
        self.assertEqual(gate.COMPRESSION_DECODE_COMPUTE_DTYPE, "torch.float32")

    def test_generation_seal_score_matching_and_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation, records = _fake_generation(root)
            provenance, provenance_hash = _fake_provenance(root, generation)
            scores_path = root / "scores.jsonl"
            rows = _score_rows(records, provenance_hash)
            _write_scores(scores_path, rows)
            matched, identity = gate.load_and_match_scores(
                generation, scores_path, provenance
            )
            self.assertEqual(len(matched), gate.EXPECTED_DEV_PROMPTS * 4)
            self.assertEqual(identity["evaluator_provenance_sha256"], provenance_hash)

            rogue = generation / "unsealed.txt"
            rogue.write_text("not in the artifact contract", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "file set mismatch"):
                gate.verify_generation_directory(generation, rehash_audio=True)
            rogue.unlink()

            generation_link = root / "generation-link"
            generation_link.symlink_to(generation, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "directory must not be a symlink"):
                gate.verify_generation_directory(generation_link, rehash_audio=True)

            audio = generation / records[0]["path"]
            original_payload = audio.read_bytes()
            external_audio = root / "external.wav"
            external_audio.write_bytes(original_payload)
            audio.unlink()
            audio.symlink_to(external_audio)
            with self.assertRaisesRegex(ValueError, "regular non-symlink"):
                gate.verify_generation_directory(generation, rehash_audio=True)
            audio.unlink()
            audio.write_bytes(original_payload)

            audio.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "audio hash mismatch"):
                gate.verify_generation_directory(generation, rehash_audio=True)

    def test_decision_formula_selects_highest_eligible_q_and_seals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation, records = _fake_generation(root)
            quality, external = _fake_sealed_external_artifacts(
                root, generation, records
            )
            output = root / "decision"
            args = argparse.Namespace(
                generation_dir=generation,
                external_evaluation_dir=external,
                quality_dir=quality,
                output_dir=output,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(gate.run_decision(args), 0)
            verified = gate.verify_decision(output)
            self.assertEqual(verified["status"], "selected")
            self.assertEqual(verified["selected_cfg_scale"], 3.0)
            decision = json.loads(
                (output / "cfg_scale_decision.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                decision["input_hashes"][
                    "external_evaluation_artifact_seal_sha256"
                ],
                gate.sha256_file(external / "artifact_seal.json"),
            )
            self.assertEqual(
                decision["input_hashes"]["quality_artifact_seal_sha256"],
                gate.sha256_file(quality / "artifact_seal.json"),
            )
            with self.assertRaises(FileExistsError):
                with contextlib.redirect_stdout(io.StringIO()):
                    gate.run_decision(args)

    def test_decision_rejects_tampered_external_seal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation, records = _fake_generation(root)
            quality, external = _fake_sealed_external_artifacts(
                root, generation, records
            )
            args = argparse.Namespace(
                generation_dir=generation,
                external_evaluation_dir=external,
                quality_dir=quality,
                output_dir=root / "decision",
            )

            seal_path = external / "artifact_seal.json"
            seal = json.loads(seal_path.read_text(encoding="utf-8"))
            seal["score_records"] -= 1
            seal_path.unlink()
            _write_json(seal_path, seal)
            with self.assertRaisesRegex(ValueError, "external evaluation seal"):
                gate.run_decision(args)

    def test_external_provenance_rejects_protocol_and_offline_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation, _ = _fake_generation(root)
            provenance, _ = _fake_provenance(root, generation)
            value = json.loads(provenance.read_text(encoding="utf-8"))
            value["protocol"]["fad_computed"] = True
            provenance.unlink()
            _write_json(provenance, value)
            with self.assertRaisesRegex(ValueError, "protocol mismatch"):
                gate.validate_evaluator_provenance(provenance)

            value["protocol"] = dict(gate.EXTERNAL_EVALUATION_PROTOCOL)
            value["offline_environment"]["HF_HUB_OFFLINE"] = "0"
            provenance.unlink()
            _write_json(provenance, value)
            with self.assertRaisesRegex(ValueError, "offline environment"):
                gate.validate_evaluator_provenance(provenance)

    def test_exact_q_tie_prefers_lower_scale(self) -> None:
        records = []
        for index in range(gate.EXPECTED_DEV_PROMPTS):
            sample_id = "s{:03d}".format(index)
            base = index / 50.0
            for anchor, gain in (
                ("no_cfg", 0.0),
                ("cfg_2.0", 0.4),
                ("cfg_3.0", 0.4),
                ("cfg_5.0", 0.2),
            ):
                records.append(
                    {
                        "sample_id": sample_id,
                        "condition_id": anchor,
                        "metrics": {
                            "muq_mi": base + gain,
                            "audiobox_ce": base + gain,
                            "audiobox_pq": base + gain,
                            "music_clap": base,
                        },
                    }
                )
        decision = gate.decide_from_scores(
            records, bootstrap_seed=1, bootstrap_replicates=100
        )
        self.assertEqual(decision["status"], "selected")
        self.assertEqual(decision["selected_cfg_scale"], 2.0)

    def test_score_set_and_provenance_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation, records = _fake_generation(root)
            provenance, provenance_hash = _fake_provenance(root, generation)
            scores_path = root / "scores.jsonl"
            _write_scores(scores_path, _score_rows(records, provenance_hash)[:-1])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                gate.load_and_match_scores(generation, scores_path, provenance)

            value = json.loads(provenance.read_text(encoding="utf-8"))
            value["status"] = "unverified"
            bad_provenance = root / "bad_provenance.json"
            _write_json(bad_provenance, value)
            with self.assertRaisesRegex(ValueError, "not explicitly accepted"):
                gate.validate_evaluator_provenance(bad_provenance)


if __name__ == "__main__":
    unittest.main()
