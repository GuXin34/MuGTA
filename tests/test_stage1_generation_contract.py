import json
from pathlib import Path
import tempfile
import unittest

from ptc_opd.stage1_control import publish_pilot_eval_manifest
from ptc_opd.stage1_generation import (
    GENERATION_SEEDS,
    _verify_generation_config,
    derive_generation_seed,
    expected_keys,
    load_eval_manifest_artifact,
    verify_generation_artifact,
    verify_trained_run_source_binding,
)


H = "1" * 64


def valid_base_config():
    return {
        "schema_version": "ptc-opd-stage1-generation-v1",
        "model_id": "facebook/musicgen-small",
        "source_kind": "base_no_cfg",
        "condition_id": "base_no_cfg",
        "method": None,
        "train_seed": None,
        "learning_rate": None,
        "checkpoint_step": 0,
        "base_checkpoint": {
            "checkpoint_sha256": H,
            "state_dict_sha256": H,
            "compression_state_dict_sha256": H,
        },
        "trained_checkpoint": None,
        "stage1_run": None,
        "stage1_run_tree_sha256": None,
        "audiocraft_source_sha256": H,
        "cfg_decision": {
            "decision_file_sha256": H,
            "decision_payload_sha256": H,
            "scientific_config_sha256": H,
            "selected_cfg_scale": 5.0,
            "loaded_t5_identity_sha256": H,
        },
        "eval_manifest_artifact_seal_sha256": H,
        "eval_manifest_sha256": H,
        "prompt_count": 128,
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
            "loaded_t5_identity": {"identity_sha256": H},
            "precision_contract": {
                "lm_parameter_dtype": "torch.float32",
                "compression_parameter_dtype": "torch.float32",
                "conditioner_parameter_dtype": "torch.float32",
                "conditioner_compute_dtype": "torch.float32",
                "lm_generation_compute_dtype": "torch.bfloat16",
                "compression_decode_compute_dtype": "torch.float32",
            },
        },
        "base_lm_state_sha256": H,
        "loaded_trained_lm_state_sha256": None,
    }


class Stage1GenerationContractTests(unittest.TestCase):
    def test_generation_verifier_rejects_symlink_root_before_parsing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.mkdir()
            link = root / "generation-link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "root must not be a symlink"):
                verify_generation_artifact(
                    link,
                    eval_manifest_dir=root / "unused-manifest",
                )

    def test_eval_manifest_verifier_rejects_symlink_root_before_parsing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.mkdir()
            link = root / "manifest-link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "root must not be a symlink"):
                load_eval_manifest_artifact(link)

    def _run_source_fixture(self):
        base = {
            "checkpoint_sha256": "2" * 64,
            "state_dict_sha256": "3" * 64,
            "compression_state_dict_sha256": "4" * 64,
        }
        source_sha = "5" * 64
        cfg = {
            "selected_cfg_scale": 5.0,
            "decision_file_sha256": "6" * 64,
            "decision_payload_sha256": "7" * 64,
            "scientific_config_sha256": "8" * 64,
            "generation_identity": {
                "checkpoint_sha256": base["checkpoint_sha256"],
                "state_dict_sha256": base["state_dict_sha256"],
                "compression_state_dict_sha256": base[
                    "compression_state_dict_sha256"
                ],
                "audiocraft_source_sha256": source_sha,
                "loaded_t5_identity_sha256": "9" * 64,
            },
        }
        run = {
            "student_checkpoint_sha256": base["checkpoint_sha256"],
            "teacher_checkpoint_sha256": base["checkpoint_sha256"],
            "student_state_dict_sha256": base["state_dict_sha256"],
            "teacher_state_dict_sha256": base["state_dict_sha256"],
            "audiocraft_source_identity": {"tree_sha256": source_sha},
            "cfg_scale_decision": dict(cfg),
            "cfg_generation_binding": {
                "checkpoint_sha256": base["checkpoint_sha256"],
                "state_dict_sha256": base["state_dict_sha256"],
                "audiocraft_source_sha256": source_sha,
            },
            "loaded_t5_identity": {"identity_sha256": "9" * 64},
            "config": {
                "cfg_scale_decision_file_sha256": cfg[
                    "decision_file_sha256"
                ],
                "cfg_scale_decision_payload_sha256": cfg[
                    "decision_payload_sha256"
                ],
                "cfg_scale_scientific_config_sha256": cfg[
                    "scientific_config_sha256"
                ],
                "cfg_generation_checkpoint_sha256": base[
                    "checkpoint_sha256"
                ],
                "cfg_generation_state_dict_sha256": base["state_dict_sha256"],
                "cfg_generation_audiocraft_source_sha256": source_sha,
                "cfg_generation_loaded_t5_identity_sha256": "9" * 64,
                "teacher_cfg_scale": 5.0,
            },
        }
        return run, base, source_sha, cfg

    def test_trained_generation_rejects_cross_base_or_cfg_lineage(self):
        run, base, source_sha, cfg = self._run_source_fixture()
        verify_trained_run_source_binding(
            run,
            base_checkpoint=base,
            audiocraft_source_sha256=source_sha,
            cfg_scale_decision=cfg,
        )

        tampered = dict(run)
        tampered["student_checkpoint_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "checkpoint binding"):
            verify_trained_run_source_binding(
                tampered,
                base_checkpoint=base,
                audiocraft_source_sha256=source_sha,
                cfg_scale_decision=cfg,
            )

        tampered = dict(run)
        tampered["cfg_scale_decision"] = {
            **cfg,
            "decision_payload_sha256": "b" * 64,
        }
        with self.assertRaisesRegex(ValueError, "CFG decision"):
            verify_trained_run_source_binding(
                tampered,
                base_checkpoint=base,
                audiocraft_source_sha256=source_sha,
                cfg_scale_decision=cfg,
            )

    def test_seed_is_repeatable_paired_and_domain_separated_by_base_seed(self):
        a = derive_generation_seed("id-1", 31001)
        self.assertEqual(a, derive_generation_seed("id-1", 31001))
        self.assertNotEqual(a, derive_generation_seed("id-1", 31002))
        self.assertNotEqual(a, derive_generation_seed("id-2", 31001))
        self.assertGreaterEqual(a, 0)
        self.assertLess(a, 2**63)

    def test_expected_key_plan_is_exact_128_times_two_shape_independent(self):
        records = [
            {"sample_id": "b", "prompt": "B"},
            {"sample_id": "a", "prompt": "A"},
        ]
        self.assertEqual(
            expected_keys(records),
            [("a", GENERATION_SEEDS[0]), ("a", GENERATION_SEEDS[1]),
             ("b", GENERATION_SEEDS[0]), ("b", GENERATION_SEEDS[1])],
        )

    def test_unregistered_generation_seed_is_rejected(self):
        with self.assertRaises(ValueError):
            derive_generation_seed("id", 999)

    def test_base_generation_config_is_closed_and_fp32_codec_decode(self):
        manifest = {"artifact_seal_sha256": H, "manifest_sha256": H}
        _verify_generation_config(valid_base_config(), manifest)

        extra = valid_base_config()
        extra["unreviewed"] = True
        with self.assertRaisesRegex(ValueError, "field set"):
            _verify_generation_config(extra, manifest)

        bf16_decode = valid_base_config()
        bf16_decode["precision"]["compression_decode_compute"] = "torch.bfloat16"
        with self.assertRaisesRegex(ValueError, "precision"):
            _verify_generation_config(bf16_decode, manifest)

    def test_generation_config_binds_both_manifest_files(self):
        with self.assertRaisesRegex(ValueError, "supplied eval manifest"):
            _verify_generation_config(
                valid_base_config(),
                {"artifact_seal_sha256": H, "manifest_sha256": "2" * 64},
            )

    def test_pilot_manifest_producer_and_generation_consumer_are_aligned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "dev.full.jsonl"
            with source.open("w", encoding="utf-8") as stream:
                for index in range(300):
                    stream.write(
                        json.dumps(
                            {
                                "sample_id": "sample-{:03d}".format(index),
                                "prompt": "prompt {}".format(index),
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
            artifact = root / "pilot"
            publish_pilot_eval_manifest(source, artifact)
            records, identity = load_eval_manifest_artifact(artifact)
            self.assertEqual(len(records), 128)
            self.assertEqual(identity["metadata"]["source"]["record_count"], 300)


if __name__ == "__main__":
    unittest.main()
