"""Strict-consumer tests for immutable CFG-scale decisions."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ptc_opd.cfg_decision import (
    DECISION_FILENAME,
    DECISION_SCHEMA_VERSION,
    DECISION_SIDECAR_FILENAME,
    DECISION_SIDECAR_SCHEMA_VERSION,
    REQUIRED_INPUT_HASHES,
    REQUIRED_GENERATION_IDENTITY,
    canonical_json_sha256,
    sha256_file,
    verify_cfg_scale_decision,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _fixture(root: Path, *, selected: float = 3.0, status: str = "selected") -> Path:
    root.mkdir()
    config = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "evaluation_manifest": "dev.full.jsonl",
        "candidates": [2.0, 3.0, 5.0],
        "base_anchor": "no_cfg",
        "quality_formula": {
            "q_dev": "0.5*delta_z_muq_mi + 0.5*delta_z_aesthetic",
            "aesthetic": "0.5*delta_z_audiobox_ce + 0.5*delta_z_audiobox_pq",
            "expanded": "0.5*delta_z_muq_mi + 0.25*delta_z_audiobox_ce + 0.25*delta_z_audiobox_pq",
            "standardization": "paired raw delta divided by no_cfg base sample SD",
        },
        "guardrails": {
            "q_dev_strictly_positive": True,
            "paired_bootstrap_probability_positive_gte": 0.90,
            "music_clap_delta_base_sd_gte": -0.10,
        },
        "paired_prompt_bootstrap": {
            "seed": 4703,
            "replicates": 10_000,
            "confidence_interval": 0.95,
        },
        "selection": "highest eligible q_dev",
        "tie_break": "lower cfg scale only on exact numerical q_dev equality",
    }
    decision = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "status": status,
        "selected_cfg_scale": selected if status == "selected" else None,
        "scientific_config": config,
        "scientific_config_sha256": canonical_json_sha256(config),
        "input_hashes": {
            name: format(index + 1, "064x")
            for index, name in enumerate(REQUIRED_INPUT_HASHES)
        },
        "generation_identity": {
            name: (
                "facebook/musicgen-small"
                if name == "model_id"
                else "torch.float32"
                if name in {
                    "lm_parameter_dtype",
                    "compression_parameter_dtype",
                    "conditioner_parameter_dtype",
                    "conditioner_compute_dtype",
                    "compression_decode_compute_dtype",
                }
                else "torch.bfloat16"
                if name == "lm_generation_compute_dtype"
                else "896ec7c47f5e5d1e5aa1e4b260c4405328bf009d"
                if name == "audiocraft_base_commit"
                else format(index + 20, "064x")
            )
            for index, name in enumerate(REQUIRED_GENERATION_IDENTITY)
        },
        "base_standardization": {
            name: {"mean": 0.0, "sample_sd": 1.0}
            for name in ("muq_mi", "audiobox_ce", "audiobox_pq", "music_clap")
        },
        "candidates": (
            [
                {
                    "cfg_scale": 2.0, "q_dev": 0.2, "q_dev_ci95": [0.1, 0.3],
                    "paired_bootstrap_probability_positive": 0.95,
                    "music_clap_delta_base_sd": 0.0,
                    "component_delta_base_sd": {
                        "muq_mi": 0.2, "audiobox_ce": 0.2,
                        "audiobox_pq": 0.2, "music_clap": 0.0,
                    }, "eligible": True,
                },
                {
                    "cfg_scale": 3.0, "q_dev": 0.3, "q_dev_ci95": [0.2, 0.4],
                    "paired_bootstrap_probability_positive": 0.96,
                    "music_clap_delta_base_sd": 0.0,
                    "component_delta_base_sd": {
                        "muq_mi": 0.3, "audiobox_ce": 0.3,
                        "audiobox_pq": 0.3, "music_clap": 0.0,
                    }, "eligible": True,
                },
                {
                    "cfg_scale": 5.0, "q_dev": -0.1, "q_dev_ci95": [-0.2, 0.0],
                    "paired_bootstrap_probability_positive": 0.4,
                    "music_clap_delta_base_sd": 0.0,
                    "component_delta_base_sd": {
                        "muq_mi": -0.1, "audiobox_ce": -0.1,
                        "audiobox_pq": -0.1, "music_clap": 0.0,
                    }, "eligible": False,
                },
            ]
            if status == "selected"
            else [
                {
                    "cfg_scale": scale, "q_dev": q_dev,
                    "q_dev_ci95": [q_dev - 0.1, q_dev + 0.1],
                    "paired_bootstrap_probability_positive": 0.1,
                    "music_clap_delta_base_sd": 0.0,
                    "component_delta_base_sd": {
                        "muq_mi": q_dev, "audiobox_ce": q_dev,
                        "audiobox_pq": q_dev, "music_clap": 0.0,
                    }, "eligible": False,
                }
                for scale, q_dev in ((2.0, -0.2), (3.0, -0.3), (5.0, -0.1))
            ]
        ),
    }
    decision["decision_payload_sha256"] = canonical_json_sha256(decision)
    decision_path = root / DECISION_FILENAME
    _write_json(decision_path, decision)
    _write_json(
        root / DECISION_SIDECAR_FILENAME,
        {
            "schema_version": DECISION_SIDECAR_SCHEMA_VERSION,
            "path": DECISION_FILENAME,
            "sha256": sha256_file(decision_path),
            "decision_payload_sha256": decision["decision_payload_sha256"],
        },
    )
    return root


class CFGDecisionConsumerTest(unittest.TestCase):
    def test_accepts_selected_consistent_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = verify_cfg_scale_decision(
                _fixture(Path(temporary) / "decision")
            )
            self.assertEqual(result.selected_cfg_scale, 3.0)
            self.assertEqual(len(result.decision_file_sha256), 64)

    def test_rejects_stop_tamper_and_rule_inconsistency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            stopped = _fixture(base / "stopped", status="stopped_no_eligible_scale")
            with self.assertRaisesRegex(ValueError, "did not select"):
                verify_cfg_scale_decision(stopped)
            inspected = verify_cfg_scale_decision(stopped, require_selected=False)
            self.assertEqual(inspected.status, "stopped_no_eligible_scale")
            self.assertIsNone(inspected.selected_cfg_scale)

            tampered = _fixture(base / "tampered")
            with (tampered / DECISION_FILENAME).open("a", encoding="utf-8") as stream:
                stream.write(" ")
            with self.assertRaisesRegex(ValueError, "file hash"):
                verify_cfg_scale_decision(tampered)

            inconsistent = _fixture(base / "inconsistent", selected=2.0)
            with self.assertRaisesRegex(ValueError, "selection rule"):
                verify_cfg_scale_decision(inconsistent)

    def test_rejects_unexpected_directory_member(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = _fixture(Path(temporary) / "decision")
            (path / "manual-note.txt").write_text("not immutable\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "members differ"):
                verify_cfg_scale_decision(path)

    def test_rejects_dtype_and_guardrail_flag_tamper_even_when_rehashed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            dtype_path = _fixture(base / "dtype")
            decision_file = dtype_path / DECISION_FILENAME
            decision = json.loads(decision_file.read_text(encoding="utf-8"))
            decision["generation_identity"]["lm_generation_compute_dtype"] = "torch.float16"
            decision["decision_payload_sha256"] = canonical_json_sha256(
                {key: value for key, value in decision.items() if key != "decision_payload_sha256"}
            )
            _write_json(decision_file, decision)
            _write_json(
                dtype_path / DECISION_SIDECAR_FILENAME,
                {
                    "schema_version": DECISION_SIDECAR_SCHEMA_VERSION,
                    "path": DECISION_FILENAME,
                    "sha256": sha256_file(decision_file),
                    "decision_payload_sha256": decision["decision_payload_sha256"],
                },
            )
            with self.assertRaisesRegex(ValueError, "precision identity"):
                verify_cfg_scale_decision(dtype_path)

            decode_path = _fixture(base / "decode")
            decision_file = decode_path / DECISION_FILENAME
            decision = json.loads(decision_file.read_text(encoding="utf-8"))
            decision["generation_identity"][
                "compression_decode_compute_dtype"
            ] = "torch.bfloat16"
            decision["decision_payload_sha256"] = canonical_json_sha256(
                {key: value for key, value in decision.items() if key != "decision_payload_sha256"}
            )
            _write_json(decision_file, decision)
            _write_json(
                decode_path / DECISION_SIDECAR_FILENAME,
                {
                    "schema_version": DECISION_SIDECAR_SCHEMA_VERSION,
                    "path": DECISION_FILENAME,
                    "sha256": sha256_file(decision_file),
                    "decision_payload_sha256": decision["decision_payload_sha256"],
                },
            )
            with self.assertRaisesRegex(ValueError, "precision identity"):
                verify_cfg_scale_decision(decode_path)

            flag_path = _fixture(base / "eligible")
            decision_file = flag_path / DECISION_FILENAME
            decision = json.loads(decision_file.read_text(encoding="utf-8"))
            decision["candidates"][0]["eligible"] = False
            decision["decision_payload_sha256"] = canonical_json_sha256(
                {key: value for key, value in decision.items() if key != "decision_payload_sha256"}
            )
            _write_json(decision_file, decision)
            _write_json(
                flag_path / DECISION_SIDECAR_FILENAME,
                {
                    "schema_version": DECISION_SIDECAR_SCHEMA_VERSION,
                    "path": DECISION_FILENAME,
                    "sha256": sha256_file(decision_file),
                    "decision_payload_sha256": decision["decision_payload_sha256"],
                },
            )
            with self.assertRaisesRegex(ValueError, "eligible flag"):
                verify_cfg_scale_decision(flag_path)


if __name__ == "__main__":
    unittest.main()
