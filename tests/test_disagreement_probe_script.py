from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ptc_opd.phenomena import CELL_SCHEMA_VERSION, prompt_sha256


spec = importlib.util.spec_from_file_location(
    "run_disagreement_probe", ROOT / "scripts" / "run_disagreement_probe.py"
)
assert spec is not None and spec.loader is not None
probe_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe_script)


def _cell(
    sample_id: str,
    seed: int,
    time_index: int,
    selected: bool,
    *,
    q_index: int = 0,
) -> dict:
    return {
        "schema_version": CELL_SCHEMA_VERSION,
        "sample_id": sample_id,
        "prompt_sha256": prompt_sha256("prompt " + sample_id),
        "rollout_seed": seed,
        "q": q_index,
        "t": time_index,
        "temporal_decile": min(9, (10 * time_index) // 500),
        "js": float(2 - min(time_index, 1)),
        "forward_kl": float(time_index + 1),
        "teacher_entropy": 2.0,
        "student_entropy": 1.0,
        "sampled_token_logp_teacher": -0.9,
        "sampled_token_logp_student": -1.0,
        "a_q": 1.0,
        "top50_js": selected,
    }


def _audit_cell(sample_id: str, seed: int, time_index: int) -> dict:
    return {
        "schema_version": "ptc-opd-disagreement-topk-audit-v1",
        "sample_id": sample_id,
        "prompt_sha256": prompt_sha256("prompt " + sample_id),
        "rollout_seed": seed,
        "q": 0,
        "t": time_index,
        "sampled_token_id": 7,
        "student_top_token_ids": list(range(32)),
        "student_top_logits": [float(31 - index) for index in range(32)],
        "teacher_top_token_ids": list(range(32, 64)),
        "teacher_top_logits": [float(31 - index) / 2.0 for index in range(32)],
    }


def _write_debug_probe(root: Path, sample_ids=("a", "b")) -> Path:
    directory = root / "sealed_probe"
    directory.mkdir()
    cells = []
    audits = []
    for sample_id in sample_ids:
        for seed in probe_script.PRIMARY_ROLLOUT_SEEDS:
            cells.extend(
                [
                    _cell(sample_id, seed, 0, True),
                    _cell(sample_id, seed, 1, False),
                ]
            )
            audits.extend(
                [_audit_cell(sample_id, seed, 0), _audit_cell(sample_id, seed, 1)]
            )
    cell_path = directory / probe_script.PROBE_CELLS_FILENAME
    audit_path = directory / probe_script.PROBE_AUDIT_FILENAME
    metadata_path = directory / probe_script.PROBE_METADATA_FILENAME
    probe_script.write_jsonl_gzip_atomic(cell_path, cells)
    probe_script.write_jsonl_gzip_atomic(audit_path, audits)
    config = {
        "schema_version": probe_script.PROBE_CONFIG_SCHEMA_VERSION,
        "primary_contract": False,
        "device_type": "cpu",
        "rollout_seeds": list(probe_script.PRIMARY_ROLLOUT_SEEDS),
    }
    metadata = {
        "schema_version": probe_script.PROBE_RUN_SCHEMA_VERSION,
        "scientific_status": "cpu_debug_only",
        "primary_contract_passed": False,
        "scientific_config": config,
        "scientific_config_sha256": probe_script.sha256_json(config),
        "student_teacher_same_checkpoint_and_state": True,
        "condition_tensors_reused_for_rollout_and_scoring": True,
        "strict_raw_branch_contract_before_intersection": True,
        "probe_sample_ids": list(sample_ids),
        "audit_sample_ids": list(sample_ids),
        "records_per_sequence": 2,
        "cell_count": len(cells),
        "expected_cell_count": len(cells),
        "audit_prompt_count": len(sample_ids),
        "audit_top_k": 32,
        "audit_cell_count": len(audits),
        "expected_audit_cell_count": len(audits),
        "outputs": {
            cell_path.name: probe_script.artifact_member(cell_path),
            audit_path.name: probe_script.artifact_member(audit_path),
        },
    }
    probe_script.write_json_atomic(metadata_path, metadata)
    probe_script.write_artifact_seal(
        directory / probe_script.PROBE_SEAL_FILENAME,
        schema_version=probe_script.PROBE_SEAL_SCHEMA_VERSION,
        status="complete_cpu_debug",
        members=[cell_path, audit_path, metadata_path],
    )
    probe_script.verify_probe_artifact(directory, require_primary=False)
    return directory


class DisagreementProbeScriptTest(unittest.TestCase):
    def test_cfg_precision_consumer_uses_canonical_fp32_decode_contract(self) -> None:
        self.assertEqual(
            probe_script.EXPECTED_PRECISION_IDENTITY,
            {
                "lm_parameter_dtype": "torch.float32",
                "compression_parameter_dtype": "torch.float32",
                "conditioner_parameter_dtype": "torch.float32",
                "conditioner_compute_dtype": "torch.float32",
                "lm_generation_compute_dtype": "torch.bfloat16",
                "compression_decode_compute_dtype": "torch.float32",
            },
        )

    def test_checkpoint_snapshot_must_be_fully_dereferenced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "snapshot"
            blobs = snapshot / "blobs"
            blobs.mkdir(parents=True)
            (blobs / "lm.bin").write_bytes(b"lm")
            (snapshot / "compression_state_dict.bin").write_bytes(b"codec")
            (snapshot / "state_dict.bin").symlink_to(blobs / "lm.bin")
            with self.assertRaisesRegex(ValueError, "fully dereferenced"):
                probe_script.verify_checkpoint_snapshot(snapshot)

    def test_prepare_frozen_lm_moves_unregistered_t5_and_sets_device(self) -> None:
        class Conditioner(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.device = "wrong"
                self.__dict__["t5"] = torch.nn.Linear(2, 2)

        class Provider(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conditioner = Conditioner()

        class LM(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.condition_provider = Provider()

        lm = LM()
        external = lm.condition_provider.conditioner.__dict__["t5"]
        probe_script._prepare_frozen_lm(lm, torch.device("cpu"))
        self.assertFalse(lm.training)
        self.assertFalse(lm.condition_provider.training)
        self.assertFalse(external.training)
        self.assertEqual(lm.condition_provider.conditioner.device, "cpu")
        self.assertFalse(any(parameter.requires_grad for parameter in lm.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in external.parameters()))

    def test_a2_architecture_is_frozen_to_musicgen_small(self) -> None:
        class Transformer:
            dim = 1024
            num_layers = 24
            num_heads = 16

        lm = type(
            "LM",
            (),
            {
                "num_codebooks": 4,
                "card": 2048,
                "cfg": type("Cfg", (), {"transformer_lm": Transformer()})(),
            },
        )()
        self.assertEqual(
            probe_script._require_musicgen_small_architecture(lm),
            probe_script.MUSICGEN_SMALL_ARCHITECTURE,
        )
        lm.cfg.transformer_lm.dim = 1536
        with self.assertRaisesRegex(RuntimeError, "MusicGen-small architecture"):
            probe_script._require_musicgen_small_architecture(lm)

    def test_manifest_validation_exact_dev_subset_and_blank_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dev_path = root / probe_script.DEV_MANIFEST_BASENAME
            probe_path = root / probe_script.PROBE_MANIFEST_BASENAME
            dev = []
            for index in range(300):
                dev.append(
                    {
                        "sample_id": "id-{:03d}".format(index),
                        "prompt": "prompt {} café".format(index),
                        "source_row_sha256": hashlib.sha256(
                            "source {}".format(index).encode("utf-8")
                        ).hexdigest(),
                    }
                )
            dev_path.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in dev),
                encoding="utf-8",
            )
            probe_path.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in dev[:256]),
                encoding="utf-8",
            )
            probe, loaded_dev = probe_script.validate_probe_is_exact_dev_subset(
                probe_path, dev_path
            )
            self.assertEqual(len(probe), 256)
            self.assertEqual(len(loaded_dev), 300)

            mutated = list(dev[:256])
            mutated[4] = dict(mutated[4], prompt=mutated[4]["prompt"] + " ")
            probe_path.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in mutated),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                probe_script.validate_probe_is_exact_dev_subset(probe_path, dev_path)

            probe_path.write_text(
                json.dumps(dev[0], ensure_ascii=False) + "\n\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                probe_script.load_probe_manifest(probe_path, expected_prompts=1)

    def test_manifest_test_name_and_missing_source_row_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            forbidden = root / "test.full.jsonl"
            forbidden.write_text('{"sample_id":"x","prompt":"p"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                probe_script.load_probe_manifest(forbidden, expected_prompts=1)
            primary = root / probe_script.PROBE_MANIFEST_BASENAME
            primary.write_text('{"sample_id":"x","prompt":"p"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                probe_script.load_probe_manifest(primary, expected_prompts=1)

    def test_gzip_writer_refuses_overwrite_and_blank_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cells.jsonl.gz"
            records = [_cell("x", 31001, 0, True), _cell("x", 31001, 1, False)]
            self.assertEqual(probe_script.write_jsonl_gzip_atomic(path, records), 2)
            self.assertEqual(list(probe_script.iter_jsonl_gzip(path)), records)
            with self.assertRaises(FileExistsError):
                probe_script.write_jsonl_gzip_atomic(path, records)

    def test_completeness_gate_rejects_duplicate_and_wrong_lattice(self) -> None:
        records = []
        for q_index in range(4):
            for time_index in range(500 - q_index):
                records.append(
                    _cell("a", 31001, time_index, time_index < (500 - q_index + 1) // 2, q_index=q_index)
                )
        self.assertEqual(
            len(
                list(
                    probe_script.validated_probe_records(
                        records,
                        expected_sample_ids=["a"],
                        expected_seeds=[31001],
                        expected_records_per_sequence=1994,
                    )
                )
            ),
            1994,
        )
        duplicate = list(records)
        duplicate[-1] = dict(duplicate[0])
        with self.assertRaises(ValueError):
            list(
                probe_script.validated_probe_records(
                    duplicate,
                    expected_sample_ids=["a"],
                    expected_seeds=[31001],
                    expected_records_per_sequence=1994,
                )
            )
        bad_decile = [dict(records[0], temporal_decile=9)] + records[1:]
        with self.assertRaises(ValueError):
            list(
                probe_script.validated_probe_records(
                    bad_decile,
                    expected_sample_ids=["a"],
                    expected_seeds=[31001],
                    expected_records_per_sequence=1994,
                )
            )

    def test_audit_gate_rejects_wrong_top32_length_and_token(self) -> None:
        record = _audit_cell("a", 31001, 0)
        bad_length = dict(record, student_top_logits=[0.0] * 31)
        with self.assertRaises(ValueError):
            list(
                probe_script.validated_probe_records(
                    [bad_length],
                    expected_sample_ids=["a"],
                    expected_seeds=[31001],
                    expected_records_per_sequence=1,
                    audit_top_k=32,
                )
            )
        bad_token = dict(record, teacher_top_token_ids=[2048] + list(range(31)))
        with self.assertRaises(ValueError):
            list(
                probe_script.validated_probe_records(
                    [bad_token],
                    expected_sample_ids=["a"],
                    expected_seeds=[31001],
                    expected_records_per_sequence=1,
                    audit_top_k=32,
                )
            )

    def test_raw_three_branch_contract_is_called_before_cfg_arithmetic(self) -> None:
        codes = torch.zeros((1, 4, 500), dtype=torch.long)
        student_mask = torch.ones((1, 4, 500), dtype=torch.bool)
        teacher_mask = torch.cat(
            (
                torch.zeros((1, 4, 500), dtype=torch.bool),
                torch.ones((1, 4, 500), dtype=torch.bool),
            ),
            dim=0,
        )
        student_output = types.SimpleNamespace(
            logits=torch.zeros((1, 4, 500, 1)), mask=student_mask
        )
        teacher_output = types.SimpleNamespace(
            logits=torch.zeros((2, 4, 500, 1)), mask=teacher_mask
        )

        class LM:
            card = 2048

            def __init__(self, output):
                self.output = output

            def compute_predictions(self, *args, **kwargs):
                return self.output

        captured = {}

        def strict(codes_arg, card, masks, logits):
            captured["masks"] = masks
            captured["logits"] = logits
            return torch.zeros_like(student_mask)

        with mock.patch.object(probe_script, "batch_condition_tensors", return_value={}), mock.patch.object(
            probe_script, "strict_validate_musicgen_batch", side_effect=strict
        ) as gate:
            probe_script._score_probe_no_grad(
                LM(student_output),
                LM(teacher_output),
                codes,
                {},
                {},
                {},
                teacher_cfg_scale=3.0,
                bf16=False,
            )
        gate.assert_called_once()
        self.assertIs(captured["masks"]["student"], student_mask)
        self.assertTrue(captured["masks"]["teacher_cond"].eq(False).all())
        self.assertTrue(captured["masks"]["teacher_null"].eq(True).all())

    def test_debug_cuda_is_rejected_before_any_artifact_can_be_written(self) -> None:
        args = probe_script.build_parser().parse_args(
            [
                "probe",
                "--probe-manifest",
                "missing-probe",
                "--dev-manifest",
                "missing-dev",
                "--checkpoint",
                "missing-checkpoint",
                "--audiocraft-root",
                "missing-source",
                "--codebook-prior-artifact-dir",
                "missing-prior",
                "--cfg-scale-decision-dir",
                "missing-decision",
                "--output-dir",
                "missing-output",
                "--device",
                "cuda:0",
                "--bf16",
                "--batch-size",
                "3",
                "--allow-cpu-debug",
            ]
        )
        with self.assertRaisesRegex(ValueError, "only valid with --device cpu"):
            probe_script.run_probe(args)

    def test_parser_has_only_artifact_directory_interfaces(self) -> None:
        parser = probe_script.build_parser()
        subparsers = next(
            action for action in parser._actions if hasattr(action, "choices") and action.choices
        )
        probe_options = {
            option
            for action in subparsers.choices["probe"]._actions
            for option in action.option_strings
        }
        self.assertIn("--cfg-scale-decision-dir", probe_options)
        self.assertIn("--codebook-prior-artifact-dir", probe_options)
        self.assertIn("--output-dir", probe_options)
        self.assertNotIn("--teacher-cfg-scale", probe_options)
        self.assertNotIn("--prior", probe_options)
        summary_options = {
            option
            for action in subparsers.choices["summarize"]._actions
            for option in action.option_strings
        }
        self.assertEqual(
            {"--probe-dir", "--output-dir"} - summary_options,
            set(),
        )
        self.assertNotIn("--input-cells", summary_options)

    def test_sealed_probe_verifier_rejects_extra_and_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = _write_debug_probe(root)
            extra = directory / "unexpected.txt"
            extra.write_text("x", encoding="utf-8")
            with self.assertRaises(ValueError):
                probe_script.verify_probe_artifact(directory, require_primary=False)
            extra.unlink()
            with (directory / probe_script.PROBE_CELLS_FILENAME).open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaises(ValueError):
                probe_script.verify_probe_artifact(directory, require_primary=False)

    def test_summarize_consumes_and_emits_only_sealed_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            probe_dir = _write_debug_probe(root)
            output_dir = root / "summary"
            rc = probe_script.main(
                [
                    "summarize",
                    "--probe-dir",
                    str(probe_dir),
                    "--output-dir",
                    str(output_dir),
                    "--bootstrap-replicates",
                    "16",
                    "--allow-nonprimary-input",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {
                    probe_script.SUMMARY_CSV_FILENAME,
                    probe_script.SUMMARY_NPZ_FILENAME,
                    probe_script.SUMMARY_JSON_FILENAME,
                    probe_script.SUMMARY_SEAL_FILENAME,
                },
            )
            verified = probe_script.verify_summary_artifact(
                output_dir, require_primary=False
            )
            payload = verified["metadata"]
            self.assertEqual(payload["prompt_count"], 2)
            self.assertEqual(payload["sequence_count"], 4)
            self.assertEqual(payload["bootstrap"]["unit"], "prompt")
            with self.assertRaises(FileExistsError):
                probe_script.main(
                    [
                        "summarize",
                        "--probe-dir",
                        str(probe_dir),
                        "--output-dir",
                        str(output_dir),
                        "--bootstrap-replicates",
                        "16",
                        "--allow-nonprimary-input",
                    ]
                )

    def test_primary_summary_rejects_debug_probe_without_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            probe_dir = _write_debug_probe(root, sample_ids=("a",))
            output_dir = root / "summary"
            with self.assertRaises(ValueError):
                probe_script.main(
                    [
                        "summarize",
                        "--probe-dir",
                        str(probe_dir),
                        "--output-dir",
                        str(output_dir),
                    ]
                )
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
