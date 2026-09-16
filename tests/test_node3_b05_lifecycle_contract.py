"""Model-free regression guards for the patch06 B05 lifecycle contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = ROOT / "scripts" / "run_node3_gate.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("patch06_gate_contract", GATE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import node-3 gate")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GATE = _load_gate()


def _write_attempt0(run: Path, marker: str = "same") -> None:
    logs = run / "logs"
    logs.mkdir(parents=True)
    (logs / "attempt-0000.json").write_text(
        json.dumps({"marker": marker}, sort_keys=True) + "\n", encoding="utf-8"
    )
    (logs / "metrics.attempt-0000.jsonl").write_text(
        json.dumps({"marker": marker}, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run / "run_manifest.json").write_text(
        json.dumps({"marker": marker}, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run / "status.json").write_text(
        json.dumps({"status": "running", "optimizer_step": 0}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (run / "FAILED.json").write_text(
        json.dumps({"marker": marker}, sort_keys=True) + "\n", encoding="utf-8"
    )
    for step in (0, 1):
        checkpoint = run / "checkpoints" / "step-{:05d}".format(step)
        checkpoint.mkdir(parents=True)
        (checkpoint / "checkpoint.pt").write_bytes(
            "{}-{}".format(marker, step).encode("ascii")
        )
        (checkpoint / "SHA256.json").write_text(
            json.dumps({"marker": marker, "step": step}, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class ColdResumeRotationTest(unittest.TestCase):
    def test_rotation_retains_completed_and_restores_untouched_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            completed = root / "resume"
            seed = root / "resume.seed"
            retained = root / "resume.cold-a"
            _write_attempt0(completed)
            _write_attempt0(seed)
            (completed / "DONE.json").write_text("{}\n", encoding="utf-8")

            result = GATE.rotate_completed_resume_with_seed(
                completed=completed,
                seed=seed,
                retained=retained,
                runs_root=root,
            )

            self.assertEqual(result["status"], "rotated")
            self.assertEqual(
                result["restored_seed_closed_world"]["status"], "closed_world"
            )
            self.assertTrue((retained / "DONE.json").is_file())
            self.assertTrue((completed / "FAILED.json").is_file())
            self.assertFalse(seed.exists())
            self.assertEqual(
                GATE._attempt0_identities(retained),
                GATE._attempt0_identities(completed),
            )

    def test_rotation_rejects_tampered_seed_without_moving_anything(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            completed = root / "resume"
            seed = root / "resume.seed"
            retained = root / "resume.cold-a"
            _write_attempt0(completed, "a")
            _write_attempt0(seed, "b")
            with self.assertRaisesRegex(GATE.Node3GateError, "seed bytes differ"):
                GATE.rotate_completed_resume_with_seed(
                    completed=completed,
                    seed=seed,
                    retained=retained,
                    runs_root=root,
                )
            self.assertTrue(completed.is_dir())
            self.assertTrue(seed.is_dir())
            self.assertFalse(retained.exists())

    def test_closed_world_seed_rejects_extra_root_or_nested_member(self) -> None:
        for location in ("root", "logs", "checkpoints"):
            with self.subTest(location=location), tempfile.TemporaryDirectory() as temporary:
                run = Path(temporary).resolve() / "resume.seed"
                _write_attempt0(run)
                parent = run if location == "root" else run / location
                (parent / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    GATE.Node3GateError, "member set differs"
                ):
                    GATE._validate_preresume_seed_closed_world(run)

    def test_rotation_rejects_tampered_status_without_moving_anything(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runs_root = Path(temporary).resolve()
            completed = runs_root / "attempt.resume"
            seed = runs_root / "attempt.resume.seed"
            retained = runs_root / "attempt.resume.cold-a"
            _write_attempt0(completed)
            _write_attempt0(seed)
            (seed / "status.json").write_text(
                json.dumps({"status": "tampered", "optimizer_step": 0}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(GATE.Node3GateError, "bytes differ"):
                GATE.rotate_completed_resume_with_seed(
                    completed=completed,
                    seed=seed,
                    retained=retained,
                    runs_root=runs_root,
                )

            self.assertTrue(completed.is_dir())
            self.assertTrue(seed.is_dir())
            self.assertFalse(retained.exists())

    def test_rotation_rejects_existing_retention_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            completed = root / "resume"
            seed = root / "resume.seed"
            retained = root / "resume.cold-a"
            _write_attempt0(completed)
            _write_attempt0(seed)
            retained.mkdir()
            with self.assertRaisesRegex(GATE.Node3GateError, "already exists"):
                GATE.rotate_completed_resume_with_seed(
                    completed=completed,
                    seed=seed,
                    retained=retained,
                    runs_root=root,
                )

    def test_second_rename_failure_rolls_completed_run_back(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            completed = root / "resume"
            seed = root / "resume.seed"
            retained = root / "resume.cold-a"
            _write_attempt0(completed)
            _write_attempt0(seed)
            (completed / "DONE.json").write_text("{}\n", encoding="utf-8")
            original_rename = Path.rename

            def fail_seed_rename(path, target):
                if path == seed:
                    raise OSError("injected second rename failure")
                return original_rename(path, target)

            with mock.patch.object(Path, "rename", new=fail_seed_rename):
                with self.assertRaisesRegex(OSError, "injected"):
                    GATE.rotate_completed_resume_with_seed(
                        completed=completed,
                        seed=seed,
                        retained=retained,
                        runs_root=root,
                    )

            self.assertTrue((completed / "DONE.json").is_file())
            self.assertTrue(seed.is_dir())
            self.assertFalse(retained.exists())


class LifecycleSourceContractTest(unittest.TestCase):
    def test_b05_uses_exact_cold_cold_and_strict_warm_cold_consumers(self) -> None:
        source = GATE_PATH.read_text(encoding="utf-8")
        self.assertIn('"copy_preresume_seed"', source)
        self.assertIn('"latest_step1_cold_resume_a"', source)
        self.assertIn('"latest_step1_cold_resume_b"', source)
        self.assertIn('"cold_resume_checkpoint_exact_equivalence"', source)
        self.assertIn('"warm_cold_numerical_continuity"', source)
        self.assertIn('"analyze_node3_checkpoint_drift.py"', source)
        self.assertIn("_validate_resume_state_restore(cold_a_step2)", source)
        self.assertIn("_validate_resume_state_restore(resumed_step2)", source)
        self.assertIn("_compare_cold_step2_metrics_exact", source)
        self.assertIn('Path("status.json")', source)
        self.assertIn(
            '"b05_policy_schema_version": B05_LIFECYCLE_SCHEMA_VERSION', source
        )
        self.assertLess(
            source.index('active_gate = "B05_resume_equivalence"'),
            source.index('"cold_resume_a_verify_before_rotation"'),
        )
        self.assertNotIn("torch.allclose", source)


if __name__ == "__main__":
    unittest.main()
