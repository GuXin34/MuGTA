"""Focused CPU tests for the warm/cold numerical-continuity consumer."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_analyzer():
    path = ROOT / "scripts" / "analyze_node3_checkpoint_drift.py"
    spec = importlib.util.spec_from_file_location("node3_checkpoint_drift", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import checkpoint drift analyzer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANALYZE = _load_analyzer()


def _rng(rank: int):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(700 + rank)
    return {
        "python": (3, (rank, rank + 1, rank + 2), None),
        "torch_cpu": torch.randint(0, 255, (16,), dtype=torch.uint8, generator=generator),
        "torch_cuda_local": torch.randint(
            0, 255, (16,), dtype=torch.uint8, generator=generator
        ),
    }


def _payload(step: int, *, config: str = "a"):
    size = 100_000
    exp_avg = torch.full((size,), 5.0e-4 if step == 1 else 1.0e-3)
    exp_avg_sq = torch.full((size,), 5.0e-7 if step == 1 else 1.0e-6)
    learning_rate = 6.0e-8 if step == 1 else 1.2e-7
    return {
        "metadata": {
            "schema_version": ANALYZE.CHECKPOINT_SCHEMA_VERSION,
            "config_sha256": config * 64,
            "manifest_sha256": "b" * 64,
            "student_checkpoint_sha256": "c" * 64,
            "teacher_checkpoint_sha256": "c" * 64,
            "teacher_state_sha256_initial": "d" * 64,
            "ddp_reducer_identity_sha256": "e" * 64,
            "optimizer_step": step,
            "global_microstep": 4 * step,
        },
        "student_state": {
            "weight": torch.full((size,), 1.0e-2),
            "counter": torch.tensor(step, dtype=torch.int64),
        },
        "optimizer_state": {
            "state": {
                0: {
                    "step": torch.tensor(float(step)),
                    "exp_avg": exp_avg,
                    "exp_avg_sq": exp_avg_sq,
                }
            },
            "param_groups": [
                {
                    "lr": learning_rate,
                    "betas": (0.9, 0.95),
                    "eps": 1.0e-8,
                    "weight_decay": 0.0,
                    "params": [0],
                }
            ],
        },
        "scheduler_state": {
            "type": "linear_warmup_then_constant",
            "warmup_optimizer_steps": 50,
            "completed_optimizer_steps": step,
            "learning_rate": learning_rate,
        },
        "rng_state_by_rank": [_rng(rank) for rank in range(8)],
    }


def _one_ulp_up(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(tensor, torch.full_like(tensor, float("inf")))


def _bounded_cold_step2():
    payload = _payload(2, config="f")
    payload["student_state"]["weight"][0] = _one_ulp_up(
        payload["student_state"]["weight"][0]
    )
    for field in ("exp_avg", "exp_avg_sq"):
        tensor = payload["optimizer_state"]["state"][0][field]
        tensor[:10_000] = _one_ulp_up(tensor[:10_000])
    return payload


def _write_checkpoint(run: Path, step: int, payload) -> None:
    directory = run / "checkpoints" / "step-{:05d}".format(step)
    directory.mkdir(parents=True)
    checkpoint = directory / "checkpoint.pt"
    torch.save(payload, checkpoint)
    (directory / "SHA256.json").write_text(
        json.dumps(
            {
                "path": "checkpoint.pt",
                "optimizer_step": step,
                "sha256": ANALYZE.sha256_file(checkpoint),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _make_runs(root: Path, cold_step2=None):
    warm = root / "warm"
    cold = root / "cold"
    step1 = _payload(1)
    _write_checkpoint(warm, 1, copy.deepcopy(step1))
    cold_step1 = copy.deepcopy(step1)
    cold_step1["metadata"]["config_sha256"] = "f" * 64
    _write_checkpoint(cold, 1, cold_step1)
    _write_checkpoint(warm, 2, _payload(2))
    _write_checkpoint(
        cold,
        2,
        _bounded_cold_step2() if cold_step2 is None else cold_step2,
    )
    return warm, cold


class UlpMetricTest(unittest.TestCase):
    def test_adjacent_float32_values_are_one_ulp_apart(self) -> None:
        left = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float32)
        right = _one_ulp_up(left)
        self.assertEqual(ANALYZE.max_ulp_distance(left, right), 1)

    def test_signed_zero_has_zero_ulp_distance(self) -> None:
        left = torch.tensor([-0.0], dtype=torch.float32)
        right = torch.tensor([0.0], dtype=torch.float32)
        self.assertEqual(ANALYZE.max_ulp_distance(left, right), 0)

    def test_negative_min_subnormal_is_one_step_below_zero(self) -> None:
        zero = torch.tensor([0.0], dtype=torch.float32)
        negative_min = torch.nextafter(
            zero, torch.tensor([float("-inf")], dtype=torch.float32)
        )
        self.assertEqual(ANALYZE.max_ulp_distance(negative_min, zero), 1)

    def test_near_zero_subnormal_is_governed_by_abs_not_ulp(self) -> None:
        left = torch.tensor([0.0], dtype=torch.float32)
        right = torch.tensor([torch.finfo(torch.float32).tiny / 2], dtype=torch.float32)
        raw_ulp = ANALYZE.max_ulp_distance(left, right)
        floored_ulp = ANALYZE.max_ulp_distance(left, right, 2.0 ** -40)
        self.assertGreater(raw_ulp, ANALYZE.FROZEN_CAPS["student"]["max_ulp"])
        self.assertEqual(floored_ulp, 0)
        metrics = ANALYZE._tensor_metrics(left, right, "near_zero", "student")
        self.assertEqual(metrics["below_ulp_floor_changed_count"], 1)
        self.assertLessEqual(
            metrics["below_ulp_floor_max_abs"],
            ANALYZE.FROZEN_CAPS["student"]["max_below_ulp_floor_abs"],
        )

    def test_small_moment_is_governed_by_abs_not_misleading_ulp(self) -> None:
        left = torch.tensor([0.0], dtype=torch.float32)
        bounded = torch.tensor([1.0e-10], dtype=torch.float32)
        raw_ulp = ANALYZE.max_ulp_distance(left, bounded)
        metrics = ANALYZE._tensor_metrics(left, bounded, "moment", "exp_avg")
        self.assertGreater(raw_ulp, ANALYZE.FROZEN_CAPS["exp_avg"]["max_ulp"])
        self.assertEqual(metrics["max_ulp"], 0)
        self.assertLessEqual(
            metrics["below_ulp_floor_max_abs"],
            ANALYZE.FROZEN_CAPS["exp_avg"]["max_below_ulp_floor_abs"],
        )

        excessive = torch.tensor([3.0e-10], dtype=torch.float32)
        excessive_metrics = ANALYZE._tensor_metrics(
            left, excessive, "moment", "exp_avg"
        )
        aggregate = ANALYZE._aggregate_tensor_metrics(
            [excessive_metrics], "exp_avg"
        )
        violations = ANALYZE._cap_violations(
            "exp_avg", [excessive_metrics], aggregate
        )
        self.assertTrue(any("max_abs" in violation for violation in violations))


class WarmColdContinuityTest(unittest.TestCase):
    def test_bounded_ulp_drift_passes_with_per_tensor_and_derived_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            warm, cold = _make_runs(Path(temporary))
            result = ANALYZE.analyze_runs(warm, cold)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["decision"]["violation_count"], 0)
        self.assertEqual(result["step_1"]["status"], "exact")
        student = result["step_2"]["student"]
        self.assertEqual(student["aggregate"]["changed_count"], 1)
        self.assertEqual(student["aggregate"]["max_ulp"], 1)
        moments = result["step_2"]["optimizer"]
        self.assertEqual(moments["exp_avg"]["aggregate"]["changed_count"], 10_000)
        self.assertIn(
            "inferred_clipped_gradient", result["step_2"]["derived"]
        )
        self.assertIn(
            "adam_moment_update_direction", result["step_2"]["derived"]
        )

    def test_run_local_config_hash_is_the_only_metadata_exception(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            warm, cold = _make_runs(Path(temporary))
            result = ANALYZE.analyze_runs(warm, cold)
            self.assertEqual(result["status"], "passed")

            checkpoint = cold / "checkpoints" / "step-00002" / "checkpoint.pt"
            payload = torch.load(checkpoint, map_location="cpu")
            payload["metadata"]["manifest_sha256"] = "9" * 64
            torch.save(payload, checkpoint)
            sidecar = checkpoint.parent / "SHA256.json"
            sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
            sidecar_payload["sha256"] = ANALYZE.sha256_file(checkpoint)
            sidecar.write_text(json.dumps(sidecar_payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ANALYZE.ContinuityError, "metadata differs"):
                ANALYZE.analyze_runs(warm, cold)

    def test_step1_must_be_complete_payload_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            warm, cold = _make_runs(Path(temporary))
            checkpoint = cold / "checkpoints" / "step-00001" / "checkpoint.pt"
            payload = torch.load(checkpoint, map_location="cpu")
            payload["student_state"]["counter"] += 1
            torch.save(payload, checkpoint)
            sidecar = checkpoint.parent / "SHA256.json"
            sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
            sidecar_payload["sha256"] = ANALYZE.sha256_file(checkpoint)
            sidecar.write_text(json.dumps(sidecar_payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ANALYZE.ContinuityError, "step-1"):
                ANALYZE.analyze_runs(warm, cold)

    def test_large_student_corruption_is_rejected_by_frozen_caps(self) -> None:
        payload = _bounded_cold_step2()
        payload["student_state"]["weight"][0] += 1.0e-5
        with tempfile.TemporaryDirectory() as temporary:
            warm, cold = _make_runs(Path(temporary), payload)
            result = ANALYZE.analyze_runs(warm, cold)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(
            any("student_state" in value and "max_abs" in value for value in result["decision"]["violations"])
        )

    def test_optimizer_corruption_is_rejected_by_frozen_caps(self) -> None:
        payload = _bounded_cold_step2()
        payload["optimizer_state"]["state"][0]["exp_avg"][0] += 1.0e-6
        with tempfile.TemporaryDirectory() as temporary:
            warm, cold = _make_runs(Path(temporary), payload)
            result = ANALYZE.analyze_runs(warm, cold)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(
            any("exp_avg" in value and "max_abs" in value for value in result["decision"]["violations"])
        )

    def test_nonfinite_float_fails_closed(self) -> None:
        payload = _bounded_cold_step2()
        payload["student_state"]["weight"][3] = float("nan")
        with tempfile.TemporaryDirectory() as temporary:
            warm, cold = _make_runs(Path(temporary), payload)
            with self.assertRaisesRegex(ANALYZE.ContinuityError, "NaN/Inf"):
                ANALYZE.analyze_runs(warm, cold)

    def test_optimizer_param_groups_and_nonmoment_state_are_exact(self) -> None:
        for field in ("group", "step"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                payload = _bounded_cold_step2()
                if field == "group":
                    payload["optimizer_state"]["param_groups"][0]["eps"] = 2.0e-8
                else:
                    payload["optimizer_state"]["state"][0]["step"] += 1.0
                warm, cold = _make_runs(Path(temporary), payload)
                with self.assertRaisesRegex(
                    ANALYZE.ContinuityError, "param_groups|optimizer_state"
                ):
                    ANALYZE.analyze_runs(warm, cold)

    def test_cli_writes_failed_json_and_returns_nonzero(self) -> None:
        payload = _bounded_cold_step2()
        payload["student_state"]["weight"][0] += 1.0e-5
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            warm, cold = _make_runs(root, payload)
            output = root / "decision.json"
            return_code = ANALYZE.main(
                [
                    "--uninterrupted-run",
                    str(warm),
                    "--resumed-run",
                    str(cold),
                    "--output",
                    str(output),
                ]
            )
            result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(return_code, 1)
        self.assertEqual(result["status"], "failed")
        self.assertGreater(result["decision"]["violation_count"], 0)

    def test_source_has_no_generic_approximate_equality_call(self) -> None:
        source = (
            ROOT / "scripts" / "analyze_node3_checkpoint_drift.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("torch." + "all" + "close", source)


if __name__ == "__main__":
    unittest.main()
