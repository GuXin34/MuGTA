"""CPU/Torch tests for state-level node-3 resume equivalence."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_comparator():
    path = ROOT / "scripts" / "compare_node3_checkpoints.py"
    spec = importlib.util.spec_from_file_location("node3_state_comparator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import checkpoint comparator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COMPARE = _load_comparator()


def _sha256(path: Path) -> str:
    return COMPARE.sha256_file(path)


def _rng(rank: int):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(100 + rank)
    return {
        "python": (3, (rank, rank + 1, rank + 2), None),
        "torch_cpu": torch.randint(0, 255, (16,), dtype=torch.uint8, generator=generator),
        "torch_cuda_local": torch.randint(
            0, 255, (16,), dtype=torch.uint8, generator=generator
        ),
    }


def _payload(step: int, *, mutation: float = 0.0):
    student = {
        "weight": torch.tensor([[1.0 + step + mutation, 2.0]], dtype=torch.float32),
        "counter": torch.tensor(step, dtype=torch.int64),
    }
    optimizer = {
        "state": {
            0: {
                "step": torch.tensor(float(step)),
                "exp_avg": torch.tensor([0.1 * step + mutation, 0.2]),
                "exp_avg_sq": torch.tensor([0.01, 0.02]),
            }
        },
        "param_groups": [
            {
                "lr": 3.0e-6 * min(step + 1, 50) / 50,
                "betas": (0.9, 0.95),
                "eps": 1.0e-8,
                "weight_decay": 0.0,
                "params": [0],
            }
        ],
    }
    return {
        "metadata": {
            "schema_version": COMPARE.CHECKPOINT_SCHEMA_VERSION,
            "config_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "student_checkpoint_sha256": "c" * 64,
            "teacher_checkpoint_sha256": "c" * 64,
            "teacher_state_sha256_initial": "d" * 64,
            "ddp_reducer_identity_sha256": "e" * 64,
            "optimizer_step": step,
            "global_microstep": 4 * step,
        },
        "student_state": student,
        "optimizer_state": optimizer,
        "scheduler_state": {
            "type": "linear_warmup_then_constant",
            "warmup_optimizer_steps": 50,
            "completed_optimizer_steps": step,
            "learning_rate": optimizer["param_groups"][0]["lr"],
        },
        "rng_state_by_rank": [_rng(rank) for rank in range(8)],
    }


def _write_checkpoint(run: Path, step: int, *, mutation: float = 0.0) -> None:
    directory = run / "checkpoints" / "step-{:05d}".format(step)
    directory.mkdir(parents=True)
    checkpoint = directory / "checkpoint.pt"
    torch.save(_payload(step, mutation=mutation), checkpoint)
    (directory / "SHA256.json").write_text(
        json.dumps(
            {
                "path": "checkpoint.pt",
                "optimizer_step": step,
                "sha256": _sha256(checkpoint),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


class CanonicalStateHashTest(unittest.TestCase):
    def test_mapping_order_is_irrelevant_but_tensor_change_is_detected(self) -> None:
        first = {"b": [1, 2], "a": torch.tensor([1.0, 2.0])}
        second = {"a": torch.tensor([1.0, 2.0]), "b": [1, 2]}
        self.assertEqual(
            COMPARE.canonical_state_sha256(first),
            COMPARE.canonical_state_sha256(second),
        )
        second["a"][0] += 1.0
        self.assertNotEqual(
            COMPARE.canonical_state_sha256(first),
            COMPARE.canonical_state_sha256(second),
        )

    def test_tensor_layout_is_part_of_the_canonical_identity(self) -> None:
        contiguous = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        noncontiguous = contiguous.t().contiguous().t()
        self.assertTrue(torch.equal(contiguous, noncontiguous))
        self.assertNotEqual(contiguous.stride(), noncontiguous.stride())
        self.assertNotEqual(
            COMPARE.canonical_state_sha256({"value": contiguous}),
            COMPARE.canonical_state_sha256({"value": noncontiguous}),
        )

        offset_zero = torch.tensor([1.0, 2.0, 3.0, 4.0])
        offset_one = torch.tensor([99.0, 1.0, 2.0, 3.0, 4.0])[1:]
        self.assertTrue(torch.equal(offset_zero, offset_one))
        self.assertEqual(offset_zero.shape, offset_one.shape)
        self.assertEqual(offset_zero.stride(), offset_one.stride())
        self.assertEqual(offset_zero.storage_offset(), 0)
        self.assertEqual(offset_one.storage_offset(), 1)
        self.assertNotEqual(
            COMPARE.canonical_state_sha256({"value": offset_zero}),
            COMPARE.canonical_state_sha256({"value": offset_one}),
        )

    def test_sparse_tensor_layout_fails_closed(self) -> None:
        sparse = torch.sparse_coo_tensor(
            torch.tensor([[0]], dtype=torch.int64),
            torch.tensor([1.0]),
            size=(2,),
        )
        with self.assertRaisesRegex(
            COMPARE.StateEquivalenceError, "layout must be torch.strided"
        ):
            COMPARE.canonical_state_sha256({"value": sparse})

    def test_paired_nonfinite_tensors_fail_closed(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaisesRegex(
                COMPARE.StateEquivalenceError, "NaN/Inf"
            ):
                COMPARE.canonical_state_sha256(
                    {"value": torch.tensor([value], dtype=torch.float32)}
                )


class CheckpointEquivalenceTest(unittest.TestCase):
    def test_mmap_loader_passes_plain_string_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            _write_checkpoint(run, 1)
            observed = {}

            def torch_21_load(
                f, map_location=None, weights_only=None, mmap=None
            ):
                observed.update(
                    {
                        "filename": f,
                        "map_location": map_location,
                        "weights_only": weights_only,
                        "mmap": mmap,
                    }
                )
                if type(f) is not str:
                    raise ValueError(
                        "f must be a string filename in order to use mmap argument"
                    )
                return _payload(1)

            with mock.patch.object(COMPARE.torch, "load", new=torch_21_load):
                fingerprint = COMPARE.fingerprint_checkpoint(run, 1)

            self.assertIs(type(observed["filename"]), str)
            self.assertEqual(observed["map_location"], "cpu")
            self.assertIs(observed["weights_only"], False)
            self.assertIs(observed["mmap"], True)
            self.assertEqual(fingerprint["step"], 1)

    def test_steps_one_and_two_compare_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            uninterrupted = root / "uninterrupted"
            resumed = root / "resumed"
            for run in (uninterrupted, resumed):
                for step in (1, 2):
                    _write_checkpoint(run, step)
            result = COMPARE.compare_runs(uninterrupted, resumed)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(set(result["comparisons"]), {"step_1", "step_2"})
            self.assertTrue(
                result["comparisons"]["step_2"]["exact_fields"][
                    "optimizer_state_sha256"
                ]
            )
            self.assertEqual(
                len(
                    result["comparisons"]["step_2"]["resumed"][
                        "rng_state_per_rank_sha256"
                    ]
                ),
                8,
            )

    def test_student_optimizer_scheduler_rng_and_ddp_tamper_fail_closed(self) -> None:
        for target in ("student", "optimizer", "scheduler", "rng", "ddp_identity"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                uninterrupted = root / "uninterrupted"
                resumed = root / "resumed"
                for run in (uninterrupted, resumed):
                    for step in (1, 2):
                        _write_checkpoint(run, step)
                directory = resumed / "checkpoints" / "step-00002"
                checkpoint = directory / "checkpoint.pt"
                payload = torch.load(checkpoint, map_location="cpu")
                if target == "student":
                    payload["student_state"]["weight"][0, 0] += 1.0
                elif target == "optimizer":
                    payload["optimizer_state"]["state"][0]["exp_avg"][0] += 1.0
                elif target == "scheduler":
                    payload["scheduler_state"]["learning_rate"] += 1.0e-7
                elif target == "rng":
                    payload["rng_state_by_rank"][3]["torch_cpu"][0] ^= 1
                else:
                    payload["metadata"]["ddp_reducer_identity_sha256"] = "f" * 64
                torch.save(payload, checkpoint)
                sidecar = directory / "SHA256.json"
                sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
                sidecar_payload["sha256"] = _sha256(checkpoint)
                sidecar.write_text(
                    json.dumps(sidecar_payload, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    COMPARE.StateEquivalenceError, "state equivalence failed"
                ):
                    COMPARE.compare_runs(uninterrupted, resumed)


if __name__ == "__main__":
    unittest.main()
