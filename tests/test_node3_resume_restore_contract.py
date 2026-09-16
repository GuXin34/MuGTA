"""CPU tests for the exact pre-forward resume-state restore audit."""

from __future__ import annotations

import copy
import importlib.util
import inspect
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch import nn


def _load_runner_module():
    path = ROOT / "scripts" / "train_stage1.py"
    spec = importlib.util.spec_from_file_location("train_stage1_resume_restore", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import train_stage1.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner_module()


def _model_and_optimizer():
    model = nn.Sequential(nn.Linear(4, 3), nn.LayerNorm(3))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-6,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    return model, optimizer


def _trained_checkpoint_payload():
    torch.manual_seed(2027)
    model, optimizer = _model_and_optimizer()
    inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 8.0
    model(inputs).square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    payload = {
        "metadata": {
            "schema_version": RUNNER.CHECKPOINT_SCHEMA_VERSION,
            "optimizer_step": 1,
            "global_microstep": 4,
        },
        "student_state": copy.deepcopy(model.state_dict()),
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "scheduler_state": {
            "type": "linear_warmup_then_constant",
            "warmup_optimizer_steps": 50,
            "completed_optimizer_steps": 1,
            "learning_rate": 3.0e-6,
        },
    }
    return payload


def _restored_live_state(payload):
    model, optimizer = _model_and_optimizer()
    model.load_state_dict(payload["student_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    return model, optimizer


class CanonicalFiniteTensorTreeTests(unittest.TestCase):
    def test_mapping_order_is_irrelevant_but_container_and_leaf_types_are_not(self):
        tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        first = {
            "tensor": tensor,
            7: [None, True, -4, 0.25, ("x", False)],
        }
        second = {
            7: [None, True, -4, 0.25, ("x", False)],
            "tensor": tensor.clone(),
        }
        self.assertEqual(
            RUNNER.canonical_finite_tensor_tree_sha256(first),
            RUNNER.canonical_finite_tensor_tree_sha256(second),
        )
        self.assertNotEqual(
            RUNNER.canonical_finite_tensor_tree_sha256([1, 2]),
            RUNNER.canonical_finite_tensor_tree_sha256((1, 2)),
        )
        self.assertNotEqual(
            RUNNER.canonical_finite_tensor_tree_sha256({"x": True}),
            RUNNER.canonical_finite_tensor_tree_sha256({"x": 1}),
        )

    def test_dtype_shape_stride_storage_offset_and_bytes_are_identity(self):
        base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        transposed = base.t()
        same_values_contiguous = transposed.contiguous()
        self.assertEqual(transposed.shape, same_values_contiguous.shape)
        self.assertTrue(torch.equal(transposed, same_values_contiguous))
        self.assertNotEqual(
            RUNNER.canonical_finite_tensor_tree_sha256(transposed),
            RUNNER.canonical_finite_tensor_tree_sha256(same_values_contiguous),
        )

        offset_view = torch.arange(10, dtype=torch.int64)[2:6]
        offset_zero = offset_view.clone()
        self.assertEqual(offset_view.stride(), offset_zero.stride())
        self.assertTrue(torch.equal(offset_view, offset_zero))
        self.assertNotEqual(offset_view.storage_offset(), offset_zero.storage_offset())
        self.assertNotEqual(
            RUNNER.canonical_finite_tensor_tree_sha256(offset_view),
            RUNNER.canonical_finite_tensor_tree_sha256(offset_zero),
        )

        self.assertNotEqual(
            RUNNER.canonical_finite_tensor_tree_sha256(
                torch.tensor([1], dtype=torch.int32)
            ),
            RUNNER.canonical_finite_tensor_tree_sha256(
                torch.tensor([1], dtype=torch.int64)
            ),
        )
        self.assertNotEqual(
            RUNNER.canonical_finite_tensor_tree_sha256(torch.tensor([1, 2])),
            RUNNER.canonical_finite_tensor_tree_sha256(torch.tensor([[1, 2]])),
        )
        changed = base.clone()
        changed[0, 0] = 99.0
        self.assertNotEqual(
            RUNNER.canonical_finite_tensor_tree_sha256(base),
            RUNNER.canonical_finite_tensor_tree_sha256(changed),
        )

        source = inspect.getsource(RUNNER._update_resume_state_hash)
        self.assertLess(
            source.index('"stride": list(logical.stride())'),
            source.index('source = logical.to(device="cpu")'),
        )
        self.assertLess(
            source.index('"storage_offset": int(logical.storage_offset())'),
            source.index('source = logical.to(device="cpu")'),
        )

    def test_nonfinite_layout_and_unapproved_types_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            RUNNER.canonical_finite_tensor_tree_sha256(torch.tensor([float("nan")]))
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            RUNNER.canonical_finite_tensor_tree_sha256({"lr": float("inf")})
        with self.assertRaisesRegex(ValueError, "torch.strided"):
            RUNNER.canonical_finite_tensor_tree_sha256(
                torch.sparse_coo_tensor(
                    torch.tensor([[0], [1]]), torch.tensor([1.0]), (2, 2)
                )
            )
        with self.assertRaisesRegex(TypeError, "exact str or int"):
            RUNNER.canonical_finite_tensor_tree_sha256({True: 1})
        with self.assertRaisesRegex(TypeError, "unsupported"):
            RUNNER.canonical_finite_tensor_tree_sha256({"bad": {1, 2}})


class ResumeRestoreAuditTests(unittest.TestCase):
    def test_loaded_student_and_optimizer_are_exact_and_rank_local(self):
        payload = _trained_checkpoint_payload()
        model, optimizer = _restored_live_state(payload)
        audit = RUNNER.audit_resume_state_restore(
            checkpoint_payload=payload,
            student_lm=model,
            optimizer=optimizer,
            rank=3,
        )
        self.assertEqual(
            audit["schema_version"], RUNNER.RESUME_STATE_RESTORE_SCHEMA_VERSION
        )
        self.assertEqual(audit["rank"], 3)
        self.assertTrue(audit["captured_after_checkpoint_load"])
        self.assertTrue(audit["captured_before_next_forward"])
        self.assertTrue(audit["exact"])
        self.assertEqual(
            audit["exact_fields"],
            {
                "optimizer_state_sha256": True,
                "student_state_sha256": True,
            },
        )
        self.assertEqual(audit["checkpoint"], audit["live"])

    def test_student_or_optimizer_bit_drift_is_rejected_without_tolerance(self):
        payload = _trained_checkpoint_payload()
        model, optimizer = _restored_live_state(payload)
        with torch.no_grad():
            next(model.parameters()).view(-1)[0].add_(torch.finfo(torch.float32).eps)
        with self.assertRaisesRegex(RuntimeError, "student_state_sha256"):
            RUNNER.audit_resume_state_restore(
                checkpoint_payload=payload,
                student_lm=model,
                optimizer=optimizer,
                rank=0,
            )

        model, optimizer = _restored_live_state(payload)
        checkpoint_hashes = {
            "student_state_sha256": RUNNER.canonical_finite_tensor_tree_sha256(
                payload["student_state"]
            ),
            "optimizer_state_sha256": RUNNER.canonical_finite_tensor_tree_sha256(
                payload["optimizer_state"]
            ),
        }
        first_parameter = next(iter(optimizer.state))
        optimizer.state[first_parameter]["exp_avg"].view(-1)[0].add_(1.0)
        with self.assertRaisesRegex(
            RuntimeError, "payload state changed|optimizer_state_sha256"
        ):
            RUNNER.audit_resume_state_restore(
                checkpoint_payload=payload,
                student_lm=model,
                optimizer=optimizer,
                rank=0,
                checkpoint_hashes_before_load=checkpoint_hashes,
            )

    def test_load_resume_returns_rng_and_state_audits_before_forward(self):
        payload = _trained_checkpoint_payload()
        rng_state = {
            "python": random.Random(19).getstate(),
            "torch_cpu": torch.arange(16, dtype=torch.uint8),
            "torch_cuda_local": torch.arange(16, 32, dtype=torch.uint8),
        }
        payload["rng_state_by_rank"] = [rng_state]

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary) / "step-00001"
            checkpoint_dir.mkdir()
            checkpoint = checkpoint_dir / "checkpoint.pt"
            torch.save(payload, checkpoint)
            sidecar = {
                "path": "checkpoint.pt",
                "optimizer_step": 1,
                "sha256": RUNNER.sha256_file(checkpoint),
            }
            (checkpoint_dir / "SHA256.json").write_text(
                json.dumps(sidecar), encoding="utf-8"
            )

            model, optimizer = _model_and_optimizer()
            restored_rng = {}

            def set_rng(state):
                restored_rng["value"] = state

            def get_rng():
                return restored_rng["value"]

            with mock.patch.object(
                RUNNER.dist, "get_world_size", return_value=1
            ), mock.patch.object(RUNNER, "set_rng_state", side_effect=set_rng), mock.patch.object(
                RUNNER, "get_rng_state", side_effect=get_rng
            ):
                step, microstep, rng_audit, state_audit = RUNNER.load_resume(
                    path=checkpoint_dir,
                    student_lm=model,
                    optimizer=optimizer,
                    expected_metadata={
                        "schema_version": RUNNER.CHECKPOINT_SCHEMA_VERSION,
                        "optimizer_step": 0,
                        "global_microstep": 0,
                    },
                    device=torch.device("cpu"),
                    rank=0,
                )

        self.assertEqual((step, microstep), (1, 4))
        self.assertTrue(rng_audit["exact"])
        self.assertTrue(state_audit["exact"])
        self.assertEqual(state_audit["rank"], 0)

        load_source = inspect.getsource(RUNNER.load_resume)
        self.assertLess(
            load_source.index("state_restore_audit = audit_resume_state_restore"),
            load_source.index("return optimizer_step"),
        )
        training_source = inspect.getsource(RUNNER.run_training)
        self.assertLess(
            training_source.index(") = load_resume("),
            training_source.index("codes = student_lm.generate("),
        )
        self.assertIn(
            '"resume_state_restore": resume_state_restore_local', training_source
        )


if __name__ == "__main__":
    unittest.main()
