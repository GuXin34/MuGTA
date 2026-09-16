"""Model-free guards for the patch05 fixed DDP resume topology."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "scripts" / "train_stage1.py"
COMPARE_PATH = ROOT / "scripts" / "compare_node3_checkpoints.py"
TRAIN_UTILS_PATH = ROOT / "src" / "ptc_opd" / "train_utils.py"
GATE_PATH = ROOT / "scripts" / "run_node3_gate.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _module_constants(tree: ast.Module):
    values = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
        ):
            values[node.targets[0].id] = node.value.value
    return values


class FixedDDPResumeContractTest(unittest.TestCase):
    def test_ddp_constructor_uses_the_frozen_no_rebuild_policy(self) -> None:
        tree = _tree(TRAIN_PATH)
        constants = _module_constants(tree)
        self.assertEqual(constants["DDP_BUCKET_CAP_MB"], 25)
        self.assertIs(constants["DDP_FIND_UNUSED_PARAMETERS"], True)
        self.assertIs(constants["DDP_STATIC_GRAPH"], False)
        self.assertIs(constants["DDP_GRADIENT_AS_BUCKET_VIEW"], False)

        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "DDP"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
        expected = {
            "bucket_cap_mb": "DDP_BUCKET_CAP_MB",
            "find_unused_parameters": "DDP_FIND_UNUSED_PARAMETERS",
            "gradient_as_bucket_view": "DDP_GRADIENT_AS_BUCKET_VIEW",
            "static_graph": "DDP_STATIC_GRAPH",
        }
        for keyword, constant in expected.items():
            self.assertIn(keyword, keywords)
            self.assertIsInstance(keywords[keyword], ast.Name)
            self.assertEqual(keywords[keyword].id, constant)

    def test_training_records_bucket_and_rollout_identities(self) -> None:
        source = TRAIN_PATH.read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count("ddp_reducer_audit(ddp_scorer)"), 2)
        self.assertIn('"rollout_codes_sha256": tensor_sha256(codes)', source)
        self.assertIn('"all_trainable_gradients_present": True', source)
        self.assertIn('"has_rebuilt_buckets": False', source)
        self.assertIn('"torch_version": str(torch.__version__)', source)
        self.assertIn('"torch_cuda_runtime": str(torch.version.cuda)', source)

        gate_source = GATE_PATH.read_text(encoding="utf-8")
        self.assertIn('"torch_version": "2.1.0+cu121"', gate_source)
        self.assertIn('"torch_cuda_runtime": "12.1"', gate_source)

    def test_checkpoint_v3_binds_reducer_identity_through_terminal_seal(self) -> None:
        train_source = TRAIN_PATH.read_text(encoding="utf-8")
        utils_source = TRAIN_UTILS_PATH.read_text(encoding="utf-8")
        compare_source = COMPARE_PATH.read_text(encoding="utf-8")
        self.assertIn('ptc-opd-stage1-checkpoint-v3', utils_source)
        self.assertIn('ptc-opd-stage1-run-v3', utils_source)
        self.assertIn('ptc-opd-stage1-seal-v5', train_source)
        self.assertIn('ptc-opd-stage1-done-v5', train_source)
        self.assertIn('ddp_reducer_identity_sha256', utils_source)
        self.assertIn('metadata_ddp_reducer_identity_sha256', train_source)
        self.assertIn('ddp_reducer_identity_sha256=manifest_identity[', train_source)
        self.assertIn('ptc-opd-stage1-checkpoint-v3', compare_source)
        self.assertIn('metadata lacks DDP reducer identity', compare_source)

    def test_checkpoint_comparator_remains_exact_and_rejects_nonfinite(self) -> None:
        tree = _tree(COMPARE_PATH)
        source = COMPARE_PATH.read_text(encoding="utf-8")
        allclose_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "allclose"
        ]
        self.assertEqual(allclose_calls, [])
        self.assertIn('ptc-opd-node3-state-equivalence-v2', source)
        self.assertIn('"student_state_sha256"', source)
        self.assertIn('"optimizer_state_sha256"', source)
        self.assertIn("torch.isfinite(source)", source)
        self.assertIn("checkpoint tensor contains NaN/Inf", source)


if __name__ == "__main__":
    unittest.main()
