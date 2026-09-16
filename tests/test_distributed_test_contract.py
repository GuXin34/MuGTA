"""Model-free source guard for the real-DDP two-rank regression."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests" / "test_distributed.py"


def _function(name: str) -> ast.FunctionDef:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError("expected exactly one function named {!r}".format(name))
    return matches[0]


class DistributedTestContract(unittest.TestCase):
    def test_worker_uses_real_ddp_without_manual_gradient_all_reduce(self) -> None:
        worker = _function("_worker")
        ddp_calls = [
            node
            for node in ast.walk(worker)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "DDP"
        ]
        manual_gradient_reductions = [
            node
            for node in ast.walk(worker)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "dist"
            and node.func.attr == "all_reduce"
        ]
        loss_calls = [
            node
            for node in ast.walk(worker)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ptc_opd_loss"
        ]
        ddp_gradients = [
            node
            for node in ast.walk(worker)
            if isinstance(node, ast.Attribute)
            and node.attr == "grad"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "student"
            and isinstance(node.value.value, ast.Attribute)
            and node.value.value.attr == "module"
            and isinstance(node.value.value.value, ast.Name)
            and node.value.value.value.id == "ddp_student"
        ]
        self.assertEqual(len(ddp_calls), 1)
        self.assertEqual(manual_gradient_reductions, [])
        self.assertEqual(len(loss_calls), 1)
        self.assertIsInstance(loss_calls[0].args[0], ast.Call)
        self.assertIsInstance(loss_calls[0].args[0].func, ast.Name)
        self.assertEqual(loss_calls[0].args[0].func.id, "ddp_student")
        self.assertEqual(len(ddp_gradients), 1)

    def test_reference_compares_the_complete_shared_gradient(self) -> None:
        test = _function(
            "test_two_rank_unequal_counts_matches_concatenated_reference"
        )
        skip_calls = [
            node
            for node in ast.walk(test)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "skipTest"
        ]
        matches = []
        for node in ast.walk(test):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "assert_close"
                and len(node.args) >= 2
            ):
                continue
            left, right = node.args[:2]
            if (
                isinstance(left, ast.Subscript)
                and isinstance(left.value, ast.Name)
                and left.value.id == "saved"
                and isinstance(left.slice, ast.Constant)
                and left.slice.value == "gradient"
                and isinstance(right, ast.Attribute)
                and isinstance(right.value, ast.Name)
                and right.value.id == "student"
                and right.attr == "grad"
            ):
                matches.append(node)
        self.assertEqual(skip_calls, [])
        self.assertEqual(len(matches), 1)


if __name__ == "__main__":
    unittest.main()
