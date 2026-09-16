"""Model-free source guard for Torch 2.1 mmap filename compatibility."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError("expected exactly one function named {!r}".format(name))
    return matches[0]


class TorchLoadMmapPathCompatibilityTest(unittest.TestCase):
    def _assert_compatible_loader(
        self,
        relative_path: str,
        function_name: str,
        path_variable: str,
        options_variable: str,
    ) -> None:
        function = _function(ROOT / relative_path, function_name)
        loads = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr == "load"
        ]
        self.assertEqual(len(loads), 1)
        load = loads[0]
        self.assertEqual(len(load.args), 1)
        filename = load.args[0]
        self.assertIsInstance(filename, ast.Call)
        self.assertIsInstance(filename.func, ast.Name)
        self.assertEqual(filename.func.id, "str")
        self.assertEqual(len(filename.args), 1)
        self.assertIsInstance(filename.args[0], ast.Name)
        self.assertEqual(filename.args[0].id, path_variable)
        self.assertTrue(
            any(
                keyword.arg is None
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == options_variable
                for keyword in load.keywords
            )
        )

        mmap_assignments = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == options_variable
            and isinstance(node.targets[0].slice, ast.Constant)
            and node.targets[0].slice.value == "mmap"
            and isinstance(node.value, ast.Constant)
            and node.value.value is True
        ]
        self.assertEqual(len(mmap_assignments), 1)

    def test_state_comparator_uses_string_filename_with_mmap(self) -> None:
        self._assert_compatible_loader(
            "scripts/compare_node3_checkpoints.py",
            "fingerprint_checkpoint",
            "checkpoint",
            "options",
        )

    def test_stage1_metadata_consumer_uses_string_filename_with_mmap(self) -> None:
        self._assert_compatible_loader(
            "scripts/train_stage1.py",
            "_load_checkpoint_metadata",
            "checkpoint_path",
            "load_options",
        )


if __name__ == "__main__":
    unittest.main()
