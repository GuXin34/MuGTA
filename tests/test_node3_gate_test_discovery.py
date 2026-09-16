"""Model-free regressions for node-3 A05 diagnostics and A06 discovery."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def _load_gate():
    path = ROOT / "scripts" / "run_node3_gate.py"
    spec = importlib.util.spec_from_file_location("node3_patch03_gate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import run_node3_gate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GATE = _load_gate()


class Node3GatePatch03RegressionTest(unittest.TestCase):
    def test_a06_target_output_requires_three_real_passes_and_no_skip(self) -> None:
        passing_lines = [
            "{} ... ok".format(name) for name in GATE.DISTRIBUTED_TARGET_TESTS
        ]
        passing = "\n".join(
            passing_lines
            + [
                "",
                "----------------------------------------------------------------------",
                "Ran 3 tests in 0.123s",
                "",
                "OK",
                "",
            ]
        )
        result = GATE.validate_distributed_target_test_output("", passing)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["ran_test_count"], 3)
        self.assertEqual(result["skip_count"], 0)

        skipped = passing.replace(
            passing_lines[-1],
            "{} ... skipped 'network unavailable'".format(
                GATE.DISTRIBUTED_TARGET_TESTS[-1]
            ),
        ).replace("\nOK\n", "\nOK (skipped=1)\n")
        with self.assertRaisesRegex(GATE.Node3GateError, "did not execute and pass"):
            GATE.validate_distributed_target_test_output("", skipped)

        wrong_count = passing.replace("Ran 3 tests", "Ran 2 tests")
        with self.assertRaisesRegex(GATE.Node3GateError, "exactly 3 tests"):
            GATE.validate_distributed_target_test_output("", wrong_count)

        nonexact_ok = passing.replace("\nOK\n", "\nOK (skipped=1)\n")
        with self.assertRaisesRegex(GATE.Node3GateError, "exactly OK"):
            GATE.validate_distributed_target_test_output("", nonexact_ok)

    def test_a06_output_validation_precedes_the_eight_process_probe(self) -> None:
        source_path = ROOT / "scripts" / "run_node3_gate.py"
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"), filename=str(source_path)
        )
        execute = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "execute"
        )
        validation_calls = [
            node
            for node in ast.walk(execute)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "validate_distributed_target_test_output"
        ]
        probe_calls = [
            node
            for node in ast.walk(execute)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "eight_process_ratio"
        ]
        self.assertEqual(len(validation_calls), 1)
        self.assertEqual(len(probe_calls), 1)
        self.assertLess(validation_calls[0].lineno, probe_calls[0].lineno)

    def test_tampered_retained_decision_sidecar_is_named_in_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            decision = directory / "cfg_scale_decision.json"
            sidecar = directory / "cfg_scale_decision.sha256.json"
            decision.write_text(
                json.dumps(
                    {
                        "status": "selected",
                        "selected_cfg_scale": 5.0,
                        "generation_identity": {
                            "model_id": "facebook/musicgen-small"
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            sidecar.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(
                GATE, "RETAINED_SMALL_DECISION_SHA256", GATE.sha256_file(decision)
            ), mock.patch.object(
                GATE,
                "RETAINED_SMALL_DECISION_SIDECAR_SHA256",
                GATE.sha256_file(sidecar),
            ):
                GATE._verify_retained_small_decision(directory)
                sidecar.write_text('{"tampered":true}\n', encoding="utf-8")
                with self.assertRaisesRegex(
                    GATE.Node3GateError,
                    r"cfg_scale_decision\.sha256\.json sidecar differs from the Gate-1 pin",
                ):
                    GATE._verify_retained_small_decision(directory)

    def test_a06_discovery_bypasses_hostile_top_level_tests_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workpack = root / "workpack"
            local_tests = workpack / "tests"
            hostile_tests = root / "hostile_site" / "tests"
            local_tests.mkdir(parents=True)
            hostile_tests.mkdir(parents=True)
            hostile_tests.joinpath("__init__.py").write_text(
                'ORIGIN = "hostile-installed-package"\n', encoding="utf-8"
            )
            local_tests.joinpath("test_distributed.py").write_text(
                "import unittest\n"
                "\n"
                "class LocalDistributedTest(unittest.TestCase):\n"
                "    def test_local_file_wins(self):\n"
                "        self.assertTrue(True)\n",
                encoding="utf-8",
            )
            environment = dict(os.environ)
            environment.update(
                {
                    "PYTHONPATH": str(root / "hostile_site"),
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )

            dotted = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "tests.test_distributed",
                    "-v",
                ],
                cwd=str(workpack),
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(dotted.returncode, 0)
            self.assertIn(
                "No module named 'tests.test_distributed'",
                dotted.stdout + dotted.stderr,
            )

            command = GATE._distributed_target_test_command(
                Path(sys.executable), local_tests
            )
            self.assertEqual(
                command,
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    str(local_tests),
                    "-p",
                    "test_distributed.py",
                    "-v",
                ],
            )
            self.assertTrue(Path(command[5]).is_absolute())
            discovered = subprocess.run(
                command,
                cwd=str(workpack),
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(discovered.returncode, 0, discovered.stderr)
            self.assertIn("test_local_file_wins", discovered.stdout + discovered.stderr)
            self.assertIn("OK", discovered.stdout + discovered.stderr)


if __name__ == "__main__":
    unittest.main()
