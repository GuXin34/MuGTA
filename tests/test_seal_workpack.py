"""Tests for the closed-world portable workpack source seal."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "seal_workpack.py"
SPEC = importlib.util.spec_from_file_location("seal_workpack", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not import seal_workpack.py")
SEAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SEAL
SPEC.loader.exec_module(SEAL)


def make_fixture(root: Path) -> None:
    root_files = {
        ".gitignore": "__pycache__/\n",
        "BASELINE_PROVENANCE.md": "baseline\n",
        "CHANGELOG.md": "changes\n",
        "README.md": "readme\n",
        "pyproject.toml": "[project]\nname='fixture'\n",
        "LICENSE": "fixture license\n",
    }
    for name, payload in root_files.items():
        (root / name).write_text(payload, encoding="utf-8")
    for dirname in SEAL.MANAGED_DIRECTORIES:
        (root / dirname).mkdir()
        (root / dirname / "managed.txt").write_text(
            "{}\n".format(dirname), encoding="utf-8"
        )


class WorkpackSealTest(unittest.TestCase):
    def test_generate_is_sorted_deterministic_and_verify_is_closed_world(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture(root)
            # Noise and large products must never enter the source seal.
            (root / ".DS_Store").write_bytes(b"noise")
            (root / "scripts" / "__pycache__").mkdir()
            (root / "scripts" / "__pycache__" / "x.pyc").write_bytes(b"cache")
            (root / "docs" / "tmp").mkdir()
            (root / "docs" / "tmp" / "generated.txt").write_text(
                "generated", encoding="utf-8"
            )
            for dirname in (
                "manifests",
                "vendor",
                "artifacts",
                "runs",
                "console_logs",
                "generated_audio",
                "checkpoints",
                "tmp",
            ):
                (root / dirname).mkdir()
                (root / dirname / "large.bin").write_bytes(b"large")

            first = SEAL.generate(root)
            manifest = root / SEAL.MANIFEST_NAME
            first_payload = manifest.read_bytes()
            second = SEAL.generate(root)
            self.assertEqual(first, second)
            self.assertEqual(first_payload, manifest.read_bytes())
            self.assertEqual(first, SEAL.verify(root))

            names = list(SEAL.parse_manifest(first_payload))
            self.assertEqual(names, sorted(names))
            self.assertIn("LICENSE", names)
            self.assertIn("scripts/managed.txt", names)
            self.assertNotIn(SEAL.MANIFEST_NAME, names)
            self.assertFalse(any("__pycache__" in name for name in names))
            self.assertFalse(any(name.startswith("manifests/") for name in names))
            self.assertFalse(any(name.startswith("docs/tmp/") for name in names))
            self.assertFalse(
                list(root.glob(".WORKPACK_MANIFEST.sha256.tmp.*")),
                "atomic writer left a temporary file behind",
            )

    def test_verify_rejects_changed_missing_and_new_managed_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture(root)
            SEAL.generate(root)

            target = root / "docs" / "managed.txt"
            target.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(SEAL.SealError, "changed=.*docs/managed.txt"):
                SEAL.verify(root)

            SEAL.generate(root)
            target.unlink()
            with self.assertRaisesRegex(SEAL.SealError, "missing=.*docs/managed.txt"):
                SEAL.verify(root)

            target.write_text("docs\n", encoding="utf-8")
            SEAL.generate(root)
            (root / "src" / "new_module.py").write_text("VALUE = 1\n", encoding="utf-8")
            with self.assertRaisesRegex(SEAL.SealError, "added=.*src/new_module.py"):
                SEAL.verify(root)

            (root / "src" / "new_module.py").unlink()
            SEAL.generate(root)
            (root / "notebooks").mkdir()
            (root / "notebooks" / "analysis.md").write_text(
                "new managed tree\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(SEAL.SealError, "added=.*notebooks/analysis.md"):
                SEAL.verify(root)

    def test_symbolic_links_are_rejected_without_following_them(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture(root)
            outside = root.parent / "outside-seal-target.txt"
            outside.write_text("outside\n", encoding="utf-8")
            try:
                (root / "docs" / "escape.md").symlink_to(outside)
                with self.assertRaisesRegex(SEAL.SealError, "symbolic links"):
                    SEAL.generate(root)
            finally:
                outside.unlink()

    def test_manifest_parser_rejects_escape_absolute_duplicate_and_self(self) -> None:
        digest = "0" * 64
        bad_payloads = (
            "{}  ../escape\n".format(digest),
            "{}  /absolute\n".format(digest),
            "{}  a\n{}  a\n".format(digest, digest),
            "{}  {}\n".format(digest, SEAL.MANIFEST_NAME),
            "{}  b\n{}  a\n".format(digest, digest),
            "not-a-hash  src/a.py\n",
        )
        for payload in bad_payloads:
            with self.subTest(payload=payload), self.assertRaises(SEAL.SealError):
                SEAL.parse_manifest(payload.encode("utf-8"))

    def test_verify_rejects_malformed_or_tampered_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture(root)
            SEAL.generate(root)
            manifest = root / SEAL.MANIFEST_NAME
            manifest.write_text("bad\n", encoding="utf-8")
            with self.assertRaisesRegex(SEAL.SealError, "malformed"):
                SEAL.verify(root)

    def test_cli_returns_nonzero_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture(root)
            self.assertEqual(SEAL.main(["generate", "--root", str(root)]), 0)
            self.assertEqual(SEAL.main(["verify", "--root", str(root)]), 0)
            (root / "tests" / "managed.txt").unlink()
            self.assertEqual(SEAL.main(["verify", "--root", str(root)]), 1)


if __name__ == "__main__":
    unittest.main()
