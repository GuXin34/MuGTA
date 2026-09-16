from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_audiocraft_overlay.py"
SPEC = importlib.util.spec_from_file_location("check_audiocraft_overlay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class OverlayCheckerTest(unittest.TestCase):
    def git(self, repo: Path, *arguments: str) -> subprocess.CompletedProcess:
        environment = os.environ.copy()
        environment.update(
            {
                "GIT_AUTHOR_NAME": "Overlay Test",
                "GIT_AUTHOR_EMAIL": "overlay@example.invalid",
                "GIT_COMMITTER_NAME": "Overlay Test",
                "GIT_COMMITTER_EMAIL": "overlay@example.invalid",
            }
        )
        return subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )

    def make_repo_and_patch(self, root: Path) -> tuple[Path, Path, str]:
        repo = root / "audiocraft"
        patches = root / "patches" / "audiocraft"
        repo.mkdir(parents=True)
        patches.mkdir(parents=True)
        self.git(repo, "init", "--quiet")
        source = repo / "model.py"
        source.write_text("LOSS = 'uniform'\n", encoding="utf-8")
        self.git(repo, "add", "model.py")
        self.git(repo, "commit", "--quiet", "-m", "baseline")
        head = self.git(repo, "rev-parse", "HEAD").stdout.strip()

        source.write_text("LOSS = 'ptc-opd'\n", encoding="utf-8")
        patch_text = self.git(repo, "diff", "--", "model.py").stdout
        (patches / "0001-ptc-opd.patch").write_text(patch_text, encoding="utf-8")
        source.write_text("LOSS = 'uniform'\n", encoding="utf-8")
        self.assertEqual(self.git(repo, "status", "--porcelain").stdout, "")
        return repo, patches, head

    def test_expected_commit_constant_is_frozen(self) -> None:
        self.assertEqual(
            MODULE.EXPECTED_AUDIOCRAFT_COMMIT,
            "896ec7c47f5e5d1e5aa1e4b260c4405328bf009d",
        )

    def test_rejects_wrong_head(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo, patches, _ = self.make_repo_and_patch(Path(temporary))
            with self.assertRaisesRegex(MODULE.OverlayCheckError, "HEAD mismatch"):
                MODULE.check_overlay(repo, patches, expected_head="0" * 40)

    def test_rejects_dirty_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo, patches, head = self.make_repo_and_patch(Path(temporary))
            (repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")
            with self.assertRaisesRegex(MODULE.OverlayCheckError, "worktree is dirty"):
                MODULE.check_overlay(repo, patches, expected_head=head)

    def test_rejects_missing_patch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo, patches, head = self.make_repo_and_patch(Path(temporary))
            (patches / "0001-ptc-opd.patch").unlink()
            with self.assertRaisesRegex(MODULE.OverlayCheckError, "no .patch files"):
                MODULE.check_overlay(repo, patches, expected_head=head)

    def test_valid_patch_is_checked_without_application(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo, patches, head = self.make_repo_and_patch(Path(temporary))
            result = MODULE.check_overlay(repo, patches, expected_head=head)
            self.assertEqual(result.head, head)
            self.assertEqual([path.name for path in result.patch_paths], ["0001-ptc-opd.patch"])
            self.assertEqual((repo / "model.py").read_text(encoding="utf-8"), "LOSS = 'uniform'\n")
            self.assertEqual(self.git(repo, "status", "--porcelain").stdout, "")

            commands = MODULE.format_apply_commands(result)
            self.assertIn("git -C", commands)
            self.assertIn("0001-ptc-opd.patch", commands)
            self.assertNotIn("--check", commands)


if __name__ == "__main__":
    unittest.main()
