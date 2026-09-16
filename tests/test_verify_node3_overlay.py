"""Regression tests for the post-apply AudioCraft overlay verifier."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_node3_overlay.py"
SPEC = importlib.util.spec_from_file_location("verify_node3_overlay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class AppliedOverlayVerifierTest(unittest.TestCase):
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

    def make_applied_overlay(
        self, root: Path
    ) -> tuple[Path, Path, str, str, str, str]:
        repo = root / "audiocraft"
        patch = root / "patches" / "0001-explicit-no-cfg-generation.patch"
        lm_path = repo / "audiocraft" / "models" / "lm.py"
        test_path = repo / "tests" / "models" / "test_lm_no_cfg.py"
        lm_path.parent.mkdir(parents=True)
        test_path.parent.mkdir(parents=True)
        patch.parent.mkdir(parents=True)

        self.git(repo, "init", "--quiet")
        base_lines = ["LINE_{:02d} = {}\n".format(index, index) for index in range(40)]
        base_lines[20] = "MODE = 'base'\n"
        lm_path.write_text("".join(base_lines), encoding="utf-8")
        self.git(repo, "add", "audiocraft/models/lm.py")
        self.git(repo, "commit", "--quiet", "-m", "baseline")
        head = self.git(repo, "rev-parse", "HEAD").stdout.rstrip("\r\n")

        patched_lines = list(base_lines)
        patched_lines[20] = "MODE = 'explicit-no-cfg'\n"
        lm_path.write_text("".join(patched_lines), encoding="utf-8")
        test_path.write_text("def test_explicit_no_cfg():\n    assert True\n", encoding="utf-8")
        self.git(repo, "add", "-N", "tests/models/test_lm_no_cfg.py")
        patch.write_text(
            self.git(
                repo,
                "diff",
                "--binary",
                "--",
                "audiocraft/models/lm.py",
                "tests/models/test_lm_no_cfg.py",
            ).stdout,
            encoding="utf-8",
        )

        self.git(repo, "reset", "--hard", "HEAD")
        if test_path.exists():
            test_path.unlink()
        self.git(repo, "apply", str(patch))
        self.assertEqual(
            set(
                self.git(
                    repo, "status", "--porcelain=v1", "--untracked-files=all"
                ).stdout.splitlines()
            ),
            MODULE.EXPECTED_STATUS,
        )
        return (
            repo,
            patch,
            head,
            MODULE.sha256_file(patch),
            MODULE.sha256_file(lm_path),
            MODULE.sha256_file(test_path),
        )

    def expected_identity(
        self,
        *,
        head: str,
        patch_sha256: str,
        lm_sha256: str,
        test_sha256: str,
    ):
        return mock.patch.multiple(
            MODULE,
            EXPECTED_AUDIOCRAFT_COMMIT=head,
            EXPECTED_PATCH_SHA256=patch_sha256,
            EXPECTED_LM_SHA256=lm_sha256,
            EXPECTED_TEST_SHA256=test_sha256,
        )

    def test_require_git_removes_only_line_terminators(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["git"],
            returncode=0,
            stdout=" M audiocraft/models/lm.py\r\n?? tests/models/test_lm_no_cfg.py\r\n",
            stderr="",
        )
        with mock.patch.object(MODULE, "_git", return_value=completed):
            observed = MODULE._require_git(Path("/unused"), "status")
        self.assertEqual(
            observed,
            " M audiocraft/models/lm.py\r\n?? tests/models/test_lm_no_cfg.py",
        )

    def test_require_git_reports_an_empty_failure(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["git"], returncode=1, stdout="\n", stderr=" \r\n"
        )
        with mock.patch.object(MODULE, "_git", return_value=completed):
            with self.assertRaisesRegex(
                MODULE.OverlayVerificationError, "unknown error"
            ):
                MODULE._require_git(Path("/unused"), "rev-parse", "HEAD")

    def test_production_identity_constants_are_frozen(self) -> None:
        self.assertEqual(
            MODULE.EXPECTED_AUDIOCRAFT_COMMIT,
            "896ec7c47f5e5d1e5aa1e4b260c4405328bf009d",
        )
        self.assertEqual(
            MODULE.EXPECTED_PATCH_SHA256,
            "fb141d4b37f1068b2adb113f80253d3c3f17a5fde86e8423071931d5ccc1daaf",
        )
        self.assertEqual(
            MODULE.EXPECTED_LM_SHA256,
            "07a37138ded33cfa3efd9764fd4abae89daf8297595cc29535bc660aeea61e53",
        )
        self.assertEqual(
            MODULE.EXPECTED_TEST_SHA256,
            "d2a2940cdbf1794a697b63b54cfee4c4df9ab45fc966c9a921e58cbcd410afed",
        )

    def test_real_applied_overlay_accepts_porcelain_leading_space(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo, patch, head, patch_sha, lm_sha, test_sha = self.make_applied_overlay(
                Path(temporary)
            )
            with self.expected_identity(
                head=head,
                patch_sha256=patch_sha,
                lm_sha256=lm_sha,
                test_sha256=test_sha,
            ):
                result = MODULE.verify_applied_overlay(repo, patch)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(
                result["schema_version"], "ptc-opd-node3-overlay-verification-v2"
            )
            self.assertEqual(set(result["git_status"]), MODULE.EXPECTED_STATUS)
            self.assertIn(" M audiocraft/models/lm.py", result["git_status"])
            self.assertEqual(result["patch_sha256"], patch_sha)
            self.assertEqual(result["lm_sha256"], lm_sha)
            self.assertEqual(result["test_sha256"], test_sha)
            self.assertEqual(int(result["lm_mode"], 8) & 0o111, 0)
            self.assertEqual(int(result["test_mode"], 8) & 0o111, 0)

    def test_real_applied_overlay_rejects_wrong_head(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo, patch, _, patch_sha, lm_sha, test_sha = self.make_applied_overlay(
                Path(temporary)
            )
            with self.expected_identity(
                head="0" * 40,
                patch_sha256=patch_sha,
                lm_sha256=lm_sha,
                test_sha256=test_sha,
            ):
                with self.assertRaisesRegex(
                    MODULE.OverlayVerificationError, "HEAD mismatch"
                ):
                    MODULE.verify_applied_overlay(repo, patch)

    def test_real_applied_overlay_rejects_an_extra_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo, patch, head, patch_sha, lm_sha, test_sha = self.make_applied_overlay(
                Path(temporary)
            )
            (repo / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
            with self.expected_identity(
                head=head,
                patch_sha256=patch_sha,
                lm_sha256=lm_sha,
                test_sha256=test_sha,
            ):
                with self.assertRaisesRegex(
                    MODULE.OverlayVerificationError, "exact two-path change"
                ):
                    MODULE.verify_applied_overlay(repo, patch)

    def test_real_applied_overlay_rejects_extra_bytes_in_expected_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo, patch, head, patch_sha, lm_sha, test_sha = self.make_applied_overlay(
                Path(temporary)
            )
            lm_path = repo / "audiocraft" / "models" / "lm.py"
            with lm_path.open("a", encoding="utf-8") as stream:
                stream.write("UNEXPECTED = True\n")
            self.assertEqual(
                set(
                    self.git(
                        repo, "status", "--porcelain=v1", "--untracked-files=all"
                    ).stdout.splitlines()
                ),
                MODULE.EXPECTED_STATUS,
            )
            self.git(repo, "apply", "--reverse", "--check", str(patch))
            with self.expected_identity(
                head=head,
                patch_sha256=patch_sha,
                lm_sha256=lm_sha,
                test_sha256=test_sha,
            ):
                with self.assertRaisesRegex(
                    MODULE.OverlayVerificationError, "post-image SHA-256 mismatch"
                ):
                    MODULE.verify_applied_overlay(repo, patch)

    def test_real_applied_overlay_rejects_reverse_incompatible_patch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo, patch, head, _, lm_sha, test_sha = self.make_applied_overlay(
                Path(temporary)
            )
            patch.write_text(
                patch.read_text(encoding="utf-8").replace(
                    "explicit-no-cfg", "different-post-image"
                ),
                encoding="utf-8",
            )
            with self.expected_identity(
                head=head,
                patch_sha256=MODULE.sha256_file(patch),
                lm_sha256=lm_sha,
                test_sha256=test_sha,
            ):
                with self.assertRaisesRegex(
                    MODULE.OverlayVerificationError, "do not exactly reverse"
                ):
                    MODULE.verify_applied_overlay(repo, patch)

    def test_real_applied_overlay_rejects_executable_bit_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo, patch, head, patch_sha, lm_sha, test_sha = self.make_applied_overlay(
                Path(temporary)
            )
            (repo / "audiocraft" / "models" / "lm.py").chmod(0o755)
            with self.expected_identity(
                head=head,
                patch_sha256=patch_sha,
                lm_sha256=lm_sha,
                test_sha256=test_sha,
            ):
                with self.assertRaisesRegex(
                    MODULE.OverlayVerificationError, "must be non-executable"
                ):
                    MODULE.verify_applied_overlay(repo, patch)

    def test_real_applied_overlay_rejects_patch_byte_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo, patch, head, patch_sha, lm_sha, test_sha = self.make_applied_overlay(
                Path(temporary)
            )
            with patch.open("a", encoding="utf-8") as stream:
                stream.write("\n")
            with self.expected_identity(
                head=head,
                patch_sha256=patch_sha,
                lm_sha256=lm_sha,
                test_sha256=test_sha,
            ):
                with self.assertRaisesRegex(
                    MODULE.OverlayVerificationError, "patch SHA-256 mismatch"
                ):
                    MODULE.verify_applied_overlay(repo, patch)


if __name__ == "__main__":
    unittest.main()
