#!/usr/bin/env python3
"""Fail-closed verification of the already-applied node-3 AudioCraft overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess
from typing import Any, Dict, Optional, Sequence


EXPECTED_AUDIOCRAFT_COMMIT = "896ec7c47f5e5d1e5aa1e4b260c4405328bf009d"
EXPECTED_STATUS = {
    " M audiocraft/models/lm.py",
    "?? tests/models/test_lm_no_cfg.py",
}
EXPECTED_PATCH_SHA256 = "fb141d4b37f1068b2adb113f80253d3c3f17a5fde86e8423071931d5ccc1daaf"
EXPECTED_LM_SHA256 = "07a37138ded33cfa3efd9764fd4abae89daf8297595cc29535bc660aeea61e53"
EXPECTED_TEST_SHA256 = "d2a2940cdbf1794a697b63b54cfee4c4df9ab45fc966c9a921e58cbcd410afed"


class OverlayVerificationError(RuntimeError):
    pass


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _require_git(root: Path, *arguments: str) -> str:
    completed = _git(root, *arguments)
    if completed.returncode != 0:
        detail = (
            completed.stderr.strip()
            or completed.stdout.strip()
            or "unknown error"
        )
        raise OverlayVerificationError(
            "git {} failed: {}".format(" ".join(arguments), detail)
        )
    # Porcelain v1 uses the leading character as the staged-status column.
    # str.strip() would turn an unstaged line such as ``" M path"`` into
    # ``"M path"`` and make an exact applied overlay impossible to verify.
    # Remove line terminators only; every other byte is semantically relevant.
    return completed.stdout.rstrip("\r\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_applied_overlay(audiocraft_root: Path, patch: Path) -> Dict[str, Any]:
    if audiocraft_root.is_symlink() or not audiocraft_root.is_dir():
        raise OverlayVerificationError("AudioCraft root must be a regular directory")
    root = audiocraft_root.resolve()
    if patch.is_symlink() or not patch.is_file():
        raise OverlayVerificationError("overlay patch must be a regular file")
    patch_sha256 = sha256_file(patch)
    if patch_sha256 != EXPECTED_PATCH_SHA256:
        raise OverlayVerificationError(
            "overlay patch SHA-256 mismatch: expected {}, found {}".format(
                EXPECTED_PATCH_SHA256, patch_sha256
            )
        )
    top = Path(_require_git(root, "rev-parse", "--show-toplevel")).resolve()
    if top != root:
        raise OverlayVerificationError("AudioCraft path is not the Git worktree root")
    head = _require_git(root, "rev-parse", "--verify", "HEAD")
    if head != EXPECTED_AUDIOCRAFT_COMMIT:
        raise OverlayVerificationError(
            "AudioCraft HEAD mismatch: expected {}, found {}".format(
                EXPECTED_AUDIOCRAFT_COMMIT, head
            )
        )
    status_lines = set(
        _require_git(root, "status", "--porcelain=v1", "--untracked-files=all").splitlines()
    )
    if status_lines != EXPECTED_STATUS:
        raise OverlayVerificationError(
            "applied overlay is not the exact two-path change: {}".format(
                sorted(status_lines)
            )
        )
    reverse = _git(root, "apply", "--reverse", "--check", str(patch.resolve()))
    if reverse.returncode != 0:
        detail = reverse.stderr.strip() or reverse.stdout.strip() or "unknown error"
        raise OverlayVerificationError(
            "overlay bytes do not exactly reverse against the pinned patch: {}".format(
                detail
            )
        )
    _require_git(root, "diff", "--check")
    test_path = root / "tests" / "models" / "test_lm_no_cfg.py"
    lm_path = root / "audiocraft" / "models" / "lm.py"
    if any(path.is_symlink() or not path.is_file() for path in (test_path, lm_path)):
        raise OverlayVerificationError("applied overlay paths must be regular files")
    lm_sha256 = sha256_file(lm_path)
    test_sha256 = sha256_file(test_path)
    observed_postimage = {
        "audiocraft/models/lm.py": lm_sha256,
        "tests/models/test_lm_no_cfg.py": test_sha256,
    }
    expected_postimage = {
        "audiocraft/models/lm.py": EXPECTED_LM_SHA256,
        "tests/models/test_lm_no_cfg.py": EXPECTED_TEST_SHA256,
    }
    if observed_postimage != expected_postimage:
        raise OverlayVerificationError(
            "applied overlay post-image SHA-256 mismatch: expected {}, found {}".format(
                expected_postimage, observed_postimage
            )
        )
    lm_mode = stat.S_IMODE(lm_path.stat().st_mode)
    test_mode = stat.S_IMODE(test_path.stat().st_mode)
    executable_modes = {
        name: mode
        for name, mode in {
            "audiocraft/models/lm.py": lm_mode,
            "tests/models/test_lm_no_cfg.py": test_mode,
        }.items()
        if mode & 0o111
    }
    if executable_modes:
        raise OverlayVerificationError(
            "applied overlay files must be non-executable: found {}".format(
                {name: oct(mode) for name, mode in executable_modes.items()}
            )
        )
    return {
        "schema_version": "ptc-opd-node3-overlay-verification-v2",
        "status": "passed",
        "audiocraft_root": str(root),
        "audiocraft_head": head,
        "patch": str(patch.resolve()),
        "patch_sha256": patch_sha256,
        "git_status": sorted(status_lines),
        "lm_sha256": lm_sha256,
        "test_sha256": test_sha256,
        "lm_mode": oct(lm_mode),
        "test_mode": oct(test_mode),
        "reverse_apply_check": "passed",
        "diff_check": "passed",
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    workpack = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audiocraft-root", type=Path, required=True)
    parser.add_argument(
        "--patch",
        type=Path,
        default=workpack / "patches" / "audiocraft" / "0001-explicit-no-cfg-generation.patch",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = verify_applied_overlay(args.audiocraft_root, args.patch)
    except OverlayVerificationError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
