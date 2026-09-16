#!/usr/bin/env python3
"""Read-only validation for the pinned AudioCraft patch overlay.

The checker deliberately has no code path that invokes ``git apply`` without
``--check``.  Applying a validated overlay remains a separate, explicit human
action.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


EXPECTED_AUDIOCRAFT_COMMIT = "896ec7c47f5e5d1e5aa1e4b260c4405328bf009d"


class OverlayCheckError(RuntimeError):
    """Raised when an overlay precondition or dry-run check fails."""


@dataclass(frozen=True)
class OverlayCheckResult:
    audiocraft_root: Path
    head: str
    patch_paths: Sequence[Path]
    dirty_allowed: bool


def _run_git(root: Path, arguments: Sequence[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        raise OverlayCheckError(f"cannot execute git: {exc}") from exc


def _git_output(root: Path, arguments: Sequence[str], operation: str) -> str:
    completed = _run_git(root, arguments)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise OverlayCheckError(f"{operation} failed: {detail}")
    return completed.stdout.strip()


def discover_patches(patch_dir: Path) -> List[Path]:
    """Return regular ``*.patch`` files in stable lexical order."""
    if not patch_dir.is_dir():
        raise OverlayCheckError(f"patch directory is missing: {patch_dir}")
    patches = sorted(path.resolve() for path in patch_dir.glob("*.patch") if path.is_file())
    if not patches:
        raise OverlayCheckError(f"no .patch files found in: {patch_dir}")
    return patches


def check_overlay(
    audiocraft_root: Path,
    patch_dir: Path,
    *,
    expected_head: str = EXPECTED_AUDIOCRAFT_COMMIT,
    allow_dirty: bool = False,
) -> OverlayCheckResult:
    """Validate provenance, cleanliness, and every patch without writing."""
    root = audiocraft_root.expanduser().resolve()
    patches_root = patch_dir.expanduser().resolve()
    if not root.is_dir():
        raise OverlayCheckError(f"AudioCraft root is not a directory: {root}")

    top_level_text = _git_output(
        root, ["rev-parse", "--show-toplevel"], "locating Git worktree"
    )
    top_level = Path(top_level_text).resolve()
    if top_level != root:
        raise OverlayCheckError(
            f"--audiocraft-root must be the Git worktree root: "
            f"received {root}, Git reports {top_level}"
        )

    head = _git_output(root, ["rev-parse", "--verify", "HEAD"], "reading Git HEAD")
    if head != expected_head:
        raise OverlayCheckError(
            f"AudioCraft HEAD mismatch: expected {expected_head}, found {head}"
        )

    status = _git_output(
        root,
        ["status", "--porcelain=v1", "--untracked-files=all"],
        "checking Git worktree status",
    )
    if status and not allow_dirty:
        preview = "\n".join(status.splitlines()[:20])
        suffix = "\n..." if len(status.splitlines()) > 20 else ""
        raise OverlayCheckError(
            "AudioCraft worktree is dirty; refuse to validate by default:\n"
            f"{preview}{suffix}"
        )

    patches = discover_patches(patches_root)
    # `git apply --check p1 p2 ...` validates the cumulative overlay. Checking
    # each patch independently against the pristine tree can reject a valid p2
    # that intentionally depends on p1.
    completed = _run_git(
        root, ["apply", "--check", *[str(path) for path in patches]]
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        names = ", ".join(path.name for path in patches)
        raise OverlayCheckError(
            f"cumulative git apply --check failed for [{names}]: {detail}"
        )

    return OverlayCheckResult(
        audiocraft_root=root,
        head=head,
        patch_paths=tuple(patches),
        dirty_allowed=allow_dirty,
    )


def format_apply_commands(result: OverlayCheckResult) -> str:
    """Render commands for manual review; this function never executes them."""
    words = [
        "git",
        "-C",
        str(result.audiocraft_root),
        "apply",
        *[str(path) for path in result.patch_paths],
    ]
    return " ".join(shlex.quote(word) for word in words)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    workpack_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Read-only validation of the pinned AudioCraft overlay."
    )
    parser.add_argument(
        "--audiocraft-root",
        type=Path,
        required=True,
        help="Path to the AudioCraft Git worktree pinned by this workpack.",
    )
    parser.add_argument(
        "--patch-dir",
        type=Path,
        default=workpack_root / "patches" / "audiocraft",
        help="Overlay patch directory (default: workpack patches/audiocraft).",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Diagnostic override: check a dirty worktree instead of refusing it.",
    )
    parser.add_argument(
        "--print-apply-command",
        action="store_true",
        help="After successful checks, print manual git apply commands; do not run them.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = check_overlay(
            args.audiocraft_root,
            args.patch_dir,
            allow_dirty=args.allow_dirty,
        )
    except OverlayCheckError as exc:
        print(f"[fail] {exc}", file=sys.stderr)
        return 1

    cleanliness = "dirty override enabled" if result.dirty_allowed else "clean"
    print(f"[ok] AudioCraft HEAD: {result.head}")
    print(f"[ok] AudioCraft worktree: {cleanliness}")
    print(
        "[ok] cumulative git apply --check: "
        + ", ".join(path.name for path in result.patch_paths)
    )
    print(f"OVERLAY CHECK PASSED ({len(result.patch_paths)} patch(es))")
    if args.print_apply_command:
        print("# Manual commands only; the checker did not execute these:")
        print(format_apply_commands(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
