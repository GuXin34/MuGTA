#!/usr/bin/env python3
"""Read-only portability and provenance checks for the PTC-OPD workpack."""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path
from typing import Iterable


EXPECTED_COMMITS = {
    Path("third_party/audiocraft"): "896ec7c47f5e5d1e5aa1e4b260c4405328bf009d",
    Path("third_party/MuQ-Eval"): "60a88f8ac0909ca1fd1a78af3660f1fc376977a1",
}


def iter_python_files(root: Path) -> Iterable[Path]:
    for parent in (root / "src", root / "scripts", root / "tests"):
        if parent.exists():
            yield from sorted(parent.rglob("*.py"))


def check_python_syntax(root: Path) -> list[str]:
    failures: list[str] = []
    files = list(iter_python_files(root))
    if not files:
        failures.append("workpack contains no Python files")
        return failures
    for path in files:
        try:
            # The accepted training environment is Python 3.9.  Parse against
            # that grammar even when this portability check runs on a newer
            # packaging machine, so newer-only syntax cannot be sealed by
            # accident and fail only after the workpack is copied remotely.
            ast.parse(
                path.read_text(encoding="utf-8"),
                filename=str(path),
                feature_version=(3, 9),
            )
        except (OSError, SyntaxError, UnicodeError) as exc:
            failures.append(f"syntax/read failure: {path}: {exc}")
    print(f"[ok] parsed {len(files)} Python files")
    return failures


def git_head(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "git rev-parse failed"
        raise RuntimeError(detail)
    return completed.stdout.strip()


def check_base_project(base_root: Path) -> list[str]:
    failures: list[str] = []
    for relative, expected in EXPECTED_COMMITS.items():
        repo = base_root / relative
        if not repo.is_dir():
            failures.append(f"missing base repository: {repo}")
            continue
        try:
            actual = git_head(repo)
        except RuntimeError as exc:
            failures.append(f"cannot identify {repo}: {exc}")
            continue
        if actual != expected:
            failures.append(
                f"commit mismatch for {repo}: expected {expected}, found {actual}"
            )
        else:
            print(f"[ok] {relative}: {actual}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-project-root",
        type=Path,
        help="Base project containing third_party/audiocraft and MuQ-Eval.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    failures = check_python_syntax(root)
    if args.base_project_root is not None:
        failures.extend(check_base_project(args.base_project_root.resolve()))
    else:
        print("[info] base-project commit checks skipped (no path supplied)")

    if failures:
        for failure in failures:
            print(f"[fail] {failure}", file=sys.stderr)
        return 1
    print("WORKPACK CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
