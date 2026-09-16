#!/usr/bin/env python3
"""Generate or verify the portable workpack source-integrity manifest.

The fixed output is ``WORKPACK_MANIFEST.sha256`` in the workpack root.  Only
source and protocol inputs are sealed; large/generated research products are
explicitly excluded.  Verification is closed-world over that managed scope:
missing files, newly added files, altered bytes, malformed entries, and
symbolic links all fail.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


MANIFEST_NAME = "WORKPACK_MANIFEST.sha256"
MANAGED_DIRECTORIES = (
    "configs",
    "docs",
    "patches",
    "scripts",
    "src",
    "tests",
)
REQUIRED_ROOT_FILES = (
    ".gitignore",
    "BASELINE_PROVENANCE.md",
    "CHANGELOG.md",
    "README.md",
    "pyproject.toml",
)
EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "artifacts",
        "build",
        "checkpoints",
        "console_logs",
        "generated_audio",
        "manifests",
        "metrics",
        "runs",
        "dist",
        "tmp",
        "vendor",
    }
)
EXCLUDED_FILE_NAMES = frozenset({".DS_Store", MANIFEST_NAME})
MANIFEST_PATTERN = re.compile(r"^([0-9a-f]{64})  (.+)$")


class SealError(RuntimeError):
    """A deterministic integrity or workpack-layout failure."""


def _is_excluded_directory_name(name: str) -> bool:
    return name in EXCLUDED_DIRECTORY_NAMES or name.endswith(".egg-info")


def _resolve_root(root: Path) -> Path:
    if root.is_symlink():
        raise SealError("workpack root may not be a symbolic link")
    try:
        resolved = root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise SealError("workpack root does not exist: {}".format(root)) from exc
    if not resolved.is_dir():
        raise SealError("workpack root is not a directory: {}".format(root))
    return resolved


def _validate_relative_path(value: str) -> PurePosixPath:
    if not value or "\x00" in value or "\n" in value or "\r" in value:
        raise SealError("manifest contains an invalid path")
    if "\\" in value:
        raise SealError("manifest paths must use portable '/' separators")
    path = PurePosixPath(value)
    if path.is_absolute() or value.startswith("/"):
        raise SealError("manifest path may not be absolute: {!r}".format(value))
    if any(part in {"", ".", ".."} for part in path.parts):
        raise SealError("manifest path escapes or is non-canonical: {!r}".format(value))
    if path.as_posix() != value:
        raise SealError("manifest path is non-canonical: {!r}".format(value))
    return path


def _relative_name(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise SealError("managed path escapes workpack root: {}".format(path)) from exc
    name = relative.as_posix()
    _validate_relative_path(name)
    return name


def _walk_managed_directory(directory: Path, root: Path) -> Iterable[Tuple[str, Path]]:
    """Walk without following links, pruning every documented generated tree."""

    try:
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
    except OSError as exc:
        raise SealError("cannot scan managed directory {}: {}".format(directory, exc)) from exc
    for entry in entries:
        path = Path(entry.path)
        try:
            if entry.is_symlink():
                raise SealError("symbolic links are forbidden in managed scope: {}".format(path))
            if entry.is_dir(follow_symlinks=False):
                if _is_excluded_directory_name(entry.name):
                    continue
                yield from _walk_managed_directory(path, root)
            elif entry.is_file(follow_symlinks=False):
                if entry.name in EXCLUDED_FILE_NAMES:
                    continue
                yield _relative_name(path, root), path
            else:
                raise SealError(
                    "non-regular filesystem entry in managed scope: {}".format(path)
                )
        except OSError as exc:
            raise SealError("cannot inspect managed path {}: {}".format(path, exc)) from exc


def discover_managed_files(root: Path) -> Dict[str, Path]:
    """Return the complete current managed file set keyed by portable path."""

    root = _resolve_root(root)
    managed: Dict[str, Path] = {}

    for filename in REQUIRED_ROOT_FILES:
        path = root / filename
        if path.is_symlink() or not path.is_file():
            raise SealError("required root file is missing or not regular: {}".format(filename))
    for dirname in MANAGED_DIRECTORIES:
        directory = root / dirname
        if directory.is_symlink() or not directory.is_dir():
            raise SealError(
                "required managed directory is missing or not regular: {}".format(dirname)
            )

    # Every root-level regular file and every non-generated directory tree is
    # managed, not just the mandatory starting set. A future LICENSE,
    # notebooks/, or protocol package is therefore a closed-world addition
    # that verification cannot silently overlook.
    try:
        with os.scandir(root) as iterator:
            root_entries = sorted(iterator, key=lambda item: item.name)
    except OSError as exc:
        raise SealError("cannot scan workpack root {}: {}".format(root, exc)) from exc
    for entry in root_entries:
        path = Path(entry.path)
        if entry.is_symlink():
            raise SealError("symbolic links are forbidden in workpack source scope: {}".format(path))
        if entry.is_file(follow_symlinks=False):
            if entry.name not in EXCLUDED_FILE_NAMES:
                managed[_relative_name(path, root)] = path
        elif entry.is_dir(follow_symlinks=False):
            if _is_excluded_directory_name(entry.name):
                continue
            for name, child_path in _walk_managed_directory(path, root):
                if name in managed:
                    raise SealError("duplicate managed path discovered: {}".format(name))
                managed[name] = child_path
        else:
            raise SealError(
                "non-regular filesystem entry at workpack root: {}".format(path)
            )

    for name, path in managed.items():
        if name in EXCLUDED_FILE_NAMES:
            raise SealError("internal error: excluded file was managed: {}".format(path))

    return dict(sorted(managed.items()))


def sha256_regular_file(path: Path) -> str:
    """Hash a stable regular file and fail if it changes during the read."""

    if path.is_symlink() or not path.is_file():
        raise SealError("cannot hash non-regular file: {}".format(path))
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise SealError("file changed while hashing: {}".format(path))
    return digest.hexdigest()


def compute_records(root: Path) -> Dict[str, str]:
    return {
        name: sha256_regular_file(path)
        for name, path in discover_managed_files(root).items()
    }


def encode_manifest(records: Mapping[str, str]) -> bytes:
    lines: List[str] = []
    previous = None
    for name in sorted(records):
        _validate_relative_path(name)
        digest = records[name]
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise SealError("invalid SHA-256 value for {}".format(name))
        if previous is not None and name <= previous:
            raise SealError("manifest paths must be strictly sorted")
        lines.append("{}  {}\n".format(digest, name))
        previous = name
    if not lines:
        raise SealError("refusing to write an empty workpack manifest")
    return "".join(lines).encode("utf-8")


def parse_manifest(payload: bytes) -> Dict[str, str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SealError("workpack manifest is not UTF-8") from exc
    if not text or not text.endswith("\n"):
        raise SealError("workpack manifest must be nonempty and newline-terminated")
    records: Dict[str, str] = {}
    previous = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = MANIFEST_PATTERN.fullmatch(line)
        if match is None:
            raise SealError("malformed manifest line {}".format(line_number))
        digest, name = match.groups()
        _validate_relative_path(name)
        if name == MANIFEST_NAME:
            raise SealError("manifest may not hash itself")
        if previous is not None and name <= previous:
            raise SealError("manifest paths are duplicated or not sorted")
        records[name] = digest
        previous = name
    return records


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(".{}.tmp.{}".format(path.name, os.getpid()))
    if temporary.exists() or temporary.is_symlink():
        raise SealError("temporary manifest path already exists: {}".format(temporary))
    descriptor = None
    try:
        descriptor = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise SealError("atomic manifest write failed: {}".format(exc)) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def generate(root: Path) -> Dict[str, str]:
    root = _resolve_root(root)
    records = compute_records(root)
    _atomic_write(root / MANIFEST_NAME, encode_manifest(records))
    return records


def verify(root: Path) -> Dict[str, str]:
    root = _resolve_root(root)
    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise SealError("{} is missing or not a regular file".format(MANIFEST_NAME))
    expected = parse_manifest(manifest_path.read_bytes())
    actual = compute_records(root)

    expected_names = set(expected)
    actual_names = set(actual)
    missing = sorted(expected_names - actual_names)
    added = sorted(actual_names - expected_names)
    changed = sorted(
        name
        for name in expected_names & actual_names
        if expected[name] != actual[name]
    )
    if missing or added or changed:
        details = []
        if missing:
            details.append("missing={}".format(missing))
        if added:
            details.append("added={}".format(added))
        if changed:
            details.append("changed={}".format(changed))
        raise SealError("workpack verification failed: " + "; ".join(details))
    return actual


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("generate", "verify"))
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="workpack root; defaults to the parent of scripts/",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        records = generate(args.root) if args.mode == "generate" else verify(args.root)
    except SealError as exc:
        print("WORKPACK SEAL FAILED: {}".format(exc), file=sys.stderr)
        return 1
    print(
        "WORKPACK {} OK: {} managed files".format(args.mode.upper(), len(records))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
