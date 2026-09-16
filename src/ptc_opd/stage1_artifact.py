"""Small, dependency-free primitives for Stage-1 control-plane artifacts.

The scientific runners import PyTorch and AudioCraft.  The control plane must
still be independently testable on a CPU login node, so hashing, strict JSON,
checksum manifests, and atomic artifact publication live in this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CHECKSUM_LINE_RE = re.compile(r"^([0-9a-f]{64})  (.+)$")


class Stage1ArtifactError(ValueError):
    """A closed artifact or one of its identities is invalid."""


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise Stage1ArtifactError("not a regular file: {}".format(path))
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise Stage1ArtifactError("file changed while hashing: {}".format(path))
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    # Artifact files include a trailing newline; scientific-config identities
    # conventionally do not.  Keep this helper newline-free for config hashes.
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json_strict(path: Path) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Stage1ArtifactError("JSON is missing or not regular: {}".format(path))

    def reject_constant(value: str) -> None:
        raise Stage1ArtifactError(
            "{} contains forbidden non-finite {}".format(path, value)
        )

    def unique_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Stage1ArtifactError(
                    "{} contains duplicate key {!r}".format(path, key)
                )
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(
                stream,
                object_pairs_hook=unique_object,
                parse_constant=reject_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Stage1ArtifactError("invalid UTF-8 JSON: {}".format(path)) from exc
    if not isinstance(value, dict):
        raise Stage1ArtifactError("JSON root must be an object: {}".format(path))
    return value


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise Stage1ArtifactError("{} must be lowercase SHA-256".format(label))
    return value


def require_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Stage1ArtifactError("{} must be numeric".format(label))
    result = float(value)
    if not math.isfinite(result):
        raise Stage1ArtifactError("{} must be finite".format(label))
    return result


def _canonical_relative(value: str) -> PurePosixPath:
    if not value or "\\" in value or "\x00" in value:
        raise Stage1ArtifactError("invalid checksum path {!r}".format(value))
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise Stage1ArtifactError("unsafe checksum path {!r}".format(value))
    if path.as_posix() != value:
        raise Stage1ArtifactError("non-canonical checksum path {!r}".format(value))
    return path


def regular_tree_files(root: Path) -> Dict[str, Path]:
    if root.is_symlink() or not root.is_dir():
        raise Stage1ArtifactError("not a regular directory: {}".format(root))
    result: Dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise Stage1ArtifactError("symlink forbidden in artifact: {}".format(path))
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            _canonical_relative(relative)
            result[relative] = path
        elif not path.is_dir():
            raise Stage1ArtifactError(
                "non-regular artifact member: {}".format(path)
            )
    return result


def sha256_tree(root: Path) -> str:
    """Hash relative names, sizes, and file digests for copy-stable identity."""

    files = regular_tree_files(root)
    if not files:
        raise Stage1ArtifactError("refusing to hash empty tree: {}".format(root))
    digest = hashlib.sha256()
    for relative, path in files.items():
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def verify_checksum_manifest(
    root: Path,
    *,
    manifest_name: str = "SHA256SUMS.txt",
    sidecar_name: str = "SHA256SUMS.txt.sha256",
    require_closed_world: bool = True,
    extra_excluded_names: Iterable[str] = (),
) -> Dict[str, str]:
    """Verify a GNU-style two-space SHA manifest and optional closed world."""

    root = root.expanduser().absolute()
    if root.is_symlink():
        raise Stage1ArtifactError("checksum artifact root must not be a symlink")
    root = root.resolve(strict=True)
    files = regular_tree_files(root)
    if manifest_name not in files or sidecar_name not in files:
        raise Stage1ArtifactError("checksum manifest or sidecar is missing")
    sidecar_lines = files[sidecar_name].read_text(encoding="utf-8").splitlines()
    expected_sidecar = "{}  {}".format(
        sha256_file(files[manifest_name]), manifest_name
    )
    if sidecar_lines != [expected_sidecar]:
        raise Stage1ArtifactError("checksum manifest sidecar mismatch")

    records: Dict[str, str] = {}
    previous = None
    lines = files[manifest_name].read_text(encoding="utf-8").splitlines()
    if not lines:
        raise Stage1ArtifactError("empty checksum manifest")
    for line_number, line in enumerate(lines, start=1):
        match = CHECKSUM_LINE_RE.fullmatch(line)
        if match is None:
            raise Stage1ArtifactError(
                "malformed checksum line {}".format(line_number)
            )
        expected, relative = match.groups()
        _canonical_relative(relative)
        if previous is not None and relative <= previous:
            raise Stage1ArtifactError("checksum paths must be unique and sorted")
        if relative in {manifest_name, sidecar_name}:
            raise Stage1ArtifactError("checksum infrastructure may not self-cover")
        target = root / relative
        if target.is_symlink() or not target.is_file():
            raise Stage1ArtifactError("checksum target missing: {}".format(relative))
        if sha256_file(target) != expected:
            raise Stage1ArtifactError("checksum mismatch: {}".format(relative))
        records[relative] = expected
        previous = relative

    if require_closed_world:
        excluded = {manifest_name, sidecar_name} | set(extra_excluded_names)
        observed = set(files) - excluded
        if observed != set(records):
            raise Stage1ArtifactError(
                "checksum closed world differs; missing={}, unexpected={}".format(
                    sorted(set(records) - observed),
                    sorted(observed - set(records)),
                )
            )
    return records


def artifact_member(path: Path) -> Dict[str, Any]:
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def verify_simple_seal(
    directory: Path,
    *,
    seal_name: str,
    schema_version: str,
    status: str,
    payload_names: Sequence[str],
) -> Dict[str, Any]:
    files = regular_tree_files(directory)
    expected = set(payload_names) | {seal_name}
    if set(files) != expected:
        raise Stage1ArtifactError(
            "sealed member set differs; missing={}, unexpected={}".format(
                sorted(expected - set(files)), sorted(set(files) - expected)
            )
        )
    seal = load_json_strict(files[seal_name])
    if seal.get("schema_version") != schema_version or seal.get("status") != status:
        raise Stage1ArtifactError("artifact seal schema/status mismatch")
    members = seal.get("members")
    if not isinstance(members, dict) or set(members) != set(payload_names):
        raise Stage1ArtifactError("artifact seal member map mismatch")
    for name in payload_names:
        if members[name] != artifact_member(files[name]):
            raise Stage1ArtifactError("artifact seal identity mismatch: {}".format(name))
    return seal


def publish_closed_json_artifact(
    output_dir: Path,
    *,
    report_name: str,
    report: Mapping[str, Any],
    seal_schema: str,
    seal_status: str,
) -> Path:
    """Atomically create a new two-member report+seal directory."""

    output_dir = output_dir.expanduser().absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise Stage1ArtifactError("output already exists: {}".format(output_dir))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".{}.".format(output_dir.name), dir=str(output_dir.parent))
    )
    try:
        report_path = staging / report_name
        report_path.write_bytes(canonical_json_bytes(dict(report)))
        with report_path.open("rb") as stream:
            os.fsync(stream.fileno())
        seal = {
            "schema_version": seal_schema,
            "status": seal_status,
            "members": {report_name: artifact_member(report_path)},
        }
        seal_path = staging / "artifact_seal.json"
        seal_path.write_bytes(canonical_json_bytes(seal))
        with seal_path.open("rb") as stream:
            os.fsync(stream.fileno())
        directory_fd = os.open(str(staging), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(str(staging), str(output_dir))
        parent_fd = os.open(str(output_dir.parent), os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def publish_closed_files_artifact(
    output_dir: Path,
    *,
    payloads: Mapping[str, bytes],
    seal_schema: str,
    seal_status: str,
) -> Path:
    """Atomically publish immutable byte payloads plus one integrity seal.

    This is the multi-payload counterpart of
    :func:`publish_closed_json_artifact`.  Relative names are deliberately
    limited to canonical top-level members: scientific artifact producers do
    not get an implicit way to escape or merge an existing directory.
    """

    output_dir = output_dir.expanduser().absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise Stage1ArtifactError("output already exists: {}".format(output_dir))
    if not payloads:
        raise Stage1ArtifactError("cannot publish an empty artifact")
    if "artifact_seal.json" in payloads:
        raise Stage1ArtifactError("payloads may not supply artifact_seal.json")
    normalized: Dict[str, bytes] = {}
    for name, payload in payloads.items():
        relative = _canonical_relative(name)
        if len(relative.parts) != 1:
            raise Stage1ArtifactError(
                "closed file payload must be top-level: {}".format(name)
            )
        if not isinstance(payload, bytes):
            raise Stage1ArtifactError("artifact payload must be bytes: {}".format(name))
        if name in normalized:
            raise Stage1ArtifactError("duplicate artifact payload: {}".format(name))
        normalized[name] = payload

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".{}.".format(output_dir.name), dir=str(output_dir.parent))
    )
    try:
        for name in sorted(normalized):
            path = staging / name
            with path.open("xb") as stream:
                stream.write(normalized[name])
                stream.flush()
                os.fsync(stream.fileno())
        members = {
            name: artifact_member(staging / name) for name in sorted(normalized)
        }
        seal_path = staging / "artifact_seal.json"
        with seal_path.open("xb") as stream:
            stream.write(
                canonical_json_bytes(
                    {
                        "schema_version": seal_schema,
                        "status": seal_status,
                        "members": members,
                    }
                )
            )
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(str(staging), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(str(staging), str(output_dir))
        parent_fd = os.open(str(output_dir.parent), os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir
