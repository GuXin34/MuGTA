#!/usr/bin/env python3
"""Build deterministic, leakage-audited MusicCaps prompt manifests.

The script reads the public MusicCaps CSV.  It never downloads audio or writes
outside ``--output-dir``.  Development prompts come only from AudioSet-train
rows and test prompts only from AudioSet-eval rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


SCHEMA_VERSION = "ptc-opd-musiccaps-manifest-v1"
CANONICAL_TOTAL = 5521
CANONICAL_TRAIN = 2663
CANONICAL_EVAL = 2858
REQUIRED_COLUMNS = {
    "ytid",
    "start_s",
    "end_s",
    "caption",
    "is_audioset_eval",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: int, namespace: str, sample_id: str) -> Tuple[str, str]:
    payload = f"{seed}\0{namespace}\0{sample_id}".encode("utf-8")
    return sha256_bytes(payload), sample_id


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


def normalize_caption(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def caption_shingles(normalized: str, width: int = 3) -> Set[Tuple[str, ...]]:
    tokens = normalized.split()
    if len(tokens) < width:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1)}


def shingle_jaccard(left: Set[Tuple[str, ...]], right: Set[Tuple[str, ...]]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


@dataclass(frozen=True)
class Row:
    source_index: int
    sample_id: str
    ytid: str
    start_s: str
    end_s: str
    caption: str
    aspect_list: str
    is_audioset_eval: bool
    normalized_caption: str
    raw_row_sha256: str

    def manifest_record(self) -> Dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "prompt": self.caption,
            "ytid": self.ytid,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "aspect_list": self.aspect_list,
            "is_audioset_eval": self.is_audioset_eval,
            "source_row_sha256": self.raw_row_sha256,
        }


def canonical_number(value: str) -> str:
    number = float(value)
    if not (number == number and abs(number) != float("inf")):
        raise ValueError(f"non-finite timestamp {value!r}")
    return format(number, ".9g")


def load_rows(path: Path) -> List[Row]:
    rows: List[Row] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"MusicCaps CSV is missing columns: {sorted(missing)}")
        for source_index, raw in enumerate(reader, start=2):
            ytid = (raw.get("ytid") or "").strip()
            caption = (raw.get("caption") or "").strip()
            if not ytid or not caption:
                raise ValueError(f"row {source_index}: empty ytid or caption")
            start_s = canonical_number(raw.get("start_s") or "")
            end_s = canonical_number(raw.get("end_s") or "")
            if float(end_s) <= float(start_s):
                raise ValueError(f"row {source_index}: end_s must exceed start_s")
            is_eval = parse_bool(raw.get("is_audioset_eval") or "")
            sample_id = f"musiccaps:{ytid}:{start_s}-{end_s}"
            canonical_raw = json.dumps(raw, sort_keys=True, ensure_ascii=False).encode("utf-8")
            normalized = normalize_caption(caption)
            if not normalized:
                raise ValueError(f"row {source_index}: caption normalizes to empty")
            rows.append(
                Row(
                    source_index=source_index,
                    sample_id=sample_id,
                    ytid=ytid,
                    start_s=start_s,
                    end_s=end_s,
                    caption=caption,
                    aspect_list=(raw.get("aspect_list") or "").strip(),
                    is_audioset_eval=is_eval,
                    normalized_caption=normalized,
                    raw_row_sha256=sha256_bytes(canonical_raw),
                )
            )
    if not rows:
        raise ValueError("MusicCaps CSV contains no rows")
    return rows


def duplicate_groups(rows: Sequence[Row], near_threshold: float) -> List[List[int]]:
    if not 0.0 <= near_threshold <= 1.0:
        raise ValueError("near-duplicate threshold must lie in [0, 1]")
    union_find = UnionFind(len(rows))
    by_source: Dict[str, int] = {}
    by_exact: Dict[str, int] = {}
    postings: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    shingles: List[Set[Tuple[str, ...]]] = []

    for index, row in enumerate(rows):
        previous_source = by_source.get(row.ytid)
        if previous_source is not None:
            union_find.union(index, previous_source)
        else:
            by_source[row.ytid] = index

        previous_exact = by_exact.get(row.normalized_caption)
        if previous_exact is not None:
            union_find.union(index, previous_exact)
        else:
            by_exact[row.normalized_caption] = index

        row_shingles = caption_shingles(row.normalized_caption)
        candidates: Set[int] = set()
        for shingle in row_shingles:
            candidates.update(postings[shingle])
        for candidate in candidates:
            # Jaccard cannot reach the threshold when set-size ratios are too
            # different; skip those pairs before constructing an intersection.
            smaller = min(len(row_shingles), len(shingles[candidate]))
            larger = max(len(row_shingles), len(shingles[candidate]))
            if larger and smaller / larger < near_threshold:
                continue
            if shingle_jaccard(row_shingles, shingles[candidate]) >= near_threshold:
                union_find.union(index, candidate)
        for shingle in row_shingles:
            postings[shingle].append(index)
        shingles.append(row_shingles)

    grouped: Dict[int, List[int]] = defaultdict(list)
    for index in range(len(rows)):
        grouped[union_find.find(index)].append(index)
    return sorted(grouped.values(), key=lambda group: min(rows[i].sample_id for i in group))


def representatives(rows: Sequence[Row], groups: Sequence[Sequence[int]]) -> Tuple[List[Row], List[Dict[str, object]]]:
    kept: List[Row] = []
    audit: List[Dict[str, object]] = []
    for group in groups:
        members = [rows[index] for index in group]
        eval_members = [row for row in members if row.is_audioset_eval]
        eligible = eval_members or members
        representative = min(eligible, key=lambda row: row.sample_id)
        kept.append(representative)
        if len(members) > 1:
            audit.append(
                {
                    "representative": representative.sample_id,
                    "members": sorted(row.sample_id for row in members),
                    "spans_official_train_eval": len({row.is_audioset_eval for row in members}) > 1,
                }
            )
    return kept, audit


def choose(rows: Sequence[Row], count: int, seed: int, namespace: str) -> List[Row]:
    if count < 0 or count > len(rows):
        raise ValueError(f"cannot choose {count} rows from pool of {len(rows)} for {namespace}")
    return sorted(rows, key=lambda row: stable_key(seed, namespace, row.sample_id))[:count]


def encoded_jsonl(records: Iterable[Dict[str, object]]) -> bytes:
    lines = [json.dumps(record, sort_keys=True, ensure_ascii=False) for record in records]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def write_new(path: Path, payload: bytes, force: bool) -> str:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return sha256_bytes(payload)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--musiccaps-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split-seed", type=int, default=2701)
    parser.add_argument("--dev-size", type=int, default=300)
    parser.add_argument("--probe-size", type=int, default=256)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--near-duplicate-jaccard", type=float, default=0.90)
    parser.add_argument("--allow-noncanonical-counts", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def build(args: argparse.Namespace) -> Dict[str, object]:
    input_path = args.musiccaps_csv.resolve()
    output_dir = args.output_dir.resolve()
    rows = load_rows(input_path)
    raw_train = sum(not row.is_audioset_eval for row in rows)
    raw_eval = len(rows) - raw_train
    if not args.allow_noncanonical_counts:
        observed = (len(rows), raw_train, raw_eval)
        expected = (CANONICAL_TOTAL, CANONICAL_TRAIN, CANONICAL_EVAL)
        if observed != expected:
            raise ValueError(
                f"noncanonical MusicCaps counts: observed {observed}, expected {expected}; "
                "use --allow-noncanonical-counts only for a documented test fixture"
            )

    groups = duplicate_groups(rows, args.near_duplicate_jaccard)
    unique_rows, duplicate_audit = representatives(rows, groups)
    train_pool = [row for row in unique_rows if not row.is_audioset_eval]
    eval_pool = [row for row in unique_rows if row.is_audioset_eval]

    development = choose(train_pool, args.dev_size, args.split_seed, "development")
    development_ids = {row.sample_id for row in development}
    training = [row for row in train_pool if row.sample_id not in development_ids]
    test = choose(eval_pool, args.test_size, args.split_seed, "test")
    probe = choose(development, args.probe_size, args.split_seed, "phenomenon-probe")

    selected = training + development + test
    if len({row.sample_id for row in selected}) != len(selected):
        raise AssertionError("sample_id overlap survived split construction")
    if len({row.normalized_caption for row in selected}) != len(selected):
        raise AssertionError("exact normalized-caption overlap survived deduplication")
    if any(row.is_audioset_eval for row in training + development):
        raise AssertionError("AudioSet-eval row leaked into train/development")
    if any(not row.is_audioset_eval for row in test):
        raise AssertionError("AudioSet-train row leaked into test")

    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "train.full.jsonl": encoded_jsonl(row.manifest_record() for row in sorted(training, key=lambda x: x.sample_id)),
        "dev.full.jsonl": encoded_jsonl(row.manifest_record() for row in sorted(development, key=lambda x: x.sample_id)),
        "phenomenon_probe.dev.jsonl": encoded_jsonl(row.manifest_record() for row in sorted(probe, key=lambda x: x.sample_id)),
        "test.full.jsonl": encoded_jsonl(row.manifest_record() for row in sorted(test, key=lambda x: x.sample_id)),
        "duplicate_groups.json": (json.dumps(duplicate_audit, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"),
    }
    planned_paths = [output_dir / name for name in payloads]
    planned_paths.append(output_dir / "split_report.json")
    if not args.force:
        existing = [str(path) for path in planned_paths if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to create a partial manifest set; existing outputs: "
                + ", ".join(existing)
            )
    hashes = {name: write_new(output_dir / name, payload, args.force) for name, payload in payloads.items()}

    report: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "input_csv": str(input_path),
        "input_csv_sha256": sha256_file(input_path),
        "split_seed": args.split_seed,
        "near_duplicate_jaccard": args.near_duplicate_jaccard,
        "raw_counts": {"all": len(rows), "audioset_train": raw_train, "audioset_eval": raw_eval},
        "deduplicated_pool_counts": {"audioset_train": len(train_pool), "audioset_eval": len(eval_pool)},
        "output_counts": {
            "train.full.jsonl": len(training),
            "dev.full.jsonl": len(development),
            "phenomenon_probe.dev.jsonl": len(probe),
            "test.full.jsonl": len(test),
        },
        "output_sha256": hashes,
        "duplicate_group_count": len(duplicate_audit),
        "cross_official_split_duplicate_group_count": sum(
            bool(group["spans_official_train_eval"]) for group in duplicate_audit
        ),
    }
    report_payload = (json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    report["split_report_sha256"] = write_new(output_dir / "split_report.json", report_payload, args.force)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build(args)
    except (OSError, ValueError, AssertionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
