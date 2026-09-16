# MusicCaps prompt-manifest runbook

This is the first data step. It creates immutable prompt manifests only; it
does not download YouTube audio. Training consumes captions, so missing source
audio is not a blocker for OPD rollouts.

## 1. Download the canonical public CSV once

Run on `node-0` in a directory visible to all four machines:

```bash
export PTC_STORAGE_ROOT=<local>/ICASSP2027/storage
mkdir -p "${PTC_STORAGE_ROOT}/data/musiccaps/source"

curl --fail --location --retry 5 \
  --output "${PTC_STORAGE_ROOT}/data/musiccaps/source/musiccaps-public.csv" \
  https://huggingface.co/datasets/google/MusicCaps/resolve/main/musiccaps-public.csv

sha256sum "${PTC_STORAGE_ROOT}/data/musiccaps/source/musiccaps-public.csv"
```

Record the observed SHA-256. Do not silently replace the CSV after manifests
have been built. MusicCaps publishes YouTube IDs, timestamps, and captions; it
does not redistribute the referenced audio in this CSV.

## 2. Build the frozen split

Use the accepted training environment and the copied workpack:

```bash
export PTC_WORKPACK_ROOT=<local>/ICASSP2027_PTC_OPD_A1R2_Workpack_20260814
export PTC_MANIFEST_ROOT="${PTC_WORKPACK_ROOT}/manifests/musiccaps-v1"

<local>/envs/ptc-opd-train-py39-cu121/bin/python \
  "${PTC_WORKPACK_ROOT}/scripts/build_musiccaps_manifests.py" \
  --musiccaps-csv "${PTC_STORAGE_ROOT}/data/musiccaps/source/musiccaps-public.csv" \
  --output-dir "${PTC_MANIFEST_ROOT}" \
  --split-seed 2701 \
  --dev-size 300 \
  --probe-size 256 \
  --test-size 500 \
  --near-duplicate-jaccard 0.90
```

The script fails unless the source has the canonical `5521 = 2663 train +
2858 eval` rows. It removes connected duplicate groups formed by shared source
ID, exact normalized caption, or three-token-shingle Jaccard at least `0.90`.
If a group crosses the official AudioSet train/eval boundary, its train rows
are not allowed back into training.

Expected files are:

```text
train.full.jsonl
dev.full.jsonl
phenomenon_probe.dev.jsonl
test.full.jsonl
duplicate_groups.json
split_report.json
```

The builder refuses a partial overwrite by default. `--force` is for a
documented rebuild only; a forced rebuild invalidates downstream artifacts
unless every resulting hash is unchanged.

## 3. Seal and compare

```bash
sha256sum "${PTC_MANIFEST_ROOT}"/* | sort
wc -l \
  "${PTC_MANIFEST_ROOT}/train.full.jsonl" \
  "${PTC_MANIFEST_ROOT}/dev.full.jsonl" \
  "${PTC_MANIFEST_ROOT}/phenomenon_probe.dev.jsonl" \
  "${PTC_MANIFEST_ROOT}/test.full.jsonl"
```

Copy `split_report.json` and the `sha256sum` output into the experiment ledger.
Every training condition reads the complete `train.full.jsonl` with the same
hash. The four machines do not receive four different training shards.

The downloaded public CSV may remain under the external storage root, but every
new split/report file is deliberately written inside the portable workpack.

Do not inspect or use `test.full.jsonl` for method, CFG-scale, learning-rate,
checkpoint, or metric selection. Stage 1 uses only the training, development,
and named phenomenon-probe manifests.
