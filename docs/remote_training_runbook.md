# Stage-1 training implementation reference（不再是执行入口）

> 当前唯一执行入口是 `../START_HERE_REMOTE.md` 和
> `STAGE1_AUTONOMY_RUNBOOK.md`。本文件来自 patch06，只保留单机八卡 runner、resume
> 与污染变量的实现说明；其中旧路径、阶段状态、任务顺序和人工裁决点均已被 autonomy
> DAG 取代。发生冲突时，以新 runbook 和注册 verifier receipt 为准。

## Retained patch06 reference: manual 4 × independent 8-GPU jobs

This runbook is the executable contract for the standalone Stage-1 runner.
Each of the four physical machines launches one completely independent
single-node job. There is no cross-machine rendezvous, no Slurm, no `sbatch`,
no `submitit.AutoExecutor`, no nested Dora/torchrun launch, and never a
`WORLD_SIZE=32` process group.

The confirmed target environment is:

```text
conda prefix: <local>/envs/ptc-opd-train-py39-cu121
Python:       3.9.18
PyTorch:      2.1 + CUDA 12.1
GPU:          NVIDIA H20, compute capability 9.0
project root: <local>/ICASSP2027
```

All paths remain command-line arguments. The examples below use the confirmed
base checkout at `<local>/ICASSP2027`, whose
AudioCraft tree is `third_party/audiocraft`. The portable workpack may be copied
as a sibling directory and is therefore configured independently.

## 1. Shell isolation on every machine

Start a fresh `tmux` session for one condition, then remove variables that may
have leaked from an earlier distributed job:

```bash
tmux new -s ptc-stage1
unset MASTER_ADDR MASTER_PORT RANK LOCAL_RANK WORLD_SIZE GROUP_RANK ROLE_RANK
unset LOCAL_WORLD_SIZE ROLE_WORLD_SIZE TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT
unset SLURM_JOB_ID SLURM_PROCID SLURM_LOCALID SLURM_NTASKS
unset OMPI_COMM_WORLD_RANK OMPI_COMM_WORLD_SIZE PMI_RANK PMI_SIZE

export PTC_PROJECT_ROOT=<local>/ICASSP2027
export PTC_WORKPACK=<local>/ICASSP2027_PTC_OPD_A1R2_Workpack_20260814
export PTC_AUDIOCRAFT=${PTC_WORKPACK}/vendor/audiocraft
export PTC_ENV_PREFIX=<local>/envs/ptc-opd-train-py39-cu121
export PTC_MANIFEST=${PTC_WORKPACK}/manifests/musiccaps-v1/train.full.jsonl
export PTC_MODEL_SCALE=small
export PTC_PRIOR_ARTIFACT_DIR=${PTC_WORKPACK}/artifacts/phase_a1_codec_prior_r2
export PTC_CFG_DECISION_DIR=${PTC_WORKPACK}/artifacts/cfg_scale/${PTC_MODEL_SCALE}/decision
: "${PTC_MUSICGEN_SMALL:?first export the exact dereferenced path from START_HERE_REMOTE.md section 2}"
export PTC_STUDENT_CKPT=${PTC_MUSICGEN_SMALL}
export PTC_TEACHER_CKPT=${PTC_MUSICGEN_SMALL}
export PTC_RUN_ROOT=${PTC_WORKPACK}/runs
export PTC_CONSOLE_ROOT=${PTC_WORKPACK}/console_logs
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
mkdir -p "${PTC_RUN_ROOT}" "${PTC_CONSOLE_ROOT}"
set -o pipefail
```

Use the same commands on node-0 through node-3. Machine labels are operational
labels only; each local torchrun assigns ranks 0–7 from scratch.

## 2. Overlay and immutable-input gates

The training entry requires the explicit no-CFG AudioCraft overlay. Preserve
the legacy checkout and create the writable deployment copy inside the
workpack once, on the shared filesystem:

```bash
mkdir -p "${PTC_WORKPACK}/vendor"
test ! -e "${PTC_AUDIOCRAFT}"
cp -a "${PTC_PROJECT_ROOT}/third_party/audiocraft" "${PTC_AUDIOCRAFT}"

"${PTC_ENV_PREFIX}/bin/python" \
  "${PTC_WORKPACK}/scripts/check_audiocraft_overlay.py" \
  --audiocraft-root "${PTC_AUDIOCRAFT}"

git -C "${PTC_AUDIOCRAFT}" apply \
  "${PTC_WORKPACK}/patches/audiocraft/0001-explicit-no-cfg-generation.patch"
```

Then run the patched test described in `docs/nocfg_overlay_notes.md`. The copy
under `vendor/` is a generated deployment artifact and stays inside the one
folder the user will copy; the baseline project checkout remains untouched.

Confirm the manifest is the complete, unsharded file and that every condition
sees the same digest:

```bash
test "$(basename "${PTC_MANIFEST}")" = train.full.jsonl
sha256sum "${PTC_MANIFEST}"
git -C "${PTC_AUDIOCRAFT}" rev-parse HEAD
git -C "${PTC_PROJECT_ROOT}/third_party/audiocraft" status --short
```

Training must not use `train.shard-of-4.node-*.jsonl`. Four machines run four
experimental conditions over the same full prompt manifest; they are not four
quarters of one dataset.

## 3. CPU-only dry run before torchrun

The dry run deliberately does not import AudioCraft, allocate CUDA, or start a
process group. It validates all local input paths, hashes the manifest and both
checkpoint trees, hashes the patched AudioCraft LM source, confirms that the
manifest contains at least one full 16-prompt physical global batch, confirms
2 × 8 × 4 = 64, verifies 10 seconds × 50 Hz = 500 frames, and prints the exact
method-to-loss mapping.

```bash
"${PTC_ENV_PREFIX}/bin/python" "${PTC_WORKPACK}/scripts/train_stage1.py" \
  --dry-run \
  --manifest "${PTC_MANIFEST}" \
  --student-checkpoint "${PTC_STUDENT_CKPT}" \
  --teacher-checkpoint "${PTC_TEACHER_CKPT}" \
  --audiocraft-root "${PTC_AUDIOCRAFT}" \
  --cfg-scale-decision-dir "${PTC_CFG_DECISION_DIR}" \
  --output-dir "${PTC_RUN_ROOT}/dry-run-not-created" \
  --mode ptc50 \
  --codebook-prior-artifact-dir "${PTC_PRIOR_ARTIFACT_DIR}" \
  --seed 2027 \
  --learning-rate 3e-6 \
  --max-optimizer-steps 1000
```

Save the printed JSON beside the run. `audiocraft_imported` must be `false`.
For `uniform100`, `random50`, and `disagreement50`, a prior path may be passed
as a common template but the runner does not pass it to the loss. `codebook100`
and `ptc50` require it.

The prior argument is the complete sealed A1 directory, never the loose
`codec_prior.json` member. The dry run rehashes all four directory members,
checks the 512-clip/Q4/codec/bootstrap contract, and requires the current
checkpoint's `compression_state_dict.bin` to match the codec used by A1.
If that file is AudioCraft's official `pretrained` indirection, the dry run
also resolves the sealed EnCodec commit from the strictly offline Hugging Face
cache and rehashes its complete file set; matching only the small wrapper file
is not sufficient.
Likewise, the selected CFG decision must have been generated for the exact
checkpoint tree, patched AudioCraft source, and loaded T5 identity used here.

## 4. Launch one independent eight-GPU job

Set a globally unique output directory. The runner refuses to overwrite it.
The example below is one PTC small-pilot job:

```bash
export PTC_RUN_ID=pilot.small.ptc50.seed-2027.20260812T120000Z
export PTC_RUN_DIR=${PTC_RUN_ROOT}/${PTC_RUN_ID}

"${PTC_ENV_PREFIX}/bin/torchrun" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  "${PTC_WORKPACK}/scripts/train_stage1.py" \
  --manifest "${PTC_MANIFEST}" \
  --student-checkpoint "${PTC_STUDENT_CKPT}" \
  --teacher-checkpoint "${PTC_TEACHER_CKPT}" \
  --audiocraft-root "${PTC_AUDIOCRAFT}" \
  --cfg-scale-decision-dir "${PTC_CFG_DECISION_DIR}" \
  --output-dir "${PTC_RUN_DIR}" \
  --mode ptc50 \
  --codebook-prior-artifact-dir "${PTC_PRIOR_ARTIFACT_DIR}" \
  --seed 2027 \
  --learning-rate 3e-6 \
  --max-optimizer-steps 1000 \
  --save-every 250 \
  --log-every 1 \
  2>&1 | tee "${PTC_CONSOLE_ROOT}/${PTC_RUN_ID}.console.log"
```

Only change `--mode`, `--seed`, the frozen selected LR, run horizon, and unique
run ID according to `configs/stage1_matrix.yaml`. The five primary mode names
plus one small-only diagnostic are exactly:

| CLI mode | Kernel mode | retention | KL weighting |
|---|---|---:|---|
| `uniform100` | `uniform` | 100% | uniform |
| `codebook100` | `codebook_only` | 100% | perceptual prior |
| `random50` | `random_stratified` | 50% per (sample, codebook) | uniform |
| `prefix50` | `prefix` | earliest 50% per (sample, codebook) | uniform |
| `disagreement50` | `disagreement` | detached-JS top 50% per (sample, codebook) | uniform |
| `ptc50` | `ptc` | same detached-JS gate | perceptual prior |

`prefix50` is permitted only for the MusicGen-small one-seed diagnostic. It is
not a medium/main condition and cannot replace any of the five primary modes.

The optimizer is frozen: AdamW, weight decay 0, betas (0.9, 0.95), epsilon
1e-8, global gradient clipping 1.0. The first 50 completed optimizer updates
warm linearly from zero to the selected LR; all later updates remain at that
LR, regardless of whether the horizon is 500, 1000, or 3000.

The example is intentionally MusicGen-small. Before Phase C, verify and consume
the bundled, byte-pinned MusicGen-medium v3 decision, then set
`PTC_MODEL_SCALE=medium` and update the checkpoint and decision variables
together. Do not regenerate the medium gate unless an explicit adjudication is
issued after verification or identity binding fails. A small-model decision
cannot authorize a medium run, even though both retained gates chose scale 5.

## 5. Why the four accumulation denominators are gated

Every rank consumes physical batch 2; eight ranks form a physical global batch
16; four `DDP.no_sync()` microsteps form effective global batch 64. Each
microstep uses the true global numerator/global denominator reduction and
backpropagates one quarter of that ratio.

That is exactly the whole-window ratio only because fixed 500-frame generation,
the fixed AudioCraft delay mask, and fixed per-(sample, codebook) retained
counts make all four global effective-weight denominators equal. The runner
records all four and fails immediately if they differ beyond relative tolerance
1e-6. It never silently optimizes an average of unequal microbatch ratios.

Do not relax `--denominator-rtol` to rescue a run. A failure means the fixed
length/mask/selector contract was violated and must be diagnosed.

Patch05 froze the Torch-2.1 DDP reducer lifecycle for every research run:
explicit `bucket_cap_mb=25` plus the pinned Torch default first-bucket cap,
`find_unused_parameters=True`, `static_graph=False`, and
`gradient_as_bucket_view=False`.  Bucket sizes, parameter layout, total
gradient bytes, and no-rebuild state are recorded per rank; the flag is used as
a topology anchor, not because unused LM parameters are expected, and the
runner fails if any trainable gradient is missing.  The retained real-H20
patch05 run nevertheless reproduced the same tiny warm/cold student and Adam
drift with this contract live.  Patch06 therefore retains the reducer setting
without claiming it was the root cause, and qualifies checkpoint-to-live plus
cold-to-cold exactness separately from strict warm/cold numerical continuity.
See `docs/NODE3_RESUME_LIFECYCLE_PATCH06.md`.

## 6. Exact resume

Resume into the same existing run directory and pass its **latest committed**
checkpoint directory. A committed checkpoint is a closed `step-NNNNN/`
directory containing exactly `checkpoint.pt` and `SHA256.json`. Before launch,
list the checkpoint directories, inspect the highest step, and set the resume
path explicitly:

```bash
find "${PTC_RUN_DIR}/checkpoints" -mindepth 1 -maxdepth 1 -type d \
  -name 'step-[0-9][0-9][0-9][0-9][0-9]' -print | sort
export PTC_RESUME_CHECKPOINT=${PTC_RUN_DIR}/checkpoints/step-00250
test -f "${PTC_RESUME_CHECKPOINT}/checkpoint.pt"
test -f "${PTC_RESUME_CHECKPOINT}/SHA256.json"
```

The runner independently enumerates the checkpoint root and rejects an older
step even if that step is internally valid. It also rejects partial step
directories, stale staging directories, symlinks, and any unexpected member of
`checkpoints/`; inspect and quarantine such evidence outside the run before
retrying rather than guessing around it.

The runner verifies config, full-manifest, initial student checkpoint, initial
teacher checkpoint, and initial in-memory teacher-state hashes before restoring
student/optimizer/progress. It verifies `global_microstep = optimizer_step × 4`.
The LR is then derived from the restored completed optimizer step, so warmup
continues exactly.  The new process constructs the same fixed initial bucket
layout as the uninterrupted process; do not change DDP flags to silence the
expected no-unused-parameter performance warning.

```bash
"${PTC_ENV_PREFIX}/bin/torchrun" \
  --standalone --nnodes=1 --nproc_per_node=8 \
  "${PTC_WORKPACK}/scripts/train_stage1.py" \
  --manifest "${PTC_MANIFEST}" \
  --student-checkpoint "${PTC_STUDENT_CKPT}" \
  --teacher-checkpoint "${PTC_TEACHER_CKPT}" \
  --audiocraft-root "${PTC_AUDIOCRAFT}" \
  --cfg-scale-decision-dir "${PTC_CFG_DECISION_DIR}" \
  --output-dir "${PTC_RUN_DIR}" \
  --mode ptc50 --codebook-prior-artifact-dir "${PTC_PRIOR_ARTIFACT_DIR}" \
  --seed 2027 --learning-rate 3e-6 --max-optimizer-steps 1000 \
  --save-every 250 --log-every 1 \
  --resume "${PTC_RESUME_CHECKPOINT}"
```

Never resume a failed condition using another condition's checkpoint. Do not
delete the original `FAILED.json`; a successful same-run resume records that
failure hash as superseded, and only the final `DONE.json` is a successful run
commit. Each checkpoint is a closed directory containing `checkpoint.pt` and
`SHA256.json`, published by one directory rename. Create a new run ID if a
non-resumable scientific input changed.

Every launch publishes a new, exclusive log pair and never reopens an earlier
metrics file:

```text
logs/attempt-0000.json
logs/metrics.attempt-0000.jsonl
logs/attempt-0001.json
logs/metrics.attempt-0001.jsonl
...
```

The attempt metadata records whether the launch is initial/resume, its exact
start checkpoint and hashes, restored optimizer/global-microstep progress,
config hash, host, and matching metrics path. Optimizer steps must be strictly
increasing inside each metrics segment. A failed attempt may contain an
uncommitted tail beyond the checkpoint used by the next attempt, so do not
naively concatenate the JSONL files. For a canonical no-duplicate curve, retain
from each non-final attempt only records at or before the next attempt's
`start_optimizer_step`, then append the final attempt. `SEALED.json` records the
path, SHA-256, record count, and step range of every attempt metadata/metrics
file; `DONE.json` commits that seal.

## 7. Seal and verify a copied run

A successful runner now publishes the v5 terminal chain. `SEALED.json` binds
the path, SHA-256, and byte size of `run_manifest.json` and the immutable
`status.json`; inventories every committed checkpoint and its sidecar; checks
the embedded metadata of every checkpoint; and binds every attempt log plus an
optional superseded failure. Every checkpoint's embedded `config_sha256` must
equal the canonical resolved-config hash in `run_manifest.json`, its
optimizer/global-microstep values must match its directory, and its frozen
teacher identity must equal the manifest and final live teacher identities.
`DONE.json`, written last, binds the SHA-256 and byte size of `SEALED.json` and
also repeats the manifest SHA-256 and byte size. Therefore `DONE.json` is the
only success commit; `SEALED.json` without `DONE.json` remains repairable by
the existing exact-resume path and is not success.

Run the closed-world consumer immediately after completion. Store its output
outside the run directory, because an extra file inside the run is correctly
treated as corruption:

```bash
test -f "${PTC_RUN_DIR}/DONE.json"
export PTC_VERIFY_LOG=${PTC_CONSOLE_ROOT}/stage1-verify.json
"${PTC_ENV_PREFIX}/bin/python" \
  "${PTC_WORKPACK}/scripts/verify_stage1_run.py" \
  --run-dir "${PTC_RUN_DIR}" | tee "${PTC_VERIFY_LOG}"
```

The result must have `"status": "verified"`. The consumer rejects symlinks,
unexpected top-level members, partial/unexpected checkpoint members, altered
manifest/status/log bytes, stale or changed checkpoint sidecars/payloads,
checkpoint metadata with a foreign config, and any mismatch in the
`run_manifest → SEALED → DONE` chain. It hashes all checkpoint payloads and
uses mmap-capable `torch.load` for their small metadata dictionaries, so allow
time for full sequential disk reads without requiring another GPU job.

After copying or synchronizing a run to another filesystem, invoke the same
command with `--run-dir` set to the destination and archive that second JSON
beside the transfer log. The manifest contains the original resolved input
paths for provenance, but the verifier intentionally does not dereference
those paths, so verification remains executable after the run directory is
copied. Do not edit paths in `run_manifest.json` after transfer.

## 8. Required remote gates before research runs

Do not execute this section as a hand-written checklist. The authoritative
node-3 procedure is `docs/node3_gate_runbook.md`, whose single orchestrator
implements 15 fail-closed checks and seals every command, return code, stdout,
stderr, runtime audit, and checksum under one UTC evidence directory.

The orchestrator deliberately runs two distinct two-update jobs: an
uninterrupted reference and a same-config job with a dual-key, gate-only stop
after committed step 1, followed by stale-step rejection and resume from the
latest step. Full-rank sample/gate/memory audits are part of sealed metrics;
CPU FP32/BF16/conditioner/no-CFG runtime contracts are part of the sealed run
manifest.  Patch05 additionally seals per-microstep rollout-code hashes and one
fixed DDP reducer identity across all ranks, attempts, and steps. The
qualification calls the resume exact only after finite, layout-aware step-1 and
step-2 student/optimizer/scheduler/eight-rank-RNG canonical hashes match, and
the frozen step-2 loss/gradient tolerance checks pass. It verifies the resumed
run both before and after a real directory copy.  There is no checkpoint-state
`allclose` waiver.

Only `STATUS.json` with `gate_count=15`, `passed_count=15`, all individual
statuses `passed`, plus a successful evidence checksum verification constitutes
the remote gate. The existence of this runner and its CPU tests is not evidence
that AudioCraft/NCCL/BF16/H20 execution passed. A failed or unexecuted remote
gate remains `PENDING`/`NO-GO`; it must never be converted to PASS by editing
the status or regenerating a source manifest in place.
