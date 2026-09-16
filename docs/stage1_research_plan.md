# Stage 1 research plan: finding and testing the PTC-OPD phenomenon

**Freeze candidate:** 2026-08-12  
**Scope:** MusicGen only; four independent machines, eight GPUs per machine  
**Purpose:** turn the central claim into a falsifiable sequence of measurements,
correctness checks, pilots, and three-seed experiments.

The software, hardware, evaluator, and FAD acceptance checks are treated as
passed prerequisites. This document does not reopen the environment audit.

## 1. Decision to be earned

The proposed paper may claim that uniform OPD is a poor allocation of
supervision over MusicGen's time-codebook lattice only if Stage 1 establishes
all three links below:

1. cumulative EnCodec codebooks make reproducibly unequal perceptual marginal
   contributions;
2. a CFG teacher and the no-CFG student disagree non-uniformly over valid
   `(time, codebook)` positions on student rollouts;
3. selecting the high-value half of those positions transfers the teacher gain
   at least as well as a matched random half, while retaining most of the
   uniform-OPD quality gain.

The first two links are observations. The third is the causal intervention.
Maps alone cannot support an effectiveness claim, and a favorable pilot alone
cannot support the proposed mechanism if the maps are uniform.

The primary token score and loss are frozen as

```text
r_tq = JS(stopgrad(p_T,tq), stopgrad(p_S,tq))
g_btq = TopRho_t(r_btq; rho=0.5)  # independently inside every (sample b, codebook q)

L_PTC = sum_btq g_btq * a_q * KL(p_T,btq || p_S,btq)
        / sum_btq g_btq * a_q
```

The selector and the codebook prior have deliberately separate roles. The
selector ranks detached JS only and always keeps the same prescribed number of
time positions in each `(sample, codebook)` group. The perceptual prior `a_q`
enters only as an effective-weight-normalized KL weight; it never changes the
primary ranking. A global selector based on `a_q*JS` is retained as a one-seed
diagnostic, not as the proposed method.

All probabilities and divergences are evaluated in float32 at frozen
distillation temperature `tau=1`. Padding, special
positions, and positions invalid under MusicGen's delay pattern are excluded.
The teacher is the frozen initialization checkpoint evaluated with CFG; the
student is evaluated without CFG on trajectories sampled by the current
student. The initial KL direction is forward KL.

## 2. Immutable data contract

Before the first scientific run, build and hash these manifests:

| Manifest | Unit and size | Permitted use |
|---|---:|---|
| `train.full.jsonl` | all eligible training prompts | every training condition |
| `dev.full.jsonl` | 300 held-out prompts | tuning, gates, and pilot evaluation |
| `phenomenon_probe.dev.jsonl` | 256 fixed prompts drawn from development | disagreement/loss maps only |
| `a1-r2/codec_calibration.train.jsonl` | 512 deterministic, input-eligible 10 s excerpts selected after auditing all 7,994 original FMA-small candidates | perceptual codebook marginal only |
| `test.full.jsonl` | 500 fixed test prompts | untouched until method freeze |

The split seed is `2701`. Exact caption duplicates, near duplicates, and common
source IDs must not cross train/development/test. The calibration set cannot
contain development or test source IDs. The probe manifest is a named subset
of development and is never used as training input.

The default codec-calibration source is the official `mdeff/fma` FMA-small
archive: 8,000 approximately 30-second tracks. Its required archive SHA-1 is
`ade154f733639d52e35e32f5593efe5be76c6d70`. After verifying that archive,
decode all 7,994 candidates and derive each candidate's fixed 10-second start
offset from a domain-separated SHA-256 modulo the decoded valid start range.
Run the exact AudioCraft 32-kHz mono conversion and frozen A1-R2 eligibility
test on every deterministic segment; record every rejected track and reason.
Only then rank eligible track IDs by
`SHA256("ptc-opd-codec-cal-v1|2701|" + track_id)` and take the first 512. The
manifest records the source audio SHA-256, decoded duration, segment offset,
extracted PCM SHA-256, and eligibility identity for every selected item; its
paired report audits all 7,994 candidates. No MusicCaps test audio,
AudioSet-eval audio, development audio, or generated test audio may estimate
`a_q`.

Every independent training job reads the complete `train.full.jsonl` with the
same SHA-256. A copied local manifest is acceptable only when its hash matches.
The distributed sampler may divide minibatches among the eight ranks *inside a
single job*; that is ordinary data parallelism and is not a four-machine data
split.

Files named `train.shard-of-4.node-{0..3}.jsonl` are prohibited as training
inputs. Four-way shards may be created only for parallel preprocessing,
frozen-checkpoint generation, or offline evaluation. Such a task is complete
only after the four outputs are reunited and their ID multiset is proven equal
to the unsharded source manifest.

## 3. Phase A — phenomenon discovery

Phase A runs with `facebook/musicgen-small` and the exact EnCodec/tokenizer
revision intended for the main model. It consumes no test prompts.

### A1. Progressive codebook perceptual marginal

For every calibration clip, encode once to canonical EnCodec codes before
MusicGen delay-pattern expansion. Detect `Q` at runtime and preserve the
original codebook identity. Decode cumulative codebook prefixes:

```text
x_hat^(0): waveform silence, zeros_like(x), at x's aligned analysis length
x_hat^(q): decode codebooks 0,...,q-1 from the same encoded codes, then crop
           to x's original aligned waveform length, q in 1,...,Q
```

The silence baseline is defined in waveform space; do not obtain it by decoding
a zero latent because decoder bias and wrapper behavior can differ. Every
`q>=1` reconstruction reuses the single canonical encoding of that clip; it is
not re-encoded recursively. No codebook may be reordered by energy, index, or observed performance. Audio
is converted once to the codec's native sample rate and channel count; no
post-reconstruction loudness normalization is applied.

The primary distance is a fixed multi-resolution STFT distance:

```text
d(x,y) = mean over FFT sizes {512,1024,2048} of
         [spectral-convergence(x,y) + log-magnitude-L1(x,y)]
hop      = FFT / 4
window   = FFT
epsilon  = 1e-7

Delta_iq = d(x_i, x_hat_i^(q-1)) - d(x_i, x_hat_i^(q))
Delta_q  = mean_i Delta_iq
a_q      = max(Delta_q, 1e-8) / sum_j max(Delta_j, 1e-8)
```

Persist the per-clip `Delta_iq` in gzip-compressed JSONL, the aggregate raw and clipped `Delta_q`, and
the sum-to-one `a_q`. Its global scale would cancel in the effective-weight
denominator, but this representation is canonical. Do not tune this metric against MuQ, CLAP, FAD, or the
held-out test set.

Primary diagnostics are:

- total variation from uniform, `TV(a,u)=0.5*sum_q |a_q-1/Q|`;
- largest-to-smallest positive marginal ratio;
- rank agreement between two split halves defined by calibration-item hash;
- a clip-level bootstrap confidence interval with 10,000 replicates and seed
  `4701`.

### A2. Teacher/student disagreement map

Select the teacher CFG scale on all 300 prompts in `dev.full.jsonl` from
`{2.0, 3.0, 5.0}` using the frozen teacher-anchor rule in Section 5, then
freeze it for every MusicGen-small method. Generation uses a matched sample-ID-derived seed
for the explicit no-CFG and three CFG anchors, 10-second raw float WAVs, and no
loudness normalization. The immutable decision and sidecar are produced by
`scripts/cfg_scale_gate.py`; A2 and training consume that artifact rather than
an unbound hand-entered scale.
`CFG=1` is used only as a correctness control.

The decision is bound to the exact model checkpoint, patched AudioCraft tree,
and loaded T5 identity. Before the medium scale-up gate, repeat the identical
development procedure with the frozen MusicGen-medium checkpoint and seal a
separate medium decision. Reusing the numeric scale selected on small without
that model-specific gate is forbidden; the training runner verifies the exact
checkpoint/source/T5 binding.

For each of the 256 probe prompts, draw two 10-second no-CFG trajectories from
the pretrained, pre-OPD student with generation seeds `31001` and `31002`. Evaluate the
frozen CFG teacher and no-CFG student on the identical student prefix. Record,
for valid cells only:

- symmetric JS and forward KL;
- codebook ID and unexpanded codec-frame index;
- temporal decile;
- teacher entropy, student entropy, and sampled-token log probabilities;
- `a_q`, detached `JS_tq`, and whether the cell belongs to the JS top 50%
  within its own `(sample,codebook)` group.

Persist dense scalar maps as gzip-compressed JSONL or reduced CSV tables, not
Parquet (the frozen training environment does not require PyArrow) and not
full-vocabulary logits. A
fixed audit subset of 16 prompts may retain top-32 teacher/student logits and
token IDs for debugging.

Aggregate each codebook by its mean per valid cell so that delay-pattern count
differences do not masquerade as disagreement differences. Report codebook,
temporal-decile, prompt, and joint `(decile,q)` summaries. Bootstrap by prompt,
with the two rollout seeds nested within prompt, for 10,000 replicates using
seed `4702`.

### A3. Uniform-OPD allocation and observed loss mass

Measure two quantities and keep their names distinct:

1. **nominal uniform allocation:** the normalized count of valid cells receiving
   an equal coefficient under uniform OPD;
2. **observed loss mass:** each cell's forward-KL contribution divided by the
   total forward KL on that sequence.

For each sequence and for every codebook/temporal decile, report:

```text
U(group) = valid cells in group / all valid cells
W_a(q)   = a_q * N_q / sum_j (a_j * N_j)
K(group) = sum KL in group / sum KL
C50_JS,bq = sum_t g_btq * JS_btq / sum_t JS_btq
C50_KL,bq = sum_t g_btq * KL_btq / sum_t KL_btq
```

Also report `TV(U_q,W_a(q))`, every codebook's `C50_JS,bq` and `C50_KL,bq`,
their prompt-bootstrap aggregates, and uniform loss mass by within-`(b,q)` JS
quantile. The first comparison isolates codebook weighting; `C50_JS` isolates
within-codebook temporal targetability; `C50_KL` checks whether that selector
also captures the actual forward-KL numerator. Do not collapse them into a
global `a_q*JS` ranking in the primary analysis.
This analysis makes the exact statement defensible: uniform OPD assigns equal
nominal coefficients to valid positions even when their perceptual marginals
and teacher/student disagreement are unequal. It must not be paraphrased as
"half of the compute is saved" because the shared Transformer forward remains.

## 4. Phase B — MusicGen-small correctness and causal pilot

### B1. Mandatory correctness suite

All tests below must pass before comparing methods:

1. With identical teacher/student weights and `CFG=1`, float32 mean KL is below
   `1e-6` and maximum valid-cell KL is below `1e-5` on the fixed audit batch.
2. With every mask enabled and all `a_q` equal, the new loss reproduces uniform OPD:
   relative loss error below `1e-6` and relative gradient error below `1e-5`.
3. With all `a_q` equal, `ptc50` and `disagreement50` have identical gates,
   loss, and gradients within the same tolerances.
4. `random50` and the JS gate select the same `k_bq` for every `(sample,
   codebook)`; ties in the JS gate are deterministic.
5. Changing padding token values cannot change the loss; invalid logits may be
   NaN without contaminating the result, and every invalid delay-pattern cell
   has exactly zero weight and zero gradient.
6. Encode → pattern build → pattern revert preserves `(frame,q)` identity for
   every valid audit token.
7. The frozen teacher's parameter SHA-256 is identical before and after a
   two-update distributed smoke run.
8. The same seed and manifest produce identical first-batch IDs and gates
   before and after resume. Resume accepts only the latest complete committed
   checkpoint in the same run, creates a new attempt log pair, and leaves every
   earlier attempt file byte-identical.
9. The eight-rank loss and gradient match a concatenated single-process
   reference; DDP reduces global numerators and denominators, never a mean of
   unequal per-rank means.
10. Under the fixed Torch-2.1 initial-bucket policy, every resumed rank restores
    finite/layout-aware student and optimizer state exactly before its first
    forward; two independent cold resumes from the same step-1 seed have
    bit-exact student, optimizer, scheduler, eight-rank RNG, rollout codes and
    reducer identity.  Warm-uninterrupted versus cold-resumed floating state
    also passes the frozen field-specific ULP/absolute/relative-L2 continuity
    contract while all control-plane fields remain exact.
11. A 500-step PTC stability run has no NaN/Inf, no silent zero-selected batch,
   and at least 5% peak-memory margin.

A failed correctness item stops scientific interpretation. Fixing an
implementation defect requires rerunning every downstream artifact made with
that code revision.

### B2. One-seed small pilot

Before the component pilot, freeze the learning rate with a uniform-only
500-update development sweep over the predeclared small grid in the Stage-1
matrix at train seed `2027`. Select the highest `Q_dev` setting that passes the
CLAP and stability guardrails; ties prefer the lower learning rate. This sweep
and the CFG-scale selection are the only Stage-1 hyperparameter selections,
and both are sealed before the five primary component jobs plus the small-only
prefix diagnostic start. The objective is pure
OPD: it is not mixed with CE, GRPO, RSFT, or another loss. Its coefficient is
fixed at `1.0` and is not tuned because under a single-loss objective it is
confounded with learning rate.

All jobs use AdamW with `betas=(0.9,0.95)`, `eps=1e-8`, and zero weight decay,
plus global gradient-norm clipping at `1.0`. The selected learning rate warms
linearly from zero over the first 50 optimizer updates and then stays constant.
The schedule is deliberately horizon-independent, so a 500-update sweep, a
1,000-update pilot, and a 3,000-update primary run do not silently optimize
different cosine schedules. Zero weight decay keeps the parameter update tied
to the declared pure-OPD objective rather than adding an unreported shrinkage
term.

Run six independent 1,000-update jobs with train seed `2027`. The sixth is a
small-only diagnostic frozen before observing pilot outcomes because A2 showed
a strong early-time disagreement gradient:

| Method | Selector | Codebook loss weight |
|---|---|---|
| `uniform100` | all valid cells | `1` |
| `codebook100` | all valid cells | perceptual `a_q` |
| `random50` | random `k_bq` time cells inside each `(b,q)` | `1` |
| `prefix50` | earliest `k_bq` valid time cells inside each `(b,q)` | `1` |
| `disagreement50` | JS top-`k_bq` inside each `(b,q)` | `1` |
| `ptc50` | the identical JS top-`k_bq` gate | perceptual `a_q` |

Every job uses the same prompt count, valid rollout-token budget, effective
global batch (`64`), optimizer updates, and checkpoint opportunities
`{0,250,500,1000}`. For every sample and codebook, set
`k_bq=ceil(0.5*N_bq)`. Random-50 keeps exactly `k_bq` uniformly sampled valid
time cells; disagreement-50 and PTC keep exactly `k_bq` highest detached-JS
cells and share an identical gate. Random masks are deterministically derived
from recorded run/step/sample IDs and a separate seed namespace, not an
unrecorded global RNG state. All selected losses are normalized by their selected effective weight.
This gives clean component contrasts: random versus disagreement isolates
targeting, prefix versus disagreement checks that JS targeting is not merely an
early-token heuristic, and disagreement versus PTC isolates the codebook prior.
`prefix50` is diagnostic-only and is not added to the medium primary matrix.

Evaluate checkpoints on 128 fixed development prompts with two matched sample
seeds. Intermediate checkpoints are learning curves; the gate is evaluated at
the fixed final step `1000`. Use MuQ quality, Audiobox Aesthetics, music-CLAP relevance, MERT
prompt-level diversity, and FAD as a pipeline check. No test prompt is touched,
and this one-seed pilot produces a go/no-go decision rather than a paper-level
significance claim.

## 5. Predeclared falsification and pivot rules

All confidence intervals in this section are 95% intervals from the bootstrap
specified above. Thresholds are evaluated on development/calibration data,
never on test data.

### Phenomenon gates

- **P1, perceptual non-uniformity (A1-R2):** pass only if the lower confidence
  bound of `TV(a,u)` exceeds `0.05`, the two split halves have Kendall tau-b at
  least the exact threshold `2/3`, and the maximum leave-one-out prior TV is at
  most `0.05`. Median and 1%-trimmed-mean rankings are sensitivity checks; each
  must have Kendall tau-b at least `2/3` versus the arithmetic-mean primary
  ranking. At `N=512`, 1% trimming removes `floor(0.01*N)=5` observations from
  each tail independently for each codebook. If P1 fails, stop before A2 and
  remove the perceptual codebook prior from the headline method; do not describe
  codebook inequality as established.
- **P2, targetable disagreement:** pass only if the lower confidence bound of
  prompt-aggregated `C50` exceeds `0.60`, where each top half is selected by JS
  independently inside `(sample,codebook)`. Report every `C50_q` and temporal
  decile as descriptive decompositions. If P2 fails, top-50% targeting has no
  measured substrate and PTC-OPD stops.
- **P3, uniform-allocation mismatch:** pass only if the lower confidence bound
  of `TV(U_q,W_a(q))` exceeds `0.05` **and** the lower confidence bound of
  aggregate `C50_KL` exceeds `0.60`. If P1 passes but P3 fails, retain the
  reconstruction/JS observations but do not claim that they imply materially
  misallocated uniform-OPD loss mass on the actual valid-token lattice.

No required sign is imposed on the correlation between `a_q` and per-codebook
JS: complementarity and redundancy are both reportable. It also does not affect
the primary selector. Raw negative perceptual marginals must be shown rather
than hidden by clipping.

### Teacher and pilot gates

For development gating, standardize MuQ MI, Audiobox CE, and Audiobox PQ using
the no-CFG base samples. Define `Aesthetic` as the equal average of the two
Audiobox standardized paired changes, and define

```text
Aesthetic(m) = 0.5 * Delta z_Audiobox-CE(m vs base)
             + 0.5 * Delta z_Audiobox-PQ(m vs base)

Q_dev(m) = 0.5 * Delta z_MuQ-MI(m vs base)
         + 0.5 * Aesthetic(m)
         = 0.5 * Delta z_MuQ-MI(m vs base)
         + 0.25 * Delta z_Audiobox-CE(m vs base)
         + 0.25 * Delta z_Audiobox-PQ(m vs base).
```

The expanded `0.50/0.25/0.25` expression is exactly the predeclared
`0.5 MuQ + 0.5 Aesthetic` rule, not a changed metric. The CFG-scale bootstrap
resamples paired prompts 10,000 times with seed `4703`. Among candidates that
pass all gates, select the highest `Q_dev`; only an exact numerical tie prefers
the lower scale.

Music-CLAP is a guardrail rather than part of this composite. The CFG teacher
gate passes when `Q_dev(teacher)>0`, the paired-bootstrap probability of a
positive quality difference is at least `0.90`, and its CLAP change is no worse
than `-0.10` base standard deviations. If no CFG candidate passes, audit the
teacher construction once; if it still fails, OPD has no validated teacher
advantage and the study stops.

Given a positive uniform gain, define

```text
retention = Q_dev(PTC) / Q_dev(uniform).
```

The small pilot passes only if all of the following hold at the predeclared
checkpoint-selection step:

- uniform OPD has positive `Q_dev` relative to base;
- PTC retention is at least `0.80`;
- PTC quality is no worse than random-50% by more than `0.10` base standard
  deviations;
- PTC prompt-level MERT diversity is higher than uniform OPD in point estimate;
- disagreement has higher `Q_dev` than prefix-50% in point estimate; otherwise
  the temporal selector is not claimed to outperform a trivial early-position
  control;
- PTC has finite metrics and stable optimization.

One rerun is permitted only for a documented implementation or infrastructure
failure. Hyperparameter fishing is not a rerun reason. If PTC and random remain
indistinguishable, ordinary loss reweighting is not marketed as a new method;
the planned pivot is codec-geometry-aware OPD using the same instrumentation.

## 6. Phase C — MusicGen-medium scale-up gate

Scale up only after every correctness item and P1--P3 pass and the small pilot
meets its gate. Reuse `a_q` only if small and medium report the identical codec
revision and codebook configuration; otherwise recompute it.

First verify the bundled, byte-pinned MusicGen-medium v3 decision and bind it
to the exact medium checkpoint, AudioCraft tree, prompt manifest, and T5
identity. Rerun the `{2,3,5}` medium gate only after explicit adjudication if
that verification or binding fails. All primary jobs must consume the medium
decision; the small decision cannot authorize a medium checkpoint even though
both retained decisions chose scale 5.

Run one `uniform100` and one `ptc50` medium job for 250 updates. After 50
warmup updates, time 200 complete updates including rollout, teacher query,
backward, and eight-GPU communication. The medium gate requires:

- standalone `WORLD_SIZE=8` on one physical machine per job;
- identical data-manifest and frozen-teacher hashes across the two jobs;
- no NaN/Inf or zero-selected batch;
- at least 5% peak-memory margin on every rank;
- P90 complete-step time and its rollout/teacher/backward/communication
  decomposition recorded, with PTC no more than 1.25 times uniform;
- the all-selected/equal-weight equivalence check repeated on medium;
- the initial medium disagreement probe satisfies `C50>0.60` in point estimate;
- checkpoint 250 decodes valid 10-second audio for all fixed smoke prompts.

Failure for memory or throughput permits one engineering adjustment applied to
*all* medium conditions (physical batch/gradient accumulation, activation
checkpointing, or logging frequency). It cannot change the effective global
batch, loss definition, prompt budget, or retention.

## 7. Phase D — three-seed main experiment

The main matrix is five methods × training seeds `{2027,2028,2029}` = 15
independent MusicGen-medium jobs. Each job runs 3,000 optimizer updates with
checkpoints at `{0,250,500,1000,2000,3000}` and effective global batch 64. It
uses the same frozen teacher, model-specific sealed CFG scale, fixed loss coefficient `1.0`, learning rate,
generation policy, and complete `train.full.jsonl` hash.

Primary comparisons are `ptc50` versus `uniform100` and `ptc50`
versus `random50`. Codebook-only and disagreement-only are the component
ablations. Base no-CFG and the frozen CFG teacher are inference-only anchors.
No off-policy KD, alternate KL direction, or retention sweep may displace a
missing primary seed. After the primary matrix is sealed, one seed may compare
the frozen within-`(sample,codebook)` JS selector with a global `a_q*JS`
selector; that result is diagnostic and cannot redefine the main method.

The same post-primary diagnostic wave must include one
`shuffled_prior_ptc50` control. It applies a single seed-predeclared permutation
of the frozen `a_q` across codebook IDs while keeping the PTC gate, prompt order,
optimizer settings, and normalization identical. This is not another tuned
method: it tests whether any non-uniform/early-codebook bias would suffice.
If PTC does not beat this control, the paper may report non-uniform weighting
but cannot attribute the gain specifically to the progressive-reconstruction
prior. A reversed-prior control may be reported descriptively under the same
one-seed diagnostic budget, but cannot replace the shuffled control.

### Four-machine manual schedule

`node-0` through `node-3` are operational labels, not distributed ranks. Each
table cell is a separate one-machine/eight-GPU job. A wave advances only after
all its successful jobs have sealed their run artifacts; a failed job resumes
with the same run ID, seed, and manifest.

| Wave | node-0 | node-1 | node-2 | node-3 |
|---|---|---|---|---|
| 1 | uniform100 / 2027 | codebook100 / 2027 | random50 / 2027 | disagreement50 / 2027 |
| 2 | ptc50 / 2027 | uniform100 / 2028 | codebook100 / 2028 | random50 / 2028 |
| 3 | disagreement50 / 2028 | ptc50 / 2028 | uniform100 / 2029 | codebook100 / 2029 |
| 4 | random50 / 2029 | disagreement50 / 2029 | spare/evaluation | ptc50 / 2029 |

Every job launches only through
`torchrun --standalone --nnodes=1 --nproc_per_node=8`. There is no Dora,
submitit, rendezvous between machines, `WORLD_SIZE=32`, Slurm, or training
condition attached to a quarter of the data. At most one training job occupies
each machine. The schedule rotates every method over three physical machines
where possible and leaves one failover/evaluation slot in the final wave.

## 8. Run and artifact contract

Every run has a globally unique ID:

```text
{phase}.{model}.{method}.seed-{seed}.{YYYYMMDDTHHMMSSZ}
```

It writes only inside `${PTC_RUN_ROOT}/${run_id}/` with this minimum layout:

```text
run_manifest.json          immutable config, input/source hashes, environment, command
status.json                immutable start record (`running`)
logs/attempt-XXXX.json     immutable launch/start-checkpoint metadata
logs/metrics.attempt-XXXX.jsonl  append-only records for that launch only
checkpoints/step-XXXXX/checkpoint.pt  weights, optimizer, per-rank RNG, metadata
checkpoints/step-XXXXX/SHA256.json    same-directory commit sidecar
SEALED.json                immutable successful terminal record
FAILED.json                immutable failure record when applicable
DONE.json                  sole final success commit; absent means unsealed
```

Each checkpoint directory appears atomically only after both members validate.
Same-run resume must name the highest committed `step-XXXXX` directory; an
older valid step, a partial/staging directory, a symlink, or any unexpected
checkpoint-root member fails closed. Every launch creates the next contiguous
`attempt-XXXX.json` / `metrics.attempt-XXXX.jsonl` pair with exclusive creation
and never reopens an earlier metrics segment. Attempt metadata binds the config
hash, host, launch kind, restored optimizer/global-microstep progress, exact
start-checkpoint directory and hashes, and its matching metrics path.

`DONE.json` is written last; an earlier `FAILED.json` from a recoverable run is
retained for audit and its hash is explicitly superseded by the final seal.
`run_manifest.json` records at least the resolved configuration and its hash,
student/teacher checkpoint hashes, initial in-memory state hashes, codebook
prior hash when used, patched `lm.py` hash, physical hostname, Python/Torch/CUDA
and GPU capability, launcher/world size, full-manifest SHA-256, and the exact
launch command. Evaluator identities belong to later evaluation artifacts, not
the training runner. Console output is captured under the workpack-level
`console_logs/` directory by the runbook.

Each current training-step record in `metrics.attempt-XXXX.jsonl` contains
exactly the runner's current core telemetry contract:

- `schema_version`, `event`, `mode`, `optimizer_step`, `global_microstep`,
  `learning_rate`, `loss`, and `gradient_norm`;
- `global_denominators`, `denominator_window_constant`,
  `selected_cells_rank0`, `valid_cells_rank0`, `step_seconds`,
  `cuda_max_memory_allocated`, and `cuda_max_memory_reserved`.
- the nested all-rank audit additionally binds four microstep sample/gate/
  rollout-code identities, complete trainable-gradient coverage, fixed DDP
  bucket/layout identity with no rebuild, deterministic flags, conditioner
  state, and per-rank peak memory.

Optimizer steps are strictly increasing inside a segment. A failed attempt may
have an uncommitted tail that overlaps the next attempt after rollback to the
latest checkpoint. Canonical curves therefore keep each non-final segment only
through the next attempt's `start_optimizer_step` and then append the final
segment; raw segments remain untouched. On success, `SEALED.json` records each
attempt's metadata/metrics paths and SHA-256 digests, start progress/checkpoint,
record count, and first/last logged optimizer step. `DONE.json` commits that
seal, so post-seal log changes invalidate idempotent terminal verification.

Per-codebook KL/JS/entropy/C50 and detailed timing decomposition are produced by
the dedicated phenomenon/evaluation instrumentation or must be added as a
versioned runner extension before claiming those diagnostics from training
logs. This contract does not pretend the current runner logs fields it does not.

Phenomenon artifacts use stable IDs, never row order, to join prompts, rollouts,
frames, and codebooks. Evaluation manifests count every scheduled sample and
record generation failures; no best-of-N replacement is allowed. Full rollout
logits are not persisted.

## 9. Stage 1 completion rule

Stage 1 is complete only when:

1. B1 correctness is entirely green;
2. P1--P3 have sealed tables, plots, bootstrap intervals, and a written
   pass/fail decision;
3. the small pilot and medium scale gate have explicit decisions;
4. all 15 main jobs are either sealed or the method has been formally
   falsified under the rules above;
5. every compared training job proves the same `train.full.jsonl` hash and
   `WORLD_SIZE=8` single-machine execution.

A failed scientific gate is a valid research result. It should trigger the
declared pivot or stop, not silent threshold changes, selective seeds, or
post-hoc metric replacement.
