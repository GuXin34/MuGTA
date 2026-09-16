# Stage-1 pilot-prep DDP / telemetry paired benchmark

## 1. Purpose and authority

This is an engineering performance assay required before any 500-step Stage-1
job.  It quantifies the cost of the frozen production DDP policy
`find_unused_parameters=True` together with patch06 step telemetry and hashes.
It does **not** choose a new training policy and does not generate a model that
may be evaluated, resumed, or cited as a scientific result.

The production decision is frozen before measurement:

```text
production_policy_decision = retain_true_full
false/min outputs authorized for science = false
```

Even if a counterfactual arm is much faster, the remote operator may not edit
`scripts/train_stage1.py`, disable production auditing, or use a benchmark
checkpoint.  A policy change requires a separately reviewed Stage-1 contract.

Prerequisites:

1. T5 closure is `equivalent_all_inputs`, rc 0;
2. retained A1-R2, small CFG, A2/A3, and node-3 seals verify read-only;
3. the real-checkpoint B1 correctness audit has passed;
4. no 500-step LR/stability run has started.

## 2. Why this cannot be a CLI flag on the formal runner

The frozen runner is bound to:

```text
scripts/train_stage1.py SHA-256
697a450da4848a1589b257816e2572dbc681ae54e4535d300d2fda39257df7c5
```

Its DDP constructor, reducer identity, checkpoint metadata, run manifest, and
terminal seal all require the production True/no-rebuild policy.  Adding a
`--find-unused` flag to that file would weaken the formal consumer contract.

`scripts/benchmark_stage1_perf.py` therefore imports the frozen file by exact
SHA, keeps the same model/data/loss/sampling/optimizer path, and applies a
process-local benchmark adapter.  The adapter changes all run/checkpoint/DONE
schemas to `ptc-opd-benchmark-only-*` and inserts:

```text
benchmark_only = true
scientific_use_forbidden = true
formal_stage1_consumer_must_reject = true
```

The formal `verify_stage1_run.py` expects `ptc-opd-stage1-run-v3`; it must reject
the derived run at schema validation.  Benchmark evidence has its own
fail-closed verifier and seal.

No source-generated copy of the 2,900-line training loop is maintained.  Hooks
around AdamW, LR boundaries, reducer observation, state hashes, checkpoint
commit, logging, and terminal commit provide timing while the scientific loop
remains byte-identical.

## 3. Four arms and Williams schedule

| ID | Name | DDP | Observational telemetry |
|---|---|---:|---|
| A | `true_full` | `find_unused_parameters=True` | production patch06 |
| B | `false_full` | `find_unused_parameters=False` | production-strength recording |
| C | `true_min` | `find_unused_parameters=True` | hashes/gather/fsync disabled |
| D | `false_min` | `find_unused_parameters=False` | hashes/gather/fsync disabled |

`min` retains all scientific safety checks: BF16, finite logits/loss/gradient,
rollout shape/range, denominator equality, missing-gradient rejection,
conditioner freeze, gradient clip, optimizer, sampling, data, and batch math.
It uses `log_every=41` for a 40-step benchmark and replaces only the two
observational tensor hashes with constant placeholders.  The original small
microstep dictionary construction remains, so the measured full-minus-min
telemetry cost is slightly conservative rather than exaggerated.

Each node runs every arm once in a balanced order:

```text
node-0: A B D C
node-1: B C A D
node-2: C D B A
node-3: D A C B
```

Every period contains all four arms once across the cluster, and every ordered
carry-over pair occurs once.  The jobs remain four independent single-node
process groups; this schedule is coordination metadata, not multi-node DDP.

## 4. Frozen workload and timing

```text
model                MusicGen-small
method               uniform100
seed                 2027
learning rate        3e-6
physical batch       2/GPU
world size           8
gradient accumulation 4
effective batch      64
duration / frames    10 seconds / 500 codec frames
teacher CFG          retained small scale 5 decision
finite checking      enabled
warmup               10 optimizer steps
measured             30 optimizer steps
block                 5 steps (6 blocks)
```

At completed steps 10, 15, 20, 25, 30, 35, and 40, every rank synchronizes CUDA
and uses a barrier outside the measured interval.  A block starts after the
barrier and ends after CUDA completion but before the next barrier.  Rank-local
times are retained; the complete-block duration is the maximum across ranks.

This captures full-arm JSON encoding, per-step `fsync`, and stdout flush, which
the formal runner's legacy `step_seconds` omits.  Initial/final checkpoint,
teacher-state hash, and terminal seal are recorded separately and are not
misreported as steady-state step time.

For the False arms, actual bucket lifecycle is recorded at construction,
timing boundaries, and full-audit steps.  Bucket rebuilding is allowed only in
this counterfactual evidence.  The True arms fail immediately if
`has_rebuilt_buckets` is ever true.

## 5. Remote preflight

Use the same prepared environment and shared immutable assets on all nodes:

```bash
export PTC_AUTONOMY=<local>/ICASSP2027_PTC_OPD_Stage1Autonomy_20260819
export PTC_RETAINED=<local>/ICASSP2027_PTC_OPD_A1R2_Workpack_20260814
export PTC_ENV=<local>/envs/ptc-opd-train-py39-cu121
export PTC_MANIFEST=${PTC_RETAINED}/manifests/musiccaps-v1/train.full.jsonl
export PTC_AUDIOCRAFT=${PTC_RETAINED}/vendor/audiocraft
export PTC_CFG_SMALL=${PTC_RETAINED}/artifacts/cfg_scale/small/decision
: "${PTC_MUSICGEN_SMALL:?export the exact dereferenced small checkpoint path}"

test -x "${PTC_ENV}/bin/torchrun"
test -f "${PTC_AUTONOMY}/scripts/benchmark_stage1_perf.py"
test -f "${PTC_AUTONOMY}/scripts/run_perf_node.py"
test -f "${PTC_MANIFEST}"
test -d "${PTC_AUDIOCRAFT}"
test -d "${PTC_CFG_SMALL}"
test -d "${PTC_MUSICGEN_SMALL}"

sha256sum "${PTC_AUTONOMY}/scripts/train_stage1.py"
```

`PTC_AUTONOMY` is the new source/controller workpack and remains the home of
all benchmark outputs.  The training manifest, AudioCraft checkout, and sealed
small CFG decision are retained immutable inputs from `PTC_RETAINED`; the new
portable source bundle intentionally does not recreate those generated assets.

The last digest must equal the frozen value in section 2.  Do not set
`PTC_NODE3_GATE=1`: that switch measures the two-update deterministic
qualification environment, not the pilot environment.

## 6. One command per node

Node-0 generates one UTC and sends that exact literal to the other three
nodes.  The parent is shared, so each worker checks only its own new leaf:

```bash
# 仅 node-0 执行一次，并把打印出的字面值复制到 node-1～3：
export PTC_PERF_UTC=$(date -u +%Y%m%dT%H%M%SZ)
printf '%s\n' "${PTC_PERF_UTC}"

# 四台机器都从这里开始；node-1～3 不得自行运行 date：
: "${PTC_PERF_UTC:?copy the one coordinator-issued UTC literal to all nodes}"
export PTC_PERF_ROOT=${PTC_AUTONOMY}/artifacts/stage1/performance/${PTC_PERF_UTC}
export PTC_NODE_LABEL=node-0  # set node-1/node-2/node-3 on those hosts
mkdir -p "${PTC_PERF_ROOT}"
test ! -e "${PTC_PERF_ROOT}/${PTC_NODE_LABEL}"
```

On node-0:

```bash
"${PTC_ENV}/bin/python" "${PTC_AUTONOMY}/scripts/run_perf_node.py" \
  --node-label "${PTC_NODE_LABEL}" \
  --torchrun "${PTC_ENV}/bin/torchrun" \
  --manifest "${PTC_MANIFEST}" \
  --student-checkpoint "${PTC_MUSICGEN_SMALL}" \
  --teacher-checkpoint "${PTC_MUSICGEN_SMALL}" \
  --audiocraft-root "${PTC_AUDIOCRAFT}" \
  --cfg-scale-decision-dir "${PTC_CFG_SMALL}" \
  --output-dir "${PTC_PERF_ROOT}/${PTC_NODE_LABEL}"
```

Run the identical command on node-1, node-2, and node-3, changing only
`PTC_NODE_LABEL`.  Do not run `date` independently on those nodes.  The
orchestrator itself:

- removes torchrun/Slurm/MPI pollution variables;
- unsets `PTC_NODE3_GATE` and `CUBLAS_WORKSPACE_CONFIG`;
- exposes exactly GPUs 0--7;
- freezes offline HF mode and `OMP_NUM_THREADS=1`;
- records eight-row `nvidia-smi` inventory before and after;
- launches four new standalone torchrun processes sequentially;
- stops on the first failure without deleting evidence;
- verifies and seals every successful arm and then the node packet.

There is no arm or node resume.  If any node sequence fails, retain the failed
global root, diagnose the operational fault, issue one new global UTC, and
rerun all four nodes' complete Williams sequences.  Aggregation may consume
only four node leaves from that one global root; never splice nodes or periods
from different attempts.

## 7. Verification and four-node aggregation

Verify each node independently:

```bash
for node in node-0 node-1 node-2 node-3; do
  "${PTC_ENV}/bin/python" "${PTC_AUTONOMY}/scripts/verify_perf_benchmark.py" \
    --node-dir "${PTC_PERF_ROOT}/${node}" \
    --train-manifest "${PTC_MANIFEST}" \
    --small-cfg-dir "${PTC_CFG_SMALL}" \
    --audiocraft-dir "${PTC_AUDIOCRAFT}" \
    --musicgen-small-dir "${PTC_MUSICGEN_SMALL}"
done
```

四个 live-input 参数是 node verifier 的必填契约。它会覆盖本节点全部四个 arm，
同时核对 producer benchmark manifest、formal run config、checkpoint/state hashes、
CFG generation identity 与 AudioCraft source identity；仅有结构/计时 seal 不构成通过。

After all four return `status=verified`, aggregate once on node-0:

```bash
test ! -e "${PTC_PERF_ROOT}/aggregate"
"${PTC_ENV}/bin/python" "${PTC_AUTONOMY}/scripts/aggregate_perf_benchmark.py" \
  --node-dir "${PTC_PERF_ROOT}/node-0" \
  --node-dir "${PTC_PERF_ROOT}/node-1" \
  --node-dir "${PTC_PERF_ROOT}/node-2" \
  --node-dir "${PTC_PERF_ROOT}/node-3" \
  --output-dir "${PTC_PERF_ROOT}/aggregate"

"${PTC_ENV}/bin/python" "${PTC_AUTONOMY}/scripts/verify_perf_benchmark.py" \
  --aggregate-dir "${PTC_PERF_ROOT}/aggregate"
```

The aggregation computes within-node paired ratios before combining nodes:

```text
DDP/full  = true_full / false_full
DDP/min   = true_min  / false_min
audit/T   = true_full / true_min
audit/F   = false_full / false_min
interaction = DDP/full / DDP/min
```

It reports the median and full node range.  It does not pretend that 120
autocorrelated steps are 120 independent experimental units.

## 8. Gates and autonomous continuation

Hard gates:

- every arm completes exactly 40 updates on eight ranks;
- every rank contributes six valid five-step blocks;
- no NaN/Inf, OOM, missing gradient, or denominator failure;
- True policy never rebuilds buckets;
- production `true_full` peak reserved-memory margin is at least 5%;
- every arm/node/aggregate seal verifies.

Measurement-quality rule:

```text
robust CV = 1.4826 * MAD / median
```

If any arm/node exceeds 10%, aggregate status is
`inconclusive_extend_measurement`; run a new complete balanced attempt rather
than deleting slow blocks.  Otherwise, and if the hard resource gate passes,
the status authorizes autonomous continuation to the next already-frozen
Stage-1 task.  It never authorizes a DDP/logging policy change.

No performance ceiling was preregistered.  Therefore a large overhead is
reported, not converted into an improvised failure threshold.  The aggregate
also projects 500/1000-step compute time excluding separately reported
checkpoint pauses.

## 9. Minimal return protocol

Keep all formal-run/checkpoint trees remote.  Return only the aggregate:

```text
aggregate/STATUS.json
aggregate/paired_summary.json
aggregate/paired_summary.csv
aggregate/ARTIFACT_SEAL.json
```

The milestone message should include:

```text
stage: pilot-prep DDP/telemetry benchmark
four_node_gate: passed | resource_gate_failed
measurement_quality: complete | inconclusive_extend_measurement
true_full median / conservative p90 seconds per step
DDP/full and audit/True median ratios + node ranges
minimum memory margin
production_policy_decision: retain_true_full
redline_touched: false
required_action
```

Full arm trees are returned only on a verifier failure or explicit review
request.

## 10. CPU-only workpack test

This verifies schedules, closed-world seals, rank/block contracts, tamper
rejection, paired ratios, memory gate, and the formal-schema separation without
CUDA or AudioCraft:

```bash
cd "${PTC_AUTONOMY}"
"${PTC_ENV}/bin/python" -m unittest -v tests.test_perf_benchmark
```

Passing CPU tests do not substitute for the real H20 benchmark.
