# Method freeze v0.1

Date: 2026-08-12

Status: frozen for Stage 1. A change requires a dated changelog entry and all
affected comparisons must be rerun; do not tune the definition on the test set.

## 1. Problem statement

MusicGen predicts several EnCodec codebooks at every codec frame. Standard
token-level OPD can average the teacher-student divergence over every valid
time-codebook position. That objective implicitly gives equal supervision
budget to positions that may differ in two ways:

1. a codebook's marginal contribution to reconstructed audio can be unequal;
2. along a student trajectory, teacher and student may already agree at many
   positions while a smaller subset carries most of the mismatch.

The paper must first measure both forms of non-uniformity. The proposed method
is rejected or narrowed if either is not reproducible.

## 2. Canonical tensor convention

The workpack uses the native AudioCraft prediction layout:

```text
teacher_logits: [B, Q, T, V]
student_logits: [B, Q, T, V]
valid_mask:     [B, Q, T]
codebook_prior: [Q]
```

`valid_mask` is the intersection of the audio padding mask and AudioCraft's
pattern-validity mask. Invalid delayed-pattern and padding positions never
enter selection, normalization, or diagnostics.

The frozen teacher distribution is computed on the exact student-sampled
prefix. Teacher logits are detached. The rollout action itself is not
differentiated through; gradients enter only through the teacher-forced student
logits on its sampled trajectory.

## 3. Offline codebook prior

Let `x_hat^(q)` be the EnCodec reconstruction using the first `q` residual
codebooks, and let `d` initially be a frozen multi-resolution STFT distance.

```text
Delta_q = E[d(x, x_hat^(q-1)) - d(x, x_hat^(q))]
a_q     = max(Delta_q, epsilon) / sum_j max(Delta_j, epsilon)
```

The sum-one representation is deliberate: uniform `a_q=1/Q` exactly recovers
the uniform weighted loss. Because the final loss divides by selected effective
weight, multiplying every `a_q` by one common scalar cannot change the loss or
its gradient. The prior is
estimated only on a training-side calibration manifest. It is frozen before
the development comparison and never estimated from generated test audio.

Negative or zero empirical marginal improvements are floored at `epsilon` and
reported; they are not silently deleted. A split-half bootstrap must show the
rank/profile is reproducible before the learned prior supports a paper claim.

## 4. Per-position distributions

Logits are converted to float32 for the divergence calculation even under BF16
training. With distillation temperature `tau`:

```text
p_T = softmax(teacher_logits / tau)
p_S = softmax(student_logits / tau)
```

The token loss is initially forward KL:

```text
d_tq = tau^2 * KL(p_T || p_S).
```

Reverse KL is a one-seed diagnostic, not a tunable alternative in the main
table. The selector uses detached Jensen-Shannon divergence:

```text
j_tq = JS(stopgrad(p_T), stopgrad(p_S)).
```

This prevents the discrete choice of selected positions from becoming a
gradient path.

## 5. Main selector: matched per-sample/per-codebook budget

For every `(sample b, codebook q)` independently, select exactly
`ceil(rho * n_bq)` valid time positions with the largest `j_btq`, where `n_bq`
is that row's number of valid positions. If `rho=0`, select none; if `rho=1`,
select every valid position. Ties use a deterministic stable order.

This is the primary selector rather than global `a_q * JS` top-k. The reason is
experimental identifiability: Random-50%, Disagreement-50%, and PTC-50% then
select exactly the same number of tokens in every `(b,q)` stratum. The PTC
comparison cannot win merely by reallocating more selected tokens to an early
codebook.

Within one codebook `a_q` is constant and therefore cannot change a time-only
ranking. Codebook awareness enters the loss weighting, not the main gate.

Global top-k using `a_q * JS` is retained as a one-seed diagnostic named
`global_joint_selector`; it is not the headline PTC method.

## 6. Normalized loss

For binary gate `g_btq` and method-specific codebook weights `w_q`:

```text
L = sum_btq valid_btq * g_btq * w_q * d_btq
    / sum_btq valid_btq * g_btq * w_q.
```

The denominator is the actual selected effective weight. It prevents retention
or codebook weights from silently changing the loss coefficient. A batch with
no selected valid positions is a configuration/data error, not a zero-loss
training step.

## 7. Five primary trained conditions

| ID | Gate | Codebook weight |
|---|---|---|
| `uniform100` | every valid position | `1` |
| `codebook100` | every valid position | `a_q` |
| `random50` | random matched count in each `(b,q)` | `1` |
| `disagreement50` | JS top-50% in each `(b,q)` | `1` |
| `ptc50` | same JS top-50% rule | `a_q` |

The random gate is seeded from recorded run/step/sample identifiers and is
reproducible after resume. It is never sampled from an unrecorded global RNG
state.

The component logic is now exact:

- `codebook100 - uniform100` isolates codebook weighting;
- `disagreement50 - random50` isolates targeted time selection at matched
  sparsity;
- `ptc50 - disagreement50` isolates codebook weighting under the same selector;
- `ptc50 - random50` tests the combined method against matched random
  sparsification.

## 8. Optimization role

The initial implementation uses OPD alone to isolate supervision allocation.
It does not add GRPO, a learned reward, or offline SFT to the primary loss. All
conditions start from the same public checkpoint and receive identical prompt
order, rollout token count, optimizer updates, checkpoint opportunities, and
generation settings.

The distributed runtime is also part of the frozen protocol.  Under pinned
Torch 2.1, every condition uses explicit `bucket_cap_mb=25` together with the
pinned Torch default first-bucket cap, `find_unused_parameters=True`,
`static_graph=False`, and
   `gradient_as_bucket_view=False`.  This public configuration prevents the
   first-iteration bucket rebuild considered by patch05.  The real H20 patch05
   run proved that mitigation was not sufficient to eliminate the observed
   warm/cold drift, so the reducer hypothesis is no longer treated as causal
   closure.  The setting is retained in patch06 to avoid changing training
   arithmetic while the resume acceptance category is corrected.  It is not
an assertion that the LM has unused parameters: each synchronized update must
produce a gradient for every trainable parameter.  The complete bucket/layout
   identity and no-rebuild state are sealed per rank.  Patch05 H20 evidence
   showed this mitigation was insufficient to make warm/cold process histories
   bit-identical; patch06 therefore qualifies serialized restore semantics
   directly.  The protocol remains pending until patch06 passes 15/15.

The teacher is the frozen starting checkpoint evaluated with CFG; the student
is initialized from the same checkpoint and rolls out without CFG. Before the
main run, the development gate must verify that the chosen CFG scale improves
the frozen no-CFG model under the fixed evaluation suite. If it does not, CFG
distillation has no useful target and training stops for diagnosis.

## 9. Required invariants

The implementation is unacceptable unless all hold:

1. identical teacher/student logits yield numerical zero in float32;
2. `uniform100` equals a direct valid-token KL mean in value and gradient;
3. `codebook100` with all `a_q=1` equals `uniform100`;
4. `ptc50` with all `a_q=1` equals `disagreement50` in gate, value, and gradient;
5. random and targeted 50% gates have equal counts for every `(b,q)` stratum;
6. invalid/padded logits may be NaN without contaminating the loss because they
   are masked before any reduction;
7. selected positions have finite logits and finite divergence;
8. teacher parameters receive no gradient and their pre/post hashes match;
9. resuming a run reproduces subsequent random gates and prompt order;
10. all reductions under DDP use global numerators and denominators, not a mean
    of unequal per-rank means;
11. the DDP reducer bucket/layout identity is equal across all ranks and across
    uninterrupted/resumed processes, with no post-first-step rebuild;
12. the node-3 qualification requires checkpoint-to-live student/optimizer
    restoration to be finite, layout-aware and bit exact on all eight ranks,
    plus bit-exact step-1/2 student, optimizer, scheduler and RNG state across
    two independent cold resumes from the same checkpoint.  Warm-uninterrupted
    versus cold-resumed floating state must satisfy the frozen field-specific
    ULP/absolute/relative-L2 continuity contract; a generic numerical `allclose`
    waiver is not part of the method.

## 10. Claim boundary

PTC-OPD is a supervision-allocation objective. Selecting token losses does not
avoid MusicGen's shared Transformer or teacher forward passes. No speed, memory,
or `2x` claim is allowed unless end-to-end measurements demonstrate it.

The intended claim is not that early codebooks are universally semantic. The
paper may say only that the frozen progressive reconstruction assay found a
reproducible non-uniform perceptual contribution profile for the tested codec.
