# Literature positioning freeze

Date checked: 2026-08-12  
Scope: public papers relevant to on-policy distillation (OPD), classifier-free
guidance (CFG) distillation, audio/music generation, and non-uniform token
supervision. This is a claim-boundary document, not an experimental result.

## 1. Closest precedents

| Work | Domain and teacher | Reduction over positions | What overlaps with this work | Remaining distinction |
|---|---|---|---|---|
| Cideron et al., *Diversity-Rewarded CFG Distillation*, arXiv:2410.06084v1 | Text-to-music with MusicLM/MusicRL-R; the frozen initialization augmented with CFG is the teacher and the current no-CFG student supplies online partial generations. | Eq. 2 minimizes `KL(student || CFG teacher)` summed over all `L` generated positions. | Same-architecture CFG-to-no-CFG on-policy music distillation; it also directly documents quality gain and diversity loss under pure CFG distillation. | It does not study MusicGen's two-dimensional `(codebook,time)` lattice, an offline RVQ codebook prior, fixed-budget position selection, or the component controls used here. Its headline remedy adds a learned diversity reward and model merging; ours instead isolates supervision allocation under a pure-OPD comparison. |
| Dai et al., *Pushing the Frontier of Full-Song Generation*, arXiv:2607.20253v3 | Full-song autoregressive generation with an eight-codebook RVQ tokenizer and a frozen OPD teacher. | Sec. 2.6.4 Eq. 17 averages forward KL over the set `M` of every valid response-token position. | Direct multi-codebook music OPD precedent for all-valid-position averaging. | The report does not decompose OPD by RVQ codebook and time, estimate a perceptual codebook prior, or compare matched random and disagreement-selected budgets. Its SFT objective separately gives the first codebook a 5:1 aggregate weight over residual codebooks (Sec. 2.3.2, Eqs. 2-3), which motivates asymmetry but does not validate our prior. |
| Hu et al., *CORD*, arXiv:2601.16547v1 | Audio-language reasoning; an audio-conditioned rollout is aligned to an in-model text-conditioned teacher. | Sec. 3.3 Eq. 3 identifies uniform temporal KL averaging. Sec. 3.4 selects the top `K=20` reverse-KL tokens, assigns them weight `alpha>1`, leaves other tokens at weight 1, and multiplies by an early-position decay (Eqs. 4-7). | Strong prior art for diagnosing a skewed token-divergence distribution and importance-weighting high-disagreement positions. | It is not audio-token or music generation and has no RVQ codebook axis or codec-perceptual prior. It reweights rather than binary-selecting a matched per-codebook budget, uses reverse KL for both score and loss, adds position decay and a GRPO sequence objective, and has no matched random sparsification control. |

Primary sources:

- <https://arxiv.org/abs/2410.06084>
- <https://arxiv.org/abs/2607.20253>
- <https://arxiv.org/abs/2601.16547>

## 2. Adjacent audio OPD work

| Work | Relevant mechanism | Boundary for this paper |
|---|---|---|
| Lin et al., *Data-Efficient On-Policy Distillation for Automatic Speech Recognition* (Ark-ASR), arXiv:2605.28139v1 | Forms a union of teacher and student top-`k` **vocabulary supports** at each transcript position, then evaluates KL on that support. | Vocabulary truncation is not temporal position selection and the output is text, not a multi-codebook codec grid. |
| Cao et al., *X-OPD*, arXiv:2603.24596v3, and Fu et al., *X3-OPD*, arXiv:2607.21550v1 | Cross-modal OPD transfers text reasoning into speech/audio-language students along student trajectories. | These align text-token reasoning and do not allocate losses across generated RVQ codebooks. |
| Xie et al., *LS-MOPD*, arXiv:2608.03610v1 | Routes multilingual ASR rollouts among language-specialized teachers and studies teacher aggregation/prefix consistency. | Its weighting is across teachers/languages, not MusicGen codec-time cells. |
| Zhao et al., *OPOD*, arXiv:2607.20918v3 | Routes student rollouts conditioned on text/image/audio inputs to modality teachers and applies one-sided guidance when a teacher raises the sampled token probability. | It is already a token-position-selective OPD precedent, but addresses multimodal teacher consolidation rather than CFG distribution matching or codec-lattice allocation. |
| Wu et al., *Adaptive Accompaniment with ReaLchords*, ICML 2024 / PMLR 235 | Uses an on-policy KL term from a future-aware teacher while learning online symbolic chord accompaniment. | It is a closely related privileged-teacher idea in symbolic interactive music, but not CFG distillation or acoustic RVQ token generation. |

These works are relevant context, but none of them should be cited as evidence that
MusicGen's individual codebooks have unequal perceptual value. That is an empirical
question for Phase A1.

## 3. Prohibited novelty claims

Do not write any of the following:

- “the first on-policy distillation method for music generation”;
- “the first method to distill CFG into a music model”;
- “the first multi-codebook music OPD method”;
- “the first OPD method to focus on high-disagreement positions”;
- “the first token- or position-selective OPD method”;
- “the first non-uniformly weighted multi-codebook music objective”;
- “all prior OPD weights every time position equally”;
- “prior work proves early EnCodec codebooks are semantic”;
- “selecting half the cells halves training compute.”

The first four are contradicted by the closest precedents. The fifth confuses
reconstruction contribution with semantics. The sixth ignores the shared rollout,
teacher, and student Transformer computation.

## 4. Defensible contribution wording

Until the planned experiments finish, use observation-seeking language:

> We investigate whether CFG-distillation supervision is uniformly valuable on
> MusicGen's multi-codebook codec lattice. We separately estimate a frozen
> progressive-reconstruction prior over RVQ codebooks and measure teacher-student
> disagreement along no-CFG student trajectories. PTC-OPD then combines a matched
> within-codebook top-JS temporal budget with normalized codebook weighting.

If and only if Phase A and the causal pilot pass their predeclared gates, this can
be strengthened to:

> Experiments reveal reproducible heterogeneity along both axes of MusicGen's
> codec lattice. Under an equal per-codebook token budget, targeting high-JS time
> positions outperforms matched random selection, while the frozen perceptual
> prior provides an additional gain without changing which positions are selected.

This framing makes three contributions without relying on a fragile “first” claim:

1. a two-axis measurement study specific to a public MusicGen/EnCodec stack;
2. a factorized allocation rule in which temporal selection and codebook weighting
   are experimentally identifiable;
3. exact-budget component controls and a quality-diversity evaluation that directly
   confronts the known diversity cost of CFG distillation.

## 5. Experiments required by the literature

The following are not optional after this audit:

1. `uniform100`, because both CFG distillation and full-song OPD establish the
   all-position baseline;
2. `random50`, because CORD makes “high-divergence positions matter” a prior-art
   hypothesis rather than a sufficient novelty claim;
3. `disagreement50`, isolated from the codebook prior;
4. `codebook100`, isolated from temporal selection;
5. `ptc50`, using the exact same JS gate as `disagreement50`;
6. multiple samples per prompt plus an explicit diversity metric and blind listening,
   because diversity loss is already known for music CFG distillation;
7. end-to-end wall-clock and peak-memory reporting only as measurements, with no
   theoretical speedup claim;
8. confidence intervals over prompts and seeds, not only a single aggregate score.

## 6. Search boundary

The matrix reflects the local paper collection plus a primary-source arXiv search
performed on 2026-08-12. A final camera-ready literature refresh is mandatory.
New work after this date can narrow the wording further; it cannot be ignored merely
because the Stage-1 method was frozen earlier.
