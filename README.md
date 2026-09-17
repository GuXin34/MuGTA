# MUGTA: On-Policy Guidance Transfer and Supervision Allocation for Music Generation

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%E2%80%933.11-blue.svg)](https://www.python.org)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1.0-ee4c2c.svg)](https://pytorch.org)

> Reference implementation of the paper **"MuGTA: On-Policy Guidance Transfer and Supervision Allocation for Music Generation"**.
>
> The paper studies **on-policy distillation (OPD)** for text-to-music language models. We show that the *frozen-CFG-teacher* baseline already recovers most of the perceptual gain of CFG-guided sampling **while roughly halving decoder latency**, and we investigate two orthogonal supervision-selection schemes — **codebook-aware weighting** (`a_q` derived from a codec prior) and **perceptual-trajectory (PTC) top-K JS masking** — under a strict factorial protocol on MusicGen-small and MusicGen-medium.

---

## Table of Contents
1. [Highlights](#highlights)
2. [Method Overview](#method-overview)
3. [Repository Layout](#repository-layout)
4. [Installation](#installation)
5. [Quickstart](#quickstart)
6. [Reproducing the Paper](#reproducing-the-paper)
7. [Evaluation Protocol](#evaluation-protocol)
8. [Known Limitations](#known-limitations)
9. [Citation](#citation)
10. [License](#license)
11. [Acknowledgements](#acknowledgements)

---

## Highlights

- **Frozen-CFG-teacher OPD baseline** for MusicGen-small / medium — a single-forward student that matches CFG-5 teacher perceptual quality at **~1.9× lower per-audio latency** (14.8 s vs 28+ s on RTX 4090, batch=1, 10 s clips).
- A reproducible **6-method factorial protocol** for supervision-selection ablations:
  `uniform100`, `codebook100`, `random50`, `prefix50`, `disagreement50`, `ptc50`.
- **Codec-derived codebook weights `a_q`** (A1-R2 protocol, byte-reproducible from EnCodec).
- **Perceptual-trajectory top-K JS gate** with strict paired-CI evaluation (128 prompts × 2 seeds, prompt-level bootstrap).
- **AudioCraft overlay patch** enabling explicit `use_cfg=False` generation with byte-preserved parity to upstream (`patches/audiocraft/0001-explicit-no-cfg-generation.patch`).
- **Sealed audit chain** — every training / evaluation stage carries a controller-ledger receipt with SHA-256 provenance for the manifest, contract, and evidence tree.

## Method Overview

MuGTA studies three orthogonal design axes for on-policy distillation of MusicGen:

| Axis | Symbol | Values investigated |
|---|---|---|
| Teacher CFG signal | `use_cfg` | frozen CFG-5 vs no-CFG |
| Codebook weighting | `a_q` | uniform vs codec-prior-derived |
| Position-selection gate | `S(t)` | full-position vs top-K JS (K = 50 %) vs random-50 vs prefix-50 vs codebook-100 |

The student is optimized with a **forward-KL objective** against a frozen teacher, computed on the delayed-pattern valid mask of the raw codec coordinate `[B, Q, T, V]`. See the paper §3 for exact formulas; the reference implementation lives in [`src/ptc_opd/losses.py`](src/ptc_opd/losses.py) and [`src/ptc_opd/sampling.py`](src/ptc_opd/sampling.py).

## Repository Layout

```
ptc_opd_release/
├── src/ptc_opd/               # Core Python package
│   ├── losses.py              # Forward-KL + PTC / codebook-weighted variants
│   ├── sampling.py            # Position-selection strategies (JS, random, prefix)
│   ├── phenomena.py           # Codec-prior aware phenomenon detectors
│   ├── codec_prior_artifact.py  # A1-R2 sealed codec prior
│   ├── perceptual_prior.py    # PTC trajectory computation
│   ├── stage1_*.py            # Stage-1 controller / ledger / generation / metrics
│   ├── musicgen_contract.py   # Frozen contract for MusicGen models
│   ├── audiocraft_adapter.py  # AudioCraft wrapper (no upstream fork)
│   └── reproducibility.py     # Seed / RNG / determinism helpers
├── scripts/                   # Training / evaluation / audit pipeline
│   ├── train_stage1.py        # Main training entry
│   ├── generate_stage1_audio.py
│   ├── eval_stage1_quality.py
│   ├── eval_stage1_diversity_fad.py
│   ├── eval_stage1_clap.py
│   ├── cfg_scale_gate.py      # CFG-teacher generation gate
│   ├── estimate_codec_prior.py
│   ├── benchmark_stage1_perf.py
│   └── verify_stage1_*.py     # Producer-verifier scripts
├── patches/audiocraft/        # Overlay patch on AudioCraft 1.4.0a2
│   ├── 0001-explicit-no-cfg-generation.patch
│   └── tests/models/test_lm_no_cfg.py
├── configs/
│   ├── stage1_matrix.yaml     # 6-method factorial matrix
│   └── stage1_autonomy_contract.json
├── docs/                      # Runbooks and provenance notes
│   ├── STAGE1_AUTONOMY_RUNBOOK.md
│   ├── PERF_BENCHMARK_RUNBOOK.md
│   ├── cfg_scale_gate_runbook.md
│   ├── remote_training_runbook.md
│   ├── data_manifest_runbook.md
│   └── audiocraft_integration_audit.md
├── tests/                     # 39 unit / contract tests
├── CITATION.cff
├── LICENSE                    # Apache-2.0
└── pyproject.toml
```

## Installation

MuGTA targets **Python 3.9 – 3.11** and **PyTorch 2.1.0 + CUDA 12.1**.

```bash
git clone https://github.com/GuXin34/MuGTA.git
cd MuGTA

# 1. Create environment
conda create -n mugta python=3.9 -y
conda activate mugta

# 2. Install PyTorch (CUDA 12.1)
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 3. Install MuGTA and dev tooling
pip install -e ".[dev]"

# 4. Apply the AudioCraft no-CFG overlay
git clone https://github.com/facebookresearch/audiocraft.git third_party/audiocraft
cd third_party/audiocraft
git checkout 896ec7c47f5e5d1e5aa1e4b260c4405328bf009d   # 1.4.0a2
git apply ../../patches/audiocraft/0001-explicit-no-cfg-generation.patch
pip install -e .
cd ../..

# 5. Verify install
pytest tests/ -x -q
```

> **Evaluators needed** for the full paper reproduction (MuQ, AudioBox, CLAP, MERT). Download instructions and SHA-pinned model identities are in [`docs/STAGE1_AUTONOMY_RUNBOOK.md`](docs/STAGE1_AUTONOMY_RUNBOOK.md) §2. HuggingFace mirrors will be uploaded post-review.

## Quickstart

```python
from ptc_opd import losses, sampling
import torch

# Toy 2-sample 4-codebook 100-frame example
B, Q, T, V = 2, 4, 100, 2048
student_logits = torch.randn(B, Q, T, V, requires_grad=True)
teacher_logits = torch.randn(B, Q, T, V)
valid = torch.ones(B, Q, T, dtype=torch.bool)
a_q = torch.tensor([1.0, 0.87, 0.62, 0.51])   # codec prior weights

# Forward-KL with codebook weighting + top-50% JS gate
loss = losses.ptc_opd_loss(
    student_logits, teacher_logits, valid,
    codebook_weights=a_q,
    selector=sampling.top_k_js_gate(rho=0.5, scope="codebook"),
)
loss.backward()
```

## Reproducing the Paper

The full pipeline is codified as a 16-node autonomy runbook. A minimal 5-step version:

```bash
# 1. Estimate codec prior (A1-R2 protocol; sha-pinned, ~15 min on 1 A100)
python scripts/estimate_codec_prior.py --output artifacts/codec_prior/

# 2. Train the 6 factorial methods (Small pilot, 500 steps each)
bash scripts/stage1_controller_shell.sh --scale small --steps 500

# 3. Generate audio for 128 prompts × 2 seeds
python scripts/generate_stage1_audio.py \
    --checkpoint-dir artifacts/stage1/small_pilot/ \
    --prompts docs/prompts/musiccaps_128.jsonl \
    --seeds 31001 31002

# 4. Full evaluation battery (MuQ + AudioBox + CLAP + MERT + FAD)
python scripts/eval_stage1_quality.py       --run-dir artifacts/stage1/small_pilot/
python scripts/eval_stage1_clap.py          --run-dir artifacts/stage1/small_pilot/
python scripts/eval_stage1_diversity_fad.py --run-dir artifacts/stage1/small_pilot/

# 5. Bootstrap paired CIs + factorial contrasts
python scripts/build_stage1_pilot_summary.py \
    --run-dir artifacts/stage1/small_pilot/ \
    --output artifacts/stage1/small_pilot_summary.json
```

For the full MusicGen-medium cross-scale protocol, see [`docs/STAGE1_AUTONOMY_RUNBOOK.md`](docs/STAGE1_AUTONOMY_RUNBOOK.md).

## Evaluation Protocol

- **Prompts:** 128 prompt clusters from MusicCaps dev-split (bootstrap unit)
- **Seeds:** 2 audio seeds per prompt (pre-averaged before bootstrap)
- **Metrics:** MuQ, AudioBox CE / PQ, CLAP-Music, MERT-diversity, FAD (VGGish + CLAP + MERT)
- **Statistical test:** prompt-cluster bootstrap paired CIs at 95 %, `n_boot=10 000`
- **Composite:** Q_dev = weighted paired difference over aligned per-audio z-scores; see paper §4.2

All hyperparameters (seed=2027, LR=3e-6, CFG=5, effective batch=64, warmup=50) are frozen in [`configs/stage1_matrix.yaml`](configs/stage1_matrix.yaml) and verified by 39 contract tests.

## Known Limitations

- Studied only on MusicGen-small (300 M) and MusicGen-medium (1.5 B). MusicGen-large is out of the paper's compute budget.
- Text encoder = frozen T5-base per MusicGen release; we do not vary the conditioner.
- Evaluators are English-centric (MusicCaps captions); non-English prompt generalization is future work.
- FAD reference embeddings computed on our own FMA-small subset (SHA-pinned, see [`docs/data_manifest_runbook.md`](docs/data_manifest_runbook.md)).

## Citation

If you use MuGTA in your research, please cite:

```bibtex
@inproceedings{mugta2027,
  title     = {MuGTA: On-Policy Guidance Transfer and Supervision Allocation for Music Generation},
  author    = {Anonymous},
  booktitle = {ICASSP},
  year      = {2027},
  note      = {Under review}
}
```
Machine-readable metadata is also available in [`CITATION.cff`](CITATION.cff).

## License

MuGTA is released under the **Apache License 2.0** — see [`LICENSE`](LICENSE).

The AudioCraft overlay patch in `patches/audiocraft/` is derivative of Meta's AudioCraft (MIT License, © Meta Platforms) and inherits the MIT terms as noted in the patch header. All other code is original to this repository.

## Acknowledgements

We thank the maintainers of the following open-source projects.

**Backbone & generation:**
- [MusicGen / AudioCraft](https://github.com/facebookresearch/audiocraft)
- [EnCodec](https://github.com/facebookresearch/encodec)
- [T5](https://huggingface.co/t5-base)

**Evaluators:**
- [MuQ](https://github.com/tencent-ailab/MuQ)
- [MERT](https://huggingface.co/m-a-p/MERT-v1-95M)
- [LAION-CLAP](https://github.com/LAION-AI/CLAP)
- [AudioBox Aesthetics](https://ai.meta.com/research/publications/audiobox-aesthetics/)

**FAD:**
- [frechet_audio_distance](https://github.com/microsoft/fadtk)
- [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish)
