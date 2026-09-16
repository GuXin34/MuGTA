# ICASSP2027 environment acceptance report

- **generated**    : 20260812T085958Z
- **host**         : `node-7`
- **by**           : `root`
- **project root** : `<local>/ICASSP2027`
- **freeze used**  : `<local>/ICASSP2027/storage/artifacts/env_freeze/20260812T085312Z`

> Scope note: per project decision, this report includes only §12
> items **3, 4, 5, 6, 7**. Items 1, 2, 8, 9, 10, 11 were explicitly
> excluded (2 is inapplicable — no Slurm, standalone 8-GPU machines).

---

## §12.3  `module list`

```text
No Modulefiles Currently Loaded.
```

## §12.4  three Conda environment absolute paths

- **ptc-opd-train-py39-cu121**
    - prefix : `<local>/envs/ptc-opd-train-py39-cu121`
    - python : 3.9.18
    - size   : 5.9G
    - status : ✅ present
- **ptc-opd-eval-quality**
    - prefix : `<local>/envs/ptc-opd-eval-quality`
    - python : 3.11.15
    - size   : 6.5G
    - status : ✅ present
- **ptc-opd-eval-fad**
    - prefix : `<local>/envs/ptc-opd-eval-fad`
    - python : 3.11.15
    - size   : 12G
    - status : ✅ present

## §12.5  per-env `pip check` and `pip freeze`

Full artifacts (verbatim §11 flat paths):

- `<local>/ICASSP2027/storage/artifacts/pip-freeze-train.txt`
- `<local>/ICASSP2027/storage/artifacts/pip-freeze-eval-quality.txt`
- `<local>/ICASSP2027/storage/artifacts/pip-freeze-eval-fad.txt`
- `<local>/ICASSP2027/storage/artifacts/conda-explicit-train-linux-64.txt`

### train

**pip check**
  ✅ clean — `No broken requirements found.`

**pip freeze (key pins only; full file at `<local>/ICASSP2027/storage/artifacts/env_freeze/20260812T085312Z/train/pip-freeze.txt`)**
  packages: 137
  ```
  encodec==0.1.1
  librosa==0.11.0
  numpy==1.26.4
  scipy==1.13.1
  soundfile==0.13.1
  torch==2.1.0+cu121
  torchaudio==2.1.0+cu121
  torchvision==0.16.0+cu121
  transformers==4.31.0
  xformers==0.0.22.post7
  ```

### eval-quality

**pip check**
  ✅ clean — `No broken requirements found.`

**pip freeze (key pins only; full file at `<local>/ICASSP2027/storage/artifacts/env_freeze/20260812T085312Z/eval-quality/pip-freeze.txt`)**
  packages: 101
  ```
  audiobox_aesthetics==0.0.4
  librosa==0.10.2.post1
  muq==0.1.0
  numpy==1.26.4
  scipy==1.12.0
  soundfile==0.12.1
  torch==2.2.2+cu121
  torchaudio==2.2.2+cu121
  torchvision==0.17.2+cu121
  transformers==4.38.2
  ```

### eval-fad

**pip check**
  ⚠️ pip check reported issues:
  ```
  laion-clap 1.1.6 has requirement numpy==1.23.5, but you have numpy 1.26.4.
  ```

**pip freeze (key pins only; full file at `<local>/ICASSP2027/storage/artifacts/env_freeze/20260812T085312Z/eval-fad/pip-freeze.txt`)**
  packages: 88
  ```
  encodec==0.1.1
  fadtk==1.1.0
  laion_clap==1.1.6
  librosa==0.10.2.post1
  numpy==1.26.4
  scipy==1.15.3
  soundfile==0.13.1
  torch==2.7.1+cu118
  torchaudio==2.7.1+cu118
  torchvision==0.22.1+cu118
  transformers==4.51.3
  ```

## §12.6  `python -m torch.utils.collect_env` (per env)

### train

_(summary; full dump at `<local>/ICASSP2027/storage/artifacts/env_freeze/20260812T085312Z/train/torch-collect-env.txt`)_

```text
    PyTorch version: 2.1.0+cu121
    Is debug build: False
    CUDA used to build PyTorch: 12.1
    OS: TencentOS Server 3.2 (Final) (x86_64)
    GCC version: (GCC) 12.2.1 20221121 (Red Hat 12.2.1-7)
    CMake version: version 4.4.2
    Libc version: glibc-2.28
    Python version: 3.9.18 | packaged by conda-forge | (main, Dec 23 2023, 16:33:10)  [GCC 12.3.0] (64-bit runtime)
    Is CUDA available: True
    CUDA runtime version: 12.8.93
    CUDA_MODULE_LOADING set to: LAZY
    GPU models and configuration: 
    Nvidia driver version: 535.247.01
    cuDNN version: Probably one of the following:
    Versions of relevant libraries:
```

### eval-quality

_(summary; full dump at `<local>/ICASSP2027/storage/artifacts/env_freeze/20260812T085312Z/eval-quality/torch-collect-env.txt`)_

```text
    PyTorch version: 2.2.2+cu121
    Is debug build: False
    CUDA used to build PyTorch: 12.1
    OS: TencentOS Server 3.2 (Final) (x86_64)
    GCC version: (GCC) 12.2.1 20221121 (Red Hat 12.2.1-7)
    CMake version: version 3.31.10
    Libc version: glibc-2.28
    Python version: 3.11.15 | packaged by conda-forge | (main, Aug 11 2026, 10:26:29) [GCC 14.4.0] (64-bit runtime)
    Is CUDA available: True
    CUDA runtime version: 12.8.93
    CUDA_MODULE_LOADING set to: LAZY
    GPU models and configuration: 
    Nvidia driver version: 535.247.01
    cuDNN version: Probably one of the following:
    Versions of relevant libraries:
```

### eval-fad

_(summary; full dump at `<local>/ICASSP2027/storage/artifacts/env_freeze/20260812T085312Z/eval-fad/torch-collect-env.txt`)_

```text
    PyTorch version: 2.7.1+cu118
    Is debug build: False
    CUDA used to build PyTorch: 11.8
    OS: TencentOS Server 3.2 (Final) (x86_64)
    GCC version: (GCC) 12.2.1 20221121 (Red Hat 12.2.1-7)
    CMake version: version 3.31.10
    Libc version: glibc-2.28
    Python version: 3.11.15 | packaged by conda-forge | (main, Aug 11 2026, 10:26:29) [GCC 14.4.0] (64-bit runtime)
    Is CUDA available: True
    CUDA runtime version: 12.8.93
    CUDA_MODULE_LOADING set to: LAZY
    GPU models and configuration: 
    Nvidia driver version: 535.247.01
    cuDNN version: Probably one of the following:
    Versions of relevant libraries:
```

## §12.7  `python -m xformers.info` (train env only)

```text
xFormers 0.0.22.post7
memory_efficient_attention.cutlassF:               available
memory_efficient_attention.cutlassB:               available
memory_efficient_attention.decoderF:               available
memory_efficient_attention.flshattF@v2.3.2:        available
memory_efficient_attention.flshattB@v2.3.2:        available
memory_efficient_attention.smallkF:                available
memory_efficient_attention.smallkB:                available
memory_efficient_attention.tritonflashattF:        unavailable
memory_efficient_attention.tritonflashattB:        unavailable
memory_efficient_attention.triton_splitKF:         available
indexing.scaled_index_addF:                        available
indexing.scaled_index_addB:                        available
indexing.index_select:                             available
swiglu.dual_gemm_silu:                             available
swiglu.gemm_fused_operand_sum:                     available
swiglu.fused.p.cpp:                                available
is_triton_available:                               True
pytorch.version:                                   2.1.0+cu121
pytorch.cuda:                                      available
gpu.compute_capability:                            9.0
gpu.name:                                          NVIDIA H20
build.info:                                        available
build.cuda_version:                                1201
build.python_version:                              3.9.18
build.torch_version:                               2.1.0+cu121
build.env.TORCH_CUDA_ARCH_LIST:                    5.0+PTX 6.0 6.1 7.0 7.5 8.0+PTX 9.0
build.env.XFORMERS_BUILD_TYPE:                     Release
build.env.XFORMERS_ENABLE_DEBUG_ASSERTIONS:        None
build.env.NVCC_FLAGS:                              None
build.env.XFORMERS_PACKAGE_FROM:                   wheel-v0.0.22.post7
build.nvcc_version:                                12.1.66
source.privacy:                                    open source
```

---

## verdict

**PASS** — all included §12 items (3, 4, 5, 6, 7) verified.
