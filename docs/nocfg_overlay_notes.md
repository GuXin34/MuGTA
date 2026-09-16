# AudioCraft explicit no-CFG generation overlay

## Scope and frozen base

This overlay targets the frozen AudioCraft checkout at commit
`896ec7c47f5e5d1e5aa1e4b260c4405328bf009d`. It changes only
`LMModel.generate` and `LMModel._sample_next_token`, plus one focused test file.
It does not change delayed-codebook patterns, masks, token sampling, training, or
the high-level MusicGen parameter setter.

The distributable overlay is:

- `patches/audiocraft/0001-explicit-no-cfg-generation.patch`
- `patches/audiocraft/tests/models/test_lm_no_cfg.py` (a standalone copy of
  the test added by the patch)

## API and semantics

Both internal token sampling and public LM generation gain a new keyword:

```python
use_cfg: bool = True
```

The trailing position and the `True` default preserve legacy positional and
keyword calls. `LMModel.generate` additionally accepts a trailing optional
`condition_tensors` dictionary for either conditional-only generation (B) or
one-pass CFG (2B, conditional examples followed by null examples). The behavior is:

| Conditions | `use_cfg` | Model forward per decoding step |
|---|---:|---|
| present | `True` | Existing CFG behavior: 2B one-pass, two B-sized passes, or 3B double CFG |
| present | `False` | One B-sized conditional-only pass |
| absent | either | One B-sized unconditional pass |

With `use_cfg=False`, `cfg_coef`, `cfg_coef_beta`, and `two_step_cfg` are
intentionally ignored. The flag is authoritative: even if a caller retains
double-CFG or two-step-CFG settings in an existing configuration, generation
constructs a plain conditioning dictionary and never an unconditional branch.
`_sample_next_token` asserts that this condition object is a dictionary, so a
two-step tuple cannot silently enter the conditional-only path.

Streaming remains one forward per decoding step. No unconditional streaming
state is installed or updated in no-CFG mode; the existing generation-level
state object is merely retained so the private method signature stays backward
compatible. Pattern construction, valid-token masks, prompt handling, and
pattern reversion are untouched.

For rollout/scoring parity, callers can compute a frozen conditional tensor
once and reuse the exact object during generation:

```python
condition_tensors = musicgen.lm.condition_provider(
    musicgen.lm.condition_provider.tokenize(attributes)
)
tokens = musicgen.lm.generate(
    condition_tensors=condition_tensors,
    max_gen_len=750,
    use_cfg=False,
)
```

Precomputed tensors are mutually exclusive with nonempty `conditions`. With
`use_cfg=False`, their batch dimension is B. With `use_cfg=True`, it must be
even and is interpreted as 2B in conditional-then-null order. If neither a
prompt nor `num_samples` is given, generation infers the logical B from the
tensors and checks that every condition entry has the same batch dimension.
This bypasses tokenization and conditioning-provider recomputation while
preserving the same streaming and pattern behavior.

The CFG-scale gate deliberately makes one FP32 conditioner call for the 2B
conditional+null batch, then passes the B view to no-CFG and the original 2B
tensors to each CFG anchor. This keeps conditioner computation shared rather
than quietly recomputing it under a different autocast context.

For direct LM use:

```python
tokens = musicgen.lm.generate(
    conditions=attributes,
    max_gen_len=750,
    use_cfg=False,
)
```

The high-level `MusicGen.set_generation_params` is deliberately outside this
minimal patch. A high-level caller can pass the option through the existing
`BaseGenModel` forwarding path with:

```python
musicgen.generation_params["use_cfg"] = False
```

## Apply and verify

Run these commands from any machine that has the clean frozen AudioCraft clone:

```bash
git -C /path/to/audiocraft rev-parse HEAD
git -C /path/to/audiocraft apply --check \
  /path/to/workpack/patches/audiocraft/0001-explicit-no-cfg-generation.patch
git -C /path/to/audiocraft apply \
  /path/to/workpack/patches/audiocraft/0001-explicit-no-cfg-generation.patch
cd /path/to/audiocraft
pytest -q tests/models/test_lm_no_cfg.py
```

The tests cover:

1. float32 conditional-only logits and greedy-token parity against standard
   one-pass CFG with `cfg_coef=1`;
2. B versus 2B forward batch size, proving that the unconditional branch was
   actually removed rather than algebraically cancelled;
3. delayed-pattern generation shape plus AudioCraft's built-in sequence/mask
   checks, including prompt removal;
4. exact legacy-default versus explicit-`use_cfg=True` output parity;
5. exact B-sized no-CFG precomputed-condition reuse without provider
   recomputation;
6. exact 2B precomputed-CFG parity with the ordinary CFG path, provider bypass,
   API exclusivity, and odd-batch rejection; and
7. rejection of tuple-shaped two-step CFG tensors on the no-CFG path.

The `git apply` commands above are deployment instructions. The patch itself
adds `tests/models/test_lm_no_cfg.py`, so the shown pytest path exists after
application. This workpack was validated using `git apply --check` against the
clean frozen checkout; the frozen checkout itself was not modified. Both
changed Python files also pass bytecode compilation. Runtime pytest execution
remains a deployment gate on the train machine because the packaging
workstation does not have a compatible AudioCraft pytest environment.
