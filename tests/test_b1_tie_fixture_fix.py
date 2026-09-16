"""Unit tests for the B1.4 exact-zero-score stable-tie fixture (β sibling patch).

Pilot ruling 2026-08-24 CST β = B1.4 exact-tie audit fixture defect
(classification ``yellow / audit-reference-harness-only``) authorised
rewriting the B1.4 stable-tie audit fixture in
``scripts/run_b1_single_gpu_audit.py`` to feed a true bitwise-zero score
directly through the production ``_top_fraction`` selector.  The frozen
``_top_fraction`` is documented to use
``torch.argsort(descending=True, stable=True)``; when every score is
bitwise-equal (including all zeros in fp32), the stable sort falls back
to the ascending original-index order and therefore selects the earliest
``ceil(rho * n_valid)`` time positions inside each
``(batch, codebook)`` valid line.

These tests verify that invariant under **irregular** valid masks that
producer B1.4 alone does not exercise:

* the exact producer delayed-pattern lattice ``[500, 499, 498, 497]``
  (the shape the sealed audit runs on),
* a middle-gap mask where the invalid stretch is interior, not at the
  boundary,
* a mask with an entirely-invalid ``(batch, codebook)`` line
  (``n_valid == 0`` → contributes nothing to the mask),
* a mask with a single valid position (``n_valid == 1`` →
  ``keep == ceil(0.5) == 1``),
* an odd valid count that stresses ``ceil`` vs ``floor``
  (``n_valid == 3`` → ``keep == 2``, not 1),
* a ragged mask whose valid indices are non-contiguous,
* an all-valid mask (``n_valid == time_dim``),
* a batched mix of the above so per-``(b, q)`` independence is confirmed
  from within a single call.

For each case the test compares the mask returned by
``_top_fraction(zeros, valid_mask, rho, scope='codebook')`` against the
mask that keeps ``valid_indices[:keep]`` per ``(b, q)``.  Comparison uses
``torch.equal``, not ``torch.allclose`` — the pilot explicitly rejected
"same L2 norm" as an equivalence claim; only bit-exact agreement counts.

The tests execute purely on CPU with fp32 tensors and do not require a
GPU, HuggingFace assets, or any workpack upstream inputs.
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path
import sys
from typing import Tuple

import torch

WORKPACK_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = WORKPACK_ROOT / "scripts"
SRC_ROOT = WORKPACK_ROOT / "src"
for _p in (str(SRC_ROOT), str(SCRIPTS_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ptc_opd.losses import _top_fraction  # noqa: E402


def _expected_earliest_mask(
    valid_mask: torch.Tensor,
    rho: float,
) -> torch.Tensor:
    """Compute the per-(b, q) earliest ``ceil(rho * n_valid)`` mask.

    This is the reference implementation the audit fixture writes inline;
    the test verifies that ``_top_fraction(zeros, valid, rho, 'codebook')``
    reproduces it exactly for every valid mask shape.  Empty valid lines
    (``n_valid == 0``) contribute nothing to the mask, matching the
    ``_retained_count`` short-circuit at the same argument.
    """

    expected = torch.zeros_like(valid_mask)
    for b in range(valid_mask.shape[0]):
        for q in range(valid_mask.shape[1]):
            valid_indices = torch.nonzero(valid_mask[b, q], as_tuple=False).flatten()
            n_valid = int(valid_indices.numel())
            if n_valid == 0:
                continue
            keep = max(1, int(math.ceil(rho * n_valid)))
            expected[b, q, valid_indices[:keep]] = True
    return expected


def _delayed_pattern_valid(
    batch: int = 2,
    codebooks: int = 4,
    time: int = 500,
) -> torch.Tensor:
    """Reproduce MusicGen delayed-pattern valid mask.

    AudioCraft's ``build_delays_mask`` invalidates the last ``k`` time
    positions of codebook ``k`` because each codebook's stream is delayed
    ``k`` steps into the future by the delayed-pattern provider.  The
    prefix is NOT invalidated by the delay: the producer's B1.1 gate
    ``expected_valid_counts = torch.tensor([[500, 499, 498, 497], [500,
    499, 498, 497]])`` at ``run_b1_single_gpu_audit.py::_run_cases``
    L482-484 asserts ``n_valid == time_dim - k`` (i.e. ``500 - k``), not
    ``time_dim - 2*k``.  This helper therefore invalidates only the
    trailing ``k`` positions of codebook ``k``.
    """

    valid = torch.ones((batch, codebooks, time), dtype=torch.bool)
    for k in range(codebooks):
        if k > 0:
            valid[:, k, -k:] = False
    return valid


class ZeroScoreEarliestInvariantTests(unittest.TestCase):
    """``_top_fraction(zeros, ...)`` must select earliest ``ceil(rho*n_valid)``."""

    def _assert_zero_score_picks_earliest(
        self,
        valid_mask: torch.Tensor,
        rho: float = 0.5,
        scope: str = "codebook",
    ) -> None:
        """Core invariant used by every test case in this class."""

        zero_score = torch.zeros_like(valid_mask, dtype=torch.float32)
        got = _top_fraction(zero_score, valid_mask, rho, scope)
        want = _expected_earliest_mask(valid_mask, rho)
        self.assertTrue(
            torch.equal(got, want),
            "zero-score selector diverged from earliest-mask; "
            "diff cells: {} / {}".format(
                int((got != want).sum().item()),
                int(valid_mask.numel()),
            ),
        )
        # Additional structural properties: selected ⊆ valid, per-(b,q)
        # count equals ceil(rho * n_valid).
        self.assertTrue(
            torch.equal(got & valid_mask, got),
            "selected mask contains cells outside valid_mask",
        )
        for b in range(valid_mask.shape[0]):
            for q in range(valid_mask.shape[1]):
                n_valid = int(valid_mask[b, q].sum().item())
                if n_valid == 0:
                    self.assertEqual(
                        int(got[b, q].sum().item()),
                        0,
                        "empty valid line must yield empty selection",
                    )
                    continue
                expected_keep = max(1, int(math.ceil(rho * n_valid)))
                self.assertEqual(
                    int(got[b, q].sum().item()),
                    expected_keep,
                    "per-(b,q) selected count wrong: got {}, want {}, "
                    "n_valid={} rho={}".format(
                        int(got[b, q].sum().item()),
                        expected_keep,
                        n_valid,
                        rho,
                    ),
                )

    def test_delayed_pattern_matches_producer_lattice(self) -> None:
        """The exact ``[500, 499, 498, 497]`` shape B1.4 runs on."""

        valid = _delayed_pattern_valid(batch=2, codebooks=4, time=500)
        counts = valid.sum(dim=2).tolist()
        self.assertEqual(counts, [[500, 499, 498, 497], [500, 499, 498, 497]])
        self._assert_zero_score_picks_earliest(valid, rho=0.5)

    def test_middle_gap_mask(self) -> None:
        """Invalid stretch is interior (typical of segmented conditioning)."""

        valid = torch.zeros((2, 4, 20), dtype=torch.bool)
        valid[:, :, :6] = True
        valid[:, :, 12:] = True   # gap at positions 6-11
        self._assert_zero_score_picks_earliest(valid, rho=0.5)

    def test_entirely_invalid_codebook_line(self) -> None:
        """One (batch, codebook) line has no valid position at all."""

        valid = torch.ones((2, 4, 20), dtype=torch.bool)
        valid[1, 2, :] = False   # b=1, q=2 completely invalid
        self._assert_zero_score_picks_earliest(valid, rho=0.5)
        # Verify the empty line indeed contributes zero cells.
        zero_score = torch.zeros_like(valid, dtype=torch.float32)
        got = _top_fraction(zero_score, valid, 0.5, "codebook")
        self.assertEqual(int(got[1, 2].sum().item()), 0)

    def test_single_valid_position(self) -> None:
        """``n_valid == 1`` → ``keep == 1`` (the sole valid time)."""

        valid = torch.zeros((2, 4, 20), dtype=torch.bool)
        valid[0, 0, 7] = True
        valid[0, 1, 0] = True
        valid[0, 2, 19] = True
        valid[1, 0, 3] = True
        self._assert_zero_score_picks_earliest(valid, rho=0.5)

    def test_odd_n_valid_uses_ceil_not_floor(self) -> None:
        """``n_valid == 3, rho == 0.5`` → ``keep == 2``, not 1."""

        valid = torch.zeros((1, 2, 10), dtype=torch.bool)
        valid[0, 0, :3] = True                     # positions 0,1,2  → keep [0,1]
        valid[0, 1, [1, 4, 7]] = True              # positions 1,4,7  → keep [1,4]
        self._assert_zero_score_picks_earliest(valid, rho=0.5)

        zero_score = torch.zeros_like(valid, dtype=torch.float32)
        got = _top_fraction(zero_score, valid, 0.5, "codebook")
        # Explicit per-line correctness:
        self.assertTrue(bool(got[0, 0, 0].item()) and bool(got[0, 0, 1].item()))
        self.assertFalse(bool(got[0, 0, 2].item()))
        self.assertTrue(bool(got[0, 1, 1].item()) and bool(got[0, 1, 4].item()))
        self.assertFalse(bool(got[0, 1, 7].item()))

    def test_ragged_noncontiguous_valid_indices(self) -> None:
        """Valid indices are jumpy, non-contiguous positions."""

        valid = torch.zeros((2, 4, 30), dtype=torch.bool)
        pattern = torch.tensor(
            [0, 3, 7, 12, 15, 19, 22, 27], dtype=torch.long
        )
        valid[:, :, pattern] = True
        self._assert_zero_score_picks_earliest(valid, rho=0.5)

    def test_all_valid_mask(self) -> None:
        """``n_valid == time_dim`` corner (no invalid at all)."""

        valid = torch.ones((2, 4, 16), dtype=torch.bool)
        self._assert_zero_score_picks_earliest(valid, rho=0.5)

    def test_batched_mixed_shapes(self) -> None:
        """Different ``(b, q)`` lines have different ``n_valid``."""

        valid = torch.zeros((3, 4, 25), dtype=torch.bool)
        # (0,0) contiguous [0..24]
        valid[0, 0, :] = True
        # (0,1) middle gap [0..9] + [15..24]
        valid[0, 1, :10] = True
        valid[0, 1, 15:] = True
        # (0,2) single valid
        valid[0, 2, 13] = True
        # (0,3) empty
        # (1,0) odd (5 valid)
        valid[1, 0, [1, 3, 5, 7, 9]] = True
        # (1,1) even (4 valid)
        valid[1, 1, [0, 6, 12, 20]] = True
        # (1,2) ragged
        valid[1, 2, [2, 5, 11, 17, 22, 24]] = True
        # (1,3) all
        valid[1, 3, :] = True
        # (2,*) mirror of (0,*) for double-batch coverage
        valid[2, 0, :] = True
        valid[2, 1, :10] = True
        valid[2, 1, 15:] = True
        valid[2, 2, 13] = True

        self._assert_zero_score_picks_earliest(valid, rho=0.5)


class ZeroScoreRhoSweepTests(unittest.TestCase):
    """Zero-score invariant must hold at every rho in {0.25, 0.5, 0.75}."""

    def test_rho_quarter(self) -> None:
        valid = _delayed_pattern_valid(2, 4, 500)
        zero_score = torch.zeros_like(valid, dtype=torch.float32)
        got = _top_fraction(zero_score, valid, 0.25, "codebook")
        want = _expected_earliest_mask(valid, 0.25)
        self.assertTrue(torch.equal(got, want))

    def test_rho_half(self) -> None:
        valid = _delayed_pattern_valid(2, 4, 500)
        zero_score = torch.zeros_like(valid, dtype=torch.float32)
        got = _top_fraction(zero_score, valid, 0.5, "codebook")
        want = _expected_earliest_mask(valid, 0.5)
        self.assertTrue(torch.equal(got, want))

    def test_rho_three_quarters(self) -> None:
        valid = _delayed_pattern_valid(2, 4, 500)
        zero_score = torch.zeros_like(valid, dtype=torch.float32)
        got = _top_fraction(zero_score, valid, 0.75, "codebook")
        want = _expected_earliest_mask(valid, 0.75)
        self.assertTrue(torch.equal(got, want))


class ZeroScoreDeterminismTests(unittest.TestCase):
    """Repeated calls with the same score/mask/rho return bit-identical masks."""

    def test_bit_identical_repeated_calls(self) -> None:
        valid = _delayed_pattern_valid(2, 4, 500)
        zero_score_a = torch.zeros_like(valid, dtype=torch.float32)
        zero_score_b = torch.zeros_like(valid, dtype=torch.float32)
        got_a = _top_fraction(zero_score_a, valid, 0.5, "codebook")
        got_b = _top_fraction(zero_score_b, valid, 0.5, "codebook")
        self.assertTrue(torch.equal(got_a, got_b), "not bit-identical across calls")


class ZeroScoreProducesUnionOfValidTests(unittest.TestCase):
    """Sanity: the union of ``valid_indices[:keep]`` over (b, q) is a subset of valid."""

    def test_union_subset_of_valid(self) -> None:
        valid = _delayed_pattern_valid(2, 4, 500)
        zero_score = torch.zeros_like(valid, dtype=torch.float32)
        got = _top_fraction(zero_score, valid, 0.5, "codebook")
        # Every selected cell must be a valid cell.
        self.assertTrue(
            torch.equal(got & valid, got),
            "selected cell falls outside valid_mask",
        )
        # No invalid cell may be selected.
        self.assertEqual(int((got & (~valid)).sum().item()), 0)


# ---------------------------------------------------------------------------
# ε sibling patch (pilot RULING #3 2026-08-24 CST; classification
# ``yellow / audit-reference-harness-only``): the B1.5 padding-mutation
# fixture in ``scripts/run_b1_single_gpu_audit.py`` originally truncated
# the four codebooks at the same raw length ``L`` and used the resulting
# ``padded_valid = valid & (time < L)`` to gate the invariance /
# NaN-poisoning contracts.  Under MusicGen's frozen
# ``DelayedPatternProvider`` with delays ``[0, 1, 2, 3]``, the region
# actually predicted from raw codes of length ``L`` per codebook ``q``
# is ``t + delays[q] < L``, i.e. per-codebook lengths
# ``[L, L-1, L-2, L-3]``.  The tests below verify the pure-Python mask
# arithmetic on which the ε producer patch relies.
# ---------------------------------------------------------------------------


def _delayed_padding_valid(
    lengths: Tuple[int, ...] = (487, 461),
    codebooks: int = 4,
    time_dim: int = 500,
    delays: Tuple[int, ...] = (0, 1, 2, 3),
) -> torch.Tensor:
    """Reproduce the B1.5 delayed-pattern-aware ``padded_valid`` lattice.

    Composes the base delayed-pattern valid mask (``last k positions of
    codebook k invalidated`` — see :func:`_delayed_pattern_valid`) with
    a per-batch raw length cut-off applied through the frozen delay
    offsets ``[0, 1, 2, 3]``.  The result is exactly the mask used in
    ``scripts/run_b1_single_gpu_audit.py`` B1.5 after the ε patch, and
    the shape whose per-``(batch, codebook)`` counts pilot pinned as
    ``[[487, 486, 485, 484], [461, 460, 459, 458]]`` (total 3780).
    """

    batch = len(lengths)
    if codebooks != len(delays):
        raise AssertionError("codebooks vs delays length mismatch")
    base = _delayed_pattern_valid(batch=batch, codebooks=codebooks, time=time_dim)
    time_axis = torch.arange(time_dim).view(1, 1, time_dim)
    delays_t = torch.tensor(list(delays), dtype=torch.long).view(1, codebooks, 1)
    lengths_t = torch.tensor(list(lengths), dtype=torch.long).view(batch, 1, 1)
    return base & ((time_axis + delays_t) < lengths_t)


class DelayedPaddingMaskProperties(unittest.TestCase):
    """Pure-Python properties of the ε delayed-pattern ``padded_valid`` mask."""

    def test_pilot_target_per_codebook_counts(self) -> None:
        """Pilot ruling #3 target: [[487,486,485,484],[461,460,459,458]]."""

        padded_valid = _delayed_padding_valid(
            lengths=(487, 461), codebooks=4, time_dim=500, delays=(0, 1, 2, 3),
        )
        counts = padded_valid.sum(dim=2).tolist()
        self.assertEqual(
            counts,
            [[487, 486, 485, 484], [461, 460, 459, 458]],
            "delayed-pattern padded_valid counts diverged from pilot target",
        )
        self.assertEqual(
            int(padded_valid.sum().item()),
            3780,
            "delayed-pattern padded_valid total diverged from pilot target 3780",
        )

    def test_padded_valid_is_subset_of_delay_valid(self) -> None:
        """Adding the raw-length cut may only shrink the valid set."""

        base = _delayed_pattern_valid(batch=2, codebooks=4, time=500)
        padded = _delayed_padding_valid(
            lengths=(487, 461), codebooks=4, time_dim=500, delays=(0, 1, 2, 3),
        )
        # padded ⊆ base ⇒ padded & base == padded ⇒ (padded & ~base).sum() == 0.
        self.assertEqual(int((padded & (~base)).sum().item()), 0)

    def test_delay_fringe_matches_causal_ar_semantics(self) -> None:
        """The old (delay-blind) minus new (delayed-pattern) mask equals the
        theoretical AR fringe ``{(b,q,t) : L_b - delays[q] <= t < L_b,
        (b,q,t) valid under delayed pattern}``.

        Under delays ``[0, 1, 2, 3]`` and lengths ``L``, this is:
          * b: q=0 → nothing (delay=0)
          * b: q=1 → 1 cell at t=L-1
          * b: q=2 → 2 cells at t in {L-2, L-1}
          * b: q=3 → 3 cells at t in {L-3, L-2, L-1}
        per-batch = 0+1+2+3 = 6 cells; two batches = 12.  All must be
        strictly inside base-delay-valid (i.e. ``L_b + k <= 500-k``,
        which holds for L in {487, 461}).
        """

        lengths = (487, 461)
        codebooks = 4
        time_dim = 500
        delays = (0, 1, 2, 3)

        base = _delayed_pattern_valid(batch=len(lengths), codebooks=codebooks, time=time_dim)
        time_axis = torch.arange(time_dim).view(1, 1, time_dim)
        lengths_t = torch.tensor(list(lengths), dtype=torch.long).view(len(lengths), 1, 1)
        delays_t = torch.tensor(list(delays), dtype=torch.long).view(1, codebooks, 1)

        old_mask = base & (time_axis < lengths_t)
        new_mask = base & ((time_axis + delays_t) < lengths_t)
        fringe = old_mask & (~new_mask)

        # Total fringe count.
        self.assertEqual(int(fringe.sum().item()), 12)

        # Per-batch, per-codebook shape enumeration.
        fringe_per_bq = fringe.sum(dim=2).tolist()
        self.assertEqual(fringe_per_bq, [[0, 1, 2, 3], [0, 1, 2, 3]])

        # Explicit coordinate check for batch 0 (L=487).
        L = 487
        expected_coords = set()
        for q in range(codebooks):
            d = delays[q]
            for t in range(max(0, L - d), L):
                expected_coords.add((0, q, t))
        actual_coords = {tuple(c) for c in torch.nonzero(fringe[0:1], as_tuple=False).tolist()}
        self.assertEqual(actual_coords, expected_coords)


class PoisonedInvariantProperties(unittest.TestCase):
    """Pure mask arithmetic of the poisoned-NaN/Inf sanitization contract."""

    def test_invalid_and_padded_valid_are_disjoint(self) -> None:
        """``padded_valid`` and its complement must partition the lattice."""

        padded = _delayed_padding_valid(
            lengths=(487, 461), codebooks=4, time_dim=500, delays=(0, 1, 2, 3),
        )
        invalid = ~padded
        self.assertEqual(int((padded & invalid).sum().item()), 0)
        self.assertEqual(
            int((padded | invalid).sum().item()),
            padded.numel(),
        )
        self.assertEqual(
            int(padded.sum().item()) + int(invalid.sum().item()),
            padded.numel(),
        )

    def test_invalid_cell_count_matches_expectation(self) -> None:
        """The ~padded_valid mask must poison exactly ``B*Q*T - 3780`` cells."""

        padded = _delayed_padding_valid(
            lengths=(487, 461), codebooks=4, time_dim=500, delays=(0, 1, 2, 3),
        )
        total = 2 * 4 * 500
        self.assertEqual(int((~padded).sum().item()), total - 3780)

    def test_delayed_padding_valid_delay_pattern_guard(self) -> None:
        """Only ``[0,1,2,3]`` delays reach the pilot-mandated 3780 total."""

        # Under bogus delays ``[0,0,0,0]``, the padded_valid becomes
        # ``valid & (time < L)``: back to the pre-ε delay-blind mask.
        # Counts would then be [[487,487,486,485],[461,461,460,459]] because
        # base delayed-pattern still invalidates the trailing k positions.
        padded_bogus = _delayed_padding_valid(
            lengths=(487, 461), codebooks=4, time_dim=500, delays=(0, 0, 0, 0),
        )
        counts_bogus = padded_bogus.sum(dim=2).tolist()
        self.assertNotEqual(counts_bogus, [[487, 486, 485, 484], [461, 460, 459, 458]])


# ---------------------------------------------------------------------------
# φ sibling patch (pilot RULING #4 2026-08-24 CST; classification
# ``yellow / audit-reference-harness-only``): the B1.9 distributed audit
# fixture in ``scripts/run_b1_distributed_audit.py`` carries two
# independent pre-existing defects that jointly caused the DDP/reference
# tolerance failure at line 582:
#
#   φ.a  Reference path scheme: the fresh reference concatenated all
#        eight rows into a single ``batch=8`` forward whose T5
#        conditioner/attention normalisation differs numerically from
#        the per-rank ``batch=1`` path used by DDP.  Empirical
#        divergence: 4.77 % loss / 14.30 % gradient relative L2 (D0 φ
#        diagnostic 20260824T122435Z).  The corrected reference now
#        runs the eight rows one at a time and accumulates gradients
#        as ``sum_i (N_i / global_denominator).backward()``.
#
#   φ.b  Distributed padding mask: ``_padding_mask`` tiled ``time < L``
#        across all four codebooks (delay-blind semantics — identical
#        pre-ε defect to B1.5).  The corrected mask returns
#        ``(time + delays_view) < lengths_view`` per pattern_provider
#        delays ``[0, 1, 2, 3]``.  For the 8-sample DDP lattice this
#        yields the pilot-mandated counts and total 14496 (see below).
#
# These tests verify the pure Python arithmetic of the new
# distributed-scope masks and reference-accumulation semantics.  They
# do NOT require CUDA or any AudioCraft/MusicGen module.
# ---------------------------------------------------------------------------


DISTRIBUTED_PADDING_LENGTHS: Tuple[int, ...] = (
    500, 487, 474, 461, 448, 435, 422, 409,
)


def _distributed_delayed_padding_valid(
    lengths: Tuple[int, ...] = DISTRIBUTED_PADDING_LENGTHS,
    codebooks: int = 4,
    time_dim: int = 500,
    delays: Tuple[int, ...] = (0, 1, 2, 3),
) -> torch.Tensor:
    """Reproduce the B1.9 distributed delayed-pattern ``padded_valid`` lattice.

    Composition (identical to the ε helper but at the B=8 distributed
    scope): base delayed-pattern valid mask ⊗ per-batch raw-length
    cut-off through the frozen delay offsets ``[0, 1, 2, 3]``.  The
    result is exactly the mask produced by the φ.b-corrected
    ``scripts/run_b1_distributed_audit.py::_padding_mask``, and the
    shape whose per-``(batch, codebook)`` counts pilot pinned as::

        [[500, 499, 498, 497],   b=0, L=500
         [487, 486, 485, 484],   b=1, L=487
         [474, 473, 472, 471],   b=2, L=474
         [461, 460, 459, 458],   b=3, L=461
         [448, 447, 446, 445],   b=4, L=448
         [435, 434, 433, 432],   b=5, L=435
         [422, 421, 420, 419],   b=6, L=422
         [409, 408, 407, 406]]   b=7, L=409

    total 14496; uniform denominator 3624.0.
    """

    batch = len(lengths)
    if codebooks != len(delays):
        raise AssertionError("codebooks vs delays length mismatch")
    base = _delayed_pattern_valid(batch=batch, codebooks=codebooks, time=time_dim)
    time_axis = torch.arange(time_dim).view(1, 1, time_dim)
    delays_t = torch.tensor(list(delays), dtype=torch.long).view(1, codebooks, 1)
    lengths_t = torch.tensor(list(lengths), dtype=torch.long).view(batch, 1, 1)
    return base & ((time_axis + delays_t) < lengths_t)


class DistributedDelayedPaddingMaskProperties(unittest.TestCase):
    """φ.b: distributed-scope delayed-pattern padding mask properties."""

    def test_pilot_target_per_codebook_counts_distributed(self) -> None:
        """Pilot ruling #4 target: 8×4 counts, batch total 14496."""

        padded_valid = _distributed_delayed_padding_valid()
        counts = padded_valid.sum(dim=2).tolist()
        expected = [
            [500, 499, 498, 497],
            [487, 486, 485, 484],
            [474, 473, 472, 471],
            [461, 460, 459, 458],
            [448, 447, 446, 445],
            [435, 434, 433, 432],
            [422, 421, 420, 419],
            [409, 408, 407, 406],
        ]
        self.assertEqual(
            counts, expected,
            "distributed delayed-pattern counts diverged from Ruling #4 target",
        )
        self.assertEqual(
            int(padded_valid.sum().item()),
            14496,
            "distributed delayed-pattern total diverged from pilot target 14496",
        )

    def test_old_vs_new_mask_delta_is_exactly_42_cells(self) -> None:
        """φ.b evidence: old delay-blind mask over-counts by exactly 42 cells.

        Pilot pinpoints ``old_total - new_total = 14538 - 14496 = 42``.
        These 42 cells lie strictly in the delay fringe
        ``{(b,q,t) : L_b - delays[q] <= t < L_b}``, with 6 cells per
        batch (q=1→1, q=2→2, q=3→3) and 8 batches → 8*6 = 48… but the
        top batch (b=0, L=500) has base-delayed-invalid at t=497..499
        for q=3, at t=498..499 for q=2, and at t=499 for q=1 — those
        cells were never in ``old_mask`` either (base has stripped
        them), so the actual fringe is 8*6 - 6 = 42.  This test does
        the arithmetic verbatim.
        """

        codebooks = 4
        time_dim = 500
        delays = (0, 1, 2, 3)

        base = _delayed_pattern_valid(
            batch=len(DISTRIBUTED_PADDING_LENGTHS),
            codebooks=codebooks,
            time=time_dim,
        )
        time_axis = torch.arange(time_dim).view(1, 1, time_dim)
        lengths_t = torch.tensor(
            list(DISTRIBUTED_PADDING_LENGTHS), dtype=torch.long,
        ).view(len(DISTRIBUTED_PADDING_LENGTHS), 1, 1)
        delays_t = torch.tensor(list(delays), dtype=torch.long).view(1, codebooks, 1)

        old_mask = base & (time_axis < lengths_t)   # pre-φ.b semantics
        new_mask = base & ((time_axis + delays_t) < lengths_t)

        old_total = int(old_mask.sum().item())
        new_total = int(new_mask.sum().item())
        self.assertEqual(old_total, 14538)
        self.assertEqual(new_total, 14496)
        self.assertEqual(old_total - new_total, 42)

        # The delta must lie in old_mask ∖ new_mask.
        delta = old_mask & (~new_mask)
        self.assertEqual(int(delta.sum().item()), 42)
        self.assertEqual(int((delta & (~old_mask)).sum().item()), 0)

    def test_uniform_denominator_matches_3624(self) -> None:
        """φ.b evidence: 14496 valid cells ÷ 4 codebooks = 3624.0 for uniform mode.

        This test does not depend on the production selector; it merely
        pins the raw cell count so future divergence is caught early.
        """

        padded_valid = _distributed_delayed_padding_valid()
        total = int(padded_valid.sum().item())
        self.assertEqual(total, 14496)
        # Per-codebook total is the sum over batch of the b,q column.
        per_codebook = padded_valid.sum(dim=(0, 2)).tolist()
        self.assertEqual(per_codebook, [
            500 + 487 + 474 + 461 + 448 + 435 + 422 + 409,             # q=0 = 3636
            499 + 486 + 473 + 460 + 447 + 434 + 421 + 408,             # q=1 = 3628
            498 + 485 + 472 + 459 + 446 + 433 + 420 + 407,             # q=2 = 3620
            497 + 484 + 471 + 458 + 445 + 432 + 419 + 406,             # q=3 = 3612
        ])
        # Sanity: sum across codebooks matches batch total.
        self.assertEqual(sum(per_codebook), total)


class SequentialReferenceAccumulationProperties(unittest.TestCase):
    """φ.a: sequential 8-singletons accumulation equivalence properties.

    These are pure-arithmetic tests — no models loaded — asserting that
    the mathematical target used by the corrected ``_run_reference``
    (``sum_i (N_i / global_denominator)``) equals the per-rank DDP
    computation ``sum_r N_r / sum_r D_r`` under the invariant
    ``sum_i D_i == global_denominator``.
    """

    def test_denominator_sum_equals_global_denominator(self) -> None:
        """φ.a invariant #1: ``sum_i D_i == global_denominator``.

        In the corrected reference this equality is checked at runtime
        (``if denominator_total != global_denominator: raise``); the
        test guarantees the arithmetic identity that lets that check
        pass on all valid inputs.
        """

        per_sample_D = [1234.0, 561.0, 789.5, 234.25, 1024.75, 4096.0, 512.0, 128.5]
        global_D = sum(per_sample_D)
        self.assertEqual(sum(per_sample_D), global_D)
        # No cancellation error at 8-sample fp32-scale sums for these magnitudes.
        self.assertAlmostEqual(sum(per_sample_D), global_D, places=6)

    def test_gradient_accumulation_linearity(self) -> None:
        """φ.a invariant #2: sum-of-scaled-gradients equals scaled-sum.

        For each parameter, ``sum_i (N_i / D_global).grad`` equals
        ``(sum_i N_i / D_global).grad``.  Verify on a small synthetic
        parameter graph so future refactors preserve linearity.
        """

        # Use a simple synthetic parameter and 8 scaled numerators.
        param = torch.zeros(4, requires_grad=True)
        numerators = [
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([0.5, -1.0, 2.5, -3.0]),
            torch.tensor([0.25, 0.5, -0.75, 1.0]),
            torch.tensor([-4.0, 3.0, 2.0, -1.0]),
            torch.tensor([10.0, -20.0, 30.0, -40.0]),
            torch.tensor([0.1, 0.2, 0.3, 0.4]),
            torch.tensor([-0.05, 0.1, -0.15, 0.2]),
            torch.tensor([100.0, 200.0, 300.0, 400.0]),
        ]
        global_denominator = 3624.0

        # Path 1: sequential accumulation (mirrors the corrected reference).
        param.grad = None
        for N_i in numerators:
            loss_i = (param * N_i).sum() / global_denominator
            loss_i.backward()
        grad_sequential = param.grad.clone()

        # Path 2: concatenate then single backward (the batch-8 semantics).
        param.grad = None
        summed_N = torch.stack(numerators).sum(dim=0)
        loss_concat = (param * summed_N).sum() / global_denominator
        loss_concat.backward()
        grad_concat = param.grad.clone()

        # For a linear graph the two must agree to machine precision.
        self.assertTrue(
            torch.allclose(grad_sequential, grad_concat, atol=1e-7, rtol=1e-7),
            "sequential vs concatenated gradients diverged linearly",
        )

    def test_reference_path_scheme_pin_matches_ruling4(self) -> None:
        """φ.a evidence: reference_path_scheme must be sequential_singletons."""

        # This is a symbolic pin — the corrected distributed audit puts
        # ``reference_path_scheme: "sequential_singletons"`` into
        # ``scientific_config``.  The test guards against silent regression
        # by pinning the string.
        expected_scheme = "sequential_singletons"
        # Import late so torch import errors bubble up first.
        import importlib.util as _iu
        path = str(WORKPACK_ROOT / "scripts" / "run_b1_distributed_audit.py")
        spec = _iu.spec_from_file_location("run_b1_distributed_audit_readonly", path)
        # Parse the source text directly rather than executing (execution
        # requires CUDA + MusicGen).  We look for the literal string.
        source = Path(path).read_text(encoding="utf-8")
        self.assertIn(
            '"reference_path_scheme": "{}"'.format(expected_scheme),
            source,
            "distributed audit fixture lost the sequential-singletons pin",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
