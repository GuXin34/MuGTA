"""Adversarial unit tests for the gather-first ``_direct_forward_kl_loss``.

Pilot ruling ``2026-08-21T08:30Z`` (R1'-Gather, classification
``yellow / audit-reference-harness-only``) authorises a defensive rewrite
of the audit reference harness ``_direct_forward_kl_loss``.  The rewrite
must satisfy four adversarial invariants when non-finite values are
present at *invalid* logit positions (which is the by-design behaviour of
AudioCraft's ``lm.py`` delayed-pattern padding):

* the returned ``loss`` is finite,
* the returned ``token_kl`` is finite,
* the returned ``token_kl`` is exactly zero at every invalid cell,
* the gradient of ``loss`` w.r.t. the student logit leaf is finite and
  exactly zero at every invalid cell.

The tests also assert the two positive-path invariants:

* on an all-valid lattice with finite logits, the oracle agrees with a
  naive full-lattice reference to within fp64 tolerance,
* if a non-finite value appears at a **valid** cell, the oracle refuses to
  proceed (raises ``FloatingPointError``).

The tests run purely on CPU with fp32/fp64 tensors and do not require a
GPU, HuggingFace assets, or any workpack upstream inputs.  They are safe
to run in the workpack's ``tests/`` scope and are picked up by pytest.
"""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
WORKPACK_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = WORKPACK_ROOT / "scripts"
for _p in (str(SCRIPTS_ROOT),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_b1_single_gpu_audit import _direct_forward_kl_loss  # noqa: E402


def _build_toy_lattice(
    *,
    batch: int = 2,
    codebooks: int = 4,
    steps: int = 6,
    vocab: int = 8,
    seed: int = 1234,
    device: torch.device = torch.device("cpu"),
):
    """Return ``(student, teacher, valid_mask)`` with a delayed-pattern shape.

    The valid mask mimics MusicGen's delayed-pattern padding: for codebook
    ``k`` the leading ``k`` positions are invalid and the trailing ``k``
    positions are invalid as well (there is no scientific meaning to the
    exact pattern; we only need a mask that has *some* invalid cells so we
    can inject adversarial values there).
    """

    generator = torch.Generator(device=device).manual_seed(seed)
    student = torch.randn(
        (batch, codebooks, steps, vocab),
        generator=generator,
        dtype=torch.float32,
        device=device,
    )
    teacher = torch.randn(
        (batch, codebooks, steps, vocab),
        generator=generator,
        dtype=torch.float32,
        device=device,
    )
    valid_mask = torch.ones(
        (batch, codebooks, steps),
        dtype=torch.bool,
        device=device,
    )
    for k in range(codebooks):
        if k > 0:
            valid_mask[:, k, :k] = False
            valid_mask[:, k, -k:] = False
    return student, teacher, valid_mask


def _inject_invalid(
    student: torch.Tensor,
    teacher: torch.Tensor,
    valid_mask: torch.Tensor,
    fill_value: float,
) -> None:
    """In-place replace **invalid** logit rows with ``fill_value``.

    Uses index broadcasting so both the student and teacher have the same
    non-finite footprint that ``lm.py`` produces on real MusicGen runs.
    """

    invalid = ~valid_mask.unsqueeze(-1).expand_as(student)
    student.masked_fill_(invalid, fill_value)
    teacher.masked_fill_(invalid, fill_value)


class DirectForwardKLLossAdversarialTests(unittest.TestCase):
    """R1'-Gather oracle survives non-finite invalid cells with finite grad."""

    def _run_case(self, fill_value: float) -> None:
        student, teacher, valid_mask = _build_toy_lattice()
        _inject_invalid(student, teacher, valid_mask, fill_value)

        leaf = student.detach().clone().requires_grad_(True)
        loss, token_kl = _direct_forward_kl_loss(leaf, teacher, valid_mask)

        # Loss must be finite.
        self.assertTrue(
            torch.isfinite(loss).item(),
            "loss must be finite even when invalid cells hold {}".format(fill_value),
        )

        # token_kl must be finite everywhere and exactly zero at invalid cells.
        self.assertTrue(
            bool(torch.isfinite(token_kl).all().item()),
            "token_kl must be finite everywhere; invalid={}, fill={}".format(
                int((~valid_mask).sum().item()),
                fill_value,
            ),
        )
        invalid_kl = token_kl[~valid_mask]
        self.assertTrue(
            torch.all(invalid_kl == 0).item(),
            "token_kl must be EXACTLY zero at every invalid cell "
            "(got max abs {} at fill={})".format(
                float(invalid_kl.abs().max().item()) if invalid_kl.numel() else 0.0,
                fill_value,
            ),
        )

        # Gradient must be finite everywhere and exactly zero at invalid cells.
        gradient = torch.autograd.grad(loss, leaf, retain_graph=False)[0]
        self.assertTrue(
            bool(torch.isfinite(gradient).all().item()),
            "gradient must be finite everywhere; invalid fill={}".format(fill_value),
        )
        invalid_grad = gradient[(~valid_mask).unsqueeze(-1).expand_as(gradient)]
        self.assertTrue(
            torch.all(invalid_grad == 0).item(),
            "gradient must be EXACTLY zero at every invalid cell "
            "(got max abs {} at fill={})".format(
                float(invalid_grad.abs().max().item()) if invalid_grad.numel() else 0.0,
                fill_value,
            ),
        )

    def test_invalid_cells_filled_with_nan(self) -> None:
        """AudioCraft's actual by-design fill value on delayed-pattern padding."""

        self._run_case(float("nan"))

    def test_invalid_cells_filled_with_positive_inf(self) -> None:
        """Fail-open would produce inf logits; oracle must still stay finite."""

        self._run_case(float("inf"))

    def test_invalid_cells_filled_with_negative_inf(self) -> None:
        """Fail-open (opposite sign) — also common in log_softmax fault modes."""

        self._run_case(float("-inf"))

    def test_valid_cells_finite_agrees_with_naive_reference(self) -> None:
        """On an all-valid lattice, the oracle must agree with the textbook KL.

        We use the naive full-lattice softmax reference (which is safe here
        because there are no invalid cells) and require agreement to within
        fp32 relative tolerance ``1e-6`` (matching
        ``b1_prestability_contract.RELATIVE_LOSS_ERROR_LT``).
        """

        student, teacher, _ = _build_toy_lattice()
        valid_mask = torch.ones(student.shape[:-1], dtype=torch.bool)
        student.requires_grad_(True)

        loss_gather, token_kl_gather = _direct_forward_kl_loss(
            student, teacher, valid_mask
        )

        # Naive reference (safe: all valid).
        s_logp = F.log_softmax(student.detach().float(), dim=-1)
        t_logp = F.log_softmax(teacher.detach().float(), dim=-1)
        token_kl_ref = (t_logp.exp() * (t_logp - s_logp)).sum(dim=-1)
        loss_ref = token_kl_ref.mean()

        rel_loss = (
            abs(float(loss_gather.item()) - float(loss_ref.item()))
            / max(abs(float(loss_ref.item())), 1.0e-12)
        )
        self.assertLess(rel_loss, 1.0e-6, "gather oracle disagrees with naive KL")

        # token_kl agreement at every cell.
        max_abs_diff = float((token_kl_gather - token_kl_ref).abs().max().item())
        self.assertLess(
            max_abs_diff,
            1.0e-5,
            "per-cell token_kl disagrees with naive reference: max abs {}".format(
                max_abs_diff,
            ),
        )

    def test_nonfinite_at_valid_cell_raises(self) -> None:
        """If NaN/Inf leaks into a valid cell, the oracle must refuse."""

        student, teacher, valid_mask = _build_toy_lattice()
        # Inject a NaN at a valid cell of the student.
        b, k, t = 0, 0, 0
        self.assertTrue(bool(valid_mask[b, k, t].item()), "test setup: cell must be valid")
        student[b, k, t, 0] = float("nan")

        with self.assertRaises(FloatingPointError):
            _direct_forward_kl_loss(student, teacher, valid_mask)

    def test_nonfinite_at_valid_cell_teacher_raises(self) -> None:
        """Same guard applies to the teacher logits."""

        student, teacher, valid_mask = _build_toy_lattice()
        b, k, t = 0, 0, 0
        self.assertTrue(bool(valid_mask[b, k, t].item()), "test setup: cell must be valid")
        teacher[b, k, t, 0] = float("inf")

        with self.assertRaises(FloatingPointError):
            _direct_forward_kl_loss(student, teacher, valid_mask)

    def test_empty_valid_mask_raises_value_error(self) -> None:
        """Zero-valid lattice must not silently produce NaN via 0/0 mean()."""

        student, teacher, _ = _build_toy_lattice()
        valid_mask = torch.zeros(student.shape[:-1], dtype=torch.bool)

        with self.assertRaises(ValueError):
            _direct_forward_kl_loss(student, teacher, valid_mask)

    def test_token_kl_shape_matches_valid_mask(self) -> None:
        """``token_kl`` must return in the ``valid_mask`` shape, not per-vocab."""

        student, teacher, valid_mask = _build_toy_lattice()
        _inject_invalid(student, teacher, valid_mask, float("nan"))

        loss, token_kl = _direct_forward_kl_loss(student, teacher, valid_mask)
        self.assertEqual(tuple(token_kl.shape), tuple(valid_mask.shape))
        self.assertEqual(token_kl.dtype, torch.float32)
        # Sanity: loss is a scalar tensor.
        self.assertEqual(loss.dim(), 0)


class DirectForwardKLLossDeterminismTests(unittest.TestCase):
    """Two independent calls with the same inputs must return bit-identical outputs."""

    def test_bit_identical_repeated_calls(self) -> None:
        student, teacher, valid_mask = _build_toy_lattice()
        _inject_invalid(student, teacher, valid_mask, float("nan"))

        # Independent leaves for two independent grad computations.
        leaf_a = student.detach().clone().requires_grad_(True)
        leaf_b = student.detach().clone().requires_grad_(True)
        loss_a, token_kl_a = _direct_forward_kl_loss(leaf_a, teacher, valid_mask)
        loss_b, token_kl_b = _direct_forward_kl_loss(leaf_b, teacher, valid_mask)
        grad_a = torch.autograd.grad(loss_a, leaf_a, retain_graph=False)[0]
        grad_b = torch.autograd.grad(loss_b, leaf_b, retain_graph=False)[0]

        self.assertTrue(torch.equal(loss_a, loss_b), "loss not bit-identical")
        self.assertTrue(torch.equal(token_kl_a, token_kl_b), "token_kl not bit-identical")
        self.assertTrue(torch.equal(grad_a, grad_b), "gradient not bit-identical")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
