"""Unit tests for the framework-independent PTC-OPD loss kernel."""

import math
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F

from ptc_opd import ptc_opd_loss


def _forward_token_kl(student, teacher):
    student_logp = F.log_softmax(student.float(), dim=-1)
    teacher_logp = F.log_softmax(teacher.detach().float(), dim=-1)
    return (teacher_logp.exp() * (teacher_logp - student_logp)).sum(dim=-1)


def _reverse_token_kl(student, teacher):
    student_logp = F.log_softmax(student.float(), dim=-1)
    teacher_logp = F.log_softmax(teacher.detach().float(), dim=-1)
    return (student_logp.exp() * (student_logp - teacher_logp)).sum(dim=-1)


def _ranked_example():
    """Return logits whose JS ranking is q0:t0>... and q1:t0>...."""

    student = torch.zeros(1, 2, 4, 2, dtype=torch.float32)
    teacher = torch.zeros_like(student)
    teacher[0, 0, :, 0] = torch.tensor([8.0, 7.0, 6.0, 5.0])
    teacher[0, 1, :, 0] = torch.tensor([1.0, 0.8, 0.6, 0.4])
    valid = torch.ones(1, 2, 4, dtype=torch.bool)
    return student, teacher, valid


class TestLossEquivalence(unittest.TestCase):
    def test_identical_logits_give_numerical_zero(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 3, 4, 7)
        output = ptc_opd_loss(
            logits.clone().requires_grad_(),
            logits,
            mode="uniform",
        )
        torch.testing.assert_close(
            output.loss, torch.zeros_like(output.loss), atol=2e-7, rtol=0.0
        )

    def test_uniform_equals_valid_token_mean_forward_kl(self):
        torch.manual_seed(1)
        student = torch.randn(2, 3, 4, 5, requires_grad=True)
        student_direct = student.detach().clone().requires_grad_()
        teacher = torch.randn_like(student)
        valid = torch.tensor(
            [
                [
                    [True, True, False, False],
                    [True, False, False, False],
                    [True, True, True, False],
                ],
                [
                    [True, True, True, True],
                    [False, False, False, False],
                    [True, False, True, False],
                ],
            ]
        )

        output = ptc_opd_loss(
            student, teacher, valid_mask=valid, mode="uniform"
        )
        expected = _forward_token_kl(student_direct, teacher)[valid].mean()

        torch.testing.assert_close(output.loss, expected)
        self.assertTrue(torch.equal(output.selected_mask, valid))
        self.assertEqual(output.selection_scope, "all")
        output.loss.backward()
        expected.backward()
        torch.testing.assert_close(student.grad, student_direct.grad)

    def test_codebook_only_equals_effective_weighted_mean(self):
        torch.manual_seed(2)
        student = torch.randn(1, 2, 3, 7)
        teacher = torch.randn_like(student)
        valid = torch.tensor([[[True, True, False], [True, True, True]]])
        prior = torch.tensor([3.0, 1.0])

        output = ptc_opd_loss(
            student,
            teacher,
            valid_mask=valid,
            mode="codebook_only",
            codebook_weights=prior,
        )
        token_kl = _forward_token_kl(student, teacher)
        normalized = prior / prior.sum()
        effective = valid.float() * normalized.view(1, 2, 1)
        expected = (effective * token_kl).sum() / effective.sum()

        torch.testing.assert_close(output.loss, expected)
        torch.testing.assert_close(output.codebook_weights, normalized)
        torch.testing.assert_close(output.effective_weight_sum, effective.sum())

    def test_equal_codebook_prior_recovers_uniform_value_and_gradient(self):
        torch.manual_seed(21)
        student_uniform = torch.randn(2, 3, 4, 5, requires_grad=True)
        student_codebook = student_uniform.detach().clone().requires_grad_()
        teacher = torch.randn_like(student_uniform)
        valid = torch.rand(2, 3, 4) > 0.2
        uniform = ptc_opd_loss(
            student_uniform, teacher, valid_mask=valid, mode="uniform"
        )
        codebook = ptc_opd_loss(
            student_codebook,
            teacher,
            valid_mask=valid,
            mode="codebook_only",
            codebook_weights=torch.ones(3),
        )
        torch.testing.assert_close(uniform.loss, codebook.loss)
        uniform.loss.backward()
        codebook.loss.backward()
        torch.testing.assert_close(student_uniform.grad, student_codebook.grad)

    def test_reverse_kl_matches_manual_formula(self):
        torch.manual_seed(3)
        student = torch.randn(2, 2, 3, 5)
        teacher = torch.randn_like(student)
        valid = torch.tensor(
            [
                [[True, True, False], [True, True, True]],
                [[False, True, True], [True, False, True]],
            ]
        )
        output = ptc_opd_loss(
            student,
            teacher,
            valid_mask=valid,
            mode="uniform",
            kl_direction="reverse",
        )
        expected = _reverse_token_kl(student, teacher)[valid].mean()

        torch.testing.assert_close(output.loss, expected)
        self.assertEqual(output.kl_direction, "reverse")

    def test_temperature_one_preserves_manual_equivalence(self):
        torch.manual_seed(31)
        student = torch.randn(1, 2, 4, 5)
        teacher = torch.randn_like(student)
        default = ptc_opd_loss(student, teacher, mode="uniform")
        explicit = ptc_opd_loss(
            student, teacher, mode="uniform", temperature=1.0
        )
        torch.testing.assert_close(default.loss, explicit.loss)
        torch.testing.assert_close(explicit.loss, _forward_token_kl(student, teacher).mean())

    def test_temperature_scales_logits_and_kl_by_tau_squared(self):
        torch.manual_seed(32)
        student = torch.randn(1, 2, 3, 6)
        teacher = torch.randn_like(student)
        tau = 2.5
        output = ptc_opd_loss(
            student, teacher, mode="uniform", temperature=tau
        )
        expected = tau**2 * _forward_token_kl(student / tau, teacher / tau).mean()
        torch.testing.assert_close(output.loss, expected)
        self.assertEqual(output.temperature, tau)

        student_logp = F.log_softmax(student / tau, dim=-1)
        teacher_logp = F.log_softmax(teacher / tau, dim=-1)
        log_mixture = torch.logaddexp(student_logp, teacher_logp) - math.log(2.0)
        expected_js = 0.5 * (
            (student_logp.exp() * (student_logp - log_mixture)).sum(dim=-1)
            + (teacher_logp.exp() * (teacher_logp - log_mixture)).sum(dim=-1)
        )
        torch.testing.assert_close(output.js_divergence, expected_js)

    def test_bqtv_and_explicit_btqv_are_equivalent(self):
        torch.manual_seed(4)
        student_bqtv = torch.randn(2, 3, 4, 6)
        teacher_bqtv = torch.randn_like(student_bqtv)
        valid_bqt = torch.rand(2, 3, 4) > 0.2
        prior = torch.tensor([0.55, 0.30, 0.15])

        canonical = ptc_opd_loss(
            student_bqtv,
            teacher_bqtv,
            valid_mask=valid_bqt,
            mode="ptc",
            rho=0.5,
            codebook_weights=prior,
            layout="BQTV",
        )
        explicit_btqv = ptc_opd_loss(
            student_bqtv.permute(0, 2, 1, 3),
            teacher_bqtv.permute(0, 2, 1, 3),
            valid_mask=valid_bqt.permute(0, 2, 1),
            mode="ptc",
            rho=0.5,
            codebook_weights=prior,
            layout="BTQV",
        )

        torch.testing.assert_close(canonical.loss, explicit_btqv.loss)
        torch.testing.assert_close(canonical.token_kl, explicit_btqv.token_kl)
        self.assertTrue(
            torch.equal(canonical.selected_mask, explicit_btqv.selected_mask)
        )
        self.assertEqual(explicit_btqv.input_layout, "BTQV")


class TestSelectors(unittest.TestCase):
    def test_prefix_protocol_selects_earliest_matched_count_per_codebook(self):
        torch.manual_seed(400)
        student = torch.randn(2, 2, 6, 5)
        teacher = torch.randn_like(student)
        valid = torch.tensor(
            [
                [
                    [True, False, True, True, True, False],
                    [False, True, True, False, True, False],
                ],
                [
                    [True, True, True, True, True, True],
                    [False, False, False, False, False, False],
                ],
            ]
        )
        output = ptc_opd_loss(
            student, teacher, valid_mask=valid, mode="prefix", rho=0.5
        )
        expected = torch.tensor(
            [
                [
                    [True, False, True, False, False, False],
                    [False, True, True, False, False, False],
                ],
                [
                    [True, True, True, False, False, False],
                    [False, False, False, False, False, False],
                ],
            ]
        )
        self.assertEqual(output.mode, "prefix")
        self.assertEqual(output.selection_scope, "codebook")
        self.assertTrue(torch.equal(output.selected_mask, expected))
        self.assertTrue(
            torch.equal(
                output.selected_counts_per_codebook,
                torch.tensor([[2, 2], [3, 0]]),
            )
        )
        self.assertFalse(bool((output.selected_mask & ~valid).any().item()))

    def test_prefix_uses_uniform_weights_and_matches_other_fractional_budgets(self):
        student, teacher, valid = _ranked_example()
        prefix = ptc_opd_loss(student, teacher, valid_mask=valid, mode="prefix")
        random = ptc_opd_loss(
            student,
            teacher,
            valid_mask=valid,
            mode="random_stratified",
            random_seed=9,
        )
        disagreement = ptc_opd_loss(
            student, teacher, valid_mask=valid, mode="disagreement"
        )
        self.assertTrue(
            torch.equal(
                prefix.selected_counts_per_codebook,
                random.selected_counts_per_codebook,
            )
        )
        self.assertTrue(
            torch.equal(
                prefix.selected_counts_per_codebook,
                disagreement.selected_counts_per_codebook,
            )
        )
        torch.testing.assert_close(
            prefix.codebook_weights, torch.tensor([0.5, 0.5])
        )

    def test_protocol_ptc_is_top_rho_within_each_batch_codebook(self):
        student, teacher, valid = _ranked_example()
        output = ptc_opd_loss(
            student,
            teacher,
            valid_mask=valid,
            mode="ptc",
            rho=0.5,
            codebook_weights=torch.tensor([0.8, 0.2]),
        )
        expected = torch.tensor(
            [[[True, True, False, False], [True, True, False, False]]]
        )

        self.assertEqual(output.selection_scope, "codebook")
        self.assertEqual(output.requested_rho, 0.5)
        self.assertEqual(output.resolved_rho, 0.5)
        self.assertTrue(torch.equal(output.selected_mask, expected))
        self.assertTrue(
            torch.equal(
                output.selected_counts_per_codebook,
                torch.tensor([[2, 2]]),
            )
        )
        torch.testing.assert_close(output.selection_score, output.js_divergence)

    def test_ptc_and_disagreement_masks_match_but_weights_differ(self):
        student, teacher, valid = _ranked_example()
        disagreement = ptc_opd_loss(
            student,
            teacher,
            valid_mask=valid,
            mode="disagreement",
            rho=0.5,
        )
        ptc = ptc_opd_loss(
            student,
            teacher,
            valid_mask=valid,
            mode="ptc",
            rho=0.5,
            codebook_weights=torch.tensor([0.9, 0.1]),
        )

        self.assertTrue(torch.equal(disagreement.selected_mask, ptc.selected_mask))
        torch.testing.assert_close(
            disagreement.codebook_weights, torch.tensor([0.5, 0.5])
        )
        torch.testing.assert_close(
            ptc.codebook_weights, torch.tensor([0.9, 0.1])
        )
        self.assertFalse(
            torch.equal(disagreement.effective_weights, ptc.effective_weights)
        )

    def test_equal_prior_ptc_equals_disagreement_value_and_gradient(self):
        torch.manual_seed(41)
        student_disagreement = torch.randn(2, 3, 5, 6, requires_grad=True)
        student_ptc = student_disagreement.detach().clone().requires_grad_()
        teacher = torch.randn_like(student_disagreement)
        valid = torch.rand(2, 3, 5) > 0.2
        disagreement = ptc_opd_loss(
            student_disagreement,
            teacher,
            valid_mask=valid,
            mode="disagreement",
            rho=0.5,
        )
        ptc = ptc_opd_loss(
            student_ptc,
            teacher,
            valid_mask=valid,
            mode="ptc",
            rho=0.5,
            codebook_weights=torch.ones(3),
        )
        self.assertTrue(torch.equal(disagreement.selected_mask, ptc.selected_mask))
        torch.testing.assert_close(disagreement.loss, ptc.loss)
        disagreement.loss.backward()
        ptc.loss.backward()
        torch.testing.assert_close(
            student_disagreement.grad, student_ptc.grad
        )

    def test_sequence_global_is_explicit_diagnostic_not_default(self):
        student, teacher, valid = _ranked_example()
        primary = ptc_opd_loss(
            student, teacher, valid_mask=valid, mode="ptc", rho=0.5
        )
        global_diagnostic = ptc_opd_loss(
            student,
            teacher,
            valid_mask=valid,
            mode="ptc",
            rho=0.5,
            selection_scope="global_sequence",
        )

        self.assertEqual(primary.selection_scope, "codebook")
        self.assertEqual(global_diagnostic.selection_scope, "sequence")
        self.assertTrue(
            torch.equal(
                primary.selected_counts_per_codebook,
                torch.tensor([[2, 2]]),
            )
        )
        self.assertTrue(
            torch.equal(
                global_diagnostic.selected_counts_per_codebook,
                torch.tensor([[4, 0]]),
            )
        )

    def test_global_joint_ptc_ranks_aq_times_js(self):
        student = torch.zeros(1, 2, 2, 2)
        teacher = torch.zeros_like(student)
        # q0 has much higher raw disagreement, but its very small prior makes
        # q1 win the explicit joint diagnostic ranking.
        teacher[0, 0, :, 0] = torch.tensor([8.0, 7.0])
        teacher[0, 1, :, 0] = torch.tensor([1.0, 0.8])
        prior = torch.tensor([0.001, 0.999])
        primary = ptc_opd_loss(
            student,
            teacher,
            mode="ptc",
            rho=0.5,
            codebook_weights=prior,
        )
        diagnostic = ptc_opd_loss(
            student,
            teacher,
            mode="ptc",
            rho=0.5,
            codebook_weights=prior,
            selection_scope="global_joint_selector",
        )

        self.assertTrue(
            torch.equal(
                primary.selected_counts_per_codebook,
                torch.tensor([[1, 1]]),
            )
        )
        self.assertTrue(
            torch.equal(
                diagnostic.selected_counts_per_codebook,
                torch.tensor([[0, 2]]),
            )
        )
        torch.testing.assert_close(
            diagnostic.selection_score,
            diagnostic.js_divergence * prior.view(1, 2, 1),
        )

    def test_random_protocol_is_deterministic_and_stratified_with_padding(self):
        torch.manual_seed(5)
        student = torch.randn(2, 2, 5, 4)
        teacher = torch.randn_like(student)
        valid = torch.tensor(
            [
                [
                    [True, True, True, True, True],
                    [True, True, True, False, False],
                ],
                [
                    [True, False, False, False, False],
                    [False, False, False, False, False],
                ],
            ]
        )
        first = ptc_opd_loss(
            student,
            teacher,
            valid_mask=valid,
            mode="random_stratified",
            rho=0.4,
            random_seed=2027,
        )
        second = ptc_opd_loss(
            student,
            teacher,
            valid_mask=valid,
            mode="random_stratified",
            rho=0.4,
            random_seed=2027,
        )

        self.assertTrue(torch.equal(first.selected_mask, second.selected_mask))
        self.assertTrue(
            torch.equal(
                first.selected_counts_per_codebook,
                torch.tensor([[2, 2], [1, 0]]),
            )
        )
        self.assertFalse(bool((first.selected_mask & ~valid).any().item()))
        self.assertEqual(first.selection_scope, "codebook")

        targeted = ptc_opd_loss(
            student,
            teacher,
            valid_mask=valid,
            mode="disagreement",
            rho=0.4,
        )
        self.assertTrue(
            torch.equal(
                first.selected_counts_per_codebook,
                targeted.selected_counts_per_codebook,
            )
        )

    def test_seeded_generators_reproduce_random_mask(self):
        student = torch.zeros(1, 2, 8, 3)
        teacher = torch.ones_like(student)
        valid = torch.ones(1, 2, 8, dtype=torch.bool)
        first_generator = torch.Generator().manual_seed(99)
        second_generator = torch.Generator().manual_seed(99)

        first = ptc_opd_loss(
            student,
            teacher,
            valid_mask=valid,
            mode="random_stratified",
            rho=0.5,
            generator=first_generator,
        )
        second = ptc_opd_loss(
            student,
            teacher,
            valid_mask=valid,
            mode="random_stratified",
            rho=0.5,
            generator=second_generator,
        )
        self.assertTrue(torch.equal(first.selected_mask, second.selected_mask))

    def test_generator_state_restore_reproduces_subsequent_mask(self):
        student = torch.zeros(1, 2, 8, 3)
        teacher = torch.ones_like(student)
        valid = torch.ones(1, 2, 8, dtype=torch.bool)
        generator = torch.Generator().manual_seed(2027)
        ptc_opd_loss(
            student,
            teacher,
            valid_mask=valid,
            mode="random_stratified",
            generator=generator,
        )
        resume_state = generator.get_state()
        expected_next = ptc_opd_loss(
            student,
            teacher,
            valid_mask=valid,
            mode="random_stratified",
            generator=generator,
        )

        restored = torch.Generator()
        restored.set_state(resume_state)
        actual_next = ptc_opd_loss(
            student,
            teacher,
            valid_mask=valid,
            mode="random_stratified",
            generator=restored,
        )
        self.assertTrue(
            torch.equal(expected_next.selected_mask, actual_next.selected_mask)
        )

    def test_supplied_random_scores_define_mask_exactly(self):
        student = torch.zeros(2, 2, 4, 3)
        teacher = torch.ones_like(student)
        valid = torch.tensor(
            [
                [[True, True, True, True], [True, True, False, False]],
                [[True, True, True, True], [True, True, True, True]],
            ]
        )
        scores = torch.tensor(
            [
                [[0.1, 0.9, 0.3, 0.8], [0.7, 0.2, float("nan"), float("nan")]],
                [[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.9, 0.8]],
            ]
        )
        output = ptc_opd_loss(
            student,
            teacher,
            valid_mask=valid,
            mode="random_stratified",
            rho=0.5,
            random_scores=scores,
        )
        expected = torch.tensor(
            [
                [[False, True, False, True], [True, False, False, False]],
                [[True, True, False, False], [False, False, True, True]],
            ]
        )
        self.assertTrue(torch.equal(output.selected_mask, expected))
        self.assertFalse(output.selection_score.requires_grad)
        self.assertFalse(bool((output.selected_mask & ~valid).any().item()))

    def test_supplied_random_scores_follow_explicit_btq_layout(self):
        student = torch.zeros(1, 2, 4, 3)
        teacher = torch.ones_like(student)
        scores_bqt = torch.tensor(
            [[[0.1, 0.2, 0.9, 0.8], [0.9, 0.1, 0.8, 0.2]]]
        )
        canonical = ptc_opd_loss(
            student,
            teacher,
            mode="random_stratified",
            rho=0.5,
            random_scores=scores_bqt,
        )
        btq = ptc_opd_loss(
            student.permute(0, 2, 1, 3),
            teacher.permute(0, 2, 1, 3),
            mode="random_stratified",
            rho=0.5,
            layout="BTQV",
            random_scores=scores_bqt.permute(0, 2, 1),
        )
        self.assertTrue(torch.equal(canonical.selected_mask, btq.selected_mask))

    def test_fractional_group_count_uses_ceil(self):
        student = torch.zeros(1, 1, 3, 2)
        teacher = torch.tensor([[[[[3.0, 0.0], [2.0, 0.0], [1.0, 0.0]]]]])
        # Remove the accidental extra singleton introduced for readability.
        teacher = teacher.reshape_as(student)
        output = ptc_opd_loss(
            student,
            teacher,
            mode="disagreement",
            rho=0.5,
        )
        self.assertEqual(int(output.selected_mask.sum().item()), 2)


class TestBoundariesAndNumerics(unittest.TestCase):
    def test_rho_zero_is_differentiable_exact_zero(self):
        torch.manual_seed(6)
        student = torch.randn(1, 2, 3, 4, requires_grad=True)
        teacher = torch.randn_like(student)
        output = ptc_opd_loss(
            student, teacher, mode="ptc", rho=0.0, allow_empty=True
        )

        self.assertEqual(float(output.loss.item()), 0.0)
        self.assertEqual(float(output.effective_weight_sum.item()), 0.0)
        self.assertFalse(bool(output.selected_mask.any().item()))
        output.loss.backward()
        torch.testing.assert_close(student.grad, torch.zeros_like(student))

    def test_rho_one_ptc_equals_codebook_only(self):
        torch.manual_seed(7)
        student = torch.randn(2, 3, 4, 5)
        teacher = torch.randn_like(student)
        valid = torch.rand(2, 3, 4) > 0.25
        weights = torch.tensor([0.6, 0.3, 0.1])

        ptc = ptc_opd_loss(
            student,
            teacher,
            valid_mask=valid,
            mode="ptc",
            rho=1.0,
            codebook_weights=weights,
        )
        codebook_only = ptc_opd_loss(
            student,
            teacher,
            valid_mask=valid,
            mode="codebook_only",
            codebook_weights=weights,
        )

        torch.testing.assert_close(ptc.loss, codebook_only.loss)
        self.assertTrue(torch.equal(ptc.selected_mask, valid))

    def test_mode_specific_rho_defaults_are_resolved(self):
        student = torch.zeros(1, 1, 4, 2)
        teacher = torch.tensor([[[[4.0, 0.0], [3.0, 0.0], [2.0, 0.0], [1.0, 0.0]]]])
        uniform = ptc_opd_loss(student, teacher, mode="uniform")
        ptc = ptc_opd_loss(student, teacher, mode="ptc")
        self.assertIsNone(uniform.requested_rho)
        self.assertIsNone(ptc.requested_rho)
        self.assertEqual(uniform.resolved_rho, 1.0)
        self.assertEqual(ptc.resolved_rho, 0.5)
        self.assertEqual(int(uniform.selected_mask.sum().item()), 4)
        self.assertEqual(int(ptc.selected_mask.sum().item()), 2)

    def test_teacher_and_selector_are_detached_student_gradient_is_finite(self):
        torch.manual_seed(8)
        student = torch.randn(1, 2, 4, 5, requires_grad=True)
        teacher = torch.randn(1, 2, 4, 5, requires_grad=True)
        output = ptc_opd_loss(
            student,
            teacher,
            mode="ptc",
            rho=0.5,
            codebook_weights=torch.tensor([0.7, 0.3]),
        )
        output.loss.backward()

        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(student.grad)
        self.assertTrue(bool(torch.isfinite(student.grad).all().item()))
        self.assertFalse(output.selection_score.requires_grad)
        self.assertFalse(output.js_divergence.requires_grad)
        unselected_gradient = student.grad[~output.selected_mask]
        torch.testing.assert_close(
            unselected_gradient, torch.zeros_like(unselected_gradient)
        )

    def test_zero_valid_tokens_with_padding_garbage_returns_safe_zero(self):
        student = torch.full((1, 2, 3, 4), float("nan"), requires_grad=True)
        teacher = torch.full_like(student, float("inf"))
        valid = torch.zeros(1, 2, 3, dtype=torch.bool)

        output = ptc_opd_loss(
            student,
            teacher,
            valid_mask=valid,
            mode="ptc",
            rho=0.5,
            allow_empty=True,
        )
        self.assertTrue(math.isfinite(float(output.loss.item())))
        self.assertEqual(float(output.loss.item()), 0.0)
        self.assertEqual(float(output.realized_retention.item()), 0.0)
        self.assertEqual(int(output.valid_counts_per_codebook.sum().item()), 0)
        output.loss.backward()
        torch.testing.assert_close(student.grad, torch.zeros_like(student))

    def test_nonfinite_padding_is_ignored(self):
        student = torch.randn(1, 1, 3, 4)
        teacher = torch.randn_like(student)
        valid = torch.tensor([[[True, True, False]]])
        student[0, 0, 2] = float("nan")
        teacher[0, 0, 2] = float("inf")
        student.requires_grad_()

        output = ptc_opd_loss(
            student, teacher, valid_mask=valid, mode="uniform"
        )
        self.assertTrue(math.isfinite(float(output.loss.item())))
        self.assertFalse(bool(output.selected_mask[0, 0, 2].item()))
        output.loss.backward()
        torch.testing.assert_close(
            student.grad[0, 0, 2], torch.zeros_like(student.grad[0, 0, 2])
        )

    def test_nonfinite_valid_logits_raise(self):
        valid = torch.ones(1, 1, 2, dtype=torch.bool)
        for bad_value in (float("nan"), float("inf"), float("-inf")):
            student = torch.zeros(1, 1, 2, 3)
            teacher = torch.zeros_like(student)
            student[0, 0, 0, 0] = bad_value
            with self.subTest(bad_value=bad_value):
                with self.assertRaises(FloatingPointError):
                    ptc_opd_loss(
                        student,
                        teacher,
                        valid_mask=valid,
                        mode="uniform",
                    )

    def test_effective_normalization_is_invariant_to_prior_scale(self):
        torch.manual_seed(9)
        student = torch.randn(1, 2, 5, 4)
        teacher = torch.randn_like(student)
        first = ptc_opd_loss(
            student,
            teacher,
            mode="ptc",
            rho=0.6,
            codebook_weights=torch.tensor([3.0, 1.0]),
        )
        second = ptc_opd_loss(
            student,
            teacher,
            mode="ptc",
            rho=0.6,
            codebook_weights=torch.tensor([30.0, 10.0]),
        )
        torch.testing.assert_close(first.loss, second.loss)
        self.assertTrue(torch.equal(first.selected_mask, second.selected_mask))


class TestInputValidation(unittest.TestCase):
    def setUp(self):
        self.student = torch.zeros(1, 2, 3, 4)
        self.teacher = torch.ones_like(self.student)

    def test_invalid_rho_rejected(self):
        for rho in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(rho=rho), self.assertRaises(ValueError):
                ptc_opd_loss(self.student, self.teacher, rho=rho)

    def test_all_position_modes_reject_fractional_rho(self):
        for mode in ("uniform", "codebook_only"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                ptc_opd_loss(
                    self.student, self.teacher, mode=mode, rho=0.5
                )

    def test_nonpositive_codebook_weight_rejected(self):
        with self.assertRaises(ValueError):
            ptc_opd_loss(
                self.student,
                self.teacher,
                codebook_weights=torch.tensor([1.0, 0.0]),
            )

    def test_non_boolean_mask_rejected(self):
        with self.assertRaises(TypeError):
            ptc_opd_loss(
                self.student,
                self.teacher,
                valid_mask=torch.ones(1, 2, 3),
            )

    def test_layout_is_never_guessed(self):
        with self.assertRaises(ValueError):
            ptc_opd_loss(self.student, self.teacher, layout="auto")

    def test_random_sources_are_mutually_exclusive_and_mode_scoped(self):
        random_scores = torch.zeros(1, 2, 3)
        for kwargs in (
            {
                "generator": torch.Generator().manual_seed(1),
                "random_seed": 1,
            },
            {"random_seed": 1, "random_scores": random_scores},
            {
                "generator": torch.Generator().manual_seed(1),
                "random_scores": random_scores,
            },
        ):
            with self.subTest(kwargs=sorted(kwargs)), self.assertRaises(ValueError):
                ptc_opd_loss(
                    self.student,
                    self.teacher,
                    mode="random_stratified",
                    **kwargs,
                )
        with self.assertRaises(ValueError):
            ptc_opd_loss(
                self.student,
                self.teacher,
                mode="ptc",
                random_scores=random_scores,
            )

    def test_random_scores_validate_shape_dtype_device_and_valid_finiteness(self):
        with self.assertRaises(TypeError):
            ptc_opd_loss(
                self.student,
                self.teacher,
                mode="random_stratified",
                random_scores=torch.zeros(1, 2, 3, dtype=torch.long),
            )
        with self.assertRaises(ValueError):
            ptc_opd_loss(
                self.student,
                self.teacher,
                mode="random_stratified",
                random_scores=torch.zeros(1, 3, 2),
            )
        bad = torch.zeros(1, 2, 3)
        bad[0, 0, 0] = float("inf")
        with self.assertRaises(FloatingPointError):
            ptc_opd_loss(
                self.student,
                self.teacher,
                mode="random_stratified",
                random_scores=bad,
            )

    def test_empty_batch_and_empty_selection_raise_by_default(self):
        no_valid = torch.zeros(1, 2, 3, dtype=torch.bool)
        with self.assertRaises(ValueError):
            ptc_opd_loss(
                self.student,
                self.teacher,
                valid_mask=no_valid,
                mode="uniform",
            )
        with self.assertRaises(ValueError):
            ptc_opd_loss(self.student, self.teacher, mode="ptc", rho=0.0)

    def test_invalid_temperature_rejected(self):
        for temperature in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                ptc_opd_loss(
                    self.student, self.teacher, temperature=temperature
                )

    def test_numerator_retains_graph_for_global_ddp_ratio(self):
        student = self.student.clone().requires_grad_()
        output = ptc_opd_loss(student, self.teacher, mode="uniform")
        self.assertTrue(output.numerator.requires_grad)
        self.assertFalse(output.effective_weight_sum.requires_grad)
        output.numerator.backward()
        self.assertIsNotNone(student.grad)


if __name__ == "__main__":
    unittest.main()
