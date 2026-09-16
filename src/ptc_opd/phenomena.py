"""Phase-A2/A3 scalar maps and prompt-nested statistics.

This module contains no AudioCraft imports.  It converts already-scored
``[B, Q, T, V]`` trajectories into a compact valid-cell schema and aggregates
those records without retaining vocabulary-sized tensors.  The only model-side
dependency is :func:`ptc_opd.losses.ptc_opd_loss`, which is used to ensure that
the phenomenon probe and the training loss share exactly the same forward-KL,
detached-JS, validity, and stable top-50 implementation.

Bootstrap resampling is performed at the prompt level.  Both rollout seeds for
a prompt are kept together whenever that prompt is drawn, so rollout variation
is nested inside the prompt rather than treated as 512 independent examples.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .losses import ptc_opd_loss


CELL_SCHEMA_VERSION = "ptc-opd-disagreement-cell-v1"
SUMMARY_SCHEMA_VERSION = "ptc-opd-disagreement-summary-v1"
PRIMARY_ROLLOUT_SEEDS = (31001, 31002)
PRIMARY_BOOTSTRAP_SEED = 4702
PRIMARY_BOOTSTRAP_REPLICATES = 10000

REQUIRED_CELL_FIELDS = (
    "schema_version",
    "sample_id",
    "prompt_sha256",
    "rollout_seed",
    "q",
    "t",
    "temporal_decile",
    "js",
    "forward_kl",
    "teacher_entropy",
    "student_entropy",
    "sampled_token_logp_teacher",
    "sampled_token_logp_student",
    "a_q",
    "top50_js",
)

_VECTOR_FIELDS = (
    "valid_count",
    "weighted_count",
    "js_sum",
    "forward_kl_sum",
    "teacher_entropy_sum",
    "student_entropy_sum",
    "sampled_token_logp_teacher_sum",
    "sampled_token_logp_student_sum",
    "selected_js_sum",
    "selected_kl_sum",
)

_RATIO_METRICS = (
    "mean_js",
    "mean_forward_kl",
    "mean_teacher_entropy",
    "mean_student_entropy",
    "mean_sampled_token_logp_teacher",
    "mean_sampled_token_logp_student",
    "nominal_u",
    "weighted_w_a",
    "kl_mass_k",
    "c50_js",
    "c50_kl",
)


def prompt_sha256(prompt: str) -> str:
    """Return the canonical UTF-8 SHA-256 used instead of raw prompt text."""

    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a non-empty string")
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _validate_probe_inputs(
    student_logits: Tensor,
    teacher_logits: Tensor,
    sampled_codes: Tensor,
    valid_mask: Tensor,
    sample_ids: Sequence[str],
    prompts: Sequence[str],
    rollout_seeds: Sequence[int],
) -> Tuple[int, int, int, int]:
    if not isinstance(student_logits, Tensor) or not isinstance(teacher_logits, Tensor):
        raise TypeError("student_logits and teacher_logits must be tensors")
    if student_logits.ndim != 4 or teacher_logits.shape != student_logits.shape:
        raise ValueError("student/teacher logits must have the same [B, Q, T, V] shape")
    if not student_logits.is_floating_point() or not teacher_logits.is_floating_point():
        raise TypeError("student/teacher logits must be floating-point tensors")
    if student_logits.device != teacher_logits.device:
        raise ValueError("student/teacher logits must share a device")
    batch, codebooks, time, vocabulary = student_logits.shape
    if min(batch, codebooks, time, vocabulary) <= 0:
        raise ValueError("B, Q, T, and V must all be positive")
    if sampled_codes.shape != (batch, codebooks, time):
        raise ValueError("sampled_codes must have shape [B, Q, T]")
    if sampled_codes.dtype != torch.long:
        raise TypeError("sampled_codes must have dtype torch.long")
    if sampled_codes.device != student_logits.device:
        raise ValueError("sampled_codes and logits must share a device")
    if valid_mask.shape != (batch, codebooks, time) or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean [B, Q, T]")
    if valid_mask.device != student_logits.device:
        raise ValueError("valid_mask and logits must share a device")
    if len(sample_ids) != batch or len(prompts) != batch or len(rollout_seeds) != batch:
        raise ValueError("sample_ids, prompts, and rollout_seeds must each have length B")
    if len(set(sample_ids)) != batch:
        raise ValueError("sample_ids must be unique inside a probe batch")
    for sample_id in sample_ids:
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("every sample_id must be a non-empty string")
    for seed in rollout_seeds:
        if isinstance(seed, bool):
            raise TypeError("rollout seeds must be integers")
        int(seed)
    valid_codes = sampled_codes[valid_mask]
    if valid_codes.numel() and not bool(
        ((valid_codes >= 0) & (valid_codes < vocabulary)).all().item()
    ):
        raise ValueError("sampled token IDs at valid cells must lie in [0, V)")
    return batch, codebooks, time, vocabulary


def cell_records_from_logits(
    student_logits: Tensor,
    teacher_logits: Tensor,
    sampled_codes: Tensor,
    valid_mask: Tensor,
    *,
    sample_ids: Sequence[str],
    prompts: Sequence[str],
    rollout_seeds: Sequence[int],
    codebook_weights: Tensor,
    check_finite: bool = True,
) -> Iterator[Dict[str, object]]:
    """Yield one scalar-only record for every valid ``(b,q,t)`` cell.

    The function invokes the frozen OPD loss in ``disagreement`` mode with
    ``rho=0.5``.  Consequently ``forward_kl``, ``js``, and ``top50_js`` are not
    parallel reimplementations: they are the exact training diagnostics.  The
    sampled-token log probability is under the conditional no-CFG student.
    Raw prompt text and vocabulary-sized arrays are never emitted.
    """

    batch, codebooks, time, _ = _validate_probe_inputs(
        student_logits,
        teacher_logits,
        sampled_codes,
        valid_mask,
        sample_ids,
        prompts,
        rollout_seeds,
    )
    # The probe is diagnostic-only.  Do not preserve any student graph over
    # 256 x 2 full trajectories; the same loss kernel still supplies the exact
    # frozen scalar divergences and gate under no_grad.
    student_probe_logits = student_logits.detach()
    with torch.no_grad():
        loss_output = ptc_opd_loss(
            student_probe_logits,
            teacher_logits,
            valid_mask=valid_mask,
            mode="disagreement",
            rho=0.5,
            codebook_weights=codebook_weights,
            kl_direction="forward",
            selection_scope="protocol",
            layout="BQTV",
            temperature=1.0,
            check_finite=check_finite,
        )

    with torch.no_grad():
        safe_student = torch.where(
            valid_mask.unsqueeze(-1),
            student_logits.detach().float(),
            torch.zeros((), dtype=torch.float32, device=student_logits.device),
        )
        safe_teacher = torch.where(
            valid_mask.unsqueeze(-1),
            teacher_logits.detach().float(),
            torch.zeros((), dtype=torch.float32, device=teacher_logits.device),
        )
        student_logp = F.log_softmax(safe_student, dim=-1)
        teacher_logp = F.log_softmax(safe_teacher, dim=-1)
        student_entropy = -(student_logp.exp() * student_logp).sum(dim=-1)
        teacher_entropy = -(teacher_logp.exp() * teacher_logp).sum(dim=-1)
        safe_codes = torch.where(
            valid_mask, sampled_codes, torch.zeros_like(sampled_codes)
        )
        sampled_student_logp = student_logp.gather(
            -1, safe_codes.unsqueeze(-1)
        ).squeeze(-1)
        sampled_teacher_logp = teacher_logp.gather(
            -1, safe_codes.unsqueeze(-1)
        ).squeeze(-1)

    prompt_hashes = [prompt_sha256(prompt) for prompt in prompts]
    prior = torch.as_tensor(
        codebook_weights, dtype=torch.float32, device=student_logits.device
    ).detach()
    if prior.ndim != 1 or prior.shape[0] != codebooks:
        raise ValueError("codebook_weights must have length Q")
    if not bool(torch.isfinite(prior).all().item()) or not bool((prior > 0).all().item()):
        raise ValueError("codebook_weights must be finite and strictly positive")
    prior = (prior / prior.sum()).cpu()
    token_kl = loss_output.token_kl.detach().cpu()
    js = loss_output.js_divergence.detach().cpu()
    selected = loss_output.selected_mask.detach().cpu()
    valid_cpu = valid_mask.detach().cpu()
    student_entropy = student_entropy.detach().cpu()
    teacher_entropy = teacher_entropy.detach().cpu()
    sampled_student_logp = sampled_student_logp.detach().cpu()
    sampled_teacher_logp = sampled_teacher_logp.detach().cpu()

    for batch_index in range(batch):
        for q_index in range(codebooks):
            for time_index in torch.nonzero(
                valid_cpu[batch_index, q_index], as_tuple=False
            ).flatten().tolist():
                temporal_decile = min(9, (10 * int(time_index)) // time)
                record: Dict[str, object] = {
                    "schema_version": CELL_SCHEMA_VERSION,
                    "sample_id": sample_ids[batch_index],
                    "prompt_sha256": prompt_hashes[batch_index],
                    "rollout_seed": int(rollout_seeds[batch_index]),
                    "q": int(q_index),
                    "t": int(time_index),
                    "temporal_decile": int(temporal_decile),
                    "js": float(js[batch_index, q_index, time_index].item()),
                    "forward_kl": float(token_kl[batch_index, q_index, time_index].item()),
                    "teacher_entropy": float(
                        teacher_entropy[batch_index, q_index, time_index].item()
                    ),
                    "student_entropy": float(
                        student_entropy[batch_index, q_index, time_index].item()
                    ),
                    "sampled_token_logp_teacher": float(
                        sampled_teacher_logp[
                            batch_index, q_index, time_index
                        ].item()
                    ),
                    "sampled_token_logp_student": float(
                        sampled_student_logp[
                            batch_index, q_index, time_index
                        ].item()
                    ),
                    "a_q": float(prior[q_index].item()),
                    "top50_js": bool(selected[batch_index, q_index, time_index].item()),
                }
                yield record


def audit_topk_records(
    student_logits: Tensor,
    teacher_logits: Tensor,
    sampled_codes: Tensor,
    valid_mask: Tensor,
    *,
    sample_ids: Sequence[str],
    prompts: Sequence[str],
    rollout_seeds: Sequence[int],
    top_k: int = 32,
) -> Iterator[Dict[str, object]]:
    """Yield optional top-k logits for a small, explicitly bounded audit set."""

    batch, codebooks, time, vocabulary = _validate_probe_inputs(
        student_logits,
        teacher_logits,
        sampled_codes,
        valid_mask,
        sample_ids,
        prompts,
        rollout_seeds,
    )
    if top_k <= 0 or top_k > vocabulary:
        raise ValueError("top_k must lie in [1, V]")
    safe_student = torch.where(
        valid_mask.unsqueeze(-1), student_logits.detach().float(), 0.0
    )
    safe_teacher = torch.where(
        valid_mask.unsqueeze(-1), teacher_logits.detach().float(), 0.0
    )
    student_values, student_ids = torch.topk(safe_student, top_k, dim=-1)
    teacher_values, teacher_ids = torch.topk(safe_teacher, top_k, dim=-1)
    # Copy each dense batch once.  Indexing CUDA tensors inside the Python
    # record loop would otherwise introduce one device synchronization per
    # field and valid cell (about 64k cells in the frozen audit).
    student_values = student_values.cpu()
    student_ids = student_ids.cpu()
    teacher_values = teacher_values.cpu()
    teacher_ids = teacher_ids.cpu()
    sampled_codes_cpu = sampled_codes.detach().cpu()
    valid_cpu = valid_mask.detach().cpu()
    prompt_hashes = [prompt_sha256(prompt) for prompt in prompts]
    for batch_index in range(batch):
        for q_index in range(codebooks):
            for time_index in torch.nonzero(
                valid_cpu[batch_index, q_index], as_tuple=False
            ).flatten().tolist():
                yield {
                    "schema_version": "ptc-opd-disagreement-topk-audit-v1",
                    "sample_id": sample_ids[batch_index],
                    "prompt_sha256": prompt_hashes[batch_index],
                    "rollout_seed": int(rollout_seeds[batch_index]),
                    "q": int(q_index),
                    "t": int(time_index),
                    "sampled_token_id": int(
                        sampled_codes_cpu[batch_index, q_index, time_index].item()
                    ),
                    "student_top_token_ids": student_ids[
                        batch_index, q_index, time_index
                    ].tolist(),
                    "student_top_logits": student_values[
                        batch_index, q_index, time_index
                    ].tolist(),
                    "teacher_top_token_ids": teacher_ids[
                        batch_index, q_index, time_index
                    ].tolist(),
                    "teacher_top_logits": teacher_values[
                        batch_index, q_index, time_index
                    ].tolist(),
                }


def _empty_vector() -> List[float]:
    return [0.0] * len(_VECTOR_FIELDS)


def _add_vector(target: List[float], source: Sequence[float]) -> None:
    for index, value in enumerate(source):
        target[index] += float(value)


def _record_vector(record: Mapping[str, object]) -> List[float]:
    js = float(record["js"])
    kl = float(record["forward_kl"])
    selected = bool(record["top50_js"])
    return [
        1.0,
        float(record["a_q"]),
        js,
        kl,
        float(record["teacher_entropy"]),
        float(record["student_entropy"]),
        float(record["sampled_token_logp_teacher"]),
        float(record["sampled_token_logp_student"]),
        js if selected else 0.0,
        kl if selected else 0.0,
    ]


def _group_sort_key(key: Tuple[object, ...]) -> Tuple[int, Tuple[object, ...]]:
    order = {
        "overall": 0,
        "codebook": 1,
        "temporal_decile": 2,
        "joint": 3,
        "within_q_js_decile": 4,
    }
    return order[str(key[0])], key[1:]


def _validate_cell_record(record: Mapping[str, object]) -> None:
    missing = [field for field in REQUIRED_CELL_FIELDS if field not in record]
    if missing:
        raise ValueError("cell record is missing fields: {}".format(missing))
    if record["schema_version"] != CELL_SCHEMA_VERSION:
        raise ValueError("unsupported cell schema version")
    if not isinstance(record["sample_id"], str) or not record["sample_id"]:
        raise ValueError("sample_id must be non-empty")
    prompt_hash = record["prompt_sha256"]
    if (
        not isinstance(prompt_hash, str)
        or len(prompt_hash) != 64
        or any(character not in "0123456789abcdef" for character in prompt_hash)
    ):
        raise ValueError("prompt_sha256 must be a lowercase SHA-256 hex digest")
    q_index = int(record["q"])
    time_index = int(record["t"])
    decile = int(record["temporal_decile"])
    if q_index < 0 or time_index < 0 or decile not in range(10):
        raise ValueError("q/t must be non-negative and temporal_decile in [0, 9]")
    for field in (
        "js",
        "forward_kl",
        "teacher_entropy",
        "student_entropy",
        "sampled_token_logp_teacher",
        "sampled_token_logp_student",
        "a_q",
    ):
        value = float(record[field])
        if not math.isfinite(value):
            raise FloatingPointError("{} must be finite".format(field))
    if float(record["js"]) < 0.0:
        raise ValueError("JS must be non-negative")
    if float(record["forward_kl"]) < -1.0e-5:
        raise ValueError("forward KL is materially negative")
    if float(record["a_q"]) <= 0.0:
        raise ValueError("a_q must be positive")
    if not isinstance(record["top50_js"], bool):
        raise TypeError("top50_js must be boolean")


@dataclass(frozen=True)
class PhenomenonSummary:
    """Point estimates, prompt-bootstrap intervals, and optional replicates."""

    rows: List[Dict[str, object]]
    tv: Dict[str, object]
    prompt_count: int
    sequence_count: int
    cell_count: int
    codebook_count: int
    rollout_seeds: Tuple[int, ...]
    bootstrap_seed: int
    bootstrap_replicates: int
    bootstrap_arrays: Optional[Dict[str, Tensor]]
    rollout_seeds_by_prompt: Dict[str, Tuple[int, ...]]


class PhenomenonAccumulator:
    """Streaming accumulator for contiguous sequence blocks of cell records."""

    def __init__(self) -> None:
        self._current_key: Optional[Tuple[str, int]] = None
        self._current_records: List[Mapping[str, object]] = []
        self._closed_sequences: set = set()
        self._prompt_hash_by_id: Dict[str, str] = {}
        self._rollouts_by_prompt: Dict[str, set] = defaultdict(set)
        self._prompt_groups: Dict[
            str, Dict[Tuple[object, ...], List[float]]
        ] = defaultdict(dict)
        self._prior_by_q: Dict[int, float] = {}
        self._cell_count = 0

    def consume(self, record: Mapping[str, object]) -> None:
        _validate_cell_record(record)
        key = (str(record["sample_id"]), int(record["rollout_seed"]))
        if self._current_key is None:
            if key in self._closed_sequences:
                raise ValueError("sequence blocks must be contiguous")
            self._current_key = key
        elif key != self._current_key:
            self._flush_sequence()
            if key in self._closed_sequences:
                raise ValueError("duplicate/non-contiguous sequence block {}".format(key))
            self._current_key = key
        self._current_records.append(record)

    def consume_many(self, records: Iterable[Mapping[str, object]]) -> None:
        for record in records:
            self.consume(record)

    def rollout_seeds_by_prompt(self) -> Dict[str, Tuple[int, ...]]:
        """Return observed prompt/rollout membership without exposing internals."""

        self._flush_sequence()
        return {
            sample_id: tuple(sorted(seeds))
            for sample_id, seeds in self._rollouts_by_prompt.items()
        }

    def _add_group(
        self,
        prompt_id: str,
        key: Tuple[object, ...],
        vector: Sequence[float],
    ) -> None:
        groups = self._prompt_groups[prompt_id]
        target = groups.setdefault(key, _empty_vector())
        _add_vector(target, vector)

    def _flush_sequence(self) -> None:
        if self._current_key is None:
            return
        if not self._current_records:
            raise RuntimeError("internal empty sequence block")
        sample_id, rollout_seed = self._current_key
        prompt_hashes = {str(record["prompt_sha256"]) for record in self._current_records}
        if len(prompt_hashes) != 1:
            raise ValueError("one sequence contains multiple prompt hashes")
        prompt_hash = next(iter(prompt_hashes))
        previous_hash = self._prompt_hash_by_id.setdefault(sample_id, prompt_hash)
        if previous_hash != prompt_hash:
            raise ValueError("prompt hash changed across rollouts for {}".format(sample_id))
        if rollout_seed in self._rollouts_by_prompt[sample_id]:
            raise ValueError("duplicate rollout seed for {}".format(sample_id))
        self._rollouts_by_prompt[sample_id].add(rollout_seed)

        seen_cells = set()
        records_by_q: Dict[int, List[Mapping[str, object]]] = defaultdict(list)
        for record in self._current_records:
            q_index = int(record["q"])
            time_index = int(record["t"])
            cell_key = (q_index, time_index)
            if cell_key in seen_cells:
                raise ValueError("duplicate cell {} in sequence {}".format(cell_key, self._current_key))
            seen_cells.add(cell_key)
            records_by_q[q_index].append(record)
            prior = float(record["a_q"])
            if q_index in self._prior_by_q and not math.isclose(
                self._prior_by_q[q_index], prior, rel_tol=1.0e-6, abs_tol=1.0e-8
            ):
                raise ValueError("a_q changed across records for q={}".format(q_index))
            self._prior_by_q[q_index] = prior

            vector = _record_vector(record)
            decile = int(record["temporal_decile"])
            self._add_group(sample_id, ("overall",), vector)
            self._add_group(sample_id, ("codebook", q_index), vector)
            self._add_group(sample_id, ("temporal_decile", decile), vector)
            self._add_group(sample_id, ("joint", q_index, decile), vector)

        for q_index, q_records in records_by_q.items():
            expected_selected = int(math.ceil(len(q_records) * 0.5))
            selected_count = sum(bool(record["top50_js"]) for record in q_records)
            if selected_count != expected_selected:
                raise ValueError(
                    "top50 count mismatch for sequence {}, q={}: {} versus {}".format(
                        self._current_key, q_index, selected_count, expected_selected
                    )
                )
            # Ascending stable ranks make bin 9 the highest-JS decile.  Time is
            # an explicit secondary key, so ties are deterministic.
            ordered = sorted(q_records, key=lambda record: (float(record["js"]), int(record["t"])))
            count = len(ordered)
            for rank, record in enumerate(ordered):
                js_decile = min(9, (10 * rank) // count)
                self._add_group(
                    sample_id,
                    ("within_q_js_decile", q_index, js_decile),
                    _record_vector(record),
                )

        self._cell_count += len(self._current_records)
        self._closed_sequences.add(self._current_key)
        self._current_key = None
        self._current_records = []

    def finalize(
        self,
        *,
        bootstrap_replicates: int = PRIMARY_BOOTSTRAP_REPLICATES,
        bootstrap_seed: int = PRIMARY_BOOTSTRAP_SEED,
        retain_bootstrap_arrays: bool = False,
    ) -> PhenomenonSummary:
        self._flush_sequence()
        if not self._closed_sequences:
            raise ValueError("no cell records were supplied")
        if bootstrap_replicates < 0:
            raise ValueError("bootstrap_replicates must be non-negative")
        if not self._prior_by_q:
            raise ValueError("no codebooks were observed")
        codebooks = sorted(self._prior_by_q)
        if codebooks != list(range(len(codebooks))):
            raise ValueError("observed codebooks must be contiguous from zero")
        prior_sum = sum(self._prior_by_q.values())
        if not math.isclose(prior_sum, 1.0, rel_tol=1.0e-5, abs_tol=1.0e-6):
            raise ValueError("a_q must sum to one; observed {}".format(prior_sum))

        prompt_ids = sorted(self._prompt_groups)
        group_keys = sorted(
            {key for groups in self._prompt_groups.values() for key in groups},
            key=_group_sort_key,
        )
        rollout_sets = [
            tuple(sorted(self._rollouts_by_prompt[sample_id])) for sample_id in prompt_ids
        ]
        if any(seeds != rollout_sets[0] for seeds in rollout_sets[1:]):
            raise ValueError("every prompt must contain the same rollout-seed set")
        prompt_index = {sample_id: index for index, sample_id in enumerate(prompt_ids)}
        group_index = {key: index for index, key in enumerate(group_keys)}
        contributions = torch.zeros(
            (len(prompt_ids), len(group_keys), len(_VECTOR_FIELDS)), dtype=torch.float64
        )
        for sample_id, groups in self._prompt_groups.items():
            for key, vector in groups.items():
                contributions[prompt_index[sample_id], group_index[key]] = torch.tensor(
                    vector, dtype=torch.float64
                )

        # A prompt is the outer statistical unit.  Average its nested rollout
        # contributions first; with the frozen two equal-length rollouts this
        # also equals the global valid-cell estimate.
        rollout_counts = torch.tensor(
            [len(self._rollouts_by_prompt[sample_id]) for sample_id in prompt_ids],
            dtype=torch.float64,
        )
        prompt_contributions = contributions / rollout_counts.view(-1, 1, 1)
        point = prompt_contributions.sum(dim=0)
        overall_index = group_index[("overall",)]
        point_metrics = _metrics_from_totals(point, overall_index)

        bootstrap: Dict[str, Tensor] = {
            metric: torch.empty(
                (bootstrap_replicates, len(group_keys)), dtype=torch.float64
            )
            for metric in _RATIO_METRICS
        }
        tv_bootstrap = torch.empty((bootstrap_replicates,), dtype=torch.float64)
        if bootstrap_replicates:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(bootstrap_seed))
            flattened = prompt_contributions.reshape(len(prompt_ids), -1)
            codebook_indices = torch.tensor(
                [group_index[("codebook", q_index)] for q_index in codebooks],
                dtype=torch.long,
            )
            chunk_size = 128
            for start in range(0, bootstrap_replicates, chunk_size):
                stop = min(bootstrap_replicates, start + chunk_size)
                chunk = stop - start
                draws = torch.randint(
                    len(prompt_ids),
                    (chunk, len(prompt_ids)),
                    generator=generator,
                )
                draw_counts = torch.zeros(
                    (chunk, len(prompt_ids)), dtype=torch.float64
                )
                draw_counts.scatter_add_(
                    1, draws, torch.ones_like(draws, dtype=torch.float64)
                )
                totals = (draw_counts @ flattened).reshape(
                    chunk, len(group_keys), len(_VECTOR_FIELDS)
                )
                metrics = _metrics_from_totals(totals, overall_index)
                for metric in _RATIO_METRICS:
                    bootstrap[metric][start:stop] = metrics[metric]
                codebook_u = metrics["nominal_u"].index_select(1, codebook_indices)
                codebook_w = metrics["weighted_w_a"].index_select(1, codebook_indices)
                tv_bootstrap[start:stop] = 0.5 * (codebook_u - codebook_w).abs().sum(dim=1)

        rows: List[Dict[str, object]] = []
        for index, key in enumerate(group_keys):
            row: Dict[str, object] = {
                "group_type": key[0],
                "q": None,
                "temporal_decile": None,
                "within_q_js_decile": None,
                "valid_cells": int(round(float(point[index, 0].item()))),
                "a_q": None,
            }
            if key[0] == "codebook":
                row["q"] = int(key[1])
                row["a_q"] = self._prior_by_q[int(key[1])]
            elif key[0] == "temporal_decile":
                row["temporal_decile"] = int(key[1])
            elif key[0] == "joint":
                row["q"] = int(key[1])
                row["temporal_decile"] = int(key[2])
                row["a_q"] = self._prior_by_q[int(key[1])]
            elif key[0] == "within_q_js_decile":
                row["q"] = int(key[1])
                row["within_q_js_decile"] = int(key[2])
                row["a_q"] = self._prior_by_q[int(key[1])]

            for metric in _RATIO_METRICS:
                value = float(point_metrics[metric][index].item())
                row[metric] = value if math.isfinite(value) else None
                if bootstrap_replicates:
                    low, high = _finite_interval(bootstrap[metric][:, index])
                    row[metric + "_ci95_low"] = low
                    row[metric + "_ci95_high"] = high
                else:
                    row[metric + "_ci95_low"] = None
                    row[metric + "_ci95_high"] = None
            rows.append(row)

        codebook_u = torch.stack(
            [point_metrics["nominal_u"][group_index[("codebook", q)]] for q in codebooks]
        )
        codebook_w = torch.stack(
            [point_metrics["weighted_w_a"][group_index[("codebook", q)]] for q in codebooks]
        )
        tv_value = float((0.5 * (codebook_u - codebook_w).abs().sum()).item())
        if bootstrap_replicates:
            tv_low, tv_high = _finite_interval(tv_bootstrap)
        else:
            tv_low, tv_high = None, None
        tv = {
            "value": tv_value,
            "ci95_low": tv_low,
            "ci95_high": tv_high,
            "definition": "0.5 * sum_q abs(U_q - W_a(q))",
        }
        rollout_seeds = tuple(
            sorted({seed for seeds in self._rollouts_by_prompt.values() for seed in seeds})
        )
        arrays: Optional[Dict[str, Tensor]] = None
        if retain_bootstrap_arrays:
            arrays = dict(bootstrap)
            arrays["tv_u_vs_w"] = tv_bootstrap
        return PhenomenonSummary(
            rows=rows,
            tv=tv,
            prompt_count=len(prompt_ids),
            sequence_count=len(self._closed_sequences),
            cell_count=self._cell_count,
            codebook_count=len(codebooks),
            rollout_seeds=rollout_seeds,
            bootstrap_seed=int(bootstrap_seed),
            bootstrap_replicates=int(bootstrap_replicates),
            bootstrap_arrays=arrays,
            rollout_seeds_by_prompt={
                sample_id: tuple(sorted(seeds))
                for sample_id, seeds in self._rollouts_by_prompt.items()
            },
        )


def _safe_ratio(numerator: Tensor, denominator: Tensor) -> Tensor:
    result = torch.full_like(numerator, float("nan"), dtype=torch.float64)
    return torch.where(denominator != 0, numerator / denominator, result)


def _metrics_from_totals(totals: Tensor, overall_index: int) -> Dict[str, Tensor]:
    """Convert additive totals ``[..., G, F]`` into all reported ratios."""

    count = totals[..., 0]
    weighted_count = totals[..., 1]
    js_sum = totals[..., 2]
    kl_sum = totals[..., 3]
    overall_count = count[..., overall_index].unsqueeze(-1)
    overall_weighted = weighted_count[..., overall_index].unsqueeze(-1)
    overall_kl = kl_sum[..., overall_index].unsqueeze(-1)
    return {
        "mean_js": _safe_ratio(js_sum, count),
        "mean_forward_kl": _safe_ratio(kl_sum, count),
        "mean_teacher_entropy": _safe_ratio(totals[..., 4], count),
        "mean_student_entropy": _safe_ratio(totals[..., 5], count),
        "mean_sampled_token_logp_teacher": _safe_ratio(totals[..., 6], count),
        "mean_sampled_token_logp_student": _safe_ratio(totals[..., 7], count),
        "nominal_u": _safe_ratio(count, overall_count),
        "weighted_w_a": _safe_ratio(weighted_count, overall_weighted),
        "kl_mass_k": _safe_ratio(kl_sum, overall_kl),
        "c50_js": _safe_ratio(totals[..., 8], js_sum),
        "c50_kl": _safe_ratio(totals[..., 9], kl_sum),
    }


def _finite_interval(values: Tensor) -> Tuple[Optional[float], Optional[float]]:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return None, None
    quantiles = torch.quantile(
        finite, torch.tensor([0.025, 0.975], dtype=finite.dtype)
    )
    return float(quantiles[0].item()), float(quantiles[1].item())


def summarize_cell_records(
    records: Iterable[Mapping[str, object]],
    *,
    bootstrap_replicates: int = PRIMARY_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = PRIMARY_BOOTSTRAP_SEED,
    retain_bootstrap_arrays: bool = False,
) -> PhenomenonSummary:
    """Convenience wrapper around :class:`PhenomenonAccumulator`."""

    accumulator = PhenomenonAccumulator()
    accumulator.consume_many(records)
    return accumulator.finalize(
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        retain_bootstrap_arrays=retain_bootstrap_arrays,
    )


__all__ = [
    "CELL_SCHEMA_VERSION",
    "PRIMARY_BOOTSTRAP_REPLICATES",
    "PRIMARY_BOOTSTRAP_SEED",
    "PRIMARY_ROLLOUT_SEEDS",
    "PhenomenonAccumulator",
    "PhenomenonSummary",
    "SUMMARY_SCHEMA_VERSION",
    "audit_topk_records",
    "cell_records_from_logits",
    "prompt_sha256",
    "summarize_cell_records",
]
