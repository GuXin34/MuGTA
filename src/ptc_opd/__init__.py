"""PTC-OPD package with dependency-lazy public exports.

Control-plane consumers such as artifact verifiers intentionally do not need
PyTorch.  Public tensor symbols retain their original names but their modules
are imported only when the symbol is first requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple


_EXPORTS: Dict[str, Tuple[str, str]] = {
    "AudioCraftTrajectoryScores": ("audiocraft_adapter", "AudioCraftTrajectoryScores"),
    "batch_condition_tensors": ("audiocraft_adapter", "batch_condition_tensors"),
    "score_audiocraft_trajectory": ("audiocraft_adapter", "score_audiocraft_trajectory"),
    "OPDLossOutput": ("losses", "OPDLossOutput"),
    "ptc_opd_loss": ("losses", "ptc_opd_loss"),
    "GlobalRatioOutput": ("distributed", "GlobalRatioOutput"),
    "globally_normalized_loss": ("distributed", "globally_normalized_loss"),
    "DEFAULT_FFT_SIZES": ("perceptual_prior", "DEFAULT_FFT_SIZES"),
    "ProgressivePriorOutput": ("perceptual_prior", "ProgressivePriorOutput"),
    "mrstft_distance": ("perceptual_prior", "mrstft_distance"),
    "normalize_marginals": ("perceptual_prior", "normalize_marginals"),
    "progressive_prior": ("perceptual_prior", "progressive_prior"),
    "CELL_SCHEMA_VERSION": ("phenomena", "CELL_SCHEMA_VERSION"),
    "PRIMARY_BOOTSTRAP_REPLICATES": ("phenomena", "PRIMARY_BOOTSTRAP_REPLICATES"),
    "PRIMARY_BOOTSTRAP_SEED": ("phenomena", "PRIMARY_BOOTSTRAP_SEED"),
    "PRIMARY_ROLLOUT_SEEDS": ("phenomena", "PRIMARY_ROLLOUT_SEEDS"),
    "PhenomenonAccumulator": ("phenomena", "PhenomenonAccumulator"),
    "PhenomenonSummary": ("phenomena", "PhenomenonSummary"),
    "audit_topk_records": ("phenomena", "audit_topk_records"),
    "cell_records_from_logits": ("phenomena", "cell_records_from_logits"),
    "summarize_cell_records": ("phenomena", "summarize_cell_records"),
    "BatchAssignment": ("sampling", "BatchAssignment"),
    "DeterministicDistributedBatchSampler": (
        "sampling",
        "DeterministicDistributedBatchSampler",
    ),
    "keyed_random_scores": ("sampling", "keyed_random_scores"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name)) from exc
    value = getattr(import_module(".{}".format(module_name), __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
