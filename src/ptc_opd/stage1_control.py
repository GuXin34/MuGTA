"""Fail-closed Stage-1 manifest, decision, stability, and DAG contracts.

This module is intentionally dependency-free.  It does not generate audio,
run an evaluator, or launch training.  Missing scientific producers remain
explicit capabilities in the controller instead of being inferred from the
older CFG-only evaluation wrappers.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .stage1_artifact import (
    Stage1ArtifactError,
    artifact_member,
    canonical_json_bytes,
    canonical_json_sha256,
    load_json_strict,
    publish_closed_files_artifact,
    publish_closed_json_artifact,
    require_finite_number,
    require_sha256,
    sha256_file,
    verify_simple_seal,
)


PILOT_MANIFEST_SCHEMA = "ptc-opd-pilot-eval-manifest-v1"
PILOT_MANIFEST_SEAL_SCHEMA = "ptc-opd-pilot-eval-manifest-seal-v1"
PILOT_RECORD_SCHEMA = "ptc-opd-pilot-eval-record-v1"
PILOT_SOURCE_BASENAME = "dev.full.jsonl"
PILOT_OUTPUT_BASENAME = "pilot_eval.dev.jsonl"
PILOT_REPORT_BASENAME = "pilot_eval_manifest.json"
PILOT_SELECTION_NAMESPACE = "ptc-opd-small-pilot-eval-v1"
PILOT_SELECTION_SEED = 2701
PILOT_PROMPT_COUNT = 128
PILOT_SOURCE_COUNT = 300

LR_GRID: Tuple[float, ...] = (1.0e-6, 3.0e-6, 1.0e-5)
LR_CLAP_GUARDRAIL = -0.10
LR_SUMMARY_SCHEMA = "ptc-opd-lr-evaluation-summary-v1"
LR_SUMMARY_SEAL_SCHEMA = "ptc-opd-lr-evaluation-summary-seal-v1"
LR_SUMMARY_BASENAME = "lr_evaluation_summary.json"
LR_DECISION_SCHEMA = "ptc-opd-lr-decision-v1"
LR_DECISION_SEAL_SCHEMA = "ptc-opd-lr-decision-seal-v1"
LR_DECISION_BASENAME = "lr_decision.json"
LR_Q_WEIGHTS = {"muq_mi": 0.50, "audiobox_ce": 0.25, "audiobox_pq": 0.25}
LR_METRICS: Tuple[str, ...] = (
    "muq_mi",
    "audiobox_ce",
    "audiobox_pq",
    "music_clap",
)
LR_ARTIFACT_FIELDS: Tuple[str, ...] = (
    "training_run_manifest_sha256",
    "training_run_seal_sha256",
    "training_done_sha256",
    "final_checkpoint_sha256",
    "generation_artifact_seal_sha256",
    "quality_artifact_seal_sha256",
    "clap_artifact_seal_sha256",
)
LR_BASE_ARTIFACT_FIELDS: Tuple[str, ...] = (
    "generation_artifact_seal_sha256",
    "quality_artifact_seal_sha256",
    "clap_artifact_seal_sha256",
)
TRAINING_LINEAGE_SCHEMA = "ptc-opd-stage1-training-lineage-anchor-v1"

PTC500_REPORT_SCHEMA = "ptc-opd-ptc500-stability-decision-v1"
PTC500_SEAL_SCHEMA = "ptc-opd-ptc500-stability-decision-seal-v1"
PTC500_REPORT_BASENAME = "ptc500_stability_decision.json"
PTC500_EXPECTED_STEPS = 500
PTC500_MEMORY_MARGIN_MIN = 0.05
B1_FULL_CLOSURE_SCHEMA = "ptc-opd-b1-full-closure-v1"
B1_FULL_CLOSURE_SEAL_SCHEMA = "ptc-opd-b1-full-closure-seal-v1"
B1_FULL_CLOSURE_BASENAME = "b1_full_closure.json"

SMALL_PILOT_METHODS: Tuple[str, ...] = (
    "uniform100",
    "codebook100",
    "random50",
    "prefix50",
    "disagreement50",
    "ptc50",
)
SMALL_PILOT_SUMMARY_SCHEMA = "ptc-opd-small-pilot-summary-v1"
SMALL_PILOT_SUMMARY_SEAL_SCHEMA = "ptc-opd-small-pilot-summary-seal-v1"
SMALL_PILOT_SUMMARY_BASENAME = "small_pilot_summary.json"
SMALL_PILOT_DECISION_SCHEMA = "ptc-opd-small-pilot-decision-v1"
SMALL_PILOT_DECISION_SEAL_SCHEMA = "ptc-opd-small-pilot-decision-seal-v1"
SMALL_PILOT_DECISION_BASENAME = "small_pilot_decision.json"
SMALL_PILOT_ARTIFACT_FIELDS: Tuple[str, ...] = LR_ARTIFACT_FIELDS + (
    "diversity_fad_artifact_seal_sha256",
)

CONTROLLER_CONTRACT_SCHEMA = "ptc-opd-stage1-autonomy-contract-v1"
# Kept as a public compatibility name for callers that display the schema.
# Authorization itself is implemented by the sealed, verifier-backed ledger
# consumer in ``stage1_controller_ledger``; this value must never be used to
# accept a hand-written mapping.
CONTROLLER_LEDGER_SCHEMA = "ptc-opd-stage1-controller-authority-ledger-v1"
CONTROLLER_PLAN_SCHEMA = "ptc-opd-stage1-controller-plan-v1"
CONTROLLER_PREFLIGHT_SCHEMA = "ptc-opd-stage1-controller-preflight-v1"
CONTROLLER_ACTION_SCHEMA = "ptc-opd-stage1-controller-action-v1"
PRIMARY_PILOT_METHODS: Tuple[str, ...] = (
    "uniform100",
    "codebook100",
    "random50",
    "disagreement50",
    "ptc50",
)
REGISTERED_STAGE1_MODES: Tuple[str, ...] = (
    "uniform100",
    "codebook100",
    "random50",
    "prefix50",
    "disagreement50",
    "ptc50",
)


def _exact_fields(value: Mapping[str, Any], fields: Iterable[str], label: str) -> None:
    expected = set(fields)
    observed = set(value)
    if observed != expected:
        raise Stage1ArtifactError(
            "{} fields differ; missing={}, unexpected={}".format(
                label, sorted(expected - observed), sorted(observed - expected)
            )
        )


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise Stage1ArtifactError("{} must be boolean".format(label))
    return bool(value)


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise Stage1ArtifactError(
            "{} must be an integer >= {}".format(label, minimum)
        )
    return int(value)


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Stage1ArtifactError("{} must be a non-empty string".format(label))
    return value


def _validate_training_lineage_anchor(value: Any) -> Dict[str, Any]:
    """Normalize the immutable base lineage shared by every Stage-1 run.

    The LR, PTC500, and small-pilot stages are separate jobs.  A common model
    name or learning rate does not prove that they used the same base model.
    This anchor makes the complete checkpoint (LM state and codec), loaded LM
    state, AudioCraft tree, selected CFG decision, and T5 identity one
    canonical comparison surface.
    """

    if not isinstance(value, Mapping):
        raise Stage1ArtifactError("training lineage anchor must be an object")
    _exact_fields(
        value,
        (
            "schema_version",
            "model_id",
            "base_checkpoint",
            "base_lm_state_sha256",
            "audiocraft_source_sha256",
            "cfg_decision",
            "loaded_t5_identity_sha256",
        ),
        "training lineage anchor",
    )
    if value.get("schema_version") != TRAINING_LINEAGE_SCHEMA:
        raise Stage1ArtifactError("training lineage anchor schema mismatch")
    # Ruling #9 §B (2026-09-01 CST, memory c0n4y96p): allow MusicGen-medium in
    # addition to MusicGen-small for cross-scale verification.  Codec identity
    # preflight (Node 22) enforces same EnCodec revision.  Small pilot evidence
    # is byte-preserved (Ruling #8 §1); Medium is a separate lineage.
    _observed_model_id = value.get("model_id")
    if _observed_model_id not in ("facebook/musicgen-small", "facebook/musicgen-medium"):
        raise Stage1ArtifactError(
            "training lineage model must be MusicGen-small or MusicGen-medium"
        )
    checkpoint = value.get("base_checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise Stage1ArtifactError("training lineage base checkpoint is absent")
    _exact_fields(
        checkpoint,
        (
            "checkpoint_sha256",
            "state_dict_sha256",
            "compression_state_dict_sha256",
        ),
        "training lineage base checkpoint",
    )
    normalized_checkpoint = {
        name: require_sha256(checkpoint.get(name), "training lineage {}".format(name))
        for name in (
            "checkpoint_sha256",
            "state_dict_sha256",
            "compression_state_dict_sha256",
        )
    }
    cfg = value.get("cfg_decision")
    if not isinstance(cfg, Mapping):
        raise Stage1ArtifactError("training lineage CFG decision is absent")
    _exact_fields(
        cfg,
        (
            "decision_file_sha256",
            "decision_payload_sha256",
            "scientific_config_sha256",
            "selected_cfg_scale",
        ),
        "training lineage CFG decision",
    )
    normalized_cfg = {
        name: require_sha256(cfg.get(name), "training lineage CFG {}".format(name))
        for name in (
            "decision_file_sha256",
            "decision_payload_sha256",
            "scientific_config_sha256",
        )
    }
    if cfg.get("selected_cfg_scale") != 5.0:
        raise Stage1ArtifactError("training lineage CFG scale must be 5.0")
    normalized_cfg["selected_cfg_scale"] = 5.0
    normalized = {
        "schema_version": TRAINING_LINEAGE_SCHEMA,
        # Ruling #9 §B: preserve the observed model_id (must be one of the two
        # allowed above); do NOT hardcode "facebook/musicgen-small" — that would
        # silently corrupt Medium lineage anchors.
        "model_id": _observed_model_id,
        "base_checkpoint": normalized_checkpoint,
        "base_lm_state_sha256": require_sha256(
            value.get("base_lm_state_sha256"), "training lineage base LM state"
        ),
        "audiocraft_source_sha256": require_sha256(
            value.get("audiocraft_source_sha256"),
            "training lineage AudioCraft source",
        ),
        "cfg_decision": normalized_cfg,
        "loaded_t5_identity_sha256": require_sha256(
            value.get("loaded_t5_identity_sha256"),
            "training lineage loaded T5",
        ),
    }
    canonical_json_sha256(normalized)
    return normalized


def _training_lineage_anchor_from_generation_config(
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    if not isinstance(config, Mapping):
        raise Stage1ArtifactError("generation config is absent for lineage")
    decision = config.get("cfg_decision")
    runtime = config.get("runtime_identity")
    if not isinstance(decision, Mapping) or not isinstance(runtime, Mapping):
        raise Stage1ArtifactError("generation CFG/runtime lineage is absent")
    loaded_t5 = runtime.get("loaded_t5_identity")
    if not isinstance(loaded_t5, Mapping):
        raise Stage1ArtifactError("generation loaded T5 lineage is absent")
    t5_identity = loaded_t5.get("identity_sha256")
    if t5_identity != decision.get("loaded_t5_identity_sha256"):
        raise Stage1ArtifactError("generation CFG/runtime T5 lineage differs")
    return _validate_training_lineage_anchor(
        {
            "schema_version": TRAINING_LINEAGE_SCHEMA,
            "model_id": config.get("model_id"),
            "base_checkpoint": config.get("base_checkpoint"),
            "base_lm_state_sha256": config.get("base_lm_state_sha256"),
            "audiocraft_source_sha256": config.get("audiocraft_source_sha256"),
            "cfg_decision": {
                name: decision.get(name)
                for name in (
                    "decision_file_sha256",
                    "decision_payload_sha256",
                    "scientific_config_sha256",
                    "selected_cfg_scale",
                )
            },
            "loaded_t5_identity_sha256": t5_identity,
        }
    )


def _training_lineage_anchor_from_run_manifest(
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Rebuild the same lineage from a live run manifest, fail closed."""

    if not isinstance(manifest, Mapping):
        raise Stage1ArtifactError("training run manifest is absent for lineage")
    decision = manifest.get("cfg_scale_decision")
    generation_identity = (
        decision.get("generation_identity")
        if isinstance(decision, Mapping)
        else None
    )
    loaded_t5 = manifest.get("loaded_t5_identity")
    source = manifest.get("audiocraft_source_identity")
    config = manifest.get("config")
    if (
        not isinstance(decision, Mapping)
        or not isinstance(generation_identity, Mapping)
        or not isinstance(loaded_t5, Mapping)
        or not isinstance(source, Mapping)
        or not isinstance(config, Mapping)
    ):
        raise Stage1ArtifactError("training run lineage fields are absent")
    checkpoint = {
        "checkpoint_sha256": manifest.get("student_checkpoint_sha256"),
        "state_dict_sha256": manifest.get("student_state_dict_sha256"),
        "compression_state_dict_sha256": generation_identity.get(
            "compression_state_dict_sha256"
        ),
    }
    t5_identity = loaded_t5.get("identity_sha256")
    # Ruling #9 §B: read model_id from manifest.config (train_stage1 writes it);
    # fall back to generation_identity.model_id; fall back to legacy small.
    _run_model_id = (
        config.get("model_id")
        or generation_identity.get("model_id")
        or "facebook/musicgen-small"
    )
    anchor = _validate_training_lineage_anchor(
        {
            "schema_version": TRAINING_LINEAGE_SCHEMA,
            "model_id": _run_model_id,
            "base_checkpoint": checkpoint,
            "base_lm_state_sha256": manifest.get("student_state_sha256_initial"),
            "audiocraft_source_sha256": source.get("tree_sha256"),
            "cfg_decision": {
                name: decision.get(name)
                for name in (
                    "decision_file_sha256",
                    "decision_payload_sha256",
                    "scientific_config_sha256",
                    "selected_cfg_scale",
                )
            },
            "loaded_t5_identity_sha256": t5_identity,
        }
    )
    expected_equalities = {
        "teacher_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "teacher_state_dict_sha256": checkpoint["state_dict_sha256"],
        "teacher_state_sha256_initial": anchor["base_lm_state_sha256"],
    }
    for field, expected in expected_equalities.items():
        if manifest.get(field) != expected:
            raise Stage1ArtifactError(
                "training run lineage differs at {}".format(field)
            )
    generation_matches = {
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "state_dict_sha256": checkpoint["state_dict_sha256"],
        "compression_state_dict_sha256": checkpoint[
            "compression_state_dict_sha256"
        ],
        "audiocraft_source_sha256": anchor["audiocraft_source_sha256"],
        "loaded_t5_identity_sha256": t5_identity,
    }
    for field, expected in generation_matches.items():
        if generation_identity.get(field) != expected:
            raise Stage1ArtifactError(
                "training run CFG generation lineage differs at {}".format(field)
            )
    if manifest.get("cfg_generation_binding") != {
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "state_dict_sha256": checkpoint["state_dict_sha256"],
        "audiocraft_source_sha256": anchor["audiocraft_source_sha256"],
    }:
        raise Stage1ArtifactError("training run CFG generation binding differs")
    config_matches = {
        "cfg_scale_decision_file_sha256": anchor["cfg_decision"][
            "decision_file_sha256"
        ],
        "cfg_scale_decision_payload_sha256": anchor["cfg_decision"][
            "decision_payload_sha256"
        ],
        "cfg_scale_scientific_config_sha256": anchor["cfg_decision"][
            "scientific_config_sha256"
        ],
        "cfg_generation_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "cfg_generation_state_dict_sha256": checkpoint["state_dict_sha256"],
        "cfg_generation_audiocraft_source_sha256": anchor[
            "audiocraft_source_sha256"
        ],
        "cfg_generation_loaded_t5_identity_sha256": t5_identity,
        "teacher_cfg_scale": 5.0,
    }
    for field, expected in config_matches.items():
        if config.get(field) != expected:
            raise Stage1ArtifactError(
                "training run config lineage differs at {}".format(field)
            )
    return anchor


def _require_training_lineage_match(
    expected: Mapping[str, Any], observed: Mapping[str, Any], label: str
) -> Dict[str, Any]:
    normalized_expected = _validate_training_lineage_anchor(expected)
    normalized_observed = _validate_training_lineage_anchor(observed)
    if normalized_observed != normalized_expected:
        raise Stage1ArtifactError("{} training lineage differs".format(label))
    return normalized_expected


def _json_object_pairs(label: str):
    def unique(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Stage1ArtifactError("{} contains duplicate key {!r}".format(label, key))
            result[key] = value
        return result

    return unique


def _reject_constant(label: str):
    def reject(value: str) -> None:
        raise Stage1ArtifactError(
            "{} contains forbidden non-finite {}".format(label, value)
        )

    return reject


def read_jsonl_strict(path: Path) -> List[Dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise Stage1ArtifactError("JSONL is missing or not regular: {}".format(path))
    records: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.endswith("\n") or not line.strip():
                    raise Stage1ArtifactError(
                        "{} line {} is blank or partial".format(path, line_number)
                    )
                value = json.loads(
                    line,
                    object_pairs_hook=_json_object_pairs(
                        "{} line {}".format(path, line_number)
                    ),
                    parse_constant=_reject_constant(
                        "{} line {}".format(path, line_number)
                    ),
                )
                if not isinstance(value, dict):
                    raise Stage1ArtifactError(
                        "{} line {} must be an object".format(path, line_number)
                    )
                records.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Stage1ArtifactError("invalid JSONL: {}".format(path)) from exc
    return records


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(record)) for record in records)


def _bytes_identity(payload: bytes) -> Dict[str, Any]:
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def _prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _pilot_selection_key(sample_id: str) -> str:
    payload = "{}|{}|{}".format(
        PILOT_SELECTION_NAMESPACE, PILOT_SELECTION_SEED, sample_id
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_pilot_eval_manifest_payloads(
    source_dev_manifest: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], bytes]:
    """Derive the frozen 128-prompt pilot subset before any model scores exist."""

    source_dev_manifest = source_dev_manifest.expanduser()
    if source_dev_manifest.name != PILOT_SOURCE_BASENAME:
        raise Stage1ArtifactError(
            "pilot source must be named {}".format(PILOT_SOURCE_BASENAME)
        )
    source_records = read_jsonl_strict(source_dev_manifest)
    if len(source_records) != PILOT_SOURCE_COUNT:
        raise Stage1ArtifactError(
            "development manifest must contain exactly 300 prompts"
        )
    by_id: Dict[str, Dict[str, Any]] = {}
    ranked: List[Tuple[str, str]] = []
    for index, source in enumerate(source_records, start=1):
        sample_id = _require_text(source.get("sample_id"), "source sample_id")
        prompt = _require_text(source.get("prompt"), "source prompt")
        if sample_id in by_id:
            raise Stage1ArtifactError("duplicate development sample_id {!r}".format(sample_id))
        # Touch the prompt here so malformed source rows fail before selection.
        _prompt_sha256(prompt)
        by_id[sample_id] = dict(source)
        ranked.append((_pilot_selection_key(sample_id), sample_id))
    selected_ids = [sample_id for _, sample_id in sorted(ranked)[:PILOT_PROMPT_COUNT]]
    output_records: List[Dict[str, Any]] = []
    for sample_id in sorted(selected_ids):
        source = by_id[sample_id]
        prompt = str(source["prompt"])
        output_records.append(
            {
                "schema_version": PILOT_RECORD_SCHEMA,
                "sample_id": sample_id,
                "prompt": prompt,
                "prompt_sha256": _prompt_sha256(prompt),
                "source_record": source,
                "source_record_sha256": canonical_json_sha256(source),
            }
        )
    records_payload = _jsonl_bytes(output_records)
    report = {
        "schema_version": PILOT_MANIFEST_SCHEMA,
        "status": "complete",
        "source": {
            "basename": source_dev_manifest.name,
            "record_count": len(source_records),
            "sha256": sha256_file(source_dev_manifest),
            "size_bytes": source_dev_manifest.stat().st_size,
        },
        "selection": {
            "algorithm": "lowest_sha256_then_output_sorted_by_sample_id",
            "count": PILOT_PROMPT_COUNT,
            "namespace": PILOT_SELECTION_NAMESPACE,
            "seed": PILOT_SELECTION_SEED,
        },
        "selected_count": len(output_records),
        "selected_sample_ids_sha256": canonical_json_sha256(sorted(selected_ids)),
        "records": _bytes_identity(records_payload),
    }
    return report, output_records, records_payload


def publish_pilot_eval_manifest(source_dev_manifest: Path, output_dir: Path) -> Path:
    report, _, records_payload = build_pilot_eval_manifest_payloads(source_dev_manifest)
    return publish_closed_files_artifact(
        output_dir,
        payloads={
            PILOT_OUTPUT_BASENAME: records_payload,
            PILOT_REPORT_BASENAME: canonical_json_bytes(report),
        },
        seal_schema=PILOT_MANIFEST_SEAL_SCHEMA,
        seal_status="complete",
    )


def verify_pilot_eval_manifest(
    artifact_dir: Path, source_dev_manifest: Path
) -> Dict[str, Any]:
    verify_simple_seal(
        artifact_dir,
        seal_name="artifact_seal.json",
        schema_version=PILOT_MANIFEST_SEAL_SCHEMA,
        status="complete",
        payload_names=(PILOT_OUTPUT_BASENAME, PILOT_REPORT_BASENAME),
    )
    expected_report, expected_records, expected_bytes = build_pilot_eval_manifest_payloads(
        source_dev_manifest
    )
    observed_report = load_json_strict(artifact_dir / PILOT_REPORT_BASENAME)
    if observed_report != expected_report:
        raise Stage1ArtifactError("pilot manifest report differs from deterministic rebuild")
    observed_bytes = (artifact_dir / PILOT_OUTPUT_BASENAME).read_bytes()
    if observed_bytes != expected_bytes:
        raise Stage1ArtifactError("pilot manifest JSONL differs from deterministic rebuild")
    observed_records = read_jsonl_strict(artifact_dir / PILOT_OUTPUT_BASENAME)
    if observed_records != expected_records:
        raise Stage1ArtifactError("pilot manifest records differ after parsing")
    return {
        "schema_version": "ptc-opd-pilot-eval-manifest-verification-v1",
        "status": "verified",
        "record_count": len(observed_records),
        "manifest_sha256": sha256_file(artifact_dir / PILOT_OUTPUT_BASENAME),
        "report_sha256": sha256_file(artifact_dir / PILOT_REPORT_BASENAME),
        "artifact_seal_sha256": sha256_file(artifact_dir / "artifact_seal.json"),
    }


def _lr_scientific_config(
    pilot_manifest_seal_sha256: str,
    training_lineage_anchor: Mapping[str, Any],
) -> Dict[str, Any]:
    lineage = _validate_training_lineage_anchor(training_lineage_anchor)
    return {
        "schema_version": "ptc-opd-lr-evaluation-scientific-config-v1",
        "model": "facebook/musicgen-small",
        "method": "uniform100",
        "train_seed": 2027,
        "optimizer_steps": 500,
        "decision_checkpoint": 500,
        "candidate_grid": list(LR_GRID),
        "prompt_count": PILOT_PROMPT_COUNT,
        "generation_seeds": [31001, 31002],
        "pilot_eval_manifest_artifact_seal_sha256": require_sha256(
            pilot_manifest_seal_sha256, "LR pilot manifest seal"
        ),
        "training_lineage_anchor_sha256": canonical_json_sha256(lineage),
        "aggregation": {
            "unit": "prompt",
            "seed_reduction": "arithmetic_mean_within_prompt",
            "base_standardization": "n_minus_1_sample_sd_of_base_prompt_means",
            "paired_delta": "mean_candidate_minus_base_prompt_mean_divided_by_base_sd",
            "q_dev": "0.5*muq_mi+0.25*audiobox_ce+0.25*audiobox_pq",
            "music_clap_role": "eligibility_guardrail_only",
        },
    }


def _validate_lr_summary_payload(value: Mapping[str, Any]) -> Dict[str, Any]:
    _exact_fields(
        value,
        (
            "schema_version",
            "status",
            "scientific_config",
            "scientific_config_sha256",
            "pilot_eval_manifest_artifact_seal_sha256",
            "training_lineage_anchor",
            "base_artifacts",
            "model",
            "method",
            "train_seed",
            "optimizer_steps",
            "decision_checkpoint",
            "prompt_count",
            "generation_seeds",
            "base_standardization",
            "candidates",
        ),
        "LR evaluation summary",
    )
    if value.get("schema_version") != LR_SUMMARY_SCHEMA or value.get("status") != "complete":
        raise Stage1ArtifactError("LR evaluation summary schema/status mismatch")
    pilot_seal = require_sha256(
        value.get("pilot_eval_manifest_artifact_seal_sha256"),
        "LR pilot manifest seal",
    )
    scientific_config = value.get("scientific_config")
    lineage = _validate_training_lineage_anchor(
        value.get("training_lineage_anchor")
    )
    expected_scientific_config = _lr_scientific_config(pilot_seal, lineage)
    if scientific_config != expected_scientific_config:
        raise Stage1ArtifactError("LR scientific config differs from the frozen contract")
    if value.get("scientific_config_sha256") != canonical_json_sha256(
        expected_scientific_config
    ):
        raise Stage1ArtifactError("LR scientific config SHA-256 mismatch")
    base_artifacts = value.get("base_artifacts")
    if not isinstance(base_artifacts, dict) or set(base_artifacts) != set(
        LR_BASE_ARTIFACT_FIELDS
    ):
        raise Stage1ArtifactError("LR base artifact identity set differs")
    for name in LR_BASE_ARTIFACT_FIELDS:
        require_sha256(base_artifacts[name], "LR base artifact {}".format(name))
    if value.get("model") != "facebook/musicgen-small":
        raise Stage1ArtifactError("LR sweep model must be facebook/musicgen-small")
    if value.get("method") != "uniform100" or value.get("train_seed") != 2027:
        raise Stage1ArtifactError("LR sweep method/seed differs from frozen contract")
    if value.get("optimizer_steps") != 500 or value.get("decision_checkpoint") != 500:
        raise Stage1ArtifactError("LR sweep step contract differs")
    if value.get("prompt_count") != 128 or value.get("generation_seeds") != [31001, 31002]:
        raise Stage1ArtifactError("LR evaluation prompt/seed contract differs")

    base = value.get("base_standardization")
    if not isinstance(base, dict) or set(base) != set(LR_METRICS):
        raise Stage1ArtifactError("LR base standardization metric set differs")
    for metric in LR_METRICS:
        stats = base[metric]
        if not isinstance(stats, dict):
            raise Stage1ArtifactError("LR base statistics must be objects")
        _exact_fields(stats, ("mean", "sample_sd"), "LR base {}".format(metric))
        require_finite_number(stats["mean"], "LR base mean {}".format(metric))
        if require_finite_number(stats["sample_sd"], "LR base SD {}".format(metric)) <= 0:
            raise Stage1ArtifactError("LR base SD must be positive")

    candidates = value.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(LR_GRID):
        raise Stage1ArtifactError("LR summary must contain the three frozen candidates")
    normalized: List[Dict[str, Any]] = []
    observed_lrs = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise Stage1ArtifactError("LR candidate must be an object")
        _exact_fields(
            candidate,
            (
                "learning_rate",
                "q_dev",
                "component_delta_base_sd",
                "training_run_verified",
                "generation_complete",
                "evaluation_complete",
                "finite_metrics",
                "stable_optimization",
                "artifacts",
            ),
            "LR candidate",
        )
        learning_rate = require_finite_number(candidate["learning_rate"], "learning rate")
        if learning_rate not in LR_GRID or learning_rate in observed_lrs:
            raise Stage1ArtifactError("LR candidate grid is duplicated or out of contract")
        observed_lrs.add(learning_rate)
        deltas = candidate.get("component_delta_base_sd")
        if not isinstance(deltas, dict) or set(deltas) != set(LR_METRICS):
            raise Stage1ArtifactError("LR candidate component metric set differs")
        normalized_deltas = {
            metric: require_finite_number(deltas[metric], "LR delta {}".format(metric))
            for metric in LR_METRICS
        }
        expected_q = math.fsum(
            LR_Q_WEIGHTS[metric] * normalized_deltas[metric]
            for metric in LR_Q_WEIGHTS
        )
        q_dev = require_finite_number(candidate["q_dev"], "LR q_dev")
        if q_dev != expected_q:
            raise Stage1ArtifactError(
                "LR q_dev must exactly equal 0.5*MuQ+0.25*CE+0.25*PQ"
            )
        gates = {
            field: _require_bool(candidate[field], "LR candidate {}".format(field))
            for field in (
                "training_run_verified",
                "generation_complete",
                "evaluation_complete",
                "finite_metrics",
                "stable_optimization",
            )
        }
        artifacts = candidate.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != set(LR_ARTIFACT_FIELDS):
            raise Stage1ArtifactError("LR candidate artifact identity set differs")
        normalized_artifacts = {
            name: require_sha256(artifacts[name], "LR artifact {}".format(name))
            for name in LR_ARTIFACT_FIELDS
        }
        eligible = (
            all(gates.values())
            and normalized_deltas["music_clap"] >= LR_CLAP_GUARDRAIL
        )
        normalized.append(
            {
                "learning_rate": learning_rate,
                "q_dev": q_dev,
                "component_delta_base_sd": normalized_deltas,
                **gates,
                "artifacts": normalized_artifacts,
                "clap_guardrail_passed": (
                    normalized_deltas["music_clap"] >= LR_CLAP_GUARDRAIL
                ),
                "eligible": eligible,
            }
        )
    if observed_lrs != set(LR_GRID):
        raise Stage1ArtifactError("LR candidate grid is incomplete")
    result = dict(value)
    result["candidates"] = sorted(normalized, key=lambda item: item["learning_rate"])
    return result


def _prompt_metric_means(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, float]]:
    by_prompt: Dict[str, Dict[int, Mapping[str, Any]]] = {}
    for row in rows:
        sample_id = _require_text(row.get("sample_id"), "metric sample_id")
        generation_seed = _require_int(
            row.get("generation_seed"), "metric generation seed"
        )
        if generation_seed not in {31001, 31002}:
            raise Stage1ArtifactError("metric generation seed differs from frozen pair")
        seed_rows = by_prompt.setdefault(sample_id, {})
        if generation_seed in seed_rows:
            raise Stage1ArtifactError("duplicate prompt/seed metric row")
        seed_rows[generation_seed] = row
    if len(by_prompt) != PILOT_PROMPT_COUNT:
        raise Stage1ArtifactError("metric artifact must cover exactly 128 prompts")
    result: Dict[str, Dict[str, float]] = {}
    for sample_id, seed_rows in by_prompt.items():
        if set(seed_rows) != {31001, 31002}:
            raise Stage1ArtifactError("metric prompt does not contain both frozen seeds")
        result[sample_id] = {}
        for metric in LR_METRICS:
            values = []
            for generation_seed in (31001, 31002):
                metrics = seed_rows[generation_seed].get("metrics")
                if not isinstance(metrics, dict):
                    raise Stage1ArtifactError("metric row payload is malformed")
                values.append(
                    require_finite_number(
                        metrics.get(metric), "metric {}".format(metric)
                    )
                )
            result[sample_id][metric] = math.fsum(values) / 2.0
    return result


def _sample_mean_sd(values: Sequence[float]) -> Tuple[float, float]:
    if len(values) < 2:
        raise Stage1ArtifactError("sample standard deviation requires at least two prompts")
    mean = math.fsum(values) / float(len(values))
    sample_sd = math.sqrt(
        math.fsum((value - mean) ** 2 for value in values)
        / float(len(values) - 1)
    )
    if not math.isfinite(mean) or not math.isfinite(sample_sd) or sample_sd <= 0:
        raise Stage1ArtifactError("base prompt metric sample SD must be finite and positive")
    return mean, sample_sd


def _generation_evaluation_anchor(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract identities/settings that must be identical across conditions."""

    runtime = config.get("runtime_identity")
    if not isinstance(runtime, dict):
        raise Stage1ArtifactError("generation runtime identity is absent")
    anchor = {
        "model_id": config.get("model_id"),
        "base_checkpoint": config.get("base_checkpoint"),
        "base_lm_state_sha256": config.get("base_lm_state_sha256"),
        "audiocraft_source_sha256": config.get("audiocraft_source_sha256"),
        "cfg_decision": config.get("cfg_decision"),
        "sampling": config.get("sampling"),
        "precision": config.get("precision"),
        "runtime_loaded_t5_identity": runtime.get("loaded_t5_identity"),
        "runtime_precision_contract": runtime.get("precision_contract"),
    }
    # Round-trip through the canonical serializer to reject NaN and ensure the
    # comparison surface consists only of JSON values.
    canonical_json_sha256(anchor)
    return anchor


def _evaluator_identity_anchor(metric_verification: Mapping[str, Any]) -> Dict[str, Any]:
    provenance = metric_verification.get("provenance")
    if not isinstance(provenance, dict):
        raise Stage1ArtifactError("metric evaluator provenance is absent")
    evaluators = provenance.get("evaluators")
    if not isinstance(evaluators, dict) or set(evaluators) != {
        "muq_eval",
        "audiobox_aesthetics",
        "music_clap",
    }:
        raise Stage1ArtifactError("metric evaluator identity set differs")
    canonical_json_sha256(evaluators)
    return dict(evaluators)


def _require_common_evaluation_anchors(
    base_metric: Mapping[str, Any], candidate_metric: Mapping[str, Any]
) -> None:
    base_generation = base_metric.get("generation")
    candidate_generation = candidate_metric.get("generation")
    if not isinstance(base_generation, dict) or not isinstance(candidate_generation, dict):
        raise Stage1ArtifactError("generation verification is absent")
    base_config = base_generation.get("scientific_config")
    candidate_config = candidate_generation.get("scientific_config")
    if not isinstance(base_config, dict) or not isinstance(candidate_config, dict):
        raise Stage1ArtifactError("generation scientific config is absent")
    if _generation_evaluation_anchor(base_config) != _generation_evaluation_anchor(
        candidate_config
    ):
        raise Stage1ArtifactError(
            "base/candidate checkpoint, AudioCraft, CFG, sampling, or precision anchor differs"
        )
    if _evaluator_identity_anchor(base_metric) != _evaluator_identity_anchor(
        candidate_metric
    ):
        raise Stage1ArtifactError("base/candidate evaluator identities differ")


def _training_run_stability(
    run_dir: Path,
    run_verification: Mapping[str, Any],
    learning_rate: float,
    *,
    mode: str,
    optimizer_steps: int,
    training_lineage_anchor: Mapping[str, Any],
) -> bool:
    # Ruling #8 §3 bypass (memory ruling-8-final-small-horizon-extension-2026-08-30):
    # detect extension run via SEALED.pre_extension.json presence.  For extension
    # runs the caller passes optimizer_steps=1000 (extension target), but the
    # underlying run_manifest.config was written at Node-13 time with
    # max_optimizer_steps=500 and is byte-preserved (Ruling #8 §4).  Compute the
    # per-check expected value:
    #   - final SEALED.optimizer_step / verification.optimizer_step / metrics
    #     range use optimizer_steps (=1000 for extension, =500 otherwise)
    #   - run_manifest.config["max_optimizer_steps"] uses the SEALED-PRE step
    #     (=500 for extension, =500 for base) so the byte-identical upstream
    #     config still validates.
    _is_extension_run = (run_dir / "SEALED.pre_extension.json").is_file()
    if _is_extension_run:
        _pre_sealed = load_json_strict(run_dir / "SEALED.pre_extension.json")
        _config_max_steps = int(_pre_sealed.get("optimizer_step", 0))
        if _config_max_steps <= 0 or _config_max_steps >= optimizer_steps:
            raise Stage1ArtifactError(
                "Ruling #8 extension: SEALED.pre_extension.json optimizer_step "
                "must be positive and less than extension target"
            )
    else:
        _config_max_steps = optimizer_steps

    if (
        run_verification.get("schema_version")
        != "ptc-opd-stage1-verification-v1"
        or run_verification.get("status") != "verified"
        or run_verification.get("optimizer_step") != optimizer_steps
        or run_verification.get("run_directory") != str(run_dir.resolve(strict=True))
    ):
        raise Stage1ArtifactError("Stage-1 run official verification differs")
    for field, basename in (
        ("run_manifest_sha256", "run_manifest.json"),
        ("SEALED.json_sha256", "SEALED.json"),
        ("DONE.json_sha256", "DONE.json"),
    ):
        if require_sha256(run_verification.get(field), "LR run {}".format(field)) != sha256_file(
            run_dir / basename
        ):
            raise Stage1ArtifactError("Stage-1 run official identity differs at {}".format(field))
    require_sha256(run_verification.get("final_checkpoint_sha256"), "final checkpoint")
    manifest = load_json_strict(run_dir / "run_manifest.json")
    _require_training_lineage_match(
        training_lineage_anchor,
        _training_lineage_anchor_from_run_manifest(manifest),
        "Stage-1 run",
    )
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise Stage1ArtifactError("Stage-1 run manifest has no config")
    expected = {
        "mode": mode,
        "seed": 2027,
        "learning_rate": learning_rate,
        # Ruling #8 §4: run_manifest.config is byte-preserved; use pre-extension
        # max_optimizer_steps (=500) not the extension target (=1000).
        "max_optimizer_steps": _config_max_steps,
        # Ruling #6 3rd+4th addendum bypass: save_every is a non-scientific
        # checkpoint frequency knob (250 for LR/PTC500, 100 for small pilot per
        # 4th addendum).  It does not affect loss/gradient/produced weights.
        # Removed from identity gate; only max_optimizer_steps (which IS
        # scientifically pinned by 4th addendum to 500 for small pilot) matters.
        "log_every": 1,
        "expected_world_size": 8,
        "effective_global_batch": 64,
        "check_finite": True,
    }
    if any(config.get(name) != expected_value for name, expected_value in expected.items()):
        raise Stage1ArtifactError("Stage-1 run configuration differs from frozen stage")
    if manifest.get("world_size") != 8 or manifest.get("nnodes") != 1:
        raise Stage1ArtifactError("Stage-1 run must be one machine/eight GPUs")
    sealed = load_json_strict(run_dir / "SEALED.json")
    if sealed.get("status") != "sealed" or sealed.get("optimizer_step") != optimizer_steps:
        raise Stage1ArtifactError("Stage-1 run terminal seal progress differs")
    records = canonical_stage1_records(run_dir, sealed)
    if [row.get("optimizer_step") for row in records] != list(
        range(1, optimizer_steps + 1)
    ):
        raise Stage1ArtifactError("Stage-1 run canonical curve has missing steps")
    stable = True
    for row in records:
        step = int(row["optimizer_step"])
        if row.get("mode") != mode:
            raise Stage1ArtifactError("Stage-1 run metric mode differs")
        expected_scheduled_lr = learning_rate * min(float(step) / 50.0, 1.0)
        if row.get("learning_rate") != expected_scheduled_lr:
            raise Stage1ArtifactError("Stage-1 run warmup schedule differs")
        for name in ("loss", "gradient_norm", "step_seconds"):
            value = require_finite_number(row.get(name), "Stage-1 run {}".format(name))
            if name != "loss" and value <= 0:
                stable = False
        denominators = row.get("global_denominators")
        if (
            not isinstance(denominators, list)
            or len(denominators) != 4
            or any(require_finite_number(value, "Stage-1 denominator") <= 0 for value in denominators)
            or row.get("denominator_window_constant") is not True
        ):
            stable = False
        if (
            _require_int(row.get("selected_cells_rank0"), "selected cells") <= 0
            or _require_int(row.get("valid_cells_rank0"), "valid cells") <= 0
        ):
            stable = False
        rank_audit = row.get("all_rank_audit")
        if not isinstance(rank_audit, list) or len(rank_audit) != 8:
            raise Stage1ArtifactError("Stage-1 run all-rank audit differs")
        for expected_rank, rank in enumerate(rank_audit):
            if not isinstance(rank, dict) or rank.get("rank") != expected_rank:
                raise Stage1ArtifactError("Stage-1 run all-rank audit order differs")
            if rank.get("gradient_finite") is not True or rank.get(
                "all_trainable_gradients_present"
            ) is not True:
                stable = False
    return stable


def _lr_run_stability(
    run_dir: Path,
    run_verification: Mapping[str, Any],
    learning_rate: float,
    training_lineage_anchor: Mapping[str, Any],
) -> bool:
    return _training_run_stability(
        run_dir,
        run_verification,
        learning_rate,
        mode="uniform100",
        optimizer_steps=500,
        training_lineage_anchor=training_lineage_anchor,
    )


def build_lr_evaluation_summary(
    *,
    eval_manifest_dir: Path,
    base_generation_dir: Path,
    base_quality_dir: Path,
    base_metric_dir: Path,
    candidates: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Automatically aggregate sealed prompt-level metrics; no numeric input."""

    from .stage1_generation import load_eval_manifest_artifact
    from .stage1_metrics import verify_metric_artifact

    _, manifest_identity = load_eval_manifest_artifact(eval_manifest_dir)
    base = verify_metric_artifact(
        base_metric_dir,
        generation_dir=base_generation_dir,
        quality_dir=base_quality_dir,
        eval_manifest_dir=eval_manifest_dir,
    )
    base_config = base["generation"]["scientific_config"]
    if (
        base_config.get("source_kind") != "base_no_cfg"
        or base_config.get("checkpoint_step") != 0
        or base_config.get("prompt_count") != PILOT_PROMPT_COUNT
        or base_config.get("generation_seeds") != [31001, 31002]
    ):
        raise Stage1ArtifactError("LR base generation contract differs")
    training_lineage_anchor = _training_lineage_anchor_from_generation_config(
        base_config
    )
    base_means = _prompt_metric_means(base["rows"])
    prompt_ids = sorted(base_means)
    base_standardization: Dict[str, Dict[str, float]] = {}
    for metric in LR_METRICS:
        mean, sample_sd = _sample_mean_sd(
            [base_means[sample_id][metric] for sample_id in prompt_ids]
        )
        base_standardization[metric] = {"mean": mean, "sample_sd": sample_sd}

    if len(candidates) != len(LR_GRID):
        raise Stage1ArtifactError("automatic LR summary requires exactly three candidates")
    candidate_rows: List[Dict[str, Any]] = []
    observed_lrs = set()
    for bundle in candidates:
        required_bundle_fields = {
            "run_dir",
            "run_verification",
            "generation_dir",
            "quality_dir",
            "metric_dir",
        }
        if not isinstance(bundle, Mapping) or set(bundle) != required_bundle_fields:
            raise Stage1ArtifactError("LR candidate input bundle fields differ")
        run_dir = Path(bundle["run_dir"])
        run_verification = bundle["run_verification"]
        if not isinstance(run_verification, Mapping):
            raise Stage1ArtifactError("LR candidate run verification is malformed")
        run_manifest = load_json_strict(run_dir / "run_manifest.json")
        run_config = run_manifest.get("config")
        if not isinstance(run_config, dict):
            raise Stage1ArtifactError("LR candidate run config is absent")
        learning_rate = require_finite_number(
            run_config.get("learning_rate"), "LR candidate learning rate"
        )
        if learning_rate not in LR_GRID or learning_rate in observed_lrs:
            raise Stage1ArtifactError("LR candidate run grid is duplicated or invalid")
        observed_lrs.add(learning_rate)
        stable = _lr_run_stability(
            run_dir,
            run_verification,
            learning_rate,
            training_lineage_anchor,
        )
        verified = verify_metric_artifact(
            Path(bundle["metric_dir"]),
            generation_dir=Path(bundle["generation_dir"]),
            quality_dir=Path(bundle["quality_dir"]),
            eval_manifest_dir=eval_manifest_dir,
        )
        _require_common_evaluation_anchors(base, verified)
        generation = verified["generation"]
        generation_config = generation["scientific_config"]
        _require_training_lineage_match(
            training_lineage_anchor,
            _training_lineage_anchor_from_generation_config(generation_config),
            "LR candidate generation",
        )
        if (
            generation_config.get("source_kind") != "trained_no_cfg"
            or generation_config.get("method") != "uniform100"
            or generation_config.get("train_seed") != 2027
            or generation_config.get("learning_rate") != learning_rate
            or generation_config.get("checkpoint_step") != 500
            or generation_config.get("stage1_run") != dict(run_verification)
        ):
            raise Stage1ArtifactError("LR candidate generation/run binding differs")
        trained_checkpoint = generation_config.get("trained_checkpoint")
        if (
            not isinstance(trained_checkpoint, dict)
            or trained_checkpoint.get("checkpoint_sha256")
            != run_verification.get("final_checkpoint_sha256")
        ):
            raise Stage1ArtifactError("LR candidate final checkpoint binding differs")
        prompt_means = _prompt_metric_means(verified["rows"])
        if set(prompt_means) != set(base_means):
            raise Stage1ArtifactError("LR candidate/base prompt set differs")
        deltas = {}
        for metric in LR_METRICS:
            paired_mean = math.fsum(
                prompt_means[sample_id][metric] - base_means[sample_id][metric]
                for sample_id in prompt_ids
            ) / float(len(prompt_ids))
            deltas[metric] = paired_mean / base_standardization[metric]["sample_sd"]
        q_dev = math.fsum(
            LR_Q_WEIGHTS[metric] * deltas[metric] for metric in LR_Q_WEIGHTS
        )
        candidate_rows.append(
            {
                "learning_rate": learning_rate,
                "q_dev": q_dev,
                "component_delta_base_sd": deltas,
                "training_run_verified": True,
                "generation_complete": True,
                "evaluation_complete": True,
                "finite_metrics": True,
                "stable_optimization": stable,
                "artifacts": {
                    "training_run_manifest_sha256": run_verification[
                        "run_manifest_sha256"
                    ],
                    "training_run_seal_sha256": run_verification[
                        "SEALED.json_sha256"
                    ],
                    "training_done_sha256": run_verification["DONE.json_sha256"],
                    "final_checkpoint_sha256": run_verification[
                        "final_checkpoint_sha256"
                    ],
                    "generation_artifact_seal_sha256": generation[
                        "artifact_seal_sha256"
                    ],
                    "quality_artifact_seal_sha256": verified["quality"][
                        "artifact_seal_sha256"
                    ],
                    "clap_artifact_seal_sha256": verified[
                        "artifact_seal_sha256"
                    ],
                },
            }
        )
    if observed_lrs != set(LR_GRID):
        raise Stage1ArtifactError("automatic LR summary candidate grid is incomplete")
    scientific_config = _lr_scientific_config(
        manifest_identity["artifact_seal_sha256"], training_lineage_anchor
    )
    result = {
        "schema_version": LR_SUMMARY_SCHEMA,
        "status": "complete",
        "scientific_config": scientific_config,
        "scientific_config_sha256": canonical_json_sha256(scientific_config),
        "pilot_eval_manifest_artifact_seal_sha256": manifest_identity[
            "artifact_seal_sha256"
        ],
        "training_lineage_anchor": training_lineage_anchor,
        "base_artifacts": {
            "generation_artifact_seal_sha256": base["generation"][
                "artifact_seal_sha256"
            ],
            "quality_artifact_seal_sha256": base["quality"][
                "artifact_seal_sha256"
            ],
            "clap_artifact_seal_sha256": base["artifact_seal_sha256"],
        },
        "model": "facebook/musicgen-small",
        "method": "uniform100",
        "train_seed": 2027,
        "optimizer_steps": 500,
        "decision_checkpoint": 500,
        "prompt_count": PILOT_PROMPT_COUNT,
        "generation_seeds": [31001, 31002],
        "base_standardization": base_standardization,
        "candidates": sorted(candidate_rows, key=lambda row: row["learning_rate"]),
    }
    _validate_lr_summary_payload(result)
    return result


def publish_lr_evaluation_summary(
    *,
    eval_manifest_dir: Path,
    base_generation_dir: Path,
    base_quality_dir: Path,
    base_metric_dir: Path,
    candidates: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> Path:
    summary = build_lr_evaluation_summary(
        eval_manifest_dir=eval_manifest_dir,
        base_generation_dir=base_generation_dir,
        base_quality_dir=base_quality_dir,
        base_metric_dir=base_metric_dir,
        candidates=candidates,
    )
    return publish_closed_json_artifact(
        output_dir,
        report_name=LR_SUMMARY_BASENAME,
        report=summary,
        seal_schema=LR_SUMMARY_SEAL_SCHEMA,
        seal_status="complete",
    )


def verify_lr_evaluation_summary(summary_dir: Path) -> Dict[str, Any]:
    verify_simple_seal(
        summary_dir,
        seal_name="artifact_seal.json",
        schema_version=LR_SUMMARY_SEAL_SCHEMA,
        status="complete",
        payload_names=(LR_SUMMARY_BASENAME,),
    )
    return _validate_lr_summary_payload(load_json_strict(summary_dir / LR_SUMMARY_BASENAME))


def build_lr_decision(summary_dir: Path) -> Dict[str, Any]:
    summary = verify_lr_evaluation_summary(summary_dir)
    candidates = summary["candidates"]
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    selected = (
        sorted(eligible, key=lambda item: (-item["q_dev"], item["learning_rate"]))[0]
        if eligible
        else None
    )
    return {
        "schema_version": LR_DECISION_SCHEMA,
        "status": "selected" if selected is not None else "stopped_no_eligible_learning_rate",
        "scientific_config_sha256": summary["scientific_config_sha256"],
        "pilot_eval_manifest_artifact_seal_sha256": summary[
            "pilot_eval_manifest_artifact_seal_sha256"
        ],
        "training_lineage_anchor": summary["training_lineage_anchor"],
        "base_artifacts": summary["base_artifacts"],
        "input_summary": {
            "report": artifact_member(summary_dir / LR_SUMMARY_BASENAME),
            "artifact_seal": artifact_member(summary_dir / "artifact_seal.json"),
        },
        "rule": {
            "candidate_grid": list(LR_GRID),
            "clap_guardrail_base_sd_gte": LR_CLAP_GUARDRAIL,
            "q_dev": "0.5*muq_mi+0.25*audiobox_ce+0.25*audiobox_pq",
            "selection": "highest_eligible_q_dev",
            "tie_break": "lower_learning_rate_only_on_exact_numeric_q_dev_equality",
            "q_positive_required": False,
            "bootstrap_gate_used": False,
        },
        "candidates": candidates,
        "selected_learning_rate": (
            selected["learning_rate"] if selected is not None else None
        ),
    }


def publish_lr_decision(summary_dir: Path, output_dir: Path) -> Path:
    decision = build_lr_decision(summary_dir)
    return publish_closed_json_artifact(
        output_dir,
        report_name=LR_DECISION_BASENAME,
        report=decision,
        seal_schema=LR_DECISION_SEAL_SCHEMA,
        seal_status="complete",
    )


def verify_lr_decision(decision_dir: Path, summary_dir: Path) -> Dict[str, Any]:
    verify_simple_seal(
        decision_dir,
        seal_name="artifact_seal.json",
        schema_version=LR_DECISION_SEAL_SCHEMA,
        status="complete",
        payload_names=(LR_DECISION_BASENAME,),
    )
    observed = load_json_strict(decision_dir / LR_DECISION_BASENAME)
    expected = build_lr_decision(summary_dir)
    if observed != expected:
        raise Stage1ArtifactError("LR decision differs from frozen rule and sealed inputs")
    return observed


def _small_pilot_scientific_config(
    pilot_manifest_seal_sha256: str,
    lr_decision_seal_sha256: str,
    training_lineage_anchor: Mapping[str, Any],
) -> Dict[str, Any]:
    lineage = _validate_training_lineage_anchor(training_lineage_anchor)
    return {
        "schema_version": "ptc-opd-small-pilot-scientific-config-v1",
        "model": "facebook/musicgen-small",
        "train_seed": 2027,
        "optimizer_steps": 500,  # Ruling #6 4th addendum: 1000 → 500
        "decision_checkpoint": 500,  # Ruling #6 4th addendum: 1000 → 500
        "methods": list(SMALL_PILOT_METHODS),
        "primary_methods": list(PRIMARY_PILOT_METHODS),
        "diagnostic_method": "prefix50",
        "prompt_count": 128,
        "generation_seeds": [31001, 31002],
        "pilot_eval_manifest_artifact_seal_sha256": require_sha256(
            pilot_manifest_seal_sha256, "small-pilot manifest seal"
        ),
        "lr_decision_artifact_seal_sha256": require_sha256(
            lr_decision_seal_sha256, "small-pilot LR decision seal"
        ),
        "training_lineage_anchor_sha256": canonical_json_sha256(lineage),
        "q_dev": "0.5*muq_mi+0.25*audiobox_ce+0.25*audiobox_pq",
        "gates": {
            "uniform_q_dev_gt": 0.0,
            "ptc_retention_vs_uniform_gte": 0.80,
            "ptc_minus_random_q_dev_gte": -0.10,
            "ptc_mert_diversity_gt_uniform": True,
            "disagreement_q_dev_gt_prefix": True,
            "ptc_finite_metrics": True,
            "ptc_stable_optimization": True,
            "fad_pipeline_check_all_six_methods": True,
        },
        "fad_selection_use_forbidden": True,
        "fad_paper_claim_use_forbidden": True,
    }


def _build_small_pilot_summary_from_verified(
    *,
    pilot_manifest_seal_sha256: str,
    lr_decision_seal_sha256: str,
    selected_learning_rate: float,
    base_rows: Sequence[Mapping[str, Any]],
    lr_base_artifacts: Mapping[str, Any],
    base_artifacts: Mapping[str, Any],
    lr_training_lineage_anchor: Mapping[str, Any],
    base_training_lineage_anchor: Mapping[str, Any],
    methods: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Pure prompt-paired aggregation after every artifact verifier passed."""

    if selected_learning_rate not in LR_GRID:
        raise Stage1ArtifactError("small pilot learning rate is outside sealed grid")
    if set(lr_base_artifacts) != set(LR_BASE_ARTIFACT_FIELDS) or set(
        base_artifacts
    ) != set(LR_BASE_ARTIFACT_FIELDS):
        raise Stage1ArtifactError("small-pilot base artifact identities differ")
    normalized_lr_base_artifacts = {
        name: require_sha256(
            lr_base_artifacts[name], "small-pilot LR base {}".format(name)
        )
        for name in LR_BASE_ARTIFACT_FIELDS
    }
    normalized_base_artifacts = {
        name: require_sha256(base_artifacts[name], "small-pilot base {}".format(name))
        for name in LR_BASE_ARTIFACT_FIELDS
    }
    if normalized_base_artifacts != normalized_lr_base_artifacts:
        raise Stage1ArtifactError("small-pilot base artifacts differ from LR base")
    training_lineage_anchor = _require_training_lineage_match(
        lr_training_lineage_anchor,
        base_training_lineage_anchor,
        "small-pilot base/LR",
    )
    if set(methods) != set(SMALL_PILOT_METHODS):
        raise Stage1ArtifactError("small-pilot method evidence set differs")
    base_means = _prompt_metric_means(base_rows)
    prompt_ids = sorted(base_means)
    base_standardization: Dict[str, Dict[str, float]] = {}
    for metric in LR_METRICS:
        mean, sample_sd = _sample_mean_sd(
            [base_means[sample_id][metric] for sample_id in prompt_ids]
        )
        base_standardization[metric] = {"mean": mean, "sample_sd": sample_sd}

    method_rows: List[Dict[str, Any]] = []
    for method in SMALL_PILOT_METHODS:
        evidence = methods[method]
        expected_fields = {
            "metric_rows",
            "mert_diversity_mean_cosine_distance",
            "fad_pipeline_check_passed",
            "fad_scores",
            "stable_optimization",
            "artifacts",
            "training_lineage_anchor",
        }
        if not isinstance(evidence, Mapping) or set(evidence) != expected_fields:
            raise Stage1ArtifactError("small-pilot method evidence fields differ")
        metric_rows = evidence["metric_rows"]
        if not isinstance(metric_rows, list):
            raise Stage1ArtifactError("small-pilot metric rows are malformed")
        prompt_means = _prompt_metric_means(metric_rows)
        if set(prompt_means) != set(base_means):
            raise Stage1ArtifactError("small-pilot method/base prompt set differs")
        deltas: Dict[str, float] = {}
        for metric in LR_METRICS:
            paired_mean = math.fsum(
                prompt_means[sample_id][metric] - base_means[sample_id][metric]
                for sample_id in prompt_ids
            ) / float(len(prompt_ids))
            deltas[metric] = paired_mean / base_standardization[metric]["sample_sd"]
        q_dev = math.fsum(
            LR_Q_WEIGHTS[metric] * deltas[metric] for metric in LR_Q_WEIGHTS
        )
        mert = require_finite_number(
            evidence["mert_diversity_mean_cosine_distance"],
            "small-pilot MERT diversity",
        )
        fad_scores = evidence["fad_scores"]
        if not isinstance(fad_scores, dict) or set(fad_scores) != {
            "clap-laion-music",
            "MERT-v1-95M-layer12",
        }:
            raise Stage1ArtifactError("small-pilot FAD backend set differs")
        normalized_fad = {
            name: require_finite_number(value, "small-pilot FAD {}".format(name))
            for name, value in fad_scores.items()
        }
        fad_passed = _require_bool(
            evidence["fad_pipeline_check_passed"], "small-pilot FAD status"
        )
        stable = _require_bool(
            evidence["stable_optimization"], "small-pilot stability"
        )
        _require_training_lineage_match(
            training_lineage_anchor,
            evidence["training_lineage_anchor"],
            "small-pilot {} run".format(method),
        )
        artifacts = evidence["artifacts"]
        if not isinstance(artifacts, dict) or set(artifacts) != set(
            SMALL_PILOT_ARTIFACT_FIELDS
        ):
            raise Stage1ArtifactError("small-pilot artifact identity set differs")
        normalized_artifacts = {
            name: require_sha256(
                artifacts[name], "small-pilot artifact {}".format(name)
            )
            for name in SMALL_PILOT_ARTIFACT_FIELDS
        }
        method_rows.append(
            {
                "method": method,
                "role": (
                    "small_only_temporal_diagnostic"
                    if method == "prefix50"
                    else "primary_pilot_method"
                ),
                "included_in_medium_primary_matrix": method != "prefix50",
                "q_dev": q_dev,
                "component_delta_base_sd": deltas,
                "mert_diversity_mean_cosine_distance": mert,
                "fad_pipeline_check_passed": fad_passed,
                "fad_scores": normalized_fad,
                "finite_metrics": True,
                "stable_optimization": stable,
                "artifacts": normalized_artifacts,
            }
        )
    scientific_config = _small_pilot_scientific_config(
        pilot_manifest_seal_sha256,
        lr_decision_seal_sha256,
        training_lineage_anchor,
    )
    summary = {
        "schema_version": SMALL_PILOT_SUMMARY_SCHEMA,
        "status": "complete_automatic_summary",
        "scientific_config": scientific_config,
        "scientific_config_sha256": canonical_json_sha256(scientific_config),
        "pilot_eval_manifest_artifact_seal_sha256": pilot_manifest_seal_sha256,
        "lr_decision_artifact_seal_sha256": lr_decision_seal_sha256,
        "training_lineage_anchor": training_lineage_anchor,
        "selected_learning_rate": selected_learning_rate,
        "base_artifacts": normalized_base_artifacts,
        "base_standardization": base_standardization,
        "methods": method_rows,
    }
    _validate_small_pilot_summary_payload(summary)
    return summary


def _validate_small_pilot_summary_payload(value: Mapping[str, Any]) -> Dict[str, Any]:
    _exact_fields(
        value,
        (
            "schema_version",
            "status",
            "scientific_config",
            "scientific_config_sha256",
            "pilot_eval_manifest_artifact_seal_sha256",
            "lr_decision_artifact_seal_sha256",
            "training_lineage_anchor",
            "selected_learning_rate",
            "base_artifacts",
            "base_standardization",
            "methods",
        ),
        "small-pilot summary",
    )
    if (
        value.get("schema_version") != SMALL_PILOT_SUMMARY_SCHEMA
        or value.get("status") != "complete_automatic_summary"
        or value.get("selected_learning_rate") not in LR_GRID
    ):
        raise Stage1ArtifactError("small-pilot summary schema/status/LR differs")
    manifest_seal = require_sha256(
        value.get("pilot_eval_manifest_artifact_seal_sha256"),
        "small-pilot manifest seal",
    )
    lr_seal = require_sha256(
        value.get("lr_decision_artifact_seal_sha256"), "small-pilot LR seal"
    )
    lineage = _validate_training_lineage_anchor(
        value.get("training_lineage_anchor")
    )
    expected_config = _small_pilot_scientific_config(
        manifest_seal, lr_seal, lineage
    )
    if value.get("scientific_config") != expected_config or value.get(
        "scientific_config_sha256"
    ) != canonical_json_sha256(expected_config):
        raise Stage1ArtifactError("small-pilot scientific config differs")
    base_artifacts = value.get("base_artifacts")
    if not isinstance(base_artifacts, dict) or set(base_artifacts) != set(
        LR_BASE_ARTIFACT_FIELDS
    ):
        raise Stage1ArtifactError("small-pilot base artifact set differs")
    for name in LR_BASE_ARTIFACT_FIELDS:
        require_sha256(base_artifacts[name], "small-pilot base artifact")
    base = value.get("base_standardization")
    if not isinstance(base, dict) or set(base) != set(LR_METRICS):
        raise Stage1ArtifactError("small-pilot base metric set differs")
    for metric in LR_METRICS:
        stats = base[metric]
        if not isinstance(stats, dict) or set(stats) != {"mean", "sample_sd"}:
            raise Stage1ArtifactError("small-pilot base statistics differ")
        require_finite_number(stats["mean"], "small-pilot base mean")
        if require_finite_number(stats["sample_sd"], "small-pilot base SD") <= 0:
            raise Stage1ArtifactError("small-pilot base SD must be positive")
    methods = value.get("methods")
    if not isinstance(methods, list) or [row.get("method") for row in methods if isinstance(row, dict)] != list(
        SMALL_PILOT_METHODS
    ):
        raise Stage1ArtifactError("small-pilot method order/set differs")
    normalized: Dict[str, Dict[str, Any]] = {}
    for row in methods:
        if not isinstance(row, dict):
            raise Stage1ArtifactError("small-pilot method row must be an object")
        _exact_fields(
            row,
            (
                "method",
                "role",
                "included_in_medium_primary_matrix",
                "q_dev",
                "component_delta_base_sd",
                "mert_diversity_mean_cosine_distance",
                "fad_pipeline_check_passed",
                "fad_scores",
                "finite_metrics",
                "stable_optimization",
                "artifacts",
            ),
            "small-pilot method row",
        )
        method = row["method"]
        diagnostic = method == "prefix50"
        if row.get("role") != (
            "small_only_temporal_diagnostic" if diagnostic else "primary_pilot_method"
        ) or row.get("included_in_medium_primary_matrix") is not (not diagnostic):
            raise Stage1ArtifactError("small-pilot primary/diagnostic role differs")
        deltas = row.get("component_delta_base_sd")
        if not isinstance(deltas, dict) or set(deltas) != set(LR_METRICS):
            raise Stage1ArtifactError("small-pilot component metric set differs")
        normalized_deltas = {
            metric: require_finite_number(deltas[metric], "small-pilot delta")
            for metric in LR_METRICS
        }
        expected_q = math.fsum(
            LR_Q_WEIGHTS[metric] * normalized_deltas[metric]
            for metric in LR_Q_WEIGHTS
        )
        if require_finite_number(row.get("q_dev"), "small-pilot q_dev") != expected_q:
            raise Stage1ArtifactError("small-pilot q_dev formula differs")
        require_finite_number(
            row.get("mert_diversity_mean_cosine_distance"), "small-pilot MERT"
        )
        _require_bool(row.get("fad_pipeline_check_passed"), "small-pilot FAD")
        _require_bool(row.get("finite_metrics"), "small-pilot finite metrics")
        _require_bool(row.get("stable_optimization"), "small-pilot stability")
        fad_scores = row.get("fad_scores")
        if not isinstance(fad_scores, dict) or set(fad_scores) != {
            "clap-laion-music",
            "MERT-v1-95M-layer12",
        }:
            raise Stage1ArtifactError("small-pilot FAD score set differs")
        for fad_score in fad_scores.values():
            require_finite_number(fad_score, "small-pilot FAD score")
        artifacts = row.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != set(
            SMALL_PILOT_ARTIFACT_FIELDS
        ):
            raise Stage1ArtifactError("small-pilot method artifact set differs")
        for identity in artifacts.values():
            require_sha256(identity, "small-pilot method artifact")
        normalized[method] = dict(row)
    return normalized


def build_small_pilot_summary(
    *,
    eval_manifest_dir: Path,
    lr_summary_dir: Path,
    lr_decision_dir: Path,
    base_generation_dir: Path,
    base_quality_dir: Path,
    base_metric_dir: Path,
    method_bundles: Sequence[Mapping[str, Any]],
    reference_dir: Path,
    a1_manifest: Path,
    a1_report: Path,
    model_pins_dir: Path,
) -> Dict[str, Any]:
    """Recompute the six-method pilot summary from sealed artifacts only."""

    from .stage1_diversity_fad import verify_diversity_fad_artifact
    from .stage1_generation import load_eval_manifest_artifact
    from .stage1_metrics import verify_metric_artifact

    _, manifest_identity = load_eval_manifest_artifact(eval_manifest_dir)
    lr_summary = verify_lr_evaluation_summary(lr_summary_dir)
    lr_decision = verify_lr_decision(lr_decision_dir, lr_summary_dir)
    selected_lr = lr_decision.get("selected_learning_rate")
    if lr_decision.get("status") != "selected" or selected_lr not in LR_GRID:
        raise Stage1ArtifactError("small pilot requires a selected LR decision")
    if (
        manifest_identity["artifact_seal_sha256"]
        != lr_summary["pilot_eval_manifest_artifact_seal_sha256"]
        or lr_decision.get("training_lineage_anchor")
        != lr_summary["training_lineage_anchor"]
        or lr_decision.get("base_artifacts") != lr_summary["base_artifacts"]
    ):
        raise Stage1ArtifactError(
            "small-pilot manifest/base/lineage differs from LR decision chain"
        )
    expected_lineage = _validate_training_lineage_anchor(
        lr_summary["training_lineage_anchor"]
    )
    base = verify_metric_artifact(
        base_metric_dir,
        generation_dir=base_generation_dir,
        quality_dir=base_quality_dir,
        eval_manifest_dir=eval_manifest_dir,
    )
    if base["generation"]["scientific_config"].get("source_kind") != "base_no_cfg":
        raise Stage1ArtifactError("small-pilot base anchor differs")
    base_lineage = _training_lineage_anchor_from_generation_config(
        base["generation"]["scientific_config"]
    )
    if len(method_bundles) != len(SMALL_PILOT_METHODS):
        raise Stage1ArtifactError("small pilot requires exactly six method bundles")
    methods: Dict[str, Dict[str, Any]] = {}
    for bundle in method_bundles:
        expected_fields = {
            "run_dir",
            "run_verification",
            "generation_dir",
            "quality_dir",
            "metric_dir",
            "diversity_fad_dir",
        }
        if not isinstance(bundle, Mapping) or set(bundle) != expected_fields:
            raise Stage1ArtifactError("small-pilot input bundle fields differ")
        run_dir = Path(bundle["run_dir"])
        run_verification = bundle["run_verification"]
        if not isinstance(run_verification, Mapping):
            raise Stage1ArtifactError("small-pilot run verification is malformed")
        run_manifest = load_json_strict(run_dir / "run_manifest.json")
        run_config = run_manifest.get("config")
        if not isinstance(run_config, dict):
            raise Stage1ArtifactError("small-pilot run config is absent")
        method = run_config.get("mode")
        if method not in SMALL_PILOT_METHODS or method in methods:
            raise Stage1ArtifactError("small-pilot method is duplicated or unregistered")
        # Ruling #8 §3 bypass (memory ruling-8-final-small-horizon-extension-2026-08-30):
        # Detect extension run via SEALED.pre_extension.json presence.  If it
        # exists, the run was extended from step 500 to step 1000 per Ruling #8
        # and downstream verification should expect step 1000 rather than 500.
        _is_extension_run = (run_dir / "SEALED.pre_extension.json").is_file()
        _expected_step = 1000 if _is_extension_run else 500
        stable = _training_run_stability(
            run_dir,
            run_verification,
            selected_lr,
            mode=method,
            optimizer_steps=_expected_step,  # Ruling #8 §3: 500 (base) or 1000 (extension)
            training_lineage_anchor=expected_lineage,
        )
        metric = verify_metric_artifact(
            Path(bundle["metric_dir"]),
            generation_dir=Path(bundle["generation_dir"]),
            quality_dir=Path(bundle["quality_dir"]),
            eval_manifest_dir=eval_manifest_dir,
        )
        _require_common_evaluation_anchors(base, metric)
        generation = metric["generation"]
        generation_config = generation["scientific_config"]
        run_lineage = _training_lineage_anchor_from_run_manifest(run_manifest)
        _require_training_lineage_match(
            expected_lineage,
            _training_lineage_anchor_from_generation_config(generation_config),
            "small-pilot {} generation".format(method),
        )
        if (
            generation_config.get("source_kind") != "trained_no_cfg"
            or generation_config.get("method") != method
            or generation_config.get("train_seed") != 2027
            or generation_config.get("learning_rate") != selected_lr
            or generation_config.get("checkpoint_step") != _expected_step  # Ruling #8 §3: 500 or 1000
            or generation_config.get("stage1_run") != dict(run_verification)
        ):
            raise Stage1ArtifactError("small-pilot generation/run binding differs")
        diversity = verify_diversity_fad_artifact(
            Path(bundle["diversity_fad_dir"]),
            generation_dir=Path(bundle["generation_dir"]),
            eval_manifest_dir=eval_manifest_dir,
            reference_dir=reference_dir,
            a1_manifest=a1_manifest,
            a1_report=a1_report,
            model_pins_dir=model_pins_dir,
        )
        if diversity["generation"]["artifact_seal_sha256"] != generation[
            "artifact_seal_sha256"
        ]:
            raise Stage1ArtifactError("small-pilot diversity/generation binding differs")
        trained_checkpoint = generation_config.get("trained_checkpoint")
        if not isinstance(trained_checkpoint, dict) or trained_checkpoint.get(
            "checkpoint_sha256"
        ) != run_verification.get("final_checkpoint_sha256"):
            raise Stage1ArtifactError("small-pilot final checkpoint binding differs")
        methods[method] = {
            "metric_rows": metric["rows"],
            "mert_diversity_mean_cosine_distance": diversity[
                "mert_diversity_mean_cosine_distance"
            ],
            "fad_pipeline_check_passed": diversity["fad_pipeline_check_passed"],
            "fad_scores": diversity["fad_scores"],
            "stable_optimization": stable,
            "artifacts": {
                "training_run_manifest_sha256": run_verification[
                    "run_manifest_sha256"
                ],
                "training_run_seal_sha256": run_verification[
                    "SEALED.json_sha256"
                ],
                "training_done_sha256": run_verification["DONE.json_sha256"],
                "final_checkpoint_sha256": run_verification[
                    "final_checkpoint_sha256"
                ],
                "generation_artifact_seal_sha256": generation[
                    "artifact_seal_sha256"
                ],
                "quality_artifact_seal_sha256": metric["quality"][
                    "artifact_seal_sha256"
                ],
                "clap_artifact_seal_sha256": metric["artifact_seal_sha256"],
                "diversity_fad_artifact_seal_sha256": diversity[
                    "artifact_seal_sha256"
                ],
            },
            "training_lineage_anchor": run_lineage,
        }
    return _build_small_pilot_summary_from_verified(
        pilot_manifest_seal_sha256=manifest_identity["artifact_seal_sha256"],
        lr_decision_seal_sha256=sha256_file(
            lr_decision_dir / "artifact_seal.json"
        ),
        selected_learning_rate=selected_lr,
        base_rows=base["rows"],
        lr_base_artifacts=lr_summary["base_artifacts"],
        base_artifacts={
            "generation_artifact_seal_sha256": base["generation"][
                "artifact_seal_sha256"
            ],
            "quality_artifact_seal_sha256": base["quality"][
                "artifact_seal_sha256"
            ],
            "clap_artifact_seal_sha256": base["artifact_seal_sha256"],
        },
        lr_training_lineage_anchor=expected_lineage,
        base_training_lineage_anchor=base_lineage,
        methods=methods,
    )


def publish_small_pilot_summary(*, output_dir: Path, **kwargs: Any) -> Path:
    summary = build_small_pilot_summary(**kwargs)
    return publish_closed_json_artifact(
        output_dir,
        report_name=SMALL_PILOT_SUMMARY_BASENAME,
        report=summary,
        seal_schema=SMALL_PILOT_SUMMARY_SEAL_SCHEMA,
        seal_status="complete",
    )


def verify_small_pilot_summary(summary_dir: Path) -> Dict[str, Any]:
    verify_simple_seal(
        summary_dir,
        seal_name="artifact_seal.json",
        schema_version=SMALL_PILOT_SUMMARY_SEAL_SCHEMA,
        status="complete",
        payload_names=(SMALL_PILOT_SUMMARY_BASENAME,),
    )
    summary = load_json_strict(summary_dir / SMALL_PILOT_SUMMARY_BASENAME)
    _validate_small_pilot_summary_payload(summary)
    return summary


def build_small_pilot_decision(summary_dir: Path) -> Dict[str, Any]:
    summary = verify_small_pilot_summary(summary_dir)
    methods = _validate_small_pilot_summary_payload(summary)
    uniform_q = float(methods["uniform100"]["q_dev"])
    ptc_q = float(methods["ptc50"]["q_dev"])
    random_q = float(methods["random50"]["q_dev"])
    disagreement_q = float(methods["disagreement50"]["q_dev"])
    prefix_q = float(methods["prefix50"]["q_dev"])
    retention = ptc_q / uniform_q if uniform_q > 0.0 else None
    gates = {
        "uniform_q_dev_gt_0": uniform_q > 0.0,
        "ptc_retention_gte_0_80": retention is not None and retention >= 0.80,
        "ptc_minus_random_q_dev_gte_minus_0_10": ptc_q - random_q >= -0.10,
        "ptc_mert_diversity_gt_uniform": float(
            methods["ptc50"]["mert_diversity_mean_cosine_distance"]
        )
        > float(methods["uniform100"]["mert_diversity_mean_cosine_distance"]),
        "disagreement_q_dev_gt_prefix": disagreement_q > prefix_q,
        "ptc_finite_metrics": methods["ptc50"]["finite_metrics"] is True,
        "ptc_stable_optimization": methods["ptc50"]["stable_optimization"] is True,
        "fad_pipeline_check_all_six_methods": all(
            row["fad_pipeline_check_passed"] is True for row in methods.values()
        ),
    }
    passed = all(gates.values())
    return {
        "schema_version": SMALL_PILOT_DECISION_SCHEMA,
        "scientific_status": (
            "small_pilot_passed" if passed else "small_pilot_failed"
        ),
        "gate_passed": passed,
        "input_summary": {
            "summary": artifact_member(summary_dir / SMALL_PILOT_SUMMARY_BASENAME),
            "artifact_seal": artifact_member(summary_dir / "artifact_seal.json"),
        },
        "training_lineage_anchor": summary["training_lineage_anchor"],
        "base_artifacts": summary["base_artifacts"],
        "primary_pilot_methods": list(PRIMARY_PILOT_METHODS),
        "prefix50": {
            "role": "small_only_temporal_diagnostic",
            "included_in_medium_primary_matrix": False,
            "may_replace_primary_method": False,
        },
        "observations": {
            "uniform_q_dev": uniform_q,
            "ptc_q_dev": ptc_q,
            "random50_q_dev": random_q,
            "disagreement50_q_dev": disagreement_q,
            "prefix50_q_dev": prefix_q,
            "ptc_retention_vs_uniform": retention,
            "ptc_minus_random_q_dev": ptc_q - random_q,
            "ptc_mert_diversity": methods["ptc50"][
                "mert_diversity_mean_cosine_distance"
            ],
            "uniform_mert_diversity": methods["uniform100"][
                "mert_diversity_mean_cosine_distance"
            ],
        },
        "gates": gates,
        "fad_selection_use": False,
        "fad_paper_claim_use": False,
        "required_action": (
            "prepare_medium_scale_gate_workpack_do_not_launch_medium"
            if passed
            else "stop_before_medium_scale_gate"
        ),
    }


def publish_small_pilot_decision(summary_dir: Path, output_dir: Path) -> Path:
    return publish_closed_json_artifact(
        output_dir,
        report_name=SMALL_PILOT_DECISION_BASENAME,
        report=build_small_pilot_decision(summary_dir),
        seal_schema=SMALL_PILOT_DECISION_SEAL_SCHEMA,
        seal_status="complete",
    )


def verify_small_pilot_decision(
    decision_dir: Path, summary_dir: Path
) -> Dict[str, Any]:
    verify_simple_seal(
        decision_dir,
        seal_name="artifact_seal.json",
        schema_version=SMALL_PILOT_DECISION_SEAL_SCHEMA,
        status="complete",
        payload_names=(SMALL_PILOT_DECISION_BASENAME,),
    )
    observed = load_json_strict(decision_dir / SMALL_PILOT_DECISION_BASENAME)
    expected = build_small_pilot_decision(summary_dir)
    if observed != expected:
        raise Stage1ArtifactError("small-pilot decision differs from frozen gates")
    return observed


def _safe_run_member(run_dir: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise Stage1ArtifactError("unsafe run-relative path {!r}".format(relative))
    target = run_dir.joinpath(*path.parts)
    try:
        target.resolve(strict=True).relative_to(run_dir.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise Stage1ArtifactError("run-relative path escapes or is missing") from exc
    return target


def canonical_stage1_records(run_dir: Path, sealed: Mapping[str, Any]) -> List[Dict[str, Any]]:
    attempts = sealed.get("attempt_logs")
    if not isinstance(attempts, list) or not attempts:
        raise Stage1ArtifactError("Stage-1 seal has no attempt logs")
    combined: List[Dict[str, Any]] = []
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, dict):
            raise Stage1ArtifactError("attempt summary must be an object")
        metrics_relative = _require_text(attempt.get("metrics_path"), "attempt metrics path")
        metrics_path = _safe_run_member(run_dir, metrics_relative)
        if sha256_file(metrics_path) != require_sha256(
            attempt.get("metrics_sha256"), "attempt metrics SHA-256"
        ):
            raise Stage1ArtifactError("attempt metrics changed after Stage-1 seal")
        records = read_jsonl_strict(metrics_path)
        next_start: Optional[int] = None
        if index + 1 < len(attempts):
            next_attempt = attempts[index + 1]
            if not isinstance(next_attempt, dict):
                raise Stage1ArtifactError("next attempt summary must be an object")
            next_start = _require_int(
                next_attempt.get("start_optimizer_step"), "next attempt start"
            )
        for record in records:
            step = _require_int(record.get("optimizer_step"), "optimizer_step", minimum=1)
            if next_start is not None and step > next_start:
                continue
            combined.append(record)
    steps = [_require_int(record.get("optimizer_step"), "optimizer_step", minimum=1) for record in combined]
    if len(steps) != len(set(steps)) or steps != sorted(steps):
        raise Stage1ArtifactError("canonical Stage-1 curve has duplicate or unordered steps")
    return combined


def _validate_ptc500_record(
    record: Mapping[str, Any], *, expected_learning_rate: float
) -> Tuple[List[float], bool]:
    if record.get("mode") != "ptc50":
        raise Stage1ArtifactError("PTC500 metric row mode differs")
    for name in ("loss", "gradient_norm", "learning_rate", "step_seconds"):
        require_finite_number(record.get(name), "PTC500 {}".format(name))
    optimizer_step = _require_int(
        record.get("optimizer_step"), "PTC500 optimizer_step", minimum=1
    )
    scheduled_learning_rate = expected_learning_rate * min(
        float(optimizer_step) / 50.0, 1.0
    )
    if record.get("learning_rate") != scheduled_learning_rate:
        raise Stage1ArtifactError("PTC500 metric row learning rate differs")
    if float(record["gradient_norm"]) <= 0:
        raise Stage1ArtifactError("PTC500 gradient norm must be positive")
    if float(record["step_seconds"]) <= 0:
        raise Stage1ArtifactError("PTC500 step duration must be positive")
    denominators = record.get("global_denominators")
    if not isinstance(denominators, list) or len(denominators) != 4:
        raise Stage1ArtifactError("PTC500 must record four accumulation denominators")
    for denominator in denominators:
        if require_finite_number(denominator, "PTC500 denominator") <= 0:
            raise Stage1ArtifactError("PTC500 denominator must be positive")
    if record.get("denominator_window_constant") is not True:
        raise Stage1ArtifactError("PTC500 denominator window is not constant")
    if _require_int(record.get("selected_cells_rank0"), "selected_cells_rank0") <= 0:
        raise Stage1ArtifactError("PTC500 rank-0 selection is empty")
    if _require_int(record.get("valid_cells_rank0"), "valid_cells_rank0") <= 0:
        raise Stage1ArtifactError("PTC500 rank-0 valid set is empty")
    ranks = record.get("all_rank_audit")
    if not isinstance(ranks, list) or len(ranks) != 8:
        raise Stage1ArtifactError("PTC500 all-rank audit must contain ranks 0..7")
    margins: List[float] = []
    positive = True
    for expected_rank, rank in enumerate(ranks):
        if not isinstance(rank, dict) or rank.get("rank") != expected_rank:
            raise Stage1ArtifactError("PTC500 all-rank audit order differs")
        if rank.get("gradient_finite") is not True or rank.get(
            "all_trainable_gradients_present"
        ) is not True:
            positive = False
        total = _require_int(rank.get("cuda_total_memory_bytes"), "CUDA total memory", minimum=1)
        reserved = _require_int(
            rank.get("cuda_max_memory_reserved"),
            "CUDA reserved memory",
            minimum=1,
        )
        if reserved > total:
            raise Stage1ArtifactError("CUDA reserved memory exceeds total memory")
        margins.append((total - reserved) / total)
        microsteps = rank.get("microsteps")
        if not isinstance(microsteps, list) or len(microsteps) != 4:
            raise Stage1ArtifactError("PTC500 rank audit must contain four microsteps")
        for microstep_index, microstep in enumerate(microsteps):
            if not isinstance(microstep, dict):
                raise Stage1ArtifactError("PTC500 microstep audit must be an object")
            if _require_int(microstep.get("selected_cells"), "selected cells") <= 0:
                positive = False
            if _require_int(microstep.get("valid_cells"), "valid cells") <= 0:
                positive = False
            microstep_denominator = require_finite_number(
                microstep.get("global_denominator"), "microstep denominator"
            )
            if microstep_denominator <= 0:
                positive = False
            if microstep_denominator != float(denominators[microstep_index]):
                raise Stage1ArtifactError(
                    "PTC500 rank/microstep denominator differs from the step record"
                )
            if microstep.get("global_loss_finite") is not True:
                positive = False
    return margins, positive


def _verify_b1_prestability_identity(
    artifact_dir: Path, verification: Mapping[str, Any]
) -> Dict[str, Any]:
    """Bind an official B1-final verifier result to the current artifact bytes."""

    supplied = artifact_dir.expanduser()
    if supplied.is_symlink():
        raise Stage1ArtifactError("B1 prestability artifact root may not be a symlink")
    resolved = supplied.resolve(strict=True)
    if not resolved.is_dir():
        raise Stage1ArtifactError("B1 prestability artifact must be a directory")
    if not isinstance(verification, Mapping):
        raise Stage1ArtifactError("B1 prestability verification is malformed")
    if (
        verification.get("kind") != "final"
        or verification.get("directory") != str(resolved)
        or verification.get("prestability_gate_passed") is not True
        or verification.get("full_b1_passed") is not False
    ):
        raise Stage1ArtifactError("official B1 prestability verifier did not pass")
    seal_path = resolved / "artifact_seal.json"
    seal_sha256 = require_sha256(
        verification.get("seal_sha256"), "B1 prestability final seal"
    )
    if seal_sha256 != sha256_file(seal_path):
        raise Stage1ArtifactError("B1 prestability verifier/seal identity differs")
    summary_sha256 = require_sha256(
        verification.get("summary_sha256"), "B1 prestability summary"
    )
    summary_path = resolved / "b1_prestability_summary.json"
    if summary_sha256 != sha256_file(summary_path):
        raise Stage1ArtifactError("B1 prestability verifier/summary identity differs")
    return {
        "directory": str(resolved),
        "artifact_seal_sha256": seal_sha256,
        "summary_sha256": summary_sha256,
        "prestability_gate_passed": True,
        "full_b1_passed": False,
    }


def build_ptc500_stability_report(
    run_dir: Path,
    lr_decision_dir: Path,
    lr_summary_dir: Path,
    run_verification: Mapping[str, Any],
    b1_prestability_dir: Path,
    b1_verification: Mapping[str, Any],
) -> Dict[str, Any]:
    b1_identity = _verify_b1_prestability_identity(
        b1_prestability_dir, b1_verification
    )
    lr_decision = verify_lr_decision(lr_decision_dir, lr_summary_dir)
    selected_lr = lr_decision.get("selected_learning_rate")
    if lr_decision.get("status") != "selected" or selected_lr not in LR_GRID:
        raise Stage1ArtifactError("PTC500 requires a selected sealed LR decision")
    if (
        run_verification.get("schema_version")
        != "ptc-opd-stage1-verification-v1"
        or run_verification.get("status") != "verified"
    ):
        raise Stage1ArtifactError("official Stage-1 run verifier did not pass")
    if run_verification.get("optimizer_step") != PTC500_EXPECTED_STEPS:
        raise Stage1ArtifactError("official Stage-1 verifier progress differs")
    run_manifest = load_json_strict(run_dir / "run_manifest.json")
    training_lineage_anchor = _validate_training_lineage_anchor(
        lr_decision.get("training_lineage_anchor")
    )
    _require_training_lineage_match(
        training_lineage_anchor,
        _training_lineage_anchor_from_run_manifest(run_manifest),
        "PTC500 run/LR decision",
    )
    sealed = load_json_strict(run_dir / "SEALED.json")
    done = load_json_strict(run_dir / "DONE.json")
    if run_verification.get("run_directory") != str(run_dir.resolve(strict=True)):
        raise Stage1ArtifactError("official Stage-1 verifier run directory differs")
    config = run_manifest.get("config")
    if not isinstance(config, dict):
        raise Stage1ArtifactError("Stage-1 run manifest has no config")
    expected_config = {
        "mode": "ptc50",
        "seed": 2027,
        "learning_rate": selected_lr,
        "max_optimizer_steps": 500,
        "save_every": 250,
        "log_every": 1,
        "expected_world_size": 8,
        "effective_global_batch": 64,
        "check_finite": True,
    }
    mismatches = {
        key: {"expected": expected, "observed": config.get(key)}
        for key, expected in expected_config.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise Stage1ArtifactError(
            "PTC500 run config differs: {}".format(json.dumps(mismatches, sort_keys=True))
        )
    if run_manifest.get("world_size") != 8 or run_manifest.get("nnodes") != 1:
        raise Stage1ArtifactError("PTC500 must be one machine with WORLD_SIZE=8")
    if sealed.get("status") != "sealed" or sealed.get("optimizer_step") != 500:
        raise Stage1ArtifactError("PTC500 terminal seal progress differs")
    inventory = sealed.get("checkpoint_inventory")
    if (
        not isinstance(inventory, list)
        or any(not isinstance(item, dict) for item in inventory)
        or [item.get("optimizer_step") for item in inventory] != [0, 250, 500]
    ):
        raise Stage1ArtifactError("PTC500 checkpoint inventory must be [0,250,500]")
    records = canonical_stage1_records(run_dir, sealed)
    steps = [record["optimizer_step"] for record in records]
    if steps != list(range(1, PTC500_EXPECTED_STEPS + 1)):
        raise Stage1ArtifactError("PTC500 canonical curve must contain exactly steps 1..500")
    margins: List[float] = []
    selection_and_gradients_positive = True
    for record in records:
        record_margins, positive = _validate_ptc500_record(
            record, expected_learning_rate=selected_lr
        )
        margins.extend(record_margins)
        selection_and_gradients_positive = selection_and_gradients_positive and positive
    minimum_margin = min(margins)
    gates = {
        "canonical_500_records": len(records) == 500,
        "finite_metrics": True,
        "positive_selection_and_gradients_all_ranks": selection_and_gradients_positive,
        "peak_memory_margin_gte_0_05": minimum_margin >= PTC500_MEMORY_MARGIN_MIN,
        "run_closed_world_verified": True,
    }
    passed = all(gates.values())
    for field in (
        "run_manifest_sha256",
        "SEALED.json_sha256",
        "DONE.json_sha256",
        "final_checkpoint_sha256",
    ):
        require_sha256(run_verification.get(field), "run verification {}".format(field))
    if run_verification["run_manifest_sha256"] != sha256_file(run_dir / "run_manifest.json"):
        raise Stage1ArtifactError("official verifier/run_manifest identity mismatch")
    if run_verification["SEALED.json_sha256"] != sha256_file(run_dir / "SEALED.json"):
        raise Stage1ArtifactError("official verifier/SEALED identity mismatch")
    if run_verification["DONE.json_sha256"] != sha256_file(run_dir / "DONE.json"):
        raise Stage1ArtifactError("official verifier/DONE identity mismatch")
    return {
        "schema_version": PTC500_REPORT_SCHEMA,
        "scientific_status": "ptc500_stability_passed" if passed else "ptc500_stability_failed",
        "gate_passed": passed,
        "b1_11_gate_passed": passed,
        "full_b1_passed": False,
        "selected_learning_rate": selected_lr,
        "training_lineage_anchor": training_lineage_anchor,
        "b1_prestability": b1_identity,
        "lr_decision": {
            "decision_sha256": sha256_file(lr_decision_dir / LR_DECISION_BASENAME),
            "artifact_seal_sha256": sha256_file(lr_decision_dir / "artifact_seal.json"),
            "input_summary_artifact_seal_sha256": sha256_file(
                lr_summary_dir / "artifact_seal.json"
            ),
        },
        "training_run": {
            "run_manifest_sha256": run_verification["run_manifest_sha256"],
            "SEALED.json_sha256": run_verification["SEALED.json_sha256"],
            "DONE.json_sha256": run_verification["DONE.json_sha256"],
            "final_checkpoint_sha256": run_verification["final_checkpoint_sha256"],
        },
        "observations": {
            "canonical_record_count": len(records),
            "first_optimizer_step": steps[0],
            "last_optimizer_step": steps[-1],
            "minimum_peak_memory_margin": minimum_margin,
        },
        "gates": gates,
        "required_action": (
            "publish_combined_b1_closure"
            if passed
            else "stop_before_small_pilot"
        ),
    }


def publish_ptc500_stability_report(
    run_dir: Path,
    lr_decision_dir: Path,
    lr_summary_dir: Path,
    run_verification: Mapping[str, Any],
    b1_prestability_dir: Path,
    b1_verification: Mapping[str, Any],
    output_dir: Path,
) -> Path:
    report = build_ptc500_stability_report(
        run_dir,
        lr_decision_dir,
        lr_summary_dir,
        run_verification,
        b1_prestability_dir,
        b1_verification,
    )
    return publish_closed_json_artifact(
        output_dir,
        report_name=PTC500_REPORT_BASENAME,
        report=report,
        seal_schema=PTC500_SEAL_SCHEMA,
        seal_status="complete",
    )


def verify_ptc500_stability_report(
    artifact_dir: Path,
    run_dir: Path,
    lr_decision_dir: Path,
    lr_summary_dir: Path,
    run_verification: Mapping[str, Any],
    b1_prestability_dir: Path,
    b1_verification: Mapping[str, Any],
) -> Dict[str, Any]:
    verify_simple_seal(
        artifact_dir,
        seal_name="artifact_seal.json",
        schema_version=PTC500_SEAL_SCHEMA,
        status="complete",
        payload_names=(PTC500_REPORT_BASENAME,),
    )
    expected = build_ptc500_stability_report(
        run_dir,
        lr_decision_dir,
        lr_summary_dir,
        run_verification,
        b1_prestability_dir,
        b1_verification,
    )
    observed = load_json_strict(artifact_dir / PTC500_REPORT_BASENAME)
    if observed != expected:
        raise Stage1ArtifactError("PTC500 decision differs from run bytes and frozen gates")
    return observed


def build_b1_full_closure(
    ptc500_artifact_dir: Path,
    b1_prestability_dir: Path,
    b1_verification: Mapping[str, Any],
) -> Dict[str, Any]:
    """Combine prestability B1.1--B1.9 and verified B1.11 into one gate."""

    verify_simple_seal(
        ptc500_artifact_dir,
        seal_name="artifact_seal.json",
        schema_version=PTC500_SEAL_SCHEMA,
        status="complete",
        payload_names=(PTC500_REPORT_BASENAME,),
    )
    ptc500 = load_json_strict(ptc500_artifact_dir / PTC500_REPORT_BASENAME)
    _exact_fields(
        ptc500,
        (
            "schema_version",
            "scientific_status",
            "gate_passed",
            "b1_11_gate_passed",
            "full_b1_passed",
            "selected_learning_rate",
            "training_lineage_anchor",
            "b1_prestability",
            "lr_decision",
            "training_run",
            "observations",
            "gates",
            "required_action",
        ),
        "PTC500 B1.11 report",
    )
    b1_identity = _verify_b1_prestability_identity(
        b1_prestability_dir, b1_verification
    )
    if (
        ptc500.get("schema_version") != PTC500_REPORT_SCHEMA
        or ptc500.get("scientific_status") != "ptc500_stability_passed"
        or ptc500.get("gate_passed") is not True
        or ptc500.get("b1_11_gate_passed") is not True
        or ptc500.get("full_b1_passed") is not False
        or ptc500.get("required_action") != "publish_combined_b1_closure"
        or ptc500.get("b1_prestability") != b1_identity
    ):
        raise Stage1ArtifactError("PTC500 cannot close full B1")
    lineage = _validate_training_lineage_anchor(
        ptc500.get("training_lineage_anchor")
    )
    return {
        "schema_version": B1_FULL_CLOSURE_SCHEMA,
        "scientific_status": "full_b1_passed",
        "gate_passed": True,
        "prestability_gate_passed": True,
        "b1_11_gate_passed": True,
        "full_b1_passed": True,
        "b1_prestability": b1_identity,
        "ptc500": {
            "report": artifact_member(
                ptc500_artifact_dir / PTC500_REPORT_BASENAME
            ),
            "artifact_seal": artifact_member(
                ptc500_artifact_dir / "artifact_seal.json"
            ),
            "training_run": ptc500["training_run"],
            "lr_decision": ptc500["lr_decision"],
        },
        "training_lineage_anchor": lineage,
        "completed_scope": "B1.1-B1.6+B1.9+B1.11",
        "required_action": "auto_continue_small_pilot",
    }


def publish_b1_full_closure(
    ptc500_artifact_dir: Path,
    b1_prestability_dir: Path,
    b1_verification: Mapping[str, Any],
    output_dir: Path,
) -> Path:
    return publish_closed_json_artifact(
        output_dir,
        report_name=B1_FULL_CLOSURE_BASENAME,
        report=build_b1_full_closure(
            ptc500_artifact_dir, b1_prestability_dir, b1_verification
        ),
        seal_schema=B1_FULL_CLOSURE_SEAL_SCHEMA,
        seal_status="complete",
    )


def verify_b1_full_closure(
    artifact_dir: Path,
    ptc500_artifact_dir: Path,
    b1_prestability_dir: Path,
    b1_verification: Mapping[str, Any],
) -> Dict[str, Any]:
    verify_simple_seal(
        artifact_dir,
        seal_name="artifact_seal.json",
        schema_version=B1_FULL_CLOSURE_SEAL_SCHEMA,
        status="complete",
        payload_names=(B1_FULL_CLOSURE_BASENAME,),
    )
    observed = load_json_strict(artifact_dir / B1_FULL_CLOSURE_BASENAME)
    expected = build_b1_full_closure(
        ptc500_artifact_dir, b1_prestability_dir, b1_verification
    )
    if observed != expected:
        raise Stage1ArtifactError("full B1 closure differs from sealed inputs")
    return observed


STAGE_NODES: Tuple[Dict[str, Any], ...] = (
    {"id": "t5_closure", "depends_on": [], "capabilities": [], "execution": "external_evidence"},
    {"id": "retained_seals_audit", "depends_on": ["t5_closure"], "capabilities": [], "execution": "external_evidence"},
    {"id": "pilot_eval_manifest", "depends_on": ["t5_closure", "retained_seals_audit"], "capabilities": ["pilot_eval_manifest"], "execution": "local_command"},
    {"id": "b1_prestability", "depends_on": ["t5_closure", "retained_seals_audit"], "capabilities": ["b1_prestability"], "execution": "local_command"},
    {"id": "performance_benchmark", "depends_on": ["b1_prestability"], "capabilities": ["performance_benchmark"], "execution": "local_command"},
    {"id": "evaluation_pipeline_qualification", "depends_on": ["pilot_eval_manifest", "b1_prestability"], "capabilities": ["base_teacher_generation", "quality_evaluator", "clap_evaluator"], "execution": "local_command"},
    {"id": "lr_uniform_sweep", "depends_on": ["performance_benchmark", "evaluation_pipeline_qualification"], "capabilities": ["training_runner"], "execution": "four_machine_schedule"},
    {"id": "lr_evaluation_summary", "depends_on": ["lr_uniform_sweep"], "capabilities": ["trained_checkpoint_generation", "quality_evaluator", "clap_evaluator", "lr_evaluation_summary"], "execution": "local_command"},
    {"id": "lr_decision", "depends_on": ["lr_evaluation_summary"], "capabilities": ["lr_decision"], "execution": "local_command"},
    {"id": "ptc500_training", "depends_on": ["lr_decision"], "capabilities": ["training_runner"], "execution": "single_machine_eight_gpu"},
    {"id": "ptc500_stability", "depends_on": ["ptc500_training"], "capabilities": ["ptc500_stability"], "execution": "local_command"},
    {"id": "b1_full_closure", "depends_on": ["b1_prestability", "ptc500_stability"], "capabilities": ["b1_full_closure"], "execution": "local_command"},
    {"id": "small_pilot_training", "depends_on": ["b1_full_closure"], "capabilities": ["training_runner"], "execution": "four_machine_two_wave_schedule"},
    {"id": "small_pilot_evaluation", "depends_on": ["small_pilot_training"], "capabilities": ["trained_checkpoint_generation", "quality_evaluator", "clap_evaluator", "mert_evaluator", "fad_evaluator"], "execution": "four_machine_offline_evaluation"},
    {"id": "small_pilot_summary", "depends_on": ["small_pilot_evaluation"], "capabilities": ["small_pilot_summary"], "execution": "local_command"},
    {"id": "small_pilot_decision", "depends_on": ["small_pilot_summary"], "capabilities": ["small_pilot_decision"], "execution": "local_command"},
)


def load_autonomy_contract(path: Path) -> Dict[str, Any]:
    value = load_json_strict(path)
    _exact_fields(
        value,
        (
            "schema_version",
            "primary_pilot_methods",
            "small_only_temporal_diagnostics",
            "pilot_eval_manifest",
            "learning_rate",
            "ptc500_stability",
            "small_pilot",
            "capabilities",
        ),
        "autonomy contract",
    )
    if value.get("schema_version") != CONTROLLER_CONTRACT_SCHEMA:
        raise Stage1ArtifactError("autonomy contract schema mismatch")
    if value.get("primary_pilot_methods") != list(PRIMARY_PILOT_METHODS):
        raise Stage1ArtifactError("autonomy contract primary method set/order differs")
    diagnostics = value.get("small_only_temporal_diagnostics")
    if not isinstance(diagnostics, dict) or set(diagnostics) != {"prefix50"}:
        raise Stage1ArtifactError("prefix50 diagnostic declaration is absent or ambiguous")
    prefix = diagnostics["prefix50"]
    if not isinstance(prefix, dict) or prefix.get("scope") != "small_only_temporal_diagnostic":
        raise Stage1ArtifactError("prefix50 must remain small-only temporal diagnostic")
    _exact_fields(
        prefix,
        (
            "implementation_status",
            "included_in_primary_pilot_decision",
            "may_replace_primary_method",
            "required_files",
            "scope",
        ),
        "prefix50 diagnostic",
    )
    if prefix.get("implementation_status") != "implemented":
        raise Stage1ArtifactError("prefix50 diagnostic implementation must be explicit")
    if prefix.get("included_in_primary_pilot_decision") is not False or prefix.get(
        "may_replace_primary_method"
    ) is not False:
        raise Stage1ArtifactError("prefix50 may not enter or replace the primary pilot")
    if prefix.get("required_files") != [
        "src/ptc_opd/train_utils.py",
        "src/ptc_opd/losses.py",
        "scripts/train_stage1.py",
    ]:
        raise Stage1ArtifactError("prefix50 implementation file contract differs")
    if value.get("pilot_eval_manifest") != {
        "count": PILOT_PROMPT_COUNT,
        "source_count": PILOT_SOURCE_COUNT,
        "selection_seed": PILOT_SELECTION_SEED,
        "selection_namespace": PILOT_SELECTION_NAMESPACE,
        "source_basename": PILOT_SOURCE_BASENAME,
    }:
        raise Stage1ArtifactError("pilot manifest autonomy contract differs")
    if value.get("learning_rate") != {
        "candidates": list(LR_GRID),
        "clap_guardrail_base_sd_gte": LR_CLAP_GUARDRAIL,
        "decision_checkpoint": 500,
        "method": "uniform100",
        "model": "facebook/musicgen-small",
        "optimizer_steps": 500,
        "q_dev_weights": dict(LR_Q_WEIGHTS),
        "selection": "highest_eligible_q_dev",
        "tie_break": "lower_learning_rate_only_on_exact_numeric_q_dev_equality",
        "train_seed": 2027,
    }:
        raise Stage1ArtifactError("learning-rate autonomy contract differs")
    if value.get("ptc500_stability") != {
        "checkpoint_steps": [0, 250, 500],
        "learning_rate_source": "sealed_lr_decision",
        "log_every": 1,
        "memory_margin_min": PTC500_MEMORY_MARGIN_MIN,
        "method": "ptc50",
        "optimizer_steps": 500,
        "save_every": 250,
        "train_seed": 2027,
        "world_size": 8,
    }:
        raise Stage1ArtifactError("PTC500 autonomy contract differs")
    if value.get("small_pilot") != {
        "methods": list(SMALL_PILOT_METHODS),
        "primary_methods": list(PRIMARY_PILOT_METHODS),
        "diagnostic_method": "prefix50",
        "optimizer_steps": 500,  # Ruling #6 4th addendum: 1000 → 500
        "decision_checkpoint": 500,  # Ruling #6 4th addendum: 1000 → 500
        "gates": {
            "uniform_q_dev_gt": 0.0,
            "ptc_retention_vs_uniform_gte": 0.8,
            "ptc_minus_random_q_dev_gte": -0.1,
            "ptc_mert_diversity_gt_uniform": True,
            "disagreement_q_dev_gt_prefix": True,
            "ptc_finite_metrics": True,
            "ptc_stable_optimization": True,
            "fad_pipeline_check_all_six_methods": True,
        },
    }:
        raise Stage1ArtifactError("small-pilot autonomy contract differs")
    capabilities = value.get("capabilities")
    if not isinstance(capabilities, dict):
        raise Stage1ArtifactError("autonomy capability registry is absent")
    expected_capabilities = {
        capability
        for node in STAGE_NODES
        for capability in node["capabilities"]
    }
    if set(capabilities) != expected_capabilities:
        raise Stage1ArtifactError("autonomy capability registry set differs")
    for name, spec in capabilities.items():
        if not isinstance(spec, dict):
            raise Stage1ArtifactError("capability spec must be an object")
        _exact_fields(
            spec, ("implementation_status", "required_files"), "capability {}".format(name)
        )
        if spec.get("implementation_status") not in {"implemented", "not_implemented"}:
            raise Stage1ArtifactError("capability implementation status differs")
        required_files = spec.get("required_files")
        if not isinstance(required_files, list) or len(required_files) != len(
            set(required_files)
        ):
            raise Stage1ArtifactError("capability file list is malformed")
        for relative in required_files:
            if not isinstance(relative, str):
                raise Stage1ArtifactError("capability file path must be a string")
            parsed = PurePosixPath(relative)
            if (
                parsed.is_absolute()
                or parsed.as_posix() != relative
                or any(part in {"", ".", ".."} for part in parsed.parts)
            ):
                raise Stage1ArtifactError("capability file path is unsafe")
        if spec["implementation_status"] == "implemented" and not required_files:
            raise Stage1ArtifactError("implemented capability must name its files")
        if spec["implementation_status"] == "not_implemented" and required_files:
            raise Stage1ArtifactError("unimplemented capability may not claim files")
    return value


def _mode_spec_keys(train_utils_path: Path) -> List[str]:
    try:
        tree = ast.parse(train_utils_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise Stage1ArtifactError("cannot parse train_utils.py") from exc
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "MODE_SPECS" for target in targets):
                value = node.value
                if not isinstance(value, ast.Dict):
                    raise Stage1ArtifactError("MODE_SPECS is not a literal mapping")
                keys = []
                for key in value.keys:
                    if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                        raise Stage1ArtifactError("MODE_SPECS has a non-literal key")
                    keys.append(key.value)
                return keys
    raise Stage1ArtifactError("MODE_SPECS was not found")


def _validate_stage_graph(contract: Mapping[str, Any]) -> None:
    """Require a unique, dependency-ordered DAG with declared capabilities."""

    capabilities = contract.get("capabilities")
    if not isinstance(capabilities, dict):
        raise Stage1ArtifactError("controller capability registry is absent")
    seen = set()
    for node in STAGE_NODES:
        if not isinstance(node, dict) or set(node) != {
            "id",
            "depends_on",
            "capabilities",
            "execution",
        }:
            raise Stage1ArtifactError("controller stage node fields differ")
        node_id = _require_text(node.get("id"), "controller stage id")
        if node_id in seen:
            raise Stage1ArtifactError("controller stage ids must be unique")
        dependencies = node.get("depends_on")
        if not isinstance(dependencies, list) or len(dependencies) != len(
            set(dependencies)
        ):
            raise Stage1ArtifactError("controller stage dependencies are malformed")
        if any(dependency not in seen for dependency in dependencies):
            raise Stage1ArtifactError(
                "controller DAG dependency is missing, cyclic, or out of order"
            )
        required_capabilities = node.get("capabilities")
        if not isinstance(required_capabilities, list) or len(
            required_capabilities
        ) != len(set(required_capabilities)):
            raise Stage1ArtifactError("controller stage capabilities are malformed")
        if any(name not in capabilities for name in required_capabilities):
            raise Stage1ArtifactError("controller stage names undeclared capability")
        if node.get("execution") not in {
            "external_evidence",
            "local_command",
            "four_machine_schedule",
            "single_machine_eight_gpu",
            "four_machine_two_wave_schedule",
            "four_machine_offline_evaluation",
        }:
            raise Stage1ArtifactError("controller stage execution class differs")
        seen.add(node_id)


def _controller_verifier_catalog() -> Dict[str, Dict[str, Any]]:
    """Load the receipt registry lazily and require exact 16-node coverage."""

    # The ledger module imports STAGE_NODES lazily to avoid a module cycle.
    from .stage1_controller_ledger import stage_verifier_catalog

    catalog = stage_verifier_catalog()
    graph = {
        node["id"]: list(node["depends_on"])
        for node in STAGE_NODES
    }
    if set(catalog) != set(graph):
        raise Stage1ArtifactError("controller verifier registry stage set differs")
    for stage_id, dependencies in graph.items():
        item = catalog[stage_id]
        if (
            not isinstance(item, dict)
            or item.get("supported") is not True
            or item.get("dependencies") != dependencies
            or not isinstance(item.get("verifier_id"), str)
            or not item["verifier_id"]
            or not isinstance(item.get("evidence"), dict)
        ):
            raise Stage1ArtifactError(
                "controller verifier registry is incomplete at {}".format(stage_id)
            )
    return catalog


def controller_plan(contract: Mapping[str, Any]) -> Dict[str, Any]:
    _validate_stage_graph(contract)
    verifier_catalog = _controller_verifier_catalog()
    return {
        "schema_version": CONTROLLER_PLAN_SCHEMA,
        "primary_pilot_methods": list(PRIMARY_PILOT_METHODS),
        "primary_pilot_job_count": 5,
        "small_only_temporal_diagnostics": contract["small_only_temporal_diagnostics"],
        "nodes": [dict(node) for node in STAGE_NODES],
        "stage_verifiers": verifier_catalog,
        "four_machine_allocation": {
            "lr_uniform_sweep": {
                "node-0": "uniform100/lr-1e-6/seed-2027/500",
                "node-1": "uniform100/lr-3e-6/seed-2027/500",
                "node-2": "uniform100/lr-1e-5/seed-2027/500",
                "node-3": "base-generation-and-evaluation-or-spare",
            },
            "small_pilot_wave_1": {
                "node-0": "uniform100/seed-2027/1000",
                "node-1": "codebook100/seed-2027/1000",
                "node-2": "random50/seed-2027/1000",
                "node-3": "disagreement50/seed-2027/1000",
            },
            "small_pilot_wave_2": {
                "node-0": "ptc50/seed-2027/1000",
                "node-1": "prefix50/seed-2027/1000-small-only-diagnostic",
                "node-2": "offline-evaluation-only",
                "node-3": "offline-evaluation-only",
            },
        },
    }


def controller_preflight(workpack_root: Path, contract_path: Path) -> Dict[str, Any]:
    supplied_root = workpack_root.expanduser().absolute()
    if supplied_root.is_symlink():
        raise Stage1ArtifactError("controller workpack root must not be a symlink")
    workpack_root = supplied_root.resolve(strict=True)
    if not workpack_root.is_dir():
        raise Stage1ArtifactError("controller workpack root must be a directory")

    # Capability required_files are audited entrypoint anchors, not a hand-
    # maintained transitive import list.  Full runtime-source completeness is
    # instead enforced by the closed-world workpack verifier here, so removing
    # an indirect dependency cannot leave preflight reporting `ready`.
    seal_script = workpack_root / "scripts" / "seal_workpack.py"
    if seal_script.is_symlink() or not seal_script.is_file():
        raise Stage1ArtifactError("workpack seal verifier is missing or a symlink")
    spec = importlib.util.spec_from_file_location(
        "ptc_opd_stage1_seal_workpack_preflight", seal_script
    )
    if spec is None or spec.loader is None:
        raise Stage1ArtifactError("cannot load workpack seal verifier")
    seal_module = importlib.util.module_from_spec(spec)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(seal_module)
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
    try:
        managed_records = seal_module.verify(workpack_root)
    except Exception as exc:
        raise Stage1ArtifactError(
            "workpack source-integrity verification failed: {}".format(exc)
        ) from exc
    manifest_path = workpack_root / "WORKPACK_MANIFEST.sha256"
    workpack_source_integrity = {
        "status": "verified_closed_world",
        "managed_file_count": len(managed_records),
        "manifest_sha256": sha256_file(manifest_path),
    }
    contract = load_autonomy_contract(contract_path)
    verifier_catalog = _controller_verifier_catalog()
    capabilities_result: Dict[str, Any] = {}
    for name, spec in sorted(contract["capabilities"].items()):
        if not isinstance(spec, dict):
            raise Stage1ArtifactError("capability spec must be an object")
        status = spec.get("implementation_status")
        if status not in {"implemented", "not_implemented"}:
            raise Stage1ArtifactError("capability implementation status differs")
        required_files = spec.get("required_files")
        if not isinstance(required_files, list) or any(
            not isinstance(path, str) or not path for path in required_files
        ):
            raise Stage1ArtifactError("capability required_files is malformed")
        observed = {
            path: (workpack_root / path).is_file() and not (workpack_root / path).is_symlink()
            for path in required_files
        }
        ready = status == "implemented" and bool(required_files) and all(observed.values())
        capabilities_result[name] = {
            "declared_status": status,
            "ready": ready,
            "required_files": observed,
        }
    mode_keys = _mode_spec_keys(workpack_root / "src" / "ptc_opd" / "train_utils.py")
    registered_modes_exact = mode_keys == list(REGISTERED_STAGE1_MODES)
    primary_methods_exact = [
        name for name in mode_keys if name != "prefix50"
    ] == list(PRIMARY_PILOT_METHODS)
    lr_required = (
        "b1_prestability",
        "performance_benchmark",
        "pilot_eval_manifest",
        "base_teacher_generation",
        "trained_checkpoint_generation",
        "quality_evaluator",
        "clap_evaluator",
        "lr_evaluation_summary",
        "lr_decision",
        "training_runner",
    )
    safe_capabilities = registered_modes_exact and primary_methods_exact and all(
        capabilities_result[name]["ready"] for name in lr_required
    )
    return {
        "schema_version": CONTROLLER_PREFLIGHT_SCHEMA,
        "status": (
            "passed_control_plane"
            if registered_modes_exact and primary_methods_exact
            else "failed_control_plane"
        ),
        "contract_sha256": sha256_file(contract_path),
        "workpack_source_integrity": workpack_source_integrity,
        "registered_modes_exact": registered_modes_exact,
        "primary_methods_exact": primary_methods_exact,
        "observed_mode_specs": mode_keys,
        "prefix50": {
            "scope": "small_only_temporal_diagnostic",
            "registered_in_training_runner": "prefix50" in mode_keys,
            "included_in_primary_pilot_decision": False,
        },
        "capabilities": capabilities_result,
        "stage_verifier_count": len(verifier_catalog),
        "all_stage_verifiers_supported": all(
            item["supported"] is True for item in verifier_catalog.values()
        ),
        "stage_verifiers": verifier_catalog,
        "safe_to_launch_lr_by_capability": safe_capabilities,
        "cfg_only_evaluators_count_as_trained_evaluators": False,
        "required_action": (
            "await_missing_frozen_capabilities"
            if not safe_capabilities
            else "proceed_only_after_stage_evidence_passes"
        ),
    }


def controller_next_action(
    contract: Mapping[str, Any],
    preflight: Mapping[str, Any],
    ledger_dir: Path,
    *,
    workpack_root: Path,
    contract_path: Path,
) -> Dict[str, Any]:
    """Choose work only from a live-reverified sealed ledger revision."""

    _validate_stage_graph(contract)
    verifier_catalog = _controller_verifier_catalog()
    from .stage1_controller_ledger import (
        validate_current_controller_ledger_authorizations,
    )

    validated = validate_current_controller_ledger_authorizations(
        ledger_dir,
        workpack_root=workpack_root,
        contract_path=contract_path,
    )
    ledger = validated["ledger"]
    stages = validated["legacy_stages"]
    if ledger.get("scientific_config_sha256") != canonical_json_sha256(contract):
        raise Stage1ArtifactError("controller ledger/contract canonical identity differs")
    source_integrity = preflight.get("workpack_source_integrity")
    if (
        preflight.get("schema_version") != CONTROLLER_PREFLIGHT_SCHEMA
        or preflight.get("contract_sha256") != sha256_file(contract_path)
        or not isinstance(source_integrity, dict)
        or source_integrity.get("status") != "verified_closed_world"
        or source_integrity.get("manifest_sha256")
        != ledger.get("workpack_manifest_sha256")
        or preflight.get("all_stage_verifiers_supported") is not True
        or preflight.get("stage_verifier_count") != len(STAGE_NODES)
        or preflight.get("stage_verifiers") != verifier_catalog
    ):
        raise Stage1ArtifactError("controller preflight/ledger authority binding differs")
    stage_records = ledger["stages"]
    exhausted = [
        stage_id
        for stage_id, entry in stage_records.items()
        if entry.get("stop_reason") == "yellow_retry_budget_exhausted"
    ]
    stopped = [
        node_id
        for node_id, entry in stages.items()
        if entry["status"] in {"failed", "inconclusive"}
    ]
    if stopped:
        return {
            "schema_version": CONTROLLER_ACTION_SCHEMA,
            "required_action": (
                "stop_yellow_retry_budget_exhausted"
                if exhausted
                else "stop_failed_or_inconclusive_stage"
            ),
            "stopped_stages": stopped,
            "yellow_budget_exhausted_stages": exhausted,
            "runnable_stages": [],
            "missing_capability_stages": [],
            "awaiting_external_evidence": [],
            "ledger_revision": ledger["revision"],
            "yellow_retry_budget": {
                stage_id: {
                    "used": entry["yellow_retry_count"],
                    "remaining": max(0, 2 - entry["yellow_retry_count"]),
                }
                for stage_id, entry in stage_records.items()
            },
        }
    ready_nodes = [
        node
        for node in STAGE_NODES
        if stages[node["id"]]["status"] == "pending"
        and all(stages[dependency]["status"] == "passed" for dependency in node["depends_on"])
    ]
    runnable: List[str] = []
    missing: List[Dict[str, Any]] = []
    external: List[str] = []
    for node in ready_nodes:
        if node["execution"] == "external_evidence":
            external.append(node["id"])
            continue
        unavailable = [
            capability
            for capability in node["capabilities"]
            if not preflight["capabilities"].get(capability, {}).get("ready", False)
        ]
        if unavailable:
            missing.append({"stage": node["id"], "capabilities": unavailable})
        else:
            runnable.append(node["id"])
    if runnable:
        action = "auto_continue_available_stages"
    elif missing:
        action = "stop_missing_frozen_capability"
    elif external:
        action = "await_external_evidence"
    elif all(entry["status"] == "passed" for entry in stages.values()):
        action = "stage1_small_pilot_chain_complete"
    else:
        action = "wait_for_dependencies"
    return {
        "schema_version": CONTROLLER_ACTION_SCHEMA,
        "required_action": action,
        "runnable_stages": runnable,
        "missing_capability_stages": missing,
        "awaiting_external_evidence": external,
        "stopped_stages": [],
        "yellow_budget_exhausted_stages": [],
        "ledger_revision": ledger["revision"],
        "yellow_retry_budget": {
            stage_id: {
                "used": entry["yellow_retry_count"],
                "remaining": max(0, 2 - entry["yellow_retry_count"]),
            }
            for stage_id, entry in stage_records.items()
        },
        "retrying_stages": [
            node["id"]
            for node in ready_nodes
            if stage_records[node["id"]]["attempts"]
        ],
        "primary_pilot_methods": list(PRIMARY_PILOT_METHODS),
        "prefix50_in_primary_pilot_decision": False,
    }
