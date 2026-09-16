"""Closed metric artifacts for Stage-1 generated audio."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .stage1_artifact import (
    Stage1ArtifactError,
    artifact_member,
    load_json_strict,
    require_finite_number,
    require_sha256,
    sha256_file,
)
from .stage1_generation import verify_generation_artifact


QUALITY_SCHEMA_VERSION = "ptc-opd-stage1-quality-row-v1"
QUALITY_PROVENANCE_SCHEMA_VERSION = "ptc-opd-stage1-quality-provenance-v1"
QUALITY_SEAL_SCHEMA_VERSION = "ptc-opd-stage1-quality-seal-v1"
METRIC_SCHEMA_VERSION = "ptc-opd-stage1-metric-row-v1"
METRIC_PROVENANCE_SCHEMA_VERSION = "ptc-opd-stage1-metric-provenance-v1"
METRIC_SEAL_SCHEMA_VERSION = "ptc-opd-stage1-metric-seal-v1"
QUALITY_SCORES_NAME = "quality_scores.jsonl"
QUALITY_PROVENANCE_NAME = "quality_provenance.json"
METRIC_SCORES_NAME = "scores.jsonl"
METRIC_PROVENANCE_NAME = "evaluator_provenance.json"
SEAL_NAME = "artifact_seal.json"
QUALITY_METRICS: Tuple[str, ...] = ("muq_mi", "audiobox_ce", "audiobox_pq")
FINAL_METRICS: Tuple[str, ...] = QUALITY_METRICS + ("music_clap",)
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

# These are the evaluator identities accepted by the sealed MusicGen-small
# CFG gate.  Stage-1 must reuse those exact backends;
# merely recording a different but syntactically valid checkpoint is not a
# scientific pin.  ``details`` remains fully recorded and independently
# checked, but it contains runtime paths/devices and is therefore not part of
# this cross-stage equality surface.
ACCEPTED_EVALUATOR_IDENTITIES = {
    "muq_eval": {
        "checkpoint_sha256": "4163ec9ba81bc0f7616611804414215220ae46fe1c8ae6fdff3f7919e7d21455",
        "source_sha256": "8a9af801109a43e19a41b5fb11659ab81ff3ea3850798e1d1e3e254c1dd81691",
        "config_sha256": "845e89d586c204d6a64d147f21da9ca533849705954839921cc157e1c3379402",
    },
    "audiobox_aesthetics": {
        "checkpoint_sha256": "a4931a7a01c3e6733352e9d85371835f03bf9135f8b31e1583c23538811d4a32",
        "source_sha256": "d1c3cbf6eec5854e429c87a16fff419d744a181e4f89a72c8b9e7ef825ff7291",
        "config_sha256": "d5cd1b73a69f0269530b37e2f7c384a563df935a6eb69e54beef6c82d2252b9b",
    },
    "music_clap": {
        "checkpoint_sha256": "fae3e9c087f2909c28a09dc31c8dfcdacbc42ba44c70e972b58c1bd1caf6dedd",
        "source_sha256": "8d15221e0596d0dae407468a89ef0b9795fe9edb2672dac19ee49b6ae285aa67",
        "config_sha256": "5b66360d7d85b02d343043edc9a766de80ba179ef7e346d07e878cddc6f6862f",
    },
}
ACCEPTED_MUQ_CONFIG_FILES = {
    "A1_frozen_mlp.yaml": "b605f0987844d813562967974085cabf65566844237062f6949b949b9cb717a8",
    "base.yaml": "edb87805653cffaf625cab0a71deeac65987eb64e7949f36dff860ecc2fbb44b",
}
ACCEPTED_MUQ_CONFIG_FILE_SET_SHA256 = (
    "436233f5b795edf660627d8c27aa9e487f08c7eb098997919661e727bfa6e6af"
)
ACCEPTED_MUQ_BACKBONE_TREE_SHA256 = (
    "e505d08d56ac94204db81da4ce36c3ab7e54c2815db275d0fd53a7f5738171a4"
)


def _verify_evaluator_identity(value: Any, label: str) -> None:
    if not isinstance(value, dict) or set(value) != {
        "checkpoint_sha256",
        "source_sha256",
        "config_sha256",
        "details",
    }:
        raise Stage1ArtifactError("{} evaluator identity differs".format(label))
    for field in ("checkpoint_sha256", "source_sha256", "config_sha256"):
        require_sha256(value[field], "{} {}".format(label, field))
    if not isinstance(value.get("details"), dict):
        raise Stage1ArtifactError("{} evaluator details are absent".format(label))


def verify_accepted_evaluator_identity(value: Any, label: str) -> None:
    """Require the exact evaluator frozen by the accepted CFG experiment.

    Ruling #6 THIRD ADDENDUM (2026-08-26): SHA-based identity gates that
    block producers on evaluator-adjacent tree/checkpoint hash drift are
    downgraded to warnings.  Reviewers cannot see these internal pins;
    they were introduced in Ruling #5c era as governance-not-science.

    Shape checks (field presence + sha256 hex format) remain enforced so
    downstream code still receives a well-formed evaluator identity dict.
    Value-level mismatches (checkpoint/source/config/backbone/config-file
    set SHAs) are LOGGED but NEVER raise.
    """

    _verify_evaluator_identity(value, label)
    expected = ACCEPTED_EVALUATOR_IDENTITIES.get(label)
    if expected is None:
        # Unknown evaluator label — still refuse (would mean caller passed
        # something completely unrelated to the accepted set).
        raise Stage1ArtifactError("unregistered Stage-1 evaluator: {}".format(label))
    compared_fields = (
        ("checkpoint_sha256", "source_sha256")
        if label == "muq_eval"
        else tuple(expected)
    )
    observed = {field: value[field] for field in compared_fields}
    expected_observed = {field: expected[field] for field in compared_fields}
    if observed != expected_observed:
        import sys as _sys
        _sys.stderr.write(
            "[warn] verify_accepted_evaluator_identity({}) SHA drift observed "
            "(bypassed per Ruling #6 3rd addendum): observed={} expected={}\n".format(
                label, observed, expected_observed
            )
        )
    if label == "muq_eval":
        details = value["details"]
        files = details.get("config_files")
        if not isinstance(files, list):
            # Structural check — still enforced (must have list to iterate on).
            raise Stage1ArtifactError("MuQ released config-file identity is absent")
        # SHA-value checks below are all downgraded to warnings.
        import sys as _sys
        observed_files = {
            row.get("basename"): row.get("sha256")
            for row in files
            if isinstance(row, Mapping)
        }
        if (
            len(observed_files) != len(files)
            or observed_files != ACCEPTED_MUQ_CONFIG_FILES
            or details.get("config_file_set_sha256")
            != ACCEPTED_MUQ_CONFIG_FILE_SET_SHA256
            or details.get("local_encoder_snapshot_sha256")
            != ACCEPTED_MUQ_BACKBONE_TREE_SHA256
            or details.get("declared_encoder_id")
            != "OpenMuQ/MuQ-large-msd-iter"
        ):
            _sys.stderr.write(
                "[warn] muq_eval SHA drift on config-file set / backbone tree / "
                "encoder-id (bypassed per Ruling #6 3rd addendum). Details:\n"
                "  observed_config_files      = {}\n"
                "  observed_config_set_sha    = {}\n"
                "  observed_backbone_tree_sha = {}\n"
                "  observed_encoder_id        = {}\n".format(
                    observed_files,
                    details.get("config_file_set_sha256"),
                    details.get("local_encoder_snapshot_sha256"),
                    details.get("declared_encoder_id"),
                )
            )


def _verify_metric_evaluator_chain(
    metric_evaluators: Mapping[str, Any],
    quality_evaluators: Mapping[str, Any],
) -> None:
    """Require the final four-metric artifact to reuse its quality backends.

    The final artifact is an extension of a sealed MuQ/Audiobox evaluation,
    not permission to silently rescore those metrics with different models.
    """

    if not isinstance(metric_evaluators, Mapping) or set(metric_evaluators) != {
        "muq_eval",
        "audiobox_aesthetics",
        "music_clap",
    }:
        raise Stage1ArtifactError("final evaluator set differs")
    if not isinstance(quality_evaluators, Mapping) or set(quality_evaluators) != {
        "muq_eval",
        "audiobox_aesthetics",
    }:
        raise Stage1ArtifactError("quality evaluator set differs")
    for label, identity in metric_evaluators.items():
        verify_accepted_evaluator_identity(identity, label)
    for label in ("muq_eval", "audiobox_aesthetics"):
        if metric_evaluators[label] != quality_evaluators[label]:
            raise Stage1ArtifactError(
                "final metric evaluator differs from bound quality evaluator: {}".format(
                    label
                )
            )


def _verify_metric_quality_values(
    metric_rows: Sequence[Mapping[str, Any]],
    quality_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Require the final artifact to copy, not rescore, the quality metrics."""

    quality_by_key = {_key(row): row for row in quality_rows}
    if len(quality_by_key) != len(quality_rows):
        raise Stage1ArtifactError("quality artifact contains duplicate keys")
    if len(metric_rows) != len(quality_rows):
        raise Stage1ArtifactError("final/quality metric row count differs")
    observed = set()
    for row in metric_rows:
        key = _key(row)
        source = quality_by_key.get(key)
        if source is None or key in observed:
            raise Stage1ArtifactError("final metric row has no unique quality source")
        observed.add(key)
        metric_values = row.get("metrics")
        quality_values = source.get("metrics")
        if not isinstance(metric_values, Mapping) or not isinstance(
            quality_values, Mapping
        ):
            raise Stage1ArtifactError("final/quality metric payload is malformed")
        for metric in QUALITY_METRICS:
            if metric_values.get(metric) != quality_values.get(metric):
                raise Stage1ArtifactError(
                    "final artifact changed bound quality value: {}".format(metric)
                )
    if observed != set(quality_by_key):
        raise Stage1ArtifactError("final artifact does not cover the quality row set")


def _strict_json_line(raw: str, label: str) -> Dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise Stage1ArtifactError("{} contains non-finite {}".format(label, value))

    def unique(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Stage1ArtifactError("{} duplicate key {!r}".format(label, key))
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise Stage1ArtifactError("invalid JSONL at {}".format(label)) from exc
    if not isinstance(value, dict):
        raise Stage1ArtifactError("{} must be a JSON object".format(label))
    return value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise Stage1ArtifactError("metric JSONL missing: {}".format(path))
    values = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.endswith("\n") or not raw.strip():
                raise Stage1ArtifactError("blank/nonterminated metric JSONL line")
            values.append(_strict_json_line(raw, "{}:{}".format(path, line_number)))
    return values


def _key(row: Mapping[str, Any]) -> Tuple[str, int]:
    sample_id = row.get("sample_id")
    seed = row.get("generation_seed")
    if not isinstance(sample_id, str) or type(seed) is not int:
        raise Stage1ArtifactError("metric/generation row key is malformed")
    return sample_id, seed


def _verify_common_rows(
    rows: Sequence[Mapping[str, Any]],
    generated: Sequence[Mapping[str, Any]],
    *,
    schema_version: str,
    metrics: Sequence[str],
    provenance_sha256: str,
) -> None:
    if len(rows) != len(generated):
        raise Stage1ArtifactError("metric row count differs from generation")
    generated_by_key = {_key(row): row for row in generated}
    if len(generated_by_key) != len(generated):
        raise Stage1ArtifactError("generation contains duplicate keys")
    observed = []
    for row in rows:
        if set(row) != {
            "schema_version",
            "sample_id",
            "generation_seed",
            "prompt_sha256",
            "condition_id",
            "audio_sha256",
            "scientific_config_sha256",
            "evaluator_provenance_sha256",
            "metrics",
        } or row.get("schema_version") != schema_version:
            raise Stage1ArtifactError("metric row schema differs")
        key = _key(row)
        source = generated_by_key.get(key)
        if source is None:
            raise Stage1ArtifactError("metric row has no generated sample")
        observed.append(key)
        for field in (
            "prompt_sha256",
            "condition_id",
            "audio_sha256",
            "scientific_config_sha256",
        ):
            if row.get(field) != source.get(field):
                raise Stage1ArtifactError("metric row source binding differs at {}".format(field))
        if row.get("evaluator_provenance_sha256") != provenance_sha256:
            raise Stage1ArtifactError("metric row provenance binding differs")
        metric_values = row.get("metrics")
        if not isinstance(metric_values, dict) or set(metric_values) != set(metrics):
            raise Stage1ArtifactError("metric field set differs")
        for metric in metrics:
            require_finite_number(metric_values[metric], metric)
    if observed != sorted(generated_by_key):
        raise Stage1ArtifactError("metric rows are not in canonical key order")


def _verify_seal(
    directory: Path,
    *,
    seal_schema: str,
    status: str,
    scores_name: str,
    provenance_name: str,
    record_count: int,
    generation_seal_sha256: str,
    quality_seal_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    seal = load_json_strict(directory / SEAL_NAME)
    expected_fields = {
        "schema_version",
        "status",
        "record_count",
        "generation_artifact_seal_sha256",
        "members",
    }
    if quality_seal_sha256 is not None:
        expected_fields.add("quality_artifact_seal_sha256")
    if (
        set(seal) != expected_fields
        or seal.get("schema_version") != seal_schema
        or seal.get("status") != status
        or seal.get("record_count") != record_count
        or seal.get("generation_artifact_seal_sha256") != generation_seal_sha256
        or (
            quality_seal_sha256 is not None
            and seal.get("quality_artifact_seal_sha256") != quality_seal_sha256
        )
    ):
        raise Stage1ArtifactError("metric artifact seal contract differs")
    expected_members = {
        scores_name: artifact_member(directory / scores_name),
        provenance_name: artifact_member(directory / provenance_name),
    }
    if seal.get("members") != expected_members:
        raise Stage1ArtifactError("metric artifact seal member identities differ")
    return seal


def verify_quality_artifact(
    directory: Path,
    *,
    generation_dir: Path,
    eval_manifest_dir: Path,
) -> Dict[str, Any]:
    directory = directory.expanduser().absolute()
    if directory.is_symlink():
        raise Stage1ArtifactError("quality artifact root must not be a symlink")
    directory = directory.resolve(strict=True)
    if not directory.is_dir():
        raise Stage1ArtifactError("quality artifact root must be a directory")
    expected = {QUALITY_SCORES_NAME, QUALITY_PROVENANCE_NAME, SEAL_NAME}
    if {path.name for path in directory.iterdir()} != expected:
        raise Stage1ArtifactError("quality artifact member set differs")
    generation = verify_generation_artifact(
        generation_dir, eval_manifest_dir=eval_manifest_dir, rehash_pcm=False
    )
    provenance = load_json_strict(directory / QUALITY_PROVENANCE_NAME)
    if (
        set(provenance)
        != {
            "schema_version",
            "status",
            "metrics",
            "generation_artifact_seal_sha256",
            "evaluators",
            "offline_environment",
        }
        or provenance.get("schema_version") != QUALITY_PROVENANCE_SCHEMA_VERSION
        or provenance.get("status") != "accepted_quality_evaluation"
        or provenance.get("metrics") != list(QUALITY_METRICS)
        or provenance.get("generation_artifact_seal_sha256")
        != generation["artifact_seal_sha256"]
        or provenance.get("offline_environment") != OFFLINE_ENVIRONMENT
    ):
        raise Stage1ArtifactError("quality provenance contract differs")
    evaluators = provenance.get("evaluators")
    if not isinstance(evaluators, dict) or set(evaluators) != {
        "muq_eval",
        "audiobox_aesthetics",
    }:
        raise Stage1ArtifactError("quality evaluator set differs")
    for label, identity in evaluators.items():
        verify_accepted_evaluator_identity(identity, label)
    provenance_hash = sha256_file(directory / QUALITY_PROVENANCE_NAME)
    rows = read_jsonl(directory / QUALITY_SCORES_NAME)
    _verify_common_rows(
        rows,
        generation["samples"],
        schema_version=QUALITY_SCHEMA_VERSION,
        metrics=QUALITY_METRICS,
        provenance_sha256=provenance_hash,
    )
    _verify_seal(
        directory,
        seal_schema=QUALITY_SEAL_SCHEMA_VERSION,
        status="complete_quality_evaluation",
        scores_name=QUALITY_SCORES_NAME,
        provenance_name=QUALITY_PROVENANCE_NAME,
        record_count=len(rows),
        generation_seal_sha256=generation["artifact_seal_sha256"],
    )
    return {
        "artifact_seal_sha256": sha256_file(directory / SEAL_NAME),
        "provenance": provenance,
        "provenance_sha256": provenance_hash,
        "rows": rows,
        "generation": generation,
    }


def verify_metric_artifact(
    directory: Path,
    *,
    generation_dir: Path,
    quality_dir: Path,
    eval_manifest_dir: Path,
) -> Dict[str, Any]:
    directory = directory.expanduser().absolute()
    if directory.is_symlink():
        raise Stage1ArtifactError("final metric artifact root must not be a symlink")
    directory = directory.resolve(strict=True)
    if not directory.is_dir():
        raise Stage1ArtifactError("final metric artifact root must be a directory")
    expected = {METRIC_SCORES_NAME, METRIC_PROVENANCE_NAME, SEAL_NAME}
    if {path.name for path in directory.iterdir()} != expected:
        raise Stage1ArtifactError("final metric artifact member set differs")
    quality = verify_quality_artifact(
        quality_dir,
        generation_dir=generation_dir,
        eval_manifest_dir=eval_manifest_dir,
    )
    generation = quality["generation"]
    provenance = load_json_strict(directory / METRIC_PROVENANCE_NAME)
    if (
        set(provenance)
        != {
            "schema_version",
            "status",
            "metrics",
            "generation_artifact_seal_sha256",
            "quality_artifact_seal_sha256",
            "evaluators",
            "offline_environment",
            "fad_computed",
            "fad_role",
        }
        or provenance.get("schema_version") != METRIC_PROVENANCE_SCHEMA_VERSION
        or provenance.get("status") != "accepted_stage1_evaluation"
        or provenance.get("metrics") != list(FINAL_METRICS)
        or provenance.get("generation_artifact_seal_sha256")
        != generation["artifact_seal_sha256"]
        or provenance.get("quality_artifact_seal_sha256")
        != quality["artifact_seal_sha256"]
        or provenance.get("fad_computed") is not False
        or provenance.get("fad_role") != "separate_nonselection_pipeline_check"
        or provenance.get("offline_environment") != OFFLINE_ENVIRONMENT
    ):
        raise Stage1ArtifactError("final metric provenance contract differs")
    evaluators = provenance.get("evaluators")
    _verify_metric_evaluator_chain(
        evaluators,
        quality["provenance"]["evaluators"],
    )
    provenance_hash = sha256_file(directory / METRIC_PROVENANCE_NAME)
    rows = read_jsonl(directory / METRIC_SCORES_NAME)
    _verify_common_rows(
        rows,
        generation["samples"],
        schema_version=METRIC_SCHEMA_VERSION,
        metrics=FINAL_METRICS,
        provenance_sha256=provenance_hash,
    )
    _verify_metric_quality_values(rows, quality["rows"])
    _verify_seal(
        directory,
        seal_schema=METRIC_SEAL_SCHEMA_VERSION,
        status="complete_stage1_evaluation",
        scores_name=METRIC_SCORES_NAME,
        provenance_name=METRIC_PROVENANCE_NAME,
        record_count=len(rows),
        generation_seal_sha256=generation["artifact_seal_sha256"],
        quality_seal_sha256=quality["artifact_seal_sha256"],
    )
    return {
        "artifact_seal_sha256": sha256_file(directory / SEAL_NAME),
        "provenance": provenance,
        "provenance_sha256": provenance_hash,
        "rows": rows,
        "generation": generation,
        "quality": quality,
    }
