"""Verifier-backed controller receipts and a fail-closed Stage-1 ledger.

The legacy controller ledger accepted ``passed`` plus any syntactically valid
SHA-256.  This module deliberately accepts neither an operator-supplied status
nor an operator-supplied evidence digest.  A stage attempt is represented by a
closed receipt whose registered verifier is rerun against live, rehashed
evidence every time the receipt or ledger is consumed.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .stage1_artifact import (
    Stage1ArtifactError,
    artifact_member,
    canonical_json_bytes,
    canonical_json_sha256,
    load_json_strict,
    publish_closed_json_artifact,
    regular_tree_files,
    require_sha256,
    sha256_file,
    sha256_tree,
    verify_simple_seal,
)


RECEIPT_SCHEMA = "ptc-opd-stage1-controller-verifier-receipt-v1"
RECEIPT_SEAL_SCHEMA = "ptc-opd-stage1-controller-verifier-receipt-seal-v1"
RECEIPT_BASENAME = "verifier_receipt.json"
LEDGER_SCHEMA = "ptc-opd-stage1-controller-authority-ledger-v1"
LEDGER_SEAL_SCHEMA = "ptc-opd-stage1-controller-authority-ledger-seal-v1"
LEDGER_BASENAME = "controller_ledger.json"
AUTHORITY_BASENAME = "CONTROLLER_AUTHORITY.json"
AUTHORITY_SCHEMA = "ptc-opd-stage1-controller-current-authority-v1"
WORKPACK_MANIFEST_BASENAME = "WORKPACK_MANIFEST.sha256"
CONTRACT_RELATIVE_PATH = "configs/stage1_autonomy_contract.json"
MAX_YELLOW_RETRIES_PER_STAGE = 2
TERMINAL_STAGE_STATUSES = frozenset({"passed", "failed", "inconclusive"})

# These are the five retained data identities accepted before Stage-1.  Merely
# naming a different file ``train.full.jsonl`` (or one of the other basenames)
# must never create a new scientific branch under the same autonomy contract.
FROZEN_RETAINED_FILE_SHA256 = {
    "train_manifest": "756aeed40eda04adca27148f36688c5ce884723e6ad23bc53c68c39502f5da25",
    "dev_manifest": "63ecd2a5267efcd8d8c2cd3f64a27cf60431d8aee2d78ab8bf393b4ecdbcb82a",
    "probe_manifest": "556a92181090e8c9e1dee3948b2811c8f590903f980508724eb29c4edbcc634f",
    "a1_manifest": "f88578a57592b6bf1592f9b0710e9dfe93e287cf157598631ba250c942447d25",
    "a1_report": "3cc1c581e9367cde1e8a740b0680fb33cb0665e75a5f68c50597cda6256de48d",
}

TRAINING_STAGE_RUN_LABELS = {
    "lr_uniform_sweep": ("run_1e6", "run_3e6", "run_1e5"),
    "ptc500_training": ("run_dir",),
    "small_pilot_training": tuple(
        method + "_run_dir"
        for method in (
            "uniform100", "codebook100", "random50", "prefix50",
            "disagreement50", "ptc50",
        )
    ),
}

# Every tuple is (ancestor stage, ancestor evidence label).  These are exact
# identity bindings, not basename checks: a scientifically valid but different
# artifact is rejected when it was not the artifact authorized upstream.
STAGE_EVIDENCE_BINDINGS: Dict[str, Dict[str, Tuple[str, str]]] = {
    "retained_seals_audit": {
        "t5_closure_dir": ("t5_closure", "closure_dir"),
    },
    "pilot_eval_manifest": {
        "source_dev_manifest": ("retained_seals_audit", "dev_manifest"),
    },
    "b1_prestability": {
        name: ("retained_seals_audit", name)
        for name in (
            "probe_manifest", "dev_manifest", "small_cfg_dir", "a1_dir",
            "a2_dir", "node3_dir", "t5_closure_dir", "audiocraft_dir",
            "musicgen_small_dir",
        )
    },
    "performance_benchmark": {
        name: ("retained_seals_audit", name)
        for name in (
            "train_manifest",
            "small_cfg_dir",
            "audiocraft_dir",
            "musicgen_small_dir",
        )
    },
    "evaluation_pipeline_qualification": {
        "eval_manifest_dir": ("pilot_eval_manifest", "artifact_dir"),
        "musicgen_small_dir": ("retained_seals_audit", "musicgen_small_dir"),
        "audiocraft_dir": ("retained_seals_audit", "audiocraft_dir"),
        "small_cfg_dir": ("retained_seals_audit", "small_cfg_dir"),
    },
    "lr_evaluation_summary": {
        "eval_manifest_dir": ("pilot_eval_manifest", "artifact_dir"),
        "base_generation_dir": (
            "evaluation_pipeline_qualification", "base_generation_dir"
        ),
        "base_quality_dir": (
            "evaluation_pipeline_qualification", "base_quality_dir"
        ),
        "base_metric_dir": (
            "evaluation_pipeline_qualification", "base_metric_dir"
        ),
        "candidate_1e6_run_dir": ("lr_uniform_sweep", "run_1e6"),
        "candidate_3e6_run_dir": ("lr_uniform_sweep", "run_3e6"),
        "candidate_1e5_run_dir": ("lr_uniform_sweep", "run_1e5"),
    },
    "lr_decision": {
        "summary_dir": ("lr_evaluation_summary", "summary_dir"),
    },
    "ptc500_training": {
        "lr_summary_dir": ("lr_evaluation_summary", "summary_dir"),
        "lr_decision_dir": ("lr_decision", "decision_dir"),
    },
    "ptc500_stability": {
        "run_dir": ("ptc500_training", "run_dir"),
        "lr_summary_dir": ("ptc500_training", "lr_summary_dir"),
        "lr_decision_dir": ("ptc500_training", "lr_decision_dir"),
        "b1_dir": ("b1_prestability", "artifact_dir"),
    },
    "b1_full_closure": {
        "ptc500_dir": ("ptc500_stability", "artifact_dir"),
        "b1_dir": ("b1_prestability", "artifact_dir"),
    },
    "small_pilot_training": {
        "b1_full_dir": ("b1_full_closure", "artifact_dir"),
        "lr_summary_dir": ("lr_evaluation_summary", "summary_dir"),
        "lr_decision_dir": ("lr_decision", "decision_dir"),
    },
    "small_pilot_evaluation": {
        "eval_manifest_dir": ("pilot_eval_manifest", "artifact_dir"),
        "base_generation_dir": (
            "evaluation_pipeline_qualification", "base_generation_dir"
        ),
        "base_quality_dir": (
            "evaluation_pipeline_qualification", "base_quality_dir"
        ),
        "base_metric_dir": (
            "evaluation_pipeline_qualification", "base_metric_dir"
        ),
        "a1_manifest": ("retained_seals_audit", "a1_manifest"),
        "a1_report": ("retained_seals_audit", "a1_report"),
        **{
            method + "_run_dir": ("small_pilot_training", method + "_run_dir")
            for method in (
                "uniform100", "codebook100", "random50", "prefix50",
                "disagreement50", "ptc50",
            )
        },
    },
    "small_pilot_summary": {
        "eval_manifest_dir": ("small_pilot_evaluation", "eval_manifest_dir"),
        "lr_summary_dir": ("lr_evaluation_summary", "summary_dir"),
        "lr_decision_dir": ("lr_decision", "decision_dir"),
        "base_generation_dir": (
            "small_pilot_evaluation", "base_generation_dir"
        ),
        "base_quality_dir": ("small_pilot_evaluation", "base_quality_dir"),
        "base_metric_dir": ("small_pilot_evaluation", "base_metric_dir"),
        "reference_dir": ("small_pilot_evaluation", "reference_dir"),
        "a1_manifest": ("small_pilot_evaluation", "a1_manifest"),
        "a1_report": ("small_pilot_evaluation", "a1_report"),
        "model_pins_dir": ("small_pilot_evaluation", "model_pins_dir"),
        **{
            "{}_{}_dir".format(method, kind): (
                "small_pilot_evaluation" if kind != "run" else "small_pilot_training",
                "{}_{}_dir".format(method, kind),
            )
            for method in (
                "uniform100", "codebook100", "random50", "prefix50",
                "disagreement50", "ptc50",
            )
            for kind in ("run", "generation", "quality", "metric", "diversity")
        },
    },
    "small_pilot_decision": {
        "summary_dir": ("small_pilot_summary", "summary_dir"),
    },
}

LINEAGE_SOURCE_STAGE = {
    "performance_benchmark": "b1_prestability",
    "evaluation_pipeline_qualification": "b1_prestability",
    "lr_uniform_sweep": "evaluation_pipeline_qualification",
    "lr_evaluation_summary": "lr_uniform_sweep",
    "lr_decision": "lr_evaluation_summary",
    "ptc500_training": "lr_decision",
    "ptc500_stability": "ptc500_training",
    "b1_full_closure": "ptc500_stability",
    "small_pilot_training": "b1_full_closure",
    "small_pilot_evaluation": "small_pilot_training",
    "small_pilot_summary": "small_pilot_evaluation",
    "small_pilot_decision": "small_pilot_summary",
}


@dataclass(frozen=True)
class StageVerifierSpec:
    stage_id: str
    verifier_id: str
    dependencies: Tuple[str, ...]
    evidence_kinds: Tuple[Tuple[str, str], ...]
    handler: Optional[Callable[[Mapping[str, Path], Path], Dict[str, Any]]]

    @property
    def supported(self) -> bool:
        return self.handler is not None


def _exact_fields(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    expected_set = set(expected)
    if set(value) != expected_set:
        raise Stage1ArtifactError(
            "{} fields differ; missing={}, unexpected={}".format(
                label,
                sorted(expected_set - set(value)),
                sorted(set(value) - expected_set),
            )
        )


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise Stage1ArtifactError("{} must be nonempty canonical text".format(label))
    return value


def _resolve_root_before_symlink(path: Path, label: str, *, kind: str) -> Path:
    supplied = Path(path).expanduser().absolute()
    if supplied.is_symlink():
        raise Stage1ArtifactError("{} root may not be a symlink".format(label))
    resolved = supplied.resolve(strict=True)
    if kind == "tree":
        if not resolved.is_dir():
            raise Stage1ArtifactError("{} must be a directory".format(label))
    elif kind == "file":
        if not resolved.is_file() or resolved.is_symlink():
            raise Stage1ArtifactError("{} must be a regular file".format(label))
    else:
        raise Stage1ArtifactError("unsupported evidence kind {!r}".format(kind))
    return resolved


def _run_json_command(argv: Sequence[str], label: str) -> Dict[str, Any]:
    completed = subprocess.run(
        list(argv), check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise Stage1ArtifactError(
            "{} failed rc={}: {}".format(label, completed.returncode, detail)
        )

    def unique(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Stage1ArtifactError("{} emitted duplicate JSON key".format(label))
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise Stage1ArtifactError("{} emitted non-finite {}".format(label, value))

    try:
        value = json.loads(
            completed.stdout, object_pairs_hook=unique, parse_constant=reject
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise Stage1ArtifactError("{} did not emit one strict JSON object".format(label)) from exc
    if not isinstance(value, dict):
        raise Stage1ArtifactError("{} JSON root is not an object".format(label))
    return value


def _official_run(workpack_root: Path, run_dir: Path) -> Dict[str, Any]:
    result = _run_json_command(
        (
            sys.executable,
            str(workpack_root / "scripts" / "verify_stage1_run.py"),
            "--run-dir",
            str(run_dir),
        ),
        "official Stage-1 run verifier",
    )
    if result.get("status") != "verified" or result.get("run_directory") != str(
        run_dir.resolve(strict=True)
    ):
        raise Stage1ArtifactError("official Stage-1 run verification differs")
    for field, basename in (
        ("run_manifest_sha256", "run_manifest.json"),
        ("SEALED.json_sha256", "SEALED.json"),
        ("DONE.json_sha256", "DONE.json"),
    ):
        if require_sha256(result.get(field), field) != sha256_file(run_dir / basename):
            raise Stage1ArtifactError("official run identity differs at {}".format(field))
    require_sha256(result.get("final_checkpoint_sha256"), "final checkpoint")
    return result


def _official_b1(workpack_root: Path, artifact_dir: Path) -> Dict[str, Any]:
    result = _run_json_command(
        (
            sys.executable,
            str(workpack_root / "scripts" / "verify_b1_prestability.py"),
            "final",
            str(artifact_dir),
        ),
        "official B1 prestability verifier",
    )
    if (
        result.get("kind") != "final"
        or result.get("prestability_gate_passed") is not True
        or result.get("full_b1_passed") is not False
    ):
        raise Stage1ArtifactError("official B1 prestability result differs")
    return result


def _handler_t5(evidence: Mapping[str, Path], workpack: Path) -> Dict[str, Any]:
    # The retained verifier is the frozen consumer for the accepted patch02 packet.
    script = workpack / "scripts" / "verify_stage1_upstreams.py"
    namespace: Dict[str, Any] = {"__file__": str(script), "__name__": "_ptc_t5_verifier"}
    exec(compile(script.read_bytes(), str(script), "exec"), namespace)
    result = namespace["_verify_t5"](evidence["closure_dir"])
    return {"status": "verified", "gate_passed": True, "t5": result}


def _validate_retained_generation_binding(
    checkpoint: Mapping[str, Any],
    audiocraft: Mapping[str, Any],
    generation_identity: Mapping[str, Any],
) -> Dict[str, Any]:
    """Bind retained model/source trees to the selected CFG generation identity."""

    if not isinstance(generation_identity, Mapping):
        raise Stage1ArtifactError("retained CFG generation identity is absent")
    expected = {
        "model_id": "facebook/musicgen-small",
        "checkpoint_sha256": checkpoint.get("checkpoint_sha256"),
        "state_dict_sha256": checkpoint.get("state_dict_sha256"),
        "compression_state_dict_sha256": checkpoint.get(
            "compression_state_dict_sha256"
        ),
        "audiocraft_base_commit": audiocraft.get("audiocraft_base_commit"),
        "audiocraft_source_sha256": audiocraft.get(
            "audiocraft_source_sha256"
        ),
        "audiocraft_lm_sha256": audiocraft.get("audiocraft_lm_sha256"),
    }
    for field, value in expected.items():
        if generation_identity.get(field) != value:
            category = (
                "MusicGen checkpoint"
                if field in {
                    "checkpoint_sha256",
                    "state_dict_sha256",
                    "compression_state_dict_sha256",
                }
                else "AudioCraft source"
                if field.startswith("audiocraft_")
                else "model"
            )
            raise Stage1ArtifactError(
                "retained {} identity differs at {}".format(category, field)
            )
    return {
        "checkpoint_identity": dict(checkpoint),
        "audiocraft_identity": dict(audiocraft),
        "cfg_generation_binding": {
            field: generation_identity[field] for field in expected
        },
    }


def _handler_retained(evidence: Mapping[str, Path], workpack: Path) -> Dict[str, Any]:
    copied = _run_json_command(
        (
            sys.executable,
            str(workpack / "scripts" / "verify_stage1_upstream_artifact.py"),
            str(evidence["upstream_dir"]),
        ),
        "retained-upstream verifier",
    )
    if copied.get("status") != "passed":
        raise Stage1ArtifactError("retained-upstream verification did not pass")

    report = load_json_strict(evidence["upstream_dir"] / "upstream_verification.json")
    root_labels = {
        "a1": "a1_dir",
        "small_cfg": "small_cfg_dir",
        "a2": "a2_dir",
        "a3": "a3_dir",
        "node3": "node3_dir",
        "t5": "t5_closure_dir",
    }
    live_trees = {
        name: sha256_tree(evidence[label]) for name, label in root_labels.items()
    }
    if (
        report.get("upstream_tree_sha256_before") != live_trees
        or report.get("upstream_tree_sha256_after") != live_trees
    ):
        raise Stage1ArtifactError("retained readiness tree hashes differ from live roots")
    expected_archive = evidence["t5_closure_dir"].with_suffix(".tar.gz")
    if evidence["t5_archive"] != expected_archive or sha256_file(
        evidence["t5_archive"]
    ) != report.get("t5_archive_sha256_before") or report.get(
        "t5_archive_sha256_before"
    ) != report.get("t5_archive_sha256_after"):
        raise Stage1ArtifactError("retained T5 archive binding differs")

    script = workpack / "scripts" / "verify_stage1_upstreams.py"
    namespace: Dict[str, Any] = {"__file__": str(script), "__name__": "_ptc_upstream_verifier"}
    exec(compile(script.read_bytes(), str(script), "exec"), namespace)
    from .cfg_decision import verify_cfg_scale_decision
    from .codec_prior_artifact import load_codec_prior_artifact

    scripts = str(workpack / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import run_disagreement_probe as probe

    a1 = load_codec_prior_artifact(evidence["a1_dir"])
    cfg = verify_cfg_scale_decision(evidence["small_cfg_dir"], require_selected=True)
    checkpoint = probe.verify_checkpoint_snapshot(evidence["musicgen_small_dir"])
    audiocraft = probe.verify_audiocraft_source(evidence["audiocraft_dir"])
    live_generation_identity = _validate_retained_generation_binding(
        checkpoint, audiocraft, cfg.generation_identity
    )
    live_sections = {
        "a1": {
            "artifact_seal_sha256": a1.artifact_seal_sha256,
            "codec_prior_sha256": a1.codec_prior_sha256,
            "prior": list(a1.prior),
        },
        "small_cfg": {
            "decision_file_sha256": cfg.decision_file_sha256,
            "decision_payload_sha256": cfg.decision_payload_sha256,
            "scientific_config_sha256": cfg.scientific_config_sha256,
            "selected_cfg_scale": cfg.selected_cfg_scale,
        },
        "a2_a3": namespace["_verify_a2_a3"](
            evidence["a2_dir"], evidence["a3_dir"]
        ),
        "node3": namespace["_verify_node3"](evidence["node3_dir"]),
        "t5": namespace["_verify_t5"](evidence["t5_closure_dir"]),
    }
    for name, value in live_sections.items():
        if report.get(name) != value:
            raise Stage1ArtifactError(
                "retained readiness section differs from live {} evidence".format(name)
            )
    for label, expected_sha256 in FROZEN_RETAINED_FILE_SHA256.items():
        if sha256_file(evidence[label]) != expected_sha256:
            raise Stage1ArtifactError("retained {} identity differs".format(label))
    if cfg.selected_cfg_scale != 5.0:
        raise Stage1ArtifactError("retained small CFG decision must select 5.0")
    return {
        "status": "verified",
        "gate_passed": True,
        "result": copied,
        "live_sections": live_sections,
        "live_tree_sha256": live_trees,
        "live_generation_identity": live_generation_identity,
        "frozen_file_sha256": dict(FROZEN_RETAINED_FILE_SHA256),
    }


def _handler_pilot(evidence: Mapping[str, Path], _workpack: Path) -> Dict[str, Any]:
    from .stage1_control import verify_pilot_eval_manifest

    result = verify_pilot_eval_manifest(
        evidence["artifact_dir"], evidence["source_dev_manifest"]
    )
    return {"status": "verified", "gate_passed": True, "result": result}


def _training_lineage_anchor_from_b1_single_result(
    single: Mapping[str, Any]
) -> Dict[str, Any]:
    """Build the canonical Stage-1 lineage from the audited live B1 models."""

    from .stage1_control import TRAINING_LINEAGE_SCHEMA, _validate_training_lineage_anchor

    scientific = single.get("scientific_config")
    if not isinstance(scientific, Mapping):
        raise Stage1ArtifactError("B1 single scientific config is absent for lineage")
    checkpoint = scientific.get("checkpoint_identity")
    audiocraft = scientific.get("audiocraft_identity")
    decision = scientific.get("cfg_decision")
    loaded_t5 = single.get("loaded_t5_identity")
    if not isinstance(checkpoint, Mapping):
        raise Stage1ArtifactError("B1 checkpoint identity is absent for lineage")
    if not isinstance(audiocraft, Mapping):
        raise Stage1ArtifactError("B1 AudioCraft identity is absent for lineage")
    if not isinstance(decision, Mapping):
        raise Stage1ArtifactError("B1 CFG decision is absent for lineage")
    if not isinstance(loaded_t5, Mapping):
        raise Stage1ArtifactError("B1 loaded T5 identity is absent for lineage")

    state_fields = (
        "student_state_sha256_before",
        "student_state_sha256_after",
        "teacher_state_sha256_before",
        "teacher_state_sha256_after",
    )
    state_hashes = [
        require_sha256(single.get(field), "B1 lineage {}".format(field))
        for field in state_fields
    ]
    if len(set(state_hashes)) != 1:
        raise Stage1ArtifactError("B1 audited student/teacher states differ")
    t5_identity = require_sha256(
        loaded_t5.get("identity_sha256"), "B1 lineage loaded T5"
    )
    if t5_identity != decision.get("loaded_t5_identity_sha256"):
        raise Stage1ArtifactError("B1 CFG and loaded T5 lineage differs")

    return _validate_training_lineage_anchor(
        {
            "schema_version": TRAINING_LINEAGE_SCHEMA,
            "model_id": scientific.get("model_id"),
            "base_checkpoint": checkpoint,
            "base_lm_state_sha256": state_hashes[0],
            "audiocraft_source_sha256": audiocraft.get(
                "audiocraft_source_sha256"
            ),
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


def _handler_b1(evidence: Mapping[str, Path], workpack: Path) -> Dict[str, Any]:
    result = _official_b1(workpack, evidence["artifact_dir"])
    scripts = str(workpack / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import run_disagreement_probe as probe
    from .cfg_decision import verify_cfg_scale_decision
    from .codec_prior_artifact import load_codec_prior_artifact

    probe_rows, dev_rows = probe.validate_probe_is_exact_dev_subset(
        evidence["probe_manifest"], evidence["dev_manifest"]
    )
    if len(probe_rows) != 256 or len(dev_rows) != 300:
        raise Stage1ArtifactError("B1 retained probe/dev cardinality differs")
    checkpoint = probe.verify_checkpoint_snapshot(evidence["musicgen_small_dir"])
    audiocraft = probe.verify_audiocraft_source(evidence["audiocraft_dir"])
    cfg = verify_cfg_scale_decision(evidence["small_cfg_dir"], require_selected=True)
    a1 = load_codec_prior_artifact(evidence["a1_dir"])
    a2 = probe.verify_probe_artifact(evidence["a2_dir"], require_primary=True)
    single = load_json_strict(evidence["artifact_dir"] / "single_gpu_results.json")
    scientific = single.get("scientific_config")
    if not isinstance(scientific, dict):
        raise Stage1ArtifactError("B1 single scientific config is absent")
    t5_root = evidence["t5_closure_dir"]
    t5_output = t5_root / "output_dir" if (t5_root / "output_dir").is_dir() else t5_root
    expected = {
        "checkpoint_identity": checkpoint,
        "audiocraft_identity": audiocraft,
        "probe_manifest_sha256": sha256_file(evidence["probe_manifest"]),
        "dev_manifest_sha256": sha256_file(evidence["dev_manifest"]),
        "a1_artifact_seal_sha256": a1.artifact_seal_sha256,
        "a2_artifact_seal_sha256": a2["artifact_seal_sha256"],
    }
    for field, value in expected.items():
        if scientific.get(field) != value:
            raise Stage1ArtifactError("B1 live input differs at {}".format(field))
    if scientific.get("cfg_decision") != {
        "selected_cfg_scale": cfg.selected_cfg_scale,
        "decision_file_sha256": cfg.decision_file_sha256,
        "decision_payload_sha256": cfg.decision_payload_sha256,
        "scientific_config_sha256": cfg.scientific_config_sha256,
        "loaded_t5_identity_sha256": cfg.generation_identity[
            "loaded_t5_identity_sha256"
        ],
    }:
        raise Stage1ArtifactError("B1 live CFG decision identity differs")
    node3 = scientific.get("node3_identity")
    if not isinstance(node3, dict) or node3.get("status_sha256") != sha256_file(
        evidence["node3_dir"] / "STATUS.json"
    ):
        raise Stage1ArtifactError("B1 live Node-3 identity differs")
    t5 = scientific.get("t5_closure_identity")
    if not isinstance(t5, dict) or t5.get("artifact_seal_sha256") != sha256_file(
        t5_output / "artifact_seal.json"
    ):
        raise Stage1ArtifactError("B1 live T5 identity differs")
    lineage = _training_lineage_anchor_from_b1_single_result(single)
    return {
        "status": "verified",
        "gate_passed": True,
        "training_lineage_anchor": lineage,
        "result": result,
        "live_input_identity": expected,
    }


def _handler_perf(evidence: Mapping[str, Path], workpack: Path) -> Dict[str, Any]:
    node_results = []
    for index in range(4):
        node_results.append(
            _run_json_command(
                (
                    sys.executable,
                    str(workpack / "scripts" / "verify_perf_benchmark.py"),
                    "--node-dir",
                    str(evidence["node{}_dir".format(index)]),
                    "--train-manifest",
                    str(evidence["train_manifest"]),
                    "--small-cfg-dir",
                    str(evidence["small_cfg_dir"]),
                    "--audiocraft-dir",
                    str(evidence["audiocraft_dir"]),
                    "--musicgen-small-dir",
                    str(evidence["musicgen_small_dir"]),
                ),
                "performance node verifier",
            )
        )
    aggregate = _run_json_command(
        (
            sys.executable,
            str(workpack / "scripts" / "verify_perf_benchmark.py"),
            "--aggregate-dir",
            str(evidence["aggregate_dir"]),
        ),
        "performance aggregate verifier",
    )
    if aggregate.get("status") != "verified" or any(
        row.get("status") != "verified" for row in node_results
    ):
        raise Stage1ArtifactError("performance verifier result differs")
    labels = {row.get("node_label") for row in node_results}
    if labels != {"node-0", "node-1", "node-2", "node-3"}:
        raise Stage1ArtifactError("performance node set differs")
    live_input_identities = [row.get("live_input_identity") for row in node_results]
    if (
        not isinstance(live_input_identities[0], dict)
        or any(item != live_input_identities[0] for item in live_input_identities[1:])
        or any(row.get("verified_input_arm_count") != 4 for row in node_results)
    ):
        raise Stage1ArtifactError("performance node live-input bindings differ")
    scripts = str(workpack / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import perf_benchmark_common as perf

    from .stage1_control import _training_lineage_anchor_from_run_manifest

    node_dirs = [evidence["node{}_dir".format(index)] for index in range(4)]
    lineage_anchor: Optional[Dict[str, Any]] = None
    formal_run_count = 0
    for node_dir in node_dirs:
        for arm_dir in perf.node_arm_directories(node_dir).values():
            manifest = load_json_strict(arm_dir / "formal_run" / "run_manifest.json")
            observed_anchor = _training_lineage_anchor_from_run_manifest(manifest)
            if lineage_anchor is None:
                lineage_anchor = observed_anchor
            elif observed_anchor != lineage_anchor:
                raise Stage1ArtifactError(
                    "performance formal-run training lineage differs across arms"
                )
            formal_run_count += 1
    if formal_run_count != 16 or lineage_anchor is None:
        raise Stage1ArtifactError("performance formal-run coverage differs")
    recomputed = perf.aggregate_node_directories(node_dirs)
    observed = load_json_strict(evidence["aggregate_dir"] / "paired_summary.json")
    for field, expected in recomputed.items():
        if observed.get(field) != expected:
            raise Stage1ArtifactError(
                "performance aggregate differs from four live nodes at {}".format(field)
            )
    input_nodes = observed.get("input_nodes")
    if not isinstance(input_nodes, dict) or set(input_nodes) != labels:
        raise Stage1ArtifactError("performance aggregate input-node binding differs")
    by_label = {str(row["node_label"]): row for row in node_results}
    for index in range(4):
        label = "node-{}".format(index)
        if input_nodes[label] != {
            "directory": str(evidence["node{}_dir".format(index)]),
            "artifact_seal_sha256": by_label[label]["artifact_seal_sha256"],
        }:
            raise Stage1ArtifactError("performance aggregate node identity differs")
    return {
        "status": "verified",
        "gate_passed": aggregate.get("hard_resource_gate_passed") is True,
        "measurement_quality": aggregate.get("measurement_quality"),
        "aggregate": aggregate,
        "verified_input_arm_count": formal_run_count,
        "live_input_identity": live_input_identities[0],
        "training_lineage_anchor": lineage_anchor,
        "node_artifact_seals": {
            str(row["node_label"]): row["artifact_seal_sha256"] for row in node_results
        },
    }


def _run_lineage_and_config(
    workpack: Path,
    run_dir: Path,
    *,
    mode: str,
    learning_rate: float,
    optimizer_steps: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from .stage1_control import _training_lineage_anchor_from_run_manifest
    from .train_utils import Stage1Config, validate_config

    verified = _official_run(workpack, run_dir)
    manifest = load_json_strict(run_dir / "run_manifest.json")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise Stage1ArtifactError("Stage-1 run config is absent")
    try:
        resolved_config = Stage1Config(**config)
        mode_spec = validate_config(resolved_config)
    except (TypeError, ValueError) as exc:
        raise Stage1ArtifactError("Stage-1 run config is not a valid frozen config") from exc
    expected = {
        "mode": mode,
        "seed": 2027,
        "learning_rate": learning_rate,
        "weight_decay": 0.0,
        "max_optimizer_steps": optimizer_steps,
        "save_every": 250,
        "log_every": 1,
        "rank_batch_size": 2,
        "expected_world_size": 8,
        "grad_accum_steps": 4,
        "effective_global_batch": 64,
        "duration_seconds": 10.0,
        "codec_frame_rate": 50.0,
        "token_frames": 500,
        "rollout_temperature": 1.0,
        "rollout_top_k": 250,
        "rollout_top_p": 0.0,
        "teacher_cfg_scale": 5.0,
        "distillation_temperature": 1.0,
        "grad_clip_norm": 1.0,
        "teacher_forward_mode": "batched",
        "check_finite": True,
        "random_mask_namespace": 5701,
        "optimizer_schedule": "linear_warmup_then_constant",
        "warmup_optimizer_steps": 50,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_eps": 1.0e-8,
        "kl_direction": "forward",
        "denominator_rtol": 1.0e-6,
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise Stage1ArtifactError(
                "Stage-1 run config differs at {} for {}".format(field, mode)
            )
    if str(Path(config.get("output_dir", "")).expanduser().resolve(strict=True)) != str(
        run_dir.resolve(strict=True)
    ):
        raise Stage1ArtifactError("Stage-1 run config output directory differs")
    if manifest.get("world_size") != 8 or manifest.get("nnodes") != 1:
        raise Stage1ArtifactError("Stage-1 run must be one machine/eight GPUs")
    if verified.get("optimizer_step") != optimizer_steps:
        raise Stage1ArtifactError("official run optimizer-step differs")
    prior_path = config.get("codebook_prior_artifact_dir")
    if mode_spec.requires_perceptual_prior:
        if not isinstance(prior_path, str) or not prior_path:
            raise Stage1ArtifactError("weighted Stage-1 run lacks the frozen A1 prior")
        if manifest.get("codebook_prior_artifact") is None:
            raise Stage1ArtifactError("weighted Stage-1 run lacks prior identity")
    elif prior_path is not None or manifest.get("codebook_prior_artifact") is not None:
        raise Stage1ArtifactError("uniform-weight Stage-1 run may not consume a prior")
    manifest_path = _resolve_root_before_symlink(
        Path(resolved_config.manifest), "Stage-1 training manifest", kind="file"
    )
    if manifest.get("manifest_sha256") != sha256_file(manifest_path):
        raise Stage1ArtifactError("Stage-1 run manifest hash differs from live input")
    return verified, _training_lineage_anchor_from_run_manifest(manifest)


def _handler_evaluation_qualification(
    evidence: Mapping[str, Path], workpack: Path
) -> Dict[str, Any]:
    from .stage1_control import _training_lineage_anchor_from_generation_config
    from .cfg_decision import verify_cfg_scale_decision
    from .stage1_metrics import verify_metric_artifact

    scripts = str(workpack / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import run_disagreement_probe as probe

    checkpoint = probe.verify_checkpoint_snapshot(evidence["musicgen_small_dir"])
    audiocraft = probe.verify_audiocraft_source(evidence["audiocraft_dir"])
    cfg = verify_cfg_scale_decision(evidence["small_cfg_dir"], require_selected=True)
    generation_identity = cfg.generation_identity
    if not isinstance(generation_identity, Mapping):
        raise Stage1ArtifactError("retained CFG generation identity is absent")
    retained_matches = {
        "model_id": "facebook/musicgen-small",
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "state_dict_sha256": checkpoint["state_dict_sha256"],
        "compression_state_dict_sha256": checkpoint[
            "compression_state_dict_sha256"
        ],
        "audiocraft_source_sha256": audiocraft["audiocraft_source_sha256"],
    }
    for field, expected in retained_matches.items():
        if generation_identity.get(field) != expected:
            raise Stage1ArtifactError(
                "retained CFG generation identity differs at {}".format(field)
            )
    expected_cfg = {
        "decision_file_sha256": cfg.decision_file_sha256,
        "decision_payload_sha256": cfg.decision_payload_sha256,
        "scientific_config_sha256": cfg.scientific_config_sha256,
        "selected_cfg_scale": cfg.selected_cfg_scale,
        "loaded_t5_identity_sha256": generation_identity.get(
            "loaded_t5_identity_sha256"
        ),
    }
    verified: Dict[str, Any] = {}
    for name, source_kind in (
        ("base", "base_no_cfg"),
        ("teacher", "frozen_cfg_teacher"),
    ):
        result = verify_metric_artifact(
            evidence["{}_metric_dir".format(name)],
            generation_dir=evidence["{}_generation_dir".format(name)],
            quality_dir=evidence["{}_quality_dir".format(name)],
            eval_manifest_dir=evidence["eval_manifest_dir"],
        )
        config = result["generation"]["scientific_config"]
        if config.get("source_kind") != source_kind:
            raise Stage1ArtifactError("evaluation qualification source kind differs")
        if config.get("model_id") != "facebook/musicgen-small":
            raise Stage1ArtifactError("evaluation retained model identity differs")
        if config.get("base_checkpoint") != checkpoint:
            raise Stage1ArtifactError(
                "evaluation retained MusicGen checkpoint identity differs"
            )
        if config.get("audiocraft_source_sha256") != audiocraft.get(
            "audiocraft_source_sha256"
        ):
            raise Stage1ArtifactError(
                "evaluation retained AudioCraft source identity differs"
            )
        if config.get("cfg_decision") != expected_cfg:
            raise Stage1ArtifactError("evaluation retained CFG decision identity differs")
        verified[name] = result
    base_lineage = _training_lineage_anchor_from_generation_config(
        verified["base"]["generation"]["scientific_config"]
    )
    teacher_lineage = _training_lineage_anchor_from_generation_config(
        verified["teacher"]["generation"]["scientific_config"]
    )
    if base_lineage != teacher_lineage:
        raise Stage1ArtifactError("base/teacher evaluation lineage differs")
    return {
        "status": "verified",
        "gate_passed": True,
        "training_lineage_anchor": base_lineage,
        "retained_generation_binding": {
            "base_checkpoint": checkpoint,
            "audiocraft_source_sha256": audiocraft[
                "audiocraft_source_sha256"
            ],
            "cfg_decision": expected_cfg,
        },
        "base_metric_seal": verified["base"]["artifact_seal_sha256"],
        "teacher_metric_seal": verified["teacher"]["artifact_seal_sha256"],
    }


def _handler_lr_sweep(evidence: Mapping[str, Path], workpack: Path) -> Dict[str, Any]:
    results = []
    lineages = []
    for label, learning_rate in (("1e6", 1.0e-6), ("3e6", 3.0e-6), ("1e5", 1.0e-5)):
        verified, lineage = _run_lineage_and_config(
            workpack,
            evidence["run_{}".format(label)],
            mode="uniform100",
            learning_rate=learning_rate,
            optimizer_steps=500,
        )
        results.append(verified)
        lineages.append(lineage)
    if any(lineage != lineages[0] for lineage in lineages[1:]):
        raise Stage1ArtifactError("LR sweep training lineage differs across candidates")
    return {
        "status": "verified",
        "gate_passed": True,
        "training_lineage_anchor": lineages[0],
        "run_verifications": results,
    }


def _handler_ptc500_training(
    evidence: Mapping[str, Path], workpack: Path
) -> Dict[str, Any]:
    from .stage1_control import verify_lr_decision

    decision = verify_lr_decision(evidence["lr_decision_dir"], evidence["lr_summary_dir"])
    learning_rate = decision.get("selected_learning_rate")
    if decision.get("status") != "selected" or learning_rate not in {
        1.0e-6, 3.0e-6, 1.0e-5
    }:
        raise Stage1ArtifactError("PTC500 training lacks a selected LR")
    verified, lineage = _run_lineage_and_config(
        workpack,
        evidence["run_dir"],
        mode="ptc50",
        learning_rate=float(learning_rate),
        optimizer_steps=500,
    )
    if decision.get("training_lineage_anchor") != lineage:
        raise Stage1ArtifactError("PTC500 run lineage differs from LR decision")
    return {
        "status": "verified",
        "gate_passed": True,
        "training_lineage_anchor": lineage,
        "run_verification": verified,
    }


def _handler_small_training(
    evidence: Mapping[str, Path], workpack: Path
) -> Dict[str, Any]:
    from .stage1_control import verify_b1_full_closure, verify_lr_decision

    decision = verify_lr_decision(evidence["lr_decision_dir"], evidence["lr_summary_dir"])
    learning_rate = decision.get("selected_learning_rate")
    if decision.get("status") != "selected" or not isinstance(learning_rate, (int, float)):
        raise Stage1ArtifactError("small-pilot training lacks a selected LR")
    # The closure's own live registry receipt is the dependency; this direct seal
    # read prevents swapping a different closure directory under this group.
    closure = load_json_strict(evidence["b1_full_dir"] / "b1_full_closure.json")
    if closure.get("full_b1_passed") is not True or closure.get("gate_passed") is not True:
        raise Stage1ArtifactError("small-pilot training requires full B1 closure")
    results = []
    lineages = []
    for method in (
        "uniform100", "codebook100", "random50", "prefix50", "disagreement50", "ptc50"
    ):
        verified, lineage = _run_lineage_and_config(
            workpack,
            evidence[method + "_run_dir"],
            mode=method,
            learning_rate=float(learning_rate),
            optimizer_steps=500,  # Ruling #6 4th addendum: 1000 → 500
        )
        results.append(verified)
        lineages.append(lineage)
    if any(lineage != lineages[0] for lineage in lineages[1:]):
        raise Stage1ArtifactError("small-pilot run lineages differ")
    if decision.get("training_lineage_anchor") != lineages[0]:
        raise Stage1ArtifactError("small-pilot lineage differs from LR decision")
    return {
        "status": "verified",
        "gate_passed": True,
        "training_lineage_anchor": lineages[0],
        "run_verifications": results,
    }


def _small_evaluation_evidence_names() -> Tuple[Tuple[str, str], ...]:
    result = [
        ("eval_manifest_dir", "tree"),
        ("base_generation_dir", "tree"),
        ("base_quality_dir", "tree"),
        ("base_metric_dir", "tree"),
        ("reference_dir", "tree"),
        ("a1_manifest", "file"),
        ("a1_report", "file"),
        ("model_pins_dir", "tree"),
    ]
    for method in (
        "uniform100", "codebook100", "random50", "prefix50", "disagreement50", "ptc50"
    ):
        result.append(("{}_run_dir".format(method), "tree"))
        for kind in ("generation", "quality", "metric", "diversity"):
            result.append(("{}_{}_dir".format(method, kind), "tree"))
    return tuple(result)


def _require_small_checkpoint_1000(
    generation_config: Mapping[str, Any], live_run: Mapping[str, Any]
) -> None:
    # Ruling #6 FOURTH ADDENDUM (2026-08-28): pilot-authorized reduction of
    # small-pilot training from 1000 → 500 optimizer steps based on LR sweep
    # convergence evidence (loss plateau between step 281 and step 500).
    # Function name retained to minimize sibling-patch surface; contract
    # updated to require checkpoint_step == 500.
    if generation_config.get("checkpoint_step") != 500:
        raise Stage1ArtifactError("small-pilot generation must use checkpoint step 500")
    trained_checkpoint = generation_config.get("trained_checkpoint")
    if not isinstance(trained_checkpoint, Mapping) or trained_checkpoint.get(
        "checkpoint_sha256"
    ) != live_run.get("final_checkpoint_sha256"):
        raise Stage1ArtifactError("small-pilot generation final checkpoint identity differs")


def _handler_small_evaluation(
    evidence: Mapping[str, Path], workpack: Path
) -> Dict[str, Any]:
    from .stage1_control import _training_lineage_anchor_from_generation_config
    from .stage1_diversity_fad import verify_diversity_fad_artifact
    from .stage1_metrics import verify_metric_artifact

    base = verify_metric_artifact(
        evidence["base_metric_dir"],
        generation_dir=evidence["base_generation_dir"],
        quality_dir=evidence["base_quality_dir"],
        eval_manifest_dir=evidence["eval_manifest_dir"],
    )
    if base["generation"]["scientific_config"].get("source_kind") != "base_no_cfg":
        raise Stage1ArtifactError("small-pilot evaluation base anchor differs")
    base_lineage = _training_lineage_anchor_from_generation_config(
        base["generation"]["scientific_config"]
    )
    results: Dict[str, Any] = {}
    for method in (
        "uniform100", "codebook100", "random50", "prefix50", "disagreement50", "ptc50"
    ):
        metric = verify_metric_artifact(
            evidence[method + "_metric_dir"],
            generation_dir=evidence[method + "_generation_dir"],
            quality_dir=evidence[method + "_quality_dir"],
            eval_manifest_dir=evidence["eval_manifest_dir"],
        )
        config = metric["generation"]["scientific_config"]
        if config.get("source_kind") != "trained_no_cfg" or config.get("method") != method:
            raise Stage1ArtifactError("small-pilot trained generation identity differs")
        stage1_run = config.get("stage1_run")
        if not isinstance(stage1_run, dict):
            raise Stage1ArtifactError("trained generation lacks official run identity")
        run_dir = evidence[method + "_run_dir"]
        run_manifest = load_json_strict(run_dir / "run_manifest.json")
        run_config = run_manifest.get("config")
        if not isinstance(run_config, dict) or not isinstance(
            run_config.get("learning_rate"), (int, float)
        ):
            raise Stage1ArtifactError("small-pilot run learning rate is absent")
        live_run, run_lineage = _run_lineage_and_config(
            workpack,
            run_dir,
            mode=method,
            learning_rate=float(run_config["learning_rate"]),
            optimizer_steps=500,  # Ruling #6 4th addendum: 1000 → 500
        )
        if (
            live_run != stage1_run
            or sha256_tree(run_dir) != config.get("stage1_run_tree_sha256")
            or config.get("train_seed") != 2027
            or config.get("learning_rate") != run_config["learning_rate"]
            or _training_lineage_anchor_from_generation_config(config) != run_lineage
            or run_lineage != base_lineage
        ):
            raise Stage1ArtifactError("trained generation/live run tree binding differs")
        _require_small_checkpoint_1000(config, live_run)
        diversity = verify_diversity_fad_artifact(
            evidence[method + "_diversity_dir"],
            generation_dir=evidence[method + "_generation_dir"],
            eval_manifest_dir=evidence["eval_manifest_dir"],
            reference_dir=evidence["reference_dir"],
            a1_manifest=evidence["a1_manifest"],
            a1_report=evidence["a1_report"],
            model_pins_dir=evidence["model_pins_dir"],
        )
        if diversity["generation"]["artifact_seal_sha256"] != metric["generation"][
            "artifact_seal_sha256"
        ]:
            raise Stage1ArtifactError("small-pilot metric/diversity generation differs")
        results[method] = {
            "run": live_run,
            "metric_seal": metric["artifact_seal_sha256"],
            "diversity_fad_seal": diversity["artifact_seal_sha256"],
        }
    return {
        "status": "verified",
        "gate_passed": True,
        "training_lineage_anchor": base_lineage,
        "methods": results,
    }


def _lr_summary_evidence_names() -> Tuple[Tuple[str, str], ...]:
    result = [
        ("summary_dir", "tree"),
        ("eval_manifest_dir", "tree"),
        ("base_generation_dir", "tree"),
        ("base_quality_dir", "tree"),
        ("base_metric_dir", "tree"),
    ]
    for label in ("1e6", "3e6", "1e5"):
        for kind in ("run", "generation", "quality", "metric"):
            result.append(("candidate_{}_{}_dir".format(label, kind), "tree"))
    return tuple(result)


def _handler_lr_summary(evidence: Mapping[str, Path], workpack: Path) -> Dict[str, Any]:
    from .stage1_control import (
        LR_SUMMARY_BASENAME,
        build_lr_evaluation_summary,
        verify_lr_evaluation_summary,
    )

    candidates = []
    run_identities = []
    for label in ("1e6", "3e6", "1e5"):
        run_dir = evidence["candidate_{}_run_dir".format(label)]
        verified = _official_run(workpack, run_dir)
        run_identities.append(verified)
        candidates.append(
            {
                "run_dir": run_dir,
                "run_verification": verified,
                "generation_dir": evidence["candidate_{}_generation_dir".format(label)],
                "quality_dir": evidence["candidate_{}_quality_dir".format(label)],
                "metric_dir": evidence["candidate_{}_metric_dir".format(label)],
            }
        )
    expected = build_lr_evaluation_summary(
        eval_manifest_dir=evidence["eval_manifest_dir"],
        base_generation_dir=evidence["base_generation_dir"],
        base_quality_dir=evidence["base_quality_dir"],
        base_metric_dir=evidence["base_metric_dir"],
        candidates=candidates,
    )
    observed = verify_lr_evaluation_summary(evidence["summary_dir"])
    if observed != expected:
        raise Stage1ArtifactError("LR summary differs from live official runs/evaluations")
    return {
        "status": "verified",
        "gate_passed": True,
        "training_lineage_anchor": expected.get("training_lineage_anchor"),
        "summary_sha256": sha256_file(evidence["summary_dir"] / LR_SUMMARY_BASENAME),
        "artifact_seal_sha256": sha256_file(
            evidence["summary_dir"] / "artifact_seal.json"
        ),
        "run_verifications": run_identities,
    }


def _handler_lr_decision(evidence: Mapping[str, Path], _workpack: Path) -> Dict[str, Any]:
    from .stage1_control import LR_DECISION_BASENAME, verify_lr_decision

    decision = verify_lr_decision(evidence["decision_dir"], evidence["summary_dir"])
    return {
        "status": "verified",
        "gate_passed": decision.get("status") == "selected",
        "training_lineage_anchor": decision.get("training_lineage_anchor"),
        "scientific_status": decision.get("status"),
        "selected_learning_rate": decision.get("selected_learning_rate"),
        "decision_sha256": sha256_file(evidence["decision_dir"] / LR_DECISION_BASENAME),
        "artifact_seal_sha256": sha256_file(
            evidence["decision_dir"] / "artifact_seal.json"
        ),
    }


def _handler_ptc500(evidence: Mapping[str, Path], workpack: Path) -> Dict[str, Any]:
    from .stage1_control import PTC500_REPORT_BASENAME, verify_ptc500_stability_report

    run = _official_run(workpack, evidence["run_dir"])
    b1 = _official_b1(workpack, evidence["b1_dir"])
    decision = verify_ptc500_stability_report(
        evidence["artifact_dir"],
        evidence["run_dir"],
        evidence["lr_decision_dir"],
        evidence["lr_summary_dir"],
        run,
        evidence["b1_dir"],
        b1,
    )
    return {
        "status": "verified",
        "gate_passed": decision.get("gate_passed") is True,
        "b1_11_gate_passed": decision.get("b1_11_gate_passed") is True,
        "training_lineage_anchor": decision.get("training_lineage_anchor"),
        "report_sha256": sha256_file(evidence["artifact_dir"] / PTC500_REPORT_BASENAME),
        "artifact_seal_sha256": sha256_file(
            evidence["artifact_dir"] / "artifact_seal.json"
        ),
        "run_verification": run,
        "b1_prestability_seal_sha256": b1.get("seal_sha256"),
    }


def _handler_b1_closure(evidence: Mapping[str, Path], workpack: Path) -> Dict[str, Any]:
    from .stage1_control import B1_FULL_CLOSURE_BASENAME, verify_b1_full_closure

    b1 = _official_b1(workpack, evidence["b1_dir"])
    closure = verify_b1_full_closure(
        evidence["artifact_dir"], evidence["ptc500_dir"], evidence["b1_dir"], b1
    )
    return {
        "status": "verified",
        "gate_passed": closure.get("gate_passed") is True,
        "full_b1_passed": closure.get("full_b1_passed") is True,
        "training_lineage_anchor": closure.get("training_lineage_anchor"),
        "report_sha256": sha256_file(
            evidence["artifact_dir"] / B1_FULL_CLOSURE_BASENAME
        ),
        "artifact_seal_sha256": sha256_file(
            evidence["artifact_dir"] / "artifact_seal.json"
        ),
        "b1_prestability_seal_sha256": b1.get("seal_sha256"),
    }


def _small_summary_evidence_names() -> Tuple[Tuple[str, str], ...]:
    result = [
        ("summary_dir", "tree"),
        ("eval_manifest_dir", "tree"),
        ("lr_summary_dir", "tree"),
        ("lr_decision_dir", "tree"),
        ("base_generation_dir", "tree"),
        ("base_quality_dir", "tree"),
        ("base_metric_dir", "tree"),
        ("reference_dir", "tree"),
        ("a1_manifest", "file"),
        ("a1_report", "file"),
        ("model_pins_dir", "tree"),
    ]
    for method in (
        "uniform100", "codebook100", "random50", "prefix50", "disagreement50", "ptc50"
    ):
        for kind in ("run", "generation", "quality", "metric", "diversity"):
            result.append(("{}_{}_dir".format(method, kind), "tree"))
    return tuple(result)


def _handler_small_summary(evidence: Mapping[str, Path], workpack: Path) -> Dict[str, Any]:
    from .stage1_control import (
        SMALL_PILOT_SUMMARY_BASENAME,
        build_small_pilot_summary,
        verify_small_pilot_summary,
    )

    bundles = []
    run_identities = []
    for method in (
        "uniform100", "codebook100", "random50", "prefix50", "disagreement50", "ptc50"
    ):
        run_dir = evidence["{}_run_dir".format(method)]
        verified = _official_run(workpack, run_dir)
        run_identities.append(verified)
        bundles.append(
            {
                "run_dir": run_dir,
                "run_verification": verified,
                "generation_dir": evidence["{}_generation_dir".format(method)],
                "quality_dir": evidence["{}_quality_dir".format(method)],
                "metric_dir": evidence["{}_metric_dir".format(method)],
                "diversity_fad_dir": evidence["{}_diversity_dir".format(method)],
            }
        )
    expected = build_small_pilot_summary(
        eval_manifest_dir=evidence["eval_manifest_dir"],
        lr_summary_dir=evidence["lr_summary_dir"],
        lr_decision_dir=evidence["lr_decision_dir"],
        base_generation_dir=evidence["base_generation_dir"],
        base_quality_dir=evidence["base_quality_dir"],
        base_metric_dir=evidence["base_metric_dir"],
        method_bundles=bundles,
        reference_dir=evidence["reference_dir"],
        a1_manifest=evidence["a1_manifest"],
        a1_report=evidence["a1_report"],
        model_pins_dir=evidence["model_pins_dir"],
    )
    observed = verify_small_pilot_summary(evidence["summary_dir"])
    if observed != expected:
        raise Stage1ArtifactError("small-pilot summary differs from live run/evaluation evidence")
    return {
        "status": "verified",
        "gate_passed": True,
        "training_lineage_anchor": observed.get("training_lineage_anchor"),
        "summary_sha256": sha256_file(
            evidence["summary_dir"] / SMALL_PILOT_SUMMARY_BASENAME
        ),
        "artifact_seal_sha256": sha256_file(
            evidence["summary_dir"] / "artifact_seal.json"
        ),
        "run_verifications": run_identities,
    }


def _handler_small_decision(evidence: Mapping[str, Path], _workpack: Path) -> Dict[str, Any]:
    from .stage1_control import SMALL_PILOT_DECISION_BASENAME, verify_small_pilot_decision

    decision = verify_small_pilot_decision(
        evidence["decision_dir"], evidence["summary_dir"]
    )
    return {
        "status": "verified",
        "gate_passed": decision.get("gate_passed") is True,
        "training_lineage_anchor": decision.get("training_lineage_anchor"),
        "scientific_status": decision.get("scientific_status"),
        "required_action": decision.get("required_action"),
        "decision_sha256": sha256_file(
            evidence["decision_dir"] / SMALL_PILOT_DECISION_BASENAME
        ),
        "artifact_seal_sha256": sha256_file(
            evidence["decision_dir"] / "artifact_seal.json"
        ),
    }


def _stage_graph() -> Tuple[Dict[str, Any], ...]:
    from .stage1_control import STAGE_NODES

    return tuple(dict(node) for node in STAGE_NODES)


def _build_registry() -> Dict[str, StageVerifierSpec]:
    graph = {node["id"]: tuple(node["depends_on"]) for node in _stage_graph()}
    definitions: Dict[str, Tuple[str, Tuple[Tuple[str, str], ...], Optional[Callable[..., Any]]]] = {
        "t5_closure": ("ptc-opd-t5-closure-official-v1", (("closure_dir", "tree"),), _handler_t5),
        "retained_seals_audit": (
            "ptc-opd-retained-upstreams-official-v2",
            (
                ("upstream_dir", "tree"),
                ("a1_dir", "tree"),
                ("small_cfg_dir", "tree"),
                ("a2_dir", "tree"),
                ("a3_dir", "tree"),
                ("node3_dir", "tree"),
                ("t5_closure_dir", "tree"),
                ("t5_archive", "file"),
                ("train_manifest", "file"),
                ("dev_manifest", "file"),
                ("probe_manifest", "file"),
                ("a1_manifest", "file"),
                ("a1_report", "file"),
                ("audiocraft_dir", "tree"),
                ("musicgen_small_dir", "tree"),
            ),
            _handler_retained,
        ),
        "pilot_eval_manifest": ("ptc-opd-pilot-manifest-bit-rebuild-v1", (("artifact_dir", "tree"), ("source_dev_manifest", "file")), _handler_pilot),
        "b1_prestability": (
            "ptc-opd-b1-prestability-final-v2",
            (
                ("artifact_dir", "tree"),
                ("probe_manifest", "file"),
                ("dev_manifest", "file"),
                ("small_cfg_dir", "tree"),
                ("a1_dir", "tree"),
                ("a2_dir", "tree"),
                ("node3_dir", "tree"),
                ("t5_closure_dir", "tree"),
                ("audiocraft_dir", "tree"),
                ("musicgen_small_dir", "tree"),
            ),
            _handler_b1,
        ),
        "performance_benchmark": (
            "ptc-opd-performance-four-node-live-inputs-v2",
            (
                ("aggregate_dir", "tree"),
                ("node0_dir", "tree"),
                ("node1_dir", "tree"),
                ("node2_dir", "tree"),
                ("node3_dir", "tree"),
                ("train_manifest", "file"),
                ("small_cfg_dir", "tree"),
                ("audiocraft_dir", "tree"),
                ("musicgen_small_dir", "tree"),
            ),
            _handler_perf,
        ),
        "evaluation_pipeline_qualification": ("ptc-opd-evaluation-pipeline-group-v1", (("eval_manifest_dir", "tree"), ("musicgen_small_dir", "tree"), ("audiocraft_dir", "tree"), ("small_cfg_dir", "tree"), ("base_generation_dir", "tree"), ("base_quality_dir", "tree"), ("base_metric_dir", "tree"), ("teacher_generation_dir", "tree"), ("teacher_quality_dir", "tree"), ("teacher_metric_dir", "tree")), _handler_evaluation_qualification),
        "lr_uniform_sweep": ("ptc-opd-lr-sweep-official-runs-v1", (("run_1e6", "tree"), ("run_3e6", "tree"), ("run_1e5", "tree")), _handler_lr_sweep),
        "lr_evaluation_summary": ("ptc-opd-lr-summary-live-group-v1", _lr_summary_evidence_names(), _handler_lr_summary),
        "lr_decision": ("ptc-opd-lr-decision-frozen-rule-v1", (("decision_dir", "tree"), ("summary_dir", "tree")), _handler_lr_decision),
        "ptc500_training": ("ptc-opd-ptc500-official-run-v1", (("run_dir", "tree"), ("lr_summary_dir", "tree"), ("lr_decision_dir", "tree")), _handler_ptc500_training),
        "ptc500_stability": ("ptc-opd-ptc500-b1-11-live-v1", (("artifact_dir", "tree"), ("run_dir", "tree"), ("lr_summary_dir", "tree"), ("lr_decision_dir", "tree"), ("b1_dir", "tree")), _handler_ptc500),
        "b1_full_closure": ("ptc-opd-b1-full-closure-live-v1", (("artifact_dir", "tree"), ("ptc500_dir", "tree"), ("b1_dir", "tree")), _handler_b1_closure),
        "small_pilot_training": ("ptc-opd-small-pilot-six-official-runs-v1", tuple((method + "_run_dir", "tree") for method in ("uniform100", "codebook100", "random50", "prefix50", "disagreement50", "ptc50")) + (("lr_summary_dir", "tree"), ("lr_decision_dir", "tree"), ("b1_full_dir", "tree")), _handler_small_training),
        "small_pilot_evaluation": ("ptc-opd-small-pilot-evaluation-group-v1", _small_evaluation_evidence_names(), _handler_small_evaluation),
        "small_pilot_summary": ("ptc-opd-small-pilot-summary-live-group-v1", _small_summary_evidence_names(), _handler_small_summary),
        "small_pilot_decision": ("ptc-opd-small-pilot-decision-frozen-gates-v1", (("decision_dir", "tree"), ("summary_dir", "tree")), _handler_small_decision),
    }
    if set(definitions) != set(graph):
        raise Stage1ArtifactError("stage verifier catalogue does not match controller DAG")
    return {
        stage_id: StageVerifierSpec(
            stage_id=stage_id,
            verifier_id=definition[0],
            dependencies=graph[stage_id],
            evidence_kinds=definition[1],
            handler=definition[2],
        )
        for stage_id, definition in definitions.items()
    }


def stage_verifier_registry() -> Dict[str, StageVerifierSpec]:
    return _build_registry()


def stage_verifier_catalog() -> Dict[str, Dict[str, Any]]:
    return {
        stage_id: {
            "verifier_id": spec.verifier_id,
            "supported": spec.supported,
            "dependencies": list(spec.dependencies),
            "evidence": {name: kind for name, kind in spec.evidence_kinds},
        }
        for stage_id, spec in stage_verifier_registry().items()
    }


def _authority_context(workpack_root: Path, contract_path: Path) -> Dict[str, Any]:
    root = _resolve_root_before_symlink(workpack_root, "workpack", kind="tree")
    expected_contract = root / CONTRACT_RELATIVE_PATH
    supplied_contract = Path(contract_path).expanduser().absolute()
    if supplied_contract.is_symlink():
        raise Stage1ArtifactError("autonomy contract may not be a symlink")
    contract = supplied_contract.resolve(strict=True)
    if contract != expected_contract.resolve(strict=True):
        raise Stage1ArtifactError("autonomy contract is not the canonical workpack contract")
    from .stage1_control import load_autonomy_contract

    contract_value = load_autonomy_contract(contract)
    manifest = root / WORKPACK_MANIFEST_BASENAME
    if manifest.is_symlink() or not manifest.is_file():
        raise Stage1ArtifactError("workpack manifest is missing or a symlink")
    completed = subprocess.run(
        (
            sys.executable,
            str(root / "scripts" / "seal_workpack.py"),
            "verify",
            "--root",
            str(root),
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise Stage1ArtifactError(
            "workpack closed-world verification failed: {}".format(
                completed.stderr.strip() or completed.stdout.strip()
            )
        )
    return {
        "workpack_root": root,
        "contract": contract_value,
        "contract_sha256": canonical_json_sha256(contract_value),
        "workpack_manifest_sha256": sha256_file(manifest),
    }


def _snapshot_evidence(
    spec: StageVerifierSpec, evidence_paths: Mapping[str, Path]
) -> Tuple[Dict[str, Path], Dict[str, Dict[str, Any]]]:
    expected = {name: kind for name, kind in spec.evidence_kinds}
    if set(evidence_paths) != set(expected):
        raise Stage1ArtifactError(
            "{} evidence fields differ; missing={}, unexpected={}".format(
                spec.stage_id,
                sorted(set(expected) - set(evidence_paths)),
                sorted(set(evidence_paths) - set(expected)),
            )
        )
    resolved: Dict[str, Path] = {}
    identities: Dict[str, Dict[str, Any]] = {}
    seen = set()
    for name in sorted(expected):
        kind = expected[name]
        path = _resolve_root_before_symlink(
            Path(evidence_paths[name]), "{} evidence {}".format(spec.stage_id, name), kind=kind
        )
        if path in seen:
            raise Stage1ArtifactError("two evidence labels resolve to the same path")
        seen.add(path)
        resolved[name] = path
        if kind == "file":
            identities[name] = {
                "kind": "file",
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        else:
            files = regular_tree_files(path)
            identities[name] = {
                "kind": "tree",
                "path": str(path),
                "sha256": sha256_tree(path),
                "file_count": len(files),
            }
    return resolved, identities


def _merge_ancestor_reports(
    target: Dict[str, Dict[str, Any]], verified: Mapping[str, Any]
) -> None:
    reports = verified.get("ancestor_reports")
    if not isinstance(reports, Mapping):
        report = verified.get("report")
        if not isinstance(report, dict):
            raise Stage1ArtifactError("verified dependency has no report graph")
        reports = {str(report.get("stage_id")): report}
    for stage_id, report in reports.items():
        if not isinstance(stage_id, str) or not isinstance(report, dict):
            raise Stage1ArtifactError("dependency report graph is malformed")
        previous = target.get(stage_id)
        if previous is not None and previous != report:
            raise Stage1ArtifactError(
                "dependency graph contains two receipts for {}".format(stage_id)
            )
        target[stage_id] = report


def _enforce_training_input_bindings(
    stage_id: str,
    evidence: Mapping[str, Dict[str, Any]],
    ancestor_reports: Mapping[str, Mapping[str, Any]],
) -> None:
    labels = TRAINING_STAGE_RUN_LABELS.get(stage_id)
    if labels is None:
        return
    retained = ancestor_reports.get("retained_seals_audit")
    if not isinstance(retained, Mapping) or not isinstance(
        retained.get("evidence"), Mapping
    ):
        raise Stage1ArtifactError("training stage lacks retained input authority")
    accepted = retained["evidence"]
    required = {
        "train_manifest", "small_cfg_dir", "a1_dir", "audiocraft_dir",
        "musicgen_small_dir",
    }
    if not required.issubset(accepted):
        raise Stage1ArtifactError("retained training input identities are incomplete")
    selected_lr = None
    if stage_id in {"ptc500_training", "small_pilot_training"}:
        lr_report = ancestor_reports.get("lr_decision")
        lr_result = lr_report.get("verifier_result") if isinstance(lr_report, Mapping) else None
        if not isinstance(lr_result, Mapping):
            raise Stage1ArtifactError("training stage lacks selected-LR receipt")
        selected_lr = lr_result.get("selected_learning_rate")
    prior_modes = {"codebook100", "ptc50"}
    observed_lrs = set()
    for label in labels:
        identity = evidence.get(label)
        if not isinstance(identity, Mapping) or not isinstance(identity.get("path"), str):
            raise Stage1ArtifactError("training run evidence identity is malformed")
        run_dir = Path(identity["path"])
        manifest = load_json_strict(run_dir / "run_manifest.json")
        config = manifest.get("config")
        if not isinstance(config, dict):
            raise Stage1ArtifactError("training run config is absent")
        mode = config.get("mode")
        path_bindings = {
            "manifest": "train_manifest",
            "cfg_scale_decision_dir": "small_cfg_dir",
            "audiocraft_root": "audiocraft_dir",
            "student_checkpoint": "musicgen_small_dir",
            "teacher_checkpoint": "musicgen_small_dir",
        }
        for config_field, accepted_label in path_bindings.items():
            value = config.get(config_field)
            expected_kind = "file" if accepted_label == "train_manifest" else "tree"
            if not isinstance(value, str) or str(
                _resolve_root_before_symlink(
                    Path(value), "training {}".format(config_field), kind=expected_kind
                )
            ) != accepted[accepted_label]["path"]:
                raise Stage1ArtifactError(
                    "training run {} is not retained {}".format(
                        config_field, accepted_label
                    )
                )
        if (
            manifest.get("manifest_sha256") != accepted["train_manifest"]["sha256"]
            or accepted["train_manifest"]["sha256"]
            != FROZEN_RETAINED_FILE_SHA256["train_manifest"]
        ):
            raise Stage1ArtifactError("training run used a non-frozen MusicCaps manifest")
        prior_path = config.get("codebook_prior_artifact_dir")
        if mode in prior_modes:
            if not isinstance(prior_path, str) or str(
                _resolve_root_before_symlink(
                    Path(prior_path), "training A1 prior", kind="tree"
                )
            ) != accepted["a1_dir"]["path"]:
                raise Stage1ArtifactError("weighted training run used a different A1 prior")
        elif prior_path is not None:
            raise Stage1ArtifactError("uniform-weight training run consumed an A1 prior")
        learning_rate = config.get("learning_rate")
        if not isinstance(learning_rate, (int, float)) or isinstance(
            learning_rate, bool
        ):
            raise Stage1ArtifactError("training learning rate is malformed")
        observed_lrs.add(float(learning_rate))
        if selected_lr is not None and float(learning_rate) != selected_lr:
            raise Stage1ArtifactError("training run did not use the selected LR")
    if stage_id == "lr_uniform_sweep" and observed_lrs != {
        1.0e-6, 3.0e-6, 1.0e-5
    }:
        raise Stage1ArtifactError("LR sweep run set differs from the frozen grid")
    if stage_id == "small_pilot_training" and len(observed_lrs) != 1:
        raise Stage1ArtifactError("small-pilot methods do not share one selected LR")


def _enforce_dependency_bindings(
    stage_id: str,
    evidence: Mapping[str, Dict[str, Any]],
    ancestor_reports: Mapping[str, Mapping[str, Any]],
    verifier_result: Mapping[str, Any],
) -> None:
    for current_label, (ancestor_stage, ancestor_label) in STAGE_EVIDENCE_BINDINGS.get(
        stage_id, {}
    ).items():
        ancestor = ancestor_reports.get(ancestor_stage)
        ancestor_evidence = ancestor.get("evidence") if isinstance(ancestor, Mapping) else None
        if (
            not isinstance(ancestor_evidence, Mapping)
            or current_label not in evidence
            or ancestor_label not in ancestor_evidence
            or evidence[current_label] != ancestor_evidence[ancestor_label]
        ):
            raise Stage1ArtifactError(
                "{} evidence {} is not the authorized {}:{} identity".format(
                    stage_id, current_label, ancestor_stage, ancestor_label
                )
            )
    _enforce_training_input_bindings(stage_id, evidence, ancestor_reports)
    source_stage = LINEAGE_SOURCE_STAGE.get(stage_id)
    if source_stage is not None:
        source_report = ancestor_reports.get(source_stage)
        source_result = (
            source_report.get("verifier_result")
            if isinstance(source_report, Mapping)
            else None
        )
        if (
            not isinstance(source_result, Mapping)
            or verifier_result.get("training_lineage_anchor") is None
            or verifier_result.get("training_lineage_anchor")
            != source_result.get("training_lineage_anchor")
        ):
            raise Stage1ArtifactError(
                "{} training lineage differs from {}".format(stage_id, source_stage)
            )


def _classify_result(result: Mapping[str, Any]) -> Tuple[str, str]:
    if result.get("gate_passed") is False:
        return "failed", "scientific_gate"
    quality = result.get("measurement_quality")
    if isinstance(quality, str) and quality.startswith("inconclusive"):
        return "pending", "retryable_inconclusive"
    if result.get("status") != "verified" or result.get("gate_passed") is not True:
        raise Stage1ArtifactError("registered verifier did not return a classified result")
    return "passed", "none"


def _normalize_error(exc: BaseException) -> Dict[str, Any]:
    return {
        "schema_version": "ptc-opd-stage1-verifier-error-v1",
        "status": "verifier_error",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _normalize_authority_error(exc: BaseException) -> Dict[str, Any]:
    return {
        "schema_version": "ptc-opd-stage1-authority-integrity-error-v1",
        "status": "authority_integrity_failure",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _receipt_reference(directory: Path, report: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "artifact_dir": str(directory.resolve(strict=True)),
        "stage_id": report["stage_id"],
        "stage_status": report["stage_status"],
        "failure_class": report["attempt"]["failure_class"],
        "attempt_number": report["attempt"]["number"],
        "attempt_kind": report["attempt"]["kind"],
        "verifier_id": report["verifier_id"],
        "report_sha256": sha256_file(directory / RECEIPT_BASENAME),
        "artifact_seal_sha256": sha256_file(directory / "artifact_seal.json"),
    }


def _ledger_reference(directory: Path, ledger: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "artifact_dir": str(directory.resolve(strict=True)),
        "revision": ledger["revision"],
        "report_sha256": sha256_file(directory / LEDGER_BASENAME),
        "artifact_seal_sha256": sha256_file(directory / "artifact_seal.json"),
    }


def _authority_path(controller_root: Path) -> Path:
    return controller_root / AUTHORITY_BASENAME


def _read_controller_authority(controller_root: Path) -> Dict[str, Any]:
    path = _authority_path(controller_root)
    value = load_json_strict(path)
    _exact_fields(
        value,
        ("schema_version", "current_ledger", "pending_receipt"),
        "controller current authority",
    )
    if value.get("schema_version") != AUTHORITY_SCHEMA or not isinstance(
        value.get("current_ledger"), dict
    ):
        raise Stage1ArtifactError("controller current authority is malformed")
    return value


def _write_controller_authority(
    controller_root: Path, value: Mapping[str, Any], *, exclusive: bool
) -> None:
    root = controller_root.expanduser().absolute()
    if root.is_symlink():
        raise Stage1ArtifactError("controller authority root may not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve(strict=True)
    payload = canonical_json_bytes(dict(value)) + b"\n"
    target = _authority_path(root)
    if exclusive:
        descriptor = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return
    temporary = root / ".{}.{}.tmp".format(AUTHORITY_BASENAME, os.getpid())
    descriptor = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(str(temporary), str(target))


@contextlib.contextmanager
def _controller_authority_lock(controller_root: Path) -> Iterable[None]:
    root = controller_root.expanduser().absolute()
    if root.is_symlink():
        raise Stage1ArtifactError("controller authority root may not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve(strict=True)
    descriptor = os.open(
        str(root / ".CONTROLLER_AUTHORITY.lock"),
        os.O_RDWR | os.O_CREAT,
        0o600,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _assert_current_ledger(ledger_dir: Path, ledger: Mapping[str, Any]) -> Dict[str, Any]:
    directory = Path(ledger_dir).expanduser().absolute().resolve(strict=True)
    authority = _read_controller_authority(directory.parent)
    if authority.get("current_ledger") != _ledger_reference(directory, ledger):
        raise Stage1ArtifactError("supplied ledger is stale or belongs to another branch")
    return authority


def _issue_stage_verifier_receipt_locked(
    *,
    stage_id: str,
    evidence_paths: Mapping[str, Path],
    dependency_receipt_dirs: Mapping[str, Path],
    workpack_root: Path,
    contract_path: Path,
    output_dir: Path,
    attempt_number: int,
    attempt_kind: str = "initial",
    ledger_dir: Optional[Path] = None,
) -> Path:
    context = _authority_context(workpack_root, contract_path)
    registry = stage_verifier_registry()
    if stage_id not in registry or not registry[stage_id].supported:
        raise Stage1ArtifactError("stage verifier is unsupported: {}".format(stage_id))
    spec = registry[stage_id]
    if type(attempt_number) is not int or attempt_number <= 0:
        raise Stage1ArtifactError("attempt number must be a positive integer")
    if attempt_kind not in {"initial", "yellow"}:
        raise Stage1ArtifactError("attempt kind must be initial or yellow")
    if ledger_dir is None:
        raise Stage1ArtifactError("every attempt requires the current verified ledger")
    normalized = validate_controller_ledger_authorizations(
        ledger_dir, workpack_root=workpack_root, contract_path=contract_path
    )
    authority = _assert_current_ledger(ledger_dir, normalized["ledger"])
    if authority.get("pending_receipt") is not None:
        raise Stage1ArtifactError("a verifier receipt is already pending controller record")
    stage = normalized["ledger"]["stages"][stage_id]
    if attempt_kind == "initial":
        if attempt_number != 1:
            raise Stage1ArtifactError("initial attempt must be exactly attempt 1")
        if stage["attempts"]:
            raise Stage1ArtifactError("initial attempt already exists")
    else:
        if stage["yellow_retry_count"] >= MAX_YELLOW_RETRIES_PER_STAGE:
            raise Stage1ArtifactError("yellow retry budget exhausted before verifier launch")
        if not stage["attempts"] or stage["status"] != "pending":
            raise Stage1ArtifactError("yellow attempt requires one prior pending initial attempt")
        if attempt_number != len(stage["attempts"]) + 1:
            raise Stage1ArtifactError("yellow attempt number must continue ledger history")

    resolved, before = _snapshot_evidence(spec, evidence_paths)
    if set(dependency_receipt_dirs) != set(spec.dependencies):
        raise Stage1ArtifactError("dependency receipt set differs from controller DAG")
    dependencies: Dict[str, Dict[str, Any]] = {}
    ancestor_reports: Dict[str, Dict[str, Any]] = {}
    receipt_cache: Dict[str, Dict[str, Any]] = {}
    for dependency in spec.dependencies:
        verified = verify_stage_verifier_receipt(
            dependency_receipt_dirs[dependency],
            workpack_root=workpack_root,
            contract_path=contract_path,
            _cache=receipt_cache,
        )
        if verified["report"]["stage_id"] != dependency or verified["report"][
            "stage_status"
        ] != "passed":
            raise Stage1ArtifactError("dependency receipt did not pass: {}".format(dependency))
        if normalized["ledger"]["stages"][dependency].get(
            "terminal_receipt"
        ) != verified["reference"]:
            raise Stage1ArtifactError(
                "dependency receipt is not the current ledger authorization: {}".format(
                    dependency
                )
            )
        dependencies[dependency] = verified["reference"]
        _merge_ancestor_reports(ancestor_reports, verified)

    assert spec.handler is not None
    try:
        result = spec.handler(resolved, context["workpack_root"])
    except Exception as exc:
        result = _normalize_error(exc)
        stage_status = "pending"
        failure_class = "verifier_error"
    else:
        try:
            _enforce_dependency_bindings(stage_id, before, ancestor_reports, result)
        except Exception as exc:
            result = _normalize_authority_error(exc)
            stage_status = "failed"
            failure_class = "authority_integrity"
        else:
            stage_status, failure_class = _classify_result(result)
    _, after = _snapshot_evidence(spec, resolved)
    if before != after:
        raise Stage1ArtifactError("evidence changed while the registered verifier ran")
    report = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "sealed_verifier_receipt",
        "stage_id": stage_id,
        "stage_status": stage_status,
        "verifier_id": spec.verifier_id,
        "contract_sha256": context["contract_sha256"],
        "workpack_manifest_sha256": context["workpack_manifest_sha256"],
        "parent_ledger": _ledger_reference(
            Path(ledger_dir).expanduser().absolute().resolve(strict=True),
            normalized["ledger"],
        ),
        "attempt": {
            "number": attempt_number,
            "kind": attempt_kind,
            "failure_class": failure_class,
        },
        "dependencies": dependencies,
        "evidence": before,
        "evidence_set_sha256": canonical_json_sha256(before),
        "verifier_result": result,
        "verifier_result_sha256": canonical_json_sha256(result),
    }
    published = publish_closed_json_artifact(
        output_dir,
        report_name=RECEIPT_BASENAME,
        report=report,
        seal_schema=RECEIPT_SEAL_SCHEMA,
        seal_status="sealed_verifier_receipt",
    )
    sealed_report = load_json_strict(published / RECEIPT_BASENAME)
    updated_authority = {
        "schema_version": AUTHORITY_SCHEMA,
        "current_ledger": authority["current_ledger"],
        "pending_receipt": _receipt_reference(published, sealed_report),
    }
    _write_controller_authority(
        Path(ledger_dir).expanduser().absolute().resolve(strict=True).parent,
        updated_authority,
        exclusive=False,
    )
    return published


def issue_stage_verifier_receipt(
    *,
    stage_id: str,
    evidence_paths: Mapping[str, Path],
    dependency_receipt_dirs: Mapping[str, Path],
    workpack_root: Path,
    contract_path: Path,
    output_dir: Path,
    attempt_number: int,
    attempt_kind: str = "initial",
    ledger_dir: Optional[Path] = None,
) -> Path:
    if ledger_dir is None:
        raise Stage1ArtifactError("every attempt requires the current verified ledger")
    controller_root = Path(ledger_dir).expanduser().absolute().resolve(strict=True).parent
    with _controller_authority_lock(controller_root):
        return _issue_stage_verifier_receipt_locked(
            stage_id=stage_id,
            evidence_paths=evidence_paths,
            dependency_receipt_dirs=dependency_receipt_dirs,
            workpack_root=workpack_root,
            contract_path=contract_path,
            output_dir=output_dir,
            attempt_number=attempt_number,
            attempt_kind=attempt_kind,
            ledger_dir=ledger_dir,
        )


def verify_stage_verifier_receipt(
    receipt_dir: Path,
    *,
    workpack_root: Path,
    contract_path: Path,
    _seen: Optional[set[str]] = None,
    _cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    supplied = Path(receipt_dir).expanduser().absolute()
    if supplied.is_symlink():
        raise Stage1ArtifactError("receipt artifact root may not be a symlink")
    directory = supplied.resolve(strict=True)
    marker = str(directory)
    cache: Dict[str, Dict[str, Any]] = {} if _cache is None else _cache
    if marker in cache:
        return cache[marker]
    seen = set() if _seen is None else _seen
    if marker in seen:
        raise Stage1ArtifactError("receipt dependency cycle detected")
    seen.add(marker)
    try:
        verify_simple_seal(
            directory,
            seal_name="artifact_seal.json",
            schema_version=RECEIPT_SEAL_SCHEMA,
            status="sealed_verifier_receipt",
            payload_names=(RECEIPT_BASENAME,),
        )
        report = load_json_strict(directory / RECEIPT_BASENAME)
        _exact_fields(
            report,
            (
                "schema_version", "status", "stage_id", "stage_status",
                "verifier_id", "contract_sha256", "workpack_manifest_sha256",
                "parent_ledger", "attempt", "dependencies", "evidence", "evidence_set_sha256",
                "verifier_result", "verifier_result_sha256",
            ),
            "verifier receipt",
        )
        context = _authority_context(workpack_root, contract_path)
        if (
            report.get("schema_version") != RECEIPT_SCHEMA
            or report.get("status") != "sealed_verifier_receipt"
            or report.get("contract_sha256") != context["contract_sha256"]
            or report.get("workpack_manifest_sha256")
            != context["workpack_manifest_sha256"]
        ):
            raise Stage1ArtifactError("receipt authority binding differs")
        registry = stage_verifier_registry()
        stage_id = report.get("stage_id")
        if stage_id not in registry or not registry[stage_id].supported:
            raise Stage1ArtifactError("receipt names an unsupported stage verifier")
        spec = registry[stage_id]
        if report.get("verifier_id") != spec.verifier_id:
            raise Stage1ArtifactError("receipt verifier ID differs from registry")
        parent = report.get("parent_ledger")
        if not isinstance(parent, dict) or set(parent) != {
            "artifact_dir", "revision", "report_sha256", "artifact_seal_sha256"
        }:
            raise Stage1ArtifactError("receipt parent-ledger reference is malformed")
        parent_validated = validate_controller_ledger_authorizations(
            Path(parent["artifact_dir"]),
            workpack_root=workpack_root,
            contract_path=contract_path,
            _receipt_cache=cache,
        )
        if _ledger_reference(
            Path(parent["artifact_dir"]), parent_validated["ledger"]
        ) != parent:
            raise Stage1ArtifactError("receipt parent-ledger binding differs")
        evidence = report.get("evidence")
        if not isinstance(evidence, dict) or report.get(
            "evidence_set_sha256"
        ) != canonical_json_sha256(evidence):
            raise Stage1ArtifactError("receipt evidence-set identity differs")
        paths = {
            name: Path(identity["path"])
            for name, identity in evidence.items()
            if isinstance(identity, dict) and isinstance(identity.get("path"), str)
        }
        resolved, before = _snapshot_evidence(spec, paths)
        if before != evidence:
            raise Stage1ArtifactError("live evidence differs from receipt")
        dependencies = report.get("dependencies")
        if not isinstance(dependencies, dict) or set(dependencies) != set(spec.dependencies):
            raise Stage1ArtifactError("receipt dependency set differs")
        ancestor_reports: Dict[str, Dict[str, Any]] = {}
        for dependency in spec.dependencies:
            reference = dependencies[dependency]
            if not isinstance(reference, dict) or set(reference) != {
                "artifact_dir", "stage_id", "stage_status", "failure_class",
                "attempt_number", "attempt_kind", "verifier_id", "report_sha256",
                "artifact_seal_sha256",
            }:
                raise Stage1ArtifactError("dependency receipt reference is malformed")
            verified_dependency = verify_stage_verifier_receipt(
                Path(reference["artifact_dir"]),
                workpack_root=workpack_root,
                contract_path=contract_path,
                _seen=seen,
                _cache=cache,
            )
            if (
                verified_dependency["reference"] != reference
                or reference["stage_id"] != dependency
                or reference["stage_status"] != "passed"
            ):
                raise Stage1ArtifactError("dependency receipt live verification differs")
            _merge_ancestor_reports(ancestor_reports, verified_dependency)

        attempt = report.get("attempt")
        if not isinstance(attempt, dict) or set(attempt) != {
            "number", "kind", "failure_class"
        }:
            raise Stage1ArtifactError("receipt attempt record is malformed")
        if type(attempt["number"]) is not int or attempt["number"] <= 0:
            raise Stage1ArtifactError("receipt attempt number differs")
        if attempt["kind"] not in {"initial", "yellow"}:
            raise Stage1ArtifactError("receipt attempt kind differs")
        assert spec.handler is not None
        if attempt["failure_class"] == "verifier_error":
            if report.get("stage_status") != "pending":
                raise Stage1ArtifactError("verifier-error receipt classification differs")
            try:
                live_result = spec.handler(resolved, context["workpack_root"])
                _enforce_dependency_bindings(
                    stage_id, before, ancestor_reports, live_result
                )
                _classify_result(live_result)
            except Exception as exc:
                result = _normalize_error(exc)
            else:
                raise Stage1ArtifactError("receipt verifier failure no longer reproduces")
            stage_status = "pending"
            failure_class = "verifier_error"
        elif attempt["failure_class"] == "authority_integrity":
            if report.get("stage_status") != "failed":
                raise Stage1ArtifactError(
                    "authority-integrity receipt classification differs"
                )
            live_result = spec.handler(resolved, context["workpack_root"])
            try:
                _enforce_dependency_bindings(
                    stage_id, before, ancestor_reports, live_result
                )
            except Exception as exc:
                result = _normalize_authority_error(exc)
            else:
                raise Stage1ArtifactError(
                    "receipt authority-integrity failure no longer reproduces"
                )
            stage_status = "failed"
            failure_class = "authority_integrity"
        else:
            result = spec.handler(resolved, context["workpack_root"])
            _enforce_dependency_bindings(stage_id, before, ancestor_reports, result)
            stage_status, failure_class = _classify_result(result)
            if attempt["failure_class"] != failure_class:
                raise Stage1ArtifactError("receipt failure class differs from verifier result")
        _, after = _snapshot_evidence(spec, resolved)
        if before != after:
            raise Stage1ArtifactError("live evidence changed during receipt verification")
        if (
            report.get("stage_status") != stage_status
            or report.get("verifier_result") != result
            or report.get("verifier_result_sha256") != canonical_json_sha256(result)
        ):
            raise Stage1ArtifactError("receipt result differs from live registered verifier")
        verified = {
            "report": report,
            "reference": _receipt_reference(directory, report),
            "artifact_seal_sha256": sha256_file(directory / "artifact_seal.json"),
            "ancestor_reports": {stage_id: report, **ancestor_reports},
        }
        cache[marker] = verified
        return verified
    finally:
        seen.remove(marker)


def _initialize_controller_ledger_locked(
    *, workpack_root: Path, contract_path: Path, output_dir: Path
) -> Path:
    context = _authority_context(workpack_root, contract_path)
    controller_root = Path(output_dir).expanduser().absolute().parent
    if _authority_path(controller_root).exists() or _authority_path(
        controller_root
    ).is_symlink():
        raise Stage1ArtifactError("controller ledger was already initialized")
    stages = {
        node["id"]: {
            "status": "pending",
            "yellow_retry_count": 0,
            "attempts": [],
            "terminal_receipt": None,
            "stop_reason": None,
        }
        for node in _stage_graph()
    }
    ledger = {
        "schema_version": LEDGER_SCHEMA,
        "revision": 0,
        "previous_ledger": None,
        "scientific_config_sha256": context["contract_sha256"],
        "workpack_manifest_sha256": context["workpack_manifest_sha256"],
        "stages": stages,
    }
    published = publish_closed_json_artifact(
        output_dir,
        report_name=LEDGER_BASENAME,
        report=ledger,
        seal_schema=LEDGER_SEAL_SCHEMA,
        seal_status="active_controller_ledger",
    )
    _write_controller_authority(
        published.parent,
        {
            "schema_version": AUTHORITY_SCHEMA,
            "current_ledger": _ledger_reference(published, ledger),
            "pending_receipt": None,
        },
        exclusive=True,
    )
    return published


def initialize_controller_ledger(
    *, workpack_root: Path, contract_path: Path, output_dir: Path
) -> Path:
    controller_root = Path(output_dir).expanduser().absolute().parent
    with _controller_authority_lock(controller_root):
        return _initialize_controller_ledger_locked(
            workpack_root=workpack_root,
            contract_path=contract_path,
            output_dir=output_dir,
        )


def validate_controller_ledger_authorizations(
    ledger_dir: Path,
    *,
    workpack_root: Path,
    contract_path: Path,
    _receipt_cache: Optional[Dict[str, Dict[str, Any]]] = None,
    _ledger_seen: Optional[set[str]] = None,
) -> Dict[str, Any]:
    context = _authority_context(workpack_root, contract_path)
    supplied = Path(ledger_dir).expanduser().absolute()
    if supplied.is_symlink():
        raise Stage1ArtifactError("controller ledger root may not be a symlink")
    directory = supplied.resolve(strict=True)
    ledger_seen = set() if _ledger_seen is None else _ledger_seen
    marker = str(directory)
    if marker in ledger_seen:
        raise Stage1ArtifactError("controller ledger chain contains a cycle")
    ledger_seen.add(marker)
    verify_simple_seal(
        directory,
        seal_name="artifact_seal.json",
        schema_version=LEDGER_SEAL_SCHEMA,
        status="active_controller_ledger",
        payload_names=(LEDGER_BASENAME,),
    )
    ledger = load_json_strict(directory / LEDGER_BASENAME)
    _exact_fields(
        ledger,
        (
            "schema_version", "revision", "scientific_config_sha256",
            "workpack_manifest_sha256", "previous_ledger", "stages",
        ),
        "controller authority ledger",
    )
    if (
        ledger.get("schema_version") != LEDGER_SCHEMA
        or type(ledger.get("revision")) is not int
        or ledger.get("revision") < 0
        or ledger.get("scientific_config_sha256") != context["contract_sha256"]
        or ledger.get("workpack_manifest_sha256") != context["workpack_manifest_sha256"]
    ):
        raise Stage1ArtifactError("controller ledger authority binding differs")
    previous_reference = ledger.get("previous_ledger")
    previous_validated: Optional[Dict[str, Any]] = None
    if ledger["revision"] == 0:
        if previous_reference is not None:
            raise Stage1ArtifactError("revision zero may not name a previous ledger")
    else:
        if not isinstance(previous_reference, dict) or set(previous_reference) != {
            "artifact_dir", "revision", "report_sha256", "artifact_seal_sha256"
        }:
            raise Stage1ArtifactError("controller previous-ledger reference is malformed")
        previous_path = Path(previous_reference["artifact_dir"])
        if previous_path.expanduser().absolute().parent != directory.parent:
            raise Stage1ArtifactError("controller ledger revision changed authority root")
        previous_validated = validate_controller_ledger_authorizations(
            previous_path,
            workpack_root=workpack_root,
            contract_path=contract_path,
            _receipt_cache=_receipt_cache,
            _ledger_seen=ledger_seen,
        )
        previous_ledger = previous_validated["ledger"]
        if (
            _ledger_reference(previous_path, previous_ledger) != previous_reference
            or previous_ledger["revision"] + 1 != ledger["revision"]
        ):
            raise Stage1ArtifactError("controller ledger revision linkage differs")
    graph = {node["id"]: tuple(node["depends_on"]) for node in _stage_graph()}
    stages = ledger.get("stages")
    if not isinstance(stages, Mapping) or set(stages) != set(graph):
        raise Stage1ArtifactError("controller ledger stage set differs from DAG")
    legacy: Dict[str, Dict[str, Any]] = {}
    verified_terminal: Dict[str, Dict[str, Any]] = {}
    receipt_cache = {} if _receipt_cache is None else _receipt_cache
    for stage_id in graph:
        entry = stages[stage_id]
        if not isinstance(entry, Mapping):
            raise Stage1ArtifactError("ledger stage entry must be an object")
        _exact_fields(
            entry,
            (
                "status", "yellow_retry_count", "attempts", "terminal_receipt",
                "stop_reason",
            ),
            "ledger stage entry",
        )
        attempts = entry.get("attempts")
        if not isinstance(attempts, list):
            raise Stage1ArtifactError("ledger attempt history must be a list")
        yellow_count = 0
        verified_attempts = []
        for expected_number, reference in enumerate(attempts, start=1):
            if not isinstance(reference, dict) or reference.get("attempt_number") != expected_number:
                raise Stage1ArtifactError("ledger attempt sequence is not contiguous")
            receipt = verify_stage_verifier_receipt(
                Path(reference["artifact_dir"]),
                workpack_root=workpack_root,
                contract_path=contract_path,
                _cache=receipt_cache,
            )
            if receipt["reference"] != reference or reference["stage_id"] != stage_id:
                raise Stage1ArtifactError("ledger attempt receipt reference differs")
            if expected_number == 1:
                report = receipt["report"]
                if report["attempt"]["kind"] != "initial":
                    raise Stage1ArtifactError("first ledger attempt must be initial")
            elif receipt["report"]["attempt"]["kind"] != "yellow":
                raise Stage1ArtifactError("all attempts after initial must be yellow")
            if receipt["report"]["attempt"]["kind"] == "yellow":
                yellow_count += 1
            verified_attempts.append(receipt)
        if yellow_count > MAX_YELLOW_RETRIES_PER_STAGE or entry.get(
            "yellow_retry_count"
        ) != yellow_count:
            raise Stage1ArtifactError("ledger yellow retry budget/count differs")
        status = entry.get("status")
        terminal = entry.get("terminal_receipt")
        if status == "pending":
            if (
                terminal is not None
                or entry.get("stop_reason") is not None
                or (attempts and attempts[-1]["stage_status"] != "pending")
            ):
                raise Stage1ArtifactError("pending ledger stage has terminal authorization")
            legacy[stage_id] = {"status": "pending", "artifact_sha256": None}
        elif status in TERMINAL_STAGE_STATUSES:
            exhausted = entry.get("stop_reason") == "yellow_retry_budget_exhausted"
            if exhausted:
                if (
                    status != "failed"
                    or yellow_count != MAX_YELLOW_RETRIES_PER_STAGE
                    or not attempts
                    or terminal != attempts[-1]
                    or terminal["stage_status"] != "pending"
                ):
                    raise Stage1ArtifactError("yellow-budget terminal stage differs")
            elif (
                entry.get("stop_reason") is not None
                or not attempts
                or terminal != attempts[-1]
                or terminal["stage_status"] != status
            ):
                raise Stage1ArtifactError("terminal ledger stage is not bound to its last receipt")
            verified_terminal[stage_id] = terminal
            legacy[stage_id] = {
                "status": status,
                "artifact_sha256": terminal["artifact_seal_sha256"],
            }
        else:
            raise Stage1ArtifactError("ledger stage status differs")
    for stage_id, terminal in verified_terminal.items():
        report = load_json_strict(Path(terminal["artifact_dir"]) / RECEIPT_BASENAME)
        expected_dependencies = graph[stage_id]
        for dependency in expected_dependencies:
            if dependency not in verified_terminal or verified_terminal[dependency][
                "stage_status"
            ] != "passed":
                raise Stage1ArtifactError("terminal stage lacks a passed dependency")
            if report["dependencies"][dependency] != verified_terminal[dependency]:
                raise Stage1ArtifactError("terminal stage dependency receipt is not current")
    if previous_validated is not None:
        previous_stages = previous_validated["ledger"]["stages"]
        changed = [
            stage_id
            for stage_id in graph
            if previous_stages[stage_id] != stages[stage_id]
        ]
        if len(changed) != 1:
            raise Stage1ArtifactError("ledger revision must append exactly one stage attempt")
        changed_stage = changed[0]
        before_entry = previous_stages[changed_stage]
        after_entry = stages[changed_stage]
        if (
            after_entry["attempts"][:-1] != before_entry["attempts"]
            or len(after_entry["attempts"]) != len(before_entry["attempts"]) + 1
        ):
            raise Stage1ArtifactError("ledger revision did not append one receipt")
        appended = after_entry["attempts"][-1]
        appended_report = load_json_strict(
            Path(appended["artifact_dir"]) / RECEIPT_BASENAME
        )
        if appended_report.get("parent_ledger") != previous_reference:
            raise Stage1ArtifactError("appended receipt is bound to a stale ledger")
    ledger_seen.remove(marker)
    return {"ledger": dict(ledger), "legacy_stages": legacy}


def _record_controller_receipt_locked(
    ledger_dir: Path,
    receipt_dir: Path,
    *,
    workpack_root: Path,
    contract_path: Path,
    output_dir: Path,
) -> Path:
    validated_result = validate_controller_ledger_authorizations(
        ledger_dir, workpack_root=workpack_root, contract_path=contract_path
    )
    validated = validated_result["ledger"]
    authority = _assert_current_ledger(ledger_dir, validated)
    receipt = verify_stage_verifier_receipt(
        receipt_dir, workpack_root=workpack_root, contract_path=contract_path
    )
    report = receipt["report"]
    stage_id = report["stage_id"]
    entry = validated["stages"][stage_id]
    if authority.get("pending_receipt") != receipt["reference"]:
        raise Stage1ArtifactError("receipt is not the one pending in controller authority")
    current_reference = _ledger_reference(
        Path(ledger_dir).expanduser().absolute().resolve(strict=True), validated
    )
    if report.get("parent_ledger") != current_reference:
        raise Stage1ArtifactError("receipt was issued from a stale controller ledger")
    if Path(output_dir).expanduser().absolute().parent.resolve(strict=True) != Path(
        ledger_dir
    ).expanduser().absolute().resolve(strict=True).parent:
        raise Stage1ArtifactError("next ledger revision must remain in one authority root")
    if entry["status"] != "pending":
        raise Stage1ArtifactError("cannot append to a terminal ledger stage")
    reference = receipt["reference"]
    if reference["attempt_number"] != len(entry["attempts"]) + 1:
        raise Stage1ArtifactError("receipt attempt number does not continue ledger history")
    attempt_kind = report["attempt"]["kind"]
    if not entry["attempts"] and attempt_kind != "initial":
        raise Stage1ArtifactError("first recorded attempt must be initial")
    if entry["attempts"] and attempt_kind != "yellow":
        raise Stage1ArtifactError("retry attempts must be classified yellow")
    if attempt_kind == "yellow" and entry["yellow_retry_count"] >= MAX_YELLOW_RETRIES_PER_STAGE:
        raise Stage1ArtifactError("third yellow retry is forbidden")
    result = json.loads(json.dumps(validated))
    result["previous_ledger"] = current_reference
    target = result["stages"][stage_id]
    target["attempts"].append(reference)
    if attempt_kind == "yellow":
        target["yellow_retry_count"] += 1
    if reference["stage_status"] in TERMINAL_STAGE_STATUSES:
        target["status"] = reference["stage_status"]
        target["terminal_receipt"] = reference
    elif target["yellow_retry_count"] >= MAX_YELLOW_RETRIES_PER_STAGE:
        target["status"] = "failed"
        target["terminal_receipt"] = reference
        target["stop_reason"] = "yellow_retry_budget_exhausted"
    result["revision"] += 1
    published = publish_closed_json_artifact(
        output_dir,
        report_name=LEDGER_BASENAME,
        report=result,
        seal_schema=LEDGER_SEAL_SCHEMA,
        seal_status="active_controller_ledger",
    )
    validate_controller_ledger_authorizations(
        published, workpack_root=workpack_root, contract_path=contract_path
    )
    _write_controller_authority(
        published.parent,
        {
            "schema_version": AUTHORITY_SCHEMA,
            "current_ledger": _ledger_reference(published, result),
            "pending_receipt": None,
        },
        exclusive=False,
    )
    return published


def record_controller_receipt(
    ledger_dir: Path,
    receipt_dir: Path,
    *,
    workpack_root: Path,
    contract_path: Path,
    output_dir: Path,
) -> Path:
    controller_root = Path(ledger_dir).expanduser().absolute().resolve(strict=True).parent
    with _controller_authority_lock(controller_root):
        return _record_controller_receipt_locked(
            ledger_dir,
            receipt_dir,
            workpack_root=workpack_root,
            contract_path=contract_path,
            output_dir=output_dir,
        )


def validate_current_controller_ledger_authorizations(
    ledger_dir: Path, *, workpack_root: Path, contract_path: Path
) -> Dict[str, Any]:
    """Validate a ledger and require it to be the sole current action authority."""

    validated = validate_controller_ledger_authorizations(
        ledger_dir, workpack_root=workpack_root, contract_path=contract_path
    )
    authority = _assert_current_ledger(ledger_dir, validated["ledger"])
    if authority.get("pending_receipt") is not None:
        raise Stage1ArtifactError(
            "controller has a pending receipt that must be recorded before next action"
        )
    return validated


__all__ = [
    "CONTRACT_RELATIVE_PATH",
    "LEDGER_BASENAME",
    "LEDGER_SCHEMA",
    "LEDGER_SEAL_SCHEMA",
    "MAX_YELLOW_RETRIES_PER_STAGE",
    "RECEIPT_BASENAME",
    "RECEIPT_SCHEMA",
    "StageVerifierSpec",
    "initialize_controller_ledger",
    "issue_stage_verifier_receipt",
    "record_controller_receipt",
    "stage_verifier_catalog",
    "stage_verifier_registry",
    "validate_current_controller_ledger_authorizations",
    "validate_controller_ledger_authorizations",
    "verify_stage_verifier_receipt",
]
