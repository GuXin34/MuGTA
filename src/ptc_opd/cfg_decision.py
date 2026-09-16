"""Strict consumer for the immutable development CFG-scale decision artifact."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Tuple


DECISION_SCHEMA_VERSION = "ptc-opd-cfg-scale-decision-v3"
DECISION_SIDECAR_SCHEMA_VERSION = "ptc-opd-cfg-scale-decision-sidecar-v3"
DECISION_FILENAME = "cfg_scale_decision.json"
DECISION_SIDECAR_FILENAME = "cfg_scale_decision.sha256.json"
ALLOWED_CFG_SCALES: Tuple[float, ...] = (2.0, 3.0, 5.0)
REQUIRED_INPUT_HASHES = (
    "generation_run_sha256",
    "generation_samples_jsonl_sha256",
    "generation_artifact_seal_sha256",
    "generation_scientific_config_sha256",
    "scores_jsonl_sha256",
    "evaluator_provenance_sha256",
    "external_evaluation_artifact_seal_sha256",
    "quality_artifact_seal_sha256",
    "quality_scores_jsonl_sha256",
    "quality_evaluator_provenance_sha256",
    "external_evaluation_protocol_sha256",
    "external_evaluation_offline_environment_sha256",
)
REQUIRED_GENERATION_IDENTITY = (
    "model_id",
    "lm_parameter_dtype",
    "compression_parameter_dtype",
    "conditioner_parameter_dtype",
    "conditioner_compute_dtype",
    "lm_generation_compute_dtype",
    "compression_decode_compute_dtype",
    "manifest_sha256",
    "checkpoint_sha256",
    "state_dict_sha256",
    "compression_state_dict_sha256",
    "audiocraft_base_commit",
    "audiocraft_source_sha256",
    "audiocraft_lm_sha256",
    "loaded_t5_identity_sha256",
)
PINNED_AUDIOCRAFT_BASE_COMMIT = "896ec7c47f5e5d1e5aa1e4b260c4405328bf009d"
ALLOWED_MODEL_IDS: Tuple[str, ...] = (
    "facebook/musicgen-small",
    "facebook/musicgen-medium",
)
EXPECTED_PRECISION_IDENTITY = {
    "lm_parameter_dtype": "torch.float32",
    "compression_parameter_dtype": "torch.float32",
    "conditioner_parameter_dtype": "torch.float32",
    "conditioner_compute_dtype": "torch.float32",
    "lm_generation_compute_dtype": "torch.bfloat16",
    "compression_decode_compute_dtype": "torch.float32",
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_QUALITY_FORMULA = {
    "q_dev": "0.5*delta_z_muq_mi + 0.5*delta_z_aesthetic",
    "aesthetic": "0.5*delta_z_audiobox_ce + 0.5*delta_z_audiobox_pq",
    "expanded": "0.5*delta_z_muq_mi + 0.25*delta_z_audiobox_ce + 0.25*delta_z_audiobox_pq",
    "standardization": "paired raw delta divided by no_cfg base sample SD",
}
EXPECTED_GUARDRAILS = {
    "q_dev_strictly_positive": True,
    "paired_bootstrap_probability_positive_gte": 0.90,
    "music_clap_delta_base_sd_gte": -0.10,
}


@dataclass(frozen=True)
class CFGScaleDecision:
    directory: str
    status: str
    selected_cfg_scale: Optional[float]
    decision_file_sha256: str
    decision_payload_sha256: str
    scientific_config_sha256: str
    input_hashes: Mapping[str, str]
    generation_identity: Mapping[str, str]


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("CFG decision member is missing or not regular: {}".format(path))
    def reject_constant(value: str) -> None:
        raise ValueError("CFG decision JSON contains forbidden {}".format(value))

    def unique_object(pairs: Tuple[Tuple[str, Any], ...]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("CFG decision JSON contains duplicate key {!r}".format(key))
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as stream:
        value = json.load(
            stream,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    if not isinstance(value, dict):
        raise ValueError("CFG decision member must be a JSON object: {}".format(path))
    return value


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("{} must be a lowercase SHA-256 hex string".format(name))
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("{} must be numeric".format(name))
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError("{} must be finite".format(name))
    return resolved


def verify_cfg_scale_decision(
    path: Path, *, require_selected: bool = True
) -> CFGScaleDecision:
    """Verify hashes, rule-consistency, and selected status of one decision.

    This is intentionally stricter than a generic artifact inspector: a stop
    decision is valid provenance but is *not* consumable by A2 or training and
    therefore raises here.
    """

    supplied = path.expanduser()
    if supplied.is_symlink():
        raise ValueError("CFG decision directory must not be a symlink")
    directory = supplied.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("CFG decision path must be a regular directory")
    entries = list(directory.iterdir())
    observed_members = {member.name for member in entries}
    expected_members = {DECISION_FILENAME, DECISION_SIDECAR_FILENAME}
    if observed_members != expected_members:
        raise ValueError(
            "CFG decision directory members differ; missing={}, unexpected={}".format(
                sorted(expected_members - observed_members),
                sorted(observed_members - expected_members),
            )
        )
    if any(member.is_symlink() or not member.is_file() for member in entries):
        raise ValueError("CFG decision members must be regular files")

    decision_path = directory / DECISION_FILENAME
    sidecar_path = directory / DECISION_SIDECAR_FILENAME
    decision = _load_json_object(decision_path)
    sidecar = _load_json_object(sidecar_path)
    if set(decision) != {
        "schema_version",
        "status",
        "selected_cfg_scale",
        "scientific_config",
        "scientific_config_sha256",
        "input_hashes",
        "generation_identity",
        "base_standardization",
        "candidates",
        "decision_payload_sha256",
    }:
        raise ValueError("CFG decision top-level field set mismatch")
    if set(sidecar) != {
        "schema_version",
        "path",
        "sha256",
        "decision_payload_sha256",
    }:
        raise ValueError("CFG decision sidecar field set mismatch")
    if decision.get("schema_version") != DECISION_SCHEMA_VERSION:
        raise ValueError("CFG decision schema mismatch")
    if sidecar.get("schema_version") != DECISION_SIDECAR_SCHEMA_VERSION:
        raise ValueError("CFG decision sidecar schema mismatch")
    if sidecar.get("path") != DECISION_FILENAME:
        raise ValueError("CFG decision sidecar path mismatch")
    observed_file_hash = sha256_file(decision_path)
    if _require_sha256(sidecar.get("sha256"), "sidecar.sha256") != observed_file_hash:
        raise ValueError("CFG decision file hash mismatch")

    payload_hash = _require_sha256(
        decision.get("decision_payload_sha256"), "decision_payload_sha256"
    )
    without_payload_hash = dict(decision)
    without_payload_hash.pop("decision_payload_sha256")
    if canonical_json_sha256(without_payload_hash) != payload_hash:
        raise ValueError("CFG decision canonical payload hash mismatch")
    if sidecar.get("decision_payload_sha256") != payload_hash:
        raise ValueError("CFG decision sidecar payload hash mismatch")

    status = decision.get("status")
    if status not in {"selected", "stopped_no_eligible_scale"}:
        raise ValueError("CFG decision status is not recognized")
    selected: Optional[float]
    if status == "selected":
        selected = _finite_number(
            decision.get("selected_cfg_scale"), "selected_cfg_scale"
        )
        if selected not in ALLOWED_CFG_SCALES:
            raise ValueError("selected CFG scale is outside the frozen candidates")
    else:
        if decision.get("selected_cfg_scale") is not None:
            raise ValueError("stopped CFG decision must not expose a scale")
        selected = None

    scientific_config = decision.get("scientific_config")
    if not isinstance(scientific_config, dict):
        raise ValueError("CFG decision has no scientific_config object")
    if set(scientific_config) != {
        "schema_version",
        "evaluation_manifest",
        "candidates",
        "base_anchor",
        "quality_formula",
        "guardrails",
        "paired_prompt_bootstrap",
        "selection",
        "tie_break",
    }:
        raise ValueError("CFG decision scientific-config field set mismatch")
    if scientific_config.get("schema_version") != DECISION_SCHEMA_VERSION:
        raise ValueError("CFG decision embedded scientific-config schema mismatch")
    scientific_hash = _require_sha256(
        decision.get("scientific_config_sha256"), "scientific_config_sha256"
    )
    if canonical_json_sha256(scientific_config) != scientific_hash:
        raise ValueError("CFG decision scientific_config hash mismatch")
    if scientific_config.get("candidates") != list(ALLOWED_CFG_SCALES):
        raise ValueError("CFG decision candidate set mismatch")
    if scientific_config.get("evaluation_manifest") != "dev.full.jsonl":
        raise ValueError("CFG decision development-manifest contract mismatch")
    if scientific_config.get("base_anchor") != "no_cfg":
        raise ValueError("CFG decision base-anchor contract mismatch")
    if scientific_config.get("quality_formula") != EXPECTED_QUALITY_FORMULA:
        raise ValueError("CFG decision quality formula mismatch")
    if scientific_config.get("guardrails") != EXPECTED_GUARDRAILS:
        raise ValueError("CFG decision guardrail contract mismatch")
    if scientific_config.get("selection") != "highest eligible q_dev":
        raise ValueError("CFG decision selection rule mismatch")
    if scientific_config.get("tie_break") != "lower cfg scale only on exact numerical q_dev equality":
        raise ValueError("CFG decision tie-break rule mismatch")
    bootstrap = scientific_config.get("paired_prompt_bootstrap")
    if bootstrap != {
        "seed": 4703,
        "replicates": 10_000,
        "confidence_interval": 0.95,
    }:
        raise ValueError("CFG decision bootstrap contract mismatch")

    input_hashes = decision.get("input_hashes")
    if not isinstance(input_hashes, dict) or set(input_hashes) != set(REQUIRED_INPUT_HASHES):
        raise ValueError("CFG decision input hash set mismatch")
    normalized_inputs = {
        name: _require_sha256(input_hashes.get(name), "input_hashes.{}".format(name))
        for name in REQUIRED_INPUT_HASHES
    }
    generation_identity = decision.get("generation_identity")
    if not isinstance(generation_identity, dict) or set(generation_identity) != set(
        REQUIRED_GENERATION_IDENTITY
    ):
        raise ValueError("CFG decision generation identity set mismatch")
    normalized_generation = {
        name: str(generation_identity[name]) for name in REQUIRED_GENERATION_IDENTITY
    }
    if normalized_generation["model_id"] not in ALLOWED_MODEL_IDS:
        raise ValueError("generation model_id is outside MusicGen small/medium")
    for name, expected in EXPECTED_PRECISION_IDENTITY.items():
        if normalized_generation[name] != expected:
            raise ValueError("generation precision identity mismatch for {}".format(name))
    if normalized_generation["audiocraft_base_commit"] != PINNED_AUDIOCRAFT_BASE_COMMIT:
        raise ValueError("generation AudioCraft commit is not the pinned base")
    enum_names = {"model_id", "audiocraft_base_commit"} | set(
        EXPECTED_PRECISION_IDENTITY
    )
    for name, value in normalized_generation.items():
        if name not in enum_names:
            _require_sha256(value, "generation_identity.{}".format(name))

    base_standardization = decision.get("base_standardization")
    required_metrics = {"muq_mi", "audiobox_ce", "audiobox_pq", "music_clap"}
    if not isinstance(base_standardization, dict) or set(base_standardization) != required_metrics:
        raise ValueError("CFG base-standardization metric set mismatch")
    for metric in sorted(required_metrics):
        statistic = base_standardization[metric]
        if not isinstance(statistic, dict) or set(statistic) != {"mean", "sample_sd"}:
            raise ValueError("CFG base-standardization fields mismatch for {}".format(metric))
        _finite_number(statistic["mean"], "base_standardization.{}.mean".format(metric))
        if _finite_number(
            statistic["sample_sd"],
            "base_standardization.{}.sample_sd".format(metric),
        ) <= 0.0:
            raise ValueError("CFG base sample SD must be strictly positive")

    candidates = decision.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(ALLOWED_CFG_SCALES):
        raise ValueError("CFG decision must contain exactly three candidate results")
    by_scale: Dict[float, Mapping[str, Any]] = {}
    for index, item in enumerate(candidates):
        if not isinstance(item, dict):
            raise ValueError("CFG candidate {} is not an object".format(index))
        scale = _finite_number(item.get("cfg_scale"), "candidate.cfg_scale")
        if scale in by_scale or scale not in ALLOWED_CFG_SCALES:
            raise ValueError("CFG candidate scales are duplicate or outside the freeze")
        if set(item) != {
            "cfg_scale",
            "q_dev",
            "q_dev_ci95",
            "paired_bootstrap_probability_positive",
            "music_clap_delta_base_sd",
            "component_delta_base_sd",
            "eligible",
        }:
            raise ValueError("CFG candidate fields differ from the frozen schema")
        if not isinstance(item.get("eligible"), bool):
            raise ValueError("CFG candidate eligible flag must be boolean")
        q_dev = _finite_number(item.get("q_dev"), "candidate.q_dev")
        probability = _finite_number(
            item.get("paired_bootstrap_probability_positive"),
            "candidate.paired_bootstrap_probability_positive",
        )
        clap_delta = _finite_number(
            item.get("music_clap_delta_base_sd"),
            "candidate.music_clap_delta_base_sd",
        )
        if not 0.0 <= probability <= 1.0:
            raise ValueError("CFG candidate bootstrap probability must lie in [0,1]")
        interval = item.get("q_dev_ci95")
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("CFG candidate q_dev_ci95 must contain two values")
        low = _finite_number(interval[0], "candidate.q_dev_ci95[0]")
        high = _finite_number(interval[1], "candidate.q_dev_ci95[1]")
        if low > high:
            raise ValueError("CFG candidate confidence interval is reversed")
        components = item.get("component_delta_base_sd")
        required_components = required_metrics
        if not isinstance(components, dict) or set(components) != required_components:
            raise ValueError("CFG candidate component metric set mismatch")
        for name in sorted(required_components):
            _finite_number(components[name], "candidate.component.{}".format(name))
        formula_q = (
            0.5 * float(components["muq_mi"])
            + 0.25 * float(components["audiobox_ce"])
            + 0.25 * float(components["audiobox_pq"])
        )
        if not math.isclose(q_dev, formula_q, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError("CFG candidate q_dev violates the frozen quality formula")
        if not math.isclose(
            clap_delta,
            float(components["music_clap"]),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("CFG candidate music-CLAP guardrail statistic mismatch")
        expected_eligible = (
            q_dev > 0.0
            and probability >= 0.90
            and clap_delta >= -0.10
        )
        if item["eligible"] is not expected_eligible:
            raise ValueError("CFG candidate eligible flag violates frozen guardrails")
        by_scale[scale] = item
    if tuple(sorted(by_scale)) != ALLOWED_CFG_SCALES:
        raise ValueError("CFG candidate results do not cover 2/3/5")
    eligible = [item for item in by_scale.values() if item["eligible"]]
    if status == "selected":
        if not eligible:
            raise ValueError("selected decision has no eligible candidate")
        recomputed = sorted(
            eligible,
            key=lambda item: (
                -_finite_number(item["q_dev"], "candidate.q_dev"),
                _finite_number(item["cfg_scale"], "candidate.cfg_scale"),
            ),
        )[0]
        if float(recomputed["cfg_scale"]) != selected:
            raise ValueError("selected CFG scale is inconsistent with frozen selection rule")
    elif eligible:
        raise ValueError("stopped CFG decision still contains an eligible candidate")
    if require_selected and status != "selected":
        raise ValueError(
            "CFG development gate did not select an eligible teacher scale"
        )

    return CFGScaleDecision(
        directory=str(directory),
        status=str(status),
        selected_cfg_scale=selected,
        decision_file_sha256=observed_file_hash,
        decision_payload_sha256=payload_hash,
        scientific_config_sha256=scientific_hash,
        input_hashes=normalized_inputs,
        generation_identity=normalized_generation,
    )


__all__ = [
    "ALLOWED_CFG_SCALES",
    "CFGScaleDecision",
    "DECISION_FILENAME",
    "DECISION_SCHEMA_VERSION",
    "DECISION_SIDECAR_FILENAME",
    "DECISION_SIDECAR_SCHEMA_VERSION",
    "REQUIRED_INPUT_HASHES",
    "REQUIRED_GENERATION_IDENTITY",
    "canonical_json_sha256",
    "sha256_file",
    "verify_cfg_scale_decision",
]
