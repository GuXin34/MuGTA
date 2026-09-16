"""Pure contracts and verifier for trained Stage-1 audio generation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .stage1_artifact import (
    Stage1ArtifactError,
    artifact_member,
    canonical_json_sha256,
    load_json_strict,
    require_finite_number,
    require_sha256,
    sha256_file,
    sha256_tree,
)


GENERATION_SCHEMA_VERSION = "ptc-opd-stage1-generation-v1"
SAMPLE_SCHEMA_VERSION = "ptc-opd-stage1-generation-sample-v1"
SEAL_SCHEMA_VERSION = "ptc-opd-stage1-generation-seal-v1"
CONFIG_NAME = "scientific_config.json"
SAMPLES_NAME = "samples.jsonl"
SEAL_NAME = "artifact_seal.json"
EVAL_MANIFEST_NAME = "pilot_eval.dev.jsonl"
EVAL_MANIFEST_METADATA_NAME = "pilot_eval_manifest.json"
EVAL_MANIFEST_SCHEMA_VERSION = "ptc-opd-pilot-eval-manifest-v1"
EVAL_MANIFEST_SEAL_SCHEMA_VERSION = "ptc-opd-pilot-eval-manifest-seal-v1"
EVAL_RECORD_SCHEMA_VERSION = "ptc-opd-pilot-eval-record-v1"
EVAL_SELECTION = {
    "algorithm": "lowest_sha256_then_output_sorted_by_sample_id",
    "count": 128,
    "namespace": "ptc-opd-small-pilot-eval-v1",
    "seed": 2701,
}
GENERATION_SEEDS: Tuple[int, int] = (31001, 31002)
SEED_NAMESPACE = "ptc-opd-small-pilot-generation-v1"
EXPECTED_PROMPTS = 128
EXPECTED_SAMPLE_RATE = 32_000
EXPECTED_FRAMES = 320_000
EXPECTED_CHANNELS = 1
MIN_CATASTROPHIC_RMS = 1.0e-7
GENERATION_CONFIG_FIELDS = {
    "schema_version",
    "model_id",
    "source_kind",
    "condition_id",
    "method",
    "train_seed",
    "learning_rate",
    "checkpoint_step",
    "base_checkpoint",
    "trained_checkpoint",
    "stage1_run",
    "stage1_run_tree_sha256",
    "audiocraft_source_sha256",
    "cfg_decision",
    "eval_manifest_artifact_seal_sha256",
    "eval_manifest_sha256",
    "prompt_count",
    "generation_seeds",
    "derived_seed_namespace",
    "duration_seconds",
    "sample_rate",
    "codec_frame_rate",
    "token_frames",
    "sampling",
    "precision",
    "student_inference_cfg",
    "teacher_cfg_scale",
    "no_loudness_normalization",
    "replace_failed_audio",
    "best_of_n",
    "runtime_identity",
    "base_lm_state_sha256",
    "loaded_trained_lm_state_sha256",
}
SAMPLE_FIELDS = {
    "schema_version",
    "sample_id",
    "prompt_sha256",
    "generation_seed",
    "derived_seed",
    "condition_id",
    "method",
    "checkpoint_step",
    "path",
    "audio_sha256",
    "audio_frames",
    "audio_channels",
    "audio_sample_rate",
    "audio_subtype",
    "audio_peak_abs",
    "audio_rms",
    "scientific_config_sha256",
}


def derive_generation_seed(sample_id: str, generation_seed: int) -> int:
    if generation_seed not in GENERATION_SEEDS:
        raise Stage1ArtifactError("generation seed is outside the frozen pair")
    if not isinstance(sample_id, str) or not sample_id:
        raise Stage1ArtifactError("sample_id must be nonempty")
    payload = "{}|{}|{}".format(
        SEED_NAMESPACE, generation_seed, sample_id
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _strict_json_line(raw: str, label: str) -> Dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise Stage1ArtifactError("{} contains non-finite {}".format(label, value))

    def unique(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Stage1ArtifactError(
                    "{} contains duplicate key {!r}".format(label, key)
                )
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise Stage1ArtifactError("invalid JSONL record: {}".format(label)) from exc
    if not isinstance(value, dict):
        raise Stage1ArtifactError("{} must be a JSON object".format(label))
    return value


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise Stage1ArtifactError("JSONL is missing or not regular: {}".format(path))
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.endswith("\n") or not raw.strip():
                raise Stage1ArtifactError(
                    "{}:{} is blank or not newline terminated".format(path, line_number)
                )
            yield _strict_json_line(raw, "{}:{}".format(path, line_number))


def _load_test_prompts_manifest_lax(
    directory: Path,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Post-Milestone-7 lax loader for test_prompts artifacts (Ruling #9 §6).

    Bypasses the sealed 128-prompt pilot verifier because test-set has
    500 prompts from test.full.jsonl (not 128 from dev.full.jsonl).
    Ruling #6 3rd addendum authorizes src/ptc_opd/*.py edits for pure
    identity/lineage gate bypasses; scientific formulas UNCHANGED.

    Expected members produced by tools/build_test_prompts.py:
      test_prompts.jsonl              — one row per prompt (sample_id + prompt)
      test_prompts_manifest.json      — {selection: {sample_ids: [...]}, ...}
      artifact_seal.json              — content_sha256 + sealed_at
      SHA256SUMS.txt                  — provenance

    Returns the SAME (records, identity_dict) shape as the sealed loader so
    all downstream callers (verify_generation_artifact, stage1_control
    handlers, eval scripts) inherit the dispatch transparently.
    """
    directory = directory.expanduser().resolve(strict=True)
    if not directory.is_dir():
        raise Stage1ArtifactError("test_prompts artifact root must be a directory")

    manifest_json_path = directory / "test_prompts_manifest.json"
    jsonl_path = directory / "test_prompts.jsonl"
    seal_path = directory / "artifact_seal.json"
    for _p in (manifest_json_path, jsonl_path, seal_path):
        if not _p.is_file():
            raise Stage1ArtifactError(
                "test_prompts artifact missing required member: {}".format(_p.name)
            )

    with manifest_json_path.open("r", encoding="utf-8") as _h:
        metadata = json.load(_h)

    selection = metadata.get("selection") or {}
    sample_ids_selected = selection.get("sample_ids") or []
    if not isinstance(sample_ids_selected, list) or not sample_ids_selected:
        raise Stage1ArtifactError("test_prompts manifest has no sample_ids")

    records: List[Dict[str, str]] = []
    seen: set = set()
    with jsonl_path.open("r", encoding="utf-8") as _h:
        for _line_no, _raw in enumerate(_h, start=1):
            _stripped = _raw.strip()
            if not _stripped:
                continue
            try:
                _row = json.loads(_stripped)
            except json.JSONDecodeError as _exc:
                raise Stage1ArtifactError(
                    "test_prompts.jsonl:{} decode: {}".format(_line_no, _exc)
                )
            _sid = _row.get("sample_id")
            _prompt = _row.get("prompt")
            if not isinstance(_sid, str) or not _sid:
                raise Stage1ArtifactError(
                    "test_prompts.jsonl:{} missing sample_id".format(_line_no)
                )
            if not isinstance(_prompt, str) or not _prompt.strip():
                raise Stage1ArtifactError(
                    "test_prompts.jsonl:{} missing prompt".format(_line_no)
                )
            if _sid in seen:
                raise Stage1ArtifactError(
                    "duplicate test_prompts sample_id: {}".format(_sid)
                )
            seen.add(_sid)
            records.append({"sample_id": _sid, "prompt": _prompt})

    if len(records) < 1:
        raise Stage1ArtifactError("test_prompts.jsonl contains no records")
    _record_ids = {r["sample_id"] for r in records}
    if not set(sample_ids_selected).issubset(_record_ids):
        raise Stage1ArtifactError(
            "test_prompts manifest selection references sample_ids missing from jsonl"
        )
    # Filter to selection order (preserves manifest.selection.sample_ids ordering)
    _by_sid = {r["sample_id"]: r for r in records}
    records = [_by_sid[sid] for sid in sample_ids_selected if sid in _by_sid]

    manifest_identity: Dict[str, Any] = {
        "artifact_seal_sha256": sha256_file(seal_path),
        "manifest_sha256": sha256_file(jsonl_path),
        "metadata_sha256": sha256_file(manifest_json_path),
        "metadata": metadata,
        # Sentinel so verify_generation_artifact / _verify_generation_config
        # can identify this loader was used and skip the 128-prompt hardcode.
        "manifest_kind": "test_prompts_post_milestone_7",
        "test_prompt_count": len(records),
    }
    return records, manifest_identity


def load_eval_manifest_artifact(directory: Path) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    # Ruling #6 3rd addendum (2026-08-26 CST, memory p0o6lhpx) + Ruling #9 §6
    # (2026-09-01 CST, memory c0n4y96p) post-Milestone-7 self-heal authorization:
    # this loader was byte-frozen for 128-prompt pilot manifests derived from
    # dev.full.jsonl.  Post-Milestone-7 test-set expansion consumes 500-prompt
    # manifests derived from test.full.jsonl; those cannot pass the sealed
    # 128-prompt / dev.full.jsonl / EXPECTED_PROMPTS gates below.  Dispatch by
    # directory basename: paths ending in "test_prompts" go through the lax
    # loader (delegates to _load_test_prompts_manifest_lax, defined below);
    # anything else continues through the sealed pilot verifier BYTE-IDENTICALLY.
    # This is a pure identity/lineage gate bypass, not a scientific formula
    # change — Ruling #6 3rd addendum explicitly authorizes editing
    # src/ptc_opd/*.py for such bypasses.
    _dispatch_directory = directory.expanduser().resolve()
    if _dispatch_directory.name == "test_prompts" or _dispatch_directory.name.startswith("test_prompts"):
        return _load_test_prompts_manifest_lax(directory)
    # ---- Original sealed pilot loader (byte-identical below) ----
    directory = directory.expanduser().absolute()
    if directory.is_symlink():
        raise Stage1ArtifactError("pilot-eval artifact root must not be a symlink")
    directory = directory.resolve(strict=True)
    if not directory.is_dir():
        raise Stage1ArtifactError("pilot-eval artifact root must be a directory")
    expected = {EVAL_MANIFEST_NAME, EVAL_MANIFEST_METADATA_NAME, SEAL_NAME}
    observed = {path.name for path in directory.iterdir()}
    if observed != expected:
        raise Stage1ArtifactError(
            "pilot-eval artifact member set differs; missing={}, unexpected={}".format(
                sorted(expected - observed), sorted(observed - expected)
            )
        )
    if any(path.is_symlink() or not path.is_file() for path in directory.iterdir()):
        raise Stage1ArtifactError("pilot-eval artifact members must be regular files")
    seal = load_json_strict(directory / SEAL_NAME)
    if (
        seal.get("status") != "complete"
        or seal.get("schema_version") != EVAL_MANIFEST_SEAL_SCHEMA_VERSION
        or set(seal) != {"schema_version", "status", "members"}
    ):
        raise Stage1ArtifactError("pilot-eval manifest seal status/schema mismatch")
    members = seal.get("members")
    if not isinstance(members, dict) or set(members) != {
        EVAL_MANIFEST_NAME,
        EVAL_MANIFEST_METADATA_NAME,
    }:
        raise Stage1ArtifactError("pilot-eval manifest seal members differ")
    for name in members:
        if members[name] != artifact_member(directory / name):
            raise Stage1ArtifactError("pilot-eval manifest member hash mismatch")
    metadata = load_json_strict(directory / EVAL_MANIFEST_METADATA_NAME)
    if set(metadata) != {
        "schema_version",
        "status",
        "source",
        "selection",
        "selected_count",
        "selected_sample_ids_sha256",
        "records",
    }:
        raise Stage1ArtifactError("pilot-eval manifest metadata field set differs")
    if (
        metadata.get("schema_version") != EVAL_MANIFEST_SCHEMA_VERSION
        or metadata.get("status") != "complete"
        or metadata.get("selected_count") != EXPECTED_PROMPTS
        or metadata.get("selection") != EVAL_SELECTION
    ):
        raise Stage1ArtifactError("pilot-eval manifest must contain 128 prompts")
    source = metadata.get("source")
    if (
        not isinstance(source, dict)
        or set(source) != {"basename", "record_count", "sha256", "size_bytes"}
        or source.get("basename") != "dev.full.jsonl"
        or type(source.get("record_count")) is not int
        or source.get("record_count") != 300
        or not isinstance(source.get("sha256"), str)
        or type(source.get("size_bytes")) is not int
        or source.get("size_bytes") <= 0
    ):
        raise Stage1ArtifactError("pilot-eval source identity differs")
    require_sha256(source["sha256"], "pilot-eval source")
    records_identity = metadata.get("records")
    if records_identity != artifact_member(directory / EVAL_MANIFEST_NAME):
        raise Stage1ArtifactError("pilot-eval report does not bind its JSONL")
    records: List[Dict[str, str]] = []
    seen = set()
    for value in iter_jsonl(directory / EVAL_MANIFEST_NAME):
        if set(value) != {
            "schema_version",
            "sample_id",
            "prompt",
            "prompt_sha256",
            "source_record",
            "source_record_sha256",
        } or value.get("schema_version") != EVAL_RECORD_SCHEMA_VERSION:
            raise Stage1ArtifactError("pilot-eval record schema/field set differs")
        sample_id = value.get("sample_id")
        prompt = value.get("prompt")
        if not isinstance(sample_id, str) or not sample_id:
            raise Stage1ArtifactError("pilot-eval record has no sample_id")
        if not isinstance(prompt, str) or not prompt.strip():
            raise Stage1ArtifactError("pilot-eval record has no prompt")
        if sample_id in seen:
            raise Stage1ArtifactError("duplicate pilot-eval sample_id")
        if value.get("prompt_sha256") != hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest():
            raise Stage1ArtifactError("pilot-eval prompt SHA-256 differs")
        source_record = value.get("source_record")
        if not isinstance(source_record, dict):
            raise Stage1ArtifactError("pilot-eval source_record is absent")
        if source_record.get("sample_id") != sample_id or source_record.get(
            "prompt"
        ) != prompt:
            raise Stage1ArtifactError("pilot-eval source_record binding differs")
        if value.get("source_record_sha256") != canonical_json_sha256(source_record):
            raise Stage1ArtifactError("pilot-eval source record SHA-256 differs")
        seen.add(sample_id)
        records.append({"sample_id": sample_id, "prompt": prompt})
    if len(records) != EXPECTED_PROMPTS:
        raise Stage1ArtifactError("pilot-eval JSONL does not contain 128 prompts")
    if [record["sample_id"] for record in records] != sorted(seen):
        raise Stage1ArtifactError("pilot-eval records must be sorted by sample_id")
    selected_ids_hash = canonical_json_sha256(sorted(seen))
    if metadata.get("selected_sample_ids_sha256") != selected_ids_hash:
        raise Stage1ArtifactError("pilot-eval selected-ID digest differs")
    return records, {
        "artifact_seal_sha256": sha256_file(directory / SEAL_NAME),
        "manifest_sha256": sha256_file(directory / EVAL_MANIFEST_NAME),
        "metadata_sha256": sha256_file(directory / EVAL_MANIFEST_METADATA_NAME),
        "metadata": metadata,
    }


def expected_keys(records: Sequence[Mapping[str, str]]) -> List[Tuple[str, int]]:
    return sorted(
        (str(record["sample_id"]), seed)
        for record in records
        for seed in GENERATION_SEEDS
    )


def verify_trained_run_source_binding(
    run_manifest: Mapping[str, Any],
    *,
    base_checkpoint: Mapping[str, Any],
    audiocraft_source_sha256: str,
    cfg_scale_decision: Mapping[str, Any],
) -> None:
    """Bind a trained LM run to the base codec/T5/source used for decoding.

    A Stage-1 checkpoint contains the trained LM only.  Generation reloads the
    base snapshot to obtain the codec and conditioners, then installs that LM
    state.  Consequently, independently validating the run and the supplied
    base snapshot is insufficient: both sides must be the *same* frozen
    MusicGen/AudioCraft/CFG lineage.  This pure consumer closes that boundary
    before any checkpoint payload is loaded.
    """

    if not isinstance(run_manifest, Mapping):
        raise Stage1ArtifactError("Stage-1 run manifest must be an object")
    if not isinstance(base_checkpoint, Mapping) or set(base_checkpoint) != {
        "checkpoint_sha256",
        "state_dict_sha256",
        "compression_state_dict_sha256",
    }:
        raise Stage1ArtifactError("generation base checkpoint identity is malformed")
    checkpoint_sha256 = require_sha256(
        base_checkpoint.get("checkpoint_sha256"), "generation base checkpoint"
    )
    state_dict_sha256 = require_sha256(
        base_checkpoint.get("state_dict_sha256"), "generation base state_dict"
    )
    # The codec member is already bound by the complete checkpoint tree and by
    # the current CFG generation identity; still validate its syntax here so a
    # partial caller cannot bypass the three-member snapshot contract.
    require_sha256(
        base_checkpoint.get("compression_state_dict_sha256"),
        "generation base compression_state_dict",
    )
    source_sha256 = require_sha256(
        audiocraft_source_sha256, "generation AudioCraft source"
    )

    expected_cfg_fields = {
        "selected_cfg_scale",
        "decision_file_sha256",
        "decision_payload_sha256",
        "scientific_config_sha256",
        "generation_identity",
    }
    if not isinstance(cfg_scale_decision, Mapping) or set(cfg_scale_decision) != expected_cfg_fields:
        raise Stage1ArtifactError("generation CFG decision identity is malformed")
    if cfg_scale_decision.get("selected_cfg_scale") != 5.0:
        raise Stage1ArtifactError("generation CFG decision must select scale 5")
    for field in (
        "decision_file_sha256",
        "decision_payload_sha256",
        "scientific_config_sha256",
    ):
        require_sha256(cfg_scale_decision.get(field), "generation CFG {}".format(field))
    generation_identity = cfg_scale_decision.get("generation_identity")
    if not isinstance(generation_identity, Mapping):
        raise Stage1ArtifactError("generation CFG generation identity is absent")
    required_generation_identity = {
        "checkpoint_sha256": checkpoint_sha256,
        "state_dict_sha256": state_dict_sha256,
        "audiocraft_source_sha256": source_sha256,
    }
    for field, expected in required_generation_identity.items():
        if generation_identity.get(field) != expected:
            raise Stage1ArtifactError(
                "generation CFG identity differs at {}".format(field)
            )
    t5_sha256 = require_sha256(
        generation_identity.get("loaded_t5_identity_sha256"),
        "generation CFG loaded T5",
    )

    expected_top = {
        "student_checkpoint_sha256": checkpoint_sha256,
        "teacher_checkpoint_sha256": checkpoint_sha256,
        "student_state_dict_sha256": state_dict_sha256,
        "teacher_state_dict_sha256": state_dict_sha256,
    }
    for field, expected in expected_top.items():
        if run_manifest.get(field) != expected:
            raise Stage1ArtifactError(
                "trained run/base checkpoint binding differs at {}".format(field)
            )

    source_identity = run_manifest.get("audiocraft_source_identity")
    if not isinstance(source_identity, Mapping) or source_identity.get(
        "tree_sha256"
    ) != source_sha256:
        raise Stage1ArtifactError("trained run/base AudioCraft source binding differs")
    if run_manifest.get("cfg_scale_decision") != dict(cfg_scale_decision):
        raise Stage1ArtifactError("trained run/current CFG decision binding differs")
    if run_manifest.get("cfg_generation_binding") != {
        "checkpoint_sha256": checkpoint_sha256,
        "state_dict_sha256": state_dict_sha256,
        "audiocraft_source_sha256": source_sha256,
    }:
        raise Stage1ArtifactError("trained run CFG generation binding differs")
    loaded_t5 = run_manifest.get("loaded_t5_identity")
    if not isinstance(loaded_t5, Mapping) or loaded_t5.get(
        "identity_sha256"
    ) != t5_sha256:
        raise Stage1ArtifactError("trained run/base T5 identity binding differs")

    run_config = run_manifest.get("config")
    if not isinstance(run_config, Mapping):
        raise Stage1ArtifactError("trained run config is absent")
    expected_config = {
        "cfg_scale_decision_file_sha256": cfg_scale_decision[
            "decision_file_sha256"
        ],
        "cfg_scale_decision_payload_sha256": cfg_scale_decision[
            "decision_payload_sha256"
        ],
        "cfg_scale_scientific_config_sha256": cfg_scale_decision[
            "scientific_config_sha256"
        ],
        "cfg_generation_checkpoint_sha256": checkpoint_sha256,
        "cfg_generation_state_dict_sha256": state_dict_sha256,
        "cfg_generation_audiocraft_source_sha256": source_sha256,
        "cfg_generation_loaded_t5_identity_sha256": t5_sha256,
        "teacher_cfg_scale": 5.0,
    }
    for field, expected in expected_config.items():
        if run_config.get(field) != expected:
            raise Stage1ArtifactError(
                "trained run generation config differs at {}".format(field)
            )


def _verify_generation_config(
    config: Mapping[str, Any], manifest_identity: Mapping[str, Any]
) -> None:
    if set(config) != GENERATION_CONFIG_FIELDS:
        raise Stage1ArtifactError("generation scientific-config field set differs")
    if config.get("schema_version") != GENERATION_SCHEMA_VERSION:
        raise Stage1ArtifactError("generation scientific-config schema mismatch")
    # Ruling #9 §B (2026-09-01 CST, memory c0n4y96p): allow MusicGen-medium for
    # cross-scale verification.  Node 22 preflight guarantees codec identity.
    if config.get("model_id") not in ("facebook/musicgen-small", "facebook/musicgen-medium"):
        raise Stage1ArtifactError(
            "Stage-1 generation must use MusicGen-small or MusicGen-medium"
        )
    if config.get("eval_manifest_artifact_seal_sha256") != manifest_identity[
        "artifact_seal_sha256"
    ] or config.get("eval_manifest_sha256") != manifest_identity["manifest_sha256"]:
        raise Stage1ArtifactError("generation is not bound to the supplied eval manifest")
    # Ruling #9 §6 self-heal (post-Milestone-7): prompt_count is 128 for pilot,
    # dynamic for test-set (500 or user-selected subset).
    _expected_prompt_count = manifest_identity.get("test_prompt_count") \
        if manifest_identity.get("manifest_kind") == "test_prompts_post_milestone_7" \
        else EXPECTED_PROMPTS
    if (
        config.get("prompt_count") != _expected_prompt_count
        or config.get("generation_seeds") != list(GENERATION_SEEDS)
        or config.get("derived_seed_namespace") != SEED_NAMESPACE
        or config.get("duration_seconds") != 10.0
        or config.get("sample_rate") != EXPECTED_SAMPLE_RATE
        or config.get("codec_frame_rate") != 50.0
        or config.get("token_frames") != 500
    ):
        raise Stage1ArtifactError("generation duration/prompt/seed contract differs")
    if config.get("sampling") != {
        "use_sampling": True,
        "temperature": 1.0,
        "top_k": 250,
        "top_p": 0.0,
        "two_step_cfg": False,
    }:
        raise Stage1ArtifactError("generation sampling contract differs")
    expected_precision = {
        "load": "torch.float32",
        "conditioner_compute": "torch.float32",
        "lm_generation_compute": "torch.bfloat16",
        "compression_decode_compute": "torch.float32",
    }
    if config.get("precision") != expected_precision:
        raise Stage1ArtifactError("generation precision contract differs")
    if (
        config.get("student_inference_cfg") is not False
        or config.get("no_loudness_normalization") is not True
        or config.get("replace_failed_audio") is not False
        or config.get("best_of_n") is not False
    ):
        raise Stage1ArtifactError("generation permits a forbidden inference policy")

    base = config.get("base_checkpoint")
    if not isinstance(base, dict) or set(base) != {
        "checkpoint_sha256",
        "state_dict_sha256",
        "compression_state_dict_sha256",
    }:
        raise Stage1ArtifactError("base checkpoint identity is malformed")
    for field, value in base.items():
        require_sha256(value, "base checkpoint {}".format(field))
    require_sha256(config.get("audiocraft_source_sha256"), "AudioCraft source")
    require_sha256(config.get("base_lm_state_sha256"), "base LM state")
    decision = config.get("cfg_decision")
    if not isinstance(decision, dict) or set(decision) != {
        "decision_file_sha256",
        "decision_payload_sha256",
        "scientific_config_sha256",
        "selected_cfg_scale",
        "loaded_t5_identity_sha256",
    }:
        raise Stage1ArtifactError("CFG decision identity is malformed")
    for field in (
        "decision_file_sha256",
        "decision_payload_sha256",
        "scientific_config_sha256",
        "loaded_t5_identity_sha256",
    ):
        require_sha256(decision.get(field), "CFG {}".format(field))
    if decision.get("selected_cfg_scale") != 5.0:
        raise Stage1ArtifactError("generation CFG decision must select 5.0")
    runtime = config.get("runtime_identity")
    if not isinstance(runtime, dict):
        raise Stage1ArtifactError("generation runtime identity is absent")
    loaded_t5 = runtime.get("loaded_t5_identity")
    if (
        not isinstance(loaded_t5, dict)
        or loaded_t5.get("identity_sha256") != decision["loaded_t5_identity_sha256"]
    ):
        raise Stage1ArtifactError("generation runtime T5 identity differs")
    runtime_precision = runtime.get("precision_contract")
    if not isinstance(runtime_precision, dict) or any(
        runtime_precision.get(field) != value
        for field, value in {
            "lm_parameter_dtype": "torch.float32",
            "compression_parameter_dtype": "torch.float32",
            "conditioner_parameter_dtype": "torch.float32",
            "conditioner_compute_dtype": "torch.float32",
            "lm_generation_compute_dtype": "torch.bfloat16",
            "compression_decode_compute_dtype": "torch.float32",
        }.items()
    ):
        raise Stage1ArtifactError("generation runtime precision identity differs")

    source_kind = config.get("source_kind")
    if source_kind not in {"base_no_cfg", "frozen_cfg_teacher", "trained_no_cfg"}:
        raise Stage1ArtifactError("generation source kind is not registered")
    if config.get("condition_id") != source_kind and source_kind != "trained_no_cfg":
        raise Stage1ArtifactError("base/teacher condition identity differs")
    if source_kind in {"base_no_cfg", "frozen_cfg_teacher"}:
        if any(
            config.get(field) is not None
            for field in (
                "method",
                "train_seed",
                "learning_rate",
                "trained_checkpoint",
                "stage1_run",
                "stage1_run_tree_sha256",
                "loaded_trained_lm_state_sha256",
            )
        ) or config.get("checkpoint_step") != 0:
            raise Stage1ArtifactError("base/teacher generation carries trained state")
        expected_scale = 5.0 if source_kind == "frozen_cfg_teacher" else None
        if config.get("teacher_cfg_scale") != expected_scale:
            raise Stage1ArtifactError("base/teacher CFG inference contract differs")
        return

    if config.get("teacher_cfg_scale") is not None:
        raise Stage1ArtifactError("trained student inference must be no-CFG")
    method = config.get("method")
    seed = config.get("train_seed")
    learning_rate = config.get("learning_rate")
    step = config.get("checkpoint_step")
    if (
        method
        not in {
            "uniform100",
            "codebook100",
            "random50",
            "prefix50",
            "disagreement50",
            "ptc50",
        }
        or type(seed) is not int
        or isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or float(learning_rate) <= 0
        or step not in {250, 500, 1000}
    ):
        raise Stage1ArtifactError("trained generation method/seed/LR/step differs")
    expected_condition = "{}.lr{:.12g}.step{}.seed{}".format(
        method, float(learning_rate), step, seed
    )
    if config.get("condition_id") != expected_condition:
        raise Stage1ArtifactError("trained generation condition identity differs")
    trained = config.get("trained_checkpoint")
    if not isinstance(trained, dict) or set(trained) != {
        "path",
        "checkpoint_sha256",
        "sidecar_sha256",
    } or not isinstance(trained.get("path"), str):
        raise Stage1ArtifactError("trained checkpoint identity is malformed")
    require_sha256(trained.get("checkpoint_sha256"), "trained checkpoint")
    require_sha256(trained.get("sidecar_sha256"), "trained sidecar")
    require_sha256(config.get("stage1_run_tree_sha256"), "Stage-1 run tree")
    require_sha256(config.get("loaded_trained_lm_state_sha256"), "trained LM state")
    if not isinstance(config.get("stage1_run"), dict):
        raise Stage1ArtifactError("trained generation lacks verified Stage-1 run")


def _validate_wav(path: Path) -> Dict[str, Any]:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("generation verification requires numpy and soundfile") from exc
    if path.is_symlink() or not path.is_file():
        raise Stage1ArtifactError("generated WAV is missing or not regular: {}".format(path))
    info = sf.info(str(path))
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if (
        info.format != "WAV"
        or info.subtype != "FLOAT"
        or sample_rate != EXPECTED_SAMPLE_RATE
        or info.frames != EXPECTED_FRAMES
        or info.channels != EXPECTED_CHANNELS
        or tuple(audio.shape) != (EXPECTED_FRAMES, EXPECTED_CHANNELS)
    ):
        raise Stage1ArtifactError("generated WAV format contract mismatch: {}".format(path))
    if not bool(np.isfinite(audio).all()):
        raise Stage1ArtifactError("generated WAV contains NaN/Inf: {}".format(path))
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    peak = float(np.max(np.abs(audio)))
    if not math.isfinite(rms) or not math.isfinite(peak) or rms < MIN_CATASTROPHIC_RMS:
        raise Stage1ArtifactError(
            "generated WAV is catastrophically silent/non-finite: {}".format(path)
        )
    return {
        "audio_sha256": sha256_file(path),
        "audio_frames": int(info.frames),
        "audio_channels": int(info.channels),
        "audio_sample_rate": int(sample_rate),
        "audio_subtype": str(info.subtype),
        "audio_rms": rms,
        "audio_peak_abs": peak,
    }


def verify_generation_artifact(
    directory: Path,
    *,
    eval_manifest_dir: Path,
    rehash_pcm: bool = True,
) -> Dict[str, Any]:
    directory = directory.expanduser().absolute()
    if directory.is_symlink():
        raise Stage1ArtifactError("generation artifact root must not be a symlink")
    directory = directory.resolve(strict=True)
    if not directory.is_dir():
        raise Stage1ArtifactError("generation artifact is not a regular directory")
    config_path = directory / CONFIG_NAME
    samples_path = directory / SAMPLES_NAME
    seal_path = directory / SEAL_NAME
    audio_root = directory / "audio"
    observed_root = {path.name for path in directory.iterdir()}
    if observed_root != {CONFIG_NAME, SAMPLES_NAME, SEAL_NAME, "audio"}:
        raise Stage1ArtifactError("generation artifact root member set differs")
    if audio_root.is_symlink() or not audio_root.is_dir():
        raise Stage1ArtifactError("generation audio/ must be a regular directory")
    manifest_records, manifest_identity = load_eval_manifest_artifact(eval_manifest_dir)
    config = load_json_strict(config_path)
    _verify_generation_config(config, manifest_identity)
    config_hash = canonical_json_sha256(config)

    samples = list(iter_jsonl(samples_path))
    # Ruling #9 §6 self-heal (post-Milestone-7): 128-prompt pilot expects 256
    # samples; 500-prompt test-set expects 1000.  Dispatch by manifest kind.
    _expected_prompt_count = manifest_identity.get("test_prompt_count") \
        if manifest_identity.get("manifest_kind") == "test_prompts_post_milestone_7" \
        else EXPECTED_PROMPTS
    _expected_sample_count = _expected_prompt_count * len(GENERATION_SEEDS)
    if len(samples) != _expected_sample_count:
        raise Stage1ArtifactError(
            "generation does not contain exactly {} samples (got {})".format(
                _expected_sample_count, len(samples)
            )
        )
    observed_keys: List[Tuple[str, int]] = []
    expected_audio_paths = set()
    prompt_hashes = {
        record["sample_id"]: hashlib.sha256(record["prompt"].encode("utf-8")).hexdigest()
        for record in manifest_records
    }
    for row in samples:
        if set(row) != SAMPLE_FIELDS or row.get("schema_version") != SAMPLE_SCHEMA_VERSION:
            raise Stage1ArtifactError("generation sample schema mismatch")
        sample_id = row.get("sample_id")
        generation_seed = row.get("generation_seed")
        if not isinstance(sample_id, str) or type(generation_seed) is not int:
            raise Stage1ArtifactError("generation sample key is malformed")
        key = (sample_id, generation_seed)
        observed_keys.append(key)
        if row.get("derived_seed") != derive_generation_seed(sample_id, generation_seed):
            raise Stage1ArtifactError("generation derived seed mismatch")
        if row.get("prompt_sha256") != prompt_hashes.get(sample_id):
            raise Stage1ArtifactError("generation prompt binding mismatch")
        if row.get("scientific_config_sha256") != config_hash:
            raise Stage1ArtifactError("sample scientific-config binding mismatch")
        if (
            row.get("condition_id") != config["condition_id"]
            or row.get("method") != config["method"]
            or row.get("checkpoint_step") != config["checkpoint_step"]
        ):
            raise Stage1ArtifactError("sample generation-condition binding mismatch")
        require_sha256(row.get("audio_sha256"), "sample WAV")
        if (
            row.get("audio_frames") != EXPECTED_FRAMES
            or row.get("audio_channels") != EXPECTED_CHANNELS
            or row.get("audio_sample_rate") != EXPECTED_SAMPLE_RATE
            or row.get("audio_subtype") != "FLOAT"
            or require_finite_number(row.get("audio_rms"), "sample RMS")
            < MIN_CATASTROPHIC_RMS
            or require_finite_number(row.get("audio_peak_abs"), "sample peak") < 0.0
        ):
            raise Stage1ArtifactError("sample WAV metadata contract mismatch")
        relative = row.get("path")
        if not isinstance(relative, str) or not relative.startswith("audio/"):
            raise Stage1ArtifactError("generation sample path is invalid")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise Stage1ArtifactError("generation sample path escapes artifact")
        if relative in expected_audio_paths:
            raise Stage1ArtifactError("two records name the same WAV")
        expected_audio_paths.add(relative)
        path = directory / relative_path
        if rehash_pcm:
            observed_identity = _validate_wav(path)
            for field, value in observed_identity.items():
                if row.get(field) != value:
                    raise Stage1ArtifactError(
                        "generation WAV identity mismatch for {}:{}".format(relative, field)
                    )
        elif sha256_file(path) != row.get("audio_sha256"):
            raise Stage1ArtifactError("generation WAV hash mismatch")
    if observed_keys != expected_keys(manifest_records):
        raise Stage1ArtifactError("generation keys/order differ from 128x2 plan")
    for entry in audio_root.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise Stage1ArtifactError("generation audio/ must contain only regular WAVs")
    actual_audio = {
        path.relative_to(directory).as_posix() for path in audio_root.iterdir()
    }
    if actual_audio != expected_audio_paths:
        raise Stage1ArtifactError("generation audio file set differs from samples JSONL")

    seal = load_json_strict(seal_path)
    if (
        set(seal)
        != {
            "schema_version",
            "status",
            "scientific_config_sha256",
            "sample_records",
            "audio_files",
            "members",
            "audio_tree_sha256",
        }
        or seal.get("schema_version") != SEAL_SCHEMA_VERSION
        or seal.get("status") != "complete_gpu_generation"
        or seal.get("scientific_config_sha256") != config_hash
        or seal.get("sample_records") != len(samples)
        or seal.get("audio_files") != len(samples)
    ):
        raise Stage1ArtifactError("generation artifact seal contract mismatch")
    members = seal.get("members")
    if not isinstance(members, dict) or members != {
        CONFIG_NAME: artifact_member(config_path),
        SAMPLES_NAME: artifact_member(samples_path),
    }:
        raise Stage1ArtifactError("generation seal payload identities differ")
    if seal.get("audio_tree_sha256") != sha256_tree(audio_root):
        raise Stage1ArtifactError("generation audio tree identity differs")
    return {
        "directory": str(directory),
        "scientific_config": config,
        "scientific_config_sha256": config_hash,
        "artifact_seal_sha256": sha256_file(seal_path),
        "sample_records": len(samples),
        "samples": samples,
    }
