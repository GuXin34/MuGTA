#!/usr/bin/env python3
"""Generate sealed 128x2 Stage-1 evaluation audio from base or trained small LM.

Scientific failures are never repaired by changing a seed or sampling another
candidate.  A failed attempt remains outside the requested final directory and
must be classified by the controller before the identical config may retry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


WORKPACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKPACK_ROOT / "src"))
sys.path.insert(0, str(WORKPACK_ROOT / "scripts"))

from ptc_opd.cfg_decision import verify_cfg_scale_decision  # noqa: E402
from ptc_opd.stage1_artifact import (  # noqa: E402
    Stage1ArtifactError,
    artifact_member,
    canonical_json_bytes,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
    sha256_tree,
)
from ptc_opd.stage1_generation import (  # noqa: E402
    CONFIG_NAME,
    GENERATION_SCHEMA_VERSION,
    GENERATION_SEEDS,
    SAMPLE_SCHEMA_VERSION,
    SAMPLES_NAME,
    SEAL_NAME,
    SEAL_SCHEMA_VERSION,
    derive_generation_seed,
    load_eval_manifest_artifact,
    verify_trained_run_source_binding,
    verify_generation_artifact,
)
MODEL_ID = "facebook/musicgen-small"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-kind",
        choices=("base_no_cfg", "frozen_cfg_teacher", "trained_no_cfg"),
        required=True,
    )
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--audiocraft-root", type=Path, required=True)
    parser.add_argument("--cfg-scale-decision-dir", type=Path, required=True)
    parser.add_argument("--eval-manifest-dir", type=Path, required=True)
    parser.add_argument("--stage1-run", type=Path)
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def _torch_load(path: Path) -> Any:
    import inspect
    import torch

    options: Dict[str, Any] = {"map_location": "cpu"}
    parameters = inspect.signature(torch.load).parameters
    if "weights_only" in parameters:
        options["weights_only"] = False
    if "mmap" in parameters:
        options["mmap"] = True
    return torch.load(str(path), **options)

def _load_test_prompts_manifest(
    directory: Path,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Lax loader for post-Milestone-7 test_prompts artifacts.

    Ruling #9 §6 (2026-09-01 CST, memory c0n4y96p) authorizes unsealed
    wrapper/harness self-heal.  This loader does NOT run through the sealed
    128-prompt pilot verifier (`load_eval_manifest_artifact`), because
    test-set has 500 prompts from test.full.jsonl, not 128 from dev.full.jsonl.

    Expected members (produced by tools/build_test_prompts.py):
      test_prompts.jsonl              — one row per prompt (source_record schema)
      test_prompts_manifest.json      — {selection: {sample_ids: [...]}, ...}
      artifact_seal.json              — {content_sha256, sealed_at, ...}
      SHA256SUMS.txt                  — provenance

    Returns:
      (records, identity_dict) with the SAME shape as load_eval_manifest_artifact
      so downstream config-writing code is unchanged.
    """
    import hashlib

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
    seen = set()
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
    # Records must be a superset of selection (sanity check)
    _record_ids = {r["sample_id"] for r in records}
    if not set(sample_ids_selected).issubset(_record_ids):
        raise Stage1ArtifactError(
            "test_prompts manifest selection references sample_ids missing from jsonl"
        )
    # Filter to selection order (preserves manifest.selection.sample_ids ordering)
    _by_sid = {r["sample_id"]: r for r in records}
    records = [_by_sid[sid] for sid in sample_ids_selected if sid in _by_sid]

    def _sha256_file(_p: Path) -> str:
        _h = hashlib.sha256()
        with _p.open("rb") as _fh:
            for _chunk in iter(lambda: _fh.read(1 << 20), b""):
                _h.update(_chunk)
        return _h.hexdigest()

    manifest_identity = {
        "artifact_seal_sha256": _sha256_file(seal_path),
        "manifest_sha256": _sha256_file(jsonl_path),
        "metadata_sha256": _sha256_file(manifest_json_path),
        "metadata": metadata,
        # Sentinel so downstream can identify this loader was used:
        "manifest_kind": "test_prompts_post_milestone_7",
    }
    return records, manifest_identity


def _resolve_source(args: argparse.Namespace) -> Dict[str, Any]:
    from cfg_scale_gate import _validate_audiocraft_source, _validate_local_snapshot

    checkpoint_identity = _validate_local_snapshot(args.base_checkpoint)
    source_identity = _validate_audiocraft_source(args.audiocraft_root)
    decision = verify_cfg_scale_decision(args.cfg_scale_decision_dir, require_selected=True)
    generation_identity = decision.generation_identity
    # Ruling #9 §B (2026-09-01 CST, memory c0n4y96p): infer model_id from
    # base-checkpoint basename (which contains "musicgen-small" or
    # "musicgen-medium").  _validate_local_snapshot does NOT return model_id;
    # cfg decision's generation_identity is the authoritative source for
    # matching, so we derive checkpoint's model_id from basename and confirm
    # it matches the CFG decision's model_id.
    _basename = args.base_checkpoint.name.lower()
    if "musicgen-medium" in _basename:
        _checkpoint_model_id = "facebook/musicgen-medium"
    elif "musicgen-small" in _basename:
        _checkpoint_model_id = "facebook/musicgen-small"
    else:
        raise Stage1ArtifactError(
            "cannot infer model_id from base-checkpoint basename {!r}; "
            "must contain 'musicgen-small' or 'musicgen-medium'".format(_basename)
        )
    required_matches = {
        "model_id": _checkpoint_model_id,
        "checkpoint_sha256": checkpoint_identity["checkpoint_sha256"],
        "state_dict_sha256": checkpoint_identity["state_dict_sha256"],
        "compression_state_dict_sha256": checkpoint_identity[
            "compression_state_dict_sha256"
        ],
        "audiocraft_source_sha256": source_identity["audiocraft_source_sha256"],
    }
    for field, observed in required_matches.items():
        if generation_identity.get(field) != observed:
            raise Stage1ArtifactError(
                "base/source differs from CFG decision at {} (checkpoint={!r}, cfg_decision={!r})".format(
                    field, observed, generation_identity.get(field)
                )
            )
    if decision.selected_cfg_scale != 5.0:
        raise Stage1ArtifactError(
            "CFG decision must select scale 5.0 (Ruling #9 §B: same for medium)"
        )

    result: Dict[str, Any] = {
        "base_checkpoint": checkpoint_identity,
        "audiocraft": source_identity,
        "cfg_decision": {
            "decision_file_sha256": decision.decision_file_sha256,
            "decision_payload_sha256": decision.decision_payload_sha256,
            "scientific_config_sha256": decision.scientific_config_sha256,
            "selected_cfg_scale": decision.selected_cfg_scale,
            "loaded_t5_identity_sha256": generation_identity[
                "loaded_t5_identity_sha256"
            ],
        },
        # Ruling #9 §B: propagate the resolved model_id (small OR medium) so
        # main() can use it for downstream architecture verification.
        "model_id": _checkpoint_model_id,
        "stage1_run": None,
        "checkpoint_step": 0,
        "trained_checkpoint": None,
        "condition_id": args.source_kind,
        "method": None,
        "train_seed": None,
        "learning_rate": None,
    }
    if args.source_kind == "trained_no_cfg":
        if args.stage1_run is None or args.checkpoint_step is None:
            raise Stage1ArtifactError(
                "trained_no_cfg requires --stage1-run and --checkpoint-step"
            )
        from train_stage1 import checkpoint_paths, verify_sealed_stage1_run

        supplied_run_dir = args.stage1_run.expanduser().absolute()
        if supplied_run_dir.is_symlink():
            raise Stage1ArtifactError("Stage-1 run root must not be a symlink")
        run_dir = supplied_run_dir.resolve(strict=True)
        if not run_dir.is_dir():
            raise Stage1ArtifactError("Stage-1 run root must be a directory")
        verified = verify_sealed_stage1_run(run_dir)
        manifest = load_json_strict(run_dir / "run_manifest.json")
        config = manifest.get("config")
        if not isinstance(config, dict):
            raise Stage1ArtifactError("Stage-1 run manifest has no config")
        verify_trained_run_source_binding(
            manifest,
            base_checkpoint=result["base_checkpoint"],
            audiocraft_source_sha256=result["audiocraft"][
                "audiocraft_source_sha256"
            ],
            cfg_scale_decision={
                "selected_cfg_scale": decision.selected_cfg_scale,
                "decision_file_sha256": decision.decision_file_sha256,
                "decision_payload_sha256": decision.decision_payload_sha256,
                "scientific_config_sha256": decision.scientific_config_sha256,
                "generation_identity": dict(decision.generation_identity),
            },
        )
        step = int(args.checkpoint_step)
        if step not in {250, 500, 1000}:
            raise Stage1ArtifactError("evaluation checkpoint step must be 250/500/1000")
        if step > int(verified["optimizer_step"]):
            raise Stage1ArtifactError("requested checkpoint is beyond the sealed run")
        checkpoint_dir = run_dir / "checkpoints" / "step-{:05d}".format(step)
        checkpoint, sidecar = checkpoint_paths(checkpoint_dir)
        sidecar_payload = load_json_strict(sidecar)
        if sidecar_payload.get("sha256") != sha256_file(checkpoint):
            raise Stage1ArtifactError("trained checkpoint sidecar mismatch")
        method = config.get("mode")
        seed = config.get("seed")
        learning_rate = config.get("learning_rate")
        if not isinstance(method, str) or type(seed) is not int or not isinstance(
            learning_rate, (int, float)
        ):
            raise Stage1ArtifactError("Stage-1 run method/seed/LR is malformed")
        result.update(
            {
                "stage1_run": verified,
                "stage1_run_tree_sha256": sha256_tree(run_dir),
                "checkpoint_step": step,
                "trained_checkpoint": {
                    "path": str(checkpoint_dir),
                    "checkpoint_sha256": sidecar_payload["sha256"],
                    "sidecar_sha256": sha256_file(sidecar),
                },
                "condition_id": "{}.lr{:.12g}.step{}.seed{}".format(
                    method, float(learning_rate), step, seed
                ),
                "method": method,
                "train_seed": seed,
                "learning_rate": float(learning_rate),
            }
        )
    elif args.stage1_run is not None or args.checkpoint_step is not None:
        raise Stage1ArtifactError("base/teacher generation forbids Stage-1 run arguments")
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(dict(value)))
        stream.flush()
        os.fsync(stream.fileno())


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for value in values:
            stream.write(
                json.dumps(
                    dict(value),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def execute(args: argparse.Namespace) -> Path:
    import torch
    from cfg_scale_gate import (
        _audiocraft_import_path,
        _generate_one,
        _precompute_prompt_condition_tensors,
        _prepare_and_identify_frozen_model,
        _write_raw_float_wav,
    )
    from ptc_opd.train_utils import hash_module_state

    output = args.output_dir.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("refusing to overwrite generation output: {}".format(output))
    failure_record = output.with_name(output.name + ".FAILED.json")
    if failure_record.exists() or failure_record.is_symlink():
        raise FileExistsError(
            "a prior attempt record exists; retry in a new attempt directory: {}".format(
                failure_record
            )
        )
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("formal Stage-1 generation requires CUDA BF16")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise Stage1ArtifactError("--device must be CUDA")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Ruling #9 §6 self-heal (post-Milestone-7 test-set expansion, 2026-09-04):
    # load_eval_manifest_artifact enforces 128-prompt pilot contract
    # (dev.full.jsonl derived; hard-coded EXPECTED_PROMPTS=128, source
    # basename=="dev.full.jsonl", record_count==300).  Test-set has 500
    # prompts from test.full.jsonl and does not fit that schema.  We dispatch
    # by manifest dir basename: paths ending in "test_prompts" go through
    # _load_test_prompts_manifest (lax loader defined below); anything else
    # continues to hit the sealed pilot loader byte-identically.
    _manifest_basename = args.eval_manifest_dir.name
    if _manifest_basename == "test_prompts" or _manifest_basename.startswith("test_prompts"):
        prompts, manifest_identity = _load_test_prompts_manifest(args.eval_manifest_dir)
    else:
        prompts, manifest_identity = load_eval_manifest_artifact(args.eval_manifest_dir)
    source = _resolve_source(args)
    # Ruling #9 §B (2026-09-01 CST, memory c0n4y96p): use the model_id resolved
    # from the actual base-checkpoint basename (source["model_id"]) instead of
    # the legacy MODEL_ID constant which was pinned to musicgen-small.
    _run_model_id = source["model_id"]
    config = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "model_id": _run_model_id,
        "source_kind": args.source_kind,
        "condition_id": source["condition_id"],
        "method": source["method"],
        "train_seed": source["train_seed"],
        "learning_rate": source["learning_rate"],
        "checkpoint_step": source["checkpoint_step"],
        "base_checkpoint": source["base_checkpoint"],
        "trained_checkpoint": source["trained_checkpoint"],
        "stage1_run": source["stage1_run"],
        "stage1_run_tree_sha256": source.get("stage1_run_tree_sha256"),
        "audiocraft_source_sha256": source["audiocraft"]["audiocraft_source_sha256"],
        "cfg_decision": source["cfg_decision"],
        "eval_manifest_artifact_seal_sha256": manifest_identity[
            "artifact_seal_sha256"
        ],
        "eval_manifest_sha256": manifest_identity["manifest_sha256"],
        "prompt_count": len(prompts),
        "generation_seeds": list(GENERATION_SEEDS),
        "derived_seed_namespace": "ptc-opd-small-pilot-generation-v1",
        "duration_seconds": 10.0,
        "sample_rate": 32000,
        "codec_frame_rate": 50.0,
        "token_frames": 500,
        "sampling": {
            "use_sampling": True,
            "temperature": 1.0,
            "top_k": 250,
            "top_p": 0.0,
            "two_step_cfg": False,
        },
        "precision": {
            "load": "torch.float32",
            "conditioner_compute": "torch.float32",
            "lm_generation_compute": "torch.bfloat16",
            "compression_decode_compute": "torch.float32",
        },
        "student_inference_cfg": False,
        "teacher_cfg_scale": 5.0 if args.source_kind == "frozen_cfg_teacher" else None,
        "no_loudness_normalization": True,
        "replace_failed_audio": False,
        "best_of_n": False,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".{}.partial.".format(output.name), dir=str(output.parent)))
    try:
        with _audiocraft_import_path(args.audiocraft_root.resolve(strict=True)):
            from audiocraft.models import MusicGen
            from audiocraft.models.loaders import load_compression_model, load_lm_model

            lm = load_lm_model(str(args.base_checkpoint.resolve(strict=True)), device="cpu")
            compression = load_compression_model(
                str(args.base_checkpoint.resolve(strict=True)), device="cpu"
            )
            base_lm_hash = hash_module_state(lm)
            loaded_trained_hash = None
            if args.source_kind == "trained_no_cfg":
                checkpoint_path = Path(source["trained_checkpoint"]["path"]) / "checkpoint.pt"
                payload = _torch_load(checkpoint_path)
                if not isinstance(payload, dict) or set(payload) != {
                    "metadata",
                    "student_state",
                    "optimizer_state",
                    "scheduler_state",
                    "rng_state_by_rank",
                }:
                    raise Stage1ArtifactError("trained checkpoint payload field set differs")
                state = payload["student_state"]
                if not isinstance(state, Mapping):
                    raise Stage1ArtifactError("trained checkpoint has no student state")
                lm.load_state_dict(state, strict=True)
                loaded_trained_hash = hash_module_state(lm)
                if loaded_trained_hash == base_lm_hash:
                    raise Stage1ArtifactError("trained checkpoint is unchanged from initialization")
                del payload
            model = MusicGen(str(args.base_checkpoint), compression, lm)
            # Ruling #9 §B: pass the dynamic model_id so cfg_scale_gate's
            # MODEL_ARCHITECTURES lookup finds the correct medium (1536×48×24)
            # or small (1024×24×16) transformer contract.
            runtime_identity = _prepare_and_identify_frozen_model(
                model, device, model_id=_run_model_id
            )
            if runtime_identity["loaded_t5_identity"]["identity_sha256"] != source[
                "cfg_decision"
            ]["loaded_t5_identity_sha256"]:
                raise Stage1ArtifactError("loaded T5 differs from frozen CFG decision")
            config["runtime_identity"] = runtime_identity
            config["base_lm_state_sha256"] = base_lm_hash
            config["loaded_trained_lm_state_sha256"] = loaded_trained_hash
            config_hash = canonical_json_sha256(config)
            _write_json(staging / CONFIG_NAME, config)

            rows: List[Dict[str, Any]] = []
            for prompt_index, record in enumerate(prompts):
                sample_id = record["sample_id"]
                prompt = record["prompt"]
                conditional, cfg_batched = _precompute_prompt_condition_tensors(model, prompt)
                for generation_seed in GENERATION_SEEDS:
                    derived_seed = derive_generation_seed(sample_id, generation_seed)
                    _seed_everything(derived_seed)
                    condition = (
                        "cfg" if args.source_kind == "frozen_cfg_teacher" else "no_cfg"
                    )
                    audio = _generate_one(
                        model,
                        conditional,
                        cfg_batched,
                        condition,
                        5.0 if condition == "cfg" else None,
                        derived_seed,
                    )
                    stem = "{:04d}-{}-s{}.wav".format(
                        prompt_index,
                        hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:16],
                        generation_seed,
                    )
                    relative = Path("audio") / stem
                    identity = _write_raw_float_wav(staging / relative, audio, 32000)
                    # Recompute RMS from the exact serialized float samples in the
                    # verifier.  Store it here as well for anomaly audits.
                    import numpy as np
                    import soundfile as sf

                    roundtrip, _ = sf.read(
                        str(staging / relative), dtype="float32", always_2d=True
                    )
                    rms = float(np.sqrt(np.mean(np.square(roundtrip, dtype=np.float64))))
                    if not np.isfinite(rms) or rms < 1.0e-7:
                        raise Stage1ArtifactError(
                            "catastrophically silent generated WAV; no replacement is allowed"
                        )
                    rows.append(
                        {
                            "schema_version": SAMPLE_SCHEMA_VERSION,
                            "sample_id": sample_id,
                            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                            "generation_seed": generation_seed,
                            "derived_seed": derived_seed,
                            "condition_id": source["condition_id"],
                            "method": source["method"],
                            "checkpoint_step": source["checkpoint_step"],
                            "path": relative.as_posix(),
                            **identity,
                            "audio_rms": rms,
                            "scientific_config_sha256": config_hash,
                        }
                    )
            rows.sort(key=lambda row: (str(row["sample_id"]), int(row["generation_seed"])))
            _write_jsonl(staging / SAMPLES_NAME, rows)

            # Generation is long enough that a mounted manifest, checkpoint,
            # sealed run, CFG decision, or source tree could change while the
            # 256 WAVs are being produced.  Re-resolve every frozen input and
            # refuse to seal mixed-provenance output if any identity drifted.
            # Ruling #9 §6 self-heal: same dispatch as §main() opening — test
            # prompts go through the lax loader, pilot manifest through sealed.
            _post_basename = args.eval_manifest_dir.name
            if _post_basename == "test_prompts" or _post_basename.startswith("test_prompts"):
                post_prompts, post_manifest_identity = _load_test_prompts_manifest(
                    args.eval_manifest_dir
                )
            else:
                post_prompts, post_manifest_identity = load_eval_manifest_artifact(
                    args.eval_manifest_dir
                )
            post_source = _resolve_source(args)
            if post_prompts != prompts or post_manifest_identity != manifest_identity:
                raise Stage1ArtifactError(
                    "eval manifest changed during generation"
                )
            if canonical_json_sha256(post_source) != canonical_json_sha256(source):
                raise Stage1ArtifactError(
                    "checkpoint/run/CFG/source identity changed during generation"
                )
            seal = {
                "schema_version": SEAL_SCHEMA_VERSION,
                "status": "complete_gpu_generation",
                "scientific_config_sha256": config_hash,
                "sample_records": len(rows),
                "audio_files": len(rows),
                "members": {
                    CONFIG_NAME: artifact_member(staging / CONFIG_NAME),
                    SAMPLES_NAME: artifact_member(staging / SAMPLES_NAME),
                },
                "audio_tree_sha256": sha256_tree(staging / "audio"),
            }
            _write_json(staging / SEAL_NAME, seal)
            verify_generation_artifact(
                staging, eval_manifest_dir=args.eval_manifest_dir, rehash_pcm=True
            )
        os.replace(str(staging), str(output))
    except BaseException as exc:
        # Partial bytes are not scientific data.  Keep a small immutable failure
        # record beside the requested target; never delete/replace an individual
        # WAV and never publish a partial generation directory.
        failure = failure_record
        if not failure.exists():
            _write_json(
                failure,
                {
                    "schema_version": "ptc-opd-stage1-generation-failure-v1",
                    "scientific_config_sha256": canonical_json_sha256(config),
                    "replacement_audio_generated": False,
                    "best_of_n": False,
                    "partial_directory_published": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
        shutil.rmtree(staging, ignore_errors=True)
        raise
    verify_generation_artifact(
        output, eval_manifest_dir=args.eval_manifest_dir, rehash_pcm=False
    )
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        output = execute(args)
    except (OSError, ValueError, RuntimeError, Stage1ArtifactError) as exc:
        print("STAGE1 GENERATION FAILED: {}".format(exc), file=sys.stderr)
        return 3
    print(
        json.dumps(
            {
                "status": "complete_gpu_generation",
                "output_dir": str(output),
                "artifact_seal_sha256": sha256_file(output / SEAL_NAME),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
