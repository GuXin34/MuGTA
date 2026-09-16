#!/usr/bin/env python3
"""Finish the frozen CFG-scale gate in the eval-fad environment.

Despite the environment name, this wrapper computes *no FAD*.  It verifies
the generation and partial-quality artifacts, explicitly loads the local
music-CLAP checkpoint as non-fusion HTSAT-base, computes normalized paired
prompt/audio cosine scores, and atomically publishes the two files consumed
by ``cfg_scale_gate.py decide`` plus an integrity seal.
"""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import math
from pathlib import Path
import sys
from typing import Dict, List, Mapping, Optional, Sequence


SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import cfg_eval_common as common


MUSIC_CLAP_CHECKPOINT_BASENAME = "music_audioset_epoch_15_esc_90.14.pt"
# The predecessor environment originally documented a different upstream hash,
# but both retained, sealed CFG evaluations used this exact re-published HF
# payload.  Full-pickle loading is permitted only after this byte identity has
# passed, never merely because the basename matches.
PINNED_MUSIC_CLAP_CHECKPOINT_SHA256 = (
    "fae3e9c087f2909c28a09dc31c8dfcdacbc42ba44c70e972b58c1bd1caf6dedd"
)


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("music-CLAP runtime configuration contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item") and callable(getattr(value, "item")):
        scalar = value.item()
        if scalar is value:
            raise TypeError("unsupported music-CLAP configuration value")
        return _jsonable(scalar)
    raise TypeError("unsupported music-CLAP configuration value {!r}".format(type(value)))


def _require_trusted_music_clap_checkpoint(path: Path) -> tuple[Path, str]:
    checkpoint = common.require_local_file(
        path,
        "music-CLAP checkpoint",
        basename=MUSIC_CLAP_CHECKPOINT_BASENAME,
    )
    observed = common.sha256_file(checkpoint)
    if observed != PINNED_MUSIC_CLAP_CHECKPOINT_SHA256:
        raise RuntimeError(
            "music-CLAP checkpoint SHA-256 mismatch: expected {}, observed {}".format(
                PINNED_MUSIC_CLAP_CHECKPOINT_SHA256,
                observed,
            )
        )
    return checkpoint, observed


def _load_ckpt_with_scoped_full_pickle(model: object, torch_module: object, checkpoint: Path) -> None:
    """Apply the torch>=2.6 compatibility override only during load_ckpt()."""

    original_torch_load = torch_module.load

    def torch_load_full_pickle(*load_args, **load_kwargs):
        # Preserve an explicit caller choice; only supply the missing default
        # used by laion-clap 1.1.6.
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    try:
        torch_module.load = torch_load_full_pickle
        model.load_ckpt(str(checkpoint))
    finally:
        torch_module.load = original_torch_load


class MusicClapBackend:
    def __init__(self, args: argparse.Namespace):
        import torch
        import laion_clap
        from laion_clap import CLAP_Module

        checkpoint, checkpoint_sha256 = _require_trusted_music_clap_checkpoint(
            Path(args.clap_checkpoint)
        )
        device = str(args.device)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("music-CLAP evaluation requested CUDA but CUDA is unavailable")
        model = CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=device)
        # laion-clap 1.1.6 calls torch.load() without weights_only=.  PyTorch
        # >=2.6 changed that default to True, which rejects the numpy scalar
        # stored in this public checkpoint.  The checkpoint bytes were gated
        # above before any unsafe deserialization; now inject False only while
        # this one trusted load_ckpt call is active, restoring torch.load even
        # if laion-clap raises.
        _load_ckpt_with_scoped_full_pickle(model, torch, checkpoint)
        if bool(getattr(model, "enable_fusion", True)):
            raise RuntimeError("music-CLAP unexpectedly enabled feature fusion")

        laion_identity = common.imported_package_identity(laion_clap, "laion_clap")
        try:
            import clap_module

            core_identity = common.imported_package_identity(clap_module, "clap_module")
        except ImportError:
            core_identity = laion_identity
        source_components = {
            "laion_clap_python_sha256": laion_identity["source_sha256"],
            "clap_module_python_sha256": core_identity["source_sha256"],
        }
        source_details = {
            "laion_clap": common.package_file_hashes(
                Path(str(laion_identity["package_root"]))
            ),
            "clap_module": common.package_file_hashes(
                Path(str(core_identity["package_root"]))
            ),
        }
        # CLAP_Module exposes a bound ``tokenizer(texts)`` method and stores
        # the actual HuggingFace tokenizer in ``self.tokenize``.  The former
        # is the inference API; the latter owns get_vocab() for provenance.
        tokenizer_method = getattr(model, "tokenizer", None)
        tokenizer_object = getattr(model, "tokenize", None)
        if not callable(tokenizer_method):
            raise RuntimeError("music-CLAP has no callable tokenizer(texts) method")
        if tokenizer_object is None or not callable(getattr(tokenizer_object, "get_vocab", None)):
            raise RuntimeError("cannot fingerprint the music-CLAP tokenizer object")
        runtime_config = {
            "constructor": {
                "enable_fusion": False,
                "amodel": "HTSAT-base",
                "device": device,
            },
            "loaded_model_cfg": _jsonable(getattr(model, "model_cfg", {})),
            "tokenizer_vocab_sha256": common.sha256_json(tokenizer_object.get_vocab()),
            "tokenizer_api": "CLAP_Module.tokenizer(texts) using CLAP_Module.tokenize",
            "audio_api": "get_audio_embedding_from_filelist(use_tensor=True)",
            "text_api": "get_text_embedding(use_tensor=True)",
            "similarity": "L2-normalize audio/text embeddings independently, then paired dot",
            "fad_computed": False,
        }
        try:
            package_version = metadata.version("laion-clap")
        except metadata.PackageNotFoundError:
            package_version = "unknown"
        self.model = model
        self._torch = torch
        self.identity = {
            "checkpoint_sha256": checkpoint_sha256,
            "source_sha256": common.sha256_json(source_components),
            "config_sha256": common.sha256_json(runtime_config),
            "details": {
                "checkpoint_basename": checkpoint.name,
                "package_version": package_version,
                "source_components": source_components,
                "source_file_hashes": source_details,
                "runtime_config": runtime_config,
            },
        }

    def score_batch(self, audio_paths: Sequence[Path], prompts: Sequence[str]) -> List[float]:
        if len(audio_paths) != len(prompts):
            raise ValueError("music-CLAP audio/prompt batch length mismatch")
        with self._torch.inference_mode():
            audio = self.model.get_audio_embedding_from_filelist(
                x=[str(path) for path in audio_paths], use_tensor=True
            )
            text = self.model.get_text_embedding(x=list(prompts), use_tensor=True)
        audio_tensor = self._torch.as_tensor(audio, dtype=self._torch.float32).detach().cpu()
        text_tensor = self._torch.as_tensor(text, dtype=self._torch.float32).detach().cpu()
        if (
            audio_tensor.ndim != 2
            or text_tensor.ndim != 2
            or audio_tensor.shape != text_tensor.shape
            or audio_tensor.shape[0] != len(audio_paths)
            or audio_tensor.shape[1] <= 0
        ):
            raise RuntimeError(
                "music-CLAP embeddings must be paired [B,D], got {} and {}".format(
                    tuple(audio_tensor.shape), tuple(text_tensor.shape)
                )
            )
        audio_norms = self._torch.linalg.vector_norm(audio_tensor, dim=-1, keepdim=True)
        text_norms = self._torch.linalg.vector_norm(text_tensor, dim=-1, keepdim=True)
        if (
            not self._torch.isfinite(audio_tensor).all()
            or not self._torch.isfinite(text_tensor).all()
            or not self._torch.isfinite(audio_norms).all()
            or not self._torch.isfinite(text_norms).all()
            or bool((audio_norms <= 0).any())
            or bool((text_norms <= 0).any())
        ):
            raise ValueError("music-CLAP embeddings contain NaN/Inf or a zero norm")
        normalized_audio = audio_tensor / audio_norms
        normalized_text = text_tensor / text_norms
        cosine = (normalized_audio * normalized_text).sum(dim=-1)
        return [common.finite_float(value, "music-CLAP cosine") for value in cosine.tolist()]


def _score_clap(
    backend: object,
    generation_dir: Path,
    records: Sequence[Mapping[str, object]],
    batch_size: int,
) -> List[float]:
    output: List[float] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        paths = [generation_dir / Path(str(record["path"])) for record in batch]
        prompts = [str(record["prompt"]) for record in batch]
        values = backend.score_batch(paths, prompts)
        if len(values) != len(batch):
            raise RuntimeError("music-CLAP backend returned the wrong batch length")
        output.extend(common.finite_float(value, "music-CLAP cosine") for value in values)
    return output


def run_music_clap(
    args: argparse.Namespace,
    *,
    clap_backend_factory=MusicClapBackend,
) -> int:
    if int(args.batch_size) <= 0:
        raise ValueError("batch-size must be positive")
    generation_dir = common.require_local_directory(Path(args.generation_dir), "generation artifact")
    quality_dir = common.require_local_directory(Path(args.quality_dir), "quality artifact")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("refusing to overwrite external evaluation artifact: {}".format(output_dir))

    generation_identity, generation_records = common.load_verified_generation(generation_dir)
    quality_provenance, quality_rows, quality_generation = common.verify_quality_artifact(
        quality_dir, generation_dir
    )
    if common.generation_binding(generation_identity) != common.generation_binding(quality_generation):
        raise ValueError("quality artifact belongs to a different generation artifact")
    quality_by_key = {
        (str(row["sample_id"]), str(row["condition_id"])): row for row in quality_rows
    }
    if len(quality_by_key) != len(quality_rows):
        raise ValueError("quality artifact contains duplicate keys")

    with common.staged_directory(output_dir) as staging:
        with common.deny_network_connections():
            clap_backend = clap_backend_factory(args)
            clap_identity = common.validate_evaluator_identity(clap_backend.identity, "music_clap")
            clap_scores = _score_clap(
                clap_backend, generation_dir, generation_records, int(args.batch_size)
            )
            del clap_backend
        if len(clap_scores) != len(generation_records):
            raise RuntimeError("music-CLAP output count mismatch")

        quality_evaluators = quality_provenance.get("evaluators")
        if not isinstance(quality_evaluators, dict):
            raise ValueError("quality provenance evaluator set is absent")
        final_provenance = {
            "schema_version": common.gate.EVALUATOR_PROVENANCE_SCHEMA_VERSION,
            "status": "accepted_external_evaluation",
            "metrics": list(common.gate.REQUIRED_METRICS),
            "evaluators": {
                "muq_eval": common.validate_evaluator_identity(
                    quality_evaluators.get("muq_eval"), "muq_eval"
                ),
                "audiobox_aesthetics": common.validate_evaluator_identity(
                    quality_evaluators.get("audiobox_aesthetics"), "audiobox_aesthetics"
                ),
                "music_clap": clap_identity,
            },
            "generation": common.generation_binding(generation_identity),
            "quality_artifact": {
                "artifact_seal_sha256": common.sha256_file(quality_dir / "artifact_seal.json"),
                "quality_scores_sha256": common.sha256_file(quality_dir / "quality_scores.jsonl"),
                "quality_provenance_sha256": common.sha256_file(
                    quality_dir / "quality_provenance.json"
                ),
            },
            "protocol": dict(common.gate.EXTERNAL_EVALUATION_PROTOCOL),
            "offline_environment": dict(common.OFFLINE_ENVIRONMENT),
        }
        provenance_path = staging / "evaluator_provenance.json"
        common.write_json(provenance_path, final_provenance)
        provenance_hash = common.sha256_file(provenance_path)

        final_rows: List[Dict[str, object]] = []
        for generated, clap_score in zip(generation_records, clap_scores):
            key = (str(generated["sample_id"]), str(generated["condition_id"]))
            quality = quality_by_key.get(key)
            if quality is None:
                raise ValueError("quality join is missing {}".format(key))
            metrics = quality.get("metrics")
            if not isinstance(metrics, dict):
                raise ValueError("quality metrics are absent for {}".format(key))
            final_rows.append(
                {
                    "schema_version": common.gate.SCORE_SCHEMA_VERSION,
                    "sample_id": generated["sample_id"],
                    "prompt_sha256": generated["prompt_sha256"],
                    "condition": generated["condition"],
                    "cfg_scale": generated["cfg_scale"],
                    "condition_id": generated["condition_id"],
                    "audio_sha256": generated["audio_sha256"],
                    "scientific_config_sha256": generated["scientific_config_sha256"],
                    "evaluator_provenance_sha256": provenance_hash,
                    "metrics": {
                        "muq_mi": common.finite_float(metrics.get("muq_mi"), "muq_mi"),
                        "audiobox_ce": common.finite_float(
                            metrics.get("audiobox_ce"), "audiobox_ce"
                        ),
                        "audiobox_pq": common.finite_float(
                            metrics.get("audiobox_pq"), "audiobox_pq"
                        ),
                        "music_clap": common.finite_float(clap_score, "music_clap"),
                    },
                }
            )
        scores_path = staging / "scores.jsonl"
        count = common.write_jsonl(scores_path, final_rows)
        matched, _ = common.gate.load_and_match_scores(
            generation_dir, scores_path, provenance_path
        )
        if len(matched) != count:
            raise RuntimeError("CFG gate rejected the completed external score set")
        common.write_json(
            staging / "artifact_seal.json",
            common.build_external_seal(
                scores_path=scores_path,
                provenance_path=provenance_path,
                quality_dir=quality_dir,
                record_count=count,
                generation_identity=generation_identity,
            ),
        )
        common.verify_external_artifact(staging, generation_dir, quality_dir)

    print(
        json.dumps(
            {
                "event": "cfg_external_evaluation_artifact_published",
                "output_dir": str(output_dir),
                "score_records": len(generation_records),
                "fad_computed": False,
                "artifact_seal_sha256": common.sha256_file(output_dir / "artifact_seal.json"),
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="compute music-CLAP and publish final gate inputs")
    run.add_argument("--generation-dir", type=Path, required=True)
    run.add_argument("--quality-dir", type=Path, required=True)
    run.add_argument("--clap-checkpoint", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--batch-size", type=int, default=8)

    verify = subparsers.add_parser("verify", help="rehash the completed external artifact")
    verify.add_argument("--generation-dir", type=Path, required=True)
    verify.add_argument("--quality-dir", type=Path, required=True)
    verify.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return run_music_clap(args)
    result = common.verify_external_artifact(
        Path(args.output_dir), Path(args.generation_dir), Path(args.quality_dir)
    )
    print(json.dumps({"event": "cfg_external_evaluation_artifact_verified", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
