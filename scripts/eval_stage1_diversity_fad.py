#!/usr/bin/env python3
"""Run offline MERT diversity and finite-only FAD pipeline checks.

All source WAVs and the CLAP checkpoint are copied into an independent
temporary workspace.  FADtk's ``convert/``, ``embeddings/``, and ``stats/``
caches can therefore never be created inside a generation/reference artifact.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple


WORKPACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKPACK_ROOT / "src"))

from ptc_opd.stage1_artifact import (  # noqa: E402
    Stage1ArtifactError,
    canonical_json_bytes,
    publish_closed_files_artifact,
    sha256_file,
    sha256_tree,
)
from ptc_opd.stage1_diversity_fad import (  # noqa: E402
    CLAP_BACKEND,
    CLAP_CHECKPOINT_BASENAME,
    DIVERSITY_ROW_SCHEMA,
    DIVERSITY_ROWS,
    EVALUATION_SCHEMA,
    EVALUATION_SEAL_SCHEMA,
    EVALUATION_SUMMARY,
    EVALUATOR_PROVENANCE,
    FADTK_PYTHON_FILE_COUNT,
    FADTK_SOURCE_SHA256,
    FADTK_UPSTREAM_COMMIT,
    FADTK_UPSTREAM_TAG,
    FADTK_VERSION,
    MERT_LAYER,
    MERT_MODEL_ID,
    MODEL_PINS_REPORT,
    PROVENANCE_SCHEMA,
    dereferenced_tree_identity,
    python_source_identity,
    verify_diversity_fad_artifact,
    verify_mert_scientific_snapshot,
    verify_model_pins_artifact,
    verify_reference_artifact,
)
from ptc_opd.stage1_generation import GENERATION_SEEDS, verify_generation_artifact  # noqa: E402


OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


def _canonical_existing(path: Path, label: str, *, directory: bool) -> Path:
    absolute = path.expanduser().absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise Stage1ArtifactError("{} path contains a symlink".format(label))
    resolved = absolute.resolve(strict=True)
    if directory and not resolved.is_dir():
        raise Stage1ArtifactError("{} must be a directory".format(label))
    if not directory and not resolved.is_file():
        raise Stage1ArtifactError("{} must be a regular file".format(label))
    return resolved


def _canonical_new_output(path: Path) -> Path:
    supplied = path.expanduser().absolute()
    parent = _canonical_existing(supplied.parent, "evaluation output parent", directory=True)
    output = parent / supplied.name
    if output.exists() or output.is_symlink():
        raise FileExistsError("refusing to overwrite evaluation artifact")
    return output


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--generation-dir", type=Path, required=True)
    result.add_argument("--eval-manifest-dir", type=Path, required=True)
    result.add_argument("--reference-dir", type=Path, required=True)
    result.add_argument("--a1-manifest", type=Path, required=True)
    result.add_argument("--a1-report", type=Path, required=True)
    result.add_argument("--model-pins-dir", type=Path, required=True)
    result.add_argument("--mert-snapshot", type=Path, required=True)
    result.add_argument("--clap-checkpoint", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--device", default="cuda:0")
    return result


@contextlib.contextmanager
def deny_network_connections() -> Iterator[None]:
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create = socket.create_connection

    def blocked(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("network access is forbidden during Stage-1 evaluation")

    socket.socket.connect = blocked  # type: ignore[assignment]
    socket.socket.connect_ex = blocked  # type: ignore[assignment]
    socket.create_connection = blocked  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[assignment]
        socket.socket.connect_ex = original_connect_ex  # type: ignore[assignment]
        socket.create_connection = original_create  # type: ignore[assignment]


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                dict(row),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _copy_audio(source_rows: Sequence[Mapping[str, Any]], source_root: Path, target: Path) -> List[Path]:
    target.mkdir()
    copied: List[Path] = []
    for index, row in enumerate(source_rows):
        relative = row.get("path")
        audio_hash = row.get("audio_sha256")
        if not isinstance(relative, str) or not isinstance(audio_hash, str):
            raise Stage1ArtifactError("audio source row lacks path/hash")
        source = (source_root / relative).resolve(strict=True)
        try:
            source.relative_to(source_root.resolve(strict=True))
        except ValueError as exc:
            raise Stage1ArtifactError("audio source path escapes artifact") from exc
        if source.is_symlink() or not source.is_file() or sha256_file(source) != audio_hash:
            raise Stage1ArtifactError("audio source identity differs before staging")
        name = "{:04d}-{}".format(index, source.name)
        destination = target / name
        shutil.copyfile(source, destination)
        if sha256_file(destination) != audio_hash:
            raise Stage1ArtifactError("staging copy SHA-256 differs")
        copied.append(destination)
    return copied


def _fadtk_runtime() -> Tuple[Any, Path, str]:
    version = importlib.metadata.version("fadtk")
    if version != FADTK_VERSION:
        raise Stage1ArtifactError("fadtk must be exactly {}".format(FADTK_VERSION))
    import fadtk

    source = getattr(fadtk, "__file__", None)
    if not isinstance(source, str):
        raise Stage1ArtifactError("cannot locate imported fadtk")
    return fadtk, Path(source).resolve().parent, version


def _cache_paths(fadtk: Any, model: Any, fad: Any, files: Sequence[Path]) -> List[Path]:
    from fadtk.utils import get_cache_embedding_path

    results = []
    for path in files:
        fad.cache_embedding_file(path)
        cache = get_cache_embedding_path(model.name, path)
        if cache.is_symlink() or not cache.is_file():
            raise Stage1ArtifactError("fadtk did not publish a regular embedding cache")
        results.append(cache)
    return results


def _finite_fad(value: Any, backend: str) -> float:
    if isinstance(value, bool) or isinstance(value, (str, bytes, bytearray, complex)):
        raise Stage1ArtifactError("{} FAD is not numeric".format(backend))
    try:
        # fadtk/scipy may return a NumPy scalar rather than a Python float.
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Stage1ArtifactError("{} FAD is not numeric".format(backend)) from exc
    if not math.isfinite(result):
        raise Stage1ArtifactError("{} FAD is non-finite".format(backend))
    return result


def _make_mert(fadtk: Any, snapshot: Path, device: Any) -> Any:
    model = fadtk.MERTModel(size="v1-95M", layer=MERT_LAYER)
    model.huggingface_id = str(snapshot)
    model.device = device
    return model


def _make_clap(fadtk: Any, checkpoint: Path, device: Any) -> Any:
    # Avoid CLAPLaionModel.__init__: it owns a package-local checkpoint path and
    # downloads when absent.  Initializing the public base fields plus the two
    # documented subclass fields gives load_model/_get_embedding the exact
    # fadtk 1.1.0 state while keeping every mutable byte in staging.
    model = object.__new__(fadtk.CLAPLaionModel)
    fadtk.ModelLoader.__init__(model, CLAP_BACKEND, 512, 48_000)
    model.type = "music"
    model.model_file = checkpoint
    model.device = device
    return model


def _mert_diversity_rows(
    *,
    generation_rows: Sequence[Mapping[str, Any]],
    cache_paths: Sequence[Path],
    provenance_hash: str,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    import numpy as np

    cache_by_key: Dict[Tuple[str, int], Path] = {}
    source_by_key: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for source, cache in zip(generation_rows, cache_paths):
        key = (str(source["sample_id"]), int(source["generation_seed"]))
        if key in cache_by_key:
            raise Stage1ArtifactError("duplicate generation key")
        cache_by_key[key] = cache
        source_by_key[key] = source
    sample_ids = sorted({key[0] for key in cache_by_key})
    rows: List[Dict[str, Any]] = []
    distances: List[float] = []
    for sample_id in sample_ids:
        vectors = []
        embedding_hashes: Dict[str, str] = {}
        frame_counts: Dict[str, int] = {}
        audio_hashes: Dict[str, str] = {}
        condition_id = None
        for seed in GENERATION_SEEDS:
            key = (sample_id, seed)
            cache = cache_by_key[key]
            embedding = np.load(cache, allow_pickle=False)
            if (
                embedding.ndim != 2
                or embedding.shape[0] <= 0
                or embedding.shape[1] != 768
                or not bool(np.isfinite(embedding).all())
            ):
                raise Stage1ArtifactError("MERT embedding shape/finite contract differs")
            pooled = np.mean(embedding.astype(np.float64), axis=0, dtype=np.float64)
            norm = float(np.linalg.norm(pooled))
            if not math.isfinite(norm) or norm <= 0.0:
                raise Stage1ArtifactError("MERT pooled embedding has zero/non-finite norm")
            vectors.append(pooled / norm)
            source = source_by_key[key]
            source_condition = str(source["condition_id"])
            if condition_id is None:
                condition_id = source_condition
            elif condition_id != source_condition:
                raise Stage1ArtifactError("seed pair crosses generation conditions")
            embedding_hashes[str(seed)] = sha256_file(cache)
            frame_counts[str(seed)] = int(embedding.shape[0])
            audio_hashes[str(seed)] = str(source["audio_sha256"])
        similarity = float(np.dot(vectors[0], vectors[1]))
        similarity = max(-1.0, min(1.0, similarity))
        distance = 1.0 - similarity
        rows.append(
            {
                "schema_version": DIVERSITY_ROW_SCHEMA,
                "sample_id": sample_id,
                "condition_id": condition_id,
                "generation_seeds": list(GENERATION_SEEDS),
                "audio_sha256_by_seed": audio_hashes,
                "embedding_sha256_by_seed": embedding_hashes,
                "embedding_frame_count_by_seed": frame_counts,
                "model_id": MERT_MODEL_ID,
                "layer": MERT_LAYER,
                "pooling": "arithmetic_mean_over_frame_axis",
                "normalization": "L2_after_frame_mean",
                "cosine_similarity": similarity,
                "cosine_distance": distance,
                "evaluator_provenance_sha256": provenance_hash,
            }
        )
        distances.append(distance)
    return rows, distances


def run(args: argparse.Namespace) -> Path:
    import numpy as np
    import torch

    # Ruling #6 3rd addendum bypass: PyTorch 2.6 changed torch.load's default
    # weights_only from False to True, breaking laion_clap's
    # clap_module/factory.py:54 which calls torch.load(ckpt, map_location=...)
    # without weights_only.  The CLAP checkpoint contains
    # numpy.core.multiarray.scalar objects, which are safe to trust (it's the
    # frozen CLAP-LAION music_audioset checkpoint pinned by upstream), but the
    # new default rejects them.  Monkey-patch torch.load to default
    # weights_only=False so laion_clap can load its checkpoint.  We do NOT
    # touch laion_clap library code (avoid polluting the conda env).
    _original_torch_load = torch.load
    def _torch_load_full(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return _original_torch_load(*load_args, **load_kwargs)
    torch.load = _torch_load_full

    generation_input = _canonical_existing(
        args.generation_dir, "generation", directory=True
    )
    eval_manifest_root = _canonical_existing(
        args.eval_manifest_dir, "evaluation manifest", directory=True
    )
    reference_root = _canonical_existing(args.reference_dir, "reference", directory=True)
    model_pins_root = _canonical_existing(
        args.model_pins_dir, "model pins", directory=True
    )
    mert_root = _canonical_existing(args.mert_snapshot, "MERT snapshot", directory=True)
    a1_manifest = _canonical_existing(args.a1_manifest, "A1 manifest", directory=False)
    a1_report = _canonical_existing(args.a1_report, "A1 report", directory=False)
    clap_checkpoint = _canonical_existing(
        args.clap_checkpoint, "CLAP checkpoint", directory=False
    )
    if not torch.cuda.is_available():
        raise RuntimeError("formal MERT/FAD evaluation requires CUDA")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise Stage1ArtifactError("--device must be CUDA")
    torch.cuda.set_device(device)
    # FADtk 1.1.0's ModelLoader.get_embedding compares its device with the
    # index-free ``torch.device('cuda')`` before moving tensors to CPU.  Use
    # that exact representation after selecting the requested current device;
    # ``cuda`` now resolves to args.device without triggering its CUDA->NumPy
    # bug for an explicitly indexed device object.
    fadtk_device = torch.device("cuda")
    for name, value in OFFLINE_ENVIRONMENT.items():
        os.environ[name] = value
    generation = verify_generation_artifact(
        generation_input,
        eval_manifest_dir=eval_manifest_root,
        rehash_pcm=False,
    )
    generation_root = Path(generation["directory"])
    reference = verify_reference_artifact(
        reference_root, a1_manifest=a1_manifest, a1_report=a1_report
    )
    fadtk, fadtk_root, fadtk_version = _fadtk_runtime()
    pins = verify_model_pins_artifact(
        model_pins_root,
        mert_snapshot=mert_root,
        clap_checkpoint=clap_checkpoint,
        fadtk_package_root=fadtk_root,
        fadtk_version=fadtk_version,
    )
    output = _canonical_new_output(args.output_dir)
    protected_roots = [
        generation_root,
        reference_root,
        model_pins_root,
        mert_root,
    ]
    if any(root == output or root in output.parents or output in root.parents for root in protected_roots):
        raise Stage1ArtifactError("evaluation output must be outside every immutable input")

    generation_before = sha256_tree(generation_root)
    reference_before = sha256_tree(reference_root)
    mert_before = dereferenced_tree_identity(mert_root)["tree_sha256"]
    clap_before = sha256_file(clap_checkpoint)
    compute = Path(
        tempfile.mkdtemp(prefix=".{}.compute.".format(output.name), dir=str(output.parent))
    )
    try:
        for directory in (compute / "hf_home", compute / "torch_home", compute / "tmp"):
            directory.mkdir()
        # Ruling #6 3rd addendum bypass: the original hermetic override
        # points HF_HOME at an empty compute-local dir, which forces
        # laion_clap's BertModel.from_pretrained("bert-base-uncased") to
        # fail under HF_HUB_OFFLINE=1.  Preserve TORCH_HOME/TMPDIR isolation
        # but route HF cache to the pre-populated shared hub cache instead.
        _shared_hf_home = os.environ.get(
            "PTC_SHARED_HF_HOME",
            "<local>/models/ICASSP2027/huggingface",
        )
        os.environ["HF_HOME"] = _shared_hf_home
        os.environ["HF_HUB_CACHE"] = str(Path(_shared_hf_home) / "hub")
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(_shared_hf_home) / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(Path(_shared_hf_home) / "hub")
        os.environ["TORCH_HOME"] = str(compute / "torch_home")
        os.environ["TMPDIR"] = str(compute / "tmp")
        os.environ["TORCH_HOME"] = str(compute / "torch_home")
        os.environ["TMPDIR"] = str(compute / "tmp")
        generated_audio = _copy_audio(
            generation["samples"], generation_root, compute / "generated_audio"
        )
        reference_audio = _copy_audio(
            reference["rows"], reference_root, compute / "reference_audio"
        )
        resource_root = compute / "resources"
        resource_root.mkdir()
        staged_clap = resource_root / CLAP_CHECKPOINT_BASENAME
        shutil.copyfile(clap_checkpoint, staged_clap)
        if sha256_file(staged_clap) != clap_before:
            raise Stage1ArtifactError("staged CLAP checkpoint differs")

        evaluator_details = {
            "fadtk": {
                "distribution_version": fadtk_version,
                "upstream_tag": FADTK_UPSTREAM_TAG,
                "upstream_commit": FADTK_UPSTREAM_COMMIT,
                **python_source_identity(fadtk_root),
            },
            "mert": {
                "model_id": MERT_MODEL_ID,
                "revision": pins["report"]["mert"]["revision"],
                "layer": MERT_LAYER,
                "embedding_cache_dtype": "float16_by_fadtk_ModelLoader_get_embedding",
                "diversity": "frame arithmetic mean -> L2 -> paired cosine distance",
                "fad": "fadtk.FrechetAudioDistance.score over frame embeddings",
            },
            "clap_laion_music": {
                "backend": CLAP_BACKEND,
                "checkpoint_sha256": clap_before,
                "checkpoint_loaded_from_staging_copy": True,
                "fad": "fadtk.FrechetAudioDistance.score",
            },
            "runtime_device": str(device),
        }
        provenance = {
            "schema_version": PROVENANCE_SCHEMA,
            "status": "accepted_offline_evaluation",
            "generation_artifact_seal_sha256": generation["artifact_seal_sha256"],
            "reference_artifact_seal_sha256": reference["artifact_seal_sha256"],
            "model_pins_artifact_seal_sha256": pins["artifact_seal_sha256"],
            "offline_environment": dict(OFFLINE_ENVIRONMENT),
            "network_access_forbidden": True,
            "evaluator": evaluator_details,
            # Filled after both backends and rehashed before publication.
            "generation_integrity": None,
            "reference_integrity": None,
            "resource_integrity": None,
            "staging_policy": {
                "generation_audio_copied_to_independent_staging": True,
                "reference_audio_copied_to_independent_staging": True,
                "clap_checkpoint_copied_to_independent_staging": True,
                "fadtk_cache_roots_inside_staging_only": True,
                "generation_artifact_mutation_forbidden": True,
            },
        }

        with deny_network_connections():
            verify_mert_scientific_snapshot(mert_root)
            mert_model = _make_mert(
                fadtk, mert_root, fadtk_device
            )
            mert_fad = fadtk.FrechetAudioDistance(
                mert_model, audio_load_worker=1, load_model=True
            )
            if hasattr(mert_model.model, "eval"):
                mert_model.model.eval()
            mert_reference_cache = _cache_paths(
                fadtk, mert_model, mert_fad, reference_audio
            )
            mert_generated_cache = _cache_paths(
                fadtk, mert_model, mert_fad, generated_audio
            )
            mert_fad_score = _finite_fad(
                mert_fad.score(compute / "reference_audio", compute / "generated_audio"),
                "MERT-v1-95M-layer12",
            )
            del mert_fad, mert_model
            torch.cuda.empty_cache()

            clap_model = _make_clap(fadtk, staged_clap, fadtk_device)
            clap_fad = fadtk.FrechetAudioDistance(
                clap_model, audio_load_worker=1, load_model=True
            )
            if hasattr(clap_model.model, "eval"):
                clap_model.model.eval()
            _cache_paths(fadtk, clap_model, clap_fad, reference_audio)
            _cache_paths(fadtk, clap_model, clap_fad, generated_audio)
            clap_fad_score = _finite_fad(
                clap_fad.score(compute / "reference_audio", compute / "generated_audio"),
                CLAP_BACKEND,
            )
            del clap_fad, clap_model
            torch.cuda.empty_cache()

        generation_after = sha256_tree(generation_root)
        reference_after = sha256_tree(reference_root)
        mert_after = dereferenced_tree_identity(mert_root)["tree_sha256"]
        verify_mert_scientific_snapshot(mert_root)
        clap_after = sha256_file(clap_checkpoint)
        if generation_before != generation_after:
            raise Stage1ArtifactError("generation artifact mutated during FAD evaluation")
        if reference_before != reference_after:
            raise Stage1ArtifactError("reference artifact mutated during FAD evaluation")
        if mert_before != mert_after or clap_before != clap_after:
            raise Stage1ArtifactError("model resources mutated during FAD evaluation")
        provenance["generation_integrity"] = {
            "before_tree_sha256": generation_before,
            "after_tree_sha256": generation_after,
            "unchanged": True,
        }
        provenance["reference_integrity"] = {
            "before_tree_sha256": reference_before,
            "after_tree_sha256": reference_after,
            "unchanged": True,
        }
        provenance["resource_integrity"] = {
            "mert_snapshot_before_tree_sha256": mert_before,
            "mert_snapshot_after_tree_sha256": mert_after,
            "clap_checkpoint_before_sha256": clap_before,
            "clap_checkpoint_after_sha256": clap_after,
            "unchanged": True,
        }
        provenance_bytes = canonical_json_bytes(provenance)
        provenance_hash = __import__("hashlib").sha256(provenance_bytes).hexdigest()
        diversity_rows, distances = _mert_diversity_rows(
            generation_rows=generation["samples"],
            cache_paths=mert_generated_cache,
            provenance_hash=provenance_hash,
        )
        mean_distance = math.fsum(distances) / len(distances)
        condition_ids = {str(row["condition_id"]) for row in generation["samples"]}
        if len(condition_ids) != 1:
            raise Stage1ArtifactError("generation artifact must contain one condition")
        summary = {
            "schema_version": EVALUATION_SCHEMA,
            "status": "complete_diversity_fad_pipeline_check",
            "condition_id": next(iter(condition_ids)),
            "generation_artifact_seal_sha256": generation["artifact_seal_sha256"],
            "reference_artifact_seal_sha256": reference["artifact_seal_sha256"],
            "model_pins_artifact_seal_sha256": pins["artifact_seal_sha256"],
            "evaluator_provenance_sha256": provenance_hash,
            "mert_diversity": {
                "model_id": MERT_MODEL_ID,
                "layer": MERT_LAYER,
                "prompt_pair_count": len(distances),
                "pairing": "same_prompt_seed31001_vs_seed31002",
                "pooling": "frame_mean_then_L2",
                "mean_cosine_distance": mean_distance,
                "min_cosine_distance": min(distances),
                "max_cosine_distance": max(distances),
                "selection_role": "pilot_point_estimate_gate",
            },
            "fad_pipeline_checks": {
                CLAP_BACKEND: {
                    "score": clap_fad_score,
                    "finite": True,
                    "pipeline_check_passed": True,
                    "selection_use_forbidden": True,
                    "paper_claim_use_forbidden": True,
                },
                "MERT-v1-95M-layer12": {
                    "score": mert_fad_score,
                    "finite": True,
                    "pipeline_check_passed": True,
                    "selection_use_forbidden": True,
                    "paper_claim_use_forbidden": True,
                },
            },
            "selection_contract": {
                "mert_diversity_used_only_as_predeclared_pilot_point_estimate": True,
                "fad_used_for_model_or_checkpoint_selection": False,
                "fad_used_for_paper_claim": False,
            },
        }
        closed_candidate = compute / "closed_artifact"
        publish_closed_files_artifact(
            closed_candidate,
            payloads={
                EVALUATION_SUMMARY: canonical_json_bytes(summary),
                EVALUATOR_PROVENANCE: provenance_bytes,
                DIVERSITY_ROWS: _jsonl_bytes(diversity_rows),
            },
            seal_schema=EVALUATION_SEAL_SCHEMA,
            seal_status="complete_diversity_fad_pipeline_check",
        )
        # A final directory is never exposed until the complete CPU verifier
        # has accepted the closed candidate against every immutable input.
        verify_diversity_fad_artifact(
            closed_candidate,
            generation_dir=generation_root,
            eval_manifest_dir=eval_manifest_root,
            reference_dir=reference_root,
            a1_manifest=a1_manifest,
            a1_report=a1_report,
            model_pins_dir=model_pins_root,
        )
        os.replace(str(closed_candidate), str(output))
    finally:
        shutil.rmtree(compute, ignore_errors=True)
    verify_diversity_fad_artifact(
        output,
        generation_dir=generation_root,
        eval_manifest_dir=eval_manifest_root,
        reference_dir=reference_root,
        a1_manifest=a1_manifest,
        a1_report=a1_report,
        model_pins_dir=model_pins_root,
    )
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    try:
        output = run(args)
    except (OSError, RuntimeError, ValueError, Stage1ArtifactError) as exc:
        print(
            json.dumps(
                {"status": "invalid", "error_type": type(exc).__name__, "error": str(exc)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 3
    print(json.dumps({"status": "complete", "output_dir": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
