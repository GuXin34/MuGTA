#!/usr/bin/env python3
"""Run the quality half of the frozen CFG-scale gate.

The command is intended for the Python 3.11 ``ptc-opd-eval-quality``
environment.  It evaluates the sealed generation artifact with the exact
pinned MuQ-Eval source plus local A1 files and with a local Audiobox
Aesthetics checkpoint.  It never downloads a model and atomically publishes
a sealed partial artifact for the music-CLAP environment.
"""

from __future__ import annotations

import argparse
import importlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import cfg_eval_common as common


ENVIRONMENT_REPORT_BASENAME = "environment_acceptance_20260812.md"
EXPECTED_QUALITY_RUNTIME_VERSIONS = {
    "muq": "0.1.0",
    "numpy": "1.26.4",
    "soundfile": "0.12.1",
    "torch": "2.2.2+cu121",
    "torchaudio": "2.2.2+cu121",
}


def _module_under(module: object, root: Path, label: str) -> None:
    source = getattr(module, "__file__", None)
    if not isinstance(source, str) or not source:
        raise RuntimeError("cannot locate loaded {} module".format(label))
    try:
        Path(source).resolve().relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            "{} was imported outside the pinned source root: {}".format(label, source)
        ) from exc


def _jsonable(value: object) -> object:
    """Convert checkpoint configuration scalars into canonical JSON values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("checkpoint configuration contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item") and callable(getattr(value, "item")):
        scalar = value.item()
        if scalar is value:
            raise TypeError("unsupported checkpoint configuration value {!r}".format(type(value)))
        return _jsonable(scalar)
    raise TypeError("unsupported checkpoint configuration value {!r}".format(type(value)))


def _torch_load(path: Path, *, map_location: object) -> object:
    import torch

    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def _distribution_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "required eval-quality distribution is not installed: {}".format(distribution)
        ) from exc


def _quality_runtime_identity(
    environment_report: Path,
    *,
    imported_modules: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Bind the accepted environment report and the packages used at runtime."""

    if imported_modules is None:
        imported_modules = {
            name: importlib.import_module(name)
            for name in ("numpy", "soundfile", "torch", "torchaudio")
        }
    if set(imported_modules) != {"numpy", "soundfile", "torch", "torchaudio"}:
        raise ValueError("runtime module identity must cover numpy/soundfile/torch/torchaudio")

    observed = {
        distribution: _distribution_version(distribution)
        for distribution in sorted(EXPECTED_QUALITY_RUNTIME_VERSIONS)
    }
    mismatches = {
        distribution: {
            "expected": expected,
            "observed": observed[distribution],
        }
        for distribution, expected in sorted(EXPECTED_QUALITY_RUNTIME_VERSIONS.items())
        if observed[distribution] != expected
    }
    if mismatches:
        raise RuntimeError(
            "eval-quality runtime version mismatch: {}".format(
                json.dumps(mismatches, sort_keys=True)
            )
        )
    imported_versions = {
        name: str(getattr(module, "__version__", ""))
        for name, module in sorted(imported_modules.items())
    }
    import_mismatches = {
        name: {
            "distribution_metadata": observed[name],
            "imported_module": version,
        }
        for name, version in sorted(imported_versions.items())
        if version != observed[name]
    }
    if import_mismatches:
        raise RuntimeError(
            "eval-quality imported module/version mismatch: {}".format(
                json.dumps(import_mismatches, sort_keys=True)
            )
        )
    report = common.require_local_file(
        environment_report,
        "accepted environment report",
        basename=ENVIRONMENT_REPORT_BASENAME,
    )
    return {
        "accepted_environment_report": {
            "basename": report.name,
            "sha256": common.sha256_file(report),
        },
        "distributions": observed,
        "imported_modules": imported_versions,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": str(Path(sys.executable).resolve()),
        },
    }


def _muq_source_identity(
    *,
    muq_eval_source: Mapping[str, object],
    muq_module: object,
) -> Tuple[str, Dict[str, object]]:
    """Combine the pinned evaluator checkout with the imported MuQ package."""

    package = common.imported_package_identity(muq_module, "muq")
    package_root = Path(str(package["package_root"]))
    components = {
        "muq_eval_repository": {
            "git_commit": str(muq_eval_source["git_commit"]),
            "tracked_source_sha256": common.require_sha256(
                muq_eval_source["source_sha256"], "MuQ-Eval tracked source"
            ),
        },
        "muq_python_package": {
            "distribution": "muq",
            "distribution_version": _distribution_version("muq"),
            "python_source_sha256": common.require_sha256(
                package["source_sha256"], "muq Python package source"
            ),
        },
    }
    details = {
        "components": components,
        "muq_python_package": {
            "package_root": str(package_root),
            **common.package_file_hashes(package_root),
        },
    }
    return common.sha256_json(components), details


def _load_muq_configuration(config_path: Path, backbone: Path) -> Tuple[object, Dict[str, object]]:
    from omegaconf import OmegaConf

    config_path = common.require_local_file(config_path, "MuQ-Eval A1 config")
    cfg = OmegaConf.load(str(config_path))
    config_files = [config_path]
    if "defaults" in cfg:
        base_path = common.require_local_file(config_path.parent / "base.yaml", "MuQ-Eval base config")
        base_cfg = OmegaConf.load(str(base_path))
        cfg = OmegaConf.merge(base_cfg, cfg)
        config_files.append(base_path)

    required = (
        str(cfg.get("experiment", {}).get("name", "")) == "A1_frozen_mlp",
        str(cfg.get("model", {}).get("encoder", "")) == "muq",
        str(cfg.get("model", {}).get("tuning_mode", "")) == "frozen",
        str(cfg.get("loss", {}).get("type", "")) == "mse",
        int(cfg.get("data", {}).get("sample_rate", -1)) == 24000,
        int(cfg.get("data", {}).get("clip_samples", -1)) == 240000,
    )
    if not all(required):
        raise ValueError("MuQ-Eval configuration is not the released A1 FP32 inference contract")
    heads = [str(item.get("name", "")) for item in cfg.model.heads]
    if "MI" not in heads:
        raise ValueError("MuQ-Eval A1 configuration has no MI head")

    raw_encoder_id = str(cfg.model.encoder_id)
    if raw_encoder_id != "OpenMuQ/MuQ-large-msd-iter":
        raise ValueError(
            "MuQ-Eval A1 must declare encoder_id OpenMuQ/MuQ-large-msd-iter"
        )
    cfg.model.encoder_id = str(backbone)
    effective = OmegaConf.to_container(cfg, resolve=True)
    config_file_set_sha256 = common.hash_config_files(config_files)
    effective_config_sha256 = common.sha256_json(effective)
    runtime_config = {
        "effective_model_config_sha256": effective_config_sha256,
        "released_config_file_set_sha256": config_file_set_sha256,
        "local_encoder_snapshot_sha256": common.sha256_local_tree(backbone),
        "audio_processor": {
            "target_sr": 24000,
            "clip_samples": 240000,
            "mode": "center",
        },
        "parameter_dtype": "torch.float32",
        "autocast": False,
    }
    identity = {
        # Provenance's config hash identifies the exact resolved configuration
        # that actually constructs the model plus preprocessing/precision.
        # Individual released YAML bytes remain recorded below.
        "config_sha256": common.sha256_json(runtime_config),
        "runtime_config": runtime_config,
        "config_file_set_sha256": config_file_set_sha256,
        "config_files": [
            {"basename": path.name, "sha256": common.sha256_file(path)}
            for path in sorted(config_files, key=lambda item: item.name)
        ],
        "declared_encoder_id": raw_encoder_id,
        "local_encoder_snapshot_sha256": runtime_config["local_encoder_snapshot_sha256"],
        "effective_config_sha256": effective_config_sha256,
        "audio_processor": {
            "target_sr": 24000,
            "clip_samples": 240000,
            "mode": "center",
        },
        "parameter_dtype": "torch.float32",
        "autocast": False,
    }
    return cfg, identity


class MuQBackend:
    def __init__(self, args: argparse.Namespace):
        import numpy as np
        import soundfile as sf
        import torch

        self._np = np
        self._sf = sf
        source_root = common.require_local_directory(Path(args.muq_eval_root), "MuQ-Eval source root")
        source_identity = common.pinned_git_source_identity(source_root)
        runtime_identity = _quality_runtime_identity(Path(args.environment_report))
        config_path = common.require_local_file(Path(args.muq_config), "MuQ-Eval A1 config")
        state_path = common.require_local_file(
            Path(args.muq_state_dict), "MuQ-Eval A1 state dict", basename="model_state_dict.pt"
        )
        backbone = common.require_local_directory(Path(args.muq_backbone), "local MuQ backbone")
        cfg, config_identity = _load_muq_configuration(config_path, backbone)

        for name, module in list(sys.modules.items()):
            if name == "src" or name.startswith("src."):
                _module_under(module, source_root, name)
        sys.path.insert(0, str(source_root))
        try:
            model_module = importlib.import_module("src.model")
            data_module = importlib.import_module("src.data")
        finally:
            try:
                sys.path.remove(str(source_root))
            except ValueError:
                pass
        _module_under(model_module, source_root, "src.model")
        _module_under(data_module, source_root, "src.data")
        muq_module = importlib.import_module("muq")
        source_sha256, muq_source_details = _muq_source_identity(
            muq_eval_source=source_identity,
            muq_module=muq_module,
        )

        device = torch.device(str(args.device))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("MuQ evaluation requested CUDA but CUDA is unavailable")
        model = model_module.MusicQualityModel(cfg)
        state = _torch_load(state_path, map_location="cpu")
        state_container = "raw_state_dict"
        if isinstance(state, dict) and "model_state_dict" in state:
            state_container = "model_state_dict"
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "model_state" in state:
            state_container = "model_state"
            state = state["model_state"]
        if not isinstance(state, Mapping):
            raise TypeError("MuQ-Eval model_state_dict.pt does not contain a state mapping")
        model.load_state_dict(state, strict=True)
        model.float().to(device).requires_grad_(False).eval()
        wrong_parameters = {
            name: str(parameter.dtype)
            for name, parameter in model.named_parameters()
            if parameter.is_floating_point() and parameter.dtype != torch.float32
        }
        wrong_buffers = {
            name: str(buffer.dtype)
            for name, buffer in model.named_buffers()
            if buffer.is_floating_point() and buffer.dtype != torch.float32
        }
        if wrong_parameters or wrong_buffers:
            raise RuntimeError("MuQ-Eval model is not entirely FP32")

        self.model = model
        self.device = device
        self.processor = data_module.AudioProcessor(target_sr=24000, clip_samples=240000)
        self.identity = {
            "checkpoint_sha256": common.sha256_file(state_path),
            # This is a composite of the pinned MuQ-Eval checkout and the
            # installed ``muq`` package that executes the encoder, not merely
            # the thin evaluator repository.
            "source_sha256": source_sha256,
            "config_sha256": config_identity["config_sha256"],
            "details": {
                "source_git_commit": source_identity["git_commit"],
                "source_identity": muq_source_details,
                "runtime_environment": runtime_identity,
                **config_identity,
                "runtime_device": str(device),
                "model_class": "src.model.MusicQualityModel",
                "checkpoint_state_container": state_container,
            },
        }

    def score_batch(self, audio_paths: Sequence[Path]) -> List[float]:
        import torch

        processed = []
        for path in audio_paths:
            array, source_sr = self._sf.read(str(path), dtype="float32", always_2d=True)
            mono = self._np.asarray(array.mean(axis=1), dtype=self._np.float32)
            waveform = self.processor.process(mono, int(source_sr), mode="center")
            if tuple(waveform.shape) != (240000,):
                raise RuntimeError("MuQ AudioProcessor returned shape {}".format(tuple(waveform.shape)))
            processed.append(waveform)
        batch = torch.stack(processed, dim=0).to(self.device, dtype=torch.float32)
        with torch.inference_mode():
            predictions = self.model(batch)
        expected = getattr(self.model, "_last_expected_scores", {})
        values = expected.get("MI") if isinstance(expected, dict) else None
        if values is None and isinstance(predictions, dict):
            values = predictions.get("MI")
        if not isinstance(values, torch.Tensor) or values.ndim != 1 or values.shape[0] != len(audio_paths):
            raise RuntimeError("MuQ-Eval MI output must be one scalar per input")
        return [common.finite_float(value, "MuQ MI") for value in values.detach().float().cpu().tolist()]


class AudioboxBackend:
    def __init__(self, args: argparse.Namespace):
        import torch
        import audiobox_aesthetics
        from audiobox_aesthetics.infer import initialize_predictor

        checkpoint = common.require_local_file(
            Path(args.audiobox_checkpoint), "Audiobox Aesthetics checkpoint", basename="checkpoint.pt"
        )
        checkpoint_payload = _torch_load(checkpoint, map_location="cpu")
        if not isinstance(checkpoint_payload, Mapping):
            raise TypeError("Audiobox checkpoint must contain a mapping")
        if "model_cfg" not in checkpoint_payload or "target_transform" not in checkpoint_payload:
            raise ValueError("Audiobox checkpoint lacks embedded model_cfg/target_transform")
        embedded_config = {
            "model_cfg": _jsonable(checkpoint_payload["model_cfg"]),
            "target_transform": _jsonable(checkpoint_payload["target_transform"]),
        }
        predictor = initialize_predictor(str(checkpoint))
        if not callable(getattr(predictor, "forward", None)):
            raise TypeError("initialize_predictor(local_checkpoint) returned no forward method")
        source_identity = common.imported_package_identity(audiobox_aesthetics, "audiobox_aesthetics")
        source_details = common.package_file_hashes(Path(str(source_identity["package_root"])))
        try:
            package_version = metadata.version("audiobox-aesthetics")
        except metadata.PackageNotFoundError:
            package_version = "unknown"
        self.predictor = predictor
        self.identity = {
            "checkpoint_sha256": common.sha256_file(checkpoint),
            "source_sha256": source_identity["source_sha256"],
            "config_sha256": common.sha256_json(embedded_config),
            "details": {
                "embedded_checkpoint_config": embedded_config,
                "package_version": package_version,
                **source_details,
                "runtime_device": str(getattr(predictor, "device", "unknown")),
                "api": "audiobox_aesthetics.infer.initialize_predictor(local_ckpt).forward",
                "axes_consumed": ["CE", "PQ"],
            },
        }

    def score_batch(self, audio_paths: Sequence[Path]) -> List[Dict[str, float]]:
        values = self.predictor.forward([{"path": str(path)} for path in audio_paths])
        if not isinstance(values, list) or len(values) != len(audio_paths):
            raise RuntimeError("Audiobox forward must return one result per input")
        results: List[Dict[str, float]] = []
        for value in values:
            if not isinstance(value, Mapping):
                raise TypeError("Audiobox result must be an object")
            results.append(
                {
                    "audiobox_ce": common.finite_float(value.get("CE"), "Audiobox CE"),
                    "audiobox_pq": common.finite_float(value.get("PQ"), "Audiobox PQ"),
                }
            )
        return results


def _audio_paths(generation_dir: Path, records: Sequence[Mapping[str, object]]) -> List[Path]:
    return [generation_dir / Path(str(record["path"])) for record in records]


def _score_muq(backend: object, paths: Sequence[Path], batch_size: int) -> List[float]:
    output: List[float] = []
    for start in range(0, len(paths), batch_size):
        values = backend.score_batch(paths[start : start + batch_size])
        if len(values) != len(paths[start : start + batch_size]):
            raise RuntimeError("MuQ backend returned the wrong batch length")
        output.extend(common.finite_float(value, "MuQ MI") for value in values)
    return output


def _score_audiobox(backend: object, paths: Sequence[Path], batch_size: int) -> List[Dict[str, float]]:
    output: List[Dict[str, float]] = []
    for start in range(0, len(paths), batch_size):
        values = backend.score_batch(paths[start : start + batch_size])
        if len(values) != len(paths[start : start + batch_size]):
            raise RuntimeError("Audiobox backend returned the wrong batch length")
        for value in values:
            if not isinstance(value, Mapping):
                raise TypeError("Audiobox backend result must be an object")
            output.append(
                {
                    "audiobox_ce": common.finite_float(value.get("audiobox_ce"), "Audiobox CE"),
                    "audiobox_pq": common.finite_float(value.get("audiobox_pq"), "Audiobox PQ"),
                }
            )
    return output


def run_quality(
    args: argparse.Namespace,
    *,
    muq_backend_factory=MuQBackend,
    audiobox_backend_factory=AudioboxBackend,
) -> int:
    if int(args.batch_size) <= 0:
        raise ValueError("batch-size must be positive")
    generation_dir = common.require_local_directory(Path(args.generation_dir), "generation artifact")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("refusing to overwrite quality artifact: {}".format(output_dir))
    generation_identity, records = common.load_verified_generation(generation_dir)
    paths = _audio_paths(generation_dir, records)

    with common.staged_directory(output_dir) as staging:
        with common.deny_network_connections():
            muq_backend = muq_backend_factory(args)
            muq_identity = common.validate_evaluator_identity(muq_backend.identity, "muq_eval")
            muq_scores = _score_muq(muq_backend, paths, int(args.batch_size))
            del muq_backend

            audiobox_backend = audiobox_backend_factory(args)
            audiobox_identity = common.validate_evaluator_identity(
                audiobox_backend.identity, "audiobox_aesthetics"
            )
            audiobox_scores = _score_audiobox(audiobox_backend, paths, int(args.batch_size))
            del audiobox_backend

        if not (len(records) == len(muq_scores) == len(audiobox_scores)):
            raise RuntimeError("quality evaluator output count mismatch")
        provenance = {
            "schema_version": common.QUALITY_PROVENANCE_SCHEMA_VERSION,
            "status": "accepted_quality_evaluation",
            "metrics": list(common.QUALITY_METRICS),
            "generation": common.generation_binding(generation_identity),
            "evaluators": {
                "muq_eval": muq_identity,
                "audiobox_aesthetics": audiobox_identity,
            },
            "offline_environment": dict(common.OFFLINE_ENVIRONMENT),
        }
        provenance_path = staging / "quality_provenance.json"
        common.write_json(provenance_path, provenance)
        provenance_hash = common.sha256_file(provenance_path)
        quality_rows = [
            common.quality_row(
                generated,
                provenance_sha256=provenance_hash,
                muq_mi=muq_score,
                audiobox_ce=audiobox_score["audiobox_ce"],
                audiobox_pq=audiobox_score["audiobox_pq"],
            )
            for generated, muq_score, audiobox_score in zip(records, muq_scores, audiobox_scores)
        ]
        scores_path = staging / "quality_scores.jsonl"
        count = common.write_jsonl(scores_path, quality_rows)
        common.write_json(
            staging / "artifact_seal.json",
            common.build_quality_seal(
                scores_path=scores_path,
                provenance_path=provenance_path,
                record_count=count,
                generation_identity=generation_identity,
            ),
        )
        common.verify_quality_artifact(staging, generation_dir)

    print(
        json.dumps(
            {
                "event": "cfg_quality_artifact_published",
                "output_dir": str(output_dir),
                "score_records": len(records),
                "artifact_seal_sha256": common.sha256_file(output_dir / "artifact_seal.json"),
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="evaluate and atomically publish quality scores")
    run.add_argument("--generation-dir", type=Path, required=True)
    run.add_argument("--muq-eval-root", type=Path, required=True)
    run.add_argument("--muq-config", type=Path, required=True)
    run.add_argument("--muq-state-dict", type=Path, required=True)
    run.add_argument("--muq-backbone", type=Path, required=True)
    run.add_argument("--audiobox-checkpoint", type=Path, required=True)
    run.add_argument(
        "--environment-report",
        type=Path,
        default=SCRIPTS_ROOT.parent / "docs" / ENVIRONMENT_REPORT_BASENAME,
        help="sealed acceptance report for the frozen eval-quality environment",
    )
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--batch-size", type=int, default=8)

    verify = subparsers.add_parser("verify", help="rehash a sealed quality artifact")
    verify.add_argument("--generation-dir", type=Path, required=True)
    verify.add_argument("--quality-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return run_quality(args)
    provenance, rows, identity = common.verify_quality_artifact(
        Path(args.quality_dir), Path(args.generation_dir)
    )
    print(
        json.dumps(
            {
                "event": "cfg_quality_artifact_verified",
                "score_records": len(rows),
                "model_id": identity["generation_identity"]["model_id"],
                "quality_provenance_sha256": common.sha256_file(
                    Path(args.quality_dir) / "quality_provenance.json"
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
