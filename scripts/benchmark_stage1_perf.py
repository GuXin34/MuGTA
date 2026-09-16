#!/usr/bin/env python3
"""Benchmark-only adapter around the frozen Stage-1 training runner.

The adapter imports ``train_stage1.py`` by exact SHA-256, changes only the DDP
policy/observational instrumentation requested by one of four frozen benchmark
arms, and measures complete five-step blocks.  It deliberately changes every
run/checkpoint/terminal schema to a benchmark-only namespace.  Consequently a
fresh invocation of the formal ``verify_stage1_run.py`` must reject these
outputs before considering any checkpoint payload.

This executable is a torchrun worker.  Use ``run_perf_node.py`` for the frozen
four-arm Williams sequence; do not launch this file as a scientific run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import time
import traceback
from typing import Any, Dict, List, Mapping, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from perf_benchmark_common import (  # noqa: E402
    BENCHMARK_SCHEMA_VERSION,
    BLOCK_STEPS,
    COMPONENT_NAMES,
    EXPECTED_BLOCKS,
    MEASURED_STEPS,
    PERF_ATTEMPT_SCHEMA_VERSION,
    PERF_CHECKPOINT_SCHEMA_VERSION,
    PERF_DONE_SCHEMA_VERSION,
    PERF_REDUCER_SCHEMA_VERSION,
    PERF_RUN_SCHEMA_VERSION,
    PERF_SEAL_SCHEMA_VERSION,
    RANK_SCHEMA_VERSION,
    TOTAL_STEPS,
    TRAIN_STAGE1_SHA256,
    WARMUP_STEPS,
    WORLD_SIZE,
    arm_definition,
    canonical_json_sha256,
    read_json,
    sha256_file,
    write_json_exclusive,
)


class BenchmarkRuntimeError(RuntimeError):
    pass


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=(
        "true_full", "false_full", "true_min", "false_min"
    ))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--audiocraft-root", type=Path, required=True)
    parser.add_argument("--cfg-scale-decision-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--formal-run-dir", type=Path, required=True)
    return parser.parse_args(argv)


def source_paths() -> Dict[str, Path]:
    root = SCRIPT_DIR.parent
    return {
        "train_stage1.py": root / "scripts" / "train_stage1.py",
        "losses.py": root / "src" / "ptc_opd" / "losses.py",
        "distributed.py": root / "src" / "ptc_opd" / "distributed.py",
        "train_utils.py": root / "src" / "ptc_opd" / "train_utils.py",
    }


def verified_source_hashes() -> Dict[str, str]:
    observed = {name: sha256_file(path) for name, path in source_paths().items()}
    if observed["train_stage1.py"] != TRAIN_STAGE1_SHA256:
        raise BenchmarkRuntimeError(
            "frozen train_stage1.py changed: expected {}, observed {}".format(
                TRAIN_STAGE1_SHA256, observed["train_stage1.py"]
            )
        )
    return observed


def load_train_module() -> Any:
    path = source_paths()["train_stage1.py"]
    spec = importlib.util.spec_from_file_location("_ptc_opd_perf_source_train", path)
    if spec is None or spec.loader is None:
        raise BenchmarkRuntimeError("cannot create train_stage1 import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TimingState:
    def __init__(self, *, arm: Mapping[str, Any], rank: int) -> None:
        self.arm = dict(arm)
        self.rank = rank
        self.optimizer_ready = False
        self.completed_steps = 0
        self.measurement_active = False
        self.block_started_at: Optional[float] = None
        self.block_start_step: Optional[int] = None
        self.blocks: List[Dict[str, Any]] = []
        self.components = {name: 0.0 for name in COMPONENT_NAMES}
        self.reducer_module: Any = None
        self.reducer_contract: Optional[Dict[str, Any]] = None
        self.reducer_lifecycle: List[Dict[str, Any]] = []
        self.phase_events: List[Dict[str, Any]] = []

    def component(self, name: str, seconds: float) -> None:
        if self.measurement_active:
            self.components[name] += float(seconds)


def _raw_reducer_snapshot(train: Any, module: Any, arm: Mapping[str, Any]) -> Dict[str, Any]:
    logging_data = module._get_ddp_logging_data()
    if not isinstance(logging_data, dict):
        raise BenchmarkRuntimeError("DDP logging data is not a dictionary")
    bucket_sizes = logging_data.get("bucket_sizes")
    if not isinstance(bucket_sizes, str) or re.fullmatch(
        r"[1-9][0-9]*(?:,\s*[1-9][0-9]*)*", bucket_sizes
    ) is None:
        raise BenchmarkRuntimeError("DDP bucket sizes are not canonical")
    named_trainable = [
        (name, parameter)
        for name, parameter in module.module.named_parameters()
        if parameter.requires_grad
    ]
    if not named_trainable:
        raise BenchmarkRuntimeError("DDP benchmark module has no trainable parameters")
    parameter_layout = [
        {
            "name": name,
            "dtype": str(parameter.dtype),
            "shape": list(parameter.shape),
            "numel": int(parameter.numel()),
            "element_size": int(parameter.element_size()),
        }
        for name, parameter in named_trainable
    ]
    parsed_bucket_sizes = [int(item) for item in re.split(r",\s*", bucket_sizes)]
    trainable_gradient_bytes = sum(
        int(parameter.numel()) * int(parameter.element_size())
        for _, parameter in named_trainable
    )
    if sum(parsed_bucket_sizes) != trainable_gradient_bytes:
        raise BenchmarkRuntimeError("DDP buckets do not cover trainable gradient bytes")
    observed = {
        "schema_version": PERF_REDUCER_SCHEMA_VERSION,
        "policy_id": "benchmark-find-unused-{}-v1".format(
            str(bool(arm["find_unused_parameters"])).lower()
        ),
        "torch_version": str(train.torch.__version__),
        "torch_cuda_runtime": str(train.torch.version.cuda),
        "find_unused_parameters": bool(module.find_unused_parameters),
        "static_graph": bool(module.static_graph),
        "gradient_as_bucket_view": bool(module.gradient_as_bucket_view),
        "bucket_cap_bytes": int(module.bucket_bytes_cap),
        "has_rebuilt_buckets": bool(module._has_rebuilt_buckets),
        "bucket_sizes": bucket_sizes,
        "bucket_count": len(parsed_bucket_sizes),
        "trainable_parameter_tensor_count": len(named_trainable),
        "trainable_parameter_numel": sum(int(parameter.numel()) for _, parameter in named_trainable),
        "trainable_gradient_bytes": trainable_gradient_bytes,
        "parameter_layout_sha256": canonical_json_sha256(parameter_layout),
    }
    if observed["find_unused_parameters"] is not bool(arm["find_unused_parameters"]):
        raise BenchmarkRuntimeError("DDP find_unused_parameters differs from arm")
    if observed["static_graph"] is not False or observed["gradient_as_bucket_view"] is not False:
        raise BenchmarkRuntimeError("DDP benchmark changed static/bucket-view policy")
    if observed["bucket_cap_bytes"] != 25 * 1024 * 1024:
        raise BenchmarkRuntimeError("DDP benchmark bucket cap differs")
    observed["identity_sha256"] = canonical_json_sha256(observed)
    return observed


def install_benchmark_adapters(train: Any, train_utils: Any, state: TimingState) -> None:
    arm = state.arm
    train.RUN_SCHEMA_VERSION = PERF_RUN_SCHEMA_VERSION
    train.CHECKPOINT_SCHEMA_VERSION = PERF_CHECKPOINT_SCHEMA_VERSION
    train.ATTEMPT_SCHEMA_VERSION = PERF_ATTEMPT_SCHEMA_VERSION
    train.SEAL_SCHEMA_VERSION = PERF_SEAL_SCHEMA_VERSION
    train.DONE_SCHEMA_VERSION = PERF_DONE_SCHEMA_VERSION
    train.DDP_REDUCER_SCHEMA_VERSION = PERF_REDUCER_SCHEMA_VERSION
    train.DDP_REDUCER_POLICY_ID = "benchmark-only-ddp-policy-v1"
    train.DDP_FIND_UNUSED_PARAMETERS = bool(arm["find_unused_parameters"])
    train_utils.RUN_SCHEMA_VERSION = PERF_RUN_SCHEMA_VERSION
    train_utils.CHECKPOINT_SCHEMA_VERSION = PERF_CHECKPOINT_SCHEMA_VERSION

    original_ddp_audit = train.ddp_reducer_audit

    def benchmark_ddp_audit(module: Any) -> Dict[str, Any]:
        started = time.perf_counter()
        observed = _raw_reducer_snapshot(train, module, arm)
        elapsed = time.perf_counter() - started
        state.component("reducer_audit_seconds", elapsed)
        state.reducer_module = module
        state.reducer_lifecycle.append(
            {
                "source": "runner_audit",
                "completed_step": state.completed_steps,
                **observed,
            }
        )
        if state.reducer_contract is None:
            if observed["has_rebuilt_buckets"] is not False:
                raise BenchmarkRuntimeError("DDP was rebuilt at construction")
            state.reducer_contract = dict(observed)
        if arm["find_unused_parameters"]:
            if observed["has_rebuilt_buckets"] is not False:
                raise BenchmarkRuntimeError("production True policy rebuilt buckets")
            if observed["identity_sha256"] != state.reducer_contract["identity_sha256"]:
                raise BenchmarkRuntimeError("production True reducer identity drifted")
            return observed
        # The frozen source's terminal chain requires one construction identity.
        # Return that benchmark-only identity while preserving the actual False
        # lifecycle above in rank evidence.  No formal consumer accepts this run.
        return dict(state.reducer_contract)

    train.ddp_reducer_audit = benchmark_ddp_audit

    original_codes_hash = train.tensor_sha256
    original_gate_hash = train.boolean_tensor_sha256

    if arm["audit_mode"] == "full":
        def timed_codes_hash(value: Any) -> str:
            started = time.perf_counter()
            result = original_codes_hash(value)
            state.component("rollout_codes_sha_seconds", time.perf_counter() - started)
            return result

        def timed_gate_hash(value: Any) -> str:
            started = time.perf_counter()
            result = original_gate_hash(value)
            state.component("selected_gate_sha_seconds", time.perf_counter() - started)
            return result

        train.tensor_sha256 = timed_codes_hash
        train.boolean_tensor_sha256 = timed_gate_hash
    else:
        train.tensor_sha256 = lambda _value: "0" * 64
        train.boolean_tensor_sha256 = lambda _value: "0" * 64

    original_all_gather_object = train.dist.all_gather_object

    def timed_all_gather_object(output: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
        is_step_audit = isinstance(value, dict) and "microsteps" in value
        started = time.perf_counter()
        result = original_all_gather_object(output, value, *args, **kwargs)
        if is_step_audit:
            state.component("audit_all_gather_seconds", time.perf_counter() - started)
        return result

    train.dist.all_gather_object = timed_all_gather_object

    original_jsonl_append = train.jsonl_append

    def timed_jsonl_append(path: Path, value: Mapping[str, Any]) -> None:
        started = time.perf_counter()
        original_jsonl_append(path, value)
        state.component("json_encode_write_fsync_seconds", time.perf_counter() - started)

    train.jsonl_append = timed_jsonl_append

    original_print_json = train.print_json

    def timed_print_json(value: Mapping[str, Any]) -> None:
        started = time.perf_counter()
        original_print_json(value)
        state.component("stdout_flush_seconds", time.perf_counter() - started)

    train.print_json = timed_print_json

    original_write_json = train.write_json_exclusive

    def benchmark_write_json(path: Path, value: Mapping[str, Any]) -> None:
        payload = dict(value)
        if path.name == "run_manifest.json":
            payload.update(
                {
                    "benchmark_only": True,
                    "scientific_use_forbidden": True,
                    "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
                    "benchmark_arm": arm["name"],
                    "formal_stage1_consumer_must_reject": True,
                }
            )
        original_write_json(path, payload)

    train.write_json_exclusive = benchmark_write_json

    original_adamw = train.torch.optim.AdamW

    class TimedAdamW(original_adamw):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            state.optimizer_ready = True

        def step(self, *args: Any, **kwargs: Any) -> Any:
            result = super().step(*args, **kwargs)
            state.completed_steps += 1
            if state.completed_steps > TOTAL_STEPS:
                raise BenchmarkRuntimeError("optimizer exceeded 40 benchmark updates")
            return result

    train.torch.optim.AdamW = TimedAdamW

    def record_boundary(completed_step: int, *, final: bool = False) -> None:
        if state.reducer_module is None:
            raise BenchmarkRuntimeError("timing boundary has no DDP reducer")
        train.torch.cuda.synchronize()
        now = time.perf_counter()
        if state.block_started_at is not None:
            if state.block_start_step is None:
                raise BenchmarkRuntimeError("timing state has no block start step")
            state.blocks.append(
                {
                    "block_index": len(state.blocks),
                    "start_completed_step": state.block_start_step,
                    "end_completed_step": completed_step,
                    "steps": completed_step - state.block_start_step,
                    "seconds": now - state.block_started_at,
                }
            )
        lifecycle = _raw_reducer_snapshot(train, state.reducer_module, arm)
        state.reducer_lifecycle.append(
            {
                "source": "timing_boundary",
                "completed_step": completed_step,
                **lifecycle,
            }
        )
        if arm["find_unused_parameters"] and lifecycle["has_rebuilt_buckets"] is not False:
            raise BenchmarkRuntimeError("production True reducer rebuilt at boundary")
        train.dist.barrier()
        if final:
            state.measurement_active = False
            state.block_started_at = None
            state.block_start_step = None
        else:
            state.measurement_active = True
            state.block_started_at = time.perf_counter()
            state.block_start_step = completed_step

    original_learning_rate = train.learning_rate_for_update

    def timed_learning_rate(base_lr: float, completed_updates: int, warmup_updates: int) -> float:
        if state.optimizer_ready:
            if completed_updates != state.completed_steps:
                raise BenchmarkRuntimeError("optimizer/LR progress drift in benchmark")
            if completed_updates in {10, 15, 20, 25, 30, 35}:
                record_boundary(completed_updates)
        return original_learning_rate(base_lr, completed_updates, warmup_updates)

    train.learning_rate_for_update = timed_learning_rate

    original_hash_module_state = train.hash_module_state

    def timed_hash_module_state(module: Any) -> str:
        if (
            state.completed_steps == TOTAL_STEPS
            and state.measurement_active
            and len(state.blocks) == EXPECTED_BLOCKS - 1
        ):
            record_boundary(TOTAL_STEPS, final=True)
        started = time.perf_counter()
        result = original_hash_module_state(module)
        state.phase_events.append(
            {
                "event": "hash_module_state",
                "completed_step": state.completed_steps,
                "rank": state.rank,
                "seconds": time.perf_counter() - started,
            }
        )
        return result

    train.hash_module_state = timed_hash_module_state

    original_save_checkpoint = train.save_checkpoint

    def timed_save_checkpoint(*args: Any, **kwargs: Any) -> None:
        started = time.perf_counter()
        original_save_checkpoint(*args, **kwargs)
        state.phase_events.append(
            {
                "event": "save_checkpoint",
                "completed_step": state.completed_steps,
                "rank": state.rank,
                "seconds": time.perf_counter() - started,
            }
        )

    train.save_checkpoint = timed_save_checkpoint

    original_commit_terminal = train.commit_success_terminal

    def timed_commit_terminal(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        result = original_commit_terminal(*args, **kwargs)
        state.phase_events.append(
            {
                "event": "commit_success_terminal",
                "completed_step": state.completed_steps,
                "rank": state.rank,
                "seconds": time.perf_counter() - started,
            }
        )
        return result

    train.commit_success_terminal = timed_commit_terminal

    # Retain a reference so a debugger can prove that the production function
    # was replaced deliberately, not lost through an import accident.
    train._benchmark_original_ddp_reducer_audit = original_ddp_audit


def train_argv(args: argparse.Namespace, arm: Mapping[str, Any]) -> List[str]:
    log_every = 1 if arm["audit_mode"] == "full" else TOTAL_STEPS + 1
    return [
        "--manifest", str(args.manifest.resolve()),
        "--student-checkpoint", str(args.student_checkpoint.resolve()),
        "--teacher-checkpoint", str(args.teacher_checkpoint.resolve()),
        "--audiocraft-root", str(args.audiocraft_root.resolve()),
        "--cfg-scale-decision-dir", str(args.cfg_scale_decision_dir.resolve()),
        "--output-dir", str(args.formal_run_dir.resolve()),
        "--mode", "uniform100",
        "--seed", "2027",
        "--learning-rate", "3e-6",
        "--max-optimizer-steps", str(TOTAL_STEPS),
        "--save-every", "10000",
        "--log-every", str(log_every),
        "--rank-batch-size", "2",
        "--expected-world-size", "8",
        "--grad-accum-steps", "4",
        "--effective-global-batch", "64",
        "--duration-seconds", "10",
        "--codec-frame-rate", "50",
        "--token-frames", "500",
        "--teacher-forward-mode", "batched",
    ]


def validate_invocation(args: argparse.Namespace, hashes: Mapping[str, str]) -> Dict[str, Any]:
    if os.environ.get("PTC_NODE3_GATE") is not None:
        raise BenchmarkRuntimeError("PTC_NODE3_GATE must be unset for pilot-like timing")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is not None:
        raise BenchmarkRuntimeError("CUBLAS_WORKSPACE_CONFIG must be unset for pilot-like timing")
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        if name not in os.environ:
            raise BenchmarkRuntimeError("launch benchmark with torchrun; {} is absent".format(name))
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != WORLD_SIZE:
        raise BenchmarkRuntimeError("benchmark WORLD_SIZE must be exactly 8")
    evidence = args.evidence_dir.resolve()
    expected_formal = evidence / "formal_run"
    if args.formal_run_dir.resolve() != expected_formal:
        raise BenchmarkRuntimeError("formal run must be evidence-dir/formal_run")
    manifest = read_json(evidence / "benchmark_manifest.json")
    arm = arm_definition(args.arm)
    if (
        manifest.get("schema_version") != BENCHMARK_SCHEMA_VERSION
        or manifest.get("benchmark_only") is not True
        or manifest.get("arm") != arm["name"]
        or manifest.get("arm_id") != arm["arm_id"]
        or manifest.get("source_hashes") != dict(hashes)
    ):
        raise BenchmarkRuntimeError("orchestrator benchmark manifest differs")
    return {"rank": rank, "world_size": world_size, "manifest": manifest, "arm": arm}


def write_rank_payload(
    *,
    args: argparse.Namespace,
    state: TimingState,
    hashes: Mapping[str, str],
    train: Any,
) -> Path:
    if len(state.blocks) != EXPECTED_BLOCKS:
        raise BenchmarkRuntimeError(
            "expected {} timing blocks, observed {}".format(EXPECTED_BLOCKS, len(state.blocks))
        )
    if state.completed_steps != TOTAL_STEPS:
        raise BenchmarkRuntimeError("benchmark did not complete exactly 40 updates")
    device = train.torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    payload = {
        "schema_version": RANK_SCHEMA_VERSION,
        "benchmark_only": True,
        "scientific_use_forbidden": True,
        "status": "passed",
        "arm": state.arm["name"],
        "arm_id": state.arm["arm_id"],
        "rank": state.rank,
        "local_rank": int(os.environ["LOCAL_RANK"]),
        "world_size": int(os.environ["WORLD_SIZE"]),
        "host": socket.gethostname(),
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
        "timing_block_steps": BLOCK_STEPS,
        "source_hashes": dict(hashes),
        "blocks": state.blocks,
        "components_seconds": (
            state.components
            if state.arm["audit_mode"] == "full"
            else {name: 0.0 for name in COMPONENT_NAMES}
        ),
        "reducer_lifecycle": state.reducer_lifecycle,
        "phase_events": state.phase_events,
        "cuda_max_memory_allocated": int(train.torch.cuda.max_memory_allocated(device)),
        "cuda_max_memory_reserved": int(train.torch.cuda.max_memory_reserved(device)),
        "cuda_total_memory_bytes": int(train.torch.cuda.get_device_properties(device).total_memory),
        "runtime": {
            "python": sys.version,
            "torch": str(train.torch.__version__),
            "cuda_runtime": str(train.torch.version.cuda),
            "gpu_name": train.torch.cuda.get_device_name(device),
            "gpu_capability": list(train.torch.cuda.get_device_capability(device)),
            "ptc_node3_gate": None,
            "cublas_workspace_config": None,
        },
    }
    ranks = args.evidence_dir.resolve() / "ranks"
    ranks.mkdir(parents=True, exist_ok=True)
    path = ranks / "rank-{:02d}.json".format(state.rank)
    write_json_exclusive(path, payload)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    hashes = verified_source_hashes()
    invocation = validate_invocation(args, hashes)
    train = load_train_module()
    import ptc_opd.train_utils as train_utils

    state = TimingState(arm=invocation["arm"], rank=invocation["rank"])
    install_benchmark_adapters(train, train_utils, state)
    parsed_train_args = train.parse_args(train_argv(args, invocation["arm"]))
    config = train.make_config(parsed_train_args)
    try:
        train.run_training(config, parsed_train_args)
        write_rank_payload(args=args, state=state, hashes=hashes, train=train)
    except BaseException as exc:
        ranks = args.evidence_dir.resolve() / "ranks"
        ranks.mkdir(parents=True, exist_ok=True)
        error_path = ranks / "rank-{:02d}.ERROR.json".format(invocation["rank"])
        if not error_path.exists():
            write_json_exclusive(
                error_path,
                {
                    "schema_version": RANK_SCHEMA_VERSION,
                    "benchmark_only": True,
                    "status": "failed",
                    "rank": invocation["rank"],
                    "arm": invocation["arm"]["name"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
