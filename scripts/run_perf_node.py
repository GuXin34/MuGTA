#!/usr/bin/env python3
"""Run one node's frozen four-arm Williams performance sequence.

Each child is an independent ``torchrun --standalone --nproc_per_node=8`` job.
There is never a cross-node process group.  Run this same orchestrator on
node-0 through node-3 with the matching ``--node-label``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from perf_benchmark_common import (  # noqa: E402
    ARM_DEFINITIONS,
    ARM_STATUS_SCHEMA_VERSION,
    BENCHMARK_SCHEMA_VERSION,
    BLOCK_STEPS,
    MEASURED_STEPS,
    NODE_STATUS_SCHEMA_VERSION,
    TOTAL_STEPS,
    TRAIN_STAGE1_SHA256,
    WARMUP_STEPS,
    arm_definition,
    arm_seal_paths,
    node_seal_paths,
    schedule_for_node,
    sha256_file,
    verify_arm_directory,
    verify_node_directory,
    write_json_exclusive,
    write_seal,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-label", required=True, choices=("node-0", "node-1", "node-2", "node-3"))
    parser.add_argument("--torchrun", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--audiocraft-root", type=Path, required=True)
    parser.add_argument("--cfg-scale-decision-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--worker",
        type=Path,
        default=SCRIPT_DIR / "benchmark_stage1_perf.py",
    )
    return parser.parse_args(argv)


def source_hashes() -> Dict[str, str]:
    root = SCRIPT_DIR.parent
    paths = {
        "train_stage1.py": root / "scripts" / "train_stage1.py",
        "losses.py": root / "src" / "ptc_opd" / "losses.py",
        "distributed.py": root / "src" / "ptc_opd" / "distributed.py",
        "train_utils.py": root / "src" / "ptc_opd" / "train_utils.py",
    }
    observed = {name: sha256_file(path) for name, path in paths.items()}
    if observed["train_stage1.py"] != TRAIN_STAGE1_SHA256:
        raise RuntimeError("frozen train_stage1.py SHA-256 differs")
    return observed


def isolated_environment() -> Dict[str, str]:
    environment = dict(os.environ)
    polluted = {
        "MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE",
        "GROUP_RANK", "ROLE_RANK", "LOCAL_WORLD_SIZE", "ROLE_WORLD_SIZE",
        "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT",
        "SLURM_JOB_ID", "SLURM_PROCID", "SLURM_LOCALID", "SLURM_NTASKS",
        "OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE", "PMI_RANK", "PMI_SIZE",
        "PTC_NODE3_GATE", "CUBLAS_WORKSPACE_CONFIG",
    }
    for name in polluted:
        environment.pop(name, None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "OMP_NUM_THREADS": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def gpu_inventory() -> Dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,memory.total,compute_cap,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process.returncode != 0:
        raise RuntimeError("nvidia-smi inventory failed: {}".format(process.stderr.strip()))
    rows = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    if len(rows) != 8:
        raise RuntimeError("benchmark requires exactly eight visible nvidia-smi rows")
    return {"command": command, "rows": rows}


def worker_command(args: argparse.Namespace, arm_dir: Path, arm_name: str) -> List[str]:
    return [
        str(args.torchrun.resolve()),
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=8",
        str(args.worker.resolve()),
        "--arm", arm_name,
        "--manifest", str(args.manifest.resolve()),
        "--student-checkpoint", str(args.student_checkpoint.resolve()),
        "--teacher-checkpoint", str(args.teacher_checkpoint.resolve()),
        "--audiocraft-root", str(args.audiocraft_root.resolve()),
        "--cfg-scale-decision-dir", str(args.cfg_scale_decision_dir.resolve()),
        "--evidence-dir", str(arm_dir.resolve()),
        "--formal-run-dir", str((arm_dir / "formal_run").resolve()),
    ]


def arm_manifest(
    *,
    args: argparse.Namespace,
    arm_id: str,
    period: int,
    command: Sequence[str],
    hashes: Mapping[str, str],
) -> Dict[str, Any]:
    definition = arm_definition(arm_id)
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_only": True,
        "scientific_use_forbidden": True,
        "formal_stage1_consumer_must_reject": True,
        "node_label": args.node_label,
        "period": period,
        "arm_id": definition["arm_id"],
        "arm": definition["name"],
        "find_unused_parameters": definition["find_unused_parameters"],
        "audit_mode": definition["audit_mode"],
        "model": "musicgen-small",
        "mode": "uniform100",
        "seed": 2027,
        "learning_rate": 3.0e-6,
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
        "timing_block_steps": BLOCK_STEPS,
        "max_optimizer_steps": TOTAL_STEPS,
        "world_size": 8,
        "rank_batch_size": 2,
        "grad_accum_steps": 4,
        "effective_global_batch": 64,
        "check_finite": True,
        "source_hashes": dict(hashes),
        "inputs": {
            "manifest": str(args.manifest.resolve()),
            "student_checkpoint": str(args.student_checkpoint.resolve()),
            "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
            "audiocraft_root": str(args.audiocraft_root.resolve()),
            "cfg_scale_decision_dir": str(args.cfg_scale_decision_dir.resolve()),
        },
        "launch_command": list(command),
    }


def run_arm(
    *,
    args: argparse.Namespace,
    arm_id: str,
    period: int,
    hashes: Mapping[str, str],
    environment: Mapping[str, str],
) -> Dict[str, Any]:
    definition = arm_definition(arm_id)
    arm_dir = args.output_dir.resolve() / "period-{:02d}.{}".format(period, definition["name"])
    arm_dir.mkdir(parents=False, exist_ok=False)
    command = worker_command(args, arm_dir, definition["name"])
    write_json_exclusive(
        arm_dir / "benchmark_manifest.json",
        arm_manifest(
            args=args,
            arm_id=arm_id,
            period=period,
            command=command,
            hashes=hashes,
        ),
    )
    console_path = arm_dir / "console.log"
    started_utc = utc_now()
    started = time.perf_counter()
    with console_path.open("xb") as console:
        process = subprocess.run(
            command,
            stdout=console,
            stderr=subprocess.STDOUT,
            env=dict(environment),
            cwd=str(SCRIPT_DIR.parent),
        )
        console.flush()
        os.fsync(console.fileno())
    wall_seconds = time.perf_counter() - started
    status_payload = {
        "schema_version": ARM_STATUS_SCHEMA_VERSION,
        "benchmark_only": True,
        "scientific_use_forbidden": True,
        "status": "passed" if process.returncode == 0 else "failed",
        "redline_touched": False,
        "arm": definition["name"],
        "arm_id": definition["arm_id"],
        "node_label": args.node_label,
        "period": period,
        "started_utc": started_utc,
        "ended_utc": utc_now(),
        "returncode": process.returncode,
        "wall_seconds": wall_seconds,
    }
    write_json_exclusive(arm_dir / "STATUS.json", status_payload)
    if process.returncode != 0:
        raise RuntimeError(
            "{} failed with rc {}; evidence preserved at {}".format(
                definition["name"], process.returncode, arm_dir
            )
        )
    write_seal(arm_dir, scope="arm", relative_paths=arm_seal_paths())
    verification = verify_arm_directory(arm_dir)
    return {
        "arm": definition["name"],
        "period": period,
        "directory": str(arm_dir),
        "artifact_seal_sha256": verification["artifact_seal_sha256"],
        "summary": verification["summary"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError("refusing to overwrite {}".format(args.output_dir))
    if not args.torchrun.is_file() or not os.access(args.torchrun, os.X_OK):
        raise ValueError("--torchrun must be an executable regular file")
    if not args.worker.is_file():
        raise ValueError("benchmark worker is missing")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    environment = isolated_environment()
    sequence = schedule_for_node(args.node_label)
    node_config = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_only": True,
        "scientific_use_forbidden": True,
        "node_label": args.node_label,
        "physical_hostname": socket.gethostname(),
        "arm_sequence": list(sequence),
        "arm_names": [ARM_DEFINITIONS[item]["name"] for item in sequence],
        "source_hashes": hashes,
        "environment_overrides": {
            name: environment[name]
            for name in (
                "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "PYTHONUNBUFFERED"
            )
        },
        "pollution_variables_unset": [
            "MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE",
            "GROUP_RANK", "LOCAL_WORLD_SIZE", "SLURM_JOB_ID", "PTC_NODE3_GATE",
            "CUBLAS_WORKSPACE_CONFIG",
        ],
        "gpu_inventory_before": gpu_inventory(),
        "created_utc": utc_now(),
    }
    write_json_exclusive(args.output_dir / "NODE_CONFIG.json", node_config)
    arm_results = []
    try:
        for period, arm_id in enumerate(sequence, start=1):
            arm_results.append(
                run_arm(
                    args=args,
                    arm_id=arm_id,
                    period=period,
                    hashes=hashes,
                    environment=environment,
                )
            )
    except BaseException as exc:
        write_json_exclusive(
            args.output_dir / "STATUS.json",
            {
                "schema_version": NODE_STATUS_SCHEMA_VERSION,
                "benchmark_only": True,
                "status": "failed",
                "redline_touched": False,
                "node_label": args.node_label,
                "completed_arms": [item["arm"] for item in arm_results],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "required_action": "preserve_and_diagnose_operationally",
            },
        )
        raise
    write_json_exclusive(
        args.output_dir / "STATUS.json",
        {
            "schema_version": NODE_STATUS_SCHEMA_VERSION,
            "benchmark_only": True,
            "scientific_use_forbidden": True,
            "status": "passed",
            "redline_touched": False,
            "node_label": args.node_label,
            "arm_sequence": list(sequence),
            "completed_arms": [item["arm"] for item in arm_results],
            "arm_artifact_seals": {
                item["arm"]: item["artifact_seal_sha256"] for item in arm_results
            },
            "gpu_inventory_after": gpu_inventory(),
            "ended_utc": utc_now(),
            "required_action": "aggregate_after_all_four_nodes_pass",
        },
    )
    write_seal(
        args.output_dir,
        scope="node",
        relative_paths=node_seal_paths(args.output_dir),
    )
    result = verify_node_directory(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
