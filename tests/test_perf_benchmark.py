"""CPU-only tests for the benchmark-only Stage-1 performance contract."""

from __future__ import annotations

import json
import contextlib
import io
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from argparse import Namespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from perf_benchmark_common import (  # noqa: E402
    ARM_DEFINITIONS,
    ARM_STATUS_SCHEMA_VERSION,
    BENCHMARK_SCHEMA_VERSION,
    BLOCK_STEPS,
    COMPONENT_NAMES,
    FORMAL_RUN_SCHEMA_VERSION,
    MEASURED_STEPS,
    NODE_STATUS_SCHEMA_VERSION,
    PERF_DONE_SCHEMA_VERSION,
    PERF_RUN_SCHEMA_VERSION,
    PERF_SEAL_SCHEMA_VERSION,
    RANK_SCHEMA_VERSION,
    TOTAL_STEPS,
    TRAIN_STAGE1_SHA256,
    WARMUP_STEPS,
    WILLIAMS_SCHEDULE,
    PerfContractError,
    aggregate_node_directories,
    arm_seal_paths,
    audiocraft_source_identity,
    build_live_input_identity,
    canonical_json_sha256,
    node_seal_paths,
    sha256_file,
    sha256_path,
    validate_williams_schedule,
    verify_arm_directory,
    verify_node_directory,
    verify_node_input_bindings,
    write_json_exclusive,
    write_seal,
    write_summary_csv,
)
from benchmark_stage1_perf import train_argv, verified_source_hashes  # noqa: E402
from verify_perf_benchmark import (  # noqa: E402
    _canonical_artifact_root,
    main as verify_perf_main,
    parse_args as parse_verifier_args,
    verify_aggregate,
)


SECONDS_PER_STEP = {
    "true_full": 2.00,
    "false_full": 1.80,
    "true_min": 1.60,
    "false_min": 1.50,
}


def _rank_payload(arm_id: str, rank: int) -> dict:
    definition = ARM_DEFINITIONS[arm_id]
    seconds_per_step = SECONDS_PER_STEP[definition["name"]]
    blocks = []
    for block_index in range(MEASURED_STEPS // BLOCK_STEPS):
        start = WARMUP_STEPS + block_index * BLOCK_STEPS
        # Rank 7 is the deterministic straggler and therefore defines the
        # global complete-step block used by the aggregator.
        rank_factor = 1.0 + rank * 0.001
        blocks.append(
            {
                "block_index": block_index,
                "start_completed_step": start,
                "end_completed_step": start + BLOCK_STEPS,
                "steps": BLOCK_STEPS,
                "seconds": seconds_per_step * BLOCK_STEPS * rank_factor,
            }
        )
    components = {
        name: (0.01 if definition["audit_mode"] == "full" else 0.0)
        for name in COMPONENT_NAMES
    }
    lifecycle = [
        {
            "completed_step": 0,
            "has_rebuilt_buckets": False,
            "find_unused_parameters": definition["find_unused_parameters"],
        },
        {
            "completed_step": 10,
            "has_rebuilt_buckets": (
                False if definition["find_unused_parameters"] else True
            ),
            "find_unused_parameters": definition["find_unused_parameters"],
        },
        {
            "completed_step": 40,
            "has_rebuilt_buckets": (
                False if definition["find_unused_parameters"] else True
            ),
            "find_unused_parameters": definition["find_unused_parameters"],
        },
    ]
    return {
        "schema_version": RANK_SCHEMA_VERSION,
        "benchmark_only": True,
        "scientific_use_forbidden": True,
        "status": "passed",
        "arm": definition["name"],
        "arm_id": arm_id,
        "rank": rank,
        "local_rank": rank,
        "world_size": 8,
        "host": "fake-node",
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
        "timing_block_steps": BLOCK_STEPS,
        "source_hashes": {"train_stage1.py": TRAIN_STAGE1_SHA256},
        "blocks": blocks,
        "components_seconds": components,
        "reducer_lifecycle": lifecycle,
        "phase_events": [
            {"event": "save_checkpoint", "completed_step": 0, "seconds": 1.0},
            {"event": "save_checkpoint", "completed_step": 40, "seconds": 1.0},
        ],
        "cuda_max_memory_allocated": 700,
        "cuda_max_memory_reserved": 800,
        "cuda_total_memory_bytes": 1000,
    }


def _make_live_inputs(root: Path, name: str = "live") -> dict:
    input_root = root / name
    input_root.mkdir()
    train_manifest = input_root / "train.full.jsonl"
    train_manifest.write_text(
        json.dumps({"sample_id": name, "prompt": "frozen prompt"}) + "\n",
        encoding="utf-8",
    )
    checkpoint = input_root / "musicgen-small"
    checkpoint.mkdir()
    (checkpoint / "state_dict.bin").write_bytes(b"student-state-" + name.encode())
    (checkpoint / "compression_state_dict.bin").write_bytes(
        b"codec-state-" + name.encode()
    )
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    source = input_root / "audiocraft"
    (source / "audiocraft" / "models").mkdir(parents=True)
    (source / "audiocraft" / "models" / "lm.py").write_text(
        "# frozen {}\n".format(name), encoding="utf-8"
    )
    source_identity = audiocraft_source_identity(source)
    checkpoint_identity = {
        "checkpoint_sha256": sha256_path(checkpoint),
        "state_dict_sha256": sha256_file(checkpoint / "state_dict.bin"),
        "compression_state_dict_sha256": sha256_file(
            checkpoint / "compression_state_dict.bin"
        ),
    }
    cfg = input_root / "small-cfg"
    cfg.mkdir()
    generation_identity = {
        "model_id": "facebook/musicgen-small",
        **checkpoint_identity,
        "audiocraft_source_sha256": source_identity["tree_sha256"],
        "audiocraft_lm_sha256": sha256_file(
            source / "audiocraft" / "models" / "lm.py"
        ),
        "loaded_t5_identity_sha256": canonical_json_sha256(
            {"fixture": name, "asset": "t5"}
        ),
    }
    decision = {
        "schema_version": "ptc-opd-cfg-scale-decision-v3",
        "selected_cfg_scale": 5.0,
        "decision_payload_sha256": canonical_json_sha256(
            {"fixture": name, "asset": "decision"}
        ),
        "scientific_config_sha256": canonical_json_sha256(
            {"fixture": name, "asset": "science"}
        ),
        "generation_identity": generation_identity,
    }
    write_json_exclusive(cfg / "cfg_scale_decision.json", decision)
    write_json_exclusive(
        cfg / "cfg_scale_decision.sha256.json",
        {
            "sha256": sha256_file(cfg / "cfg_scale_decision.json"),
            "decision_payload_sha256": decision["decision_payload_sha256"],
        },
    )
    return {
        "train_manifest": train_manifest,
        "small_cfg_dir": cfg,
        "audiocraft_dir": source,
        "musicgen_small_dir": checkpoint,
    }


def _copy_live_inputs(paths: dict, root: Path, name: str = "alternate") -> dict:
    target = root / name
    target.mkdir()
    train = target / "train.full.jsonl"
    shutil.copy2(paths["train_manifest"], train)
    copied = {"train_manifest": train}
    for key, leaf in (
        ("small_cfg_dir", "small-cfg"),
        ("audiocraft_dir", "audiocraft"),
        ("musicgen_small_dir", "musicgen-small"),
    ):
        destination = target / leaf
        shutil.copytree(paths[key], destination)
        copied[key] = destination
    return copied


def _make_arm(
    root: Path,
    arm_id: str,
    period: int,
    node_label: str,
    live_input_identity: dict = None,
) -> Path:
    definition = ARM_DEFINITIONS[arm_id]
    arm_dir = root / "period-{:02d}.{}".format(period, definition["name"])
    (arm_dir / "ranks").mkdir(parents=True)
    (arm_dir / "formal_run").mkdir()
    benchmark_manifest = {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "benchmark_only": True,
            "scientific_use_forbidden": True,
            "arm_id": arm_id,
            "arm": definition["name"],
            "node_label": node_label,
            "period": period,
            "source_hashes": {"train_stage1.py": TRAIN_STAGE1_SHA256},
        }
    formal_manifest = {
        "schema_version": PERF_RUN_SCHEMA_VERSION,
        "benchmark_only": True,
        "scientific_use_forbidden": True,
    }
    if live_input_identity is not None:
        paths = live_input_identity["paths"]
        checkpoint = live_input_identity["checkpoint_identity"]
        source = live_input_identity["audiocraft_source_identity"]
        cfg = live_input_identity["cfg_decision_identity"]
        generation = cfg["generation_identity"]
        benchmark_manifest["inputs"] = {
            "manifest": paths["train_manifest"],
            "student_checkpoint": paths["musicgen_small_dir"],
            "teacher_checkpoint": paths["musicgen_small_dir"],
            "audiocraft_root": paths["audiocraft_dir"],
            "cfg_scale_decision_dir": paths["small_cfg_dir"],
        }
        formal_manifest.update(
            {
                "config": {
                    **benchmark_manifest["inputs"],
                    "cfg_scale_decision_file_sha256": cfg[
                        "decision_file_sha256"
                    ],
                    "cfg_scale_decision_payload_sha256": cfg[
                        "decision_payload_sha256"
                    ],
                    "cfg_scale_scientific_config_sha256": cfg[
                        "scientific_config_sha256"
                    ],
                    "cfg_generation_checkpoint_sha256": checkpoint[
                        "checkpoint_sha256"
                    ],
                    "cfg_generation_state_dict_sha256": checkpoint[
                        "state_dict_sha256"
                    ],
                    "cfg_generation_audiocraft_source_sha256": source[
                        "tree_sha256"
                    ],
                    "cfg_generation_loaded_t5_identity_sha256": generation[
                        "loaded_t5_identity_sha256"
                    ],
                    "teacher_cfg_scale": cfg["selected_cfg_scale"],
                },
                "manifest_sha256": live_input_identity["manifest_sha256"],
                "student_checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "teacher_checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "student_state_dict_sha256": checkpoint["state_dict_sha256"],
                "teacher_state_dict_sha256": checkpoint["state_dict_sha256"],
                "audiocraft_lm_sha256": live_input_identity[
                    "audiocraft_lm_sha256"
                ],
                "audiocraft_source_identity": source,
                "cfg_scale_decision": cfg,
                "cfg_generation_binding": {
                    "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                    "state_dict_sha256": checkpoint["state_dict_sha256"],
                    "audiocraft_source_sha256": source["tree_sha256"],
                },
                "loaded_t5_identity": {
                    "identity_sha256": generation["loaded_t5_identity_sha256"]
                },
            }
        )
    write_json_exclusive(arm_dir / "benchmark_manifest.json", benchmark_manifest)
    for rank in range(8):
        write_json_exclusive(
            arm_dir / "ranks" / "rank-{:02d}.json".format(rank),
            _rank_payload(arm_id, rank),
        )
    write_json_exclusive(
        arm_dir / "formal_run" / "run_manifest.json",
        formal_manifest,
    )
    write_json_exclusive(
        arm_dir / "formal_run" / "SEALED.json",
        {
            "schema_version": PERF_SEAL_SCHEMA_VERSION,
            "benchmark_only": True,
            "status": "sealed",
        },
    )
    write_json_exclusive(
        arm_dir / "formal_run" / "DONE.json",
        {
            "schema_version": PERF_DONE_SCHEMA_VERSION,
            "benchmark_only": True,
            "status": "complete",
            "SEALED.json_sha256": sha256_file(arm_dir / "formal_run" / "SEALED.json"),
            "run_manifest_sha256": sha256_file(arm_dir / "formal_run" / "run_manifest.json"),
        },
    )
    (arm_dir / "console.log").write_bytes(b"")
    write_json_exclusive(
        arm_dir / "STATUS.json",
        {
            "schema_version": ARM_STATUS_SCHEMA_VERSION,
            "benchmark_only": True,
            "scientific_use_forbidden": True,
            "status": "passed",
            "arm": definition["name"],
            "arm_id": arm_id,
            "returncode": 0,
        },
    )
    write_seal(arm_dir, scope="arm", relative_paths=arm_seal_paths())
    return arm_dir


def _make_node(
    parent: Path,
    node_label: str,
    live_input_identity: dict = None,
    arm_input_identities: dict = None,
) -> Path:
    node_dir = parent / node_label
    node_dir.mkdir()
    sequence = WILLIAMS_SCHEDULE[node_label]
    write_json_exclusive(
        node_dir / "NODE_CONFIG.json",
        {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "benchmark_only": True,
            "scientific_use_forbidden": True,
            "node_label": node_label,
            "arm_sequence": list(sequence),
        },
    )
    for period, arm_id in enumerate(sequence, start=1):
        arm_live = live_input_identity
        if arm_input_identities is not None and arm_id in arm_input_identities:
            arm_live = arm_input_identities[arm_id]
        _make_arm(
            node_dir,
            arm_id,
            period,
            node_label,
            live_input_identity=arm_live,
        )
    write_json_exclusive(
        node_dir / "STATUS.json",
        {
            "schema_version": NODE_STATUS_SCHEMA_VERSION,
            "benchmark_only": True,
            "scientific_use_forbidden": True,
            "status": "passed",
            "node_label": node_label,
        },
    )
    write_seal(node_dir, scope="node", relative_paths=node_seal_paths(node_dir))
    return node_dir


class PerfBenchmarkContractTest(unittest.TestCase):
    def test_worker_is_bound_to_frozen_train_source(self) -> None:
        self.assertEqual(
            verified_source_hashes()["train_stage1.py"], TRAIN_STAGE1_SHA256
        )

    def test_worker_argv_has_frozen_workload_and_no_resume(self) -> None:
        fake = Namespace(
            manifest=Path("/tmp/manifest.jsonl"),
            student_checkpoint=Path("/tmp/student"),
            teacher_checkpoint=Path("/tmp/teacher"),
            audiocraft_root=Path("/tmp/audiocraft"),
            cfg_scale_decision_dir=Path("/tmp/cfg"),
            formal_run_dir=Path("/tmp/formal_run"),
        )
        full = train_argv(fake, {"audit_mode": "full"})
        minimal = train_argv(fake, {"audit_mode": "min"})
        self.assertNotIn("--resume", full)
        self.assertEqual(full[full.index("--mode") + 1], "uniform100")
        self.assertEqual(full[full.index("--max-optimizer-steps") + 1], "40")
        self.assertEqual(full[full.index("--log-every") + 1], "1")
        self.assertEqual(minimal[minimal.index("--log-every") + 1], "41")

    def test_williams_square_balances_periods_and_carryover(self) -> None:
        validate_williams_schedule()
        arms = set(ARM_DEFINITIONS)
        for period in range(4):
            self.assertEqual(
                {sequence[period] for sequence in WILLIAMS_SCHEDULE.values()}, arms
            )
        pairs = {
            (sequence[index], sequence[index + 1])
            for sequence in WILLIAMS_SCHEDULE.values()
            for index in range(3)
        }
        self.assertEqual(pairs, {(a, b) for a in arms for b in arms if a != b})

    def test_benchmark_schema_cannot_equal_formal_stage1_schema(self) -> None:
        self.assertNotEqual(PERF_RUN_SCHEMA_VERSION, FORMAL_RUN_SCHEMA_VERSION)

    def test_arm_and_node_closed_world_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            node = _make_node(root, "node-0")
            self.assertEqual(verify_node_directory(node)["status"], "verified")
            arm = node / "period-01.true_full"
            result = verify_arm_directory(arm)
            self.assertTrue(result["formal_stage1_consumer_must_reject"])
            self.assertEqual(result["summary"]["arm"], "true_full")

    def test_node_verifier_binds_all_four_arms_to_live_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = _make_live_inputs(root)
            live = build_live_input_identity(**paths)
            node = _make_node(root, "node-0", live_input_identity=live)
            result = verify_node_input_bindings(node, **paths)
            self.assertEqual(result["verified_input_arm_count"], 4)
            self.assertEqual(set(result["arm_input_bindings"]), set(SECONDS_PER_STEP))
            self.assertEqual(result["live_input_identity"], live)
            output = io.StringIO()
            argv = [
                "--node-dir",
                str(node),
                "--train-manifest",
                str(paths["train_manifest"]),
                "--small-cfg-dir",
                str(paths["small_cfg_dir"]),
                "--audiocraft-dir",
                str(paths["audiocraft_dir"]),
                "--musicgen-small-dir",
                str(paths["musicgen_small_dir"]),
            ]
            with contextlib.redirect_stdout(output):
                self.assertEqual(verify_perf_main(argv), 0)
            cli_result = json.loads(output.getvalue())
            self.assertEqual(cli_result["verified_input_arm_count"], 4)
            self.assertEqual(cli_result["live_input_identity"], live)

    def test_node_cli_requires_complete_live_input_quartet(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_verifier_args(["--node-dir", "/tmp/node-0"])
        parsed = parse_verifier_args(
            [
                "--node-dir",
                "/tmp/node-0",
                "--train-manifest",
                "/tmp/train.full.jsonl",
                "--small-cfg-dir",
                "/tmp/cfg",
                "--audiocraft-dir",
                "/tmp/audiocraft",
                "--musicgen-small-dir",
                "/tmp/musicgen-small",
            ]
        )
        self.assertEqual(parsed.node_dir, Path("/tmp/node-0"))

    def test_valid_path_swaps_are_rejected_for_each_scientific_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = _make_live_inputs(root)
            alternate = _copy_live_inputs(paths, root)
            live = build_live_input_identity(**paths)
            node = _make_node(root, "node-0", live_input_identity=live)
            for swapped_name in sorted(paths):
                with self.subTest(swapped_name=swapped_name):
                    expected = dict(paths)
                    expected[swapped_name] = alternate[swapped_name]
                    with self.assertRaisesRegex(
                        PerfContractError, "path differs from the live input"
                    ):
                        verify_node_input_bindings(node, **expected)

    def test_alternate_valid_inputs_in_fourth_arm_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = _make_live_inputs(root)
            alternate = _copy_live_inputs(paths, root)
            live = build_live_input_identity(**paths)
            alternate_live = build_live_input_identity(**alternate)
            node = _make_node(
                root,
                "node-0",
                live_input_identity=live,
                arm_input_identities={"C": alternate_live},
            )
            with self.assertRaisesRegex(
                PerfContractError, "path differs from the live input"
            ):
                verify_node_input_bindings(node, **paths)

    def test_live_train_manifest_byte_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = _make_live_inputs(root)
            live = build_live_input_identity(**paths)
            node = _make_node(root, "node-0", live_input_identity=live)
            paths["train_manifest"].write_text(
                '{"sample_id":"changed","prompt":"different"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                PerfContractError, "formal input identity differs at manifest_sha256"
            ):
                verify_node_input_bindings(node, **paths)

    def test_rank_tamper_is_rejected_by_arm_seal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            node = _make_node(root, "node-0")
            arm = node / "period-01.true_full"
            rank_path = arm / "ranks" / "rank-00.json"
            payload = json.loads(rank_path.read_text(encoding="utf-8"))
            payload["blocks"][0]["seconds"] *= 0.5
            rank_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(PerfContractError, "sealed payload changed"):
                verify_arm_directory(arm)

    def test_four_node_aggregation_is_paired_and_never_switches_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nodes = [_make_node(root, label) for label in sorted(WILLIAMS_SCHEDULE)]
            summary = aggregate_node_directories(nodes)
            expected_ddp_full = SECONDS_PER_STEP["true_full"] / SECONDS_PER_STEP["false_full"]
            expected_audit_true = SECONDS_PER_STEP["true_full"] / SECONDS_PER_STEP["true_min"]
            self.assertAlmostEqual(
                summary["ratios"]["ddp_cost_under_full_audit"]["median_ratio"],
                expected_ddp_full,
            )
            self.assertAlmostEqual(
                summary["ratios"]["audit_cost_under_true"]["median_ratio"],
                expected_audit_true,
            )
            self.assertEqual(summary["production_policy_decision"], "retain_true_full")
            self.assertFalse(summary["false_or_min_outputs_authorized_for_science"])
            self.assertFalse(summary["pilot_policy_change_authorized"])
            self.assertTrue(summary["hard_resource_gate_passed"])

    def test_aggregate_cli_verifier_executes_closed_world_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nodes = [_make_node(root, label) for label in sorted(WILLIAMS_SCHEDULE)]
            summary = aggregate_node_directories(nodes)
            summary["created_utc"] = "20260819T000000Z"
            aggregate = root / "aggregate"
            aggregate.mkdir()
            write_json_exclusive(aggregate / "paired_summary.json", summary)
            write_summary_csv(aggregate / "paired_summary.csv", summary)
            write_json_exclusive(
                aggregate / "STATUS.json",
                {"status": "passed"},
            )
            write_seal(
                aggregate,
                scope="aggregate",
                relative_paths=[
                    "STATUS.json",
                    "paired_summary.csv",
                    "paired_summary.json",
                ],
            )
            result = verify_aggregate(aggregate)
            self.assertEqual(result["status"], "verified")
            self.assertEqual(result["scope"], "aggregate")

    def test_true_policy_rebuild_is_rejected(self) -> None:
        payload = _rank_payload("A", 0)
        payload["reducer_lifecycle"][-1]["has_rebuilt_buckets"] = True
        from perf_benchmark_common import validate_rank_payload

        with self.assertRaisesRegex(PerfContractError, "rebuilt buckets"):
            validate_rank_payload(payload, "true_full")

    def test_verifier_rejects_root_symlink_before_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            node = _make_node(root, "node-0")
            alias = root / "node-alias"
            alias.symlink_to(node, target_is_directory=True)
            with self.assertRaisesRegex(PerfContractError, "symlink"):
                _canonical_artifact_root(alias, "node")


if __name__ == "__main__":
    unittest.main()
