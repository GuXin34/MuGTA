#!/usr/bin/env python3
"""Verify every retained Stage-1 prerequisite and publish one readiness seal.

This is a read-only consumer.  It never rebuilds A1, CFG, A2/A3, Node-3, or
T5 artifacts.  The complete trees are hashed before and after validation so a
passing report also proves that verification did not mutate an input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Sequence


WORKPACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKPACK_ROOT / "src"))
sys.path.insert(0, str(WORKPACK_ROOT / "scripts"))

from ptc_opd.cfg_decision import verify_cfg_scale_decision  # noqa: E402
from ptc_opd.codec_prior_artifact import load_codec_prior_artifact  # noqa: E402
from ptc_opd.stage1_artifact import (  # noqa: E402
    Stage1ArtifactError,
    canonical_json_sha256,
    load_json_strict,
    publish_closed_json_artifact,
    require_sha256,
    sha256_file,
    sha256_tree,
    verify_checksum_manifest,
)


SCHEMA_VERSION = "ptc-opd-stage1-upstream-verification-v1"
SEAL_SCHEMA_VERSION = "ptc-opd-stage1-upstream-verification-seal-v1"
REPORT_NAME = "upstream_verification.json"
EXPECTED_T5_ARTIFACT_SEAL_SHA256 = (
    "850f0fab6f8e757ca3d82a01e27c1f3361338d2b75f98db825a48de5853ec7f4"
)
EXPECTED_T5_ARCHIVE_SHA256 = (
    "868cec5a4664566529813ce8ceeb6f263cd69bc61951f5c110bc4cec7ea84af8"
)
EXPECTED_T5_PACKET_STATUS_SHA256 = (
    "d9a09d65f8f57d96cc1ffd7c6205bf2c16704025c4e85e73bd725fe97ae5c1b1"
)
NODE3_GATES = (
    "A01_workpack_seal",
    "A02_workpack_check",
    "A03_overlay_exact",
    "A04_overlay_test",
    "A05_full_tests",
    "A06_eight_process_ratio",
    "B01_uninterrupted_two_updates",
    "B02_controlled_step1_failure",
    "B03_attempt0_immutable",
    "B04_stale_rejected_latest_accepted",
    "B05_resume_equivalence",
    "B06_copied_run_verification",
    "B07_cpu_fp32_conditioners_nocfg",
    "B08_bf16_finite_denominator_teacher",
    "B09_all_rank_memory",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workpack-root", type=Path, default=WORKPACK_ROOT)
    parser.add_argument("--a1-dir", type=Path, required=True)
    parser.add_argument("--small-cfg-decision-dir", type=Path, required=True)
    parser.add_argument("--a2-probe-dir", type=Path, required=True)
    parser.add_argument("--a3-summary-dir", type=Path, required=True)
    parser.add_argument("--node3-evidence-dir", type=Path, required=True)
    parser.add_argument("--t5-closure-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _verify_workpack(root: Path) -> Dict[str, Any]:
    from seal_workpack import MANIFEST_NAME, verify

    records = verify(root)
    return {
        "managed_file_count": len(records),
        "manifest_sha256": sha256_file(root / MANIFEST_NAME),
    }


def _verify_a2_a3(probe_dir: Path, summary_dir: Path) -> Dict[str, Any]:
    # Delayed import keeps --help and pure control-plane unit tests independent
    # of torch.  Formal use runs in the accepted training environment.
    from run_disagreement_probe import verify_probe_artifact, verify_summary_artifact

    probe = verify_probe_artifact(probe_dir, require_primary=True)
    summary = verify_summary_artifact(summary_dir, require_primary=True)
    metadata = summary["metadata"]
    rows = metadata.get("summary_rows")
    if not isinstance(rows, list):
        raise Stage1ArtifactError("A3 summary_rows is absent")
    overall = [row for row in rows if isinstance(row, dict) and row.get("group_type") == "overall"]
    if len(overall) != 1:
        raise Stage1ArtifactError("A3 must contain exactly one overall row")
    row = overall[0]
    tv = metadata.get("tv_u_vs_w")
    if not isinstance(tv, dict):
        raise Stage1ArtifactError("A3 tv_u_vs_w is absent")
    p2 = float(row.get("c50_js_ci95_low", float("-inf"))) > 0.60
    p3 = (
        float(tv.get("ci95_low", float("-inf"))) > 0.05
        and float(row.get("c50_kl_ci95_low", float("-inf"))) > 0.60
    )
    if not p2 or not p3:
        raise Stage1ArtifactError("A3 P2/P3 phenomenon gate failed")
    if metadata.get("input_probe_artifact_seal_sha256") != probe.get(
        "artifact_seal_sha256"
    ):
        raise Stage1ArtifactError("A3 is not bound to the supplied A2 probe")
    return {
        "a2_artifact_seal_sha256": probe["artifact_seal_sha256"],
        "a3_artifact_seal_sha256": summary["artifact_seal_sha256"],
        "p2_c50_js_ci95_low": row["c50_js_ci95_low"],
        "p2_passed": p2,
        "p3_c50_kl_ci95_low": row["c50_kl_ci95_low"],
        "p3_tv_u_vs_w_ci95_low": tv["ci95_low"],
        "p3_passed": p3,
    }


def _verify_node3(directory: Path) -> Dict[str, Any]:
    records = verify_checksum_manifest(
        directory,
        extra_excluded_names=("SHA256SUMS.verify.log",),
    )
    verify_log = (directory / "SHA256SUMS.verify.log").read_text(encoding="utf-8")
    if "status=passed\n" not in verify_log:
        raise Stage1ArtifactError("Node-3 checksum verify log did not pass")
    status = load_json_strict(directory / "STATUS.json")
    if status.get("gate_count") != 15 or status.get("passed_count") != 15:
        raise Stage1ArtifactError("Node-3 is not 15/15")
    if status.get("failure") is not None:
        raise Stage1ArtifactError("Node-3 records a failure")
    gates = status.get("gates")
    if not isinstance(gates, dict) or set(gates) != set(NODE3_GATES):
        raise Stage1ArtifactError("Node-3 gate ID set mismatch")
    for gate_id in NODE3_GATES:
        item = gates[gate_id]
        if not isinstance(item, dict) or item.get("status") != "passed":
            raise Stage1ArtifactError("Node-3 gate did not pass: {}".format(gate_id))
    authorization = status.get("authorization")
    if not isinstance(authorization, dict) or authorization.get(
        "research_training_authorized"
    ) is not False:
        raise Stage1ArtifactError("Node-3 evidence has unexpected authorization")
    return {
        "status_sha256": sha256_file(directory / "STATUS.json"),
        "checksum_manifest_sha256": sha256_file(directory / "SHA256SUMS.txt"),
        "covered_file_count": len(records),
        "passed_count": 15,
    }


def _find_t5_output(closure: Path) -> Path:
    direct = closure / "output_dir"
    if direct.is_dir():
        return direct
    if (closure / "STATUS.json").is_file():
        return closure
    raise Stage1ArtifactError("T5 closure has no output_dir/STATUS.json")


def _resolve_regular_directory(path: Path, label: str) -> Path:
    candidate = path.expanduser().absolute()
    if candidate.is_symlink():
        raise Stage1ArtifactError("{} root must not be a symlink".format(label))
    resolved = candidate.resolve(strict=True)
    if not resolved.is_dir():
        raise Stage1ArtifactError("{} root must be a directory".format(label))
    return resolved


def _verify_t5(closure: Path) -> Dict[str, Any]:
    closure = _resolve_regular_directory(closure, "T5 closure")
    output = _find_t5_output(closure)
    verify_checksum_manifest(output)
    if output != closure:
        verify_checksum_manifest(closure)

    status = load_json_strict(output / "STATUS.json")
    accepted_status = status.get("status") == "equivalent_all_inputs" or status.get(
        "scientific_status"
    ) == "equivalent_all_inputs"
    if not accepted_status or status.get("waiver_closed") is not True:
        raise Stage1ArtifactError("T5 waiver is not closed")
    if status.get("pilot_blocked") is not False:
        raise Stage1ArtifactError("T5 status still blocks the pilot")

    seal_path = output / "artifact_seal.json"
    if sha256_file(seal_path) != EXPECTED_T5_ARTIFACT_SEAL_SHA256:
        raise Stage1ArtifactError("T5 patch02 artifact seal pin mismatch")
    seal = load_json_strict(seal_path)
    if (
        set(seal)
        != {
            "schema_version",
            "status",
            "scientific_config_sha256",
            "members",
        }
        or seal.get("schema_version")
        != "ptc-opd-t5-tokenization-equivalence-seal-v3"
        or seal.get("status") != "equivalent_all_inputs"
    ):
        raise Stage1ArtifactError("T5 artifact seal status mismatch")
    require_sha256(seal.get("scientific_config_sha256"), "T5 scientific config")
    members = seal.get("members")
    if not isinstance(members, dict) or set(members) != {
        "STATUS.json",
        "environment.json",
        "per_case.jsonl.gz",
        "t5_tokenization_equivalence.json",
    }:
        raise Stage1ArtifactError("T5 artifact seal has no members")
    for name, identity in members.items():
        if not isinstance(name, str) or not isinstance(identity, dict):
            raise Stage1ArtifactError("T5 artifact seal member malformed")
        path = output / name
        if set(identity) != {"sha256", "size_bytes"}:
            raise Stage1ArtifactError("T5 artifact member identity malformed")
        if sha256_file(path) != require_sha256(identity["sha256"], "T5 member"):
            raise Stage1ArtifactError("T5 artifact member hash mismatch")
        if path.stat().st_size != identity["size_bytes"]:
            raise Stage1ArtifactError("T5 artifact member size mismatch")

    result = load_json_strict(output / "t5_tokenization_equivalence.json")
    comparison = result.get("comparison")
    decision = result.get("decision")
    if not isinstance(comparison, dict) or not isinstance(decision, dict):
        raise Stage1ArtifactError("T5 result lacks comparison/decision")
    if any(
        comparison.get(field) != expected
        for field, expected in {
            "total_case_count": 4132,
            "formal_batch_count": 1460,
            "formal_entry_count": 5838,
            "manifest_role_occurrences": 2919,
            "unique_nonempty_prompts": 2663,
        }.items()
    ):
        raise Stage1ArtifactError("T5 closure case count differs from 4132")
    mismatch_fields = [
        key
        for key, value in comparison.items()
        if key.endswith("mismatches") and value != 0
    ]
    if mismatch_fields:
        raise Stage1ArtifactError(
            "T5 closure records mismatches: {}".format(sorted(mismatch_fields))
        )
    if (
        comparison.get("candidate_transcript_sha256")
        != comparison.get("second_candidate_transcript_sha256")
        or comparison.get("reference_transcript_sha256")
        != comparison.get("second_reference_transcript_sha256")
    ):
        raise Stage1ArtifactError("T5 independent reload transcripts differ")
    for field in (
        "candidate_transcript_sha256",
        "second_candidate_transcript_sha256",
        "reference_transcript_sha256",
        "second_reference_transcript_sha256",
    ):
        require_sha256(comparison.get(field), "T5 {}".format(field))
    if decision.get("waiver_closed") is not True or decision.get(
        "pilot_blocked"
    ) is not False or decision.get("required_action") != "none":
        raise Stage1ArtifactError("T5 scientific decision is not auto-continue")

    delivery_path = closure / "DELIVERY_STATUS.json"
    if delivery_path.is_file():
        if sha256_file(delivery_path) != EXPECTED_T5_PACKET_STATUS_SHA256:
            raise Stage1ArtifactError("T5 delivery status pin mismatch")
        delivery = load_json_strict(delivery_path)
        if (
            delivery.get("formal_rc") != 0
            or delivery.get("scientific_status") != "equivalent_all_inputs"
            or delivery.get("waiver_closed") is not True
        ):
            raise Stage1ArtifactError("T5 delivery packet did not pass")
    else:
        raise Stage1ArtifactError("T5 DELIVERY_STATUS.json is required")

    archive = closure.with_suffix(".tar.gz")
    if (
        archive.is_symlink()
        or not archive.is_file()
        or sha256_file(archive) != EXPECTED_T5_ARCHIVE_SHA256
    ):
        raise Stage1ArtifactError("T5 return archive pin mismatch or archive missing")
    return {
        "artifact_seal_sha256": EXPECTED_T5_ARTIFACT_SEAL_SHA256,
        "archive_sha256": EXPECTED_T5_ARCHIVE_SHA256,
        "delivery_status_sha256": EXPECTED_T5_PACKET_STATUS_SHA256,
        "case_count": 4132,
        "reload_count": 2,
        "mismatch_count": 0,
        "scientific_status": "equivalent_all_inputs",
    }


def execute(args: argparse.Namespace) -> Path:
    roots = {
        "a1": _resolve_regular_directory(args.a1_dir, "A1-R2"),
        "small_cfg": _resolve_regular_directory(
            args.small_cfg_decision_dir, "small CFG"
        ),
        "a2": _resolve_regular_directory(args.a2_probe_dir, "A2 probe"),
        "a3": _resolve_regular_directory(args.a3_summary_dir, "A3 summary"),
        "node3": _resolve_regular_directory(
            args.node3_evidence_dir, "Node-3 evidence"
        ),
        "t5": _resolve_regular_directory(args.t5_closure_dir, "T5 closure"),
    }
    t5_archive_path = roots["t5"].with_suffix(".tar.gz")
    t5_archive_before = sha256_file(t5_archive_path)
    before = {name: sha256_tree(path) for name, path in roots.items()}
    workpack = _verify_workpack(
        _resolve_regular_directory(args.workpack_root, "Stage-1 workpack")
    )
    a1 = load_codec_prior_artifact(roots["a1"])
    cfg = verify_cfg_scale_decision(roots["small_cfg"], require_selected=True)
    if cfg.selected_cfg_scale != 5.0:
        raise Stage1ArtifactError("retained small CFG decision must select 5.0")
    a2_a3 = _verify_a2_a3(roots["a2"], roots["a3"])
    node3 = _verify_node3(roots["node3"])
    t5 = _verify_t5(roots["t5"])
    after = {name: sha256_tree(path) for name, path in roots.items()}
    t5_archive_after = sha256_file(t5_archive_path)
    if before != after or t5_archive_before != t5_archive_after:
        raise Stage1ArtifactError("an upstream artifact changed during verification")

    scientific_config = {
        "schema_version": SCHEMA_VERSION,
        "t5_patch02_artifact_seal_sha256": EXPECTED_T5_ARTIFACT_SEAL_SHA256,
        "t5_patch02_archive_sha256": EXPECTED_T5_ARCHIVE_SHA256,
        "t5_patch02_delivery_status_sha256": EXPECTED_T5_PACKET_STATUS_SHA256,
        "required_node3_gates": list(NODE3_GATES),
        "a2_p2_c50_js_ci95_low_strictly_gt": 0.60,
        "a3_p3_tv_ci95_low_strictly_gt": 0.05,
        "a3_p3_c50_kl_ci95_low_strictly_gt": 0.60,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "scientific_status": "ready_for_b1_prestability",
        "gate_passed": True,
        "operationally_accepted": True,
        "redline_touched": False,
        "full_b1_passed": False,
        "pending": ["B1_prestability_real_checkpoint", "B1.11_ptc500_stability"],
        "scientific_config": scientific_config,
        "scientific_config_sha256": canonical_json_sha256(scientific_config),
        "workpack": workpack,
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
        "a2_a3": a2_a3,
        "node3": node3,
        "t5": t5,
        "upstream_tree_sha256_before": before,
        "upstream_tree_sha256_after": after,
        "t5_archive_sha256_before": t5_archive_before,
        "t5_archive_sha256_after": t5_archive_after,
        "upstream_seals_before_after_equal": True,
        "required_action": "auto_continue_to_b1_prestability",
    }
    return publish_closed_json_artifact(
        args.output_dir,
        report_name=REPORT_NAME,
        report=report,
        seal_schema=SEAL_SCHEMA_VERSION,
        seal_status="ready_for_b1_prestability",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        output = execute(args)
    except (OSError, ValueError, Stage1ArtifactError) as exc:
        print("STAGE1 UPSTREAM VERIFICATION FAILED: {}".format(exc), file=sys.stderr)
        return 2
    print(json.dumps({"status": "passed", "output_dir": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
