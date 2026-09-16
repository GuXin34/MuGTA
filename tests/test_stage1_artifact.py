from pathlib import Path
import json
import tempfile
import unittest

from ptc_opd.stage1_artifact import (
    Stage1ArtifactError,
    canonical_json_sha256,
    publish_closed_files_artifact,
    publish_closed_json_artifact,
    sha256_file,
    sha256_tree,
    verify_checksum_manifest,
    verify_simple_seal,
)


class Stage1ArtifactTests(unittest.TestCase):
    def test_closed_json_artifact_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifact"
            publish_closed_json_artifact(
                output,
                report_name="report.json",
                report={"value": 1},
                seal_schema="test-seal-v1",
                seal_status="passed",
            )
            verify_simple_seal(
                output,
                seal_name="artifact_seal.json",
                schema_version="test-seal-v1",
                status="passed",
                payload_names=("report.json",),
            )
            with self.assertRaises(Stage1ArtifactError):
                publish_closed_json_artifact(
                    output,
                    report_name="report.json",
                    report={"value": 2},
                    seal_schema="test-seal-v1",
                    seal_status="passed",
                )

    def test_checksum_manifest_is_closed_world_and_tamper_evident(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.txt").write_text("a\n", encoding="utf-8")
            (root / "SHA256SUMS.txt").write_text(
                "{}  a.txt\n".format(sha256_file(root / "a.txt")), encoding="utf-8"
            )
            (root / "SHA256SUMS.txt.sha256").write_text(
                "{}  SHA256SUMS.txt\n".format(
                    sha256_file(root / "SHA256SUMS.txt")
                ),
                encoding="utf-8",
            )
            self.assertEqual(set(verify_checksum_manifest(root)), {"a.txt"})
            (root / "extra.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(Stage1ArtifactError):
                verify_checksum_manifest(root)

    def test_closed_multi_file_artifact_roundtrip_and_unsafe_names_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "artifact"
            publish_closed_files_artifact(
                output,
                payloads={"a.jsonl": b"{}\n", "report.json": b'{"ok":true}\n'},
                seal_schema="test-files-seal-v1",
                seal_status="complete",
            )
            verify_simple_seal(
                output,
                seal_name="artifact_seal.json",
                schema_version="test-files-seal-v1",
                status="complete",
                payload_names=("a.jsonl", "report.json"),
            )
            with self.assertRaises(Stage1ArtifactError):
                publish_closed_files_artifact(
                    root / "unsafe",
                    payloads={"../escape": b"x"},
                    seal_schema="x",
                    seal_status="x",
                )

    def test_tree_identity_is_copy_stable_and_name_sensitive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            one = root / "one"
            two = root / "two"
            one.mkdir()
            two.mkdir()
            (one / "x").write_bytes(b"same")
            (two / "x").write_bytes(b"same")
            self.assertEqual(sha256_tree(one), sha256_tree(two))
            (two / "x").rename(two / "y")
            self.assertNotEqual(sha256_tree(one), sha256_tree(two))

    def test_config_hash_is_canonical_and_newline_free(self):
        self.assertEqual(
            canonical_json_sha256({"b": 2, "a": 1}),
            canonical_json_sha256({"a": 1, "b": 2}),
        )


if __name__ == "__main__":
    unittest.main()
