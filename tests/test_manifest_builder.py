from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_musiccaps_manifests.py"
SPEC = importlib.util.spec_from_file_location("build_musiccaps_manifests", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ManifestBuilderTest(unittest.TestCase):
    def make_csv(self, root: Path) -> Path:
        path = root / "musiccaps.csv"
        fieldnames = [
            "ytid",
            "start_s",
            "end_s",
            "caption",
            "aspect_list",
            "is_audioset_eval",
        ]
        rows = [
            {"ytid": "tr0", "start_s": "0", "end_s": "10", "caption": "Warm jazz trio with brushed drums", "aspect_list": "jazz", "is_audioset_eval": "0"},
            {"ytid": "tr1", "start_s": "1", "end_s": "11", "caption": "Fast electronic dance beat and bright synth", "aspect_list": "edm", "is_audioset_eval": "false"},
            {"ytid": "tr2", "start_s": "2", "end_s": "12", "caption": "Solo acoustic guitar plays a calm melody", "aspect_list": "guitar", "is_audioset_eval": "no"},
            {"ytid": "tr3", "start_s": "3", "end_s": "13", "caption": "Heavy rock drums and distorted electric guitar", "aspect_list": "rock", "is_audioset_eval": "0"},
            {"ytid": "tr4", "start_s": "4", "end_s": "14", "caption": "Orchestral strings build a dramatic theme", "aspect_list": "orchestra", "is_audioset_eval": "0"},
            {"ytid": "ev0", "start_s": "5", "end_s": "15", "caption": "Slow ambient pads with distant bells", "aspect_list": "ambient", "is_audioset_eval": "1"},
            {"ytid": "ev1", "start_s": "6", "end_s": "16", "caption": "Funky bass groove with syncopated drums", "aspect_list": "funk", "is_audioset_eval": "true"},
            {"ytid": "ev2", "start_s": "7", "end_s": "17", "caption": "Gentle piano chords under a violin melody", "aspect_list": "classical", "is_audioset_eval": "yes"},
            # Exact normalized duplicate crossing the official split.  The eval
            # representative wins and the train duplicate cannot leak.
            {"ytid": "ev3", "start_s": "8", "end_s": "18", "caption": "WARM jazz trio—with brushed drums!", "aspect_list": "jazz", "is_audioset_eval": "1"},
        ]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_build_is_deterministic_and_leak_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_csv(root)
            output_a, output_b = root / "a", root / "b"
            common = [
                "--musiccaps-csv", str(source),
                "--allow-noncanonical-counts",
                "--dev-size", "2",
                "--probe-size", "1",
                "--test-size", "2",
            ]
            args_a = MODULE.parse_args(common + ["--output-dir", str(output_a)])
            args_b = MODULE.parse_args(common + ["--output-dir", str(output_b)])
            report_a = MODULE.build(args_a)
            report_b = MODULE.build(args_b)
            self.assertEqual(report_a["output_sha256"], report_b["output_sha256"])

            def read_jsonl(name: str):
                return [json.loads(line) for line in (output_a / name).read_text().splitlines()]

            train = read_jsonl("train.full.jsonl")
            dev = read_jsonl("dev.full.jsonl")
            test = read_jsonl("test.full.jsonl")
            self.assertTrue(all(not row["is_audioset_eval"] for row in train + dev))
            self.assertTrue(all(row["is_audioset_eval"] for row in test))
            selected_ids = [row["sample_id"] for row in train + dev + test]
            self.assertEqual(len(selected_ids), len(set(selected_ids)))
            self.assertEqual(report_a["cross_official_split_duplicate_group_count"], 1)

    def test_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_csv(root)
            output = root / "out"
            arguments = MODULE.parse_args([
                "--musiccaps-csv", str(source),
                "--output-dir", str(output),
                "--allow-noncanonical-counts",
                "--dev-size", "2",
                "--probe-size", "1",
                "--test-size", "2",
            ])
            MODULE.build(arguments)
            with self.assertRaises(FileExistsError):
                MODULE.build(arguments)


if __name__ == "__main__":
    unittest.main()
