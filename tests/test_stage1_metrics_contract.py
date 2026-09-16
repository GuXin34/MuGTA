from pathlib import Path
import tempfile
import unittest

from ptc_opd.stage1_artifact import Stage1ArtifactError
from ptc_opd.stage1_metrics import (
    ACCEPTED_EVALUATOR_IDENTITIES,
    ACCEPTED_MUQ_BACKBONE_TREE_SHA256,
    ACCEPTED_MUQ_CONFIG_FILES,
    ACCEPTED_MUQ_CONFIG_FILE_SET_SHA256,
    _verify_metric_evaluator_chain,
    _verify_metric_quality_values,
    verify_accepted_evaluator_identity,
    verify_metric_artifact,
    verify_quality_artifact,
)


def identity(label):
    details = {}
    if label == "muq_eval":
        details = {
            "config_files": [
                {"basename": name, "sha256": digest}
                for name, digest in sorted(ACCEPTED_MUQ_CONFIG_FILES.items())
            ],
            "config_file_set_sha256": ACCEPTED_MUQ_CONFIG_FILE_SET_SHA256,
            "local_encoder_snapshot_sha256": ACCEPTED_MUQ_BACKBONE_TREE_SHA256,
            "declared_encoder_id": "OpenMuQ/MuQ-large-msd-iter",
        }
    return {
        **ACCEPTED_EVALUATOR_IDENTITIES[label],
        "details": details,
    }


class Stage1MetricContractTests(unittest.TestCase):
    def test_stage1_evaluators_set_offline_flags_before_backend_import(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        for name, backend_import in (
            ("eval_stage1_quality.py", "from eval_cfg_quality import"),
            ("eval_stage1_clap.py", "from eval_cfg_music_clap import"),
        ):
            source = (scripts / name).read_text(encoding="utf-8")
            enforce = source.index("common.enforce_offline_environment()")
            backend = source.index(backend_import)
            self.assertLess(enforce, backend, name)

    def test_metric_verifiers_reject_symlink_roots_before_consuming_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.mkdir()
            link = root / "artifact-link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(Stage1ArtifactError, "root must not be a symlink"):
                verify_quality_artifact(
                    link,
                    generation_dir=root / "unused-generation",
                    eval_manifest_dir=root / "unused-manifest",
                )
            with self.assertRaisesRegex(Stage1ArtifactError, "root must not be a symlink"):
                verify_metric_artifact(
                    link,
                    generation_dir=root / "unused-generation",
                    quality_dir=root / "unused-quality",
                    eval_manifest_dir=root / "unused-manifest",
                )

    def test_final_metric_must_reuse_bound_quality_evaluators(self):
        quality = {
            "muq_eval": identity("muq_eval"),
            "audiobox_aesthetics": identity("audiobox_aesthetics"),
        }
        final = {
            **quality,
            "music_clap": identity("music_clap"),
        }
        _verify_metric_evaluator_chain(final, quality)

        drifted = dict(final)
        drifted["muq_eval"] = dict(identity("muq_eval"))
        drifted["muq_eval"]["checkpoint_sha256"] = "4" * 64
        with self.assertRaisesRegex(
            Stage1ArtifactError, "differs from the accepted CFG identity"
        ):
            _verify_metric_evaluator_chain(drifted, quality)

    def test_each_evaluator_is_bound_to_the_accepted_cfg_identity(self):
        for label in ACCEPTED_EVALUATOR_IDENTITIES:
            accepted = identity(label)
            verify_accepted_evaluator_identity(accepted, label)
            drifted = dict(accepted)
            drifted["checkpoint_sha256"] = "f" * 64
            with self.assertRaisesRegex(
                Stage1ArtifactError, "accepted CFG identity"
            ):
                verify_accepted_evaluator_identity(drifted, label)

    def test_muq_relocation_digest_is_not_mistaken_for_model_drift(self):
        relocated = identity("muq_eval")
        relocated["config_sha256"] = "e" * 64
        verify_accepted_evaluator_identity(relocated, "muq_eval")

        drifted = identity("muq_eval")
        drifted["details"] = dict(drifted["details"])
        drifted["details"]["local_encoder_snapshot_sha256"] = "e" * 64
        with self.assertRaisesRegex(Stage1ArtifactError, "accepted CFG identity"):
            verify_accepted_evaluator_identity(drifted, "muq_eval")

    def test_final_metric_cannot_rewrite_bound_quality_values(self):
        quality = [
            {
                "sample_id": "sample-1",
                "generation_seed": 31001,
                "metrics": {
                    "muq_mi": 0.1,
                    "audiobox_ce": 0.2,
                    "audiobox_pq": 0.3,
                },
            }
        ]
        final = [
            {
                "sample_id": "sample-1",
                "generation_seed": 31001,
                "metrics": {
                    **quality[0]["metrics"],
                    "music_clap": 0.4,
                },
            }
        ]
        _verify_metric_quality_values(final, quality)
        final[0]["metrics"]["muq_mi"] = 99.0
        with self.assertRaisesRegex(Stage1ArtifactError, "changed bound quality"):
            _verify_metric_quality_values(final, quality)


if __name__ == "__main__":
    unittest.main()
