from pathlib import Path
import tempfile
import unittest

from ptc_opd.stage1_artifact import Stage1ArtifactError


class Stage1UpstreamVerifierTests(unittest.TestCase):
    def test_regular_directory_resolver_rejects_root_symlink(self):
        # Importing the script is intentionally delayed so this pure contract
        # test does not pull any GPU dependency into test discovery.
        import importlib.util

        script = Path(__file__).resolve().parents[1] / "scripts" / "verify_stage1_upstreams.py"
        spec = importlib.util.spec_from_file_location("verify_stage1_upstreams_test", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.mkdir()
            link = root / "upstream-link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(Stage1ArtifactError, "must not be a symlink"):
                module._resolve_regular_directory(link, "test upstream")


if __name__ == "__main__":
    unittest.main()
