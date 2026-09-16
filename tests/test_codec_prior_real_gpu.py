"""Explicitly gated real-checkpoint smoke; skipped on local/CI machines by default."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import unittest


WORKPACK = Path(__file__).resolve().parents[1]


def load_estimator():
    path = WORKPACK / "scripts" / "estimate_codec_prior.py"
    spec = importlib.util.spec_from_file_location("estimate_codec_prior_real", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(
    os.environ.get("PTC_RUN_REAL_CODEC_A1_SMOKE") == "1",
    "training-machine-only real MusicGen codec smoke; not evidence that full A1 ran",
)
class RealCodecPrefixSmoke(unittest.TestCase):
    def test_standard_mono_codec_accepts_q1_prefix(self) -> None:
        estimator = load_estimator()
        audiocraft_root = Path(os.environ["PTC_A1_AUDIOCRAFT_ROOT"])
        checkpoint = Path(os.environ["PTC_A1_CODEC_CHECKPOINT"])
        expected_sha = os.environ["PTC_A1_CODEC_SHA256"].lower()
        payload = estimator._checkpoint_payload_path(checkpoint)
        self.assertEqual(estimator.sha256_file(payload), expected_sha)
        _, _, torch, _, load_compression_model = estimator._load_runtime(
            audiocraft_root, "cuda:0"
        )
        model = load_compression_model(str(checkpoint.resolve()), device="cuda:0")
        model.requires_grad_(False)
        model.eval()
        contract = estimator.validate_codec_model_contract(model)
        self.assertEqual(contract["channels"], 1)
        self.assertEqual(contract["sample_rate"], 32_000)
        self.assertEqual(contract["frame_rate"], 50.0)
        self.assertEqual(contract["cardinality"], 2_048)
        self.assertEqual(contract["num_codebooks"], 4)
        waveform = torch.zeros(
            1, int(model.channels), int(model.sample_rate), device="cuda:0", dtype=torch.float32
        )
        with torch.inference_mode():
            codes, scale = model.encode(waveform)
            self.assertGreaterEqual(int(codes.shape[1]), 2)
            reconstruction = model.decode(codes[:, :1, :], scale)
        self.assertEqual(tuple(reconstruction.shape[:2]), tuple(waveform.shape[:2]))
        self.assertGreaterEqual(int(reconstruction.shape[-1]), int(waveform.shape[-1]))


if __name__ == "__main__":
    unittest.main()
