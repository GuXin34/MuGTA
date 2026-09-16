"""CPU tests for source, offline, and external-T5 reproducibility identities."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch import nn

from ptc_opd.reproducibility import (
    audiocraft_source_identity,
    loaded_t5_identity,
    require_offline_hf_environment,
)


class _Processor:
    def __init__(self, payload: bytes = b"sentencepiece") -> None:
        self.payload = payload

    def serialized_model_proto(self) -> bytes:
        return self.payload


class _Tokenizer:
    def __init__(self, payload: bytes = b"sentencepiece") -> None:
        self.sp_model = _Processor(payload)
        self.special_tokens_map = {"eos_token": "</s>", "pad_token": "<pad>"}

    def get_vocab(self):
        return {"<pad>": 0, "</s>": 1, "music": 2}


class _Config:
    def to_dict(self):
        return {
            "d_model": 2,
            "_name_or_path": "/machine/specific/cache/t5-base",
            "transformers_version": "test",
        }


class _Conditioner(nn.Module):
    def __init__(self, payload: bytes = b"sentencepiece") -> None:
        super().__init__()
        self.name = "t5-base"
        self.finetune = False
        self.t5_tokenizer = _Tokenizer(payload)
        encoder = nn.Linear(2, 2)
        encoder.config = _Config()
        self.__dict__["t5"] = encoder


class _Provider(nn.Module):
    def __init__(self, payload: bytes = b"sentencepiece") -> None:
        super().__init__()
        self.description = _Conditioner(payload)


class _LM(nn.Module):
    def __init__(self, payload: bytes = b"sentencepiece") -> None:
        super().__init__()
        self.condition_provider = _Provider(payload)


class ReproducibilityIdentityTest(unittest.TestCase):
    def test_offline_environment_is_exact(self) -> None:
        accepted = {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
        self.assertEqual(require_offline_hf_environment(accepted), accepted)
        with self.assertRaisesRegex(RuntimeError, "immutable local caches"):
            require_offline_hf_environment({"HF_HUB_OFFLINE": "true"})

    def test_loaded_t5_identity_is_path_independent_and_sensitive(self) -> None:
        torch.manual_seed(7)
        first = _LM()
        torch.manual_seed(7)
        second = _LM()
        identity_a = loaded_t5_identity(first)
        identity_b = loaded_t5_identity(second)
        self.assertEqual(identity_a, identity_b)

        with torch.no_grad():
            second.condition_provider.description.__dict__["t5"].weight[0, 0].add_(1)
        self.assertNotEqual(
            identity_a["identity_sha256"],
            loaded_t5_identity(second)["identity_sha256"],
        )

        torch.manual_seed(7)
        changed_tokenizer = _LM(payload=b"different sentencepiece")
        self.assertNotEqual(
            identity_a["identity_sha256"],
            loaded_t5_identity(changed_tokenizer)["identity_sha256"],
        )

    def test_loaded_t5_identity_requires_unique_frozen_t5_base(self) -> None:
        lm = _LM()
        lm.condition_provider.description.name = "t5-large"
        with self.assertRaisesRegex(ValueError, "t5-base"):
            loaded_t5_identity(lm)
        lm.condition_provider.description.name = "t5-base"
        lm.condition_provider.description.finetune = True
        with self.assertRaisesRegex(ValueError, "frozen"):
            loaded_t5_identity(lm)

    def test_source_tree_hash_ignores_git_and_cache_but_tracks_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "audiocraft" / "models"
            source.mkdir(parents=True)
            lm = source / "lm.py"
            lm.write_text("VALUE = 1\n", encoding="utf-8")
            (root / ".git").mkdir()
            (root / ".git" / "index").write_bytes(b"first")
            (source / "__pycache__").mkdir()
            (source / "__pycache__" / "lm.pyc").write_bytes(b"cache")
            first = audiocraft_source_identity(root)
            (root / ".git" / "index").write_bytes(b"second")
            (source / "__pycache__" / "lm.pyc").write_bytes(b"changed cache")
            self.assertEqual(first, audiocraft_source_identity(root))
            lm.write_text("VALUE = 2\n", encoding="utf-8")
            self.assertNotEqual(first, audiocraft_source_identity(root))

    def test_source_tree_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "audiocraft" / "models"
            source.mkdir(parents=True)
            (source / "lm.py").write_text("x\n", encoding="utf-8")
            target = root / "target.txt"
            target.write_text("target\n", encoding="utf-8")
            (source / "link.py").symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symlinks"):
                audiocraft_source_identity(root)


if __name__ == "__main__":
    unittest.main()
