#!/usr/bin/env python3
"""Verify a copied Stage-1 generation artifact and every raw-float WAV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence


WORKPACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKPACK_ROOT / "src"))

from ptc_opd.stage1_artifact import sha256_file  # noqa: E402
from ptc_opd.stage1_generation import (  # noqa: E402
    SEAL_NAME,
    verify_generation_artifact,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--eval-manifest-dir", type=Path, required=True)
    parser.add_argument("--skip-pcm-rehash", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = verify_generation_artifact(
            args.generation_dir,
            eval_manifest_dir=args.eval_manifest_dir,
            rehash_pcm=not args.skip_pcm_rehash,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print("STAGE1 GENERATION VERIFY FAILED: {}".format(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "verified",
                "sample_records": result["sample_records"],
                "artifact_seal_sha256": sha256_file(
                    args.generation_dir / SEAL_NAME
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
