#!/usr/bin/env python3
"""Build or verify the offline MERT/CLAP/fadtk identity artifact."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Optional, Sequence


WORKPACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKPACK_ROOT / "src"))

from ptc_opd.stage1_artifact import (  # noqa: E402
    Stage1ArtifactError,
    publish_closed_json_artifact,
)
from ptc_opd.stage1_diversity_fad import (  # noqa: E402
    FADTK_VERSION,
    MODEL_PINS_REPORT,
    MODEL_PINS_SEAL_SCHEMA,
    build_model_pins_report,
    verify_model_pins_artifact,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("build", "verify"):
        command = sub.add_parser(name)
        command.add_argument("--mert-snapshot", type=Path, required=True)
        command.add_argument("--clap-checkpoint", type=Path, required=True)
        command.add_argument(
            "--artifact-dir" if name == "verify" else "--output-dir",
            type=Path,
            required=True,
        )
    return result


def _fadtk_identity() -> tuple[Path, str]:
    version = importlib.metadata.version("fadtk")
    if version != FADTK_VERSION:
        raise Stage1ArtifactError("fadtk must be exactly {}".format(FADTK_VERSION))
    import fadtk

    source = getattr(fadtk, "__file__", None)
    if not isinstance(source, str):
        raise Stage1ArtifactError("cannot locate imported fadtk package")
    return Path(source).resolve().parent, version


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    try:
        package_root, version = _fadtk_identity()
        if args.command == "build":
            report = build_model_pins_report(
                mert_snapshot=args.mert_snapshot,
                clap_checkpoint=args.clap_checkpoint,
                fadtk_package_root=package_root,
                fadtk_version=version,
            )
            output = publish_closed_json_artifact(
                args.output_dir,
                report_name=MODEL_PINS_REPORT,
                report=report,
                seal_schema=MODEL_PINS_SEAL_SCHEMA,
                seal_status="complete_local_model_pins",
            )
        else:
            output = args.artifact_dir
        verified = verify_model_pins_artifact(
            output,
            mert_snapshot=args.mert_snapshot,
            clap_checkpoint=args.clap_checkpoint,
            fadtk_package_root=package_root,
            fadtk_version=version,
        )
    except (OSError, RuntimeError, ValueError, Stage1ArtifactError) as exc:
        print(
            json.dumps(
                {"status": "invalid", "error_type": type(exc).__name__, "error": str(exc)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 3
    print(
        json.dumps(
            {
                "status": "verified",
                "artifact_dir": str(Path(output).absolute()),
                "artifact_seal_sha256": verified["artifact_seal_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
