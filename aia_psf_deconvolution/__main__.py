from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from aia_psf_deconvolution.analytics import analyze_run
from aia_psf_deconvolution.config import TREATMENTS, Settings, select_treatments
from aia_psf_deconvolution.dataio import augment_manifest
from aia_psf_deconvolution.execution import run_sweep, stage_deconvolve
from aia_psf_deconvolution.execution.scratch import clear_scratch
from aia_psf_deconvolution.results import ResultWriter


def _treatment_ids(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    result: list[str] = []
    for value in values:
        result.extend(item.strip() for item in value.split(",") if item.strip())
    return result


def _scene_ids(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    result: list[str] = []
    for value in values:
        result.extend(item.strip() for item in value.split(",") if item.strip())
    return result


def _metadata_path(run_dir: Path) -> Path:
    return run_dir / "run_metadata.json"


def _write_metadata(path: Path, metadata: dict[str, object]) -> None:
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _execute(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    treatments = select_treatments(_treatment_ids(args.treatments))
    scenes = _scene_ids(getattr(args, "scenes", None))
    force_recompute = getattr(args, "force_recompute", False)
    limit = 1 if args.smoke else args.limit
    run_id, run_dir = settings.create_run_dir()
    config_hash = settings.config_hash([item.treatment_id for item in treatments])
    writer = ResultWriter(run_dir / "results.jsonl", run_id, config_hash)
    clear_scratch(settings.scratch_root)

    metadata: dict[str, object] = {
        "run_id": run_id,
        "config_hash": config_hash,
        "command": args.command,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "settings": settings.as_serializable(),
        "treatments": [item.treatment_id for item in treatments],
        "limit": limit,
        "scenes": scenes,
    }
    metadata_path = _metadata_path(run_dir)
    _write_metadata(metadata_path, metadata)
    print(f"[run] {run_id} -> {run_dir}", flush=True)

    try:
        manifest = augment_manifest(
            settings.data_root,
            run_dir / "augmented_manifest.csv",
            limit=limit,
            scene_ids=scenes,
        )
        if args.command == "augment":
            metadata["status"] = "complete"
            return 0

        references = stage_deconvolve(
            manifest, settings, run_dir, force_recompute=force_recompute
        )
        if args.command == "deconvolve":
            metadata["status"] = "complete"
            return 0

        run_sweep(manifest, references, treatments, settings, writer)
        if args.command == "all":
            outputs = analyze_run(run_dir)
            print(f"[analyze] report -> {outputs['report']}", flush=True)
        metadata["status"] = "complete"
        return 0
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["finished_at"] = datetime.now(UTC).isoformat()
        _write_metadata(metadata_path, metadata)
        clear_scratch(settings.scratch_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aia-experiment", description="AIA FITS storage experiment"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("augment", "deconvolve", "sweep", "all"):
        child = subparsers.add_parser(command)
        child.add_argument("--limit", type=int)
        child.add_argument("--smoke", action="store_true")
        child.add_argument("--force-recompute", action="store_true")
        child.add_argument(
            "--scenes",
            nargs="+",
            help="Scene IDs separated by spaces or commas",
        )
        child.add_argument(
            "--treatments",
            nargs="+",
            help="Treatment IDs separated by spaces or commas",
        )
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("run_dir", type=Path)
    subparsers.add_parser("list-treatments")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list-treatments":
        for treatment in TREATMENTS:
            print(treatment.treatment_id)
        return 0
    if args.command == "analyze":
        outputs = analyze_run(args.run_dir.resolve())
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0
    return _execute(args)


if __name__ == "__main__":
    sys.exit(main())
