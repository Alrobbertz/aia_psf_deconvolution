from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from aia_psf_deconvolution.analytics import (
    analyze_poisson_run,
    analyze_psf_weighting_run,
    analyze_run,
)
from aia_psf_deconvolution.config import TREATMENTS, Settings, select_treatments
from aia_psf_deconvolution.dataio import augment_manifest
from aia_psf_deconvolution.execution import (
    run_sweep,
    stage_deconvolve,
    stage_deconvolve_bid,
    stage_poisson_noise,
    stage_psf_weighting,
    stage_scene_out_of_frame,
)
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


def _execute_poisson_noise(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    if (
        settings.deconvolution_method != "bid"
        or settings.psf_source != "hofmeister_2025"
    ):
        raise ValueError(
            "poisson-noise requires AIA_DECONV_METHOD=bid and "
            "AIA_PSF_SOURCE=hofmeister_2025 (got deconvolution_method="
            f"{settings.deconvolution_method!r}, psf_source={settings.psf_source!r})"
        )
    scenes = _scene_ids(getattr(args, "scenes", None))
    force_recompute = getattr(args, "force_recompute", False)
    limit = 1 if args.smoke else args.limit
    run_id, run_dir = settings.create_run_dir()
    config_hash = settings.config_hash([])
    writer = ResultWriter(run_dir / "poisson_results.jsonl", run_id, config_hash)
    clear_scratch(settings.scratch_root)

    metadata: dict[str, object] = {
        "run_id": run_id,
        "config_hash": config_hash,
        "command": args.command,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "settings": settings.as_serializable(),
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
        bid_references = stage_deconvolve_bid(
            manifest, settings, run_dir, force_recompute=force_recompute
        )
        stage_poisson_noise(manifest, bid_references, run_dir, writer)
        outputs = analyze_poisson_run(run_dir)
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


PSF_WEIGHTING_WAVELENGTHS = [94, 131, 171, 193, 211, 304, 335]


def _execute_psf_weighting(args: argparse.Namespace) -> int:
    """Compare the Grigis (aiapy) and Hofmeister PSFs. Both PSF sources are
    fixed by the experiment, so `AIA_PSF_SOURCE` / `AIA_DECONV_METHOD` are
    ignored; each PSF is paired with its own deconvolution method.
    """
    settings = Settings.from_env()
    grigis_settings = dataclasses.replace(
        settings, psf_source="grigis_2012", deconvolution_method="aiapy_rl"
    )
    hofmeister_settings = dataclasses.replace(
        settings, psf_source="hofmeister_2025", deconvolution_method="bid"
    )
    scenes = _scene_ids(getattr(args, "scenes", None))
    force_recompute = getattr(args, "force_recompute", False)
    limit = 1 if args.smoke else args.limit
    run_id, run_dir = settings.create_run_dir()
    config_hash = grigis_settings.config_hash(
        ["psf_weighting", "grigis_2012/aiapy_rl", "hofmeister_2025/bid"]
    )
    writer = ResultWriter(run_dir / "out_of_frame_results.jsonl", run_id, config_hash)
    clear_scratch(settings.scratch_root)

    metadata: dict[str, object] = {
        "run_id": run_id,
        "config_hash": config_hash,
        "command": args.command,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "settings": {
            "grigis_2012": grigis_settings.as_serializable(),
            "hofmeister_2025": hofmeister_settings.as_serializable(),
        },
        "wavelengths": PSF_WEIGHTING_WAVELENGTHS,
        "psf_only": args.psf_only,
        "limit": limit,
        "scenes": scenes,
    }
    metadata_path = _metadata_path(run_dir)
    _write_metadata(metadata_path, metadata)
    print(f"[run] {run_id} -> {run_dir}", flush=True)

    try:
        stage_psf_weighting(
            settings,
            run_dir,
            PSF_WEIGHTING_WAVELENGTHS,
            force_recompute=force_recompute,
        )
        if not args.psf_only:
            manifest = augment_manifest(
                settings.data_root,
                run_dir / "augmented_manifest.csv",
                limit=limit,
                scene_ids=scenes,
            )
            references = {
                "grigis_2012": stage_deconvolve(
                    manifest, grigis_settings, run_dir, force_recompute=force_recompute
                ),
                "hofmeister_2025": stage_deconvolve_bid(
                    manifest,
                    hofmeister_settings,
                    run_dir,
                    force_recompute=force_recompute,
                ),
            }
            stage_scene_out_of_frame(manifest, references, settings, writer)
        outputs = analyze_psf_weighting_run(run_dir)
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

    poisson_noise = subparsers.add_parser("poisson-noise")
    poisson_noise.add_argument("--limit", type=int)
    poisson_noise.add_argument("--smoke", action="store_true")
    poisson_noise.add_argument("--force-recompute", action="store_true")
    poisson_noise.add_argument(
        "--scenes",
        nargs="+",
        help="Scene IDs separated by spaces or commas",
    )

    psf_weighting = subparsers.add_parser(
        "psf-weighting",
        help="Compare Grigis (aiapy) and Hofmeister PSF weighting vs distance",
    )
    psf_weighting.add_argument("--limit", type=int)
    psf_weighting.add_argument("--smoke", action="store_true")
    psf_weighting.add_argument("--force-recompute", action="store_true")
    psf_weighting.add_argument(
        "--scenes",
        nargs="+",
        help="Scene IDs separated by spaces or commas",
    )
    psf_weighting.add_argument(
        "--psf-only",
        action="store_true",
        help="Only the scene-independent PSF metrics; skip the out-of-frame analysis",
    )
    analyze_psf = subparsers.add_parser("analyze-psf-weighting")
    analyze_psf.add_argument("run_dir", type=Path)
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
    if args.command == "poisson-noise":
        return _execute_poisson_noise(args)
    if args.command == "psf-weighting":
        return _execute_psf_weighting(args)
    if args.command == "analyze-psf-weighting":
        outputs = analyze_psf_weighting_run(args.run_dir.resolve())
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0
    return _execute(args)


if __name__ == "__main__":
    sys.exit(main())
