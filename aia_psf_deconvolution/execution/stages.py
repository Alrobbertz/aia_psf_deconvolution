from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import sunpy.map
from aiapy.calibrate.utils import get_error_table
from astropy.table import QTable

from aia_psf_deconvolution.config.settings import Settings
from aia_psf_deconvolution.physics.deconvolution import (
    deconvolve_to_reference,
    deconvolve_to_reference_bid,
)
from aia_psf_deconvolution.physics.geometry import on_disk_mask
from aia_psf_deconvolution.physics.psf_cache import (
    get_or_create_hofmeister_psf,
    get_or_create_psf,
)
from aia_psf_deconvolution.results.writer import ResultWriter
from aia_psf_deconvolution.verification.poisson_noise import (
    aiapy_error_map,
    poisson_error_stats,
    sqrt_dn_error_map,
)


def stage_deconvolve(
    manifest: pd.DataFrame,
    settings: Settings,
    run_dir: Path,
    force_recompute: bool = False,
) -> pd.DataFrame:
    psf_dir = settings.psf_cache_dir
    reference_dir = settings.reference_cache_dir
    rows: list[dict[str, object]] = []

    for wavelength, group in manifest.groupby("wavelength_angstrom", sort=True):
        wavelength = int(wavelength)
        print(f"[deconvolve] calculating/loading {wavelength} A PSF", flush=True)
        psf = get_or_create_psf(psf_dir, wavelength, force_recompute=force_recompute)
        for row in group.itertuples(index=False):
            output_path = reference_dir / f"{row.image_id}.npy"
            print(f"[deconvolve] {row.image_id}", flush=True)
            metrics = deconvolve_to_reference(
                Path(row.source_path),
                psf,
                output_path,
                settings.iterations,
                settings.clip_negative,
                force_recompute=force_recompute,
            )
            rows.append(
                {
                    "image_id": row.image_id,
                    "scene_id": row.scene_id,
                    "scene_type": row.scene_type,
                    "wavelength_angstrom": wavelength,
                    **metrics,
                }
            )
        del psf

    references = pd.DataFrame(rows)
    references.to_csv(run_dir / "reference_manifest.csv", index=False)
    return references


def stage_deconvolve_bid(
    manifest: pd.DataFrame,
    settings: Settings,
    run_dir: Path,
    force_recompute: bool = False,
    tolerance: float = 0.1,
) -> pd.DataFrame:
    """Hofmeister/BID counterpart to `stage_deconvolve`.

    Purely additive: does not call or modify `stage_deconvolve`. Fixed BID
    method parameters (large_psf=True, pad=True, estimate_background=True,
    constrain_positive=True) are set in `deconvolve_to_reference_bid`, per
    the design doc. `tolerance` defaults to 0.1 (not settings-driven, since
    it is fixed for this experiment, unlike the treatment-swept storage
    pipeline).
    """
    psf_dir = settings.psf_cache_dir
    reference_dir = settings.reference_cache_dir
    rows: list[dict[str, object]] = []

    for wavelength, group in manifest.groupby("wavelength_angstrom", sort=True):
        wavelength = int(wavelength)
        print(f"[deconvolve-bid] loading {wavelength} A Hofmeister PSF", flush=True)
        psf = get_or_create_hofmeister_psf(
            psf_dir, wavelength, force_recompute=force_recompute
        )
        for row in group.itertuples(index=False):
            output_path = reference_dir / f"{row.image_id}.npy"
            print(f"[deconvolve-bid] {row.image_id}", flush=True)
            metrics = deconvolve_to_reference_bid(
                Path(row.source_path),
                psf,
                output_path,
                settings.iterations,
                tolerance,
                force_recompute=force_recompute,
            )
            rows.append(
                {
                    "image_id": row.image_id,
                    "scene_id": row.scene_id,
                    "scene_type": row.scene_type,
                    "wavelength_angstrom": wavelength,
                    **metrics,
                }
            )
        del psf

    references = pd.DataFrame(rows)
    references.to_csv(run_dir / "bid_reference_manifest.csv", index=False)
    return references


def stage_poisson_noise(
    manifest: pd.DataFrame,
    bid_references: pd.DataFrame,
    run_dir: Path,
    writer: ResultWriter,
) -> None:
    """Compute both Poisson-error metrics, before/after BID deconvolution,
    over three regions (full/on_disk/off_disk), for every image in
    `manifest`. Writes one long-format result row per
    `(image_id, stage, region, metric)` via `writer.append(...)`.
    """
    reference_by_image = bid_references.set_index("image_id")["reference_path"]
    error_tables: dict[int, QTable] = {}

    for row in manifest.itertuples(index=False):
        image_id = row.image_id
        wavelength = int(row.wavelength_angstrom)
        print(f"[poisson-noise] {image_id}", flush=True)

        reference_path = reference_by_image.get(image_id)
        if reference_path is None:
            print(
                f"[poisson-noise] skipping {image_id}: no BID reference found",
                flush=True,
            )
            continue

        if wavelength not in error_tables:
            error_tables[wavelength] = get_error_table()
        error_table = error_tables[wavelength]

        source_map = sunpy.map.Map(Path(row.source_path))
        before = np.ascontiguousarray(source_map.data, dtype=np.float64)
        after = np.ascontiguousarray(
            np.load(Path(reference_path), mmap_mode="r"), dtype=np.float64
        )
        disk_mask = on_disk_mask(source_map)

        stages = {"before": before, "after": after}
        regions: dict[str, np.ndarray | None] = {
            "full": None,
            "on_disk": disk_mask,
            "off_disk": ~disk_mask,
        }
        for stage_name, stage_image in stages.items():
            metric_maps = {
                "sqrt_dn": sqrt_dn_error_map(stage_image),
                "aiapy_estimate_error": aiapy_error_map(
                    stage_image, wavelength, error_table=error_table
                ),
            }
            for metric_name, error_map in metric_maps.items():
                for region_name, region_mask in regions.items():
                    stats = poisson_error_stats(error_map, mask=region_mask)
                    writer.append(
                        {
                            "image_id": image_id,
                            "scene_id": row.scene_id,
                            "scene_type": row.scene_type,
                            "wavelength_angstrom": wavelength,
                            "stage": stage_name,
                            "region": region_name,
                            "metric": metric_name,
                            **stats,
                        }
                    )
