from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import sunpy.map
from aiapy.calibrate.utils import get_error_table
from astropy.table import QTable

from aia_psf_deconvolution.config.settings import Settings
from aia_psf_deconvolution.physics.deconvolution import (
    array_sha256,
    deconvolve_to_reference,
    deconvolve_to_reference_bid,
)
from aia_psf_deconvolution.physics.geometry import on_disk_mask
from aia_psf_deconvolution.physics.psf_cache import (
    get_or_create_hofmeister_psf,
    get_or_create_psf,
)
from aia_psf_deconvolution.physics.psf_metrics import (
    STANDARD_RADII_PX,
    cumulative_fraction_curve,
    distance_grid,
    embed_psf,
    fraction_beyond,
    fraction_threshold_px,
    log_bin_edges,
    profile_at_radius,
    psf_center,
    radial_profile,
    ring_weights,
)
from aia_psf_deconvolution.results.writer import ResultWriter
from aia_psf_deconvolution.verification.out_of_frame import out_of_frame_metrics
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


PSF_WEIGHTING_SOURCES: tuple[str, ...] = ("grigis_2012", "hofmeister_2025")


def load_native_psf(
    settings: Settings, psf_source: str, wavelength: int, force_recompute: bool = False
) -> np.ndarray:
    """Load a PSF at its native array size (4096^2 Grigis, 8192^2 Hofmeister)."""
    cache_dir = settings.psf_cache_root / psf_source
    if psf_source == "grigis_2012":
        psf = get_or_create_psf(cache_dir, wavelength, force_recompute=force_recompute)
    elif psf_source == "hofmeister_2025":
        psf = get_or_create_hofmeister_psf(cache_dir, wavelength)
    else:
        raise ValueError(f"Unsupported PSF source for psf-weighting: {psf_source}")
    return np.ascontiguousarray(psf, dtype=np.float64)


def stage_psf_weighting(
    settings: Settings,
    run_dir: Path,
    wavelengths: list[int],
    force_recompute: bool = False,
) -> dict[str, pd.DataFrame]:
    """Goals 1 and 2: radial PSF-weight profile and cumulative fraction of light
    beyond r, for every wavelength x PSF source. Scene independent.

    Writes ``psf_metadata.csv``, ``psf_radial_profiles.csv``,
    ``psf_cumulative_fraction.csv`` and ``psf_weighting_table.csv`` to
    ``run_dir``.
    """
    metadata_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    table_rows: list[dict[str, object]] = []
    total = len(wavelengths) * len(PSF_WEIGHTING_SOURCES)
    step = 0

    for wavelength in wavelengths:
        for psf_source in PSF_WEIGHTING_SOURCES:
            step += 1
            print(
                f"[psf-weighting {step}/{total}] {wavelength} A {psf_source} -> radial metrics",
                flush=True,
            )
            psf = load_native_psf(settings, psf_source, wavelength, force_recompute)
            center = psf_center(psf)
            distance = distance_grid(psf.shape, center)
            psf_sum = float(psf.sum())
            curve = cumulative_fraction_curve(ring_weights(psf, distance))
            key = {"psf_source": psf_source, "wavelength_angstrom": wavelength}

            metadata_rows.append(
                {
                    **key,
                    "shape_rows": psf.shape[0],
                    "shape_cols": psf.shape[1],
                    "center_row": center[0],
                    "center_col": center[1],
                    "psf_sum": psf_sum,
                    "psf_min": float(psf.min()),
                    "center_fraction": float(psf[center] / psf_sum),
                    "max_distance_px": float(distance.max()),
                    "psf_sha256": array_sha256(psf),
                }
            )
            for row in radial_profile(
                psf, distance, log_bin_edges(float(distance.max()))
            ):
                profile_rows.append(
                    {
                        **key,
                        **row,
                        "mean_weight_normalized": row["mean_weight"] / psf_sum,
                    }
                )
            curve_rows.extend(
                {**key, "radius_px": k, "fraction_beyond": float(value)}
                for k, value in enumerate(curve)
            )
            for radius in STANDARD_RADII_PX:
                threshold = fraction_threshold_px(radius)
                profile = profile_at_radius(psf, distance, float(radius))
                table_rows.append(
                    {
                        **key,
                        "radius_px": radius,
                        "fraction_threshold_px": threshold,
                        "fraction_beyond": fraction_beyond(curve, threshold),
                        **profile,
                        "profile_mean_weight_normalized": profile["profile_mean_weight"]
                        / psf_sum,
                    }
                )
            del psf, distance, curve

    outputs = {
        "psf_metadata": pd.DataFrame(metadata_rows),
        "psf_radial_profiles": pd.DataFrame(profile_rows),
        "psf_cumulative_fraction": pd.DataFrame(curve_rows),
        "psf_weighting_table": pd.DataFrame(table_rows),
    }
    for name, frame in outputs.items():
        frame.to_csv(run_dir / f"{name}.csv", index=False)
    return outputs


def stage_scene_out_of_frame(
    manifest: pd.DataFrame,
    references_by_source: dict[str, pd.DataFrame],
    settings: Settings,
    writer: ResultWriter,
) -> None:
    """Optional analysis: forward-convolve each PSF's own deconvolved scene and
    record the share of the signal scattered out of the 4096^2 frame.

    Loop order is wavelength-outer, PSF-source, scene-inner so that one
    8192^2 PSF is resident at a time. Failed cells are written as rows with
    ``status="failed"``.
    """
    groups = list(manifest.groupby("wavelength_angstrom", sort=True))
    sources = [
        source for source in PSF_WEIGHTING_SOURCES if source in references_by_source
    ]
    for w_index, (wavelength, group) in enumerate(groups, start=1):
        wavelength = int(wavelength)
        for psf_source in sources:
            references = references_by_source[psf_source].set_index("image_id")
            print(
                f"[out-of-frame {w_index}/{len(groups)}] {wavelength} A {psf_source} -> loading PSF",
                flush=True,
            )
            psf = load_native_psf(settings, psf_source, wavelength)
            large_psfs: dict[tuple[int, int], np.ndarray] = {}
            for row in group.itertuples(index=False):
                print(
                    f"[out-of-frame {w_index}/{len(groups)}] {psf_source} {row.image_id}",
                    flush=True,
                )
                result: dict[str, object] = {
                    "image_id": row.image_id,
                    "scene_id": row.scene_id,
                    "scene_type": row.scene_type,
                    "wavelength_angstrom": wavelength,
                    "psf_source": psf_source,
                }
                try:
                    if row.image_id not in references.index:
                        raise FileNotFoundError(
                            f"no {psf_source} reference for {row.image_id}"
                        )
                    reference = references.loc[row.image_id]
                    scene = np.ascontiguousarray(
                        np.load(Path(reference["reference_path"]), mmap_mode="r"),
                        dtype=np.float64,
                    )
                    target = (2 * scene.shape[0], 2 * scene.shape[1])
                    if target not in large_psfs:
                        large_psfs[target] = (
                            psf if psf.shape == target else embed_psf(psf, target)
                        )
                    result.update(
                        {
                            "reference_path": str(reference["reference_path"]),
                            **out_of_frame_metrics(scene, large_psfs[target]),
                            "status": "completed",
                        }
                    )
                    del scene
                except Exception as exc:  # retain failed cells as rows
                    result.update(
                        {
                            "status": "failed",
                            "failure_stage": "out_of_frame",
                            "failure_reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
                writer.append(result)
            del psf, large_psfs
