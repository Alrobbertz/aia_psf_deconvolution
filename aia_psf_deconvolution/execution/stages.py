from __future__ import annotations

from pathlib import Path

import pandas as pd

from aia_psf_deconvolution.config.settings import Settings
from aia_psf_deconvolution.physics.deconvolution import deconvolve_to_reference
from aia_psf_deconvolution.physics.psf_cache import get_or_create_psf


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
