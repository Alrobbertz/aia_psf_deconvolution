from __future__ import annotations

from pathlib import Path

import aiapy.psf
import astropy.units as u
import numpy as np


def psf_cache_path(cache_dir: Path, wavelength: int) -> Path:
    return cache_dir / f"psf_{float(wavelength)}.npy"


def get_or_create_psf(
    cache_dir: Path, wavelength: int, force_recompute: bool = False
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = psf_cache_path(cache_dir, wavelength)
    if force_recompute or not path.exists():
        psf = np.asarray(
            aiapy.psf.calculate_psf(int(wavelength) * u.angstrom), dtype=np.float64
        )
        if psf.shape != (4096, 4096):
            raise ValueError(f"Unexpected PSF shape for {wavelength} A: {psf.shape}")
        np.save(path, psf, allow_pickle=False)
        del psf
    return np.load(path, mmap_mode="r")
