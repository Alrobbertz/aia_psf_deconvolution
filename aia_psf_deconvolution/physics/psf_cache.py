from __future__ import annotations

from pathlib import Path

import aiapy.psf
import astropy.units as u
import numpy as np
from astropy.io import fits

HOFMEISTER_PSF_SHAPE = (8192, 8192)


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


def hofmeister_psf_path(cache_dir: Path, wavelength: int) -> Path:
    return cache_dir / f"psf_aia_{int(wavelength)}.fits"


def get_or_create_hofmeister_psf(
    cache_dir: Path, wavelength: int, force_recompute: bool = False
) -> np.ndarray:
    """Load a precomputed Hofmeister et al. PSF for the given wavelength.

    Unlike `get_or_create_psf` (grigis_2012, computed on demand via
    `aiapy.psf.calculate_psf`), Hofmeister PSFs are static files shipped
    alongside the dataset at `psf_aia_{wavelength}.fits` -- there is nothing
    to (re)compute, so `force_recompute` is accepted only for signature
    symmetry with `get_or_create_psf` and has no effect.
    """
    path = hofmeister_psf_path(cache_dir, wavelength)
    if not path.is_file():
        raise FileNotFoundError(f"Hofmeister PSF not found for {wavelength} A: {path}")
    with fits.open(path) as hdul:
        psf = np.asarray(hdul[0].data, dtype=np.float64)
    if psf.shape != HOFMEISTER_PSF_SHAPE:
        raise ValueError(
            f"Unexpected Hofmeister PSF shape for {wavelength} A: {psf.shape}"
        )
    return psf
