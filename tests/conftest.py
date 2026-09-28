"""Shared pytest fixtures providing fully synthetic AIA-like test data.

None of these fixtures read from or depend on any real downloaded SDO/AIA
data. Everything is generated on the fly in pytest's `tmp_path`, so the test
suite has no dependency on a workspace-level dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits


def _build_header(wavelength: int, shape: tuple[int, int]) -> fits.Header:
    """Build a minimal but complete AIA-like FITS header.

    Includes the WCS, instrument, and observer keywords required by
    `interoperability_checks()` (WCS equivalence, sunpy.map.Map construction)
    and `_saturation_metrics()` (DATAMAX).
    """
    header = fits.Header()
    header["TELESCOP"] = "SDO/AIA"
    header["INSTRUME"] = "AIA_3"
    header["WAVELNTH"] = wavelength
    header["WAVEUNIT"] = "angstrom"
    header["DATE-OBS"] = "2019-02-07T11:59:57.000"
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRPIX1"] = shape[1] / 2 + 0.5
    header["CRPIX2"] = shape[0] / 2 + 0.5
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CDELT1"] = 0.6
    header["CDELT2"] = 0.6
    header["CROTA2"] = 0.0
    header["HGLN_OBS"] = 0.0
    header["HGLT_OBS"] = 0.0
    header["DSUN_OBS"] = 1.5e11
    header["RSUN_OBS"] = 960.0
    header["RSUN_REF"] = 6.96e8
    header["DATAMAX"] = 16383
    header["EXPTIME"] = 2.9
    return header


def _synthetic_array(shape: tuple[int, int]) -> np.ndarray:
    y, x = np.mgrid[: shape[0], : shape[1]]
    return (1000 + 50 * np.sin(x / 5) + 20 * np.cos(y / 7)).astype(np.float32)


def _write_fits(path: Path, wavelength: int, shape: tuple[int, int]) -> None:
    header = _build_header(wavelength, shape)
    data = _synthetic_array(shape)
    hdul = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=data, header=header)])
    hdul.writeto(path, overwrite=True)


@pytest.fixture
def synthetic_header() -> fits.Header:
    """A single AIA-like header (171 A, 64x64) with no backing file."""
    return _build_header(171, (64, 64))


@pytest.fixture
def synthetic_fits_path(tmp_path: Path) -> Path:
    """Path to a small synthetic 171 A, 64x64 FITS file (Primary + ImageHDU)."""
    path = tmp_path / "synthetic_171A.fits"
    _write_fits(path, wavelength=171, shape=(64, 64))
    return path


@pytest.fixture
def synthetic_data_root(tmp_path: Path) -> Path:
    """A temp data root with two synthetic scenes and matching manifest CSVs.

    Mirrors the schema `augment_manifest()` expects: scene_manifest.csv,
    download_manifest.csv, fits_verification.csv, and hek_window_summary.csv,
    joined on (scene_id, wavelength_angstrom) [hek on scene_id only]. Backing
    FITS files are tiny (8x8) synthetic images written at plain relative
    paths (no "data/aia_scene_experiment/" marker), which `normalize_source_path`
    treats as already-relative.
    """
    data_root = tmp_path / "aia_scene_experiment"
    data_root.mkdir()

    scenes = {
        "quiet_test_scene": ("quiet_sun", (171, 193)),
        "active_test_scene": ("active_region", (94, 131, 171)),
    }

    scene_rows: list[dict[str, object]] = []
    download_rows: list[dict[str, object]] = []
    verification_rows: list[dict[str, object]] = []
    hek_rows: list[dict[str, object]] = []

    for scene_id, (scene_type, wavelengths) in scenes.items():
        scene_dir = data_root / scene_type / scene_id
        scene_dir.mkdir(parents=True)
        for wavelength in wavelengths:
            filename = f"aia.lev1.{wavelength}A_test_{scene_id}.fits"
            path = scene_dir / filename
            _write_fits(path, wavelength=wavelength, shape=(8, 8))
            local_path = f"{scene_type}/{scene_id}/{filename}"

            scene_rows.append(
                {
                    "scene_id": scene_id,
                    "scene_type": scene_type,
                    "anchor_time": "2019-02-07T12:00:00",
                    "wavelength_angstrom": wavelength,
                }
            )
            download_rows.append(
                {
                    "scene_id": scene_id,
                    "wavelength_angstrom": wavelength,
                    "local_path": local_path,
                    "bytes": path.stat().st_size,
                    "status": "downloaded",
                    "errors": "",
                }
            )
            verification_rows.append(
                {
                    "scene_id": scene_id,
                    "wavelength_angstrom": wavelength,
                    "exists": True,
                    "shape_ok": True,
                    "wavelength_ok": True,
                    "level_metadata": 1.0,
                    "level_ok": True,
                    "map_status": "ok",
                }
            )
        hek_rows.append({"scene_id": scene_id, "query_status": "ok"})

    pd.DataFrame(scene_rows).to_csv(data_root / "scene_manifest.csv", index=False)
    pd.DataFrame(download_rows).to_csv(data_root / "download_manifest.csv", index=False)
    pd.DataFrame(verification_rows).to_csv(
        data_root / "fits_verification.csv", index=False
    )
    pd.DataFrame(hek_rows).to_csv(data_root / "hek_window_summary.csv", index=False)

    return data_root
