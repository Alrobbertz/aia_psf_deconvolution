from __future__ import annotations

import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
import sunpy.map
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.wcs import FITSFixedWarning

from aia_psf_deconvolution.config.treatments import Treatment
from aia_psf_deconvolution.dataio.fitsio import clean_output_header, image_hdu_index

REQUIRED_PROVENANCE = ("AIAEXPID", "AIATRMT", "AIAREF", "AIAITER", "AIACLIP")
WCS_KEYS = (
    "CTYPE1",
    "CTYPE2",
    "CUNIT1",
    "CUNIT2",
    "CRPIX1",
    "CRPIX2",
    "CRVAL1",
    "CRVAL2",
    "CDELT1",
    "CDELT2",
)


def _wcs_equivalent(source_header: fits.Header, output_header: fits.Header) -> bool:
    source_header = clean_output_header(source_header, integer=False)
    for key in WCS_KEYS:
        if source_header.get(key) != output_header.get(key):
            return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        source_wcs = WCS(source_header)
        output_wcs = WCS(output_header)
    pixels = np.array([[0.0, 0.0], [2047.5, 2047.5], [4095.0, 4095.0]])
    source_world = np.asarray(source_wcs.all_pix2world(pixels, 0))
    output_world = np.asarray(output_wcs.all_pix2world(pixels, 0))
    return bool(np.allclose(source_world, output_world, rtol=0.0, atol=1e-10))


def interoperability_checks(
    path: Path,
    source_header: fits.Header,
    treatment: Treatment,
    expected_shape: tuple[int, int],
) -> dict[str, object]:
    index = image_hdu_index(path)
    with fits.open(path, memmap=False, checksum=True) as hdul:
        hdu = hdul[index]
        output_header = hdu.header.copy()
        shape_ok = tuple(hdu.data.shape) == expected_shape

    with fits.open(
        path, disable_image_compression=True, memmap=True, checksum=True
    ) as hdul:
        raw_hdu = hdul[index]
        raw_header = raw_hdu.header.copy()
        checksum_ok = raw_hdu.verify_checksum() == 1 and raw_hdu.verify_datasum() == 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        output_map = sunpy.map.Map(path)
        source_map = sunpy.map.Map(
            np.zeros((2, 2)), clean_output_header(source_header, integer=False)
        )
    unit_ok = str(output_map.unit) == str(source_map.unit)
    submap = output_map.submap(
        u.Quantity([0, 0], u.pixel), top_right=u.Quantity([15, 15], u.pixel)
    )
    submap_ok = tuple(submap.data.shape) == (16, 16)
    provenance_ok = all(key in output_header for key in REQUIRED_PROVENANCE)
    wcs_ok = _wcs_equivalent(source_header, output_header)

    quantization_ok = True
    zquantiz = raw_header.get("ZQUANTIZ")
    zdither0 = raw_header.get("ZDITHER0")
    if treatment.is_quantized:
        expected_mode = {
            -1: "NO_DITHER",
            1: "SUBTRACTIVE_DITHER_1",
            2: "SUBTRACTIVE_DITHER_2",
        }[int(treatment.quantize_method)]
        quantization_ok = zquantiz == expected_mode
        if treatment.quantize_method in (1, 2):
            quantization_ok = quantization_ok and zdither0 is not None

    return {
        "astropy_reopen_pass": True,
        "sunpy_reopen_pass": True,
        "shape_pass": shape_ok,
        "unit_pass": unit_ok,
        "wcs_pass": wcs_ok,
        "submap_pass": submap_ok,
        "checksum_pass": checksum_ok,
        "provenance_pass": provenance_ok,
        "quantization_header_pass": quantization_ok,
        "zquantiz": zquantiz,
        "zdither0": zdither0,
        "zblank_present": "ZBLANK" in raw_header,
        "interop_pass": bool(
            shape_ok
            and unit_ok
            and wcs_ok
            and submap_ok
            and checksum_ok
            and provenance_ok
            and quantization_ok
        ),
    }
