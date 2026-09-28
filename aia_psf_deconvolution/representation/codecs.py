from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from astropy.io import fits

from aia_psf_deconvolution.config.treatments import Treatment
from aia_psf_deconvolution.dataio.fitsio import clean_output_header


def dither_seed(source_sha256: str) -> int:
    digest = hashlib.sha256(source_sha256.encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") % 1000 + 1


def provenance_header(
    source_header: fits.Header,
    treatment: Treatment,
    reference_sha256: str,
    iterations: int,
    clip_negative: bool,
) -> fits.Header:
    header = clean_output_header(source_header, integer=treatment.is_integer)
    header["AIAEXPID"] = ("AIASTOR1", "AIA storage experiment schema")
    header["AIATRMT"] = (treatment.treatment_id, "Storage treatment ID")
    header["AIAREF"] = (reference_sha256[:16], "Reference SHA-256 prefix")
    header["AIAITER"] = (iterations, "Richardson-Lucy iterations")
    header["AIACLIP"] = (bool(clip_negative), "Negative values clipped")
    if treatment.is_integer:
        header["BLANK"] = (
            int(np.iinfo(np.dtype(treatment.dtype)).min),
            "Reserved missing-data sentinel",
        )
    return header


def build_hdu(
    array: np.ndarray,
    header: fits.Header,
    treatment: Treatment,
    seed: int,
) -> fits.ImageHDU | fits.CompImageHDU:
    if treatment.codec == "NONE":
        return fits.ImageHDU(data=array, header=header, name="SCI")

    kwargs: dict[str, object] = {
        "data": array,
        "header": header,
        "name": "SCI",
        "compression_type": treatment.codec,
    }
    tile_shape = treatment.tile_shape(array.shape)
    if tile_shape is not None:
        kwargs["tile_shape"] = tile_shape
    if treatment.path == "A":
        kwargs["quantize_level"] = treatment.quantize_level
        if treatment.quantize_method is not None:
            kwargs["quantize_method"] = treatment.quantize_method
            kwargs["dither_seed"] = seed
    return fits.CompImageHDU(**kwargs)


def write_fits(
    path: Path,
    array: np.ndarray,
    header: fits.Header,
    treatment: Treatment,
    seed: int,
) -> None:
    hdu = build_hdu(array, header, treatment, seed)
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=False, checksum=True)
