from __future__ import annotations

import math
from pathlib import Path

from astropy.io import fits

_STRUCTURAL_PREFIXES = ("NAXIS", "ZNAXIS", "ZTILE", "ZNAME", "ZVAL")
_STRUCTURAL_KEYS = {
    "SIMPLE",
    "XTENSION",
    "BITPIX",
    "PCOUNT",
    "GCOUNT",
    "TFIELDS",
    "ZIMAGE",
    "ZBITPIX",
    "ZCMPTYPE",
    "CHECKSUM",
    "DATASUM",
    "BSCALE",
    "BZERO",
    "BLANK",
}


def read_image_header(path: Path) -> fits.Header:
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                return hdu.header.copy()
    raise ValueError(f"No image HDU found in {path}")


def clean_output_header(source: fits.Header, integer: bool) -> fits.Header:
    header = source.copy()
    for key in list(header.keys()):
        if key in _STRUCTURAL_KEYS or key.startswith(_STRUCTURAL_PREFIXES):
            del header[key]
            continue
        try:
            value = header.get(key)
        except Exception:
            del header[key]
            continue
        if isinstance(value, float) and not math.isfinite(value):
            del header[key]
    if integer:
        header["BSCALE"] = (1, "DN per stored integer count")
        header["BZERO"] = (0, "Integer storage zero point")
    return header


def image_hdu_index(path: Path) -> int:
    with fits.open(path, memmap=False) as hdul:
        for index, hdu in enumerate(hdul):
            if hdu.data is not None:
                return index
    raise ValueError(f"No image HDU found in {path}")
