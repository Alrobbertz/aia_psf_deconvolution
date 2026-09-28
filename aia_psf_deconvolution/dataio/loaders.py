from __future__ import annotations

from pathlib import Path

import numpy as np
import sunpy.map
from astropy.io import fits

from .fitsio import image_hdu_index


def load_map(path: Path) -> sunpy.map.GenericMap:
    return sunpy.map.Map(path)


def load_array(path: Path) -> np.ndarray:
    index = image_hdu_index(path)
    with fits.open(path, memmap=False) as hdul:
        return np.asarray(hdul[index].data)


def load_reference(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r")
