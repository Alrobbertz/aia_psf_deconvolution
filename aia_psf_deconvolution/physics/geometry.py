"""Solar-disk geometry helpers.

`aiapy.utils` has no disk/limb masking capability (only `check_quality_flag`,
`sdo_location`, `telescope_number`, `validate_channel`); disk geometry lives
in `sunpy.map` instead, so this module is a thin wrapper around sunpy's
off-the-shelf functions rather than a hand-derived Tx/Ty/rsun_obs formula.
"""

from __future__ import annotations

import numpy as np
import sunpy.map


def on_disk_mask(sunpy_map: sunpy.map.GenericMap) -> np.ndarray:
    """Boolean mask of pixels on the visible solar disk.

    True where the pixel's helioprojective coordinate falls on the solar
    disk, False elsewhere (including off-limb pixels).
    """
    coordinates = sunpy.map.all_coordinates_from_map(sunpy_map)
    return np.asarray(sunpy.map.coordinate_is_on_solar_disk(coordinates), dtype=bool)
