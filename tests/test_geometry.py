from __future__ import annotations

import numpy as np
import sunpy.map
from astropy.io import fits

from aia_psf_deconvolution.physics.geometry import on_disk_mask


def _map_with_disk_crossing(shape: tuple[int, int] = (64, 64)) -> sunpy.map.GenericMap:
    """A synthetic AIA-like map whose field of view straddles the solar limb.

    CDELT is deliberately large (50 arcsec/pixel) relative to RSUN_OBS (960
    arcsec) so the center of the frame is on-disk and the corners are
    off-disk -- unlike the tiny/zoomed-in fixtures elsewhere in this test
    suite, where the whole frame sits well inside the disk.
    """
    header = fits.Header()
    header["TELESCOP"] = "SDO/AIA"
    header["INSTRUME"] = "AIA_3"
    header["WAVELNTH"] = 171
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
    header["CDELT1"] = 50.0
    header["CDELT2"] = 50.0
    header["CROTA2"] = 0.0
    header["HGLN_OBS"] = 0.0
    header["HGLT_OBS"] = 0.0
    header["DSUN_OBS"] = 1.5e11
    header["RSUN_OBS"] = 960.0
    header["RSUN_REF"] = 6.96e8
    data = np.ones(shape, dtype=np.float32)
    return sunpy.map.Map(data, header)


def test_on_disk_mask_center_true_corner_false() -> None:
    smap = _map_with_disk_crossing()
    mask = on_disk_mask(smap)
    assert mask.dtype == bool
    assert mask.shape == (64, 64)
    assert mask[32, 32]
    assert not mask[0, 0]
    assert 0 < mask.sum() < mask.size
