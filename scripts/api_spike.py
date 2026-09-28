from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import aiapy.psf
import astropy.units as u
import numpy as np
from astropy.io import fits


def run(source: Path) -> dict[str, object]:
    psf = aiapy.psf.calculate_psf(171 * u.angstrom)
    result: dict[str, object] = {
        "psf_shape": list(psf.shape),
        "psf_dtype": str(psf.dtype),
        "psf_bytes": int(psf.nbytes),
    }
    if psf.shape != (4096, 4096) or psf.dtype != np.float64:
        raise AssertionError("Unexpected PSF shape or dtype")
    del psf

    with fits.open(source, memmap=False) as hdul:
        source_array = np.asarray(
            next(hdu.data for hdu in hdul if hdu.data is not None)
        )
    array = np.asarray(source_array, dtype=np.float64)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for codec in ("GZIP_1", "GZIP_2"):
            path = root / f"{codec}.fits"
            hdu = fits.CompImageHDU(array, compression_type=codec, quantize_level=0.0)
            fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, checksum=True)
            with fits.open(path, memmap=False) as hdul:
                decoded = np.asarray(hdul[1].data)
            exact = bool(
                np.array_equal(
                    np.ascontiguousarray(array).view(np.uint8),
                    np.ascontiguousarray(decoded).view(np.uint8),
                )
            )
            result[f"{codec.lower()}_bit_exact"] = exact
            if not exact:
                raise AssertionError(f"{codec} was not bit exact")

        sample = array[:64, :64]
        for mode in (-1, 1, 2):
            path = root / f"rice_{mode}.fits"
            hdu = fits.CompImageHDU(
                sample,
                compression_type="RICE_1",
                tile_shape=(1, sample.shape[1]),
                quantize_level=16.0,
                quantize_method=mode,
                dither_seed=17,
            )
            fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)
            with fits.open(path, disable_image_compression=True) as hdul:
                header = hdul[1].header
            expected = {
                -1: "NO_DITHER",
                1: "SUBTRACTIVE_DITHER_1",
                2: "SUBTRACTIVE_DITHER_2",
            }[mode]
            if header.get("ZQUANTIZ") != expected:
                raise AssertionError(f"Unexpected ZQUANTIZ for mode {mode}")
            if (header.get("ZTILE1"), header.get("ZTILE2")) != (64, 1):
                raise AssertionError("tile_shape did not create a FITS row strip")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        type=Path,
        nargs="?",
        default=Path(
            "test_files/aia.l1lev1d.171A_2017_09_10T20_00_09.35Z.image_l1lev1d.fits"
        ),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.source), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
