from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from astropy.io import fits

from aia_psf_deconvolution.config.treatments import Treatment

from .codecs import dither_seed, provenance_header, write_fits
from .paths import represent


def encode(
    output_path: Path,
    reference: np.ndarray,
    source_header: fits.Header,
    source_sha256: str,
    reference_sha256: str,
    treatment: Treatment,
    iterations: int,
    clip_negative: bool,
) -> tuple[np.ndarray, float, int]:
    represented = represent(reference, treatment)
    header = provenance_header(
        source_header,
        treatment,
        reference_sha256,
        iterations,
        clip_negative,
    )
    seed = dither_seed(source_sha256)
    started = time.perf_counter()
    write_fits(output_path, represented, header, treatment, seed)
    elapsed = time.perf_counter() - started
    return represented, elapsed, seed
