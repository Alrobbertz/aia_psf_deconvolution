from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from astropy.io import fits

from aia_psf_deconvolution.config.treatments import Treatment
from aia_psf_deconvolution.dataio.fitsio import image_hdu_index


def read_decoded(path: Path) -> tuple[np.ndarray, float]:
    index = image_hdu_index(path)
    started = time.perf_counter()
    with fits.open(path, memmap=False) as hdul:
        decoded = np.asarray(hdul[index].data)
    return decoded, time.perf_counter() - started


def _bit_pattern_equal(expected: np.ndarray, actual: np.ndarray) -> bool:
    if expected.shape != actual.shape:
        return False
    if (
        expected.dtype.kind != actual.dtype.kind
        or expected.dtype.itemsize != actual.dtype.itemsize
    ):
        return False
    native_dtype = expected.dtype.newbyteorder("=")
    left = np.ascontiguousarray(expected.astype(native_dtype, copy=False)).view(
        np.uint8
    )
    right = np.ascontiguousarray(actual.astype(native_dtype, copy=False)).view(np.uint8)
    return bool(np.array_equal(left, right))


def fidelity_metrics(
    path: Path,
    reference: np.ndarray,
    represented: np.ndarray,
    treatment: Treatment,
) -> tuple[dict[str, object], np.ndarray]:
    decoded, read_seconds = read_decoded(path)
    if decoded.shape != reference.shape:
        raise ValueError(
            f"Decoded shape {decoded.shape} does not match reference {reference.shape}"
        )

    delta = np.asarray(decoded, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    finite = np.isfinite(delta)
    if finite.all():
        absolute = np.abs(delta)
        mae = float(np.mean(absolute))
        maximum = float(np.max(absolute))
        median = float(np.median(absolute))
        rmse = float(np.sqrt(np.mean(np.square(delta))))
        p99, p999 = np.percentile(absolute, [99.0, 99.9])
        mean_signed = float(np.mean(delta))
    else:
        mae = maximum = median = rmse = mean_signed = float("nan")
        p99 = p999 = float("nan")

    byte_exact = None
    if treatment.is_byte_exact_control:
        byte_exact = _bit_pattern_equal(represented, decoded)
    source_lsb_pass = bool(finite.all() and maximum <= 0.5)
    metrics: dict[str, object] = {
        "read_seconds": read_seconds,
        "mae": mae,
        "max_abs_error": maximum,
        "median_abs_error": median,
        "rmse": rmse,
        "p99_abs_error": float(p99),
        "p999_abs_error": float(p999),
        "mean_signed_error": mean_signed,
        "source_lsb_pass": source_lsb_pass,
        "byte_exact_pass": byte_exact,
    }
    return metrics, decoded
