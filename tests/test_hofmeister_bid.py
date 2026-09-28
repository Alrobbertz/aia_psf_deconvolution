from __future__ import annotations

import numpy as np
import pytest

from aia_psf_deconvolution.physics.hofmeister_bid import (
    convolve_image,
    deconvolve_bid,
    deconvolve_richardson_lucy,
)


def _delta_psf(size: int) -> np.ndarray:
    psf = np.zeros((size, size), dtype=np.float64)
    psf[size // 2, size // 2] = 1.0
    return psf


def _gaussian_psf(size: int, sigma: float) -> np.ndarray:
    coords = np.arange(size) - size // 2
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return (psf / psf.sum()).astype(np.float64)


def _synthetic_image(size: int) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    image = 100 + 10 * np.sin(xx / 4) + 5 * np.cos(yy / 3)
    return image.astype(np.float64)


def test_convolve_image_identity_with_delta_psf() -> None:
    image = _synthetic_image(64)
    psf = _delta_psf(64)
    convolved = convolve_image(image, psf)
    assert np.allclose(convolved, image, atol=1e-6)


def test_deconvolve_bid_recovers_blurred_source() -> None:
    source = _synthetic_image(64)
    psf = _gaussian_psf(64, sigma=1.5)
    blurred = convolve_image(source, psf)
    recovered = deconvolve_bid(blurred, psf, iterations=50, tolerance=1e-6)
    blurred_error = np.max(np.abs(blurred - source))
    recovered_error = np.max(np.abs(recovered - source))
    assert recovered_error < 0.5 * blurred_error


def test_deconvolve_richardson_lucy_recovers_blurred_source() -> None:
    # pad=False avoids the RL update's known 0/0 instability at hard-zero-padded
    # borders (the img/predicted ratio is undefined there); the source is strictly
    # positive and roughly periodic over the tile, so unpadded FFT convolution is safe.
    source = _synthetic_image(64)
    psf = _gaussian_psf(64, sigma=1.5)
    blurred = convolve_image(source, psf, pad=False)
    recovered = deconvolve_richardson_lucy(blurred, psf, iterations=50, pad=False)
    blurred_error = np.max(np.abs(blurred - source))
    recovered_error = np.max(np.abs(recovered - source))
    assert recovered_error < 0.5 * blurred_error


def test_deconvolve_bid_large_psf_runs() -> None:
    source = _synthetic_image(64)
    psf = _gaussian_psf(128, sigma=1.5)
    blurred = convolve_image(source, psf, large_psf=True)
    recovered = deconvolve_bid(blurred, psf, iterations=10, large_psf=True)
    assert recovered.shape == source.shape


def test_deconvolve_bid_gpu_without_cupy_raises_clear_error() -> None:
    image = _synthetic_image(64)
    psf = _gaussian_psf(64, sigma=1.5)
    with pytest.raises(ImportError, match="cupy"):
        deconvolve_bid(image, psf, iterations=1, use_gpu=True)


def test_deconvolve_richardson_lucy_gpu_without_cupy_raises_clear_error() -> None:
    image = _synthetic_image(64)
    psf = _gaussian_psf(64, sigma=1.5)
    with pytest.raises(ImportError, match="cupy"):
        deconvolve_richardson_lucy(image, psf, iterations=1, use_gpu=True)


def test_convolve_image_gpu_without_cupy_raises_clear_error() -> None:
    image = _synthetic_image(64)
    psf = _gaussian_psf(64, sigma=1.5)
    with pytest.raises(ImportError, match="cupy"):
        convolve_image(image, psf, use_gpu=True)
