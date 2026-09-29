from __future__ import annotations

import numpy as np
import pytest

from aia_psf_deconvolution.verification.poisson_noise import (
    aiapy_error_map,
    poisson_error_stats,
    sqrt_dn_error_map,
)


def test_sqrt_dn_error_map_clips_negatives() -> None:
    image = np.array([[-4.0, 0.0], [4.0, 16.0]])
    error = sqrt_dn_error_map(image)
    assert error[0, 0] == 0.0
    assert error[0, 1] == 0.0
    assert error[1, 0] == pytest.approx(2.0)
    assert error[1, 1] == pytest.approx(4.0)


def test_aiapy_error_map_is_finite_and_positive() -> None:
    image = np.array([[0.0, 100.0], [1000.0, 16383.0]])
    error = aiapy_error_map(image, wavelength=171)
    assert error.shape == image.shape
    assert error.dtype == np.float64
    assert np.all(np.isfinite(error))
    assert np.all(error >= 0.0)


def test_poisson_error_stats_known_array() -> None:
    # DN=0.25 -> sqrt(0.25) = 0.5, exactly at the 0.5 DN threshold.
    error_map = np.array([[0.0, 0.5], [1.0, 2.0]])
    stats = poisson_error_stats(error_map)
    assert stats["count"] == 4
    assert stats["min"] == pytest.approx(0.0)
    assert stats["max"] == pytest.approx(2.0)
    assert stats["median"] == pytest.approx(0.75)
    assert stats["mean"] == pytest.approx(0.875)
    # values >= 0.5: 0.5, 1.0, 2.0 -> 3 of 4
    assert stats["fraction_ge_0p5dn"] == pytest.approx(0.75)


def test_poisson_error_stats_with_mask() -> None:
    error_map = np.array([[0.0, 10.0], [0.2, 0.3]])
    mask = np.array([[True, False], [True, True]])
    stats = poisson_error_stats(error_map, mask=mask)
    assert stats["count"] == 3
    assert stats["max"] == pytest.approx(0.3)
    assert stats["fraction_ge_0p5dn"] == pytest.approx(0.0)


def test_poisson_error_stats_empty_mask_returns_nans() -> None:
    error_map = np.array([[1.0, 2.0]])
    mask = np.array([[False, False]])
    stats = poisson_error_stats(error_map, mask=mask)
    assert stats["count"] == 0
    assert np.isnan(stats["mean"])
