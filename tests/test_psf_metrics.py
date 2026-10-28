from __future__ import annotations

import numpy as np
import pytest

from aia_psf_deconvolution.physics.psf_metrics import (
    cumulative_fraction_curve,
    distance_grid,
    embed_psf,
    fraction_beyond,
    fraction_threshold_px,
    log_bin_edges,
    profile_at_radius,
    psf_center,
    radial_profile,
    ring_weights,
)


def _curve(psf: np.ndarray) -> np.ndarray:
    distance = distance_grid(psf.shape, psf_center(psf))
    return cumulative_fraction_curve(ring_weights(psf, distance))


def _random_centered_psf(size: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    psf = rng.random((size, size))
    psf[size // 2, size // 2] = psf.sum()
    return psf / psf.sum()


def test_delta_psf_has_no_light_beyond_center() -> None:
    psf = np.zeros((9, 9))
    psf[4, 4] = 1.0
    curve = _curve(psf)
    assert fraction_beyond(curve, 0) == 0.0
    assert fraction_beyond(curve, 100) == 0.0


def test_hand_built_psf_rings() -> None:
    psf = np.zeros((9, 9))
    psf[4, 4] = 0.5
    for row, col in ((3, 4), (5, 4), (4, 3), (4, 5)):
        psf[row, col] = 0.1
    for row, col in ((3, 3), (3, 5), (5, 3), (5, 5)):
        psf[row, col] = 0.025
    curve = _curve(psf)
    assert fraction_beyond(curve, 0) == pytest.approx(0.5)
    assert fraction_beyond(curve, 1) == pytest.approx(0.1)  # diagonals at sqrt(2)
    assert fraction_beyond(curve, 2) == pytest.approx(0.0)


def test_curve_matches_direct_mask_sums() -> None:
    psf = _random_centered_psf()
    distance = distance_grid(psf.shape, psf_center(psf))
    curve = _curve(psf)
    for radius in (0, 1, 3, 5, 10, 16, 22):
        expected = psf[distance > radius].sum() / psf.sum()
        assert fraction_beyond(curve, radius) == pytest.approx(expected, abs=1e-15)
    assert np.all(np.diff(curve) <= 1e-15)


def test_off_center_peak_raises() -> None:
    psf = np.zeros((8, 8))
    psf[3, 4] = 1.0
    with pytest.raises(ValueError, match="not at the array center"):
        psf_center(psf)


def test_fraction_threshold_for_one_pixel_row() -> None:
    assert fraction_threshold_px(1) == 0
    assert fraction_threshold_px(10) == 10


def test_embed_psf_keeps_center_and_sum() -> None:
    psf = _random_centered_psf(size=4)
    embedded = embed_psf(psf, (8, 8))
    assert psf_center(embedded) == (4, 4)
    assert embedded.sum() == pytest.approx(psf.sum())
    with pytest.raises(ValueError):
        embed_psf(np.ones((10, 10)), (8, 8))


def test_radial_profile_of_uniform_psf() -> None:
    size = 32
    psf = np.full((size, size), 1.0 / size**2)
    psf[size // 2, size // 2] *= 2  # make the center the unique maximum
    distance = distance_grid(psf.shape, psf_center(psf))
    rows = radial_profile(
        psf, distance, log_bin_edges(float(distance.max()), per_decade=10)
    )
    outer = [row for row in rows if row["r_lo_px"] > 1.0]
    assert outer
    assert all(row["mean_weight"] == pytest.approx(1.0 / size**2) for row in outer)
    inscribed = [row for row in rows if row["r_hi_px"] <= size // 2 - 1]
    assert all(row["completeness"] == 1.0 for row in inscribed)
    assert rows[-1]["completeness"] < 1.0
    covered = sum(row["mean_weight"] * row["n_pixels"] for row in rows)
    in_range = psf[
        (distance >= 0.5) & (distance < log_bin_edges(float(distance.max()), 10)[-1])
    ].sum()
    assert covered == pytest.approx(in_range)


def test_profile_at_radius_one_covers_first_two_rings() -> None:
    psf = _random_centered_psf(size=16)
    distance = distance_grid(psf.shape, psf_center(psf))
    profile = profile_at_radius(psf, distance, 1.0)
    assert profile["profile_n_pixels"] == 8
    assert profile["profile_completeness"] == 1.0
