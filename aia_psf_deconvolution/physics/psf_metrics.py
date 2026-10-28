"""Radial PSF-weight metrics: radial profile and cumulative fraction beyond r.

Conventions (see docs/psf_weighting_comparison.md):

- The PSF center is the array element at ``shape // 2`` and must also be the
  PSF maximum. This matches `convolve_image` (which rolls by ``shape // 2``)
  and `aiapy.psf.calculate_psf` (peak at (2048, 2048)).
- Distance ``d`` is the Euclidean distance, in detector pixels, from the
  center pixel to each pixel center.
- The fraction beyond ``r`` is the share of the total PSF weight with
  ``d > r`` (strict). For integer ``r`` this is computed exactly from
  pixel rings indexed by ``ceil(d)``.
"""

from __future__ import annotations

import numpy as np

AIA_DETECTOR_SIZE_PX = 4096
STANDARD_RADII_PX: tuple[int, ...] = (1, 10, 100, 1000, 2048, 2896)
PROFILE_HALF_WIDTH_DEX = 0.05


def fraction_threshold_px(radius_px: int) -> int:
    """Threshold for the cumulative fraction at a standard radius.

    The 1 px row reports everything except the center pixel (``d > 0``,
    equivalently ``d >= 1``); every other row uses ``d > r``.
    """
    return 0 if radius_px == 1 else radius_px


def psf_center(psf: np.ndarray) -> tuple[int, int]:
    expected = (psf.shape[0] // 2, psf.shape[1] // 2)
    peak = tuple(int(value) for value in np.unravel_index(np.argmax(psf), psf.shape))
    if peak != expected:
        raise ValueError(f"PSF maximum at {peak} is not at the array center {expected}")
    return expected


def distance_grid(shape: tuple[int, int], center: tuple[int, int]) -> np.ndarray:
    rows = np.arange(shape[0], dtype=np.float64) - center[0]
    cols = np.arange(shape[1], dtype=np.float64) - center[1]
    return np.hypot(rows[:, None], cols[None, :])


def ring_weights(psf: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """PSF weight summed over pixel rings indexed by ``ceil(d)``.

    Element ``k`` holds the weight with ``k - 1 < d <= k`` (element 0 is the
    center pixel).
    """
    index = np.ceil(distance).astype(np.int64).ravel()
    return np.bincount(index, weights=np.asarray(psf, dtype=np.float64).ravel())


def cumulative_fraction_curve(rings: np.ndarray) -> np.ndarray:
    """Fraction of total weight with ``d > k`` for ``k = 0 .. len(rings) - 1``.

    Sums from the outside in so that small tail values keep float64 precision.
    """
    total = float(rings.sum())
    tail_from = np.cumsum(rings[::-1])[::-1]
    beyond = np.zeros_like(rings)
    beyond[:-1] = tail_from[1:]
    return beyond / total


def fraction_beyond(curve: np.ndarray, threshold_px: int) -> float:
    if threshold_px >= curve.size:
        return 0.0
    return float(curve[threshold_px])


def log_bin_edges(r_max: float, per_decade: int = 50) -> np.ndarray:
    step = 1.0 / per_decade
    exponents = np.arange(np.log10(0.5), np.log10(r_max) + step, step)
    return 10.0**exponents


def radial_profile(
    psf: np.ndarray, distance: np.ndarray, bin_edges: np.ndarray
) -> list[dict[str, float]]:
    """Mean PSF weight per pixel in annuli ``[lo, hi)``; empty bins are dropped."""
    index = np.digitize(distance.ravel(), bin_edges) - 1
    valid = (index >= 0) & (index < bin_edges.size - 1)
    index = index[valid]
    weights = np.asarray(psf, dtype=np.float64).ravel()[valid]
    n_bins = bin_edges.size - 1
    sums = np.bincount(index, weights=weights, minlength=n_bins)
    counts = np.bincount(index, minlength=n_bins)
    inner_radius = _inscribed_radius(psf.shape)
    rows: list[dict[str, float]] = []
    for lo, hi, total, count in zip(
        bin_edges[:-1], bin_edges[1:], sums, counts, strict=True
    ):
        if count == 0:
            continue
        rows.append(
            {
                "r_lo_px": float(lo),
                "r_hi_px": float(hi),
                "r_center_px": float(np.sqrt(lo * hi)),
                "n_pixels": int(count),
                "completeness": _completeness(lo, hi, int(count), inner_radius),
                "mean_weight": float(total / count),
            }
        )
    return rows


def profile_at_radius(
    psf: np.ndarray, distance: np.ndarray, radius_px: float
) -> dict[str, float]:
    """Mean PSF weight per pixel in a +/-0.05 dex annulus around ``radius_px``,
    widened to at least ``[r - 0.5, r + 0.5)`` so it always holds one pixel ring.
    """
    lo = min(radius_px * 10.0**-PROFILE_HALF_WIDTH_DEX, radius_px - 0.5)
    hi = max(radius_px * 10.0**PROFILE_HALF_WIDTH_DEX, radius_px + 0.5)
    mask = (distance >= lo) & (distance < hi)
    count = int(mask.sum())
    mean_weight = (
        float(np.asarray(psf, dtype=np.float64)[mask].mean()) if count else float("nan")
    )
    return {
        "profile_r_lo_px": float(lo),
        "profile_r_hi_px": float(hi),
        "profile_n_pixels": count,
        "profile_completeness": _completeness(
            lo, hi, count, _inscribed_radius(psf.shape)
        ),
        "profile_mean_weight": mean_weight,
    }


def embed_psf(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Place ``psf`` in a zero array of ``shape`` so that center maps to center."""
    if psf.shape[0] > shape[0] or psf.shape[1] > shape[1]:
        raise ValueError(f"Cannot embed PSF of shape {psf.shape} into {shape}")
    output = np.zeros(shape, dtype=np.float64)
    row0 = shape[0] // 2 - psf.shape[0] // 2
    col0 = shape[1] // 2 - psf.shape[1] // 2
    output[row0 : row0 + psf.shape[0], col0 : col0 + psf.shape[1]] = psf
    return output


def _inscribed_radius(shape: tuple[int, int]) -> float:
    """Largest radius whose full circle lies inside the array (center at shape//2)."""
    return float(
        min(
            shape[0] // 2,
            shape[1] // 2,
            shape[0] - shape[0] // 2 - 1,
            shape[1] - shape[1] // 2 - 1,
        )
    )


def _completeness(lo: float, hi: float, count: int, inner_radius: float) -> float:
    """Share of the annulus area inside the array (1.0 for fully inscribed annuli)."""
    if hi <= inner_radius:
        return 1.0
    area = np.pi * (hi**2 - lo**2)
    return float(min(1.0, count / area))
