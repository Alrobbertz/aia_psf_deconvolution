"""Pixel-wise Poisson (shot-noise) error metrics for AIA images.

Reports two independent metrics per pixel (see
`docs/poisson_noise_experiment.md` for the full derivation and caveats):

1. `sqrt_dn_error_map` -- literal `sqrt(DN)`, a simple conservative lower
   bound. Only exactly correct if one DN corresponds to one counted quantum
   event; AIA's ADC rescales real photon counts through quantum yield and
   gain (combined factor > 1), so this understates the true shot noise.
2. `aiapy_error_map` -- aiapy's off-the-shelf, gain-corrected error estimate
   (shot + read + quantization + compression noise, Boerner et al. 2012
   constants, ported from SolarSoft's `aia_bp_estimate_error.pro`).
   `aiapy.calibrate.estimate_error` is designed for *observed* (raw,
   pre-deconvolution) DN counts; applying it to a post-deconvolution
   reference image is a repurposing beyond its designed use case.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
from aiapy.calibrate import estimate_error
from astropy.table import QTable


def sqrt_dn_error_map(image: np.ndarray) -> np.ndarray:
    """Literal `sqrt(clip(DN, 0, None))` Poisson error, in DN."""
    return np.sqrt(np.clip(image, 0, None))


def aiapy_error_map(
    image: np.ndarray, wavelength: int, error_table: QTable | None = None
) -> np.ndarray:
    """Gain-corrected Poisson error via `aiapy.calibrate.estimate_error`.

    `error_table` can be precomputed once per wavelength via
    `aiapy.calibrate.utils.get_error_table()` and reused across images to
    avoid re-fetching it for every call.
    """
    counts = image * (u.DN / u.pix)
    error = estimate_error(
        counts, int(wavelength) * u.angstrom, error_table=error_table
    )
    return np.asarray(error.to_value(u.DN / u.pix), dtype=np.float64)


def poisson_error_stats(
    error_map: np.ndarray, mask: np.ndarray | None = None
) -> dict[str, object]:
    """Descriptive statistics for a Poisson-error map, optionally masked."""
    values = error_map if mask is None else error_map[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p05": float("nan"),
            "q1": float("nan"),
            "median": float("nan"),
            "q3": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
            "fraction_ge_0p5dn": float("nan"),
        }
    p05, q1, median, q3, p95 = np.percentile(values, [5, 25, 50, 75, 95])
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(p05),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "p95": float(p95),
        "max": float(np.max(values)),
        "fraction_ge_0p5dn": float(np.mean(values >= 0.5)),
    }
