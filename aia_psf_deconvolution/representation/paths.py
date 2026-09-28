from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aia_psf_deconvolution.config.treatments import Treatment


@dataclass
class RangeIntegrityError(ValueError):
    dtype: str
    observed_min: float
    observed_max: float
    allowed_min: int
    allowed_max: int
    reason: str

    def __str__(self) -> str:
        return (
            f"{self.reason}: {self.dtype} allows [{self.allowed_min}, "
            f"{self.allowed_max}], observed rounded range "
            f"[{self.observed_min}, {self.observed_max}]"
        )


def _represent_integer(reference: np.ndarray, dtype: str) -> np.ndarray:
    if not np.isfinite(reference).all():
        raise RangeIntegrityError(dtype, float("nan"), float("nan"), 0, 0, "nonfinite")
    info = np.iinfo(np.dtype(dtype))
    rounded = np.rint(reference)
    observed_min = float(np.min(rounded))
    observed_max = float(np.max(rounded))
    if observed_min < info.min or observed_max > info.max:
        raise RangeIntegrityError(
            dtype,
            observed_min,
            observed_max,
            int(info.min),
            int(info.max),
            "integer_overflow",
        )
    sentinel = int(info.min)
    if np.any(rounded == sentinel):
        raise RangeIntegrityError(
            dtype,
            observed_min,
            observed_max,
            int(info.min),
            int(info.max),
            "blank_sentinel_collision",
        )
    return rounded.astype(dtype, copy=False)


def represent(reference: np.ndarray, treatment: Treatment) -> np.ndarray:
    if treatment.path == "A":
        return np.asarray(reference, dtype=np.dtype(treatment.dtype))
    return _represent_integer(reference, treatment.dtype)
