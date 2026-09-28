from __future__ import annotations

import numpy as np

from aia_psf_deconvolution.config.treatments import Treatment


def range_integrity_metrics(
    reference: np.ndarray,
    represented: np.ndarray,
    decoded: np.ndarray,
    treatment: Treatment,
) -> dict[str, object]:
    reference_nonfinite = int(reference.size - np.isfinite(reference).sum())
    decoded_nonfinite = int(decoded.size - np.isfinite(decoded).sum())
    result: dict[str, object] = {
        "reference_nonfinite": reference_nonfinite,
        "decoded_nonfinite": decoded_nonfinite,
        "new_nonfinite": max(0, decoded_nonfinite - reference_nonfinite),
        "integer_code_min": None,
        "integer_code_max": None,
        "integer_code_utilization": None,
        "range_integrity_pass": decoded_nonfinite <= reference_nonfinite,
    }
    if not treatment.is_integer:
        return result

    info = np.iinfo(represented.dtype)
    code_min = int(np.min(represented))
    code_max = int(np.max(represented))
    utilization = (code_max - code_min) / float(info.max - info.min)
    sentinel = int(info.min)
    sentinel_collision = bool(np.any(represented == sentinel))
    result.update(
        {
            "integer_code_min": code_min,
            "integer_code_max": code_max,
            "integer_code_utilization": utilization,
            "blank_sentinel_collision": sentinel_collision,
            "range_integrity_pass": bool(
                result["range_integrity_pass"] and not sentinel_collision
            ),
        }
    )
    return result
