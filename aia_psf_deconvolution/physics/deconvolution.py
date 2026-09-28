from __future__ import annotations

import hashlib
import time
from pathlib import Path

import aiapy.psf
import numpy as np
import sunpy.map


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def deconvolve_to_reference(
    source_path: Path,
    psf: np.ndarray,
    output_path: Path,
    iterations: int,
    clip_negative: bool,
    force_recompute: bool = False,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Cache-hit case: load existing reference if available and not forcing recompute
    if not force_recompute and output_path.exists():
        reference = np.load(output_path, mmap_mode="r")
        reference = np.ascontiguousarray(reference, dtype=np.float64)
        return {
            "reference_path": str(output_path),
            "reference_sha256": array_sha256(reference),
            "reference_dtype": str(reference.dtype),
            "reference_bytes": int(reference.nbytes),
            "deconvolution_seconds": 0.0,
            "reference_cached": True,
            "reference_min": float(np.min(reference)),
            "reference_max": float(np.max(reference)),
            "reference_nonfinite": int(reference.size - np.isfinite(reference).sum()),
        }

    # Cache-miss case: compute deconvolution and save
    source_map = sunpy.map.Map(source_path)
    started = time.perf_counter()
    result_map = aiapy.psf.deconvolve(
        source_map,
        psf=psf,
        iterations=iterations,
        clip_negative=clip_negative,
    )
    elapsed = time.perf_counter() - started
    reference = np.ascontiguousarray(result_map.data, dtype=np.float64)
    np.save(output_path, reference, allow_pickle=False)
    return {
        "reference_path": str(output_path),
        "reference_sha256": array_sha256(reference),
        "reference_dtype": str(reference.dtype),
        "reference_bytes": int(reference.nbytes),
        "deconvolution_seconds": elapsed,
        "reference_cached": False,
        "reference_min": float(np.min(reference)),
        "reference_max": float(np.max(reference)),
        "reference_nonfinite": int(reference.size - np.isfinite(reference).sum()),
    }
