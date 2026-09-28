from __future__ import annotations

from pathlib import Path


def size_metrics(
    encoded_path: Path,
    source_file_bytes: int,
    reference_bytes: int,
    source_integer_bytes: int,
    pixel_count: int,
) -> dict[str, float | int]:
    encoded_bytes = encoded_path.stat().st_size
    return {
        "encoded_bytes": encoded_bytes,
        "encoded_mib": encoded_bytes / 1024**2,
        "bits_per_pixel": encoded_bytes * 8.0 / pixel_count,
        "ratio_source_fits": encoded_bytes / source_file_bytes,
        "ratio_reference_f64": encoded_bytes / reference_bytes,
        "ratio_source_integer": encoded_bytes / source_integer_bytes,
    }
