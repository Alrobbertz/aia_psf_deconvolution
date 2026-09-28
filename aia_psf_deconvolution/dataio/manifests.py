from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

IMAGE_KEYS = ["scene_id", "wavelength_angstrom"]
DATASET_MARKER = "data/aia_scene_experiment/"


def _package_versions() -> str:
    versions: dict[str, str] = {}
    for package in ("aiapy", "astropy", "numpy", "pandas", "scipy", "sunpy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "missing"
    return ";".join(f"{key}={value}" for key, value in sorted(versions.items()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_source_path(raw_path: str, data_root: Path) -> tuple[str, Path]:
    normalized = str(raw_path).replace(chr(92), "/")
    lower = normalized.lower()
    marker_index = lower.rfind(DATASET_MARKER)
    if marker_index >= 0:
        relative = normalized[marker_index + len(DATASET_MARKER) :]
    else:
        candidate = Path(normalized)
        if candidate.is_absolute():
            raise ValueError(f"Path is outside expected dataset root: {raw_path}")
        relative = normalized
    relative = relative.lstrip("/")
    resolved = (data_root / Path(relative)).resolve()
    root = data_root.resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError(f"Resolved path escapes data root: {raw_path}")
    return relative, resolved


def _read_manifest(data_root: Path, name: str) -> pd.DataFrame:
    path = data_root / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _assert_unique(frame: pd.DataFrame, keys: list[str], name: str) -> None:
    duplicates = frame.duplicated(keys, keep=False)
    if duplicates.any():
        raise ValueError(f"{name} contains duplicate keys: {keys}")


def _join_manifests(data_root: Path) -> pd.DataFrame:
    scene = _read_manifest(data_root, "scene_manifest.csv")
    download = _read_manifest(data_root, "download_manifest.csv")
    verification = _read_manifest(data_root, "fits_verification.csv")
    hek = _read_manifest(data_root, "hek_window_summary.csv")

    for frame, name in (
        (scene, "scene_manifest"),
        (download, "download_manifest"),
        (verification, "fits_verification"),
    ):
        _assert_unique(frame, IMAGE_KEYS, name)
    _assert_unique(hek, ["scene_id"], "hek_window_summary")

    download = download[
        IMAGE_KEYS + ["local_path", "bytes", "status", "errors"]
    ].rename(
        columns={
            "bytes": "source_file_bytes",
            "status": "download_status",
            "errors": "download_errors",
        }
    )
    verification = verification[
        IMAGE_KEYS
        + [
            "exists",
            "shape_ok",
            "wavelength_ok",
            "level_metadata",
            "level_ok",
            "map_status",
        ]
    ]
    hek = hek.drop(
        columns=["scene_type", "anchor_time", "expected_event"], errors="ignore"
    )

    joined = scene.merge(download, on=IMAGE_KEYS, how="inner", validate="one_to_one")
    joined = joined.merge(
        verification, on=IMAGE_KEYS, how="inner", validate="one_to_one"
    )
    joined = joined.merge(hek, on="scene_id", how="left", validate="many_to_one")
    if len(joined) != len(scene):
        raise ValueError("Manifest joins changed the image count")
    if joined["query_status"].isna().any():
        raise ValueError("At least one image has no HEK scene context")
    return joined.sort_values(["wavelength_angstrom", "scene_id"]).reset_index(
        drop=True
    )


def _inspect_source(path: Path) -> dict[str, object]:
    with fits.open(path, disable_image_compression=True, memmap=True) as hdul:
        raw = hdul[1].header
        zbitpix = int(raw.get("ZBITPIX", raw.get("BITPIX")))
        width = int(raw.get("ZNAXIS1", raw.get("NAXIS1")))
        height = int(raw.get("ZNAXIS2", raw.get("NAXIS2")))
        zscale = float(raw.get("ZSCALE", 1.0))
        zzero = float(raw.get("ZZERO", 0.0))
    with fits.open(path, memmap=False) as hdul:
        image_hdu = next(hdu for hdu in hdul if hdu.data is not None)
        data = np.asarray(image_hdu.data)
        loaded_dtype = str(data.dtype)
        header = image_hdu.header
        saturation = _saturation_metrics(data, header)
    return {
        "source_sha256": _sha256(path),
        "source_loaded_dtype": loaded_dtype,
        "source_zbitpix": zbitpix,
        "source_bscale": zscale,
        "source_bzero": zzero,
        "source_integer_bytes": width * height * abs(zbitpix) // 8,
        "source_width": width,
        "source_height": height,
        **saturation,
    }


def _adc_ceiling(data: np.ndarray, header) -> float:
    """Best available AIA 14-bit ADC saturation ceiling in DN.

    AIA Level 1 pixels saturate at the detector's 14-bit ADC limit (nominally
    16383 DN), NOT at the int16 storage-container limit of 32767 DN. Prefer the
    per-file DATAMAX header when present; otherwise fall back to the observed
    maximum, clamped to the nominal 14-bit ceiling.
    """
    nominal = 16383.0
    datamax = header.get("DATAMAX")
    try:
        if datamax is not None:
            value = float(datamax)
            if value > 0:
                return value
    except (TypeError, ValueError):
        pass
    observed = float(np.nanmax(data)) if data.size else nominal
    return min(observed, nominal) if observed > 0 else nominal


def _saturation_metrics(data: np.ndarray, header) -> dict[str, object]:
    ceiling = _adc_ceiling(data, header)
    finite = np.isfinite(data)
    total = int(finite.sum())
    saturated = int(np.count_nonzero(finite & (data >= ceiling)))
    fraction = (saturated / total) if total else 0.0
    return {
        "source_adc_ceiling_dn": ceiling,
        "source_max_dn": float(np.nanmax(data)) if data.size else float("nan"),
        "source_saturated_pixels": saturated,
        "source_saturated_fraction": fraction,
        "source_is_saturated": bool(saturated > 0),
    }


def augment_manifest(
    data_root: Path,
    output_path: Path,
    limit: int | None = None,
    scene_ids: list[str] | None = None,
) -> pd.DataFrame:
    joined = _join_manifests(data_root)
    if scene_ids:
        available = set(joined["scene_id"].unique())
        missing = sorted(set(scene_ids) - available)
        if missing:
            raise ValueError(f"Unknown scene IDs: {', '.join(missing)}")
        joined = joined[joined["scene_id"].isin(scene_ids)].reset_index(drop=True)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        joined = joined.head(limit).copy()
    if joined.empty:
        raise ValueError("No images selected after applying scene/limit filters")

    details: list[dict[str, object]] = []
    package_versions = _package_versions()
    for row in joined.itertuples(index=False):
        relative, source_path = normalize_source_path(row.local_path, data_root)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        inspected = _inspect_source(source_path)
        details.append(
            {
                "image_id": f"{row.scene_id}__{int(row.wavelength_angstrom):04d}A",
                "source_relative_path": relative,
                "source_path": str(source_path),
                "source_file_bytes": source_path.stat().st_size,
                "package_versions": package_versions,
                **inspected,
            }
        )
    detail_frame = pd.DataFrame(details)
    result = pd.concat(
        [joined.drop(columns=["local_path", "source_file_bytes"]), detail_frame],
        axis=1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result
