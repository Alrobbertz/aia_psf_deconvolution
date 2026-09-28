from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RATIO_COLUMNS = (
    "ratio_source_fits",
    "ratio_reference_f64",
    "ratio_source_integer",
)


def geometric_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if numeric.size == 0:
        return float("nan")
    if np.any(numeric < 0):
        raise ValueError("Geometric mean is undefined for negative values")
    if np.any(numeric == 0):
        return 0.0
    return float(np.exp(np.mean(np.log(numeric))))


def load_results(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"No result rows found in {path}")
    return pd.read_json(path, lines=True)


def aggregate_treatments(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for treatment_id, group in results.groupby("treatment_id", sort=True):
        completed = group[group["status"] == "completed"]
        row: dict[str, object] = {
            "treatment_id": treatment_id,
            "path": group["path"].iloc[0],
            "representation_dtype": group["representation_dtype"].iloc[0],
            "codec": group["codec"].iloc[0],
            "expected_images": len(group),
            "completed_images": len(completed),
            "failed_images": int((~group["overall_pass"].fillna(False)).sum()),
            "all_images_pass": bool(group["overall_pass"].fillna(False).all()),
            "gm_mae": geometric_mean(completed.get("mae", pd.Series(dtype=float))),
            "zero_mae_images": int(
                (completed.get("mae", pd.Series(dtype=float)) == 0).sum()
            ),
            "worst_max_abs_error": pd.to_numeric(
                completed.get("max_abs_error", pd.Series(dtype=float)), errors="coerce"
            ).max(),
            "median_mae": pd.to_numeric(
                completed.get("mae", pd.Series(dtype=float)), errors="coerce"
            ).median(),
        }
        for column in RATIO_COLUMNS:
            values = completed.get(column, pd.Series(dtype=float))
            row[f"gm_{column}"] = geometric_mean(values)
            row[f"median_{column}"] = pd.to_numeric(values, errors="coerce").median()
            row[f"worst_{column}"] = pd.to_numeric(values, errors="coerce").max()
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_strata(results: pd.DataFrame) -> pd.DataFrame:
    completed = results[results["status"] == "completed"]
    rows: list[dict[str, object]] = []
    keys = ["treatment_id", "wavelength_angstrom", "scene_type"]
    for key, group in completed.groupby(keys, sort=True):
        row: dict[str, object] = dict(zip(keys, key, strict=True))
        row.update(
            {
                "images": len(group),
                "gm_mae": geometric_mean(group["mae"]),
                "median_mae": group["mae"].median(),
                "min_mae": group["mae"].min(),
                "max_mae": group["mae"].max(),
                "worst_max_abs_error": group["max_abs_error"].max(),
            }
        )
        for column in RATIO_COLUMNS:
            row[f"gm_{column}"] = geometric_mean(group[column])
            row[f"median_{column}"] = group[column].median()
            row[f"min_{column}"] = group[column].min()
            row[f"max_{column}"] = group[column].max()
        rows.append(row)
    return pd.DataFrame(rows)
