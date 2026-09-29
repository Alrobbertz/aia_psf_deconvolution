"""Per-stratum aggregation for the Poisson noise experiment."""

from __future__ import annotations

import pandas as pd


def aggregate_poisson_by_stratum(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-image Poisson-error stats into per-stratum summaries.

    Groups by `(scene_type, region, stage, metric)`. Percentile columns
    (p05/q1/median/q3/p95) are averaged across images -- an approximation,
    consistent with the framework's existing descriptive-not-formal-inference
    style (see `analytics.aggregate`), not recomputed from pooled pixels.
    """
    rows: list[dict[str, object]] = []
    keys = ["scene_type", "region", "stage", "metric"]
    for key, group in results.groupby(keys, sort=True):
        row: dict[str, object] = dict(zip(keys, key, strict=True))
        row.update(
            {
                "images": len(group),
                "mean_of_means": group["mean"].mean(),
                "min_of_mins": group["min"].min(),
                "max_of_maxes": group["max"].max(),
                "mean_p05": group["p05"].mean(),
                "mean_q1": group["q1"].mean(),
                "mean_median": group["median"].mean(),
                "mean_q3": group["q3"].mean(),
                "mean_p95": group["p95"].mean(),
                "mean_fraction_ge_0p5dn": group["fraction_ge_0p5dn"].mean(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
