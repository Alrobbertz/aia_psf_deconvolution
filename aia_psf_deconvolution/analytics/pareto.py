from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASELINES = {
    "source_fits": "gm_ratio_source_fits",
    "reference_f64": "gm_ratio_reference_f64",
    "source_integer": "gm_ratio_source_integer",
}


def _frontier(frame: pd.DataFrame, x_column: str, y_column: str) -> pd.Series:
    values = frame[[x_column, y_column]].to_numpy(dtype=float)
    frontier = np.ones(len(frame), dtype=bool)
    for index, (x_value, y_value) in enumerate(values):
        dominates = (
            (values[:, 0] <= x_value)
            & (values[:, 1] <= y_value)
            & ((values[:, 0] < x_value) | (values[:, 1] < y_value))
        )
        frontier[index] = not bool(np.any(dominates))
    return pd.Series(frontier, index=frame.index)


def pareto_table(summary: pd.DataFrame) -> pd.DataFrame:
    eligible = summary[
        summary["all_images_pass"]
        & summary["gm_mae"].notna()
        & summary["worst_max_abs_error"].le(0.5)
    ].copy()
    tables: list[pd.DataFrame] = []
    for baseline, x_column in BASELINES.items():
        subset = eligible.dropna(subset=[x_column, "gm_mae"]).copy()
        subset["baseline"] = baseline
        subset["is_pareto"] = _frontier(subset, x_column, "gm_mae")
        subset["size_metric"] = subset[x_column]
        tables.append(subset)
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def plot_pareto(summary: pd.DataFrame, output_path: Path) -> None:
    eligible = summary[summary["all_images_pass"]].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    colors = {"A": "#2676b8", "B": "#d1495b"}
    for axis, (baseline, x_column) in zip(axes, BASELINES.items(), strict=True):
        for path, group in eligible.groupby("path"):
            axis.scatter(
                group[x_column],
                group["gm_mae"],
                label=f"Path {path}",
                color=colors.get(path, "#555555"),
                alpha=0.8,
            )
        anchor = eligible[eligible["treatment_id"] == "B_i16_rice_bs32"]
        if not anchor.empty:
            axis.scatter(
                anchor[x_column],
                anchor["gm_mae"],
                marker="*",
                s=170,
                color="#f2b134",
                edgecolor="black",
                label="L1 recipe anchor",
                zorder=5,
            )
        axis.set_title(baseline.replace("_", " ").title())
        axis.set_xlabel("Geometric mean size ratio")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("GM-MAE (DN)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
