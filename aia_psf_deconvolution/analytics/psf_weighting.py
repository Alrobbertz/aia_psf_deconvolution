"""Aggregation, validation and figures for the PSF weighting comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

PSF_SOURCES: tuple[str, ...] = ("grigis_2012", "hofmeister_2025")
PSF_LABELS = {
    "grigis_2012": "Grigis et al. (2012), aiapy",
    "hofmeister_2025": "Hofmeister et al. (2025)",
}
PSF_COLORS = {"grigis_2012": "#2676b8", "hofmeister_2025": "#d1495b"}
SCENE_TYPE_ORDER = ("quiet_sun", "active_region", "flare", "flare_saturated")
FRAME_HALF_WIDTH_PX = 2048
FRAME_DIAGONAL_PX = 2896

# Hofmeister et al. (2025), Sect. 5 / Fig. 14: share of light beyond r over the
# seven EUV channels.
PAPER_HOFMEISTER_RANGES = {10: (0.23, 0.29), 100: (0.11, 0.15), 1000: (0.03, 0.10)}
# Hofmeister et al. (2025), Fig. 14a: total diffracted + scattered light, read
# from the figure to the nearest percent.
PAPER_HOFMEISTER_TOTAL = {
    94: 0.43,
    131: 0.52,
    171: 0.41,
    193: 0.49,
    211: 0.43,
    304: 0.37,
    335: 0.55,
}
PAPER_TOTAL_TOLERANCE = 0.03
# Values measured on the 171 A PSFs during method development (rounded as shown);
# guards against silent changes to the PSF inputs or to the metric code.
REGRESSION_171 = {
    ("hofmeister_2025", 10): (0.262, 0.0005),
    ("hofmeister_2025", 100): (0.121, 0.0005),
    ("hofmeister_2025", 1000): (0.071, 0.0005),
    ("hofmeister_2025", 2048): (0.050, 0.0005),
    ("grigis_2012", 10): (0.185, 0.0005),
    ("grigis_2012", 100): (0.030, 0.0005),
    ("grigis_2012", 1000): (0.0013, 0.00005),
}
PSF_SUM_TOLERANCE = 1e-3
# Lower y-limits for figures. The Grigis PSF is built by FFT convolution, so
# its far wings carry float64 round-off (~1e-20 per pixel, ~1e-13 cumulative)
# rather than physical signal; the limits keep that floor off the plots.
PROFILE_FLOOR = 1e-12
CUMULATIVE_FLOOR = 1e-6


def radius_label(radius_px: int) -> str:
    return ">=1 px" if radius_px == 1 else f">{radius_px} px"


def comparison_table(table: pd.DataFrame) -> pd.DataFrame:
    """Wide Grigis vs Hofmeister table at the standard radii (Goals 1 and 2)."""
    wide = table.pivot_table(
        index=["wavelength_angstrom", "radius_px"],
        columns="psf_source",
        values=["fraction_beyond", "profile_mean_weight_normalized"],
    )
    wide.columns = [f"{metric}__{source}" for metric, source in wide.columns]
    wide = wide.reset_index()
    for metric in ("fraction_beyond", "profile_mean_weight_normalized"):
        grigis = wide.get(f"{metric}__grigis_2012")
        hofmeister = wide.get(f"{metric}__hofmeister_2025")
        if grigis is None or hofmeister is None:
            continue
        wide[f"{metric}__difference"] = hofmeister - grigis
        with np.errstate(divide="ignore", invalid="ignore"):
            wide[f"{metric}__ratio"] = np.where(grigis > 0, hofmeister / grigis, np.nan)
    wide.insert(2, "radius_label", wide["radius_px"].map(radius_label))
    return wide


def validation_checks(table: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Gates (must pass) and indicative literature comparisons.

    ``kind == "gate"`` rows check internal consistency and the 171 A
    regression values; ``kind == "indicative"`` rows compare against values
    read from the published figures and are reported, not enforced.
    """
    rows: list[dict[str, object]] = []
    for item in metadata.itertuples(index=False):
        rows.append(
            {
                "kind": "gate",
                "check": "psf_sum_is_one",
                "psf_source": item.psf_source,
                "wavelength_angstrom": int(item.wavelength_angstrom),
                "observed": float(item.psf_sum),
                "expected": f"1 +/- {PSF_SUM_TOLERANCE:g}",
                "passed": abs(float(item.psf_sum) - 1.0) <= PSF_SUM_TOLERANCE,
            }
        )
        rows.append(
            {
                "kind": "gate",
                "check": "psf_non_negative",
                "psf_source": item.psf_source,
                "wavelength_angstrom": int(item.wavelength_angstrom),
                "observed": float(item.psf_min),
                "expected": ">= 0",
                "passed": float(item.psf_min) >= 0.0,
            }
        )

    indexed = table.set_index(["psf_source", "wavelength_angstrom", "radius_px"])[
        "fraction_beyond"
    ]
    for (source, radius), (value, tolerance) in REGRESSION_171.items():
        key = (source, 171, radius)
        if key not in indexed.index:
            continue
        observed = float(indexed[key])
        rows.append(
            {
                "kind": "gate",
                "check": f"regression_171_fraction_beyond_{radius}px",
                "psf_source": source,
                "wavelength_angstrom": 171,
                "observed": observed,
                "expected": f"{value:g} +/- {tolerance:g}",
                "passed": abs(observed - value) <= tolerance,
            }
        )

    hofmeister = table[table["psf_source"] == "hofmeister_2025"]
    for row in hofmeister.itertuples(index=False):
        radius = int(row.radius_px)
        wavelength = int(row.wavelength_angstrom)
        observed = float(row.fraction_beyond)
        if radius in PAPER_HOFMEISTER_RANGES:
            low, high = PAPER_HOFMEISTER_RANGES[radius]
            rows.append(
                {
                    "kind": "indicative",
                    "check": f"paper_range_fraction_beyond_{radius}px",
                    "psf_source": "hofmeister_2025",
                    "wavelength_angstrom": wavelength,
                    "observed": observed,
                    "expected": f"{low:g} to {high:g}",
                    "passed": low <= round(observed, 2) <= high,
                }
            )
        if radius == 1 and wavelength in PAPER_HOFMEISTER_TOTAL:
            expected = PAPER_HOFMEISTER_TOTAL[wavelength]
            rows.append(
                {
                    "kind": "indicative",
                    "check": "paper_total_diffracted_scattered",
                    "psf_source": "hofmeister_2025",
                    "wavelength_angstrom": wavelength,
                    "observed": observed,
                    "expected": f"{expected:g} +/- {PAPER_TOTAL_TOLERANCE:g}",
                    "passed": abs(observed - expected) <= PAPER_TOTAL_TOLERANCE,
                }
            )
    return pd.DataFrame(rows)


def aggregate_out_of_frame(results: pd.DataFrame) -> pd.DataFrame:
    """Out-of-frame share by PSF x wavelength x scene type (completed rows only)."""
    keys = ["psf_source", "wavelength_angstrom", "scene_type"]
    if results.empty:
        return pd.DataFrame(columns=[*keys, "n_completed", "n_failed"])
    status = results.groupby(keys)["status"]
    counts = pd.DataFrame(
        {
            "n_completed": status.apply(lambda s: int((s == "completed").sum())),
            "n_failed": status.apply(lambda s: int((s != "completed").sum())),
        }
    )
    completed = results[results["status"] == "completed"]
    if completed.empty or "out_of_frame_share" not in completed:
        return counts.reset_index()
    shares = completed.groupby(keys)["out_of_frame_share"].agg(
        mean_share="mean", min_share="min", max_share="max"
    )
    summary = counts.join(shares).reset_index()
    summary["scene_type"] = pd.Categorical(
        summary["scene_type"], categories=_scene_order(summary["scene_type"]), ordered=True
    )
    return summary.sort_values(keys).reset_index(drop=True)


def plot_radial_profiles(profiles: pd.DataFrame, output_path: Path | None = None) -> Figure:
    """Goal 1: azimuthally averaged PSF weight per pixel vs distance."""
    fig, axes = _wavelength_grid(profiles)
    for axis, (wavelength, group) in zip(axes, profiles.groupby("wavelength_angstrom"), strict=False):
        for source, subset in group.groupby("psf_source"):
            subset = subset[subset["r_hi_px"] >= 1.0]
            axis.loglog(
                subset["r_center_px"],
                subset["mean_weight_normalized"],
                color=PSF_COLORS.get(source),
                label=PSF_LABELS.get(source, source),
                lw=1.2,
            )
        _frame_markers(axis)
        axis.set_title(f"{int(wavelength)} A")
        axis.set_xlabel("Distance r (px)")
        axis.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("Mean PSF weight per pixel")
    axes[0].set_ylim(PROFILE_FLOOR, 0.1)
    _finish(fig, axes, "Radial PSF profile (azimuthal mean)", output_path)
    return fig


def plot_cumulative_fraction(
    curves: pd.DataFrame,
    output_path: Path | None = None,
    scale: str = "log",
    r_max: float | None = None,
    r_min: float | None = None,
) -> Figure:
    """Goal 2: fraction of total PSF weight beyond distance r.

    ``scale`` is "log" or "linear"; ``r_min`` / ``r_max`` limit the x-axis to
    r_min <= r <= r_max px.
    """
    if scale not in ("log", "linear"):
        raise ValueError(f"scale must be 'log' or 'linear', got {scale!r}")
    if r_min is not None and r_max is not None and r_min >= r_max:
        raise ValueError(f"r_min ({r_min}) must be less than r_max ({r_max})")
    if r_min is not None:
        curves = curves[curves["radius_px"] >= r_min]
    if r_max is not None:
        curves = curves[curves["radius_px"] <= r_max]
    fig, axes = _wavelength_grid(curves)
    for axis, (wavelength, group) in zip(axes, curves.groupby("wavelength_angstrom"), strict=False):
        for source, subset in group.groupby("psf_source"):
            if scale == "log":
                subset = subset[(subset["radius_px"] >= 1) & (subset["fraction_beyond"] > 0)]
            axis.plot(
                subset["radius_px"],
                100.0 * subset["fraction_beyond"],
                color=PSF_COLORS.get(source),
                label=PSF_LABELS.get(source, source),
                lw=1.2,
            )
        if scale == "log":
            axis.set_xscale("log")
            axis.set_yscale("log")
        _frame_markers(axis, r_max, r_min)
        axis.set_title(f"{int(wavelength)} A")
        axis.set_xlabel("Distance r (px)")
        axis.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("Light beyond r (%)")
    peak = 100.0 * curves["fraction_beyond"].max()
    if scale == "log":
        axes[0].set_ylim(100.0 * CUMULATIVE_FLOOR, min(100.0, 2.0 * peak))
    else:
        axes[0].set_xlim(left=0.0)
        axes[0].set_ylim(0.0, 1.05 * peak)
    if r_min is not None:
        axes[0].set_xlim(left=r_min)
    if r_max is not None:
        axes[0].set_xlim(right=r_max)
    title = "Cumulative fraction of PSF weight beyond r"
    if r_min is not None and r_max is not None:
        title += f" ({r_min:g} <= r <= {r_max:g} px)"
    elif r_min is not None:
        title += f" (r >= {r_min:g} px)"
    elif r_max is not None:
        title += f" (r <= {r_max:g} px)"
    _finish(fig, axes, title, output_path)
    return fig


def plot_fraction_at_radii(table: pd.DataFrame, output_path: Path | None = None) -> Figure:
    """Grouped bars of the fraction beyond each standard radius, per wavelength."""
    radii = sorted(table["radius_px"].unique())
    fig, axes = plt.subplots(1, len(radii), figsize=(3.2 * len(radii), 3.8), sharey=False)
    axes = np.atleast_1d(axes)
    wavelengths = sorted(table["wavelength_angstrom"].unique())
    positions = np.arange(len(wavelengths))
    width = 0.4
    for axis, radius in zip(axes, radii, strict=True):
        subset = table[table["radius_px"] == radius]
        for offset, source in zip((-width / 2, width / 2), PSF_SOURCES, strict=True):
            values = (
                subset[subset["psf_source"] == source]
                .set_index("wavelength_angstrom")["fraction_beyond"]
                .reindex(wavelengths)
            )
            axis.bar(
                positions + offset,
                100.0 * values.to_numpy(),
                width,
                color=PSF_COLORS[source],
                label=PSF_LABELS[source],
            )
        axis.set_xticks(positions, [str(int(w)) for w in wavelengths], rotation=45)
        axis.set_title(radius_label(int(radius)))
        axis.set_xlabel("Wavelength (A)")
        axis.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("Light beyond r (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    if output_path is not None:
        fig.savefig(output_path, dpi=160)
    return fig


def plot_out_of_frame(results: pd.DataFrame, output_path: Path | None = None) -> Figure:
    """Scene-dependent out-of-frame share: one point per scene, by wavelength."""
    completed = results[results["status"] == "completed"]
    scene_types = _scene_order(completed["scene_type"])
    markers = dict(zip(SCENE_TYPE_ORDER, ("o", "s", "^", "D"), strict=True))
    wavelengths = sorted(completed["wavelength_angstrom"].unique())
    fig, axis = plt.subplots(figsize=(10, 4.5))
    for s_index, source in enumerate(PSF_SOURCES):
        source_rows = completed[completed["psf_source"] == source]
        for scene_type in scene_types:
            rows = source_rows[source_rows["scene_type"] == scene_type]
            if rows.empty:
                continue
            x = rows["wavelength_angstrom"].map({w: i for i, w in enumerate(wavelengths)})
            jitter = (s_index - 0.5) * 0.3 + (
                SCENE_TYPE_ORDER.index(scene_type) - 1.5
            ) * 0.05
            axis.scatter(
                x + jitter,
                100.0 * rows["out_of_frame_share"],
                color=PSF_COLORS[source],
                marker=markers.get(scene_type, "o"),
                alpha=0.75,
                label=f"{PSF_LABELS[source]}: {scene_type}",
            )
    axis.set_xticks(range(len(wavelengths)), [str(int(w)) for w in wavelengths])
    axis.set_xlim(-0.5, len(wavelengths) - 0.5)
    axis.set_xlabel("Wavelength (A)")
    axis.set_ylabel("Signal scattered out of frame (%)")
    axis.set_yscale("symlog", linthresh=0.01)
    top = 100.0 * completed["out_of_frame_share"].max() if not completed.empty else 1.0
    axis.set_ylim(0.0, max(top, 0.01) * 2.0)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=160)
    return fig


def _scene_order(values: pd.Series) -> list[str]:
    present = set(values.astype(str))
    ordered = [scene for scene in SCENE_TYPE_ORDER if scene in present]
    return ordered + sorted(present - set(ordered))


def _wavelength_grid(frame: pd.DataFrame) -> tuple[Figure, list[plt.Axes]]:
    n_panels = max(1, frame["wavelength_angstrom"].nunique())
    n_cols = min(4, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, grid = plt.subplots(
        n_rows, n_cols, figsize=(4.0 * n_cols, 3.4 * n_rows), sharex=True, sharey=True,
        squeeze=False,
    )
    axes = list(grid.ravel())
    for axis in axes[n_panels:]:
        axis.set_visible(False)
    return fig, axes[:n_panels]


def _frame_markers(
    axis: plt.Axes, r_max: float | None = None, r_min: float | None = None
) -> None:
    """Mark the frame half-width and half-diagonal, with legend entries and in-panel labels."""
    markers = (
        (FRAME_HALF_WIDTH_PX, "--", "right", f"frame edge ({FRAME_HALF_WIDTH_PX} px)"),
        (FRAME_DIAGONAL_PX, ":", "left", f"frame corner ({FRAME_DIAGONAL_PX:.0f} px)"),
    )
    for radius, style, align, label in markers:
        if (r_max is not None and radius > r_max) or (r_min is not None and radius < r_min):
            continue
        axis.axvline(radius, color="0.4", ls=style, lw=0.8, label=label)
        axis.text(
            radius,
            0.97,
            f" {radius:.0f} px ",
            transform=axis.get_xaxis_transform(),
            rotation=90,
            ha=align,
            va="top",
            fontsize=7,
            color="0.3",
            clip_on=True,
        )


def _finish(fig: Figure, axes: list[plt.Axes], title: str, output_path: Path | None) -> None:
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(title)
    fig.legend(handles, labels, loc="lower right", frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if output_path is not None:
        fig.savefig(output_path, dpi=160)
