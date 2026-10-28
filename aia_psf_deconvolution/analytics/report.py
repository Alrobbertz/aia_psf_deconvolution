from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .aggregate import aggregate_strata, aggregate_treatments, load_results
from .pareto import pareto_table, plot_pareto
from .poisson_aggregate import aggregate_poisson_by_stratum
from .psf_weighting import (
    aggregate_out_of_frame,
    comparison_table,
    plot_cumulative_fraction,
    plot_fraction_at_radii,
    plot_out_of_frame,
    plot_radial_profiles,
    validation_checks,
)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "None."
    selected = frame[columns].copy()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in selected.itertuples(index=False, name=None):
        rendered: list[str] = []
        for value in values:
            if pd.isna(value):
                rendered.append("")
            elif isinstance(value, float):
                rendered.append(f"{value:.6g}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def analyze_run(run_dir: Path) -> dict[str, Path]:
    results = load_results(run_dir / "results.jsonl")
    summary = aggregate_treatments(results)
    strata = aggregate_strata(results)
    pareto = pareto_table(summary)
    analytics_dir = run_dir / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)

    summary_path = analytics_dir / "treatment_summary.csv"
    strata_path = analytics_dir / "stratum_summary.csv"
    pareto_path = analytics_dir / "pareto.csv"
    plot_path = analytics_dir / "pareto.png"
    report_path = analytics_dir / "report.md"
    summary.to_csv(summary_path, index=False)
    strata.to_csv(strata_path, index=False)
    pareto.to_csv(pareto_path, index=False)
    plot_pareto(summary, plot_path)

    controls = summary[
        summary["treatment_id"].str.startswith("A_")
        & summary["treatment_id"].str.contains("none|gzip", regex=True)
        & summary["all_images_pass"]
    ]
    candidates = pareto[pareto.get("is_pareto", False)] if not pareto.empty else pareto
    rejected = summary[~summary["all_images_pass"]]
    columns = [
        "treatment_id",
        "gm_ratio_source_fits",
        "gm_mae",
        "worst_max_abs_error",
    ]
    report = "\n".join(
        [
            "# AIA Storage Pilot Report",
            "",
            "## Byte-Exact Controls",
            "",
            _markdown_table(controls, columns),
            "",
            "## Pareto Candidates",
            "",
            _markdown_table(candidates.drop_duplicates("treatment_id"), columns),
            "",
            "## Rejected Treatments",
            "",
            _markdown_table(
                rejected,
                ["treatment_id", "failed_images", "worst_max_abs_error"],
            ),
            "",
        ]
    )
    report_path.write_text(report, encoding="utf-8")
    return {
        "summary": summary_path,
        "strata": strata_path,
        "pareto": pareto_path,
        "plot": plot_path,
        "report": report_path,
    }


def analyze_poisson_run(run_dir: Path) -> dict[str, Path]:
    """Aggregate `poisson_results.jsonl` and write a headline markdown report.

    Mirrors `analyze_run`, but for the Poisson noise experiment: a separate
    results file, a `(scene_type, region, stage, metric)` stratum table, and
    a headline comparing both Poisson metrics (`sqrt_dn` vs.
    `aiapy_estimate_error`) side-by-side.
    """
    results = load_results(run_dir / "poisson_results.jsonl")
    strata = aggregate_poisson_by_stratum(results)
    analytics_dir = run_dir / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)

    strata_path = analytics_dir / "poisson_stratum_summary.csv"
    report_path = analytics_dir / "poisson_report.md"
    strata.to_csv(strata_path, index=False)

    on_disk_after = strata[
        (strata["region"] == "on_disk") & (strata["stage"] == "after")
    ].sort_values(["metric", "scene_type"])
    columns = ["scene_type", "metric", "mean_of_means", "mean_fraction_ge_0p5dn"]

    headlines: list[str] = []
    for metric_name, metric_group in on_disk_after.groupby("metric", sort=True):
        detail = ", ".join(
            f"{scene_type}: {fraction:.4g}"
            for scene_type, fraction in zip(
                metric_group["scene_type"],
                metric_group["mean_fraction_ge_0p5dn"],
                strict=True,
            )
        )
        headlines.append(
            f"- `{metric_name}` (on-disk, post-deconvolution): fraction of pixels "
            f"with Poisson error >= 0.5 DN by scene_type -- {detail}"
        )

    report = "\n".join(
        [
            "# AIA Poisson Noise Report",
            "",
            "## Headline: fraction of pixels with Poisson error >= 0.5 DN",
            "",
            *headlines,
            "",
            "## On-Disk, Post-Deconvolution, by Scene Type and Metric",
            "",
            _markdown_table(on_disk_after, columns),
            "",
            "## Full Stratum Summary",
            "",
            _markdown_table(
                strata.sort_values(["scene_type", "region", "stage", "metric"]),
                ["scene_type", "region", "stage", "metric", *columns[2:]],
            ),
            "",
        ]
    )
    report_path.write_text(report, encoding="utf-8")
    return {"strata": strata_path, "report": report_path}


def analyze_psf_weighting_run(run_dir: Path) -> dict[str, Path]:
    """Tables, validation checks, figures and a markdown report for a
    `psf-weighting` run. The scene out-of-frame section is included only when
    `out_of_frame_results.jsonl` exists and is non-empty.
    """
    metadata = pd.read_csv(run_dir / "psf_metadata.csv")
    table = pd.read_csv(run_dir / "psf_weighting_table.csv")
    profiles = pd.read_csv(run_dir / "psf_radial_profiles.csv")
    curves = pd.read_csv(run_dir / "psf_cumulative_fraction.csv")
    analytics_dir = run_dir / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)

    comparison = comparison_table(table)
    checks = validation_checks(table, metadata)
    outputs = {
        "comparison": analytics_dir / "psf_weighting_comparison.csv",
        "validation": analytics_dir / "psf_weighting_validation.csv",
        "profiles_plot": analytics_dir / "psf_radial_profiles.png",
        "cumulative_plot": analytics_dir / "psf_cumulative_fraction.png",
        "bars_plot": analytics_dir / "psf_fraction_at_radii.png",
        "report": analytics_dir / "psf_weighting_report.md",
    }
    comparison.to_csv(outputs["comparison"], index=False)
    checks.to_csv(outputs["validation"], index=False)
    for plot, frame, key in (
        (plot_radial_profiles, profiles, "profiles_plot"),
        (plot_cumulative_fraction, curves, "cumulative_plot"),
        (plot_fraction_at_radii, table, "bars_plot"),
    ):
        plt.close(plot(frame, outputs[key]))

    fraction_columns = [
        "wavelength_angstrom",
        "radius_label",
        "fraction_beyond__grigis_2012",
        "fraction_beyond__hofmeister_2025",
        "fraction_beyond__difference",
    ]
    profile_columns = [
        "wavelength_angstrom",
        "radius_px",
        "profile_mean_weight_normalized__grigis_2012",
        "profile_mean_weight_normalized__hofmeister_2025",
        "profile_mean_weight_normalized__ratio",
    ]
    gates = checks[checks["kind"] == "gate"] if not checks.empty else checks
    failed_gates = gates[~gates["passed"]] if not gates.empty else gates
    indicative = checks[checks["kind"] == "indicative"] if not checks.empty else checks
    check_columns = ["check", "psf_source", "wavelength_angstrom", "observed", "expected", "passed"]

    sections = [
        "# PSF Weighting, Diffraction, and Scattering Comparison Report",
        "",
        f"Run: `{run_dir.name}`",
        "",
        "## Validation Gates",
        "",
        f"{len(gates) - len(failed_gates)} of {len(gates)} gates passed.",
        "",
        _markdown_table(failed_gates, check_columns) if not failed_gates.empty else "No failed gates.",
        "",
        "## Goal 2: Fraction of Light Beyond r",
        "",
        "`>=1 px` excludes only the center pixel (d > 0); other rows are d > r.",
        "",
        _markdown_table(comparison, fraction_columns),
        "",
        f"![Cumulative fraction]({outputs['cumulative_plot'].name})",
        "",
        f"![Fraction at standard radii]({outputs['bars_plot'].name})",
        "",
        "## Goal 1: Radial PSF Weight Profile",
        "",
        "Mean weight per pixel (PSF normalized to unit sum) in a +/-0.05 dex annulus "
        "around each radius; ratio is Hofmeister / Grigis.",
        "",
        _markdown_table(comparison, profile_columns),
        "",
        f"![Radial profiles]({outputs['profiles_plot'].name})",
        "",
        "## Literature Comparison (indicative)",
        "",
        _markdown_table(indicative, check_columns),
        "",
    ]

    results_path = run_dir / "out_of_frame_results.jsonl"
    if results_path.is_file() and results_path.stat().st_size > 0:
        results = load_results(results_path)
        summary = aggregate_out_of_frame(results)
        outputs["out_of_frame_summary"] = analytics_dir / "out_of_frame_summary.csv"
        outputs["out_of_frame_plot"] = analytics_dir / "out_of_frame_share.png"
        summary.to_csv(outputs["out_of_frame_summary"], index=False)
        if "out_of_frame_share" in results:
            plt.close(plot_out_of_frame(results, outputs["out_of_frame_plot"]))
        sections += [
            "## Scene-Dependent Out-of-Frame Share",
            "",
            "Each PSF is forward-convolved with its own deconvolved scene; "
            "`flare_saturated` is reported as a separate stratum.",
            "",
            _markdown_table(
                summary,
                [
                    "psf_source",
                    "wavelength_angstrom",
                    "scene_type",
                    "n_completed",
                    "n_failed",
                    *[c for c in ("mean_share", "min_share", "max_share") if c in summary],
                ],
            ),
            "",
            f"![Out-of-frame share]({outputs['out_of_frame_plot'].name})",
            "",
        ]

    outputs["report"].write_text("\n".join(sections), encoding="utf-8")
    return outputs
