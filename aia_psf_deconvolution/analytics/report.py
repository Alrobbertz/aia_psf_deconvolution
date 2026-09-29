from __future__ import annotations

from pathlib import Path

import pandas as pd

from .aggregate import aggregate_strata, aggregate_treatments, load_results
from .pareto import pareto_table, plot_pareto
from .poisson_aggregate import aggregate_poisson_by_stratum


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
