from __future__ import annotations

from pathlib import Path

import pandas as pd

from .aggregate import aggregate_strata, aggregate_treatments, load_results
from .pareto import pareto_table, plot_pareto


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
