from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aia_psf_deconvolution.analytics import analyze_psf_weighting_run
from aia_psf_deconvolution.analytics.psf_weighting import (
    comparison_table,
    validation_checks,
)
from aia_psf_deconvolution.config.settings import Settings
from aia_psf_deconvolution.dataio import augment_manifest
from aia_psf_deconvolution.execution.stages import (
    stage_psf_weighting,
    stage_scene_out_of_frame,
)
from aia_psf_deconvolution.physics.psf_metrics import STANDARD_RADII_PX
from aia_psf_deconvolution.results.writer import ResultWriter

SCENE_SIZE = 64


def _gaussian_psf(size: int, sigma: float) -> np.ndarray:
    coords = np.arange(size) - size // 2
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2)) + 1e-9
    return psf / psf.sum()


@pytest.fixture
def fake_psfs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "aia_psf_deconvolution.execution.stages.get_or_create_psf",
        lambda cache_dir, wavelength, force_recompute=False: _gaussian_psf(
            SCENE_SIZE, 1.0
        ),
    )
    monkeypatch.setattr(
        "aia_psf_deconvolution.execution.stages.get_or_create_hofmeister_psf",
        lambda cache_dir, wavelength, force_recompute=False: _gaussian_psf(
            2 * SCENE_SIZE, 4.0
        ),
    )


def _settings(tmp_path: Path, data_root: Path) -> Settings:
    return Settings(
        data_root=data_root,
        runs_root=tmp_path / "runs",
        scratch_root=tmp_path / "scratch",
        psf_cache_root=tmp_path / "psfs",
        deconvolved_cache_root=tmp_path / "deconvolved_scenes",
    )


def test_stage_psf_weighting_outputs(
    tmp_path: Path, synthetic_data_root: Path, fake_psfs: None
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outputs = stage_psf_weighting(
        _settings(tmp_path, synthetic_data_root), run_dir, [171, 193]
    )

    for name in outputs:
        assert (run_dir / f"{name}.csv").is_file()
    table = outputs["psf_weighting_table"]
    assert len(table) == 2 * 2 * len(STANDARD_RADII_PX)
    metadata = outputs["psf_metadata"]
    assert set(metadata["psf_source"]) == {"grigis_2012", "hofmeister_2025"}
    assert np.allclose(metadata["psf_sum"], 1.0)

    curves = outputs["psf_cumulative_fraction"]
    for (source, wavelength), group in table.groupby(
        ["psf_source", "wavelength_angstrom"]
    ):
        curve = curves[
            (curves["psf_source"] == source)
            & (curves["wavelength_angstrom"] == wavelength)
        ]
        lookup = curve.set_index("radius_px")["fraction_beyond"]
        for row in group.itertuples(index=False):
            expected = lookup.get(row.fraction_threshold_px, 0.0)
            assert row.fraction_beyond == pytest.approx(expected)

    wide = comparison_table(table)
    broader = wide[wide["radius_px"] == 10]
    assert (broader["fraction_beyond__difference"] > 0).all()

    checks = validation_checks(table, metadata)
    gates = checks[checks["kind"] == "gate"]
    assert gates[gates["check"].str.startswith("psf_")]["passed"].all()


def test_stage_scene_out_of_frame_rows_and_report(
    tmp_path: Path, synthetic_data_root: Path, fake_psfs: None
) -> None:
    settings = _settings(tmp_path, synthetic_data_root)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    stage_psf_weighting(settings, run_dir, [171])
    manifest = augment_manifest(synthetic_data_root, run_dir / "augmented_manifest.csv")
    assert len(manifest) > 0

    rng = np.random.default_rng(3)
    reference_rows = []
    for row in manifest.itertuples(index=False):
        path = tmp_path / f"{row.image_id}.npy"
        np.save(path, rng.random((SCENE_SIZE, SCENE_SIZE)) + 1.0)
        reference_rows.append({"image_id": row.image_id, "reference_path": str(path)})
    references = {
        "grigis_2012": pd.DataFrame(reference_rows),
        "hofmeister_2025": pd.DataFrame(reference_rows[1:]),
    }
    writer = ResultWriter(run_dir / "out_of_frame_results.jsonl", "run", "hash")
    stage_scene_out_of_frame(manifest, references, settings, writer)

    rows = [
        json.loads(line)
        for line in (run_dir / "out_of_frame_results.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2 * len(manifest)
    failed = [row for row in rows if row["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["psf_source"] == "hofmeister_2025"
    assert failed[0]["failure_stage"] == "out_of_frame"
    completed = [row for row in rows if row["status"] == "completed"]
    by_source = {
        source: np.mean(
            [r["out_of_frame_share"] for r in completed if r["psf_source"] == source]
        )
        for source in ("grigis_2012", "hofmeister_2025")
    }
    assert 0.0 < by_source["grigis_2012"] < by_source["hofmeister_2025"]

    outputs = analyze_psf_weighting_run(run_dir)
    report = outputs["report"].read_text()
    assert "Scene-Dependent Out-of-Frame Share" in report
    for key in ("profiles_plot", "cumulative_plot", "bars_plot", "out_of_frame_plot"):
        assert outputs[key].is_file()
