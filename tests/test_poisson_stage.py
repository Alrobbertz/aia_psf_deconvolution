from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aia_psf_deconvolution.config.settings import Settings
from aia_psf_deconvolution.dataio import augment_manifest
from aia_psf_deconvolution.execution.stages import (
    stage_deconvolve_bid,
    stage_poisson_noise,
)
from aia_psf_deconvolution.results.writer import ResultWriter

N_STAGES = 2
N_REGIONS = 3
N_METRICS = 2


def _fake_hofmeister_psf(size: int = 16) -> np.ndarray:
    """A tiny normalized kernel standing in for the real 8192x8192 PSF."""
    coords = np.arange(size) - size // 2
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    psf = np.exp(-(xx**2 + yy**2) / (2 * 2.0**2))
    return (psf / psf.sum()).astype(np.float64)


def _settings(tmp_path: Path, data_root: Path) -> Settings:
    return Settings(
        data_root=data_root,
        runs_root=tmp_path / "runs",
        scratch_root=tmp_path / "scratch",
        psf_cache_root=tmp_path / "psfs",
        deconvolved_cache_root=tmp_path / "deconvolved_scenes",
        psf_source="hofmeister_2025",
        deconvolution_method="bid",
    )


def test_stage_poisson_noise_row_count(
    tmp_path: Path, synthetic_data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "aia_psf_deconvolution.execution.stages.get_or_create_hofmeister_psf",
        lambda cache_dir, wavelength, force_recompute=False: _fake_hofmeister_psf(),
    )

    settings = _settings(tmp_path, synthetic_data_root)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    manifest = augment_manifest(synthetic_data_root, run_dir / "augmented_manifest.csv")
    n_images = len(manifest)
    assert n_images > 0

    bid_references = stage_deconvolve_bid(
        manifest, settings, run_dir, force_recompute=True
    )
    assert len(bid_references) == n_images

    writer = ResultWriter(run_dir / "poisson_results.jsonl", "run-1", "hash-1")
    stage_poisson_noise(manifest, bid_references, run_dir, writer)

    rows = [
        json.loads(line)
        for line in (run_dir / "poisson_results.jsonl").read_text().splitlines()
    ]
    assert len(rows) == n_images * N_STAGES * N_REGIONS * N_METRICS

    for row in rows:
        assert row["stage"] in {"before", "after"}
        assert row["region"] in {"full", "on_disk", "off_disk"}
        assert row["metric"] in {"sqrt_dn", "aiapy_estimate_error"}
        assert "fraction_ge_0p5dn" in row
        assert row["count"] >= 0
