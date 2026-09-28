import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aia_psf_deconvolution.analytics.aggregate import geometric_mean
from aia_psf_deconvolution.analytics.report import analyze_run
from aia_psf_deconvolution.results.writer import ResultWriter


def test_result_writer_is_one_json_object_per_line(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    writer = ResultWriter(path, "run-1", "hash-1")
    writer.append({"image_id": "image", "value": np.float64(2.5)})
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run-1"
    assert rows[0]["value"] == 2.5


def test_geometric_mean_zero_semantics() -> None:
    assert geometric_mean(pd.Series([1.0, 4.0])) == pytest.approx(2.0)
    assert geometric_mean(pd.Series([0.0, 4.0])) == 0.0


def test_analysis_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path
    writer = ResultWriter(run_dir / "results.jsonl", "run-1", "hash-1")
    for treatment_id, passed, mae in (
        ("A_f64_none", True, 0.0),
        ("B_i32_rice_bs32", True, 0.2),
        ("B_i16_rice_bs32", False, 0.2),
    ):
        for index in range(3):
            writer.append(
                {
                    "image_id": f"image-{index}",
                    "scene_id": f"scene-{index}",
                    "scene_type": "quiet_sun",
                    "wavelength_angstrom": 171,
                    "treatment_id": treatment_id,
                    "path": treatment_id[0],
                    "representation_dtype": "float64",
                    "codec": "RICE_1",
                    "status": "completed",
                    "overall_pass": passed,
                    "mae": mae,
                    "max_abs_error": 0.4 if passed else 0.6,
                    "ratio_source_fits": 1.5 + index / 10,
                    "ratio_reference_f64": 0.2,
                    "ratio_source_integer": 0.8,
                }
            )
    outputs = analyze_run(run_dir)
    for path in outputs.values():
        assert path.is_file(), path
    summary = pd.read_csv(outputs["summary"])
    assert len(summary) == 3
