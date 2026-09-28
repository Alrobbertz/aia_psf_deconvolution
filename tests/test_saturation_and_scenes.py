from pathlib import Path

import numpy as np
import pytest

from aia_psf_deconvolution.dataio.manifests import (
    _adc_ceiling,
    _saturation_metrics,
    augment_manifest,
)


class _Header(dict):
    """Minimal stand-in for an astropy Header for saturation tests."""


def test_adc_ceiling_prefers_datamax() -> None:
    data = np.zeros((4, 4), dtype=np.float32)
    assert _adc_ceiling(data, _Header(DATAMAX=16383)) == 16383.0


def test_adc_ceiling_falls_back_to_nominal_when_missing() -> None:
    data = np.full((4, 4), 12000.0, dtype=np.float32)
    # No DATAMAX: observed max (12000) is below nominal 16383 and is used.
    assert _adc_ceiling(data, _Header()) == 12000.0


def test_adc_ceiling_clamps_to_14bit_when_no_header() -> None:
    data = np.full((4, 4), 20000.0, dtype=np.float32)
    # Without DATAMAX an over-ceiling observed max is clamped to 16383.
    assert _adc_ceiling(data, _Header()) == 16383.0


def test_saturation_counts_pixels_at_ceiling() -> None:
    data = np.zeros((10, 10), dtype=np.float32)
    data[0, :5] = 16383.0  # 5 pegged pixels
    metrics = _saturation_metrics(data, _Header(DATAMAX=16383))
    assert metrics["source_saturated_pixels"] == 5
    assert metrics["source_saturated_fraction"] == pytest.approx(5 / 100)
    assert metrics["source_is_saturated"]
    assert metrics["source_adc_ceiling_dn"] == 16383.0


def test_unsaturated_frame_reports_zero() -> None:
    data = np.full((8, 8), 500.0, dtype=np.float32)
    metrics = _saturation_metrics(data, _Header(DATAMAX=16383))
    assert metrics["source_saturated_pixels"] == 0
    assert not metrics["source_is_saturated"]


def test_augment_rejects_unknown_scene(
    synthetic_data_root: Path, tmp_path: Path
) -> None:
    output = tmp_path / "augmented.csv"
    with pytest.raises(ValueError):
        augment_manifest(synthetic_data_root, output, scene_ids=["does_not_exist"])


def test_augment_scopes_to_requested_scene(
    synthetic_data_root: Path, tmp_path: Path
) -> None:
    output = tmp_path / "augmented.csv"
    frame = augment_manifest(
        synthetic_data_root, output, scene_ids=["quiet_test_scene"]
    )
    assert set(frame["scene_id"]) == {"quiet_test_scene"}
    assert len(frame) == 2
    for column in (
        "source_saturated_pixels",
        "source_saturated_fraction",
        "source_adc_ceiling_dn",
    ):
        assert column in frame.columns
