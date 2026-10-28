from __future__ import annotations

import numpy as np
import pytest

from aia_psf_deconvolution.verification.out_of_frame import out_of_frame_metrics


def test_delta_psf_keeps_all_light_in_frame() -> None:
    size = 16
    psf = np.zeros((2 * size, 2 * size))
    psf[size, size] = 1.0
    scene = np.random.default_rng(1).random((size, size)) + 1.0
    metrics = out_of_frame_metrics(scene, psf)
    assert metrics["out_of_frame_share"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["scene_total_dn"] == pytest.approx(scene.sum())


def test_uniform_scene_matches_closed_form() -> None:
    size = 16
    rng = np.random.default_rng(2)
    psf = rng.random((2 * size, 2 * size))
    psf[size, size] = psf.sum()
    psf /= psf.sum()
    scene = np.ones((size, size))

    offsets = np.arange(2 * size) - size
    overlap = np.clip(size - np.abs(offsets), 0, None)
    expected_in_frame = float((psf * overlap[:, None] * overlap[None, :]).sum())
    expected_share = 1.0 - expected_in_frame / size**2

    metrics = out_of_frame_metrics(scene, psf)
    assert metrics["out_of_frame_share"] == pytest.approx(expected_share, rel=1e-9)
    assert 0.0 < metrics["out_of_frame_share"] < 1.0


def test_psf_must_be_twice_the_scene() -> None:
    with pytest.raises(ValueError, match="twice the scene shape"):
        out_of_frame_metrics(np.ones((8, 8)), np.ones((8, 8)))
