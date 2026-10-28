"""Scene-dependent share of light that a PSF scatters out of the detector frame."""

from __future__ import annotations

import numpy as np

from ..physics.hofmeister_bid import convolve_image


def out_of_frame_metrics(scene: np.ndarray, psf_large: np.ndarray) -> dict[str, object]:
    """Forward-convolve ``scene`` with a PSF of twice the frame size and measure
    how much of the total signal lands outside the frame.

    ``psf_large`` must be centered at ``shape // 2`` and span twice the scene
    shape, so every scene pixel can scatter to every other position; the
    padded linear convolution has no wrap-around into the frame.
    """
    if psf_large.shape != (2 * scene.shape[0], 2 * scene.shape[1]):
        raise ValueError(
            f"PSF shape {psf_large.shape} must be twice the scene shape {scene.shape}"
        )
    scene = np.asarray(scene, dtype=np.float64)
    total_in = float(scene.sum())
    psf_sum = float(psf_large.sum())
    convolved = convolve_image(scene, psf_large, large_psf=True)
    in_frame = float(convolved.sum())
    return {
        "scene_total_dn": total_in,
        "scene_min_dn": float(scene.min()),
        "scene_max_dn": float(scene.max()),
        "psf_sum": psf_sum,
        "in_frame_after_dn": in_frame,
        "out_of_frame_dn": total_in * psf_sum - in_frame,
        "out_of_frame_share": 1.0 - in_frame / (total_in * psf_sum),
    }
