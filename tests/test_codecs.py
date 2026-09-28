from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from aia_psf_deconvolution.config.treatments import TREATMENTS
from aia_psf_deconvolution.dataio.fitsio import read_image_header
from aia_psf_deconvolution.representation.encoder import encode
from aia_psf_deconvolution.verification.fidelity import fidelity_metrics
from aia_psf_deconvolution.verification.interop import interoperability_checks


@pytest.fixture
def reference_array() -> np.ndarray:
    y, x = np.mgrid[:64, :64]
    return (1000 + 50 * np.sin(x / 5) + 20 * np.cos(y / 7)).astype(np.float64)


@pytest.fixture
def source_header(synthetic_fits_path: Path) -> fits.Header:
    return read_image_header(synthetic_fits_path)


@pytest.mark.parametrize("treatment", TREATMENTS, ids=lambda t: t.treatment_id)
def test_all_treatments_round_trip(
    treatment, reference_array: np.ndarray, source_header: fits.Header, tmp_path: Path
) -> None:
    path = tmp_path / f"{treatment.treatment_id}.fits"
    represented, _, _ = encode(
        path,
        reference_array,
        source_header,
        "a" * 64,
        "b" * 64,
        treatment,
        25,
        True,
    )
    metrics, _ = fidelity_metrics(path, reference_array, represented, treatment)
    assert metrics["max_abs_error"] <= 0.5
    if treatment.is_byte_exact_control:
        assert metrics["byte_exact_pass"]
    if treatment.row_strip:
        with fits.open(path, disable_image_compression=True) as hdul:
            assert hdul[1].header["ZTILE1"] == 64
            assert hdul[1].header["ZTILE2"] == 1


@pytest.mark.parametrize("treatment_id", ("A_f64_rice_q16_dither1", "B_i16_rice_bs32"))
def test_representative_interoperability(
    treatment_id: str,
    reference_array: np.ndarray,
    source_header: fits.Header,
    tmp_path: Path,
) -> None:
    treatment = next(item for item in TREATMENTS if item.treatment_id == treatment_id)
    path = tmp_path / f"{treatment_id}.fits"
    encode(
        path,
        reference_array,
        source_header,
        "a" * 64,
        "b" * 64,
        treatment,
        25,
        True,
    )
    checks = interoperability_checks(
        path, source_header, treatment, reference_array.shape
    )
    assert checks["interop_pass"], checks
