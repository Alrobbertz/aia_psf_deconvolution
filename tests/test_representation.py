import numpy as np
import pytest

from aia_psf_deconvolution.config.treatments import TREATMENTS
from aia_psf_deconvolution.representation.paths import RangeIntegrityError, represent


def treatment(treatment_id: str):
    return next(item for item in TREATMENTS if item.treatment_id == treatment_id)


def test_integer_path_uses_bankers_rounding() -> None:
    source = np.array([[0.5, 1.5, 2.5, 3.5]], dtype=np.float64)
    actual = represent(source, treatment("B_i16_none"))
    np.testing.assert_array_equal(actual, np.array([[0, 2, 2, 4]], dtype=np.int16))


def test_int16_overflow_is_structured_failure() -> None:
    source = np.array([[0.0, 40000.0]], dtype=np.float64)
    with pytest.raises(RangeIntegrityError) as excinfo:
        represent(source, treatment("B_i16_none"))
    assert excinfo.value.reason == "integer_overflow"


def test_blank_sentinel_is_rejected() -> None:
    source = np.array([[-32768.0, 0.0]], dtype=np.float64)
    with pytest.raises(RangeIntegrityError) as excinfo:
        represent(source, treatment("B_i16_none"))
    assert excinfo.value.reason == "blank_sentinel_collision"
