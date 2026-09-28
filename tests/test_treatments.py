from aia_psf_deconvolution.config.treatments import TREATMENTS, select_treatments


def test_registry_counts() -> None:
    assert len(TREATMENTS) == 32
    assert sum(item.path == "A" for item in TREATMENTS) == 24
    assert sum(item.path == "B" for item in TREATMENTS) == 8
    assert sum(item.is_byte_exact_control for item in TREATMENTS) == 6
    assert sum(item.is_quantized for item in TREATMENTS) == 18
    assert len({item.treatment_id for item in TREATMENTS}) == 32


def test_row_strip_uses_numpy_axis_order() -> None:
    rice = next(item for item in TREATMENTS if item.treatment_id == "B_i16_rice_bs32")
    assert rice.tile_shape((64, 128)) == (1, 128)


def test_selection_preserves_requested_order() -> None:
    selected = select_treatments(["B_i32_none", "A_f64_none"])
    assert [item.treatment_id for item in selected] == ["B_i32_none", "A_f64_none"]
