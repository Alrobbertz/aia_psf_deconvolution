from pathlib import Path

from aia_psf_deconvolution.dataio.manifests import (
    augment_manifest,
    normalize_source_path,
)


def test_windows_path_is_normalized_under_data_root(tmp_path: Path) -> None:
    raw = r"C:\workspace\data\aia_scene_experiment\quiet_sun\scene\image.fits"
    relative, resolved = normalize_source_path(raw, tmp_path)
    assert relative == "quiet_sun/scene/image.fits"
    assert resolved == (tmp_path / relative).resolve()


def test_augment_one_row(synthetic_data_root: Path, tmp_path: Path) -> None:
    output = tmp_path / "augmented.csv"
    frame = augment_manifest(synthetic_data_root, output, limit=1)
    assert len(frame) == 1
    assert output.is_file()
    assert int(frame.iloc[0]["source_integer_bytes"]) == 8 * 8 * 32 // 8
    assert frame.iloc[0]["source_loaded_dtype"] == ">f4"
