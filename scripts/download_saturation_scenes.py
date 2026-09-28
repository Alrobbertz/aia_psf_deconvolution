"""Acquire saturated-flare stress-test scenes and append manifest rows.

This extends the pilot dataset with a distinct ``flare_saturated`` stratum
without disturbing the original 63-image acquisition. It downloads one nearest
full-frame Level 1 image per supported wavelength for each configured event and
appends schema-matching rows to the four manifest CSVs the experiment reads.

Run with the project virtual environment, which has sunpy/aiapy installed:

    .venv\\Scripts\\python.exe scripts/download_saturation_scenes.py

The script is idempotent: scenes already present in ``scene_manifest.csv`` are
skipped so it can be re-run safely.
"""

from __future__ import annotations

import re
from pathlib import Path

import astropy.units as u
import pandas as pd
import sunpy.map
from astropy.time import Time
from sunpy.net import Fido
from sunpy.net import attrs as a

DATA_ROOT = Path("data/aia_scene_experiment")
WAVELENGTHS = (94, 131, 171, 193, 211, 304, 335)
IMAGE_HALF_WINDOW = 7 * u.second
VALIDATION_HALF_WINDOW = 12 * u.hour

SCENES = [
    {
        "scene_id": "flare_saturated_20170906_x9p3",
        "scene_type": "flare_saturated",
        "anchor_time": "2017-09-06T12:02:00",
        "expected_event": "X9.3 flare, NOAA AR 12673",
        "selection_note": (
            "Peak-time frame; documented multi-channel ADC saturation. "
            "Stress test for int16 storage of deconvolved saturated flares."
        ),
        "goes_class": "X9.3",
        "noaa_ar": "12673",
    },
    {
        "scene_id": "flare_saturated_20240514_x8p7",
        "scene_type": "flare_saturated",
        "anchor_time": "2024-05-14T16:51:00",
        "expected_event": "X8.7 flare, NOAA AR 13664",
        "selection_note": (
            "Peak-time frame; Cycle-25 extreme flare with reported AIA "
            "saturation. Independent event from the 2017 case."
        ),
        "goes_class": "X8.7",
        "noaa_ar": "13664",
    },
]


def wavelength_as_int(value) -> int:
    if hasattr(value, "to_value"):
        raw = value.to_value(u.angstrom)
        try:
            return int(round(float(raw)))
        except (TypeError, ValueError):
            return int(round(float(raw[0])))
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    if match is None:
        raise ValueError(f"Cannot parse wavelength from {value!r}")
    return int(round(float(match.group())))


def select_nearest(result, anchor: Time):
    candidates: dict[int, list] = {w: [] for w in WAVELENGTHS}
    for block_index, block in enumerate(result):
        cols = block.colnames
        wcol = next(
            (c for c in ("Wavelength", "wavelength", "wave") if c in cols), None
        )
        scol = next(
            (c for c in ("Start Time", "start_time", "time") if c in cols), None
        )
        if wcol is None or scol is None:
            continue
        for row_index in range(len(block)):
            wavelength = wavelength_as_int(block[row_index][wcol])
            if wavelength not in candidates:
                continue
            start_time = Time(block[row_index][scol])
            offset = abs((start_time - anchor).to_value(u.second))
            candidates[wavelength].append(
                (offset, start_time.isot, block_index, row_index)
            )
    selected: dict[int, tuple] = {}
    for wavelength, rows in candidates.items():
        rows.sort(key=lambda item: (item[0], item[1]))
        if rows:
            selected[wavelength] = rows[0]
    return selected


def _load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_ROOT / name)


def _append_csv(name: str, rows: list[dict], columns: list[str]) -> None:
    existing = _load_csv(name)
    addition = pd.DataFrame(rows)
    for column in columns:
        if column not in addition.columns:
            addition[column] = ""
    addition = addition[columns]
    combined = pd.concat([existing, addition], ignore_index=True)
    combined.to_csv(DATA_ROOT / name, index=False)


def process_scene(scene: dict) -> dict[str, list[dict]]:
    anchor = Time(scene["anchor_time"])
    start = anchor - IMAGE_HALF_WINDOW
    end = anchor + IMAGE_HALF_WINDOW
    result = Fido.search(a.Time(start, end), a.Instrument.aia, a.Level(1))
    selected = select_nearest(result, anchor)
    missing = sorted(set(WAVELENGTHS) - set(selected))
    if missing:
        raise RuntimeError(f"{scene['scene_id']} missing wavelengths: {missing}")

    target_dir = DATA_ROOT / scene["scene_type"] / scene["scene_id"]
    target_dir.mkdir(parents=True, exist_ok=True)

    scene_rows, download_rows, verify_rows = [], [], []
    for wavelength in WAVELENGTHS:
        offset, start_iso, block_index, row_index = selected[wavelength]
        record = result[block_index, row_index]
        fetched = Fido.fetch(
            record, path=target_dir / "{file}", site="NSO", overwrite=False
        )
        errors = "; ".join(str(e) for e in getattr(fetched, "errors", []))
        path = Path(sorted(str(f) for f in fetched)[0])
        smap = sunpy.map.Map(path)
        actual = int(round(smap.wavelength.to_value(u.angstrom)))
        level_value = smap.meta.get("lvl_num", smap.meta.get("level"))
        scene_rows.append(
            {
                "scene_id": scene["scene_id"],
                "scene_type": scene["scene_type"],
                "anchor_time": scene["anchor_time"],
                "wavelength_angstrom": wavelength,
                "search_start": start.isot,
                "search_end": end.isot,
                "candidate_records": 1,
                "selected_records": 1,
                "selected_start_time": start_iso,
                "selected_offset_seconds": offset,
                "scene_candidate_records": sum(len(b) for b in result),
                "response_blocks": len(result),
                "search_status": "ok",
                "expected_event": scene["expected_event"],
                "selection_note": scene["selection_note"],
            }
        )
        download_rows.append(
            {
                "scene_id": scene["scene_id"],
                "scene_type": scene["scene_type"],
                "anchor_time": scene["anchor_time"],
                "wavelength_angstrom": actual,
                "local_path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "status": "downloaded",
                "errors": errors,
            }
        )
        verify_rows.append(
            {
                "scene_id": scene["scene_id"],
                "scene_type": scene["scene_type"],
                "wavelength_angstrom": actual,
                "local_path": str(path.resolve()),
                "exists": path.is_file(),
                "shape_ok": tuple(smap.data.shape) == (4096, 4096),
                "wavelength_ok": actual == wavelength,
                "level_metadata": level_value,
                "level_ok": level_value is None or str(level_value).startswith("1"),
                "map_status": "ok",
            }
        )
        print(f"  {scene['scene_id']} {wavelength}A -> {path.name}", flush=True)

    hek_row = {
        "scene_id": scene["scene_id"],
        "scene_type": scene["scene_type"],
        "anchor_time": scene["anchor_time"],
        "expected_event": scene["expected_event"],
        "validation_start": (anchor - VALIDATION_HALF_WINDOW).isot,
        "validation_end": (anchor + VALIDATION_HALF_WINDOW).isot,
        "swpc_flare_records": "",
        "c_class_records": "",
        "m_class_records": "",
        "x_class_records": 1,
        "major_near_anchor_records": 1,
        "x_near_anchor_records": 1,
        "goes_classes": scene["goes_class"],
        "noaa_active_regions_in_flare_reports": scene["noaa_ar"],
        "automatic_criterion": "Saturated flare stress-test scene (manually curated)",
        "automatic_characteristic_pass": True,
        "query_status": "manual",
    }
    return {
        "scene": scene_rows,
        "download": download_rows,
        "verify": verify_rows,
        "hek": [hek_row],
    }


def main() -> None:
    existing_scenes = set(_load_csv("scene_manifest.csv")["scene_id"].unique())
    scene_cols = list(_load_csv("scene_manifest.csv").columns)
    download_cols = list(_load_csv("download_manifest.csv").columns)
    verify_cols = list(_load_csv("fits_verification.csv").columns)
    hek_cols = list(_load_csv("hek_window_summary.csv").columns)

    for scene in SCENES:
        if scene["scene_id"] in existing_scenes:
            print(f"skip {scene['scene_id']} (already in manifest)", flush=True)
            continue
        print(f"processing {scene['scene_id']} ...", flush=True)
        out = process_scene(scene)
        _append_csv("scene_manifest.csv", out["scene"], scene_cols)
        _append_csv("download_manifest.csv", out["download"], download_cols)
        _append_csv("fits_verification.csv", out["verify"], verify_cols)
        _append_csv("hek_window_summary.csv", out["hek"], hek_cols)
        print(f"appended manifest rows for {scene['scene_id']}", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
