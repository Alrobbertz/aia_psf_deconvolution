# Scene Data Configuration

This experiment framework does not ship or download AIA data itself. It reads
an existing dataset from a directory tree you provide, pointed to by the
`AIA_DATA_ROOT` environment variable.

## Expected layout

```text
<AIA_DATA_ROOT>/
|-- active_region/
|-- flare/
|-- flare_saturated/
|-- quiet_sun/
|-- scene_manifest.csv
|-- download_manifest.csv
|-- fits_verification.csv
|-- hek_window_summary.csv
|-- aia_availability.csv
`-- hek_flare_events.csv
```

Only true SDO/AIA Level 1 FITS files are supported (RICE_1-compressed int16,
`BLOCKSIZE=32`, row-strip tiling). Level 1.5 data is not supported.

## Manifest schemas

Four manifests are joined directly by the experiment:

- `scene_manifest.csv`: scene and wavelength selection. Keyed on
  `(scene_id, wavelength_angstrom)`.
- `download_manifest.csv`: source FITS location (`local_path`), compressed
  size (`bytes`), and download `status`/`errors`. Keyed on
  `(scene_id, wavelength_angstrom)`.
- `fits_verification.csv`: shape, wavelength, level, and read validation
  columns (`exists`, `shape_ok`, `wavelength_ok`, `level_metadata`,
  `level_ok`, `map_status`). Keyed on `(scene_id, wavelength_angstrom)`.
- `hek_window_summary.csv`: scene-level flare context, including a
  `query_status` column that must be non-null for every scene used. Keyed on
  `scene_id`.

Image-level manifests are inner-joined on `(scene_id, wavelength_angstrom)`;
the HEK summary is left-joined on `scene_id`. Paths recorded in `local_path`
may be absolute or Windows-style; they are normalized against
`AIA_DATA_ROOT` at augmentation time.

## Building your own dataset

Two helpers in this repository can populate a compatible dataset from public
AIA/HEK sources:

- [`notebooks/aia-scene-data-download.ipynb`](../notebooks/aia-scene-data-download.ipynb):
  interactive notebook for querying and downloading a stratified sample of
  quiet-Sun, active-region, and flare scenes.
- [`scripts/download_saturation_scenes.py`](../scripts/download_saturation_scenes.py):
  script for adding a curated `flare_saturated` stress-test stratum.

Point `AIA_DATA_ROOT` at the resulting directory before running any pipeline
stage. See the main [README](../README.md) for the full list of environment
variables.

## Pointing Docker Compose at your dataset

`AIA_DATA_ROOT` configures the path the Python code reads *inside* the
container (`/data/aia_scene_experiment` by default). It does not tell Docker
where your dataset lives on the *host*.

By default, [`docker-compose.yml`](../docker-compose.yml) bind-mounts a `data/`
directory that is a sibling of `docker-compose.yml` itself:

```yaml
volumes:
  - ${DATA_DIR:-./data}/aia_scene_experiment:/data/aia_scene_experiment:ro
  - ${DATA_DIR:-./data}/psfs:/data/psfs
  - ${DATA_DIR:-./data}/deconvolved_scenes:/data/deconvolved_scenes
```

If your dataset lives somewhere else on the host, set the `DATA_DIR`
environment variable to that directory (it must contain the
`aia_scene_experiment/`, `psfs/`, and `deconvolved_scenes/` subdirectories
described above) and every mount resolves relative to it:

```bash
export DATA_DIR=/absolute/path/to/data
docker compose run --rm experiment list-treatments
```

You can also place `DATA_DIR=/absolute/path/to/data` in a `.env` file next to
`docker-compose.yml`; Compose loads it automatically without needing an
explicit `export`. Leave `DATA_DIR` unset to use the default `./data` sibling
directory.
