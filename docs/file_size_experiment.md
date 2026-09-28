# File Size Experiment Reference

This page is the detailed reference for the storage-fidelity pilot. See the
[README](../README.md) for a quickstart. See
[Scene Data Configuration](scene_configuration.md) for how to obtain and
configure the input dataset.

The experiment is a storage-fidelity pilot. It measures file size, numerical
error relative to a canonical float64 deconvolution, FITS integrity, metadata,
WCS behavior, and SunPy interoperability. It does not evaluate the physical
accuracy of the PSF model or Richardson-Lucy deconvolution.

## Experiment Matrix

Every selected image receives all 32 treatments:

- 24 floating-point treatments:
  - float32 and float64 representations
  - uncompressed, `GZIP_1`, and `GZIP_2` byte-exact controls
  - `RICE_1` at quantization levels 8, 16, and 32
  - no dither, subtractive dither 1, and subtractive dither 2
- 8 integer treatments:
  - int16 and int32 representations at `BSCALE=1` DN
  - uncompressed, `GZIP_1`, `GZIP_2`, and `RICE_1`

The full run contains 63 images x 32 treatments = 2,016 result rows.

List the canonical treatment IDs with:

```bash
docker compose run --rm experiment list-treatments
```

## Prerequisites

- Docker Desktop with Docker Compose v2
- At least 3 GiB of memory available to the container
- Approximately 10 GiB of free disk space for a full run
- Input FITS files and manifests under `AIA_DATA_ROOT` (see
  [Scene Data Configuration](scene_configuration.md))

The Docker image is not pinned to a processor architecture. Docker therefore
builds a native `amd64` image on x86-64 Windows/Linux and a native `arm64`
image on Apple Silicon. Avoid forcing `linux/arm64` on an x86-64 host because
emulation makes the numerical workload dramatically slower.

## Build the Docker Image

From the repository root:

```bash
docker compose build
```

To refresh the Python base image:

```bash
docker compose build --pull
```

Confirm that the image architecture matches the Docker host when diagnosing
performance problems:

```bash
docker info --format '{{.Architecture}}'
docker image inspect sdac-experiment --format '{{.Architecture}}'
```

The exact image name can vary if the Compose project name is overridden. Use
`docker compose images` to find it.

## Run Automated Tests

The test suite uses `pytest` and relies entirely on synthetic, in-memory
fixtures (see `tests/conftest.py`) -- no real AIA data is required to run it:

```bash
pip install -e ".[dev]"
pytest
```

The tests cover the 32-treatment registry, all codec round trips on synthetic
arrays, integer overflow and BLANK handling, manifest augmentation, JSONL
writing, result aggregation, Pareto output, WCS, checksums, and SunPy reopen.

An optional full-frame API check is available in the local Python environment:

```bash
python scripts/api_spike.py
```

It verifies the 4096 x 4096 float64 PSF, full-frame lossless GZIP behavior,
dither metadata, and FITS row-strip orientation.

## Run a Smoke Test

The smoke test processes one real 4096 x 4096 image through all 32 treatments,
including PSF calculation, deconvolution, encoding, verification, and analysis:

```bash
mkdir -p runs scratch data/psfs data/deconvolved_scenes
docker compose run --rm experiment all --smoke
```

Expected output:

- One new directory under `runs/`
- 32 unique rows in `results.jsonl`
- `run_metadata.json` with `"status": "complete"`
- No files left under `scratch/`
- CSV, Markdown, and PNG analysis artifacts under `analytics/`

## Run the Full Experiment

Build and start the full experiment in the background:

```bash
mkdir -p runs scratch data/psfs data/deconvolved_scenes
docker compose up --build --detach
```

Follow progress without tying the experiment lifecycle to the terminal:

```bash
docker compose logs --follow
```

Pressing Ctrl+C while following logs stops only the log viewer. Check the
container separately:

```bash
docker compose ps
```

The included helper performs the same build, detached start, and log follow:

```bash
./run.sh
```

The full run executes serially and is expected to take hours on a workstation.
It creates 63 float64 references and 2,016 treatment result rows. Encoded FITS
files are deleted after verification, so they do not accumulate on disk.

To stop an experiment intentionally:

```bash
docker compose stop
```

There is no resume behavior. An interrupted run remains on disk for diagnosis,
but the next execution creates a new run ID and starts from scratch.

## Useful Partial Runs

Process the first three manifest rows through all treatments:

```bash
docker compose run --rm experiment all --limit 3
```

Run selected treatments only:

```bash
docker compose run --rm experiment all --limit 3 \
  --treatments B_i16_rice_bs32 B_i32_rice_bs32 A_f64_rice_q16_nodither
```

Available pipeline endpoints are:

- `augment`: create and inspect the joined input manifest
- `deconvolve`: augment inputs and create PSF/reference caches
- `sweep`: augment, deconvolve, and run treatments without analytics
- `all`: run every stage, including analytics
- `analyze RUN_DIR`: regenerate analytics for an existing run

Regenerate analysis inside Docker with:

```bash
docker compose run --rm experiment analyze /runs/RUN_ID
```

## Output Locations

Host directories are mounted as follows:

| Host path | Container path | Purpose |
|---|---|---|
| `./data/aia_scene_experiment` | `/data/aia_scene_experiment` | Read-only source data and manifests |
| `./data/psfs` | `/data/psfs` | PSF cache (run-independent) |
| `./data/deconvolved_scenes` | `/data/deconvolved_scenes` | Reference float64 cache (run-independent) |
| `./runs` | `/runs` | Run metadata, results, and analysis |
| `./scratch` | `/scratch` | Ephemeral encoded FITS files |

Every execution creates a unique UTC timestamp and random-suffix run ID:

```text
runs/20260730T190743Z_05bbbd7e/
|-- run_metadata.json
|-- augmented_manifest.csv
|-- reference_manifest.csv
|-- results.jsonl
`-- analytics/
    |-- treatment_summary.csv
    |-- stratum_summary.csv
    |-- pareto.csv
    |-- pareto.png
    `-- report.md
```

### Run Metadata

`run_metadata.json` records:

- Run ID and configuration hash
- Command and selected treatments
- Effective paths and deconvolution settings
- Start and finish timestamps
- Final status (`running`, `complete`, or `failed`)
- A top-level error when a pipeline stage fails

### Augmented Manifest

`augmented_manifest.csv` freezes the selected inputs and records each source
file's normalized path, SHA-256, compressed bytes, physical integer payload,
loaded dtype, FITS scaling, dimensions, validation state, scene context, and
package versions.

### Reference Manifest and Cache

`reference_manifest.csv` records one canonical float64 deconvolution per image:

- Reference path (under `data/deconvolved_scenes/<method>/<psf_source>/`), SHA-256, dtype, and bytes
- Minimum, maximum, and nonfinite count
- Coarse deconvolution elapsed time
- `reference_cached` flag: `true` if loaded from cache, `false` if newly computed

PSFs are cached per wavelength at `data/psfs/<psf_source>/psf_{wavelength}.npy`.
Deconvolved references are cached at `data/deconvolved_scenes/<method>/<psf_source>/{image_id}.npy`.
These caches are run-independent and persist across multiple executions. For the full experiment,
expect approximately 8 GiB of references and 0.9 GiB of PSFs (Grigis 2012 source).

## Result Format

`results.jsonl` is the primary analysis input. It contains one JSON object per
image-treatment pair. JSONL permits streaming writes and preserves completed
cell results if a later cell fails.

A successful full run should contain exactly 2,016 lines:

```bash
wc -l runs/RUN_ID/results.jsonl
```

Important field groups are:

### Identity and Reproduction

- `run_id`, `config_hash`
- `image_id`, `scene_id`, `scene_type`, `wavelength_angstrom`
- `treatment_id`, `path`, `representation_dtype`, `codec`
- `quantize_level`, `quantize_method`, `dither_seed`

### Status and Guardrails

- `status`: `completed`, `failed`, or `error`
- `overall_pass`
- `failure_stage`, `failure_reason`, and `error` when applicable
- `source_lsb_pass`, `byte_exact_pass`, `range_integrity_pass`
- `interop_pass`, `checksum_pass`, `wcs_pass`, `unit_pass`
- `shape_pass`, `submap_pass`, and `provenance_pass`

Failure rows are retained. Fields that could not be measured before a failure
may be absent or null.

### Pixel Error in DN

- `mae`: mean absolute error against the float64 reference
- `max_abs_error`: maximum absolute error
- `median_abs_error`, `rmse`
- `p99_abs_error`, `p999_abs_error`
- `mean_signed_error`
- `reference_nonfinite`, `decoded_nonfinite`, `new_nonfinite`

The source-LSB guardrail is `max_abs_error <= 0.5 DN`. One DN is one integer
count in the Level 1 source representation. This evaluates storage error only,
not observational, calibration, PSF-model, or deconvolution uncertainty.

### Size Metrics

- `encoded_bytes`, `encoded_mib`, `bits_per_pixel`
- `ratio_source_fits`: encoded whole FITS / compressed Level 1 whole FITS
- `ratio_reference_f64`: encoded whole FITS / float64 pixel payload
- `ratio_source_integer`: encoded whole FITS / physical source integer payload

The primary size target is a geometric mean `ratio_source_fits` near or below
2.0, subject to all fidelity and interoperability guardrails.

### Operational Telemetry

- `write_seconds`
- `read_seconds`

These timings are coarse telemetry. The pilot does not rank treatments by
workstation timing; dedicated performance tests are deferred to NASA NCCS.

## Analysis Artifacts

The `all` command automatically produces:

- `treatment_summary.csv`: one row per treatment, including geometric-mean MAE,
  zero-MAE count, worst maximum error, pass counts, and size-ratio summaries
- `stratum_summary.csv`: treatment summaries by wavelength and scene type
- `pareto.csv`: eligible size-error frontier records for all three baselines
- `pareto.png`: visual size versus GM-MAE comparison
- `report.md`: byte-exact controls, Pareto candidates, and rejected treatments

GM-MAE is calculated by first computing MAE for each image and then taking the
ordinary geometric mean across images. The zero-MAE image count is reported
because any zero component makes an ordinary geometric mean zero.

## Environment Configuration

Compose supplies these environment variables:

| Variable | Container default | Meaning |
|---|---|---|
| `AIA_DATA_ROOT` | `/data/aia_scene_experiment` | Input dataset root |
| `AIA_RUNS_ROOT` | `/runs` | Persistent run output root |
| `AIA_SCRATCH_ROOT` | `/scratch` | Ephemeral FITS workspace |
| `AIA_PSF_CACHE_ROOT` | `/data/psfs` | PSF cache root |
| `AIA_DECONVOLVED_CACHE_ROOT` | `/data/deconvolved_scenes` | Deconvolved reference cache root |
| `AIA_PSF_SOURCE` | `grigis_2012` | PSF source used to compute references |
| `AIA_DECONV_METHOD` | `aiapy_rl` | Deconvolution method used to compute references |
| `AIA_RL_ITERATIONS` | `25` | Richardson-Lucy iterations |
| `AIA_CLIP_NEGATIVE` | `true` | Clip negative RL output values |
| `AIA_RANDOM_SEED` | `20260730` | Run-level seed recorded in configuration |

Changing settings or treatment selection changes the recorded configuration
hash. Per-image FITS dither seeds are derived deterministically from the source
SHA-256. For the planned pilot, retain the defaults.

Docker Compose additionally reads a host-side `DATA_DIR` variable (not passed
into the container) that controls where the `data/aia_scene_experiment`,
`data/psfs`, and `data/deconvolved_scenes` bind mounts are read from on the
host. See [Scene Data Configuration](scene_configuration.md#pointing-docker-compose-at-your-dataset)
for details.

## Reproduction Checklist

Before trusting a full run, verify:

1. `pytest` passes.
2. `docker compose build` completes using the host-native architecture.
3. `docker compose run --rm experiment all --smoke` produces 32 result rows.
4. The smoke run has `status: complete` and an empty `scratch/` directory.
5. The full run produces 63 reference rows and 2,016 unique result rows.
6. Every intended treatment has 63 image rows in `treatment_summary.csv`.
7. `run_metadata.json`, source hashes, reference hashes, package versions, and
   the configuration hash are retained with the analysis outputs.

## Troubleshooting

### Docker appears much slower than local Python

Check host and image architectures. They should match. ARM emulation on an
x86-64 Windows host can make PSF and FFT-heavy work tens of times slower.

### JAX reports that a TPU backend is unavailable

This is expected on a workstation. JAX probes optional accelerators and falls
back to CPU. The message does not indicate an experiment failure.

### Ctrl+C stopped the experiment

`docker compose up` runs attached and Ctrl+C stops its containers. Use
`docker compose up --detach`, then inspect with `docker compose logs --follow`.

### The container cannot write to `runs/` or `scratch/`

Create both directories on the host before starting Compose and ensure Docker
Desktop can write to the repository directory:

```bash
mkdir -p runs scratch
```

### A run was interrupted

Do not append to or reuse its output directory. Start a new execution. The new
run receives a new ID and recomputes references and treatment results.
