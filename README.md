# AIA Point Spread Function (PSF) Image Deconvolution Experiments

This repository contains code for experimentation with the Point Spread Function (PSF)
for the Solar Dynamics Observatory (SDO) Atmospheric Imaging Assembly (AIA) instrument.

## Prerequisites

- Docker Desktop with Docker Compose v2 (recommended), or Python >= 3.12 for
  a local install
- An AIA scene dataset -- see [Scene Data Configuration](docs/scene_configuration.md)

## Quickstart

Install locally in editable mode:

```bash
pip install -e ".[dev]"
pytest
```

Or build and run the experiment container:

```bash
docker compose build
mkdir -p runs scratch data/psfs data/deconvolved_scenes
docker compose run --rm experiment list-treatments
docker compose run --rm experiment all --smoke
```

## Documentation

- [Scene Data Configuration](docs/scene_configuration.md) -- how to obtain and
  configure the input AIA dataset
- [File Size Experiment Reference](docs/file_size_experiment.md) -- experiment
  matrix, running the full pipeline, output/result formats, environment
  configuration, and troubleshooting
- [Poisson Noise Experiment Reference](docs/poisson_noise_experiment.md) --
  pixel-wise Poisson (shot-noise) error before/after Hofmeister/BID
  deconvolution, via the `poisson-noise` subcommand
- [PSF Weighting, Diffraction, and Scattering Comparison](docs/psf_weighting_comparison.md) --
  Grigis (aiapy) vs. Hofmeister PSF radial profile, fraction of light beyond r,
  and scene-dependent out-of-frame share, via the `psf-weighting` subcommand

