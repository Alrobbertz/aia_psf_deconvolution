# Point Spread Function Weighting, Diffraction, and Scattering Comparison

## Background

The [total irradiance conservation](total_irradiance_conservation.md) experiment produced an
unexpected result for the "PSF weight beyond a given radius". Hofmeister, Savin & Hahn (2025)
report similar findings in general terms. Describing their Figures 14 and 15:

> Depending on the AIA channel, 23%-29% of light is scattered farther than 10 px, 11%-15% is
> scattered farther than 100 px, and 3%-10% is scattered farther than 1000 px.

Figures 14 and 15 of that paper show PSF weight beyond 1000 px, but the paper gives no table of it
and no comparison with the earlier Grigis et al. (2012) PSFs that `aiapy` implements. Our
irradiance experiment looked beyond 1000 px, out to 2048 px. That is the edge of the detector's
field of view for a source at the center of the solar disk.

The surprising result was at 171 Å. About 5% of the Hofmeister PSF weight lies beyond 2048 px.
Deconvolution with this PSF therefore puts back into the frame a significant share of light that
was scattered out of the field of view.

The shaded bands in Figures 14 and 15 show fit uncertainty, from 100 Monte Carlo runs of the PSF
fit. They do not show dependence on scene content. The range quoted above (3–10%) is for light
beyond 1000 px, not beyond 2048 px. This experiment measures the PSFs directly at 1000, 2048 and
2896 px (the frame diagonal):

- **Beyond 1000 px:** 131 Å has the most light, 10.5%.
- **Beyond 2048 px:** 131 Å has 7.4% and 171 Å has 5.0% (see [Results](#results)).

How much light actually leaves the frame depends on the scene. That is the purpose of the
optional [scene-dependent out-of-frame share](#scene-dependent-out-of-frame-share) analysis.

## Motivation

This study sets out to bring forward results that are understated in Hofmeister et al. (2025), and
to compare them with the PSFs available before that paper.

1. There is no thorough comparison between the Hofmeister (2025) and Grigis (2012) PSFs of either
   the PSF weight, or the share of light diffracted and scattered, beyond a given distance in
   pixels.
2. There is no table of this comparison. Reading the values from complex figures makes analysis
   harder.
3. Our earlier irradiance conservation experiment measured PSF weight only at 171 Å, and did not
   measure the share of diffracted light at all.

## Goals

1. **Radial PSF weight profile.** Compare the Hofmeister (2025) and Grigis (2012) PSFs for all 7
   EUV wavelengths: the azimuthally averaged weight per pixel against distance r. Report it as
   figures and as a table at the standard radii.
2. **Cumulative fraction of light beyond r.** Compare the same PSFs for all 7 wavelengths: the
   share of the total PSF weight, i.e. diffracted plus scattered light, that lands farther than r
   from the source pixel. Report it as figures and as a table at the standard radii.

**Standard radii:** ≥1 px, 10, 100, 1000, 2048 and 2896 px.

- 2048 px is half the 4096 px frame, so it is the edge of the frame for a source at the center of
  the disk.
- 2896 px is the frame diagonal (≈ 2048·√2). No pair of pixels in the frame is farther apart.

**Optional analysis:** the scene-dependent out-of-frame share. This is the share of a real
scene's signal that each PSF moves outside the 4096² frame.

**Out of scope:**

- splitting the PSF into components (core, diffraction, scattering);
- closed-form out-of-frame metrics as a deliverable;
- Monte Carlo uncertainty bands on the PSFs.

## Definitions

- **PSF center.** The array element at `shape // 2`, which must also be the PSF maximum. This is
  (2048, 2048) for the 4096² Grigis PSF and (4096, 4096) for the 8192² Hofmeister PSF. The code
  raises an error if the maximum is anywhere else.
- **Distance d.** The Euclidean distance, in detector pixels, from the center pixel to the center
  of each pixel.
- **PSF weight.** Each PSF is normalized to unit sum before metrics are computed, so a weight is
  the share of a point source's light that lands in that pixel.
- **Radial profile, P(r) (Goal 1).** The mean weight per pixel in an annulus around r.
  - In figures, annuli are logarithmic bins, 50 per decade, starting at 0.5 px.
  - In tables, the annulus is r·10^±0.05 (±0.05 dex). It is widened to at least
    [r − 0.5, r + 0.5) so that it always contains a whole ring of pixels.
  - Each row also records `completeness`: the share of the annulus area that lies inside the
    array. It is less than 1 where the circle runs past the array edge.
- **Fraction beyond r, F(>r) (Goal 2).** The share of total PSF weight with d > r (strict).
  - The **≥1 px** row is F(>0), which counts everything except the center pixel. For the
    Hofmeister PSF this is the paper's "total diffracted and scattered light".
  - The calculation sums the weight in rings indexed by ⌈d⌉, then takes a reverse cumulative sum
    from the outside in. For integer r it is exact; a unit test checks it against direct masked
    sums.
- **Out-of-frame share.** For a scene S on the 4096² frame and a PSF K spanning 8192²:
  1 − Σ_frame(S ⊛ K) / (ΣS · ΣK). Here ⊛ is a zero-padded linear convolution in which every
  pixel can scatter to every other pixel position, with no wrap-around back into the frame.

## PSF inputs

| PSF | Array | Source | Paired deconvolution |
|---|---|---|---|
| Grigis et al. (2012), `grigis_2012` | 4096², float64, peak at (2048, 2048) | `aiapy.psf.calculate_psf()` per wavelength, cached as `.npy` | Richardson–Lucy, 25 iterations, negatives clipped (`aiapy_rl`) |
| Hofmeister et al. (2025), `hofmeister_2025` | 8192², peak at (4096, 4096) | Published PSF FITS files, `psf_aia_{wavelength}.fits` | Basic Iterative Deconvolution, BID (`bid`) |

Wavelengths: 94, 131, 171, 193, 211, 304 and 335 Å.

**Grigis PSF.** `aiapy` builds each filter's PSF from two parts:

- a Gaussian core carrying 82% of the light, with σ ≈ 1/3 px per filter and ≈ 0.47 px combined;
- diffraction spikes from the entrance and focal-plane meshes, carrying the other 18%.

The two filters are combined by FFT convolution. Spikes that would fall outside the 4096² array
are cut off and the PSF is renormalized.

**Hofmeister PSF.** It spans twice the frame, so every detector pixel can scatter light to every
other detector pixel and beyond.

## Method

The experiment has two stages. Both run from the `psf-weighting` CLI subcommand.

### Stage 1: PSF metrics (Goals 1 and 2, independent of the scene)

For each wavelength and each PSF:

1. Load the PSF at its native size and check that the center is the maximum.
2. Build the distance grid d.
3. Compute the ring weights and the cumulative curve F(>k) for every integer k, out to the
   largest distance in the array.
4. Compute the log-binned radial profile with `completeness`.
5. At each standard radius, compute F(>r) and the table-annulus P(r).
6. Record metadata: sum, minimum, center-pixel share, largest distance, and the SHA-256 of the
   array.

The loop is wavelength-outer, so only one 8192² PSF is held in memory at a time.

### Stage 2: scene-dependent out-of-frame share (optional analysis)

Each PSF is paired with **its own** deconvolved scene, the best estimate of the true scene under
that PSF model:

- Grigis uses its Richardson–Lucy reference;
- Hofmeister uses its BID reference.

Both references come from the cache-aware deconvolution stages. Missing references are computed
on demand.

For each wavelength, then each PSF, then each scene:

1. If needed, embed the PSF in a zero array twice the scene size. The 4096² Grigis PSF goes into
   an 8192² array, centered.
2. Forward-convolve the scene with `convolve_image(..., large_psf=True)`.
3. Record the out-of-frame share, together with the scene's total, minimum and maximum.

Failed cells are kept as rows with `status`, `failure_stage` and `failure_reason`. The
`flare_saturated` stratum (2017-09-06 X9.3, 2024-05-14 X8.7) is reported separately from
`flare`.

### Analysis

`analyze_psf_weighting_run(run_dir)` produces:

- a wide comparison table at the standard radii, with Hofmeister − Grigis differences and
  Hofmeister / Grigis ratios;
- the validation table described below;
- an out-of-frame summary by PSF × wavelength × scene type;
- four figures;
- a markdown report.

## Validation checks

| Kind | Check | Expected |
|---|---|---|
| Gate | PSF sum | 1 ± 10⁻³ |
| Gate | PSF minimum | ≥ 0 |
| Gate | 171 Å regression, Hofmeister F(>10, >100, >1000, >2048) | 0.262, 0.121, 0.071, 0.050 (± 0.0005) |
| Gate | 171 Å regression, Grigis F(>10, >100, >1000) | 0.185, 0.030 (± 0.0005); 0.0013 (± 0.00005) |
| Indicative | Hofmeister F(>10), F(>100), F(>1000), compared with the paper's ranges | 23–29%, 11–15%, 3–10% |
| Indicative | Hofmeister F(≥1 px), compared with the paper's Fig. 14a totals | 94: 43%, 131: 52%, 171: 41%, 193: 49%, 211: 43%, 304: 37%, 335: 55% (± 3 pp) |

- **Gates** check internal consistency, and must pass.
- **Indicative checks** compare with values read off published figures. They are reported, not
  enforced.
- **Unit tests** use synthetic inputs:
  - for a delta PSF, F(>0) = 0 and the out-of-frame share is 0;
  - a hand-built PSF gives exact ring values;
  - the cumulative curve matches direct masked sums;
  - for a uniform scene, the out-of-frame share matches the closed form
    1 − Σ w(dx, dy)(N − |dx|)(N − |dy|) / N².

## Running

From the workspace root, with the dataset and PSF caches set up as described in the
[package README](../README.md):

```bash
export AIA_DATA_ROOT="$PWD/data/aia_scene_experiment" AIA_RUNS_ROOT="$PWD/runs" \
       AIA_SCRATCH_ROOT="$PWD/scratch" AIA_PSF_CACHE_ROOT="$PWD/data/psfs" \
       AIA_DECONVOLVED_CACHE_ROOT="$PWD/data/deconvolved_scenes"

python -m aia_psf_deconvolution psf-weighting --psf-only   # Goals 1 and 2 only (~30 s)
python -m aia_psf_deconvolution psf-weighting --smoke      # + out-of-frame on 1 image
python -m aia_psf_deconvolution psf-weighting              # full: 77 images x 2 PSFs
python -m aia_psf_deconvolution analyze-psf-weighting runs/<RUN_ID>
```

`AIA_PSF_SOURCE` and `AIA_DECONV_METHOD` are ignored by this subcommand, because both PSF and
deconvolution pairs are fixed. The flags `--limit`, `--scenes` and `--force-recompute` behave as
in the other subcommands.

A full run is long only when Richardson–Lucy references are missing from the cache. Each one
takes several minutes. The full run reported below, with its references already cached, took
about 20 minutes. Use [`notebooks/psf_weighting_comparison.ipynb`](../notebooks/psf_weighting_comparison.ipynb)
to review a run.

## Outputs

Under `runs/<RUN_ID>/`:

| File | Content |
|---|---|
| `run_metadata.json` | Settings for both PSF/method pairs, flags, status, timestamps |
| `psf_metadata.csv` | One row per PSF × wavelength: shape, center, sum, minimum, center share, largest distance, SHA-256 |
| `psf_radial_profiles.csv` | Goal 1, log-binned: `r_lo_px`, `r_hi_px`, `r_center_px`, `n_pixels`, `completeness`, `mean_weight`, `mean_weight_normalized` |
| `psf_cumulative_fraction.csv` | Goal 2, every integer radius: `radius_px`, `fraction_beyond` (d > radius) |
| `psf_weighting_table.csv` | Standard radii: `fraction_threshold_px`, `fraction_beyond`, `profile_*` |
| `reference_manifest.csv`, `bid_reference_manifest.csv` | The deconvolved references used by Stage 2 |
| `out_of_frame_results.jsonl` | One row per image × PSF |
| `analytics/psf_weighting_report.md` | The report |
| `analytics/psf_weighting_comparison.csv` | Wide comparison table |
| `analytics/psf_weighting_validation.csv` | Validation checks |
| `analytics/out_of_frame_summary.csv` | Out-of-frame summary |
| `analytics/*.png` | Figures: radial profiles, cumulative fraction, bars at the standard radii, out-of-frame share |

## Results

### Goals 1 and 2 (independent of the scene)

Source: full run `20261028T123753Z_e8a037f6`. These PSF metrics are identical to those of the
earlier PSF-only/smoke run `20261027T210620Z_a2d614b9`: same PSF SHA-256 hashes, and a zero
difference in every table value. All 35 gates passed. Of the 28 indicative checks, 27 passed. The
one miss is 304 Å F(>10) = 22.2%, against the paper's lower bound of 23%.

**Fraction of light beyond r, F(>r), in % (Goal 2)**

Grigis et al. (2012) / `aiapy`:

| Wavelength (Å) | ≥1 px | >10 px | >100 px | >1000 px | >2048 px | >2896 px |
|---|---|---|---|---|---|---|
| 94 | 32.4 | 13.6 | 1.9 | <0.001 | <0.001 | <0.001 |
| 131 | 34.5 | 18.3 | 1.7 | 0.050 | <0.001 | <0.001 |
| 171 | 35.8 | 18.5 | 3.0 | 0.132 | <0.001 | <0.001 |
| 193 | 36.3 | 18.6 | 4.2 | 0.189 | <0.001 | <0.001 |
| 211 | 36.7 | 18.7 | 5.0 | 0.206 | <0.001 | <0.001 |
| 304 | 37.8 | 19.0 | 7.2 | 0.417 | 0.069 | <0.001 |
| 335 | 38.0 | 19.3 | 8.6 | 0.443 | 0.088 | <0.001 |

Hofmeister et al. (2025):

| Wavelength (Å) | ≥1 px | >10 px | >100 px | >1000 px | >2048 px | >2896 px |
|---|---|---|---|---|---|---|
| 94 | 41.8 | 26.6 | 14.1 | 9.2 | 6.3 | 4.1 |
| 131 | 52.8 | 29.5 | 15.3 | 10.5 | 7.4 | 4.9 |
| 171 | 41.0 | 26.2 | 12.1 | 7.1 | 5.0 | 3.3 |
| 193 | 49.3 | 27.4 | 12.5 | 6.1 | 4.1 | 2.6 |
| 211 | 43.5 | 25.9 | 12.2 | 4.5 | 2.8 | 1.8 |
| 304 | 37.4 | 22.2 | 11.3 | 2.5 | 1.5 | 0.90 |
| 335 | 54.6 | 25.1 | 13.2 | 5.0 | 3.2 | 2.1 |

**Mean PSF weight per pixel at r, P(r), for a PSF normalized to unit sum (Goal 1)**

Grigis et al. (2012) / `aiapy`:

| Wavelength (Å) | r = 1 px | r = 10 px | r = 100 px | r = 1000 px | r = 2048 px | r = 2896 px |
|---|---|---|---|---|---|---|
| 94 | 1.58e-02 | 2.58e-04 | 3.76e-08 | 3.19e-11 | <1e-12 | <1e-12 |
| 131 | 1.75e-02 | 7.55e-06 | 3.21e-07 | 3.36e-10 | <1e-12 | <1e-12 |
| 171 | 1.73e-02 | 1.31e-05 | 9.05e-07 | 6.56e-10 | <1e-12 | <1e-12 |
| 193 | 1.68e-02 | 5.66e-06 | 1.90e-06 | 6.83e-10 | 7.32e-12 | <1e-12 |
| 211 | 1.64e-02 | 2.17e-06 | 1.03e-06 | 7.70e-10 | 4.01e-11 | <1e-12 |
| 304 | 1.44e-02 | 2.23e-05 | 7.85e-08 | 8.61e-10 | 1.30e-10 | 3.02e-10 |
| 335 | 1.39e-02 | 1.42e-05 | 1.98e-06 | 8.05e-10 | 2.08e-10 | 2.02e-10 |

Hofmeister et al. (2025):

| Wavelength (Å) | r = 1 px | r = 10 px | r = 100 px | r = 1000 px | r = 2048 px | r = 2896 px |
|---|---|---|---|---|---|---|
| 94 | 9.20e-03 | 1.63e-04 | 2.01e-07 | 4.77e-09 | 2.01e-09 | 1.36e-09 |
| 131 | 1.61e-02 | 5.57e-05 | 4.51e-07 | 4.90e-09 | 2.22e-09 | 1.57e-09 |
| 171 | 1.16e-02 | 2.91e-05 | 8.25e-07 | 3.62e-09 | 1.54e-09 | 1.06e-09 |
| 193 | 1.48e-02 | 5.41e-05 | 1.61e-06 | 3.79e-09 | 1.35e-09 | 8.80e-10 |
| 211 | 1.14e-02 | 3.49e-05 | 1.17e-06 | 3.40e-09 | 1.03e-09 | 6.22e-10 |
| 304 | 8.53e-03 | 3.77e-05 | 2.55e-07 | 2.05e-09 | 5.59e-10 | 3.20e-10 |
| 335 | 1.67e-02 | 6.97e-05 | 1.41e-06 | 3.31e-09 | 1.13e-09 | 7.02e-10 |

The radial profiles are spiky for both PSFs at r ≲ 300 px, because the diffraction spikes cross
the annuli. A value at a single radius in the table can therefore land on or between spikes. Use
the figures, or the full `psf_radial_profiles.csv`, to judge trends.

### Scene-dependent out-of-frame share

Source: full run `20261028T123753Z_e8a037f6`.

- **Coverage:** 11 scenes (3 each of `quiet_sun`, `active_region` and `flare`, and 2 of
  `flare_saturated`) × 7 wavelengths gives 77 images. With 2 PSFs that is 154 rows, and all
  completed with no failures.
- **Inputs:** each PSF is forward-convolved with its own deconvolved scene, the Richardson–Lucy
  reference for Grigis and the BID reference for Hofmeister.

**Share of the signal scattered out of the 4096² frame, in %, mean per scene type**

Grigis et al. (2012) / `aiapy`:

| Scene type | 94 Å | 131 Å | 171 Å | 193 Å | 211 Å | 304 Å | 335 Å |
|---|---|---|---|---|---|---|---|
| quiet_sun | 0.113 | 0.073 | 0.078 | 0.108 | 0.130 | 0.196 | 0.600 |
| active_region | 0.072 | 0.062 | 0.084 | 0.125 | 0.157 | 0.165 | 0.362 |
| flare | 0.047 | 0.074 | 0.086 | 0.191 | 0.141 | 0.160 | 0.344 |
| flare_saturated | 0.047 | 0.136 | 0.084 | 0.166 | 0.124 | 0.164 | 0.458 |
| all images, min–max | 0.026–0.115 | 0.022–0.251 | 0.073–0.094 | 0.088–0.379 | 0.098–0.172 | 0.121–0.227 | 0.221–0.608 |

Hofmeister et al. (2025):

| Scene type | 94 Å | 131 Å | 171 Å | 193 Å | 211 Å | 304 Å | 335 Å |
|---|---|---|---|---|---|---|---|
| quiet_sun | 6.59 | 7.44 | 4.96 | 4.20 | 3.02 | 1.61 | 3.89 |
| active_region | 6.47 | 7.36 | 4.96 | 4.22 | 3.05 | 1.55 | 3.55 |
| flare | 6.31 | 7.31 | 4.97 | 4.33 | 3.02 | 1.54 | 3.51 |
| flare_saturated | 6.31 | 7.48 | 4.96 | 4.30 | 2.97 | 1.55 | 3.67 |
| all images, min–max | 6.07–6.79 | 6.91–7.90 | 4.93–4.99 | 4.09–4.77 | 2.88–3.12 | 1.47–1.66 | 3.29–3.90 |

**Comparison per wavelength**

| | 94 Å | 131 Å | 171 Å | 193 Å | 211 Å | 304 Å | 335 Å |
|---|---|---|---|---|---|---|---|
| Hofmeister, mean over all images (%) | 6.43 | 7.39 | 4.96 | 4.26 | 3.02 | 1.57 | 3.65 |
| Hofmeister F(>2048 px) (%) | 6.3 | 7.4 | 5.0 | 4.1 | 2.8 | 1.5 | 3.2 |
| Grigis, mean over all images (%) | 0.072 | 0.082 | 0.083 | 0.146 | 0.139 | 0.172 | 0.439 |
| Median per-image ratio, Hofmeister / Grigis | 92 | 104 | 60 | 36 | 21 | 8.9 | 9.4 |

Overall:

- **Hofmeister:** 1.47–7.90% of the signal leaves the frame (median 4.2%). The lowest values are
  at 304 Å (the X2.2 2011-02-15 and X3.1 2014-10-24 flares, 1.47–1.49%). The highest are at
  131 Å, for the X8.7 2024-05-14 saturated flare (7.90%) and the X8.2 2017-09-10 limb flare
  (7.74%).
- **Grigis:** 0.022–0.61% leaves the frame (median 0.12%). The highest values are the three
  quiet-Sun scenes at 335 Å (0.59–0.61%).
- **Ratio:** per image, the Hofmeister share is 6–320 times the Grigis share (median 35).

## Interpretation

- **Near the core (≤ 10 px).** Both PSFs move roughly a third or more of the light off the source
  pixel. For Grigis, part of the ≥1 px share is the Gaussian core spilling into neighboring
  pixels, not diffraction. F(≥1 px) is therefore not "diffracted light" for Grigis.
- **Intermediate distances (100 px).** Hofmeister puts 11–15% of the light beyond 100 px.
  Grigis puts 1.7–8.6% there, and its value rises with wavelength because the diffraction spikes
  spread out more at longer wavelengths.
- **Large distances (≥ 1000 px).** This is the main difference between the models:
  - Grigis puts at most 0.44% of the light beyond 1000 px. At 94 Å, its diffraction orders end
    near 900 px.
  - Hofmeister puts 2.5–10.5% there, from its long-range scattering wings.
  - Beyond the edge of the frame for a source at the center of the disk (2048 px), Hofmeister
    still has 1.5–7.4%, against ≤ 0.09% for Grigis.

  A Grigis-based deconvolution cannot restore light that was scattered off the detector. A
  Hofmeister/BID deconvolution does restore it, and so raises the total in-frame irradiance.
- **Out-of-frame share.** This is the scene-integrated form of the same effect.
  - **Hofmeister follows the PSF.** Its share is set mainly by the PSF, and the scene matters
    little. Across all images the mean share is within about 0.5 pp of F(>2048 px) at every
    wavelength. Within a wavelength, the range across all 11 scenes is at most 1 pp (131 Å),
    and only 0.06 pp at 171 Å. Scenes whose bright emission sits closer to the frame edge
    lose a little more: the limb flare of 2017-09-10, the X8.7 flare of 2024-05-14, and
    quiet-Sun scenes, where a larger share of the light comes from the limb and off-limb
    corona.
  - **Grigis depends more on the scene, in relative terms.** Its share exceeds its own
    F(>2048 px) (≤ 0.09%), because emission at the limb is only about 450 px from the frame
    edge, where the Grigis diffraction orders still reach. The absolute values stay below
    0.61%.
  - **Practical consequence.** For full-disk studies, a Hofmeister/BID correction restores the
    roughly 1.5–8% of the true signal that the PSF scattered off the detector. How much depends
    mostly on wavelength. Under the Grigis PSF, the most that could have left the frame is
    about 0.6%, and a Grigis-based correction does not restore it.
  - **`flare_saturated` is not an outlier.** Its means match the other scene types to within
    about 0.2 pp (Hofmeister) and 0.15 pp (Grigis). The analysis keeps it as a separate
    stratum.

## Scope and caveats

- **Grigis wings.** The far wings of the Grigis PSF are float64 FFT round-off, about 10⁻²⁰ per
  pixel and about 10⁻¹³ cumulative. Table entries below 0.001% (F) or 10⁻¹² (P) are shown as
  upper bounds, and the figures are clipped above this floor.
- **Grigis array size.** The Grigis PSF is defined only on a 4096² array. Its values at
  r > 2048 px come from the array corners (`completeness` < 1). By construction they cannot show
  light reaching the far side of the frame.
- **Hofmeister array size.** The Hofmeister PSF is defined on 8192². Its F(>r) for r > 4096 px
  includes only the array corners.
- **Richardson–Lucy references.** These are computed with a circular FFT on the frame, so all
  light stays inside the frame. Using them as the "true" scene for the Grigis forward model is
  self-consistent but does not add back any light from beyond the frame.
- **Small sample per scene type.** There are 3 scenes per scene type, and 2 for
  `flare_saturated`. Differences between scene types of a few tenths of a percentage point
  (Hofmeister), or a few hundredths (Grigis), should not be over-interpreted.
- **Literature values.** The values from Hofmeister et al. (2025) were read from figures and
  rounded to the nearest percent. Treat the indicative checks accordingly.
- **Storage format.** The experiment does not alter or re-encode any L1 data, and is separate
  from the storage pilot's DN error guardrails.

## References

- Grigis, P., Su, Y., & Weber, M. (2012). *AIA PSF characterization and image deconvolution.*
  AIA team technical note; implemented in `aiapy.psf`.
- Hofmeister, S. J., Savin, D. W., & Hahn, M. (2025). Updated SDO/AIA EUV point-spread functions.
  arXiv:2410.08967. PSF data: doi:10.7910/DVN/DYT4ZL.
- Hofmeister, S. J., et al. (2024). *The Basic Iterative Deconvolution: A fast instrumental
  point-spread function deconvolution method that corrects for light that is scattered out of the
  field of view of a detector.* Solar Physics 299, 77. doi:10.1007/s11207-024-02321-y.
  Code: https://github.com/stefanhofmeister/PSF-Tools.
- Barnes, W. T., et al. (2020). *aiapy: A Python package for analyzing solar EUV image data from
  AIA.* Journal of Open Source Software 5(55), 2801.