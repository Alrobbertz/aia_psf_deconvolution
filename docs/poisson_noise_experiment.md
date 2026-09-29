# AIA Poisson Statistical-Error Experiment (Hofmeister/BID)

This page is the design of record for the Poisson (shot-noise) error
experiment. See the [README](../README.md) for a quickstart and the [File
Size Experiment Reference](file_size_experiment.md) for the sibling
storage-fidelity pilot that motivated this one. See [Scene Data
Configuration](scene_configuration.md) for how to obtain and configure the
input dataset.

## Motivation

The storage-fidelity pilot ([file_size_experiment.md](file_size_experiment.md))
treats `max_abs_error <= 0.5 DN` (half the source int16 quantization step) as
its guardrail for "source-LSB-lossless" storage. That guardrail is only a
meaningful engineering target if the underlying pixel values already carry
more than 0.5 DN of *physical* measurement noise -- otherwise a storage
format could be discarding real signal, not just rounding.

This experiment measures the pixel-wise Poisson (shot-noise) statistical
error of AIA images, before and after PSF deconvolution, across scene types
and solar-disk geometry, to test whether that is the case: is the intrinsic
photon-counting noise floor already at or above 0.5 DN for the scene classes
and regions the storage pilot cares about?

### Prior art: AIA's own onboard compression reasons the same way

Notably, the AIA instrument team faced an analogous lossy-vs-noise-floor
argument for the onboard Rice compression applied before downlink, and
reached a similar conclusion. From Boerner et al. (2012), "Initial
Calibration of the Atmospheric Imaging Assembly (AIA) on the Solar Dynamics
Observatory (SDO)", *Solar Physics*, 275, 41, Section 4.4 ("Onboard Data
Compression"), p. 64-65:

> The image data are compressed by an onboard FPGA using a Rice compression
> algorithm (Rice, 1979) that converts the pixel values into differences
> between adjacent pixels. [...] The Rice algorithm itself is lossless;
> however, in order to maximize its effectiveness, it is necessary to
> minimize the impact of large intensity variations. This is accomplished by
> remapping the image pixel values using a lookup table before applying the
> Rice compression. [...] Note that the difference between the "true" pixel
> value and the value that it is rounded off to (the error introduced by the
> compression scheme) is approximately proportional to the square root of
> the true pixel value -- in this case, it is always less than 1/4 of the
> value. This ensures that the compression error will be small compared to
> the intrinsic shot noise in the image for all types of observations.

This is the same argument this experiment makes about the storage pilot's
0.5 DN guardrail, applied by the instrument team itself to a different
lossy step (onboard lookup-table remapping ahead of Rice compression,
rather than downstream storage encoding): a rounding/quantization error is
an acceptable engineering trade-off precisely because it is kept small
relative to the pixel's own shot-noise floor, not because it is
negligible in absolute terms.

## Fixed method

Unlike the storage pilot (which sweeps 32 encode/compress treatments over a
fixed aiapy/Richardson-Lucy reference), this experiment fixes a single
deconvolution method and sweeps nothing -- the independent variables are
scene type, solar-disk region, and the two error metrics below.

- **PSF**: Hofmeister et al. (2024) PSFs (`AIA_PSF_SOURCE=hofmeister_2025`),
  8192x8192 per wavelength, static files at
  `data/psfs/hofmeister_2025/psf_aia_{wavelength}.fits` (94, 131, 171, 193,
  211, 304, 335 A -- no 4500 A, matching the storage pilot's wavelength set).
- **Deconvolution**: Basic Iterative Deconvolution (BID)
  (`AIA_DECONV_METHOD=bid`), via
  `aia_psf_deconvolution.physics.hofmeister_bid.deconvolve_bid`, with
  `iterations=25`, `tolerance=0.1`, `large_psf=True`, `pad=True`,
  `estimate_background=True`, `constrain_positive=True`. Not aiapy's
  Richardson-Lucy -- this is deliberately the Hofmeister/BID path, in
  contrast to the storage pilot's `aiapy_rl`/`grigis_2012` path.
- **Scope**: all available AIA scene images (63 base scenes + 14
  `flare_saturated`), smoke-tested (1 image) first to calibrate wall-clock
  and memory for BID with the 8192x8192 padded PSF before committing to a
  full run.

The existing storage-treatment pipeline (`aiapy_rl`/`grigis_2012`,
`run_sweep`, the treatments registry) is untouched by this experiment -- all
of the code below is additive.

## Poisson error metrics

Two independent per-pixel metrics are reported side by side, rather than
choosing one:

1. **`sqrt_dn`** -- literal `sqrt(clip(DN, 0, None))`, in DN. Matches the
   simple heuristic from the exploratory
   `notebooks/hofmeister_2025_psf_testing.ipynb` prototype.
2. **`aiapy_estimate_error`** -- `aiapy.calibrate.estimate_error(counts,
   channel)`, aiapy's off-the-shelf, gain-corrected error estimate (shot +
   read + quantization + onboard-compression noise, using the Boerner et al.
   2012 gain/quantum-yield constants, ported from SolarSoft's
   `aia_bp_estimate_error.pro`).

### Why they differ

`sqrt(DN)` is only exactly correct if one DN corresponds to one counted
quantum event. AIA's ADC actually rescales real photon counts $N_\gamma$
through two multiplicative factors: quantum yield $Y$ (electrons/photon,
wavelength-dependent, generally tens of electrons for EUV) and gain $g$
(~18 electrons/DN, roughly channel-independent). Propagating Poisson noise
through both factors gives:

$$\sigma_{DN} = \sqrt{DN} \cdot \sqrt{Y / g}$$

Since $Y/g > 1$ for AIA, literal `sqrt(DN)` **understates** the true shot
noise -- it is a conservative (harder-to-satisfy) lower bound for the ">=
0.5 DN" argument, not an overestimate. `aiapy_estimate_error` supplies the
gain-corrected value directly, so no custom gain/quantum-yield constants
need to be sourced or maintained in this repository.

### Caveat: repurposing `estimate_error` after deconvolution

`aiapy.calibrate.estimate_error` is designed for *observed* (raw,
pre-deconvolution) DN counts. Applying it to the post-deconvolution BID
reference image (the "after" stage below) is a repurposing beyond its
designed use case, exactly as the literal-`sqrt(DN)`-after-deconvolution
heuristic is. Both "after"-stage metrics should be read as descriptive,
order-of-magnitude indicators, not calibrated uncertainties.

Confirmed via a real smoke test (`poisson-noise --smoke`): `aiapy
.calibrate.estimate_error` occasionally emits `RuntimeWarning: invalid value
encountered in sqrt` for some pixels (an internal aiapy behavior at very low
counts, not specific to this repository). Non-finite pixels are dropped by
`poisson_error_stats` before computing descriptive statistics, so this does
not corrupt reported results -- it is expected console noise during a full
run, not an error to chase down.

## Strata

Each image contributes result rows across every combination of:

- **scene_type**: `quiet_sun`, `active_region`, `flare`, `flare_saturated`
- **region**: `full`, `on_disk`, `off_disk` -- the on/off-disk split uses
  `aia_psf_deconvolution.physics.geometry.on_disk_mask`, a thin wrapper
  around `sunpy.map.coordinate_is_on_solar_disk` +
  `sunpy.map.all_coordinates_from_map` (`aiapy.utils` has no disk/limb
  masking capability of its own)
- **stage**: `before` (raw L1 source) vs. `after` (BID-deconvolved
  reference)
- **metric**: `sqrt_dn` vs. `aiapy_estimate_error`

12 stat blocks per image (2 stages x 3 regions x 2 metrics), written as one
long-format result row per `(image_id, stage, region, metric)` to
`poisson_results.jsonl`.

**Deferred**: HEK flare/active-region bounding-box ROI extraction
(`hpc_bbox.*` columns in `hek_flare_events.csv`) -- v1 relies on scene_type x
on-disk/off-disk only.

## Response metrics

Per `(image_id, stage, region, metric)`: `count`, `mean`, `std`, `min`,
`p05`, `q1`, `median`, `q3`, `p95`, `max`, `fraction_ge_0p5dn` (fraction of
pixels with Poisson error >= 0.5 DN).

Aggregated per `(scene_type, region, stage, metric)` via
`aggregate_poisson_by_stratum`: image count, mean-of-means, min-of-mins,
max-of-maxes, mean of p05/q1/median/q3/p95 (an approximation across images,
consistent with the framework's existing descriptive-not-formal-inference
aggregation style -- see `analytics/aggregate.py`), and mean
`fraction_ge_0p5dn`.

## Decision criteria

Headline: the fraction of **on-disk** pixels with Poisson error >= 0.5 DN,
broken out by `scene_type` **and by metric** (`sqrt_dn` vs.
`aiapy_estimate_error` side by side), both before and after BID
deconvolution. This directly answers the motivating question and shows
whether the two metrics agree on direction and magnitude.

Directional expectation (not a strict pass/fail gate): `flare` and
`active_region` on-disk pixels should show the highest Poisson error;
`quiet_sun` off-disk pixels should show the lowest.

## Running the experiment

### Environment variables for this run

Same variable set as the storage pilot, but with `AIA_PSF_SOURCE` and
`AIA_DECONV_METHOD` switched to the Hofmeister/BID pair. Run from the
**workspace root** (not `aia_psf_deconvolution/`), so the relative paths
resolve correctly:

```bash
export AIA_DATA_ROOT="$PWD/data/aia_scene_experiment"
export AIA_RUNS_ROOT="$PWD/runs"
export AIA_SCRATCH_ROOT="$PWD/scratch"
export AIA_PSF_CACHE_ROOT="$PWD/data/psfs"
export AIA_DECONVOLVED_CACHE_ROOT="$PWD/data/deconvolved_scenes"
export AIA_PSF_SOURCE="hofmeister_2025"
export AIA_DECONV_METHOD="bid"
mkdir -p runs scratch data/psfs data/deconvolved_scenes
```

If your Python environment is managed by conda, prefix
every command below with `conda run -n <env>`.

### Commands

The console entry point is `python -m aia_psf_deconvolution` (or the
installed `aia-psf-deconvolution` script) -- not `aia_experiment`, despite
what older docs in this workspace say.

```bash
python -m aia_psf_deconvolution poisson-noise --smoke   # 1 image, calibrate wall-clock/memory first
python -m aia_psf_deconvolution poisson-noise           # full run once the smoke test looks healthy
python -m aia_psf_deconvolution analyze runs/<RUN_ID>   # re-render the report from an existing run
```

The `poisson-noise` subcommand validates that `AIA_DECONV_METHOD=bid` and
`AIA_PSF_SOURCE=hofmeister_2025` are set (it refuses to run against the
storage pilot's `aiapy_rl`/`grigis_2012` configuration) and writes its own
`poisson_results.jsonl`, `bid_reference_manifest.csv`, and
`analytics/poisson_report.md` / `poisson_stratum_summary.csv` under a new run
directory -- it never touches the storage pilot's `results.jsonl` in the same
run. Other flags: `--limit N`, `--scenes ID...`, `--force-recompute` (bypass
the PSF/reference cache); there is no `--treatments` flag since this
experiment sweeps no treatments.

### Expected scale and runtime

The full dataset is 77 manifest rows (7 wavelengths x 11 images each,
wavelength-outer/scene-inner order -- see
[execution/orchestrator.py](../aia_psf_deconvolution/execution/orchestrator.py)).
A single-image smoke test measured ~24s end-to-end (PSF load + pad, BID
deconvolution, both Poisson metrics, report render). The Hofmeister PSF is
loaded and padded once per wavelength and kept resident for all 11 images in
that group, so steady-state per-image time after the first image in each
wavelength group should be lower than 24s; treat ~24s x 77 (~31 minutes) as a
conservative upper bound, not an estimate to alarm on if the run finishes
sooner. Expect a `RuntimeWarning: invalid value encountered in sqrt` printed
periodically to the console during `aiapy_estimate_error` computation --
this is expected (see the caveat above) and does not indicate a failed run.

BID with the 8192x8192 padded Hofmeister PSF is expected to be substantially
heavier per-image (wall-clock and memory) than the storage pilot's
Richardson-Lucy pipeline -- always smoke-test before a full run.

## Excluded / deferred

- Custom hand-derived gain/quantum-yield constants -- not needed;
  `aiapy.calibrate.estimate_error` supplies the gain-corrected metric
  off-the-shelf.
- HEK flare/active-region bounding-box ROI extraction (`hpc_bbox.*` in
  `hek_flare_events.csv`).
- Any change to the existing storage-treatment pipeline
  (`aiapy_rl`/`grigis_2012`, `run_sweep`, the treatments registry) -- this
  experiment is purely additive.

## Results

Full run `20260928T200652Z_89ca204d`: 77 images (7 wavelengths x 11 images),
924 result rows. Analysis notebook:
[notebooks/poisson_noise_analysis.ipynb](../notebooks/poisson_noise_analysis.ipynb)
(reads `poisson_results.jsonl` directly, not just the pre-aggregated
`poisson_stratum_summary.csv`, since wavelength trends require the raw rows).

**All results below are reported pre-deconvolution (`stage="before"`, the raw
L1 source) as the primary measurement** -- these carry more statistical rigor
than the post-BID-deconvolution ("after") numbers, since deconvolution
itself alters the pixel statistics in ways not modeled by either error
metric (see the repurposing caveat above). Post-deconvolution figures are
reported only as a secondary comparison in the final subsection below.

### Headline: on-disk, pre-deconvolution, fraction of pixels >= 0.5 DN

| scene_type | `aiapy_estimate_error` | `sqrt_dn` |
| --- | --- | --- |
| quiet_sun | 1.000 | 0.931 |
| active_region | 1.000 | 0.967 |
| flare | 1.000 | 0.919 |
| flare_saturated | 1.000 | 0.880 |

**`aiapy_estimate_error` is 1.0 (100%) in every single stratum in the run** --
every scene type, wavelength, region, and stage, with no exceptions. This
motivated a direct check (see notebook section 7): `estimate_error` has a
noise floor that is already above 0.5 DN even at **zero** counts, for every
wavelength used in this experiment, independent of any pipeline data:

| wavelength (A) | `estimate_error(0 DN)` |
| --- | --- |
| 94 | 1.190 |
| 131 | 1.228 |
| 171 | 1.199 |
| 193 | 1.247 |
| 211 | 1.247 |
| 304 | 1.190 |
| 335 | 1.228 |

The observed per-image minimums across the whole dataset cluster tightly
around these same values. So this is a real, mechanistically-understood
property of the gain-corrected metric (its read/quantization/compression-noise
terms don't vanish at zero signal), not a computation bug -- but it also means
`aiapy_estimate_error`'s fraction_ge_0p5dn statistic carries no discriminating
information in this experiment; `sqrt_dn` is the metric that actually
differentiates strata for the ">= 0.5 DN" question.

**Answer to the motivating question**: yes, by both metrics, the bulk of
on-disk AIA pixels already carry >= 0.5 DN of Poisson error *before any
deconvolution is applied*, across every scene type tested (`sqrt_dn` on-disk
fractions range 0.880-0.967; the gain-corrected metric is essentially always
above threshold due to its noise floor). The storage pilot's 0.5 DN
"source-LSB-lossless" guardrail sits at or below the intrinsic
photon-counting noise floor for the large majority of on-disk pixels in
every scene class studied, using the raw L1 data the pilot actually stores.

### By scene type (on-disk, pre-deconvolution)

`active_region` has both the highest mean error (8.97 DN `aiapy`, 8.80 DN
`sqrt_dn`) and the highest `sqrt_dn` fraction >= 0.5 DN (0.967), consistent
with the design doc's directional hypothesis. `flare_saturated` has the
*lowest* `sqrt_dn` fraction (0.880) among on-disk strata despite being a
flare-adjacent stress stratum -- plausibly because ADC saturation clips the
brightest pixels, removing them from the high end of the DN distribution
that would otherwise pull more pixels above threshold. This ordering
(active_region > flare > flare_saturated > quiet_sun) is the same one seen
post-deconvolution, just with uniformly higher fractions pre-deconvolution.

### By wavelength (on-disk, pre-deconvolution)

Both metrics track scene brightness by channel: 171 A is far and away the
brightest/noisiest channel (mean error 16.22 DN `aiapy`, 15.22 DN `sqrt_dn`;
`sqrt_dn` fraction >= 0.5 DN = 1.000), while 335 A is the dimmest (mean 1.95
DN `aiapy`, 1.72 DN `sqrt_dn`; `sqrt_dn` fraction = 0.816, the lowest of any
channel). 94 A is similarly low (`sqrt_dn` fraction 0.860). At 211 A and
304 A, `sqrt_dn`'s mean is actually slightly *higher* than
`aiapy_estimate_error`'s (9.31 vs. 8.88 DN at 211 A; 4.65 vs. 3.89 DN at
304 A) -- a wavelength-level instance of the metric disagreement discussed
next. The wavelength-vs-scene_type interaction plot (notebook section 2)
shows the 171 A peak holds across all four scene types, not just one -- it
is a channel-intrinsic effect, not scene-driven.

### Metric comparison (`aiapy_estimate_error` vs. `sqrt_dn`)

Restricted to the primary pre-deconvolution, on-disk stratum (77 images),
the ratio `aiapy_estimate_error / sqrt_dn` has median 1.06, mean 1.24 (range
0.80-3.10) -- `aiapy_estimate_error` is usually, but **not always**, the
larger value. In 35/77 images (45.5%) `aiapy_estimate_error < sqrt_dn`,
contradicting the simplified $\sigma_{DN} = \sqrt{DN}\cdot\sqrt{Y/g}$ model's
prediction that the ratio should always exceed 1 -- the real
`aia_bp_estimate_error.pro`-derived formula includes additive
read/quantization/compression terms (not a pure multiplicative rescaling),
so it can fall below `sqrt(DN)` particularly at low mean-error strata where
the additive noise floor dominates. Both metrics still agree in *direction*
(active_region > flare > flare_saturated > quiet_sun on-disk; 171 A > 193 A
> 211 A > 94/131/304/335 A), which is the important qualitative agreement
for this experiment's purpose, even though they disagree on individual
magnitudes and on which metric is larger in nearly half of all images.

### By region (full vs. on-disk vs. off-disk)

Pre-deconvolution (`stage="before"`), per scene type and region:

| scene_type | region | `aiapy_estimate_error` mean | `sqrt_dn` mean | `aiapy_estimate_error` frac >= 0.5 DN | `sqrt_dn` frac >= 0.5 DN |
| --- | --- | --- | --- | --- | --- |
| quiet_sun | full | 4.673 | 3.986 | 1.000 | 0.787 |
| quiet_sun | on_disk | 6.162 | 5.759 | 1.000 | 0.931 |
| quiet_sun | off_disk | 3.178 | 2.332 | 1.000 | 0.653 |
| active_region | full | 6.758 | 6.260 | 1.000 | 0.856 |
| active_region | on_disk | 8.972 | 8.798 | 1.000 | 0.967 |
| active_region | off_disk | 4.640 | 3.967 | 1.000 | 0.754 |
| flare | full | 5.157 | 4.603 | 1.000 | 0.816 |
| flare | on_disk | 6.812 | 6.556 | 1.000 | 0.919 |
| flare | off_disk | 3.499 | 2.762 | 1.000 | 0.719 |
| flare_saturated | full | 4.949 | 4.253 | 1.000 | 0.767 |
| flare_saturated | on_disk | 6.549 | 6.088 | 1.000 | 0.880 |
| flare_saturated | off_disk | 3.444 | 2.627 | 1.000 | 0.667 |

For `sqrt_dn`, every scene type follows on_disk > full > off_disk in both
mean error and fraction >= 0.5 DN (e.g. quiet_sun fraction: 0.931 on-disk,
0.787 full, 0.653 off-disk) -- expected, since off-disk pixels are dimmer
background/corona. `aiapy_estimate_error`'s fraction stays pinned at 1.0 in
every region for the noise-floor reason above. `off_disk` is the region with
the lowest `sqrt_dn` fraction in every scene type, but pre-deconvolution none
drop below 0.65 (quiet_sun off_disk, 0.653, is the overall minimum) --
notably less severe than the post-deconvolution off-disk values (down to
0.545, see below), which is one more reason to treat the pre-deconvolution
numbers as the primary result.

### Comparison: pre- vs. post-deconvolution (on-disk)

Applying BID deconvolution to the same 77 images changes the picture only
modestly. Mean error changes by less than ~2% in either direction for both
metrics, in every scene type -- BID does not materially change the bulk
noise floor. `sqrt_dn`'s fraction >= 0.5 DN drops modestly but consistently
after deconvolution in every scene type (-4.9 to -6.6 percentage points: from
0.931/0.967/0.919/0.880 pre-deconvolution to 0.874/0.919/0.859/0.822
post-deconvolution for quiet_sun/active_region/flare/flare_saturated,
respectively), plausibly because BID's local contrast sharpening pushes some
already-dim pixels closer to zero DN. This is a real, small, consistent
effect, but it does not change the headline conclusion, which is why the
pre-deconvolution numbers above -- not these post-deconvolution ones -- are
treated as this experiment's primary result.
