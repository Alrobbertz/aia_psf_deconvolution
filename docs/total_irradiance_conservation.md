# AIA Total Irradiance Conservation Under PSF Deconvolution (Grigis RL vs. Hofmeister BID)

This page reports the total-DN experiment in
[hofmeister_2025_psf_testing.ipynb](../notebooks/hofmeister_2025_psf_testing.ipynb).
It answers three questions:

1. Is the total DN in the frame conserved through deconvolution?
2. If not, why not?
3. Is the difference expected from the PSF definitions in (Hofmeister et al. (2025)) or
   from the `aiapy` / Grigis (2012) implementation?

## Summary

- **Grigis (aiapy RL): conserved.** The total changes by +0.013%. All of that
  change comes from clipping 264,793 negative input pixels. The residual is
  0.0000%.
- **Hofmeister (BID): not conserved, and not expected to be.** The output
  total is +5.49% above the input. The model attributes +5.27% to light the PSF
  scatters out of the frame, and +0.23% to an in-frame fit change caused by the
  positivity constraint.
- **Both results follow from the PSFs.** The Grigis PSF has almost no
  long-range diffuse wing, so a periodic RL convolution conserves its sum. The
  Hofmeister PSF puts 7.1% of its weight beyond 1000 px. For a uniform
  full-frame scene the out-of-frame share is 5.7%, so an increase of about 5%
  is expected from the PSF alone.

## Method

- **Input:** one 171 Å L1 image,
  `data/test_files/aia.lev1.171A_2017_09_10T20_00_09.35Z.image_lev1.fits`
  (4096 × 4096, int16, `EXPTIME` 2.000169 s).
- **Grigis (2012):** `aiapy.psf.deconvolve` with the 4096 × 4096 Grigis 171 Å
  PSF, Richardson-Lucy, 25 iterations, `clip_negative=True`.
- **Hofmeister (2025):** `deconvolve_bid` with the 8192 × 8192 Hofmeister 171 Å
  PSF, `large_psf=True` (zero-pad to 8192), 25 iterations, tolerance 0.1,
  `constrain_positive=True`.
- **Isolation run:** the same BID call with `constrain_positive=False`. This
  removes the positivity term so the out-of-frame term can be read on its own.
- **Totals:** float64 sums over the full 4096 × 4096 frame, in DN.
- **Out-of-frame share:** the BID output is blurred with the same PSF through
  `convolve_image(..., large_psf=True)`, and only the in-frame part is summed.
  The out-of-frame share is the output total minus that in-frame sum. This is a
  model estimate: the output is cropped to the frame, so the flux is not stored
  anywhere.
- **Wing fractions:** fraction of each PSF's total weight beyond a given
  radius from its center (8192 px for Hofmeister, 4096 px for Grigis).

## Results

### Total DN before and after deconvolution

| Image | Total DN | % change | Negative px |
|---|---|---|---|
| Original L1 | 2.610050e+09 | — | 264,793 |
| Grigis (2012), RL | 2.610392e+09 | +0.0131 | 0 |
| Hofmeister (2025), BID | 2.753443e+09 | +5.4939 | 0 |

### Grigis: change explained by clipping

| Quantity | Value |
|---|---|
| Flux added by clipping negative input pixels | 3.421780e+05 DN (+0.0131%) |
| Grigis change not explained by clipping | +0.0000% |

### Hofmeister: budget

| Quantity | Constrained (used) | Unconstrained |
|---|---|---|
| Total change of BID output | +5.4939% | +5.2476% |
| In-frame re-convolved change | +0.2260% | +0.0000% |
| Out-of-frame share (model) | +5.2679% | +5.2476% |
| Max \|re-convolved − original\| | 111.34 DN | 0.03 DN |
| Negative output pixels | 0 | 2,232,319 |

Cost of the positivity constraint (constrained minus unconstrained total):
+0.2463%.

### PSF weight beyond a given radius

| Radius from PSF center | Hofmeister 171 Å | Grigis 171 Å |
|---|---|---|
| > 10 px | 26.2% | 18.5% |
| > 100 px | 12.1% | 3.0% |
| > 1000 px | 7.1% | 0.13% |
| > 2048 px | 5.0% | ~0 (array edge at 2048 px) |

The Hofmeister values lie within the paper's across-channel ranges for AIA:
23–29% beyond 10 px, 11–15% beyond 100 px, and 3–10% beyond 1000 px.

### Out-of-frame share for a uniform full-frame scene

Using the same forward model on a uniform 4096 × 4096 image, the out-of-frame
share is 5.74%. The unconstrained run on this L1 image gives 5.25%. The dark
corners of the real image reduce the share relative to the uniform case.

## Interpretation

### Why the Grigis total is conserved

Richardson-Lucy with a normalized PSF and periodic FFT boundaries preserves the
sum of the image: light scattered past one edge wraps back into the frame. The
Grigis PSF includes only the PSF core and the diffraction pattern, with no
diffuse long-range wing (per the paper's
[introduction of the original PSFs](../../Docs/Hofmeister_2025/RevisedAIAPSFs.tex)).
Its weight beyond 1000 px is 0.13%, so almost nothing leaves the frame. The
only change left is the one caused by clipping negative input pixels.

### Why the Hofmeister total is not conserved

The Hofmeister PSF is normalized to 1 on the 8192 px grid, so the convolution
itself does not create flux. The change comes from the cropping. About 5% of the
weight of a scene's light lands outside the 4096 px frame, because the diffuse
wing extends across the full detector. The BID zero-pads the image so that this
light can be recovered, and the output is then cropped. The output therefore
sums to more than the input.

The paper states this directly: RL "conserves flux within the image, it cannot
retrieve these photons and thus should not be used to deconvolve AIA images with
our PSFs" (lines 741–744). The BID is the method the authors recommend.

### Is the difference expected?

Yes, for the reasons above. Three checks support this:

1. The Grigis PSF has no diffuse wing, and its total is conserved.
2. The Hofmeister 171 Å wing fractions match the paper's published ranges.
3. A uniform full-frame scene loses 5.7% of its light out of the frame under
   the same forward model. The 5.3% from the real image is in the same range.

The positivity constraint is a separate effect. It adds 0.23% through an in-frame
fit change. The unconstrained run fits the frame to 0.03 DN, so the
constraint is the cause of the in-frame residual. The paper does not discuss
positivity; this term comes from the BID implementation.

## Scope and caveats

- **One image, one wavelength.** These results are for a single 171 Å L1 image.
  The out-of-frame share depends on scene brightness near the frame edges, so
  other scenes will differ.
- **No independent truth.** The out-of-frame term is a model estimate. The
  paper calls the medium- to long-distance diffuse scatter the least certain
  part of the fit, with calibration artifacts of about 0.2 DN in lunar-occulted
  pixels. The out-of-frame term inherits that uncertainty.
- **Negative input pixels.** The 264,793 negative pixels (1.58% of the frame)
  are off-disk and within the read-noise scatter near zero. The clipping adds
  0.013% of total flux.
- **In-frame residual not localized.** The constrained fit leaves a maximum
  per-pixel mismatch of 111 DN. Its location has not been identified.
- **Shift invariance.** The PSF is assumed constant across the frame, as the
  paper also assumes.
