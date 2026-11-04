# orient: orientation from the image

SpecVersion `orient/1`. The code is `baillarger.orient` (the sampler) and
`baillarger.orient_noise` (the noise file). This is a sampler contract
(`docs/version-2-plan.md`, §1 and §3c step 3): the sampler computes its
output on demand, for one superblock and its halo, from published files
only. Nothing per voxel is stored. The one stored output is the noise file,
a small JSON file for each acquisition.

The specs in `docs/formats/` stay at version 1 until a release, and are
edited in place (plan, §8). This version gives the local cortex level and
the noise file. The structure tensor, the data normal, its sign and its
coherence are added to this document by an edit in place (phase 1.2, the
second PR).

## Words

| Word | Meaning |
|---|---|
| echo | One magnitude image of the multi-echo FLASH, I_e, e = 1 to E, in echo order. |
| log echo | log max(I_e, `FLOOR`), `FLOOR` = 1.0. |
| cortex | The voxels with the cortex label of the hemisphere (`label-set/1`): 42 (R) or 3 (L). |
| local cortex level | The level of a log echo in the cortex at a point, with the laminae averaged out (below). |
| detrended log echo | The log echo minus its local cortex level, in cortex. |
| block, superblock, level | As in `ray-sampler/1`: blocks of `cells` voxels a side from voxel (0, 0, 0); a superblock is 2^m × 2^m × 2^m blocks; m is the level. |
| taps | The weights of a Gaussian of σ voxels at the integer offsets −r to r, r = ceil(`TRUNCATE` × σ), `TRUNCATE` = 3.0, divided by their sum. |
| normalized convolution | G(x m) / G(m) over a mask m, with G the separable convolution with the taps on each axis, and zero outside the grid. NaN where G(m) < `MIN_WEIGHT` (1e-06). |

## Data inputs

| Input | Spec | What the sampler reads |
|---|---|---|
| The echoes | the scan (NIfTI magnitude images, uncompressed or gzipped) | The intensities |
| The corrected segmentation | `wm-dseg/1` (`_desc-wmtopo_dseg.nii.gz`) | The labels, the affine, and the sidecar (`FilledLabels`, `Hemisphere`) |
| The noise file | this document, "The noise file" | C's parts (the tensor only) |

The echoes and the segmentation must have the same grid and the same affine
(within 1e-5). The sampler refuses others. It reads them through the chunked
reader (`baillarger.blocks`), and never reads a volume whole.

`wm-dseg/1` has no block size, so the block size is a parameter, `cells`,
with the default 64 (`baillarger.mesh.BLOCK`).

## The local cortex level

```python
sampler = OrientSampler(echoes, dseg, cells=64)
r = sampler.level(superblock, level, halo_mm=0.0, sigma_mm=SIGMA, device="numpy")
```

The level of echo e at a cortex voxel is the normalized convolution of the
log echo over the cortex mask, with the taps of σ_a = sigma_mm / vs_a voxels
on axis a (vs_a: the voxel size on axis a). It is NaN outside the cortex.

It is spatial, not a trend along depth, so it needs no columns. σ sets two
goals that pull against each other: the laminae must not leak into the
level, and the level must follow real changes of the cortex's level, above
all at areal borders (plan, §3c step 3). `SIGMA` = 0.64 mm is the default
until phase 1.2 chooses σ by its views (Part D).

### Parameters

| Parameter | Type | Default | Rule |
|---|---|---|---|
| `superblock` | tuple of 3 integers | (required) | (I, J, K), each ≥ 0 |
| `level` | integer | (required) | m ≥ 0 |
| `halo_mm` | number | 0.0 | 0 ≤ halo_mm, finite, in mm |
| `sigma_mm` | number | `SIGMA` (0.64) | σ > 0, finite, in mm |
| `device` | string or `Device` | `numpy` | `numpy`, `cpu`, `cuda` or `mps` (`baillarger.backend`) |

### `Level`

| Field | Type | Meaning |
|---|---|---|
| `level` | float32 array (E, box shape) | The local cortex level of each log echo. NaN outside the cortex |
| `box` | tuple of 3 slices | The output box, in grid points: `blocks.superblock_box` with halo_mm |
| `sigma_mm` | number | σ of the call, in mm |

The read box is the output box plus the taps' radius on each axis, clipped
to the grid.

### Determinism and devices

- **numpy** computes in float64 and gives float32. Each value uses only the
  voxels within the taps' radius, in the same order, whatever box holds it.
  So a voxel has the same value, bit for bit, from every superblock, and the
  same value as a call on the whole grid.
- **PyTorch** (`cpu`, `cuda`, `mps`) computes in float32, with
  `conv1d` for the separable convolutions, no float64 and no atomic
  additions. It gives the numpy value within float32 rounding: the tests
  require a relative difference below 4e-6 of the log value.

## The noise covariance C

C is the covariance across echoes of one component of the gradients of the
detrended log echoes (per voxel step, at the derivative scale
`SD` = 1 voxel). It holds everything in the gradients that is not laminar signal.
The tensor whitens by it: J = Σ Gᵀ C⁻¹ G (plan, §3c step 3).

### The model

At a voxel, with I_e = exp(the local cortex level of echo e):

    C = g · diag(σ_e / I_e) · R · diag(σ_e / I_e) + C_rest

| Part | Meaning |
|---|---|
| g | `GradientGain`: the variance of one component of the gradient per unit variance of white noise, `orient.gradient_gain(SD)`, the sum of the squared taps of the derivative filter (the taps of `SD`, then central differences). |
| σ_e | `ThermalSigma`: the thermal noise of echo e before the log, the standard deviation of each real and imaginary channel. In the log, at high SNR, the noise is σ_e / I_e. |
| R | `ThermalCorrelation`: the correlation of the thermal noise across echoes. |
| C_rest | `ResidualCovariance`: the rest, in the units of the gradients. It is about constant relative to the signal (plan, §2), so in the log it does not scale with I_e. |

### The thermal part

Measured in background far from tissue:

1. **The blocks.** Blocks of `BLOCK` = 32 voxels a side on a grid from voxel 0,
   whose box grown by `FAR` = 20 voxels on each axis has no label other than 0
   and lies in the grid.
2. **Per block and echo:** the mean μ of the magnitude, and the variance v of
   its residual about a Gaussian local mean of `BG_SCALE` = 4 voxels (the
   residual removes ghosts and slow structure). The local mean takes about
   0.1 % of the variance, which the fit ignores.
3. **The channels and σ.** Background magnitude is a scaled chi
   distribution with 2N degrees of freedom (N = 1: Rayleigh; a sum of squares
   of N channels: N). μ² / (μ² + v) = Γ(N + ½)² / (N Γ(N)²) gives N per
   block and echo (bisection; N in [0.5, 1e4]). `ThermalChannels` is the
   median N. σ_e is the median over the blocks of √((μ² + v) / (2N)).
4. **R.** For a scaled chi distribution whose channels are independent and
   correlated across echoes by R in each channel, the correlation of the
   squared magnitudes is R². c_ef is the median over the blocks of the
   correlation of the residuals of M_e² and M_f² (`ThermalSquaredCorrelation`),
   and R_ef = sign(c_ef) √|c_ef|. The sign of R cannot be measured; it is
   taken from c.
5. Medians, not means, so that the few blocks with signal (on this scan, at
   the two ends of axis 1, plan §2) do not move the result.
   `ThermalRegion.SignalBlockFraction` gives the share of blocks with
   c_12 > 0.5.

### The rest

Measured in the cortex, superblock by superblock (level 1 by default):

1. The detrended log echoes, their gradients G_e (`orient.gradients`), and
   the tensor J with the weights C_thermal⁻¹ at the superblock's median level
   (the identity if that matrix is singular).
2. n, J's leading eigenvector, and its coherence c (integration scale
   `SI` = 2 voxels).
3. **The voxels:** cortex voxels of the output box, whose whole window is in
   the cortex (the cortex's weight in a Gaussian of √(SD² + SI²) voxels is
   above `INTERIOR` = 0.95), with c ≥ 0.5.
4. **The residual:** r_e = G_e − (G_e · n) n. Its two components in the plane
   each add r_e · r_f / 2 (`ResidualTotal`, the mean over the voxels).
5. **The thermal part there:** the mean of g diag(σ_e/I_e) R diag(σ_e/I_e)
   over the same voxels (`ResidualThermal`).
6. C_rest = `ResidualTotal` − `ResidualThermal`.

**A known bias.** n is fitted in the same window as the noise, so it
follows part of the noise, and the residual loses that part. On a phantom
with known C (`tests/test_orient_noise.py`) the diagonal of C_rest is low by
2 to 5 %, most in the echo with the largest share of J.

`ResidualByCoherence` repeats the rest at c ≥ 0.3 and c ≥ 0.7, so that a
reader can see how much of it depends on the choice of voxels.

### The noise file

One file for each acquisition, in the BIDS derivatives root:

    sub-<sub>/ses-<ses>/anat/sub-<sub>_ses-<ses>[_acq-<acq>][_run-<run>]_desc-orientnoise_noise.json

The entities are those of the first echo (sub, ses, acq, run). BIDS has no
rule for this file; the suffix `noise` and the description `orientnoise`
are our choice. The file is JSON with SpecVersion `orient-noise/1`. Matrices
are E × E, flattened row-major. The command:

    python -m baillarger.orient_noise --echoes E1 E2 E3 --dseg DSEG --out DERIVATIVES [--level 1] [--sigma 0.64] [--report REPORT.json]

| Key | Type | Required | Meaning |
|---|---|---|---|
| `SpecVersion` | string | yes | `orient-noise/1` |
| `Description` | string | yes | What the file is |
| `Sources` | array of strings | yes | The echoes, in echo order, then the segmentation |
| `SourcesSHA256` | object | yes | SHA-256 of each source |
| `Provenance` | object | yes | Created, Function, Version, Commit, Dirty |
| `Echoes` | integer | yes | E |
| `GradientGain` | number | yes | g |
| `DerivativeScale` | number | yes | `SD`, voxels |
| `ThermalSigma` | array of numbers | yes | σ_e, E values, in the units of the echoes |
| `ThermalChannels` | number | yes | N of the chi fit |
| `ThermalCorrelation` | array of numbers | yes | R |
| `ThermalSquaredCorrelation` | array of numbers | yes | c |
| `ThermalRegion` | object | yes | `Kind`, `BlockVoxels`, `FarVoxels`, `LocalMeanScaleVoxels`, `Blocks`, `ChannelsIQR`, `SignalBlockFraction` |
| `ResidualCovariance` | array of numbers | yes | C_rest |
| `ResidualTotal` | array of numbers | yes | The mean in-plane residual products |
| `ResidualThermal` | array of numbers | yes | The mean thermal part over the same voxels |
| `ResidualRegion` | object | yes | `Kind`, `Coherence`, `InteriorWeight`, `Voxels`, `Level`, `BlockCells`, `SigmaMM`, `IntegrationScale` |
| `ResidualByCoherence` | array of objects | yes | For c ≥ 0.5, 0.3 and 0.7: `Coherence`, `Voxels`, `ResidualCovariance`, `ResidualTotal`, `ResidualThermal` |
| `Model` | string | yes | `thermal+rest`: the tensor whitens by both parts |

The numbers are the acquisition's. No code builds them in.
