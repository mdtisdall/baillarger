# orient: orientation from the image

SpecVersion `orient/1`. The code is `baillarger.orient` (the sampler) and
`baillarger.orient_noise` (the noise file). This is a sampler contract
(`docs/version-2-plan.md`, §1 and §3c step 3): the sampler computes its
output on demand, for one superblock and its halo, from published files
only. Nothing per voxel is stored. The one stored output is the noise file,
a small JSON file for each acquisition.

The specs in `docs/formats/` stay at version 1 until a release, and are
edited in place (plan, §8). This version gives the local cortex level, the
whitened structure tensor, the data normal, its sign and its coherence, and
the noise file.

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
| The WM distance | the sampler `wm-distance/1` on the same segmentation | The sign of the data normal |

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

## The tensor and the data normal

```python
sampler = OrientSampler(echoes, dseg, cells=64, noise=noise_json)
r = sampler.orient(
    superblock,
    level,
    halo_mm=0.0,
    sigma_mm=SIGMA,
    device="numpy",
    tensor=False,
    echo_tensor=False,
)
```

### The method

1. **The detrended log echoes:** the log echo minus its local cortex level
   (σ = sigma_mm), in cortex; 0 elsewhere.
2. **The gradients G_e:** the normalized convolution of the detrended log
   echo over the cortex with the taps of `SD` = 1 voxel, then the central
   difference (s[i + 1] − s[i − 1]) / 2 on each axis, divided by the voxel
   size of the axis (so G_e is in mm⁻¹); 0 outside the cortex.
3. **C at each voxel:** the model of the noise file, with I_e the exp of the
   echo's level. C is per voxel step and G is per mm; for isotropic voxels
   this scales J by one factor, which changes neither the normal nor the
   coherence. For anisotropic voxels the model ignores the difference
   between the axes.
4. **J** = Σ_ef (C⁻¹)_ef G_e G_fᵀ at each voxel, then the normalized
   convolution of each of its six components (`PAIRS`: xx, xy, xz, yy, yz,
   zz) over the cortex with the taps of `SI` = 2 voxels. NaN outside the
   cortex.
5. **The data normal:** J's eigenvector of the largest eigenvalue, made
   canonical (its component of largest magnitude positive).
6. **The coherence** c = (λ₁ − λ₂)/(λ₁ + λ₂), λ₁ ≥ λ₂ the two largest
   eigenvalues, clipped to [0, 1].
7. **The sign:** D is the WM-distance sampler's value with the cap
   `SIGN_CAP` = 5.0 mm (cortex is at most about 4.5 mm thick). ∇D on each
   axis is the central difference where both neighbours are finite, else the
   one-sided difference with the finite neighbour, else 0; it is not defined
   where D is not finite or no axis has a finite neighbour. With
   cos = n · ∇D / |∇D|, the sign is +1 where cos > `SIGN_EPS` (1e-3), −1
   where cos < −`SIGN_EPS`, and 0 elsewhere, and where ∇D is not defined. The
   output normal is the sign times the canonical normal, or the canonical
   normal where the sign is 0. So the normal points away from the WM where
   the sign is not 0.

The WM distance is through tissue, with outside as an obstacle, so the sign
does not flip beside an open sulcus (a straight-line distance flips it,
plan §3c step 3). `SIGN_EPS` keeps the sign from depending on rounding
where the normal is perpendicular to ∇D; there the sign is 0.

The read box is the output box plus, on each axis, the level's taps, `SD`'s
taps and 1 voxel, and `SI`'s taps. The WM distance is read for the output
box plus one voxel.

### Parameters of `orient`

| Parameter | Type | Default | Rule |
|---|---|---|---|
| `superblock` | tuple of 3 integers | (required) | (I, J, K), each ≥ 0 |
| `level` | integer | (required) | m ≥ 0 |
| `halo_mm` | number | 0.0 | 0 ≤ halo_mm, finite, in mm |
| `sigma_mm` | number | `SIGMA` (0.64) | σ of the level, > 0, in mm |
| `device` | string or `Device` | `numpy` | `numpy`, `cpu`, `cuda` or `mps` |
| `tensor` | boolean | false | Give J |
| `echo_tensor` | boolean | false | Give each echo's own tensor |

The sampler refuses a call without a noise file (`ValueError`).

### `Orientation`

| Field | Type | Meaning |
|---|---|---|
| `normal` | float32 array (3, box shape) | The data normal (unit, mm axes of the grid). NaN outside the cortex |
| `coherence` | float32 array (box shape) | c in [0, 1]. NaN outside the cortex |
| `sign` | int8 array (box shape) | The factor (+1, −1) that turned the canonical normal away from the WM; 0 where not defined, and outside the cortex |
| `tensor` | float32 array (6, box shape), or None | J's components (`PAIRS`), if `tensor` |
| `echo_tensor` | float32 array (E, 6, box shape), or None | Each echo's own tensor: G_e G_eᵀ smoothed at `SI` over the cortex, not weighted, if `echo_tensor` |
| `box` | tuple of 3 slices | The output box |
| `sigma_mm` | number | σ of the level, in mm |

### Determinism and devices

- **numpy** (float64, output float32): the normal, the coherence and J are
  the same, bit for bit, from every superblock. The sign uses the WM
  distance, which two superblocks give within its `TOLERANCE` (1e-5 mm);
  `SIGN_EPS` keeps that from changing the sign except at |cos| within about
  1e-5 of `SIGN_EPS`.
- **PyTorch** (float32): the separable convolutions use `conv1d`, the
  inverse of C `torch.linalg.inv`, and the eigenvectors `torch.linalg.eigh`,
  all on the device. No float64 and no atomic additions. The tests require,
  against numpy: J within 1e-4 of its trace, the normal within 1e-3 rad and
  the coherence within 1e-4 where c > 0.2, and the same sign there.
- A plane of laminae gives the normal that the discrete filters see: on
  axis a, sin(k_a) / vs_a, with k_a the wavenumber in radians per voxel.
  At a period of 5 voxels and n = (1, 2, 2)/3 that is 1.7° from n.

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

    python -m baillarger.orient_noise --echoes E1 E2 E3 --dseg DSEG --out DERIVATIVES [--level 1] [--sigma 0.64] [--cache-mb 256] [--report REPORT.json]

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
