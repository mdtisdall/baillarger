# Phase 1.2: `orient`

Stream 1, phase 2 of `docs/version-2-plan.md` (§3d). Read the plan's §2
(*The image*, *Noise on this acquisition*, *No bias-field correction*), §3c
step 3 in full, and the §3c Open items on `orient`. The rules in
`docs/plans/round-2.md` apply. The same session did phase 1.1.

## Goal

An on-demand sampler, for a superblock and its halo, of:

1. the local cortex level of each log echo;
2. the whitened multichannel structure tensor J = Σ Gᵀ C⁻¹ G of all the
   echoes;
3. the data normal (J's leading eigenvector, signed by the WM distance of
   phase 1.1) and its coherence.

Also: the noise covariance C, measured on this acquisition; the choice of
σ for the local cortex level, by diagnostics; and the agreement of the
white surface's normals with the data normal.

## Done when

1. `docs/formats/orient.md` gives the sampler's contract, version
   `orient/1`, and the format of the measured noise covariance.
2. The exact tests below pass, on numpy and on torch (MPS on the laptop),
   with equal results up to float32 rounding.
3. C is measured on the hemisphere and published as a small file.
4. σ has a provisional value, chosen by the leakage and border views.
5. The PR bodies give the diagnostics in "The hemisphere runs".
6. `nix develop --command scripts/check` passes on each PR.

## Facts from the current code

### Tensor tools (`streamlines.py`, numpy and scipy, no torch)

- `SD, SI = 1.0, 2.0` (98): derivative and integration scales in voxels.
- `gradient` (105, `np.gradient`), `masked_smooth(x, mask, sigma)` (114,
  `G(x m)/G(m)`).
- `structure_tensor(img, mask, sd, si)` (122–131): one image, 6
  components. No multichannel form. No whitening.
- `_coherence` (143), `eigen(tensor, chunk)` (148–160), `eigenvalues`
  (372), `n_eff(sd, si)` (382), `step_noise` (400).
- Sign: `wm_direction`, `signed_wm_direction`, `tissue_wm_direction`
  (163–232), `oriented_normal` (235–246).
- Tests: `tests/test_streamlines.py` 36, 51, 62, 77, 302.

### The local cortex level

- `column_traces.py`: `TREND = 4.0` voxels (156). `tissue()` (200–240):
  for echoes 2 and 3, `log(max(echo, 1))` minus its `masked_smooth` over
  the cortex mask.
- `base_tensor` (262–266): the plain sum of the tensors of echoes 2 and 3.
- `data_normal` (269–289) duplicates `eigen`.
- `pial_labels.py`: `_trend_contrast` (231–241); the noise:
  `gradient_gain(sd)` (153), `wishart_min_factor` (164), `echo_noise`
  (185), per echo, with no covariance across echoes.
- Do not change `streamlines.py`, `column_traces.py` or `pial_labels.py`.
  Write `orient` as a new module, and copy what you need.

### The noise scripts (now on `main`)

- `prototypes/scripts/noise_corr.py`, `noise_excess.py`,
  `noise_cross_echo.py`, `noise_cross_echo_map.py`,
  `noise_cross_echo_plot.py`. They are the recipe for C. They read
  `<data_dir> <out_dir>` and use nibabel, numpy and scipy only.
- Their results are in `scratch/noise-corr/` (not in git). The plan's §2
  names `prototypes/results/noise/`, which does not exist. Send this
  finding to the coordinator.

### The data

- The echoes: `scratch/v2/echoes/`, three uncompressed `.nii` files
  (`..._echo-{1,2,3}_part-mag_FLASH.nii`), each with a `.source.json`.
- The labels and the surface: `scratch/v2/derivatives/`.

### Surface diagnostics

- On `main` there is no code for the neighbour-angle spread or for
  misfacing faces. The neighbour angle is in
  `prototypes/scripts/antialias.py` (`edge_angles`, 435–439; `pct`).
  `misfacing` is in `wm_sites.py` of PR #81, which phase 1.3 ports. Do not
  depend on it. Write the agreement diagnostic as its own function.

## Design

### Part A: the local cortex level

1. log I_e of each echo (with a floor of 1, as now), and its normalized
   convolution over the cortex mask with a Gaussian of σ in mm.
2. On demand, per superblock with a halo of 3σ.
3. σ is a parameter of the sampler. Its default is chosen in Part D.

### Part B: the noise covariance C

Follow the plan, §3c step 3, *C is measured per acquisition*.

1. **Thermal part:** from background far from tissue (as
   `noise_cross_echo.py` does): σ_e per echo and the correlation across
   echoes. In the log at a voxel it is diag(σ_e / I_e) R diag(σ_e / I_e),
   with I_e from the echo's local cortex level (exp of Part A).
2. **The rest:** the gradients' residual in cortex, after the local level
   and the laminar direction are removed: G minus its projection on J's
   leading direction, in high-coherence voxels. Estimate its covariance
   across echoes. Compare with the thermal part. The plan's §2 gives the
   expected size on this acquisition.
3. **The model for C at a voxel:** the thermal part scaled by the voxel's
   I_e, plus the rest. In the log the rest is about constant relative to
   the signal (§2: 7.5–13.6 % of the signal). Write the model and its
   evidence in the spec. If the thermal part dominates, the thermal C is
   enough (plan §3c step 3). Report which.
4. **The file:** a small JSON derivative, `_desc-orientnoise_noise.json`
   (BIDS-like name; record the choice in the spec), with σ_e, R, the
   residual covariance, the region it was measured in, and the sources.
   The sampler reads it.
5. A command that measures C on an acquisition and writes the file. The
   numbers are this acquisition's. Nothing in the code builds them in.

### Part C: the tensor, the data normal and the coherence

1. G: the gradients of each echo's log minus its local level, at derivative
   scale 1 voxel, by masked normalized convolution over cortex.
2. J = Σ Gᵀ C⁻¹ G over the window, integration scale 2 voxels. The scales
   stay in voxels (plan §3c step 3 gives the reason).
3. The data normal: J's leading eigenvector. Coherence c = (λ₁ − λ₂)/(λ₁ +
   λ₂), clipped to [0, 1].
4. **The sign:** the normal points away from the WM: the sign of n · ∇D,
   with D from phase 1.1's sampler. Ask it for a cap of 5 mm (cortex is at
   most about 4.5 mm thick). Where ∇D is not defined (NaN, inf), the sign
   is 0 and the output says so.
5. **Torch and numpy.** numpy is the reference. Torch for CUDA and MPS. The
   convolutions are separable. Test device equality up to float32
   rounding. No float64 in torch kernels, no atomics (plan §2,
   *Precision*). No discrete step that rounding can flip.
6. **Output per cortex voxel** of the superblock: the 6 components of J
   (on request), the signed normal, the coherence, the sign flag. Nothing
   is stored.
7. `orient` also gives each echo's own tensor on request, for the
   diagnostics.

### Part D: the choice of σ

Follow the plan, §3c step 3, *The local cortex level*. Start at 0.64 mm (4
voxels). Try at least 0.32, 0.64, 0.96, 1.28 mm.

1. **Leakage:** the level's correlation with depth along normal rays at the
   test sites. Use the ray sampler (`rays.RaySampler`, method `normal`) to
   get depth along the rays.
2. **Borders:** the level's ramp across known areal borders. Use the V1/V2
   border at the V1V2 site (where the stria ends). Give how you found it.
3. Make the views. Choose a provisional σ. Give the views and the reason in
   the PR body under "Decisions for review". Do not choose by one number.

### Part E: the diagnostics

1. **Agreement with the surface:** for each white-surface vertex, the angle
   between its normal and the data normal at the cortex voxels within 1
   voxel outward along its normal, weighted by coherence. A function of its
   own, with a test.
2. Each echo's share of J's trace, by depth and by site.
3. Where one echo's normal departs from the combined one: views.
4. For each of the three array axes: the agreement and the coherence
   against the normal's angle to that axis (plan §3c step 3).
5. The plan's check: the echoes agree within 10° in about 91 % of interior
   cortex (SF F1, echoes 2 and 3). Give the same number for the new
   combination.

## Exact tests

- A plane of laminae in a phantom (each echo a different amplitude of the
  same pattern): the normal is the plane's normal, coherence near 1.
- A sphere of laminae: the normal is radial, signed outward.
- Whitening: with C = diag, the result equals the weighted sum of the
  echoes' tensors. With a C that makes one echo pure noise, that echo has
  no weight.
- The local level of a constant image is the constant. Its value at a
  voxel does not depend on the superblock.
- The sign beside an open sulcus does not flip (the case of
  `tests/test_streamlines.py` 315).
- The same output, bit for bit on numpy, from two superblocks.
- numpy against torch within float32 rounding.
- The C measurement on synthetic noise with a known C recovers it within
  its sampling error.

## The hemisphere runs

Read `scratch/v2/echoes/` and `scratch/v2/derivatives/`. Write to
`scratch/phase-1.2/`. Measure one superblock first. Ask the coordinator for
a slot for each run above 8 GB or above 4 cores for 10 minutes.

1. C on the hemisphere (Part B). Its values against the plan's §2 numbers.
2. σ (Part D) at the three test sites.
3. `orient` over the whole hemisphere cortex, superblock by superblock, for
   the diagnostics of Part E. Give the time and the peak. Give the
   agreement as a map on the overview mesh (PNG).
4. Figures in `reports/figures/phase-1.2/`, each well under 1 MB.

## Work breakdown

| Step | Work | Who | Can run with |
|---|---|---|---|
| 1 | The spec draft, with the noise file format | Phase session | 2, 3 |
| 2 | Phantoms: laminae planes and spheres, multi-echo, known C | Sonnet sub-agent | 1, 3 |
| 3 | Part A: the local level, numpy and torch | Sonnet sub-agent | 1, 2 |
| 4 | Part B: the C measurement and its command | Phase session | after 3 |
| 5 | Part C: J, normal, coherence, sign; torch | Phase session | after 3; the sign after 1.1's PR |
| 6 | Part E: the agreement function and its test | Sonnet sub-agent | after 5 |
| 7 | Part D and the hemisphere runs | Phase session | after 4, 5, 6 |

Parts A and B do not need phase 1.1. Start them while phase 1.1's PR waits.

## PRs

1. `feature/orient-noise`: Parts A and B, the spec, and C on the
   hemisphere. Branch from `origin/main`.
2. `feature/orient`: Parts C, D and E. Stacked on PR 1; merge
   `feature/wm-distance-sampler` (phase 1.1) into it. First line of the
   body: "Stacked on #N; contains #M".

Tell the coordinator when each PR opens.

## Decisions for review (made by the coordinator)

- `orient` asks the WM-distance sampler for a cap of 5 mm.
- The sign is 0 where the WM distance is not defined.
- C is published as a small JSON file per acquisition.
- σ is provisional until the user reviews the views.
