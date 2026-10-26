# Plan: contrast bands, regions and surfaces (laminar refinement)

Planning file for the joint estimation of laminar bands, areal regions and
cortical surfaces in baillarger. The method is implemented in this package
(`src/baillarger/`). When the method and its weights are settled, the lasting
content goes into the module docstrings and the README, and this file is
deleted.

The plan started as a LayNii plan (the ex vivo prototype used LayNii's
`LN2_LAYERS` and `LN2_COLUMNS`). baillarger does not use LayNii: the rim
labels come from the aseg, and the columns and the depth frame come from the
white surface (see **Joint surfaces**).

## Status

Updated after #21. "Steps" are the numbers of **Solver** and **Joint
surfaces**; "tests" those of **Correctness tests**.

**Merged** (on `main`):

| PR | Module | What |
|---|---|---|
| #4 | `topology`, `wellcomposed` | simple points, Euler characteristic, well-composed repair |
| #5 | `phantoms` | P1–P9 |
| #7 | `growth` | topological growth, fluid against tissue, `thin_seam`, `mincut_seam` |
| #8 | `mesh`, `mesh_intersect` | marching tetrahedra, mesh topology, RAS VTK, triangle intersection |
| #9 | `data` | patch crops of the three echoes and the aseg, rim labels |
| #10 | `frame` | white mesh, D, streamlines, columns, depth; missing tissue |
| #11 | `columns` | model columns, column graph, voxel table, per-streamline r |
| #12 | `bands` | step 3b (band DP, K by cross-validation), fixed-K baseline |
| #13 | `gain` | steps 2 and 3 |
| #14 | `surfaces` | step 4 |
| #15 | `regions` | step 1 (Ward merging, re-fit levels) |
| #16 | `bottom` | step 5 |
| #17 | `tv` | step 6 |
| #18 | `refine` | steps 1–6 together (`Refiner`, `Params`) |
| #19 | `output` | the outputs of a run; `python -m baillarger.output` |
| #20 | `bands`, `columns`, `output` | one-column regions in `choose_k`; small graph pieces dropped |
| #21 | `joint`, `columns`, `output` | outer loop step 0 (geometric seams as fluid); the fused rule |

(#6 changed `CLAUDE.md` only.) Tests 1–6 and 8 pass; test 9's phantom (P8)
is an expected failure of the seam growth, the open problem that step 7 is
for. A run on V1V2 (all levels, step-0 seams) takes about 4.5 minutes.

**In progress:** nothing. The step-0 work is merged (#21): `joint` places the
seams with the seam growth, `run_patch` uses them as fluid, and the fused
rule counts only streamlines that never exit. What it leaves: a bank whose
streamlines meet the facing bank's at the ridge before they reach its seam
still has no exit (see the finding under **Columns and depth frame**).

**Next**, in this order (decided after the literature review, see
**Literature and ideas to try**):

1. Equivolume r, as an option of the frame (equidistant or equivolume), from
   the divergence of the streamlines' flow (the prototype's
   `surface_geometry.equivolume`); compare both on V1V2 (band sharpness, K_R,
   the stria's width) before step 7 builds on r. Done as an option
   (`column_data(depth=...)`, `--depth`); the default stays equidistant. The
   comparison on V1V2 (below, **Columns and depth frame**) is not conclusive: the data
   term is lower with equivolume r, but on the raw profiles the correction
   for the curvature is about twice too strong (div u checked against the
   mesh curvature; smoothing does not help). All candidates in the full
   model on V1V2 (below): the white mesh's curvature smoothed at 1 mm does
   best; the pial surface's curvature is too noisy. To decide after review.
2. A Hotelling T² test between the column profiles of two neighbouring
   regions: reported for each merge (which merges join distinct areas), and
   used in test A for each WM bin on its own (step 5).
3. Seams as barriers in the frame: D recomputed with the seams as
   obstacles, and a streamline that reaches a seam ends there with its own end
   class, which counts as an exit. A scratch version (without the end class)
   was inconclusive. Once Λ exists (step 7), compare it with D's path cost
   weighted by Λ, on P1, P8 and V1V2.
4. Step 7: the fluid template μ_F and Σ_F, the costs Λ, the fluid against
   tissue growth with Λ, the cut placement with Λ, the repair, the thickness.
5. The outer loop, and tests 7, 9 and 10.
6. The evaluation sweep (with the measures of the earlier column-filter
   evaluation, to port), then the other two patches (PCC_SPL, M1S1). The
   sweep must compare our K rule with an absolute-error held-out loss and
   with mBIC (see **Literature and ideas to try**).

Waiting for a decision:

- How the merging reaches the levels below the number of graph pieces
  (3 and 2 regions on V1V2 with the step-0 seams; see step 1); the T² test
  (next step 2) may make those levels matter less.
- The changes marked "changed" below that depart from the first plan (the K
  rule, the window of each surface, scipy's min-cut, the fused rule).
- Whether PyMaxflow (GPL) may be a dependency.

## Goal

A filter that refines geometric layers with the observed laminar contrast,
for 100 µm ex vivo data. It balances three pressures:

1. **Within-layer total variation.** TV between neighbouring columns, inside
   each layer, on the filtered intensity and on the layer boundary depths.
2. **Bottom layer.** For each region, the deepest depth (into the
   `-extend_wm` layers: u-fibres, superficial WM) down to which the region
   keeps both its areal borders and its laminar transitions.
3. **Nudged layers.** Each column's layer boundaries move so that layers are
   homogeneous inside and distinct from their neighbours, pooled over each
   region. The layers are **contrast bands**: each region gets only as many
   as its laminar contrast supports (K_R, usually well below 13), chosen by
   cross-validation.

The regions are estimated together with the layers, by merging columns.
Laminar contrast is normalized per column, over small units, whatever the
size of the region.

The pial surface and the seams in fused (kissing) cortex are estimated in the
same model (see **Joint surfaces**). The layers need the surfaces, and the
surfaces in fused cortex need the laminar contrast, so neither is an input to
the other: the fluid above the cortex is one more band, and an outer loop
alternates between the model and the geometry. The white surface is fixed. It
anchors the columns and the depth frame.

## Data and terms

Input, per patch (V1V2, PCC_SPL, M1S1; the boxes are in the data manifest):

- The three FLASH echoes, cropped (see **Joint surfaces**). f is their
  vector; the scalar forms below are the one-echo case.
- Rim labels: 1 outside, 2 WM, 3 cortex, 0 other, from the aseg (labels 0,
  41, 42; everything else is 0).
- Columns c, about 0.7 mm across, anchored on the white surface (see
  **Joint surfaces**). Column graph: columns whose patches touch on the white
  mesh and whose cortex voxels share faces; w_cc' = the shared faces, each
  counted by its depth d as max(0, 1 − d / base_mm) (default 1 mm), so that
  columns touching only near the top (facing banks) count little or not at
  all (`baillarger.columns`).
- Continuous depth m(v): the relative depth r(v) of the joint frame, in
  [0, 1] in the cortex (0 = white surface, 1 = pial surface) and negative
  (−distance / T) in the WM below the white surface. T is the exit depth of
  the voxel's own streamline (see **Columns and depth frame**).
- Geometric layers: 25 or 13 equal bins of m. With adaptive bands they are
  only the fixed-K baseline (see Evaluation); the bands are fitted on the
  continuous depth m.

Terms:

- *Region* R: a connected set of columns.
- *Band*: a layer of the refined layering, a depth interval of nearly
  constant normalized intensity. Region R has K_R bands; neighbouring regions
  can have different K_R.
- *Template* μ_R: the region's band profile (one value for each band), zero
  mean and unit norm over the region's common bands.
- *Common bands* of a region: the bands that at least `p_common` (default
  0.8) of its columns have voxels in.
- *Boundaries* z_c = (z_c,1 < … < z_c,K−1), K = K_R: the depths, on the m
  axis, that separate the bands of column c. Band of voxel v: k_c(v) = the
  number of boundaries below m(v), plus 1.
- *Region boundaries* Z_R = (Z_R,1 < … < Z_R,K−1): the band boundaries of the
  region's pooled profile (step 3b), in relative depth.
- *Depth grid*: G fine depth bins of width h (default 1/50 of the local
  thickness) on the m axis, from the deepest WM depth to the pial cut.

## Model

For voxel v in column c of region R:

    f(v) = a_c + s_c · μ_R(k_c(v)) + noise

a_c (offset) and s_c (gain) normalize column c: they remove its intensity
level and contrast, so only the profile **shape** separates regions. They
form a smooth field over the column graph, which is also smooth across
region borders (a bias field does not stop at an areal border).

Energy, for a fixed partition into regions:

    E = Σ_c Σ_{v ∈ c} (f(v) − a_c − s_c μ_R(k_c(v)))² / σ_c²           data (pressure 3)
      + κ Σ_{c~c'} w_cc' [(a_c − a_c')² + (s_c − s_c')²] / σ²             gain/offset field
      + λ_z Σ_k Σ_{c~c'} w_cc' |z_c,k − z_c',k|                           TV on boundaries (pressure 1)
      + η Σ_c Σ_k (z_c,k − Z_R,k)²                                        pull to the region's bands
      s.t. z_c,k+1 − z_c,k ≥ t_min                                        minimum band thickness

σ_c is the noise level of column c, from the variance inside its cells; it
makes the data term dimensionless. Z_R are the region's band boundaries
(step 3b), at the same relative depth in each column; at the first re-fit,
before any bands exist, they are the geometric boundaries of the fixed-K
baseline. Minimizing the within-band residual is the same as maximizing
the between-layer / within-layer (Fisher, Otsu) ratio for the fixed total
variance, so this is pressure 3. It is scored on the raw data, not on the
filtered u, so the TV on u cannot make the layers look homogeneous.

TV on the intensity (pressure 1) is a separate step on the normalized cell
means x̂_c,k = (cell mean − a_c) / s_c in the nudged bands. Two columns of
different regions can have different bands, so the TV compares profiles on
the common depth grid: each column's band values are written to the grid
bins it covers (piecewise constant), and

    min_u  ½ Σ_c,k n_c,k (u_c,k − x̂_c,k)²
         + λ_g Σ_{c~c'} w_cc' ||P_c u_c − P_c' u_c'||₂          group term (as LayNii's LN2_COLUMN_SMOOTH_TV_NORM)
         + λ_l Σ_g Σ_{c~c'} w_cc' |(P_c u_c)_g − (P_c' u_c')_g|  per-depth term

where P_c maps column c's band values to the grid bins (weighted by bin
width, so the scale does not depend on h), over the bins where both columns
have voxels. Inside a region with equal bands this is the plain band TV.
λ_l = 0 is the group form; λ_l > 0 the group + per-depth form. The two are
compared (see Evaluation).

## Joint surfaces

The model above needs the depth of each voxel, so it needs the pial surface.
In fused (kissing) cortex the segmentation joins two banks into one block, and
the pial surface between them is the hard part: there, the image alone is
often not enough, but the laminar contrast of each bank is. So the pial
surface and the seams are unknowns of the same model, and an outer loop
alternates between the model (steps 1–5) and the geometry (step 7).

### Data: three echoes

Each voxel has a vector x(v) of the three FLASH echo magnitudes (TE 9.37,
20.71, 32.05 ms; the echoes are in register with the FLASH, which is echo 2).
The model becomes

    x(v) = a_c + s_c ⊙ μ_R(k_c(v)) + noise,    noise ~ N(0, Σ_c)

with a per-echo offset a_c and gain s_c (vectors of 3) and vector templates
μ_R,k. **Changed:** the noise is diagonal over the echoes, Σ_c =
diag(σ²_c,e), a variance per column and echo pooled inside the cells of 13
equal bins of r (`columns`); the first plan had a full 3 × 3 Σ_c. So every
step is a sum over the echoes of its one-echo form: the gain/offset field
and the templates per echo (steps 2 and 3), and the merge costs (step 1),
the band DP (step 3b), the surface costs (step 4) and the bottom tests
(step 5) sum their one-echo residuals; the prefix sums hold Σ x̂ and Σ x̂² per
bin and echo. With E = 1 each step gives the prototype's outputs on the same
inputs (about 2,700 comparisons, #12–#17). The fluid's Σ_F (step 7) stays a
full 3 × 3 covariance.

Measured on V1V2 (signal over the cortex median, by echo):

| Voxels | Echo 1 | Echo 2 | Echo 3 |
|---|---|---|---|
| Collapsed sulcus lines (512 voxels) | 0.20 | 0.23 | 0.31 |
| Stria-like dark band | 0.81 | 0.77 | 0.74 |
| Ordinary cortex | 1.12 | 1.17 | 1.25 |

Fluid is dim at every echo. The stria is tissue with a faster decay. The
vector keeps the two apart; one echo shows only a difference of level.

### Fluid band

Every column has one more band above its cortical bands: the fluid band F
(Fomblin or CSF, in an open sulcus, outside the specimen and in a seam). Its
template μ_F is one vector for the whole patch, not one per region, and it is
not scaled by the column gain: fluid has no tissue gain. μ_F and its
covariance come from the outside label farther than 0.5 mm from tissue. The
pial surface of a column is the boundary between its top cortical band and F.

### Columns and depth frame

The frame depends only on the fixed white surface, so it does not change in
the loop:

- **Domain.** Cortex, seams and the fluid inside the crop (rim labels 3 and
  1), without missing tissue.
- **Missing tissue.** Blocks cut out of the specimen before the scan, which
  the aseg labels as tissue (imputed) but where the FLASH is black: tissue
  where the FLASH smoothed by 2 voxels is below 0.25 × the cortex median, in
  blocks of at least 250 voxels, grown by 0.3 mm (`frame.missing_tissue`).
  Not in the domain; its WM is no source of D; a streamline stops on entering
  it and is cut: its voxels take no column and it does not count for the
  thickness. The white mesh is made from all the WM, imputed WM too, so the
  edge of a cut does not become a false white surface. V1V2: 416,865 voxels
  (14.6 % of its cortex label).
- **Distance.** D(v), the distance from the white surface (mm), along paths in
  the domain (not through the WM), by second-order fast marching (scikit-fmm;
  p99 error 66 µm on a ball, against 415 µm for 26-neighbour Dijkstra).
- **Streamlines.** From each white-mesh vertex, along the gradient of D
  (smoothed by 1 voxel), outward. A streamline stops at the ridge of D (where
  streamlines from facing banks meet: in fused cortex, the medial surface
  between the two banks), at missing tissue, at the crop faces, or at
  D = D_max (default 4 mm, about twice the thickest cortex).
- **Columns.** Patches of white-mesh vertices about 0.7 mm across: geodesic
  Voronoi cells of the white mesh, from farthest-point seeds until every
  vertex is within 0.35 mm of a seed. Each voxel goes to the column of the
  nearest streamline point within 1.5 mm (streamlines fan out over gyral
  crowns; with 0.35 mm, 3.6 % of the V1V2 cortex fell between them), so each
  voxel is in one column. Unlike columns built from the rim, these do not
  depend on the pial surface. V1V2: 21,034 columns, of which 14,397 have
  cortex.
- **Depth.** d(v): the distance along the voxel's streamline (mm). Relative
  depth r(v) = d(v) / T_p, where T_p is the depth at which the voxel's own
  streamline exits the cortex into fluid (`frame.path_exit`). A streamline
  that never exits (it ends inside the cortex: at a ridge in fused cortex,
  the crop, an excision or D_max) gives its voxels no r, and they stay out of
  steps 1–6. Per column, r = d / T_c put 7 % of the V1V2 cortex beyond
  r = 1.2 (T_c was set by truncated streamlines); per streamline, 1.2 %. The
  column thickness T_c is the median T_p of its streamlines (for the
  thickness prior). The data term and the geometry use d; the band DP (step
  3b), the pull to Z_R, the TV on z and t_min use r.

r is equidistant by default. `column_data(depth="equivolume")` gives the
equivolume r of Waehnert et al. 2014 instead: along each streamline the
cross-section A of a flow tube changes as d ln A / ds = div u (u: the unit
direction of the streamlines, div u from central differences of the
normalized gradient of D, clipped to ±10 per mm), V(s) = ∫ A ds, and
r = V(d) / V(T_p) (r = D / V(T_p) in the WM). Tests: on a flat slab the two
depths agree (< 0.01); round a WM ball (R0 1.6 mm, cortex to 3.6 mm, 0.16 mm
voxels) the equivolume r is within 0.026 (median) and 0.117 (p99) of the
exact (R³ − R0³) / (R1³ − R0³), against 0.15 / 0.21 for the equidistant r.

Comparison on V1V2 (with step-0 seams; same voxels, same 11,922 columns;
levels 256 to 5):

| level | data term, equidistant | data term, equivolume | change |
|---|---|---|---|
| 256 | 3.494e6 | 3.437e6 | −1.6 % |
| 64 | 3.696e6 | 3.604e6 | −2.5 % |
| 16 | 4.020e6 | 3.778e6 | −6.0 % |
| 5 | 4.168e6 | 3.920e6 | −6.0 % |

The column-weighted mean K_R is the same (4.86–5.03 in both). The stria (in
each region with K_R ≥ 3, the middle band that is a local minimum of echo 2;
on the columns where both runs find it; width in mm from each column's own
voxels, d against r):

| level | columns | width (mm), ED / EV | centre r, ED / EV | Spearman(centre r, curvature), ED / EV |
|---|---|---|---|---|
| 256 | 386 | 0.34 / 0.34 | 0.32 / 0.45 | +0.26 / −0.34 |
| 128 | 362 | 0.36 / 0.36 | 0.24 / 0.43 | +0.14 / −0.57 |
| 64 | 213 | 0.35 / 0.33 | 0.28 / 0.44 | +0.32 / −0.73 |

(curvature: the column's median of ln(V(T_p) / T_p), > 0 where the flow
tubes widen, over crowns.) That test is circular: the curvature and the
equivolume r both come from div u, so noise in div u alone can make the
equivolume correlation negative. The checks after it:

- div u against the white mesh. Per column, the mean div u over the first
  0.3 mm of the streamlines against the area-weighted k1 + k2 of the white
  mesh (cotangent Laplacian; the mesh vertices are in voxel units, so per mm
  divide by 0.16): correlation 0.45, about the same spread (5–95 %: div u
  −0.97 to +0.94 per mm, mesh −1.3 to +1.5). Over the whole column, the
  curvature proxy above against the mesh curvature: Spearman 0.79, same
  range. On the ball div u is unbiased (median error −0.04 per mm, 5–95 %
  −0.42 to +0.32, against 2/R of 0.56 to 1.21). So div u is noisy point by
  point but right at the scale of a column.
- Smoothing does not help. Smoothing D's gradient more (σ 1.5, 2, 3 voxels)
  or div u itself (normalized convolution, σ 1, 2, 3) lowers the point
  noise (SD 1.9 to 0.6 per mm) but also the agreement with the mesh (r 0.45
  to 0.35, slope 1.06 to 0.57). The code keeps σ = 1 and no smoothing of
  div u.
- The stria against the mesh curvature (independent of div u), same columns
  as above: Spearman of the centre r: equidistant +0.11, +0.03, +0.15;
  equivolume −0.19, −0.34, −0.41 (levels 256, 128, 64). Weaker than with
  the proxy, but still reversed.
- Without the model. On the 1,876 columns where either run finds the stria
  at level 256, the mean echo-2 profile (z-scored per column, 25 bins of r)
  in each quintile of mesh curvature, with div u scaled by α (α = 0 is
  equidistant, α = 1 equivolume):

  | α | stria minimum trend (r per quintile) | profile RMS between quintiles | R² of one profile for all |
  |---|---|---|---|
  | 0 | +0.011 | 0.130 | 0.329 |
  | 0.25 | +0.006 | 0.098 | 0.351 |
  | 0.5 | −0.006 | 0.087 | 0.353 |
  | 0.75 | −0.020 | 0.091 | 0.337 |
  | 1 | −0.024 | 0.108 | 0.311 |

  The profiles line up best at α ≈ 0.25–0.5; full equivolume is worse than
  equidistant on this test (though better on the model's data term).
- Area from the white mesh instead of div u: each streamline takes the
  principal curvatures k1, k2 of its vertex (mean curvature from the
  cotangent Laplacian, Gaussian from the angle defect, smoothed along the
  mesh at σ), and d ln A / ds = k1 / (1 + k1 s) + k2 / (1 + k2 s) (offset
  surfaces; 1 + k s floored at 0.1 past the focal distance). Same test:

  | σ (mm) | trend | RMS | R² | ×0.5: trend | RMS | R² |
  |---|---|---|---|---|---|---|
  | 0.5 | −0.008 | 0.083 | 0.264 | −0.002 | 0.074 | 0.304 |
  | 1 | −0.002 | 0.075 | 0.280 | +0.004 | 0.080 | 0.319 |
  | 2 | +0.016 | 0.076 | 0.262 | +0.014 | 0.095 | 0.309 |
  | 3 | +0.021 | 0.087 | 0.237 | +0.021 | 0.100 | 0.286 |

  At σ 0.5–1 mm the stria's trend is flat, but one profile explains less of
  the voxels than with equidistant r (0.33) or half div u (0.35): the
  profiles are blurred. At σ 2–3 mm the trend comes back. So the mesh
  areas are not better than div u here.
- div u is never clipped at ±10 per mm on V1V2; it is 0 (weak gradient or
  the edge of the domain) at 3.7 % of the streamline points.
- Correction to the row above: its mesh smoothing weighted each vertex by
  its valence, which biases the Gaussian curvature on marching-tetrahedra
  meshes (on a ball: 0.017 instead of 0.077 per mm²). With mass-keeping
  smoothing the ball and cylinder give the exact area ratios.

**Full model on V1V2, all candidates** (scratch runs, with step-0 seams; the
candidates replace div u along the streamlines):

- div u, half div u: as above.
- mesh offset: the white mesh's k1, k2 at the streamline's vertex (smoothed
  along the mesh at 1 mm), A(s) = (1 + k1 s)(1 + k2 s).
- two-surface (Waehnert's closed form, A linear in depth between the two
  ends): A_out / A_in = R from the white surface's curvature (R_in =
  1 + H T + K T²), the pial surface's at the exit point (R_out =
  1 / (1 − H T + K T²); pial mesh from WM + cortex), or their geometric mean.
  On the ball and the cylinder all three are within 0.013 (median) of the
  exact equivolume r, against 0.024–0.030 for div u.

| frame | data term ×1e6, level 256 / 32 / 5 | raw profiles: stria trend / RMS / R² | stria columns (256) | Spearman(stria centre r, mesh k): 256 / 128 / 64 |
|---|---|---|---|---|
| equidistant | 3.494 / 3.844 / 4.168 | +0.012 / 0.130 / 0.329 | 1,613 | +0.15 / +0.12 / +0.17 |
| div u | 3.437 / 3.696 / 3.920 | −0.024 / 0.108 / 0.311 | 649 | +0.03 / −0.31 / −0.36 |
| half div u | 3.466 / 3.741 / 3.987 | −0.006 / 0.087 / 0.353 | 743 | −0.12 / −0.22 / −0.15 |
| mesh offset | 3.533 / 3.834 / 4.083 | −0.002 / 0.085 / 0.346 | 1,830 | +0.07 / +0.04 / +0.20 |
| two-surface, both | 3.553 / 3.871 / 4.144 | +0.009 / 0.085 / 0.326 | 1,989 | +0.01 / +0.04 / +0.10 |
| two-surface, pial | 3.542 / 3.848 / 4.153 | −0.135 / 0.084 / 0.286 | 1,690 | −0.05 / +0.00 / +0.11 |
| two-surface, white | 3.528 / 3.829 / 4.065 | +0.001 / 0.089 / 0.346 | 1,917 | +0.04 / +0.21 / +0.11 |

K_R is about 5 everywhere. The stria's width is 0.29–0.36 mm in every
frame that uses the white surface only; with the pial surface, the stria
band found is often a thin band near the pial surface (centre r 0.77–0.94,
width 0.05–0.13 mm), so those rows' stria numbers are not the stria.

Findings: the pial surface's curvature at the exit point is too noisy to
use (the pial mesh closes thin sulci, and the exit point is on whichever
bank is nearest); the two-surface hypothesis is not supported here. The
white-surface curvature smoothed at 1 mm (mesh offset, or the two-surface
form with R_in) lines the profiles up as well as half div u, with no
trend, finds the stria in more columns than equidistant, and lowers the
data term by 1–2.5 %. div u corrects too much although it agrees with
the mesh curvature per column; it is the voxel-scale div u, deeper in the
cortex (convergence in fundi, ridges), that the 1 mm white-surface
curvature leaves out. Proposed default for step 7: equivolume r from the
white mesh's smoothed curvature (mesh offset); to decide after review.

**Again with the edge zones out and λ_z = 5** (#24; scratch runs of the
same candidates, pial-based ones left out):

| frame | data term ×1e6, level 256 / 32 / 5 | raw profiles: trend / RMS / R² | stria columns 256 / 64 / 32 / 16 | Spearman(stria centre r, mesh k) 256 / 128 / 64 | step across regions, 32 |
|---|---|---|---|---|---|
| equidistant | 3.144 / 3.406 / 3.672 | +0.017 / 0.177 / 0.486 | 2,868 / 1,565 / 1,113 / 0 | +0.30 / +0.34 / +0.19 | 0.105 |
| div u | 3.101 / 3.249 / 3.351 | −0.031 / 0.131 / 0.482 | 1,130 / 839 / 513 / 148 | −0.19 / +0.00 / +0.01 | 0.105 |
| half div u | 3.132 / 3.331 / 3.477 | −0.006 / 0.129 / 0.529 | 2,116 / 1,249 / 330 / 0 | −0.01 / +0.04 / +0.12 | 0.095 |
| mesh offset | 3.181 / 3.370 / 3.539 | −0.006 / 0.109 / 0.525 | 2,546 / 2,310 / 1,232 / 721 | −0.07 / −0.08 / −0.02 | 0.085 |
| two-surface, white | 3.164 / 3.385 / 3.573 | −0.006 / 0.120 / 0.525 | 2,724 / 2,030 / 1,302 / 121 | −0.01 / −0.01 / −0.02 | 0.085 |

(raw profiles now on 2,935 V1-like columns.) With the edges out, the
picture is the same and sharper: the white-surface curvature at 1 mm (mesh
offset, or the two-surface form with R_in) has no stria trend in the raw
profiles or in the model's bands, the tightest profiles between curvature
groups, the least step between regions, and keeps the stria at coarse
levels (721 columns at level 16, against none with equidistant r). Its
data term is 1 % above equidistant at level 256 and 3.6 % below at level 5.
div u fits the data term best but turns the trend over. Proposed: the mesh
offset as the equivolume method.

So the overcorrection is real and not noise: the flow tubes of the distance
from the WM widen (or narrow) about twice as fast as the layers do. A likely
reason: they follow the white surface's curvature only, while Waehnert et
al.'s equivolume model uses the areas of both the white and the pial surface,
and the pial surface is flatter than the WM over crowns and in fundi. To try
before step 7 uses r: a two-surface equivolume (the tube's area at the pial
end from the pial surface, or a Laplace field between the two surfaces), and
a scaled div u (α = 0.5) only as an empirical baseline. The stria is found in
fewer columns with equivolume r at levels 256–64 (649 vs 1,613 at 256) and
in more at 32 (168 vs 104).

**Columns in steps 1–6** (`columns.model_columns`): a column takes part when
it has cortex voxels on streamlines that exit into fluid, is not cut by
missing tissue, and at most half of its streamlines are fused. A streamline
is fused when it ends at a ridge inside the cortex without having exited
first (#21: with seams, a streamline can exit at a seam and go on to a ridge
in the facing bank's cortex; counted as fused, it kept even the bank that
crossed the seam out). Then `drop_small_pieces` (#20) removes the columns in
connected pieces of the column graph with fewer than 5 columns: they can
never merge, and one column cannot cross-validate its bands.

**Finding (still open): streamlines regenerated in the loop.** With the fixed
frame, both banks of a fused stretch stop at the ridge midway between their
WM surfaces. When step 7 puts the seam at the true contact, nearer the
thinner bank (P8: banks of 12 and 18 voxels), the thicker bank's
streamlines stop at the ridge inside its own cortex and never reach the seam,
and its voxels between the seam and the ridge go to the thinner bank's
streamlines. With the seam as a barrier to D and to the streamlines, the
thicker bank's streamlines reach the seam (P8: 2.84 mm against 2.88 mm to the
contact). So, at each outer iteration: the columns (patches of the white
mesh) stay fixed; D is recomputed with the current seams as barriers, the
streamlines are traced again from the same vertices, and a streamline that
reaches a seam exits there. Test 8 then reads: with no seams, the frame does
not depend on the pial surface. Evidence so far: P8 (above); P1 with the
step-0 seams as fluid only (#21, `test_joint`): the cut lies at x = 28, the
ridge at x = 31.5, so bank A's streamlines exit at the seam (> 80 % of its
patch voxels join the model) and bank B's do not (< 50 %). A scratch run on
V1V2 with the seams as barriers covered less, not more (26 % of the cortex,
against 64 % with the seams as fluid), but its exit test looked only one
sample past a streamline's end and missed most seams: inconclusive. Next
step 1 of **Status**.

### 7. Geometry (fixed regions, templates, gain/offset, bands)

Unknowns: the tissue/fluid label of each voxel of the domain (this gives the
pial surface and the seams), then T_c.

1. **Voxel costs from the model.** For voxel v in column c:
   - L_F(v) = ½ (x(v) − μ_F)ᵀ Σ_F⁻¹ (x(v) − μ_F): the cost of fluid;
   - L_T(v) = the same for the column's band at r(v) (the top band for
     r(v) ≥ Z_R,K−1), with the column's gain and offset: the cost of tissue;
   - Λ(v) = L_T(v) − L_F(v): positive where v looks like fluid.
2. **Fluid against tissue** (`grow_competing`, with Λ in place of the
   darkness measure). Tissue grows from the WM and joins only through
   simple points, so it keeps the topology of the WM. Fluid flows in from the
   outside. Per mm, fluid is cheap where Λ is high and tissue is expensive
   there; one weight β (sweep) sets how strongly Λ acts. A smooth-thickness
   prior adds γ · max(0, d(v) − T̂_c) to the tissue cost, where T̂_c is the
   thickness of the column's neighbours from the previous iteration (with the
   TV of step 4). Where two banks touch with no fluid between them, topology
   forces a cut, and the cut goes to the fluid: the seam.
3. **Cut placement.** Thin the cuts (`thin_seam`), then move each cut onto
   the cheapest separating surface near it (`mincut_seam`), with Λ as the
   voxel cost in place of the darkness.
4. **Well-composed tissue** (`repair`), so that the meshes have
   the topology of the voxels.
5. **Thickness.** T_p of each streamline: where it exits the tissue into
   fluid or a seam; T_c = the median T_p over the column's streamlines.

Step 2 is a growth, not the minimum of an energy, so the outer loop does not
always lower E. The loop records E after each step.

### Outer loop

0. Start from the geometric surfaces: the white surface from the WM labels,
   and the pial surface and seams from the seam growth with the echo-2
   darkness (as the ex vivo prototype), which give T_c. Done (#21):
   `joint.initial_seams` runs `grow_competing`, `thin_seam` and `mincut_seam`
   on the echo-2 FLASH, with missing tissue left out; `output.run_patch` uses
   the seams as fluid (`--no-seams` leaves them out). On V1V2 (24 s; 44,728
   seam voxels; 616 cuts placed by min-cut) they raise the cortex in the
   model from 58 % to 69 %, leave the column graph in 5 pieces instead of 8,
   and the merging reached every level down to 4 regions (a scratch run before
   the fused-rule fix). At 256, 64 and 16 regions, Dan's point
   (66, 62, 62) lies on a border between two regions. A bank whose
   streamlines meet the facing bank's at the ridge before they reach its seam
   still has no exit: the seams as barriers (finding above).
1. Steps 1–5 of the model, in the fixed frame.
2. Step 7, with the regions of the finest re-fit level whose regions all have
   at least N_geo columns (default 50), so that each template is pooled over
   many columns.
3. Repeat from 1 until fewer than n_stop voxels (default 0.1 % of the cortex)
   change label, or for it_max iterations (default 5). (Not built: the
   stopping rule is untested.)

The templates are pooled over many columns, so the columns with a clear pial
surface set them, and the fused columns are fitted to them. In a fused
stretch with no visible line, the cut then goes where both banks' profiles
end, not to the midpoint between the WM surfaces.

## Solver

### 0. Data preparation

- For each patch: crop the three echoes and the aseg to the patch box, and
  make the rim labels from the aseg.
- For the fixed-K baseline only: merge sparse edge layers. A geometric
  layer with fewer voxels than `f_sparse` (default 0.25) × the median layer
  count in the patch is merged into its inner neighbour. The adaptive
  bands work on the depth grid and need no such rule.

### 1. Regions: agglomerative merging

The merge loop is the outer loop; it gives the full hierarchy.

1. Start: each column is a region.
2. Merge cost: the increase of the data term when two touching regions share
   one template (Ward linkage on the normalized column profiles, with the
   column graph as connectivity), using the current gain/offset and
   boundaries.
3. Merge the cheapest pair; update the costs of its neighbours.
4. **Re-fit levels.** A full re-fit (steps 2–5 below) after each merge is too
   expensive. The re-fit runs when the number of regions reaches one of the
   levels, and the merge costs are then recomputed. **Changed:** a fixed list,
   256, 128, 64, 32, 16, 8, 5, 4, 3, 2 (`refine.DEFAULT_LEVELS`), not N₀ / 2^j.
   Before the first re-fit the gain/offset is each column's cortex mean and
   SD, per echo; the merge costs need no boundaries (they compare the
   profiles on the depth grid). At each re-fit (`refine.Refiner.refit`): the
   bands of each region (3b), which start the columns' boundaries at Z_R; then
   3 rounds of templates, gain/offset, templates, boundaries (4), templates;
   then the bottoms (5) and the TV (6). The energy terms are recorded after
   each round; on V1V2 the total falls at every level.
   Each re-fit chooses K_R again for every region (step 3b), so a merged
   region can have more or fewer bands than its parts.
5. Stop at one region, or when no regions touch. Nothing chooses a level: the
   output is the merge tree plus the full results at each re-fit level.

The merge cost uses the profiles on the depth grid up to m = 0.85 (no
pial-edge bins), each bin weighted by s²/σ², summed over the echoes (`regions`;
with E = 1 the prototype's merge tree, 468 merges). The prototype's
border-length (Potts) term is an option, off.

**Merging stops at the number of connected pieces of the column graph**: it
joins only touching regions. V1V2 had 19 pieces (fused and cut columns left
out, and the crop), so no level below 19 was reached; with the small pieces
dropped (#20) 10, and with the step-0 seams 4 (6,066, 4,796, 151 and 18
columns), so 3 and 2 regions are still out of reach. `run.json` reports the
pieces and the levels not reached. Open: merge across pieces by Ward cost
once no touching pair is left (one try, without re-fits, gave 8,462 against
940 columns, unrelated to Dan's points), link the pieces through the fused
cortex once step 7 places its seams, or set the levels per piece.

### 2. Gain/offset field (fixed regions, templates, boundaries)

Quadratic: a sparse linear system in 2 unknowns per column and echo (data
term plus graph Laplacian, scaled by the echo's median noise), one per echo,
solved directly (`scipy.sparse.linalg.spsolve`; 0.2 s for V1V2's 9,402 columns
× 3 echoes). A tiny ridge keeps a column with nothing to fit at its previous
values. κ = 0 gives the independent per-column least squares; κ → ∞ one gain
and offset per echo (test 2).

### 3. Templates (fixed gain/offset, boundaries)

μ_R,k = Σ s_c (f(v) − a_c) / Σ s_c², over the voxels of region R in band k
(weights 1/σ²_c), per echo. Then centre and scale μ_R to zero mean and unit
RMS over the region's voxels, per echo, and change a_c and s_c to match, so
the model is unchanged; a flat template (K = 1) is zero and its mean goes into
the offsets. **Changed:** over all the region's voxels, as the prototype, not
over the common bands (p_common is not used).

### 3b. Bands of each region (fixed gain/offset)

For each region, an optimal segmentation of its pooled profile on the depth
grid, in relative depth (each column's own m), with the normalized voxels
x̂(v) = (f(v) − a_c) / s_c:

1. For each grid bin g: Σ x̂, Σ x̂² and the voxel count, over the region's
   voxels in the bin. Prefix sums give the residual of any interval in O(1).
2. Dynamic programming (Bellman segmentation): for each K = 1..K_max
   (default 8), the K-band split of the grid with the least within-band
   residual, with each band at least t_min wide. One pass gives every K;
   O(K_max · G²) per region. The split for K need not be a refinement of
   the split for K − 1.
3. **Choose K by cross-validation.** Split the region's columns into two
   halves A and B at random (fixed seed); the halves share no voxels. Fit
   the split for each K on A; measure its residual on B, with B's voxels
   normalized by their own gain and offset; swap A and B and add. K_R is the
   least K whose held-out residual is within the larger of one standard
   error (of the paired difference over the held-out columns) and 5 % of the
   whole held-out gain (residual at K = 1 minus the least) of the least one.
   (Re-fitting each held-out column's gain and offset to A's template, as
   first planned and as the prototype did, gives every K ≥ 2 a gain fitted
   on the held-out data that a flat template cannot use: a flat profile then
   never got K = 1. The standard error alone stops nothing in a large
   region, where every extra band of a smooth trend is a real improvement:
   K = K_max on V1V2's large components; with the 5 % term, K = 5 or 6.)
4. Z_R = the best split for K_R on all columns.

A region of fewer than 6 columns splits its voxels for the cross-validation,
not its columns. **#20:** a region of one column then has a single held-out
column and no standard error, so only the 5 % gain term applies (it fell
back to the least residual, and on V1V2 one-column regions got K up to 8;
`test_bands.test_one_column_region`).

Each column's boundaries are then nudged around Z_R in step 4. A region
with K_R = 1 has no boundaries (no laminar contrast that generalizes).

### 4. Boundaries: optimal surfaces (fixed gain/offset, templates)

Exact min-cut for the K_R−1 surfaces of all columns together (Li, Wu, Chen &
Sonka 2006; Ishikawa 2003). The data term splits into one term for each
surface: with e_k(v) = (f(v) − a_c − s_c μ_R,k)² / σ_c²,

    Σ_v e_{k_c(v)}(v) = Σ_v e_1(v) + Σ_k Σ_{v : m(v) ≥ z_c,k} (e_k+1(v) − e_k(v)),

so the cost of surface k at depth d in column c is a cumulative sum over the
column's voxels sorted by depth. The prior η(z − Z_R)² is a unary term;
TV on z is a convex pairwise term between columns; the minimum thickness is
a hard constraint between neighbouring surfaces of one column. All of these
fit the construction, so the cut gives the global optimum of this step.

TV edges between columns of different regions have weight 0: their bands
need not correspond (K_R can differ), and boundaries can jump at a region
border. Each region is then its own min-cut problem.

**Coarse to fine.** The graph has columns × depth samples × surfaces nodes.
With fixed layers this is large (V1V2 with 24 surfaces at full resolution:
about 4,500 × 250 × 24 = 27 million); with bands it has only K_R − 1
surfaces per region, which is much smaller. The coarse level is the
region's DP split Z_R on the depth grid (step 3b); the fine level is the
min-cut in a band of ± `w_band` around Z_R, with a depth step of 1/200 of the
thickness. `w_band` is half the narrower of the two bands next to each
surface (changed: first planned as half the region's narrowest band for
every surface, which on V1V2 is the thin band at the pial edge, r 0.94 to 1;
every surface could then move only ± 0.02 to 0.03 and 30 % of them stopped
at the window's edge, against 17 % and a 6 % lower data term with the
window of each surface; `surfaces.candidates`, `window="region"` keeps the
first plan). The fixed-K baseline uses the same two levels, with the
13-layer bins as the coarse level.

The min-cut is `scipy.sparse.csgraph.maximum_flow` on costs scaled to 32-bit
integers (see **Layout and dependencies**); on 25 random regions it gives the
prototype's depths in all 1,130 boundaries (test 1 by brute force passes).

### 5. Bottom (pressure 2), for each region at each re-fit level

The profiles are extended into the WM, in steps of one depth-grid bin
(normalized with the column's own a_c, s_c). The WM part of the profile is
segmented with the same DP as step 3b, continuing the region's bands, so
a band can also reach below the GM/WM border. For each candidate bottom b
(the WM depth down to b included):

- **Test A, areal border.** For each neighbouring region R', over the
  columns within `D` (default 2 links of the column graph) of the R–R'
  border: the separation of the two groups of profiles (between-group /
  within-group variance, over the bins and the echoes). It fails when it
  drops by more than `δ_A` (default 0.1) from its value with no WM layers.
- **Test B, laminar transitions.** The fraction of variance of the region's
  columns that the extended template explains. It fails when it drops by
  more than `δ_B` (default 0.1) from its value with no WM layers.

b_R is the deepest b for which every depth from 0 to b passes both tests.
Only columns that have WM voxels down to b vote; the search stops
("voters") where fewer than max(3, 20 %) of the region's columns do.
Columns without WM voxels take their region's b_R. As built (`bottom`, as the
prototype): the WM part of test B is the region's mean in each WM bin, not a
DP continuing the bands; the WM extension is the WM within 0.5 mm of the
white surface, each voxel in the column of its nearest white vertex.

**Finding (still open): both tests react late.** Each compares a total over
all the bins from b up with its value at b = 0, so a WM bin with no contrast
only dilutes it. On a synthetic patch of two areas (6 × 4 columns, 12 WM
bins) whose contrast stops at the WM, test A stopped 3 bins (0.075 of the
thickness) too deep; with a contrast fading over about 5 bins, neither test
stopped before the deepest bin; test B, which noise-only WM bins barely move,
never stopped unless the WM voxels were much noisier than the cortex. To
try: test each new bin on its own (the between-group separation of bin b,
and the variance the template explains in bin b, against the cortex bins'),
so that a bin with no contrast fails at once. On V1V2 (step-0 seams) most
regions stop on test A or on the voters, at about −0.12 to −0.26 of the
thickness; test B stopped 2 to 6 regions per level. Waiting for a decision.

### 6. Intensity TV (for each output level)

Chambolle–Pock with the per-depth term added (its dual prox is a clip of
each entry) and the grid maps P_c in the difference operator. **Changed:**
fixed steps from the operator norm (power iteration), as the prototype, not
diagonal preconditioning with adaptive steps. With three echoes the group
term's vector holds every bin and every echo, so an areal border acts on the
whole multi-echo profile; the per-depth term is per bin and echo (`tv`).
u does not feed back into steps 1–5. V1V2 (9,402 columns, 5 bands): 8 s for
λ_g = 0.5, 14 s with λ_l = 0.1.

## Weights (normalized units)

| Weight | Unit | Default (first guess) |
|---|---|---|
| λ_g, λ_l (TV on u) | voxels per shared face, on normalized profiles (as TV_NORM) | sweep |
| λ_z (TV on z) | per shared face, z in fractions of local thickness | sweep |
| η (pull to geometry) | per column, z in fractions of local thickness | sweep |
| κ (gain/offset smoothness) | per shared face, a and s relative to σ | sweep |
| t_min (minimum band thickness) | fraction of local thickness | 0.05 |
| K_max | bands | 8 |
| h (depth grid) | fraction of local thickness | 1/50 |
| δ_A, δ_B | relative drop | 0.1 |
| p_common, f_sparse | fractions | 0.8, 0.25 |
| β (Λ in the growth) | per unit of Λ | sweep |
| γ (thickness prior in the growth) | per mm beyond T̂_c | sweep |
| D_max (streamline length) | mm | 4 |
| N_geo (columns per region for step 7) | columns | 50 |
| n_stop, it_max (outer loop) | voxels, iterations | 0.1 % of cortex, 5 |
| gain_frac (K rule, step 3b) | fraction of the held-out gain | 0.05 |
| column_mm, reach (frame) | mm | 0.7, 1.5 |
| base_mm (column graph) | mm | 1 |
| fused_frac, min_piece (model columns) | fraction, columns | 0.5, 5 |
| wm_mm (WM extension) | mm | 0.5 |
| m_max (merge profiles) | relative depth | 0.85 |
| n_inner (rounds per re-fit) | rounds | 3 |

The sweep weights start at the prototype's defaults (`refine.Params`): κ = 1,
λ_z = 1, η = 10, λ_g = 0.5, λ_l = 0; none has been swept yet.

## Outputs (per patch)

- Merge tree (parent, cost, size for each merge), and per re-fit level:
  - region labels (column and voxel images);
  - K_R, Z_R and the cross-validated residual for each K = 1..K_max, per
    region (CSV);
  - band labels and the continuous boundaries z_c (one value per column and
    boundary, CSV, and a band image);
  - gain and offset per column;
  - bottom b_R per region, and a bottom-depth image;
  - filtered image: each voxel shifted by the change of its cell mean, as
    the TV_NORM default (u in normalized units).
- A JSON with parameters, energies per term and solver statistics.
- Joint surfaces, per outer iteration: the tissue/fluid labels, the seam,
  T_c per column (CSV), the pial mesh (RAS, for ITK-SNAP), Λ as an image, and
  E after each step.

Written now (`output`, #19 and #21): `merge_tree.csv`; `run.json` (parameters,
column counts and why the others are out, the seams' report, the graph
pieces and the levels not reached, per level the energies, K and seconds);
`seams.nii.gz`; and per level `regions.nii.gz`, `bands.nii.gz` (the WM down
to the region's bottom as −1, which stands in for a bottom-depth image),
`filtered.nii.gz` (4D, per echo), `regions.csv` (K, Z, template, bottom and
why it stopped, cross-validated residuals) and `columns.csv` (frame column,
region, T_c, a, s, σ per echo, z). Not yet: the column-level region image,
the joint-surface outputs.

## Evaluation

For λ_l / λ_g ∈ {0, 0.1, 0.3, 1} over a λ_g sweep, at the 2-region level and
at the level nearest to the earlier k = 2 split:

- The measures of the earlier column-filter evaluation (to be ported):
  roughness, contrast, isolated, border, and dist_dan (the distance to Dan's
  areal boundary points).
- Laminar measures: the pooled within-layer residual fraction per region
  (data term / total variance), and, in V1V2, whether the stria of Gennari
  stays inside its layers: the width (in layers) of the stria's dip in the
  V1 template and in u, against the unfiltered profile.
- Whether the nudge helps: the same measures with η → ∞ (no nudge).
- **Fixed against adaptive layers.** The whole filter with fixed geometric
  layers (13 bins, and 25 bins with the sparse edge layers merged) against
  adaptive bands, on all the measures above.
- **Bands per region.** Maps of K_R and Z_R per region, and the
  cross-validated residual against K for each region: for example, whether
  V1 needs more bands (the stria) than V2.
- **Joint surfaces.** On V1V2, at (69, 65, 69), (68, 60, 71), (81, 49, 162)
  and (61, 94, 78): the seams and the pial surface after each outer
  iteration, against the geometric start. The digital handles of the pial
  solid against those of the WM (16 on V1V2), and the meshes in ITK-SNAP.
- **Band stability.** Whether the band boundaries agree between the two
  cross-validation halves (the distance between Z_R fitted on A and on B),
  and between neighbouring re-fit levels of the region hierarchy.

## Correctness tests (prototype)

1. Optimal surfaces against brute force on a tiny problem (3 columns,
   2 surfaces, 10 depths).
2. Gain/offset with κ = 0 equals per-column least squares; with κ → ∞ it
   equals one global gain and offset.
3. Intensity TV: λ = 0 gives x̂; λ_g large gives the mean profile over each
   connected component; λ_l = 0 equals the group-TV problem of LayNii's
   `LN2_COLUMN_SMOOTH_TV_NORM` (before its normalization), checked against a
   direct convex solve on a small case.
4. Merging with re-fits off and equal column weights equals scikit-learn
   Ward clustering with the same connectivity.
5. A column copied with a gain and an offset gets the same boundaries and
   template.
6. Band DP against brute force on a short grid (G = 12, K ≤ 4), and on a
   synthetic piecewise-constant profile with noise: cross-validation finds
   the true K and the boundaries within one grid bin.
7. The phantoms (P1–P9) with three synthetic echoes (fluid dim at every
   echo, a stria band with a faster decay): step 7 keeps the results of the
   seam growth on them (seam on the dark line and through its
   holes, no seam on the stria, no seam at a sulcus end, the WM topology
   kept).
8. The frame does not depend on the pial surface: changing the rim changes
   neither the streamlines nor d.
9. A phantom with a known band template in both banks and a fused stretch
   with no visible line, the banks of different thicknesses: the cut goes
   where each bank's profile ends, within 1 voxel, not to the midpoint.
10. On the phantoms, the outer loop stops (n_stop) within it_max iterations.

Status: 1–6 pass (`test_surfaces`, `test_gain`, `test_tv`, `test_regions`,
`test_surfaces` for 5, `test_bands`); 8 passes (`test_frame`); 7, 9 and 10
wait for step 7 and the outer loop (P8's cut at the natural contact is an
expected failure of the seam growth, `test_growth`).

## Layout and dependencies

In `src/baillarger/`, with the tests in `tests/`:

- the topology core: `topology`, `wellcomposed`, `growth` (topology-preserving
  growth, the fluid-against-tissue competition, seam thinning and the
  min-cut cut placement);
- meshes: `mesh`, `mesh_intersect`;
- the phantoms: `phantoms`;
- data: `data` (patch crops, rim labels, the three echoes);
- the frame: `frame`; the model's inputs: `columns`;
- the model: `regions`, `gain`, `bands`, `surfaces`, `bottom`, `tv`, `refine`
  (tests 1–6); see **Model implementation**;
- the joint surfaces: `joint` (step 0 now; step 7 and the outer loop next,
  tests 7–10);
- runs and outputs: `output`;
- the evaluation sweep (not started).

Dependencies (through uv): numpy, scipy, nibabel, numba, scikit-fmm (fast
marching for D); scikit-image and scikit-learn for the tests. Min cuts use `scipy.sparse.csgraph.maximum_flow`; it
takes 32-bit integer capacities, so the optimal surfaces scale their real
costs to integers (optimal to within the rounding; on random regions the
same depths as the prototype's real-valued PyMaxflow cut). PyMaxflow is
GPL-licensed, so it is not a dependency unless the project's licence allows
it; a 3,906-column region of V1V2 takes about 2 s with scipy.

## Model implementation

Steps 1–6 port the ex vivo prototype's `layer_refine.py` (one echo, LayNii
columns), generalized to E echoes on the frame's columns and depth. With
E = 1, each ported function gives the prototype's outputs on the same
synthetic inputs; its LayNii-specific parts (column splitting, fused-zone
files, the equivolume metric) are not ported.

All done:

| PR | Module | Contents | Tests | Merged |
|---|---|---|---|---|
| A | `columns` | model columns, column graph, voxel table (column, r, echoes), noise per column and echo, depth-grid sums | slab depth and weights by brute force; no links across P1's fused banks | #11 |
| B | `bands` | band DP for every K ≤ K_max with t_min; K by 2-fold cross-validation; fixed-K baseline | test 6 | #12 |
| C | `gain` | gain/offset field (sparse solve, 2 unknowns per column and echo); templates | test 2 (test 5 with D) | #13 |
| D | `surfaces` | optimal-surface min-cut per region, coarse to fine | tests 1, 5 | #14 |
| E | `regions` | Ward merge loop, re-fit levels, merge tree | test 4 (scikit-learn) | #15 |
| F | `bottom` | WM extension of the bands, tests A and B | synthetic bottoms | #16 |
| G | `tv` | intensity TV (Chambolle–Pock), group and per-depth terms | test 3 | #17 |
| H1 | `refine` | steps 1–6 together, in memory | end to end on a two-area phantom | #18 |
| H2 | `output` | the outputs, the command line | the files of one level | #19 |
| fix | `bands`, `columns` | one-column regions in the K rule; small graph pieces | | #20 |

Decisions for the model:

- **Noise.** Diagonal over the echoes: σ_c is a variance per column and echo.
  The fluid's Σ_F in step 7 stays a full 3 × 3 covariance.
- **Column graph.** Mesh adjacency and the base-weighted shared faces (see
  **Data and terms**). On V1V2, links between columns more than 3 mm apart
  along the white mesh keep 0.1 % of the weight, against 2.5 % with plain
  face counts; on P1 there is no link across the fused stretch (66 with
  plain faces). `links="faces"` keeps the plain rule for comparison.
- **K rule.** Cross-validation only (the prototype's other rules are not
  ported), with the held-out residual not re-fitted and the least K within
  max(one standard error, 5 % of the held-out gain) of the minimum (step 3b);
  a one-column region uses the 5 % term alone (#20). Changed from the first
  plan: on synthetic regions (K = 1 to 4, one or three echoes, 20 seeds each)
  it finds the true K in 160 of 160; the prototype's rule in 87 of 160, and a
  flat profile's K = 1 in none. Waiting for your confirmation.
- **Merge cost.** Plain Ward; the prototype's border-length term is an option,
  off.
- **Columns in steps 1–6.** Columns with cortex voxels on streamlines that
  exit into fluid, not cut by missing tissue, and with at most half of their
  streamlines fused (see **Columns and depth frame**). Changed by #21: with
  the step-0 seams as fluid and the fused rule fixed, V1V2 keeps 11,922 of
  21,034 columns and 68.5 % of its cortex label (it was 9,402 and 56 %; the
  first plan had 66 % of the cortex without missing tissue). What is still
  out: excised 14.6 %, columns with no exit, the crop, D_max, and the banks
  whose seam lies beyond their ridge. A D_max of 6 or 8 mm does not raise the
  share.
- **Column width.** 0.7 mm (from the prototype), to be swept in the
  evaluation (`column_mm`). The prototype's own sweep (0.5 to 2 mm, raw FLASH,
  a two-group test) found no clear optimum.
- **Min-cut solver.** scipy, not PyMaxflow (GPL); same depths on random
  regions. Waiting for your decision on PyMaxflow.

## Decided

| Topic | Decision |
|---|---|
| Pressure 2 boundaries | Both the areal borders and the laminar transitions. |
| Regions | Estimated jointly, by merging columns; the full hierarchy is the output. |
| Normalization | Smooth per-column gain/offset field against the region template; shape only. |
| Nudge | Per column, with TV smoothness across columns and a pull to geometry. |
| TV | On the intensity and on the boundary depths. Intensity TV: compare group norm against group + per-depth. |
| Bottom tests | Relative drop (still open: they react late, see step 5). |
| Number of layers | Adaptive contrast bands: K_R per region, from an optimal split (DP) of the region's pooled profile, K chosen by cross-validation. Neighbouring regions can differ in K_R; the TV on u compares them on a common depth grid; K_R is chosen again at each re-fit level. |
| Size | Coarse to fine: the region's band split, then the min-cut in a band around it. |
| Edge layers | Merged into their neighbours (fixed-K baseline only). |
| Weights | Normalized units. |
| Evaluation | Existing measures plus laminar ones; fixed against adaptive layers, K_R maps and residual-vs-K curves, band stability. |
| Implementation | Python, in this package. No LayNii dependency. |
| Surfaces | Estimated jointly with the layers: the fluid is a band, and an outer loop alternates between the model and the geometry. |
| White surface | Fixed. It anchors the frame. |
| Columns and depth | From streamlines of the distance to the white surface, independent of the pial surface; r per streamline (exit depth). |
| Data | The three echoes as one vector per voxel. |
| Seams | Fluid against tissue (topology-preserving competing growth), with costs from the fluid and band templates in place of the darkness measure. Step 0 starts from the darkness seams, as fluid (#21). |
| Noise | Diagonal over the echoes in steps 1–6; full Σ_F for the fluid. |
| Column graph | Mesh adjacency and the base-weighted shared faces. |

## Choices made in this plan (not yet confirmed)

Each with its status: confirmed (the evidence supports it as built), changed
(built otherwise; why) or still open.

- u does not feed back into the regions, boundaries or bottom. Still open:
  built so (`Refiner.refit` computes u last), not compared with feeding it
  back.
- Re-fit only at the levels N₀ / 2^j and 2–5 regions; Ward costs in between.
  Changed: a fixed list, 256 to 2 (step 1). Still open whether the levels
  below the graph's pieces can be reached (step 1).
- TV on z is cut at region borders; the gain/offset field is not. Still open:
  built so, not compared.
- The tests A and B and their defaults (D, δ_A, δ_B). Still open: they react
  late (step 5); D is 2 links of the column graph.
- The coarse-to-fine schedule and depth steps (h, w_band, the 1/200 step).
  h = 1/50 and the 1/200 step: confirmed as built (tests 1 and 6 pass; no
  sweep). w_band: changed to the window of each surface (step 4).
- The sparse-layer rule (f_sparse) and the common-band rule (p_common).
  f_sparse: confirmed (`test_bands.test_fixed_bands_merge_sparse_edges`).
  p_common: changed, not used (templates normalized over all the region's
  voxels, step 3).
- The band DP works in relative depth (each column's own m), so a band
  boundary sits at the same fraction of thickness in every column before
  the nudge. Confirmed, with r per streamline: exact on the flat slab
  (`test_columns`); on V1V2 r > 1.2 for 1.5 % of the cortex without seams
  and 3.2 % with them, against 7 % with r per column.
- Joint surfaces: the min-cut cut placement (step 7.3) is kept. On V1V2 with
  the echo-2 darkness it removed the ladder seams at (69, 65, 69) and
  (68, 60, 71), and the pial solid had 21 digital handles against 23 without
  it. (The genus of the trimmed pial mesh rose from 34.5 to 38, but from
  boundary loops at the trim, not from handles.) Confirmed for step 0: used
  in `joint.initial_seams` (616 cuts placed on V1V2); still open for step 7,
  with Λ as the cost.
- Joint surfaces: the form of the growth costs (β), the thickness prior in
  the growth (γ, T̂_c), the region level for step 7 (N_geo), the stopping rule,
  the streamline stops (ridge, D_max), and equidistant r first. β, γ, N_geo
  and the stopping rule: still open (step 7 not built). D_max = 4 mm:
  confirmed (6 or 8 mm did not raise the cortex in the model on V1V2). The
  ridge stop: still open (with seams, the barrier finding). Equidistant r
  first: equivolume r is an option now; the default stays equidistant until
  the overcorrection (**Columns and depth frame**) is understood.
- The cross-validation split into random halves of columns, with the held-out
  columns' gain and offset re-fitted to the training template; K_max = 8;
  t_min = 0.05. The split into halves of columns: confirmed (test 6 passes).
  The re-fit: changed (step 3b). K_max = 8 and t_min = 0.05: still open (no
  sweep; V1V2's regions use K = 4 to 6, so K_max does not bind).

## Literature and ideas to try

From a literature search on search results and abstracts, not on the full
papers: each source is to read before acting on it. Reviewed against the
plan: the decisions taken (equivolume r first, the T² test, both seam
methods compared, the K-rule comparison in the evaluation) are in **Status**.

- **K choice (step 3b).** Our cross-validation over-selected K, and the 5 %
  floor we added because of it, match what Pein & Shah (arXiv:2112.03220)
  found with squared-error cross-validation for change-point models: it can
  under- or over-estimate the number of changes; they propose an
  absolute-error loss or modified hold-out sets (R package
  `crossvalidationCP`; checked on the abstract). To compare in the
  evaluation sweep (required, Next 6): an absolute-error held-out loss, and
  mBIC as in cghseg (Picard et al., Biostatistics 2011). The group fused lasso (Bleakley & Vert 2011) is the
  convex analogue of the shared-band split. Arlot & Celisse (2010) survey
  cross-validation when the noise varies (as σ_c does here).
- **Test A (step 5) and a level of the merge tree.** The Mahalanobis distance
  with a Hotelling T² test between neighbouring blocks of profiles
  (Schleicher, Amunts, Zilles et al. 1998–99: the Mahalanobis distance was
  the most sensitive measure, and significant maxima mark borders; checked on
  abstracts and later papers that use the method). Waehnert et al. 2016
  (NeuroImage 125) is an MRI framework of equivolume layers and quantitative
  T1 profiles; that it uses T² is not confirmed by its abstract. Decided
  (Next 2): test each WM bin on its own with it, and report for each merge
  whether the T² between the two regions is significant.
- **Depth frame.** Waehnert et al. 2014 (NeuroImage 93) found that
  equivolume layers (Bok's principle) fit the myelinated bands of ex vivo
  cortex better than Laplace or equidistant ones (checked on the abstract).
  Decided (Next 1): equivolume r before step 7. Implementations: nighres,
  highres-cortex (Leprince 2015), LayNii `-equivol`; ours follows the
  streamlines (the prototype's flow-divergence method). Done as an option;
  the first comparison on V1V2 is not conclusive (see **Columns and depth frame**).
- **Seams.** Osechinskiy & Kruggel 2012 (Int J Biomed Imaging 2012:870196)
  compute the field of an inhomogeneous dielectric layer of grey matter round
  conductive white matter, on ex vivo images of 0.25 to 0.35 mm; that the
  weighting is by the GM probability, and that it puts the collision ridge
  on the true sulcal centreline, is not confirmed from the abstract.
  Decided (Next 3): weight D's path cost by Λ, compared with the seams as
  barriers. Compare step 7 with CRUISE/ACE (Han et al. 2004).
- **Regions.** To try: polish a Ward level with α-expansion and template
  re-fits, with label costs (Delong et al., IJCV 2012; PEARL, Isack & Boykov
  2012), so that the columns at region borders can move.
- **Step 4.** The cost decomposition inside a region is Haeker et al.,
  IPMI 2007: to cite. LOGISMOS (Yin et al. 2010) could couple the two banks
  or the region borders; check the Iowa patent (US 8,571,278) first.
