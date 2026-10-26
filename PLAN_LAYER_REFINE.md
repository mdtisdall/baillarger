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

with a per-echo offset a_c and gain s_c (vectors of 3), vector templates
μ_R,k, and a 3 × 3 noise covariance Σ_c from the residuals inside the cells.
The data term is the Mahalanobis form of the scalar one. The gain/offset
field and the templates are fitted per echo as in steps 2 and 3. The merge
costs (step 1) and the band DP (step 3b) use the same Mahalanobis residual:
the prefix sums hold Σ x̂ and Σ x̂ x̂ᵀ for each grid bin.

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
- **Distance.** D(v), the distance from the white surface (mm), along paths in
  the domain (not through the WM).
- **Streamlines.** From each white-mesh vertex, along the gradient of D
  (smoothed by 1 voxel), outward. A streamline stops at the ridge of D (where
  streamlines from facing banks meet: in fused cortex, the medial surface
  between the two banks), at missing tissue, at the crop faces, or at
  D = D_max (default 4 mm, about twice the thickest cortex).
- **Columns.** Patches of white-mesh vertices about 0.7 mm across. Each voxel
  goes to the column of the nearest streamline that reaches it, so each voxel
  is in one column. Unlike columns built from the rim, these do not depend
  on the pial surface.
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

r is equidistant in the first version. An equivolume correction of r is a
later option.

**Finding (to confirm): streamlines regenerated in the loop.** With the fixed
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
not depend on the pial surface.

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
   darkness (as the ex vivo prototype), which give T_c.
1. Steps 1–5 of the model, in the fixed frame.
2. Step 7, with the regions of the finest re-fit level whose regions all have
   at least N_geo columns (default 50), so that each template is pooled over
   many columns.
3. Repeat from 1 until fewer than n_stop voxels (default 0.1 % of the cortex)
   change label, or for it_max iterations (default 5).

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
   expensive. The re-fit runs at the levels where the number of regions
   reaches N₀ / 2^j (and at 2, 3, 4, 5 regions), and the merge costs are then
   recomputed. Before the first re-fit, the boundaries are geometric (the
   fixed-K baseline) and the gain/offset is the per-column mean and RMS.
   Each re-fit chooses K_R again for every region (step 3b), so a merged
   region can have more or fewer bands than its parts.
5. Stop at one region. Nothing chooses a level: the output is the merge tree
   plus the full results at each re-fit level.

### 2. Gain/offset field (fixed regions, templates, boundaries)

Quadratic: a sparse linear system in 2 unknowns per column (data term plus
graph Laplacian), solved with conjugate gradients. κ = 0 gives the
independent per-column least squares.

### 3. Templates (fixed gain/offset, boundaries)

μ_R,k = Σ s_c (f(v) − a_c) / Σ s_c², over the voxels of region R in band k.
Then centre and scale μ_R to zero mean and unit norm over the common bands,
and change a_c and s_c to match, so the model is unchanged.

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
   the split for each K on A; on B, re-fit each column's gain and offset to
   A's template (2 parameters per column) and measure the residual; swap A
   and B and add. K_R is the K with the least held-out residual.
4. Z_R = the best split for K_R on all columns.

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
min-cut in a band of ± `w_band` (default half of the narrowest band of the
region) around Z_R, with a depth step of 1/200 of the thickness. The
fixed-K baseline uses the same two levels, with the 13-layer bins as the
coarse level.

### 5. Bottom (pressure 2), for each region at each re-fit level

The profiles are extended into the WM, in steps of one depth-grid bin
(normalized with the column's own a_c, s_c). The WM part of the profile is
segmented with the same DP as step 3b, continuing the region's bands, so
a band can also reach below the GM/WM border. For each candidate bottom b
(the WM depth down to b included):

- **Test A, areal border.** For each neighbouring region R', over the
  columns within `D` (default 2 column widths) of the R–R' border: the
  separation of the two groups of profiles (between-group / within-group
  variance). It fails when it drops by more than `δ_A` (default 0.1) from its
  value with no WM layers.
- **Test B, laminar transitions.** The fraction of variance of the region's
  columns that the extended template explains. It fails when it drops by
  more than `δ_B` (default 0.1) from its value with no WM layers.

b_R is the deepest b for which every depth from 0 to b passes both tests.
Only columns that have WM voxels down to b vote; columns without WM voxels
take their region's b_R.

### 6. Intensity TV (for each output level)

Chambolle–Pock with diagonal preconditioning and adaptive steps, as in
LayNii's `LN2_COLUMN_SMOOTH_TV_NORM`, with the per-depth term added (its dual prox is
a clip of each entry) and the grid maps P_c in the difference operator.
u does not feed back into steps 1–5.

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

## Layout and dependencies

In `src/baillarger/` (names are working names), with the tests in `tests/`:

- the topology core: simple points, Euler characteristic, well-composed
  repair, topology-preserving growth, the fluid-against-tissue competition,
  seam thinning and the min-cut cut placement;
- meshes: marching tetrahedra, triangle intersection tests, streamlines;
- the phantoms;
- data: patch crops, rim labels, the three echoes;
- the model: regions, gain/offset, templates, band DP, optimal surfaces,
  bottom, intensity TV (tests 1–6); see **Model implementation** for the
  modules and the order;
- the joint surfaces: frame, step 7 and the outer loop (tests 7–10);
- the evaluation sweep.

Dependencies (through uv): numpy, scipy, nibabel, scikit-image,
scikit-learn, numba. Min cuts use `scipy.sparse.csgraph.maximum_flow`;
PyMaxflow (from PyPI) only if that is too slow for the optimal surfaces.

## Model implementation

Steps 1–6 port the ex vivo prototype's `layer_refine.py` (one echo, LayNii
columns), generalized to E echoes on the frame's columns and depth. With
E = 1, each ported function must give the prototype's outputs on the same
synthetic inputs; its LayNii-specific parts (column splitting, fused-zone
files, the equivolume metric) are not ported.

| PR | Module | Contents | Tests |
|---|---|---|---|
| A | `columns` | model columns, column graph, voxel table (column, r, echoes), noise per column and echo, depth-grid sums | slab depth and weights by brute force; no links across P1's fused banks |
| B | `bands` | band DP for every K ≤ K_max with t_min; K by 2-fold cross-validation; fixed-K baseline | test 6 |
| C | `gain` | gain/offset field (sparse solve, 2E unknowns per column); templates | tests 2, 5 |
| D | `surfaces` | optimal-surface min-cut per region, coarse to fine | test 1 |
| E | `regions` | Ward merge loop, re-fit levels, merge tree | test 4 (scikit-learn) |
| F | `bottom` | WM extension of the bands, tests A and B | synthetic bottoms |
| G | `tv` | intensity TV (Chambolle–Pock), group and per-depth terms | test 3 |
| H | `refine` | one run on a patch, the outputs above | end to end on a phantom; V1V2 |

Decisions for the model:

- **Noise.** Diagonal over the echoes: σ_c is a variance per column and echo.
  The fluid's Σ_F in step 7 stays a full 3 × 3 covariance.
- **Column graph.** Mesh adjacency and the base-weighted shared faces (see
  **Data and terms**). On V1V2, links between columns more than 3 mm apart
  along the white mesh keep 0.1 % of the weight, against 2.5 % with plain
  face counts; on P1 there is no link across the fused stretch (66 with
  plain faces). `links="faces"` keeps the plain rule for comparison.
- **K rule.** Cross-validation only (the prototype's other rules are not
  ported).
- **Merge cost.** Plain Ward; the prototype's border-length term is an option,
  off.
- **Columns in steps 1–6.** Columns with cortex voxels on streamlines that
  exit into fluid, not cut by missing tissue, and with at most half of their
  streamlines ending at a ridge inside the cortex (fused). The others wait
  for step 7. To revisit if this leaves too much out: on V1V2 it keeps 9,402
  of 21,034 columns and 66 % of the cortex (the rest: fused 20 %, crop 8 %,
  D_max 6 %; a D_max of 6 or 8 mm does not raise the share).
- **Column width.** 0.7 mm (from the prototype), to be swept in the
  evaluation (`column_mm`).

## Decided

| Topic | Decision |
|---|---|
| Pressure 2 boundaries | Both the areal borders and the laminar transitions. |
| Regions | Estimated jointly, by merging columns; the full hierarchy is the output. |
| Normalization | Smooth per-column gain/offset field against the region template; shape only. |
| Nudge | Per column, with TV smoothness across columns and a pull to geometry. |
| TV | On the intensity and on the boundary depths. Intensity TV: compare group norm against group + per-depth. |
| Bottom tests | Relative drop. |
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
| Seams | Fluid against tissue (topology-preserving competing growth), with costs from the fluid and band templates in place of the darkness measure. |

## Choices made in this plan (not yet confirmed)

- u does not feed back into the regions, boundaries or bottom.
- Re-fit only at the levels N₀ / 2^j and 2–5 regions; Ward costs in between.
- TV on z is cut at region borders; the gain/offset field is not.
- The tests A and B and their defaults (D, δ_A, δ_B).
- The coarse-to-fine schedule and depth steps (h, w_band, the 1/200 step).
- The sparse-layer rule (f_sparse) and the common-band rule (p_common).
- The band DP works in relative depth (each column's own m), so a band
  boundary sits at the same fraction of thickness in every column before
  the nudge.
- Joint surfaces: the min-cut cut placement (step 7.3) is kept. On V1V2 with
  the echo-2 darkness it removed the ladder seams at (69, 65, 69) and
  (68, 60, 71), and the pial solid had 21 digital handles against 23 without
  it. (The genus of the trimmed pial mesh rose from 34.5 to 38, but from
  boundary loops at the trim, not from handles.)
- Joint surfaces: the form of the growth costs (β), the thickness prior in
  the growth (γ, T̂_c), the region level for step 7 (N_geo), the stopping rule,
  the streamline stops (ridge, D_max), and equidistant r first.
- The cross-validation split into random halves of columns, with the held-out
  columns' gain and offset re-fitted to the training template; K_max = 8;
  t_min = 0.05.
