# τ on a coarse volume mesh

**Goal.** Put the laminar depth τ on a coarse volume mesh between the
white surface and the top surface Σ, instead of on the voxels. The mesh
conforms to Σ, so τ = 0 on Σ by construction. Σ stays the explicit
surface of `docs/plans/top-surface.md` (branch `feature/top-surface`,
PR mdtisdall/baillarger#222): the gap, the test for self-intersection and
the explicit contacts do not change.

**Status.** A design proposal. There is no code. Every open choice is a
question for the user (see *Open questions*). Some choices need
measurements first (see *Measurements first*).

**Why.** The T5 solve of the top-surface plan stalls on site 1 for two
reasons.

1. E jumps when Σ moves. The continuous Σ and the discrete voxel τ meet
   in two places:
   - domain membership: a voxel enters the domain at s = 0 with all its
     residuals at full weight (600–5,000 nats each, nearly all
     equivolume, measured on site 1).
   - the ghosts at θh: τ = 0 where an axis crosses Σ, with θ clipped at
     0.05.

   The signed distance (commit 4b27f68) removed most jumps of the stop
   term. A fade-in by s is coded, but not tested.
2. The τ LM does not converge. The last steps are 0.01–0.1 mm after 600
   iterations. A diagnosis of this runs in another session. This plan
   does not depend on its result, but the solver section uses it.

On a mesh that conforms to Σ, no voxel enters or leaves the domain, and
there are no ghosts. The nodes on Σ have τ = 0. If the mesh nodes are
smooth functions of the vertices of Σ, E is a continuous function of Σ.

The user expects contrast to be smooth over wide patches of cortex:
wider than 10 voxels (1.6 mm at 0.16 mm) in many places, and more so
along the laminae. τ then needs much fewer unknowns than voxels. The
voxel data (n̂, the labels, later the contrast) stays at full
resolution: the mesh samples it at quadrature points.

## What stays the same

- Σ, with T1, T2, T4 (its role), T6 and T7 of the top-surface plan. The
  checks of `baillarger.mesh_checks`. The contact definition and the
  contact measures.
- The energy terms and constants of `global_tau_energy.py`, in the
  scaled form (no division by |∇τ|). `W_EQ` is still provisional.
- LM with a Gauss–Newton matrix, as now (see *Solver*).
- The snapshots and the movie page. The τ field on the mesh is sampled
  on the voxel grid for the slices.

## Design

### M1. Form of the mesh

Three forms are possible.

- **A. Prism stacks (recommended).** A triangulation of the base surface,
  extruded along columns from the bottom (an inner offset of the white
  surface) to Σ. Each base triangle gives one stack of prisms (wedges).
  The top face of each stack is a face of Σ (or a patch of Σ, see M2).
  τ = 0 on the top nodes. This needs no 3D mesher: extrusion is a few
  lines of code. The mesh follows the anatomy: depth on one axis, the
  surface on the other two.
- **B. Cut cells on an octree.** A coarse octree over the volume. Σ and
  the white surface cut the cells. The energy integrates over the cut
  part of each cell. τ = 0 on Σ is a weak condition (Nitsche's method).
  The mesh does not move with Σ. Only the cut changes. The cut is
  continuous in Σ, but small cut cells make the problem badly
  conditioned and need a stabilisation term. The prism form has none of
  this.
- **C. Tetrahedra from our own mesher.** A constrained Delaunay mesh
  between the two surfaces. TetGen, CGAL and gmsh are excluded (see
  *Construction*), so this is a large new piece of code. The tets do not
  follow the depth axis, so anisotropic sizing (coarse along the
  surface, finer in depth) is hard.

The rest of this plan uses form A.

### M2. Lateral resolution: τ's top level against Σ's vertices

Σ needs its own resolution: the coarse start keeps every white vertex
within ε = h/4 of Σ, with edges L = 3/4 × √(8εR), clipped to [h, 5h].
On site 1 this gives 7,781 vertices at a median edge of 1.27 h. τ can be
coarser than that. Two forms:

- **Conforming.** Every vertex of Σ is the top of one column of the τ
  mesh. Simple: the stacks are the faces of Σ. But τ then has Σ's
  resolution. On site 1 this is about 47,000 unknowns (7,781 columns ×
  about 6 nodes), 4.7 × fewer than the 221,739 voxels, not 100 ×.
- **Nested (two scales).** The top of the τ mesh is a coarse
  triangulation, and Σ is a nested refinement of it. Each vertex of Σ
  stores its coarse top face and its barycentric coordinates in that
  face. The geometry of a coarse stack is a blend from its base to the
  fine patch of Σ above it, and τ's shape functions are defined on the
  reference prism. T7's refinement (longest-edge bisection) already
  keeps Σ nested. Collapses and flips do not, so Σ's own remeshing must
  stay inside each coarse face. Cost: the blend is piecewise (one piece
  for each fine face of Σ), so ∇τ has small jumps at the fine edges
  inside a coarse stack. Quadrature runs over the fine sub-prisms.

Estimates for site 1 (Σ's area about 10,900 h², 278 mm², and about 6
unknowns for each column, see M3):

| τ lateral edge | Columns | τ unknowns | Fewer than voxels |
|---|---|---|---|
| Σ's vertices (conforming) | 7,781 | 46,700 | 4.7 × |
| 2 h | 3,140 | 18,800 | 12 × |
| 5 h | 500 | 3,000 | 74 × |
| 10 h | 125 | 750 | 300 × |
| 20 h | 31 | 190 | 1,200 × |

At 20 h (3.2 mm) a stack is a third of the 10 mm crop. The crop then
decides more than the data, so site 1 can not test that size.

### M3. Depth: levels and shape functions

τ along a column is close to linear (the mm pull). Equivolume makes it
bend at crowns and fundi, and the contrast later moves it between the
laminae. Options for each stack:

- **Linear in depth, many levels** (for example 8). Easy, but linear
  prisms have no second derivatives inside an element, and equivolume and
  smooth need them (see M4).
- **Quadratic in depth, 2 elements** (5 nodes on each column, the top at
  τ = 0). Linear across the surface. 9 nodes on each wedge.
- **Quadratic in all directions** (18-node wedges). The second
  derivatives exist in every direction. More nodes on each column.

Below the white surface, one more layer of prisms reaches 1.5 voxels
inward (the inner voxels of the voxel solve). It carries the pull only,
as now (R1). With quadratic depth this adds 2 nodes. So "about 6
unknowns for each column" above.

### M4. Second derivatives

The scaled equivolume term (Δτ − 2ĝᵀHĝ)² and the smooth term |PHP|²
need the Hessian H of τ. On C0 elements, ∇τ can jump across an element
face, and H then has a part on the face. Options:

- **C0 interior penalty.** H inside each element, and a penalty on the
  jump of ∇τ across each inner face (Brenner's C0-IP method). The jump
  term says "no kink below the top", which is the rule that columns do
  not meet in tissue. Kinks at τ = 0 are on the boundary, so they cost
  nothing.
- **Recovered gradient.** Project ∇τ onto continuous nodal values g, then
  differentiate g. A smooth H, but one more linear map in every
  residual.
- **Flux form for equivolume.** For each element, (the net flux of
  v = ∇τ/|∇τ|² through its faces) ² / volume. This is the
  incompressibility rule in its integral form. It needs only first
  derivatives, on the faces. It does not cover the smooth term.

The ridge problem of the voxel grid (a voxel in tissue with ∇τ = 0)
can only occur in the mesh form if τ has a ridge inside an element. The
explicit Σ makes the top a boundary, as in the 1D study, so there is no
ridge in tissue at a contact.

### M5. Depth refinement and the number of bands (user question, 2026-11-19)

**Question.** The cortex is a set of laminar bands, and the pattern of
bands changes along the cortex. τ at one place probably has about 10
bands, with τ smooth between them. Can the mesh refine and coarsen in
depth, to follow the number of bands in each region?

**Answer.** Yes. In form A this is simpler than general 3D adaptivity:
the lateral structure stays one 2D triangulation, and each column has a
1D refinement in depth. Only the side faces between two neighbouring
columns need care. One point first: the bands are a property of the
contrast as a function of τ, not of τ itself. τ is smooth through a band.
But each visible band edge must lie on a level set of τ once a contrast
term is in the energy. So the depth unknowns that a column needs are
about the number of visible band edges there, plus a few for equivolume.
Where a band fades (for example layer IV in agranular cortex), its edges
no longer hold τ, and the column can use fewer unknowns. The fade is a
change of facies, not a fault: τ stays continuous.

Options:

- **D1. A fixed number of levels everywhere** (M3). No depth adaptivity.
  Simple. Correct if the measurement (*Measurements first*, item 4) shows
  that a few levels hold the profiles everywhere.
- **D2. Depth bisection in each column.** Each column has a binary tree
  of depth intervals. Where two neighbouring columns differ, the extra
  nodes on the shared side face are hanging nodes: their τ is the
  interpolation of the coarser column's nodes, so τ stays continuous. A
  2:1 rule (neighbours differ by at most one bisection) keeps the
  constraints local. A split keeps the function space nested, so E does
  not change. A merge is an event between steps, logged, as a remeshing
  of Σ is now.
- **D3. Depth degree in each column (p-adaptivity).** Each column has its
  own polynomial degree in depth (1–4). A shared side face takes the
  lower degree. Fewer data structures than D2, but a high degree can
  oscillate between band edges.
- **D4. Levels as level sets (the inverse form).** Each level of the
  mesh has a fixed τ, and the unknowns are the positions of the nodes
  along their columns. τ is then strictly monotone along every column by
  construction, so a ridge in tissue can not occur. But the energy
  becomes more nonlinear (the unknowns are geometry), and a level that
  ends in one region needs the same hanging-node rule as D2. A band that
  fades does not remove a level here: the level stays as a knot, at its
  normal spacing.

The indicator for depth refinement: before a contrast term exists, an
estimate of the error of τ in depth (its second or third derivative
along the column). After, the number of band edges that the column's
profile resolves above the noise, or the alignment residual in each
depth interval.

### M6. A multi-scale form (user question, 2026-11-19)

**Question.** The laminar patterns are probably hierarchical over the
cortex: near regions have more similar contrast than far regions. Is a
multi-scale representation of this useful to simplify τ further, or is
local refinement and coarsening better?

**Answer.** Two different things are hierarchical here, and the answer
is different for each.

- *τ's geometry.* With a nested mesh (bisection in M2 and D2), the
  hierarchy is already in the data structure. A hierarchical basis
  (or lifting wavelets on the nested triangulation, Sweldens) gives the
  same space of τ in a different basis. It does not lower the number of
  unknowns, which is already small (about 10⁶–10⁷ for the whole brain,
  see *Scaling*). It helps in two other ways: as a preconditioner
  (multilevel, so that the CG iterations do not grow with the mesh
  size), and as a coarse-to-fine continuation (solve the coarse levels
  first, then add detail). That is the user's earlier idea of growing
  from regions that are easy. A sparsity prior on the detail
  coefficients could replace the refine and coarsen decisions, but it is
  not least squares (an IRLS form, as for the Student-t term).
- *The laminar pattern.* The similarity of near regions is a property of
  the contrast as a function of (surface position, τ): a regional
  profile template. A multi-scale model fits this well: a mean profile,
  regional deviations, local deviations, with the scales from the
  variograms. It changes no part of τ's mesh. It is the prior that a
  later contrast term uses ("neighbours' contrast matches at equal τ").
  It is also the compressibility check of the unfolded image.

So the multi-scale structure is most useful in the pattern model. For
τ, local refinement on a nested mesh is sufficient, and the nesting
keeps the hierarchical basis available as a solver tool later.

## Construction (no GPL or LGPL)

TetGen (AGPL), CGAL (GPL) and gmsh (GPL) are excluded. Form A does not
need a 3D mesher.

1. **Base.** The coarse top triangulation: Σ's coarse start
   (`sigma_mesh.py`, our own code), or a coarser triangulation that Σ
   refines (M2). The base point of each column is the point on the
   white surface that its top vertex grew from. Σ already records it as
   `origin` for the fold rule.
2. **Extrusion.** The nodes of a column are on the line from its base
   point b to its top vertex V: x = b + λ (V − b), with fixed fractions
   λ for each level. The inner layer continues the line below b by 1.5
   voxels.
3. **Checks.** Every element has a positive Jacobian at its corners and
   at its quadrature points. If the boundary (the inner surface, Σ and
   the crop sides) is embedded and every Jacobian is positive, the mesh
   covers the region once, with no overlap (a degree argument). So the
   existing checks of Σ, plus the Jacobian signs, are sufficient.
4. **Libraries.** numpy and scipy (BSD), JAX (Apache-2.0) for the energy
   and its derivatives, numba (BSD), pyamg (MIT). MFEM and PyMFEM
   (BSD-3) have wedge elements of any order, and the 1D study used MFEM.
   But ∂E/∂V through the geometry is easier in JAX. scikit-fem (BSD-3)
   has wedge meshes: its element orders must be checked. SuiteSparse's
   CHOLMOD is LGPL and GPL, so it is excluded. SuperLU (in scipy, BSD) is
   allowed.

## Moving Σ

- The nodes are affine in the vertices of Σ (step 2 of *Construction*).
  The τ values stay on the nodes when Σ moves (a Lagrangian τ). So E is a
  composition of smooth maps, apart from the data interpolation. There
  are no voxels that enter the domain, and no ghosts.
- **Data interpolation.** Trilinear interpolation is continuous, but its
  derivative jumps where a quadrature point crosses a voxel plane. So E
  is continuous in Σ, with a gradient that is piecewise smooth. A cubic
  B-spline makes E twice differentiable, at 64 lookups instead of 8.
- **Skew.** If Σ moves far along the surface (the relaxation of T7),
  the line from b to V leans, and the prisms become skewed. A remap of the
  base points along the white surface fixes this. It is a remeshing
  event between steps: τ is interpolated onto the new mesh, and the
  change of E is logged, as for a remeshing of Σ now.
- **Splits.** A bisection that keeps Σ's shape keeps E (the old τ is
  exact in the new space). A collapse or a flip does not. Those are
  events.
- **A step of Σ** is accepted only if all Jacobians stay positive, in
  addition to the gap, the triangle test and the fold rule. Otherwise it
  is halved, as now.
- **Joint or alternating solve.** Because E is smooth in (τ, V), one LM
  over both is possible. T5 alternates now. See the questions.

## Energy terms on the mesh

Each voxel term becomes an integral over the mesh, multiplied by 1/h³,
so that the total is in the same nats as the voxel sum. Inside Σ the
tissue fraction is f = 1, so f drops out of the τ terms. Only the stop
term sees the boundary.

**Quadrature.** τ is a polynomial in each element, so the τ-only terms
(smooth, equivolume, pull) need only a Gauss rule of fixed order. The
data terms (align, stop, later contrast) must sample the voxel data at
about one point per voxel, or they alias. So the data terms use a
composite rule: each element is split into sub-cells of about one voxel.
This keeps the evaluation cost linear in the voxels, but the unknowns
and the linear algebra are much smaller.

**Data sampling.** n̂ has no sign, so its components can not be
interpolated. Interpolate the tensor T = coh n̂n̂ᵀ instead: the align
term is linear in it (below). The label log-odds c is a scalar.

| Term | Voxel form now | On the mesh |
|---|---|---|
| align | κ coh \|∇τ × n̂\|² | κ (coh \|∇τ\|² − ∇τᵀ T ∇τ) at the data points, T and coh trilinear |
| smooth | \|P H P\|² / (2σ_t²) | the same at the Gauss points, H from M4 |
| equivolume | wq (Δτ − 2ĝᵀHĝ)², wq = W_EQ / (2 A_owner) | the same at the Gauss points, or the flux form (M4). A_owner becomes the base area of the column |
| pull | \|∇τ − ĝ\|² / (2σ_p²) | the same at the Gauss points, also in the inner layer |
| thickness | t = −τ(root), trilinear, for each 0.5 mm patch | t = −τ at the column's base node, or τ interpolated at the patch roots (question) |
| tsmooth | Student-t on adjacent patches | the same, on adjacent columns or patches |
| stop | Σ c clip(0.5 − s/h, 0, 1) over the voxels | S1: (1/h³) ∫ c dx over the mesh, c trilinear. S2: the voxel ramp on the signed distance, as now |

**The stop term.** With S1, ∂E_stop/∂V is the integral of c over Σ
times the normal motion: Σ moves out where c < 0 (cortex) and in where
c > 0 (label 0). The equilibrium is the level c = 0 of the interpolated
labels, about half a voxel from the cortex voxel centres, the same as
the ramp. S1 needs no band, no inside test and no signed distance.

## Sizing and refinement

Sizing has three sources:

1. **τ's shape.** An error estimate from τ's Hessian: lateral edge
   L ≈ √(8 ε_τ / |H_t|), with H_t the tangential part and ε_τ a bound on
   the error of τ (in mm). This is the size map of the planned
   measurement.
2. **The data's scale.** The variograms of the contrast along τ's level
   sets: how far the pattern stays the same.
3. **Σ's shape**, only in the conforming form (M2).

Refinement is at:

- **Crowns and fundi.** High curvature. The columns fan out or converge,
  and equivolume bends τ in depth. Lateral and depth refinement.
- **Contacts.** Both sides of a contact are fine near the seam, so that
  the top faces of the two sides can follow each other within the gap.
- **Excisions.** Zero thickness (see *Contacts, excisions and
  abrasions*). The thickness changes fast near the edge.
- **Abrasions.** Σ moves out into the medium, and the thickness changes
  fast. Refine where the thickness gradient between columns is large.
- **Force variation.** As T7 now: where ∂E/∂V varies strongly across a
  face.

Coarsening merges the children of a bisection where all the sources
allow it. It is an event between steps.

## Contacts, excisions and abrasions

- **The gap.** Unchanged: two parts of Σ keep h/4 between them. The two
  sides of a contact are two separate stacks of prisms. They share no
  node, and no term couples them across the gap. Each side has τ = 0 on
  its own top.
- **Contacts** are measured on Σ as now (T2, T6): a vertex whose ray
  meets a facing part of Σ with no voxel centre between them. The
  contact measures do not change.
- **Merges.** The energy decides a merge (thickness, neighbour
  thickness), not a rule. The mesh has no rule of its own here.
- **Excisions.** A column whose white face faces label 0 has V = b, and
  its stack has zero height. Options: drop the zero-height elements from
  the integrals and fix τ = 0 on their nodes, or keep a minimum height.
  The thickness prior (log t, clipped at 0.02 mm) is wrong for a real
  zero thickness. Zero-thickness tops are valid at excisions and go to
  the review.
- **Abrasions.** No special case. The stacks become longer where Σ moves
  out into the medium, and the stop term (c > 0 in label 0) pulls Σ in.
  This is the test of T8.

## Solver

- **LM with a Gauss–Newton matrix**, as now. The residuals are at the
  quadrature points. Each element gives a dense block J_eᵀJ_e (9 × 9 for
  the 9-node wedge), assembled into a sparse matrix. No colouring and no
  jvp passes are necessary.
- **The linear solve.** The mesh is a thick surface, so its graph is
  close to 2D. A sparse direct solve with nested dissection costs about
  n^1.5 operations and n log n memory. For 10⁶ unknowns this is small.
  SuperLU (BSD) on each region. CG with the exact inverse of each
  column's block (the column-line preconditioner of the voxel solve, now
  exact) or pyamg is the alternative.
- **The LM convergence problem** (steps of 0.01–0.1 mm after 600
  iterations) is not solved by the mesh alone. Its diagnosis runs
  elsewhere. The mesh can help in three ways: fewer unknowns, an exact
  linear solve (no CG error in the step), and a smaller condition number
  (it grows with the square of the number of elements across the
  domain). This is not tested.
- **Σ and τ.** Alternation (T5), or one joint LM over (τ, V) with the
  checks of Σ as step acceptance. The joint form uses the smoothness of
  E in V. The τ block can be removed by its Schur complement, because it
  is small.

## Scaling: the whole brain at 50 µm

Estimates, for the analysis only:

- Cortex: about 2 × 10⁵ mm² of surface, about 2.5 mm thick, about
  5 × 10⁵ mm³. At 50 µm: about 4 × 10⁹ voxels, and so 4 × 10⁹ unknowns
  for the voxel τ.
- τ mesh: its size is set by the anatomy (mm), not by the voxel size.
  At a lateral edge of 0.8–1.6 mm: 0.9–3.6 × 10⁵ columns. With 3 ×
  for refinement and about 6 nodes for each column: about 10⁶–10⁷
  unknowns, 400–4,000 × fewer than the voxels.
- Σ: its edge is tied to h (ε = h/4, edges at most 5 h = 0.25 mm). That
  gives about 4 × 10⁶ vertices, 1.1 × 10⁷ coordinates. Σ then has more
  unknowns than τ. Whether Σ's longest edge must stay at 5 h is a
  question.
- **Evaluation** (each energy or Jacobian pass): one data point for each
  voxel. With T (6 floats) and c (1 float) in float32: about 28 bytes for
  each voxel, so about 110 GB read for each pass. This is more than one
  GPU's memory, so the pass streams blocks with a halo (the blocked
  structure of the GPU rule). The pass is bandwidth-bound. If the
  variograms show that n̂ is smooth over several voxels, T can be stored
  at a coarser grid, which divides the bytes by 8 for each factor of 2.
  The labels stay at full resolution near Σ.
- **Assembly:** about 16 residuals × 9² for each data point, about
  5 × 10¹² operations, a few seconds on a GPU.
- **Linear solve:** about 10¹⁰ operations for 5 × 10⁶ unknowns (nested
  dissection). Memory: about 1 GB.
- **Σ's checks:** O(n log n) for the triangle test with a tree, as now.

The cost of each iteration stays linear in the voxels (the data pass),
but the number of iterations should depend on the mesh, not on the
voxels. That is not measured.

## Relation to the columns stream (#179)

PR mdtisdall/baillarger#179 makes the refined columns a stored output
and leaves their form open. One candidate is "the map on a 0.3–0.6 mm
grid". The prism mesh is such a map: (base point, λ) → x is a discrete,
invertible map from a reference slab (the base surface × [0, 1]) onto
the ribbon. The Jacobian check makes it invertible.

But the mesh lines are not the columns of τ (the flow lines of v or of
∇τ) unless they are made to follow them. Options:

- **(a)** The mesh is only a discretisation. The columns are traced from
  τ afterwards. The angle between the mesh lines and ∇τ is a diagnostic.
- **(b)** The base points slide along the white surface so that the mesh
  lines follow ∇τ (r-adaptivity). Then the mesh is the stored columns,
  and the column map and τ are refined together (the user's idea of
  2026-11-08: probes collect evidence, columns are refined in parallel).
- **(c)** The inverse form (D4): the levels are τ's level sets, so the
  mesh is the laminar coordinate system.

The mesh coordinates (base point, τ) are also the unfolded frame for the
compressibility check and for the variograms.

## Measurements first

The planned measurement decides whether this plan is worth its cost. It
needs a converged τ, so it waits for the LM diagnosis.

1. **Size map** from τ's Hessian on site 1: the lateral edge that keeps
   the error of τ below ε_τ, mapped over the site. With ε_τ = h/4 and
   h/2.
2. **Prism fits** at L = 2, 5, 10 and 20 voxels: fit τ on prism meshes
   of each size (least squares on the voxel τ), and map the error and
   the change of each energy term. The table in M2 gives the unknowns.
3. **Variograms of the contrast** along τ's level sets: for each echo,
   at several depths, by distance along the level set. The shape shows
   how fast the pattern changes, and whether it has more than one scale
   (M6).
4. **Band count** (new, for M5): the number of band edges that each
   column's profile resolves above the noise, mapped over site 1, and
   later at a site with a clear stria (V1).
5. **Continuity** (after a prototype): the 80 moves of 0.0005 h along the
   normals, as in Steps 3 of the top-surface plan. E must have no jumps.
   The analytic slope against the measured slope.
6. **Quadrature density:** E and its gradient at 1, 2 and 3 data points
   for each voxel along each axis, to show the aliasing.
7. **Mesh against columns:** the angle between the mesh lines and ∇τ
   (for #179).

If τ does not converge soon, items 1–3 can run on the best τ that the
current solve gives, as a first view. The results then also contain the
solve's error.

## Open questions

Each question has options and a recommendation (R). No choice is made
before the user answers.

1. **Form of the mesh** (M1). A prism stacks, B cut cells on an octree,
   C tetrahedra from our own mesher. R: A.
2. **Lateral resolution** (M2). Conforming to Σ, or nested (τ coarser
   than Σ). R: nested, because conforming gives only 4.7 × fewer
   unknowns on site 1. Start with conforming as the first prototype,
   because it is simpler and it tests the continuity in Σ.
3. **Shape functions in depth** (M3). Linear with many levels, quadratic
   in depth with 2 elements, or quadratic in all directions. R:
   quadratic in depth, linear across, until the prism fits show the
   error.
4. **Second derivatives** (M4). C0 interior penalty, recovered gradient,
   or the flux form for equivolume. R: C0 interior penalty, because its
   jump term is the rule "no kink below the top".
5. **Depth adaptivity** (M5). D1 fixed, D2 bisection with hanging nodes,
   D3 degree for each column, D4 levels as level sets. R: D1 for the
   first prototype, then D2 if the band count (measurement 4) varies
   much across a region.
6. **Multi-scale** (M6). Local refinement only, a hierarchical basis for
   τ, or a multi-scale pattern model. R: local refinement on a nested
   mesh for τ now. The multi-scale model for the pattern when the
   contrast term is designed, with scales from the variograms. A
   hierarchical basis later, only as a preconditioner, if the CG
   iterations grow.
7. **Base points** (Construction). Σ's `origin` on the white surface
   (Lagrangian), or the closest point on the white surface again at each
   step. R: `origin`, because the closest point jumps at the medial axis,
   and E would then jump too.
8. **Columns in the mesh** (#179). (a) a discretisation only, (b) mesh
   lines that follow ∇τ, (c) the inverse form. R: (a), with the angle as
   a diagnostic. Decide after measurement 7.
9. **Stop term** (S1 or S2). R: S1, the volume integral of c.
10. **Data interpolation.** Trilinear or cubic B-spline. R: trilinear
    first. Cubic only if the gradient's kinks slow LM.
11. **Thickness terms.** At the column bases, or at the 0.5 mm patches.
    R: at the 0.5 mm patches, so that the priors keep their meaning when
    the mesh size changes.
12. **Excisions.** Drop the zero-height elements, or keep a minimum
    height. R: drop them and fix τ = 0, and send each one to the review
    list.
13. **Solve.** Alternate τ and Σ (T5), or one joint LM. R: joint, after
    a test that compares the two on site 1.
14. **Linear solver.** Sparse direct (SuperLU) or CG with column blocks
    or pyamg. R: SuperLU on site 1 (exact steps help the LM diagnosis).
    Analyse again for the whole brain.
15. **Σ's longest edge** at 50 µm: stay at 5 h, or a size in mm. R: ask
    after the nested form works. Σ then carries more unknowns than τ.
16. **Crop sides.** Free (natural) sides, or the rim fixed at τ0 as in
    T5 (`rim_fixed`). R: as T5, so that the two solves can be compared.

## Tests

Unit tests with exact answers where possible: the extrusion of a flat
slab gives τ steps of h with no error, the Jacobian check finds an
inverted prism, a bisection keeps E, the align term with an interpolated
tensor equals the voxel form at the voxel centres, and the stop term S1
against a numerical volume fraction. Real-data checks are on site 1 and
on the abrasion site, not on phantoms.

## Rules

- No GPL or LGPL dependencies.
- Always 3D.
- Outside medium, never CSF.
- Diagnostics for a person to read, not pass/fail criteria.
- Heavy runs under `scripts/memguard`, one at a time, with a cap of
  about 1.5 × the peak.
- No code or test names `scratch/`: paths are arguments.
- Each new open choice goes to the user.
