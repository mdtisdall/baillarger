# τ on a coarse field mesh behind Σ

**Goal.** Represent the laminar depth τ as a field on a basis that is
coarser than the voxels, behind the explicit top surface Σ of
`docs/plans/top-surface.md` (branch `feature/top-surface`, PR
mdtisdall/baillarger#222). The voxel data (n̂, the labels, later the
contrast) stays at full resolution. Only the unknowns of τ become
fewer.

**Status.** A design proposal and one local prototype
(`prototypes/scripts/tau_coarse.py`, branch `feature/tau-coarse`, commit
45fe574, not pushed). Every open choice is a question for the user (see
*Open questions*). The convergence experiments wait for a Σ in the
cortex and for the user's choice about the scale bias (see
*Dependencies*).

## Decisions so far (user, 2026-11-19)

- **τ is a field.** Mesh points on the flow lines of τ have failed
  several times. Straight prism stacks cross or collapse where the
  columns bend. A mesh that bends with the columns stores τ in its
  geometry, not as values at a resolution. Both forms are out.
- **The open question is the basis.** Which basis holds τ, coarser than
  the voxels and not tied to the voxel lattice.
- **Σ stays an explicit surface.** Contacts are found on Σ, never in the
  field. Both are needed: the continuous Σ and the discrete τ behind it.
- **The τ basis must not couple the two sides of a contact.** Each basis
  function has one copy for each connected piece of the domain in its
  support.
- **The user expects contrast to be smooth over wide patches**, wider
  than 10 voxels (1.6 mm at 0.16 mm) in many places, and more along the
  laminae. So τ can be coarse.
- **Test an irregular τ mesh.** The T5 session moved to a coarse Σ
  first, refined once the rough shape of τ is known (finding 9). The
  user wants the same idea tested here with an irregular mesh for τ:
  nodes where the size map asks for them, not on a regular lattice and
  not on the flow lines of τ.

## What the voxel solver work found (2026-11-19)

The global-τ session repaired and measured the voxel τ of T5 on site 1
(65³ voxels, 0.16 mm). These results set the design below. Commits are
on `feature/top-surface` (PR #222). The offset series is in
`scratch/top-surface/site1-grow/offset-sw-series.log`, with its τ files
next to it. A T5 run with the new stencil is in progress (`t5-sw/`).

1. **Signed distance (4b27f68).** The stop ramp, the inside test and τ0
   use the signed distance s to Σ. E_stop jumps in 7 of 80 small steps
   (before: 22), and the largest jump is 15 (before: 218). The rest come
   from faces with very small altitudes, and from the open crop edge.
   *For the mesh:* s is the input of the cut quadrature (M3) and of the
   stop term. Keep it.
2. **Domain membership.** A voxel enters the domain at s = 0 with all its
   residuals at full weight, so E_τ at a frozen τ jumps. The clip of θ
   at 0.05 is not the cause (ablation). Most voxels enter with a small
   change. A few outliers give 600–2,000 each. The fade-in by s
   (c5b0066) is coded and off. A cut-cell form (a weight for each voxel
   from its volume fraction inside Σ) is consistent, but on a voxel τ
   it needs ghosts from both sides and new stencils. *For the mesh:* on
   a basis, the cut-cell form is only a set of quadrature weights (M3).
3. **Decoupled corner cells (fixed in c5fb5b1).** On an axis with a
   ghost on one side and a real neighbour on the other, the old stencil
   (τ_R − 0)/((1 + θ)h) left out the cell's own τ. At offset 0, 401 of
   the 494 tissue cells had this on all three axes, so their τ was
   free (down to −0.39 mm where it should be about −0.003 mm). The
   three-point (Shortley–Weller) derivative couples them again. Median
   thickness at offset 0 went from 0.388 to 0.216 mm, and E_τ from
   38,747 to 77,732. The old low E_τ came from the free cells, whose τ
   made the columns look thick. *For the mesh:* the
   prototype samples τ at the voxel centres and then applies the voxel
   stencils, so its partial runs had the same fault. An analytic
   gradient of a basis always contains the local coefficients (M4).
4. **Scale bias of the scaled terms.** In `lm_scaled`, align, smooth and
   equivolume are not normalised by |∇τ|. A stretch of τ by k multiplies
   them by k², so they reward a flat τ in tissue. Only the pull and the
   thickness prior push back. The per-term thickening test showed this
   with the old stencil. With the new stencil, the bias is still in the
   terms, but from offset 0 to 4 h the thickness pressure outweighs it
   (finding 7). The tissue terms grow from 559 to 14,642 as the domain
   grows from 14,612 to 54,216 slots. The median thickness at 4 h is
   0.57 mm. That is about what the geometry gives: h is the voxel edge
   (0.16 mm), so 4 h moves Σ only 0.64 mm out of the white surface, and
   the columns can not be near the prior median of 2.7 mm yet. The
   solved τ is close to a distance: the median |∇τ| at the voxels with a
   central difference on every axis is 0.97, 1.11, 1.26, 1.26, 1.09 and
   0.90 at 0, 0.25, 0.5, 1, 2 and 4 h (p10–p90 at 4 h: 0.59–1.07). So
   at 4 h τ is compressed by about 10 % at the median. Part of that may
   be the bias. Whether it grows as Σ moves farther out is not measured.
   The
   lagged 1/|∇τ| weights (c5b0066, off) still diverge at 0.25 h: inside
   one round, a small patch next to Σ goes flat (|∇τ| ≈ 5 × 10⁻⁵), and
   the weight of the next round explodes. They are not needed to pass
   the barrier. The bias is open. *For the mesh:* the bias is in the
   energy, not in the representation, so the mesh has it too. The
   prototype's partial runs show it: with no bound on the coefficients,
   8,000–13,000 voxels went to τ > 0 and τ flattened toward 0. A coarse
   basis can not make a single small patch flat, so the divergence of
   the weights may be weaker there. That is not tested.
5. **Soft single-voxel modes.** The ten smallest eigenvalues of the
   Gauss–Newton matrix (0.2–0.57, against a median diagonal of 88) each
   sit on 1–7 inner (WM) voxels with ghost sides. These voxels have the
   pull term only, which fixes |∇τ| but not its direction. Broad
   directions are near 10⁹, so the ratio is about 10¹⁰. The LM stops by
   its tolerance on E, and the large last steps (0.02–0.06 mm) are
   moves along these soft modes. *For the mesh:* a basis with supports
   of L voxels has no single-voxel mode. Small cut pieces of a support
   are the mesh form of the same problem (M5).
6. **"Not converging" was partly a misreading.** The voxel LM stops when
   E falls by less than 10⁻⁶ for each unknown. A large last step does
   not show that E has not converged. Two starts end at different
   minima (E_τ 38,748 and 34,737, mostly in thickness and tsmooth).
   *For the mesh:* the comparison of convergence uses E, the time, and
   the difference between two starts, not the size of the last step.
   My earlier note to the solver session judged the partial runs by the
   step size. That reading was wrong.
7. **The barrier near the white surface was an artefact.** At the old
   T5 result, Σ is at the white surface: the domain is a shell of 494
   tissue voxels on 14,118 inner voxels. With the old stencil, E_τ rose
   from 38.7k to 62.1k over the first 0.5 h outward, and T5's small
   steps could not cross that. With the three-point stencil, E_τ falls
   at every offset (border held, unweighted τ):

   | Offset | E_τ | E (with stop) | Median thickness | Roots at the floor |
   |---|---|---|---|---|
   | 0 | 77,732 | 73,920 | 0.216 mm | 201 |
   | 0.25 h | 66,005 | 57,780 | 0.185 mm | 26 |
   | 0.5 h | 61,247 | 46,880 | 0.185 mm | 15 |
   | 1 h | 52,363 | 23,694 | 0.235 mm | 4 |
   | 2 h | 40,742 | −17,070 | 0.363 mm | 1 |
   | 4 h | 36,764 | −80,743 | 0.573 mm | 1 |

   Mostly the thickness term falls (55.6k to 17.3k). At 1, 2 and 4 h,
   the last LM call from the warm start took no accepted step, so these
   solves are not fully converged. The plain normal offset folds Σ
   locally (319 of 14,053 faces at 4 h). *For the mesh:* the prototype
   used the T6 start Σ, at the white surface. A test of a coarse τ needs
   a Σ in the cortex (question 9). The 2 h and 4 h offsets, and their
   voxel τ, are now a usable reference. A T5 result that has moved out
   would be better, because it has no offset folds.
8. **Open crop border.** A normal offset of Σ moves its border vertices
   off the crop face, and the inside test then leaks through the gap.
   Holding the border vertices in their box face fixes it. *For the
   mesh:* the same rule applies to every Σ that a test uses.
9. **T5 descends now, but Σ is rough (T5 run `t5-sw`, in progress).**
   With the three-point stencil, E fell from about 31,000 to −4,587 in
   21 steps, with no intersections (the old run stopped at step 4). But
   neighbouring vertices of Σ move by very different amounts. The run
   splits edges down to 1 h, so each vertex sees only a few voxel-line
   crossings (θ) and a few stop-ramp voxels. E has no term on the shape
   of Σ, so the forces on neighbouring vertices are almost independent.
   This explanation comes from how E is built and is not measured yet.
   8,467 vertex moves were refused at step 21. The user chose to start
   with a coarser Σ mesh and to refine it once the rough shape of τ is
   known. *For the mesh:* the same coarse-to-fine order applies to τ
   (M6). Two parts of this plan also make the forces on Σ denser: the
   cut weights f (M3) give a force from every voxel within h/2 of Σ,
   not only from the lines that cross it, and the Z1 residuals (M4) tie
   each point of Σ to a smooth τ, which couples neighbouring vertices
   through the shared coefficients. Neither is a smoothness term on Σ.

**Conclusion.** The faults found are in two places. Some are at the
coupling of the continuous Σ and the voxel τ: membership, ghosts, the
decoupled stencil and small cells. A smooth basis with cut quadrature
removes most of these by construction. The others are in the energy:
the scale bias. A basis does not fix these, so they must be fixed for
the voxel τ and the coarse τ alike. The prototype mixed the two: coarse
unknowns, but the voxel stencils. The next prototype changes that (see
*Prototype*). The offset series adds one point: a single coupling fault
(the decoupled stencil) made a barrier in E large enough to stop T5. So
a coupling that is correct by construction matters for the descent of
Σ, not only for accuracy.

## Design

### M1. The basis

- **A. Cubic B-splines on a regular lattice (recommended).** Spacing L
  in mm, any value, with any offset from the voxel lattice. τ =
  Σ_k c_k B_k(x). The basis is C², so ∇τ and the Hessian H exist
  everywhere and are analytic. A tensor product makes the evaluation
  cheap (3 × 4 weights in 1D for each point). The prototype uses it.
- **B. Hierarchical B-splines (THB) on an octree.** Form A with local
  refinement. Each refined cell replaces its basis functions by those of
  half the spacing. The truncation keeps a partition of unity. This is
  the adaptive form of A (M6).
- **C. Trilinear (Q1) on an octree.** C⁰ only. H needs a recovered
  gradient or an interior penalty, and the equivolume and smooth terms
  are then weakest where τ meets Σ. This is the place that failed in the
  voxel form.
- **D. Tetrahedra that conform to Σ.** A permissive constrained
  tetrahedral mesher does not exist (TetGen AGPL, CGAL and gmsh GPL).
  Out.
- **E. An irregular mesh, cut by Σ (chosen: E1, user 2026-11-19).** The τ mesh does not have to conform to Σ, because Σ
  cuts it (M3) as it cuts the lattice. So no constrained mesher is
  needed. The nodes go where a size field asks for them, and they stay
  fixed in space while Σ moves. Two forms:
  - **E1. Meshfree, moving least squares (MLS) with compact C² weights
    (Wendland).** Each node has a support (a ball, or an ellipsoid).
    The MLS shape functions reproduce every polynomial of the chosen
    degree (2 or 3), and they are C² if the weights are. So ∇τ and H
    are analytic, as for A. The supports can be ellipsoids, longer
    along the laminae and shorter across them, oriented by the column
    direction n̂ of the image (an orientation, not a node path). That
    gives the anisotropy that an axis-aligned lattice can not give.
    Quadrature is the usual difficulty of meshfree methods. Here the
    quadrature points are the voxel centres (M3), so it is not a
    problem. The shape functions at the voxel centres are a fixed sparse
    matrix U (as in the prototype), with the matrices of their first and
    second derivatives. Each voxel needs one small solve (10 × 10 for
    degree 2) to build them, once for each node layout.
  - **E2. Tetrahedra of a Delaunay triangulation of the nodes** (Qhull
    through `scipy.spatial.Delaunay`. The Qhull licence is permissive,
    not GPL). Linear or quadratic elements, C⁰ only. H needs a recovered
    gradient or an interior penalty, with the same limit as C. Long thin
    tets need care to keep the elements valid.

  Node placement for E: the size field of measurement 2 (finer at
  crowns, fundi, contacts and excisions), filled by Poisson-disk
  sampling or by a few Lloyd steps, in the cortex label dilated by the
  inner layer. Copies by piece (M2) and extension for small pieces (M5)
  apply to E as to A: aggregation is the meshfree and FEM form of the
  extension. A refinement adds nodes. With MLS this changes the shape
  functions near the new nodes, so it is a projection event (M9), not
  a nested refinement as in THB.

A is already prototyped. It is the reference for E: the same energy,
cut weights, copies and extension, with a regular basis. The lattice is
fixed in space and does not move with Σ. An axis-aligned
lattice can not be coarse along the laminae and fine across them. With
cubic functions, 2–3 cells across the cortex can hold a τ that bends
smoothly in depth, so this matters less than for linear elements. The
coarseness measurement (item 2) tests it.

### M2. Copies at contacts

Each basis function has one copy for each connected piece of
(its support) ∩ (the domain). Two quadrature points are in the same
piece if a path of real neighbours joins them inside the support. A
real neighbour is a neighbour that the contact rule does not cut (the
`nbv` of the voxel solve). So a function whose support spans a contact
gets two copies, and τ on the two sides is independent. The prototype
does this with a union-find, and an exact test checks that a V-shaped τ
across a cut is held exactly.

Options for the connectivity:

- **K1. Voxel connectivity, as now (recommended).** The contact rule is
  "no voxel centre between the two sides", so voxel connectivity agrees
  with it.
- **K2. Connectivity from Σ.** A quadrature cell that Σ cuts twice is
  split into its parts. This is sub-voxel, and it would need contact
  measures at the same scale.

A piece that splits when a contact forms adds a copy. The old τ is still
in the new space (both copies take the same value), so E does not jump.
A merge of two pieces removes a copy. τ is then projected onto the
smaller space. This is an event between steps, logged as a remeshing of
Σ is now.

### M3. Cut quadrature and continuity in Σ

The quadrature points are the voxel centres. Each one has the weight
f = clip(0.5 − s/h, 0, 1), the same volume fraction as the stop ramp.
Voxels with a centre outside Σ but less than h/2 away join with f < 0.5.
The basis has values there, so they need no ghosts.

- E is a continuous function of Σ. As Σ passes a voxel centre, the
  voxel's weight changes linearly. No term switches on at once, and
  there is no clip.
- ∂E/∂V has the term (residual² × ∂f/∂s × ∂s/∂V), as the stop term has
  now.
- The data terms read n̂, coh and the labels at the voxel centres. They
  need no interpolation.
- f is exact for a flat Σ parallel to a voxel face, and an O(h)
  approximation otherwise. An exact clip of each voxel by Σ is the
  alternative. It is more accurate and costs more (question 4).

Inner (WM) voxels behind the white surface keep the pull term only, as
now. They are quadrature points, not unknowns.

### M4. τ = 0 on Σ

With a basis, τ = 0 on Σ is not exact. Options:

- **Z1. A residual at points on Σ (recommended).** At each vertex of Σ
  (or at a quadrature rule on each face): r = τ(x) / σ_Σ, with
  σ_Σ = h/4. This states "τ = 0 on Σ to within the gap", so its scale
  is a prior, not a tuned constant. It is least squares, so it fits the
  LM. The points move with Σ, so ∂E/∂V has the term ∇τ(x) · ∂x/∂V,
  which is analytic.
- **Z2. Nitsche's method.** Consistent for a quadratic energy with a
  Dirichlet condition. Our energy is not the Dirichlet energy, and
  Nitsche's form is not least squares. Not recommended.
- **Z3. Weighted B-splines (WEB-splines, Höllig).** τ = w(x) Σ c_k B_k,
  with w a smooth distance to Σ, so τ = 0 on Σ exactly. The signed
  distance to a triangle mesh has kinks at the edges and at the medial
  axis, and these enter H. A smoothed w would be needed.
- **Z4. Voxel ghosts, as in the prototype.** The voxel stencils with
  ghosts at θh. This brings back the stencil faults (finding 3). Out.

The bound τ ≤ 0 inside Σ: the prototype had no bound, and τ > 0 grew
(finding 4). The options are a one-sided residual max(τ, 0)/σ_Σ at the
quadrature points, or no bound. A bound on coefficients is not a bound
on τ. R: the one-sided residual, and report how many points it holds.
Measure this again after the scale bias is fixed, because the bias
caused most of the τ > 0.

### M5. Small cut pieces

A copy whose piece has a very small total weight is badly determined.
In the prototype at L = 8, 4,219 pieces had fewer than 8 voxels. Before
the fit ridge and the damping floor (`FIT_RIDGE`, `DFLOOR`), the fit of
τ0 reached 3 × 10¹⁸ mm. These two constants hide the symptom. They do
not fix it. Options:

- **X1. Extension (extended B-splines of Höllig, or the aggregation of Badia)
  (recommended).** A copy is inner if its piece contains at least one
  whole lattice cell with f = 1. Each outer copy is not an unknown: it
  is the extrapolation of the inner copies of a nearby whole cell in
  the same piece. The space still holds every cubic polynomial, and the
  constants go away. Some cells change between inner and outer when Σ
  moves. Outer to inner adds an unknown (no jump). Inner to outer is a
  projection event.
- **X2. Ghost penalty (Burman).** A penalty on the jumps of derivatives
  across the faces of cut cells. On a C² basis only the third
  derivatives jump, so the penalty must be on those. It adds a weight,
  which is a constant to set.
- **X3. Keep the ridge and the damping floor.** Not recommended.

### M6. Size and refinement

- **Uniform L first.** L = 2, 4, 8 voxels, and L = 1 as the reference
  at the voxel scale in the same discretisation (question 6).
- **Local refinement (THB) later.** Refine where the size map (item 2)
  asks for it: crowns and fundi (curvature), contacts (both sides fine
  near the seam), excisions and abrasions (fast changes of thickness),
  and a large change of ∂E/∂V across a face of Σ. A refinement keeps
  the old τ in the new space, so E does not jump. A coarsening is a
  projection event.
- **The number of bands (user question, 2026-11-19).** The bands are a
  property of the contrast as a function of τ, not of τ. τ is smooth
  through a band. So τ needs depth resolution only where τ must bend
  between band edges, which a contrast term will show. With an
  axis-aligned basis, refinement in depth also refines along the
  laminae. Options: D1 no depth adaptivity, D2 THB refinement where the
  band count needs it. R: D1, then D2 if measurement 6 shows a need.
- **A multi-scale form (user question, 2026-11-19).** A hierarchical
  B-spline basis is multi-scale by construction. It helps as a
  coarse-to-fine continuation (L = 8, then 4, 2, 1: the "b" runs of the
  experiment) and as a multigrid preconditioner. The hierarchy of the
  laminar pattern over the cortex (near regions alike) is a property of
  the contrast. It belongs to the prior of a later contrast term, with
  its scales from the variograms. It does not change τ's basis.

### M7. Energy terms

All terms are sums over the quadrature points with the weights f. ∇τ
and H are analytic from the basis.

| Term | Voxel form now | On the basis |
|---|---|---|
| align | κ coh \|∇τ × n̂\|², finite differences | the same at the voxel centres, ∇τ analytic |
| smooth | \|P H P\|² / (2σ_t²) | the same, H analytic (no one-sided differences next to Σ) |
| equivolume | wq (Δτ − 2ĝᵀHĝ)² | the same, H analytic |
| pull | \|∇τ − ĝ\|² / (2σ_p²) | the same, also at the inner voxels |
| thickness | −τ(root), trilinear in the 8 voxels | −τ(root) from the basis, exact at the root |
| tsmooth | Student-t on adjacent patches | unchanged |
| stop | Σ c f over the voxels | unchanged (it uses Σ only) |
| τ = 0 on Σ | ghosts at θh | Z1 residuals at points on Σ |

The scale bias of finding 4 is in align, smooth and equivolume in both
columns of the table. Its fix is a change of the energy, made for the
voxel solve and used here as it is.

### M8. Solver

- **LM with a Gauss–Newton matrix in the coefficients**, as in the
  prototype: M = Uᵀ(JᵀJ)U. Each residual depends on the 64 coefficients
  of its point, so J can be built directly in the coefficients, without
  the voxel Jacobian. A row of M has at most 7³ = 343 entries.
- **Linear solve.** Exact SuperLU with a symmetric ordering on site 1.
  For the whole brain: CG with a geometric multigrid on the lattice
  hierarchy, or nested dissection on the band of the cortex.
- **Σ and τ.** Alternate as T5, or one joint LM over (c, V). E is
  continuous in V (M3), so the joint form is possible. R: alternate
  first, as the voxel T5, so that the two can be compared.

### M9. Moving Σ

The lattice does not move. Σ moves through it, and these change:

- the weights f (continuous, M3).
- the points of the Z1 residuals (continuous, M4).
- the pieces of M2 and the inner and outer copies of M5. These are
  discrete. A change that makes the space larger keeps E. A change that
  makes it smaller is a logged projection event, as a remeshing of Σ is
  now.

The continuity check (item 4) measures how often the events happen and
how large they are.

## Contacts, excisions and abrasions

- **Contacts** are found and measured on Σ as now (T2, T6), with the
  gap h/4. The copies of M2 keep the two sides apart in τ.
- **Merges** are decided by the energy, not by a rule of the basis.
- **Excisions.** A zero-thickness column has no tissue points. The
  thickness prior (log t, floored at 0.02 mm) is wrong there. Each one
  goes to the review list, as in the top-surface plan.
- **Abrasions.** No special case. The stop term pulls Σ in from label 0.

## Scaling: the whole brain at 50 µm

Estimates, for the analysis only:

- Cortex: about 5 × 10⁵ mm³, so about 4 × 10⁹ voxels at 50 µm.
- Unknowns: about (cortex volume) / L³, plus the copies at contacts.
  At L = 0.8 mm (16 voxels, about 3 cells across the cortex): about
  10⁶. At L = 0.4 mm: about 8 × 10⁶. The axis-aligned lattice costs
  more than a mesh that is coarse only along the laminae. The size map
  decides which L is enough.
- With E1 and ellipsoid supports, for example 1.6 mm along the laminae
  and 0.6 mm across them: about (cortex volume) / (1.6² × 0.6 mm³),
  about 3 × 10⁵ nodes, before refinement. Each MLS shape function at a
  voxel has more nonzeros than a cubic B-spline (about 20–40 nodes in
  range, against 64), so the cost of each evaluation pass is similar.
- Evaluation: every voxel near the cortex is a quadrature point. Each
  point needs 3 × 4 1D weights and their derivatives, and 64
  coefficients. The pass reads n̂ (as the tensor coh n̂n̂ᵀ), the label and
  s, about 30 bytes for each voxel, about 120 GB for each pass. The pass
  streams blocks with a halo of 2 L (the blocked GPU rule).
- Linear solve: multigrid on the lattice hierarchy, linear in the
  unknowns. Not measured.
- Σ has more unknowns than τ at 50 µm (edges tied to h). Whether Σ's
  longest edge must stay at 5 h is a question for later.

## Relation to the columns stream (#179)

PR mdtisdall/baillarger#179 makes the columns a stored output, with
the form open. On a C² τ, the columns are the flow lines of ∇τ, traced
after the solve. The coefficients are a compact stored form of the
laminar coordinate. The column map is derived from them, not stored in
the mesh. The angle between the traced columns and n̂ is a diagnostic.

## Prototype

`prototypes/scripts/tau_coarse.py` (commit 45fe574, local):

- **Has:** cubic B-splines on a lattice of L voxels, copies by piece
  (K1), the fit of τ0, LM on the coefficients with M = Uᵀ(JᵀJ)U and an
  exact SuperLU step. Exact tests: cubics are held, a V-shape across a
  cut is held, M equals the dense product.
- **Does not have:** analytic derivatives (it uses the voxel stencils
  on τ sampled at the voxel centres), the cut weights f, the Z1
  residuals, the extension of M5, the bound on τ. It has the two
  constants `FIT_RIDGE` and `DFLOOR`.
- **Partial runs** (site 1, T6 start Σ, old stencil, scale bias):
  voxel LM 5.24 × 10⁶ → 176,192 in 20 iterations. L = 8: 529,259 →
  325,906 after 86 iterations. These numbers are not valid for a
  comparison: the stencil decoupled most tissue cells, the energy
  favoured a flat τ, and Σ was at the white surface.

The next prototype: replace the voxel stencils by analytic derivatives
(M7), add f (M3), Z1 (M4) and the extension (M5), and remove the two
constants. Then add the irregular basis E1 behind the same interface
(a sparse U and its derivative matrices at the voxel centres), so A and
E1 run on the same energy, Σ and solver. A test then checks that at L = 1 the energy is close to the
voxel energy with the three-point stencil.

## Dependencies

1. **The scale bias (finding 4).** Fix it in the energy, or measure with
   it. It no longer blocks T5 near the white surface, so it does not
   block a first comparison. But a comparison with the bias measures how
   fast each form reaches a biased minimum, so the report must say so
   (question 12).
2. **The voxel baseline.** Done: the offset series with the three-point
   stencil (finding 7). Its 2 h and 4 h τ files are the voxel reference
   until a T5 result in the cortex exists.
3. **A Σ in the cortex** for the tests (question 9). The T5 run with the
   new stencil (in progress) shows whether T5 now moves Σ out.

## Measurements

Diagnostics for a person to read, not pass/fail criteria.

1. **Noise floor.** The voxel τ from two starts. Their difference
   bounds how well any coarse τ can be judged.
2. **Coarseness.** A size map from τ's Hessian (L ≈ √(8 ε_τ / |H|),
   tangential and in depth), and fits of a converged τ at L = 1, 2, 4,
   8 voxels: the error in τ and in the direction of ∇τ, by region
   (banks, crowns, fundi, contacts).
3. **Small pieces.** The number and the weight of the outer copies at
   each L, and the condition of M with and without the extension.
4. **Continuity.** Σ moved 80 times by 0.0005 h along its normals (the
   border held): E, the analytic slope against the measured slope, and
   the events of M9.
5. **Convergence.** Voxel LM against L = 8, 4, 2, 1, and coarse to fine
   (8 → 4 → 2 → 1): E against the time, and the difference between two
   starts. One run at a time, under `scripts/memguard`.
6. **Contrast.** Variograms of the echoes along τ's level sets, and the
   band count of each column. These test the premise of a coarse τ and
   size M6.

## Open questions

**Answers for the E1 prototype (user, 2026-11-19).** Accept every
recommendation below for this prototype (questions 2–8, 10, 11, 13 and
14). Question 12: compare now, with the bias in both forms, and label
the results as biased. Question 9: run E1 through the same T5 evolution
as the T5 session (Σ moves, τ solved on E1 behind it), not only at a
fixed Σ. So the E1 τ must give T5 its ∂E/∂V (M9).

Each question has options and a recommendation (R). No choice is made
before the user answers.

1. **Basis** (M1). Answered (user, 2026-11-19): E1, meshfree MLS with
   C² weights, on an irregular set of nodes cut by Σ. A (the B-spline
   lattice, already prototyped) is the reference on the same energy and
   the same Σ.
2. **Copies** (M2). K1 voxel connectivity, K2 from Σ. R: K1.
3. **τ = 0 on Σ** (M4). Z1 residuals at h/4, Z2 Nitsche, Z3
   WEB-splines. R: Z1.
4. **Cut weights** (M3). The ramp f from s, or an exact clip of each
   voxel. R: the ramp first, then compare on site 1.
5. **Small pieces** (M5). X1 extension, X2 ghost penalty, X3 the
   constants. R: X1.
6. **Reference.** Compare with the voxel LM, or with L = 1 in the same
   discretisation, or both. R: both. L = 1 separates the effect of the
   coarseness from the effect of the discretisation.
7. **Bound τ ≤ 0** (M4). A one-sided residual, or none. R: the residual,
   measured after the scale bias is fixed.
8. **Solve** (M8). Alternate (T5) or joint. R: alternate first.
9. **Σ for the tests.** The old T5 result (at the white surface), an
   offset of it with the border held (2 h or 4 h, with the voxel τ of
   the offset series), or the result of the T5 run with the new stencil.
   R: the new T5 result if it moves Σ into the cortex, because it has
   no offset folds. Otherwise the 4 h offset.
10. **Depth adaptivity** (M6). D1 or D2. R: D1, then D2 from the band
    count.
11. **Branch.** Keep `feature/tau-coarse` local until the next prototype
    works, or push it as a draft PR. R: local, as the user said.
12. **Scale bias in the first comparison** (Dependencies, item 1).
    Wait for a fix of the energy, or compare the voxel and the coarse
    forms now, with the bias in both. R: compare now, on the same
    energy, and label the results as biased. The comparison is about the
    representation, and the bias is the same in both forms. Repeat it
    after the fix.
13. **Supports of E1.** Balls, or ellipsoids oriented by n̂. R: balls
    for the first test, ellipsoids after the size map shows how much
    the two directions differ. A ball needs no orientation field, so
    the first test has one cause less.
14. **Resolution of Σ and τ.** The T5 session starts Σ coarse (finding
    9). τ's nodes can follow the same schedule (coarse, then refined
    where the size map asks), or stay at one size. R: the same
    coarse-to-fine schedule, with τ's node spacing never finer than Σ's
    edge in the first stage.

## Tests

Exact tests where possible, on real patches of site 1 for anything
past the basis:

- the basis holds every cubic, and the analytic ∇τ and H equal those of
  the cubic.
- a V-shape across a cut is held with two copies, and not with one.
- the extension holds every cubic, with the outer copies removed.
- E is continuous in Σ on a patch as one voxel centre crosses Σ, and
  ∂E/∂V matches finite differences.
- J in the coefficients against the dense Jacobian.

## Rules

- No GPL or LGPL dependencies.
- Always 3D.
- Outside medium, never CSF.
- Diagnostics for a person to read, not pass/fail criteria.
- Heavy runs under `scripts/memguard`, one at a time, with a cap of
  about 1.5 × the peak.
- No code or test names `scratch/`: paths are arguments.
- Each new open choice goes to the user.
