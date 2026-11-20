# Top surface and τ behind it

**Goal.** Find the laminar depth τ of the continuous model (top anchor,
equivolume, tangential smoothness, a weak pull toward mm) with the top as
an explicit surface. Contacts are the places where that surface touches
itself. The cost of each iteration must be linear in the number of voxels
and vertices.

**Why.** The global τ solve (PR mdtisdall/baillarger#221, report
`reports/Global sparse tau solve.md`) found two blocks on site 1:

1. On the voxel grid, a ridge of τ in tissue has a voxel with ∇τ = 0. There
   v = ∇τ/|∇τ|² of the equivolume term is not defined. This froze the
   solve from both starts (without a contact and with the contact seeded).
2. Where the labels are cortex on both sides of a fused contact, no term
   pushes τ up to 0. The ridge stays where the start put it.

In the 1D study the ridge was a kink on an element boundary, so v was
never evaluated there. An explicit τ = 0 surface does the same in 3D: the
ridge is a boundary, and τ is solved only behind it.

## Decisions

- **Two parts** (user, 2026-11-18). A τ = 0 surface Σ, grown outward from
  the white surface as the start. A τ field filled in behind Σ, which can
  move Σ, but cannot move it so that it intersects itself.
- **Σ is a pial surface** (user, 2026-11-18). This changes the earlier
  rule "no pial surface" for the column solver. The white surface is still
  judged by its normals only.
- **Abrasion** (user, 2026-11-18). There is no jump variable for missing
  laminae (the stop-jump model of 2026-11-11 is replaced). First, measure
  how far the neighbour-consistency terms push Σ out into the medium at an
  abrasion. Only if that is not sufficient, evaluate other forces there.
- **Kept from the global τ solve:** the energy terms and constants
  (`global_tau_energy.py`, `W_EQ` still provisional), the LM + CG solve
  with the coloured Jacobian and the column-line preconditioner, the
  domain margin, the inner voxels with the pull only, the thickness terms
  per patch, the snapshots and the movie page. No bank, no 5 mm relation,
  no face test.

## Design

The user's choices of 2026-11-18 (T1–T8).

- **T1. Form of Σ.** A copy of the white mesh, deformed outward. The
  no-crossing rule uses the triangle–triangle test of
  `baillarger.mesh_intersect`. Σ may be remeshed where it is much
  compressed or stretched (T7). Σ does not keep a vertex-to-vertex
  correspondence with the white surface: the τ field gives the relation
  between the two surfaces.
- **T2. Gap.** Two parts of Σ keep a minimum gap of h/4 (0.04 mm at
  0.16 mm, 12.5 µm at 0.05 mm). The triangle test's tolerance is 1e-4 ×
  the mean edge length, and the edges stay about one voxel long (T7), so
  the gap is about 2,500 × the tolerance at each resolution. "Another
  part" of Σ for a point: a face outside the point's local patch, which is
  the part of Σ inside the ball of radius h/2 that is connected to the
  point (faces reached from the point's own faces across shared vertices,
  each closer than h/2). The no-crossing rule also tests faces that share
  one vertex (a fold at the vertex), which the triangle test skips. The
  fold rule ("reference + dihedral", user, 2026-11-18): no face may turn
  by 90 degrees or more from the white face it grew from (through
  `origin`), and no two faces with a common edge may fold back past 154
  degrees (`mesh_checks.FOLD_COS`). Only new folds count; the white
  surface's own creases are counted, not removed. All these checks are
  `baillarger.mesh_checks`, shared with the WM side. Open point: on a
  synthetic slab with one label-0 voxel in the cortex, the fold rule keeps
  the neighbours of the stopped vertex as a narrow shaft that never stops
  (not TOP, and never at rest). Whether this occurs in real tissue is not known;
  the site-1 run decides (user, 2026-11-18). A
  contact is two parts of Σ with no voxel centre between them along their
  normals.
  The gap is a numerical margin, not a statement about tissue: at 0.05 mm
  a fused line can be several voxels wide, and the stop evidence then
  decides between a contact and a gap with medium.
- **T3. τ behind Σ.** The unknowns are the domain voxels whose centre is
  inside Σ (between the white surface and Σ) and the inner voxels. On an
  axis where the neighbour is across Σ, the first derivative uses τ = 0 at
  the crossing, at the distance θh (0 < θ ≤ 1) from the voxel centre. The
  second-level derivatives are one-sided there. θ depends on the vertex
  positions, so the energy has a derivative with respect to Σ. At a
  contact each voxel sees only its own side of Σ.
- **T4. Stop term.** c × clip(0.5 − s/h, 0, 1) for each voxel, s the
  signed distance from the voxel centre to Σ (positive outside). This is
  the ramp of the global solve, taken from Σ, not from τ. It is continuous
  in the vertex positions. Where c < 0 (cortex label) it pushes Σ out;
  where c > 0 (label 0) it pulls Σ in. s = ±|x − q| with q the closest
  point on Σ, exact within 1.5 voxels of Σ (`top_tau.signed_distance_band`);
  the sign is from the pseudonormal of the face, edge or vertex that q lies
  on. ∂s/∂V_i = −w_i u (w the barycentric weights of q, u = (x − q)/s).
  "Inside Σ" is s < 0 in the band; each connected part off the band takes
  the side of the band voxels next to it (user, 2026-11-19: "Signed
  distance"; it replaced the axis crossings and the line votes, which made
  E jump as Σ moved). The ghosts of T3 still use the axis crossings; a
  comment in `sigma_top` says to test them.
- **T5. Solve** (`top_solve.py`). The energy is the scaled τ energy (the
  equivolume, align and smooth terms have no division by |∇τ|) plus the
  stop term. Alternate: (a) τ with Σ fixed (LM on the scaled energy with
  ĝ = ĝ(τ), estimator "lm_scaled", warm start, the rim fixed; user,
  2026-11-19); (b) a step of Σ along −∂E/∂V, at most h/2 for each vertex,
  halved until it keeps the gap and makes no intersection and no new fold
  (FreeSurfer's rule), accepted if E falls. Repeat until Σ and τ do not
  change. Open: E_τ is not continuous in Σ (below): a voxel that enters
  the domain adds to E_τ at once.
- **T6. Start.** Expand Σ from the white surface along the vertex normals
  in steps of h/4, driven only by the stop term of T4 (out in the cortex
  label, stop at label 0). The white mesh lies inside the WM voxels, so a
  vertex in WM keeps moving; it stops as "wm" after 3 voxels still in WM.
  Each step keeps the gap and is checked for intersections and new folds.
  A move that breaks one is halved up to 4 times (h/4 to h/64), then refused for
  that step. A refused vertex stays MOVING and tries again in each step;
  `held` counts the steps in a row it was refused. There is no contact
  state (user, 2026-11-18): a state set by the path (troughs, notches of
  the white surface) is not a record of the tissue. The start ends when no
  vertex takes a step (Σ no longer changes, an equilibrium of the push and
  the constraints). The contacts are then measured from the geometry, with
  the definition of T2: a vertex whose ray along its normal meets a facing
  part of Σ (within 2 voxels) with no voxel centre between them (none whose
  projection on the ray is between the two and whose distance from the ray
  is at most half a voxel). Each contact has the facing distance and `held`
  (how long the push pressed it). In T5 the same measure applies to the
  equilibrium of the whole energy. A vertex whose
  white face faces label 0 directly does not move (zero thickness at an
  excision).
- **T7. Mesh quality.** Coarse to fine (user, 2026-11-19):
  - **Coarse start** (`sigma_mesh.py`). An adaptive remeshing of the white
    mesh. The target edge is 3/4 × sqrt(8 ε R), ε = h/4, R the curvature
    radius (smoothed over 2.5 h), at most the distance to a facing sheet,
    clipped to [h, 5 h]. Each pass: splits and collapses toward the
    target, edge flips where the two diagonals pass within ε of each
    other, a tangential relaxation, then projection onto the white mesh.
    A deviation check keeps every white vertex within ε of Σ. Every change
    keeps the gap and makes no intersection and no new fold. The crop's
    boundary does not change.
  - **Refinement in T5** (`mesh=refine`). No relaxation and no collapse.
    After each accepted step, the longest edge of a marked face is split
    at its midpoint (Σ's shape does not change), down to edges of h. A
    face is marked if the force varies strongly across it (max |g_i − g_j|
    ≥ max |g_i|), if a facing part of Σ is nearer than its longest edge,
    or if a vertex was refused 2 steps in a row or halved in this step.
  - The earlier mode (`mesh=remesh`): a tangential relaxation after each
    step, then splits of edges longer than 2 × the white median edge and
    collapses below 0.5 ×.
  No normal smoothing: the shape of Σ comes only from the τ energy and
  the stop term, so the abrasion test sees the neighbour-consistency terms
  on their own.
- **T8. Abrasion site.** Chosen later, when steps 1–3 work on site 1.

## Watching the optimization

The user wants to see how Σ and τ evolve, to understand the behaviour of
the optimization (2026-11-18). Every run can write snapshots
(`snapshots=true`), and a movie page shows them while the run goes on.

- **Snapshots.** One after each step of the start (T6), one before the
  first and after each accepted LM iteration of each τ solve, and one after
  each Σ step (T5) and each remeshing (T7). Each snapshot has τ in the
  domain, Σ (vertices and faces, because remeshing changes them), the
  displacement of each vertex in that step, the contacts, and the energy
  by term.
- **Frames.** Three slices through the site: echo 2, τ in colour, the white
  surface and Σ as cut lines (blue and red), the contacts in cyan. The
  title gives the snapshot name, the energy and the number of contacts.
- **3D view.** Beside the slices, Σ in the crop, coloured by how far each
  vertex moved in that step, with the contacts marked. It can be rotated,
  and it keeps its angle from frame to frame.
- **Page.** One page with a slider, play, and "follow the newest", as for
  the global τ movie, served on 127.0.0.1. The 3D view uses three.js (MIT),
  loaded with an exact version.

## Steps

1. Σ start (T6) on site 1 (`scratch/tube-sites/site1-fixed/c10mm/site.npz`):
   the contact area, where it is, and the cost. Compare with the 8,851
   contact faces of the first global run.
2. τ behind a fixed Σ (T3, T4): the τ solve on the shell. Measure the
   largest |v|, the cortex voxels outside Σ, the topology of {τ < 0}, and
   the cost.
3. Alternation (T5, T7) on site 1: how far Σ moves, the contacts after
   each outer iteration, intersections (always 0), energy by term.
   Current state, site 1 (`site1-grow/c10mm`):
   - The coarse start has 7,781 vertices (white mesh: 37,286). The median
     edge is 1.27 h: the deviation check stops most collapses before the
     target (up to 5 h). Every white vertex is within h/4 of it. No
     interior face has an altitude/longest-edge ratio below 0.46 (p1).
     All checks are 0. ∂E/∂V = 0 at 10% of the vertices, 83% of them on
     the crop boundary (white mesh: 32%).
   - E at the coarse start is 40,620; at the white mesh it is 25,783. 399
     voxel centres within h/4 of the white surface change side and
     switch on the align, equivolume and smooth terms.
   - The lagged τ solve does not converge on the coarse start: a
     two-cycle (at 86% of the voxels that move, the next lag moves them
     back; the largest |Δτ| stays at 0.295 mm from 10 to 300 lags; E
     changes by thousands from lag to lag). The lagged step freezes ĝ and
     is not a descent step on the scaled E.
   - LM on the scaled E (lm_scaled) descends: E_τ on the coarse start is
     36,456 after 10 iterations, 34,955 after 30 and 34,908 after 100
     (lagged: 44,600–46,500). The equivolume term falls from 8,105 to 106.
   - T5 from the coarse start with lm_scaled (30 iterations per solve):
     3 accepted steps (E 30,953 → 30,381), then no decrease in 8 tries,
     down to steps of 0.002 h. The force indicator marks about 2,250
     faces; few edges are 2 h or longer (0, 0, 2 splits).
   - ∂E_stop/∂V is right: along a uniform move of 0.0005 h on the normals,
     the analytic slope is −7.12 per step and the measured median is
     −7.24. But E is not continuous in Σ:
     - E_stop jumps in 22 of 80 such steps (the jumps sum to 969, the
       largest 218). Causes: crossings that enter or leave the one-voxel
       window (|Δf| ≈ 0.33 each), changes of the best axis, and the inside
       vote. A move of 0.0005 h flips "inside" at 12–601 voxels, mostly
       far from Σ (of 601, 4 have a crossing within one voxel): the vote
       is taken along whole grid lines. No vertex lies on a grid line.
     - E_τ falls for steps in both directions (−16 to −77 at 0.00125–0.02
       h), so it is not smooth in Σ either; its domain is the inside
       voxels.
     - Along −∂E/∂V, E rises (+26 to +54); along +∂E/∂V it falls (−42 to
       −460): the jumps, not the slope, decide each step.
   - With the signed distance (T4), on the T5 result Σ, the same 80 moves
     of 0.0005 h along the normals:
     - E_stop jumps in 7 steps (sum 36, largest 14.9; before: 22, 969,
       218). Analytic slope −7.18 per step, measured median −7.19. 6 of the
       7 are sign flips (|s| 1.0–1.4 h, f 1 → 0) at vertices of a cluster
       of very small faces (edges 0.06–0.3 h, altitudes 0.02–0.05 h) in
       the T5 result; 1 is at the open crop edge. 21 faces of 14,053 have
       an altitude below 0.01 h.
     - The band has 29,463 voxels; no connected part off the band has
       neighbours on both sides. The old line votes disagree with the new
       inside test at 320 voxels.
     - E_τ at a frozen τ: each new domain voxel (τ = s ≈ 0) adds 600–5,000,
       all in the equivolume term. It has 3 ghost sides with θ clipped at
       0.05. From α = 0 to +0.02 h, E_τ rises from 34,737 to 335,441.
     - E_τ with τ solved again (2 × 30 LM iterations, warm) at steps of
       0.0005 h from −0.002 h to +0.006 h: equivolume stays at 124–162,
       but E_τ rises by +8 to +418 per step (mostly the thickness term),
       larger in steps where more voxels enter (11–16 voxels: +177 to
       +418). The analytic ∂E_τ/∂V (through θ) predicts −11.8 per step.
       The LM last steps are 0.007–0.097 mm, so part of this is the solve.
4. The abrasion site (T8, chosen later): how far Σ goes into the medium,
   against the distance its neighbours give.
5. Report: what worked, what failed with exact numbers, open questions.

## Measures

For a person to read, not pass/fail:

- contacts as explicit events: area, position, and the contact measures
  of `global_tau_contacts.py` (label 0 near the face, thickness outlier,
  profile correlation);
- the disagreement at each contact: where each side's energy alone would
  put Σ, as a signed distance (positive = a gap, negative = the two
  sides' evidence disagrees; never read as overlap);
- intersections of Σ (must stay 0) and the minimum distance between parts
  of Σ that are not contacts;
- isolated TOP vertices (user, 2026-11-18): a review list of the vertices
  stopped at label 0, most isolated first, so that a person can correct
  label errors as they appear. For each: how far its neighbours moved past
  it, the size of the label-0 region that stopped it, whether that region
  reaches the crop edge, and the label-0 voxels around the stopping voxel.
  A TOP vertex is final in the start; a small region with a large lag is a
  likely label error in the cortex;
- equivolume: the largest |v| and the share of the energy in the top 1% of
  voxels;
- thickness per patch; at the abrasion site, the thickness against the
  neighbours' median;
- cost of each step and each outer iteration; memory peak.

## Tests

Unit tests on small synthetic arrays and meshes, with exact answers where
possible: the crossing distance θ and its derivative with respect to a
vertex; the stop term against a numerical volume fraction; the
no-crossing step on two facing planes (they stop with no intersection and
become a contact); the τ fill on a flat shell gives τ steps of h; the
start on a slab stops at label 0. Real-data checks are on site 1 and the
abrasion site, not on phantoms.

## Rules

- No GPL or LGPL dependencies.
- Heavy runs under `scripts/memguard`, one at a time, the cap about 1.5×
  the peak.
- Outside medium, never CSF.
- No code or test names `scratch/`: paths are arguments.
- Each new open choice goes to the user. No provisional decisions.
