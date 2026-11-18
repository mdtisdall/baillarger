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
  the gap is about 2,500 × the tolerance at each resolution. A contact is
  two parts of Σ with no voxel centre between them along their normals.
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
  where c > 0 (label 0) it pulls Σ in.
- **T5. Solve.** Alternate: (a) the τ LM solve with Σ fixed; (b) a step of
  Σ along the gradient of the energy with respect to the vertex positions,
  at most h/2 for each vertex, halved until it keeps the gap and makes no
  intersection (FreeSurfer's rule). Repeat until Σ and τ do not change.
- **T6. Start.** Expand Σ from the white surface along the vertex normals
  in steps of h/4, driven only by the stop term of T4 (out in the cortex
  label, stop at label 0). Each step keeps the gap and is checked for
  intersections. A vertex whose move would break the gap stops there:
  these are the first contacts. A vertex whose white face faces label 0
  directly does not move (zero thickness at an excision).
- **T7. Mesh quality.** After each Σ step, a tangential relaxation (each
  vertex moves only in its tangent plane, toward the centroid of its
  neighbours). Edges longer than 2 × the white mesh's median edge are
  split, and edges shorter than 0.5 × are collapsed, where the change keeps
  the gap and makes no intersection. No normal smoothing: the shape of Σ
  comes only from the τ energy and the stop term, so the abrasion test
  sees the neighbour-consistency terms on their own.
- **T8. Abrasion site.** Chosen later, when steps 1–3 work on site 1.

## Steps

1. Σ start (T6) on site 1 (`scratch/tube-sites/site1-fixed/c10mm/site.npz`):
   the contact area, where it is, and the cost. Compare with the 8,851
   contact faces of the first global run.
2. τ behind a fixed Σ (T3, T4): the τ solve on the shell. Measure the
   largest |v|, the cortex voxels outside Σ, the topology of {τ < 0}, and
   the cost.
3. Alternation (T5, T7) on site 1: how far Σ moves, the contacts after
   each outer iteration, intersections (always 0), energy by term.
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
