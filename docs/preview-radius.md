# The radius of a local preview of an edit

Status: design and evidence. `baillarger.antialias.local_solve` uses the rule (see
"The code").

## Purpose

An edit of the dseg changes the white surface only near the edit. For interactive
editing, a preview shows the effect of an edit quickly, and with a small error. The full
regeneration (`white_dseg`, then `white_surf`) runs after the user accepts the edits.

A preview of the antialiased field (`baillarger.antialias`) solves the field only within a
radius R of the changed voxels. Beyond R, it holds the field at the previous solution.
This document gives R from properties of the edit, and the error that the preview has
near the edit.

The preview's other steps are not in this document: the topology check of an edit, and
the marching of the changed blocks only.

## The problem

The field phi minimises the thin-plate energy on the free band, |sd| <= BAND
(BAND = 5 voxels). These constraints apply: phi >= EPS on the mask, and phi <= -EPS off
the mask. Outside the band, phi = sd, the signed distance clipped to +-(BAND + 3). The
gradient applies two stencil passes, which reach 2 voxels into the fixed layer. Thus the
fixed layer holds the value and the slope at each face of the band. The band is a sheet
that is clamped on its two faces.

## Derivation

1. Source. An edit changes the mask, so it changes sd. Because sd is clipped at
   BAND + 3, sd changes only within S = BAND + 3 = 8 voxels of the changed voxels. The
   change of phi starts there.

2. Decay. Away from the source, the change delta of phi obeys the biharmonic equation in
   the free band, if the set of active sign constraints does not change. On the faces of
   the band, delta = 0 and d(delta)/dn = 0. In a strip of half-width c that is clamped on
   its two faces, the solutions are delta = exp(-lambda x) f(y). For symmetric f,
   f = A cos(lambda y) + B y sin(lambda y). The clamped conditions give
   sin 2z + 2z = 0 (symmetric) and sin 2z - 2z = 0 (antisymmetric), with z = lambda c.
   These are the Papkovich-Fadle eigenfunctions of the clamped strip (Saint-Venant decay).
   The slowest root is z1 = 2.1062 + 1.1254i. The antisymmetric root is
   3.7488 + 1.3843i. Both roots were found numerically. Thus delta decays as
   exp(-x / l), with

       l = c / 2.106.

   The imaginary part gives an oscillation with a wavelength of 2 pi c / 1.125. It does
   not change the envelope.

3. Half-width c. In a flat band, c = BAND = 5 voxels, so l = 2.4 voxels. Where the bands
   of two sulcal banks merge, or the bands of the two sides of a thin WM blade merge, the
   free region is wider: c is up to about 10 voxels, and l is up to about 5 voxels. At an
   edit, c is the largest distance from a free voxel to a non-free voxel near the edit.

4. Constraints. The sign constraints add conditions only where they are active. The
   derivation ignores them, so it is the decay of the linear problem. The tests below
   show that the constraints make the decay faster. Thus l = c / 2.106 is an upper bound
   on the decay length of the real problem.

5. The error of the preview. Because the preview holds phi at the previous solution
   beyond R, it puts an error at the seam that is approximately delta(R). That error
   decays back towards the edit at the same rate. Thus, near the edit, the error is
   approximately

       e(R) = delta0 exp(-(R - S) / l).

   delta0 is the size of the change of the field that the edit causes. It increases with
   the size of the edit.

## The radius

For a preview error tau near the edit, the rule is:

    R = S + (c / 2.106) ln(delta0 / tau),    S = BAND + 3.

The rule uses the derived (linear) decay length. The tests show that this length is
longer than the real decay length, so the rule is conservative.

Example: a flat band (c = 5) and delta0 = 1 voxel. For tau = 0.01 voxel, R = 19 voxels.
For tau = 1e-3 voxel, R = 25 voxels. R increases only with the log of the precision.

The work of a preview is (the voxels in the box round the edit) x (iterations). The box
has (2R + size of the edit)^3 voxels, and only the band in it is free. A small box has a
smaller condition number than the full problem, so it needs fewer iterations.

## The code

`baillarger.antialias.local_solve` applies the rule with tau = 0.01 voxel
(`PREVIEW_TAU`). Its smallest R is 16 voxels (`PREVIEW_MIN_RADIUS`): the evidence below
gives an error of 0.002 voxel or less at R = 16 for each edit. c is the largest distance
from a free voxel to a non-free voxel, over the voxels within 4 voxels of the changed
voxels (`band_half_width`). delta0 is the larger of 1 and half the largest side of the
box of the changed voxels (voxels). The changed voxels are the given voxels whose mask
value differs from the sign of the earlier field.

The solve is `antialias.thin_plate` on the box of the changed voxels plus R + S + 2
voxels, with the earlier field as its start (`start`) and the voxels farther than R from
the changed voxels held (`held`). The margin past R is S + 2, so that sd (computed on the
box) is exact within R of the changed voxels. The prototype used a margin of 3 and sd
from a larger box; the two give the same problem within R.

The viewer's surface preview (`baillarger.viewer.surface_preview`) calls
`local_solve` for each group of changed voxels (26-connected cells of 2 x 16 voxels),
with the working field (the stored field and the patches of earlier solves) as the
earlier field.

## Evidence

Script: `prototypes/scripts/antialias_preview_radius.py`. Data: the right hemisphere of
sub-INDD121850 (0.16 mm voxels). The previous field phi_A is the converged field of the
dseg of `main` at commit e773d53 (1,826 iterations, the same count as the production run).

Sites: six boundary voxels, two in each class of c (5.5, 6.6 to 7.3, 8.8 to 9.5 voxels).
At each site, a ball of radius 1, 2, 3, 5 or 8 voxels was set inside the mask (bump) or
outside the mask (dent): 60 edits of 1 to 1,245 changed voxels. For each edit, the preview
was solved for R = 8 to 32 voxels, with the full solver tolerance. The reference is the
preview at R = 56. The error is the largest |phi_R - phi_ref| over the band voxels within
S of the edit, in voxels.

Results:

- Decay with the constraints: l = 0.95 to 1.9 voxels, for all sizes and all c. The decay
  length does not increase with c, and it does not change with the size of the edit.

- Decay without the constraints (the linear problem; ball of radius 3; reference R = 40):
  l = 2.0 to 3.5 voxels. For flat bands (c = 5.5), the measured l = 2.5 to 3.0, and the
  derivation gives 2.6. Where the bands merge, the measured l is shorter than c / 2.106.
  One fit on a flat curve near the solver floor gave 7.3. Thus the derivation describes
  the linear problem, and the constraints make the real decay approximately two times
  faster.

- Active constraints near the edit (within 14 voxels, |phi| < EPS + 1e-4): 1 to 3 % of
  the band voxels, and 5 to 16 % of the voxels next to the boundary. The active
  constraints are sparse points, not a full layer. But they make the decay shorter.

- Size of the edit: at a given R, the error of a ball of radius 8 is approximately 10 to
  30 times the error of a ball of radius 1 (delta0). This adds approximately 2 voxels to R.

- The largest error near the edit, over all sites:

  | R (voxels) | 8 | 10 | 12 | 14 | 16 | 20 |
  |---|---|---|---|---|---|---|
  | balls of radius 1 to 3 | 0.3 | 0.07 | 0.016 | 0.002 | 8e-4 | 6e-4 |
  | balls of radius 5 to 8 | 0.8 | 0.2 | 0.07 | 0.014 | 0.002 | 8e-4 |

  Below approximately 5e-4 voxel, the comparison is at the solver tolerance.

- Cost at R = 14, ball of radius 3: a box of approximately 64,000 voxels, 10,000 to 15,000
  of them free, and 280 to 380 iterations. This is approximately 2e7 voxel-iterations.
  The full solve is approximately 4e11 (1,826 x 2.2e8).

- A phantom (smoothed noise, merged bands, a flip of 6^3 voxels, R = 12) had a largest
  vertex shift away from the edit of 0.55 voxel. The derivation gives 0.2 to 0.6.

## Not tested

- Real topology corrections: handle fills and cuts at bridges and in narrow sulci. Their
  changed voxels can be long and thin, and their sites can have a different density of
  active constraints. The `Moves` of the dseg sidecar and the fixes diff give a test set.
- Edits larger than a ball of radius 8, for example a filled excision.
- A bound on the decay with the constraints. Step 4 is an argument, not a proof.
