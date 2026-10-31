# Plan: the WM mask defined by invariants

Planning file for the `feature/wm-invariants` branch. `dseg`'s corrected WM
mask is defined by properties it must have, and its algorithm is chosen to
guarantee them, in place of procedures whose effect depends on their order
and their radii. When the work is done, the lasting content goes into
`docs/formats/wm-dseg.md` and the module docstrings, and this file is
deleted.

## Why

The white surface folds back on itself within a few faces at about 90 places
on the example hemisphere (faces turning against their corners' normals).
About two thirds are at WM/GM. Most sit next to `dseg`'s handle fixes, which
are one voxel thick (a cut leaves two stubs a voxel apart; a fill leaves a
plug a voxel thick); most of the rest are on thin WM threads of the aseg.

The procedural options of PR #83 (fill cost, widened cuts, thickened fills,
trimmed threads) showed which direction helps, on the hemisphere
(`scratch/topo-variants/`, R = 1 voxel):

| Variant | Sites | GM sites, near a fix | GM sites, no fix nearby |
|---|---|---|---|
| none | 90 | 36 | 23 |
| fills weighed 100 times (cuts preferred) | 136 | 71 | 23 |
| cuts widened | 85 | 35 | 23 |
| fills thickened | 76 | 24 | 24 |
| threads trimmed | 61 | 35 | 5 |
| cuts preferred, widened; threads trimmed | 42 | 21 | 5 |

Removing WM (inward) where it is thin or next to a narrow gap helps most;
preferring cuts helps only when the cuts are then widened. But each option
is a procedure: what the result satisfies depends on the order of the steps
and on R, and the site count checks it only afterwards, by proxy.

## The invariants

M is the corrected mask, A the input mask (the filled labels after the manual
edits). Each voxel outside A has a side, from its input label: **GM** (the
cortex label only: 42 right, 3 left) or **other** (everything else not
filled: background 0, hippocampus, amygdala, ...). Hippocampus and amygdala
are other: the mask may grow into them. Some of the aseg's cortex is water
mislabelled as cortex; the invariant follows the label.

1. **Topology (hard).** M's Betti numbers (Kuhn adjacency) are (b0, 0, 0).
2. **No growth into GM (hard).** M takes no GM voxel: M \ A has no voxel
   with the GM label. Where WM meets cortex, the mask only moves inward (a
   column root inside the WM is safe; one in the cortex is not). Where WM
   meets anything else, it may move in either direction.
3. **Thickness.** Every voxel of M lies in a ball of radius r (mm) inside M:
   M is its own opening by that ball. No WM feature is thinner than about
   2r.
4. **Gap.** Every voxel outside M that lies between parts of M lies in a ball
   of radius g (mm) outside M: no gap between WM surfaces is narrower than
   about 2g. A one-voxel cut violates it.
5. **Smallest change (the objective).** Among the masks that satisfy 1 to 4,
   the one that changes the fewest voxels; approximated (below), since the
   exact problem with a topology constraint is NP-hard in general.

1 and 2 always hold. 3 and 4 hold everywhere they can without breaking 1 or
2; each place where they cannot is an **exception**, listed in the output
with its kind and position. For example: a thin WM bridge between two thick
parts that is not a handle can be neither removed (topology) nor thickened
into cortex (2).

r and g are in mm, so that they mean the same at 160 µm and at 50 µm.

## Exceptions are the review queue

The review queue (PR #80) becomes the list of exceptions: the places where
the invariants could not be met automatically, with their kind (thin bridge,
narrow gap that cannot be widened, ...). It replaces the proxy (faces
turning against their corners' normals), which stays as a diagnostic.

## How each invariant is guaranteed

Every change is a simple point (Kuhn adjacency), or a whole set that keeps
the homology (`handles._set_simple`), so 1 holds at each step; and the end
check of `wm_topo` stays.

- **1 and 2: the handle and cavity fixes.** A fix that would add a GM voxel
  is not allowed: a background wall holding GM voxels cannot be filled, so
  that handle is cut. This replaces the single fill cost of #83 by a cost
  per voxel: other voxels count once, GM voxels cannot be taken. A cavity is
  filled (on the example hemisphere every cavity voxel is background, label
  0). A cavity holding GM is a serious error that needs a person: `dseg`
  refuses and lists each one (size, centre), to be resolved by a manual edit
  and a rerun.
- **3 and 4: a homotopic opening, inward.** Repeat until nothing changes:
  find the voxels of M outside its opening by r, and those within g of a
  gap narrower than 2g; take them in connected pieces, thinnest first; remove
  a piece whole when that keeps the topology. A piece that cannot be removed
  whole is left as it is, not thinned (thinning a bridge to a chain one voxel
  across makes it worse), and becomes an exception. Removing only, this
  keeps 1 and 2 by construction; and its result does not depend on the order
  of the pieces within a pass, since pieces are not adjacent.
- **On the other side, either direction.** Where a violation of 3 or 4 is
  at WM next to non-GM voxels, adding voxels (simple points, never GM) is
  allowed as well as removing: a thin WM blade against fluid may be
  thickened rather than removed, and of the two the one that changes fewer
  voxels is taken (invariant 5). The first version removes only, which is
  allowed on both sides; the outward choice is added and compared on the
  hemisphere after it.

## What the example hemisphere's fixes take now

From the current `dseg` (`scratch/white-main`):

| Fix | Voxels | Input labels |
|---|---|---|
| Cavity fills | 6,307 in 278 pockets | all background (0) |
| Handle fills | 570 in 29 places | 484 cortex, 85 background, 1 hippocampus; 22 of the 29 places hold cortex |
| Handle cuts | 589 in 52 places | relabelled 529 cortex, 60 background |

So invariant 2 changes most of today's fills (22 places) into cuts, and the
cavity fills stay as they are.

## Tests

- Exact shapes: a handle whose fill would take cortex is cut, and one whose
  fill takes background is filled when that is smaller; a cavity of
  background is filled; a thread
  in cortex is removed; a cavity holding cortex is refused and listed; a
  one-voxel cut's gap is widened to at least 2g; a
  thin bridge is kept whole and listed as an exception; a thin blade against
  fluid (if outward is used there).
- Properties on random objects with random GM and other labels: 1 and 2
  exactly; 3 and 4 everywhere but the exceptions; each exception is a place
  where 3 or 4 fails.
- The default output stays the reference output until the change is chosen
  (`tests/test_handles.py` digests), or the digests change in the PR that
  makes the invariants the default.
- The hemisphere: the exceptions by kind, the voxels changed by side, the
  surface area, and the tight-fold sites against the table above.

## What happens to PR #83

Its fill cost becomes the per-voxel cost above; widened cuts and trimmed
threads become the opening of 3 and 4; thickened fills go (they grow into
cortex). #83 is closed when this lands, its table kept here.

## Open

- **r and g.** `dseg --thickness-mm R --gap-mm G` (0: off); a range is run
  on the hemisphere and the default set from the exceptions and the area
  lost, not from the site count alone. r has a floor of about 1.5 voxels:
  below it the voxel staircase itself breaks the invariant (with r = 1
  voxel a cube's edges are all 'thin' and a ball loses 24 voxels; with 1.5 a
  cube loses only its 8 corners and a ball nothing; with 1.8 neither).
- **Where the opening runs.** Over the whole mask (thin aseg threads
  included, as trimmed threads did: about 41,000 voxels at R = 1, much of it
  single-voxel bumps and corners), or only near the fixes. The invariant
  says everywhere; the hemisphere runs show what that costs.
