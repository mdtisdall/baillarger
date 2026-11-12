# Flow-field columns on phantoms

The first stage of `docs/plans/column-solver.md`. It builds the field form
of the column solver (arm C there) and tests it on phantoms only, before
any chain arm and before any real data. Read first:

- `docs/plans/column-solver.md`: *The model*, *Stopping and contacts*,
  *The field form (arm C)*, *Adaptive refinement and coarsening (arm C)*
  and *Inference*. This plan uses those definitions and does not repeat
  them;
- `prototypes/scripts/fold_phantoms.py` and
  `prototypes/scripts/bent_phantom.py` (the phantoms and their truth);
- `reports/Bent columns trial.md`, *The fit on the phantoms*;
- `docs/plans/round-2.md`, *Rules for every phase session*. Those rules
  apply here, with the changes in *Organisation* below.

This is a prototype plan. Nothing in the ray sampler or `src/` changes.

**Status:** draft for review. No code is written.

## Goal

Find the theoretical gaps in the field form, mainly in how it represents
kissing surfaces: a contact where the stop surface is pressed against
itself, a hairpin where it folds back, and a gap where it opens. The
phantoms have known geometry, so a gap shows as a clear failure, not as
noise in real data. The result decides whether the field form goes on to
real data, and what in it must change first.

## The design decisions under test

Each run must be able to show which decision failed. So each decision has
its own question, its own measure and its own failure signature. A failure
of the evidence (the stop field, the contrast) is kept apart from a
failure of the representation by the oracle runs (*Inputs*).

| # | Design decision (`column-solver.md`) | Question | Measure | Failure signature |
|---|---|---|---|---|
| R1 | The top is a node state (stopped), solved with the flow, not a separate surface | 1, 3, 4 | the stop boundary against the true stop surface | the boundary is right with the oracle stop field and wrong with the image one: evidence. Wrong with both: representation |
| R2 | The speed jumps to zero at the stop. τ jumps to τ_top | 4, 5 | the last levels' spacing near the stop, the jumps | levels pile up near the stop (a smooth decline), or jumps are nonzero on intact cortex |
| R3 | The jump has a prior that is almost always zero | 4 | jumps on the abraded patch and elsewhere | the jump absorbs a misfit: positive jumps on intact cortex, or no jump on the patch |
| R4 | τ_top = 0 is the origin | 5, 6 | levels against truth, the anchor comparison | the two origins give different levels after one constant, or many more sweeps |
| R5 | No cortex overlaps, by the form of the field | 1, 3 | flow lines that cross the stop set, points on two columns | any count above zero |
| R6 | No interpolation across the stop boundary | 1, 3 | the moving field on one side of a contact with the other side changed | the field on one side changes when only the other side changes |
| R7 | Non-crossing of columns by form, with a check for sinks | all | sinks, crossings, focal points inside the cortex | any count above zero away from the stop set |
| R8 | The grid is adaptive and may go finer than the image, with prior-shaped cells marked | 2, 7 | nodes, cells finer than the image, their marks | a focal point that the octree does not remove, or fine cells where no constraint needs them |
| R9 | Refinement is accepted by the free energy, with no hand-set threshold | 7 | refinements on the slab and on noise alone | any refinement accepted there |
| R10 | Uncertainty per column comes from sampled fields | 8 | z-scores of stops and levels | SD of z far from 1 |
| R11 | A moving node holds its own direction, tied to τ by the consistency term | all | the angle between the direction and ∇τ | a large angle where the truth has none (all phantoms here have orthogonal truth) |

## Questions

Each question names the phantoms that answer it (*Phantoms*). Each gives
the expected result if the design is right. A different result is a
finding, not a reason to tune.

1. **Fused contact.** In the fused part of the connected kiss, does the
   field stop each bank's columns on the contact, with no flow line across
   it, with and without the dark band? Expected: with the oracle stop
   field, every stop within one voxel of the true contact, and zero
   crossings. With the image stop field and no dark band, the stops spread
   wider, and the spread shows in the uncertainty.
2. **Hairpin.** At the bottom of a fundus, as g goes from 0.48 mm to
   0.04 mm, do the columns end on the hairpin with no focal point inside
   the cortex? Where do the octree's cells go finer than the image, and are
   they marked prior-shaped? Expected: on a uniform grid at image
   resolution, focal points appear at g = 0.04 mm. The octree removes them
   with cells finer than the image only within about one hairpin radius of
   the tip.
3. **The end of a gap.** Where the connected kiss changes from open to
   fused, the gap ends. With the smooth change the stop surface folds back
   across the slot. With the sharp change it turns through near-corners.
   Expected: the smooth change behaves like the open and fused parts on
   either side. The sharp change may show grazing columns, sinks or a
   wrong boundary near the end face. Each such place is a finding.
4. **The stop as a state.** Does the stopped region take the right form:
   the outside medium at the open slot, a sheet thinner than a cell at the
   fused contact, and a sheet that follows the hairpin round its tip? On
   the abraded phantom, do the jumps match the missing depth, and are they
   zero elsewhere? Expected: the stopped sheet at the contact is resolved
   only where the octree refines. Jumps within one τ candidate step of the
   truth without noise.
5. **τ at a contact.** Both banks reach τ_top at the contact. Does the
   solved τ agree there when the banks differ in thickness (2.5 and
   1.2 mm)? Does each bank get its own speed? Expected: the speed ratio of
   the banks near 2.5/1.2, and the levels match the truth after one shared
   monotone map.
6. **The origin of τ across a fold.** Solve the connected kiss with
   τ_top = 0, and again with one anchor column far from the fundus.
   Expected: the same levels after one constant. The top origin needs fewer
   sweeps, because every complete column is tied to the origin at its own
   stop. Report the sweep counts with and without the coarse grid.
7. **Grid.** On every phantom, the spacing scan, then the octree. Does the
   octree match the finest uniform grid's columns with fewer nodes? Does
   the free-energy test refuse refinement on the slab and on noise alone?
8. **Uncertainty.** With noise, are the per-column spreads from sampled
   fields calibrated: z = (true − mean)/σ with an SD near 1 for the stops
   and the τ levels?

## Truth under the stop rule

The phantoms' own truth must be restated for the rule that columns stop
at the stop surface (`column-solver.md`, *Stopping and contacts*). The
user decided (2026-11-11): cortex stops at a contact and does not overlap.
So at a fused contact each bank's column stops on the contact:

- **Connected kiss.** The depth function of the curved fundus with a slot
  half-width that changes along the fundus line: φ = 1 − (r − g(z))/t(θ).
  The true columns are its orthogonal trajectories in 3D (∇φ has a z part
  where g changes), traced as in `bent_phantom.py`. They stop on the
  slot's faces, which are the stop surface. Where g is below a voxel, the
  two faces are a fused contact, and each bank's column stops on it. The
  true τ is a monotone function of φ, the same for both banks.
- **Fundus and curved.** The true columns are the orthogonal trajectories
  of the depth function, as now. They end on the slot's edge (r = g),
  which is the stop surface. As g goes to 0, they end on the slot's
  half-plane and converge on its edge line. That line is part of the stop
  surface, so columns that meet there do not count as a focal point.
- **Abraded.** The truth of the curved fundus. On the patch, each column
  stops where it meets the abraded face, and its true jump is
  τ_top − τ(φ at the face).
- **The truth is a definition.** `bent_phantom.py` says so for its depth
  function. The adapters state it for each phantom.

**The true τ.** For every phantom here, the true τ is φ − 1 (so τ_top = 0)
times one constant per phantom: the mean bank thickness. So the unit is
near one millimetre of the thicker bank's depth, and the speed prior's
typical value is 1. The levels measure fits one shared monotone map, so
the constant does not affect the result.

## Phantoms

All phantoms: voxels of 0.16 mm (`fold_phantoms.H`), the oblique rotation
`fold_phantoms.R_DEFAULT`, the image of `fold_phantoms.make` from the
profile of the site `V1V2_banks` (user, 2026-11-11), with noise 0 and with
noise 1.0 (the site's matched SD), seed 0. Question 8 adds seeds 1–4.

| Name | Source | Parameters | Answers |
|---|---|---|---|
| `slab` | new | flat cortex 2.4 mm thick between a WM half-space and the outside medium; box 64 × 64 × 48 voxels | exactness, R9 |
| `crown_w0.32`, `crown_w0.64`, `crown_w1.28` | `fold_phantoms.standard` | t 2.4 | divergence, cover |
| `ckiss_smooth` | new, from `bent_phantom.curved` | box 96 × 96 × 128 voxels; tA 2.5, tB 1.2, width 180°; g(z) = 0.32 mm for z < −2.5 mm, 0.02 mm for z > 2.5 mm, the quintic S of `bent_phantom` between them (5 mm) | 1, 3, 4, 5, 6 |
| `ckiss_sharp` | new | as `ckiss_smooth`, with the change over 0.16 mm (one voxel) at z = 0 | 3 |
| `ckiss_smooth_flat`, `ckiss_sharp_flat` | new | as above, with `fold_phantoms.flat_top` (at 0.85) for the profile where g < 0.16 mm: the fused contact shows no dark band | 1, 4 |
| `fundus_g0.48`, `fundus_g0.16` | `fold_phantoms.standard` | t 2.4 | 2 |
| `fundus_g0.04` | new parameter value | t 2.4 | 2, 4 |
| `curved_g0.16`, `curved_g0.04` | `bent_phantom.curved` | tA 2.5, tB 1.2, shape 80³ | 2, 5 |
| `abraded` | new, from `curved_g0.16` | bank A, two disc patches of radius 2 mm, centres 3 mm and 8 mm from the slot's edge: the top 0.3 mm and 0.9 mm replaced by the outside medium (label 1, the medium's level) | 4: jumps, R3 |
| `tau` | new | as `slab`, box 96 × 64 × 48, with: thickness 2.4 × (1 + 0.25 sin(2πu/6 mm)) along u; a flat band (the profile constant over relative depth 0.40–0.55); an areal border at u = 0 where two bands change amplitude by × 0.5 and no band moves | the warp alone (`column-solver.md`, exact tests) |
| `noise` | new | the `slab`'s box, with noise only (no structure; cortex label everywhere) | R9 |

The features file is `scratch/onepass/fold_features.json` in the main
checkout (made by `fold_measure.py`), passed as an argument. Tests use a
synthetic features dict, as `test_fold_phantoms.py` does, so CI needs no
site data. The concentric kiss of `fold_phantoms.py` is not run (user,
2026-11-11): real brains have connected WM, so the connected kiss is
enough.

## The solver for this stage

`column-solver.md` leaves D1, D4 and D6 open. This stage needs concrete
values to run. The values below are the defaults for stage 1. Each one is
written into every run's record. A scan of a setting (× 0.5 and × 2) is a
finding to report, never a setting to tune. The checkpoint (PR 7) proposes
the values for D1, D4 and D6 from the evidence.

### Inputs

Each run takes three inputs, each from one of two sources:

| Input | Image source | Oracle source |
|---|---|---|
| Stop field: the log-odds that a point is stopped | the point-wise stop terms below, from the phantom's image and labels | the true stopped region, blurred by a Gaussian of 0.5 voxel SD, as a probability clipped to [0.02, 0.98] |
| Tensor: the leading eigenvector and coherence | `baillarger.orient` on the phantom's echoes, as for the site | the true column direction, coherence 1 |
| Contrast: the detrended echoes 2 and 3 | `fold_phantoms.make` (`det`) | the same: the contrast has no oracle |

The **oracle runs** isolate the representation: with the oracle stop field
and tensor, a failure is the field form's own. The **image runs** test the
evidence on top of it. Every question runs both.

**The point-wise stop terms** (image source). The terms of
`top_confidence.py`, evaluated at grid points, not per ray:

- the label term: high where the label is the outside medium;
- the sheet term: the dark band's sheetness (the Hessian of the detrended
  echo 2 at the band scale, 2 voxels);
- the mirror term: the profile below the point, mirrored, against the
  profile above it, along the tensor direction, over 1 mm;
- the partial-volume term: the outside medium's fraction in the voxel,
  from the echo levels of the WM, cortex and medium.

The weights follow `top_confidence.py`. Where that file needs a site
statistic (the floor S_floor, the thickness prior), the phantom's own
noise-free image gives it. The checkpoint report states which terms carry the
contact on each phantom.

### Node states

- **Grid.** Nodes at cell corners. The uniform spacings: 0.64, 0.48, 0.32,
  0.16 mm. The octree: coarsest 0.64 mm, finest 0.02 mm (1/8 voxel), 2:1
  balance.
- **Moving candidates.** τ: 9 values round the node's predicted τ (the
  mean of its neighbours' τ, moved along the predicted direction), Δτ =
  0.04 apart. Direction: 7 values, the predicted direction and 6 round it
  at 15°. So 63 moving candidates.
- **Stopped state.** One more state, with τ = τ_top = 0.
- **Pruning.** Keep the candidates that hold 0.999 of the mass, at least 4
  and at most 32. Log how often the final choice is at the edge of the kept
  set.
- **Re-centring.** After each width step, the candidate sets are moved to
  the new predictions, and the distributions are carried over by
  interpolation in τ and direction.

### Terms on the grid

Each term is between one node and one adjacent node (a cell edge), or on
one node. σ comes from data where it can (`column-solver.md`, *The
energy*).

| Term | Form on the grid |
|---|---|
| Consistency | on one moving node: (d·∇τ − 1/v)² / σ_c², with ∇τ from the node's τ and its neighbours' expected τ, and v the speed |
| Contrast match | on an edge: the echoes sampled at full resolution at the two nodes' positions moved along their directions to equal τ, smoothed at the band scale (2 voxels), weight falling where the profiles differ (D2: the provisional form of `column-solver.md`) |
| Alignment | on one node: the angle between the direction and the tensor's eigenvector, weighted by coherence |
| Bending | on an edge: the change of direction between the two nodes, per mm |
| Speed prior | on one node: log v near log 1, SD 0.3. On an edge: smooth across |
| Smoothness | the FEM prior below, on τ and the direction |
| Stop | on the stopped state: the stop field's log-odds |
| Jump | on an edge between a moving node at τ and a stopped node: the jump −τ − (the level that the step reaches), with the prior: a point mass at 0 of 0.999, and an exponential above 0 with mean 0.5. A negative jump is not a state |
| Start | the column's start points (2 mm behind the white surface, along its normal) seed τ through the speed prior. No offset is fixed except by the origin |

**The FEM prior.** Trilinear (Q1) elements on the octree's cells, with the
stiffness matrix K and the mass matrix M: precision λ (K + M / ℓ²), with
ℓ = 0.5 mm. The prior then means the same at every spacing, which the
free-energy comparisons between grids need (`column-solver.md`,
*Acceptance by evidence*).

### Inference

As `joint_gpu.py`: mean field, 8 colour groups (the parity of the three
node indices), the pair width 4 → 2 → 1. At most 200 sweeps per width.
A width ends when the largest change of any q is below 1e-4, or when the
free energy changes by less than 1e-7 of its value. A rise of the free
energy in any sweep is a defect, logged and reported. The coarse grid
(question 6 only): solve at 0.64 mm first, then use its distributions as
priors on the fine grid.

### The integrator

- RK4 on the posterior-mean direction, step 0.01 mm (the step of the
  truth), from each root of the phantom's white surface.
- **Masked interpolation.** Inside a cell, only moving nodes enter the
  interpolation, with weights renormalised. A cell whose moving nodes lie
  on both sides of a stopped sheet (the sheet passes between corners with
  no stopped node at a corner) is an **unresolved contact cell**. The
  integrator stops there and counts it. A count above zero means the grid
  did not resolve the sheet.
- **The stop.** In each step, the stop probability is interpolated along
  the step. The column stops where it crosses 0.5, found by 6 bisections,
  as `bent_phantom._last_step`.
- **Determinism.** The roots are processed in a fixed order, and the result
  does not depend on the order. The same input gives the same columns to
  the last bit on the CPU.

### The octree

- **Indicators.** The hierarchical surplus of τ and direction, and the
  local residual of the contrast, alignment and stop terms, each weighted
  by the distance that the nearby columns or stops would move.
- **Marking.** A cell is nominated for refinement where its indicator is
  above twice the median, and for coarsening where it is below half the
  median. All cells are marked in one pass, in Morton order, so the result
  does not depend on the order of updates.
- **Acceptance.** A nominated change is kept only if the free energy
  improves (*Acceptance by evidence*). The thresholds above only nominate.
  They do not decide.
- **Prior-shaped cells.** A cell finer than 0.16 mm is marked.
- **Stop.** At most 6 rounds of mark, refine or coarsen, and solve again.

## Measures

Diagnostics and views, not pass/fail rules, for each phantom, input
source, grid and noise level. Each gives the median, the 90th and the 99th
percentile, and a view of where the largest values are.

- **Stops:** the distance from each column's stop to the true stop
  surface (mm).
- **Paths:** the distance from each column to its true column
  (`bent_phantom.path_errors`, or `fold_eval.errors` for straight truth).
- **Levels:** the distance from each solved τ level to the true level,
  after the one shared monotone map is fitted (mm).
- **Speeds:** the ratio of the two banks' speeds, against the ratio of
  their thicknesses.
- **Spacing near the stop:** the distance between the last 5 levels before
  the stop, against the truth (R2).
- **Topology** (counts, with places): flow lines that cross the stop set;
  points on two columns (overlap); focal points inside the cortex; sinks
  (the field falls to zero or reverses away from the stop set); unresolved
  contact cells.
- **Independence across the contact** (R6): solve the connected kiss
  again with bank B's image changed (its profile shifted by 0.2 mm in
  depth). The change of bank A's field, by distance from the contact.
- **The stop boundary:** its position against the true stop surface near
  the contact, the hairpin and the gap's end. The thickness of the stopped
  sheet at the contact. Where the octree refines to resolve it.
- **Jumps:** the jump at each stop, against the true jump on the abraded
  patches and against zero elsewhere. The probability of a jump.
- **Direction and ∇τ:** the angle between them (R11).
- **Grid:** nodes per spacing. Where the octree refines and coarsens.
  Prior-shaped cells.
- **Uncertainty:** the spread of the stops and levels from 32 sampled
  fields, and the SD of z.
- **Inference:** sweeps per width, the free energy per sweep, the pruning
  log.
- **Cost:** time and peak memory.

## Exact tests

Exact tests use noise 0 and the oracle inputs unless stated. They are in
`prototypes/scripts/test_ff_*.py`, added to `testpaths` in
`pyproject.toml`. Each tolerance is fixed when the test is written, before
any run on the phantoms in *Questions*.

- **Truth adapters:** on each phantom, the true stops lie on the true stop
  surface to within 0.01 mm, and no two true columns cross.
- **New phantoms:** the labels match the closed-form regions at every voxel
  centre. g(z) of the connected kiss matches its definition.
- **Slab:** the columns are straight and normal to the bank (angle below
  1°), the levels are planes (deviation below 0.02 mm), and the octree
  stays at 0.64 mm.
- **Noise alone:** the free-energy test refuses every refinement.
- **Origin:** with an anchor column, moving the anchor changes every τ by
  one constant (to 1e-6) and nothing else. With the top as the origin,
  every complete column's stop is at τ = 0 (to one Δτ).
- **τ phantom:** the speeds match the true speeds to 5%, the levels across
  the flat band stay aligned, and no level shifts across the areal border
  (below 0.04 mm).
- **Masked interpolation:** on a synthetic two-cell grid, a moving value
  never takes weight from a stopped node or from the far side of a sheet.
- **Integrator:** the same input gives the same columns to the last bit.
  On a constant field, RK4 gives the straight line to 1e-9.
- **Free energy:** it does not rise in any sweep on the slab and the
  crowns.
- **Connected kiss:** no flow line crosses the fused contact, each bank's
  stops lie within one voxel of it, and no cortex overlaps.
- **Sub-voxel fundus** (octree): the columns end on the slot, no focal
  point is inside the cortex, and the tip cells are finer than the image
  and marked prior-shaped.
- **Abraded:** the jumps match the true jumps to within one Δτ, and are
  zero elsewhere.

## Run matrix, outputs and memory

- **The core matrix:** 15 phantoms × 2 input sources × 2 noise levels on
  the uniform grid at 0.16 mm, and on the octree. The spacing scan (4
  spacings) runs with the oracle inputs and noise 1.0 only. Question 8 adds
  4 seeds on `ckiss_smooth`, `curved_g0.16` and `abraded`, image inputs.
- **Scripts** take their output directory as an argument, and never name
  `scratch/`. Runs write to `scratch/flow-field-phantoms/<pr>/<phantom>/
  <source>_<noise>_<grid>/`: `field.npz` (τ, direction, q of the kept
  states, the stop probability, per node), `columns.npz` (the paths, the
  stops, the jumps, the levels, the z-scores), `run.json` (every setting
  and default of this plan, the input versions, the free energy per sweep,
  the pruning log, time and peak memory) and the views.
- **Figures** for the report go to `reports/figures/flow-field-phantoms/`,
  each well under 1 MB.
- **Memory.** The largest box is 96 × 96 × 128 voxels. At 0.16 mm and 64
  states per node in float32, q is about 300 MB. The octree's finest
  cells near the contact may add as much again. Expected peak below 4 GB
  per run, so the runs are light (*Organisation*). The first run of each
  kind measures its peak under `scripts/memguard` with a cap of 8 GB. If a
  run is above 4 GB, the matrix runs one job at a time.

## Organisation

### Roles

- **The coordinator** is a Claude Code session started separately. It
  works from this plan. It starts one **phase session** for each PR below,
  keeps the order of the PRs, and is the only link to the user.
- **A phase session** does one PR. It does the tasks of its PR, gives the
  sub-tasks to **sub-agents** (the `parallel-agents` skill), reviews every
  diff, runs the checks, and opens the PR.
- **A sub-agent** does one sub-task, in the files that its brief names.
  It does not run git commands that write. It reports to its phase
  session. The phase session sends to the coordinator what needs the
  coordinator.

### What needs the coordinator's permission

A phase session asks the coordinator, and waits for a reply, before it
does one of these. While it waits, it continues with other tasks of its
PR. A sub-agent asks its phase session, which asks the
coordinator.

1. A run with a peak above 8 GB, or with more than 4 cores for more than
   10 minutes (the heavy-run rule of `round-2.md`).
2. A change to a file that another PR owns, or to `ff_types.py` after
   PR 1 is open.
3. A change to a phantom, a truth adapter, a default of *The solver for
   this stage*, or a tolerance of *Exact tests*. After any result has been
   seen, such a change also needs a disclosure in the PR body: what was
   seen, and why the change is not chosen to improve it.
4. The removal, skip or weakening of a test.
5. A new dependency. Give its licence. No GPL dependency is permitted.
6. An edit to `docs/plans/`, `docs/formats/`, `docs/version-2-plan.md` or
   `src/`.
7. Work that is not in the PR's tasks.
8. A sub-agent model weaker than the one that this plan gives.

A phase session does not need permission to: edit the files of its own
PR, run light jobs, commit and push its branch, open its PR, and make a
provisional choice where this plan asks for a choice by diagnostics (with
the views and the choice in the PR body).

### What goes back to the coordinator

A sub-agent or a phase session does not act on work outside its scope. It
reports it. Each phase session's final report to the coordinator has these
lists. Each item gives what was found, the evidence (a file, a run, a
figure), and what it affects.

- **Findings outside scope:** a defect or a surprise in another PR's
  files, in existing code, or in the phantoms.
- **Proposed further work:** a task that a result calls for. Do not start
  it.
- **Changes to the plan:** a place where this plan is wrong or not
  specific enough. The coordinator keeps one list and applies it in one PR
  at the checkpoint.
- **Decisions for review:** each provisional choice, its reason, and what
  changes if the user reverses it.
- **Questions for the user.**

The coordinator decides what to do with each item, within *An overnight
run*. It collects the questions and the decisions for the user in its
final report, and never presents another session's approval as the
user's.

### An overnight run

The stage runs from PR 1 to PR 7 with no input from the user (user,
2026-11-11). So:

1. **Nothing waits for the user.** Where this plan says "ask", the phase
   session asks the coordinator. The coordinator answers from this plan.
   Where the plan does not decide, the coordinator makes a provisional
   decision, records it under "Decisions for review" in the PR body, and
   work continues.
2. **Nothing waits for a merge.** Every PR is stacked (*Git*). The user
   merges after the stage, in order.
3. **When a design decision fails** (a failure signature of *The design
   decisions under test* with the oracle inputs), the coordinator may make
   one provisional change for that decision (user, 2026-11-11):
   - The PR that found the failure is finished as planned, with the
     failure as its result. Its results are kept and are not run again
     in place.
   - The change goes in its own stacked PR (branch
     `feature/ff-change-r<n>`), with the failure's evidence, the change,
     its reason, and the affected questions run again. Its body begins
     "Provisional design change for R<n>".
   - The change must follow the user's rules (no overlap, no faults in
     cortex, no tuned constants, simple rules). It must not change a
     truth adapter, a phantom or an exact test's tolerance.
   - One change per decision. If the changed design also fails, record
     the second failure and continue. Do not make a second change.
   - Later PRs build on the original design, not on the change, unless the
     original makes them unable to run. Then the coordinator stacks them on
     the change and says so in their bodies.
4. **Where the stage ends.** After PR 7, the coordinator stops. Stage 2 of
   `column-solver.md` waits for the user's reading of the checkpoint.
5. **A failure of the tools** (a crash, a CI failure, a memory cap): the
   phase session finds the cause and fixes it in its own files. If the
   fix needs a file of another PR, it asks the coordinator. If a task
   cannot be done, the phase session records why, finishes the other
   tasks, and opens its PR with the gap stated.
6. **The coordinator's final report** to the user, at the end: the PR
   numbers in merge order, the checkpoint's table, every provisional
   decision and change, the lists of *What goes back to the coordinator*,
   and the questions for the user.

### NumPy or Torch

Each sub-agent whose task has a kernel over nodes, edges, cells or
columns (the terms, the mean field, the integrator, the octree's FEM
prior, the sampled fields) decides by analysis whether to write it in
Torch. Not by wall time on the laptop.

1. Before it writes the kernel, the sub-agent writes a short analysis:
   the operations and the memory traffic per sweep, at the phantom size
   and at the target size of `column-solver.md` (4·10⁶–4·10⁷ grid nodes),
   the loop structure, and whether the work is regular and data-parallel.
2. Torch changes the cost by removing Python loops and by running in
   parallel on the GPU. It does not change the complexity. A kernel that
   NumPy already runs as whole-array operations gains little from Torch on
   the CPU.
3. Use Torch if the analysis shows a large gain at the target size, and
   the Torch code is the same algorithm with no large increase in
   complexity: no custom kernels, no sparse tensors, and the device rules
   of `bent_terms.py` (no `index_add_` or `scatter_add_`). Use
   `baillarger.backend.device`. Else use NumPy.
4. The exact tests run on the CPU in float64, so CI needs no GPU. A run on
   MPS in float32 is checked against the CPU on one phantom, and the
   difference is reported. The integrator's bit-for-bit determinism is
   required on the CPU.
5. The analysis and the choice go in the PR body under "Decisions for
   review". The choice does not need the coordinator's permission. Torch is
   already a dependency.

### Git

Each PR has its own branch and worktree (`start-task`), stacked on the PR
below it (`round-2.md`, *Git and PRs*, items 5–7). PR 1 is stacked on
#201 (the branch `docs/column-solver-plan`, which holds this plan), so its
body begins "Stacked on #201". No phase session waits for a merge. When a
PR below changes, the phase session merges its branch into its own, and
does not rebase a pushed branch. The user merges after the stage. Nothing
in this stage merges to `main` without the user's instruction that names
the PR.

## Phases (PRs)

Each PR ends with views for the user in its body. The diagram shows the
order. PRs on one row can run at the same time, in separate phase
sessions.

```
PR 1 (contracts, phantoms, truth) ──┬── PR 2 (inputs) ──┐
                                    └── PR 3 (solver core, uniform grid) ──┬── PR 4 (kissing surfaces)
                                                                          └── PR 5 (octree) ── PR 6 (hairpins, jumps, grid, calibration) ── PR 7 (checkpoint)
```

PR 2 and PR 3 can run at the same time: PR 3 tests with the oracle
inputs, which PR 1 gives. PR 4 and PR 5 can run at the same time: PR 4
uses the uniform grid only. PR 6 needs PR 2, PR 4 and PR 5.

In the tables, "Parallel" names the sub-tasks that can run at the same
time, each in its own sub-agent. The briefs are written for the
sub-agents, in Simplified Technical English.

### PR 1. Contracts, phantoms and truth (branch `feature/ff-phantoms`, stacked on #201)

**Task 1.1. Contracts** (the phase session, not a sub-agent; model:
opus). Write `prototypes/scripts/ff_types.py`: the dataclasses of a
phantom (image, labels, features, frame), a truth (stop surface, columns,
τ, jumps), the inputs (stop field, tensor), a grid, a field and a run
record, with docstrings. Every later PR builds on this file. Do this
first. The sub-tasks below start after it.

**Task 1.2. New phantoms.** Parallel: 1.2a, 1.2b, 1.2c.

- **1.2a. Connected kiss** (sonnet; files `ff_ckiss.py`,
  `test_ff_ckiss.py`). Brief: "Make the connected kiss phantom in
  `ff_ckiss.py`. Start from `bent_phantom.curved`. Make the slot half-width
  a function g(z), with the values in the plan's *Phantoms* table. Make the
  smooth and the sharp variants. Make the variants without the dark band
  with `fold_phantoms.flat_top`. Use `fold_phantoms.make` for the image.
  Return the dataclass of `ff_types.py`. Write tests that compare the
  labels with the closed-form regions at every voxel centre. Use a
  synthetic features dict, as `test_fold_phantoms.py` does. Do not change
  `bent_phantom.py` or `fold_phantoms.py`."
- **1.2b. Slab, noise, sub-voxel fundus, abraded** (sonnet; files
  `ff_simple.py`, `test_ff_simple.py`). Brief: "Make the phantoms `slab`,
  `noise`, `fundus_g0.04` and `abraded` in `ff_simple.py`. Use the
  parameters in the plan's *Phantoms* table. For `abraded`, replace the
  top of bank A in two disc patches by the outside medium, at the depths
  that the table gives. Set the label of the removed tissue to 1. Return
  the dataclass of `ff_types.py`. Write tests of the labels at every
  voxel centre. Do not change `bent_phantom.py` or `fold_phantoms.py`."
- **1.2c. τ phantom** (opus; files `ff_tau.py`, `test_ff_tau.py`).
  Brief: "Make the τ phantom in `ff_tau.py`. Use the thickness function,
  the flat band and the areal border in the plan's *Phantoms* table. Write
  the true τ, the true speeds and the true offsets with the image. Write
  tests that the image profile is constant across the flat band and that
  no band moves at the areal border."

**Task 1.3. Truth adapters** (after 1.2a and 1.2b).

- **1.3a. Trajectory truth** (opus; files `ff_truth.py`,
  `test_ff_truth.py`). Brief: "Write the truth under the stop rule for
  the connected kiss, the fundus, the curved fundus and the abraded
  phantom, as the plan's *Truth under the stop rule* gives it. Trace the
  orthogonal trajectories of φ in 3D with RK4 at 0.01 mm, as
  `bent_phantom.py` does. Stop each true column on the true stop surface.
  Give the true τ as φ − 1 times the phantom's constant. Give the true
  jump on the abraded patches. Write the tests in the plan's *Exact tests*,
  item *Truth adapters*."

**Task 1.4. Views** (sonnet, after 1.3; file `ff_phantom_views.py`).
Brief: "Make one figure per phantom: a section through the image, the
labels, the true stop surface and 30 true columns. Write each figure to
the directory that the first argument gives. Keep each PNG under 1 MB."

### PR 2. Inputs (branch `feature/ff-inputs`, stacked on PR 1)

**Task 2.1. Point-wise stop terms** (opus; files `ff_stop.py`,
`test_ff_stop.py`). Brief: "Write the image stop field of the plan's
*Inputs*. Compute the label, sheet, mirror and partial-volume terms at
any set of points. Follow the forms and weights of `top_confidence.py`.
Take each site statistic from the phantom's noise-free image. Do not
change `top_confidence.py`. Write tests on the slab: the stop
probability is above 0.9 in the outside medium and below 0.1 deeper than
0.5 mm in the cortex."

**Task 2.2. Oracle inputs and the tensor** (sonnet; files
`ff_oracle.py`, `test_ff_oracle.py`). Brief: "Write the oracle stop field
and the oracle tensor of the plan's *Inputs*, from the truth of PR 1.
Write the image tensor with `baillarger.orient`, with the settings that
the site uses. Write tests that the oracle stop field is 0.5 on the true
stop surface to within 0.02 mm."

Parallel: 2.1 and 2.2.

**Task 2.3. Views of the inputs** (sonnet, after 2.1 and 2.2). Brief:
"For each phantom, show the image stop field, the oracle stop field and
their difference on one section. Show each stop term on its own at the
fused contact of `ckiss_smooth` and `ckiss_smooth_flat`."

### PR 3. The solver core on a uniform grid (branch `feature/ff-solver`, stacked on PR 1)

**Task 3.1. The grid and the FEM prior** (opus; files `ff_grid.py`,
`test_ff_grid.py`). Brief: "Write a uniform grid with nodes at cell
corners, at a given spacing, over a phantom's box. Write the Q1 stiffness
and mass matrices of the plan's FEM prior. Write a test that the prior's
correlation length in mm is the same at 0.64 mm and 0.16 mm spacing. The
octree is not in this task. Write the grid so that PR 5 can replace it
with an octree through the same interface in `ff_types.py`."

**Task 3.2. The terms** (opus; files `ff_terms.py`, `test_ff_terms.py`).
Brief: "Write each term of the plan's *Terms on the grid* as a function of
the node states. Write the moving candidates and the stopped state of
*Node states*. Write the jump prior. A negative jump is not a state. Write
a test for each term on a two-node grid, with values computed by hand."

**Task 3.3. The mean field** (opus; files `ff_meanfield.py`,
`test_ff_meanfield.py`; after 3.1 and 3.2). Brief: "Write the mean field
of the plan's *Inference*. Use the colour groups, the width schedule, the
stop rules, the pruning and the re-centring that the plan gives. Write the
free energy, and log it per sweep. Log a rise as a defect. Do not change a
default. Write a test that the free energy does not rise on the slab."

**Task 3.4. The integrator and the checks** (opus; files
`ff_integrate.py`, `ff_checks.py`, `test_ff_integrate.py`). Brief: "Write
the integrator of the plan's *The integrator*. Use masked interpolation.
Count the unresolved contact cells. Stop each column at the 0.5 crossing
of the stop probability, with 6 bisections. Write the checks of the
plan's *Measures*, item *Topology*. Write the tests of the plan's *Exact
tests*, items *Masked interpolation* and *Integrator*."

Parallel: 3.1, 3.2 and 3.4 (3.4 uses a field that a test gives, not the
solver). Then 3.3.

**Task 3.5. Runs and views** (sonnet, after 3.3 and 3.4; file
`ff_run.py`). Brief: "Write the run script. It takes the phantom, the
input source, the noise, the grid and the output directory as arguments.
It writes the files of the plan's *Run matrix, outputs and memory*. Run
the slab, the three crowns and the τ phantom with the oracle inputs at
0.16 mm. Make the views of *Measures* for each run. Report the results of
the slab and τ exact tests."

### PR 4. Kissing surfaces (branch `feature/ff-kiss`, stacked on PR 3 and containing PR 2)

Questions 1, 3, 4 (the stopped region), 5 and 6, on the uniform grid at
0.16 mm. The octree is not used here.

**Task 4.1. Runs** (sonnet). Brief: "Run every connected kiss variant
with both input sources and both noise levels at 0.16 mm. Run
`ckiss_smooth` again with an anchor column, and again with the coarse grid
first. Run the independence test of the plan's *Measures*. Measure the
peak memory of the first run under `scripts/memguard` with a cap of 8 GB."

**Task 4.2. Views and the report section** (opus, after 4.1). Brief:
"For questions 1, 3, 4, 5 and 6, make the views and the numbers of the
plan's *Measures*. Show the stop boundary at the contact, at the gap's end
and at the open slot, for the oracle and the image inputs side by side.
For each design decision R1–R7 and R11 in the plan's table, write which
result supports it and which result does not. Write the section
*Kissing surfaces* of `reports/Flow-field phantoms.md`."

### PR 5. The octree (branch `feature/ff-octree`, stacked on PR 3)

**Task 5.1. The octree** (opus; files `ff_octree.py`,
`test_ff_octree.py`). Brief: "Write the octree of the plan's *The octree*,
behind the grid interface of `ff_types.py`. Keep 2:1 balance. Write the
indicators, the marking in Morton order, and the acceptance by free
energy. Mark the cells finer than 0.16 mm as prior-shaped. Move the
distributions to children and to parents as `column-solver.md` gives.
Write tests: the slab stays at 0.64 mm, noise alone refuses every
refinement, and the marking does not depend on the order of updates."

**Task 5.2. The FEM prior on the octree** (opus; files
`ff_octree_fem.py`, `test_ff_octree_fem.py`; parallel with 5.1, joined
after). Brief: "Write the Q1 stiffness and mass matrices on a 2:1
balanced octree, with hanging nodes constrained. Write a test that the
prior's correlation length in mm is the same on a uniform grid and on an
octree of the same box."

### PR 6. Hairpins, jumps, the grid and calibration (branch `feature/ff-hairpins`, stacked on PR 5, containing PR 4)

**Task 6.1. Hairpins** (sonnet). Brief: "Run the fundus phantoms and the
curved phantoms of question 2 on the uniform grid at 0.16 mm and on the
octree, with both input sources and both noise levels. Count the focal
points inside the cortex. Map the cells finer than 0.16 mm."

**Task 6.2. Jumps** (sonnet). Brief: "Run the abraded phantom on the
octree with both input sources and both noise levels. Compare each jump
with the true jump. Map the jumps and their probabilities."

**Task 6.3. The spacing scan** (sonnet). Brief: "Run every phantom with
the oracle inputs and noise 1.0 at 0.64, 0.48, 0.32 and 0.16 mm, and on
the octree. Give the measures of the plan's *Measures*, item *Grid*,
against the number of nodes."

**Task 6.4. Calibration** (sonnet). Brief: "Run question 8: seeds 1–4 on
`ckiss_smooth`, `curved_g0.16` and `abraded`, image inputs, on the
octree. Draw 32 fields from the node distributions for each run. Give the
SD of z for the stops and the levels."

Parallel: 6.1, 6.2, 6.3 and 6.4, one job at a time if a run is heavy.

**Task 6.5. Views and the report sections** (opus, after 6.1–6.4).
Brief: "Write the sections *Hairpins*, *Jumps*, *Grid* and *Uncertainty*
of `reports/Flow-field phantoms.md`. For each design decision R2, R3 and
R7–R10 in the plan's table, write which result supports it and which
result does not."

### PR 7. Checkpoint (branch `docs/ff-checkpoint`, stacked on PR 6)

**Task 7.1** (the coordinator's phase session, opus). Brief: "Write the
checkpoint of `reports/Flow-field phantoms.md`. Give one row per design
decision R1–R11: supported, not supported, or not decided by these
phantoms, with the figure that shows it. List the gaps that were found and
what each needs. Propose values for D1, D4 and D6 of `column-solver.md`,
with the evidence. Give the user the choice: the field form goes on as
it is, goes on changed, or goes on with chains at hairpins." The
coordinator applies its list of plan changes in the same PR.

Torch follows *NumPy or Torch* in every PR. The later stages of
`column-solver.md` decide the kernels that run at the target scale.

## Decisions for the user

None are open in this plan. D-F1 is decided (*Truth under the stop rule*).
The origin of τ is decided (τ_top = 0), and question 6 compares it with an
anchor column. The defaults in *The solver for this stage* are
provisional: they are for review in this plan, and PR 7 proposes D1, D4
and D6 from the evidence.

## Not in this plan

- The chain arms (A, B) and the comparison between representations: later
  stages of `column-solver.md`.
- Real data, the coarse-to-fine schedule over the column graph, and the
  compressibility measure.
- Kernels written for the target scale. The Torch choice here is made per
  task, by analysis (*NumPy or Torch*).
