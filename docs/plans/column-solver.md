# Column solver: probes first or growth

Stream 2 of `docs/version-2-plan.md`: the refined columns (§4b, item 5)
and the open question "which paths are right in real cortex" (§4c, *Open*).
Read first:

- `docs/version-2-plan.md` §4b–4d (columns) and §5c, step 3 (the joint
  warp);
- `reports/Bent columns trial.md`;
- `reports/Column top stage.md`, *A joint solver (prototype, 2026-11-10)*
  (in PR #200), and `prototypes/scripts/joint_gpu.py`;
- `reports/Fused sulci and thickness priors.md` (the Gaussian core, the
  border weight, calibration).

This is a prototype plan. Nothing in the ray sampler or `src/` changes.
The rules in `docs/plans/round-2.md` apply. Heavy runs go under
`scripts/memguard`, one at a time.

**Status:** draft for review. No code is written. Stage 1 is its own
plan: `docs/plans/flow-field-phantoms.md`.

## Goal

Find out how to make the refined columns, and decide one question: is it
worth casting rays as probes to start the solve? Two strategies use the
same energy and differ only in where the solve starts and in which order it
updates:

- **A, probes first.** Cast the rays, correct the tops, then solve the
  whole length of every column together.
- **B, growth.** Grow all the columns from their starts, a small step at a
  time, and correct them at each step.

A third arm, **C**, holds the same unknowns as fields on a grid instead of
as chains, to compare the two representations. Arm C is tested first, on
phantoms only (*Phases*, stage 1), because its representation of kissing
surfaces has the most open theory.

The plan also fixes what the solver writes, so that the result can become
the stored refined columns (§4b, item 5) and the editor can re-solve a
region after a label edit.

## What is known

- **The top solver works as a probabilistic model.** `joint_gpu.py` keeps
  a distribution over candidate tops on each column, with the costs as log
  probabilities, and solves it by mean field in colour groups, from a wide
  agreement width to a narrow one. On the whole hemisphere (1,179,148
  columns, MPS) it took 65 s, and the objective did not rise in any sweep.
  It chooses a point on each column's ray. It does not move the ray.
- **Its exact test did not compare the methods.** Best-first restored all
  the false candidates and the mean field about half. The user found that
  this is an artefact of the segmentation, which is quite bad in places.
  So every comparison in this plan uses sites where the labels are correct
  or were repaired.
- **The bent-columns fit hardly bent the columns** (median bend at the top
  0.14–0.17°), did not converge in 4 rounds, and did not remove the
  sawtooth. Its barrier and crowding terms held the probes back. The best
  probe used only agreement, alignment, length and bending.
- **The large errors at the fundus come from the direction of the rays.**
  Rays from the fundus floor pass the slot tip and run up the wall's cortex
  at a shallow angle. They meet the outside late or not at all. The labels
  there are close to correct (user).
- **Straight rays fold** below about 1.5 mm in a quarter of the trial box.
  They can be probes but not columns.
- **The joint warp** (§5c, step 3; `joint_site.py`) gives each trace
  ordered levels τ and solves the pial surface and the depth together. It
  uses straight traces, weighted L1 costs and QPBO, not probabilities.

## The model (shared by all arms)

### Columns in τ

Each column b is a path x_b(τ) through the ribbon, parametrised by a shared
"time" τ. Neighbouring columns at equal τ are at the same laminar level.
Three things are unknown for each column:

1. **Its path.** Where it goes laterally, so it can bend.
2. **Its speed** along the path, ds/dτ. The speeds are set so that
   neighbours' contrast matches at equal τ. This is seismic flattening
   applied to the columns themselves.
3. **Its offset.** The τ at which the column starts. The start is a fixed
   point: 2 mm behind the root along the root's normal (the sampler's
   inward cap). No probe is needed to find it. The
   offsets let neighbouring columns align, both at the GM/WM transition and
   at the top.

**The origin of τ is the top: τ_top = 0** (user, 2026-11-11). Every
complete column ends at the same level, τ_top (*Stopping and contacts*),
so τ = 0 is the outermost laminar level: the pial edge, or the contact
where two banks meet. Every value in the cortex is negative and says how
far below the top, in laminar terms, a point lies. The pia is the
better-defined end: the transition to outside medium is sharp, and there
is no sharp GM/WM boundary. Limits:

- **The unit is arbitrary.** The speed prior's typical value fixes it, so
  τ is not millimetres. Read τ by its order and by equality between
  columns.
- **Columns that stop early** (excisions, abrasion) keep the same origin:
  they stop below it, by their jump.
- **At a fused contact with no visible notch**, the top is inferred from
  the shared stop and the alignment with neighbours, and the origin has
  that uncertainty.
- **In the WM below the cortex** there is little contrast, so τ there comes
  mostly from the speed prior. A column's start has a τ value but not a
  meaningful laminar depth.

The column graph (the across-triangle mesh graph of `white-surf/1`) is one
connected graph, also around excisions. The solver checks that it is
connected. A disconnected part is a defect to report and repair. An
**anchor column** (one point on one column as τ = 0) is no longer the
default. It is kept only as the comparison run in the flow-field plan's
question 6.

### The chain of each column

Each column is a chain of nodes, one at each τ step τ_k = k·Δτ. The state
of a node is one of:

- **not begun**: τ_k is before the column's offset;
- **a position**: one candidate point of the path at τ_k;
- **ended**: τ_k is after the column's top.

The offset is the step where the chain leaves "not begun". The top is the
step where it enters "ended". So the offset and the top are both
transitions of the chain, and both are aligned across neighbours like any
other level. A zero-thickness top at an excision is a chain that ends
right after the GM/WM levels. The top needs no separate solve: the
"ended" transition reads the shared stop field (below).

The candidates at a node are a small set of points round the node's
expected position: steps of several lengths (the speed) in several
directions within a cone round the current heading (the bend). The form of
the set is decision D1.

### Stopping and contacts

A top is a probability of stopping at each point of space. So the stop
evidence is one **stop field** over the ribbon, shared by every column
and every arm: the top stage's terms (`top_confidence.py`), evaluated at
points, not per ray. The mirror term is already contact evidence: it looks
for the column's profile mirrored on the far side.

The stop set is **one continuous surface**: the pial side of the cortex.
It never ends. At the bottom of a sulcus it folds back on itself in a
tight hairpin.

- **Open sulcus:** two parts of the surface face each other with outside
  medium between them. Each bank's columns end on their own part.
- **Contact:** the surface is pressed against itself across a slot of zero
  width. Columns from both sides end on the same sheet, which is two
  layers of the one surface. The contact is a stop set reached from more
  than one direction. This is the version-2 theory: every collision is a
  front meeting itself (§4c, step 1).
- Between them, the two parts come closer by degrees, as a fold and a
  contact differ only along a continuum.
- **Hairpin:** where the surface turns back at the bottom of a sulcus, its
  curvature is very high, but nothing meets and nothing ends. Columns from
  the floor and both walls converge toward the hairpin and end on it,
  roughly along its normal.

No step labels banks, and no column is paired with another. Each column
only has to stop on the shared set. Which side of a contact a point is on
follows from where the columns stop.

**No result concludes that cortex overlaps** (user, 2026-11-11). A point
of tissue belongs to one bank. Gaps are allowed. An overlap in an estimate
means only that the banks' evidence disagrees, so it is reported, never
drawn: at each contact site, the signed distance between where each
bank's evidence alone would stop (positive: a gap; negative: the old
overlap). This replaces the overlap view of version-2 §5c, step 3
("neither is forced to meet the other"). The chains need an exclusion term
for this (below). The field form excludes overlaps by its form.

**The top level and the jump.** All complete columns end at one level of
τ, τ_top, and no column passes it: τ ≤ τ_top everywhere in the cortex.
Where a column stops, τ jumps from its value at arrival to τ_top. The jump
is the part of the laminar sequence that is missing:

- **zero** at a pia or a contact, where the column ends with its whole
  sequence. Both sides of a contact have a zero jump;
- **positive** where the top layers are missing: an excision, or tissue
  where the top surface was abraded away (user, 2026-11-11: a realistic
  case that fits the model). A region of positive jumps most likely shows
  such damage. A zero-thickness top is the largest jump;
- **never negative.**

The jump has a prior that is almost always zero. A positive jump is an
expensive explanation, chosen only where the stop evidence pays for it, so
the solver cannot hide a misfit in it. The jump and its probability are
an output: a map of missing laminae, and review items where a probable
jump has weak evidence of damage. Complete columns that end at visibly
different levels with an intact top would show as small jumps with no
damage in the image. That is a sign that the model is wrong there, and the
review separates it from damage.

### The energy

All the terms are log probabilities, as in the top solver. Each term is a
cost on one node, on two consecutive nodes of one column, or on two nodes
of neighbouring columns at the same τ step. So the energy of a partial
column (the steps grown so far) is defined, which growth needs.

| Term | Acts on | What it does |
|---|---|---|
| Contrast match | neighbours at equal τ | The echoes at the two candidates agree, at the band scale of at least 2–3 voxels. Thinner layers follow the τ levels. A weight falls where the deep profiles differ, so an areal border is a change of facies, not forced into a match (D2). |
| Alignment | one column, step k → k+1 | The step direction follows the tensor's leading eigenvector, weighted by coherence. |
| Bending | one column, k−1, k, k+1 | The heading changes slowly. |
| Speed prior | one column, and neighbours | The speed is smooth along τ and across neighbours, and near a typical value. Where the contrast is flat, this term alone sets the speed. |
| Separation | neighbours at equal τ | The two positions keep their order along the edge between the roots. A crossing has a high cost. |
| Start | the "not begun" → position transition | The column's first position is its fixed start point. |
| Stop | the position → "ended" transition | The stop field at the last position, and the prior on the jump τ_top − τ at the stop. |
| Exclusion | any two columns at the same point | The expected occupancy of space by the columns, on a grid. Occupancy by two columns whose roots are far apart on the mesh has a high cost. This is what keeps the banks from overlapping, because columns of opposite banks are not neighbours on the mesh. |

The weights come from data where they can (σ from the spread of the
measured quantity), as in the top stage. A weight scan (each weight × 0.1
and × 10) is a finding to report, never a setting to tune.

**Non-crossing is not a guarantee.** The mean field gives each node its
own distribution, and a crossing is a property of the joint configuration.
The separation term acts only in expectation. So the solver checks the
chosen configuration for folds and overlaps after every solve, in every
chain arm.

**No focal point inside the cortex.** Columns converge strongly toward a
hairpin, but a correct set of columns ends on the stop surface before any
two meet. Columns that meet inside the cortex mean the map from the roots
is folded. Every arm checks this after every solve and reports it as an
error, like a fold.

### The field form (arm C)

The same unknowns can be held as fields on a grid instead of as chains.
Each grid node is in one of two states:

- **moving:** a candidate τ and column direction. A local consistency term
  ties them: the speed along the direction equals 1/|∇τ| in that
  direction;
- **stopped:** speed 0 and τ = τ_top.

Each node holds a distribution over its moving candidates and the stopped
state. The columns are the flow lines of the direction field, from each
column's start. A column that reaches a stopped node ends there. So the
**top probability at a point is the probability that its node is
stopped**, and the stop surface is the boundary between moving and stopped
nodes. It is solved in the same mean field as everything else, not located
separately. The stop field is the data term on the stopped state. With τ
stored at the node, every term is between adjacent grid nodes, so the mean
field is the same kind of problem as the tops. With only a velocity
stored, the τ at a point would be an integral over the whole field below
it.

**The speed jumps to zero at the top.** It does not decline smoothly. A
smoothly declining speed would let the columns reach the top only as
τ → ∞, and would squeeze the last levels into a vanishing distance. So
the speed stays nonzero up to the boundary and is zero beyond it, and τ
jumps to τ_top there (*The top level and the jump*). The integrator stops
each column where it crosses the boundary, at a sub-cell position from the
stop probabilities.

- **Non-crossing by form.** A field that is smooth in the moving region
  gives flow lines that cannot cross, so the posterior mean field, or the
  per-node choices after smooth interpolation, give columns that do not
  cross, whatever the uncertainty. The solver checks only where a moving
  node's field falls toward zero or reverses away from the stop boundary,
  which makes columns meet at a sink.
- **The cases of the stop surface.** At an open sulcus the stopped region
  is the outside medium. At a contact it is a stopped sheet thinner than
  a cell, reached from both sides: the columns on each side flow into it
  and end, nothing crosses it, so each point lies on one column and cortex
  cannot overlap. At a hairpin the stopped sheet folds back. At an
  excision or abraded tissue the stopped region begins below τ_top, with
  a positive jump.
- **No interpolation across the stop boundary.** The field is never
  interpolated between a moving and a stopped node, or between the moving
  nodes on the two sides of a thin stopped sheet. The boundary's position
  is part of the solved state. At a contact the stopped sheet is thinner
  than a cell, so the octree refines there until the sheet is resolved,
  below a voxel if needed, with those cells marked prior-shaped.
- **Hairpins.** Near a hairpin, the columns end on the surface roughly
  along its normal, and the normal turns through about 180° over a
  distance equal to the hairpin's radius. So the field's direction turns
  as fast. This is a question of resolution, not a singularity: the field
  is smooth on the cortex side wherever the grid resolves the radius. The
  natural chart there is the stop surface itself (position on the surface
  and distance from it), in which a hairpin is only a strongly curved part
  of a smooth surface. `bent_phantom.py` uses the special case, polar
  coordinates about the slot's edge.
- **The grid is adaptive.** The grid holds how the geometry changes (the
  bends and the speed), not the image. In the smooth interior of a bank
  that changes over fractions of a millimetre or more, so the grid there
  can be coarser than the image: 0.3–0.6 mm, the scale that the data
  support (version-2 §4c, *Open*). Near tight folds (crowns down to
  0.32 mm on the phantoms) a coarse grid makes the columns too smooth, so
  the grid there goes down to image resolution. Whether the coarse spacing
  is safe is tested, not assumed (*Evaluation*, D6).
- **What the grid does not limit.** The field gives a separate continuous
  column from every root, at any density: the columns are not snapped to
  the grid. The contrast match samples the image at full resolution along
  the interpolated τ, so the thin bands are matched at image resolution.
- **Fallback: chains at hairpins.** If the spacing scan shows that the
  field fails at hairpins even at image resolution, the field is used in
  bank interiors and chains in a tube round each hairpin. The cost is a
  seam where chain ends must match the field's flow lines. This is not
  built unless the scan calls for it.
- **Per-column uncertainty** comes from samples: draw fields from the node
  distributions, smooth them, integrate the columns, and take the spread
  of each column's levels and top.
- **Storage.** The field and the offsets on the start surface encode every
  column. The integrator and its step become part of the spec, and must be
  deterministic.

### Adaptive refinement and coarsening (arm C)

The grid is an octree. It is refined where it cannot follow the data and
coarsened where a parent cell already describes its children:

- **Indicators.** The hierarchical surplus: the difference between a
  cell's solution and the value interpolated from its parent. A large
  surplus keeps or refines the cell, a small one coarsens it. And the local
  residual: the part of the contrast, alignment and stop terms that a cell
  cannot reduce. Both are weighted by their effect on the columns: how far
  the columns or the stops would move. An error where no column passes
  does not count.
- **Acceptance by evidence.** More cells always fit the data better, so a
  refinement is kept only if the finer model's free energy (a bound on the
  evidence) is better than the coarser one's. The free energy includes the
  constraint terms (separation, exclusion), not only the data, so a
  refinement that removes a fold or a focal point is accepted even where
  the contrast terms barely change. This penalises detail that
  only fits noise, with no hand-set threshold. It needs a smoothness prior
  that means the same at every spacing: FEM weights with a mass matrix,
  so the correlation length is physical (the fused-sulci report).
- **Finer than the image where the geometry needs it.** The grid describes
  the coordinate map, not detail in the image, so there is no limit at
  image resolution. At a fused fundus the hairpin's radius can be smaller
  than a voxel, and a grid that cannot follow it forces the columns to
  meet. Below a voxel, the shape still has constraints: the stop surface's
  sub-voxel position (partial volume in the stop probability), the white
  surface (a sub-voxel mesh), no folds, no overlap, and the smoothness of
  cortex. Where nothing is violated, finer cells bring no gain and the
  acceptance test refuses them.
- **Prior-shaped cells.** Inside a cell finer than the image, the shape
  comes mostly from the prior and the constraints, and the contrast match
  reads interpolated values. Each such cell is marked prior-shaped, like
  the rule-only provenance, and its broad distributions are reported, not
  hidden.
- **Hysteresis.** The threshold to refine is higher than the threshold to
  coarsen, so a cell does not change back and forth between sweeps.
- **2:1 balance.** Neighbouring cells differ by at most one level, so
  interpolation stays consistent.
- **Moving the distributions.** A refined cell's children take the
  parent's distribution as their prior. Coarsened children give the parent
  their combined distribution.
- **Determinism.** The marking does not depend on the order of updates, so
  the same input gives the same grid. After a label edit, only the region
  round the edit is marked again.

### Inference

Mean field over all the nodes, as in `joint_gpu.py`: each node holds a
distribution q over its states, updated from its own terms plus the
expected terms of its neighbours (along its chain and across at equal τ).
The pair widths start wide and narrow in steps. Colour groups of the
column graph update together. The objective is the free energy, and it
must not rise in any sweep. A rise is a defect.

**Pruning.** Each node keeps the candidates that hold most of its
probability mass (a mass rule, not a fixed count). A later backward step
cannot restore a pruned candidate. So the solver logs how often the final
choice is near the edge of the kept set.

### Coarse to fine

Every arm solves a coarse level first and then the fine level. This
changes how the solver works, not the energy: the fine level's energy is
the energy above.

- **The coarse level.** Coarse columns, one root per coarse cell, as in the
  top stage (cells of 2 voxels, about 0.32 mm). The coarse graph links
  cells whose roots are neighbours on the white mesh, so it stays connected
  and never links roots that face opposite ways across a thin WM blade. The
  coarse τ step is at most the band scale (2–3 voxels), so the contrast
  that sets the speed is not blurred. Coarsen across columns much more
  than along depth.
- **What it gives.** Mean field moves evidence about one neighbour per
  sweep, so on the fine graph the offsets far from columns with clear tops
  settle slowly. On the coarse graph the same distance takes few sweeps. The
  coarse level fits in memory for a whole hemisphere, and the fine level
  then runs by superblock with a halo.
- **How it passes down.** The coarse distributions become priors on the
  fine nodes, not fixed values. Where the coarse level is uncertain, the
  fine level's candidates spread wider. Features smaller than a coarse
  cell (tight fundi, thin blades) cannot be represented coarsely, so the
  fine level must be free to correct them.
- **In growth.** B-fwd and B-fb grow on the coarse level first, then on the
  fine level with the coarse priors.
- **In the field form.** Arm C solves a coarse grid first, and its
  distributions become priors on the finer grid.

Coarse to fine is tested as one factor (fine only against coarse to fine),
not as more arms (*Evaluation*).

## The strategies

The energy, the candidates, the pruning rule, the stop field and the
coarse-to-fine schedule are the same in every arm. Arms A and B use
chains. Arm C uses the field form, with the same terms written locally on
the grid.

| Arm | Start | Update order |
|---|---|---|
| **A** (probes first) | The top stage's rays and its corrected tops. The candidates are centred on the probe paths, with the offset and the speed from a first fit of the contrast along the probes. The top stage traces one ray per coarse cell, so every column of the phantoms and the fundus box gets its own probe. | All the nodes of all the columns, until the objective stops falling. |
| **B-fwd** (growth, forward only) | The columns' starts. | One τ step at a time, outward. A node is final when the front passes it. |
| **B-fb** (growth, forward and backward) | As B-fwd. | As B-fwd, then sweeps over all the nodes, back from the tops and out again, until the objective stops falling. |
| **C** (field form) | The tensor's leading eigenvector for the direction, and τ from distance along it, at the coarse level. | All the grid nodes, until the objective stops falling. |

- **A against B-fb** answers the main question. Both end as solves of the
  whole length. Only the start differs.
- **B-fwd against B-fb** measures how much evidence from deeper in the
  column (mainly the top) must flow back to shallow steps.
- **C against the better chain arm** compares the representations: the
  guarantee of non-crossing and no overlap against the resolution of the
  grid at tight folds.
- In B, a column's offset is unknown when the front reaches its start. The
  "not begun" state carries that: until the front has grown far enough to
  see the contrast, a column holds a distribution over when it begins. The
  plan records how many steps each column needs before its offset settles.

Two arms are not in the first round. They can be added if the first round
calls for them:

- **Fixed-lag growth.** B with a backward window of L steps. Memory stays
  bounded by L.
- **Growth with lookahead probes.** B with short probe rays from each
  candidate. This mixes in the probes, so it is not a clean B.

## Evaluation

### Sites

Only where the labels are correct or were repaired.

| Site | Truth | What it tests |
|---|---|---|
| Fold phantoms (`fold_phantoms.py`) and the curved fundus phantom (`bent_phantom.py`) | Known true columns and levels | Path error; exact recovery |
| τ phantoms, the slab, sub-voxel fundi, the connected kiss, the abraded phantom (new, stage 1) | Known offsets, speeds, stops, jumps and columns | The warp alone; the origin of τ; flat contrast; an areal border; kissing surfaces; missing laminae |
| The fundus box of the bent-columns trial (true vertex 12182770, hemisphere R, radius 5 mm), with the painted labels | The user's reading of the fundus; the failing stretch | The direction failure; the sawtooth |
| M1S1 (`prototypes/data/M1S1_points_full0.json`, 23 points) | Marked points | The profile split; tops at a contact |
| V1V2 banks (`prototypes/data/V1V2_banks_points_full0.json`) | Marked points | A striate border |
| The four superblocks of the top solver's region (`1_1_1, 1_1_2, 2_1_1, 2_1_2`) | None. Read only at columns whose sites the user has reviewed and found label-correct. | Scale, memory, folds at scale |

### Measures

Diagnostics and views, not pass/fail rules. Each run shows every measure
for every arm. The arms' views go to the user blind (with neutral labels),
and the user's reading is recorded before the arms are named.

- **Final energy and its parts.** If two arms give different columns, the
  one with the lower energy is the one the energy prefers. If that one is
  also closer to the truth, the other arm failed to optimise. If it is
  further from the truth, the energy is wrong, and the plan records which
  term.
- **Folds** in the chosen configuration: the sign of the local Jacobian of
  the map from the roots, the count of crossing neighbour pairs, and where
  they are.
- **Tops** against the user's marked tops.
- **Path error** on the phantoms. Straight rays: 0.18–0.41 mm at crowns.
- **The profile split** at the marked points. Straight normal rays:
  0.17 mm at M1/S1, 0.075 mm at V1V2.
- **Alignment** to the tensor where coherence ≥ 0.7, as the median and
  the 90th percentile angle.
- **Level shifts between neighbours** (the "no faults" rule, §4b): a
  shift of a level by more than 1–2 voxels between neighbours, or a step
  along a border between groups of columns.
- **Calibration.** Report the uncertainty of each column's levels relative
  to its own top (τ_top = 0), and of the offset difference between
  neighbours. Mask the visible tops in blocks, solve again, and check
  that z = (t_true − μ)/σ has an SD near 1 and that 95 % intervals cover
  about 95 %.
- **Pruning:** the share of final choices near the edge of the kept set.
- **Coarse to fine,** as one factor: each arm on the fundus box and the
  curved phantom, fine only and coarse to fine. The final energies; the
  sweeps to converge; where the fine level moves away from the coarse
  answer, and whether that is at the features smaller than a coarse cell;
  and whether the coarse level's errors at the tight fundus remain in the
  fine result.
- **Contacts:** at each contact site, the stop sheet's position and
  probability against the user's reading, and the signed distance between
  the banks' own stops. For the chain arms, the overlaps found by the check
  before the exclusion term resolved them.
- **Field only (arm C):** where the field falls to zero or reverses away
  from a stop sheet; the per-column spread from samples; the columns'
  error at tight folds against the chains'.
- **The grid spacing scan (arm C):** on the curved phantom and the fundus
  box, arm C at several uniform spacings, from 0.6 mm down to image
  resolution, then the adaptive grid. For each spacing: the path error,
  the error at the hairpin, the stops against the known surface, and
  where the error starts to grow. The uniform spacings are the reference
  for the adaptive grid: does it match the finest uniform grid's columns
  with a fraction of the nodes, and does the evidence test stop refining
  before the residual reaches the noise level? This is the evidence for
  D6, and for the fallback (chains at hairpins).
- **Focal points:** the count and the places where columns meet inside the
  cortex, per arm.
- **Jumps:** the map of jumps at the stops, with their probability, against
  the known missing depth on the abraded phantom, and their regions at the
  real sites, read with the user.
- **Prior-shaped cells (arm C):** where they are, and the spread of the
  columns through them.
- **Compressibility of the unfolded image.** The columns' samples at each τ
  step are already computed for the contrast match, so the unfolded image
  costs nothing more. Predict each fine column's profile from its coarse
  parent (the coarse-to-fine levels) and report the energy of the residual
  per unit volume of tissue, level by level. This is a lifting-style
  wavelet step on the column hierarchy, about the cost of one sweep. It
  needs no ground truth, so it also works at the superblocks. It overlaps
  the contrast match term, which is its nearest-neighbour form. What it
  adds is the longer range: coordinates that agree with each neighbour but
  drift over millimetres. Like the contrast term, and unlike the tensor,
  it sees the offsets and speeds. Three guards:
  - **No collapse.** A map that sends many columns through the same tissue
    makes the image trivially compressible, so the measure counts only
    maps that cover the ribbon one to one, per unit volume covered.
  - **Tilt is invisible to it.** Columns all tilted the same way still give
    smooth profiles, stretched by 1/cos θ. Alignment stays a separate
    check.
  - **Areal borders are real.** The image is piecewise smooth across a
    border, so the measure is read with the borders in mind.

  It is dropped if it adds nothing to the contrast term at the first
  sites.
- **Growth only:** steps to settle each offset; how far B-fwd's columns
  move when B-fb's backward sweeps run.
- **Cost:** time and peak memory per arm, and a memory model for a 50 µm
  whole brain (below).

### Exact tests (stages 1 and 2)

- On a τ phantom with known offsets and speeds, the solver recovers the
  offsets and the speeds, with every complete column's top at τ = 0.
- In the comparison run with an anchor column instead, moving the anchor
  changes every offset by one constant and changes nothing else.
- A phantom whose contrast is flat over a band: the speed there follows the
  speed prior, and the levels on both sides stay aligned.
- A phantom with an areal border (the profile's content changes, the
  layers do not shift): no level shifts across the border.
- A graph with two components: the solver reports the defect and stops.
- A contact phantom with a known contact sheet, and an open-sulcus phantom
  with a known gap: every arm puts the stops on the known surface, and no
  arm's output has overlapping cortex.
- Arm C on a straight phantom gives the same columns as the chains.
- Arm C on the contact phantom: no flow line crosses the known surface,
  the stopped sheet lies on it, and the moving field on one side does not
  depend on the other side (no interpolation across the sheet).
- On a phantom with a hairpin of known radius, every arm's columns end on
  the hairpin, with no focal point inside the cortex. Arm C's stopped
  sheet follows the hairpin round its tip.
- On a phantom with an abraded patch (the top layers removed to a known
  depth), the jumps there match the missing depth, and the jumps elsewhere
  are zero.
- Adaptive refinement on a straight phantom stays at the coarsest level,
  and on noise alone the evidence test refuses every refinement.
- A hairpin phantom with a radius below one voxel: every arm's columns end
  on the hairpin with no focal point. Arm C's cells at the tip are finer
  than the image and are marked prior-shaped.
- On a straight phantom with straight true columns, no arm bends them.
- On the same phantom, fine only and coarse to fine reach the same columns
  and the same energy.
- On a phantom with a fundus narrower than a coarse cell, the fine level
  corrects the coarse answer at the fundus.

## Memory at the target scale

The estimate for a 50 µm whole brain, about 3.5·10⁸ columns, with about
150 τ steps per column (6 mm at 0.04 mm) and about 20 kept candidates per
node, in float32:

- **A and B-fb** hold every node: 3.5·10⁸ × 150 × 20 × 4 B ≈ 4 TB for q
  alone, with the evidence of the same order. So they must run by
  superblock with a halo, in sweeps, with q on disk between sweeps.
- **B-fwd** holds the front: 3.5·10⁸ × 20 × 4 B ≈ 28 GB, also by
  superblock.
- **Arm C** holds the grid nodes only. With a whole brain's ribbon and
  margins at about 10⁶ mm³ (an assumption), a grid of 0.3–0.6 mm has about
  4·10⁶–4·10⁷ nodes: about 0.4–3 GB for q at 20 candidates, plus the finer
  grid near tight folds. It does not depend on the image resolution.
- **The coarse level** has about 12 times fewer columns (hemisphere R:
  1,179,148 coarse columns against 13,763,166 white-mesh vertices) and a τ
  step 2–3 times larger. That is about 23–35 times less than the fine
  level, about 120–180 GB for q at 50 µm. That is still too large for one
  machine, so a
  whole brain needs more than one coarse level, or the coarse level by
  superblock with a larger halo. Stage 4 decides which from the measured
  sizes.

These are estimates from assumptions. Stage 4 measures the real step count,
candidate count and halo width, and gives the analysis again. The choice
between the arms is by accuracy first. Memory and the cost of a local
re-solve decide only between arms that are equally accurate.

## What the solver writes

The prototype writes scratch `.npz` files with these fields, so that the
stored form (§4b, item 5) can be chosen from them:

- per column: the edge key; the chosen path at each τ step; the offset and
  the top (or censored); the most probable state's probability at each
  step; the offset difference to each neighbour with its uncertainty;
  provenance (from data, or rule-only);
- arm C also: the grid's τ and direction (the posterior mean and the
  most probable candidate at each node), the offsets on the start surface,
  and the integrator and its step;
- per run: the arm, the origin (the top, or an anchor column in the
  comparison run), the input versions, the energy and its
  parts per sweep, and the pruning log.

The stored form is chosen after the stage 3 checkpoint, with the result. A compact
candidate: the path as a shear spline off the probe ray (as in the
bent-columns trial), plus the τ map as a monotone spline and the offset.

**Local re-solve for the editor.** After a label edit, the columns in the
edit's blocks and a margin are solved again, with the columns outside the
margin fixed, at the fine level, with the coarse level fixed unless the
edit changes it. Stage 4 measures, for each arm, the time of a re-solve at a
column site, and how far the change reaches past the edit.

## Phases

The work runs in stages. Each phase ends with views for the user. A
checkpoint needs the user's reading before the next phase starts.

**Stage 1: the field form on phantoms** (`docs/plans/flow-field-phantoms.md`).
Arm C alone, on phantoms only. The phantoms have known geometry, so gaps
in how the field represents kissing surfaces (contacts, hairpins, the end
of a gap) show as clear failures. Its checkpoint (F4) decides whether the
field form goes on as it is, changed, or with chains at hairpins, before
any later stage starts. The truth adapters under the stop rule, the new
phantoms and the τ phantoms of stage 1 are reused by every later stage.

**Stage 2: the chain arms on the same phantoms.**

1. **The chain model.** The chain, the terms, the exclusion term, the mean
   field on the CPU (NumPy reference), and the coarse level with its priors
   on the fine level. Exact tests on the stage 1 phantoms.
2. **Arms A, B-fwd and B-fb** on the stage 1 phantoms, fine only and coarse
   to fine, against arm C's stage 1 results on the same phantoms.

**Stage 3: real data at the fundus box.**

3. **Every arm that goes on**, on the fundus box of the bent-columns trial
   with its painted labels, fine only and coarse to fine, and arm C with its
   grid from stage 1.
4. **Comparison and checkpoint.** The blind views, the energies, and the
   measures above. The user decides which arms and which representation go
   on, whether coarse to fine stays, and whether fixed-lag growth or
   lookahead runs.

**Stage 4: scale and write-up.**

5. **M1S1, V1V2 and the four superblocks** with the arms that go on: the
   measures, the compressibility measure, the memory model from measured
   sizes, and the local re-solve timing. Torch on MPS for the kernels that
   run at this scale, checked against the reference.
6. **Write-up.** A report in `reports/`, and the stored form proposed as a
   change to `docs/version-2-plan.md` §4b–4d, with the no-overlap rule and
   the signed-distance view in §5c, step 3. The format spec comes with
   its own plan.

## Decisions for the user

- **D1. The form of the candidates.** Discrete candidates (a fan of steps
  round the heading) with the mean field above, or a continuous Gaussian
  form, which keeps the posterior in closed form (the fused-sulci report's
  argument). The plan assumes discrete, because the top solver is
  discrete and works.
- **D2. The contrast match.** Which echoes, the band scale, and the form of
  the border weight.
- **D3. The sites.** Whether the fundus box's painted labels are enough, and
  which marked points to add.
- **D4. The τ step** Δτ, and the number of candidates kept per node.
- **D5. The coarse level.** The cell size (the plan assumes the top stage's
  2 voxels), the coarse τ step, and how wide the coarse priors are.
- **D6. The field grid** (arm C): the coarsest spacing, the indicators'
  weighting, and the hysteresis between refining and coarsening. The
  spacing scan gives the evidence.

## Not in this plan

- Any change to the ray sampler or to "normal rays stay straight" (user,
  #47). A result here is evidence for a separate decision.
- The format spec of the stored columns.
- The viewer's column preview.
- **A hierarchical model**, in which coarse nodes stay in the model as
  latent variables (for example a bank's shared offset or thickness) and
  both levels are solved together. It is the next step if the coarse level
  shows bank-level structure. Plain long-range links between columns are
  not used: they change the energy and count the same evidence many times,
  so the probabilities become too confident.
- **Wavelets in laminar coordinates.** In the correct laminar coordinates
  the cortex's contrast should be highly compressible: it varies along
  depth and is nearly constant across. The column solve is, in this view,
  a search for the coordinates that give the shortest description of the
  image plus the map. Later work extends the octree and the lifting step
  to a full wavelet representation in those coordinates: as the objective,
  as a compact form for the 50 µm data, and for the viewer.
- The laminar depth metric. The τ levels are contrast levels. Equal-volume
  depth within a column stays a separate role.
