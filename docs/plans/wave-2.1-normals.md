# Wave 2.1: a review of the white surface's normals

Wave 2.1 of `docs/version-2-plan.md` (§9, *Waves*). Read the plan's §1
(working rules) and §4b (*What a correct column looks like*), and
`docs/formats/white-surf.md`. The rules in `docs/plans/round-2.md` apply,
with the changes in this file.

"Wave 2.1" is not phase 2.1 (the equivolume cross-section). Always write
"wave 2.1" in full.

## Why this wave comes first

Every column is cast along the published normal of its root. If the normals
change at the voxel scale between neighbours, every column inherits that
error. A review of the column samples then finds normal problems, at a
higher cost.

There is evidence that this can be so. Phase 2.1 (#125) found an outward
focal point within 3 mm for 37 % of the columns, in a speckle pattern that
does not follow the folds. Two neighbouring rays at an angle θ and a
distance d apart cross at about d / θ. So the focal points are a
normal-to-neighbour signal. #125 blames its 1-ring estimate of A(s). The
normals themselves can also be the cause. Wave 2.1 tells the two apart.

What wave 2.1 finds can change wave 2.5. The wave 2.5 plans (#141) wait
until wave 2.1 is done.

## Goal

1. The input set has the new names of the normals (#142).
2. Seven maps on the white mesh, from the mesh and its normals only: no
   echoes, no rays, no `orient`.
3. A generic overlay in the viewer: any per-vertex GIFTI map on the
   overview, with a colour bar and a ranked list that opens the vertex.
4. The user reviews the maps and the sites they rank. The review finds the
   types of error in the surface and its normals, and the prevalence of
   each type.
5. Each error type gets a decision: repair now, or defer with a recorded
   provisional limit. Wave 2.5 starts only when every known type has a
   decision.

These maps test whether the normals agree with themselves and with the
mesh. They do not test whether the normals are correct. Normals can be
smooth and point the wrong way. The data normal (`orient`) is the
independent check, later.

## Steps

| Step | What | Who |
|---|---|---|
| 1 | Merge #142, after the user's instruction that names it | the coordinator |
| 2 | Make `white_surf` and `white_overview` again on `scratch/v2/derivatives-main/`, one at a time under `scripts/memguard` | the coordinator |
| 3 | The normal maps (Part A) | session N |
| 4 | The generic overlay (Part B) | session N |
| 5 | The review: the error types and their prevalence (see "The error types") | the user, with the coordinator |
| 6 | A decision for each type: repair now, or defer. A repair that changes code or a method gets its own plan and PR (see "Repairs"). After each repair, the input set is made again, and the counts are measured again | the user decides; the coordinator writes the repair plans |
| 7 | The changes to the wave 2.5 plans (#141) | the coordinator, with the user |

Steps 5 and 6 repeat, one type at a time, until the stop rule holds.

Step 3 starts after step 2. Step 4 can start at once: it tests on phantom
files.

### Step 2 in detail

1. `white_surf` and `white_overview` run from `main` after #142, with the
   commands of `scratch/v2/logs/main/` (step 3 and step 4). The other files
   of the set do not change.
2. The old `desc-normals` and `desc-normlen` files move to
   `scratch/v2/derivatives-main-stale/`. They are not deleted.
3. The surface and the overview must be byte-identical to the set made
   before #142, apart from the sidecar. #142 changes only names and a key.
   If they differ, stop and report.

## Facts from the current code

All references are to `main` at 1f67376, with #142.

- The published normals: `desc-smoothnormals_white.func.gii`, (n, 3) f32,
  unit, in the voxel axes of the `dseg`, not scanner RAS: `white_surf`
  smooths them on the vertices in voxel coordinates and writes them with no
  transform. For the current input set the affine is diagonal and positive
  (0.16028, 0.16, 0.16028 mm), so the difference is at most about 0.1°.
  `docs/formats/white-surf.md` does not name the frame yet. The area-weighted vertex normals take
  `NormalSmoothingSteps` (10) steps. Each step sets n to 0.5 n + 0.5 × (the
  mean of n over the neighbours), then divides n by its norm
  (`mesh_smoothing.smooth_normals_and_length`, `mesh_smoothing.py:139`).
- |n̄|: `desc-smoothnormlen_white.func.gii`, (n,) f32, the norm of m. m
  takes the same 10 steps with no division. So m is linear in the start
  normals: m = M¹⁰ n₀. The code returns |m|, not m.
- The raw normals: `decimate.normals(v, f)` (`decimate.py:125`), per vertex
  the sum of its faces' cross products (twice the area times the unit
  normal). Not unit.
- Misfacing: `wm_sites.misfacing(v, f)` (`wm_sites.py:140`), per face: the
  face's normal · (N[a] + N[b] + N[c]) ≤ 0, with N the raw normals. It runs
  in chunks on the whole mesh: 19 s, peak 5.3 GB (#121).
- The edges: `mesh_smoothing.edges(f)` (`mesh_smoothing.py:36`), (k, 2),
  each once.
- The overview: `WhiteOverview.load(path)` gives `vertex_ids` and the carry
  CSR (`white_overview.py:202`). `carried(faces) -> (ids, face)`.
  `prototypes/scripts/orient_hemi_figures.py:66` (`face_values`) gives a
  value per overview face from its corners and its carried true vertices.
- The viewer: the overview is grey (`colour3d`, `static/index.html:189-192`).
  The per-vertex `col` buffer exists (`GL.bcol`). No route serves per-vertex
  values. A click on the overview gives `selectVertex(v, frame)`, and the
  columns pane loads `V.selection.true_vertex`. The review queue table
  (`static/review.js:86-97`) is a table that opens a site.
- The mesh: 13,763,166 vertices and 27,526,328 faces, one piece. The
  overview: 172,285 vertices.

## Part A: the normal maps (PR 1)

1. A module `src/baillarger/normal_maps.py` holds the map functions and the
   ranked-list function, so that they have tests. No test imports from
   `prototypes/` today.
2. A prototype script, `prototypes/scripts/normal_maps.py`, runs them on a
   surface. Its arguments: the surface, an output directory and a figures
   directory. It writes one GIFTI file of named scalar maps,
   `normal_maps.func.gii` with an external `.dat`, in the output
   directory, in the form of the published normals (`gifti.write_external`).
   It also writes the figures and the ranked lists.
3. The maps file is not a published output, and it has no spec.

### The maps (one value for each vertex)

| Name | Value | Unit |
|---|---|---|
| `nbrmax` | the largest angle between the vertex's smoothed normal and the smoothed normals of its 1-ring neighbours | degree |
| `nbrmaxpermm` | for the same neighbour, that angle divided by the edge length | degree / mm |
| `normlen` | \|n̄\|, read from the published file | — |
| `renorm` | the angle between the smoothed normal n and m / \|m\| | degree |
| `smoothing` | the angle between the raw normal (made unit) and n | degree |
| `misfacingraw` | the number of faces that touch the vertex and are misfacing against the raw normals (the `wm_sites` rule) | faces |
| `misfacingsmooth` | the same, against the smoothed normals: the face's normal · (n[a] + n[b] + n[c]) ≤ 0 | faces |

Notes:

- `nbrmax` and `nbrmaxpermm` are the direct test: smooth normals give small
  angles that change slowly along the mesh. The angle per mm corrects for
  edges of different length.
- `renorm` shows where the division at each step changed the normal, against
  the linear smoothing. m needs a function that returns m, not only |m|.
  Put it in `normal_maps.py`. Do not change `mesh_smoothing`.
- `misfacingsmooth` matters most for the columns: the rays use the smoothed
  normals.
- A vertex that is not in any face, or has no neighbours, gets NaN.

### The outputs for the review

In the figures directory:

1. A histogram of each map, with its median, its 99th and 99.9th
   percentiles, and the share above a few round values (for example 10°,
   30° and 60° for an angle). These are for description, not pass or fail.
2. A map of each on the overview, from lateral and medial views, as PNG.
   Each overview face shows the largest value of its corners and its carried
   true vertices, so that one bad true vertex is not averaged away. Each PNG
   is well under 1 MB.
3. A ranked list for each map, as a TSV: take the largest value, keep its
   vertex as a site, drop the vertices within 1 mm of it, and repeat, for
   200 sites. Columns: rank, vertex, edge key, RAS x, y, z, the value, and
   the other six maps at that vertex. No threshold.
4. Scatter plots, as density plots: `nbrmax` against `normlen`, and `renorm`
   against `normlen`. They show whether the maps give the same places or
   different ones.

The figures that go in the PR body go to `reports/figures/normal-maps/`.

### Exact tests

In `tests/test_normal_maps.py`, for the functions of
`baillarger.normal_maps`:

- A plane: every map is 0, except `normlen`, which is 1.
- A sphere of radius R with mean edge length h: `nbrmaxpermm` is about
  (180 / π) / R, and `renorm` is about 0.
- \|m\| from `normal_maps` equals the published `normlen` within 1e-5, and
  `smooth_normals_and_length`'s length.
- `misfacingraw` gives the same faces as `wm_sites.misfacing`.
- A phantom with one face turned over: `misfacingraw` and
  `misfacingsmooth` count it at its three corners.
- The ranked list: no two sites within 1 mm, the first site is the largest
  value, NaN is never a site.

### The run

On `scratch/v2/derivatives-main/`, after step 2. Output:
`scratch/wave-2.1/maps`. Figures: `scratch/wave-2.1/figures`. Expected
peak: several GB for 13.8 M vertices and 27.5 M faces. Measure it on the
overview mesh first, then ask the coordinator for a slot if it goes above
8 GB.

## Part B: the generic overlay (PR 2)

1. A new module `viewer/overlay.py`. The flag `--overlay PATH` (repeatable)
   gives a GIFTI file of scalar maps on the white mesh. The server refuses a
   file whose Dim0 is not the white surface's vertex count. Without the
   flag, the overlay routes give 404 and the page hides the overlay
   controls.
2. Routes:
   - `GET /overlay/meta`: for each file and each map: the file, the map's
     name (its GIFTI `Name`), and its 1st, 50th and 99th percentiles of its
     finite values.
   - `GET /overlay/<file>/<map>?agg=max|min`: f32, one value for each
     overview vertex: the largest (or smallest) value of the overview vertex
     and of the true vertices carried by the overview faces that touch it.
   - `GET /overlay/<file>/<map>/sites?n=200&spacing=1.0&order=high|low`: the
     ranked list, as in Part A.
   - `GET /overlay/at?v=...`: all maps at the given true vertices.
3. A new page module `static/overlay.js`:
   - a map choice, `max` or `min`, a sequential colour map, a range from
     the 1st to the 99th percentile that the user can change, and a colour
     bar;
   - it writes `GL.bcol` through a new `colour` hook in `index.html`. "None"
     gives the grey of today;
   - the ranked list as a table (rank, value, vertex). A row click selects
     the true vertex through the shared state and turns the 3D view to it,
     and the columns pane loads it;
   - the selected vertex's values of all maps in a small table.
4. Nothing in Part B knows about normals. It shows any per-vertex map. Wave
   2.5 can use it for its signals.

### Exact tests

- `/overlay/<file>/<map>` on a phantom surface and overview equals the
  max-of (or min-of) computed with numpy from the carry CSR.
- The ranked list: as in Part A.
- A file with the wrong vertex count: the server refuses to start.
- Without `--overlay`, `main()` serves every route of 4.0, 4.0.5, 4.2 and
  4.3, and the overlay routes give 404 (extend
  `test_the_white_mode_serves_every_route`).

### The check

In the built-in browser pane, on the new set with Part A's file: show each
map, open the first three sites of `nbrmax` and of `misfacingsmooth`, and
show the columns pane at one of them. Give screenshots, the time of each
route, and the server's peak memory in the PR body.

## Work breakdown

| Step | Work | Who | Can run with |
|---|---|---|---|
| 1 | Part B: `viewer/overlay.py`, its routes and tests, on phantom files | Sonnet sub-agent, the session reviews | 2 |
| 2 | Part A: `normal_maps.py` and its tests | Phase session (Opus) | 1 |
| 3 | Part B: `static/overlay.js` and the `colour` hook | Phase session | after 1 |
| 4 | Part A: the run and the figures | Phase session | after 2 and step 2 of the wave |
| 5 | Part B: the check, with Part A's file | Phase session | after 3, 4 |

## PRs

1. `feature/normal-maps`: Part A (the module, the script and the tests) and
   its figures. Branch from
   `origin/main` after #142 merges.
2. `feature/viewer-overlay`: Part B. Branch from `origin/main`.

Tell the coordinator when each PR opens.

## The review (step 5)

The goal of the review is a catalogue of the error types and the prevalence
of each. The repair plan comes after the catalogue, from the scale and the
scope of each type. Do not build repair tools before a type is in the
catalogue.

The user reviews the maps in the overlay, the figures, and the sites that
the ranked lists open. The questions that lead the review:

1. Do the smoothed normals change at the voxel scale between neighbours, and
   where: over the whole surface, or in some kinds of places (thin blades,
   fundi, crowns, topology fixes, excision walls)?
2. Do the high values sit on surface errors or on real anatomy?
3. Do the misfacing faces against the smoothed normals sit in the same
   places as against the raw normals?
4. Does the renormalisation change the normal much anywhere?

### The loop for one error type

1. **Identify.** The user finds a site, and the coordinator finds its cause
   with views of that site. Record the type: its definition, the stage that
   makes it (input segmentation, `white_dseg`, the meshing, the normals, the
   columns), its effect on the columns, and one or two example sites.
2. **Count.** Find the candidates of the type with the simplest signal that
   catches it. The count of candidates is an upper bound. Record what the
   signal cannot find, so that a low count does not hide the type.
3. **Sample.** The user reviews a random sample of the candidates (about 10
   to 20), spread over their sizes. The share that are errors (the
   precision) times the count gives the prevalence, as a count of sites and
   as an area.
4. **Decide.** Repair now, or defer. A deferral records the type as a
   provisional limit (plan §1, working rules), so that later results do not
   blame its errors on another component. A repair of tens of sites can go
   through the review queue as `dseg` manual edits. A repair of thousands of
   sites needs a change to the method, with its own plan (see "Repairs").
5. **Measure again.** After a repair, make the input set again and measure
   the counts of every type again.

Repair in the order of the pipeline: the input segmentation first, then
`white_dseg`, the meshing, and the normals. A repair upstream changes the
counts downstream. For example, a repair of the WM label at type A sites
removes some type B slivers and some column fans.

### Repairs

Wave 2.1 decides which types to repair. It does not hold the repair work.

- A repair that changes code or a method gets its own plan in
  `docs/plans/`, on its own branch, with its own PR. The plan gives the
  design, the tests, and the counts to measure again. The user approves the
  plan before the work starts.
- A repair by `dseg` manual edits at tens of sites goes through the review
  queue. It needs no plan of its own.
- The catalogue's "Decision" column links to the repair plan or PR.

Wave 2.5 waits until each repair is merged and measured again.

### The stop rule

The search for new types stops when the user reviews a random sample of
sites over the whole surface, not only the top of the ranked lists, and
finds no new type. Then every known type is a claim that can be checked.

### The error types

The current state of the catalogue. Input set: `scratch/v2/derivatives-main/`
(white-surface sidecar SHA-256 1d53bc66182c3760…), with the maps of #146.

| Type | Stage | Effect | Signal and candidates | Sample | Decision |
|---|---|---|---|---|---|
| **A. The WM label stops short.** The label ends at a flat, grid-aligned face, and the WM in the echoes goes on past it. Where the face meets the other side of the WM, the label is one voxel thick, and the surface folds back on itself. | input segmentation (it is in the excised `dseg`, before `white_dseg`) | the smoothing mixes the opposite normals of the two sides; the columns fan out | WM half-thickness (the largest EDT within 1 mm inward along the normal) ≤ 1 voxel and `normlen` < 0.9, merged at 2 mm: 165 regions, 45 of 0.5 mm² or more. At ≤ 1.5 voxels: 219 regions, 76 of 0.5 mm² or more. It cannot tell missed WM from WM that is really thin. It cannot find a stop where the surface does not fold. | 1 site (vertex 10909480: missed WM that the user judges repairable by hand) | open |
| **B. Sliver triangles.** Where the level set passes within a few µm of a grid node, the crossings on that node's edges fall almost on top of each other. Edges of 2–12 µm, against a typical 0.1 mm. | the meshing (marching tetrahedra, no remesh) | noisy raw normals and `nbrmaxpermm`; a small effect on the smoothed normals | `nbrmaxpermm` > 1000: 2.4 % of the vertices; 80 % of them are within 0.05 voxel of a grid node | the mechanism is confirmed | open. A candidate repair: before the marching, set each node value with \|φ\| < ε to ±ε. Each vertex stays on its own grid edge, so the edge keys and the guarantee of no self-intersection hold. |
| The largest thin regions (about 90 mm² each, near (11, −46, −25) and (−6, −41, −50) mm) | not known | not known | the 2–5 largest regions of type A's signal | none | not a type yet |
| Misfacing faces (question 3) | not known | not known | `misfacingsmooth` ≥ 1 at 0.032 % of the vertices, `misfacingraw` ≥ 1 at 0.013 % | none | not a type yet |

The frame of the published normals (voxel axes, see "Facts from the current
code") is a gap in the spec, not an error type. It goes to its own PR.

### When wave 2.1 is done

1. The stop rule holds.
2. Every type in the catalogue has a decision. Each repair, with its own
   plan and PR where it changes code or a method, is merged and measured
   again.
3. The changes to the wave 2.5 plans (#141) are written. Then #141 can be
   taken out of draft.

Each finding names the input set and the map.

## Decisions for review (made by the coordinator)

- The maps are a prototype output with no spec. Their functions are in
  `src/` so that they have tests. They are in standard GIFTI,
  so the overlay and other tools can read them.
- The overlay is generic: it shows any per-vertex GIFTI map. It does not
  depend on what wave 2.5 decides about its signals.
- The overview shows the largest value under each overview vertex, not a
  point sample, so that a bad true vertex is not hidden by the decimation.
- The ranked lists drop the vertices within 1 mm of a site. No threshold.
