# Wave 2.5, step 2: signals per column

Session S. Read the plan's §1 (working rules), §4b (*What a correct column
looks like*) and §9 (wave 2.5). The rules in `docs/plans/round-2.5.md` and
`round-2.md` apply.

## Goal

One batch command computes, for the whole hemisphere, the signals that show
where a column can be wrong. It writes them as published per-vertex,
per-edge and per-face files in one format, `column-signals/1`. The viewer
(session V) shows them as an overlay and as ranked lists.

Each signal comes from a working version. Each signal has its provisional
limits in the sidecar, so that the user does not blame the column for an
error of the signal.

The signals:

| Signal | From (working version) | Domain |
|---|---|---|
| The angle between the data normal and the surface normal | 1.2 `orient` (#130, #135) | vertex |
| The mean coherence at the root | 1.2 `orient` | vertex |
| The data normal | 1.2 `orient` | vertex |
| Misfacing | `wm_sites.misfacing` (#121) | vertex |
| The distance to the first focal point, outward and inward | 2.1 equivolume (#125) | vertex |
| The start and the length of the first cortex run | 2.0 rays (intervals) | vertex |
| The neighbour shift | 3.1 pair-shift step (#131) | edge, face, vertex |

## Done when

1. `python -m baillarger.column_signals` writes all the signals of the
   format below on the new input set (`scratch/v2/derivatives-main/`).
2. The spec `docs/formats/column-signals.md` exists, at version 1, and
   `tests/test_bids.py` checks its sidecar table.
3. The exact tests below pass.
4. The PR bodies give the hemisphere run's time and peak, a map of each
   signal on the overview (PNG), its histogram, and its provisional limits.
5. `nix develop --command scripts/check` passes on each PR.

## Facts from the current code

All references are to `main` at 1f67376.

### The data normal (1.2)

- `orient.OrientSampler(echoes, dseg, cells=64, cache_bytes, noise)`
  (`orient.py:480`). `.orient(superblock, level, halo_mm, sigma_mm=0.64,
  device="numpy") -> Orientation` (`orient.py:583`).
- `Orientation` is per voxel on the box: `normal` (3, box), `coherence`
  (box), `sign` (box). It is NaN outside cortex. Nothing per voxel is
  stored (`docs/formats/orient.md:6-8`).
- `orient` needs the noise file `_desc-orientnoise_noise.json`. The command
  `python -m baillarger.orient_noise --echoes ... --dseg ... --out DERIV`
  writes it (`orient.md:275`). The new input set does not have it yet.
- `orient_diag.surface_agreement(x, n, normal, coherence, sign, box,
  affine) -> (angle_deg, weight)` (`orient_diag.py:20`) reduces it to one
  value per root. It samples at 0.5 and 1.0 voxel outward along the vertex
  normal (`SAMPLES`, `orient_diag.py:17`), forms m = Σ c·s·d, and gives
  the angle between m and n_v. `weight` is Σ c, a sum, not a mean. It does
  not return the voxel count or m.
- The prototype driver `prototypes/scripts/orient_hemi.py` loops over the
  cortex superblocks (`orient_noise.superblocks`), with a halo of 2 voxels,
  uses the owned roots of `RaySampler.roots` and their published normal, and
  writes resumable per-superblock `.npz` files. On the old set (#135): 119
  superblocks at level 1, numpy, 1 core, σ 0.64: 43 min, peak 9.3 GB.
  Median agreement about 12°, 45 % within 10°, 6.6 % above 45°.
- Provisional limits: σ is open (0.64 against 0.96, plan §3c). The sign is
  0 where the WM distance is not defined (cap 5 mm). The coherence is low at
  contacts and fundi (0.05–0.38). The local level does not stop at fused
  banks. A small dependence on the array axes.

### Misfacing

- `wm_sites.misfacing(v, f) -> Bool` (`wm_sites.py:140`) is per face. A
  face is misfacing when its normal · (N[a] + N[b] + N[c]) ≤ 0, with N the
  raw area-weighted vertex normals of `decimate.normals`, not the published
  normals. It runs on the whole mesh in chunks.
- No per-vertex or per-face array is stored. `write_sites` stores only the
  site rows. Cost on the whole mesh (#121): 19 s, peak 5.3 GB.

### The focal points (2.1)

- `RaySampler.sample(..., method="equivolume")` fills `Rays.focus` (n, 2):
  r_lo < 0 and r_hi > 0 in mm, ±inf where there is none, (0, 0) where
  A(0) ≤ 0 (`rays.py:134`). The values do not depend on `e_in` and
  `e_out`.
- The prototype `prototypes/scripts/equivolume_hemisphere.py` writes a
  per-vertex `focus.npy`. Cost on the old set (#125): level 1, 107
  superblocks, 13.76 M columns, 238 s, peak 9.2 GB.
- Provisional limit (#125): the 1-ring A(s) follows voxel-scale curvature.
  37 % of the columns have an outward focal point within 3 mm. The map is
  speckle and does not follow the folds.

### The first cortex run

- `Rays` intervals are flat arrays: `interval_start` (n+1), `interval_s0`,
  `interval_s1` (mm), `interval_label`, `interval_class` (`rays.py:98-129`).
  They tile [−2, 8] mm (`CAP_IN`, `CAP_OUT`) and do not depend on the
  extents or the step. `CORTEX` is class 2 (`rays.py:57-61`).
- `flattening.extents(rays, margin=0.5)` (`flattening.py:99`) finds the
  first cortex interval with s1 > 0. It returns padded extents, not the
  run itself.
- Provisional limits: the run is capped at 8 mm. An interval ends at any
  label change, so one stray voxel of another label breaks a run. Fused
  banks give long runs (39 % of V1V2 rays above 3.5 mm, #135).
- Cost: a normal-method pass on the hemisphere is about 20–26 s, peak
  7.4–8.2 GB (#108, #125).

### The neighbour shifts (3.1)

- `flattening.pair_shifts(F, M, pairs, step, energy_ref=None, chunk) ->
  PairShifts` (`flattening.py:357`): `shift` (P, n) and `weight` (P, n). The
  shift varies with depth: one value at each sample of the pair's first
  column. It wraps `flatten.shifts` (`flatten.py:73`), a 1-D Lucas–Kanade
  that **clips each shift to ±0.3 mm** (`smax`).
- The features: `pad(cv)` (`flattening.py:147`) and `features(p, scales)`
  (`flattening.py:220`): echoes 2 and 3, log minus a 1-D trend of 0.64 mm,
  divided by their SD. `scales` and `energy_ref` depend on the call unless
  they are given (#131, finding 3).
- `mesh_edge_pairs(faces, vertex, voxel) -> (P, 2)` (`flattening.py:285`)
  gives each white-mesh edge between the columns of a call once.
  `orient(pairs, edge_key)` (`flattening.py:272`) puts the smaller edge key
  first, so that a pair gives the same result in every call.
- `flatten_superblock` (`flattening.py:491`) runs the whole chain and then
  the warp solve. No function stops after the shifts. No code sums shifts
  round a face.
- Cost: on the M1S1 superblock (7, 4, 7), 53,026 columns and 157,251 edge
  pairs (#131). The cost of the shifts alone is not measured. The column
  volume samples at about 5 M samples/s with numpy, about 6 min for the
  hemisphere (#109).
- The whole-mesh edges: `mesh_smoothing.edges(f)` (`mesh_smoothing.py:36`),
  (k, 2), sorted.

### Batch runs and published files

- Superblocks: `np.unique(sampler.block_ijk >> m, axis=0)` in Morton order,
  or `orient_noise.superblocks(s, level)` for the cortex boxes. The owned
  roots of all superblocks cover each vertex once. Level 1 at 160 µm.
- No `src/` module loops over superblocks to write a per-vertex output. The
  per-vertex files that exist (normals, normlen, edgekeys) are written in
  one whole-mesh pass by `write_white_surface`.
- A per-vertex file: `derivative_path(root, ents, "white", ".func.gii",
  hemi=h, desc=<name>)` (`bids.py:53`), with the data in an external `.dat`
  through `gifti.write_external` (`gifti.py:55`; f32, i32, u8; 1-D or 2-D).
  Sidecar helpers: `bids.write_sidecar`, `provenance`, `sources`.
- The overview: `WhiteOverview.load(path)` gives `vertex_ids` and the carry
  CSR (`white_overview.py:202`). A full-mesh value maps to the overview as
  `value[vertex_ids]`.

## The format: `column-signals/1`

All files are next to the white surface, with BIDS names
`<sub>_<ses>_hemi-<h>_desc-<name>_white.func.gii` and an external `.dat`.
One sidecar, `<sub>_<ses>_hemi-<h>_desc-signals_white.json`, describes all
of them.

### Vertex signals (one row for each vertex of the white surface)

| desc | dtype, shape | Unit | Value |
|---|---|---|---|
| `orientangle` | f32 (n,) | degree | angle between m and the published normal, in [0, 180]; NaN where no voxel counts |
| `orientcoh` | f32 (n,) | — | Σ c divided by the number of voxels that count; NaN where none |
| `datanormal` | f32 (n, 3) | — | m / \|m\| in RAS; NaN where no voxel counts |
| `misfacing` | u8 (n,) | faces | the number of misfacing faces that touch the vertex |
| `focusout` | f32 (n,) | mm | r_hi; +inf where none; NaN where A(0) ≤ 0 |
| `focusin` | f32 (n,) | mm | −r_lo; +inf where none; NaN where A(0) ≤ 0 |
| `cortexs0` | f32 (n,) | mm | s0 of the first cortex interval with s1 > 0 (can be < 0); NaN where none |
| `cortexlen` | f32 (n,) | mm | s1 − s0 of that interval; NaN where none |
| `shiftmax` | f32 (n,) | mm | the largest \|`edgeshift` root value\| of the edges that touch the vertex |
| `circmax` | f32 (n,) | mm | the largest \|`facecirc`\| of the faces that touch the vertex |

### Edge signals (one row for each edge of `mesh_smoothing.edges(f)`)

| desc | dtype, shape | Value |
|---|---|---|
| `edges` | i32 (P, 2) | the vertex pair (a, b), a < b, in the order of `mesh_smoothing.edges` |
| `edgeshift` | f32 (P, 2) | the shift of b's profile against a's, in mm: column 0 is the weighted mean over the first 1 mm of a's first cortex run (the root value), column 1 over the whole run |
| `edgeweight` | f32 (P,) | Σ weight over the samples of the root value |
| `edgeclip` | f32 (P,) | the weighted share of the samples of the root value where \|shift\| = smax |

### Face signals (one row for each face of the white surface)

| desc | dtype, shape | Value |
|---|---|---|
| `facecirc` | f32 (m,) | the sum of the signed root values of `edgeshift` round the face, in the face's winding order, in mm; NaN if an edge is NaN |

For correct columns the shifts come from a smooth depth field, so the sum
round a face is near 0. A sum far from 0 means that no depth field fits the
three shifts.

### The sidecar

- `SpecVersion`: `column-signals/1`.
- `Sources` and `SourcesSHA256`: the white surface, the dseg, the echoes,
  and the noise file. `Provenance`.
- `InputSet`: the derivatives root, and the SHA-256 of the white-surface
  sidecar.
- `Level`, `HaloMM`, `SigmaMM`, `Method` (of the rays), `Scales` and
  `EnergyRef` (the fixed values of the pair-shift step), `SMaxMM`.
- `Signals`: for each desc, `File`, `Domain` (vertex, edge or face),
  `Unit`, `From` (the working version and its PRs), `LookAt` (`high` or
  `low`: the end of the range where the user looks first), and `Limits`
  (the provisional limits, as text).
- A signal that a run did not make is not in `Signals`. The viewer shows
  only the signals that are in `Signals`.

## Design

### Part A: the command and the vertex signals (PR 1)

1. A new module `src/baillarger/column_signals.py`. CLI:
   `python -m baillarger.column_signals SURF --echoes E1 E2 E3 --noise NOISE
   --work DIR [--only orient,misfacing,focus,cortex,shifts] [--level 1]
   [--device numpy]`. It writes next to SURF.
2. The work directory holds one resumable result for each superblock and
   each signal group. A rerun skips the superblocks that are done. A final
   step collects them into the published files. This is the pattern of the
   prototype drivers.
3. The orient group: the logic of `prototypes/scripts/orient_hemi.py`,
   moved into `src/`. Add the voxel count and m to the reduction without
   a change to `surface_agreement`'s return value: a new function next to
   it, which `surface_agreement` calls. Keep the halo of 2 voxels and σ
   0.64.
4. The misfacing group: one whole-mesh pass. For each vertex, the number of
   misfacing faces that touch it.
5. The focus group: `RaySampler.sample(..., method="equivolume", e_in=0,
   e_out=0)` for each superblock, owned roots only.
6. The cortex group: one normal-method pass. A new function
   `rays.first_cortex_run(rays) -> (s0, s1)` gives the interval that
   `flattening.extents` uses. Do not change `extents`. A test shows that the
   two find the same interval.
7. The noise file: run `orient_noise` on the new input set first, as its own
   step. Its time and peak go in the PR body.

### Part B: the neighbour shifts (PR 2)

1. Split `flatten_superblock` into the shifts and the solve: a new function
   `flattening.edge_shifts(...)` returns the `PairShifts` of a call and the
   data that the solve needs. `flatten_superblock` calls it. Its results do
   not change, byte for byte (test).
2. Fixed `Scales` and `EnergyRef`: a first pass computes them from the
   owned columns of all cortex superblocks. All calls use them. They go in
   the sidecar.
3. Each edge belongs to the superblock that owns its first column (the
   smaller edge key, as `orient` gives). The halo is large enough to sample
   the second column. A test shows that an edge across a superblock border
   gets the same value as in one call that holds both columns.
4. The reduction to the two edge values and the face sums, as the format
   gives. The sign: the shift of b against a, with a < b by vertex id in the
   published file, whatever order the call used.
5. Measure one superblock first (M1S1, (7, 4, 7)): time, peak and the share
   of clipped edges. Send them to the coordinator before the hemisphere run.

### Provisional limits to state in the sidecar

- `orientangle`, `orientcoh`, `datanormal`: those of 1.2 above.
- `misfacing`: it uses the raw vertex normals, not the published ones.
- `focusout`, `focusin`: the 1-ring A(s) follows voxel-scale curvature
  (#125).
- `cortexs0`, `cortexlen`: the cap at 8 mm, broken runs, fused banks.
- `edgeshift`, `facecirc`, `shiftmax`, `circmax`: the shift is clipped at
  ±0.3 mm, so a large jump shows as a value at the clip (see `edgeclip`),
  not as its true size. It needs profile contrast. The stria makes its
  contrast differ by area. Low `edgeweight` means little evidence.

Do not raise `smax` to remove the clip in this phase. Report the share of
clipped edges, with a map. Whether to raise it, or to use
`flatten.dynamic_shifts`, is a question for the review.

## Exact tests

- Each vertex signal on a phantom equals the value of the function it comes
  from, called directly on the same columns.
- Planar layers at a known tilt θ against the surface: `orientangle` is θ
  within the phantom's tolerance, and `datanormal` is the layer normal.
- The owned roots: each vertex has one value. The vertex signals for level
  0 and level 1 are equal (byte for byte for misfacing, the focus and the
  cortex run; within a stated tolerance for orient).
- Resume: a run that stops after k superblocks and starts again gives the
  same files, byte for byte, as one run.
- `first_cortex_run` and `flattening.extents` find the same interval.
- Two regions of columns with layers offset by δ < smax: the edges across
  the border have root value δ, the other edges 0. With δ > smax:
  `edgeclip` is 1 on the border edges.
- A smooth depth field gives `facecirc` near 0 on every face. A phantom
  with a step that ends inside the mesh gives large `facecirc` on the faces
  round the end of the step.
- An edge across a superblock border has the same value as in one call.
- `flatten_superblock` gives the same result as before the split, byte for
  byte.
- The sidecar passes `bids.write_sidecar` with the spec, and
  `tests/test_bids.py` checks the spec table.

## The hemisphere run

1. Wait for the coordinator's "go". The coordinator first finishes the new
   input set and its comparison.
2. Run `orient_noise`, then each group of Part A, one at a time, under
   `scripts/memguard`, each after a slot from the coordinator. Expected
   peaks from the old set: orient 9.3 GB (43 min), focus 9.2 GB (4 min),
   cortex about 8 GB, misfacing 5.3 GB.
3. Part B: after the one-superblock measurement, ask for a slot with the
   expected peak and time.
4. Work directory: `scratch/wave-2.5-signals/work`. Figures:
   `scratch/wave-2.5-signals/figures`, and the ones for the PR body in
   `reports/figures/column-signals/` (PNG, each well under 1 MB).
5. A check across input sets, by statistics only: the orient angle's
   median, its share within 10° and its share above 45°, against the old
   set's 12°, 45 % and 6.6 % (#135). Report the difference. Do not compare
   vertex by vertex.

## Work breakdown

| Step | Work | Who | Can run with |
|---|---|---|---|
| 1 | The spec `docs/formats/column-signals.md` and its `test_bids.py` row | Phase session | 2 |
| 2 | The misfacing and cortex groups, `first_cortex_run`, their tests | Sonnet sub-agent | 1, 3 |
| 3 | The command, the work directory, resume, collect; the orient and focus groups | Phase session (Opus) | 2 |
| 4 | The tests of step 3 on phantoms | Sonnet sub-agent | after 3 |
| 5 | PR 1's hemisphere run and figures | Phase session | after 4 |
| 6 | Part B: `edge_shifts`, the split test, the fixed scales | Phase session (Opus) | 5 |
| 7 | Part B: the edge and face reduction, its phantom tests | Sonnet sub-agent, the phase session reviews | after 6 |
| 8 | PR 2's one-superblock measurement, then the hemisphere run | Phase session | after 7 |

## PRs

1. `feature/column-signals`: the spec, Part A, and its hemisphere run.
   Branch from `origin/main`.
2. `feature/neighbour-shifts`: Part B and its hemisphere run. Stacked on PR
   1.

Tell the coordinator when each PR opens.

## Decisions for review (made by the coordinator)

- One batch command for all the signals, with one sidecar, so that the
  viewer reads one format. The groups run one at a time.
- The root value of the edge shift is the weighted mean over the first 1 mm
  of cortex, because the samples are anchored at the WM and a jump near the
  root is the strongest sign (§4b). The whole-run value is stored next to
  it.
- `Scales` and `EnergyRef` are fixed for the whole hemisphere, so that an
  edge's value does not depend on the superblock.
- The clip at ±0.3 mm stays. The share of clipped edges is a signal of its
  own.
- No thresholds. The ranked lists in the viewer order sites by value.
