# Phase 3.1: flattening

Stream 3, phase 1 of `docs/version-2-plan.md` (§5d). Read the plan's §2
(*Domain edges*), §5c steps 1 and 2, and the stream 3 scaling track, item 1.
The rules in `docs/plans/round-2.md` apply.

## Goal

Flattening on the column-volume sampler, per superblock, with a solve that
is local: shifts between neighbouring traces, and the fitted warp, for any
part of the hemisphere. Phases 3.3 and 3.4 build on it in wave 3.

## Done when

1. A function flattens the columns of one superblock and its halo, from
   the column-volume sampler, and returns the shifts and the warp of the
   owned columns.
2. The local solve agrees with the global solve (`flatten.fit_warp`) within
   a stated tolerance, on synthetic data and on a real region. The result
   does not depend on where the superblock boundaries fall, within the same
   tolerance.
3. The PR body gives the analysis of the solve at 50 µm whole brain (see
   "Analysis").
4. It runs on the hemisphere at the three test sites (V1V2, M1S1,
   PCC_SPL). The PR body gives the diagnostics in "The hemisphere run".
5. `nix develop --command scripts/check` passes.

## Facts from the current code

### `src/baillarger/flatten.py` (356 lines)

- numpy and scipy only. It imports nothing from `baillarger`. It does not
  use the old pipeline modules.
- `shifts(Fv, Fw, Mv, Mw, step, sig_d=0.16, sig_w=0.32, n_iter=10,
  smax=0.3, s0=None)` (74): Lucas–Kanade shifts between pairs.
- `dp_lags` (126), `dynamic_shifts` (157): dynamic warping (#57, #60). The
  plan uses it for long-range matching only. Leave it as it is.
- `pair_weights` (232).
- `fit_warp(pairs, S, W, n_traces, step, dtau=0.08, tau_max=6.0, eps=0.2,
  n_outer=4, anchor=10.0, maxiter=3000, rtol=1e-7, r0=None, log=None)`
  (241): unknowns r (N, K), K = tau_max/dtau + 1. Data rows, smoothness
  rows, and anchor rows r_v(0) = 0. Gauss–Newton with 4 outer passes. Each
  inner solve is one `scipy.sparse.linalg.cg` over all columns, with a
  Jacobi preconditioner. This is the global solve.
- `residuals`, `resample`, `tau_of`.
- `tests/test_flatten.py` (330 lines): synthetic grids of traces with
  known warps.

### Callers

- `viewer/filters.py` (`compute`, 244–320) and the prototypes
  (`prototypes/scripts/flatten_volume.py` and others) call it on the old
  `column_volume_<site>.npz`. Do not change them in this phase. Phase 4.3
  owns `viewer/filters.py`.
- `prototypes/scripts/flatten_volume.py` `fit()`: drops traces within 2 mm
  of a box face, divides echoes 2 and 3 by their SD, and makes pairs with
  `cKDTree(root).query_pairs(0.25)` kept where the normals agree (dot >
  0.7).

### The prototype result

- `reports/Crossing paths and contacts from laminar profiles.md` 495–515:
  V1V2 64,046 traces, 310,978 pairs, 5.3 min, 5.7 GB, weighted RMS misfit
  0.037 → 0.033 mm. M1S1 72,350 traces, 338,794 pairs, 6.1 min, 6.8 GB,
  0.046 → 0.042 mm. ε = 0.2, 4 outer passes, 1,600–2,000 CG iterations
  each.

### The column-volume sampler (`column_volume.py`, `column-volume.md`)

- `ColumnVolume(rays, echo (N, E) float32, valid (N,) bool)`. Rows are
  flat. Column c is `sample_start[c]:sample_start[c+1]`.
- Every column shares the grid s_k = k·step. So padding by k is exact.
- `ColumnVolumeSampler.sample(rays, device=None)`.
- `take_columns(rays, lo, hi)` makes batches.
- `PENDING` (22) is a stale string. Remove it.

### The mesh graph

- `mesh_graph.local_graph(v, f, blocks, superblock, level, cells, halo_mm,
  voxel_mm)` gives the graph of the superblock and its halo, with mesh
  edges and across-triangle edges in mm.

## Design

### Part A: the adapter

1. A new module, `src/baillarger/flattening.py`. Keep `flatten.py` as the
   solver library.
2. From a `ColumnVolume`, make the padded features (nf, N, n) and the mask,
   on the common k grid.
3. **Features:** echoes 2 and 3, each divided by its SD over the call's
   cortex samples, as in the prototype. Keep this choice. Record it in the
   docstring. (Stream 1's `orient` weighs all echoes by a noise model.
   Whether flattening should do the same is a plan finding for wave 3. Send
   it to the coordinator. Do not change it now.)
4. **Extents:** take each column from the start of the label's WM to the
   end of its cortex interval, with a margin of 0.5 mm each side, from the
   label intervals. This is the first-extent rule of §5c step 3, without
   the censoring (phase 3.3 adds that).
5. **Pairs:** columns whose roots are neighbours on the white mesh: the
   mesh edges of `local_graph`. This is the plan's rule (§5c step 2). Also
   make the prototype's pairs (0.25 mm, dot > 0.7) as a diagnostic, and
   compare the two on one site.

### Part B: the local solve

The anchor rows tie each column's r to 0 at τ = 0. So the coupling through
the data rows may decay with distance along the mesh. Measure it before you
choose the design.

1. **The decay.** On a real region, perturb the shifts of one column and
   measure the change in r of the other columns against their distance
   along the mesh. Give the curve. From it, choose the halo h (mm) where
   the change falls below 1 % of the warp's resolution (dtau · typical
   slope).
2. **If h is a few mm or less:** solve each superblock with its halo by
   `fit_warp` and keep the owned columns. This is a local solve.
3. **If h is larger:** overlapping Schwarz: solve the superblocks in Morton
   order with a halo, with the halo columns' r taken from the last pass of
   their own superblocks, and repeat until the change is below a stated
   tolerance. Give the number of passes.
4. Give the halo in mm, never in blocks (plan §2).
5. Leave the Gauss–Newton outer passes as they are.

Tell the coordinator which design the decay picks, and the halo.

### Part C: the result

1. A function `flatten_superblock(...)` that returns, for the owned
   columns: their edge keys, the shifts to each neighbour, the weights, τ,
   r, and the solve's log (CG iterations, misfit per pass).
2. No file and no spec. The output is internal (plan §5c step 2). Phase 3.3
   calls the function.

## Exact tests

- On the synthetic grids of `tests/test_flatten.py`, laid out as a phantom
  white surface with a column volume: the adapter's padded features equal
  the sampler's samples.
- The local solve equals the global solve within the stated tolerance, on
  a grid split into several superblocks.
- The result of a column does not change, within the tolerance, when the
  superblock level changes or the grid shifts by one block.
- Mesh-edge pairs on a plane phantom: each pair is a mesh edge, and each
  edge appears once.

## Analysis

Write this in the PR body (plan §1, working rules):

1. Unknowns, nonzeros and bytes of the normal matrix per column, and the
   working set of one superblock at 160 µm and at 50 µm.
2. CG iterations against superblock size, and the cost per column.
3. The read factor ((B + 2h)/B)³ for the chosen h, at each level.
4. Whether a torch version is needed. Give the arithmetic intensity of the
   CG step. Write a torch version only if the analysis shows that the
   solve does not fit or is far too slow at 50 µm. Otherwise, write how it
   would be done.

## The hemisphere run

Read `scratch/v2/derivatives/` and the echoes. Write to
`scratch/phase-3.1/`. A prototype box took 6–7 GB. Measure one superblock
first. Ask the coordinator for a slot for each run above 8 GB.

1. Flatten the superblocks that cover each of the three test sites, with
   their halos. A full hemisphere pass is not needed in wave 2. Give an
   estimate of its time and peak from the measured superblocks.
2. Diagnostics (views, not pass/fail numbers): the weighted RMS misfit per
   pass, against the prototype's numbers; flattened sections through the
   user's marked points (M1/S1 23, V1V2 35); maps of the misfit on the
   overview mesh; the local-against-global difference on one site.
3. Figures as PNG in `reports/figures/phase-3.1/`, each well under 1 MB.

## Work breakdown

| Step | Work | Who | Can run with |
|---|---|---|---|
| 1 | Part A: padding, features, extents | Sonnet sub-agent | 2 |
| 2 | Part A: mesh-edge pairs, and the prototype's pairs | Sonnet sub-agent | 1 |
| 3 | The synthetic phantom fixtures | Sonnet sub-agent | 1, 2 |
| 4 | Part B: the decay measurement on one site | Phase session | after 1 and 2 |
| 5 | Part B: the local solve or Schwarz | Phase session | after 4 |
| 6 | Part C and the tests | Phase session | after 5 |
| 7 | The analysis and the hemisphere run | Phase session | after 6 |

## PRs

1. `feature/flattening`: everything. Branch from `origin/main`. If it grows
   above about 1500 changed lines, put Part A in its own PR first and stack
   Parts B and C on it.

Tell the coordinator when each PR opens.

## Decisions for review (made by the coordinator)

- Features: echoes 2 and 3 divided by their SD, as in the prototype.
- Pairs: mesh edges of the white surface.
- First extents: label WM start to cortex end, with 0.5 mm margins.
- No torch version unless the analysis calls for it.
