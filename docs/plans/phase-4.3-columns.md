# Phase 4.3: columns in the viewer

Stream 4, phase 3 of `docs/version-2-plan.md` (§6d). Read the plan's §4c
step 4, §5c step 1, and §6. The rules in `docs/plans/round-2.md` apply.

## Goal

The viewer shows columns from the published samplers, for what is on
screen, anywhere on the hemisphere. It replaces the viewer code that reads
prototype `.npz` files: `viewer/data.py`, `filters.py`, `labels.py`,
`streamlines.py` and their parts of the page.

1. A columns pane in `--white` mode: the columns near the selected vertex,
   from the ray sampler, with their label intervals and their echo
   profiles from the column-volume sampler.
2. The filter-job pattern and the TCK writer, kept, on the new data.
3. The old site mode and its modules go, in a last PR.

## Done when

1. The columns pane runs on the hemisphere in `--white` mode. The PR body
   has screenshots: columns in the 3D view and on the brick tiles, a
   section of profiles, and the label intervals of one column.
2. No response scales with the whole mesh or volume. A request names a
   vertex and a radius, and the server samples only the superblocks that
   the radius meets.
3. The exact tests below pass.
4. `nix develop --command scripts/check` passes on each PR.

## Facts from the current code

### The samplers

- `rays.RaySampler(surf, dseg, cache_bytes)`; `.roots(superblock, level,
  halo_mm)`; `.sample(superblock, level, *, method="normal", e_in, e_out,
  step, halo_mm) -> Rays`. `Rays` holds flat arrays: column c's samples are
  `sample_start[c]:sample_start[c+1]`. It holds positions and label
  intervals, not echo values.
- `column_volume.ColumnVolumeSampler(echoes, voxel_to_ras, echo_affines,
  grid_shape)`; `.sample(rays, device=None) -> ColumnVolume(rays, echo (N,
  E), valid (N,))`. `take_columns(rays, lo, hi)`.
- A superblock is `cells << level` voxels a side. Level 1 at 160 µm on the
  laptop (#108).
- Phase 2.1 (another session) adds A(s) and V(s) to `Rays` in wave 2. Do
  not depend on them. If they arrive before you finish, show them as an
  option. Ask the coordinator.

### The old viewer code (site mode)

- `viewer/data.py` (589 lines, `Site`): reads `column_volume_<name>.npz`,
  `joint_<tag>.npz`, `pia_<name>.npz`, `edit_priority_<tag>.npz`, and
  echo 2. Methods: `locate`, `vertex`, `snap`, `use_filter`, `layers`,
  `section_at`, `mesh_payload`, `streamlines`, `tracts`, `meta`, and
  others.
- `viewer/filters.py` (439 lines): `Params`, `Result`, `FilterJobs` (331).
  The filter-job pattern: one job at a time in a thread, its state under a
  lock, results cached as `filter_<name>_<key>.npz`, the page polls `GET
  /filters` and posts `/filters/run` and `/filters/use`. It imports
  `baillarger.flatten`.
- `viewer/labels.py` (170 lines): the old pial-label review
  (`pial_labels_<name>.npz`).
- `viewer/streamlines.py` (164 lines): `tck(lines)` (130–143) and
  `read_tck` (146–164). The TCK body starts with a NaN row (NiiVue 0.69
  drops the first line), puts a NaN row after each line, and ends with an
  inf row.
- `viewer/sections.py` (215 lines): `section_curve` (105–192), used by
  `Site.section_at` and `GET /section`.
- `server.py`: site mode and labels-only mode use these. `--white` mode does
  not. `main()` (526–534) requires a column-volume `.npz` for site mode.
- Tests: `tests/test_viewer_data.py` (253 lines), `_filters.py` (253),
  `_labels.py` (71), `_streamlines.py` (169), `_sections.py` (207).
- `docs/plans/phase-4.0-viewer.md`, "Old code": phase 4.3 replaces these
  modules.
- `.claude/launch.json` has several entries for site mode in old
  worktrees.

## Shared files with other phases

`server.py` and `index.html` are shared with phases 4.0.5 and 4.2 (session
F). Put the new pane in `viewer/columns.py` and `static/columns.js`. In
`server.py` and `index.html`, add only the hooks.

PR 3 (the removal) changes `server.py` and `index.html` a lot. Do not start
it until the coordinator says that 4.0.5's and 4.2's PRs are open. Then
make PR 3's branch from `origin/main` and keep the conflicts for the user's
merge order, or stack it as the coordinator says.

`viewer/filters.py` imports `baillarger.flatten`. Phase 3.1 does not change
`flatten.py`'s API. Do not change `flatten.py`.

## Design

### Part A: the server side (`viewer/columns.py`)

1. `--white` mode builds one `RaySampler` and one `ColumnVolumeSampler`
   from the white surface's sidecar `Sources` and the echoes.
2. `GET /columns?v=&radius=&method=&every=`: the columns whose roots lie
   within `radius` mm (along the surface: use `mesh_graph.local_graph`)
   of vertex v, every n-th column. It samples only the superblocks that the
   radius meets, with the halo it needs. Return binary arrays: positions,
   s, label intervals, the derived class, and each column's edge key.
3. `GET /columns/profiles?...`: the same columns' echo samples, from the
   column-volume sampler.
4. `GET /columns/<key>.tck`: the columns as TCK, with the kept writer, for
   NiiVue and for export.
5. A limit on columns per request. Give it and the reason.
6. Cache by vertex, radius, method and the sources' SHA.

### Part B: the filter-job pattern

1. Move `FilterJobs` and the TCK writer into modules that do not read
   prototype files: for example `viewer/jobs.py` and `viewer/tck.py`. Keep
   their tests.
2. A column request that takes longer than about 2 s runs as a job. Give
   the measured times.
3. Flattening as a filter is not in this phase. Phase 3.1 publishes it as
   a function in wave 2. Its pane is phase 4.4's.

### Part C: the page (`static/columns.js`)

1. Columns in the 3D surface view, coloured by derived class or by the
   label of each interval.
2. Columns on the brick tiles, where they cross the tile's plane (on
   `#bvo`).
3. A profile section: the selected columns' echo profiles side by side,
   along s, with the root at s = 0 and the label intervals drawn, like a
   seismic section (plan §5c step 1).
4. One column's details: its label intervals in a table, its |n̄|, its
   class.
5. The selection drives everything through the shared state.

### Part D: the removal (PR 3)

1. Remove site mode and labels-only mode from `server.py`, and the routes
   that only they serve.
2. Remove `viewer/data.py`, `viewer/labels.py`, `viewer/streamlines.py`
   (after Part B moves the writer), and the parts of `viewer/filters.py`
   that read `.npz` files. Remove the page code that only they use.
3. Before you remove a module, list its importers (`git grep`). Prototypes
   in `prototypes/` that import a removed module: list them in the PR body.
   Do not change them.
4. `viewer/sections.py`: keep it if the new profile section uses it,
   otherwise remove it with its route. Say which in the PR body.
5. Remove the `.claude/launch.json` entries that start site mode.
6. The tests of removed code go with it. No other test changes.

## Exact tests

- `/columns` for a vertex and a radius returns exactly the columns that
  `RaySampler` gives for those roots, byte for byte.
- A request near a superblock corner gives the same column as a request
  far from it (the halo is right).
- The radius selection along the surface, on a phantom.
- The TCK writer: the tests move with it, unchanged.
- The job pattern: the tests move with it.
- After the removal, the server starts in `--white` mode and serves every
  4.0, 4.0.5 and 4.3 route.

## The hemisphere check

1. Start the viewer in `--white` mode on `scratch/v2/derivatives/` with the
   echoes. Add a `.claude/launch.json` entry for your worktree.
2. In the built-in browser pane: select a vertex at the V1V2 site and at the
   M1S1 site; show columns at radius 2 mm; show the profile section; open
   one column.
3. Give the time of each request and the server's peak memory. A request
   samples one or a few superblocks. If the peak goes above 8 GB, ask the
   coordinator for a slot.

## Work breakdown

| Step | Work | Who | Can run with |
|---|---|---|---|
| 1 | Part B: move `FilterJobs` and the TCK writer, with their tests | Sonnet sub-agent | 2 |
| 2 | Part A: `/columns` and `/columns/profiles` | Phase session | 1 |
| 3 | Part A: the tests | Sonnet sub-agent | after 2 |
| 4 | Part C: the page | Phase session | after 2 |
| 5 | The hemisphere check | Phase session | after 4 |
| 6 | Part D: the removal | Sonnet sub-agent, the phase session reviews | after the coordinator says go |

## PRs

1. `refactor/viewer-jobs-tck`: Part B. Branch from `origin/main`.
2. `feature/viewer-columns`: Parts A and C and the check. Stacked on PR 1.
3. `refactor/viewer-drop-site-mode`: Part D. Stacked on PR 2, after the
   coordinator says go.

Tell the coordinator when each PR opens.

## Decisions for review (made by the coordinator)

- The removal of site mode is the last PR of the phase, apart from the
  new pane, so the user can merge the pane without the removal.
- Flattening is not a filter in the viewer in wave 2.
