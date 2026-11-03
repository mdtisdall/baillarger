# Phase 1.0: the pipeline on the purple-mri aseg

Stream 1, phase 0 of `docs/version-2-plan.md` (§3d). This phase needs no
other phase. Every other stream waits on its outputs. It is in round 1
(`docs/plans/round-1.md`).

## Goal

Publish the three outputs that streams 2–4 need, for the one hemisphere that
we have:

- `wm-dseg/1`: the corrected labels and the fixes image.
- `white-surf/1`: the white surface, with a new file for the normal length
  |n̄|.
- `white-overview/1`: the decimated display mesh.

The input is the purple-mri aseg, with its excisions relabelled to outside.
The code uses the current labels-only path: handle removal ordered by
distance, and antialiasing by smoothness only.

## Done when

1. `white.py` is split into three modules (see "Design, part A"). Each has
   its own command line.
2. A new module relabels the aseg's excisions to outside and writes the
   excision map (see "Design, part B").
3. `white-surf/1` has the normal-length file, in the code, the spec and the
   tests (see "Design, part C").
4. The phantom tests for cut faces pass (see "Tests").
5. The hemisphere outputs are in `scratch/v2/derivatives/` and every output
   passes its reader's `check`.
6. The PR bodies give the diagnostics in "The hemisphere run".
7. `nix develop --command scripts/check` passes on each PR.

## Facts from the current code

All paths are relative to the repository root.

### `src/baillarger/white.py` (1058 lines)

| Lines | Part | Contents |
|---|---|---|
| 91–417 | dseg | `FILLED_LABELS`, `WM_LABEL`, `LUT_NAMES`, `DSEG_SPEC`, `FIXES_SPEC`, `hemisphere`, `cavities`, `relabel_cuts`, `WMTopo`, `wm_topo`, `write_wm_topo`, `filled_labels` |
| 420–465 | box only, except `mesh_pieces` | `white_mesh`, `white_mesh_normals` (box only); `mesh_pieces` (the surface and the overview use it) |
| 468–783 | surface | `SURF_SPEC`, `surface_paths`, `mm_to_voxel`, `block_bounds`, `write_white_surface`, `SurfaceFiles`, `WhiteSurface` |
| 785–1003 | overview | `OVERVIEW_SPEC`, `overview_paths`, `write_white_overview`, `surface_points`, `WhiteOverview` |
| 1008– | command line | `dseg`, `mesh` (surface, then overview), `overview` |

- `white.py` does not import `frame`, `topology` (the module),
  `wellcomposed`, `growth` or `joint`.
- `handles.py`, `wm_edits.py` and `decimate.py` do not import any module that
  the plan drops. "Port as is" for these three means: keep them as they are.
- `antialias.py` has only a band path (`thin_plate`, `BAND = 5.0`). Make no
  change to it in this phase. The blocked solve is scaling item 4.

### Importers of `white`

- `src/baillarger/column_traces.py:126` (`filled_labels`,
  `white_mesh_normals`).
- `src/baillarger/viewer/overview.py:24` (`WhiteOverview`, `WhiteSurface`,
  `overview_paths`).
- Tests: `test_bids.py` (spec parametrization at lines 9 and 95–99),
  `test_white.py`, `test_white_dseg.py`, `test_white_surf.py`,
  `test_white_overview.py`, `test_wm_edits.py`, `test_viewer_overview.py`,
  `test_pial_labels.py`, `test_column_traces.py`.
- Prototypes: `antialias.py:42`, `antialias_profile.py:175`.
  `wm_handles.py` and `wm_handles_boxes.py` are already broken. Leave them.

### The excision detector

- `frame.missing_tissue(rim, flash, voxel_mm, frac=0.25, sigma=2.0,
  min_vox=250, margin_mm=0.3)` at `frame.py:136–165`. It uses only numpy and
  `scipy.ndimage`.
- It reads rim labels from `data.rim_labels` (1 outside, 2 WM, 3 cortex).
  These rim labels use right-hemisphere aseg values only (41, 42).
- The callers give it echo 2.
- Tests: `test_frame.py:120` and `:136` on `excised_slab()`.
- About 20 prototypes and `column_traces.py:124` import it from `frame`.

### Normals

- `white.py:619–622`: `S.smooth_normals(v, f)` without torch, else
  `MeshOps(...).normals(v)`.
- `mesh_smoothing.smooth_normals` (lines 139–147) and
  `mesh_torch.MeshOps.smooth_normals` (lines 212–218) do 10 steps of
  `n = 0.5 n + 0.5 (A n)/deg`, then `n /= |n|`. |n̄| is the norm before the
  last division.
- `test_mesh_torch.py:62` already tests that the two give the same normals.

### Specs

- `docs/formats/`: `wm-dseg.md`, `white-surf.md`, `white-overview.md`,
  `wm-edits.md`.
- `bids.SidecarSpec` enforces each sidecar. `test_bids.py:102`
  (`test_spec_documents_list_the_code_keys`) checks that each doc's
  "Sidecar keys" table lists the same keys as the code.

### The data and the last hemisphere run

- `data/` (a link to the main checkout's `data/`):
  `sub-INDD121850_ses-7THemix20250516`, three echoes and `_aseg.nii.gz`.
  Grid 700 × 384 × 1148 at 0.16028 × 0.16 × 0.16028 mm, right hemisphere.
  `data.py:27–31` holds the names.
- The last run (`docs/scaling.md:136–142, 182`):

| Step | Result | Peak memory |
|---|---|---|
| dseg | Betti (1, 111, 182) → (1, 0, 0), 3.4 min | 21.6 GB |
| mesh | 12.7 M vertices, 25.3 M faces, 1,651 iterations on MPS | 20–24 GB |
| overview | 4 workers | 10.9 GB |

## Old code

The user's decision: a module that the plan drops stays until the phase that
replaces its callers deletes it. In this phase:

- Do not delete `frame`, `topology`, `wellcomposed`, `growth`, `joint`,
  `columns`, `column_traces`, `pial_labels` or `output`.
- `white.py` keeps only the box-only part (`white_mesh`,
  `white_mesh_normals`) and a re-export of `filled_labels`, so that
  `column_traces` continues to work. Write in its docstring that it is
  legacy and that stream 2's phase 0 deletes it.
- `frame.missing_tissue` becomes a re-export of the new function, with the
  old signature. The old callers continue to work.

## Design

### Part A: split `white.py`

Make three modules. Move the code. Do not change what it does.

| New module | Contents | Command line |
|---|---|---|
| `src/baillarger/white_dseg.py` | the dseg part, with its specs and `filled_labels` | `python -m baillarger.white_dseg ASEG [--edits EDITS] --out DIR` |
| `src/baillarger/white_surf.py` | the surface part, `mesh_pieces`, `WhiteSurface` | `python -m baillarger.white_surf DSEG --out DIR [--block-cells N] [--device D]` |
| `src/baillarger/white_overview.py` | the overview part, `WhiteOverview` | `python -m baillarger.white_overview SURF [--eps E] [--workers N]` |

1. The current `mesh` command runs the surface and then the overview. After
   the split, they are two commands. Write the order in each module's
   docstring.
2. Change the importers in "Importers of `white`" to import from the new
   modules. The prototypes `antialias.py` and `antialias_profile.py` also
   change.
3. Change the spec docs: each one names its writing command and module.
4. This part is a refactor. Use the `parallel-agents` rules for refactors:
   compare the outputs on the phantoms before and after, byte for byte, with
   a baseline worktree on `origin/main`.

### Part B: excisions to outside

Make `src/baillarger/excisions.py`. It is a bootstrap tool. It runs only on
an aseg that imputes tissue, such as purple-mri's.

1. Move the body of `frame.missing_tissue` into
   `excisions.detect(aseg, flash, voxel_mm, hemi, ...)`. Keep the same
   parameters and defaults. Take the WM and cortex labels from the aseg and
   the hemisphere ("R": 41 and 42; "L": 2 and 3), not from rim labels.
2. Use float32, not float64.
3. Write `excisions.relabel(aseg, mask)`. It sets each voxel in the mask to
   0 (outside). It returns a new array. It does not change its input.
4. Add a command line:
   `python -m baillarger.excisions ASEG FLASH --hemi R --out DIR`. FLASH is
   echo 2. It writes two files with BIDS-derivative names and sidecars:
   - `_desc-excised_dseg.nii.gz`: the aseg with the excisions set to 0. This
     is the input label for `white_dseg`.
   - `_desc-excisions_mask.nii.gz`: the excision map. Stream 3 uses it as a
     test set (plan §5c, Open).
5. Write a spec for these two files: `docs/formats/wm-excisions.md`, version
   `wm-excisions/1`. Record the detector's parameters, the input SHAs and the
   number of voxels relabelled. Add a `SidecarSpec` and add the doc to the
   `test_bids.py` parametrization. Send the spec to the coordinator before
   you write the code.
6. `wm-edits/1` pins edits to the SHA of the input aseg. After this change,
   the input of `white_dseg` is the excised aseg. If edits exist for the
   hemisphere, tell the coordinator. Do not change `wm-edits/1` in this
   phase. Phase 1.3 revises it.

### Part C: the normal length |n̄|

1. Add a function to `mesh_smoothing` and a method to `mesh_torch.MeshOps`
   that return the unit normals and |n̄|. |n̄| is the norm of the smoothed
   normal before the last division. Keep the old functions. They call the
   new ones.
2. `white_surf` writes `_desc-normlen_white.func.gii` (float32, one value per
   vertex, in [0, 1]).
3. Add the file to `surface_paths`, `SURF_SPEC` (a new key, for example
   `NormLenFile`), `docs/formats/white-surf.md` and `WhiteSurface.load`.
4. Tests:
   - On a plane, |n̄| = 1 at every vertex away from the edge.
   - On a thin blade (two faces one or two voxels apart), |n̄| falls towards
     0 at the crest. Give the expected value from the geometry.
   - numpy and torch give the same |n̄| to float32 precision.

## Tests for cut faces

The plan asks for two phantom tests (§3d, phase 0). Put them in
`tests/test_white_dseg.py`.

1. **A tunnel.** Make a WM phantom with an excision (outside) that goes
   through it and leaves a tunnel. Run `wm_topo`. The result has Betti
   (1, 0, 0), and the fixes image shows a fill (a plug) in the tunnel.
2. **A piece cut off.** Make a WM phantom where an excision cuts a piece off.
   Run `wm_topo`. b0 is 2. Each piece is closed and has genus 0.

Move the two `missing_tissue` tests from `test_frame.py` to a new
`tests/test_excisions.py`. Rewrite `:136` without `build_frame`. Keep
`test_frame.py` working through the re-export.

## The hemisphere run

Do this after parts A, B and C. Follow the rules for heavy runs in
`round-1.md`: ask the coordinator before each heavy step.

| Step | Command | Cap |
|---|---|---|
| 1 | `python -m baillarger.excisions` on the aseg and echo 2 | measure it; start at 16 GB |
| 2 | `python -m baillarger.white_dseg` on the excised aseg | 33 GB |
| 3 | `python -m baillarger.white_surf` on the dseg | 36 GB |
| 4 | `python -m baillarger.white_overview` on the surface | 17 GB |

Write all outputs to `scratch/v2/derivatives/`. Then run each reader's
`check` on its output.

### Diagnostics for the PR body

The plan prefers views to pass/fail numbers. Give these:

1. Excisions: the number of voxels relabelled, the number of components,
   and the percentage of the aseg's cortex label. Give slice views of the
   excision map over echo 2 at three or four excisions. Put the images in
   `scratch/v2/figures/` and attach them to the PR.
2. dseg: Betti before and after. The count of each kind of fix. Compare with
   the last run (Betti (1, 111, 182) before). Explain each large difference.
3. The surface: the number of pieces, vertices and faces. Self-intersecting
   faces (must be 0). The antialias iterations.
4. |n̄|: a histogram. The fraction of vertices below 0.9 and below 0.6.
   A view of where |n̄| is lowest.
5. Each step's peak memory and time.

If a result looks wrong, diagnose it and write a test with an exact answer
before you fix it. Do not change a constant to hide a symptom.

## Work breakdown

| Step | Work | Who | Can run with |
|---|---|---|---|
| 1 | Part A: the split, the importers, the spec docs | Sonnet sub-agent; the phase session checks the byte-for-byte comparison | 2, 3 |
| 2 | Part B: `excisions.py`, its tests, the re-export in `frame` | Sonnet sub-agent (after the coordinator answers about the spec) | 1, 3 |
| 3 | Part C, step 1: the normal-length functions in `mesh_smoothing` and `mesh_torch`, and their tests | Sonnet sub-agent | 1, 2 |
| 4 | Part C, steps 2–3: the new file in `white_surf`, the spec and the reader | Phase session | after 1 and 3 |
| 5 | The cut-face phantom tests | Phase session | after 1 |
| 6 | The hemisphere run and the diagnostics | Phase session | after 2, 4 and 5 |

Steps 1, 2 and 3 change different files. Step 2 must not change `white.py`
or the new modules. Step 3 must not change `white.py` or the new modules.

## PRs

Stack them in this order:

1. `refactor/white-split`: part A only.
2. `feature/excisions`: part B and the cut-face tests. Stacked on PR 1.
3. `feature/white-normlen`: part C and the hemisphere run report. Stacked
   on PR 2.

Phase 1.5 stacks one PR on PR 1 (it moves the label constants). Tell the
coordinator when PR 1 is open.

## Coordination

- Phase 1.5 writes the label-set spec. It does not change the white modules
  except in its stacked PR. If its decisions change `FILLED_LABELS`, the
  hemisphere outputs change. The coordinator decides whether to run them
  again in this phase.
- Send the `wm-excisions/1` spec to the coordinator before you write the
  code.
