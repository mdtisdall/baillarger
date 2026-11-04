# Phase 1.0.5: small pieces

Stream 1, phase 0.5 of `docs/version-2-plan.md` (§3d). Read the plan's §3c
step 2, *Small pieces* and *The rule for cut faces*. The rules in
`docs/plans/round-2.md` apply.

## Goal

`dseg` removes the pieces of the filled mask that are smaller than
`CrumbMM3` (1 mm³). Each voxel of such a piece takes the most common label
of its non-filled 26-neighbours, as a fix of its own kind. On the
hemisphere, b0 goes from 22 to 1.

## Done when

1. The exact tests below pass.
2. `docs/formats/wm-dseg.md` gives the new fix kind and its count, and
   `tests/test_bids.py` passes.
3. The hemisphere outputs are rerun into `scratch/v2/derivatives-1.0.5/`:
   `dseg`, the white surface and the overview. b0 is 1. The PR body gives
   the diagnostics in "The hemisphere run".
4. `nix develop --command scripts/check` passes.

## Facts from the current code

### `src/baillarger/white_dseg.py`

- `wm_topo` (172–269): edits applied (189); the filled mask
  `m = np.isin(a0, filled)` (193), cropped to its bounding box plus 2 voxels
  (194–201); `before = kuhn_betti(x)` (204); cavities (206–207);
  `remove_handles(x)` (209); `cut, fill` (212); labels rewritten (214–217);
  the assertion `isin(labels, filled) == obj` (218); `after =
  kuhn_betti(obj)` (220); RuntimeError if b1 or b2 is not 0 (222–228). b0 is
  not checked.
- `FIXES` (66): `{1: handle_cut, 2: handle_fill, 3: cavity_fill, 4:
  manual_edit}`. The fixes image is uint8 (231–237).
- `WMTopo` (152–169). `moves` (239–259).
- `write_wm_topo` (317–338) writes the counts: `CavityVoxels`,
  `HandleFillVoxels`, `HandleCutVoxels`, `CutLabels`, `ManualVoxels`,
  `Moves`. `DSEG_SPEC` (69–85). `FIXES_SPEC` (87).
- The fixes TSV comes from `FIXES` (313). The fixes sidecar description
  (344–345) has the four kinds in a fixed string.
- The command: `python -m baillarger.white_dseg ASEG [--edits EDITS] --out
  DERIVATIVES` (`main`, 363–375).

### Pieces

- `handles.kuhn_structure()` (`handles.py` 130) and `handles.kuhn_betti`
  (207–217). b0 comes from `ndi.label(a, kuhn_structure())`.
- No helper gives a map of piece sizes. `excisions.drop_fragments`
  (`excisions.py` 165–176) and `excisions.cut_off` (179–198) use
  `ndi.label` and `np.bincount` with a mm³ test: `size × prod(voxel_mm) <
  max_mm3`.
- `white_surf.mesh_pieces(v, f, voxel_mm)` (`white_surf.py` 53) gives each
  mesh piece's genus and boundary loops.

### `CrumbMM3`

- `excisions.py` 58–60: `FRAGMENT_MM3, CRUMB_MM3 = 0.1, 1.0`.
- `excisions.py` imports `LUT_NAMES` from `white_dseg` (44). So
  `white_dseg` cannot import `CRUMB_MM3` from `excisions`. Move the
  constant to `labels.py`, and import it in both modules. `wm-excisions/1`
  keeps its `CrumbMM3` key.

### Tests

- `tests/test_white_dseg.py`: `SHAPE = (40, 36, 30)`, `VOX = (0.2, 0.16,
  0.2)`, `synthetic_aseg()` (22), `excised_block()` (145). The current
  phantoms have two filled pieces well above 1 mm³ (b0 = 2). The rule must
  keep them.

### The hemisphere

- Input: `scratch/v2/derivatives/sub-INDD121850/ses-7THemix20250516/anat/
  sub-INDD121850_ses-7THemix20250516_desc-excised_dseg.nii.gz`.
- Betti now: (22, 190, 95) → (22, 0, 0). Fixes: cavity 1,448, fill 9,798,
  cut 4,461 voxels.
- The commands and logs of the last run: `scratch/v2/logs/e/` (`step*.log`,
  `commit.txt`). Follow the same steps.

## Shared files with other phases

- Phase 1.3 (session C) changes `white_dseg.py` and `wm-dseg.md` after you.
  Its fix-table PR stacks on your PR. Open your PR early, then tell the
  coordinator.
- Phase 4.0.5 (session F) draws kind 5 in its colour table. The name is
  `small_piece`.
- Open PR #83 also changes `wm_topo`. Do not merge it or adapt to it.

## Design

1. After the handle and cavity correction, label the pieces of the filled
   mask with Kuhn adjacency. A piece is small if its voxel count ×
   prod(voxel_mm) < `CRUMB_MM3`.
2. Each voxel of a small piece takes the most common label of its
   non-filled 26-neighbours, in the labels after the correction. **Tie
   rule:** the smallest label value wins. This is a coordinator decision
   for review. Write it in the spec.
3. If a voxel of a small piece has no non-filled 26-neighbour, raise an
   error. It cannot occur for a piece below 1 mm³ at this voxel size, but
   the error makes the rule exact. Give the reason in the error.
4. Record the fix as kind 5, `small_piece`, in `FIXES`, the fixes image and
   the TSV. Add the count key `SmallPieceVoxels` and the piece count
   `SmallPieces` to `DSEG_SPEC`, and both rows to `wm-dseg.md`. Make the
   fixes sidecar description come from `FIXES`, not from a fixed string.
5. Add a `Moves` entry for each small piece: kind, voxel count, centre.
6. Update `obj` and the assertion at line 218. Run the Betti check after
   the rule.
7. Do not join a speck to the main piece across its diagonal (plan §3c step
   2 gives the reason).

## Exact tests

- A voxel that touches the main piece only across an edge diagonal: it is
  removed, and it takes the label of its neighbours.
- An isolated two-voxel speck surrounded by cortex: both voxels become
  cortex.
- A piece just above 1 mm³ is kept. A piece just below is removed. Use the
  anisotropic `VOX` so the mm³ test is exact.
- The tie rule, on a speck with equal counts of two labels.
- The two large pieces of the current phantoms stay. All earlier tests in
  `tests/test_white_dseg.py` pass unchanged, except the counts that the
  new keys add.
- The fixes image has code 5 exactly at the removed voxels.

## The hemisphere run

`dseg` peaks at about 22 GB and the white surface at 20–24 GB. Each is a
heavy job. Ask the coordinator for a slot for each step. Write only to
`scratch/v2/derivatives-1.0.5/`. Do not change `scratch/v2/derivatives/`.

1. `dseg` on the excised aseg. Then the white surface. Then the overview.
2. Diagnostics for the PR body:
   - Betti before and after, and the number of small pieces and voxels.
   - For each small piece: its size, its centre, its distance to the
     excision map (`desc-excisions_mask`), and the label it took. Compare
     with the plan's 21 specks (19 diagonal-only, 2 at 0.45 and 1.3 mm).
   - The mesh pieces of the new surface (`mesh_pieces`): one piece, genus 0.
   - The vertex and face counts against the old surface.
   - The time and the peak of each step.

## Work breakdown

| Step | Work | Who | Can run with |
|---|---|---|---|
| 1 | Move `CRUMB_MM3` to `labels.py` | Haiku sub-agent | 2 |
| 2 | The phantoms and the exact tests | Sonnet sub-agent | 1, 3 |
| 3 | The rule in `wm_topo`, the fix kind, the spec | Phase session | 2 |
| 4 | The hemisphere run | Phase session | after 3 |

## PRs

1. `feature/dseg-small-pieces`: everything. Branch from `origin/main`.
   Open it as soon as the code and the tests pass. Add the hemisphere
   results to the PR body afterwards.

Tell the coordinator when the PR opens. Phase 1.3 stacks on it.

## Decisions for review (made by the coordinator)

- The fix kind 5 is `small_piece`.
- The tie rule: the smallest label value wins.
- The new outputs go to `scratch/v2/derivatives-1.0.5/`. The user decides
  when they replace `scratch/v2/derivatives/`.
