# Phase 4.0.5: overlays on the brick pane

Stream 4, phase 0.5 of `docs/version-2-plan.md` (§6d). Read the plan's §6
fully. The rules in `docs/plans/round-2.md` apply. The same session does
phase 4.2 after this phase (`phase-4.2-review-pane.md`).

## Goal

The brick pane shows the labels and the outlines of the white surface on
the echo, at every zoom level, for the whole hemisphere. The review panes
(4.2, 4.5) draw on these overlays. NiiVue is not used for them, because
NiiVue loads one box into the browser and does not scale to a 50 µm whole
brain.

1. A label-brick pyramid: the most common label of each 2×2×2 block.
2. More than one named volume on the server.
3. A label overlay in the brick shader: nearest sampling, a colour table,
   and an opacity.
4. Outlines of the true white surface on each tile.

## Done when

1. The exact tests in each part pass.
2. The brick pane on the hemisphere shows echo 2 with the labels, the fixes
   image and the surface outlines. The PR body has screenshots at two zoom
   levels and at a topology fix. Use the built-in browser pane.
3. No response scales with the whole mesh or volume (plan §6b).
4. `nix develop --command scripts/check` passes on each PR.

## Facts from the current code

### `src/baillarger/viewer/bricks.py` (241 lines)

- `halve(a, dtype)` (95–109): the mean of each 2×2×2 block, one pair of
  x-slices at a time. At an odd far edge it averages the voxels that are
  present.
- `level_shapes` (75–83), `level_affine` (65–72).
- `BrickSource` protocol (53–62): `brick`, `dtype`, `levels`, `read(level,
  b)`.
- `ArrayPyramid` (112–163) calls `halve` at 122–124. It has no hook for
  another reducer. `from_nifti(path, lo, shape, brick)`.
- `wire_dtype` (86–92): `<i2` if the input is integer and fits int16,
  else `<f4`. A label above 32767 becomes float32. That is wrong for
  labels.
- `encode` (173–178): bricks little-endian, x fastest, no header.
- `LazyVolume(make, name)` (217–241) builds the pyramid on first use.
- `tests/test_viewer_bricks.py`: `test_coarse_levels_are_block_means` (64)
  is the model for an exact test of the mode.

### `src/baillarger/viewer/server.py` (556 lines)

- One volume only: `Handler.volume` (145), built in `main()` (535–551).
  `volume_route` (240): `/volume`, `/volume/meta`, `/volume/bricks`.
- `--white` mode serves `/meta`, `/mesh`, `/vertex`, `/snap`, `/white.gii`
  and the `/white/*` box routes (`BOX_ROUTES`, 127–133).
- The server binds 127.0.0.1 (474).

### `src/baillarger/viewer/static/index.html` (1123 lines)

- The whole front end is one file, with one inline module script. NiiVue
  0.69 comes from a pinned CDN URL (159–160).
- The brick pane ("BV"): `initBricks` (668–706), `bvLoad` (771), `bvRender`
  (790), `bvOverlay` (814–830), `bvEvents` (846).
- The shader `BVFS` (650–665): a page table (`usampler3D`, R32UI) and an
  atlas (R16I or R32F). It starts at the zoom's level and falls back to
  coarser loaded levels. Greyscale only.
- `#bvo` is a transparent 2D canvas over the WebGL canvas. `bvOverlay`
  draws the crosshair and the side letters on it.
- The NiiVue label overlays (`LAB`, `loadLabelOverlay`, 1010–1044) are
  hidden in brick mode.
- `server.py` serves `/static/<file>` (321–326), for `.html`, `.js` and
  `.css`.

### `src/baillarger/viewer/white_view.py` (199 lines)

- `WhiteView.surface_faces(c)` gives the true mesh's faces in a box, from
  the block index. `self.W.v`, `self.W.f` hold the true surface.
- The fixes overlay: 1 WM mask, then fixes + 1 (2 cut, 3 fill, 4 cavity).

### `src/baillarger/viewer/sections.py` (215 lines)

- `section_curve` (105–192) cuts the mesh with one plane and follows one
  curve from a root. It is not a full cut of a plane. Its crossing code
  (132–148) can be reused.

### The data

- `scratch/v2/derivatives/sub-INDD121850/ses-7THemix20250516/anat/`:
  `_desc-wmtopo_dseg.nii.gz`, `_desc-wmtopofixes_dseg.nii.gz`, the white
  surface (`_hemi-R_white.surf.gii` and `.dat`) and the overview.
- The echoes: `scratch/v2/echoes/`, three uncompressed `.nii` files.
- The echo pyramid of the hemisphere builds in 6.8 s at 1.4 GB peak (#84).

## Shared files with other phases

`server.py` and `index.html` are shared with phase 4.3 (another session)
and with your own phase 4.2. To keep merges small:

1. Put new Python code in new modules: `viewer/volumes.py` (named volumes)
   and `viewer/outlines.py`.
2. Put new page code in new static files: `static/overlays.js`. Load it from
   `index.html` with one `<script type="module">` line.
3. In `server.py` and `index.html`, add only the hooks: imports, one
   dispatch entry for each route prefix, and the pane's DOM block.
4. Phase 4.3 follows the same rule. If you must change code in `server.py`
   or `index.html` beyond the hooks, ask the coordinator first.

## Design

### Part A: the label pyramid

1. Give `ArrayPyramid` a reducer parameter: `mean` (now) or `mode`. Keep
   the default, so the echo pyramid does not change, byte for byte.
2. `mode`: the most common label of each 2×2×2 block. At an odd edge, count
   only the voxels present. **Tie rule:** the smallest label value wins.
   This rule is a coordinator decision for review. It is deterministic. A
   tie of outside (0) against tissue shows outside at coarse levels only.
3. Labels keep an integer type. Add an `<i4` wire type (R32I atlas) for
   labels above int16. Never send labels as float.
4. Analysis for the PR body: time and bytes of the mode against the mean,
   on the hemisphere, and its cost at a 50 µm whole brain (it is local per
   block, so it can run per chunk of the NIfTI-Zarr store later).

Exact tests: a block with a clear majority; a tie (smallest wins); an odd
edge; the echo pyramid unchanged byte for byte; labels above 32767 stay
exact.

### Part B: named volumes

1. A volume registry: `Handler.volumes`, a dict from name to `LazyVolume`.
2. Routes: `/volume/<name>/meta` and `/volume/<name>/bricks`. Keep
   `/volume`, `/volume/meta` and `/volume/bricks` as the echo, so the page
   and its tests keep working.
3. `/volume/list`: the names, their kinds (image or labels) and dtypes.
4. In `--white` mode, the server finds the `_desc-wmtopo_dseg` and the
   fixes image from the white surface's sidecar `Sources`. Add arguments
   to give other paths.
5. Each pyramid is built on first use, one at a time, under a lock.

Exact tests: two volumes with different shapes; `/volume/list`; an unknown
name is 404; the old routes give the same bytes as before.

### Part C: the label overlay in the shader

1. A second atlas and page table for the label volume, with its own
   `w2v[]` and shapes. The label volume can have a different grid.
2. Nearest sampling for labels. Never interpolate a label.
3. A colour table as a small texture: label value to RGBA. Take the colours
   from FreeSurferColorLUT for the label set (`labels.py`, `label-set/1`).
   The fixes image has its own table (cut, fill, cavity, manual; and kind
   5, `small_piece`, from phase 1.0.5).
4. Controls: which label volume (labels, fixes, none), the opacity, and an
   outline-only mode (draw a label's edge, not its fill).
5. The overlay uses the same level fallback as the echo.
6. Show the label name under the cursor in the tile's text line.

Tests: the colour table for each label in the set (a Python test of the
table data); the page reads the label under the crosshair correctly (a
browser check at a known voxel).

### Part D: surface outlines

1. `outlines.py`: all segments where the true mesh crosses an
   axis-aligned plane, inside a box. Pick the faces with the block index
   (`white-surf/1`). Do not cut the whole mesh.
2. Route: `/outline?axis=&mm=&box=`. It returns float32 segment ends in
   scanner mm. The box is the tile's view.
3. **A limit on size.** If the box holds more than a fixed number of
   faces, cut the overview mesh instead and say so in the response. The
   overview is for display only (plan §3c, step 5), and an outline is
   display. Give the limit and the reason in the PR body.
4. The page draws the segments on `#bvo`, for each tile, after each pan,
   zoom or crosshair move. Cache by axis, plane and box.

Exact tests: the cut of a sphere phantom is a polygon at the right radius;
a plane that misses the mesh gives no segments; only faces in the box are
read (count the faces read); the overview fallback above the limit.

## The hemisphere check

1. Start the viewer in `--white` mode on `scratch/v2/derivatives/` and
   echo 2, with the brick pane (`?vol=bricks`). Add a `.claude/launch.json`
   entry for your worktree. The label pyramid adds about 0.7–1.4 GB. The
   viewer's run is not a heavy job if its peak stays under 8 GB.
2. Check in the built-in browser pane: the labels, the fixes image and the
   outlines at two zoom levels; at one handle fix; the label name under the
   cursor; the console has no errors.
3. Screenshots in the PR body, each well under 1 MB.

## Work breakdown

| Step | Work | Who | Can run with |
|---|---|---|---|
| 1 | Part A: the mode reducer and the `<i4` wire type | Sonnet sub-agent | 2, 4 |
| 2 | Part B: named volumes | Sonnet sub-agent | 1, 4 |
| 3 | Part C: the shader and the controls | Phase session | after 1 and 2 |
| 4 | Part D: `outlines.py` and its route | Sonnet sub-agent | 1, 2 |
| 5 | Part D: drawing on `#bvo` | Phase session | after 3 and 4 |
| 6 | The hemisphere check | Phase session | after 5 |

## PRs

1. `feature/brick-overlays`: Parts A–D and the check. Branch from
   `origin/main`. If the PR grows above about 1500 changed lines, split
   Part D into `feature/brick-outlines`, stacked on it.

Tell the coordinator when the PR opens. Phase 4.2 merges this branch.

## Decisions for review (made by the coordinator)

- The mode's tie rule: the smallest label value wins.
- The outline falls back to the overview above a face limit.
- New code goes in new modules and static files, to keep `server.py` and
  `index.html` merges small.
