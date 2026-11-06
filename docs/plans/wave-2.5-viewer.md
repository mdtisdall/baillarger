# Wave 2.5, step 3: viewer tools for the review

Session V. Stream 4. Read the plan's §1 (working rules), §4b (*What a
correct column looks like*), §6 and §9 (wave 2.5), and
`docs/plans/wave-2.5-signals.md` (the `column-signals/1` files and the
provisional `edges.npz`). The rules
in `docs/plans/round-2.5.md` and `round-2.md` apply.

## Goal

The user can go from a signal to the column, see the column's signals next
to its raw samples, and write a note that opens the same view again after a
fix.

1. The signals as an overlay on the overview, and a ranked list of sites
   for each signal. A click on a site opens its column.
2. The selected column's signals in the columns pane, and its data normal
   drawn next to its surface normal.
3. A notes log. Each note records the site, what the user saw, the cause
   that the user blames, and the view, so that the view opens again
   exactly.
4. Fixes for the known viewer bugs.
5. The user's feature requests (Part D).

No new section view. The profile section of the columns pane (#128) is the
view for the review.

## Done when

1. The viewer runs on the new input set (`scratch/v2/derivatives-main/`)
   with the hemisphere signals. The PR bodies have screenshots: an overlay,
   a ranked list, a column opened from the list with its signals and its two
   normals, and a note opened again after a page reload, with the same
   section.
2. The viewer starts and serves every route of 4.0, 4.0.5, 4.2 and 4.3
   without a signals file and without a notes log.
3. No response scales with the whole mesh except the overlay, which is one
   value for each overview vertex (about 170 k).
4. The exact tests below pass.
5. `nix develop --command scripts/check` passes on each PR.

## Facts from the current code

All references are to `main` at 1f67376.

### The overview surface

- The overview is grey. `colour3d()` fills `bcol` with 110/255
  (`static/index.html:189-192`). The per-vertex `col` buffer exists
  (`GL.bcol`, `index.html:175-178`). Nothing else writes it.
- `/mesh?level=k&faces=1` gives `Surface.mesh_payload`
  (`viewer/surface.py:151-179`).
- Overview vertex k is true vertex `ov.vertex_ids[k]` (`white_view.py:68,
  103`). The viewer does not use the carry files.
- No route serves per-vertex values. No colour map for continuous values
  exists.
- `pick3d` finds the nearest overview vertex on screen (`index.html:221-232`).
  A click calls `selectVertex(v, screenFrame())`, then `/vertex?v=`.

### A table to reuse

- The stream 1 review queue table (`static/review.js:86-97`): a row click
  calls `open(id)` and centres the brick tiles. It is keyed by voxel site,
  not by vertex. The review pane takes the place of the surface pane
  (`review.js:64`).
- No ranked list opens a column or a vertex.

### The columns pane

- Server `viewer/columns.py`: `GET /columns/meta`, `GET /columns?v=&radius=
  [&method=&every=&e_in=&e_out=&step=]`, `GET /columns/profiles`, `GET
  /columns/<key>.tck` (`route`, lines 539-576). `v` is a true vertex.
- The arrays sent (`RAY_ARRAYS`, lines 192-210) include `edge_key`,
  `vertex`, `root` and `normal`. They do not include the focus, a data
  normal or a coherence.
- Page `static/columns.js`: state `C` (line 23). `load()` uses
  `V.selection.true_vertex` (lines 88-114).
- The section (`section()`, lines 203-222): u is the 3D view's screen right
  at the time of the selection (`V.frame.right`), made normal to the picked
  column's normal. **u is computed, not stored**, and it is not in the
  shared state. The half width (`chalf`), the echo (`cecho`) and the show
  choice (`cshow`) are not in the shared state either.
- The details panel `#cdet` (`details()`, lines 261-272): vertex, edge key,
  owned, superblock, distance, |n̄|, derived class, and the intervals.
- No normal is drawn. `normal` only orients the section plane (line 206).

### Logs that the viewer writes

- No notes log exists. The only POST routes apart from `/state` are the
  review routes (`server.py:294-304`).
- The pattern to follow: `wm_edits.EditLog` (`wm_edits.py:296-420`): an
  exclusive `fcntl.flock` on the `.jsonl` for each append, read back, assign
  the id, check the record, append, and write over an unfinished last line.
  A sidecar with the sources.
- The path comes from a CLI flag (`--edits`, `review.py:644-681`).

### The input set

- The viewer does not show which input set it serves. `/meta`
  (`white_view.py:70-96`) has no sources. `Columns.sources`
  (`columns.py:301-309`) is a hash used only in the cache key.
- The white-surface sidecar holds `Sources`, `SourcesSHA256` and
  `DataSHA256`.
- Edge keys: `_desc-edgekeys_white.func.gii`, int64 as an (n, 2) int32
  pair. They identify a vertex across remeshes.

### The server

- `Handler` class attributes: `shared`, `white`, `volume`, `volumes`,
  `outlines`, `columns`, `wmreview`, `lock` (`server.py:116-304`). Each pane
  is a module with a `route(h, ...)` function. `do_GET` calls them in a
  fixed order.
- `serve(site, port, *, white, volume, volumes, outlines, columns,
  wmreview)` (`server.py:307-333`). `main()` flags: `--white`, `--cache`,
  `--echo`, `--dseg`, `--fixes`, `--echo-dir`, `--sites`, `--edits`,
  `--port`.
- Page hooks: `window.VIEWER.hooks {selection, state, draw3d, resize}`
  (`index.html:606-614`). Brick-pane plug-ins `window.BVP`. The panes
  through `?panes=` (`index.html:82-87, 622-626`).

### Tests

- All viewer tests are Python (`tests/test_viewer_*.py`). There are no JS
  tests. `test_viewer_columns.py::test_the_white_mode_serves_every_route`
  (lines 313-388) starts `main()` and checks every route.

### The known bugs

1. `/labels/list` gave 404 in `--white` mode (#126). After #134 no page
   code asks for it, and a test asserts the 404. Not checked in a browser.
2. `bvFit` throws when the page loads in a hidden pane: "bvRange … reading
   'shape'", from `initBricks` (`index.html:413, 421-431, 448-452`). The
   probable cause: with a zero-size canvas, the layout gives a negative
   scale, `log2` gives NaN and `M.levels[NaN]` is undefined. This is an
   inference, not a test.
3. A ctrl-wheel event moves the view centre far in `bvZoom`
   (`index.html:559-562, 586-591`). Seen with the browser tool's event only.
   A real trackpad is not tested.
4. `.claude/launch.json` still has entries for the site mode that #134
   removed. The file is not tracked by git: it is local to the main
   checkout.

## Design

### Part A: the bugs (PR 1)

1. Bug 2: do not lay out or fit a pane with a zero-size canvas. Fit when
   the pane gets a size (the `resize` hook). Reproduce it first in the
   browser pane with a hidden pane, then fix it.
2. Bug 3: a wheel event with ctrl zooms about the cursor, with the same
   step as a plain zoom, and does not move the centre by more than the
   zoom about the cursor gives. Reproduce it first. In the PR body, ask the
   user to check a real trackpad.
3. Bug 1: check in the browser pane that the page makes no request that
   gives 404 in `--white` mode. Give the network list in the PR body.
4. Bug 4: remove the site-mode entries from `.claude/launch.json` in the
   main checkout, and add one entry for the new input set. This is a local
   change, not part of the PR. Say in the PR body that it is done.

### Part B: the signals (PR 2)

1. A new module `viewer/signals.py`. It reads the `column-signals/1`
   files: `desc-signals_white.func.gii` (the named maps, with each map's
   metadata), `desc-datanormal_white.func.gii` and their sidecar. It
   memory-maps the `.dat` files. The flag `--signals PATH` gives the
   sidecar. Without the flag, the viewer uses the sidecar next to the white
   surface if it exists. Without one, the signals routes give 404 and the
   page hides the signal controls.
2. The flag `--edges PATH` gives the provisional `edges.npz`. It is
   optional. Without it, `/signals/at` gives no edges and faces, and the
   page hides them.
3. It checks that the sidecar's input set and the `white_sha256` of
   `edges.npz` are those of the white surface that `--white` gives. If not,
   it refuses to start with a clear message.
4. Routes:
   - `GET /signals/meta`: for each map in the file: the name, unit,
     `LookAt`, `From`, `Limits` (from the map's metadata), and the 1st, 50th and
     99th percentiles of its finite values.
   - `GET /signals/<name>/overview`: f32, one value for each overview
     vertex. The value is the worst one (the `LookAt`
     end) of the overview vertex and of the true vertices carried by the
     overview faces that touch it. A point sample would hide a bad true
     vertex inside an overview face.
   - `GET /signals/<name>/sites?n=200&spacing=1.0`: the ranked list. Take
     the worst finite value, keep it as a site, drop all vertices within
     `spacing` mm of it in 3D, and repeat until n sites. Each site: rank,
     true vertex, edge key, value. No threshold.
   - `GET /signals/at?v=v1,v2,...`: all maps and the data normal at the
     given true vertices, and, with `--edges`, for each vertex its edges (the other vertex, the two
     `edgeshift` values, `edgeweight`, `edgeclip`) and its faces
     (`facecirc`). At most the columns pane's `max_columns` vertices.
5. Cache the overview values and the site lists for each signal. They do
   not change while the server runs.
6. A new page module `static/signals.js`:
   - The overlay: a signal choice, a sequential colour map (the `LookAt` end
     is the bright end), a range from the 1st to the 99th percentile that
     the user can change, and a colour bar. It writes `GL.bcol` through a
     new `colour` hook in `index.html`. "None" gives the grey of today.
   - The provisional limits of the chosen signal are shown under the colour
     bar, always, as text.
   - The ranked list: a table (rank, value, vertex) for the chosen signal.
     A row click selects the true vertex through the shared state, turns the
     3D view to the site, and the columns pane loads it.
7. In the columns pane (`static/columns.js`):
   - The details panel shows the picked column's signals, with units, and
     its edges sorted by |root shift|.
   - A strip above the section: one cell for each column in u order,
     coloured by the chosen signal with the overlay's colour map, so that a
     step in the section can be read against the signal.
   - In the 3D view: at the picked root, the surface normal (the column's
     `normal`) and the data normal (`datanormal`) as two 1 mm segments in
     two colours, with a legend.

### Part C: the notes log (PR 3)

1. The format `review-notes/1`, in `docs/formats/review-notes.md`: a
   `.jsonl` log and a JSON sidecar, as `wm-edits/1` does. Two kinds of
   record, append only:
   - `note`: `Id`, `Time`, `InputSet` (the derivatives root and the SHA-256
     of the white-surface sidecar), `Vertex`, `EdgeKey`, `RootRAS`, `View`,
     `Saw` (the user's text), `Cause` (one of `surface`, `casting`,
     `working-version`, `unattributed`), `WorkingVersion` (text, when
     `Cause` is `working-version`, for example "1.2 orient"), `Signals`
     (the vertex signals at the time, by name).
   - `recheck`: `Id`, `Time`, `NoteId`, `InputSet`, `Vertex`, `Match` (see
     item 4), `Saw`, `Result` (one of `gone`, `still-there`, `changed`).
   - `View`: `URAS` (the section axis u, a unit vector in RAS),
     `HalfWidthMM`, `Echo`, `Show`, `RadiusMM`, `Every`, `EInMM`, `EOutMM`,
     `Method`, and the 3D view's rotation.
2. A new module `viewer/notes.py`. The flag `--notes PATH` gives the log.
   The log is made if it is absent. Each append holds an exclusive
   `fcntl.flock`, as `EditLog` does. Without the flag, the notes routes
   give 404 and the page hides the notes controls.
3. Routes: `GET /notes`, `POST /notes`, `POST /notes/<id>/recheck`, `GET
   /notes/<id>/open`.
4. Open: the server finds the note's column in the current input set.
   - The same input set: the stored vertex. `Match` is `same`.
   - Another input set: the vertex with the same edge key. `Match` is
     `edge-key`.
   - No such edge key: the root nearest to `RootRAS`, if it is within one
     voxel. `Match` is `nearest`, with the distance.
   - Else `Match` is `none`, and the page says so. It does not open a
     different column.
5. `/meta` gives the input set (root and sidecar SHA-256), and the page
   shows it in the header.
6. In `static/columns.js`: u, the half width, the echo and the show choice
   go into the shared state `columns`. When a note opens, the section uses
   the stored u until the user picks a new column. The section's text line
   says "u from note N".
7. A notes panel: the list of notes (id, cause, the first words of `Saw`,
   the last recheck result), "new note" with the fields of item 1, "open",
   and "recheck".

### Part D: the user's requests (PR 4)

The user adds requests here during wave 2.1's review. Each request gets an
exact test.

1. **Go to a vertex.** A box in the header takes a true vertex number and
   makes it the WM selection, as a click on that vertex does. The address
   parameter `?v=<true vertex>` does the same when the page opens. A new
   route gives the selection for a true vertex: its position in mm, its
   voxel, and the nearest overview vertex. A number out of range gives an
   error in the log, and the selection does not change.
2. **The profile section follows the surface.** Now the section takes the
   columns whose roots lie within the half width of a plane through the
   selected root, and sorts them by u. Where the WM is one voxel thick, the
   plane cuts the surface two times, close together, and the sort
   interleaves the two cuts. The profile then shows alternating bands (seen
   at true vertex 10909480). The new section is a line along the surface:
   - The curve is the cut of the same plane with the white mesh, only the
     piece that goes through the selected root. It runs in both directions
     from the root, to the column radius along the curve.
   - The section shows one column for each step along the curve: the
     column whose root is nearest to the curve at that step. The step is
     the mean spacing of the roots near the selection.
   - It sorts them by their arc length along the curve, from the root
     (negative on one side).
   - A strip of columns on each side of the curve is not interleaved into
     the section. On flat, regular cortex (true vertex 12218203), the
     current strip of ±0.25 mm (3 voxels) put columns
     from different lines across the strip next to each other. Columns close
     across the strip were alike; columns 0.16–0.5 mm apart differed 24
     times more in their mean level. The profile showed alternating bands.
   - The half-width menu goes. A "mean across" option shows, at each step,
     the mean of the columns within one voxel of the curve.
   - Where the surface turns around a thin edge, the curve goes around the
     turn, and the profile shows the turn at one place.
   The section does not label the sides of the WM, and it does not use the
   sign of the normals.

## Exact tests

- `/signals/<name>/overview` on a phantom surface and overview equals the
  worst-of value computed with numpy from the carry CSR.
- `/signals/<name>/sites`: the first site is the worst finite value. No two
  sites are within `spacing`. NaN vertices are never sites. A `LookAt: low`
  signal ranks from the lowest.
- `/signals/at` gives the file's values for those vertices, and the edges
  and faces that touch them.
- A signals sidecar or an `edges.npz` for another white surface: the
  server refuses to start.
- Without `edges.npz`, `/signals/at` gives the maps and no edges.
- Without a signals sidecar and a notes log, `main()` serves every route of
  4.0, 4.0.5, 4.2 and 4.3, and the new routes give 404 (extend
  `test_the_white_mode_serves_every_route`).
- The notes log: append and read back. Two threads that append at the same
  time give two records with different ids. An unfinished last line is
  written over. The sidecar passes its spec, and `tests/test_bids.py`
  checks the spec table.
- Open: the same set gives `same`. A phantom remesh (the vertices in another
  order, the same edge keys) gives `edge-key` and the right vertex. A
  missing edge key gives `nearest` within one voxel, else `none`.

- Go to a vertex: the route gives the true vertex's position, the voxel
  that `Overlays.voxels` (#144) gives, and the overview vertex that `/snap`
  gives for that position. A vertex out of range gives an error.
- The section along the surface, on phantom meshes:
  - A flat sheet: the order is the same as the order by u.
  - A thin fold (two parallel sheets 0.16 mm apart, joined by a half
    cylinder): the order has the columns of the root's sheet, then the
    columns of the turn, then the columns of the other sheet. No column of
    one sheet comes between two columns of the other.
  - A second sheet near the plane, but not joined to the root's sheet
    within the column radius: none of its columns is in the section.
  - A flat sheet whose image changes across the curve and not along it:
    the section's columns change smoothly with arc length, with no
    alternation (the correlation at lag 1 is not less than at lag 2).

## The hemisphere check

1. Start the viewer on the new input set with the echoes, the signals and a
   notes log in `scratch/wave-2.5-viewer/`. Until session S publishes the
   hemisphere signals, use phantom files or a partial run.
2. In the built-in browser pane: choose each signal as the overlay; open the
   first three sites of `orientangle` and of `shiftmax` from the ranked
   list; write a note at one of them; reload the page; open the note again
   and compare the section.
3. Give the time of each route on the hemisphere and the server's peak
   memory. If the peak goes above 8 GB, ask the coordinator for a slot.

## Work breakdown

| Step | Work | Who | Can run with |
|---|---|---|---|
| 1 | Part A: reproduce and fix bugs 2 and 3 in the browser pane; bugs 1 and 4 | Phase session | 2 |
| 2 | Part B: `viewer/signals.py`, its routes and tests, on phantom files | Sonnet sub-agent, the phase session reviews | 1 |
| 3 | Part B: `static/signals.js`, the `colour` hook, the columns-pane additions | Phase session (Opus) | after 2 |
| 4 | Part C: the spec and `viewer/notes.py`, with its tests | Sonnet sub-agent, the phase session reviews | 3 |
| 5 | Part C: the notes panel, the stored u, open and recheck on the page | Phase session | after 3, 4 |
| 5b | Part D: the user's requests, with their tests | Phase session | 1, 2 |
| 6 | The hemisphere check | Phase session | after 5 and S's PR 1 |

## PRs

1. `fix/viewer-wave2-bugs`: Part A. Branch from `origin/main`.
2. `feature/viewer-signals`: Part B. Branch from `origin/main`.
3. `feature/viewer-notes`: Part C. Stacked on PR 2, because both change
   `columns.js`.
4. `feature/viewer-requests`: Part D. Branch from `origin/main`.

Tell the coordinator when each PR opens.

## Decisions for review (made by the coordinator)

- The overlay shows the worst value of the true vertices under each
  overview vertex, not a point sample, so that a bad column is not hidden by
  the overview's decimation.
- The ranked list drops the vertices within 1 mm of a site, so that one bad
  place gives one row, not hundreds. No threshold decides what is listed.
- A note on another input set opens by edge key, then by position within one
  voxel, else not at all. The viewer never opens a different column without
  saying so.
- The notes log is its own format (`review-notes/1`), not part of the stream
  1 answer log (`wm-edits/1`): a note does not change the labels.
