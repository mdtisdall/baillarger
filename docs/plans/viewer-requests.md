# Viewer requests from the wave 2.1 review

The user's feature requests for the viewer, from the wave 2.1 review. They
are built now, during wave 2.1, so that the review can tell viewer artefacts
from errors in the data. Read the plan's §1 (working rules) and
`docs/plans/wave-2.1-normals.md` ("The review").

The user adds requests here during the review. Each request gets an exact
test. The work is one PR, `feature/viewer-requests`, from `origin/main`.

## Goal

1. **Go to a vertex**, from a box or from the page's address.
2. **A profile section that follows the surface**, so that the section
   itself makes no bands.

## Facts from the current code

- The page: `src/baillarger/viewer/static/index.html`. The shared state
  (`POST /state`, `GET /events`) holds the WM selection: `{vertex,
  true_vertex, world_mm, voxel, trace, root_mm}`, where `vertex` is an
  overview vertex. `VIEWER.select(sel, frame)` sets it.
- `GET /snap?mm=x,y,z` gives the overview vertex nearest a point.
- `Overlays.voxels` (`src/baillarger/viewer/overlay.py`, #144) gives the
  voxel nearest each true vertex.
- The columns pane: `static/columns.js` and
  `src/baillarger/viewer/columns.py`. A request takes the columns whose
  roots lie within the radius of the selected true vertex, along the
  surface, on the local graph of the vertex's superblock. The profile
  section (`section()` in `columns.js`) now keeps the columns whose roots
  lie within a half width (default 0.25 mm) of the plane through the
  selected root that holds its normal and the section axis u, and sorts
  them by u. The page has only the overview mesh; the true mesh is on the
  server.

## Request 1: go to a vertex

- A box in the header takes a true vertex number. Enter makes that vertex
  the WM selection, as a click on it does, and turns the 3D view to it.
- The address parameter `?v=<true vertex>` does the same when the page
  opens.
- A new route, `GET /vertex/true?v=<true vertex>`, gives the selection for a
  true vertex: `true_vertex`, `world_mm`, `voxel` (as `Overlays.voxels`),
  and `vertex` (the overview vertex that `/snap` gives for that position).
- A number that is not a true vertex gives an error in the log, and the
  selection does not change.

## Request 2: the profile section follows the surface

Two problems of the current section, both seen in the review:

- Where the WM is one voxel thick, the plane cuts the surface two times,
  close together, and the sort by u interleaves the two cuts. The profile
  shows alternating bands (true vertex 10909480).
- On flat, regular cortex, the strip of ±0.25 mm (3 voxels) puts columns
  from different lines across the strip next to each other. Columns close
  across the strip were alike; columns 0.16–0.5 mm apart differed 24 times
  more in their mean level. The profile shows alternating bands (true
  vertex 12218203).

The new section:

1. **The curve.** The cut of the same plane (through the selected root,
   holding its normal and u) with the true white mesh. Only the piece of the
   cut that goes through the selected root. It runs in both directions from
   the root, to the column radius along the curve.
2. **The columns.** One column for each step along the curve: the column
   whose root is nearest to the curve point at that step. The step is half
   a voxel of arc length (`SECTION_STEP_VOXELS`), not a spacing of the roots:
   the mesh has many vertices a few hundredths of a millimetre apart, so a
   spacing of the roots depends on the mesh and varies from site to site.
   Where roots are closer than half a voxel, "nearest" shows one a step and
   "mean across" averages those within one voxel.
3. **The order.** Arc length along the curve, from the root (negative on one
   side). Where the surface turns around a thin edge, the curve goes around
   the turn, and the profile shows the turn at one place.
4. **"Mean across".** An option: at each step, the mean of the columns
   whose roots lie within one voxel of the curve point.
5. The half-width menu goes.

The section does not label the sides of the WM, and it does not use the
sign of the normals.

The server computes the curve, because it has the true mesh: a new route
`GET /columns/section?v=&radius=&u=x,y,z&mode=nearest|mean` (with the
other parameters of `/columns`) gives, for each step, the arc length and
the column indices of the `/columns` answer. The page sends u (the 3D
view's screen right at the time of the selection, as now) and draws the
answer.

## Request 3: the layout of the panes (after the user's test of #151)

1. The surface pane has no "White surface" heading; its bar keeps the level menu.
2. "columns: off | section | all" in the surface pane's bar (the 3D lines) and in the volume pane's bar (the bricks
   tiles). section, the default: the columns of the section (one a step; mean across, the columns of each step's
   mean). all: every column in the radius (the cap applies). The volume pane's controls act on the bricks viewer only
   and are disabled with NiiVue (NiiVue goes under its own plan).
3. A "section roots" checkbox in each of those bars, apart from the menu: the 3D view as before; bricks, the roots
   within half a voxel of a tile's slice. The picked root stays white.
4. A "log" toggle in the header, per window (localStorage). While the log is hidden, an error puts a red dot on the
   toggle until the log is shown. The information line stays.
5. The section's roots are coloured by their step's place along the section (plasma, 0.12 to 0.95 of it, with a
   dark rim in the slices only: in the 3D view the roots are too close for rims), and the same colours run in a
   thin bar along the top of the section. Mean across: every root of a step has that step's colour.
6. A tooltip over the section gives a column's details (true vertex, arc, distance, |n̄|, derived class, label
   intervals); one line under the section names the picked column. The details list goes.

Every setting except the log toggle goes through the shared state. No route changes: the roots of "section" are the
`roots` of `/columns/section`, which the route test already checks.

## Exact tests

- Go to a vertex: the route gives the true vertex's position, the voxel
  that `Overlays.voxels` gives, and the overview vertex that `/snap` gives
  for that position. A vertex out of range gives an error.
- The section, on phantom meshes:
  - A flat sheet: the order is the same as the order by u.
  - A thin fold (two parallel sheets 0.16 mm apart, joined by a half
    cylinder): the order has the columns of the root's sheet, then the
    columns of the turn, then the columns of the other sheet. No column of
    one sheet comes between two columns of the other.
  - A second sheet near the plane, but not joined to the root's sheet
    within the column radius: none of its columns is in the section.
  - A flat sheet whose image changes across the curve and not along it: the
    section's columns change smoothly with arc length, with no alternation
    (the correlation at lag 1 is not less than at lag 2).
- `test_the_white_mode_serves_every_route` covers the new routes.

## The check

In the built-in browser pane, on `scratch/v2/derivatives-main/`:

1. `?v=10909480` opens that vertex. The box opens 12218203.
2. At 10909480 the section shows the turn around the thin edge at one place,
   without alternating bands. At 12218203 it shows no alternating bands.
3. Screenshots of both sections, before and after, in the PR body.

## Work breakdown

| Step | Work | Who |
|---|---|---|
| 1 | The route for request 1, the box and `?v=`, with the test | the session |
| 2 | The section route in `columns.py`, with the phantom tests | the session |
| 3 | `columns.js`: the new section, the "mean across" option | the session |
| 4 | The check | the session |

## Decisions for review (made by the coordinator)

- The curve is computed on the server, on the true mesh. The page's overview
  mesh is too coarse for it.
- "Nearest" is the default. "Mean across" averages columns, so it hides
  their own variation; it is an option, not the default.
