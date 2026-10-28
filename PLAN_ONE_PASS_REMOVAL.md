# Plan: the column traces without the one pass

Next PR after `feature/streamlines` (the port of the tracers into `baillarger.streamlines` and
`baillarger.column_traces`). Branch: `feature/traces-without-one-pass` (it changes behaviour, so
not `refactor/`).

## Goal

`baillarger.column_traces` builds the traces from one surface (the white mesh) and one field (the
fitted field T), with no voxel seed graph and no fast-marching pass. Every job the one pass does
for the traces is done by something the traces already compute. The one pass stays available to
the prototypes (`prototypes/scripts/onepass_lib.py`, about 15 callers) as recorded evidence; only
its copy in the package goes.

## What the one pass does today

In `column_traces.trace_columns` (`one_pass`, `OnePass`, `setup`, `run`, `geodesic_patches`,
`_shifted_pairs`):

| Role | How | Read by |
|---|---|---|
| A. Fallback tracer, where the primary turned or reached the cap (M1S1: 15,301 turns + 164 caps, 14 % of roots) | `streamlines.ascend` on the equal-arrival time `u` | `pos`, `ns`, `outc`, `tracer` |
| B. The fallback's contact stop | `far()`: the arrival label's patch is > 5 mm from the root's patch | outcome CONTACT, class TOCONTACT (M1S1: 5,035) |
| C. Root and end patches, and patch-to-patch distances along the white surface | geodesic Voronoi patches on the voxel seed graph (26-neighbours with agreeing EDT normals); `D` by Dijkstra; mesh roots mapped to the nearest seed voxel | the classes CROSS / OWN (5 mm rule); `root_patch`, `end_patch` |

In the prototype `column_volume.py` only (horizons, step 5):

| Role | How | Read by |
|---|---|---|
| D. Thickness evidence per patch | `click_evidence.Thickness(B, s, h)`: open thickness along the pass's fronts (`onepass_lib.open_thickness`), face-quadric curvature (`curv_lib.curvature`), `shell_graph`, a GMRF over shell-voxel nodes | `mu_f`, `sd_f` |
| E. Horizons "one pass, equal" and "one pass, field" | where each crossing trace enters the far bank's arrival labels (equal pass; field-weighted pass with `mu_f`, `sd_f`) | `pia_horizon.py` (scores, views), `chart.py` (export); comparison only |
| F. Horizon "field prior" | the pairwise posterior on L from `mu_f`, `sd_f` | `pia_horizon.py`, `flatten_volume.py`, `chart.py`; comparison only |
| G. Profile templates near each end | open traces with roots within 6 mm (`D`) | horizon "profile, top" |

Nothing on the main path reads E or F. `joint_site.py` uses `H_top` and `H_far WM`, and
`pia_horizon.py` builds its pia from `H_top`.

## Replacements

### A, B: the fallback is the ascent of T, with the contact found by descending the far side

- The fitted field T is computed for every tracer, `current` included. It is about 10 s of
  pyamg on a 141³ box, and `fitted` and `fitted_blend` compute it already.
- The fallback is `streamlines.ascend(T, ...)` from the root. T is 0 on the WM and rises through
  the cortex:
  - in open cortex, the ascent reaches the fluid (OPEN);
  - in fused cortex, T rises from both banks' WM to a ridge between them, so the ascent stops
    there (RIDGE). That ridge is the contact.
- At a ridge, decide contact or not by **descending the far side**. Step across the ridge along
  the unsigned direction, then descend −grad T to the WM. The foot is the far bank's root (its
  nearest mesh vertex, by the side rule below).
  - If that root is ≥ SAME_MM along the white surface from the trace's own root, the outcome is
    CONTACT and the class TOCONTACT. The trace keeps its samples up to the ridge, and the far
    root is stored (new key `far_root`).
  - Otherwise it is a ridge within one bank (FAILED, as RIDGE is today).
- This is the same backward projection as the WM extension discussed earlier, so the code is
  shared (`streamlines.descend`).
- `ascend`'s `far` callback goes; the contact decision moves after the path ends.

### C: patches and distances on the white mesh

- Mesh edge graph in mm: `baillarger.mesh_smoothing.edges`, weighted by edge length × voxel size.
- Patches are geodesic Voronoi cells on the mesh, centres at least PATCH_MM = 0.96 mm apart (six
  0.16 mm voxels, as today). `geodesic_patches` moves unchanged into `baillarger.mesh_smoothing`
  (or a small `baillarger.mesh_graph`), because it is generic over a csr graph.
- `D` (patch centre to patch centre, mm) is Dijkstra on the mesh graph, limited to 8 mm as today.
- `root_patch` is the patch of the root vertex itself, with no mapping to a seed voxel.
- **End patch, side rule.** A trace that ends in WM entered it through a face whose outward normal
  faces the trace: n · d_last < 0. Its end vertex is the nearest vertex (k-d tree, k = 8) that
  satisfies that rule. This keeps the two sides of a thin blade apart, the property the seed
  graph's agreeing normals were built for.
- The mesh is the label's surface, so it has the label's topology: genus 0 with `--wm-changes`.
  Geodesics around a blade's tip are then correct.

### D, E, F, G: the prototype's horizons

E and F are debugging information, so they go, along with D, which only they use. The npz no
longer has `H_one pass, equal`, `H_one pass, field` or `H_field prior`, and the viewers no longer
show them:
- `pia_horizon.py` (its HOR list, the scores at the marked points, the section views);
- `flatten_volume.py` (HOR);
- `chart.py` (HOR and its export layers 4–6, with the remaining layers renumbered);
- `edit_priority.py`: the editor's "|joint − field prior|" layer (it reads `H_field prior`). This is
  the edit-priority session's tooling ([edit-priority-maps-and-l](#local_ef135e9a-5b37-4137-b25c-62c1637a0276)),
  so the layer's removal is agreed with that session and lands in this PR or just before it.

**Their replacement: the T ridge**, a default horizon from `column_traces` (key `H_T ridge`). On
every crossing trace and every TOCONTACT fallback, it is the arc length of T's maximum along the
trace: the midline where T rises from the two banks' WM. It replaces "one pass, equal", which was
where the equal-arrival fronts met, and costs one argmax per trace. The thickness-weighted splits
("one pass, field", "field prior") have no replacement in the traces. A thickness-aware split is
the main pipeline's job (`joint_site`, `joint_levels`), and "profile, top" stays as the
data-driven split.

Each viewer keeps "top", "far WM", "fallback contact" (where it shows it) and "profile, top", and
shows "T ridge" in place of "one pass, equal". G (the
templates for "profile, top") uses the mesh patches' `D`. `click_evidence.Thickness` and the
field-weighted pass are no longer called from `column_volume.py`. `onepass_lib.py`
and `click_evidence.py` stay for the other prototype scripts.

## Code changes

- `baillarger.streamlines`: add `descend(T, labels, dface, x0, vox, nmax)` (−grad T to the WM;
  stops at WM, face, missing, length), and drop `ascend`'s `far` argument (CONTACT is decided
  after the path ends).
- `baillarger.mesh_smoothing` (or `baillarger.mesh_graph`):
  - `edge_graph(v, f, voxel_mm)` (csr, mm);
  - `geodesic_patches(graph, spacing)` (moved);
  - `patch_distances(graph, rep, limit_mm)`.
- `baillarger.column_traces`:
  - delete `setup`, `run`, `Setup`, `OnePass`, `one_pass`, `_shifted_pairs`, `_N13` and the
    `baillarger.fronts` import;
  - add `Patches` (mesh patches, `D`, `patch_of(vertex)`, `end_vertex(x, d_last)`) and the
    T-ascent fallback with the far-side descent;
  - `ColumnTraces` gains `far_root`, `T` and `ridge_mm` (the T-ridge horizon), and loses
    `one_pass`. `oriented_normal` is still
    needed for the fitted field.
- `prototypes/scripts/column_volume.py`: E, F and D removed (no `Thickness`, no field-weighted
  pass, no `pairwise_posterior`); G from the mesh `D`.
- `prototypes/scripts/column_tracers.py`: `Pass` and `with_fallback` call the new package
  functions, so `fold_eval.py` and `column_diag.py` keep working.
- Viewers (`pia_horizon.py`, `flatten_volume.py`, `chart.py`, and the editor's layer in
  `edit_priority.py`): the three horizons removed. Before
  editing, check with the sessions that may be working in them; they read the same npz keys, so
  their change ships with this PR.

## Tests (exact answers first)

- `descend` on a linear T is a straight line to the WM, in unit steps.
- **Fallback on the kissing cylinders** (`fold_phantoms` kiss, noise-free; or a minimal copy in
  `tests/`):
  - the ascent from a root on one bank stops at the mid-gap ridge;
  - the far-side descent reaches the other cylinder's foot point, within one step;
  - the class is TOCONTACT and `far_root` is the true far root.
- Fallback on an open slab: the ascent reaches the fluid; OPEN.
- T ridge on the kissing cylinders: on a crossing trace through the gap, `ridge_mm` is at the
  gap's midline, within one step.
- A ridge within one bank (a crown with T's maximum inside the cortex, built by hand): FAILED,
  not TOCONTACT.
- Mesh patches on a plane:
  - centres at least the spacing apart, in mm;
  - each vertex in its nearest centre's patch;
  - `D` equals the analytic geodesic between centres within edge-length error.
- **Blade:** on a thin WM blade's mesh, the geodesic between the two sides goes round the tip, and
  the side rule maps an end on side A to a side-A vertex.
- **Anisotropic voxels:** `edge_graph` distances are in mm (0.16 × 0.16 × 0.32).
- Determinism: two builds are bit-identical.

## Verification (diagnostics, not pass/fail)

- **Phantoms.** `fold_eval.py` with both fallbacks (pass versus T), per phantom:
  - share of the right class (it knows each trace's true class);
  - contact error of fallback traces against the true contact;
  - end error.
  
  Expect the gain at contacts and tight fundi, where T's ridge is the geometric midline, and check
  crowns, where T has no ridge.
- **Real data**, V1V2 and M1S1 side by side with the current build:
  - class counts;
  - own-bank count;
  - marked points reached by crossing traces;
  - for each crossing and TOCONTACT trace, the distance along it between the pass's contact ("one
    pass, equal") and the T ridge, mapped on the marked planes, with the scores at the marked
    points;
  - classes that change, on the marked planes (where and why);
  - runtime and peak memory: the Thickness step goes, about 100 s on M1S1.
- Report: a subsection result per role and estimator. The fallback (pass ascent against T
  ascent) and the patches (seed graph against mesh) are recorded separately, so that a change in
  one is not read as a change in the other.

## Order

1. `descend`, mesh graph functions, and their tests.
2. Mesh patches in `column_traces`, still with the pass fallback, so the patch change can be
   measured alone. Diagnostics run.
3. The T fallback with the far-side descent; delete the package's one pass. Diagnostics run.
4. Remove the one-pass horizons from `column_volume.py` and the viewers (after checking with the
   sessions that may be working in them).
5. Report subsection, checks, PR.

## Done when

- `baillarger.column_traces` imports nothing from `baillarger.fronts`, and has no seed graph.
- The tests pass.
- The phantom and real-data diagnostics are in the report, per role and estimator.
- `column_volume.py` no longer runs the one pass. The npz has no one-pass horizons, and the viewers
  do not show them.

## Not in this PR

- The WM extension of each column (backward tracing into the WM; it shares `descend`).
- The coverage diagnostic, and gap-filling from uncovered voxels.
- The fitted field bounded by the white mesh, and the curvature or equivolume prior.
- A thickness prior on the mesh (if the field prior is wanted again, it comes from the main
  pipeline's thickness prior, not from the one pass).
