# Phase 2.1: the equivolume cross-section

Stream 2, phase 1 of `docs/version-2-plan.md` (§4d). Read the plan's §2
(*Domain edges*), §4c steps 2 and 4, and §5c step 5 (*Depth within a
column*). The rules in `docs/plans/round-2.md` apply.

## Goal

The ray sampler gets the method **equivolume**: the same straight rays as
**normal**, with each column's cross-section A(s) and volume V(s), and a
mark where A(s) ≤ 0. Stream 3 normalises V between its two named levels
(phase 3.5).

## Done when

1. `docs/formats/ray-sampler.md` describes `equivolume`, A(s), V(s) and the
   mark, by an edit in place. The "Not in version 1" section goes.
2. The exact tests below pass, and every test in `tests/test_rays.py` and
   `tests/test_column_volume.py` passes.
3. The sampler runs `equivolume` on the hemisphere. The PR body gives the
   diagnostics in "The hemisphere run".
4. `nix develop --command scripts/check` passes.

## Facts from the current code

### `src/baillarger/rays.py` (527 lines)

- `METHODS = ("normal",)` (45). `sample()` refuses other methods with
  `ValueError` (240).
- `Rays` (97): `roots, method, step, k_in, k_out, sample_start, s,
  position, interval_start, interval_s0, interval_s1, interval_label,
  interval_class, derived_class`. No A or V field.
- `RaySampler.__init__` (134–168) memory-maps the vertices `x`, the
  normals `n`, the edge keys, the normal lengths and the block index. It
  does not read the faces or the face ranges (`idx[:,5:7]`).
- `_halo_box` (171), `roots()` (188–220): owned vertices, then halo
  vertices, each in ascending vertex order.
- `_samples()` (508): `p(s) = x0 + s·n` in float64, rounded to float32.
- `_voxel_rays`, `_to_voxel` (343–358): float64 element by element, so
  the result does not depend on the batch.

### `docs/formats/ray-sampler.md`

- Lines 9–11: phase 2.1 adds `equivolume`, A(s) and V(s) by an edit in
  place.
- "Not in version 1" lists the method, A(s) in mm², V(s) in mm³ with
  V(−e_in) = 0, and the mark where A(s) ≤ 0.
- The doc test requires the output table to list the `Rays` fields in
  order, with their types.
- Determinism, item 1: "A column depends only on its root". The 1-ring
  changes this.
- Samples: "Sample k is at the same s for every extent that contains it. A
  longer extent only adds samples."

### The prototype

- `column_traces.offset_equivolume` (`column_traces.py` 608–625): it offsets
  the whole mesh by i·step along each vertex's own normal, computes the dual
  area vector with `column_front.dual_areas` (`column_front.py` 100), takes
  the root's dual area vector on the root's normal, clips at 0, normalises
  by A at the root, and integrates by the trapezoid rule.
- Its tests (`tests/test_column_traces.py` about 720–775): ball
  (r³ − r0³)/(3 r0²), cylinder (r² − r0²)/(2 r0), tube, slab. Median within
  2 %.

### Callers

- `column_volume.py` imports `Rays` (19). `take_columns` (47–82) slices
  each field by hand. Add each new field there, or a batch carries stale
  arrays.
- `tests/test_column_volume.py` `make_rays` (59, 99–121) builds a `Rays`
  with all 14 fields as keywords.
- `tests/test_rays.py`: line 68 asserts `METHODS == ("normal",)`; line 506
  lists `{"method": "equivolume"}` as refused; `as_column` (465) compares
  each field of each column.
- `mesh_graph.local_graph` reads the faces of the halo's blocks but does
  not return them.
- `white_surf.WhiteSurface.load(surf, block)` (377) can load one block's
  faces. A block's faces use its own vertices and some of earlier blocks'
  vertices (`white-surf.md` 130).

## Design

### The cross-section is a quadratic in s

The cross-section of a column at s is the root's dual area vector on the
1-ring moved along each vertex's own normal, taken on the root's normal:

A(s) = n₀ · N₀(s), where N₀(s) = Σ over the faces f round the root of
(1/6) (x_b(s) − x_a(s)) × (x_c(s) − x_a(s)), and x_i(s) = x_i + s n_i.

Each x_i(s) is linear in s, so each cross product is quadratic in s, and
A(s) = a₀ + a₁ s + a₂ s², exactly. Compute the three coefficients per
column, in float64. Then:

1. A(s) at each sample is exact, and does not depend on the step or the
   extent.
2. V(s) is the exact integral of max(A, 0): a cubic between the real roots
   of A. It needs no trapezoid rule.
3. The mark: A(s) ≤ 0 at the sample.

On a plane A is constant. On a sphere with radial unit normals,
x_i(s) = x_i (1 + s/R), so A(s) = A₀ (1 + s/R)², exactly.

Check this derivation before you code it. If it is wrong, tell the
coordinator.

### The origin of V

The spec now says V(−e_in) = 0. That conflicts with the spec's rule that a
sample does not depend on the extent. **Coordinator decision for review:**
V is measured from the inward cap, V(−CAP_IN) = 0, for every call. Then V
at each s is the same for every extent, and only differences of V enter
the depth (plan §4c step 2). Write this in the spec. Send the plan finding
to the coordinator: §4c step 2 says V(−e_in) = 0.

### The faces

1. `RaySampler` reads the faces and the face ranges of the block index.
2. For a call, take the faces of every block whose box meets the halo box
   widened by 2 voxels. A root's 1-ring faces are then all present. Test
   this.
3. For each root, find its 1-ring faces. Use one vertex-to-faces map for
   the call, built from the faces read.
4. A vertex on a mesh boundary does not occur: `white-surf/1` pieces are
   closed. Assert it.

### The outputs

1. Per sample: `area` (A(s), mm², float32), `volume` (V(s), mm³, float32),
   `area_nonpositive` (bool).
2. Per column: `area_coef` (a₀, a₁, a₂, float64), so that a reader can
   evaluate A at any s.
3. For `normal`, these arrays have zero length. Write this rule in the spec.
   Ask the coordinator if you prefer another rule.
4. Add each field to `take_columns` in `column_volume.py`, and to the spec
   tables in order.
5. Determinism: "A column depends only on its root and its 1-ring (their
   positions and normals)". Edit the spec.

## Exact tests

- A plane: A is constant, and V is linear in s.
- A sphere built with exact radial unit normals (a phantom written as
  `white-surf/1`): A(s) = A₀ (1 + s/R)² to float rounding, and V is its
  integral.
- The marching-tetrahedra ball of `tests/test_rays.py`: A(s)/A(0) against
  (1 + s/R)², with the median and the spread reported. Its normals are
  smoothed, so this test has a tolerance. Give the reason for the
  tolerance in the test.
- A focal point: a cylinder of radius r, with s inward past r. A reaches 0
  at s = −r, the mark is set there and beyond, and V stops rising.
- V does not depend on the extent or the step.
- A column is the same, bit for bit, in every superblock (extend
  `test_a_column_is_the_same_in_every_superblock`).
- `normal` gives the same samples as before, byte for byte, and zero-length
  A and V arrays.
- Update the tests at `tests/test_rays.py` 68 and 506, `as_column`, and
  `make_rays` in `tests/test_column_volume.py`.

## The hemisphere run

Read `scratch/v2/derivatives/`. Write to `scratch/phase-2.1/`. The normal
pass of the hemisphere takes 27 s. Measure one superblock first. Ask the
coordinator for a slot if the full pass peaks above 8 GB.

Diagnostics for the PR body (views, not pass/fail numbers):

1. The fraction of samples with A ≤ 0, by s, and against |n̄|.
2. The histogram of A(+2 mm)/A(0) and A(−1 mm)/A(0).
3. A map of A(+2 mm)/A(0) on the overview mesh, as a PNG.
4. The time and the peak memory, against `normal`.
5. Analysis: the cost per column of the 1-ring (faces read, flops) at 50
   µm whole brain, and the extra halo of 2 voxels.

## Work breakdown

| Step | Work | Who | Can run with |
|---|---|---|---|
| 1 | Check the quadratic derivation; write the spec edit | Phase session | — |
| 2 | Faces in `RaySampler`, the vertex-to-faces map | Sonnet sub-agent | 3 |
| 3 | The exact sphere and cylinder phantoms in `white-surf/1` | Sonnet sub-agent | 2 |
| 4 | A, V, the mark, `area_coef`; `take_columns` | Phase session | after 1 and 2 |
| 5 | The tests | Phase session, with a Sonnet sub-agent for test updates | after 3 and 4 |
| 6 | The hemisphere run and the diagnostics | Phase session | after 5 |

## PRs

1. `feature/ray-sampler-equivolume`: everything. Branch from `origin/main`.

Tell the coordinator when the PR opens.

## Decisions for review (made by the coordinator)

- V(−CAP_IN) = 0 for every call, in place of V(−e_in) = 0.
- A, V and the mark are zero-length for `normal`.
- A per-column `area_coef` field gives A(s) in closed form.
