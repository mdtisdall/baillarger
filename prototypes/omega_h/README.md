# Omega_h with a boundary lock

This directory builds [SCOREC Omega_h](https://github.com/SCOREC/omega_h) with two patches. Omega_h adapts the tetrahedral mesh of the τ region. Σ is a boundary of that mesh. Our own code owns Σ and checks it (T5's checker and a surface remesher). Omega_h may change the interior and may split the edges of Σ, but it must not remove or move a vertex of Σ.

## Source and licence

| | |
|---|---|
| Repository | https://github.com/SCOREC/omega_h (the maintained fork of sandialabs/omega_h) |
| Commit | `7647c8a52547a737f06de91d3f47952643ca5b64` (2026-10-26, `master`) |
| Licence | BSD-2-Clause (Sandia notice). The patches use the same licence. |
| Build tools | CMake (BSD-3), Ninja (Apache-2.0), pybind11 (BSD-3), zlib (zlib licence) |

No GPL or AGPL code is in the build.

## Build

From the repository root:

```bash
nix develop --command prototypes/omega_h/build.sh WORK_DIR
```

- The script clones the pinned commit into `WORK_DIR/src`, applies `patches/*.patch`, and builds into `WORK_DIR/build`. Then it runs Omega_h's own tests (`ctest`).
- If `WORK_DIR/src` exists, the script builds it as it is. Use this to work on the patches in a clone.
- The build is serial: no MPI, no Kokkos, no OpenMP. `Omega_h_THROW=ON` makes errors into C++ exceptions, so Python gets a `RuntimeError` and does not abort.
- The compiler comes from the devShell. CMake, Ninja, pybind11 and zlib come from the flake's nixpkgs (`nix build --inputs-from .`). `flake.nix` does not change.
- The module is `WORK_DIR/build/src/PyOmega_h.cpython-312-*.so`. It is built for the project's `.venv` Python 3.12.
- A clean build on the laptop takes about 2 minutes with 4 jobs (peak memory 1.5 GB).

Python tests:

```bash
nix develop --command env PYTHONPATH=WORK_DIR/build/src uv run pytest prototypes/scripts/test_omega_h_lock.py
```

Without the module on `PYTHONPATH`, the tests are skipped. CI skips them.

Later, for the GPU: build Kokkos with CUDA and set `Omega_h_USE_Kokkos=ON`. The lock check is a `parallel_for` like the checks next to it, so it runs on the device.

## The patches

### 0001: an opt-in vertex lock (`AdaptOpts::lock_tag`)

- `AdaptOpts::lock_tag` names an `I8` tag with one component. The default is an empty name, and then nothing changes.
- When the name is set, no edge collapse removes a vertex whose tag value is not 0. The check is in `coarsen_element_based1` (`Omega_h_coarsen.cpp`), next to `check_collapse_class`. All collapses go through it: `coarsen_by_size`, `coarsen_slivers` and `coarsen_verts`.
- `validate()` (in `adapt`) stops with an error if the tag is not on every dimension, is not `I8` with one component, or is not transferred with `OMEGA_H_INHERIT`.
- New C++ test `src/unit_lock.cpp` (ctest `run_unit_lock`).

Why a tag on every dimension with `OMEGA_H_INHERIT`: Omega_h drops a tag during adaptation unless a transfer rule keeps it. An `I8` vertex tag has no rule. `OMEGA_H_INHERIT` keeps the values of entities that stay, and it gives each new entity the value of its parent. A split point gets the value of the split edge. An edge cut across a face gets the value of the face. So the values on edges and faces decide if new boundary vertices are locked. Omega_h needs no new transfer code for this.

What moves vertices during adaptation:

| Step | Moves or removes existing vertices? |
|---|---|
| Refinement (edge split) | No. The new vertex is at the midpoint of the edge. A midpoint of a flat triangle edge is on that triangle. |
| Edge collapse | Removes the collapsed vertex. The lock stops this for locked vertices. The other vertices keep their coordinates (they are copied). |
| Edge swap (3D) | No. Swaps change only interior edges (`swap_part1` marks edges with `class_dim == dim`). |
| Snap to the model | Only with EGADS (`snap_and_satisfy_quality`). This build does not have EGADS. |
| `warp_to_limit` | Yes, every vertex, by the caller's `warp` tag. This is the prescribed motion of Σ. The lock does not stop it. |
| `approach_metric` | No. It changes only the metric. |

`grep` for `set_coords` and `"coordinates"` in `src/Omega_h_*.cpp` found no other writes in the adaptation path.

**Split points: what we chose.** The patch does not decide. The caller sets the edge and face values. For Σ, set `lock = 1` on the vertices, edges and faces of Σ, and `0` on the tetrahedra and on all interior entities. Then every vertex on Σ stays, also the vertices that splits make, and Σ is only subdivided. The reason is a measured result (below): if split points are not locked, a later collapse can remove them, and the boundary is then not a subdivision of the input triangles any more, also on flat faces. On a curved Σ that changes the geometry. Our surface remesher can remove the extra vertices of Σ later.

### 0002: Python bindings

- `warp_to_limit(mesh, opts, exit_on_stall=False, max_niters=40)`.
- `AdaptOpts`: the length and quality goals, the `should_*` switches, `nsliver_layers`, `lock_tag`, and `set_transfer(name, Transfer)` / `transfer(name)` for `xfer_opts.type_map`. (A bound `std::map` would be a copy, so a direct assignment would have no effect.)
- `build_from_elems_and_coords(mesh, family, dim, ev2v, coords)` from NumPy arrays.
- `finalize_classification(mesh)`.
- Tags from NumPy already worked (`mesh.add_tag(dim, name, ncomps, array)`, `mesh.get_tag_array(dim, name)`).

## Use from Python

```python
import numpy as np
import PyOmega_h as oh

lib = oh.OmegaHLibrary()
mesh = oh.new_empty_mesh(lib)
oh.build_from_elems_and_coords(mesh, oh.SIMPLEX, 3, tets.ravel(), points.ravel())
oh.classify_by_angles(mesh, np.pi / 4)

opts = oh.AdaptOpts(mesh)
for d in range(4):
    on_sigma = mesh.get_tag_array(d, "class_dim") < 3
    mesh.add_tag(d, "lock", 1, (on_sigma & (d < 3)).astype(np.int8))
opts.lock_tag = "lock"
opts.set_transfer("lock", oh.INHERIT)

mesh.add_tag(oh.VERT, "metric", 1, 1.0 / h**2)  # h: NumPy array, isotropic size per vertex
mesh.add_tag(oh.VERT, "warp", 3, displacement.ravel())
while oh.warp_to_limit(mesh, opts):
    oh.adapt(mesh, opts)
```

## Test results (2026-11-20, laptop, serial build)

C++ (`unit_lock`, 4×4×4 box, `build_box`):

```
lock coarsen: verts 125 -> 104, boundary kept 98 of 98
control coarsen: boundary kept 29 of 98
split points (locked 1): original kept 98 of 98, after refine kept 770 of 770
  final boundary inside the original triangles: 1
split points (locked 0): original kept 98 of 98, after refine kept 98 of 770
  final boundary inside the original triangles: 0
```

All 41 Omega_h tests pass (`ctest -j1`).

Python (`prototypes/scripts/test_omega_h_lock.py`, a 4×4×4 grid of tetrahedra from NumPy with a curved top):

- With the lock, coarsening keeps every boundary vertex at identical coordinates. The boundary triangles are a subdivision of the input triangles (same area, every new triangle in one old triangle). All tetrahedra have positive volume and quality.
- Without the lock (control), coarsening removes boundary vertices, and the boundary is not a subdivision of the input.
- Refine, then coarsen: every split point on the boundary stays.
- `warp_to_limit` with an affine warp, with `adapt` after each step: each locked vertex ends at its point plus its warp (to 1e-12). The boundary is a subdivision of the warped input.
- A lock tag without the `INHERIT` transfer gives a `RuntimeError`.

## Upstream

The patches are written to be offered to SCOREC. Nothing is public yet. The user decides each public step.
