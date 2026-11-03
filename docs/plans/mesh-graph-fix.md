# Shared code: the `mesh_graph` duplicate-edge fix

`docs/version-2-plan.md` §7 says: fix the `edge_graph` bug in its own PR,
before stream 2's phase 0 and the viewer's flat map use it. It is in round 1
(`docs/plans/round-1.md`).

The plan also says that #69 has the fix. That is not true. #69 only
describes the bug. This task writes the fix.

## Goal

`mesh_graph.edge_graph` gives each edge a weight equal to its length in mm,
with no exceptions.

## Done when

1. Every stored entry (i, j) of `edge_graph(v, f, voxel_mm)` equals the
   distance in mm between vertex i and vertex j.
2. The regression tests in "Tests" pass. They fail on `main` today.
3. `nix develop --command scripts/check` passes.

## Facts from the current code

- `src/baillarger/mesh_graph.py:53` `edge_graph(v, f, voxel_mm)` returns a
  `csr_matrix`. It holds the mesh edges (`mesh_smoothing.edges(f)`) and the
  across-triangle edges (`_across(f)`, line 37), weighted by length in mm.
- The bug is at lines 62–63:

  ```
  G = coo_matrix((w, (E[:, 0], E[:, 1])), shape=(len(v),) * 2).tocsr()
  out: csr_matrix = G.maximum(G.T).tocsr()
  ```

  `tocsr()` adds duplicate (i, j) pairs. An across edge that is also a mesh
  edge, in the same order, gets twice its length. On a small
  marching-tetrahedra ball this is 2.8 % of the entries. On a tetrahedron it
  is every edge.
- The current test (`tests/test_mesh_graph.py:20`) uses a plane only. No
  across edge is a mesh edge on that plane, so the test does not find the
  bug.
- Importers: `column_traces.py:121`, `pial_labels.py:126`, and the prototypes
  `turn_experiment.py`, `bend_sweep.py`, `turn_diag.py`.
  `viewer/surface.py:35` has its own function with the same name. It is not
  this module. Do not change it.

## Design

1. Put each edge in order: (min(i, j), max(i, j)).
2. Remove duplicate pairs with `np.unique(..., axis=0)` before the COO
   matrix is made.
3. Compute the weight of each unique pair from the vertex positions.
4. Make the matrix symmetric from the unique pairs.

Do not change the set of edges. Only the weights change.

## Tests

Add these to `tests/test_mesh_graph.py`:

1. **A tetrahedron.** Every stored weight equals the length of its edge.
2. **A small marching-tetrahedra ball.** Use `phantoms` and `mesh`. Every
   stored weight equals the distance between its two vertices, to float
   precision. Check that some across edges are also mesh edges on this
   ball, so the test covers the bug.
3. **The edge set does not change.** On the ball, the sparsity pattern is
   the same as before the fix.

## Effects on other tests

The fix changes geodesic distances where the bug doubled an edge. Run the
tests of `column_traces` and `pial_labels`. If a test fails, find which
expected value depended on the bug. Write the cause in the PR body. Change
the expected value only when the new value is the correct one. Do not change
a tolerance.

## Work breakdown

This task is small. The phase session does it alone. A sub-agent is not
necessary.

## PR

One PR: `fix/mesh-graph-duplicate-edges`.

## Coordination

- Send the finding about #69 to the coordinator. The coordinator records the
  change to the plan.
- Stream 2's phase 0 makes the "local form" of `mesh_graph` (a graph for each
  superblock). That is not part of this task.
