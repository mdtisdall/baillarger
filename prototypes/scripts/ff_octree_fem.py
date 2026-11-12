"""The FEM prior on the octree (docs/plans/flow-field-phantoms.md, *The FEM prior*;
PR 5, task 5.2).

The prior is λ (K + M / ℓ²) with λ = 1 / (sd² · ℓ), as in `ff_grid` (its module
docstring). K and M are the Q1 stiffness and mass matrices in mm units. Each leaf of
side h adds h · K_ref and h³ · M_ref (`ff_grid.q1_reference`) on its 8 corners.

Two forms
---------

- Non-conforming (`nonconforming_matrices`, n × n). Every node is free, also a hanging
  node. A large leaf does not see the hanging nodes on its faces and edges. So the
  field is not continuous across a coarse/fine interface. The mean field uses this
  form (coordinator decision CD-10): each node has its own state.
- Conforming (`conforming_matrices`, n_free × n_free). A hanging node takes the
  value Σ w · (its parents' values). P (`constraint_matrix`, n × n_free) gives every
  node's value from the free nodes' values, with the chains resolved (a parent that
  is hanging is replaced by its own parents). K_c = Pᵀ K P and M_c = Pᵀ M P, with K
  and M from the non-conforming assembly. The field is then continuous: on a face of a
  large leaf, the small leaves' trilinear fields equal the large leaf's field. The
  solve does not use this form (CD-10). It is the reference for the prior's length.

On an octree with no hanging nodes the two forms are the same, and they equal
`ff_grid.fem_matrices` of the uniform grid at that spacing, bit for bit (the leaves
are assembled in the C order of their anchors, the order of the uniform grid's cells;
tested on the base octree and on an octree with all leaves at level 1).

Chains. In the octrees of the tests, no parent of a hanging node is hanging. The
reason: a parent p is a corner of the large leaf B, and one of the small leaves on
B's face or edge touches p. If p were hanging on a leaf one level coarser than B, that
small leaf would touch it at p, two levels apart, which 2:1 balance across corners
does not permit. `constraint_matrix` resolves chains all the same (tested on a stub).

The prior's length (test_ff_octree_fem.py). On a 57³-voxel box, with the fit line in
0.16-mm leaves, both forms give the decay length of the uniform 0.16-mm grid (0.5017
mm conforming, 0.5026 mm non-conforming). Where the line crosses a 0.16/0.32-mm
interface, the conforming form is continuous (C within 7 % of the reference), but the
non-conforming form has a step: C falls to 0.65–0.67 of the reference beyond the
interface (a finding; the test is xfail).

NumPy or Torch (the plan's *NumPy or Torch*)
--------------------------------------------

- Work of the assembly. The same as `ff_grid.fem_matrices` (its module docstring):
  64 entries of K and 64 of M for each leaf, one multiplication for each entry. The
  leaf count C is near the node count n. Target size (n = 4·10⁶ to 4·10⁷): about
  2.6·10⁸ to 2.6·10⁹ entries for each matrix. The cost is the memory traffic of the
  sum of the duplicate entries (`ff_grid._assemble`: chunks reduced to CSR, then added
  in pairs). Before the assembly, one sort of the C anchors (O(C log C)).
- Work of the constraint. P has one entry for each free node and at most 4 for each
  hanging node whose parents are free (more for a chain). Pᵀ K P is two sparse
  products (scipy SpGEMM, compiled code). Its cost is near Σ over the rows of K of
  (entries in the row) × (entries in the matching rows of P): near nnz(K) where most
  nodes are free. The chains are resolved by repeated products P ← A P (A: one level
  of parents), one product for each level of the chain (no product when there is
  no chain).
- Memory. K and M as in `ff_grid` (about 20 and 26 entries for each node, 12 B for
  each entry). K_c and M_c have a few more entries in each row near an interface (the
  hanging nodes' couplings move to their parents), and fewer rows. At the target size:
  the same order as K and M, 1–13 GB for each matrix (`ff_grid`). During Pᵀ K P, K,
  K P and the result are in memory together.
- Measured (CPU, three runs: `_measure` twice, and the same calls under
  tracemalloc): the box of 96 × 96 × 128 voxels, the middle third in x at level 2
  (0.16 mm), level 0 and 1 elsewhere: 416 256 leaves, 437 219 nodes, 23 376 hanging.
  Times: `nonconforming_matrices` 1.06–1.30 s,
  `constraint_matrix` 0.018–0.03 s, `conforming_matrices` (with its own
  non-conforming assembly) 1.31–1.67 s. nnz for each row: K 20.3, M 26.3, K_c 20.9,
  M_c 26.7; P has 475 955 entries for 413 843 free nodes. Matrix memory (data +
  indices + indptr): K 103 MiB, M 133 MiB, K_c 101 MiB, M_c 128 MiB. Peak of the
  memory that NumPy and scipy allocate in each call (tracemalloc, MallocLargeCache=0):
  413 MiB for `nonconforming_matrices` (242 MiB held after it), 46 MiB for
  `constraint_matrix`, 511 MiB for `conforming_matrices`. Process peak RSS 729 MiB, of
  which 357 MiB before the first call (the octree).
- Loop structure. The assembly: one Python loop over the chunks (`ff_grid._assemble`).
  The constraint: one Python loop with one product for each level of the chains, and
  one check (one pass when there is no chain). All other work is whole-array NumPy or
  compiled scipy.sparse code.
- Regularity. Not regular: the leaves have different sizes, the gather of the corners
  is irregular, and the duplicates are summed by a scatter-add. P has a variable
  number of entries in each row.
- Choice: NumPy and scipy.sparse, for the reasons of `ff_grid`. The assembly and the
  constraint are done one time for each octree, and their results are sparse
  matrices. In Torch the sum of duplicates is a scatter-add (`index_add_`) and
  the results are sparse tensors. The plan's device rules do not permit either.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
import ff_grid as FG  # noqa: E402
import ff_types as T  # noqa: E402

Array = T.Array
IntArray = T.IntArray


class _LeavesInCOrder:
    """The parts of an octree that `ff_grid._assemble` reads (n_nodes, cells), with
    the leaves in the C order of their anchors (x slowest, z fastest)."""

    def __init__(self, grid: Any) -> None:
        anchor = np.asarray(grid.anchor, np.int64)
        order = np.lexsort((anchor[:, 2], anchor[:, 1], anchor[:, 0]))
        self.n_nodes = int(grid.n_nodes)
        self.cells = np.ascontiguousarray(np.asarray(grid.cells, np.int64)[order])
        self.cell_size_mm = np.asarray(grid.cell_size_mm, np.float64)[order]


def nonconforming_matrices(grid: Any) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """(K, M) (n × n CSR, mm units): each leaf of side h adds h · K_ref and h³ · M_ref
    on its 8 corners. Every node is free, also a hanging node. Explicit zeros are
    removed from K. K is assembled first, then M (`ff_grid._assemble`)."""
    leaves = _LeavesInCOrder(grid)
    if np.any(leaves.cells < 0):
        raise ValueError("a leaf corner is not a node")
    k_ref, m_ref = FG.q1_reference()
    h = leaves.cell_size_mm
    K = FG._assemble(leaves, h, k_ref)  # type: ignore[arg-type]
    K.eliminate_zeros()
    M = FG._assemble(leaves, h**3, m_ref)  # type: ignore[arg-type]
    return K, M


def constraint_matrix(grid: Any) -> tuple[sp.csr_matrix, IntArray]:
    """(P, free_ids): P (n × n_free CSR) gives every node's value from the free nodes'
    values. free_ids (n_free,) are the node ids that are not hanging, sorted; column j
    of P is node free_ids[j]. The row of a free node is the identity. The row of a
    hanging node is Σ w · (its parents' rows), with the chains resolved: a parent that
    is hanging is replaced by its own parents, until all are free. The weights are
    1/2 and 1/4, so the entries of P are exact. Raises ValueError if the chains do not
    end (a cycle)."""
    n = int(grid.n_nodes)
    h_ids, h_par, h_w = grid.hanging
    h_ids = np.asarray(h_ids, np.int64)
    is_hanging = np.zeros(n, bool)
    is_hanging[h_ids] = True
    free_ids = np.flatnonzero(~is_hanging).astype(np.int64)
    # A: one level of parents (n × n). Free rows: the identity. Hanging rows: the
    # parents with weight > 0.
    par = np.asarray(h_par, np.int64)
    w = np.asarray(h_w, np.float64)
    use = w > 0
    rows = np.concatenate([free_ids, np.repeat(h_ids, use.sum(1))])
    cols = np.concatenate([free_ids, par[use]])
    vals = np.concatenate([np.ones(len(free_ids)), w[use]])
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    A.sum_duplicates()
    A.sort_indices()
    P = A
    for _ in range(n + 1):
        Pc = P.tocsc()
        if Pc[:, h_ids].nnz == 0:
            break
        P = sp.csr_matrix(A @ P)
        P.sum_duplicates()
        P.sort_indices()
    else:  # pragma: no cover - a cycle in the hanging parents
        raise ValueError("the chains of hanging parents do not end")
    P = sp.csr_matrix(P.tocsc()[:, free_ids])
    P.eliminate_zeros()
    P.sort_indices()
    return P, free_ids


def conforming_matrices(
    grid: Any,
) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, IntArray]:
    """(K_c, M_c, P, free_ids): K_c = Pᵀ K P and M_c = Pᵀ M P (n_free × n_free CSR,
    rows and columns in the order of free_ids), with K and M from
    `nonconforming_matrices` and P from `constraint_matrix`. Explicit zeros are removed
    from K_c."""
    K, M = nonconforming_matrices(grid)
    P, free_ids = constraint_matrix(grid)
    Pt = sp.csr_matrix(P.T)
    K_c = sp.csr_matrix(Pt @ (K @ P))
    del K
    K_c.sum_duplicates()
    K_c.eliminate_zeros()
    K_c.sort_indices()
    M_c = sp.csr_matrix(Pt @ (M @ P))
    del M
    M_c.sum_duplicates()
    M_c.sort_indices()
    return K_c, M_c, P, free_ids


def nonconforming_precision(grid: Any, length_mm: float, sd: float) -> sp.csr_matrix:
    """λ (K + M / ℓ²) with λ = 1 / (sd² · ℓ), ℓ = length_mm (n × n CSR), from
    `nonconforming_matrices`."""
    K, M = nonconforming_matrices(grid)
    return FG.precision_from(K, M, length_mm, sd)


def conforming_precision(grid: Any, length_mm: float, sd: float) -> sp.csr_matrix:
    """λ (K_c + M_c / ℓ²) with λ = 1 / (sd² · ℓ), ℓ = length_mm (n_free × n_free CSR,
    in the order of `constraint_matrix(grid)[1]`), from `conforming_matrices`."""
    K_c, M_c, _, _ = conforming_matrices(grid)
    return FG.precision_from(K_c, M_c, length_mm, sd)


def _measure() -> dict[str, Any]:
    """Time and matrix memory of both forms on one phantom-size octree: the box of
    96 × 96 × 128 voxels, the level-0 leaves with x in the middle third refined to level
    2 (0.16 mm), then balanced."""
    import time

    import ff_octree as FO

    g = FO.OctreeGrid.base((96, 96, 128))
    x0 = g.anchor[:, 0]
    third = int(g.extent[0]) // 3
    g = g.refine(np.flatnonzero((x0 >= third) & (x0 < 2 * third)))
    g = g.refine(np.flatnonzero((g.level == 1) & (g.anchor[:, 0] >= third)
                                & (g.anchor[:, 0] < 2 * third)))  # fmt: skip
    out: dict[str, Any] = dict(
        n_nodes=g.n_nodes, n_cells=g.n_cells, n_hanging=len(g.hanging[0])
    )
    t = time.perf_counter()
    K, M = nonconforming_matrices(g)
    out["t_nonconforming_s"] = time.perf_counter() - t

    def mib(A: sp.csr_matrix) -> float:
        return float(A.data.nbytes + A.indices.nbytes + A.indptr.nbytes) / 2**20

    out["K_nnz_per_node"] = K.nnz / K.shape[0]
    out["M_nnz_per_node"] = M.nnz / M.shape[0]
    out["K_MiB"], out["M_MiB"] = mib(K), mib(M)
    del K, M
    t = time.perf_counter()
    P, free_ids = constraint_matrix(g)
    out["t_constraint_s"] = time.perf_counter() - t
    out["P_nnz"], out["n_free"] = P.nnz, len(free_ids)
    t = time.perf_counter()
    K_c, M_c, _, _ = conforming_matrices(g)
    out["t_conforming_s"] = time.perf_counter() - t
    out["Kc_nnz_per_node"] = K_c.nnz / K_c.shape[0]
    out["Mc_nnz_per_node"] = M_c.nnz / M_c.shape[0]
    out["Kc_MiB"], out["Mc_MiB"] = mib(K_c), mib(M_c)
    return out


if __name__ == "__main__":
    for k, v in _measure().items():
        print(f"{k}: {v}")
