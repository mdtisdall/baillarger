"""The uniform grid and the FEM prior of the flow-field solver (PR 3, task 3.1).

`UniformGrid` puts nodes at the corners of equal cubic cells over a phantom's box. It
implements the `ff_types.Grid` protocol. PR 5 puts an octree behind the same protocol.

The helpers below `UniformGrid` use only the `Grid` protocol, so that the octree can use
them: `axis_neighbours`, `edge_vectors_mm`, `fem_matrices`, `fem_precision`,
`node_weights_vox`, `edge_weights_vox`, `face_weights_vox`.

The FEM prior. Q1 (trilinear) elements on the cells, in mm. K is the stiffness matrix
(∫ ∇φ_a · ∇φ_b), M the mass matrix (∫ φ_a φ_b). The precision is λ (K + M / ℓ²) with
λ = 1 / (sd² · ℓ). In 3D this is the Matérn case α = 1: the Green's function of
λ (−Δ + 1/ℓ²) is e^(−r/ℓ) / (4π λ r). It has no pointwise variance, so the variance of a
node rises as the spacing falls. The covariance at a fixed distance r > 0 converges.

NumPy or Torch (the plan's *NumPy or Torch*), for `fem_matrices`:

- Work. Each cell gives 64 entries of K and 64 of M (8 × 8 corners). The number of
  cells C is near the number of nodes n. Phantom size (n ≈ 1.2·10⁶ at 0.16 mm on the
  box of 96 × 96 × 128 voxels): about 7.7·10⁷ entries for each matrix. Target size
  (n = 4·10⁶ to 4·10⁷): 2.6·10⁸ to 2.6·10⁹ entries. The arithmetic is one
  multiplication for each entry. The cost is the memory traffic of the sum of the
  duplicate entries.
- Memory. The cells go in chunks of CHUNK_CELLS cells. One chunk has 64 · CHUNK_CELLS
  entries of (row, column, value), int32 indices. Each chunk is reduced to CSR
  (duplicates summed). The chunk matrices are added in pairs, level by level, and each
  part is released when it has been added. K is assembled first, then M. The result
  has about 27 entries per node in M and 21 in K (K is zero between the two ends of a
  cell edge), 12 B per entry. Phantom size: about 0.4 GB for M and 0.3 GB for K.
  Target size: 1.3–13 GB for M and 1.0–10 GB for K. At the target size, store only
  the precision, or apply the constant stencil of a uniform grid without a matrix.
- Measured (96 × 96 × 128 voxels at 0.16 mm, 1.18·10⁶ nodes, CPU): 2.9 s; nnz 20.5
  per node in K and 26.5 in M; 282 MiB for K and 362 MiB for M. Process peak with
  the macOS setting MallocLargeCache=0 (freed memory goes back to the system): 1190
  MiB, of which 164 MiB before the call (the earlier assembly, which concatenated the
  COO parts of all chunks before one conversion: 1734 MiB). With the default macOS
  allocator the peak RSS was 2418 MiB, because the allocator keeps freed large blocks
  resident.
- Loop structure. One Python loop over the chunks for each matrix (about 20 at the
  phantom size, 60 to 600 at the target size). In each chunk, whole-array NumPy
  operations and one scipy COO-to-CSR conversion (compiled code). Then about log2 of
  the chunk count levels of sparse additions (compiled code).
- Regularity. On the uniform grid the pattern is regular. On the octree it is an
  irregular gather and a scatter-add of the duplicates.
- Choice: NumPy and scipy.sparse. The assembly is done one time for each grid, and
  its result is a sparse matrix. In Torch the sum of duplicates is a scatter-add
  (`index_add_`) and the result is a sparse tensor. The plan's device rules do not
  permit either. NumPy already runs each chunk as whole-array operations.
"""

from __future__ import annotations

import math
from functools import cached_property
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from ff_types import VOXEL_MM, Array, BoolArray, Grid, IntArray

CHUNK_CELLS = 1 << 16  # cells per chunk in fem_matrices


# ---------------------------------------------------------------- the uniform grid


class UniformGrid:
    """A uniform grid with nodes at cell corners over a box of voxels.

    The node with index (i, j, k) is at the voxel position (i, j, k) · h_vox, with
    h_vox = spacing_mm / VOXEL_MM. Along axis a there are
    ceil((shape_vox[a] − 1) / h_vox) + 1 nodes, so the nodes cover every voxel centre.
    The node id is np.ravel_multi_index((i, j, k), dims) (C order). The cell id is the
    same function of the index of its corner 0 and dims − 1.
    """

    kind: Literal["uniform", "octree"]

    def __init__(self, shape_vox: tuple[int, int, int], spacing_mm: float) -> None:
        if spacing_mm <= 0:
            raise ValueError("spacing_mm must be positive")
        if any(int(s) < 2 for s in shape_vox):
            raise ValueError("each side of the box must have at least 2 voxels")
        self.kind = "uniform"
        self.shape_vox = (int(shape_vox[0]), int(shape_vox[1]), int(shape_vox[2]))
        self.spacing_mm = float(spacing_mm)
        self.h_vox = self.spacing_mm / VOXEL_MM
        # round before ceil, so that 0.48 / 0.16 = 2.9999999999999996 counts as 3
        d = [math.ceil(round((s - 1) / self.h_vox, 9)) + 1 for s in self.shape_vox]
        self.dims = (d[0], d[1], d[2])
        self.cell_dims = (d[0] - 1, d[1] - 1, d[2] - 1)

    # -------------------------------------------------- counts and positions

    @property
    def n_nodes(self) -> int:
        return self.dims[0] * self.dims[1] * self.dims[2]

    @property
    def n_cells(self) -> int:
        c = self.cell_dims
        return c[0] * c[1] * c[2]

    @cached_property
    def node_index(self) -> IntArray:
        """(n, 3) the index (i, j, k) of each node."""
        idx = np.indices(self.dims, dtype=np.int64).reshape(3, -1).T
        return np.ascontiguousarray(idx)

    @cached_property
    def nodes(self) -> Array:
        """(n, 3) node positions, voxel space."""
        return self.node_index.astype(np.float64) * self.h_vox

    @cached_property
    def node_level(self) -> IntArray:
        """(n,) zeros: a uniform grid has one level."""
        return np.zeros(self.n_nodes, np.int64)

    # -------------------------------------------------- cells

    @cached_property
    def _cell_index(self) -> IntArray:
        """(C, 3) the node index (i, j, k) of corner 0 of each cell."""
        idx = np.indices(self.cell_dims, dtype=np.int64).reshape(3, -1).T
        return np.ascontiguousarray(idx)

    @cached_property
    def cells(self) -> IntArray:
        """(C, 8) node ids of the corners, corner index i + 2 j + 4 k."""
        ni, nj, nk = self.dims
        base = np.ravel_multi_index(tuple(self._cell_index.T), self.dims)
        offs = np.array(
            [((c & 1) * nj + ((c >> 1) & 1)) * nk + ((c >> 2) & 1) for c in range(8)],
            np.int64,
        )
        return np.asarray(base[:, None] + offs[None, :], np.int64)

    @cached_property
    def cell_origin(self) -> Array:
        """(C, 3) corner 0 of each cell, voxel space."""
        return self._cell_index.astype(np.float64) * self.h_vox

    @cached_property
    def cell_size_mm(self) -> Array:
        """(C,) the side of each cell (mm): spacing_mm for every cell."""
        return np.full(self.n_cells, self.spacing_mm)

    @cached_property
    def prior_shaped(self) -> BoolArray:
        """(C,) True if the cell side is less than VOXEL_MM."""
        return np.asarray(self.cell_size_mm < VOXEL_MM * (1.0 - 1e-9), np.bool_)

    # -------------------------------------------------- edges, hanging nodes, colours

    @cached_property
    def edges(self) -> IntArray:
        """(E, 2) node ids of each cell edge, i < j, each edge once, sorted."""
        ids = np.arange(self.n_nodes, dtype=np.int64).reshape(self.dims)
        parts = []
        for a in range(3):
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[a] = slice(0, -1)
            hi[a] = slice(1, None)
            parts.append(
                np.stack([ids[tuple(lo)].ravel(), ids[tuple(hi)].ravel()], axis=1)
            )
        e = np.concatenate(parts)
        order = np.lexsort((e[:, 1], e[:, 0]))
        return np.ascontiguousarray(e[order])

    @property
    def hanging(self) -> tuple[IntArray, IntArray, Array]:
        """Empty arrays: a uniform grid has no hanging nodes."""
        return (
            np.zeros(0, np.int64),
            np.zeros((0, 4), np.int64),
            np.zeros((0, 4), np.float64),
        )

    @cached_property
    def colour(self) -> IntArray:
        """(n,) the parity of the node index: (i mod 2) + 2 (j mod 2) + 4 (k mod 2)."""
        p = self.node_index % 2
        return np.asarray(p[:, 0] + 2 * p[:, 1] + 4 * p[:, 2], np.int64)

    @property
    def n_colours(self) -> int:
        return 8

    # -------------------------------------------------- queries

    def locate(self, X: Array) -> tuple[IntArray, Array]:
        """(cell id (N,), local coordinates (N, 3) in [0, 1]) of each voxel position.

        A point outside the nodes' box gets cell −1 and local coordinates 0. A point on
        a face between two cells goes to the cell with the larger corner-0 index (local
        0 on that axis). A point on the last node plane goes to the last cell (local 1).
        """
        t = np.asarray(X, np.float64).reshape(-1, 3) / self.h_vox
        top = np.array(self.cell_dims, np.float64)
        inside = np.all((t >= 0.0) & (t <= top), axis=1)
        c = np.clip(np.floor(t), 0, top - 1).astype(np.int64)
        local = np.where(inside[:, None], t - c, 0.0)
        cid = np.ravel_multi_index(tuple(c.T), self.cell_dims)
        cid = np.where(inside, cid, -1).astype(np.int64)
        return cid, np.asarray(local, np.float64)

    def morton(self) -> IntArray:
        """(C,) the Morton key of each cell: the bits of (i, j, k) of corner 0
        interleaved, bit b of i at 3b, of j at 3b + 1, of k at 3b + 2."""
        idx = self._cell_index
        key = np.zeros(len(idx), np.int64)
        for b in range(21):
            for a in range(3):
                key |= ((idx[:, a] >> b) & 1) << (3 * b + a)
        return key

    def arrays(self) -> dict[str, npt.NDArray[Any]]:
        """The arrays that define the grid, keys prefixed "grid_"."""
        return {
            "grid_kind": np.array(self.kind),
            "grid_shape_vox": np.array(self.shape_vox, np.int64),
            "grid_spacing_mm": np.array(self.spacing_mm),
            "grid_h_vox": np.array(self.h_vox),
            "grid_dims": np.array(self.dims, np.int64),
            "grid_node_level": self.node_level,
            "grid_cells": self.cells,
            "grid_cell_origin": self.cell_origin,
            "grid_cell_size_mm": self.cell_size_mm,
        }


# ---------------------------------------------------------------- generic helpers


def axis_neighbours(grid: Grid) -> IntArray:
    """(n, 6) the node ids at +x, −x, +y, −y, +z, −z of each node; −1 where none.

    From grid.edges and grid.nodes. Raises ValueError if an edge is not along an axis,
    or if two edges give the same neighbour slot.
    """
    e = grid.edges
    d = grid.nodes[e[:, 1]] - grid.nodes[e[:, 0]]
    ad = np.abs(d)
    axis = np.argmax(ad, axis=1)
    length = ad[np.arange(len(e)), axis]
    if np.any(ad.sum(1) - length > 1e-9 * np.maximum(length, 1.0)):
        raise ValueError("an edge is not along an axis")
    pos = d[np.arange(len(e)), axis] > 0
    # slot of the second node as seen from the first, and the reverse
    s_fwd = 2 * axis + np.where(pos, 0, 1)
    s_bwd = 2 * axis + np.where(pos, 1, 0)
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    slot = np.concatenate([s_fwd, s_bwd])
    flat = src * 6 + slot
    if len(np.unique(flat)) != len(flat):
        raise ValueError("two edges give the same neighbour slot")
    nbr = np.full(grid.n_nodes * 6, -1, np.int64)
    nbr[flat] = dst
    return nbr.reshape(-1, 6)


def edge_vectors_mm(grid: Grid) -> Array:
    """(E, 3) x_j − x_i in mm for each edge (i, j) of grid.edges."""
    e = grid.edges
    return np.asarray((grid.nodes[e[:, 1]] - grid.nodes[e[:, 0]]) * VOXEL_MM)


def q1_reference() -> tuple[Array, Array]:
    """(K_ref, M_ref) (8, 8): the Q1 element matrices of the unit cube, corner order
    i + 2 j + 4 k. For a cube of side h (mm): K^e = h · K_ref, M^e = h³ · M_ref."""
    k1 = np.array([[1.0, -1.0], [-1.0, 1.0]])
    m1 = np.array([[2.0, 1.0], [1.0, 2.0]]) / 6.0

    def kron3(a0: Array, a1: Array, a2: Array) -> Array:
        # np.kron(A, B)[2 p + q] pairs row p of A with row q of B: axis 2 is the slowest
        return np.asarray(np.kron(a2, np.kron(a1, a0)))

    K = kron3(k1, m1, m1) + kron3(m1, k1, m1) + kron3(m1, m1, k1)
    M = kron3(m1, m1, m1)
    # every entry of K is a multiple of 1/36 and of M of 1/216; rounding removes the
    # rounding error of the sums (K is then exactly 0 between the ends of a cell edge)
    return np.round(K * 36.0) / 36.0, np.round(M * 216.0) / 216.0


def fem_matrices(grid: Grid) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """(K, M): the Q1 stiffness and mass matrices (n × n CSR) in mm units.

    Each cell of side h adds h · K_ref and h³ · M_ref (`q1_reference`). Explicit zeros
    are removed from K. Raises NotImplementedError if the grid has hanging nodes.

    K is assembled first, then M (`_assemble`), so the parts of only one matrix are in
    memory at a time.
    """
    if len(grid.hanging[0]):
        raise NotImplementedError("hanging nodes are not in PR 3")
    k_ref, m_ref = q1_reference()
    h = np.asarray(grid.cell_size_mm, np.float64)
    K = _assemble(grid, h, k_ref)
    K.eliminate_zeros()
    M = _assemble(grid, h**3, m_ref)
    return K, M


def _assemble(grid: Grid, scale: Array, ref: Array) -> sp.csr_matrix:
    """Σ over the cells of scale_c · ref on the cell's corners (n × n CSR, duplicates
    summed, indices sorted). The cells go in chunks of CHUNK_CELLS. Each chunk becomes
    a CSR matrix (its duplicate entries summed). The chunk matrices are then added in
    pairs, level by level (a fixed order), and each part is released when it has been
    added. So the parts take about the size of the result, not the 64 COO entries of
    each cell."""
    n = grid.n_nodes
    cells = grid.cells
    r = ref.ravel()
    itype = np.int32 if n < 2**31 - 1 else np.int64
    parts: list[sp.csr_matrix | None] = []
    for s in range(0, len(cells), CHUNK_CELLS):
        c = cells[s : s + CHUNK_CELLS].astype(itype)
        rows = np.repeat(c, 8, axis=1).ravel()  # entry a·8 + b → corner a
        cols = np.tile(c, (1, 8)).ravel()  # entry a·8 + b → corner b
        vals = (scale[s : s + CHUNK_CELLS, None] * r[None, :]).ravel()
        parts.append(sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr())
        del rows, cols, vals
    if not parts:
        return sp.csr_matrix((n, n))
    while len(parts) > 1:
        nxt: list[sp.csr_matrix | None] = []
        for i in range(0, len(parts) - 1, 2):
            a, b = parts[i], parts[i + 1]
            assert a is not None and b is not None
            parts[i] = parts[i + 1] = None
            nxt.append(a + b)
            del a, b
        if len(parts) % 2:
            nxt.append(parts[-1])
        parts = nxt
    last = parts.pop()
    assert last is not None
    out = sp.csr_matrix(last)
    out.sum_duplicates()
    out.sort_indices()
    return out


def precision_from(
    K: sp.csr_matrix, M: sp.csr_matrix, length_mm: float, sd: float
) -> sp.csr_matrix:
    """λ (K + M / ℓ²) with λ = 1 / (sd² · ℓ), ℓ = length_mm (CSR, n × n), from the
    matrices of `fem_matrices`."""
    lam = 1.0 / (sd * sd * length_mm)
    Q = (K + M * (1.0 / (length_mm * length_mm))) * lam
    return sp.csr_matrix(Q)


def fem_precision(grid: Grid, length_mm: float, sd: float) -> sp.csr_matrix:
    """λ (K + M / ℓ²) with λ = 1 / (sd² · ℓ), ℓ = length_mm (CSR, n × n)."""
    K, M = fem_matrices(grid)
    return precision_from(K, M, length_mm, sd)


def node_weights_from_mass(M: sp.csr_matrix) -> Array:
    """(n,) the row sums of the mass matrix M divided by VOXEL_MM³."""
    return np.asarray(M.sum(axis=1)).ravel() / VOXEL_MM**3


def node_weights_vox(grid: Grid) -> Array:
    """(n,) the row sums of M divided by VOXEL_MM³: the volume of each node, in
    voxels."""
    _, M = fem_matrices(grid)
    return node_weights_from_mass(M)


def _cell_edge_pairs() -> list[tuple[int, int]]:
    """The 12 edges of a cell: pairs of corner indices (a, b), b = a with one bit
    set."""
    return [(a, a | (1 << ax)) for ax in range(3) for a in range(8) if not a >> ax & 1]


def edge_weights_vox(grid: Grid) -> Array:
    """(E,) for each edge of grid.edges: the sum over the cells that hold it of
    (cell volume / 12) / VOXEL_MM³. Raises ValueError if a cell edge is not in
    grid.edges."""
    e = grid.edges
    n = np.int64(grid.n_nodes)
    key = e[:, 0] * n + e[:, 1]
    cells = grid.cells
    vol = np.asarray(grid.cell_size_mm, np.float64) ** 3 / 12.0 / VOXEL_MM**3
    w = np.zeros(len(e))
    for a, b in _cell_edge_pairs():
        lo = np.minimum(cells[:, a], cells[:, b])
        hi = np.maximum(cells[:, a], cells[:, b])
        k = lo * n + hi
        pos = np.searchsorted(key, k)
        pos_c = np.minimum(pos, len(key) - 1)
        if np.any(key[pos_c] != k):
            raise ValueError("a cell edge is not in grid.edges")
        w += np.bincount(pos_c, weights=vol, minlength=len(e))
    return w


def face_weights_vox(grid: Grid) -> Array:
    """(E,) (edge length / VOXEL_MM)²: the area that an edge stands for, in voxel
    faces."""
    length = np.linalg.norm(edge_vectors_mm(grid), axis=1)
    return np.asarray((length / VOXEL_MM) ** 2)
