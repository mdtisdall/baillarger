"""The octree of the flow-field solver (docs/plans/flow-field-phantoms.md, *The octree*;
PR 5, task 5.1).

`OctreeGrid` implements the `ff_types.Grid` protocol on a 2:1 balanced octree over a
phantom's box. The mean field (`ff_meanfield.solve`) runs on it unchanged, except for
one line in `ff_meanfield.grid_arrays` (coordinator decision CD-10): a grid that has a
`fem_matrices` method gives its own K and M.

Integer coordinates
-------------------

Every position in the octree is an integer number of *units*. One unit is the side of
the finest cell (settings.octree_finest_mm, 0.02 mm = 1/8 voxel). The coarsest cells
(level 0, settings.octree_coarsest_mm, 0.64 mm) have a side of 2^L units, with
L = log2(coarsest / finest) = 5. A cell of level l has a side of 2^(L − l) units. The
level-0 cells tile the same box as `ff_grid.UniformGrid(shape, coarsest)`: the box of
the nodes starts at voxel 0 and has (d_a − 1) level-0 cells along axis a, with
d_a = ceil((shape_a − 1) / h0_vox) + 1.

A leaf is (level, anchor): the anchor is its corner 0 in units. The leaves are kept in
Morton order of their anchors (a unique key, because leaves do not overlap).

Hanging nodes and the mean field (CD-10)
----------------------------------------

A node that lies on the middle of an edge, or the centre of a face, of a larger leaf
is a hanging node. `hanging` gives its parents (the 2 ends of the edge, weight 1/2,
or the 4 corners of the face, weight 1/4). A parent can itself be hanging (a chain);
`ff_octree_fem` resolves the chains. The mean field gives every node, also a hanging
node, its own state (the `ff_types.Grid` docstring says that a hanging node has no
free state: the plan change is reported). So `fem_matrices` is the non-conforming
assembly (Q1 on each leaf's 8 corners). The conforming form (hanging nodes
constrained) is `ff_octree_fem.conforming_matrices`.

Edges are split at hanging nodes: the edge of a large leaf whose middle is a node
gives two edges. So every edge is between two nodes with no node between them, and
`ff_grid.axis_neighbours` stays valid.

Colours: a greedy colouring over every clique of interacting nodes (the 8 corners of
each leaf, and each node with its 3 axis neighbours of each octant), in node order. So
no two nodes of one colour share a term. The count can be above 8; each run records
it.

σ_c: `spacing_mm` is the coarsest spacing (the same for every octree of one box), so
the consistency weight σ_c = Δτ / spacing does not change when one cell is refined,
and the free energies of two octrees differ only by the cells that changed. It is a
provisional decision: a per-node σ_c would need a change to ff_terms.

Edge weights: `ff_grid.edge_weights_vox` needs every leaf edge in `edges`, which a
split edge is not. So the octree gives its own `edge_weights_vox` (each half of a split
leaf edge takes half of that leaf's share), and `ff_meanfield.grid_arrays` uses it as
it uses `fem_matrices`.

Adaptation (task 5.1)
---------------------

`indicators` (per leaf, mm), `mark` (one pass, in Morton order), `adapt` (the rounds and
the acceptance by free energy), `transfer` (the distributions to another octree, for
the record and the tests) and `hanging_step` (the step of the field at the hanging
nodes). Each docstring gives the exact rule.

NumPy or Torch (the plan's *NumPy or Torch*)
--------------------------------------------

Symbols: C leaves, n nodes (n is about 1.4 C on the octrees measured here), L =
max_level + 1 = 6 levels. Phantom size: up to about 10⁶ nodes; target size: 4·10⁶ to
4·10⁷ nodes.

- Construction (`refine`, `_split`, `balanced`, `coarsen`). `_split` is whole-array
  work, O(C). `balanced` repeats `unbalanced` until no leaf is too coarse: each pass
  is 26 calls of `find_leaf` on C points. `find_leaf` is one `searchsorted` per level
  on the C sorted keys: O(L · N log C) for N points. So one balance pass is about
  26 · L · C · log C key comparisons; the passes are at most L. `coarsen` has a Python
  loop over the nominated parents (`_merge`).
- Nodes, hanging nodes, edges. Whole-array work on the (C, 8, 3) corners and
  `np.unique` sorts: O(C log C). `edges` uses `np.unique(axis=0)` on about 12 C rows,
  which is slow (row sort by lexicographic compare).
- Colouring. A greedy colouring in node order: one Python loop over the n nodes (by
  design sequential: each node reads its neighbours' colours). The adjacency is built
  from the cliques as int64 COO lists (about 56 entries per leaf, 2 per edge, 96 per
  node from the 8 octants), then CSR. A vectorised alternative (for example a fixed
  colour from the level and the parity of the node index, or a parallel colouring
  such as Jones–Plassmann) is a finding, not done here.
- Indicators and marking. (C, 8, 8) trilinear weights and gathers of node values:
  about 8 · 8 · 10 flops per leaf, regular and data parallel, O(C) memory traffic
  (about 1 kB per leaf in float64 for the weights and the gathered values). The
  marking is a median and an `np.unique` of the sibling keys: O(C log C).
- Transfer. Copies of the kept nodes, and for each new node a gather of 8 corners ×
  prune_max slots and an argsort of 8 · prune_max masses: O(n_new · 256 log 256).

Measured (laptop CPU, one run, NumPy): a box of 96 × 96 × 128 voxels, refined by 3
rounds of random nominations to 1.11·10⁶ leaves, 1.58·10⁶ nodes, 7.9·10⁵ hanging
nodes, 17 colours. The 3rd refinement (with its balance) 13.4 s; cells 0.24 s; hanging
1.4 s; edges 12.2 s; colouring 18.2 s (about 11.5 µs per node); edge_weights_vox 1.9 s;
find_leaf of 10⁶ points 0.15 s; one `unbalanced` pass 4.7 s; indicators 3.3 s; mark
1.4 s; transfer to an octree with 1.8·10⁴ new nodes 1.4 s. The peak RSS of the
process was 11.0 GiB (about 7 kB per node; the step that sets the peak was not
identified). By linear extrapolation (an estimate): at 4·10⁷ nodes the colouring
loop takes about 8 minutes, and the construction as written needs about 280 GB. So
the construction, the edges and the colouring must be rewritten in chunks (or with a
fixed colouring) before the target size; the indicators and the transfer scale.

Choice: NumPy. The construction is integer sorting and searching on irregular keys
(torch.searchsorted could run `find_leaf` on a GPU, but the construction runs once
per round); the colouring is sequential; the indicators and the transfer are already
whole-array NumPy and run once per round, a small cost next to one solve. The cost at
the target size is set by memory, not by the arithmetic, and Torch does not change
that.
"""

from __future__ import annotations

import dataclasses
import math
import sys
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
import ff_grid as FG  # noqa: E402
import ff_types as T  # noqa: E402

Array = T.Array
IntArray = T.IntArray
BoolArray = T.BoolArray

# corner c of a cell is at anchor + size · CORNER_BITS[c] (order i + 2 j + 4 k)
CORNER_BITS = np.array([[(c >> a) & 1 for a in range(3)] for c in range(8)], np.int64)
# the 12 edges of a cell as corner pairs (a, b), b = a with one bit set
EDGE_PAIRS = np.array(
    [(a, a | (1 << ax)) for ax in range(3) for a in range(8) if not (a >> ax) & 1],
    np.int64,
)
# the 6 faces of a cell: (axis, side) and their 4 corners
FACE_CORNERS = np.array(
    [
        [c for c in range(8) if ((c >> ax) & 1) == side]
        for ax in range(3)
        for side in (0, 1)
    ],
    np.int64,
)
# the 26 neighbour offsets in {−1, 0, 1}³ \ 0
NEIGHBOUR_OFFSETS = np.array(
    [
        (i, j, k)
        for i in (-1, 0, 1)
        for j in (-1, 0, 1)
        for k in (-1, 0, 1)
        if (i, j, k) != (0, 0, 0)
    ],
    np.int64,
)


def morton_key(anchor: IntArray) -> IntArray:
    """(N,) the Morton key of integer points (N, 3): bit b of axis a at 3 b + a."""
    a = np.asarray(anchor, np.int64)
    key = np.zeros(len(a), np.int64)
    for b in range(21):
        for ax in range(3):
            key |= ((a[:, ax] >> b) & 1) << (3 * b + ax)
    return key


class OctreeGrid:
    """A 2:1 balanced octree with nodes at the leaf corners (module docstring).

    shape_vox: the phantom's box (voxels). level (C,), anchor (C, 3): the leaves, in
    units. The leaves must tile the box of the level-0 cells. They are sorted into
    Morton order here. Use `OctreeGrid.base`, `refine`, `coarsen` and `balanced` to
    make octrees; the constructor checks that the leaves tile the box and that the tree
    is balanced (ValueError otherwise).
    """

    kind: Literal["uniform", "octree"]

    def __init__(
        self,
        shape_vox: tuple[int, int, int],
        level: IntArray,
        anchor: IntArray,
        settings: T.SolverSettings = T.DEFAULTS,
        *,
        check: bool = True,
    ) -> None:
        self.kind = "octree"
        self.settings = settings
        self.shape_vox = (int(shape_vox[0]), int(shape_vox[1]), int(shape_vox[2]))
        self.coarsest_mm = float(settings.octree_coarsest_mm)
        self.finest_mm = float(settings.octree_finest_mm)
        ratio = self.coarsest_mm / self.finest_mm
        self.max_level = int(round(math.log2(ratio)))
        if abs(2.0**self.max_level - ratio) > 1e-9 * ratio:
            raise ValueError("coarsest / finest must be a power of 2")
        self.unit_vox = self.finest_mm / T.VOXEL_MM  # voxels per unit
        self.h0_units = 1 << self.max_level
        h0_vox = self.coarsest_mm / T.VOXEL_MM
        d = [math.ceil(round((s - 1) / h0_vox, 9)) + 1 for s in self.shape_vox]
        self.base_dims = (d[0] - 1, d[1] - 1, d[2] - 1)  # level-0 cells per axis
        self.extent = np.array(self.base_dims, np.int64) * self.h0_units  # units
        self.spacing_mm = self.coarsest_mm  # σ_c (module docstring)
        lev = np.asarray(level, np.int64).ravel()
        anc = np.asarray(anchor, np.int64).reshape(-1, 3)
        if len(lev) != len(anc):
            raise ValueError("level and anchor differ in length")
        if np.any(lev < 0) or np.any(lev > self.max_level):
            raise ValueError("a level is out of range")
        order = np.argsort(morton_key(anc), kind="stable")
        self.level = np.ascontiguousarray(lev[order])
        self.anchor = np.ascontiguousarray(anc[order])
        if check:
            self._check()

    # -------------------------------------------------- construction

    @classmethod
    def base(
        cls, shape_vox: tuple[int, int, int], settings: T.SolverSettings = T.DEFAULTS
    ) -> OctreeGrid:
        """The octree of level-0 cells only (the uniform grid at the coarsest
        spacing)."""
        z_lev, z_anc = np.zeros(0, np.int64), np.zeros((0, 3), np.int64)
        probe = cls(shape_vox, z_lev, z_anc, settings, check=False)
        idx = np.indices(probe.base_dims, dtype=np.int64).reshape(3, -1).T
        return cls(
            shape_vox, np.zeros(len(idx), np.int64), idx * probe.h0_units, settings
        )

    def _with(
        self, level: IntArray, anchor: IntArray, check: bool = True
    ) -> OctreeGrid:
        return OctreeGrid(self.shape_vox, level, anchor, self.settings, check=check)

    @property
    def size_units(self) -> IntArray:
        """(C,) the side of each leaf in units."""
        return np.asarray(np.left_shift(1, self.max_level - self.level), np.int64)

    def _split(self, ids: IntArray) -> tuple[IntArray, IntArray]:
        """The leaves after splitting `ids` into 8 children each (not balanced)."""
        ids = np.unique(np.asarray(ids, np.int64))
        if np.any(self.level[ids] >= self.max_level):
            raise ValueError("a leaf at the finest level cannot be refined")
        keep = np.ones(len(self.level), bool)
        keep[ids] = False
        half = self.size_units[ids] // 2
        ch_anchor = (
            self.anchor[ids][:, None, :] + half[:, None, None] * CORNER_BITS[None]
        ).reshape(-1, 3)
        ch_level = np.repeat(self.level[ids] + 1, 8)
        return (np.concatenate([self.level[keep], ch_level]),
                np.concatenate([self.anchor[keep], ch_anchor]))  # fmt: skip

    def refine(self, ids: IntArray) -> OctreeGrid:
        """Split the leaves `ids` into 8 children each, then restore 2:1 balance by
        refining more leaves (`balanced`)."""
        lev, anc = self._split(ids)
        return self._with(lev, anc, check=False).balanced()

    def balanced(self) -> OctreeGrid:
        """The octree with 2:1 balance across faces, edges and corners: leaves that are
        more than one level coarser than a neighbour are split, until none is."""
        t = self
        while True:
            bad = t.unbalanced()
            if len(bad) == 0:
                return t._with(t.level, t.anchor, check=True)
            lev, anc = t._split(bad)
            t = t._with(lev, anc, check=False)

    def unbalanced(self) -> IntArray:
        """(B,) sorted ids of the leaves that are more than one level coarser than one
        of their 26 neighbours."""
        size = self.size_units
        out = []
        for o in NEIGHBOUR_OFFSETS:
            p = self.anchor + np.where(o > 0, size[:, None], np.where(o < 0, -1, 0))
            nb = self.find_leaf(p)
            ok = nb >= 0
            src = np.flatnonzero(ok)
            nb = nb[ok]
            coarse = self.level[nb] < self.level[src] - 1
            out.append(nb[coarse])
        if not out:
            return np.zeros(0, np.int64)
        return np.unique(np.concatenate(out))

    def coarsen(self, parents_level: IntArray, parents_anchor: IntArray) -> OctreeGrid:
        """Merge the 8 children of each given parent (level, anchor) into the parent.
        A parent whose 8 children are not all leaves, or whose merge would break 2:1
        balance, is skipped. Returns the new octree (balanced)."""
        pl = np.asarray(parents_level, np.int64).ravel()
        pa = np.asarray(parents_anchor, np.int64).reshape(-1, 3)
        live = np.ones(len(pl), bool)
        while True:
            lev, anc, merged = self._merge(pl[live], pa[live])
            t = self._with(lev, anc, check=False)
            bad = t.unbalanced()
            ids_live = np.flatnonzero(live)
            # a merged parent that is too coarse for a neighbour is not merged
            bad_keys = set(
                zip(
                    t.level[bad].tolist(),
                    map(tuple, t.anchor[bad].tolist()),
                    strict=True,
                )
            )
            drop = [
                k
                for k, (l_, a_) in enumerate(zip(pl[live], pa[live], strict=True))
                if (int(l_), tuple(int(v) for v in a_)) in bad_keys or not merged[k]
            ]
            if not drop:
                return t._with(t.level, t.anchor, check=True)
            live[ids_live[drop]] = False

    def _merge(
        self, pl: IntArray, pa: IntArray
    ) -> tuple[IntArray, IntArray, BoolArray]:
        keep = np.ones(len(self.level), bool)
        merged = np.zeros(len(pl), bool)
        for k in range(len(pl)):
            half = 1 << (self.max_level - int(pl[k]) - 1)
            ch = pa[k][None, :] + half * CORNER_BITS
            ids = self.find_leaf(ch)
            if (
                np.all(ids >= 0)
                and np.all(self.level[ids] == pl[k] + 1)
                and len(set(ids.tolist())) == 8
            ):
                keep[ids] = False
                merged[k] = True
        lev = np.concatenate([self.level[keep], pl[merged]])
        anc = np.concatenate([self.anchor[keep], pa[merged]])
        return lev, anc, merged

    def _check(self) -> None:
        size = self.size_units
        if np.any(self.anchor % size[:, None] != 0):
            raise ValueError("a leaf is not aligned to its level")
        if np.any(self.anchor < 0) or np.any(self.anchor + size[:, None] > self.extent):
            raise ValueError("a leaf is outside the box")
        vol = int(np.sum(size.astype(object) ** 3))
        if vol != int(np.prod(self.extent.astype(object))):
            raise ValueError("the leaves do not tile the box")
        # with the volume equal, no overlap <=> every leaf's centre finds itself
        own = self.find_leaf(
            self.anchor + size[:, None] // 2 if self.max_level else self.anchor
        )
        if np.any(own != np.arange(len(size))):
            raise ValueError("the leaves overlap")
        if len(self.unbalanced()):
            raise ValueError("the octree is not 2:1 balanced")

    # -------------------------------------------------- lookup

    @cached_property
    def _lookup(self) -> tuple[IntArray, IntArray]:
        """(sorted keys, leaf ids): key = level · S³ + linear index of the anchor."""
        key = self._key(self.level, self.anchor)
        order = np.argsort(key, kind="stable")
        return key[order], order.astype(np.int64)

    def _key(self, level: IntArray, p: IntArray) -> IntArray:
        S = int(self.extent.max()) + 1
        p = np.asarray(p, np.int64)
        return (
            np.asarray(level, np.int64) * S**3 + (p[:, 0] * S + p[:, 1]) * S + p[:, 2]
        )

    def find_leaf(self, P: IntArray) -> IntArray:
        """(N,) the leaf that holds each integer point P (N, 3) (a point on a face goes
        to the leaf on its upper side: the leaf that has it in [anchor, anchor + size));
        −1 for a point outside [0, extent)."""
        P = np.asarray(P, np.int64).reshape(-1, 3)
        out = np.full(len(P), -1, np.int64)
        inside = np.all((P >= 0) & (P < self.extent), axis=1)
        keys, ids = self._lookup
        if len(keys) == 0:
            return out
        for lv in range(self.max_level + 1):
            todo = inside & (out < 0)
            if not todo.any():
                break
            s = 1 << (self.max_level - lv)
            a = (P[todo] // s) * s
            k = self._key(np.full(len(a), lv), a)
            pos = np.minimum(np.searchsorted(keys, k), len(keys) - 1)
            hit = keys[pos] == k
            sel = np.flatnonzero(todo)
            out[sel[hit]] = ids[pos[hit]]
        return out

    # -------------------------------------------------- nodes and cells

    @cached_property
    def _corner_points(self) -> IntArray:
        """(C, 8, 3) the corners of every leaf, in units."""
        return np.asarray(
            self.anchor[:, None, :] + self.size_units[:, None, None] * CORNER_BITS[None]
        )

    def _node_key(self, p: IntArray) -> IntArray:
        S = int(self.extent.max()) + 1
        p = np.asarray(p, np.int64)
        return np.asarray((p[..., 0] * S + p[..., 1]) * S + p[..., 2])

    @cached_property
    def _node_keys(self) -> IntArray:
        """(n,) sorted keys of the node points (C order of the coordinates)."""
        return np.unique(self._node_key(self._corner_points).ravel())

    def node_id(self, p: IntArray) -> IntArray:
        """(…) the node id of integer points p (…, 3); −1 where p is not a node."""
        k = self._node_key(p)
        keys = self._node_keys
        pos = np.minimum(np.searchsorted(keys, k), len(keys) - 1)
        return np.where(keys[pos] == k, pos, -1).astype(np.int64)

    @cached_property
    def node_units(self) -> IntArray:
        """(n, 3) the integer coordinates of the nodes."""
        S = int(self.extent.max()) + 1
        k = self._node_keys
        return np.stack([k // (S * S), (k // S) % S, k % S], axis=1).astype(np.int64)

    @property
    def n_nodes(self) -> int:
        return len(self._node_keys)

    @property
    def n_cells(self) -> int:
        return len(self.level)

    @cached_property
    def nodes(self) -> Array:
        """(n, 3) node positions, voxel space."""
        return self.node_units.astype(np.float64) * self.unit_vox

    @cached_property
    def cells(self) -> IntArray:
        """(C, 8) node ids of each leaf's corners, corner order i + 2 j + 4 k."""
        return self.node_id(self._corner_points)

    @cached_property
    def node_level(self) -> IntArray:
        """(n,) the level of the finest leaf that touches the node."""
        out = np.zeros(self.n_nodes, np.int64)
        np.maximum.at(out, self.cells.ravel(), np.repeat(self.level, 8))
        return out

    @cached_property
    def cell_origin(self) -> Array:
        return self.anchor.astype(np.float64) * self.unit_vox

    @cached_property
    def cell_size_mm(self) -> Array:
        return self.size_units.astype(np.float64) * self.finest_mm

    @cached_property
    def prior_shaped(self) -> BoolArray:
        """(C,) True for a leaf finer than the image (side <
        settings.prior_shaped_below_mm)."""
        return np.asarray(
            self.cell_size_mm < self.settings.prior_shaped_below_mm * (1.0 - 1e-9),
            np.bool_,
        )

    # -------------------------------------------------- hanging nodes and edges

    @cached_property
    def hanging(self) -> tuple[IntArray, IntArray, Array]:
        """(ids (H,), parents (H, 4), weights (H, 4)): the nodes on the middle of an
        edge (2 parents, 1/2 each) or the centre of a face (4 parents, 1/4 each) of a
        larger leaf. Sorted by id. A parent can be hanging itself."""
        big = np.flatnonzero(self.size_units >= 2)
        ids_l: list[IntArray] = []
        par_l: list[IntArray] = []
        w_l: list[Array] = []
        cp = self._corner_points[big]
        cn = self.cells[big]
        for a, b in EDGE_PAIRS:
            mid = (cp[:, a] + cp[:, b]) // 2
            nid = self.node_id(mid)
            hit = nid >= 0
            ids_l.append(nid[hit])
            p = np.full((int(hit.sum()), 4), 0, np.int64)
            p[:, 0], p[:, 1] = cn[hit, a], cn[hit, b]
            par_l.append(p)
            w_l.append(np.tile([0.5, 0.5, 0.0, 0.0], (int(hit.sum()), 1)))
        for fc in FACE_CORNERS:
            ctr = (cp[:, fc[0]] + cp[:, fc[3]]) // 2
            nid = self.node_id(ctr)
            hit = nid >= 0
            ids_l.append(nid[hit])
            par_l.append(cn[hit][:, fc])
            w_l.append(np.full((int(hit.sum()), 4), 0.25))
        if not ids_l:
            return np.zeros(0, np.int64), np.zeros((0, 4), np.int64), np.zeros((0, 4))
        ids = np.concatenate(ids_l)
        par = np.concatenate(par_l)
        w = np.concatenate(w_l)
        ids, first = np.unique(ids, return_index=True)
        return ids, par[first], w[first]

    @cached_property
    def edges(self) -> IntArray:
        """(E, 2) node ids of each leaf edge, split at its middle where the middle is a
        node; i < j, each edge once, sorted."""
        cp = self._corner_points
        cn = self.cells
        parts = []
        for a, b in EDGE_PAIRS:
            mid = (cp[:, a] + cp[:, b]) // 2
            nid = np.where(self.size_units >= 2, self.node_id(mid), -1)
            split = nid >= 0
            parts.append(np.stack([cn[~split, a], cn[~split, b]], 1))
            parts.append(np.stack([cn[split, a], nid[split]], 1))
            parts.append(np.stack([nid[split], cn[split, b]], 1))
        e = np.concatenate(parts)
        e = np.sort(e, axis=1)
        e = np.unique(e, axis=0)
        return np.ascontiguousarray(e)

    # -------------------------------------------------- colours

    @cached_property
    def _colouring(self) -> IntArray:
        """Greedy colouring in node order over the cliques: the 8 corners of each
        leaf, the two ends of each edge, and each node with its axis neighbours of one
        octant."""
        n = self.n_nodes
        rows: list[IntArray] = []
        cols: list[IntArray] = []

        def clique(c: IntArray) -> None:
            k = c.shape[1]
            for i in range(k):
                for j in range(k):
                    if i != j:
                        ok = (c[:, i] >= 0) & (c[:, j] >= 0)
                        rows.append(c[ok, i])
                        cols.append(c[ok, j])

        clique(self.cells)
        clique(self.edges)
        nb = FG.axis_neighbours(self)
        for sx in (0, 1):
            for sy in (0, 1):
                for sz in (0, 1):
                    c = np.stack(
                        [np.arange(n), nb[:, sx], nb[:, 2 + sy], nb[:, 4 + sz]], 1
                    )
                    clique(c)
        ones = np.ones(sum(len(r) for r in rows), np.int8)
        rc = (np.concatenate(rows), np.concatenate(cols))
        A = sp.csr_matrix((ones, rc), shape=(n, n))
        A.sum_duplicates()
        colour = np.full(n, -1, np.int64)
        ip, ix = A.indptr, A.indices
        for i in range(n):
            used = colour[ix[ip[i] : ip[i + 1]]]
            used = used[used >= 0]
            c = 0
            if len(used):
                taken = np.zeros(int(used.max()) + 2, bool)
                taken[used] = True
                c = int(np.argmin(taken))
            colour[i] = c
        return colour

    @property
    def colour(self) -> IntArray:
        return self._colouring

    @property
    def n_colours(self) -> int:
        return int(self._colouring.max()) + 1 if self.n_nodes else 0

    # -------------------------------------------------- queries

    def locate(self, X: Array) -> tuple[IntArray, Array]:
        """(leaf id (N,), local coordinates (N, 3) in [0, 1]) of voxel positions X. −1
        and 0 outside the box of the nodes. On a face between leaves: the leaf on the
        upper side; on the last node plane: the last leaf (local 1)."""
        t = np.asarray(X, np.float64).reshape(-1, 3) / self.unit_vox
        top = self.extent.astype(np.float64)
        inside = np.all((t >= 0.0) & (t <= top), axis=1)
        p = np.clip(np.floor(t), 0, top - 1).astype(np.int64)
        cid = self.find_leaf(p)
        cid = np.where(inside, cid, -1)
        safe = np.maximum(cid, 0)
        local = (t - self.anchor[safe]) / self.size_units[safe][:, None]
        local = np.where(inside[:, None], np.clip(local, 0.0, 1.0), 0.0)
        return cid.astype(np.int64), np.asarray(local, np.float64)

    def morton(self) -> IntArray:
        """(C,) the Morton key of each leaf's anchor (the leaves are stored in this
        order)."""
        return morton_key(self.anchor)

    def edge_weights_vox(self) -> Array:
        """(E,) the volume weight of each edge of `edges`, in voxels (as
        `ff_grid.edge_weights_vox`, which needs every leaf edge in `edges`): the sum
        over the leaves that hold the edge of (leaf volume / 12) / VOXEL_MM³. Where a
        leaf edge is split at its middle node, each half gets half of that leaf's
        share."""
        e = self.edges
        n = np.int64(self.n_nodes)
        key = e[:, 0] * n + e[:, 1]
        cp = self._corner_points
        cn = self.cells
        vol = self.cell_size_mm**3 / 12.0 / T.VOXEL_MM**3
        w = np.zeros(len(e))

        def add(a: IntArray, b: IntArray, v: Array) -> None:
            k = np.minimum(a, b) * n + np.maximum(a, b)
            pos = np.minimum(np.searchsorted(key, k), len(key) - 1)
            if np.any(key[pos] != k):
                raise ValueError("a leaf edge is not in edges")
            w[:] += np.bincount(pos, weights=v, minlength=len(e))

        for a, b in EDGE_PAIRS:
            mid = (cp[:, a] + cp[:, b]) // 2
            nid = np.where(self.size_units >= 2, self.node_id(mid), -1)
            split = nid >= 0
            add(cn[~split, a], cn[~split, b], vol[~split])
            add(cn[split, a], nid[split], 0.5 * vol[split])
            add(nid[split], cn[split, b], 0.5 * vol[split])
        return w

    def fem_matrices(self) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """(K, M) for the mean field: the non-conforming Q1 assembly on each leaf's 8
        corners (CD-10; `ff_octree_fem.nonconforming_matrices`)."""
        import ff_octree_fem as FOF

        return FOF.nonconforming_matrices(self)

    def level_counts(self) -> dict[str, Any]:
        """Leaves and nodes per level, the hanging nodes, the colours and the
        prior-shaped leaves (for run.json)."""
        lk, lc = np.unique(self.level, return_counts=True)
        nk, nc = np.unique(self.node_level, return_counts=True)
        lv = {int(k): int(v) for k, v in zip(lk, lc, strict=True)}
        nl = {int(k): int(v) for k, v in zip(nk, nc, strict=True)}
        return dict(
            leaves_per_level=lv,
            nodes_per_level=nl,
            cell_mm_per_level={int(k): self.coarsest_mm / 2**k for k in lv},
            n_nodes=self.n_nodes,
            n_cells=self.n_cells,
            n_hanging=int(len(self.hanging[0])),
            n_colours=self.n_colours,
            n_prior_shaped=int(self.prior_shaped.sum()),
        )

    def arrays(self) -> dict[str, npt.NDArray[Any]]:
        """The arrays that define the grid, keys prefixed "grid_"."""
        h_ids, h_par, h_w = self.hanging
        return {
            "grid_kind": np.array(self.kind),
            "grid_shape_vox": np.array(self.shape_vox, np.int64),
            "grid_coarsest_mm": np.array(self.coarsest_mm),
            "grid_finest_mm": np.array(self.finest_mm),
            "grid_unit_vox": np.array(self.unit_vox),
            "grid_level": self.level,
            "grid_anchor": self.anchor,
            "grid_node_level": self.node_level,
            "grid_cells": self.cells,
            "grid_cell_origin": self.cell_origin,
            "grid_cell_size_mm": self.cell_size_mm,
            "grid_prior_shaped": self.prior_shaped,
            "grid_hanging_ids": h_ids,
            "grid_hanging_parents": h_par,
            "grid_hanging_weights": h_w,
            "grid_colour": self.colour,
        }


# ---------------------------------------------------------------- the indicators

INDICATOR_PARTS = ("tau", "direction", "stop", "alignment")


def parent_box(grid: OctreeGrid) -> tuple[IntArray, IntArray]:
    """(lo (C, 3), hi (C, 3)) in units: the parent cell of each leaf. The parent of a
    leaf of side s is the aligned cell of side 2 s that holds it. At level 0 it is the
    aligned virtual cell of side 2 h0, clipped to the box (so hi − lo is s or 2 s on
    each axis)."""
    s = grid.size_units[:, None]
    lo = (grid.anchor // (2 * s)) * (2 * s)
    hi = np.minimum(lo + 2 * s, grid.extent[None, :])
    return np.asarray(lo), np.asarray(hi)


def _parent_interpolation(grid: OctreeGrid) -> tuple[IntArray, Array]:
    """(parent corner node ids (C, 8), weights (C, 8, 8)): row [c, k] gives the weights
    of the parent's 8 corners at the leaf's corner k. Every parent corner is a node: the
    parent's octants are covered by leaves of the leaf's level or finer (a coarser leaf
    would hold the leaf), and the clipped virtual parent's corners are level-0 nodes."""
    lo, hi = parent_box(grid)
    Q = np.where(CORNER_BITS[None, :, :] == 1, hi[:, None, :], lo[:, None, :])
    pid = grid.node_id(Q)
    if np.any(pid < 0):
        raise AssertionError("a parent corner is not a node")
    P = grid._corner_points
    t = (P - lo[:, None, :]) / (hi - lo)[:, None, :].astype(np.float64)
    W = T.trilinear_weights(t.reshape(-1, 3)).reshape(len(lo), 8, 8)
    return pid, W


def indicators(
    grid: OctreeGrid,
    field: T.Field,
    speed: Array,
    tensor_dir: Array,
    coherence: Array,
) -> dict[str, Array]:
    """The refinement indicators of each leaf, in mm (the plan's *The octree*,
    *Indicators*): how far the columns or the stops would move if the leaf were not
    there. Returns the parts "tau", "direction", "stop", "alignment" and their sum
    "total", each (C,).

    Definitions. For leaf L of side h (mm) with corners k = 0..7 (nodes i_k), the
    parent is `parent_box` and the parent's corners are nodes j_0..j_7 with the
    trilinear weights W_k of corner k in the parent. m_i = 1 − q_stop_i is the moving
    probability of node i. τ_i is the mean τ (conditional on moving), d_i the
    posterior-mean direction, v_i = speed_i (mm per unit τ, exp(u)).

    - The parent's value at corner k is the masked interpolation of `ff_meanfield.
      initial_from_field` (the value that a merge would give): for τ and d, the weights
      W_k · m_j, renormalised; d then normalised. Where the masked weights sum to 0 the
      surplus is 0. For q_stop the plain weights W_k.
    - tau = max_k m_ik · v_ik · |τ_ik − τ̂_k|.
    - direction = max_k m_ik · angle(d_ik, d̂_k) · h (0 where d_ik or d̂_k is zero).
    - stop = (max_k m_ik) · max_k |q_stop,ik − q̂_stop,k| · h: the move of the stop
      level inside the leaf, if q_stop changes by 1 across one leaf side.
    - alignment = max_k m_ik · coherence_ik · arccos|d_ik · e_ik| · h, with e the
      tensor's eigenvector (sign-free): the residual angle of the alignment term.
    - total = tau + direction + stop + alignment.

    The weight m (the moving probability) removes errors where no column passes. The
    contrast residual is not in the total: it needs the per-node contrast energy,
    which `ff_terms` does not give (a change to ff_terms, not in this task). A corner
    shared with the parent has surplus 0 (W_k is then a unit vector)."""
    pid, W = _parent_interpolation(grid)
    cells = grid.cells
    m = np.asarray(field.p_moving, np.float64)
    tau = np.nan_to_num(field.tau_mean())
    d = field.direction_mean()
    qs = np.asarray(field.q_stop, np.float64)
    h = grid.cell_size_mm

    Wm = W * m[pid][:, None, :]
    sw = Wm.sum(2)
    has = sw > 0
    tau_hat = np.where(
        has, np.einsum("ckj,cj->ck", Wm, tau[pid]) / np.maximum(sw, 1e-300), 0.0
    )
    D = np.einsum("ckj,cjx->ckx", Wm, d[pid])
    nr = np.linalg.norm(D, axis=2)
    dn = D / np.maximum(nr, 1e-300)[..., None]
    q_hat = np.einsum("ckj,cj->ck", W, qs[pid])

    mi = m[cells]
    tau_part = np.where(has, mi * speed[cells] * np.abs(tau[cells] - tau_hat), 0.0).max(
        1
    )
    di = d[cells]
    ok = (nr > 0) & (np.linalg.norm(di, axis=2) > 0)
    ang = np.arccos(np.clip(np.einsum("ckx,ckx->ck", di, dn), -1.0, 1.0))
    dir_part = np.where(ok, mi * ang, 0.0).max(1) * h
    stop_part = mi.max(1) * np.abs(qs[cells] - q_hat).max(1) * h
    e = tensor_dir[cells]
    al = np.arccos(np.clip(np.abs(np.einsum("ckx,ckx->ck", di, e)), 0.0, 1.0))
    al = np.where(np.linalg.norm(di, axis=2) > 0, al, 0.0)
    align_part = (mi * coherence[cells] * al).max(1) * h
    out = dict(tau=tau_part, direction=dir_part, stop=stop_part, alignment=align_part)
    out["total"] = tau_part + dir_part + stop_part + align_part
    return {k: np.asarray(v, np.float64) for k, v in out.items()}


def indicators_of_solve(
    grid: OctreeGrid, field: T.Field, extras: dict[str, Any], inputs: T.Inputs
) -> dict[str, Array]:
    """`indicators` with the speed of a solve (extras["speed"]) and the tensor at the
    nodes (`ff_meanfield.tensor_at_nodes`)."""
    import ff_meanfield as MF

    tdir, coh, _ = MF.tensor_at_nodes(inputs.tensor, grid.nodes)
    return indicators(grid, field, np.asarray(extras["speed"], np.float64), tdir, coh)


# ---------------------------------------------------------------- the marking


def mark_leaves(
    level: IntArray,
    anchor: IntArray,
    total: Array,
    max_level: int,
    settings: T.SolverSettings = T.DEFAULTS,
) -> dict[str, Any]:
    """The nominations of one marking pass (the plan's *Marking*). The leaves can come
    in any order: the result depends only on the set of (level, anchor, total).

    med = the median of total over all leaves. Refinement: the leaves with
    total > settings.refine_factor · med and level < max_level. Coarsening: each group
    of 8 leaves with the same parent (level ≥ 1, all 8 octants are leaves) whose 8
    totals are all < settings.coarsen_factor · med. The thresholds only nominate.

    Returns refine_level (R,), refine_anchor (R, 3), coarsen_level (P,) and
    coarsen_anchor (P, 3) (the parents), each in Morton order of the anchor (then by
    level), and median, refine_threshold, coarsen_threshold."""
    lev = np.asarray(level, np.int64).ravel()
    anc = np.asarray(anchor, np.int64).reshape(-1, 3)
    tot = np.asarray(total, np.float64).ravel()
    med = float(np.median(tot)) if len(tot) else 0.0
    up = settings.refine_factor * med
    down = settings.coarsen_factor * med
    ref = (tot > up) & (lev < max_level)
    r_l, r_a = _morton_sorted(lev[ref], anc[ref])
    # sibling groups: key = (parent level, parent anchor)
    fine = lev >= 1
    pl = lev[fine] - 1
    side = np.left_shift(1, max_level - pl)
    pa = (anc[fine] // side[:, None]) * side[:, None]
    small = tot[fine] < down
    if len(pl):
        keys = np.concatenate([pl[:, None], pa], 1)
        uk, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        n_small = np.bincount(inv, weights=small.astype(np.float64), minlength=len(uk))
        ok = (cnt == 8) & (n_small == 8)
        c_l, c_a = _morton_sorted(uk[ok, 0], uk[ok, 1:])
    else:
        c_l, c_a = np.zeros(0, np.int64), np.zeros((0, 3), np.int64)
    return dict(
        refine_level=r_l,
        refine_anchor=r_a,
        coarsen_level=c_l,
        coarsen_anchor=c_a,
        median=med,
        refine_threshold=up,
        coarsen_threshold=down,
    )


def _morton_sorted(level: IntArray, anchor: IntArray) -> tuple[IntArray, IntArray]:
    lev = np.asarray(level, np.int64).ravel()
    anc = np.asarray(anchor, np.int64).reshape(-1, 3)
    order = np.lexsort((lev, morton_key(anc)))
    return np.ascontiguousarray(lev[order]), np.ascontiguousarray(anc[order])


def mark(
    grid: OctreeGrid, total: Array, settings: T.SolverSettings | None = None
) -> dict[str, Any]:
    """`mark_leaves` on the leaves of a grid (stored in Morton order, so the pass goes
    over them in Morton order)."""
    s = grid.settings if settings is None else settings
    return mark_leaves(grid.level, grid.anchor, total, grid.max_level, s)


def leaf_ids(grid: OctreeGrid, level: IntArray, anchor: IntArray) -> IntArray:
    """(N,) the ids of the leaves (level, anchor) of the grid; ValueError if one is not
    a leaf."""
    ids = grid.find_leaf(np.asarray(anchor, np.int64).reshape(-1, 3))
    lev = np.asarray(level, np.int64).ravel()
    if np.any(ids < 0) or np.any(grid.level[np.maximum(ids, 0)] != lev):
        raise ValueError("a nominated cell is not a leaf")
    if np.any(np.any(grid.anchor[ids] != np.asarray(anchor).reshape(-1, 3), axis=1)):
        raise ValueError("a nominated cell is not a leaf")
    return ids


def apply_marks(
    grid: OctreeGrid, marks: dict[str, Any]
) -> tuple[OctreeGrid, OctreeGrid]:
    """(refined, coarsened): the grid with all nominated refinements (balanced), and the
    grid with all nominated coarsenings (`OctreeGrid.coarsen` skips a parent whose merge
    breaks the balance). Each is the grid itself where nothing is nominated."""
    refined = grid
    if len(marks["refine_level"]):
        refined = grid.refine(
            leaf_ids(grid, marks["refine_level"], marks["refine_anchor"])
        )
    coarsened = grid
    if len(marks["coarsen_level"]):
        coarsened = grid.coarsen(marks["coarsen_level"], marks["coarsen_anchor"])
    return refined, coarsened


def same_tree(a: OctreeGrid, b: OctreeGrid) -> bool:
    """True if the two octrees have the same box and the same leaves."""
    return (
        a.shape_vox == b.shape_vox
        and np.array_equal(a.level, b.level)
        and np.array_equal(a.anchor, b.anchor)
    )


# ---------------------------------------------------------------- the transfer


def transfer(field: T.Field, new_grid: OctreeGrid) -> T.Field:
    """The field moved to another octree of the same box (column-solver.md, *Moving the
    distributions*), for the record and the tests.

    - A node of new_grid that is also a node of field.grid keeps its slots exactly
      (τ, direction, q, q_stop, n_kept).
    - Another node (inside or on the boundary of an old leaf that was refined) takes
      the trilinear mixture of the distributions of the 8 corners of the old leaf that
      holds it (`OctreeGrid.find_leaf`: on a face, the leaf on the upper side; at the
      box's upper end, the last leaf). Each corner candidate is one candidate of the
      mixture, with mass w_c · q_c,k (two equal candidates of two corners stay two
      slots). The prune_max candidates of largest mass are kept (ties: the corner
      order, then the slot), in that order. q_stop = Σ_c w_c q_stop,c. The kept
      moving masses are scaled to sum to 1 − q_stop; where no moving mass is kept,
      q_stop = 1. So the total probability is 1 at every node.

    On coarsening every node of the new octree is a node of the old one, so every node
    keeps its own distribution; the distributions of the removed nodes are not merged
    into the parent's corners (provisional).

    Limit: `ff_meanfield.solve(init=...)` reads only the means of the init field
    (q_stop, mean τ and mean direction, `initial_from_field`), not these distributions.
    So the trials of `adapt` do not use this function."""
    old = field.grid
    if not isinstance(old, OctreeGrid):
        raise TypeError("transfer needs an OctreeGrid field")
    if old.shape_vox != new_grid.shape_vox or old.max_level != new_grid.max_level:
        raise ValueError("the two octrees are not of the same box")
    K = field.tau.shape[1]
    P = new_grid.node_units
    oid = old.node_id(P)
    n = len(P)
    tau = np.full((n, K), np.nan)
    D = np.zeros((n, K, 3))
    q = np.zeros((n, K))
    qs = np.zeros(n)
    nk = np.zeros(n, np.int64)
    same = oid >= 0
    tau[same] = field.tau[oid[same]]
    D[same] = field.direction[oid[same]]
    q[same] = field.q[oid[same]]
    qs[same] = field.q_stop[oid[same]]
    nk[same] = field.n_kept[oid[same]]
    new = np.flatnonzero(~same)
    if len(new):
        p = P[new]
        leaf = old.find_leaf(np.minimum(p, old.extent - 1))
        if np.any(leaf < 0):
            raise AssertionError("a new node is outside the old octree")
        local = (p - old.anchor[leaf]) / old.size_units[leaf][:, None].astype(
            np.float64
        )
        w = T.trilinear_weights(local)
        cn = old.cells[leaf]
        mass = (w[:, :, None] * field.q[cn]).reshape(len(new), 8 * K)
        ctau = field.tau[cn].reshape(len(new), 8 * K)
        cdir = field.direction[cn].reshape(len(new), 8 * K, 3)
        order = np.argsort(-mass, axis=1, kind="stable")[:, :K]
        km = np.take_along_axis(mass, order, 1)
        valid = km > 0
        kt = np.where(valid, np.take_along_axis(ctau, order, 1), np.nan)
        kd = np.where(
            valid[..., None], np.take_along_axis(cdir, order[..., None], 1), 0.0
        )
        q_stop = np.clip((w * field.q_stop[cn]).sum(1), 0.0, 1.0)
        s = km.sum(1)
        q_stop = np.where(s > 0, q_stop, 1.0)
        scale = np.where(s > 0, (1.0 - q_stop) / np.maximum(s, 1e-300), 0.0)
        tau[new] = kt
        D[new] = kd
        q[new] = km * scale[:, None]
        qs[new] = q_stop
        nk[new] = valid.sum(1)
    return T.Field(grid=new_grid, tau=tau, direction=D, q=q, q_stop=qs, n_kept=nk)


# ---------------------------------------------------------------- the adapt loop


LogFn = Any


def _terms(log: dict[str, Any]) -> dict[str, float]:
    return {k: float(v) for k, v in log["terms"][-1].items() if k != "width"}


def _trial_record(
    kind: str,
    n_nominated: int,
    before: OctreeGrid,
    after: OctreeGrid,
    t_before: dict[str, float],
    t_after: dict[str, float] | None,
    log_after: dict[str, Any] | None,
) -> dict[str, Any]:
    rec: dict[str, Any] = dict(
        kind=kind,
        n_nominated=int(n_nominated),
        changed=not same_tree(before, after),
        level_counts_before=before.level_counts(),
        level_counts_after=after.level_counts(),
        free_energy_before=t_before["free_energy"],
        terms_before=t_before,
    )
    if t_after is None:
        rec.update(free_energy_after=None, delta_free_energy=None, terms_after=None,
                   delta_terms=None, accepted=False, solve=None)  # fmt: skip
        return rec
    assert log_after is not None
    rec.update(
        free_energy_after=t_after["free_energy"],
        delta_free_energy=t_after["free_energy"] - t_before["free_energy"],
        terms_after=t_after,
        delta_terms={k: t_after[k] - t_before[k] for k in t_after},
        accepted=bool(t_after["free_energy"] < t_before["free_energy"]),
        solve=dict(
            sweeps=log_after["sweeps"],
            end_reason=log_after["end_reason"],
            defects=log_after["defects"],
            time_total_s=log_after["time_total_s"],
        ),
    )
    return rec


def adapt(
    phantom_or_shape: T.Phantom | tuple[int, int, int],
    inputs: T.Inputs,
    noise_sd: Any,
    settings: T.SolverSettings = T.DEFAULTS,
    *,
    profile_dtype: Any = np.float64,
    log_fn: LogFn | None = None,
    max_sweeps: int | None = None,
) -> tuple[OctreeGrid, T.Field, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """The adaptive octree (the plan's *The octree*). Returns (grid, field, extras, log,
    record): the final octree, its solve (`ff_meanfield.solve`) and a JSON-able record
    of every round.

    1. Solve on `OctreeGrid.base` (the level-0 cells only).
    2. Up to settings.octree_rounds rounds. Each round: `indicators_of_solve`, then
       `mark` (one pass), then two trials:
       - the refinement trial: all nominated refinements at once (`OctreeGrid.refine`,
         then 2:1 balance); one solve on the new octree, init = the current field;
       - the coarsening trial: all nominated coarsenings at once, applied to the grid
         after the refinement trial (`OctreeGrid.coarsen` skips a parent whose 8
         children are no longer leaves or whose merge breaks the balance); one solve,
         init = the current field.
       A trial is accepted only if the free energy at the end of its last width
       (log["terms"][-1]["free_energy"]) is lower than that of the current solve
       (R9, *Acceptance by evidence*). A trial whose octree equals the current one is
       not solved and not accepted.
    3. The loop stops when a round accepts nothing.

    Decision for review: one trial for all nominated refinements of a round and one
    for all nominated coarsenings (not one per cell). It is deterministic and costs 2
    solves per round; a group with a good and a bad change is kept or refused as a
    whole. The coarsening uses the indicators of the grid before the refinement trial.

    max_sweeps: if given, replaces settings.max_sweeps (for tests; the record says
    so). In a trial, delta_terms gives the change of each expected term, and of the
    entropy H (which enters F as −H), from the current solve to the trial.

    The record: settings, rounds (per round: level_counts, the median and the
    thresholds, the nominated counts, the trials of `_trial_record`), the final
    level_counts, the prior-shaped leaves (level, anchor, origin in voxels, side in
    mm) and `hanging_step` of the final field."""
    import ff_meanfield as MF

    if max_sweeps is not None:
        settings = dataclasses.replace(settings, max_sweeps=int(max_sweeps))
    shape = (
        phantom_or_shape.shape
        if isinstance(phantom_or_shape, T.Phantom)
        else phantom_or_shape
    )
    say = log_fn if log_fn is not None else (lambda _m: None)

    def solve(
        g: OctreeGrid, init: T.Field | None
    ) -> tuple[T.Field, dict[str, Any], dict[str, Any]]:
        return MF.solve(g, inputs, noise_sd, settings, init=init,
                        profile_dtype=profile_dtype, log_fn=log_fn)  # fmt: skip

    grid = OctreeGrid.base(tuple(int(s) for s in shape), settings)  # type: ignore[arg-type]
    say(f"octree base: {grid.n_nodes} nodes")
    field, extras, log = solve(grid, None)
    cur_terms = _terms(log)
    record: dict[str, Any] = dict(
        settings=settings.record(),
        max_sweeps_override=max_sweeps,
        base=dict(level_counts=grid.level_counts(), terms=cur_terms,
                  defects=log["defects"], sweeps=log["sweeps"]),
        rounds=[],
    )  # fmt: skip
    for r in range(settings.octree_rounds):
        ind = indicators_of_solve(grid, field, extras, inputs)
        marks = mark(grid, ind["total"], settings)
        rnd: dict[str, Any] = dict(
            round=r,
            level_counts=grid.level_counts(),
            indicator_median=marks["median"],
            refine_threshold=marks["refine_threshold"],
            coarsen_threshold=marks["coarsen_threshold"],
            indicator_parts_median={
                k: float(np.median(ind[k])) for k in INDICATOR_PARTS
            },
            n_refine_nominated=int(len(marks["refine_level"])),
            n_coarsen_nominated=int(len(marks["coarsen_level"])),
            trials=[],
        )
        accepted = False
        # the refinement trial
        if len(marks["refine_level"]):
            g_new = grid.refine(
                leaf_ids(grid, marks["refine_level"], marks["refine_anchor"])
            )
            f2, e2, l2 = solve(g_new, field)
            t2 = _terms(l2)
            tr = _trial_record(
                "refine", len(marks["refine_level"]), grid, g_new, cur_terms, t2, l2
            )
            rnd["trials"].append(tr)
            say(f"round {r} refine: F {cur_terms['free_energy']:.10g} -> "
                f"{t2['free_energy']:.10g} accepted={tr['accepted']}")  # fmt: skip
            if tr["accepted"]:
                grid, field, extras, log, cur_terms = g_new, f2, e2, l2, t2
                accepted = True
        # the coarsening trial
        if len(marks["coarsen_level"]):
            g_new = grid.coarsen(marks["coarsen_level"], marks["coarsen_anchor"])
            if same_tree(g_new, grid):
                tr = _trial_record("coarsen", len(marks["coarsen_level"]), grid, g_new,
                                   cur_terms, None, None)  # fmt: skip
            else:
                f2, e2, l2 = solve(g_new, field)
                t2 = _terms(l2)
                tr = _trial_record("coarsen", len(marks["coarsen_level"]), grid, g_new,
                                   cur_terms, t2, l2)  # fmt: skip
                say(f"round {r} coarsen: F {cur_terms['free_energy']:.10g} -> "
                    f"{t2['free_energy']:.10g} accepted={tr['accepted']}")  # fmt: skip
                if tr["accepted"]:
                    grid, field, extras, log, cur_terms = g_new, f2, e2, l2, t2
                    accepted = True
            rnd["trials"].append(tr)
        record["rounds"].append(rnd)
        if not accepted:
            break
    ps = np.flatnonzero(grid.prior_shaped)
    record["final"] = dict(
        level_counts=grid.level_counts(),
        terms=cur_terms,
        prior_shaped=dict(
            count=int(len(ps)),
            level=grid.level[ps].tolist(),
            anchor=grid.anchor[ps].tolist(),
            origin_vox=grid.cell_origin[ps].tolist(),
            side_mm=grid.cell_size_mm[ps].tolist(),
        ),
        hanging_step=hanging_step(grid, field, settings),
    )
    return grid, field, extras, log, record


# ---------------------------------------------------------------- the hanging step


def _stats2(x: Array) -> dict[str, float | None]:
    x = np.asarray(x, np.float64)
    if len(x) == 0:
        return dict(median=None, max=None)
    return dict(median=float(np.median(x)), max=float(np.max(x)))


def hanging_coarse_leaf(grid: OctreeGrid) -> IntArray:
    """(H,) for each hanging node (in the order of `hanging`), the larger leaf on whose
    edge middle or face centre it lies (the lowest leaf id if there are more)."""
    ids = grid.hanging[0]
    out = np.full(grid.n_nodes, np.iinfo(np.int64).max, np.int64)
    big = np.flatnonzero(grid.size_units >= 2)
    cp = grid._corner_points[big]
    for a, b in EDGE_PAIRS:
        nid = grid.node_id((cp[:, a] + cp[:, b]) // 2)
        hit = nid >= 0
        np.minimum.at(out, nid[hit], big[hit])
    for fc in FACE_CORNERS:
        nid = grid.node_id((cp[:, fc[0]] + cp[:, fc[3]]) // 2)
        hit = nid >= 0
        np.minimum.at(out, nid[hit], big[hit])
    return np.asarray(out[ids])


def hanging_step(
    grid: OctreeGrid, field: T.Field, settings: T.SolverSettings = T.DEFAULTS
) -> dict[str, Any]:
    """The step of the field at the hanging nodes (CD-10: a hanging node has its own
    free state, so the field can step there). JSON-able.

    Node view, at the hanging nodes where the node and all its parents (weight > 0) are
    moving (q_stop < settings.stop_level): tau_diff = |τ_own − Σ w τ_parent| (mean τ),
    angle_deg = the angle between the own mean direction and the normalised Σ w d_parent
    (where both are nonzero), q_stop_diff = |q_stop,own − Σ w q_stop,parent| (the
    parents are the direct ones of `hanging`; a parent can be hanging).

    Integrator view (`ff_integrate.masked_sample`), at the same nodes: the field at the
    node's position moved 1e-6 voxel toward the centre of the coarse leaf
    (`hanging_coarse_leaf`), and moved 1e-6 voxel toward the centre of the leaf of
    lowest id that has the node as a corner (the fine side). int_angle_deg is the angle
    between the two masked directions, int_tau_diff the difference of the two masked τ
    (where both are finite and the directions nonzero).

    Each gives its median and max; n_hanging, n_moving (node view) and n_integrator
    give the counts."""
    import ff_integrate as FI

    ids, par, w = grid.hanging
    out: dict[str, Any] = dict(n_hanging=int(len(ids)))
    if len(ids) == 0:
        for k in (
            "tau_diff",
            "angle_deg",
            "q_stop_diff",
            "int_angle_deg",
            "int_tau_diff",
        ):
            out[k] = _stats2(np.zeros(0))
        out.update(n_moving=0, n_integrator=0)
        return out
    lvl = settings.stop_level
    qs = field.q_stop
    tau = np.nan_to_num(field.tau_mean())
    d = field.direction_mean()
    used = w > 0
    mov = qs < lvl
    ok = mov[ids] & np.all(~used | mov[par], axis=1)
    t_par = (w * tau[par]).sum(1)
    q_par = (w * qs[par]).sum(1)
    D = np.einsum("hp,hpx->hx", w, d[par])
    nD = np.linalg.norm(D, axis=1)
    nd = np.linalg.norm(d[ids], axis=1)
    cosang = np.einsum("hx,hx->h", d[ids], D) / np.maximum(nD * nd, 1e-300)
    ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    ok_ang = ok & (nD > 0) & (nd > 0)
    out["tau_diff"] = _stats2(np.abs(tau[ids] - t_par)[ok])
    out["angle_deg"] = _stats2(ang[ok_ang])
    out["q_stop_diff"] = _stats2(np.abs(qs[ids] - q_par)[ok])
    out["n_moving"] = int(ok.sum())
    # the integrator's view
    X = grid.nodes[ids]
    coarse = hanging_coarse_leaf(grid)
    centre = grid.cell_origin + 0.5 * (grid.size_units * grid.unit_vox)[:, None]
    fine = np.full(grid.n_nodes, np.iinfo(np.int64).max, np.int64)
    np.minimum.at(fine, grid.cells.ravel(), np.repeat(np.arange(grid.n_cells), 8))
    fine = fine[ids]

    def nudge(leaf: IntArray) -> Array:
        v = centre[leaf] - X
        return X + 1e-6 * v / np.linalg.norm(v, axis=1)[:, None]

    P = np.concatenate([nudge(coarse), nudge(fine)])
    Ds, ts, _, cell, _ = FI.masked_sample(field, P, settings)
    h = len(ids)
    Dc, Df = Ds[:h], Ds[h:]
    tc, tf = ts[:h], ts[h:]
    nc = np.linalg.norm(np.nan_to_num(Dc), axis=1)
    nf = np.linalg.norm(np.nan_to_num(Df), axis=1)
    oki = (
        ok
        & (nc > 0)
        & (nf > 0)
        & np.isfinite(tc)
        & np.isfinite(tf)
        & (cell[:h] >= 0)
        & (cell[h:] >= 0)
    )
    c2 = np.einsum("hx,hx->h", np.nan_to_num(Dc), np.nan_to_num(Df)) / np.maximum(
        nc * nf, 1e-300
    )
    out["int_angle_deg"] = _stats2(np.degrees(np.arccos(np.clip(c2, -1.0, 1.0)))[oki])
    out["int_tau_diff"] = _stats2(np.abs(tc - tf)[oki])
    out["n_integrator"] = int(oki.sum())
    out["coarse_leaf_check"] = int(
        np.sum(cell[:h] != coarse)
    )  # points that left the coarse leaf
    return out


# ---------------------------------------------------------------- null inputs

NULL_P = 0.02  # the null stop probability (the oracle clip)


class NullStop:
    """The null stop field of the noise phantom: log-odds log(0.02 / 0.98) everywhere
    (nothing stopped: the noise phantom is cortex everywhere; 0.02 is the oracle clip).
    The noise phantom has no geometry, so it has no oracle; its image stop field is not
    a null (PR 2: p > 0.5 at 99 % of the points)."""

    source: T.InputSource = "image"

    def log_odds(self, X: Array) -> Array:
        return np.full(len(X), np.log(NULL_P / (1.0 - NULL_P)))

    def terms(self, X: Array) -> dict[str, Array]:
        return {"null": self.log_odds(X)}


class NullTensor:
    """The null tensor of the noise phantom: coherence 0 (no direction)."""

    source: T.InputSource = "image"

    def at(self, X: Array) -> tuple[Array, Array]:
        d = np.zeros((len(X), 3))
        d[:, 0] = 1.0
        return d, np.zeros(len(X))


def null_inputs(phantom: T.Phantom) -> T.Inputs:
    """The null inputs of the noise phantom (`NullStop`, `NullTensor`, its det)."""
    return T.Inputs(stop=NullStop(), tensor=NullTensor(), contrast=phantom.det)
