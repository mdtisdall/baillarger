"""Exact triangle-triangle intersection counts for triangle meshes, in
parallel on the CPU with numba.

Candidate pairs come from a uniform grid over the triangles' bounding boxes;
each pair is tested with the interval test of Moller (1997): each triangle
must straddle the other's plane, and the two triangles' intervals on the line
where the planes meet must overlap. Touching pairs (a vertex or edge on the
other triangle, within a tolerance) and coplanar pairs count as not
intersecting. In a self test, triangles that share a vertex are not tested
(neighbours touch by construction).

The results are those of a plain numpy implementation (the reference in the
tests, mesh_intersect_ref), bit for bit. Grid: each box goes into the cells it
covers, with keys cx * 73856093 ^ cy * 19349663 ^ cz * 83492791; boxes that
span more than max_span cells in some direction (stretched triangles) are
compared with all boxes directly. Each overlapping pair is visited once, in the
one cell max(c0_a, c0_b) (the cell of the lowest corner of the boxes'
intersection, which both boxes cover), so no pair list needs to be built or
sorted. Cells whose keys collide share a group of the sorted keys; a pair is
skipped there unless both entries are for that one cell. The entries are
visited in blocks in a prange loop, in a shuffled block order so that crowded
cells are spread over the threads.

max_pairs: the candidate count compared with it is the reference's (the number
of pairs within each group of equal keys, then the boxes that stretched boxes
overlap). Above max_pairs, a RuntimeError is raised: crowded triangles are a
sign of a collapsed mesh.

Pair test: the reference's float64 operations in the same order, so that the
results are identical, also near the tolerance. The dot products are
fma(x2, y2, fma(x1, y1, x0 * y0)) with llvm.fma: the order in which the numpy of
the prototype summed np.einsum("ij,ij->i"). Other numpy builds can sum einsum
in another order (numpy 2.5 on arm64 does), so there the reference can differ
in the last bits, which matters only for a pair within rounding of the
tolerance. np.cross, np.linalg.norm and .mean use separate multiplies and adds,
left to right.

The grid cells are computed in the dtype of the input, as in the reference (so
that max_pairs acts the same); the pair tests are always in float64, so for
float32 vertices the results can differ from the reference, which then tests
in float32. The chunk arguments are accepted and not used.
"""

from collections.abc import Callable
from typing import Any, cast, overload

import numba
import numpy as np
import numpy.typing as npt
from llvmlite import ir
from numba.core import cgutils
from numba.extending import intrinsic

from baillarger._jit import njit, prange

type Arr = npt.NDArray[Any]
type Bool = npt.NDArray[np.bool_]
type Int = npt.NDArray[np.int64]

_BLOCK = 512  # sorted grid entries per work block


def _fma_typing(typingctx: Any, a: Any, b: Any, c: Any) -> tuple[Any, Any]:
    """a * b + c with one rounding (llvm.fma.f64)."""
    sig = numba.float64(numba.float64, numba.float64, numba.float64)

    def codegen(context: Any, builder: Any, signature: Any, args: Any) -> Any:
        d = ir.DoubleType()
        fn = cgutils.get_or_insert_function(
            builder.module, ir.FunctionType(d, [d, d, d]), "llvm.fma.f64"
        )
        return builder.call(fn, args)

    return sig, codegen


_fma = cast(Callable[[float, float, float], float], intrinsic(_fma_typing))


@njit(inline="always")
def _dot(x0: float, x1: float, x2: float, y0: float, y1: float, y2: float) -> float:
    # the order of np.einsum("ij,ij->i"): fma(x2, y2, fma(x1, y1, x0 * y0))
    return _fma(x2, y2, _fma(x1, y1, x0 * y0))


@njit(parallel=True)
def _dot_rows_kernel(x: Arr, y: Arr, out: Arr) -> None:
    for i in prange(len(x)):
        out[i] = _dot(x[i, 0], x[i, 1], x[i, 2], y[i, 0], y[i, 1], y[i, 2])


def _dot_rows(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Row dot products of (M, 3) arrays with _dot (for the tests)."""
    x = np.ascontiguousarray(x, np.float64)
    y = np.ascontiguousarray(y, np.float64)
    out = np.empty(len(x))
    _dot_rows_kernel(x, y, out)
    return out


@njit(inline="always")
def _norm(x0: float, x1: float, x2: float) -> float:
    return float(np.sqrt((x0 * x0 + x1 * x1) + x2 * x2))


@njit(inline="always")
def _mean_edge(T: Arr, a: int) -> float:
    # np.linalg.norm(np.diff(T, axis=1, append=T[:, :1]), axis=2).mean(1)
    e0 = _norm(
        T[a, 1, 0] - T[a, 0, 0], T[a, 1, 1] - T[a, 0, 1], T[a, 1, 2] - T[a, 0, 2]
    )
    e1 = _norm(
        T[a, 2, 0] - T[a, 1, 0], T[a, 2, 1] - T[a, 1, 1], T[a, 2, 2] - T[a, 1, 2]
    )
    e2 = _norm(
        T[a, 0, 0] - T[a, 2, 0], T[a, 0, 1] - T[a, 2, 1], T[a, 0, 2] - T[a, 2, 2]
    )
    return ((e0 + e1) + e2) / 3.0


@njit(inline="always")
def _plane(T: Arr, a: int) -> tuple[float, float, float, float]:
    u0, u1, u2 = (
        T[a, 1, 0] - T[a, 0, 0],
        T[a, 1, 1] - T[a, 0, 1],
        T[a, 1, 2] - T[a, 0, 2],
    )
    v0, v1, v2 = (
        T[a, 2, 0] - T[a, 0, 0],
        T[a, 2, 1] - T[a, 0, 1],
        T[a, 2, 2] - T[a, 0, 2],
    )
    n0 = u1 * v2 - u2 * v1
    n1 = u2 * v0 - u0 * v2
    n2 = u0 * v1 - u1 * v0
    d = -_dot(n0, n1, n2, T[a, 0, 0], T[a, 0, 1], T[a, 0, 2])
    return n0, n1, n2, d


@njit(inline="always")
def _interval(
    d0: float, d1: float, d2: float, p0: float, p1: float, p2: float
) -> tuple[float, float]:
    # the reference's _interval for one triangle: edge crossings and vertices on
    # the plane
    lo, hi = np.inf, -np.inf
    if d0 * d1 < 0:
        t = p0 + (p1 - p0) * d0 / (d0 - d1)
        lo, hi = min(lo, t), max(hi, t)
    if d1 * d2 < 0:
        t = p1 + (p2 - p1) * d1 / (d1 - d2)
        lo, hi = min(lo, t), max(hi, t)
    if d2 * d0 < 0:
        t = p2 + (p0 - p2) * d2 / (d2 - d0)
        lo, hi = min(lo, t), max(hi, t)
    if d0 == 0:
        lo, hi = min(lo, p0), max(hi, p0)
    if d1 == 0:
        lo, hi = min(lo, p1), max(hi, p1)
    if d2 == 0:
        lo, hi = min(lo, p2), max(hi, p2)
    return lo, hi


@njit
def _tt(A: Arr, a: int, B: Arr, b: int, rel_tol: float) -> bool:
    """mesh_intersect.tri_tri for the one pair A[a], B[b]."""
    tol = rel_tol * (0.5 * (_mean_edge(A, a) + _mean_edge(B, b)))
    m0, m1, m2, e2 = _plane(B, b)
    thr = tol * _norm(m0, m1, m2)
    da0 = _dot(A[a, 0, 0], A[a, 0, 1], A[a, 0, 2], m0, m1, m2) + e2
    da1 = _dot(A[a, 1, 0], A[a, 1, 1], A[a, 1, 2], m0, m1, m2) + e2
    da2 = _dot(A[a, 2, 0], A[a, 2, 1], A[a, 2, 2], m0, m1, m2) + e2
    if abs(da0) < thr:
        da0 = 0.0
    if abs(da1) < thr:
        da1 = 0.0
    if abs(da2) < thr:
        da2 = 0.0
    if (da0 >= 0 and da1 >= 0 and da2 >= 0) or (da0 <= 0 and da1 <= 0 and da2 <= 0):
        return False
    n0, n1, n2, e1 = _plane(A, a)
    thr = tol * _norm(n0, n1, n2)
    db0 = _dot(B[b, 0, 0], B[b, 0, 1], B[b, 0, 2], n0, n1, n2) + e1
    db1 = _dot(B[b, 1, 0], B[b, 1, 1], B[b, 1, 2], n0, n1, n2) + e1
    db2 = _dot(B[b, 2, 0], B[b, 2, 1], B[b, 2, 2], n0, n1, n2) + e1
    if abs(db0) < thr:
        db0 = 0.0
    if abs(db1) < thr:
        db1 = 0.0
    if abs(db2) < thr:
        db2 = 0.0
    if (db0 >= 0 and db1 >= 0 and db2 >= 0) or (db0 <= 0 and db1 <= 0 and db2 <= 0):
        return False
    # line where the planes meet: D = n1 x n2
    D0 = n1 * m2 - n2 * m1
    D1 = n2 * m0 - n0 * m2
    D2 = n0 * m1 - n1 * m0
    pa0 = _dot(A[a, 0, 0], A[a, 0, 1], A[a, 0, 2], D0, D1, D2)
    pa1 = _dot(A[a, 1, 0], A[a, 1, 1], A[a, 1, 2], D0, D1, D2)
    pa2 = _dot(A[a, 2, 0], A[a, 2, 1], A[a, 2, 2], D0, D1, D2)
    pb0 = _dot(B[b, 0, 0], B[b, 0, 1], B[b, 0, 2], D0, D1, D2)
    pb1 = _dot(B[b, 1, 0], B[b, 1, 1], B[b, 1, 2], D0, D1, D2)
    pb2 = _dot(B[b, 2, 0], B[b, 2, 1], B[b, 2, 2], D0, D1, D2)
    a0, a1 = _interval(da0, da1, da2, pa0, pa1, pa2)
    b0, b1 = _interval(db0, db1, db2, pb0, pb1, pb2)
    return min(a1, b1) - max(a0, b0) > tol * _norm(D0, D1, D2)


@njit(parallel=True)
def _tri_tri_kernel(A: Arr, B: Arr, rel_tol: float, out: Bool) -> None:
    for k in prange(len(A)):
        out[k] = _tt(A, k, B, k, rel_tol)


def tri_tri(A: npt.ArrayLike, B: npt.ArrayLike, rel_tol: float = 1e-4) -> Bool:
    """Whether triangles A[k] and B[k] (M, 3, 3) intersect: mesh_intersect.tri_tri."""
    A = np.ascontiguousarray(A, np.float64)
    B = np.ascontiguousarray(B, np.float64)
    out = np.zeros(len(A), bool)
    if len(A):
        _tri_tri_kernel(A, B, float(rel_tol), out)
    return out


# ---- the grid ----------------------------------------------------------------


@njit(parallel=True)
def _entries(
    small: Arr, c0: Arr, span: Arr, off: Arr, keys: Arr, tris: Arr, dpk: Arr
) -> None:
    # one entry for each cell of each small box: the reference's cell key, the box,
    # and the cell's offset from c0 packed as dx * 16 + dy * 4 + dz (spans <= max_span)
    for s in prange(len(small)):
        t = small[s]
        k = off[s]
        for dx in range(span[t, 0]):
            for dy in range(span[t, 1]):
                for dz in range(span[t, 2]):
                    cx, cy, cz = c0[t, 0] + dx, c0[t, 1] + dy, c0[t, 2] + dz
                    keys[k] = cx * 73856093 ^ cy * 19349663 ^ cz * 83492791
                    tris[k] = t
                    dpk[k] = dx * 16 + dy * 4 + dz
                    k += 1


@njit(inline="always")
def _overlap(lo: Arr, hi: Arr, i: int, j: int) -> bool:
    return bool(
        lo[i, 0] <= hi[j, 0]
        and lo[j, 0] <= hi[i, 0]
        and lo[i, 1] <= hi[j, 1]
        and lo[j, 1] <= hi[i, 1]
        and lo[i, 2] <= hi[j, 2]
        and lo[j, 2] <= hi[i, 2]
    )


@njit(inline="always")
def _visit(
    i: int,
    j: int,
    mode: int,
    k: int,
    pairs: Arr,
    ns: int,
    A: Arr,
    B: Arr,
    FA: Arr,
    FB: Arr,
    IA: Arr,
    IB: Arr,
    share: bool,
    mA: Arr,
    mB: Arr,
) -> int:
    """Pair i < j of overlapping boxes. mode 0: count it; 1: write it at pairs[k];
    2: test it (with i < ns <= j when ns >= 0; i, j - ns index A, B) and mark hits.
    Returns 1 when the pair counts."""
    if mode == 0:
        return 1
    if mode == 1:
        pairs[k, 0] = i
        pairs[k, 1] = j
        return 1
    a, b = i, j
    if ns >= 0:
        b = j - ns
    if share:
        if IA[a] == IB[b]:
            return 0
        for u in range(3):
            for v in range(3):
                if FA[a, u] == FB[b, v]:
                    return 0
    if _tt(A, a, B, b, 1e-4):
        mA[a] = True
        mB[b] = True
        return 1
    return 0


@njit(parallel=True)
def _scan_grid(
    perm: Arr,
    gend: Arr,
    tris: Arr,
    dpk: Arr,
    c0: Arr,
    lo: Arr,
    hi: Arr,
    mode: int,
    off: Arr,
    pairs: Arr,
    ns: int,
    A: Arr,
    B: Arr,
    FA: Arr,
    FB: Arr,
    IA: Arr,
    IB: Arr,
    share: bool,
    mA: Arr,
    mB: Arr,
    cnt: Arr,
) -> None:
    """Each overlapping pair of small boxes once, in the cell max(c0_a, c0_b)."""
    ne = len(tris)
    for kb in prange(len(perm)):
        blk = perm[kb]
        k = off[blk]
        n = 0
        for e in range(blk * _BLOCK, min(ne, (blk + 1) * _BLOCK)):
            a = tris[e]
            d = np.int64(dpk[e])
            ax, ay, az = c0[a, 0] + d // 16, c0[a, 1] + (d // 4) % 4, c0[a, 2] + d % 4
            for f in range(e + 1, gend[e]):
                b = tris[f]
                if b == a:
                    continue
                # the same cell (not only the same key), and that cell is
                # max(c0_a, c0_b)
                d = np.int64(dpk[f])
                if (
                    c0[b, 0] + d // 16 != ax
                    or c0[b, 1] + (d // 4) % 4 != ay
                    or c0[b, 2] + d % 4 != az
                ):
                    continue
                if (
                    max(c0[a, 0], c0[b, 0]) != ax
                    or max(c0[a, 1], c0[b, 1]) != ay
                    or max(c0[a, 2], c0[b, 2]) != az
                ):
                    continue
                i, j = min(a, b), max(a, b)
                if ns >= 0 and not (i < ns <= j):
                    continue
                if not _overlap(lo, hi, i, j):
                    continue
                n += _visit(
                    i, j, mode, k + n, pairs, ns, A, B, FA, FB, IA, IB, share, mA, mB
                )
        cnt[blk] = n


@njit(parallel=True)
def _scan_big(
    bigs: Arr,
    isbig: Arr,
    lo: Arr,
    hi: Arr,
    mode: int,
    off: Arr,
    pairs: Arr,
    ns: int,
    A: Arr,
    B: Arr,
    FA: Arr,
    FB: Arr,
    IA: Arr,
    IB: Arr,
    share: bool,
    mA: Arr,
    mB: Arr,
    cnt: Arr,
) -> None:
    """Each pair of a stretched box with an overlapping box once. mode -1: count all
    boxes that each stretched box overlaps (the reference's count for max_pairs)."""
    nb = len(lo)
    for kb in prange(len(bigs)):
        t = bigs[kb]
        k = off[kb]
        n = 0
        for j in range(nb):
            if j == t:
                continue
            if mode >= 0 and isbig[j] and j < t:
                continue  # visited from j
            if not _overlap(lo, hi, t, j):
                continue
            if mode < 0:
                n += 1
                continue
            i, jj = min(t, j), max(t, j)
            if ns >= 0 and not (i < ns <= jj):
                continue
            n += _visit(
                i, jj, mode, k + n, pairs, ns, A, B, FA, FB, IA, IB, share, mA, mB
            )
        cnt[kb] = n


class _Grid:
    """The reference's grid over boxes [lo, hi] (M, 3), its max_pairs check, and scans
    of the overlapping pairs."""

    def __init__(
        self,
        lo: npt.ArrayLike,
        hi: npt.ArrayLike,
        cell: float | None = None,
        max_pairs: int | None = None,
        max_span: int = 4,
    ) -> None:
        # cells in the boxes' own dtype, as the reference computes them
        lo, hi = np.asarray(lo), np.asarray(hi)
        self.lo = np.ascontiguousarray(lo, np.float64)
        self.hi = np.ascontiguousarray(hi, np.float64)
        self.n = len(lo)
        ext = hi - lo
        if cell is None:
            cell = 2.0 * np.median(ext.max(1)) + 1e-12 if len(lo) else 1.0
        c0 = np.floor(lo / cell).astype(np.int64)
        c1 = np.floor(hi / cell).astype(np.int64)
        span = c1 - c0 + 1
        big = np.any(span > max_span, 1)
        small = np.where(~big)[0]
        # dx * 16 + dy * 4 + dz needs spans of at most 4
        assert max_span <= 4, "max_span > 4 is not supported"
        sz = np.prod(span[small], 1)
        off = np.r_[0, np.cumsum(sz)].astype(np.int64)
        ne = int(off[-1])
        keys = np.empty(ne, np.int64)
        tris = np.empty(ne, np.int64)
        dpk = np.empty(ne, np.uint8)
        if len(small):
            _entries(small, c0, span, off, keys, tris, dpk)
        o = np.argsort(keys)
        keys, self.tris, self.dpk = keys[o], tris[o], dpk[o]
        total = 0
        if len(small):
            start = np.r_[0, np.flatnonzero(np.diff(keys)) + 1]
            size = np.diff(np.r_[start, ne])
            total = int((size * (size - 1) // 2).sum())
            if max_pairs is not None and total > max_pairs:
                raise RuntimeError(
                    f"{total} candidate pairs (more than {max_pairs}); "
                    f"largest cell has {size.max()} triangles"
                )
            self.gend = np.repeat(start + size, size)
        else:
            self.gend = np.zeros(0, np.int64)
        self.c0, self.isbig, self.bigs = c0, big, np.where(big)[0]
        if max_pairs is not None and len(self.bigs):
            cnt = self._big(-1)
            if total + int(cnt.sum()) > max_pairs:
                raise RuntimeError(
                    f"more than {max_pairs} candidate pairs, "
                    f"with {big.sum()} stretched triangles"
                )
        nblk = -(-ne // _BLOCK)
        # shuffled block order: crowded cells spread over the threads
        self.perm = np.random.default_rng(0).permutation(nblk).astype(np.int64)

    _no3 = np.zeros((0, 3, 3))
    _noF = np.zeros((0, 3), np.int64)
    _noI = np.zeros(0, np.int64)
    _noM = np.zeros(0, bool)

    def _grid(
        self,
        mode: int,
        off: Arr | None = None,
        pairs: Arr | None = None,
        ns: int = -1,
        A: Arr = _no3,
        B: Arr = _no3,
        FA: Arr = _noF,
        FB: Arr = _noF,
        IA: Arr = _noI,
        IB: Arr = _noI,
        share: bool = False,
        mA: Arr = _noM,
        mB: Arr = _noM,
    ) -> Int:
        cnt = np.zeros(len(self.perm), np.int64)
        if len(self.perm):
            _scan_grid(
                self.perm,
                self.gend,
                self.tris,
                self.dpk,
                self.c0,
                self.lo,
                self.hi,
                mode,
                np.zeros(len(self.perm), np.int64) if off is None else off,
                np.zeros((0, 2), np.int64) if pairs is None else pairs,
                ns,
                A,
                B,
                FA,
                FB,
                IA,
                IB,
                share,
                mA,
                mB,
                cnt,
            )
        return cnt

    def _big(
        self,
        mode: int,
        off: Arr | None = None,
        pairs: Arr | None = None,
        ns: int = -1,
        A: Arr = _no3,
        B: Arr = _no3,
        FA: Arr = _noF,
        FB: Arr = _noF,
        IA: Arr = _noI,
        IB: Arr = _noI,
        share: bool = False,
        mA: Arr = _noM,
        mB: Arr = _noM,
    ) -> Int:
        cnt = np.zeros(len(self.bigs), np.int64)
        if len(self.bigs):
            _scan_big(
                self.bigs,
                self.isbig,
                self.lo,
                self.hi,
                mode,
                np.zeros(len(self.bigs), np.int64) if off is None else off,
                np.zeros((0, 2), np.int64) if pairs is None else pairs,
                ns,
                A,
                B,
                FA,
                FB,
                IA,
                IB,
                share,
                mA,
                mB,
                cnt,
            )
        return cnt

    def pairs(self) -> Int:
        """All overlapping pairs (i < j), sorted as np.unique sorts them."""
        cg, cb = self._grid(0), self._big(0)
        og = np.r_[0, np.cumsum(cg)].astype(np.int64)
        ob = og[-1] + np.r_[0, np.cumsum(cb)].astype(np.int64)
        P = np.empty((int(ob[-1]), 2), np.int64)
        self._grid(1, og[:-1], P)
        self._big(1, ob[:-1], P)
        return P[np.argsort(P[:, 0] * max(self.n, 1) + P[:, 1], kind="stable")]

    def test(self, **kw: Any) -> int:
        """Test the pairs (see _visit); returns the number of intersecting pairs."""
        return int(self._grid(2, **kw).sum() + self._big(2, **kw).sum())


def candidate_pairs(
    lo: npt.ArrayLike,
    hi: npt.ArrayLike,
    cell: float | None = None,
    max_pairs: int | None = None,
    max_span: int = 4,
) -> Int:
    """mesh_intersect.candidate_pairs: pairs (i < j) of boxes [lo, hi] (M, 3) that
    overlap, sorted; RuntimeError above max_pairs candidate pairs."""
    return _Grid(lo, hi, cell, max_pairs, max_span).pairs()


@overload
def intersecting_faces(
    V: npt.ArrayLike,
    F: npt.ArrayLike,
    V2: None = None,
    F2: None = None,
    chunk: int = 500000,
    max_pairs: int | None = None,
) -> tuple[Bool, int]: ...


@overload
def intersecting_faces(
    V: npt.ArrayLike,
    F: npt.ArrayLike,
    V2: npt.ArrayLike,
    F2: npt.ArrayLike,
    chunk: int = 500000,
    max_pairs: int | None = None,
) -> tuple[Bool, Bool, int]: ...


def intersecting_faces(
    V: npt.ArrayLike,
    F: npt.ArrayLike,
    V2: npt.ArrayLike | None = None,
    F2: npt.ArrayLike | None = None,
    chunk: int = 500000,
    max_pairs: int | None = None,
) -> tuple[Bool, int] | tuple[Bool, Bool, int]:
    """mesh_intersect.intersecting_faces: faces that intersect another face, within
    (V, F) when V2 is None (pairs sharing a vertex skipped), else between (V, F) and
    (V2, F2). Returns a boolean mask over F (and over F2 for two meshes) and the
    number of intersecting pairs."""
    F = np.ascontiguousarray(F, np.int64)
    V = np.asarray(V)
    T1 = np.ascontiguousarray(V[F], np.float64)
    if V2 is None:
        Tv = V[F]  # boxes in V's dtype, as the reference
        g = _Grid(Tv.min(1), Tv.max(1), max_pairs=max_pairs)
        m1 = np.zeros(len(T1), bool)
        ids = np.arange(len(F), dtype=np.int64)
        n = g.test(A=T1, B=T1, FA=F, FB=F, IA=ids, IB=ids, share=True, mA=m1, mB=m1)
        return m1, n
    F2 = np.ascontiguousarray(F2, np.int64)
    V2 = np.asarray(V2)
    T2 = np.ascontiguousarray(V2[F2], np.float64)
    T = np.concatenate([V[F], V2[F2]])
    g = _Grid(T.min(1), T.max(1), max_pairs=max_pairs)
    m1, m2 = np.zeros(len(T1), bool), np.zeros(len(T2), bool)
    n = g.test(ns=len(T1), A=T1, B=T2, mA=m1, mB=m2)
    return m1, m2, n


def faces_hitting(
    V: npt.ArrayLike,
    F: npt.ArrayLike,
    sub: npt.ArrayLike,
    V2: npt.ArrayLike | None = None,
    F2: npt.ArrayLike | None = None,
    chunk: int = 500000,
    max_pairs: int | None = None,
) -> Bool:
    """mesh_intersect.faces_hitting: which of the faces F[sub] intersect a face of
    (V, F) (sub's own faces and faces sharing a vertex with them not counted) or, with
    V2, a face of (V2, F2). Only the faces near sub are tested."""
    sub = np.asarray(sub)
    F = np.ascontiguousarray(F, np.int64)
    V = np.asarray(V)
    Fs = np.ascontiguousarray(F[sub])
    Ts = V[Fs]  # boxes in V's dtype, as the reference
    Fo = F if V2 is None else np.ascontiguousarray(F2, np.int64)
    To = V[F] if V2 is None else np.asarray(V2)[Fo]
    lo_s, hi_s = Ts.min(1), Ts.max(1)
    lo_o, hi_o = To.min(1), To.max(1)
    # only the other faces whose boxes reach the box around sub
    near = np.all((lo_o <= hi_s.max(0)) & (lo_s.min(0) <= hi_o), 1)
    on = np.where(near)[0].astype(np.int64)
    g = _Grid(
        np.concatenate([lo_s, lo_o[on]]),
        np.concatenate([hi_s, hi_o[on]]),
        max_pairs=max_pairs,
    )
    out = np.zeros(len(sub), bool)
    g.test(
        ns=len(sub),
        A=np.ascontiguousarray(Ts, np.float64),
        B=np.ascontiguousarray(To[on], np.float64),
        FA=Fs,
        FB=np.ascontiguousarray(Fo[on]),
        IA=np.ascontiguousarray(sub, np.int64),
        IB=on,
        share=V2 is None,
        mA=out,
        mB=np.zeros(len(on), bool),
    )
    return out
