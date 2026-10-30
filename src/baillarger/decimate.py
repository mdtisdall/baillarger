"""Decimation of a triangle mesh by blocks, in parallel, with three guarantees:
the topology is kept (manifold edges, components, genus), every input vertex
stays within eps of the output surface, and the output's vertices are input
vertices (they never move).

Half-edge collapses: a vertex a is merged into a neighbour b, which keeps its
position. They are taken cheapest first by quadric error (Garland & Heckbert
1997: the area-weighted sum of squared distances from b to the planes of the
input faces merged into it). A collapse is made only if all hold:

1. Link condition (Dey, Edelsbrunner, Guha & Nekhayev 1999): edge ab has two
   faces, the common neighbours of a and b are exactly the two vertices c, d
   opposite it, and faces acd and bcd are not both there (a tetrahedron).
   Then the collapse keeps the mesh a manifold of the same topology.
2. No fold: each face that the collapse moves keeps a normal at less than 90
   degrees to its old one (and so a positive area).
3. Distance: every input vertex that the region around a stands for is within
   eps of one of the faces the collapse makes. Each face carries the input
   vertices removed in its region: a collapse gathers a and the points of the
   faces around a and gives each to its nearest new face. So every input
   vertex is an output vertex or within eps of the face that carries it.

A rejected collapse is tried again when the region around a or b changes.

Blocks: in a pass, each vertex belongs to the cube of side `block` that holds
it (the grid shifted by `offset` blocks). A vertex that shares a face with a
vertex of another block, or lies on the mesh's boundary, is locked, and an
edge collapses only when both its ends are unlocked. Then every face around
an unlocked vertex lies in its block: the blocks change disjoint faces and
read disjoint regions, so they run in parallel, and any order of them is one
serial sequence of valid collapses with the same guarantees. The locked bands
are reduced by the next pass, on the grid shifted by half a block. The result
depends on the block grid; the guarantees do not.

Each block runs compiled (decimate_block, numba). decimate_block_py is the same
worker in Python, the reference: the two give the same result bit for bit (the same
arithmetic in the same order, the faces round a vertex in increasing order), which
the tests check. The compiled worker queues each collapse once per update, not once
per vertex that it touches; a repeated entry would pop right after the first and
change nothing.

Memory: a pass makes its blocks one at a time and keeps a few in flight (twice the
workers), folding each result in as it comes; a block returns the quadrics of the
vertices it keeps only. The quadrics and the final distances are computed CHUNK at a
time, in the same order, so the results are the same bits. What stays whole is the
mesh's own arrays (positions, faces, quadrics) and its block and lock keys; a whole
brain would need the first pass read block by block from the mesh's files.
"""

import heapq
import os
import resource
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    as_completed,
    wait,
)
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from baillarger._jit import njit
from baillarger.mesh_smoothing import boundary_vertices

type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]
type Vec = tuple[float, float, float]
type Quad = tuple[float, ...]  # (xx, xy, xz, xw, yy, yz, yw, zz, zw, ww)


# ---------------------------------------------------------------- geometry
#: faces at a time in quadrics, and points at a time in carried_distances: the
#: temporary arrays scale with this, not with the mesh
CHUNK = 1 << 20


def _face_quadrics(v: Float, f: Int) -> Float:
    """(m, 10): each face's area times its plane quadric."""
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    a2 = np.linalg.norm(n, axis=1)
    ok = a2 > 0
    u = np.zeros_like(n)
    u[ok] = n[ok] / a2[ok, None]
    p = np.c_[u, -np.sum(u * v[f[:, 0]], axis=1)]  # (m, 4)
    w = a2 / 2
    iu = np.triu_indices(4)
    K: Float = w[:, None] * (p[:, iu[0]] * p[:, iu[1]])
    return K


def quadrics(v: Float, f: Int) -> Float:
    """Per vertex, the sum over its faces of area times the plane quadric
    (n, d)(n, d)^T of the face's plane n.x + d = 0, as its 10 upper-triangle
    entries (xx, xy, xz, xw, yy, yz, yw, zz, zw, ww). In chunks of faces, in
    the same order of additions as one whole-mesh pass (each corner in turn,
    faces in order), so the sums are the same bits."""
    Q = np.zeros((len(v), 10))
    for k in range(3):
        for a in range(0, len(f), CHUNK):
            fc = f[a : a + CHUNK]
            np.add.at(Q, fc[:, k], _face_quadrics(v, fc))
    return Q


def _cost(q: Quad, p: Vec) -> float:
    x, y, z = p
    return (
        q[0] * x * x
        + 2 * (q[1] * x * y + q[2] * x * z + q[3] * x)
        + q[4] * y * y
        + 2 * (q[5] * y * z + q[6] * y)
        + q[7] * z * z
        + 2 * q[8] * z
        + q[9]
    )


def _sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec, b: Vec) -> Vec:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def point_triangle_distance(p: Vec, a: Vec, b: Vec, c: Vec) -> float:
    """Distance from point p to triangle abc (Ericson, Real-Time Collision
    Detection, 5.1.5: the closest point by Voronoi region)."""
    ab, ac, ap = _sub(b, a), _sub(c, a), _sub(p, a)
    d1, d2 = _dot(ab, ap), _dot(ac, ap)
    if d1 <= 0 and d2 <= 0:
        q = a
    else:
        bp = _sub(p, b)
        d3, d4 = _dot(ab, bp), _dot(ac, bp)
        cp = _sub(p, c)
        d5, d6 = _dot(ab, cp), _dot(ac, cp)
        vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
        if d3 >= 0 and d4 <= d3:
            q = b
        elif d6 >= 0 and d5 <= d6:
            q = c
        elif vc <= 0 and d1 >= 0 and d3 <= 0:
            t = d1 / (d1 - d3)
            q = (a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2])
        elif vb <= 0 and d2 >= 0 and d6 <= 0:
            t = d2 / (d2 - d6)
            q = (a[0] + t * ac[0], a[1] + t * ac[1], a[2] + t * ac[2])
        elif va <= 0 and d4 - d3 >= 0 and d5 - d6 >= 0:
            t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            q = (
                b[0] + t * (c[0] - b[0]),
                b[1] + t * (c[1] - b[1]),
                b[2] + t * (c[2] - b[2]),
            )
        else:
            s = 1.0 / (va + vb + vc)
            v, w = vb * s, vc * s
            q = (
                a[0] + v * ab[0] + w * ac[0],
                a[1] + v * ab[1] + w * ac[1],
                a[2] + v * ab[2] + w * ac[2],
            )
    e = _sub(p, q)
    return float(np.sqrt(_dot(e, e)))


# ---------------------------------------------------------------- one block
@dataclass
class Block:
    """One block's part of a pass: its faces (indices into gids) and their
    vertices (global ids gids, positions P, locked, quadrics Q), and per face
    the input vertices it carries (CSR: cptr, cids, and their positions
    cpos)."""

    gids: Int
    P: Float
    locked: npt.NDArray[np.bool_]
    Q: Float
    F: Int
    cptr: Int
    cids: Int
    cpos: Float


@dataclass
class BlockResult:
    F: Int  # faces kept, indices into the block's gids
    Q: Float  # quadrics of the block's vertices (with qv: of those vertices)
    cptr: Int
    cids: Int
    collapses: int
    qv: Int | None = None  # the vertices that Q is of (None: all)


def decimate_block_py(B: Block, eps: float) -> BlockResult:
    """Collapse the block's edges whose ends are both unlocked, cheapest first,
    while a collapse passes the three tests (module docstring). The reference,
    in Python, for decimate_block: the faces round a vertex are taken in
    increasing order, which decides the nearest new face of a point that is
    equally near two, and the order of the points a face carries."""
    P: list[Vec] = [tuple(p) for p in B.P.tolist()]
    gid: list[int] = B.gids.tolist()
    locked: list[bool] = B.locked.tolist()
    Q: list[Quad] = [tuple(q) for q in B.Q.tolist()]
    F: list[list[int]] = B.F.tolist()
    ptr: list[int] = B.cptr.tolist()
    cid: list[int] = B.cids.tolist()
    cpos: list[Vec] = [tuple(p) for p in B.cpos.tolist()]
    carried: list[list[tuple[int, Vec]]] = [
        [(cid[k], cpos[k]) for k in range(ptr[i], ptr[i + 1])] for i in range(len(F))
    ]
    alive = [True] * len(F)
    vf: list[set[int]] = [set() for _ in P]
    for i, t in enumerate(F):
        for x in t:
            vf[x].add(i)
    stamp = [0] * len(P)
    heap: list[tuple[float, int, int, int, int]] = []

    def nbrs(x: int) -> set[int]:
        return {y for i in vf[x] for y in F[i]} - {x}

    def push(a: int, b: int) -> None:
        if not (locked[a] or locked[b]):
            c = _cost(Q[a], P[b]) + _cost(Q[b], P[b])
            heapq.heappush(heap, (c, a, b, stamp[a], stamp[b]))

    for a in range(len(P)):
        if not locked[a]:
            for b in nbrs(a):
                push(a, b)

    def has_face(x: int, y: int, z: int) -> bool:
        return any(y in F[i] and z in F[i] for i in vf[x])

    def collapse(a: int, b: int) -> bool:
        fa = sorted(vf[a])
        shared = [i for i in fa if b in F[i]]
        if len(shared) != 2:
            return False
        opp = {x for i in shared for x in F[i]} - {a, b}
        if len(opp) != 2 or nbrs(a) & nbrs(b) != opp:
            return False
        c, d = opp
        if has_face(a, c, d) and has_face(b, c, d):
            return False
        moved = [i for i in fa if i not in shared]
        new: list[tuple[Vec, Vec, Vec]] = []
        for i in moved:
            old = [P[x] for x in F[i]]
            tri = [P[b] if x == a else P[x] for x in F[i]]
            n0 = _cross(_sub(old[1], old[0]), _sub(old[2], old[0]))
            n1 = _cross(_sub(tri[1], tri[0]), _sub(tri[2], tri[0]))
            if _dot(n0, n1) <= 0:
                return False
            new.append((tri[0], tri[1], tri[2]))
        pts = [(gid[a], P[a])] + [p for i in fa for p in carried[i]]
        home: list[int] = []
        for _, p in pts:
            ds = [point_triangle_distance(p, *t) for t in new]
            k = min(range(len(ds)), key=ds.__getitem__)
            if ds[k] > eps:
                return False
            home.append(k)
        for i in shared:
            alive[i] = False
            for x in F[i]:
                vf[x].discard(i)
            carried[i] = []
        for i in moved:
            F[i] = [b if x == a else x for x in F[i]]
            vf[b].add(i)
            carried[i] = []
        for (g, p), k in zip(pts, home, strict=True):
            carried[moved[k]].append((g, p))
        vf[a] = set()
        return True

    n = 0
    while heap:
        _, a, b, sa, sb = heapq.heappop(heap)
        if sa != stamp[a] or sb != stamp[b] or not vf[a] or b not in nbrs(a):
            continue
        around = nbrs(a)
        if not collapse(a, b):
            continue
        n += 1
        Q[b] = tuple(x + y for x, y in zip(Q[a], Q[b], strict=True))
        changed = around | {b}
        for x in changed:
            stamp[x] += 1
        for x in changed:
            for y in nbrs(x):
                push(x, y)
                push(y, x)
    keep = [i for i in range(len(F)) if alive[i]]
    counts = [len(carried[i]) for i in keep]
    return BlockResult(
        F=np.array([F[i] for i in keep], np.int64).reshape(-1, 3),
        Q=np.array(Q, float).reshape(-1, 10),
        cptr=np.r_[0, np.cumsum(counts)].astype(np.int64),
        cids=np.array([g for i in keep for g, _ in carried[i]], np.int64),
        collapses=n,
    )


# ---------------------------------------------------------------- one block, compiled
# The same algorithm as decimate_block_py, on arrays, with the same arithmetic in
# the same order, so it gives the same bits. The faces round a vertex are kept in
# increasing order (vf, cnt); the points a face carries are a linked list of nodes
# in the order they were given to it (head, tail, nxt). Pushes after a collapse are
# deduplicated: a repeated entry pops right after the first and changes nothing.


@njit(cache=True)
def _nb_cost(Q: Float, a: int, p: Float, b: int) -> float:
    x, y, z = p[b, 0], p[b, 1], p[b, 2]
    q = Q[a]
    out: float = (
        q[0] * x * x
        + 2 * (q[1] * x * y + q[2] * x * z + q[3] * x)
        + q[4] * y * y
        + 2 * (q[5] * y * z + q[6] * y)
        + q[7] * z * z
        + 2 * q[8] * z
        + q[9]
    )
    return out


@njit(cache=True)
def _nb_pt(
    px: float,
    py: float,
    pz: float,
    ax: float,
    ay: float,
    az: float,
    bx: float,
    by: float,
    bz: float,
    cx: float,
    cy: float,
    cz: float,
) -> float:
    """point_triangle_distance on scalars."""
    abx, aby, abz = bx - ax, by - ay, bz - az
    acx, acy, acz = cx - ax, cy - ay, cz - az
    apx, apy, apz = px - ax, py - ay, pz - az
    d1 = abx * apx + aby * apy + abz * apz
    d2 = acx * apx + acy * apy + acz * apz
    if d1 <= 0 and d2 <= 0:
        qx, qy, qz = ax, ay, az
    else:
        bpx, bpy, bpz = px - bx, py - by, pz - bz
        d3 = abx * bpx + aby * bpy + abz * bpz
        d4 = acx * bpx + acy * bpy + acz * bpz
        cpx, cpy, cpz = px - cx, py - cy, pz - cz
        d5 = abx * cpx + aby * cpy + abz * cpz
        d6 = acx * cpx + acy * cpy + acz * cpz
        vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
        if d3 >= 0 and d4 <= d3:
            qx, qy, qz = bx, by, bz
        elif d6 >= 0 and d5 <= d6:
            qx, qy, qz = cx, cy, cz
        elif vc <= 0 and d1 >= 0 and d3 <= 0:
            t = d1 / (d1 - d3)
            qx, qy, qz = ax + t * abx, ay + t * aby, az + t * abz
        elif vb <= 0 and d2 >= 0 and d6 <= 0:
            t = d2 / (d2 - d6)
            qx, qy, qz = ax + t * acx, ay + t * acy, az + t * acz
        elif va <= 0 and d4 - d3 >= 0 and d5 - d6 >= 0:
            t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            qx = bx + t * (cx - bx)
            qy = by + t * (cy - by)
            qz = bz + t * (cz - bz)
        else:
            s = 1.0 / (va + vb + vc)
            v, w = vb * s, vc * s
            qx = ax + v * abx + w * acx
            qy = ay + v * aby + w * acy
            qz = az + v * abz + w * acz
    ex, ey, ez = px - qx, py - qy, pz - qz
    return float(np.sqrt(ex * ex + ey * ey + ez * ez))


@njit(cache=True)
def _nb_nbrs(x: int, F: Int, vf: Int, cnt: Int) -> Int:
    """The neighbours of vertex x (any order, each once)."""
    out = np.empty(3 * cnt[x], np.int64)
    k = 0
    for j in range(cnt[x]):
        for y in F[vf[x, j]]:
            if y == x:
                continue
            seen = False
            for t in range(k):
                if out[t] == y:
                    seen = True
                    break
            if not seen:
                out[k] = y
                k += 1
    return out[:k]


@njit(cache=True)
def _nb_has(arr: Int, y: int) -> bool:
    for t in range(arr.size):
        if arr[t] == y:
            return True
    return False


@njit(cache=True)
def _nb_has_face(x: int, y: int, z: int, F: Int, vf: Int, cnt: Int) -> bool:
    for j in range(cnt[x]):
        t = F[vf[x, j]]
        if (t[0] == y or t[1] == y or t[2] == y) and (
            t[0] == z or t[1] == z or t[2] == z
        ):
            return True
    return False


@njit(cache=True)
def _nb_remove(x: int, i: int, vf: Int, cnt: Int) -> None:
    for j in range(cnt[x]):
        if vf[x, j] == i:
            for t in range(j, cnt[x] - 1):
                vf[x, t] = vf[x, t + 1]
            cnt[x] -= 1
            return


@njit(cache=True)
def _nb_insert(x: int, i: int, vf: Int, cnt: Int) -> None:
    j = cnt[x]
    while j > 0 and vf[x, j - 1] > i:
        vf[x, j] = vf[x, j - 1]
        j -= 1
    vf[x, j] = i
    cnt[x] += 1


@njit(cache=True)
def _nb_collapse(
    a: int,
    b: int,
    eps: float,
    P: Float,
    gid: Int,
    F: Int,
    alive: npt.NDArray[np.bool_],
    vf: Int,
    cnt: Int,
    head: Int,
    tail: Int,
    nxt: Int,
    node_g: Int,
    node_p: Float,
    nn: Int,
) -> bool:
    """collapse() of decimate_block_py: a into b if the three tests pass."""
    na = cnt[a]
    fa = vf[a, :na].copy()
    shared = np.empty(na, np.int64)
    ns = 0
    for j in range(na):
        t = F[fa[j]]
        if t[0] == b or t[1] == b or t[2] == b:
            shared[ns] = fa[j]
            ns += 1
    if ns != 2:
        return False
    opp = np.empty(6, np.int64)
    no = 0
    for j in range(2):
        for y in F[shared[j]]:
            if y != a and y != b and not _nb_has(opp[:no], y):
                opp[no] = y
                no += 1
    if no != 2:
        return False
    Na, Nb = _nb_nbrs(a, F, vf, cnt), _nb_nbrs(b, F, vf, cnt)
    common = 0
    for y in Na:
        if _nb_has(Nb, y):
            if not _nb_has(opp[:2], y):
                return False
            common += 1
    if common != 2:
        return False
    c, d = opp[0], opp[1]
    if _nb_has_face(a, c, d, F, vf, cnt) and _nb_has_face(b, c, d, F, vf, cnt):
        return False
    moved = np.empty(na - 2, np.int64)
    nm = 0
    for j in range(na):
        if fa[j] != shared[0] and fa[j] != shared[1]:
            moved[nm] = fa[j]
            nm += 1
    tri = np.empty((nm, 3, 3))
    for m in range(nm):
        t = F[moved[m]]
        o0, o1, o2 = P[t[0]], P[t[1]], P[t[2]]
        for r in range(3):
            src = b if t[r] == a else t[r]
            tri[m, r, 0], tri[m, r, 1], tri[m, r, 2] = P[src, 0], P[src, 1], P[src, 2]
        e1x, e1y, e1z = o1[0] - o0[0], o1[1] - o0[1], o1[2] - o0[2]
        e2x, e2y, e2z = o2[0] - o0[0], o2[1] - o0[1], o2[2] - o0[2]
        n0x, n0y, n0z = (
            e1y * e2z - e1z * e2y,
            e1z * e2x - e1x * e2z,
            e1x * e2y - e1y * e2x,
        )
        T = tri[m]
        f1x, f1y, f1z = T[1, 0] - T[0, 0], T[1, 1] - T[0, 1], T[1, 2] - T[0, 2]
        f2x, f2y, f2z = T[2, 0] - T[0, 0], T[2, 1] - T[0, 1], T[2, 2] - T[0, 2]
        n1x, n1y, n1z = (
            f1y * f2z - f1z * f2y,
            f1z * f2x - f1x * f2z,
            f1x * f2y - f1y * f2x,
        )
        if n0x * n1x + n0y * n1y + n0z * n1z <= 0:
            return False
    # the points: a (a new node), then the nodes of a's faces in order
    npts = 1
    for j in range(na):
        k = head[fa[j]]
        while k >= 0:
            npts += 1
            k = nxt[k]
    pts = np.empty(npts, np.int64)
    pa = nn[0]
    node_g[pa] = gid[a]
    node_p[pa, 0], node_p[pa, 1], node_p[pa, 2] = P[a, 0], P[a, 1], P[a, 2]
    pts[0] = pa
    q = 1
    for j in range(na):
        k = head[fa[j]]
        while k >= 0:
            pts[q] = k
            q += 1
            k = nxt[k]
    home = np.empty(npts, np.int64)
    for q in range(npts):
        p = node_p[pts[q]]
        best, bk = 0.0, -1
        for m in range(nm):
            T = tri[m]
            dm = _nb_pt(
                p[0],
                p[1],
                p[2],
                T[0, 0],
                T[0, 1],
                T[0, 2],
                T[1, 0],
                T[1, 1],
                T[1, 2],
                T[2, 0],
                T[2, 1],
                T[2, 2],
            )
            if bk < 0 or dm < best:
                best, bk = dm, m
        if best > eps:
            return False
        home[q] = bk
    # commit
    nn[0] += 1
    for j in range(2):
        i = shared[j]
        alive[i] = False
        for x in F[i]:
            _nb_remove(x, i, vf, cnt)
        head[i] = -1
        tail[i] = -1
    for m in range(nm):
        i = moved[m]
        for r in range(3):
            if F[i, r] == a:
                F[i, r] = b
        _nb_insert(b, i, vf, cnt)
        head[i] = -1
        tail[i] = -1
    for q in range(npts):
        i, k = moved[home[q]], pts[q]
        nxt[k] = -1
        if head[i] < 0:
            head[i] = k
        else:
            nxt[tail[i]] = k
        tail[i] = k
    cnt[a] = 0
    return True


@njit(cache=True)
def _nb_grow(vf: Int, need: int) -> Int:
    out = -np.ones((vf.shape[0], max(need, 2 * vf.shape[1])), np.int64)
    out[:, : vf.shape[1]] = vf
    return out


@njit(cache=True)
def _nb_block(
    P: Float,
    gid: Int,
    locked: npt.NDArray[np.bool_],
    Q0: Float,
    F0: Int,
    cptr: Int,
    cids: Int,
    cpos: Float,
    eps: float,
) -> tuple[Int, Float, Int, Int, int, int]:
    """decimate_block on arrays. Returns faces kept, quadrics, the carried CSR,
    the collapses and the pushes."""
    n, m = P.shape[0], F0.shape[0]
    F, Q = F0.copy(), Q0.copy()
    alive = np.ones(m, np.bool_)
    cnt = np.zeros(n, np.int64)
    for i in range(m):
        for x in F[i]:
            cnt[x] += 1
    vf = -np.ones((n, max(16, 2 * (cnt.max() if n else 0))), np.int64)
    cnt[:] = 0
    for i in range(m):
        for x in F[i]:
            vf[x, cnt[x]] = i
            cnt[x] += 1
    cap = cids.size + n
    node_g = np.empty(cap, np.int64)
    node_p = np.empty((cap, 3))
    nxt = -np.ones(cap, np.int64)
    head = -np.ones(m, np.int64)
    tail = -np.ones(m, np.int64)
    nn = np.zeros(1, np.int64)
    for i in range(m):
        for k in range(cptr[i], cptr[i + 1]):
            j = nn[0]
            nn[0] += 1
            node_g[j] = cids[k]
            node_p[j] = cpos[k]
            if head[i] < 0:
                head[i] = j
            else:
                nxt[tail[i]] = j
            tail[i] = j
    stamp = np.zeros(n, np.int64)
    heap = [(0.0, 0, 0, 0, 0)]
    heap.pop()
    pushes = 0
    for a in range(n):
        if not locked[a]:
            for b in _nb_nbrs(a, F, vf, cnt):
                if not locked[b]:
                    c = _nb_cost(Q, a, P, b) + _nb_cost(Q, b, P, b)
                    heapq.heappush(heap, (c, a, b, stamp[a], stamp[b]))
                    pushes += 1
    ncol = 0
    while len(heap):
        _, a, b, sa, sb = heapq.heappop(heap)
        if sa != stamp[a] or sb != stamp[b] or cnt[a] == 0:
            continue
        around = _nb_nbrs(a, F, vf, cnt)
        if not _nb_has(around, b):
            continue
        if cnt[b] + cnt[a] > vf.shape[1]:
            vf = _nb_grow(vf, cnt[b] + cnt[a])
        if not _nb_collapse(
            a, b, eps, P, gid, F, alive, vf, cnt, head, tail, nxt, node_g, node_p, nn
        ):
            continue
        ncol += 1
        for r in range(10):
            Q[b, r] = Q[a, r] + Q[b, r]
        changed = np.empty(around.size + 1, np.int64)
        changed[: around.size] = around
        changed[around.size] = b
        for x in changed:
            stamp[x] += 1
        # every (x, y) and (y, x) for x changed, y its neighbour: each once
        parts = []
        for x in changed:
            ys = _nb_nbrs(int(x), F, vf, cnt)
            k2 = np.empty(2 * ys.size, np.int64)
            k2[: ys.size] = x * n + ys
            k2[ys.size :] = ys * n + x
            parts.append(k2)
        tot = 0
        for k2 in parts:
            tot += k2.size
        keys = np.empty(tot, np.int64)
        o = 0
        for k2 in parts:
            keys[o : o + k2.size] = k2
            o += k2.size
        keys = np.unique(keys)
        for key in keys:
            x, y = key // n, key % n
            if not (locked[x] or locked[y]):
                c = _nb_cost(Q, x, P, y) + _nb_cost(Q, y, P, y)
                heapq.heappush(heap, (c, x, y, stamp[x], stamp[y]))
                pushes += 1
    nk = 0
    for i in range(m):
        if alive[i]:
            nk += 1
    Fk = np.empty((nk, 3), np.int64)
    ptr = np.zeros(nk + 1, np.int64)
    r = 0
    for i in range(m):
        if alive[i]:
            Fk[r] = F[i]
            k = head[i]
            c2 = 0
            while k >= 0:
                c2 += 1
                k = nxt[k]
            ptr[r + 1] = ptr[r] + c2
            r += 1
    out_c = np.empty(ptr[nk], np.int64)
    r = 0
    for i in range(m):
        if alive[i]:
            k, o = head[i], ptr[r]
            while k >= 0:
                out_c[o] = node_g[k]
                o += 1
                k = nxt[k]
            r += 1
    return Fk, Q, ptr, out_c, ncol, pushes


def decimate_block(B: Block, eps: float) -> BlockResult:
    """decimate_block_py compiled (numba): the same result, bit for bit."""
    F, Q, cptr, cids, n, _ = _nb_block(
        np.ascontiguousarray(B.P, np.float64),
        np.ascontiguousarray(B.gids, np.int64),
        np.ascontiguousarray(B.locked, np.bool_),
        np.ascontiguousarray(B.Q, np.float64),
        np.ascontiguousarray(B.F, np.int64),
        np.ascontiguousarray(B.cptr, np.int64),
        np.ascontiguousarray(B.cids, np.int64),
        np.ascontiguousarray(B.cpos, np.float64).reshape(-1, 3),
        float(eps),
    )
    return BlockResult(F=F, Q=Q, cptr=cptr, cids=cids, collapses=int(n))


def _run_block(args: tuple[Block, float, bool]) -> BlockResult:
    """A block in a pass: its result with the quadrics of the vertices it keeps
    only (a removed vertex's quadric is never read again)."""
    B, eps, reference = args
    r = decimate_block_py(B, eps) if reference else decimate_block(B, eps)
    qv = np.unique(r.F)
    return BlockResult(
        F=r.F, Q=r.Q[qv], cptr=r.cptr, cids=r.cids, collapses=r.collapses, qv=qv
    )


# ---------------------------------------------------------------- passes
def _peak_gb() -> float:
    """This process's peak resident memory so far, GB (as baillarger.timing, which
    this module does not import: it brings PyTorch into every worker)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(r) / (2**30 if sys.platform == "darwin" else 2**20)


class _Laps:
    """Each call logs the step that just ended: its time and the peak memory."""

    def __init__(self, log: Callable[[str], object] | None) -> None:
        self.log, self.t = log, time.time()

    def __call__(self, name: str) -> None:
        if self.log is not None:
            now = time.time()
            self.log(f"  {name:<22} {now - self.t:7.1f} s   peak {_peak_gb():5.1f} GB")
            self.t = now


def _csr_take(ptr: Int, data: Int, rows: Int) -> tuple[Int, Int]:
    """The rows `rows` of a CSR list (ptr, data), as a new CSR list."""
    lens = ptr[rows + 1] - ptr[rows]
    out_ptr = np.r_[0, np.cumsum(lens)].astype(np.int64)
    idx = np.repeat(ptr[rows] - out_ptr[:-1], lens) + np.arange(out_ptr[-1])
    return out_ptr, data[idx]


def _csr_concat(parts: Sequence[tuple[Int, Int]]) -> tuple[Int, Int]:
    ptrs, datas, base = [np.zeros(1, np.int64)], [], 0
    for p, d in parts:
        ptrs.append(p[1:] + base)
        datas.append(d)
        base += int(p[-1])
    return np.concatenate(ptrs), np.concatenate(datas or [np.zeros(0, np.int64)])


@dataclass
class Mesh:
    """A mesh during decimation: input positions v (never changed), faces f
    (into v), quadrics Q per input vertex, and per face the input vertices it
    carries (cptr, cids)."""

    v: Float
    f: Int
    Q: Float
    cptr: Int
    cids: Int


def decimate_pass(
    M: Mesh,
    eps: float,
    block: float,
    offset: float = 0.0,
    origin: Float | None = None,
    workers: int = 1,
    reference: bool = False,
    log: Callable[[str], object] | None = None,
) -> tuple[Mesh, dict[str, Any]]:
    """One pass: the blocks of the grid of side `block` (from origin, default
    the minimum of v, shifted by offset blocks), each decimated on its own,
    in parallel with workers > 1; reference: with the Python worker. log: each
    step's time and peak memory."""
    lap = _Laps(log)
    v, f = M.v, M.f
    o = v.min(0) if origin is None else np.asarray(origin, float)
    g = np.floor((v - o) / block + offset).astype(np.int64)
    g -= g.min(0)
    n = g.max(0) + 1
    key = (g[:, 0] * n[1] + g[:, 1]) * n[2] + g[:, 2]
    fk = key[f]
    inb = (fk[:, 0] == fk[:, 1]) & (fk[:, 1] == fk[:, 2])
    locked = boundary_vertices(f, len(v))
    locked[f[~inb]] = True
    rows = np.flatnonzero(inb)
    order = rows[np.argsort(fk[rows, 0], kind="stable")]
    cuts = np.flatnonzero(np.diff(fk[order, 0])) + 1
    lap("blocks and locks")
    kept = [np.flatnonzero(~inb)]

    def blocks() -> Iterator[tuple[Int, Block]]:
        """The blocks with an unlocked vertex, made one at a time; the others
        are kept as they are."""
        for idx in np.split(order, cuts) if len(order) else []:
            gids, inv = np.unique(f[idx], return_inverse=True)
            if locked[gids].all():
                kept.append(idx)
                continue
            cptr, cids = _csr_take(M.cptr, M.cids, idx)
            yield (
                gids,
                Block(
                    gids=gids,
                    P=v[gids],
                    locked=locked[gids],
                    Q=M.Q[gids],
                    F=inv.reshape(-1, 3).astype(np.int64),
                    cptr=cptr,
                    cids=cids,
                    cpos=v[cids],
                ),
            )

    # each block's result, by its place in the order (gids, result); the
    # quadrics go into Q as results come, so that only a few blocks' data are
    # held at once
    out: dict[int, tuple[Int, BlockResult]] = {}
    Q = M.Q.copy()

    def fold(j: int, gids: Int, r: BlockResult) -> None:
        Q[gids if r.qv is None else gids[r.qv]] = r.Q
        out[j] = (
            gids,
            BlockResult(r.F, np.zeros((0, 10)), r.cptr, r.cids, r.collapses),
        )

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            pending: dict[Future[BlockResult], tuple[int, Int]] = {}
            for j, (gids, B) in enumerate(blocks()):
                pending[ex.submit(_run_block, (B, eps, reference))] = (j, gids)
                del B
                while len(pending) >= 2 * workers:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for fu in done:
                        fold(*pending.pop(fu), fu.result())
            for fu in as_completed(pending):
                fold(*pending[fu], fu.result())
    else:
        for j, (gids, B) in enumerate(blocks()):
            fold(j, gids, _run_block((B, eps, reference)))
    lap("run blocks")
    results = [out[j] for j in range(len(out))]
    keep = np.sort(np.concatenate(kept))
    f_new = np.concatenate([f[keep]] + [g[r.F] for g, r in results])
    cptr, cids = _csr_concat(
        [_csr_take(M.cptr, M.cids, keep)] + [(r.cptr, r.cids) for _, r in results]
    )
    lap("join")
    summary = dict(
        blocks=len(results),
        faces_before=len(f),
        faces_after=len(f_new),
        collapses=sum(r.collapses for _, r in results),
        locked=int(locked.sum()),
    )
    return Mesh(v=v, f=f_new.astype(np.int64), Q=Q, cptr=cptr, cids=cids), summary


# ---------------------------------------------------------------- whole run
@dataclass
class Decimated:
    """The output: vertex_ids (the input vertex of each output vertex), faces
    (into vertex_ids), per face the input vertices it carries (cptr, cids:
    input ids), the largest distance from a carried vertex to its face, and a
    summary per pass."""

    vertex_ids: Int
    faces: Int
    cptr: Int
    cids: Int
    max_distance: float
    passes: list[dict[str, Any]] = field(default_factory=list)


def carried_distances(v: Float, f: Int, cptr: Int, cids: Int) -> Float:
    """Per carried input vertex, its distance to the face (f, into v) that
    carries it (CHUNK points at a time)."""
    face = np.repeat(np.arange(len(f)), np.diff(cptr))
    out = np.empty(len(cids))
    for a in range(0, len(cids), CHUNK):
        fc, t = face[a : a + CHUNK], f[face[a : a + CHUNK]]
        out[a : a + len(fc)] = triangle_distances(
            v[cids[a : a + CHUNK]], v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
        )
    return out


def triangle_distances(p: Float, a: Float, b: Float, c: Float) -> Float:
    """Row-wise distance from points p to triangles abc (all (k, 3)): the
    vectorised form of point_triangle_distance."""

    def dot(x: Float, y: Float) -> Float:
        out: Float = np.einsum("ij,ij->i", x, y)
        return out

    ab, ac, ap, bp, cp = b - a, c - a, p - a, p - b, p - c
    d1, d2, d3, d4, d5, d6 = (
        dot(ab, ap),
        dot(ac, ap),
        dot(ab, bp),
        dot(ac, bp),
        dot(ab, cp),
        dot(ac, cp),
    )
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ab = d1 / (d1 - d3)
        t_ac = d2 / (d2 - d6)
        t_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        s = 1.0 / (va + vb + vc)
    q = a + (vb * s)[:, None] * ab + (vc * s)[:, None] * ac
    regions = [
        ((d1 <= 0) & (d2 <= 0), a),
        ((d3 >= 0) & (d4 <= d3), b),
        ((d6 >= 0) & (d5 <= d6), c),
        ((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + t_ab[:, None] * ab),
        ((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + t_ac[:, None] * ac),
        ((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), b + t_bc[:, None] * (c - b)),
    ]
    done = np.zeros(len(p), bool)
    for m, x in regions:
        m = m & ~done
        q[m] = x[m]
        done |= m
    out: Float = np.linalg.norm(p - q, axis=1)
    return out


def decimate(
    v: npt.ArrayLike,
    f: npt.ArrayLike,
    eps: float,
    block: float,
    passes: int = 2,
    workers: int | None = None,
    log: Callable[[str], object] = print,
    reference: bool = False,
) -> Decimated:
    """Decimate mesh (v, f) within eps (in v's units) on blocks of side
    `block`: `passes` passes, on the grid shifted by half a block in every
    other pass, each with `workers` processes (default: the CPU count)."""
    lap = _Laps(log)
    V = np.asarray(v, np.float64)
    Fa = np.asarray(f, np.int64)
    M = Mesh(
        v=V,
        f=Fa,
        Q=quadrics(V, Fa),
        cptr=np.zeros(len(Fa) + 1, np.int64),
        cids=np.zeros(0, np.int64),
    )
    lap("quadrics")
    nw = (os.cpu_count() or 1) if workers is None else workers
    origin = V.min(0)
    out: list[dict[str, Any]] = []
    for k in range(passes):
        t0 = time.time()
        M, s = decimate_pass(
            M, eps, block, 0.5 * (k % 2), origin, nw, reference, log=log
        )
        s["seconds"] = round(time.time() - t0, 2)
        out.append(s)
        log(
            f"decimate pass {k + 1}/{passes}: {s['blocks']} blocks, "
            f"{s['faces_before']:,} -> {s['faces_after']:,} faces "
            f"({s['seconds']} s)"
        )
        lap.t = time.time()
    vid = np.unique(M.f)
    d = carried_distances(V, M.f, M.cptr, M.cids)
    lap("distances")
    return Decimated(
        vertex_ids=vid,
        faces=np.searchsorted(vid, M.f).astype(np.int64),
        cptr=M.cptr,
        cids=M.cids,
        max_distance=float(d.max()) if d.size else 0.0,
        passes=out,
    )
