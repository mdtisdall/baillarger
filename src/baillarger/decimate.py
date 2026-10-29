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
"""

import heapq
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from baillarger.mesh_smoothing import boundary_vertices

type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]
type Vec = tuple[float, float, float]
type Quad = tuple[float, ...]  # (xx, xy, xz, xw, yy, yz, yw, zz, zw, ww)


# ---------------------------------------------------------------- geometry
def quadrics(v: Float, f: Int) -> Float:
    """Per vertex, the sum over its faces of area times the plane quadric
    (n, d)(n, d)^T of the face's plane n.x + d = 0, as its 10 upper-triangle
    entries (xx, xy, xz, xw, yy, yz, yw, zz, zw, ww)."""
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    a2 = np.linalg.norm(n, axis=1)
    ok = a2 > 0
    u = np.zeros_like(n)
    u[ok] = n[ok] / a2[ok, None]
    p = np.c_[u, -np.sum(u * v[f[:, 0]], axis=1)]  # (m, 4)
    w = a2 / 2
    iu = np.triu_indices(4)
    K = w[:, None] * (p[:, iu[0]] * p[:, iu[1]])  # (m, 10)
    Q = np.zeros((len(v), 10))
    for k in range(3):
        np.add.at(Q, f[:, k], K)
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
    Q: Float  # quadrics of the block's vertices
    cptr: Int
    cids: Int
    collapses: int


def decimate_block(B: Block, eps: float) -> BlockResult:
    """Collapse the block's edges whose ends are both unlocked, cheapest first,
    while a collapse passes the three tests (module docstring)."""
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
        shared = [i for i in vf[a] if b in F[i]]
        if len(shared) != 2:
            return False
        opp = {x for i in shared for x in F[i]} - {a, b}
        if len(opp) != 2 or nbrs(a) & nbrs(b) != opp:
            return False
        c, d = opp
        if has_face(a, c, d) and has_face(b, c, d):
            return False
        moved = [i for i in vf[a] if i not in shared]
        new: list[tuple[Vec, Vec, Vec]] = []
        for i in moved:
            old = [P[x] for x in F[i]]
            tri = [P[b] if x == a else P[x] for x in F[i]]
            n0 = _cross(_sub(old[1], old[0]), _sub(old[2], old[0]))
            n1 = _cross(_sub(tri[1], tri[0]), _sub(tri[2], tri[0]))
            if _dot(n0, n1) <= 0:
                return False
            new.append((tri[0], tri[1], tri[2]))
        pts = [(gid[a], P[a])] + [p for i in vf[a] for p in carried[i]]
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


def _run_block(args: tuple[Block, float]) -> BlockResult:
    return decimate_block(*args)


# ---------------------------------------------------------------- passes
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
) -> tuple[Mesh, dict[str, Any]]:
    """One pass: the blocks of the grid of side `block` (from origin, default
    the minimum of v, shifted by offset blocks), each decimated on its own,
    in parallel with workers > 1."""
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
    jobs: list[Block] = []
    kept = [np.flatnonzero(~inb)]
    for idx in np.split(order, cuts) if len(order) else []:
        gids, inv = np.unique(f[idx], return_inverse=True)
        if locked[gids].all():
            kept.append(idx)
            continue
        cptr, cids = _csr_take(M.cptr, M.cids, idx)
        jobs.append(
            Block(
                gids=gids,
                P=v[gids],
                locked=locked[gids],
                Q=M.Q[gids],
                F=inv.reshape(-1, 3).astype(np.int64),
                cptr=cptr,
                cids=cids,
                cpos=v[cids],
            )
        )
    args = [(b, eps) for b in jobs]
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(
                ex.map(_run_block, args, chunksize=max(1, len(jobs) // (4 * workers)))
            )
    else:
        results = [_run_block(a) for a in args]
    keep = np.sort(np.concatenate(kept))
    Q = M.Q.copy()
    for b, r in zip(jobs, results, strict=True):
        Q[b.gids] = r.Q
    f_new = np.concatenate(
        [f[keep]] + [b.gids[r.F] for b, r in zip(jobs, results, strict=True)]
    )
    cptr, cids = _csr_concat(
        [_csr_take(M.cptr, M.cids, keep)] + [(r.cptr, r.cids) for r in results]
    )
    summary = dict(
        blocks=len(jobs),
        faces_before=len(f),
        faces_after=len(f_new),
        collapses=sum(r.collapses for r in results),
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
    carries it."""
    face = np.repeat(np.arange(len(f)), np.diff(cptr))
    return triangle_distances(v[cids], v[f[face, 0]], v[f[face, 1]], v[f[face, 2]])


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
) -> Decimated:
    """Decimate mesh (v, f) within eps (in v's units) on blocks of side
    `block`: `passes` passes, on the grid shifted by half a block in every
    other pass, each with `workers` processes (default: the CPU count)."""
    V = np.asarray(v, np.float64)
    Fa = np.asarray(f, np.int64)
    M = Mesh(
        v=V,
        f=Fa,
        Q=quadrics(V, Fa),
        cptr=np.zeros(len(Fa) + 1, np.int64),
        cids=np.zeros(0, np.int64),
    )
    nw = (os.cpu_count() or 1) if workers is None else workers
    origin = V.min(0)
    out: list[dict[str, Any]] = []
    for k in range(passes):
        t0 = time.time()
        M, s = decimate_pass(M, eps, block, 0.5 * (k % 2), origin, nw)
        s["seconds"] = round(time.time() - t0, 2)
        out.append(s)
        log(
            f"decimate pass {k + 1}/{passes}: {s['blocks']} blocks, "
            f"{s['faces_before']:,} -> {s['faces_after']:,} faces "
            f"({s['seconds']} s)"
        )
    vid = np.unique(M.f)
    d = carried_distances(V, M.f, M.cptr, M.cids)
    return Decimated(
        vertex_ids=vid,
        faces=np.searchsorted(vid, M.f).astype(np.int64),
        cptr=M.cptr,
        cids=M.cids,
        max_distance=float(d.max()) if d.size else 0.0,
        passes=out,
    )
