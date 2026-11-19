"""Validity checks of a triangle mesh, shared by the WM side (decimate; the
white_surf tests, which check its guarantee by construction) and the τ = 0 top
surface Σ.

The checks, each per face (or per vertex for the gap) with a count of pairs:

- intersections: two faces with no common vertex that cross
  (baillarger.mesh_intersect.intersecting_faces). That test skips every pair
  of faces that share a vertex, by design, so two more tests cover them:
- shared-vertex folds: two faces that share exactly one vertex and cross
  beyond it (the edge opposite the shared vertex in one face crosses the
  other face; Möller-Trumbore, ends and edges within a relative tolerance not
  counted). Faces that lie in one plane are not found here.
- shared-edge folds (creases): two faces that share an edge and are folded
  back on each other, the cosine of the angle between their normals at most
  a cut-off (default FOLD_COS = -0.9, 154°; the user's choice of
  2026-11-18). A fold in one plane (a vertex pushed past its 1-ring in the
  plane) has opposed normals and is found here.
- turned faces: a face whose normal is at 90 degrees or more to a reference
  direction per face: the face's own normal in a previous state (turned
  since the last state) or a fixed reference (turned against, for example,
  the input surface that a face stands for), as decimate's tests 2 and 4.
- the gap: a point closer than `gap` to a face of another part of the
  surface (a face with no vertex in the point's local patch: the faces
  reached from the point's own faces across shared vertices, each closer
  than 2 × gap to the point).

check_mesh runs them on a whole mesh. Checker keeps a live copy of a mesh
that changes by local edits (moved vertices; faces and vertices added and
removed) and checks each edit in time proportional to the faces near it, not
to the mesh: an edit is proposed, checked, then committed or rolled back.
link_condition is the edge-collapse test of exactly two common neighbours.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import sparse

from baillarger import mesh_intersect as MI
from baillarger._jit import njit, prange

Float = npt.NDArray[np.float64]
Int = npt.NDArray[np.int64]
Bool = npt.NDArray[np.bool_]

#: shared-edge folds: the cosine of the angle between the two faces' normals
#: at or below which they count as folded back (cos 154.16°; user, 2026-11-18)
FOLD_COS = -0.9
#: shared-vertex folds: the relative tolerance of the segment-triangle test
SEG_TOL = 1e-6
#: Checker: above this share of changed vertices, the checks run on the
#: whole mesh (speed only)
LOCAL_SHARE = 0.25


# ------------------------------------------------------------ topology


def edge_keys(F: Int, n: int, directed: bool = False) -> tuple[Int, Int]:
    """The distinct edges as keys a × n + b (a < b, or both directions when
    directed), sorted, and how often each occurs. 1-D keys: much faster to
    sort than rows."""
    a = np.r_[F[:, 0], F[:, 1], F[:, 2]].astype(np.int64)
    b = np.r_[F[:, 1], F[:, 2], F[:, 0]].astype(np.int64)
    if directed:
        a, b = np.r_[a, b], np.r_[b, a]
    else:
        a, b = np.minimum(a, b), np.maximum(a, b)
    k, c = np.unique(a * n + b, return_counts=True)
    return k.astype(np.int64), c.astype(np.int64)


def vertex_neighbours(F: Int, n: int) -> tuple[Int, Int]:
    """The vertices' neighbours as CSR (ptr, nbr), each row sorted."""
    k, _ = edge_keys(F, n, directed=True)
    ptr = np.zeros(n + 1, np.int64)
    np.add.at(ptr, k // n + 1, 1)
    return np.cumsum(ptr), k % n


def vertex_faces(F: Int, n: int) -> tuple[Int, Int]:
    """The faces round each vertex as CSR (vf_ptr, vf), each row in
    increasing face order."""
    cnt = np.bincount(F.ravel(), minlength=n)
    ptr = np.r_[0, np.cumsum(cnt)].astype(np.int64)
    vf = (np.argsort(F.ravel(), kind="stable") // 3).astype(np.int64)
    return ptr, vf


def boundary_vertices(F: Int, n: int) -> Bool:
    """Vertices on an edge with one face."""
    k, c = edge_keys(F, n)
    b = np.zeros(n, bool)
    kb = k[c == 1]
    b[kb // n] = True
    b[kb % n] = True
    return b


@dataclass(frozen=True)
class Topology:
    """A mesh's faces with its adjacency: neighbours (ptr, nbr), faces round
    each vertex (vf_ptr, vf) and the boundary vertices."""

    F: Int
    n: int
    ptr: Int
    nbr: Int
    vf_ptr: Int
    vf: Int
    boundary: Bool

    @classmethod
    def of(cls, F: npt.ArrayLike, n: int) -> "Topology":
        """The topology of faces F on n vertices."""
        Fi = np.ascontiguousarray(F, np.int64)
        ptr, nbr = vertex_neighbours(Fi, n)
        vf_ptr, vf = vertex_faces(Fi, n)
        return cls(Fi, n, ptr, nbr, vf_ptr, vf, boundary_vertices(Fi, n))


def edge_face_pairs(F: Int, n: int) -> tuple[Int, Int]:
    """The edges with exactly two faces: their vertices (k, 2), a < b,
    sorted, and their two faces (k, 2), in increasing order. Edges with one
    face (the boundary) or more than two (not a manifold) are left out."""
    a = np.r_[F[:, 0], F[:, 1], F[:, 2]].astype(np.int64)
    b = np.r_[F[:, 1], F[:, 2], F[:, 0]].astype(np.int64)
    face = np.tile(np.arange(len(F), dtype=np.int64), 3)
    key = np.minimum(a, b) * n + np.maximum(a, b)
    o = np.argsort(key, kind="stable")
    key, face = key[o], face[o]
    # the runs of equal keys, and their lengths
    start = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    length = np.diff(np.r_[start, len(key)])
    i = start[length == 2]
    k = key[i]
    E = np.c_[k // n, k % n].astype(np.int64)
    P = np.sort(np.c_[face[i], face[i + 1]], axis=1).astype(np.int64)
    return E, P


def link_condition(ptr: Int, nbr: Int, E: Int) -> Bool:
    """Per edge E[k] = (a, b): a and b have exactly two common neighbours
    (an interior edge whose collapse keeps the mesh a manifold, but for a
    tetrahedron, which this test does not see). Vectorised as a sparse
    product of the adjacency rows."""
    n = len(ptr) - 1
    A = sparse.csr_matrix((np.ones(len(nbr), np.int32), nbr, ptr), shape=(n, n))
    common = np.asarray(A[E[:, 0]].multiply(A[E[:, 1]]).sum(1)).ravel()
    out: Bool = common == 2
    return out


# ------------------------------------------------------------ normals


def face_normals(V: Float, F: Int) -> Float:
    """The faces' cross products (twice the area times the unit normal)."""
    out: Float = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    return out


def vertex_normals(V: Float, F: Int) -> Float:
    """Area-weighted unit vertex normals (the faces' orientation)."""
    cr = face_normals(V, F)
    N = np.zeros((len(V), 3))
    for k in range(3):
        np.add.at(N, F[:, k], cr)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    return N


# ------------------------------------------------------------ pair tests


@njit(cache=True, inline="always")
def _ptd_xyz(
    px: float, py: float, pz: float,
    ax: float, ay: float, az: float,
    bx: float, by: float, bz: float,
    cx: float, cy: float, cz: float,
) -> float:  # fmt: skip
    """Distance from the point p to the triangle a, b, c (Ericson, closest
    point), in scalars: no temporary arrays."""
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
        if d3 >= 0 and d4 <= d3:
            qx, qy, qz = bx, by, bz
        else:
            vc = d1 * d4 - d3 * d2
            if vc <= 0 and d1 >= 0 and d3 <= 0:
                t = d1 / (d1 - d3)
                qx, qy, qz = ax + t * abx, ay + t * aby, az + t * abz
            else:
                cpx, cpy, cpz = px - cx, py - cy, pz - cz
                d5 = abx * cpx + aby * cpy + abz * cpz
                d6 = acx * cpx + acy * cpy + acz * cpz
                if d6 >= 0 and d5 <= d6:
                    qx, qy, qz = cx, cy, cz
                else:
                    vb = d5 * d2 - d1 * d6
                    if vb <= 0 and d2 >= 0 and d6 <= 0:
                        t = d2 / (d2 - d6)
                        qx, qy, qz = ax + t * acx, ay + t * acy, az + t * acz
                    else:
                        va = d3 * d6 - d5 * d4
                        if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
                            t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
                            qx = bx + t * (cx - bx)
                            qy = by + t * (cy - by)
                            qz = bz + t * (cz - bz)
                        else:
                            den = 1.0 / (va + vb + vc)
                            v_, w_ = vb * den, vc * den
                            qx = ax + abx * v_ + acx * w_
                            qy = ay + aby * v_ + acy * w_
                            qz = az + abz * v_ + acz * w_
    dx, dy, dz = px - qx, py - qy, pz - qz
    d: float = np.sqrt(dx * dx + dy * dy + dz * dz)
    return d


@njit(cache=True, inline="always")
def _ptd(V: Float, p: int, ia: int, ib: int, ic: int) -> float:
    """Distance from vertex p to the triangle of vertices ia, ib, ic."""
    return _ptd_xyz(
        V[p, 0], V[p, 1], V[p, 2], V[ia, 0], V[ia, 1], V[ia, 2],
        V[ib, 0], V[ib, 1], V[ib, 2], V[ic, 0], V[ic, 1], V[ic, 2],
    )  # fmt: skip


@njit(cache=True, inline="always")
def _seg_tri(V: Float, ip: int, iq: int, ia: int, ib: int, ic: int, tol: float) -> bool:
    """The segment of vertices ip, iq crosses the triangle ia, ib, ic
    (Möller-Trumbore); points within tol (relative, in the barycentric and
    segment parameters) of an edge or an end do not count. Scalars only."""
    dx, dy, dz = V[iq, 0] - V[ip, 0], V[iq, 1] - V[ip, 1], V[iq, 2] - V[ip, 2]
    ax, ay, az = V[ia, 0], V[ia, 1], V[ia, 2]
    e1x, e1y, e1z = V[ib, 0] - ax, V[ib, 1] - ay, V[ib, 2] - az
    e2x, e2y, e2z = V[ic, 0] - ax, V[ic, 1] - ay, V[ic, 2] - az
    pvx, pvy, pvz = dy * e2z - dz * e2y, dz * e2x - dx * e2z, dx * e2y - dy * e2x
    det = e1x * pvx + e1y * pvy + e1z * pvz
    if abs(det) < 1e-300:
        return False
    inv = 1.0 / det
    tx, ty, tz = V[ip, 0] - ax, V[ip, 1] - ay, V[ip, 2] - az
    u = (tx * pvx + ty * pvy + tz * pvz) * inv
    if u <= tol or u >= 1 - tol:
        return False
    qx, qy, qz = ty * e1z - tz * e1y, tz * e1x - tx * e1z, tx * e1y - ty * e1x
    v = (dx * qx + dy * qy + dz * qz) * inv
    if v <= tol or u + v >= 1 - tol:
        return False
    t = (e2x * qx + e2y * qy + e2z * qz) * inv
    hit: bool = tol < t < 1 - tol
    return hit


# ------------------------------------------------------------ shared-vertex folds


@njit(cache=True, parallel=True)
def _foldover_kernel(
    V: Float, F: Int, vf_ptr: Int, vf: Int, xs: Int, tol: float, out: Bool
) -> int:
    """Pairs of faces round each vertex x of xs that share exactly one vertex
    (x) and cross each other beyond it: the edge opposite x in one face
    crosses the other face. Marks both faces in out (every thread writes the
    same value); returns the number of pairs (each is found at its one
    shared vertex, so once)."""
    cnt = np.zeros(len(xs), np.int64)
    for ix in prange(len(xs)):
        x = xs[ix]
        for i in range(vf_ptr[x], vf_ptr[x + 1]):
            f = vf[i]
            for j in range(i + 1, vf_ptr[x + 1]):
                g = vf[j]
                sh = 0
                for k in range(3):
                    for m in range(3):
                        if F[f, k] == F[g, m]:
                            sh += 1
                if sh != 1:
                    continue
                # the edge of f opposite x, and of g
                k = 0
                while F[f, k] != x:
                    k += 1
                f0, f1 = F[f, (k + 1) % 3], F[f, (k + 2) % 3]
                k = 0
                while F[g, k] != x:
                    k += 1
                g0, g1 = F[g, (k + 1) % 3], F[g, (k + 2) % 3]
                hit = _seg_tri(V, f0, f1, F[g, 0], F[g, 1], F[g, 2], tol)
                if not hit:
                    hit = _seg_tri(V, g0, g1, F[f, 0], F[f, 1], F[f, 2], tol)
                if hit:
                    out[f] = True
                    out[g] = True
                    cnt[ix] += 1
    total = int(cnt.sum())
    return total


def near_vertices(ptr: Int, nbr: Int, moved: Bool) -> Bool:
    """The moved vertices and their neighbours."""
    near = moved.copy()
    near[nbr[np.repeat(moved, np.diff(ptr))]] = True
    return near


def shared_vertex_folds(
    V: npt.ArrayLike,
    F: npt.ArrayLike,
    top: Topology | None = None,
    moved: Bool | None = None,
    tol: float = SEG_TOL,
) -> tuple[Bool, int]:
    """Faces that share exactly one vertex with another face and cross it
    beyond that vertex (mesh_intersect skips these pairs): a mask over F and
    the number of pairs. With `moved`, only at the moved vertices and their
    neighbours (a fold needs a moved vertex in one of its faces)."""
    Vf = np.ascontiguousarray(V, np.float64)
    Fi = np.ascontiguousarray(F, np.int64)
    out = np.zeros(len(Fi), bool)
    if moved is None:
        xs = np.arange(len(Vf), dtype=np.int64)
        vf_ptr, vf = (top.vf_ptr, top.vf) if top else vertex_faces(Fi, len(Vf))
    else:
        t = Topology.of(Fi, len(Vf)) if top is None else top
        xs = np.flatnonzero(near_vertices(t.ptr, t.nbr, moved)).astype(np.int64)
        vf_ptr, vf = t.vf_ptr, t.vf
    n = _foldover_kernel(Vf, Fi, vf_ptr, vf, xs, tol, out)
    return out, int(n)


# ------------------------------------------------------------ shared-edge folds


def shared_edge_folds(
    V: npt.ArrayLike,
    F: npt.ArrayLike,
    fold_cos: float = FOLD_COS,
    pairs: Int | None = None,
) -> tuple[Bool, int]:
    """The dihedral test: faces that share an edge with a face folded back
    on them, the cosine of the angle between the two normals at most
    fold_cos (-1: flat back on each other). A mask over F and the number of
    such edges. pairs: the edge-face pairs to test (edge_face_pairs' second
    array, or some of its rows); default all. A face of zero area is in no
    fold here (turned_faces counts it)."""
    Vf = np.asarray(V, np.float64)
    Fi = np.ascontiguousarray(F, np.int64)
    P = edge_face_pairs(Fi, len(Vf))[1] if pairs is None else pairs
    N = face_normals(Vf, Fi)
    L = np.linalg.norm(N, axis=1)
    a, b = P[:, 0], P[:, 1]
    num = np.einsum("ij,ij->i", N[a], N[b])
    den = L[a] * L[b]
    fold = (den > 0) & (num <= fold_cos * den)
    out = np.zeros(len(Fi), bool)
    out[a[fold]] = True
    out[b[fold]] = True
    return out, int(fold.sum())


# ------------------------------------------------------------ turned faces


def turned_faces(V: npt.ArrayLike, F: npt.ArrayLike, reference: Float) -> Bool:
    """Faces whose normal is at 90 degrees or more to reference[f] (a
    direction per face, any length): with the face normals of a previous
    state, the faces turned since that state; with a fixed reference, the
    faces turned against it. A face of zero area counts as turned (as in
    decimate's tests 2 and 4)."""
    N = face_normals(np.asarray(V, np.float64), np.asarray(F, np.int64))
    out: Bool = np.einsum("ij,ij->i", N, reference) <= 0
    return out


# ------------------------------------------------------------ the live mesh
#
# Checker keeps its own copy of the mesh as arrays that change in place:
#
# - V (capacity × 3) and F (capacity × 3); a free face has F[f] = -1, a free
#   vertex has no corner (vhead[v] = -1). Free ids are on stacks (vfree,
#   ffree) and are used again.
# - the faces round each vertex: corner c = 3 f + k (vertex F[f, k]) in a
#   doubly linked list per vertex (vhead, cnext, cprev).
# - a grid of the face centroids, cells of size `cell`, as a hash table of
#   the occupied cells (hkey, hhead; open addressing, linear probes) with a
#   doubly linked list of faces per cell (gnext, gprev; fcell = the face's
#   slot).
#
# Each operation costs O(1) per face it touches (expected, for the hash), so
# an edit and its check cost O(the faces near the edit), not O(n).

#: cell coordinates: 21 bits per axis, offset so that -2**20 .. 2**20 - 1 fit
_KOFF = 1 << 20
_KMAX = (1 << 21) - 1
#: Fibonacci hashing: 2**64 / the golden ratio
_HMUL = np.uint64(11400714819323198485)
#: the counters array: high-water marks of vertex and face ids, sizes of the
#: free stacks, occupied hash slots, the last marker
_NV, _NF, _NVFREE, _NFFREE, _HUSED, _MARK = 0, 1, 2, 3, 4, 5
#: the live mesh as one tuple for the kernels (see Checker._M)
type _Live = tuple[
    Float, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Float, Float
]
#: the marker arrays (see Checker._K)
type _Marks = tuple[Int, Int, Int, Int, Int, Int, Int, Int, Int, Int]
#: the parameters array: cell size, largest face radius (a bound), gap, the
#: patch radius, the crease cut-off
_CELL, _RMAX, _GAP, _RLOC, _FOLD = 0, 1, 2, 3, 4


@njit(cache=True, inline="always")
def _q(x: float, lo: float, cell: float) -> int:
    """The cell coordinate of x (offset, clipped to the key range)."""
    q = int(np.floor((x - lo) / cell)) + _KOFF
    return min(max(q, 0), _KMAX)


@njit(cache=True, inline="always")
def _pack(i: int, j: int, k: int) -> int:
    """The key of cell (i, j, k)."""
    return (i << 42) | (j << 21) | k


@njit(cache=True, inline="always")
def _hslot(hkey: Int, key: int) -> int:
    """The slot that holds key, or the empty slot where it goes."""
    mask = len(hkey) - 1
    s = int((np.uint64(key) * _HMUL) >> np.uint64(32)) & mask
    while hkey[s] != key and hkey[s] != -1:
        s = (s + 1) & mask
    return s


@njit(cache=True, inline="always")
def _centre(V: Float, F: Int, f: int) -> tuple[float, float, float]:
    """The centroid of face f."""
    a, b, c = F[f, 0], F[f, 1], F[f, 2]
    return (
        (V[a, 0] + V[b, 0] + V[c, 0]) / 3.0,
        (V[a, 1] + V[b, 1] + V[c, 1]) / 3.0,
        (V[a, 2] + V[b, 2] + V[c, 2]) / 3.0,
    )


@njit(cache=True, inline="always")
def _radius(V: Float, F: Int, f: int) -> float:
    """The largest distance from face f's centroid to its vertices."""
    cx, cy, cz = _centre(V, F, f)
    r = 0.0
    for k in range(3):
        v = F[f, k]
        dx, dy, dz = V[v, 0] - cx, V[v, 1] - cy, V[v, 2] - cz
        r = max(r, np.sqrt(dx * dx + dy * dy + dz * dz))
    return r


@njit(cache=True)
def _glink(
    V: Float, F: Int, fcell: Int, gnext: Int, gprev: Int, hkey: Int,
    hhead: Int, cnt: Int, prm: Float, lo: Float, f: int,
) -> None:  # fmt: skip
    """Put face f in the grid cell of its centroid; raise the radius bound."""
    cell = prm[_CELL]
    cx, cy, cz = _centre(V, F, f)
    key = _pack(_q(cx, lo[0], cell), _q(cy, lo[1], cell), _q(cz, lo[2], cell))
    s = _hslot(hkey, key)
    if hkey[s] == -1:
        hkey[s] = key
        hhead[s] = -1
        cnt[_HUSED] += 1
    h = hhead[s]
    gprev[f] = -1
    gnext[f] = h
    if h >= 0:
        gprev[h] = f
    hhead[s] = f
    fcell[f] = s
    r = _radius(V, F, f)
    if r > prm[_RMAX]:
        prm[_RMAX] = r


@njit(cache=True, inline="always")
def _gunlink(fcell: Int, gnext: Int, gprev: Int, hhead: Int, f: int) -> None:
    """Take face f out of its grid cell (the cell's key stays)."""
    p, n = gprev[f], gnext[f]
    if p >= 0:
        gnext[p] = n
    else:
        hhead[fcell[f]] = n
    if n >= 0:
        gprev[n] = p
    fcell[f] = -1


@njit(cache=True, inline="always")
def _clink(F: Int, vhead: Int, cnext: Int, cprev: Int, f: int) -> None:
    """Add face f's corners to its vertices' lists."""
    for k in range(3):
        c = 3 * f + k
        v = F[f, k]
        h = vhead[v]
        cprev[c] = -1
        cnext[c] = h
        if h >= 0:
            cprev[h] = c
        vhead[v] = c


@njit(cache=True, inline="always")
def _cunlink(F: Int, vhead: Int, cnext: Int, cprev: Int, f: int) -> None:
    """Take face f's corners out of its vertices' lists."""
    for k in range(3):
        c = 3 * f + k
        p, n = cprev[c], cnext[c]
        if p >= 0:
            cnext[p] = n
        else:
            vhead[F[f, k]] = n
        if n >= 0:
            cprev[n] = p


@njit(cache=True)
def _link_all(
    V: Float, F: Int, vhead: Int, cnext: Int, cprev: Int, fcell: Int,
    gnext: Int, gprev: Int, hkey: Int, hhead: Int, cnt: Int, prm: Float,
    lo: Float, corners: bool,
) -> None:  # fmt: skip
    """Put every face (F[f, 0] ≥ 0, f < the high-water mark) in the grid,
    and (corners) in the vertices' lists."""
    for f in range(cnt[_NF]):
        if F[f, 0] < 0:
            continue
        if corners:
            _clink(F, vhead, cnext, cprev, f)
        _glink(V, F, fcell, gnext, gprev, hkey, hhead, cnt, prm, lo, f)


@njit(cache=True)
def _links_of(F: Int, vhead: Int, cnext: Int, vs: Int) -> int:
    """The number of faces round the vertices vs (with repeats): a bound on
    the grid links that moving them makes."""
    n = 0
    for v in vs:
        c = vhead[v]
        while c >= 0:
            n += 1
            c = cnext[c]
    return n


@njit(cache=True)
def _apply(
    M: _Live,
    mv: Int, to: Float, rm: Int, add: Int, addv: Float,
) -> tuple[Int, Int, Float, Int, Float]:  # fmt: skip
    """Make an edit in place: new vertices at addv, faces rm removed, the
    vertices mv moved to `to`, faces `add` added (a vertex -1 - j there is
    new vertex j). Returns the new vertex and face ids, the moved vertices'
    old positions, the removed faces' rows, and the old triangles of the
    removed faces and of the faces round the moved vertices (k × 3 × 3, for
    the region the edit changed)."""
    (
        V,
        F,
        vhead,
        cnext,
        cprev,
        fcell,
        gnext,
        gprev,
        hkey,
        hhead,
        vfree,
        ffree,
        cnt,
        prm,
        lo,
    ) = M
    newv = np.empty(len(addv), np.int64)
    for j in range(len(addv)):
        if cnt[_NVFREE] > 0:
            cnt[_NVFREE] -= 1
            v = vfree[cnt[_NVFREE]]
        else:
            v = cnt[_NV]
            cnt[_NV] += 1
        V[v] = addv[j]
        vhead[v] = -1
        newv[j] = v
    # the old triangles
    nt = len(rm) + _links_of(F, vhead, cnext, mv)
    Told = np.empty((nt, 3, 3))
    t = 0
    for f in rm:
        for k in range(3):
            Told[t, k] = V[F[f, k]]
        t += 1
    for v in mv:
        c = vhead[v]
        while c >= 0:
            g = c // 3
            for k in range(3):
                Told[t, k] = V[F[g, k]]
            t += 1
            c = cnext[c]
    rows = np.empty((len(rm), 3), np.int64)
    for i in range(len(rm)):
        f = rm[i]
        rows[i] = F[f]
        _gunlink(fcell, gnext, gprev, hhead, f)
        _cunlink(F, vhead, cnext, cprev, f)
        F[f] = -1
    old = np.empty((len(mv), 3))
    for i in range(len(mv)):
        v = mv[i]
        old[i] = V[v]
        V[v] = to[i]
        c = vhead[v]
        while c >= 0:
            g = c // 3
            _gunlink(fcell, gnext, gprev, hhead, g)
            _glink(V, F, fcell, gnext, gprev, hkey, hhead, cnt, prm, lo, g)
            c = cnext[c]
    newf = np.empty(len(add), np.int64)
    for i in range(len(add)):
        if cnt[_NFFREE] > 0:
            cnt[_NFFREE] -= 1
            f = ffree[cnt[_NFFREE]]
        else:
            f = cnt[_NF]
            cnt[_NF] += 1
        for k in range(3):
            u = add[i, k]
            F[f, k] = newv[-1 - u] if u < 0 else u
        _clink(F, vhead, cnext, cprev, f)
        _glink(V, F, fcell, gnext, gprev, hkey, hhead, cnt, prm, lo, f)
        newf[i] = f
    return newv, newf, old, rows, Told


@njit(cache=True)
def _undo(
    M: _Live,
    mv: Int, old: Float, rm: Int, rows: Int, newv: Int, newf: Int,
) -> None:  # fmt: skip
    """Undo _apply: remove the new faces, put the moved vertices back,
    restore the removed faces (same ids) and free the new vertices."""
    (
        V,
        F,
        vhead,
        cnext,
        cprev,
        fcell,
        gnext,
        gprev,
        hkey,
        hhead,
        vfree,
        ffree,
        cnt,
        prm,
        lo,
    ) = M
    for f in newf[::-1]:
        _gunlink(fcell, gnext, gprev, hhead, f)
        _cunlink(F, vhead, cnext, cprev, f)
        F[f] = -1
        ffree[cnt[_NFFREE]] = f
        cnt[_NFFREE] += 1
    for i in range(len(mv) - 1, -1, -1):
        v = mv[i]
        V[v] = old[i]
        c = vhead[v]
        while c >= 0:
            g = c // 3
            _gunlink(fcell, gnext, gprev, hhead, g)
            _glink(V, F, fcell, gnext, gprev, hkey, hhead, cnt, prm, lo, g)
            c = cnext[c]
    for i in range(len(rm) - 1, -1, -1):
        f = rm[i]
        F[f] = rows[i]
        _clink(F, vhead, cnext, cprev, f)
        _glink(V, F, fcell, gnext, gprev, hkey, hhead, cnt, prm, lo, f)
    for v in newv[::-1]:
        vhead[v] = -1
        vfree[cnt[_NVFREE]] = v
        cnt[_NVFREE] += 1


@njit(cache=True)
def _settle(
    M: _Live,
    rm: Int, rows: Int, newv: Int,
) -> Int:  # fmt: skip
    """Keep an edit: the removed faces' ids are free, and so are the
    vertices with no face left (of the removed faces, or new). Returns the
    freed vertices."""
    (
        V,
        F,
        vhead,
        cnext,
        cprev,
        fcell,
        gnext,
        gprev,
        hkey,
        hhead,
        vfree,
        ffree,
        cnt,
        prm,
        lo,
    ) = M
    for f in rm:
        ffree[cnt[_NFFREE]] = f
        cnt[_NFFREE] += 1
    out = [int(x) for x in range(0)]
    for i in range(len(rows) + len(newv)):
        for k in range(3 if i < len(rows) else 1):
            v = rows[i, k] if i < len(rows) else newv[i - len(rows)]
            if vhead[v] != -1:
                continue
            vhead[v] = -2  # freed: not seen again in this loop
            out.append(v)
    res = np.empty(len(out), np.int64)
    for i in range(len(out)):
        v = out[i]
        vhead[v] = -1
        V[v] = np.nan
        vfree[cnt[_NVFREE]] = v
        cnt[_NVFREE] += 1
        res[i] = v
    return res


# ------------------------------------------------------------ the local checks


@njit(cache=True)
def _patch(
    p: int, V: Float, F: Int, vhead: Int, cnext: Int, rloc: float,
    stamp: Int, mark: int, queue: Int,
) -> None:  # fmt: skip
    """Stamp p's local patch with mark: the faces reached from p's own faces
    across shared vertices, each closer than rloc to p."""
    stamp[p] = mark
    qh, qt = 0, 1
    queue[0] = p
    while qh < qt:
        u = queue[qh]
        qh += 1
        c = vhead[u]
        while c >= 0:
            g = c // 3
            c = cnext[c]
            a, b, cc = F[g, 0], F[g, 1], F[g, 2]
            if stamp[a] == mark and stamp[b] == mark and stamp[cc] == mark:
                continue
            if _ptd(V, p, a, b, cc) >= rloc:
                continue
            for w in (a, b, cc):
                if stamp[w] != mark:
                    stamp[w] = mark
                    queue[qt] = w
                    qt += 1


@njit(cache=True)
def _check_kernel(
    M: _Live,
    K: _Marks,
    cv: Int, addf: Int, Told: Float, full: bool, cross: bool, want_dmin: bool,
) -> tuple[Int, Int, int, Int, int, int, Int, int, Int, Float]:  # fmt: skip
    """The checks of the live mesh near an edit.

    Changed faces: addf and the faces round the changed vertices cv. The
    affected vertices: cv and every vertex closer than the patch radius
    (rloc = 2 × gap) to a changed face (now) or to a triangle of Told (the
    faces as they were): a vertex's local patch and its distances to faces
    change only there. full: every vertex and face instead.

    - the gap, at each affected vertex p: the faces closer than gap that are
      not in p's local patch (a pair (p, face) counts once per vertex).
    - cross (not with full): the intersecting pairs with a changed face
      (mesh_intersect's pair test, pairs that share a vertex skipped), and
      the shared-vertex folds round the vertices of the changed faces.
    - creases: the edges of the changed faces whose two faces are folded
      back past the cut-off (edges with one face, or more than two, are
      left out).

    Returns: bad vertices (gap or cross), facing vertices (gap pairs whose
    normals are opposed), the number of gap pairs, the cross faces, the
    numbers of intersecting pairs and of shared-vertex folds, the crease
    faces and the number of creases, and the affected vertices with dmin
    (want_dmin: the distance to the nearest face of another part among the
    candidates, inf if none)."""
    (
        V,
        F,
        vhead,
        cnext,
        cprev,
        fcell,
        gnext,
        gprev,
        hkey,
        hhead,
        vfree,
        ffree,
        cnt,
        prm,
        lo,
    ) = M
    vA, vbad, vfac, vx, stamp, ring, queue, fC, fbad, fcr = K
    cell, Rmax, gap, rloc, fold_cos = (
        prm[_CELL],
        prm[_RMAX],
        prm[_GAP],
        prm[_RLOC],
        prm[_FOLD],
    )
    cnt[_MARK] += 1
    m = cnt[_MARK]
    # the changed faces
    cf = [int(x) for x in range(0)]
    if full:
        for f in range(cnt[_NF]):
            if F[f, 0] >= 0:
                fC[f] = m
                cf.append(f)
    else:
        for f in addf:
            if F[f, 0] >= 0 and fC[f] != m:
                fC[f] = m
                cf.append(f)
        for v in cv:
            c = vhead[v]
            while c >= 0:
                g = c // 3
                c = cnext[c]
                if fC[g] != m:
                    fC[g] = m
                    cf.append(g)
    # the affected vertices
    av = [int(x) for x in range(0)]
    if full:
        for v in range(cnt[_NV]):
            if vhead[v] >= 0:
                vA[v] = m
                av.append(v)
    else:
        for v in cv:
            if vhead[v] >= 0 and vA[v] != m:
                vA[v] = m
                av.append(v)
        nT = len(Told) + len(cf)
        for t in range(nT):
            if t < len(Told):
                ax, ay, az = Told[t, 0, 0], Told[t, 0, 1], Told[t, 0, 2]
                bx, by, bz = Told[t, 1, 0], Told[t, 1, 1], Told[t, 1, 2]
                qx, qy, qz = Told[t, 2, 0], Told[t, 2, 1], Told[t, 2, 2]
            else:
                f = cf[t - len(Told)]
                a, b, c = F[f, 0], F[f, 1], F[f, 2]
                ax, ay, az = V[a, 0], V[a, 1], V[a, 2]
                bx, by, bz = V[b, 0], V[b, 1], V[b, 2]
                qx, qy, qz = V[c, 0], V[c, 1], V[c, 2]
            ox, oy, oz = (
                (ax + bx + qx) / 3.0,
                (ay + by + qy) / 3.0,
                (az + bz + qz) / 3.0,
            )
            r = 0.0
            for px, py, pz in ((ax, ay, az), (bx, by, bz), (qx, qy, qz)):
                r = max(r, np.sqrt((px - ox) ** 2 + (py - oy) ** 2 + (pz - oz) ** 2))
            # a vertex within rloc of the triangle: one of its faces has its
            # centroid within rloc + r + Rmax of the triangle's centroid
            rad = int(np.ceil((rloc + r + Rmax) / cell))
            i0, j0, k0 = _q(ox, lo[0], cell), _q(oy, lo[1], cell), _q(oz, lo[2], cell)
            for i in range(max(i0 - rad, 0), min(i0 + rad, _KMAX) + 1):
                for j in range(max(j0 - rad, 0), min(j0 + rad, _KMAX) + 1):
                    for k in range(max(k0 - rad, 0), min(k0 + rad, _KMAX) + 1):
                        key = _pack(i, j, k)
                        s = _hslot(hkey, key)
                        if hkey[s] != key:
                            continue
                        g = hhead[s]
                        while g >= 0:
                            for w in range(3):
                                u = F[g, w]
                                if vA[u] == m:
                                    continue
                                d = _ptd_xyz(
                                    V[u, 0], V[u, 1], V[u, 2], ax, ay, az,
                                    bx, by, bz, qx, qy, qz,
                                )  # fmt: skip
                                if d < rloc:
                                    vA[u] = m
                                    av.append(u)
                            g = gnext[g]
    # the gap at each affected vertex
    bad = [int(x) for x in range(0)]
    fac = [int(x) for x in range(0)]
    dmin = np.full(len(av), np.inf)
    ng = 0
    radg = int(np.ceil((gap + Rmax) / cell))
    for ip in range(len(av)):
        p = av[ip]
        cnt[_MARK] += 1
        mp = cnt[_MARK]
        ring[p] = mp
        c = vhead[p]
        while c >= 0:
            g = c // 3
            c = cnext[c]
            ring[F[g, 0]] = mp
            ring[F[g, 1]] = mp
            ring[F[g, 2]] = mp
        i0, j0, k0 = (
            _q(V[p, 0], lo[0], cell),
            _q(V[p, 1], lo[1], cell),
            _q(V[p, 2], lo[2], cell),
        )
        built = False
        nx = 0.0
        ny = 0.0
        nz = 0.0
        normal = False
        for _rep in range(2):  # pass 0: is any face near? pass 1: with the patch
            near = False
            for i in range(max(i0 - radg, 0), min(i0 + radg, _KMAX) + 1):
                for j in range(max(j0 - radg, 0), min(j0 + radg, _KMAX) + 1):
                    for k in range(max(k0 - radg, 0), min(k0 + radg, _KMAX) + 1):
                        key = _pack(i, j, k)
                        s = _hslot(hkey, key)
                        if hkey[s] != key:
                            continue
                        f = hhead[s]
                        while f >= 0:
                            g = f
                            f = gnext[f]
                            a, b, cc = F[g, 0], F[g, 1], F[g, 2]
                            if a == p or b == p or cc == p:
                                continue
                            if built and (
                                stamp[a] == mp or stamp[b] == mp or stamp[cc] == mp
                            ):
                                continue
                            if (
                                not built
                                and not want_dmin
                                and (ring[a] == mp or ring[b] == mp or ring[cc] == mp)
                            ):
                                continue
                            d = _ptd(V, p, a, b, cc)
                            if not built:
                                if d < gap or want_dmin:
                                    near = True
                                continue
                            if d < dmin[ip]:
                                dmin[ip] = d
                            if d < gap:
                                ng += 1
                                for w in (p, a, b, cc):
                                    if vbad[w] != m:
                                        vbad[w] = m
                                        bad.append(w)
                                if not normal:
                                    cc2 = vhead[p]
                                    while cc2 >= 0:
                                        h = cc2 // 3
                                        cc2 = cnext[cc2]
                                        e0, e1, e2 = F[h, 0], F[h, 1], F[h, 2]
                                        ux = V[e1, 0] - V[e0, 0]
                                        uy = V[e1, 1] - V[e0, 1]
                                        uz = V[e1, 2] - V[e0, 2]
                                        wx = V[e2, 0] - V[e0, 0]
                                        wy = V[e2, 1] - V[e0, 1]
                                        wz = V[e2, 2] - V[e0, 2]
                                        nx += uy * wz - uz * wy
                                        ny += uz * wx - ux * wz
                                        nz += ux * wy - uy * wx
                                    normal = True
                                ux = V[b, 0] - V[a, 0]
                                uy = V[b, 1] - V[a, 1]
                                uz = V[b, 2] - V[a, 2]
                                wx = V[cc, 0] - V[a, 0]
                                wy = V[cc, 1] - V[a, 1]
                                wz = V[cc, 2] - V[a, 2]
                                dot = (
                                    nx * (uy * wz - uz * wy)
                                    + ny * (uz * wx - ux * wz)
                                    + nz * (ux * wy - uy * wx)
                                )
                                if dot < 0:
                                    for w in (p, a, b, cc):
                                        if vfac[w] != m:
                                            vfac[w] = m
                                            fac.append(w)
            if built or not near:
                break
            _patch(p, V, F, vhead, cnext, rloc, stamp, mp, queue)
            built = True
    # crossings and shared-vertex folds
    xf = [int(x) for x in range(0)]
    ni = 0
    nf = 0
    if cross:
        A = np.empty((1, 3, 3))
        B = np.empty((1, 3, 3))
        radt = int(np.ceil(2 * Rmax / cell))
        for f in cf:
            for w in range(3):
                A[0, w] = V[F[f, w]]
            ox, oy, oz = _centre(V, F, f)
            i0, j0, k0 = _q(ox, lo[0], cell), _q(oy, lo[1], cell), _q(oz, lo[2], cell)
            for i in range(max(i0 - radt, 0), min(i0 + radt, _KMAX) + 1):
                for j in range(max(j0 - radt, 0), min(j0 + radt, _KMAX) + 1):
                    for k in range(max(k0 - radt, 0), min(k0 + radt, _KMAX) + 1):
                        key = _pack(i, j, k)
                        s = _hslot(hkey, key)
                        if hkey[s] != key:
                            continue
                        g = hhead[s]
                        while g >= 0:
                            h = g
                            g = gnext[g]
                            if h == f or (fC[h] == m and h < f):
                                continue  # each pair once
                            sh = False
                            for a in range(3):
                                for b in range(3):
                                    if F[f, a] == F[h, b]:
                                        sh = True
                            if sh:
                                continue
                            ov = True
                            for a in range(3):
                                flo = min(A[0, 0, a], A[0, 1, a], A[0, 2, a])
                                fhi = max(A[0, 0, a], A[0, 1, a], A[0, 2, a])
                                glo = min(V[F[h, 0], a], V[F[h, 1], a], V[F[h, 2], a])
                                ghi = max(V[F[h, 0], a], V[F[h, 1], a], V[F[h, 2], a])
                                if fhi < glo or ghi < flo:
                                    ov = False
                            if not ov:
                                continue
                            for w in range(3):
                                B[0, w] = V[F[h, w]]
                            if MI._tt(A, 0, B, 0, 1e-4):
                                ni += 1
                                for e in (f, h):
                                    if fbad[e] != m:
                                        fbad[e] = m
                                        xf.append(e)
        # shared-vertex folds round the vertices of the changed faces
        for f in cf:
            for w in range(3):
                x = F[f, w]
                if vx[x] == m:
                    continue
                vx[x] = m
                ci = vhead[x]
                while ci >= 0:
                    fi = ci // 3
                    cj = cnext[ci]
                    while cj >= 0:
                        gj = cj // 3
                        cj = cnext[cj]
                        shared = 0
                        for k in range(3):
                            for k2 in range(3):
                                if F[fi, k] == F[gj, k2]:
                                    shared += 1
                        if shared != 1:
                            continue
                        k = 0
                        while F[fi, k] != x:
                            k += 1
                        f0, f1 = F[fi, (k + 1) % 3], F[fi, (k + 2) % 3]
                        k = 0
                        while F[gj, k] != x:
                            k += 1
                        g0, g1 = F[gj, (k + 1) % 3], F[gj, (k + 2) % 3]
                        hit = _seg_tri(V, f0, f1, F[gj, 0], F[gj, 1], F[gj, 2], SEG_TOL)
                        if not hit:
                            hit = _seg_tri(
                                V, g0, g1, F[fi, 0], F[fi, 1], F[fi, 2], SEG_TOL
                            )
                        if hit:
                            nf += 1
                            for e in (fi, gj):
                                if fbad[e] != m:
                                    fbad[e] = m
                                    xf.append(e)
                    ci = cnext[ci]
        for e in xf:
            for w in range(3):
                u = F[e, w]
                if vbad[u] != m:
                    vbad[u] = m
                    bad.append(u)
    # creases on the edges of the changed faces
    crf = [int(x) for x in range(0)]
    ncr = 0
    for f in cf:
        for w in range(3):
            a, b = F[f, w], F[f, (w + 1) % 3]
            other = -1
            nface = 1
            c = vhead[a]
            while c >= 0:
                h = c // 3
                c = cnext[c]
                if h == f:
                    continue
                if F[h, 0] == b or F[h, 1] == b or F[h, 2] == b:
                    nface += 1
                    other = h
            if nface != 2 or (fC[other] == m and other < f):
                continue
            # the two normals (cross products)
            e0, e1, e2 = F[f, 0], F[f, 1], F[f, 2]
            ux, uy, uz = V[e1, 0] - V[e0, 0], V[e1, 1] - V[e0, 1], V[e1, 2] - V[e0, 2]
            wx, wy, wz = V[e2, 0] - V[e0, 0], V[e2, 1] - V[e0, 1], V[e2, 2] - V[e0, 2]
            n0, n1, n2 = uy * wz - uz * wy, uz * wx - ux * wz, ux * wy - uy * wx
            e0, e1, e2 = F[other, 0], F[other, 1], F[other, 2]
            ux, uy, uz = V[e1, 0] - V[e0, 0], V[e1, 1] - V[e0, 1], V[e1, 2] - V[e0, 2]
            wx, wy, wz = V[e2, 0] - V[e0, 0], V[e2, 1] - V[e0, 1], V[e2, 2] - V[e0, 2]
            m0, m1, m2 = uy * wz - uz * wy, uz * wx - ux * wz, ux * wy - uy * wx
            den = np.sqrt(n0 * n0 + n1 * n1 + n2 * n2) * np.sqrt(
                m0 * m0 + m1 * m1 + m2 * m2
            )
            if den > 0 and n0 * m0 + n1 * m1 + n2 * m2 <= fold_cos * den:
                ncr += 1
                for e in (f, other):
                    if fcr[e] != m:
                        fcr[e] = m
                        crf.append(e)
    out_bad = np.empty(len(bad), np.int64)
    for i in range(len(bad)):
        out_bad[i] = bad[i]
    out_fac = np.empty(len(fac), np.int64)
    for i in range(len(fac)):
        out_fac[i] = fac[i]
    out_xf = np.empty(len(xf), np.int64)
    for i in range(len(xf)):
        out_xf[i] = xf[i]
    out_cr = np.empty(len(crf), np.int64)
    for i in range(len(crf)):
        out_cr[i] = crf[i]
    out_av = np.empty(len(av), np.int64)
    for i in range(len(av)):
        out_av[i] = av[i]
    return out_bad, out_fac, ng, out_xf, ni, nf, out_cr, ncr, out_av, dmin


@dataclass
class EditCheck:
    """The checks near one edit (Checker.propose). Vertex and face ids are
    the Checker's own (see Checker.V and Checker.F)."""

    #: vertices in a gap violation or on a crossing or shared-vertex fold
    bad: Int
    #: vertices of gap violations between sheets that face each other
    facing: Int
    gap_pairs: int
    #: faces in a crossing or a shared-vertex fold
    cross_faces: Int
    intersecting_pairs: int
    vertex_fold_pairs: int
    #: faces at a crease (shared-edge fold) on an edge of a changed face
    crease_faces: Int
    creases: int
    #: the affected vertices (the gap was checked there) and their dmin
    affected: Int
    dmin: Float
    #: the ids the edit gave to its new vertices and faces
    new_vertices: Int
    new_faces: Int

    @property
    def ok(self) -> bool:
        """No violation of any check."""
        return not (
            self.gap_pairs
            or self.intersecting_pairs
            or self.vertex_fold_pairs
            or self.creases
        )


def _grow[T: np.generic](a: npt.NDArray[T], n: int, fill: float) -> npt.NDArray[T]:
    """a with its first axis at least n long (doubled), new rows = fill."""
    if len(a) >= n:
        return a
    out: npt.NDArray[T] = np.full((max(n, 2 * len(a)), *a.shape[1:]), fill, a.dtype)
    out[: len(a)] = a
    return out


class Checker:
    """The gap, crossing, shared-vertex fold and crease checks of a mesh
    that changes by local edits: moved vertices, and faces and vertices
    added and removed. The Checker owns a live copy of the mesh (V, F, the
    faces round each vertex, a hashed grid of the face centroids) and keeps
    it current, so each check costs O(the faces near the edit), not O(n).

    An edit is tentative: propose() makes it and checks it, then commit()
    keeps it or rollback() undoes it. check(V, changed) keeps the interface
    of a fixed topology (positions in, masks out)."""

    def __init__(
        self,
        top: Topology,
        V: Float,
        gap: float,
        slack: float = 0.0,
        fold_cos: float = FOLD_COS,
    ):
        n, nf = len(V), len(top.F)
        self.gap, self.slack = gap, slack
        self._V: Float = np.full((n + n // 4 + 16, 3), np.nan)
        self._V[:n] = V
        self._F: Int = np.full((nf + nf // 4 + 16, 3), -1, np.int64)
        self._F[:nf] = top.F
        self._cnt: Int = np.array([n, nf, 0, 0, 0, 0], np.int64)
        self._prm: Float = np.array([0.0, 0.0, gap, 2 * gap, fold_cos])
        self._vfree: Int = np.zeros(len(self._V), np.int64)
        self._ffree: Int = np.zeros(len(self._F), np.int64)
        self._vhead: Int = np.full(len(self._V), -1, np.int64)
        self._cnext: Int = np.full(3 * len(self._F), -1, np.int64)
        self._cprev: Int = np.full(3 * len(self._F), -1, np.int64)
        self._fcell: Int = np.full(len(self._F), -1, np.int64)
        self._gnext: Int = np.full(len(self._F), -1, np.int64)
        self._gprev: Int = np.full(len(self._F), -1, np.int64)
        self._vmarks: list[Int] = [np.zeros(len(self._V), np.int64) for _ in range(7)]
        self._fmarks: list[Int] = [np.zeros(len(self._F), np.int64) for _ in range(3)]
        self._pending: tuple[Int, Float, Int, Int, Int, Int] | None = None
        self._corners = False
        self.rebuild()

    # ---------------------------------------------------------- state

    def _M(self) -> _Live:
        return (
            self._V, self._F, self._vhead, self._cnext, self._cprev, self._fcell,
            self._gnext, self._gprev, self._hkey, self._hhead, self._vfree,
            self._ffree, self._cnt, self._prm, self._lo,
        )  # fmt: skip

    def _K(self) -> _Marks:
        return (*self._vmarks, *self._fmarks)  # type: ignore[return-value]

    @property
    def V(self) -> Float:
        """The live positions (rows of free vertices are NaN)."""
        return self._V[: self._cnt[_NV]]

    @property
    def F(self) -> Int:
        """The live faces (rows of free faces are -1)."""
        return self._F[: self._cnt[_NF]]

    @property
    def cell(self) -> float:
        return float(self._prm[_CELL])

    def faces_of(self, v: int) -> Int:
        """The faces round vertex v."""
        out = []
        c = int(self._vhead[v])
        while c >= 0:
            out.append(c // 3)
            c = int(self._cnext[c])
        return np.array(out, np.int64)

    def rebuild(self, links: int = 0) -> None:
        """The grid again from the live faces: cell = gap + the largest face
        radius + slack, a hash table of at least four slots per face and per
        grid link to come."""
        nf = int(self._cnt[_NF])
        live = self._F[:nf, 0] >= 0
        Fl = self._F[:nf][live]
        T = self._V[Fl]
        C = T.mean(1)
        R = float(np.sqrt(((T - C[:, None]) ** 2).sum(-1)).max()) if len(Fl) else 0.0
        cell = self.gap + R + self.slack
        self._prm[_CELL] = cell if cell > 0 else 1.0
        self._prm[_RMAX] = R
        self._lo = (C.min(0) if len(Fl) else np.zeros(3)) - self._prm[_CELL]
        H = 1 << max(4, int(np.ceil(np.log2(4 * max(len(Fl) + links, 1)))))
        self._hkey = np.full(H, -1, np.int64)
        self._hhead = np.full(H, -1, np.int64)
        self._cnt[_HUSED] = 0
        _link_all(
            self._V, self._F, self._vhead, self._cnext, self._cprev, self._fcell,
            self._gnext, self._gprev, self._hkey, self._hhead, self._cnt,
            self._prm, self._lo, not self._corners,
        )  # fmt: skip
        self._corners = True

    def _reserve(self, nv: int, nf: int, links: int) -> None:
        """Room for nv new vertices, nf new faces and `links` grid links."""
        if self._cnt[_NVFREE] < nv:
            need = int(self._cnt[_NV]) + nv
            if need > len(self._V):
                self._V = _grow(self._V, need, np.nan)
                n = len(self._V)
                self._vfree = _grow(self._vfree, n, 0)
                self._vhead = _grow(self._vhead, n, -1)
                self._vmarks = [_grow(a, n, 0) for a in self._vmarks]
        if self._cnt[_NFFREE] < nf:
            need = int(self._cnt[_NF]) + nf
            if need > len(self._F):
                self._F = _grow(self._F, need, -1)
                n = len(self._F)
                self._ffree = _grow(self._ffree, n, 0)
                self._fcell = _grow(self._fcell, n, -1)
                self._gnext = _grow(self._gnext, n, -1)
                self._gprev = _grow(self._gprev, n, -1)
                self._cnext = _grow(self._cnext, 3 * n, -1)
                self._cprev = _grow(self._cprev, 3 * n, -1)
                self._fmarks = [_grow(a, n, 0) for a in self._fmarks]
        if 2 * (int(self._cnt[_HUSED]) + links) > len(self._hkey):
            self.rebuild(links)

    # ---------------------------------------------------------- edits

    def propose(
        self,
        move: npt.ArrayLike | None = None,
        to: npt.ArrayLike | None = None,
        remove_faces: npt.ArrayLike | None = None,
        add_faces: npt.ArrayLike | None = None,
        add_vertices: npt.ArrayLike | None = None,
        want_dmin: bool = False,
    ) -> EditCheck:
        """Make an edit and check it; commit() keeps it, rollback() undoes
        it. In this order: add_vertices (positions) become new vertices,
        remove_faces go, the vertices `move` go to `to`, add_faces come (a
        vertex -1 - j in add_faces is new vertex j). A vertex left with no
        face is freed at commit."""
        if self._pending is not None:
            raise RuntimeError("an edit is pending: commit() or rollback() first")
        mv = np.asarray([] if move is None else move, np.int64).ravel()
        P = np.asarray(np.zeros((0, 3)) if to is None else to, np.float64).reshape(
            -1, 3
        )
        rm = np.asarray([] if remove_faces is None else remove_faces, np.int64).ravel()
        add = np.asarray(np.zeros((0, 3)) if add_faces is None else add_faces, np.int64)
        add = add.reshape(-1, 3)
        addv = np.asarray(
            np.zeros((0, 3)) if add_vertices is None else add_vertices, np.float64
        ).reshape(-1, 3)
        if len(P) != len(mv):
            raise ValueError("move and to differ in length")
        self._reserve(
            len(addv),
            len(add),
            len(add) + _links_of(self._F, self._vhead, self._cnext, mv),
        )
        newv, newf, old, rows, Told = _apply(self._M(), mv, P, rm, add, addv)
        self._pending = (mv, old, rm, rows, newv, newf)
        cv = np.r_[mv, newv]
        return self._run(cv, newf, Told, False, True, want_dmin, newv, newf)

    def _run(
        self, cv: Int, addf: Int, Told: Float, full: bool, cross: bool,
        want_dmin: bool, newv: Int, newf: Int,
    ) -> EditCheck:  # fmt: skip
        r = _check_kernel(self._M(), self._K(), cv, addf, Told, full, cross, want_dmin)
        bad, fac, ng, xf, ni, nf, crf, ncr, av, dmin = r
        return EditCheck(bad, fac, ng, xf, ni, nf, crf, ncr, av, dmin, newv, newf)

    def commit(self) -> Int:
        """Keep the pending edit. Returns the vertices it freed."""
        if self._pending is None:
            raise RuntimeError("no edit is pending")
        mv, old, rm, rows, newv, newf = self._pending
        self._pending = None
        freed = _settle(self._M(), rm, rows, newv)
        if self.gap + self._prm[_RMAX] > 2 * self._prm[_CELL]:
            self.rebuild()  # the faces grew: a larger cell
        return freed

    def rollback(self) -> None:
        """Undo the pending edit."""
        if self._pending is None:
            raise RuntimeError("no edit is pending")
        mv, old, rm, rows, newv, newf = self._pending
        self._pending = None
        _undo(self._M(), mv, old, rm, rows, newv, newf)

    def check_all(self, want_dmin: bool = False) -> EditCheck:
        """Every check on the whole live mesh (the crossings and folds with
        mesh_intersect's own grid and shared_vertex_folds)."""
        e = np.zeros(0, np.int64)
        r = self._run(e, e, np.zeros((0, 3, 3)), True, False, want_dmin, e, e)
        F = self.F
        live = np.flatnonzero(F[:, 0] >= 0)
        Fl = F[live]
        V = np.where(np.isnan(self.V), 0.0, self.V)
        im, ni = MI.intersecting_faces(V, Fl)
        vm, nf = shared_vertex_folds(V, Fl)
        xf = live[im | vm]
        r.cross_faces, r.intersecting_pairs, r.vertex_fold_pairs = xf, int(ni), int(nf)
        r.bad = np.union1d(r.bad, F[xf].ravel())
        return r

    # ---------------------------------------------------------- fixed topology

    def check(
        self,
        V: Float,
        changed: Bool | None = None,
        facing: Bool | None = None,
        want_dmin: bool = False,
    ) -> tuple[Bool, int, int, Float]:
        """For a fixed topology: the checks at positions V (all n vertices).
        Vertices that differ from the live copy but are not in `changed` are
        taken as they are (they passed an earlier check); then the changed
        vertices are moved, checked, and moved back. Returns a mask of the
        vertices in a gap violation or on a crossing or shared-vertex fold,
        the numbers of gap pairs (at the affected vertices) and of crossing
        pairs (folds included), and dmin (want_dmin; inf elsewhere).
        `facing` (if given) is set True at the vertices of gap violations
        between sheets that face each other. Without `changed`, or when more
        than LOCAL_SHARE of the vertices changed, the whole mesh is checked
        (faster then). Costs O(n) for the comparison and the masks; use
        propose() for O(the edit)."""
        n = len(V)
        if self._pending is not None:
            raise RuntimeError("an edit is pending: commit() or rollback() first")
        if n != self._cnt[_NV] or self._cnt[_NVFREE] or self._cnt[_NFFREE]:
            raise ValueError("check() needs the Checker's original vertices and faces")
        Vf = np.asarray(V, np.float64)
        diff = (Vf != self._V[:n]).any(1)
        chg = np.zeros(n, bool) if changed is None else np.asarray(changed, bool)
        quiet = np.flatnonzero(diff & ~chg)
        e, e3, p3 = np.zeros(0, np.int64), np.zeros((0, 3), np.int64), np.zeros((0, 3))
        if len(quiet):  # earlier moves that the caller kept: no check
            self._reserve(0, 0, _links_of(self._F, self._vhead, self._cnext, quiet))
            _apply(self._M(), quiet, Vf[quiet], e, e3, p3)
        mv = np.flatnonzero(chg) if changed is not None else np.zeros(0, np.int64)
        wide = changed is None or float(chg.mean()) > LOCAL_SHARE
        if wide:
            mv = (
                np.flatnonzero(diff & chg)
                if changed is not None
                else np.zeros(0, np.int64)
            )
        self._reserve(0, 0, _links_of(self._F, self._vhead, self._cnext, mv))
        _, _, old, _, Told = _apply(self._M(), mv, Vf[mv], e, e3, p3)
        try:
            if wide:
                r = self.check_all(want_dmin)
            else:
                r = self._run(mv, e, Told, False, True, want_dmin, e, e)
        finally:
            _undo(self._M(), mv, old, e, e3, e, e)
        if self.gap + self._prm[_RMAX] > 2 * self._prm[_CELL]:
            self.rebuild()
        bad = np.zeros(n, bool)
        bad[r.bad] = True
        if facing is not None:
            facing[r.facing] = True
        dmin = np.full(n, np.inf)
        dmin[r.affected] = r.dmin
        return bad, r.gap_pairs, r.intersecting_pairs + r.vertex_fold_pairs, dmin

    def export(self) -> tuple[Float, Int, Int]:
        """The live mesh compacted: positions, faces, and for each old
        vertex id its new id (-1 if free)."""
        F = self.F[self.F[:, 0] >= 0]
        used = np.zeros(len(self.V), bool)
        used[F.ravel()] = True
        new = np.full(len(self.V), -1, np.int64)
        new[used] = np.arange(int(used.sum()))
        return self.V[used].copy(), new[F], new


def gap_check(
    top: Topology,
    V: Float,
    gap: float,
    facing: Bool | None = None,
    want_dmin: bool = False,
) -> tuple[Bool, int, Float]:
    """The gap test alone on the whole mesh: vertices in a gap violation
    (the point and the face's vertices), the number of violating pairs, and
    (want_dmin) each vertex's distance to the nearest other part within gap
    + the largest face radius (inf beyond, and everywhere without
    want_dmin). `facing` is set True at the vertices of the violations
    between sheets that face each other (normals opposed)."""
    c = Checker(top, V, gap)
    e = np.zeros(0, np.int64)
    r = c._run(e, e, np.zeros((0, 3, 3)), True, False, want_dmin, e, e)
    n = len(V)
    bad = np.zeros(n, bool)
    bad[r.bad] = True
    if facing is not None:
        facing[r.facing] = True
    dmin = np.full(n, np.inf)
    dmin[r.affected] = r.dmin
    return bad, r.gap_pairs, dmin


# ------------------------------------------------------------ all checks


@dataclass
class MeshChecks:
    """The checks of one mesh: masks over the faces (the gap: over the
    vertices) and pair counts. turned and gap are None when not asked for."""

    intersecting: Bool
    intersecting_pairs: int
    vertex_folds: Bool
    vertex_fold_pairs: int
    edge_folds: Bool
    edge_fold_pairs: int
    fold_cos: float
    turned: Bool | None = None
    gap: Bool | None = None
    gap_facing: Bool | None = None
    gap_pairs: int = 0

    def counts(self) -> dict[str, int]:
        """The numbers of faces (vertices for the gap) and pairs, by name."""
        out = dict(
            intersecting_faces=int(self.intersecting.sum()),
            intersecting_pairs=self.intersecting_pairs,
            shared_vertex_fold_faces=int(self.vertex_folds.sum()),
            shared_vertex_fold_pairs=self.vertex_fold_pairs,
            crease_faces=int(self.edge_folds.sum()),
            creases=self.edge_fold_pairs,
        )
        if self.turned is not None:
            out["turned_faces"] = int(self.turned.sum())
        if self.gap is not None:
            out["gap_vertices"] = int(self.gap.sum())
            out["gap_pairs"] = self.gap_pairs
        return out


def check_mesh(
    V: npt.ArrayLike,
    F: npt.ArrayLike,
    fold_cos: float = FOLD_COS,
    reference: Float | None = None,
    gap: float | None = None,
    top: Topology | None = None,
) -> MeshChecks:
    """All the checks on the whole mesh (V, F): intersections
    (mesh_intersect, pairs that share a vertex skipped), shared-vertex
    folds, shared-edge folds (creases) at fold_cos; with `reference` (a
    direction per face), the faces turned against it; with `gap`, the gap
    violations. top: the topology of F, if already made."""
    Vf = np.ascontiguousarray(V, np.float64)
    Fi = np.ascontiguousarray(F, np.int64)
    im, ni = MI.intersecting_faces(Vf, Fi)
    vm, nv = shared_vertex_folds(Vf, Fi, top)
    em, ne = shared_edge_folds(Vf, Fi, fold_cos)
    out = MeshChecks(im, int(ni), vm, nv, em, ne, fold_cos)
    if reference is not None:
        out.turned = turned_faces(Vf, Fi, reference)
    if gap is not None:
        t = Topology.of(Fi, len(Vf)) if top is None else top
        fac = np.zeros(len(Vf), bool)
        out.gap, out.gap_pairs, _ = gap_check(t, Vf, gap, fac)
        out.gap_facing = fac
    return out
