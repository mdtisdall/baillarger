"""Validity checks of a triangle mesh, shared by the WM side (decimate,
white_surf) and the τ = 0 top surface Σ.

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

check_mesh runs them on a whole mesh. Checker keeps a face grid for one
topology while the vertices move and checks only the pairs with a changed
vertex. link_condition is the edge-collapse test of exactly two common
neighbours.
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
def _ptd(V: Float, p: int, ia: int, ib: int, ic: int) -> float:
    """Distance from vertex p to the triangle of vertices ia, ib, ic
    (Ericson, closest point), in scalars: no temporary arrays."""
    ax, ay, az = V[ia, 0], V[ia, 1], V[ia, 2]
    abx, aby, abz = V[ib, 0] - ax, V[ib, 1] - ay, V[ib, 2] - az
    acx, acy, acz = V[ic, 0] - ax, V[ic, 1] - ay, V[ic, 2] - az
    px, py, pz = V[p, 0], V[p, 1], V[p, 2]
    apx, apy, apz = px - ax, py - ay, pz - az
    d1 = abx * apx + aby * apy + abz * apz
    d2 = acx * apx + acy * apy + acz * apz
    if d1 <= 0 and d2 <= 0:
        qx, qy, qz = ax, ay, az
    else:
        bpx, bpy, bpz = px - V[ib, 0], py - V[ib, 1], pz - V[ib, 2]
        d3 = abx * bpx + aby * bpy + abz * bpz
        d4 = acx * bpx + acy * bpy + acz * bpz
        if d3 >= 0 and d4 <= d3:
            qx, qy, qz = V[ib, 0], V[ib, 1], V[ib, 2]
        else:
            vc = d1 * d4 - d3 * d2
            if vc <= 0 and d1 >= 0 and d3 <= 0:
                t = d1 / (d1 - d3)
                qx, qy, qz = ax + t * abx, ay + t * aby, az + t * abz
            else:
                cpx, cpy, cpz = px - V[ic, 0], py - V[ic, 1], pz - V[ic, 2]
                d5 = abx * cpx + aby * cpy + abz * cpz
                d6 = acx * cpx + acy * cpy + acz * cpz
                if d6 >= 0 and d5 <= d6:
                    qx, qy, qz = V[ic, 0], V[ic, 1], V[ic, 2]
                else:
                    vb = d5 * d2 - d1 * d6
                    if vb <= 0 and d2 >= 0 and d6 <= 0:
                        t = d2 / (d2 - d6)
                        qx, qy, qz = ax + t * acx, ay + t * acy, az + t * acz
                    else:
                        va = d3 * d6 - d5 * d4
                        if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
                            t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
                            bx, by, bz = V[ib, 0], V[ib, 1], V[ib, 2]
                            qx = bx + t * (V[ic, 0] - bx)
                            qy = by + t * (V[ic, 1] - by)
                            qz = bz + t * (V[ic, 2] - bz)
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


# ------------------------------------------------------------ the gap


@njit(cache=True)
def _cells(P: Float, lo: Float, cell: float, dims: Int) -> Int:
    """Cell key of each point (cells of size `cell` from lo; -1 outside)."""
    out = np.empty(P.shape[0], np.int64)
    for i in range(P.shape[0]):
        k = 0
        ok = True
        for a in range(3):
            q = int(np.floor((P[i, a] - lo[a]) / cell))
            if q < 0 or q >= dims[a]:
                ok = False
            k = k * dims[a] + q
        out[i] = k if ok else -1
    return out


@njit(cache=True)
def _patch(
    p: int,
    V: Float,
    F: Int,
    vf_ptr: Int,
    vf: Int,
    rloc: float,
    stamp: Int,
    queue: Int,
) -> None:
    """Stamp p's local patch: the faces reached from p's own faces across
    shared vertices, each closer than rloc to p."""
    stamp[p] = p
    qh, qt = 0, 1
    queue[0] = p
    while qh < qt:
        u = queue[qh]
        qh += 1
        for k in range(vf_ptr[u], vf_ptr[u + 1]):
            g = vf[k]
            a, b, c = F[g, 0], F[g, 1], F[g, 2]
            if stamp[a] == p and stamp[b] == p and stamp[c] == p:
                continue
            if _ptd(V, p, a, b, c) >= rloc:
                continue
            for w in (a, b, c):
                if stamp[w] != p:
                    stamp[w] = p
                    queue[qt] = w
                    qt += 1


@njit(cache=True)
def _gap_kernel(
    V: Float,
    F: Int,
    vf_ptr: Int,
    vf: Int,
    order: Int,
    skeys: Int,
    lo: Float,
    cell: float,
    dims: Int,
    gap: float,
    rloc: float,
    check: Bool,
    moved: Bool,
    all_moved: bool,
    want_dmin: bool,
    VN: Float,
    FN: Float,
    bad: Bool,
    facing: Bool,
    dmin: Float,
) -> int:
    """For each vertex p with check[p]: the faces whose centroid is in the
    27 cells around p (cell ≥ gap + the largest face radius, so every face
    within gap is among them). p's local patch (_patch) is skipped: a face
    with a vertex in the patch is in it or at least rloc > gap away. A face
    closer than gap marks p and its three vertices bad, and also `facing`
    when the face's normal points against p's normal (two sheets that face
    each other: a contact). Unless all_moved, a pair counts only if p or a
    vertex of the face moved (pairs of still vertices did not change). The
    patch is built only when a face is closer than gap (or want_dmin: then
    dmin[p] = the distance to the nearest other part among the candidates,
    inf if none). Returns the number of pairs."""
    n = V.shape[0]
    stamp = np.full(n, -1, np.int64)
    ring = np.full(n, -1, np.int64)
    queue = np.empty(n, np.int64)
    npairs = 0
    q = np.empty(3, np.int64)
    for p in range(n):
        if not check[p]:
            continue
        # p and its 1-ring: a face with one of these vertices that is closer
        # than rloc is in p's patch, so pass 0 need not look at it
        ring[p] = p
        for k in range(vf_ptr[p], vf_ptr[p + 1]):
            g = vf[k]
            ring[F[g, 0]] = p
            ring[F[g, 1]] = p
            ring[F[g, 2]] = p
        for a in range(3):
            q[a] = int(np.floor((V[p, a] - lo[a]) / cell))
        built = False
        for _rep in range(2):  # pass 0: is any face near? pass 1: with the patch
            near = False
            for di in range(-1, 2):
                i0 = q[0] + di
                if i0 < 0 or i0 >= dims[0]:
                    continue
                for dj in range(-1, 2):
                    j0 = q[1] + dj
                    if j0 < 0 or j0 >= dims[1]:
                        continue
                    for dk in range(-1, 2):
                        k0 = q[2] + dk
                        if k0 < 0 or k0 >= dims[2]:
                            continue
                        key = (i0 * dims[1] + j0) * dims[2] + k0
                        s0 = np.searchsorted(skeys, key)
                        s1 = np.searchsorted(skeys, key + 1)
                        for t in range(s0, s1):
                            f = order[t]
                            a, b, c = F[f, 0], F[f, 1], F[f, 2]
                            if a == p or b == p or c == p:
                                continue
                            if not all_moved and not (
                                moved[p] or moved[a] or moved[b] or moved[c]
                            ):
                                continue
                            if built and (
                                stamp[a] == p or stamp[b] == p or stamp[c] == p
                            ):
                                continue
                            if (
                                not built
                                and not want_dmin
                                and (ring[a] == p or ring[b] == p or ring[c] == p)
                            ):
                                continue
                            d = _ptd(V, p, a, b, c)
                            if not built:
                                if d < gap or want_dmin:
                                    near = True
                                continue
                            if d < dmin[p]:
                                dmin[p] = d
                            if d < gap:
                                bad[p] = True
                                bad[a] = True
                                bad[b] = True
                                bad[c] = True
                                dot = (
                                    VN[p, 0] * FN[f, 0]
                                    + VN[p, 1] * FN[f, 1]
                                    + VN[p, 2] * FN[f, 2]
                                )
                                if dot < 0:
                                    facing[p] = True
                                    facing[a] = True
                                    facing[b] = True
                                    facing[c] = True
                                npairs += 1
            if built or not near:
                break
            _patch(p, V, F, vf_ptr, vf, rloc, stamp, queue)
            built = True
    return npairs


@njit(cache=True)
def _tri_kernel(
    T: Float,
    F: Int,
    order: Int,
    skeys: Int,
    lo: Float,
    cell: float,
    dims: Int,
    rad: int,
    fchg: Bool,
    all_chg: bool,
    out: Bool,
) -> int:
    """Intersecting face pairs (mesh_intersect's pair test _tt, rel_tol
    1e-4; pairs that share a vertex are skipped, as in
    mesh_intersect.intersecting_faces) with at least one changed face
    (all_chg: every pair). Candidates: faces whose grid centroid is within
    rad cells of the face's centroid (rad × cell ≥ twice the largest face
    radius + the largest move since the grid was built). Marks both faces in
    out; returns the number of pairs."""
    n = 0
    for f in range(F.shape[0]):
        if not (all_chg or fchg[f]):
            continue
        cx = (T[f, 0, 0] + T[f, 1, 0] + T[f, 2, 0]) / 3.0
        cy = (T[f, 0, 1] + T[f, 1, 1] + T[f, 2, 1]) / 3.0
        cz = (T[f, 0, 2] + T[f, 1, 2] + T[f, 2, 2]) / 3.0
        q0 = int(np.floor((cx - lo[0]) / cell))
        q1 = int(np.floor((cy - lo[1]) / cell))
        q2 = int(np.floor((cz - lo[2]) / cell))
        for i0 in range(q0 - rad, q0 + rad + 1):
            if i0 < 0 or i0 >= dims[0]:
                continue
            for j0 in range(q1 - rad, q1 + rad + 1):
                if j0 < 0 or j0 >= dims[1]:
                    continue
                for k0 in range(q2 - rad, q2 + rad + 1):
                    if k0 < 0 or k0 >= dims[2]:
                        continue
                    key = (i0 * dims[1] + j0) * dims[2] + k0
                    s0 = np.searchsorted(skeys, key)
                    s1 = np.searchsorted(skeys, key + 1)
                    for t in range(s0, s1):
                        g = order[t]
                        if g == f:
                            continue
                        if (all_chg or fchg[g]) and g < f:
                            continue  # each pair once
                        sh = False
                        for a in range(3):
                            for b in range(3):
                                if F[f, a] == F[g, b]:
                                    sh = True
                        if sh:
                            continue
                        ov = True
                        for a in range(3):
                            flo = min(T[f, 0, a], T[f, 1, a], T[f, 2, a])
                            fhi = max(T[f, 0, a], T[f, 1, a], T[f, 2, a])
                            glo = min(T[g, 0, a], T[g, 1, a], T[g, 2, a])
                            ghi = max(T[g, 0, a], T[g, 1, a], T[g, 2, a])
                            if fhi < glo or ghi < flo:
                                ov = False
                        if not ov:
                            continue
                        if MI._tt(T, f, T, g, 1e-4):
                            out[f] = True
                            out[g] = True
                            n += 1
    return n


class Checker:
    """The gap, intersection and shared-vertex fold checks for one mesh
    topology, with one face grid (centroids, cell = gap + the largest face
    radius + slack) reused while the vertices move: a check at new positions
    is exact while gap + the largest face radius + the largest move since
    the grid was built ≤ cell (else the grid is built again at the new
    positions)."""

    def __init__(self, top: Topology, V: Float, gap: float, slack: float = 0.0):
        self.top, self.gap, self.slack = top, gap, slack
        self._build(V)

    def _build(self, V: Float) -> None:
        T = V[self.top.F]
        C = T.mean(1)
        R = float(np.sqrt(((T - C[:, None]) ** 2).sum(-1)).max())
        self.V0 = V.copy()
        self.cell = cell = self.gap + R + self.slack
        self.lo: Float = C.min(0) - cell
        self.dims: Int = (np.floor((C.max(0) + cell - self.lo) / cell) + 1).astype(
            np.int64
        )
        key = _cells(C, self.lo, cell, self.dims)
        self.order: Int = np.argsort(key, kind="stable").astype(np.int64)
        self.skeys: Int = key[self.order]

    def _ready(self, V: Float) -> tuple[Float, int]:
        T = V[self.top.F]
        C = T.mean(1)
        R = float(np.sqrt(((T - C[:, None]) ** 2).sum(-1)).max())
        dmax = float(np.sqrt(((V - self.V0) ** 2).sum(1)).max()) if len(V) else 0.0
        if self.gap + R + dmax > self.cell:
            self._build(V)
            dmax = 0.0
        rad = int(np.ceil((2 * R + dmax) / self.cell))
        return T, max(rad, 1)

    def near(self, V: Float, changed: Bool) -> Bool:
        """Vertices whose cell is within two cells of a changed vertex's
        cell: every vertex closer than 2 × cell ≥ gap + an edge length to a
        changed vertex, so every vertex within gap of a changed face."""
        keys = _cells(V, self.lo, self.cell, self.dims)
        q = np.floor((V[changed] - self.lo) / self.cell).astype(np.int64)
        r = range(-2, 3)
        off = np.array([(a, b, c) for a in r for b in r for c in r])
        Q = (q[:, None, :] + off[None]).reshape(-1, 3)
        Q = Q[np.all((Q >= 0) & (Q < self.dims), 1)]
        mk = np.unique((Q[:, 0] * self.dims[1] + Q[:, 1]) * self.dims[2] + Q[:, 2])
        if not len(mk):
            return np.zeros(len(V), bool)
        pos = np.clip(np.searchsorted(mk, keys), 0, len(mk) - 1)
        out: Bool = (keys >= 0) & (mk[pos] == keys)
        return out

    def check(
        self,
        V: Float,
        changed: Bool | None = None,
        facing: Bool | None = None,
        want_dmin: bool = False,
    ) -> tuple[Bool, int, int, Float]:
        """Vertices in a gap violation or on an intersecting face (or a
        shared-vertex fold), the numbers of gap pairs and of intersecting
        pairs (folds included), and dmin (see gap_check). With `changed`,
        only the pairs with a changed vertex are tested: exact for the new
        violations when the other vertices did not move since a check that
        found none among them. `facing` (if given) is set True at the
        vertices of gap violations between sheets that face each other."""
        t = self.top
        T, rad = self._ready(V)
        n = len(V)
        allc = changed is None
        chg = np.ones(n, bool) if changed is None else changed
        # when most vertices changed, the whole-mesh paths are faster (the
        # extra pairs among unchanged vertices only add to `bad` there, which
        # the callers mask with the changed vertices)
        wide = allc or float(chg.mean()) > LOCAL_SHARE
        chk = np.ones(n, bool) if wide else self.near(V, chg)
        bad = np.zeros(n, bool)
        fac = np.zeros(n, bool) if facing is None else facing
        dmin = np.full(n, np.inf)
        VN = vertex_normals(V, t.F)
        FN = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        ng = _gap_kernel(
            V, t.F, t.vf_ptr, t.vf, self.order, self.skeys, self.lo, self.cell,
            self.dims, self.gap, 2 * self.gap, chk, chg, allc, want_dmin, VN, FN,
            bad, fac, dmin,
        )  # fmt: skip
        if wide:  # mesh_intersect's own box grid is faster for many faces
            fm, ni = MI.intersecting_faces(V, t.F)
        else:
            fm = np.zeros(len(t.F), bool)
            ni = _tri_kernel(
                np.ascontiguousarray(T), t.F, self.order, self.skeys, self.lo,
                self.cell, self.dims, rad, chg[t.F].any(1), False, fm,
            )  # fmt: skip
        fo, nf = shared_vertex_folds(V, t.F, t, None if allc else chg)
        fm |= fo
        ni += nf
        if ni:
            bad[t.F[fm].ravel()] = True
        return bad, ng, ni, dmin


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
    T = V[top.F]
    n = len(V)
    bad = np.zeros(n, bool)
    fac = np.zeros(n, bool) if facing is None else facing
    dmin = np.full(n, np.inf)
    one = np.ones(n, bool)
    ng = _gap_kernel(
        V, top.F, top.vf_ptr, top.vf, c.order, c.skeys, c.lo, c.cell, c.dims, gap,
        2 * gap, one, one, True, want_dmin, vertex_normals(V, top.F),
        np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), bad, fac, dmin,
    )  # fmt: skip
    return bad, ng, dmin


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
