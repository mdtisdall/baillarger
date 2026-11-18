"""The τ = 0 top surface Σ (plan docs/plans/top-surface.md, the user's
choices T1-T8 of 2026-11-18): step 1, the start (T6) with the gap (T2) and
the mesh quality (T7).

Σ is a copy of the white mesh (T1), in mm (voxel index × voxel size).
The start moves the vertices out along their normals in steps of h/4. A
vertex moves while the voxel at its new position is cortex (label 42).
It stops for good at TOP, WM and EDGE:
  - TOP: the new position is label 0 (outside medium), or a vertex whose
    white face meets label 0 directly (zero thickness at an excision);
  - WM: the vertex moved more than WM_LIMIT voxels from the white surface
    and is still in white matter (label 41). A vertex in WM keeps moving
    (user, 2026-11-18): the white mesh lies inside the WM voxels, and the
    stop term has c = 0 in WM;
  - EDGE: the new position is outside the crop;
  - BLOCKED (for this step): the move breaks the gap or makes an
    intersection even after the step is halved HALVINGS times (T5's rule),
    and the near face of the other part points against the vertex's normal:
    a first contact (T6). A refused vertex whose violation is not between
    facing sheets stays MOVING. Both try again in each step (user,
    2026-11-18), so a BLOCKED vertex is free again when the other part moves
    away.
The gap (T2): no point of Σ is closer than h/4 to a face of another part
of Σ. Intersections: baillarger.mesh_intersect, plus faces that share one
vertex and cross beyond it (a fold at the vertex, which the triangle test
skips). "Another part": a face with no vertex in the local patch of the
point. The local patch is the part of Σ inside the ball of radius 2 × gap
(h/2) that is connected to the point: the faces reached from the point's
own faces across shared vertices, each closer than h/2 to the point (user,
2026-11-18).
After each step: a tangential relaxation and edge splits / collapses (T7)
that keep the gap and make no intersection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import global_tau_prep as TP
import numba as nb
import numpy as np
from scipy import sparse

from baillarger import mesh_intersect as MI

MOVING, TOP, WM, EDGE, BLOCKED = 0, 1, 2, 3, 4
STATE_NAMES = ("moving", "top", "wm", "edge", "blocked")
HALVINGS = 4  # T5: a step is halved until it keeps the gap (h/2 ... h/64 here)
RELAX = 0.5  # T7: the fraction of the tangential move toward the neighbours' centroid


@dataclass
class StartSettings:
    gap_vox: float = 0.25  # T2
    step_vox: float = 0.25  # T6
    split_ratio: float = 2.0  # T7: × the white mesh's median edge
    collapse_ratio: float = 0.5  # T7
    wm_limit_vox: float = 3.0  # user, 2026-11-18
    max_steps: int = 200  # a safety limit only (50 voxels); reported if reached


@dataclass
class Mesh:
    V: np.ndarray  # (n, 3) mm
    F: np.ndarray  # (m, 3)
    origin: np.ndarray  # (n, 3) mm: the white position (interpolated by remeshing), for display only
    state: np.ndarray  # (n,)
    travel: np.ndarray  # (n,) mm moved along the normals since the start
    last: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (n,) mm moved in the last step (display)
    boundary: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))

    def __post_init__(self) -> None:
        if len(self.last) != len(self.V):
            self.last = np.zeros(len(self.V))
        self.update_topology()

    def update_topology(self) -> None:
        self.ptr, self.nbr = vertex_neighbours(self.F, len(self.V))
        self.boundary = boundary_vertices(self.F, len(self.V))
        n = len(self.V)
        self.vf_ptr = np.r_[0, np.cumsum(np.bincount(self.F.ravel(), minlength=n))].astype(np.int64)
        self.vf = (np.argsort(self.F.ravel(), kind="stable") // 3).astype(np.int64)


# ------------------------------------------------------------ mesh helpers


def vertex_neighbours(F: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    e = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]
    e = np.r_[e, e[:, ::-1]]
    e = np.unique(e, axis=0)
    ptr = np.zeros(n + 1, np.int64)
    np.add.at(ptr, e[:, 0] + 1, 1)
    return np.cumsum(ptr), e[:, 1].astype(np.int64)


def boundary_vertices(F: np.ndarray, n: int) -> np.ndarray:
    e = np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1)
    u, c = np.unique(e, axis=0, return_counts=True)
    b = np.zeros(n, bool)
    b[u[c == 1].ravel()] = True
    return b


def normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    n, _ = TP.vertex_normals_areas(V, F)
    return n


@nb.njit(cache=True, inline="always")
def _ptd(V: np.ndarray, p: int, ia: int, ib: int, ic: int) -> float:
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
    return np.sqrt(dx * dx + dy * dy + dz * dz)


@nb.njit(cache=True)
def _pt_tri(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Distance from the point p to the triangle abc."""
    V = np.empty((4, 3))
    V[0], V[1], V[2], V[3] = p, a, b, c
    return _ptd(V, 0, 1, 2, 3)


@nb.njit(cache=True)
def _cells(
    P: np.ndarray, lo: np.ndarray, cell: float, dims: np.ndarray
) -> np.ndarray:
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


@nb.njit(cache=True)
def _patch(
    p: int, V: np.ndarray, F: np.ndarray, vf_ptr: np.ndarray, vf: np.ndarray, rloc: float,
    stamp: np.ndarray, queue: np.ndarray,
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


@nb.njit(cache=True)
def _gap_kernel(
    V: np.ndarray,
    F: np.ndarray,
    vf_ptr: np.ndarray,
    vf: np.ndarray,
    order: np.ndarray,
    skeys: np.ndarray,
    lo: np.ndarray,
    cell: float,
    dims: np.ndarray,
    gap: float,
    rloc: float,
    check: np.ndarray,
    moved: np.ndarray,
    all_moved: bool,
    want_dmin: bool,
    VN: np.ndarray,
    FN: np.ndarray,
    bad: np.ndarray,
    facing: np.ndarray,
    dmin: np.ndarray,
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
    queue = np.empty(n, np.int64)
    npairs = 0
    q = np.empty(3, np.int64)
    for p in range(n):
        if not check[p]:
            continue
        for a in range(3):
            q[a] = int(np.floor((V[p, a] - lo[a]) / cell))
        built = False
        for rep in range(2):  # pass 0: is any face near? pass 1: with the patch
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
                            if not all_moved and not (moved[p] or moved[a] or moved[b] or moved[c]):
                                continue
                            if built and (stamp[a] == p or stamp[b] == p or stamp[c] == p):
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
                                if VN[p, 0] * FN[f, 0] + VN[p, 1] * FN[f, 1] + VN[p, 2] * FN[f, 2] < 0:
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


@nb.njit(cache=True, inline="always")
def _seg_tri(V: np.ndarray, ip: int, iq: int, ia: int, ib: int, ic: int, tol: float) -> bool:
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
    return tol < t < 1 - tol


@nb.njit(cache=True)
def _foldover_kernel(
    V: np.ndarray, F: np.ndarray, vf_ptr: np.ndarray, vf: np.ndarray, xs: np.ndarray, out: np.ndarray
) -> int:
    """Pairs of faces that share exactly one vertex and cross each other
    beyond it: the edge opposite the shared vertex in one face crosses the
    other face. (baillarger.mesh_intersect skips pairs that share a vertex.)"""
    n = 0
    for x in xs:
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
                hit = _seg_tri(V, f0, f1, F[g, 0], F[g, 1], F[g, 2], 1e-6) or _seg_tri(
                    V, g0, g1, F[f, 0], F[f, 1], F[f, 2], 1e-6
                )
                if hit:
                    out[f] = True
                    out[g] = True
                    n += 1
    return n


def foldovers(m: Mesh, V: np.ndarray, moved: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Folds at shared vertices; with `moved`, only at the moved vertices
    and their neighbours (a fold needs a moved vertex in one of its faces)."""
    out = np.zeros(len(m.F), bool)
    if moved is None:
        xs = np.arange(len(V))
    else:
        near = moved.copy()
        near[m.nbr[np.repeat(moved, np.diff(m.ptr))]] = True
        xs = np.flatnonzero(near)
    n = _foldover_kernel(V, m.F, m.vf_ptr, m.vf, xs.astype(np.int64), out)
    return out, n


def gap_check(
    m: Mesh,
    V: np.ndarray,
    gap: float,
    check: np.ndarray | None = None,
    facing: np.ndarray | None = None,
    moved: np.ndarray | None = None,
    want_dmin: bool = False,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Vertices in a gap violation (the point and the face's vertices), the
    number of violating pairs, and (want_dmin) each checked vertex's
    distance to the nearest other part within gap + the largest face radius
    (inf beyond, and inf everywhere without want_dmin). If `facing` is
    given, it is set True at the vertices of the violations between sheets
    that face each other (normals opposed). With `moved`, only the pairs
    with a moved vertex are tested."""
    F = m.F
    T = V[F]
    C = T.mean(1)
    R = float(np.sqrt(((T - C[:, None]) ** 2).sum(-1)).max())
    cell = gap + R
    lo = C.min(0) - cell
    dims = (np.floor((C.max(0) + cell - lo) / cell) + 1).astype(np.int64)
    key = _cells(C, lo, cell, dims)
    order = np.argsort(key, kind="stable").astype(np.int64)
    skeys = key[order]
    n = len(V)
    chk = np.ones(n, bool) if check is None else check
    bad = np.zeros(n, bool)
    fac = np.zeros(n, bool) if facing is None else facing
    dmin = np.full(n, np.inf)
    VN = normals(V, F)
    FN = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    mv = np.ones(n, bool) if moved is None else moved
    npairs = _gap_kernel(
        V, F, m.vf_ptr, m.vf, order, skeys, lo, cell, dims, gap, 2 * gap, chk, mv, moved is None,
        want_dmin, VN, FN, bad, fac, dmin,
    )
    return bad, npairs, dmin


def violations(
    m: Mesh, V: np.ndarray, gap: float, facing: np.ndarray | None = None, moved: np.ndarray | None = None
) -> tuple[np.ndarray, int, int]:
    """Vertices in a gap violation or on an intersecting face; the numbers
    of gap pairs and intersecting pairs. `facing`: as for gap_check. With
    `moved`, the gap and fold tests count only the pairs with a moved vertex
    (the triangle test is always on the whole mesh)."""
    bad, ng, _ = gap_check(m, V, gap, facing=facing, moved=moved)
    fm, ni = MI.intersecting_faces(V, m.F)
    fo, nf = foldovers(m, V, moved)
    fm |= fo
    ni += nf
    if ni:
        bad[m.F[fm].ravel()] = True
    return bad, ng, ni


# ------------------------------------------------------------ the start (T6)


def label_at(site: TP.Site, V: np.ndarray) -> np.ndarray:
    return TP.sample_labels(site.labels, V / site.vox)


def start_mesh(site: TP.Site) -> Mesh:
    V = site.v * site.vox
    n = len(V)
    return Mesh(V.copy(), site.f.copy(), V.copy(), np.zeros(n, np.int64), np.zeros(n))


def stop_by_label(lab: np.ndarray, travel: np.ndarray, wm_limit: float) -> np.ndarray:
    st = np.full(len(lab), MOVING, np.int64)
    st[lab == TP.OUTSIDE] = TOP
    st[(lab == TP.WM) & (travel > wm_limit)] = WM
    st[lab < 0] = EDGE
    st[~np.isin(lab, (TP.CORTEX, TP.WM)) & (st == MOVING)] = TOP  # any other label: stop
    return st


def expand_step(
    m: Mesh, site: TP.Site, s: StartSettings, h: float
) -> tuple[np.ndarray, dict[str, Any]]:
    """One start step; returns the displacement of each vertex and counts."""
    gap, d = s.gap_vox * h, s.step_vox * h
    N = normals(m.V, m.F)
    # a BLOCKED vertex tries again in each step (user, 2026-11-18)
    act = (m.state == MOVING) | (m.state == BLOCKED)
    m.state[act] = MOVING
    lab = label_at(site, m.V + d * N)
    st = stop_by_label(lab, m.travel + d, s.wm_limit_vox * h)
    stop = act & (st != MOVING)
    stopped = {STATE_NAMES[k]: int((stop & (st == k)).sum()) for k in (TOP, WM, EDGE)}
    m.state[stop] = st[stop]
    mov = act & ~stop
    step = np.where(mov, d, 0.0)
    halved = np.zeros(len(m.V), bool)
    V0 = m.V.copy()
    refused = np.zeros(len(m.V), bool)
    contact = np.zeros(len(m.V), bool)
    for k in range(HALVINGS + 1):
        Vn = V0 + step[:, None] * N
        fac = np.zeros(len(m.V), bool)
        bad, _, _ = violations(m, Vn, gap, fac, moved=step > 0)
        badm = bad & (step > 0)
        if not badm.any():
            break
        if k < HALVINGS:
            step[badm] *= 0.5
            halved |= badm
        else:
            step[badm] = 0.0
            refused |= badm
            contact[badm] = fac[badm]
    Vn = V0 + step[:, None] * N
    # a revert can leave a violation with a vertex that still moves: stop those
    # too, until none is left (each pass stops at least one vertex)
    while True:
        fac = np.zeros(len(m.V), bool)
        bad, _, _ = violations(m, Vn, gap, fac, moved=step > 0)
        badm = bad & (step > 0)
        if not badm.any():
            break
        step[badm] = 0.0
        refused |= badm
        contact[badm] = fac[badm]
        Vn = V0 + step[:, None] * N
    # a refused vertex is a contact (BLOCKED) only where the sheets face each
    # other (user, 2026-11-18); otherwise it stays MOVING and tries again
    m.state[refused & contact] = BLOCKED
    m.V = Vn
    m.travel = m.travel + step
    disp = step.copy()
    info = {
        "moved": int((step > 0).sum()),
        "halved": int(halved.sum()),
        "refused": int(refused.sum()),
        "blocked": int((refused & contact).sum()),
        "stopped_new": stopped,
    }
    return disp, info


# ------------------------------------------------------------ mesh quality (T7)


def relax(m: Mesh, gap: float) -> tuple[np.ndarray, int]:
    """One tangential relaxation step for the interior vertices; moves that
    break the gap or make an intersection are undone. Returns the
    displacement length and the number of vertices moved."""
    n = len(m.V)
    N = normals(m.V, m.F)
    cnt = np.diff(m.ptr)
    S = np.zeros((n, 3))
    np.add.at(S, np.repeat(np.arange(n), cnt), m.V[m.nbr])
    cen = S / np.maximum(cnt, 1)[:, None]
    dlt = cen - m.V
    dlt -= (dlt * N).sum(1)[:, None] * N
    dlt *= RELAX
    dlt[m.boundary | (cnt == 0)] = 0.0
    V0 = m.V.copy()
    while True:  # each pass undoes at least one move
        Vn = V0 + dlt
        bad, _, _ = violations(m, Vn, gap, moved=np.abs(dlt).sum(1) > 0)
        badm = bad & (np.abs(dlt).sum(1) > 0)
        if not badm.any():
            break
        dlt[badm] = 0.0
    m.V = V0 + dlt
    return np.linalg.norm(dlt, axis=1), int((np.abs(dlt).sum(1) > 0).sum())


def remesh(m: Mesh, gap: float, lmax: float, lmin: float) -> tuple[int, int]:
    """One batch of edge splits (longer than lmax, at the midpoint: the
    shape does not change), then one batch of collapses (shorter than lmin,
    to the midpoint). Each batch takes edges that are the longest (splits)
    or shortest (collapses) among the candidates near them, so the changes
    do not touch each other. A collapse needs: interior edge, both ends in
    the same state, the link condition, and no face normal turned by more
    than 90 degrees. The gap, the intersections and the folds are checked on
    the result: if the batch adds gap pairs or intersecting pairs, it is
    undone and done again without the collapses near the violations (at
    most 3 times; then no collapse in this pass)."""
    splits = _split(m, lmax)
    _, ng0, ni0 = violations(m, m.V, gap)
    exclude = np.zeros(len(m.V), bool)
    for _ in range(3):
        keep = (m.V.copy(), m.F.copy(), m.origin.copy(), m.state.copy(), m.travel.copy(), m.last.copy())
        col, newid = _collapse(m, lmin, exclude)
        if col == 0:
            return splits, 0
        bad, ng, ni = violations(m, m.V, gap)
        if ng <= ng0 and ni <= ni0:
            return splits, col
        old = np.flatnonzero(newid >= 0)
        hit = np.zeros(len(newid), bool)
        hit[old] = bad[newid[old]]
        m.V, m.F, m.origin, m.state, m.travel, m.last = keep
        m.update_topology()
        # exclude the bad vertices and their neighbours
        nb = np.repeat(np.arange(len(m.V)), np.diff(m.ptr))
        exclude |= hit
        exclude[m.nbr[hit[nb]]] = True
    return splits, 0


def _edges(F: np.ndarray) -> np.ndarray:
    e = np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1)
    return np.unique(e, axis=0)


def _local_best(E: np.ndarray, rank: np.ndarray, n: int, ptr: np.ndarray | None, nbr: np.ndarray | None) -> np.ndarray:
    """Edges whose rank (smaller is better, all distinct) is the best among
    the candidate edges at their two ends (ptr is None), or at any vertex
    of the closed 1-rings of their ends (ptr, nbr given)."""
    best = np.full(n, np.iinfo(np.int64).max)
    np.minimum.at(best, E[:, 0], rank)
    np.minimum.at(best, E[:, 1], rank)
    if ptr is not None and nbr is not None:
        src = np.repeat(np.arange(n), np.diff(ptr))
        b2 = best.copy()
        np.minimum.at(b2, src, best[nbr])
        best = b2
    return (best[E[:, 0]] == rank) & (best[E[:, 1]] == rank)


def _split(m: Mesh, lmax: float) -> int:
    """Splits the edges longer than lmax that are the longest candidate at
    both ends (so each face has at most one split edge)."""
    F, V = m.F, m.V
    n0 = len(V)
    u = _edges(F)
    L = np.linalg.norm(V[u[:, 0]] - V[u[:, 1]], axis=1)
    c = L > lmax
    if not c.any():
        return 0
    E = u[c]
    rank = np.empty(len(E), np.int64)
    rank[np.lexsort((np.arange(len(E)), -L[c]))] = np.arange(len(E))
    sel = E[_local_best(E, rank, n0, None, None)]
    k = len(sel)
    mid = n0 + np.arange(k)
    skey = sel[:, 0] * n0 + sel[:, 1]
    srt = np.argsort(skey)
    skey, mid = skey[srt], mid[srt]
    newF = [F]
    hit = np.zeros(len(F), bool)
    for j in range(3):
        a, b, cc = F[:, j], F[:, (j + 1) % 3], F[:, (j + 2) % 3]
        key = np.minimum(a, b) * n0 + np.maximum(a, b)
        pos = np.clip(np.searchsorted(skey, key), 0, k - 1)
        h = skey[pos] == key
        if h.any():
            kk = mid[pos[h]]
            newF += [np.c_[a[h], kk, cc[h]], np.c_[kk, b[h], cc[h]]]
            hit |= h
    newF[0] = F[~hit]
    m.V = np.r_[V, 0.5 * (V[sel[:, 0]] + V[sel[:, 1]])]
    for nm in ("origin", "travel", "last"):
        x = getattr(m, nm)
        setattr(m, nm, np.r_[x, 0.5 * (x[sel[:, 0]] + x[sel[:, 1]])])
    sa, sb = m.state[sel[:, 0]], m.state[sel[:, 1]]
    m.state = np.r_[m.state, np.where((sa == MOVING) | (sb == MOVING), MOVING, sa)]
    m.F = np.concatenate(newF).astype(np.int64)
    m.update_topology()
    return k


def _collapse(m: Mesh, lmin: float, exclude: np.ndarray) -> tuple[int, np.ndarray]:
    """Collapses, to the midpoint, the interior edges shorter than lmin with
    both ends in the same state that are the shortest candidate in the
    closed 1-rings of both ends (so no two collapses touch the same face),
    pass the link condition (exactly two common neighbours), and turn no
    face normal by more than 90 degrees. Returns the number of collapses
    and the new index of each old vertex (-1: removed)."""
    F, V = m.F, m.V
    n = len(V)
    ident = np.arange(n)
    u = _edges(F)
    L = np.linalg.norm(V[u[:, 0]] - V[u[:, 1]], axis=1)
    a, b = u[:, 0], u[:, 1]
    c = (L < lmin) & ~m.boundary[a] & ~m.boundary[b] & ~exclude[a] & ~exclude[b] & (m.state[a] == m.state[b])
    if not c.any():
        return 0, ident
    E, Lc = u[c], L[c]
    rank = np.empty(len(E), np.int64)
    rank[np.lexsort((np.arange(len(E)), Lc))] = np.arange(len(E))
    E = E[_local_best(E, rank, n, m.ptr, m.nbr)]
    # link condition: exactly two common neighbours
    A = sparse.csr_matrix((np.ones(len(m.nbr)), m.nbr, m.ptr), shape=(n, n))
    common = np.asarray(A[E[:, 0]].multiply(A[E[:, 1]]).sum(1)).ravel()
    E = E[common == 2]
    if len(E) == 0:
        return 0, ident
    remap = ident.copy()
    remap[E[:, 1]] = E[:, 0]
    V2 = V.copy()
    V2[E[:, 0]] = 0.5 * (V[E[:, 0]] + V[E[:, 1]])
    F2 = remap[F]
    gone = (F2[:, 0] == F2[:, 1]) | (F2[:, 1] == F2[:, 2]) | (F2[:, 0] == F2[:, 2])
    # no flipped face: an edge whose fan has a face turned by > 90 degrees is not collapsed
    owner = np.full(n, -1, np.int64)
    owner[E[:, 0]] = np.arange(len(E))
    owner[E[:, 1]] = np.arange(len(E))
    fo = owner[F].max(1)  # the collapse that changes this face (at most one)
    ch = (fo >= 0) & ~gone
    n_old = np.cross(V[F[ch, 1]] - V[F[ch, 0]], V[F[ch, 2]] - V[F[ch, 0]])
    n_new = np.cross(V2[F2[ch, 1]] - V2[F2[ch, 0]], V2[F2[ch, 2]] - V2[F2[ch, 0]])
    flip = np.zeros(len(E), bool)
    flip[fo[ch][(n_old * n_new).sum(1) <= 0]] = True
    if flip.any():
        E = E[~flip]
        if len(E) == 0:
            return 0, ident
        remap = ident.copy()
        remap[E[:, 1]] = E[:, 0]
        V2 = V.copy()
        V2[E[:, 0]] = 0.5 * (V[E[:, 0]] + V[E[:, 1]])
        F2 = remap[F]
        gone = (F2[:, 0] == F2[:, 1]) | (F2[:, 1] == F2[:, 2]) | (F2[:, 0] == F2[:, 2])
    for nm in ("origin", "travel", "last"):
        x = getattr(m, nm).copy()
        x[E[:, 0]] = 0.5 * (x[E[:, 0]] + x[E[:, 1]])
        setattr(m, nm, x)
    used = np.ones(n, bool)
    used[E[:, 1]] = False
    newid = -np.ones(n, np.int64)
    newid[used] = np.arange(int(used.sum()))
    m.V = V2[used]
    m.origin, m.travel, m.last = m.origin[used], m.travel[used], m.last[used]
    m.state = m.state[used]
    m.F = newid[F2[~gone]]
    m.update_topology()
    return len(E), newid


# ------------------------------------------------------------ the whole start


@dataclass
class StartRun:
    site: TP.Site
    s: StartSettings
    progress: Any = None
    snapshot: Any = None  # callable(name, mesh, disp, meta)
    log: list[dict[str, Any]] = field(default_factory=list)

    def run(self) -> Mesh:
        site, s = self.site, self.s
        h = float(site.vox.mean())
        gap = s.gap_vox * h
        m = start_mesh(site)
        e = np.sort(np.r_[m.F[:, [0, 1]], m.F[:, [1, 2]], m.F[:, [2, 0]]], 1)
        u = np.unique(e, axis=0)
        med = float(np.median(np.linalg.norm(m.V[u[:, 0]] - m.V[u[:, 1]], axis=1)))
        self.median_edge = med
        lmax, lmin = s.split_ratio * med, s.collapse_ratio * med
        # vertices whose white face meets label 0 directly do not move (excision)
        lab0 = label_at(site, m.V + 0.5 * h * normals(m.V, m.F))
        m.state[lab0 == TP.OUTSIDE] = TOP
        t0 = time.time()
        sp, co = remesh(m, gap, lmax, lmin)
        bad, ng, ni = violations(m, m.V, gap)
        self.initial = {
            "vertices": len(m.V),
            "faces": len(m.F),
            "zero_thickness": int((m.state == TOP).sum()),
            "splits": sp,
            "collapses": co,
            "gap_pairs": ng,
            "intersections": ni,
            "s": time.time() - t0,
        }
        if self.snapshot:
            self.snapshot("start-00", m, np.zeros(len(m.V)), {"step": 0, **self.initial})
        for k in range(1, s.max_steps + 1):
            t0 = time.time()
            disp, info = expand_step(m, site, s, h)
            moved_any = info["moved"] > 0
            r, nrel = relax(m, gap)
            # displacement shown per vertex: the normal step plus the relaxation;
            # carried through the remeshing (new vertices: the mean of the edge's ends)
            m.last = np.sqrt(disp**2 + r**2)
            sp, co = remesh(m, gap, lmax, lmin)
            disp = m.last
            bad, ng, ni = violations(m, m.V, gap)
            states = {nm: int((m.state == i).sum()) for i, nm in enumerate(STATE_NAMES)}
            rec = {
                "step": k,
                **info,
                "relaxed": nrel,
                "splits": sp,
                "collapses": co,
                "gap_pairs": ng,
                "intersections": ni,
                "vertices": len(m.V),
                "faces": len(m.F),
                "states": states,
                "s": time.time() - t0,
            }
            self.log.append(rec)
            if self.progress:
                self.progress(
                    f"step {k}: moved {info['moved']}, halved {info['halved']}, refused {info['refused']} (blocked {info['blocked']}), "
                    f"stopped {info['stopped_new']}, splits {sp}, collapses {co}, gap pairs {ng}, "
                    f"intersections {ni}, {states}, {rec['s']:.1f} s"
                )
            if self.snapshot:
                self.snapshot(f"start-{k:02d}", m, disp, rec)
            if not moved_any:  # nothing moved: the next step would be the same
                break
        return m
