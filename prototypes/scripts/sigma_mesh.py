"""A coarse start for Σ (user, 2026-11-19: "how coarse can we make the
initial mesh, and then refine it as needed?"), made from the full white
mesh (marching tetrahedra) of a site.

Target edge length per vertex (the sizing field):
  - shape: a flat triangle with edges L over a surface of curvature radius R
    deviates from it by about L² / (8R). With the bound eps = EPS_VOX × h
    this gives L <= sqrt(8 eps R) = sqrt(8 eps / κ), κ the larger absolute
    principal curvature (baillarger.frame.mesh_curvatures, smoothed over
    CURV_VOX × h along the surface). The remeshing keeps edges up to 4/3 ×
    the target, so the target is 3/4 of that bound;
  - facing sheets: L <= the distance along the outward normal to a facing
    part of the white surface (top_surface._facing_hits, up to LMAX_VOX × h),
    so that faces of two sheets across thin tissue stay apart;
  - L is clipped to [LMIN_VOX, LMAX_VOX] × h.
Each vertex of Σ takes the target of its nearest white vertex. The
sizing is an estimate; the bound itself is a check (the deviation): every
white vertex stays within eps of Σ, and Σ's vertices stay on the white
mesh (projection), so the start is within eps of the white surface both
ways (at the white vertices).

One pass (isotropic remeshing, Botsch & Kobbelt 2004, with Σ's checks):
  1. split the edges longer than 4/3 × target, collapse those shorter than
     4/5 × target (top_surface.remesh: the gap, intersections, folds, and
     the deviation);
  2. flip the interior edges whose flip raises the smaller angle of their
     two faces (toward Delaunay), only where the quad is nearly planar: the
     old and the new diagonal (a, b) and (c, d) pass within eps of each
     other, so the flip moves Σ by less than eps; the flips are checked like
     a collapse;
  3. a tangential relaxation (top_surface.relax), then each moved vertex is
     projected back onto the white mesh (its closest point); moves that
     break a check (with the deviation) are undone.
The passes stop when a pass makes no split, collapse or flip, or after
max_passes. The crop's boundary vertices do not move and are not
collapsed. The result's origin is its own positions (the reference of the
fold rule).
"""

from __future__ import annotations

import time
from typing import Any

import global_tau_prep as TP
import numpy as np
import top_surface as TS
from scipy.spatial import cKDTree

from baillarger import mesh_checks as MC
from baillarger.decimate import closest_points
from baillarger.frame import mesh_curvatures

EPS_VOX = 0.25  # the shape bound eps = h/4 (the gap of T2)
CURV_VOX = 2.5  # the curvature's smoothing scale
LMIN_VOX = 1.0  # the energy reaches Σ through grid lines one voxel apart
LMAX_VOX = 5.0


def target_range(lmin_vox: float) -> tuple[float, float]:
    """The clip range of the target (× h) for a minimum edge lmin_vox × h.
    The remeshing collapses edges below 4/5 × target, so a target of at
    least lmin_vox / (4/5) keeps the edges at or above lmin_vox (where the
    checks allow the collapses). With the default LMIN_VOX it is the old
    range [LMIN_VOX, LMAX_VOX]."""
    if lmin_vox <= LMIN_VOX:
        return LMIN_VOX, LMAX_VOX
    lo = lmin_vox / 0.8
    return lo, max(LMAX_VOX, lo)


def sizing(site: TP.Site, lmin_vox: float = LMIN_VOX) -> np.ndarray:
    """The target edge length (mm) at each white vertex."""
    h = float(site.vox.mean())
    tlo, thi = target_range(lmin_vox)
    V = site.v * site.vox
    F = site.f
    k1, k2 = mesh_curvatures(V, F, site.normal, CURV_VOX * h)
    kap = np.maximum(np.abs(k1), np.abs(k2))
    L = 0.75 * np.sqrt(8.0 * EPS_VOX * h / np.maximum(kap, 1e-12))
    # facing sheets
    FN = MC.face_normals(V, F)
    cen = V[F].mean(1)
    rad = float(np.linalg.norm(V[F] - cen[:, None], axis=2).max())
    lmax = thi * h
    act = np.arange(len(V))
    lists = cKDTree(cen).query_ball_point(
        V + 0.5 * lmax * site.normal, 0.5 * lmax + rad
    )
    ptr = np.zeros(len(act) + 1, np.int64)
    ptr[1:] = np.cumsum([len(x) for x in lists])
    idx = np.fromiter((j for x in lists for j in x), np.int64, int(ptr[-1]))
    face_d, _ = TS._facing_hits(V, site.normal, F, FN, act, ptr, idx, lmax)
    L = np.minimum(L, face_d)
    return np.clip(L, tlo * h, thi * h)


def closest_on(
    V: np.ndarray, F: np.ndarray, tree: cKDTree, P: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The closest point on the mesh (V, F) to each point P, among the k
    faces whose centroids (in tree) are nearest; its distance and face."""
    k = min(k, len(F))
    _, cand = tree.query(P, k=k)
    cand = cand.reshape(len(P), k)
    best = np.full(len(P), np.inf)
    out = P.copy()
    face = np.full(len(P), -1, np.int64)
    for j in range(k):
        f = F[cand[:, j]]
        d, b0, b1, b2 = closest_points(P, V[f[:, 0]], V[f[:, 1]], V[f[:, 2]])
        q = (
            b0[:, None] * V[f[:, 0]]
            + b1[:, None] * V[f[:, 1]]
            + b2[:, None] * V[f[:, 2]]
        )
        better = d < best
        best[better] = d[better]
        out[better] = q[better]
        face[better] = cand[better, j]
    return out, best, face


class Reference:
    """The white mesh: the target at each vertex, closest points on it, and
    the deviation check."""

    def __init__(self, site: TP.Site, eps: float | None = None, lmin_vox: float = LMIN_VOX):
        h = float(site.vox.mean())
        self.eps = EPS_VOX * h if eps is None else eps
        self.V = site.v * site.vox
        self.F = site.f
        self.size = sizing(site, lmin_vox)
        #: the deviation is a check only at the default minimum edge; with a
        #: larger minimum (user, 2026-11-19: "Minimum wins") it is measured
        #: and logged, so that it does not keep the short edges
        self.check = lmin_vox <= LMIN_VOX
        self.vtree = cKDTree(self.V)
        self.ftree = cKDTree(self.V[self.F].mean(1))

    def target(self, m: TS.Mesh) -> np.ndarray:
        return self.size[self.vtree.query(m.V)[1]]

    def project(self, P: np.ndarray, k: int = 12) -> tuple[np.ndarray, np.ndarray]:
        """The closest point on the white mesh to each point, and its distance."""
        q, d, _ = closest_on(self.V, self.F, self.ftree, P, k)
        return q, d

    def deviation(
        self, V: np.ndarray, F: np.ndarray, changed: np.ndarray, k: int = 24
    ) -> np.ndarray:
        """The vertices of the faces of Σ (V, F) that are the nearest to a
        white vertex more than eps away, for the white vertices near the
        changed vertices of Σ (within the longest edge of their faces)."""
        bad = np.zeros(len(V), bool)
        if not getattr(self, "check", True):  # a Reference built without __init__ (tests) checks
            return bad
        fc = changed[F].any(1)
        if not fc.any():
            return bad
        Fc = F[fc]
        r = float(np.linalg.norm(V[Fc] - V[Fc][:, [1, 2, 0]], axis=2).max())
        lists = self.vtree.query_ball_point(V[changed], r)
        w = np.unique(np.fromiter((j for x in lists for j in x), np.int64))
        if not len(w):
            return bad
        _, d, face = closest_on(V, F, cKDTree(V[F].mean(1)), self.V[w], k)
        bad[F[face[d > self.eps]].ravel()] = True
        return bad


def segment_distance(p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """The distance between segments (p0, p1) and (q0, q1), row by row."""
    u, v, w = p1 - p0, q1 - q0, p0 - q0
    a, b, c = (u * u).sum(1), (u * v).sum(1), (v * v).sum(1)
    d, e = (u * w).sum(1), (v * w).sum(1)
    den = a * c - b * b
    s = np.where(den > 1e-300 * np.maximum(a * c, 1e-300), (b * e - c * d) / np.where(den > 0, den, 1.0), 0.0)
    s = np.clip(s, 0.0, 1.0)
    t = np.clip((b * s + e) / np.maximum(c, 1e-300), 0.0, 1.0)
    s = np.clip((b * t - d) / np.maximum(a, 1e-300), 0.0, 1.0)
    return np.linalg.norm(w + s[:, None] * u - t[:, None] * v, axis=1)


def _angles(V: np.ndarray, T: np.ndarray) -> np.ndarray:
    """(k, 3) the angles of triangles T (k, 3) at their corners."""
    out = np.zeros(T.shape)
    for j in range(3):
        a, b, c = V[T[:, j]], V[T[:, (j + 1) % 3]], V[T[:, (j + 2) % 3]]
        u, w = b - a, c - a
        cs = (u * w).sum(1) / np.maximum(
            np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1), 1e-300
        )
        out[:, j] = np.arccos(np.clip(cs, -1.0, 1.0))
    return out


def flip_pass(m: TS.Mesh, ref: Reference, gap: float, fold_cos: float) -> int:
    """One batch of edge flips (see the module docstring); returns the
    number made."""
    F, V = m.F, m.V
    n = len(V)
    # half-edges: face f has (F[f, j], F[f, j+1]); the twin has them reversed
    a = F.ravel()
    b = F[:, [1, 2, 0]].ravel()
    c = F[:, [2, 0, 1]].ravel()
    fid = np.repeat(np.arange(len(F)), 3)
    key = a * n + b
    order = np.argsort(key)
    tk = b * n + a
    pos = np.searchsorted(key[order], tk)
    pos = np.clip(pos, 0, len(key) - 1)
    has = key[order][pos] == tk
    twin = np.where(has, order[pos], -1)
    sel = has & (a < b)  # each interior edge once
    h1 = np.flatnonzero(sel)
    h2 = twin[h1]
    ea, eb, ec, ed = a[h1], b[h1], c[h1], c[h2]
    f1, f2 = fid[h1], fid[h2]
    ok = (ec != ed) & ~m.boundary[ea] & ~m.boundary[eb]
    # the new edge must not exist
    ekey = np.minimum(a, b) * n + np.maximum(a, b)
    ex = np.unique(ekey)
    nk = np.minimum(ec, ed) * n + np.maximum(ec, ed)
    p2 = np.clip(np.searchsorted(ex, nk), 0, len(ex) - 1)
    ok &= ex[p2] != nk
    # nearly planar: the two diagonals pass within eps of each other
    nrm = np.cross(V[eb] - V[ea], V[ec] - V[ea])
    ok &= segment_distance(V[ea], V[eb], V[ec], V[ed]) <= ref.eps
    old = np.minimum(
        _angles(V, np.c_[ea, eb, ec]).min(1), _angles(V, np.c_[eb, ea, ed]).min(1)
    )
    T1, T2 = np.c_[ea, ed, ec], np.c_[ed, eb, ec]
    new = np.minimum(_angles(V, T1).min(1), _angles(V, T2).min(1))
    # the flipped faces keep the orientation (no normal turned by 90 degrees)
    n1, n2 = MC.face_normals(V, T1), MC.face_normals(V, T2)
    ok &= ((n1 * n2).sum(1) > 0) & ((n1 * nrm).sum(1) > 0)
    gain = new - old
    ok &= gain > 1e-6
    cand = np.flatnonzero(ok)
    if not len(cand):
        return 0
    cand = cand[np.argsort(-gain[cand], kind="stable")]
    used = np.zeros(len(F), bool)
    vused = np.zeros(n, bool)
    pick = []
    for i in cand:  # independent flips: no shared face or vertex
        q = (ea[i], eb[i], ec[i], ed[i])
        if used[f1[i]] or used[f2[i]] or vused[list(q)].any():
            continue
        used[f1[i]] = used[f2[i]] = True
        vused[list(q)] = True
        pick.append(i)
    pick = np.array(pick, np.int64)
    exclude = np.zeros(n, bool)
    for _ in range(3):
        pk = pick[~exclude[ea[pick]] & ~exclude[eb[pick]]]
        if not len(pk):
            return 0
        saved = m.save()
        F2 = F.copy()
        F2[f1[pk]] = T1[pk]
        F2[f2[pk]] = T2[pk]
        m.F = F2
        m.update_topology()
        moved = np.zeros(n, bool)
        moved[np.r_[ea[pk], eb[pk], ec[pk], ed[pk]]] = True
        bad, _, _, _ = MC.Checker(m.top, m.V, gap).check(m.V, moved)
        bad &= MC.near_vertices(m.ptr, m.nbr, moved)
        fb, _, _ = TS.new_folds(m, m.V, None, moved, fold_cos)
        bad |= fb | ref.deviation(m.V, m.F, moved)
        if not bad.any():
            return len(pk)
        m.restore(saved)
        F = m.F
        exclude |= MC.near_vertices(m.ptr, m.nbr, bad)
    return 0


def relax_project(
    m: TS.Mesh, ref: Reference, gap: float, fold_cos: float
) -> tuple[int, float]:
    """A tangential relaxation (top_surface.relax), then the moved vertices
    projected onto the white mesh; a projection that breaks a check is
    undone (the vertex goes back to its place before the relaxation).
    Returns the vertices moved and the largest projection distance."""
    V0 = m.V.copy()
    chk = MC.Checker(m.top, m.V, gap, slack=float(np.median(ref.size)))
    _, nrel = TS.relax(m, chk, fold_cos)
    mv = np.flatnonzero(np.abs(m.V - V0).sum(1) > 0)
    if not len(mv):
        return 0, 0.0
    q, d = ref.project(m.V[mv])
    Vn = m.V.copy()
    Vn[mv] = q
    changed = np.zeros(len(m.V), bool)
    changed[mv] = True
    while changed.any():
        bad, _, _, _ = chk.check(Vn, changed)
        fb, _, _ = TS.new_folds(m, Vn, V0, changed, fold_cos)
        dv = ref.deviation(Vn, m.F, changed)
        badm = (bad | fb | dv) & (np.abs(Vn - V0).sum(1) > 0)
        if not badm.any():
            break
        Vn[badm] = V0[badm]
        changed = badm
    m.V = Vn
    return int((np.abs(Vn - V0).sum(1) > 0).sum()), float(d.max())


def deviation_all(ref: Reference, V: np.ndarray, F: np.ndarray, k: int = 24) -> np.ndarray:
    """The distance (mm) from each white vertex to Σ (V, F)."""
    _, d, _ = closest_on(V, F, cKDTree(V[F].mean(1)), ref.V, k)
    return d


def _deviation_log(ref: Reference, m: TS.Mesh, h: float) -> dict[str, Any]:
    d = deviation_all(ref, m.V, m.F)
    e = np.linalg.norm(m.V[m.F[:, [0, 1, 2]]] - m.V[m.F[:, [1, 2, 0]]], axis=2).ravel() / h
    return {"deviation_max_vox": float(d.max()) / h, "deviation_over_eps": int((d > ref.eps).sum()),
            "edge_vox_p1_50_99": [float(q) for q in np.percentile(e, [1, 50, 99])]}


def coarse_start(
    site: TP.Site,
    max_passes: int = 30,
    fold_cos: float = MC.FOLD_COS,
    progress: Any = None,
    lmin_vox: float = LMIN_VOX,
) -> tuple[TS.Mesh, Reference, list[dict[str, Any]]]:
    """The coarse start. lmin_vox: the minimum edge (× h; user, 2026-11-19:
    4 h for the coarse stage, refine once τ has its rough shape). Above the
    default, the deviation from the white surface is not a check
    ("Minimum wins"): each pass logs its largest value and the white
    vertices more than eps from Σ."""
    h = float(site.vox.mean())
    gap = 0.25 * h
    ref = Reference(site, lmin_vox=lmin_vox)
    m = TS.start_mesh(site)
    log = []
    for k in range(max_passes):
        t0 = time.time()
        sp, co = TS.remesh(
            m,
            gap,
            4.0 / 3.0,
            4.0 / 5.0,
            fold_cos,
            size=ref.target,
            extra=lambda mm, mv: ref.deviation(mm.V, mm.F, mv),
        )
        fl = flip_pass(m, ref, gap, fold_cos)
        nr, dmax = relax_project(m, ref, gap, fold_cos)
        rec = {
            "pass": k,
            "splits": sp,
            "collapses": co,
            "flips": fl,
            "relaxed": nr,
            "projection_max_vox": dmax / h,
            **_deviation_log(ref, m, h),
            "vertices": len(m.V),
            "faces": len(m.F),
            "s": time.time() - t0,
        }
        log.append(rec)
        if progress:
            progress(rec)
        if sp == 0 and co == 0 and fl == 0:
            break
    m.origin = m.V.copy()
    m.travel = np.zeros(len(m.V))
    m.update_topology()
    return m, ref, log


def main() -> None:
    """usage: sigma_mesh.py SITE_NPZ OUT_NPZ [max_passes] [lmin_vox]
    Writes OUT_NPZ (V mm, F, and the pass log as JSON) for top_solve_run.py start=."""
    import json
    import sys

    site = TP.load_site(sys.argv[1])
    mp = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    lv = float(sys.argv[4]) if len(sys.argv) > 4 else LMIN_VOX
    m, _, log = coarse_start(
        site, max_passes=mp, progress=lambda r: print(r, flush=True), lmin_vox=lv
    )
    np.savez(sys.argv[2], V=m.V, F=m.F, log=json.dumps(log))


if __name__ == "__main__":
    main()
