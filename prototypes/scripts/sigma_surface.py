"""The top surface Σ as a mesh, its checks and its remesh: the parts of
top_surface.py and sigma_mesh.py (prototype branch claude/tau-shell-cc08e1)
that the Σ stop stage uses, unchanged (the plan docs/plans/top-surface.md
of the prototype branch: the gap T2, the mesh quality T7).

Σ is a mesh in mm (voxel index × voxel size) with per-vertex fields
(`Mesh`). The checks are baillarger.mesh_checks, shared with the WM side:
  - the gap (T2): no point of Σ is closer than h/4 to a face of another part
    of Σ. "Another part": a face with no vertex in the local patch of the
    point. The local patch is the part of Σ inside the ball of radius
    2 × gap (h/2) that is connected to the point: the faces reached from the
    point's own faces across shared vertices, each closer than h/2 to the
    point (user, 2026-11-18);
  - intersections: faces with no common vertex that cross, and faces that
    share one vertex and cross beyond it (a fold at the vertex);
  - folds ("reference + dihedral", user, 2026-11-18): no face may turn by
    90 degrees or more from the white surface it grew from (the face at
    `origin`; FOLD_REFERENCE "none" turns this part off), and no two faces
    with a common edge may fold back on each other (the cosine of the angle
    between their normals at most fold_cos, mesh_checks.FOLD_COS: 154
    degrees). In a move, only new folds count: a fold that was there before
    the move does not stop it.
After each step: a tangential relaxation (`relax`) and edge splits /
collapses / flips (`remesh`, SigmaRemesher on baillarger.surface_remesh)
that keep the gap and make no intersection and no new fold. `contacts`
measures the contacts of Σ from the geometry. `sizing` is the target edge
length at the white vertices (the curvature size field of a coarse start).
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import crop_site as CS
import numpy as np
from scipy.spatial import cKDTree

from baillarger import mesh_checks as MC
from baillarger import mesh_sizing as MS
from baillarger import surface_remesh as SR

MOVING, TOP, WM, EDGE = 0, 1, 2, 3
STATE_NAMES = ("moving", "top", "wm", "edge")
HALVINGS = 4  # T5: a step of h/4 is halved until it keeps the gap (h/8 ... h/64 here)
RELAX = 0.5  # T7: the fraction of the tangential move toward the neighbours' centroid
#: the per-vertex fields of a Mesh (remeshing and save/restore handle them all)
VERTEX_FIELDS = ("V", "origin", "state", "travel", "last", "stopvox", "held")
# The reference of the fold test: "origin" (the faces at `origin`), or "none"
# (no reference: only the dihedral part of the test). An experiment switch,
# set by stop_solve from SolveSettings.fold_reference (2026-11-22: with Σ far
# from the start, faces sit at the 90° limit from the origin reference).
FOLD_REFERENCE = "origin"


@dataclass
class Mesh:
    V: np.ndarray  # (n, 3) mm
    F: np.ndarray  # (m, 3)
    # (n, 3) mm: the white position (interpolated by remeshing); the faces at
    # `origin` are the reference of the fold test
    origin: np.ndarray
    state: np.ndarray  # (n,)
    travel: np.ndarray  # (n,) mm moved along the normals since the start
    last: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (n,) mm moved in the last step (display)
    # (n, 3) voxel whose label stopped a TOP vertex (-1: none), for the isolated-TOP review list
    stopvox: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.int64))
    # (n,) the steps in a row in which the vertex's move was refused (0: it moved)
    held: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self) -> None:
        if len(self.last) != len(self.V):
            self.last = np.zeros(len(self.V))
        if len(self.stopvox) != len(self.V):
            self.stopvox = np.full((len(self.V), 3), -1, np.int64)
        if len(self.held) != len(self.V):
            self.held = np.zeros(len(self.V))
        self.update_topology()

    def update_topology(self) -> None:
        """The adjacency of F, the edge-face pairs and the reference
        normals (the faces at `origin`): after any change of F or origin."""
        self.top = MC.Topology.of(self.F, len(self.V))
        self.F = self.top.F
        self.ptr, self.nbr, self.boundary = self.top.ptr, self.top.nbr, self.top.boundary
        self.pairs = MC.edge_face_pairs(self.F, len(self.V))[1]
        self.ref = MC.face_normals(self.origin, self.F)
        if FOLD_REFERENCE == "none":
            self.ref = np.zeros_like(self.ref)  # a reference of zero area turns nothing
        elif FOLD_REFERENCE != "origin":
            raise ValueError(f"FOLD_REFERENCE: {FOLD_REFERENCE}")

    def save(self) -> dict[str, np.ndarray]:
        """A copy of F and the per-vertex fields (for restore)."""
        out = {nm: getattr(self, nm).copy() for nm in VERTEX_FIELDS}
        out["F"] = self.F.copy()
        return out

    def restore(self, saved: dict[str, np.ndarray]) -> None:
        for nm, x in saved.items():
            setattr(self, nm, x)
        self.update_topology()


# ------------------------------------------------------------ mesh helpers


def normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    n, _ = CS.vertex_normals_areas(V, F)
    return n


def _edges(F: np.ndarray) -> np.ndarray:
    n = int(F.max()) + 1
    k, _ = MC.edge_keys(F, n)
    return np.c_[k // n, k % n]


# ------------------------------------------------------------ the fold test


def _turned(N: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Faces turned by 90 degrees or more from the reference R (a reference
    of zero area turns nothing; a face of zero area is turned)."""
    out: np.ndarray = ((N * R).sum(1) <= 0) & ((R * R).sum(1) > 0)
    return out


def _creased(N: np.ndarray, P: np.ndarray, fold_cos: float) -> np.ndarray:
    """Edge-face pairs P folded back on each other: the test of
    mesh_checks.shared_edge_folds, one result per pair."""
    a, b = N[P[:, 0]], N[P[:, 1]]
    num = (a * b).sum(1)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    out: np.ndarray = (den > 0) & (num <= fold_cos * den)
    return out


def new_folds(
    m: Mesh, V: np.ndarray, V0: np.ndarray | None, changed: np.ndarray, fold_cos: float
) -> tuple[np.ndarray, int, int]:
    """The fold test ("reference + dihedral") of the faces with a changed
    vertex, at positions V: faces turned from their reference (m.ref, the
    faces at `origin`), and edge-face pairs folded back on each other. With
    V0, only the folds that are not there at V0 count (new folds); without
    V0, all count. Returns the vertices of the folded faces, the number of
    turned faces and the number of folded pairs."""
    fc = changed[m.F].any(1)
    P = m.pairs[fc[m.pairs].any(1)]
    idx = np.unique(np.r_[np.flatnonzero(fc), P.ravel()])
    loc = np.full(len(m.F), -1, np.int64)
    loc[idx] = np.arange(len(idx))
    Fi, Pl, R = m.F[idx], loc[P], m.ref[idx]
    N = MC.face_normals(V, Fi)
    tf = fc[idx] & _turned(N, R)
    cp = _creased(N, Pl, fold_cos)
    if V0 is not None:
        N0 = MC.face_normals(V0, Fi)
        tf &= ~_turned(N0, R)
        cp &= ~_creased(N0, Pl, fold_cos)
    bad = np.zeros(len(V), bool)
    bad[Fi[tf].ravel()] = True
    bad[Fi[Pl[cp]].ravel()] = True
    return bad, int(tf.sum()), int(cp.sum())


def counts(m: Mesh, gap: float, fold_cos: float) -> dict[str, int]:
    """All the checks on the whole of Σ (mesh_checks.check_mesh; turned:
    against the reference, the faces at `origin`), for the log."""
    ref = m.ref if FOLD_REFERENCE == "origin" else None
    return MC.check_mesh(m.V, m.F, fold_cos, reference=ref, gap=gap, top=m.top).counts()



def start_mesh(site: CS.Site) -> Mesh:
    V = site.v * site.vox
    n = len(V)
    return Mesh(V.copy(), site.f.copy(), V.copy(), np.zeros(n, np.int64), np.zeros(n))


# ------------------------------------------------------------ contacts, from the geometry

#: how far along its normal a vertex looks for a facing part of Σ (voxels)
FACING_SEARCH_VOX = 2.0


def contacts(m: Mesh, vox: np.ndarray) -> dict[str, np.ndarray]:
    """The contacts of Σ, measured from the geometry (plan T2: two parts of
    Σ with no voxel centre between them along their normals). For each
    MOVING vertex: `facing_mm`, the distance along its outward normal to the
    nearest facing part of Σ (a face whose normal points against it) within
    FACING_SEARCH_VOX voxels (inf: none); `contact`, true where no voxel
    centre lies between the two: none whose projection on the ray is inside
    (0, facing_mm) and whose distance from the ray is at most half a voxel."""
    h = float(vox.mean())
    lmax = FACING_SEARCH_VOX * h
    N = normals(m.V, m.F)
    best = MS.facing_distances(m.V, m.F, N, lmax)
    best[m.state != MOVING] = np.inf
    contact = np.zeros(len(m.V), bool)
    k = np.flatnonzero(np.isfinite(best))
    if len(k):
        o = m.V[k] / vox
        d = N[k]
        L = best[k] / h
        base = np.floor(o).astype(np.int64)
        free = np.ones(len(k), bool)
        r = int(np.ceil(FACING_SEARCH_VOX)) + 1
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                for dk in range(-r, r + 1):
                    p = base + np.array([di, dj, dk]) - o  # voxel centre - ray origin (voxels)
                    t = (p * d).sum(1)
                    lat = np.linalg.norm(p - t[:, None] * d, axis=1)
                    free &= ~((t > 0) & (t < L) & (lat <= 0.5))
        contact[k] = free
    return {"facing_mm": best, "contact": contact}


# ------------------------------------------------------------ mesh quality (T7)


def relax(
    m: Mesh, chk: MC.Checker, fold_cos: float, normal: float = 0.0, only: np.ndarray | None = None
) -> tuple[np.ndarray, int]:
    """One tangential relaxation step for the interior vertices (with only,
    for those of them where only is True); with
    normal > 0, the MOVING vertices also move along
    their normal by `normal` × the normal part of the move toward the
    neighbours' centroid, and that part is added to their travel. Moves
    that break the gap, make an intersection or make a new fold are undone
    (later checks test only the undone vertices). Returns the displacement
    length and the number of vertices moved."""
    n = len(m.V)
    N = normals(m.V, m.F)
    cnt = np.diff(m.ptr)
    S = np.zeros((n, 3))
    np.add.at(S, np.repeat(np.arange(n), cnt), m.V[m.nbr])
    cen = S / np.maximum(cnt, 1)[:, None]
    dlt = cen - m.V
    dn = (dlt * N).sum(1)
    dlt -= dn[:, None] * N
    dlt *= RELAX
    if normal > 0:
        act = m.state == MOVING
        dlt += np.where(act, normal * dn, 0.0)[:, None] * N
    dlt[m.boundary | (cnt == 0)] = 0.0
    if only is not None:
        dlt[~only] = 0.0
    V0 = m.V.copy()
    changed = np.abs(dlt).sum(1) > 0
    while changed.any():  # each pass undoes at least one move
        Vn = V0 + dlt
        bad, _, _, _ = chk.check(Vn, changed)
        fb, _, _ = new_folds(m, Vn, V0, changed, fold_cos)
        badm = (bad | fb) & (np.abs(dlt).sum(1) > 0)
        if not badm.any():
            break
        dlt[badm] = 0.0
        changed = badm
    m.V = V0 + dlt
    m.travel = m.travel + (dlt * N).sum(1)
    return np.linalg.norm(dlt, axis=1), int((np.abs(dlt).sum(1) > 0).sum())


class StartSizes:
    """The remesh target of Σ during a run: each vertex of the start mesh
    keeps edge_ratio × the mean length of its edges there; a later vertex
    takes the target of the start vertex nearest to its origin (the white
    position it grew from). So the remesh keeps the start's local sizes (an
    adaptive coarse start stays adaptive) while Σ stretches or shrinks."""

    def __init__(self, m: Mesh, edge_ratio: float = 1.0):
        E = _edges(m.F)
        L = np.linalg.norm(m.V[E[:, 0]] - m.V[E[:, 1]], axis=1)
        tot = np.bincount(E.ravel(), np.r_[L, L], len(m.V))
        cnt = np.bincount(E.ravel(), minlength=len(m.V))
        self.size = edge_ratio * tot / np.maximum(cnt, 1)
        self.tree = cKDTree(m.origin)

    def __call__(self, m: Mesh) -> np.ndarray:
        out: np.ndarray = self.size[self.tree.query(m.origin)[1]]
        return out


class _NewVertexOrigin(dict):  # type: ignore[type-arg]
    """The origins during a split: every vertex not in `old` (the new one)
    has the origin `mid`."""

    def __init__(self, old: set[int], mid: np.ndarray):
        super().__init__()
        self.old, self.mid = old, mid

    def __contains__(self, v: object) -> bool:
        return v not in self.old and isinstance(v, int) and v >= 0

    def __getitem__(self, v: int) -> np.ndarray:
        return self.mid


class SigmaRemesher(SR.Remesher):
    """baillarger.surface_remesh.Remesher on Σ (user, 2026-11-20: one
    surface remesher, T5's in-run remesh included), carrying Σ's per-vertex
    fields: a split's new vertex takes the mean of its edge's ends (state:
    MOVING if either end is, else the first end's; a stopping voxel only if
    TOP), a collapse's kept vertex the mean of the two (origin, travel,
    last). A collapse needs both ends in the same state (the T7 rule).

    The reference is Σ itself before the remesh, with a target edge length
    per vertex (StartSizes): splits and collapses put their points back on
    that Σ. The normal check
    and the outward bound are off (they measure against the white surface,
    which Σ has left)."""

    def __init__(self, m: Mesh, gap: float, target: float | np.ndarray, fold_cos: float, h: float):
        ref = SR.Reference(m.V, m.F, np.broadcast_to(np.asarray(target, float), (len(m.V),)).copy())
        super().__init__(ref, m.V, m.F, gap, fold_cos, h=h)
        self.last_edit: MC.EditCheck | None = None
        self._origin_now: dict[int, np.ndarray] = {}
        self.fields = {nm: getattr(m, nm).copy() for nm in VERTEX_FIELDS if nm != "V"}

    def _field_room(self, v: int) -> None:
        for nm, x in self.fields.items():
            if v >= len(x):
                pad = np.zeros((max(v + 1, 2 * len(x)) - len(x), *x.shape[1:]), x.dtype)
                self.fields[nm] = np.concatenate([x, pad])

    def _try(self, kind: str, *args: Any, **kw: Any) -> MC.EditCheck | None:
        self.last_edit = super()._try(kind, *args, **kw)
        return self.last_edit

    def turned(self, faces: np.ndarray) -> bool:
        """Remesher's test (against Σ before the remesh) and the T6 fold
        rule's: a face turned by 90 degrees or more from its reference, the
        face at the vertices' origins (a split's new vertex: the mean of the
        edge's ends' origins; a collapse's kept vertex: of both ends')."""
        if super().turned(faces):
            return True
        if not len(faces):
            return False
        Fc = self.chk.F[faces]
        O = np.empty((len(Fc), 3, 3))
        for i, row in enumerate(Fc):
            for j, v in enumerate(row):
                v = int(v)
                O[i, j] = self._origin_now[v] if v in self._origin_now else self.fields["origin"][v]
        R = SR.tri_cross(O)
        N = SR.face_cross(self.chk.V, Fc)
        return bool(_turned(N, R).any())

    def split(self, a: int, b: int) -> bool:
        f = self.fields
        old = {int(v) for x in self.edge_faces(a, b) for v in self.chk.F[x]}
        mid = 0.5 * (f["origin"][a] + f["origin"][b])
        self._origin_now = _NewVertexOrigin(old, mid)
        ok = super().split(a, b)
        self._origin_now = {}
        if not ok:
            return False
        assert self.last_edit is not None
        v = int(self.last_edit.new_vertices[0])
        self._field_room(v)
        sa, sb = f["state"][a], f["state"][b]
        st = MOVING if MOVING in (sa, sb) else sa
        f["state"][v] = st
        sv = f["stopvox"][a] if f["stopvox"][a][0] >= 0 else f["stopvox"][b]
        f["stopvox"][v] = sv if st == TOP else -1
        for nm in ("origin", "travel", "last", "held"):
            f[nm][v] = 0.5 * (f[nm][a] + f[nm][b])
        return True

    def collapse(self, a: int, b: int) -> bool:
        f = self.fields
        if f["state"][a] != f["state"][b]:
            self.refused["collapse_state"] += 1
            return False
        interior = not (self.rim[a] or self.rim[b])
        if interior:  # Remesher keeps a, at the midpoint
            self._origin_now = {a: 0.5 * (f["origin"][a] + f["origin"][b])}
        ok = super().collapse(a, b)
        self._origin_now = {}
        if not ok:
            return False
        keep = a if len(self.chk.faces_of(a)) else b
        if interior:
            for nm in ("origin", "travel", "last"):
                f[nm][keep] = 0.5 * (f[nm][a] + f[nm][b])
        return True

    def result(self) -> Mesh:
        """Σ as a new Mesh (compacted)."""
        V, F, new = self.chk.export()
        self._field_room(len(new) - 1)
        used = new >= 0
        out = {nm: x[: len(new)][used] for nm, x in self.fields.items()}
        return Mesh(V, F, out["origin"], out["state"], out["travel"], out["last"], out["stopvox"], out["held"])


def remesh(
    m: Mesh, gap: float, target: float | np.ndarray, fold_cos: float, h: float, passes: int = 1,
    collapse: bool = True, log: list[dict[str, Any]] | None = None, carry: dict[str, Any] | None = None,
) -> tuple[Mesh, int, int]:
    """Σ remeshed toward a target edge length (one, or per vertex; e.g.
    StartSizes) with SigmaRemesher: `passes`
    rounds of splits, collapses (unless collapse is False) and valence
    flips, no relaxation (T7's own tangential `relax` does that); the rounds
    end early at one that changes nothing. Returns
    the new Σ (m itself when nothing changed) and the numbers of splits and
    collapses. log: gets one record per round run (splits, collapses, flips,
    the refused tries by reason, seconds of each part), and one for the
    setup (the Remesher and its Checker) and the result.

    carry: a holder that the caller keeps between calls. The refusals that this
    call remembers (gap, crossing, crease) are stored in it, and the next call
    reuses them where nothing within the checks' reach has changed since (the
    same result; Remesher.memo_import). The log's last record gets
    memo_carried_in, memo_carried_alive and memo_carried_out."""
    t0 = time.time()
    rm = SigmaRemesher(m, gap, target, fold_cos, h)
    n_in = n_alive = 0
    if carry is not None and rm.memo:
        n_in, n_alive = rm.memo_import(carry.get("memo"))
    t_setup = time.time() - t0
    carry_s = 0.0

    def keep() -> int:
        nonlocal carry_s
        if carry is None or not rm.memo:
            return 0
        t = time.time()
        carry["memo"] = rm.memo_export(*rm.chk.export())
        carry_s = round(time.time() - t, 3)
        return len(carry["memo"]["entries"])

    def info(n_out: int) -> dict[str, Any]:
        if carry is None or not rm.memo:
            return {}
        return {"memo_carried_in": n_in, "memo_carried_alive": n_alive, "memo_carried_out": n_out, "carry_s": carry_s}
    sp = co = fl = 0
    for k in range(passes):
        before = Counter(rm.refused)
        hits0 = rm.memo_hits
        t1 = time.time()
        s1 = rm.split_all()
        t2 = time.time()
        c1 = rm.collapse_all() if collapse else 0
        t3 = time.time()
        f1 = rm.flip_all()
        t4 = time.time()
        sp, co, fl = sp + s1, co + c1, fl + f1
        if log is not None:
            log.append({"pass": k + 1, "splits": s1, "collapses": c1, "flips": f1,
                        "refused": dict(rm.refused - before), "memo_hits": rm.memo_hits - hits0,
                        "s": {"split": round(t2 - t1, 3), "collapse": round(t3 - t2, 3), "flip": round(t4 - t3, 3)}})
        if s1 == 0 and c1 == 0 and f1 == 0:
            break
    if sp == 0 and co == 0 and fl == 0:
        n_out = keep()
        if log is not None:
            log.append({"setup_s": round(t_setup, 3), "result_s": 0.0, **info(n_out)})
        return m, 0, 0
    t5 = time.time()
    out = rm.result()
    t_res = round(time.time() - t5, 3)
    n_out = keep()
    if log is not None:
        log.append({"setup_s": round(t_setup, 3), "result_s": t_res, **info(n_out)})
    return out, sp, co


# ------------------------------------------------------------ the size field of the white surface

EPS_VOX = 0.25  # the shape bound eps = h/4 (the gap of T2)
CURV_VOX = 2.5  # the curvature's smoothing scale
LMIN_VOX = 1.0  # the energy reaches Σ through grid lines one voxel apart
LMAX_VOX = 5.0


def target_range(lmin_vox: float) -> tuple[float, float]:
    """The clip range of the target (× h) for a minimum edge lmin_vox × h.
    The remesher collapses edges below 4/5 × target, so a target of at
    least lmin_vox / (4/5) keeps the edges at or above lmin_vox (where the
    checks allow the collapses). With the default LMIN_VOX it is the range
    [LMIN_VOX, LMAX_VOX]."""
    if lmin_vox <= LMIN_VOX:
        return LMIN_VOX, LMAX_VOX
    lo = lmin_vox / SR.LO
    return lo, max(LMAX_VOX, lo)


def sizing(site: CS.Site, lmin_vox: float = LMIN_VOX) -> np.ndarray:
    """The target edge length (mm) at each white vertex."""
    h = float(site.vox.mean())
    tlo, thi = target_range(lmin_vox)
    V = site.v * site.vox
    facing = MS.facing_distances(V, site.f, site.normal, thi * h)
    L, _ = MS.curvature_sizes(V, site.f, site.normal, EPS_VOX * h, CURV_VOX * h, tlo * h, thi * h,
                              scale=0.75, facing=facing)
    return L

