"""The τ = 0 top surface Σ (plan docs/plans/top-surface.md, the user's
choices T1-T8 of 2026-11-18): step 1, the start (T6) with the gap (T2) and
the mesh quality (T7).

Σ is a copy of the white mesh (T1), in mm (voxel index × voxel size).
The start moves the vertices out along their normals in steps of h/4. A
vertex moves while the voxel at its new position is cortex (label 42) or
white matter (label 41). It stops for good at TOP, WM and EDGE:
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
The checks are baillarger.mesh_checks, shared with the WM side:
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
    `origin`), and no two faces with a common edge may fold back on each
    other (the cosine of the angle between their normals at most fold_cos,
    mesh_checks.FOLD_COS: 154 degrees). In a move, only new folds count: a
    fold that was there before the move does not stop it (the white surface
    has creases of its own; the log counts them, nothing removes them).
After each step: a tangential relaxation and edge splits / collapses (T7)
that keep the gap and make no intersection and no new fold.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import global_tau_prep as TP
import numpy as np

from baillarger import mesh_checks as MC

MOVING, TOP, WM, EDGE, BLOCKED = 0, 1, 2, 3, 4
STATE_NAMES = ("moving", "top", "wm", "edge", "blocked")
HALVINGS = 4  # T5: a step of h/4 is halved until it keeps the gap (h/8 ... h/64 here)
RELAX = 0.5  # T7: the fraction of the tangential move toward the neighbours' centroid
#: the per-vertex fields of a Mesh (remeshing and save/restore handle them all)
VERTEX_FIELDS = ("V", "origin", "state", "travel", "last", "stopvox")


@dataclass
class StartSettings:
    gap_vox: float = 0.25  # T2
    step_vox: float = 0.25  # T6
    split_ratio: float = 2.0  # T7: × the white mesh's median edge
    collapse_ratio: float = 0.5  # T7
    wm_limit_vox: float = 3.0  # user, 2026-11-18
    max_steps: int = 200  # a safety limit only (50 voxels); reported if reached
    fold_cos: float = MC.FOLD_COS  # the dihedral fold test (user, 2026-11-18)


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

    def __post_init__(self) -> None:
        if len(self.last) != len(self.V):
            self.last = np.zeros(len(self.V))
        if len(self.stopvox) != len(self.V):
            self.stopvox = np.full((len(self.V), 3), -1, np.int64)
        self.update_topology()

    def update_topology(self) -> None:
        """The adjacency of F, the edge-face pairs and the reference
        normals (the faces at `origin`): after any change of F or origin."""
        self.top = MC.Topology.of(self.F, len(self.V))
        self.F = self.top.F
        self.ptr, self.nbr, self.boundary = self.top.ptr, self.top.nbr, self.top.boundary
        self.pairs = MC.edge_face_pairs(self.F, len(self.V))[1]
        self.ref = MC.face_normals(self.origin, self.F)

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
    n, _ = TP.vertex_normals_areas(V, F)
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
    return MC.check_mesh(m.V, m.F, fold_cos, reference=m.ref, gap=gap, top=m.top).counts()


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
    m: Mesh, site: TP.Site, s: StartSettings, h: float, chk: MC.Checker
) -> tuple[np.ndarray, dict[str, Any]]:
    """One start step; returns the displacement of each vertex and counts.
    The first check tests the pairs with a moving vertex; each later check
    only the pairs with a vertex whose step changed (halved or refused)."""
    d = s.step_vox * h
    N = normals(m.V, m.F)
    # a BLOCKED vertex tries again in each step (user, 2026-11-18)
    act = (m.state == MOVING) | (m.state == BLOCKED)
    m.state[act] = MOVING
    lab = label_at(site, m.V + d * N)
    st = stop_by_label(lab, m.travel + d, s.wm_limit_vox * h)
    stop = act & (st != MOVING)
    top = stop & (st == TOP)
    m.stopvox[top] = np.round((m.V[top] + d * N[top]) / site.vox).astype(np.int64)
    stopped = {STATE_NAMES[k]: int((stop & (st == k)).sum()) for k in (TOP, WM, EDGE)}
    m.state[stop] = st[stop]
    mov = act & ~stop
    step = np.where(mov, d, 0.0)
    halved = np.zeros(len(m.V), bool)
    V0 = m.V.copy()
    refused = np.zeros(len(m.V), bool)
    contact = np.zeros(len(m.V), bool)
    folded = np.zeros(len(m.V), bool)
    changed = step > 0
    k = 0
    # halve up to HALVINGS times, then refuse; a refusal can leave a violation
    # with a vertex that still moves: refuse those too, until none is left
    # (each pass changes at least one vertex)
    while changed.any():
        Vn = V0 + step[:, None] * N
        fac = np.zeros(len(m.V), bool)
        bad, _, _, _ = chk.check(Vn, changed, fac)
        fb, _, _ = new_folds(m, Vn, V0, changed, s.fold_cos)
        folded |= fb & (step > 0)
        badm = (bad | fb) & (step > 0)
        if not badm.any():
            break
        if k < HALVINGS:
            step[badm] *= 0.5
            halved |= badm
        else:
            step[badm] = 0.0
            refused |= badm
            contact[badm] = fac[badm]
        changed = badm
        k += 1
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
        "fold_limited": int(folded.sum()),
        "stopped_new": stopped,
    }
    return disp, info


# ------------------------------------------------------------ mesh quality (T7)


def relax(m: Mesh, chk: MC.Checker, fold_cos: float) -> tuple[np.ndarray, int]:
    """One tangential relaxation step for the interior vertices; moves that
    break the gap, make an intersection or make a new fold are undone (later
    checks test only the undone vertices). Returns the displacement length
    and the number of vertices moved."""
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
    return np.linalg.norm(dlt, axis=1), int((np.abs(dlt).sum(1) > 0).sum())


def remesh(m: Mesh, gap: float, lmax: float, lmin: float, fold_cos: float) -> tuple[int, int]:
    """One batch of edge splits (longer than lmax, at the midpoint: the
    shape does not change), then one batch of collapses (shorter than lmin,
    to the midpoint). Each batch takes edges that are the longest (splits)
    or shortest (collapses) among the candidates near them, so the changes
    do not touch each other. A collapse needs: interior edge, both ends in
    the same state, the link condition, and no face normal turned by more
    than 90 degrees. The result is checked at the merged vertices and their
    faces: the gap, the intersections and the folds (turned from the
    reference, or folded back on a neighbour). If a check fails there, the
    batch is undone and done again without the collapses near the failing
    vertices (at most 3 times; then no collapse in this pass). Returns the
    numbers of splits and collapses."""
    splits = _split(m, lmax)
    exclude = np.zeros(len(m.V), bool)
    for _ in range(3):
        saved = m.save()
        col, newid, merged = _collapse(m, lmin, exclude)
        if col == 0:
            return splits, 0
        moved = np.zeros(len(m.V), bool)
        moved[newid[merged]] = True
        bad, _, _, _ = MC.Checker(m.top, m.V, gap).check(m.V, moved)
        # with many changed vertices the checks run on the whole mesh: keep
        # the failures at the merged vertices and their neighbours
        bad &= MC.near_vertices(m.ptr, m.nbr, moved)
        fb, _, _ = new_folds(m, m.V, None, moved, fold_cos)
        bad |= fb
        if not bad.any():
            return splits, col
        old = np.flatnonzero(newid >= 0)
        hit = np.zeros(len(newid), bool)
        hit[old] = bad[newid[old]]
        m.restore(saved)
        # exclude the failing vertices and their neighbours (the removed end
        # of a merged edge is a neighbour of the kept end)
        exclude |= MC.near_vertices(m.ptr, m.nbr, hit)
    return splits, 0


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
    a, b = sel[:, 0], sel[:, 1]
    sa, sb = m.state[a], m.state[b]
    state = np.where((sa == MOVING) | (sb == MOVING), MOVING, sa)
    # a new vertex keeps a stopping voxel only if it is TOP
    sv = np.where(m.stopvox[a] >= 0, m.stopvox[a], m.stopvox[b])
    sv[state != TOP] = -1
    new = {"state": state, "stopvox": sv}
    for nm in VERTEX_FIELDS:
        x = getattr(m, nm)
        setattr(m, nm, np.r_[x, new[nm] if nm in new else 0.5 * (x[a] + x[b])])
    m.F = np.concatenate(newF).astype(np.int64)
    m.update_topology()
    return k


def _collapse(m: Mesh, lmin: float, exclude: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Collapses, to the midpoint, the interior edges shorter than lmin with
    both ends in the same state that are the shortest candidate in the
    closed 1-rings of both ends (so no two collapses touch the same face),
    pass the link condition (exactly two common neighbours), and turn no
    face normal by more than 90 degrees. Returns the number of collapses,
    the new index of each old vertex (-1: removed) and the old indices of
    the merged (kept) vertices."""
    F, V = m.F, m.V
    n = len(V)
    ident = np.arange(n)
    none = np.zeros(0, np.int64)
    u = _edges(F)
    L = np.linalg.norm(V[u[:, 0]] - V[u[:, 1]], axis=1)
    a, b = u[:, 0], u[:, 1]
    c = (L < lmin) & ~m.boundary[a] & ~m.boundary[b] & ~exclude[a] & ~exclude[b] & (m.state[a] == m.state[b])
    if not c.any():
        return 0, ident, none
    E, Lc = u[c], L[c]
    rank = np.empty(len(E), np.int64)
    rank[np.lexsort((np.arange(len(E)), Lc))] = np.arange(len(E))
    E = E[_local_best(E, rank, n, m.ptr, m.nbr)]
    E = E[MC.link_condition(m.ptr, m.nbr, E)]
    if len(E) == 0:
        return 0, ident, none

    def apply(E: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        remap = ident.copy()
        remap[E[:, 1]] = E[:, 0]
        V2 = V.copy()
        V2[E[:, 0]] = 0.5 * (V[E[:, 0]] + V[E[:, 1]])
        F2 = remap[F]
        gone = (F2[:, 0] == F2[:, 1]) | (F2[:, 1] == F2[:, 2]) | (F2[:, 0] == F2[:, 2])
        return V2, F2, gone

    V2, F2, gone = apply(E)
    # no flipped face: an edge whose fan has a face turned by >= 90 degrees is not collapsed
    owner = np.full(n, -1, np.int64)
    owner[E[:, 0]] = np.arange(len(E))
    owner[E[:, 1]] = np.arange(len(E))
    fo = owner[F].max(1)  # the collapse that changes this face (at most one)
    ch = (fo >= 0) & ~gone
    n_old = MC.face_normals(V, F[ch])
    n_new = MC.face_normals(V2, F2[ch])
    flip = np.zeros(len(E), bool)
    flip[fo[ch][(n_old * n_new).sum(1) <= 0]] = True
    if flip.any():
        E = E[~flip]
        if len(E) == 0:
            return 0, ident, none
        V2, F2, gone = apply(E)
    m.V = V2
    for nm in ("origin", "travel", "last"):
        x = getattr(m, nm).copy()
        x[E[:, 0]] = 0.5 * (x[E[:, 0]] + x[E[:, 1]])
        setattr(m, nm, x)
    used = np.ones(n, bool)
    used[E[:, 1]] = False
    newid = -np.ones(n, np.int64)
    newid[used] = np.arange(int(used.sum()))
    for nm in VERTEX_FIELDS:
        setattr(m, nm, getattr(m, nm)[used])
    m.F = newid[F2[~gone]]
    m.update_topology()
    return len(E), newid, E[:, 0].copy()


# ------------------------------------------------------------ review: isolated TOP vertices


def isolated_top(m: Mesh, labels: np.ndarray) -> dict[str, np.ndarray]:
    """A review list of the TOP vertices, most isolated first: no rule, only
    measures for a person (user, 2026-11-18). For each TOP vertex:
      lag_mm: the median distance moved by its mesh neighbours minus its own
        (large: the neighbours moved on past it, a pit at a TOP vertex);
      comp_voxels: the size of the 6-connected label-0 region that holds the
        voxel that stopped it (small: maybe a label error in the cortex);
      comp_at_crop_edge: whether that region reaches the edge of the array
        (then its size is cut by the crop);
      label0_3x3x3: label-0 voxels in the 3 × 3 × 3 box around that voxel.
    Returns the vertex ids and these arrays, sorted by lag_mm, largest first."""
    from scipy import ndimage as ndi

    top = np.flatnonzero((m.state == TOP) & (m.stopvox[:, 0] >= 0))
    lag = np.zeros(len(top))
    for i, v in enumerate(top):
        nb_ = m.nbr[m.ptr[v] : m.ptr[v + 1]]
        lag[i] = np.median(m.travel[nb_]) - m.travel[v] if len(nb_) else 0.0
    zero = labels == TP.OUTSIDE
    comp, ncomp = ndi.label(zero)
    size = np.bincount(comp.ravel(), minlength=ncomp + 1)
    edge = np.zeros(ncomp + 1, bool)
    for ax in range(3):
        for k in (0, labels.shape[ax] - 1):
            edge[np.unique(np.take(comp, k, axis=ax))] = True
    sv = np.clip(m.stopvox[top], 0, np.array(labels.shape) - 1)
    c = comp[sv[:, 0], sv[:, 1], sv[:, 2]]
    pad = np.pad(zero, 1)
    box = np.zeros(len(top), np.int64)
    for di in range(3):
        for dj in range(3):
            for dk in range(3):
                box += pad[sv[:, 0] + di, sv[:, 1] + dj, sv[:, 2] + dk]
    o = np.argsort(-lag, kind="stable")
    return {
        "vertex": top[o],
        "lag_mm": lag[o],
        "comp_voxels": np.where(c > 0, size[c], 0)[o],
        "comp_at_crop_edge": np.where(c > 0, edge[c], False)[o],
        "label0_3x3x3": box[o],
        "stopvox": sv[o],
    }


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
        u = _edges(m.F)
        med = float(np.median(np.linalg.norm(m.V[u[:, 0]] - m.V[u[:, 1]], axis=1)))
        self.median_edge = med
        lmax, lmin = s.split_ratio * med, s.collapse_ratio * med
        # vertices whose white face meets label 0 directly do not move (excision)
        p0 = m.V + 0.5 * h * normals(m.V, m.F)
        z0 = label_at(site, p0) == TP.OUTSIDE
        m.state[z0] = TOP
        m.stopvox[z0] = np.round(p0[z0] / site.vox).astype(np.int64)
        t0 = time.time()
        # the white surface's own checks (creases, folds) before any change
        white = counts(m, gap, s.fold_cos)
        sp, co = remesh(m, gap, lmax, lmin, s.fold_cos)
        self.initial = {
            "vertices": len(m.V),
            "faces": len(m.F),
            "zero_thickness": int((m.state == TOP).sum()),
            "white": white,
            "splits": sp,
            "collapses": co,
            **counts(m, gap, s.fold_cos),
            "s": time.time() - t0,
        }
        if self.snapshot:
            self.snapshot("start-00", m, np.zeros(len(m.V)), {"step": 0, **self.initial})
        for k in range(1, s.max_steps + 1):
            t0 = time.time()
            # one face grid for the step's checks (no vertex moves more than
            # h/4 along the normal; the relaxation is covered by the rebuild test)
            chk = MC.Checker(m.top, m.V, gap, slack=s.step_vox * h)
            disp, info = expand_step(m, site, s, h, chk)
            moved_any = info["moved"] > 0
            r, nrel = relax(m, chk, s.fold_cos)
            # displacement shown per vertex: the normal step plus the relaxation;
            # carried through the remeshing (new vertices: the mean of the edge's ends)
            m.last = np.sqrt(disp**2 + r**2)
            sp, co = remesh(m, gap, lmax, lmin, s.fold_cos)
            # all the checks on the whole of Σ, once per step, for the log
            c = counts(m, gap, s.fold_cos)
            disp = m.last
            states = {nm: int((m.state == i).sum()) for i, nm in enumerate(STATE_NAMES)}
            rec = {
                "step": k,
                **info,
                "relaxed": nrel,
                "splits": sp,
                "collapses": co,
                **c,
                "vertices": len(m.V),
                "faces": len(m.F),
                "states": states,
                "s": time.time() - t0,
            }
            self.log.append(rec)
            if self.progress:
                self.progress(
                    f"step {k}: moved {info['moved']}, halved {info['halved']}, refused {info['refused']} (blocked {info['blocked']}), "
                    f"fold-limited {info['fold_limited']}, stopped {info['stopped_new']}, splits {sp}, "
                    f"collapses {co}, checks {c}, {states}, {rec['s']:.1f} s"
                )
            if self.snapshot:
                self.snapshot(f"start-{k:02d}", m, disp, rec)
            if not moved_any:  # nothing moved: the next step would be the same
                break
        return m
