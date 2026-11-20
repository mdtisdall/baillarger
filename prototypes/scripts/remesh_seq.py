"""Sequential isotropic remeshing (Botsch & Kobbelt 2004) of the coarse Σ
start, with every operation checked on its own (user, 2026-11-19:
"Sequential rewrite", after the evaluation of the batch remesh in
top_surface.remesh).

The mesh lives in a mesh_checks.Checker. Each operation is one
Checker.propose(); the operation is kept (commit) only if it makes no gap
violation, crossing, shared-vertex fold or crease, no face turned
against the white surface, and passes the normal check (below). Else it is undone (rollback) and counted as
refused, with its reason. So no batch undo, no retries, no deadlock.

One pass, in the order of Botsch & Kobbelt:
  1. split the edges longer than 4/3 × target at their midpoint (projected
     onto the white surface), the longest first, until none is left or the
     rest are refused;
  2. collapse the edges shorter than 4/5 × target, the shortest first, but
     not when an edge of the result would be longer than 4/3 × target;
  3. flip the interior edges whose flip brings the valences nearer to 6
     (4 on the rim);
  4. a 3D smoothing of each interior vertex (see below), not projected.
target = the mean of the two ends' targets; a vertex's target is the
sizing field at its nearest white vertex (sigma_mesh.sizing).

The fold reference (user, 2026-11-19: "White normal"): a face is turned
when its normal is at 90 degrees or more to the normal of the white face
at the closest point to its centroid.

The normal check (user, 2026-11-19: "inward is actually fine ... Pushing
outward (into GM) is worse ... having the faces correctly oriented is more
important than precise location"; theta 30 degrees, outward 0.5 voxels).
It replaces the no-cut check. Each white vertex is assigned to its closest
Σ face and keeps its signed distance to Σ (+: Σ is outward of it, toward
GM). A Σ face stands for the white vertices assigned to it: S, the sum of
their area-weighted white normals. An operation is refused ("normal") when
a face it changes, or a face whose assigned white vertices it changes, is
more than theta from its S, and the worst such face is worse than before
the operation (so a start that already exceeds theta somewhere does not
block every operation there). It is refused ("outward") when it leaves a
white vertex more than d_out inside Σ's outer side (Σ outward of it), and
farther than before. Inward offsets are free. After an edit, the white
vertices near the old and the new faces (within the faces' radius + 2 h +
d_out) and the white vertices assigned to the removed or moved faces get
their closest face again (Checker.closest). A farther white vertex keeps
its face: the assignment is exact only near the edit. A face with no white
vertex assigned keeps only the turned test.

The relaxation is a 3D smoothing (Taubin 1995: a Jacobi sweep toward the
mean of the neighbours with lambda = 0.5, then one with mu = -0.53, which
does not shrink the surface), with no projection onto the white surface.
(A Gauss-Seidel sweep, each vertex toward neighbours already moved, shrank
a sphere of radius 10 by about 0.1 per pass.) Splits and
collapses still project their new points onto the white surface.

The rim (the crop's boundary) does not relax. Corners stay where they are.
A collapse of a rim edge or of a spoke (one end on the rim) moves the
other end onto the kept end; an edge between two rim vertices that is not
on the rim does not collapse. A rim edge splits at its midpoint (on the
same box face as its ends).
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

import global_tau_prep as TP
import numpy as np
import sigma_mesh as SM
import top_surface as TS
from scipy.spatial import cKDTree

from baillarger import mesh_checks as MC
from baillarger.decimate import closest_points

HI, LO = 4.0 / 3.0, 4.0 / 5.0


class White:
    """The white surface: closest points, its normals there, the target."""

    def __init__(self, V: np.ndarray, F: np.ndarray, size: np.ndarray):
        self.V, self.F, self.size = V, F, size
        N = MC.face_normals(V, F)
        self.N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-300)
        self.vtree = cKDTree(V)
        self.ftree = cKDTree(V[F].mean(1))

    @classmethod
    def of_site(cls, site: TP.Site, lmin_vox: float) -> White:
        return cls(site.v * site.vox, site.f, SM.sizing(site, lmin_vox))

    def target(self, P: np.ndarray) -> np.ndarray:
        return self.size[self.vtree.query(P)[1]]

    def project(self, P: np.ndarray, k: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The closest points on the white mesh, their distances and white
        faces, among the k faces whose centroids are nearest (as
        sigma_mesh.closest_on, in one call for all candidates)."""
        k = min(k, len(self.F))
        _, cand = self.ftree.query(P, k=k)
        cand = cand.reshape(len(P), k)
        f = self.F[cand.ravel()]
        Pr = np.repeat(P, k, axis=0)
        A, B, C = self.V[f[:, 0]], self.V[f[:, 1]], self.V[f[:, 2]]
        d, b0, b1, b2 = closest_points(Pr, A, B, C)
        j = np.argmin(d.reshape(len(P), k), 1)
        i = np.arange(len(P)) * k + j
        Q = b0[i, None] * A[i] + b1[i, None] * B[i] + b2[i, None] * C[i]
        return Q, d[i], cand[np.arange(len(P)), j]


def box_corners(V: np.ndarray, rim: np.ndarray, rim_edges: np.ndarray, lo: np.ndarray,
                hi: np.ndarray, tol: float) -> np.ndarray:
    """Rim vertices that are corners of a crop box (lo, hi): on two box
    planes or more, or with no box plane in common with both rim
    neighbours (a vertex's planes: those within tol, else the nearest)."""
    D = np.c_[V - lo, hi - V]  # distances to the 6 planes
    on = D < tol
    on[np.arange(len(V)), np.argmin(D, 1)] |= ~on.any(1)
    corner = rim & (on.sum(1) >= 2)
    nb: dict[int, list[int]] = {}
    for a, b in rim_edges:
        nb.setdefault(int(a), []).append(int(b))
        nb.setdefault(int(b), []).append(int(a))
    for v, us in nb.items():
        if len(us) != 2 or not (on[v] & on[us[0]] & on[us[1]]).any():
            corner[v] = True
    return corner


def angle_corners(V: np.ndarray, rim: np.ndarray, rim_edges: np.ndarray,
                  cos_max: float = np.cos(np.radians(45))) -> np.ndarray:
    """Rim vertices where the rim turns by more than 45 degrees (or that
    do not have exactly two rim neighbours)."""
    nb: dict[int, list[int]] = {}
    for a, b in rim_edges:
        nb.setdefault(int(a), []).append(int(b))
        nb.setdefault(int(b), []).append(int(a))
    corner = np.zeros(len(V), bool)
    for v, us in nb.items():
        if len(us) != 2:
            corner[v] = True
            continue
        u, w = V[v] - V[us[0]], V[us[1]] - V[v]
        c = u @ w / max(np.linalg.norm(u) * np.linalg.norm(w), 1e-300)
        corner[v] = c < cos_max
    return corner & rim


def rim_of(F: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """The boundary vertices (mask) and the boundary edges."""
    k, c = MC.edge_keys(F, n)
    kb = k[c == 1]
    E = np.c_[kb // n, kb % n]
    rim = np.zeros(n, bool)
    rim[E.ravel()] = True
    return rim, E


class Remesher:
    """The sequential remesher on a live mesh (V, F) over a White."""

    def __init__(self, white: White, V: np.ndarray, F: np.ndarray, gap: float,
                 fold_cos: float = MC.FOLD_COS, corner: np.ndarray | None = None,
                 theta_deg: float | None = None, d_out: float | None = None,
                 h: float = 1.0):
        self.white, self.gap = white, gap
        self.cos_t = None if theta_deg is None else float(np.cos(np.radians(theta_deg)))
        self.d_out = d_out
        self.rn = 2.0 * h + (d_out or 0.0)
        self.rs = 2.0 * HI * float(white.size.max())
        self.wn = MC.vertex_normals(white.V, white.F)
        cr = MC.face_normals(white.V, white.F)
        self.Wn = np.zeros((len(white.V), 3))  # area-weighted white normals
        for k in range(3):
            np.add.at(self.Wn, white.F[:, k], cr)
        top = MC.Topology.of(F, len(V))
        self.chk = MC.Checker(top, np.asarray(V, np.float64), gap, fold_cos=fold_cos)
        rim, E = rim_of(top.F, len(V))
        self.rim = rim
        self.corner = angle_corners(V, rim, E) if corner is None else corner & rim
        self.t = white.target(V)
        self.refused: Counter[str] = Counter()
        self.places: np.ndarray = np.zeros((0, 3))
        # the assignment of white vertices to Σ faces
        nw = len(white.V)
        self.wf = np.full(nw, -1, np.int64)
        self.wd = np.zeros(nw)
        self.Sf = np.zeros((len(self.chk.F), 3))
        #: the number of white vertices assigned to each face (a face with
        #: none is unassigned; its S may keep a round-off residue)
        self.Cf = np.zeros(len(self.chk.F), np.int64)
        self.fw: dict[int, set[int]] = {}
        if self.cos_t is not None or self.d_out is not None:
            _, _, f = self.chk.closest(white.V, self.rs)
            self.wf[:] = f
            for w in np.flatnonzero(f >= 0):
                self.Sf[f[w]] += self.Wn[w]
                self.Cf[f[w]] += 1
                self.fw.setdefault(int(f[w]), set()).add(int(w))

    @property
    def checking(self) -> bool:
        return self.cos_t is not None or self.d_out is not None

    # ------------------------------------------------------------ helpers

    def _grow(self) -> None:
        n = len(self.chk.V)
        if len(self.rim) < n:
            k = n - len(self.rim)
            self.rim = np.r_[self.rim, np.zeros(k, bool)]
            self.corner = np.r_[self.corner, np.zeros(k, bool)]
            self.t = np.r_[self.t, np.zeros(k)]
        if len(self.Sf) < len(self.chk.F):
            k = len(self.chk.F) - len(self.Sf)
            self.Sf = np.r_[self.Sf, np.zeros((k, 3))]
            self.Cf = np.r_[self.Cf, np.zeros(k, np.int64)]

    def nbrs(self, v: int) -> set[int]:
        F = self.chk.F
        return {int(u) for f in self.chk.faces_of(v) for u in F[f]} - {v}

    def edge_faces(self, a: int, b: int) -> list[int]:
        F = self.chk.F
        return [int(f) for f in self.chk.faces_of(a) if b in F[f]]

    def turned(self, faces: np.ndarray) -> bool:
        """A face turned against the white normal at its centroid's closest
        point (or of zero area)."""
        if not len(faces):
            return False
        V, F = self.chk.V, self.chk.F[faces]
        n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        _, _, wf = self.white.project(V[F].mean(1))
        return bool(((n * self.white.N[wf]).sum(1) <= 0).any())

    def _tris(self, faces: np.ndarray) -> np.ndarray:
        return self.chk.V[self.chk.F[faces]]

    def _near_white(self, T: np.ndarray) -> np.ndarray:
        """The white vertices within rn of the triangles T (a superset:
        within rn + the triangle's radius of its centroid)."""
        if not len(T):
            return np.zeros(0, np.int64)
        C = T.mean(1)
        R = np.linalg.norm(T - C[:, None], axis=2).max(1) + self.rn
        lists = self.white.vtree.query_ball_point(C, R)
        return np.unique(np.fromiter((j for x in lists for j in x), np.int64))

    def _cos(self, faces: np.ndarray, S: np.ndarray, C: np.ndarray) -> np.ndarray:
        """cos(face normal, S) per face; 1 where no white vertex is assigned
        (C = 0)."""
        if not len(faces):
            return np.ones(0)
        n = MC.face_normals(self.chk.V, self.chk.F[faces])
        ns = np.linalg.norm(n, axis=1) * np.linalg.norm(S, axis=1)
        return np.where(np.asarray(C) > 0, (n * S).sum(1) / np.maximum(ns, 1e-300), 1.0)

    def _normal_check(self, ofaces: np.ndarray, cos_old: dict[int, float], old: np.ndarray,
                      r: MC.EditCheck, faces: np.ndarray, removed: set[int]) -> tuple[str, Any]:
        """After a proposed edit: the new assignment of the affected white
        vertices and the reason to refuse ("" if none)."""
        chk = self.chk
        self._grow()
        w = np.union1d(self._near_white(old), self._near_white(self._tris(faces)))
        extra = [x for f in ofaces for x in self.fw.get(int(f), ())]
        w = np.union1d(w, np.asarray(extra, np.int64))
        if not len(w):
            return "", None
        W = self.white.V[w]
        # the closest face within r is the closest face: search a small
        # radius first, then larger ones for the misses only
        T = self._tris(faces)
        r1 = self.rn + 2.0 * float(np.linalg.norm(T - T.mean(1)[:, None], axis=2).max()) if len(T) else self.rn
        Q, d, f = chk.closest(W, r1)
        for rr in (self.rs, None):
            miss = ~np.isfinite(d)
            if not miss.any():
                break
            if rr is None:
                rr = float(np.abs(self.wd[w[miss]]).max()) + self.rs
            Q2, d2, f2 = chk.closest(W[miss], rr)
            Q[miss], d[miss], f[miss] = Q2, d2, f2
        found = np.isfinite(d)
        side = np.where(found, ((Q - W) * self.wn[w]).sum(1), 0.0)
        sd = np.where(found, np.where(side >= 0, d, -d), self.wd[w])
        f = np.where(found, f, -1)
        if self.d_out is not None:
            lim = np.maximum(self.d_out, np.maximum(self.wd[w], 0.0))
            if (found & (sd > lim + 1e-12)).any():
                return "outward", None
        # tentative sums
        new_ids = {int(x) for x in r.new_faces}
        S: dict[int, np.ndarray] = {x: np.zeros(3) for x in removed | new_ids}
        C: dict[int, int] = {x: 0 for x in removed | new_ids}

        def touch(x: int) -> None:
            if x not in S:
                S[x], C[x] = self.Sf[x].copy(), int(self.Cf[x])

        for j, x in enumerate(w):
            fo, fn = int(self.wf[x]), int(f[j])
            if fo == fn and fo not in removed and fo not in new_ids:
                continue
            if fo >= 0 and fo not in removed:
                touch(fo)
                S[fo] = S[fo] - self.Wn[x]
                C[fo] -= 1
            if fn >= 0:
                touch(fn)
                S[fn] = S[fn] + self.Wn[x]
                C[fn] += 1
        for x in faces:
            touch(int(x))
        live = [x for x in S if chk.F[x, 0] >= 0]
        if self.cos_t is not None and live:
            la = np.asarray(live, np.int64)
            after = self._cos(la, np.array([S[x] for x in live]), np.array([C[x] for x in live]))
            before = [cos_old[x] if x in cos_old else None for x in live]
            other = [x for x, b in zip(live, before, strict=True) if b is None and x not in new_ids]
            if other:
                cb = self._cos(np.asarray(other, np.int64), self.Sf[other], self.Cf[other])
                for x, c in zip(other, cb, strict=True):
                    cos_old[x] = float(c)
            worst_before = min([*cos_old.values(), 1.0])
            if after.min() < min(self.cos_t, worst_before) - 1e-12:
                return "normal", None
        return "", (w, f, sd, S, C, removed)

    def _apply(self, state: Any) -> None:
        w, f, sd, S, C, removed = state
        for x in removed:
            self.fw.pop(x, None)
        for j, x in enumerate(w):
            fo, fn = int(self.wf[x]), int(f[j])
            if fo >= 0 and fo not in removed:
                s = self.fw.get(fo)
                if s is not None:
                    s.discard(int(x))
            if fn >= 0:
                self.fw.setdefault(fn, set()).add(int(x))
        self.wf[w] = f
        self.wd[w] = sd
        for x, v in S.items():
            self.Sf[x] = v
            self.Cf[x] = C[x]

    def _try(self, kind: str, **edit: Any) -> MC.EditCheck | None:
        """Propose an edit; keep it if every check passes (returns its
        check), else undo it (None)."""
        mv0 = [int(v) for v in edit.get("move", [])]
        if self.checking:
            rem = [int(x) for x in edit.get("remove_faces", [])]
            ofaces = np.unique(np.r_[np.asarray(rem, np.int64),
                                     *[self.chk.faces_of(v) for v in mv0]]).astype(np.int64)
            old = self._tris(ofaces)
            cos_old = dict(zip((int(x) for x in ofaces),
                               (float(c) for c in self._cos(ofaces, self.Sf[ofaces], self.Cf[ofaces])),
                               strict=True))
        r = self.chk.propose(**edit)
        reason, state = "", None
        if r.gap_pairs:
            reason = "gap"
        elif r.intersecting_pairs or r.vertex_fold_pairs:
            reason = "cross"
        elif r.creases:
            reason = "crease"
        else:
            mv = np.asarray(edit.get("move", []), np.int64)
            faces = np.unique(np.r_[r.new_faces, *[self.chk.faces_of(int(v)) for v in mv]].astype(np.int64))
            if self.turned(faces):
                reason = "turned"
            elif self.checking:
                reason, state = self._normal_check(ofaces, cos_old, old, r, faces, set(rem))
        if reason:
            self.chk.rollback()
            self.refused[f"{kind}_{reason}"] += 1
            return None
        self.chk.commit()
        self._grow()
        if state is not None:
            self._apply(state)
        return r

    def _edges(self) -> tuple[np.ndarray, np.ndarray]:
        """The live edges (a < b) and their lengths over their targets."""
        F = self.chk.F
        F = F[F[:, 0] >= 0]
        E = np.unique(np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1), axis=0)
        V = self.chk.V
        L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
        return E, L / (0.5 * (self.t[E[:, 0]] + self.t[E[:, 1]]))

    def ratio(self, a: int, b: int) -> float:
        V = self.chk.V
        return float(np.linalg.norm(V[a] - V[b]) / (0.5 * (self.t[a] + self.t[b])))

    # ------------------------------------------------------------ operations

    def split(self, a: int, b: int) -> bool:
        ef = self.edge_faces(a, b)
        if len(ef) not in (1, 2):
            self.refused["split_nonmanifold"] += 1
            return False
        mid = 0.5 * (self.chk.V[a] + self.chk.V[b])
        rim_edge = len(ef) == 1
        m = mid if rim_edge else self.white.project(mid[None])[0][0]
        F = self.chk.F
        add = []
        for f in ef:
            row = [int(u) for u in F[f]]
            k = next(i for i in range(3) if {row[i], row[(i + 1) % 3]} == {a, b})
            p, q, o = row[k], row[(k + 1) % 3], row[(k + 2) % 3]
            add += [[p, -1, o], [-1, q, o]]
        r = self._try("split", add_vertices=[m], remove_faces=ef, add_faces=add)
        if r is None:
            return False
        v = int(r.new_vertices[0])
        self.rim[v] = rim_edge
        self.corner[v] = False
        self.t[v] = float(self.white.target(m[None])[0])
        return True

    def collapse(self, a: int, b: int) -> bool:
        ef = self.edge_faces(a, b)
        ra, rb = bool(self.rim[a]), bool(self.rim[b])
        common = self.nbrs(a) & self.nbrs(b)
        V = self.chk.V
        if ra and rb:
            if len(ef) != 1:
                self.refused["collapse_chord"] += 1
                return False
            if self.corner[a] and self.corner[b]:
                self.refused["collapse_corners"] += 1
                return False
            keep, rem = (b, a) if self.corner[b] else (a, b)
            pos = V[keep].copy()
            need = 1
        elif ra or rb:
            keep, rem = (a, b) if ra else (b, a)
            pos = V[keep].copy()
            need = 2
        else:
            keep, rem = a, b
            pos = self.white.project((0.5 * (V[a] + V[b]))[None])[0][0]
            need = 2
        if len(ef) != need or len(common) != need:
            self.refused["collapse_link"] += 1
            return False
        tk = float(self.white.target(pos[None])[0]) if not (ra or rb) else self.t[keep]
        for u in (self.nbrs(rem) | self.nbrs(keep)) - {keep, rem}:
            if np.linalg.norm(pos - V[u]) > HI * 0.5 * (tk + self.t[u]):
                self.refused["collapse_long"] += 1
                return False
        rest = [int(f) for f in self.chk.faces_of(rem) if f not in ef]
        Fr = self.chk.F[rest].copy()
        Fr[Fr == rem] = keep
        if self._try("collapse", move=[keep], to=[pos], remove_faces=ef + rest, add_faces=Fr) is None:
            return False
        self.t[keep] = tk
        return True

    def valence_target(self, v: int) -> int:
        return 4 if self.rim[v] else 6

    def flip(self, a: int, b: int) -> bool | None:
        """Flip edge (a, b) if that brings the valences nearer to their
        targets; None if the flip would not improve them (no try)."""
        ef = self.edge_faces(a, b)
        if len(ef) != 2:
            return None
        F = self.chk.F
        f, g = ef
        row = [int(u) for u in F[f]]
        k = next(i for i in range(3) if row[i] == a)
        if row[(k + 1) % 3] != b:  # orient: f = (a, b, c)
            a, b = b, a
            k = next(i for i in range(3) if row[i] == a)
        c = row[(k + 2) % 3]
        d = next(int(u) for u in F[g] if u not in (a, b))
        if d in self.nbrs(c):
            return None
        val = {v: len(self.nbrs(v)) for v in (a, b, c, d)}
        before = sum((val[v] - self.valence_target(v)) ** 2 for v in (a, b, c, d))
        val[a] -= 1
        val[b] -= 1
        val[c] += 1
        val[d] += 1
        if min(val[a], val[b]) < 3:
            return None
        after = sum((val[v] - self.valence_target(v)) ** 2 for v in (a, b, c, d))
        if after >= before:
            return None
        return self._try("flip", remove_faces=[f, g], add_faces=[[c, a, d], [c, d, b]]) is not None

    def relax(self, v: int, target: np.ndarray) -> bool:
        """Move v to target (3D, no projection)."""
        if self.rim[v]:
            return False
        p = self.chk.V[v]
        if np.linalg.norm(target - p) < 1e-9 * self.t[v]:
            return False
        if self._try("relax", move=[v], to=[target]) is None:
            return False
        self.t[v] = float(self.white.target(target[None])[0])
        return True

    # ------------------------------------------------------------ passes

    def split_all(self, sweeps: int = 10) -> int:
        n = 0
        for _ in range(sweeps):
            E, r = self._edges()
            o = np.argsort(-r)
            o = o[r[o] > HI]
            k = 0
            for a, b in E[o]:
                a, b = int(a), int(b)
                if b not in self.nbrs(a) or self.ratio(a, b) <= HI:
                    continue
                k += self.split(a, b)
            n += k
            if k == 0:
                break
        return n

    def collapse_all(self) -> int:
        E, r = self._edges()
        o = np.argsort(r)
        o = o[r[o] < LO]
        n = 0
        for a, b in E[o]:
            a, b = int(a), int(b)
            if len(self.chk.faces_of(a)) == 0 or len(self.chk.faces_of(b)) == 0:
                continue
            if b not in self.nbrs(a) or self.ratio(a, b) >= LO:
                continue
            n += self.collapse(a, b)
        return n

    def flip_all(self) -> int:
        E, _ = self._edges()
        n = 0
        for a, b in E:
            a, b = int(a), int(b)
            if len(self.chk.faces_of(a)) == 0 or b not in self.nbrs(a):
                continue
            n += bool(self.flip(a, b))
        return n

    def relax_all(self, lam: float = 0.5, mu: float = -0.53) -> int:
        """One Taubin step: a sweep with lam, then one with mu. Each sweep
        is a Jacobi step: every vertex moves toward the mean of its
        neighbours' positions at the start of the sweep (the form that does
        not shrink); the moves are then tried one at a time."""
        n = 0
        for step in (lam, mu):
            live = self.chk.F[self.chk.F[:, 0] >= 0]
            vs = [int(v) for v in np.unique(live) if not self.rim[v]]
            V = self.chk.V
            tg = {}
            for v in vs:
                nb = list(self.nbrs(v))
                if nb:
                    tg[v] = V[v] + step * (V[nb].mean(0) - V[v])
            for v, q in tg.items():
                n += self.relax(v, q)
        return n

    def measures(self, h: float) -> dict[str, Any]:
        V, F, _ = self.chk.export()
        E, r = self._edges()
        rim, _ = rim_of(F, len(V))
        nb = MC.vertex_neighbours(F, len(V))[0]
        deg = np.diff(nb)
        ang = np.degrees(SM._angles(V, F).min(1))
        return {
            "vertices": len(V),
            "faces": len(F),
            "edge_over_target_p1_5_50_95_99": [round(float(x), 3) for x in np.percentile(r, [1, 5, 50, 95, 99])],
            "edge_in_range": round(float(((r >= LO) & (r <= HI)).mean()), 4),
            "valence_interior": {int(k): int(c) for k, c in zip(*np.unique(deg[~rim], return_counts=True), strict=True)},
            "min_angle_deg_p1_5_50": [round(float(x), 1) for x in np.percentile(ang, [1, 5, 50])],
            **surface_measures(V, F, self.white, self.Wn, self.wn, h, self.places),
        }


def dihedral(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per interior edge: the angle (degrees) between its two faces' normals,
    and the edge's midpoint."""
    n = len(V)
    E = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]
    fid = np.tile(np.arange(len(F)), 3)
    key = np.minimum(E[:, 0], E[:, 1]) * n + np.maximum(E[:, 0], E[:, 1])
    o = np.argsort(key, kind="stable")
    key, fid, E = key[o], fid[o], E[o]
    same = key[1:] == key[:-1]
    f1, f2 = fid[:-1][same], fid[1:][same]
    N = MC.face_normals(V, F)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-300)
    c = np.clip((N[f1] * N[f2]).sum(1), -1, 1)
    mid = 0.5 * (V[E[:-1][same, 0]] + V[E[:-1][same, 1]])
    return np.degrees(np.arccos(c)), mid


def surface_measures(V: np.ndarray, F: np.ndarray, white: White, Wn: np.ndarray, wn: np.ndarray,
                     h: float, places: np.ndarray, radius: float = 1.5) -> dict[str, Any]:
    """Measures of Σ (V, F) against the white surface, from scratch: each
    white vertex's closest Σ face, its signed offset (+: Σ outward of it),
    each Σ face's angle to S (its assigned white vertices' summed
    area-weighted normals), and the dihedral angles (roughness), over all
    and within radius (mm) of each of the places (mm)."""
    q, dev, fc = SM.closest_on(V, F, cKDTree(V[F].mean(1)), white.V, 24)
    side = ((q - white.V) * wn).sum(1)
    out = np.where(side > 0, dev, 0.0) / h
    din = np.where(side < 0, dev, 0.0) / h
    S = np.zeros((len(F), 3))
    np.add.at(S, fc, Wn)
    has = np.linalg.norm(S, axis=1) > 0
    n = MC.face_normals(V, F)
    fa = np.degrees(np.arccos(np.clip((n * S).sum(1) / np.maximum(
        np.linalg.norm(n, axis=1) * np.linalg.norm(S, axis=1), 1e-300), -1, 1)))[has]
    di, mid = dihedral(V, F)
    res: dict[str, Any] = {
        "deviation_max_vox": round(float(dev.max()) / h, 3),
        "outward_max_vox": round(float(out.max()), 3),
        "outward_over_0.5_1_vox": [int((out > t).sum()) for t in (0.5, 1)],
        "inside_max_vox": round(float(din.max()), 3),
        "inside_over_0.5_1_2_4_vox": [int((din > t).sum()) for t in (0.5, 1, 2, 4)],
        "face_angle_p50_90_99_max": [round(float(x), 1) for x in np.percentile(fa, [50, 90, 99, 100])],
        "faces_over_30_45_60": [int((fa > t).sum()) for t in (30, 45, 60)],
        "faces_unassigned": int((~has).sum()),
        "dihedral_p50_90_99_max": [round(float(x), 1) for x in np.percentile(di, [50, 90, 99, 100])],
    }
    if len(places):
        res["dihedral_at_places_p50_90_max"] = []
        for p in places:
            m = np.linalg.norm(mid - p, axis=1) < radius
            res["dihedral_at_places_p50_90_max"].append(
                [round(float(x), 1) for x in np.percentile(di[m], [50, 90, 100])] if m.any() else [])
    return res


def remesh(rm: Remesher, h: float, max_passes: int = 20, progress: Any = None) -> list[dict[str, Any]]:
    """Passes until one makes no split, collapse or flip (or max_passes)."""
    log = []
    for k in range(max_passes):
        t0 = time.time()
        rm.refused.clear()
        sp = rm.split_all()
        co = rm.collapse_all()
        fl = rm.flip_all()
        rl = rm.relax_all()
        rec = {"pass": k, "splits": sp, "collapses": co, "flips": fl, "relaxed": rl,
               "refused": dict(rm.refused), **rm.measures(h), "s": round(time.time() - t0, 1)}
        log.append(rec)
        if progress:
            progress(rec)
        if sp == 0 and co == 0 and fl == 0:
            break
    return log


def coarse_start(site: TP.Site, lmin_vox: float, max_passes: int = 20,
                 progress: Any = None, theta_deg: float | None = None,
                 d_out_vox: float | None = None,
                 places_vox: np.ndarray | None = None) -> tuple[TS.Mesh, list[dict[str, Any]], dict[str, Any]]:
    """The coarse Σ start of a site, sequentially remeshed; returns the
    mesh, the pass log and the checks of the start (the white mesh)."""
    h = float(site.vox.mean())
    gap = 0.25 * h
    white = White.of_site(site, lmin_vox)
    V, F = white.V.copy(), site.f.copy()
    rim, E = rim_of(F, len(V))
    hi = (np.array(site.labels.shape) - 1) * site.vox
    corner = box_corners(V, rim, E, np.zeros(3), hi, 1e-6 * h)
    rm = Remesher(white, V, F, gap, corner=corner, theta_deg=theta_deg,
                  d_out=None if d_out_vox is None else d_out_vox * h, h=h)
    if places_vox is not None:
        rm.places = np.asarray(places_vox, float) * site.vox
    start = rm.chk.check_all()
    start_info = {"gap_pairs": start.gap_pairs, "intersecting_pairs": start.intersecting_pairs,
                  "vertex_fold_pairs": start.vertex_fold_pairs, "creases": start.creases,
                  "rim": int(rim.sum()), "corners": int(corner.sum()), **rm.measures(h)}
    if progress:
        progress({"start": start_info})
    log = remesh(rm, h, max_passes, progress)
    Vo, Fo, _ = rm.chk.export()
    m = TS.Mesh(Vo, Fo, Vo.copy(), np.zeros(len(Vo), np.int64), np.zeros(len(Vo)))
    return m, log, start_info


def main() -> None:
    """usage: remesh_seq.py SITE_NPZ OUT_NPZ [max_passes] [lmin_vox]
    [theta_deg] [d_out_vox] [places.json]
    (theta_deg, d_out_vox: the normal check; "none" turns one off;
    places.json: a list of {"at_vox": [i, j, k]} for the roughness there)"""
    import json
    import sys

    def opt(i: int, default: float | None) -> float | None:
        if len(sys.argv) <= i:
            return default
        return None if sys.argv[i] == "none" else float(sys.argv[i])

    site = TP.load_site(sys.argv[1])
    mp = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    lv = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0
    places = None
    if len(sys.argv) > 7:
        places = np.array([p["at_vox"] for p in json.load(open(sys.argv[7]))], float)
    m, log, start = coarse_start(site, lv, mp, progress=lambda r: print(json.dumps(r), flush=True),
                                 theta_deg=opt(5, None), d_out_vox=opt(6, None), places_vox=places)
    np.savez(sys.argv[2], V=m.V, F=m.F, log=json.dumps(log), start=json.dumps(start))


if __name__ == "__main__":
    main()
