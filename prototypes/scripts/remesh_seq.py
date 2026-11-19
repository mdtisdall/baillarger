"""Sequential isotropic remeshing (Botsch & Kobbelt 2004) of the coarse Σ
start, with every operation checked on its own (user, 2026-11-19:
"Sequential rewrite", after the evaluation of the batch remesh in
top_surface.remesh).

The mesh lives in a mesh_checks.Checker. Each operation is one
Checker.propose(); the operation is kept (commit) only if it makes no gap
violation, crossing, shared-vertex fold or crease, and no face turned
against the white surface. Else it is undone (rollback) and counted as
refused, with its reason. So no batch undo, no retries, no deadlock.

One pass, in the order of Botsch & Kobbelt:
  1. split the edges longer than 4/3 × target at their midpoint (projected
     onto the white surface), the longest first, until none is left or the
     rest are refused;
  2. collapse the edges shorter than 4/5 × target, the shortest first, but
     not when an edge of the result would be longer than 4/3 × target;
  3. flip the interior edges whose flip brings the valences nearer to 6
     (4 on the rim);
  4. a tangential relaxation of each interior vertex toward the mean of its
     neighbours, then projected onto the white surface.
target = the mean of the two ends' targets; a vertex's target is the
sizing field at its nearest white vertex (sigma_mesh.sizing).

The fold reference (user, 2026-11-19: "White normal"): a face is turned
when its normal is at 90 degrees or more to the normal of the white face
at the closest point to its centroid.

The no-cut check (user, 2026-11-19: "No-cut check only", after the review
of the thin WM fingers that the coarsening cut off): with eps given, an
operation is also refused when it leaves a white vertex more than eps from
Σ with Σ on its WM side (Σ cuts into the white surface). Σ may lie farther
than eps on the outer side (across a sulcus). Each white vertex keeps its
signed distance to Σ (+ outside, - inside; 0 at the start, where Σ is the
white mesh); after an edit, the white vertices near the old and the new
faces get theirs again (Checker.closest, within a search radius of twice
the largest edge). A white vertex with no Σ within that radius keeps its
old distance if it was more than eps outside, else it is cut.

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
                 eps: float | None = None):
        self.white, self.gap, self.eps = white, gap, eps
        #: the white vertices' signed distances to Σ (the no-cut check)
        self.wd = np.zeros(len(white.V))
        self.wn = MC.vertex_normals(white.V, white.F)
        self.rs = 2.0 * HI * float(white.size.max())
        top = MC.Topology.of(F, len(V))
        self.chk = MC.Checker(top, np.asarray(V, np.float64), gap, fold_cos=fold_cos)
        rim, E = rim_of(top.F, len(V))
        self.rim = rim
        self.corner = angle_corners(V, rim, E) if corner is None else corner & rim
        self.t = white.target(V)
        self.refused: Counter[str] = Counter()

    # ------------------------------------------------------------ helpers

    def _grow(self) -> None:
        n = len(self.chk.V)
        if len(self.rim) < n:
            k = n - len(self.rim)
            self.rim = np.r_[self.rim, np.zeros(k, bool)]
            self.corner = np.r_[self.corner, np.zeros(k, bool)]
            self.t = np.r_[self.t, np.zeros(k)]

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
        """The white vertices within eps of the triangles T (a superset:
        within eps + the triangle's radius of its centroid)."""
        if not len(T):
            return np.zeros(0, np.int64)
        C = T.mean(1)
        R = np.linalg.norm(T - C[:, None], axis=2).max(1) + self.eps
        lists = self.white.vtree.query_ball_point(C, R)
        return np.unique(np.fromiter((j for x in lists for j in x), np.int64))

    def _cut(self, old: np.ndarray, new: np.ndarray) -> tuple[bool, np.ndarray, np.ndarray]:
        """The no-cut check after an edit: (cut, the white vertices
        checked, their new signed distances)."""
        w = np.union1d(self._near_white(old), self._near_white(new))
        if not len(w):
            return False, w, np.zeros(0)
        W = self.white.V[w]
        Q, d, _ = self.chk.closest(W, self.rs)
        found = np.isfinite(d)
        side = np.where(found, ((Q - W) * self.wn[w]).sum(1), 0.0)
        sd = np.where(found, np.where(side >= 0, d, -d), self.wd[w])
        cut = (found & (d > self.eps) & (side < 0)) | (~found & (self.wd[w] <= self.eps))
        return bool(cut.any()), w, sd

    def _try(self, kind: str, **edit: Any) -> MC.EditCheck | None:
        """Propose an edit; keep it if every check passes (returns its
        check), else undo it (None)."""
        mv0 = [int(v) for v in edit.get("move", [])]
        if self.eps is not None:
            ofaces = np.unique(np.r_[np.asarray(edit.get("remove_faces", []), np.int64),
                                     *[self.chk.faces_of(v) for v in mv0]]).astype(np.int64)
            old = self._tris(ofaces)
        r = self.chk.propose(**edit)
        reason = ""
        if r.gap_pairs:
            reason = "gap"
        elif r.intersecting_pairs or r.vertex_fold_pairs:
            reason = "cross"
        elif r.creases:
            reason = "crease"
        else:
            mv = np.asarray(edit.get("move", []), np.int64)
            faces = np.r_[r.new_faces, *[self.chk.faces_of(int(v)) for v in mv]].astype(np.int64)
            if self.turned(np.unique(faces)):
                reason = "turned"
            elif self.eps is not None:
                cut, w, sd = self._cut(old, self._tris(np.unique(faces)))
                if cut:
                    reason = "nocut"
        if reason:
            self.chk.rollback()
            self.refused[f"{kind}_{reason}"] += 1
            return None
        self.chk.commit()
        if self.eps is not None:
            self.wd[w] = sd
        self._grow()
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

    def relax(self, v: int) -> bool:
        if self.rim[v]:
            return False
        nb = list(self.nbrs(v))
        if not nb:
            return False
        V = self.chk.V
        p = V[v]
        q = V[nb].mean(0)
        _, _, wf = self.white.project(p[None])
        n = self.white.N[wf[0]]
        d = q - p
        d -= (d @ n) * n
        new = self.white.project((p + d)[None])[0][0]
        if np.linalg.norm(new - p) < 1e-9 * self.t[v]:
            return False
        if self._try("relax", move=[v], to=[new]) is None:
            return False
        self.t[v] = float(self.white.target(new[None])[0])
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

    def relax_all(self) -> int:
        live = self.chk.F[self.chk.F[:, 0] >= 0]
        n = 0
        for v in np.unique(live):
            n += self.relax(int(v))
        return n

    def measures(self, h: float) -> dict[str, Any]:
        V, F, _ = self.chk.export()
        E, r = self._edges()
        rim, _ = rim_of(F, len(V))
        nb = MC.vertex_neighbours(F, len(V))[0]
        deg = np.diff(nb)
        ang = np.degrees(SM._angles(V, F).min(1))
        q, dev, _ = SM.closest_on(V, F, cKDTree(V[F].mean(1)), self.white.V, 24)
        inside = ((q - self.white.V) * self.wn).sum(1) < 0
        din = np.where(inside, dev, 0.0) / h
        return {
            "vertices": len(V),
            "faces": len(F),
            "edge_over_target_p1_5_50_95_99": [round(float(x), 3) for x in np.percentile(r, [1, 5, 50, 95, 99])],
            "edge_in_range": round(float(((r >= LO) & (r <= HI)).mean()), 4),
            "valence_interior": {int(k): int(c) for k, c in zip(*np.unique(deg[~rim], return_counts=True), strict=True)},
            "min_angle_deg_p1_5_50": [round(float(x), 1) for x in np.percentile(ang, [1, 5, 50])],
            "deviation_max_vox": round(float(dev.max()) / h, 3),
            "deviation_p99_vox": round(float(np.percentile(dev, 99)) / h, 3),
            "inside_max_vox": round(float(din.max()), 3),
            "inside_over_0.5_1_2_4_vox": [int((din > t).sum()) for t in (0.5, 1, 2, 4)],
        }


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
                 progress: Any = None, eps_vox: float | None = None) -> tuple[TS.Mesh, list[dict[str, Any]], dict[str, Any]]:
    """The coarse Σ start of a site, sequentially remeshed; returns the
    mesh, the pass log and the checks of the start (the white mesh)."""
    h = float(site.vox.mean())
    gap = 0.25 * h
    white = White.of_site(site, lmin_vox)
    V, F = white.V.copy(), site.f.copy()
    rim, E = rim_of(F, len(V))
    hi = (np.array(site.labels.shape) - 1) * site.vox
    corner = box_corners(V, rim, E, np.zeros(3), hi, 1e-6 * h)
    rm = Remesher(white, V, F, gap, corner=corner, eps=None if eps_vox is None else eps_vox * h)
    start = rm.chk.check_all()
    start_info = {"gap_pairs": start.gap_pairs, "intersecting_pairs": start.intersecting_pairs,
                  "vertex_fold_pairs": start.vertex_fold_pairs, "creases": start.creases,
                  "rim": int(rim.sum()), "corners": int(corner.sum())}
    if progress:
        progress({"start": start_info})
    log = remesh(rm, h, max_passes, progress)
    Vo, Fo, _ = rm.chk.export()
    m = TS.Mesh(Vo, Fo, Vo.copy(), np.zeros(len(Vo), np.int64), np.zeros(len(Vo)))
    return m, log, start_info


def main() -> None:
    """usage: remesh_seq.py SITE_NPZ OUT_NPZ [max_passes] [lmin_vox] [eps_vox]
    (eps_vox: the no-cut check's eps in voxels; none: no check)"""
    import json
    import sys

    site = TP.load_site(sys.argv[1])
    mp = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    lv = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0
    ev = float(sys.argv[5]) if len(sys.argv) > 5 else None
    m, log, start = coarse_start(site, lv, mp, progress=lambda r: print(json.dumps(r), flush=True),
                                 eps_vox=ev)
    np.savez(sys.argv[2], V=m.V, F=m.F, log=json.dumps(log), start=json.dumps(start))


if __name__ == "__main__":
    main()
