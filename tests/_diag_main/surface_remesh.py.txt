"""Sequential isotropic remeshing (Botsch & Kobbelt 2004) of a surface over a
reference surface, every operation checked on its own: the one surface
remesher for the coarse Σ start and for T5's remesh (user, 2026-11-19:
"Sequential rewrite"; 2026-11-20: one remesher).

Ported from prototypes/scripts/remesh_seq.py (feature/top-surface, fd733d3,
a3857d4, bd5441d, 6f0ea61). The batch remesh of top_surface.remesh and the
coarse start of sigma_mesh.coarse_start are replaced by it.

The mesh lives in a mesh_checks.Checker. Each operation is one
Checker.propose(); the operation is kept (commit) only if it makes no gap
violation, crossing, shared-vertex fold or crease, no face turned against
the reference surface, and passes the normal check (below). Else it is
undone (rollback) and counted as refused, with its reason. So no batch
undo, no retries, no deadlock.

One pass, in the order of Botsch & Kobbelt:
  1. split the edges longer than 4/3 × target at their midpoint (projected
     onto the reference), the longest first, until none is left or the rest
     are refused;
  2. collapse the edges shorter than 4/5 × target, the shortest first, but
     not when an edge of the result would be longer than 4/3 × target;
  3. flip the interior edges whose flip brings the valences nearer to 6
     (4 on the rim);
  4. a 3D smoothing of each interior vertex (see below), not projected.
target = the mean of the two ends' targets; a vertex's target is the size
at its nearest reference vertex (baillarger.mesh_sizing).

Turned faces: a face is turned when its normal is at 90 degrees or more to
the normal of the reference face at the closest point to its centroid
(user, 2026-11-19: "White normal").

The normal check (user, 2026-11-19: orientation first, inward fine, outward
worse; theta 30 degrees, outward 0.5 voxels). Each reference vertex is
assigned to its closest face of the mesh and keeps its signed distance to
it (+: the mesh is outward of it). A face stands for the reference vertices
assigned to it: S, the sum of their area-weighted normals. An operation is
refused ("normal") when a face it changes, or a face whose assigned
vertices it changes, is more than theta from its S, and the worst such face
is worse than before the operation (so a start that already exceeds theta
somewhere does not block every operation there). It is refused ("outward")
when it leaves a reference vertex more than d_out inside the mesh's outer
side, and farther than before. Inward offsets are free. After an edit, the
reference vertices near the old and the new faces (within the faces' radius
+ 2 h + d_out) and those assigned to the removed or moved faces get their
closest face again (Checker.closest). A face with no reference vertex
assigned keeps only the turned test.

The relaxation is a 3D smoothing (Taubin 1995: a Jacobi sweep toward the
mean of the neighbours with lambda = 0.5, then one with mu = -0.53, which
does not shrink the surface), with no projection. Splits and collapses still
project their new points onto the reference surface.

The rim (the boundary, a crop's box) does not relax. Corners stay where they
are. A collapse of a rim edge or of a spoke (one end on the rim) moves the
other end onto the kept end; an edge between two rim vertices that is not on
the rim does not collapse. A rim edge splits at its midpoint.

Known limit (tau-shell, 2026-11-20): with no shape target, collapses refused
near folds leave sliver faces and creases at size transitions.
"""

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from baillarger import mesh_checks as MC
from baillarger.decimate import closest_points

Float = npt.NDArray[np.float64]
Int = npt.NDArray[np.int64]
Bool = npt.NDArray[np.bool_]

#: split above HI × target, collapse below LO × target
HI, LO = 4.0 / 3.0, 4.0 / 5.0


def closest_on(
    V: Float, F: Int, P: Float, k: int = 24, tree: cKDTree | None = None
) -> tuple[Float, Float, Int]:
    """The closest point on the mesh (V, F) to each point P, among the k faces
    whose centroids are nearest (tree: a cKDTree of the face centroids, if
    already made); its distance and its face."""
    k = min(k, len(F))
    t = cKDTree(V[F].mean(1)) if tree is None else tree
    _, cand = t.query(P, k=k)
    cand = np.asarray(cand, np.int64).reshape(len(P), k)
    f = F[cand.ravel()]
    Pr = np.repeat(P, k, axis=0)
    A, B, C = V[f[:, 0]], V[f[:, 1]], V[f[:, 2]]
    d, b0, b1, b2 = closest_points(Pr, A, B, C)
    j = np.argmin(d.reshape(len(P), k), 1)
    i = np.arange(len(P)) * k + j
    Q: Float = b0[i, None] * A[i] + b1[i, None] * B[i] + b2[i, None] * C[i]
    face: Int = cand[np.arange(len(P)), j]
    dist: Float = d[i]
    return Q, dist, face


class Reference:
    """The reference surface (the white surface): closest points, its unit
    face normals, and the target size at its vertices."""

    def __init__(self, V: npt.ArrayLike, F: npt.ArrayLike, size: npt.ArrayLike):
        self.V: Float = np.asarray(V, np.float64)
        self.F: Int = np.asarray(F, np.int64)
        self.size: Float = np.asarray(size, np.float64)
        N = MC.face_normals(self.V, self.F)
        self.N: Float = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-300)
        self.vtree = cKDTree(self.V)
        self.ftree = cKDTree(self.V[self.F].mean(1))

    def target(self, P: Float) -> Float:
        """The size at the nearest reference vertex of each point."""
        out: Float = self.size[np.asarray(self.vtree.query(P)[1], np.int64)]
        return out

    def project(self, P: Float, k: int = 12) -> tuple[Float, Float, Int]:
        """The closest points on the reference, their distances and faces,
        among the k faces whose centroids are nearest."""
        return closest_on(self.V, self.F, np.asarray(P, np.float64), k, self.ftree)


def rim_of(F: Int, n: int) -> tuple[Bool, Int]:
    """The boundary vertices (mask) and the boundary edges."""
    k, c = MC.edge_keys(F, n)
    kb = k[c == 1]
    E: Int = np.c_[kb // n, kb % n].astype(np.int64)
    rim: Bool = np.zeros(n, bool)
    rim[E.ravel()] = True
    return rim, E


def _rim_neighbours(rim_edges: Int) -> dict[int, list[int]]:
    nb: dict[int, list[int]] = {}
    for a, b in rim_edges:
        nb.setdefault(int(a), []).append(int(b))
        nb.setdefault(int(b), []).append(int(a))
    return nb


def box_corners(
    V: Float, rim: Bool, rim_edges: Int, lo: Float, hi: Float, tol: float
) -> Bool:
    """Rim vertices that are corners of a crop box (lo, hi): on two box
    planes or more, or with no box plane in common with both rim
    neighbours (a vertex's planes: those within tol, else the nearest)."""
    D = np.c_[V - lo, hi - V]  # distances to the 6 planes
    on = D < tol
    on[np.arange(len(V)), np.argmin(D, 1)] |= ~on.any(1)
    corner: Bool = rim & (on.sum(1) >= 2)
    for v, us in _rim_neighbours(rim_edges).items():
        if len(us) != 2 or not (on[v] & on[us[0]] & on[us[1]]).any():
            corner[v] = True
    return corner


def angle_corners(
    V: Float, rim: Bool, rim_edges: Int, cos_max: float = float(np.cos(np.radians(45)))
) -> Bool:
    """Rim vertices where the rim turns by more than 45 degrees (or that do
    not have exactly two rim neighbours)."""
    corner: Bool = np.zeros(len(V), bool)
    for v, us in _rim_neighbours(rim_edges).items():
        if len(us) != 2:
            corner[v] = True
            continue
        u, w = V[v] - V[us[0]], V[us[1]] - V[v]
        c = float(u @ w) / max(float(np.linalg.norm(u) * np.linalg.norm(w)), 1e-300)
        corner[v] = c < cos_max
    out: Bool = corner & rim
    return out


@dataclass
class _Assignment:
    """A tentative new assignment of reference vertices to faces."""

    w: Int
    f: Int
    sd: Float
    S: dict[int, Float]
    C: dict[int, int]
    removed: set[int]


class Remesher:
    """The sequential remesher on a live mesh (V, F) over a Reference.

    gap: the Checker's gap; theta_deg and d_out: the normal check and the
    outward bound (None: off); corner: rim vertices that stay (default: where
    the rim turns by more than 45 degrees); h: the voxel size (the normal
    check's search radius)."""

    def __init__(
        self,
        ref: Reference,
        V: npt.ArrayLike,
        F: npt.ArrayLike,
        gap: float,
        fold_cos: float = MC.FOLD_COS,
        corner: Bool | None = None,
        theta_deg: float | None = None,
        d_out: float | None = None,
        h: float = 1.0,
    ):
        Vf = np.asarray(V, np.float64)
        self.ref, self.gap = ref, gap
        self.cos_t = None if theta_deg is None else float(np.cos(np.radians(theta_deg)))
        self.d_out = d_out
        self.rn = 2.0 * h + (d_out or 0.0)
        self.rs = 2.0 * HI * float(ref.size.max())
        self.wn = MC.vertex_normals(ref.V, ref.F)
        cr = MC.face_normals(ref.V, ref.F)
        #: the area-weighted normals of the reference vertices
        self.Wn: Float = np.zeros((len(ref.V), 3))
        for k in range(3):
            np.add.at(self.Wn, ref.F[:, k], cr)
        top = MC.Topology.of(np.asarray(F, np.int64), len(Vf))
        self.chk = MC.Checker(top, Vf, gap, fold_cos=fold_cos)
        rim, E = rim_of(top.F, len(Vf))
        self.rim: Bool = rim
        self.corner: Bool = (
            angle_corners(Vf, rim, E) if corner is None else corner & rim
        )
        self.t: Float = ref.target(Vf)
        self.refused: Counter[str] = Counter()
        #: places (mm) where measures() reports the dihedral angles
        self.places: Float = np.zeros((0, 3))
        nw = len(ref.V)
        self.wf: Int = np.full(nw, -1, np.int64)
        self.wd: Float = np.zeros(nw)
        self.Sf: Float = np.zeros((len(self.chk.F), 3))
        #: the number of reference vertices assigned to each face (a face
        #: with none is unassigned; its S may keep a round-off residue)
        self.Cf: Int = np.zeros(len(self.chk.F), np.int64)
        self.fw: dict[int, set[int]] = {}
        if self.checking:
            _, _, f = self.chk.closest(ref.V, self.rs)
            self.wf[:] = f
            for w in np.flatnonzero(f >= 0):
                self.Sf[f[w]] += self.Wn[w]
                self.Cf[f[w]] += 1
                self.fw.setdefault(int(f[w]), set()).add(int(w))

    @property
    def checking(self) -> bool:
        """The normal check or the outward bound is on."""
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
        """The neighbours of vertex v."""
        F = self.chk.F
        return {int(u) for f in self.chk.faces_of(v) for u in F[f]} - {v}

    def edge_faces(self, a: int, b: int) -> list[int]:
        """The faces of edge (a, b)."""
        F = self.chk.F
        return [int(f) for f in self.chk.faces_of(a) if b in F[f]]

    def turned(self, faces: Int) -> bool:
        """A face turned against the reference normal at its centroid's
        closest point (or of zero area)."""
        if not len(faces):
            return False
        V, F = self.chk.V, self.chk.F[faces]
        n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        _, _, wf = self.ref.project(V[F].mean(1))
        return bool(((n * self.ref.N[wf]).sum(1) <= 0).any())

    def _tris(self, faces: Int) -> Float:
        out: Float = self.chk.V[self.chk.F[faces]]
        return out

    def _near_ref(self, T: Float) -> Int:
        """The reference vertices within rn of the triangles T (a superset:
        within rn + the triangle's radius of its centroid)."""
        if not len(T):
            return np.zeros(0, np.int64)
        C = T.mean(1)
        R = np.linalg.norm(T - C[:, None], axis=2).max(1) + self.rn
        lists: Any = self.ref.vtree.query_ball_point(C, R)
        return np.unique(np.fromiter((j for x in lists for j in x), np.int64))

    def _cos(self, faces: Int, S: Float, C: npt.ArrayLike) -> Float:
        """cos(face normal, S) per face; 1 where no reference vertex is
        assigned (C = 0)."""
        if not len(faces):
            return np.ones(0)
        n = MC.face_normals(self.chk.V, self.chk.F[faces])
        ns = np.linalg.norm(n, axis=1) * np.linalg.norm(S, axis=1)
        out: Float = np.where(
            np.asarray(C) > 0, (n * S).sum(1) / np.maximum(ns, 1e-300), 1.0
        )
        return out

    def _normal_check(
        self,
        ofaces: Int,
        cos_old: dict[int, float],
        old: Float,
        r: MC.EditCheck,
        faces: Int,
        removed: set[int],
    ) -> tuple[str, _Assignment | None]:
        """After a proposed edit: the new assignment of the affected
        reference vertices and the reason to refuse ("" if none)."""
        chk = self.chk
        self._grow()
        w = np.union1d(self._near_ref(old), self._near_ref(self._tris(faces)))
        extra = [x for f in ofaces for x in self.fw.get(int(f), ())]
        w = np.union1d(w, np.asarray(extra, np.int64)).astype(np.int64)
        if not len(w):
            return "", None
        W = self.ref.V[w]
        # the closest face within r is the closest face: search a small
        # radius first, then larger ones for the misses only
        T = self._tris(faces)
        r1 = self.rn
        if len(T):
            r1 += 2.0 * float(np.linalg.norm(T - T.mean(1)[:, None], axis=2).max())
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
        S: dict[int, Float] = {x: np.zeros(3) for x in removed | new_ids}
        C: dict[int, int] = dict.fromkeys(removed | new_ids, 0)

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
            after = self._cos(la, np.array([S[x] for x in live]), [C[x] for x in live])
            other = [x for x in live if x not in cos_old and x not in new_ids]
            if other:
                oa = np.asarray(other, np.int64)
                cb = self._cos(oa, self.Sf[oa], self.Cf[oa])
                for y, c in zip(other, cb, strict=True):
                    cos_old[y] = float(c)
            worst_before = min([*cos_old.values(), 1.0])
            if after.min() < min(self.cos_t, worst_before) - 1e-12:
                return "normal", None
        return "", _Assignment(w, f.astype(np.int64), sd, S, C, removed)

    def _apply(self, a: _Assignment) -> None:
        for x in a.removed:
            self.fw.pop(x, None)
        for j, x in enumerate(a.w):
            fo, fn = int(self.wf[x]), int(a.f[j])
            if fo >= 0 and fo not in a.removed:
                s = self.fw.get(fo)
                if s is not None:
                    s.discard(int(x))
            if fn >= 0:
                self.fw.setdefault(fn, set()).add(int(x))
        self.wf[a.w] = a.f
        self.wd[a.w] = a.sd
        for x, v in a.S.items():
            self.Sf[x] = v
            self.Cf[x] = a.C[x]

    def _try(
        self,
        kind: str,
        move: list[int] | None = None,
        to: Float | None = None,
        remove_faces: list[int] | None = None,
        add_faces: npt.ArrayLike | None = None,
        add_vertices: Float | None = None,
    ) -> MC.EditCheck | None:
        """Propose an edit; keep it if every check passes (returns its
        check), else undo it (None)."""
        mv0 = [] if move is None else [int(v) for v in move]
        rem = [] if remove_faces is None else [int(x) for x in remove_faces]
        ofaces = np.zeros(0, np.int64)
        old = np.zeros((0, 3, 3))
        cos_old: dict[int, float] = {}
        if self.checking:
            ofaces = np.unique(
                np.r_[np.asarray(rem, np.int64), *[self.chk.faces_of(v) for v in mv0]]
            ).astype(np.int64)
            old = self._tris(ofaces)
            cs = self._cos(ofaces, self.Sf[ofaces], self.Cf[ofaces])
            cos_old = {int(x): float(c) for x, c in zip(ofaces, cs, strict=True)}
        r = self.chk.propose(
            move=mv0,
            to=to,
            remove_faces=rem,
            add_faces=add_faces,
            add_vertices=add_vertices,
        )
        reason, state = "", None
        if r.gap_pairs:
            reason = "gap"
        elif r.intersecting_pairs or r.vertex_fold_pairs:
            reason = "cross"
        elif r.creases:
            reason = "crease"
        else:
            faces = np.unique(
                np.r_[r.new_faces, *[self.chk.faces_of(v) for v in mv0]].astype(
                    np.int64
                )
            )
            if self.turned(faces):
                reason = "turned"
            elif self.checking:
                reason, state = self._normal_check(
                    ofaces, cos_old, old, r, faces, set(rem)
                )
        if reason:
            self.chk.rollback()
            self.refused[f"{kind}_{reason}"] += 1
            return None
        self.chk.commit()
        self._grow()
        if state is not None:
            self._apply(state)
        return r

    def _edges(self) -> tuple[Int, Float]:
        """The live edges (a < b) and their lengths over their targets."""
        F = self.chk.F
        F = F[F[:, 0] >= 0]
        E: Int = np.unique(
            np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1), axis=0
        )
        V = self.chk.V
        L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
        out: Float = L / (0.5 * (self.t[E[:, 0]] + self.t[E[:, 1]]))
        return E, out

    def ratio(self, a: int, b: int) -> float:
        """The length of edge (a, b) over its target."""
        V = self.chk.V
        return float(np.linalg.norm(V[a] - V[b]) / (0.5 * (self.t[a] + self.t[b])))

    # ------------------------------------------------------------ operations

    def split(self, a: int, b: int) -> bool:
        """Split edge (a, b) at its midpoint (projected, unless on the rim)."""
        ef = self.edge_faces(a, b)
        if len(ef) not in (1, 2):
            self.refused["split_nonmanifold"] += 1
            return False
        mid = 0.5 * (self.chk.V[a] + self.chk.V[b])
        rim_edge = len(ef) == 1
        m = mid if rim_edge else self.ref.project(mid[None])[0][0]
        F = self.chk.F
        add = []
        for f in ef:
            row = [int(u) for u in F[f]]
            k = next(i for i in range(3) if {row[i], row[(i + 1) % 3]} == {a, b})
            p, q, o = row[k], row[(k + 1) % 3], row[(k + 2) % 3]
            add += [[p, -1, o], [-1, q, o]]
        r = self._try("split", add_vertices=m[None], remove_faces=ef, add_faces=add)
        if r is None:
            return False
        v = int(r.new_vertices[0])
        self.rim[v] = rim_edge
        self.corner[v] = False
        self.t[v] = float(self.ref.target(m[None])[0])
        return True

    def collapse(self, a: int, b: int) -> bool:
        """Collapse edge (a, b): to its projected midpoint, or onto the rim
        end (a spoke) or the end that is not a corner (a rim edge)."""
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
            pos = self.ref.project((0.5 * (V[a] + V[b]))[None])[0][0]
            need = 2
        if len(ef) != need or len(common) != need:
            self.refused["collapse_link"] += 1
            return False
        tk = (
            float(self.ref.target(pos[None])[0])
            if not (ra or rb)
            else float(self.t[keep])
        )
        for u in (self.nbrs(rem) | self.nbrs(keep)) - {keep, rem}:
            if np.linalg.norm(pos - V[u]) > HI * 0.5 * (tk + self.t[u]):
                self.refused["collapse_long"] += 1
                return False
        rest = [int(f) for f in self.chk.faces_of(rem) if f not in ef]
        Fr = self.chk.F[rest].copy()
        Fr[Fr == rem] = keep
        r = self._try(
            "collapse", move=[keep], to=pos[None], remove_faces=ef + rest, add_faces=Fr
        )
        if r is None:
            return False
        self.t[keep] = tk
        return True

    def valence_target(self, v: int) -> int:
        """6 inside, 4 on the rim."""
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
        r = self._try("flip", remove_faces=[f, g], add_faces=[[c, a, d], [c, d, b]])
        return r is not None

    def relax(self, v: int, target: Float) -> bool:
        """Move v to target (3D, no projection)."""
        if self.rim[v]:
            return False
        p = self.chk.V[v]
        if np.linalg.norm(target - p) < 1e-9 * self.t[v]:
            return False
        if self._try("relax", move=[v], to=target[None]) is None:
            return False
        self.t[v] = float(self.ref.target(target[None])[0])
        return True

    # ------------------------------------------------------------ passes

    def split_all(self, sweeps: int = 10) -> int:
        """Split the long edges, longest first, until none is left."""
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
        """Collapse the short edges, shortest first."""
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
        """Flip the edges whose flip improves the valences."""
        E, _ = self._edges()
        n = 0
        for a, b in E:
            a, b = int(a), int(b)
            if len(self.chk.faces_of(a)) == 0 or b not in self.nbrs(a):
                continue
            n += bool(self.flip(a, b))
        return n

    def relax_all(self, lam: float = 0.5, mu: float = -0.53) -> int:
        """One Taubin step: a sweep with lam, then one with mu. Each sweep is
        a Jacobi step: every vertex moves toward the mean of its neighbours'
        positions at the start of the sweep (the form that does not shrink);
        the moves are then tried one at a time."""
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
        """Edge lengths over target, valences, angles and the measures
        against the reference (surface_measures)."""
        V, F, _ = self.chk.export()
        _, r = self._edges()
        rim, _ = rim_of(F, len(V))
        deg = np.diff(MC.vertex_neighbours(F, len(V))[0])
        ang = np.degrees(MC.face_angles(V, F).min(1))
        vi, vc = np.unique(deg[~rim], return_counts=True)
        return {
            "vertices": len(V),
            "faces": len(F),
            "edge_over_target_p1_5_50_95_99": [
                round(float(x), 3) for x in np.percentile(r, [1, 5, 50, 95, 99])
            ],
            "edge_in_range": round(float(((r >= LO) & (r <= HI)).mean()), 4),
            "valence_interior": {int(k): int(c) for k, c in zip(vi, vc, strict=True)},
            "min_angle_deg_p1_5_50": [
                round(float(x), 1) for x in np.percentile(ang, [1, 5, 50])
            ],
            **surface_measures(V, F, self.ref, self.Wn, self.wn, h, self.places),
        }


def surface_measures(
    V: Float,
    F: Int,
    ref: Reference,
    Wn: Float,
    wn: Float,
    h: float,
    places: Float,
    radius: float = 1.5,
) -> dict[str, Any]:
    """Measures of the mesh (V, F) against the reference, from scratch: each
    reference vertex's closest face, its signed offset (+: the mesh outward
    of it), each face's angle to S (its assigned reference vertices' summed
    area-weighted normals), and the dihedral angles (roughness), over all
    and within radius of each of the places."""
    q, dev, fc = closest_on(V, F, ref.V, 24)
    side = ((q - ref.V) * wn).sum(1)
    out = np.where(side > 0, dev, 0.0) / h
    din = np.where(side < 0, dev, 0.0) / h
    S = np.zeros((len(F), 3))
    np.add.at(S, fc, Wn)
    has = np.linalg.norm(S, axis=1) > 0
    n = MC.face_normals(V, F)
    den = np.maximum(np.linalg.norm(n, axis=1) * np.linalg.norm(S, axis=1), 1e-300)
    fa = np.degrees(np.arccos(np.clip((n * S).sum(1) / den, -1, 1)))[has]
    di, mid = MC.edge_dihedrals(V, F)

    def pct(x: Float, p: list[int]) -> list[float]:
        return [round(float(y), 1) for y in np.nanpercentile(x, p)] if len(x) else []

    res: dict[str, Any] = {
        "deviation_max_vox": round(float(dev.max()) / h, 3),
        "outward_max_vox": round(float(out.max()), 3),
        "outward_over_0.5_1_vox": [int((out > t).sum()) for t in (0.5, 1)],
        "inside_max_vox": round(float(din.max()), 3),
        "inside_over_0.5_1_2_4_vox": [int((din > t).sum()) for t in (0.5, 1, 2, 4)],
        "face_angle_p50_90_99_max": pct(fa, [50, 90, 99, 100]),
        "faces_over_30_45_60": [int((fa > t).sum()) for t in (30, 45, 60)],
        "faces_unassigned": int((~has).sum()),
        "dihedral_p50_90_99_max": pct(di, [50, 90, 99, 100]),
    }
    if len(places):
        res["dihedral_at_places_p50_90_max"] = [
            pct(di[np.linalg.norm(mid - p, axis=1) < radius], [50, 90, 100])
            for p in places
        ]
    return res


def remesh(
    rm: Remesher,
    h: float,
    max_passes: int = 20,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Passes until one makes no split, collapse or flip (or max_passes);
    the log of each pass."""
    log = []
    for k in range(max_passes):
        t0 = time.time()
        rm.refused.clear()
        sp = rm.split_all()
        co = rm.collapse_all()
        fl = rm.flip_all()
        rl = rm.relax_all()
        rec = {
            "pass": k,
            "splits": sp,
            "collapses": co,
            "flips": fl,
            "relaxed": rl,
            "refused": dict(rm.refused),
            **rm.measures(h),
            "s": round(time.time() - t0, 1),
        }
        log.append(rec)
        if progress:
            progress(rec)
        if sp == 0 and co == 0 and fl == 0:
            break
    return log
