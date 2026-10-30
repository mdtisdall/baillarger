"""The editor's re-solve: a correction field on the white mesh (plan: reports/Pial editing
and edit priority.md, §3, *The fast re-solve*; §6 step 3). Prototype only.

The pial thickness is t = t_base + dt. dt is the posterior mean of a Gaussian field on
the white mesh's vertices, conditioned on the edits.

- Prior: mean 0, precision Q = tau^2 L M^-1 L with L = kappa^2 M + K, K the cotangent
  stiffness matrix and M the lumped (barycentric) mass matrix: the SPDE form of a
  Matern field with nu = 1 on a surface (Lindgren, Rue and Lindstrom 2011, alpha = 2).
  Range rho = sqrt(8) / kappa (correlation about 0.14 at rho; kappa r K1(kappa r) in the
  plane); marginal SD sigma0 with tau^2 = 1 / (4 pi kappa^2 sigma0^2). The kappa^2 term
  is the level term: dt returns to 0 far from edits. The mesh's open edges (box faces)
  are natural (Neumann) boundaries.
- Q^-1 = tau^-2 L^-1 M L^-1, so only L is factored (sparse LU, once per range); Q is
  never formed. Each observation a_k (a barycentric row) adds one column
  x_k = Q^-1 a_k (two solves with the factor), cached.
- Observations: y_k = a_k . t with SD sigma_e. Posterior mean
  dt = X (A X + sigma_e^2 I)^-1 (y - A t_base), X = [x_1 ... x_k]. Adding edits one at
  a time gives the same dt as solving for all at once (the formula has no order).

Units: mm for the mesh and the field.

Session (below) ties the field to pial_height's geometry: a click becomes a top edit
(the carried root and arc length at the click), a contact edit (that plus the opposing
column passing nearest the click), or a free edit (a 3D displacement field on the pial
vertices, the same prior per component, observed at the pial point nearest the click).
Edits are stored as world positions and replayed onto any base (replay).
"""

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse.linalg import splu

sys.path.insert(0, str(Path(__file__).parent.parent))

type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]


# ------------------------------------------------------------------ mesh matrices


def cotan_matrices(v: Float, f: Int) -> tuple[sparse.csr_matrix, Float]:
    """Cotangent stiffness K (positive semidefinite, rows sum to 0) and lumped mass
    (a third of the area of each triangle at each corner)."""
    n = len(v)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cr = np.cross(b - a, c - a)
    dbl = np.linalg.norm(cr, axis=1)  # twice the area
    dbl_safe = np.maximum(dbl, 1e-300)
    rows, cols, vals = [], [], []
    # the cotangent at corner k weights the opposite edge (i, j)
    for k, i, j in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        pk, pi, pj = v[f[:, k]], v[f[:, i]], v[f[:, j]]
        cot = np.sum((pi - pk) * (pj - pk), 1) / dbl_safe
        w = 0.5 * cot
        rows += [f[:, i], f[:, j], f[:, i], f[:, j]]
        cols += [f[:, j], f[:, i], f[:, i], f[:, j]]
        vals += [-w, -w, w, w]
    K = sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsr()
    mass = np.zeros(n)
    for k in range(3):
        np.add.at(mass, f[:, k], dbl / 6.0)
    return K, mass


# ------------------------------------------------------------------ the prior


def weld(v: Float, f: Int, rel: float) -> tuple[Int, Int, int]:
    """Vertices within rel x the median edge length of each other, joined (transitively)
    into one. Returns (welded index of every vertex, the faces on welded indices without
    those that collapse, the number of vertices merged away)."""
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    e = np.linalg.norm(v[f[:, [1, 2, 0]]] - v[f], axis=-1)
    tol = rel * float(np.median(e))
    pairs = cKDTree(v).query_pairs(tol, output_type="ndarray")
    n = len(v)
    if len(pairs) == 0:
        return np.arange(n), f, 0
    G = sparse.coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)
    )
    _, rep = connected_components(G, directed=False)
    fw = rep[f]
    keep = (fw[:, 0] != fw[:, 1]) & (fw[:, 1] != fw[:, 2]) & (fw[:, 2] != fw[:, 0])
    return rep.astype(np.int64), fw[keep], n - int(rep.max()) - 1


class Prior:
    """The Matern-type prior on a mesh, factored for solves with Q."""

    def __init__(
        self,
        v_mm: Float,
        f: Int,
        range_mm: float = 2.0,
        sd_mm: float = 0.5,
        weld_rel: float = 1e-3,
    ):
        t0 = time.perf_counter()
        self.v, self.f = np.asarray(v_mm, np.float64), np.asarray(f, np.int64)
        # Vertices closer than weld_rel x the median edge are one point: a triangle with
        # such an edge has (near) zero area and cotangent weights up to 1e294, which
        # makes L singular or the factorisation slow and wrong (V1V2: 124 such triangles
        # after the VTK's 0.1 um rounding). They are welded; the triangles that collapse
        # are dropped. The field lives on the welded vertices; rep maps each vertex to
        # its welded one.
        self.rep, fw, self.n_welded = weld(self.v, self.f, weld_rel)
        vw = np.zeros((self.rep.max() + 1, 3))
        vw[self.rep] = self.v
        self.K, self.mass = cotan_matrices(vw, fw)
        # vertices with no area left (only collapsed triangles) would make L singular
        self.mass = np.maximum(self.mass, 1e-12 * max(self.mass.max(), 1e-300))
        self.seconds_matrices = time.perf_counter() - t0
        self.set_range(range_mm, sd_mm)

    def set_range(self, range_mm: float, sd_mm: float) -> None:
        t0 = time.perf_counter()
        self.range_mm, self.sd_mm = float(range_mm), float(sd_mm)
        self.kappa = np.sqrt(8.0) / self.range_mm
        self.tau2 = 1.0 / (4.0 * np.pi * self.kappa**2 * self.sd_mm**2)
        L = (self.kappa**2 * sparse.diags(self.mass) + self.K).tocsc()
        # L is symmetric positive definite: pivot on the diagonal so the symmetric
        # ordering survives (SuperLU's default partial pivoting destroys it: on a 320k
        # vertex mesh, > 10 min and > 11 GB against 2.4 s and 1.3 GB)
        self.lu = splu(
            L,
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0.0,
            options=dict(SymmetricMode=True),
        )
        self.seconds_factor = time.perf_counter() - t0

    @property
    def n(self) -> int:
        return len(self.v)

    def cov_times(self, b: Float) -> Float:
        """Q^-1 b (b: (n,) or (n, k), on the original vertices)."""
        b = np.asarray(b, np.float64)
        nw = len(self.mass)
        if b.ndim == 1:
            bw = np.bincount(self.rep, b, minlength=nw)
        else:
            bw = np.zeros((nw, b.shape[1]))
            np.add.at(bw, self.rep, b)
        y = self.lu.solve(bw)
        y = (self.mass[:, None] * y) if y.ndim == 2 else self.mass * y
        return (self.lu.solve(y) / self.tau2)[self.rep]

    def dense_precision(self) -> Float:
        """Q as a dense matrix on the welded vertices (small meshes only; for tests)."""
        L = (self.kappa**2 * sparse.diags(self.mass) + self.K).toarray()
        return self.tau2 * L @ np.diag(1.0 / self.mass) @ L


# ------------------------------------------------------------------ a field conditioned on point observations


@dataclass
class Obs:
    idx: Int  # vertices (3,)
    w: Float  # barycentric weights (3,)
    y: float  # observed value (mm)


class Field:
    """Posterior mean of base + a prior-distributed correction, given observations."""

    def __init__(self, prior: Prior, base: Float, sd_obs_mm: float = 0.03):
        self.prior = prior
        self.base = np.asarray(base, np.float64)
        self.sd_obs = float(sd_obs_mm)
        self.obs: dict[int, Obs] = {}
        self.cols: dict[int, Float] = {}  # Q^-1 a_k
        self._next = 0

    def row(self, o: Obs) -> Float:
        a = np.zeros(self.prior.n)
        np.add.at(a, o.idx, o.w)
        return a

    def add(self, idx, w, y: float, col: Float | None = None) -> int:
        """An observation; col: its column Q^-1 a when already known (fields sharing a
        prior and a row share it)."""
        if not np.isfinite(y) or not np.all(np.isfinite(w)):
            raise ValueError(f"observation not finite: y {y}, weights {w}")
        o = Obs(np.asarray(idx, np.int64), np.asarray(w, np.float64), float(y))
        k = self._next
        self._next += 1
        self.obs[k] = o
        self.cols[k] = self.prior.cov_times(self.row(o)) if col is None else col
        return k

    def remove(self, k: int) -> None:
        self.obs.pop(k)
        self.cols.pop(k)

    def set_base(self, base: Float) -> None:
        self.base = np.asarray(base, np.float64)

    def refresh(self) -> None:
        """Recompute the cached columns (after the prior's range changes)."""
        for k, o in self.obs.items():
            self.cols[k] = self.prior.cov_times(self.row(o))

    def correction(self) -> Float:
        if not self.obs:
            return np.zeros(self.prior.n)
        keys = list(self.obs)
        X = np.stack([self.cols[k] for k in keys], 1)
        A_idx = [self.obs[k] for k in keys]
        AX = np.stack(
            [np.sum(X[o.idx] * o.w[:, None], 0) for o in A_idx], 0
        )  # (k, k) = A X
        S = 0.5 * (AX + AX.T) + self.sd_obs**2 * np.eye(len(keys))
        r = np.array([o.y - np.dot(self.base[o.idx], o.w) for o in A_idx])
        return X @ np.linalg.solve(S, r)

    def mean(self) -> Float:
        return self.base + self.correction()


# ------------------------------------------------------------------ geometry helpers


def barycentric(P: Float, tri: Float) -> Float:
    """Barycentric coordinates of points P (n, 3) in triangles tri (n, 3, 3), projected
    onto the triangle's plane."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    v0, v1, v2 = b - a, c - a, P - a
    d00 = np.sum(v0 * v0, 1)
    d01 = np.sum(v0 * v1, 1)
    d11 = np.sum(v1 * v1, 1)
    d20 = np.sum(v2 * v0, 1)
    d21 = np.sum(v2 * v1, 1)
    den = np.maximum(d00 * d11 - d01 * d01, 1e-300)
    w1 = (d11 * d20 - d01 * d21) / den
    w2 = (d00 * d21 - d01 * d20) / den
    return np.stack([1 - w1 - w2, w1, w2], 1)


def polyline_nearest(pts: Float, x: Float) -> tuple[float, float]:
    """Distance from x to a polyline (K, 3; nan rows ignored) and the arc length of the
    nearest point."""
    p = pts[np.isfinite(pts).all(1)]
    if len(p) < 2:
        return np.inf, np.nan
    a, b = p[:-1], p[1:]
    ab = b - a
    L = np.sum(ab * ab, 1)
    u = np.clip(np.sum((x - a) * ab, 1) / np.maximum(L, 1e-300), 0, 1)
    q = a + u[:, None] * ab
    d = np.linalg.norm(q - x, axis=1)
    i = int(np.argmin(d))
    s0 = np.concatenate([[0.0], np.cumsum(np.sqrt(L))])
    return float(d[i]), float(s0[i] + u[i] * np.sqrt(L[i]))


class Locator:
    """Closest points on a triangle mesh, with the vertex-to-face table built once
    (mesh_smoothing.closest_on_mesh rebuilds it on every call). move() swaps the
    vertex positions (same faces) and rebuilds only the KD-tree."""

    def __init__(self, v: Float, f: Int, k: int = 8):
        nv = len(v)
        order = np.argsort(f.ravel(), kind="stable")
        fv = f.ravel()[order]
        fid = order // 3
        st = np.searchsorted(fv, np.arange(nv + 1))
        val = np.diff(st)
        self.table = -np.ones((nv, int(val.max())), np.int64)
        for j in range(self.table.shape[1]):
            has = val > j
            self.table[has, j] = fid[st[:-1][has] + j]
        self.f, self.k = f, k
        self.move(v)

    def move(self, v: Float) -> None:
        from scipy.spatial import cKDTree

        self.v = np.asarray(v, np.float64)
        self.tree = cKDTree(self.v)

    def closest(self, X: Float) -> tuple[Float, Int, Float]:
        """Closest points, their faces and barycentric weights, for points X (n, 3)."""
        import mesh_smoothing as M

        X = np.atleast_2d(np.asarray(X, np.float64))
        _, nn = self.tree.query(X, k=self.k)
        cand = self.table[nn].reshape(len(X), -1)
        best = np.full(len(X), np.inf)
        P, F, B = np.zeros_like(X), np.zeros(len(X), np.int64), np.zeros((len(X), 3))
        for j in range(cand.shape[1]):
            c = cand[:, j]
            ok = c >= 0
            cc = np.where(ok, c, 0)
            tri = self.f[cc]
            q, bary = M._closest_on_tri(
                X, self.v[tri[:, 0]], self.v[tri[:, 1]], self.v[tri[:, 2]]
            )
            d = np.where(ok, np.sum((q - X) ** 2, 1), np.inf)
            better = d < best
            best[better], P[better], F[better], B[better] = (
                d[better],
                q[better],
                cc[better],
                bary[better],
            )
        return P, F, B


# ------------------------------------------------------------------ the editing session


@dataclass
class Edit:
    kind: str  # "top", "contact", "free"
    mm: list  # world position
    vox: list  # crop voxel position (0-based)
    roots: list = field(
        default_factory=list
    )  # root positions (voxels) at the time of the edit
    s_mm: list = field(default_factory=list)  # arc lengths at the time of the edit
    miss_mm: list = field(default_factory=list)
    fallback: bool = False  # a top edit that fell back to a free edit
    time: float = 0.0
    obs: list = field(
        default_factory=list
    )  # (field name, observation key) pairs, not saved
    vertex: int = -1  # a drag's base vertex
    pushes: list = field(
        default_factory=list
    )  # a drag's pushes: [vertex, obs pairs, target voxels]


class Session:
    """Edits on one patch: white mesh v (voxels), faces f, outward normals n; the march
    Geometry g; the traced Columns cols of every vertex; the base thickness t_base
    (voxels, per vertex); affine: voxel -> world mm."""

    def __init__(
        self,
        v,
        f,
        n,
        g,
        cols,
        t_base,
        affine=None,
        range_mm=2.0,
        sd_mm=0.5,
        sd_edit_mm=0.03,
        miss_max_vox=1.0,
        contact_min_sep_mm=1.0,
    ):
        import pial_height as P

        self.P = P
        self.h = P.H_MM
        self.v, self.f, self.n = (
            np.asarray(v, np.float64),
            np.asarray(f, np.int64),
            np.asarray(n, np.float64),
        )
        self.g, self.cols = g, cols
        self.affine = np.eye(4) if affine is None else np.asarray(affine, np.float64)
        self.prior = Prior(self.v * self.h, self.f, range_mm, sd_mm)
        self.t_base = np.asarray(t_base, np.float64)  # voxels
        self.thick = Field(self.prior, self.t_base * self.h, sd_edit_mm)
        self.free = [
            Field(self.prior, np.zeros(len(self.v)), sd_edit_mm) for _ in range(3)
        ]
        self.miss_max = miss_max_vox
        self.contact_min_sep = contact_min_sep_mm / self.h
        self.edits: list[Edit] = []
        from scipy.spatial import cKDTree

        self.tree = cKDTree(self.v)
        self.white = Locator(self.v, self.f)
        self._pial = None  # Locator over the current pial positions (free edits)

    # -- coordinates
    def to_mm(self, x_vox):
        x = np.asarray(x_vox, np.float64)
        return x @ self.affine[:3, :3].T + self.affine[:3, 3]

    def to_vox(self, x_mm):
        A = np.linalg.inv(self.affine)
        return np.asarray(x_mm, np.float64) @ A[:3, :3].T + A[:3, 3]

    # -- columns
    def _root_obs(self, R):
        """Face vertices and barycentric weights of the white-mesh point nearest R."""
        _, F, B = self.white.closest(np.atleast_2d(R))
        return self.f[F[0]], B[0]

    def _column_of(self, idx, w):
        """The column through a mesh point: barycentric blend of its vertices' polylines."""
        pts = np.tensordot(w, self.cols.pts[idx].astype(np.float64), axes=(0, 0))
        n = int(self.cols.n[idx].min())
        pts[n:] = np.nan
        return pts

    def thickness(self) -> Float:
        """Final thickness per vertex (voxels): the base plus the correction."""
        if not self.thick.obs:
            return self.t_base.copy()
        return self.t_base + self.thick.correction() / self.h

    def positions(self, rows=None, t=None, free=None) -> Float:
        """Pial vertex positions (voxels) of all vertices or of `rows`: the column point
        at the final thickness, plus the free displacement. t, free: the thickness
        (voxels) and the free displacement (mm, (n, 3)) of all vertices, if known."""
        rows = np.arange(len(self.v)) if rows is None else np.asarray(rows, np.int64)
        t = self.thickness() if t is None else t
        p = self.P.point_at(self.cols, t[rows], rows)
        bad = ~np.isfinite(p).all(1)
        p[bad] = self.v[rows[bad]]
        if free is None and any(fd.obs for fd in self.free):
            free = self.free_displacement()
        if free is not None:
            p = p + free[rows] / self.h
        return p

    def free_displacement(self) -> Float | None:
        if not any(fd.obs for fd in self.free):
            return None
        return np.stack([fd.correction() for fd in self.free], 1)

    # -- edits
    @staticmethod
    def _finite(x_vox):
        x = np.asarray(x_vox, np.float64)
        if x.shape != (3,) or not np.all(np.isfinite(x)):
            raise ValueError(f"edit position not finite: {x_vox}")
        return x

    def add_top(self, x_vox, kind="top", defer_free=False) -> Edit:
        """A top at the click: the carried root and arc length there. If the click is
        more than miss_max from that root's column it falls back to a free edit, applied
        now or (defer_free) left for the caller to apply after every top edit."""
        x = self._finite(x_vox)
        R, s = self.P.click_column(self.g, x)
        idx, w = self._root_obs(R[0])
        miss, s_line = polyline_nearest(self._column_of(idx, w), x)
        e = Edit(
            kind,
            self.to_mm(x).tolist(),
            x.tolist(),
            [R[0].tolist()],
            [float(s[0]) * self.h],
            [miss * self.h],
            time=time.time(),
        )
        if not np.isfinite(s[0]) or miss > self.miss_max:
            e.fallback = True
            if defer_free:
                self.edits.append(e)
                return e
            return self._free(x, e)
        k = self.thick.add(idx, w, float(s[0]) * self.h)
        e.obs.append(("thick", k))
        self.edits.append(e)
        return e

    def add_contact(self, x_vox, defer_free=False) -> Edit:
        """A top for the column the march carries to x, and one for the nearest column
        that reaches x from the other side (roots more than contact_min_sep apart,
        directions at x opposed)."""
        x = self._finite(x_vox)
        e = self.add_top(x, kind="contact", defer_free=defer_free)
        if e.fallback:
            return e
        R1 = np.asarray(e.roots[0])
        idx1, w1 = self._root_obs(R1)
        c1 = self._column_of(idx1, w1)
        d1 = self._direction_at(c1, x)
        L = float(np.nanmax(self.cols.stop_s))
        cand = np.array(self.tree.query_ball_point(x, L), np.int64)
        cand = cand[np.linalg.norm(self.v[cand] - R1, axis=1) > self.contact_min_sep]
        d, r, s_line = self._opposing(cand, x, d1)
        if r < 0 or d > self.miss_max:
            e.miss_mm.append(float(d) * self.h)
            return e  # the second side not found: the edit stays a top edit, flagged by its miss
        k = self.thick.add([r, r, r], [1.0, 0.0, 0.0], s_line * self.h)
        e.obs.append(("thick", k))
        e.roots.append(self.v[r].tolist())
        e.s_mm.append(s_line * self.h)
        e.miss_mm.append(d * self.h)
        return e

    def _opposing(self, cand, x, d1):
        """Among the columns of vertices cand: the one passing nearest x whose direction
        there opposes d1. Returns (distance, vertex or -1, arc length at the nearest point)."""
        if cand.size == 0:
            return np.inf, -1, np.nan
        p = self.cols.pts[cand].astype(np.float64)  # (m, K, 3)
        a, b = p[:, :-1], p[:, 1:]
        ab = b - a
        L2 = np.sum(ab * ab, -1)
        ok = np.isfinite(L2) & (L2 > 0)
        L2s = np.where(ok, L2, 1.0)
        u = np.clip(np.nan_to_num(np.sum((x - a) * ab, -1) / L2s), 0, 1)
        q = a + u[..., None] * ab
        d = np.where(ok, np.linalg.norm(q - x, axis=-1), np.inf)
        tang = ab / np.sqrt(L2s)[..., None]
        d = np.where(np.sum(tang * d1, -1) < 0, d, np.inf)  # opposed directions only
        j = np.argmin(d, 1)
        m = np.arange(len(cand))
        dm = d[m, j]
        i = int(np.argmin(dm))
        if not np.isfinite(dm[i]):
            return np.inf, -1, np.nan
        seg = np.where(ok[i], np.sqrt(L2s[i]), 0.0)
        s0 = np.concatenate([[0.0], np.cumsum(seg)])
        return float(dm[i]), int(cand[i]), float(s0[j[i]] + u[i, j[i]] * seg[j[i]])

    @staticmethod
    def _direction_at(pts, x):
        p = pts[np.isfinite(pts).all(1)]
        i = int(np.argmin(np.linalg.norm(p - x, axis=1)))
        i = min(max(i, 1), len(p) - 1)
        t = p[i] - p[i - 1]
        return t / (np.linalg.norm(t) + 1e-12)

    def add_free(self, x_vox) -> Edit:
        x = self._finite(x_vox)
        return self._free(
            x, Edit("free", self.to_mm(x).tolist(), x.tolist(), time=time.time())
        )

    def _free(self, x, e: Edit, append: bool = True) -> Edit:
        """Observe the free displacement at the point nearest x of the pial surface
        without free displacement (so free edits do not depend on each other's order)."""
        pos = self.positions(free=np.zeros((len(self.v), 3)))
        if self._pial is None:
            self._pial = Locator(pos, self.f)
        else:
            self._pial.move(pos)
        Pc, F, B = self._pial.closest(x[None])
        w = B[0]
        idx = self.f[F[0]]
        target = (x - Pc[0]) * self.h
        for a, fd in enumerate(self.free):
            e.obs.append((f"free{a}", fd.add(idx, w, float(target[a]))))
        e.miss_mm.append(float(np.linalg.norm(x - Pc[0])) * self.h)
        if append:
            self.edits.append(e)
        return e

    def rebuild(self, points) -> list[Edit]:
        """All edits from scratch, from a list of (kind, x_vox): first every top and
        contact edit, then the free edits and the tops that fell back to free, against
        the pial surface the tops give. The result depends only on the list."""
        while self.edits:
            self.remove(len(self.edits) - 1)
        made: list = [None] * len(points)
        for i, (kind, x) in enumerate(points):
            if kind == "top":
                made[i] = self.add_top(x, defer_free=True)
            elif kind == "contact":
                made[i] = self.add_contact(x, defer_free=True)
        for i, (kind, x) in enumerate(points):
            if kind == "free":
                made[i] = self.add_free(x)
            elif made[i].fallback:
                self._free(np.asarray(made[i].vox), made[i], append=False)
        return made

    def remove(self, i: int) -> None:
        e = self.edits.pop(i)
        for name, k in e.obs:
            (self.thick if name == "thick" else self.free[int(name[-1])]).remove(k)

    def set_range(self, range_mm: float) -> None:
        self.prior.set_range(range_mm, self.prior.sd_mm)
        for fd in [self.thick, *self.free]:
            fd.refresh()

    # -- persistence
    def save(self, path) -> None:
        out = [
            {k: val for k, val in asdict(e).items() if k != "obs"} for e in self.edits
        ]
        Path(path).write_text(
            json.dumps(dict(range_mm=self.prior.range_mm, edits=out), indent=1)
        )

    def replay(self, path) -> list[Edit]:
        """Apply the edits of a file to this session, from their world positions."""
        d = json.loads(Path(path).read_text())
        if abs(d.get("range_mm", self.prior.range_mm) - self.prior.range_mm) > 1e-12:
            self.set_range(d["range_mm"])
        return self.rebuild(
            [(e["kind"], self.to_vox(np.asarray(e["mm"]))) for e in d["edits"]]
        )


# ------------------------------------------------------------------ touching: one set of definitions
#
# - Facing: two places on the pial surface face each other when their unit normals
#   point against each other, dot < FACING. A point's normal is its triangle's normal;
#   a vertex's is the area-weighted mean of its triangles'.
# - Distance: from a point to a triangle's surface (its closest point), never vertex to
#   vertex.
# - Touching: a pial vertex within delta of a facing pial triangle, or within delta of
#   the white surface (a different surface: no facing test), on a triangle that does not
#   cross (crossing takes precedence).
# - A contact point's other sheet: the nearest facing pial triangle within the contact
#   radius: the same facing test and distance, searched further. "At a contact" means
#   within that radius of a contact point.

FACING = -0.5


def face_normals(pos: Float, f: Int, sign: Float | None = None) -> Float:
    """Unit triangle normals at positions pos; sign (+-1 per triangle) turns them to the
    base surface's outward side."""
    fn = np.cross(pos[f[:, 1]] - pos[f[:, 0]], pos[f[:, 2]] - pos[f[:, 0]])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-300
    return fn if sign is None else fn * sign[:, None]


def vertex_normals_from(pos: Float, f: Int, fn: Float) -> Float:
    area = 0.5 * np.linalg.norm(
        np.cross(pos[f[:, 1]] - pos[f[:, 0]], pos[f[:, 2]] - pos[f[:, 0]]), axis=1
    )
    vn = np.zeros_like(pos)
    for k in range(3):
        np.add.at(vn, f[:, k], fn * area[:, None])
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-300)


def longest_edge(pos: Float, f: Int) -> float:
    return float(np.linalg.norm(pos[f[:, [1, 2, 0]]] - pos[f], axis=-1).max())


def nearest_facing(
    pos: Float,
    f: Int,
    fn: Float,
    table: Int,
    X: Float,
    NX: Float,
    dist: float,
    tree=None,
    chunk: int = 4000,
    e: float | None = None,
) -> tuple[Float, Int, Float, Float]:
    """nearest_facing_chunk over the points in chunks (bounded memory; the same result).
    tree, e: the KD tree of pos and the longest edge, when already known."""
    from scipy.spatial import cKDTree

    X, NX = np.atleast_2d(X), np.atleast_2d(NX)
    tree = cKDTree(pos) if tree is None else tree
    e = longest_edge(pos, f) if e is None else e
    parts = [
        _nearest_facing_chunk(
            pos, f, fn, table, X[i : i + chunk], NX[i : i + chunk], dist, tree, e
        )
        for i in range(0, len(X), chunk)
    ]
    if not parts:
        return np.zeros(0), np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros((0, 3))
    return tuple(np.concatenate([p[k] for p in parts]) for k in range(4))  # type: ignore[return-value]


def _nearest_facing_chunk(
    pos, f, fn, table, X, NX, dist, tree, e
) -> tuple[Float, Int, Float, Float]:
    """For points X (n, 3) with normals NX: the nearest point of a facing triangle of
    (pos, f) within dist (surface distance). Returns (distance, inf where none; face, -1;
    closest point; barycentric weights). table: the vertex-to-face table of f; e: its
    longest edge (a triangle within dist has a vertex within dist + e)."""
    import mesh_smoothing as M

    n = len(X)
    out_d = np.full(n, np.inf)
    out_f = -np.ones(n, np.int64)
    out_p = np.full((n, 3), np.nan)
    out_b = np.full((n, 3), np.nan)
    from scipy.spatial import cKDTree

    nb = cKDTree(X).sparse_distance_matrix(tree, dist + e, output_type="ndarray")
    qi, vj = nb["i"].astype(np.int64), nb["j"].astype(np.int64)
    if qi.size == 0:
        return out_d, out_f, out_p, out_b
    cand = table[vj]  # (pairs, max faces per vertex)
    qi = np.repeat(qi, cand.shape[1])
    fc = cand.ravel()
    ok = fc >= 0
    qi, fc = qi[ok], fc[ok]
    keep = (
        np.sum(fn[fc] * NX[qi], 1) < FACING
    )  # facing first: far fewer pairs to de-duplicate
    qi, fc = qi[keep], fc[keep]
    key = np.unique(qi * len(f) + fc)
    qi, fc = key // len(f), key % len(f)
    if qi.size == 0:
        return out_d, out_f, out_p, out_b
    tri = f[fc]
    q, bary = M._closest_on_tri(X[qi], pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]])
    d = np.linalg.norm(q - X[qi], axis=1)
    order = np.lexsort((d, qi))
    first = order[np.r_[True, qi[order][1:] != qi[order][:-1]]]
    w = d[first] <= dist
    sel = first[w]
    out_d[qi[sel]], out_f[qi[sel]], out_p[qi[sel]], out_b[qi[sel]] = (
        d[sel],
        fc[sel],
        q[sel],
        bary[sel],
    )
    return out_d, out_f, out_p, out_b


def touching_vertices(
    pos: Float,
    f: Int,
    table: Int,
    sign: Float,
    delta: float,
    white: "Locator | None",
    verts: Int | None = None,
    tree=None,
    e: float | None = None,
) -> npt.NDArray[np.bool_]:
    """Which vertices (all, or `verts` only: the rest False) touch (see above): within
    delta of a facing pial triangle or of the white surface. tree, e: the KD tree of pos
    and the longest edge, when already known."""
    from scipy.spatial import cKDTree

    fn = face_normals(pos, f, sign)
    vn = vertex_normals_from(pos, f, fn)
    tree = cKDTree(pos) if tree is None else tree
    e = longest_edge(pos, f) if e is None else e
    touch = np.zeros(len(pos), bool)
    # only vertices with a facing vertex nearby can touch a facing triangle; test those
    if verts is None:
        pairs = tree.query_pairs(delta + e, output_type="ndarray")
        cand = (
            np.unique(pairs[np.sum(vn[pairs[:, 0]] * vn[pairs[:, 1]], 1) < 0].ravel())
            if len(pairs)
            else np.zeros(0, np.int64)
        )
    else:
        verts = np.asarray(verts, np.int64)
        nb = cKDTree(pos[verts]).sparse_distance_matrix(
            tree, delta + e, output_type="ndarray"
        )
        qi, vj = nb["i"].astype(np.int64), nb["j"].astype(np.int64)
        opp = np.sum(vn[verts[qi]] * vn[vj], 1) < 0
        cand = verts[np.unique(qi[opp])]
    if cand.size:
        d, _, _, _ = nearest_facing(
            pos, f, fn, table, pos[cand], vn[cand], delta, tree, e=e
        )
        touch[cand[np.isfinite(d)]] = True
    if white is not None:
        q = np.arange(len(pos)) if verts is None else verts
        if not hasattr(white, "longest"):
            white.longest = longest_edge(white.v, white.f)
        dw, _ = white.tree.query(pos[q], distance_upper_bound=delta + white.longest)
        cw = q[np.isfinite(dw)]
        if cw.size:
            Pw, _, _ = white.closest(pos[cw])
            touch[cw[np.linalg.norm(Pw - pos[cw], axis=1) <= delta]] = True
    return touch


def touching_faces(
    pos: Float,
    f: Int,
    table: Int,
    sign: Float,
    delta: float,
    white: "Locator | None",
    cross: npt.NDArray[np.bool_],
) -> npt.NDArray[np.bool_]:
    """Triangles of (pos, f) that touch: a vertex touching (touching_vertices), and the
    triangle itself not crossing."""
    return touching_vertices(pos, f, table, sign, delta, white)[f].any(1) & ~cross


# ------------------------------------------------------------------ a session on a contoured surface


class SurfaceSession:
    """Edits on a base pial surface that is a mesh of its own (a contour), not a height
    over the white mesh: every point says "the surface passes here". The displacement is
    a 3D field on the pial mesh (one Field per component, the same prior), observed at
    the point of the base surface nearest the click, so the result depends only on the
    list of points. A contact point also observes the nearest point of a facing sheet
    (vertex normals opposed, dot < -0.5) within contact_radius_mm, so both sheets of a
    carved contact move to the click. v, n: voxels and outward unit normals; affine:
    voxel -> world mm."""

    def __init__(
        self,
        v,
        f,
        n,
        affine=None,
        range_mm=2.0,
        sd_mm=0.5,
        sd_edit_mm=0.03,
        contact_radius_mm=1.0,
    ):
        import pial_height as P

        self.P, self.h = P, P.H_MM
        self.v, self.f, self.n = (
            np.asarray(v, np.float64),
            np.asarray(f, np.int64),
            np.asarray(n, np.float64),
        )
        self.affine = np.eye(4) if affine is None else np.asarray(affine, np.float64)
        self.prior = Prior(self.v * self.h, self.f, range_mm, sd_mm)
        self.free = [
            Field(self.prior, np.zeros(len(self.v)), sd_edit_mm) for _ in range(3)
        ]
        self.loc = Locator(self.v, self.f)
        # each triangle's normal turned to the base surface's outward side, once
        raw = face_normals(self.v, self.f)
        self.sign = np.where(np.sum(raw * self.n[self.f].mean(1), 1) < 0, -1.0, 1.0)
        self.fn = raw * self.sign[:, None]
        self.contact_radius = contact_radius_mm / self.h
        self.edits: list[Edit] = []

    to_mm = Session.to_mm
    to_vox = Session.to_vox
    _finite = staticmethod(Session._finite)

    def free_displacement(self):
        if not any(fd.obs for fd in self.free):
            return None
        return np.stack([fd.correction() for fd in self.free], 1)

    def positions(self, rows=None, free=None) -> Float:
        rows = np.arange(len(self.v)) if rows is None else np.asarray(rows, np.int64)
        d = self.free_displacement() if free is None else free
        p = self.v[rows].copy()
        if d is not None:
            p += d[rows] / self.h
        return p

    def _observe(self, e: Edit, x, idx, w, P0) -> None:
        """The three components share the prior and the row, so one column serves all."""
        target = (x - P0) * self.h
        col = self.prior.cov_times(
            self.free[0].row(
                Obs(np.asarray(idx, np.int64), np.asarray(w, np.float64), 0.0)
            )
        )
        for a, fd in enumerate(self.free):
            e.obs.append((f"free{a}", fd.add(idx, w, float(target[a]), col)))
        e.miss_mm.append(float(np.linalg.norm(x - P0)) * self.h)

    def _facing(self, x, n1):
        """A contact point's other sheet: the nearest facing triangle within the contact
        radius (nearest_facing, the touching definitions' facing test and distance)."""
        d, fi, q, b = nearest_facing(
            self.v,
            self.f,
            self.fn,
            self.loc.table,
            x[None],
            n1[None],
            self.contact_radius,
            self.loc.tree,
        )
        if not np.isfinite(d[0]):
            return None
        return q[0], fi[0], b[0]

    def touching(
        self, pos, delta_vox: float, white: "Locator | None", cross
    ) -> npt.NDArray[np.bool_]:
        return touching_faces(
            pos, self.f, self.loc.table, self.sign, delta_vox, white, cross
        )

    def rebuild(self, points) -> list[Edit]:
        """All edits from scratch, from a list of (kind, x_vox) or, for a drag,
        ("drag", target_vox, from_vox[, pushes]): a base vertex (the one at from_vox)
        moved to the target, and the drag's pushes, (target_vox, from_vox) pairs: base
        vertices of another sheet moved ahead of it. A vertex dragged more than once keeps
        its last drag (with its pushes); the earlier ones are returned with no observation
        (superseded)."""
        for fd in self.free:
            for k in list(fd.obs):
                fd.remove(k)
        self.edits = []
        last_drag = {}
        for i, p in enumerate(points):
            if p[0] == "drag":
                last_drag[self.drag_vertex(p[2])] = i
        for i, p in enumerate(points):
            kind, x = p[0], self._finite(p[1])
            e = Edit(kind, self.to_mm(x).tolist(), x.tolist(), time=time.time())
            if kind == "drag":
                r = self.drag_vertex(p[2])
                e.vertex = r
                if last_drag[r] == i:
                    self._observe(
                        e, x, np.array([r, r, r]), np.array([1.0, 0.0, 0.0]), self.v[r]
                    )
                    e.miss_mm = [
                        float(np.linalg.norm(self.v[r] - np.asarray(p[2]))) * self.h
                    ]
                    if len(p) > 3 and p[3]:
                        self.add_pushes(
                            e,
                            [self._finite(t) for t, _ in p[3]],
                            [self.drag_vertex(fr) for _, fr in p[3]],
                        )
            else:
                P0, F, B = self.loc.closest(x[None])
                self._observe(e, x, self.f[F[0]], B[0], P0[0])
                if kind == "contact":
                    hit = self._facing(x, self.fn[F[0]])
                    if hit is not None:
                        q, fi, b = hit
                        self._observe(e, x, self.f[fi], b, q)
            e.roots = [[] for _ in e.miss_mm]  # one entry per observed sheet
            self.edits.append(e)
        return self.edits

    def set_drag_target(self, e: Edit, x_vox) -> None:
        """Move a drag edit's target without new solves: only its observed values change."""
        names = dict(e.obs)
        x = self._finite(x_vox)
        k0 = names["free0"]
        idx = self.free[0].obs[k0].idx[0]
        for a in range(3):
            self.free[a].obs[names[f"free{a}"]].y = float(
                (x[a] - self.v[idx][a]) * self.h
            )
        e.vox = x.tolist()
        e.mm = self.to_mm(x).tolist()

    def add_drag(self, target_vox, from_vox) -> Edit:
        """One drag added to the current edits without a rebuild: the same as appending it
        to the list and rebuilding (an earlier drag of the same vertex loses its
        observation; the last drag wins)."""
        x = self._finite(target_vox)
        r = self.drag_vertex(from_vox)
        for old in self.edits:
            if old.kind == "drag" and old.vertex == r and old.obs:
                for name, k in old.obs + [o for _, obs, _ in old.pushes for o in obs]:
                    self.free[int(name[-1])].remove(k)
                old.obs, old.miss_mm, old.roots, old.pushes = [], [], [], []
        e = Edit("drag", self.to_mm(x).tolist(), x.tolist(), time=time.time(), vertex=r)
        self._observe(e, x, np.array([r, r, r]), np.array([1.0, 0.0, 0.0]), self.v[r])
        e.miss_mm = [float(np.linalg.norm(self.v[r] - np.asarray(from_vox))) * self.h]
        e.roots = [[]]
        self.edits.append(e)
        return e

    def add_pushes(self, e: Edit, targets_vox, verts) -> None:
        """Pushes of drag e: base vertices verts moved to targets_vox, with one solve for
        all their columns (each observes one vertex, as a drag does)."""
        verts = np.asarray(verts, np.int64)
        if verts.size == 0:
            return
        A = np.zeros((self.prior.n, verts.size))
        A[verts, np.arange(verts.size)] = 1.0
        cols = self.prior.cov_times(A)
        for j, (q, x) in enumerate(zip(verts, targets_vox, strict=True)):
            d = (np.asarray(x) - self.v[q]) * self.h
            obs = [
                (
                    f"free{a}",
                    fd.add(
                        np.array([q, q, q]),
                        np.array([1.0, 0.0, 0.0]),
                        float(d[a]),
                        cols[:, j],
                    ),
                )
                for a, fd in enumerate(self.free)
            ]
            e.pushes.append([int(q), obs, np.asarray(x, np.float64).copy()])

    def set_push_target(self, e: Edit, j: int, x_vox) -> None:
        """Move push j of drag e without new solves: only its observed values change."""
        q, obs, _ = e.pushes[j]
        x = self._finite(x_vox)
        for a, (name, k) in enumerate(obs):
            self.free[a].obs[k].y = float((x[a] - self.v[q][a]) * self.h)
        e.pushes[j][2] = x.copy()

    def drag_vertex(self, from_vox) -> int:
        """The base vertex a drag starts from (its base position is saved with it)."""
        return int(self.loc.tree.query(self._finite(from_vox))[1])

    def set_range(self, range_mm: float) -> None:
        self.prior.set_range(range_mm, self.prior.sd_mm)
        for fd in self.free:
            fd.refresh()
