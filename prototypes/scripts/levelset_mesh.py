"""The white mesh for levelset_fill on real data: the start band (signed distance, foot point, normal),
roots as mesh vertices, and the thickness field on the mesh (test plan §2, §4).

The field: log t = m + (a0 + w(x)) H(x) + z(x) + noise, as collide.Field, but on the mesh's vertices
with a finite-element (cotangent) stiffness C and a lumped mass M, in mm, so that the lengths are the
same 2 mm (z) and 6 mm (w) whatever the vertex spacing:
- second order (alpha = 2, as collide's (kappa^2 I + L)^T (kappa^2 I + L)): Q = tau K M^-1 K;
- first order (alpha = 1): Q = tau K;
with K = kappa^2 M + C, kappa = 1 / length. tau is set, as collide.prior_op, so that the mean marginal
variance at 40 probe vertices is SD^2. The posterior mean comes from a sparse solve; predictive SDs at
chosen vertices from perturb-and-MAP samples.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numba import njit, prange
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]

# collide.py's constants
SIG_Z, ELL_Z, SIG_W, ELL_W = 0.2, 2.0, 0.3, 6.0
A0_MU, A0_SD, M_SD, SIG_EPS = 0.5, 0.5, 1.0, 0.08
M_MU = float(np.log(2.5))


# ------------------------------------------------------------------ closest point


@njit(cache=True)
def _closest_on_triangle(p, a, b, c):
    """Closest point to p on triangle abc (Ericson, Real-Time Collision Detection 5.1.5) and its
    barycentric coordinates."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = ab @ ap
    d2 = ac @ ap
    if d1 <= 0 and d2 <= 0:
        return a, 1.0, 0.0, 0.0
    bp = p - b
    d3 = ab @ bp
    d4 = ac @ bp
    if d3 >= 0 and d4 <= d3:
        return b, 0.0, 1.0, 0.0
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        return a + v * ab, 1 - v, v, 0.0
    cp = p - c
    d5 = ab @ cp
    d6 = ac @ cp
    if d6 >= 0 and d5 <= d6:
        return c, 0.0, 0.0, 1.0
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        return a + w * ac, 1 - w, 0.0, w
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b), 0.0, 1 - w, w
    den = 1.0 / (va + vb + vc)
    v = vb * den
    w = vc * den
    return a + ab * v + ac * w, 1 - v - w, v, w


@njit(cache=True, parallel=True)
def _closest(pts, cand, verts, faces):
    n = pts.shape[0]
    foot = np.zeros((n, 3))
    bary = np.zeros((n, 3))
    tri = np.zeros(n, np.int64)
    dist = np.full(n, np.inf)
    for i in prange(n):
        p = pts[i]
        for j in range(cand.shape[1]):
            t = cand[i, j]
            if t < 0:
                continue
            a = verts[faces[t, 0]]
            b = verts[faces[t, 1]]
            c = verts[faces[t, 2]]
            q, u, v, w = _closest_on_triangle(p, a, b, c)
            dd = np.sqrt(np.sum((p - q) ** 2))
            if dd < dist[i]:
                dist[i] = dd
                foot[i] = q
                bary[i, 0] = u
                bary[i, 1] = v
                bary[i, 2] = w
                tri[i] = t
    return dist, foot, bary, tri


@dataclass
class Mesh:
    verts: Float  # voxel coordinates
    faces: Int
    normals: Float  # unit, out of the WM
    vox: Float  # mm per voxel along each axis
    H: Float | None = None  # mean curvature per vertex (per mm)

    @property
    def mm(self) -> Float:
        return self.verts * self.vox


def closest(mesh: Mesh, pts: Float, k: int = 24) -> tuple[Float, Float, Float, Int, Float]:
    """Signed distance (voxels, > 0 on the side the normals point to), foot point, outward normal at the
    foot (barycentric), triangle and barycentric coordinates, for points (n, 3) in voxel coordinates.
    Candidates: the k triangles with the nearest centroids (in voxels)."""
    cen = mesh.verts[mesh.faces].mean(1)
    _, cand = cKDTree(cen).query(pts, k=k)
    dist, foot, bary, tri = _closest(np.ascontiguousarray(pts, float), cand.astype(np.int64),
                                     mesh.verts, mesh.faces)
    nrm = np.einsum("ij,ijk->ik", bary, mesh.normals[mesh.faces[tri]])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    sgn = np.sign(np.sum((pts - foot) * nrm, 1))
    sgn[sgn == 0] = 1
    return sgn * dist, foot, nrm, tri, bary


def vertex_of(mesh: Mesh, tri: Int, bary: Float) -> Int:
    """The vertex of each triangle with the largest barycentric weight."""
    return mesh.faces[tri, np.argmax(bary, 1)]


# ------------------------------------------------------------------ FEM matrices and graph


def fem(mesh: Mesh) -> tuple[sparse.csc_matrix, Float]:
    """Cotangent stiffness C (positive semi-definite) and lumped mass (vertex areas), in mm."""
    v = mesh.mm
    f = mesh.faces
    nv = len(v)
    rows, cols, vals = [], [], []
    area = np.zeros(nv)
    for k in range(3):
        i, j, o = f[:, k], f[:, (k + 1) % 3], f[:, (k + 2) % 3]
        a, b = v[i] - v[o], v[j] - v[o]
        cr = np.maximum(np.linalg.norm(np.cross(a, b), axis=1), 1e-12)
        cot = np.einsum("ij,ij->i", a, b) / cr
        rows += [i, j, i, j]
        cols += [j, i, i, j]
        vals += [-0.5 * cot, -0.5 * cot, 0.5 * cot, 0.5 * cot]
        np.add.at(area, f[:, k], 0.5 * cr / 3)
    C = sparse.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nv, nv))
    return C, np.maximum(area, 1e-12)


def edge_graph(mesh: Mesh) -> sparse.csr_matrix:
    """Mesh edges with their lengths in mm."""
    f = mesh.faces
    E = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1), axis=0)
    ln = np.linalg.norm(mesh.mm[E[:, 0]] - mesh.mm[E[:, 1]], axis=1)
    G = sparse.coo_matrix((ln, (E[:, 0], E[:, 1])), shape=(len(mesh.verts),) * 2)
    return (G + G.T).tocsr()


def geodesic(G: sparse.csr_matrix, src: Int, limit: float = 60.0) -> Float:
    return dijkstra(G, indices=src, limit=limit)


# ------------------------------------------------------------------ the field


class MeshField:
    def __init__(self, mesh: Mesh, H: Float, alpha: int = 2, seed: int = 0):
        self.n = len(mesh.verts)
        self.H = H
        self.alpha = alpha
        self.C, self.area = fem(mesh)
        self.M = sparse.diags(self.area).tocsc()
        self.Minv = sparse.diags(1.0 / self.area).tocsc()
        self.pz = self._prior(ELL_Z, SIG_Z, seed)
        self.pw = self._prior(ELL_W, SIG_W, seed + 1)

    def _prior(self, ell, sig, seed):
        K = ((1 / ell) ** 2 * self.M + self.C).tocsc()
        lu = splu(K)
        rng = np.random.default_rng(seed)
        probe = rng.choice(self.n, size=min(40, self.n), replace=False)
        E = np.zeros((self.n, len(probe)))
        E[probe, np.arange(len(probe))] = 1
        X = lu.solve(E)
        if self.alpha == 2:  # Q^-1 = K^-1 M K^-1 / tau: var_i = |M^1/2 K^-1 e_i|^2 / tau
            v = (self.area[:, None] * X**2).sum(0).mean()
            tau = v / sig**2
            Q = tau * (K @ self.Minv @ K)
        else:  # Q^-1 = K^-1 / tau
            v = X[probe, np.arange(len(probe))].mean()
            tau = v / sig**2
            Q = tau * K
        return dict(K=K, lu=lu, tau=tau, Q=Q.tocsc())

    def _sample_prior(self, p, rng):
        """A draw with precision p['Q']."""
        xi = rng.standard_normal(self.n)
        if self.alpha == 2:  # x = K^-1 M^1/2 xi / sqrt(tau)
            return p["lu"].solve(np.sqrt(self.area) * xi) / np.sqrt(p["tau"])
        raise NotImplementedError("predictive SDs are only implemented for the second-order field")

    def _prior_inv(self, p, r):
        """The prior covariance times r: K^-1 M K^-1 r / tau (alpha 2) or K^-1 r / tau (alpha 1)."""
        if self.alpha == 2:
            return p["lu"].solve(self.area * p["lu"].solve(r)) / p["tau"]
        return p["lu"].solve(r) / p["tau"]

    def fit(self, y: Float, trend: bool = True, m_mu: float = M_MU, nsamp: int = 0, at: Int | None = None,
            seed: int = 1, direct: bool = False, rtol: float = 1e-8) -> dict:
        """y: log budget per vertex (nan: no evidence). Returns the posterior mean of log t per vertex,
        (m, a0) and, with nsamp > 0, the predictive SD (posterior plus noise) at the vertices `at`.
        The posterior precision Q = P + A^T A / eps^2 is solved by conjugate gradients preconditioned
        with the prior covariance P^-1 (two solves with K's LU); P^-1 Q = I + P^-1 A^T A / eps^2 is well
        conditioned. A direct sparse LU of Q (direct=True) fills in beyond memory at 3e5 vertices
        (13.6 GB and rising at M1S1); test_mesh_field_cg_matches_direct checks the two agree."""
        from scipy.sparse.linalg import LinearOperator, cg

        n, H = self.n, self.H
        o = np.flatnonzero(np.isfinite(y))
        nu = 2 * n + 2 if trend else n + 1
        im = nu - 2 if trend else n
        ia = nu - 1
        sc_var = np.array([M_SD**2] + ([A0_SD**2] if trend else []))
        pm = np.zeros(nu)
        pm[im] = m_mu
        if trend:
            pm[ia] = A0_MU
        rows = np.r_[np.arange(len(o)), np.arange(len(o))]
        cols = np.r_[o, np.full(len(o), im)]
        vals = np.ones(2 * len(o))
        if trend:
            rows = np.r_[rows, np.arange(len(o)), np.arange(len(o))]
            cols = np.r_[cols, n + o, np.full(len(o), ia)]
            vals = np.r_[vals, H[o], H[o]]
        A = sparse.csr_matrix((vals, (rows, cols)), shape=(len(o), nu))
        AtA = (A.T @ A).tocsr() / SIG_EPS**2
        pz, pw = self.pz, self.pw
        sc0 = n * (2 if trend else 1)

        def Pmul(x):
            out = np.empty_like(x)
            out[:n] = pz["Q"] @ x[:n]
            if trend:
                out[n:2 * n] = pw["Q"] @ x[n:2 * n]
            out[sc0:] = x[sc0:] / sc_var
            return out

        def Pinv(r):
            out = np.empty_like(r)
            out[:n] = self._prior_inv(pz, r[:n])
            if trend:
                out[n:2 * n] = self._prior_inv(pw, r[n:2 * n])
            out[sc0:] = r[sc0:] * sc_var
            return out

        Qop = LinearOperator((nu, nu), matvec=lambda x: Pmul(x) + AtA @ x)
        Mop = LinearOperator((nu, nu), matvec=Pinv)
        if direct:
            blocks = [pz["Q"]] + ([pw["Q"]] if trend else [])
            P = sparse.block_diag(blocks + [sparse.diags(1 / sc_var)]).tocsc()
            lu = splu((P + AtA).tocsc())
            solve = lu.solve
        else:
            def solve(b, x0=None):
                x, info = cg(Qop, b, x0=x0, rtol=rtol, maxiter=2000, M=Mop)
                if info != 0:
                    raise RuntimeError(f"cg did not converge ({info})")
                return x
        xhat = solve(Pmul(pm) + A.T @ y[o] / SIG_EPS**2)

        def pred(x):
            out = x[:n] + x[im]
            if trend:
                out = out + H * (x[n:2 * n] + x[ia])
            return out

        res = dict(mu=pred(xhat), m=float(xhat[im]), a0=float(xhat[ia]) if trend else 0.0,
                   a=(xhat[n:2 * n] + xhat[ia]) if trend else np.zeros(n))
        if nsamp and at is not None:
            rng = np.random.default_rng(seed)
            S, A0s = [], []
            for _ in range(nsamp):
                x0 = pm.copy()
                x0[:n] += self._sample_prior(pz, rng)
                if trend:
                    x0[n:2 * n] += self._sample_prior(pw, rng)
                    x0[ia] += A0_SD * rng.standard_normal()
                x0[im] += M_SD * rng.standard_normal()
                yp = y[o] + SIG_EPS * rng.standard_normal(len(o))
                xs = x0 + solve(A.T @ (yp - A @ x0) / SIG_EPS**2)
                S.append(pred(xs)[at])
                A0s.append(xs[ia] if trend else 0.0)
            S = np.array(S)
            res["sd_at"] = np.sqrt(S.std(0) ** 2 + SIG_EPS**2)
            res["a0_sd"] = float(np.std(A0s))
        return res


def pieces_without_evidence(mesh: Mesh, has: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """Per vertex: True where the vertex's connected piece of the mesh holds no evidence."""
    G = edge_graph(mesh)
    npc, lab = connected_components(G, directed=False)
    ev = np.bincount(lab[has], minlength=npc) > 0
    return ~ev[lab]
