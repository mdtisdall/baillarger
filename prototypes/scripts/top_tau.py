"""τ behind a fixed top surface Σ (T3 of docs/plans/top-surface.md), and the
step direction ∇τ for the Σ start (user, 2026-11-19: "Step along ∇τ").

Between two start steps, the τ solve of global_tau_solver runs on the voxel
centres between the white surface and Σ, with τ = 0 where an axis crosses Σ
(a ghost value at θh, 0 < θ ≤ 1, from the voxel centre), a warm start from
the last τ and at most LM_ITERS Levenberg-Marquardt iterations. The energy
is that of global_tau_energy without the stop term (Σ is fixed), with the
thickness prior kept (user, 2026-11-19). The direction of a vertex of Σ is
the mean of ∇τ over the τ voxels inside Σ within one voxel of it, weighted
by 1 - distance (voxels); where none is near, the caller uses the mesh
normal (user, 2026-11-19).

Inside Σ: for each axis, the grid lines through the voxel centres are cut
with the faces of Σ. Along a line, a voxel centre is inside when the next
crossing above it leaves Σ (the face normal has a positive component on the
axis) or, with no crossing above, the last crossing below enters it. Each
axis with a crossing on the voxel's line votes; inside needs a majority of
the votes. The domain is the voxels inside Σ that are outside the white
surface (site.inside) or are inner voxels of global_tau_solver.make_domain;
every domain voxel gets the ghost where an axis crosses Σ, and a connected
part with no ghost is left out (its τ has no τ = 0 to fix it).
"""

from __future__ import annotations

import time
from typing import Any

import global_tau_energy as GE
import global_tau_prep as TP
import global_tau_solver as GS
import jax.numpy as jnp
import numba as nb
import numpy as np
from scipy import ndimage as ndi

LM_ITERS = 3  # user, 2026-11-19: "A few LM iterations"
THETA_MIN = 0.05  # a voxel centre on Σ: the ghost at 0.05 h, not at 0
EPS_B, EPS_C = 1.37e-7, 2.91e-7  # voxels: the offset of the grid lines (no exact edge hits)


@nb.njit(cache=True)
def _axis_crossings(
    P: np.ndarray, F: np.ndarray, a: int, shape: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The crossings of the grid lines along axis a (through the voxel
    centres, integer coordinates on the other two axes) with the faces of
    the mesh (P in voxel coordinates): line coordinates (u, v), the position
    t on the axis, the sign of the face normal's component on the axis, and
    the face."""
    b, c = (a + 1) % 3, (a + 2) % 3
    cap = 16 * len(F) + 16
    U = np.empty(cap, np.int64)
    W = np.empty(cap, np.int64)
    T = np.empty(cap, np.float64)
    S = np.empty(cap, np.int64)
    Fx = np.empty(cap, np.int64)
    n = 0
    for f in range(len(F)):
        p0, p1, p2 = P[F[f, 0]], P[F[f, 1]], P[F[f, 2]]
        nx = (p1[1] - p0[1]) * (p2[2] - p0[2]) - (p1[2] - p0[2]) * (p2[1] - p0[1])
        ny = (p1[2] - p0[2]) * (p2[0] - p0[0]) - (p1[0] - p0[0]) * (p2[2] - p0[2])
        nz = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        nrm = (nx, ny, nz)
        na = nrm[a]
        if na == 0.0:
            continue
        lo_b = int(np.ceil(min(p0[b], p1[b], p2[b]) - EPS_B))
        hi_b = int(np.floor(max(p0[b], p1[b], p2[b]) - EPS_B))
        lo_c = int(np.ceil(min(p0[c], p1[c], p2[c]) - EPS_C))
        hi_c = int(np.floor(max(p0[c], p1[c], p2[c]) - EPS_C))
        for u in range(max(lo_b, 0), min(hi_b, shape[b] - 1) + 1):
            for w in range(max(lo_c, 0), min(hi_c, shape[c] - 1) + 1):
                # the line is moved by a tiny offset (EPS_B, EPS_C), so it
                # never passes exactly through an edge or a vertex of Σ and
                # each crossing counts once
                ub, wc = u + EPS_B, w + EPS_C
                x0, y0 = p0[b], p0[c]
                x1, y1 = p1[b], p1[c]
                x2, y2 = p2[b], p2[c]
                den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
                if den == 0.0:
                    continue
                l0 = ((y1 - y2) * (ub - x2) + (x2 - x1) * (wc - y2)) / den
                l1 = ((y2 - y0) * (ub - x2) + (x0 - x2) * (wc - y2)) / den
                l2 = 1.0 - l0 - l1
                if l0 < 0 or l1 < 0 or l2 < 0:
                    continue
                if n == cap:
                    raise ValueError("_axis_crossings: buffer full")
                U[n] = u
                W[n] = w
                T[n] = l0 * p0[a] + l1 * p1[a] + l2 * p2[a]
                S[n] = 1 if na > 0 else -1
                Fx[n] = f
                n += 1
    return U[:n], W[:n], T[:n], S[:n], Fx[:n]


@nb.njit(cache=True)
def _line_votes(
    U: np.ndarray, W: np.ndarray, T: np.ndarray, S: np.ndarray, Fx: np.ndarray,
    a: int, shape: np.ndarray, vote: np.ndarray, voted: np.ndarray,
    ghost: np.ndarray, gdist: np.ndarray, gface: np.ndarray,
) -> None:
    """Per voxel on each line with crossings: an inside vote (next crossing
    above leaves, or none above and the last below enters), and the ghost
    sides: a crossing between the centre and the neighbour's centre, with
    θ = its distance from the centre (the nearest crossing on that side)
    and its face (gface)."""
    b, c = (a + 1) % 3, (a + 2) % 3
    # sort by line, then by t
    key = (U * shape[c] + W).astype(np.float64) * (4.0 * shape[a] + 8.0) + (T + 2.0)
    order = np.argsort(key, kind="mergesort")
    i = 0
    N = len(order)
    idx = np.zeros(3, np.int64)
    while i < N:
        j = i
        line = U[order[i]] * shape[c] + W[order[i]]
        while j < N and U[order[j]] * shape[c] + W[order[j]] == line:
            j += 1
        u, w = U[order[i]], W[order[i]]
        ts = T[order[i:j]]
        ss = S[order[i:j]]
        fs = Fx[order[i:j]]
        m = j - i
        for k in range(shape[a]):
            # next crossing above k
            q = 0
            while q < m and ts[q] <= k:
                q += 1
            inside = False
            if q < m:
                inside = ss[q] > 0
            else:
                inside = ss[m - 1] < 0
            idx[a] = k
            idx[b] = u
            idx[c] = w
            vote[idx[0], idx[1], idx[2]] += 1 if inside else -1
            voted[idx[0], idx[1], idx[2]] += 1
            # ghost sides: a crossing in (k, k + 1] (plus side), [k - 1, k) (minus side)
            if q < m and ts[q] <= k + 1:
                ghost[idx[0], idx[1], idx[2], a, 1] = True
                gdist[idx[0], idx[1], idx[2], a, 1] = ts[q] - k
                gface[idx[0], idx[1], idx[2], a, 1] = fs[q]
            if q > 0 and ts[q - 1] >= k - 1:
                ghost[idx[0], idx[1], idx[2], a, 0] = True
                gdist[idx[0], idx[1], idx[2], a, 0] = k - ts[q - 1]
                gface[idx[0], idx[1], idx[2], a, 0] = fs[q - 1]
        i = j


def sigma_top(
    site: TP.Site, V_mm: np.ndarray, F: np.ndarray, tau_prev: np.ndarray | None, s: GS.Settings
) -> tuple[GS.SigmaTop, dict[str, int]]:
    """The fixed top surface Σ on the site's grid (see the module docstring):
    domain, inner voxels, ghosts (θ ≥ THETA_MIN) and τ0 (the last τ where it
    exists, else -(distance to the voxels beyond Σ - h/2)), and counts.

    Every domain voxel is inside Σ: the tissue voxels, and the inner voxels
    inside Σ (while Σ still lies in the WM voxels, at the start, it is
    between inner voxels). Every domain voxel with a neighbour across Σ has
    the ghost τ = 0 on that side (T3). A connected part of the domain (6
    neighbours, the stencil) with no ghost has no τ = 0 to fix its offset:
    it is left out, and counted ("unanchored_voxels")."""
    shape = np.array(site.labels.shape, np.int64)
    P = V_mm / site.vox
    Fi = np.ascontiguousarray(F, np.int64)
    vote = np.zeros(tuple(shape), np.int64)
    voted = np.zeros(tuple(shape), np.int64)
    ghost = np.zeros((*shape, 3, 2), bool)
    gdist = np.ones((*shape, 3, 2))
    gface = np.full((*shape, 3, 2), -1, np.int64)
    for a in range(3):
        U, W, T, S, Fx = _axis_crossings(np.ascontiguousarray(P), Fi, a, shape)
        _line_votes(U, W, T, S, Fx, a, shape, vote, voted, ghost, gdist, gface)
    inside_sigma = (voted > 0) & (vote > 0)
    # every voxel's crossings within one voxel (before the domain mask): the
    # stop term's band and the derivatives with respect to Σ (top_solve)
    geo = {"inside": inside_sigma, "cross": ghost.copy(), "theta": gdist.copy(), "face": gface}
    _, inner = GS.make_domain(site, s)
    tissue = inside_sigma & ~site.inside
    inner &= inside_sigma
    domain = tissue | inner
    ghost &= domain[..., None, None]
    comp, n = ndi.label(domain)
    anchored = np.zeros(n + 1, bool)
    anchored[np.unique(comp[ghost.any((3, 4))])] = True
    anchored[0] = False
    free = domain & ~anchored[comp]
    domain &= ~free
    inner &= domain
    ghost &= domain[..., None, None]
    gdist = np.clip(gdist, THETA_MIN, 1.0)
    h = float(site.vox.mean())
    d = ndi.distance_transform_edt(inside_sigma)
    tau0 = -(d - 0.5) * h
    if tau_prev is not None:
        ok = domain & np.isfinite(tau_prev)
        tau0[ok] = tau_prev[ok]
    info = {
        "tissue_voxels": int((domain & ~inner).sum()),
        "inner_voxels": int(inner.sum()),
        "ghost_sides": int(ghost.sum()),
        "unanchored_parts": int(n - anchored[1:].sum()),
        "unanchored_voxels": int(free.sum()),
    }
    sig = GS.SigmaTop(domain, inner, ghost, gdist, tau0)
    sig.geo = geo
    return sig, info


class TauBehind:
    """τ behind Σ for one site, solved again before each start step."""

    def __init__(self, site: TP.Site, lm_iters: int = LM_ITERS):
        self.site = site
        self.s = GS.Settings(lm_iters=lm_iters, face_test=False)
        t0 = time.time()
        self.pt = TP.make_patches(site, self.s.patch_mm)
        self.centre_vox = np.array(site.labels.shape) / 2.0
        self.tau_grid: np.ndarray | None = None
        self.log: list[dict[str, Any]] = []
        self.setup_s = time.time() - t0

    def solve(self, V_mm: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray, GS.GlobalTau | None]:
        """τ behind Σ = (V_mm, F): returns τ on the grid (nan off the
        domain), ∇τ on the grid ((X, Y, Z, 3), nan off the domain: every
        domain voxel is inside Σ) and the solver (None when the domain is
        empty)."""
        t0 = time.time()
        sig, ginfo = sigma_top(self.site, V_mm, F, self.tau_grid, self.s)
        shape = self.site.labels.shape
        grad = np.full((*shape, 3), np.nan)
        if not sig.domain.any():
            self.log.append({**ginfo, "s": time.time() - t0})
            return np.full(shape, np.nan), grad, None
        gt = GS.GlobalTau(self.site, self.pt, self.s, self.centre_vox, sigma=sig)
        info, _ = gt.solve_global("sigma")
        pb = gt._pb
        x = np.zeros(pb.S1)
        x[: pb.S] = gt.tau
        g = np.asarray(GE.gradient(jnp.asarray(x), pb.P))[: pb.S]
        tau = gt.G.grid(gt.tau)
        X = tuple(gt.G.coords.T - GS.PAD)
        grad[X] = g
        grad[~sig.domain] = np.nan
        self.tau_grid = tau
        self.log.append(
            {
                **ginfo,
                "tau_min": float(np.nanmin(tau)),
                "tau_max": float(np.nanmax(tau)),
                "lm_iters": len(info["iters"]),
                "e0": info["e0"],
                "e1": info["e1"],
                "s": time.time() - t0,
            }
        )
        return tau, grad, gt

    def directions(self, V_mm: np.ndarray, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per vertex: the unit mean of ∇τ over the τ voxels inside Σ within
        one voxel, weighted by 1 - distance (voxels), and whether it has one."""
        p = V_mm / self.site.vox
        shape = np.array(grad.shape[:3])
        lo = np.floor(p).astype(np.int64)
        acc = np.zeros((len(p), 3))
        wsum = np.zeros(len(p))
        for di in (-1, 0, 1, 2):
            for dj in (-1, 0, 1, 2):
                for dk in (-1, 0, 1, 2):
                    q = lo + np.array([di, dj, dk])
                    ok = np.all((q >= 0) & (q < shape), 1)
                    d = np.linalg.norm(q - p, axis=1)
                    w = np.where(ok, np.maximum(1.0 - d, 0.0), 0.0)
                    qc = np.clip(q, 0, shape - 1)
                    g = grad[qc[:, 0], qc[:, 1], qc[:, 2]]
                    good = (w > 0) & np.isfinite(g).all(1)
                    acc[good] += w[good, None] * g[good]
                    wsum[good] += w[good]
        n = np.linalg.norm(acc, axis=1)
        has = (wsum > 0) & (n > 0)
        D = np.zeros_like(acc)
        D[has] = acc[has] / n[has, None]
        return D, has
