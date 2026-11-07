"""A curved fundus phantom whose true columns bend (plan: docs/plans/bent-columns-
trial.md, phase 4). All 3D, voxels of 0.16 mm.

Geometry. The geometry is defined in mm in a local frame (x, y, z), as in
fold_phantoms.py. The phantom does not change along z. The slot's core K is
the half-plane {x = 0, y >= 0}, as for the "fundus" of fold_phantoms.py. For a
point Y, its foot on K is (0, max(y, 0), z). Let q = (x, min(y, 0)), r = |q| (the
distance from K) and theta = atan2(x, -min(y, 0)), in [-pi/2, pi/2]:

- On the banks (y >= 0): r = |x|, theta = +pi/2 (bank A, x > 0) or -pi/2
  (bank B, x < 0).
- Round the slot's edge (y < 0): polar coordinates about the edge, theta = 0
  straight below it.

The cortex thickness is t(theta) = tB + (tA - tB) S(u), with
u = theta / width + 1/2 clipped to [0, 1] and S(u) = 6u^5 - 15u^4 + 10u^3. S has
zero first and second derivatives at u = 0 and u = 1. So t is constant on each
bank and changes smoothly round the fundus, from tB (bank B) to tA (bank A).
The default is width = 180 degrees, so t changes over the whole half turn.

Depth function. phi = 1 - (r - g) / t(theta). This is the relative depth:

- phi = 0 is the white surface r = g + t(theta).
- phi = 1 is the pia r = g: the edge of a slot of outside medium of half-width
  g above the fundus, with a round end of radius g.
- phi < 0 is WM (label 2), 0 <= phi <= 1 is cortex (label 0), phi > 1 is the
  outside medium (label 1). These are the labels of fold_phantoms.py.

On a flat bank, phi is linear in the distance across the bank. There it is the
equidistant depth, the equal-volume depth and the Laplace solution of the
slab, all at once. With tA = tB, phi is the relative depth of the
fold_phantoms "fundus". Round the fundus, phi is a closed form, not a harmonic
function. The gradient of phi is never zero (its r component is -1/t), so
each orthogonal trajectory goes from phi = 0 to phi = 1. phi is smooth except
at y = 0, where it is continuously differentiable and its second derivatives
jump (as the distance to K does).

True columns. The true column through a root is the orthogonal trajectory of
phi through that root: the curve that follows grad(phi) / |grad(phi)|, traced
with RK4 at a step of STEP = 0.01 mm. It starts where the trajectory through
the root meets phi = 0 and ends where it meets phi = 1. The last step is
shorter, so that the end lies on phi = 1.

The truth is a choice of definition: the orthogonal trajectories of this
depth function. It is not a fact of tissue. Another depth function with the
same white and pial surfaces gives other true columns.

Image and labels. make_curved calls fold_phantoms.make with this phantom: the
same partial volume over 3^3 sub-samples of the profile against relative depth
in the cortex, the WM and fluid levels outside, labels at voxel centres, noise
on the log intensity, and features made as there. Coordinates are voxel index
space, with the oblique rotation of fold_phantoms.py.

Errors. fold_eval.errors needs straight true columns (a foot, a direction and
two distances). These columns bend, so path_errors is the adapter. It gives
the same four arrays as fold_eval.errors, with the distance from the true
path in place of the distance from the true line.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fold_phantoms as FP  # noqa: E402

H = FP.H  # voxel (mm)
STEP = 0.01  # step of the true columns (mm)
SMAX = 12.0  # the longest trace (mm)


@dataclass
class Curved(FP.Fold):
    """A Fold with kind "bent". p holds tA, tB, g (mm) and width (radians)."""

    # ---- the polar frame about the slot's edge
    def _polar(self, Y):
        Y = np.asarray(Y, float)
        x, yq = Y[:, 0], np.minimum(Y[:, 1], 0.0)
        r = np.hypot(x, yq)
        th = np.arctan2(x, -yq)
        return x, yq, r, th, Y[:, 1] < 0

    def _u(self, th):
        return np.clip(th / self.p["width"] + 0.5, 0.0, 1.0)

    def thickness(self, th):
        """Cortex thickness t(theta) (mm)."""
        u = self._u(np.asarray(th, float))
        S = u**3 * (10 - 15 * u + 6 * u**2)
        return self.p["tB"] + (self.p["tA"] - self.p["tB"]) * S

    def dthickness(self, th):
        """dt / dtheta (mm per radian)."""
        u = self._u(np.asarray(th, float))
        dS = 30 * u**2 * (1 - u) ** 2
        return (self.p["tA"] - self.p["tB"]) * dS / self.p["width"]

    # ---- the depth function
    def phi(self, Y):
        """Relative depth (0 at the WM, 1 at the pia), at local mm points (N, 3)."""
        _, _, r, th, _ = self._polar(Y)
        return 1.0 - (r - self.p["g"]) / self.thickness(th)

    def grad_phi(self, Y):
        """grad(phi) (1/mm), at local mm points (N, 3)."""
        x, yq, r, th, low = self._polar(Y)
        rs = np.maximum(r, 1e-12)
        t, dt = self.thickness(th), self.dthickness(th)
        dr = np.stack([x / rs, yq / rs], 1)
        dth = np.stack([-yq / rs**2, np.where(low, x / rs**2, 0.0)], 1)
        G = np.zeros((len(x), 3))
        G[:, :2] = -dr / t[:, None] + ((r - self.p["g"]) * dt / t**2)[:, None] * dth
        return G

    # ---- the interface of fold_phantoms.render and make
    def foot(self, Y):
        c = np.zeros_like(np.asarray(Y, float))
        c[:, 1] = np.maximum(Y[:, 1], 0.0)
        c[:, 2] = Y[:, 2]
        return c

    def geom(self, Y):
        """label (0 cortex, 1 fluid, 2 WM), relative depth phi (nan outside the
        cortex), bank (0 A, 1 B), distance r from the core, foot c, unit u."""
        Y = np.asarray(Y, float)
        p = self.phi(Y)
        lab = np.where(p < 0, 2, np.where(p > 1, 1, 0)).astype(np.int8)
        rel = np.where(lab == 0, p, np.nan)
        _, _, r, th, _ = self._polar(Y)
        bank = (th < 0).astype(np.int8)
        c = self.foot(Y)
        u = (Y - c) / np.maximum(r, 1e-12)[:, None]
        return lab, rel, bank, r, c, u

    def column(self, X):
        raise NotImplementedError("true columns bend: use true_columns and path_errors")


def curved(
    shape=(80, 80, 80),
    tA=2.5,
    tB=1.2,
    g=0.16,
    width_deg=180.0,
    origin_mm=(0.0, -1.0, 0.0),
    R=None,
):
    """The curved fundus: bank A (x > 0) of cortex tA thick, bank B (x < 0) of
    tB, a slot of outside medium of half-width g between them, all in mm.
    origin_mm: where the slot's edge sits, from the box centre (in scanner
    axes, as in fold_phantoms.Fold)."""
    p = dict(tA=float(tA), tB=float(tB), g=float(g), width=np.radians(width_deg))
    R = FP.R_DEFAULT.copy() if R is None else np.asarray(R, float)
    return Curved("bent", tuple(shape), p, R, tuple(origin_mm))


def make_curved(phantom, feats, noise=1.0, seed=0):
    """The image, labels and noise SDs, as fold_phantoms.make gives them."""
    return FP.make(phantom, None, noise=noise, seed=seed, feats=feats)


# ---------------------------------------------------------------- true columns


def _direction(ph, Y, sign):
    G = ph.grad_phi(Y)
    return sign[:, None] * G / np.linalg.norm(G, axis=1)[:, None]


def _rk4(ph, Y, sign, h):
    k1 = _direction(ph, Y, sign)
    k2 = _direction(ph, Y + 0.5 * h * k1, sign)
    k3 = _direction(ph, Y + 0.5 * h * k2, sign)
    k4 = _direction(ph, Y + h * k3, sign)
    return Y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def _last_step(ph, Y, sign, level, h, iters=6):
    """The RK4 step from Y of the length (at most h) that ends on phi = level,
    so that the end stays on the trajectory. Then Newton steps along
    grad(phi) remove the last rounding."""
    s = np.clip(
        np.abs(level - ph.phi(Y)) / np.linalg.norm(ph.grad_phi(Y), axis=1), 0, h
    )
    for _ in range(iters):
        Yn = _rk4(ph, Y, sign, s[:, None])
        g = np.linalg.norm(ph.grad_phi(Yn), axis=1)
        s = np.clip(s + sign * (level - ph.phi(Yn)) / g, 0, h)
    return _to_level(ph, _rk4(ph, Y, sign, s[:, None]), level)


def _to_level(ph, Y, level, iters=4):
    """Newton steps along grad(phi) from Y to phi = level."""
    Y = Y.copy()
    for _ in range(iters):
        G = ph.grad_phi(Y)
        Y += ((level - ph.phi(Y)) / np.sum(G * G, 1))[:, None] * G
    return Y


def _flow(ph, Y0, level, h=STEP, smax=SMAX):
    """Follow the orthogonal trajectory from each point of Y0 (local mm) toward
    phi = level, in steps of h. Returns the points (n, m, 3), padded with nan,
    and the count of points of each trace. The last point lies on the level."""
    n = len(Y0)
    m = int(np.ceil(smax / h)) + 2
    out = np.full((n, m, 3), np.nan)
    out[:, 0] = Y0
    ns = np.ones(n, int)
    sign = np.sign(level - ph.phi(Y0))
    done = sign == 0
    ns[done] = 1
    Y = Y0.copy()
    for k in range(1, m - 1):
        a = np.flatnonzero(~done)
        if a.size == 0:
            break
        Yn = _rk4(ph, Y[a], sign[a], h)
        cross = sign[a] * (ph.phi(Yn) - level) >= 0
        go, end = a[~cross], a[cross]
        out[go, k] = Yn[~cross]
        Y[go] = Yn[~cross]
        ns[go] = k + 1
        if end.size:
            out[end, k] = _last_step(ph, Y[end], sign[end], level, h)
            ns[end] = k + 1
            done[end] = True
    if not done.all():
        raise RuntimeError(f"{(~done).sum()} traces did not reach phi = {level}")
    for i in np.flatnonzero(sign == 0):
        out[i, 0] = _to_level(ph, Y0[i : i + 1], level)[0]
    return out, ns


def true_columns(ph, roots_vox, h=STEP):
    """The true column of each root (voxel coordinates, (N, 3)): a list of
    (m_i, 3) arrays in voxel coordinates, from phi = 0 (the white surface) to
    phi = 1 (the pia), in steps of h mm. The column is the orthogonal
    trajectory through the root. A root that is not on phi = 0 is first moved
    along its trajectory to phi = 0."""
    Y = ph.to_local(np.atleast_2d(roots_vox))
    P, ns = _flow(ph, Y, 0.0, h)
    Y0 = P[np.arange(len(Y)), ns - 1]
    P, ns = _flow(ph, Y0, 1.0, h)
    return [ph.to_vox(P[i, : ns[i]]) for i in range(len(Y))]


def pad(paths):
    """Paths as a padded array (N, m, 3) (nan after the end) and their counts,
    the form of the tracers' outputs."""
    ns = np.array([len(p) for p in paths], int)
    pos = np.full((len(paths), ns.max(), 3), np.nan)
    for i, p in enumerate(paths):
        pos[i, : len(p)] = p
    return pos, ns


def _dist_to_polyline(P, T):
    """The distance of each point of P (m, 3) from the polyline T (k, 3)."""
    A, d = T[:-1], np.diff(T, axis=0)
    dd = np.maximum(np.sum(d * d, 1), 1e-30)
    v = P[:, None] - A[None]
    s = np.clip(np.sum(v * d[None], 2) / dd[None], 0.0, 1.0)
    return np.linalg.norm(v - s[..., None] * d[None], axis=2).min(1)


def path_errors(ph, paths, pos, ns):
    """fold_eval.errors for bent true columns. paths: the true columns (voxel
    coordinates, from true_columns). pos (N, m, 3), ns: the traces (voxel
    coordinates). Per trace, in mm: the largest and the mean distance from the
    true path, over the trace's samples with 0 <= phi <= 1; the distance from
    the trace's last sample to the true path's end (on the pia); the contact
    error (nan: this phantom has no contact)."""
    n = pos.shape[0]
    emax, emean, eend, econ = (np.full(n, np.nan) for _ in range(4))
    for i in range(n):
        Y = ph.to_local(pos[i, : ns[i]].astype(float))
        T = ph.to_local(paths[i])
        p = ph.phi(Y)
        m = (p >= 0) & (p <= 1)
        if m.any():
            e = _dist_to_polyline(Y[m], T)
            emax[i], emean[i] = e.max(), e.mean()
        eend[i] = np.linalg.norm(Y[-1] - T[-1])
    return emax, emean, eend, econ
