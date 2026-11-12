"""The τ phantom of the flow-field phantoms stage (docs/plans/flow-field-phantoms.md, *Phantoms*,
row `tau`). Its use: "the warp alone" (docs/plans/column-solver.md, *Exact tests (stages 1 and
2)*). A solver must recover the true speeds and the true offsets, with the top of every complete
column at τ = 0. In the flat band, the speed must follow the speed prior and the levels on the two
sides must stay aligned. No level may shift across the areal border.

Local frame (mm), as in fold_phantoms.Fold, with the oblique rotation fold_phantoms.R_DEFAULT:

- x is across the cortex. u = the local y is along the bank: the thickness changes along u.
  z is along the bank: nothing changes along z.
- WM is x < 0 (a flat white surface at x = 0). The cortex is 0 <= x <= t(u). The outside medium
  is x > t(u). The thickness is t(u) = T0 (1 + A sin(2 pi u / P)), with T0 = 2.4 mm, A = 0.25 and
  P = 6 mm. So t goes from 1.8 mm to 3.0 mm.
- The box. The plan gives 96 x 64 x 48 voxels. The long side (96) must cover u. The column of
  R_DEFAULT for the local y is nearest to voxel axis 1, so the default shape is (64, 96, 48):
  96 voxels (15.36 mm, about 2.5 periods) on voxel axis 1.
- The origin. The box centre is the local point (T0 / 2, 0, 0): the cortex is centred across x
  (for the mean thickness), and the areal border u = 0 goes through the box centre. So
  origin_mm = -T0 / 2 * R[:, 0] (fold_phantoms.Fold.to_vox gives the box centre for that point).

Depth. The relative depth is phi = x / t(u), with the same formula everywhere in the box (phi < 0
in WM, phi > 1 in the outside medium). Its gradient is (1 / t, -x t'(u) / t^2, 0): it has a u
part where t changes, so the levels of phi are not parallel to the white surface. The true
columns are the orthogonal trajectories of phi. They bend a small amount: they are NOT straight.
This module does not trace them. The truth adapter (PR 1 task 1.3a) traces them from this
Geometry.

The true τ is (phi - 1) * tau_scale, with tau_scale = T0 = 2.4 mm (the mean thickness). The true
speed is ff_types.true_speed: 1 / (tau_scale |grad phi|). Where t'(u) = 0 (u at the extrema of
the sine), the speed is t(u) / tau_scale. The true offset of a root is the true τ at the root's
start point: START_DEPTH_MM (2 mm) behind the root along the white surface normal. The white
surface is flat, so the normal is +x and the start point of a root at (0, u, z) is (-2, u, z).
There phi = -2 / t(u), and the offset is (-2 / t(u) - 1) * tau_scale. true_speed_at and
true_offset_at give these at voxel positions.

Stop surface. The stop surface is the pia x = t(u). stop_sdf is the exact distance to it (not
x - t(u), which is not a distance on a wavy surface). It is a 1D minimisation over the foot's
u': a grid search, then Newton steps. Its sign is the sign of x - t(u). bank is 0 everywhere.

Image. Per point, the value is the profile at relative depth phi (profile_tau), with two changes:

(a) The flat band. Over relative depth BAND = (0.40, 0.55), the profile is held at its value at
    0.40. The depth map m(r) is r below 0.40, 0.40 on the band, and 0.40 + (r - 0.55) (1 - 0.40)
    / (1 - 0.55) above it; the profile is p(m(r)). So the band is inserted into the profile: the
    profile above the band is the site's profile from 0.40 to 1, compressed into 0.55 to 1. The
    profile is continuous at the two edges of the band by construction (no jump, so no fake
    band). Its slope changes at the edges (a kink).
(b) The areal border. At u > BORDER_U (0 mm), two bands have their amplitude multiplied by
    BORDER_SCALE (0.5). Band k is B_k(r) = W_k(r) (p(r) - trend(r)), with p the profile from (a).
    The trend is the least-squares line of p over relative depth 0 to 1 (TREND_SAMPLES equally
    spaced samples). W_k is 1 on the window's core and goes to 0 by a raised cosine over each
    taper. The profile is p - (1 - s(u)) (B_1 + B_2), with s = BORDER_SCALE at u > BORDER_U and 1
    elsewhere. So each side is the same background p - B_1 - B_2 plus s(u) times the same bands:
    a band changes only by a constant factor, and its extremum stays at the same depth. The
    windows (start, core start, core end, end) are WINDOWS: (0.10, 0.13, 0.24, 0.27) for the small
    bright band near 0.18 of the V1V2_banks profile, and (0.66, 0.70, 0.90, 0.93) for its large
    bright band near 0.86 (after the map of (a)). The extremum of each B_k of that profile is in
    the core. Both windows are outside the flat band, so the flat band is the same on the two
    sides. The windows avoid the dark band near 0.36 (it touches the flat band) and the dark top
    near 0.96 (it is next to the stop surface).

Render and features. render_tau copies the logic of fold_phantoms.render (3^3 sub-samples of
partial volume, labels at voxel centres, the WM and medium levels), with profile_tau in place of
profile_at. make_tau copies the logic of fold_phantoms.make (noise matched to the site's high-pass
statistic, the detrend, the dict). With no flat band and a border scale of 1 they give the same
arrays as fold_phantoms.make (a test checks this).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ff_types as T  # noqa: E402
import fold_phantoms as FP  # noqa: E402
from curv_lib import face_dist  # noqa: E402

H = FP.H
SHAPE = (64, 96, 48)  # the plan's 96 x 64 x 48, with the 96 on voxel axis 1 (the local y = u)
T0 = 2.4  # mm, the mean thickness (= tau_scale)
AMP = 0.25
PERIOD_MM = 6.0
BAND = (0.40, 0.55)  # the flat band (relative depth)
BORDER_U = 0.0  # mm
BORDER_SCALE = 0.5
WINDOWS = ((0.10, 0.13, 0.24, 0.27), (0.66, 0.70, 0.90, 0.93))  # (start, core start, core end, end)
TREND_SAMPLES = 1001
SDF_GRID = 65  # the grid search of stop_sdf
SDF_NEWTON = 12
SDF_CHUNK = 20000


# ---------------------------------------------------------------- the profile


def band_map(rel: Any, band: tuple[float, float] | None = BAND) -> np.ndarray:
    """The depth map m(r) of the flat band: r below the band, band[0] on it, and the interval
    (band[1], 1] mapped linearly onto (band[0], 1]. The identity if band is None."""
    r = np.asarray(rel, float)
    if band is None:
        return r
    lo, hi = band
    return np.where(r < lo, r, np.where(r <= hi, lo, lo + (r - hi) * (1.0 - lo) / (1.0 - hi)))


def window_weight(rel: Any, window: Any) -> np.ndarray:
    """W(r) of one window (start, core start, core end, end): 1 on the core, a raised cosine on
    each taper, 0 outside."""
    r = np.asarray(rel, float)
    a, b, c, d = window
    up = 0.5 - 0.5 * np.cos(np.pi * np.clip((r - a) / (b - a), 0.0, 1.0))
    dn = 0.5 + 0.5 * np.cos(np.pi * np.clip((r - c) / (d - c), 0.0, 1.0))
    return np.asarray(np.where(r < b, up, np.where(r <= c, 1.0, dn)))


def trend_tau(prof: Any, band: tuple[float, float] | None = BAND) -> tuple[float, float]:
    """(a, b) of the trend a + b r: the least-squares line of the flat-band profile over relative
    depth 0 to 1 (TREND_SAMPLES equal steps)."""
    r = np.linspace(0.0, 1.0, TREND_SAMPLES)
    p = FP.profile_at(np.asarray(prof, float), band_map(r, band))
    b, a = np.polyfit(r, p, 1)
    return float(a), float(b)


def bands_tau(prof: Any, rel: Any, band: tuple[float, float] | None = BAND,
              windows: Any = WINDOWS) -> np.ndarray:
    """The bands B_k(r) = W_k(r) (p(r) - trend(r)) at relative depths rel: shape (len(windows),
    *rel.shape). p is the profile with the flat band (a)."""
    prof = np.asarray(prof, float)
    r = np.asarray(rel, float)
    p = FP.profile_at(prof, band_map(r, band))
    a, b = trend_tau(prof, band)
    return np.stack([window_weight(r, w) * (p - (a + b * r)) for w in windows])


def profile_tau(prof: Any, rel: Any, u: Any, band: tuple[float, float] | None = BAND,
                scale: float = BORDER_SCALE, windows: Any = WINDOWS,
                border_u: float = BORDER_U) -> np.ndarray:
    """The τ phantom's profile at relative depths rel and local u (mm), arrays of one shape.
    prof: the site profile (bins over [0, 1]). See the module docstring, (a) and (b)."""
    prof = np.asarray(prof, float)
    r = np.asarray(rel, float)
    p = FP.profile_at(prof, band_map(r, band))
    if scale == 1.0:
        return np.asarray(p)
    s = np.where(np.asarray(u, float) > border_u, scale, 1.0)
    return np.asarray(p - (1.0 - s) * bands_tau(prof, r, band, windows).sum(0))


# ---------------------------------------------------------------- the geometry


@dataclass
class Tau(FP.Fold):
    """The τ phantom's geometry (kind "tau"). p holds t0, amp, period (mm). Implements
    ff_types.Geometry. geom gives the interface of fold_phantoms.render."""

    # ---- the thickness
    def thickness(self, u: Any) -> np.ndarray:
        p = self.p
        return np.asarray(p["t0"] * (1.0 + p["amp"] * np.sin(2 * np.pi * np.asarray(u, float) / p["period"])))

    def dthickness(self, u: Any) -> np.ndarray:
        p = self.p
        k = 2 * np.pi / p["period"]
        return np.asarray(p["t0"] * p["amp"] * k * np.cos(k * np.asarray(u, float)))

    def d2thickness(self, u: Any) -> np.ndarray:
        p = self.p
        k = 2 * np.pi / p["period"]
        return np.asarray(-p["t0"] * p["amp"] * k * k * np.sin(k * np.asarray(u, float)))

    # ---- ff_types.Geometry
    def phi(self, Y: Any) -> np.ndarray:
        Y = np.asarray(Y, float)
        return np.asarray(Y[:, 0] / self.thickness(Y[:, 1]))

    def grad_phi(self, Y: Any) -> np.ndarray:
        Y = np.asarray(Y, float)
        t, dt = self.thickness(Y[:, 1]), self.dthickness(Y[:, 1])
        G = np.zeros((len(Y), 3))
        G[:, 0] = 1.0 / t
        G[:, 1] = -Y[:, 0] * dt / t**2
        return G

    def label(self, Y: Any) -> np.ndarray:
        ph = self.phi(Y)
        return np.where(ph < 0, T.WM, np.where(ph > 1, T.MEDIUM, T.CORTEX)).astype(np.int8)

    def stopped(self, Y: Any) -> np.ndarray:
        return np.asarray(self.label(Y) == T.MEDIUM)

    def bank(self, Y: Any) -> np.ndarray:
        return np.zeros(len(np.asarray(Y)), np.int8)

    def stop_sdf(self, Y: Any) -> np.ndarray:
        """The signed distance (mm) to the pia x = t(u): the minimum over u' of
        sqrt((x - t(u'))^2 + (u - u')^2), signed as x - t(u). The foot u' is within |x - t(u)| of
        u (that is the distance to the point (t(u), u)). A grid search of SDF_GRID points on that
        interval, then SDF_NEWTON Newton steps on the half derivative of the squared distance,
        kept only where they lower it."""
        Y = np.asarray(Y, float)
        out = np.empty(len(Y))
        for i in range(0, len(Y), SDF_CHUNK):
            out[i:i + SDF_CHUNK] = self._sdf(Y[i:i + SDF_CHUNK, 0], Y[i:i + SDF_CHUNK, 1])
        return out

    def _sdf(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        v = np.abs(x - self.thickness(u))
        s = np.linspace(-1.0, 1.0, SDF_GRID)
        U = u[:, None] + v[:, None] * s[None, :]
        q = (x[:, None] - self.thickness(U)) ** 2 + (U - u[:, None]) ** 2
        k = np.argmin(q, 1)
        ub = U[np.arange(len(u)), k]
        qb = q[np.arange(len(u)), k]
        w = ub.copy()
        for _ in range(SDF_NEWTON):
            t, dt, d2 = self.thickness(w), self.dthickness(w), self.d2thickness(w)
            g = (w - u) - (x - t) * dt
            h = 1.0 + dt * dt - (x - t) * d2
            step = np.where(h > 1e-9, g / np.where(h > 1e-9, h, 1.0), 0.0)
            w = np.clip(w - step, u - v, u + v)
        qw = (x - self.thickness(w)) ** 2 + (w - u) ** 2
        d = np.sqrt(np.minimum(qb, qw))
        return np.asarray(np.where(x > self.thickness(u), d, -d))

    # ---- the interface of fold_phantoms.render
    def foot(self, Y: Any) -> np.ndarray:
        """The foot on the white surface along x (not the true column's root: columns bend)."""
        c = np.asarray(Y, float).copy()
        c[:, 0] = 0.0
        return c

    def geom(self, Y: Any) -> tuple[np.ndarray, ...]:
        """label (0 cortex, 1 medium, 2 WM), relative depth phi (nan outside the cortex), bank 0,
        the distance x from the white surface, the foot (0, u, z), the unit +x."""
        Y = np.asarray(Y, float)
        ph = self.phi(Y)
        lab = self.label(Y)
        rel = np.where(lab == T.CORTEX, ph, np.nan)
        u = np.zeros_like(Y)
        u[:, 0] = 1.0
        return lab, rel, self.bank(Y), Y[:, 0].copy(), self.foot(Y), u

    def column(self, X: Any) -> Any:
        raise NotImplementedError("true columns bend: trace the orthogonal trajectories of phi")

    def white_normal_vox(self, n: int) -> np.ndarray:
        """The white surface's unit normal toward the cortex (+x), in voxel axes, n times."""
        return np.repeat(self.dir_to_vox(np.array([[1.0, 0.0, 0.0]])), n, axis=0)


def geometry(shape: tuple[int, int, int] = SHAPE, t0: float = T0, amp: float = AMP,
             period: float = PERIOD_MM, R: np.ndarray | None = None) -> Tau:
    """The τ geometry, with the cortex centred in the box (origin_mm = -t0 / 2 * R[:, 0])."""
    R = FP.R_DEFAULT.copy() if R is None else np.asarray(R, float)
    origin = tuple(float(v) for v in -0.5 * t0 * R[:, 0])
    return Tau("tau", tuple(int(s) for s in shape), dict(t0=float(t0), amp=float(amp),  # type: ignore[arg-type]
                                                          period=float(period)), R, origin)  # type: ignore[arg-type]


# ---------------------------------------------------------------- the image


def render_tau(geo: Tau, feats: Any, band: tuple[float, float] | None = BAND,
               scale: float = BORDER_SCALE, windows: Any = WINDOWS, border_u: float = BORDER_U,
               sub: int = 3) -> tuple[np.ndarray, list[np.ndarray]]:
    """Noise-free log intensity per echo (partial volume over sub^3 samples) and centre labels.
    The logic of fold_phantoms.render, with profile_tau for the cortex."""
    shape = geo.shape
    idx = np.indices(shape).reshape(3, -1).T.astype(float)
    lab, _, _, _, _, _ = geo.geom(geo.to_local(idx))
    labels = lab.reshape(shape)
    imgs = []
    offs = (np.arange(sub) + 0.5) / sub - 0.5
    for e in ("echo2", "echo3"):
        prof = np.array(feats[e]["profile"])
        acc = np.zeros(idx.shape[0])
        for ox in offs:
            for oy in offs:
                for oz in offs:
                    Y = geo.to_local(idx + np.array([ox, oy, oz]))
                    lb, rel, _, _, _, _ = geo.geom(Y)
                    val = np.full(len(Y), feats[e]["wm_level"])
                    val[lb == 1] = feats[e]["fluid_level"]
                    c = lb == 0
                    val[c] = profile_tau(prof, rel[c], Y[c, 1], band, scale, windows, border_u)
                    acc += val
        imgs.append((acc / sub**3).reshape(shape))
    return labels, imgs


def make_tau(geo: Tau, feats: Any, noise: float = 1.0, seed: int = 0, **profile: Any) -> dict[str, Any]:
    """The image dict of fold_phantoms.make, from render_tau. The logic is a copy of
    fold_phantoms.make (noise SD matched to the site's high-pass statistic, the detrend)."""
    labels, imgs = render_tau(geo, feats, **profile)
    ctx, wm = labels == 0, labels == 2
    rng = np.random.default_rng(seed)
    det, sds = {}, []
    for k, (e, L0) in enumerate(zip(("echo2", "echo3"), imgs)):
        s0 = FP.hp_stat(FP.detrend(L0, ctx), ctx)
        k1 = FP.hp_stat(FP.detrend(np.random.default_rng(seed + 99).standard_normal(L0.shape), ctx), ctx)
        sd = np.sqrt(max(feats[e]["hp_stat"] ** 2 - s0**2, 0.0)) / k1
        sds.append(float(sd))
        L = L0 + noise * sd * rng.standard_normal(L0.shape)
        det[k + 1] = FP.detrend(L, ctx)
    rim = np.where(wm, 2, np.where(ctx, 3, 1)).astype(np.int16)
    vox = np.array([H, H, H])
    return dict(det=det, disp=det[1], ctx=ctx, wm=wm, rim=rim, miss=np.zeros_like(ctx), vox=vox,
                dface=face_dist(labels.shape, vox), labels=labels.astype(np.int16), noise_sd=sds,
                fold=geo, lo=np.zeros(3, int))


# ---------------------------------------------------------------- the phantom


def tau(feats: Any, site: str = "synthetic", noise: float = 0.0, seed: int = 0,
        shape: tuple[int, int, int] | None = None) -> T.Phantom:
    """The τ phantom of the plan's table. shape None gives SHAPE."""
    shape = SHAPE if shape is None else tuple(int(s) for s in shape)  # type: ignore[assignment]
    geo = geometry(shape)  # type: ignore[arg-type]
    image = make_tau(geo, feats, noise=noise, seed=seed)
    params = dict(shape=list(shape), origin_mm=list(geo.origin_mm), R="R_DEFAULT", voxel_mm=H,
                  t0=T0, amp=AMP, period_mm=PERIOD_MM, thickness="t0 (1 + amp sin(2 pi u / period_mm))",
                  white_x=0.0, band=list(BAND), band_value="profile at band[0]",
                  border_u_mm=BORDER_U, border_scale=BORDER_SCALE, windows=[list(w) for w in WINDOWS],
                  trend="least-squares line over relative depth [0, 1]", tau_scale=T0,
                  start_depth_mm=T.START_DEPTH_MM)
    return T.Phantom(name="tau", geometry=geo, image=image, features=feats, site=site,
                     noise=float(noise), seed=int(seed), tau_scale=T0, params=params)


def true_speed_at(ph: T.Phantom, X: Any) -> np.ndarray:
    """The true speed (mm per unit τ) at voxel positions X: ff_types.true_speed."""
    assert ph.geometry is not None
    return T.true_speed(ph.geometry, ph.tau_scale, np.asarray(X, float))


def true_offset_at(ph: T.Phantom, roots: Any) -> np.ndarray:
    """The true offset of each root (voxel positions on phi = 0): the true τ at its start point,
    START_DEPTH_MM behind the root along the white surface normal (+x)."""
    geo = ph.geometry
    assert isinstance(geo, Tau)
    roots = np.asarray(roots, float)
    starts = roots - (T.START_DEPTH_MM / H) * geo.white_normal_vox(len(roots))
    return T.true_tau(geo, ph.tau_scale, starts)
