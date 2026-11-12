"""The connected kiss phantoms of the flow-field stage (docs/plans/flow-field-phantoms.md,
*Phantoms*): `ckiss_smooth`, `ckiss_sharp`, `ckiss_smooth_flat`, `ckiss_sharp_flat`.

The geometry is that of `bent_phantom.Curved`: a slot of outside medium between bank A (x > 0,
thickness tA) and bank B (x < 0, thickness tB), with a round end at the fundus. Here the half-width
of the slot depends on z: g(z). For z below the transition, g is wide (0.32 mm). For z above it, g is
thin (0.02 mm, thinner than a voxel: a fused contact). The two banks are one connected sheet of
cortex that goes from an open slot to a fused contact.

With r the distance from the half-plane K = {x = 0, y >= 0} in the x-y plane (z does not enter r),
and t(theta) the thickness of `bent_phantom`:

    phi = 1 - (r - g(z)) / t(theta)

Labels: WM for phi < 0, CORTEX for 0 <= phi <= 1, MEDIUM for phi > 1 (that is, r < g(z)).

The transition is the quintic S of `bent_phantom` over 5 mm (smooth) or over one voxel (sharp).

The flat variants hold the profile at relative depth 0.85 (`fold_phantoms.flat_top`) where
g(z) < 0.16 mm. In the fused part the contact then shows no dark band. This needs a render of
our own: the profile changes in space, and `fold_phantoms.render` selects only by the fold's kind.
For the plain case (no flat) `render` gives the same arrays as `fold_phantoms.make`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bent_phantom as BP  # noqa: E402
import ff_types as T  # noqa: E402
import fold_phantoms as FP  # noqa: E402

NAMES = ("ckiss_smooth", "ckiss_sharp", "ckiss_smooth_flat", "ckiss_sharp_flat")

TA, TB, WIDTH_DEG = 2.5, 1.2, 180.0
G_WIDE, G_THIN = 0.32, 0.02
SMOOTH_HALF = 2.5  # mm: the smooth transition is over z in [-2.5, 2.5]
SHARP_HALF = 0.5 * FP.H  # mm: the sharp transition is over one voxel
FUSED_G = 0.16  # mm: where g is below it, the contact is fused
SHAPE = (96, 96, 128)
ORIGIN_MM = (0.0, -1.0, 0.0)
TAU_SCALE = 0.5 * (TA + TB)
_K1, _K2, _ZOOMS = 201, 21, 5  # samples of the first pass, of each zoom, and the zoom count
_CHUNK = 20000


def _quintic(u: Any) -> Any:
    return u**3 * (10 - 15 * u + 6 * u**2)


@dataclass
class ConnectedKiss(BP.Curved):
    """A Curved fold whose slot half-width g depends on z. `p` holds tA, tB, width (rad), g_wide,
    g_thin, half (mm: half the length of the transition) and flat (bool)."""

    # ---- the slot half-width
    def g(self, z: Any) -> Any:
        """The slot half-width g (mm) at local z (mm)."""
        p = self.p
        u = np.clip((np.asarray(z, float) + p["half"]) / (2 * p["half"]), 0.0, 1.0)
        S = _quintic(u)  # same as g_wide + (g_thin - g_wide) S, but exact at S = 0 and 1
        return p["g_wide"] * (1.0 - S) + p["g_thin"] * S

    def dg(self, z: Any) -> Any:
        """dg / dz (no unit: mm per mm)."""
        p = self.p
        z = np.asarray(z, float)
        u = (z + p["half"]) / (2 * p["half"])
        inside = (u > 0) & (u < 1)
        uc = np.clip(u, 0.0, 1.0)
        dS = 30 * uc**2 * (1 - uc) ** 2
        return np.where(inside, (p["g_thin"] - p["g_wide"]) * dS / (2 * p["half"]), 0.0)

    # ---- the depth function
    def phi(self, Y: Any) -> Any:
        Y = np.asarray(Y, float)
        _, _, r, th, _ = self._polar(Y)
        return 1.0 - (r - self.g(Y[:, 2])) / self.thickness(th)

    def grad_phi(self, Y: Any) -> Any:
        Y = np.asarray(Y, float)
        G = super().grad_phi(Y)  # the in-plane part is that of Curved, with g(z) in place of g
        x, yq, r, th, low = self._polar(Y)
        rs = np.maximum(r, 1e-12)
        t, dt = self.thickness(th), self.dthickness(th)
        gz = self.g(Y[:, 2])
        dr = np.stack([x / rs, yq / rs], 1)
        dth = np.stack([-yq / rs**2, np.where(low, x / rs**2, 0.0)], 1)
        G[:, :2] = -dr / t[:, None] + ((r - gz) * dt / t**2)[:, None] * dth
        G[:, 2] = self.dg(Y[:, 2]) / t
        return G

    # ---- the interface of ff_types.Geometry
    def label(self, Y: Any) -> Any:
        p = self.phi(Y)
        return np.where(p < 0, T.WM, np.where(p > 1, T.MEDIUM, T.CORTEX)).astype(np.int8)

    def stopped(self, Y: Any) -> Any:
        return self.label(Y) == T.MEDIUM

    def bank(self, Y: Any) -> Any:
        _, _, _, th, _ = self._polar(Y)
        return (th < 0).astype(np.int8)

    def stop_sdf(self, Y: Any) -> Any:
        """The signed distance to the slot region {r' < g(z')}: positive inside the slot (there
        g(z) - r, a value of which only the sign is exact), negative in tissue. In tissue it is
        minus min over z' of sqrt(max(r - g(z'), 0)^2 + (z - z')^2)."""
        Y = np.asarray(Y, float)
        _, _, r, _, _ = self._polar(Y)
        z = Y[:, 2]
        dr = r - self.g(z)
        out = -dr
        idx = np.flatnonzero(dr > 0)
        for s in range(0, len(idx), _CHUNK):
            i = idx[s : s + _CHUNK]
            out[i] = -self._tissue_distance(r[i], z[i], dr[i])
        return out

    def _tissue_distance(self, r: Any, z: Any, d0: Any) -> Any:
        """min over z' in [z - d0, z + d0] of the distance to the slot cross-section at z'.
        d0 = r - g(z) is an upper bound (z' = z). A coarse pass, then zooms round the best z'."""
        lo, hi = z - d0, z + d0
        best = np.full(len(z), np.inf)
        for k, n in enumerate([_K1] + [_K2] * _ZOOMS):
            s = np.linspace(0.0, 1.0, n)
            zs = lo[:, None] + (hi - lo)[:, None] * s[None]
            f = np.hypot(np.maximum(r[:, None] - self.g(zs), 0.0), z[:, None] - zs)
            j = np.argmin(f, axis=1)
            fj = f[np.arange(len(z)), j]
            best = np.minimum(best, fj)
            step = (hi - lo) / (n - 1)
            zb = zs[np.arange(len(z)), j]
            lo, hi = zb - step, zb + step
        return best


# ---------------------------------------------------------------- the image


def _values(fold: ConnectedKiss, feats: Any, e: str, Y: Any, flat: bool) -> Any:
    """The noise-free log intensity of echo e at local points Y."""
    lb, rel, _, _, _, _ = fold.geom(Y)
    prof = np.array(feats[e]["profile"])
    val = np.full(len(Y), feats[e]["wm_level"])
    val[lb == 1] = feats[e]["fluid_level"]
    c = lb == 0
    if flat:  # the fused part has no dark top
        fused = c & (fold.g(Y[:, 2]) < FUSED_G)
        val[fused] = FP.profile_at(FP.flat_top(prof), rel[fused])
        c = c & ~fused
    val[c] = FP.profile_at(prof, rel[c])
    return val


def render(fold: ConnectedKiss, feats: Any, flat: bool, sub: int = 3) -> Any:
    """As `fold_phantoms.render` (partial volume over sub^3 samples), with the flat top in the
    fused part if `flat`. Returns the labels and the two noise-free images."""
    shape = fold.shape
    idx = np.indices(shape).reshape(3, -1).T.astype(float)
    lab = fold.geom(fold.to_local(idx))[0]
    imgs = []
    offs = (np.arange(sub) + 0.5) / sub - 0.5
    for e in ("echo2", "echo3"):
        acc = np.zeros(idx.shape[0])
        for ox in offs:
            for oy in offs:
                for oz in offs:
                    acc += _values(fold, feats, e, fold.to_local(idx + np.array([ox, oy, oz])), flat)
        imgs.append((acc / sub**3).reshape(shape))
    return lab.reshape(shape), imgs


def make(fold: ConnectedKiss, feats: Any, noise: float, seed: int, flat: bool) -> dict[str, Any]:
    """As `fold_phantoms.make`, with `render` above."""
    labels, imgs = render(fold, feats, flat)
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
    vox = np.array([FP.H] * 3)
    return dict(det=det, disp=det[1], ctx=ctx, wm=wm, rim=rim, miss=np.zeros_like(ctx), vox=vox,
                dface=FP.face_dist(labels.shape, vox), labels=labels.astype(np.int16),
                noise_sd=sds, fold=fold, lo=np.zeros(3, int))


# ---------------------------------------------------------------- the builder


def geometry(sharp: bool, shape: Any = None, origin_mm: Any = None) -> ConnectedKiss:
    p = dict(tA=TA, tB=TB, g=G_WIDE, width=np.radians(WIDTH_DEG), g_wide=G_WIDE, g_thin=G_THIN,
             half=SHARP_HALF if sharp else SMOOTH_HALF, flat=False)
    return ConnectedKiss("bent", tuple(shape or SHAPE), p, FP.R_DEFAULT.copy(),
                         tuple(ORIGIN_MM if origin_mm is None else origin_mm))


def ckiss(variant: str, feats: Any, site: str = "synthetic", noise: float = 0.0, seed: int = 0,
          shape: Any = None, origin_mm: Any = None) -> T.Phantom:
    """The connected kiss phantom `variant` (one of NAMES)."""
    if variant not in NAMES:
        raise ValueError(f"unknown variant {variant!r}; use one of {NAMES}")
    sharp, flat = "sharp" in variant, variant.endswith("_flat")
    geo = geometry(sharp, shape, origin_mm)
    geo.p["flat"] = flat
    image = make(geo, feats, noise, seed, flat)
    params = dict(shape=list(geo.shape), origin_mm=list(geo.origin_mm), R="R_DEFAULT", tA=TA, tB=TB,
                  width_deg=WIDTH_DEG, g_wide=G_WIDE, g_thin=G_THIN, z_half_mm=geo.p["half"],
                  transition="sharp" if sharp else "smooth", flat=flat, flat_at=0.85,
                  fused_g_mm=FUSED_G)
    return T.Phantom(name=variant, geometry=geo, image=image, features=feats, site=site,
                     noise=float(noise), seed=int(seed), tau_scale=TAU_SCALE, params=params)
