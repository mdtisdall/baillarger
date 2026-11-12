"""The simple phantoms of the flow-field stage (docs/plans/flow-field-phantoms.md, task 1.2b):
slab, noise, fundus (also g = 0.04), crown, curved and abraded.

Each phantom has a geometry class that implements `ff_types.Geometry`. The classes are
subclasses of `fold_phantoms.Fold` or `bent_phantom.Curved`, so `fold_phantoms.make` renders
them. A builder returns an `ff_types.Phantom`.

Frames. A local frame (x, y, z) in mm, as in `fold_phantoms`. The local origin sits at
`origin_mm` (scanner axes) from the box centre. A local point Y is at R Y + origin_mm from the
box centre.

Labels are CORTEX 0, MEDIUM 1, WM 2. The sign of `stop_sdf` is positive in the stopped region.

- slab: WM for x < 0, cortex 0 <= x <= t, medium for x > t. phi = x / t.
- noise: the slab's box. Cortex label everywhere. The image is the profile at relative depth
  0.5, plus noise. It has no geometry.
- fundus: `fold_phantoms.Fold("fundus")`. phi = 1 - (r - g) / t, r the distance from K.
- crown: `fold_phantoms.Fold("crown")`. phi = (r - w) / t.
- curved: `bent_phantom.Curved`.
- abraded: curved, with two discs of bank A replaced by the outside medium. phi is not changed.
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

H = FP.H
T_DEFAULT = 2.4  # the cortex thickness of slab, noise, fundus and crown (mm)
PROFILE_AT = 0.5  # the relative depth of the noise phantom's constant profile

NAMES = [
    "slab",
    "noise",
    "fundus_g0.48",
    "fundus_g0.16",
    "fundus_g0.04",
    "crown_w0.32",
    "crown_w0.64",
    "crown_w1.28",
    "curved_g0.16",
    "curved_g0.04",
    "abraded",
]


# ---------------------------------------------------------------- shared methods


class _Common:
    """The methods that every geometry here shares. `geom` comes from the Fold."""

    def label(self, Y: T.Array) -> Any:
        return self.geom(Y)[0]  # type: ignore[attr-defined]

    def stopped(self, Y: T.Array) -> Any:
        return self.label(Y) == T.MEDIUM


def _unit_x(n: int) -> T.Array:
    u = np.zeros((n, 3))
    u[:, 0] = 1.0
    return u


# ---------------------------------------------------------------- slab and noise


@dataclass
class SlabGeometry(_Common, FP.Fold):
    """A flat cortex of thickness p["t"]. kind "slab"."""

    def geom(self, Y: T.Array) -> Any:
        Y = np.asarray(Y, float)
        t = self.p["t"]
        x = Y[:, 0]
        lab = np.where(x < 0, T.WM, np.where(x > t, T.MEDIUM, T.CORTEX)).astype(np.int8)
        rel = np.where(lab == T.CORTEX, x / t, np.nan)
        c = Y.copy()
        c[:, 0] = 0.0
        return lab, rel, np.zeros(len(Y), np.int8), x.copy(), c, _unit_x(len(Y))

    def phi(self, Y: T.Array) -> T.Array:
        return np.asarray(Y, float)[:, 0] / self.p["t"]

    def grad_phi(self, Y: T.Array) -> T.Array:
        return _unit_x(len(Y)) / self.p["t"]

    def stop_sdf(self, Y: T.Array) -> T.Array:
        return np.asarray(Y, float)[:, 0] - self.p["t"]

    def bank(self, Y: T.Array) -> Any:
        return np.zeros(len(Y), np.int8)


@dataclass
class NoiseFold(FP.Fold):
    """The renderer of the noise phantom. It is cortex everywhere, at relative depth
    PROFILE_AT. It is not a Geometry: the noise phantom has no truth. kind "noise"."""

    def geom(self, Y: T.Array) -> Any:
        Y = np.asarray(Y, float)
        n = len(Y)
        return (np.zeros(n, np.int8), np.full(n, PROFILE_AT), np.zeros(n, np.int8),
                np.zeros(n), np.zeros_like(Y), _unit_x(n))


# ---------------------------------------------------------------- fundus and crown


@dataclass
class FundusGeometry(_Common, FP.Fold):
    """`fold_phantoms.Fold("fundus")` with the depth function. p holds g and t.
    r is the distance from K, the half-plane {x = 0, y >= 0}."""

    def _r(self, Y: T.Array) -> T.Array:
        Y = np.asarray(Y, float)
        return np.asarray(np.hypot(Y[:, 0], np.minimum(Y[:, 1], 0.0)))

    def phi(self, Y: T.Array) -> T.Array:
        return 1.0 - (self._r(Y) - self.p["g"]) / self.p["t"]

    def grad_phi(self, Y: T.Array) -> T.Array:
        Y = np.asarray(Y, float)
        v = np.zeros_like(Y)
        v[:, 0] = Y[:, 0]
        v[:, 1] = np.minimum(Y[:, 1], 0.0)
        r = np.maximum(self._r(Y), 1e-12)
        return -v / (r * self.p["t"])[:, None]

    def stop_sdf(self, Y: T.Array) -> T.Array:
        return self.p["g"] - self._r(Y)

    def bank(self, Y: T.Array) -> Any:
        return np.where(np.asarray(Y, float)[:, 0] >= 0, 0, 1).astype(np.int8)


@dataclass
class CrownGeometry(_Common, FP.Fold):
    """`fold_phantoms.Fold("crown")` with the depth function. p holds w and t.
    r is the distance from K, the half-line {x = y = 0, z <= 0}."""

    def _v(self, Y: T.Array) -> T.Array:
        v = np.array(Y, float)
        v[:, 2] = np.maximum(v[:, 2], 0.0)
        return v

    def _r(self, Y: T.Array) -> T.Array:
        return np.asarray(np.linalg.norm(self._v(Y), axis=1))

    def phi(self, Y: T.Array) -> T.Array:
        return (self._r(Y) - self.p["w"]) / self.p["t"]

    def grad_phi(self, Y: T.Array) -> T.Array:
        r = np.maximum(self._r(Y), 1e-12)
        return self._v(Y) / (r * self.p["t"])[:, None]

    def stop_sdf(self, Y: T.Array) -> T.Array:
        return self._r(Y) - (self.p["w"] + self.p["t"])

    def bank(self, Y: T.Array) -> Any:
        return np.zeros(len(Y), np.int8)


# ---------------------------------------------------------------- curved and abraded


@dataclass
class CurvedGeometry(_Common, BP.Curved):
    """`bent_phantom.Curved` with the stop surface. The slot is the set r < g, with r the
    distance from K, so the distance to it is r - g where r >= g."""

    def stop_sdf(self, Y: T.Array) -> T.Array:
        r = self._polar(Y)[2]
        return self.p["g"] - r

    def bank(self, Y: T.Array) -> Any:
        return np.where(np.asarray(Y, float)[:, 0] < 0, 1, 0).astype(np.int8)


@dataclass(frozen=True)
class Patch:
    """One abraded disc on the pia of bank A. centre = (x, y, z) in the local frame, with
    x = g (the pia of the flat bank). The removed tissue is x > 0, rho < radius and
    0 <= x - centre[0] < depth, with rho the distance from the line through the centre
    along x."""

    centre: tuple[float, float, float]
    radius: float
    depth: float

    def rho(self, Y: T.Array) -> T.Array:
        Y = np.asarray(Y, float)
        return np.asarray(np.hypot(Y[:, 1] - self.centre[1], Y[:, 2] - self.centre[2]))

    def removed(self, Y: T.Array) -> Any:
        x = np.asarray(Y, float)[:, 0]
        d = x - self.centre[0]
        return (x > 0) & (self.rho(Y) < self.radius) & (d >= 0) & (d < self.depth)

    def solid(self, Y: T.Array) -> Any:
        """The stopped solid of the patch: rho < radius and 0 <= x < centre x + depth."""
        x = np.asarray(Y, float)[:, 0]
        return (self.rho(Y) < self.radius) & (x >= 0) & (x < self.centre[0] + self.depth)

    def sdf(self, Y: T.Array) -> T.Array:
        """Signed distance to the solid, positive inside. Exact: a capped cylinder."""
        x = np.asarray(Y, float)[:, 0]
        h = 0.5 * (self.centre[0] + self.depth)
        a = self.rho(Y) - self.radius
        b = np.abs(x - h) - h
        out = np.hypot(np.maximum(a, 0.0), np.maximum(b, 0.0))
        inside = np.minimum(np.maximum(a, b), 0.0)
        return -(out + inside)

    def record(self) -> dict[str, Any]:
        return dict(centre=list(self.centre), radius=self.radius, depth=self.depth)


def abraded_patches(g: float) -> tuple[Patch, ...]:
    return (Patch((g, 3.0, 0.0), 2.0, 0.3), Patch((g, 8.0, 0.0), 2.0, 0.9))


@dataclass
class AbradedGeometry(CurvedGeometry):
    """The curved fundus with `patches` of bank A replaced by the outside medium. phi, grad_phi
    and bank are those of the curved fundus. In a patch the label is MEDIUM."""

    patches: tuple[Patch, ...] = ()

    def geom(self, Y: T.Array) -> Any:
        lab, rel, bank, r, c, u = super().geom(Y)
        rem = np.zeros(len(lab), bool)
        for p in self.patches:
            rem |= p.removed(Y)
        lab = lab.copy()
        rel = rel.copy()
        lab[rem] = T.MEDIUM
        rel[rem] = np.nan
        return lab, rel, bank, r, c, u

    def patch_sdf(self, Y: T.Array, k: int) -> T.Array:
        """Signed distance to the solid of patch k, positive inside."""
        return self.patches[k].sdf(Y)

    def patch_of(self, Y: T.Array) -> Any:
        """The index of the patch whose solid holds each point, or -1 for none."""
        out = np.full(len(Y), -1, np.int64)
        for k, p in enumerate(self.patches):
            out[p.solid(Y)] = k
        return out

    def stop_sdf(self, Y: T.Array) -> T.Array:
        """max of the slot's and the patches' signed distances. In tissue this is minus the
        distance to the union, so it is exact there."""
        s = super().stop_sdf(Y)
        for p in self.patches:
            s = np.maximum(s, p.sdf(Y))
        return s


def abraded_box(
    g: float = 0.16,
    tA: float = 2.5,
    tB: float = 1.2,
    patches: tuple[Patch, ...] | None = None,
    R: T.Array | None = None,
    margin_mm: float = 1.0,
    multiple: int = 16,
) -> tuple[int, tuple[float, float, float]]:
    """The smallest cube side N (a multiple of `multiple` voxels) and the origin_mm such that
    all of these lie at least margin_mm inside the range of the voxel centres:
    the patches with their full depth to the white surface (x from 0 to g + tA), the fundus'
    round end (y down to -(g + tA)), and bank B between the round end and the last patch's far
    side. Each part is taken as its local bounding box, so the test is conservative. The
    constraint is convex, so the corners of the boxes decide. The origin centres the corners'
    extent in each scanner axis."""
    R = FP.R_DEFAULT if R is None else np.asarray(R, float)
    pts = abraded_patches(g) if patches is None else patches
    z0 = min(p.centre[2] - p.radius for p in pts)
    z1 = max(p.centre[2] + p.radius for p in pts)
    y1 = max(p.centre[1] + p.radius for p in pts)
    boxes = [((0.0, g + tA), (p.centre[1] - p.radius, p.centre[1] + p.radius),
              (p.centre[2] - p.radius, p.centre[2] + p.radius)) for p in pts]
    boxes.append(((-(g + tA), g + tA), (-(g + tA), 0.0), (z0, z1)))  # the round end
    boxes.append(((-(g + tB), 0.0), (0.0, y1), (z0, z1)))  # bank B
    Y = np.array([[x, y, z] for bx in boxes for x in bx[0] for y in bx[1] for z in bx[2]])
    S = Y @ R.T  # scanner axes, relative to the local origin
    lo, hi = S.min(0), S.max(0)
    ext = float((hi - lo).max())
    n_min = (ext + 2.0 * margin_mm) / H + 1.0
    N = int(multiple * np.ceil(n_min / multiple))
    origin = -(hi + lo) / 2.0
    return N, (float(origin[0]), float(origin[1]), float(origin[2]))


# ---------------------------------------------------------------- the geometry factory


def _params(shape: tuple[int, int, int], origin: tuple[float, float, float], **kw: Any) -> dict[str, Any]:
    d: dict[str, Any] = dict(shape=[int(s) for s in shape], origin_mm=[float(o) for o in origin],
                             R="R_DEFAULT")
    d.update(kw)
    return d


def _slab_origin(t: float = T_DEFAULT) -> tuple[float, float, float]:
    """The cortex mid-plane x = t / 2 passes through the box centre: the local point
    (t / 2, 0, 0) is at R (t / 2, 0, 0) + origin_mm = 0 from the centre."""
    o = -(t / 2.0) * FP.R_DEFAULT[:, 0]
    return (float(o[0]), float(o[1]), float(o[2]))


def _curved_p(tA: float, tB: float, g: float, width_deg: float) -> dict[str, float]:
    return dict(tA=float(tA), tB=float(tB), g=float(g), width=float(np.radians(width_deg)))


def geometry(name: str, shape: tuple[int, int, int] | None = None,
             origin_mm: tuple[float, float, float] | None = None) -> tuple[Any, float, dict[str, Any]]:
    """(geometry, tau_scale, params) of a phantom of NAMES, without the image. For "noise" the
    geometry is the NoiseFold (the renderer). shape=None gives the plan's box."""
    R = FP.R_DEFAULT
    t = T_DEFAULT
    if name in ("slab", "noise"):
        shp = shape or (64, 64, 48)
        org = origin_mm or _slab_origin(t)
        cls = SlabGeometry if name == "slab" else NoiseFold
        kind = "slab" if name == "slab" else "noise"
        geo = cls(kind, tuple(shp), dict(t=t), R.copy(), tuple(org))
        extra = dict(t=t) if name == "slab" else dict(t=t, relative_depth=PROFILE_AT)
        return geo, t, _params(shp, org, **extra)
    if name.startswith("fundus_g"):
        g = float(name[len("fundus_g"):])
        n = int(2 * (g + t + 2.8) / H) + 1
        shp = shape or (n, n, n)
        org = origin_mm or (0.0, 0.0, 0.0)
        geo = FundusGeometry("fundus", tuple(shp), dict(g=g, t=t), R.copy(), tuple(org))
        return geo, t, _params(shp, org, g=g, t=t)
    if name.startswith("crown_w"):
        w = float(name[len("crown_w"):])
        n = int(2 * (w + t + 2.8) / H) + 1
        shp = shape or (n, n, n)
        org = origin_mm or (0.0, 0.0, 0.0)
        geo = CrownGeometry("crown", tuple(shp), dict(w=w, t=t), R.copy(), tuple(org))
        return geo, t, _params(shp, org, w=w, t=t)
    if name.startswith("curved_g"):
        g = float(name[len("curved_g"):])
        shp = shape or (80, 80, 80)
        org = origin_mm or (0.0, -1.0, 0.0)
        p = _curved_p(2.5, 1.2, g, 180.0)
        geo = CurvedGeometry("bent", tuple(shp), p, R.copy(), tuple(org))
        return geo, (2.5 + 1.2) / 2.0, _params(shp, org, tA=2.5, tB=1.2, g=g, width_deg=180.0)
    if name == "abraded":
        g, tA, tB = 0.16, 2.5, 1.2
        patches = abraded_patches(g)
        N, org0 = abraded_box(g, tA, tB, patches)
        shp = shape or (N, N, N)
        org = origin_mm or org0
        p = _curved_p(tA, tB, g, 180.0)
        geo = AbradedGeometry("bent", tuple(shp), p, R.copy(), tuple(org), patches)
        prm = _params(shp, org, tA=tA, tB=tB, g=g, width_deg=180.0,
                      patches=[q.record() for q in patches], box_margin_mm=1.0,
                      box_rule="smallest cube, side a multiple of 16 voxels; patches to x = g + tA, "
                               "round end and bank B inside the voxel-centre range by 1 mm")
        return geo, (tA + tB) / 2.0, prm
    raise KeyError(name)


# ---------------------------------------------------------------- the builders


def _finish(name: str, geo: Any, tau_scale: float, params: dict[str, Any], feats: Any, site: str,
            noise: float, seed: int, as_geometry: bool = True) -> T.Phantom:
    image = FP.make(geo, site, noise=noise, seed=seed, feats=feats)
    return T.Phantom(name=name, geometry=geo if as_geometry else None, image=image, features=feats,
                     site=site, noise=float(noise), seed=int(seed), tau_scale=float(tau_scale),
                     params=params)


def slab(feats: Any, site: str = "synthetic", noise: float = 0.0, seed: int = 0,
         shape: tuple[int, int, int] | None = None) -> T.Phantom:
    geo, ts, prm = geometry("slab", shape)
    return _finish("slab", geo, ts, prm, feats, site, noise, seed)


def noise_only(feats: Any, site: str = "synthetic", noise: float = 0.0, seed: int = 0,
               shape: tuple[int, int, int] | None = None) -> T.Phantom:
    geo, ts, prm = geometry("noise", shape)
    return _finish("noise", geo, ts, prm, feats, site, noise, seed, as_geometry=False)


def fundus(g: float, feats: Any, site: str = "synthetic", noise: float = 0.0, seed: int = 0,
           shape: tuple[int, int, int] | None = None) -> T.Phantom:
    name = f"fundus_g{g:.2f}"
    geo, ts, prm = geometry(name, shape)
    return _finish(name, geo, ts, prm, feats, site, noise, seed)


def crown(w: float, feats: Any, site: str = "synthetic", noise: float = 0.0, seed: int = 0,
          shape: tuple[int, int, int] | None = None) -> T.Phantom:
    name = f"crown_w{w:.2f}"
    geo, ts, prm = geometry(name, shape)
    return _finish(name, geo, ts, prm, feats, site, noise, seed)


def curved(g: float, feats: Any, site: str = "synthetic", noise: float = 0.0, seed: int = 0,
           shape: tuple[int, int, int] | None = None) -> T.Phantom:
    name = f"curved_g{g:.2f}"
    geo, ts, prm = geometry(name, shape)
    return _finish(name, geo, ts, prm, feats, site, noise, seed)


def abraded(feats: Any, site: str = "synthetic", noise: float = 0.0, seed: int = 0,
            shape: tuple[int, int, int] | None = None,
            origin_mm: tuple[float, float, float] | None = None) -> T.Phantom:
    geo, ts, prm = geometry("abraded", shape, origin_mm)
    return _finish("abraded", geo, ts, prm, feats, site, noise, seed)


def build(name: str, feats: Any, site: str = "synthetic", noise: float = 0.0, seed: int = 0,
          shape: tuple[int, int, int] | None = None) -> T.Phantom:
    """The phantom `name` of NAMES."""
    if name not in NAMES:
        raise KeyError(name)
    if name == "slab":
        return slab(feats, site, noise, seed, shape)
    if name == "noise":
        return noise_only(feats, site, noise, seed, shape)
    if name == "abraded":
        return abraded(feats, site, noise, seed, shape)
    if name.startswith("fundus_g"):
        return fundus(float(name[len("fundus_g"):]), feats, site, noise, seed, shape)
    if name.startswith("crown_w"):
        return crown(float(name[len("crown_w"):]), feats, site, noise, seed, shape)
    return curved(float(name[len("curved_g"):]), feats, site, noise, seed, shape)


if __name__ == "__main__":
    N, org = abraded_box()
    print("abraded box", N, org)
