"""Fold phantoms with known columns (plan: reports/Crossing paths and contacts from laminar
profiles.md, *Column traces at folds*). All 3D, 0.16 mm voxels.

Every phantom is a family of offset surfaces of a core set K: the laminae are the level sets of
the distance to K, so the true column through any point is the straight segment along the
direction from its foot on K, and the true column's two ends are known in closed form.

- crown: K a half-line (the axis of a capsule's WM core, radius w). WM within w of K, cortex
  from w to w + t, fluid beyond. The tip is a hemisphere (both curvatures 1/w), the shaft a
  cylinder (1/w and 0). Columns run outward from the WM.
- fundus: K a half-plane (the slot of fluid at the bottom of a sulcus, half-width g). Fluid
  within g of K, cortex from g to g + t, WM beyond. The banks are flat; round the slot's edge the
  white surface is a concave cylinder of radius g + t. Columns run from the WM toward K and
  converge at the fundus.
- kiss: K a line (the axis of concentric cylinders). Bank A (inner, convex) from Rc - tA to Rc,
  bank B (outer, concave) from Rc to Rc + tB, apposed at the contact cylinder Rc with no fluid,
  except in an open sector of opening angle `open_deg`, where a fluid gap of width g straddles
  the contact. Columns are radial: in the fused sector they run from one bank's WM through the
  contact to the other bank's WM; in the open sector they end at the gap.

Image: per voxel, the mean over 3^3 sub-samples of the site's median open profile (echo 2 and 3,
trend removed, against relative depth 0 at the WM to 1 at the top; fold_measure.py) in cortex,
and the site's WM and fluid levels outside (partial volume at the edges as in the data). Labels
at voxel centres. Noise: Gaussian on the log intensity, its SD per echo set so that the
high-pass statistic of the trend-removed echo in the cortex (fold_measure.hp_stat) matches the
site's. The features are then made exactly as curv_lib.load (trend removed over the cortex at
flatten_fused2.SHP voxels, masked).

Coordinates: voxel index space (voxel centres at integers); the geometry is defined in mm in a
local frame, y = R^T (X - c) h, with R an oblique rotation so that nothing is grid-aligned.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import flatten_fused2 as F
from curv_lib import face_dist
from mesh_smoothing import rotation

H = 0.16  # voxel (mm)
HERE = Path(__file__).parent
R_DEFAULT = rotation([1.0, 2.0, 3.0], 25.0)


@dataclass
class Fold:
    kind: str  # crown, fundus, kiss
    shape: tuple[int, int, int]
    p: dict = field(default_factory=dict)
    R: np.ndarray = field(default_factory=lambda: R_DEFAULT.copy())
    origin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)  # where the local origin sits, from the box centre

    # ---- frames
    def to_local(self, X):
        """Voxel coordinates (N, 3) to local mm."""
        c = (np.array(self.shape) - 1) / 2.0
        return ((np.asarray(X, float) - c) * H - np.array(self.origin_mm)) @ self.R

    def to_vox(self, Y):
        c = (np.array(self.shape) - 1) / 2.0
        return (np.asarray(Y, float) @ self.R.T + np.array(self.origin_mm)) / H + c

    def dir_to_vox(self, U):
        return np.asarray(U, float) @ self.R.T

    # ---- the core: foot point and distance
    def foot(self, Y):
        Y = np.asarray(Y, float)
        c = np.zeros_like(Y)
        if self.kind == "crown":  # half-line {(0, 0, z): z <= 0}
            c[:, 2] = np.minimum(Y[:, 2], 0.0)
        elif self.kind == "fundus":  # half-plane {x = 0, y >= 0}
            c[:, 1] = np.maximum(Y[:, 1], 0.0)
            c[:, 2] = Y[:, 2]
        elif self.kind == "kiss":  # line {(0, 0, z)}
            c[:, 2] = Y[:, 2]
        return c

    def geom(self, Y):
        """label (0 cortex, 1 fluid, 2 WM), relative depth (0 at the WM, 1 at the bank's top; nan
        outside the cortex), bank (0 A, 1 B), distance r from the core, foot c, unit u."""
        Y = np.asarray(Y, float)
        c = self.foot(Y)
        v = Y - c
        r = np.linalg.norm(v, axis=1)
        u = v / np.maximum(r, 1e-12)[:, None]
        p = self.p
        lab = np.full(len(Y), 2, np.int8)
        rel = np.full(len(Y), np.nan)
        bank = np.zeros(len(Y), np.int8)
        if self.kind == "crown":
            w, t = p["w"], p["t"]
            ctx = (r >= w) & (r <= w + t)
            lab[ctx] = 0
            lab[r > w + t] = 1
            rel[ctx] = (r[ctx] - w) / t
        elif self.kind == "fundus":
            g, t = p["g"], p["t"]
            ctx = (r >= g) & (r <= g + t)
            lab[ctx] = 0
            lab[r < g] = 1
            rel[ctx] = (g + t - r[ctx]) / t
        elif self.kind == "kiss":
            Rc, tA, tB, g = p["Rc"], p["tA"], p["tB"], p["g"]
            a = (r >= Rc - tA) & (r <= Rc)
            b = (r > Rc) & (r <= Rc + tB)
            lab[a | b] = 0
            rel[a] = (r[a] - (Rc - tA)) / tA
            rel[b] = (Rc + tB - r[b]) / tB
            bank[b] = 1
            gap = self.in_open(Y) & (np.abs(r - Rc) < g / 2)
            lab[gap] = 1
            rel[gap] = np.nan
        return lab, rel, bank, r, c, u

    def in_open(self, Y):
        """kiss: inside the open sector (angle about the local x axis within open_deg / 2)."""
        ang = np.degrees(np.arctan2(Y[:, 1], Y[:, 0]))
        return np.abs(ang) < self.p["open_deg"] / 2

    # ---- truth per root
    def column(self, X):
        """The true column through each root X (voxel coords, (N, 3)): foot c and unit u (local mm),
        the distances from the core where it starts (the root's WM surface) and ends (the top,
        or for a fused kiss root the far bank's WM), the contact distance (nan if none), and the
        white surface's mean curvature at the root (1/mm, + where convex)."""
        Y = self.to_local(X)
        lab, rel, bank, r, c, u = self.geom(Y)
        p = self.p
        n = len(Y)
        r0, r1, rc, hm = (np.full(n, np.nan) for _ in range(4))
        if self.kind == "crown":
            r0[:], r1[:] = p["w"], p["w"] + p["t"]
            hm[:] = np.where(Y[:, 2] > 0, 1.0 / p["w"], 0.5 / p["w"])
        elif self.kind == "fundus":
            r0[:], r1[:] = p["g"] + p["t"], p["g"]
            hm[:] = np.where(Y[:, 1] < 0, -0.5 / (p["g"] + p["t"]), 0.0)
        elif self.kind == "kiss":
            Rc, tA, tB, g = p["Rc"], p["tA"], p["tB"], p["g"]
            A = r < Rc
            op = self.in_open(Y)
            r0 = np.where(A, Rc - tA, Rc + tB)
            r1 = np.where(op, np.where(A, Rc - g / 2, Rc + g / 2), np.where(A, Rc + tB, Rc - tA))
            rc = np.where(op, np.nan, Rc)
            hm = np.where(A, 0.5 / (Rc - tA), -0.5 / (Rc + tB))
        return dict(c=c, u=u, r0=r0, r1=r1, rc=rc, hmean=hm)


# ---------------------------------------------------------------- the standard set

def standard(t=2.4):
    """The phantoms of the results: three crowns, two fundi, the kissing pair with and without the
    top's dark band at the contact (kiss_flat: the fused contact shows no notch)."""
    n_crown = lambda w: (int(2 * (w + t + 2.8) / H) + 1,) * 3  # noqa: E731
    out = {}
    for w in (0.32, 0.64, 1.28):
        out[f"crown_w{w:.2f}"] = Fold("crown", n_crown(w), dict(w=w, t=t), origin_mm=(0.0, 0.0, 0.0))
    for g in (0.16, 0.48):
        n = int(2 * (g + t + 2.8) / H) + 1
        out[f"fundus_g{g:.2f}"] = Fold("fundus", (n, n, n), dict(g=g, t=t), origin_mm=(0.0, 0.0, 0.0))
    kp = dict(Rc=3.6, tA=2.0, tB=2.6, g=0.32, open_deg=120.0)
    nk = int(2 * (kp["Rc"] + kp["tB"] + 2.8) / H) + 1
    out["kiss"] = Fold("kiss", (nk, nk, 64), dict(kp, flat=False))
    out["kiss_flat"] = Fold("kiss", (nk, nk, 64), dict(kp, flat=True))
    return out


# ---------------------------------------------------------------- the image

def features(site, path=None):
    J = json.load(open(path or HERE / "fold_features.json"))[site]
    return J


def profile_at(prof, rel):
    """The profile (bins over [0, 1]) at relative depths rel."""
    nb = len(prof)
    xs = (np.arange(nb) + 0.5) / nb
    return np.interp(rel, xs, prof)


def flat_top(prof, at=0.85):
    """The profile with its top (relative depth > at) held at its value at `at` (no dark top)."""
    nb = len(prof)
    xs = (np.arange(nb) + 0.5) / nb
    v = np.interp(at, xs, prof)
    return np.where(xs > at, v, prof)


def hp_stat(det, ctx):
    from scipy import ndimage as ndi

    m = ctx.astype(float)
    r = det - F.mconv(det, m, 1.0)
    inner = ndi.binary_erosion(ctx)
    v = r[inner]
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def render(fold, feats, sub=3):
    """Noise-free log intensity per echo (partial volume over sub^3 samples) and centre labels."""
    shape = fold.shape
    idx = np.indices(shape).reshape(3, -1).T.astype(float)
    lab, _, _, _, _, _ = fold.geom(fold.to_local(idx))
    labels = lab.reshape(shape)
    imgs = []
    flat = fold.kind == "kiss" and fold.p.get("flat", False)
    for e in ("echo2", "echo3"):
        prof = np.array(feats[e]["profile"])
        pf = flat_top(prof) if flat else prof
        acc = np.zeros(idx.shape[0])
        offs = (np.arange(sub) + 0.5) / sub - 0.5
        for ox in offs:
            for oy in offs:
                for oz in offs:
                    Y = fold.to_local(idx + np.array([ox, oy, oz]))
                    lb, rel, bank, _, _, _ = fold.geom(Y)
                    val = np.full(len(Y), feats[e]["wm_level"])
                    val[lb == 1] = feats[e]["fluid_level"]
                    c = lb == 0
                    if flat:  # the fused sector's banks have no dark top
                        fused = c & ~fold.in_open(Y)
                        val[fused] = profile_at(pf, rel[fused])
                        c = c & ~fused
                    val[c] = profile_at(prof, rel[c])
                    acc += val
        imgs.append((acc / sub ** 3).reshape(shape))
    return labels, imgs


def detrend(L, ctx):
    mb = ctx.astype(float)
    return (L - F.mconv(L, mb, F.SHP)) * mb


def make(fold, site, noise=1.0, seed=0, feats=None):
    """A phantom as a curv_lib.load-like dict (det {1: echo 2, 2: echo 3}, ctx, wm, rim, vox, dface,
    miss) plus labels (0 cortex, 1 fluid, 2 WM), the noise SDs used, and the fold. noise scales the
    matched SD (0: noise-free)."""
    feats = feats or features(site)
    labels, imgs = render(fold, feats)
    ctx, wm = labels == 0, labels == 2
    rng = np.random.default_rng(seed)
    det, sds = {}, []
    for k, (e, L0) in enumerate(zip(("echo2", "echo3"), imgs)):
        s0 = hp_stat(detrend(L0, ctx), ctx)
        k1 = hp_stat(detrend(np.random.default_rng(seed + 99).standard_normal(L0.shape), ctx), ctx)
        sd = np.sqrt(max(feats[e]["hp_stat"] ** 2 - s0 ** 2, 0.0)) / k1
        sds.append(float(sd))
        L = L0 + noise * sd * rng.standard_normal(L0.shape)
        det[k + 1] = detrend(L, ctx)
    rim = np.where(wm, 2, np.where(ctx, 3, 1)).astype(np.int16)
    vox = np.array([H, H, H])
    return dict(det=det, disp=det[1], ctx=ctx, wm=wm, rim=rim, miss=np.zeros_like(ctx), vox=vox,
                dface=face_dist(labels.shape, vox), labels=labels.astype(np.int16), noise_sd=sds,
                fold=fold, lo=np.zeros(3, int))
