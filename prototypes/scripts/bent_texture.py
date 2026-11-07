"""Texture in the box of the bent-columns trial
(docs/plans/bent-columns-trial.md, phase 2): the eigenvalues of the
noise-whitened gradient tensor of orient, the fibre measure, the laminar
coherence, and views for the user's checkpoint H1 ("is there radial texture
in this box?"). The views are diagnostics. This module does not decide if
there is texture.

The measures, from the eigenvalues l1 >= l2 >= l3 of the tensor J at a voxel:

- the fibre measure f = (l2 - l3) / (l2 + l3), clipped to [0, 1], and its
  direction, the eigenvector of l3. Radial fibres give gradients across the
  fibre in two directions: l1 ~ l2 >> l3, so f is near 1 and the eigenvector
  of l3 is along the fibre.
- the laminar coherence c = (l1 - l2) / (l1 + l2), clipped to [0, 1], and the
  leading eigenvector, as orient.leading gives them. Laminae give
  l1 >> l2 ~ l3.
- the strength l2 + l3, for a later weight. It is large where the gradients
  have two strong directions.

l2 and l3 of a lamina come from the noise. Then f has the distribution that
the noise alone gives, and that distribution is not 0. The views show a
reference from an image of pure noise.

Directions: the eigenvectors are in the frame of the voxel axes (gradients
per mm along each axis). to_scanner turns them to scanner space, the frame of
the rays' normals.

Validity: a value of orient is right only where the array holds its whole
reach (orient.orient_reach). The voxels nearer than the reach to the
array's edge are not valid (NaN). The box of bent_box has a margin of 1 mm,
less than the reach (22 voxels at SIGMA). With --grow, orient runs on the
echoes and the dseg read again on the box grown by the reach, and the
results are cut to the box.

Outliers (optional, --outlier-k): a voxel of the cortex is an outlier where
it is dark in every echo: its detrended log echo (log echo minus orient's
local cortex level) is below -k robust SDs in each echo. The robust SD of an
echo is 1.4826 times the median absolute deviation of its detrended log echo
over the cortex. The outliers grow by OUTLIER_DILATE voxels (the derivative
scale) inside the cortex, and they leave the mask of the tensor only (not
the labels). A small vessel in the cortex is such an outlier: dark from the
first echo on, and with strong gradients round it. orient's masked
convolutions then fill its place from the cortex round it. k is set before a
run and is not tuned to the result.

The views sample the tensor along the straight rays of the inner columns,
from s = 0 to the first outside. They show the cortex beside references: the
WM, the outside medium, a pure-noise image and a laminar image (fields).

Run from the repository root (heavy runs under scripts/memguard):
    nix develop --command scripts/memguard 8 uv run python \\
        prototypes/scripts/bent_texture.py --box BOX.npz --noise NOISE.json \\
        --out DIR
It writes DIR/texture.npz (the fields), DIR/texture.json (summary numbers)
and the figures DIR/*.png.
"""

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from scipy.ndimage import binary_dilation, distance_transform_edt, map_coordinates

sys.path.insert(0, str(Path(__file__).parent))
import bent_box as BB  # noqa: E402

from baillarger import orient as O  # noqa: E402
from baillarger import rays as R  # noqa: E402
from baillarger.blocks import open_volume  # noqa: E402
from baillarger.orient_noise import read_noise  # noqa: E402

type F64 = npt.NDArray[np.float64]
type Bool = npt.NDArray[np.bool_]
type I64 = npt.NDArray[np.int64]

#: the parts of the split of the columns
BANK_N = -1
FUNDUS = 0
BANK_P = 1
PART_NAMES = {BANK_N: "bank -u", FUNDUS: "fundus", BANK_P: "bank +u"}
#: the regions and reference images of fields
REGIONS = ("cortex", "wm", "outside", "noise", "laminar")
#: a column is in the fundus if its normal is within this angle (degrees) of
#: the plane normal to u (|n . u| < sin of it)
FUNDUS_DEG = 20.0
#: the depth bins along the straight rays, mm
BIN_MM = 0.16
#: the views show a depth bin only if it has at least this many samples
MIN_SAMPLES = 100
#: the seed of the reference images
SEED = 0
#: the period of the laminar reference image, mm
LAMINA_MM = 0.8
#: the outliers grow by this many voxels inside the cortex
OUTLIER_DILATE = 1
#: the robust SD from the median absolute deviation (normal distribution)
MAD_SD = 1.4826


# ---------------------------------------------------------------- measures


@dataclass(frozen=True)
class Texture:
    """The eigen-decomposition of a tensor field (6, ...) and the measures.
    NaN where the tensor is not finite."""

    #: (3, ...) l1 >= l2 >= l3
    values: F64
    #: (..., 3) the unit eigenvector of l1, canonical sign (orient.canonical)
    leading: F64
    #: (..., 3) the unit eigenvector of l3, canonical sign: the fibre direction
    fibre: F64
    #: (...) (l2 - l3) / (l2 + l3) in [0, 1]
    f: F64
    #: (...) (l1 - l2) / (l1 + l2) in [0, 1]
    c: F64
    #: (...) l2 + l3
    strength: F64


def texture(t: npt.ArrayLike) -> Texture:
    """The Texture of a tensor field t (6, ...), components in the order of
    orient.PAIRS."""
    a = np.asarray(t, np.float64)
    shp = a.shape[1:]
    flat = a.reshape(6, -1)
    ok = np.all(np.isfinite(flat), 0)
    M = np.zeros((flat.shape[1], 3, 3))
    for q, (i, j) in enumerate(O.PAIRS):
        M[:, i, j] = M[:, j, i] = np.where(ok, flat[q], 0.0)
    w, V = np.linalg.eigh(M)  # ascending
    l1, l2, l3 = w[:, 2], w[:, 1], w[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.clip((l2 - l3) / (l2 + l3), 0.0, 1.0)
        c = np.clip((l1 - l2) / (l1 + l2), 0.0, 1.0)
    f = np.where(ok & (l2 + l3 > 0), f, np.nan)
    c = np.where(ok & (l1 + l2 > 0), c, np.nan)
    vals = np.where(ok[None], np.stack([l1, l2, l3]), np.nan)
    lead = np.where(ok[:, None], V[:, :, 2], np.nan)
    fib = np.where(ok[:, None], V[:, :, 0], np.nan)
    return Texture(
        values=vals.reshape(3, *shp),
        leading=O.canonical(lead.reshape(*shp, 3)),
        fibre=O.canonical(fib.reshape(*shp, 3)),
        f=f.reshape(shp),
        c=c.reshape(shp),
        strength=np.where(ok, l2 + l3, np.nan).reshape(shp),
    )


def valid_mask(shape: Sequence[int], reach: Sequence[int]) -> Bool:
    """True at the voxels of a box of shape that are at least reach[a]
    voxels from the box's edge on each axis a."""
    m = np.zeros(tuple(shape), bool)
    sl = tuple(slice(r, n - r) for n, r in zip(shape, reach, strict=True))
    m[sl] = True
    return m


def sample_tensor(t: F64, points: npt.ArrayLike, order: Literal[0, 1] = 1) -> F64:
    """The tensor field t (6, X, Y, Z) at points (P, 3) in box voxel
    coordinates: (6, P). order 1 interpolates each component trilinearly
    (a NaN in the 8 voxels round a point gives NaN). order 0 takes the
    nearest voxel. NaN outside the box."""
    p = np.asarray(points, np.float64).T
    out = np.stack(
        [
            map_coordinates(t[q], p, order=order, mode="constant", cval=np.nan)
            for q in range(6)
        ]
    )
    return out


def to_scanner(v: npt.ArrayLike, affine: F64, voxel_mm: npt.ArrayLike) -> F64:
    """Unit directions v (..., 3) in the frame of the voxel axes (mm along
    each axis) to unit directions in scanner space. The map is
    affine[:3, :3] diag(1 / voxel_mm)."""
    D = np.asarray(affine, np.float64)[:3, :3] / np.asarray(voxel_mm, np.float64)
    w = np.asarray(v, np.float64) @ D.T
    out: F64 = w / np.linalg.norm(w, axis=-1, keepdims=True)
    return out


def axis_angle(v: F64, n: F64) -> F64:
    """The angle in degrees, 0 to 90, between the axes v (..., 3) and
    n (..., 3), both unit: arccos |v . n|."""
    d = np.abs(np.sum(v * n, -1))
    out: F64 = np.degrees(np.arccos(np.clip(d, 0.0, 1.0)))
    return out


def noise_image(
    level: F64, sigma: npt.ArrayLike, corr: npt.ArrayLike, rng: np.random.Generator
) -> F64:
    """exp(level) (E, ...) plus Gaussian noise with the covariance
    sigma_e sigma_f R_ef between the echoes, white in space."""
    s = np.asarray(sigma, np.float64)
    C = np.outer(s, s) * np.asarray(corr, np.float64)
    L = np.linalg.cholesky(C)
    z = rng.standard_normal((len(s), level[0].size))
    out: F64 = np.exp(level) + (L @ z).reshape(level.shape)
    return out


# ---------------------------------------------------------------- columns


def split_banks(
    normals: npt.ArrayLike, vertex_index: int, use: Bool | None = None
) -> tuple[npt.NDArray[np.int8], F64]:
    """The split of the columns into two banks and the fundus, by their
    normals (n, 3). u is the direction of the largest spread of the normals
    (of the columns in use), made perpendicular to the normal of the vertex
    column, with its component of largest magnitude positive. A column is
    BANK_P if n . u > sin(FUNDUS_DEG), BANK_N if n . u < -sin(FUNDUS_DEG),
    else FUNDUS. Returns the part (n,) and u (3,)."""
    n = np.asarray(normals, np.float64)
    m = n if use is None else n[use]
    d = m - m.mean(0)
    _, V = np.linalg.eigh(d.T @ d)
    nv = n[vertex_index] / np.linalg.norm(n[vertex_index])
    u = V[:, 2] - (V[:, 2] @ nv) * nv
    u = O.canonical(u / np.linalg.norm(u))
    a = n @ u
    cut = np.sin(np.radians(FUNDUS_DEG))
    part = np.where(a > cut, BANK_P, np.where(a < -cut, BANK_N, FUNDUS))
    return part.astype(np.int8), u


def first_outside(
    interval_start: I64, s0: npt.ArrayLike, s1: npt.ArrayLike, cls: npt.ArrayLike
) -> F64:
    """The s (mm) of the first outside on each column's straight ray: the
    first interval of class OUTSIDE that ends at s > 0, from max(s0, 0).
    NaN if there is none (censored)."""
    s0a = np.asarray(s0, np.float64)
    s1a = np.asarray(s1, np.float64)
    hit = (np.asarray(cls) == R.OUTSIDE) & (s1a > 0)
    out = np.full(len(interval_start) - 1, np.nan)
    for k in range(len(out)):
        q = np.flatnonzero(hit[interval_start[k] : interval_start[k + 1]])
        if len(q):
            out[k] = max(s0a[interval_start[k] + q[0]], 0.0)
    return out


@dataclass(frozen=True)
class RaySamples:
    """The straight-ray samples of the chosen columns between s = 0 and the
    first outside (CAP_OUT if censored), with the texture there."""

    column: I64
    s: F64
    f: F64
    c: F64
    strength: F64
    #: degrees, 0 to 90: the fibre direction and the leading eigenvector
    #: against the ray (the column's normal)
    angle_fibre: F64
    angle_leading: F64


def ray_samples(b: BB.Box, J: F64, columns: I64, top: F64) -> RaySamples:
    """The RaySamples of the given columns: the tensor J interpolated
    trilinearly (sample_tensor, order 1) at the straight samples."""
    r = b.rays
    count = np.diff(r.sample_start)
    col = np.repeat(np.arange(len(count)), count)
    s = r.s.astype(np.float64)
    end = np.where(np.isfinite(top), top, R.CAP_OUT)
    pick = np.isin(col, columns) & (s >= 0) & (s < end[col])
    idx = np.flatnonzero(pick)
    p = b.to_box(r.position[idx])
    tx = texture(sample_tensor(J, p, order=1))
    n = b.normals.astype(np.float64)[col[idx]]
    return RaySamples(
        column=col[idx],
        s=s[idx],
        f=tx.f,
        c=tx.c,
        strength=tx.strength,
        angle_fibre=axis_angle(to_scanner(tx.fibre, b.affine, b.voxel_mm), n),
        angle_leading=axis_angle(to_scanner(tx.leading, b.affine, b.voxel_mm), n),
    )


# ---------------------------------------------------------------- fields


@dataclass(frozen=True)
class Grown:
    """The echoes and labels of a box grown by the reach of orient on each
    side (cut to the grid), and the slices of the box in it."""

    echoes: F64
    labels: npt.NDArray[np.int32]
    crop: tuple[slice, slice, slice]


def read_grown(b: BB.Box, side: dict[str, Any], reach: Sequence[int]) -> Grown:
    """Read the echoes and the dseg that bent_box read (paths in its sidecar
    side) on the box grown by reach voxels, with the painted crops of the
    sidecar applied to the labels (bent_box.paint). The part in the box must
    equal the box's echoes and labels, else ValueError."""
    srcs = [open_volume(Path(e)) for e in side["echoes"]]
    lab = open_volume(Path(side["dseg"]))
    shape = np.asarray(lab.shape[:3], np.int64)
    n = np.asarray(b.labels.shape, np.int64)
    lo = np.maximum(b.offset - np.asarray(reach), 0)
    hi = np.minimum(b.offset + n + np.asarray(reach), shape)
    sl = tuple(slice(int(x), int(y)) for x, y in zip(lo, hi, strict=True))
    echoes = np.stack([np.asarray(s.read(sl), np.float64) for s in srcs])
    labels = np.ascontiguousarray(lab.read(sl), np.int32)
    if side.get("paint"):
        labels, _ = BB.paint(labels, lo, [Path(c) for c in side["paint"]])
    o = b.offset - lo
    crop = (
        slice(int(o[0]), int(o[0] + n[0])),
        slice(int(o[1]), int(o[1] + n[1])),
        slice(int(o[2]), int(o[2] + n[2])),
    )
    if not np.array_equal(labels[crop], b.labels):
        raise ValueError("the grown labels differ from the box's labels")
    if not np.array_equal(echoes[(slice(None), *crop)].astype(np.float32), b.echoes):
        raise ValueError("the grown echoes differ from the box's echoes")
    return Grown(echoes, labels, crop)


def outliers(
    echoes: F64,
    cortex: Bool,
    voxel_mm: npt.ArrayLike,
    k: float,
    sigma_mm: float = O.SIGMA,
    dilate: int = OUTLIER_DILATE,
) -> tuple[Bool, F64]:
    """The outliers of the cortex (see the module docstring) and the robust
    SD of each echo's detrended log echo. echoes (E, ...), cortex (...)."""
    ctx = np.asarray(cortex, bool)
    lv = O.local_level(echoes, ctx, sigma_mm, voxel_mm)
    det = O.log_echo(echoes) - lv
    use = ctx & np.all(np.isfinite(det), 0)
    sd = np.array(
        [MAD_SD * float(np.median(np.abs(d[use] - np.median(d[use])))) for d in det]
    )
    z = det / sd.reshape((-1,) + (1,) * ctx.ndim)
    dark = use & np.all(np.nan_to_num(z, nan=0.0) < -k, 0)
    if dilate > 0 and dark.any():
        dark = binary_dilation(dark, iterations=dilate) & ctx
    return dark, sd


def fields(
    b: BB.Box,
    noise: O.NoiseModel,
    sigma_mm: float = O.SIGMA,
    grown: Grown | None = None,
    outlier_k: float | None = None,
) -> dict[str, Any]:
    """The tensors on the box (NaN where not valid), each orient_arrays on
    its own mask: the cortex, the WM and the outside medium; and two
    reference images on the cortex mask, both with the thermal noise of the
    model (noise_image): "noise", the cortex level only; "laminar", the
    level plus laminae parallel to the WM (a cosine of the distance to the
    WM, period LAMINA_MM, with the data's detrended variance). With grown,
    orient runs on the grown arrays and the results are cut to the box. With
    outlier_k, the outliers (outliers) leave the cortex mask of every tensor
    here; "outliers" holds them (cut to the box)."""
    vs = b.voxel_mm
    reach = O.orient_reach(vs, sigma_mm)
    if grown is None:
        echoes = b.echoes.astype(np.float64)
        labels = np.asarray(b.labels)
        crop = tuple(slice(0, k) for k in labels.shape)
    else:
        echoes, labels, crop = grown.echoes, grown.labels, grown.crop
    valid = valid_mask(labels.shape, reach)
    inbox = np.zeros(labels.shape, bool)
    inbox[crop] = True
    masks = {
        "cortex": labels == b.cortex,
        "wm": np.isin(labels, b.filled),
        "outside": labels == 0,
    }
    out: dict[str, Any] = {"reach": reach}
    odd = np.zeros(labels.shape, bool)
    out["outlier_sd"] = None
    if outlier_k is not None:
        odd, out["outlier_sd"] = outliers(
            echoes, masks["cortex"], vs, outlier_k, sigma_mm
        )
        masks["cortex"] = masks["cortex"] & ~odd
    for k, m in masks.items():
        a = O.orient_arrays(echoes, m, noise, vs, sigma_mm)
        out[k] = np.where(valid, a["tensor"], np.nan)
        if k == "cortex":
            out["level"] = a["level"]
    ctx = masks["cortex"]
    lv = np.where(ctx, out["level"], 0.0)
    rng = np.random.default_rng(SEED)
    img = noise_image(lv, noise.sigma, noise.R, rng)
    a = O.orient_arrays(img, ctx, noise, vs, sigma_mm)
    out["noise"] = np.where(valid, a["tensor"], np.nan)
    # laminae only: the log level plus a cosine of the distance to the WM,
    # with the detrended variance of the data in each echo (valid cortex of
    # the box)
    det = O.log_echo(echoes) - lv
    use = ctx & valid & inbox
    sd = np.array([np.std(det[e][use]) for e in range(len(det))])
    dist = distance_transform_edt(~masks["wm"], sampling=vs)
    wave = np.cos(2 * np.pi * dist / LAMINA_MM)
    lam = lv + np.sqrt(2) * sd[:, None, None, None] * wave
    img = noise_image(lam, noise.sigma, noise.R, rng)
    a = O.orient_arrays(img, ctx, noise, vs, sigma_mm)
    out["laminar"] = np.where(valid, a["tensor"], np.nan)
    c3 = (slice(None), *crop)
    for k in REGIONS:
        out[k] = out[k][c3]
    out["level"] = out["level"][c3]
    out["valid"] = valid[crop]
    out["masks"] = {k: m[crop] for k, m in masks.items()}
    out["outliers"] = odd[crop]
    out["detrended_sd"] = sd
    out["grown"] = grown is not None
    return out


# ---------------------------------------------------------------- summaries


def _q(x: F64) -> tuple[float, float, float]:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    a, m, c = np.percentile(x, [25, 50, 75])
    return float(a), float(m), float(c)


def by_depth(rs: RaySamples, sel: Bool, edges: F64) -> dict[str, F64]:
    """Per depth bin: the count of finite samples, and the quartiles of f,
    c and the two angles."""
    k = np.digitize(rs.s[sel], edges) - 1
    out: dict[str, list[Any]] = {
        n: [] for n in ("count", "f", "c", "angle_fibre", "angle_leading")
    }
    for i in range(len(edges) - 1):
        m = k == i
        fin = m & np.isfinite(rs.f[sel])
        out["count"].append(int(fin.sum()))
        for name in ("f", "c", "angle_fibre", "angle_leading"):
            out[name].append(_q(getattr(rs, name)[sel][fin]))
    return {k2: np.asarray(v, np.float64) for k2, v in out.items()}


def _r(x: float) -> float | None:
    return None if not np.isfinite(x) else round(float(x), 4)


# ---------------------------------------------------------------- views


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def view_split(
    path: Path,
    b: BB.Box,
    part: npt.NDArray[np.int8],
    u: F64,
    top: F64,
    vi: int,
    cover: F64,
) -> None:
    """The roots in two projections, coloured by part, by the first outside
    and by cover (the share of each column's ray samples with a valid
    value), the histogram of n . u with the cuts, and the first outside of
    each part."""
    plt = _plt()
    nv = b.normals[vi].astype(np.float64)
    t = np.cross(nv, u)
    d = b.roots.astype(np.float64) - b.roots[vi]
    x_t, x_u, x_n = d @ t, d @ u, d @ nv
    inn = b.inner
    col = {BANK_N: "tab:blue", FUNDUS: "tab:orange", BANK_P: "tab:green"}
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for p, cc in col.items():
        m = inn & (part == p)
        lab = f"{PART_NAMES[p]} ({m.sum()})"
        ax[0, 0].scatter(x_t[m], x_u[m], s=1, c=cc, label=lab, rasterized=True)
        ax[0, 1].scatter(x_u[m], x_n[m], s=1, c=cc, label=lab, rasterized=True)
    for a in (ax[0, 0], ax[0, 1]):
        h = ~inn
        xx = x_t if a is ax[0, 0] else x_u
        yy = x_u if a is ax[0, 0] else x_n
        a.scatter(xx[h], yy[h], s=1, c="0.8", label="halo", rasterized=True)
        a.plot(xx[vi], yy[vi], "r*", ms=14, label="vertex")
        a.set_aspect("equal")
        lg = a.legend(fontsize=8)
        for hd in lg.legend_handles[:-1]:
            hd.set_sizes([20])
    ax[0, 0].set(
        xlabel="t = n_v x u (mm)",
        ylabel="u, across the sulcus (mm)",
        title="roots, top view",
    )
    ax[0, 1].set(
        xlabel="u (mm)", ylabel="n_v, vertex normal (mm)", title="roots, cut view"
    )
    cen = ~np.isfinite(top) & inn
    sc = ax[1, 0].scatter(
        x_t[inn & ~cen],
        x_u[inn & ~cen],
        s=1,
        c=top[inn & ~cen],
        cmap="viridis",
        vmin=0,
        vmax=6,
        rasterized=True,
    )
    ax[1, 0].scatter(
        x_t[cen], x_u[cen], s=1, c="red", label="censored", rasterized=True
    )
    ax[1, 0].plot(x_t[vi], x_u[vi], "k*", ms=14)
    fig.colorbar(sc, ax=ax[1, 0], label="first outside on the straight ray (mm)")
    ax[1, 0].set(xlabel="t (mm)", ylabel="u (mm)", title="first outside, inner columns")
    ax[1, 0].set_aspect("equal")
    lg = ax[1, 0].legend(fontsize=8)
    lg.legend_handles[0].set_sizes([20])
    a = b.normals[inn].astype(np.float64) @ u
    ax[1, 1].hist(a, bins=60, color="0.4")
    cut = np.sin(np.radians(FUNDUS_DEG))
    for v in (-cut, cut):
        ax[1, 1].axvline(v, color="r")
    ax[1, 1].set(
        xlabel="n . u",
        ylabel="inner columns",
        title=f"split: |n . u| < sin {FUNDUS_DEG:g} deg is fundus",
    )
    sc = ax[0, 2].scatter(
        x_t[inn],
        x_u[inn],
        s=1,
        c=cover[inn],
        cmap="magma",
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    ax[0, 2].plot(x_t[vi], x_u[vi], "c*", ms=14)
    fig.colorbar(sc, ax=ax[0, 2], label="share of samples with a finite f")
    ax[0, 2].set(xlabel="t (mm)", ylabel="u (mm)", title="cover of the ray samples")
    ax[0, 2].set_aspect("equal")
    rng = (0.0, R.CAP_OUT)
    for p, cc in col.items():
        m = inn & (part == p) & np.isfinite(top)
        ax[1, 2].hist(
            top[m],
            bins=40,
            range=rng,
            histtype="step",
            color=cc,
            lw=1.5,
            label=f"{PART_NAMES[p]}, {int((inn & (part == p)).sum() - m.sum())} "
            "censored",
        )
    ax[1, 2].set(
        xlabel="first outside on the straight ray (mm)",
        ylabel="inner columns",
        title="first outside by part",
    )
    ax[1, 2].legend(fontsize=8)
    fig.suptitle(
        f"vertex {b.vertex}: u = ({u[0]:.2f}, {u[1]:.2f}, {u[2]:.2f}) scanner, "
        "the largest spread of the inner normals, perpendicular to n_v"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def view_depth(
    path: Path,
    prof: dict[int, dict[str, F64]],
    edges: F64,
    measure: str,
    angle: str,
    refs: dict[str, tuple[float, float, float]],
    hi: dict[int, F64],
    lam: dict[int, dict[str, F64]],
    title: str,
) -> None:
    """For each part: the measure by depth (median and quartiles) with the
    references, and the angle by depth (median and quartiles; and the median
    of the samples where the measure is above the noise's 90th percentile).
    lam is the laminar reference image on the same rays."""
    plt = _plt()
    mid = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey="row")
    styles = {
        "noise": ("0.5", ":"),
        "wm": ("tab:purple", "--"),
        "outside": ("k", "-."),
        "laminar": ("tab:olive", "--"),
    }
    for j, p in enumerate((BANK_N, FUNDUS, BANK_P)):
        d = prof[p]
        ok = d["count"] >= MIN_SAMPLES
        q = d[measure]
        a0 = ax[0, j]
        a0.fill_between(mid[ok], q[ok, 0], q[ok, 2], color="tab:blue", alpha=0.3)
        a0.plot(mid[ok], q[ok, 1], color="tab:blue", label="cortex on rays")
        for k, (lo, med, up) in refs.items():
            cc, ls = styles[k]
            a0.axhline(med, color=cc, ls=ls, label=f"{k} median")
            if k == "noise":
                a0.axhspan(lo, up, color="0.5", alpha=0.15, label="noise quartiles")
        ql = lam[p][measure]
        a0.plot(mid[ok], ql[ok, 1], color="tab:olive", label="laminar ref. on rays")
        a0.set(title=f"{PART_NAMES[p]}", ylim=(0, 1))
        a2 = a0.twinx()
        a2.plot(mid, d["count"], color="0.7", lw=0.8)
        a2.set_yticks([])
        a1 = ax[1, j]
        g = d[angle]
        a1.fill_between(mid[ok], g[ok, 0], g[ok, 2], color="tab:red", alpha=0.25)
        a1.plot(mid[ok], g[ok, 1], color="tab:red", label="median, all samples")
        h = hi[p]
        okh = np.isfinite(h) & ok
        a1.plot(mid[okh], h[okh], color="k", label=f"median, {measure} > noise p90")
        gl = lam[p][angle]
        a1.plot(mid[ok], gl[ok, 1], color="tab:olive", label="laminar ref., median")
        a1.axhline(60.0, color="0.5", ls=":", label="median of random axes")
        a1.set(xlabel="s along the straight ray (mm)", ylim=(0, 90))
    ax[0, 0].set_ylabel(measure)
    ax[1, 0].set_ylabel(f"angle of {angle.split('_')[1]} to the ray (deg)")
    ax[0, 2].legend(fontsize=7, loc="upper right")
    ax[1, 2].legend(fontsize=7, loc="lower right")
    fig.suptitle(
        title + f" (bins of {BIN_MM} mm, shown with >= {MIN_SAMPLES} samples; "
        "grey line: sample count)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def view_reference(path: Path, tex: dict[str, Texture]) -> None:
    """The histograms of f, c and l3 at the valid voxels of each region."""
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    cols = {
        "cortex": "tab:blue",
        "wm": "tab:purple",
        "outside": "k",
        "noise": "0.5",
        "laminar": "tab:olive",
    }
    for k, t in tex.items():
        f = t.f[np.isfinite(t.f)]
        c = t.c[np.isfinite(t.c)]
        l3 = t.values[2][np.isfinite(t.values[2])]
        kw = dict(histtype="step", density=True, color=cols[k], lw=1.5)
        ax[0].hist(f, bins=50, range=(0, 1), label=f"{k} ({len(f)} voxels)", **kw)
        ax[1].hist(c, bins=50, range=(0, 1), label=k, **kw)
        pos = l3[l3 > 0]
        ax[2].hist(np.log10(pos), bins=80, label=k, **kw)
    ax[0].set(xlabel="f = (l2 - l3)/(l2 + l3)", ylabel="density", title="fibre measure")
    ax[1].set(xlabel="c = (l1 - l2)/(l1 + l2)", title="laminar coherence")
    ax[2].set(xlabel="log10 l3 (whitened)", title="smallest eigenvalue")
    ax[0].legend(fontsize=8)
    fig.suptitle(
        "valid voxels of the box. noise: the cortex level + thermal noise; "
        f"laminar: + laminae parallel to the WM ({LAMINA_MM} mm); cortex mask"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _plane(b: BB.Box, vi: int, e1: F64, e2: F64, h: float) -> tuple[F64, F64, F64]:
    """A plane through the vertex root spanned by e1 (horizontal) and e2
    (vertical) in scanner space, as box voxel coordinates (H, W, 3) at
    step h mm: e1 from -4 to 4 mm, e2 from -2 to 6 mm."""
    a = np.arange(-4.0, 4.0 + 1e-9, h)
    c = np.arange(-2.0, 6.0 + 1e-9, h)
    A, C = np.meshgrid(a, c, indexing="xy")
    x = b.roots[vi].astype(np.float64) + A[..., None] * e1 + C[..., None] * e2
    return b.to_box(x), a, c


def view_slices(path: Path, b: BB.Box, J: F64, u: F64, vi: int, top: F64) -> None:
    """Two planes through the vertex: echo 2, the cortex edge, the vertex
    ray, and the fibre direction projected on the plane as short lines
    coloured by f (in the valid cortex)."""
    plt = _plt()
    from matplotlib.collections import LineCollection

    nv = b.normals[vi].astype(np.float64)
    t = np.cross(nv, u)
    fig, ax = plt.subplots(1, 2, figsize=(14, 7.5))
    for a, e1, name in (
        (ax[0], u, "across the sulcus (u, n_v)"),
        (ax[1], t, "along the fundus (t, n_v)"),
    ):
        p, ga, gc = _plane(b, vi, e1, nv, 0.08)
        img = map_coordinates(
            b.echoes[1].astype(np.float64), p.reshape(-1, 3).T, order=1, mode="nearest"
        ).reshape(p.shape[:2])
        lab = map_coordinates(
            b.labels, p.reshape(-1, 3).T, order=0, mode="nearest"
        ).reshape(p.shape[:2])
        ext = (ga[0], ga[-1], gc[0], gc[-1])
        a.imshow(img, origin="lower", extent=ext, cmap="gray")
        a.contour(
            ga, gc, (lab == b.cortex).astype(float), [0.5], colors="y", linewidths=0.6
        )
        a.contour(ga, gc, (lab == 0).astype(float), [0.5], colors="c", linewidths=0.6)
        # lines every 2 voxels
        q, qa, qc = _plane(b, vi, e1, nv, 0.32)
        tx = texture(sample_tensor(J, q.reshape(-1, 3), order=1))
        v = to_scanner(tx.fibre, b.affine, b.voxel_mm)
        A, C = np.meshgrid(qa, qc, indexing="xy")
        x0, y0 = A.ravel(), C.ravel()
        dx, dy = v @ e1, v @ nv
        ok = np.isfinite(tx.f)
        half = 0.14
        segs = np.stack(
            [
                np.stack([x0 - half * dx, y0 - half * dy], -1),
                np.stack([x0 + half * dx, y0 + half * dy], -1),
            ],
            1,
        )[ok]
        lc = LineCollection(list(segs), cmap="plasma", linewidths=1.6)
        lc.set_array(tx.f[ok])
        lc.set_clim(0, 1)
        a.add_collection(lc)
        end = top[vi] if np.isfinite(top[vi]) else R.CAP_OUT
        a.plot([0, 0], [0, min(end, 6.0)], "w--", lw=1)
        a.plot(0, 0, "r*", ms=10)
        a.set(
            xlim=ext[:2], ylim=ext[2:], xlabel="mm", ylabel="along n_v (mm)", title=name
        )
        fig.colorbar(
            lc,
            ax=a,
            fraction=0.04,
            label="f (line: fibre direction projected; length = in-plane part)",
        )
    fig.suptitle(
        f"vertex {b.vertex}, echo 2; yellow: cortex edge, cyan: outside edge, "
        "white dashed: vertex ray to its first outside"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------- command line


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--box", type=Path, required=True, help="bent_box's box.npz")
    p.add_argument("--noise", type=Path, required=True, help="orient-noise/1 file")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--sigma", type=float, default=O.SIGMA, help="level sigma, mm")
    p.add_argument(
        "--grow",
        action="store_true",
        help="read the echoes and dseg of the box's sidecar (box.json next to "
        "the box) on the box grown by orient's reach, so that the whole box "
        "is valid",
    )
    p.add_argument(
        "--outlier-k",
        type=float,
        default=None,
        help="remove the voxels dark in every echo by more than k robust SDs "
        "(and OUTLIER_DILATE voxels round them) from the tensor's mask",
    )
    a = p.parse_args(argv)
    t0 = time.perf_counter()
    b = BB.load(a.box)
    noise = read_noise(a.noise)
    if len(noise.sigma) != b.echoes.shape[0]:
        raise ValueError(
            f"{a.noise.name}: {len(noise.sigma)} echoes, the box has "
            f"{b.echoes.shape[0]}"
        )
    grown = None
    if a.grow:
        side = json.loads((a.box.parent / BB.SIDECAR).read_text())
        grown = read_grown(b, side, O.orient_reach(b.voxel_mm, a.sigma))
    F = fields(b, noise, a.sigma, grown, a.outlier_k)
    del grown
    tex = {k: texture(F[k]) for k in REGIONS}
    t1 = time.perf_counter()

    vi = int(np.flatnonzero(b.rays.roots.vertex == b.vertex)[0])
    part, u = split_banks(b.normals, vi, use=b.inner)
    r = b.rays
    top = first_outside(
        r.interval_start, r.interval_s0, r.interval_s1, r.interval_class
    )
    inner = np.flatnonzero(b.inner)
    rs = ray_samples(b, F["cortex"], inner, top)
    # the samples are at multiples of the step: edges half a step below them
    edges = np.arange(0.0, R.CAP_OUT + BIN_MM / 2, BIN_MM) - r.step / 2
    mid = (edges[:-1] + edges[1:]) / 2
    refs_f = {k: _q(tex[k].f) for k in ("wm", "outside", "noise")}
    refs_c = {k: _q(tex[k].c) for k in ("wm", "outside", "noise")}
    rl = ray_samples(b, F["laminar"], inner, top)
    p90_f = float(np.nanpercentile(tex["noise"].f, 90))
    p90_c = float(np.nanpercentile(tex["noise"].c, 90))
    prof: dict[int, dict[str, F64]] = {}
    lam: dict[int, dict[str, F64]] = {}
    hi_f: dict[int, F64] = {}
    hi_c: dict[int, F64] = {}
    kbin = np.digitize(rs.s, edges) - 1
    for q in (BANK_N, FUNDUS, BANK_P):
        sel = part[rs.column] == q
        prof[q] = by_depth(rs, sel, edges)
        lam[q] = by_depth(rl, part[rl.column] == q, edges)
        hf, hc = [], []
        for i in range(len(mid)):
            m = sel & (kbin == i)
            mf = m & (rs.f > p90_f)
            mc = m & (rs.c > p90_c)
            hf.append(
                np.median(rs.angle_fibre[mf]) if mf.sum() >= MIN_SAMPLES else np.nan
            )
            hc.append(
                np.median(rs.angle_leading[mc]) if mc.sum() >= MIN_SAMPLES else np.nan
            )
        hi_f[q] = np.asarray(hf)
        hi_c[q] = np.asarray(hc)

    a.out.mkdir(parents=True, exist_ok=True)
    n_col = len(b.rays.sample_start) - 1
    tot = np.bincount(rs.column, minlength=n_col)
    fin = np.bincount(rs.column, weights=np.isfinite(rs.f), minlength=n_col)
    with np.errstate(invalid="ignore", divide="ignore"):
        cover = fin / tot
    view_split(a.out / "split_map.png", b, part, u, top, vi, cover)
    view_depth(
        a.out / "fibre_by_depth.png",
        prof,
        edges,
        "f",
        "angle_fibre",
        refs_f,
        hi_f,
        lam,
        "fibre measure f and the angle of the fibre direction (eigenvector of l3) "
        "to the straight ray",
    )
    view_depth(
        a.out / "laminar_by_depth.png",
        prof,
        edges,
        "c",
        "angle_leading",
        refs_c,
        hi_c,
        lam,
        "laminar coherence c and the angle of the leading eigenvector to the "
        "straight ray",
    )
    view_reference(a.out / "reference.png", tex)
    view_slices(a.out / "slices.png", b, F["cortex"], u, vi, top)
    t2 = time.perf_counter()

    tc = tex["cortex"]
    np.savez(
        a.out / "texture.npz",
        tensor=F["cortex"].astype(np.float32),
        values=tc.values.astype(np.float32),
        leading=tc.leading.astype(np.float32),
        fibre=tc.fibre.astype(np.float32),
        f=tc.f.astype(np.float32),
        c=tc.c.astype(np.float32),
        strength=tc.strength.astype(np.float32),
        valid=F["valid"],
        level=F["level"].astype(np.float32),
        part=part,
        u=u,
        first_outside=top,
        offset=b.offset,
        outliers=F["outliers"],
    )
    ctx = F["masks"]["cortex"]
    summary: dict[str, Any] = dict(
        box=str(a.box.resolve()),
        noise=str(a.noise.resolve()),
        sigma_mm=a.sigma,
        reach_voxels=list(F["reach"]),
        grown=F["grown"],
        outlier_k=a.outlier_k,
        outlier_dilate=OUTLIER_DILATE,
        outlier_voxels=int(F["outliers"].sum()),
        outlier_sd=None
        if F["outlier_sd"] is None
        else [_r(x) for x in F["outlier_sd"]],
        cortex_voxels=int(ctx.sum()),
        cortex_valid_voxels=int((ctx & F["valid"]).sum()),
        strength="l2 + l3 (whitened)",
        sampling="tensor interpolated trilinearly at the straight samples, "
        "0 <= s < first outside (CAP_OUT if censored), inner columns",
        split=dict(
            method="u: largest spread of the inner normals, perpendicular to "
            f"n_v; fundus if |n . u| < sin {FUNDUS_DEG} deg",
            u=[round(float(x), 4) for x in u],
            columns={PART_NAMES[q]: int((b.inner & (part == q)).sum()) for q in prof},
            first_outside_median_mm={
                PART_NAMES[q]: _r(float(np.nanmedian(top[b.inner & (part == q)])))
                for q in prof
            },
            censored={
                PART_NAMES[q]: int((b.inner & (part == q) & ~np.isfinite(top)).sum())
                for q in prof
            },
        ),
        detrended_sd=[_r(x) for x in F["detrended_sd"]],
        reference_quartiles=dict(
            f={k: [_r(x) for x in _q(tex[k].f)] for k in tex},
            c={k: [_r(x) for x in _q(tex[k].c)] for k in tex},
            l3={k: [_r(x) for x in _q(tex[k].values[2])] for k in tex},
            noise_p90=dict(f=_r(p90_f), c=_r(p90_c)),
        ),
        depth_bins_mm=[round(float(x), 3) for x in mid],
        by_part={
            PART_NAMES[q]: dict(
                count=prof[q]["count"].astype(int).tolist(),
                f_median=[_r(x) for x in prof[q]["f"][:, 1]],
                c_median=[_r(x) for x in prof[q]["c"][:, 1]],
                angle_fibre_median=[_r(x) for x in prof[q]["angle_fibre"][:, 1]],
                angle_leading_median=[_r(x) for x in prof[q]["angle_leading"][:, 1]],
                laminar_ref_f_median=[_r(x) for x in lam[q]["f"][:, 1]],
                laminar_ref_c_median=[_r(x) for x in lam[q]["c"][:, 1]],
            )
            for q in prof
        },
        seconds=dict(fields=round(t1 - t0, 1), views=round(t2 - t1, 1)),
    )
    (a.out / "texture.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(
        f"cortex {int(ctx.sum())} voxels ({summary['cortex_valid_voxels']} valid); "
        f"{len(rs.s)} ray samples; fields {t1 - t0:.1f} s, views {t2 - t1:.1f} s"
    )


if __name__ == "__main__":
    main()
