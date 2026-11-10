"""The bent paths, the soft tops, the lengths and costs 1-7 of the
bent-columns trial (docs/plans/bent-columns-trial.md, "The model" and
phase 3), in torch. Every cost is differentiable with respect to the path
parameters. Cost 7 is laminar alignment (an optional arm), not radial
texture. closest_outside is a view, not a cost.

Units: mm. n columns, NS samples along each path, K spline coefficients.

The path
--------
Column c has a root r_c, a unit normal n_c and a tangent frame (e1_c, e2_c)
(frame). Only the outward part bends:

    x_c(s) = r_c + s n_c + d_c(s),  s_k = k STEP,  k = 0 .. NS - 1,

with STEP = rays.STEP and s up to S_MAX = rays.CAP_OUT. The shear is
d_c(s) = a_c(s) e1_c + b_c(s) e2_c. a and b are cubic B-splines in s with
uniform knots KNOT_MM apart on [0, S_MAX], with the value d_c(0) = 0 fixed.
The slope at the root is free, so a path can bend at the root (basis). The
parameters are the coefficients (n, 2, K): [:, 0] for a, [:, 1] for b. The
coefficients of a column with moving = False are multiplied by 0, so a
fixed (halo) column stays straight and gets a zero gradient. Zero
coefficients give r + s n with the arithmetic of the ray sampler
(rays._samples), so the positions are the straight samples exactly.

Because (n, e1, e2) is orthonormal: |x'| = sqrt(1 + a'^2 + b'^2) and
|x''|^2 = a''^2 + b''^2. The unit tangent is t = x' / |x'|, with
x' = n + a' e1 + b' e2. This takes |n| = 1; the box's float32 normals are
unit to float32 rounding.

The soft maps
-------------
Three boolean volumes give three soft maps: outside medium (O), WM (W, the
filled labels) and cortex (C). Each is the volume smoothed by a Gaussian of
SIGMA_VOX = 1 voxel on each axis (0.16 mm at 0.16 mm voxels), truncated at
TRUNCATE = 4 sigma, edges by the nearest voxel (scipy gaussian_filter,
mode "nearest"), computed once in numpy. The path is sampled trilinearly
(bent_sample.trilinear) in box voxels, mapped from mm by the 4 x 4 affine
mm_to_vox. A point outside the volume takes the value of the nearest edge
voxel (Paths.leaves tells which columns leave it).

The soft top, mask, length
--------------------------
m_c(s) = 1 - max_{s' <= s} O(x_c(s')): the running maximum (running_max,
with a gradient to the sample that holds the maximum). O is clipped to
[0, 1] first. The mask does not increase along s. A column is censored if
its mask is still >= 0.5 at S_MAX. The length is the trapezoid sum
L_c = sum_k w_k m_c(s_k) |x_c'(s_k)|, w_k = STEP, and STEP / 2 at the two
ends. For a straight path across a flat top, O is symmetric about the top,
so L is the distance to the top (the test checks it within one STEP).

The warp levels
---------------
A fixed warp (flattening.Warp: s_lo, tau (K_w,), r (n, K_w)) puts level j
of column c at s = s_lo + tau_j + r[c, j]. These positions are constants
(Levels). A value along the path is read there by linear interpolation
between the two samples round it. A level with s < 0 (the inward part,
not in the path) or s > S_MAX does not count (Levels.valid = 0).

The costs
---------
1. cost_agreement: the features at equal warped depth, between the two
   columns of each mesh edge, weighted by m_c m_c' at the level.
2. cost_length: sum of L over the columns that are not censored.
3. cost_length_agreement: form A (edges) or form B (umbrella Laplacian).
4. cost_bending and cost_shear: bending stiffness, and the difference of
   neighbours' shear at equal s.
5. cost_crossing: the barrier and the crowding of each face at equal warped
   depth.
6. cost_return: a path that enters WM again after it left it.
7. cost_alignment: the path follows the data normal v (the leading
   eigenvector of orient's tensor, across the laminae), weighted by the
   coherence c (align_field). An optional arm: the fit runs with weight 0
   too, so that the data normal stays an independent check.

The view
--------
closest_outside: the smallest distance from a path to the outside medium.
It separates a column that almost reaches the top from a censored column
that is far from it. It is a diagnostic, not a cost.

Every choice that the plan leaves open is stated in the docstring of its
function. total adds the weighted costs for the fit.

Device rules: no sparse tensors and no index_add_ or scatter_add_. Every
gather by an index goes through Gather, whose backward is a gather of a
padded inverse table. The running maximum is a log-step scan, and its
backward uses cumsum and searchsorted.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from bent_sample import inside, trilinear
from scipy.interpolate import BSpline
from scipy.ndimage import distance_transform_edt, gaussian_filter

from baillarger import flattening
from baillarger import orient as O
from baillarger import rays as R
from baillarger.mesh_smoothing import boundary_vertices, edges

type Arr = npt.ArrayLike
type I64 = npt.NDArray[np.int64]
type F64 = npt.NDArray[np.float64]

#: the sample step along a path and the outward cap, mm (the ray sampler's)
STEP = R.STEP
S_MAX = R.CAP_OUT
#: the number of samples s_k = k STEP in [0, S_MAX]
NS = int(round(S_MAX / STEP)) + 1
#: the spacing of the spline knots, mm
KNOT_MM = 0.5
#: the number of free spline coefficients per component: the
#: S_MAX / KNOT_MM + 3 cubic B-splines on [0, S_MAX], less one for d(0) = 0
K = int(round(S_MAX / KNOT_MM)) + 3 - 1
#: the smoothing of the soft maps, voxels, and its truncation, in sigmas
SIGMA_VOX = 1.0
TRUNCATE = 4.0
#: the trend of the features along a path, mm (flattening's)
TREND_MM = flattening.TREND_MM
#: the feature echoes (0-based) that from_box takes from a box's echoes
FEATURE_ECHOES = flattening.FEATURE_ECHOES
#: the smallest weight sum of the trend (as flattening's 1e-3)
TREND_DEN = 1e-3
#: the rows of a backward gather per chunk: the product with the row size
#: and the table width is at most this many elements
CHUNK_ELEMS = 1 << 24
#: the order of the six components of a symmetric 3 x 3 field (orient's)
PAIRS = O.PAIRS
#: the terms of total, in order
TERMS = (
    "agreement",
    "length",
    "length_agreement",
    "bending",
    "shear",
    "barrier",
    "crowding",
    "return",
    "alignment",
)


# ---------------------------------------------------------------- tools


def _table(src: I64, dst: I64, n: int, pad: int) -> I64:
    """(n, w): for each value 0..n-1 of src, the dst values of its entries,
    padded with pad (as mesh_torch._table). w >= 1."""
    o = np.argsort(src, kind="stable")
    src, dst = src[o], dst[o]
    cnt = np.bincount(src, minlength=n)
    start = np.r_[0, np.cumsum(cnt)[:-1]]
    rank = np.arange(len(src)) - start[src]
    w = max(int(cnt.max()) if n and len(cnt) else 0, 1)
    out = np.full((n, w), pad, np.int64)
    out[src, rank] = dst
    return out


class _Take(torch.autograd.Function):
    """x[index] with a backward that gathers the inverse table (no
    scatter, no atomic add)."""

    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, index: torch.Tensor, inverse: torch.Tensor
    ) -> torch.Tensor:
        ctx.save_for_backward(inverse)
        return x[index]

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, g: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (inv,) = ctx.saved_tensors
        g = g.contiguous()
        gp = torch.cat([g, g.new_zeros((1, *g.shape[1:]))])
        row = max(int(np.prod(g.shape[1:])), 1)
        per = max(1, CHUNK_ELEMS // (row * inv.shape[1]))
        out = g.new_empty((len(inv), *g.shape[1:]))
        for i in range(0, len(inv), per):
            out[i : i + per] = gp[inv[i : i + per]].sum(1)
        return out, None, None


class Gather:
    """x[index] for x with `rows` rows, index of any shape: the result has
    the shape index.shape + x.shape[1:]. The inverse table is built once,
    in numpy. pad: the index of a padding row (a constant row of x, such as
    a row of zeros). Its entries are left out of the inverse table, so that
    row gets no gradient and the table stays narrow."""

    def __init__(
        self,
        index: Arr,
        rows: int,
        device: torch.device | str,
        pad: int | None = None,
    ) -> None:
        idx = np.asarray(index, np.int64)
        flat = idx.ravel()
        if len(flat) and (flat.min() < 0 or flat.max() >= rows):
            raise ValueError("index out of range")
        self.shape = idx.shape
        self.rows = rows
        self.index = torch.as_tensor(flat, device=device)
        pos = np.arange(len(flat), dtype=np.int64)
        keep = flat != pad if pad is not None else np.ones(len(flat), bool)
        inv = _table(flat[keep], pos[keep], rows, len(flat))
        self.inverse = torch.as_tensor(inv, device=device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != self.rows:
            raise ValueError(f"x has {x.shape[0]} rows, not {self.rows}")
        out: torch.Tensor = _Take.apply(x, self.index, self.inverse)
        return out.reshape(*self.shape, *x.shape[1:])


def _scan_max(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The running maximum along the last axis and the index of the sample
    that holds it (the last one of a tie): a log-step scan."""
    n = x.shape[-1]
    v = x
    i = torch.arange(n, device=x.device).expand(x.shape).contiguous()
    sh = 1
    while sh < n:
        pv, pi = v[..., :-sh], i[..., :-sh]
        cv, ci = v[..., sh:], i[..., sh:]
        prev = pv > cv
        v = torch.cat([v[..., :sh], torch.where(prev, pv, cv)], -1)
        i = torch.cat([i[..., :sh], torch.where(prev, pi, ci)], -1)
        sh *= 2
    return v, i


class _RunningMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        v, i = _scan_max(x)
        ctx.save_for_backward(i)
        return v

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, g: torch.Tensor) -> torch.Tensor:
        # The index i_k does not decrease along k. So the gradient of x_j is
        # the sum of g_k over the k with i_k = j: C(j) - C(j - 1), with C(j)
        # the sum of g_k over the k with i_k <= j.
        (i,) = ctx.saved_tensors
        lead = g.shape[:-1]
        n = g.shape[-1]
        g2 = g.reshape(-1, n)
        i2 = i.reshape(-1, n).contiguous()
        cs = torch.cat([g2.new_zeros((len(g2), 1)), torch.cumsum(g2, -1)], -1)
        j = torch.arange(n, device=g.device).expand(len(g2), n).contiguous()
        cnt = torch.searchsorted(i2, j, right=True)
        c = torch.gather(cs, 1, cnt)
        out = c - torch.cat([c.new_zeros((len(c), 1)), c[:, :-1]], -1)
        return out.reshape(*lead, n)


def running_max(x: torch.Tensor) -> torch.Tensor:
    """max_{k' <= k} x_k' along the last axis. The gradient goes to the
    sample that holds the maximum (the last one of a tie)."""
    out: torch.Tensor = _RunningMax.apply(x)
    return out


def running_min(x: torch.Tensor) -> torch.Tensor:
    """min_{k' <= k} x_k' along the last axis (running_max of -x)."""
    return -running_max(-x)


# ---------------------------------------------------------------- geometry


def frame(normals: Arr) -> tuple[F64, F64]:
    """A tangent frame (e1, e2), each (n, 3), perpendicular to each unit
    normal, with e2 = n x e1. e1 is n x a, normalized, with a the axis of
    the smallest |n| component (the first of a tie). It is deterministic.
    It can jump between neighbours where that axis changes; no cost depends
    on the frame (cost 4 compares the 3D shear)."""
    n = np.asarray(normals, np.float64)
    ax = np.eye(3)[np.argmin(np.abs(n), axis=1)]
    e1 = np.cross(n, ax)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(n, e1)
    e2 /= np.linalg.norm(e2, axis=1, keepdims=True)
    return e1, e2


def basis() -> tuple[F64, F64, F64]:
    """The basis (NS, K) of the shear on the samples s_k, and its first and
    second derivatives in s.

    The cubic B-splines B_0 .. B_{M-1} (M = S_MAX / KNOT_MM + 3) have the
    uniform knots t_i = (i - 3) KNOT_MM. At s = 0 only B_0, B_1, B_2 are
    nonzero, with the values 1/6, 2/3, 1/6. The constraint d(0) = 0 removes
    B_1's coefficient: c_1 = -(c_0 + c_2) / 4. The K = M - 1 free
    coefficients are c_0, c_2, ..., c_{M-1}. Row 0 of the value basis is
    then set to exactly 0 (it is 0 to rounding)."""
    m = K + 1
    t = KNOT_MM * (np.arange(m + 4) - 3.0)
    s = np.arange(NS) * STEP
    spl = BSpline(t, np.eye(m), 3, extrapolate=True)
    z = np.zeros((m, K))
    z[0, 0] = 1.0
    z[1, 0] = z[1, 1] = -0.25
    z[2:, 1:] = np.eye(K - 1)
    out = [
        np.asarray(spl.derivative(d)(s) if d else spl(s), np.float64) @ z
        for d in range(3)
    ]
    out[0][0] = 0.0
    return out[0], out[1], out[2]


def _gauss_matrix(sigma: float, n: int) -> F64:
    """(n, n): G with (G x)_k = sum_k' g(k - k') x_k', g the kernel of
    scipy's gaussian_filter1d (truncate 4), zeros beyond the ends (mode
    "constant"). A matrix in place of a convolution: no im2col copy."""
    r = int(4.0 * sigma + 0.5)
    x = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-0.5 * (x / sigma) ** 2)
    g /= g.sum()
    k = np.arange(n)
    off = k[:, None] - k[None, :]
    out: F64 = np.where(np.abs(off) <= r, g[np.clip(off + r, 0, 2 * r)], 0.0)
    return out


def _soft(mask: Arr) -> F64:
    out: F64 = gaussian_filter(
        np.asarray(mask, np.float64), SIGMA_VOX, mode="nearest", truncate=TRUNCATE
    )
    return out


def _face_neighbours(faces: I64) -> I64:
    """(F, w): the faces that share an edge with each face, padded with F."""
    nf = len(faces)
    if nf == 0:
        return np.zeros((0, 1), np.int64)
    e = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), 1
    )
    nv = int(faces.max()) + 1
    key = e[:, 0] * nv + e[:, 1]
    fid = np.tile(np.arange(nf), 3)
    o = np.lexsort((fid, key))
    key, fid = key[o], fid[o]
    src: list[I64] = []
    dst: list[I64] = []
    # every pair of faces on one edge (an edge has 2 faces on a manifold)
    start = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    stop = np.r_[start[1:], len(key)]
    for a, b in zip(start, stop, strict=True):
        if b - a > 1:
            f = fid[a:b]
            p, q = np.meshgrid(f, f, indexing="ij")
            keep = p != q
            src.append(p[keep])
            dst.append(q[keep])
    if not src:
        return np.full((nf, 1), nf, np.int64)
    return _table(np.concatenate(src), np.concatenate(dst), nf, nf)


def _to_scanner(v: F64, affine: F64, voxel_mm: Arr) -> F64:
    """Unit directions (..., 3) in the frame of the voxel axes to unit
    directions in scanner space: D v / |D v|, D = affine[:3, :3]
    diag(1 / voxel_mm) (as bent_texture.to_scanner)."""
    D = affine[:3, :3] / np.asarray(voxel_mm, np.float64)
    w = v @ D.T
    out: F64 = w / np.linalg.norm(w, axis=-1, keepdims=True)
    return out


def align_field(
    leading: Arr, c: Arr, affine: Arr, voxel_mm: Arr
) -> npt.NDArray[np.float32]:
    """The laminar field of cost 7: A = c (I - v v^T), (6, X, Y, Z) float32,
    components in the order of PAIRS (xx, xy, xz, yy, yz, zz), scanner
    frame.

    leading (X, Y, Z, 3): the unit leading eigenvector in the frame of the
    voxel axes (bent_texture's "leading"). It goes to scanner space by
    the map of bent_texture.to_scanner, affine[:3, :3] diag(1 / voxel_mm),
    then normalized (written here, so this module does not import
    bent_texture; a test checks the two agree). c (X, Y, Z): the coherence
    (l1 - l2) / (l1 + l2). Where c or v is not finite (no valid tensor:
    outside the cortex, outliers), A = 0. A does not depend on the sign of
    v. t^T A t = c sin^2 of the angle between a unit t and v."""
    lead = np.asarray(leading, np.float64)
    cc = np.asarray(c, np.float64)
    if lead.shape != (*cc.shape, 3):
        raise ValueError(f"leading {lead.shape} and c {cc.shape} do not match")
    ok = np.isfinite(cc) & np.all(np.isfinite(lead), -1)
    v = _to_scanner(
        np.where(ok[..., None], lead, [0.0, 0.0, 1.0]),
        np.asarray(affine, np.float64),
        voxel_mm,
    )
    w = np.where(ok, cc, 0.0)
    eye = np.eye(3)
    out = np.stack([w * (eye[i, j] - v[..., i] * v[..., j]) for i, j in PAIRS])
    return out.astype(np.float32)


def _voxel_mm(mm_to_vox: F64) -> F64:
    """(3,) the length in mm of one voxel step along each voxel axis: the
    norms of the columns of the inverse of the linear part."""
    out: F64 = np.linalg.norm(np.linalg.inv(mm_to_vox[:3, :3]), axis=0)
    return out


# ---------------------------------------------------------------- the inputs


@dataclass
class Inputs:
    """The fixed inputs of the costs, on one device, in one dtype."""

    device: torch.device
    dtype: torch.dtype
    n: int
    K: int
    #: (NS,) s_k, mm, and the trapezoid weights of the sums along s
    s: torch.Tensor
    ds: torch.Tensor
    #: (NS, K) the basis and its derivatives
    B0: torch.Tensor
    B1: torch.Tensor
    B2: torch.Tensor
    #: (n, 3) mm: roots, normals and the frame
    root: torch.Tensor
    normal: torch.Tensor
    e1: torch.Tensor
    e2: torch.Tensor
    #: (n,) 1 for a column that moves, 0 for a fixed one
    moving: torch.Tensor
    #: (n, 3) the roots in voxels; (3, 3) the linear part of mm_to_vox
    root_vox: torch.Tensor
    lin: torch.Tensor
    #: (3 + E, X, Y, Z): O, W, C and the feature echoes (nan -> 1)
    volumes: torch.Tensor
    #: (E,) the fixed feature scales
    scales: torch.Tensor
    #: (NS, NS) the trend's Gaussian along s, as a matrix
    trend: torch.Tensor
    #: the edges (m, 2) and the gathers of their two ends (n rows)
    edges: I64
    edge0: Gather
    edge1: Gather
    #: the faces (F, 3) and the gathers of their corners (n rows)
    faces: I64
    corner: tuple[Gather, Gather, Gather]
    #: (F, 3) r1 - r0 and r2 - r0, mm; (F, 3) the unit face normal, oriented
    #: with the root normals; (F,) the root triangle's area, mm^2
    dr1: torch.Tensor
    dr2: torch.Tensor
    nu: torch.Tensor
    a0: torch.Tensor
    #: the faces that share an edge with each face (F + 1 rows, padded),
    #: and their number (F,)
    face_nbr: Gather
    face_deg: torch.Tensor
    #: the neighbours of each column (n + 1 rows, padded), their number (n,),
    #: and 1 for an interior column (not on the mesh's boundary, at least
    #: one neighbour)
    nbr: Gather
    deg: torch.Tensor
    interior: torch.Tensor
    #: (6, X, Y, Z) the laminar field of cost 7 (align_field), or None
    align: torch.Tensor | None = None
    #: (X, Y, Z) mm: the distance from each voxel centre to the centre of
    #: the nearest outside voxel (0 in the outside), for closest_outside
    dist: torch.Tensor | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        x, y, z = self.volumes.shape[1:]
        return int(x), int(y), int(z)

    @classmethod
    def from_arrays(
        cls,
        roots: Arr,
        normals: Arr,
        faces: Arr,
        mm_to_vox: Arr,
        outside: Arr,
        wm: Arr,
        cortex: Arr,
        echoes: Arr,
        scales: Arr,
        *,
        mesh_edges: Arr | None = None,
        moving: Arr | None = None,
        align: Arr | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "Inputs":
        """From plain arrays: roots (n, 3) mm, unit normals (n, 3), faces
        (F, 3) column indices, mm_to_vox (4, 4), boolean volumes outside,
        wm and cortex (X, Y, Z), the feature echoes (E, X, Y, Z) on the same
        voxels, and their fixed scales (E,). mesh_edges default: the edges of
        the faces. moving default: all columns move. align: the laminar
        field (6, X, Y, Z) of align_field on the same voxels, or None (no
        cost 7). The distance map of the outside uses the voxel size of
        mm_to_vox."""
        dev = torch.device(device)
        r = np.asarray(roots, np.float64)
        nrm = np.asarray(normals, np.float64)
        n = len(r)
        f = np.asarray(faces, np.int64).reshape(-1, 3)
        if mesh_edges is None:
            e = edges(f).astype(np.int64) if len(f) else np.zeros((0, 2), np.int64)
        else:
            e = np.asarray(mesh_edges, np.int64).reshape(-1, 2)
        mv = np.ones(n, bool) if moving is None else np.asarray(moving, bool)
        a = np.asarray(mm_to_vox, np.float64)
        echo = np.nan_to_num(np.asarray(echoes, np.float64), nan=1.0)
        out_b = np.asarray(outside, bool)
        vols = np.concatenate(
            [np.stack([_soft(out_b), _soft(wm), _soft(cortex)]), echo]
        )
        if align is not None:
            al = np.asarray(align, np.float64)
            if al.shape != (6, *out_b.shape):
                raise ValueError(f"align {al.shape}: not (6, *{out_b.shape})")
        dist = (
            distance_transform_edt(~out_b, sampling=_voxel_mm(a))
            if out_b.any()
            else np.full(out_b.shape, np.inf)
        )
        e1, e2 = frame(nrm)
        b0, b1, b2 = basis()
        s = np.arange(NS) * STEP
        ds = np.full(NS, STEP)
        ds[0] = ds[-1] = STEP / 2

        # the faces: root edges, the oriented normal and the area
        if len(f):
            d1 = r[f[:, 1]] - r[f[:, 0]]
            d2 = r[f[:, 2]] - r[f[:, 0]]
            cr = np.cross(d1, d2)
            area2 = np.linalg.norm(cr, axis=1)
            if np.any(area2 <= 0):
                raise ValueError("a face with zero area at the roots")
            nu = cr / area2[:, None]
            sign = np.sign((nu * nrm[f].sum(1)).sum(1))
            nu *= np.where(sign == 0, 1.0, sign)[:, None]
            a0 = area2 / 2
        else:
            d1 = d2 = nu = np.zeros((0, 3))
            a0 = np.zeros(0)
        fn = _face_neighbours(f)
        fdeg = (fn < len(f)).sum(1)

        # the neighbours of each column, for form B
        src = np.r_[e[:, 0], e[:, 1]]
        dst = np.r_[e[:, 1], e[:, 0]]
        nb = _table(src, dst, n, n)
        deg = np.bincount(src, minlength=n)
        interior = (deg > 0) & ~boundary_vertices(f, n) if len(f) else np.zeros(n, bool)

        def T(x: Arr) -> torch.Tensor:
            return torch.as_tensor(np.asarray(x, np.float64)).to(dtype).to(dev)

        return cls(
            device=dev,
            dtype=dtype,
            n=n,
            K=K,
            s=T(s),
            ds=T(ds),
            B0=T(b0),
            B1=T(b1),
            B2=T(b2),
            root=T(r),
            normal=T(nrm),
            e1=T(e1),
            e2=T(e2),
            moving=T(mv.astype(np.float64)),
            root_vox=T(r @ a[:3, :3].T + a[:3, 3]),
            lin=T(a[:3, :3]),
            volumes=T(vols),
            scales=T(np.asarray(scales, np.float64)),
            trend=T(_gauss_matrix(TREND_MM / STEP, NS)),
            edges=e,
            edge0=Gather(e[:, 0], n, dev),
            edge1=Gather(e[:, 1], n, dev),
            faces=f,
            corner=(
                Gather(f[:, 0], n, dev),
                Gather(f[:, 1], n, dev),
                Gather(f[:, 2], n, dev),
            ),
            dr1=T(d1),
            dr2=T(d2),
            nu=T(nu),
            a0=T(a0),
            face_nbr=Gather(fn, len(f) + 1, dev, pad=len(f)),
            face_deg=T(fdeg),
            nbr=Gather(nb, n + 1, dev, pad=n),
            deg=T(np.maximum(deg, 1)),
            interior=T(interior.astype(np.float64)),
            align=None if align is None else T(al),
            dist=T(dist),
        )

    @classmethod
    def from_box(
        cls,
        box: Any,
        device: torch.device | str,
        scales: Arr,
        dtype: torch.dtype = torch.float32,
        align: Arr | None = None,
    ) -> "Inputs":
        """From a bent_box.Box: outside = labels == 0, WM = labels in the
        box's filled labels, cortex = labels == the box's cortex label. The
        echoes are the box's FEATURE_ECHOES. The inner columns move; the
        halo columns are fixed. align: the laminar field on the box grid
        (align_field of bent_texture's "leading" and "c", with the box's
        affine and voxel_mm), or None."""
        lab = np.asarray(box.labels)
        return cls.from_arrays(
            roots=np.asarray(box.roots, np.float64),
            normals=np.asarray(box.normals, np.float64),
            faces=box.faces,
            mm_to_vox=box.mm_to_box,
            outside=lab == 0,
            wm=np.isin(lab, np.asarray(box.filled)),
            cortex=lab == int(box.cortex),
            echoes=np.asarray(box.echoes)[list(FEATURE_ECHOES)],
            scales=scales,
            mesh_edges=box.edges,
            moving=box.inner,
            align=align,
            device=device,
            dtype=dtype,
        )


@dataclass
class Levels:
    """The fixed warp levels of the columns: level j of column c is at
    s = s_lo + tau_j + r[c, j]. valid (n, Kw) is 1 where that s is in
    [0, S_MAX]. A value is read by linear interpolation between the samples
    lo and lo + 1, with the fraction frac."""

    s: torch.Tensor
    valid: torch.Tensor
    frac: torch.Tensor
    lo: Gather
    hi: Gather

    @classmethod
    def make(cls, inputs: Inputs, s_lo: float, tau: Arr, r: Arr) -> "Levels":
        rr = np.asarray(r, np.float64)
        q = float(s_lo) + np.asarray(tau, np.float64)[None] + rr
        if q.shape[0] != inputs.n:
            raise ValueError(f"r has {q.shape[0]} rows, not {inputs.n}")
        valid = (q >= 0) & (q <= S_MAX)
        t = np.clip(q, 0, S_MAX) / STEP
        lo = np.clip(np.floor(t), 0, NS - 2).astype(np.int64)
        frac = t - lo
        row = np.arange(inputs.n, dtype=np.int64)[:, None] * NS
        rows = inputs.n * NS
        dev, dt = inputs.device, inputs.dtype
        return cls(
            s=torch.as_tensor(q).to(dt).to(dev),
            valid=torch.as_tensor(valid.astype(np.float64)).to(dt).to(dev),
            frac=torch.as_tensor(frac).to(dt).to(dev),
            lo=Gather(row + lo, rows, dev),
            hi=Gather(row + lo + 1, rows, dev),
        )

    @classmethod
    def from_warp(cls, inputs: Inputs, warp: Any) -> "Levels":
        """From a flattening.Warp (s_lo, tau, r) on the same columns."""
        return cls.make(inputs, warp.s_lo, warp.tau, warp.r)

    def read(self, values: torch.Tensor) -> torch.Tensor:
        """values (n, NS, C) -> (n, Kw, C) at the levels (not multiplied by
        valid)."""
        n, ns, c = values.shape
        v = values.reshape(n * ns, c)
        a, b = self.lo(v), self.hi(v)
        out: torch.Tensor = torch.lerp(a, b, self.frac[..., None])
        return out


# ---------------------------------------------------------------- the paths


@dataclass
class Paths:
    """The paths of a set of coefficients and the soft maps along them."""

    #: (NS,) s_k, mm
    s: torch.Tensor
    #: (n, NS, 3) s n + d, mm (the position less the root)
    y: torch.Tensor
    #: (n, NS, 3) the shear d, mm
    d: torch.Tensor
    #: (n, NS) |x'| and |x''|^2; (n, NS, 3) the unit tangent x' / |x'|
    speed: torch.Tensor
    curv2: torch.Tensor
    tangent: torch.Tensor
    #: (n, NS, 3) the positions in voxels
    vox: torch.Tensor
    #: (n, NS) O, W, C along the path; (E, n, NS) the feature echoes
    outside: torch.Tensor
    wm: torch.Tensor
    cortex: torch.Tensor
    echo: torch.Tensor
    #: (n, 3) the roots, mm
    root: torch.Tensor

    @property
    def x(self) -> torch.Tensor:
        """(n, NS, 3) the positions, mm."""
        return self.root[:, None] + self.y

    def leaves(self, shape: tuple[int, int, int]) -> torch.Tensor:
        """(n,) True for a column with a sample outside the volume."""
        return ~inside(shape, self.vox).all(-1)


def paths(params: torch.Tensor, inputs: Inputs) -> Paths:
    """The paths of the coefficients (n, 2, K) (see the module docstring)."""
    c = params * inputs.moving[:, None, None]
    a = c[:, 0] @ inputs.B0.T
    b = c[:, 1] @ inputs.B0.T
    a1 = c[:, 0] @ inputs.B1.T
    b1 = c[:, 1] @ inputs.B1.T
    a2 = c[:, 0] @ inputs.B2.T
    b2 = c[:, 1] @ inputs.B2.T
    d = a[..., None] * inputs.e1[:, None] + b[..., None] * inputs.e2[:, None]
    y = inputs.s[None, :, None] * inputs.normal[:, None] + d
    vox = inputs.root_vox[:, None] + y @ inputs.lin.T
    f = trilinear(inputs.volumes, vox)
    speed = torch.sqrt(1 + a1 * a1 + b1 * b1)
    dx = (
        inputs.normal[:, None]
        + a1[..., None] * inputs.e1[:, None]
        + b1[..., None] * inputs.e2[:, None]
    )
    return Paths(
        s=inputs.s,
        y=y,
        d=d,
        speed=speed,
        curv2=a2 * a2 + b2 * b2,
        tangent=dx / speed[..., None],
        vox=vox,
        outside=f[0],
        wm=f[1],
        cortex=f[2],
        echo=f[3:],
        root=inputs.root,
    )


@dataclass
class Top:
    #: (n, NS) the mask; (n,) the soft length, mm; (n,) bool, detached
    mask: torch.Tensor
    length: torch.Tensor
    censored: torch.Tensor


def soft_top(p: Paths, inputs: Inputs) -> Top:
    """The mask, the soft length and the censored flags (see the module
    docstring)."""
    m = 1 - running_max(p.outside.clamp(0, 1))
    length = (m * p.speed * inputs.ds).sum(-1)
    return Top(m, length, (m[:, -1] >= 0.5).detach())


# ---------------------------------------------------------------- cost 1


def _smooth_s(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Gaussian smoothing along the last axis by the matrix g
    (_gauss_matrix)."""
    out: torch.Tensor = x @ g.T
    return out


def features(p: Paths, mask: torch.Tensor, inputs: Inputs) -> torch.Tensor:
    """(E, n, NS): the flattening features along the paths, in torch.

    As flattening.features: log(max(echo, 1)), minus a normalized Gaussian
    trend of TREND_MM along s, divided by the fixed scales. Choices:
    - The trend's weight is the mask times the soft cortex C (flattening
      uses the binary cortex label). The mask keeps the samples above the
      top out of the trend.
    - The feature is multiplied by C: the soft form of flattening's zero
      outside the cortex.
    - The trend divides by max(weight sum, TREND_DEN) and is not set to 0
      where the sum is small (the factor C does that).
    - Only the outward samples s >= 0 enter the trend; flattening's
      features of the straight rays also use the inward samples, so the two
      differ within about TREND_MM of the root."""
    L = torch.log(p.echo.clamp(min=1.0))
    w = mask * p.cortex
    num = _smooth_s(w[None] * L, inputs.trend)
    den = _smooth_s(w, inputs.trend).clamp(min=TREND_DEN)
    return p.cortex[None] * (L - num / den[None]) / inputs.scales[:, None, None]


def cost_agreement(
    feat: torch.Tensor,
    mask: torch.Tensor,
    levels: Levels,
    inputs: Inputs,
    normalize: bool = False,
) -> torch.Tensor:
    """Cost 1: sum over the edges (c, c') and the levels j of
    w (f_c - f_c')^2, summed over the features, with w = m_c m_c' at the
    level (times valid on both). normalize: divide each edge's sum by the
    sum of its weights w over the levels (max 1e-12)."""
    E = feat.shape[0]
    v = torch.cat([feat.permute(1, 2, 0), mask[..., None]], -1)
    lv = levels.read(v)
    m = lv[..., E] * levels.valid
    f = lv[..., :E]
    f0, f1 = inputs.edge0(f), inputs.edge1(f)
    w = inputs.edge0(m) * inputs.edge1(m)
    per = (w * ((f0 - f1) ** 2).sum(-1)).sum(-1)
    if normalize:
        per = per / w.sum(-1).clamp(min=1e-12)
    return per.sum()


# ---------------------------------------------------------------- costs 2, 3


def cost_length(length: torch.Tensor, censored: torch.Tensor) -> torch.Tensor:
    """Cost 2: sum of L over the columns that are not censored. The censored
    set is fixed for a round (passed in)."""
    return torch.where(censored, torch.zeros_like(length), length).sum()


def cost_length_agreement(
    length: torch.Tensor,
    censored: torch.Tensor,
    free_length: torch.Tensor,
    inputs: Inputs,
    form: str = "A",
) -> torch.Tensor:
    """Cost 3. A censored column has the length free_length[c] (a separate
    parameter that enters only here) in place of L_c.

    Form A: sum over the edges of (L_c - L_c')^2.
    Form B: sum over the interior columns of (L_c - mean of the neighbours'
    L)^2 (umbrella Laplacian, equal weights). A column on the boundary of
    the box's mesh (an edge with one face) has neighbours on one side only,
    so its Laplacian of a linear L is not 0: it is left out of the sum. It
    still enters as a neighbour of interior columns. Halo columns that are
    not on the boundary are in the sum."""
    L = torch.where(censored, free_length, length)
    if form == "A":
        return ((inputs.edge0(L) - inputs.edge1(L)) ** 2).sum()
    if form == "B":
        Lp = torch.cat([L, L.new_zeros(1)])
        mean = inputs.nbr(Lp).sum(-1) / inputs.deg
        return (inputs.interior * (L - mean) ** 2).sum()
    raise ValueError(f"form {form!r}: not 'A' or 'B'")


# ---------------------------------------------------------------- cost 4


def cost_bending(p: Paths, mask: torch.Tensor, inputs: Inputs) -> torch.Tensor:
    """Cost 4, part 1: sum over the columns and samples of
    w_k m |x''|^2 (= |d''|^2). Weighted by the mask: a bend above the top
    costs nothing, as in the other terms."""
    return (mask * p.curv2 * inputs.ds).sum()


def cost_shear(p: Paths, mask: torch.Tensor, inputs: Inputs) -> torch.Tensor:
    """Cost 4, part 2: sum over the edges and samples of
    w_k m_c m_c' |d_c - d_c'|^2, with d the 3D shear (it does not depend on
    the frames)."""
    d0, d1 = inputs.edge0(p.d), inputs.edge1(p.d)
    w = inputs.edge0(mask) * inputs.edge1(mask)
    return (w * inputs.ds * ((d0 - d1) ** 2).sum(-1)).sum()


# ---------------------------------------------------------------- cost 5


def gate(m: torch.Tensor) -> torch.Tensor:
    """clamp(2 m - 1, 0, 1): 0 where m <= 0.5, 1 at m = 1, continuous."""
    return (2 * m - 1).clamp(0, 1)


def face_areas(
    p: Paths, mask: torch.Tensor, levels: Levels, inputs: Inputs
) -> tuple[torch.Tensor, torch.Tensor]:
    """(F, Kw) a / a0 of each face at each level, and its weight: the
    product of gate(m) of the three corners at the level, times valid. a is
    the signed area (cross(p1 - p0, p2 - p0) . nu) / 2 of the three points
    at equal warped depth."""
    lv = levels.read(torch.cat([p.y, mask[..., None]], -1))
    g = gate(lv[..., 3]) * levels.valid
    c0, c1, c2 = (cg(lv[..., :3]) for cg in inputs.corner)
    u = inputs.dr1[:, None] + c1 - c0
    v = inputs.dr2[:, None] + c2 - c0
    a = (torch.linalg.cross(u, v) * inputs.nu[:, None]).sum(-1) / 2
    w = inputs.corner[0](g) * inputs.corner[1](g) * inputs.corner[2](g)
    return a / inputs.a0[:, None], w


def cost_crossing(
    p: Paths,
    mask: torch.Tensor,
    levels: Levels,
    inputs: Inputs,
    extend: float | None = None,
    relative: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cost 5: (barrier, crowding), from face_areas (x = a / a0, weight w).

    Barrier: sum over faces and levels of w phi(x), phi = -log x. The
    weight is the product of the gates (face_areas): it is 0 unless all
    three masks are > 0.5, and it goes to 0 continuously, so the cost has
    no step where a mask crosses 0.5. A term with w = 0 is 0 whatever x
    is (a crossing above a top costs nothing).
    - extend None (pure barrier): a term with w > 0 and x <= 0 is +inf, so
      the sum is +inf. Its gradient is then not of use.
    - extend = delta > 0: below x = delta, phi is the quadratic
      continuation of -log at delta: -log delta - (x - delta) / delta
      + (x - delta)^2 / (2 delta^2). It is finite for all x.
    The levels are discrete: a triangle that becomes degenerate between two
    levels and comes back with the same orientation is not seen.

    Crowding: sum over faces and levels of
    w_f * (sum_g w_g / n_f) * (y_f - ybar_f)^2, y = log(max(x, floor)),
    ybar_f the w-weighted mean of y over the n_f faces that share an edge
    with f, at the same level. floor = extend, or 1e-12 for the pure
    barrier (a guard only: with x <= 0 the barrier is +inf). relative
    True: y is log(a / a0), so equal growth of the faces against the root
    mesh costs nothing. relative False: y is log(a), as the plan writes it,
    so an irregular root mesh costs something at every level."""
    x, w = face_areas(p, mask, levels, inputs)
    if extend is None:
        xs = torch.where(x > 0, x, torch.ones_like(x))
        phi = torch.where(x > 0, -torch.log(xs), torch.full_like(x, torch.inf))
        floor = 1e-12
    else:
        dl = float(extend)
        if not dl > 0:
            raise ValueError("extend must be > 0")
        xs = torch.where(x > dl, x, torch.ones_like(x))
        q = -np.log(dl) - (x - dl) / dl + (x - dl) ** 2 / (2 * dl * dl)
        phi = torch.where(x > dl, -torch.log(xs), q)
        floor = dl
    barrier = torch.where(w > 0, w * phi, torch.zeros_like(phi)).sum()

    y = torch.log(x.clamp(min=floor))
    if not relative:
        y = y + torch.log(inputs.a0)[:, None]
    pad = y.new_zeros((1, y.shape[1]))
    yn = inputs.face_nbr(torch.cat([y, pad]))  # (F, w, Kw)
    wn = inputs.face_nbr(torch.cat([w, pad]))
    sw = wn.sum(1)
    ybar = (wn * yn).sum(1) / sw.clamp(min=1e-12)
    nf = inputs.face_deg.clamp(min=1)[:, None]
    crowd = (w * (sw / nf) * (y - ybar) ** 2).sum()
    return barrier, crowd


# ---------------------------------------------------------------- cost 6


def cost_return(p: Paths, mask: torch.Tensor, inputs: Inputs) -> torch.Tensor:
    """Cost 6: sum over columns and samples of
    w_k m g relu(W - W_min)^2, with W_min = min_{s' <= s} W(s') (the
    running minimum, with a gradient) and g = clamp(1 - 2 W_min, 0, 1): 0
    until the running minimum falls below 0.5 (the path has left the WM),
    1 where it is 0. The gate is continuous, in place of a step at 0.5. A
    path along which W does not increase gets exactly 0."""
    wmin = running_min(p.wm)
    g = (1 - 2 * wmin).clamp(0, 1)
    r = torch.relu(p.wm - wmin)
    return (mask * g * r * r * inputs.ds).sum()


# ---------------------------------------------------------------- cost 7


def cost_alignment(p: Paths, mask: torch.Tensor, inputs: Inputs) -> torch.Tensor:
    """Cost 7: sum over columns and samples of w_k m t^T A t, with A the
    laminar field (align_field) sampled trilinearly at the path and t the
    unit tangent. It is c sin^2 of the angle between t and v: 0 along v, c
    across it. A is sampled per component, so between voxels it is not
    exactly of the form c (I - v v^T). Above the top it costs nothing."""
    if inputs.align is None:
        raise ValueError("cost_alignment: inputs.align is None (no laminar field)")
    A = trilinear(inputs.align, p.vox)  # (6, n, NS)
    t = p.tangent
    q = p.speed.new_zeros(p.speed.shape)
    for k, (i, j) in enumerate(PAIRS):
        f = 1.0 if i == j else 2.0
        q = q + f * A[k] * t[..., i] * t[..., j]
    return (mask * q * inputs.ds).sum()


# ---------------------------------------------------------------- the view


def closest_outside(p: Paths, inputs: Inputs) -> torch.Tensor:
    """(n,) mm, detached: for each column, the smallest value along its path
    (samples 0 .. NS - 1) of the distance map inputs.dist, sampled
    trilinearly. The map is the distance to the centre of the nearest
    outside voxel (scipy distance_transform_edt with the voxel size), so a
    path that reaches deep into the outside gives 0, and a path at distance
    d below a flat top face gives about d + half a voxel. A view (a
    diagnostic), not a cost: it separates "almost reaches the top" from
    "censored"."""
    if inputs.dist is None:
        raise ValueError("closest_outside: inputs.dist is None")
    with torch.no_grad():
        out: torch.Tensor = trilinear(inputs.dist, p.vox.detach()).amin(-1)
    return out


# ---------------------------------------------------------------- the sum


def total(
    params: torch.Tensor,
    inputs: Inputs,
    weights: Mapping[str, float],
    warp: Any,
    form: str = "A",
    *,
    censored: torch.Tensor | None = None,
    free_length: torch.Tensor | None = None,
    normalize: bool = False,
    extend: float | None = None,
    relative: bool = True,
) -> dict[str, torch.Tensor]:
    """Every term of TERMS, and "total": the sum of weight * term over the
    terms with a weight that is not 0 (a missing weight is 0; an unknown
    name is an error). A term with weight 0 is still computed and reported.
    "alignment" needs inputs.align: without it the term is not in the
    result, and a weight that is not 0 is an error.

    warp: a Levels, or a flattening.Warp on the same columns. censored
    (n,) bool: default the censored flags of these params (give the
    round's set to keep the cost smooth). free_length (n,): the length of
    the censored columns in cost 3; default their own soft length,
    detached."""
    bad = set(weights) - set(TERMS)
    if bad:
        raise ValueError(f"unknown terms: {sorted(bad)}")
    if inputs.align is None and float(weights.get("alignment", 0.0)) != 0.0:
        raise ValueError("the weight of alignment is not 0, but inputs.align is None")
    lv = warp if isinstance(warp, Levels) else Levels.from_warp(inputs, warp)
    p = paths(params, inputs)
    top = soft_top(p, inputs)
    m = top.mask
    cen = top.censored if censored is None else censored
    free = top.length.detach() if free_length is None else free_length
    barrier, crowd = cost_crossing(p, m, lv, inputs, extend, relative)
    out = {
        "agreement": cost_agreement(features(p, m, inputs), m, lv, inputs, normalize),
        "length": cost_length(top.length, cen),
        "length_agreement": cost_length_agreement(top.length, cen, free, inputs, form),
        "bending": cost_bending(p, m, inputs),
        "shear": cost_shear(p, m, inputs),
        "barrier": barrier,
        "crowding": crowd,
        "return": cost_return(p, m, inputs),
    }
    if inputs.align is not None:
        out["alignment"] = cost_alignment(p, m, inputs)
    tot = params.new_zeros(())
    for k in out:
        wk = float(weights.get(k, 0.0))
        if wk != 0.0:
            tot = tot + wk * out[k]
    out["total"] = tot
    return out
