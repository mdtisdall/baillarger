"""Flattening on the column-volume sampler (plan: docs/version-2-plan.md,
section 5c, step 2; docs/plans/phase-3.1-flattening.md).

This module is the adapter between the samplers (ray-sampler/1,
column-volume/1, the mesh of white-surf/1) and the solver library
baillarger.flatten. It does not change the solver.

1. Extents. Each column goes from the start of the label's cortex (the end
   of the root's WM) to the end of the first cortex interval outward, with a
   margin of MARGIN_MM on each side, within the caps (extents). This is the
   first-extent rule of section 5c, step 3, without the censoring.
2. Padding. Every column shares the grid s_k = k step, so the samples of a
   call go into arrays (n_columns, n) on one grid from s_lo, with no
   interpolation (pad).
3. Features. Echoes 2 and 3, as in the prototype
   (prototypes/scripts/flatten_volume.py): the log of the echo, minus its
   local cortex level, zero outside the label's cortex, each divided by a
   scale (features). The local cortex level is a normalized Gaussian of
   TREND_MM along the column, over the column's cortex samples. (The
   prototype used a 3D Gaussian of 4 voxels over the box's cortex; the
   column volume has samples along the columns only.) The scale is the SD
   of the feature over the cortex samples of a set of columns. Give the
   scales as an argument to make a column's result independent of the call
   (scales_of).
4. Pairs. The edges of the white mesh between two columns of the call
   (mesh_edge_pairs), each edge once. proximity_pairs gives the
   prototype's pairs (roots within 0.25 mm, normals with a dot product
   above 0.7) for comparison.
5. Shifts and weights between the two columns of each pair
   (flatten.shifts, flatten.pair_weights), then the warp
   (flatten.fit_warp), anchored at the root: r_v = 0 at s = 0.

The grid of the warp: level j is at s = s_lo + j dtau along the column,
before the warp. The horizon of level j on column v is at
s = s_lo + tau_j + r_v(tau_j).

The gauge. The data rows of the warp see only differences r_w - r_v. So
at each level, a constant added to every column of a connected set does
not change them. fit_warp gives the solution with mean_v r_v(tau) = 0 over
the columns of the solve, at each level: the level tau is the mean depth of
its horizon over the solve's columns. A local solve (one superblock with its
halo) therefore has its own gauge, and no halo removes the difference. It
can be large: on M1S1, the mean of a region's warp over a 2 mm cube was
about 0.15 mm.

The solve has two levels:

1. Local: flatten_superblock solves each superblock with its halo
   (HALO_MM, in mm). This fixes the shape of the warp near the owned
   columns, up to one constant per level.
2. Coarse: gauge_offsets gives one offset c_b(tau) per superblock b and
   level. It is the least-squares fit of (r_b' + c_b') = (r_b + c_b) on the
   columns that two superblocks compute (owned by b, in the halo of b'),
   with the sum of r + c over the owned columns equal to 0 in each
   connected set of superblocks (for the hemisphere: the gauge of one
   global solve). There is one unknown per superblock and level, so the
   coarse problem is small at any scale.

Units: mm.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from baillarger import flatten as FL
from baillarger import rays as R
from baillarger.column_volume import ColumnVolume
from baillarger.mesh_smoothing import edges

type Bool = npt.NDArray[np.bool_]
type F32 = npt.NDArray[np.float32]
type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]

#: the margin of the extents beyond the label's cortex, each side, mm
MARGIN_MM = 0.5
#: the echoes of the features (0-based: echoes 2 and 3)
FEATURE_ECHOES = (1, 2)
#: the scale of the local cortex level along a column, mm (the prototype's
#: 4 voxels at 0.16 mm)
TREND_MM = 0.64
#: the spacing of the warp's levels, mm, and the smoothness weight
DTAU = 0.08
EPS = 0.2
#: the prototype's pairs: roots within this distance (mm), normals with a
#: dot product above NORMAL_DOT
PAIR_MM = 0.25
NORMAL_DOT = 0.7


# ---------------------------------------------------------------- extents


def extents(rays: R.Rays, margin: float = MARGIN_MM) -> tuple[F64, F64]:
    """e_in and e_out (n,) of each column, mm, from its label intervals.

    The cortex interval of a column is its first interval of class CORTEX
    that ends outward of the root (s1 > 0). e_out is its end plus margin,
    and e_in is minus its start plus margin (the start of the cortex is the
    end of the root's WM in the labels), each clipped to [0, cap]. A column
    with no such interval gets e_in = e_out = margin."""
    n = len(rays.roots.vertex)
    e_in = np.full(n, margin)
    e_out = np.full(n, margin)
    cls, s0, s1 = rays.interval_class, rays.interval_s0, rays.interval_s1
    ist = rays.interval_start
    hit = (cls == R.CORTEX) & (s1 > 0)
    col = np.repeat(np.arange(n), np.diff(ist))
    q = np.flatnonzero(hit)
    if len(q):
        # the first hit of each column
        c, first = np.unique(col[q], return_index=True)
        qq = q[first]
        e_in[c] = -s0[qq].astype(np.float64) + margin
        e_out[c] = s1[qq].astype(np.float64) + margin
    return np.clip(e_in, 0.0, R.CAP_IN), np.clip(e_out, 0.0, R.CAP_OUT)


# ---------------------------------------------------------------- padding


@dataclass(frozen=True)
class Padded:
    """The samples of a call on one grid: index i is at s = s_lo + i step."""

    s_lo: float
    step: float
    #: (E, n_columns, n) the echoes, NaN where a column has no valid sample
    echo: F64
    #: (n_columns, n) where a column has a valid sample
    valid: Bool
    #: (n_columns, n) where the sample is in a CORTEX interval
    cortex: Bool


def grid_start(e_in_max: float, dtau: float = DTAU) -> float:
    """s_lo: the start of the grid, at or below -e_in_max, on the level grid
    (a multiple of dtau), so that the root s = 0 is a level."""
    return -dtau * float(np.ceil(e_in_max / dtau - 1e-9))


def pad(
    cv: ColumnVolume,
    echoes: tuple[int, ...] = FEATURE_ECHOES,
    s_lo: float | None = None,
    n: int | None = None,
) -> Padded:
    """The samples of cv as arrays on the grid s_lo + i step, i < n.

    s_lo defaults to grid_start of the call's largest e_in; n to the last
    sample of the call. s_lo must be a multiple of the step. A sample of a
    column at s_k = k step goes to index k - k_lo, k_lo = s_lo / step; no
    value is interpolated."""
    rays = cv.rays
    step = rays.step
    if s_lo is None:
        s_lo = grid_start(float(rays.k_in.max(initial=0)) * step)
    k_lo = int(round(s_lo / step))
    if abs(k_lo * step - s_lo) > 1e-6:
        raise ValueError(f"s_lo {s_lo}: not a multiple of the step {step}")
    if n is None:
        n = int(rays.k_out.max(initial=0)) - k_lo + 1
    m = len(rays.roots.vertex)
    counts = np.diff(rays.sample_start)
    col = np.repeat(np.arange(m), counts)
    k = np.rint(rays.s.astype(np.float64) / step).astype(np.int64)
    i = k - k_lo
    ok = (i >= 0) & (i < n)
    echo = np.full((len(echoes), m, n), np.nan)
    valid = np.zeros((m, n), bool)
    cortex = np.zeros((m, n), bool)
    ci, ii = col[ok], i[ok]
    for e, a in enumerate(echoes):
        echo[e, ci, ii] = cv.echo[ok, a]
    valid[ci, ii] = cv.valid[ok]
    # the class of each sample: the interval s0 <= s < s1 of its column, by
    # one search over keys that order the columns, then s (|s| <= 10 mm)
    ist = rays.interval_start
    icol = np.repeat(np.arange(m), np.diff(ist)).astype(np.float64)
    key_iv = icol * 64.0 + rays.interval_s1.astype(np.float64)
    key_s = col[ok].astype(np.float64) * 64.0 + rays.s[ok].astype(np.float64)
    j = np.searchsorted(key_iv, key_s, side="right")
    j = np.minimum(j, ist[ci + 1] - 1)
    cortex[ci, ii] = rays.interval_class[j] == R.CORTEX
    echo[:, ~valid] = np.nan
    return Padded(float(s_lo), step, echo, valid, cortex & valid)


# ---------------------------------------------------------------- features


def _masked_smooth(a: F64, m: Bool, sig: float) -> F64:
    out: F64 = FL.masked_smooth(a, m, sig)
    return out


def raw_features(p: Padded, trend_mm: float = TREND_MM) -> F64:
    """(E, n_columns, n): the log of each echo minus its local cortex level,
    zero outside the cortex, before the scale."""
    L = np.log(np.maximum(np.nan_to_num(p.echo, nan=1.0), 1.0))
    m = p.cortex
    trend = _masked_smooth(L, m[None], trend_mm / p.step)
    out: F64 = np.where(m[None], L - trend, 0.0)
    return out


def scales_of(raw: F64, cortex: Bool, rows: Bool | None = None) -> F64:
    """(E,): the SD of each raw feature over the cortex samples of the
    columns `rows` (default: all)."""
    m = cortex if rows is None else cortex & rows[:, None]
    out: F64 = np.array([float(np.std(raw[e][m])) for e in range(raw.shape[0])])
    return out


def features(
    p: Padded, scales: F64 | None = None, trend_mm: float = TREND_MM
) -> tuple[F64, Bool, F64]:
    """The features (E, n_columns, n), their mask (the valid samples) and the
    scales used (default: scales_of over the call's cortex samples)."""
    raw = raw_features(p, trend_mm)
    sc = scales_of(raw, p.cortex) if scales is None else np.asarray(scales, float)
    F: F64 = raw / sc[:, None, None]
    return F, p.valid, sc


# ---------------------------------------------------------------- pairs


@dataclass(frozen=True)
class SurfaceFaces:
    """The faces of white-surf/1 with its block index, to find the mesh
    edges between columns: faces (m, 3), possibly memory-mapped; for each
    block, its face range and the bounding box (voxel coordinates) of the
    vertices its faces use."""

    faces: npt.NDArray[np.integer[Any]]
    face_start: I64
    face_stop: I64
    box_min: F64
    box_max: F64

    @classmethod
    def load(cls, surf: Any) -> "SurfaceFaces":
        """From the _white.surf.gii of white-surf/1 and its block index."""
        from pathlib import Path

        from nibabel.gifti.gifti import GiftiImage

        from baillarger.bids import read_sidecar, sidecar_path
        from baillarger.white_surf import BLOCK_COLUMNS, SURF_SPEC

        surf = Path(surf)
        side = read_sidecar(sidecar_path(surf), SURF_SPEC)
        g = GiftiImage.from_filename(str(surf), mmap=True)
        idx = np.loadtxt(
            surf.with_name(side["BlocksFile"]), skiprows=1, ndmin=2
        ).reshape(-1, len(BLOCK_COLUMNS))
        return cls(
            g.darrays[1].data,
            idx[:, 5].astype(np.int64),
            idx[:, 6].astype(np.int64),
            idx[:, 7:10].astype(np.float64),
            idx[:, 10:13].astype(np.float64),
        )


def orient(pairs: I64, key: I64) -> I64:
    """The pairs with the column of the smaller key first, sorted by
    column index. A pair's shift is read on its first column (flatten.shifts,
    flatten.fit_warp), so the order within a pair must not depend on the
    call: with the edge keys it does not."""
    p = np.asarray(pairs, np.int64).reshape(-1, 2)
    swap = key[p[:, 0]] > key[p[:, 1]]
    out: I64 = np.where(swap[:, None], p[:, ::-1], p)
    if len(out):
        out = out[np.lexsort((out[:, 1], out[:, 0]))]
    return out


def mesh_edge_pairs(sf: SurfaceFaces, vertex: I64, voxel: F64) -> I64:
    """The mesh edges between two columns, as pairs (P, 2) of column indices,
    smaller first, each edge once, sorted. Give them to orient before the
    shifts.

    vertex (n,): the root's vertex id of each column; voxel (n, 3): its
    voxel coordinates. Every face that uses a root belongs to a block whose
    bounding box holds that root, so the faces of the blocks whose box meets
    the roots' bounding box hold every edge between two roots."""
    if len(vertex) == 0:
        return np.zeros((0, 2), np.int64)
    lo, hi = voxel.min(0), voxel.max(0)
    near = np.all((sf.box_max >= lo - 1e-3) & (sf.box_min <= hi + 1e-3), axis=1)
    rows = [
        np.asarray(sf.faces[a:b], np.int64)
        for a, b in zip(sf.face_start[near], sf.face_stop[near], strict=True)
        if b > a
    ]
    if not rows:
        return np.zeros((0, 2), np.int64)
    f = np.concatenate(rows)
    order = np.argsort(vertex, kind="stable")
    sv = vertex[order]
    # keep the faces with at least two roots; then their edges
    pos = np.clip(np.searchsorted(sv, f), 0, len(sv) - 1)
    inn = sv[pos] == f
    f = f[inn.sum(1) >= 2]
    if not len(f):
        return np.zeros((0, 2), np.int64)
    # edges() keys by the faces' own indices: compact them first
    ids, inv = np.unique(f, return_inverse=True)
    e = ids[edges(inv.reshape(-1, 3))]
    p = np.clip(np.searchsorted(sv, e), 0, len(sv) - 1)
    both = np.all(sv[p] == e, axis=1)
    pc = order[p[both]]
    pc = np.sort(pc, axis=1)
    out: I64 = np.unique(pc, axis=0) if len(pc) else pc
    return out


def proximity_pairs(
    root_mm: npt.ArrayLike,
    normal: npt.ArrayLike,
    radius: float = PAIR_MM,
    dot: float = NORMAL_DOT,
) -> I64:
    """The prototype's pairs: roots within radius (mm) whose normals have a
    dot product above dot; (P, 2), smaller first, sorted."""
    x = np.asarray(root_mm, np.float64)
    nr = np.asarray(normal, np.float64)
    p = np.array(sorted(cKDTree(x).query_pairs(radius)), np.int64).reshape(-1, 2)
    out: I64 = p[(nr[p[:, 0]] * nr[p[:, 1]]).sum(1) > dot]
    return out


# ---------------------------------------------------------------- shifts


@dataclass(frozen=True)
class PairShifts:
    """Shifts between the columns of each pair, at each sample of the first
    (flatten.shifts), and their weights (flatten.pair_weights)."""

    pairs: I64
    #: (P, n) mm
    shift: F32
    #: (P, n)
    weight: F32
    #: the energy reference of the weights
    energy_ref: float


def pair_shifts(
    F: F64,
    M: Bool,
    pairs: I64,
    step: float,
    energy_ref: float | None = None,
    chunk: int = 20000,
) -> PairShifts:
    """The shift and weight of each pair, in chunks of pairs. energy_ref
    defaults to the median gradient energy over the valid samples of all
    pairs (the prototype's rule). Give it to make a column's result
    independent of the call."""
    P, n = len(pairs), F.shape[2]
    S = np.zeros((P, n), np.float32)
    NC = np.zeros((P, n), np.float32)
    EN = np.zeros((P, n), np.float32)
    VA = np.zeros((P, n), bool)
    for a in range(0, P, chunk):
        p = pairs[a : a + chunk]
        s, ncc, en, va = FL.shifts(
            F[:, p[:, 0]], F[:, p[:, 1]], M[p[:, 0]], M[p[:, 1]], step
        )
        S[a : a + chunk], NC[a : a + chunk] = s, ncc
        EN[a : a + chunk], VA[a : a + chunk] = en, va
    eref = float(np.median(EN[VA])) if energy_ref is None else float(energy_ref)
    W = NC  # in place, chunk by chunk: no float64 copy of the whole
    for a in range(0, P, chunk):
        c = slice(a, a + chunk)
        W[c] = FL.pair_weights(
            NC[c].astype(np.float64), EN[c].astype(np.float64), VA[c], eref
        )
    return PairShifts(pairs, S, W, eref)


# ---------------------------------------------------------------- the warp


@dataclass(frozen=True)
class Warp:
    """The fitted warp of a set of columns: r (n_columns, K) on the levels
    tau_j = j dtau; the horizon of level j on column v is at
    s = s_lo + tau_j + r[v, j]; r = 0 at the root's level."""

    s_lo: float
    tau: F64
    r: F64
    anchor_level: int
    info: list[dict[str, Any]]


def fit(
    ps: PairShifts,
    n_columns: int,
    s_lo: float,
    step: float,
    dtau: float = DTAU,
    eps: float = EPS,
    r0: F64 | None = None,
    log: Callable[[str], object] | None = None,
    **kw: Any,
) -> Warp:
    """flatten.fit_warp over the grid of the shifts, anchored at the root."""
    n = ps.shift.shape[1]
    ja = int(round(-s_lo / dtau))
    tau_max = dtau * np.floor((n - 1) * step / dtau + 1e-9)
    tau, r, info = FL.fit_warp(
        ps.pairs,
        ps.shift,
        ps.weight,
        n_columns,
        step,
        dtau=dtau,
        tau_max=float(tau_max),
        eps=eps,
        r0=r0,
        log=log,
        anchor_level=ja,
        **kw,
    )
    return Warp(s_lo, tau, r, ja, info)


# ---------------------------------------------------------------- a superblock

#: the halo of the local solve, mm (from the decay measurement of phase 3.1)
HALO_MM = 2.0


@dataclass(frozen=True)
class Flattened:
    """The flattening of the owned columns of one superblock, in the gauge of
    its own call (see "The gauge" in the module text: add the superblock's
    offsets from gauge_offsets).

    Level j of the warp is at s = s_lo + tau[j] before the warp; the
    horizon of level j on owned column c is at s = s_lo + tau[j] + r[c, j],
    and r = 0 at the root (level anchor_level). The pairs are the mesh
    edges with at least one owned end; the shift and the weight of pair p
    are at the samples s = s_lo + i step of its first column."""

    #: (n,) the edge keys of the owned columns
    edge_key: I64
    s_lo: float
    step: float
    tau: F64
    #: (n, K) mm
    r: F64
    anchor_level: int
    #: (P, 2) the edge keys of the two columns of each pair, smaller first
    pair_key: I64
    #: (P, n_samples) mm, and the weights
    shift: F32
    weight: F32
    #: the feature scales and the energy reference that were used
    scales: F64
    energy_ref: float
    #: the solve's log: one entry per outer pass (cg iterations, misfit)
    info: list[dict[str, Any]]
    #: columns and pairs of the call, with the halo
    n_columns: int
    n_pairs: int
    #: (n,) the weighted RMS misfit r_w - r_v - s_vw of each owned column
    #: over its pairs and levels, mm (NaN with a total weight below 0.05)
    misfit: F64
    #: (n,) the root's vertex in white-surf/1 of each owned column
    vertex: I64
    #: the halo columns: their edge keys (h,) and their warp (h, K), for the
    #: offsets between superblocks (gauge_offsets)
    halo_key: I64
    halo_r: F64
    #: with keep_echo: the padded echoes of the owned columns (Padded)
    echo: "Padded | None" = None


def flatten_superblock(
    ray_sampler: R.RaySampler,
    cv_sampler: Any,
    faces: SurfaceFaces,
    superblock: tuple[int, int, int],
    level: int,
    halo_mm: float = HALO_MM,
    scales: F64 | None = None,
    energy_ref: float | None = None,
    device: str | None = None,
    margin: float = MARGIN_MM,
    log: Callable[[str], object] | None = None,
    keep_echo: bool = False,
    **fit_kw: Any,
) -> Flattened:
    """The flattening of one superblock: the columns of the superblock and
    its halo (halo_mm, in mm) from the samplers, the shifts on the mesh
    edges, and one solve of the warp over all of them (a local solve). It
    returns the owned columns' warp and the pairs that touch them.

    cv_sampler is a column_volume.ColumnVolumeSampler on the echoes. Give
    scales and energy_ref (for example from one call, or for a whole
    hemisphere) so that a column's result does not depend on the
    superblock; by default each call computes its own."""
    r0 = ray_sampler.sample(superblock, level, e_in=0.0, e_out=0.0, halo_mm=halo_mm)
    e_in, e_out = extents(r0, margin)
    del r0
    rays = ray_sampler.sample(
        superblock, level, e_in=e_in, e_out=e_out, halo_mm=halo_mm
    )
    cv = cv_sampler.sample(rays, device)
    p = pad(cv, FEATURE_ECHOES)
    del cv
    F, M, sc = features(p, scales)
    inv = ray_sampler.inverse
    vox = (inv[:3, :3] @ rays.roots.root.astype(np.float64).T).T + inv[:3, 3]
    key = rays.roots.edge_key
    pairs = orient(mesh_edge_pairs(faces, rays.roots.vertex, vox), key)
    ps = pair_shifts(F, M, pairs, p.step, energy_ref)
    del F, M
    n = len(rays.roots.vertex)
    if log:
        log(f"{n:,} columns, {len(pairs):,} pairs, grid {p.echo.shape[2]} samples")
    warp = fit(ps, n, p.s_lo, p.step, log=log, **fit_kw)
    own = rays.roots.owned
    touch = own[pairs[:, 0]] | own[pairs[:, 1]]
    wt, mis = FL.residuals(
        pairs[touch], ps.shift[touch], ps.weight[touch], warp.r, warp.tau, p.step
    )
    num, den = np.zeros(n), np.zeros(n)
    for c in (0, 1):
        np.add.at(num, pairs[touch, c], (wt * mis**2).sum(1))
        np.add.at(den, pairs[touch, c], wt.sum(1))
    misfit = np.where(den >= 0.05, np.sqrt(num / np.maximum(den, 1e-12)), np.nan)
    echo = None
    if keep_echo:
        echo = Padded(p.s_lo, p.step, p.echo[:, own], p.valid[own], p.cortex[own])
    return Flattened(
        edge_key=key[own],
        s_lo=p.s_lo,
        step=p.step,
        tau=warp.tau,
        r=warp.r[own],
        anchor_level=warp.anchor_level,
        pair_key=key[pairs[touch]],
        shift=ps.shift[touch],
        weight=ps.weight[touch],
        scales=sc,
        energy_ref=ps.energy_ref,
        info=warp.info,
        n_columns=n,
        n_pairs=len(pairs),
        misfit=misfit[own],
        vertex=rays.roots.vertex[own],
        halo_key=key[~own],
        halo_r=warp.r[~own],
        echo=echo,
    )


# ---------------------------------------------------------------- the gauge


def common_levels(f: Flattened, r: F64, lev: I64, dtau: float = DTAU) -> F64:
    """r (m, K) of one call on the common levels lev (L,): level l is at
    s = l dtau before the warp. A level below or above the call's grid takes
    the call's first or last level: there the warp has no data, and its
    smoothness rows keep r constant."""
    own = np.rint((f.s_lo + f.tau) / dtau).astype(np.int64)
    j = np.clip(lev - own[0], 0, len(own) - 1)
    out: F64 = r[:, j]
    return out


def gauge_offsets(results: list[Flattened], dtau: float = DTAU) -> tuple[I64, F64]:
    """The offsets c (B, L) of B superblocks' warps on the common levels
    lev (L,): r of an owned column of superblock b, on the common levels,
    plus c[b] is the warp in one gauge for all the superblocks.

    At each level, c solves a least-squares problem with one unknown per
    superblock: for each column that superblock b owns and superblock b'
    has in its halo, (r_b' + c_b') - (r_b + c_b) = 0; and in each connected
    set of superblocks, the sum over their owned columns of r + c is 0 (the
    gauge of one solve over all their columns: fit_warp gives
    mean_v r_v(tau) = 0). The problem is a weighted graph Laplacian over the
    superblocks with one constraint per connected set, solved exactly
    (sparse LU of its saddle-point matrix)."""
    from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import spsolve

    B = len(results)
    lo = min(int(np.rint((f.s_lo) / dtau)) for f in results)
    hi = max(int(np.rint((f.s_lo + f.tau[-1]) / dtau)) for f in results)
    lev = np.arange(lo, hi + 1, dtype=np.int64)
    L = len(lev)
    # the owner of each column, and its warp on the common levels
    keys = np.concatenate([f.edge_key for f in results])
    owner = np.concatenate(
        [np.full(len(f.edge_key), b, np.int64) for b, f in enumerate(results)]
    )
    if len(np.unique(keys)) != len(keys):
        raise ValueError("a column is owned by two superblocks")
    order = np.argsort(keys)
    skeys = keys[order]
    r_own = np.concatenate([common_levels(f, f.r, lev, dtau) for f in results])
    # the overlaps: (b, b', column row in r_own, the halo row in b')
    rows_b, rows_h, d_list = [], [], []
    for bh, f in enumerate(results):
        if not len(f.halo_key):
            continue
        p = np.clip(np.searchsorted(skeys, f.halo_key), 0, len(skeys) - 1)
        hit = skeys[p] == f.halo_key
        q = order[p[hit]]
        rows_b.append(owner[q])
        rows_h.append(np.full(len(q), bh, np.int64))
        d_list.append(common_levels(f, f.halo_r[hit], lev, dtau) - r_own[q])
    bo = np.concatenate(rows_b) if rows_b else np.zeros(0, np.int64)
    bh_ = np.concatenate(rows_h) if rows_h else np.zeros(0, np.int64)
    d = np.concatenate(d_list) if d_list else np.zeros((0, L))
    # rows c_b - c_b' = r_b' - r_b, for the overlaps between two superblocks
    two = bo != bh_
    bo, bh_, d = bo[two], bh_[two], d[two]
    m = len(bo)
    A = coo_matrix(
        (
            np.r_[np.ones(m), -np.ones(m)],
            (np.r_[np.arange(m), np.arange(m)], np.r_[bo, bh_]),
        ),
        shape=(m, B),
    ).tocsr()
    G = coo_matrix((np.ones(m), (bo, bh_)), shape=(B, B))
    ncomp, comp = connected_components(G, directed=False)
    n_own = np.bincount(owner, minlength=B).astype(np.float64)
    sum_own = np.zeros((B, L))
    np.add.at(sum_own, owner, r_own)
    # the constraints: sum over the owned columns of the set of (r + c) = 0
    C = csr_matrix((n_own, (comp, np.arange(B))), shape=(ncomp, B))
    g = np.zeros((ncomp, L))
    np.add.at(g, comp, -sum_own)
    AtA = (A.T @ A).tocsr()
    K = vstack([hstack([AtA, C.T]), hstack([C, csr_matrix((ncomp, ncomp))])]).tocsc()
    rhs = np.vstack([np.asarray(A.T @ d), g])
    sol = spsolve(K, rhs)
    c: F64 = np.asarray(sol, np.float64).reshape(B + ncomp, L)[:B]
    return lev, c


def in_gauge(f: Flattened, c_b: F64, lev: I64, dtau: float = DTAU) -> F64:
    """The owned columns' warp of f on the common levels lev, plus its
    offsets c_b (L,): the warp in the gauge of gauge_offsets. The horizon
    of common level l is at s = l dtau + r."""
    out: F64 = common_levels(f, f.r, lev, dtau) + c_b[None]
    return out
