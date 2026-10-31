"""The pial surface as a height over the white mesh (plan: reports/Pial editing and edit
priority.md, §2 and §6 step 1). Prototype only.

The pial surface is the white mesh with each vertex moved up its own column to that
column's top. Pieces:

- white_surface: finding 28's surface: the WM label contoured by marching tetrahedra,
  relaxed tangentially, smoothed along the mesh by twicing at 2 voxels (0.32 mm);
  outward unit normals.
- march_from_mesh: one unweighted march (mesh_smoothing.march_band: the package's
  scheme with per-seed start values) from a narrow band round the white mesh, through
  cortex and fluid (the geometry piece of the main plan's *The fronts as PDE problems*:
  no thickness beliefs enter). Carries arrival, arc length and the root position.
- characteristic_field: the columns' direction, D grad T normalized, with D = M^-1 the
  dual metric. Under an anisotropic metric the characteristics are along D grad T, not
  along grad T.
- trace_columns: from each white vertex, RK2 along the characteristic field, recording
  samples every `rec` voxels. Extensions (plan §2):
  - past the ridge of arrival, when the ridge lies in cortex: along the data's layer
    normal, signed by continuity with the last direction (the column's own direction
    where the layer normal is missing);
  - into fluid: straight on along the last direction, up to `fluid_mm` past the entry.
  A column stops at the length cap, on entering WM (after the first voxel), outside the
  domain or the box, and `fluid_mm` past the fluid entry.
- tops of base models on the same columns, as arc length along the column:
  - top_label: the first crossing of the fluid label (trilinear indicator at 0.5); the
    ridge of arrival where there is no fluid before it;
  - top_onepass: the first sample in fluid or in the territory of a front whose root
    patch is more than sep_min from the column's own patch along the white surface
    (onepass_lib's contact rule), placed midway between the last own and first
    foreign sample.
- click_column: a click's column from the carried root position and arc length,
  interpolated at the click (no search over polylines), and its miss distance.

Units: voxels of the 1x grid (voxel centres at integer coordinates) unless stated; the
patches' voxels are 0.16028 x 0.16 x 0.16028 mm, treated as isotropic H_MM (0.2 %).
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi

import mesh_smoothing as M
from baillarger._jit import njit
from baillarger.fronts import riemann_from_direction

type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]
type Bool = npt.NDArray[np.bool_]

H_MM = 0.16
# voxel codes for the tracer
OUT, CTX, FLUID, WM = 0, 1, 2, 3
# why a column stopped
CAP, ENTER_WM, LEFT, FLUID_END, NO_DIRECTION = 1, 2, 3, 4, 5
STOP_NAMES = {
    CAP: "length cap",
    ENTER_WM: "entered WM",
    LEFT: "left the domain",
    FLUID_END: "end of the fluid extension",
    NO_DIRECTION: "no direction",
}
# how a model's top was placed
TOP_FLUID, TOP_FOREIGN, TOP_RIDGE, TOP_NONE = 1, 2, 3, 4
TOP_NAMES = {
    TOP_FLUID: "fluid",
    TOP_FOREIGN: "another front",
    TOP_RIDGE: "ridge of arrival",
    TOP_NONE: "none (column end)",
}


# ------------------------------------------------------------------ white surface


def white_surface(wm: Bool, sigma: float = 2.0) -> tuple[Float, Int, Float]:
    """Vertices (voxel coordinates), faces and outward unit normals of the twicing
    surface (finding 28). Outward is checked on the label and fixed if needed."""
    v0, f = M.contour(wm, 0.0)
    v = M.twice(M.relax(v0, f), f, sigma)
    n = M.smooth_normals(v, f)
    # outward = away from the WM: the label 1 voxel out along n should be non-WM
    shape = np.array(wm.shape)
    ahead = np.clip(np.round(v + n).astype(np.int64), 0, shape - 1)
    behind = np.clip(np.round(v - n).astype(np.int64), 0, shape - 1)
    score = np.mean(~wm[tuple(ahead.T)]) - np.mean(~wm[tuple(behind.T)])
    if score < 0:
        n = -n
    return v, f, n


# ------------------------------------------------------------------ march


@dataclass
class Geometry:
    """The march from the white mesh and what the columns need."""

    arrival: Float  # inf where unreached
    length: Float  # arc length, nan where unreached
    root: Float  # (..., 3) carried root position (voxels), nan where unreached
    direction: Float  # (..., 3) the metric's unit direction (zero: isotropic)
    ratio: float
    codes: npt.NDArray[np.int8]  # OUT, CTX, FLUID, WM


def march_from_mesh(
    v: Float,
    f: Int,
    wm: Bool,
    ctx: Bool,
    fluid: Bool,
    direction: Float,
    ratio: float,
    band: float = 2.5,
) -> Geometry:
    """One unweighted march from the white mesh through cortex and fluid. direction:
    unit vectors (..., 3) (zero: isotropic there); ratio: cost across it."""
    dom = ctx | fluid
    Mt = riemann_from_direction(direction, ratio)
    D = None if ratio == 1.0 else np.linalg.inv(Mt)
    X = np.stack(np.indices(wm.shape), -1).astype(np.float64)
    seeds, t, s, r = M.surface_seeds(dom, wm, X, M.mesh_closest(v, f), D, band=band)
    del X, D
    out = M.march_band(Mt, dom, seeds, t, s, r)
    codes = np.zeros(wm.shape, np.int8)
    codes[ctx] = CTX
    codes[fluid] = FLUID
    codes[wm] = WM
    return Geometry(
        out.arrival,
        out.length,
        out.root,
        np.asarray(direction, np.float64),
        ratio,
        codes,
    )


def characteristic_field(g: Geometry) -> npt.NDArray[np.float32]:
    """Unit D grad T (zero where T or its gradient is undefined)."""
    T = np.where(np.isfinite(g.arrival), g.arrival, np.nan)
    grad = np.stack(np.gradient(T), -1)
    v = g.direction
    if g.ratio != 1.0:
        along = np.sum(grad * v, -1, keepdims=True)
        grad = along * v + (grad - along * v) / g.ratio**2
    nrm = np.linalg.norm(grad, axis=-1, keepdims=True)
    ok = np.isfinite(nrm) & (nrm > 1e-9)
    c = np.where(ok, grad / np.where(ok, nrm, 1.0), 0.0)
    return c.astype(np.float32)


# ------------------------------------------------------------------ tracing


@njit(cache=True)
def _interp_vec(F, x, ref, signed):
    """Trilinear interpolation of a vector field at x; with signed=False each corner's
    vector is first turned to agree with ref (unsigned fields, e.g. the layer normal).
    Corners whose vector is zero are left out. Returns a unit vector, or zeros."""
    nx, ny, nz = F.shape[0], F.shape[1], F.shape[2]
    i0 = int(np.floor(x[0]))
    j0 = int(np.floor(x[1]))
    k0 = int(np.floor(x[2]))
    fx = x[0] - i0
    fy = x[1] - j0
    fz = x[2] - k0
    out = np.zeros(3)
    wsum = 0.0
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                i = i0 + di
                j = j0 + dj
                k = k0 + dk
                if i < 0 or j < 0 or k < 0 or i >= nx or j >= ny or k >= nz:
                    continue
                w = (
                    (fx if di else 1 - fx)
                    * (fy if dj else 1 - fy)
                    * (fz if dk else 1 - fz)
                )
                a = F[i, j, k, 0]
                b = F[i, j, k, 1]
                c = F[i, j, k, 2]
                if a == 0.0 and b == 0.0 and c == 0.0:
                    continue
                if not signed and a * ref[0] + b * ref[1] + c * ref[2] < 0:
                    w = -w
                out[0] += w * a
                out[1] += w * b
                out[2] += w * c
                wsum += abs(w)
    n = np.sqrt(out[0] ** 2 + out[1] ** 2 + out[2] ** 2)
    if wsum < 0.25 or n < 1e-9:
        return np.zeros(3)
    return out / n


@njit(cache=True)
def _interp_scalar(F, x):
    """Trilinear interpolation of a finite scalar field (nan corners left out)."""
    nx, ny, nz = F.shape
    i0 = int(np.floor(x[0]))
    j0 = int(np.floor(x[1]))
    k0 = int(np.floor(x[2]))
    fx = x[0] - i0
    fy = x[1] - j0
    fz = x[2] - k0
    s = 0.0
    wsum = 0.0
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                i = i0 + di
                j = j0 + dj
                k = k0 + dk
                if i < 0 or j < 0 or k < 0 or i >= nx or j >= ny or k >= nz:
                    continue
                val = F[i, j, k]
                if not np.isfinite(val):
                    continue
                w = (
                    (fx if di else 1 - fx)
                    * (fy if dj else 1 - fy)
                    * (fz if dk else 1 - fz)
                )
                s += w * val
                wsum += w
    if wsum < 1e-9:
        return np.nan
    return s / wsum


@njit(cache=True)
def _code(codes, x):
    i = int(np.floor(x[0] + 0.5))
    j = int(np.floor(x[1] + 0.5))
    k = int(np.floor(x[2] + 0.5))
    if (
        i < 0
        or j < 0
        or k < 0
        or i >= codes.shape[0]
        or j >= codes.shape[1]
        or k >= codes.shape[2]
    ):
        return -1
    return codes[i, j, k]


@njit(cache=True)
def _trace_all(
    p0,
    n0,
    C,
    T,
    N,
    codes,
    h,
    L_max,
    fluid_ext,
    wm_grace,
    rec,
    K,
    out_pts,
    out_n,
    ridge_s,
    fluid_s,
    stop_s,
    stop_why,
    ext_used,
):
    nrec = int(round(rec / h))
    for r in range(p0.shape[0]):
        x = p0[r].copy()
        d = n0[r].copy()
        s = 0.0
        out_pts[r, 0] = x
        cnt = 1
        mode = 0  # 0 along the characteristics, 1 along the layer normal, 2 straight on in fluid
        ridge_s[r] = np.nan
        fluid_s[r] = np.nan
        why = CAP
        Tx = _interp_scalar(T, x)
        step = 0
        while s < L_max - 1e-9:
            c = _code(codes, x)
            d2 = d.copy()
            if mode == 0 and s > wm_grace:
                d1 = _interp_vec(C, x, d, True)
                ok = (d1[0] != 0.0 or d1[1] != 0.0 or d1[2] != 0.0) and (
                    d1[0] * d[0] + d1[1] * d[1] + d1[2] * d[2] > 0.0
                )
                if ok:
                    d2 = _interp_vec(C, x + 0.5 * h * d1, d1, True)
                    ok = (d2[0] != 0.0 or d2[1] != 0.0 or d2[2] != 0.0) and (
                        d2[0] * d1[0] + d2[1] * d1[1] + d2[2] * d1[2] > 0.0
                    )
                if ok:
                    Tn = _interp_scalar(T, x + h * d2)
                    ok = Tn > Tx
                if not ok:
                    if c == CTX:
                        mode = 1
                        ridge_s[r] = s
                        ext_used[r] = True
                    elif c == FLUID:
                        mode = 2
                        ridge_s[r] = s
                    else:
                        why = NO_DIRECTION
                        break
            elif mode == 0:
                # leaving the WM: along the characteristics where defined, else the normal
                d1 = _interp_vec(C, x, d, True)
                if (d1[0] != 0.0 or d1[1] != 0.0 or d1[2] != 0.0) and (
                    d1[0] * d[0] + d1[1] * d[1] + d1[2] * d[2] > 0.0
                ):
                    d2 = d1
            if mode == 1:
                d1 = _interp_vec(N, x, d, False)
                if d1[0] == 0.0 and d1[1] == 0.0 and d1[2] == 0.0:
                    d1 = d.copy()
                d2 = _interp_vec(N, x + 0.5 * h * d1, d1, False)
                if d2[0] == 0.0 and d2[1] == 0.0 and d2[2] == 0.0:
                    d2 = d1
            if mode == 2:
                d2 = d.copy()
            x = x + h * d2
            d = d2
            s += h
            step += 1
            Tx = _interp_scalar(T, x)
            c = _code(codes, x)
            if step % nrec == 0 and cnt < K:
                out_pts[r, cnt] = x
                cnt += 1
            if c < 0 or c == OUT:
                why = LEFT
                break
            if c == WM and s > wm_grace:
                why = ENTER_WM
                break
            if c == FLUID:
                if np.isnan(fluid_s[r]):
                    fluid_s[r] = s
                mode = 2
                if s - fluid_s[r] > fluid_ext:
                    why = FLUID_END
                    break
        if cnt < K and step % nrec != 0:
            out_pts[r, cnt] = x
            cnt += 1
        out_n[r] = cnt
        stop_s[r] = s
        stop_why[r] = why


@dataclass
class Columns:
    pts: npt.NDArray[np.float32]  # (nv, K, 3) samples; nan beyond n
    n: Int  # samples per column
    s: Float  # (nv, K) arc length of each sample (voxels); nan beyond n
    ridge_s: Float  # arc length where the ridge of arrival was met (nan: none)
    fluid_s: Float  # arc length of the first fluid voxel (nan: none)
    stop_s: Float
    stop_why: Int
    extended: Bool  # the column was extended past a ridge in cortex


def trace_columns(
    v: Float,
    n: Float,
    g: Geometry,
    C,
    layer_normal: Float,
    h: float = 0.25,
    L_max_mm: float = 5.0,
    fluid_mm: float = 1.0,
    grace: float = 1.0,
    rec: float = 0.5,
) -> Columns:
    """Trace every vertex's column (see the module docstring). layer_normal: unit,
    unsigned (zero where unknown). Lengths in voxels except the _mm arguments."""
    L_max = L_max_mm / H_MM
    K = int(np.ceil(L_max / rec)) + 2
    nv = len(v)
    pts = np.full((nv, K, 3), np.nan)
    cnt = np.zeros(nv, np.int64)
    ridge_s = np.full(nv, np.nan)
    fluid_s = np.full(nv, np.nan)
    stop_s = np.zeros(nv)
    stop_why = np.zeros(nv, np.int64)
    ext = np.zeros(nv, np.bool_)
    T = np.where(np.isfinite(g.arrival), g.arrival, np.nan)
    _trace_all(
        np.ascontiguousarray(v, np.float64),
        np.ascontiguousarray(n, np.float64),
        np.ascontiguousarray(C, np.float64),
        T,
        np.ascontiguousarray(layer_normal, np.float64),
        g.codes,
        h,
        L_max,
        fluid_mm / H_MM,
        grace,
        rec,
        K,
        pts,
        cnt,
        ridge_s,
        fluid_s,
        stop_s,
        stop_why,
        ext,
    )
    seg = np.linalg.norm(np.diff(pts, axis=1), axis=-1)
    s = np.concatenate([np.zeros((nv, 1)), np.cumsum(np.nan_to_num(seg), 1)], 1)
    s[np.arange(K)[None, :] >= cnt[:, None]] = np.nan
    return Columns(
        pts.astype(np.float32), cnt, s, ridge_s, fluid_s, stop_s, stop_why, ext
    )


def point_at(cols: Columns, t: Float, rows: Int | None = None) -> Float:
    """The point at arc length t (voxels) on each column, or on the columns `rows`
    with t given for those (linear between samples; nan where t is nan or beyond the
    column)."""
    rows = np.arange(len(cols.n)) if rows is None else np.asarray(rows, np.int64)
    t = np.asarray(t, np.float64)
    out = np.full((len(rows), 3), np.nan)
    ok = np.isfinite(t)
    S = cols.s[rows]
    # first sample at or beyond t (the arc lengths increase along a column)
    idx = np.sum(
        np.where(np.isfinite(S), S, np.inf) < np.where(ok, t, -np.inf)[:, None], 1
    )
    good = ok & (idx >= 1) & (idx < cols.n[rows])
    j = np.flatnonzero(good)
    r, i = rows[j], idx[j]
    s0, s1 = cols.s[r, i - 1], cols.s[r, i]
    w = ((t[j] - s0) / np.maximum(s1 - s0, 1e-12))[:, None]
    out[j] = (1 - w) * cols.pts[r, i - 1] + w * cols.pts[r, i]
    at0 = ok & (np.abs(t) < 1e-12)
    out[at0] = cols.pts[rows[at0], 0]
    return out


# ------------------------------------------------------------------ tops of base models


def _sample_trilinear(F: Float, P: Float) -> Float:
    return ndi.map_coordinates(F, P.reshape(-1, 3).T, order=1, mode="nearest").reshape(
        P.shape[:-1]
    )


def top_label(cols: Columns, fluid: Bool) -> tuple[Float, Int]:
    """Arc length (voxels) of the first crossing of the fluid label (trilinear
    indicator at 0.5) along each column, and how it was placed. Where the column
    meets no fluid before the ridge of arrival: the ridge."""
    P = np.nan_to_num(cols.pts.astype(np.float64), nan=0.0)
    f = _sample_trilinear(fluid.astype(np.float64), P)
    valid = np.isfinite(cols.s)
    f[~valid] = np.nan
    nv, K = f.shape
    t = np.full(nv, np.nan)
    kind = np.full(nv, TOP_NONE, np.int64)
    above = (f >= 0.5) & valid
    above[:, 0] = False
    first = np.where(above.any(1), above.argmax(1), -1)
    for r in np.flatnonzero(first > 0):
        i = first[r]
        f0, f1 = f[r, i - 1], f[r, i]
        w = (0.5 - f0) / (f1 - f0) if f1 != f0 else 0.0
        t[r] = cols.s[r, i - 1] + w * (cols.s[r, i] - cols.s[r, i - 1])
        kind[r] = TOP_FLUID
    ridge_first = np.isfinite(cols.ridge_s) & (~np.isfinite(t) | (cols.ridge_s < t))
    t[ridge_first] = cols.ridge_s[ridge_first]
    kind[ridge_first] = TOP_RIDGE
    rest = ~np.isfinite(t)
    t[rest] = cols.stop_s[rest]
    return t, kind


def near_patches(setup, sep_min: float) -> Int:
    """Sorted keys a * npatch + b of the patch pairs whose centres are within sep_min
    (voxels) of each other along the white surface (onepass_lib's rule; a with itself
    included)."""
    from scipy.sparse.csgraph import dijkstra

    reps = setup.rep
    npatch = reps.size
    keys = []
    for c0 in range(0, npatch, 256):
        src = reps[c0 : c0 + 256]
        d = dijkstra(setup.graph, indices=src, limit=sep_min)
        i, j = np.nonzero(d[:, reps] <= sep_min)
        keys.append((i + c0).astype(np.int64) * npatch + j)
    return np.unique(
        np.concatenate(keys + [np.arange(npatch, dtype=np.int64) * (npatch + 1)])
    )


def top_onepass(
    cols: Columns,
    label: Int,
    near_keys: Int,
    npatch: int,
    fluid: Bool,
    own_from: float = 1.0,
) -> tuple[Float, Int]:
    """Arc length of the first sample in fluid or in a far front's territory, placed
    midway between it and the sample before. The column's own patch: the label of its
    first cortex sample at least own_from voxels up. near_keys: from near_patches."""
    nv, K = cols.s.shape
    valid = np.isfinite(cols.s)
    Pi = np.round(np.nan_to_num(cols.pts.astype(np.float64), nan=0.0)).astype(np.int64)
    Pi = np.clip(Pi, 0, np.array(label.shape) - 1).reshape(-1, 3).T
    lab = label[tuple(Pi)].reshape(nv, K)
    fl = fluid[tuple(Pi)].reshape(nv, K) & valid
    lab = np.where(valid, lab, -1)
    own_ok = valid & (cols.s >= own_from) & (lab >= 0) & ~fl
    has = own_ok.any(1)
    i0 = np.where(has, own_ok.argmax(1), K)
    a = np.where(has, lab[np.arange(nv), np.minimum(i0, K - 1)], -1)
    key = a[:, None] * npatch + lab
    pos = np.searchsorted(near_keys, key)
    is_near = near_keys[np.minimum(pos, near_keys.size - 1)] == key
    foreign = valid & (lab >= 0) & ~is_near & (a[:, None] >= 0)
    after = np.arange(K)[None, :] > i0[:, None]
    hit = (fl | foreign) & after
    first = np.where(hit.any(1), hit.argmax(1), -1)
    t = np.full(nv, np.nan)
    kind = np.full(nv, TOP_NONE, np.int64)
    r = np.flatnonzero(first > 0)
    i = first[r]
    t[r] = 0.5 * (cols.s[r, i - 1] + cols.s[r, i])
    kind[r] = np.where(fl[r, i], TOP_FLUID, TOP_FOREIGN)
    rest = ~np.isfinite(t)
    t[rest] = cols.stop_s[rest]
    return t, kind


# ------------------------------------------------------------------ clicks


@njit(cache=True)
def _click(length, root, x, same):
    """Arc length and root at x from the 8 surrounding voxels, blending only the corners
    whose root lies within `same` voxels of the root at the corner with the largest
    weight: across a ridge the corners belong to different fronts, and averaging their
    roots gives a point on neither."""
    nx, ny, nz = length.shape
    i0 = int(np.floor(x[0]))
    j0 = int(np.floor(x[1]))
    k0 = int(np.floor(x[2]))
    fx = x[0] - i0
    fy = x[1] - j0
    fz = x[2] - k0
    ws = np.zeros(8)
    ii = np.zeros((8, 3), np.int64)
    ok = np.zeros(8, np.bool_)
    c = 0
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                i = i0 + di
                j = j0 + dj
                k = k0 + dk
                ii[c, 0] = i
                ii[c, 1] = j
                ii[c, 2] = k
                ws[c] = (
                    (fx if di else 1 - fx)
                    * (fy if dj else 1 - fy)
                    * (fz if dk else 1 - fz)
                )
                ok[c] = 0 <= i < nx and 0 <= j < ny and 0 <= k < nz
                if ok[c]:
                    ok[c] = np.isfinite(length[i, j, k]) and np.isfinite(
                        root[i, j, k, 0]
                    )
                c += 1
    ref = -1
    for c in range(8):
        if ok[c] and (ref < 0 or ws[c] > ws[ref]):
            ref = c
    R = np.full(3, np.nan)
    if ref < 0:
        return np.nan, R
    r0 = root[ii[ref, 0], ii[ref, 1], ii[ref, 2]]
    sw = 0.0
    sl = 0.0
    sr = np.zeros(3)
    for c in range(8):
        if not ok[c]:
            continue
        rc = root[ii[c, 0], ii[c, 1], ii[c, 2]]
        if (rc[0] - r0[0]) ** 2 + (rc[1] - r0[1]) ** 2 + (
            rc[2] - r0[2]
        ) ** 2 > same * same:
            continue
        # the corner's arc length carried to x along its column (unit root -> corner);
        # with all 8 corners kept these terms cancel (sum w_c x_c = x)
        u0 = ii[c, 0] - rc[0]
        u1 = ii[c, 1] - rc[1]
        u2 = ii[c, 2] - rc[2]
        un = np.sqrt(u0 * u0 + u1 * u1 + u2 * u2)
        ds = 0.0
        if un > 1e-9:
            ds = (
                u0 * (x[0] - ii[c, 0]) + u1 * (x[1] - ii[c, 1]) + u2 * (x[2] - ii[c, 2])
            ) / un
        sw += ws[c]
        sl += ws[c] * (length[ii[c, 0], ii[c, 1], ii[c, 2]] + ds)
        sr += ws[c] * rc
    return sl / sw, sr / sw


def click_column(g: Geometry, x: Float, same: float = 3.0) -> tuple[Float, Float]:
    """For points x (N, 3): the carried root position and arc length, interpolated
    trilinearly over the reached voxels whose roots agree (within `same` voxels; see
    _click)."""
    x = np.atleast_2d(np.asarray(x, np.float64))
    L = np.ascontiguousarray(g.length, np.float64)
    Rt = np.ascontiguousarray(g.root, np.float64)
    out = [_click(L, Rt, p, same) for p in x]
    return np.array([o[1] for o in out]), np.array([o[0] for o in out])


# ------------------------------------------------------------------ mesh sections (figures)


def plane_section(v: Float, f: Int, axis: int, c: float) -> Float:
    """Segments (M, 2, 3) where the mesh crosses coordinate axis = c."""
    d = v[:, axis] - c
    df = d[f]
    pos = df > 0
    cross = pos.any(1) & ~pos.all(1)
    segs = []
    for tri, dd in zip(f[cross], df[cross]):
        pts = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if (dd[i] > 0) != (dd[j] > 0):
                w = dd[i] / (dd[i] - dd[j])
                pts.append(v[tri[i]] + w * (v[tri[j]] - v[tri[i]]))
        if len(pts) == 2:
            segs.append(pts)
    return np.array(segs).reshape(-1, 2, 3)
