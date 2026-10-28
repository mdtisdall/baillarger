"""Column streamlines along the data's layer normal: the direction fields and
the integrators that trace them.

A column is traced from a root on the white mesh, outward along the normal of
the laminae. The laminae are read from the data, not from the WM distance (as
baillarger.frame does): the structure tensor of the trend-removed echoes has
its leading eigenvector across the layers, and its coherence says how clearly.
Positions are voxel index coordinates; vox is the voxel size (mm, per axis);
arc length is in mm. Everything is 3D.

Direction fields
- structure_tensor: the gradient outer products of an image, each smoothed by
  a Gaussian of si voxels over a mask (masked_smooth: normalized convolution),
  after the image itself is smoothed by sd voxels. The six components are
  (xx, xy, xz, yy, yz, zz). Summed over the two trend-removed echoes, this is
  the base tensor.
- eigen: the leading unit eigenvector n and the coherence
  c = (w3 - w2) / (w3 + w2) of a tensor field.
- oriented_normal: F = c n sgn + lam g, normalized, where g is the unit
  gradient of the smoothed WM distance (wm_direction) and sgn the sign of
  n . g. The data's normal, signed to point away from the nearest WM, with g
  as a weak prior where the data are not coherent.
- fitted_field: T with T = 0 on the WM voxels next to the cortex and grad T
  fitted in least squares to F over the 6-neighbour edges of the cortex
  ("columns as the characteristics of a fitted field"). Its gradient is one
  smooth field, so its streamlines cannot cross. No top boundary is needed:
  in fused cortex, where two banks touch, F flips where g does, so T has a
  ridge between the banks. Solved by smoothed-aggregation multigrid (pyamg)
  with conjugate gradients on the normal equations; deterministic (the same
  T on every run).
- gradient_tensor: the rank-1 tensor u u^T of the unit gradient of a field,
  so that the unsigned integrator follows the field and passes its ridges.

Integrators (many paths at once; positions (n, nmax, 3) float32, nan past
each end, sample i at arc length i * step_mm; the number of samples; and the
outcome code)
- trace: midpoint (RK2) steps along the leading eigenvector of a tensor
  field, interpolated trilinearly component by component, its sign chosen to
  agree with the previous step. With blend_mm, the direction near the root
  is a blend w d_data + (1 - w) n_mesh, w = min(1, c s / blend_mm) (s the arc
  length, c the interpolated tensor's coherence): the mesh normal at the root,
  the data's direction after blend_mm / c. This keeps the start off the
  partial-volume edge of the white surface, where the data tensor is
  unreliable.
- ascend: steps up the gradient of a scalar field, the fallback where the
  primary trace turns; descend: steps down it, from a point in the cortex to
  the WM.

Stops (outcome codes): WM (after wm_after mm), FLUID, MISSING (missing
tissue), OTHER labels, FACE (within face_mm of a box face, or outside),
TURN (the direction turned more than turn_deg within turn_mm: the trace has
left the column, usually at a fused stretch or a fundus), LENGTH (nmax
samples). ascend adds CONTACT (a caller's test: the voxel belongs to a root
far from the path's own) and RIDGE (the field stops increasing).

Labels: 0 cortex, 1 fluid, 2 WM, 3 missing tissue, other values other.

Validated (reports/Crossing paths and contacts from laminar profiles.md,
*Column traces at folds*) on fold phantoms with known columns (crowns,
fundi, touching banks) at the measured noise: the fitted field with the
blended start (column_traces.TRACERS "fitted_blend") is the most accurate at
tight fundi and at contacts.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pyamg
from scipy import ndimage as ndi
from scipy import sparse

type Bool = npt.NDArray[np.bool_]
type F32 = npt.NDArray[np.float32]
type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]
type Traces = tuple[F32, I64, I64]

#: Outcome codes of trace and ascend.
RUNNING, WM, FLUID, MISSING, FACE, OTHER, TURN, LENGTH, CONTACT, RIDGE = range(10)
OUTCOME = (
    "running",
    "WM",
    "fluid",
    "missing",
    "face",
    "other",
    "turn",
    "length",
    "contact",
    "ridge",
)
#: Sample spacing along the paths (mm).
STEP_MM = 0.04
#: The structure tensor's derivative and integration scales (voxels).
SD, SI = 1.0, 2.0
_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


# ---------------------------------------------------------------- fields


def gradient(f: npt.ArrayLike, spacing: npt.ArrayLike = (1.0, 1.0, 1.0)) -> list[F64]:
    """np.gradient along each of the three axes (central differences, one-sided
    at the edges), with the spacing of each axis."""
    sp = np.asarray(spacing, np.float64)
    return [
        np.asarray(np.gradient(f, float(sp[k]), axis=k), np.float64) for k in range(3)
    ]


def masked_smooth(x: npt.ArrayLike, mask: npt.ArrayLike, sigma: float) -> F64:
    """Normalized convolution: G(x m) / G(m), G a Gaussian of sigma voxels."""
    m = np.asarray(mask, np.float64)
    num = ndi.gaussian_filter(np.asarray(x, np.float64) * m, sigma)
    out: F64 = num / np.maximum(ndi.gaussian_filter(m, sigma), 1e-6)
    return out


def structure_tensor(
    img: npt.ArrayLike, mask: npt.ArrayLike, sd: float = SD, si: float = SI
) -> F64:
    """The six components (6, ...) of the structure tensor of img over mask:
    the gradient of the image smoothed by sd, its outer products smoothed by
    si (voxels; both masked_smooth)."""
    g = np.stack(gradient(masked_smooth(img, mask, sd)), -1)
    return np.stack(
        [masked_smooth(g[..., i] * g[..., j], mask, si) for i, j in _PAIRS], 0
    )


def _matrices(t: npt.NDArray[Any]) -> F64:
    """Symmetric 3 x 3 matrices (n, 3, 3) from six components (6, n)."""
    M = np.empty((t.shape[1], 3, 3))
    M[:, 0, 0], M[:, 0, 1], M[:, 0, 2] = t[0], t[1], t[2]
    M[:, 1, 1], M[:, 1, 2], M[:, 2, 2] = t[3], t[4], t[5]
    M[:, 1, 0], M[:, 2, 0], M[:, 2, 1] = t[1], t[2], t[4]
    return M


def _coherence(w: F64) -> F64:
    c: F64 = np.clip((w[:, 2] - w[:, 1]) / np.maximum(w[:, 2] + w[:, 1], 1e-12), 0, 1)
    return c


def eigen(tensor: F64, chunk: int = 2_000_000) -> tuple[F32, F32]:
    """The leading unit eigenvector (..., 3) and the coherence (...) of a
    tensor field (6, ...), float32."""
    shp = tensor.shape[1:]
    flat = tensor.reshape(6, -1)
    n = flat.shape[1]
    vec = np.zeros((n, 3), np.float32)
    coh = np.zeros(n, np.float32)
    for a in range(0, n, chunk):
        w, V = np.linalg.eigh(_matrices(flat[:, a : a + chunk]))
        vec[a : a + chunk] = V[:, :, 2]
        coh[a : a + chunk] = _coherence(w)
    return vec.reshape(*shp, 3), coh.reshape(shp)


def wm_direction(wm: Bool, vox: npt.ArrayLike, sigma: float = 1.0) -> F64:
    """The unit gradient (..., 3) of the WM distance (mm) smoothed by sigma
    voxels: the direction away from the nearest WM."""
    v = np.asarray(vox, np.float64)
    d_wm = ndi.distance_transform_edt(~wm, sampling=v)
    g = np.stack(gradient(ndi.gaussian_filter(d_wm, sigma), v), -1)
    out: F64 = g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-12)
    return out


def oriented_normal(
    n: npt.NDArray[np.floating[Any]],
    coh: npt.NDArray[np.floating[Any]],
    g: F64,
    lam: float = 0.1,
) -> F64:
    """F = c n sgn + lam g, normalized (sgn the sign of n . g, 1 where it is 0)."""
    sgn = np.sign((n * g).sum(-1))
    sgn[sgn == 0] = 1
    F = coh[..., None] * n * sgn[..., None] + lam * g
    out: F64 = F / (np.linalg.norm(F, axis=-1, keepdims=True) + 1e-12)
    return out


def _fill(T: F64) -> F64:
    """Nearest-value fill of the nan voxels, for gradients and interpolation."""
    bad = np.isnan(T)
    if not bad.any():
        return T
    _, ind = ndi.distance_transform_edt(bad, return_indices=True)
    out: F64 = T[tuple(ind)]
    return out


#: The prolongation smoother of the multigrid: Jacobi weighted by local
#: (Gershgorin) bounds. pyamg's default weighting estimates a spectral radius
#: from a random start vector (numpy's global generator), so its solutions
#: differ from run to run within the tolerance; this one is deterministic.
_SMOOTH = ("jacobi", {"omega": 4.0 / 3.0, "degree": 1, "weighting": "local"})


def fitted_field(
    tensor: F64,
    ctx: Bool,
    wm: Bool,
    vox: npt.ArrayLike,
    lam: float = 0.1,
    tol: float = 1e-8,
) -> tuple[F64, dict[str, Any]]:
    """T on the full grid (mm) and the solver's info.

    The domain is the cortex and the WM voxels 6-adjacent to it; T = 0 on the
    WM ones. Over every 6-neighbour edge of the domain, T1 - T0 is fitted to
    F_a h_a (F = oriented_normal of the tensor's normal, its coherence zero
    outside the cortex, at the edge midpoint; h_a the voxel size along the
    edge). Outside the domain T is the nearest domain value.
    """
    v = np.asarray(vox, np.float64)
    n, c = eigen(tensor)
    c = np.where(ctx, c, 0.0)
    Fv = oriented_normal(n, c, wm_direction(wm, v), lam)
    dom = ctx | (wm & ndi.binary_dilation(ctx))
    fixed = dom & wm
    idx = -np.ones(dom.shape, np.int64)
    nd = int(dom.sum())
    idx[dom] = np.arange(nd)
    free = ~fixed[dom]
    fmap = -np.ones(nd, np.int64)
    fmap[free] = np.arange(free.sum())
    rows, cols, vals, rhs = [], [], [], []
    r = 0
    for a in range(3):
        sl0: list[slice] = [slice(None)] * 3
        sl1: list[slice] = [slice(None)] * 3
        sl0[a], sl1[a] = slice(0, -1), slice(1, None)
        i0, i1 = idx[tuple(sl0)], idx[tuple(sl1)]
        ok = (i0 >= 0) & (i1 >= 0)
        pos = np.argwhere(ok)
        i0, i1 = i0[ok], i1[ok]
        p1 = pos.copy()
        p1[:, a] += 1
        f = 0.5 * (Fv[..., a][tuple(pos.T)] + Fv[..., a][tuple(p1.T)]) * v[a]
        rr = np.arange(r, r + len(i0))
        for ii, s_ in ((i1, 1.0), (i0, -1.0)):
            fr = fmap[ii]
            keep = fr >= 0
            rows.append(rr[keep])
            cols.append(fr[keep])
            vals.append(np.full(keep.sum(), s_))
        rhs.append(f)
        r += len(i0)
    A = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(r, int(free.sum())),
    )
    b = np.concatenate(rhs)
    N = (A.T @ A + 1e-6 * sparse.identity(A.shape[1])).tocsr()
    ml = pyamg.smoothed_aggregation_solver(N, symmetry="symmetric", smooth=_SMOOTH)
    res: list[float] = []
    t = ml.solve(A.T @ b, tol=tol, accel="cg", maxiter=500, residuals=res)
    T = np.full(dom.shape, np.nan)
    tv = np.zeros(nd)
    tv[free] = t
    T[dom] = tv
    info = dict(
        solver="pyamg sa+cg",
        iters=len(res),
        resid=float(res[-1] / max(res[0], 1e-30)),
    )
    return _fill(T), info


def gradient_tensor(T: F64, vox: npt.ArrayLike) -> F64:
    """The rank-1 tensor u u^T (6, ...) of u, the unit gradient of T (mm)."""
    g = np.stack(gradient(T, vox), -1)
    u = g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-12)
    return np.stack([u[..., i] * u[..., j] for i, j in _PAIRS], 0)


# ---------------------------------------------------------------- integrators


def tensor_at(tensor: F64, x: F64) -> F64:
    """3 x 3 tensors (n, 3, 3) at positions x (n, 3), trilinear in each
    component (nearest value outside the grid)."""
    t = np.stack(
        [ndi.map_coordinates(c, x.T, order=1, mode="nearest") for c in tensor], 0
    )
    return _matrices(t)


def _flip(d: F64, ref: F64) -> F64:
    out: F64 = d * np.where((d * ref).sum(1) < 0, -1.0, 1.0)[:, None]
    return out


def _stop_codes(
    xa: F64,
    labels: npt.NDArray[np.integer[Any]],
    dface: F64,
    s_mm: float,
    face_mm: float,
    wm_after: float,
) -> I64:
    """Outcome per active path at its new position (RUNNING if none)."""
    shape = np.array(labels.shape)
    q = np.clip(np.round(xa).astype(int), 0, shape - 1)
    lab = labels[tuple(q.T)]
    inside = np.all((xa >= 0) & (xa <= shape - 1), 1)
    code = np.full(len(xa), RUNNING)
    code[(lab == 2) & (s_mm > wm_after)] = WM
    code[lab == 1] = FLUID
    code[lab == 3] = MISSING
    code[(lab > 3) | (lab < 0)] = OTHER
    code[(code == RUNNING) & ((dface[tuple(q.T)] < face_mm) | ~inside)] = FACE
    return code


def trace(
    tensor: F64,
    labels: npt.NDArray[np.integer[Any]],
    dface: F64,
    x0: npt.ArrayLike,
    d0: npt.ArrayLike,
    vox: npt.ArrayLike,
    nmax: int,
    blend_mm: float | None = None,
    step_mm: float = STEP_MM,
    face_mm: float = 2.0,
    wm_after: float = 0.3,
    turn_deg: float = 60.0,
    turn_mm: float = 0.3,
) -> Traces:
    """Streamlines of the leading eigenvector of `tensor` from x0 (n, 3),
    started along d0 (n, 3; the mesh normal). blend_mm None: the data's
    direction from the first step; otherwise blended in from the mesh normal
    (module docstring). Returns positions, sample counts and outcomes."""
    x = np.asarray(x0, np.float64).copy()
    n = x.shape[0]
    d_in = np.asarray(d0, np.float64)
    nm = d_in / np.linalg.norm(d_in, axis=1, keepdims=True)
    d = nm.copy()
    h = step_mm / np.asarray(vox, np.float64)
    pos = np.full((n, nmax, 3), np.nan, np.float32)
    pos[:, 0] = x
    nsamp = np.ones(n, np.int64)
    out = np.full(n, RUNNING, np.int64)
    lag = max(1, int(round(turn_mm / step_mm)))
    hist = np.zeros((lag + 1, n, 3))
    hist[:] = d
    act = np.arange(n)
    cos_turn = np.cos(np.radians(turn_deg))

    def direction(xa: F64, ref: F64, s: float, nma: F64) -> F64:
        w, V = np.linalg.eigh(tensor_at(tensor, xa))
        v = _flip(V[:, :, 2], ref)
        if blend_mm is None:
            return v
        b = np.minimum(1.0, _coherence(w) * s / blend_mm)[:, None]
        dd = b * v + (1 - b) * nma
        out_: F64 = dd / np.maximum(np.linalg.norm(dd, axis=1, keepdims=True), 1e-12)
        return out_

    for i in range(1, nmax):
        if not act.size:
            break
        xa, da, nma = x[act], d[act], nm[act]
        s = (i - 1) * step_mm
        d1 = direction(xa, da, s, nma)
        d2 = direction(xa + 0.5 * d1 * h, d1, s + 0.5 * step_mm, nma)
        xa = xa + d2 * h
        x[act], d[act] = xa, d2
        pos[act, i] = xa
        nsamp[act] = i + 1
        code = _stop_codes(xa, labels, dface, i * step_mm, face_mm, wm_after)
        old = hist[i % (lag + 1)][act]  # the direction lag steps ago
        hist[i % (lag + 1)][act] = d2
        if i > lag:
            turned = np.abs((old * d2).sum(1)) < cos_turn
            code[(code == RUNNING) & turned] = TURN
        if i == nmax - 1:
            code[code == RUNNING] = LENGTH
        out[act] = code
        act = act[code == RUNNING]
    return pos, nsamp, out


def ascend(
    u: F64,
    labels: npt.NDArray[np.integer[Any]],
    dface: F64,
    x0: npt.ArrayLike,
    vox: npt.ArrayLike,
    nmax: int,
    far: Callable[[I64, I64], Bool] | None = None,
    step_mm: float = STEP_MM,
    face_mm: float = 2.0,
    wm_after: float = 0.3,
) -> Traces:
    """Paths up grad u from x0 (n, 3), unit steps of step_mm in mm.
    far(i_paths, q) -> bool per path, if given: the voxel q (rounded
    positions) belongs to a root far from that path's own (CONTACT). RIDGE
    where grad u vanishes or u decreases (after wm_after mm). Arrays as
    trace."""
    g = np.stack(gradient(u), 0)
    x = np.asarray(x0, np.float64).copy()
    n = x.shape[0]
    v = np.asarray(vox, np.float64)
    shape = np.array(labels.shape)
    pos = np.full((n, nmax, 3), np.nan, np.float32)
    pos[:, 0] = x
    nsamp = np.ones(n, np.int64)
    out = np.full(n, RUNNING, np.int64)
    prev = ndi.map_coordinates(u, x.T, order=1, mode="nearest")
    act = np.arange(n)
    for i in range(1, nmax):
        if not act.size:
            break
        xa = x[act]
        gv = np.stack(
            [ndi.map_coordinates(c, xa.T, order=1, mode="nearest") for c in g], 1
        )
        nv = np.linalg.norm(gv * (1.0 / v), axis=1, keepdims=True)  # in mm
        dirn = gv / v / np.maximum(nv, 1e-12)
        xa = xa + dirn * step_mm / v
        x[act] = xa
        pos[act, i] = xa
        nsamp[act] = i + 1
        q = np.clip(np.round(xa).astype(int), 0, shape - 1)
        ui = ndi.map_coordinates(u, xa.T, order=1, mode="nearest")
        code = _stop_codes(xa, labels, dface, i * step_mm, face_mm, wm_after)
        # a label stop wins over a vanished gradient, which wins over a face
        flat = (nv[:, 0] < 1e-9) & ((code == RUNNING) | (code == FACE))
        code[flat] = RIDGE
        if far is not None:
            code[(code == RUNNING) & far(act, q)] = CONTACT
        fell = (ui < prev[act] - 1e-6) & (i * step_mm > wm_after)
        code[(code == RUNNING) & fell] = RIDGE
        prev[act] = ui
        if i == nmax - 1:
            code[code == RUNNING] = LENGTH
        out[act] = code
        act = act[code == RUNNING]
    return pos, nsamp, out


def descend(
    u: F64,
    labels: npt.NDArray[np.integer[Any]],
    dface: F64,
    x0: npt.ArrayLike,
    vox: npt.ArrayLike,
    nmax: int,
    step_mm: float = STEP_MM,
    face_mm: float = 2.0,
) -> Traces:
    """Paths down grad u from x0 (n, 3): ascend on -u, stopping on the first
    sample in the WM (no wm_after). For the fitted field T (0 on the WM),
    the path from a point in the cortex to its root. RIDGE where u stops
    decreasing (a minimum away from the WM). Arrays as trace."""
    return ascend(-u, labels, dface, x0, vox, nmax, None, step_mm, face_mm, 0.0)
