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


def signed_wm_direction(wm: Bool, vox: npt.ArrayLike, sigma: float = 1.0) -> F64:
    """The unit gradient (..., 3) of the signed WM distance (mm, positive
    outside the WM) smoothed by sigma voxels: away from the WM's edge, on
    both sides of it."""
    v = np.asarray(vox, np.float64)
    sd = ndi.distance_transform_edt(~wm, sampling=v) - ndi.distance_transform_edt(
        wm, sampling=v
    )
    g = np.stack(gradient(ndi.gaussian_filter(sd, sigma), v), -1)
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
    fluid: Bool | None = None,
    wm_band_mm: float = 0.0,
) -> tuple[F64, dict[str, Any]]:
    """T on the full grid (mm) and the solver's info.

    The domain is the cortex and the WM voxels 6-adjacent to it; T = 0 on the
    WM ones. Over every 6-neighbour edge of the domain, T1 - T0 is fitted to
    F_a h_a (F = oriented_normal of the tensor's normal, its coherence zero
    outside the cortex, at the edge midpoint; h_a the voxel size along the
    edge). Outside the domain T is the nearest domain value.

    Extended (fluid given, or wm_band_mm > 0): the domain also takes the
    fluid voxels 6-adjacent to the cortex and the WM voxels within wm_band_mm
    of the WM's edge, so that grad T is T's own up to the fluid and into the
    WM (with the nearest-value fill alone, it flattens within a voxel of the
    domain's edge). There the data has no layers (coherence 0), so F is the
    unit gradient of the smoothed signed WM distance (positive outside the
    WM), and T is negative in the WM band.
    """
    v = np.asarray(vox, np.float64)
    n, c = eigen(tensor)
    c = np.where(ctx, c, 0.0)
    near = ndi.binary_dilation(ctx)
    extended = fluid is not None or wm_band_mm > 0
    g = signed_wm_direction(wm, v) if extended else wm_direction(wm, v)
    Fv = oriented_normal(n, c, g, lam)
    dom = ctx | (wm & near)
    if fluid is not None:
        dom |= fluid & near
    if wm_band_mm > 0:
        dom |= wm & (ndi.distance_transform_edt(wm, sampling=v) <= wm_band_mm)
    fixed = wm & near
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


#: The 13 lattice directions (one of each opposite pair of the 26 neighbours).
_DIR13 = [
    (a, b, c)
    for a in (-1, 0, 1)
    for b in (-1, 0, 1)
    for c in (-1, 0, 1)
    if (a, b, c) > (0, 0, 0)
]


def eigenvalues(tensor: F64, chunk: int = 2_000_000) -> F32:
    """The eigenvalues (..., 3), ascending, of a tensor field (6, ...)."""
    shp = tensor.shape[1:]
    flat = tensor.reshape(6, -1)
    out = np.zeros((flat.shape[1], 3), np.float32)
    for a in range(0, flat.shape[1], chunk):
        out[a : a + chunk] = np.linalg.eigvalsh(_matrices(flat[:, a : a + chunk]))
    return out.reshape(*shp, 3)


def n_eff(sd: float = SD, si: float = SI) -> float:
    """The number of independent samples of gradient noise in the structure
    tensor's integration window: 1 / sum_t (w * w)(t) rho(t), w the Gaussian
    window of si voxels (sum 1), rho the normalized autocorrelation of one
    component of the gradient (central differences) of white noise smoothed
    by a Gaussian of sd voxels. Exact for the discrete filters."""
    m = int(np.ceil(4 * (sd + si))) + 2
    d = np.zeros((2 * m + 1,) * 3)
    d[m, m, m] = 1.0
    r = np.gradient(ndi.gaussian_filter(d, sd), axis=0)
    rho = np.real(np.fft.ifftn(np.abs(np.fft.fftn(r)) ** 2))
    rho /= rho.flat[0]
    w = ndi.gaussian_filter(d, si)
    w /= w.sum()
    ww = np.real(np.fft.ifftn(np.abs(np.fft.fftn(w)) ** 2))
    return float(1.0 / np.sum(ww * rho))


def step_noise(
    tensor: F64,
    ctx: Bool,
    g: F64,
    vox: npt.ArrayLike,
    lam: float = 0.1,
    sd: float = SD,
    si: float = SI,
) -> tuple[F64, float]:
    """The predicted noise (standard deviation, mm) of T's change over one
    voxel step, per voxel, and the gradient noise power, both from the
    structure tensor itself (no calibration).

    Noise in the image adds its gradient power sigma2 to each of the
    tensor's eigenvalues; the layers add their gradient power S to the
    largest only; the cortex's curvature and partial volume move power into
    the middle one. So sigma2 is the median smallest eigenvalue over the
    cortex, and S = lam1 - sigma2. The leading eigenvector's angular error
    (first-order perturbation) has variance sigma2 (S + sigma2) / (S^2 N),
    N = n_eff(sd, si). F = c n sgn + lam g (normalized, g the WM direction)
    carries that error times c / |c n sgn + lam g|. T, the least-squares
    potential of F, keeps the curl-free part of F's error; for an isotropic
    error, one component of it carries a ninth of the variance, so a step
    of h (the mean voxel size) varies with h |dF| / 3. T is solved over the
    whole domain, so the variance is spread over F's correlation length (a
    Gaussian of si voxels)."""
    h = float(np.mean(np.asarray(vox, np.float64)))
    lam_ = eigenvalues(tensor).astype(np.float64)
    sigma2 = float(np.median(lam_[..., 0][ctx]))
    S_ = np.maximum(lam_[..., 2] - sigma2, 1e-12 * max(sigma2, 1e-30))
    ang2 = sigma2 * (S_ + sigma2) / (S_**2 * n_eff(sd, si))
    n, c = eigen(tensor)
    c = np.where(ctx, c, 0.0)
    sgn = np.sign((n * g).sum(-1))
    sgn[sgn == 0] = 1
    Fn = np.linalg.norm(c[..., None] * n * sgn[..., None] + lam * g, axis=-1)
    gain = c / np.maximum(Fn, 1e-12)
    var = np.where(ctx, (h * gain) ** 2 * ang2 / 9.0, 0.0)
    out: F64 = np.sqrt(ndi.gaussian_filter(var, si))
    return out, sigma2


def crest_voxels(
    T: F64,
    mask: Bool,
    noise: F64,
    kappa: F64,
    vox: npt.ArrayLike,
    alpha: float = 1e-3,
    free: Bool | None = None,
) -> Bool:
    """The voxels of mask where two rises of T meet in a kink: along some
    lattice direction e (of the 26 neighbours; L its length in mm), T rises
    significantly into the voxel, does not rise significantly beyond it, and
    its slope changes across the voxel by more than a smooth level set's
    curvature allows:
      a = T(x) - T(x - e) > k s,  b = T(x + e) - T(x) < k s,
      a - b > L^2 kappa(x) + sqrt(2) k s,
    s = noise(x) L / h (step_noise, for a step of L), kappa the curvature of
    T's level sets (1/mm), k the one-sided normal quantile for alpha / 26
    (alpha: false alarms per voxel).

    In one bank T rises steadily along its flow (b > k s). A straight line
    tangent to a curved level set has a smooth maximum of T at the tangent
    point, but there the slope changes across the voxel by at most L^2 kappa
    (the second difference of a smooth field). At a kink, where two banks'
    rises meet (a contact), the slope changes by L (cos a1 + cos a2), the
    angles at which e meets the two banks' gradients, however short the
    step. free: the voxels where T was solved for (not pinned at 0 on the
    WM's edge, not filled outside the domain; fitted_free); x - e, x and
    x + e must all be free, since the pinned staircase and the fill make
    voxel-scale kinks of their own. Box-edge voxels are not tested."""
    from statistics import NormalDist

    v = np.asarray(vox, np.float64)
    h = float(np.mean(v))
    k = NormalDist().inv_cdf(1.0 - alpha / 26.0)
    out = np.zeros(T.shape, bool)
    c = (slice(1, -1),) * 3
    t0, n0, kap = T[c], noise[c], kappa[c]
    fr = np.ones(T.shape, bool) if free is None else free
    f0 = fr[c]
    for e in _DIR13:
        L = float(np.linalg.norm(np.asarray(e, np.float64) * v))
        ks = k * n0 * L / h
        curv = L**2 * kap
        fw = tuple(slice(1 + d, T.shape[i] - 1 + d) for i, d in enumerate(e))
        bw = tuple(slice(1 - d, T.shape[i] - 1 - d) for i, d in enumerate(e))
        tf, tb = T[fw], T[bw]
        ok = f0 & fr[fw] & fr[bw]
        for up, dn in ((tb, tf), (tf, tb)):
            a_ = t0 - up
            b_ = dn - t0
            out[c] |= ok & (a_ > ks) & (b_ < ks) & (a_ - b_ > curv + np.sqrt(2.0) * ks)
    crest: Bool = out & mask
    return crest


def fitted_free(
    ctx: Bool,
    wm: Bool,
    vox: npt.ArrayLike,
    fluid: Bool | None = None,
    wm_band_mm: float = 0.0,
) -> Bool:
    """The voxels fitted_field solves T for (with the same domain options):
    its domain without the WM voxels 6-adjacent to the cortex, where T is
    pinned at 0."""
    v = np.asarray(vox, np.float64)
    near = ndi.binary_dilation(ctx)
    dom = ctx | (wm & near)
    if fluid is not None:
        dom |= fluid & near
    if wm_band_mm > 0:
        dom |= wm & (ndi.distance_transform_edt(wm, sampling=v) <= wm_band_mm)
    out: Bool = dom & ~(wm & near)
    return out


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
    turn_deg: float | None = 60.0,
    turn_mm: float = 0.3,
) -> Traces:
    """Streamlines of the leading eigenvector of `tensor` from x0 (n, 3),
    started along d0 (n, 3; the mesh normal). blend_mm None: the data's
    direction from the first step; otherwise blended in from the mesh normal
    (module docstring). turn_deg None: no TURN stop. Returns positions,
    sample counts and outcomes."""
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
    cos_turn = -1.0 if turn_deg is None else np.cos(np.radians(turn_deg))

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
    step_mm: float = STEP_MM,
    face_mm: float = 2.0,
    wm_after: float = 0.3,
    turn_deg: float | None = None,
    turn_mm: float = 0.3,
    far: Callable[[I64, I64], Bool] | None = None,
) -> Traces:
    """Paths up grad u from x0 (n, 3), unit steps of step_mm in mm. RIDGE
    where grad u vanishes or u decreases (after wm_after mm): the last
    sample is then the first one past the maximum. With turn_deg, TURN where
    the direction turned more than turn_deg within turn_mm, as trace: where
    the path reaches a crest of u that rises along its length, it turns to
    follow the crest instead of stopping. far(i_paths, q) -> bool per path,
    if given: the voxel q (rounded positions) belongs to a root far from that
    path's own (CONTACT). Arrays as trace."""
    g = np.stack(gradient(u), 0)
    x = np.asarray(x0, np.float64).copy()
    n = x.shape[0]
    v = np.asarray(vox, np.float64)
    pos = np.full((n, nmax, 3), np.nan, np.float32)
    pos[:, 0] = x
    nsamp = np.ones(n, np.int64)
    out = np.full(n, RUNNING, np.int64)
    prev = ndi.map_coordinates(u, x.T, order=1, mode="nearest")
    act = np.arange(n)
    lag = max(1, int(round(turn_mm / step_mm)))
    hist = np.zeros((lag + 1, n, 3))
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
        ui = ndi.map_coordinates(u, xa.T, order=1, mode="nearest")
        code = _stop_codes(xa, labels, dface, i * step_mm, face_mm, wm_after)
        if far is not None:
            q = np.clip(np.round(xa).astype(int), 0, np.array(labels.shape) - 1)
            code[(code == RUNNING) & far(act, q)] = CONTACT
        # a label stop wins over a vanished gradient, which wins over a face
        flat = (nv[:, 0] < 1e-9) & ((code == RUNNING) | (code == FACE))
        code[flat] = RIDGE
        fell = (ui < prev[act] - 1e-6) & (i * step_mm > wm_after)
        code[(code == RUNNING) & fell] = RIDGE
        prev[act] = ui
        if turn_deg is not None:
            dm = dirn * v  # unit direction in mm
            old = hist[i % (lag + 1)][act]
            hist[i % (lag + 1)][act] = dm
            if i > lag:
                turned = (old * dm).sum(1) < np.cos(np.radians(turn_deg))
                code[(code == RUNNING) & turned] = TURN
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
    return ascend(-u, labels, dface, x0, vox, nmax, step_mm, face_mm, 0.0)


def sample(img: npt.ArrayLike, pos: F32, chunk: int = 20000) -> F32:
    """img at every sample of paths (n, m, 3), trilinear (nearest value
    outside the grid); nan where pos is nan. (n, m) float32."""
    n, m, _ = pos.shape
    im = np.asarray(img, np.float64)
    out = np.full((n, m), np.nan, np.float32)
    for a in range(0, n, chunk):
        p = pos[a : a + chunk].reshape(-1, 3)
        ok = np.isfinite(p[:, 0])
        val = np.full(p.shape[0], np.nan, np.float32)
        val[ok] = ndi.map_coordinates(
            im, p[ok].T.astype(float), order=1, mode="nearest"
        )
        out[a : a + chunk] = val.reshape(-1, m)
    return out
