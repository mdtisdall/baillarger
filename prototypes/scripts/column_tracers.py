"""Column tracers for the column volume, the current one and the candidates for folds (plan:
reports/Crossing paths and contacts from laminar profiles.md, *Column traces at folds*). Pure
functions, no argv; all 3D. Positions in voxel index coordinates; vox the voxel size (mm).

Every tracer returns arrays as crossing_lib.trace_array (positions (n, nmax, 3) float32, the
number of samples, the outcome code) from roots x0 started along d0 (the white mesh's normal).

- current: crossing_lib.trace_array on the summed structure tensor of the two trend-removed echoes
  (flatten_fused2.tensor: derivative scale SD, integration scale SI voxels).
- (i) laplace: streamlines of grad phi, phi the anisotropic Laplace potential between the WM
  (phi = 0) and the fluid (phi = 1), div(D grad phi) = 0 in the cortex, D = I - c n n^T (n the
  data's layer normal, c its coherence): conduction along the layers is 1 and across them 1 - c,
  so phi equalizes along the data's laminae where they are coherent and is the plain Laplace
  potential (Jones et al. 2000) where they are not. Its gradient is a smooth field, so its
  streamlines cannot cross. In fused cortex phi rises from both banks' WM to a ridge; the trace
  follows the unsigned direction (rank-1 tensor u u^T), so it passes the ridge into the far bank.
- (i-b) fitted: streamlines of grad T, T = 0 on all WM and grad T fitted in least squares to the
  oriented data normal F = c n sgn + 0.1 g (normalized; g the unit gradient of the smoothed WM
  distance, sgn the sign of n . g): curv_lib.fit_all_wm's field, the main plan's "columns as
  characteristics of a fitted field". It needs no top: in fused cortex F flips where g does, so T
  has a ridge between the banks, which the unsigned trace passes as (i)'s.
- (ii) multiscale: the tensor at SI and at 2 SI, each normalized by its trace, blended by the
  fine scale's coherence: T = c T_SI + (1 - c) T_2SI.
- (iii) blend: trace_array's integrator, with the direction a blend of the data's normal and the
  mesh normal, d = w d_data + (1 - w) n_mesh, w = min(1, c s / blend_mm) (s the arc length): the
  mesh normal at the root, the data's normal after blend_mm / c.
- fitted_blend: (i-b)'s field traced with (iii)'s start (its rank-1 tensor has coherence 1, so
  the mesh normal hands over to it within blend_mm).
- (iv) per bank: the tensor accumulated only over voxels on the same bank, the bank given by the
  direction away from the nearest WM (g, the unit gradient of the smoothed WM distance): with the
  weights w_k = max(0, g . e_k)^2 over the six axis directions (sum 1), T(x) = sum_k w_k(x)
  mconv(J w_k), so voxels whose g are opposite (the two banks of a contact) never mix.

current, fitted and fitted_blend, the white mesh, the fallback (ascent of the equal-arrival pass)
and the classes are the package's (baillarger.streamlines, baillarger.column_traces); this module
wraps them with the same calls as the candidates, which stay here for fold_eval.py and
column_diag.py.
"""

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.spatial import cKDTree

import crossing_lib as X
import flatten_fused2 as F
from baillarger import column_traces as PCT
from baillarger import streamlines as S

OPEN, CROSS, OWN, TOCONTACT, FAILED = range(5)
CLASS = dict(enumerate(PCT.CLASS))
TRACERS = ["current", "laplace", "fitted", "fitted_blend", "multiscale", "blend", "perbank"]


# ---------------------------------------------------------------- mesh (as column_volume.py)

def white_mesh(wm):
    return PCT.white_mesh(wm)


def pick_vertices(v, n):
    """One vertex per voxel cell and normal side (column_traces.root_vertices)."""
    return PCT.root_vertices(v, n)


# ---------------------------------------------------------------- tensors

def tensor(img, m, sd=F.SD, si=F.SI):
    """flatten_fused2.tensor with its scales as arguments."""
    g = np.stack(np.gradient(F.mconv(img, m, sd)), -1)
    return np.stack([F.mconv(g[..., i] * g[..., j], m, si)
                     for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))], 0)


def base_tensor(B):
    m = B["ctx"].astype(float)
    return F.tensor(B["det"][1], m) + F.tensor(B["det"][2], m)


def eig(Tc, chunk=2_000_000):
    """Leading unit eigenvector (..., 3) and coherence (w3 - w2) / (w3 + w2) of a tensor field."""
    return S.eigen(Tc, chunk)


def normalized(Tc):
    tr = Tc[0] + Tc[3] + Tc[5]
    return Tc / np.maximum(tr, 1e-12)[None]


def multiscale_tensor(B, base=None):
    """(ii): c T_SI + (1 - c) T_2SI, each normalized by its trace, c the fine scale's coherence."""
    m = B["ctx"].astype(float)
    T1 = normalized(base_tensor(B) if base is None else base)
    T2 = normalized(tensor(B["det"][1], m, si=2 * F.SI) + tensor(B["det"][2], m, si=2 * F.SI))
    _, c = eig(T1)
    return c[None] * T1 + (1 - c[None]) * T2


def wm_direction(wm, vox):
    """Unit gradient of the smoothed WM distance (away from the nearest WM)."""
    return S.wm_direction(wm, vox)


def perbank_tensor(B):
    """(iv): T(x) = sum_k w_k(x) mconv(J w_k)(x), w_k = max(0, g . e_k)^2 over the six axis
    directions, J the gradient outer products of each feature over the cortex."""
    m = B["ctx"].astype(float)
    g = wm_direction(B["wm"], B["vox"])
    grads = [np.stack(np.gradient(F.mconv(B["det"][e], m, F.SD)), -1) for e in (1, 2)]
    J = [sum(gr[..., i] * gr[..., j] for gr in grads) for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))]
    del grads
    out = np.zeros((6,) + m.shape)
    for a in range(3):
        for sgn in (1.0, -1.0):
            w = np.maximum(0.0, sgn * g[..., a]) ** 2
            mw = m * w
            if not mw.any():
                continue
            for k in range(6):
                out[k] += w * F.mconv(J[k], mw, F.SI)
    return out


def rank1(u):
    """The tensor u u^T (6 components) of a unit vector field (..., 3)."""
    return np.stack([u[..., i] * u[..., j] for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))], 0)


# ---------------------------------------------------------------- (i) the anisotropic Laplace potential

OFF27 = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)])


def _k(off):
    return (off[0] + 1) * 9 + (off[1] + 1) * 3 + (off[2] + 1)


def laplace_system(labels, nrm_field, coh, vox, floor=0.02):
    """The linear system of laplace over the cortex voxels: A (csr, symmetric positive definite),
    the right-hand side, the unknowns' mask, and the WM and fluid Dirichlet masks."""
    shape = labels.shape
    vox = np.asarray(vox, float)
    V = float(np.prod(vox))
    ctx = labels == 0
    near = ndi.binary_dilation(ctx, np.ones((3, 3, 3), bool))
    dirW = near & (labels == 2)
    dirF = near & (labels == 1)
    dom = ctx | dirW | dirF
    S = np.zeros((27,) + shape, np.float64)
    iso = np.maximum(1.0 - coh, floor) * dom
    # edges
    for a in range(3):
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[a], sl1[a] = slice(0, -1), slice(1, None)
        sl0, sl1 = tuple(sl0), tuple(sl1)
        both = dom[sl0] & dom[sl1]
        w = np.where(both, 0.5 * (iso[sl0] + iso[sl1]), 0.0) * V / vox[a] ** 2
        e = np.zeros(3, int)
        e[a] = 1
        S[13][sl0] += w
        S[13][sl1] += w
        S[_k(e)][sl0] -= w
        S[_k(-e)][sl1] -= w
    # cells: M = c (I - n n^T), averaged over the 8 corners of each cell
    M = [c_ * ((i == j) - nrm_field[..., i] * nrm_field[..., j]) for c_ in [coh]
         for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))]
    corners = np.array([(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    csl = lambda o: tuple(slice(o[a], shape[a] - 1 + o[a]) for a in range(3))  # noqa: E731
    cell_ok = np.ones(tuple(s - 1 for s in shape), bool)
    Mc = [np.zeros(cell_ok.shape) for _ in range(6)]
    for o in corners:
        cell_ok &= dom[csl(o)]
        for q in range(6):
            Mc[q] += M[q][csl(o)] / 8
    Mfull = {}
    for q, (i, j) in enumerate(((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))):
        Mfull[(i, j)] = Mfull[(j, i)] = np.where(cell_ok, Mc[q], 0.0)
    del Mc, M
    sgn = 2 * corners - 1  # s_a(corner)
    for ii, oi in enumerate(corners):
        for jj, oj in enumerate(corners):
            coef = np.zeros(cell_ok.shape)
            for a in range(3):
                for b in range(3):
                    coef += sgn[ii, a] * sgn[jj, b] * Mfull[(a, b)] / (16 * vox[a] * vox[b])
            S[_k(oj - oi)][csl(oi)] += V * coef
    del Mfull
    # the system over the unknowns (cortex), Dirichlet values moved to the right-hand side
    unk = ctx
    nid = -np.ones(shape, np.int64)
    nid[unk] = np.arange(unk.sum())
    phiD = np.where(dirF, 1.0, 0.0)
    P = np.argwhere(unk)
    rows, cols, vals = [], [], []
    rhs = np.zeros(len(P))
    for k, off in enumerate(OFF27):
        s = S[k][unk]
        nz = s != 0
        if not nz.any():
            continue
        q = P[nz] + off
        inb = np.all((q >= 0) & (q < np.array(shape)), 1)
        r = np.flatnonzero(nz)[inb]
        q = q[inb]
        sv = s[nz][inb]
        qi = nid[tuple(q.T)]
        u = qi >= 0
        rows.append(r[u])
        cols.append(qi[u])
        vals.append(sv[u])
        np.subtract.at(rhs, r[~u], sv[~u] * phiD[tuple(q[~u].T)])
    del S
    n = len(P)
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return A, rhs, unk, dirW, dirF


def laplace(labels, nrm_field, coh, vox, floor=0.02, tol=1e-8, verbose=False):
    """phi with div(D grad phi) = 0 on the cortex (labels 0), phi = 0 on WM (2) and 1 on fluid (1)
    voxels 26-adjacent to the cortex, no flux elsewhere (missing, other, box faces).
    D = (1 - c) I + c (I - n n^T): the isotropic part on the 6-neighbour edges (its weight floored
    at `floor`, a numerical guard against checkerboard modes where c -> 1), the rest on 2x2x2
    cells from the cell-centre gradient. Finite-element-like energy, so the matrix is symmetric
    positive definite. Returns phi on the full grid (Dirichlet values on their voxels, nearest
    domain value elsewhere) and the solver's info."""
    shape = labels.shape
    A, rhs, unk, dirW, dirF = laplace_system(labels, nrm_field, coh, vox, floor)
    n = A.shape[0]
    A = A + sparse.identity(n) * (1e-9 * A.diagonal().mean())  # isolated pieces stay solvable
    try:
        import pyamg

        ml = pyamg.smoothed_aggregation_solver(A, symmetry="symmetric")
        res = []
        x = ml.solve(rhs, tol=tol, accel="cg", maxiter=500, residuals=res)
        info = dict(solver="pyamg sa+cg", iters=len(res), resid=float(res[-1] / max(res[0], 1e-30)))
    except ImportError:
        from scipy.sparse.linalg import cg

        Dg = sparse.diags(1.0 / A.diagonal())
        x, it = cg(A, rhs, rtol=tol, maxiter=20000, M=Dg)
        info = dict(solver="cg jacobi", flag=int(it))
    if verbose:
        print(info, flush=True)
    phi = np.full(shape, np.nan)
    phi[unk] = x
    phi[dirW] = 0.0
    phi[dirF] = 1.0
    bad = np.isnan(phi)
    _, ind = ndi.distance_transform_edt(bad, return_indices=True)
    return phi[tuple(ind)], info


def laplace_tensor(B, labels, Tc=None):
    """(i): the rank-1 tensor of the unit gradient of the anisotropic Laplace potential, the
    potential, and the solver's info."""
    Tc = base_tensor(B) if Tc is None else Tc
    n, c = eig(Tc)
    c = np.where(labels == 0, c, 0.0)
    phi, info = laplace(labels, n, c, B["vox"])
    g = np.stack(np.gradient(phi, *B["vox"]), -1)
    u = g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-12)
    return rank1(u), phi, info


# ---------------------------------------------------------------- (i-b) the fitted field

def fitted(B, Tc=None, lam=0.1, tol=1e-8):
    """streamlines.fitted_field on the box: T (full grid) and the solver's info."""
    Tc = base_tensor(B) if Tc is None else Tc
    return S.fitted_field(Tc, B["ctx"], B["wm"], B["vox"], lam, tol)


def fitted_tensor(B, Tc=None):
    T, info = fitted(B, Tc)
    return S.gradient_tensor(T, B["vox"]), T, info


# ---------------------------------------------------------------- (iii) the blended start

def trace_blend(Tc, labels, dface, x0, d0, vox, nmax, blend_mm=0.3, **kw):
    """streamlines.trace with the blended start."""
    return S.trace(Tc, labels, dface, x0, d0, vox, nmax, blend_mm, **kw)


# ---------------------------------------------------------------- fields per tracer

def fields(B, labels, which=TRACERS, verbose=False, base=None):
    """The tensor field each tracer integrates (blend uses the base tensor), and notes. base: the
    base tensor, if already made."""
    out, notes = {}, {}
    base = base_tensor(B) if base is None else base
    for name in which:
        if name in ("current", "blend"):
            out[name] = base
        elif name == "multiscale":
            out[name] = multiscale_tensor(B, base)
        elif name == "perbank":
            out[name] = perbank_tensor(B)
        elif name in ("fitted", "fitted_blend"):
            out[name], _, notes[name] = fitted_tensor(B, base)
        elif name == "laplace":
            out[name], _, notes["laplace"] = laplace_tensor(B, labels, base)
            if verbose:
                print("laplace:", notes["laplace"], flush=True)
    return out, notes


def primary(name, Tc, labels, dface, x0, d0, vox, nmax, **kw):
    if name in ("blend", "fitted_blend"):
        return trace_blend(Tc, labels, dface, x0, d0, vox, nmax, **kw)
    return S.trace(Tc, labels, dface, x0, d0, vox, nmax, None, **kw)


# ---------------------------------------------------------------- the fallback and classes (column_volume.py)

def orientation(B):
    """The one pass's direction: the coherence-weighted data normal, signed by the WM distance
    gradient, plus 0.1 x that gradient (column_traces.trace_columns)."""
    nh, coh = PCT.data_normal(base_tensor(B), B["ctx"])
    return S.oriented_normal(nh, coh, S.wm_direction(B["wm"], B["vox"]), 0.1)


class _Tissue:
    """The fields of column_traces.Tissue that one_pass and finish read, from a curv_lib.load
    dict."""

    def __init__(self, B, labels=None):
        self.wm, self.ctx, self.vox = B["wm"], B["ctx"], np.asarray(B["vox"], float)
        self.dface, self.labels = B["dface"], labels


class Pass:
    """column_traces' white mesh, its patches and the one pass on the box, for the fallback and
    the classes."""

    def __init__(self, B, patch=6, ratio=3.0, limit_mm=8.0):
        assert patch == PCT.PATCH and ratio == PCT.RATIO and limit_mm == PCT.LIMIT_MM
        self.v, self.f, self.nrm = PCT.white_mesh(B["wm"])
        self.P = PCT.mesh_patches(self.v, self.f, np.asarray(B["vox"], float))
        self.op = PCT.one_pass(_Tissue(B), orientation(B), self.v, self.nrm, self.P)
        self.s, self.f_eq, self.D, self.u = self.op.setup, self.op.fronts, self.P.D, self.op.u
        self.tree = cKDTree(self.v)

    def vertex_of(self, x):
        """The mesh vertex at each root (the roots are this mesh's vertices)."""
        d, j = self.tree.query(np.asarray(x, float))
        assert np.all(d < 1e-6), "roots must be vertices of column_traces.white_mesh"
        return j

    def patch_at(self, x):
        return self.P.of_vertex[self.vertex_of(x)]


def with_fallback(Pa, B, labels, x0, pos1, ns1, out1, nmax, same_mm=5.0):
    """column_traces.finish: the fallback where the primary turned or ran to the cap, and the
    classes. Returns pos, ns, outc, tracer (0 primary, 1 fallback), cls, root_patch, end_patch."""
    assert nmax == PCT.NMAX and same_mm == PCT.SAME_MM
    tv = Pa.vertex_of(x0)
    pos, ns, outc, tracer, end_patch, cls = PCT.finish(
        _Tissue(B, labels), Pa.v, Pa.nrm, tv, Pa.P, Pa.op, pos1, ns1, out1)
    return pos, ns, outc, tracer, cls, Pa.P.of_vertex[tv], end_patch


# ---------------------------------------------------------------- the two marked sites

SITES = {"V1V2_banks": (198, 58, 65), "M1S1": (378, 224, 448)}


def load_site(site, size=141):
    """The box, tissue labels (0 cortex, 1 fluid, 2 WM, 3 missing, 4 other) and marked points
    (0-based box voxels) of a site, as column_volume.py builds them."""
    import json
    from pathlib import Path

    import curv_lib as C

    lo = np.array(SITES[site])
    hi = lo + size
    B = C.load(lo, hi, f"column_volume_{site}")
    labels = np.full(B["rim"].shape, 4, np.int16)
    labels[B["ctx"]] = 0
    labels[B["rim"] == 1] = 1
    labels[B["wm"]] = 2
    labels[B["miss"]] = 3
    pf = Path(__file__).parent.parent / "data" / f"{site}_points_full0.json"
    P = np.zeros((0, 3))
    if pf.exists():
        P = np.array(json.load(open(pf))["points_full0"], float) - lo
        P = P[np.all((P >= 0) & (P < hi - lo), 1)]
    return dict(B=B, labels=labels, P=P, vox=B["vox"], lo=lo, hi=hi, points_file=str(pf))
