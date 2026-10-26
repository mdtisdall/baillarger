"""The gain/offset field and the templates (plan steps 2 and 3).

Model, per echo e (the noise is diagonal over the echoes, so the echoes
decouple in both steps): x_e(v) = a_ce + s_ce μ_R,k,e + noise, for voxel v in
column c of region R and band k.

Step 2, for fixed templates and bands: a and s minimize
Σ_v (x_e − a_ce − s_ce t_e(v))² / σ²_ce + κ / σ̄²_e Σ_c~c' w_cc' ((a_ce − a_c'e)²
+ (s_ce − s_c'e)²), with t_e(v) the template value of the voxel and σ̄²_e the
median noise of echo e: a sparse linear system in 2 unknowns per column and
echo. κ = 0 gives the independent per-column least squares; κ → ∞ one gain
and one offset for all the columns of a connected piece of the graph. A tiny
ridge keeps a degenerate column (no voxels, or a flat template) at its
previous a, s.

Step 3, for fixed gain/offset and bands: μ_R,k,e = Σ s (x − a) / σ² / Σ s² / σ²
over the voxels of region R in band k; then each region's template is
centred and scaled to zero mean and unit RMS over its voxels (per echo), and
a and s change to match, so the model is unchanged. A flat template (K = 1)
is all zero, and its mean goes into the offsets.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]


def _solve_gain_echo(
    col: I64,
    f: F64,
    t: F64,
    sigma2: F64,
    n: int,
    edges: I64,
    ew: F64,
    kappa: float,
    a0: F64,
    s0: F64,
) -> tuple[F64, F64]:
    """Step 2 for one echo (see the module notes)."""
    c, tt = col, t
    w = 1.0 / sigma2[c]
    sw, st, stt = (np.bincount(c, w * v, n) for v in (np.ones_like(tt), tt, tt * tt))
    sf, sft = np.bincount(c, w * f, n), np.bincount(c, w * f * tt, n)
    ridge = 1e-6 * np.maximum(sw, 1e-12)  # keeps degenerate columns at a0, s0
    A: Any = sp.bmat(
        [[sp.diags(sw + ridge), sp.diags(st)], [sp.diags(st), sp.diags(stt + ridge)]],
        format="csr",
    )
    rhs = np.concatenate([sf + ridge * a0, sft + ridge * s0])
    if kappa > 0:
        sig2 = np.median(sigma2)
        i, j = edges[:, 0], edges[:, 1]
        C = sp.coo_matrix((ew, (i, j)), (n, n))
        Wm = (C + C.T).tocsr()
        L = (sp.diags(np.asarray(Wm.sum(1)).ravel()) - Wm) * (kappa / sig2)
        A = A + sp.block_diag([L, L], format="csr")
    x = spla.spsolve(A.tocsc(), rhs)
    return x[:n], x[n:]


def solve_gain(
    col: I64,
    f: F64,
    t: F64,
    sigma2: F64,
    edges: I64,
    ew: F64,
    kappa: float,
    a0: F64,
    s0: F64,
) -> tuple[F64, F64]:
    """Step 2: the gain s and offset a of every column and echo ((n, E) each).

    col, f, t: the voxels' columns, echo values (V, E) and template values
    (V, E); sigma2 (n, E): the noise; edges, ew: the column graph; a0, s0
    (n, E): the previous values, where a column has nothing to fit.
    """
    n, E = sigma2.shape
    a = np.empty((n, E))
    s = np.empty((n, E))
    for e in range(E):
        a[:, e], s[:, e] = _solve_gain_echo(
            col, f[:, e], t[:, e], sigma2[:, e], n, edges, ew, kappa, a0[:, e], s0[:, e]
        )
    return a, s


def update_templates(
    col: I64,
    reg: I64,
    bandv: I64,
    f: F64,
    a: F64,
    s: F64,
    sigma2: F64,
    K: dict[int, int],
) -> dict[int, F64]:
    """Step 3: the template of each region ({R: (K_R, E)}) by weighted least
    squares, then zero mean and unit RMS per echo; a and s ((n, E)) change in
    place to match. reg: the region of each column (0..nR-1); col, bandv, f:
    the voxels' columns, bands and echo values (V, E)."""
    E = f.shape[1]
    nR = int(reg.max()) + 1
    kmax = max(K.values())
    c = col
    r = reg[c]
    idx = r * kmax + bandv
    mus = {R: np.zeros((k, E)) for R, k in K.items()}
    for e in range(E):
        w = 1.0 / sigma2[c, e]
        sn = np.bincount(idx, w * s[c, e] * (f[:, e] - a[c, e]), nR * kmax)
        sd = np.bincount(idx, w * s[c, e] ** 2, nR * kmax)
        mu_all = np.where(sd > 0, sn / np.maximum(sd, 1e-300), 0.0).reshape(nR, kmax)
        # normalize each region's template over its voxels
        wt = np.bincount(idx, w, nR * kmax).reshape(nR, kmax)
        for R, k in K.items():
            mu = mu_all[R, :k]
            ww = wt[R, :k]
            tot = ww.sum()
            if tot <= 0:
                continue
            mean = (ww * mu).sum() / tot
            sd_ = np.sqrt((ww * (mu - mean) ** 2).sum() / tot)
            colsR = reg == R
            if sd_ <= 1e-12:  # K = 1 or flat: offset only
                a[colsR, e] += s[colsR, e] * mean
                continue
            mus[R][:, e] = (mu - mean) / sd_
            a[colsR, e] += s[colsR, e] * mean
            s[colsR, e] *= sd_
    return mus
