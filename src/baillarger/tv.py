"""TV on the normalized band values (plan step 6).

For each column c and band k, the cell mean x̂_c,k,e = mean of (x_e − a_ce) /
s_ce over the cell's voxels (n_c,k of them), per echo. The filtered values u
minimize

    ½ Σ_c,k,e n_c,k (u_c,k,e − x̂_c,k,e)²
    + λ_g Σ_c~c' w_cc' ||P_c u_c − P_c' u_c'||₂
    + λ_l Σ_c~c' w_cc' Σ_g,e √h |(P_c u_c)_g,e − (P_c' u_c')_g,e|

where P_c writes column c's band values to the depth-grid bins it covers
(√h per bin, so that the scale does not depend on h), over the bins where
both columns have voxels. The group term is LayNii's LN2_COLUMN_SMOOTH_TV_NORM
form; with three echoes its vector holds every bin and every echo, so an
areal border acts on the whole multi-echo profile at once. λ_l > 0 adds the
per-depth (and per-echo) term. Solved by Chambolle-Pock (primal-dual), with
the steps from the operator norm (power iteration); u does not feed back
into steps 1-5.
"""

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from baillarger.bands import band_of

type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]


def tv_bands(
    col: I64,
    m: F64,
    f: F64,
    a: F64,
    s: F64,
    bandv: I64,
    z: F64,
    Kc: I64,
    edges: I64,
    ew: F64,
    h: float,
    lambda_g: float,
    lambda_l: float,
    n_iter: int = 2000,
    tol: float = 1e-5,
) -> tuple[F64, F64, F64]:
    """u (n, K_max, E), x̂ (n, K_max, E) and the cell counts n (n, K_max).

    col, m, f, bandv: the cortex voxels (column, r, echoes (V, E), band); a,
    s (n, E); z (n, K_max − 1): each column's boundaries; Kc (n): its number
    of bands; edges, ew: the column graph.
    """
    n, E = a.shape
    G = int(np.ceil(1.0 / h - 1e-9))
    kmax = int(Kc.max())
    c = col
    xh = (f - a[c]) / np.where(s[c] != 0, s[c], 1.0)
    cell = c * kmax + bandv
    ncell = np.bincount(cell, minlength=n * kmax).astype(np.float64)
    xhat = np.stack(
        [
            np.bincount(cell, xh[:, e], n * kmax) / np.maximum(ncell, 1)
            for e in range(E)
        ],
        1,
    )
    if lambda_g <= 0 and lambda_l <= 0:
        return (
            xhat.reshape(n, kmax, E),
            xhat.reshape(n, kmax, E),
            ncell.reshape(n, kmax),
        )
    # the band of each grid bin, per column; the bins where the column has voxels
    centres = ((np.arange(G) + 0.5) * h).astype(np.float64)
    gb = np.zeros((n, G), np.int64)
    for i in range(n):
        gb[i] = band_of(centres, z[i][: Kc[i] - 1]) if Kc[i] > 1 else 0
    gv = np.clip((m / h).astype(np.int64), 0, G - 1)
    has = np.zeros((n, G), bool)
    has[c, gv] = True
    # difference operator: one row per (link, common bin)
    ei, ej = edges[:, 0], edges[:, 1]
    both = has[ei] & has[ej]
    e_idx, g_idx = np.nonzero(both)
    rows = np.arange(len(e_idx))
    ci_, cj_ = ei[e_idx], ej[e_idx]
    ui = ci_ * kmax + gb[ci_, g_idx]
    uj = cj_ * kmax + gb[cj_, g_idx]
    sq = np.sqrt(h)
    D = sp.coo_matrix(
        (
            np.concatenate([np.full(len(rows), sq), np.full(len(rows), -sq)]),
            (np.concatenate([rows, rows]), np.concatenate([ui, uj])),
        ),
        (len(rows), n * kmax),
    ).tocsr()
    we = ew[e_idx]
    # operator norm (power iteration)
    v = np.random.default_rng(0).standard_normal(n * kmax)
    nv = 1.0
    for _ in range(50):
        v = D.T @ (D @ v)
        nv = float(np.linalg.norm(v))
        v /= nv
    L2 = nv * (1 + (1 if lambda_l > 0 else 0))
    tau = sigma = 0.99 / np.sqrt(L2)
    u = xhat.copy()
    ub = u.copy()
    y1 = np.zeros((len(rows), E))
    y2 = np.zeros((len(rows), E))
    bound_g = lambda_g * ew
    bound_l = (lambda_l * we * sq)[:, None]  # per entry: λ_l w Σ_g √h |.|
    nc = ncell[:, None]
    for _ in range(n_iter):
        Du = D @ ub
        if lambda_g > 0:
            y1 = y1 + sigma * Du
            nrm = np.sqrt(np.bincount(e_idx, (y1 * y1).sum(1), len(ew)))
            scale = np.minimum(1.0, bound_g / np.maximum(nrm, 1e-300))
            y1 *= scale[e_idx][:, None]
        if lambda_l > 0:
            y2 = np.clip(y2 + sigma * Du, -bound_l, bound_l)
        u_old = u
        u = u - tau * (D.T @ (y1 + y2))
        u = (u + tau * nc * xhat) / (1 + tau * nc)
        ub = 2 * u - u_old
        if np.linalg.norm(u - u_old) <= tol * max(float(np.linalg.norm(u)), 1e-12):
            break
    return u.reshape(n, kmax, E), xhat.reshape(n, kmax, E), ncell.reshape(n, kmax)
