"""Bands of a region (step 3b): the optimal K-band split of its pooled profile
on the depth grid, for every K, and K by cross-validation.

The voxels are normalized by their column's gain and offset, per echo:
x̂_e(v) = (f_e(v) − a_ce) / s_ce, with weight w_e(v) = s_ce² / σ²_ce. The
residual of a band is Σ_e Σ_v w_e (x̂_e − μ_e)², separable over the echoes
(the noise is diagonal), so the cost of an interval of the grid is the sum
over the echoes of the one-echo cost, from prefix sums of Σw, Σw x̂ and
Σw x̂² in each bin.

Band DP (Bellman segmentation): for each K = 1..k_max, the K-band split of
the grid with the least residual, each band at least t_min wide. One pass
gives every K; the split for K need not refine the split for K − 1.

Cross-validation: the region's columns are split into two halves at random
(fixed seed; a region with fewer than min_cols_cv columns splits its voxels
instead). The split for each K is fitted on one half and its residual
measured on the other, with the held-out voxels' own normalization; then the
halves swap. K_R is the least K whose held-out residual is within the larger
of one standard error and gain_frac (5 %) of the whole held-out gain of the
least one (choose_k), and Z_R the best split for K_R on all the region's
voxels.

The prototype fitted each held-out column's gain and offset again to the
template (refit=True) and took the least residual (one_se=False). A one-band
template is flat, so it leaves the gain unused while every K >= 2 fits it on
the held-out data itself: K = 1 always lost, and among the larger K, whose
residuals differ by a few in 10,000, the minimum fell at random. On
synthetic regions (20 seeds each) that rule found a flat profile's K = 1 in
none, against all with the default; over eight cases with K = 1 to 4 and one
or three echoes the default found the true K in 160 of 160. On the large
graph components of V1V2 (1,000 to 4,000 columns) the standard error alone
gave K = 8 = K_max, each extra band still lowering the held-out residual;
with the 5 % gain term, K = 5 or 6.

fixed_bands gives the fixed-K baseline: n_layers equal bins of r, with the
sparse edge layers merged into their inner neighbour.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]


def band_of(m: F64, z: F64) -> I64:
    """Band (0-based) of each depth m, for the boundaries z (one row per
    voxel, or one shared row)."""
    if z.ndim == 1:
        out: I64 = np.searchsorted(z, m, side="right").astype(np.int64)
        return out
    return (m[:, None] >= z).sum(1).astype(np.int64)


def fit_affine(f: F64, t: F64, w: F64, col: I64, n: int) -> tuple[F64, F64, F64]:
    """Per column weighted least squares f = a + s t (one echo). Returns a, s
    and the residual of each column."""
    sw = np.bincount(col, w, n)
    st = np.bincount(col, w * t, n)
    stt = np.bincount(col, w * t * t, n)
    sf = np.bincount(col, w * f, n)
    sft = np.bincount(col, w * f * t, n)
    sff = np.bincount(col, w * f * f, n)
    det = sw * stt - st * st
    ok = det > 1e-9 * np.maximum(sw * stt, 1e-300)
    a = np.where(sw > 0, sf / np.maximum(sw, 1e-300), 0.0)
    s = np.zeros(n)
    a[ok] = (stt[ok] * sf[ok] - st[ok] * sft[ok]) / det[ok]
    s[ok] = (sw[ok] * sft[ok] - st[ok] * sf[ok]) / det[ok]
    res = sff - a * sf - s * sft
    return a, s, np.maximum(res, 0.0)


def band_dp(
    W: F64, S: F64, Q: F64, k_max: int, min_w: int
) -> dict[int, tuple[I64, float]]:
    """Optimal K-band splits of a binned profile, for K = 1..k_max.

    W, S, Q: per bin (and per echo: shape (G,) or (G, E)), the sums of
    weights, of weight x value and of weight x value². Each band has at least
    min_w bins. Returns {K: (cuts, residual)}: cuts are the bins where bands
    2..K start.
    """
    W2, S2, Q2 = (np.asarray(v, np.float64).reshape(len(W), -1) for v in (W, S, Q))
    G = len(W2)
    z = np.zeros((1, W2.shape[1]))
    cW, cS, cQ = (np.concatenate([z, np.cumsum(v, axis=0)]) for v in (W2, S2, Q2))

    def sse(i: Any, j: Any) -> Any:  # bins i..j-1; i, j arrays or scalars
        w = cW[j] - cW[i]
        s = cS[j] - cS[i]
        r = cQ[j] - cQ[i] - np.where(w > 0, s * s / np.where(w > 0, w, 1), 0.0)
        return r.sum(-1)

    inf = np.inf
    cost = np.full((k_max + 1, G + 1), inf)
    arg = np.zeros((k_max + 1, G + 1), np.int64)
    for j in range(min_w, G + 1):
        cost[1, j] = sse(0, j)
    for k in range(2, k_max + 1):
        for j in range(k * min_w, G + 1):
            i = np.arange((k - 1) * min_w, j - min_w + 1)
            c = cost[k - 1, i] + sse(i, j)
            b = int(np.argmin(c))
            cost[k, j], arg[k, j] = c[b], i[b]
    out: dict[int, tuple[I64, float]] = {}
    for k in range(1, k_max + 1):
        if not np.isfinite(cost[k, G]):
            break
        cuts, j = [], G
        for kk in range(k, 1, -1):
            j = int(arg[kk, j])
            cuts.append(j)
        out[k] = (np.array(cuts[::-1], np.int64), float(cost[k, G]))
    return out


def grid_stats(
    xh: F64, w: F64, g: I64, G: int, sel: npt.NDArray[np.bool_] | None = None
) -> tuple[F64, F64, F64]:
    """Per grid bin and echo: Σw, Σw x̂, Σw x̂² (each (G, E)). xh, w: (V, E)."""
    if sel is not None:
        xh, w, g = xh[sel], w[sel], g[sel]
    E = xh.shape[1]
    out = [np.empty((G, E)) for _ in range(3)]
    for e in range(E):
        out[0][:, e] = np.bincount(g, w[:, e], G)
        out[1][:, e] = np.bincount(g, w[:, e] * xh[:, e], G)
        out[2][:, e] = np.bincount(g, w[:, e] * xh[:, e] * xh[:, e], G)
    return out[0], out[1], out[2]


def band_fits(
    col: I64,
    m: F64,
    f: F64,
    xh: F64,
    w: F64,
    sigma2: F64,
    n: int,
    h: float,
    k_max: int,
    t_min: float,
    rng: np.random.Generator,
    min_cols_cv: int = 6,
    refit: bool = False,
) -> dict[str, Any]:
    """The band splits of one region for every K, and their cross-validated
    residuals.

    refit: measure the held-out residual after fitting each held-out column's
    gain and offset again to the template (as the prototype); by default the
    held-out voxels keep their normalization, Σ w (x̂ − μ)².

    The arrays are the region's cortex voxels: col (column 0..n-1), m (r),
    f (raw echoes, (V, E)), xh and w (normalized values and weights, (V, E));
    sigma2 (n, E) the noise of every column. Returns a dict with g (grid bin
    of each voxel), fits ({K: cuts} on all the voxels), sse (in-sample
    residual for each K), cv (held-out residual for each K; inf where K is
    not possible), percol (held-out residual per K and column) and cols (the
    region's columns).
    """
    G = int(np.ceil(1.0 / h - 1e-9))
    min_w = max(1, int(round(t_min / h)))
    g = np.minimum((m / h).astype(np.int64), G - 1)
    g = np.maximum(g, 0)
    cols = np.unique(col)
    if len(cols) >= min_cols_cv:
        half = np.zeros(n, bool)
        half[rng.permutation(cols)[: len(cols) // 2]] = True
        inA = half[col]
    else:
        inA = rng.random(len(col)) < 0.5
    E = f.shape[1]
    cv = np.zeros(k_max)
    percol = np.zeros((k_max, n))
    for A in (inA, ~inA):
        Wg, Sg, Qg = grid_stats(xh, w, g, G, A)
        fits = band_dp(Wg, Sg, Qg, k_max, min_w)
        B = ~A
        for k, (cuts, _) in fits.items():
            edges = np.concatenate([[0], cuts, [G]])
            band = np.searchsorted(cuts, g[B], side="right")
            for e in range(E):
                mu = np.array(
                    [
                        Sg[i:j, e].sum() / max(Wg[i:j, e].sum(), 1e-300)
                        for i, j in zip(edges[:-1], edges[1:], strict=True)
                    ]
                )
                if refit:
                    # held-out columns: their gain and offset fitted again to
                    # the template, on the raw data
                    _, _, res = fit_affine(
                        f[B, e], mu[band], 1.0 / sigma2[col[B], e], col[B], n
                    )
                else:
                    res = np.bincount(col[B], w[B, e] * (xh[B, e] - mu[band]) ** 2, n)
                cv[k - 1] += res.sum()
                percol[k - 1] += res
        for k in range(len(fits) + 1, k_max + 1):
            cv[k - 1] = np.inf
    full = band_dp(*grid_stats(xh, w, g, G), k_max, min_w)
    return dict(
        g=g,
        fits={k: c for k, (c, _) in full.items()},
        sse=np.array([e for _, e in full.values()]),
        cv=cv,
        percol=percol[:, cols],
        cols=cols,
    )


def bands_for_region(
    col: I64,
    m: F64,
    f: F64,
    xh: F64,
    w: F64,
    sigma2: F64,
    n: int,
    h: float,
    k_max: int,
    t_min: float,
    rng: np.random.Generator,
    min_cols_cv: int = 6,
    refit: bool = False,
    one_se: bool = True,
    gain_frac: float = 0.05,
) -> tuple[int, F64, F64]:
    """Step 3b for one region: K_R, its boundaries Z_R (relative depths) and
    the cross-validated residual of every K.

    K_R: see choose_k. refit=True, one_se=False give the prototype's rule.
    """
    bf = band_fits(
        col, m, f, xh, w, sigma2, n, h, k_max, t_min, rng, min_cols_cv, refit
    )
    cv = bf["cv"]
    K = choose_k(cv, bf["percol"], one_se, gain_frac)
    return K, bf["fits"][K].astype(np.float64) * h, cv


def choose_k(cv: F64, percol: F64, one_se: bool = True, gain_frac: float = 0.05) -> int:
    """K from the cross-validated residuals cv (one per K) and percol (per K
    and held-out column).

    With one_se: the least K whose residual is within the larger of one
    standard error (of the paired difference over the held-out columns) and
    gain_frac of the whole held-out gain cv[K=1] - min(cv) of the least one.
    The standard error alone stops nothing in a large region: real profiles
    have smooth trends, and with a million voxels every extra band is a real
    improvement, however small. Without one_se: the least residual.
    """
    kmin = int(np.argmin(cv))
    if not one_se or percol.shape[1] < 2:
        return kmin + 1
    tol_gain = gain_frac * (cv[0] - cv[kmin]) if np.isfinite(cv[0]) else 0.0
    for k in range(kmin + 1):
        if not np.isfinite(cv[k]):
            continue
        d = percol[k] - percol[kmin]
        se = np.sqrt(len(d)) * d.std(ddof=1)
        if cv[k] <= cv[kmin] + max(se, tol_gain):
            return k + 1
    return kmin + 1


def fixed_bands(m: F64, n_layers: int, f_sparse: float) -> F64:
    """The fixed-K baseline: n_layers equal bins of r as bands, the sparse edge
    layers (fewer voxels than f_sparse x the median layer) merged into their
    inner neighbour. m: r of the cortex voxels. Returns the boundaries."""
    lay = np.minimum(np.ceil(m * n_layers).astype(np.int64), n_layers)
    cnt = np.bincount(lay, minlength=n_layers + 1)[1:]
    med = np.median(cnt[cnt > 0])
    keep = np.ones(n_layers, bool)
    # edge layers only: walk in from each end while the layer is sparse
    for order in (range(n_layers), range(n_layers - 1, -1, -1)):
        for k in order:
            if cnt[k] < f_sparse * med:
                keep[k] = False
            else:
                break
    first, last = np.flatnonzero(keep)[[0, -1]]
    return np.arange(first + 1, last + 1) / n_layers
