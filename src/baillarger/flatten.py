"""Flattening in the column volume (plan: reports/Crossing paths and contacts from
laminar profiles.md, *Flattening in the volume*). Pure functions.

- shifts: for pairs of neighbouring traces (v, w), the shift s(z) with
  f_w(z + s) ~ f_v(z), by iterated gradient matching along the trace (Lucas-Kanade in
  1D), all features together, with a local correlation and gradient energy for the
  weight.
- dynamic_shifts: the same output by dynamic warping (Hale 2013, smooth dynamic image
  warping): squared alignment errors over a lag grid, smoothed along the trace, a
  minimum-cost lag path under a strain bound on a coarse depth grid (dp_lags, exact),
  interpolated, then optionally polished by the gradient iterations. Global along the
  trace, so it reaches shifts past the gradient estimator's capture range (about half
  a band spacing).
- fit_warp: the horizon volume z_v(tau) = tau + r_v(tau), from
  r_w - r_v = s_vw(tau + r_v) over all pairs and levels,
  eps (r_v(tau + dtau) - r_v(tau)) = 0, r_v(0) = 0; weighted least squares by
  conjugate gradients, with Gauss-Newton outer iterations (the shift is read where v's
  horizon is). The per-level constant left free by the pair equations is fixed by the
  eps rows: mean_v r_v(tau) = 0 in each connected piece, so tau is the mean depth of
  its horizon.
- resample / tau_of: a trace at constant tau, and the tau of a depth on a trace.

Units: mm. Traces are sampled at `step` mm from the root; a trace's features are nan
past its end.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg

type Float = npt.NDArray[np.float64]
type Bool = npt.NDArray[np.bool_]
type Int = npt.NDArray[np.int64]
type Arr = npt.NDArray[Any]


def masked_smooth(a: Arr, m: Arr, sig: float) -> Float:
    """Normalized Gaussian smoothing along the last axis of a (..., n) over the mask m
    (broadcast to a)."""
    m = np.broadcast_to(m, a.shape)
    num = ndi.gaussian_filter1d(np.where(m, a, 0.0), sig, axis=-1, mode="constant")
    den = ndi.gaussian_filter1d(m.astype(float), sig, axis=-1, mode="constant")
    out: Float = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0.0)
    return out


def interp_last(a: Arr, q: Arr) -> tuple[Float, Bool]:
    """a (..., n) at fractional indices q (broadcast to a's leading shape, last axis
    any length), linear. Returns (values, inside)."""
    n = a.shape[-1]
    inside: Bool = (q >= 0) & (q <= n - 1)
    qc = np.clip(q, 0, n - 1 - 1e-6)
    i0 = np.floor(qc).astype(np.int64)
    t = qc - i0
    shape = a.shape[:-1] + q.shape[-1:]
    i0 = np.broadcast_to(i0, shape)
    t = np.broadcast_to(t, shape)
    v: Float = (
        np.take_along_axis(a, i0, -1) * (1 - t) + np.take_along_axis(a, i0 + 1, -1) * t
    )
    return v, inside


def shifts(
    Fv: Arr,
    Fw: Arr,
    Mv: Bool,
    Mw: Bool,
    step: float,
    sig_d: float = 0.16,
    sig_w: float = 0.32,
    n_iter: int = 10,
    smax: float = 0.3,
    s0: Float | None = None,
) -> tuple[Float, Float, Float, Bool]:
    """Shift s (mm) with f_w(z + s) ~ f_v(z) at every sample z of v.

    Fv, Fw: (nf, k, n) features of k pairs' traces (any value where invalid);
    Mv, Mw: (k, n) valid. Derivative scale sig_d and window sig_w (mm), as the
    structure tensor's. s0 (k, n) mm: the start (default 0).
    Returns s (k, n) mm, ncc (k, n) local correlation after the shift, energy (k, n)
    the window's gradient energy per feature, valid (k, n)."""
    sd, sw = sig_d / step, sig_w / step
    a = masked_smooth(Fv, Mv[None], sd)
    b = masked_smooth(Fw, Mw[None], sd)
    gb = np.gradient(b, axis=-1)
    z = np.arange(Fv.shape[-1], dtype=float)
    s: Float = np.zeros(Mv.shape) if s0 is None else s0 / step

    def win(x: Arr) -> Float:
        out: Float = ndi.gaussian_filter1d(x, sw, axis=-1, mode="constant")
        return out

    mwf = Mw.astype(float)
    for _ in range(n_iter):
        q = z + s
        bw, ok = interp_last(b, q)
        gw, _ = interp_last(gb, q)
        mw, _ = interp_last(mwf, q)
        valid = Mv & ok & (mw > 0.999)
        e = (bw - a) * valid
        num = win((gw * e).sum(0))
        den = win((gw * gw * valid).sum(0))
        s = np.clip(s - num / (den + 1e-12), -smax / step, smax / step)
    q = z + s
    bw, ok = interp_last(b, q)
    mw, _ = interp_last(mwf, q)
    valid = Mv & ok & (mw > 0.999)
    gw, _ = interp_last(gb, q)
    energy = win((gw * gw * valid).sum(0)) / Fv.shape[0]
    cnt = win(valid.astype(float))
    c = np.maximum(cnt, 1e-6)
    ma = win((a * valid).sum(0)) / c / Fv.shape[0]
    mb = win((bw * valid).sum(0)) / c / Fv.shape[0]
    a0 = (a - ma) * valid
    b0 = (bw - mb) * valid
    ncc = win((a0 * b0).sum(0)) / np.sqrt(
        win((a0 * a0).sum(0)) * win((b0 * b0).sum(0)) + 1e-12
    )
    return s * step, ncc, energy, valid


def dp_lags(E: Float, m: int) -> Int:
    """Lag indices l (k, n) minimizing sum_i E[:, i, l_i] subject to
    |l_{i+1} - l_i| <= m, per pair.

    E (k, n, nl) costs. Accumulation forward, backtrack from the cheapest last lag;
    ties go to the unchanged lag."""
    k, n, nl = E.shape
    offs = np.r_[0, np.ravel(np.c_[-np.arange(1, m + 1), np.arange(1, m + 1)])]
    D = E[:, 0].astype(np.float64)
    back = np.zeros((k, n, nl), np.int8)
    for i in range(1, n):
        P = np.full((offs.size, k, nl), np.inf)
        for j, o in enumerate(offs):  # predecessor of lag l is l + o
            if o >= 0:
                P[j, :, : nl - o] = D[:, o:]
            else:
                P[j, :, -o:] = D[:, :o]
        a = np.argmin(P, axis=0)
        D = E[:, i] + np.take_along_axis(P, a[None], 0)[0]
        back[:, i] = a
    lags = np.zeros((k, n), np.int64)
    lags[:, -1] = np.argmin(D, axis=1)
    rows = np.arange(k)
    for i in range(n - 1, 0, -1):
        lags[:, i - 1] = lags[:, i] + offs[back[rows, i, lags[:, i]]]
    return lags


def dynamic_shifts(
    Fv: Arr,
    Fw: Arr,
    Mv: Bool,
    Mw: Bool,
    step: float,
    sig_d: float = 0.16,
    sig_w: float = 0.32,
    smax: float = 0.3,
    dl: float | None = None,
    coarse: float = 0.16,
    strain: float = 0.2,
    polish: int = 10,
    chunk: int = 2048,
) -> tuple[Float, Float, Float, Bool]:
    """Shift s (mm) with f_w(z + s) ~ f_v(z), by dynamic warping; the arguments and
    output as `shifts`.

    Errors e(z, s) = sum over features (a(z) - b(z + s))^2 of the smoothed features,
    on lags s in [-smax, smax] at spacing dl (default step / 4, b interpolated). Where
    a or b(z + s) is invalid the error is the mean of the row's valid lags (no
    preference); a row with none is 0. Errors are smoothed along z (Gaussian sig_w),
    the lag path is found on depths every `coarse` mm with lag changes of at most
    strain * coarse, and interpolated linearly to every sample (|ds/dz| <= strain).
    Then `polish` gradient iterations (`shifts`) from that start refine it below dl;
    0 keeps it as is. The gradient iterations only fix the shift's average over their
    window (win(g e) = 0 at each z), so a start with ripples shorter than the window
    (the path's kinks every `coarse` mm) keeps them; the start is first smoothed at
    sig_w, which keeps both bounds. Pairs in chunks of `chunk`."""
    dl = step / 4 if dl is None else dl
    sd = sig_d / step
    nl_half = int(np.floor(smax / dl + 1e-9))
    lagv = np.arange(-nl_half, nl_half + 1) * dl
    b_ = max(1, int(round(coarse / step)))
    m = int(np.floor(strain * b_ * step / dl + 1e-9))
    k, n = Mv.shape
    z = np.arange(n, dtype=float)
    ic = np.arange(0, n, b_)
    s = np.zeros((k, n))
    mwf = Mw.astype(float)
    for c0 in range(0, k, chunk):
        sl = slice(c0, min(k, c0 + chunk))
        a = masked_smooth(Fv[:, sl], Mv[sl][None], sd)
        bb = masked_smooth(Fw[:, sl], Mw[sl][None], sd)
        kc = a.shape[1]
        E = np.zeros((kc, n, lagv.size))
        V = np.zeros((kc, n, lagv.size), bool)
        for j, lg in enumerate(lagv):
            q = np.broadcast_to(z + lg / step, (kc, n))
            bw, ok = interp_last(bb, q)
            mw, _ = interp_last(mwf[sl], q)
            ok = ok & (mw > 0.999) & Mv[sl]
            E[:, :, j] = ((bw - a) ** 2).sum(0)
            V[:, :, j] = ok
        cnt = V.sum(2)
        rowmean = np.where(cnt > 0, (E * V).sum(2) / np.maximum(cnt, 1), 0.0)
        E = np.where(V, E, rowmean[..., None])
        E = ndi.gaussian_filter1d(E, sig_w / step, axis=1, mode="nearest")
        lags = dp_lags(E[:, ic], m)
        sc = lagv[lags]  # (kc, len(ic)) mm
        for r in range(kc):
            s[c0 + r] = np.interp(z, ic, sc[r])
    if polish > 0:
        s = ndi.gaussian_filter1d(s, sig_w / step, axis=1, mode="nearest")
    return shifts(
        Fv, Fw, Mv, Mw, step, sig_d=sig_d, sig_w=sig_w, n_iter=polish, smax=smax, s0=s
    )


def pair_weights(ncc: Float, energy: Float, valid: Bool, energy_ref: float) -> Float:
    """Weight of a pair's shift: positive correlation times the gradient energy's
    saturation."""
    out: Float = np.where(
        valid, np.maximum(ncc, 0.0) * energy / (energy + energy_ref), 0.0
    )
    return out


def fit_warp(
    pairs: Int,
    S: Arr,
    W: Arr,
    n_traces: int,
    step: float,
    dtau: float = 0.08,
    tau_max: float = 6.0,
    eps: float = 0.2,
    n_outer: int = 4,
    anchor: float = 10.0,
    maxiter: int = 3000,
    rtol: float = 1e-7,
    r0: Float | None = None,
    log: Callable[[str], object] | None = None,
) -> tuple[Float, Float, list[dict[str, Any]]]:
    """Fit r (n_traces, K) on tau_j = j dtau, j < K = tau_max / dtau + 1.

    pairs (P, 2) trace indices (v, w); S, W (P, n) the pair's shift (mm) and weight at
    each sample of v. Returns tau (K,), r (n_traces, K), info (per outer iteration: cg
    iterations, weighted rms)."""
    K = int(round(tau_max / dtau)) + 1
    tau: Float = np.arange(K, dtype=float) * dtau
    N = n_traces
    r = np.zeros((N, K)) if r0 is None else r0.copy()
    v, w = pairs[:, 0], pairs[:, 1]
    idx = np.arange(N * K).reshape(N, K)
    # eps and anchor rows: fixed
    ev = idx[:, :-1].ravel()
    rows_e = np.arange(ev.size)
    Ae = coo_matrix(
        (
            np.r_[np.full(ev.size, -eps), np.full(ev.size, eps)],
            (np.r_[rows_e, rows_e], np.r_[ev, ev + 1]),
        ),
        shape=(ev.size, N * K),
    ).tocsr()
    Aa = coo_matrix(
        (np.full(N, anchor), (np.arange(N), idx[:, 0])), shape=(N, N * K)
    ).tocsr()
    R = (Ae.T @ Ae + Aa.T @ Aa).tocsr()
    info: list[dict[str, Any]] = []
    for it in range(n_outer):
        q = (tau[None] + r[v]) / step  # (P, K) samples on v
        s, ins = interp_last(S, q)
        wt, _ = interp_last(W, q)
        wt = np.where(ins, np.maximum(wt, 0.0), 0.0)
        p_, j_ = np.nonzero(wt > 0)
        sw = np.sqrt(wt[p_, j_])
        rows = np.arange(p_.size)
        Ad = coo_matrix(
            (
                np.r_[sw, -sw],
                (np.r_[rows, rows], np.r_[idx[w[p_], j_], idx[v[p_], j_]]),
            ),
            shape=(p_.size, N * K),
        ).tocsr()
        bd = sw * s[p_, j_]
        M = (Ad.T @ Ad + R).tocsr()
        rhs = Ad.T @ bd
        dg = M.diagonal()
        pre = diags(1.0 / np.where(dg > 0, dg, 1.0))
        n_it = [0]

        def cb(_: Any, n_it: list[int] = n_it) -> None:
            n_it[0] += 1

        x, flag = cg(
            M, rhs, x0=r.ravel(), rtol=rtol, maxiter=maxiter, M=pre, callback=cb
        )
        r = x.reshape(N, K)
        res = Ad @ x - bd
        den = float(np.sum(sw**2))
        rms = float(np.sqrt(np.sum(res**2) / den)) if den > 0 else 0.0
        info.append(
            dict(
                outer=it,
                cg_iters=n_it[0],
                cg_flag=int(flag),
                data_rows=int(p_.size),
                weighted_rms_mm=rms,
            )
        )
        if log:
            log(
                f"outer {it}: {p_.size:,} data rows, cg {n_it[0]} iterations "
                f"(flag {flag}), weighted rms {rms:.4f} mm"
            )
    return tau, r, info


def residuals(
    pairs: Int, S: Arr, W: Arr, r: Float, tau: Float, step: float
) -> tuple[Float, Float]:
    """Per pair and level: weight and misfit r_w - r_v - s_vw(tau + r_v)."""
    v, w = pairs[:, 0], pairs[:, 1]
    q = (tau[None] + r[v]) / step
    s, ins = interp_last(S, q)
    wt, _ = interp_last(W, q)
    wt = np.where(ins, np.maximum(wt, 0.0), 0.0)
    out: Float = r[w] - r[v] - s
    return wt, out


def resample(F: Arr, ns: Arr, z: Arr, step: float) -> Float:
    """Features F (nf, N, n) at depths z (N, K) mm: (nf, N, K), nan past each trace's
    end."""
    q = z / step
    v, ins = interp_last(np.nan_to_num(F), q)
    ok = ins & (q <= (ns[:, None] - 1))
    out: Float = np.where(ok[None], v, np.nan)
    return out


def tau_of(depth: Arr, z: Arr, tau: Arr) -> Float:
    """The tau of each trace's depth (N,) mm, from its horizons z (N, K) (made
    monotone first)."""
    out = np.full(len(depth), np.nan)
    zm = np.maximum.accumulate(z, axis=1)
    for i, d in enumerate(depth):
        if np.isfinite(d):
            out[i] = np.interp(d, zm[i], tau, right=np.nan)
    return out
