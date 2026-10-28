"""Flattening in the column volume (plan: reports/Crossing paths and contacts from laminar profiles.md,
*Flattening in the volume*). Pure functions, no argv.

- shifts: for pairs of neighbouring traces (v, w), the shift s(z) with f_w(z + s) ~ f_v(z), by iterated
  gradient matching along the trace (Lucas-Kanade in 1D), all features together, with a local
  correlation and gradient energy for the weight.
- fit_warp: the horizon volume z_v(tau) = tau + r_v(tau), from r_w - r_v = s_vw(tau + r_v) over all
  pairs and levels, eps (r_v(tau + dtau) - r_v(tau)) = 0, r_v(0) = 0; weighted least squares by
  conjugate gradients, with Gauss-Newton outer iterations (the shift is read where v's horizon is).
  The per-level constant left free by the pair equations is fixed by the eps rows: mean_v r_v(tau) = 0
  in each connected piece, so tau is the mean depth of its horizon.
- resample / tau_of: a trace at constant tau, and the tau of a depth on a trace.

Units: mm. Traces are sampled at `step` mm from the root; a trace's features are nan past its end.
"""

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg


def masked_smooth(a, m, sig):
    """Normalized Gaussian smoothing along the last axis of a (..., n) over the mask m (broadcast to a)."""
    m = np.broadcast_to(m, a.shape)
    num = ndi.gaussian_filter1d(np.where(m, a, 0.0), sig, axis=-1, mode="constant")
    den = ndi.gaussian_filter1d(m.astype(a.dtype), sig, axis=-1, mode="constant")
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0.0)


def interp_last(a, q):
    """a (..., n) at fractional indices q (broadcast to a's leading shape, last axis any length),
    linear. Returns (values, inside)."""
    n = a.shape[-1]
    inside = (q >= 0) & (q <= n - 1)
    qc = np.clip(q, 0, n - 1 - 1e-6)
    i0 = np.floor(qc).astype(np.int64)
    t = qc - i0
    shape = a.shape[:-1] + q.shape[-1:]
    i0 = np.broadcast_to(i0, shape)
    t = np.broadcast_to(t, shape)
    v = np.take_along_axis(a, i0, -1) * (1 - t) + np.take_along_axis(a, i0 + 1, -1) * t
    return v, inside


def shifts(Fv, Fw, Mv, Mw, step, sig_d=0.16, sig_w=0.32, n_iter=10, smax=0.3):
    """Shift s (mm) with f_w(z + s) ~ f_v(z) at every sample z of v.

    Fv, Fw: (nf, k, n) features of k pairs' traces (any value where invalid); Mv, Mw: (k, n) valid.
    Derivative scale sig_d and window sig_w (mm), as the structure tensor's.
    Returns s (k, n) mm, ncc (k, n) local correlation after the shift, energy (k, n) the window's
    gradient energy per feature, valid (k, n)."""
    sd, sw = sig_d / step, sig_w / step
    a = masked_smooth(Fv, Mv[None], sd)
    b = masked_smooth(Fw, Mw[None], sd)
    gb = np.gradient(b, axis=-1)
    z = np.arange(Fv.shape[-1], dtype=float)
    s = np.zeros(Mv.shape)
    win = lambda x: ndi.gaussian_filter1d(x, sw, axis=-1, mode="constant")  # noqa: E731
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
    ncc = win((a0 * b0).sum(0)) / np.sqrt(win((a0 * a0).sum(0)) * win((b0 * b0).sum(0)) + 1e-12)
    return s * step, ncc, energy, valid


def pair_weights(ncc, energy, valid, energy_ref):
    """Weight of a pair's shift: positive correlation times the gradient energy's saturation."""
    return np.where(valid, np.maximum(ncc, 0.0) * energy / (energy + energy_ref), 0.0)


def fit_warp(pairs, S, W, n_traces, step, dtau=0.08, tau_max=6.0, eps=0.2, n_outer=4, anchor=10.0,
             maxiter=3000, rtol=1e-7, r0=None, log=None):
    """Fit r (n_traces, K) on tau_j = j dtau, j < K = tau_max / dtau + 1.

    pairs (P, 2) trace indices (v, w); S, W (P, n) the pair's shift (mm) and weight at each sample
    of v. Returns tau (K,), r (n_traces, K), info (per outer iteration: cg iterations, weighted rms)."""
    K = int(round(tau_max / dtau)) + 1
    tau = np.arange(K) * dtau
    N = n_traces
    r = np.zeros((N, K)) if r0 is None else r0.copy()
    v, w = pairs[:, 0], pairs[:, 1]
    idx = np.arange(N * K).reshape(N, K)
    # eps and anchor rows: fixed
    ev = idx[:, :-1].ravel()
    rows_e = np.arange(ev.size)
    Ae = coo_matrix((np.r_[np.full(ev.size, -eps), np.full(ev.size, eps)],
                     (np.r_[rows_e, rows_e], np.r_[ev, ev + 1])), shape=(ev.size, N * K)).tocsr()
    Aa = coo_matrix((np.full(N, anchor), (np.arange(N), idx[:, 0])), shape=(N, N * K)).tocsr()
    R = (Ae.T @ Ae + Aa.T @ Aa).tocsr()
    info = []
    for it in range(n_outer):
        q = (tau[None] + r[v]) / step  # (P, K) samples on v
        s, ins = interp_last(S, q)
        wt, _ = interp_last(W, q)
        wt = np.where(ins, np.maximum(wt, 0.0), 0.0)
        p_, j_ = np.nonzero(wt > 0)
        sw = np.sqrt(wt[p_, j_])
        rows = np.arange(p_.size)
        Ad = coo_matrix((np.r_[sw, -sw], (np.r_[rows, rows], np.r_[idx[w[p_], j_], idx[v[p_], j_]])),
                        shape=(p_.size, N * K)).tocsr()
        bd = sw * s[p_, j_]
        M = (Ad.T @ Ad + R).tocsr()
        rhs = Ad.T @ bd
        dg = M.diagonal()
        pre = diags(1.0 / np.where(dg > 0, dg, 1.0))
        n_it = [0]

        def cb(_):
            n_it[0] += 1

        x, flag = cg(M, rhs, x0=r.ravel(), rtol=rtol, maxiter=maxiter, M=pre, callback=cb)
        r = x.reshape(N, K)
        res = (Ad @ x - bd)
        rms = float(np.sqrt(np.sum(res ** 2) / np.sum(sw ** 2)))
        info.append(dict(outer=it, cg_iters=n_it[0], cg_flag=int(flag), data_rows=int(p_.size), weighted_rms_mm=rms))
        if log:
            log(f"outer {it}: {p_.size:,} data rows, cg {n_it[0]} iterations (flag {flag}), weighted rms {rms:.4f} mm")
    return tau, r, info


def residuals(pairs, S, W, r, tau, step):
    """Per pair and level: weight and misfit r_w - r_v - s_vw(tau + r_v)."""
    v, w = pairs[:, 0], pairs[:, 1]
    q = (tau[None] + r[v]) / step
    s, ins = interp_last(S, q)
    wt, _ = interp_last(W, q)
    wt = np.where(ins, np.maximum(wt, 0.0), 0.0)
    return wt, r[w] - r[v] - s


def resample(F, ns, z, step):
    """Features F (nf, N, n) at depths z (N, K) mm: (nf, N, K), nan past each trace's end."""
    q = z / step
    v, ins = interp_last(np.nan_to_num(F), q)
    ok = ins & (q <= (ns[:, None] - 1))
    return np.where(ok[None], v, np.nan)


def tau_of(depth, z, tau):
    """The tau of each trace's depth (N,) mm, from its horizons z (N, K) (made monotone first)."""
    out = np.full(len(depth), np.nan)
    zm = np.maximum.accumulate(z, axis=1)
    for i, d in enumerate(depth):
        if np.isfinite(d):
            out[i] = np.interp(d, zm[i], tau, right=np.nan)
    return out
