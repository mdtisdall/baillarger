"""The joint warp's levels from column data (plan: reports/Crossing paths and contacts from laminar
profiles.md, *The joint warp*). Pure functions, no argv.

- build: a baillarger.surfcut.Model for levels tau = TAU (the last is the pia) on depth grids of step h, one
  window per variable (centre +- half, mm), with the terms of the plan: conservation (inner levels,
  an exact table over both labels, tapered by the starting pia), lateral smoothness, ordering, the
  vertical prior (an exact table), open-top anchors, ties. Nothing depends on the window's centre,
  so moving a window never changes the energy.
- solve_pass: one pass; re-centres windows hit at their edge and repeats (up to `recentre` times).
- coarse_to_fine: a coarse pass, then a fine pass with windows set by the coarse certification,
  solved by QPBO with pruning (baillarger.surfcut.qpbo_pruned).
"""

from dataclasses import dataclass, field

import numpy as np

import baillarger.surfcut as WC

TAU = (1 / 3, 2 / 3, 1.0)


@dataclass
class Data:
    step: float  # sample step along traces (mm)
    ns: np.ndarray  # (n,) samples per trace
    top: np.ndarray  # (n,) open top (mm), nan where not an anchor
    mu: np.ndarray  # (n,) curvature trend of thickness (mm)
    pairs: np.ndarray  # (P, 2) lateral neighbours
    S: np.ndarray  # (P, m) shift s_vw at each sample of v (mm)
    W: np.ndarray  # (P, m) its weight (0-1)
    ties: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), int)
    )  # (T, 2) facing pia pairs
    tie_L: np.ndarray = field(
        default_factory=lambda: np.zeros(0)
    )  # (T,) trace length (mm)
    tie_w: np.ndarray = field(
        default_factory=lambda: np.zeros(0)
    )  # (T,) facing factor (0-1)
    c0: np.ndarray = (
        None  # (n,) the starting pia (mm): fixes the conservation taper for all passes
    )
    tau: tuple = TAU  # the levels' tau (the last is the pia)
    band_cost: list = (
        None  # per inner level: None, or (n, nt) cost on band_tf (a band's evidence)
    )
    band_tf: np.ndarray = None  # (nt,) tau grid of band_cost
    warp_tg: np.ndarray = None  # (J + 1,) tau grid of warp_z
    warp_z: np.ndarray = (
        None  # (n, J + 1) a warp, to read band_cost at a depth: tau_v(depth)
    )
    _g_full: np.ndarray = (
        None  # cache: the monotone neighbour maps g(z) = z + s_vw(z) (see build)
    )

    def g_full(self):
        """g(z) = z + s_vw(z) at every sample of v, projected onto non-decreasing maps (a warp between
        neighbours cannot fold; folds are estimation errors). Computed once."""
        if self._g_full is None:
            zgrid = self.step * np.arange(self.S.shape[1])[None]
            self._g_full = np.maximum.accumulate(zgrid + self.S.astype(float), axis=1)
        return self._g_full


def interp_rows(x, xp, fp):
    """np.interp row by row: x (n, k) at xp (n, m) increasing per row, fp (n, m); clamped at each
    row's ends like np.interp. Vectorized (one searchsorted over offset rows)."""
    n, m = xp.shape
    span = float(np.nanmax(xp) - np.nanmin(xp)) + float(np.nanmax(np.abs(x))) + 1.0
    off = (np.arange(n) * 4.0 * span)[:, None]
    flat = (xp + off).ravel()
    j = (
        np.searchsorted(flat, (x + off).ravel(), side="right").reshape(x.shape)
        - np.arange(n)[:, None] * m
    )
    j = np.clip(j, 1, m - 1)
    rows = np.arange(n)[:, None]
    x0, x1 = xp[rows, j - 1], xp[rows, j]
    f0, f1 = fp[rows, j - 1], fp[rows, j]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(x1 > x0, (x - x0) / (x1 - x0), 0.0)
    out = f0 + np.clip(t, 0.0, 1.0) * (f1 - f0)
    out = np.where(x <= xp[:, :1], fp[:, :1], out)
    return np.where(x >= xp[:, -1:], fp[:, -1:], out)


@dataclass
class Params:
    w_cons: float = 1.0  # per mm, times the shift weight
    w_smooth: float = 0.05
    w_vert: float = 0.1
    w_band: float = (
        0.5  # weight of a band's evidence on its level (per unit of stacked profile)
    )
    anchor: float = 5.0  # cost of leaving an open top: its cap
    capture: float = 1.0  # mm over which the anchor's cost rises to its cap
    w_tie: float = 1.0
    min_gap: float = 0.1  # mm between levels
    taper_top: float = 0.16  # no conservation within this of the pia
    taper_wm: float = 0.3  # half weight within this of the WM
    vert: str = "pia"  # the vertical prior: "pia" (each inner level to the pia) or "chain" (to the next inner level)


def build(D, h, centre, half, prm=Params(), tau=None):
    """centre, half: (L, n) mm. Variable index l * n + v. Returns (Model, origins (L, n) labels)."""
    tau = D.tau if tau is None else tau
    L, n = len(tau), D.ns.size
    lo = np.maximum(0.0, centre - half)
    o = np.round(lo / h).astype(int)
    K = np.maximum(1, np.round((centre + half) / h).astype(int) - o + 1)
    var = lambda l, v: l * n + np.asarray(v)  # noqa: E731
    unary = [np.zeros(k) for k in K.ravel()]
    # open-top anchors on the pia level
    for v in np.flatnonzero(np.isfinite(D.top)):
        d = h * (o[-1, v] + np.arange(K[-1, v]))
        # robust with a capture range: rises linearly to the cap over `capture` mm, flat beyond
        off = np.maximum(0.0, np.abs(d - D.top[v]) - h)
        unary[var(L - 1, v)] = np.minimum(prm.anchor, prm.anchor * off / prm.capture)
    # a band's evidence on its level: the column's stacked profile at the label's depth, read
    # through a warp (depth -> tau)
    if D.band_cost is not None:
        wz = np.maximum.accumulate(
            D.warp_z, axis=1
        )  # a warp that dips would make depth -> tau undefined
        wt = np.broadcast_to(D.warp_tg[None], wz.shape)
        tfr = np.broadcast_to(D.band_tf[None], (n, D.band_tf.size))
        for l in range(L - 1):
            if D.band_cost[l] is None:
                continue
            kmax = int(K[l].max())
            d = h * (o[l][:, None] + np.arange(kmax)[None])
            cost = prm.w_band * interp_rows(interp_rows(d, wz, wt), tfr, D.band_cost[l])
            for v in range(n):
                unary[var(l, v)] = unary[var(l, v)] + cost[v, : K[l, v]]
    hinge, hard, tie, table = [], [], [], []
    a, b = D.pairs[:, 0], D.pairs[:, 1]
    q_max = D.S.shape[1] - 1
    c0 = (
        np.where(np.isfinite(D.c0), D.c0, np.inf)
        if D.c0 is not None
        else np.full(n, np.inf)
    )
    for l, t in enumerate(tau):
        # lateral smoothness, all levels
        hinge.append(
            (
                var(l, a),
                var(l, b),
                np.round(t * (D.mu[b] - D.mu[a]) / h).astype(int),
                np.full(a.size, prm.w_smooth * h),
            )
        )
        if t >= 1.0:
            continue
        # conservation, exact over both labels: w |z_w - g(z_v)|, g(z) = z + s_vw(z), tapered by c0
        if a.size:
            Ka, Kb = K[l, a], K[l, b]
            ka, kb = int(Ka.max()), int(Kb.max())
            zv = h * (o[l, a][:, None] + np.arange(ka)[None])  # (P, ka)
            zw = h * (o[l, b][:, None] + np.arange(kb)[None])  # (P, kb)
            # the neighbour map g(z) = z + s_vw(z), projected onto non-decreasing maps (a warp between
            # neighbours cannot fold; folds are estimation errors): then the table is submodular
            g_full = D.g_full()
            q = np.clip(np.round(zv / D.step).astype(int), 0, q_max)
            rows = np.arange(a.size)[:, None]
            s = g_full[rows, q] - zv
            # one weight per pair and level, read at the starting estimate (as the taper): a weight
            # that varied with z_v would break submodularity
            c0a, c0b = c0[a], c0[b]
            q0 = np.clip(
                np.round(t * np.minimum(c0a, D.step * q_max) / D.step).astype(int),
                0,
                q_max,
            )
            w0 = D.W[np.arange(a.size), q0].astype(float)
            w0 = np.where(
                (t * c0a > c0a - prm.taper_top) | (t * c0b > c0b - prm.taper_top),
                0.0,
                w0,
            )
            w0 = np.where(t * c0a < prm.taper_wm, 0.5 * w0, w0)
            T = (
                prm.w_cons
                * w0[:, None, None]
                * np.abs(zw[:, None, :] - zv[:, :, None] - s[:, :, None])
            )
            valid = (np.arange(ka)[None, :, None] < Ka[:, None, None]) & (
                np.arange(kb)[None, None, :] < Kb[:, None, None]
            )
            T = np.where(valid, T, np.nan)
            keep = np.nanmax(np.where(valid, T, 0.0), axis=(1, 2)) > 0
            if keep.any():
                table.append((var(l, a[keep]), var(l, b[keep]), T[keep]))
        # ordering, per column
        v = np.arange(n)
        hard.append(
            (
                var(l, v),
                var(l + 1, v),
                np.full(n, int(np.ceil(prm.min_gap / h - 1e-9))),
                np.full(n, 10**6),
            )
        )
        # vertical prior, exact: w |z_l - r z_m| with m the pia (r = tau_l) or, for "chain", the next
        # inner level (r = tau_l / tau_(l+1)); the last inner level has none in "chain"
        if prm.vert == "chain":
            if l + 1 >= L - 1:
                continue
            m, ratio = l + 1, t / tau[l + 1]
        else:
            m, ratio = L - 1, t
        kl, km = int(K[l].max()), int(K[m].max())
        zl = h * (o[l][:, None] + np.arange(kl)[None])
        zm = h * (o[m][:, None] + np.arange(km)[None])
        TV = prm.w_vert * np.abs(zl[:, :, None] - ratio * zm[:, None, :])
        validv = (np.arange(kl)[None, :, None] < K[l][:, None, None]) & (
            np.arange(km)[None, None, :] < K[m][:, None, None]
        )
        table.append((var(l, v), var(m, v), np.where(validv, TV, np.nan)))
    if D.ties.size:
        tie.append(
            (
                var(L - 1, D.ties[:, 0]),
                var(L - 1, D.ties[:, 1]),
                np.round(D.tie_L / h).astype(int),
                prm.w_tie * h * D.tie_w,
            )
        )
    return WC.Model(K.ravel(), o.ravel(), unary, hard, hinge, tie, table), o


def depths(M, o, labels, h, L):
    return h * (np.asarray(M.o) + labels).reshape(L, -1)


def edge_hits(M, labels):
    """Variables at a window edge that is not the depth range's own end (origin 0 at the bottom)."""
    K, o = np.asarray(M.K), np.asarray(M.o)
    low = (labels == 0) & (o > 0)
    high = (labels == K - 1) & (K > 1)
    return low | high


def solve_pass(D, h, centre, half, prm=Params(), solver=WC.qpbo, recentre=3, log=None):
    """One pass. Windows hit at an edge are re-centred on the solution and the pass repeated."""
    say = log or (lambda m: None)
    L = centre.shape[0]
    hist = []
    for it in range(recentre + 1):
        M, o = build(D, h, centre, half, prm)
        warm = None
        if (
            it > 0
        ):  # the previous solution, inside the new window: QPBO-I never ends above it
            warm = np.clip(
                np.round(z.ravel() / h).astype(int) - np.asarray(M.o),
                0,
                np.asarray(M.K) - 1,
            )
        r = solver(M, start_labels=warm, log=log)
        z = depths(M, o, r.labels, h, L)
        hits = edge_hits(M, r.labels).reshape(L, -1)
        low = ((r.labels == 0) & (np.asarray(M.o) > 0)).reshape(L, -1)
        hist.append(
            dict(
                edge_hits=int(hits.sum()),
                energy=r.energy,
                lower_bound=r.lower_bound,
                certified=float(np.mean(r.certified)),
                hits=hits,
                low=low,
                centre=centre.copy(),
                z=z,
            )
        )
        say(
            f"pass h={h}: try {it}: energy {r.energy:.3f}, bound {r.lower_bound:.3f}, "
            f"certified {np.mean(r.certified):.1%}, edge hits {int(hits.sum())}"
        )
        if not hits.any():
            break
        centre = np.where(hits, z, centre)
    converged = not hits.any()
    hist[-1]["converged"] = converged
    if not converged:
        say(
            f"WARNING pass h={h}: not converged after {recentre} re-centrings: {int(hits.sum())} variables "
            f"still at a window edge"
        )
    cert = np.asarray(r.certified).reshape(L, -1)
    return z, cert, r, hist


def coarse_to_fine(
    D,
    start,
    prm=Params(),
    h_coarse=0.16,
    h_fine=0.04,
    half_coarse=1.0,
    half_cert=0.16,
    half_unc=0.48,
    recentre=20,
    log=None,
):
    """start: (L, n) mm. Returns fine depths (L, n), the fine result dict and both passes' history."""
    zc, cc, rc, hc = solve_pass(
        D,
        h_coarse,
        start,
        np.full(start.shape, half_coarse),
        prm,
        WC.qpbo,
        recentre,
        log=log,
    )
    half = np.where(cc, half_cert, half_unc)
    zf, cf, rf, hf = solve_pass(
        D, h_fine, zc, half, prm, WC.qpbo_pruned, recentre, log=log
    )
    return zf, cf, rf, dict(coarse=hc, fine=hf, coarse_depths=zc)


def fill_warp(
    pairs,
    S,
    W,
    step,
    levels,
    tau=TAU,
    J=24,
    eps=0.2,
    anchor=10.0,
    taper_top=0.16,
    n_outer=3,
    maxiter=3000,
    rtol=1e-7,
    log=None,
):
    """The continuous warp z_v(tau), tau = j / J, between solved levels (L, n) at `tau` (the last is
    the pia; J must put every tau on the grid). Least squares:
      - z_v(0) = 0 and z_v(tau_l) = level (weight `anchor`);
      - conservation z_w(tau) - z_v(tau) = s_vw(z_v(tau)), weight W, linearized at the current warp
        (Gauss-Newton), counted only where both samples lie more than taper_top below their own
        column's pia: nothing past the pia enters;
      - eps ((z_(j+1) - z_j) - (the same for the straight fill between levels)): a weak pull to the
        straight fill (distance; equal volume when its estimator is validated).
    Returns tg (J + 1,), z (n, J + 1), info."""
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import cg

    from flatten_lib import interp_last

    L, n = levels.shape
    tg = np.arange(J + 1) / J
    jl = np.round(np.asarray(tau) * J).astype(int)
    assert np.allclose(jl / J, tau), "every level must lie on the tau grid"
    pia = levels[-1]
    knots_t = np.r_[0.0, tau]
    zl = np.stack([np.interp(tg, knots_t, np.r_[0.0, levels[:, v]]) for v in range(n)])
    z = zl.copy()
    idx = np.arange(n * (J + 1)).reshape(n, J + 1)
    # fixed rows: anchors and the vertical pull
    ra = np.r_[idx[:, 0], idx[:, jl].ravel()]
    ba = np.r_[np.zeros(n), levels.T.ravel()]
    Aa = coo_matrix(
        (np.full(ra.size, anchor), (np.arange(ra.size), ra)),
        shape=(ra.size, n * (J + 1)),
    ).tocsr()
    ev = idx[:, :-1].ravel()
    re_ = np.arange(ev.size)
    Ae = coo_matrix(
        (
            np.r_[np.full(ev.size, -eps), np.full(ev.size, eps)],
            (np.r_[re_, re_], np.r_[ev, ev + 1]),
        ),
        shape=(ev.size, n * (J + 1)),
    ).tocsr()
    be = eps * np.diff(zl, axis=1).ravel()
    R = (Aa.T @ Aa + Ae.T @ Ae).tocsr()
    rR = Aa.T @ (anchor * ba) + Ae.T @ be
    v, w = pairs[:, 0], pairs[:, 1]
    info = []
    for it in range(n_outer):
        q = z[v] / step
        s, ins = interp_last(S, q)
        wt, _ = interp_last(W, q)
        inside = (
            (z[v] < pia[v][:, None] - taper_top)
            & (z[w] < pia[w][:, None] - taper_top)
            & (tg[None] < 1.0)
        )
        wt = np.where(ins & inside, np.maximum(wt, 0.0), 0.0)
        p_, j_ = np.nonzero(wt > 0)
        sw = np.sqrt(wt[p_, j_])
        rows = np.arange(p_.size)
        Ad = coo_matrix(
            (
                np.r_[sw, -sw],
                (np.r_[rows, rows], np.r_[idx[w[p_], j_], idx[v[p_], j_]]),
            ),
            shape=(p_.size, n * (J + 1)),
        ).tocsr()
        bd = sw * s[p_, j_]
        M = (Ad.T @ Ad + R).tocsr()
        rhs = Ad.T @ bd + rR
        dg = M.diagonal()
        x, flag = cg(
            M,
            rhs,
            x0=z.ravel(),
            rtol=rtol,
            maxiter=maxiter,
            M=diags(1.0 / np.where(dg > 0, dg, 1.0)),
        )
        z = x.reshape(n, J + 1)
        rms = float(np.sqrt(np.sum((Ad @ x - bd) ** 2) / max(np.sum(sw**2), 1e-12)))
        info.append(
            dict(
                outer=it, cg_flag=int(flag), data_rows=int(p_.size), weighted_rms_mm=rms
            )
        )
        if log:
            log(
                f"fill outer {it}: {p_.size:,} data rows, cg flag {flag}, weighted rms {rms:.4f} mm"
            )
    return tg, z, info
