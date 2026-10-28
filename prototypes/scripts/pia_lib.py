"""The first-pia horizon (plan: reports/Crossing paths and contacts from laminar profiles.md,
*The first-pia horizon*). Pure functions, no argv.

- min_surface: one label c_v per column minimizing sum_v D[v, c_v] subject to
  lo_p <= c_w - c_v <= hi_p for every pair p = (v, w), exactly, as a minimum closed set (Li et al.
  2006; Picard 1976) by a minimum s-t cut.
- bound_labels: the label range of a pair from a bound on detrended thickness.
- anchor_penalty: the cost a trace pays for leaving the range an open neighbour's top allows.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

SCALE = 1000  # costs to integers


def min_surface(D, pairs, lo, hi):
    """Labels c (n,) in [0, K) minimizing sum D[v, c_v] with lo <= c_w - c_v <= hi per pair.

    D (n, K) finite costs; pairs (P, 2); lo, hi (P,) integer label offsets with lo <= hi.
    Returns c, or raises if the bounds alone are infeasible."""
    n, K = D.shape
    Di = np.round(np.asarray(D, float) * SCALE).astype(np.int64)
    w = np.empty_like(Di)
    w[:, 1:] = Di[:, 1:] - Di[:, :-1]
    w[:, 0] = 0
    INF = int(np.abs(w).sum()) + 1
    node = np.arange(n * K).reshape(n, K)
    s, t = n * K, n * K + 1
    R, C, V = [], [], []

    def arcs(a, b, cap):
        R.append(np.asarray(a).ravel().astype(np.int32))
        C.append(np.asarray(b).ravel().astype(np.int32))
        V.append(np.broadcast_to(np.int64(cap) if np.isscalar(cap) else cap, np.asarray(a).ravel().shape).ravel())

    # every column's bottom is in the closed set
    arcs(np.full(n, s), node[:, 0], INF)
    # the closed set is a lower set in each column
    arcs(node[:, 1:], node[:, :-1], INF)
    # c_w >= c_v + lo: (v, k) -> (w, k + lo); c_v >= c_w - hi: (w, k) -> (v, k - hi)
    k = np.arange(K)
    for (a, b, off) in ((pairs[:, 0], pairs[:, 1], lo), (pairs[:, 1], pairs[:, 0], -hi)):
        j = k[None] + np.asarray(off)[:, None]  # (P, K) target label on b
        src = node[a][:, :]
        inside = (j >= 1) & (j <= K - 1)
        arcs(src[inside], node[b][:, :][inside.nonzero()[0], j[inside]], INF)
        over = j > K - 1  # (a, k) cannot be in the set
        arcs(src[over], np.full(int(over.sum()), t), INF)
    neg, pos = w < 0, w > 0
    arcs(np.full(int(neg.sum()), s), node[neg], -w[neg])
    arcs(node[pos], np.full(int(pos.sum()), t), w[pos])
    rows, cols, caps = np.concatenate(R), np.concatenate(C), np.concatenate(V)
    G = coo_matrix((caps, (rows, cols)), shape=(n * K + 2,) * 2).tocsr()
    G.sum_duplicates()
    res = maximum_flow(G, s, t, method="dinic")
    if res.flow_value >= INF:
        raise ValueError("the bounds are infeasible")
    resid = (G - res.flow).tocsr()
    resid.data = (resid.data > 0).astype(np.int8)
    resid.eliminate_zeros()
    reach = breadth_first_order(resid, s, directed=True, return_predecessors=False)
    S = np.zeros(n * K + 2, bool)
    S[reach] = True
    inS = S[:n * K].reshape(n, K)
    return K - 1 - np.argmax(inS[:, ::-1], axis=1)


def min_surfaces(grids, costs, cons):
    """Several surfaces per column, each on its own depth grid, exactly (Li et al. 2006's
    multi-surface construction; any increasing grids).

    grids: list over surfaces of increasing depth arrays (K_s,); costs: list of (n, K_s) finite
    costs; cons: list of (sa, sb, ia, ib, lo, hi): depth_sb[ib] - depth_sa[ia] in [lo, hi] (mm),
    arrays over constraints. Returns a list over surfaces of label indices (n,)."""
    n = costs[0].shape[0]
    Ks = [len(g) for g in grids]
    off = np.cumsum([0] + [n * k for k in Ks])
    N = int(off[-1])
    s, t = N, N + 1
    node = [off[i] + np.arange(n * k).reshape(n, k) for i, k in enumerate(Ks)]
    w_all, R, C, V = [], [], [], []

    def arcs(a, b, cap):
        a = np.asarray(a).ravel()
        R.append(a.astype(np.int32))
        C.append(np.asarray(b).ravel().astype(np.int32))
        V.append(np.full(a.size, cap, np.int64) if np.isscalar(cap) else np.asarray(cap, np.int64).ravel())

    for D in costs:
        Di = np.round(np.asarray(D, float) * SCALE).astype(np.int64)
        w = np.empty_like(Di)
        w[:, 1:] = Di[:, 1:] - Di[:, :-1]
        w[:, 0] = 0
        w_all.append(w)
    INF = int(sum(np.abs(w).sum() for w in w_all)) + 1
    for i in range(len(grids)):
        arcs(np.full(n, s), node[i][:, 0], INF)
        arcs(node[i][:, 1:], node[i][:, :-1], INF)
    for sa, sb, ia, ib, lo, hi in cons:
        ga, gb = np.asarray(grids[sa]), np.asarray(grids[sb])
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        # depth_b >= depth_a + lo: (a, k) -> (b, first j with gb[j] >= ga[k] + lo)
        # depth_a >= depth_b - hi: (b, k) -> (a, first j with ga[j] >= gb[k] - hi)
        for (A_, B_, gA, gB, ia_, ib_, x) in ((sa, sb, ga, gb, ia, ib, lo), (sb, sa, gb, ga, ib, ia, -hi)):
            j = np.searchsorted(gB, gA[None, :] + x[:, None] - 1e-9, side="left")  # (P, K_A)
            src = node[A_][ia_]  # (P, K_A)
            inside = (j >= 1) & (j <= len(gB) - 1)
            pi, ki = np.nonzero(inside)
            arcs(src[pi, ki], node[B_][ib_[pi], j[pi, ki]], INF)
            over = j > len(gB) - 1
            arcs(src[over], np.full(int(over.sum()), t), INF)
            del j, src, inside, over, pi, ki
    for i, w in enumerate(w_all):
        neg, pos = w < 0, w > 0
        arcs(np.full(int(neg.sum()), s), node[i][neg], -w[neg])
        arcs(node[i][pos], np.full(int(pos.sum()), t), w[pos])
    rows, cols, caps = np.concatenate(R), np.concatenate(C), np.concatenate(V)
    del R, C, V
    G = coo_matrix((caps, (rows, cols)), shape=(N + 2,) * 2).tocsr()
    del rows, cols, caps
    G.sum_duplicates()
    res = maximum_flow(G, s, t, method="dinic")
    if res.flow_value >= INF:
        raise ValueError("the bounds are infeasible")
    resid = (G - res.flow).tocsr()
    del G, res
    resid.data = (resid.data > 0).astype(np.int8)
    resid.eliminate_zeros()
    reach = breadth_first_order(resid, s, directed=True, return_predecessors=False)
    S = np.zeros(N + 2, bool)
    S[reach] = True
    out = []
    for i, k in enumerate(Ks):
        inS = S[off[i]:off[i + 1]].reshape(n, k)
        out.append(k - 1 - np.argmax(inS[:, ::-1], axis=1))
    return out


def snap_out(lo, hi, h):
    """Widen [lo, hi] (mm) outward to multiples of the grid step h."""
    return np.floor(np.asarray(lo) / h + 1e-9) * h, np.ceil(np.asarray(hi) / h - 1e-9) * h


def bound_labels(dmu, gd, step):
    """lo, hi label offsets of c_w - c_v from |(c_w - mu_w) - (c_v - mu_v)| <= g d, with
    dmu = mu_w - mu_v and gd = g d (mm). Rounded outward (never tighter than the bound)."""
    return np.floor((dmu - gd) / step).astype(np.int64), np.ceil((dmu + gd) / step).astype(np.int64)


def anchor_penalty(depths, top, dmu, gd, big):
    """Cost (len(depths),) of each depth for a trace next to an open trace whose top is `top`:
    0 inside [top + dmu - gd, top + dmu + gd] (mm), `big` outside."""
    return np.where((depths >= top + dmu - gd - 1e-9) & (depths <= top + dmu + gd + 1e-9), 0.0, big)
