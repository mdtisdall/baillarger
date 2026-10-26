"""Band boundaries by optimal surfaces (plan step 4): one exact min-cut for
the K − 1 boundaries of all the columns of a region.

For fixed gain/offset and template, surface k of column c sits at one of the
candidate depths d_k: a window around Z_R,k, in steps of zstep, of ± half the
narrower of the two bands next to the surface (candidates). The plan and the
prototype used ± half the region's narrowest band for every surface; on
V1V2 that is the thin band at the pial edge (r 0.94 to 1), so every surface
could move only ± 0.02 to 0.03, and 30 % of the columns' surfaces stopped at
the window's edge; with the window of each surface, 17 %, and the data term
6 % lower after two passes (window="region" keeps the prototype's).

With e_k(v) = Σ_e (x_e − a_ce − s_ce μ_k,e)² / σ²_ce, the data term splits
into one cost per surface: Σ_v e_{k(v)}(v) = Σ_v e_1(v) + Σ_k Σ_{v: m(v) ≥ z_k}
(e_{k+1}(v) − e_k(v)), so the cost of surface k at depth d is a sum over the
column's voxels above d. Each
candidate after the first is a binary node x_ckp = [z_ck ≥ d_k[p]] (Ishikawa;
Li, Wu, Chen and Sonka 2006):

- unary: the data cost and the pull η (z − Z_R,k)², through the source and
  sink edges;
- x_ckp = 1 implies x_ck,p−1 = 1 (infinite edges), so each column's nodes
  give one depth;
- the minimum thickness: z_k ≥ d implies z_{k+1} ≥ d + t_min;
- the TV between touching columns of the region: λ_z w_cc' |z_ck − z_c'k|, as
  λ_z w_cc' × (the step to the next candidate) on the edge between their
  nodes of the same candidate.

The minimum cut is the global optimum of this step. scipy's maximum_flow
takes 32-bit integer capacities, so the costs are scaled to integers, with
the total of the finite capacities below 2^30 (an infinite edge is one more
than that total); the cut is then optimal to within the rounding, about
(number of cut edges) x (total cost) / 2^31. PyMaxflow (the prototype's
solver, real-valued) is GPL-licensed and not a dependency.
"""

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

type Bool = npt.NDArray[np.bool_]
type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]


def candidates(
    Z: F64, zstep: float, window: Literal["surface", "region"] = "surface"
) -> list[F64]:
    """The candidate depths of each surface: a window around Z_k, in steps of
    zstep, inside (0, 1). window="surface": ± half the narrower of the two
    bands next to the surface; "region": ± half the region's narrowest band
    (the prototype). Never less than zstep."""
    gaps = np.diff(np.concatenate([[0.0], Z, [1.0]]))
    out = []
    for k, z in enumerate(Z):
        if window == "surface":
            half = max(0.5 * min(gaps[k], gaps[k + 1]), zstep)
        else:
            half = max(0.5 * gaps.min(), zstep)
        lo, hi = max(z - half, zstep), min(z + half, 1.0 - zstep)
        out.append(
            z
            + zstep
            * np.arange(np.ceil((lo - z) / zstep), np.floor((hi - z) / zstep) + 1)
        )
    return out


def st_min_cut(
    n: int,
    tail: I64,
    head: I64,
    cap: F64,
    src_cap: F64,
    snk_cap: F64,
    inf_edges: Bool | None = None,
) -> Bool:
    """Minimum s-t cut of a graph of n nodes with real capacities: edges
    tail -> head (cap; where inf_edges, infinite), and from the source to each
    node (src_cap, paid when the node is on the sink side) and from each node
    to the sink (snk_cap, inf where infinite; paid when on the source side).
    Returns the source side (bool over the nodes)."""
    inf_e = np.zeros(len(cap), bool) if inf_edges is None else inf_edges
    inf_t = ~np.isfinite(snk_cap)
    finite = cap[~inf_e].sum() + src_cap.sum() + snk_cap[~inf_t].sum()
    scale = (2.0**30) / finite if finite > 0 else 1.0
    big = int(round(finite * scale)) + 1

    def q(x: F64) -> I64:
        out: I64 = np.round(x * scale).astype(np.int64)
        return out

    s, t = n, n + 1
    c_e = np.where(inf_e, big, q(np.where(inf_e, 0.0, cap)))
    c_t = np.where(inf_t, big, q(np.where(inf_t, 0.0, snk_cap)))
    rows = np.concatenate([tail, np.full(n, s), np.arange(n)])
    cols = np.concatenate([head, np.arange(n), np.full(n, t)])
    c = np.concatenate([c_e, q(src_cap), c_t])
    keep = c > 0
    G = sparse.csr_matrix(
        (c[keep].astype(np.int32), (rows[keep], cols[keep])), shape=(n + 2, n + 2)
    )
    G.sum_duplicates()
    if G.data.size and int(G.data.max()) >= 2**31 - 1:
        raise OverflowError("capacities do not fit 32 bits")
    # scipy accepts csr_matrix here; its stubs list only csr_array
    F = maximum_flow(G, s, t, method="dinic").flow  # type: ignore[arg-type]
    R: Any = (G - F).tocsr()  # residual capacities (F is antisymmetric)
    R.data = (R.data > 0).astype(np.int8)
    R.eliminate_zeros()
    order = breadth_first_order(R, s, directed=True, return_predecessors=False)
    side = np.zeros(n + 2, bool)
    side[order] = True
    out: Bool = side[:n]
    return out


def surfaces_region(
    cols: I64,
    col: I64,
    m: F64,
    f: F64,
    sigma2: F64,
    Z: F64,
    mu: F64,
    a: F64,
    s: F64,
    edges: I64,
    ew: F64,
    zstep: float,
    eta: float,
    lambda_z: float,
    t_min: float,
    window: Literal["surface", "region"] = "surface",
) -> F64:
    """The boundaries z (len(cols), K − 1) of the columns cols (sorted ids)
    of one region.

    col, m, f: the region's cortex voxels (column ids, r, echo values (V, E));
    sigma2, a, s: (n, E) for all the columns; Z (K − 1), mu (K, E): the
    region's bands and template; edges, ew: the links between columns of the
    region.
    """
    K = len(mu)
    nc = len(cols)
    if K < 2:
        return np.zeros((nc, 0))
    cand = candidates(Z, zstep, window)
    pos = np.full(int(max(cols.max(), col.max())) + 1, -1, np.int64)
    pos[cols] = np.arange(nc)
    ci = pos[col]
    # unary costs U[k][i, p]: surface k of column i at d_k[p]
    U = []
    for k in range(K - 1):
        res1 = f - a[col] - s[col] * mu[k + 1]
        res0 = f - a[col] - s[col] * mu[k]
        de = np.sum((res1**2 - res0**2) / sigma2[col], axis=1)
        d = cand[k]
        pidx = np.searchsorted(d, m, side="right")  # m above candidates < pidx
        acc = np.zeros((nc, len(d) + 1))
        np.add.at(acc, (ci, pidx), de)
        tail_sum = np.cumsum(acc[:, ::-1], 1)[:, ::-1]  # voxels with m >= d[p]
        U.append(tail_sum[:, 1:] + eta * (d - Z[k]) ** 2)
    # nodes: x_ckp for p >= 1
    sizes = [nc * (len(d) - 1) for d in cand]
    base = np.concatenate([[0], np.cumsum(sizes)])
    nn = int(base[-1])

    def node(k: int, i: Any, p: Any) -> Any:
        return base[k] + np.asarray(i) * (len(cand[k]) - 1) + (np.asarray(p) - 1)

    src_cap = np.zeros(nn)
    snk_cap = np.zeros(nn)
    T: list[I64] = []
    H: list[I64] = []
    C: list[F64] = []
    INF: list[Bool] = []

    def add(tl: Any, hd: Any, cp: Any, inf: bool) -> None:
        tl_a, hd_a = np.asarray(tl), np.asarray(hd)
        cp_a = np.broadcast_to(np.asarray(cp, float), tl_a.shape)
        T.append(tl_a.ravel().astype(np.int64))
        H.append(hd_a.ravel().astype(np.int64))
        C.append(cp_a.ravel().astype(float))
        INF.append(np.full(tl_a.size, inf))

    ii = np.arange(nc)
    for k in range(K - 1):
        Pk = len(cand[k])
        if Pk < 2:
            continue
        dU = np.diff(U[k], axis=1)  # coefficient of x_ckp
        ids = node(k, ii[:, None], np.arange(1, Pk)[None, :])
        src_cap[ids] = np.where(dU < 0, -dU, 0.0)  # paid when x = 0
        snk_cap[ids] = np.where(dU > 0, dU, 0.0)  # paid when x = 1
        if Pk > 2:  # x_p = 1 implies x_{p-1} = 1
            add(ids[:, 1:], ids[:, :-1], 0.0, True)
    # ordering with the minimum thickness: z_k >= d_k[p] => z_k+1 >= d_k[p] + t_min
    for k in range(K - 2):
        dk, dn = cand[k], cand[k + 1]
        for pp in range(1, len(dk)):
            need = dk[pp] + t_min - 1e-9
            qn = int(np.searchsorted(dn, need))
            if qn == 0:
                continue
            if qn >= len(dn):
                snk_cap[node(k, ii, pp)] = np.inf
            else:
                add(node(k, ii, pp), node(k + 1, ii, qn), 0.0, True)
    # TV between touching columns of the region
    if lambda_z > 0 and len(edges):
        ei, ej = pos[edges[:, 0]], pos[edges[:, 1]]
        for k in range(K - 1):
            dd = np.diff(cand[k])
            if len(dd) == 0:
                continue
            capm = lambda_z * ew[:, None] * dd[None, :]
            p1 = np.arange(1, len(cand[k]))[None, :]
            a_ids = node(k, ei[:, None], p1)
            b_ids = node(k, ej[:, None], p1)
            add(a_ids, b_ids, capm, False)
            add(b_ids, a_ids, capm, False)
    z = np.zeros((nc, K - 1))
    if nn == 0:
        for k in range(K - 1):
            z[:, k] = cand[k][0]
        return z
    tail = np.concatenate(T) if T else np.zeros(0, np.int64)
    head = np.concatenate(H) if H else np.zeros(0, np.int64)
    cap = np.concatenate(C) if C else np.zeros(0)
    inf = np.concatenate(INF) if INF else np.zeros(0, bool)
    side = st_min_cut(nn, tail, head, cap, src_cap, snk_cap, inf)
    for k in range(K - 1):
        Pk = len(cand[k])
        if Pk < 2:
            z[:, k] = cand[k][0]
            continue
        x = side[node(k, ii[:, None], np.arange(1, Pk)[None, :])]
        z[:, k] = cand[k][x.sum(1)]
    return z
