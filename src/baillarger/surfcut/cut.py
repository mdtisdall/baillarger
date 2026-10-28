"""The exact minimum of the submodular terms, by a minimum s-t cut.

Ishikawa's construction (2003, "Exact optimization for Markov random fields
with convex priors"): node (i, k) stands for x_i >= k, chained by infinite
arcs so that each variable's nodes on the source side form a prefix; the
labels are then a minimum closed set (Picard 1976). Unary costs become
terminal arcs (telescoped); a hard bound or an L1 hinge on a difference
becomes arcs between two chains; a submodular table becomes, by its second
differences, arcs between two chains (Schlesinger and Flach 2006). Ties and
the positive second differences of tables are not submodular and are left
out here (baillarger.surfcut.qpbo takes them).
"""

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_array
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

from baillarger.surfcut.model import (
    Float,
    Int,
    Model,
    floats,
    hard_terms,
    hinge_terms,
    ints,
    table_parts,
)

SCALE = 1000  # costs to integers for scipy's maximum flow


def _chunks(m: int, size: int) -> Iterator[slice]:
    for c in range(0, m, size):
        yield slice(c, min(c + size, m))


def graph(
    M: Model, extra_unary: dict[int, Float] | None = None, chunk: int = 100000
) -> tuple[Int, Int, Float, Int, int, int]:
    """The s-t graph of unary, hard, hinge and the submodular part of table
    terms (ties left out), as arcs with float capacities (inf for hard arcs).
    Node start[i] + k is 'x_i >= k'; node start[i] is always on the source
    side. Returns rows, cols, caps, start, s, t."""
    n, K, o = M.n, ints(M.K), ints(M.o)
    start = np.r_[0, np.cumsum(K)].astype(np.int64)
    N = int(start[-1])
    s, t_ = N, N + 1
    U = np.concatenate([floats(u) for u in M.unary]) if n else floats([])
    for i, v in (extra_unary or {}).items():
        U[start[i] : start[i + 1]] += v
    var = np.repeat(np.arange(n), K)
    lab = np.arange(N) - start[var]
    R: list[Int] = []
    C: list[Int] = []
    V: list[Float] = []

    def add(f: npt.ArrayLike, tt: npt.ArrayLike, c: npt.ArrayLike) -> None:
        fr = ints(f).ravel()
        R.append(fr)
        C.append(np.broadcast_to(ints(tt), fr.shape).astype(np.int64).ravel())
        V.append(np.broadcast_to(floats(c), fr.shape).astype(np.float64).ravel())

    a_h, b_h, lo_h, hi_h = hard_terms(M)
    a_g, b_g, t_g, w_g = hinge_terms(M)
    # hinges: w max(0, x_b - x_a - t) and w max(0, x_a - x_b + t)
    for A_, B_, T_, W_ in ((a_g, b_g, t_g, w_g), (b_g, a_g, -t_g, w_g)):
        for sl in _chunks(A_.size, chunk):
            A, B, W = A_[sl], B_[sl], W_[sl]
            tl = (T_[sl] - (o[B] - o[A]))[
                :, None
            ]  # label space: max(0, k_b - k_a - tl)
            Ka, Kb = K[A][:, None], K[B][:, None]
            kk = np.arange(int(Ka.max()))[None]
            val = W[:, None] * np.maximum(0, -tl - kk)  # the j <= 0 part, a unary on a
            m0 = (kk < Ka) & (val > 0)
            U += np.bincount((start[A][:, None] + kk)[m0], val[m0], minlength=N)
            j = np.arange(1, max(int(Kb.max()), 2))[None]
            mj = j < Kb
            ja = j - tl
            m = mj & (ja >= 1) & (ja <= Ka - 1)
            Wb = np.broadcast_to(W[:, None], m.shape)
            add((start[B][:, None] + j)[m], (start[A][:, None] + ja)[m], Wb[m])
            m2 = mj & (ja >= Ka)
            add((start[B][:, None] + j)[m2], t_, Wb[m2])
    # tables: D <= 0 gives D y_ai y_bj = D y_ai + (-D) y_ai (1 - y_bj),
    # a unary on a and an arc (a, i) -> (b, j)
    for ta, tb, T in M.table:
        ta, tb = ints(ta), ints(tb)
        _, ua, ub, D = table_parts(ta, tb, floats(T), K)
        Dn = np.minimum(D, 0.0)
        ia = np.arange(T.shape[1])[None]
        ma = ia < K[ta][:, None]
        cum = np.cumsum(Dn.sum(2), axis=1)  # sum_{i <= k} sum_j D(i, j)
        U += np.bincount((start[ta][:, None] + ia)[ma], (ua + cum)[ma], minlength=N)
        jb = np.arange(T.shape[2])[None]
        mb = jb < K[tb][:, None]
        U += np.bincount((start[tb][:, None] + jb)[mb], ub[mb], minlength=N)
        p_, i_, j_ = np.nonzero(Dn < 0)
        add(start[ta][p_] + i_, start[tb][p_] + j_, -Dn[p_, i_, j_])
    # unary -> terminal arcs (telescoped); the chains; the source side
    nd = np.flatnonzero(lab >= 1)
    d = U[nd] - U[nd - 1]
    add(np.full(int((d < 0).sum()), s), nd[d < 0], -d[d < 0])
    add(nd[d > 0], t_, d[d > 0])
    add(np.full(n, s), start[:-1], np.inf)
    add(nd, nd - 1, np.inf)
    # hard: x_b >= x_a + lo; x_a >= x_b - hi
    for A_, B_, X_ in ((a_h, b_h, lo_h), (b_h, a_h, -hi_h)):
        for sl in _chunks(A_.size, chunk):
            A, B = A_[sl], B_[sl]
            xl = (X_[sl] - (o[B] - o[A]))[:, None]
            Ka, Kb = K[A][:, None], K[B][:, None]
            k = np.arange(int(Ka.max()))[None]
            mk = k < Ka
            j = k + xl
            m = mk & (j >= 1) & (j <= Kb - 1)
            add((start[A][:, None] + k)[m], (start[B][:, None] + j)[m], np.inf)
            m2 = mk & (j >= Kb)
            add((start[A][:, None] + k)[m2], t_, np.inf)
    rows, cols, caps = np.concatenate(R), np.concatenate(C), np.concatenate(V)
    keep = caps > 0
    return rows[keep], cols[keep], caps[keep], start, s, t_


def cut(M: Model, extra_unary: dict[int, Float] | None = None) -> Int:
    """The exact minimum of unary, hard and hinge terms and the submodular
    part of the tables (ties and positive second differences left out).
    extra_unary: costs added to some variables' unary terms. Returns the
    labels (n,). Raises ValueError if the hard bounds are infeasible. Costs
    are rounded to 1 / SCALE."""
    rows, cols, caps, start, s, t_ = graph(M, extra_unary)
    N = s
    inf = ~np.isfinite(caps)
    ci = np.zeros(caps.size, np.int64)
    ci[~inf] = np.round(caps[~inf] * SCALE).astype(np.int64)
    INF = int(ci.sum()) + 1
    ci[inf] = INF
    keep = ci > 0
    G = coo_array(
        (ci[keep], (rows[keep].astype(np.int32), cols[keep].astype(np.int32))),
        shape=(N + 2, N + 2),
    ).tocsr()
    G.sum_duplicates()
    res = maximum_flow(G, s, t_, method="dinic")
    if res.flow_value >= INF:
        raise ValueError("the hard bounds are infeasible")
    resid = (G - res.flow).tocsr()
    resid.data = (resid.data > 0).astype(np.int8)
    resid.eliminate_zeros()
    S = np.zeros(N + 2, bool)
    S[breadth_first_order(resid, s, directed=True, return_predecessors=False)] = True
    return (np.add.reduceat(S[:N].astype(np.int64), start[:-1]) - 1).astype(np.int64)
