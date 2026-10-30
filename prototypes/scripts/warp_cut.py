"""The joint warp's levels (plan: reports/Crossing paths and contacts from laminar profiles.md,
*The joint warp*). Pure functions, no argv.

Variables sit on one depth lattice of step h: variable i takes label k in [0, K_i), at depth
h (o_i + k) with its own integer origin o_i. Terms, on absolute positions x_i = o_i + k_i:
- unary costs (per variable, per label);
- hard bounds lo <= x_b - x_a <= hi;
- L1 hinges w |x_b - x_a - t|, w >= 0;
- ties w |x_a + x_b - L| (a sum: not submodular on a loop);
- pairwise tables theta(k_a, k_b): submodular where every second difference
  D(i, j) = theta(i, j) - theta(i-1, j) - theta(i, j-1) + theta(i-1, j-1) <= 0.

- energy: the total, ties included.
- cut: the exact minimum of unary + hard + hinge terms and the submodular part of the tables
  (positive second differences dropped), ties ignored (Ishikawa 2003; Schlesinger and Flach 2006;
  a minimum closed set, Picard 1976, by a minimum s-t cut). The reference, and QPBO-I's start.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

SCALE = 1000
BIG_FACTOR = 1e3  # hard terms in qpbo: this times the sum of all finite costs


@dataclass
class Model:
    K: np.ndarray  # (n,) labels per variable
    o: np.ndarray  # (n,) integer origins
    unary: list  # per variable: (K_i,) costs
    hard: list = field(default_factory=list)  # (a, b, lo, hi) arrays
    hinge: list = field(default_factory=list)  # (a, b, t, w) arrays
    tie: list = field(default_factory=list)  # (a, b, L, w) arrays
    table: list = field(
        default_factory=list
    )  # (a, b, T): T (m, Ka_max, Kb_max), T[p, k_a, k_b]; nan padding

    @property
    def n(self):
        return len(self.K)


def _cat(terms, k):
    if not terms:
        return [np.zeros(0, int if j < 2 or k == "hard" else float) for j in range(4)]
    return [
        np.concatenate([np.atleast_1d(np.asarray(t[j])) for t in terms])
        for j in range(4)
    ]


def energy(M, k):
    """Total energy of labels k (n,), ties included; inf if a hard bound is broken."""
    k = np.asarray(k, int)
    x = np.asarray(M.o, int) + k
    start = np.r_[0, np.cumsum(np.asarray(M.K, int))]
    e = (
        float(
            np.concatenate([np.asarray(u, float) for u in M.unary])[
                start[:-1] + k
            ].sum()
        )
        if M.n
        else 0.0
    )
    a, b, lo, hi = _cat(M.hard, "hard")
    if a.size and np.any((x[b] - x[a] < lo) | (x[b] - x[a] > hi)):
        return np.inf
    a, b, t, w = _cat(M.hinge, "hinge")
    e += float(np.sum(w * np.abs(x[b] - x[a] - t)))
    a, b, L, w = _cat(M.tie, "tie")
    e += float(np.sum(w * np.abs(x[a] + x[b] - L)))
    for a, b, T in M.table:
        a, b = np.asarray(a, int), np.asarray(b, int)
        e += float(np.sum(T[np.arange(a.size), k[a], k[b]]))
    return e


def _table_parts(a, b, T, K):
    """For a chunk of tables: valid mask, the unary functions on a and b (theta(i, 0) - theta(0, 0),
    theta(0, j) - theta(0, 0)) and the second differences D (i, j >= 1; 0 outside)."""
    Ka, Kb = K[a][:, None, None], K[b][:, None, None]
    ii = np.arange(T.shape[1])[None, :, None]
    jj = np.arange(T.shape[2])[None, None, :]
    valid = (ii < Ka) & (jj < Kb)
    Tz = np.where(valid, T, 0.0)
    ua = Tz[:, :, 0] - Tz[:, :1, 0]
    ub = Tz[:, 0, :] - Tz[:, 0, :1]
    D = np.zeros_like(Tz)
    D[:, 1:, 1:] = Tz[:, 1:, 1:] - Tz[:, :-1, 1:] - Tz[:, 1:, :-1] + Tz[:, :-1, :-1]
    D = np.where(valid, D, 0.0)
    return valid, ua, ub, D


def _chunks(m, size):
    for c in range(0, m, size):
        yield slice(c, min(c + size, m))


def _graph(M, extra_unary=None, chunk=100000):
    """The s-t graph of unary + hard + hinge terms (ties ignored), as arcs with float capacities
    (inf for hard arcs). Node (i, k) is 'x_i >= k'; node (i, 0) is always on the source side.
    Vectorized over terms. Returns rows, cols, caps, start (node offsets), s, t."""
    n, K, o = M.n, np.asarray(M.K, int), np.asarray(M.o, int)
    start = np.r_[0, np.cumsum(K)]
    N = int(start[-1])
    s, t_ = N, N + 1
    U = np.concatenate([np.asarray(M.unary[i], float) for i in range(n)])
    if extra_unary:
        for i, v in extra_unary.items():
            U[start[i] : start[i + 1]] += v
    var = np.repeat(np.arange(n), K)
    lab = np.arange(N) - start[var]
    R, C, V = [], [], []

    def add(f, tt, c):
        f = np.asarray(f).ravel()
        R.append(f.astype(np.int64))
        C.append(np.broadcast_to(np.asarray(tt), f.shape).astype(np.int64).ravel())
        V.append(np.broadcast_to(np.asarray(c, float), f.shape).astype(float).ravel())

    a_h, b_h, lo_h, hi_h = _cat(M.hard, "hard")
    a_g, b_g, t_g, w_g = _cat(M.hinge, "hinge")
    # hinges: w max(0, x_b - x_a - t) and w max(0, x_a - x_b + t)
    for A_, B_, T_, W_ in ((a_g, b_g, t_g, w_g), (b_g, a_g, -t_g, w_g)):
        for sl in _chunks(A_.size, chunk):
            A, B, W = A_[sl], B_[sl], W_[sl]
            tl = (T_[sl] - (o[B] - o[A])).astype(int)[
                :, None
            ]  # label space: max(0, k_b - k_a - tl)
            Ka, Kb = K[A][:, None], K[B][:, None]
            kk = np.arange(int(Ka.max()))[None]
            mk = kk < Ka
            val = W[:, None] * np.maximum(0, -tl - kk)  # the j <= 0 part, on a
            idx = (start[A][:, None] + kk)[mk & (val > 0)]
            U += np.bincount(idx, (val)[mk & (val > 0)], minlength=N)
            j = np.arange(1, max(int(Kb.max()), 2))[None]
            mj = j < Kb
            ja = j - tl
            m = mj & (ja >= 1) & (ja <= Ka - 1)
            Wb = np.broadcast_to(W[:, None], m.shape)
            add((start[B][:, None] + j)[m], (start[A][:, None] + ja)[m], Wb[m])
            m2 = mj & (ja >= Ka)
            add((start[B][:, None] + j)[m2], t_, Wb[m2])
    # tables: theta = theta(0,0) + ua(k_a) + ub(k_b) + sum_{i<=k_a, j<=k_b} D(i, j); D <= 0 kept:
    # D y_ai y_bj = D y_ai + (-D) y_ai (1 - y_bj): a unary on a and an arc (a, i) -> (b, j)
    for a_, b_, T_ in M.table:
        a_, b_ = np.asarray(a_, int), np.asarray(b_, int)
        valid, ua, ub, D = _table_parts(a_, b_, np.asarray(T_, float), K)
        Dn = np.minimum(D, 0.0)
        ia = np.arange(T_.shape[1])[None]
        ma = ia < K[a_][:, None]
        cum = np.cumsum(Dn.sum(2), axis=1)  # sum_{i <= k} sum_j D(i, j)
        U += np.bincount((start[a_][:, None] + ia)[ma], (ua + cum)[ma], minlength=N)
        jb = np.arange(T_.shape[2])[None]
        mb = jb < K[b_][:, None]
        U += np.bincount((start[b_][:, None] + jb)[mb], ub[mb], minlength=N)
        p_, i_, j_ = np.nonzero(Dn < 0)
        add(start[a_][p_] + i_, start[b_][p_] + j_, -Dn[p_, i_, j_])
    # unary -> terminal arcs (telescoped)
    nd = np.flatnonzero(lab >= 1)
    d = U[nd] - U[nd - 1]
    add(np.full(int((d < 0).sum()), s), nd[d < 0], -d[d < 0])
    add(nd[d > 0], t_, d[d > 0])
    add(np.full(n, s), start[:-1], np.inf)
    add(nd, nd - 1, np.inf)
    # hard: x_b >= x_a + lo ; x_a >= x_b - hi
    for A_, B_, X_ in ((a_h, b_h, lo_h), (b_h, a_h, -hi_h)):
        for sl in _chunks(A_.size, chunk):
            A, B = A_[sl], B_[sl]
            xl = (X_[sl] - (o[B] - o[A])).astype(int)[:, None]
            Ka, Kb = K[A][:, None], K[B][:, None]
            k = np.arange(int(Ka.max()))[None]
            mk = k < Ka
            j = k + xl
            m = mk & (j >= 1) & (j <= Kb - 1)
            add((start[A][:, None] + k)[m], (start[B][:, None] + j)[m], np.inf)
            m2 = mk & (j >= Kb)
            add((start[A][:, None] + k)[m2], t_, np.inf)
    rows = np.concatenate(R)
    cols = np.concatenate(C)
    caps = np.concatenate(V)
    keep = caps > 0
    return rows[keep], cols[keep], caps[keep], start, s, t_


def cut(M, extra_unary=None):
    """Exact minimum of unary + hard + hinge terms (ties ignored). extra_unary: dict i -> (K_i,)
    added to variable i's unary. Returns labels k (n,). Raises if the hard bounds are infeasible."""
    rows, cols, caps, start, s, t_ = _graph(M, extra_unary)
    N = s
    inf = ~np.isfinite(caps)
    ci = np.zeros(caps.size, np.int64)
    ci[~inf] = np.round(caps[~inf] * SCALE).astype(np.int64)
    INF = int(ci.sum()) + 1
    ci[inf] = INF
    keep = ci > 0
    G = coo_matrix(
        (ci[keep], (rows[keep].astype(np.int32), cols[keep].astype(np.int32))),
        shape=(N + 2,) * 2,
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
    K = np.asarray(M.K, int)
    return np.add.reduceat(S[:N].astype(int), start[:-1]) - 1


def qpbo(M, start_labels=None, improve=True, log=None):
    """All terms, ties included, by QPBO (Kolmogorov and Rother 2007; thinqpbo, double precision)
    on the binary encoding y_(i,k) = [x_i >= k], k >= 1: the submodular s-t graph of _graph(), plus
    each tie w |k_a + k_b - L'| written as
        w |k_b - L'| + sum_j y_(a,j) (2 w [k_b >= L' - j + 1] - w),
    where [k_b >= m] is 1, 0 or y_(b,m): a non-submodular pairwise term (E11 = 2w).

    Returns a dict: labels (n,) after QPBO-I improvement of start_labels (default: cut() without the
    ties), certified (n,) bool, true where every y of the variable is weakly persistent and
    consistent (so the label agrees with some global minimum), the certified labels, the energy,
    and the lower bound (QPBO's twice lower bound / 2, constants of the encoding added)."""
    import thinqpbo

    say = log or (lambda m: None)
    n, K, o = M.n, np.asarray(M.K, int), np.asarray(M.o, int)
    a_t, b_t, L_t, w_t = _cat(M.tie, "tie")
    a_t, b_t = a_t.astype(int), b_t.astype(int)
    Lp = (L_t - o[a_t] - o[b_t]).astype(int)
    extra = {}
    for b, L, w in zip(b_t, Lp, w_t):  # the w |k_b - L'| part, on b
        extra[int(b)] = extra.get(int(b), 0) + w * np.abs(np.arange(K[b]) - L)
    rows, cols, caps, start, s, t_ = _graph(M, extra)
    say(f"graph: {rows.size:,} arcs")
    bstart = np.r_[0, np.cumsum(K - 1)]
    NB = int(bstart[-1])
    var = np.repeat(np.arange(n), K)
    lab = np.arange(int(start[-1])) - start[var]
    lab = np.r_[lab, -1, -1]  # s, t
    var = np.r_[var, 0, 0]

    def bid(nodes):
        return bstart[var[nodes]] + lab[nodes] - 1

    fin = np.isfinite(caps)
    big = BIG_FACTOR * (
        1.0 + float(caps[fin].sum()) + float(np.sum(w_t)) * float(K.max()) * 2
    )
    caps = np.where(fin, caps, big)
    ms = rows == s
    v = cols[ms]
    keep = lab[v] >= 1
    E0 = np.bincount(bid(v[keep]), caps[ms][keep], minlength=NB).astype(float)
    mt = cols == t_
    u = rows[mt]
    keep = lab[u] >= 1
    E1 = np.bincount(bid(u[keep]), caps[mt][keep], minlength=NB).astype(float)
    mo = ~ms & ~mt
    u, v, c = rows[mo], cols[mo], caps[mo]
    keep = lab[v] >= 1
    u, v, c = u[keep], v[keep], c[keep]
    from_zero = lab[u] == 0
    E0 += np.bincount(bid(v[from_zero]), c[from_zero], minlength=NB)
    pi = [bid(u[~from_zero])]
    pj = [bid(v[~from_zero])]
    p10 = [c[~from_zero]]
    p11 = [np.zeros(int((~from_zero).sum()))]
    del rows, cols, caps, u, v, c
    # ties: sum_j y_(a,j) (2 w [k_b >= m_j] - w), m_j = L' - j + 1
    if a_t.size:
        jj = np.arange(1, int(K[a_t].max()))[None]
        mj = jj < K[a_t][:, None]
        m = Lp[:, None] - jj + 1
        ya = bstart[a_t][:, None] + jj - 1
        W = np.broadcast_to(w_t[:, None], m.shape)
        lo_ = mj & (m <= 0)
        hi_ = mj & (m >= K[b_t][:, None])
        mid = mj & ~lo_ & ~hi_
        E1 += np.bincount(ya[lo_], W[lo_], minlength=NB)
        E1 -= np.bincount(ya[hi_ | mid], W[hi_ | mid], minlength=NB)
        yb = bstart[b_t][:, None] + m - 1
        pi.append(ya[mid])
        pj.append(yb[mid])
        p10.append(np.zeros(int(mid.sum())))
        p11.append(2.0 * W[mid])
    # tables: the positive second differences (not in the s-t graph), E11 = D
    for a_, b_, T_ in M.table:
        a_, b_ = np.asarray(a_, int), np.asarray(b_, int)
        valid, ua, ub, D = _table_parts(a_, b_, np.asarray(T_, float), K)
        p_, i_, j_ = np.nonzero(D > 0)
        pi.append(bstart[a_][p_] + i_ - 1)
        pj.append(bstart[b_][p_] + j_ - 1)
        p10.append(np.zeros(p_.size))
        p11.append(D[p_, i_, j_])
    pi, pj = np.concatenate(pi).astype(np.int64), np.concatenate(pj).astype(np.int64)
    p10, p11 = np.concatenate(p10), np.concatenate(p11)
    # merge parallel terms (same ordered pair) before handing them over
    key = pi * max(NB, 1) + pj
    uk, inv = np.unique(key, return_inverse=True)
    pi, pj = uk // max(NB, 1), uk % max(NB, 1)
    p10 = np.bincount(inv, p10, minlength=uk.size)
    p11 = np.bincount(inv, p11, minlength=uk.size)
    say(f"binary: {NB:,} nodes, {uk.size:,} pairwise terms")
    g = thinqpbo.QPBODouble(NB, int(uk.size))
    if NB:
        g.add_node(NB)
    nz = np.flatnonzero((E0 != 0) | (E1 != 0))
    if nz.size:
        np.vectorize(g.add_unary_term, otypes=[bool])(nz, E0[nz], E1[nz])
    if uk.size:
        z = np.zeros(uk.size)
        np.vectorize(g.add_pairwise_term, otypes=[int])(pi, pj, z, z, p10, p11)
    say("terms added")
    g.merge_parallel_edges()
    say("parallel edges merged")
    g.solve()
    say("solve done")
    g.compute_weak_persistencies()
    say("weak persistencies done")
    allb = np.arange(NB)
    yl = np.vectorize(g.get_label, otypes=[int])(allb) if NB else np.zeros(0, int)
    lower = 0.5 * g.compute_twice_lower_bound()
    say(f"labels read: {int(np.sum(yl < 0)):,} of {NB:,} binary nodes unlabeled")
    # per variable: certified if every y is labeled and the chain is consistent
    has = K > 1
    certified = np.ones(n, bool)
    cert_lab = np.zeros(n, int)
    if NB:
        bvar = np.repeat(np.arange(n), K - 1)
        unl = np.bincount(bvar, (yl < 0).astype(float), minlength=n) > 0
        same = bvar[1:] == bvar[:-1]
        bad_chain = np.zeros(n, bool)
        inc = same & (yl[1:] > yl[:-1])
        bad_chain[bvar[1:][inc]] = True
        certified = ~unl & ~bad_chain
        cert_lab = np.bincount(
            bvar, np.clip(yl, 0, 1).astype(float), minlength=n
        ).astype(int)
    cert_lab = np.where(certified, cert_lab, -1)
    # QPBO-I from the start labeling
    k0 = cut(M) if start_labels is None else np.asarray(start_labels, int)
    y0 = np.zeros(NB, int)
    if NB:
        y0 = (np.arange(NB) - bstart[bvar] + 1 <= k0[bvar]).astype(int)
    if improve and NB:
        np.vectorize(g.set_label, otypes=[object])(allb, y0)
        g.improve()
        say("improve done")
        yl2 = np.vectorize(g.get_label, otypes=[int])(allb)
        k = np.bincount(bvar, np.clip(yl2, 0, 1).astype(float), minlength=n).astype(int)
        chain_ok = not np.any(same & (yl2[1:] > yl2[:-1]))
    else:
        k, chain_ok = k0, True
    e = energy(M, k)
    e0 = energy(M, k0)
    if e0 < e:  # never worse than the start
        k, e = k0, e0
    # the encoding drops constants: relate the two scales at the (feasible) start labeling
    offset = e0 - _binary_energy(E0, E1, pi, pj, p10, p11, y0)
    return dict(
        labels=k,
        energy=e,
        certified=certified,
        certified_labels=cert_lab,
        lower_bound=lower + offset,
        chain_ok=chain_ok,
        binary_nodes=NB,
        pairwise=int(uk.size),
    )


def _binary_energy(E0, E1, pi, pj, p10, p11, y):
    y = np.asarray(y, int)
    e = float(np.sum(np.where(y == 1, E1, E0)))
    if len(pi):
        yi, yj = y[pi], y[pj]
        e += float(
            np.sum(p10 * ((yi == 1) & (yj == 0)) + p11 * ((yi == 1) & (yj == 1)))
        )
    return e


def reduce(M, fixed, labels, big=None):
    """Fix the variables where fixed (n,) is True at labels; return (the model of the others, the
    index map free -> original, the constant). Terms between a fixed and a free variable become
    unary costs on the free one (a hard bound: cost `big` outside it). For every labeling k of the
    free variables: energy(reduced, k) + constant == energy(M, k merged with the fixed labels),
    wherever the fixed variables' own hard bounds hold."""
    K, o = np.asarray(M.K, int), np.asarray(M.o, int)
    fixed = np.asarray(fixed, bool)
    labels = np.asarray(labels, int)
    free = np.flatnonzero(~fixed)
    new = -np.ones(M.n, int)
    new[free] = np.arange(free.size)
    x = o + np.where(fixed, labels, 0)
    U = [np.asarray(M.unary[i], float).copy() for i in free]
    const = float(sum(float(M.unary[i][labels[i]]) for i in np.flatnonzero(fixed)))
    if big is None:
        big = 1e3 * (
            1.0
            + sum(float(np.ptp(u)) for u in M.unary)
            + float(np.sum(_cat(M.hinge, "hinge")[3])) * max(K)
            + float(np.sum(_cat(M.tie, "tie")[3])) * max(K)
        )
    hard, hinge, tie = [], [], []
    a, b, lo, hi = _cat(M.hard, "hard")
    for aa, bb, l_, h_ in zip(a, b, lo, hi):
        if fixed[aa] and fixed[bb]:
            continue
        if not fixed[aa] and not fixed[bb]:
            hard.append(([new[aa]], [new[bb]], [l_], [h_]))
        elif fixed[aa]:  # lo <= x_b - x_a <= hi on free b
            xb = o[bb] + np.arange(K[bb])
            U[new[bb]] += np.where((xb - x[aa] < l_) | (xb - x[aa] > h_), big, 0.0)
        else:
            xa = o[aa] + np.arange(K[aa])
            U[new[aa]] += np.where((x[bb] - xa < l_) | (x[bb] - xa > h_), big, 0.0)
    a, b, t, w = _cat(M.hinge, "hinge")
    for aa, bb, tt, ww in zip(a, b, t, w):
        if fixed[aa] and fixed[bb]:
            const += float(ww * abs(x[bb] - x[aa] - tt))
        elif not fixed[aa] and not fixed[bb]:
            hinge.append(([new[aa]], [new[bb]], [tt], [ww]))
        elif fixed[aa]:
            U[new[bb]] += ww * np.abs(o[bb] + np.arange(K[bb]) - x[aa] - tt)
        else:
            U[new[aa]] += ww * np.abs(x[bb] - o[aa] - np.arange(K[aa]) - tt)
    a, b, L, w = _cat(M.tie, "tie")
    for aa, bb, LL, ww in zip(a, b, L, w):
        if fixed[aa] and fixed[bb]:
            const += float(ww * abs(x[aa] + x[bb] - LL))
        elif not fixed[aa] and not fixed[bb]:
            tie.append(([new[aa]], [new[bb]], [LL], [ww]))
        elif fixed[aa]:
            U[new[bb]] += ww * np.abs(x[aa] + o[bb] + np.arange(K[bb]) - LL)
        else:
            U[new[aa]] += ww * np.abs(o[aa] + np.arange(K[aa]) + x[bb] - LL)
    tab = []
    for a_, b_, T_ in M.table:
        for p_, (aa, bb) in enumerate(zip(np.asarray(a_, int), np.asarray(b_, int))):
            Tp = np.asarray(T_[p_], float)[: K[aa], : K[bb]]
            if fixed[aa] and fixed[bb]:
                const += float(Tp[labels[aa], labels[bb]])
            elif not fixed[aa] and not fixed[bb]:
                tab.append((new[aa], new[bb], Tp))
            elif fixed[aa]:
                U[new[bb]] += Tp[labels[aa], :]
            else:
                U[new[aa]] += Tp[:, labels[bb]]
    tables = []
    if tab:
        ka = max(t[2].shape[0] for t in tab)
        kb = max(t[2].shape[1] for t in tab)
        T = np.full((len(tab), ka, kb), np.nan)
        for p_, t in enumerate(tab):
            T[p_, : t[2].shape[0], : t[2].shape[1]] = t[2]
        tables = [(np.array([t[0] for t in tab]), np.array([t[1] for t in tab]), T)]
    pack = lambda terms: (
        [tuple(np.concatenate([np.asarray(t[j]) for t in terms]) for j in range(4))]
        if terms
        else []
    )  # noqa: E731
    return (
        Model(K[free], o[free], U, pack(hard), pack(hinge), pack(tie), tables),
        free,
        const,
    )


def qpbo_pruned(M, start_labels=None, log=None):
    """QPBO on M; then the certified variables fixed (reduce) and QPBO-I on the rest only.
    Returns qpbo()'s dict with labels/energy from the pruned completion, and n_free."""
    r = qpbo(M, improve=False, log=log)
    cert = r["certified"]
    if cert.all():
        k = r["certified_labels"].copy()
        r.update(labels=k, energy=energy(M, k), n_free=0)
        return r
    R, free, const = reduce(M, cert, np.where(cert, r["certified_labels"], 0))
    r2 = qpbo(
        R,
        start_labels=None if start_labels is None else np.asarray(start_labels)[free],
        log=log,
    )
    k = np.where(cert, r["certified_labels"], 0)
    k[free] = r2["labels"]
    e = energy(M, k)
    if (
        start_labels is not None and energy(M, start_labels) < e
    ):  # never worse than the start
        k, e = np.asarray(start_labels, int).copy(), energy(M, start_labels)
    r.update(labels=k, energy=e, n_free=int(free.size))
    return r
