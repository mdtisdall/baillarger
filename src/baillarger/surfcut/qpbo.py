"""Every term, ties and non-submodular tables included, by QPBO.

QPBO (roof duality; Kolmogorov and Rother 2007, "Minimizing nonsubmodular
functions with graph cuts - a review"; thinqpbo, double precision) on the
binary encoding y_(i,k) = [x_i >= k], k >= 1: the submodular graph of
baillarger.surfcut.cut, plus each tie w |k_a + k_b - L'| written as

    w |k_b - L'| + sum_j y_(a,j) (2 w [k_b >= L' - j + 1] - w),

where [k_b >= m] is 1, 0 or y_(b,m) (a pairwise term with E11 = 2 w, not
submodular), and each table's positive second differences as E11 terms.

It returns a partial labelling that agrees with some global minimum (weak
persistency) and a lower bound; QPBO-I (Rother et al. 2007, "Optimizing
binary MRFs via extended roof duality") then completes it from a start
labelling and never ends above that start. Where every binary node of a
variable is labelled, the variable is certified. Where only some are, the
labelled ones still bound it: y_(i,k) = 1 says x_i >= k, y_(i,k) = 0 says
x_i < k, and some global minimum lies in every variable's interval at once
(the depths QPBO could not rule out; zero width where certified). On a
submodular model every variable is certified and the labelling is the global
minimum.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from baillarger.surfcut._thinqpbo import new_graph, seed_rand
from baillarger.surfcut.cut import cut, graph
from baillarger.surfcut.model import (
    Bool,
    Float,
    Hard,
    Hinge,
    Int,
    Model,
    Table,
    Tie,
    energy,
    floats,
    hard_terms,
    hinge_terms,
    ints,
    table_parts,
    tie_terms,
)

HARD = 1e3  # a hard term's cost: this times the total of every finite cost


@dataclass
class Result:
    labels: Int  # after QPBO-I from the start labelling
    energy: float
    certified: (
        Bool  # every binary node labelled and consistent: agrees with some minimum
    )
    certified_labels: Int  # -1 where not certified
    lower_bound: float
    chain_ok: bool  # QPBO-I left every variable's binary nodes consistent
    binary_nodes: int
    pairwise: int
    interval_lo: Int  # lowest label the labelled binary nodes allow
    interval_hi: Int  # highest; lo == hi == the label where certified
    n_free: int = -1  # qpbo_pruned: variables left to QPBO-I


def _binary_energy(
    E0: Float, E1: Float, pi: Int, pj: Int, p10: Float, p11: Float, y: Int
) -> float:
    e = float(np.sum(np.where(y == 1, E1, E0)))
    if pi.size:
        yi, yj = y[pi], y[pj]
        e += float(
            np.sum(p10 * ((yi == 1) & (yj == 0)) + p11 * ((yi == 1) & (yj == 1)))
        )
    return e


def qpbo(
    M: Model,
    start_labels: npt.ArrayLike | None = None,
    improve: bool = True,
    log: Callable[[str], None] | None = None,
    seed: int = 0,
) -> Result:
    """Every term by QPBO, then QPBO-I from start_labels (default: cut(),
    the optimum without the non-submodular terms). QPBO-I visits the variables
    in an order drawn from C rand(); it is reseeded with `seed` first, so the
    result depends only on the inputs."""
    say = log or (lambda m: None)
    n, K, o = M.n, ints(M.K), ints(M.o)
    a_t, b_t, L_t, w_t = tie_terms(M)
    Lp = L_t - o[a_t] - o[b_t]
    extra: dict[int, Float] = {}
    for b, L, w in zip(b_t, Lp, w_t, strict=True):  # the w |k_b - L'| part, on b
        extra[int(b)] = extra.get(int(b), floats(np.zeros(K[b]))) + w * np.abs(
            np.arange(K[b]) - L
        )
    rows, cols, caps, start, s, t_ = graph(M, extra)
    say(f"graph: {rows.size:,} arcs")
    bstart = np.r_[0, np.cumsum(K - 1)].astype(np.int64)
    NB = int(bstart[-1])
    var = np.r_[np.repeat(np.arange(n), K), 0, 0]
    lab = np.r_[np.arange(int(start[-1])) - start[var[:-2]], -1, -1]  # s, t: -1

    def bid(nodes: Int) -> Int:
        return ints(bstart[var[nodes]] + lab[nodes] - 1)

    fin = np.isfinite(caps)
    big = HARD * (
        1.0 + float(caps[fin].sum()) + float(np.sum(w_t)) * float(K.max(initial=1)) * 2
    )
    caps = np.where(fin, caps, big)
    ms = rows == s
    v = cols[ms]
    keep = lab[v] >= 1
    E0 = np.bincount(bid(v[keep]), caps[ms][keep], minlength=NB).astype(np.float64)
    mt = cols == t_
    u = rows[mt]
    keep = lab[u] >= 1
    E1 = np.bincount(bid(u[keep]), caps[mt][keep], minlength=NB).astype(np.float64)
    mo = ~ms & ~mt
    u, v, c = rows[mo], cols[mo], caps[mo]
    keep = lab[v] >= 1
    u, v, c = u[keep], v[keep], c[keep]
    zero = lab[u] == 0  # from a node always on the source side: a unary on v
    E0 += np.bincount(bid(v[zero]), c[zero], minlength=NB)
    PI, PJ, P10, P11 = (
        [bid(u[~zero])],
        [bid(v[~zero])],
        [c[~zero]],
        [np.zeros(int((~zero).sum()))],
    )
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
        PI.append(ints(ya[mid]))
        PJ.append(ints(yb[mid]))
        P10.append(np.zeros(int(mid.sum())))
        P11.append(2.0 * W[mid])
    # tables: the positive second differences, E11 = D
    for ta, tb, T in M.table:
        ta, tb = ints(ta), ints(tb)
        _, _, _, D = table_parts(ta, tb, floats(T), K)
        p_, i_, j_ = np.nonzero(D > 0)
        PI.append(ints(bstart[ta][p_] + i_ - 1))
        PJ.append(ints(bstart[tb][p_] + j_ - 1))
        P10.append(np.zeros(p_.size))
        P11.append(D[p_, i_, j_])
    pi, pj = np.concatenate(PI), np.concatenate(PJ)
    # merge parallel terms (the same ordered pair)
    key = pi * max(NB, 1) + pj
    uk, inv = np.unique(key, return_inverse=True)
    pi, pj = ints(uk // max(NB, 1)), ints(uk % max(NB, 1))
    p10 = np.bincount(inv, np.concatenate(P10), minlength=uk.size).astype(np.float64)
    p11 = np.bincount(inv, np.concatenate(P11), minlength=uk.size).astype(np.float64)
    say(f"binary: {NB:,} nodes, {uk.size:,} pairwise terms")
    g = new_graph(NB, int(uk.size))
    if NB:
        g.add_node(NB)
    nz = np.flatnonzero((E0 != 0) | (E1 != 0))
    if nz.size:
        np.vectorize(g.add_unary_term, otypes=[object])(nz, E0[nz], E1[nz])
    if uk.size:
        z = np.zeros(uk.size)
        np.vectorize(g.add_pairwise_term, otypes=[int])(pi, pj, z, z, p10, p11)
    g.merge_parallel_edges()
    g.solve()
    g.compute_weak_persistencies()
    allb = np.arange(NB)
    yl = ints(np.vectorize(g.get_label, otypes=[np.int64])(allb)) if NB else ints([])
    lower = 0.5 * g.compute_twice_lower_bound()
    say(f"qpbo: {int(np.sum(yl < 0)):,} of {NB:,} binary nodes unlabelled")
    certified = np.ones(n, bool)
    cert_lab = np.zeros(n, np.int64)
    bvar = np.repeat(np.arange(n), K - 1)
    lo_k, hi_k = np.zeros(n, np.int64), ints(K - 1)
    same = bvar[1:] == bvar[:-1]
    if NB:
        unl = np.bincount(bvar, (yl < 0).astype(np.float64), minlength=n) > 0
        bad = np.zeros(n, bool)
        bad[bvar[1:][same & (yl[1:] > yl[:-1])]] = True
        certified = ~unl & ~bad
        cert_lab = np.bincount(
            bvar, np.clip(yl, 0, 1).astype(np.float64), minlength=n
        ).astype(np.int64)
        kb = np.arange(NB) - bstart[bvar] + 1  # y_(i,k): k
        np.maximum.at(lo_k, bvar[yl == 1], kb[yl == 1])
        np.minimum.at(hi_k, bvar[yl == 0], kb[yl == 0] - 1)
    cert_lab = ints(np.where(certified, cert_lab, -1))
    k0 = cut(M) if start_labels is None else ints(start_labels)
    y0 = (
        ints((np.arange(NB) - bstart[bvar] + 1 <= k0[bvar]).astype(np.int64))
        if NB
        else ints([])
    )
    k, chain_ok = k0, True
    if improve and NB:
        np.vectorize(g.set_label, otypes=[object])(allb, y0)
        seed_rand(seed)
        g.improve()
        yl2 = np.vectorize(g.get_label, otypes=[np.int64])(allb)
        k = ints(np.bincount(bvar, np.clip(yl2, 0, 1).astype(np.float64), minlength=n))
        chain_ok = not bool(np.any(same & (yl2[1:] > yl2[:-1])))
    e, e0 = energy(M, k), energy(M, k0)
    if e0 < e:  # never above the start
        k, e = k0, e0
    # the encoding drops constants: relate the two scales at the start labelling
    offset = e0 - _binary_energy(E0, E1, pi, pj, p10, p11, y0)
    return Result(
        k,
        e,
        certified,
        cert_lab,
        lower + offset,
        chain_ok,
        NB,
        int(uk.size),
        lo_k,
        hi_k,
    )


def reduce(
    M: Model, fixed: npt.ArrayLike, labels: npt.ArrayLike, big: float | None = None
) -> tuple[Model, Int, float]:
    """Fix the variables where fixed (n,) is true at labels. Returns the model
    of the others, the map from its variables to M's, and a constant: terms
    between a fixed and a free variable become unary costs on the free one (a
    hard bound: cost `big` outside it). For every labelling k of the free
    variables, energy(reduced, k) + constant == energy(M, k with the fixed
    labels), wherever the bounds between fixed variables hold."""
    K, o = ints(M.K), ints(M.o)
    fx = np.asarray(fixed, bool)
    lab = ints(labels)
    free = ints(np.flatnonzero(~fx))
    new = -np.ones(M.n, np.int64)
    new[free] = np.arange(free.size)
    x = o + np.where(fx, lab, 0)
    U = [floats(M.unary[i]).copy() for i in free]
    const = float(sum(float(M.unary[i][lab[i]]) for i in np.flatnonzero(fx)))
    if big is None:
        big = HARD * (
            1.0
            + sum(float(np.ptp(u)) for u in M.unary)
            + float(np.sum(hinge_terms(M)[3])) * float(K.max(initial=1))
            + float(np.sum(tie_terms(M)[3])) * float(K.max(initial=1))
        )
    hard: list[Hard] = []
    hinge: list[Hinge] = []
    tie: list[Tie] = []
    for aa, bb, lo, hi in zip(*hard_terms(M), strict=True):
        if fx[aa] and fx[bb]:
            continue
        if not fx[aa] and not fx[bb]:
            hard.append((ints(new[aa]), ints(new[bb]), ints(lo), ints(hi)))
        elif fx[aa]:  # lo <= x_b - x_a <= hi, on free b
            xb = o[bb] + np.arange(K[bb])
            U[new[bb]] += np.where((xb - x[aa] < lo) | (xb - x[aa] > hi), big, 0.0)
        else:
            xa = o[aa] + np.arange(K[aa])
            U[new[aa]] += np.where((x[bb] - xa < lo) | (x[bb] - xa > hi), big, 0.0)
    for aa, bb, tt, ww in zip(*hinge_terms(M), strict=True):
        if fx[aa] and fx[bb]:
            const += float(ww * abs(x[bb] - x[aa] - tt))
        elif not fx[aa] and not fx[bb]:
            hinge.append((ints(new[aa]), ints(new[bb]), ints(tt), floats(ww)))
        elif fx[aa]:
            U[new[bb]] += ww * np.abs(o[bb] + np.arange(K[bb]) - x[aa] - tt)
        else:
            U[new[aa]] += ww * np.abs(x[bb] - o[aa] - np.arange(K[aa]) - tt)
    for aa, bb, LL, ww in zip(*tie_terms(M), strict=True):
        if fx[aa] and fx[bb]:
            const += float(ww * abs(x[aa] + x[bb] - LL))
        elif not fx[aa] and not fx[bb]:
            tie.append((ints(new[aa]), ints(new[bb]), ints(LL), floats(ww)))
        elif fx[aa]:
            U[new[bb]] += ww * np.abs(x[aa] + o[bb] + np.arange(K[bb]) - LL)
        else:
            U[new[aa]] += ww * np.abs(o[aa] + np.arange(K[aa]) + x[bb] - LL)
    tabs: list[tuple[int, int, Float]] = []
    for ta, tb, T in M.table:
        for p_, (aa, bb) in enumerate(zip(ints(ta), ints(tb), strict=True)):
            Tp = floats(T[p_])[: K[aa], : K[bb]]
            if fx[aa] and fx[bb]:
                const += float(Tp[lab[aa], lab[bb]])
            elif not fx[aa] and not fx[bb]:
                tabs.append((int(new[aa]), int(new[bb]), Tp))
            elif fx[aa]:
                U[new[bb]] += Tp[lab[aa], :]
            else:
                U[new[aa]] += Tp[:, lab[bb]]
    tables: list[Table] = []
    if tabs:
        ka = max(t[2].shape[0] for t in tabs)
        kb = max(t[2].shape[1] for t in tabs)
        TT = np.full((len(tabs), ka, kb), np.nan)
        for p_, t in enumerate(tabs):
            TT[p_, : t[2].shape[0], : t[2].shape[1]] = t[2]
        tables = [(ints([t[0] for t in tabs]), ints([t[1] for t in tabs]), TT)]
    return Model(K[free], o[free], U, hard, hinge, tie, tables), free, const


def qpbo_pruned(
    M: Model,
    start_labels: npt.ArrayLike | None = None,
    log: Callable[[str], None] | None = None,
    seed: int = 0,
) -> Result:
    """QPBO on M; then the certified variables fixed (reduce) and QPBO-I on
    the rest only, from start_labels (never ending above them), with `seed` as
    in qpbo."""
    r = qpbo(M, improve=False, log=log)
    if r.certified.all():
        k = r.certified_labels.copy()
        r.labels, r.energy, r.n_free = k, energy(M, k), 0
        return r
    R, free, _ = reduce(M, r.certified, np.where(r.certified, r.certified_labels, 0))
    r2 = qpbo(
        R,
        start_labels=None if start_labels is None else ints(start_labels)[free],
        log=log,
        seed=seed,
    )
    k = ints(np.where(r.certified, r.certified_labels, 0))
    k[free] = r2.labels
    e = energy(M, k)
    if start_labels is not None and energy(M, start_labels) < e:
        k, e = ints(start_labels).copy(), energy(M, start_labels)
    r.labels, r.energy, r.n_free = k, e, int(free.size)
    return r
