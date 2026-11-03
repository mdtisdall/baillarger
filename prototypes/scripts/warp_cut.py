"""The joint warp's levels (plan: reports/Crossing paths and contacts from laminar profiles.md,
*The joint warp*). Pure functions, no argv. The solvers are thin wrappers that call
baillarger.surfcut (our own QPBO, on scipy's maximum flow); this module keeps the prototype's
Model dataclass and the dict return formats.

Variables sit on one depth lattice of step h: variable i takes label k in [0, K_i), at depth
h (o_i + k) with its own integer origin o_i. Terms, on absolute positions x_i = o_i + k_i:
- unary costs (per variable, per label);
- hard bounds lo <= x_b - x_a <= hi;
- L1 hinges w |x_b - x_a - t|, w >= 0;
- ties w |x_a + x_b - L| (a sum: not submodular on a loop);
- pairwise tables theta(k_a, k_b): submodular where every second difference
  D(i, j) = theta(i, j) - theta(i-1, j) - theta(i, j-1) + theta(i-1, j-1) <= 0.

- energy: the total, ties included.
- cut: baillarger.surfcut.cut, the exact minimum of unary + hard + hinge terms and the submodular
  part of the tables (positive second differences dropped), ties ignored. QPBO-I's start.
- qpbo, qpbo_pruned, reduce: baillarger.surfcut.qpbo, qpbo_pruned, reduce.
"""

from dataclasses import dataclass, field

import numpy as np

import baillarger.surfcut as SC

@dataclass
class Model:
    K: np.ndarray  # (n,) labels per variable
    o: np.ndarray  # (n,) integer origins
    unary: list  # per variable: (K_i,) costs
    hard: list = field(default_factory=list)  # (a, b, lo, hi) arrays
    hinge: list = field(default_factory=list)  # (a, b, t, w) arrays
    tie: list = field(default_factory=list)  # (a, b, L, w) arrays
    table: list = field(default_factory=list)  # (a, b, T): T (m, Ka_max, Kb_max), T[p, k_a, k_b]; nan padding

    @property
    def n(self):
        return len(self.K)


def _cat(terms, k):
    if not terms:
        return [np.zeros(0, int if j < 2 or k == "hard" else float) for j in range(4)]
    return [np.concatenate([np.atleast_1d(np.asarray(t[j])) for t in terms]) for j in range(4)]


def energy(M, k):
    """Total energy of labels k (n,), ties included; inf if a hard bound is broken."""
    k = np.asarray(k, int)
    x = np.asarray(M.o, int) + k
    start = np.r_[0, np.cumsum(np.asarray(M.K, int))]
    e = float(np.concatenate([np.asarray(u, float) for u in M.unary])[start[:-1] + k].sum()) if M.n else 0.0
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


def _to_sc(M):
    """The prototype Model as a baillarger.surfcut.Model (int64 labels and indices, float64 costs)."""
    i64 = lambda x: np.atleast_1d(np.asarray(x, np.int64))  # noqa: E731
    f64 = lambda x: np.atleast_1d(np.asarray(x, np.float64))  # noqa: E731
    return SC.Model(
        i64(M.K),
        i64(M.o),
        [f64(u) for u in M.unary],
        [(i64(a), i64(b), i64(lo), i64(hi)) for a, b, lo, hi in M.hard],
        [(i64(a), i64(b), i64(t), f64(w)) for a, b, t, w in M.hinge],
        [(i64(a), i64(b), i64(L), f64(w)) for a, b, L, w in M.tie],
        [(i64(a), i64(b), f64(T)) for a, b, T in M.table],
    )


def _from_sc(S):
    return Model(S.K, S.o, S.unary, S.hard, S.hinge, S.tie, S.table)


def _result(r):
    return dict(
        labels=r.labels,
        energy=r.energy,
        certified=r.certified,
        certified_labels=r.certified_labels,
        lower_bound=r.lower_bound,
        chain_ok=r.chain_ok,
        binary_nodes=r.binary_nodes,
        pairwise=r.pairwise,
        interval_lo=r.interval_lo,
        interval_hi=r.interval_hi,
        **({"n_free": r.n_free} if r.n_free >= 0 else {}),
    )


def cut(M, extra_unary=None):
    """baillarger.surfcut.cut: exact minimum of unary + hard + hinge terms (ties ignored).
    extra_unary: dict i -> (K_i,) added to variable i's unary. Returns labels k (n,). Raises if the
    hard bounds are infeasible."""
    extra = None if extra_unary is None else {int(i): np.asarray(v, np.float64) for i, v in extra_unary.items()}
    return SC.cut(_to_sc(M), extra)


def qpbo(M, start_labels=None, improve=True, log=None, seed=0):
    """All terms, ties included, by baillarger.surfcut.qpbo (our own QPBO, Kolmogorov and Rother
    2007, on scipy's maximum flow), then QPBO-I from start_labels (default: cut() without the ties);
    its order is drawn from numpy.random.Generator(PCG64(seed)).

    Returns a dict: labels (n,) after QPBO-I improvement of start_labels, certified (n,) bool, true
    where every y of the variable is weakly persistent and consistent (so the label agrees with some
    global minimum), the certified labels (-1 elsewhere), the energy, the lower bound, chain_ok,
    binary_nodes, pairwise, and the intervals interval_lo, interval_hi."""
    return _result(SC.qpbo(_to_sc(M), start_labels, improve, log, seed))


def reduce(M, fixed, labels, big=None):
    """baillarger.surfcut.reduce: fix the variables where fixed (n,) is True at labels; return (the
    model of the others, the index map free -> original, the constant). Terms between a fixed and a
    free variable become unary costs on the free one (a hard bound: cost `big` outside it). For
    every labeling k of the free variables: energy(reduced, k) + constant == energy(M, k merged with
    the fixed labels), wherever the fixed variables' own hard bounds hold."""
    R, free, const = SC.reduce(_to_sc(M), fixed, labels, big)
    return _from_sc(R), free, const


def qpbo_pruned(M, start_labels=None, log=None, seed=0):
    """baillarger.surfcut.qpbo_pruned: QPBO on M; then the certified variables fixed (reduce) and
    QPBO-I on the rest only. Returns qpbo()'s dict with labels/energy from the pruned completion,
    and n_free."""
    return _result(SC.qpbo_pruned(_to_sc(M), start_labels, log, seed))
