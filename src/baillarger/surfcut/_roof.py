"""QPBO (roof duality) and QPBO-I on energies of binary variables.

Written from Kolmogorov and Rother (2007), "Minimizing nonsubmodular functions
with graph cuts - a review" (the doubled graph, its labelling rule, weak
persistency) and Rother et al. (2007), "Optimizing binary MRFs via extended
roof duality" (QPBO-I). Only numpy and scipy (both BSD) are used.

The energy is

    E(y) = sum_p U0_p [y_p = 0] + U1_p [y_p = 1]
         + sum_e A_e [00] + B_e [01] + C_e [10] + D_e [11]   on (y_i, y_j),

with inf for a forbidden configuration. Each pairwise table is put in normal
form (every row and every column has a zero; the rest goes to the unary
terms), and then each cost becomes two arcs on a graph with two nodes for
each variable, p (y_p) and its mirror p' (1 - y_p). The source side of a cut
is label 0. A term on (y_i, y_j) puts arcs between i and j and between j' and
i' where it is submodular (B, C), and between i and j' and between j and i'
where it is not (A, D). Each arc carries the whole cost, so the cut of a
labelling and its mirror is twice its energy, and the minimum cut is twice
the roof-duality lower bound.

From a minimum cut (S, T): y_p = 0 where p is in S and p' in T, y_p = 1 where
p is in T and p' in S, and no label elsewhere. Strong persistency: the labels
of the smallest source set (reachable from s in the residual graph); they
agree with every minimum of E. Weak persistency: the labels of a larger
minimum cut, chosen with the strongly connected components of the residual
graph; they agree with some minimum of E.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_array, csr_array
from scipy.sparse.csgraph import connected_components

from baillarger.surfcut._flow import SCALE, min_cut
from baillarger.surfcut.model import Bool, Float, Int, floats, ints


@dataclass
class Binary:
    """An energy of binary variables (see the module docstring)."""

    U0: Float
    U1: Float
    i: Int
    j: Int
    A: Float
    B: Float
    C: Float
    D: Float

    @property
    def n(self) -> int:
        return int(self.U0.size)


def binary_energy(P: Binary, y: npt.ArrayLike) -> float:
    """E(y); inf if y takes a forbidden configuration."""
    y = ints(y) if P.n else ints([])
    e = float(np.sum(np.where(y == 1, P.U1, P.U0)))
    if P.i.size:
        yi, yj = y[P.i], y[P.j]
        e += float(
            np.sum(
                np.where(
                    yi == 0, np.where(yj == 0, P.A, P.B), np.where(yj == 0, P.C, P.D)
                )
            )
        )
    return e


def _less(x: Float, m: Float) -> Float:
    """x - m, with inf - inf = 0 (a row or column that is all inf moves to the
    unary term whole)."""
    with np.errstate(invalid="ignore"):
        return np.where(np.isinf(m), 0.0, x - m)


def _normal_form(P: Binary) -> tuple[Binary, float]:
    """The same energy with the pairwise tables of each unordered pair merged
    and put in normal form, and the unary terms with min 0. Returns it and the
    constant taken out."""
    n = P.n
    U0, U1 = floats(P.U0).copy(), floats(P.U1).copy()
    i, j = ints(P.i), ints(P.j)
    A, B, C, D = floats(P.A), floats(P.B), floats(P.C), floats(P.D)
    for x in (U0, U1, A, B, C, D):
        if np.any(np.isnan(x)) or np.any(x == -np.inf):
            raise ValueError("costs must be finite or +inf")
    sw = i > j  # (j, i): B and C trade places
    i, j = np.where(sw, j, i), np.where(sw, i, j)
    B, C = np.where(sw, C, B), np.where(sw, B, C)
    self_ = i == j  # a term on one variable: its diagonal is a unary term
    U0 += np.bincount(i[self_], A[self_], minlength=n)
    U1 += np.bincount(i[self_], D[self_], minlength=n)
    i, j, A, B, C, D = (x[~self_] for x in (i, j, A, B, C, D))
    key = i * max(n, 1) + j
    uk, inv = np.unique(key, return_inverse=True)
    i, j = ints(uk // max(n, 1)), ints(uk % max(n, 1))
    A, B, C, D = (np.bincount(inv, x, minlength=uk.size) for x in (A, B, C, D))
    for row, (x, y) in enumerate(((A, B), (C, D))):  # rows: y_i = 0, 1
        m = np.minimum(x, y)
        (U0 if row == 0 else U1)[:] += np.bincount(i, m, minlength=n)
        x[:], y[:] = _less(x, m), _less(y, m)
    for col, (x, y) in enumerate(((A, C), (B, D))):  # columns: y_j = 0, 1
        m = np.minimum(x, y)
        (U0 if col == 0 else U1)[:] += np.bincount(j, m, minlength=n)
        x[:], y[:] = _less(x, m), _less(y, m)
    m = np.minimum(U0, U1)
    if np.any(np.isinf(m)):
        raise ValueError("a variable has no allowed label")
    return Binary(U0 - m, U1 - m, i, j, A, B, C, D), float(m.sum())


def _sink_first_rank(n_comp: int, cu: Int, cv: Int, labels: Int) -> Int:
    """A rank of the components such that every arc cu -> cv between two
    components goes from a higher rank to a lower one (a reverse topological
    order). scipy labels components in this order (as Tarjan's algorithm
    finishes them); that is checked, and if it does not hold the rank is
    computed here (Kahn's algorithm from the sinks)."""
    if not np.any(labels[cu] < labels[cv]):
        return labels
    uk = np.unique(cu * n_comp + cv)
    a, b = uk // n_comp, uk % n_comp
    out = np.bincount(a, minlength=n_comp)
    order = np.argsort(b, kind="stable")  # arcs grouped by their head
    bstart = np.searchsorted(b[order], np.arange(n_comp + 1))
    rank = -np.ones(n_comp, np.int64)
    nxt = 0
    ready = np.flatnonzero(out == 0)
    while ready.size:
        rank[ready] = nxt + np.arange(ready.size)
        nxt += ready.size
        heads = np.concatenate(
            [order[bstart[c] : bstart[c + 1]] for c in ready] or [ints([])]
        )
        pred = a[heads]
        np.subtract.at(out, pred, 1)
        ready = np.unique(pred[out[pred] == 0])
    return ints(rank)


@dataclass
class Roof:
    weak: Int  # -1 where unlabelled; agrees with some minimum
    strong: Int  # -1 where unlabelled; agrees with every minimum
    lower_bound: float
    log: str


def roof(P: Binary, scale: float = SCALE) -> Roof:
    """QPBO: the roof-duality lower bound and the persistent labels of E. The
    costs are rounded down to 1 / scale (baillarger.surfcut._flow), so the
    bound is valid for the true costs. Raises ValueError if every labelling is
    forbidden or the hard cost reaches 2^61."""
    n = P.n
    if n == 0:
        e = ints([])
        return Roof(e, e, 0.0, "roof: no variables")
    Q, const = _normal_form(P)
    s, t = 2 * n, 2 * n + 1
    p = np.arange(n)
    pb = p + n  # the mirror nodes
    i, j, ib, jb = Q.i, Q.j, Q.i + n, Q.j + n
    S_, T_ = np.full(n, s), np.full(n, t)
    R: list[Int] = []
    C_: list[Int] = []
    V: list[Float] = []
    for fr, to, c in (
        (S_, p, Q.U1),  # U1 y_p: cut when p is on the sink side
        (pb, T_, Q.U1),
        (p, T_, Q.U0),
        (S_, pb, Q.U0),
        (i, j, Q.B),  # B (1 - y_i) y_j
        (jb, ib, Q.B),
        (j, i, Q.C),  # C y_i (1 - y_j)
        (ib, jb, Q.C),
        (jb, i, Q.D),  # D y_i y_j
        (ib, j, Q.D),
        (i, jb, Q.A),  # A (1 - y_i)(1 - y_j)
        (j, ib, Q.A),
    ):
        m = c > 0
        R.append(ints(fr[m]))
        C_.append(ints(to[m]))
        V.append(floats(c[m]))
    mc = min_cut(
        np.concatenate(R), np.concatenate(C_), np.concatenate(V), 2 * n + 2, s, t, scale
    )
    if mc.value >= mc.capacities.hard:
        raise ValueError("every labelling takes a forbidden configuration")
    src, snk = mc.source_side, mc.sink_side
    strong = -np.ones(n, np.int64)
    strong[src[p]] = 0
    strong[snk[p]] = 1
    # a larger minimum cut: the residual graph's components between the two
    # smallest sides, taken into the source side in reverse topological order
    # wherever the mirror component is not taken first (as in 2-SAT); a component
    # that holds its own mirror goes to the source side and labels nothing
    S = src.copy()
    mid = np.flatnonzero(~src & ~snk)
    if mid.size:
        idx = -np.ones(2 * n + 2, np.int64)
        idx[mid] = np.arange(mid.size)
        Rm = csr_array(mc.residual[mid][:, mid])
        nc, lab = connected_components(Rm, directed=True, connection="strong")
        Rc = coo_array(Rm)
        cu, cv = ints(lab[Rc.row]), ints(lab[Rc.col])
        d = cu != cv
        rank = _sink_first_rank(nc, cu[d], cv[d], ints(lab))
        mirror = np.where(mid < n, mid + n, mid - n)
        if np.any(idx[mirror] < 0):
            raise AssertionError("the residual graph is not symmetric")
        cm = lab[idx[mirror]]
        own = rank[lab] <= rank[cm]  # equal: the component holds its own mirror
        S[mid[own]] = True
    weak = -np.ones(n, np.int64)
    weak[S[p] & ~S[pb]] = 0
    weak[~S[p] & S[pb]] = 1
    lb = const + mc.value / (2.0 * scale)
    log = (
        f"roof: {n:,} variables, {Q.i.size:,} pairs; scale {scale:g}, hard cost "
        f"{mc.capacities.hard:,}, largest rounding error "
        f"{mc.capacities.max_error:.3g}, total {mc.capacities.total_error:.3g}; "
        f"{int(np.sum(weak < 0)):,} unlabelled"
    )
    return Roof(ints(weak), ints(strong), lb, log)


def restrict(P: Binary, free: Bool, y: Int) -> tuple[Binary, Int]:
    """The energy of the free variables with the others fixed at y (their own
    terms dropped: a constant). Returns it and its variables' indices in P."""
    idx = ints(np.flatnonzero(free))
    new = -np.ones(P.n, np.int64)
    new[idx] = np.arange(idx.size)
    U0, U1 = floats(P.U0[idx]).copy(), floats(P.U1[idx]).copy()
    fi, fj = free[P.i], free[P.j]
    m = fi & ~fj  # y_j fixed: a unary term on i
    yj = y[P.j[m]]
    U0 += np.bincount(
        new[P.i[m]], np.where(yj == 0, P.A[m], P.B[m]), minlength=idx.size
    )
    U1 += np.bincount(
        new[P.i[m]], np.where(yj == 0, P.C[m], P.D[m]), minlength=idx.size
    )
    m = ~fi & fj
    yi = y[P.i[m]]
    U0 += np.bincount(
        new[P.j[m]], np.where(yi == 0, P.A[m], P.C[m]), minlength=idx.size
    )
    U1 += np.bincount(
        new[P.j[m]], np.where(yi == 0, P.B[m], P.D[m]), minlength=idx.size
    )
    m = fi & fj
    Q = Binary(
        U0, U1, ints(new[P.i[m]]), ints(new[P.j[m]]), P.A[m], P.B[m], P.C[m], P.D[m]
    )
    return Q, idx


def improve(
    P: Binary,
    start: Int,
    labels: Int,
    rng: np.random.Generator,
    passes: int = 4,
    scale: float = SCALE,
    log: Callable[[str], None] | None = None,
) -> Int:
    """QPBO-I (Rother et al. 2007): a full labelling, from `start` with QPBO's
    labels (`labels`, -1 where none) put in, never above E(start).

    The unlabelled variables U are solved with the rest fixed. In each pass, in
    an order drawn from rng: fix one variable of each connected part of the
    undecided variables at its current label, run QPBO on the other undecided
    ones, and take the labels it gives (they agree with some minimum of the
    energy with the fixed ones held, so the energy does not increase); those
    variables are decided. Repeat until every variable of U is decided. The
    parts are independent, so fixing one variable in each at the same time is
    the same as fixing them one at a time. A step whose labels would increase
    the energy (only by rounding) is not taken. Stop after `passes` passes, or
    after a pass that gives no decrease."""
    say = log or (lambda m: None)
    y = ints(start).copy()
    have = labels >= 0
    cand = y.copy()
    cand[have] = labels[have]
    if binary_energy(P, cand) <= binary_energy(P, y):
        y = cand
    else:
        have = np.zeros(P.n, bool)
    Q, U = restrict(P, ~have, y)
    if Q.n == 0:
        return y
    yq = y[U].copy()
    e = binary_energy(Q, yq)
    for pss in range(passes):
        e_pass = e
        pos = np.empty(Q.n, np.int64)
        pos[rng.permutation(Q.n)] = np.arange(Q.n)
        undecided = np.ones(Q.n, bool)
        steps = 0
        while undecided.any():
            both = undecided[Q.i] & undecided[Q.j]
            G = coo_array(
                (np.ones(int(both.sum()), np.int8), (Q.i[both], Q.j[both])),
                shape=(Q.n, Q.n),
            )
            _, comp = connected_components(G, directed=False)
            first = np.full(Q.n, np.iinfo(np.int64).max)
            np.minimum.at(first, comp[undecided], pos[undecided])
            undecided[undecided & (pos == first[comp])] = False  # fixed
            steps += 1
            if not undecided.any():
                break
            sub, idx = restrict(Q, undecided, yq)
            r = roof(sub, scale)
            lab = r.weak >= 0
            cand = yq.copy()
            cand[idx[lab]] = r.weak[lab]
            ec = binary_energy(Q, cand)
            if ec <= e:
                yq, e = cand, ec
            undecided[idx[lab]] = False
        say(
            f"qpbo-i pass {pss + 1}: {Q.n:,} variables, {steps} steps, "
            f"energy of these variables {e:.6g}"
        )
        if e >= e_pass:
            break
    y[U] = yq
    return y
