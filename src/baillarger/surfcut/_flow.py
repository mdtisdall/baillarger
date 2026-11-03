"""Integer capacities, the maximum flow, and the minimum cut.

Costs are floats; the maximum flow runs on int64 capacities. Each finite cost
is multiplied by one scale factor and rounded DOWN. A flow of the rounded
graph is then a flow of the true graph, so its value is a lower bound of the
true minimum cut (weak duality): a bound computed from it is valid for the
true costs, not only for the rounded ones. Rounding to the nearest integer
does not have this property (a bound above the true minimum is possible), and
int32 with costs rounded to 1/1000 is too coarse for real models: on the V1V2
box the rounding errors of the active costs summed to +1.9.

A hard arc (an infinite cost) gets 1 + the sum of the finite capacities, so
that no finite cut reaches it; that hard cost must stay below 2^61, so that
the sum of two of them (the largest residual capacity) fits in int64. Hard
arcs from s or into t are contracted first: every node that s reaches by hard
arcs is put into s, and every node that reaches t by hard arcs into t. So no
hard arc leaves s or enters t, and the flow out of s is at most the finite
total.

The maximum flow itself is behind one function, max_flow, so that another
int64 engine (our own, blocked or on a GPU) can replace it. The outputs used
here (the smallest source and sink sides, and the residual graph's strongly
connected components) are the same for every maximum flow (Picard and
Queyranne 1980), so they do not depend on the engine.

The engine now is OR-Tools' SimpleMaxFlow (push-relabel, Apache-2.0). It
fails to import if the HiGHS of the highspy package is loaded first in the
same process (a symbol clash; scipy's own HiGHS does not clash): do not add
highspy or cvxpy to this environment (tests/reference has its own).
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_array
from scipy.sparse.csgraph import breadth_first_order

from baillarger.surfcut.model import Bool, Float, Int, floats, ints

SCALE = 1e9  # costs to integers: one unit is 1 / SCALE
LIMIT = 2**61  # the hard cost must be below this


@dataclass
class Capacities:
    caps: npt.NDArray[np.int64]
    hard: int  # the capacity of a hard arc: 1 + the sum of the finite capacities
    scale: float
    max_error: float  # the largest rounding error of one arc, in cost units
    total_error: float  # the sum of the rounding errors of all arcs, in cost units


def capacities(caps: npt.ArrayLike, scale: float = SCALE) -> Capacities:
    """Float capacities (>= 0; inf for a hard arc) to int64: each finite one
    times `scale`, rounded down; each hard one 1 + the sum of the finite ones.
    Raises ValueError if that hard cost is 2^61 or more."""
    c = floats(caps)
    if np.any(np.isnan(c)) or np.any(c < 0):
        raise ValueError("capacities must be >= 0 (inf for a hard arc)")
    fin = np.isfinite(c)
    x = c[fin] * scale
    r = np.floor(x)
    if float(r.sum()) >= 2.0**62:  # before the cast to int64
        total = int(float(r.sum()))
    else:
        total = int(r.astype(np.int64).sum())
    if total + 1 >= LIMIT:
        raise ValueError(
            f"the hard cost 1 + {total} (the sum of the finite capacities at "
            f"scale {scale:g}) is 2^61 or more"
        )
    hard = total + 1
    out = np.full(c.size, hard, np.int64)
    out[fin] = r.astype(np.int64)
    err = x - r
    return Capacities(
        out,
        hard,
        scale,
        float(np.max(err, initial=0.0)) / scale,
        float(err.sum()) / scale,
    )


def max_flow(tails: Int, heads: Int, caps: Int, n_nodes: int, s: int, t: int) -> Int:
    """A maximum flow from s to t on arcs tails -> heads with int64 capacities
    (no parallel arcs, no self-loops): the flow on each arc."""
    from ortools.graph.python import max_flow as ortools_max_flow

    m = ortools_max_flow.SimpleMaxFlow()
    if tails.size:
        m.add_arcs_with_capacity(
            tails.astype(np.int32), heads.astype(np.int32), caps.astype(np.int64)
        )
    m.add_arc_with_capacity(s, t, 0)  # every node index up to t exists
    status = m.solve(s, t)
    if status != m.OPTIMAL:
        raise RuntimeError(f"maximum flow: status {status}")
    return ints(m.flows(np.arange(tails.size)))


def _reach(arcs: csr_array, start: int) -> Bool:
    seen = np.zeros(arcs.shape[0], bool)
    seen[breadth_first_order(arcs, start, directed=True, return_predecessors=False)] = (
        True
    )
    return seen


@dataclass
class MinCut:
    source_side: Bool  # reachable from s in the residual graph: the smallest source set
    sink_side: Bool  # reaches t in the residual graph: the smallest sink set
    residual: csr_array  # the arcs with residual capacity > 0 (1 each)
    value: int  # the capacity of the minimum cut, in integer units
    capacities: Capacities


def min_cut(
    rows: Int,
    cols: Int,
    caps: Float,
    n_nodes: int,
    s: int,
    t: int,
    scale: float = SCALE,
) -> MinCut:
    """The minimum s-t cut of a graph of arcs rows -> cols with float
    capacities (inf: hard). Parallel arcs are merged. value >= capacities.hard
    means that every cut crosses a hard arc."""
    cap = capacities(caps, scale)
    rows, cols = ints(rows), ints(cols)
    keep = (cap.caps > 0) & (rows != cols)
    key = rows[keep] * n_nodes + cols[keep]
    uk, inv = np.unique(key, return_inverse=True)  # sorted: the same graph every run
    w = np.zeros(uk.size, np.int64)
    np.add.at(w, inv, cap.caps[keep])
    w = np.minimum(w, cap.hard)
    u, v = uk // n_nodes, uk % n_nodes
    # contract the hard arcs from s and into t
    hard = w >= cap.hard
    H = csr_array(
        (np.ones(int(hard.sum()), np.int8), (u[hard], v[hard])), shape=(n_nodes,) * 2
    )
    in_s, in_t = _reach(H, s), _reach(csr_array(H.T), t)
    if np.any(in_s & in_t):
        return MinCut(
            in_s, in_t, csr_array((n_nodes,) * 2, dtype=np.int8), cap.hard, cap
        )
    node = np.arange(n_nodes)
    node[in_s], node[in_t] = s, t
    cu, cv = node[u], node[v]
    keep = cu != cv
    key = cu[keep] * n_nodes + cv[keep]
    uk, inv = np.unique(key, return_inverse=True)
    cw = np.zeros(uk.size, np.int64)
    np.add.at(cw, inv, w[keep])
    cw = np.minimum(cw, cap.hard)
    cu, cv = uk // n_nodes, uk % n_nodes
    f = max_flow(cu, cv, cw, n_nodes, s, t)
    res = np.r_[cw - f, f]
    rr, rc = np.r_[cu, cv][res > 0], np.r_[cv, cu][res > 0]
    R = csr_array((np.ones(rr.size, np.int8), (rr, rc)), shape=(n_nodes,) * 2)
    R.sum_duplicates()
    R.data[:] = 1
    src = _reach(R, s) | in_s
    snk = _reach(csr_array(R.T), t) | in_t
    value = int(np.sum(cw[src[cu] & ~src[cv]], dtype=np.int64))
    return MinCut(src, snk, R, value, cap)
