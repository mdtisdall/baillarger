"""Integer capacities for scipy's maximum flow, and the minimum cut.

scipy.sparse.csgraph.maximum_flow takes int32 capacities (with int64 it has
returned a flow of 0 and no error, #14). Float costs are multiplied by one
scale factor and rounded; a hard arc (an infinite cost) gets 1 + the sum of
the finite capacities, so that no finite cut reaches it. That hard cost must
stay below 2^30: a residual capacity is at most the sum of the two opposite
arcs, and the sum of two hard costs must fit in int32.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_array
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

from baillarger.surfcut.model import Bool, Float, Int, floats, ints

SCALE = 1000.0  # costs to integers: one unit is 1 / SCALE
LIMIT = 2**30  # the hard cost must be below this


@dataclass
class Capacities:
    caps: npt.NDArray[np.int32]
    hard: int  # the capacity of a hard arc: 1 + the sum of the finite capacities
    scale: float
    max_error: float  # the largest rounding error of one finite arc, in cost units


def capacities(caps: npt.ArrayLike, scale: float = SCALE) -> Capacities:
    """Float capacities (>= 0; inf for a hard arc) to int32: each finite one
    times `scale`, rounded to the nearest integer; each hard one 1 + the sum of
    the rounded finite ones. Raises ValueError if that hard cost is 2^30 or
    more."""
    c = floats(caps)
    if np.any(np.isnan(c)) or np.any(c < 0):
        raise ValueError("capacities must be >= 0 (inf for a hard arc)")
    fin = np.isfinite(c)
    x = c[fin] * scale
    r = np.rint(x)
    total = float(r.sum())
    if total + 1 >= LIMIT:
        raise ValueError(
            f"the hard cost 1 + {total:.0f} (the sum of the finite capacities at "
            f"scale {scale:g}) is 2^30 or more; scipy's maximum flow takes int32 "
            "capacities"
        )
    hard = int(total) + 1
    out = np.full(c.size, hard, np.int32)
    out[fin] = r.astype(np.int32)
    err = float(np.max(np.abs(x - r), initial=0.0)) / scale
    return Capacities(out, hard, scale, err)


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
    capacities (inf: hard), by Dinic's maximum flow on int32 capacities.
    Parallel arcs are merged. value >= capacities.hard means that every cut
    crosses a hard arc."""
    cap = capacities(caps, scale)
    rows, cols = ints(rows), ints(cols)
    keep = (cap.caps > 0) & (rows != cols)
    key = rows[keep] * n_nodes + cols[keep]
    uk, inv = np.unique(key, return_inverse=True)  # sorted: the same graph every run
    w = np.zeros(uk.size, np.int64)
    np.add.at(w, inv, cap.caps[keep].astype(np.int64))
    w = np.minimum(w, cap.hard)
    G = csr_array(
        (
            w.astype(np.int32),
            ((uk // n_nodes).astype(np.int32), (uk % n_nodes).astype(np.int32)),
        ),
        shape=(n_nodes, n_nodes),
    )
    flow = maximum_flow(G, s, t, method="dinic").flow
    R = csr_array(G - flow)
    R.data = (R.data > 0).astype(np.int8)
    R.eliminate_zeros()
    src = np.zeros(n_nodes, bool)
    src[breadth_first_order(R, s, directed=True, return_predecessors=False)] = True
    snk = np.zeros(n_nodes, bool)
    RT = csr_array(R.T)
    snk[breadth_first_order(RT, t, directed=True, return_predecessors=False)] = True
    Gc = G.tocoo()
    cross = src[Gc.row] & ~src[Gc.col]
    value = int(np.sum(Gc.data[cross], dtype=np.int64))
    return MinCut(src, snk, R, value, cap)
