"""Regions by agglomerative merging (plan step 1).

Each column starts as its own region. The cost of merging two touching
regions is the Ward cost of their normalized profiles on the depth grid: the
increase of the within-region residual when they share one profile,
Σ_e Σ_g W1 W2 / (W1 + W2) (m1 − m2)², with W the sum of the weights
s² / σ² and m the weighted mean of x̂ = (x − a) / s in each grid bin, over the
bins both regions have (and, per echo, the noise is diagonal). The cheapest
pair merges; the costs of its neighbours are updated; the loop ends at one
region, or when no regions touch. The bins above m_max (default 0.85, the
pial edge) are left out of the profiles.

At the re-fit levels (when the number of regions reaches one of them) the
caller re-fits the model (steps 2-5) through refit(regions) and gives new
profiles through stats(), and the costs are recomputed. border (off by
default) subtracts border x the shared faces from the cost, which prefers
compact regions (the prototype's Potts term).
"""

import heapq
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]


def column_grid_stats(
    col: I64,
    m: F64,
    f: F64,
    a: F64,
    s: F64,
    sigma2: F64,
    n: int,
    h: float,
    m_max: float = 0.85,
) -> tuple[F64, F64]:
    """Per column, grid bin and echo: W = Σ s² / σ² and S = Σ W x̂, over the
    cortex voxels with m <= m_max ((n, G, E) each)."""
    G = int(np.ceil(1.0 / h - 1e-9))
    sel = m <= m_max
    c = col[sel]
    g = np.clip((m[sel] / h).astype(np.int64), 0, G - 1)
    idx = c * G + g
    E = f.shape[1]
    W = np.empty((n, G, E))
    S = np.empty((n, G, E))
    for e in range(E):
        se = np.where(s[c, e] != 0, s[c, e], 1.0)
        xh = (f[sel, e] - a[c, e]) / se
        w = s[c, e] ** 2 / sigma2[c, e]
        W[..., e] = np.bincount(idx, w, n * G).reshape(n, G)
        S[..., e] = np.bincount(idx, w * xh, n * G).reshape(n, G)
    return W, S


def ward_cost(w1: F64, s1: F64, w2: F64, s2: F64) -> float:
    """The Ward cost of merging two profiles (sums W, S per bin and echo)."""
    ok = (w1 > 0) & (w2 > 0)
    m1 = s1[ok] / w1[ok]
    m2 = s2[ok] / w2[ok]
    return float(np.sum(w1[ok] * w2[ok] / (w1[ok] + w2[ok]) * (m1 - m2) ** 2))


def merge_regions(
    n: int,
    edges: I64,
    ew: F64,
    stats: Callable[[], tuple[F64, F64]],
    levels: list[int] | None = None,
    refit: Callable[[dict[int, list[int]]], Any] | None = None,
    border: float = 0.0,
) -> tuple[list[tuple[int, int, int, int, float, int]], dict[int, Any]]:
    """The merge loop (see the module notes).

    n columns, the column graph (edges, ew), stats() the per-column (W, S)
    profiles. At each of levels (numbers of regions), refit(regions) is
    called with {region id: columns} and its result kept, then stats() is
    called again. Returns the merge tree, one row per merge: (step, region 1,
    region 2, new region, cost, columns of the new region), with the regions
    numbered as the columns 0..n-1 and the new ones n, n+1, ...; and
    {level: result of refit}.
    """
    regions: dict[int, list[int]] = {i: [i] for i in range(n)}
    adj: dict[int, dict[int, float]] = {i: {} for i in range(n)}
    for (i, j), w in zip(edges.tolist(), ew.tolist(), strict=True):
        adj[i][j] = w
        adj[j][i] = w
    W, S = stats()
    RW = {i: W[i].copy() for i in range(n)}
    RS = {i: S[i].copy() for i in range(n)}

    def cost(r1: int, r2: int) -> float:
        return ward_cost(RW[r1], RS[r1], RW[r2], RS[r2]) - border * adj[r1][r2]

    def rebuild() -> list[tuple[float, int, int]]:
        hp = [(cost(r1, r2), r1, r2) for r1 in regions for r2 in adj[r1] if r1 < r2]
        heapq.heapify(hp)
        return hp

    heap = rebuild()
    tree: list[tuple[int, int, int, int, float, int]] = []
    next_id = n
    todo = sorted([L for L in (levels or []) if L < n], reverse=True)
    out: dict[int, Any] = {}
    while len(regions) > 1:
        if todo and len(regions) == todo[0]:
            L = todo.pop(0)
            if refit is not None:
                out[L] = refit(regions)
            W, S = stats()
            for r, cols in regions.items():
                RW[r], RS[r] = W[cols].sum(0), S[cols].sum(0)
            heap = rebuild()
        if not heap:
            break  # no touching regions left
        cst, r1, r2 = heapq.heappop(heap)
        if r1 not in regions or r2 not in regions:
            continue
        new = next_id
        next_id += 1
        regions[new] = regions.pop(r1) + regions.pop(r2)
        RW[new], RS[new] = RW.pop(r1) + RW.pop(r2), RS.pop(r1) + RS.pop(r2)
        a1, a2 = adj.pop(r1), adj.pop(r2)
        merged: dict[int, float] = {}
        for nb, w in list(a1.items()) + list(a2.items()):
            if nb not in (r1, r2):
                merged[nb] = merged.get(nb, 0.0) + w
        adj[new] = merged
        for q, w in merged.items():
            adj[q].pop(r1, None)
            adj[q].pop(r2, None)
            adj[q][new] = w
            heapq.heappush(heap, (cost(new, q), min(new, q), max(new, q)))
        tree.append((len(tree) + 1, r1, r2, new, cst, len(regions[new])))
    return tree, out


def regions_at(
    tree: list[tuple[int, int, int, int, float, int]], n: int, k: int
) -> I64:
    """The region (0..k'-1) of each column when k regions are left (or as
    few as the tree reaches)."""
    groups: dict[int, list[int]] = {i: [i] for i in range(n)}
    for _, r1, r2, new, _, _ in tree[: max(0, n - k)]:
        groups[new] = groups.pop(r1) + groups.pop(r2)
    lab = np.empty(n, np.int64)
    for i, cols in enumerate(sorted(groups.values(), key=min)):
        lab[cols] = i
    return lab
