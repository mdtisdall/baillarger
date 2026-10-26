"""The bottom depth of each region (plan step 5): how deep into the WM the
region keeps both its areal borders and its laminar transitions.

The profiles extend into the WM in steps of one depth-grid bin (r < 0, the
WM voxels of the voxel table, normalized with their column's gain and
offset). For each candidate bottom b, the WM down to b is added and:

- Test A, areal border: for each neighbouring region, over the columns within
  border_hops links of the border on each side, the separation of the two
  groups of column profiles (between-group over within-group variance, over
  the bins from b up and every echo). It fails when it drops by more than
  delta_a from its value with no WM bins.
- Test B, laminar transitions: the fraction of the region's variance that its
  template explains (in the WM: the region's mean in each WM bin), over the
  voxels from b up and every echo. It fails when it drops by more than
  delta_b from its value with no WM bins.

b_R is the deepest bin for which every depth from 0 to b passes both tests;
the search also stops where fewer than max(3, 20 %) of the region's columns
have WM voxels that deep ("voters"). Returns b_R (≤ 0, in r) and why the
search stopped one bin deeper: "A", "B", "voters" or "end".

Both tests compare a total over all the bins from b up with its value at
b = 0, so they react late (to confirm, see the plan): on a synthetic patch
whose areal contrast stops at the WM, test A stopped 3 bins (0.075 of the
thickness) too deep, and test B, which noise-only WM bins barely move, did
not stop at all.
"""

import numpy as np
import numpy.typing as npt

type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]


def bottom_depths(
    col: I64,
    m: F64,
    f: F64,
    a: F64,
    s: F64,
    sigma2: F64,
    reg: I64,
    regions: dict[int, list[int]],
    nbrs: dict[int, set[int]],
    tmpl_v: F64,
    edges: I64,
    h: float,
    border_hops: int = 2,
    delta_a: float = 0.1,
    delta_b: float = 0.1,
) -> dict[int, tuple[float, str]]:
    """The bottom of each region (see the module notes).

    col, m, f: all the voxels of the table, cortex (m >= 0) and WM (m < 0),
    with their echoes (V, E); a, s, sigma2 (n, E); reg: the region of each
    column; regions: {region: columns}; nbrs: {region: neighbouring
    regions}; tmpl_v (V, E): the template value of each cortex voxel (the WM
    voxels' are replaced); edges: the column graph.
    """
    n, E = a.shape
    c = col
    w = s[c] ** 2 / sigma2[c]
    xh = (f - a[c]) / np.where(s[c] != 0, s[c], 1.0)
    gidx = np.floor(m / h).astype(np.int64)  # cortex bins >= 0, WM bins < 0
    gmin = int(gidx.min())
    out: dict[int, tuple[float, str]] = {}
    # column profiles on the grid (all depths, every echo), for test A
    off = -gmin
    ng = int(gidx.max()) + off + 1
    cell = c * ng + (gidx + off)
    cw = np.stack(
        [np.bincount(cell, w[:, e], n * ng).reshape(n, ng) for e in range(E)], -1
    )
    cs = np.stack(
        [
            np.bincount(cell, w[:, e] * xh[:, e], n * ng).reshape(n, ng)
            for e in range(E)
        ],
        -1,
    )
    prof = np.where(cw > 0, cs / np.maximum(cw, 1e-300), np.nan)
    cw1 = cw[..., 0]
    # columns near each border (border_hops), for test A
    adj: list[list[int]] = [[] for _ in range(n)]
    for i, j in edges.tolist():
        adj[i].append(j)
        adj[j].append(i)

    def near(R: int, R2: int) -> I64:
        A = [ci for ci in regions[R] if any(reg[j] == R2 for j in adj[ci])]
        seen, front = set(A), list(A)
        for _ in range(border_hops - 1):
            nxt = []
            for ci in front:
                for j in adj[ci]:
                    if reg[j] == R and j not in seen:
                        seen.add(j)
                        nxt.append(j)
            front = nxt
        return np.array(sorted(seen), np.int64)

    def separation(c1: I64, c2: I64, lo: int) -> float:
        g = slice(lo + off, ng)
        p1 = prof[c1, g].reshape(len(c1), -1)
        p2 = prof[c2, g].reshape(len(c2), -1)
        n1, n2 = np.sum(~np.isnan(p1), 0), np.sum(~np.isnan(p2), 0)
        m1 = np.nansum(p1, 0) / np.maximum(n1, 1)
        m2 = np.nansum(p2, 0) / np.maximum(n2, 1)
        ok = (n1 > 0) & (n2 > 0)
        between = np.sum((n1 * n2 / np.maximum(n1 + n2, 1) * (m1 - m2) ** 2)[ok])
        within = np.nansum(((p1 - m1) ** 2)[:, ok]) + np.nansum(((p2 - m2) ** 2)[:, ok])
        return float(between / max(within, 1e-300))

    for R, cols in regions.items():
        inR = np.isin(c, cols)
        # test B: the fraction of variance explained; WM part = the region's
        # mean in each WM grid bin
        vb = np.flatnonzero(inR)
        gb = gidx[vb]
        wb, xb = w[vb], xh[vb]
        t = tmpl_v[vb].copy()
        wm = gb < 0
        if wm.any():
            k = gb[wm] - gmin
            for e in range(E):
                mw = np.bincount(k, wb[wm, e] * xb[wm, e]) / np.maximum(
                    np.bincount(k, wb[wm, e]), 1e-300
                )
                t[wm, e] = mw[k]
        mean = np.array([np.average(xb[:, e], weights=wb[:, e]) for e in range(E)])

        def fve(
            lo: int,
            gb: I64 = gb,
            wb: F64 = wb,
            xb: F64 = xb,
            t: F64 = t,
            mean: F64 = mean,
        ) -> float:
            sl = gb >= lo
            sse = np.sum(wb[sl] * (xb[sl] - t[sl]) ** 2)
            sst = np.sum(wb[sl] * (xb[sl] - mean) ** 2)
            return float(1 - sse / max(sst, 1e-300))

        f0 = fve(0)
        pairs = [(near(R, R2), near(R2, R)) for R2 in sorted(nbrs.get(R, ()))]
        pairs = [(c1, c2) for c1, c2 in pairs if len(c1) > 1 and len(c2) > 1]
        s0 = [separation(c1, c2, 0) for c1, c2 in pairs]
        b = 0.0
        ncols = len(cols)
        why = "end"
        for lo in range(-1, gmin - 1, -1):
            if np.sum(cw1[cols, lo + off] > 0) == 0:
                continue
            # columns that reach this deep: WM voxels at depth lo or below
            voters = np.sum(cw1[cols, : lo + off + 1].sum(1) > 0)
            if voters < max(3, 0.2 * ncols):
                why = "voters"
                break
            if fve(lo) < (1 - delta_b) * f0:
                why = "B"
                break
            if any(
                separation(c1, c2, lo) < (1 - delta_a) * s_
                for (c1, c2), s_ in zip(pairs, s0, strict=True)
            ):
                why = "A"
                break
            b = lo * h
        out[R] = (b, why)
    return out
