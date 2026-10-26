"""Tests for baillarger.regions: plan test 4 (the merging is Ward clustering
with the column graph as connectivity), levels, components and the border
term."""

import numpy as np
import numpy.typing as npt
from sklearn.cluster import AgglomerativeClustering

from baillarger import regions as rg

type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]


def grid_graph(nx: int, ny: int) -> I64:
    e = []
    for x in range(nx):
        for y in range(ny):
            c = x * ny + y
            if y < ny - 1:
                e.append((c, c + 1))
            if x < nx - 1:
                e.append((c, c + ny))
    return np.array(e, np.int64)


def test_merge_is_ward() -> None:
    """Plan test 4: with no re-fits and equal weights, the merge tree cut at k
    regions is scikit-learn's Ward clustering with the same connectivity."""
    rng = np.random.default_rng(5)
    nx, ny, G = 6, 5, 50
    vals = rng.standard_normal(G)
    n = nx * ny
    # every column: the same values, permuted over the bins
    X = np.stack([vals[rng.permutation(G)] for _ in range(n)])
    W = np.ones((n, G, 1))
    S = X[..., None].copy()
    edges = grid_graph(nx, ny)
    tree, _ = rg.merge_regions(n, edges, np.ones(len(edges)), lambda: (W, S))
    conn = np.zeros((n, n))
    conn[edges[:, 0], edges[:, 1]] = conn[edges[:, 1], edges[:, 0]] = 1
    for k in (8, 4, 2):
        ours = rg.regions_at(tree, n, k)
        sk = AgglomerativeClustering(
            n_clusters=k, linkage="ward", connectivity=conn
        ).fit(X)
        theirs = sk.labels_
        a = sorted(sorted(np.flatnonzero(ours == c).tolist()) for c in range(k))
        b = sorted(sorted(np.flatnonzero(theirs == c).tolist()) for c in range(k))
        assert a == b


def test_three_echoes_add_their_ward_costs() -> None:
    rng = np.random.default_rng(1)
    w1, w2 = rng.uniform(0.5, 2, (10, 3)), rng.uniform(0.5, 2, (10, 3))
    s1, s2 = rng.normal(size=(10, 3)), rng.normal(size=(10, 3))
    tot = sum(rg.ward_cost(w1[:, e], s1[:, e], w2[:, e], s2[:, e]) for e in range(3))
    assert np.isclose(rg.ward_cost(w1, s1, w2, s2), tot)


def test_levels_refit_and_components() -> None:
    """Two components: the merging stops at two regions; refit is called at
    each level with the regions of that level."""
    rng = np.random.default_rng(2)
    n = 10
    edges = np.array([(i, i + 1) for i in range(4)] + [(i, i + 1) for i in range(5, 9)])
    W = np.ones((n, 5, 1))
    S = rng.normal(size=(n, 5, 1))
    seen: list[int] = []

    def refit(regions: dict[int, list[int]]) -> int:
        seen.append(len(regions))
        return sum(len(c) for c in regions.values())

    tree, out = rg.merge_regions(
        n, edges, np.ones(len(edges)), lambda: (W, S), levels=[6, 3, 2], refit=refit
    )
    assert seen == [6, 3, 2] and out == {6: 10, 3: 10, 2: 10}
    assert len(tree) == n - 2
    lab = rg.regions_at(tree, n, 1)
    assert sorted(set(lab.tolist())) == [0, 1]
    assert len(set(lab[:5].tolist())) == 1 and len(set(lab[5:].tolist())) == 1


def test_border_prefers_the_longer_shared_border() -> None:
    """Region 0 touches 1 along a long border and 2 along a short one; the
    profiles make 2 slightly cheaper by Ward alone."""
    W = np.ones((3, 4, 1))
    S = np.array([[0.0, 0, 0, 0], [0.12, 0, 0, 0], [0.1, 0, 0, 0]])[..., None]
    edges = np.array([[0, 1], [0, 2]], np.int64)
    ew = np.array([20.0, 1.0])
    tree, _ = rg.merge_regions(3, edges, ew, lambda: (W, S))
    assert tree[0][1:3] == (0, 2)
    tree, _ = rg.merge_regions(3, edges, ew, lambda: (W, S), border=0.001)
    assert tree[0][1:3] == (0, 1)
