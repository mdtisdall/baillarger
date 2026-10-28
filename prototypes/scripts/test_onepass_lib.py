"""Tests of onepass_lib with exact answers or exact properties.

Run from the repository root:
    nix develop --command uv run pytest -q prototypes/scripts/test_onepass_lib.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).parent))
from onepass_lib import (  # noqa: E402
    contact_points,
    contacts,
    open_thickness,
    pure_voronoi,
    run,
    score_points,
    score_points_sides,
    separation,
    setup,
)

S6 = ndi.generate_binary_structure(3, 1)


def half_space(n: int, normal: tuple[float, float, float], t: float):
    """WM below an oblique plane through the box centre, cortex within t of
    the WM, fluid above."""
    g = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).astype(float)
    v = np.array(normal) / np.linalg.norm(normal)
    h = (g - (n - 1) / 2) @ v
    wm = h < 0
    edt = ndi.distance_transform_edt(~wm)
    cortex = ~wm & (edt <= t)
    return wm, cortex, ~wm & ~cortex


def two_plates(nx: int = 40, n: int = 24):
    """WM plates at x < 4 and x >= nx - 4, filled with cortex between."""
    wm = np.zeros((nx, n, n), bool)
    wm[:4] = True
    wm[nx - 4 :] = True
    return wm, ~wm


# ---------------------------------------------------------------- seeds


@pytest.mark.parametrize(
    "normal", [(0, 0, 1), (0.3, 0.2, 1.0), (1.0, 1.0, 1.0), (0.1, 0.9, 0.4)]
)
def test_seeds_are_one_layer(normal) -> None:
    wm, cortex, _ = half_space(30, normal, 8)
    s = setup(wm, cortex, patch=6)
    expect = wm & ndi.binary_dilation(cortex, S6)
    assert np.array_equal(s.seeds, expect)
    _, n26 = ndi.label(s.seeds, structure=np.ones((3, 3, 3)))
    assert n26 == 1
    assert np.array_equal(s.domain, cortex | s.seeds)


def test_seed_graph_is_connected_on_a_plane() -> None:
    from scipy.sparse.csgraph import connected_components

    wm, cortex, _ = half_space(30, (0.3, 0.2, 1.0), 8)
    s = setup(wm, cortex, patch=6)
    ncomp, _ = connected_components(s.graph, directed=False)
    assert ncomp == 1


# ---------------------------------------------------------------- patches


@pytest.mark.parametrize("spacing", [4, 6, 9])
def test_patches_are_a_geodesic_voronoi_partition(spacing: int) -> None:
    wm, cortex, _ = half_space(34, (0.3, 0.2, 1.0), 6)
    s = setup(wm, cortex, patch=spacing)
    comp = s.patch.ravel()[s.sflat]
    assert np.all(comp >= 0)
    assert np.array_equal(np.unique(comp), np.arange(s.npatch))
    # every seed is assigned to its nearest centre, within the spacing
    d = dijkstra(s.graph, indices=s.rep)
    assert np.array_equal(np.argmin(d, axis=0), comp) or np.allclose(
        d[comp, np.arange(comp.size)], d.min(0)
    )
    assert d.min(0).max() < spacing
    # centres are at least the spacing apart along the graph
    dc = d[:, s.rep]
    np.fill_diagonal(dc, np.inf)
    assert dc.min() >= spacing
    # each patch is connected in the seed graph
    for lab in range(s.npatch):
        sub = s.graph[comp == lab][:, comp == lab]
        from scipy.sparse.csgraph import connected_components

        assert connected_components(sub, directed=False)[0] == 1


def test_patch_sizes_have_no_slivers() -> None:
    """Greedy centres can leave small cells; report the spread and require
    that none is a near-empty sliver on a plane."""
    wm, cortex, _ = half_space(40, (0.3, 0.2, 1.0), 6)
    s = setup(wm, cortex, patch=6)
    sizes = np.bincount(s.patch.ravel()[s.sflat])
    interior = np.all((s.centre > 4) & (s.centre < 35), axis=1)
    assert np.percentile(sizes[interior], 5) >= 0.25 * np.median(sizes[interior])


def test_thin_blade_sides_are_not_joined() -> None:
    """A WM blade 2 voxels thick (crest at z = 19) with cortex on both sides:
    below the crest no seed-graph edge joins the two sides, and no patch holds
    seeds of both (a patch has radius < 4, and going round the crest from
    z < 13 takes more than 12)."""
    n = 30
    wm = np.zeros((n, n, n), bool)
    wm[14:16, :, :20] = True
    wm[:, :, :2] = True  # a base, so the two sides join round the blade's root
    cortex = ~wm & (ndi.distance_transform_edt(~wm) <= 6)
    s = setup(wm, cortex, patch=4)
    x = np.stack(np.unravel_index(s.sflat, wm.shape), -1)[:, 0]
    z = np.stack(np.unravel_index(s.sflat, wm.shape), -1)[:, 2]
    up = (z > 4) & (z < 13)
    left = (x == 14) & up
    right = (x == 15) & up
    g = s.graph.tocoo()
    assert not np.any(left[g.row] & right[g.col])
    comp = s.patch.ravel()[s.sflat]
    assert not set(comp[left]) & set(comp[right])


# ---------------------------------------------------------------- separation


def test_separation_on_a_plane_is_close_to_euclidean() -> None:
    wm, cortex, _ = half_space(34, (0.3, 0.2, 1.0), 6)
    s = setup(wm, cortex, patch=5)
    rng = np.random.default_rng(0)
    a = rng.integers(0, s.npatch, 200)
    b = rng.integers(0, s.npatch, 200)
    sep = separation(s, a, b, limit=np.inf)
    eu = np.linalg.norm(s.centre[a] - s.centre[b], axis=1)
    k = eu > 5
    ratio = sep[k] / eu[k]
    # never shorter than straight; 26-neighbour paths on the staircased seed
    # layer overestimate (measured median 1.10 on this plane)
    assert np.all(ratio >= 1 - 1e-9)
    assert np.median(ratio) < 1.15


def test_separation_round_a_blade() -> None:
    """Across a blade the separation goes round its edge, not through it."""
    n = 30
    wm = np.zeros((n, n, n), bool)
    wm[14:16, :, :20] = True
    wm[:, :, :2] = True
    cortex = ~wm & (ndi.distance_transform_edt(~wm) <= 6)
    s = setup(wm, cortex, patch=4)
    c = s.centre
    left = np.flatnonzero((c[:, 0] == 14) & (np.abs(c[:, 2] - 15) < 2))
    right = np.flatnonzero((c[:, 0] == 15) & (np.abs(c[:, 2] - 15) < 2))
    sep = separation(s, left[:1].repeat(right.size), right, limit=np.inf)
    # round the top edge (z = 19) and back: at least 2 * (19 - 15) + 1
    assert np.all(sep > 8)


# ---------------------------------------------------------------- contacts


def test_no_contacts_on_a_flat_or_curved_open_ribbon() -> None:
    """Fronts that never meet head-on give no contact pairs, with equal and
    with smoothly varying weights."""
    for wm, cortex, _ in (
        half_space(30, (0.3, 0.2, 1.0), 8),
        ball_case(),
    ):
        s = setup(wm, cortex, patch=6)
        for mu in (np.full(s.npatch, 8.0), 8.0 + 0.05 * s.centre[:, 0]):
            f = run(s, mu, 0.1 * mu)
            c = contacts(s, f, mu, 0.1 * mu)
            assert c.p.size == 0


def ball_case():
    n = 36
    g = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).astype(float)
    r = np.linalg.norm(g - 17.5, axis=-1)
    wm = r <= 8
    cortex = ~wm & (ndi.distance_transform_edt(~wm) <= 8)
    return wm, cortex, ~wm & ~cortex


@pytest.mark.parametrize(
    ("mu_a", "mu_b", "law"),
    [(10.0, 10.0, "ratio"), (12.0, 18.0, "ratio"), (9.0, 23.0, "ratio"),
     (12.0, 18.0, "const"), (14.0, 20.0, "const")],
)
def test_contact_position_between_plates(mu_a: float, mu_b: float, law: str) -> None:
    """Two plates, L = 33 between seed centres, cortex between: the contact is
    where (x - x_a - mu_a) / sigma_a = (x_b - x - mu_b) / sigma_b, at
    sub-voxel accuracy, and the two arrivals add up to L."""
    wm, cortex = two_plates()
    s = setup(wm, cortex, patch=6)
    xa, xb = 3.0, 36.0
    left = s.centre[:, 0] < 20
    mu = np.where(left, mu_a, mu_b)
    sig = 0.1 * mu if law == "ratio" else np.ones(s.npatch)
    sa, sb = (0.1 * mu_a, 0.1 * mu_b) if law == "ratio" else (1.0, 1.0)
    # (x - xa - mu_a) / sa = (xb - x - mu_b) / sb
    x_exact = (sb * (xa + mu_a) + sa * (xb - mu_b)) / (sa + sb)
    f = run(s, mu, sig)
    c = contacts(s, f, mu, sig)
    pts = contact_points(c, wm.shape)
    k = (c.axis == 0) & np.all((pts[:, 1:] > 4) & (pts[:, 1:] < 19), axis=1)
    assert k.sum() > 100
    err = pts[k, 0] - x_exact
    assert np.median(np.abs(err)) < 0.1
    assert np.max(np.abs(err)) < 0.3
    assert np.allclose(c.da[k] + c.db[k], xb - xa, atol=0.3)


def test_contact_separation() -> None:
    """Separate WM bodies are infinitely far apart, so their meeting is always
    a contact. On one connected WM (a U: two plates joined by a floor), the
    roots meeting in the filled gap are round the U, about 2 * 15 + 33 apart,
    so the meeting is a contact for sep_min below that and not above it."""
    wm, cortex = two_plates()
    s = setup(wm, cortex, patch=6)
    f = run(s)
    c = contacts(s, f)
    assert c.p.size > 0
    assert np.all(np.isinf(c.sep))
    wm = wm.copy()
    wm[:, :, :3] = True  # the floor joins the plates
    cortex = ~wm & (np.arange(24)[None, None, :] >= 3)
    s = setup(wm, cortex, patch=6)
    f = run(s)
    top = contacts(s, f)
    pts = contact_points(top, wm.shape)
    # above z = 18 only plate meets plate (the floor's corner sheets, at 45
    # degrees from (3, 2) and (36, 2), reach the midline at z = 18)
    high = (pts[:, 2] > 18.5) & (np.abs(pts[:, 0] - 19.5) < 1.5)
    assert high.sum() > 50
    # round the U along the seed layer: up one plate face from the floor's
    # seeds (z = 2), across 33, down the other (inf: beyond the search limit)
    fin = high & np.isfinite(top.sep)
    expect = 2 * (pts[fin, 2] - 2) + 33
    assert fin.sum() > 20
    # centres are up to a patch radius from the meeting roots
    assert np.all(np.abs(top.sep[fin] - expect) < 12)
    assert contacts(s, f, sep_min=200).p.size == 0


# ---------------------------------------------------------------- open tops


def test_open_thickness_on_a_flat_slab() -> None:
    """Cortex of t voxels over flat WM: every patch reads t + 0.5 (seed
    centre to the cortex/fluid face)."""
    n, t = 30, 9
    wm = np.zeros((n, n, n), bool)
    wm[:, :, :5] = True
    cortex = np.zeros_like(wm)
    cortex[:, :, 5 : 5 + t] = True
    fluid = ~wm & ~cortex
    s = setup(wm, cortex, patch=6)
    mu, has = open_thickness(s, fluid)
    assert has.all()
    assert np.allclose(mu, t + 0.5)


# ---------------------------------------------------------------- crystal vs pure


def curved_case():
    """A curved, partly fused shape: two WM bodies (a ball and a slab) with
    cortex that meets between them and fluid elsewhere."""
    n = 36
    g = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).astype(float)
    wm = (np.linalg.norm(g - np.array([17.5, 17.5, 24.0]), axis=-1) <= 6) | (g[..., 2] < 6)
    cortex = ~wm & (ndi.distance_transform_edt(~wm) <= 9)
    return wm, cortex


def test_crystal_growth_equals_pure_voronoi_for_start_times() -> None:
    """Equal speeds, random start times: additively weighted regions are
    star-shaped from their sources, so crystal growth and the pure weighted
    Voronoi partition agree, up to discretization at the boundaries."""
    wm, cortex = curved_case()
    s = setup(wm, cortex, patch=6)
    rng = np.random.default_rng(3)
    mu = rng.uniform(5, 12, s.npatch)
    sig = np.ones(s.npatch)
    f = run(s, mu, sig)
    plab, pbest = pure_voronoi(s, mu, sig)
    reach = s.domain & (f.label >= 0)
    assert reach.sum() == s.domain.sum()
    agree = f.label[reach] == plab[reach]
    # disagreements are only next to a pure boundary: there each label's own
    # update is one-sided, so its arrival is late by up to about a voxel
    # (measured: median 0.46, 95th percentile 1.17 at disagreeing voxels)
    edge = np.zeros_like(reach)
    for ax in range(3):
        d = np.diff(plab, axis=ax) != 0
        sl = [slice(None)] * 3
        sl[ax] = slice(0, -1)
        edge[tuple(sl)] |= d
        sl[ax] = slice(1, None)
        edge[tuple(sl)] |= d
    near = ndi.binary_dilation(edge, iterations=1)
    assert np.all(near[reach][~agree])
    # weighted arrival: crystal growth can only be later than pure
    assert np.all(f.arrival[reach] >= pbest[reach] - 0.05 * np.abs(pbest[reach]) - 0.5)


def test_slow_pure_regions_lie_inside_crystal_regions() -> None:
    """Two plates, the left slow (speed 0.5) and the right fast (speed 1),
    equal start: the slow side's pure region (one slab, star-shaped from it)
    lies inside its crystal-growth region."""
    wm, cortex = two_plates()
    s = setup(wm, cortex, patch=6)
    left = s.centre[:, 0] < 20
    sig = np.where(left, 0.5, 1.0)
    mu = np.zeros(s.npatch)
    f = run(s, mu, sig)
    plab, _ = pure_voronoi(s, mu, sig)
    slow_pure = s.domain & left[np.maximum(plab, 0)] & (plab >= 0)
    slow_cry = s.domain & left[np.maximum(f.label, 0)] & (f.label >= 0)
    inner = slow_pure & ~ndi.binary_dilation(~slow_pure, iterations=1)
    assert np.all(slow_cry[inner])


def test_column_tops_ignore_fluid_met_sideways() -> None:
    """A slab of t = 9 over flat WM with a block of cortex 6 higher on half of
    it: next to fluid sideways (the block's side face) is not a top with
    along=True, so every patch reads its own column's thickness, 9.5 or 15.5;
    the default rule lets the side face into the patches at the step."""
    n = 36
    wm = np.zeros((n, n, n), bool)
    wm[:, :, :5] = True
    cortex = np.zeros_like(wm)
    cortex[:, :, 5:14] = True
    cortex[18:, :, 14:20] = True
    fluid = ~wm & ~cortex
    s = setup(wm, cortex, patch=6)
    mu, has = open_thickness(s, fluid, along=True)
    x = s.centre[:, 0]
    assert np.allclose(mu[x <= 14], 9.5)
    assert np.allclose(mu[x >= 21], 15.5)
    # with the default rule the side face (arc lengths 10-15) belongs to the
    # thick side's patches at the step and pulls them below 15.5
    mu0, _ = open_thickness(s, fluid)
    near = (x >= 17) & (x < 22)
    assert np.any(mu0[near] < 15.4)


def test_column_tops_in_a_thin_gap() -> None:
    """Plates with ribbons of 12 and 18 voxels and a 2-voxel fluid gap (P8's
    geometry): the thick ribbon's tops are nearer the other plate's WM, so a
    direction signed by the WM distance points the wrong way there. The tops
    must come from the front's own flow: 12.5 and 18.5 on the two sides."""
    nx, n = 40, 24
    wm = np.zeros((nx, n, n), bool)
    wm[:4] = True
    wm[36:] = True  # seeds at x = 3 and x = 36
    cortex = np.zeros_like(wm)
    cortex[4:16] = True  # 12 voxels
    cortex[18:36] = True  # 18 voxels, after the gap at x = 16, 17
    fluid = ~wm & ~cortex
    # x = 18, 19: nearer the left WM (15, 16) than their own (18, 17)
    s = setup(wm, cortex, patch=6)
    mu, has = open_thickness(s, fluid, along=True)
    left = s.centre[:, 0] < 20
    assert has.all()
    assert np.allclose(mu[left], 12.5)
    assert np.allclose(mu[~left], 18.5)


@pytest.mark.parametrize("flip", [1.0, -1.0])
def test_score_points_between_plates(flip: float) -> None:
    """Plates with seeds at x = 3 and 36, mu 12 (left, thinner) and 18, sigma
    0.1 mu: the contact is at x = 3 + 33 * 12 / 30 = 16.2. A point at 18.2 lies
    2 voxels on the thick side, so the contact is beyond it toward the thinner
    side: offset +2 h, S_thin 15.2 h, S_thick 17.8 h. A point at 14.2: offset
    -2 h. The sign of the column direction given does not matter."""
    wm, cortex = two_plates()
    s = setup(wm, cortex, patch=6)
    left = s.centre[:, 0] < 20
    mu = np.where(left, 12.0, 18.0)
    sig = 0.1 * mu
    f = run(s, mu, sig)
    c = contacts(s, f, mu, sig)
    cp = contact_points(c, wm.shape)
    h = 0.16
    P = np.array([[18.2, 11.0, 12.0], [14.2, 12.0, 11.0]])
    u = np.array([[flip, 0.0, 0.0], [-flip, 0.0, 0.0]])
    rows = score_points(P, u, c, cp, mu, sig, h)
    assert rows[0]["offset"] == pytest.approx(2 * h, abs=0.02 * h * 10)
    assert rows[1]["offset"] == pytest.approx(-2 * h, abs=0.02 * h * 10)
    assert rows[0]["S_thin"] == pytest.approx(15.2 * h, abs=0.2 * h)
    assert rows[0]["S_thick"] == pytest.approx(17.8 * h, abs=0.2 * h)
    assert rows[0]["thin_mu_minus_S"] == pytest.approx((12 - 15.2) * h, abs=0.2 * h)
    assert rows[1]["S_thin"] == pytest.approx(11.2 * h, abs=0.2 * h)
    assert rows[1]["S_thick"] == pytest.approx(21.8 * h, abs=0.2 * h)


@pytest.mark.parametrize("flip", [1.0, -1.0])
def test_score_points_sides_on_a_u(flip: float) -> None:
    """The U of test_contact_separation (plates joined by a floor), mu 12 on the left
    plate and 18 on the right, sigma 0.1 mu, every boundary kept (sep_min 0): the
    sides found locally at each point give the analytic contact x = 16.2 above the
    floor's corner sheets, with no separation threshold."""
    wm, _ = two_plates()
    wm = wm.copy()
    wm[:, :, :3] = True
    cortex = ~wm & (np.arange(24)[None, None, :] >= 3)
    s = setup(wm, cortex, patch=6)
    left = s.centre[:, 0] < 20
    mu = np.where(left, 12.0, 18.0)
    sig = 0.1 * mu
    f = run(s, mu, sig)
    c = contacts(s, f, mu, sig, sep_min=0.0)
    cp = contact_points(c, wm.shape)
    h = 0.16
    P = np.array([[18.2, 11.0, 20.0], [14.2, 12.0, 21.0]])
    u = np.array([[flip, 0.0, 0.0], [-flip, 0.0, 0.0]])
    rows = score_points_sides(s, f, P, u, c, cp, mu, sig, h)
    assert rows[0]["offset"] == pytest.approx(2 * h, abs=0.2 * h)
    assert rows[1]["offset"] == pytest.approx(-2 * h, abs=0.2 * h)
    assert rows[0]["S_thin"] == pytest.approx(15.2 * h, abs=0.3 * h)
    assert rows[0]["S_thick"] == pytest.approx(17.8 * h, abs=0.3 * h)
