"""Exact-answer tests of layer_maps_lib on synthetic traces.
    nix develop --command uv run pytest -q prototypes/scripts/test_layer_maps.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import layer_maps_lib as LM  # noqa: E402

TG = np.linspace(0, 1, 97)
EDGES = [0.0, 1 / 3, 2 / 3, 1.0]  # on the grid (32 and 64 of 96)


def plane(nx=60, ny=12, h=0.16, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.arange(nx) * h, np.arange(ny) * h, indexing="ij")
    root = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    root[:, :2] += rng.uniform(-jitter, jitter, (len(root), 2))
    nrm = np.tile([0.0, 0.0, 1.0], (len(root), 1))
    return root, nrm


def warp(thick):
    """Piecewise linear warps through the cumulative thicknesses at EDGES (N, 3) -> (N, 97)."""
    knots = np.c_[np.zeros(len(thick)), np.cumsum(thick, 1)]
    return np.stack([np.interp(TG, EDGES, k) for k in knots])


def thinning(x, x0=5.0, w=2.0, t=0.5):
    """A layer of thickness t that thins linearly to zero over [x0 - w, x0] and is absent beyond."""
    return t * np.clip((x0 - x) / w, 0, 1)


def test_isopachs_are_the_interval_thicknesses():
    root, _ = plane()
    th = np.c_[np.full(len(root), 0.6), thinning(root[:, 0]), np.full(len(root), 0.8)]
    t = LM.isopachs(warp(th), TG, EDGES)
    assert np.allclose(t, th, atol=1e-12)
    # slabs not on the levels: z is linear in tau within each interval
    s = LM.isopachs(warp(th), TG, [0, 1 / 6, 0.5])
    assert np.allclose(s[:, 0], th[:, 0] / 2, atol=1e-12)
    assert np.allclose(s[:, 1], th[:, 0] / 2 + th[:, 1] / 2, atol=1e-12)


def test_a_dip_in_the_fill_is_removed_before_reading():
    z = np.linspace(0, 2, 97)[None].copy()
    z[0, 40:44] = z[0, 39] - 0.05  # the fill dips
    assert np.all(np.diff(LM.monotone(z)[0]) >= 0)
    assert LM.isopachs(z, TG, [0, 1])[0, 0] == 2.0


def test_thinning_layer_recovered_through_the_curvature_trend():
    """t = t_true exp(a H), a = 0.8 per (1/mm), H random per root. The trend is fitted on traces
    where the layer exists; detrended, the layer's thickness and where it vanishes are recovered."""
    root, nrm = plane(nx=80, ny=30)
    pr = LM.neighbour_pairs(root, nrm, 0.5)
    rng = np.random.default_rng(1)
    H = rng.normal(0, 0.3, len(root))
    x = root[:, 0]
    true = np.c_[np.full(len(root), 0.6), thinning(x), np.full(len(root), 0.8)]
    t = LM.isopachs(warp(true * np.exp(0.8 * H)[:, None]), TG, EDGES)
    a0, _ = LM.curvature_trend(t[:, 0], H, pr)
    assert abs(a0 - 0.8) < 1e-9  # a constant layer: exact
    a1, n1 = LM.curvature_trend(t[:, 1], H, pr)
    # not exact: pairs on the taper where both pass the floor favour larger H (truncation)
    assert n1 < len(pr) and abs(a1 - 0.8) < 0.05
    # an areal change that follows curvature: the layer's thick part is on more convex roots
    Hc = H + 0.5 * (x < 2.5)
    tc = LM.isopachs(warp(true * np.exp(0.8 * Hc)[:, None]), TG, EDGES)
    assert abs(LM.curvature_trend(tc[:, 1], Hc, pr)[0] - 0.8) < 0.05
    ok = tc[:, 1] > 0.1  # the plain fit of log t on H takes it as the trend
    plain = np.polyfit(Hc[ok], np.log(tc[ok, 1]), 1)[0]
    assert plain - 0.8 > 0.3
    d = LM.detrend(t[:, 1], H, a1)
    assert np.all(d[x >= 5.0] == 0)
    ins = x < 4.5
    assert np.median(np.abs(d[ins] / true[ins, 1] - 1)) < 0.05
    # where the detrended layer falls below 0.04 mm: within one spacing of the true zero
    gone = np.array([np.median(d[np.isclose(x, xx)]) < 0.04 for xx in np.unique(x)])
    x_gone = np.unique(x)[np.argmax(gone)]
    assert abs(x_gone - (5.0 - 2.0 * 0.04 / 0.5)) <= 0.16 + 1e-9


def test_lateral_gradient_exact_for_a_linear_field():
    root, nrm = plane(nx=20, ny=20, jitter=0.05)
    pr = LM.neighbour_pairs(root, nrm, 0.4)
    g = LM.lateral_gradient(0.3 * root[:, 0] - 0.7 * root[:, 1] + 2, root, nrm, pr)
    assert np.allclose(g, [0.3, -0.7, 0], atol=1e-9)
    # a tilted plane: the gradient lies in it
    R = np.array([[1, 0, 0], [0, np.cos(0.6), -np.sin(0.6)], [0, np.sin(0.6), np.cos(0.6)]])
    root2, nrm2 = root @ R.T, nrm @ R.T
    g2 = LM.lateral_gradient(0.3 * root[:, 0] - 0.7 * root[:, 1], root2, nrm2, pr)
    assert np.allclose(g2, np.array([0.3, -0.7, 0]) @ R.T, atol=1e-9)
    # a root with no neighbours, and one with a nan value: nan, the rest unchanged
    root4 = np.r_[root, [[50.0, 50, 0]]]
    val4 = np.r_[0.3 * root[:, 0] - 0.7 * root[:, 1], 1.0]
    val4[0] = np.nan
    pr4 = LM.neighbour_pairs(root4, np.r_[nrm, [[0, 0, 1.0]]], 0.4)
    g4 = LM.lateral_gradient(val4, root4, np.r_[nrm, [[0, 0, 1.0]]], pr4)
    assert np.isnan(g4[[0, -1]]).all() and np.allclose(g4[1:-1], [0.3, -0.7, 0], atol=1e-6)
    # opposite normals do not pair
    nrm3 = nrm.copy()
    nrm3[root[:, 0] > 1.5] *= -1
    assert np.all(np.sign(nrm3[LM.neighbour_pairs(root, nrm3, 0.4)][:, :, 2]).prod(1) > 0)


def test_pinch_out_is_localised():
    """The middle layer pinches out over [3, 5] mm; its fraction falls to 0 there and its lateral
    gradient is nonzero only on the taper (within the neighbour radius of it)."""
    root, nrm = plane(nx=80, ny=12)
    x = root[:, 0]
    th = np.c_[np.full(len(root), 0.6), thinning(x), np.full(len(root), 0.8)]
    t = LM.isopachs(warp(th), TG, EDGES)
    pr = LM.neighbour_pairs(root, nrm, 0.35)
    r, gr = LM.pinch_out(t, t.sum(1), root, nrm, pr)
    assert np.allclose(r[:, 1], th[:, 1] / th.sum(1))
    assert np.all(r[x >= 5.0, 1] == 0)
    inner = (x > 0.35) & (x < x.max() - 0.35)
    on = gr[inner, 1] > 1e-6
    xs = x[inner][on]
    assert xs.min() >= 3.0 - 0.35 - 1e-9 and xs.max() <= 5.0 + 0.35 + 1e-9
    assert np.all(gr[inner & (x > 3.4) & (x < 4.6), 1] > 0.05)
    # the analytic gradient of r = t2 / (t1 + t2 + t3) mid-taper, x = 4: t2 = 0.125, dt2/dx = -0.25
    mid = np.isclose(x, 4.0, atol=0.08) & inner
    exp_ = 0.25 * 1.4 / (1.4 + 0.25 * (5.0 - x[mid])) ** 2
    assert np.allclose(gr[mid, 1], exp_, rtol=0.05)


def blend(x, x0=5.0, w=0.3):
    return 1 / (1 + np.exp((x - x0) / w))


def test_facies_two_areas_with_a_gradual_border():
    """Area A (x < 5): a dark band at tau 0.4 and a bright one at 0.8; area B: only a bright band
    at 0.2. A logistic border of width 0.3 mm. Two facies, pure far from the border, mixed at it."""
    root, _ = plane(nx=100, ny=20, h=0.1)
    x = root[:, 0]
    tf = np.linspace(0, 1, 49)
    g = lambda c: np.exp(-0.5 * ((tf - c) / 0.04) ** 2)  # noqa: E731
    pa, pb = -1.0 * g(0.4) + 0.8 * g(0.8), 0.8 * g(0.2)
    s = blend(x)[:, None]
    rng = np.random.default_rng(2)
    X = s * pa + (1 - s) * pb + rng.normal(0, 0.15, (len(x), tf.size))
    X[3] = np.nan  # a trace without a profile
    f = LM.facies(X, ks=(1, 2, 3, 4))
    # a third, transition component may be chosen; folded, two end-members remain
    assert len(f["ends"]) == 2, (f["icl"], f["transitions"])
    assert np.all(np.isnan(f["memberships"][3])) and f["labels"][3] == -1
    # which end-member is A: the one whose mean has the dark band
    ia = int(np.argmin(f["means"][f["ends"], np.argmin(np.abs(tf - 0.4))]))
    m = f["end_memberships"][:, ia]
    far = np.abs(x - 5.0) > 1.0
    xs = np.unique(x)
    med = np.array([np.nanmedian(m[x == xx]) for xx in xs])
    # column medians: pure far from the border (single noisy traces stray into the transition)
    assert np.all(med[xs < 4.0] > 0.95) and np.all(med[xs > 6.0] < 0.05)
    ok = np.isfinite(m)
    assert (m[ok & (x < 4.0)] > 0.9).mean() > 0.95 and (m[ok & (x > 6.0)] < 0.1).mean() > 0.95
    # the 0.5 crossing of the column-median membership: at the border
    x_half = xs[np.argmax(med < 0.5)]
    assert abs(x_half - 5.0) <= 0.2
    # and they follow the true blend fraction across the border
    assert np.max(np.abs(med - blend(xs))) < 0.15
    # mixed memberships lie near the border only, and some exist there
    e = LM.entropy(np.nan_to_num(f["end_memberships"], nan=0.5))
    e[3] = 0
    mixed = e > 0.3
    near = np.abs(x - 5.0) < 0.5
    assert mixed[near].mean() > 0.3 and mixed[far].mean() < 0.03


def test_a_sharp_border_gives_two_components():
    root, _ = plane(nx=100, ny=20, h=0.1)
    x = root[:, 0]
    tf = np.linspace(0, 1, 49)
    g = lambda c: np.exp(-0.5 * ((tf - c) / 0.04) ** 2)  # noqa: E731
    s = (x < 5.0)[:, None].astype(float)
    X = s * (-g(0.4)) + (1 - s) * 0.8 * g(0.2) + np.random.default_rng(3).normal(0, 0.15, (len(x), tf.size))
    f = LM.facies(X, ks=(1, 2, 3, 4))
    assert f["k"] == 2 and not f["transitions"]
    assert np.all(f["labels"] == np.where(x < 5.0, f["labels"][0], 1 - f["labels"][0]))


def test_small_anomalies_are_set_aside():
    """Two areas (sharp border) and 2 % of traces with a wild profile (a strong dark plateau)."""
    root, _ = plane(nx=100, ny=20, h=0.1)
    x = root[:, 0]
    tf = np.linspace(0, 1, 49)
    g = lambda c: np.exp(-0.5 * ((tf - c) / 0.04) ** 2)  # noqa: E731
    s = (x < 5.0)[:, None].astype(float)
    rng = np.random.default_rng(4)
    X = s * (-g(0.4)) + (1 - s) * 0.8 * g(0.2) + rng.normal(0, 0.15, (len(x), tf.size))
    odd = np.zeros(len(x), bool)
    odd[rng.choice(len(x), 40, replace=False)] = True
    X[odd] += -3.0 * (tf > 0.4)
    f, f0, aside = LM.facies_set_aside(X, ks=(1, 2, 3, 4))
    assert np.array_equal(aside, odd)
    assert len(f["ends"]) == 2 and np.isnan(f["end_memberships"][odd]).all()
    lab = f["end_memberships"][~odd].argmax(1)
    assert np.all(lab == np.where(x[~odd] < 5.0, lab[0], 1 - lab[0]))


def test_fold_transitions_exact():
    a, b, c = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
    means = np.stack([a, 0.3 * a + 0.7 * b, b, c])
    ends, M, trans = LM.fold_transitions(means)
    assert ends == [0, 2, 3] and len(trans) == 1 and abs(trans[0][3] - 0.3) < 1e-12
    assert np.allclose(M[1], [0.3, 0.7, 0]) and np.allclose(M[[0, 2, 3]], np.eye(3))
    # three distinct end-members fold nothing
    assert LM.fold_transitions(np.eye(3))[0] == [0, 1, 2]


def test_smooth_on_roots_keeps_a_constant_and_averages():
    pr = np.array([[0, 1], [1, 2]])
    v = np.array([1.0, 2.0, 3.0])
    out = LM.smooth_on_roots(v, pr, np.array([1.0, 1.0]))
    assert np.allclose(out, [1.5, 2.0, 2.5])
    assert np.allclose(LM.smooth_on_roots(np.ones((3, 4)), pr, np.array([0.5, 2.0])), 1)
