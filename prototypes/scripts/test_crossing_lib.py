"""Tests of crossing_lib with exact answers or exact properties.

Run from the repository root:
    nix develop --command uv run --with matplotlib pytest -q prototypes/scripts/test_crossing_lib.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import crossing_lib as X  # noqa: E402
from flatten_fused2 import tensor  # noqa: E402

VOX = np.array([0.16, 0.16, 0.16])


def band_profile(depth_from_top, t):
    """A laminar profile as a function of depth below the top (mm): a bright superficial band
    (0-0.3 mm) and a dark stria at 60 % of the depth from the top."""
    d = np.asarray(depth_from_top, float)
    return 1.0 * np.exp(-0.5 * ((d - 0.15) / 0.1) ** 2) - 1.2 * np.exp(-0.5 * ((d - 0.6 * t) / 0.12) ** 2)


def apposed_slabs(tA, tB, n=(0, 40, 40)):
    """Two slabs along axis 0: WM | cortex A (tA mm) | cortex B (tB mm) | WM, contact at x = tA.
    Returns labels (0 cortex, 2 WM), image, contact position (voxel index), WM faces."""
    wm_vox = 12
    nA, nB = tA / VOX[0], tB / VOX[0]
    nx = int(np.ceil(2 * wm_vox + nA + nB)) + 1
    xi = np.arange(nx) + 0.5  # voxel centres in voxel units
    a0 = wm_vox  # WM/cortex boundary A
    c = a0 + nA  # contact
    b0 = c + nB
    lab = np.where((xi < a0) | (xi > b0), 2, 0).astype(np.int16)
    s = (xi - a0) * VOX[0]  # distance from A's WM (mm)
    prof = np.where(xi < c, band_profile(tA - s, tA), band_profile(s - tA, tB))
    lab3 = np.broadcast_to(lab[:, None, None], (nx, n[1], n[2])).copy()
    img = np.broadcast_to(prof[:, None, None], (nx, n[1], n[2])).copy()
    img[lab3 == 2] = 0.0
    return lab3, img, c - 0.5, (a0 - 0.5, b0 - 0.5)


# ---------------------------------------------------------------- depth maps

@pytest.mark.parametrize("kind", ["linear", "top"])
def test_depth_map_ends(kind):
    for t in (1.5, 2.4, 3.3):
        r = X.depth_map(np.array([0.0, t]), t, 2.4, kind)
        assert r[0] == pytest.approx(0.0) and r[1] == pytest.approx(2.4)
        s = np.linspace(0, t, 200)
        assert np.all(np.diff(X.depth_map(s, t, 2.4, kind)) > 0)


def test_top_anchored_keeps_the_top_absolute():
    t0 = 2.4
    for t in (2.0, 3.3):
        s = np.array([t - 0.5, t - 1.0])  # 0.5 and 1.0 mm below the top, inside a = 1.2 mm
        assert X.depth_map(s, t, t0, "top") == pytest.approx(t0 - np.array([0.5, 1.0]))


# ---------------------------------------------------------------- tracer and profile

def test_trace_crosses_a_contact_and_ends_on_both_WM():
    tA, tB = 2.0, 3.0
    lab, img, c, (wa, wb) = apposed_slabs(tA, tB)
    ctx = (lab == 0).astype(float)
    Tc = tensor(img, ctx)
    dface = np.full(lab.shape, 99.0)
    x0 = np.array([[c, 20.0, 20.0], [c, 20.0, 20.0], [c - 4, 15.0, 22.0]])
    d0 = np.array([[1.0, 0.2, 0.0], [-1.0, 0.1, 0.0], [1.0, 0.0, 0.0]])
    paths, out = X.trace(Tc, lab, dface, x0, d0, VOX)
    assert list(out) == [X.WM, X.WM, X.WM]
    # the sign is carried through the contact: the third path starts on A's side going +x and
    # reaches B's WM, not A's
    # each stops at its first step inside WM: within one step (0.25 voxel) past the WM face
    assert 0 <= paths[0][-1, 0] - wb <= 0.3
    assert 0 <= wa - paths[1][-1, 0] <= 0.3
    assert 0 <= paths[2][-1, 0] - wb <= 0.3
    for p in paths:  # straight along the normal
        assert np.ptp(p[:, 1]) < 0.3 and np.ptp(p[:, 2]) < 0.3


def test_trace_stops_at_fluid_face_and_length():
    lab = np.zeros((30, 30, 30), np.int16)
    lab[25:] = 1
    img = np.broadcast_to(np.sin(np.arange(30) / 2.0)[:, None, None], lab.shape).copy()
    Tc = tensor(img, np.ones(lab.shape))
    dface = np.full(lab.shape, 99.0)
    dface[:, :, :3] = 0.0
    x0 = np.array([[10.0, 15.0, 15.0]])
    _, out = X.trace(Tc, lab, dface, x0, np.array([[1.0, 0, 0]]), VOX)
    assert out[0] == X.FLUID
    _, out = X.trace(Tc, lab, dface, x0, np.array([[-1.0, 0, 0]]), VOX, lmax=1.0)
    assert out[0] == X.LENGTH
    _, out = X.trace(Tc, lab, dface, np.array([[10.0, 15.0, 5.0]]), np.array([[0.0, 0.0, -1.0]]), VOX)
    assert out[0] in (X.FACE, X.TURN)  # the normal is along axis 0: a path started across it turns


def test_profile_masked_average():
    img = np.broadcast_to(np.arange(40.0)[:, None, None], (40, 20, 20)).copy()
    mask = np.ones(img.shape, bool)
    mask[:, :, 10:] = False
    img[~mask] = -1000.0  # outside the mask: must not leak
    p, _ = X.resample(np.array([[5.0, 10.0, 9.0], [30.0, 10.0, 9.0]]), VOX)
    pr = X.profile([img], mask, p, VOX, radius=0.3)[0]
    assert np.allclose(pr, p[:, 0], atol=1e-6)


# ---------------------------------------------------------------- templates and split

def synthetic_columns(t_list, noise, rng):
    out = []
    for t in t_list:
        s = np.arange(0, t + 1e-9, X.STEP_MM)
        out.append(np.array([band_profile(t - s, t) + noise * rng.standard_normal(s.size)]))
    return out


def test_template_recovers_the_profile():
    rng = np.random.default_rng(0)
    t = rng.uniform(2.2, 2.6, 60)
    tpl = X.build_template(synthetic_columns(t, 0.05, rng), t, "linear")
    s = np.arange(0, 2.4 + 1e-9, X.STEP_MM)
    mu, sd = X.expect(tpl, s, 2.4)
    assert np.max(np.abs(mu[0] - band_profile(2.4 - s, 2.4))) < 0.12
    assert np.median(sd) == pytest.approx(0.05, rel=0.35)


def splice(tA, tB, noise, rng):
    """A contact profile from A's root to B's root, split at tA."""
    sA = np.arange(0, tA, X.STEP_MM)
    sB = np.arange(tA, tA + tB + 1e-9, X.STEP_MM)
    s = np.r_[sA, sB]
    y = np.where(s < tA, band_profile(tA - s, tA), band_profile(s - tA, tB))
    return np.array([y + noise * rng.standard_normal(s.size)]), s[-1]


T0, NOISE = 2.5, 0.5


def model_columns(t_list, kind, noise, rng):
    """Columns generated from the model a template of this kind assumes: a reference profile on
    [0, T0] (band_profile at T0) read at depth_map(s, t, T0, kind)."""
    out = []
    for t in t_list:
        s = np.arange(0, t + 1e-9, X.STEP_MM)
        r = X.depth_map(s, t, T0, kind)
        out.append(np.array([band_profile(T0 - r, T0) + noise * rng.standard_normal(s.size)]))
    return out


def model_splice(tA, tB, kind, noise, rng):
    """A contact from the same model: A's column up to tA, then B's reversed; split at tA."""
    s = np.arange(0, tA + tB + 1e-9, X.STEP_MM)
    L = s[-1]
    rA = X.depth_map(np.minimum(s, tA), tA, T0, kind)
    rB = X.depth_map(np.minimum(L - s, L - tA), L - tA, T0, kind)
    y = np.where(s < tA, band_profile(T0 - rA, T0), band_profile(T0 - rB, T0))
    return np.array([y + noise * rng.standard_normal(s.size)]), L


@pytest.mark.parametrize("kind", ["linear", "top"])
def test_profile_likelihood_finds_the_split_and_is_calibrated(kind):
    """Exact model, independent noise: the split is found and z = (mean - truth) / sd has SD
    near 1 over draws (noise large enough that the posterior is wider than the grid step)."""
    rng = np.random.default_rng(1)
    d = rng.uniform(0.0, 0.5, 40)
    t = np.r_[T0 + d, T0 - d, T0]  # median exactly T0
    tpl = X.build_template(model_columns(t, kind, NOISE, rng), t, kind)
    assert tpl.t0 == pytest.approx(T0)
    z, err = [], []
    for _ in range(60):
        tA, tB = rng.uniform(1.8, 2.4), rng.uniform(2.6, 3.4)
        y, L = model_splice(tA, tB, kind, NOISE, rng)
        xs = X.xgrid(L, step_mm=0.005)
        r = X.summarize(xs, X.loglik(y, tpl, tpl, L, xs, eff_mm=X.STEP_MM))
        err.append(r["map"] - tA)
        z.append((r["mean"] - tA) / r["sd"])
    assert np.median(np.abs(err)) < 0.1
    assert 0.7 < np.std(z) < 1.4


def test_flat_profile_returns_the_pairwise_posterior():
    rng = np.random.default_rng(2)
    t = rng.uniform(2.0, 3.0, 30)
    cols = [np.zeros((1, int(ti / X.STEP_MM) + 1)) + 0.1 * rng.standard_normal((1, int(ti / X.STEP_MM) + 1))
            for ti in t]
    tpl = X.build_template(cols, t, "linear")
    tpl.med[:] = 0.0
    tpl.sd[:] = 0.1
    L = 5.0
    y = np.zeros((1, int(L / X.STEP_MM) + 1))
    xs = X.xgrid(L, 0.05, 0.95, 0.005)
    ll = X.loglik(y, tpl, tpl, L, xs)
    assert np.ptp(ll) < 1e-6
    r = X.summarize(xs, ll + X.gauss_prior(xs, L, 2.0, 0.3, 2.6, 0.4))
    x, sd = X.pairwise_posterior(L, 2.0, 0.3, 2.6, 0.4)
    assert r["mean"] == pytest.approx(x, abs=0.005)
    assert r["sd"] == pytest.approx(sd, abs=0.005)


def test_symmetry_and_darkest_on_a_mirrored_profile():
    rng = np.random.default_rng(3)
    y, L = splice(2.0, 3.0, 0.0, rng)
    xs = X.xgrid(L)
    sc = X.symmetry(y, L, xs)
    assert xs[np.nanargmax(sc)] == pytest.approx(2.0, abs=0.05)
    # the darkest point is a stria (60 % of the way down each side), not the contact: E-dark reads
    # a dark line at the contact only where the image has one
    yd = y.copy()
    s = np.arange(y.shape[1]) * X.STEP_MM
    yd[0] -= 2.0 * np.exp(-0.5 * ((s - 2.0) / 0.05) ** 2)
    assert X.darkest(yd[0], L) == pytest.approx(2.0, abs=0.05)


# ---------------------------------------------------------------- the column volume

def test_trace_array_classes_and_lengths():
    tA, tB = 2.0, 3.0
    lab, img, c, (wa, wb) = apposed_slabs(tA, tB)
    lab_open = lab.copy()
    lab_open[int(np.ceil(c)) + 2:][lab[int(np.ceil(c)) + 2:] == 0] = 1  # bank B replaced by fluid
    Tc = tensor(img, (lab == 0).astype(float))
    dface = np.full(lab.shape, 99.0)
    y = np.linspace(8, 30, 6)
    x0 = np.stack([np.full(6, wa + 0.01), y, np.full(6, 20.0)], 1)
    d0 = np.tile([1.0, 0, 0], (6, 1))
    pos, ns, out = X.trace_array(Tc, lab, dface, x0, d0, VOX, 200)
    assert np.all(out == X.WM)
    L = (ns - 1) * X.STEP_MM
    assert np.allclose(L, tA + tB, atol=0.06)
    assert np.all(np.isnan(pos[np.arange(6), ns])) and np.all(np.isfinite(pos[np.arange(6), ns - 1]))
    pos, ns, out = X.trace_array(Tc, lab_open, dface, x0, d0, VOX, 200)
    assert np.all(out == X.FLUID)


def test_ascend_stops_at_the_contact_label():
    # u: distance from x = 0 (voxels); labels: 0 cortex, 2 WM at x < 1; 'far' beyond x = 20
    shape = (40, 10, 10)
    xi = np.arange(shape[0], dtype=float)[:, None, None] * np.ones(shape)
    u = np.maximum(xi - 1.0, 0.0)
    lab = np.zeros(shape, np.int16)
    lab[:1] = 2
    dface = np.full(shape, 99.0)
    x0 = np.array([[1.2, 5.0, 5.0], [1.2, 4.0, 4.0]])
    pos, ns, out = X.ascend(u, lab, dface, x0, VOX, 400, lambda i, q: q[:, 0] >= 20)
    assert np.all(out == X.CONTACT)
    assert np.allclose(pos[np.arange(2), ns - 1, 0], 20.0, atol=0.3)


def test_sample_traces_and_order_along():
    img = np.broadcast_to(np.arange(30.0)[:, None, None], (30, 8, 8)).copy()
    pos = np.full((2, 5, 3), np.nan, np.float32)
    pos[0, :3] = [[1, 2, 2], [2.5, 2, 2], [4, 2, 2]]
    s = X.sample_traces([img], pos)[0]
    assert np.allclose(s[0, :3], [1, 2.5, 4]) and np.all(np.isnan(s[0, 3:])) and np.all(np.isnan(s[1]))
    from scipy.sparse import diags
    n = 7
    g = diags([np.ones(n - 1), np.ones(n - 1)], [1, -1]).tocsr()
    perm = np.array([3, 0, 6, 1, 5, 2, 4])
    o, d = X.order_along(perm, g)
    assert list(o) in (list(range(n)), list(range(n))[::-1])
    assert np.allclose(d, np.arange(n))


def test_exp_chart_plane_and_cylinder():
    rng = np.random.default_rng(4)
    x = np.c_[rng.uniform(-3, 3, (50, 2)), np.zeros(50)]
    d = np.linalg.norm(x, axis=1)
    c = X.exp_chart(x, np.zeros(3), [0, 0, 1.0], d)
    # a plane: the chart is the plane itself up to a rotation
    assert np.allclose(np.linalg.norm(c, axis=1), d)
    assert np.allclose(np.linalg.norm(c[:, None] - c[None], axis=2), np.linalg.norm(x[:, None] - x[None], axis=2), atol=1e-9)
    # a cylinder of radius 2 about the y axis, x0 on top: geodesic distance along the circumference
    th = np.linspace(-1.0, 1.0, 21)
    x = np.c_[2 * np.sin(th), np.zeros(21), 2 * np.cos(th)]
    c = X.exp_chart(x, [0, 0, 2.0], [0, 0, 1.0], 2 * np.abs(th))
    e = c[-1] / np.linalg.norm(c[-1])  # the chart's axes are an arbitrary rotation
    assert np.allclose(c, np.outer(2 * th, e), atol=1e-9)
