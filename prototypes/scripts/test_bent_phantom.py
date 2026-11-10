"""Tests of bent_phantom: the curved fundus, its depth function, its true
columns and its image.

Run from the repository root:
    nix develop --command uv run --with matplotlib --with pyamg \
        pytest -q prototypes/scripts/test_bent_phantom.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import bent_phantom as BP  # noqa: E402
import column_tracers as CT  # noqa: E402
import fold_phantoms as FP  # noqa: E402

# The features inline, as test_fold_phantoms.py makes them.
FEATS = {
    e: dict(
        profile=list(np.sin(np.linspace(0, 3 * np.pi, 50))),
        wm_level=-0.3,
        fluid_level=-2.0,
        hp_stat=0.0,
    )
    for e in ("echo2", "echo3")
}
SHAPE = (64, 64, 64)


def phantom(tA=2.5, tB=1.2):
    return BP.curved(SHAPE, tA=tA, tB=tB)


def roots(ph, margin_mm=1.0):
    """The white mesh's roots (column_tracers), at least margin_mm from a box face."""
    B = BP.make_curved(ph, FEATS, noise=0.0)
    v, _, nrm = CT.white_mesh(B["wm"])
    x0 = v[CT.pick_vertices(v, nrm)]
    i = np.clip(np.round(x0).astype(int), 0, np.array(SHAPE) - 1)
    return x0[B["dface"][tuple(i.T)] >= margin_mm]


def chord_dev(P):
    """The largest distance (mm) of the points of a path from its chord."""
    a, b = P[0], P[-1]
    d = (b - a) / np.linalg.norm(b - a)
    v = P - a
    return float(np.linalg.norm(v - (v @ d)[:, None] * d, axis=1).max())


def arc(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


# ---------------------------------------------------------------- geometry


def test_depth_and_labels_on_the_flat_banks():
    """On each flat bank, the depth is linear across the bank: 1 at the pia
    (|x| = g), 0 at the WM (|x| = g + t)."""
    ph = phantom()
    g, tA, tB = ph.p["g"], ph.p["tA"], ph.p["tB"]
    y = 2.0
    Y = np.array(
        [
            [g, y, 0.3],
            [g + tA, y, 0.3],
            [g + 0.25 * tA, y, 0.3],
            [-g, y, 0.3],
            [-(g + tB), y, 0.3],
            [-(g + 0.5 * tB), y, 0.3],
        ]
    )
    assert np.allclose(ph.phi(Y), [1, 0, 0.75, 1, 0, 0.5])
    Yl = np.array([[0.05, y, 0], [g + 1.0, y, 0], [g + tA + 0.1, y, 0], [0, -0.1, 0]])
    lab, rel, bank, *_ = ph.geom(Yl)
    assert list(lab) == [1, 0, 2, 1]  # fluid, cortex, WM, fluid (the slot's end)
    assert rel[1] == pytest.approx(1 - (1.0) / tA)
    assert bank[1] == 0 and ph.geom(Y[3:4])[2][0] == 1


def test_thickness_changes_smoothly_round_the_fundus():
    """The white surface's distance from the slot's edge goes from g + tB to
    g + tA round the fundus, without a step."""
    ph = phantom()
    th = np.linspace(-np.pi / 2, np.pi / 2, 721)
    t = ph.thickness(th)
    assert t[0] == pytest.approx(ph.p["tB"]) and t[-1] == pytest.approx(ph.p["tA"])
    assert np.all(np.diff(t) >= 0)
    assert np.abs(np.diff(t)).max() < 0.02


def test_gradient_equals_finite_differences():
    ph = phantom()
    rng = np.random.default_rng(0)
    Y = rng.uniform(-3, 3, (400, 3))
    lab = ph.geom(Y)[0]
    Y = Y[lab == 0]
    G = ph.grad_phi(Y)
    e = 1e-6
    fd = np.stack(
        [(ph.phi(Y + e * d) - ph.phi(Y - e * d)) / (2 * e) for d in np.eye(3)], 1
    )
    assert np.allclose(G, fd, atol=1e-5)


# ---------------------------------------------------------------- true columns


def test_equal_banks_columns_are_straight():
    """Equal banks: every true column is straight within 0.01 mm and as long
    as the cortex is thick."""
    ph = phantom(tA=1.8, tB=1.8)
    X0 = roots(ph)
    paths = BP.true_columns(ph, X0)
    for P in paths:
        Y = ph.to_local(P)
        assert chord_dev(Y) < 0.01
        assert arc(Y) == pytest.approx(1.8, abs=0.01)


def test_unequal_banks_far_from_the_fundus_are_straight():
    """Unequal banks, roots more than 1 mm along the bank from the fundus: the
    true columns are straight within 0.01 mm, of length tA or tB."""
    ph = phantom()
    X0 = roots(ph)
    far = ph.to_local(X0)[:, 1] > 1.0
    assert far.sum() > 20
    for P in BP.true_columns(ph, X0[far]):
        Y = ph.to_local(P)
        assert chord_dev(Y) < 0.01
        L = arc(Y)
        assert min(abs(L - 2.5), abs(L - 1.2)) < 0.01


def test_unequal_banks_columns_bend_at_the_fundus():
    """Unequal banks: some true columns round the fundus are curved."""
    ph = phantom()
    X0 = roots(ph)
    near = ph.to_local(X0)[:, 1] < -0.3
    dev = [chord_dev(ph.to_local(P)) for P in BP.true_columns(ph, X0[near])]
    assert max(dev) > 0.05


def level_normal(ph, E, c):
    """The unit normal (local mm) at a point E of the level set phi = c, from
    finite differences of points on that level set. The level set is
    r = g + (1 - c) t(theta) round the slot's edge, |x| = g + (1 - c) t on the
    banks. This does not use grad_phi."""
    x, y = E[0], E[1]
    if y >= 0:
        return np.array([np.sign(x), 0.0, 0.0])
    th = np.arctan2(x, -y)
    d = 1e-6

    def pt(a):
        r = ph.p["g"] + (1 - c) * ph.thickness(np.array([a]))[0]
        return np.array([r * np.sin(a), -r * np.cos(a), 0.0])

    tg = pt(th + d) - pt(th - d)
    n = np.array([-tg[1], tg[0], 0.0])
    return n / np.linalg.norm(n)


def end_tangent(Y):
    """The unit tangent at the first point of a path, from the quadratic
    through its first three points (arc length as the parameter). A second
    point nearer than half a step (a short last step) is left out."""
    if np.linalg.norm(Y[1] - Y[0]) < 0.5 * BP.STEP:
        Y = np.r_[Y[:1], Y[2:]]
    p0, p1, p2 = Y[0], Y[1], Y[2]
    s1 = np.linalg.norm(p1 - p0)
    s2 = s1 + np.linalg.norm(p2 - p1)
    d = (
        -(1 / s1 + 1 / s2) * p0
        + s2 / (s1 * (s2 - s1)) * p1
        - s1 / (s2 * (s2 - s1)) * p2
    )
    return d / np.linalg.norm(d)


@pytest.mark.parametrize("tA,tB", [(2.5, 1.2), (1.8, 1.8)])
def test_columns_meet_wm_and_pia_at_right_angles(tA, tB):
    """The angle between a column's tangent and the level set's normal is
    below 1 degree at the WM (phi = 0) and at the pia (phi = 1)."""
    ph = phantom(tA, tB)
    X0 = roots(ph)
    worst = 0.0
    for P in BP.true_columns(ph, X0):
        Y = ph.to_local(P)
        assert ph.phi(Y[:1])[0] == pytest.approx(0, abs=1e-9)
        assert ph.phi(Y[-1:])[0] == pytest.approx(1, abs=1e-9)
        for E, T, c in (
            (Y[0], end_tangent(Y), 0.0),
            (Y[-1], -end_tangent(Y[::-1]), 1.0),
        ):
            n = level_normal(ph, E, c)
            # the normal points to higher depth: from the WM to the pia
            if ph.phi(E[None] + 1e-4 * n[None])[0] < c:
                n = -n
            worst = max(worst, np.degrees(np.arccos(np.clip(T @ n, -1, 1))))
    assert worst < 1.0


def test_step_and_ends():
    """The steps are 0.01 mm, except the last one (to the pia)."""
    ph = phantom()
    X0 = roots(ph)[:50]
    for P in BP.true_columns(ph, X0):
        st = np.linalg.norm(np.diff(ph.to_local(P), axis=0), axis=1)
        assert np.allclose(st[:-1], BP.STEP, atol=1e-9)
        assert 0 < st[-1] <= BP.STEP + 1e-9


# ---------------------------------------------------------------- the image


def offsets(sub=3):
    o = (np.arange(sub) + 0.5) / sub - 0.5
    return np.array([[a, b, c] for a in o for b in o for c in o])


def test_image_is_the_mean_of_the_profile_at_the_depths():
    """Exact: the noise-free image of each echo is, at every voxel, the mean
    over the 27 sub-samples of the profile at phi (cortex), the WM level
    (phi < 0) or the fluid level (phi > 1). The labels are those of phi at the
    voxel centre."""
    ph = BP.curved((40, 40, 40), tA=2.5, tB=1.2)
    labels, imgs = FP.render(ph, FEATS)
    idx = np.indices(ph.shape).reshape(3, -1).T.astype(float)
    p0 = ph.phi(ph.to_local(idx))
    lab = np.where(p0 < 0, 2, np.where(p0 > 1, 1, 0))
    assert np.array_equal(labels.ravel(), lab)
    for e, img in zip(("echo2", "echo3"), imgs, strict=True):
        prof = np.array(FEATS[e]["profile"])
        acc = np.zeros(len(idx))
        for o in offsets():
            p = ph.phi(ph.to_local(idx + o))
            v = np.where(p < 0, FEATS[e]["wm_level"], FEATS[e]["fluid_level"])
            c = (p >= 0) & (p <= 1)
            v[c] = FP.profile_at(prof, p[c])
            acc += v
        assert np.allclose(img.ravel(), acc / 27, atol=1e-12)


def test_laminae_lie_on_the_level_sets():
    """Near-exact: with a profile linear in depth (a + b phi), a voxel whose
    27 sub-samples are all in the cortex has the value a + b phi(centre) up to
    the partial volume of the sub-sampling. The offsets are symmetric about
    the centre, so the linear part of phi averages out. The bound is |b| times
    the largest remainder |phi(s) - phi(c) - grad phi(c) . (s - c)| over the
    voxel's sub-samples."""
    a, b = 0.2, -1.5
    nb = 50
    xs = (np.arange(nb) + 0.5) / nb
    lin = {e: dict(FEATS[e], profile=list(a + b * xs)) for e in FEATS}
    ph = BP.curved((40, 40, 40), tA=2.5, tB=1.2)
    _, imgs = FP.render(ph, lin)
    idx = np.indices(ph.shape).reshape(3, -1).T.astype(float)
    Yc = ph.to_local(idx)
    pc = ph.phi(Yc)
    G = ph.grad_phi(Yc)
    inside = np.ones(len(idx), bool)
    rem = np.zeros(len(idx))
    for o in offsets():
        Ys = ph.to_local(idx + o)
        p = ph.phi(Ys)
        # profile_at clamps outside the first and last bin centres
        inside &= (p >= xs[0]) & (p <= xs[-1])
        rem = np.maximum(rem, np.abs(p - pc - np.sum(G * (Ys - Yc), 1)))
    assert inside.sum() > 1000
    err = np.abs(imgs[0].ravel() - (a + b * pc))[inside]
    assert np.all(err <= abs(b) * rem[inside] + 1e-12)
    assert np.median(err) < 0.01 * abs(b)


def test_make_curved_returns_the_fold_phantoms_dict():
    ph = BP.curved((40, 40, 40))
    B = BP.make_curved(ph, FEATS, noise=0.0)
    B2 = FP.make(ph, None, noise=0.0, feats=FEATS)
    assert set(B) == set(B2)
    assert np.array_equal(B["labels"], B2["labels"])
    assert set(np.unique(B["labels"])) == {0, 1, 2}


# ---------------------------------------------------------------- errors


def test_path_errors():
    """A true column against itself has no error. A copy shifted 0.1 mm across
    a flat bank has a path error and an end error of 0.1 mm."""
    ph = phantom()
    X0 = roots(ph)
    far = np.flatnonzero(ph.to_local(X0)[:, 1] > 1.0)[:10]
    paths = BP.true_columns(ph, X0[far])
    pos, ns = BP.pad(paths)
    emax, emean, eend, econ = BP.path_errors(ph, paths, pos, ns)
    assert np.allclose(emax, 0, atol=1e-9) and np.allclose(eend, 0, atol=1e-9)
    assert np.all(np.isnan(econ))
    shift = ph.dir_to_vox(np.array([0.0, 0.1, 0.0])) / FP.H
    emax, emean, eend, _ = BP.path_errors(ph, paths, pos + shift, ns)
    assert np.allclose(emax, 0.1, atol=1e-6) and np.allclose(emean, 0.1, atol=1e-6)
    assert np.allclose(eend, 0.1, atol=1e-6)
