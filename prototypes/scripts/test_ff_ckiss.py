"""Tests of ff_ckiss (the connected kiss phantoms), with exact answers or exact properties.

Run from the repository root:
    nix develop --command uv run pytest -q prototypes/scripts/test_ff_ckiss.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import ff_ckiss as C  # noqa: E402
import ff_types as T  # noqa: E402
import fold_phantoms as FP  # noqa: E402

FEATS = {e: dict(profile=list(np.sin(np.linspace(0, 3 * np.pi, 50))), wm_level=-0.3, fluid_level=-2.0, hp_stat=0.0)
         for e in ("echo2", "echo3")}
SMALL = (40, 40, 64)  # holds z in [-3.5, 3.5] mm of the slot with a margin of 7 voxels
GRAD_RTOL = 1e-6  # relative tolerance of grad_phi against centred differences (h = 1e-5 mm)
SDF_TOL = 1e-3  # mm


# ---- closed forms written from the definitions (not from the module)


def S(u):
    return 6 * u**5 - 15 * u**4 + 10 * u**3


def g_def(z, sharp):
    half = 0.08 if sharp else 2.5
    u = np.clip((np.asarray(z, float) + half) / (2 * half), 0, 1)
    return 0.32 + (0.02 - 0.32) * S(u)


def phi_def(Y, sharp):
    x, y, z = Y[:, 0], Y[:, 1], Y[:, 2]
    yq = np.minimum(y, 0.0)
    r = np.hypot(x, yq)
    th = np.arctan2(x, -yq)
    u = np.clip(th / np.pi + 0.5, 0, 1)
    t = 1.2 + (2.5 - 1.2) * S(u)
    return 1 - (r - g_def(z, sharp)) / t


def labels_def(Y, sharp):
    p = phi_def(Y, sharp)
    return np.where(p < 0, 2, np.where(p > 1, 1, 0))


# ---------------------------------------------------------------- 1. labels


@pytest.mark.parametrize("sharp", [False, True])
def test_labels_at_voxel_centres_full_box(sharp):
    geo = C.geometry(sharp)
    X = np.indices(geo.shape).reshape(3, -1).T.astype(float)
    Y = geo.to_local(X)
    assert np.array_equal(geo.label(Y), labels_def(Y, sharp))
    assert np.array_equal(geo.stopped(Y), labels_def(Y, sharp) == 1)


@pytest.mark.parametrize("variant", C.NAMES)
def test_rendered_labels_match_closed_form(variant):
    ph = C.ckiss(variant, FEATS, shape=SMALL)
    geo = ph.geometry
    X = np.indices(SMALL).reshape(3, -1).T.astype(float)
    Y = geo.to_local(X)
    want = labels_def(Y, "sharp" in variant).reshape(SMALL)
    assert np.array_equal(ph.labels, want)
    # the small box holds both regimes of g and the transition
    z = Y[:, 2].reshape(SMALL)
    med = want == 1
    assert (med & (z < -2.5)).any() and (med & (z > 2.5)).any() and (med & (np.abs(z) < 2.5)).any()


# ---------------------------------------------------------------- 2. g


@pytest.mark.parametrize("sharp", [False, True])
def test_g_values(sharp):
    geo = C.geometry(sharp)
    half = 0.08 if sharp else 2.5
    z = np.concatenate([np.linspace(-4, 4, 161), np.linspace(-half, half, 41), [-half, 0.0, half]])
    assert np.allclose(geo.g(z), g_def(z, sharp), atol=1e-12, rtol=0)
    assert np.all(geo.g(z[z < -half]) == 0.32) and np.all(geo.g(z[z > half]) == 0.02)
    zz = np.linspace(-half - 1, half + 1, 4001)
    assert np.all(np.diff(geo.g(zz)) <= 1e-15)  # monotone (non-increasing)
    assert geo.g(np.array([0.0]))[0] == pytest.approx(0.17, abs=1e-12)  # S(1/2) = 1/2


def test_g_smooth_is_c2_and_dg_matches():
    geo = C.geometry(False)
    h = 1e-4
    z = np.linspace(-3.5, 3.5, 701)
    fd = (geo.g(z + h) - geo.g(z - h)) / (2 * h)
    assert np.allclose(geo.dg(z), fd, atol=1e-7, rtol=0)
    # g, g', g'' are continuous at the ends of the transition (the quintic has zero 1st and 2nd derivatives)
    for e in (-2.5, 2.5):
        d2 = lambda zz: (geo.g(zz + 1e-3) - 2 * geo.g(zz) + geo.g(zz - 1e-3)) / 1e-6  # noqa: E731
        assert abs(d2(np.array([e]))[0]) < 5e-3
        assert abs(geo.dg(np.array([e]))[0]) < 1e-12


# ---------------------------------------------------------------- 3. grad_phi


@pytest.mark.parametrize("sharp", [False, True])
def test_grad_phi_against_finite_differences(sharp):
    geo = C.geometry(sharp)
    rng = np.random.default_rng(0)
    Y = rng.uniform([-3, -2, -3.5], [3, 3, 3.5], (400, 3))
    if sharp:  # sample the one-voxel transition more densely
        Y[:100, 2] = rng.uniform(-0.08, 0.08, 100)
    # away from y = 0 (second derivatives jump) is not needed: phi is C1 there
    G = geo.grad_phi(Y)
    h = 1e-5
    fd = np.zeros_like(G)
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        fd[:, k] = (geo.phi(Y + e) - geo.phi(Y - e)) / (2 * h)
    scale = np.maximum(np.linalg.norm(G, axis=1), 1e-3)
    assert np.all(np.linalg.norm(G - fd, axis=1) / scale < GRAD_RTOL * (50.0 if sharp else 1.0))
    assert np.allclose(geo.phi(Y), phi_def(Y, sharp), atol=1e-12)
    assert np.all(np.linalg.norm(G, axis=1) > 0)


def test_grad_phi_has_a_z_part():
    geo = C.geometry(False)
    Y = np.array([[0.5, 1.0, 0.0]])
    t = 2.5  # bank A
    assert geo.grad_phi(Y)[0, 2] == pytest.approx(geo.dg(np.array([0.0]))[0] / t)
    assert geo.grad_phi(Y)[0, 2] != 0.0


# ---------------------------------------------------------------- 4. stop_sdf


def brute_sdf_tissue(geo, y, zc, x, n=400001):
    """Minus the distance from a tissue point to the slot region, by dense sampling of z'."""
    r = np.hypot(x, min(y, 0.0))
    d0 = r - geo.g(np.array([zc]))[0]
    zs = np.linspace(zc - d0, zc + d0, n)
    f = np.hypot(np.maximum(r - geo.g(zs), 0.0), zc - zs)
    return -f.min()


@pytest.mark.parametrize("sharp", [False, True])
def test_stop_sdf_properties(sharp):
    geo = C.geometry(sharp)
    # zero on the slot's edge, where g is constant
    for z, gv in [(-6.0, 0.32), (6.0, 0.02)]:
        Y = np.array([[gv, 2.0, z], [-gv, 1.0, z], [gv * np.cos(0.3), -gv * np.sin(0.3) - 0.0, z]])
        Y[2] = [gv * np.sin(0.3), -gv * np.cos(0.3), z]
        assert np.allclose(geo.stop_sdf(Y), 0.0, atol=1e-12)
    # equals r - g (negative of the distance) where g is constant far from the transition
    rng = np.random.default_rng(1)
    for z, gv in [(-6.0, 0.32), (6.0, 0.02)]:
        Y = np.column_stack([rng.uniform(-0.8, 0.8, 60), rng.uniform(-0.8, 3, 60), np.full(60, z)])
        r = np.hypot(Y[:, 0], np.minimum(Y[:, 1], 0))
        tissue = r - gv > 1e-6
        d = geo.stop_sdf(Y)
        assert np.allclose(d[tissue], -(r - gv)[tissue], atol=SDF_TOL * 1e-3, rtol=0)
        assert np.all(d[~tissue] >= -1e-9)  # inside the slot: positive
    # never more negative than -(r - g(z)), and the sign agrees with the label
    Y = rng.uniform([-1.5, -1.5, -3.5], [1.5, 2.5, 3.5], (3000, 3))
    r = np.hypot(Y[:, 0], np.minimum(Y[:, 1], 0))
    d = geo.stop_sdf(Y)
    assert np.all(d >= -(r - geo.g(Y[:, 2])) - 1e-12)
    assert np.array_equal(d > 0, r < geo.g(Y[:, 2]))
    assert np.array_equal(d > 0, geo.stopped(Y))


@pytest.mark.parametrize("sharp", [False, True])
def test_stop_sdf_against_brute_force(sharp):
    geo = C.geometry(sharp)
    half = 0.08 if sharp else 2.5
    pts = [(0.5, 1.0, half), (0.3, 0.5, 0.0), (-0.6, 2.0, -half + 0.05), (0.2, -0.4, half + 0.1),
           (0.9, 1.0, 0.3), (-0.4, 0.2, 0.05), (0.15, 3.0, half - 0.03), (0.7, -0.2, -0.2)]
    Y = np.array([[x, y, z] for x, y, z in pts])
    d = geo.stop_sdf(Y)
    for (x, y, z), di in zip(pts, d):
        assert di == pytest.approx(brute_sdf_tissue(geo, y, z, x), abs=SDF_TOL)


# ---------------------------------------------------------------- 5. flat variants


def z_at_g(geo, level):
    z = np.linspace(-geo.p["half"] - 0.1, geo.p["half"] + 0.1, 200001)
    return z[np.argmax(geo.g(z) < level)]  # first z with g < level


@pytest.mark.parametrize("sharp", [False, True])
def test_flat_values_at_points(sharp):
    geo = C.geometry(sharp)
    rng = np.random.default_rng(2)
    Y = rng.uniform([-3, -2, -4], [3, 3, 4], (4000, 3))
    lb, rel, *_ = geo.geom(Y)
    fused = geo.g(Y[:, 2]) < 0.16
    prof = np.array(FEATS["echo2"]["profile"])
    plain = C._values(geo, FEATS, "echo2", Y, False)
    flat = C._values(geo, FEATS, "echo2", Y, True)
    ctx = lb == 0
    top = ctx & fused & (rel > 0.85)
    assert top.sum() > 20
    assert np.allclose(flat[top], np.interp(0.85, (np.arange(50) + 0.5) / 50, prof))
    # the open part and everything outside the cortex are unchanged
    assert np.array_equal(flat[~(ctx & fused)], plain[~(ctx & fused)])
    # below relative depth 0.85 the fused part is unchanged too
    low = ctx & fused & (rel <= 0.85 - 1.0 / 50)
    assert np.allclose(flat[low], plain[low])


@pytest.mark.parametrize("sharp", [False, True])
def test_flat_image_equals_plain_in_the_open_part(sharp):
    name = "ckiss_sharp" if sharp else "ckiss_smooth"
    a = C.ckiss(name, FEATS, shape=SMALL)
    b = C.ckiss(name + "_flat", FEATS, shape=SMALL)
    geo = a.geometry
    Y = geo.to_local(np.indices(SMALL).reshape(3, -1).T.astype(float)).reshape(SMALL + (3,))
    zf = z_at_g(geo, 0.16)
    margin = 0.2  # mm: more than a voxel's half diagonal (0.14 mm)
    open_part = Y[..., 2] < zf - margin
    fused_part = Y[..., 2] > zf + margin
    assert open_part.any() and fused_part.any()
    # detrend mixes the image over a window, so compare the rendered values (before detrend)
    la, ia = C.render(geo, FEATS, False)
    lb, ib = C.render(geo, FEATS, True)
    assert np.array_equal(la, lb)
    for e in range(2):
        assert np.array_equal(ia[e][open_part], ib[e][open_part])
        diff = (ia[e] != ib[e]) & fused_part & (la == 0)
        assert diff.any()
        assert np.all(ib[e][diff] != ia[e][diff])


def test_plain_render_equals_fold_phantoms_make():
    geo = C.geometry(False, SMALL)
    ref = FP.make(geo, None, noise=0.0, seed=0, feats=FEATS)
    mine = C.ckiss("ckiss_smooth", FEATS, shape=SMALL).image
    assert np.array_equal(ref["labels"], mine["labels"])
    for k in (1, 2):
        assert np.array_equal(ref["det"][k], mine["det"][k])
    mine_noisy = C.ckiss("ckiss_smooth", FEATS, noise=1.0, seed=3, shape=SMALL).image
    ref_noisy = FP.make(geo, None, noise=1.0, seed=3, feats=FEATS)
    for k in (1, 2):
        assert np.array_equal(ref_noisy["det"][k], mine_noisy["det"][k])


# ---------------------------------------------------------------- 6. the phantom


@pytest.mark.parametrize("variant", C.NAMES)
def test_phantom_fields(variant):
    ph = C.ckiss(variant, FEATS, shape=SMALL)
    assert isinstance(ph, T.Phantom)
    assert ph.name == variant and ph.site == "synthetic" and ph.noise == 0.0 and ph.seed == 0
    assert ph.tau_scale == pytest.approx(1.85)
    assert isinstance(ph.geometry, T.Geometry)
    assert ph.shape == SMALL
    p = ph.params
    assert p["tA"] == 2.5 and p["tB"] == 1.2 and p["width_deg"] == 180.0 and p["R"] == "R_DEFAULT"
    assert p["flat"] == variant.endswith("_flat")
    assert p["transition"] == ("sharp" if "sharp" in variant else "smooth")
    ph.record()  # JSON-able
    assert C.NAMES == ("ckiss_smooth", "ckiss_sharp", "ckiss_smooth_flat", "ckiss_sharp_flat")
    with pytest.raises(ValueError):
        C.ckiss("nope", FEATS)


def test_default_shape_and_box_fit():
    geo = C.geometry(False)
    assert geo.shape == (96, 96, 128)
    assert tuple(geo.origin_mm) == (0.0, -1.0, 0.0)
    # the fundus line and the transition fit in the box with at least 1 mm margin
    zs = np.linspace(-3.0, 3.0, 13)
    P = np.array([[0.0, y, z] for z in zs for y in (-0.3, 0.0, 3.0)])
    V = geo.to_vox(P)
    lo = V.min(0)
    hi = np.array(geo.shape) - 1 - V.max(0)
    assert lo.min() > 1.0 / 0.16 and hi.min() > 1.0 / 0.16
