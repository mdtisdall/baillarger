"""Tests of ff_simple: the geometry of slab, noise, fundus, crown, curved and abraded.

Each closed form below is written again in this file, so that it is independent of the module.

Run from the repository root:
    nix develop --command uv run pytest -q prototypes/scripts/test_ff_simple.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import bent_phantom as BP  # noqa: E402
import ff_simple as S  # noqa: E402
import ff_types as T  # noqa: E402
import fold_phantoms as FP  # noqa: E402

FEATS = {e: dict(profile=list(np.sin(np.linspace(0, 3 * np.pi, 50))), wm_level=-0.3, fluid_level=-2.0,
                 hp_stat=0.0) for e in ("echo2", "echo3")}
FEATS_N = {e: dict(FEATS[e], hp_stat=1.0) for e in FEATS}  # with a noise level
R = FP.R_DEFAULT
HV = 0.16

# Tolerances, fixed before any run.
TOL_GRAD = 1e-6  # finite-difference gradient
TOL_SDF = 1e-3  # the signed distance against a brute-force distance (the contract's value)
TOL_ZERO = 1e-9  # the signed distance on the surface

SMALL = {
    "slab": (28, 28, 24),
    "noise": (16, 16, 16),
    "fundus_g0.48": (30, 30, 30),
    "fundus_g0.16": (30, 30, 30),
    "fundus_g0.04": (30, 30, 30),
    "crown_w0.32": (30, 30, 30),
    "crown_w0.64": (30, 30, 30),
    "crown_w1.28": (30, 30, 30),
    "curved_g0.16": (30, 30, 30),
    "curved_g0.04": (30, 30, 30),
    "abraded": (30, 30, 30),
}


# ---------------------------------------------------------------- closed forms (independent)


def vox_local(shape, origin):
    """Local mm points of all voxel centres."""
    idx = np.indices(shape).reshape(3, -1).T.astype(float)
    c = (np.array(shape) - 1) / 2.0
    return ((idx - c) * HV - np.array(origin)) @ R


def lab_slab(Y, t=2.4):
    return np.where(Y[:, 0] < 0, 2, np.where(Y[:, 0] > t, 1, 0))


def lab_fundus(Y, g, t=2.4):
    r = np.sqrt(Y[:, 0] ** 2 + np.minimum(Y[:, 1], 0.0) ** 2)
    return np.where(r < g, 1, np.where(r > g + t, 2, 0))


def lab_crown(Y, w, t=2.4):
    r = np.sqrt(Y[:, 0] ** 2 + Y[:, 1] ** 2 + np.maximum(Y[:, 2], 0.0) ** 2)
    return np.where(r < w, 2, np.where(r > w + t, 1, 0))


def lab_curved(Y, g, tA=2.5, tB=1.2, width=np.pi):
    x, yq = Y[:, 0], np.minimum(Y[:, 1], 0.0)
    r = np.hypot(x, yq)
    th = np.arctan2(x, -yq)
    u = np.clip(th / width + 0.5, 0, 1)
    t = tB + (tA - tB) * (6 * u**5 - 15 * u**4 + 10 * u**3)
    return np.where(r < g, 1, np.where(r > g + t, 2, 0))


def removed_mask(Y, g=0.16):
    out = np.zeros(len(Y), bool)
    for yc, d in ((3.0, 0.3), (8.0, 0.9)):
        rho2 = (Y[:, 1] - yc) ** 2 + Y[:, 2] ** 2
        out |= (Y[:, 0] > 0) & (rho2 < 4.0) & (Y[:, 0] - g >= 0) & (Y[:, 0] - g < d)
    return out


def lab_abraded(Y, g=0.16):
    lab = lab_curved(Y, g)
    return np.where(removed_mask(Y, g), 1, lab)


def closed_labels(name, Y):
    if name == "slab":
        return lab_slab(Y)
    if name == "noise":
        return np.zeros(len(Y), int)
    if name.startswith("fundus_g"):
        return lab_fundus(Y, float(name[8:]))
    if name.startswith("crown_w"):
        return lab_crown(Y, float(name[7:]))
    if name.startswith("curved_g"):
        return lab_curved(Y, float(name[8:]))
    return lab_abraded(Y)


# ---------------------------------------------------------------- labels


@pytest.mark.parametrize("name", S.NAMES)
def test_full_size_labels_at_every_voxel_centre(name):
    geo, _, prm = S.geometry(name)
    Y = vox_local(tuple(prm["shape"]), prm["origin_mm"])
    lab = geo.geom(Y)[0] if name == "noise" else geo.label(Y)  # the noise phantom has no Geometry
    assert np.array_equal(np.asarray(lab), closed_labels(name, Y))


@pytest.mark.parametrize("name", S.NAMES)
def test_rendered_labels_match_the_closed_form(name):
    shape = SMALL[name]
    if name == "abraded":  # put patch 1 in the small box: the local point (0.8, 3, 0) at the centre
        org = tuple(float(v) for v in -(R @ np.array([0.8, 3.0, 0.0])))
        ph = S.abraded(FEATS, shape=shape, origin_mm=org)
    else:
        ph = S.build(name, FEATS, shape=shape)
    Y = vox_local(shape, ph.params["origin_mm"])
    want = closed_labels(name, Y).reshape(shape)
    assert np.array_equal(ph.labels, want)
    if name == "abraded":
        assert removed_mask(Y).sum() > 20  # the patch is in the box
        assert np.any(lab_curved(Y, 0.16)[removed_mask(Y)] == 0)


def test_slab_frame_centres_the_cortex():
    geo, _, prm = S.geometry("slab")
    N = np.array(prm["shape"])
    centre = (N - 1) / 2.0
    Y = geo.to_local(centre[None])
    assert Y[0, 0] == pytest.approx(1.2, abs=1e-12)
    # and the local frame is the oblique R
    assert np.allclose(geo.R, R)
    assert prm["shape"] == [64, 64, 48]


def test_noise_phantom():
    ph = S.noise_only(FEATS)
    assert ph.geometry is None
    assert np.all(ph.labels == 0)
    imgs = FP.render(ph.image["fold"], FEATS)[1]
    for e, L in zip(("echo2", "echo3"), imgs):
        want = FP.profile_at(np.array(FEATS[e]["profile"]), 0.5)
        assert np.allclose(L, want, atol=1e-12, rtol=0)
    for k in (1, 2):  # the detrended image of a constant is zero
        assert np.abs(ph.det[k]).max() < 1e-9
    ph2 = S.noise_only(FEATS_N, noise=1.0, seed=3, shape=(24, 24, 24))
    assert ph2.det[1].std() > 0 and np.all(ph2.labels == 0) and ph2.tau_scale == 2.4
    again = S.noise_only(FEATS_N, noise=1.0, seed=3, shape=(24, 24, 24))
    assert np.array_equal(ph2.det[1], again.det[1])


# ---------------------------------------------------------------- depth function


def cortex_points(geo, n, lo, hi, seed=0, skip_y=True):
    rng = np.random.default_rng(seed)
    Y = rng.uniform(lo, hi, (n * 30, 3))
    ok = np.asarray(geo.label(Y)) == 0
    if skip_y:
        ok &= np.abs(Y[:, 1]) > 1e-3
    return Y[ok][:n]


@pytest.mark.parametrize("name", ["slab", "fundus_g0.16", "fundus_g0.04", "crown_w0.32", "crown_w1.28",
                                  "curved_g0.16", "curved_g0.04", "abraded"])
def test_grad_phi_is_the_gradient_of_phi(name):
    geo, _, _ = S.geometry(name, shape=(8, 8, 8))
    hi = 12.0 if name == "abraded" else 5.0
    Y = cortex_points(geo, 150, np.array([-5, -5, -5.0]), np.array([5, hi, 5.0]))
    assert len(Y) >= 50
    h = 1e-6
    G = np.zeros_like(Y)
    for a in range(3):
        e = np.zeros(3)
        e[a] = h
        G[:, a] = (geo.phi(Y + e) - geo.phi(Y - e)) / (2 * h)
    assert np.abs(G - geo.grad_phi(Y)).max() < TOL_GRAD
    assert np.all(np.linalg.norm(geo.grad_phi(Y), axis=1) > 0)
    assert np.all(np.isfinite(geo.phi(np.random.default_rng(1).uniform(-9, 9, (50, 3)))))


def test_phi_closed_forms():
    geo, _, _ = S.geometry("slab", shape=(8, 8, 8))
    assert np.allclose(geo.phi(np.array([[0, 1, 2], [1.2, 0, 0], [2.4, 5, 5], [-1.2, 0, 0]])),
                       [0, 0.5, 1, -0.5])
    geo, _, _ = S.geometry("fundus_g0.16", shape=(8, 8, 8))
    # r = 1.16 on the bank: phi = 1 - 1.0 / 2.4
    assert geo.phi(np.array([[1.16, 3.0, 0.0]]))[0] == pytest.approx(1 - 1.0 / 2.4)
    geo, _, _ = S.geometry("crown_w0.64", shape=(8, 8, 8))
    assert geo.phi(np.array([[0, 0, 0.64 + 1.2]]))[0] == pytest.approx(0.5)
    geo, _, _ = S.geometry("curved_g0.16", shape=(8, 8, 8))
    assert geo.phi(np.array([[0.16 + 1.25, 3.0, 0.0]]))[0] == pytest.approx(0.5)


# ---------------------------------------------------------------- the stop surface


def sdf_bank_tree():
    """The slot's boundary (z-invariant), in (x, y), sampled every 0.002 mm, for g given later."""

    def make(g):
        s = 0.002
        y = np.arange(0.0, 14.0 + s, s)
        bank_a = np.stack([np.full_like(y, g), y], 1)
        bank_b = np.stack([np.full_like(y, -g), y], 1)
        ph = np.linspace(-np.pi / 2, np.pi / 2, max(int(np.pi * g / s) + 2, 400))
        arc = np.stack([g * np.sin(ph), -g * np.cos(ph)], 1)
        return cKDTree(np.concatenate([bank_a, bank_b, arc]))

    return make


@pytest.mark.parametrize("name", ["fundus_g0.48", "fundus_g0.16", "fundus_g0.04", "curved_g0.16",
                                  "curved_g0.04"])
def test_slot_stop_sdf_is_the_distance_in_tissue(name):
    geo, _, prm = S.geometry(name, shape=(8, 8, 8))
    g = prm["g"]
    Y = cortex_points(geo, 200, np.array([-4, -4, -3.0]), np.array([4, 6, 3.0]), seed=2, skip_y=False)
    tree = sdf_bank_tree()(g)
    d, _ = tree.query(Y[:, :2])
    keep = d >= 0.01
    sdf = geo.stop_sdf(Y)
    assert keep.sum() > 80
    assert np.abs(sdf[keep] + d[keep]).max() < TOL_SDF


def test_slab_and_crown_stop_sdf():
    geo, _, _ = S.geometry("slab", shape=(8, 8, 8))
    Y = np.array([[2.4, 1, 1], [1.0, 0, 0], [3.0, 0, 0], [-1.0, 4, 4]])
    assert np.allclose(geo.stop_sdf(Y), [0, -1.4, 0.6, -3.4], atol=TOL_ZERO)
    assert np.allclose(geo.stop_sdf(Y), Y[:, 0] - 2.4)
    for name in ("crown_w0.32", "crown_w0.64", "crown_w1.28"):
        geo, _, prm = S.geometry(name, shape=(8, 8, 8))
        w, t = prm["w"], prm["t"]
        Rr = w + t
        # the meridian (rho, z): the shaft's wall rho = Rr for z <= 0, then the hemisphere
        s = 0.002
        zz = np.arange(-14.0, 0.0 + s, s)
        ang = np.linspace(0, np.pi / 2, int(np.pi * Rr / s))
        wall = np.concatenate([np.stack([np.full_like(zz, Rr), zz], 1),
                               np.stack([Rr * np.cos(ang), Rr * np.sin(ang)], 1)])
        tree = cKDTree(wall)
        Y = cortex_points(geo, 150, np.array([-4, -4, -4.0]), np.array([4, 4, 4.0]), seed=5, skip_y=False)
        d, _ = tree.query(np.stack([np.hypot(Y[:, 0], Y[:, 1]), Y[:, 2]], 1))
        keep = d >= 0.01
        assert keep.sum() > 50
        assert np.abs(geo.stop_sdf(Y)[keep] + d[keep]).max() < TOL_SDF


@pytest.mark.parametrize("name", ["slab", "fundus_g0.16", "fundus_g0.04", "crown_w0.64", "curved_g0.16",
                                  "curved_g0.04", "abraded"])
def test_stop_sdf_zero_on_the_surface_and_sign(name):
    geo, _, prm = S.geometry(name, shape=(8, 8, 8))
    rng = np.random.default_rng(7)
    n = 60
    if name == "slab":
        Y = np.column_stack([np.full(n, 2.4), rng.uniform(-5, 5, (n, 2))])
    elif name.startswith("crown"):
        w, t = prm["w"], prm["t"]
        a, b = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, np.pi / 2, n)
        cap = (w + t) * np.column_stack([np.cos(a) * np.cos(b), np.sin(a) * np.cos(b), np.sin(b)])
        wall = np.column_stack([(w + t) * np.cos(a), (w + t) * np.sin(a), rng.uniform(-6, 0, n)])
        Y = np.concatenate([cap, wall])
    else:
        g = prm["g"]
        ph = rng.uniform(-np.pi / 2, np.pi / 2, n)  # the slot's round end and the banks
        Y = np.column_stack([g * np.sin(ph), -g * np.cos(ph), rng.uniform(-3, 3, n)])
        Y2 = np.column_stack([rng.choice([-g, g], n), rng.uniform(0, 12, n), rng.uniform(-3, 3, n)])
        if name == "abraded":  # the slot's face inside a patch is not on the stop surface
            Y2 = Y2[geo.patch_of(Y2) < 0]
            Y = Y[geo.patch_of(Y) < 0]
        Y = np.concatenate([Y, Y2])
        if name == "abraded":
            for p in geo.patches:
                a, rr = rng.uniform(0, 2 * np.pi, n), 2.0 * np.sqrt(rng.uniform(0, 1, n))
                yc = p.centre[1]
                pia = np.column_stack([np.full(n, g + p.depth), yc + rr * np.cos(a), rr * np.sin(a)])
                side = np.column_stack([g + p.depth * rng.uniform(0, 1, n), yc + 2 * np.cos(a),
                                        2 * np.sin(a)])
                Y = np.concatenate([Y, pia, side])
    # points inside bank A's slab at y > 0 on the slot's side faces have x = +-g; a point on
    # the slot of bank B is on the surface too
    assert np.abs(geo.stop_sdf(Y)).max() < TOL_ZERO
    # sign: positive exactly in the stopped region
    P = rng.uniform(-4, 12, (4000, 3))
    P[:, 2] = rng.uniform(-3, 3, 4000)
    lab = np.asarray(geo.label(P))
    sd = geo.stop_sdf(P)
    assert np.all(sd[lab == 1] > 0) and np.all(sd[lab != 1] < 0)
    assert np.array_equal(geo.stopped(P), lab == 1)


def patch_boundary_tree(g, d, R_=2.0, yc=0.0, s=0.01):
    """Dense samples of the capped cylinder rho <= R_, 0 <= x <= g + d (axis along x, centre
    (y, z) = (yc, 0)): the pia face, the cap at x = 0 and the wall."""
    L = g + d
    ax = np.arange(-R_, R_ + s, s)
    Yg, Zg = np.meshgrid(ax, ax, indexing="ij")
    m = Yg**2 + Zg**2 <= R_**2
    disc = np.stack([Yg[m], Zg[m]], 1)
    face = np.concatenate([np.column_stack([np.full(len(disc), L), disc[:, 0] + yc, disc[:, 1]]),
                           np.column_stack([np.zeros(len(disc)), disc[:, 0] + yc, disc[:, 1]])])
    th = np.linspace(0, 2 * np.pi, int(2 * np.pi * R_ / s))[:-1]
    xs = np.linspace(0, L, int(np.ceil(L / s)) + 1)
    wall = np.array([[x, yc + R_ * np.cos(a), R_ * np.sin(a)] for x in xs for a in th])
    return cKDTree(np.concatenate([face, wall]))


@pytest.mark.parametrize("k", [0, 1])
def test_patch_sdf_against_brute_force(k):
    geo, _, _ = S.geometry("abraded", shape=(8, 8, 8))
    p = geo.patches[k]
    g = 0.16
    tree = patch_boundary_tree(g, p.depth, yc=p.centre[1])
    rng = np.random.default_rng(11)
    Y = np.column_stack([rng.uniform(-1.0, 1.5, 300), p.centre[1] + rng.uniform(-3.5, 3.5, 300),
                         rng.uniform(-3.5, 3.5, 300)])
    d, _ = tree.query(Y)
    keep = d >= 0.05
    inside = p.solid(Y)
    want = np.where(inside, d, -d)
    assert keep.sum() > 150 and inside[keep].any() and (~inside[keep]).any()
    assert np.abs(geo.patch_sdf(Y, k)[keep] - want[keep]).max() < TOL_SDF
    assert np.array_equal(geo.patch_sdf(Y, k) > 0, inside)


def test_abraded_stop_sdf_is_the_nearest_of_slot_and_patches_in_tissue():
    geo, _, _ = S.geometry("abraded", shape=(8, 8, 8))
    g = 0.16
    slot = sdf_bank_tree()(g)
    trees = [patch_boundary_tree(g, p.depth, yc=p.centre[1]) for p in geo.patches]
    rng = np.random.default_rng(13)
    Y = np.column_stack([rng.uniform(0.0, 2.5, 3000), rng.uniform(0, 11, 3000), rng.uniform(-2.5, 2.5, 3000)])
    Y = Y[np.asarray(geo.label(Y)) == 0][:400]
    ds = [slot.query(Y[:, :2])[0]] + [t.query(Y)[0] for t in trees]
    dmin = np.min(ds, axis=0)
    keep = (dmin >= 0.05) & (Y[:, 0] > 0.0)
    assert keep.sum() > 100
    assert np.abs(geo.stop_sdf(Y)[keep] + dmin[keep]).max() < TOL_SDF


def test_patch_of():
    geo, _, _ = S.geometry("abraded", shape=(8, 8, 8))
    Y = np.array([[0.2, 3.0, 0.0], [0.2, 8.0, 1.9], [0.5, 8.0, 0.0], [0.2, 5.5, 0.0],
                  [0.2, 3.0, 2.1], [1.0, 3.0, 0.0], [-0.1, 3.0, 0.0], [0.4, 8.0, 1.9]])
    assert list(geo.patch_of(Y)) == [0, 1, 1, -1, -1, -1, -1, 1]
    assert geo.patches[0].depth == 0.3 and geo.patches[1].depth == 0.9
    assert geo.patches[0].centre == (0.16, 3.0, 0.0) and geo.patches[1].radius == 2.0


# ---------------------------------------------------------------- the same labels as the sources


def test_fundus_labels_equal_the_fold():
    rng = np.random.default_rng(3)
    Y = rng.uniform(-5, 5, (20000, 3))
    for g in (0.48, 0.16, 0.04):
        geo, _, _ = S.geometry(f"fundus_g{g:.2f}", shape=(8, 8, 8))
        fold = FP.Fold("fundus", (8, 8, 8), dict(g=g, t=2.4))
        assert np.array_equal(geo.label(Y), fold.geom(Y)[0])
        assert isinstance(geo, FP.Fold) and geo.kind == "fundus"


def test_crown_labels_equal_the_fold():
    Y = np.random.default_rng(4).uniform(-5, 5, (20000, 3))
    for w in (0.32, 0.64, 1.28):
        geo, _, _ = S.geometry(f"crown_w{w:.2f}", shape=(8, 8, 8))
        fold = FP.Fold("crown", (8, 8, 8), dict(w=w, t=2.4))
        assert np.array_equal(geo.label(Y), fold.geom(Y)[0])


def test_curved_labels_equal_bent_phantom():
    Y = np.random.default_rng(5).uniform(-5, 6, (20000, 3))
    for g in (0.16, 0.04):
        geo, _, _ = S.geometry(f"curved_g{g:.2f}", shape=(8, 8, 8))
        cur = BP.curved((8, 8, 8), g=g)
        assert np.array_equal(geo.label(Y), cur.geom(Y)[0])
        assert np.array_equal(geo.phi(Y), cur.phi(Y))
        assert np.array_equal(geo.bank(Y), cur.geom(Y)[2])


def test_abraded_equals_curved_outside_the_patches_and_only_adds_medium():
    geo, _, prm = S.geometry("abraded")
    cur, _, _ = S.geometry("curved_g0.16", shape=tuple(prm["shape"]), origin_mm=tuple(prm["origin_mm"]))
    Y = vox_local(tuple(prm["shape"]), prm["origin_mm"])
    la, lc = np.asarray(geo.label(Y)), np.asarray(cur.label(Y))
    rem = removed_mask(Y)
    assert np.array_equal(la[~rem], lc[~rem])
    assert np.all(la[rem] == 1) and np.all(lc[rem] == 0)
    assert rem.sum() > 1000  # both patches are in the full box
    assert np.array_equal(geo.phi(Y), cur.phi(Y))  # phi is not changed
    assert np.array_equal(geo.stopped(Y), la == 1)


# ---------------------------------------------------------------- the abraded box


def test_abraded_box():
    geo, _, prm = S.geometry("abraded")
    N = prm["shape"][0]
    assert prm["shape"] == [N, N, N] and N % 16 == 0
    g, tA, tB = 0.16, 2.5, 1.2
    pts = []
    for yc in (3.0, 8.0):  # the patches to the white surface
        for a in np.linspace(0, 2 * np.pi, 90, endpoint=False):
            for x in np.linspace(0, g + tA, 6):
                pts.append([x, yc + 2 * np.cos(a), 2 * np.sin(a)])
    for ph in np.linspace(-np.pi / 2, np.pi / 2, 40):  # the round end, the white surface
        for z in (-2.0, 0.0, 2.0):
            pts.append([(g + tA) * np.sin(ph), -(g + tA) * np.cos(ph), z])
    for x in np.linspace(-(g + tB), 0, 5):  # bank B, from the round end to the far patch
        for y in np.linspace(0, 10, 11):
            for z in (-2.0, 2.0):
                pts.append([x, y, z])
    X = geo.to_vox(np.array(pts))
    m = 1.0 / HV
    assert X.min() >= m and X.max() <= N - 1 - m
    # no smaller cube of the same rule can hold the conservative set (boxes round the parts)
    boxes = [((0, g + tA), (1.0, 5.0), (-2.0, 2.0)), ((0, g + tA), (6.0, 10.0), (-2.0, 2.0)),
             ((-(g + tA), g + tA), (-(g + tA), 0.0), (-2.0, 2.0)), ((-(g + tB), 0.0), (0.0, 10.0), (-2.0, 2.0))]
    S0 = np.array([[x, y, z] for b in boxes for x in b[0] for y in b[1] for z in b[2]]) @ R.T
    need = (S0.max(0) - S0.min(0)).max() + 2.0
    assert (N - 1) * HV >= need and (N - 16 - 1) * HV < need
    # the module's box, run again with the same rule, gives the same answer
    N2, org2 = S.abraded_box()
    assert N2 == N and np.allclose(org2, prm["origin_mm"])


# ---------------------------------------------------------------- the images and the Phantom record


@pytest.mark.parametrize("name", S.NAMES)
def test_phantom_record(name):
    ph = S.build(name, FEATS, shape=SMALL[name])
    want_ts = {"curved_g0.16": 1.85, "curved_g0.04": 1.85, "abraded": 1.85}.get(name, 2.4)
    assert ph.name == name and ph.tau_scale == pytest.approx(want_ts)
    assert ph.site == "synthetic" and ph.noise == 0.0 and ph.seed == 0
    assert ph.shape == SMALL[name]
    if name == "noise":
        assert ph.geometry is None
    else:
        assert isinstance(ph.geometry, T.Geometry)
        assert ph.geometry.shape == SMALL[name]
    json.dumps(ph.params)  # JSON-able as it is
    json.dumps(ph.record())
    for k in ("shape", "origin_mm", "R"):
        assert k in ph.params
    assert ph.params["R"] == "R_DEFAULT"
    assert ph.shape == tuple(ph.params["shape"])


def test_full_size_params():
    sh = {n: S.geometry(n)[2]["shape"] for n in S.NAMES}
    assert sh["slab"] == [64, 64, 48] and sh["noise"] == [64, 64, 48]
    assert sh["curved_g0.16"] == [80, 80, 80] and sh["curved_g0.04"] == [80, 80, 80]
    for g in (0.48, 0.16, 0.04):
        n = int(2 * (g + 2.4 + 2.8) / HV) + 1
        assert sh[f"fundus_g{g:.2f}"] == [n, n, n]
    for w in (0.32, 0.64, 1.28):
        n = int(2 * (w + 2.4 + 2.8) / HV) + 1
        assert sh[f"crown_w{w:.2f}"] == [n, n, n]
    assert S.geometry("curved_g0.16")[2]["origin_mm"] == [0.0, -1.0, 0.0]
    assert S.geometry("abraded")[2]["g"] == 0.16


def test_rendering_equals_the_plain_sources():
    """With the same noise and seed, the subclasses give the arrays of the plain classes."""
    shape = (30, 30, 30)
    for name, plain in (
        ("fundus_g0.16", FP.Fold("fundus", shape, dict(g=0.16, t=2.4))),
        ("crown_w0.64", FP.Fold("crown", shape, dict(w=0.64, t=2.4))),
        ("curved_g0.16", BP.curved(shape, g=0.16)),
    ):
        ph = S.build(name, FEATS_N, noise=1.0, seed=2, shape=shape)
        ref = FP.make(plain, None, noise=1.0, seed=2, feats=FEATS_N)
        assert min(ref["noise_sd"]) > 0
        for k in (1, 2):
            assert np.array_equal(ph.det[k], ref["det"][k])
        assert np.array_equal(ph.labels, ref["labels"])


def test_slab_image_is_the_profile_with_partial_volume_and_noise_is_matched():
    shape = (24, 24, 20)
    ph = S.slab(FEATS_N, noise=1.0, seed=1, shape=shape)
    ph0 = S.slab(FEATS_N, noise=0.0, shape=shape)
    assert min(ph.image["noise_sd"]) > 0
    assert np.array_equal(ph.labels, ph0.labels)
    assert not np.array_equal(ph.det[1], ph0.det[1])
    # the noise-free log intensity: the mean over 3^3 sub-samples of the closed-form value
    imgs = FP.render(ph0.geometry, FEATS_N)[1]
    idx = np.indices(shape).reshape(3, -1).T.astype(float)
    c = (np.array(shape) - 1) / 2.0
    o = ph0.params["origin_mm"]
    offs = (np.arange(3) + 0.5) / 3 - 0.5
    for e, L in zip(("echo2", "echo3"), imgs):
        f = FEATS_N[e]
        prof = np.array(f["profile"])
        acc = np.zeros(len(idx))
        for ox in offs:
            for oy in offs:
                for oz in offs:
                    x = (((idx + [ox, oy, oz] - c) * HV - np.array(o)) @ R)[:, 0]
                    v = np.interp(x / 2.4, (np.arange(50) + 0.5) / 50, prof)
                    acc += np.where(x < 0, f["wm_level"], np.where(x > 2.4, f["fluid_level"], v))
        assert np.allclose(L.reshape(-1), acc / 27, atol=1e-12, rtol=0)
