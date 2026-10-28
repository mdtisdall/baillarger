"""Tests of fold_phantoms, fold_eval's error measures and column_tracers, with exact answers or
exact properties.

Run from the repository root:
    nix develop --command uv run --with matplotlib --with pyamg pytest -q prototypes/scripts/test_fold_phantoms.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import column_tracers as CT  # noqa: E402
import crossing_lib as X  # noqa: E402
import fold_eval as FE  # noqa: E402
import fold_phantoms as FP  # noqa: E402
from baillarger.frame import mesh_curvatures  # noqa: E402

FEATS = {e: dict(profile=list(np.sin(np.linspace(0, 3 * np.pi, 50))), wm_level=-0.3, fluid_level=-2.0, hp_stat=0.0)
         for e in ("echo2", "echo3")}


def folds():
    return [FP.Fold("crown", (60, 60, 60), dict(w=0.64, t=2.4)),
            FP.Fold("fundus", (60, 60, 60), dict(g=0.16, t=2.4)),
            FP.Fold("kiss", (90, 90, 40), dict(Rc=3.6, tA=2.0, tB=2.6, g=0.32, open_deg=120.0, flat=False))]


# ---------------------------------------------------------------- geometry

@pytest.mark.parametrize("fold", folds(), ids=lambda f: f.kind)
def test_frames_round_trip(fold):
    X0 = np.random.default_rng(0).uniform(0, 40, (50, 3))
    assert np.allclose(fold.to_vox(fold.to_local(X0)), X0)


@pytest.mark.parametrize("fold", folds(), ids=lambda f: f.kind)
def test_columns_are_straight_and_normal_to_the_laminae(fold):
    """Every point of a true column has the same foot and direction, and moving along the column
    changes the relative depth linearly from 0 at its start to 1 at its top."""
    rng = np.random.default_rng(1)
    Y = rng.uniform(-4, 4, (400, 3))
    lab, rel, bank, r, c, u = fold.geom(Y)
    ctx = np.flatnonzero(lab == 0)[:60]
    for i in ctx:
        rr = np.linspace(r[i] - 0.05, r[i] + 0.05, 5)
        Yc = c[i] + rr[:, None] * u[i]
        _, rel2, bank2, r2, c2, u2 = fold.geom(Yc)
        assert np.allclose(c2, c[i]) and np.allclose(u2, u[i]) and np.allclose(r2, rr)
        ok = np.isfinite(rel2) & (bank2 == bank[i])  # one bank (kiss: the slope flips at the contact)
        # relative depth is linear in r with slope +-1/t
        if ok.sum() >= 2:
            sl = np.diff(rel2[ok]) / np.diff(rr[ok])
            assert np.allclose(np.abs(sl), np.abs(sl[0]))


def test_crown_depths_and_truth():
    f = FP.Fold("crown", (60, 60, 60), dict(w=0.64, t=2.4))
    Y = np.array([[0, 0, 0.64 + 1.2], [0.64 + 0.6, 0, -2.0], [0, 0.3, 0], [0, 0, 4.0]])
    lab, rel, *_ = f.geom(Y)
    assert list(lab) == [0, 0, 2, 1]
    assert rel[0] == pytest.approx(0.5) and rel[1] == pytest.approx(0.25)
    tr = f.column(f.to_vox(Y[:2]))
    assert tr["r0"][0] == pytest.approx(0.64) and tr["r1"][0] == pytest.approx(3.04)
    assert tr["hmean"][0] == pytest.approx(1 / 0.64) and tr["hmean"][1] == pytest.approx(0.5 / 0.64)
    assert np.allclose(tr["u"][0], [0, 0, 1])


def test_fundus_truth():
    f = FP.Fold("fundus", (60, 60, 60), dict(g=0.16, t=2.4))
    Y = np.array([[1.0, 2.0, 0.3], [0.0, -1.5, 0.0], [0.1, 1.0, 0.0], [3.0, 1.0, 0.0]])
    lab, rel, *_ = f.geom(Y)
    assert list(lab) == [0, 0, 1, 2]
    assert rel[0] == pytest.approx((2.56 - 1.0) / 2.4)
    tr = f.column(f.to_vox(Y[:2]))
    assert tr["hmean"][0] == 0.0 and tr["hmean"][1] == pytest.approx(-0.5 / 2.56)
    assert np.allclose(tr["c"][1], [0, 0, 0]) and np.allclose(tr["u"][1], [0, -1, 0])


def test_kiss_truth():
    f = FP.Fold("kiss", (90, 90, 40), dict(Rc=3.6, tA=2.0, tB=2.6, g=0.32, open_deg=120.0, flat=False))
    # fused sector (angle 180 deg), bank A and B; open sector (angle 0): the gap
    Y = np.array([[-2.0, 0, 0], [-5.0, 0, 0], [3.6, 0, 0], [-3.6, 0, 0]])
    lab, rel, bank, *_ = f.geom(Y)
    assert list(lab) == [0, 0, 1, 0] and list(bank[:2]) == [0, 1]
    tr = f.column(f.to_vox(Y[:2]))
    assert tr["r0"][0] == pytest.approx(1.6) and tr["r1"][0] == pytest.approx(6.2) and tr["rc"][0] == 3.6
    assert tr["r0"][1] == pytest.approx(6.2) and tr["r1"][1] == pytest.approx(1.6)
    tr2 = f.column(f.to_vox(np.array([[2.0, 0.0, 0.0]])))
    assert tr2["r1"][0] == pytest.approx(3.6 - 0.16) and np.isnan(tr2["rc"][0])
    assert list(FE.true_class(tr)) == [CT.CROSS, CT.CROSS] and FE.true_class(tr2)[0] == CT.OPEN


def test_render_noise_free_matches_the_profile_in_the_cortex_interior():
    f = FP.Fold("crown", (50, 50, 50), dict(w=0.64, t=2.4))
    labels, imgs = FP.render(f, FEATS)
    idx = np.argwhere(labels == 0)
    lab, rel, *_ = f.geom(f.to_local(idx))
    # away from the edges (relative depth 0.2-0.8) the sub-sampled mean is the profile within
    # its curvature over a voxel
    m = (rel > 0.2) & (rel < 0.8)
    want = FP.profile_at(np.array(FEATS["echo2"]["profile"]), rel[m])
    got = imgs[0][tuple(idx[m].T)]
    assert np.median(np.abs(got - want)) < 0.02


def test_noise_matching_hits_the_target_statistic():
    f = FP.Fold("crown", (60, 60, 60), dict(w=0.64, t=2.4))
    # a smooth, weak profile, so that its own high-pass part is well under the target
    feats = {e: dict(FEATS[e], profile=list(0.05 * np.sin(np.linspace(0, np.pi, 50))), hp_stat=0.1) for e in FEATS}
    B = FP.make(f, None, noise=1.0, feats=feats)
    for k in (1, 2):
        assert FP.hp_stat(B["det"][k], B["ctx"]) == pytest.approx(0.1, rel=0.05)


def test_mesh_curvature_at_the_crown_tip():
    """baillarger.frame.mesh_curvatures on the phantom's white mesh: mean curvature 1/w at the tip
    and 1/(2w) on the shaft, within 15 %, smoothing over 0.3 mm."""
    w = 1.28
    f = FP.Fold("crown", (70, 70, 70), dict(w=w, t=2.4))
    labels, _ = FP.render(f, FEATS, sub=1)
    v, fa, nrm = CT.white_mesh(labels == 2)
    k1, k2 = mesh_curvatures(v * FP.H, fa, nrm, 0.3)
    Y = f.to_local(v)
    r = np.linalg.norm(Y - f.foot(Y), axis=1)
    inner = np.linalg.norm(v - 34.5, axis=1) < 18
    tip = inner & (Y[:, 2] > 0.5 * w) & (np.abs(r - w) < 0.2)
    shaft = inner & (Y[:, 2] < -1.0) & (np.abs(r - w) < 0.2)
    assert np.median((k1 + k2)[tip] / 2) == pytest.approx(1 / w, rel=0.15)
    assert np.median((k1 + k2)[shaft] / 2) == pytest.approx(0.5 / w, rel=0.15)


# ---------------------------------------------------------------- error measures

def test_errors_zero_on_the_true_column_and_the_shift_off_it():
    f = FP.Fold("crown", (60, 60, 60), dict(w=0.64, t=2.4))
    Y0 = np.array([[0.64, 0.0, -1.0], [0.0, 0.0, 0.64]])
    tr = f.column(f.to_vox(Y0))
    pos = np.full((2, 100, 3), np.nan, np.float32)
    ns = np.zeros(2, int)
    for i in range(2):
        s = np.arange(tr["r0"][i], tr["r1"][i] + 1e-9, 0.04)
        Y = tr["c"][i] + s[:, None] * tr["u"][i]
        if i == 1:  # shift sideways by 0.1 mm, perpendicular to u = z
            Y = Y + np.array([0.1, 0.0, 0.0])
        pos[i, :len(s)] = f.to_vox(Y)
        ns[i] = len(s)
    emax, emean, eend, econ = FE.errors(f, tr, pos, ns, np.full(3, FP.H))
    assert emax[0] == pytest.approx(0, abs=1e-5) and eend[0] == pytest.approx(0, abs=0.04)
    assert emax[1] == pytest.approx(0.1, abs=1e-5) and emean[1] == pytest.approx(0.1, abs=1e-5)


def test_contact_error_is_where_the_trace_crosses_the_contact():
    f = FP.Fold("kiss", (90, 90, 40), dict(Rc=3.6, tA=2.0, tB=2.6, g=0.32, open_deg=120.0, flat=False))
    Y0 = np.array([[-1.6, 0.0, 0.0]])
    tr = f.column(f.to_vox(Y0))
    s = np.arange(1.6, 6.2, 0.04)
    Y = s[:, None] * np.array([-1.0, 0, 0]) + np.array([0, 0.05, 0])
    pos = f.to_vox(Y)[None].astype(np.float32)
    _, _, _, econ = FE.errors(f, tr, pos, np.array([len(s)]), np.full(3, FP.H))
    assert econ[0] == pytest.approx(0.05, abs=1e-3)


# ---------------------------------------------------------------- tracers

def test_laplace_stencil_annihilates_linear_functions():
    """For constant D (any orientation, any coherence), a linear phi has zero divergence of
    D grad phi: every interior row of the assembled system (edges plus cells) applied to it gives
    zero, to rounding."""
    n = (16, 15, 14)
    labels = np.zeros(n, np.int16)
    labels[0] = 2
    labels[-1] = 1
    d = np.array([0.6, 0.64, 0.48])
    nv = np.broadcast_to(d / np.linalg.norm(d), n + (3,)).copy()
    for c in (0.0, 0.5, 0.95):
        A, rhs, unk, _, _ = CT.laplace_system(labels, nv, np.full(n, c), np.array([0.16, 0.17, 0.15]))
        P = np.argwhere(unk)
        g = np.array([0.3, -1.1, 0.7])
        phi = (P * np.array([0.16, 0.17, 0.15])) @ g
        inner = np.all((P >= 2) & (P <= np.array(n) - 3), 1)
        r = (A @ phi)[inner]
        assert np.max(np.abs(r)) < 1e-10 * np.abs(A).max() * np.abs(phi).max()


def test_laplace_slab_is_linear():
    """WM and fluid planes on both ends, no-flux sides, any coherence with the layer normal across
    the slab: phi is linear across it."""
    n = (24, 14, 14)
    labels = np.full(n, 0, np.int16)
    labels[:3] = 2
    labels[-3:] = 1
    nv = np.zeros(n + (3,))
    nv[..., 0] = 1.0
    phi, info = CT.laplace(labels, nv, np.full(n, 0.8), np.full(3, 0.16), tol=1e-12)
    want = (np.arange(n[0]) - 2) / (n[0] - 3 - 2)  # 0 at the last WM row, 1 at the first fluid row
    assert np.max(np.abs(phi[2:-2] - want[2:-2, None, None])) < 1e-6


def test_laplace_isotropic_cylinder_is_logarithmic():
    """Concentric cylinders, phi = 0 inside r0, 1 outside r1, D = I: phi = ln(r/r0)/ln(r1/r0) to
    within the voxelization of the boundaries (a few 1/100)."""
    n = 61
    idx = np.indices((n, n, 8)).astype(float)
    r = np.hypot(idx[0] - 30, idx[1] - 30) * 0.16
    r0, r1 = 1.0, 3.5
    labels = np.where(r < r0, 2, np.where(r > r1, 1, 0)).astype(np.int16)
    labels[r > 4.4] = 4  # outside the outer ring: not in the domain
    phi, _ = CT.laplace(labels, np.zeros((n, n, 8, 3)), np.zeros((n, n, 8)), np.full(3, 0.16))
    m = labels == 0
    want = np.log(r[m] / r0) / np.log(r1 / r0)
    assert np.max(np.abs(phi[m] - want)) < 0.06


def test_blend_turns_from_the_mesh_normal_to_the_data_normal_as_prescribed():
    """A constant field e1 e1^T (coherence 1) and a mesh normal e2: the direction is
    normalize(w e1 + (1 - w) e2), w = min(1, s / blend_mm), evaluated at each midpoint."""
    n = (60, 60, 60)
    Tc = np.zeros((6,) + n)
    Tc[0] = 1.0
    Tc[3] = Tc[5] = 1e-9
    labels = np.zeros(n, np.int16)
    dface = np.full(n, 99.0)
    x0 = np.array([[30.0, 20.0, 30.0]])
    d0 = np.array([[0.0, 1.0, 0.0]])
    vox = np.full(3, 0.16)
    pos, ns, out = CT.trace_blend(Tc, labels, dface, x0, d0, vox, 20, blend_mm=0.3)
    step = np.diff(pos[0, :ns[0]], axis=0) * vox
    for i, st in enumerate(step):
        s = i * X.STEP_MM + 0.5 * X.STEP_MM  # the midpoint's arc length
        w = min(1.0, s / 0.3)
        want = np.array([w, 1 - w, 0.0])
        want /= np.linalg.norm(want)
        assert np.allclose(st / np.linalg.norm(st), want, atol=1e-4)  # positions are float32


def _tilted_slabs():
    """Two banks along x with layers tilted by +-20 deg about z (a V of two banks meeting at x = 0),
    a sinusoidal laminar pattern in each, WM at both ends."""
    n = (60, 40, 30)
    idx = np.indices(n).astype(float)
    x = (idx[0] - 29.5) * 0.16
    y = (idx[1] - 19.5) * 0.16
    a = np.radians(20)
    sA = np.cos(a) * x + np.sin(a) * y
    sB = np.cos(a) * x - np.sin(a) * y
    img = np.where(x < 0, np.sin(sA / 0.25), np.sin(sB / 0.25))
    labels = np.zeros(n, np.int16)
    labels[np.abs(x) > 4.0] = 2
    ctx = labels == 0
    B = dict(det={1: img * ctx, 2: img * ctx}, ctx=ctx, wm=labels == 2, vox=np.full(3, 0.16))
    return B, x, np.array([np.cos(a), np.sin(a), 0.0])


def _angles(B, x, nA, v, lo, hi, margin=10):
    m = (x < hi) & (x > lo) & B["ctx"]
    m[:, :margin] = m[:, -margin:] = False
    m[:, :, :margin] = m[:, :, -margin:] = False
    return np.degrees(np.arccos(np.clip(np.abs(v[m] @ nA), 0, 1)))


def test_perbank_tensor_does_not_mix_the_far_bank():
    """Near the contact of a V of two banks, the base tensor mixes the far bank's orientation
    (16 deg off within 1.5 voxels of the contact, 3.5 deg at 1.5-3 voxels); the per-bank tensor
    leaves only the contact's own edge at the derivative scale (under half of that), and equals
    the base tensor away from the contact (its 1 deg floor is central differences on this
    pattern)."""
    B, x, nA = _tilted_slabs()
    vb, _ = CT.eig(CT.base_tensor(B))
    vp, _ = CT.eig(CT.perbank_tensor(B))
    for lo, hi in ((-0.24, 0.0), (-0.48, -0.24)):
        assert np.median(_angles(B, x, nA, vp, lo, hi)) < 0.7 * np.median(_angles(B, x, nA, vb, lo, hi))
    a_b, a_p = _angles(B, x, nA, vb, -3.0, -1.5), _angles(B, x, nA, vp, -3.0, -1.5)
    assert np.allclose(a_p, a_b, atol=0.05)


def test_multiscale_keeps_the_orientation_of_planar_layers():
    """Planar layers: both scales give the same normal, so the blend does too (to the base
    tensor's own discretization, about 1 deg on this pattern)."""
    B, x, nA = _tilted_slabs()
    vb, _ = CT.eig(CT.base_tensor(B))
    vm, _ = CT.eig(CT.multiscale_tensor(B))
    assert np.allclose(_angles(B, x, nA, vm, -3.0, -1.5), _angles(B, x, nA, vb, -3.0, -1.5), atol=0.1)


def test_rank1_eig_round_trip():
    u = np.random.default_rng(3).standard_normal((5, 4, 3, 3))
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    v, c = CT.eig(CT.rank1(u))
    assert np.allclose(np.abs((u * v).sum(-1)), 1, atol=1e-6) and np.allclose(c, 1, atol=1e-6)


def test_fitted_field_on_a_slab_is_the_distance_from_the_wm():
    """WM below a plane, layers parallel to it, coherence 1: F is the unit normal, so T is the
    distance from the WM's last voxel row (to solver precision) and its gradient is the normal."""
    n = (40, 16, 16)
    idx = np.indices(n).astype(float)
    img = np.sin(idx[0] * 0.16 / 0.2)
    labels = np.zeros(n, np.int16)
    labels[:5] = 2
    ctx = labels == 0
    B = dict(det={1: img * ctx, 2: img * ctx}, ctx=ctx, wm=labels == 2, vox=np.full(3, 0.16))
    T, info = CT.fitted(B)
    want = (np.arange(n[0]) - 4) * 0.16
    inner = T[5:-3, 4:-4, 4:-4]
    assert np.max(np.abs(inner - want[5:-3, None, None])) < 0.01
