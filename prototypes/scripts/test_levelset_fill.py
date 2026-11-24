"""Exact-answer tests for levelset_fill (test plan §5, items 1-7). Voxels.

Run: nix develop --command uv run --with matplotlib pytest prototypes/scripts/test_levelset_fill.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pytest

import levelset_fill as L
import levelset_phantoms as PH


def run(ph, budget, mode, **kw):
    b = budget if isinstance(budget, np.ndarray) else np.full(ph.wm.shape, float(budget))
    return L.march(ph.cortex, ph.wm, ph.band, ph.phi, ph.foot, ph.normal, b, ph.n, mode=mode, **kw)


# ------------------------------------------------------------------ 1. flat slab


@pytest.mark.parametrize("mode", [L.LENGTH, L.VOLUME])
def test_slab_exact(mode):
    ph = PH.slab(t=12)
    f = run(ph, 12.0, mode)
    c = ph.cortex
    z = np.broadcast_to(np.arange(c.shape[2])[None, None, :], c.shape).astype(float)
    ex = z - 9.5
    assert L.claimed(f)[c].all()
    np.testing.assert_allclose(f.s[c], ex[c], atol=1e-9)
    np.testing.assert_allclose(f.w[c], ex[c], atol=1e-9)
    np.testing.assert_allclose(f.tau[c], ex[c] / 12, atol=1e-9)
    np.testing.assert_allclose(f.a[c], 1.0, atol=1e-9)
    # R is the foot point
    X, Y = np.meshgrid(np.arange(c.shape[0]), np.arange(c.shape[1]), indexing="ij")
    foot = np.stack([np.broadcast_to(X[..., None], c.shape), np.broadcast_to(Y[..., None], c.shape),
                     np.full(c.shape, 9.5)], -1)
    np.testing.assert_allclose(f.R[c], foot[c], atol=1e-9)
    tp = L.tops(f, ph.fluid)
    assert tp.idx.size == c.shape[0] * c.shape[1]
    np.testing.assert_allclose(tp.s, 12.0, atol=1e-9)
    np.testing.assert_allclose(tp.w, 12.0, atol=1e-9)


# ------------------------------------------------------------------ 2. crown


def test_phantom_normals_on_the_surface():
    """The analytic surface's normal is defined at voxels lying on it (phi = 0), not from x - foot."""
    ph = PH.cylinder()
    c = ph.extra["c"]
    G = np.stack(np.meshgrid(*[np.arange(m) for m in ph.wm.shape], indexing="ij"), -1).astype(float) - c
    G[..., 1] = 0
    u = G / np.maximum(np.linalg.norm(G, axis=-1, keepdims=True), 1e-12)
    cosang = np.sum(ph.normal[ph.band] * u[ph.band], 1)
    assert cosang.min() > np.cos(np.radians(1.0)), cosang.min()


def crown_exact(ph, kind):
    r0, r1, c = ph.extra["r0"], ph.extra["r1"], ph.extra["c"]
    k = 1 if kind == "cylinder" else 2
    G = np.stack(np.meshgrid(*[np.arange(m) for m in ph.wm.shape], indexing="ij"), -1).astype(float) - c
    if k == 1:
        G[..., 1] = 0
    r = np.linalg.norm(G, axis=-1)
    vol = (r ** (k + 1) - r0 ** (k + 1)) / ((k + 1) * r0**k)
    v1 = (r1 ** (k + 1) - r0 ** (k + 1)) / ((k + 1) * r0**k)
    return r, k, r0, r1, vol, v1


def tau_to_radius(tau, mode, k, r0, r1, v1):
    if mode == L.LENGTH:
        return r0 + tau * (r1 - r0)
    return (r0 ** (k + 1) + tau * v1 * (k + 1) * r0**k) ** (1 / (k + 1))


@pytest.mark.parametrize("kind", ["cylinder", "sphere"])
@pytest.mark.parametrize("mode", [L.LENGTH, L.VOLUME])
def test_crown_level_sets(kind, mode):
    """The level sets of tau sit at the exact equal-length / equal-volume radii: the radius read from
    tau is within half a voxel of the voxel's radius (95th percentile), with no bias (median)."""
    ph = getattr(PH, kind)()
    r, k, r0, r1, vol, v1 = crown_exact(ph, kind)
    b = (r1 - r0) if mode == L.LENGTH else v1
    f = run(ph, b, mode)
    m = ph.cortex & L.claimed(f)
    assert m.sum() == ph.cortex.sum()
    err = tau_to_radius(f.tau[m], mode, k, r0, r1, v1) - r[m]
    assert abs(np.median(err)) < 0.1, np.median(err)
    assert np.percentile(np.abs(err), 95) < 0.5, np.percentile(np.abs(err), 95)
    es = f.s[m] - (r[m] - r0)
    assert abs(np.median(es)) < 0.1 and np.percentile(np.abs(es), 95) < 0.5
    # the root is on the white surface (within the start band's interpolation)
    Rc = f.R[m] - ph.extra["c"]
    if k == 1:
        Rc[:, 1] = 0
    assert np.percentile(np.abs(np.linalg.norm(Rc, axis=1) - r0), 95) < 0.1


@pytest.mark.parametrize("kind", ["cylinder", "sphere"])
def test_crown_cross_section(kind):
    """a from the divergence and a from grad R both match (r / r0)^k away from the outer edge."""
    ph = getattr(PH, kind)()
    r, k, r0, r1, vol, v1 = crown_exact(ph, kind)
    f = run(ph, v1, L.VOLUME)
    inner = ph.cortex & L.claimed(f) & (r < r1 - 1.0)
    ex = (r / r0) ** k
    e_div = f.a[inner] / ex[inner] - 1
    aR = L.a_from_R(f)
    e_R = aR[inner] / ex[inner] - 1
    assert abs(np.median(e_div)) < 0.02 and np.percentile(np.abs(e_div), 95) < 0.05, (
        np.median(e_div), np.percentile(np.abs(e_div), 95))
    assert abs(np.nanmedian(e_R)) < 0.02 and np.nanpercentile(np.abs(e_R), 95) < 0.05, (
        np.nanmedian(e_R), np.nanpercentile(np.abs(e_R), 95))
    # w through the radius it implies (w grows to ~50 voxels on the sphere; the radius is the scale that
    # matters, as for the level sets)
    rw = (r0 ** (k + 1) + f.w[inner] * (k + 1) * r0**k) ** (1 / (k + 1))
    ew = rw - r[inner]
    assert abs(np.median(ew)) < 0.1 and np.percentile(np.abs(ew), 95) < 0.5, (np.median(ew),
                                                                           np.percentile(np.abs(ew), 95))


# ------------------------------------------------------------------ 3. apposed slabs, P4 and P8


def p_budgets(ph, oracle: bool):
    """Per-root budgets by construction: plate A, plate B (the oracle: the true distance to the contact;
    else the open thicknesses 12 and 18), and the U bottom (its own open thickness, 14)."""
    wm2 = ph.wm[:, 0, :]
    xs = np.nonzero(wm2[:, 30])[0]
    fa = xs[xs < ph.extra["split"]].max() + 0.5
    fb = xs[xs > ph.extra["split"]].min() - 0.5
    kn = ph.extra["known"]
    ba, bb = (kn - fa, fb - kn) if oracle else (12.0, 18.0)
    fx, fz = ph.foot[..., 0], ph.foot[..., 2]
    bud = np.where(fz < 7.6, 14.0, np.where(fx < ph.extra["split"], ba, bb))
    return bud, (fa * bb + fb * ba) / (ba + bb)


def region_seams(ph, f):
    sm = L.seams(f)
    reg = ph.extra["region"]
    yz = np.round(sm.pos[:, 1:]).astype(int)
    inr = (sm.axis == 0) & reg[yz[:, 0], yz[:, 1]]
    return sm, inr, int(reg.sum())


@pytest.mark.parametrize("name", ["p4", "p8"])
def test_apposed_coverage(name):
    """Every cortex voxel is claimed: all of it is connected to the WM through cortex."""
    ph = getattr(PH, name)()
    f = run(ph, p_budgets(ph, True)[0], L.LENGTH)
    cov = L.coverage(f, ph.wm, ph.fluid)
    assert cov["fraction"] == 1.0, cov


@pytest.mark.parametrize("name", ["p4", "p8"])
@pytest.mark.parametrize("oracle", [True, False])
@pytest.mark.parametrize("mode", [L.LENGTH, L.VOLUME])
def test_apposed_seam(name, oracle, mode):
    """The seam lies where tau_A = tau_B predicts: at the exact contact with the oracle budgets (tau = 1
    on both sides), and at the rule's prediction with the open thicknesses (P4 +2.4, P8 -0.2 voxels)."""
    ph = getattr(PH, name)()
    bud, pred = p_budgets(ph, oracle)
    f = run(ph, bud, mode)
    sm, inr, nlines = region_seams(ph, f)
    assert inr.sum() == nlines  # one seam pair on every line through the filled patch
    # the prediction assumes a = 1 along both paths. At the patch's top and bottom rows (the lips of the
    # fluid gap) the far bank's voxel just outside the patch sees the near plate through the patch (the
    # distance-through-tissue normal tilts ~18 deg), so the volume form reads a != 1 there and moves the
    # seam (P8: 0.16 voxel); that is the volume form's answer for that direction field
    a = f.a.ravel()
    flat = inr & (np.abs(a[sm.p] - 1) < 1e-6) & (np.abs(a[sm.q] - 1) < 1e-6)
    lips = np.isin(np.round(sm.pos[:, 2]).astype(int), [36, 45])
    if mode == L.LENGTH:
        flat = inr  # a does not enter the rate
    assert np.all(flat | ~inr | lips)
    np.testing.assert_allclose(sm.pos[flat, 0], pred, atol=0.01)
    np.testing.assert_allclose(sm.tau_a[inr], sm.tau_b[inr], atol=1e-6)
    if oracle:
        np.testing.assert_allclose(pred, ph.extra["known"], atol=1e-9)
        np.testing.assert_allclose(sm.tau_a[flat], 1.0, atol=0.01)


def test_tops_do_not_depend_on_budgets():
    """The evidence is the same whatever the budgets (test plan §3): P4's tops under the oracle and the
    open-thickness budgets agree in s and w (P4's seam is in the filled patch, away from every top)."""
    ph = PH.p4()
    tps = []
    for oracle in (True, False):
        f = run(ph, p_budgets(ph, oracle)[0], L.VOLUME)
        tps.append(L.tops(f, ph.fluid))
    a, b = tps
    np.testing.assert_array_equal(a.idx, b.idx)
    np.testing.assert_allclose(a.s, b.s, atol=1e-9)
    np.testing.assert_allclose(a.w, b.w, atol=1e-9)


# ------------------------------------------------------------------ 4. wedge


def wedge_reference(ph):
    """Each bank alone over the whole domain with its own wall normal: tau_A, tau_B."""
    e = ph.extra
    sideA = ph.foot[..., 0] < e["cx"]
    out = []
    for sel, nv, mu in ((sideA, e["nA"], e["mu_a"]), (~sideA, e["nB"], e["mu_b"])):
        n3 = np.broadcast_to(np.array([nv[0], 0, nv[1]]), ph.wm.shape + (3,))
        f = L.march(ph.cortex, ph.wm, ph.band & sel, ph.phi, ph.foot, ph.normal, np.full(ph.wm.shape, mu), n3)
        out.append(f.tau)
    return out


def seam_vs_reference(ph, f, tA, tB):
    sm = L.seams(f)
    errs = []
    for x, y, z in sm.pos[sm.axis == 0]:
        yi, zi = int(round(y)), int(round(z))
        dd = tA[:, yi, zi] - tB[:, yi, zi]
        xs = np.flatnonzero(np.isfinite(dd) & ph.cortex[:, yi, zi])
        ch = [c for c in np.flatnonzero(np.diff(np.sign(dd[xs])) != 0) if xs[c + 1] == xs[c] + 1]
        if ch:
            a0, a1 = dd[xs[ch[0]]], dd[xs[ch[0]] + 1]
            errs.append(x - (xs[ch[0]] + a0 / (a0 - a1)))
    return np.array(errs)


@pytest.mark.parametrize("theta", [20, 45, 90])
def test_wedge_own_normals(theta):
    """Version A (each bank's layers follow its own wall up to the contact): the seam lies at
    tau_A = tau_B of the two banks run alone, at every angle. (Versions B and C, where the normal flips
    at the bisector, pin the seam to the flip at large angles: levelset_synth.py reports them.)"""
    ph = PH.wedge(theta)
    e = ph.extra
    bud = np.where(ph.foot[..., 0] < e["cx"], e["mu_a"], e["mu_b"])
    n, coh = PH.wedge_normals(ph, "A")
    f = L.march(ph.cortex, ph.wm, ph.band, ph.phi, ph.foot, ph.normal, bud, n, coh)
    tA, tB = wedge_reference(ph)
    err = seam_vs_reference(ph, f, tA, tB)
    assert err.size > 0.9 * np.sum(L.seams(f).axis == 0)
    assert np.abs(err).max() < 0.05, np.abs(err).max()


@pytest.mark.xfail(strict=True, reason="a known limitation, reported: where coherence is 0 and two fronts "
                   "meet at 90 degrees, the blend turns the direction in steps under 90 degrees from one "
                   "front's to the other's, so neighbours agree everywhere, no seam forms and the fronts merge")
def test_wedge_seam_at_90():
    """Version C at 90 degrees: with coherence 0 at the bisector the direction is the front's own normal
    there. The plan expects the budgets to decide a seam inside that band. What happens: directions run
    (0.71, 0.71) -> (0.28, 0.96) -> (-0.66, 0.75) across the band, each step agreeing (d . d > 0), so the
    sign agreement never separates the sides."""
    ph = PH.wedge(90)
    e = ph.extra
    bud = np.where(ph.foot[..., 0] < e["cx"], e["mu_a"], e["mu_b"])
    n, coh = PH.wedge_normals(ph, "C")
    f = L.march(ph.cortex, ph.wm, ph.band, ph.phi, ph.foot, ph.normal, bud, n, coh)
    sm = L.seams(f)
    k = sm.axis == 0
    assert k.sum() > 0
    zs = np.unique(np.round(sm.pos[k, 2]).astype(int))
    assert zs.min() <= e["z0"] + 3 and zs.size >= 0.6 * e["L"] / np.sqrt(2)


# ------------------------------------------------------------------ 5. U-slab


def test_uslab_fold():
    """R = 6 < t = 16: the front meets itself on the bisector from about the fundus centre (zo) to
    zo + (t - R) / sin 25 deg, with tau < 1 on both sides (the fold signature); R = 22 > t: no seam."""
    ph = PH.u_slab(6.0)
    e = ph.extra
    f = run(ph, 16.0, L.LENGTH)
    sm = L.seams(f)
    assert sm.p.size > 0
    np.testing.assert_allclose(sm.pos[:, 0], e["cx"], atol=0.51)
    assert sm.pos[:, 2].min() <= e["zo"] + 1.5 and sm.pos[:, 2].max() >= e["zmax"] - 1.5
    assert sm.pos[:, 2].max() <= e["zmax"] + 0.5
    assert np.all(sm.tau_a < 1.0) and np.all(sm.tau_b < 1.0)
    np.testing.assert_allclose(sm.tau_a, sm.tau_b, atol=1e-6)
    ph = PH.u_slab(22.0)
    assert L.seams(run(ph, 16.0, L.LENGTH)).p.size == 0


# ------------------------------------------------------------------ 6. omega


@pytest.mark.parametrize("mode", [L.LENGTH, L.VOLUME])
def test_omega_no_bending(mode):
    """A step in budget (12 left, 18 right) does not bend paths: near the step on the chamber floor
    every root is the oracle foot (paths along the distance gradient), the same as with equal
    budgets; the neck seam lies at the known contact (length form exact)."""
    ph = PH.omega()
    e = ph.extra
    fx = ph.foot[..., 0]
    f_step = run(ph, np.where(fx < e["cx"], e["tl"], e["tr"]), mode)
    f_eq = run(ph, 15.0, mode)
    floor = ph.cortex & L.claimed(f_step) & (np.abs(fx - e["cx"]) < 12) & (ph.foot[..., 2] < 20)
    assert floor.sum() > 500
    dR = np.linalg.norm(f_step.R - ph.foot, axis=-1)[floor]
    assert np.percentile(dR, 99) < 0.5, np.percentile(dR, 99)
    np.testing.assert_allclose(f_step.R[floor], f_eq.R[floor], atol=1e-9)
    sm = L.seams(f_step)
    yz = np.round(sm.pos[:, 1:]).astype(int)
    inr = (sm.axis == 0) & e["region"][yz[:, 0], yz[:, 1]]
    assert inr.sum() == e["region"].sum()
    err = sm.pos[inr, 0] - e["known"]
    if mode == L.LENGTH:
        np.testing.assert_allclose(err, 0.0, atol=0.01)
    else:
        assert abs(np.median(err)) < 0.05


# ------------------------------------------------------------------ 7. thin blade


def test_blade_sides_do_not_mix():
    ph = PH.blade()
    e = ph.extra
    f = run(ph, 12.0, L.LENGTH)
    m = L.claimed(f) & ph.cortex
    assert m.sum() == ph.cortex.sum()
    sides = m & (ph.foot[..., 2] < e["ztip"] - 0.6)
    assert np.all(np.sign(f.R[sides][:, 0] - e["mid"]) == np.sign(ph.foot[sides][:, 0] - e["mid"]))
    assert np.linalg.norm(f.R - ph.foot, axis=-1)[m].max() < 0.5


# ------------------------------------------------------------------ §2. carried budget


def test_carried_budget_matches_lookup():
    """A budget varying over the white surface, carried with R, equals the budget looked up at the
    carried R (sphere, b = 16 + 4 cos(polar angle) of the root)."""
    ph = PH.sphere()
    c = ph.extra["c"]
    r0 = ph.extra["r0"]

    def bfun(R):
        return 16.0 + 4.0 * (R[..., 2] - c) / r0

    f = run(ph, bfun(ph.foot), L.LENGTH)
    m = L.claimed(f) & ph.cortex
    diff = f.b[m] - bfun(f.R[m])
    assert np.abs(diff).max() < 0.05, np.abs(diff).max()


@pytest.mark.parametrize("normal", [(1.0, 1.0, 1.0), (1.0, 0.5, 0.2)])
@pytest.mark.parametrize("band_wm", [1.0, 3.0])
def test_band_depth_flat_slab(normal, band_wm):
    """The start band must reach as deep as the update looks back (test plan §3 step 1). The update
    traces back h <= sqrt 3 to the upwind cell's face; from the first voxel outside the cortex band
    (phi >= 1) that reaches phi >= 1 - sqrt 3, so a band one voxel into the WM suffices: s and R are
    exact on an oblique plane (the cube's diagonal included) with band depths 1 and 3."""
    ph = PH.plane3d(normal, band_wm=band_wm)
    f = run(ph, 12.0, L.LENGTH)
    X = np.stack(np.meshgrid(*[np.arange(n) for n in ph.wm.shape], indexing="ij"), -1)
    inner = ph.cortex & np.all(np.abs(X - 20) < 12, -1)
    assert L.claimed(f)[inner].all()
    np.testing.assert_allclose(f.s[inner], ph.phi[inner], atol=1e-9)
    np.testing.assert_allclose(f.R[inner], ph.foot[inner], atol=1e-9)


# ------------------------------------------------------------------ the mesh field


def small_mesh():
    import mesh_smoothing as MS

    import levelset_mesh as LM

    X = np.stack(np.meshgrid(*[np.arange(28)] * 3, indexing="ij"), -1).astype(float)
    wm = np.linalg.norm(X - 13.6, axis=-1) < 9.0
    v0, f = MS.contour(wm, 0.0)
    v = MS.twice(MS.relax(v0, f), f, 2.0)
    return LM.Mesh(v, f, MS.smooth_normals(v, f), np.array([0.16, 0.16, 0.16])), LM


@pytest.mark.parametrize("alpha", [1, 2])
@pytest.mark.parametrize("trend", [True, False])
def test_mesh_field_cg_matches_direct(alpha, trend):
    """The conjugate-gradient posterior mean equals the direct sparse solve."""
    mesh, LM = small_mesh()
    rng = np.random.default_rng(3)
    n = len(mesh.verts)
    H = rng.normal(0, 0.3, n)
    fld = LM.MeshField(mesh, H, alpha=alpha)
    y = np.full(n, np.nan)
    o = rng.choice(n, n // 5, replace=False)
    y[o] = np.log(2.5) + 0.2 * np.sin(mesh.mm[o, 0] * 3) + 0.05 * rng.standard_normal(o.size)
    a = fld.fit(y, trend=trend)
    b = fld.fit(y, trend=trend, direct=True)
    np.testing.assert_allclose(a["mu"], b["mu"], atol=1e-5)


def test_mesh_field_length():
    """The prior's correlation length is about ELL_Z (2 mm), whatever the vertex spacing: the posterior
    with one observation decays from it like the Matern (nu = 1) correlation, r/l K1(r/l): at 2 mm,
    about 0.6 of its peak (Lindgren's range sqrt(8) l is where it falls to about 0.14)."""
    import mesh_smoothing as MS

    import levelset_mesh as LM

    wm = np.zeros((100, 100, 10), bool)
    wm[:, :, :5] = True  # a flat sheet 16 mm across (a small ball is closed within one length)
    v0, f = MS.contour(wm, 0.0)
    mesh = LM.Mesh(v0, f, MS.smooth_normals(v0, f), np.array([0.16, 0.16, 0.16]))
    n = len(mesh.verts)
    fld = LM.MeshField(mesh, np.zeros(n), alpha=2)
    y = np.full(n, np.nan)
    i0 = int(np.argmin(np.linalg.norm(mesh.verts[:, :2] - 49.5, axis=1)))
    y[i0] = np.log(2.5) + 1.0
    r = fld.fit(y, trend=False)
    eff = r["mu"] - r["m"]
    G = LM.edge_graph(mesh)
    d = LM.geodesic(G, np.array([i0]))[0]
    from scipy.special import k1

    near = (d > 1.8) & (d < 2.2)
    ratio = np.median(eff[near]) / eff[i0]
    expect = 1.0 * k1(1.0)  # r / l = 1
    assert abs(ratio - expect) < 0.15, (ratio, expect)


# ------------------------------------------------------------------ §1: the rate never changes the path


@pytest.mark.parametrize("mode", [L.LENGTH, L.VOLUME])
@pytest.mark.parametrize("noise", [0.15, 0.4])
def test_low_coherence_paths_do_not_depend_on_budgets(mode, noise):
    """The rate never changes the path (test plan §1, §3 step 2). On the sphere with a noisy data normal
    (the true one plus Gaussian noise of SD `noise` per component) and coherence 0.5 everywhere, two
    budget fields (uniform, and varying 2:1 over the surface) give identical d, R, s, w and tops,
    except at the voxels whose side the budgets changed (moved: phase 2's owner differs from phase 1's;
    with noise 0.4 the direction field has small false seams, d_x . d_y < 0, and the budgets decide
    them: 146 voxels in the length form, 1,950 (2.8 %) in the volume form, whose noisy a makes more of
    those competitions close). Two causes of a dependence were found and fixed: blending toward the gradient of tau, and
    ordering the geometry by tau (a voxel's upwind support then depended on which roots were faster)."""
    ph = PH.sphere()
    c = ph.extra["c"]
    rng = np.random.default_rng(5)
    n = ph.n + rng.normal(0, noise, ph.n.shape)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    coh = np.full(ph.wm.shape, 0.5)
    b1 = np.full(ph.wm.shape, 16.0)
    b2 = 16.0 + 5.0 * (ph.foot[..., 2] - c) / ph.extra["r0"]
    f1 = L.march(ph.cortex, ph.wm, ph.band, ph.phi, ph.foot, ph.normal, b1, n, coh, mode=mode)
    f2 = L.march(ph.cortex, ph.wm, ph.band, ph.phi, ph.foot, ph.normal, b2, n, coh, mode=mode)
    m = L.claimed(f1) & L.claimed(f2) & ph.cortex
    assert m.sum() > 0.99 * ph.cortex.sum()
    assert np.mean(f1.blend[m] > 0) > 0.9
    moved = f1.moved | f2.moved
    if noise <= 0.15:
        assert not moved.any()
    k = m & ~moved
    for name in ("d", "R", "s", "w"):
        np.testing.assert_allclose(getattr(f1, name)[k], getattr(f2, name)[k], atol=1e-9, err_msg=name)
    t1, t2 = L.tops(f1, ph.fluid), L.tops(f2, ph.fluid)
    keep1 = ~moved.ravel()[t1.idx]
    keep2 = ~moved.ravel()[t2.idx]
    np.testing.assert_array_equal(t1.idx[keep1], t2.idx[keep2])
    np.testing.assert_allclose(t1.s[keep1], t2.s[keep2], atol=1e-9)
    np.testing.assert_allclose(t1.w[keep1], t2.w[keep2], atol=1e-9)
