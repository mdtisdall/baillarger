"""Exact-answer tests for mesh_smoothing.py.

Run: nix develop --command uv run pytest prototypes/scripts/test_mesh_smoothing.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import mesh_smoothing as M  # noqa: E402

from baillarger.fronts import march, riemann_from_direction  # noqa: E402
from baillarger.mesh import marching_tetrahedra  # noqa: E402


# ---------------------------------------------------------------- phantoms


def test_ball_truth_exact():
    ph = M.ball(R=6.0, T=8.0)
    c = ph.p["c"]
    X = c + np.array([[9.0, 0, 0], [0, -3.0, 4.0]])
    tr = ph.truth(X)
    np.testing.assert_allclose(tr.d, [3.0, -1.0])
    np.testing.assert_allclose(tr.foot, c + np.array([[6.0, 0, 0], [0, -3.6, 4.8]]))
    np.testing.assert_allclose(tr.k1, 1 / 6)


def test_graph_peak_curvature_and_orthogonality():
    A, lam = 2.0, 40.0
    ph = M.Phantom("g", (10, 10, 10), 5.0, "graph",
                   dict(Q=np.eye(3), o=np.zeros(3), z0=0.0, A=A, lam=lam))
    w = 2 * np.pi / lam
    tr = ph.truth(np.array([[0.0, 0.0, 7.0]]))
    np.testing.assert_allclose(tr.d, 5.0, atol=1e-12)
    np.testing.assert_allclose([tr.k1[0], tr.k2[0]], [A * w * w] * 2, rtol=1e-12)
    rng = np.random.default_rng(0)
    X = rng.uniform(-30, 30, (200, 3)) * [1, 1, 0.3]
    tr = ph.truth(X)
    r = X - tr.foot
    # the residual is along the normal, and the foot is on the surface
    np.testing.assert_allclose(np.cross(r, tr.n), 0, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(r, axis=1), np.abs(tr.d), atol=1e-9)
    h = ph._h(tr.foot[:, 0], tr.foot[:, 1])[0]
    np.testing.assert_allclose(tr.foot[:, 2], h, atol=1e-9)


def test_blade_offsets():
    ph = M.blade(w=3.0)
    rng = np.random.default_rng(1)
    X = ph.p["o"] + rng.uniform(-12, 12, (300, 3))
    tr = ph.truth(X)
    np.testing.assert_allclose(np.linalg.norm(X - tr.foot, axis=1), np.abs(tr.d), atol=1e-9)
    # feet lie on the blade's surface
    np.testing.assert_allclose(ph.truth(tr.foot).d, 0, atol=1e-9)
    # the normal points away from the WM: a step out raises d by the step
    np.testing.assert_allclose(ph.truth(tr.foot + 0.01 * tr.n).d, 0.01, atol=1e-6)


def test_blade_offset_curvature_against_circle_fit():
    # a short arc of the + side's cross-section through its foot, fitted by a circle
    ph = M.blade(w=4.0, a=3.0, lam=30.0, deg=0.0)
    Q, o = ph.p["Q"], ph.p["o"]
    y0 = 30.0 / 4  # a crest of the centre curve: convex toward +x
    pts = []
    for dy in np.linspace(-0.5, 0.5, 21):
        L = np.array([20.0, y0 + dy, 0.0])
        tr = ph.truth(o + L @ Q.T)
        pts.append(((tr.foot - o) @ Q)[0])
    P = np.array(pts)[:, :2]
    Aa = np.c_[2 * P, np.ones(len(P))]
    sol = np.linalg.lstsq(Aa, np.sum(P**2, 1), rcond=None)[0]
    R = np.sqrt(sol[2] + sol[0] ** 2 + sol[1] ** 2)
    tr = ph.truth(o + np.array([20.0, y0, 0.0]) @ Q.T)
    np.testing.assert_allclose(tr.k1[0], 1 / R, rtol=1e-3)


# ---------------------------------------------------------------- meshes


def test_contour_outward():
    ph = M.ball(R=6.0, T=4.0)
    wm, _, _ = ph.labels()
    for s in (0.0, 1.0):
        v, f = M.contour(wm, s)
        n = M.vertex_normals(v, f)
        assert np.all(np.sum(n * (v - ph.p["c"]), 1) > 0)


def _plane_mesh():
    g = np.stack(np.meshgrid(*[np.arange(20.0)] * 3, indexing="ij"), -1)
    nrm = np.array([0.3, -0.5, 0.81])
    nrm /= np.linalg.norm(nrm)
    F = g @ nrm
    v, f = marching_tetrahedra(F, 9.3)
    return v, f, nrm


def test_taubin_and_relax_keep_a_plane():
    v, f, nrm = _plane_mesh()
    for x in (M.taubin(v, f, 20), M.twice(v, f, 2.0), M.relax(v, f)):
        np.testing.assert_allclose(x @ nrm, 9.3, atol=1e-9)
    b = M.boundary_vertices(f, len(v))
    np.testing.assert_array_equal(M.twice(v, f, 2.0)[b], v[b])


def test_twice_and_taubin_keep_a_ball_radius_gauss_shrinks():
    ph = M.ball(R=10.0, T=3.0)
    wm, _, _ = ph.labels()
    v, f = M.contour(wm, 1.0)
    c = ph.p["c"]
    r0 = np.linalg.norm(v - c, axis=1).mean()
    n = M.gauss_steps(v, f, 2.0)
    A, deg, _ = M._umbrella(v, f)
    x = v.copy()
    for _ in range(n):
        x = x + 0.5 * ((A @ x) / deg[:, None] - x)
    rl = np.linalg.norm(x - c, axis=1).mean()
    for y in (M.twice(v, f, 2.0), M.taubin(v, f, 30)):
        rt = np.linalg.norm(y - c, axis=1).mean()
        assert abs(rt - r0) < 0.1 * abs(rl - r0)


def test_twice_never_amplifies():
    # the response of 2G - G^2 lies in [0, 1] for every eigenvector of the
    # (symmetrized) umbrella operator: no mode grows
    ph = M.ball(R=5.0, T=3.0)
    wm, _, _ = ph.labels()
    v, f = M.contour(wm, 0.0)
    A, deg, _ = M._umbrella(v, f)
    Ds = np.diag(1 / np.sqrt(deg))
    W = Ds @ A.toarray() @ Ds  # similar to D^-1 A
    g = np.linalg.eigvalsh(0.5 * np.eye(len(v)) + 0.5 * W)
    n = M.gauss_steps(v, f, 1.5)
    resp = 1 - (1 - g**n) ** 2
    assert g.min() >= -1e-12 and resp.max() <= 1 + 1e-12 and resp.min() >= -1e-12


def test_graph_truth_far_above_troughs():
    # beyond a trough's focal distance Newton from the point's own (u, v) diverged
    ph = M.wavy()
    rng = np.random.default_rng(4)
    X = ph.p["o"] + rng.uniform(-30, 30, (300, 3)) @ ph.p["Q"].T + 25 * ph.p["Q"][:, 2]
    tr = ph.truth(X)
    uu, vv = np.meshgrid(np.arange(-90, 90, 0.05), np.arange(-90, 90, 0.05), indexing="ij")
    S = np.c_[uu.ravel(), vv.ravel(), ph._h(uu.ravel(), vv.ravel())[0]] @ ph.p["Q"].T + ph.p["o"]
    from scipy.spatial import cKDTree
    dbf, _ = cKDTree(S).query(X)
    np.testing.assert_allclose(np.abs(tr.d), dbf, atol=2e-3)


def test_closest_on_mesh_projects_onto_a_plane():
    v, f, nrm = _plane_mesh()
    X = np.array([[10.0, 10.0, 3.0], [8.0, 9.0, 14.0]])
    P, _, n = M.closest_on_mesh(v, f, X)
    np.testing.assert_allclose(P, X - (X @ nrm - 9.3)[:, None] * nrm, atol=1e-9)
    np.testing.assert_allclose(n, np.broadcast_to(nrm, n.shape) * np.sign(n @ nrm)[:, None], atol=1e-9)


def test_band_seeds_metric_distance_brute_force():
    rng = np.random.default_rng(2)
    n = np.array([0.0, 0.0, 1.0])
    for _ in range(5):
        Mt = riemann_from_direction(rng.normal(size=3), 3.0)
        D = np.linalg.inv(Mt)
        x = np.array([0.1, -0.2, 0.7])
        t, s, foot = M.band_seeds(x[None], np.zeros((1, 3)), n[None], D[None])
        uv = np.stack(np.meshgrid(np.linspace(-3, 3, 1201), np.linspace(-3, 3, 1201)), -1).reshape(-1, 2)
        Y = np.c_[uv, np.zeros(len(uv))]
        r = x - Y
        best = np.sqrt(np.einsum("ni,ij,nj->n", r, Mt, r)).min()
        np.testing.assert_allclose(t[0], best, rtol=1e-4)
        assert abs(foot[0, 2]) < 1e-12
        r0 = x - foot[0]
        np.testing.assert_allclose(np.sqrt(r0 @ Mt @ r0), t[0], rtol=1e-12)


# ---------------------------------------------------------------- the march


def test_march_band_matches_the_package_from_point_seeds():
    rng = np.random.default_rng(3)
    shape = (14, 13, 12)
    dirs = rng.normal(size=shape + (3,))
    Mt = riemann_from_direction(dirs, 2.5)
    lab = -np.ones(shape, np.int64)
    lab[2, 3, 4], lab[10, 9, 7], lab[6, 11, 2] = 0, 1, 2
    mask = np.ones(shape, bool)
    mask[7, :, 5:9] = False
    ref = march(Mt, lab, mask=mask, spacing=(1.0, 1.1, 0.9))
    seeds = lab >= 0
    k = np.flatnonzero(seeds.ravel())
    pos = np.stack(np.unravel_index(k, shape), -1).astype(float)
    out = M.march_band(Mt, mask, seeds, np.zeros(3), np.zeros(3), pos, lab.ravel()[k],
                       spacing=(1.0, 1.1, 0.9))
    np.testing.assert_array_equal(out.arrival, ref.arrival)
    np.testing.assert_array_equal(out.label, ref.label)
    np.testing.assert_array_equal(np.nan_to_num(out.length, nan=-1), np.nan_to_num(ref.length, nan=-1))


@pytest.mark.parametrize("ratio", [1.0, 3.0])
def test_march_band_exact_from_an_oblique_plane(ratio):
    nrm = np.array([0.36, -0.48, 0.8])
    Q = np.linalg.qr(np.c_[nrm, [1, 0, 0], [0, 1, 0]])[0]
    Q = np.c_[Q[:, 1], Q[:, 2], Q[:, 0]] * np.sign(Q[:, 0] @ nrm)
    Q[:, 2] = nrm
    Q[:, 1] = np.cross(Q[:, 2], Q[:, 0])
    ph = M.Phantom("p", (24, 24, 24), 12.0, "graph", dict(Q=Q, o=np.full(3, 8.0), z0=0.0))
    wm, ctx, tr = ph.labels()
    X = ph.grid()
    D = np.linalg.inv(riemann_from_direction(np.broadcast_to(nrm, X.shape), ratio))
    # the band must reach as deep as the stencil: |e.n| up to 2.45 voxels at ratio 3
    band = 1.0 if ratio == 1.0 else 2.5
    seeds, t, s, r = M.surface_seeds(ctx, wm, X, M.truth_closest(ph), D, band=band)
    # the box faces get exact values too, so every voxel's upwind cone is inside
    face = np.zeros(ctx.shape, bool)
    face[[0, 1, -2, -1]] = face[:, [0, 1, -2, -1]] = face[:, :, [0, 1, -2, -1]] = True
    extra = face & ctx & ~seeds
    seeds2 = seeds | extra
    k_all = np.flatnonzero(seeds2.ravel())
    k_band = np.flatnonzero(seeds.ravel())
    pos = np.searchsorted(k_all, k_band)
    T_, S_, R_ = np.zeros(k_all.size), np.zeros(k_all.size), np.zeros((k_all.size, 3))
    T_[pos], S_[pos], R_[pos] = t, s, r
    k_ex = np.flatnonzero(extra.ravel())
    pe = np.searchsorted(k_all, k_ex)
    dt = tr.d[k_ex]
    T_[pe], S_[pe], R_[pe] = dt, dt, tr.foot[k_ex]
    Mt = riemann_from_direction(np.broadcast_to(nrm, X.shape), ratio)
    out = M.march_band(Mt, ctx, seeds2, T_, S_, R_)
    d = tr.d.reshape(ctx.shape)
    foot = tr.foot.reshape(ctx.shape + (3,))
    # columns whose foot lies inside the box, 2 voxels from its faces (others are
    # reached round the plane's cut edge)
    ok = ctx
    assert np.all(np.isfinite(out.arrival[ok]))
    np.testing.assert_allclose(out.arrival[ok], d[ok], atol=1e-9)
    np.testing.assert_allclose(out.length[ok], d[ok], atol=1e-9)
    np.testing.assert_allclose(out.root[ok], foot[ok], atol=1e-9)


def test_fin_distance_brute_force_outside():
    # outside the WM the fin's distance is the distance to its WM label, at fine resolution
    ph = M.fin(w=2.0, n=40)
    rng = np.random.default_rng(5)
    X = ph.p["o"] + rng.uniform(-10, 10, (200, 3))
    tr = ph.truth(X)
    X, tr_d, foot = X[tr.d > 0.5], tr.d[tr.d > 0.5], tr.foot[tr.d > 0.5]
    np.testing.assert_allclose(np.linalg.norm(X - foot, axis=1), tr_d, atol=1e-9)
    np.testing.assert_allclose(ph.truth(foot).d, 0, atol=1e-6)
    g = ph.grid(4).reshape(-1, 3)
    inside = g[ph.truth(g).d < 0]
    from scipy.spatial import cKDTree
    dbf, _ = cKDTree(inside).query(X)
    assert np.all(dbf >= tr_d - 1e-9) and np.all(dbf <= tr_d + 0.25)


def test_closest_on_mesh_sign_at_staircase_edges():
    # on a binary contour, grid points of a 2x grid sit in the planes of axis-aligned
    # faces; the side must come from the edge / vertex pseudonormal, not one face
    ph = M.ball(R=8.0, T=6.0)
    wm1, _, _ = ph.labels()
    v, f = M.contour(wm1, 0.0)
    G = ph.grid(2).reshape(-1, 3)
    d = ph.truth(G).d
    X = G[np.abs(d) < 3.5]
    P, _, n = M.closest_on_mesh(v, f, X)
    s = np.sum((X - P) * n, 1)
    # brute-force closest distance, and the side from the label (a point is inside
    # the binary contour when its nearest... ) -- here: |s| equals the true mesh distance
    from scipy.spatial import cKDTree
    np.testing.assert_allclose(np.abs(s), np.linalg.norm(X - P, axis=1), atol=1e-9)
    # outside the label's staircase by more than its half-voxel step: positive
    assert np.all(s[ph.truth(X).d > 0.9] > 0) and np.all(s[ph.truth(X).d < -0.9] < 0)


def test_implicit_closest_on_a_ball_and_a_thin_blade():
    # Chopp's iteration lands on the interpolant's zero set, orthogonally
    ph = M.ball(R=8.0, T=6.0)
    wm, _, _ = ph.labels()
    cl = M.implicit_closest(wm, 1.0)
    G = ph.grid(2).reshape(-1, 3)
    X = G[np.abs(ph.truth(G).d) < 2.0][::7]
    P, n = cl(X)
    from scipy import ndimage as ndi
    from scipy.special import ndtri
    F = ndi.gaussian_filter(wm.astype(float), 1.0, truncate=6.0)
    coef = ndi.spline_filter(ndtri(np.clip(F, 1e-9, 1 - 1e-9)), order=3)
    val = ndi.map_coordinates(coef, P.T, order=3, prefilter=False, mode="nearest")
    np.testing.assert_allclose(val, 0.0, atol=1e-6)
    r = X - P
    # x - foot is along the normal (the closest point on the zero set)
    rn = np.linalg.norm(r, axis=1)
    ok = rn > 1e-3
    np.testing.assert_allclose(np.abs(np.sum(r[ok] * n[ok], 1)), rn[ok], rtol=1e-6)
    # sides of a 3-voxel blade: a cortex point's foot is on its own side
    ph = M.blade(w=3.0)
    wm, ctx, tr = ph.labels(2)
    X = ph.grid(2).reshape(-1, 3)[(ctx.ravel()) & (tr.d < 1.5)][::5]
    P, n = M.implicit_closest(ph.labels()[0], 0.5)(X)
    assert np.mean(ph.truth(P).side == ph.truth(X).side) > 0.999


def test_voxel_face_mesh_is_the_label_boundary():
    wm = np.zeros((6, 6, 6), bool)
    wm[2:4, 2:5, 1:3] = True
    v, f = M.voxel_face_mesh(wm)
    # area = surface area of the box 2 x 3 x 2
    P = v[f]
    area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1).sum()
    assert area == 2 * (2 * 3 + 3 * 2 + 2 * 2)
    n = M.vertex_normals(v, f)
    c = np.array([2.5, 3.0, 1.5])
    assert np.all(np.sum(n * (v - c), 1) > 0)
    np.testing.assert_allclose(M.aspect_ratios(np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0]]),
                                               np.array([[0, 1, 2]])), 1.0)


def test_implicit_closest_converges_across_the_band():
    # with a lightly smoothed label the field is flat beyond about a voxel; points
    # there must still reach the zero set (they did not: 55 % stopped short)
    from scipy import ndimage as ndi
    ph = M.wavy()
    wm, _, _ = ph.labels()
    G = ph.grid().reshape(-1, 3)
    X = G[np.abs(ph.truth(G).d) <= 2][::20]
    P, n = M.implicit_closest(wm, 0.5)(X)
    from scipy.special import ndtri
    F = ndi.gaussian_filter(wm.astype(float), 0.5, truncate=6.0)
    coef = ndi.spline_filter(0.5 * ndtri(np.clip(F, 1e-9, 1 - 1e-9)), order=3)
    res = ndi.map_coordinates(coef, P.T, order=3, prefilter=False, mode="nearest")
    assert np.mean(np.abs(res) > 1e-6) < 0.001
