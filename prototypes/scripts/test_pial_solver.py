"""Exact-answer tests for the editor's correction field (pial_edit/solver.py; plan:
reports/Pial editing and edit priority.md, §5 tests 1-4, 6, 8, 9; test 10, timing, is
pial_edit/solver_timing.py).

- 1, no edits: the field is the base exactly; the session's pial points are the base's.
- 2, one edit with small sigma_e: the field at the edit equals it to within sigma_e.
- 3, against a dense solve: on a small flat mesh the posterior mean equals the dense
  formula base + S A^T (A S A^T + sigma_e^2 I)^-1 (y - A base), S = Q^-1 from the dense
  precision. On a large flat mesh, away from its edges, one edit's correction falls off
  as the 2D Matern (nu = 1) correlation kappa r K1(kappa r), at each range, and the prior
  SD at a vertex is sigma0.
- 4, order: edits added one at a time, in any order, or removed and added again, give the
  same field.
- 6, contact on P4: a click on the known contact (x = 25.5 in the fused patch) gives bank
  A's column its top 16 voxels up and bank B's 30 voxels up, both at the click.
- 8, replay: edits saved and replayed onto a perturbed base give the same tops at the
  edited roots.
- 9, free edits: the displaced pial surface passes through the click; far away nothing
  moves.

Run: nix develop --command uv run pytest prototypes/scripts/test_pial_solver.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import k1

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "pial_edit"))
import pial_height as P  # noqa: E402
import solver as S  # noqa: E402

from baillarger import phantoms  # noqa: E402

H = P.H_MM


def flat_mesh(nx: int, ny: int, h: float):
    """A regular grid of nx x ny vertices (spacing h mm) in the plane z = 0, each square
    split along alternating diagonals."""
    X, Y = np.meshgrid(np.arange(nx) * h, np.arange(ny) * h, indexing="ij")
    v = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
    f = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = i * ny + j, (i + 1) * ny + j, (i + 1) * ny + j + 1, i * ny + j + 1
            f += [[a, b, c], [a, c, d]] if (i + j) % 2 == 0 else [[a, b, d], [b, c, d]]
    return v, np.array(f, np.int64)


# ---------------------------------------------------------------- the field (tests 1-4)


def test_cotan_stiffness_exact_on_linear_functions():
    v, f = flat_mesh(9, 7, 0.3)
    K, mass = S.cotan_matrices(v, f)
    np.testing.assert_allclose(K @ np.ones(len(v)), 0, atol=1e-12)
    # a linear function has zero Laplacian at interior vertices
    u = 2.0 * v[:, 0] - 0.7 * v[:, 1]
    inner = (v[:, 0] > 1e-9) & (v[:, 0] < 8 * 0.3 - 1e-9) & (v[:, 1] > 1e-9) & (v[:, 1] < 6 * 0.3 - 1e-9)
    np.testing.assert_allclose((K @ u)[inner], 0, atol=1e-12)
    np.testing.assert_allclose(mass.sum(), (8 * 0.3) * (6 * 0.3))
    assert np.all(np.linalg.eigvalsh(K.toarray()) > -1e-10)


def test_coincident_vertices_are_welded():
    # one vertex duplicated: half its triangles use the copy, and a zero-area triangle
    # joins them (V1V2's 124 such triangles made L singular). Welded, the field equals
    # the clean mesh's, and the two copies get the same value.
    v, f = flat_mesh(21, 21, 0.1)
    c = 10 * 21 + 10
    clean = S.Field(S.Prior(v, f, 1.0, 0.5), np.zeros(len(v)))
    clean.add([c + 3, c + 3, c + 3], [1.0, 0, 0], 0.4)
    v2 = np.vstack([v, v[c]])
    f2 = f.copy()
    uses = np.flatnonzero((f2 == c).any(1))
    for t in uses[: len(uses) // 2]:
        f2[t][f2[t] == c] = len(v)
    f2 = np.vstack([f2, [[c, len(v), c + 1]]])
    pr = S.Prior(v2, f2, 1.0, 0.5)
    assert pr.n_welded == 1
    fd = S.Field(pr, np.zeros(len(v2)))
    fd.add([c + 3, c + 3, c + 3], [1.0, 0, 0], 0.4)
    m = fd.mean()
    np.testing.assert_allclose(m[: len(v)], clean.mean(), atol=1e-12)
    assert m[len(v)] == m[c]


def test_no_edits_is_base():
    v, f = flat_mesh(10, 10, 0.2)
    base = np.random.default_rng(0).normal(size=len(v))
    fd = S.Field(S.Prior(v, f, 1.0, 0.5), base)
    np.testing.assert_array_equal(fd.mean(), base)


def test_one_edit_small_sigma():
    v, f = flat_mesh(31, 31, 0.1)
    base = np.full(len(v), 2.5)
    fd = S.Field(S.Prior(v, f, 1.0, 0.5), base, sd_obs_mm=0.03)
    c = 15 * 31 + 15
    fd.add([c, c, c], [1.0, 0.0, 0.0], 3.1)
    m = fd.mean()
    assert abs(m[c] - 3.1) <= 0.03
    # and it is a correction: base far away
    far = np.linalg.norm(v - v[c], axis=1) > 1.4
    assert np.max(np.abs(m[far] - 2.5)) < 0.1


def test_against_dense():
    v, f = flat_mesh(12, 12, 0.3)
    pr = S.Prior(v, f, 1.0, 0.5)
    rng = np.random.default_rng(1)
    base = rng.normal(size=len(v))
    fd = S.Field(pr, base, sd_obs_mm=0.05)
    A, y = [], []
    for _ in range(4):
        t = rng.integers(len(f))
        w = rng.dirichlet(np.ones(3))
        yy = rng.normal(2.0, 0.3)
        fd.add(f[t], w, yy)
        a = np.zeros(len(v))
        np.add.at(a, f[t], w)
        A.append(a)
        y.append(yy)
    A, y = np.array(A), np.array(y)
    Sig = np.linalg.inv(pr.dense_precision())
    dense = base + Sig @ A.T @ np.linalg.solve(A @ Sig @ A.T + 0.05**2 * np.eye(len(y)), y - A @ base)
    np.testing.assert_allclose(fd.mean(), dense, rtol=1e-8, atol=1e-10)


def _matern_error(rho, n, h):
    """Prior SD at the centre vertex of an n x n flat mesh (spacing h) and the largest
    error of its covariance against sigma0^2 kappa r K1(kappa r), over 0.1-1.5 ranges,
    in units of sigma0^2."""
    v, f = flat_mesh(n, n, h)
    pr = S.Prior(v, f, rho, 0.5)
    c = (n // 2) * n + n // 2
    e = np.zeros(len(v))
    e[c] = 1.0
    col = pr.cov_times(e)
    r = np.linalg.norm(v - v[c], axis=1)
    sel = (r > 0.1 * rho) & (r < 1.5 * rho)
    kr = pr.kappa * r[sel]
    return np.sqrt(col[c]), np.max(np.abs(col[sel] / 0.25 - kr * k1(kr)))


@pytest.mark.parametrize("rho", [1.0, 2.0])
def test_matern_shape_and_sd(rho):
    # spacing 0.08 mm, the white meshes' edge length (half a voxel); 16 mm square: at
    # least 4 ranges from the centre to an edge
    sd, err = _matern_error(rho, 201, 0.08)
    print(f"\nrho {rho}: prior SD at the centre {sd:.4f} (0.5); covariance error max {err:.4f} sigma0^2")
    assert abs(sd - 0.5) / 0.5 < 0.05
    assert err < 0.02


def test_matern_error_is_discretisation():
    # halving the spacing at rho = 1 mm (16 mm square both times) shrinks the error
    sd0, e0 = _matern_error(1.0, 101, 0.16)
    sd1, e1 = _matern_error(1.0, 201, 0.08)
    print(f"\nrho 1: error {e0:.4f} -> {e1:.4f}; SD {sd0:.4f} -> {sd1:.4f}")
    assert e1 < e0 / 1.5
    assert abs(sd1 - 0.5) < abs(sd0 - 0.5)


def test_order_and_removal():
    v, f = flat_mesh(25, 25, 0.12)
    pr = S.Prior(v, f, 1.0, 0.5)
    rng = np.random.default_rng(2)
    base = rng.normal(2.5, 0.1, len(v))
    edits = [(f[rng.integers(len(f))], rng.dirichlet(np.ones(3)), rng.normal(2.5, 0.3)) for _ in range(5)]
    a = S.Field(pr, base)
    steps = []
    for e in edits:
        a.add(*e)
        steps.append(a.mean())
    b = S.Field(pr, base)
    for e in reversed(edits):
        b.add(*e)
    c = S.Field(pr, base)
    keys = [c.add(*e) for e in edits]
    c.remove(keys[2])
    c.add(*edits[2])
    np.testing.assert_allclose(b.mean(), steps[-1], atol=1e-10)
    np.testing.assert_allclose(c.mean(), steps[-1], atol=1e-10)


# ---------------------------------------------------------------- the session on P4 (tests 1, 6, 8, 9)


@pytest.fixture(scope="module")
def p4():
    rim, _, e = phantoms.p4_kissing_dark_line()
    wm, ctx, fluid = rim == 2, rim == 3, rim == 1
    xdir = np.zeros(wm.shape + (3,))
    xdir[..., 0] = 1.0
    v, f, n = P.white_surface(wm)
    g = P.march_from_mesh(v, f, wm, ctx, fluid, xdir, 3.0)
    cols = P.trace_columns(v, n, g, P.characteristic_field(g), np.where(ctx[..., None], xdir, 0.0))
    t_base, _ = P.top_label(cols, fluid)
    return dict(v=v, f=f, n=n, g=g, cols=cols, t_base=t_base, fluid=fluid)


def session(c, t_base=None):
    return S.Session(c["v"], c["f"], c["n"], c["g"], c["cols"],
                     c["t_base"] if t_base is None else t_base)


def test_session_no_edits_is_base(p4):
    s = session(p4)
    np.testing.assert_array_equal(s.thickness(), p4["t_base"])
    ref = P.point_at(p4["cols"], p4["t_base"])
    ok = np.isfinite(ref).all(1)
    np.testing.assert_array_equal(s.positions()[ok], ref[ok])


def test_top_edit_open_bank(p4):
    s = session(p4)
    x = np.array([20.0, 5.3, 30.2])  # bank A outside the patch; its WM surface is x = 9.5
    e = s.add_top(x)
    assert not e.fallback and e.miss_mm[0] < 0.1 * H
    idx, w = s._root_obs(np.asarray(e.roots[0]))
    t = np.dot(s.thickness()[idx], w)
    print(f"\ntop edit: t at the root {t:.3f} voxels (exact 10.5); miss {e.miss_mm[0] / H:.3f} voxel")
    assert abs(t - 10.5) < 0.05
    p = np.tensordot(w, s.positions()[idx], axes=(0, 0))
    assert np.linalg.norm(p - x) < 0.1 + 0.6  # the blended vertices' own spread (half a triangle)
    col = s._column_of(idx, w)
    assert np.linalg.norm(P.point_at(_one(col), np.array([t]))[0] - x) < 0.1


def _one(pts):
    """A Columns of one polyline (for point_at)."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(np.nan_to_num(seg))])
    n = int(np.isfinite(pts).all(1).sum())
    s[n:] = np.nan
    return P.Columns(pts[None].astype(np.float32), np.array([n]), s[None], np.array([np.nan]),
                     np.array([np.nan]), np.array([s[n - 1]]), np.array([0]), np.array([False]))


def test_contact_edit_p4(p4):
    s = session(p4)
    x = np.array([25.5, 19.6, 40.4])  # on the known contact, inside the fused patch
    e = s.add_contact(x)
    print(f"\ncontact: s {np.array(e.s_mm) / H} voxels (exact 16, 30); miss {np.array(e.miss_mm) / H}")
    assert len(e.roots) == 2
    np.testing.assert_allclose(np.array(e.s_mm) / H, [16.0, 30.0], atol=0.1)
    tt = s.thickness()
    idx, w = s._root_obs(np.asarray(e.roots[0]))
    assert abs(np.dot(tt[idx], w) - 16.0) < 0.05
    r2 = int(np.argmin(np.linalg.norm(s.v - np.asarray(e.roots[1]), axis=1)))
    assert abs(tt[r2] - 30.0) < 0.05
    pB = P.point_at(p4["cols"], tt)[r2]
    assert np.linalg.norm(pB - x) < e.miss_mm[1] / H + 0.1
    pA = P.point_at(_one(s._column_of(idx, w)), np.array([np.dot(tt[idx], w)]))[0]
    assert np.linalg.norm(pA - x) < 0.1


def test_replay_onto_perturbed_base(p4, tmp_path):
    s = session(p4)
    s.add_top(np.array([20.0, 5.3, 30.2]))
    s.add_contact(np.array([25.5, 19.6, 40.4]))
    path = tmp_path / "edits.json"
    s.save(path)
    v = p4["v"]
    pert = 1.5 * np.sin(v[:, 1] / 6.0) * np.cos(v[:, 2] / 9.0) + 0.3
    s2 = session(p4, p4["t_base"] + pert)
    made = s2.replay(path)
    assert [m.kind for m in made] == [e.kind for e in s.edits]
    t1, t2 = s.thickness(), s2.thickness()
    for a, b in zip(s.edits, made):
        np.testing.assert_allclose(b.s_mm, a.s_mm, atol=1e-9)
        for R in a.roots:
            idx, w = s._root_obs(np.asarray(R))
            assert abs(np.dot(t1[idx], w) - np.dot(t2[idx], w)) < 0.05
    # away from the edits the perturbed base shows through
    far = np.linalg.norm(v - np.array([25.5, 19.6, 40.4]), axis=1) > 40
    far &= np.linalg.norm(v - np.array([20.0, 5.3, 30.2]), axis=1) > 40
    np.testing.assert_allclose((t2 - t1)[far], pert[far], atol=0.05)


def test_free_edit(p4):
    s = session(p4)
    before = s.positions()
    x = np.array([22.6, 5.3, 30.2])  # 1 voxel into the fluid above bank A's top (x = 21.5)
    e = s.add_free(x)
    after = s.positions()
    import mesh_smoothing as M

    Pc, F, _ = M.closest_on_mesh(after, s.f, x[None])
    print(f"\nfree edit: the surface passes {np.linalg.norm(Pc[0] - x):.4f} voxel from the click")
    assert np.linalg.norm(Pc[0] - x) < 0.05
    moved = np.linalg.norm(after - before, axis=1)
    far = np.linalg.norm(before - x, axis=1) > 3 * 2.0 / H  # 3 ranges: correlation < 1e-3
    assert far.sum() > 1000
    assert np.max(moved[far]) < 1e-3 / H
    s.remove(0)
    np.testing.assert_allclose(s.positions(), before, atol=1e-12)


def test_click_on_ridge_does_not_mix_fronts(p4):
    # x = 32.5 is the equal-arrival ridge: the voxels on either side belong to banks A
    # (root x = 9.5) and B (55.5). The carried root must be one of them, not their mean.
    R, s = P.click_column(p4["g"], np.array([[32.5, 19.6, 40.4]]))
    assert min(abs(R[0, 0] - 9.5), abs(R[0, 0] - 55.5)) < 0.1
    assert abs(s[0] - 23.0) < 0.1


def test_contact_edit_on_ridge(p4):
    s = session(p4)
    e = s.add_contact(np.array([32.5, 19.6, 40.4]))
    print(f"\ncontact on the ridge: s {np.array(e.s_mm) / H} voxels (exact 23, 23); miss {np.array(e.miss_mm) / H}")
    assert not e.fallback and len(e.roots) == 2
    np.testing.assert_allclose(np.array(e.s_mm) / H, [23.0, 23.0], atol=0.1)
    assert max(e.miss_mm) / H < 0.2


def test_non_finite_edits_rejected(p4):
    s = session(p4)
    with pytest.raises(ValueError):
        s.add_top(np.array([np.nan, 5.0, 30.0]))
    with pytest.raises(ValueError):
        s.thick.add([0, 0, 0], [1.0, 0, 0], np.nan)
    assert not s.edits and not s.thick.obs


def test_rebuild_depends_only_on_the_list(p4):
    # tops, a contact, a top that falls back (a click in the fluid gap, far from any
    # column's line) and free points, in two orders: the same pial surface
    pts = [("top", np.array([20.0, 5.3, 30.2])), ("free", np.array([22.6, 30.0, 30.2])),
           ("contact", np.array([25.5, 19.6, 40.4])), ("top", np.array([30.0, 5.3, 48.0])),
           ("free", np.array([22.6, 5.3, 20.2]))]
    a = session(p4)
    ea = a.rebuild(pts)
    b = session(p4)
    order = [3, 1, 4, 0, 2]
    eb = b.rebuild([pts[i] for i in order])
    assert ea[3].fallback and eb[0].fallback
    np.testing.assert_allclose(b.positions(), a.positions(), atol=1e-9)
    # rebuilding again in place gives the same again
    a.rebuild(pts)
    np.testing.assert_allclose(a.positions(), b.positions(), atol=1e-9)


# ---------------------------------------------------------------- the surface session (contoured base)


@pytest.fixture(scope="module")
def slit():
    # tissue fills the box except a one-voxel slit at x = 30: the contour has two facing
    # sheets at x = 29.5 and 30.5 (binary contour, then twicing: flat parts stay put)
    tis = np.zeros((60, 40, 40), bool)
    tis[4:56, 4:36, 4:36] = True
    tis[30, 4:36, 4:36] = False
    v, f, n = P.white_surface(tis)
    return v, f, n


def test_surface_no_edits_is_base(slit):
    v, f, n = slit
    s = S.SurfaceSession(v, f, n)
    s.rebuild([])
    np.testing.assert_array_equal(s.positions(), v)


def test_surface_contact_moves_both_sheets(slit):
    v, f, n = slit
    s = S.SurfaceSession(v, f, n)
    x = np.array([30.1, 20.3, 19.7])
    (e,) = s.rebuild([("contact", x)])
    assert len(e.miss_mm) == 2  # both sheets observed
    p = s.positions()
    for sheet in (29.5, 30.5):
        on = np.flatnonzero((np.abs(v[:, 0] - sheet) < 0.05) & (np.abs(v[:, 1] - 20.3) < 0.6) & (np.abs(v[:, 2] - 19.7) < 0.6))
        assert on.size
        d = np.min(np.linalg.norm(p[on] - x, axis=1))
        print(f"\nsheet {sheet}: nearest moved vertex {d:.3f} voxel from the click")
        assert d < 0.6  # the vertex spacing; the surface itself passes through x (next test)


def test_surface_passes_through_points(slit):
    v, f, n = slit
    s = S.SurfaceSession(v, f, n)
    pts = [("contact", np.array([30.1, 20.3, 19.7])), ("free", np.array([29.2, 10.4, 12.6])),
           ("top", np.array([56.4, 20.2, 20.1]))]
    s.rebuild(pts)
    loc = S.Locator(s.positions(), f)
    for kind, x in pts:
        Pc, _, _ = loc.closest(x[None])
        print(f"\n{kind}: surface passes {np.linalg.norm(Pc[0] - x):.4f} voxel from the point")
        assert np.linalg.norm(Pc[0] - x) < 0.05


def test_surface_order_and_free_one_sheet(slit):
    v, f, n = slit
    pts = [("contact", np.array([30.1, 20.3, 19.7])), ("free", np.array([29.2, 10.4, 12.6])),
           ("top", np.array([56.4, 20.2, 20.1]))]
    a = S.SurfaceSession(v, f, n)
    a.rebuild(pts)
    b = S.SurfaceSession(v, f, n)
    b.rebuild(pts[::-1])
    np.testing.assert_allclose(a.positions(), b.positions(), atol=1e-9)
    # a free point near one sheet moves only that sheet
    c = S.SurfaceSession(v, f, n)
    (e,) = c.rebuild([("free", np.array([29.2, 10.4, 12.6]))])
    assert len(e.miss_mm) == 1
    moved = np.linalg.norm(c.positions() - v, axis=1)
    other = (np.abs(v[:, 0] - 30.5) < 0.05) & (np.linalg.norm(v[:, 1:] - [10.4, 12.6], axis=1) < 3)
    assert np.max(moved[other]) < 0.05 * np.max(moved)


def _slit_vertex(v, sheet=29.5, y=20.0, z=20.0):
    return int(np.argmin(np.linalg.norm(v - [sheet, y, z], axis=1)))


def test_drag_puts_vertex_at_target(slit):
    v, f, n = slit
    s = S.SurfaceSession(v, f, n)
    r = _slit_vertex(v)
    target = v[r] + np.array([-1.5, 0.4, 0.0])
    (e,) = s.rebuild([("drag", target, v[r])])
    p = s.positions()
    print(f"\ndrag: vertex {np.linalg.norm(p[r] - target):.4f} voxel from its target")
    assert np.linalg.norm(p[r] - target) < 0.03 / H  # sigma_edit
    assert e.miss_mm == [0.0]


def test_drags_order_and_last_wins(slit):
    v, f, n = slit
    r, q = _slit_vertex(v), _slit_vertex(v, 30.5, 12.0, 25.0)
    t1, t2, t3 = v[r] + [-1.0, 0, 0], v[r] + [-2.0, 0.5, 0], v[q] + [1.2, 0, 0]
    pts = [("drag", t1, v[r]), ("contact", np.array([30.1, 25.3, 12.7])), ("drag", t3, v[q]), ("drag", t2, v[r])]
    a = S.SurfaceSession(v, f, n)
    ea = a.rebuild(pts)
    assert ea[0].miss_mm == [] and ea[3].miss_mm == [0.0]  # the first drag of r is superseded
    assert np.linalg.norm(a.positions()[r] - t2) < 0.03 / H
    # any order of the other points, the drags of r kept in their order: the same surface
    b = S.SurfaceSession(v, f, n)
    b.rebuild([pts[2], pts[0], pts[1], pts[3]])
    np.testing.assert_allclose(b.positions(), a.positions(), atol=1e-9)
    # without the last drag, the earlier one applies again (an undo)
    c = S.SurfaceSession(v, f, n)
    c.rebuild(pts[:3])
    assert np.linalg.norm(c.positions()[r] - t1) < 0.03 / H


def test_server_saves_and_replays_drags(slit, tmp_path):
    sys.path.insert(0, str(Path(__file__).parent / "pial_edit"))
    from editor_server import Editor

    v, f, n = slit
    aff = np.diag([H, H, H, 1.0])
    aff[:3, 3] = [10.0, -20.0, 5.0]
    sess = tmp_path / "slit_session.npz"
    np.savez(sess, kind="surface", white_v=v, white_f=f, v=v, f=f, n=n, near_face=np.zeros(len(v), bool), affine=aff)
    ed = Editor(str(sess), "contour", str(tmp_path / "edits.json"))
    r = _slit_vertex(v)
    target_vox = v[r] + np.array([-1.5, 0.4, 0.0])
    ed.add("drag", ed.s.to_mm(target_vox).tolist(), r)
    ed.add("contact", ed.s.to_mm(np.array([30.1, 25.3, 12.7])).tolist())
    ed.remesh()
    p1 = ed.pos.copy()
    assert np.linalg.norm(p1[r] - target_vox) < 0.03 / H
    ed.save()
    ed2 = Editor(str(sess), "contour", str(tmp_path / "edits.json"))
    ed2.load()
    np.testing.assert_array_equal(ed2.pos, p1)
    ed.undo()  # the contact
    ed.undo()  # the drag
    ed.remesh()
    np.testing.assert_allclose(ed.pos, v, atol=1e-12)


@pytest.fixture()
def slit_editor(slit, tmp_path):
    """The slit phantom as an editor session, with a white surface inside the left block
    (its +x face at x = 23.5)."""
    sys.path.insert(0, str(Path(__file__).parent / "pial_edit"))
    from editor_server import Editor

    v, f, n = slit
    wm = np.zeros((60, 40, 40), bool)
    wm[10:24, 10:30, 10:30] = True
    wv, wf, _ = P.white_surface(wm)
    sess = tmp_path / "slit_session.npz"
    np.savez(sess, kind="surface", white_v=wv, white_f=wf, v=v, f=f, n=n, near_face=np.zeros(len(v), bool),
             affine=np.eye(4))
    ed = Editor(str(sess), "contour", str(tmp_path / "edits.json"))
    ed.remesh()
    return ed, v


def test_crossings_shown_when_not_refused(slit_editor):
    ed, v = slit_editor
    assert ed.crossings["faces"] == 0
    r = _slit_vertex(v)  # on the sheet at x = 29.5, facing +x
    ed.add("drag", (v[r] + [2.0, 0, 0]).tolist(), r)  # through the other sheet at 30.5
    ed.remesh()
    print(f"\nadded without pull-back: {ed.crossings['faces']} crossing triangles in {ed.crossings['n_clusters']} places")
    assert ed.crossings["faces"] > 0 and ed.crossings["at_contact"] == 0


def test_drag_pulled_back_before_the_other_sheet(slit_editor):
    ed, v = slit_editor
    r = _slit_vertex(v)
    head = json.loads(_head(ed.drag(r, (v[r] + [2.0, 0, 0]).tolist())))
    d = head["drag"]
    print(f"\ndrag across the slit: placed at {100 * d['fraction']:.0f} % ({d['would_cross']} crossings refused); "
          f"vertex at x = {ed.pos[r, 0]:.3f}")
    assert 0 < d["fraction"] < 1 and d["would_cross"] > 0
    assert head["crossings"]["faces"] == 0
    assert ed.pos[r, 0] < 30.5
    assert ed.points[-1]["pulled_back"]


def test_drag_pulled_back_before_the_white_surface(slit_editor):
    ed, v = slit_editor
    r = _slit_vertex(v)
    head = json.loads(_head(ed.drag(r, (v[r] + [-8.0, 0, 0]).tolist())))  # through the white face at 23.5
    print(f"\ndrag into the white surface: placed at {100 * head['drag']['fraction']:.0f} %; vertex at x = {ed.pos[r, 0]:.3f}")
    assert head["drag"]["fraction"] < 1 and head["crossings"]["faces"] == 0
    assert ed.pos[r, 0] > 23.5


def test_drag_that_crosses_nothing_is_placed_in_full(slit_editor):
    ed, v = slit_editor
    r = _slit_vertex(v)
    target = v[r] + [-1.5, 0.3, 0]
    head = json.loads(_head(ed.drag(r, target.tolist())))
    assert head["drag"]["fraction"] == 1.0 and head["drag"]["would_cross"] == 0
    assert np.linalg.norm(ed.pos[r] - target) < 0.03 / H


def _head(body: bytes) -> str:
    hl = int(np.frombuffer(body[:4], np.uint32)[0])
    return body[4 : 4 + hl].decode()


def test_touching_by_distance(slit_editor):
    ed, v = slit_editor
    # the slit's sheets are one voxel (0.16 mm) apart: not touching at 0.1 mm, touching at 0.2
    assert ed.crossings["touching_faces"] == 0
    ed.set_touch(0.2)
    ed.remesh()
    print(f"\ntouch 0.2 mm: {ed.crossings['touching_faces']} triangles in {ed.crossings['touching_places']} places")
    assert ed.crossings["touching_faces"] > 1000 and ed.crossings["faces"] == 0


def test_pulled_back_drag_touches(slit_editor):
    ed, v = slit_editor
    r = _slit_vertex(v)
    head = json.loads(_head(ed.drag(r, (v[r] + [2.0, 0, 0]).tolist())))
    assert head["drag"]["touching_pending"]
    c = json.loads(_head(ed.touching()))["crossings"]  # the second phase
    print(f"\npulled-back drag: crossing {c['faces']}, touching {c['touching_faces']} in {c['touching_places']} places")
    assert c["faces"] == 0 and c["touching_faces"] > 0
    assert ed.touch[np.isin(ed.s.f, [r]).any(1)].any()  # at the dragged vertex


def test_contact_touches_at_the_contact(slit_editor):
    ed, v = slit_editor
    ed.add("contact", [30.1, 20.3, 19.7])
    ed.remesh()
    c = ed.crossings
    print(f"\ncontact: crossing {c['faces']} ({c['at_contact']} places at contacts), touching {c['touching_faces']} "
          f"({c['touching_at_contact']} of {c['touching_places']} places at contacts)")
    assert c["touching_faces"] > 0 and c["touching_at_contact"] >= 1
    assert c["n_clusters"] == c["at_contact"]  # any crossing there is at the contact


def test_touching_is_surface_distance(slit_editor):
    # the slit's sheets, after smoothing, are 0.149-0.160 mm apart away from its open edges
    # (x 29.50-29.53 and 30.46-30.50): by surface distance nothing there touches at 0.145 mm
    # and all of it does
    # at 0.165 mm (vertex to vertex distances would read longer wherever the two sheets'
    # vertices are not opposite each other)
    ed, v = slit_editor
    inner = (np.abs(v[ed.s.f].mean(1)[:, 0] - 30.0) < 0.6) & np.all(np.abs(v[ed.s.f].mean(1)[:, 1:] - 20) < 8, 1)
    ed.set_touch(0.145)
    ed.remesh()
    # (within about 3 smoothing widths of the slit's open edges the sheets come to 0.10 mm and do touch)
    assert ed.touch[inner].sum() == 0
    ed.set_touch(0.165)
    ed.remesh()
    share = ed.touch[inner].mean()
    print(f"\ntouch 0.165 mm: {share:.3f} of the slit's inner triangles touching")
    assert share > 0.99


def test_contact_search_and_touching_agree(slit_editor):
    # the other sheet a contact point finds is at the distance the touching test measures
    ed, v = slit_editor
    s = ed.s
    x = np.array([30.1, 20.3, 19.7])
    P0, F, B = s.loc.closest(x[None])
    q, fi, b = s._facing(x, s.fn[F[0]])
    d_contact = np.linalg.norm(q - x)
    d, f2, _, _ = S.nearest_facing(s.v, s.f, s.fn, s.loc.table, x[None], s.fn[F[0]][None], 1.0)
    assert f2[0] == fi and abs(d[0] - d_contact) < 1e-12
    assert abs(d_contact - 0.6) < 0.02  # x is nearest the sheet at 30.5; the facing one is at 29.5


def test_fast_drags_equal_a_full_remesh(slit_editor):
    ed, v = slit_editor
    ed.add("contact", [30.1, 26.3, 12.7])
    ed.remesh()
    r, q = _slit_vertex(v), _slit_vertex(v, 29.5, 12.0, 25.0)
    heads = []
    for k, (vert, d) in enumerate(((r, [-1.0, 0.3, 0]), (q, [2.0, 0, 0]), (r, [-1.6, -0.2, 0]))):  # the second is pulled back
        heads.append(json.loads(_head(ed.drag(vert, (v[vert] + d).tolist()))))
        if k != 1:  # after the second drag the page's touching request is skipped: the region carries over
            ed.touching()
    assert all(h["sparse"] for h in heads)
    assert heads[1]["drag"]["fraction"] < 1
    assert heads[2]["drag"]["moved"] < len(v)
    assert ed.points[0]["kind"] == "contact" and ed.points[1]["sides"] == 0  # r's first drag superseded
    fast = (ed.pos.copy(), ed.cross.copy(), ed.touch.copy())
    print(f"\nfast drags: crossing places {ed.crossings['n_clusters']}, at the contact {ed.crossings['at_contact']}; "
          f"moved {[h['drag']['moved'] for h in heads]} vertices; checked "
          f"{[h['state']['timings'].get('checked_vertices') for h in heads]}; crossing {fast[1].sum()}, touching {fast[2].sum()}")
    # any crossing is at the contact point, where the two sheets are pulled together
    assert ed.crossings["n_clusters"] == ed.crossings["at_contact"]
    ed.remesh()  # everything again from the list
    np.testing.assert_allclose(ed.pos, fast[0], atol=1e-9)
    np.testing.assert_array_equal(ed.cross, fast[1])
    np.testing.assert_array_equal(ed.touch, fast[2])


def test_refused_drag_names_the_contacts_that_hold_it(slit_editor):
    # pulling one sheet away from the other is placed in full, unless contact points
    # beside it pin the two sheets together: then the drag is refused and names them
    ed, v = slit_editor
    b = _slit_vertex(v, 30.5)  # the sheet at x = 30.5, facing -x
    ed.add("contact", [30.1, 20.0, 6.0])  # point 0: 14 voxels (2.2 mm) away, too far to hold it
    ed.add("contact", [30.45, 21.0, 20.0])  # point 1: one voxel away
    ed.remesh()
    head = json.loads(_head(ed.drag(b, (ed.pos[b] + [1.5, 0, 0]).tolist())))
    d = head["drag"]
    print(f"\npull apart beside a contact: placed at {100 * d['fraction']:.0f} %, held by {d['held_by']}")
    assert d["fraction"] < 1 and [h["i"] for h in d["held_by"]] == [1]
    assert ed.points[-1]["held_by"] == d["held_by"]
    fast = ed.pos.copy()
    ed.remesh()  # the search put every contact point back
    np.testing.assert_allclose(ed.pos, fast, atol=1e-9)
    ed.add("contact", [30.1, 30.0, 20.0])  # point 3: 9 voxels (1.6 mm) away, holds it alone too
    ed.remesh()
    held = json.loads(_head(ed.drag(b, (ed.pos[b] + [1.5, 0, 0]).tolist())))["drag"]["held_by"]
    print(f"with a second holder: {held}")
    assert [h["i"] for h in held] == [1, 3]
    for i in (3, 1):
        ed.delete(i)
    ed.remesh()
    head = json.loads(_head(ed.drag(b, (ed.pos[b] + [1.5, 0, 0]).tolist())))
    assert head["drag"]["fraction"] == 1.0 and head["drag"]["held_by"] == []


def test_drag_beside_an_existing_crossing_is_placed(slit_editor):
    # a crossing that only shifts at its edge is not a new one: pulling sheet A away from
    # the slit 1-2 mm from an existing crossing is placed; counting every triangle that
    # did not cross before (PLACE_MM = 0) refused it at a few per cent
    ed, v = slit_editor
    a = _slit_vertex(v)
    ed.add("drag", (v[a] + [2.0, 0, 0]).tolist(), a)  # through sheet B: a crossing
    ed.remesh()
    assert ed.crossings["faces"] > 0
    q = _slit_vertex(v, 29.5, 28.0, 20.0)  # 8 voxels (1.3 mm) from it
    fractions = {}
    for place in (0.0, ed.PLACE_MM):
        ed.PLACE_MM = place
        head = json.loads(_head(ed.drag(q, (ed.pos[q] + [-0.5, 0, 0]).tolist())))
        fractions[place] = head["drag"]["fraction"]
        ed.undo()
        ed.remesh()
    print(f"\npull away beside a crossing: placed at {fractions}")
    assert fractions[0.0] < 0.2 and fractions[0.5] == 1.0


def test_refused_drag_says_where(slit_editor):
    ed, v = slit_editor
    r = _slit_vertex(v)
    d = json.loads(_head(ed.drag(r, (v[r] + [2.0, 0, 0]).tolist())))["drag"]
    assert d["fraction"] < 1 and d["blocked"]["against"] == "pial surface"
    assert abs(d["blocked"]["mm"][0] - 30.5) < 0.3 and d["blocked"]["away_mm"] < 0.5
    d = json.loads(_head(ed.drag(r, (v[r] + [-8.0, 0, 0]).tolist())))["drag"]  # into the white surface at 23.5
    assert d["blocked"]["against"] == "white surface"


def test_push_moves_the_other_sheet_ahead(slit_editor, tmp_path):
    ed, v = slit_editor
    a, b = _slit_vertex(v), _slit_vertex(v, 30.5)
    head = json.loads(_head(ed.drag(a, (v[a] + [2.0, 0, 0]).tolist(), push=True)))
    ed.touching()
    d = head["drag"]
    print(f"\npush A through B: placed at {d['fraction']}, {d['pushes']} pushes; A at x {ed.pos[a, 0]:.3f}, "
          f"B at x {ed.pos[b, 0]:.3f}; crossing {ed.crossings['faces']}")
    assert d["fraction"] == 1.0 and d["pushes"] > 0 and d["blocked"] is None
    assert ed.crossings["faces"] == 0
    assert abs(ed.pos[a, 0] - (v[a, 0] + 2.0)) < 0.03 / H
    assert ed.pos[b, 0] > ed.pos[a, 0]  # B was pushed ahead of A
    assert len(ed.points[-1]["pushes"]) == d["pushes"]
    fast = ed.pos.copy()
    ed.remesh()  # the pushes are saved with the drag
    np.testing.assert_allclose(ed.pos, fast, atol=1e-9)
    ed.save()
    ed2 = _reopen(ed, tmp_path)
    ed2.load()
    np.testing.assert_allclose(ed2.pos, fast, atol=1e-9)
    ed.undo()  # the pushes go with the drag
    ed.remesh()
    np.testing.assert_allclose(ed.pos, v, atol=1e-12)


def _reopen(ed, tmp_path):
    sys.path.insert(0, str(Path(__file__).parent / "pial_edit"))
    from editor_server import Editor

    return Editor(str(tmp_path / "slit_session.npz"), "contour", str(ed.edits_path))


def test_push_still_stops_at_the_white_surface(slit_editor):
    ed, v = slit_editor
    r = _slit_vertex(v)
    d = json.loads(_head(ed.drag(r, (v[r] + [-8.0, 0, 0]).tolist(), push=True)))["drag"]
    assert d["fraction"] < 1 and d["pushes"] == 0 and d["blocked"]["against"] == "white surface"
    assert ed.pos[r, 0] > 23.5 and ed.crossings["faces"] == 0
