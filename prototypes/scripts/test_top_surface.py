"""Tests of the Σ start (top_surface.py) on small synthetic label arrays and
meshes: the point-triangle distance, the gap test, the remeshing, and the
start on a slab (stops at label 0), on two facing sheets (stop at the gap,
no intersection: a first contact), and at a zero-thickness face."""

from __future__ import annotations

from types import SimpleNamespace

import global_tau_prep as TP
import numpy as np
import top_surface as TS

from baillarger import mesh_intersect as MI

H = 0.16


def grid_sheet(n: int, z: float, up: bool, shift: float = 0.0, step: float = 1.0):
    """A square sheet of n × n vertices at height z (voxel units), spacing
    `step` voxels, starting at x = y = 2 + shift; faces oriented so the
    normal is +z (up) or -z."""
    xs = 2 + shift + step * np.arange(n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    v = np.c_[X.ravel(), Y.ravel(), np.full(n * n, z)]
    f = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
            f += [(a, b, c), (a, c, d)]
    f = np.array(f, np.int64)
    if not up:
        f = f[:, ::-1].copy()
    return v, f


def site_of(labels: np.ndarray, v: np.ndarray, f: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(labels=labels, vox=np.full(3, H), v=v, f=f)


def test_point_triangle_distance_brute_force() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b, c, p = rng.normal(size=(4, 3))
        u = rng.random((20000, 2))
        u = u[u.sum(1) <= 1]
        q = a + u[:, :1] * (b - a) + u[:, 1:] * (c - a)
        brute = np.linalg.norm(q - p, axis=1).min()
        d = TS._pt_tri(p, a, b, c)
        assert d <= brute + 1e-12
        assert brute - d < 0.02 * np.linalg.norm(b - a)


def test_gap_check_two_sheets_and_own_sheet() -> None:
    """Two parallel sheets 0.1 voxel apart violate a gap of 0.25 voxel; one
    sheet alone has no violation (its own faces are its local patch)."""
    v1, f1 = grid_sheet(6, 5.0, True)
    v2, f2 = grid_sheet(6, 5.1, False, shift=0.3)
    one = TS.Mesh(v1 * H, f1, v1 * H, np.zeros(len(v1), np.int64), np.zeros(len(v1)))
    bad, n, dmin = TS.gap_check(one, one.V, 0.25 * H)
    assert n == 0 and not bad.any()
    V = np.r_[v1, v2] * H
    F = np.r_[f1, f2 + len(v1)]
    two = TS.Mesh(V, F, V, np.zeros(len(V), np.int64), np.zeros(len(V)))
    bad, n, dmin = TS.gap_check(two, two.V, 0.25 * H, want_dmin=True)
    assert n > 0 and bad[: len(v1)].any() and bad[len(v1) :].any()
    xy = V[:, :2] / H
    over = np.all((xy >= 2.3 - 1e-9) & (xy <= 7.0 + 1e-9), axis=1)  # above the other sheet
    np.testing.assert_allclose(dmin[over], 0.1 * H, atol=1e-9)


def test_split_keeps_the_shape() -> None:
    v, f = grid_sheet(4, 5.0, True, step=3.0)
    m = TS.Mesh(v * H, f, v * H, np.zeros(len(v), np.int64), np.zeros(len(v)))
    a0 = np.linalg.norm(np.cross(*(m.V[m.F[:, 1:]] - m.V[m.F[:, :1]]).transpose(1, 0, 2)), axis=1).sum()
    n = TS._split(m, 2.0 * H)
    assert n > 0
    a1 = np.linalg.norm(np.cross(*(m.V[m.F[:, 1:]] - m.V[m.F[:, :1]]).transpose(1, 0, 2)), axis=1).sum()
    np.testing.assert_allclose(a1, a0, rtol=1e-12)
    np.testing.assert_allclose(m.V[:, 2], 5.0 * H)
    assert len(m.F) == len(f) + 2 * n
    assert TS.boundary_vertices(m.F, len(m.V)).sum() >= TS.boundary_vertices(f, len(v)).sum()


def test_collapse_short_edges_stays_flat_and_manifold() -> None:
    v, f = grid_sheet(8, 5.0, True)
    rng = np.random.default_rng(1)
    v[:, :2] += rng.uniform(-0.3, 0.3, (len(v), 2)) * (~TS.boundary_vertices(f, len(v)))[:, None]
    m = TS.Mesh(v * H, f, v * H, np.zeros(len(v), np.int64), np.zeros(len(v)))
    nf0 = len(m.F)
    s, c = TS.remesh(m, 0.25 * H, 10 * H, 0.6 * H)
    assert s == 0 and c > 0
    assert len(m.F) == nf0 - 2 * c
    np.testing.assert_allclose(m.V[:, 2], 5.0 * H)
    e = np.sort(np.r_[m.F[:, [0, 1]], m.F[:, [1, 2]], m.F[:, [2, 0]]], 1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    assert cnt.max() == 2
    cr = np.cross(m.V[m.F[:, 1]] - m.V[m.F[:, 0]], m.V[m.F[:, 2]] - m.V[m.F[:, 0]])
    assert (cr[:, 2] > 0).all()


def test_start_on_a_slab_stops_at_label_0() -> None:
    """WM below z = 5, cortex 5 ≤ z < 15, label 0 above. The white sheet at
    z = 4.6 moves out in steps of h/4 and stops within h/4 + h/2 of the
    label-0 voxels (rounded to the nearest voxel: label 0 from z = 14.5)."""
    lab = np.zeros((14, 14, 20), np.int16)
    lab[..., :5] = TP.WM
    lab[..., 5:15] = TP.CORTEX
    v, f = grid_sheet(10, 4.6, True)
    run = TS.StartRun(site_of(lab, v, f), TS.StartSettings())
    m = run.run()
    z = m.V[:, 2] / H
    assert (m.state == TS.TOP).all()
    assert (z >= 14.5 - 0.25 - 1e-9).all() and (z < 14.5).all()
    assert MI.intersecting_faces(m.V, m.F)[1] == 0


def test_start_two_sheets_stop_at_the_gap() -> None:
    """Two sheets facing each other in cortex, no label 0 between them: they
    stop at the gap (h/4) as BLOCKED, the first contacts, with no
    intersection."""
    lab = np.full((14, 14, 20), TP.CORTEX, np.int16)
    lab[..., :5] = TP.WM
    lab[..., 15:] = TP.WM
    v1, f1 = grid_sheet(8, 4.6, True)
    v2, f2 = grid_sheet(8, 14.4, False, shift=0.4)
    v = np.r_[v1, v2]
    f = np.r_[f1, f2 + len(v1)]
    run = TS.StartRun(site_of(lab, v, f), TS.StartSettings(collapse_ratio=0.0))
    m = run.run()
    gap = 0.25 * H
    _, ng, dmin = TS.gap_check(m, m.V, gap, want_dmin=True)
    assert ng == 0
    assert MI.intersecting_faces(m.V, m.F)[1] == 0
    inner = ~m.boundary
    assert (m.state[inner] == TS.BLOCKED).mean() > 0.9
    # they stopped close to the gap: within one step (h/4) above it
    d = dmin[inner & np.isfinite(dmin)]
    assert d.min() >= gap and np.median(d) < gap + 0.25 * H + 1e-9


def test_zero_thickness_does_not_move() -> None:
    """A white sheet with label 0 directly above it does not move."""
    lab = np.zeros((14, 14, 20), np.int16)
    lab[..., :5] = TP.WM
    v, f = grid_sheet(6, 4.6, True)
    run = TS.StartRun(site_of(lab, v, f), TS.StartSettings())
    m = run.run()
    assert run.initial["zero_thickness"] == len(v)
    np.testing.assert_allclose(m.V, v * H)


def test_wm_pass_through_and_limit() -> None:
    """A vertex in WM keeps moving; it stops as WM after wm_limit voxels."""
    lab = np.full((14, 14, 20), TP.WM, np.int16)
    v, f = grid_sheet(6, 4.6, True)
    run = TS.StartRun(site_of(lab, v, f), TS.StartSettings(wm_limit_vox=2.0))
    m = run.run()
    assert (m.state == TS.WM).all()
    np.testing.assert_allclose(m.travel, 2.0 * H, atol=1e-12)


def test_foldover_at_a_shared_vertex() -> None:
    """Two faces that share only vertex 0: apart, no fold; when one is
    turned to cross the other beyond vertex 0, a fold (the triangle test of
    mesh_intersect skips this pair)."""
    F = np.array([[0, 1, 2], [0, 3, 4]], np.int64)
    flat = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [-1, 0, 0], [-1, 1, 0]], float)
    m = TS.Mesh(flat, F, flat, np.zeros(5, np.int64), np.zeros(5))
    assert TS.foldovers(m, flat)[1] == 0
    cross = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0.8, 0.3, -0.5], [0.8, 0.3, 0.5]], float)
    assert MI.intersecting_faces(cross, F)[1] == 0
    assert TS.foldovers(m, cross)[1] == 1


def test_refused_but_not_facing_retries() -> None:
    """Two sheets side by side with the same normal (not facing): a move
    that breaks the gap is refused, but the vertex is not BLOCKED (no
    contact)."""
    lab = np.full((14, 14, 20), TP.CORTEX, np.int16)
    lab[..., 17:] = 0
    v1, f1 = grid_sheet(6, 4.6, True)
    v2, f2 = grid_sheet(6, 4.7, True, shift=0.4)
    v = np.r_[v1, v2]
    f = np.r_[f1, f2 + len(v1)]
    m = TS.start_mesh(site_of(lab, v, f))
    disp, info = TS.expand_step(m, site_of(lab, v, f), TS.StartSettings(), H)
    assert info["refused"] > 0 and info["blocked"] == 0
    assert not (m.state == TS.BLOCKED).any()
