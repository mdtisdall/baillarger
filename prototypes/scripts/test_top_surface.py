"""Tests of the Σ start (top_surface.py) on small synthetic label arrays and
meshes: the remeshing, the fold rule ("reference + dihedral"), and the
start on a slab (stops at label 0), on two facing sheets (they stop at the
gap with no intersection, and the geometry gives a contact), in a narrow valley (no new fold), and at
a zero-thickness face. The checks themselves are tested in
tests/test_mesh_checks.py."""

from __future__ import annotations

from types import SimpleNamespace

import global_tau_prep as TP
import numpy as np
import top_surface as TS

from baillarger import mesh_checks as MC
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
    assert MC.boundary_vertices(m.F, len(m.V)).sum() >= MC.boundary_vertices(f, len(v)).sum()


def test_collapse_short_edges_stays_flat_and_manifold() -> None:
    v, f = grid_sheet(8, 5.0, True)
    rng = np.random.default_rng(1)
    v[:, :2] += rng.uniform(-0.3, 0.3, (len(v), 2)) * (~MC.boundary_vertices(f, len(v)))[:, None]
    m = TS.Mesh(v * H, f, v * H, np.zeros(len(v), np.int64), np.zeros(len(v)))
    nf0 = len(m.F)
    s, c = TS.remesh(m, 0.25 * H, 10 * H, 0.6 * H, MC.FOLD_COS)
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
    stop at the gap (h/4) with no intersection, the start ends because no
    vertex takes a step, the inner vertices are still MOVING (there is no
    contact state) and the measured contacts cover them: a facing sheet
    within the gap plus one step, no voxel centre between, held at the end."""
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
    _, ng, dmin = MC.gap_check(m.top, m.V, gap, want_dmin=True)
    assert ng == 0
    assert MI.intersecting_faces(m.V, m.F)[1] == 0
    inner = ~m.boundary
    assert run.ended == "no step"
    assert (m.state[inner] == TS.MOVING).all()
    c = run.contacts
    assert c["contact"][inner].mean() > 0.9
    assert (c["facing_mm"][inner & c["contact"]] < 2 * gap + 0.25 * H + 1e-9).all()
    assert (m.held[inner & c["contact"]] >= 1).all()
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


def test_refused_but_not_facing_retries() -> None:
    """Two sheets side by side with the same normal (not facing): a move
    that breaks the gap is refused, the vertex stays MOVING with held = 1,
    and the geometry finds no contact (no facing sheet)."""
    lab = np.full((14, 14, 20), TP.CORTEX, np.int16)
    lab[..., 17:] = 0
    v1, f1 = grid_sheet(6, 4.6, True)
    v2, f2 = grid_sheet(6, 4.7, True, shift=0.4)
    v = np.r_[v1, v2]
    f = np.r_[f1, f2 + len(v1)]
    m = TS.start_mesh(site_of(lab, v, f))
    chk = MC.Checker(m.top, m.V, 0.25 * H, slack=0.25 * H)
    disp, info = TS.expand_step(m, site_of(lab, v, f), TS.StartSettings(), H, chk)
    assert info["refused"] > 0
    assert (m.state == TS.MOVING).all()
    assert (m.held > 0).sum() == info["refused"]
    assert not TS.contacts(m, np.full(3, H))["contact"].any()


def test_contacts_need_no_voxel_centre_between() -> None:
    """Two facing sheets 0.6 voxels apart (a real-tissue contact need not be
    at the gap): a contact at z = 5.1 and 5.7 (no voxel centre between), not
    at z = 4.7 and 5.3 (the centres at z = 5 lie between them)."""
    for z1, z2, want in ((5.1, 5.7, True), (4.7, 5.3, False)):
        v1, f1 = grid_sheet(6, z1, True)
        v2, f2 = grid_sheet(6, z2, False, shift=0.3)
        v = np.r_[v1, v2] * H
        f = np.r_[f1, f2 + len(v1)]
        m = TS.Mesh(v, f, v, np.zeros(len(v), np.int64), np.zeros(len(v)))
        c = TS.contacts(m, np.full(3, H))
        inner = ~m.boundary
        np.testing.assert_allclose(c["facing_mm"][inner], 0.6 * H, rtol=1e-9)
        assert (c["contact"][inner] == want).all()


def test_checker_matches_mesh_intersect_and_full_check() -> None:
    """The checker's triangle test finds the same pairs as
    mesh_intersect.intersecting_faces on two sheets that cross; a check of
    the changed vertices only finds the same violations at them as a whole
    check, with the grid reused after the move."""
    v1, f1 = grid_sheet(8, 5.0, True)
    v2, f2 = grid_sheet(8, 5.0, True, shift=0.37)
    v2[:, 2] += 0.3 * (v2[:, 0] - 5.5)  # tilted: crosses the first sheet
    V = np.r_[v1, v2] * H
    F = np.r_[f1, f2 + len(v1)]
    m = TS.Mesh(V, F, V, np.zeros(len(V), np.int64), np.zeros(len(V)))
    mask, n = MI.intersecting_faces(V, F)
    bad, ng, ni, _ = MC.Checker(m.top, V, 0.25 * H).check(V)
    assert n > 0 and ni == n  # no folds here: the sheets share no vertex
    # move a few vertices of sheet 2 toward sheet 1 (no crossing yet in the
    # flat part); the local check agrees with the whole check at them
    v2b, _ = grid_sheet(8, 6.0, False, shift=0.37)
    W = np.r_[v1, v2b] * H
    Fw = np.r_[f1, f2[:, ::-1] + len(v1)]
    mw = TS.Mesh(W, Fw, W, np.zeros(len(W), np.int64), np.zeros(len(W)))
    c = MC.Checker(mw.top, W, 0.25 * H, slack=0.25 * H)
    assert c.check(W)[1] == 0
    ch = np.zeros(len(W), bool)
    ch[len(v1) + np.arange(20, 30)] = True
    W2 = W.copy()
    W2[ch, 2] -= 0.9 * H  # 0.1 voxel above sheet 1: inside the gap
    b_loc, ng_loc, ni_loc, _ = c.check(W2, ch)
    b_all, ng_all, ni_all, _ = MC.Checker(mw.top, W2, 0.25 * H).check(W2)
    assert ng_loc == ng_all > 0 and ni_loc == ni_all
    np.testing.assert_array_equal(b_loc, b_all)


def test_isolated_top_finds_a_single_label0_voxel() -> None:
    """One label-0 voxel inside the cortex of a slab stops the vertex under
    it early: it heads the review list, with a lag of more than one voxel
    and a one-voxel label-0 region; the vertices that stop at the real top
    have no lag. The fold rule keeps the neighbours from closing over it,
    so they stay as a narrow shaft (lag 1.2 voxels at the vertex, not the
    needle of more than 4 voxels that the start made without the rule).
    Whether this shaft occurs in real tissue is not known yet (user,
    2026-11-18: wait for the site-1 run)."""
    lab = np.zeros((14, 14, 20), np.int16)
    lab[..., :5] = TP.WM
    lab[..., 5:15] = TP.CORTEX
    lab[6, 6, 8] = 0  # a label error in the cortex
    v, f = grid_sheet(10, 4.6, True)
    run = TS.StartRun(site_of(lab, v, f), TS.StartSettings(collapse_ratio=0.0))
    m = run.run()
    iso = TS.isolated_top(m, lab)
    first = iso["vertex"][0]
    assert np.allclose(m.V[first, :2] / H, (6, 6), atol=0.5)
    assert iso["lag_mm"][0] > H
    assert iso["comp_voxels"][0] == 1 and not iso["comp_at_crop_edge"][0]
    assert iso["label0_3x3x3"][0] == 1
    big = iso["comp_voxels"] > 1
    assert big.any() and iso["comp_at_crop_edge"][big].all()


# ------------------------------------------------------------ the fold rule


def valley(slope: float = 3.0):
    """A narrow valley along y: z = 3 + slope × |x - 5| voxels, normals up,
    in cortex with label 0 from z = 22."""
    lab = np.full((14, 14, 24), TP.CORTEX, np.int16)
    lab[..., 22:] = 0
    v, f = grid_sheet(7, 0.0, True)
    v[:, 2] = 3.0 + slope * np.abs(v[:, 0] - 5.0)
    return site_of(lab, v, f)


def test_new_folds_dihedral_only() -> None:
    """Two faces on a common edge, each turned 80 degrees from the
    reference (+z) to opposite sides: a crease (160 degrees between the
    normals) but no face turned 90 degrees from its reference."""
    F = np.array([[0, 1, 2], [1, 0, 3]], np.int64)
    flat = np.array([[0, 0, 0], [0, 1, 0], [-1, 0.5, 0], [1, 0.5, 0]], float)
    m = TS.Mesh(flat, F, flat, np.zeros(4, np.int64), np.zeros(4))
    t = np.radians(80.0)
    ridge = np.array([[0, 0, 0], [0, 1, 0], [-np.cos(t), 0.5, -np.sin(t)], [np.cos(t), 0.5, -np.sin(t)]], float)
    ch = np.ones(4, bool)
    bad, nt, nc = TS.new_folds(m, ridge, flat, ch, MC.FOLD_COS)
    assert nt == 0 and nc == 1 and bad.all()
    # the same crease before the move is not a new fold
    assert TS.new_folds(m, ridge, ridge, ch, MC.FOLD_COS)[2] == 0
    # 70 degrees each side: 140 degrees between the normals, above the cut-off
    t = np.radians(70.0)
    ridge[2:, [0, 2]] = [[-np.cos(t), -np.sin(t)], [np.cos(t), -np.sin(t)]]
    assert TS.new_folds(m, ridge, flat, ch, MC.FOLD_COS)[2] == 0


def test_new_folds_turned_from_the_white_surface() -> None:
    """An interior vertex pushed in the plane past its 1-ring turns faces
    over (against the reference at `origin`) and folds them back on their
    neighbours; only the faces with a changed vertex are tested."""
    v, f = grid_sheet(5, 5.0, True)
    V0 = v * H
    m = TS.Mesh(V0, f, V0, np.zeros(len(v), np.int64), np.zeros(len(v)))
    V = V0.copy()
    c = 12  # the centre vertex
    V[c, 0] += 1.6 * H
    ch = np.zeros(len(v), bool)
    ch[c] = True
    bad, nt, nc = TS.new_folds(m, V, V0, ch, MC.FOLD_COS)
    assert nt > 0 and nc > 0 and bad[c]
    # no changed vertex: nothing tested
    assert TS.new_folds(m, V, V0, np.zeros(len(v), bool), MC.FOLD_COS)[0].sum() == 0


def test_start_in_a_narrow_valley_makes_no_fold(monkeypatch) -> None:
    """In a narrow valley the walls' first rows close in over the floor:
    the faces at the floor fold back on each other, inside the local patch
    of the gap test and with no crossing, so only the fold rule sees them.
    With the rule, no crease, turned face or crossing; without it (the fold
    test replaced by one that finds nothing), creases."""
    run = TS.StartRun(valley(), TS.StartSettings(max_steps=16))
    m = run.run()
    c = TS.counts(m, 0.25 * H, MC.FOLD_COS)
    assert run.initial["white"]["creases"] == 0
    assert c["creases"] == 0 and c["turned_faces"] == 0
    assert c["intersecting_pairs"] == 0 and c["shared_vertex_fold_pairs"] == 0 and c["gap_pairs"] == 0
    assert sum(r["fold_limited"] for r in run.log) > 0
    monkeypatch.setattr(TS, "new_folds", lambda m, V, V0, ch, fc: (np.zeros(len(V), bool), 0, 0))
    m2 = TS.StartRun(valley(), TS.StartSettings(max_steps=16)).run()
    c2 = TS.counts(m2, 0.25 * H, MC.FOLD_COS)
    assert c2["creases"] > 0 and c2["gap_pairs"] == 0 and c2["intersecting_pairs"] == 0


def test_remesh_undoes_a_failed_batch_and_excludes_it(monkeypatch) -> None:
    """When the check of a collapse batch fails at its merged vertices, the
    batch is undone (every per-vertex field restored) and done again
    without the collapses near them: those vertices keep their positions."""
    v, f = grid_sheet(8, 5.0, True)
    rng = np.random.default_rng(1)
    v[:, :2] += rng.uniform(-0.3, 0.3, (len(v), 2)) * (~MC.boundary_vertices(f, len(v)))[:, None]
    m = TS.Mesh(v * H, f, v * H, np.zeros(len(v), np.int64), np.arange(len(v), dtype=float))
    first: dict = {}
    real_collapse, real_folds = TS._collapse, TS.new_folds

    def collapse(m_, lmin, exclude, size=None):
        if not first:
            first["V"] = m_.V.copy()
            first["ptr"], first["nbr"] = m_.ptr, m_.nbr
        out = real_collapse(m_, lmin, exclude, size)
        if "merged" not in first:
            first["merged"] = out[2]
        return out

    calls = []

    def folds(m_, V, V0, ch, fc):
        calls.append(int(ch.sum()))
        if len(calls) == 1:  # the first batch fails at every merged vertex
            return ch.copy(), 0, 0
        return real_folds(m_, V, V0, ch, fc)

    monkeypatch.setattr(TS, "_collapse", collapse)
    monkeypatch.setattr(TS, "new_folds", folds)
    s, c = TS.remesh(m, 0.25 * H, 10 * H, 0.9 * H, MC.FOLD_COS)
    assert s == 0 and len(calls) == 2 and c > 0
    for nm in TS.VERTEX_FIELDS:
        assert len(getattr(m, nm)) == len(m.V)
    # the vertices near the failed batch were not collapsed: still there
    near = MC.near_vertices(first["ptr"], first["nbr"], np.isin(np.arange(len(first["V"])), first["merged"]))
    for p in first["V"][near]:
        assert np.isclose(m.V, p).all(1).any()
    # travel was set to the vertex index: kept vertices keep their own
    kept = np.isclose(m.travel, np.round(m.travel))
    assert kept.sum() >= near.sum()


# ------------------------------------------------------------ trough variants


def test_smooth_normals_keep_a_flat_sheet_and_open_a_valley() -> None:
    """Averaged normals stay unit length and stay +z on a flat sheet; in a
    valley they turn toward +z (they converge less)."""
    v, f = grid_sheet(6, 5.0, True)
    m = TS.Mesh(v * H, f, v * H, np.zeros(len(v), np.int64), np.zeros(len(v)))
    N = TS.smooth_normals(m, TS.normals(m.V, m.F), 3)
    np.testing.assert_allclose(np.linalg.norm(N, axis=1), 1.0)
    np.testing.assert_allclose(N[:, 2], 1.0, atol=1e-12)
    site = valley(1.0)
    m = TS.start_mesh(site)
    N0 = TS.normals(m.V, m.F)
    N3 = TS.smooth_normals(m, N0, 3)
    inner = ~m.boundary
    assert (N3[inner, 2] >= N0[inner, 2] - 1e-12).mean() > 0.8


def test_relax_along_the_normal_lifts_a_lagging_vertex() -> None:
    """With normal > 0, a MOVING vertex below its neighbours moves up toward
    them and its travel grows by the same amount; with normal = 0 it moves
    only in its tangent plane and its travel does not change."""
    v, f = grid_sheet(5, 5.0, True)
    V = v * H
    c = 12
    V[c, 2] -= 0.5 * H
    for normal, lifts in ((0.0, False), (0.5, True)):
        m = TS.Mesh(V.copy(), f, V.copy(), np.zeros(len(v), np.int64), np.zeros(len(v)))
        chk = MC.Checker(m.top, m.V, 0.25 * H, slack=0.25 * H)
        TS.relax(m, chk, MC.FOLD_COS, normal)
        dz = m.V[c, 2] - V[c, 2]
        if lifts:
            assert dz > 0.1 * H and np.isclose(m.travel[c], dz, rtol=0.05)
        else:
            assert abs(m.travel[c]) < 1e-12


def test_valley_with_each_variant_keeps_the_checks() -> None:
    """The narrow valley with either variant: no crease, turned face,
    crossing or gap pair at the end (the fold rule still holds)."""
    for s in (TS.StartSettings(max_steps=16, normal_iters=3), TS.StartSettings(max_steps=16, normal_smooth=0.5)):
        m = TS.StartRun(valley(), s).run()
        c = TS.counts(m, 0.25 * H, MC.FOLD_COS)
        assert c["creases"] == 0 and c["turned_faces"] == 0
        assert c["intersecting_pairs"] == 0 and c["shared_vertex_fold_pairs"] == 0 and c["gap_pairs"] == 0


def test_collapse_is_not_blocked_by_an_edge_that_cannot_collapse() -> None:
    """A flat grid (spacing 1) with a vertex v inside face (A, B, C), so
    that A and B have three common neighbours (C, v and the one across AB):
    AB fails the link condition. AB is made the shortest edge (0.25), and a
    second short edge (0.4) starts at a neighbour of B. In one round of
    picks (the locally shortest candidate only) AB is picked, fails, and
    blocks the other; in rounds the other edge collapses and AB stays."""
    n = 8
    xs = np.arange(n, dtype=float)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    V = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    f = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
            f += [(a, b, c), (a, c, d)]
    F = np.array(f, np.int64)
    A, B, C = 3 * n + 3, 4 * n + 3, 4 * n + 4
    k = [tuple(t) for t in F.tolist()].index((A, B, C))
    v = len(V)
    V = np.vstack([V, V[[A, B, C]].mean(0)])
    F = np.vstack([np.delete(F, k, 0), [(A, B, v), (B, C, v), (C, A, v)]])
    V[B, 0] -= 0.75  # |AB| = 0.25
    Y1, Z1 = 5 * n + 3, 6 * n + 3  # Y1 is a neighbour of B
    V[Z1, 0] -= 0.6  # |Y1 Z1| = 0.4
    m = TS.Mesh(V.copy(), F.copy(), V.copy(), np.zeros(len(V), np.int64), np.zeros(len(V)))
    m.update_topology()
    u = TS._edges(m.F)
    L = np.linalg.norm(m.V[u[:, 0]] - m.V[u[:, 1]], axis=1)
    c = (L < 0.45) & ~m.boundary[u[:, 0]] & ~m.boundary[u[:, 1]]
    E = u[c]
    assert sorted(map(sorted, E.tolist())) == sorted([sorted([A, B]), sorted([Y1, Z1])])
    # one round: only AB is picked, and it fails the link condition
    rank = np.argsort(np.argsort(L[c]))
    pick = TS._local_best(E, rank, len(m.V), m.ptr, m.nbr)
    assert sorted(E[pick].ravel().tolist()) == sorted([A, B])
    assert not MC.link_condition(m.ptr, m.nbr, E[pick]).any()
    # rounds: Y1 Z1 collapses, AB stays
    nc, newid, merged = TS._collapse(m, 0.45, np.zeros(len(V), bool))
    assert nc == 1 and merged.tolist() == [min(Y1, Z1)] and newid[A] >= 0 and newid[B] >= 0
