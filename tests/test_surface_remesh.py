"""Tests for baillarger.surface_remesh (ported from remesh_seq): a sphere,
a U-shaped plate whose inner faces are closer than the target edge (the
remesher must keep them apart), an open sheet with a box rim (corners stay,
the rim stays on the box), and a slab with a thin fin (the normal check and
the outward bound)."""

from typing import Any

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from baillarger import mesh_checks as MC
from baillarger import surface_remesh as SR
from baillarger.mesh import marching_tetrahedra

GAP = 0.25


def sphere() -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(30, dtype=float)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    v, f = marching_tetrahedra(
        np.sqrt((X - 14.3) ** 2 + (Y - 15.1) ** 2 + (Z - 14.7) ** 2), 10.0
    )
    return v.astype(np.float64), f.astype(np.int64)


def u_plate() -> tuple[np.ndarray, np.ndarray]:
    """A closed U: two plates joined at one end, their inner faces about
    two voxels apart."""
    a = np.zeros((36, 30, 24))
    a[5:30, 5:25, 6:10] = 1
    a[5:30, 5:25, 13:17] = 1
    a[26:30, 5:25, 6:17] = 1
    v, f = marching_tetrahedra(a, 0.5)
    return v.astype(np.float64), f.astype(np.int64)


def sheet() -> tuple[np.ndarray, np.ndarray]:
    """A bumpy open sheet over the square [0, 20]², spacing 0.5."""
    s = np.linspace(0.0, 20.0, 41)
    X, Y = np.meshgrid(s, s, indexing="ij")
    V = np.c_[X.ravel(), Y.ravel(), (5 + np.sin(X / 4) * np.cos(Y / 5)).ravel()]
    n = len(s)
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
            F += [[a, b, c], [a, c, d]]
    return V, np.array(F, np.int64)


def run(
    V: np.ndarray, F: np.ndarray, target: float, corner: np.ndarray | None = None
) -> tuple[SR.Remesher, list[dict[str, Any]]]:
    ref = SR.Reference(V, F, np.full(len(V), target))
    rm = SR.Remesher(ref, V.copy(), F.copy(), GAP, corner=corner)
    assert rm.chk.check_all().ok  # the start is valid
    log = SR.remesh(rm, 1.0, max_passes=12)
    return rm, log


def assert_valid(rm: SR.Remesher) -> tuple[np.ndarray, np.ndarray]:
    V, F, _ = rm.chk.export()
    c = MC.check_mesh(V, F, gap=GAP)
    assert c.counts()["intersecting_pairs"] == 0
    assert c.vertex_fold_pairs == 0 and c.edge_fold_pairs == 0 and c.gap_pairs == 0
    n = MC.face_normals(V, F)
    _, _, wf = rm.ref.project(V[F].mean(1))
    assert ((n * rm.ref.N[wf]).sum(1) > 0).all()  # none turned against white
    return V, F


@pytest.mark.parametrize(("mesh", "target"), [("sphere", 2.5), ("u_plate", 3.0)])
def test_closed_meshes_reach_the_target_and_stay_valid(
    mesh: str, target: float
) -> None:
    V0, F0 = sphere() if mesh == "sphere" else u_plate()
    rm, log = run(V0, F0, target)
    V, F = assert_valid(rm)
    last = log[-1]
    assert last["edge_in_range"] > 0.85, last
    assert len(F) < len(F0) / 4  # coarser
    # the smoothing is not projected: the vertices stay near the white
    # surface, within half the target (it rounds the U plate's 90-degree
    # box edges by about a third of the target; no check bounds it here)
    _, d, _ = rm.ref.project(V)
    assert d.max() < target / 2, d.max()
    if mesh == "u_plate":
        # the inner faces are closer than the target: some operations there
        # were refused, and the two plates stay apart
        assert sum(sum(r["refused"].values()) for r in log) > 0


def test_an_open_sheet_keeps_its_corners_and_its_rim_on_the_box() -> None:
    V0, F0 = sheet()
    rim, E = SR.rim_of(F0, len(V0))
    corner = SR.box_corners(
        V0, rim, E, np.array([0, 0, -100.0]), np.array([20, 20, 100.0]), 1e-9
    )
    assert corner.sum() == 4
    rm, log = run(V0, F0, 2.0, corner)
    V, F = assert_valid(rm)
    assert log[-1]["edge_in_range"] > 0.8, log[-1]
    # the corners are where they were
    for p in V0[corner]:
        assert np.linalg.norm(V - p, axis=1).min() < 1e-12
    # the rim is on the box
    rim2, _ = SR.rim_of(F, len(V))
    on_box = np.isclose(V[rim2, 0], 0) | np.isclose(V[rim2, 0], 20)
    on_box |= np.isclose(V[rim2, 1], 0) | np.isclose(V[rim2, 1], 20)
    assert on_box.all()


def finger() -> tuple[np.ndarray, np.ndarray]:
    """A slab with a thin fin on top (3 voxels thick, 6 tall): the fin is
    narrower than the target edge."""
    a = np.zeros((24, 20, 22))
    a[3:21, 3:17, 3:8] = 1
    a[7:17, 9:12, 8:14] = 1
    v, f = marching_tetrahedra(a, 0.5)
    return v.astype(np.float64), f.astype(np.int64)


@pytest.mark.parametrize("check", [False, True])
def test_the_normal_check_keeps_orientation_and_bounds_outward(check: bool) -> None:
    """Target 4 on a slab with a fin 3 voxels thick. Without the checks,
    the 3D smoothing moves Σ more than a voxel outward. With theta 30
    degrees and d_out 0.5: no white vertex is more than 0.5 inside Σ's
    outer side, no face is farther from its white normals than the worst
    face of the start, and some operations are refused by the check."""
    V0, F0 = finger()
    ref = SR.Reference(V0, F0, np.full(len(V0), 4.0))
    kw: dict[str, Any] = {"theta_deg": 30.0, "d_out": 0.5} if check else {}
    rm = SR.Remesher(ref, V0.copy(), F0.copy(), GAP, **kw)
    start = rm.measures(1.0)
    log = SR.remesh(rm, 1.0, max_passes=4)
    assert_valid(rm)
    last = log[-1]
    assert last["faces"] < len(F0) / 3, last  # coarser
    if not check:
        assert last["outward_max_vox"] > 1.0, last
    else:
        assert last["outward_max_vox"] <= 0.5 + 1e-9, last
        assert (
            last["face_angle_p50_90_99_max"][3]
            <= start["face_angle_p50_90_99_max"][3] + 1e-6
        ), (start, last)
        assert any(k.endswith("_normal") for r in log for k in r["refused"])
        # the running sums and counts are those of the assignment
        live = np.flatnonzero(rm.chk.F[:, 0] >= 0)
        has = rm.wf >= 0
        C = np.bincount(rm.wf[has], minlength=len(rm.Cf))
        S = np.zeros_like(rm.Sf)
        np.add.at(S, rm.wf[has], rm.Wn[has])
        assert (C[live] == rm.Cf[live]).all()
        assert np.allclose(S[live], rm.Sf[live], atol=1e-9)
        assert all(rm.wf[w] == f for f, ws in rm.fw.items() for w in ws)


def test_compiled_closest_on_is_the_numpy_one_bit_for_bit() -> None:
    """closest_on (compiled after the candidate search) gives the same bits
    as closest_on_py: points near the surface, on its vertices and edge
    midpoints (ties between faces), and with a degenerate face."""
    v, f = sphere()
    f = np.r_[f, [[f[0, 0], f[0, 0], f[0, 1]]]]  # a face of zero area
    rng = np.random.default_rng(3)
    P = np.r_[
        v[rng.integers(0, len(v), 3000)] + rng.normal(0, 0.7, (3000, 3)),
        v[:500],
        0.5 * (v[f[:500, 0]] + v[f[:500, 1]]),
    ]
    for k in (1, 12, 24):
        a = SR.closest_on(v, f, P, k)
        b = SR.closest_on_py(v, f, P, k)
        for x, y in zip(a, b, strict=True):
            assert np.array_equal(x, y, equal_nan=True)


def test_the_dot_order_is_numpys_here() -> None:
    """The order of sums found at import is the one np.einsum uses here, on
    other rows than the probe's (where none is found, closest_on runs in
    numpy)."""
    if SR.DOT_ORDER < 0:
        pytest.skip("numpy sums in an order closest_on does not know here")
    rng = np.random.default_rng(7)
    x = rng.normal(size=(20000, 3)) * 10.0 ** rng.integers(-4, 5, (20000, 1))
    y = rng.normal(size=(20000, 3))
    want = np.einsum("ij,ij->i", x, y)
    assert np.array_equal(SR._dot3_rows(x, y, SR.DOT_ORDER), want)


def test_compiled_neighbours_and_edge_faces_are_as_before() -> None:
    v, f = sphere()
    chk = MC.Checker(MC.Topology.of(f, len(v)), v, GAP)
    F = chk.F
    for a in range(0, len(v), 7):
        fs = chk.faces_of(a)
        old = {int(u) for x in fs for u in F[x]} - {a}
        assert set(chk.neighbours_of(a).tolist()) == old
        assert len(chk.neighbours_of(a)) == len(old)
        for b in old:
            want = [int(x) for x in fs if b in F[x]]
            assert chk.edge_faces(a, b).tolist() == want


def test_the_neighbour_sets_iterate_as_before() -> None:
    """Remesher.nbrs is the set that the uncompiled code built, iterated in
    the same order: relax_all sums V[list(nbrs)] in that order, so another
    order changes the bits of every relaxed position. After a remesh, so
    that ids are large and the faces round each vertex are reordered."""
    v, f = sphere()
    rm, _ = run(v, f, 0.7)
    F = rm.chk.F
    live = np.unique(F[F[:, 0] >= 0])
    assert len(live) > 100
    for a in live.tolist():
        old = {int(u) for x in rm.chk.faces_of(a) for u in F[x]} - {a}
        assert list(rm.nbrs(a)) == list(old), a


@pytest.mark.parametrize("mesh", [sphere, sheet])
def test_compiled_flip_all_is_the_loop_of_flips(mesh: Any) -> None:
    """flip_all (its test compiled) makes the same flips, in the same order,
    as flip() on each edge in turn; after collapses, so that valences are
    uneven."""
    V, F = mesh()

    def made() -> SR.Remesher:
        ref = SR.Reference(V, F, np.full(len(V), 2.5))
        rm = SR.Remesher(ref, V.copy(), F.copy(), GAP)
        rm.collapse_all()
        return rm

    a, b = made(), made()
    E, _ = b._edges()
    n_loop = 0
    for x, y in E:
        x, y = int(x), int(y)
        if len(b.chk.faces_of(x)) == 0 or y not in b.nbrs(x):
            continue
        n_loop += bool(b.flip(x, y))
    assert a.flip_all() == n_loop > 0
    for p, q in zip(a.chk.export(), b.chk.export(), strict=True):
        assert np.array_equal(p, q)


def test_compiled_cross_and_turn_test_are_numpys_bit_for_bit() -> None:
    rng = np.random.default_rng(3)
    V = rng.normal(size=(400, 3)) * 10.0 ** rng.integers(-3, 3, size=(400, 1))
    F = rng.integers(0, 400, size=(1000, 3))
    T = V[F]
    ref = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    assert np.array_equal(SR.tri_cross(T), ref)
    assert np.array_equal(SR.face_cross(V, F), ref)
    assert np.array_equal(SR.face_cross(V, F), MC.face_normals(V, F))
    R = rng.normal(size=(len(ref), 3))
    for i in range(len(ref)):
        a, b = ref[i : i + 1], R[i : i + 1]
        assert SR.any_against(a, b) == bool(((a * b).sum(1) <= 0).any())
    zero = np.zeros((1, 3))
    assert SR.any_against(zero, R[:1])  # a face of zero area is turned


def two_sheets() -> tuple[np.ndarray, np.ndarray]:
    """The bumpy sheet and a copy above it, 0.2 (closer than GAP) for x < 8,
    0.6 for x > 12: as at the hemisphere's gap sites, edits near the close
    part are refused by the gap check, pass after pass."""
    V, F = sheet()
    up = 0.2 + 0.4 * np.clip((V[:, 0] - 8) / 4, 0, 1)
    return np.r_[V, V + np.c_[0 * up, 0 * up, up]], np.r_[F, F + len(V)]


@pytest.mark.parametrize("mesh", ["two_sheets", "u_plate", "sphere"])
def test_the_memo_of_refusals_changes_nothing(mesh: str) -> None:
    V0, F0 = {"two_sheets": two_sheets, "u_plate": u_plate, "sphere": sphere}[mesh]()
    out = []
    for memo in (False, True):
        ref = SR.Reference(V0, F0, np.full(len(V0), 2.0))
        rm = SR.Remesher(ref, V0.copy(), F0.copy(), GAP)
        rm.memo = memo
        log = SR.remesh(rm, 1.0, max_passes=6)
        out.append(
            (
                rm.chk.export(),
                [(r["refused"], r["splits"], r["collapses"], r["flips"]) for r in log],
                rm.memo_hits,
            )
        )
    (a, la, _), (b, lb, hits) = out
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)
    assert la == lb  # the same edits and refusals, pass by pass
    if mesh == "two_sheets":
        assert hits > 0  # the memo answered some tries


# ---------------------------------------------------- the memo, carried over


def small_sheet() -> tuple[np.ndarray, np.ndarray]:
    """The bumpy sheet over [0, 8]², spacing 0.5 (the carried-memo tests'
    meshes are small: they must run in seconds)."""
    s = np.linspace(0.0, 8.0, 17)
    X, Y = np.meshgrid(s, s, indexing="ij")
    V = np.c_[X.ravel(), Y.ravel(), (5 + np.sin(X / 2) * np.cos(Y / 2.5)).ravel()]
    n = len(s)
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
            F += [[a, b, c], [a, c, d]]
    return V, np.array(F, np.int64)


def small_sphere() -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(16, dtype=float)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    v, f = marching_tetrahedra(
        np.sqrt((X - 7.3) ** 2 + (Y - 7.6) ** 2 + (Z - 7.4) ** 2), 5.0
    )
    return v.astype(np.float64), f.astype(np.int64)


def small_two_sheets() -> tuple[np.ndarray, np.ndarray]:
    """small_sheet and a copy above it, 0.2 (closer than GAP) for x < 3,
    0.6 for x > 5."""
    V, F = small_sheet()
    up = 0.2 + 0.4 * np.clip((V[:, 0] - 3) / 2, 0, 1)
    return np.r_[V, V + np.c_[0 * up, 0 * up, up]], np.r_[F, F + len(V)]


def offset_sheets() -> tuple[np.ndarray, np.ndarray]:
    """small_sheet and a copy shifted 0.37 in x and 0.21 in y, 0.15 to 0.3
    above it (closer than GAP), its triangles not on the first's."""
    V, F = small_sheet()
    up = 0.15 + 0.15 * V[:, 1] / 8
    W = V + np.c_[0 * up + 0.37, 0 * up + 0.21, up]
    return np.r_[V, W], np.r_[F, F + len(V)]


def _first(
    V0: np.ndarray, F0: np.ndarray
) -> tuple[SR.Remesher, np.ndarray, np.ndarray, dict[str, Any]]:
    rm = SR.Remesher(
        SR.Reference(V0, F0, np.full(len(V0), 2.0)), V0.copy(), F0.copy(), GAP
    )
    SR.remesh(rm, 1.0, max_passes=2)
    V, F, new = rm.chk.export()
    carried = rm.memo_export(V, F, new)
    assert carried is not None
    return rm, V, F, carried


def _second(
    V: np.ndarray, F: np.ndarray, carried: dict[str, Any] | None
) -> tuple[SR.Remesher, tuple[int, int], tuple[Any, ...], list[Any]]:
    rm = SR.Remesher(SR.Reference(V, F, np.full(len(V), 2.0)), V.copy(), F.copy(), GAP)
    n = rm.memo_import(carried) if carried is not None else (0, 0)
    log = SR.remesh(rm, 1.0, max_passes=3)
    rows = [(r["refused"], r["splits"], r["collapses"], r["flips"]) for r in log]
    return rm, n, rm.chk.export(), rows


def _same_run(a: tuple[Any, ...], b: tuple[Any, ...]) -> None:
    for x, y in zip(a[2], b[2], strict=True):
        assert np.array_equal(x, y)
    assert a[3] == b[3]
    ma, mb = a[0]._memo, b[0]._memo
    assert set(mb) <= set(ma)  # carried extras were never tried again
    assert all(ma[k][0] == mb[k][0] for k in mb)


def _moves(V: np.ndarray) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    far = V.copy()  # the upper sheet's far part (x > 5.5) only
    sel = (V[:, 0] > 5.5) & (V[:, 2] > 5.3)
    far[sel] += rng.normal(0, 0.01, (int(sel.sum()), 3))
    everywhere = V + rng.normal(0, 0.005, V.shape)
    return {"none": V.copy(), "far": far, "everywhere": everywhere}


@pytest.mark.parametrize(
    ("mesh", "move"),
    [
        ("two_sheets", "none"),
        ("two_sheets", "far"),
        ("two_sheets", "everywhere"),
        ("offset_sheets", "far"),
    ],
)
def test_carried_refusals_change_nothing(mesh: str, move: str) -> None:
    V0, F0 = {"two_sheets": small_two_sheets, "offset_sheets": offset_sheets}[mesh]()
    _, V, F, carried = _first(V0, F0)
    assert carried["entries"]
    V2 = _moves(V)[move]
    plain = _second(V2, F, None)
    carry = _second(V2, F, carried)
    _same_run(carry, plain)
    n_in, n_alive = carry[1]
    assert n_in == len(carried["entries"])
    if move == "none":
        assert n_alive == n_in
        assert carry[0].memo_hits > plain[0].memo_hits
    if move == "far":
        assert 0 < n_alive <= n_in
        assert carry[0].memo_hits > plain[0].memo_hits
    if move == "everywhere":
        assert n_alive == 0  # every cell is stamped


def test_a_refusal_is_not_reused_where_the_gap_has_opened() -> None:
    V0, F0 = small_two_sheets()
    _, V, F, carried = _first(V0, F0)
    gap_keys = [k for k, r, _, _ in carried["entries"] if r == "gap"]
    assert gap_keys
    V2 = V.copy()
    # the upper sheet (the component with the higher mean z) leaves
    A = coo_matrix((np.ones(len(F)), (F[:, 0], F[:, 1])), (len(V), len(V)))
    B = coo_matrix((np.ones(len(F)), (F[:, 1], F[:, 2])), (len(V), len(V)))
    nc, lab = connected_components(A + B, directed=False)
    assert nc == 2
    up = lab == (0 if V[lab == 0, 2].mean() > V[lab == 1, 2].mean() else 1)
    V2[up] += [0.0, 0.0, 3.0]
    assert (V2 != V).any()
    plain = _second(V2, F, None)
    carry = _second(V2, F, carried)
    _same_run(carry, plain)
    assert carry[1][1] == 0  # nothing near the moved sheet survives
    assert carry[0].memo_hits == plain[0].memo_hits
    # the edits that the gap refused are kept now
    assert sum(x[1] + x[2] + x[3] for x in carry[3]) > 0
    assert carry[3][0][0].get("collapse_gap", 0) == 0


def test_the_carried_memo_is_rejected_when_the_inputs_differ() -> None:
    V0, F0 = small_two_sheets()
    _, V, F, carried = _first(V0, F0)

    def build(F_: np.ndarray, gap: float = GAP) -> SR.Remesher:
        return SR.Remesher(
            SR.Reference(V, F_, np.full(len(V), 2.0)), V.copy(), F_.copy(), gap
        )

    assert build(F).memo_import(carried)[0] == len(carried["entries"])
    F2 = F.copy()
    F2[5] = F2[5][[1, 2, 0]]  # one face row, rotated
    assert build(F2).memo_import(carried) == (0, 0)
    F3 = F.copy()
    F3[[3, 4]] = F3[[4, 3]]  # two rows swapped
    assert build(F3).memo_import(carried) == (0, 0)
    assert build(F, GAP * 1.01).memo_import(carried) == (0, 0)
    rm = SR.Remesher(
        SR.Reference(V, F, np.full(len(V), 2.0)), V.copy(), F.copy(), GAP, fold_cos=-0.5
    )
    assert rm.memo_import(carried) == (0, 0)
    rm = build(F)
    rm.chk.slack = 0.1
    assert rm.memo_import(carried) == (0, 0)
    rm = build(F)
    rm.memo = False
    assert rm.memo_import(carried) == (0, 0)
    assert rm.memo_export(*rm.chk.export()) is None
    V3 = np.r_[V, V[:1]]  # a vertex more
    rm = SR.Remesher(SR.Reference(V3, F, np.full(len(V3), 2.0)), V3, F.copy(), GAP)
    assert rm.memo_import(carried) == (0, 0)
    carried2 = dict(carried, V=carried["V"].copy())
    carried2["V"][0] = np.nan  # a non-finite moved vertex: nothing is carried
    assert build(F).memo_import(carried2) == (0, 0)


def test_the_carried_keys_are_those_that_try_builds() -> None:
    V0, F0 = small_two_sheets()
    rm1, V, F, carried = _first(V0, F0)
    rm2 = SR.Remesher(SR.Reference(V, F, np.full(len(V), 2.0)), V.copy(), F.copy(), GAP)
    assert rm2.memo_import(carried)[0] == len(carried["entries"])
    seen = set()
    for key, reason, _, _ in carried["entries"]:
        kind = key[0]
        mv = np.frombuffer(key[1], np.int64)
        rem = np.frombuffer(key[3], np.int64)
        assert (mv < len(V)).all() and (rem < len(F)).all()
        if kind in seen or len(mv) != 1 and kind == "collapse":
            continue
        seen.add(kind)
        # the edit, built from the exported ids, gives the key, and the
        # Checker of the new mesh refuses it for the same reason
        to = None if key[2] is None else np.frombuffer(key[2], np.float64)
        af = None if key[4] is None else np.frombuffer(key[4], np.int64).reshape(-1, 3)
        av = (
            None if key[5] is None else np.frombuffer(key[5], np.float64).reshape(-1, 3)
        )
        assert SR.Remesher._memo_key(kind, mv, to, rem, af, av) == key
        assert key in rm2._memo
        # the edit's faces are live faces of the new mesh, and its vertices
        # exist: propose it and get the same reason
        r = rm2.chk.propose(mv, None if to is None else to.reshape(-1, 3), rem, af, av)
        got = (
            "gap"
            if r.gap_pairs
            else "cross"
            if (r.intersecting_pairs or r.vertex_fold_pairs)
            else "crease"
            if r.creases
            else ""
        )
        rm2.chk.rollback()
        assert got == reason
    assert len(seen) >= 2


# the checks' answers do not depend on the order of the face rows, of the
# faces round a vertex, or on free ids


def _reason(r: Any) -> str:
    return (
        "gap"
        if r.gap_pairs
        else "cross"
        if (r.intersecting_pairs or r.vertex_fold_pairs)
        else "crease"
        if r.creases
        else ""
    )


def _triples(chk: MC.Checker, r: Any) -> Any:
    """The result's pairs, the faces named by their vertex triples."""

    def tri(f: int) -> tuple[int, ...]:
        return tuple(sorted(int(v) for v in chk.F[f]))

    return (
        sorted(int(v) for v in r.bad),
        sorted(tri(int(f)) for f in r.cross_faces),
        sorted(tri(int(f)) for f in r.crease_faces),
        r.gap_pairs,
        r.intersecting_pairs,
        r.vertex_fold_pairs,
        r.creases,
    )


def _edits(
    chk: MC.Checker, V: np.ndarray, F: np.ndarray
) -> list[tuple[str, dict[str, Any]]]:
    """Edits the Checker refuses with each reason, found by trying moves and flips."""
    found: dict[str, tuple[str, dict[str, Any]]] = {}
    rng = np.random.default_rng(1)
    cand: list[dict[str, Any]] = []
    for v in rng.choice(len(V), min(400, len(V)), replace=False):
        for d in (
            [0, 0, 0.1],
            [0, 0, -0.3],
            [0, 0, 0.5],
            [0.3, 0.2, -0.6],
            [0, 0, 1.2],
        ):
            cand.append({"move": [int(v)], "to": (V[v] + d)[None]})
    for x in rng.choice(len(F), min(400, len(F)), replace=False):
        # a vertex pushed through the sheet
        v = int(F[x][0])
        cand.append({"move": [v], "to": (V[v] + [0, 0, 2.5])[None]})
        cand.append({"move": [v], "to": (V[v] + [0, 0, -2.5])[None]})
        cand.append({"move": [v], "to": (V[v] + [1.5, 0, 0])[None]})
    for c in cand:
        r = chk.propose(**c)
        reason = _reason(r)
        chk.rollback()
        if reason and reason not in found:
            found[reason] = (reason, c)
    return list(found.values())


@pytest.mark.parametrize("mesh", ["two_sheets", "sphere", "offset_sheets"])
def test_the_checks_do_not_read_row_order_face_order_or_free_ids(mesh: str) -> None:
    V0, F0 = {
        "two_sheets": small_two_sheets,
        "sphere": small_sphere,
        "offset_sheets": offset_sheets,
    }[mesh]()
    # the same mesh as a remesh returns it
    rm = SR.Remesher(
        SR.Reference(V0, F0, np.full(len(V0), 2.0)), V0.copy(), F0.copy(), GAP
    )
    SR.remesh(rm, 1.0, max_passes=2)
    V, F, new = rm.chk.export()
    base = MC.Checker(MC.Topology.of(F, len(V)), V, GAP)
    edits = _edits(base, V, F)
    reasons = {e[0] for e in edits}
    # the remesher's own refused edits (collapses: removed and added faces),
    # decoded from the exported memo, a few of each kind and reason
    carried = rm.memo_export(V, F, new)
    assert carried is not None
    per: dict[tuple[str, str], int] = {}
    n_collapse = 0
    for key, why, _, _ in carried["entries"]:
        if per.get((key[0], why), 0) >= 3:
            continue
        per[(key[0], why)] = per.get((key[0], why), 0) + 1
        to = None if key[2] is None else np.frombuffer(key[2], np.float64)
        e: dict[str, Any] = {
            "move": np.frombuffer(key[1], np.int64).tolist(),
            "to": None if to is None else to.reshape(-1, 3),
            "remove_faces": np.frombuffer(key[3], np.int64).tolist(),
            "add_faces": None
            if key[4] is None
            else np.frombuffer(key[4], np.int64).reshape(-1, 3),
            "add_vertices": None
            if key[5] is None
            else np.frombuffer(key[5], np.float64).reshape(-1, 3),
        }
        edits.append((why, e))
        n_collapse += key[0] == "collapse"
    if any(k[0] == "collapse" for k in per):
        assert n_collapse >= 1
        assert any(
            e.get("add_faces") is not None and len(e["remove_faces"]) > 1
            for _, e in edits
        )
    want = {
        "two_sheets": {"gap", "cross"},
        "offset_sheets": {"gap"},
        "sphere": {"cross", "crease"},
    }[mesh]
    assert want <= reasons
    rng = np.random.default_rng(7)

    def variants() -> list[tuple[str, MC.Checker, np.ndarray]]:
        out = []
        # (a) the face rows permuted: old row -> new row
        p = rng.permutation(len(F))
        Fp = np.empty_like(F)
        Fp[p] = F
        out.append(("rows", MC.Checker(MC.Topology.of(Fp, len(V)), V, GAP), p))
        # (b) another order of the faces round each vertex: edits tried and rolled back
        cb = MC.Checker(MC.Topology.of(F, len(V)), V, GAP)
        for v in rng.choice(len(V), min(300, len(V)), replace=False):
            cb.touch([int(v)], (V[v] + [0, 0, 0.05])[None])
        for v in rng.choice(len(V), min(200, len(V)), replace=False):
            fs = cb.faces_of(int(v))
            if len(fs) >= 2:
                cb.touch(remove_faces=[int(fs[0])])
        out.append(("faces", cb, np.arange(len(F))))
        # (c) free vertex and face rows: a split and its rollback, a grown Checker
        cc = MC.Checker(MC.Topology.of(F, len(V)), V, GAP)
        for _ in range(120):
            f = int(rng.integers(len(F)))
            a, b, c = (int(x) for x in F[f])
            cc.touch(
                remove_faces=[f],
                add_faces=[[a, b, -1], [b, c, -1], [c, a, -1]],
                add_vertices=V[[a, b, c]].mean(0)[None],
            )
        out.append(("free", cc, np.arange(len(F))))
        return out

    for name, chk, p in variants():
        for reason, e in edits:
            ref = base.propose(**e)
            expect = (_reason(ref), _triples(base, ref))
            base.rollback()
            # the edit in the variant: the same vertices, the face ids mapped
            e2: dict[str, Any] = dict(e)
            if "remove_faces" in e2:  # the face ids follow the rows
                e2["remove_faces"] = [int(p[x]) for x in e2["remove_faces"]]
            r = chk.propose(**e2)
            got = (_reason(r), _triples(chk, r))
            chk.rollback()
            assert expect[0] == reason, name
            assert got == expect, (name, reason)
