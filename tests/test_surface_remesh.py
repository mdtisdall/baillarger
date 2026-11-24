"""Tests for baillarger.surface_remesh (ported from remesh_seq): a sphere,
a U-shaped plate whose inner faces are closer than the target edge (the
remesher must keep them apart), an open sheet with a box rim (corners stay,
the rim stays on the box), and a slab with a thin fin (the normal check and
the outward bound)."""

from typing import Any

import numpy as np
import pytest

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
