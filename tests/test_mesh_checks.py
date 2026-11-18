"""Tests for baillarger.mesh_checks: exact small cases of each check (a folded
fan that no crossing test finds, a fold at a shared vertex, two sheets at the
gap, turned faces, the link condition), cross-checks against mesh_intersect
and a brute-force fold search on random meshes, and decimate's repair of a
shared-vertex fold."""

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from test_decimate import slabs, sphere

from baillarger import decimate as D
from baillarger import mesh_checks as C
from baillarger import mesh_intersect as MI

type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]


def flat_grid(n: int = 5, z: float = 0.0, up: bool = True) -> tuple[Float, Int]:
    """An n × n grid of unit squares in the plane z, two faces per square,
    normals +z (up) or -z."""
    xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    V = np.c_[xs.ravel(), ys.ravel(), np.full(n * n, z)].astype(np.float64)
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
            F += [[a, b, c], [a, c, d]]
    Fa = np.array(F, np.int64)
    return V, Fa if up else Fa[:, ::-1].copy()


def test_a_flat_grid_passes_every_check() -> None:
    V, F = flat_grid()
    c = C.check_mesh(V, F, reference=C.face_normals(V, F), gap=0.3)
    assert c.counts() == dict(
        intersecting_faces=0,
        intersecting_pairs=0,
        shared_vertex_fold_faces=0,
        shared_vertex_fold_pairs=0,
        crease_faces=0,
        creases=0,
        turned_faces=0,
        gap_vertices=0,
        gap_pairs=0,
    )


def test_a_folded_fan_is_found_by_the_edge_and_turn_tests_only() -> None:
    """The centre of a flat grid pushed past its 1-ring, in the plane: two
    faces flip onto their neighbours. All faces lie in one plane, so neither
    mesh_intersect nor the shared-vertex test finds them; the dihedral test
    (normals opposed, cos -1) and the turn test do."""
    V0, F = flat_grid()
    V = V0.copy()
    V[12, :2] += [1.4, 0.3]
    flipped = np.flatnonzero(C.face_normals(V, F)[:, 2] < 0)
    assert len(flipped) == 2
    c = C.check_mesh(V, F, reference=C.face_normals(V0, F))
    assert c.intersecting_pairs == 0 and c.vertex_fold_pairs == 0
    assert c.turned is not None
    np.testing.assert_array_equal(np.flatnonzero(c.turned), flipped)
    # every crease has one flipped face and one face that did not flip
    _, P = C.edge_face_pairs(F, len(V))
    folded = P[np.isin(P, flipped).sum(1) == 1]
    assert c.edge_fold_pairs == len(folded) > 0
    np.testing.assert_array_equal(np.flatnonzero(c.edge_folds), np.unique(folded))


def test_the_fold_cut_off() -> None:
    """Two faces on an edge at a dihedral angle: a fold past the cut-off only."""
    for deg, at_default in ((150.0, False), (160.0, True), (179.0, True)):
        t = np.radians(180.0 - deg)
        V = np.array(
            [[0, 0, 0], [0, 1, 0], [1, 0.5, 0], [np.cos(t), 0.5, np.sin(t)]], float
        )
        F = np.array([[0, 1, 2], [1, 0, 3]])  # consistently oriented
        assert C.shared_edge_folds(V, F)[1] == int(at_default)
        assert C.shared_edge_folds(V, F, fold_cos=np.cos(np.radians(deg - 1)))[1]
        assert not C.shared_edge_folds(V, F, fold_cos=np.cos(np.radians(deg + 1)))[1]


def shared_vertex_pair(cross: bool) -> tuple[Float, Int]:
    """Two faces with the vertex 0 in common: g's far edge crosses f (cross)
    or passes beside it."""
    x = 1.0 if cross else -1.0
    V = np.array([[0, 0, 0], [2, -1, 0], [2, 1, 0], [x, 0, -1], [x, 0, 1]], np.float64)
    return V, np.array([[0, 1, 2], [0, 3, 4]], np.int64)


def test_a_fold_at_a_shared_vertex() -> None:
    V, F = shared_vertex_pair(cross=True)
    assert MI.intersecting_faces(V, F)[1] == 0  # skipped by design
    m, n = C.shared_vertex_folds(V, F)
    assert n == 1 and m.all()
    V, F = shared_vertex_pair(cross=False)
    m, n = C.shared_vertex_folds(V, F)
    assert n == 0 and not m.any()


def test_faces_with_a_common_edge_are_not_shared_vertex_folds() -> None:
    """Even folded flat on each other: that is a crease."""
    V = np.array([[0, 0, 0], [0, 1, 0], [1, 0.5, 0], [0.9, 0.5, 1e-3]], float)
    F = np.array([[0, 1, 2], [1, 0, 3]])
    assert C.shared_vertex_folds(V, F)[1] == 0
    assert C.shared_edge_folds(V, F)[1] == 1


@pytest.mark.parametrize(("d", "facing"), [(0.2, True), (0.2, False), (0.4, True)])
def test_two_sheets_at_the_gap(d: float, facing: bool) -> None:
    """Two parallel grids d apart; gap 0.3. Closer than the gap: every vertex
    is in a violation, and `facing` where the sheets' normals are opposed.
    The sheet's own faces (its local patch) never count."""
    V1, F1 = flat_grid(4, 0.0, up=True)
    V2, F2 = flat_grid(4, d, up=not facing)
    V = np.r_[V1, V2]
    F = np.r_[F1, F2 + len(V1)]
    top = C.Topology.of(F, len(V))
    fac = np.zeros(len(V), bool)
    bad, n, _ = C.gap_check(top, V, 0.3, fac)
    if d < 0.3:
        assert bad.all() and n > 0
        assert fac.all() == facing and fac.any() == facing
    else:
        assert not bad.any() and n == 0
    # one sheet alone: no violation at any gap below the patch radius
    bad1, n1, _ = C.gap_check(C.Topology.of(F1, len(V1)), V1, 0.45)
    assert n1 == 0 and not bad1.any()


def test_turned_faces_since_the_last_state_and_against_a_reference() -> None:
    """A face turned by 80 degrees in each of two steps: not turned since
    the last state either time, turned against the first state."""
    V0 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float64)
    F = np.array([[0, 1, 2]], np.int64)

    def rot(V: Float, deg: float) -> Float:
        t = np.radians(deg)
        R = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
        out: Float = V @ R.T
        return out

    V1 = rot(V0, 80)
    V2 = rot(V1, 80)
    assert not C.turned_faces(V1, F, C.face_normals(V0, F)).any()
    assert not C.turned_faces(V2, F, C.face_normals(V1, F)).any()
    assert C.turned_faces(V2, F, C.face_normals(V0, F)).all()
    # a face of zero area counts as turned
    Vz = V0.copy()
    Vz[2] = [2, 0, 0]
    assert C.turned_faces(Vz, F, C.face_normals(V0, F)).all()


def test_the_link_condition() -> None:
    V, F = flat_grid(4)
    top = C.Topology.of(F, len(V))
    E, _ = C.edge_face_pairs(F, len(V))
    ok = C.link_condition(top.ptr, top.nbr, E)
    # interior edges of a flat grid: two common neighbours
    assert ok.all()
    # a boundary edge has one; two faces more round (5, 6) give it four
    assert not C.link_condition(top.ptr, top.nbr, np.array([[0, 1]]))[0]
    F2 = np.r_[F, [[5, 16, 17], [6, 17, 16]]]
    t2 = C.Topology.of(F2, 18)
    assert not C.link_condition(t2.ptr, t2.nbr, np.array([[5, 6]]))[0]


def test_edge_face_pairs_on_a_closed_mesh() -> None:
    v, f = sphere()
    F = np.asarray(f, np.int64)
    E, P = C.edge_face_pairs(F, len(v))
    assert len(E) == len(P) == 3 * len(F) // 2
    shared = (F[P[:, 0], :, None] == F[P[:, 1], None, :]).sum((1, 2))
    assert (shared == 2).all()
    assert C.check_mesh(v, F).counts()["creases"] == 0


def random_mesh(seed: int) -> tuple[Float, Int]:
    """A random soup of faces on few vertices: many shared vertices and
    crossings."""
    rng = np.random.default_rng(seed)
    V = rng.random((25, 3))
    F = np.array([rng.choice(25, 3, replace=False) for _ in range(60)], np.int64)
    return V, F


def brute_vertex_folds(V: Float, F: Int) -> tuple[set[int], int]:
    """Every pair of faces with exactly one common vertex x, the same rule."""
    faces: set[int] = set()
    n = 0
    for f in range(len(F)):
        for g in range(f + 1, len(F)):
            common = set(F[f].tolist()) & set(F[g].tolist())
            if len(common) != 1:
                continue
            (x,) = common
            ef = [int(u) for u in F[f] if u != x]
            eg = [int(u) for u in F[g] if u != x]
            hit = C._seg_tri.py_func(V, *ef, *F[g], C.SEG_TOL)  # type: ignore[attr-defined]
            hit = hit or C._seg_tri.py_func(V, *eg, *F[f], C.SEG_TOL)  # type: ignore[attr-defined]
            if hit:
                faces |= {f, g}
                n += 1
    return faces, n


@pytest.mark.parametrize("seed", range(5))
def test_random_meshes_against_mesh_intersect_and_brute_force(seed: int) -> None:
    V, F = random_mesh(seed)
    c = C.check_mesh(V, F)
    m, n = MI.intersecting_faces(V, F)
    np.testing.assert_array_equal(c.intersecting, m)
    assert c.intersecting_pairs == n
    faces, nf = brute_vertex_folds(V, F)
    assert c.vertex_fold_pairs == nf
    assert set(np.flatnonzero(c.vertex_folds).tolist()) == faces
    # the Checker's own pair search (the local path) finds the same crossings
    chk = C.Checker(C.Topology.of(F, len(V)), V, gap=0.0)
    T, rad = chk._ready(V)
    out = np.zeros(len(F), bool)
    k = C._tri_kernel(
        np.ascontiguousarray(T), F, chk.order, chk.skeys, chk.lo, chk.cell,
        chk.dims, rad, np.ones(len(F), bool), True, out,
    )  # fmt: skip
    np.testing.assert_array_equal(out, m)
    assert k == n


def test_moved_vertices_find_the_same_folds() -> None:
    """With `moved`, the folds round the moved vertices and their neighbours."""
    V, F = random_mesh(3)
    full, _ = C.shared_vertex_folds(V, F)
    top = C.Topology.of(F, len(V))
    moved = np.zeros(len(V), bool)
    moved[F[np.flatnonzero(full)].ravel()] = True
    local, _ = C.shared_vertex_folds(V, F, top, moved)
    np.testing.assert_array_equal(local, full)


def test_decimate_pins_a_shared_vertex_fold(monkeypatch: pytest.MonkeyPatch) -> None:
    """On two touching slabs, a repair that sees crossings only leaves faces
    folded at a shared vertex; the repair with mesh_checks pins them, so
    none remain. Creases are counted in every round."""
    v, f = slabs(1)
    kw: dict[str, Any] = dict(eps=0.08, block=1.6, workers=1, log=lambda s: None)
    real = C.check_mesh

    def crossings_only(V: Float, F: Int) -> C.MeshChecks:
        c = real(V, F)
        c.vertex_folds[:] = False
        c.vertex_fold_pairs = 0
        return c

    monkeypatch.setattr(D, "check_mesh", crossings_only)
    old = D.decimate(v, f, **kw)
    assert real(v[old.vertex_ids], old.faces).vertex_fold_pairs > 0
    monkeypatch.setattr(D, "check_mesh", real)
    new = D.decimate(v, f, **kw)
    c = real(v[new.vertex_ids], new.faces)
    assert c.vertex_fold_pairs == 0 and c.intersecting_pairs == 0
    r = new.repairs
    assert any(x["shared_vertex_fold_pairs"] > 0 for x in r)
    assert r[-1]["shared_vertex_fold_pairs"] == 0 and r[-1]["intersecting_faces"] == 0
    assert all("creases" in x for x in r)
    assert r[-1]["creases"] == c.edge_fold_pairs
    assert len(new.pinned) > len(old.pinned)
