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
    # the Checker's own local search (every face changed) finds the same
    # crossings and folds
    chk = C.Checker(C.Topology.of(F, len(V)), V, gap=0.0)
    r = chk.propose(move=np.arange(len(V)), to=V)
    assert r.intersecting_pairs == n and r.vertex_fold_pairs == nf
    want = np.flatnonzero(m | c.vertex_folds)
    np.testing.assert_array_equal(np.sort(r.cross_faces), want)
    chk.rollback()


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


# ------------------------------------------------------------ the live Checker


def audit(chk: C.Checker) -> None:
    """The Checker's live structures agree with its faces: the faces round
    each vertex, and every live face once in the grid, in its centroid's
    cell."""
    F = chk.F
    live = np.flatnonzero(F[:, 0] >= 0)
    ptr, vf = C.vertex_faces(F[live], len(chk.V))
    for v in range(len(chk.V)):
        want = live[vf[ptr[v] : ptr[v + 1]]]
        np.testing.assert_array_equal(np.sort(chk.faces_of(v)), want)
    seen = []
    for s in np.flatnonzero(chk._hkey >= 0):
        g = int(chk._hhead[s])
        while g >= 0:
            seen.append(g)
            assert chk._fcell[g] == s
            c = chk.V[F[g]].mean(0)
            q = np.floor((c - chk._lo) / chk.cell).astype(np.int64) + C._KOFF
            assert chk._hkey[s] == (q[0] << 42) | (q[1] << 21) | q[2]
            g = int(chk._gnext[g])
    assert sorted(seen) == live.tolist()


def neighbours(chk: C.Checker, v: int) -> set[int]:
    return {int(u) for f in chk.faces_of(v) for u in chk.F[f]} - {v}


def edge_faces(chk: C.Checker, a: int, b: int) -> list[int]:
    return [int(f) for f in chk.faces_of(a) if b in chk.F[f]]


def random_edit(
    chk: C.Checker, rng: np.random.Generator, step: float
) -> dict[str, Any] | None:
    """A random move, edge flip, edge split or edge collapse on the live
    mesh, as propose() arguments (None if the chosen edge does not allow
    it: not two faces, or the link condition fails)."""
    F = chk.F
    live = np.flatnonzero(F[:, 0] >= 0)
    kind = rng.choice(["move", "flip", "split", "collapse"])
    f = int(rng.choice(live))
    k = int(rng.integers(3))
    a, b = int(F[f, k]), int(F[f, (k + 1) % 3])
    if kind == "move":
        return dict(move=[a], to=chk.V[[a]] + rng.normal(0, step, (1, 3)))
    ef = edge_faces(chk, a, b)
    if len(ef) != 2:
        return None
    g = ef[0] if ef[1] == f else ef[1]
    c = int(F[f, (k + 2) % 3])
    d = int(next(u for u in F[g] if u not in (a, b)))
    if kind == "flip":
        if d in neighbours(chk, c):
            return None
        return dict(remove_faces=[f, g], add_faces=[[c, a, d], [c, d, b]])
    if kind == "split":
        m = 0.5 * (chk.V[a] + chk.V[b]) + rng.normal(0, 0.2 * step, 3)
        return dict(
            add_vertices=[m],
            remove_faces=[f, g],
            add_faces=[[a, -1, c], [-1, b, c], [b, -1, d], [-1, a, d]],
        )
    if len(neighbours(chk, a) & neighbours(chk, b)) != 2:
        return None
    rest = [int(h) for h in chk.faces_of(b) if h not in (f, g)]
    Fr = F[rest].copy()
    Fr[Fr == b] = a
    return dict(
        move=[a],
        to=[0.5 * (chk.V[a] + chk.V[b])],
        remove_faces=[f, g, *rest],
        add_faces=Fr,
    )


def whole(chk: C.Checker, gap: float) -> dict[str, Any]:
    """check_mesh on the live mesh, in the Checker's ids."""
    V, F, new = chk.export()
    old = np.flatnonzero(new >= 0)
    live = np.flatnonzero(chk.F[:, 0] >= 0)
    c = C.check_mesh(V, F, gap=gap)
    assert c.gap is not None
    cross = c.intersecting | c.vertex_folds
    return dict(
        bad=set(old[c.gap].tolist()) | set(old[F[cross].ravel()].tolist()),
        facing=set(old[c.gap_facing].tolist()) if c.gap_facing is not None else set(),
        gap_pairs=c.gap_pairs,
        intersecting_pairs=c.intersecting_pairs,
        vertex_fold_pairs=c.vertex_fold_pairs,
        creases=c.edge_fold_pairs,
        crease_faces=set(live[c.edge_folds].tolist()),
    )


@pytest.mark.parametrize(
    ("mesh", "gap", "step", "min_gaps"),
    [("sphere", 0.3, 0.8, 0), ("slabs", 0.15, 0.2, 5)],
)
def test_local_checks_of_random_edits_match_the_whole_mesh(
    mesh: str, gap: float, step: float, min_gaps: int
) -> None:
    """Random moves, flips, splits and collapses on a valid mesh, each
    proposed and checked locally: the violations equal those of check_mesh
    on the whole edited mesh (all are new, as the mesh before the edit had
    none). Valid edits are kept, the others rolled back; a rollback gives
    back the mesh exactly. The live structures stay consistent."""
    rng = np.random.default_rng(1)
    V, F = sphere() if mesh == "sphere" else slabs(2, 16)
    chk = C.Checker(C.Topology.of(F, len(V)), V, gap)
    assert chk.check_all().ok
    kinds = {True: 0, False: 0}
    gaps = 0
    tried = 0
    while tried < 150:
        e = random_edit(chk, rng, step)
        if e is None:
            continue
        tried += 1
        before = chk.export()
        r = chk.propose(**e)
        w = whole(chk, gap)
        assert set(r.bad.tolist()) == w["bad"]
        assert set(r.facing.tolist()) == w["facing"]
        assert set(r.crease_faces.tolist()) == w["crease_faces"]
        for key in ("gap_pairs", "intersecting_pairs", "vertex_fold_pairs", "creases"):
            assert getattr(r, key) == w[key], key
        kinds[r.ok] += 1
        gaps += r.gap_pairs > 0
        if r.ok:
            chk.commit()
        else:
            chk.rollback()
            after = chk.export()
            np.testing.assert_array_equal(before[0], after[0])
            np.testing.assert_array_equal(before[1], after[1])
            k = len(before[2])  # a freed new vertex may stay above the old ids
            np.testing.assert_array_equal(before[2], after[2][:k])
            assert (after[2][k:] == -1).all()
    assert kinds[True] > 25 and kinds[False] > 10 and gaps >= min_gaps  # all paths ran
    audit(chk)
    assert chk.check_all().ok


def test_a_hole_that_parts_a_patch_is_found() -> None:
    """Two strips 0.2 apart, joined by a fold along one edge: within one
    patch, so no gap violation at gap 0.3. Removing the fold's faces leaves
    two parts 0.2 apart: the local check of that edit (no vertex moved)
    finds every violation of the whole check."""
    n = 8
    x = np.arange(n, dtype=float) * 0.25
    rows = [(0.0, 0.0), (0.25, 0.0), (0.25, 0.2), (0.0, 0.2)]  # (y, z)
    V = np.array([(xi, y, z) for y, z in rows for xi in x])
    F = []
    for r in range(3):
        for i in range(n - 1):
            a, b = r * n + i, r * n + i + 1
            c, d = (r + 1) * n + i + 1, (r + 1) * n + i
            F += [[a, b, c], [a, c, d]]
    Fa = np.array(F, np.int64)
    chk = C.Checker(C.Topology.of(Fa, len(V)), V, 0.3)
    assert chk.check_all().ok
    fold = np.arange(2 * (n - 1), 4 * (n - 1))  # the faces between rows 1 and 2
    res = chk.propose(remove_faces=fold)
    w = whole(chk, 0.3)
    assert w["gap_pairs"] > 0
    assert set(res.bad.tolist()) == w["bad"] and res.gap_pairs == w["gap_pairs"]


def test_the_cost_of_a_local_check_does_not_grow_with_the_mesh() -> None:
    """One vertex moved in the middle of a flat grid: the affected vertices
    and the faces checked are the same for a 9 × 9 and a 61 × 61 grid."""
    out = []
    for n in (9, 61):
        V, F = flat_grid(n)
        chk = C.Checker(C.Topology.of(F, len(V)), V, 0.3)
        mid = (n // 2) * n + n // 2
        r = chk.propose(move=[mid], to=V[[mid]] + [0.1, 0.05, 0.2])
        xy = np.sort(V[r.affected, :2] - V[mid, :2], axis=0)
        out.append((len(r.affected), xy))
        chk.rollback()
    assert out[0][0] == out[1][0]
    np.testing.assert_allclose(out[0][1], out[1][1])


def test_closest_points_on_the_live_mesh_match_brute_force() -> None:
    """Checker.closest after some edits: the closest point, distance and
    face among the faces within r, as a search over every live face."""
    rng = np.random.default_rng(4)
    V, F = sphere()
    chk = C.Checker(C.Topology.of(F, len(V)), V, 0.3)
    done = 0
    while done < 30:
        e = random_edit(chk, rng, 0.4)
        if e is None:
            continue
        r = chk.propose(**e)
        if r.ok:
            chk.commit()
            done += 1
        else:
            chk.rollback()
    P = rng.uniform(V.min(0) - 2, V.max(0) + 2, (300, 3))
    Q, d, face = chk.closest(P, 1.5)
    live = np.flatnonzero(chk.F[:, 0] >= 0)
    T = chk.V[chk.F[live]]
    for i in range(len(P)):
        dd = np.array([C._ptd_xyz.py_func(*P[i], *t[0], *t[1], *t[2]) for t in T])  # type: ignore[attr-defined]
        k = int(np.argmin(dd))
        if dd[k] >= 1.5:
            assert face[i] == -1 and np.isinf(d[i])
        else:
            assert np.isclose(d[i], dd[k]) and np.isclose(
                np.linalg.norm(Q[i] - P[i]), d[i]
            )


def test_edge_folds_and_pair_cosines_agree_with_shared_edge_folds() -> None:
    # two faces folded 10 degrees apart (normals 170 degrees apart), then
    # one of zero area
    t = np.radians(10.0)
    V = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [np.cos(t), 0, np.sin(t)]], float)
    F = np.array([[0, 1, 2], [1, 0, 3]], np.int64)
    _, P = C.edge_face_pairs(F, len(V))
    assert np.allclose(np.degrees(np.arccos(C.pair_cosines(V, F, P))), 170.0)
    assert C.edge_folds(V, F, P).tolist() == [True]
    assert C.shared_edge_folds(V, F)[1] == 1
    assert not C.edge_folds(V, F, P, fold_cos=np.cos(np.radians(171))).any()
    V[3] = V[0]  # zero area: NaN cosine, no fold
    assert np.isnan(C.pair_cosines(V, F, P)).all()
    assert not C.edge_folds(V, F, P, fold_cos=1.0).any()
