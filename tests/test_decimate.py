"""Tests for baillarger.decimate: the three guarantees (topology, distance,
input vertices) on closed phantom meshes, for any block grid; the link
condition's exact limits; locked vertices; parallel = serial; the distance
functions."""

import numpy as np
import numpy.typing as npt
import pytest

from baillarger import decimate as D
from baillarger.mesh import marching_tetrahedra, topology

type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]


def grid(n: int) -> tuple[Float, Float, Float]:
    x = np.arange(n, dtype=float)
    return np.meshgrid(x, x, x, indexing="ij")


def sphere() -> tuple[Float, Int]:
    X, Y, Z = grid(30)
    F = np.sqrt((X - 14.3) ** 2 + (Y - 15.1) ** 2 + (Z - 14.7) ** 2)
    return marching_tetrahedra(F, 10.0)


def torus() -> tuple[Float, Int]:
    X, Y, Z = grid(34)
    r = np.sqrt((X - 16.6) ** 2 + (Y - 16.2) ** 2)
    F = np.sqrt((r - 10.0) ** 2 + (Z - 16.4) ** 2)
    return marching_tetrahedra(F, 4.0)


def two_spheres() -> tuple[Float, Int]:
    X, Y, Z = grid(34)
    a = np.sqrt((X - 9.3) ** 2 + (Y - 16.1) ** 2 + (Z - 16.7) ** 2)
    b = np.sqrt((X - 24.6) ** 2 + (Y - 16.4) ** 2 + (Z - 16.2) ** 2)
    return marching_tetrahedra(np.minimum(a, b), 6.0)


def octahedron() -> tuple[Float, Int]:
    v = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float
    )
    f = np.array(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ]
    )
    return v, f


def oriented_edges_once(f: Int) -> bool:
    """Every directed edge once and its reverse once: closed, manifold and
    consistently oriented."""
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    k = {tuple(x) for x in e.tolist()}
    return len(k) == len(e) and all((b, a) in k for a, b in k)


def max_input_distance(v: Float, out: D.Decimated) -> float:
    """Brute force: the largest distance from an input vertex to the output
    surface."""
    w, f = v[out.vertex_ids], out.faces
    used = np.unique(np.concatenate([out.vertex_ids, out.cids]))
    best = np.full(len(used), np.inf)
    for t in f:
        a, b, c = (np.broadcast_to(w[i], (len(used), 3)) for i in t)
        best = np.minimum(best, D.triangle_distances(v[used], a, b, c))
    return float(best.max())


def check(v: Float, f: Int, out: D.Decimated, eps: float) -> None:
    t0, t1 = topology(f, len(v)), topology(out.faces, len(out.vertex_ids))
    assert t1 == t0
    assert oriented_edges_once(out.faces)
    # every input vertex is an output vertex or carried by exactly one face
    ids = np.sort(np.concatenate([out.vertex_ids, out.cids]))
    np.testing.assert_array_equal(ids, np.unique(f))
    assert out.max_distance <= eps
    assert max_input_distance(v, out) <= eps + 1e-12


@pytest.mark.parametrize("mesh", [sphere, torus, two_spheres])
@pytest.mark.parametrize("block", [4.0, 9.0, 1000.0])
def test_guarantees_for_any_block_grid(mesh: object, block: float) -> None:
    v, f = mesh()  # type: ignore[operator]
    out = D.decimate(
        v, f, eps=0.3, block=block, passes=2, workers=1, log=lambda s: None
    )
    check(v, f, out, 0.3)
    assert len(out.faces) < 0.5 * len(f)


def test_large_eps_keeps_the_genus() -> None:
    v, f = torus()
    out = D.decimate(
        v, f, eps=100.0, block=1000.0, passes=1, workers=1, log=lambda s: None
    )
    check(v, f, out, 100.0)
    assert len(out.vertex_ids) >= 7  # the smallest triangulated torus


def tent() -> tuple[Float, Int]:
    """A tetrahedron with one face raised slightly at its middle (vertex 4)."""
    v = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1 / 3, 1 / 3, -0.05]], float
    )
    # outward: base 0 2 1 split round 4, sides 0 1 3, 1 2 3, 2 0 3
    f = np.array([[0, 2, 4], [2, 1, 4], [1, 0, 4], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    return v, f


def test_a_tent_goes_down_to_a_tetrahedron_and_stops() -> None:
    """The raised vertex goes; the tetrahedron left cannot be collapsed (the
    link condition)."""
    v, f = tent()
    out = D.decimate(
        v, f, eps=10.0, block=100.0, passes=1, workers=1, log=lambda s: None
    )
    check(v, f, out, 10.0)
    assert len(out.faces) == 4 and len(out.vertex_ids) == 4
    assert (surface_cosines(v, f, out) > 0).all()


def test_the_octahedron_is_kept_whole() -> None:
    """Any collapse of a regular octahedron leaves a face at 90 degrees to the
    surface it stands for, so none is made."""
    v, f = octahedron()
    out = D.decimate(
        v, f, eps=10.0, block=100.0, passes=1, workers=1, log=lambda s: None
    )
    assert len(out.faces) == 8


def test_eps_zero_removes_only_vertices_on_the_output() -> None:
    """Marching tetrahedra puts a few vertices exactly in the plane of their
    neighbours; only those may go."""
    v, f = sphere()
    out = D.decimate(
        v, f, eps=0.0, block=1000.0, passes=1, workers=1, log=lambda s: None
    )
    check(v, f, out, 0.0)
    assert len(out.faces) > 0.99 * len(f)


def test_a_pass_keeps_locked_vertices() -> None:
    """Vertices on a face that spans two blocks survive the pass."""
    v, f = sphere()
    block = 6.0
    o = v.min(0)
    g = np.floor((v - o) / block).astype(np.int64)
    span = np.any(g[f] != g[f][:, :1], axis=(1, 2))
    locked = np.unique(f[span])
    M = D.Mesh(
        v=v,
        f=f,
        Q=D.quadrics(v, f),
        cptr=np.zeros(len(f) + 1, np.int64),
        cids=np.zeros(0, np.int64),
        N=D.normals(v, f),
    )
    M2, s = D.decimate_pass(M, 0.5, block, 0.0, o, workers=1)
    assert s["collapses"] > 0
    assert np.isin(locked, M2.f).all()


def test_parallel_equals_serial() -> None:
    v, f = torus()
    a = D.decimate(v, f, eps=0.3, block=8.0, passes=2, workers=1, log=lambda s: None)
    b = D.decimate(v, f, eps=0.3, block=8.0, passes=2, workers=2, log=lambda s: None)
    np.testing.assert_array_equal(a.vertex_ids, b.vertex_ids)
    np.testing.assert_array_equal(a.faces, b.faces)
    np.testing.assert_array_equal(a.cptr, b.cptr)
    np.testing.assert_array_equal(a.cids, b.cids)


def test_triangle_distance_against_dense_sampling() -> None:
    """Exact distance <= the nearest of a fine sample of the triangle <= exact
    + the sample's spacing; the scalar and vectorised forms agree."""
    rng = np.random.default_rng(3)
    s = np.linspace(0, 1, 201)
    U, W = np.meshgrid(s, s, indexing="ij")
    ok = U + W <= 1
    U, W = U[ok], W[ok]
    for _ in range(200):
        a, b, c = rng.normal(size=(3, 3))
        if rng.random() < 0.2:
            c = a + 0.5 * (b - a) + 1e-3 * rng.normal(size=3)  # thin
        p = rng.normal(size=3) * 2
        pts = a + U[:, None] * (b - a) + W[:, None] * (c - a)
        sampled = float(np.linalg.norm(pts - p, axis=1).min())
        exact = D.point_triangle_distance(tuple(p), tuple(a), tuple(b), tuple(c))
        vec = D.triangle_distances(p[None], a[None], b[None], c[None])[0]
        h = max(np.linalg.norm(b - a), np.linalg.norm(c - a)) / 200
        assert exact <= sampled + 1e-12
        assert sampled <= exact + h
        assert vec == pytest.approx(exact, abs=1e-12)


def test_quadric_is_the_sum_of_squared_plane_distances() -> None:
    v, f = octahedron()
    Q = D.quadrics(v, f)
    p = np.array([0.3, -0.2, 0.7])
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    area = np.linalg.norm(n, axis=1) / 2
    u = n / np.linalg.norm(n, axis=1, keepdims=True)
    d = np.sum(u * (p - v[f[:, 0]]), axis=1)
    for k in range(len(v)):
        mine = np.any(f == k, axis=1)
        want = float(np.sum(area[mine] * d[mine] ** 2))
        assert D._cost(tuple(Q[k]), tuple(p)) == pytest.approx(want)


@pytest.mark.parametrize("mesh", [sphere, torus, two_spheres, octahedron])
@pytest.mark.parametrize(
    ("eps", "block"), [(0.0, 1000.0), (0.3, 4.0), (0.3, 1000.0), (100.0, 9.0)]
)
def test_the_compiled_worker_equals_the_python_reference(
    mesh: object, eps: float, block: float
) -> None:
    """Bit for bit: faces, the carried vertices and their order, the quadrics."""
    v, f = mesh()  # type: ignore[operator]
    M = D.Mesh(
        v=v,
        f=f,
        Q=D.quadrics(v, f),
        cptr=np.zeros(len(f) + 1, np.int64),
        cids=np.zeros(0, np.int64),
        N=D.normals(v, f),
    )
    for off in (0.0, 0.5):  # two passes, the second on carried points
        a, sa = D.decimate_pass(M, eps, block, off, workers=1)
        b, sb = D.decimate_pass(M, eps, block, off, workers=1, reference=True)
        assert sa == sb
        for x, y in ((a.f, b.f), (a.cptr, b.cptr), (a.cids, b.cids)):
            np.testing.assert_array_equal(x, y)
        assert np.array_equal(a.Q, b.Q)  # exactly, not within a tolerance
        M = a


def test_chunks_change_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Quadrics and the final distances in chunks of 7 give the same bits as in
    one chunk; so does the whole decimation."""
    v, f = torus()
    q1 = D.quadrics(v, f)
    a = D.decimate(v, f, eps=0.3, block=8.0, passes=2, workers=1, log=lambda s: None)
    monkeypatch.setattr(D, "CHUNK", 7)
    assert np.array_equal(D.quadrics(v, f), q1)
    b = D.decimate(v, f, eps=0.3, block=8.0, passes=2, workers=1, log=lambda s: None)
    for x, y in ((a.vertex_ids, b.vertex_ids), (a.faces, b.faces), (a.cids, b.cids)):
        np.testing.assert_array_equal(x, y)
    assert a.max_distance == b.max_distance


def surface_cosines(v: Float, f: Int, out: D.Decimated) -> Float:
    vn = np.zeros_like(v)
    nf = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    for k in range(3):
        np.add.at(vn, f[:, k], nf)
    f0 = out.vertex_ids[out.faces]
    S = vn[f0[:, 0]] + vn[f0[:, 1]] + vn[f0[:, 2]]
    np.add.at(S, np.repeat(np.arange(len(f0)), np.diff(out.cptr)), vn[out.cids])
    n = np.cross(v[f0[:, 1]] - v[f0[:, 0]], v[f0[:, 2]] - v[f0[:, 0]])
    out_: Float = np.sum(n * S, 1) / (
        np.linalg.norm(n, axis=1) * np.linalg.norm(S, axis=1)
    )
    return out_


def staircase() -> tuple[Float, Int]:
    """A closed surface like the real white mesh at small scale: marching
    tetrahedra of a binary mask (a ball with bumps), so flat facets and steps."""
    X, Y, Z = grid(34)
    r = np.sqrt((X - 16.3) ** 2 + (Y - 16.6) ** 2 + (Z - 16.2) ** 2)
    a = np.arctan2(Y - 16.6, X - 16.3)
    mask = r < 11 + 2.5 * np.sin(3 * a) * np.cos(0.4 * Z)
    v, f = marching_tetrahedra(mask.astype(float), 0.5)
    return v * 0.16, f


@pytest.mark.parametrize("eps", [0.05, 0.1, 0.2])
def test_no_output_face_faces_away_from_the_surface_it_stands_for(eps: float) -> None:
    """On a staircase like the real data, every output face points the same
    way as the input surface it stands for."""
    v, f = staircase()
    out = D.decimate(v, f, eps=eps, block=1.5, passes=2, workers=1, log=lambda s: None)
    cos = surface_cosines(v, f, out)
    assert np.sum(cos <= 0) == 0, f"{np.sum(cos <= 0)} of {len(cos)} faces flipped"


def slabs(gap: int, n: int = 26) -> tuple[Float, Int]:
    """Two bumpy slabs, one above the other, `gap` voxels apart: at gap 1 their
    facing surfaces are one voxel (0.16 mm) apart and touch at corners (one
    piece of genus > 0), like the necks of the real white surface; at gap 2,
    two pieces."""
    x, y, z = np.indices((n, n, 24))
    h = 10 + np.round(2.5 * np.sin(x / 4.0) * np.cos(y / 5.0)).astype(int)
    inside = (x > 2) & (x < n - 3) & (y > 2) & (y < n - 3)
    a = (z >= 3) & (z < h) & inside
    b = (z >= h + gap) & (z < 21) & inside
    v, f = marching_tetrahedra((a | b).astype(float), 0.5)
    return v * 0.16, f.astype(np.int64)


@pytest.mark.parametrize(
    ("gap", "eps", "n"), [(1, 0.12, 26), (1, 0.2, 26), (2, 0.08, 40)]
)
def test_repair_leaves_no_faces_intersecting(gap: int, eps: float, n: int) -> None:
    """Without repair the facing surfaces cross; with it no faces intersect,
    and the other guarantees hold."""
    from baillarger.mesh_intersect import intersecting_faces

    v, f = slabs(gap, n)
    quiet = dict(block=1.6, workers=1, log=lambda s: None)
    raw = D.decimate(v, f, eps=eps, repair=False, **quiet)  # type: ignore[arg-type]
    assert intersecting_faces(v[raw.vertex_ids], raw.faces)[0].any()
    out = D.decimate(v, f, eps=eps, **quiet)  # type: ignore[arg-type]
    assert not intersecting_faces(v[out.vertex_ids], out.faces)[0].any()
    assert out.repairs[-1]["intersecting_faces"] == 0 and len(out.repairs) >= 2
    check(v, f, out, eps)
    assert (surface_cosines(v, f, out) > 0).all()
    assert np.isin(out.pinned, out.vertex_ids).all()  # pinned vertices are kept


def test_the_repaired_result_is_the_result_with_its_pins() -> None:
    """Reusing unchanged blocks changes nothing: the repaired result equals a
    decimation with its pinned vertices given from the start, serial or not."""
    v, f = slabs(1)
    quiet = dict(block=1.6, log=lambda s: None)
    out = D.decimate(v, f, eps=0.12, workers=1, **quiet)  # type: ignore[arg-type]
    for workers in (1, 2):
        again = D.decimate(
            v,
            f,
            eps=0.12,
            workers=workers,
            repair=False,
            pinned=out.pinned,
            **quiet,  # type: ignore[arg-type]
        )
        for x, y in (
            (out.vertex_ids, again.vertex_ids),
            (out.faces, again.faces),
            (out.cids, again.cids),
        ):
            np.testing.assert_array_equal(x, y)
    reused = [s["reused"] for s in out.passes]
    assert sum(reused) > 0


def test_closest_points_barycentrics() -> None:
    """The barycentrics of the closest point: non-negative, summing to 1, and
    rebuilding the point at the distance found, which is triangle_distances'
    and the scalar point_triangle_distance's (to rounding)."""
    rng = np.random.default_rng(5)
    k = 20000
    p, a, b, c = (rng.normal(size=(k, 3)) * 3 for _ in range(4))
    thin = rng.random(k) < 0.2
    c[thin] = (
        a[thin] + 0.5 * (b[thin] - a[thin]) + 1e-3 * rng.normal(size=(thin.sum(), 3))
    )
    d, b0, b1, b2 = D.closest_points(p, a, b, c)
    np.testing.assert_array_equal(d, D.triangle_distances(p, a, b, c))
    assert min(b0.min(), b1.min(), b2.min()) >= -1e-12
    np.testing.assert_allclose(b0 + b1 + b2, 1, atol=1e-12)
    q = b0[:, None] * a + b1[:, None] * b + b2[:, None] * c
    np.testing.assert_allclose(np.linalg.norm(p - q, axis=1), d, atol=1e-9)
    for i in range(0, k, 997):
        s = D.point_triangle_distance(
            tuple(p[i]), tuple(a[i]), tuple(b[i]), tuple(c[i])
        )
        assert s == pytest.approx(d[i], abs=1e-12)


def test_reorder_keeps_each_face_with_its_carried_vertices() -> None:
    v, f = torus()
    out = D.decimate(v, f, eps=0.3, block=8.0, workers=1, log=lambda s: None)
    o = np.random.default_rng(2).permutation(len(out.faces))
    r = D.reorder(out, o)
    np.testing.assert_array_equal(r.faces, out.faces[o])
    assert r.repairs == out.repairs and r.passes == out.passes
    np.testing.assert_array_equal(r.pinned, out.pinned)
    for new, old in ((0, o[0]), (7, o[7]), (len(o) - 1, o[-1])):
        a = r.cids[r.cptr[new] : r.cptr[new + 1]]
        b = out.cids[out.cptr[old] : out.cptr[old + 1]]
        np.testing.assert_array_equal(a, np.sort(b))
    assert np.all(
        np.diff(r.cids)[np.diff(np.repeat(np.arange(len(o)), np.diff(r.cptr))) == 0] > 0
    )


def carried_geometry(v: Float, out: D.Decimated) -> tuple[Int, Float, Float, Float]:
    """Per carried vertex: its face (into out.faces), the face's normal, the
    distance and the rebuilt closest point."""
    face = np.repeat(np.arange(len(out.faces)), np.diff(out.cptr))
    t = out.vertex_ids[out.faces[face]]
    a, b, c = (v[t[:, k]] for k in range(3))
    d, b0, b1, b2 = D.closest_points(v[out.cids], a, b, c)
    q = b0[:, None] * a + b1[:, None] * b + b2[:, None] * c
    return face, np.cross(b - a, c - a), d, q


def test_on_a_plane_the_carried_points_are_on_their_faces() -> None:
    n = 30
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    v = np.c_[i.ravel() * 0.25, j.ravel() * 0.25, np.zeros(n * n)]
    F = []
    for x in range(n - 1):
        for y in range(n - 1):
            p, q, r, s = x * n + y, (x + 1) * n + y, (x + 1) * n + y + 1, x * n + y + 1
            F += [(p, q, r), (p, r, s)]
    f = np.array(F, np.int64)
    out = D.decimate(v, f, eps=1e-9, block=100.0, workers=1, log=lambda s: None)
    assert len(out.faces) < 0.2 * len(f)
    _, _, d, rebuilt = carried_geometry(v, out)
    assert len(d) == n * n - len(out.vertex_ids)
    assert d.max() <= 1e-12
    np.testing.assert_allclose(rebuilt, v[out.cids], atol=1e-12)


def test_on_a_thin_blade_each_vertex_is_carried_on_its_own_side() -> None:
    """A blade 2 voxels (0.32 mm) thick, decimated within 0.1 mm: each carried
    vertex's face has a normal on the same side as the vertex's own, and in
    the blade's middle the face lies on the vertex's sheet."""
    from baillarger.mesh_smoothing import vertex_normals

    wm = np.zeros((40, 40, 20), bool)
    wm[19:21, 4:30, :] = True
    v, f = marching_tetrahedra(wm.astype(float), 0.5)
    v = v * 0.16
    out = D.decimate(v, f, eps=0.1, block=2.0, workers=1, log=lambda s: None)
    assert len(out.faces) < 0.5 * len(f)
    face, fn, d, _ = carried_geometry(v, out)
    vn = vertex_normals(v, f)[out.cids]
    assert np.all(np.sum(fn * vn, axis=1) > 0)
    mid = (v[out.cids, 1] > 8 * 0.16) & (v[out.cids, 1] < 26 * 0.16)
    centre = v[out.vertex_ids[out.faces[face]]].mean(1)
    side = v[out.cids, 0] < 19.5 * 0.16
    assert np.array_equal(side[mid], centre[mid, 0] < 19.5 * 0.16)
