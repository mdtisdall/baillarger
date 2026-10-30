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


def test_octahedron_goes_down_to_a_tetrahedron_and_stops() -> None:
    v, f = octahedron()
    out = D.decimate(
        v, f, eps=10.0, block=100.0, passes=1, workers=1, log=lambda s: None
    )
    check(v, f, out, 10.0)
    assert len(out.faces) == 4 and len(out.vertex_ids) == 4


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
