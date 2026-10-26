"""Tests for baillarger.mesh_intersect: correctness on hand-made and test
meshes, and identical results to the numpy reference (mesh_intersect_ref)."""

from collections.abc import Callable
from fractions import Fraction
from typing import Any

import mesh_intersect_ref as ref
import numpy as np
import numpy.typing as npt
from skimage import measure

from baillarger import mesh_intersect as mi

type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]


def tri(*pts: tuple[float, float, float]) -> Float:
    return np.array(pts, float)[None]


def sphere(n: int = 30) -> tuple[Float, Int]:
    x = np.linspace(-1, 1, n)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    V, F, _, _ = measure.marching_cubes(X**2 + Y**2 + Z**2, 0.5)
    # float64: the reference tests the pairs in the vertices' dtype, the numba
    # version in float64
    return V.astype(np.float64), F.astype(np.int64)


def folded() -> tuple[Float, Int]:
    """The sphere with its top cap pushed down and sideways through the rest."""
    V, F = sphere()
    c = V.mean(0)
    W = V.copy()
    W[W[:, 2] > c[2] + 3.1] += [6.3, 0.17, -5.2]
    return W, F


def same_faces(r: tuple[Any, ...], n: tuple[Any, ...]) -> None:
    """Masks and counts equal (the tuples of intersecting_faces)."""
    assert len(r) == len(n)
    for a, b in zip(r, n, strict=True):
        if isinstance(a, np.ndarray):
            assert a.dtype == b.dtype and np.array_equal(a, b)
        else:
            assert a == b


# ---------------------------------------------------------------- correctness


def test_crossing_triangles_intersect() -> None:
    A = tri((0, 0, 0), (2, 0, 0), (0, 2, 0))
    B = tri((0.5, 0.5, -1), (0.5, 0.5, 1), (1.5, 0.2, 0.3))
    assert mi.tri_tri(A, B)[0]


def test_separated_and_parallel_triangles_do_not() -> None:
    A = tri((0, 0, 0), (2, 0, 0), (0, 2, 0))
    # parallel planes
    assert not mi.tri_tri(A, tri((0, 0, 1), (2, 0, 1), (0, 2, 1)))[0]
    # plane crossed outside A
    assert not mi.tri_tri(A, tri((3, 3, -1), (3, 3, 1), (4, 3, 0)))[0]
    # above A
    assert not mi.tri_tri(A, tri((0.5, 0.5, 0.1), (0.5, 0.5, 1), (1.5, 0.2, 0.3)))[0]


def test_marching_cubes_sphere_has_no_self_intersection() -> None:
    V, F = sphere(40)
    mask, n = mi.intersecting_faces(V, F)
    assert n == 0 and not mask.any()


def test_folded_mesh_and_two_meshes() -> None:
    # generic offsets: an integer shift would put the copy's vertices on the
    # same grid lines as the original's, which makes touching contacts, not
    # crossings
    W, F = folded()
    _, n = mi.intersecting_faces(W, F)
    assert n > 0
    # a sphere and a shifted copy cross; a sphere and a smaller concentric one
    # do not
    V, F = sphere()
    c = V.mean(0)
    _, _, n2 = mi.intersecting_faces(V, F, V + [3.37, 0.21, 0.13], F)
    assert n2 > 0
    _, _, n3 = mi.intersecting_faces(V, F, c + 0.53 * (V - c), F)
    assert n3 == 0


def _seg_tri(p: Float, q: Float, T: Float) -> bool:
    """Whether the segment pq crosses triangle T (Moller-Trumbore)."""
    e1, e2 = T[1] - T[0], T[2] - T[0]
    d = q - p
    h = np.cross(d, e2)
    a = e1 @ h
    if abs(a) < 1e-14:
        return False
    s = p - T[0]
    u = (s @ h) / a
    qq = np.cross(s, e1)
    v = (d @ qq) / a
    t = (e2 @ qq) / a
    return bool(0 <= u <= 1 and v >= 0 and u + v <= 1 and 0 <= t <= 1)


def test_random_pairs_match_brute_force() -> None:
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3000, 3, 3))
    B = rng.normal(size=(3000, 3, 3)) * 0.8 + 0.3
    fast = mi.tri_tri(A, B)
    slow = np.array(
        [
            any(_seg_tri(a[i], a[(i + 1) % 3], b) for i in range(3))
            or any(_seg_tri(b[i], b[(i + 1) % 3], a) for i in range(3))
            for a, b in zip(A, B, strict=True)
        ]
    )
    assert slow.sum() > 100
    assert (fast == slow).mean() == 1.0


def test_stretched_triangles_are_found() -> None:
    # one long sliver crossing a small sphere: it spans many grid cells
    V, F = sphere()
    c = V.mean(0)
    sliver_v = np.array(
        [c + [-40, 0.13, 0.27], c + [40, 0.31, 0.11], c + [0.2, 0.17, 40]]
    )
    _, _, n = mi.intersecting_faces(V, F, sliver_v, np.array([[0, 1, 2]]))
    assert n > 0


def test_faces_hitting_matches_full_test() -> None:
    W, F = folded()
    full, _ = mi.intersecting_faces(W, F)
    rng = np.random.default_rng(1)
    sub = rng.choice(len(F), 400, replace=False)
    sub = np.union1d(sub, np.where(full)[0][:50])
    assert np.array_equal(mi.faces_hitting(W, F, sub), full[sub])
    # against another mesh
    V, F = sphere()
    m1, _, _ = mi.intersecting_faces(V, F, V + [3.37, 0.21, 0.13], F)
    assert np.array_equal(
        mi.faces_hitting(V, F, sub, V + [3.37, 0.21, 0.13], F), m1[sub]
    )


# ---------------------------------------------------------------- against the reference


def random_pairs(rng: np.random.Generator, m: int) -> tuple[Float, Float]:
    """Random triangle pairs, a quarter of them each: general, scaled over many
    orders of magnitude, on a coarse grid (coplanar and touching cases), and
    pairs that share a vertex or an edge or lie near each other's plane."""
    q = m // 4
    A = [rng.normal(size=(q, 3, 3))]
    B = [rng.normal(size=(q, 3, 3)) * 0.8 + 0.3]
    s = np.exp(rng.normal(size=(q, 1, 1)) * 4)
    A.append(rng.normal(size=(q, 3, 3)) * s)
    B.append(rng.normal(size=(q, 3, 3)) * s * 0.8)
    A.append(rng.integers(-2, 3, size=(q, 3, 3)).astype(float))
    B.append(rng.integers(-2, 3, size=(q, 3, 3)).astype(float))
    a = rng.normal(size=(m - 3 * q, 3, 3))
    b = rng.normal(size=(m - 3 * q, 3, 3))
    k = len(a) // 3
    b[:k, 0] = a[:k, 0]  # one shared vertex
    b[k : 2 * k, :2] = a[k : 2 * k, :2]  # one shared edge
    # B's first vertex very near A's plane: distances around the tolerance
    n = np.cross(a[2 * k :, 1] - a[2 * k :, 0], a[2 * k :, 2] - a[2 * k :, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    b[2 * k :, 0] = (
        a[2 * k :, 0]
        + rng.uniform(0.2, 0.8, (len(n), 1)) * (a[2 * k :, 1] - a[2 * k :, 0])
        + n * rng.normal(size=(len(n), 1)) * 1e-4
    )
    A.append(a)
    B.append(b)
    return np.concatenate(A), np.concatenate(B)


def test_tri_tri_random_pairs_match() -> None:
    rng = np.random.default_rng(0)
    A, B = random_pairs(rng, 100000)
    r = ref.tri_tri(A, B)
    new = mi.tri_tri(A, B)
    assert new.dtype == bool and new.shape == r.shape
    bad = np.where(r != new)[0]
    assert len(bad) == 0, f"{len(bad)} pairs differ, e.g. {bad[:10]}"
    assert 1000 < r.sum() < len(r) - 1000
    # another tolerance
    assert np.array_equal(
        ref.tri_tri(A, B, rel_tol=1e-2), mi.tri_tri(A, B, rel_tol=1e-2)
    )


def test_tri_tri_hand_cases() -> None:
    A = tri((0, 0, 0), (2, 0, 0), (0, 2, 0))
    for B in (
        tri((0.5, 0.5, -1), (0.5, 0.5, 1), (1.5, 0.2, 0.3)),
        tri((0, 0, 1), (2, 0, 1), (0, 2, 1)),
        tri((3, 3, -1), (3, 3, 1), (4, 3, 0)),
        tri((0.5, 0.5, 0.1), (0.5, 0.5, 1), (1.5, 0.2, 0.3)),
    ):
        assert mi.tri_tri(A, B)[0] == ref.tri_tri(A, B)[0]
    assert len(mi.tri_tri(np.zeros((0, 3, 3)), np.zeros((0, 3, 3)))) == 0


def _fma_exact(a: float, b: float, c: float) -> float:
    """a * b + c with one rounding, computed exactly with fractions."""
    return float(Fraction(a) * Fraction(b) + Fraction(c))


def test_dot_is_two_fused_multiply_adds() -> None:
    # fma(x2, y2, fma(x1, y1, x0 * y0)): the order of the prototype's
    # np.einsum("ij,ij->i"), which the reference uses. Newer numpy builds can sum
    # in another order, so the check is against exact arithmetic, not einsum.
    rng = np.random.default_rng(3)
    x = rng.normal(size=(3000, 3)) * np.exp(rng.normal(size=(3000, 3)) * 3)
    y = rng.normal(size=(3000, 3))
    want = np.array(
        [
            _fma_exact(a[2], b[2], _fma_exact(a[1], b[1], float(a[0] * b[0])))
            for a, b in zip(x.tolist(), y.tolist(), strict=True)
        ]
    )
    got = mi._dot_rows(x, y)
    assert np.array_equal(got, want)
    # and it is not the plain sum: the fused adds change the last bits
    plain = (x[:, 0] * y[:, 0] + x[:, 1] * y[:, 1]) + x[:, 2] * y[:, 2]
    assert not np.array_equal(got, plain)


def test_candidate_pairs_match() -> None:
    V, F = sphere()
    T = V[F]
    assert np.array_equal(
        mi.candidate_pairs(T.min(1), T.max(1)), ref.candidate_pairs(T.min(1), T.max(1))
    )
    # with stretched boxes (compared directly, not in the grid) and a given cell
    rng = np.random.default_rng(2)
    lo = rng.uniform(0, 20, (3000, 3))
    hi = lo + rng.exponential(0.5, (3000, 3))
    hi[:20] += rng.uniform(5, 15, (20, 3))
    for cell in (None, 0.7, 3.0):
        assert np.array_equal(
            mi.candidate_pairs(lo, hi, cell=cell),
            ref.candidate_pairs(lo, hi, cell=cell),
        )
    assert mi.candidate_pairs(lo[:0], hi[:0]).shape == (0, 2)
    assert mi.candidate_pairs(lo[:1], hi[:1]).shape == (0, 2)


def raises(f: Callable[..., Any], *args: Any, **kw: Any) -> str | None:
    try:
        f(*args, **kw)
    except RuntimeError as e:
        return str(e)
    return None


def test_max_pairs_behaviour() -> None:
    rng = np.random.default_rng(4)
    lo = rng.uniform(0, 20, (3000, 3))
    hi = lo + rng.exponential(0.5, (3000, 3))
    hi[:20] += rng.uniform(5, 15, (20, 3))
    msgs = set()
    caps = [0, 10, 100, 1000, 3000, 10000, 20000, 30000, 50000, 100000, 10**7]
    for cap in caps:
        r = raises(ref.candidate_pairs, lo, hi, max_pairs=cap)
        n = raises(mi.candidate_pairs, lo, hi, max_pairs=cap)
        assert r == n, (cap, r, n)
        msgs.add(r)
    # both kinds of message (grid cells, stretched boxes) and no error are
    # covered
    assert any(m and "largest cell" in m for m in msgs)
    assert any(m and "stretched" in m for m in msgs)
    assert None in msgs
    # the same through intersecting_faces and faces_hitting
    V, F = sphere()
    for cap in (100, 10**7):
        # the grid is built in the vertices' dtype, as in the reference
        V32 = V.astype(np.float32)
        assert raises(ref.intersecting_faces, V32, F, max_pairs=cap) == raises(
            mi.intersecting_faces, V32, F, max_pairs=cap
        )
        assert raises(ref.intersecting_faces, V, F, max_pairs=cap) == raises(
            mi.intersecting_faces, V, F, max_pairs=cap
        )
        assert raises(ref.faces_hitting, V, F, np.arange(300), max_pairs=cap) == raises(
            mi.faces_hitting, V, F, np.arange(300), max_pairs=cap
        )


def test_intersecting_faces_on_test_meshes() -> None:
    V, F = sphere(40)
    same_faces(ref.intersecting_faces(V, F), mi.intersecting_faces(V, F))
    W, F = folded()
    r = ref.intersecting_faces(W, F)
    assert r[1] > 0
    same_faces(r, mi.intersecting_faces(W, F))
    V, F = sphere()
    c = V.mean(0)
    r2 = ref.intersecting_faces(V, F, V + [3.37, 0.21, 0.13], F)
    assert r2[2] > 0
    same_faces(r2, mi.intersecting_faces(V, F, V + [3.37, 0.21, 0.13], F))
    same_faces(
        ref.intersecting_faces(V, F, c + 0.53 * (V - c), F),
        mi.intersecting_faces(V, F, c + 0.53 * (V - c), F),
    )
    sliver_v = np.array(
        [c + [-40, 0.13, 0.27], c + [40, 0.31, 0.11], c + [0.2, 0.17, 40]]
    )
    one = np.array([[0, 1, 2]])
    r2 = ref.intersecting_faces(V, F, sliver_v, one)
    assert r2[2] > 0
    same_faces(r2, mi.intersecting_faces(V, F, sliver_v, one))
    # a small chunk changes nothing
    same_faces(ref.intersecting_faces(W, F), mi.intersecting_faces(W, F, chunk=7))


def test_faces_hitting_on_test_meshes() -> None:
    W, F = folded()
    full, _ = ref.intersecting_faces(W, F)
    rng = np.random.default_rng(1)
    sub = np.union1d(rng.choice(len(F), 400, replace=False), np.where(full)[0][:50])
    r = ref.faces_hitting(W, F, sub)
    assert r.any()
    assert np.array_equal(r, mi.faces_hitting(W, F, sub))
    V, F = sphere()
    V2 = V + [3.37, 0.21, 0.13]
    assert np.array_equal(
        ref.faces_hitting(V, F, sub, V2, F), mi.faces_hitting(V, F, sub, V2, F)
    )
    # an unsorted sub with repeats
    s2 = np.r_[sub[::-1], sub[:5]]
    assert np.array_equal(ref.faces_hitting(W, F, s2), mi.faces_hitting(W, F, s2))
