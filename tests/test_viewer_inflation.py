"""Tests of baillarger.viewer.inflation with exact answers."""

import numpy as np
import numpy.typing as npt

from baillarger.viewer import flatmap as FM
from baillarger.viewer import inflation as IN

type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]


def grid(nx: int, ny: int, wrap_x: bool = False, wrap_y: bool = False) -> Int:
    """Faces of an nx x ny vertex grid (index i * ny + j), counter-clockwise in
    (i, j)."""
    F = []
    for i in range(nx if wrap_x else nx - 1):
        for j in range(ny if wrap_y else ny - 1):
            a, b = i * ny + j, ((i + 1) % nx) * ny + j
            c, d = ((i + 1) % nx) * ny + (j + 1) % ny, i * ny + (j + 1) % ny
            F += [(a, b, c), (a, c, d)]
    return np.array(F)


def sheet(
    nx: int = 30, ny: int = 20, h: float = 0.1, bump: float = 0.0
) -> tuple[Float, Int]:
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    x, y = i.ravel() * h, j.ravel() * h
    z = bump * np.cos(2 * np.pi * x / (nx * h / 3))
    return np.c_[x, y, z], grid(nx, ny)


def test_a_uniform_plane_grid_does_not_move() -> None:
    # each interior vertex is the mean of its neighbours on a uniform grid; the boundary
    # is pinned
    v, f = sheet()
    for x in IN.inflate(v, f, levels=(50.0, 200.0)):
        assert np.max(np.abs(x - v)) < 1e-6


def test_the_boundary_stays_pinned() -> None:
    v, f = sheet(60, 30, 0.1, bump=0.5)
    b = np.concatenate(FM.boundary_loops(f))
    x = IN.inflate(v, f, levels=(100.0,))[0]
    assert (
        np.array_equal(x[b], v[b])
        and FM.face_areas(x, f).sum() < FM.face_areas(v, f).sum()
    )


def test_inflation_converges_to_the_harmonic_membrane() -> None:
    # as lam rises the interior goes to the discrete harmonic extension of the pinned
    # boundary (L_ff x_f = -L_fb x_b), monotonically
    from scipy.sparse.linalg import spsolve

    v, f = sheet(40, 30, 0.1, bump=0.5)
    v[:, 2] += 0.2 * np.sin(np.arange(len(v)) * 0.7)  # roughen the interior too
    L = IN.umbrella(f, len(v))
    onb = np.zeros(len(v), bool)
    onb[np.concatenate(FM.boundary_loops(f))] = True
    fr = ~onb
    hm = v.copy()
    hm[fr] = spsolve(L[fr][:, fr].tocsc(), -(L[fr][:, onb] @ v[onb]))
    X = IN.inflate(v, f, levels=(10.0, 100.0, 1000.0, 1e6))
    err = [np.max(np.abs(x - hm)) for x in X]
    assert err[0] > err[1] > err[2] > err[3] and err[3] < 1e-4, err


def test_tangent_transfer_is_the_face_map() -> None:
    # true mesh = a linear map S of the inflated one: a tangent vector r maps to S r
    # exactly
    v, f = sheet()
    S = np.array([[1.5, 0.4, 0], [0.1, 0.8, 0], [0.2, -0.3, 1.0]])
    true = v @ S.T
    vf = IN.vertex_faces(f, len(v))
    vi = 10 * 20 + 7
    for r in (np.array([1.0, 0, 0]), np.array([0.3, -0.7, 0])):
        assert np.allclose(IN.transfer(r, vi, v, true, f, vf), S @ r)
    # a component off the inflated face's plane is dropped
    assert np.allclose(IN.transfer(np.array([0, 0, 1.0]), vi, v, true, f, vf), 0)


def test_sulcal_depth_is_the_normal_offset() -> None:
    v, f = sheet()
    x = v.copy()
    x[:, 2] -= 0.25
    n = IN.vertex_normals(x, f)
    assert np.allclose(np.abs(n[:, 2]), 1)
    assert np.allclose(IN.sulcal_depth(v, x, f), 0.25 * n[:, 2])


def ball(c: tuple[float, float, float], r: float, n: int = 28) -> tuple[Float, Int]:
    """A closed surface: marching tetrahedra of a ball, bumped radially."""
    from baillarger.mesh import marching_tetrahedra

    x = np.arange(n, dtype=float)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    d = np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2)
    v, f = marching_tetrahedra(d, r)
    assert not FM.boundary_loops(f)  # the ball is inside the grid
    u = (v - c) / np.linalg.norm(v - c, axis=1, keepdims=True)
    v = c + (v - c) * (1 + 0.25 * np.sin(3 * u[:, 0]) * np.cos(4 * u[:, 2]))[:, None]
    return v, f.astype(np.int64)


def roundness(x: Float) -> float:
    """The spread of the radii of x whitened by its own second moments (0 for
    any ellipsoid's vertices)."""
    c, C = moments(x)
    w, U = np.linalg.eigh(C)
    z = (x - c) @ U / np.sqrt(w)
    r = np.linalg.norm(z, axis=1)
    return float(r.std() / r.mean())


def moments(x: Float) -> tuple[Float, Float]:
    c = x.mean(0)
    return c, (x - c).T @ (x - c) / len(x)


def test_a_closed_surface_keeps_its_moments_and_gets_rounder() -> None:
    v, f = ball((13.3, 14.1, 13.7), 9.0)
    assert not FM.boundary_loops(f)
    c0, C0 = moments(v)
    X = IN.inflate(v, f, levels=(10.0, 100.0, 1000.0, 1e5))
    rs = [roundness(v)] + [roundness(x) for x in X]
    for x in X:
        c, C = moments(x)
        assert np.allclose(c, c0, atol=1e-9) and np.allclose(C, C0, rtol=1e-9)
    # rounder until it reaches its limit (the three lowest modes), then it stays
    assert (
        rs[0] > rs[1] > rs[2] and abs(rs[4] - rs[3]) < 1e-3 and rs[4] < 0.5 * rs[0]
    ), rs
    r = np.linalg.norm(X[-1] - c0, axis=1)
    assert r.min() > 0.5 * r.mean()


def test_a_closed_surface_goes_to_its_three_lowest_modes() -> None:
    # as lam grows, x - c lies in the span of the three lowest nonconstant
    # eigenvectors of the umbrella Laplacian
    from scipy.linalg import eigh

    v, f = ball((10.3, 10.1, 10.7), 6.0, n=22)
    L = IN.umbrella(f, len(v)).toarray()
    _, U = eigh(L)
    B = U[:, 1:4]
    x = IN.inflate(v, f, levels=(1e3, 1e5, 1e7), steps_per_level=8)
    res = []
    for xi in x:
        d = xi - xi.mean(0)
        res.append(np.linalg.norm(d - B @ (B.T @ d)) / np.linalg.norm(d))
    assert res[0] > res[1] > res[2] and res[2] < 1e-3, res


def test_an_open_piece_is_unchanged_by_a_closed_piece_beside_it() -> None:
    s, fs = sheet(30, 20, 0.5, bump=1.0)
    b, fb = ball((13.3, 14.1, 13.7), 6.0)
    b = b + np.array([0.0, 0.0, 40.0])
    v = np.r_[s, b]
    f = np.r_[fs, fb + len(s)]
    alone = IN.inflate(s, fs, levels=(10.0, 100.0))
    both = IN.inflate(v, f, levels=(10.0, 100.0))
    cb, Cb = moments(b)
    for xa, xb in zip(alone, both, strict=True):
        # the same linear system; cg solves it to its tolerance either way
        assert np.max(np.abs(xb[: len(s)] - xa)) < 1e-5
        c, C = moments(xb[len(s) :])
        assert np.allclose(c, cb, atol=1e-9) and np.allclose(C, Cb, rtol=1e-9)
