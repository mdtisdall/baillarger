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
    # the smoothing alone (metric 0): as tau rises the interior goes to the discrete
    # harmonic extension of the pinned boundary (L_ff x_f = -L_fb x_b), monotonically
    from scipy.sparse.linalg import spsolve

    v, f = sheet(40, 30, 0.1, bump=0.5)
    v[:, 2] += 0.2 * np.sin(np.arange(len(v)) * 0.7)  # roughen the interior too
    L = IN.laplacian(v, f)
    onb = np.zeros(len(v), bool)
    onb[np.concatenate(FM.boundary_loops(f))] = True
    fr = ~onb
    hm = v.copy()
    hm[fr] = spsolve(L[fr][:, fr].tocsc(), -(L[fr][:, onb] @ v[onb]))
    X = IN.inflate(v, f, levels=(1.0, 10.0, 100.0, 1e5), metric=0.0)
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


def moments(x: Float, w: Float | None = None) -> tuple[Float, Float]:
    """Centroid and second moments of x, each row weighted by w (default 1)."""
    w = np.ones(len(x)) if w is None else w
    c = (w @ x) / w.sum()
    return c, ((x - c).T * w) @ (x - c) / w.sum()


def test_a_closed_surface_keeps_its_moments_and_gets_rounder() -> None:
    # the smoothing alone (metric 0); the moments are weighted by the true areas
    v, f = ball((13.3, 14.1, 13.7), 9.0)
    assert not FM.boundary_loops(f)
    m = IN.vertex_areas(v, f)
    c0, C0 = moments(v, m)
    X = IN.inflate(v, f, levels=(2.5, 10.0, 40.0, 4e4), metric=0.0)
    rs = [roundness(v)] + [roundness(x) for x in X]
    for x in X:
        c, C = moments(x, m)
        assert np.allclose(c, c0, atol=1e-9) and np.allclose(C, C0, rtol=1e-9)
    # rounder until it reaches its limit (the three lowest modes, checked exactly
    # below), then it stays
    assert (
        rs[0] > rs[1] > rs[2] and abs(rs[4] - rs[3]) < 1e-3 and rs[4] < 0.6 * rs[0]
    ), rs
    r = np.linalg.norm(X[-1] - c0, axis=1)
    assert r.min() > 0.5 * r.mean()


def test_a_closed_surface_goes_to_its_three_lowest_modes() -> None:
    # the smoothing alone (metric 0): as tau grows, x - c lies in the span of the
    # three lowest nonconstant modes of L u = mu M u (M: the true vertex areas)
    from scipy.linalg import eigh

    v, f = ball((10.3, 10.1, 10.7), 6.0, n=22)
    L = IN.laplacian(v, f).toarray()
    m = IN.vertex_areas(v, f)
    _, U = eigh(L, np.diag(m))
    B = U[:, 1:4]  # M-orthonormal
    x = IN.inflate(v, f, levels=(1e2, 1e4, 1e6), steps_per_level=8, metric=0.0)
    res = []
    for xi in x:
        d = xi - (m @ xi) / m.sum()
        P = B @ (B.T @ (m[:, None] * d))
        sm = np.sqrt(m)[:, None]
        res.append(np.linalg.norm(sm * (d - P)) / np.linalg.norm(sm * d))
    assert res[0] > res[1] > res[2] and res[2] < 1e-3, res


def test_an_open_piece_is_unchanged_by_a_closed_piece_beside_it() -> None:
    s, fs = sheet(30, 20, 0.5, bump=1.0)
    b, fb = ball((13.3, 14.1, 13.7), 6.0)
    b = b + np.array([0.0, 0.0, 40.0])
    v = np.r_[s, b]
    f = np.r_[fs, fb + len(s)]
    alone = IN.inflate(s, fs, levels=(5.0, 50.0))
    both = IN.inflate(v, f, levels=(5.0, 50.0))
    mb = IN.vertex_areas(b, fb)
    cb, Cb = moments(b, mb)
    for xa, xb in zip(alone, both, strict=True):
        # the same linear system, solved for both pieces at once
        assert np.max(np.abs(xb[: len(s)] - xa)) < 1e-8
        c, C = moments(xb[len(s) :], mb)
        assert np.allclose(c, cb, atol=1e-9) and np.allclose(C, Cb, rtol=1e-9)


def test_with_the_metric_term_a_closed_surface_still_gets_rounder() -> None:
    # levels with diffusion lengths sqrt(2 tau) of 1/4, 1/2 and 1 times the ball's
    # radius, as LEVELS are for a hemisphere
    v, f = ball((13.3, 14.1, 13.7), 9.0)
    X = IN.inflate(v, f, levels=(2.5, 10.0, 40.0))
    rs = [roundness(v)] + [roundness(x) for x in X]
    assert rs[0] > rs[1] > rs[2] > rs[3] and rs[3] < 0.6 * rs[0], rs


# Thin parts and uneven sampling. The right answer for navigation: every part of the
# surface keeps a clickable share of the area. Here: at least 1/4 of its true share,
# and no more than 1 % of its area shrunk by more than 2^3. The sampling must not
# change the answer: a sphere stays a sphere however its vertices are spread.


def level_set(sd: Float) -> tuple[Float, Int]:
    """The surface {sd = 0} of a signed distance (negative inside)."""
    from baillarger.mesh import marching_tetrahedra

    v, f = marching_tetrahedra(-sd, 0.0)
    assert not FM.boundary_loops(f)
    return v, f.astype(np.int64)


def keeps_its_area(v: Float, f: Int, part: npt.NDArray[np.bool_]) -> None:
    a0 = FM.face_areas(v, f)
    pf = part[f].all(1)
    share0 = a0[pf].sum() / a0.sum()
    for lv, x in zip(IN.LEVELS, IN.inflate(v, f), strict=True):
        a = FM.face_areas(x, f)
        share = a[pf].sum() / a.sum()
        la = np.log2(np.maximum(a[pf], 1e-30) / a0[pf])
        p1 = FM.wpct(la, a0[pf], [1])[0]
        assert share > share0 / 4 and p1 > -3, (lv, share0, share, p1)


def test_a_thin_finger_keeps_its_area() -> None:
    # a ball of radius 14 with a finger of radius 2 out to 26 from its centre
    n = 52
    c = np.array([18.3, 25.6, 25.4])
    g = np.stack(np.meshgrid(*[np.arange(n, dtype=float)] * 3, indexing="ij"), -1)
    p = g - c
    ball_sd = np.linalg.norm(p, axis=-1) - 14
    t = np.clip(p[..., 0], 0, 26)
    finger_sd = np.sqrt((p[..., 0] - t) ** 2 + p[..., 1] ** 2 + p[..., 2] ** 2) - 2
    v, f = level_set(np.minimum(ball_sd, finger_sd))
    keeps_its_area(v, f, v[:, 0] - c[0] > 16)


def test_the_walls_of_a_narrow_slot_keep_their_area() -> None:
    # a ball of radius 20 with a slot 3 wide, 22 long and 18 deep cut in from its top
    n = 48
    c = np.array([23.6, 23.4, 23.7])
    g = np.stack(np.meshgrid(*[np.arange(n, dtype=float)] * 3, indexing="ij"), -1)
    p = g - c
    ball_sd = np.linalg.norm(p, axis=-1) - 20
    q = np.abs(p - np.array([0, 2 + 20, 0])) - np.array([1.5, 20, 11])
    slot_sd = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)
    v, f = level_set(np.maximum(ball_sd, -slot_sd))
    d = v - c
    walls = (np.abs(d[:, 0]) < 2.5) & (d[:, 1] > 2.5) & (np.abs(d[:, 2]) < 11.5)
    keeps_its_area(v, f, walls)


def test_a_sphere_stays_a_sphere_however_it_is_sampled() -> None:
    # vertices on a sphere of radius 10, 20 times sparser on the half x < 0
    from scipy.spatial import ConvexHull

    rng = np.random.default_rng(0)
    p = rng.normal(size=(24000, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    p = 10 * p[(p[:, 0] > 0) | (rng.random(len(p)) < 0.05)]
    f = ConvexHull(p).simplices.astype(np.int64)
    nrm = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
    out = (nrm * p[f].mean(1)).sum(1)
    f[out < 0] = f[out < 0][:, ::-1]
    a0 = FM.face_areas(p, f)
    half = (p[f, 0] < 0).all(1)
    for x in IN.inflate(p, f):
        a = FM.face_areas(x, f)
        assert abs(a[half].sum() / a.sum() - a0[half].sum() / a0.sum()) < 0.01
        w = np.zeros(len(x))
        np.add.at(w, f.ravel(), np.repeat(a, 3))
        r = np.linalg.norm(x - (w @ x) / w.sum(), axis=1)
        assert r.std() / r.mean() < 0.01
