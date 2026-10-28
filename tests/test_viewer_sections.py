"""Tests of baillarger.viewer.sections with exact answers."""

import numpy as np
import numpy.typing as npt
import pytest
from scipy.sparse import coo_matrix, csr_matrix

from baillarger.viewer import sections as PS

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


def graph(v: Float, f: Int) -> csr_matrix:
    E = np.unique(np.sort(np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 1), axis=0)
    w = np.linalg.norm(v[E[:, 0]] - v[E[:, 1]], axis=1)
    G = coo_matrix((w, (E[:, 0], E[:, 1])), shape=(len(v),) * 2).tocsr()
    return G.maximum(G.T).tocsr()


def test_directions_from_a_conformal_differential() -> None:
    n = np.array([0, 0, 1.0])
    T = 0.7 * np.array(
        [[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)], [0, 0]]
    )
    tu, tv, ang = PS.directions(T, n)
    assert np.allclose(tu, [np.cos(0.3), np.sin(0.3), 0]) and np.allclose(
        tv, [-np.sin(0.3), np.cos(0.3), 0]
    )
    assert ang == pytest.approx(0, abs=1e-6)
    Tm = T * [1, -1]  # a mirrored map: v keeps the map's sense
    tu, tv, ang = PS.directions(Tm, n)
    assert tv @ Tm[:, 1] > 0 and ang == pytest.approx(0, abs=1e-6)


def test_plane_section_is_the_row_at_euclidean_distances() -> None:
    nx, ny, h = 41, 21, 0.1
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    v = np.c_[i.ravel() * h, j.ravel() * h, np.zeros(nx * ny)]
    f = grid(nx, ny)
    G = graph(v, f)
    roots_of = -np.ones(len(v), int)
    roots_of[::2] = np.arange(0, len(v), 2) // 2  # every other vertex has a trace
    root = 20 * ny + 10
    tr, d = PS.section(
        v,
        G,
        root,
        v[root],
        np.array([0, 0, 1.0]),
        np.array([1.0, 0, 0]),
        roots_of,
        half=1.5,
        band=0.04,
        f=f,
    )
    vs = np.flatnonzero(roots_of >= 0)[tr]
    assert np.all(vs % ny == 10)  # the row j = 10
    assert (
        np.allclose(d, v[vs, 0] - v[root, 0])
        and np.all(np.diff(d) > 0)
        and np.abs(d).max() <= 1.5
    )


def test_cylinder_section_follows_the_circle_by_arc() -> None:
    nt, nz, r = 60, 11, 2.0
    th, z = np.meshgrid(
        np.arange(nt) * 2 * np.pi / nt, np.linspace(0, 1, nz), indexing="ij"
    )
    v = np.c_[r * np.cos(th).ravel(), r * np.sin(th).ravel(), z.ravel()]
    f = grid(nt, nz, wrap_x=True)
    G = graph(v, f)
    roots_of = np.arange(len(v))
    root = 0 * nz + 5
    n = np.array([1.0, 0, 0])
    tr, d = PS.section(
        v,
        G,
        root,
        v[root],
        n,
        np.array([0, 1.0, 0]),
        roots_of,
        half=4.0,
        band=0.02,
        f=f,
    )
    assert np.all(tr % nz == 5)  # the ring z = 0.5
    chord = 2 * r * np.sin(np.pi / nt)
    k = tr // nz
    k = np.where(k > nt // 2, k - nt, k)  # steps from the root round the ring
    assert np.allclose(d, k * chord) and np.all(np.diff(d) > 0)
    # positive along t = +y: the first step up the ring has y > 0
    assert v[tr[d > 0][0], 1] > 0


def test_marked_hits_take_the_nearest_sample() -> None:
    pos = np.zeros((2, 5, 3), np.float32)
    pos[0, :, 2] = np.arange(5)
    pos[1, :, 2] = np.arange(5)
    pos[1, :, 0] = 10
    P = np.array([[0.0, 0, 2.1]])
    h = PS.marked_hits(pos, np.array([5, 5]), [0, 1], P, np.ones(3), within=0.3)
    assert h == {0: [(0, 2)]}


def test_a_u_fold_section_runs_round_the_fold_by_arc_length() -> None:
    # two parallel sheets 1 mm apart joined by a half cylinder (radius 0.5): the normal
    # section across the fold is the U; distances are arc lengths along it, never across
    # the gap
    r, nu, ny = 0.5, 120, 9
    s = np.linspace(0, 2 + np.pi * r + 2, nu)  # arc parameter: arm, half circle, arm

    def xyz(a: float) -> Float:
        if a <= 2:
            return np.array([2 - a, 0.0, -r])
        if a <= 2 + np.pi * r:
            t = (a - 2) / r - np.pi / 2
            return np.array([-r * np.cos(t), 0.0, r * np.sin(t)])
        return np.array([a - 2 - np.pi * r, 0.0, r])

    P = np.array([xyz(a) for a in s])
    y = np.linspace(-0.4, 0.4, ny)
    v = np.array([[p[0], yy, p[2]] for p in P for yy in y])
    f = grid(nu, ny)
    G = graph(v, f)
    roots_of = np.arange(len(v))
    j0 = ny // 2
    k0 = 30
    root = k0 * ny + j0
    t = v[(k0 + 1) * ny + j0] - v[root]
    t /= np.linalg.norm(t)
    n = np.array([0, 0, -1.0])  # the lower arm's outward side
    tr, d = PS.section(v, G, root, v[root], n, t, roots_of, half=6.0, band=0.02, f=f)
    assert np.all(tr % ny == j0)
    k = tr // ny
    chord = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    assert np.allclose(d, chord[k] - chord[k0], atol=1e-9) and np.all(np.diff(d) > 0)
    assert k.max() > nu - 20  # reaches the upper arm, round the fold


def test_rotate_by_90_swaps_u_and_v_up_to_sign() -> None:
    tu, tv = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    a, b = PS.rotate(tu, tv, 90.0)
    assert np.allclose(a, tv) and np.allclose(b, -tu)
    a, b = PS.rotate(tu, tv, 30.0)
    assert np.allclose(a, [np.sqrt(3) / 2, 0.5, 0]) and np.allclose(
        b, [-0.5, np.sqrt(3) / 2, 0]
    )
    a, b = PS.rotate(*PS.rotate(tu, tv, 25.0), -25.0)
    assert np.allclose(a, tu) and np.allclose(b, tv)


def test_snap_puts_the_nearest_axis_in_the_plane() -> None:
    n = np.array([0, 0, 1.0])
    tu, tv = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    g = np.array([0, 1.0, 0])  # the plane y = const holds t_u at 0 and t_v at 90
    assert PS.snap(tu, tv, n, g, 10.0) == pytest.approx((0.0, "u", 0.0))
    r, ax, tilt = PS.snap(tu, tv, n, g, 80.0)  # type: ignore[misc]
    assert r == pytest.approx(90.0) and ax == "v" and tilt == pytest.approx(0.0)
    r, ax, _ = PS.snap(tu, tv, n, g, -170.0)  # type: ignore[misc]
    assert r == pytest.approx(180.0) and ax == "u"
    assert PS.snap(tu, tv, n, n, 0.0) is None  # the tangent plane is that plane
    # a plane at 30 degrees round z: t_u turned by 30 lies in it; a normal tilted
    # 20 degrees towards the plane's normal leaves that tilt between the planes
    g = np.array([-np.sin(np.radians(30)), np.cos(np.radians(30)), 0])
    nt = np.array([0, -np.sin(np.radians(20)), np.cos(np.radians(20))])
    tv2 = np.cross(nt, tu)
    r, ax, tilt = PS.snap(tu, tv2, nt, g, 0.0)  # type: ignore[misc]
    a, _ = PS.rotate(tu, tv2, r)
    assert ax == "u" and a @ g == pytest.approx(0.0, abs=1e-12)
    m = np.cross(nt, a)  # the section plane's normal: the angle between the planes
    assert tilt == pytest.approx(np.degrees(np.arccos(abs(m @ g))))
    assert tilt == pytest.approx(20.0 * np.cos(np.radians(30)), abs=0.5)


def test_section_curve_is_the_row_through_the_root() -> None:
    nx, ny, h = 41, 21, 0.1
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    v = np.c_[i.ravel() * h, j.ravel() * h, np.zeros(nx * ny)]
    f = grid(nx, ny)
    G = graph(v, f)
    roots_of = np.arange(len(v))
    root = 20 * ny + 10
    n = np.array([0, 0, 1.0])
    # a plane between the rows (offset by a quarter step) so that every crossing is
    # on an edge: the curve is the line y = y_root + h / 4, x within half of the root
    x0 = v[root] + [0, h / 4, 0]
    _, _, c = PS.section_curve(
        v, G, root, x0, n, np.array([1.0, 0, 0]), roots_of, half=1.0, band=0.04, f=f
    )
    assert np.allclose(c[:, 1], x0[1]) and np.allclose(c[:, 2], 0)
    assert np.all(np.diff(c[:, 0]) > 0) and np.abs(c[:, 0] - x0[0]).max() <= 1.0 + 1e-9
    assert c[:, 0].min() < x0[0] - 0.9 and c[:, 0].max() > x0[0] + 0.9
