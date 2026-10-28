"""Tests of pia_sections with exact answers.
    nix develop --command uv run pytest -q prototypes/scripts/test_pia_sections.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import coo_matrix

sys.path.insert(0, str(Path(__file__).parent))
import pia_sections as PS  # noqa: E402
from test_flat_map import grid  # noqa: E402


def graph(v, f):
    E = np.unique(np.sort(np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 1), axis=0)
    w = np.linalg.norm(v[E[:, 0]] - v[E[:, 1]], axis=1)
    G = coo_matrix((w, (E[:, 0], E[:, 1])), shape=(len(v),) * 2).tocsr()
    return G.maximum(G.T).tocsr()


def test_directions_from_a_conformal_differential():
    n = np.array([0, 0, 1.0])
    T = 0.7 * np.array([[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)], [0, 0]])
    tu, tv, ang = PS.directions(T, n)
    assert np.allclose(tu, [np.cos(0.3), np.sin(0.3), 0]) and np.allclose(tv, [-np.sin(0.3), np.cos(0.3), 0])
    assert ang == pytest.approx(0, abs=1e-6)
    Tm = T * [1, -1]  # a mirrored map: v keeps the map's sense
    tu, tv, ang = PS.directions(Tm, n)
    assert tv @ Tm[:, 1] > 0 and ang == pytest.approx(0, abs=1e-6)


def test_plane_section_is_the_row_at_euclidean_distances():
    nx, ny, h = 41, 21, 0.1
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    v = np.c_[i.ravel() * h, j.ravel() * h, np.zeros(nx * ny)]
    f = grid(nx, ny)
    G = graph(v, f)
    roots_of = -np.ones(len(v), int)
    roots_of[::2] = np.arange(0, len(v), 2) // 2  # every other vertex has a trace
    root = 20 * ny + 10
    tr, d = PS.section(v, G, root, v[root], np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), roots_of, half=1.5,
                       band=0.04, f=f)
    vs = np.flatnonzero(roots_of >= 0)[tr]
    assert np.all(vs % ny == 10)  # the row j = 10
    assert np.allclose(d, v[vs, 0] - v[root, 0]) and np.all(np.diff(d) > 0) and np.abs(d).max() <= 1.5


def test_cylinder_section_follows_the_circle_by_arc():
    nt, nz, r = 60, 11, 2.0
    th, z = np.meshgrid(np.arange(nt) * 2 * np.pi / nt, np.linspace(0, 1, nz), indexing="ij")
    v = np.c_[r * np.cos(th).ravel(), r * np.sin(th).ravel(), z.ravel()]
    f = grid(nt, nz, wrap_x=True)
    G = graph(v, f)
    roots_of = np.arange(len(v))
    root = 0 * nz + 5
    n = np.array([1.0, 0, 0])
    tr, d = PS.section(v, G, root, v[root], n, np.array([0, 1.0, 0]), roots_of, half=4.0, band=0.02, f=f)
    assert np.all(tr % nz == 5)  # the ring z = 0.5
    chord = 2 * r * np.sin(np.pi / nt)
    k = tr // nz
    k = np.where(k > nt // 2, k - nt, k)  # steps from the root round the ring
    assert np.allclose(d, k * chord) and np.all(np.diff(d) > 0)
    # positive along t = +y: the first step up the ring has y > 0
    assert v[tr[d > 0][0], 1] > 0


def test_marked_hits_take_the_nearest_sample():
    pos = np.zeros((2, 5, 3), np.float32)
    pos[0, :, 2] = np.arange(5)
    pos[1, :, 2] = np.arange(5)
    pos[1, :, 0] = 10
    P = np.array([[0.0, 0, 2.1]])
    h = PS.marked_hits(pos, np.array([5, 5]), [0, 1], P, np.ones(3), within=0.3)
    assert h == {0: [(0, 2)]}


def test_a_u_fold_section_runs_round_the_fold_by_arc_length():
    # two parallel sheets 1 mm apart joined by a half cylinder (radius 0.5): the normal section
    # across the fold is the U; distances are arc lengths along it, never across the gap
    r, nu, ny = 0.5, 120, 9
    s = np.linspace(0, 2 + np.pi * r + 2, nu)  # arc parameter: arm, half circle, arm
    def xyz(a):
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
