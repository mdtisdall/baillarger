"""Tests of streamline_view with exact answers.
    nix develop --command uv run pytest -q prototypes/scripts/test_streamline_view.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import streamline_view as SV  # noqa: E402
from test_flat_map import grid  # noqa: E402
from test_pia_sections import graph  # noqa: E402

STEP = 0.04


def straight(roots, d, n):
    """Traces from each root along direction d, n samples at STEP (positions in mm, vox = 1)."""
    s = np.arange(201) * STEP
    pos = roots[:, None, :] + s[None, :, None] * np.asarray(d, float)
    ns = np.full(len(roots), n)
    pos[np.arange(201)[None, :] >= ns[:, None]] = np.nan
    return pos.astype(np.float32), ns


def plane(nx, ny, h, z=0.0):
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    return np.c_[i.ravel() * h, j.ravel() * h, np.full(nx * ny, z)], grid(nx, ny)


def test_from_is_the_roots_within_the_geodesic_radius_and_to_the_ends_within_it():
    # a plane at z = 0, every vertex a root, traces straight up 1 mm; and 3 traces from z = 2 down to
    # z = 0.04 whose ends land 0, 0.6 and 1.2 mm from the click
    h = 0.1
    v, f = plane(41, 41, h)
    G = graph(v, f)
    root = 20 * 41 + 20  # (2, 2, 0)
    x0 = v[root]
    up, nup = straight(v, [0, 0, 1], 26)
    far = np.array([[2.0, 2.0, 2.0], [2.6, 2.0, 2.0], [2.0, 3.2, 2.0]])
    down, ndown = straight(far, [0, 0, -1], 50)  # last sample at z = 2 - 49 * 0.04 = 0.04
    pos, ns = np.r_[up, down], np.r_[nup, ndown]
    tv = np.r_[np.arange(len(v)), -np.ones(3, int)]  # the far roots are not on this mesh
    fr, to = SV.select(G, root, x0, tv, pos, ns, np.ones(3), radius=1.0)
    d = SV.geodesic(G, root, 1.0)
    assert set(fr.tolist()) == set(np.flatnonzero(np.isfinite(d[tv[:len(v)]])).tolist())
    # along the mesh never shorter than in space; along the grid's axes the two agree
    assert (np.linalg.norm(v[fr] - x0, axis=1) <= 1.0 + 1e-9).all()
    axes = [root + k * s for s in (1, -1, 41, -41) for k in range(11)]
    assert set(axes) <= set(fr.tolist())
    assert to.tolist() == [len(v), len(v) + 1]


def test_the_opposite_bank_is_to_not_from():
    # two parallel sheets 0.5 mm apart joined far away (a U fold): roots on the other sheet are
    # 0.5 mm away in space but far along the mesh, so their traces are "to" (ending at the click),
    # not "from"
    h = 0.1
    va, fa = plane(41, 11, h, 0.0)
    vb, fb = plane(41, 11, h, 0.5)
    v, f = np.r_[va, vb], np.r_[fa, fb + len(va)]
    # join the two sheets at x = 4 with a strip of faces (a long way round)
    ia = np.arange(11) + 40 * 11
    ib = ia + len(va)
    strip = np.r_[np.c_[ia[:-1], ia[1:], ib[:-1]], np.c_[ia[1:], ib[1:], ib[:-1]]]
    f = np.r_[f, strip]
    G = graph(v, f)
    root = 10 * 11 + 5  # (1.0, 0.5, 0) on sheet a
    up, nup = straight(va, [0, 0, 1], 10)  # 0.36 mm up from sheet a: every end > 0.25 mm from the click
    dn, ndn = straight(vb, [0, 0, -1], 12)  # 0.44 mm down from sheet b: ends 0.06 mm above sheet a
    pos, ns = np.r_[up, dn], np.r_[nup, ndn]
    tv = np.arange(len(v))
    fr, to = SV.select(G, root, v[root], tv, pos, ns, np.ones(3), radius=0.25)
    assert fr.size and (fr < len(va)).all()
    assert to.size and (to >= len(va)).all()
    ends = pos[to, ns[to] - 1]
    assert (np.linalg.norm(ends - v[root], axis=1) <= 0.25).all()


def test_crop_mesh_keeps_the_faces_inside_the_box_reindexed():
    v, f = plane(11, 11, 1.0)
    vc, fc, keep = SV.crop_mesh(v, f, np.array([2.0, 2.0, -1.0]), np.array([5.0, 6.0, 1.0]))
    assert (vc >= [2, 2, -1]).all() and (vc <= [5, 6, 1]).all()
    assert len(fc) == 2 * 3 * 4  # 3 x 4 unit squares, 2 faces each
    assert np.array_equal(vc, v[keep]) and fc.max() < len(vc)


def test_gm_surface_of_a_ball_is_closed_at_its_radius_and_clipped_by_the_box():
    n, r = 41, 10.0
    x = np.indices((n, n, n)).transpose(1, 2, 3, 0) - 20.0
    ball = np.linalg.norm(x, axis=-1) <= r
    vox = np.array([0.5, 0.5, 0.5])
    v, f = SV.gm_surface(ball, vox, np.array([0, 0, 0]), np.array([n, n, n]), sigma=0.0)
    E = np.unique(np.sort(np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 1), axis=0)
    assert len(v) - len(E) + len(f) == 2  # a sphere
    rad = np.linalg.norm(v / vox - 20.0, axis=1)
    assert rad.min() > r - 1 and rad.max() < r + 1
    # the box's upper half only: every vertex in it, and the cut is open (boundary edges)
    v2, f2 = SV.gm_surface(ball, vox, np.array([0, 0, 20]), np.array([n, n, n]), sigma=0.0)
    assert (v2[:, 2] / vox[2] >= 20 - 1e-9).all()
    e2 = np.sort(np.r_[f2[:, [0, 1]], f2[:, [1, 2]], f2[:, [2, 0]]], 1)
    _, c = np.unique(e2, axis=0, return_counts=True)
    assert (c == 1).any()


def test_pia_points_lie_on_the_trace_at_their_arc_length():
    pos, ns = straight(np.zeros((2, 3)), [1, 0, 0], 51)  # 2 mm long
    p = SV.at_depth(pos, ns, np.array([0.5, 3.0]), vox=np.ones(3))
    assert p[0] == pytest.approx([0.5, 0, 0], abs=1e-6)
    assert np.isnan(p[1]).all()  # past the trace's end
