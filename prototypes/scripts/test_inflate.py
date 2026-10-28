"""Tests of inflate with exact answers.
    nix develop --command uv run pytest -q prototypes/scripts/test_inflate.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import flat_map as FM  # noqa: E402
import inflate as IN  # noqa: E402
from test_flat_map import grid  # noqa: E402


def sheet(nx=30, ny=20, h=0.1, bump=0.0):
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    x, y = i.ravel() * h, j.ravel() * h
    z = bump * np.cos(2 * np.pi * x / (nx * h / 3))
    return np.c_[x, y, z], grid(nx, ny)


def test_a_uniform_plane_grid_does_not_move():
    # each interior vertex is the mean of its neighbours on a uniform grid; the boundary is pinned
    v, f = sheet()
    for x in IN.inflate(v, f, levels=(50.0, 200.0)):
        assert np.max(np.abs(x - v)) < 1e-6


def test_the_boundary_stays_pinned():
    v, f = sheet(60, 30, 0.1, bump=0.5)
    b = np.concatenate(FM.boundary_loops(f))
    x = IN.inflate(v, f, levels=(100.0,))[0]
    assert np.array_equal(x[b], v[b]) and FM.face_areas(x, f).sum() < FM.face_areas(v, f).sum()


def test_inflation_converges_to_the_harmonic_membrane():
    # as lam rises the interior goes to the discrete harmonic extension of the pinned boundary
    # (L_ff x_f = -L_fb x_b), monotonically
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


def test_tangent_transfer_is_the_face_map():
    # true mesh = a linear map S of the inflated one: a tangent vector r maps to S r exactly
    v, f = sheet()
    S = np.array([[1.5, 0.4, 0], [0.1, 0.8, 0], [0.2, -0.3, 1.0]])
    true = v @ S.T
    vf = IN.vertex_faces(f, len(v))
    vi = 10 * 20 + 7
    for r in (np.array([1.0, 0, 0]), np.array([0.3, -0.7, 0])):
        assert np.allclose(IN.transfer(r, vi, v, true, f, vf), S @ r)
    # a component off the inflated face's plane is dropped
    assert np.allclose(IN.transfer(np.array([0, 0, 1.0]), vi, v, true, f, vf), 0)


def test_sulcal_depth_is_the_normal_offset():
    v, f = sheet()
    x = v.copy()
    x[:, 2] -= 0.25
    n = IN.vertex_normals(x, f)
    assert np.allclose(np.abs(n[:, 2]), 1)
    assert np.allclose(IN.sulcal_depth(v, x, f), 0.25 * n[:, 2])
