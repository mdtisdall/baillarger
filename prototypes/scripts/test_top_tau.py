"""Tests of the geometry of τ behind a fixed Σ (top_tau.py), with exact
answers on flat synthetic sheets: the axis crossings, inside Σ, the ghost
sides and θ, τ0, and the step directions from ∇τ. The τ solve itself is
global_tau_solver's; it is checked on real site crops (diagnostics), not
here."""

from __future__ import annotations

from types import SimpleNamespace

import global_tau_prep as TP
import global_tau_solver as GS
import numpy as np
import top_tau as TT

H = 0.16


def sheet(n: int, z: float, up: bool) -> tuple[np.ndarray, np.ndarray]:
    """A square sheet over x, y in [-1, n] (voxels) at height z, two
    triangles per cell of size (n + 1) / 4; normal +z (up) or -z."""
    xs = np.linspace(-1.0, n, 5)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    v = np.c_[X.ravel(), Y.ravel(), np.full(X.size, z)]
    f = []
    for i in range(4):
        for j in range(4):
            a, b, c, d = i * 5 + j, (i + 1) * 5 + j, (i + 1) * 5 + j + 1, i * 5 + j + 1
            f += [(a, b, c), (a, c, d)]
    F = np.array(f, np.int64)
    return v, (F if up else F[:, ::-1].copy())


def site(shape: tuple[int, int, int], white_z: float) -> SimpleNamespace:
    """Cortex everywhere; the white surface at z = white_z (inside below)."""
    lab = np.full(shape, TP.CORTEX, np.int16)
    z = np.arange(shape[2])[None, None, :] * np.ones(shape)
    dist = (z - white_z) * H  # signed distance to the white surface, mm (negative inside)
    return SimpleNamespace(labels=lab, vox=np.full(3, H), inside=dist < 0, dist=dist)


def test_axis_crossings_of_a_flat_sheet() -> None:
    """Every z line crosses the sheet once at t = 5.3, with the sign of +z;
    the x and y lines are parallel to it and cross nothing."""
    n = 6
    v, f = sheet(n, 5.3, True)
    shape = np.array([n, n, 10], np.int64)
    U, W, T, S = TT._axis_crossings(v, f, 2, shape)
    assert len(U) == n * n
    assert len({(u, w) for u, w in zip(U, W)}) == n * n
    np.testing.assert_allclose(T, 5.3)
    assert (S == 1).all()
    for a in (0, 1):
        assert len(TT._axis_crossings(v, f, a, shape)[0]) == 0


def test_sigma_top_inside_ghosts_and_theta() -> None:
    """White surface at z = 1.5, Σ at z = 5.3 (normal +z): the tissue is
    z = 2..5, the inner voxels z = 0..1 (within 1.5 voxels inside the white
    surface); the +z side of z = 5 is a ghost with θ = 0.3, and no other
    side is; τ0 = 0.5 h at z = 5 falling by h per voxel."""
    n = 6
    v, f = sheet(n, 5.3, True)
    st = site((n, n, 10), 1.5)
    sig, info = TT.sigma_top(st, v * H, f, None, GS.Settings())
    zs = np.flatnonzero(sig.domain[2, 2])
    np.testing.assert_array_equal(zs, np.arange(0, 6))
    np.testing.assert_array_equal(np.flatnonzero(sig.inner[2, 2]), [0, 1])
    g = sig.ghost
    assert g[:, :, 5, 2, 1].all() and g.sum() == n * n
    np.testing.assert_allclose(sig.gdist[:, :, 5, 2, 1], 0.3)
    np.testing.assert_allclose(sig.tau0[2, 2, 2:6], [-3.5 * H, -2.5 * H, -1.5 * H, -0.5 * H])
    assert info["unanchored_voxels"] == 0


def test_sigma_top_inside_the_wm_voxels() -> None:
    """At the start Σ lies inside the WM voxels: white surface at z = 5.5,
    Σ at z = 4.3. The domain is the inner voxel z = 4 only (z = 5 is beyond
    Σ), and it has the ghost on its +z side (θ = 0.3), so its τ is fixed."""
    n = 6
    v, f = sheet(n, 4.3, True)
    st = site((n, n, 10), 5.5)
    sig, info = TT.sigma_top(st, v * H, f, None, GS.Settings())
    np.testing.assert_array_equal(np.flatnonzero(sig.domain[2, 2]), [4])
    assert sig.inner[2, 2, 4] and sig.ghost[2, 2, 4, 2, 1]
    np.testing.assert_allclose(sig.gdist[2, 2, 4, 2, 1], 0.3)
    assert info["tissue_voxels"] == 0 and info["unanchored_voxels"] == 0


def test_sigma_top_leaves_out_an_unanchored_part() -> None:
    """A part of the domain that no axis line crosses Σ next to (a closed
    shell of tissue between two WM regions, cut off from Σ) has no τ = 0: it
    is left out and counted."""
    n = 6
    v, f = sheet(n, 5.3, True)
    st = site((n, n, 10), 1.5)
    # WM at z = 3 cuts the tissue into z = 2 (below) and z = 4..5 (next to Σ)
    st.inside = st.inside | (np.arange(10) == 3)[None, None, :]
    st.dist = np.where(st.inside, -1.0, st.dist)  # all WM deep: no inner voxels
    sig, info = TT.sigma_top(st, v * H, f, None, GS.Settings())
    np.testing.assert_array_equal(np.flatnonzero(sig.domain[2, 2]), [4, 5])
    assert info["unanchored_voxels"] == n * n  # the layer z = 2
    assert info["unanchored_parts"] == 1


def test_sigma_top_two_facing_sheets() -> None:
    """Σ = an up sheet at z = 5.3 and a down sheet at z = 5.8 (a contact
    with no voxel centre between them): each side sees only its own sheet."""
    n = 6
    v1, f1 = sheet(n, 5.3, True)
    v2, f2 = sheet(n, 5.8, False)
    v = np.r_[v1, v2]
    f = np.r_[f1, f2 + len(v1)]
    st = site((n, n, 10), 1.5)
    st.inside = st.inside | (np.arange(10) >= 9)[None, None, :]  # WM above too
    sig, info = TT.sigma_top(st, v * H, f, None, GS.Settings())
    dom = np.flatnonzero(sig.domain[2, 2] & ~sig.inner[2, 2])
    np.testing.assert_array_equal(dom, [2, 3, 4, 5, 6, 7, 8])
    np.testing.assert_allclose(sig.gdist[2, 2, 5, 2, 1], 0.3)
    np.testing.assert_allclose(sig.gdist[2, 2, 6, 2, 0], 0.2)
    assert not sig.ghost[2, 2, 5, 2, 0] and not sig.ghost[2, 2, 6, 2, 1]


def test_directions_mean_of_nearby_gradient() -> None:
    """∇τ = +z on the tissue voxels below z = 5: a vertex at z = 5.3 has the
    direction +z; a vertex more than one voxel from any such voxel has
    none."""
    shape = (6, 6, 10)
    grad = np.full((*shape, 3), np.nan)
    grad[:, :, :6] = [0.0, 0.0, 2.0]
    tb = TT.TauBehind.__new__(TT.TauBehind)
    tb.site = SimpleNamespace(vox=np.full(3, H))
    V = np.array([[2.4, 2.6, 5.3], [2.0, 2.0, 7.5]]) * H
    D, has = tb.directions(V, grad)
    assert has.tolist() == [True, False]
    np.testing.assert_allclose(D[0], [0.0, 0.0, 1.0])
