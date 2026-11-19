"""Exact tests of the coarse Σ start (sigma_mesh.py) and of T5's refinement
(top_solve.refine_marks, refine): the sizing on a sphere, edge flips on a
flat sheet, closest points, the deviation check, and midpoint splits. The
whole start runs on real site crops (diagnostics), not here."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import sigma_mesh as SM
import top_solve as TSV
import top_surface as TS
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
from test_top_tau import H, sheet

from baillarger import mesh_checks as MC


def _grid(n: int, h: float) -> tuple[np.ndarray, np.ndarray]:
    """A flat n × n vertex grid at z = 0 (spacing h), normal +z."""
    xs = np.arange(n) * h
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    V = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    f = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
            f += [(a, b, c), (a, c, d)]
    return V, np.array(f, np.int64)


def _mesh(V: np.ndarray, F: np.ndarray) -> TS.Mesh:
    return TS.Mesh(V.copy(), F.copy(), V.copy(), np.zeros(len(V), np.int64), np.zeros(len(V)))


def _no_deviation(eps: float) -> SimpleNamespace:
    return SimpleNamespace(eps=eps, deviation=lambda V, F, ch: np.zeros(len(V), bool))


def test_sizing_on_a_sphere() -> None:
    """A sphere of radius R = 20 voxels: κ = 1/R everywhere and no facing
    sheet, so the target is 3/4 sqrt(8 (h/4) R) = 0.75 sqrt(40) h = 4.74 h
    (below the cap 5 h)."""
    n, R = 48, 20.0
    g = np.indices((n, n, n)).astype(float) - (n - 1) / 2
    d = np.sqrt((g**2).sum(0)) - R
    v, f, nrm, _ = marching_cubes(-d, 0.0)
    v = v.astype(np.float64) - (n - 1) / 2
    nrm = v / np.linalg.norm(v, axis=1, keepdims=True)  # outward
    # marching_cubes faces are oriented for its own gradient; make them outward
    fn = MC.face_normals(v, f.astype(np.int64))
    if (fn * v[f].mean(1)).sum() < 0:
        f = f[:, ::-1]
    site = SimpleNamespace(v=v, f=f.astype(np.int64), vox=np.full(3, H), normal=nrm)
    L = SM.sizing(site)
    want = 0.75 * np.sqrt(8 * 0.25 * H * R * H)
    np.testing.assert_allclose(np.median(L), want, rtol=0.05)
    assert np.percentile(L, 5) > 0.85 * want and np.percentile(L, 95) < 1.15 * want


def test_flips_on_a_flat_sheet() -> None:
    """A flat grid with one interior vertex pushed along a diagonal: the
    flips do not lower the smallest angle, keep every face's normal +z (the
    orientation (a, b, c), (b, a, d) -> (a, d, c), (d, b, c)), keep the
    vertex set and the area, and break no check."""
    V, F = _grid(6, H)
    i = 2 * 6 + 2
    V[i, :2] += 0.42 * H  # toward the (3, 3) vertex
    m = _mesh(V, F)
    a0 = SM._angles(V, F).min()
    area0 = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum()
    nf = SM.flip_pass(m, _no_deviation(0.25 * H), 0.25 * H, MC.FOLD_COS)
    assert nf > 0
    assert len(m.F) == len(F)
    fn = MC.face_normals(m.V, m.F)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True)
    np.testing.assert_allclose(fn[:, 2], 1.0, atol=1e-12)
    assert SM._angles(m.V, m.F).min() >= a0
    area = np.linalg.norm(np.cross(m.V[m.F[:, 1]] - m.V[m.F[:, 0]], m.V[m.F[:, 2]] - m.V[m.F[:, 0]]), axis=1).sum()
    np.testing.assert_allclose(area, area0, rtol=1e-12)
    assert sum(TS.counts(m, 0.25 * H, MC.FOLD_COS).values()) == 0


def test_no_flip_off_the_plane() -> None:
    """The same grid with the pushed vertex also raised by h: the
    diagonals of the quads around it pass more than eps = h/4 apart (the
    edge (1, 1)-(2.42, 2.42) is at 0.35 h where it passes the other
    diagonal), so those edges stay."""
    V, F = _grid(6, H)
    i = 2 * 6 + 2
    V[i, :2] += 0.42 * H
    V[i, 2] += 1.0 * H
    m = _mesh(V, F)
    SM.flip_pass(m, _no_deviation(0.25 * H), 0.25 * H, MC.FOLD_COS)
    ring0 = set(map(tuple, np.sort(F[(F == i).any(1)], 1)))
    ring1 = set(map(tuple, np.sort(m.F[(m.F == i).any(1)], 1)))
    assert ring0 == ring1


def test_segment_distance() -> None:
    p0, p1 = np.array([[0.0, 0, 0], [0, 0, 0]]), np.array([[1.0, 0, 0], [1, 0, 0]])
    q0, q1 = np.array([[0.5, -1, 0.3], [2.0, 1, 0]]), np.array([[0.5, 1, 0.3], [3.0, 1, 0]])
    np.testing.assert_allclose(SM.segment_distance(p0, p1, q0, q1), [0.3, np.sqrt(2.0)], atol=1e-12)


def test_closest_on_a_triangle() -> None:
    V = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    F = np.array([[0, 1, 2]])
    P = np.array([[0.2, 0.3, 0.5], [2.0, -1.0, 0.0], [-0.5, 0.5, 1.0]])
    q, d, face = SM.closest_on(V, F, cKDTree(V[F].mean(1)), P, 4)
    np.testing.assert_allclose(q, [[0.2, 0.3, 0.0], [1.0, 0.0, 0.0], [0.0, 0.5, 0.0]], atol=1e-12)
    np.testing.assert_allclose(d, [0.5, np.sqrt(2.0), np.sqrt(1.25)], atol=1e-12)
    assert face.tolist() == [0, 0, 0]


def test_deviation_flags_a_cut_bump() -> None:
    """White: a flat grid with one vertex raised by 0.3 h. Σ: the flat grid.
    That white vertex is 0.3 h > eps = h/4 from Σ: the faces of Σ nearest
    to it are flagged, nothing far away; with a raise of 0.2 h, nothing."""
    V, F = _grid(7, H)
    i = 3 * 7 + 3
    for raise_h, flagged in ((0.3, True), (0.2, False)):
        W = V.copy()
        W[i, 2] = raise_h * H
        ref = SM.Reference.__new__(SM.Reference)
        ref.eps, ref.V, ref.F, ref.vtree = 0.25 * H, W, F, cKDTree(W)
        bad = ref.deviation(V, F, np.ones(len(V), bool))
        assert bad.any() == flagged
        if flagged:
            assert np.linalg.norm(V[bad] - V[i], axis=1).max() <= np.sqrt(2) * H + 1e-12


def test_refine_splits_marked_faces_at_midpoints() -> None:
    """Opposite forces on the two ends of an edge mark its faces (force
    variation); refine splits their longest edges at the midpoints, so the
    shape does not change, and the mesh passes the checks."""
    v, f = sheet(6, 5.3, True)
    V = v * H
    m = _mesh(V, f)
    g = np.zeros_like(V)
    a, b = f[10, 0], f[10, 1]
    g[a, 2], g[b, 2] = 1.0, -1.0
    s = TSV.SolveSettings()
    mk, cnt = TSV.refine_marks(m, g, np.zeros(len(V), bool), np.full(3, H), s)
    assert cnt["force"] > 0 and cnt["facing"] == 0 and cnt["held"] == 0
    nv, nf = len(m.V), len(m.F)
    sp = TSV.refine(m, mk, 1.0 * H)
    assert sp > 0
    assert len(m.V) == nv + sp
    np.testing.assert_allclose(m.V[:, 2], 5.3 * H, atol=1e-12)
    assert len(m.F) > nf
    area = np.linalg.norm(np.cross(m.V[m.F[:, 1]] - m.V[m.F[:, 0]], m.V[m.F[:, 2]] - m.V[m.F[:, 0]]), axis=1).sum()
    np.testing.assert_allclose(area / 2, (7 * H) ** 2, rtol=1e-12)
    assert sum(TS.counts(m, 0.25 * H, MC.FOLD_COS).values()) == 0


def test_refine_keeps_the_minimum_length() -> None:
    """Faces whose longest edge is below 2 × min_len are not split."""
    V, F = _grid(5, H)
    m = _mesh(V, F)
    assert TSV.refine(m, np.ones(len(F), bool), 1.0 * H) == 0
    assert TSV.refine(m, np.ones(len(F), bool), 0.5 * H) > 0
