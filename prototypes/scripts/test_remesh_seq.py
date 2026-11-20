"""Tests for remesh_seq: a sphere, a U-shaped plate whose inner faces are
closer than the target edge (the remesher must keep them apart), an open
sheet with a box rim (corners stay, the rim stays on the box), and a slab
with a thin fin (the normal check and the outward bound)."""

from __future__ import annotations

import numpy as np
import pytest
import remesh_seq as RS

from baillarger import mesh_checks as MC
from baillarger.mesh import marching_tetrahedra

GAP = 0.25


def sphere() -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(30, dtype=float)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    v, f = marching_tetrahedra(np.sqrt((X - 14.3) ** 2 + (Y - 15.1) ** 2 + (Z - 14.7) ** 2), 10.0)
    return v.astype(np.float64), f.astype(np.int64)


def u_plate() -> tuple[np.ndarray, np.ndarray]:
    """A closed U: two plates joined at one end, their inner faces about
    two voxels apart."""
    a = np.zeros((36, 30, 24))
    a[5:30, 5:25, 6:10] = 1
    a[5:30, 5:25, 13:17] = 1
    a[26:30, 5:25, 6:17] = 1
    v, f = marching_tetrahedra(a, 0.5)
    return v.astype(np.float64), f.astype(np.int64)


def sheet() -> tuple[np.ndarray, np.ndarray]:
    """A bumpy open sheet over the square [0, 20]², spacing 0.5."""
    s = np.linspace(0.0, 20.0, 41)
    X, Y = np.meshgrid(s, s, indexing="ij")
    V = np.c_[X.ravel(), Y.ravel(), (5 + np.sin(X / 4) * np.cos(Y / 5)).ravel()]
    n = len(s)
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
            F += [[a, b, c], [a, c, d]]
    return V, np.array(F, np.int64)


def run(V: np.ndarray, F: np.ndarray, target: float, corner: np.ndarray | None = None
        ) -> tuple[RS.Remesher, list[dict]]:
    white = RS.White(V, F, np.full(len(V), target))
    rm = RS.Remesher(white, V.copy(), F.copy(), GAP, corner=corner)
    assert rm.chk.check_all().ok  # the start is valid
    log = RS.remesh(rm, 1.0, max_passes=12)
    return rm, log


def assert_valid(rm: RS.Remesher) -> tuple[np.ndarray, np.ndarray]:
    V, F, _ = rm.chk.export()
    c = MC.check_mesh(V, F, gap=GAP)
    assert c.counts()["intersecting_pairs"] == 0
    assert c.vertex_fold_pairs == 0 and c.edge_fold_pairs == 0 and c.gap_pairs == 0
    n = MC.face_normals(V, F)
    _, _, wf = rm.white.project(V[F].mean(1))
    assert ((n * rm.white.N[wf]).sum(1) > 0).all()  # none turned against white
    return V, F


@pytest.mark.parametrize(("mesh", "target"), [("sphere", 2.5), ("u_plate", 3.0)])
def test_closed_meshes_reach_the_target_and_stay_valid(mesh: str, target: float) -> None:
    V0, F0 = sphere() if mesh == "sphere" else u_plate()
    rm, log = run(V0, F0, target)
    V, F = assert_valid(rm)
    last = log[-1]
    assert last["edge_in_range"] > 0.85, last
    assert len(F) < len(F0) / 4  # coarser
    # the smoothing is not projected: the vertices stay near the white
    # surface, within half the target (it rounds the U plate's 90-degree
    # box edges by about a third of the target; no check bounds it here)
    _, d, _ = rm.white.project(V)
    assert d.max() < target / 2, d.max()
    if mesh == "u_plate":
        # the inner faces are closer than the target: some operations there
        # were refused, and the two plates stay apart
        assert sum(sum(r["refused"].values()) for r in log) > 0


def test_an_open_sheet_keeps_its_corners_and_its_rim_on_the_box() -> None:
    V0, F0 = sheet()
    rim, E = RS.rim_of(F0, len(V0))
    corner = RS.box_corners(V0, rim, E, np.array([0, 0, -100.0]), np.array([20, 20, 100.0]), 1e-9)
    assert corner.sum() == 4
    rm, log = run(V0, F0, 2.0, corner)
    V, F = assert_valid(rm)
    assert log[-1]["edge_in_range"] > 0.8, log[-1]
    # the corners are where they were
    for p in V0[corner]:
        assert np.linalg.norm(V - p, axis=1).min() < 1e-12
    # the rim is on the box
    rim2, _ = RS.rim_of(F, len(V))
    on_box = np.isclose(V[rim2, 0], 0) | np.isclose(V[rim2, 0], 20)
    on_box |= np.isclose(V[rim2, 1], 0) | np.isclose(V[rim2, 1], 20)
    assert on_box.all()


def finger() -> tuple[np.ndarray, np.ndarray]:
    """A slab with a thin fin on top (3 voxels thick, 6 tall): the fin is
    narrower than the target edge."""
    a = np.zeros((24, 20, 22))
    a[3:21, 3:17, 3:8] = 1
    a[7:17, 9:12, 8:14] = 1
    v, f = marching_tetrahedra(a, 0.5)
    return v.astype(np.float64), f.astype(np.int64)


@pytest.mark.parametrize("check", [False, True])
def test_the_normal_check_keeps_orientation_and_bounds_outward(check: bool) -> None:
    """Target 4 on a slab with a fin 3 voxels thick. Without the checks,
    the 3D smoothing moves Σ more than a voxel outward. With theta 30
    degrees and d_out 0.5: no white vertex is more than 0.5 inside Σ's
    outer side, no face is farther from its white normals than the worst
    face of the start, and some operations are refused by the check."""
    V0, F0 = finger()
    white = RS.White(V0, F0, np.full(len(V0), 4.0))
    kw = {"theta_deg": 30.0, "d_out": 0.5} if check else {}
    rm = RS.Remesher(white, V0.copy(), F0.copy(), GAP, **kw)
    start = rm.measures(1.0)
    log = RS.remesh(rm, 1.0, max_passes=4)
    assert_valid(rm)
    last = log[-1]
    assert last["faces"] < len(F0) / 3, last  # coarser
    if not check:
        assert last["outward_max_vox"] > 1.0, last
    else:
        assert last["outward_max_vox"] <= 0.5 + 1e-9, last
        assert last["face_angle_p50_90_99_max"][3] <= start["face_angle_p50_90_99_max"][3] + 1e-6, (start, last)
        assert any(k.endswith("_normal") for r in log for k in r["refused"])
        # the running sums and counts are those of the assignment
        live = np.flatnonzero(rm.chk.F[:, 0] >= 0)
        has = rm.wf >= 0
        C = np.bincount(rm.wf[has], minlength=len(rm.Cf))
        S = np.zeros_like(rm.Sf)
        np.add.at(S, rm.wf[has], rm.Wn[has])
        assert (C[live] == rm.Cf[live]).all()
        assert np.allclose(S[live], rm.Sf[live], atol=1e-9)
        assert all(rm.wf[w] == f for f, ws in rm.fw.items() for w in ws)
