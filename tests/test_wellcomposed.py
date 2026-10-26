"""Tests for baillarger.wellcomposed."""

import numpy as np
from scipy import ndimage as ndi

from baillarger import phantoms as ph
from baillarger.growth import grow_topological, seam_priority
from baillarger.mesh import marching_tetrahedra, topology
from baillarger.topology import CONN_26_6, euler_6_26, euler_26_6
from baillarger.wellcomposed import critical_count, repair


def test_detects_the_three_configurations() -> None:
    a = np.zeros((4, 4, 4), bool)
    a[1, 1, 1] = a[2, 2, 1] = True  # checkerboard square
    assert critical_count(a) > 0
    b = np.zeros((4, 4, 4), bool)
    b[1, 1, 1] = b[2, 2, 2] = True  # opposite corners
    assert critical_count(b) > 0
    c = np.zeros((4, 4, 4), bool)
    c[1:3, 1:3, 1:3] = True
    c[1, 1, 1] = c[2, 2, 2] = False  # complement corners
    assert critical_count(c) > 0
    d = np.zeros((5, 5, 5), bool)
    d[1:4, 1:4, 1:4] = True
    assert critical_count(d) == 0


def test_repair_keeps_topology_and_removes_critical_configurations() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = ndi.binary_opening(rng.random((16, 16, 16)) > 0.45)
        a[0], a[-1], a[:, 0], a[:, -1], a[:, :, 0], a[:, :, -1] = 0, 0, 0, 0, 0, 0
        r, n, left = repair(a, np.ones_like(a), np.ones_like(a))
        assert euler_26_6(r) == euler_26_6(a)
        lab_a = ndi.label(a, np.ones((3, 3, 3)))[1]
        lab_r = ndi.label(r, np.ones((3, 3, 3)))[1]
        assert lab_a == lab_r
        if left == 0:
            assert critical_count(r) == 0
            # well-composed: 6- and 26-connectivity agree
            assert euler_6_26(r) == euler_26_6(r)


def test_repaired_growth_mesh_has_the_digital_topology() -> None:
    for phantom in (ph.p1_kissing_banks, ph.p4_kissing_dark_line, ph.p5_wm_handle):
        rim, flash, _ = phantom()
        pri = seam_priority(rim, flash, ph.VOXEL_MM)
        obj, seam, _ = grow_topological(rim == 2, rim == 3, pri, CONN_26_6)
        r, _, left = repair(obj, (rim == 3) & obj, seam)
        assert left == 0
        assert euler_26_6(r) == euler_26_6(obj)
        V, T = marching_tetrahedra(np.pad(r.astype(float), 1), 0.5)
        top = topology(T, len(V))
        # a closed surface of the object: chi(surface) = 2 chi(object)
        assert top["boundary_loops"] == 0
        assert top["chi"] == 2 * euler_26_6(r)
