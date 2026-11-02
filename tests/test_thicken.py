"""Tests for baillarger.handles beyond the handle removal itself: fills that
may not take given voxels (remove_handles, nofill) and the thickening of thin
WM (thin_voxels, thicken), on shapes whose answer is known exactly."""

import numpy as np
import numpy.typing as npt

from baillarger import handles as H

type Bool3 = npt.NDArray[np.bool_]


def plate() -> Bool3:
    """A plate 36 x 36 x 4 with a 7 x 7 hole: a fill of 49, a cut of 60."""
    a = np.zeros((40, 40, 12), bool)
    a[2:38, 2:38, 4:8] = True
    a[16:23, 16:23, 4:8] = False
    return a


def cube() -> Bool3:
    a = np.zeros((20, 20, 20), bool)
    a[5:15, 5:15, 5:15] = True
    return a


def test_a_fill_that_would_take_forbidden_voxels_becomes_a_cut() -> None:
    a = plate()
    assert [m.kind for m in H.remove_handles(a).moves] == ["fill"]
    hole = np.zeros_like(a)
    hole[16:23, 16:23, :] = True  # the hole is cortex: no fill may take it
    r = H.remove_handles(a, nofill=hole)
    assert [(m.kind, len(m.voxels)) for m in r.moves] == [("cut", 60)]
    assert H.kuhn_betti(r.obj) == (1, 0, 0) and not (r.obj & ~a).any()
    elsewhere = np.zeros_like(a)
    elsewhere[0] = True  # forbidden voxels the fill does not need
    r = H.remove_handles(a, nofill=elsewhere)
    assert [(m.kind, len(m.voxels)) for m in r.moves] == [("fill", 49)]


def test_the_staircase_sets_a_floor_on_s() -> None:
    """With s = 1 voxel a cube's edges are all thin; from 1.5 only its
    corners are; from 1.8 nothing."""
    assert H.thin_voxels(cube(), 1.0).sum() == 104
    corners = {(x, y, z) for x in (5, 14) for y in (5, 14) for z in (5, 14)}
    assert {tuple(p) for p in np.argwhere(H.thin_voxels(cube(), 1.5))} == corners
    assert not H.thin_voxels(cube(), 1.8).any()
    assert not H.thicken(cube(), 1.8).added.any()


def test_a_thread_becomes_a_tube_of_radius_s() -> None:
    c = np.zeros((30, 30, 40), bool)
    c[5:15, 5:15, 5:15] = True
    c[10, 10, 15:25] = True  # a thread on top
    t = H.thicken(c, 2.0)
    disc = {(10 + x, 10 + y) for x in range(-2, 3) for y in range(-2, 3)}
    disc = {p for p in disc if (p[0] - 10) ** 2 + (p[1] - 10) ** 2 <= 4}
    assert {tuple(p) for p in np.argwhere(t.obj[:, :, 20])} == disc
    assert H.kuhn_betti(t.obj) == (1, 0, 0) and not (c & ~t.obj).any()
    assert not t.blocked.any() and not H.thin_voxels(t.obj, 2.0).any()


def test_a_blade_is_thickened_to_2s_with_its_sides_flat() -> None:
    b = np.zeros((30, 30, 30), bool)
    b[5:25, 5:25, 15] = True
    t = H.thicken(b, 2.0)
    assert t.obj[8:22, 8:22, 13:18].all()
    assert not t.obj[8:22, 8:22, 12].any() and not t.obj[8:22, 8:22, 18].any()
    assert H.kuhn_betti(t.obj) == (1, 0, 0) and not t.blocked.any()


def test_two_blades_close_together_are_not_joined() -> None:
    """Thickening each to 2s would close the gap of 3 between them: the
    middle of the gap is blocked, and the two stay apart."""
    b = np.zeros((30, 30, 30), bool)
    b[5:25, 5:25, 13] = True
    b[5:25, 5:25, 17] = True
    t = H.thicken(b, 2.0)
    assert H.kuhn_betti(t.obj) == (2, 0, 0)
    assert t.obj[8:22, 8:22, 11:15].all() and t.obj[8:22, 8:22, 16:20].all()
    assert not t.obj[:, :, 15].any() and t.blocked[8:22, 8:22, 15].all()


def test_thickening_keeps_the_topology_of_random_objects() -> None:
    rng = np.random.default_rng(5)
    for seed in range(4):
        a = np.zeros((24, 24, 24), bool)
        a[5:19, 5:19, 5:19] = rng.random((14, 14, 14)) > 0.6
        t = H.thicken(a, 1.5)
        assert H.kuhn_betti(t.obj) == H.kuhn_betti(a), seed
        assert not (a & ~t.obj).any() and t.added.any()
        assert not (t.blocked & t.obj).any()
