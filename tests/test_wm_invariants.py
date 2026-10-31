"""Tests for the WM invariants in baillarger.handles: fills that may not take
given voxels (remove_handles, nofill) and the opening for thickness and gaps
(open_thin), on shapes whose answer is known exactly."""

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


def test_the_staircase_sets_a_floor_on_r() -> None:
    """With r = 1 voxel a cube's edges are all thin (and, a ring of edges, not
    removable: one exception); from 1.5 only its corners are; from 1.8
    nothing."""
    o = H.open_thin(cube(), 1.0, 0.0)
    assert not o.removed.any() and [(m.kind, len(m.voxels)) for m in o.exceptions] == [
        ("thin", 104)
    ]
    o = H.open_thin(cube(), 1.5, 0.0)
    corners = {(x, y, z) for x in (5, 14) for y in (5, 14) for z in (5, 14)}
    assert {tuple(p) for p in np.argwhere(o.removed)} == corners and not o.exceptions
    assert not H.open_thin(cube(), 1.8, 0.0).removed.any()


def test_a_thread_is_removed_and_a_neck_kept_as_an_exception() -> None:
    c = np.zeros((40, 30, 30), bool)
    c[5:15, 5:15, 5:15] = True
    c[10, 10, 15:20] = True  # a thread on top
    c[25:35, 5:15, 5:15] = True
    c[15:25, 10, 10] = True  # a neck one voxel across
    o = H.open_thin(c, 1.8, 0.0)
    assert not o.obj[10, 10, 15:20].any()
    assert o.obj[15:25, 10, 10].all()
    assert [(m.kind, len(m.voxels)) for m in o.exceptions] == [("thin", 10)]
    assert {tuple(p) for p in o.exceptions[0].voxels} == {
        (x, 10, 10) for x in range(15, 25)
    }
    assert H.kuhn_betti(o.obj) == (1, 0, 0) and not (o.obj & ~c).any()


def test_a_one_voxel_gap_is_widened() -> None:
    s = np.zeros((30, 20, 20), bool)
    s[3:27, 5:15, 5:15] = True
    s[15, 5:15, 5:15] = False  # a cut one voxel wide
    o = H.open_thin(s, 0.0, 1.5)
    gone = np.argwhere(o.removed)
    assert set(gone[:, 0].tolist()) == {14, 16} and len(gone) == 200
    assert H.kuhn_betti(o.obj) == (2, 0, 0) and not o.exceptions
    assert not H.open_thin(cube(), 0.0, 1.5).removed.any()  # no gap, no change


def test_the_opening_keeps_the_topology_and_reaches_a_fixed_point() -> None:
    rng = np.random.default_rng(5)
    for seed in range(4):
        a = np.zeros((20, 20, 20), bool)
        a[3:17, 3:17, 3:17] = rng.random((14, 14, 14)) > 0.35
        o = H.open_thin(a, 1.5, 1.5)
        assert H.kuhn_betti(o.obj) == H.kuhn_betti(a), seed
        assert not (o.obj & ~a).any()
        again = H.open_thin(o.obj, 1.5, 1.5)
        assert not again.removed.any(), seed  # nothing removable is left
        assert [len(m.voxels) for m in again.exceptions] == [
            len(m.voxels) for m in o.exceptions
        ]
