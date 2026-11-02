"""Tests for baillarger.handles beyond the handle removal itself: fills that
may not take given voxels (remove_handles, nofill), the thickening of thin
WM (thin_voxels, thicken), the widening of gaps (gap_voxels, widen_gaps)
and the removal of diagonal-only contacts
(pinched, well_compose), on shapes whose answer is known exactly."""

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


def two_cubes(second: tuple[slice, slice, slice]) -> Bool3:
    a = np.zeros((20, 20, 20), bool)
    a[5:10, 5:10, 5:10] = True
    a[second] = True
    return a


def test_each_kind_of_diagonal_contact_is_resolved_minimally() -> None:
    """Two cubes touching along an edge or at a corner. Where the Kuhn
    adjacency joins them (offsets in {0, 1}^3), the neck is filled; where it
    does not, they are parted."""
    cases = [
        # (second cube, contact voxels, Betti, added, removed)
        ((slice(10, 15), slice(10, 15), slice(5, 10)), 20, (1, 0, 0), 5, 0),
        ((slice(10, 15), slice(0, 5), slice(5, 10)), 20, (2, 0, 0), 0, 5),
        ((slice(10, 15), slice(10, 15), slice(10, 15)), 8, (1, 0, 0), 2, 0),
        ((slice(10, 15), slice(0, 5), slice(10, 15)), 8, (2, 0, 0), 0, 1),
    ]
    for second, n, betti, added, removed in cases:
        a = two_cubes(second)
        assert H.pinched(a).sum() == n and H.kuhn_betti(a) == betti
        w = H.well_compose(a)
        assert [int(w.added.sum()), int(w.removed.sum())] == [added, removed]
        assert not w.left.any() and H.kuhn_betti(w.obj) == betti


def test_well_compose_keeps_the_topology_and_ends() -> None:
    rng = np.random.default_rng(1)
    for seed in range(3):
        a = np.zeros((24, 24, 24), bool)
        a[4:20, 4:20, 4:20] = rng.random((16, 16, 16)) > 0.5
        w = H.well_compose(a)
        assert H.kuhn_betti(w.obj) == H.kuhn_betti(a), seed
        assert w.left.sum() < H.pinched(a).sum()
        assert np.array_equal(w.left, H.pinched(w.obj))


def test_contacts_are_found_in_every_plane() -> None:
    """Two cubes touching along an edge parallel to each axis in turn."""
    for ax in range(3):
        a = two_cubes((slice(10, 15), slice(10, 15), slice(5, 10)))
        a = np.moveaxis(a, 2, ax)
        assert H.pinched(a).sum() == 20, ax
        w = H.well_compose(a)
        assert not w.left.any() and H.kuhn_betti(w.obj) == H.kuhn_betti(a), ax


def test_a_one_voxel_cut_is_widened_to_2s() -> None:
    s = np.zeros((30, 20, 20), bool)
    s[3:27, 5:15, 5:15] = True
    s[15, 5:15, 5:15] = False  # a cut one voxel wide
    g = H.gap_voxels(s, 1.5)
    # all of the cut but its 4 corners, which a ball in the open background
    # beside the bar holds
    corners = {(15, y, z) for y in (5, 14) for z in (5, 14)}
    assert {tuple(p) for p in np.argwhere(g)} == {
        (15, y, z) for y in range(5, 15) for z in range(5, 15)
    } - corners
    w = H.widen_gaps(s, 1.5)
    gone = np.argwhere(w.removed)
    assert set(gone[:, 0].tolist()) == {14, 16} and len(gone) == 200
    assert H.kuhn_betti(w.obj) == (2, 0, 0) and not w.blocked.any()


def test_a_pit_and_a_wider_gap_are_not_gaps() -> None:
    p = cube()
    p[9:11, 9:11, 12:15] = False  # a pit 2 x 2 in the top face
    assert not H.gap_voxels(p, 1.5).any()
    b = np.zeros((30, 30, 30), bool)
    b[5:25, 5:25, 10:14] = True
    b[5:25, 5:25, 16:20] = True  # two blades 2 voxels apart
    assert not H.gap_voxels(b, 2.0).any()


def test_a_gap_voxel_closing_a_ring_is_found_and_the_ring_kept() -> None:
    """A ring with a one-voxel slit: the slit is a gap, but the wall beside
    it is the ring's only connection on that side, so the widening cuts what
    it can and keeps the topology."""
    a = np.zeros((30, 30, 12), bool)
    a[5:25, 5:25, 4:8] = True
    a[10:20, 10:20, 4:8] = False  # the hole: a ring
    a[15, 5:10, 4:8] = False  # a slit through one side
    w = H.widen_gaps(a, 1.5)
    assert w.gaps[15, 5:10, 4:8].sum() > 0
    assert H.kuhn_betti(w.obj) == H.kuhn_betti(a) == (1, 0, 0)
