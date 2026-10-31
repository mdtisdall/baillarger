"""Tests for the variations of the handle fixes (baillarger.handles): the fill cost
of remove_handles, thicken (a cut's gap widened, a fill's plug thickened) and
trim_threads, on shapes whose answer is known exactly."""

import numpy as np
import numpy.typing as npt

from baillarger import handles as H

type Bool3 = npt.NDArray[np.bool_]


def ring() -> Bool3:
    """A square ring, 5 voxels wide and 6 high, round a 14 x 14 hole."""
    a = np.zeros((30, 30, 12), bool)
    a[3:27, 3:27, 3:9] = True
    a[8:22, 8:22, :] = False
    return a


def plate() -> Bool3:
    """A plate 36 x 36 x 4 with a 7 x 7 hole: a fill of 49, a cut of 60."""
    a = np.zeros((40, 40, 12), bool)
    a[2:38, 2:38, 4:8] = True
    a[16:23, 16:23, 4:8] = False
    return a


def kinds(r: H.HandleRemoval) -> list[tuple[str, int]]:
    return [(m.kind, len(m.voxels)) for m in r.moves]


def test_the_fill_cost_moves_the_choice_between_cut_and_fill() -> None:
    assert kinds(H.remove_handles(plate())) == [("fill", 49)]
    r = H.remove_handles(plate(), fill_cost=2.0)  # 98 against 60
    assert kinds(r) == [("cut", 60)] and H.kuhn_betti(r.obj) == (1, 0, 0)
    assert not (r.obj & ~plate()).any()
    assert kinds(H.remove_handles(ring())) == [("cut", 30)]
    r = H.remove_handles(ring(), fill_cost=0.1)  # 19.6 against 30
    assert kinds(r) == [("fill", 196)] and H.kuhn_betti(r.obj) == (1, 0, 0)


def rod_across_a_slit() -> tuple[Bool3, Bool3, Bool3]:
    """The ring closed by a rod one voxel across a slit 9 wide: the object, the
    result of remove_handles and its cut (one voxel, at the rod's end)."""
    a = ring()
    a[3:8, 10:19, 3:9] = False
    a[5, 10:19, 6] = True
    r = H.remove_handles(a)
    return a, r.obj, a & ~r.obj


def test_a_cut_is_widened_by_simple_points() -> None:
    _, obj, cut = rod_across_a_slit()
    assert np.argwhere(cut).tolist() == [[5, 18, 6]]
    t = H.thicken(obj, cut, 1.0, add=False)
    # the rod's next voxel and the ring's face voxel across the cut
    assert np.argwhere(obj & ~t).tolist() == [[5, 17, 6], [5, 19, 6]]
    assert not (t & ~obj).any() and H.kuhn_betti(t) == (1, 0, 0)
    t2 = H.thicken(obj, cut, 2.0, add=False)
    gone = np.argwhere(obj & ~t2)
    assert len(gone) == 12 and H.kuhn_betti(t2) == (1, 0, 0)
    assert (np.linalg.norm(gone - [5, 18, 6], axis=1) <= 2.0).all()
    assert H.thicken(obj, cut, 0.0, add=False).tolist() == obj.tolist()


def test_a_fill_is_thickened_by_simple_points() -> None:
    """A block 10 thick drilled by a 3 x 3 tunnel narrowed to 1 x 1 in its
    middle: the fill is that one voxel; R 1 adds the tunnel's voxels above and
    below it."""
    b = np.zeros((20, 20, 20), bool)
    b[4:16, 4:16, 5:15] = True
    b[9:12, 9:12, :] = False
    b[9:12, 9:12, 10] = True
    b[10, 10, 10] = False
    r = H.remove_handles(b)
    fill = r.obj & ~b
    assert np.argwhere(fill).tolist() == [[10, 10, 10]]
    t = H.thicken(r.obj, fill, 1.0, add=True)
    assert np.argwhere(t & ~r.obj).tolist() == [[10, 10, 9], [10, 10, 11]]
    t2 = H.thicken(r.obj, fill, 2.0, add=True)
    added = np.argwhere(t2 & ~r.obj)
    assert len(added) == 20 and H.kuhn_betti(t2) == (1, 0, 0)
    assert (np.linalg.norm(added - [10, 10, 10], axis=1) <= 2.0).all()


def test_simple_points_never_change_the_topology() -> None:
    """Widening every voxel of a random object as a seed keeps its Betti
    numbers, adding or removing."""
    rng = np.random.default_rng(3)
    for seed in range(4):
        a = np.zeros((18, 18, 18), bool)
        a[3:15, 3:15, 3:15] = rng.random((12, 12, 12)) > 0.4
        seeds = np.zeros_like(a)
        seeds[tuple(rng.integers(3, 15, (3, 5)))] = True
        for add in (True, False):
            assert H.kuhn_betti(H.thicken(a, seeds, 2.5, add)) == H.kuhn_betti(a)
        assert H.kuhn_betti(H.trim_threads(a, 1.0)) == H.kuhn_betti(a), seed


def test_threads_are_trimmed_and_bridges_kept() -> None:
    """Two blocks joined by a neck one voxel across, a thread 5 long on top of
    one, and a rod by itself: the thread goes but for its base (the opening
    by a ball of radius 1 reaches one voxel above the block's face); the neck
    (it joins two thick parts) and the rod (no part thicker than r) stay
    whole."""
    c = np.zeros((40, 30, 30), bool)
    c[5:15, 5:15, 5:15] = True
    c[10, 10, 15:20] = True  # the thread
    c[25:35, 5:15, 5:15] = True
    c[15:25, 10, 10] = True  # the neck
    c[5:15, 25, 25] = True  # the rod
    t = H.trim_threads(c, 1.0)
    assert np.argwhere(c & ~t).tolist() == [[10, 10, z] for z in range(16, 20)]
    assert not (t & ~c).any() and H.kuhn_betti(t) == H.kuhn_betti(c) == (2, 0, 0)
    assert H.trim_threads(c, 0.0).tolist() == c.tolist()
