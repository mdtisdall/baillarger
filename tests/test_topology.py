"""Correctness and speed tests for baillarger.topology.

The random tests compare is_simple with an independent global check: in a
volume that is background outside the 3x3x3 neighbourhood of the centre, a
voxel is simple exactly when adding it (or removing it) keeps the Euler
characteristic, the number of object components and the number of background
components.
"""

import time
from collections.abc import Callable, Iterator

import numpy as np
import numpy.typing as npt
import pytest
from scipy import ndimage
from skimage.measure import euler_number

from baillarger._jit import njit
from baillarger.topology import (
    CONN_6_26,
    CONN_26_6,
    euler_6_26,
    euler_26_6,
    is_simple,
    topological_numbers,
)

type Bool3 = npt.NDArray[np.bool_]

CONNS = (CONN_26_6, CONN_6_26)
EULER: dict[int, Callable[[Bool3], int]] = {
    CONN_26_6: euler_26_6,
    CONN_6_26: euler_6_26,
}
# object structure and background structure of each connectivity pair
STRUCT = {
    CONN_26_6: (np.ones((3, 3, 3), bool), ndimage.generate_binary_structure(3, 1)),
    CONN_6_26: (ndimage.generate_binary_structure(3, 1), np.ones((3, 3, 3), bool)),
}


def topology(vol: Bool3, conn: int) -> tuple[int, int, int]:
    """(Euler characteristic, object components, background components)."""
    so, sb = STRUCT[conn]
    return EULER[conn](vol), ndimage.label(vol, so)[1], ndimage.label(~vol, sb)[1]


def square_ring(n: int = 9, side: int = 5, z: int = 4) -> tuple[Bool3, int, int]:
    """Square ring of voxels (one voxel thick) in the plane z, in an n^3 volume."""
    v = np.zeros((n, n, n), bool)
    a = (n - side) // 2
    b = a + side - 1
    v[a : b + 1, a : b + 1, z] = True
    v[a + 1 : b, a + 1 : b, z] = False
    return v, a, b


# ---------------------------------------------------------------- Euler characteristic


def test_euler_small_shapes() -> None:
    v = np.zeros((5, 5, 5), bool)
    v[2, 2, 2] = True
    assert euler_26_6(v) == 1 and euler_6_26(v) == 1
    v[1, 1, 1] = True  # corner contact
    assert euler_26_6(v) == 1 and euler_6_26(v) == 2
    v[:] = False
    v[1:4, 1:4, 1:4] = True  # solid block
    assert euler_26_6(v) == 1 and euler_6_26(v) == 1
    v[2, 2, 2] = False  # cavity
    assert euler_26_6(v) == 2 and euler_6_26(v) == 2
    r, _, _ = square_ring()  # one handle
    assert euler_26_6(r) == 0 and euler_6_26(r) == 0
    assert euler_26_6(np.zeros((4, 4, 4), bool)) == 0


def test_euler_object_on_the_array_border() -> None:
    v = np.ones((3, 4, 5), bool)
    assert euler_26_6(v) == 1 and euler_6_26(v) == 1
    v[:, 1:3, 1:4] = False  # tunnel along x
    assert euler_26_6(v) == 0 and euler_6_26(v) == 0


def test_euler_matches_skimage_on_random_volumes() -> None:
    rng = np.random.default_rng(1)
    for _ in range(200):
        v = rng.random(tuple(rng.integers(2, 9, 3))) < rng.uniform(0.2, 0.8)
        assert euler_26_6(v) == euler_number(v, connectivity=3)
        assert euler_6_26(v) == euler_number(v, connectivity=1)


# ---------------------------------------------------------------- hand-made cases


@pytest.mark.parametrize("conn", CONNS)
def test_isolated_voxel_is_not_simple(conn: int) -> None:
    v = np.zeros((5, 5, 5), bool)
    assert not is_simple(v, 2, 2, 2, conn)
    v[2, 2, 2] = True
    assert not is_simple(v, 2, 2, 2, conn)
    assert topological_numbers(v, 2, 2, 2, conn) == (0, 1)


def test_corner_contact_depends_on_connectivity() -> None:
    v = np.zeros((5, 5, 5), bool)
    v[1, 1, 1] = True  # touches (2,2,2) at a corner only
    assert is_simple(v, 2, 2, 2, CONN_26_6)
    assert not is_simple(v, 2, 2, 2, CONN_6_26)


@pytest.mark.parametrize("conn", CONNS)
def test_voxel_on_flat_face_of_block_is_simple(conn: int) -> None:
    v = np.zeros((7, 7, 7), bool)
    v[1:6, 1:6, 1:4] = True  # top face at z = 3
    assert is_simple(v, 3, 3, 4, conn)  # add on the face
    assert is_simple(v, 3, 3, 3, conn)  # remove from the face
    assert is_simple(v, 1, 1, 3, conn)  # remove a corner of the block
    assert not is_simple(v, 3, 3, 2, conn)  # interior: removal makes a cavity


@pytest.mark.parametrize("conn", CONNS)
def test_closing_a_ring_is_not_simple(conn: int) -> None:
    v, a, b = square_ring()
    v[a, a + 2, 4] = False  # open ring: a path
    assert topology(v, conn)[0] == 1
    assert not is_simple(v, a, a + 2, 4, conn)
    v2 = v.copy()
    v2[a, a + 2, 4] = True
    assert topology(v2, conn)[0] == 0
    # the end of the open path is simple
    v[a, a + 3, 4] = False
    assert is_simple(v, a, a + 4, 4, conn)


@pytest.mark.parametrize("conn", CONNS)
def test_filling_the_centre_of_a_ring_is_not_simple(conn: int) -> None:
    v = np.zeros((5, 5, 5), bool)
    v[1:4, 1:4, 2] = True
    v[2, 2, 2] = False  # 3x3 ring around the centre
    assert not is_simple(v, 2, 2, 2, conn)


@pytest.mark.parametrize("conn", CONNS)
def test_closing_a_cavity_is_not_simple(conn: int) -> None:
    v = np.zeros((9, 9, 9), bool)
    v[2:7, 2:7, 2:7] = True
    v[3:6, 3:6, 3:6] = False  # hollow shell
    v[4, 4, 2] = False  # one hole in the shell
    assert topology(v, conn)[2] == 1
    assert not is_simple(v, 4, 4, 2, conn)
    # a shell cell beside the hole is simple
    assert is_simple(v, 3, 4, 2, conn)


@pytest.mark.parametrize("conn", CONNS)
def test_accepts_uint8_input(conn: int) -> None:
    v = np.zeros((7, 7, 7), np.uint8)
    v[1:6, 1:6, 1:4] = 1
    assert is_simple(v, 3, 3, 4, conn)
    assert not is_simple(v, 3, 3, 2, conn)


# ---------------------------------------------------------------- random cases


def random_neighbourhoods(n: int, seed: int) -> Iterator[Bool3]:
    """n random 3x3x3 neighbourhoods at the centre of a 5x5x5 background volume."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        v = np.zeros((5, 5, 5), bool)
        v[1:4, 1:4, 1:4] = rng.random((3, 3, 3)) < rng.uniform(0.1, 0.9)
        yield v


def bm_numbers_scipy(nb: Bool3, conn: int) -> tuple[int, int]:
    """(T, Tb) of the centre of the 3x3x3 array nb, counted with
    scipy.ndimage.label (an implementation independent of baillarger.topology)."""
    off = np.abs(np.indices((3, 3, 3)) - 1).sum(0)  # L1 distance from the centre
    n18, n26, n6 = (off >= 1) & (off <= 2), off >= 1, off == 1
    s6, s26 = ndimage.generate_binary_structure(3, 1), np.ones((3, 3, 3), bool)
    obj, bg = nb & n26, ~nb & n26

    def adjacent(mask: Bool3) -> int:  # 6-components that touch a 6-neighbour
        lab = ndimage.label(mask & n18, s6)[0]
        return len(set(lab[n6].tolist()) - {0})

    if conn == CONN_26_6:
        return ndimage.label(obj, s26)[1], adjacent(bg)
    return adjacent(obj), ndimage.label(bg, s26)[1]


def check_random(conn: int, seed: int, centre: bool) -> int:
    """centre: the value of the centre before the change (False: add, True:
    remove).

    Checks, for each random neighbourhood:
    - the topological numbers equal an independent scipy count;
    - simple implies that the Euler characteristic, the object components and
      the background components do not change;
    - the change of the Euler characteristic when x is added is Tb - T;
    - not simple implies that one of the three changes, except when T == Tb >= 2
      for the (6, 26) pair. Then the Euler characteristic cannot change (Tb - T
      = 0) and the T object pieces and the Tb background pieces can all be
      joined outside N18*: object pieces through the corners of the 3x3x3 cube,
      background pieces through the padding. The global counts then stay the
      same although x is not simple, so these cases are counted, not failed.
      For (26, 6) this cannot happen: all object voxels of the volume are in
      N26*, so T >= 2 always splits the object.
    """
    n_simple = n_not = n_hidden = 0
    for v in random_neighbourhoods(3000, seed):
        v[2, 2, 2] = centre  # is_simple sees the volume before the change
        s = is_simple(v, 2, 2, 2, conn)
        t, tb = topological_numbers(v, 2, 2, 2, conn)
        nb = v[1:4, 1:4, 1:4].copy()
        assert (t, tb) == bm_numbers_scipy(nb, conn)
        assert s == (t == 1 and tb == 1)
        before = topology(v, conn)
        v[2, 2, 2] = not centre
        after = topology(v, conn)
        sign = -1 if centre else 1
        assert after[0] - before[0] == sign * (tb - t)
        info = (conn, centre, t, tb, np.argwhere(nb).tolist())
        if s:
            assert after == before, info
            n_simple += 1
        else:
            n_not += 1
            if after == before:
                assert conn == CONN_6_26 and t == tb >= 2, info
                n_hidden += 1
    assert n_simple > 300 and n_not > 300  # both classes are well sampled
    return n_hidden


@pytest.mark.parametrize("conn", CONNS)
def test_random_adding_centre(conn: int) -> None:
    n = check_random(conn, seed=10 + conn, centre=False)
    print(f"conn {conn}: {n} hidden non-simple cases")


@pytest.mark.parametrize("conn", CONNS)
def test_random_removing_centre(conn: int) -> None:
    n = check_random(conn, seed=20 + conn, centre=True)
    print(f"conn {conn}: {n} hidden non-simple cases")


@pytest.mark.parametrize("conn", CONNS)
def test_result_does_not_depend_on_centre(conn: int) -> None:
    for v in random_neighbourhoods(2000, 30 + conn):
        v[2, 2, 2] = False
        a = is_simple(v, 2, 2, 2, conn), topological_numbers(v, 2, 2, 2, conn)
        v[2, 2, 2] = True
        assert a == (is_simple(v, 2, 2, 2, conn), topological_numbers(v, 2, 2, 2, conn))


def test_duality_of_the_two_pairs() -> None:
    # X is (26,6)-simple exactly when the complement is (6,26)-simple
    for v in random_neighbourhoods(2000, 40):
        w = ~v
        assert is_simple(v, 2, 2, 2, CONN_26_6) == is_simple(w, 2, 2, 2, CONN_6_26)
        t, tb = topological_numbers(v, 2, 2, 2, CONN_26_6)
        assert topological_numbers(w, 2, 2, 2, CONN_6_26) == (tb, t)


# ---------------------------------------------------------------- run time


@njit
def _count_simple(obj: Bool3, pts: npt.NDArray[np.int64], conn: int) -> int:
    n = 0
    for i in range(pts.shape[0]):
        n += is_simple(obj, pts[i, 0], pts[i, 1], pts[i, 2], conn)
    return n


@pytest.mark.parametrize("conn", CONNS)
def test_run_time_one_million_calls(conn: int) -> None:
    rng = np.random.default_rng(5)
    obj = rng.random((64, 64, 64)) < 0.5
    pts = rng.integers(1, 63, (1_000_000, 3))
    _count_simple(obj, pts[:10], conn)  # compile
    t0 = time.perf_counter()
    n = _count_simple(obj, pts, conn)
    dt = time.perf_counter() - t0
    print(f"conn {conn}: 10^6 calls in {dt:.3f} s, {n} simple")
    assert dt < 10.0
