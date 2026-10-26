"""Simple-point test and Euler characteristic for 3D binary voxel objects.

A voxel x is simple for an object X when adding x to X (or removing it) does
not change the topology: the components, handles and cavities of the object
and of the background. The test uses only the 3x3x3 neighbourhood of x,
through the two topological numbers of Bertrand and Malandain:

- (26, 6) pair (object 26-connected, background 6-connected):
  T  = number of 26-components of X in N26*(x) (the 26 neighbours, no centre);
  Tb = number of 6-components of the background in N18*(x) that contain a
       6-neighbour of x.
- (6, 26) pair: the dual,
  T  = number of 6-components of X in N18*(x) that contain a 6-neighbour of x;
  Tb = number of 26-components of the background in N26*(x).

x is simple when T == 1 and Tb == 1. The centre voxel itself is never read, so
the result is the same whether x is in X or not.

Implementation: the 27 voxels of the neighbourhood are packed into the bits of
one integer (bit i = (dx+1)*9 + (dy+1)*3 + (dz+1), the centre is bit 13). For
each bit, a precomputed table gives the bits of its 6- or 26-neighbours inside
the 3x3x3 cube. A flood fill on these bit masks counts the components. There
is no allocation and no Python object, so is_simple can be called from
numba-jitted code.

The connectivity pair is an integer: CONN_26_6 = 26 for (26, 6) and
CONN_6_26 = 6 for (6, 26).

Euler characteristics (both pad the array with background, so the object may
touch the array border):
- euler_26_6: the object as the union of closed unit cubes,
  vertices - edges + faces - cubes of that union.
- euler_6_26: the cubical complex with one vertex per voxel, one edge per pair
  of 6-adjacent voxels, one square per 2x2 block of voxels in a plane and one
  cube per 2x2x2 block, all fully in the object.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from baillarger._jit import njit

CONN_26_6 = 26
CONN_6_26 = 6
_CENTRE = 13

type Volume = npt.NDArray[Any]
"""A 3D array, bool or integer; nonzero is the object."""


def _tables() -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], int, int, int]:
    off = np.array([(i // 9 - 1, (i // 3) % 3 - 1, i % 3 - 1) for i in range(27)])
    adj6 = np.zeros(27, np.int64)
    adj26 = np.zeros(27, np.int64)
    for i in range(27):
        for j in range(27):
            d = np.abs(off[i] - off[j])
            if i == j or d.max() > 1:
                continue
            adj26[i] |= 1 << j
            if d.sum() == 1:
                adj6[i] |= 1 << j
    l1 = np.abs(off).sum(1)
    n26 = sum(1 << i for i in range(27) if i != _CENTRE)
    n18 = sum(1 << i for i in range(27) if 1 <= l1[i] <= 2)
    n6 = sum(1 << i for i in range(27) if l1[i] == 1)
    return adj6, adj26, n6, n18, n26


# neighbour bit masks inside the cube, and the masks of N6*, N18*, N26*
_ADJ6, _ADJ26, _N6, _N18, _N26 = _tables()


@njit
def _count_components(mask: int, adj: npt.NDArray[np.int64], seeds: int) -> int:
    """Number of components of the bit set mask (adjacency table adj) that
    contain a bit of seeds."""
    remaining = np.int64(mask)
    count = 0
    for s in range(27):
        b = np.int64(1) << s
        if (seeds & b) == 0 or (remaining & b) == 0:
            continue
        count += 1
        remaining &= ~b
        stack = b
        while stack != 0:
            i = 0
            while (stack >> i) & 1 == 0:
                i += 1
            stack &= ~(np.int64(1) << i)
            nb = adj[i] & remaining
            remaining &= ~nb
            stack |= nb
    return count


@njit
def _neighbourhood(obj: Volume, x: int, y: int, z: int) -> int:
    """The 3x3x3 neighbourhood of (x, y, z) as 27 bits; the centre bit is 0."""
    m = np.int64(0)
    i = 0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                if i != _CENTRE and obj[x + dx, y + dy, z + dz]:
                    m |= np.int64(1) << i
                i += 1
    return int(m)


@njit
def topological_numbers(
    obj: Volume, x: int, y: int, z: int, conn: int
) -> tuple[int, int]:
    """(T, Tb) of voxel (x, y, z) in obj, for conn = CONN_26_6 (26) or
    CONN_6_26 (6). All 26 neighbours of (x, y, z) must lie inside the array:
    pad the volume before the call."""
    m = _neighbourhood(obj, x, y, z)
    bg = ~m & _N26
    if conn == CONN_26_6:
        return (
            _count_components(m & _N26, _ADJ26, _N26),
            _count_components(bg & _N18, _ADJ6, _N6),
        )
    return (
        _count_components(m & _N18, _ADJ6, _N6),
        _count_components(bg, _ADJ26, _N26),
    )


@njit
def is_simple(obj: Volume, x: int, y: int, z: int, conn: int) -> bool:
    """True when voxel (x, y, z) is simple for the object obj.

    conn: CONN_26_6 (26) for the (26, 6) pair or CONN_6_26 (6) for the (6, 26)
    pair. All 26 neighbours of (x, y, z) must lie inside the array (pad the
    volume); there is no bounds check. The value of obj at (x, y, z) itself is
    not used.
    """
    t, tb = topological_numbers(obj, x, y, z, conn)
    return t == 1 and tb == 1


@njit
def _at(obj: Volume, x: int, y: int, z: int) -> bool:
    """obj[x, y, z], with background outside the array."""
    if (
        x < 0
        or y < 0
        or z < 0
        or x >= obj.shape[0]
        or y >= obj.shape[1]
        or z >= obj.shape[2]
    ):
        return False
    return bool(obj[x, y, z] != 0)


@njit
def _euler_closed_cubes(obj: Volume) -> int:
    # Each lattice cell (vertex, edge, square, cube) of the closed-cube union is
    # counted once if any voxel that contains it is set. A lattice point (i,j,k)
    # is the low corner of voxel (i,j,k); the voxels that contain it are
    # (i-a, j-b, k-c) for a, b, c in {0, 1}.
    nx, ny, nz = obj.shape
    chi = 0
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                v = False
                for a in range(2):
                    for b in range(2):
                        for c in range(2):
                            v = v or _at(obj, i - a, j - b, k - c)
                if not v:  # no cell at this point is used
                    continue
                chi += 1
                # edges from (i,j,k) along x, y, z: 4 voxels contain each
                ex = ey = ez = False
                for a in range(2):
                    for b in range(2):
                        ex = ex or _at(obj, i, j - a, k - b)
                        ey = ey or _at(obj, i - a, j, k - b)
                        ez = ez or _at(obj, i - a, j - b, k)
                chi -= ex + ey + ez
                # squares with low corner (i,j,k), normal x, y, z: 2 voxels
                # contain each
                fx = _at(obj, i, j, k) or _at(obj, i - 1, j, k)
                fy = _at(obj, i, j, k) or _at(obj, i, j - 1, k)
                fz = _at(obj, i, j, k) or _at(obj, i, j, k - 1)
                chi += fx + fy + fz
                chi -= _at(obj, i, j, k)
    return chi


@njit
def _euler_voxel_complex(obj: Volume) -> int:
    # Cells with low corner at voxel (i,j,k): the voxel, 3 edges, 3 squares and
    # 1 cube, each present when all its voxels are set.
    nx, ny, nz = obj.shape
    chi = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if obj[i, j, k] == 0:
                    continue
                chi += 1
                x1 = _at(obj, i + 1, j, k)
                y1 = _at(obj, i, j + 1, k)
                z1 = _at(obj, i, j, k + 1)
                chi -= x1 + y1 + z1
                xy = x1 and y1 and _at(obj, i + 1, j + 1, k)
                xz = x1 and z1 and _at(obj, i + 1, j, k + 1)
                yz = y1 and z1 and _at(obj, i, j + 1, k + 1)
                chi += xy + xz + yz
                chi -= xy and xz and yz and _at(obj, i + 1, j + 1, k + 1)
    return chi


@njit
def euler_26_6(obj: Volume) -> int:
    """Euler characteristic of the union of closed unit cubes of the voxels of
    obj: vertices - edges + faces - cubes. This is the Euler characteristic
    for the (26, 6) pair."""
    return _euler_closed_cubes(obj)


@njit
def euler_6_26(obj: Volume) -> int:
    """Euler characteristic of obj for the (6, 26) pair: voxels - 6-adjacent
    pairs + full 2x2 squares - full 2x2x2 cubes."""
    return _euler_voxel_complex(obj)
