"""Make a binary voxel object well-composed without changing its topology.

A set is well-composed (Latecki) when neither it nor its complement has a
critical configuration:

- in a 2x2 square of any axis plane, two diagonal voxels in the set and the
  other two not (a checkerboard);
- in a 2x2x2 cube, two opposite corners in the set and the other six not, or
  two opposite corners not in the set and the other six in it.

For a well-composed set, 6- and 26-connectivity give the same components,
handles and cavities, and any marching cubes or marching tetrahedra surface
of it has that topology. Without this, marching tetrahedra can join or split
voxels that touch only at an edge or a corner differently from the (26, 6)
topology that the topological growth keeps, which adds handles to the pial
surface.

repair() removes the critical configurations one voxel at a time. Each change
must be a simple point for (26, 6) (baillarger.topology.is_simple), so the
topology does not change, and only allowed voxels change: removable voxels may
leave the object, addable voxels may join it. Removal is tried first.

Open problem: on the V1V2 patch the repair leaves about 550 critical voxels.
About 246 are between cortex and seam: every single-voxel fix there changes
the topology, so they need moves of several voxels at once. About 267 are
where WM touches missing tissue or the outside, where neither side may change.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from baillarger._jit import njit
from baillarger.topology import CONN_26_6, is_simple

type _U8 = npt.NDArray[np.uint8]
type _Bool = npt.NDArray[np.bool_]


@njit
def _critical_at(o: _U8, x: int, y: int, z: int) -> bool:
    """Whether one of the 12 squares or 8 cubes that contain (x, y, z) is
    critical. o: padded uint8 object array."""
    # 2x2 squares in the three axis planes
    for ax in range(3):
        for da in (-1, 0):
            for db in (-1, 0):
                # the square's corner (lowest indices), in the plane normal to ax
                c = [x, y, z]
                a1 = (ax + 1) % 3
                a2 = (ax + 2) % 3
                c[a1] += da
                c[a2] += db
                p = [0, 0, 0, 0]
                k = 0
                for i in (0, 1):
                    for j in (0, 1):
                        q = [c[0], c[1], c[2]]
                        q[a1] += i
                        q[a2] += j
                        p[k] = o[q[0], q[1], q[2]]
                        k += 1
                # p order: (0,0), (0,1), (1,0), (1,1); diagonals are (0, 3)
                # and (1, 2)
                if p[0] == p[3] and p[1] == p[2] and p[0] != p[1]:
                    return True
    # 2x2x2 cubes
    for dx in (-1, 0):
        for dy in (-1, 0):
            for dz in (-1, 0):
                s = 0
                for i in (0, 1):
                    for j in (0, 1):
                        for k in (0, 1):
                            s += o[x + dx + i, y + dy + j, z + dz + k]
                if s == 2 or s == 6:
                    # exactly two voxels differ from the other six: critical
                    # if they are opposite corners
                    v = 1 if s == 2 else 0
                    for i in (0, 1):
                        for j in (0, 1):
                            for k in (0, 1):
                                if (
                                    o[x + dx + i, y + dy + j, z + dz + k] == v
                                    and o[
                                        x + dx + 1 - i,
                                        y + dy + 1 - j,
                                        z + dz + 1 - k,
                                    ]
                                    == v
                                ):
                                    return True
    return False


@njit
def _critical_voxels(o: _U8) -> npt.NDArray[np.int64]:
    """Flat indices (in the padded array) of the voxels that are in some
    critical configuration."""
    nx, ny, nz = o.shape
    out = []
    for x in range(1, nx - 1):
        for y in range(1, ny - 1):
            for z in range(1, nz - 1):
                if _critical_at(o, x, y, z):
                    out.append(x * ny * nz + y * nz + z)
    return np.array(out, np.int64)


@njit
def _repair(
    o: _U8, removable: _Bool, addable: _Bool, max_passes: int
) -> tuple[int, int]:
    nx, ny, nz = o.shape
    changed = 0
    for _ in range(max_passes):
        crit = _critical_voxels(o)
        if crit.size == 0:
            return changed, 0
        progress = 0
        for idx in crit:
            x = idx // (ny * nz)
            y = (idx // nz) % ny
            z = idx % nz
            if not _critical_at(o, x, y, z):
                continue
            # try the voxel itself and its 26 neighbours: removal first, then
            # addition
            done = False
            for want in (0, 1):
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            a, b, c = x + dx, y + dy, z + dz
                            if (
                                a < 1
                                or b < 1
                                or c < 1
                                or a > nx - 2
                                or b > ny - 2
                                or c > nz - 2
                            ):
                                continue
                            cur = o[a, b, c]
                            if want == 0 and not (cur == 1 and removable[a, b, c]):
                                continue
                            if want == 1 and not (cur == 0 and addable[a, b, c]):
                                continue
                            if not is_simple(o, a, b, c, CONN_26_6):
                                continue
                            o[a, b, c] = 1 - cur
                            # keep the change only if this voxel's
                            # neighbourhood got no new critical voxel and the
                            # voxel we started from is no longer critical
                            ok = not _critical_at(o, x, y, z)
                            if ok:
                                for ex in (-1, 0, 1):
                                    for ey in (-1, 0, 1):
                                        for ez in (-1, 0, 1):
                                            if (
                                                ok
                                                and _critical_at(
                                                    o, a + ex, b + ey, c + ez
                                                )
                                                and not (
                                                    a + ex == x
                                                    and b + ey == y
                                                    and c + ez == z
                                                )
                                            ):
                                                # allowed only if that voxel
                                                # was critical before too
                                                o[a, b, c] = cur
                                                was = _critical_at(
                                                    o, a + ex, b + ey, c + ez
                                                )
                                                o[a, b, c] = 1 - cur
                                                if not was:
                                                    ok = False
                            if ok:
                                changed += 1
                                progress += 1
                                done = True
                                break
                            o[a, b, c] = cur
                        if done:
                            break
                    if done:
                        break
                if done:
                    break
        if progress == 0:
            break
    return changed, _critical_voxels(o).size


def critical_count(obj: npt.ArrayLike) -> int:
    """Number of voxels in a critical configuration."""
    o = np.pad(np.asarray(obj, np.uint8), 2)
    return int(_critical_voxels(o).size)


def repair(
    obj: npt.ArrayLike,
    removable: npt.ArrayLike,
    addable: npt.ArrayLike,
    max_passes: int = 50,
) -> tuple[_Bool, int, int]:
    """Make obj (3D bool) well-composed by simple-point changes.

    removable voxels may leave the object and addable voxels may join it.
    Returns the repaired object, the number of changes and the number of
    voxels still in a critical configuration (0 when the repair succeeded).
    """

    def pad(a: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return np.pad(np.asarray(a), 2)

    o = pad(np.asarray(obj, np.uint8)).copy()
    changed, left = _repair(
        o,
        pad(np.asarray(removable, bool)),
        pad(np.asarray(addable, bool)),
        max_passes,
    )
    return o[2:-2, 2:-2, 2:-2].astype(bool), int(changed), int(left)
