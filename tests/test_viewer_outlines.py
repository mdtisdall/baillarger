"""Tests of baillarger.viewer.outlines on a sphere phantom with a real block index
(marching tetrahedra in blocks of 8 cells) and on one triangle with an exact answer."""

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from baillarger.mesh import marching_tetrahedra_blocks
from baillarger.viewer import outlines as O
from baillarger.white_surf import block_bounds

type Float = npt.NDArray[np.float64]

# voxel (i, j, k) -> world: axes permuted and flipped, 0.2 mm isotropic
AFF = np.array([[0, 0.2, 0, -3.0], [0.2, 0, 0, 2.0], [0, 0, -0.2, 5.0], [0, 0, 0, 1.0]])
C_VOX = np.array([20.3, 19.6, 21.1])
R_VOX = 14.0


def to_mm(v: Float) -> Float:
    out: Float = v @ AFF[:3, :3].T + AFF[:3, 3]
    return out


class Counting(np.ndarray[Any, Any]):
    """An array that counts the rows that slicing hands out."""

    rows = 0

    def __getitem__(self, k: Any) -> Any:
        r = super().__getitem__(k)
        if isinstance(k, slice):
            type(self).rows += len(r)
        return r


def phantom() -> tuple[Float, Any, Float, Float]:
    """Sphere at C_VOX, radius R_VOX voxels: mm vertices, faces, block table, voxel
    vertices."""
    x = np.indices((40, 40, 42)).astype(float)
    field = R_VOX - np.sqrt(((x - C_VOX[:, None, None, None]) ** 2).sum(0))
    v, f, mb = marching_tetrahedra_blocks(field, 0.0, 8)
    bb = block_bounds(v, f, mb)
    table = np.c_[
        mb.origin // 8,
        mb.vertex_start,
        mb.vertex_stop,
        mb.face_start,
        mb.face_stop,
        bb,
    ]
    return to_mm(v), f, table, v


def _subdivide(pts: list[Float], tri: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    cache: dict[tuple[int, int], int] = {}

    def mid(a: int, b: int) -> int:
        k = (min(a, b), max(a, b))
        if k not in cache:
            m = pts[a] + pts[b]
            pts.append(m / np.linalg.norm(m))
            cache[k] = len(pts) - 1
        return cache[k]

    new: list[tuple[int, ...]] = []
    for a, b, c in tri:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        new += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    return new


def icosphere(level: int) -> tuple[Float, Any]:
    t = (1 + 5**0.5) / 2
    v: list[tuple[float, ...]] = [
        (-1, t, 0),
        (1, t, 0),
        (-1, -t, 0),
        (1, -t, 0),
        (0, -1, t),
        (0, 1, t),
    ]
    v += [(0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11)]
    f += [(1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8)]
    f += [(3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9)]
    f += [(4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    pts = [np.array(p, float) / np.linalg.norm(p) for p in v]
    tri: list[tuple[int, ...]] = [tuple(x) for x in f]
    for _ in range(level):
        tri = _subdivide(pts, tri)
    return np.array(pts), np.array(tri)


@pytest.fixture(scope="module")
def sphere() -> dict[str, Any]:
    vmm, f, table, _ = phantom()
    c = to_mm(C_VOX[None])[0]
    r = R_VOX * 0.2
    ov, of = icosphere(1)
    ov = ov * r + c
    return dict(vmm=vmm, f=f, table=table, c=c, r=r, ov=ov, of=of)


def make(s: dict[str, Any], **kw: Any) -> O.Outlines:
    return O.Outlines(s["vmm"], s["f"], s["table"], AFF, s["ov"], s["of"], **kw)


BIG_LO, BIG_HI = [-100.0] * 3, [100.0] * 3


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("d", [-0.9, 0.0123, 1.1])
def test_a_plane_through_the_sphere_gives_a_closed_polygon(
    sphere: dict[str, Any], axis: int, d: float
) -> None:
    s = sphere
    o = make(s).cut(axis, float(s["c"][axis] + d), BIG_LO, BIG_HI)
    assert o.source == "true"
    seg = o.segments
    assert seg.dtype == np.float32 and seg.shape[1:] == (2, 3) and len(seg) > 8
    e = seg.reshape(-1, 3)
    assert np.all(e[:, axis] == np.float32(s["c"][axis] + d))
    # bounds on the distance of an end from the centre: it lies on a face, so it is
    # within the farthest vertex and beyond the nearest face plane
    v, f = s["vmm"], s["f"]
    far = np.linalg.norm(v - s["c"], axis=1).max()
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n /= np.linalg.norm(n, axis=1)[:, None]
    near = np.abs(((v[f[:, 0]] - s["c"]) * n).sum(1)).min()
    assert far / near < 1.1  # the bounds are tight enough to mean something
    dist = np.linalg.norm(e.astype(float) - s["c"], axis=1)
    assert dist.max() <= far + 1e-5 and dist.min() >= near - 1e-5
    # closed: every end is shared by exactly two segments
    _, counts = np.unique(e, axis=0, return_counts=True)
    assert np.all(counts == 2)


def test_a_plane_that_misses_gives_nothing(sphere: dict[str, Any]) -> None:
    s = sphere
    o = make(s).cut(1, float(s["c"][1] + 1.5 * s["r"]), BIG_LO, BIG_HI)
    assert len(o.segments) == 0 and o.source == "true"
    assert encode_len(o) == 0


def encode_len(o: O.Outline) -> int:
    return len(O.encode(o))


def test_only_the_faces_in_the_box_are_read(sphere: dict[str, Any]) -> None:
    s = sphere
    f = np.asarray(s["f"]).view(Counting)
    Counting.rows = 0
    out = O.Outlines(s["vmm"], f, s["table"], AFF, s["ov"], s["of"])
    mm = float(s["c"][2])
    # a small box on the rim of the circle in the plane z = mm
    lo = [float(s["c"][0] + s["r"] - 0.3), float(s["c"][1] - 0.3), mm]
    hi = [float(s["c"][0] + s["r"] + 0.3), float(s["c"][1] + 0.3), mm]
    o = out.cut(2, mm, lo, hi)
    # the blocks that meet the box, counted here from the table in mm
    expect = 0
    for r in s["table"]:
        corners = np.array(
            [[(r[7:10], r[10:13])[i >> k & 1][k] for k in range(3)] for i in range(8)]
        )
        m = to_mm(corners)
        if np.all(m.min(0) <= np.array(hi) + 0.1) and np.all(
            m.max(0) >= np.array(lo) - 0.1
        ):
            expect += int(r[6] - r[5])
    assert 0 < expect < len(f)
    assert o.faces_in_box == expect == o.faces_read == Counting.rows
    assert len(o.segments) > 0
    e = o.segments.reshape(-1, 3)
    assert np.all(e[:, 2] == np.float32(mm))
    # each segment has an end inside the box, and the whole circle has more segments
    ins = (e[:, :2] >= np.array(lo[:2]) - 1e-4) & (e[:, :2] <= np.array(hi[:2]) + 1e-4)
    assert np.all(ins.all(1).reshape(-1, 2).any(1))
    assert len(o.segments) < len(make(s).cut(2, mm, BIG_LO, BIG_HI).segments)


def test_the_overview_above_the_limit(sphere: dict[str, Any]) -> None:
    s = sphere
    mm = float(s["c"][0] + 0.3)
    full = make(s).cut(0, mm, BIG_LO, BIG_HI)
    o = make(s, limit=10).cut(0, mm, BIG_LO, BIG_HI)
    assert o.source == "overview"
    assert o.faces_in_box == full.faces_in_box > 10
    assert 0 < o.faces_read <= len(s["of"]) < full.faces_read
    assert 0 < len(o.segments) < len(full.segments)
    e = o.segments.reshape(-1, 3).astype(float)
    # the ends lie on the overview's faces: within its farthest vertex
    far = np.linalg.norm(s["ov"] - s["c"], axis=1).max()
    assert np.linalg.norm(e - s["c"], axis=1).max() <= far + 1e-5
    # the limit is a count of true faces: equal to it is still true
    at = make(s, limit=full.faces_in_box).cut(0, mm, BIG_LO, BIG_HI)
    assert at.source == "true"


def test_one_triangle_exactly() -> None:
    v = np.array([[0, 0, 0], [2, 0, 1], [0, 2, 1.0]])
    f = np.array([[0, 1, 2]])
    table = np.array([[0, 0, 0, 0, 3, 0, 1, 0, 0, 0, 2, 2, 1.0]])
    out = O.Outlines(v, f, table, np.eye(4), v, f)
    o = out.cut(2, 0.5, [-1, -1, 0.5], [3, 3, 0.5])
    got = {tuple(p) for p in o.segments[0].tolist()}
    assert got == {(1.0, 0.0, 0.5), (0.0, 1.0, 0.5)}
    assert o.faces_read == o.faces_in_box == 1
    # a box beside it keeps nothing; a plane beside it reads nothing
    assert len(out.cut(2, 0.5, [5, 5, 0], [6, 6, 1]).segments) == 0
    assert out.cut(2, 7.0, [-1, -1, 7], [3, 3, 7]).faces_in_box == 0
    # vertices on the plane count as positive: at z = 1 the face has one negative
    # vertex, and the segment runs along the edge that lies in the plane
    on = out.cut(2, 1.0, BIG_LO, BIG_HI).segments
    assert {tuple(p) for p in on[0].tolist()} == {(2.0, 0.0, 1.0), (0.0, 2.0, 1.0)}
    with pytest.raises(ValueError):
        out.cut(3, 0.5, BIG_LO, BIG_HI)


def test_encode_and_the_route(sphere: dict[str, Any]) -> None:
    s = sphere
    out = make(s)
    mm = float(s["c"][1] + 0.2)
    o = out.cut(1, mm, BIG_LO, BIG_HI)
    body = O.encode(o)
    back = np.frombuffer(body, "<f4").reshape(-1, 2, 3)
    assert np.array_equal(back, o.segments) and len(body) == 24 * len(o.segments)
    q = {"axis": "1", "mm": repr(mm), "box": "-100,-100,-100,100,100,100"}
    body2, h = O.outline_response(out, q)
    assert body2 == body
    assert h == {
        "X-Outline-Source": "true",
        "X-Faces-Read": str(o.faces_read),
        "X-Faces-In-Box": str(o.faces_in_box),
        "X-Segments": str(len(o.segments)),
    }
    for bad in (
        {**q, "axis": "3"},
        {**q, "axis": "x"},
        {**q, "mm": "nan"},
        {**q, "mm": "1,2"},
        {**q, "box": "1,2,3"},
        {**q, "box": "a,b,c,d,e,f"},
        {k: v for k, v in q.items() if k != "box"},
    ):
        with pytest.raises(ValueError):
            O.outline_response(out, bad)
