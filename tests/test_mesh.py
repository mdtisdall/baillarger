"""Tests for baillarger.mesh: marching tetrahedra, pl_sample, topology and the
VTK writer."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from baillarger import mesh_intersect as mi
from baillarger.mesh import (
    marching_tetrahedra,
    marching_tetrahedra_blocks,
    pl_sample,
    topology,
    write_vtk,
)

type Float = npt.NDArray[np.float64]


def sphere_field(
    n: int = 40, c: tuple[float, float, float] = (19.3, 20.1, 19.7)
) -> tuple[Float, Float]:
    x = np.arange(n, dtype=float)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    F = np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2)
    return F, np.array(c)


def test_sphere_is_a_closed_genus0_surface_with_outward_normals() -> None:
    F, c = sphere_field()
    V, T = marching_tetrahedra(F, 12.0)
    top = topology(T, len(V))
    assert top == dict(
        components=1, boundary_loops=0, chi=2, genus=0.0, nonmanifold_edges=0
    )
    nrm = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    area = 0.5 * np.linalg.norm(nrm, axis=1).sum()
    assert abs(area / (4 * np.pi * 12**2) - 1) < 0.03
    assert np.all(np.sum(nrm * (V[T].mean(1) - c), 1) > 0)
    assert mi.intersecting_faces(V, T)[1] == 0


def test_levels_of_one_field_do_not_cross() -> None:
    rng = np.random.default_rng(0)
    F, _ = sphere_field()
    # rough field: many ambiguous cubes
    F = F + 0.8 * rng.standard_normal(F.shape)
    meshes = [marching_tetrahedra(F, lv) for lv in (8.0, 8.3, 8.6)]
    for (V1, T1), (V2, T2) in zip(meshes[:-1], meshes[1:], strict=True):
        assert mi.intersecting_faces(V1, T1, V2, T2)[2] == 0
    for V, T in meshes:
        assert mi.intersecting_faces(V, T)[1] == 0
        assert topology(T, len(V))["nonmanifold_edges"] == 0


def test_surface_vertices_lie_on_the_level_of_the_interpolant() -> None:
    rng = np.random.default_rng(1)
    F, _ = sphere_field()
    F = F + 0.5 * rng.standard_normal(F.shape)
    V, T = marching_tetrahedra(F, 9.0)
    assert np.abs(pl_sample(F, V) - 9.0).max() < 1e-6
    # and at triangle centroids (the interpolant is linear on each triangle's
    # tetrahedron)
    assert np.abs(pl_sample(F, V[T].mean(1)) - 9.0).max() < 1e-6


def test_no_level_crossing_gives_an_empty_mesh() -> None:
    V, T = marching_tetrahedra(np.zeros((4, 4, 4)), 1.0)
    assert V.shape == (0, 3) and T.shape == (0, 3)


def test_topology_of_a_torus_and_an_open_sheet() -> None:
    x = np.arange(48, dtype=float)
    X, Y, Z = np.meshgrid(x, x, x[:24], indexing="ij")
    torus = np.hypot(np.hypot(X - 23.7, Y - 24.2) - 12, Z - 11.6)
    V, T = marching_tetrahedra(torus, 5.0)
    top = topology(T, len(V))
    assert top["components"] == 1 and top["chi"] == 0 and top["genus"] == 1.0
    assert top["boundary_loops"] == 0
    # one triangle: a disc with one boundary loop
    top = topology(np.array([[0, 1, 2]]), 3)
    assert top == dict(
        components=1, boundary_loops=1, chi=1, genus=0.0, nonmanifold_edges=0
    )


def read_vtk(path: Path) -> tuple[Float, npt.NDArray[np.int64], dict[str, Float]]:
    """Points, triangles and point data of a legacy ASCII VTK polydata file."""
    lines = path.read_text().split("\n")
    i = next(k for k, s in enumerate(lines) if s.startswith("POINTS"))
    n = int(lines[i].split()[1])
    V = np.array(" ".join(lines[i + 1 : i + 1 + n]).split(), float).reshape(n, 3)
    j = next(k for k, s in enumerate(lines) if s.startswith("POLYGONS"))
    m = int(lines[j].split()[1])
    F = np.array(" ".join(lines[j + 1 : j + 1 + m]).split(), np.int64).reshape(m, 4)
    data = {}
    for k, s in enumerate(lines):
        if s.startswith("SCALARS"):
            data[s.split()[1]] = np.array(lines[k + 2 : k + 2 + n], float)
    return V, F[:, 1:], data


def test_write_vtk_puts_the_points_in_ras(tmp_path: Path) -> None:
    F, _ = sphere_field(20, (9.3, 10.1, 9.7))
    V, T = marching_tetrahedra(F, 6.0)
    # 0.16 mm voxels, axes flipped and swapped, and an offset
    aff = np.array(
        [
            [0.0, -0.16, 0.0, 30.5],
            [0.0, 0.0, 0.16, -12.25],
            [-0.16, 0.0, 0.0, 7.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    depth = V[:, 0].copy()
    depth[:5] = np.nan
    path = tmp_path / "s.vtk"
    write_vtk(str(path), aff, V, T, {"depth": depth})
    text = path.read_text()
    assert text.startswith("# vtk DataFile Version 3.0\ns.vtk\nASCII\n")
    R, T2, data = read_vtk(path)
    ras = V @ aff[:3, :3].T + aff[:3, 3]
    assert np.abs(R - ras).max() <= 5e-5
    assert np.array_equal(T2, T)
    assert np.all(data["depth"][:5] == -1)
    assert np.abs(data["depth"][5:] - depth[5:]).max() <= 5e-5


def canonical(
    V: Float, T: npt.NDArray[np.int64]
) -> tuple[Float, npt.NDArray[np.int64]]:
    """The mesh with its vertices sorted by coordinates and each triangle
    rotated to start at its smallest index (orientation kept), rows sorted:
    the same for any order of the vertices and faces."""
    o = np.lexsort(V.T[::-1])
    rank = np.empty(len(V), np.int64)
    rank[o] = np.arange(len(V))
    R = rank[T]
    r = np.argmin(R, axis=1)
    R = np.take_along_axis(R, (r[:, None] + np.arange(3)[None]) % 3, 1)
    return V[o], R[np.lexsort(R.T[::-1])]


def test_blocks_give_the_one_shot_surface_exactly() -> None:
    rng = np.random.default_rng(2)
    F, _ = sphere_field(30, (14.3, 15.1, 14.7))
    rough = F + 0.8 * rng.standard_normal(F.shape)
    x = np.arange(36, dtype=float)
    X, Y, Z = np.meshgrid(x, x, x[:20], indexing="ij")
    torus = np.hypot(np.hypot(X - 17.7, Y - 18.2) - 9, Z - 9.6)
    binary = (rough < 9.0).astype(np.uint8)  # integer input, points on the level
    cases = ((rough, 9.0), (torus, 4.0), (binary, 0.5), (np.round(F), 9.0))
    for G, lv in cases:
        V0, T0 = canonical(*marching_tetrahedra(G, lv, 1000))
        assert len(T0)
        for b in (1, 3, 7, 16):
            V, T = canonical(*marching_tetrahedra(G, lv, b))
            assert np.array_equal(V, V0)
            assert np.array_equal(T, T0)
            # float32: the same triangles, block by block (vertices that
            # round together make the canonical order ambiguous, so compare
            # with float64 at the same block size)
            V64, T64 = marching_tetrahedra(G, lv, b, np.float64)
            V32, T32 = marching_tetrahedra(G, lv, b, np.float32)
            assert V32.dtype == np.float32 and np.array_equal(T32, T64)
            assert np.abs(V32 - V64).max() < 1e-5


def test_the_orientation_table_holds_for_any_values() -> None:
    """_FLIP depends only on the tetrahedron and the corners' signs: the
    geometric orientation agrees with it for random values."""
    from baillarger.mesh import _CASES, _FLIP, _TETS

    rng = np.random.default_rng(4)
    for ty in range(6):
        P = _TETS[ty].astype(float)
        for pat in range(1, 15):
            sgn = np.array([(pat >> i) & 1 for i in range(4)], bool)
            for _ in range(20):
                mag = rng.uniform(1e-3, 10, 4)
                val = np.where(sgn, mag, -mag)
                order = np.argsort(~sgn, kind="stable")
                o_pos, o_val = P[order], val[order]
                g = np.linalg.solve(P[1:] - P[:1], val[1:] - val[:1])
                pairs_list = dict(_CASES)[int(sgn.sum())]
                for k, pairs in enumerate(pairs_list):
                    q = [
                        o_pos[a]
                        + o_val[a] / (o_val[a] - o_val[b]) * (o_pos[b] - o_pos[a])
                        for a, b in pairs
                    ]
                    nrm = np.cross(q[1] - q[0], q[2] - q[0])
                    assert (float(nrm @ g) < 0) == _FLIP[ty, pat, k]


def test_float32_keeps_grid_points_on_the_level() -> None:
    """Grid points exactly on the level: in float32 the crossings next to
    them round onto the grid point, but the surface stays closed."""
    F, _ = sphere_field(30, (14.3, 15.1, 14.7))
    V, T = marching_tetrahedra(np.round(F), 9.0, 8, np.float32)
    top = topology(T, len(V))
    assert top["components"] == 1 and top["genus"] == 0
    assert top["boundary_loops"] == 0 and top["nonmanifold_edges"] == 0
    assert len(np.unique(V, axis=0)) < len(V)  # some vertices did round together


def test_float32_marching_matches_float64() -> None:
    """The arithmetic in float32 (as on a GPU) against the float64
    reference, on a smooth field: the same triangles, vertices within
    float32 rounding."""
    F, _ = sphere_field(40, (19.3, 20.1, 19.7))
    V64, T64 = marching_tetrahedra(F, 12.0, 8, np.float64)
    V32, T32 = marching_tetrahedra(F, 12.0, 8, np.float32)
    assert np.array_equal(T32, T64)
    assert np.abs(V32 - V64).max() < 1e-5


def test_the_block_index() -> None:
    rng = np.random.default_rng(3)
    F, _ = sphere_field(30, (14.3, 15.1, 14.7))
    F = F + 0.8 * rng.standard_normal(F.shape)
    b = 7
    V, T, B = marching_tetrahedra_blocks(F, 9.0, block=b)
    k = len(B.origin)
    assert B.cells == b and k > 8
    # blocks in Morton order, the ranges contiguous and covering the mesh
    from baillarger.blocks import morton

    assert np.all(np.diff(morton(B.origin // b)) > 0)
    assert B.vertex_start[0] == 0 and B.vertex_stop[-1] == len(V)
    assert B.face_start[0] == 0 and B.face_stop[-1] == len(T)
    assert np.array_equal(B.vertex_start[1:], B.vertex_stop[:-1])
    assert np.array_equal(B.face_start[1:], B.face_stop[:-1])
    owner = np.repeat(np.arange(k), B.vertex_stop - B.vertex_start)
    for i in range(k):
        f = T[B.face_start[i] : B.face_stop[i]]
        # the faces lie in the block's cells
        c = np.floor(V[f].mean(1)).astype(int)
        assert np.all((c >= B.origin[i]) & (c < B.origin[i] + b))
        # they use the block's own vertices or earlier blocks'
        assert owner[f].max() == i or B.vertex_stop[i] == B.vertex_start[i]
        assert owner[f].max() <= i
        # every vertex a block owns is used by its faces
        own = np.arange(B.vertex_start[i], B.vertex_stop[i])
        assert np.isin(own, f).all()


def test_blocks_away_from_the_level_are_skipped() -> None:
    F, _ = sphere_field(40, (8.3, 8.1, 8.7))  # surface in one corner
    V0, T0 = canonical(*marching_tetrahedra(F, 5.0, block=1000))
    V, T, B = marching_tetrahedra_blocks(F, 5.0, block=8)
    assert np.all(B.origin < 16)
    V, T = canonical(V, T)
    assert np.array_equal(V, V0) and np.array_equal(T, T0)
    assert topology(T, len(V))["genus"] == 0


def test_vertex_keys_name_their_grid_edge() -> None:
    from baillarger.mesh import edge_points

    rng = np.random.default_rng(6)
    F, _ = sphere_field(30, (14.3, 15.1, 14.7))
    F = F + 0.8 * rng.standard_normal(F.shape)
    V, T, B = marching_tetrahedra_blocks(F, 9.0, 7)
    k = B.vertex_key
    assert len(k) == len(V) and len(np.unique(k)) == len(k)
    a, b = edge_points(k, F.shape)
    # the two ends are one Kuhn edge apart, and the vertex lies between them
    d = b - a
    assert np.all(np.isin(d, [-1, 0, 1])) and np.all(np.abs(d).sum(1) >= 1)
    t = np.einsum("ij,ij->i", V - a, d) / np.einsum("ij,ij->i", d, d)
    assert np.allclose(a + t[:, None] * d, V) and np.all((t > 0) & (t < 1))
    # the same key for the same vertex whatever the block size
    V2, _, B2 = marching_tetrahedra_blocks(F, 9.0, 16)
    o, o2 = np.argsort(k), np.argsort(B2.vertex_key)
    assert np.array_equal(k[o], B2.vertex_key[o2])
    assert np.array_equal(V[o], V2[o2])


def test_vertex_keys_survive_a_local_edit() -> None:
    """Remeshing after a change in one corner renumbers vertices, but every
    vertex away from the change keeps its key and position."""
    x = np.indices((40, 40, 40)).astype(float)
    ball = (np.sqrt(((x - 19.6) ** 2).sum(0)) < 14).astype(np.uint8)
    edited = ball.copy()
    edited[28:34, 28:34, 16:22] = 1  # a bump on one side
    V0, _, B0 = marching_tetrahedra_blocks(ball, 0.5, 8)
    V1, _, B1 = marching_tetrahedra_blocks(edited, 0.5, 8)
    far = np.all((V0 < 26) | (V0 > 36), axis=1) | (V0[:, 2] < 14) | (V0[:, 2] > 24)
    pos = dict(zip(B1.vertex_key.tolist(), map(tuple, V1), strict=True))
    for key, p in zip(B0.vertex_key[far].tolist(), V0[far], strict=True):
        assert pos[key] == tuple(p)


def test_morton_order_keeps_aligned_cubes_together() -> None:
    from baillarger.blocks import morton

    ijk = np.array(list(np.ndindex(8, 8, 8)))
    code = morton(ijk)
    assert len(np.unique(code)) == len(ijk) and code.max() == len(ijk) - 1
    # within a 2 x 2 x 2 group: C order (i's bit highest)
    assert morton([1, 0, 0]) == 4 and morton([0, 1, 0]) == 2 and morton([0, 0, 1]) == 1
    order = ijk[np.argsort(code)]
    # every aligned cube of 2 or 4 blocks a side is a run of consecutive blocks
    for m in (2, 4):
        cube = order // m
        runs = np.flatnonzero(np.any(np.diff(cube, axis=0) != 0, axis=1))
        assert len(runs) == (8 // m) ** 3 - 1


class SparseBall:
    """A grid of shape (more points than int64 could key by the indices of
    an edge's two ends: above 3.04e9) that is all zero but for a ball near
    its far corner; a baillarger.blocks.Source."""

    def __init__(self, shape: tuple[int, int, int], lo: tuple[int, int, int]) -> None:
        self._shape = shape
        x = np.indices((16, 16, 16)).astype(float)
        self.lo = np.array(lo)
        self.ball = (np.sqrt(((x - 7.6) ** 2).sum(0)) < 5.3).astype(np.uint8)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype[np.uint8]:
        return np.dtype(np.uint8)

    def read(self, box: tuple[slice, ...]) -> npt.NDArray[np.uint8]:
        out = np.zeros(tuple(s.stop - s.start for s in box), np.uint8)
        src, dst = [], []
        for s, lo, n in zip(box, self.lo, self.ball.shape, strict=True):
            a, b = max(s.start, lo), min(s.stop, lo + n)
            if a >= b:
                return out
            src.append(slice(a - lo, b - lo))
            dst.append(slice(a - s.start, b - s.start))
        out[tuple(dst)] = self.ball[tuple(src)]
        return out


@pytest.mark.parametrize("device", ["numpy", "cpu"])
def test_vertex_keys_on_a_grid_past_3e9_points(device: str) -> None:
    from baillarger import backend
    from baillarger.mesh import edge_key, edge_points

    shape = (2100, 2000, 1024)  # 4.3e9 points
    assert np.prod(shape, dtype=np.float64) > 3.04e9
    src = SparseBall(shape, (2080, 1980, 1004))
    V, T, B = marching_tetrahedra_blocks(
        src, 0.5, 256, np.float32, backend.device(device)
    )
    # the ball alone, meshed in its own small grid and moved into place
    v1, t1, b1 = marching_tetrahedra_blocks(src.ball, 0.5, 8, np.float32)
    big, small = canonical(V, T), canonical(v1 + src.lo.astype(np.float32), t1)
    assert np.array_equal(big[0], small[0]) and np.array_equal(big[1], small[1])
    # keys: unique, decoded to the grid edge each vertex lies on, and equal
    # to the small grid's keys moved into the big grid
    k = B.vertex_key
    assert len(np.unique(k)) == len(k) and k.max() > 2.9e10
    a, b = edge_points(k, shape)
    d = (b - a).astype(float)
    t = np.einsum("ij,ij->i", V - a, d) / np.einsum("ij,ij->i", d, d)
    assert np.allclose(a + t[:, None] * d, V, atol=1e-3)
    a1, b1_ = edge_points(b1.vertex_key, src.ball.shape)
    low = np.ravel_multi_index(tuple((a1 + src.lo).T), shape)
    assert np.array_equal(np.sort(edge_key(low, b1_ - a1)), np.sort(k))
