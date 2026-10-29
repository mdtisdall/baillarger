"""Tests for baillarger.mesh: marching tetrahedra, pl_sample, topology and the
VTK writer."""

from pathlib import Path

import numpy as np
import numpy.typing as npt

from baillarger import mesh_intersect as mi
from baillarger.mesh import marching_tetrahedra, pl_sample, topology, write_vtk

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


def canonical_faces(T: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Each triangle rotated to start at its smallest index (orientation
    kept), rows sorted."""
    r = np.argmin(T, axis=1)
    idx = (r[:, None] + np.arange(3)[None]) % 3
    R = np.take_along_axis(T, idx, 1)
    out: npt.NDArray[np.int64] = R[np.lexsort(R.T[::-1])]
    return out


def test_blocks_give_the_one_shot_mesh_exactly() -> None:
    rng = np.random.default_rng(2)
    F, _ = sphere_field(30, (14.3, 15.1, 14.7))
    rough = F + 0.8 * rng.standard_normal(F.shape)
    x = np.arange(36, dtype=float)
    X, Y, Z = np.meshgrid(x, x, x[:20], indexing="ij")
    torus = np.hypot(np.hypot(X - 17.7, Y - 18.2) - 9, Z - 9.6)
    binary = (rough < 9.0).astype(np.uint8)  # integer input, points on the level
    for G, lv in ((rough, 9.0), (torus, 4.0), (binary, 0.5), (np.round(F), 9.0)):
        V0, T0 = marching_tetrahedra(G, lv, block=1000)
        assert len(T0)
        for b in (1, 3, 7, 16):
            V, T = marching_tetrahedra(G, lv, block=b)
            assert np.array_equal(V, V0)
            assert np.array_equal(canonical_faces(T), canonical_faces(T0))


def test_blocks_away_from_the_level_are_skipped() -> None:
    F, _ = sphere_field(40, (8.3, 8.1, 8.7))  # surface in one corner
    V0, T0 = marching_tetrahedra(F, 5.0, block=1000)
    V, T = marching_tetrahedra(F, 5.0, block=8)
    assert np.array_equal(V, V0)
    assert np.array_equal(canonical_faces(T), canonical_faces(T0))
    assert topology(T, len(V))["genus"] == 0 and V.max() < 16
