"""Tests of the Omega_h boundary lock (prototypes/omega_h).

The module ``PyOmega_h`` comes from ``prototypes/omega_h/build.sh``. These
tests are skipped when it cannot be imported. To run them, put the build's
``src`` directory on ``PYTHONPATH``::

    nix develop --command env PYTHONPATH=<WORK_DIR>/build/src \\
        uv run pytest prototypes/scripts/test_omega_h_lock.py

A small curved tetrahedral mesh is built from NumPy. Its boundary stands in
for Σ: every boundary vertex, edge and face gets ``lock = 1``, so the
original vertices and every split point stay. Then the mesh is adapted to a
coarse size, and warped with ``warp_to_limit``. The boundary must keep its
vertices at their exact coordinates (moved only by the prescribed warp), its
triangles may only be subdivided, and the tetrahedra must stay valid.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

oh: Any = pytest.importorskip("PyOmega_h")

DIM = 3
LOCK = "lock"

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int32]


@pytest.fixture(scope="module")
def lib() -> Any:
    return oh.OmegaHLibrary()


def grid_tets(n: int) -> tuple[FloatArray, IntArray]:
    """Unit cube, n³ cells, 6 tetrahedra per cell (Kuhn split), curved top.

    The top face bulges (z grows by up to 0.15 at its centre), so the
    boundary triangles of the top face are not coplanar.
    """
    ax = np.linspace(0.0, 1.0, n + 1)
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    bump = 0.15 * np.sin(np.pi * pts[:, 0]) * np.sin(np.pi * pts[:, 1])
    pts[:, 2] += bump * pts[:, 2]

    def vid(i: int, j: int, k: int) -> int:
        return (i * (n + 1) + j) * (n + 1) + k

    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    tets: list[list[int]] = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for perm in perms:
                    c = [0, 0, 0]
                    path = [vid(i, j, k)]
                    for axis in perm:
                        c[axis] += 1
                        path.append(vid(i + c[0], j + c[1], k + c[2]))
                    tets.append(path)
    ev2v = np.asarray(tets, dtype=np.int32)
    # Omega_h wants positive volumes.
    p = pts[ev2v]
    vol = np.einsum(
        "ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]
    )
    flip = vol < 0
    ev2v[flip, 2], ev2v[flip, 3] = ev2v[flip, 3].copy(), ev2v[flip, 2].copy()
    return pts, ev2v


def build_mesh(lib: Any, n: int = 4) -> Any:
    pts, ev2v = grid_tets(n)
    mesh = oh.new_empty_mesh(lib)
    oh.build_from_elems_and_coords(mesh, oh.SIMPLEX, DIM, ev2v.ravel(), pts.ravel())
    oh.classify_by_angles(mesh, np.pi / 4)
    return mesh


def add_lock(mesh: Any, opts: Any) -> None:
    """lock = 1 on every boundary entity below the element dimension."""
    for d in range(DIM + 1):
        class_dim = np.asarray(mesh.get_tag_array(d, "class_dim"))
        lock = ((class_dim < DIM) & (d < DIM)).astype(np.int8)
        mesh.add_tag(d, LOCK, 1, lock)
    opts.lock_tag = LOCK
    opts.set_transfer(LOCK, oh.INHERIT)


def set_size(mesh: Any, h: float) -> None:
    if mesh.has_tag(oh.VERT, "metric"):
        mesh.remove_tag(oh.VERT, "metric")
    mesh.add_tag(oh.VERT, "metric", 1, np.full(mesh.nverts(), 1.0 / h**2))


def coords(mesh: Any) -> FloatArray:
    return np.asarray(mesh.coords(), dtype=np.float64).reshape(-1, DIM)


def boundary_points(mesh: Any) -> FloatArray:
    cd = np.asarray(mesh.get_tag_array(oh.VERT, "class_dim"))
    return coords(mesh)[cd < DIM]


def boundary_triangles(mesh: Any) -> FloatArray:
    """(m, 3, 3): the corner points of each boundary face."""
    cd = np.asarray(mesh.get_tag_array(oh.FACE, "class_dim"))
    fv = np.asarray(mesh.ask_verts_of(oh.FACE)).reshape(-1, 3)
    return coords(mesh)[fv[cd < DIM]]


def kept(points: FloatArray, mesh: Any) -> npt.NDArray[np.bool_]:
    """Which points are still mesh vertices, with exactly equal coordinates."""
    now = {tuple(p) for p in coords(mesh).tolist()}
    return np.array([tuple(p) in now for p in points.tolist()], dtype=bool)


def areas(tris: FloatArray) -> FloatArray:
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    return 0.5 * np.linalg.norm(n, axis=1)


def is_subdivision(before: FloatArray, after: FloatArray, tol: float) -> bool:
    """Each new triangle lies in one old triangle, and the area is the same."""
    o = before[:, 0]
    e1 = before[:, 1] - o
    e2 = before[:, 2] - o
    n = np.cross(e1, e2)
    nn = np.einsum("ij,ij->i", n, n)
    for tri in after:
        inside = np.ones(len(before), dtype=bool)
        for p in tri:
            d = p - o
            off = np.abs(np.einsum("ij,ij->i", d, n)) / np.sqrt(nn)
            a = np.einsum("ij,ij->i", np.cross(d, e2), n) / nn
            b = np.einsum("ij,ij->i", np.cross(e1, d), n) / nn
            inside &= (off <= tol) & (a >= -tol) & (b >= -tol) & (a + b <= 1 + tol)
        if not inside.any():
            return False
    return bool(abs(areas(before).sum() - areas(after).sum()) <= tol)


def assert_valid(mesh: Any) -> None:
    assert np.min(mesh.ask_sizes()) > 0.0
    assert np.min(mesh.ask_qualities()) > 0.0


def make_opts(mesh: Any) -> Any:
    opts = oh.AdaptOpts(mesh)
    opts.verbosity = oh.SILENT
    return opts


def test_lock_survives_coarsening(lib: Any) -> None:
    mesh = build_mesh(lib)
    opts = make_opts(mesh)
    add_lock(mesh, opts)
    points = boundary_points(mesh)
    tris = boundary_triangles(mesh)
    nverts = mesh.nverts()
    set_size(mesh, 0.6)
    oh.adapt(mesh, opts)
    assert mesh.nverts() < nverts  # the interior was coarsened
    assert kept(points, mesh).all()
    assert is_subdivision(tris, boundary_triangles(mesh), 1e-12)
    assert_valid(mesh)


def test_control_without_lock_removes_boundary_vertices(lib: Any) -> None:
    mesh = build_mesh(lib)
    opts = make_opts(mesh)
    points = boundary_points(mesh)
    tris = boundary_triangles(mesh)
    set_size(mesh, 0.6)
    oh.adapt(mesh, opts)
    assert not kept(points, mesh).all()
    # The boundary is no longer a subdivision of the input boundary.
    assert not is_subdivision(tris, boundary_triangles(mesh), 1e-12)
    assert_valid(mesh)


def test_split_points_are_kept(lib: Any) -> None:
    """Refine, then coarsen: every boundary vertex made by a split stays."""
    mesh = build_mesh(lib)
    opts = make_opts(mesh)
    add_lock(mesh, opts)
    tris0 = boundary_triangles(mesh)
    set_size(mesh, 0.12)
    oh.adapt(mesh, opts)
    points1 = boundary_points(mesh)
    tris1 = boundary_triangles(mesh)
    assert len(tris1) > len(tris0)
    assert is_subdivision(tris0, tris1, 1e-12)
    set_size(mesh, 0.6)
    oh.adapt(mesh, opts)
    assert kept(points1, mesh).all()
    assert is_subdivision(tris1, boundary_triangles(mesh), 1e-12)
    assert_valid(mesh)


def test_warp_moves_locked_vertices_only_by_the_warp(lib: Any) -> None:
    """Warp the boundary outward (Σ moves), adapt between warp steps."""
    mesh = build_mesh(lib)
    opts = make_opts(mesh)
    add_lock(mesh, opts)
    points = boundary_points(mesh)
    tris = boundary_triangles(mesh)
    centre = np.array([0.5, 0.5, 0.5])
    scale = 0.3  # an affine warp keeps flat triangles flat

    def warp_of(p: FloatArray) -> FloatArray:
        return scale * (p - centre)

    set_size(mesh, 0.35)
    mesh.add_tag(oh.VERT, "warp", DIM, warp_of(coords(mesh)).ravel())
    nsteps = 0
    while oh.warp_to_limit(mesh, opts):
        oh.adapt(mesh, opts)
        nsteps += 1
        assert nsteps < 50
    assert not mesh.has_tag(oh.VERT, "warp")
    moved = points + warp_of(points)
    now = coords(mesh)
    # Each moved point is a vertex, to the round-off of the stepped warp.
    dist = np.linalg.norm(now[None, :, :] - moved[:, None, :], axis=2).min(axis=1)
    assert dist.max() < 1e-12
    moved_tris = tris + scale * (tris - centre)
    assert is_subdivision(moved_tris, boundary_triangles(mesh), 1e-10)
    assert_valid(mesh)


def test_missing_transfer_is_an_error(lib: Any) -> None:
    mesh = build_mesh(lib, n=2)
    opts = make_opts(mesh)
    for d in range(DIM + 1):
        mesh.add_tag(d, LOCK, 1, np.zeros(mesh.nents(d), dtype=np.int8))
    opts.lock_tag = LOCK  # no INHERIT transfer set
    set_size(mesh, 0.6)
    with pytest.raises(RuntimeError, match="INHERIT"):
        oh.adapt(mesh, opts)
