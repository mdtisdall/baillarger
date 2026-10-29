"""Triangle meshes of level sets: marching tetrahedra, the interpolant it
contours, mesh topology, and a VTK writer.

Marching tetrahedra works on one fixed tetrahedralization of the voxel grid.
Each cube of 8 grid points is split into 6 tetrahedra around its main diagonal
(the Kuhn / Freudenthal split, the same in every cube, so neighbouring cubes
share faces). On a tetrahedron the field is linear, so the level set is one
triangle or one quad, with vertices on the edges whose ends straddle the
level. The surface is exactly the level set of one piecewise-linear
interpolant of the field (pl_sample): level sets at different levels cannot
cross, and none crosses itself. Marching cubes does not have this property:
its triangles depend on choices made per cube and per level.

Vertices are shared between triangles through their grid edge, so the mesh is
welded. Triangles are oriented so that their normals point toward increasing
field values.
"""

import itertools
import os
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse.csgraph import connected_components

type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]


def _kuhn_tets() -> npt.NDArray[np.int_]:
    # corner offsets of each tetrahedron: (0,0,0), +e_a, +e_a+e_b, (1,1,1) for
    # each order of the axes
    tets = []
    for perm in itertools.permutations(range(3)):
        c = [np.zeros(3, int)]
        for ax in perm:
            nxt = c[-1].copy()
            nxt[ax] = 1
            c.append(nxt)
        tets.append(np.array(c))
    return np.array(tets)  # (6, 4, 3)


_TETS = _kuhn_tets()


#: cells per side of a block in marching_tetrahedra
BLOCK = 64


def marching_tetrahedra(
    F: npt.ArrayLike, level: float, block: int = BLOCK
) -> tuple[Float, Int]:
    """Level set {F = level} of the piecewise-linear interpolant of F (3D
    array) on the Kuhn tetrahedralization. Returns vertices (voxel coordinates)
    and faces.

    The cells are taken in blocks of block cells a side, so that the work
    arrays scale with one block's surface, not the whole grid's; a block
    whose grid points are all on one side of the level is skipped. A vertex
    is keyed by its grid edge in the whole grid and computed from the edge's
    ends in a fixed order, so the blocks join exactly: the result does not
    depend on block. Vertices are sorted by their edge key."""
    Fa = np.asarray(F)
    shape = Fa.shape
    vkeys, verts, faces = [], [], []
    n_verts = 0
    for o in itertools.product(*(range(0, max(n - 1, 0), block) for n in shape)):
        sl = tuple(
            slice(a, min(a + block + 1, n)) for a, n in zip(o, shape, strict=True)
        )
        G = np.asarray(Fa[sl], np.float64) - level
        G[G == 0] = 1e-12  # no grid point exactly on the level
        pos = G > 0
        if pos.all() or not pos.any():
            continue
        k, v, f = _march_block(G, pos, np.array(o), shape)
        vkeys.append(k)
        verts.append(v)
        faces.append(f + n_verts)
        n_verts += len(k)
    if not faces:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    keys = np.concatenate(vkeys)
    _, first, inv = np.unique(keys, return_index=True, return_inverse=True)
    out_v: Float = np.concatenate(verts)[first]
    out_f: Int = inv[np.concatenate(faces)].astype(np.int64)
    return out_v, out_f


def _march_block(
    G: Float,
    pos: npt.NDArray[np.bool_],
    origin: npt.NDArray[np.int_],
    shape: tuple[int, ...],
) -> tuple[npt.NDArray[np.int64], Float, Int]:
    """One block of marching_tetrahedra: G is F - level on the block's grid
    points (one more than its cells a side), pos = G > 0, origin the block's
    first grid point. Returns the block's vertex keys (grid-edge keys of the
    whole grid, unique), vertices (whole-grid voxel coordinates) and faces
    (indices into them)."""
    nx, ny, nz = G.shape
    _, NY, NZ = shape
    size = int(np.prod(shape))
    # cells whose 8 corners do not all have the same sign
    c = [
        pos[i : nx - 1 + i, j : ny - 1 + j, k : nz - 1 + k]
        for i in (0, 1)
        for j in (0, 1)
        for k in (0, 1)
    ]
    anyp = np.logical_or.reduce(c)
    allp = np.logical_and.reduce(c)
    cells = np.argwhere(anyp & ~allp)  # (M, 3), block coordinates
    corners = cells[:, None, None, :] + _TETS[None]  # (M, 6, 4, 3)
    corners = corners.reshape(-1, 4, 3)
    val = G[corners[..., 0], corners[..., 1], corners[..., 2]]  # (T, 4)
    corners = corners + origin  # whole-grid coordinates
    lin = (corners[..., 0].astype(np.int64) * NY + corners[..., 1]) * NZ + corners[
        ..., 2
    ]
    sgn = val > 0
    npos = sgn.sum(1)
    keep = (npos > 0) & (npos < 4)
    corners, lin, val, sgn, npos = (
        corners[keep],
        lin[keep],
        val[keep],
        sgn[keep],
        npos[keep],
    )
    order = np.argsort(~sgn, axis=1, kind="stable")  # positive corners first
    r = np.arange(len(lin))[:, None]
    o_lin, o_val, o_pos = lin[r, order], val[r, order], corners[r, order]

    def edge(
        a: int, b: int, rows: npt.NDArray[np.intp]
    ) -> tuple[npt.NDArray[np.int64], Float]:
        """Edge keys and points between corners a and b (indices into the
        sorted corners), each point computed from the edge's lower-key end,
        so that every tetrahedron and block gives the same bits."""
        la, lb = o_lin[rows, a], o_lin[rows, b]
        sw = la > lb
        va = np.where(sw, o_val[rows, b], o_val[rows, a])
        vb = np.where(sw, o_val[rows, a], o_val[rows, b])
        pa = np.where(sw[:, None], o_pos[rows, b], o_pos[rows, a]).astype(np.float64)
        pb = np.where(sw[:, None], o_pos[rows, a], o_pos[rows, b]).astype(np.float64)
        key = np.minimum(la, lb) * size + np.maximum(la, lb)
        t = (va / (va - vb))[:, None]
        return key, pa + t * (pb - pa)

    tri_keys, tri_pts, tri_rows = [], [], []
    for n_pos, pairs_list in (
        (1, [[(0, 1), (0, 2), (0, 3)]]),
        (3, [[(3, 0), (3, 1), (3, 2)]]),
        (2, [[(0, 2), (0, 3), (1, 3)], [(0, 2), (1, 3), (1, 2)]]),
    ):
        rows = np.where(npos == n_pos)[0]
        if not len(rows):
            continue
        for pairs in pairs_list:
            ks, ps = zip(*[edge(a, b, rows) for a, b in pairs], strict=True)
            tri_keys.append(np.stack(ks, 1))
            tri_pts.append(np.stack(ps, 1))
            tri_rows.append(rows)
    keys = np.concatenate(tri_keys)  # (nT, 3)
    pts = np.concatenate(tri_pts)  # (nT, 3, 3)
    rows = np.concatenate(tri_rows)
    # orientation: normals toward increasing F (the gradient of the
    # tetrahedron's linear field)
    P = corners[rows].astype(np.float64)
    V = val[rows]
    M = P[:, 1:] - P[:, :1]
    g = np.linalg.solve(M, (V[:, 1:] - V[:, :1])[..., None])[..., 0]
    nrm = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
    flip = np.sum(nrm * g, 1) < 0
    keys[flip] = keys[flip][:, [0, 2, 1]]
    pts[flip] = pts[flip][:, [0, 2, 1]]
    # drop triangles that collapsed to a line (two vertices on the same edge
    # cannot happen; zero area only when the level passes within rounding of a
    # grid point)
    ok = np.linalg.norm(nrm, axis=1) > 0
    keys, pts = keys[ok], pts[ok]
    uk, first, inv = np.unique(keys.ravel(), return_index=True, return_inverse=True)
    verts: Float = pts.reshape(-1, 3)[first]
    return uk, verts, inv.reshape(-1, 3).astype(np.int64)


def pl_sample(F: npt.ArrayLike, pts: npt.ArrayLike) -> Float:
    """The piecewise-linear interpolant that marching_tetrahedra contours, at
    points (N, 3) in voxel coordinates.

    In each cube, with the fractional coordinates sorted u_a >= u_b >= u_c,
    f = (1 - u_a) f000 + (u_a - u_b) f(e_a) + (u_b - u_c) f(e_a + e_b)
    + u_c f111. Points outside the grid are clamped to it.
    """
    Fa = np.asarray(F, np.float64)
    hi = np.array(Fa.shape) - 1
    p = np.clip(np.asarray(pts, np.float64), 0, hi - 1e-9)
    base = np.floor(p).astype(int)
    u = p - base
    order = np.argsort(-u, axis=1)
    us = np.take_along_axis(u, order, 1)
    r = np.arange(len(p))
    c1 = base.copy()
    c1[r, order[:, 0]] += 1
    c2 = c1.copy()
    c2[r, order[:, 1]] += 1
    c3 = base + 1

    def val(c: npt.NDArray[np.int_]) -> Float:
        return Fa[c[:, 0], c[:, 1], c[:, 2]]

    out: Float = (
        (1 - us[:, 0]) * val(base)
        + (us[:, 0] - us[:, 1]) * val(c1)
        + (us[:, 1] - us[:, 2]) * val(c2)
        + us[:, 2] * val(c3)
    )
    return out


def topology(faces: npt.NDArray[np.integer[Any]], nv: int) -> dict[str, Any]:
    """Components, boundary loops, Euler characteristic and genus of a triangle
    mesh with nv vertices, and the number of edges with more than two faces
    (nonmanifold_edges)."""
    e = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), 1
    )
    eu, cnt = np.unique(e, axis=0, return_counts=True)
    used = np.unique(faces)
    A = sparse.coo_matrix((np.ones(len(eu)), (eu[:, 0], eu[:, 1])), shape=(nv, nv))
    _, lab = connected_components(A, directed=False)
    bd = eu[cnt == 1]
    B = sparse.coo_matrix((np.ones(len(bd)), (bd[:, 0], bd[:, 1])), shape=(nv, nv))
    _, blab = connected_components(B, directed=False)
    loops = len(np.unique(blab[np.unique(bd)])) if len(bd) else 0
    comps = len(np.unique(lab[used]))
    chi = len(used) - len(eu) + len(faces)
    return dict(
        components=comps,
        boundary_loops=loops,
        chi=int(chi),
        genus=(2 * comps - chi - loops) / 2,
        nonmanifold_edges=int((cnt > 2).sum()),
    )


def write_vtk(
    path: str,
    aff: npt.ArrayLike,
    pts_vox: npt.ArrayLike,
    tri: npt.ArrayLike,
    arrays: dict[str, npt.ArrayLike],
) -> None:
    """Write a triangle mesh as a legacy ASCII VTK polydata file.

    pts_vox are voxel coordinates; aff is the volume's NIfTI affine (4 x 4).
    The points are written in NIfTI RAS world coordinates, which is what
    ITK-SNAP 4.4 lines up with the volume. arrays are point data (one value per
    point, NaN written as -1).
    """
    a = np.asarray(aff)
    p = np.asarray(pts_vox)
    # nibabel.affines.apply_affine
    ras = (p.reshape(-1, p.shape[-1]) @ a[:-1, :-1].T + a[:-1, -1][None, :]).reshape(
        p.shape
    )
    t = np.asarray(tri)
    with open(path, "w") as f:
        f.write(
            "# vtk DataFile Version 3.0\n"
            f"{os.path.basename(path)}\nASCII\nDATASET POLYDATA\n"
        )
        f.write(f"POINTS {len(ras)} float\n")
        np.savetxt(f, ras, fmt="%.4f")
        f.write(f"POLYGONS {len(t)} {4 * len(t)}\n")
        np.savetxt(f, np.c_[np.full(len(t), 3), t], fmt="%d")
        f.write(f"POINT_DATA {len(ras)}\n")
        for name, arr in arrays.items():
            f.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
            np.savetxt(f, np.nan_to_num(np.asarray(arr), nan=-1), fmt="%.4f")
