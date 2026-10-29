"""Smoothing a triangle mesh along itself: tangential relaxation, Gaussian
smoothing without shrinkage, and smoothed vertex normals; and the mesh's
edges, boundary vertices and connected pieces.

Every function keeps the dtype of the vertices (float32 or float64). All
smoothing uses the umbrella operator (each vertex's neighbours' centroid
minus the vertex) and keeps boundary vertices (box-face cuts) fixed.

- relax: tangential relaxation. Vertices move toward their neighbours'
  centroid within the tangent plane, which evens the triangles without moving
  the surface along its normal. The marching-tetrahedra triangles of a binary
  label are already about half a voxel across, so this stands in for
  remeshing; no edge splits or collapses.
- twice: Gaussian smoothing along the mesh with its shrinkage removed by
  twicing (Tukey): x = 2 G v - G G v. G is n umbrella steps of lam
  (gauss_steps sets n from sigma), and its eigenvalues g lie in [0, 1] for
  lam <= 1/2, so the response 1 - (1 - g)^2 lies in [0, 1]: nothing is
  amplified, and the shrinkage of the low frequencies is second order.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse.csgraph import connected_components

type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]
type Bool = npt.NDArray[np.bool_]
type Faces = npt.NDArray[np.integer[Any]]


def edges(f: Faces) -> Int:
    """The undirected edges (k, 2), each once, smaller index first, sorted."""
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1)
    n = int(e.max()) + 1 if len(e) else 1
    # one int64 key per edge: much faster than np.unique(axis=0), same order
    key = np.unique(e[:, 0].astype(np.int64) * n + e[:, 1])
    out: Int = np.stack([key // n, key % n], 1)
    return out


def boundary_vertices(f: Faces, nv: int) -> Bool:
    """The vertices on an edge with one face."""
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1)
    key, cnt = np.unique(e[:, 0].astype(np.int64) * nv + e[:, 1], return_counts=True)
    b = np.zeros(nv, bool)
    once = key[cnt == 1]
    b[once // nv] = True
    b[once % nv] = True
    return b


def vertex_normals(v: Float, f: Faces) -> Float:
    """Area-weighted unit vertex normals, on the side the faces' orientation
    gives."""
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    out: Float = vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)
    return out


def umbrella(v: Float, f: Faces) -> tuple[sparse.csr_matrix, Float]:
    """The vertex adjacency matrix and each vertex's degree (at least 1), in
    the dtype of v (float32 or float64), so that smoothing keeps it."""
    E = edges(f)
    nv = len(v)
    dt = v.dtype if np.issubdtype(v.dtype, np.floating) else np.dtype(np.float64)
    C = sparse.coo_matrix((np.ones(len(E), dt), (E[:, 0], E[:, 1])), shape=(nv, nv))
    A = (C + C.T).tocsr()
    deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1).astype(dt)
    return A, deg


def relax(v: Float, f: Faces, iters: int = 5, step: float = 0.5) -> Float:
    """Tangential relaxation toward the neighbours' centroid."""
    A, deg = umbrella(v, f)
    fixed = boundary_vertices(f, len(v))
    x = v.copy()
    for _ in range(iters):
        n = vertex_normals(x, f)
        d = (A @ x) / deg[:, None] - x
        d -= np.sum(d * n, 1, keepdims=True) * n
        d[fixed] = 0
        x = x + step * d
    return x


def gauss_steps(v: Float, f: Faces, sigma: float, lam: float = 0.5) -> int:
    """Umbrella steps of size lam for a Gaussian of sigma (the units of v): on
    a regular mesh of edge e, one step multiplies a plane wave of wavenumber w
    by about 1 - lam w^2 e^2 / 4, so n steps give exp(-w^2 sigma^2 / 2) with
    sigma^2 = n lam <e^2> / 2."""
    E = edges(f)
    e2 = float(np.mean(np.sum((v[E[:, 0]] - v[E[:, 1]]) ** 2, 1)))
    return max(1, int(round(2 * sigma**2 / (lam * e2))))


def twice(v: Float, f: Faces, sigma: float, lam: float = 0.5) -> Float:
    """Gaussian smoothing of sigma along the mesh, shrinkage removed by
    twicing: 2 G v - G G v."""
    A, deg = umbrella(v, f)
    fixed = boundary_vertices(f, len(v))
    n = gauss_steps(v, f, sigma, lam)

    def G(x: Float) -> Float:
        for _ in range(n):
            d = (A @ x) / deg[:, None] - x
            d[fixed] = 0
            x = x + lam * d
        return x

    g = G(v.copy())
    out: Float = 2 * g - G(g)
    return out


def smooth_normals(v: Float, f: Faces, iters: int = 10) -> Float:
    """Area-weighted vertex normals, then iters steps of Laplacian smoothing
    along the edges (as baillarger.frame.white_mesh)."""
    n = vertex_normals(v, f)
    A, deg = umbrella(v, f)
    for _ in range(iters):
        n = 0.5 * n + 0.5 * (A @ n) / deg[:, None]
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return n


def pieces(f: Faces, nv: int) -> tuple[Int, Int]:
    """Connected pieces: a label per vertex (-1 for vertices in no face) and
    per face, numbered in order of their lowest vertex."""
    E = edges(f)
    G = sparse.coo_matrix(
        (np.ones(len(E), np.int8), (E[:, 0], E[:, 1])), shape=(nv, nv)
    )
    _, lab = connected_components(G, directed=False)
    used = np.zeros(nv, bool)
    used[f.ravel()] = True
    _, inv = np.unique(np.where(used, lab, -1), return_inverse=True)
    vlab: Int = (inv - (0 if used.all() else 1)).astype(np.int64)
    return vlab, vlab[f[:, 0]]
