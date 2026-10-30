"""PyTorch versions of the white mesh's steps, for any baillarger.backend
device: one block of marching tetrahedra (baillarger.mesh), and the
smoothing (baillarger.mesh_smoothing: relax, twice, smooth_normals).

They follow the numpy reference step for step, in float32. The umbrella
operator and the vertex normals sum over padded tables (each vertex's
neighbours, each vertex's faces) by gather, in chunks of vertices, not by
atomic scatter or sparse products: the same on CPU, CUDA and MPS, and the
same result on every run. Results differ from numpy's only in the order of
the sums (float32 rounding); tests/test_mesh_torch.py states the
tolerances. The tables are built once with numpy.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from baillarger import mesh_smoothing as S
from baillarger.mesh import _CASES, _FLIP, _TETS

type Float = npt.NDArray[np.float32]
#: float32 or float64 input
type AnyFloat = npt.NDArray[np.floating[Any]]
type Int = npt.NDArray[np.int64]
type Faces = npt.NDArray[np.integer[Any]]

#: vertices per chunk of a gather (bounds the (chunk, K, 3) temporaries)
CHUNK = 1 << 20


# ---------------------------------------------------------------- marching


def march_block(
    G: torch.Tensor, origin: npt.NDArray[np.int_], shape: tuple[int, ...]
) -> tuple[Int, Float, Int]:
    """baillarger.mesh._march_block on G's device: G is F - level (float32)
    on the block's grid points, origin its first grid point. Returns numpy
    arrays: the block's vertex keys (unique, sorted), vertices and faces."""
    dev = G.device
    nx, ny, nz = G.shape
    _, NY, NZ = shape
    pos = G > 0
    c = [
        pos[i : nx - 1 + i, j : ny - 1 + j, k : nz - 1 + k]
        for i in (0, 1)
        for j in (0, 1)
        for k in (0, 1)
    ]
    anyp, allp = c[0].clone(), c[0].clone()
    for x in c[1:]:
        anyp |= x
        allp &= x
    cells = torch.nonzero(anyp & ~allp)  # (M, 3)
    tets = torch.as_tensor(_TETS, dtype=torch.int64, device=dev)
    corners = (cells[:, None, None, :] + tets[None]).reshape(-1, 4, 3)
    ty = torch.arange(6, device=dev).repeat(len(cells))
    val = G[corners[..., 0], corners[..., 1], corners[..., 2]]  # (T, 4)
    corners = corners + torch.as_tensor(origin, dtype=torch.int64, device=dev)
    lin = (corners[..., 0] * NY + corners[..., 1]) * NZ + corners[..., 2]
    sgn = val > 0
    npos = sgn.sum(1)
    keep = (npos > 0) & (npos < 4)
    corners, lin, val, sgn, npos, ty = (
        corners[keep],
        lin[keep],
        val[keep],
        sgn[keep],
        npos[keep],
        ty[keep],
    )
    bits = torch.tensor([1, 2, 4, 8], device=dev)
    pat = (sgn.to(torch.int64) * bits).sum(1)
    # positive corners first, stably (as np.argsort(~sgn, kind="stable"))
    order = torch.argsort((~sgn).to(torch.int64), dim=1, stable=True)
    o_lin = torch.gather(lin, 1, order)
    o_val = torch.gather(val, 1, order)
    o_pos = torch.gather(corners, 1, order[..., None].expand(-1, -1, 3))
    flip_t = torch.as_tensor(_FLIP, device=dev)

    def edge(a: int, b: int, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        la, lb = o_lin[rows, a], o_lin[rows, b]
        sw = la > lb
        va = torch.where(sw, o_val[rows, b], o_val[rows, a])
        vb = torch.where(sw, o_val[rows, a], o_val[rows, b])
        ia = torch.where(sw[:, None], o_pos[rows, b], o_pos[rows, a])
        ib = torch.where(sw[:, None], o_pos[rows, a], o_pos[rows, b])
        st = ib - ia  # the edge's step, in {0, 1}^3 (mesh.edge_key)
        d = 4 * st[:, 0] + 2 * st[:, 1] + st[:, 2] - 1
        key = torch.minimum(la, lb) * 7 + d
        pa, pb = ia.to(G.dtype), ib.to(G.dtype)
        t = (va / (va - vb))[:, None]
        return key, pa + t * (pb - pa)

    tri_keys, tri_pts, tri_flip = [], [], []
    for n_pos, pairs_list in _CASES:
        rows = torch.nonzero(npos == n_pos).squeeze(1)
        if not len(rows):
            continue
        for k, pairs in enumerate(pairs_list):
            ks, ps = zip(*[edge(a, b, rows) for a, b in pairs], strict=True)
            tri_keys.append(torch.stack(ks, 1))
            tri_pts.append(torch.stack(ps, 1))
            tri_flip.append(flip_t[ty[rows], pat[rows], k])
    if not tri_keys:
        z = np.zeros(0, np.int64)
        return z, np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64)
    keys = torch.cat(tri_keys)
    pts = torch.cat(tri_pts)
    flip = torch.cat(tri_flip)
    keys[flip] = keys[flip][:, [0, 2, 1]]
    pts[flip] = pts[flip][:, [0, 2, 1]]
    uk, inv = torch.unique(keys.reshape(-1), return_inverse=True)
    # every occurrence of a key has the same bits (each crossing is computed
    # from its edge's lower-key end), so any one may land
    verts = torch.empty((len(uk), 3), dtype=G.dtype, device=dev)
    verts[inv] = pts.reshape(-1, 3)
    return (
        uk.cpu().numpy().astype(np.int64),
        verts.cpu().numpy(),
        inv.reshape(-1, 3).cpu().numpy().astype(np.int64),
    )


# ---------------------------------------------------------------- smoothing


def _table(src: Int, dst: Int, n: int, pad: int) -> Int:
    """(n, K): for each src row, its dst values, padded with pad."""
    o = np.argsort(src, kind="stable")
    src, dst = src[o], dst[o]
    cnt = np.bincount(src, minlength=n)
    start = np.r_[0, np.cumsum(cnt)[:-1]]
    rank = np.arange(len(src)) - start[src]
    out = np.full((n, max(int(cnt.max()) if n else 0, 1)), pad, np.int64)
    out[src, rank] = dst
    return out


class MeshOps:
    """The tables of one mesh on a device, and the smoothing steps on it."""

    def __init__(self, f: Faces, nv: int, dev: torch.device) -> None:
        self.dev = dev
        self.nv = nv
        fa = np.asarray(f, np.int64)
        E = S.edges(fa)
        src, dst = np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]]
        self.nbr = torch.as_tensor(_table(src, dst, nv, nv), device=dev)
        deg = np.maximum(np.bincount(src, minlength=nv), 1).astype(np.float32)
        self.deg = torch.as_tensor(deg, device=dev)[:, None]
        m = len(fa)
        fi = np.repeat(np.arange(m), 3)
        self.vf = torch.as_tensor(_table(fa.ravel(), fi, nv, m), device=dev)
        self.f = torch.as_tensor(fa, device=dev)
        fixed = S.boundary_vertices(fa, nv)
        self.fixed = torch.as_tensor(fixed, device=dev)
        self.any_fixed = bool(fixed.any())

    def _gather_sum(self, x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        """Per row of table, the sum of the rows of x it names (the pad
        index names a zero row)."""
        xp = torch.cat(
            [x, torch.zeros((1, x.shape[1]), dtype=x.dtype, device=self.dev)]
        )
        out = torch.empty((len(table), x.shape[1]), dtype=x.dtype, device=self.dev)
        for i in range(0, len(table), CHUNK):
            out[i : i + CHUNK] = xp[table[i : i + CHUNK]].sum(1)
        return out

    def mean_neighbours(self, x: torch.Tensor) -> torch.Tensor:
        return self._gather_sum(x, self.nbr) / self.deg

    def vertex_normals(self, x: torch.Tensor) -> torch.Tensor:
        """mesh_smoothing.vertex_normals."""
        p = x[self.f]
        fn = torch.linalg.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        vn = self._gather_sum(fn, self.vf)
        out: torch.Tensor = vn / (
            torch.linalg.vector_norm(vn, dim=1, keepdim=True) + 1e-12
        )
        return out

    def relax(self, x: torch.Tensor, iters: int = 5, step: float = 0.5) -> torch.Tensor:
        """mesh_smoothing.relax."""
        for _ in range(iters):
            n = self.vertex_normals(x)
            d = self.mean_neighbours(x) - x
            d = d - (d * n).sum(1, keepdim=True) * n
            if self.any_fixed:
                d[self.fixed] = 0
            x = x + step * d
        return x

    def twice(self, x: torch.Tensor, steps: int, lam: float = 0.5) -> torch.Tensor:
        """mesh_smoothing.twice, with its number of steps given
        (mesh_smoothing.gauss_steps)."""

        def G(y: torch.Tensor) -> torch.Tensor:
            for _ in range(steps):
                d = self.mean_neighbours(y) - y
                if self.any_fixed:
                    d[self.fixed] = 0
                y = y + lam * d
            return y

        g = G(x)
        return 2 * g - G(g)

    def smooth_normals(self, x: torch.Tensor, iters: int = 10) -> torch.Tensor:
        """mesh_smoothing.smooth_normals."""
        n = self.vertex_normals(x)
        for _ in range(iters):
            n = 0.5 * n + 0.5 * self.mean_neighbours(n)
            n = n / (torch.linalg.vector_norm(n, dim=1, keepdim=True) + 1e-12)
        return n

    def smooth(self, v0: AnyFloat, sigma: float) -> Float:
        """relax, then twice with sigma (as baillarger.white smooths the
        white mesh): float32 numpy in and out."""
        x = self.relax(torch.as_tensor(np.asarray(v0, np.float32), device=self.dev))
        steps = S.gauss_steps(x.cpu().numpy(), self.f.cpu().numpy(), sigma)
        out: Float = self.twice(x, steps).cpu().numpy()
        return out

    def normals(self, v: AnyFloat) -> Float:
        """smooth_normals of v: float32 numpy in and out."""
        x = torch.as_tensor(np.asarray(v, np.float32), device=self.dev)
        out: Float = self.smooth_normals(x).cpu().numpy()
        return out
