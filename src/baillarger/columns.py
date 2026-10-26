"""The inputs of the model: which columns take part, the column graph, and one
table of the voxels with their column, relative depth and echo values.

Model columns. A column of the frame (baillarger.frame) takes part in steps
1-6 when it has cortex voxels on streamlines that exit the cortex into fluid,
is not cut by missing tissue (column_cut), and not more than fused_frac of its
streamlines end at a ridge inside the cortex. A streamline that meets the
facing bank's streamlines inside the cortex is in fused cortex, where no pial
surface is known until the joint surfaces (step 7) set one. A streamline that
meets the facing bank in the fluid of an open sulcus ends past the pial
surface and does not count.

Column graph. Two model columns are neighbours when their patches touch on
the white mesh (an edge of the mesh joins them) and their cortex voxels share
faces. The weight is the number of shared faces, each counted by its depth d
(the mean of the two voxels' depths, mm) as max(0, 1 - d / base_mm). Columns
along the cortical sheet touch at the base, next to the white surface;
columns of facing banks (in fused cortex, or across a sulcus) touch only near
the top, where their streamlines meet, so their faces count little or not at
all. On V1V2, with base_mm = 1, links between columns more than 3 mm apart
along the mesh keep 0.1 % of the weight, against 2.5 % with plain face counts.
The depth comes from the fixed frame, so the graph does not change in the
outer loop. links="faces" gives the plain shared-face graph, for comparison.

Voxel table (ColumnData). Relative depth is per streamline: r = d / T_p,
where T_p is the depth at which the voxel's own streamline exits the cortex
into fluid (frame.path_exit): 0 at the white surface, 1 at the pial surface,
whatever the thickness. A streamline that never exits into fluid ends inside
the cortex (at a ridge in fused cortex, the crop, an excision or d_max), so
its voxels have no pial depth to scale by and stay out of the table. The
table holds the cortex voxels of the model columns on exiting streamlines,
and the WM voxels within wm_mm of the white surface, each with the column and
the streamline of its nearest white vertex, at r = D / T_p < 0 (D is the
signed distance of the frame, negative in the WM). With
depth="equivolume", r is the equivolume depth instead (Waehnert et al. 2014):
the flow tube of a streamline is that of the surfaces offset from the white
surface, with the white mesh's principal curvatures at its vertex smoothed
over 1 mm, A(s) = (1 + k1 s)(1 + k2 s) (frame.path_volume); its volume from
the white surface is V(s) = integral of A ds, and r = V(d) / V(T_p) in the
cortex, r = D / V(T_p) in the WM. On flat cortex the two depths are the
same; over a gyral crown (k1 + k2 > 0) the equivolume r of a voxel is lower,
in a sulcal fundus higher. The column thickness T_c is the median of T_p
over the column's streamlines. The noise of each column is the
pooled variance, per echo, inside the cells of n_layers equal bins of r in
the cortex (the fixed-K baseline's geometric layers): the model's noise is
diagonal over the echoes.

Edge zones. The cortex voxels within edge_vox voxels (default 1, 0.16 mm on
V1V2) of the white or the pial surface along their streamline (d < edge or
T_p − d < edge) are partial volumes of cortex with WM or fluid. They change
most along r (on V1V2 the outer 0.1 of r carried 9 times the mean intensity
step of the inner 0.1s, the inner 0.1 twice), so the band fit gave them
bands of their own: 30 % of the region boundaries sat below r = 0.1 or above
0.9, and the edge voxels made about half of the data term. They are left out
of the model (and of the noise) and kept in edge_index for the outputs, which
give them the band next to them. With them out (and λ_z = 5), on V1V2: no
region boundary below r = 0.1 or above 0.9 (levels 256, 32, 16), mean K_R
4.3 instead of 5.0, and the stria (a dark middle band in echo 2) found in
2,908 columns instead of 1,642 at level 256, 1,113 instead of 107 at level
32; at level 16 still in none with equidistant r.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from baillarger.frame import END_RIDGE, Frame, path_exit, path_volume, volume_at

type Bool = npt.NDArray[np.bool_]
type F32 = npt.NDArray[np.float32]
type F64 = npt.NDArray[np.float64]
type I64 = npt.NDArray[np.int64]


def fused_fraction(frame: Frame, cortex: Bool, exit_depth: F32 | None = None) -> F64:
    """For each column, the fraction of its streamlines (those with at least
    one step) that end at a ridge inside the cortex without having exited
    into fluid first (exit_depth NaN, when given; with seams, a streamline can
    exit at a seam and go on to a ridge in the facing bank's cortex); NaN for
    a column with none."""
    n = np.diff(frame.path_start)
    moved = n >= 2
    last = frame.path_points[frame.path_start[1:][moved] - 1]
    vi = np.clip(np.floor(last + 0.5).astype(np.int64), 0, np.array(cortex.shape) - 1)
    in_cortex = np.zeros(len(n), bool)
    in_cortex[moved] = cortex[vi[:, 0], vi[:, 1], vi[:, 2]]
    fused = moved & (frame.path_end == END_RIDGE) & in_cortex
    if exit_depth is not None:
        fused &= ~np.isfinite(exit_depth)
    col = frame.vertex_column
    total = np.bincount(col[moved], minlength=frame.n_columns).astype(np.float64)
    hit = np.bincount(col[fused], minlength=frame.n_columns)
    with np.errstate(invalid="ignore", divide="ignore"):
        out: F64 = np.where(total > 0, hit / total, np.nan)
    return out


def model_columns(
    frame: Frame, cortex: Bool, exit_depth: F32, fused_frac: float = 0.5
) -> tuple[Bool, dict[str, int]]:
    """Which frame columns take part in the model (see the module notes), and
    a count of why the others do not. exit_depth: path_exit of each
    streamline; a column needs cortex voxels on streamlines that exit."""
    has = (
        np.bincount(
            frame.voxel_column[cortex & (frame.voxel_column >= 0)],
            minlength=frame.n_columns,
        )
        > 0
    )
    vp = frame.voxel_path
    good = cortex & (vp >= 0)
    good[good] = np.isfinite(exit_depth[vp[good]])
    exits = np.bincount(frame.voxel_column[good], minlength=frame.n_columns) > 0
    fused = fused_fraction(frame, cortex, exit_depth) > fused_frac
    keep = has & exits & ~frame.column_cut & ~fused
    why = dict(
        columns=frame.n_columns,
        no_cortex=int((~has).sum()),
        cut=int((has & frame.column_cut).sum()),
        fused=int((has & ~frame.column_cut & fused).sum()),
        no_exit=int((has & ~frame.column_cut & ~fused & ~exits).sum()),
        model=int(keep.sum()),
    )
    return keep, why


def column_graph(
    frame: Frame,
    cortex: Bool,
    keep: Bool,
    base_mm: float = 1.0,
    links: Literal["base", "faces"] = "base",
) -> tuple[I64, F64]:
    """Links (L, 2; frame column ids, first < second) and weights between the
    columns in keep (see the module notes). links="faces": every pair of
    columns whose cortex voxels share a face, weighted by the number of
    faces."""
    c = np.where(cortex, frame.voxel_column, -1).astype(np.int64)
    c[c >= 0] = np.where(keep[c[c >= 0]], c[c >= 0], -1)
    d = frame.depth.astype(np.float64)
    A, B, W = [], [], []
    for ax in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[ax] = slice(0, -1)
        hi[ax] = slice(1, None)
        x, y = c[tuple(lo)].ravel(), c[tuple(hi)].ravel()
        m = (x >= 0) & (y >= 0) & (x != y)
        A.append(np.minimum(x[m], y[m]))
        B.append(np.maximum(x[m], y[m]))
        if links == "faces":
            W.append(np.ones(int(m.sum())))
        else:
            depth = 0.5 * (d[tuple(lo)].ravel()[m] + d[tuple(hi)].ravel()[m])
            W.append(np.clip(1.0 - depth / base_mm, 0.0, None))
    n = frame.n_columns
    key = np.concatenate(A) * n + np.concatenate(B)
    uk, inv = np.unique(key, return_inverse=True)
    w = np.bincount(inv, np.concatenate(W), minlength=len(uk))
    edges = np.stack([uk // n, uk % n], 1)
    if links == "base":
        # the two patches must touch on the white mesh
        f = frame.faces
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        ca, cb = frame.vertex_column[e[:, 0]], frame.vertex_column[e[:, 1]]
        mk = np.unique(np.minimum(ca, cb) * n + np.maximum(ca, cb))
        ok = np.isin(uk, mk) & (w > 0)
        edges, w = edges[ok], w[ok]
    return edges.astype(np.int64), w.astype(np.float64)


@dataclass(frozen=True)
class ColumnData:
    """The voxels of the model, as one table (V voxels, E echoes, n columns,
    L links). Columns are numbered 0..n-1; labels[i] is the frame's column id
    of column i."""

    shape: tuple[int, int, int]
    #: flat voxel index in the patch, column, relative depth r, echo values
    #: (V, E); cortex is r's sign: False for the WM extension
    index: I64
    col: I64
    m: F64
    f: F64
    ctx: Bool
    labels: I64
    #: thickness T_c of each column (mm): the median exit depth of its
    #: streamlines
    thickness: F64
    #: column graph: links (L, 2) between columns 0..n-1 and their weights
    edges: I64
    w: F64
    #: noise variance of each column and echo (n, E)
    sigma2: F64
    #: the edge voxels: cortex voxels of the model columns within edge_mm of
    #: the white or the pial surface along their streamline, left out of the
    #: model (partial volume); their flat index, column, and side (-1 white,
    #: +1 pial), for the outputs
    edge_index: I64 = field(default_factory=lambda: np.zeros(0, np.int64))
    edge_col: I64 = field(default_factory=lambda: np.zeros(0, np.int64))
    edge_side: I64 = field(default_factory=lambda: np.zeros(0, np.int64))

    @property
    def n(self) -> int:
        return len(self.labels)

    @property
    def n_echoes(self) -> int:
        return int(self.f.shape[1])

    def image(
        self, values: npt.ArrayLike, fill: Any = 0, dtype: npt.DTypeLike = np.float32
    ) -> npt.NDArray[Any]:
        """A patch image of one value per voxel of the table."""
        out = np.full(int(np.prod(self.shape)), fill, dtype)
        out[self.index] = np.asarray(values)
        return out.reshape(self.shape)


def _noise(col: I64, m: F64, f: F64, n: int, n_layers: int, min_dof: int = 5) -> F64:
    """Per column and echo: the pooled variance inside the cells of n_layers
    equal bins of r in [0, 1]; columns with fewer than min_dof degrees of
    freedom take the median of the others."""
    layer = np.clip((m * n_layers).astype(np.int64), 0, n_layers - 1)
    cell = col * n_layers + layer
    u, inv, cnt = np.unique(cell, return_inverse=True, return_counts=True)
    cc = u // n_layers
    dof = np.bincount(cc, cnt - 1, minlength=n)
    E = f.shape[1]
    var = np.empty((n, E))
    for e in range(E):
        mean = np.bincount(inv, f[:, e]) / cnt
        ss = np.bincount(inv, (f[:, e] - mean[inv]) ** 2)
        var[:, e] = np.bincount(cc, ss, minlength=n) / np.maximum(dof, 1)
    good = dof >= min_dof
    med = np.median(var[good], axis=0) if good.any() else np.var(f, axis=0)
    var[~good] = med
    out: F64 = np.maximum(var, 1e-6 * med)
    return out


def column_data(
    frame: Frame,
    rim: npt.NDArray[np.integer[Any]],
    echoes: npt.ArrayLike,
    missing: Bool | None = None,
    fused_frac: float = 0.5,
    base_mm: float = 1.0,
    links: Literal["base", "faces"] = "base",
    wm_mm: float = 0.5,
    n_layers: int = 13,
    edge_vox: float = 1.0,
    depth: Literal["equidistant", "equivolume"] = "equidistant",
) -> tuple[ColumnData, dict[str, int]]:
    """The voxel table of the model (see the module notes). echoes: the
    patch's echoes, shape (*shape, E). edge_vox: the width of the edge zones
    left out at the white and the pial surface, in voxels (0: none). depth:
    the relative depth r of the voxels (see the module notes). Returns the
    table and the counts of model_columns, with the number of edge voxels."""
    x = np.asarray(echoes, np.float64)
    if x.ndim == 3:
        x = x[..., None]
    cortex = rim == 3
    if missing is not None:
        cortex = cortex & ~missing
    Tp = path_exit(frame, cortex, rim == 1)
    keep, why = model_columns(frame, cortex, Tp, fused_frac)
    # each column's thickness: the median exit depth of its streamlines
    pc = frame.vertex_column
    has = np.isfinite(Tp) & (pc >= 0)
    T = np.full(frame.n_columns, np.nan)
    if has.any():
        order = np.lexsort((Tp[has], pc[has]))
        cc, tt = pc[has][order], Tp[has][order].astype(np.float64)
        starts = np.r_[0, np.flatnonzero(np.diff(cc)) + 1]
        for a0, b0 in zip(starts, np.r_[starts[1:], len(cc)], strict=True):
            T[cc[a0]] = np.median(tt[a0:b0])
    labels = np.flatnonzero(keep).astype(np.int64)
    new = -np.ones(frame.n_columns, np.int64)
    new[labels] = np.arange(len(labels))
    # cortex voxels of the model columns, on streamlines that exit
    vc, vp = frame.voxel_column, frame.voxel_path
    sel = cortex & (vc >= 0)
    sel[sel] = keep[vc[sel]] & np.isfinite(Tp[vp[sel]])
    idx_c = np.flatnonzero(sel)
    col_c = new[vc.ravel()[idx_c]]
    p_c = vp.ravel()[idx_c].astype(np.int64)
    d_c = frame.depth.ravel()[idx_c].astype(np.float64)
    T_c = Tp[p_c].astype(np.float64)
    if depth == "equivolume":
        A, V = path_volume(frame)
        VT = volume_at(frame, A, V, np.arange(len(Tp)), np.nan_to_num(Tp))
        m_c = volume_at(frame, A, V, p_c, d_c) / VT[p_c]
    else:
        m_c = d_c / T_c
    # the edge zones: partial volume with the WM or the fluid
    edge_mm = edge_vox * float(np.mean(frame.voxel_mm))
    side = np.where(d_c < edge_mm, -1, np.where(T_c - d_c < edge_mm, 1, 0))
    is_edge = side != 0
    edge_index, edge_col, edge_side = idx_c[is_edge], col_c[is_edge], side[is_edge]
    idx_c, col_c, m_c = idx_c[~is_edge], col_c[~is_edge], m_c[~is_edge]
    why["edge_voxels"] = int(is_edge.sum())
    # the WM extension: within wm_mm of the white surface, with the column and
    # the streamline of the nearest white vertex
    wm = rim == 2
    if missing is not None:
        wm = wm & ~missing
    D = frame.distance
    near = wm & np.isfinite(D) & (D >= -wm_mm)
    idx_w = np.flatnonzero(near)
    vox = np.asarray(frame.voxel_mm)
    if len(idx_w) and len(frame.verts):
        tree = cKDTree(frame.verts * vox)
        pts = np.stack(np.unravel_index(idx_w, rim.shape), 1) * vox
        _, jt = tree.query(pts, k=1)
        j = np.asarray(jt)
        cw = frame.vertex_column[j]
        ok = keep[cw] & np.isfinite(Tp[j])
        idx_w, cw, j = idx_w[ok], cw[ok], j[ok]
        col_w = new[cw]
        # below the white surface A = 1, so V = d (negative)
        scale = VT[j] if depth == "equivolume" else Tp[j].astype(np.float64)
        m_w = D.ravel()[idx_w].astype(np.float64) / scale
    else:
        idx_w = np.zeros(0, np.int64)
        col_w = np.zeros(0, np.int64)
        m_w = np.zeros(0)
    index = np.r_[idx_c, idx_w]
    col = np.r_[col_c, col_w]
    m = np.r_[m_c, m_w]
    ctx = np.r_[np.ones(len(idx_c), bool), np.zeros(len(idx_w), bool)]
    f = x.reshape(-1, x.shape[-1])[index]
    edges, w = column_graph(frame, cortex, keep, base_mm, links)
    edges = new[edges]
    sigma2 = _noise(col[ctx], m[ctx], f[ctx], len(labels), n_layers)
    s = rim.shape
    data = ColumnData(
        shape=(s[0], s[1], s[2]),
        index=index.astype(np.int64),
        col=col.astype(np.int64),
        m=m,
        f=f,
        ctx=ctx,
        labels=labels,
        thickness=T[labels],
        edges=edges,
        w=w,
        sigma2=sigma2,
        edge_index=edge_index.astype(np.int64),
        edge_col=edge_col.astype(np.int64),
        edge_side=edge_side.astype(np.int64),
    )
    return data, why


def grid_sums(
    data: ColumnData, h: float, sel: Bool | None = None
) -> tuple[F64, F64, F64]:
    """Per column and depth-grid bin, for the cortex voxels (or sel): the
    voxel count (n, G) and the sums of x and x^2 per echo (n, G, E). The grid
    has G = ceil(1 / h) bins of r in [0, 1]; voxels beyond r = 1 go to the
    last bin, and the few cortex voxels just inside the smoothed white surface
    (r slightly below 0) to the first."""
    G = int(np.ceil(1.0 / h - 1e-9))
    s = data.ctx if sel is None else sel
    c, m, f = data.col[s], data.m[s], data.f[s]
    g = np.clip((m / h).astype(np.int64), 0, G - 1)
    cell = c * G + g
    n, E = data.n, data.n_echoes
    cnt = np.bincount(cell, minlength=n * G).reshape(n, G).astype(np.float64)
    sx = np.empty((n, G, E))
    sxx = np.empty((n, G, E))
    for e in range(E):
        sx[..., e] = np.bincount(cell, f[:, e], minlength=n * G).reshape(n, G)
        sxx[..., e] = np.bincount(cell, f[:, e] ** 2, minlength=n * G).reshape(n, G)
    return cnt, sx, sxx


def drop_small_pieces(data: ColumnData, min_cols: int = 5) -> tuple[ColumnData, int]:
    """The table without the columns in connected pieces of the column graph
    with fewer than min_cols columns (an isolated column could never merge
    with anything, and one column cannot cross-validate its bands). Returns
    the new table and the number of columns left out."""
    n = data.n
    A = sparse.coo_matrix(
        (np.ones(len(data.edges)), (data.edges[:, 0], data.edges[:, 1])), shape=(n, n)
    )
    _, comp = connected_components(A, directed=False)
    keep = np.bincount(comp)[comp] >= min_cols
    if keep.all():
        return data, 0
    new = -np.ones(n, np.int64)
    new[keep] = np.arange(int(keep.sum()))
    v = keep[data.col]
    e = keep[data.edges[:, 0]] & keep[data.edges[:, 1]]
    out = ColumnData(
        shape=data.shape,
        index=data.index[v],
        col=new[data.col[v]],
        m=data.m[v],
        f=data.f[v],
        ctx=data.ctx[v],
        labels=data.labels[keep],
        thickness=data.thickness[keep],
        edges=new[data.edges[e]],
        w=data.w[e],
        sigma2=data.sigma2[keep],
        edge_index=data.edge_index[keep[data.edge_col]],
        edge_col=new[data.edge_col[keep[data.edge_col]]],
        edge_side=data.edge_side[keep[data.edge_col]],
    )
    return out, int((~keep).sum())
