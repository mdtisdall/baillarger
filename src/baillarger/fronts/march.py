"""Anisotropic fast marching from labelled sources, with arc length.

The arrival u solves the eikonal equation grad u^T D grad u = 1 inside a
domain, with u = 0 at the sources: D = M^-1 is the dual of a Riemannian
metric M, so a path costs the integral of sqrt(x'^T M x'). Every reached
voxel also gets the label of the source its arrival comes from, and the
Euclidean length of the path from that source.

The scheme (Mirebeau 2014, "Anisotropic fast-marching on cartesian grids
using lattice basis reduction"; here with Selling's decomposition,
baillarger.fronts.selling) writes D, in voxel units, as sum_i rho_i e_i e_i^T
with rho_i >= 0 and integer offsets e_i, and solves at each voxel x

    sum_i rho_i max(0, u(x) - u(x + e_i), u(x) - u(x - e_i))^2 = 1.

The scheme is monotone and causal, so the voxels can be accepted in order of
arrival, as in Dijkstra's algorithm; the stencil adapts to the metric, so the
grid favours no direction. An offset is used only when the straight segment to
it stays inside the domain, so a front never jumps a thin gap.

Labels: the update of a voxel is computed separately for each label, from that
label's accepted neighbours only, and the voxel takes the earliest. Labels
never mix inside one update, so every voxel's arrival comes from one source,
and fronts stop where they meet. The best arrival of another label at the time
a voxel is accepted is kept as well, for locating the meeting surface between
voxels.

Accuracy (measured in the tests): the scheme is first order. From a point
source, whose singularity dominates the error, the arrival is within about 4-8
% (median) at 5-15 voxels, and its direction within about 1 degree for
metrics with anisotropy 3-10 but about 5-6 degrees for near-isotropic ones
(whose Selling stencil is just the six axis neighbours). Both errors fall when
the grid is refined. A plane source with a metric of unit cost along its
normal is exact.

Arc length: along the characteristic, x' = D grad u, and the Euclidean length
grows at the rate |H D grad u| per unit of arrival (H the voxel size). It is
carried with the same upwind weights as the arrival.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from baillarger.fronts._jit import njit
from baillarger.fronts.selling import selling

type Bool = npt.NDArray[np.bool_]
type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]
type Int8 = npt.NDArray[np.int8]
type Int32 = npt.NDArray[np.int32]

_INF = np.inf


@dataclass(frozen=True)
class Fronts:
    """The result of march. Unreached voxels: arrival inf, label -1, length nan."""

    #: arrival time (the metric's distance from the nearest source)
    arrival: Float
    #: label of the source the arrival comes from
    label: Int
    #: Euclidean length of the path from that source (in the units of spacing)
    length: Float
    #: best arrival of any other label when the voxel was accepted (inf: none)
    second_arrival: Float
    #: that label (-1: none)
    second_label: Int
    #: rank of the voxel in the order of acceptance (-1: unreached)
    order: Int


def riemann_from_direction(
    direction: npt.ArrayLike, ratio: float | npt.ArrayLike
) -> Float:
    """Metric with unit cost along direction and cost ratio across it.

    M = v v^T + ratio^2 (I - v v^T), v the unit direction: a front moves at
    unit speed along v and at speed 1/ratio across it, so with this metric the
    arrival along v is the Euclidean length. Where direction is zero the
    metric is the identity (isotropic, unit cost). direction has shape
    (..., 3); ratio is a scalar or has shape (...).
    """
    v = np.asarray(direction, np.float64)
    r = np.broadcast_to(np.asarray(ratio, np.float64), v.shape[:-1])
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    u = np.divide(v, n, out=np.zeros_like(v), where=n > 0)
    vv = u[..., :, None] * u[..., None, :]
    eye = np.broadcast_to(np.eye(3), vv.shape)
    r2 = np.where(n[..., 0] > 0, r**2, 1.0)[..., None, None]
    out: Float = vv + r2 * (eye - vv)
    return out


@njit
def _stencils(mask: Bool, offsets: Int8, ids: Int, cells: Int) -> Int32:
    """For each domain voxel and each of its 12 signed offsets (2i: +e_i,
    2i + 1: -e_i), the compact id of the neighbour, or -1 when it is outside
    the domain or the segment to it leaves the domain."""
    nx, ny, nz = mask.shape
    m = cells.shape[0]
    nbr = np.full((m, 12), -1, dtype=np.int32)
    for p in range(m):
        c = cells[p]
        x0 = c // (ny * nz)
        y0 = (c // nz) % ny
        z0 = c % nz
        for i in range(6):
            ex = np.int64(offsets[p, i, 0])
            ey = np.int64(offsets[p, i, 1])
            ez = np.int64(offsets[p, i, 2])
            if ex == 0 and ey == 0 and ez == 0:
                continue
            for s in range(2):
                sg = 1 if s == 0 else -1
                x1 = x0 + sg * ex
                y1 = y0 + sg * ey
                z1 = z0 + sg * ez
                if x1 < 0 or y1 < 0 or z1 < 0 or x1 >= nx or y1 >= ny or z1 >= nz:
                    continue
                if not mask[x1, y1, z1]:
                    continue
                steps = max(abs(ex), abs(ey), abs(ez))
                ok = True
                for t in range(1, steps):
                    xa = x0 + sg * ex * t / steps
                    ya = y0 + sg * ey * t / steps
                    za = z0 + sg * ez * t / steps
                    if not mask[round(xa), round(ya), round(za)]:
                        ok = False
                        break
                if ok:
                    nbr[p, 2 * i + s] = ids[x1 * ny * nz + y1 * nz + z1]
    return nbr


@njit
def _reverse(nbr: Int32) -> tuple[Int, Int32]:
    """CSR lists of the voxels whose stencil contains each voxel."""
    m = nbr.shape[0]
    count = np.zeros(m + 1, dtype=np.int64)
    for q in range(m):
        for d in range(12):
            p = nbr[q, d]
            if p >= 0:
                count[p + 1] += 1
    for p in range(m):
        count[p + 1] += count[p]
    fill = count[:-1].copy()
    rev = np.empty(count[m], dtype=np.int32)
    for q in range(m):
        for d in range(12):
            p = nbr[q, d]
            if p >= 0:
                rev[fill[p]] = q
                fill[p] += 1
    return count, rev


@njit
def _heap_push(
    hv: Float, hp: Int, hl: Int, size: int, v: float, p: int, lab: int
) -> int:
    i = size
    hv[i] = v
    hp[i] = p
    hl[i] = lab
    while i > 0:
        j = (i - 1) // 2
        if hv[j] <= hv[i]:
            break
        hv[i], hv[j] = hv[j], hv[i]
        hp[i], hp[j] = hp[j], hp[i]
        hl[i], hl[j] = hl[j], hl[i]
        i = j
    return size + 1


@njit
def _heap_pop(hv: Float, hp: Int, hl: Int, size: int) -> int:
    size -= 1
    hv[0] = hv[size]
    hp[0] = hp[size]
    hl[0] = hl[size]
    i = 0
    while True:
        a = 2 * i + 1
        b = a + 1
        s = i
        if a < size and hv[a] < hv[s]:
            s = a
        if b < size and hv[b] < hv[s]:
            s = b
        if s == i:
            break
        hv[i], hv[s] = hv[s], hv[i]
        hp[i], hp[s] = hp[s], hp[i]
        hl[i], hl[s] = hl[s], hl[i]
        i = s
    return size


@njit
def _update(
    q: int,
    lab: int,
    rhs: float,
    nbr: Int32,
    offsets: Int8,
    weights: Float,
    accepted: npt.NDArray[np.bool_],
    arrival: Float,
    label: Int,
    length: Float,
    spacing: Float,
    wv: Float,
    wr: Float,
    wo: Float,
    ws: Float,
) -> tuple[float, float]:
    """Arrival and length at q from label lab's accepted neighbours only."""
    k = 0
    for i in range(6):
        r = weights[q, i]
        if r <= 0.0:
            continue
        best = _INF
        bs = 0.0
        sg = 0.0
        for side in range(2):
            n = nbr[q, 2 * i + side]
            if n >= 0 and accepted[n] and label[n] == lab and arrival[n] < best:
                best = arrival[n]
                bs = length[n]
                # direction from the neighbour to q, in voxels: -(+e) or -(-e)
                sg = -1.0 if side == 0 else 1.0
        if best < _INF:
            # insertion by value
            j = k
            while j > 0 and wv[j - 1] > best:
                wv[j] = wv[j - 1]
                wr[j] = wr[j - 1]
                ws[j] = ws[j - 1]
                wo[j, 0] = wo[j - 1, 0]
                wo[j, 1] = wo[j - 1, 1]
                wo[j, 2] = wo[j - 1, 2]
                j -= 1
            wv[j] = best
            wr[j] = r
            ws[j] = bs
            wo[j, 0] = sg * offsets[q, i, 0]
            wo[j, 1] = sg * offsets[q, i, 1]
            wo[j, 2] = sg * offsets[q, i, 2]
            k += 1
    if k == 0:
        return _INF, np.nan
    a = 0.0
    b = 0.0
    c = -rhs
    u = _INF
    used = 0
    for j in range(k):
        a += wr[j]
        b += wr[j] * wv[j]
        c += wr[j] * wv[j] * wv[j]
        disc = b * b - a * c
        if disc < 0.0:
            disc = 0.0
        u = (b + np.sqrt(disc)) / a
        used = j + 1
        if j + 1 == k or u <= wv[j + 1]:
            break
    # flow D grad u (voxel units) and its Euclidean speed per unit of arrival
    fx = 0.0
    fy = 0.0
    fz = 0.0
    for j in range(used):
        g = wr[j] * (u - wv[j])
        fx += g * wo[j, 0]
        fy += g * wo[j, 1]
        fz += g * wo[j, 2]
    rate = (
        np.sqrt(
            (fx * spacing[0]) ** 2 + (fy * spacing[1]) ** 2 + (fz * spacing[2]) ** 2
        )
        / rhs
    )
    s = 0.0
    tot = 0.0
    for j in range(used):
        w = wr[j] * (u - wv[j]) ** 2
        s += w * (ws[j] + (u - wv[j]) * rate)
        tot += w
    if tot > 0.0:
        s /= tot
    else:
        s = ws[0]
    return u, s


@njit
def _march(
    nbr: Int32,
    count: Int,
    rev: Int32,
    offsets: Int8,
    weights: Float,
    seeds: Int,
    seed_labels: Int,
    spacing: Float,
) -> tuple[Float, Int, Float, Float, Int, Int]:
    m = nbr.shape[0]
    arrival = np.full(m, _INF)
    label = np.full(m, -1, dtype=np.int64)
    length = np.full(m, np.nan)
    second = np.full(m, _INF)
    second_label = np.full(m, -1, dtype=np.int64)
    order = np.full(m, -1, dtype=np.int64)
    accepted = np.zeros(m, dtype=np.bool_)
    # tentative best and second best (another label), and the best's length
    t1 = np.full(m, _INF)
    l1 = np.full(m, -1, dtype=np.int64)
    s1 = np.full(m, np.nan)
    t2 = np.full(m, _INF)
    l2 = np.full(m, -1, dtype=np.int64)
    cap = 13 * m + seeds.shape[0] + 16
    hv = np.empty(cap)
    hp = np.empty(cap, dtype=np.int64)
    hl = np.empty(cap, dtype=np.int64)
    size = 0
    wv = np.empty(6)
    wr = np.empty(6)
    wo = np.empty((6, 3))
    ws = np.empty(6)
    for n in range(seeds.shape[0]):
        p = seeds[n]
        if 0.0 < t1[p]:
            t1[p] = 0.0
            l1[p] = seed_labels[n]
            s1[p] = 0.0
            size = _heap_push(hv, hp, hl, size, 0.0, p, seed_labels[n])
    rank = 0
    while size > 0:
        v = hv[0]
        p = hp[0]
        lab = hl[0]
        size = _heap_pop(hv, hp, hl, size)
        if accepted[p] or lab != l1[p] or v > t1[p]:
            continue
        accepted[p] = True
        arrival[p] = t1[p]
        label[p] = l1[p]
        length[p] = s1[p]
        second[p] = t2[p]
        second_label[p] = l2[p]
        order[p] = rank
        rank += 1
        for e in range(count[p], count[p + 1]):
            q = rev[e]
            if accepted[q]:
                continue
            u, s = _update(
                q,
                lab,
                1.0,
                nbr,
                offsets,
                weights,
                accepted,
                arrival,
                label,
                length,
                spacing,
                wv,
                wr,
                wo,
                ws,
            )
            if u == _INF:
                continue
            if lab == l1[q]:
                if u < t1[q]:
                    t1[q] = u
                    s1[q] = s
                    size = _heap_push(hv, hp, hl, size, u, q, lab)
            elif u < t1[q]:
                if l1[q] >= 0:
                    t2[q] = t1[q]
                    l2[q] = l1[q]
                t1[q] = u
                l1[q] = lab
                s1[q] = s
                size = _heap_push(hv, hp, hl, size, u, q, lab)
            elif lab == l2[q]:
                if u < t2[q]:
                    t2[q] = u
            elif u < t2[q]:
                t2[q] = u
                l2[q] = lab
    return arrival, label, length, second, second_label, order


def march(
    metric: npt.ArrayLike,
    seed_labels: npt.ArrayLike,
    mask: npt.ArrayLike | None = None,
    spacing: npt.ArrayLike = (1.0, 1.0, 1.0),
) -> Fronts:
    """Fast marching from labelled sources in a 3D domain.

    metric: the Riemannian metric M (cost), shape (nx, ny, nz, 3, 3) or (3, 3)
        for a constant one, in the units of spacing (a path of Euclidean
        length L along a unit vector v costs L sqrt(v^T M v)).
    seed_labels: integer array (nx, ny, nz), the source label (>= 0) at each
        source voxel and -1 elsewhere. Sources start at arrival 0.
    mask: the domain (default: the whole array). Fronts move only inside it,
        and never along a stencil segment that leaves it.
    spacing: voxel size along each axis.
    """
    lab = np.asarray(seed_labels)
    if lab.ndim != 3 or not np.issubdtype(lab.dtype, np.integer):
        raise ValueError("seed_labels must be a 3D integer array")
    shape = lab.shape
    dom = np.ones(shape, bool) if mask is None else np.asarray(mask, bool)
    if dom.shape != shape:
        raise ValueError(f"mask has shape {dom.shape}, seed_labels {shape}")
    if np.any((lab >= 0) & ~dom):
        raise ValueError("a source lies outside the mask")
    h = np.asarray(spacing, np.float64)
    if h.shape != (3,) or np.any(h <= 0):
        raise ValueError("spacing must be three positive numbers")
    met = np.asarray(metric, np.float64)
    if met.shape == (3, 3):
        met = np.broadcast_to(met, (*shape, 3, 3))
    if met.shape != (*shape, 3, 3):
        raise ValueError(f"metric has shape {met.shape}, expected {(*shape, 3, 3)}")
    cells = np.flatnonzero(dom.ravel()).astype(np.int64)
    ids = np.full(dom.size, -1, dtype=np.int64)
    ids[cells] = np.arange(cells.size)
    # dual metric in voxel units: H^-1 M^-1 H^-1
    md = met.reshape(-1, 3, 3)[cells]
    dual = np.linalg.inv(md) / (h[:, None] * h[None, :])
    off, w = selling(dual)
    nbr = _stencils(dom, off, ids, cells)
    count, rev = _reverse(nbr)
    seed_cells = np.flatnonzero(lab.ravel() >= 0)
    a, lb, ln, s2, l2, od = _march(
        nbr,
        count,
        rev,
        off,
        w,
        ids[seed_cells],
        lab.ravel()[seed_cells].astype(np.int64),
        h,
    )

    def full(x: npt.NDArray[np.generic], fill: float) -> npt.NDArray[np.generic]:
        out = np.full(dom.size, fill, dtype=x.dtype)
        out[cells] = x
        return out.reshape(shape)

    return Fronts(
        arrival=full(a, np.inf).astype(np.float64),
        label=full(lb, -1).astype(np.int64),
        length=full(ln, np.nan).astype(np.float64),
        second_arrival=full(s2, np.inf).astype(np.float64),
        second_label=full(l2, -1).astype(np.int64),
        order=full(od, -1).astype(np.int64),
    )
