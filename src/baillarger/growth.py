"""Topology-preserving growth of the WM through the cortex, and the seams it
leaves.

The object starts as the WM. Cortex voxels join it in order of a priority
(lowest first), and a voxel joins only when it is a simple point
(baillarger.topology.is_simple): its addition changes neither the components
nor the handles nor the cavities of the object. A voxel that is not simple now
is tested again whenever one of its neighbours joins. Cortex voxels that never
join are the seam.

Where the two banks of a collapsed sulcus touch, joining them would close a
loop round the CSF beyond the contact, a new handle, so the voxels where the
two fronts meet never join. A collapsed sulcus that ends inside the cortex (a
slit, with no CSF beyond) changes no topology when it is filled, so the caller
removes the slits the FLASH shows from the candidates first (visible_sheets).
Where the cortex wraps round the end of a sulcus, joining encloses nothing, so
no seam forms there. Dark voxels come late in the order (seam_priority), so a
necessary seam falls on a dark line when there is one. Darkness is measured as
thin dark sheets (sheetness), not raw intensity: a dark band inside one bank
(the stria of Gennari) does not change the topology and joins in the end, but
if it were costly to cross, the other bank's front could cross a fused contact
first and the band would become the seam.

grow_competing is the same growth against a second front: fluid that flows in
from the outside, fast along dark lines. thin_seam and mincut_seam then thin
the seam and move each cut onto the darkest surface between the two banks,
keeping the topology.

The volume is padded with one voxel of background, so every voxel tested has
all its 26 neighbours in the array.
"""

import heapq
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.csgraph import breadth_first_order, dijkstra, maximum_flow

from baillarger._jit import njit
from baillarger.topology import is_simple

type Bool = npt.NDArray[np.bool_]
type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]


def _offsets(shape: npt.NDArray[np.intp]) -> Int:
    """Flat-index offsets of the 26 neighbours in an array of this shape."""
    d = np.array(
        [
            (i, j, k)
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            for k in (-1, 0, 1)
            if (i, j, k) != (0, 0, 0)
        ]
    )
    offs: Int = (d[:, 0] * shape[1] * shape[2] + d[:, 1] * shape[2] + d[:, 2]).astype(
        np.int64
    )
    return offs


def _pad(a: npt.NDArray[Any], v: Any) -> npt.NDArray[Any]:
    return np.pad(a, 1, constant_values=v)


@njit
def _grow(
    o: Bool,
    c: Bool,
    pri: Float,
    shape: tuple[int, int, int],
    conn: int,
    offs: Int,
) -> Int:
    """o, c: flat padded bool arrays (object, changed in place; candidates);
    pri: flat float64; shape: the padded shape; offs: flat-index offsets of the
    26 neighbours. Returns the flat indices of the voxels in the order they
    joined."""
    nx, ny, nz = shape
    obj = o.reshape((nx, ny, nz))  # a view: is_simple sees each change
    heap = [(0.0, np.int64(0))]
    heap.pop()
    queued = np.zeros(o.size, np.bool_)
    for v in range(o.size):
        if c[v] and not o[v]:
            for k in range(26):
                j = v + offs[k]
                if o[j]:
                    heapq.heappush(heap, (pri[v], np.int64(v)))
                    queued[v] = True
                    break
    order = np.empty(int(c.sum()), np.int64)
    n = 0
    while len(heap) > 0:
        p, i = heapq.heappop(heap)
        queued[i] = False
        if o[i]:
            continue
        x = i // (ny * nz)
        y = (i // nz) % ny
        z = i % nz
        if not is_simple(obj, x, y, z, conn):
            continue  # deferred: tested again when a neighbour joins
        o[i] = True
        order[n] = i
        n += 1
        for k in range(26):
            j = i + offs[k]
            if c[j] and not o[j] and not queued[j]:
                heapq.heappush(heap, (pri[j], j))
                queued[j] = True
    return order[:n]


def grow_topological(
    obj0: npt.ArrayLike,
    cand: npt.ArrayLike,
    priority: npt.ArrayLike,
    conn: int,
) -> tuple[Bool, Bool, Int]:
    """Grow the object obj0 (3D bool) into the candidate voxels cand (3D bool)
    in order of priority (3D float), keeping its topology for the connectivity
    pair conn (CONN_26_6 or CONN_6_26 of baillarger.topology).

    Returns the final object, the seam (candidate voxels next to the object
    that never joined) and the flat indices (in the unpadded array) of the
    joined voxels in joining order. Candidate voxels with no path to the object
    are in neither.
    """
    obj = _pad(np.asarray(obj0, bool), False)
    c = _pad(np.asarray(cand, bool) & ~np.asarray(obj0, bool), False)
    pri = _pad(np.asarray(priority, np.float64), np.inf)
    sh = np.array(obj.shape)
    offs = _offsets(sh)
    o = obj.ravel()
    order_p = _grow(o, c.ravel(), pri.ravel(), (sh[0], sh[1], sh[2]), conn, offs)
    obj = o.reshape(obj.shape)[1:-1, 1:-1, 1:-1]
    cand_u = np.asarray(cand, bool) & ~np.asarray(obj0, bool)
    # seam: candidate voxels that never joined but touch the grown object
    near = ndi.binary_dilation(obj, np.ones((3, 3, 3), bool))
    seam = cand_u & ~obj & near
    # flat indices in the unpadded array
    xp, yp, zp = np.unravel_index(order_p, tuple(sh))
    order = np.ravel_multi_index((xp - 1, yp - 1, zp - 1), obj.shape)
    return obj, seam, order


def _hessian(flash: npt.ArrayLike, mask: Bool, sigma: float) -> Float:
    """Scale-normalized Hessian of the FLASH (smoothed by sigma voxels, over the
    median in mask), at the voxels of mask: shape (n, 3, 3)."""
    med = np.median(np.asarray(flash)[mask])  # in flash's own dtype
    f = np.asarray(flash, np.float64)
    H = np.zeros((int(mask.sum()), 3, 3))
    for i in range(3):
        for j in range(i, 3):
            o = [0, 0, 0]
            o[i] += 1
            o[j] += 1
            H[:, i, j] = H[:, j, i] = (
                ndi.gaussian_filter(f, sigma, order=o)[mask] * sigma**2 / med
            )
    return H


def sheetness(flash: npt.ArrayLike, mask: Bool, sigma: float = 0.7) -> Float:
    """Thin dark sheets: the largest Hessian eigenvalue of the FLASH (smoothed
    by sigma voxels, scale-normalized, over the median in mask), minus the other
    two (half of the smallest), in mask; 0 elsewhere.

    A collapsed sulcus, one voxel of CSF between two banks, gives a high value;
    a wider band such as the stria of Gennari gives a lower one.
    """
    ev = np.linalg.eigvalsh(_hessian(flash, mask, sigma))  # ascending
    out = np.zeros(mask.shape)
    out[mask] = np.maximum(ev[:, 2], 0) - np.abs(ev[:, 1]) - 0.5 * np.abs(ev[:, 0])
    return out


def sheet_centre(flash: npt.ArrayLike, mask: Bool, sigma: float = 0.7) -> Float:
    """The sheet measure (as sheetness), kept only where it is at least as high
    as at the two points one voxel away along the sheet's normal (the
    eigenvector of the largest Hessian eigenvalue), 0 elsewhere: the centre of a
    dark sheet, not the partial-volume halo beside it."""
    ev, vec = np.linalg.eigh(_hessian(flash, mask, sigma))  # ascending
    s = np.maximum(ev[:, 2], 0) - np.abs(ev[:, 1]) - 0.5 * np.abs(ev[:, 0])
    full = np.zeros(mask.shape)
    full[mask] = s
    n = vec[:, :, 2]  # the sheet's normal
    pts = np.argwhere(mask).astype(np.float64)
    up = ndi.map_coordinates(full, (pts + n).T, order=1, mode="nearest")
    dn = ndi.map_coordinates(full, (pts - n).T, order=1, mode="nearest")
    out = np.zeros(mask.shape)
    out[mask] = np.where((s >= up) & (s >= dn), s, 0.0)
    return out


def darkness(
    flash: npt.ArrayLike,
    cortex: Bool,
    sigma: float = 0.7,
    sheet_lo: float = 0.12,
    sheet_hi: float = 0.25,
    int_lo: float = 0.6,
    int_hi: float = 0.3,
    centre: bool = False,
) -> Float:
    """How much each cortex voxel looks like a fluid channel, 0 to 1.

    The larger of the sheet measure (from sheet_lo to sheet_hi; with centre,
    only at the centre of a sheet, see sheet_centre) and the darkness of the
    FLASH smoothed by sigma (over the cortex median, from int_lo down to
    int_hi).
    """
    fl = np.asarray(flash)
    rel = ndi.gaussian_filter(np.asarray(fl, np.float64), sigma) / np.median(fl[cortex])
    sh = sheet_centre(fl, cortex, sigma) if centre else sheetness(fl, cortex, sigma)
    out: Float = np.maximum(
        np.clip((sh - sheet_lo) / (sheet_hi - sheet_lo), 0, 1),
        np.clip((int_lo - rel) / (int_lo - int_hi), 0, 1),
    )
    return out


_HALF26 = [
    (i, j, k)
    for i in (-1, 0, 1)
    for j in (-1, 0, 1)
    for k in (-1, 0, 1)
    if (i, j, k) > (0, 0, 0)
]


def _shifts(
    shape: tuple[int, ...], o: tuple[int, int, int]
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    """Slices of the voxels and of their neighbours at offset o."""
    src = tuple(slice(max(0, -d), shape[k] - max(0, d)) for k, d in enumerate(o))
    dst = tuple(slice(max(0, d), shape[k] - max(0, -d)) for k, d in enumerate(o))
    return src, dst


def seam_priority(
    rim: npt.NDArray[np.integer[Any]],
    flash: npt.ArrayLike,
    voxel_mm: float | npt.ArrayLike,
    dark_weight: float = 30.0,
    sheet_lo: float = 0.12,
    sheet_hi: float = 0.25,
    sigma: float = 0.7,
    int_lo: float = 0.6,
    int_hi: float = 0.3,
) -> Float:
    """Priority of each cortex voxel for grow_topological; inf elsewhere.

    The priority is the cheapest path cost from the WM through the cortex,
    where each step costs its length (mm) x (1 + dark_weight x the mean
    darkness of its two voxels). Darkness is the larger of two measures: the
    sheet measure (sheetness) mapped linearly from sheet_lo (0) to sheet_hi (1),
    and the FLASH (smoothed by sigma, over the cortex median) mapped from int_lo
    (0) down to int_hi (1). The second covers the ends of a dark line, where it
    meets open CSF and is no longer a sheet to the Hessian; it takes only very
    dark voxels, darker than the stria.

    A dark line is then expensive to cross, so the voxels behind it are far
    from the other bank along every path: a weak spot in the line lets the
    other bank's front in only locally, and the two fronts meet on the line. (A
    delay on the dark voxels alone is not enough: one weak spot lets a front
    through, and the voxels behind the line are near in straight distance.) A
    wider, fainter band (the stria) has low sheetness and is cheap to cross, so
    it does not hold back its own bank's front.

    Defaults, from the V1V2 crop (sheetness at sigma 0.7): collapsed sulci that
    the FLASH shows have sheetness 0.16 (5th percentile) to 0.26 (median); dark
    mid-depth cortex (the stria) has 0.01 (median) and 0.11 (99th percentile).
    A weight of 30 makes one voxel of full darkness cost about 5 mm extra, more
    than the difference in distance from the two banks' WM where one bank is
    about 2 mm thicker.
    """
    vox = np.broadcast_to(np.asarray(voxel_mm, float), (3,))
    fl = np.asarray(flash)
    gm = rim == 3
    wm = rim == 2
    tis = gm | wm
    ids = -np.ones(rim.shape, np.int64)
    ids[tis] = np.arange(tis.sum())
    dark: Float
    if dark_weight:
        rel = ndi.gaussian_filter(np.asarray(fl, np.float64), sigma) / np.median(fl[gm])
        dark = np.maximum(
            np.clip(
                (sheetness(fl, gm, sigma) - sheet_lo) / (sheet_hi - sheet_lo), 0, 1
            ),
            np.clip((int_lo - rel) / (int_lo - int_hi), 0, 1),
        )
    else:
        dark = np.zeros(rim.shape)
    rows, cols, w = [], [], []
    for o in _HALF26:
        src, dst = _shifts(rim.shape, o)
        m = tis[src] & tis[dst]
        rows.append(ids[src][m])
        cols.append(ids[dst][m])
        cost = 1 + dark_weight * 0.5 * (dark[src][m] + dark[dst][m])
        w.append(float(np.linalg.norm(np.array(o) * vox)) * cost)
    n = int(tis.sum())
    G = sparse.coo_matrix(
        (np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsr()
    dist = dijkstra(G, directed=False, indices=ids[wm], min_only=True)
    D = np.full(rim.shape, np.inf)
    D[tis] = dist
    return np.where(gm, D, np.inf)


def visible_sheets(
    rim: npt.NDArray[np.integer[Any]],
    flash: npt.ArrayLike,
    voxel_mm: float | npt.ArrayLike,
    thr: float = 0.15,
    edge_mm: float = 0.17,
    sigma: float = 0.7,
) -> Bool:
    """Collapsed sulci that the FLASH shows: cortex voxels with sheetness above
    thr, in pieces connected to the outside label (rim == 1).

    Voxels within edge_mm of the outside label (dark partial volume at an
    ordinary pial surface) are left out, unless they continue a deeper sheet
    (within 2 voxels of one), so the sheet reaches the open sulcus.

    The growth must not join these voxels: a collapsed sulcus that ends inside
    the cortex is a slit, and filling it changes no topology, so only the image
    can keep it open. Where such a sheet has a hole, joining across the hole
    would close a loop round the sheet, so the topology-preserving growth leaves
    the hole open too.
    """
    vox = np.broadcast_to(np.asarray(voxel_mm, float), (3,))
    gm = rim == 3
    cand = gm & (sheetness(flash, gm, sigma) > thr)
    lab, _ = ndi.label(cand, structure=np.ones((3, 3, 3)))
    touch = np.unique(lab[ndi.binary_dilation(rim == 1) & cand])
    keep = np.isin(lab, touch[touch > 0])
    dout = ndi.distance_transform_edt(rim != 1, sampling=vox)
    deep = keep & (dout > edge_mm)
    return deep | (keep & ndi.binary_dilation(deep, iterations=2))


# ---------------------------------------------------------------------------
# Tissue against fluid: a topological watershed with two markers
# ---------------------------------------------------------------------------


@njit
def _compete(
    state: npt.NDArray[np.int8],
    obj: npt.NDArray[np.uint8],
    cost_t: Float,
    cost_f: Float,
    shape: tuple[int, int, int],
    offs: Int,
    lens: Float,
    conn: int,
) -> int:
    """One queue, two fronts. state (flat int8): 0 not available, 1 tissue,
    2 fluid, 3 cortex not yet assigned; obj (3D uint8 view) mirrors state == 1
    for the simple-point test. cost_t, cost_f: cost per mm of entering each
    voxel for the tissue and the fluid front. lens: step lengths of the 26
    offsets (mm).

    A voxel goes to the front that reaches it first. Tissue joins only through
    simple points; a voxel the tissue may not join is tested again when a
    neighbour joins, and the fluid can still take it. Returns the number of
    deferred voxels.
    """
    nx, ny, nz = shape
    n = state.size
    best_t = np.full(n, np.inf)
    best_f = np.full(n, np.inf)
    deferred = np.zeros(n, np.bool_)
    heap = [(0.0, np.int64(0), np.int64(0))]
    heap.pop()
    for v in range(n):
        if state[v] != 3:
            continue
        for k in range(26):
            j = v + offs[k]
            if state[j] == 1:
                t = lens[k] * cost_t[v]
                if t < best_t[v]:
                    best_t[v] = t
            elif state[j] == 2:
                t = lens[k] * cost_f[v]
                if t < best_f[v]:
                    best_f[v] = t
        if best_t[v] < np.inf:
            heapq.heappush(heap, (best_t[v], np.int64(v), np.int64(1)))
        if best_f[v] < np.inf:
            heapq.heappush(heap, (best_f[v], np.int64(v), np.int64(2)))
    ndef = 0
    while len(heap) > 0:
        t, i, lab = heapq.heappop(heap)
        if state[i] != 3:
            continue
        if lab == 1:
            x = i // (ny * nz)
            y = (i // nz) % ny
            z = i % nz
            if not is_simple(obj, x, y, z, conn):
                if not deferred[i]:
                    ndef += 1
                deferred[i] = True
                continue
            state[i] = 1
            obj[x, y, z] = 1
            for k in range(26):
                j = i + offs[k]
                if state[j] != 3:
                    continue
                tj = t + lens[k] * cost_t[j]
                if tj < best_t[j] or deferred[j]:
                    # a deferred voxel is tested again now that a neighbour joined
                    if tj < best_t[j]:
                        best_t[j] = tj
                    heapq.heappush(heap, (max(tj, t), np.int64(j), np.int64(1)))
        else:
            state[i] = 2
            for k in range(26):
                j = i + offs[k]
                if state[j] != 3:
                    continue
                tj = t + lens[k] * cost_f[j]
                if tj < best_f[j]:
                    best_f[j] = tj
                    heapq.heappush(heap, (tj, np.int64(j), np.int64(2)))
    return ndef


def grow_competing(
    rim: npt.NDArray[np.integer[Any]],
    flash: npt.ArrayLike,
    voxel_mm: float | npt.ArrayLike,
    conn: int = 26,
    fluid_fast: float = 10.0,
    fluid_slow: float = 30.0,
    jitter: float = 1e-4,
    seed: int = 0,
    **dark_kw: Any,
) -> tuple[Bool, Bool, int]:
    """Assign each cortex voxel (rim == 3) to tissue, grown from the WM
    (rim == 2), or to fluid, flowing in from the outside (rim == 1).

    One queue, the earlier arrival wins (an image foresting transform with two
    seed sets), and the tissue keeps its topology (it joins only through simple
    points). Voxels with rim == 0 (missing tissue, nothing) take no part.

    Costs per mm: tissue 1 everywhere (times a jitter of up to jitter, so that
    ties do not follow the voxel order); fluid 1/fluid_fast in a channel
    (darkness 1, see darkness; dark_kw goes to it) up to 1/fluid_fast +
    fluid_slow in bright cortex (darkness 0).

    The fluid runs fast along a dark line, from the open sulcus inward, and
    barely enters bright cortex; a dark band inside one bank (the stria) is
    behind bright superficial cortex, so the tissue reaches it first. Where
    topology forces a cut (a bridge of fused cortex over a sulcus that is open
    on both sides), the cut goes where the tissue closes the loop last.

    Returns tissue (bool: WM and the cortex it took), fluid (bool: the cortex
    voxels the fluid took, or that the tissue could not take: the seam) and the
    number of deferred voxels.
    """
    vox = np.broadcast_to(np.asarray(voxel_mm, float), (3,))
    cortex = rim == 3
    d = darkness(flash, cortex, **dark_kw) if fluid_slow else np.zeros(rim.shape)
    rng = np.random.default_rng(seed)
    cost_t = 1.0 + jitter * rng.random(rim.shape)
    cost_f = 1.0 / fluid_fast + fluid_slow * (1.0 - d)
    state = np.zeros(rim.shape, np.int8)
    state[rim == 2] = 1
    state[rim == 1] = 2
    state[cortex] = 3
    st = _pad(state, 0)
    ct = _pad(cost_t, np.inf)
    cf = _pad(cost_f, np.inf)
    sh = np.array(st.shape)
    dd = np.array(
        [
            (i, j, k)
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            for k in (-1, 0, 1)
            if (i, j, k) != (0, 0, 0)
        ]
    )
    offs = _offsets(sh)
    lens = np.linalg.norm(dd * vox, axis=1)
    obj = (st == 1).astype(np.uint8)
    flat = st.ravel()
    ndef = _compete(
        flat, obj, ct.ravel(), cf.ravel(), (sh[0], sh[1], sh[2]), offs, lens, conn
    )
    st = flat.reshape(sh)[1:-1, 1:-1, 1:-1]
    tissue = st == 1
    fluid = cortex & ~tissue  # taken by the fluid, or left: the seam
    return tissue, fluid, int(ndef)


def thin_seam(
    tissue: Bool,
    seam: Bool,
    flash: npt.ArrayLike,
    cortex: Bool,
    dark: Float | None = None,
    dark_keep: float = 0.5,
    sigma: float = 0.7,
    conn: int = 26,
    **dark_kw: Any,
) -> tuple[Bool, Bool]:
    """Thin the seam to what the topology needs.

    Seam voxels that are not channel-like (darkness below dark_keep) join the
    tissue again, brightest first, each only if it is a simple point
    (grow_topological). A cut that topology forces cannot vanish, so it shrinks
    to a thin sheet, and the darkest voxels are the last to go back: the sheet
    ends on the faint line between the banks. Channels the fluid followed
    (dark) stay open, as do slits with no topology of their own.

    Returns the new tissue and the new seam.
    """
    if dark is None:
        dark = darkness(flash, cortex, sigma, **dark_kw)
    eligible = seam & (dark < dark_keep)
    key = -ndi.gaussian_filter(np.asarray(flash, np.float64), sigma)  # brightest first
    obj, _, _ = grow_topological(tissue, eligible, key, conn)
    return obj, seam & ~obj


@njit
def _erode(
    o: Bool,
    cand: Bool,
    key: Float,
    shape: tuple[int, int, int],
    conn: int,
    offs: Int,
) -> Int:
    """Remove candidate voxels from the object, lowest key first, each only if
    it is a simple point; a voxel that is not simple now is tested again when a
    neighbour leaves. o, cand: flat padded bool arrays (o changed in place);
    key: flat float64. Returns the flat indices removed, in order."""
    nx, ny, nz = shape
    obj = o.reshape((nx, ny, nz))
    heap = [(0.0, np.int64(0))]
    heap.pop()
    queued = np.zeros(o.size, np.bool_)
    for v in range(o.size):
        if cand[v] and o[v]:
            heapq.heappush(heap, (key[v], np.int64(v)))
            queued[v] = True
    out = np.empty(int(cand.sum()), np.int64)
    n = 0
    while len(heap) > 0:
        kv, i = heapq.heappop(heap)
        queued[i] = False
        if not o[i]:
            continue
        x = i // (ny * nz)
        y = (i // nz) % ny
        z = i % nz
        if not is_simple(obj, x, y, z, conn):
            continue
        o[i] = False
        out[n] = i
        n += 1
        for k in range(26):
            j = i + offs[k]
            if cand[j] and o[j] and not queued[j]:
                heapq.heappush(heap, (key[j], j))
                queued[j] = True
    return out[:n]


def _pairs(m: Bool) -> tuple[Int, Int]:
    """26-neighbour pairs of the voxels in m (3D bool), each once: two arrays of
    indices into np.flatnonzero(m)."""
    ids = -np.ones(m.shape, np.int64)
    ids[m] = np.arange(int(m.sum()))
    a, b = [], []
    for o in _HALF26:
        src, dst = _shifts(m.shape, o)
        both = m[src] & m[dst]
        a.append(ids[src][both])
        b.append(ids[dst][both])
    return np.concatenate(a), np.concatenate(b)


def _min_cut(
    n: int,
    a: Int,
    b: Int,
    cab: Int,
    cba: Int,
    src: Int,
    snk: Int,
    big: int = 10**8,
) -> Bool:
    """Minimum s-t cut of a directed graph of n nodes: edges a->b with integer
    capacity cab and b->a with cba, source nodes src, sink nodes snk. Returns
    the source side (bool over the n nodes)."""
    s, t = n, n + 1
    rows = np.concatenate([a, b, np.full(len(src), s), snk])
    cols = np.concatenate([b, a, src, np.full(len(snk), t)])
    c = np.concatenate(
        [cab, cba, np.full(len(src), big), np.full(len(snk), big)]
    ).astype(np.int32)
    G = sparse.csr_matrix((c, (rows, cols)), shape=(n + 2, n + 2))
    G.sum_duplicates()
    # scipy accepts csr_matrix here; its stubs list only csr_array
    F = maximum_flow(G, s, t, method="dinic").flow  # type: ignore[arg-type]
    R = (G - F).tocsr()  # residual capacities (F is antisymmetric)
    R.data = (R.data > 0).astype(np.int8)
    R.eliminate_zeros()
    order = breadth_first_order(R, s, directed=True, return_predecessors=False)
    side = np.zeros(n + 2, bool)
    side[order] = True
    return side[:n]


def _cut_region(
    C: Bool, tissue: Bool, cortex: Bool, w: Float, band: int, min_shell: int
) -> Bool | None:
    """The minimum cut of mincut_seam in one box: C the seam voxels to move,
    the other arrays the box's tissue, cortex and voxel cost. Returns the new
    cut (bool, box shape), or None when the band has fewer than two banks."""
    S = np.ones((3, 3, 3), bool)
    d = ndi.distance_transform_edt(~C)
    B = cortex & (tissue | C) & (d <= band)
    T = B & ~C
    # every path out of the band steps through these
    shell = T & ndi.binary_dilation(tissue & ~B, S)
    labT, nT = ndi.label(T, S)
    cnt = np.bincount(labT[shell], minlength=nT + 1)
    cnt[0] = 0
    banks = [k for k in np.argsort(cnt)[::-1][:2] if cnt[k] >= min_shell]
    if len(banks) < 2:
        return None
    idx = np.flatnonzero(B)
    pos = -np.ones(B.shape, np.int64)
    pos[B] = np.arange(len(idx))
    a, b = _pairs(B)
    src = pos[shell & (labT == banks[0])]
    snk = pos[shell & (labT == banks[1])]
    wb = w[B].copy()
    wb[src] = wb[snk] = 100.0  # terminals stay tissue
    cw = np.maximum(np.round(100 * wb), 1).astype(np.int64)
    side = _min_cut(len(idx), a, b, cw[b], cw[a], src, snk)  # an edge costs its head
    mark = np.concatenate([b[side[a] & ~side[b]], a[side[b] & ~side[a]]])
    M = np.zeros(B.shape, bool)
    M.ravel()[idx[mark]] = True
    return M


def mincut_seam(
    tissue: Bool,
    seam: Bool,
    flash: npt.ArrayLike,
    cortex: Bool,
    band: int = 6,
    window: int = 20,
    margin: int = 8,
    dark: Float | None = None,
    dark_keep: float = 0.5,
    sigma: float = 0.7,
    conn: int = 26,
    min_shell: int = 8,
    **dark_kw: Any,
) -> tuple[Bool, Bool, dict[str, int]]:
    """Put each topological cut on the darkest surface that separates the same
    two banks, through the holes of a dark line as well.

    For each 26-component of the seam, the cortex within band voxels of it
    (tissue or this seam) is a graph of 26-neighbours. The terminals are the
    two banks: the band tissue voxels next to tissue outside the band (cortex
    farther away, or the WM), in the two 26-components of band tissue with the
    most such voxels (each with at least min_shell). Each edge from the first
    bank's side to the second's costs the cost of the voxel on the second's
    side, 0.05 + (1 - darkness) x the FLASH (smoothed by sigma, over the cortex
    median), and that voxel is on the new cut. A minimum cut between the banks
    then runs along a dark line, one voxel thick, and crosses its holes by the
    shortest bright bridge. Every 26-path from one bank to the other in the band
    crosses a cut edge, so the new cut separates the banks. A seam component
    with fewer than two such banks is not a separating cut and is left as it is.

    A component wider than 2 x window voxels is cut in windows: cores of window
    voxels tile the band round it, each cut in a box with margin more voxels on
    every side, and each core keeps its part of the cut.

    The move keeps the topology: the band tissue leaves the tissue, nearest the
    seam first, each voxel only if it is a simple point; then it, and the old
    seam, join again, brightest first, each only if it is a simple point, except
    the new cut and channels (darkness at least dark_keep), which stay open.

    Returns the new tissue, the new seam, and a report (dict): the number of
    seam components, of cuts made and skipped, of voxels on the new cuts, and
    of new-cut voxels that could not be peeled (not_peeled, when a cut was
    made).
    """
    if dark is None:
        dark = darkness(flash, cortex, sigma, **dark_kw)
    fl = np.asarray(flash)
    bright = ndi.gaussian_filter(np.asarray(fl, np.float64), sigma)
    w = 0.05 + (1 - dark) * np.clip(bright / np.median(fl[cortex]), 0, 2)
    S = np.ones((3, 3, 3), bool)
    lab, n = ndi.label(seam, S)
    pinned = np.zeros(seam.shape, bool)
    free = np.zeros(seam.shape, bool)
    moved = np.zeros(seam.shape, bool)
    rep = dict(components=int(n), cut=0, skipped=0, cut_voxels=0)
    for i, sl in enumerate(ndi.find_objects(lab)):
        assert sl is not None  # labels 1..n all occur
        ext = [s_.stop - s_.start for s_ in sl]
        cores: list[tuple[slice, ...]]
        if max(ext) <= 2 * window:
            cores = [sl]
        else:  # a large component: one cut per window
            # cores tile the band round the component: the new cut can lie where
            # the old seam is not
            ob = [
                slice(max(0, s_.start - band), min(dim, s_.stop + band))
                for s_, dim in zip(sl, seam.shape, strict=True)
            ]
            cores = [
                tuple(
                    slice(o.start + k * window, min(o.start + (k + 1) * window, o.stop))
                    for k, o in zip(org, ob, strict=True)
                )
                for org in np.ndindex(*[-(-(o.stop - o.start) // window) for o in ob])
            ]
        for core in cores:
            m = band if core is sl else band + margin
            box = tuple(
                slice(max(0, c.start - m - 1), min(dim, c.stop + m + 1))
                for c, dim in zip(core, seam.shape, strict=True)
            )
            if not (lab[box] == i + 1).any():
                continue
            M = _cut_region(
                lab[box] == i + 1, tissue[box], cortex[box], w[box], band, min_shell
            )
            if M is None:
                rep["skipped"] += 1
                continue
            keep = np.zeros(M.shape, bool)
            keep[
                tuple(
                    slice(c.start - b.start, c.stop - b.start)
                    for c, b in zip(core, box, strict=True)
                )
            ] = True
            if core is sl:
                keep[:] = True
            C = lab[box] == i + 1
            d = ndi.distance_transform_edt(~C)
            B = cortex[box] & (tissue[box] | C) & (d <= band)
            pinned[box] |= M & keep
            free[box] |= B & keep & ~ndi.binary_dilation(tissue[box] & ~B, S)
            moved[box] |= C & keep
            rep["cut"] += 1
    rep["cut_voxels"] = int(pinned.sum())
    if not rep["cut"]:
        return tissue.copy(), seam.copy(), rep
    # peel the band, nearest the seam first, then let everything but the new
    # cuts and channels come back
    o = _pad(np.asarray(tissue, bool), False).copy()
    sh = np.array(o.shape)
    offs = _offsets(sh)
    key = ndi.distance_transform_edt(~seam)
    flat = o.ravel()
    _erode(
        flat,
        _pad(free & tissue, False).ravel(),
        _pad(key, np.inf).ravel(),
        (sh[0], sh[1], sh[2]),
        conn,
        offs,
    )
    peeled = flat.reshape(sh)[1:-1, 1:-1, 1:-1].copy()
    removed = tissue & ~peeled
    regrow = (removed | (seam & moved)) & ~pinned & ~(seam & (dark >= dark_keep))
    obj, _, _ = grow_topological(peeled, regrow, -bright, conn)
    rep["not_peeled"] = int((free & tissue & pinned & peeled).sum())
    return obj, (seam | removed) & ~obj, rep
