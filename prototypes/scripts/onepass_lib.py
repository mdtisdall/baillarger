"""One-pass weighted competition from the white matter (plan, Next step 0).

One anisotropic fast-marching pass (baillarger.fronts.march) from every
WM-boundary voxel (the seeds: WM voxels 6-adjacent to cortex). The labels are
root patches: geodesic Voronoi cells on the seed graph (26-neighbours whose
outward normals agree, so the two sides of a thin WM blade are not joined),
with centres chosen greedily at least `patch` apart along the graph. A
patch's front is computed on full stencils inside the patch; one label per
root voxel would leave each label a one-voxel tube with one-sided updates
(2-3 times the arrival error, measured).

Each patch has a thickness belief (mu, sigma). The competition is
min_i (d_i - mu_i) / sigma_i, as start time -mu/sigma and speed sigma, with
crystal growth. A contact is a label boundary between patches whose centres
are more than `sep_min` apart along the white surface (geodesic on the seed
graph). Neighbouring patches' regions also touch where a patch's region
pinches out, so adjacency alone does not separate one front from two.

pure_voronoi is the brute-force reference: one march per patch, and the
weighted minimum taken afterwards (no crystal growth).

Everything is 3D. Units are voxels unless stated.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix

from baillarger.fronts import Fronts, march, riemann_from_direction

type Bool = npt.NDArray[np.bool_]
type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]

N6 = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
# half of the 26-neighbourhood (each unordered pair once)
N13 = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) > (0, 0, 0)
]


def shifted_pairs(a: Bool, b: Bool, off: tuple[int, int, int]) -> tuple[Int, Int]:
    """Flat indices (p, q) with q = p + off, a[p] and b[q]."""
    sl_p, sl_q = [], []
    for o, n in zip(off, a.shape, strict=True):
        sl_p.append(slice(max(0, -o), n - max(0, o)))
        sl_q.append(slice(max(0, o), n - max(0, -o)))
    both = a[tuple(sl_p)] & b[tuple(sl_q)]
    idx = np.nonzero(both)
    p = np.ravel_multi_index(
        tuple(i + s.start for i, s in zip(idx, sl_p, strict=True)), a.shape
    )
    q = np.ravel_multi_index(
        tuple(i + s.start for i, s in zip(idx, sl_q, strict=True)), a.shape
    )
    return p.astype(np.int64), q.astype(np.int64)


@dataclass
class Setup:
    wm: Bool
    cortex: Bool
    seeds: Bool  # WM voxels 6-adjacent to cortex
    domain: Bool  # cortex | seeds
    edt: Float  # distance to the nearest WM voxel centre (voxels)
    direction: Float  # (..., 3) unit EDT gradient (zero where undefined)
    patch: Int  # patch id at each seed voxel, -1 elsewhere
    npatch: int
    adjacent: Int  # sorted codes a * npatch + b (a < b) of neighbouring patches
    centre: Float  # (npatch, 3) mean position of each patch's seeds
    graph: object  # csr seed graph (edge lengths in voxels)
    rep: Int  # per patch, its centre seed (compact id)
    sflat: Int  # flat index of each seed (compact id order)


def setup(wm: Bool, cortex: Bool, patch: int = 6, direction: Float | None = None) -> Setup:
    """Seeds, domain, distance direction, and root patches."""
    near_ctx = ndi.binary_dilation(cortex, ndi.generate_binary_structure(3, 1))
    seeds = wm & near_ctx
    domain = cortex | seeds
    edt = ndi.distance_transform_edt(~wm).astype(np.float64)
    g = np.stack(np.gradient(ndi.gaussian_filter(edt, 0.7)), -1)
    nrm = np.linalg.norm(g, axis=-1, keepdims=True)
    unit = np.divide(g, nrm, out=np.zeros_like(g), where=nrm > 1e-6)
    if direction is None:
        direction = unit
    # seed graph: 26-neighbours whose outward normals agree
    sflat = np.flatnonzero(seeds.ravel())
    sid = -np.ones(seeds.size, np.int64)
    sid[sflat] = np.arange(sflat.size)
    un = unit.reshape(-1, 3)
    rows, cols, lens = [], [], []
    for off in N13:
        p, q = shifted_pairs(seeds, seeds, off)
        ok = np.sum(un[p] * un[q], -1) > 0.0
        rows.append(sid[p[ok]])
        cols.append(sid[q[ok]])
        lens.append(np.full(ok.sum(), float(np.linalg.norm(off))))
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    ln = np.concatenate(lens)
    graph = coo_matrix((ln, (r, c)), shape=(sflat.size,) * 2).tocsr()
    graph = graph.maximum(graph.T).tocsr()
    rep, comp = geodesic_patches(graph, float(patch))
    npatch = rep.size
    pid = -np.ones(seeds.shape, np.int64)
    pid.ravel()[sflat] = comp
    a, b = comp[r], comp[c]
    diff = a != b
    lo, hi = np.minimum(a[diff], b[diff]), np.maximum(a[diff], b[diff])
    adjacent = np.unique(lo * npatch + hi)
    pos = np.stack(np.unravel_index(sflat, seeds.shape), -1)
    centre = pos[rep].astype(float)
    return Setup(
        wm, cortex, seeds, domain, edt, direction, pid, npatch, adjacent, centre,
        graph, rep, sflat,
    )


def geodesic_patches(graph: object, spacing: float) -> tuple[Int, Int]:
    """Centres chosen greedily in index order, each at least `spacing` along
    the graph from every earlier one, then every node assigned to its nearest
    centre (a geodesic Voronoi partition: each cell is connected). Returns
    (centre node per patch, patch per node)."""
    from scipy.sparse.csgraph import dijkstra

    n = graph.shape[0]  # type: ignore[attr-defined]
    covered = np.zeros(n, bool)
    centres = []
    for i in range(n):
        if covered[i]:
            continue
        centres.append(i)
        d = dijkstra(graph, indices=i, limit=spacing)
        covered |= d < spacing
    cen = np.array(centres, np.int64)
    _, _, src = dijkstra(graph, indices=cen, min_only=True, return_predecessors=True)
    lookup = -np.ones(n, np.int64)
    lookup[cen] = np.arange(cen.size)
    return cen, lookup[src]


def separation(s: Setup, a: Int, b: Int, limit: float = 120.0) -> Float:
    """Geodesic distance along the white surface (the seed graph, voxels)
    between the representative seeds of patches a and b; inf beyond limit
    or when not connected."""
    from scipy.sparse.csgraph import dijkstra

    out = np.full(a.size, np.inf)
    ua, inv = np.unique(a, return_inverse=True)
    for chunk in range(0, ua.size, 256):
        src = ua[chunk : chunk + 256]
        d = dijkstra(s.graph, indices=s.rep[src], limit=limit)
        sel = (inv >= chunk) & (inv < chunk + src.size)
        out[sel] = d[inv[sel] - chunk, s.rep[b[sel]]]
    return out


def run(
    s: Setup,
    mu: Float | None = None,
    sigma: Float | None = None,
    ratio: float = 3.0,
    labels: Int | None = None,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Fronts:
    """The one pass. mu, sigma per patch (None: equal arrival)."""
    m = riemann_from_direction(s.direction, ratio)
    lab = s.patch if labels is None else labels
    if mu is None:
        return march(m, lab, mask=s.domain, spacing=spacing)
    assert sigma is not None
    return march(
        m, lab, mask=s.domain, spacing=spacing, start_time=-mu / sigma, speed=sigma
    )


def distance(f: Fronts, lab: Int, u: Float, mu: Float | None, sigma: Float | None) -> Float:
    """Metric distance d = mu + sigma u for arrival u of label lab."""
    if mu is None or sigma is None:
        return u
    ok = lab >= 0
    out = np.full(u.shape, np.nan)
    out[ok] = mu[lab[ok]] + sigma[lab[ok]] * u[ok]
    return out


@dataclass
class Contact:
    p: Int  # flat index of the voxel on label a's side
    q: Int  # its 6-neighbour on label b's side
    axis: Int  # axis of the step from p to q
    a: Int
    b: Int
    frac: Float  # sub-voxel position from p toward q (0.5 where unknown)
    exact: npt.NDArray[np.bool_]  # frac from both sides' arrivals
    da: Float  # arrival of a at the contact in distance (voxels)
    db: Float
    sep: Float  # separation of the two patches' centres along the white surface


def contacts(
    s: Setup,
    f: Fronts,
    mu: Float | None = None,
    sigma: Float | None = None,
    sep_min: float = 18.0,
) -> Contact:
    """Label boundaries between patches more than sep_min apart along the white
    surface (voxels; 18 = about 3 mm at 0.16 mm), as 6-neighbour voxel pairs in
    cortex, located between the voxels from the two labels' arrivals (the
    other label's arrival extrapolated one step where unknown)."""
    lab = f.label.ravel()
    u = f.arrival.ravel()
    u2 = f.second_arrival.ravel()
    l2 = f.second_label.ravel()
    ctx = s.cortex & (f.label >= 0)
    P, Q, AX = [], [], []
    for ax, off in enumerate(N6):
        p, q = shifted_pairs(ctx, ctx, off)
        k = lab[p] != lab[q]
        P.append(p[k])
        Q.append(q[k])
        AX.append(np.full(k.sum(), ax))
    p, q, ax = np.concatenate(P), np.concatenate(Q), np.concatenate(AX)
    a, b = lab[p], lab[q]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    code, inv = np.unique(lo * s.npatch + hi, return_inverse=True)
    sep_u = separation(s, code // s.npatch, code % s.npatch, limit=4 * sep_min)
    sep = sep_u[inv]
    far = sep > sep_min
    p, q, ax, a, b, sep = p[far], q[far], ax[far], a[far], b[far], sep[far]
    # competition values phi = u_a - u_b (weighted time; same units both sides)
    have_p = l2[p] == b
    have_q = l2[q] == a
    phi_p = u[p] - np.where(have_p, u2[p], np.nan)  # <= 0
    phi_q = np.where(have_q, u2[q], np.nan) - u[q]  # >= 0
    # where the other label's arrival is unknown, extrapolate it one step from
    # that label's own voxel behind (p - e for a, q + e for b); one voxel of
    # distance per step when that voxel is not the label's either
    shape = f.label.shape
    strides = np.array([shape[1] * shape[2], shape[2], 1])
    st = strides[ax]
    coord_p = np.stack(np.unravel_index(p, shape), -1)
    coord_q = np.stack(np.unravel_index(q, shape), -1)
    ok_bp = coord_p[np.arange(p.size), ax] > 0
    ok_bq = coord_q[np.arange(q.size), ax] < np.array(shape)[ax] - 1
    bp = np.where(ok_bp, p - st, p)
    bq = np.where(ok_bq, q + st, q)
    sa = 1.0 if sigma is None else sigma[a]
    sb = 1.0 if sigma is None else sigma[b]
    ga = np.where(ok_bp & (lab[bp] == a), u[p] - u[bp], 1.0 / sa)
    gb = np.where(ok_bq & (lab[bq] == b), u[q] - u[bq], 1.0 / sb)
    ga = np.where(ga > 0, ga, 1.0 / sa)
    gb = np.where(gb > 0, gb, 1.0 / sb)
    phi_p = np.where(have_p, phi_p, u[p] - (u[q] + gb))
    phi_q = np.where(have_q, phi_q, (u[p] + ga) - u[q])
    exact = have_p & have_q
    frac = np.full(p.size, 0.5)
    good = (phi_q - phi_p > 0) & (phi_p <= 0) & (phi_q >= 0)
    frac[good] = -phi_p[good] / (phi_q[good] - phi_p[good])
    # arrival of each label at the contact, in distance: its own arrival plus
    # frac of a step (from the other voxel where known, else extrapolated)
    step_a = np.where(have_q, u2[q] - u[p], ga)
    step_b = np.where(have_p, u2[p] - u[q], gb)
    ma = 0.0 if mu is None else mu[a]
    mb = 0.0 if mu is None else mu[b]
    da = ma + sa * (u[p] + frac * step_a)
    db = mb + sb * (u[q] + (1 - frac) * step_b)
    return Contact(p, q, ax, a, b, frac, exact, da, db, sep)


def contact_points(c: Contact, shape: tuple[int, ...]) -> Float:
    """Sub-voxel positions (n, 3) of the contact pairs."""
    pp = np.stack(np.unravel_index(c.p, shape), -1).astype(float)
    step = np.zeros((c.p.size, 3))
    step[np.arange(c.p.size), c.axis] = 1.0
    out: Float = pp + c.frac[:, None] * step
    return out


def open_thickness(
    s: Setup, fluid: Bool, ratio: float = 3.0, along: bool = False
) -> tuple[Float, Bool]:
    """Thickness evidence per patch from visible tops: the median arc length of
    an unweighted pass at the patch's cortex voxels next to fluid. Patches
    with no top take the value of the nearest patch with one (by centre).
    along: count a top only where the front runs into the fluid: a fluid
    6-neighbour at offset o, the voxel behind (at -o) on the same label with an
    earlier arrival, and the column axis (s.direction, unsigned) within 45
    degrees of o. Cortex that meets fluid sideways, where columns do not end,
    is then not evidence; the front's own flow, not the direction's sign, says
    which way is up (in a thin gap the WM-distance sign points the wrong way).
    Returns (mu per patch, has evidence)."""
    f = run(s, ratio=ratio)
    if along:
        tops = np.zeros(s.cortex.shape, bool)
        lab, arr = f.label, f.arrival
        for ax in range(3):
            n = s.cortex.shape[ax]
            for sg in (1, -1):
                # fluid ahead at +sg along ax, own label behind at -sg
                ahead = np.zeros_like(tops)
                behind_ok = np.zeros_like(tops)
                sl_v = [slice(None)] * 3
                sl_a = [slice(None)] * 3
                sl_b = [slice(None)] * 3
                if sg == 1:
                    sl_v[ax], sl_a[ax], sl_b[ax] = slice(1, n - 1), slice(2, n), slice(0, n - 2)
                else:
                    sl_v[ax], sl_a[ax], sl_b[ax] = slice(1, n - 1), slice(0, n - 2), slice(2, n)
                v, fa, fb = tuple(sl_v), tuple(sl_a), tuple(sl_b)
                ahead[v] = fluid[fa]
                behind_ok[v] = (lab[fb] == lab[v]) & (arr[fb] < arr[v])
                tops |= ahead & behind_ok & (np.abs(s.direction[..., ax]) > np.cos(np.pi / 4))
        tops &= s.cortex
    else:
        tops = s.cortex & ndi.binary_dilation(fluid, ndi.generate_binary_structure(3, 1))
    tops &= f.label >= 0
    lab = f.label[tops]
    # the top lies half a voxel beyond the voxel centre, on average
    val = f.length[tops] + 0.5
    mu = np.full(s.npatch, np.nan)
    order = np.argsort(lab)
    lab_s, val_s = lab[order], val[order]
    starts = np.flatnonzero(np.r_[True, np.diff(lab_s) != 0])
    for i, st in enumerate(starts):
        en = starts[i + 1] if i + 1 < starts.size else lab_s.size
        mu[lab_s[st]] = np.median(val_s[st:en])
    has = np.isfinite(mu)
    from scipy.spatial import cKDTree

    _, j = cKDTree(s.centre[has]).query(s.centre[~has])
    mu[~has] = mu[has][j]
    return mu, has


def pure_voronoi(
    s: Setup, mu: Float, sigma: Float, ratio: float = 3.0
) -> tuple[Int, Float]:
    """Brute-force reference without crystal growth: one unweighted march per
    patch from its own seeds over the whole domain, then each voxel's label is
    argmin_l (d_l - mu_l) / sigma_l. Returns (label, weighted arrival)."""
    m = riemann_from_direction(s.direction, ratio)
    best = np.full(s.domain.shape, np.inf)
    lab = -np.ones(s.domain.shape, np.int64)
    for l in range(s.npatch):
        seed = np.where(s.patch == l, 0, -1).astype(np.int64)
        d = march(m, seed, mask=s.domain).arrival
        w = (d - mu[l]) / sigma[l]
        win = w < best
        best[win] = w[win]
        lab[win] = l
    return lab, best


def score_points(P, u_pts, c, cp, mu_f, sd_f, h):
    """Offsets of the contact from marked points (n, 3; voxels), as in collide2.py: for each point,
    the contact points within 1 mm and within 0.4 mm of the line through it along the column
    direction u_pts; the median offset along that line (mm), + when the contact lies beyond the point
    toward the side whose mu_f (the field's belief, per label) is smaller. Per side, the arrival S at the
    point (the contact's arrival plus or minus the offset) and mu_f - S. h: mm per voxel."""
    rows = []
    for k_, Pk in enumerate(P):
        u = u_pts[k_] / (np.linalg.norm(u_pts[k_]) + 1e-12)
        d = cp - Pk
        along = d @ u
        perp = np.linalg.norm(d - along[:, None] * u, axis=1)
        near = (np.linalg.norm(d, axis=1) * h < 1.0) & (perp * h < 0.4)
        row = dict(k=k_ + 1, offset=np.nan, n=int(near.sum()), thin_mu_minus_S=np.nan, thick_mu_minus_S=np.nan,
                   thin_sd=np.nan, thick_sd=np.nan, S_thin=np.nan, S_thick=np.nan)
        if near.any():
            i = np.flatnonzero(near)
            a, b = c.a[i], c.b[i]
            step = np.zeros((i.size, 3)); step[np.arange(i.size), c.axis[i]] = 1.0  # from a's side to b's
            b_thin = mu_f[b] < mu_f[a]
            to_thin = np.where(b_thin[:, None], step, -step)
            sg = np.sign(np.sum(to_thin @ u))
            sg = 1.0 if sg == 0 else sg
            o = along[i] * sg * h  # + : the contact lies beyond the point, toward the thinner side
            thin = np.where(b_thin, b, a); thick = np.where(b_thin, a, b)
            d_thin = np.where(b_thin, c.db[i], c.da[i]) * h
            d_thick = np.where(b_thin, c.da[i], c.db[i]) * h
            S_thin = d_thin + o  # the point lies o short of the contact on the thick side
            S_thick = d_thick - o
            row.update(offset=float(np.median(o)),
                       thin_mu_minus_S=float(np.median(mu_f[thin] * h - S_thin)),
                       thick_mu_minus_S=float(np.median(mu_f[thick] * h - S_thick)),
                       thin_sd=float(np.median(sd_f[thin] * h)), thick_sd=float(np.median(sd_f[thick] * h)),
                       S_thin=float(np.median(S_thin)), S_thick=float(np.median(S_thick)),
                       sep_mm=float(np.median(np.where(np.isinf(c.sep[i]), 999, c.sep[i] * h))))
        rows.append(row)
    return rows


def score_points_sides(s, f, P, u_pts, c, cp, mu_f, sd_f, h, reach_mm=2.5):
    """As score_points, but with the two sides found locally at each point instead of a separation
    threshold (contacts should then be taken with sep_min=0): the labels at the two ends of the
    column line through the point (reach_mm either way, last cortex voxel on each side) are the two
    fronts; every pair near the line whose labels lie on opposite sides (each label nearer, along the
    white surface, to one end's label than to the other's) is a crossing. Rows as score_points, plus
    the two end labels' separation (mm)."""
    from scipy.sparse.csgraph import dijkstra

    rows = []
    steps = np.arange(-reach_mm, reach_mm + 1e-9, 0.05) / h
    shape = np.array(s.cortex.shape)
    for k_, Pk in enumerate(P):
        u = u_pts[k_] / (np.linalg.norm(u_pts[k_]) + 1e-12)
        line = np.round(Pk + steps[:, None] * u).astype(int)
        inside = np.all((line >= 0) & (line < shape), 1)
        lab = np.full(steps.size, -1)
        lab[inside] = f.label[tuple(line[inside].T)]
        ctx_ok = np.zeros(steps.size, bool)
        ctx_ok[inside] = s.cortex[tuple(line[inside].T)]
        neg = np.flatnonzero(ctx_ok & (lab >= 0) & (steps < 0))
        pos = np.flatnonzero(ctx_ok & (lab >= 0) & (steps > 0))
        empty = dict(k=k_ + 1, offset=np.nan, n=0, thin_mu_minus_S=np.nan, thick_mu_minus_S=np.nan,
                     thin_sd=np.nan, thick_sd=np.nan, S_thin=np.nan, S_thick=np.nan, end_sep_mm=np.nan)
        if neg.size == 0 or pos.size == 0:
            rows.append(empty); continue
        lm, lp = lab[neg[0]], lab[pos[-1]]  # the deepest cortex on each side
        d = dijkstra(s.graph, indices=s.rep[[lm, lp]])
        dm, dp = d[0][s.rep], d[1][s.rep]  # per patch
        side = np.where(dm < dp, -1, 1)  # -1: the front at -u, +1: the front at +u
        dd = cp - Pk
        along = dd @ u
        perp = np.linalg.norm(dd - along[:, None] * u, axis=1)
        cross = side[c.a] != side[c.b]
        near = (np.linalg.norm(dd, axis=1) * h < 1.0) & (perp * h < 0.4) & cross
        empty["end_sep_mm"] = float(dm[lp] * h)
        # the nearest crossing in any direction (unsigned): separates a line oblique to the
        # contact from a boundary that is really displaced
        empty["nearest_mm"] = float(np.min(np.linalg.norm(dd[cross], axis=1)) * h) if cross.any() else np.nan
        if not near.any():
            rows.append(empty); continue
        i = np.flatnonzero(near)
        a, b = c.a[i], c.b[i]
        # the two fronts' beliefs: the end labels (the columns through the point)
        thin_end = lm if mu_f[lm] < mu_f[lp] else lp
        sg = -1.0 if thin_end == lm else 1.0  # direction (along u) toward the thinner side
        o = along[i] * sg * h
        a_thin = side[a] == (-1 if thin_end == lm else 1)
        thin = np.where(a_thin, a, b); thick = np.where(a_thin, b, a)
        d_thin = np.where(a_thin, c.da[i], c.db[i]) * h
        d_thick = np.where(a_thin, c.db[i], c.da[i]) * h
        S_thin, S_thick = d_thin + o, d_thick - o
        rows.append(dict(k=k_ + 1, offset=float(np.median(o)), n=int(i.size),
                         thin_mu_minus_S=float(np.median(mu_f[thin] * h - S_thin)),
                         thick_mu_minus_S=float(np.median(mu_f[thick] * h - S_thick)),
                         thin_sd=float(np.median(sd_f[thin] * h)), thick_sd=float(np.median(sd_f[thick] * h)),
                         S_thin=float(np.median(S_thin)), S_thick=float(np.median(S_thick)),
                         end_sep_mm=float(dm[lp] * h), nearest_mm=empty["nearest_mm"]))
    return rows
