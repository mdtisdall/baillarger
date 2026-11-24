"""Level-set filling from the white surface (reports/Level-set filling test plan.md).

Cortex fills outward from the white surface along the data's layer normal. Every root advances in a
common normalized time tau at a rate set by its own budget:
- length form: ds/dtau = mu(R), so tau = s / mu along a path;
- volume form: ds/dtau = v(R) / a(x), so tau = w / v with w = int a ds.
The static form is an ordered upwind march in tau (a heap), semi-Lagrangian: a candidate for a voxel x
traces back along -d_x to the face of the upwind cell (p) and interpolates (trilinear) tau, R, s, w, ln a, div d and the
budget from accepted voxels whose own direction agrees with d_x (d_y . d_x > 0). That sign agreement
is what keeps the two sides of a seam apart. The direction d_x is the data normal n_x, signed to agree
with the side it is reached from, blended toward the front's own normal with weight 1 - c (c the
coherence). The front's normal is the geometric one, the least-squares gradient of the arc length s over
that side's accepted neighbours, not of tau: tau carries the budgets, and blending toward its gradient
let the budgets bend the paths where c < 1 (test_low_coherence_paths_do_not_depend_on_budgets).

Carried per voxel: tau, R (root, a point on the white surface), s (arc length), w (volume per unit of
white area), a (cross-section ratio, from d(ln a)/ds = div d), the divergence D, the budget b (carried
with R: b(x) is the budget at R(x)), d (signed direction) and the blend weight 1 - c.

Start: a narrow band (WM voxels 26-adjacent to cortex, and cortex voxels less than one voxel from the
white surface) at its exact signed distance phi to the white surface: s = phi, ln a = phi D,
w = phi + phi^2 D / 2, tau = s / b or w / b, R = the foot point, d = the surface normal there.

Seams: 6-neighbour pairs of claimed cortex whose directions do not agree (d_p . d_q <= 0); located between the
voxels where the two sides' tau, extrapolated along the pair, are equal.
Tops: claimed cortex with a fluid 6-neighbour within 45 degrees of d; s and w at the exit face.

Units: voxels (isotropic grid assumed; the real boxes are 0.16 mm within 1 %). Everything is 3D.
"""

import heapq
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numba import njit
from scipy import ndimage as ndi

from baillarger.topology import is_simple

type Bool = npt.NDArray[np.bool_]
type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]

LENGTH, VOLUME = 0, 1
FREE, CLAIMABLE, ACCEPTED, BLOCKED = 0, 1, 2, 3
A_MIN = (
    0.05  # floor on the cross-section ratio a (focal points of concave white surface)
)

_OFF26 = np.array(
    [
        (a, b, c)
        for a in (-1, 0, 1)
        for b in (-1, 0, 1)
        for c in (-1, 0, 1)
        if (a, b, c) != (0, 0, 0)
    ],
    np.int64,
)


@njit(cache=True)
def _unit(v):
    nn = np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if nn < 1e-12:
        return v * 0.0
    return v / nn


@njit(cache=True)
def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit(cache=True)
def _divergence(x, dx, shape, state, n, coh):
    """div d at voxel x (flat) for direction dx: central differences of the data normal field aligned
    with dx (each neighbour's n signed to agree with dx and blended toward dx with weight 1 - c). A
    neighbour is usable if it is tissue in the march (claimable, accepted or blocked) and its aligned
    direction is within 45 degrees of dx; otherwise the difference is one-sided with dx. Nothing here
    depends on which voxels are accepted yet: using an accepted neighbour's own d made the divergence,
    and so a and w, depend on the march order (and the order on the budgets: test
    test_tops_do_not_depend_on_budgets)."""
    ny, nz = shape[1], shape[2]
    st = (ny * nz, nz, 1)
    c45 = 0.7071067811865476
    div = 0.0
    for k in range(3):
        u = np.zeros(2)
        ok = np.zeros(2, np.bool_)
        for j in range(2):
            y = x + st[k] if j == 0 else x - st[k]
            s = state[y]
            if s == ACCEPTED or s == CLAIMABLE or s == BLOCKED:
                v = n[y] if _dot(n[y], dx) >= 0 else -n[y]
                v = _unit(coh[y] * v + (1.0 - coh[y]) * dx)
                if _dot(v, dx) >= c45:
                    u[j] = v[k]
                    ok[j] = True
        if ok[0] and ok[1]:
            div += 0.5 * (u[0] - u[1])
        elif ok[0]:
            div += u[0] - dx[k]
        elif ok[1]:
            div += dx[k] - u[1]
    return div


@njit(cache=True)
def _upwind(xi, xj, xk, dx, shape, state, d):
    """The trace-back point on the face of the upwind cell (h = 1 / max |d_i|, h in [1, sqrt 3], so x
    itself is never a corner: with h = 1 it always was, its weight dropped and the rest renormalized,
    and s(p) read low by up to 0.1 voxel per step; test_crown_level_sets). Returns h, the 8 corners'
    flat indices and trilinear weights, which corners are usable (accepted, d agreeing with dx), and
    the number of pending corners (tissue not yet accepted, with weight > 0)."""
    ny, nz = shape[1], shape[2]
    am = 0
    for i in range(1, 3):
        if abs(dx[i]) > abs(dx[am]):
            am = i
    h = 1.0 / abs(dx[am])
    p0, p1, p2 = xi - h * dx[0], xj - h * dx[1], xk - h * dx[2]
    if am == 0:
        p0 = xi - (1.0 if dx[0] > 0 else -1.0)
    elif am == 1:
        p1 = xj - (1.0 if dx[1] > 0 else -1.0)
    else:
        p2 = xk - (1.0 if dx[2] > 0 else -1.0)
    i0, j0, k0 = int(np.floor(p0)), int(np.floor(p1)), int(np.floor(p2))
    f0, f1, f2 = p0 - i0, p1 - j0, p2 - k0
    idx = np.full(8, -1, np.int64)
    wts = np.zeros(8)
    use = np.zeros(8, np.bool_)
    pending = 0
    m = 0
    for a in range(2):
        wa = f0 if a == 1 else 1.0 - f0
        for bb in range(2):
            wb = f1 if bb == 1 else 1.0 - f1
            for c in range(2):
                wc = f2 if c == 1 else 1.0 - f2
                wt = wa * wb * wc
                y = (i0 + a) * ny * nz + (j0 + bb) * nz + (k0 + c)
                if wt > 1e-12 and 0 <= y < state.size:
                    idx[m] = y
                    wts[m] = wt
                    if state[y] == ACCEPTED:
                        use[m] = _dot(d[y], dx) > 0
                    elif state[y] == CLAIMABLE or state[y] == BLOCKED:
                        pending += 1
                m += 1
    return h, idx, wts, use, pending


@njit(cache=True)
def _candidate(
    x,
    shape,
    state,
    n,
    coh,
    tau,
    s,
    w,
    la,
    D,
    b,
    R,
    d,
    mode,
    lamin,
    out_f,
    out_v,
    excl,
    by_s=False,
):
    """Best candidate for x over its two sides by the semi-Lagrangian step, leaving out the side whose
    direction agrees with excl (zeros: none). out_f: tau, s, w, la, D, b, blend, capped; out_v: R (0:3)
    and d (3:6). Returns tau (inf if none). by_s: choose between the sides by s, not tau (phase 1:
    nothing there may depend on the budgets).

    Where c < 1 the direction blends toward the front's own normal: the least-squares gradient of the
    arc length s over the side's accepted neighbours (the geometric front), not of tau, which carries
    the budgets (test_low_coherence_paths_do_not_depend_on_budgets)."""
    ny, nz = shape[1], shape[2]
    xi = x // (ny * nz)
    xj = (x // nz) % ny
    xk = x % nz
    best = np.inf
    for sg in (1.0, -1.0):
        nxs = sg * n[x]
        if _dot(nxs, excl) > 0:
            continue
        M = np.zeros((4, 4))
        rhs = np.zeros(4)
        dsum = np.zeros(3)
        cnt = 0
        for a in range(-1, 2):
            for bb in range(-1, 2):
                for c in range(-1, 2):
                    if a == 0 and bb == 0 and c == 0:
                        continue
                    y = x + a * ny * nz + bb * nz + c
                    if state[y] != ACCEPTED or _dot(d[y], nxs) <= 0:
                        continue
                    cnt += 1
                    dsum += d[y]
                    o = np.array([1.0, a, bb, c])
                    for i in range(4):
                        rhs[i] += o[i] * s[y]
                        for j in range(4):
                            M[i, j] += o[i] * o[j]
        if cnt == 0:
            continue
        cx = coh[x]
        if cx < 1.0:
            g = dsum
            if cnt >= 4:
                Mr = M + 1e-9 * np.eye(4)
                if abs(np.linalg.det(Mr)) > 1e-6:
                    sol = np.linalg.solve(Mr, rhs)
                    gg = sol[1:]
                    if (
                        _dot(gg, nxs) > 0
                    ):  # the front's normal points the way this side moves
                        g = gg
            dx = _unit(cx * nxs + (1.0 - cx) * _unit(g))
        else:
            dx = nxs.copy()
        if _dot(dx, dx) < 0.5:
            continue
        h, idx, wts, use, pend = _upwind(xi, xj, xk, dx, shape, state, d)
        W = 0.0
        tp = 0.0
        sp = 0.0
        wp = 0.0
        lap = 0.0
        Dp = 0.0
        bp = 0.0
        Rp = np.zeros(3)
        for m in range(8):
            if not use[m]:
                continue
            y = idx[m]
            wt = wts[m]
            W += wt
            tp += wt * tau[y]
            sp += wt * s[y]
            wp += wt * w[y]
            lap += wt * la[y]
            Dp += wt * D[y]
            bp += wt * b[y]
            Rp += wt * R[y]
        if W <= 1e-9:
            continue
        tp /= W
        sp /= W
        wp /= W
        lap /= W
        Dp /= W
        bp /= W
        Rp /= W
        Dx = _divergence(x, dx, shape, state, n, coh)
        lax = lap + 0.5 * h * (Dp + Dx)
        capped = 0.0
        if lax < lamin:
            lax = lamin
            capped = 1.0
        ap, ax = np.exp(lap), np.exp(lax)
        sx = sp + h
        wx = wp + 0.5 * h * (ap + ax)
        if mode == LENGTH:
            tx = tp + h / bp
        else:
            tx = tp + (wx - wp) / bp
        key = sx if by_s else tx
        if key < best:
            best = key
            out_f[0] = tx
            out_f[1] = sx
            out_f[2] = wx
            out_f[3] = lax
            out_f[4] = Dx
            out_f[5] = bp
            out_f[6] = 1.0 - cx
            out_f[7] = capped
            out_v[0:3] = Rp
            out_v[3:6] = dx
    if by_s and best < np.inf:
        return out_f[0]
    return best


@njit(cache=True)
def _phase2(
    x, shape, state, n, coh, tau, s, w, la, D, b, R, d, mode, lamin, gf, gv, of, ov
):
    """Phase 2: the candidate of x's phase-1 side is phase 1's own values (so nothing there depends on
    the budgets' order), offered when x has an accepted neighbour on that side; the other side's
    candidate is the semi-Lagrangian step. Returns the smaller, with of / ov set to it."""
    ny, nz = shape[1], shape[2]
    gd = gv[x, 3:6].copy()
    if _dot(gd, gd) < 0.5:  # phase 1 did not reach x
        return _candidate(
            x,
            shape,
            state,
            n,
            coh,
            tau,
            s,
            w,
            la,
            D,
            b,
            R,
            d,
            mode,
            lamin,
            of,
            ov,
            np.zeros(3),
        )
    tx = _candidate(
        x, shape, state, n, coh, tau, s, w, la, D, b, R, d, mode, lamin, of, ov, gd
    )
    found = False
    for a in range(-1, 2):
        for bb in range(-1, 2):
            for c in range(-1, 2):
                if a == 0 and bb == 0 and c == 0:
                    continue
                y = x + a * ny * nz + bb * nz + c
                if state[y] == ACCEPTED and _dot(d[y], gd) > 0:
                    found = True
    if not found:
        return tx
    if gf[x, 0] <= tx:
        of[:] = gf[x]
        ov[:] = gv[x]
        return gf[x, 0]
    return tx


@njit(cache=True)
def _march(
    shape,
    state,
    obj,
    n,
    coh,
    tau,
    s,
    w,
    la,
    D,
    b,
    R,
    d,
    blend,
    capped,
    mode,
    lamin,
    tau_stop,
    topo,
    order_s,
    use_geo,
    gf,
    gv,
):
    """order_s: order the heap by s (phase 1, the geometry) instead of tau. use_geo: phase 2, with
    phase 1's candidate values gf (tau, s, w, la, D, b, blend, capped) and gv (R, d) per voxel."""
    N = state.size
    ny, nz = shape[1], shape[2]
    ctau = np.full(N, np.inf)
    cf = np.zeros((N, 8))
    cv = np.zeros((N, 6))
    of = np.zeros(8)
    ov = np.zeros(6)
    heap = [(np.inf, np.int64(-1))]
    heap.pop()
    offs = np.empty(26, np.int64)
    m = 0
    for a in range(-1, 2):
        for bb in range(-1, 2):
            for c in range(-1, 2):
                if a == 0 and bb == 0 and c == 0:
                    continue
                offs[m] = a * ny * nz + bb * nz + c
                m += 1
    # band divergence and its a, w (set by the caller: tau, s, b, d; la and w here)
    for x in range(N):
        if state[x] == ACCEPTED:
            D[x] = _divergence(x, d[x], shape, state, n, coh)
    for x in range(N):
        if state[x] == ACCEPTED:
            phi = s[x]
            la[x] = phi * D[x]
            w[x] = phi + 0.5 * phi * phi * D[x]
            tau[x] = phi / b[x] if mode == LENGTH else w[x] / b[x]
    for x in range(N):
        if state[x] == ACCEPTED:
            for o in offs:
                y = x + o
                if state[y] == CLAIMABLE and ctau[y] == np.inf:
                    if use_geo:
                        t = _phase2(
                            y,
                            shape,
                            state,
                            n,
                            coh,
                            tau,
                            s,
                            w,
                            la,
                            D,
                            b,
                            R,
                            d,
                            mode,
                            lamin,
                            gf,
                            gv,
                            of,
                            ov,
                        )
                    else:
                        t = _candidate(
                            y,
                            shape,
                            state,
                            n,
                            coh,
                            tau,
                            s,
                            w,
                            la,
                            D,
                            b,
                            R,
                            d,
                            mode,
                            lamin,
                            of,
                            ov,
                            np.zeros(3),
                            order_s,
                        )
                    if t < np.inf:
                        ctau[y] = t
                        cf[y] = of
                        cv[y] = ov
                        heapq.heappush(heap, (of[1] if order_s else t, y))
    nblock = 0
    while len(heap) > 0:
        key, x = heapq.heappop(heap)
        if (state[x] != CLAIMABLE and state[x] != BLOCKED) or key != (
            cf[x, 1] if order_s else ctau[x]
        ):
            continue
        if ctau[x] > tau_stop:
            continue
        if topo:
            xi = x // (ny * nz)
            xj = (x // nz) % ny
            xk = x % nz
            if not is_simple(obj, xi, xj, xk, 26):
                if state[x] != BLOCKED:
                    nblock += 1
                state[x] = BLOCKED
                continue
        state[x] = ACCEPTED
        obj[x // (ny * nz), (x // nz) % ny, x % nz] = 1
        tau[x] = cf[x, 0]
        s[x] = cf[x, 1]
        w[x] = cf[x, 2]
        la[x] = cf[x, 3]
        D[x] = cf[x, 4]
        b[x] = cf[x, 5]
        blend[x] = cf[x, 6]
        capped[x] = cf[x, 7] > 0
        R[x] = cv[x, 0:3]
        d[x] = cv[x, 3:6]
        for o in offs:
            y = x + o
            if state[y] == CLAIMABLE or state[y] == BLOCKED:
                if use_geo:
                    tn = _phase2(
                        y,
                        shape,
                        state,
                        n,
                        coh,
                        tau,
                        s,
                        w,
                        la,
                        D,
                        b,
                        R,
                        d,
                        mode,
                        lamin,
                        gf,
                        gv,
                        of,
                        ov,
                    )
                else:
                    tn = _candidate(
                        y,
                        shape,
                        state,
                        n,
                        coh,
                        tau,
                        s,
                        w,
                        la,
                        D,
                        b,
                        R,
                        d,
                        mode,
                        lamin,
                        of,
                        ov,
                        np.zeros(3),
                        order_s,
                    )
                if tn < np.inf and (
                    tn != ctau[y] or state[y] == BLOCKED or of[1] != cf[y, 1]
                ):
                    ctau[y] = tn
                    cf[y] = of
                    cv[y] = ov
                    heapq.heappush(heap, (of[1] if order_s else tn, y))
    return nblock, 0


@dataclass
class Fill:
    shape: tuple[int, int, int]
    mode: int
    state: npt.NDArray[
        np.int8
    ]  # FREE, CLAIMABLE (never reached), ACCEPTED, BLOCKED (topology)
    tau: Float
    s: Float
    w: Float
    a: Float
    D: Float
    b: Float
    R: Float  # (..., 3)
    d: Float  # (..., 3)
    blend: Float
    capped: Bool
    band: Bool
    cortex: Bool
    nblock: int
    nrelaxed: int  # unused (kept for the field order)


def march(
    cortex: Bool,
    wm: Bool,
    band: Bool,
    phi: Float,
    foot: Float,
    normal: Float,
    budget: Float,
    n: Float,
    coh: Float | None = None,
    mode: int = LENGTH,
    tau_stop: float = np.inf,
    topology: bool = False,
    a_min: float = A_MIN,
) -> Fill:
    """The march. cortex: claimable voxels. band: the start (WM and cortex voxels), with phi (signed
    distance to the white surface, > 0 on the cortex side), foot (the foot point, (..., 3) voxel
    coordinates), normal (the outward surface normal at the foot) and budget (mu or v at the foot),
    each read at band voxels only. n: the data normal (..., 3), unsigned; coh: its coherence (1 if
    None). wm: the WM label (for the topology test's object)."""
    shape = cortex.shape
    P = 1
    ps = tuple(m + 2 * P for m in shape)

    def pad(x, fill=0):
        return np.pad(x, [(P, P)] * 3 + [(0, 0)] * (x.ndim - 3), constant_values=fill)

    N = int(np.prod(ps))
    nn = pad(n.astype(float)).reshape(-1, 3).copy()
    cc = np.ones(N) if coh is None else pad(coh.astype(float), 1.0).ravel().copy()
    bm = pad(band).ravel()

    def one(order_s, gf, gv):
        state = np.zeros(ps, np.int8)
        state[P:-P, P:-P, P:-P][cortex] = CLAIMABLE
        state[P:-P, P:-P, P:-P][band] = ACCEPTED
        obj = pad((wm | band).astype(np.uint8))
        o = dict(
            tau=np.full(N, np.nan),
            s=np.full(N, np.nan),
            w=np.full(N, np.nan),
            la=np.full(N, np.nan),
            D=np.full(N, np.nan),
            b=np.full(N, np.nan),
            R=np.full((N, 3), np.nan),
            d=np.zeros((N, 3)),
            blend=np.zeros(N),
            capped=np.zeros(N, np.bool_),
        )
        o["s"][bm] = pad(phi.astype(float)).ravel()[bm]
        o["b"][bm] = pad(budget.astype(float)).ravel()[bm]
        o["R"][bm] = pad(foot.astype(float)).reshape(-1, 3)[bm] + P
        o["d"][bm] = pad(normal.astype(float)).reshape(-1, 3)[bm]
        nn[bm] = o["d"][
            bm
        ]  # in the start band the layer normal is the surface normal (no data normal in WM)
        st = state.ravel()
        use = gf is not None
        nb, _ = _march(
            np.array(ps, np.int64),
            st,
            obj,
            nn,
            cc,
            o["tau"],
            o["s"],
            o["w"],
            o["la"],
            o["D"],
            o["b"],
            o["R"],
            o["d"],
            o["blend"],
            o["capped"],
            mode,
            float(np.log(a_min)),
            float(np.inf if order_s else tau_stop),
            bool(topology),
            order_s,
            use,
            gf if use else np.zeros((1, 8)),
            gv if use else np.zeros((1, 6)),
        )
        o["state"] = st
        o["nblock"] = nb
        return o

    # phase 1, the geometry: ordered by s, so d, R, s, w and a do not depend on the budgets; phase 2, the
    # competition: ordered by tau, each voxel's phase-1 side offered with phase 1's values, the other
    # side extended past the equal-arrival boundary by the semi-Lagrangian step (main plan, The fronts
    # as PDE problems, piece 3). Ordering a single march by tau let the budgets change the paths where
    # the direction is noisy (test_low_coherence_paths_do_not_depend_on_budgets).
    g = one(True, None, None)
    gf = np.stack(
        [
            g["tau"],
            g["s"],
            g["w"],
            g["la"],
            g["D"],
            g["b"],
            g["blend"],
            g["capped"].astype(float),
        ],
        1,
    )
    gv = np.concatenate([g["R"], g["d"]], 1)
    gf[~np.isfinite(gf)] = 0.0
    gv[~np.isfinite(gv)] = 0.0
    gv[g["state"] != ACCEPTED, 3:] = 0.0
    o = one(False, np.ascontiguousarray(gf), np.ascontiguousarray(gv))
    state, tau, s, w, la, D, b, R, d, blend, capped = (
        o[k]
        for k in ("state", "tau", "s", "w", "la", "D", "b", "R", "d", "blend", "capped")
    )
    nblock = o["nblock"]
    # voxels whose phase-2 owner differs from phase 1's (the budgets moved the seam over them)
    moved = (
        (o["state"] == ACCEPTED)
        & (g["state"] == ACCEPTED)
        & (np.sum(o["d"] * g["d"], 1) <= 0)
    )
    nrelaxed = 0
    state = state.reshape(ps)

    def crop(x, k=None):
        x = x.reshape(ps + ((k,) if k else ()))
        return x[P:-P, P:-P, P:-P].copy()

    fill = Fill(
        shape,
        mode,
        crop(state),
        crop(tau),
        crop(s),
        crop(w),
        np.exp(crop(la)),
        crop(D),
        crop(b),
        crop(R, 3) - P,
        crop(d, 3),
        crop(blend),
        crop(capped),
        band.copy(),
        cortex.copy(),
        int(nblock),
        int(nrelaxed),
    )
    fill.moved = crop(moved)
    return fill


# ------------------------------------------------------------------ after the march


def claimed(f: Fill) -> Bool:
    return f.state == ACCEPTED


def rate_inv(f: Fill) -> Float:
    """dtau/ds at each claimed voxel: 1 / mu (length) or a / v (volume)."""
    return 1.0 / f.b if f.mode == LENGTH else f.a / f.b


@dataclass
class Seam:
    p: Int  # flat index, side A
    q: Int  # flat index, side B (q = p + e_axis)
    axis: Int
    t: Float  # position from p toward q, where the extrapolated taus agree
    clipped: Bool  # the crossing fell outside the pair (t clipped to [0, 1]) or was undefined
    pos: Float  # (n, 3) voxel coordinates
    tau_a: Float
    tau_b: Float
    s_a: Float
    s_b: Float
    w_a: Float
    w_b: Float
    angle: Float  # degrees between the two arriving directions
    b_a: Float
    b_b: Float
    R_a: Float
    R_b: Float


def seams(f: Fill, cortex_only: bool = True) -> Seam:
    """6-neighbour pairs of claimed voxels whose directions do not agree (d_p . d_q <= 0), located
    between the voxels."""
    ok = claimed(f) & (f.cortex if cortex_only else True)
    shape = f.shape
    P, Q, AX = [], [], []
    dd = f.d.reshape(-1, 3)
    for ax in range(3):
        sl_p = [slice(None)] * 3
        sl_q = [slice(None)] * 3
        sl_p[ax] = slice(0, shape[ax] - 1)
        sl_q[ax] = slice(1, shape[ax])
        both = ok[tuple(sl_p)] & ok[tuple(sl_q)]
        idx = np.nonzero(both)
        p = np.ravel_multi_index(idx, shape)
        q = p + int(np.prod(shape[ax + 1 :]))
        # the seam is where the march's own sign agreement (d_x . d_y > 0) refuses to interpolate: d . d <= 0,
        # so fronts meeting at exactly 90 degrees count (with < 0 they were missed: test_wedge_seam_at_90)
        opp = np.sum(dd[p] * dd[q], 1) <= 0
        P.append(p[opp])
        Q.append(q[opp])
        AX.append(np.full(opp.sum(), ax))
    p, q, ax = np.concatenate(P), np.concatenate(Q), np.concatenate(AX)
    e = np.zeros((p.size, 3))
    e[np.arange(p.size), ax] = 1.0
    ri = rate_inv(f).ravel()
    tau, s, w, a = f.tau.ravel(), f.s.ravel(), f.w.ravel(), f.a.ravel()
    cp = np.sum(dd[p] * e, 1)  # path step along +e on side A
    cq = -np.sum(dd[q] * e, 1)  # along -e on side B
    gp = cp * ri[p]
    gq = cq * ri[q]
    den = gp + gq
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (tau[q] + gq - tau[p]) / den
    bad = ~np.isfinite(t) | (den <= 0)
    t = np.where(bad, 0.5, t)
    clipped = bad | (t < 0) | (t > 1)
    t = np.clip(t, 0, 1)
    pos = np.stack(np.unravel_index(p, shape), -1) + t[:, None] * e
    ang = np.degrees(
        np.arccos(np.clip(-np.sum(dd[p] * dd[q], 1), -1, 1))
    )  # 0 = head-on
    Rf = f.R.reshape(-1, 3)
    return Seam(
        p,
        q,
        ax,
        t,
        clipped,
        pos,
        tau[p] + t * gp,
        tau[q] + (1 - t) * gq,
        s[p] + t * cp,
        s[q] + (1 - t) * cq,
        w[p] + t * cp * a[p],
        w[q] + (1 - t) * cq * a[q],
        180.0 - ang,
        f.b.ravel()[p],
        f.b.ravel()[q],
        Rf[p],
        Rf[q],
    )


def seam_class(sm: Seam, lo: float = 0.8, hi: float = 1.25) -> npt.NDArray[np.int8]:
    """By tau on the two sides (as finding 27's d / mu classes): 0 contact (both lo..hi), 1 fold
    (both < lo: met before using the budget), 2 overlap (both > hi), 3 mixed."""
    a, b = sm.tau_a, sm.tau_b
    c = np.full(a.size, 3, np.int8)
    c[(a >= lo) & (a <= hi) & (b >= lo) & (b <= hi)] = 0
    c[(a < lo) & (b < lo)] = 1
    c[(a > hi) & (b > hi)] = 2
    return c


@dataclass
class Tops:
    idx: Int  # flat index of the top voxel
    R: Float
    s: Float  # arc length at the exit face
    w: Float
    step: Int  # (n, 3) step into the fluid


def tops(f: Fill, fluid: Bool, exclude: Bool | None = None) -> Tops:
    """Visible tops: claimed cortex with a fluid 6-neighbour o and d . o >= cos 45; s and w at the
    exit face (0.5 / (d . o) beyond the centre along d; the nearest qualifying face). exclude: voxels
    that are not evidence (near faces or missing tissue)."""
    ok = claimed(f) & f.cortex
    if exclude is not None:
        ok &= ~exclude
    best = np.full(f.shape, np.inf)
    stp = np.zeros(f.shape + (3,), np.int64)
    c45 = np.cos(np.pi / 4)
    for ax in range(3):
        for sg in (1, -1):
            o = np.zeros(3, np.int64)
            o[ax] = sg
            ahead = np.zeros(f.shape, bool)
            sl_v = [slice(None)] * 3
            sl_a = [slice(None)] * 3
            m = f.shape[ax]
            if sg == 1:
                sl_v[ax], sl_a[ax] = slice(0, m - 1), slice(1, m)
            else:
                sl_v[ax], sl_a[ax] = slice(1, m), slice(0, m - 1)
            ahead[tuple(sl_v)] = fluid[tuple(sl_a)]
            cosv = sg * f.d[..., ax]
            q = ok & ahead & (cosv >= c45 - 1e-9)
            dist = np.where(q, 0.5 / np.maximum(cosv, 1e-9), np.inf)
            upd = dist < best
            best[upd] = dist[upd]
            stp[upd] = o
    t = np.isfinite(best)
    idx = np.flatnonzero(t.ravel())
    ds = best.ravel()[idx]
    return Tops(
        idx,
        f.R.reshape(-1, 3)[idx],
        f.s.ravel()[idx] + ds,
        f.w.ravel()[idx] + f.a.ravel()[idx] * ds,
        stp.reshape(-1, 3)[idx],
    )


def a_from_R(f: Fill) -> Float:
    """Cross-section ratio from the root map: 1 / (area scaling of dR on the plane perpendicular to d),
    with central differences of R over 6-neighbours on the same side (one-sided where one is missing)."""
    ok = claimed(f)
    R, d = f.R, f.d
    G = np.full(f.shape + (3, 3), np.nan)  # G[..., :, k] = dR / dx_k
    for k in range(3):
        fwd = np.full(f.shape + (3,), np.nan)
        bwd = np.full(f.shape + (3,), np.nan)
        m = f.shape[k]
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[k], sl1[k] = slice(0, m - 1), slice(1, m)
        s0, s1 = tuple(sl0), tuple(sl1)
        agree = ok[s0] & ok[s1] & (np.sum(d[s0] * d[s1], -1) > 0)
        diff = np.where(agree[..., None], R[s1] - R[s0], np.nan)
        fwd[s0] = diff
        bwd[s1] = diff
        both = np.isfinite(fwd[..., 0]) & np.isfinite(bwd[..., 0])
        g = np.where(
            both[..., None], 0.5 * (fwd + bwd), np.where(np.isfinite(fwd), fwd, bwd)
        )
        G[..., :, k] = g
    # basis of the plane perpendicular to d
    ref = np.where(
        np.abs(d[..., :1]) < 0.9, np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    )
    e1 = np.cross(d, ref)
    e1 /= np.linalg.norm(e1, axis=-1, keepdims=True) + 1e-12
    e2 = np.cross(d, e1)
    j1 = np.einsum("...ij,...j->...i", G, e1)
    j2 = np.einsum("...ij,...j->...i", G, e2)
    J = np.linalg.norm(np.cross(j1, j2), axis=-1)
    out = np.where(
        ok & np.isfinite(J) & (J > 1e-9), 1.0 / np.where(J > 1e-9, J, 1.0), np.nan
    )
    return out


def coverage(
    f: Fill,
    wm: Bool,
    fluid: Bool,
    face: Bool | None = None,
    missing: Bool | None = None,
) -> dict:
    """Coverage of the cortex and the unclaimed connected pieces (26-connected) by cause: no
    connection to the WM; blocked by the topology test; touching a box face or missing tissue (cut
    off); otherwise left by sign agreement."""
    ctx = f.cortex
    got = claimed(f) & ctx
    un = ctx & ~got
    lab, npc = ndi.label(un, np.ones((3, 3, 3)))
    out = dict(
        cortex=int(ctx.sum()),
        claimed=int(got.sum()),
        fraction=float(got.sum() / max(ctx.sum(), 1)),
        pieces=int(npc),
    )
    if npc == 0:
        out.update(
            no_wm=0,
            topology=0,
            face_or_missing=0,
            sign=0,
            voxels=dict(no_wm=0, topology=0, face_or_missing=0, sign=0),
        )
        return out
    # cortex components that touch WM at all
    cl, _ = ndi.label(ctx, np.ones((3, 3, 3)))
    touch = np.unique(cl[ndi.binary_dilation(wm, np.ones((3, 3, 3))) & ctx])
    near_cut = np.zeros(f.shape, bool)
    for m in (face, missing):
        if m is not None:
            near_cut |= ndi.binary_dilation(m, np.ones((3, 3, 3)))
    sizes = np.bincount(lab.ravel(), minlength=npc + 1)
    cause = {"no_wm": 0, "topology": 0, "face_or_missing": 0, "sign": 0}
    vox = dict(cause)
    blocked = f.state == BLOCKED
    for i in range(1, npc + 1):
        m = lab == i
        c0 = cl[m][0]
        if c0 not in touch:
            k = "no_wm"
        elif blocked[m].any():
            k = "topology"
        elif (near_cut & m).any():
            k = "face_or_missing"
        else:
            k = "sign"
        cause[k] += 1
        vox[k] += int(sizes[i])
    out.update(cause)
    out["voxels"] = vox
    return out
