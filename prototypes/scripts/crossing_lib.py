"""Crossing paths: contacts read from the laminar profile (plan: reports/Crossing paths and contacts
from laminar profiles.md). Pure functions, no argv; all 3D.

- trace: streamlines of the unsigned layer normal (the structure tensor interpolated as a tensor,
  its leading eigenvector flipped to agree with the previous step), many at once, RK2.
- profile: disc-averaged, mask-normalized samples along a curved path at uniform arc length.
- depth maps: linear (r = s t0 / t) and top-anchored (the top A mm keep their absolute thickness,
  the deep part stretches), from a column of thickness t onto a template of thickness t0.
- Template: median profile and per-depth spread over open columns, per feature.
- split estimators along a path of length L: E0 mirror symmetry (no template), E-dark (darkest
  point near the middle), E1 profile likelihood (flat prior), E2 with a Gaussian prior on each
  side's thickness, and the prior alone (the pairwise posterior).

Units: mm along paths; positions in voxel index coordinates; vox the voxel size (mm, per axis).
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

# outcome codes of trace
RUNNING, WM, FLUID, MISSING, FACE, OTHER, TURN, LENGTH = range(8)
OUTCOME = {
    RUNNING: "running",
    WM: "WM",
    FLUID: "fluid",
    MISSING: "missing",
    FACE: "face",
    OTHER: "other",
    TURN: "turn",
    LENGTH: "length",
}
STEP_MM = 0.04  # sample spacing along paths


def _tensor_at(Tc, x):
    """3x3 tensors (n, 3, 3) at positions x (n, 3), trilinear in each of the six components."""
    t = [ndi.map_coordinates(c, x.T, order=1, mode="nearest") for c in Tc]
    M = np.empty((x.shape[0], 3, 3))
    M[:, 0, 0], M[:, 0, 1], M[:, 0, 2] = t[0], t[1], t[2]
    M[:, 1, 1], M[:, 1, 2], M[:, 2, 2] = t[3], t[4], t[5]
    M[:, 1, 0], M[:, 2, 0], M[:, 2, 1] = t[1], t[2], t[4]
    return M


def _dir_at(Tc, x, ref):
    """Unit leading eigenvector at x, flipped to agree with ref (n, 3)."""
    _, V = np.linalg.eigh(_tensor_at(Tc, x))
    d = V[:, :, 2]
    d *= np.where((d * ref).sum(1) < 0, -1.0, 1.0)[:, None]
    return d


def trace(
    Tc,
    labels,
    dface,
    x0,
    d0,
    vox,
    step_mm=STEP_MM,
    lmax=8.0,
    face_mm=2.0,
    wm_after=0.0,
    turn_deg=60.0,
    turn_mm=0.3,
):
    """Streamlines of the unsigned layer normal from x0 (n, 3; voxel coords), first direction
    agreeing with d0 (n, 3). labels: 0 cortex, 1 fluid, 2 WM, 3 missing, anything else other.
    Stops: WM (only after wm_after mm), fluid, missing, other, within face_mm of a box face, a
    turn of more than turn_deg within turn_mm, lmax. Directions are taken in voxel index space
    and steps scaled by vox (the voxels are near-isotropic).
    Returns a list of point arrays (m_i, 3) and an array of outcome codes."""
    x = np.asarray(x0, float).copy()
    n = x.shape[0]
    d = np.asarray(d0, float) / np.linalg.norm(d0, axis=1, keepdims=True)
    vox = np.asarray(vox, float)
    h = step_mm / vox
    shape = np.array(labels.shape)
    pts = [[xi.copy()] for xi in x]
    out = np.full(n, RUNNING)
    length = np.zeros(n)
    lag = max(1, int(round(turn_mm / step_mm)))
    hist = [d.copy()]
    act = np.arange(n)
    while act.size:
        xa, da = x[act], d[act]
        d1 = _dir_at(Tc, xa, da)
        xm = xa + 0.5 * d1 * h
        d2 = _dir_at(Tc, xm, d1)
        xa = xa + d2 * h
        x[act], d[act] = xa, d2
        length[act] += step_mm
        for i, xi in zip(act, xa):
            pts[i].append(xi.copy())
        q = np.clip(np.round(xa).astype(int), 0, shape - 1)
        lab = labels[tuple(q.T)]
        inside = np.all((xa >= 0) & (xa <= shape - 1), 1)
        code = np.full(act.size, RUNNING)
        code[(lab == 2) & (length[act] > wm_after)] = WM
        code[lab == 1] = FLUID
        code[lab == 3] = MISSING
        code[(lab > 3) | (lab < 0)] = OTHER
        code[(code == RUNNING) & ((dface[tuple(q.T)] < face_mm) | ~inside)] = FACE
        hist.append(d.copy())
        if len(hist) > lag:
            old = hist[-1 - lag][act]
            cosang = np.abs((old * d2).sum(1))
            code[(code == RUNNING) & (cosang < np.cos(np.radians(turn_deg)))] = TURN
        code[(code == RUNNING) & (length[act] >= lmax)] = LENGTH
        out[act] = code
        act = act[code == RUNNING]
    return [np.array(p) for p in pts], out


def arclength(p, vox):
    return np.r_[
        0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0) * np.asarray(vox), axis=1))
    ]


def resample(p, vox, step_mm=STEP_MM):
    """The path at uniform arc length (from 0 to its length, in step_mm steps)."""
    s = arclength(p, vox)
    ss = np.arange(0.0, s[-1] + 1e-9, step_mm)
    return np.stack([np.interp(ss, s, p[:, a]) for a in range(3)], 1), ss


def _disc(d, r, n_ring=6):
    """Offsets (k, 3) in mm of a disc of radius r perpendicular to d (as a5_profiles.disc)."""
    a = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(d, a)
    offs = [np.zeros(3)]
    if r > 0:
        for rr, k in ((r / 2, n_ring), (r, 2 * n_ring)):
            for t in np.linspace(0, 2 * np.pi, k, endpoint=False):
                offs.append(rr * (np.cos(t) * a + np.sin(t) * b))
    return np.array(offs)


def profile(imgs, mask, p, vox, radius=0.3):
    """Disc-averaged samples of each image along a path already at uniform arc length (p: (m, 3)),
    the average taken over the mask only (sum img*mask / sum mask), so voxels outside the mask
    (WM, fluid) do not pull it toward zero. Returns (len(imgs), m), nan where the disc holds no mask."""
    vox = np.asarray(vox, float)
    tang = np.gradient(p * vox, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12
    coords = []
    for xi, ti in zip(p, tang):
        coords.append(xi[None] + _disc(ti, radius) / vox)
    c = np.concatenate(coords, 0)
    k = c.shape[0] // p.shape[0]
    m = ndi.map_coordinates(mask.astype(float), c.T, order=1, mode="nearest").reshape(
        -1, k
    )
    msum = m.sum(1)
    out = []
    for img in imgs:
        v = (
            ndi.map_coordinates(img * mask, c.T, order=1, mode="nearest")
            .reshape(-1, k)
            .sum(1)
        )
        out.append(np.where(msum > 1e-3, v / np.maximum(msum, 1e-12), np.nan))
    return np.array(out)


# ---------------------------------------------------------------- depth maps


def depth_map(s, t, t0, kind, a_frac=0.5):
    """Template depth r (0 at the WM, t0 at the top) of arc length s in a column of thickness t.
    linear: r = s t0 / t. top: the top a = a_frac t0 keeps its absolute thickness (r = t0 - (t - s)),
    the deep part stretches (r = s (t0 - a) / (t - a)); a is capped at t - 0.2 t0 so that the deep part
    keeps some length in thin columns."""
    s = np.asarray(s, float)
    if kind == "linear":
        return s * t0 / t
    a = min(a_frac * t0, max(t - 0.2 * t0, 1e-6))
    deep = s * (t0 - a) / max(t - a, 1e-9)
    return np.where(s < t - a, deep, t0 - (t - s))


@dataclass
class Template:
    t0: float  # reference thickness (mm): the median of the columns
    r: np.ndarray  # depth grid (mm), 0 at the WM to t0
    med: np.ndarray  # (features, len(r)) median profile
    sd: np.ndarray  # (features, len(r)) spread (1.4826 MAD), floored
    n: int  # columns used
    kind: str


def build_template(
    profiles, thick, kind, step_mm=STEP_MM, sd_floor_frac=0.1, min_cols=5
):
    """profiles: list of (features, m_i) along open columns from root (s = 0) to top (s = t_i),
    thick: their t_i. Each is mapped onto r in [0, t0] with depth_map, then the median and spread
    are taken per r. sd is floored at sd_floor_frac of its median (a few columns can agree by
    chance). None if fewer than min_cols columns."""
    thick = np.asarray(thick, float)
    if thick.size < min_cols:
        return None
    t0 = float(np.median(thick))
    r = np.arange(0.0, t0 + 1e-9, step_mm)
    nf = profiles[0].shape[0]
    stack = np.full((len(profiles), nf, r.size), np.nan)
    for i, (pr, t) in enumerate(zip(profiles, thick)):
        s = np.arange(pr.shape[1]) * step_mm
        rr = depth_map(s, t, t0, kind)
        for f in range(nf):
            ok = np.isfinite(pr[f])
            if ok.sum() > 3:
                stack[i, f] = np.interp(r, rr[ok], pr[f, ok], left=np.nan, right=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # depths no column covers
        med = np.nanmedian(stack, 0)
        cover = np.sum(np.isfinite(stack), 0)
    for f in range(nf):  # depths no column covers: the nearest covered value
        ok = np.isfinite(med[f])
        if not ok.any():
            return None
        med[f, ~ok] = np.interp(r[~ok], r[ok], med[f, ok])
    # spread from each column's own samples against the median read at their depth (interpolating
    # the columns onto r first would average neighbouring samples and shrink it), in bins of
    # bin_mm, plus the median's own sampling variance (1.2533^2 sd^2 / n)
    bin_mm = 3 * step_mm
    nb = int(np.ceil((t0 + 1e-9) / bin_mm))
    sd = np.zeros((nf, r.size))
    for f in range(nf):
        res = [[] for _ in range(nb)]
        for pr, t in zip(profiles, thick):
            rr = depth_map(np.arange(pr.shape[1]) * step_mm, t, t0, kind)
            ok = np.isfinite(pr[f]) & (rr <= t0)
            d = pr[f, ok] - np.interp(rr[ok], r, med[f])
            for b, v in zip(np.minimum((rr[ok] / bin_mm).astype(int), nb - 1), d):
                res[b].append(v)
        sb = np.array(
            [
                1.4826 * np.median(np.abs(np.array(v) - np.median(v)))
                if len(v) > 4
                else np.nan
                for v in res
            ]
        )
        ok = np.isfinite(sb)
        if not ok.any():
            return None
        centres = (np.arange(nb) + 0.5) * bin_mm
        sd[f] = np.interp(r, centres[ok], sb[ok])
    sd *= np.sqrt(1.0 + 1.5708 / np.maximum(cover, 1))
    floor = sd_floor_frac * np.median(sd, 1, keepdims=True)
    sd = np.maximum(sd, np.maximum(floor, 1e-9))
    return Template(t0, r, med, sd, len(profiles), kind)


def expect(tpl, s, t):
    """Template mean and sd at arc length s of a column of thickness t: (features, len(s)) each."""
    rr = depth_map(s, t, tpl.t0, tpl.kind)
    mu = np.array([np.interp(rr, tpl.r, m) for m in tpl.med])
    sd = np.array([np.interp(rr, tpl.r, v) for v in tpl.sd])
    return mu, sd


# ---------------------------------------------------------------- split estimators


def xgrid(L, lo=0.15, hi=0.85, step_mm=0.02):
    return np.arange(lo * L, hi * L + 1e-9, step_mm)


def _side_fit(y, mu, sd):
    """Weighted least squares of y on gain * mu + offset (gain >= 0); returns the residuals over sd."""
    ok = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sd)
    if ok.sum() < 3:
        return np.zeros(0), np.zeros(0)
    w = 1.0 / sd[ok] ** 2
    X = np.c_[mu[ok], np.ones(ok.sum())]
    A = X.T @ (X * w[:, None])
    b = X.T @ (y[ok] * w)
    try:
        g, c = np.linalg.solve(A + 1e-9 * np.eye(2), b)
    except np.linalg.LinAlgError:
        g, c = 0.0, np.sum(w * y[ok]) / np.sum(w)
    if g < 0:
        g, c = 0.0, np.sum(w * y[ok]) / np.sum(w)
    return (y[ok] - g * mu[ok] - c) / sd[ok], sd[ok]


def loglik(y, tplA, tplB, L, xs, step_mm=STEP_MM, eff_mm=0.24):
    """log p(profile | split at x) for each x in xs: side A (s < x) is tplA at thickness x, side B
    (s > x) is tplB, reversed, at thickness L - x; a gain and offset per side and feature. Gaussian
    with the template's per-depth sd; the samples step_mm apart count as one every eff_mm (the
    profile is smooth over about the tensor's scales)."""
    s = np.arange(y.shape[1]) * step_mm
    out = np.full(len(xs), -np.inf)
    for j, x in enumerate(xs):
        A = s < x
        tot = 0.0
        for side, tpl, ss, t in ((A, tplA, s[A], x), (~A, tplB, L - s[~A], L - x)):
            mu, sd = expect(tpl, ss, t)
            for f in range(y.shape[0]):
                z, sdo = _side_fit(y[f, side], mu[f], sd[f])
                tot += -0.5 * np.sum(z**2) - np.sum(np.log(sdo))
        out[j] = tot * step_mm / eff_mm
    return out


def gauss_prior(xs, L, muA, sdA, muB, sdB):
    return -0.5 * ((xs - muA) / sdA) ** 2 - 0.5 * ((L - xs - muB) / sdB) ** 2


def pairwise_posterior(L, muA, sdA, muB, sdB):
    """The main plan's pairwise posterior (COHIBA): x = muA + sdA^2/(sdA^2 + sdB^2)(L - muA - muB)."""
    x = muA + sdA**2 / (sdA**2 + sdB**2) * (L - muA - muB)
    return x, sdA * sdB / np.sqrt(sdA**2 + sdB**2)


def summarize(xs, logp):
    """Posterior mean, SD and MAP on the grid from unnormalized log p."""
    if not np.isfinite(logp).any():
        return dict(mean=np.nan, sd=np.nan, map=np.nan)
    w = np.exp(logp - np.nanmax(logp))
    w = np.nan_to_num(w)
    w /= w.sum()
    m = float(np.sum(w * xs))
    return dict(
        mean=m,
        sd=float(np.sqrt(np.sum(w * (xs - m) ** 2))),
        map=float(xs[np.argmax(logp)]),
    )


def symmetry(y, L, xs, step_mm=STEP_MM, nv=40):
    """E0: for each x, the correlation between the profile at fraction v of the way from x down to
    root A and at fraction v from x down to root B (each side stretched linearly to its own
    thickness), features standardized and pooled; v in (0.05, 0.95). Returns the score per x."""
    s = np.arange(y.shape[1]) * step_mm
    z = (y - np.nanmean(y, 1, keepdims=True)) / (np.nanstd(y, 1, keepdims=True) + 1e-12)
    v = np.linspace(0.05, 0.95, nv)
    out = np.full(len(xs), np.nan)
    for j, x in enumerate(xs):
        a = np.concatenate([np.interp(x * (1 - v), s, f) for f in z])
        b = np.concatenate([np.interp(x + (L - x) * v, s, f) for f in z])
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 5:
            out[j] = np.corrcoef(a[ok], b[ok])[0, 1]
    return out


def darkest(y0, L, lo=0.3, hi=0.7, step_mm=STEP_MM, smooth_mm=0.08):
    """E-dark: the darkest point of one feature within [lo L, hi L], lightly smoothed."""
    s = np.arange(y0.size) * step_mm
    v = ndi.gaussian_filter1d(
        np.nan_to_num(y0, nan=np.nanmedian(y0)), smooth_mm / step_mm
    )
    win = (s >= lo * L) & (s <= hi * L)
    if not win.any():
        return np.nan
    return float(s[win][np.argmin(v[win])])


# ---------------------------------------------------------------- the column volume


def trace_array(
    Tc,
    labels,
    dface,
    x0,
    d0,
    vox,
    nmax,
    step_mm=STEP_MM,
    face_mm=2.0,
    wm_after=0.3,
    turn_deg=60.0,
    turn_mm=0.3,
):
    """As trace, for many paths at once, stored in arrays: positions (n, nmax, 3) float32 (nan past
    each end; sample i at arc length i step_mm), the number of samples, and the outcome. A path
    still running after nmax - 1 steps ends as LENGTH."""
    x = np.asarray(x0, float).copy()
    n = x.shape[0]
    d = np.asarray(d0, float) / np.linalg.norm(d0, axis=1, keepdims=True)
    vox = np.asarray(vox, float)
    h = step_mm / vox
    shape = np.array(labels.shape)
    pos = np.full((n, nmax, 3), np.nan, np.float32)
    pos[:, 0] = x
    nsamp = np.ones(n, int)
    out = np.full(n, RUNNING)
    lag = max(1, int(round(turn_mm / step_mm)))
    hist = np.zeros((lag + 1, n, 3))
    hist[:] = d
    act = np.arange(n)
    for i in range(1, nmax):
        if not act.size:
            break
        xa, da = x[act], d[act]
        d1 = _dir_at(Tc, xa, da)
        d2 = _dir_at(Tc, xa + 0.5 * d1 * h, d1)
        xa = xa + d2 * h
        x[act], d[act] = xa, d2
        pos[act, i] = xa
        nsamp[act] = i + 1
        q = np.clip(np.round(xa).astype(int), 0, shape - 1)
        lab = labels[tuple(q.T)]
        inside = np.all((xa >= 0) & (xa <= shape - 1), 1)
        code = np.full(act.size, RUNNING)
        code[(lab == 2) & (i * step_mm > wm_after)] = WM
        code[lab == 1] = FLUID
        code[lab == 3] = MISSING
        code[(lab > 3) | (lab < 0)] = OTHER
        code[(code == RUNNING) & ((dface[tuple(q.T)] < face_mm) | ~inside)] = FACE
        old = hist[i % (lag + 1)][act]  # the direction lag steps ago
        hist[i % (lag + 1)][act] = d2
        if i > lag:
            code[
                (code == RUNNING)
                & (np.abs((old * d2).sum(1)) < np.cos(np.radians(turn_deg)))
            ] = TURN
        if i == nmax - 1:
            code[code == RUNNING] = LENGTH
        out[act] = code
        act = act[code == RUNNING]
    return pos, nsamp, out


# outcomes of ascend beyond those of trace
CONTACT, RIDGE = 8, 9
OUTCOME.update({CONTACT: "contact", RIDGE: "ridge"})


def ascend(
    u, labels, dface, x0, vox, nmax, far, step_mm=STEP_MM, face_mm=2.0, wm_after=0.3
):
    """Columns of an arrival field: ascent of grad u from x0 (n, 3), many at once, in arrays as
    trace_array. far(i_paths, q) -> bool per path: the voxel q belongs to a root far from that path's
    own (the pass's contact). Stops: fluid, WM (after wm_after), missing, other, face, CONTACT, RIDGE
    (u no longer increases), LENGTH."""
    g = np.stack(np.gradient(u), 0)
    x = np.asarray(x0, float).copy()
    n = x.shape[0]
    vox = np.asarray(vox, float)
    shape = np.array(labels.shape)
    pos = np.full((n, nmax, 3), np.nan, np.float32)
    pos[:, 0] = x
    nsamp = np.ones(n, int)
    out = np.full(n, RUNNING)
    prev = ndi.map_coordinates(u, x.T, order=1, mode="nearest")
    act = np.arange(n)
    for i in range(1, nmax):
        if not act.size:
            break
        xa = x[act]
        v = np.stack(
            [ndi.map_coordinates(c, xa.T, order=1, mode="nearest") for c in g], 1
        )
        nv = np.linalg.norm(v * (1.0 / vox), axis=1, keepdims=True)  # direction in mm
        dirn = v / vox / np.maximum(nv, 1e-12)
        xa = xa + dirn * step_mm / vox
        x[act] = xa
        pos[act, i] = xa
        nsamp[act] = i + 1
        q = np.clip(np.round(xa).astype(int), 0, shape - 1)
        lab = labels[tuple(q.T)]
        inside = np.all((xa >= 0) & (xa <= shape - 1), 1)
        ui = ndi.map_coordinates(u, xa.T, order=1, mode="nearest")
        code = np.full(act.size, RUNNING)
        code[nv[:, 0] < 1e-9] = RIDGE
        code[(lab == 2) & (i * step_mm > wm_after)] = WM
        code[lab == 1] = FLUID
        code[lab == 3] = MISSING
        code[(lab > 3) | (lab < 0)] = OTHER
        code[(code == RUNNING) & ((dface[tuple(q.T)] < face_mm) | ~inside)] = FACE
        code[(code == RUNNING) & far(act, q)] = CONTACT
        code[(code == RUNNING) & (ui < prev[act] - 1e-6) & (i * step_mm > wm_after)] = (
            RIDGE
        )
        prev[act] = ui
        if i == nmax - 1:
            code[code == RUNNING] = LENGTH
        out[act] = code
        act = act[code == RUNNING]
    return pos, nsamp, out


def sample_traces(imgs, pos, chunk=20000):
    """Each image at every trace sample (trilinear; nan where pos is nan): (len(imgs), n, nmax) float32."""
    n, m, _ = pos.shape
    out = np.full((len(imgs), n, m), np.nan, np.float32)
    for a in range(0, n, chunk):
        p = pos[a : a + chunk].reshape(-1, 3)
        ok = np.isfinite(p[:, 0])
        for k, img in enumerate(imgs):
            v = np.full(p.shape[0], np.nan, np.float32)
            v[ok] = ndi.map_coordinates(
                img, p[ok].T.astype(float), order=1, mode="nearest"
            )
            out[k, a : a + chunk] = v.reshape(-1, m)
    return out


def order_along(idx, graph):
    """Order the nodes idx (a connected piece of graph) along their longest geodesic: a double sweep
    from an arbitrary node gives one end, and every node is placed at its distance from that end.
    Returns (idx sorted, distances)."""
    from scipy.sparse.csgraph import dijkstra

    sub = graph[idx][:, idx]
    d0 = dijkstra(sub, indices=0)
    end = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
    d = dijkstra(sub, indices=end)
    o = np.argsort(d)
    return np.asarray(idx)[o], d[o]


def exp_chart(x, x0, n0, d):
    """A local 2D chart of surface points x (k, 3; mm) around x0 with normal n0: each point at its
    geodesic distance d from x0, in the direction of its projection onto the tangent plane at x0
    (an approximation of the exponential map: radial distances exact, angles exact on a plane and
    distorted where the surface bends round within the radius). Returns (k, 2) mm."""
    n0 = np.asarray(n0, float) / np.linalg.norm(n0)
    a = np.cross(n0, [1.0, 0, 0] if abs(n0[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(n0, a)
    r = np.asarray(x, float) - np.asarray(x0, float)
    pu, pv = r @ a, r @ b
    ang = np.arctan2(pv, pu)
    return np.stack([d * np.cos(ang), d * np.sin(ang)], 1)
