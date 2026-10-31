"""Shared pieces for the closed-form equivolume tests: boxes from the full volume, white-surface
curvature, the data-steered T with T = 0 on all WM boundary, and columns from roots.

Boxes are loaded from the full volume. Box faces are never WM boundary (the erosion counts the
outside as WM), and are free in the fit (no edges cross them).

White-surface curvature: the signed WM distance (positive into cortex), smoothed by a Gaussian of
SIG voxels, with derivatives taken from the Gaussian derivative filters. At each WM-boundary
voxel the point is moved onto the zero level along the normal, and the shape operator there is
P Hess P / |grad|, P the tangent projector. Its two eigenvalues are kappa_1 >= kappa_2 (1/mm),
positive where the white surface is convex (crowns), negative where concave (fundi).
"""

import numpy as np
from scipy import ndimage as ndi

import flatten_fused2 as F
from baillarger.data import load_box
from baillarger.frame import missing_tissue
from fit_T import fill, solve_field

LAM = 0.1
STEP = 0.25  # voxels


def face_dist(shape, vox):
    """Distance (mm) of every voxel to the nearest box face."""
    idx = np.indices(shape)
    d = [np.minimum(idx[a], shape[a] - 1 - idx[a]) * vox[a] for a in range(3)]
    return np.minimum(np.minimum(d[0], d[1]), d[2])


def load(lo, hi, name="box"):
    p = load_box(tuple(int(v) for v in lo), tuple(int(v) for v in hi), name=name)
    rim, vox = p.rim, p.voxel_mm
    miss = missing_tissue(rim, p.flash, vox)
    ctx = (rim == 3) & ~miss
    wm = rim == 2
    shell = wm & ~ndi.binary_erosion(wm, border_value=1)
    mb = ctx.astype(float)
    e = {i: np.log(np.maximum(p.echoes[..., i].astype(float), 1.0)) for i in (1, 2)}
    det = {i: (L - F.mconv(L, mb, F.SHP)) * mb for i, L in e.items()}
    return dict(
        p=p,
        lo=np.array(lo),
        rim=rim,
        vox=np.array(vox),
        miss=miss,
        ctx=ctx,
        wm=wm,
        shell=shell,
        det=det,
        disp=det[1],
        dface=face_dist(rim.shape, vox),
    )


def wm_faces(wm, vox):
    """The exposed faces of the WM label: centres (mm), outward unit normals, and the WM voxel
    each face belongs to. Box faces are not exposed."""
    cs, ns, vs = [], [], []
    for a in range(3):
        for sgn in (-1, 1):
            nb = np.ones_like(wm)  # outside the box counts as WM (not exposed)
            src = [slice(None)] * 3
            dst = [slice(None)] * 3
            if sgn > 0:
                src[a] = slice(1, None)
                dst[a] = slice(0, -1)
            else:
                src[a] = slice(0, -1)
                dst[a] = slice(1, None)
            nb[tuple(dst)] = wm[tuple(src)]
            v = np.argwhere(wm & ~nb)
            e = np.zeros(3)
            e[a] = sgn
            cs.append((v + 0.5 * e) * vox)
            ns.append(np.repeat(e[None], len(v), 0))
            vs.append(v)
    return np.concatenate(cs), np.concatenate(ns), np.concatenate(vs)


def curvature(B, sig_mm, chunk=3000):
    """kappa_1 >= kappa_2 (1/mm) and the unit outward normal at every WM-boundary voxel (arrays
    over argwhere(shell)), from a weighted quadric fitted to the exposed WM faces around it.

    Only faces on the same side of the WM as the voxel (outward normals agreeing) enter, so the
    two sides of a thin WM blade never mix. Gaussian weights of sig_mm set the scale.
    """
    from scipy.spatial import cKDTree

    wm, vox = B["wm"], B["vox"]
    fc, fn, fv = wm_faces(wm, vox)
    pts = np.argwhere(B["shell"])
    idx = -np.ones(wm.shape, np.int64)
    idx[tuple(pts.T)] = np.arange(len(pts))
    own = idx[tuple(fv.T)]
    ok = own >= 0
    cnt = np.bincount(own[ok], minlength=len(pts)).astype(float)
    c0 = (
        np.stack([np.bincount(own[ok], fc[ok, a], len(pts)) for a in range(3)], 1)
        / np.maximum(cnt, 1)[:, None]
    )
    n0 = np.stack([np.bincount(own[ok], fn[ok, a], len(pts)) for a in range(3)], 1)
    bare = cnt == 0  # shell voxels exposed only diagonally: centre on the voxel
    c0[bare] = pts[bare] * vox
    tree = cKDTree(fc)
    k = int(
        min(
            len(fc),
            max(60, 1.6 * np.pi * (2.5 * sig_mm) ** 2 / (vox[0] * vox[1]) * 1.6),
        )
    )
    k1 = np.zeros(len(pts))
    k2 = np.zeros(len(pts))
    nout = np.zeros((len(pts), 3))
    for s0 in range(0, len(pts), chunk):
        sl = slice(s0, min(s0 + chunk, len(pts)))
        c, n = c0[sl], n0[sl]
        d, j = tree.query(c, k=k, distance_upper_bound=3.0 * sig_mm)
        valid = np.isfinite(d)
        j = np.where(valid, j, 0)
        P = fc[j] - c[:, None]
        N = fn[j]
        w = np.where(valid, np.exp(-0.5 * (d / sig_mm) ** 2), 0.0)
        bad = np.linalg.norm(n, axis=1) < 1e-9
        n = np.where(bad[:, None], (w[..., None] * N).sum(1), n)
        for _ in range(2):  # normal from same-side faces nearby
            n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
            same = (N * n[:, None]).sum(-1) > 0.0
            n = (w[..., None] * same[..., None] * N).sum(1)
        n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
        same = (N * n[:, None]).sum(-1) > 0.0
        ww = w * same
        ref = np.where(
            np.abs(n[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]])
        )
        e1 = np.cross(n, ref)
        e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(n, e1)
        x = (P * e1[:, None]).sum(-1)
        y = (P * e2[:, None]).sum(-1)
        h = (P * n[:, None]).sum(-1)
        Xd = np.stack([x * x, x * y, y * y, x, y, np.ones_like(x)], -1)
        A = np.einsum("nk,nki,nkj->nij", ww, Xd, Xd) + 1e-6 * np.eye(6)
        b = np.einsum("nk,nki,nk->ni", ww, Xd, h)
        q = np.linalg.solve(A, b[..., None])[..., 0]
        hx, hy = q[:, 3], q[:, 4]
        Hs = np.stack(
            [np.stack([2 * q[:, 0], q[:, 1]], 1), np.stack([q[:, 1], 2 * q[:, 2]], 1)],
            1,
        )
        gvec = np.stack([hx, hy], 1)
        I = np.eye(2)[None] + gvec[:, :, None] * gvec[:, None, :]
        II = Hs / np.sqrt(1 + hx**2 + hy**2)[:, None, None]
        Sop = np.linalg.solve(I, II)
        ev = np.sort(np.real(np.linalg.eigvals(Sop)), 1)
        # height along the outward normal bends down on a convex (crown) surface: kappa = -eig
        k1[sl], k2[sl] = -ev[:, 0], -ev[:, 1]
        nout[sl] = n
    return dict(pts=pts, k1=k1, k2=k2, n=nout)


def curvature_sd(B, sig):
    """(Superseded: mixes the two sides of thin WM blades.) kappa_1, kappa_2 from the signed WM
    distance smoothed by sig voxels."""
    wm, vox = B["wm"], B["vox"]
    sd = ndi.distance_transform_edt(~wm, sampling=vox) - ndi.distance_transform_edt(
        wm, sampling=vox
    )
    ords1 = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    g = [ndi.gaussian_filter(sd, sig, order=o) / vox[a] for a, o in enumerate(ords1)]
    pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    H = []
    for a, b in pairs:
        o = [0, 0, 0]
        o[a] += 1
        o[b] += 1
        H.append(ndi.gaussian_filter(sd, sig, order=tuple(o)) / (vox[a] * vox[b]))
    sds = ndi.gaussian_filter(sd, sig)
    pts = np.argwhere(B["shell"]).astype(float)
    at = lambda f, x: ndi.map_coordinates(f, x.T, order=1, mode="nearest")
    gx = np.stack([at(c, pts) for c in g], 1)
    n = gx / (np.linalg.norm(gx, axis=1, keepdims=True) + 1e-12)
    x0 = (
        pts - (at(sds, pts) / np.linalg.norm(gx, axis=1))[:, None] * n / vox
    )  # onto the zero level
    gx = np.stack([at(c, x0) for c in g], 1)
    gn = np.linalg.norm(gx, axis=1)
    n = gx / (gn[:, None] + 1e-12)
    h = np.stack([at(c, x0) for c in H], 1)
    M = np.stack(
        [
            np.stack([h[:, 0], h[:, 1], h[:, 2]], 1),
            np.stack([h[:, 1], h[:, 3], h[:, 4]], 1),
            np.stack([h[:, 2], h[:, 4], h[:, 5]], 1),
        ],
        1,
    )
    # tangent basis
    ref = np.where(
        np.abs(n[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]])
    )
    e1 = np.cross(n, ref)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(n, e1)
    E = np.stack([e1, e2], 2)  # (N, 3, 2)
    S = np.einsum("nia,nij,njb->nab", E, M, E) / gn[:, None, None]
    k = np.linalg.eigvalsh(S)  # ascending
    return dict(pts=pts.astype(int), k1=k[:, 1], k2=k[:, 0], n=n)


def tensor_normal(B):
    mb = B["ctx"].astype(float)
    Tc = F.tensor(B["det"][1], mb) + F.tensor(B["det"][2], mb)
    Mt = np.stack(
        [
            np.stack([Tc[0], Tc[1], Tc[2]], -1),
            np.stack([Tc[1], Tc[3], Tc[4]], -1),
            np.stack([Tc[2], Tc[4], Tc[5]], -1),
        ],
        -2,
    )
    w_, V = np.linalg.eigh(Mt)
    nh = V[..., :, 2]
    coh = np.clip(
        (w_[..., 2] - w_[..., 1]) / np.maximum(w_[..., 2] + w_[..., 1], 1e-12), 0, 1
    )
    return nh, np.where(B["ctx"], coh, 0.0)


def fit_all_wm(B, dom_ctx=None):
    """Unit-speed data-steered T with T = 0 on all WM boundary (box faces excluded), and the
    divergence of its column direction (the old equivolume route, as in cheap_test.py).
    dom_ctx: the cortex the field is solved on, if not B['ctx'] (a closed gap); the data tensor
    always uses B['ctx']."""
    shell, vox = B["shell"], B["vox"]
    nh, coh = tensor_normal(B)
    ctx = B["ctx"] if dom_ctx is None else dom_ctx
    dom = ctx | (shell & ndi.binary_dilation(ctx))
    fixed = dom & B["wm"]
    d_wm = ndi.distance_transform_edt(~B["wm"], sampling=vox)
    g = np.stack(np.gradient(ndi.gaussian_filter(d_wm, 1.0), *vox), -1)
    gh = g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-12)
    sgn = np.sign((nh * gh).sum(-1))
    sgn[sgn == 0] = 1
    dirv = coh[..., None] * nh * sgn[..., None] + LAM * gh
    dirv /= np.linalg.norm(dirv, axis=-1, keepdims=True) + 1e-12
    T, info = solve_field(dom, fixed, [dirv[..., a] for a in range(3)], tuple(vox))
    Tf = fill(T)
    gT = np.stack(np.gradient(Tf, *vox), -1)
    n1 = gT / (np.linalg.norm(gT, axis=-1, keepdims=True) + 1e-12)
    n1s = np.stack([ndi.gaussian_filter(n1[..., a], 1.0) for a in range(3)], -1)
    agree = (n1 * n1s).sum(-1) / (np.linalg.norm(n1s, axis=-1) + 1e-12)
    n1n = n1s / (np.linalg.norm(n1s, axis=-1, keepdims=True) + 1e-12)
    div = sum(np.gradient(n1n[..., a], vox[a], axis=a) for a in range(3))
    div = np.clip(np.where((agree > 0.8) & (Tf > 0.3) & ctx, div, 0.0), -2, 2)
    return dict(
        T=T,
        Tf=Tf,
        g=np.stack([gT[..., a] for a in range(3)], 0),
        div=div,
        coh=coh,
        info=info,
    )


def trace(B, Fd, seed, lmax=6.0):
    """A column from WM-boundary voxel seed up grad T, in 3D. Ends at fluid ('open'), where T
    stops increasing ('ridge'), or elsewhere. Returns points (voxel coords) and the outcome."""
    rb, vox = B["rim"], B["vox"]
    shape = np.array(rb.shape) - 1
    x = np.array(seed, float)
    pts = [x.copy()]
    prev = -np.inf
    s = 0.0
    while s < lmax:
        v = np.array([F.interp(c, x) for c in Fd["g"]])
        nv = np.linalg.norm(v)
        if nv < 1e-9:
            return np.array(pts), "stall"
        v = v / nv
        x = x + STEP * v
        s += STEP * float(np.linalg.norm(v * vox))
        pts.append(x.copy())
        q = tuple(np.clip(np.round(x).astype(int), 0, shape))
        if B["miss"][q]:
            return np.array(pts), "missing"
        if rb[q] == 1:
            return np.array(pts), "open"
        t = F.interp(Fd["Tf"], x)
        if t < prev - 1e-3:
            return np.array(pts), "ridge"
        prev = max(prev, t)
        if rb[q] == 2 and s > 0.6:
            return np.array(pts), "WM"
        if rb[q] not in (2, 3):
            return np.array(pts), "other"
    return np.array(pts), "length"


def column(B, Fd, pts, k1, k2, freeze_after=None):
    """Along one column: arc length s (0 at the white surface), echo 2 (trend removed, lightly
    smoothed), and the three depth coordinates: distance s, divergence-based volume v_div,
    closed-form volume V = s + Hm s^2 + K s^3 / 3 (Hm the mean curvature). Beyond index
    freeze_after (a ridge) the divergence is held at zero: the column's own front ends there."""
    vox = B["vox"]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0) * vox, axis=1))] + 0.08
    e = ndi.gaussian_filter1d(ndi.map_coordinates(B["disp"], pts.T, order=1), 1.5)
    dv = ndi.map_coordinates(Fd["div"], pts.T, order=1)
    if freeze_after is not None:
        dv[freeze_after + 1 :] = 0.0
    ds = np.diff(s, prepend=0.0)
    u = np.cumsum(dv * ds)
    v_div = np.cumsum(np.exp(u) * ds)
    Hm, K = 0.5 * (k1 + k2), k1 * k2
    V = s + Hm * s**2 + K * s**3 / 3.0
    area = (1 + k1 * s) * (1 + k2 * s)
    dmin = min(ndi.map_coordinates(B["dface"], pts.T, order=0).min(), 99.0)
    return dict(s=s, e=e, v_div=v_div, V=V, area_end=area[-1], dface=dmin)
