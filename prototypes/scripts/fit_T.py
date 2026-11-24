"""Minimal 3D fit of a laminar depth field T on one stretch of V1V2.

Domain: aseg cortex (without missing tissue) in a 3D box around the stretch,
plus the WM-boundary voxels, where T = 0. Fluid, excisions and crop faces are
free (no edges cross them).

Target vector field F on every edge between domain voxels (Lomask-style:
min sum_e (dT_e - F_e . h_e)^2, linear least squares, solved by CG):
  n_hat  data layer normal (summed tensor of trend-removed echoes 2 and 3),
         signed outward by the gradient of WM distance (so the ridge sits at
         the midline between WM surfaces: a stated limitation),
  w      tensor coherence (data confidence),
  g_hat  unit gradient of WM distance (the prior direction),
  F = speed * normalize(w * n_hat + LAM * g_hat).
speed = 1 (unit speed: T is mm along the data's own columns), or, with
--equivolume, exp(u) where u = ln(A / A0) comes from a least-squares fit of
grad u = div(n1) * n1, n1 being the column direction of the unit-speed fit.

Views (all computed in 3D; slices only display):
  fit_<name>_slices.png  three orthogonal slices through the stretch centre:
      T contours over trend-removed echo 3, the ridge, the angle between grad T
      and the data normal, coherence, and |grad T| (plus u with --equivolume)
  fit_<name>_flat.png    columns traced in 3D along grad T from the seeds, by
      outcome, and the flattened profiles by T, against WM-distance columns
"""

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.linalg import cg
import numba


@numba.njit
def upwind_u(order, dom, fixed, T1, n1, div, vox):
    """Carry u = ln(A/A0) outward with the front: n1 . grad u = div n1, u = 0 on the WM.

    Voxels are visited in increasing T1 (the unit-speed front's arrival). Each takes
    u from neighbours that are behind it along its own column direction and on the
    same front (n1 agrees), so nothing crosses the shock where fronts meet.
    """
    nx, ny, nz = dom.shape
    u = np.full(dom.shape, np.nan)
    for t in range(order.shape[0]):
        i, j, k = order[t, 0], order[t, 1], order[t, 2]
        if fixed[i, j, k]:
            u[i, j, k] = 0.0
            continue
        sw = 0.0
        su = 0.0
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    a_, b_, c_ = i - di, j - dj, k - dk
                    if a_ < 0 or b_ < 0 or c_ < 0 or a_ >= nx or b_ >= ny or c_ >= nz:
                        continue
                    if not dom[a_, b_, c_] or np.isnan(u[a_, b_, c_]):
                        continue
                    ex, ey, ez = di * vox[0], dj * vox[1], dk * vox[2]
                    d = np.sqrt(ex * ex + ey * ey + ez * ez)
                    al = (
                        n1[i, j, k, 0] * ex + n1[i, j, k, 1] * ey + n1[i, j, k, 2] * ez
                    ) / d
                    if al <= 0.3:
                        continue
                    same = (
                        n1[i, j, k, 0] * n1[a_, b_, c_, 0]
                        + n1[i, j, k, 1] * n1[a_, b_, c_, 1]
                        + n1[i, j, k, 2] * n1[a_, b_, c_, 2]
                    )
                    if same < 0.5 or T1[a_, b_, c_] >= T1[i, j, k]:
                        continue
                    w = al * al / d
                    su += w * (u[a_, b_, c_] + div[i, j, k] * d * al)
                    sw += w
        u[i, j, k] = su / sw if sw > 0 else 0.0
    return u


STRETCHES = {
    "openV1": ((71, 89, 54), (84, 131, 54)),
    "fused": ((32, 78, 27), (49, 114, 27)),
}
LAM = 0.1
STEP = 0.25


def edges(dom):
    """Index pairs of neighbouring domain voxels along each axis."""
    idx = -np.ones(dom.shape, np.int64)
    idx[dom] = np.arange(dom.sum())
    out = []
    for a in range(3):
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[a] = slice(0, -1)
        sl1[a] = slice(1, None)
        i0, i1 = idx[tuple(sl0)], idx[tuple(sl1)]
        ok = (i0 >= 0) & (i1 >= 0)
        pos = np.argwhere(ok)
        out.append((a, i0[ok], i1[ok], pos))
    return idx, out


def solve_field(dom, fixed, Fvec, vox, x0=None):
    """min sum over edges (T1 - T0 - F_a(mid) h_a)^2 with T = 0 on fixed voxels."""
    idx, E = edges(dom)
    n = int(dom.sum())
    free = ~fixed[dom]
    fmap = -np.ones(n, np.int64)
    fmap[free] = np.arange(free.sum())
    rows, cols, vals, rhs = [], [], [], []
    r = 0
    for a, i0, i1, pos in E:
        p0 = pos.copy()
        sl0 = tuple(p0.T)
        p1 = pos.copy()
        p1[:, a] += 1
        sl1 = tuple(p1.T)
        f = 0.5 * (Fvec[a][sl0] + Fvec[a][sl1]) * vox[a]
        m = len(i0)
        rr = np.arange(r, r + m)
        for ii, sgn in ((i1, 1.0), (i0, -1.0)):
            fr = fmap[ii]
            ok = fr >= 0
            rows.append(rr[ok])
            cols.append(fr[ok])
            vals.append(np.full(ok.sum(), sgn))
        rhs.append(f)  # fixed voxels have T = 0, so nothing moves to the rhs
        r += m
    A = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(r, int(free.sum())),
    )
    b = np.concatenate(rhs)
    N = (A.T @ A + 1e-6 * sparse.identity(A.shape[1])).tocsr()
    d = N.diagonal()
    M = sparse.diags(1.0 / d)
    t, info = cg(N, A.T @ b, x0=x0, M=M, rtol=1e-8, maxiter=5000)
    T = np.full(dom.shape, np.nan)
    tv = np.zeros(n)
    tv[free] = t
    T[dom] = tv
    return T, info


def fill(T):
    """Nearest-value fill outside the domain, for gradients and interpolation."""
    bad = np.isnan(T)
    if not bad.any():
        return T
    _, ind = ndi.distance_transform_edt(bad, return_indices=True)
    return T[tuple(ind)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=STRETCHES)
    ap.add_argument("outdir")
    ap.add_argument("scratch")
    ap.add_argument("--equivolume", action="store_true")
    ap.add_argument("--show-echo", type=int, default=2, choices=(2, 3))
    args = ap.parse_args()
    sys.path.insert(0, args.scratch)
    import flatten_fused2 as F
    from baillarger.data import load_patch
    from baillarger.frame import missing_tissue

    P1, P2 = STRETCHES[args.name]
    p = load_patch("V1V2")
    rim, vox = p.rim, p.voxel_mm
    miss = missing_tissue(rim, p.flash, vox)
    k = P1[2]
    lo = np.array([min(P1[0], P2[0]) - 30, min(P1[1], P2[1]) - 30, k - 30]).clip(0)
    hi = np.minimum(
        np.array([max(P1[0], P2[0]) + 31, max(P1[1], P2[1]) + 31, k + 31]), rim.shape
    )
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    rb, missb = rim[box], miss[box]
    ctx = (rb == 3) & ~missb
    mb = ctx.astype(float)
    wm = rb == 2
    shell = wm & ~ndi.binary_erosion(wm)
    dom = ctx | (shell & ndi.binary_dilation(ctx))
    fixed = dom & wm

    e = {i: np.log(np.maximum(p.echoes[box + (i,)].astype(float), 1.0)) for i in (1, 2)}
    det = {i: (L - F.mconv(L, mb, F.SHP)) * mb for i, L in e.items()}
    Tc = F.tensor(det[1], mb) + F.tensor(det[2], mb)
    disp = det[args.show_echo - 1]
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
    coh = (w_[..., 2] - w_[..., 1]) / np.maximum(w_[..., 2] + w_[..., 1], 1e-12)
    coh = np.where(dom, coh, 0.0)
    d_wm = ndi.distance_transform_edt(~wm, sampling=vox)
    g = np.stack(np.gradient(ndi.gaussian_filter(d_wm, 1.0), *vox), -1)
    gh = g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-12)
    sgn = np.sign((nh * gh).sum(-1))
    sgn[sgn == 0] = 1
    nh = nh * sgn[..., None]
    dirv = coh[..., None] * nh + LAM * gh
    dirv /= np.linalg.norm(dirv, axis=-1, keepdims=True) + 1e-12
    Fu = [dirv[..., a] for a in range(3)]
    print(
        f"{args.name}: box {rb.shape}, domain {dom.sum():,} voxels, fixed (WM) {fixed.sum():,}"
    )

    T1, info = solve_field(dom, fixed, Fu, vox)
    print(f"unit-speed solve: cg info {info}")
    Tf = fill(T1)
    gT = np.stack(np.gradient(Tf, *vox), -1)
    n1 = gT / (np.linalg.norm(gT, axis=-1, keepdims=True) + 1e-12)
    T, u = T1, None
    if args.equivolume:
        # divergence of the column direction, kept off the shock: where n1 disagrees
        # with its smoothed self the front is meeting itself, and div is set to 0
        n1s = np.stack([ndi.gaussian_filter(n1[..., a], 1.0) for a in range(3)], -1)
        agree = (n1 * n1s).sum(-1) / (np.linalg.norm(n1s, axis=-1) + 1e-12)
        n1n = n1s / (np.linalg.norm(n1s, axis=-1, keepdims=True) + 1e-12)
        div = sum(np.gradient(n1n[..., a], vox[a], axis=a) for a in range(3))
        T1f0 = fill(T1)
        good = ndi.binary_erosion(dom) & (agree > 0.8) & (T1f0 > 0.3)
        div = np.clip(np.where(good, div, 0.0), -2.0, 2.0)
        print(
            "div percentiles (1/mm) 2,25,50,75,98:",
            np.round(np.percentile(div[good], [2, 25, 50, 75, 98]), 2),
        )
        T1f = fill(T1)
        pts_ = np.argwhere(dom)
        order = pts_[np.argsort(T1f[dom])].astype(np.int64)
        u = upwind_u(order, dom, fixed, T1f, n1, div, np.array(vox, float))
        u = np.where(dom, u, np.nan)
        print(f"u (upwind): range {np.nanpercentile(u[ctx], [2, 98])}")
        uf = np.clip(fill(u), -2, 2)
        Fe = [np.exp(uf) * dirv[..., a] for a in range(3)]
        T, info = solve_field(dom, fixed, Fe, vox, x0=None)
        print(f"equivolume solve: cg info {info}")
    tag = "ev" if args.equivolume else "us"

    Tf = fill(T)
    gT = np.stack(np.gradient(Tf, *vox), -1)
    spd = np.where(ctx, np.linalg.norm(gT, axis=-1), np.nan)
    ng = gT / (np.linalg.norm(gT, axis=-1, keepdims=True) + 1e-12)
    ang = np.where(
        ctx, np.degrees(np.arccos(np.clip(np.abs((ng * nh).sum(-1)), 0, 1))), np.nan
    )
    # ridge: voxels where grad T points against neighbours across the midline sign flip
    ridge = ctx & (ndi.maximum_filter(sgn, 3) != ndi.minimum_filter(sgn, 3))
    print(
        f"angle(grad T, data normal) where coherence > 0.5: median "
        f"{np.nanmedian(ang[ctx & (coh > 0.5)]):.1f} deg; |grad T| median {np.nanmedian(spd):.2f}"
    )

    np.savez_compressed(
        f"{args.outdir}/state_{args.name}_{tag}.npz",
        T=T,
        u=(u if u is not None else np.array([])),
        lo=lo,
        root=np.array([-1]),
    )
    # ---------- slices view
    centre = ((P1[0] + P2[0]) // 2 - lo[0], (P1[1] + P2[1]) // 2 - lo[1], k - lo[2])
    planes = [(2, "k"), (0, "i"), (1, "j")]
    ncol = 5 if u is None else 6
    fig, axs = plt.subplots(3, ncol, figsize=(4.2 * ncol, 12.5))
    vden = np.nanpercentile(np.abs(disp[ctx]), 98)
    tmax = np.nanpercentile(T[ctx], 99)
    for r, (ax_, nm) in enumerate(planes):
        sl = [slice(None)] * 3
        sl[ax_] = centre[ax_]
        sl = tuple(sl)
        show = lambda a, x, **kw: a.imshow(
            np.asarray(x[sl]).T, origin="lower", interpolation="nearest", **kw
        )
        a = axs[r, 0]
        show(a, np.where(ctx, disp, np.nan), cmap="gray", vmin=-vden, vmax=vden)
        a.contour(
            np.where(ctx, T, np.nan)[sl].T,
            levels=np.arange(0.25, tmax, 0.25),
            cmap="turbo",
            linewidths=0.8,
        )
        a.contour(shell[sl].T.astype(float), levels=[0.5], colors="y", linewidths=0.7)
        a.contour(ridge[sl].T.astype(float), levels=[0.5], colors="m", linewidths=1.0)
        a.set_ylabel(f"slice {nm} = {centre[ax_] + lo[ax_]}", fontsize=12)
        a = axs[r, 1]
        show(a, np.where(ctx, disp, np.nan), cmap="gray", vmin=-vden, vmax=vden)
        a.set_title(
            f"echo {args.show_echo}, trend removed (no overlay)" if r == 0 else ""
        )
        a = axs[r, 2]
        im = show(a, ang, cmap="magma", vmin=0, vmax=45)
        if r == 0:
            fig.colorbar(
                im, ax=axs[:, 2], shrink=0.35, label="angle grad T vs data normal (deg)"
            )
        a = axs[r, 3]
        im = show(a, np.where(ctx, coh, np.nan), cmap="viridis", vmin=0, vmax=0.8)
        if r == 0:
            fig.colorbar(im, ax=axs[:, 3], shrink=0.35, label="coherence (data weight)")
        a = axs[r, 4]
        im = show(a, spd, cmap="coolwarm", vmin=0.5, vmax=1.5 if u is None else 2.5)
        if r == 0:
            fig.colorbar(im, ax=axs[:, 4], shrink=0.35, label="|grad T| (T per mm)")
        if u is not None:
            a = axs[r, 5]
            im = show(a, np.where(ctx, u, np.nan), cmap="PiYG", vmin=-0.6, vmax=0.6)
            if r == 0:
                fig.colorbar(im, ax=axs[:, 5], shrink=0.35, label="u = ln(A/A0)")
        for c in range(ncol):
            axs[r, c].set_xticks([])
            axs[r, c].set_yticks([])
    axs[0, 0].set_title(
        f"T contours every 0.25 over echo {args.show_echo} (trend removed);\nyellow WM, magenta ridge"
    )
    axs[0, 2].set_title("where the fit disagrees with the data")
    axs[0, 3].set_title("data confidence")
    axs[0, 4].set_title("speed of T")
    if u is not None:
        axs[0, 5].set_title("log area ratio u (equivolume)")
    fig.suptitle(
        f"3D fit of T ({'equivolume' if u is not None else 'unit speed'}), V1V2 {args.name} stretch: "
        "slices through the stretch centre",
        fontsize=14,
    )
    fig.savefig(
        f"{args.outdir}/fit_{args.name}_{tag}_e{args.show_echo}_slices.png",
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ---------- columns and flattened view
    path = F.boundary_path(
        shell[:, :, k - lo[2]],
        (P1[0] - lo[0], P1[1] - lo[1]),
        (P2[0] - lo[0], P2[1] - lo[1]),
    )
    arc = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))]) * vox[0]
    kk = k - lo[2]
    gTc = [gT[..., a] for a in range(3)]
    gwc = [gh[..., a] for a in range(3)]

    def trace(seed, comps, use_T):
        x = np.array(seed, float)
        d0 = np.array([F.interp(c, x) for c in gwc])
        d0 /= np.linalg.norm(d0)
        x = x + d0 * 1.0
        pts, s, prevT = [x.copy()], 1.0, -np.inf
        while s < 48:
            v1 = np.array([F.interp(c, x) for c in comps])
            if np.linalg.norm(v1) < 1e-9:
                return np.array(pts), "stall"
            v1 /= np.linalg.norm(v1)
            v2 = np.array([F.interp(c, x + 0.5 * STEP * v1) for c in comps])
            v2 /= np.linalg.norm(v2) + 1e-12
            x = x + STEP * v2
            s += STEP
            pts.append(x.copy())
            vv = tuple(np.clip(np.round(x).astype(int), 0, np.array(rb.shape) - 1))
            if missb[vv]:
                return np.array(pts), "missing"
            if rb[vv] == 1:
                return np.array(pts), "fluid"
            if use_T:
                tv = F.interp(Tf, x)
                if tv < prevT - 1e-3:
                    return np.array(pts), "ridge"
                prevT = max(prevT, tv)
            elif ridge[vv]:
                return np.array(pts), "ridge"
            if rb[vv] == 2 and s > 4:
                return np.array(pts), "WM"
            if rb[vv] != 3 and rb[vv] != 2:
                return np.array(pts), "other"
        return np.array(pts), "length"

    cols = {
        "fluid": "#1f78b4",
        "ridge": "#33a02c",
        "length": "#e31a1c",
        "stall": "#e31a1c",
        "WM": "#ff7f00",
        "other": "#6a3d9a",
        "missing": "#6a3d9a",
    }
    rows_ = [("grad T", gTc, True), ("WM distance", gwc, False)]
    fig, axs = plt.subplots(
        2, 4, figsize=(25, 10), gridspec_kw={"width_ratios": [1.1, 1, 1, 1]}
    )
    sl2 = disp[:, :, kk]
    for r, (nm, comps, useT) in enumerate(rows_):
        nb = 48
        img_T = np.full((nb, len(path)), np.nan)
        img_d = np.full((nb, len(path)), np.nan)
        img_n = np.full((40, len(path)), np.nan)
        counts = {}
        a = axs[r, 0]
        a.imshow(
            np.where(ctx[:, :, kk], sl2, np.nan).T,
            origin="lower",
            cmap="gray",
            vmin=-vden,
            vmax=vden,
        )
        a.contour(
            shell[:, :, kk].T.astype(float), levels=[0.5], colors="y", linewidths=0.7
        )
        for c, (i, j) in enumerate(path):
            pts, st = trace((i, j, kk), comps, useT)
            counts[st] = counts.get(st, 0) + 1
            a.plot(pts[:, 0], pts[:, 1], "-", c=cols[st], lw=0.8)
            seg = np.r_[
                0,
                np.cumsum(np.linalg.norm(np.diff(pts, axis=0) * np.array(vox), axis=1)),
            ]
            vals = ndi.map_coordinates(disp, pts.T, order=1)
            tv = ndi.map_coordinates(Tf, pts.T, order=1)
            q = np.arange(nb) * 0.1 + 0.05  # 0.1 mm or 0.1 T bins, to 4.8
            okd = q <= seg[-1]
            img_d[okd, c] = np.interp(q[okd], seg, vals)
            order = np.argsort(tv)
            okt = (q >= tv.min()) & (q <= tv.max())
            img_T[okt, c] = np.interp(q[okt], tv[order], vals[order])
            ttop = tv.max()
            if ttop > 0.5:
                qn = (np.arange(40) + 0.5) / 40 * ttop
                img_n[:, c] = np.interp(qn, tv[order], vals[order])
        a.plot(path[:, 0], path[:, 1], "y-", lw=1.8)
        a.set_xlim(path[:, 0].min() - 22, path[:, 0].max() + 28)
        a.set_ylim(path[:, 1].min() - 18, path[:, 1].max() + 18)
        a.set_title(
            f"{nm} columns (3D, projected): "
            + ", ".join(f"{k_} {v_}" for k_, v_ in counts.items())
        )
        for c_, (img, lab) in enumerate(
            ((img_T, "T"), (img_d, "depth along the column (mm)")), start=1
        ):
            axs[r, c_].imshow(
                img,
                origin="lower",
                aspect="auto",
                cmap="gray",
                vmin=-vden,
                vmax=vden,
                extent=[0, arc[-1], 0, nb * 0.1],
            )
            axs[r, c_].set_ylabel(lab)
            axs[r, c_].set_xlabel("position along the WM boundary (mm)")
            axs[r, c_].set_title(
                f"{nm} columns: echo {args.show_echo} (trend removed) by {lab.split(' (')[0]}"
            )
        a = axs[r, 3]
        a.imshow(
            img_n,
            origin="lower",
            aspect="auto",
            cmap="gray",
            vmin=-vden,
            vmax=vden,
            extent=[0, arc[-1], 0, 1],
        )
        a.set_ylabel("T / T at the column's top (0 WM, 1 pia or ridge)")
        a.set_xlabel("position along the WM boundary (mm)")
        a.set_title(f"{nm} columns: echo {args.show_echo} by normalized T")
        for yy in (0.25, 0.5, 0.75):
            a.axhline(yy, c="c", lw=0.4, ls=":")
        print(f"{nm} columns: {counts}")
    fig.suptitle(
        f"V1V2 {args.name} stretch, {'equivolume' if u is not None else 'unit-speed'} fit: "
        "columns traced in 3D along grad T, against WM-distance columns",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(
        f"{args.outdir}/fit_{args.name}_{tag}_e{args.show_echo}_flat.png", dpi=100
    )
    print("wrote views for", args.name, tag)


if __name__ == "__main__":
    main()
