"""What each bank computes, across the fused contact (k = 27 stretch).

For each method (state_*.npz from fit_T.py / prop_T.py), all computed in 3D:
  row of maps in the stretch slice: which bank each voxel's front came from
  (root on the WM; for the least-squares fit, the nearest WM voxel), T with
  contours, u (equivolume runs), and the angle between grad T and the data normal;
  plus profiles along one straight transect from bank A's WM, through the contact,
  to bank B's WM: echo 2 (trend removed), T, u, the angle, coherence, with the
  bank of each sample as a coloured band.
"""

import sys

sys.path.insert(0, sys.argv[2])
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy import ndimage as ndi

import flatten_fused2 as F
from baillarger.data import load_patch
from baillarger.frame import missing_tissue

S = sys.argv[1]
P1, P2 = (32, 78, 27), (49, 114, 27)
RUNS = [
    ("fused_us", "least-squares fit, unit speed"),
    ("fused_prop_us", "front propagation, distance"),
    ("fused_prop_ev", "front propagation, equivolume"),
]


def main():
    p = load_patch("V1V2")
    rim, vox = p.rim, p.voxel_mm
    vv = np.array(vox)
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
    e = {i: np.log(np.maximum(p.echoes[box + (i,)].astype(float), 1.0)) for i in (1, 2)}
    det = {i: (L - F.mconv(L, mb, F.SHP)) * mb for i, L in e.items()}
    Tc = F.tensor(det[1], mb) + F.tensor(det[2], mb)
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
    disp = det[1]
    kk = k - lo[2]
    path = F.boundary_path(
        shell[:, :, kk], (P1[0] - lo[0], P1[1] - lo[1]), (P2[0] - lo[0], P2[1] - lo[1])
    )
    _, near = ndi.distance_transform_edt(~wm, sampling=vox, return_indices=True)

    # transect: from the middle seed, straight along the local data normal (signed away
    # from bank A's WM), until it reaches WM again (bank B) or fluid
    seed = np.array([path[len(path) // 2][0], path[len(path) // 2][1], kk], float)
    d_wm = ndi.distance_transform_edt(~wm, sampling=vox)
    g = np.array(np.gradient(ndi.gaussian_filter(d_wm, 1.0), *vox))
    x0 = seed + 2.0 * np.array([F.interp(c, seed) for c in g]) / np.linalg.norm(
        [F.interp(c, seed) for c in g]
    )
    nd = F.normal_at(Tc, x0)
    gd = np.array([F.interp(c, x0) for c in g])
    if np.dot(nd, gd) < 0:
        nd = -nd
    pts = [seed]
    x = seed.copy()
    for _ in range(400):
        x = x + 0.25 * nd
        if np.any(np.round(x) < 0) or np.any(np.round(x) >= np.array(rb.shape)):
            break
        v = tuple(np.round(x).astype(int))
        pts.append(x.copy())
        if (rb[v] == 2 and len(pts) > 8) or rb[v] == 1:
            break
    pts = np.array(pts)
    sx = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0) * vv, axis=1))]
    ve = tuple(np.clip(np.round(pts[-1]).astype(int), 0, np.array(rb.shape) - 1))
    end_label = {2: "bank B WM", 1: "fluid"}.get(int(rb[ve]), "box edge")
    print("transect direction (voxels):", np.round(nd, 2), "start", seed)
    print(f"transect {sx[-1]:.2f} mm, ends at {end_label}")

    seedA = np.array(np.unravel_index(0, rb.shape))  # placeholder
    nrun = len(RUNS)
    fig = plt.figure(figsize=(24, 6.2 * nrun + 5))
    gs = fig.add_gridspec(nrun + 1, 5, height_ratios=[1] * nrun + [0.9])
    bank_cmap = ListedColormap(["#4a90d9", "#e8913a"])
    profiles = []
    for r, (tag, title) in enumerate(RUNS):
        st = np.load(f"{S}/state_{tag}.npz")
        T, u, root = st["T"], st["u"], st["root"]
        if root.size > 1:
            rc = np.stack(
                np.unravel_index(np.where(root >= 0, root, 0), rb.shape), -1
            ).reshape(rb.shape + (3,))
        else:
            rc = np.stack(near, -1)
        # bank: which side of the transect's midpoint plane the root lies on
        mid = pts[len(pts) // 2]
        side = ((rc - mid) * (nd * vv)).sum(-1) > 0
        bank = np.where(ctx, side.astype(float), np.nan)
        Tf = np.where(np.isnan(T), 0, T)
        gT = np.stack(np.gradient(np.where(np.isnan(T), np.nanmean(T), T), *vox), -1)
        ng = gT / (np.linalg.norm(gT, axis=-1, keepdims=True) + 1e-12)
        ang = np.where(
            ctx, np.degrees(np.arccos(np.clip(np.abs((ng * nh).sum(-1)), 0, 1))), np.nan
        )
        sl = (slice(None), slice(None), kk)
        xl = (path[:, 0].min() - 20, path[:, 0].max() + 25)
        yl = (path[:, 1].min() - 15, path[:, 1].max() + 15)
        axs = [fig.add_subplot(gs[r, c]) for c in range(5)]
        vden = np.nanpercentile(np.abs(disp[ctx]), 98)
        axs[0].imshow(
            np.where(ctx, disp, np.nan)[sl].T,
            origin="lower",
            cmap="gray",
            vmin=-vden,
            vmax=vden,
        )
        axs[0].imshow(
            bank[sl].T, origin="lower", cmap=bank_cmap, alpha=0.45, vmin=0, vmax=1
        )
        axs[0].set_title(
            f"{title}\nbank each voxel's front came from (blue A, orange B)"
        )
        im = axs[1].imshow(
            np.where(ctx, T, np.nan)[sl].T, origin="lower", cmap="viridis"
        )
        axs[1].contour(
            np.where(ctx, T, np.nan)[sl].T,
            levels=np.arange(0.25, 6, 0.25),
            colors="w",
            linewidths=0.5,
        )
        fig.colorbar(im, ax=axs[1], shrink=0.7)
        axs[1].set_title("T (contours every 0.25)")
        if u.size > 1:
            im = axs[2].imshow(
                np.where(ctx, u, np.nan)[sl].T,
                origin="lower",
                cmap="PiYG",
                vmin=-1,
                vmax=1,
            )
            fig.colorbar(im, ax=axs[2], shrink=0.7)
            axs[2].set_title("u = ln(A/A0)")
        else:
            axs[2].imshow(
                np.where(ctx, disp, np.nan)[sl].T,
                origin="lower",
                cmap="gray",
                vmin=-vden,
                vmax=vden,
            )
            axs[2].set_title("(no u: echo 2 shown)")
        im = axs[3].imshow(ang[sl].T, origin="lower", cmap="magma", vmin=0, vmax=60)
        fig.colorbar(im, ax=axs[3], shrink=0.7)
        axs[3].set_title("angle grad T vs data normal (deg)")
        im = axs[4].imshow(
            np.where(ctx, coh, np.nan)[sl].T,
            origin="lower",
            cmap="viridis",
            vmin=0,
            vmax=0.8,
        )
        fig.colorbar(im, ax=axs[4], shrink=0.7)
        axs[4].set_title("data confidence (coherence)")
        for a in axs:
            a.contour(
                shell[sl].T.astype(float), levels=[0.5], colors="y", linewidths=0.7
            )
            a.plot(pts[:, 0], pts[:, 1], "c-", lw=1.6)
            a.plot(pts[0, 0], pts[0, 1], "co", ms=5)
            a.set_xlim(*xl)
            a.set_ylim(*yl)
            a.set_xticks([])
            a.set_yticks([])
        prof = dict(
            T=ndi.map_coordinates(Tf, pts.T, order=1),
            u=ndi.map_coordinates(np.where(np.isnan(u), 0, u), pts.T, order=1)
            if u.size > 1
            else None,
            ang=ndi.map_coordinates(np.where(np.isnan(ang), 0, ang), pts.T, order=1),
            bank=ndi.map_coordinates(
                np.where(np.isnan(bank), 0.5, bank), pts.T, order=0
            ),
        )
        profiles.append((title, prof))

    # profiles along the transect
    gsb = gs[nrun, :].subgridspec(1, 4)
    ax = [fig.add_subplot(gsb[0, c]) for c in range(4)]
    echo = ndi.map_coordinates(disp, pts.T, order=1)
    cohp = ndi.map_coordinates(coh, pts.T, order=1)
    ax[0].plot(sx, echo, "k-")
    ax[0].set_title("echo 2 (trend removed) along the transect")
    ax0b = ax[0].twinx()
    ax0b.plot(sx, cohp, "g-", lw=0.8)
    ax0b.set_ylabel("coherence", color="g")
    cols = ["#1b9e77", "#d95f02", "#7570b3"]
    for (title, pr), c in zip(profiles, cols):
        ax[1].plot(sx, pr["T"], "-", c=c, label=title)
        if pr["u"] is not None:
            ax[2].plot(sx, pr["u"], "-", c=c, label=title)
        ax[3].plot(sx, pr["ang"], "-", c=c, label=title)
    ax[1].set_title(
        "T along the transect (should rise from both WMs to a peak at the contact)"
    )
    ax[2].set_title("u along the transect")
    ax[3].set_title("angle grad T vs data normal (deg)")
    for a in ax:
        a.set_xlabel(f"distance from bank A's WM (mm); transect ends at {end_label}")
        # bank band from the first run's assignment
    for (title, pr), c in zip(profiles, cols):
        b = pr["bank"]
        ch = np.flatnonzero(np.diff(b) != 0)
        for i in ch:
            ax[1].axvline(sx[i], c=c, ls=":", lw=1)
    ax[1].legend(fontsize=8)
    fig.suptitle(
        "Fused stretch, V1V2 slice k = 27 (all computed in 3D): what each bank computes; "
        "dotted lines on the T panel = where each method switches bank along the transect",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(f"{S}/two_bank_view.png", dpi=95)
    print("wrote", f"{S}/two_bank_view.png")


if __name__ == "__main__":
    main()
