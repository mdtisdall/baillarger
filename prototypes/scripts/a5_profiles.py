"""A5: does the contact have thickness? Image profiles across the contact at the user's 23 points, against
profiles across open pia on each front (no fitted field is used for the contact profiles).

All 3D, in the box of collide.py (the M1S1 test crop grown by 15 voxels), from the unclosed image.
1. Direction at each point: the normal of a plane fitted (PCA) to the user's neighbouring points, signed
   toward front 2's wall; compared with the data's layer normal (structure tensor) there.
2. Profiles from -2.5 to +2.5 mm along that direction, averaged over a disc of radius 0.3 mm around the
   point (19 parallel lines): echo 2 trend removed (the default display), raw log echo 2, raw log echo 3.
3. Per point: the darkest point of trend-removed echo 2 within +-0.8 mm of the user's point (its offset),
   its depth below the flanks (mean over 0.6-1.2 mm on both sides), the width at half depth, and that
   depth over the flanks' noise.
4. Reference: profiles across open pia on each front, at visible tops (collide2.py) 2-6 mm from any
   user point, along the front's own column direction at the top, aligned at the last cortex voxel and
   running into the fluid. Back to back (front 1's mirrored, front 2's), they show what two layer I's
   meeting with no gap would look like.

Usage: a5_profiles.py <outdir> <workdir>. Needs collide2_state.npz (for the reference tops only).
"""

import json
import sys

sys.path.insert(0, sys.argv[2])
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

import curv_lib as C
from collide import GROW
from fit_T import fill

OUT, S = sys.argv[1], sys.argv[2]
HALF, STEP, DISC = 2.5, 0.04, 0.3


def sample(img, x0, d, vox, ss, disc_offsets):
    """Mean of img along x0 + s d over parallel lines through the disc (voxel coords; d is a unit vector in mm)."""
    out = []
    for o in disc_offsets:
        line = x0[None] + (ss[:, None] * d[None] + o[None]) / vox
        out.append(ndi.map_coordinates(img, line.T, order=1, mode="nearest"))
    return np.mean(out, 0)


def disc(d, r, n_ring=6):
    a = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(d, a)
    offs = [np.zeros(3)]
    for rr in (r / 2, r):
        for t in np.linspace(
            0, 2 * np.pi, n_ring * (1 if rr < r else 2), endpoint=False
        ):
            offs.append(rr * (np.cos(t) * a + np.sin(t) * b))
    return offs


def main():
    m = json.load(open(f"{S}/M1S1_test_points.json"))
    lo = np.array(m["full_lo"]) - GROW
    hi = np.array(m["full_hi"]) + GROW
    B = C.load(lo, hi, "M1S1")
    vox = B["vox"]
    P = np.array(m["gap_all"], float) + GROW
    st = np.load(f"{S}/collide2_state.npz")
    cz = np.load(f"{S}/curv_M1S1.npz")
    pts = cz["pts"]
    w2 = np.zeros(B["rim"].shape, bool)
    w2[tuple(pts[st["wall2"]].T)] = True
    w1 = np.zeros(B["rim"].shape, bool)
    w1[tuple(pts[st["wall1"]].T)] = True
    # images
    det2 = B["disp"]
    raw2 = np.log(np.maximum(B["p"].echoes[..., 1].astype(float), 1.0))
    raw3 = np.log(np.maximum(B["p"].echoes[..., 2].astype(float), 1.0))
    nh, coh = C.tensor_normal(B)
    # 1. directions: PCA of neighbouring points, signed toward front 2's wall
    w2p = np.argwhere(w2)
    dirs, angs = [], []
    for q in P:
        dd = np.linalg.norm((P - q) * vox, axis=1)
        nb = P[np.argsort(dd)[: max(4, int((dd < 2.5).sum()))]]
        _, V = np.linalg.eigh(np.cov(((nb - nb.mean(0)) * vox).T))
        n_ = V[:, 0]
        to2 = (w2p[np.argmin(np.linalg.norm((w2p - q) * vox, axis=1))] - q) * vox
        if np.dot(n_, to2) < 0:
            n_ = -n_
        dirs.append(n_)
        dn = np.array(
            [
                ndi.map_coordinates(nh[..., a], q.reshape(3, 1), order=1)[0]
                for a in range(3)
            ]
        )
        dn /= np.linalg.norm(dn) + 1e-12
        angs.append(np.degrees(np.arccos(min(1.0, abs(np.dot(dn, n_))))))
    ss = np.arange(-HALF, HALF + 1e-9, STEP)
    prof = {k: [] for k in ("det2", "raw2", "raw3")}
    for q, d in zip(P, dirs):
        offs = disc(d, DISC)
        for k, img in (("det2", det2), ("raw2", raw2), ("raw3", raw3)):
            prof[k].append(sample(img, q, d, vox, ss, offs))
    prof = {k: np.array(v) for k, v in prof.items()}
    # 3. dark line per point (trend-removed echo 2)
    rows = []
    for k_, pr in enumerate(prof["det2"]):
        win = np.abs(ss) <= 0.8
        i0 = np.flatnonzero(win)[np.argmin(pr[win])]
        fl = ((ss - ss[i0] >= 0.6) & (ss - ss[i0] <= 1.2)) | (
            (ss[i0] - ss >= 0.6) & (ss[i0] - ss <= 1.2)
        )
        base = pr[fl].mean()
        noise = pr[fl].std()
        depth = base - pr[i0]
        half = pr[i0] + depth / 2
        j = i0
        while j > 0 and pr[j] < half:
            j -= 1
        l = i0
        while l < len(ss) - 1 and pr[l] < half:
            l += 1
        rows.append(
            dict(
                k=k_,
                off=ss[i0],
                depth=depth,
                width=ss[l] - ss[j],
                snr=depth / (noise + 1e-12),
                ang=angs[k_],
            )
        )
    print(
        "point: darkest point offset from your point (mm, + toward front 2) | depth below the flanks | width at half depth (mm) | "
        "depth / flank noise | angle between the contact normal and the data's layer normal (deg)"
    )
    for r in rows:
        print(
            f"   {r['k'] + 1:2d}: {r['off']:+.2f} | {r['depth']:.3f} | {r['width']:.2f} | {r['snr']:.1f} | {r['ang']:.0f}"
        )
    for key in ("off", "width", "snr", "ang"):
        v = np.array([r[key] for r in rows])
        print(
            f"   {key}: median {np.median(v):.2f}, IQR {np.subtract(*np.percentile(v, [75, 25])):.2f}"
        )
    # 4. reference profiles across open pia
    T = {X: fill(st[f"T{X}"].astype(float)) for X in ("1", "2")}
    ref = {}
    ev = st["ev_src"]
    dP = ndi.distance_transform_edt(~_pts_mask(P, B["rim"].shape), sampling=vox)
    fluid = B["rim"] == 1
    for X in ("1", "2"):
        tq = np.argwhere((ev == int(X)) & (dP >= 2.0) & (dP <= 6.0))
        rng = np.random.default_rng(0)
        tq = tq[rng.permutation(len(tq))[:300]]
        g = np.gradient(T[X], *vox)
        prs = {k: [] for k in ("det2", "raw2", "raw3")}
        for t in tq.astype(float):
            d = np.array(
                [ndi.map_coordinates(c, t.reshape(3, 1), order=1)[0] for c in g]
            )
            d /= np.linalg.norm(d) + 1e-12
            # align at the cortex/fluid boundary: walk outward to the first fluid sample
            line = t[None] + (np.arange(0, 1.5, STEP)[:, None] * d[None]) / vox
            f = ndi.map_coordinates(fluid.astype(float), line.T, order=0) > 0
            if not f.any():
                continue
            s0 = np.arange(0, 1.5, STEP)[np.argmax(f)] - STEP / 2
            for k, img in (
                ("det2", np.where(fluid, np.nan, det2)),
                ("raw2", raw2),
                ("raw3", raw3),
            ):
                v = sample(
                    np.nan_to_num(img, nan=0.0) if k == "det2" else img,
                    t + s0 * d / vox,
                    d,
                    vox,
                    ss,
                    disc(d, DISC),
                )
                if k == "det2":
                    v = np.where(
                        ss > 0, np.nan, v
                    )  # trend-removed echo 2 is undefined in fluid
                prs[k].append(v)
        ref[X] = {k: np.array(v) for k, v in prs.items()}
        print(
            f"reference, front {X}: {len(ref[X]['raw2'])} open-pia profiles (tops 2-6 mm from your points)"
        )
    np.save(f"{S}/a5_rows.npy", rows, allow_pickle=True)

    # ---------- views
    fig = plt.figure(figsize=(30, 15))
    gs = fig.add_gridspec(2, 4)
    order = np.arange(len(P))
    for c_, (k, lab) in enumerate(
        (
            ("det2", "echo 2, trend removed"),
            ("raw2", "log echo 2"),
            ("raw3", "log echo 3"),
        )
    ):
        a = fig.add_subplot(gs[0, c_])
        img = prof[k]
        v = np.nanpercentile(np.abs(img - np.nanmedian(img)), 98)
        a.imshow(
            img[order],
            aspect="auto",
            cmap="gray",
            extent=[-HALF, HALF, len(P) + 0.5, 0.5],
            vmin=np.nanmedian(img) - v,
            vmax=np.nanmedian(img) + v,
            interpolation="nearest",
        )
        a.axvline(0, c="#ffd23f", lw=0.8)
        if k == "det2":
            a.plot(
                [r["off"] for r in rows],
                np.arange(1, len(P) + 1),
                "o",
                mfc="none",
                mec="c",
                ms=6,
                label="darkest within 0.8 mm",
            )
            a.legend(fontsize=8, loc="lower right")
        a.set_xlabel(
            "position across the contact (mm; 0 = your point; + toward front 2's WM)"
        )
        a.set_ylabel("your contact point")
        a.set_title(f"{lab}: profiles across the contact (disc-averaged)", fontsize=12)
    a = fig.add_subplot(gs[0, 3])
    for r in rows:
        a.plot(r["ang"], r["width"], "o", c="k")
        a.annotate(
            str(r["k"] + 1),
            (r["ang"], r["width"]),
            fontsize=8,
            xytext=(3, 3),
            textcoords="offset points",
        )
    a.set_xlabel(
        "angle between the contact normal (your points) and the data's layer normal (deg)"
    )
    a.set_ylabel("dark-line width at half depth (mm)")
    a.set_title("width against obliquity: an oblique profile widens a line")
    for c_, (k, lab) in enumerate(
        (
            ("det2", "echo 2, trend removed"),
            ("raw2", "log echo 2"),
            ("raw3", "log echo 3"),
        )
    ):
        a = fig.add_subplot(gs[1, c_])
        mprof = prof[k]
        a.fill_between(
            ss,
            np.nanpercentile(mprof, 25, 0),
            np.nanpercentile(mprof, 75, 0),
            color="k",
            alpha=0.15,
        )
        a.plot(
            ss,
            np.nanmedian(mprof, 0),
            "k-",
            lw=2,
            label="contact, median of the 23 points (IQR shaded)",
        )
        if k != "det2":
            # back-to-back pia: front 1's profile mirrored (its fluid side toward +), front 2's as is (fluid toward -)
            r1 = np.nanmedian(ref["1"][k], 0)
            r2 = np.nanmedian(ref["2"][k], 0)
            a.plot(
                ss,
                r1,
                "-",
                c="#2b7bd9",
                alpha=0.8,
                label="front 1, open pia (0 = the cortex/fluid boundary; fluid +)",
            )
            a.plot(
                -ss,
                r2,
                "-",
                c="#e8913a",
                alpha=0.8,
                label="front 2, open pia, mirrored (fluid -)",
            )
            comp = np.where(ss < 0, r1, np.interp(-ss, ss, r2))
            a.plot(
                ss,
                comp,
                "--",
                c="m",
                lw=1.5,
                label="two pia back to back, no gap (front 1 side -, front 2 side +)",
            )
        else:
            r1 = np.nanmedian(ref["1"][k], 0)
            r2 = np.nanmedian(ref["2"][k], 0)
            a.plot(
                ss,
                r1,
                "-",
                c="#2b7bd9",
                alpha=0.8,
                label="front 1, open pia (cortex side only)",
            )
            a.plot(
                -ss,
                r2,
                "-",
                c="#e8913a",
                alpha=0.8,
                label="front 2, open pia, mirrored (cortex side only)",
            )
        a.axvline(0, c="#ffd23f", lw=0.8)
        a.set_xlabel("mm")
        a.set_title(f"{lab}: the contact against two layer I's meeting", fontsize=12)
        a.legend(fontsize=7)
    a = fig.add_subplot(gs[1, 3])
    a.axis("off")
    txt = [
        f"{r['k'] + 1:2d}: offset {r['off']:+.2f}  width {r['width']:.2f}  depth/noise {r['snr']:.1f}  angle {r['ang']:.0f}"
        for r in rows
    ]
    a.text(0, 1, "\n".join(txt), va="top", family="monospace", fontsize=9)
    a.set_title("per point (trend-removed echo 2)")
    fig.suptitle(
        "A5: does the contact have thickness? Profiles across your 23 points along the contact's own normal "
        "(from your points), against open pia on each front (all 3D; no fitted field used across the contact)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(f"{OUT}/a5_profiles.png", dpi=72)
    plt.close(fig)
    print("wrote a5_profiles.png")


def _pts_mask(P, shape):
    mm = np.zeros(shape, bool)
    mm[tuple(np.round(P).astype(int).T)] = True
    return mm


if __name__ == "__main__":
    main()
