"""Cheap test: do the tops predicted from each bank's stria meet at the dark contact?

1. Calibrate on the open V1 stretch: along columns of the unit-speed least-squares
   T (traced in 3D up grad T to fluid), the stria's fraction of the column, both in
   distance (f_us = s_stria / s_top) and in equivolume (f_ev = v_stria / v_top, with
   v = integral of exp(u) ds and u = integral of div(n1) ds, n1 = grad T / |grad T|).
2. On the fused stretch: each path goes up grad T from bank A's WM to the ridge, then
   down -grad T to bank B's WM (each half is its bank's own column). Along it: echo 2
   trend removed, each bank's stria (darkest point 0.5-1.6 mm from its WM), the
   observed contact (darkest point between the two strias), and each bank's predicted
   top: where its own v (or s) reaches stria / f. Beyond the ridge a bank's u is held
   at its ridge value (its own columns end there).
"""

import sys

sys.path.insert(0, sys.argv[2])
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

import flatten_fused2 as F
from baillarger.data import load_patch
from baillarger.frame import missing_tissue

S = sys.argv[1] if len(sys.argv) > 1 else "."
STR = {"openV1": ((71, 89, 54), (84, 131, 54)), "fused": ((32, 78, 27), (49, 114, 27))}
STEP = 0.25


def setup(name, p, miss):
    P1, P2 = STR[name]
    rim, vox = p.rim, p.voxel_mm
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
    L = np.log(np.maximum(p.echoes[box + (1,)].astype(float), 1.0))
    disp = (L - F.mconv(L, mb, F.SHP)) * mb
    T = np.load(f"{S}/state_{name}_us.npz")["T"]
    bad = np.isnan(T)
    _, ind = ndi.distance_transform_edt(bad, return_indices=True)
    Tf = T[tuple(ind)]
    g = np.stack(np.gradient(Tf, *vox), -1)
    n1 = g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-12)
    n1s = np.stack([ndi.gaussian_filter(n1[..., a], 1.0) for a in range(3)], -1)
    agree = (n1 * n1s).sum(-1) / (np.linalg.norm(n1s, axis=-1) + 1e-12)
    n1n = n1s / (np.linalg.norm(n1s, axis=-1, keepdims=True) + 1e-12)
    div = sum(np.gradient(n1n[..., a], vox[a], axis=a) for a in range(3))
    div = np.clip(np.where((agree > 0.8) & (Tf > 0.3) & ctx, div, 0.0), -2, 2)
    kk = k - lo[2]
    path = F.boundary_path(
        shell[:, :, kk], (P1[0] - lo[0], P1[1] - lo[1]), (P2[0] - lo[0], P2[1] - lo[1])
    )
    arc = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))]) * vox[0]
    return dict(
        rb=rb,
        ctx=ctx,
        disp=disp,
        Tf=Tf,
        g=[g[..., a] for a in range(3)],
        div=div,
        path=path,
        arc=arc,
        kk=kk,
        vox=np.array(vox),
        wm=wm,
        shell=shell,
    )


def walk(D, x, sign, stop):
    pts = [x.copy()]
    prev = F.interp(D["Tf"], x)
    for _ in range(200):
        v = np.array([F.interp(c, x) for c in D["g"]]) * sign
        nv = np.linalg.norm(v)
        if nv < 1e-9:
            break
        x = x + STEP * v / nv
        vv = tuple(np.clip(np.round(x).astype(int), 0, np.array(D["rb"].shape) - 1))
        t = F.interp(D["Tf"], x)
        pts.append(x.copy())
        r = stop(vv, t, prev)
        if r:
            return np.array(pts), r
        prev = t
    return np.array(pts), "length"


def walk_dir(D, x, d0, stop):
    """Descend -grad T from x, rejecting any step that turns back against d0."""
    pts = [x.copy()]
    prev = F.interp(D["Tf"], x)
    d_prev = d0
    for _ in range(200):
        v = -np.array([F.interp(c, x) for c in D["g"]])
        nv = np.linalg.norm(v)
        if nv < 1e-9:
            break
        v /= nv
        if np.dot(v, d_prev) < 0:
            return np.array(pts), "turned back"
        x = x + STEP * v
        d_prev = v
        vv = tuple(np.clip(np.round(x).astype(int), 0, np.array(D["rb"].shape) - 1))
        t = F.interp(D["Tf"], x)
        pts.append(x.copy())
        r = stop(vv, t, prev)
        if r:
            return np.array(pts), r
        prev = t
    return np.array(pts), "length"


def profile(D, pts):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0) * D["vox"], axis=1))]
    e = ndi.gaussian_filter1d(ndi.map_coordinates(D["disp"], pts.T, order=1), 1.5)
    dv = ndi.map_coordinates(D["div"], pts.T, order=1)
    return s, e, dv


def vol(s, dv):
    """v(s) = int exp(u) ds, u = int div ds, from s = 0."""
    ds = np.diff(s, prepend=0.0)
    u = np.cumsum(dv * ds)
    return np.cumsum(np.exp(u) * ds), u


def stria(s, e, lo=0.5, hi=1.6):
    m = (s >= lo) & (s <= hi)
    if not m.any():
        return None
    i = np.flatnonzero(m)[np.argmin(e[m])]
    return i


def main():
    p = load_patch("V1V2")
    miss = missing_tissue(p.rim, p.flash, p.voxel_mm)
    # ---- 1. calibration on open V1
    D = setup("openV1", p, miss)
    fus, fev = [], []
    for i, j in D["path"]:
        x = np.array([i, j, D["kk"]], float)
        x = x + 1.0 * np.array([F.interp(c, x) for c in D["g"]]) / (
            np.linalg.norm([F.interp(c, x) for c in D["g"]]) + 1e-12
        )
        pts, st = walk(
            D,
            x,
            +1,
            lambda vv, t, pr: (
                "fluid" if D["rb"][vv] == 1 else ("ridge" if t < pr - 1e-3 else None)
            ),
        )
        if st != "fluid":
            continue
        s, e, dv = profile(D, pts)
        s = s + 0.16
        k_ = stria(s, e)
        if k_ is None:
            continue
        v, _ = vol(s, dv)
        fus.append(s[k_] / s[-1])
        fev.append(v[k_] / v[-1])
    fus, fev = np.array(fus), np.array(fev)
    print(
        f"calibration (open V1, {len(fus)} columns): stria fraction distance {np.median(fus):.3f} "
        f"(IQR {np.subtract(*np.percentile(fus, [75, 25])):.3f}), equivolume {np.median(fev):.3f} "
        f"(IQR {np.subtract(*np.percentile(fev, [75, 25])):.3f})"
    )
    f_us, f_ev = np.median(fus), np.median(fev)

    # ---- 2. fused stretch
    D = setup("fused", p, miss)
    rows = []
    paths = []
    for c, (i, j) in enumerate(D["path"]):
        x = np.array([i, j, D["kk"]], float)
        x = x + 1.0 * np.array([F.interp(cc, x) for cc in D["g"]]) / (
            np.linalg.norm([F.interp(cc, x) for cc in D["g"]]) + 1e-12
        )
        up, st = walk(
            D,
            x,
            +1,
            lambda vv, t, pr: (
                "fluid" if D["rb"][vv] == 1 else ("ridge" if t < pr - 1e-3 else None)
            ),
        )
        if st != "ridge":
            rows.append(None)
            paths.append(None)
            continue
        # cross the crest first: keep the incoming direction for 0.4 mm, so the descent
        # starts on bank B's slope, then descend without ever turning back towards A
        din = up[-1] - up[-2]
        din /= np.linalg.norm(din) + 1e-12
        xc = up[-1] + din * (0.4 / 0.16)
        down, st2 = walk_dir(
            D,
            xc,
            din,
            lambda vv, t, pr: (
                "WM" if D["wm"][vv] else ("fluid" if D["rb"][vv] == 1 else None)
            ),
        )
        down = np.vstack([up[-1:], down])
        if st2 != "WM":
            rows.append(None)
            paths.append(None)
            continue
        pts = np.vstack([up, down[1:]])
        s, e, dv = profile(D, pts)
        s = s + 0.16
        L = s[-1] + 0.16
        ir = len(up) - 1
        sr = s[ir]
        kA = stria(s, e)
        sB = L - s
        mB = (sB >= 0.5) & (sB <= 1.6)
        kB = np.flatnonzero(mB)[np.argmin(e[mB])] if mB.any() else None
        if kA is None or kB is None or s[kB] <= s[kA] + 1.0:
            rows.append(None)
            paths.append(pts)
            continue
        mid = (s > s[kA] + 0.5) & (s < s[kB] - 0.5)
        kc = np.flatnonzero(mid)[np.argmin(e[mid])] if mid.any() else None
        # bank A: own u up to the ridge, frozen beyond
        dvA = np.where(np.arange(len(s)) <= ir, dv, 0.0)
        vA, uA = vol(s, dvA)
        tgtA = vA[kA] / f_ev
        topA_ev = np.interp(tgtA, vA, s) if tgtA <= vA[-1] else np.nan
        # bank B: from its own WM backwards
        sb = (L - s)[::-1]
        dvB = np.where(np.arange(len(s)) >= ir, dv, 0.0)[::-1]
        vB, _ = vol(sb, dvB)
        kBr = len(s) - 1 - kB
        tgtB = vB[kBr] / f_ev
        topB_ev = L - np.interp(tgtB, vB, sb) if tgtB <= vB[-1] else np.nan
        topA_us = s[kA] / f_us
        topB_us = L - (L - s[kB]) / f_us
        rows.append(
            dict(
                L=L,
                ridge=sr,
                sA=s[kA],
                sB=s[kB],
                contact=s[kc] if kc is not None else np.nan,
                cdark=e[kc] if kc is not None else np.nan,
                topA_ev=topA_ev,
                topB_ev=topB_ev,
                topA_us=topA_us,
                topB_us=topB_us,
                s=s,
                e=e,
            )
        )
        paths.append(pts)

    ok = [r for r in rows if r is not None]
    print(
        f"fused: {len(ok)} of {len(rows)} columns give WM-to-WM paths with both strias"
    )
    if not ok:
        return
    keys = [
        "L",
        "ridge",
        "sA",
        "sB",
        "contact",
        "topA_ev",
        "topB_ev",
        "topA_us",
        "topB_us",
    ]
    A = {k_: np.array([r[k_] for r in ok]) for k_ in keys}
    for nm, (a_, b_) in {
        "equivolume": ("topA_ev", "topB_ev"),
        "distance": ("topA_us", "topB_us"),
    }.items():
        print(
            f"  {nm}: A's predicted top - contact median {np.nanmedian(A[a_] - A['contact']):+.2f} mm, "
            f"B's {np.nanmedian(A[b_] - A['contact']):+.2f} mm; gap between the two predictions "
            f"median {np.nanmedian(A[b_] - A[a_]):+.2f} mm"
        )
    print(
        f"  midline (ridge) - contact median {np.nanmedian(A['ridge'] - A['contact']):+.2f} mm; "
        f"WM-to-WM length median {np.median(A['L']):.2f} mm"
    )

    # ---- figure
    idx = [c for c, r in enumerate(rows) if r is not None]
    arc = D["arc"]
    nb = 70
    img = np.full((nb, len(rows)), np.nan)
    for c in idx:
        r = rows[c]
        q = np.arange(nb) * 0.08 + 0.04
        okq = q <= r["s"][-1]
        img[okq, c] = np.interp(q[okq], r["s"], r["e"])
    fig, axs = plt.subplots(
        1, 3, figsize=(24, 8), gridspec_kw={"width_ratios": [1, 1.5, 1.5]}
    )
    a = axs[0]
    kk = D["kk"]
    v = np.nanpercentile(np.abs(D["disp"][D["ctx"]]), 98)
    a.imshow(
        np.where(D["ctx"][:, :, kk], D["disp"][:, :, kk], np.nan).T,
        origin="lower",
        cmap="gray",
        vmin=-v,
        vmax=v,
    )
    a.contour(
        D["shell"][:, :, kk].T.astype(float), levels=[0.5], colors="y", linewidths=0.7
    )
    for pt in paths:
        if pt is not None:
            a.plot(pt[:, 0], pt[:, 1], "c-", lw=0.6)
    pa = D["path"]
    a.set_xlim(pa[:, 0].min() - 20, pa[:, 0].max() + 25)
    a.set_ylim(pa[:, 1].min() - 15, pa[:, 1].max() + 15)
    a.set_title(
        "WM-to-WM paths (3D, projected): up grad T to the ridge,\nthen down bank B's own columns"
    )
    for ax_, (nm, a_, b_) in zip(
        axs[1:],
        (("equivolume", "topA_ev", "topB_ev"), ("distance", "topA_us", "topB_us")),
    ):
        ax_.imshow(
            img,
            origin="lower",
            aspect="auto",
            cmap="gray",
            vmin=-v,
            vmax=v,
            extent=[0, arc[-1], 0, nb * 0.08],
        )
        xs = arc[idx]
        ax_.plot(
            xs,
            A["contact"],
            "o",
            mfc="none",
            mec="c",
            ms=7,
            label="observed contact (darkest point)",
        )
        ax_.plot(xs, A[a_], "v", c="#4a90d9", ms=6, label="bank A's predicted top")
        ax_.plot(xs, A[b_], "^", c="#e8913a", ms=6, label="bank B's predicted top")
        ax_.plot(xs, A["ridge"], "_", c="w", ms=12, mew=2, label="midline (ridge)")
        ax_.plot(xs, A["sA"], ".", c="#4a90d9", ms=5, label="bank A's stria")
        ax_.plot(xs, A["sB"], ".", c="#e8913a", ms=5, label="bank B's stria")
        ax_.set_xlabel("position along bank A's WM boundary (mm)")
        ax_.set_ylabel("distance from bank A's WM along the path (mm)")
        ax_.set_title(
            f"{nm}: each bank's top predicted from its stria (fraction "
            f"{f_ev if nm == 'equivolume' else f_us:.2f} from open V1)"
        )
        ax_.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "Fused stretch, V1V2 k = 27 (computed in 3D): echo 2 (trend removed) along WM-to-WM paths; "
        "do the tops predicted from each bank's stria meet at the dark contact?",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(f"{S}/cheap_test.png", dpi=100)
    print("wrote", f"{S}/cheap_test.png")


if __name__ == "__main__":
    main()
