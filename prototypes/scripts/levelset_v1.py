"""Open V1: does the stria sit at constant tau under the volume form? (test plan §5 item 9.)

Finding 10's setting exactly (ev_closed.py): the V1V2 box (full-grid corner 201, 70, 65 to 344, 214,
205), the open-V1 roots (WM-boundary voxels within 1 mm of the in-plane path, crop slice k = 54), root
curvature Hm at 0.48 mm (curv_lib.curvature), the stria as the darkest point of echo 2 (trend removed,
sampled every 0.25 voxel and smoothed by 1.5 samples) 0.5-1.6 mm from the WM, open columns only (they
end at fluid), >= 2 mm from the faces.

What changes: the columns follow the level set's own signed direction d (trilinear over claimed
voxels whose d agrees with the column's), from the root voxel's foot on the white mesh, and depth is
read from the level set's fields along them. Reported for each column at the stria:
- tau of the pass (posterior budgets) in the length and the volume form;
- the column's own fractions s / s_top and w / w_top (w from the volume run's field);
each with median, IQR and Spearman rho against Hm, beside finding 10 (distance 0.435, IQR 0.059,
rho +0.63; divergence equivolume 0.449, IQR 0.073, rho -0.67).

Usage: levelset_v1.py <figures dir> <work dir> <data dir>
"""

import json
import sys

fig_dir, work, data = sys.argv[1:4]
sys.argv = [sys.argv[0], fig_dir, work, data, "open_V1"]
sys.path.insert(0, work)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import spearmanr

import curv_lib as C
import levelset_face as LF
import levelset_fill as L
import levelset_mesh as LM

LO, HI = np.array((201, 70, 65)), np.array((344, 214, 205))
STEP = 0.25  # voxels, as curv_lib.STEP


def path_roots(B):
    """ev_closed.path_roots for open V1: WM-boundary voxels within 1 mm of the in-plane boundary path."""
    import flatten_fused2 as F2

    a = np.array((285, 126, 159)) - B["lo"]
    b = np.array((298, 168, 159)) - B["lo"]
    other = [0, 1]
    sl = (slice(None), slice(None), a[2])
    p2 = F2.boundary_path(B["shell"][sl], tuple(a[other]), tuple(b[other]))
    path = np.zeros((len(p2), 3), int)
    path[:, other] = p2
    path[:, 2] = a[2]
    m = np.zeros(B["rim"].shape, bool)
    m[tuple(path.T)] = True
    d = ndi.distance_transform_edt(~m, sampling=B["vox"])
    return np.argwhere(B["shell"] & (d <= 1.0))


def sample(f, field, p, dcur):
    """Trilinear over claimed corners whose d agrees with dcur; nan if none."""
    i0 = np.floor(p).astype(int)
    fr = p - i0
    tot, acc, dacc = 0.0, 0.0, np.zeros(3)
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                q = i0 + (a, b, c)
                if np.any(q < 0) or np.any(q >= np.array(f.shape)):
                    continue
                q = tuple(q)
                if f.state[q] != L.ACCEPTED or not f.cortex[q] or f.d[q] @ dcur <= 0:
                    continue
                wt = (
                    (fr[0] if a else 1 - fr[0])
                    * (fr[1] if b else 1 - fr[1])
                    * (fr[2] if c else 1 - fr[2])
                )
                tot += wt
                acc += wt * field[q]
                dacc += wt * f.d[q]
    if tot <= 1e-9:
        return np.nan, None
    return acc / tot, dacc / tot


def trace(F, fV, fLn, start, n0, disp):
    """From the foot (start) along the volume run's d. Returns per sample s (mm), w (mm), tau_len,
    tau_vol, echo 2; and whether the column ends at fluid."""
    h = F.h
    p = start + 0.05 * n0
    dcur = n0 / np.linalg.norm(n0)
    S, W, TL, TV, E = [], [], [], [], []
    s = 0.05 * h
    dmin = 99.0
    status = "lost"
    for _ in range(int(6.0 / (STEP * h))):
        w_, dv = sample(fV, fV.w, p, dcur)
        if dv is None:
            q = tuple(np.clip(np.round(p).astype(int), 0, np.array(F.wm.shape) - 1))
            status = "open" if F.fluid[q] else ("seam" if F.ctx[q] else "other")
            break
        tl, _ = sample(fLn, fLn.tau, p, dcur)
        tv, _ = sample(fV, fV.tau, p, dcur)
        S.append(s)
        W.append(w_ * h)
        TL.append(tl)
        TV.append(tv)
        E.append(ndi.map_coordinates(disp, p[:, None], order=1)[0])
        dmin = min(
            dmin,
            F.dface[
                tuple(np.clip(np.round(p).astype(int), 0, np.array(F.wm.shape) - 1))
            ],
        )
        dn = dv / (np.linalg.norm(dv) + 1e-12)
        p = p + STEP * dn
        s += STEP * h
        dcur = dn
    return (
        np.array(S),
        np.array(W),
        np.array(TL),
        np.array(TV),
        np.array(E),
        status,
        dmin,
    )


def main():
    lo, hi = LO, HI
    F = LF.Face.__new__(LF.Face)
    # Face.__init__ with the box given (no points)
    LF.box_and_points = lambda site: (lo, hi, np.array([[71.0, 72.0, 70.0]]))
    LF.Face.__init__(F, "open_V1")
    h = F.h
    roots = path_roots(F.B)
    K = C.curvature(F.B, 0.48)
    kidx = -np.ones(F.wm.shape, np.int64)
    kidx[tuple(K["pts"].T)] = np.arange(len(K["pts"]))
    LF.log(f"open V1: {len(roots)} roots")
    # evidence, field, posterior runs (as levelset_face)
    prior = np.exp(LM.M_MU + LM.A0_MU * F.H) / h
    f_ev = F.run(prior, L.LENGTH)
    E = LF.evidence(F, f_ev)
    fld = LM.MeshField(F.mesh, F.H, alpha=2)
    post = {k: LF.budgets(F, fld, E[k])["b"] / h for k in ("len", "vol")}
    fLn = F.run(post["len"], L.LENGTH)
    fV = F.run(post["vol"], L.VOLUME)
    disp = F.B["disp"]
    rows = []
    phi, foot, nrm, _, _ = LM.closest(F.mesh, roots.astype(float))
    for r, ft, n0 in zip(roots, foot, nrm):
        i = kidx[tuple(r)]
        if i < 0:
            continue
        Hm = 0.5 * (K["k1"][i] + K["k2"][i])
        S, W, TL, TV, Ech, st, dmin = trace(F, fV, fLn, ft, n0, disp)
        if st != "open" or len(S) < 8:
            continue
        e = ndi.gaussian_filter1d(Ech, 1.5)
        m = (S >= 0.5) & (S <= 1.6)
        if not m.any():
            continue
        ks = np.flatnonzero(m)[np.argmin(e[m])]
        rows.append(
            dict(
                Hm=float(Hm),
                s_top=float(S[-1]),
                w_top=float(W[-1]),
                fr_s=float(S[ks] / S[-1]),
                fr_w=float(W[ks] / W[-1]),
                tau_len=float(TL[ks]),
                tau_vol=float(TV[ks]),
                dface=float(min(dmin, F.dface[tuple(r)])),
            )
        )
    rows = [r for r in rows if r["dface"] >= 2.0]  # as ev_closed.usable
    h_ = np.array([r["Hm"] for r in rows])
    out = dict(columns=len(rows), roots=len(roots))
    for key, lab in (
        ("fr_s", "own top, distance s / s_top"),
        ("fr_w", "own top, volume w / w_top"),
        ("tau_len", "pass tau, length form (posterior)"),
        ("tau_vol", "pass tau, volume form (posterior)"),
    ):
        v = np.array([r[key] for r in rows])
        ok = np.isfinite(v)
        q = np.percentile(v[ok], [25, 50, 75])
        out[key] = dict(
            label=lab,
            median=float(q[1]),
            iqr=float(q[2] - q[0]),
            rho=float(spearmanr(h_[ok], v[ok]).statistic),
            n=int(ok.sum()),
        )
        LF.log(
            f"   {lab:38s} median {q[1]:.3f}  IQR {q[2] - q[0]:.3f}  rho(Hm) {out[key]['rho']:+.2f}  (n {ok.sum()})"
        )
    st = np.array([r["s_top"] for r in rows])
    out["s_top"] = dict(
        median=float(np.median(st)), rho=float(spearmanr(h_, st).statistic)
    )
    json.dump(dict(out, rows=rows), open(f"{work}/levelset_openV1.json", "w"), indent=1)
    fig, axs = plt.subplots(1, 4, figsize=(17, 4), sharey=True)
    for a, key in zip(axs, ("fr_s", "fr_w", "tau_len", "tau_vol")):
        v = np.array([r[key] for r in rows])
        a.plot(h_, v, ".", ms=3, color="#2a78d6")
        a.set_title(
            f"{out[key]['label']}\nmedian {out[key]['median']:.3f}, IQR {out[key]['iqr']:.3f}, "
            f"rho {out[key]['rho']:+.2f}",
            fontsize=9,
        )
        a.set_xlabel("root mean curvature Hm (/mm, 0.48 mm)")
    axs[0].set_ylabel("stria depth")
    fig.suptitle(
        f"Open V1 (finding 10's roots and stria), {len(rows)} open columns along the level set's d; "
        "finding 10: distance 0.435, rho +0.63",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/levelset_openV1.png", dpi=110)


if __name__ == "__main__":
    main()
