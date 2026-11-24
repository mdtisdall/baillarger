"""The contoured pial base (pial_contour_base.py, slit2) rebuilt from a weighted one pass, with
and without the 23 marked M1S1 contact points as click evidence (click_evidence.py). All 3D.

Three bases on the M1S1 patch, all carved and contoured as pial_contour_base.py (contacts with
roots > 18 voxels apart along the white surface, not fold-like, >= 2 mm from a box face; slit2;
marching tetrahedra, tangential relaxation, twicing at 0.32 mm):
- equal: the editor's current base (pial_M1S1_contour_slit2_session.npz), read, not rebuilt;
- field: the thickness field's weights from column tops (finding 25's field weights);
- clicks: the same field with every point's click evidence added.
The contact classes use each pass's own beliefs (equal: the column tops, as the current base).

At each point: the distance to the carved contact's midline (halfway between the nearest sheet
and the nearest sheet facing it within 1.5 mm, along the nearest sheet's normal); where no
facing sheet is near, the contact is not carved there. In sample: the
clicks base saw every point; click_evidence.py gives the held-out numbers.

Writes, in <data dir>: pial_M1S1_contour_{field,clicks}_session.npz (editor sessions, same
format as the current base) and .vtk; <work>/click_base_M1S1.json; <figures>/click_base_M1S1.png
(the three bases on the planes through the points).

Usage: click_base.py <figures dir> <work dir> <data dir>
"""

import json
import sys
import time

sys.path.insert(0, sys.argv[2])
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

import curv_lib as C
import pial_height as P_
from baillarger.data import PATCHES
from baillarger.mesh import topology, write_vtk
from baillarger.mesh_intersect import intersecting_faces
from click_evidence import RATIO, PATCH, FACE_MM, Thickness, end_labels, orientation, score
from onepass_lib import contacts, run, setup

FIG, WORK, DATA = sys.argv[1:4]
SEP = 18.0
T0 = time.time()


def say(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def build(B, s, mu, sd, name, white):
    """Carve the pass's contacts (slit2) and contour; the session file and the counts."""
    f = run(s, mu, sd, ratio=RATIO)
    c = contacts(s, f, mu, sd, sep_min=SEP)
    ra, rb = c.da / mu[c.a], c.db / mu[c.b]
    fold = (ra <= 0.8) & (rb <= 0.8)
    near = (B["dface"].ravel()[c.p] < FACE_MM) | (B["dface"].ravel()[c.q] < FACE_MM)
    carve = ~fold & ~near
    tissue = (B["rim"] == 2) | (B["rim"] == 3) | B["miss"]
    tissue.ravel()[np.concatenate([c.p[carve], c.q[carve]])] = False
    v, fc, n = P_.white_surface(tissue)
    vi = np.clip(np.round(v).astype(int), 0, np.array(tissue.shape) - 1)
    nf = B["dface"][tuple(vi.T)] < FACE_MM
    xf, _ = intersecting_faces(v, fc)
    aff = B["p"].affine
    np.savez(f"{DATA}/pial_M1S1_contour_{name}_session.npz", kind="surface", white_v=white[0], white_f=white[1],
             v=v, f=fc, n=n, near_face=nf, affine=aff)
    write_vtk(f"{DATA}/pial_M1S1_contour_{name}.vtk", aff, v, fc, {"near_box_face": nf.astype(float)})
    topo = topology(fc, len(v))
    return (v, fc, n), dict(pairs=int(c.p.size), carved=int(carve.sum()), fold_like=int(fold.sum()),
                            vertices=int(len(v)), intersecting_faces=int(xf.sum()),
                            genus=topo.get("genus"), components=topo.get("components"))


def at_points(mesh, P, h):
    """Per point: d1, the nearest sheet (mm); and where a sheet facing it (vertex normals dot
    < -0.5) lies within 1.5 mm, the distance from the point to the carved contact's midline,
    along the nearest sheet's normal (the midline: halfway between the two sheets; the point may
    lie between them or behind either); None where no facing sheet is near (not carved)."""
    v, f, n = mesh
    tree = cKDTree(v)
    out = []
    for p in P:
        d, i = tree.query(p)
        cand = np.array(tree.query_ball_point(p, 1.5 / h), int)
        cand = cand[n[cand] @ n[i] < -0.5]
        mid = None
        if cand.size:
            j = cand[np.argmin(np.linalg.norm(v[cand] - p, axis=1))]
            mid = round(abs((p - v[i]) @ n[i] - 0.5 * ((v[j] - v[i]) @ n[i])) * h, 3)
        out.append(dict(d1_mm=round(float(d) * h, 3), midline_mm=mid))
    return out


def main():
    lo, hi = (np.array(x) for x in PATCHES["M1S1"])
    m = json.load(open(f"{DATA}/M1S1_test_points.json"))
    Pts = (np.array(m["gap_all"]) + np.array(m["full_lo"]) - lo).astype(float)
    B = C.load(lo, hi, "click_base_M1S1")
    h = float(np.mean(B["vox"]))
    dirv = orientation(B)
    s = setup(B["wm"], B["ctx"], patch=PATCH, direction=dirv)
    u_pts = dirv[tuple(np.round(Pts).astype(int).T)]
    del dirv
    say(f"patch {tuple(hi - lo)}; patches {s.npatch:,}")
    T = Thickness(B, s, h)
    mu0, sd0, nob0 = T.weights(T.y_tops)
    mu_eq = np.full(s.npatch, float(np.median(mu0))); sd_eq = 0.1 * mu_eq
    f_eq = run(s, mu_eq, sd_eq, ratio=RATIO)
    rows = score(s, f_eq, mu_eq, sd_eq, Pts, u_pts, h, np.arange(len(Pts)))
    clicks = []
    for (lm, lp), r in zip(end_labels(s, f_eq, Pts, u_pts, h), rows):
        if lm >= 0 and np.isfinite(r["S_thin"]) and r["S_thin"] > 0.3 and r["S_thick"] > 0.3:
            clicks += [(int(lm), float(r["S_thick"])), (int(lp), float(r["S_thin"]))]
    say(f"clicks: {len(clicks) // 2} of {len(Pts)} points with both sides")
    mu1, sd1, nob1 = T.weights(T.with_clicks(clicks))
    z = np.load(f"{DATA}/pial_M1S1_session.npz")
    white = (z["v"], z["f"])
    del z
    zc = np.load(f"{DATA}/pial_M1S1_contour_slit2_session.npz")
    meshes = {"equal": (zc["v"], zc["f"], zc["n"])}
    log = dict(points=len(Pts), click_points=len(clicks) // 2, no_belief=dict(field=nob0, clicks=nob1), bases={})
    for name, (mu, sd) in (("field", (mu0, sd0)), ("clicks", (mu1, sd1))):
        meshes[name], log["bases"][name] = build(B, s, mu, sd, name, white)
        say(f"{name}: {log['bases'][name]}")
    log["at_points"] = {k: at_points(mesh, Pts, h) for k, mesh in meshes.items()}
    for k, rows_ in log["at_points"].items():
        mid = np.array([r["midline_mm"] if r["midline_mm"] is not None else np.nan for r in rows_])
        log.setdefault("summary", {})[k] = dict(carved_at=int(np.isfinite(mid).sum()),
                                                midline_abs_median=round(float(np.nanmedian(np.abs(mid))), 3),
                                                midline_abs_p90=round(float(np.nanpercentile(np.abs(mid), 90)), 3),
                                                d1_median=round(float(np.median([r["d1_mm"] for r in rows_])), 3))
        print(f"   {k:7s} {log['summary'][k]}", flush=True)
    json.dump(log, open(f"{WORK}/click_base_M1S1.json", "w"), indent=1, default=float)

    # views: the three planes holding most points, the bases' sections over echo 2 (trend removed)
    img = np.where(B["ctx"], B["disp"], np.nan)
    lim = np.nanpercentile(np.abs(img), 98)
    cols = {"equal": "#8a8984", "field": "#2a78d6", "clicks": "#eb6834"}
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.4))
    for a, ax in enumerate(axs):
        vals, cnt = np.unique(np.round(Pts[:, a]).astype(int), return_counts=True)
        c = vals[np.argmax(cnt)]
        o = [i for i in range(3) if i != a]
        sl = [slice(None)] * 3; sl[a] = c
        ax.imshow(img[tuple(sl)].T, cmap="gray", origin="lower", vmin=-lim, vmax=lim, interpolation="nearest")
        for k, (v, fc, _) in meshes.items():
            for sg in P_.plane_section(v, fc, a, float(c)):
                ax.plot(sg[:, o[0]], sg[:, o[1]], color=cols[k], lw=0.9)
        on = np.abs(Pts[:, a] - c) < 1.5
        ax.plot(Pts[on, o[0]], Pts[on, o[1]], "x", color="#e34948", ms=7, mew=2)
        lo_, hi_ = Pts[:, o].min(0) - 25, Pts[:, o].max(0) + 25
        ax.set_xlim(lo_[0], hi_[0]); ax.set_ylim(lo_[1], hi_[1])
        ax.set_title(f"plane {'ijk'[a]} = {c} (patch voxels)", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("M1S1 contoured pial base: equal arrival (grey, the editor's current base), field weights (blue), "
                 "field + the 23 points as clicks (orange); points (red x); echo 2, trend removed", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/click_base_M1S1.png", dpi=110)
    say("done")


if __name__ == "__main__":
    main()
