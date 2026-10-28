"""The column volume of a box (plan: reports/Crossing paths and contacts from laminar profiles.md,
*The column volume*). All 3D: samples keep their 3D positions; the volume only re-indexes them.

1. White mesh as finding 28 (contour of the WM label, relaxed, twicing at 2 voxels, smoothed
   outward normals). Traces at one vertex per voxel cell and normal side (the vertex nearest the
   cell's centre), so that they are about a voxel apart.
2. Primary tracer: the streamline of the unsigned layer normal from the vertex along its normal
   (crossing_lib.trace_array), up to 8 mm. Where it turns or reaches the cap, the fallback: ascent
   of the equal-arrival pass's arrival (crossing_lib.ascend), stopped at fluid, WM, or the pass's
   contact (a label whose root is > SAME_MM from the trace's own along the white surface).
3. Classes: open (fluid), crossing (WM whose root is >= SAME_MM away), own bank (WM within it),
   to contact (fallback stopped at the pass's contact), failed. Roots within 2 mm of a face: flagged.
4. Features: echo 2 and echo 3 (trend removed) at each sample, centre line, no disc.
5. Horizons (mm from the root): top; far WM; the fallback's contact; on crossing traces the one
   pass's boundary (equal and field weights), the field prior's pairwise posterior on L, and the
   profile's split (E1 top-anchored) on a random subsample of crossing traces.
6. With points (optional): every trace passing within 0.3 mm of a point gives the point's depth
   along it; each horizon minus that depth is scored.
7. Views: roots by class on three planes, where primary traces turn; with points, sections along
   the white surface through them (the three planes holding most points, two banks each).

Any box of the full grid: --patch NAME (baillarger.data.PATCHES) or --box I J K (the full-grid
corner, 0-based) with --size N (or N1 N2 N3; default 141). Points: --points FILE, a JSON object
with "points_full0": [[i, j, k], ...] in 0-based full-grid voxels.
Writes <work>/column_volume_<name>.npz and .json, <figures>/column_volume_<name>_*.png.
Usage: column_volume.py <figures dir> <work dir> --name NAME (--patch P | --box I J K [--size N ...])
                        [--points FILE] [--figs-only]
The two marked faces of the plan: --name M1S1 --box 378 224 448; --name V1V2_banks --box 198 58 65.
"""

import argparse
import json
import sys
import time

sys.path.insert(0, sys.argv[2])
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

import crossing_lib as X
import curv_lib as C
import flatten_fused2 as F
import mesh_smoothing as MS
from click_evidence import Thickness, orientation
from baillarger.data import PATCHES
from onepass_lib import run, setup

ap = argparse.ArgumentParser()
ap.add_argument("fig")
ap.add_argument("work")  # second, as collide.py and click_evidence.py read argv[2] at import
ap.add_argument("--name", required=True)
ap.add_argument("--patch", choices=sorted(PATCHES))
ap.add_argument("--box", type=int, nargs=3, metavar=("I", "J", "K"))
ap.add_argument("--size", type=int, nargs="+", default=[141])
ap.add_argument("--points")
ap.add_argument("--figs-only", action="store_true")
A = ap.parse_args()
FIG, WORK, SITE, FIGS_ONLY = A.fig, A.work, A.name, A.figs_only
RATIO, PATCH, FACE_MM, SAME_MM, TPL_MM = 3.0, 6, 2.0, 5.0, 6.0
NMAX = 201  # samples: 8 mm at 0.04 mm
N_PROFILE, TPL_MAX = 3000, 400
OPEN, CROSS, OWN, TOCONTACT, FAILED = range(5)
CLASS = {OPEN: "open", CROSS: "crossing", OWN: "own bank", TOCONTACT: "to the pass's contact", FAILED: "failed"}
CCOL = {OPEN: "#1baf7a", CROSS: "#2a78d6", OWN: "#eb6834", TOCONTACT: "#c79a00", FAILED: "#e34948"}
HOR = ["top", "far WM", "fallback contact", "one pass, equal", "one pass, field", "field prior", "profile, top"]
HCOL = {"top": "#ffffff", "far WM": "#30d5ff", "fallback contact": "#ff2bd6", "one pass, equal": "#eb6834",
        "one pass, field": "#ffd23f", "field prior": "#e34948", "profile, top": "#2a78d6"}
T0 = time.time()


def say(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def white_mesh(wm):
    v0, f = MS.contour(wm, 0.0)
    v = MS.twice(MS.relax(v0, f), f, 2.0)
    return v, f, MS.smooth_normals(v, f)


def pick_vertices(v, n):
    """One vertex per voxel cell and normal side (dominant axis and sign): the one nearest the
    cell's centre."""
    ax = np.argmax(np.abs(n), 1)
    side = ax * 2 + (n[np.arange(len(n)), ax] > 0)
    cell = np.floor(v).astype(np.int64)
    key = np.c_[cell, side]
    dist = np.linalg.norm(v - (cell + 0.5), axis=1)
    o = np.lexsort((dist, *key.T[::-1]))
    k = key[o]
    first = np.r_[True, np.any(k[1:] != k[:-1], 1)]
    return np.sort(o[first])


def mesh_graph(v, f, vox):
    E = MS.edges(f)
    w = np.linalg.norm((v[E[:, 0]] - v[E[:, 1]]) * vox, axis=1)
    G = coo_matrix((w, (E[:, 0], E[:, 1])), shape=(len(v),) * 2).tocsr()
    return G.maximum(G.T).tocsr()


def box_and_points():
    """The box [lo, hi) of the full grid, and the points (0-based box voxels; (0, 3) without any)."""
    if A.patch:
        lo, hi = (np.array(b) for b in PATCHES[A.patch])
    elif A.box:
        size = np.array(A.size * 3 if len(A.size) == 1 else A.size)
        lo = np.array(A.box)
        hi = lo + size
    else:
        raise SystemExit("give --patch or --box")
    P = np.zeros((0, 3))
    if A.points:
        P = np.array(json.load(open(A.points))["points_full0"], float) - lo
        inside = np.all((P >= 0) & (P < hi - lo), 1)
        if not inside.all():
            print(f"{(~inside).sum()} points outside the box: left out")
        P = P[inside]
    return lo, hi, P


def main():
    lo, hi, P = box_and_points()
    B = C.load(lo, hi, f"column_volume_{SITE}")
    vox = B["vox"]
    h = float(np.mean(vox))
    wm, ctx, rim = B["wm"], B["ctx"], B["rim"]
    labels = np.full(rim.shape, 4, np.int16)
    labels[ctx] = 0
    labels[rim == 1] = 1
    labels[wm] = 2
    labels[B["miss"]] = 3
    feats = [B["det"][1], B["det"][2]]
    Tc = F.tensor(feats[0], ctx.astype(float)) + F.tensor(feats[1], ctx.astype(float))
    _, coh = C.tensor_normal(B)
    d_wm = ndi.distance_transform_edt(~wm, sampling=vox)
    say(f"{SITE}: box {tuple(int(a) for a in hi - lo)}; {len(P)} points")

    # ---- mesh and trace vertices
    v, f, nrm = white_mesh(wm)
    tv = pick_vertices(v, nrm)
    say(f"white mesh: {len(v):,} vertices; traces at {len(tv):,}")

    # ---- one pass, thickness field, root patches and their distances
    s = setup(wm, ctx, patch=PATCH, direction=orientation(B))
    Thk = Thickness(B, s, h)
    mu_f, sd_f, nobel = Thk.weights(Thk.y_tops)
    f_eq = run(s, ratio=RATIO)
    f_fd = run(s, mu_f, sd_f, ratio=RATIO)
    seed_pos = np.stack(np.unravel_index(s.sflat, wm.shape), -1).astype(float)
    seed_tree = cKDTree(seed_pos)
    seed_patch = s.patch.ravel()[s.sflat]
    D = np.full((s.npatch, s.npatch), np.inf, np.float32)  # patch centre to centre along the white surface (mm)
    for a in range(0, s.npatch, 200):
        d = dijkstra(s.graph, indices=s.rep[a:a + 200], limit=8.0 / h)
        D[a:a + 200] = d[:, s.rep] * h
    say(f"passes, field and patch distances done ({s.npatch:,} patches)")

    def patch_at(x):
        return seed_patch[seed_tree.query(x)[1]]

    root_patch = patch_at(v[tv])

    # ---- primary tracer
    x0 = v[tv]
    pos1, ns1, out1 = X.trace_array(Tc, labels, B["dface"], x0, nrm[tv], vox, NMAX)
    oc = np.bincount(out1, minlength=10)
    say("primary: " + ", ".join(f"{X.OUTCOME[k]} {oc[k]:,}" for k in range(10) if oc[k]))
    # where it turns
    it = np.flatnonzero(out1 == X.TURN)
    tpos = pos1[it, ns1[it] - 1].astype(float)
    turn = dict(s_mm=((ns1[it] - 1) * X.STEP_MM).tolist(),
                coh=ndi.map_coordinates(coh, tpos.T, order=1).tolist(),
                d_wm=ndi.map_coordinates(d_wm, tpos.T, order=1).tolist())

    # ---- fallback where the primary turned or ran to the cap
    fb = np.flatnonzero((out1 == X.TURN) | (out1 == X.LENGTH))
    u = np.where(wm, 0.0, np.where(np.isfinite(f_eq.arrival), f_eq.arrival,
                                   np.nanmax(f_eq.arrival[np.isfinite(f_eq.arrival)])))
    lab_eq = f_eq.label

    def far(i_paths, q):
        lab = lab_eq[tuple(q.T)]
        rp = root_patch[fb[i_paths]]
        return (lab >= 0) & (D[rp, np.maximum(lab, 0)] > SAME_MM)

    pos2, ns2, out2 = X.ascend(u, labels, B["dface"], x0[fb], vox, NMAX, far)
    oc2 = np.bincount(out2, minlength=10)
    say("fallback: " + ", ".join(f"{X.OUTCOME[k]} {oc2[k]:,}" for k in range(10) if oc2[k]))
    pos, ns = pos1, ns1.copy()
    pos[fb] = pos2
    ns[fb] = ns2
    tracer = np.zeros(len(tv), np.int8)  # 0 primary, 1 fallback
    tracer[fb] = 1
    outc = out1.copy()
    outc[fb] = out2
    del pos2

    # ---- classes
    endp = pos[np.arange(len(tv)), ns - 1].astype(float)
    end_patch = patch_at(endp)
    sep = D[root_patch, end_patch]
    cls = np.full(len(tv), FAILED, np.int8)
    cls[outc == X.FLUID] = OPEN
    cls[(outc == X.WM) & (sep >= SAME_MM)] = CROSS
    cls[(outc == X.WM) & (sep < SAME_MM)] = OWN
    cls[outc == X.CONTACT] = TOCONTACT
    near_face = B["dface"][tuple(np.clip(np.round(x0).astype(int), 0, np.array(wm.shape) - 1).T)] < FACE_MM
    L = (ns - 1) * X.STEP_MM
    say("classes: " + ", ".join(f"{CLASS[c]} {np.sum((cls == c) & ~near_face):,}" for c in CLASS)
        + f"; roots near a face {near_face.sum():,}")

    # ---- features
    fv = X.sample_traces(feats, pos)
    say("features sampled")

    # ---- horizons
    H = {k: np.full(len(tv), np.nan, np.float32) for k in HOR}
    H["top"][cls == OPEN] = L[cls == OPEN] - X.STEP_MM / 2
    H["far WM"][cls == CROSS] = L[cls == CROSS] - X.STEP_MM / 2
    H["fallback contact"][cls == TOCONTACT] = L[cls == TOCONTACT]
    ic = np.flatnonzero(cls == CROSS)
    pa, pb = root_patch[ic], end_patch[ic]
    for name, ff in (("one pass, equal", f_eq), ("one pass, field", f_fd)):
        for a in range(0, ic.size, 5000):
            j = ic[a:a + 5000]
            p = pos[j]
            ok = np.isfinite(p[..., 0])
            q = np.clip(np.round(np.nan_to_num(p)).astype(int), 0, np.array(wm.shape) - 1)
            lab = ff.label[q[..., 0], q[..., 1], q[..., 2]]
            lab0 = np.maximum(lab, 0)
            near_a = D[pa[a:a + 5000, None], lab0] < SAME_MM
            near_b = D[pb[a:a + 5000, None], lab0] < SAME_MM
            sw = ok & (lab >= 0) & near_b & ~near_a
            first = np.argmax(sw, 1)
            H[name][j] = np.where(sw.any(1), first * X.STEP_MM - X.STEP_MM / 2, np.nan)
    x_pr, _ = X.pairwise_posterior(L[ic], mu_f[pa] * h, sd_f[pa] * h, mu_f[pb] * h, sd_f[pb] * h)
    H["field prior"][ic] = x_pr
    # profile split on a subsample, templates from open traces near each end root
    rng = np.random.default_rng(0)
    io = np.flatnonzero((cls == OPEN) & ~near_face)
    tpl_cache = {}

    def template(patch):
        if patch not in tpl_cache:
            sel = io[D[patch, root_patch[io]] <= TPL_MM]
            if sel.size > TPL_MAX:
                sel = rng.choice(sel, TPL_MAX, replace=False)
            prof = [fv[:, t, :ns[t] - 1] for t in sel]
            tpl_cache[patch] = X.build_template(prof, L[sel] - X.STEP_MM, "top") if sel.size else None
        return tpl_cache[patch]

    sub = rng.choice(ic, min(N_PROFILE, ic.size), replace=False) if ic.size else np.zeros(0, int)
    for t in sub:
        tA, tB = template(root_patch[t]), template(end_patch[t])
        if tA is None or tB is None:
            continue
        y = fv[:, t, :ns[t] - 1].astype(float)
        Lt = (y.shape[1] - 1) * X.STEP_MM
        xs = X.xgrid(Lt, 0.1, 0.9, 0.04)
        r = X.summarize(xs, X.loglik(y, tA, tB, Lt, xs))
        H["profile, top"][t] = r["mean"]
    say(f"horizons done; profile split on {np.isfinite(H['profile, top']).sum():,} crossing traces")

    np.savez_compressed(
        f"{WORK}/column_volume_{SITE}.npz", v=v.astype(np.float32), f=f, nrm=nrm.astype(np.float32), tv=tv,
        pos=pos, ns=ns, fv=fv, cls=cls, tracer=tracer, out1=out1, outc=outc, root_patch=root_patch,
        end_patch=end_patch, near_face=near_face, P=P, vox=vox, lo=lo,
        disp=B["disp"].astype(np.float32), ctx=ctx, coh=coh.astype(np.float32), wm=wm,
        labels=labels.astype(np.int8),  # 0 cortex, 1 fluid, 2 WM, 3 missing, 4 other
        **{"H_" + k: H[k] for k in HOR})
    json.dump(dict(turn=turn), open(f"{WORK}/column_volume_{SITE}_turn.json", "w"))
    say("saved")


# ---------------------------------------------------------------- summaries and views

def planes_of(P):
    """The axis along which the points take the fewest values, and the three planes on it with the
    most points within 2 voxels."""
    ax = int(np.argmin([len(np.unique(np.round(P[:, a]))) for a in range(3)]))
    ks = np.unique(np.round(P[:, ax]).astype(int))
    cnt = [(k, int(np.sum(np.abs(P[:, ax] - k) <= 2))) for k in ks]
    chosen = []
    for k, _ in sorted(cnt, key=lambda t: -t[1]):
        if all(abs(k - c) > 4 for c in chosen):
            chosen.append(k)
        if len(chosen) == 3:
            break
    return ax, chosen


def figures():
    Z = np.load(f"{WORK}/column_volume_{SITE}.npz")
    tv, pos, ns, fv = Z["tv"], Z["pos"], Z["ns"], Z["fv"]
    cls, tracer, out1, outc, near_face, P, vox = Z["cls"], Z["tracer"], Z["out1"], Z["outc"], Z["near_face"], Z["P"], Z["vox"]
    H = {k: Z["H_" + k] for k in HOR}
    turn = json.load(open(f"{WORK}/column_volume_{SITE}_turn.json"))["turn"]
    keep = ~near_face
    summary = dict(site=SITE, traces=int(len(tv)), roots_near_face=int(near_face.sum()))
    summary["classes"] = {CLASS[c]: {"primary": int(np.sum(keep & (cls == c) & (tracer == 0))),
                                     "fallback": int(np.sum(keep & (cls == c) & (tracer == 1)))} for c in CLASS}
    summary["primary outcomes"] = {X.OUTCOME[k]: int(np.sum(keep & (out1 == k))) for k in range(10) if np.sum(keep & (out1 == k))}
    summary["fallback outcomes"] = {X.OUTCOME[k]: int(np.sum(keep & (tracer == 1) & (outc == k))) for k in range(10)
                                    if np.sum(keep & (tracer == 1) & (outc == k))}
    tl = {k: np.array(turn[k]) for k in turn}
    summary["turns"] = dict(n=int(tl["s_mm"].size),
                            s_mm_p10_50_90=np.percentile(tl["s_mm"], [10, 50, 90]).round(2).tolist(),
                            coh_p10_50_90=np.percentile(tl["coh"], [10, 50, 90]).round(2).tolist(),
                            d_wm_p10_50_90=np.percentile(tl["d_wm"], [10, 50, 90]).round(2).tolist(),
                            frac_within_0p3mm_of_wm=round(float(np.mean(tl["d_wm"] < 0.3)), 3))
    for c in (OPEN, CROSS):
        m = keep & (cls == c)
        hk = "top" if c == OPEN else "far WM"
        summary[f"{CLASS[c]} length p5_50_95"] = np.nanpercentile(H[hk][m], [5, 50, 95]).round(2).tolist() if m.any() else None

    # ---- scores at the points: traces passing within 0.3 mm
    rows = []
    ic = np.flatnonzero(keep & ((cls == CROSS) | (cls == TOCONTACT))) if len(P) else np.zeros(0, int)
    tree = cKDTree((P * vox).astype(float)) if len(P) else None
    for t in ic:
        p = pos[t, :ns[t]].astype(float) * vox
        d, k = tree.query(p)
        i = int(np.argmin(d))
        if d[i] <= 0.3:
            rows.append(dict(t=int(t), k=int(k[i]) + 1, depth=i * X.STEP_MM, cls=int(cls[t]),
                             **{h_: float(H[h_][t]) for h_ in HOR}))
    sc = {}
    for h_ in ["fallback contact", "one pass, equal", "one pass, field", "field prior", "profile, top"]:
        off = np.array([r[h_] - r["depth"] for r in rows])
        pts = np.array([r["k"] for r in rows])
        okk = np.isfinite(off)
        if okk.any():
            # one value per point: the median over the traces through it
            per = [np.median(np.abs(off[okk & (pts == k)])) for k in np.unique(pts[okk])]
            sc[h_] = dict(traces=int(okk.sum()), points=len(per), abs_median_per_point_median=round(float(np.median(per)), 3),
                          abs_median_per_point_p90=round(float(np.percentile(per, 90)), 3))
    if len(P):
        summary["at the points (|horizon - point depth|, mm)"] = sc
    json.dump(summary, open(f"{WORK}/column_volume_{SITE}.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    if len(P):
        ax, planes = planes_of(P)
        sections(Z, H, keep, ax, planes, vmax=np.nanpercentile(np.abs(fv[0][keep]), 97))
    else:  # no points: the box's three middle planes along its third axis
        ax, n3 = 2, Z["disp"].shape[2]
        planes = [n3 // 4, n3 // 2, 3 * n3 // 4]
    roots_view(Z, keep, ax, planes, tl, vmax=np.nanpercentile(np.abs(fv[0][keep]), 97))
    print("figures written")


def sections(Z, H, keep, ax, planes, vmax):
    v, f, nrm, tv, pos, ns, fv = Z["v"], Z["f"], Z["nrm"], Z["tv"], Z["pos"], Z["ns"], Z["fv"]
    cls, tracer, P, vox = Z["cls"], Z["tracer"], Z["P"], Z["vox"]
    G = mesh_graph(v, f, vox)
    tv_set = np.zeros(len(v), bool)
    tv_set[tv] = True
    trace_of = -np.ones(len(v), int)
    trace_of[tv] = np.arange(len(tv))
    depth = np.arange(0, 6.0 + 1e-9, X.STEP_MM)
    fig, axs = plt.subplots(len(planes) * 2, 1, figsize=(20, 7.5 * len(planes)), squeeze=False)
    axs = axs[:, 0]
    for pi_, k in enumerate(planes):
        band = np.flatnonzero(np.abs(v[:, ax] - k) < 0.75)
        Pk = P[np.abs(P[:, ax] - k) <= 2]
        c0 = Pk[np.argmin(np.linalg.norm(Pk - np.median(Pk, 0), axis=1))]
        dv = (c0 - v[band]) * vox
        facing = band[((dv * nrm[band]).sum(1) > 0)]
        dd = np.linalg.norm((c0 - v[facing]) * vox, axis=1)
        a1 = facing[np.argmin(dd)]
        g1 = dijkstra(G, indices=a1, limit=30.0)
        other = facing[~(g1[facing] < SAME_MM)]
        anchors = [a1]
        if other.size:
            anchors.append(other[np.argmin(np.linalg.norm((c0 - v[other]) * vox, axis=1))])
        sub = G[band][:, band]
        _, comp = connected_components(sub, directed=False)
        for bi, anc in enumerate(anchors):
            a = axs[pi_ * 2 + bi]
            piece = band[comp == comp[np.searchsorted(band, anc)]]
            o, dist = X.order_along(piece, G)
            da = dist[np.flatnonzero(o == anc)[0]]
            sel = (np.abs(dist - da) <= 8.0) & tv_set[o]
            o, dist = o[sel], dist[sel] - da
            tr = trace_of[o]
            img = np.full((depth.size, tr.size), np.nan)
            for j, t in enumerate(tr):
                m_ = min(ns[t], depth.size)
                img[:m_, j] = fv[0, t, :m_]
            ext = [dist[0], dist[-1], depth[-1], 0] if tr.size > 1 else None
            a.imshow(img, aspect="auto", cmap="gray", vmin=-vmax, vmax=vmax, extent=ext, interpolation="nearest")
            for h_ in HOR:
                y = H[h_][tr]
                if h_ == "profile, top":
                    a.plot(dist, y, ".", c=HCOL[h_], ms=4, label=h_)
                else:
                    a.plot(dist, y, "-", c=HCOL[h_], lw=1.2 if h_ in ("top", "far WM") else 1.0, label=h_)
            # the points' depths on traces passing within 0.3 mm
            for j, t in enumerate(tr):
                p = pos[t, :ns[t]].astype(float) * vox
                d_ = np.linalg.norm(p[:, None] - (Pk * vox)[None], axis=2)
                if d_.min() <= 0.3:
                    a.plot(dist[j], np.unravel_index(np.argmin(d_), d_.shape)[0] * X.STEP_MM, "*", c="#ffd23f", ms=8)
            for j, t in enumerate(tr):  # class strip at the bottom, tracer as the marker's fill
                a.plot(dist[j], depth[-1] - 0.08, "s", ms=4, mec=CCOL[int(cls[t])],
                       mfc=CCOL[int(cls[t])] if tracer[t] == 0 else "none")
            a.set_ylim(depth[-1], 0)
            a.set_ylabel("depth from the root (mm)")
            a.set_xlabel("position along the white surface (mm; 0 = the vertex facing the points)")
            a.set_title(f"plane {'ijk'[ax]} = {k} (0-based), bank {bi + 1}: echo 2 trend removed along each trace; "
                        f"strip: class (green open, blue crossing, orange own bank, gold to the pass's contact, red failed; "
                        f"hollow: fallback tracer); ★ your points", fontsize=10)
            if pi_ == 0 and bi == 0:
                a.legend(fontsize=7, loc="lower right", ncol=4)
    fig.tight_layout()
    fig.savefig(f"{FIG}/column_volume_{SITE}_sections.png", dpi=72)
    plt.close(fig)


def roots_view(Z, keep, ax, planes, tl, vmax):
    """Roots by class on the planes, and where the primary tracer turns."""
    v, tv, cls, tracer, P = Z["v"], Z["tv"], Z["cls"], Z["tracer"], Z["P"]
    disp, ctx = Z["disp"], Z["ctx"]
    fig, axs = plt.subplots(1, len(planes) + 1, figsize=(8 * (len(planes) + 1), 8))
    for a, k in zip(axs, planes):
        sl = np.take(np.where(ctx, disp, np.nan), k, axis=ax)
        a.imshow(sl.T, cmap="gray", origin="lower", vmin=-vmax, vmax=vmax)
        m = (np.abs(v[tv][:, ax] - k) < 0.75)
        rest = [d for d in range(3) if d != ax]
        for c in CLASS:
            for tr_, fill in ((0, True), (1, False)):
                mm_ = m & (cls == c) & (tracer == tr_)
                a.plot(v[tv][mm_, rest[0]], v[tv][mm_, rest[1]], "o", ms=2.5, mec=CCOL[c],
                       mfc=CCOL[c] if fill else "none", mew=0.6, ls="none")
        mp = np.abs(P[:, ax] - k) <= 2
        a.plot(P[mp, rest[0]], P[mp, rest[1]], "*", c="#ffd23f", ms=8)
        a.set_title(f"plane {'ijk'[ax]} = {k}: roots by class (as the sections; hollow = fallback)", fontsize=10)
    a = axs[-1]
    a.hist(tl["s_mm"], np.arange(0, 8.01, 0.2), color="#8a8984")
    a2 = a.twinx()
    a2.plot(tl["s_mm"], tl["coh"], ".", ms=1.5, c="#2a78d6", alpha=0.4)
    a2.set_ylabel("coherence at the turn (blue dots)")
    a.set_xlabel("depth along the trace where the primary tracer turned (mm)")
    a.set_title(f"primary turns: {tl['s_mm'].size:,}; within 0.3 mm of WM {np.mean(tl['d_wm'] < 0.3):.0%}", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG}/column_volume_{SITE}_roots.png", dpi=72)
    plt.close(fig)


if __name__ == "__main__":
    if not FIGS_ONLY:
        main()
    figures()
