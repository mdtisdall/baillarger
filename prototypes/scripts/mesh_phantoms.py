"""Finding 28, phantoms: start surfaces for the march against a known smooth white surface.

Part 1 (surface): for each phantom and start surface, the vertex position error (signed
distance to the true surface), the normal error, the mean-curvature error (mesh_curvatures
at 0.32 and 0.48 mm; the voxel-face quadric of curv_lib for the raw voxels), triangle
quality, and for thin blades the coverage of each true side by mesh vertices with agreeing
normals and the self-intersections (baillarger.mesh_intersect).

Part 2 (march): one front from each start surface, isotropic and ratio 3 along the true
normal; at the top voxels (d in (T - 1, T]) the arrival, arc length and carried root
position against the truth (d and the foot point); near the WM the divergence of the unit
arrival gradient (the carried-area integrand) against sum k_i / (1 + k_i d). The true
surface as a start (band from the analytic distance) gives the solver's own error; a 2x
grid with the same start surfaces shows how it scales.

Usage: mesh_phantoms.py <figures dir> <out json> [surface|march|all]
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

import curv_lib as C
import mesh_smoothing as M
from baillarger.frame import mesh_curvatures, white_mesh
from baillarger.fronts import riemann_from_direction
from baillarger.mesh_intersect import intersecting_faces

FIG, OUT = sys.argv[1], sys.argv[2]
WHAT = sys.argv[3] if len(sys.argv) > 3 else "all"
H = M.H_MM
MARGIN = 6.0
BAND = 2.5  # >= the ratio-3 stencil's reach along the normal (2.45 voxels)


def phantoms():
    return [M.ball(R=8.0), M.wavy(), M.plane(4.0), M.blade(w=1.5), M.blade(w=2.0), M.blade(w=3.0),
            M.blade(w=4.0)]


def surfaces(wm, sigmas=(0.5, 1.0, 1.5, 2.0), scales=(0.5, 1.0, 2.0, 3.0), taubin=(50,)):
    """name -> (vertices, faces, normals) for every start surface. Normals: white_mesh's
    own; for the others the same 10 steps of normal smoothing along the edges."""
    out = {}
    v0, f0 = M.contour(wm, 0.0)
    out["binary contour"] = (v0, f0, M.smooth_normals(v0, f0))
    for s in sigmas:
        v, f, n = white_mesh(wm, (H, H, H), sigma=s)
        # white_mesh flips its normals, not its faces (which point into the WM)
        out[f"white_mesh s{s:g}"] = (v, f[:, [0, 2, 1]].copy(), n)
    vr = M.relax(v0, f0)
    for c in scales:
        v = M.twice(vr, f0, c)
        out[f"twice {c * H:.2f}mm"] = (v, f0, M.smooth_normals(v, f0))
    for it in taubin:
        v = M.taubin(vr, f0, it)
        out[f"taubin x{it}"] = (v, f0, M.smooth_normals(v, f0))
    v5, f5 = M.contour(wm, 0.5)
    v5 = M.relax(v5, f5)
    for c in (1.0, 2.0):
        v = M.twice(v5, f5, c)
        out[f"s0.5+twice {c * H:.2f}mm"] = (v, f5, M.smooth_normals(v, f5))
    return out


def away(ph, X, m=MARGIN):
    n = np.array(ph.shape) - 1
    return np.all((X >= m) & (X <= n - m), 1)


def pct(x, q):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size else float("nan")


def surface_metrics(ph, v, f, n, curv=True):
    if not len(f):
        return dict(vertices=0), None, None
    tr = ph.truth(v)
    ok = away(ph, v) & away(ph, tr.foot)
    ang = np.degrees(np.arccos(np.clip(np.sum(n * tr.n, 1), -1, 1)))
    r = dict(
        vertices=int(len(v)),
        pos_bias=float(np.mean(tr.d[ok])),
        pos_abs_med=pct(np.abs(tr.d[ok]), 50),
        pos_abs_p95=pct(np.abs(tr.d[ok]), 95),
        normal_med=pct(ang[ok], 50),
        normal_p95=pct(ang[ok], 95),
        normal_raw_med=pct(np.degrees(np.arccos(np.clip(np.sum(M.vertex_normals(v, f) * tr.n, 1), -1, 1)))[ok], 50),
        minangle_p5=pct(M.min_angles(v, f), 5),
        minangle_med=pct(M.min_angles(v, f), 50),
    )
    if curv:
        Ht = 0.5 * (tr.k1 + tr.k2) / H
        for sc in (0.32, 0.48):
            k1, k2 = mesh_curvatures(v * H, f, n, sc)
            e = 0.5 * (k1 + k2) - Ht
            r[f"H_err_med_{sc}"] = pct(np.abs(e[ok]), 50)
            r[f"H_err_p95_{sc}"] = pct(np.abs(e[ok]), 95)
            r[f"H_bias_{sc}"] = float(np.mean(e[ok]))
    return r, tr, ok


def quadric_metrics(ph, wm):
    shell = wm & ~ndi.binary_erosion(wm, border_value=1)
    B = dict(wm=wm, vox=np.array([H] * 3), shell=shell)
    r = {}
    for sc in (0.32, 0.48):
        K = C.curvature(B, sc)
        X = K["pts"].astype(float)
        tr = ph.truth(X)
        ok = away(ph, X) & away(ph, tr.foot)
        if sc == 0.32:
            ang = np.degrees(np.arccos(np.clip(np.sum(K["n"] * tr.n, 1), -1, 1)))
            r.update(vertices=int(len(X)), pos_bias=float(np.mean(tr.d[ok])),
                     pos_abs_med=pct(np.abs(tr.d[ok]), 50), pos_abs_p95=pct(np.abs(tr.d[ok]), 95),
                     normal_med=pct(ang[ok], 50), normal_p95=pct(ang[ok], 95))
        e = 0.5 * (K["k1"] + K["k2"]) - 0.5 * (tr.k1 + tr.k2) / H
        r[f"H_err_med_{sc}"] = pct(np.abs(e[ok]), 50)
        r[f"H_err_p95_{sc}"] = pct(np.abs(e[ok]), 95)
        r[f"H_bias_{sc}"] = float(np.mean(e[ok]))
    return r


def blade_metrics(ph, v, f, n, ctx, tr_grid):
    """Coverage of each true side, flipped normals, self-intersections."""
    X = ph.grid().reshape(-1, 3)
    d = tr_grid.d
    near = ctx.ravel() & (d <= 1.5)
    feet, fn, side = tr_grid.foot[near], tr_grid.n[near], tr_grid.side[near]
    keep = away(ph, feet) & away(ph, X[near])
    feet, fn, side = feet[keep], fn[keep], side[keep]
    out = {}
    if len(v):
        dist, j = cKDTree(v).query(feet)
        good = (dist <= 1.0) & (np.sum(n[j] * fn, 1) > 0.5)
    else:
        good = np.zeros(len(feet), bool)
    for s in (1, -1):
        out[f"cover_{'+' if s > 0 else '-'}"] = float(good[side == s].mean())
    if not len(f):
        out.update(vertices=0, flipped=float("nan"), self_intersections=0)
        return out
    tv = ph.truth(v)
    okv = away(ph, v)
    out["flipped"] = float(np.mean(np.sum(n[okv] * tv.n[okv], 1) < 0)) if okv.any() else float("nan")
    out["side_bias_+"] = float(np.mean(tv.d[okv & (tv.side > 0)])) if okv.any() else float("nan")
    out["side_bias_-"] = float(np.mean(tv.d[okv & (tv.side < 0)])) if okv.any() else float("nan")
    out["self_intersections"] = int(intersecting_faces(v, f)[1]) if len(f) else 0
    return out


def section(v, f, axis, c):
    s = v[:, axis] - c
    segs = []
    S = s[f]
    cross = (np.sign(S[:, 0]) != np.sign(S[:, 1])) | (np.sign(S[:, 1]) != np.sign(S[:, 2]))
    for tri, st in zip(f[cross], S[cross], strict=True):
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if (st[a] > 0) != (st[b] > 0):
                t = st[a] / (st[a] - st[b])
                pts.append(v[tri[a]] + t * (v[tri[b]] - v[tri[a]]))
        if len(pts) == 2:
            segs.append(np.delete(np.array(pts), axis, 1))
    return np.array(segs)


def part_surface(res):
    blades_fig = []
    curv_fig = None
    for ph in phantoms():
        t0 = time.time()
        wm, ctx, trg = ph.labels()
        S = surfaces(wm)
        rows = {"voxel faces (quadric)": quadric_metrics(ph, wm)}
        for name, (v, f, n) in S.items():
            r, tr, ok = surface_metrics(ph, v, f, n)
            if ph.kind == "blade":
                r.update(blade_metrics(ph, v, f, n, ctx, trg))
            elif len(f):
                r["self_intersections"] = int(intersecting_faces(v, f)[1])
            rows[name] = r
            if not len(f):
                continue
            if ph.name == "wavy" and name in ("binary contour", "white_mesh s1", "twice 0.32mm"):
                curv_fig = curv_fig or []
                k1, k2 = mesh_curvatures(v * H, f, n, 0.32)
                loc = (v - ph.p["o"]) @ ph.p["Q"]
                curv_fig.append((name, loc, 0.5 * (k1 + k2) - 0.5 * (tr.k1 + tr.k2) / H, ok))
        if ph.kind == "blade":
            blades_fig.append((ph, wm, S))
        res.setdefault("surface", {})[ph.name] = rows
        print(f"{ph.name}: {time.time() - t0:.1f} s", flush=True)
        for name, r in rows.items():
            if r.get("vertices", 1) == 0:
                print(f"  {name:24s} no surface; " + (f"cover {r.get('cover_+', 0):.2f}/{r.get('cover_-', 0):.2f}" if "cover_+" in r else ""))
                continue
            extra = ""
            if "cover_+" in r:
                extra = (f" cover {r['cover_+']:.2f}/{r['cover_-']:.2f} flip {r['flipped']:.3f}"
                         f" bias+ {r['side_bias_+']:+.2f} bias- {r['side_bias_-']:+.2f}")
            print(f"  {name:24s} pos bias {r['pos_bias']:+.2f} |d| {r['pos_abs_med']:.2f}/{r['pos_abs_p95']:.2f}"
                  f" nrm {r['normal_med']:.1f}/{r['normal_p95']:.1f} raw {r.get('normal_raw_med', float('nan')):.1f} H.32 {r['H_err_med_0.32']:.3f}/{r['H_err_p95_0.32']:.3f}"
                  f" H.48 {r['H_err_med_0.48']:.3f}/{r['H_err_p95_0.48']:.3f} bias {r['H_bias_0.48']:+.3f}"
                  f" X {r.get('self_intersections', '-')}{extra}", flush=True)
    # figure: blade cross-sections
    names = ["binary contour", "white_mesh s0.5", "white_mesh s1", "white_mesh s1.5", "white_mesh s2",
             "twice 0.16mm", "twice 0.48mm"]
    fig, axs = plt.subplots(len(blades_fig), len(names), figsize=(3.0 * len(names), 3.2 * len(blades_fig)))
    for i, (ph, wm, S) in enumerate(blades_fig):
        zc = ph.shape[2] // 2
        for j, nm in enumerate(names):
            a = axs[i, j]
            a.imshow(wm[:, :, zc].T, origin="lower", cmap="Greys", vmin=0, vmax=2.5)
            sg = section(S[nm][0], S[nm][1], 2, zc) if len(S[nm][1]) else []
            if len(sg):
                a.add_collection(LineCollection(sg, colors="#e34948", lw=1.0))
            # true sides
            g = ph.grid()[:, :, zc].reshape(-1, 3)
            dd = ph.truth(g).d.reshape(wm.shape[:2])
            a.contour(np.arange(wm.shape[0]), np.arange(wm.shape[1]), dd.T, [0], colors="#2a78d6", linewidths=0.8)
            c0 = np.array(ph.shape[:2]) // 2
            a.set_xlim(c0[0] - 14, c0[0] + 14)
            a.set_ylim(c0[1] - 14, c0[1] + 14)
            a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(nm, fontsize=10)
            if j == 0:
                a.set_ylabel(ph.name)
    fig.suptitle("Thin WM blades (grey: label; blue: true surface; red: start surface), plane z = centre", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{FIG}/mesh_phantom_blades.png", dpi=110)
    plt.close(fig)
    if curv_fig:
        fig, axs = plt.subplots(1, len(curv_fig), figsize=(5.2 * len(curv_fig), 4.6))
        for a, (nm, loc, e, ok) in zip(axs, curv_fig, strict=True):
            sc = a.scatter(loc[ok, 0] * H, loc[ok, 1] * H, c=e[ok], s=2, cmap="RdBu_r", vmin=-0.4, vmax=0.4)
            a.set_title(f"{nm}: H error (1/mm), scale 0.32 mm", fontsize=10)
            a.set_xlabel("u (mm)"); a.set_ylabel("v (mm)"); a.set_aspect("equal")
        fig.colorbar(sc, ax=axs, shrink=0.8)
        fig.savefig(f"{FIG}/mesh_phantom_curvature.png", dpi=110)
        plt.close(fig)


# ------------------------------------------------------------------ march


def div_unit(u, scale):
    g = np.stack(np.gradient(u, 1.0 / scale), -1)
    gn = np.linalg.norm(g, axis=-1, keepdims=True)
    e = g / np.maximum(gn, 1e-12)
    return sum(np.gradient(e[..., a], 1.0 / scale, axis=a) for a in range(3))


def run_case(ph, start, scale, ratio, wm1):
    """One march. start: 'raw', 'raw patches', 'true', or (name, v, f) a mesh in 1x voxels."""
    wm, ctx, tr = ph.labels(scale)
    X = ph.grid(scale)
    shp = ctx.shape
    nfield = tr.n.reshape(shp + (3,))
    Mt = riemann_from_direction(nfield, ratio)
    D = None if ratio == 1.0 else np.linalg.inv(Mt)
    h = np.full(3, 1.0 / scale)
    if start in ("raw", "raw patches"):
        seeds = wm & ndi.binary_dilation(ctx, ndi.generate_binary_structure(3, 1))
        dom = ctx | seeds
        k = np.flatnonzero(seeds.ravel())
        lab = None
        if start == "raw patches":
            import onepass_lib as O

            s = O.setup(wm, ctx, patch=6)
            lab = s.patch.ravel()[k]
        out = M.march_band(Mt, dom, seeds, np.zeros(k.size), np.zeros(k.size), X.reshape(-1, 3)[k], lab,
                           spacing=h)
    else:
        if start == "true":
            closest = M.truth_closest(ph)
        elif start[2] is None:
            closest = start[1]  # an implicit start's closest-point function
        else:
            closest = M.mesh_closest(start[1], start[2])
        seeds, t, s, r = M.surface_seeds(ctx, wm, X, closest, D, band=BAND, scale=scale)
        out = M.march_band(Mt, ctx, seeds, t, s, r, spacing=h)
    d = tr.d.reshape(shp)
    foot = tr.foot.reshape(shp + (3,))
    pts = X.reshape(-1, 3)
    okp = (away(ph, pts, 4.0) & away(ph, tr.foot)).reshape(shp)
    top = ctx & (d > ph.T - 1) & okp & np.isfinite(out.arrival)
    ea = out.arrival[top] - d[top]
    el = out.length[top] - d[top]
    dr = out.root[top] - foot[top]
    nn = nfield[top]
    dt = dr - np.sum(dr * nn, 1, keepdims=True) * nn
    r = dict(n_top=int(top.sum()), reach=float(np.mean(np.isfinite(out.arrival[ctx]))),
             arr_bias=float(np.median(ea)), arr_iqr=float(np.subtract(*np.percentile(ea, [75, 25]))),
             arr_p95=pct(np.abs(ea), 95),
             len_bias=float(np.median(el)), len_iqr=float(np.subtract(*np.percentile(el, [75, 25]))),
             len_p95=pct(np.abs(el), 95),
             root_med=pct(np.linalg.norm(dr, axis=1), 50), root_tan_med=pct(np.linalg.norm(dt, axis=1), 50),
             root_tan_p95=pct(np.linalg.norm(dt, axis=1), 95))
    # the root map near the WM (within 1 mm): its jump per unit step between
    # 6-neighbours, and voxels with a neighbour whose root is > 1 mm away (Euclidean,
    # a lower bound on the distance along the surface): false collisions here,
    # where the true surface has none within the cortex
    reached = ctx & np.isfinite(out.arrival) & okp
    import onepass_lib as O

    for nm_, lim in (("near", 6.25), ("all", ph.T + 1)):
        m = reached & (d <= lim)
        idx = np.flatnonzero(m.ravel())
        Rm = out.root.reshape(-1, 3)
        js, hit = [], np.zeros(m.size, bool)
        for off in O.N6:
            pa, pb = O.shifted_pairs(m, m, off)
            jj = np.linalg.norm(Rm[pa] - Rm[pb], axis=1)
            js.append(jj * scale)
            far = jj > 1.0 / H
            hit[pa[far]] = True
            hit[pb[far]] = True
        J = np.concatenate(js)
        r[f"rootjump_{nm_}_p95"] = pct(J, 95)
        r[f"rootjump_{nm_}_p999"] = pct(J, 99.9)
        r[f"false_coll_{nm_}"] = int(hit.sum())
        r[f"false_coll_{nm_}_frac"] = float(hit.sum() / max(idx.size, 1))
    # near the WM: div of the unit arrival gradient against sum k / (1 + k d)
    u = np.where(np.isfinite(out.arrival), out.arrival, np.nan)
    dv = div_unit(u, scale)
    k1, k2 = tr.k1.reshape(shp), tr.k2.reshape(shp)
    true_div = k1 / (1 + k1 * d) + k2 / (1 + k2 * d)
    inner = ndi.binary_erosion(ctx & np.isfinite(out.arrival), iterations=2) & okp
    for lo, hi in ((1.5, 3.0), (3.0, 6.0), (6.0, 12.0)):
        m = inner & (d > lo) & (d <= hi) & np.isfinite(dv)
        e = (dv[m] - true_div[m]) / H  # per mm
        r[f"div_{lo:g}_{hi:g}_bias"] = float(np.median(e)) if m.any() else float("nan")
        r[f"div_{lo:g}_{hi:g}_iqr"] = float(np.subtract(*np.percentile(e, [75, 25]))) if m.any() else float("nan")
        r[f"div_{lo:g}_{hi:g}_truth"] = float(np.median(true_div[m]) / H) if m.any() else float("nan")
    return r, (out, d, ctx, top)


def part_march(res):
    cases = [M.ball(R=8.0), M.wavy(), M.plane(4.0), M.blade(w=2.0), M.blade(w=4.0)]
    keepviz = {}
    for ph in cases:
        wm1, ctx1, _ = ph.labels()
        S = surfaces(wm1, sigmas=(1.0, 2.0), scales=(1.0, 2.0, 3.0), taubin=())
        only = os.environ.get("STARTS")  # e.g. "twice 0.32mm": run only these starts, keep the rest
        want = only.split(",") if only else ["raw", "raw patches", "true", "binary contour", "white_mesh s1",
                                             "white_mesh s2", "twice 0.16mm", "twice 0.32mm", "twice 0.48mm",
                                             "implicit s0.5"]
        starts = [st for st in ("raw", "raw patches", "true") if st in want] + [
            (nm, v, f) for nm, (v, f, _) in S.items() if nm in want and len(f)]
        if "implicit s0.5" in want:
            starts.append(("implicit s0.5", M.implicit_closest(wm1, 0.5), None))
        for ratio in (1.0, 3.0):
            for scale in (1, 2):
                for st in starts:
                    nm = st if isinstance(st, str) else st[0]
                    if scale == 2 and nm in ("raw", "raw patches"):
                        continue
                    if scale == 2 and ph.kind == "blade":
                        continue
                    t0 = time.time()
                    r, viz = run_case(ph, st, scale, ratio, wm1)
                    res.setdefault("march", {}).setdefault(ph.name, {})[f"{nm} | ratio {ratio:g} | {scale}x"] = r
                    print(f"{ph.name:12s} {nm:22s} r{ratio:g} {scale}x  arr {r['arr_bias']:+.3f} ({r['arr_iqr']:.3f}, {r['arr_p95']:.2f})"
                          f"  len {r['len_bias']:+.3f} ({r['len_iqr']:.3f}, {r['len_p95']:.2f})"
                          f"  root tan {r['root_tan_med']:.2f}/{r['root_tan_p95']:.2f} (|root| {r['root_med']:.2f})"
                          f"  div1.5-3 {r['div_1.5_3_bias']:+.3f}/{r['div_1.5_3_iqr']:.3f} (true {r['div_1.5_3_truth']:.3f})"
                          f"  div3-6 {r['div_3_6_bias']:+.3f}/{r['div_3_6_iqr']:.3f}  rootjump {r['rootjump_near_p95']:.2f}/{r['rootjump_near_p999']:.2f}"
                          f"  falsecoll {r['false_coll_near']}/{r['false_coll_all']}  reach {r['reach']:.3f}  {time.time() - t0:.0f}s",
                          flush=True)
                    if ph.name == "wavy" and scale == 1 and ratio == 3.0 and nm in ("raw", "true", "twice 0.16mm"):
                        keepviz[nm] = viz
                    json.dump(res, open(OUT, "w"), indent=1)
    if keepviz and not os.environ.get("STARTS"):
        fig, axs = plt.subplots(2, len(keepviz), figsize=(5 * len(keepviz), 9))
        for j, (nm, (out, d, ctx, top)) in enumerate(keepviz.items()):
            y = ctx.shape[1] // 2
            err = np.where(ctx, out.arrival - d, np.nan)[:, y, :]
            a = axs[0, j]
            im = a.imshow(err.T * H, origin="lower", cmap="RdBu_r", vmin=-0.1, vmax=0.1)
            a.set_title(f"{nm}: arrival − d (mm), ratio 3, plane y = {y}", fontsize=10)
            fig.colorbar(im, ax=a, shrink=0.7)
            dv = div_unit(np.where(np.isfinite(out.arrival), out.arrival, np.nan), 1)
            a = axs[1, j]
            im = a.imshow(np.where(ctx, dv, np.nan)[:, y, :].T / H, origin="lower", cmap="PuOr", vmin=-1.5, vmax=1.5)
            a.set_title(f"{nm}: div of unit grad (1/mm)", fontsize=10)
            fig.colorbar(im, ax=a, shrink=0.7)
        fig.tight_layout()
        fig.savefig(f"{FIG}/mesh_phantom_march.png", dpi=110)
        plt.close(fig)


def tip_retraction(ph, v):
    """Per 1-voxel bin along the blade (y'), the true top (ztip + w/2) minus the highest
    mesh vertex near the blade (|x' - c(y')| < w); the full height where none is above
    the base."""
    Q, o, w, a, lam = ph.p["Q"], ph.p["o"], ph.p["w"], ph.p["a"], ph.p["lam"]
    top = ph.p["ztip"] + w / 2
    L = (v - o) @ Q
    c = a * np.sin(2 * np.pi * L[:, 1] / lam)
    on = (np.abs(L[:, 0] - c) < w) & (L[:, 2] > ph.p["zbase"] + 2)
    n = ph.shape[0]
    ys = np.arange(-n / 2 + MARGIN + 4, n / 2 - MARGIN - 4)
    out = []
    for y in ys:
        m = on & (np.abs(L[:, 1] - y) < 0.5)
        out.append(top - L[m, 2].max() if m.any() else top - ph.p["zbase"])
    return np.array(out)


def part_fin(res):
    fig, axs = None, None
    names = ["binary contour", "white_mesh s0.5", "white_mesh s1", "white_mesh s1.5", "white_mesh s2",
             "twice 0.16mm", "twice 0.32mm", "twice 0.48mm"]
    fins = [M.fin(w=1.5), M.fin(w=2.0), M.fin(w=3.0), M.fin(w=4.0)]
    fig, axs = plt.subplots(len(fins), len(names), figsize=(2.6 * len(names), 3.0 * len(fins)))
    for i, ph in enumerate(fins):
        wm, ctx, trg = ph.labels()
        S = surfaces(wm, scales=(1.0, 2.0, 3.0), taubin=())
        rows = {}
        for name, (v, f, n) in S.items():
            r = blade_metrics(ph, v, f, n, ctx, trg)
            tr_ = tip_retraction(ph, v) if len(v) else np.array([np.nan])
            r["tip_retraction_med"] = float(np.median(tr_))
            r["tip_retraction_max"] = float(np.max(tr_))
            rows[name] = r
            print(f"  {ph.name:8s} {name:22s} cover {r['cover_+']:.2f}/{r['cover_-']:.2f} X {r['self_intersections']}"
                  f" tip retraction med {r['tip_retraction_med']:.2f} max {r['tip_retraction_max']:.2f} voxels", flush=True)
        res.setdefault("fin", {})[ph.name] = rows
        # section across the blade (plane y' = const through the box centre): the
        # grid plane nearest it, x
        zc = ph.shape[1] // 2
        for j, nm in enumerate(names):
            a = axs[i, j]
            a.imshow(wm[:, zc, :].T, origin="lower", cmap="Greys", vmin=0, vmax=2.5)
            g = ph.grid()[:, zc, :].reshape(-1, 3)
            dd = ph.truth(g).d.reshape(wm.shape[0], wm.shape[2])
            a.contour(np.arange(wm.shape[0]), np.arange(wm.shape[2]), dd.T, [0], colors="#2a78d6", linewidths=0.8)
            if len(S[nm][1]):
                sg = section(S[nm][0], S[nm][1], 1, zc)
                if len(sg):
                    a.add_collection(LineCollection(sg, colors="#e34948", lw=1.0))
            a.set_xlim(ph.shape[0] / 2 - 12, ph.shape[0] / 2 + 12)
            a.set_ylim(ph.shape[2] / 2 - 14, ph.shape[2] / 2 + 10)
            a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(nm, fontsize=9)
            if j == 0:
                a.set_ylabel(ph.name)
    fig.suptitle("Thin WM fins with a tip (grey: label; blue: true surface; red: start surface); plane y = centre",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG}/mesh_phantom_fins.png", dpi=110)
    plt.close(fig)


def band_starts(wm):
    """name -> (closest function, mesh or None) for the narrow-band comparison."""
    out = {}
    vf, ff = M.voxel_face_mesh(wm)
    out["voxel faces"] = (M.mesh_closest(vf, ff), None)
    S = surfaces(wm, scales=(1.0, 2.0, 3.0), taubin=())
    for nm in ("binary contour", "white_mesh s0.5", "white_mesh s1", "white_mesh s1.5", "white_mesh s2",
               "twice 0.16mm", "twice 0.32mm", "twice 0.48mm"):
        v, f, _ = S[nm]
        out[nm] = (M.mesh_closest(v, f) if len(f) else None, (v, f))
    for sg in (0.5, 1.0):
        out[f"implicit s{sg:g}"] = (M.implicit_closest(wm, sg), None)
    return out


def neighbour_jumps(mask, P, scale):
    """|P_i - P_j| per unit step over 6-neighbour pairs inside mask; P over mask's
    True voxels in flat order."""
    idx = -np.ones(mask.size, np.int64)
    fl = np.flatnonzero(mask.ravel())
    idx[fl] = np.arange(fl.size)
    import onepass_lib as O

    out = []
    for off in O.N6:
        a, b = O.shifted_pairs(mask, mask, off)
        out.append(np.linalg.norm(P[idx[a]] - P[idx[b]], axis=1) * scale)
    return np.concatenate(out)


def part_band(res):
    cases = [M.ball(R=8.0), M.wavy(), M.plane(4.0), M.blade(w=1.5), M.blade(w=2.0), M.blade(w=3.0), M.fin(w=2.0)]
    for ph in cases:
        wm1, _, _ = ph.labels()
        starts = band_starts(wm1)
        for scale in (1, 2):
            t0 = time.time()
            G = ph.grid(scale)
            shp = G.shape[:-1]
            Xa = G.reshape(-1, 3)
            tr = ph.truth(Xa)
            band = (np.abs(tr.d) <= 2.0) & away(ph, Xa) & away(ph, tr.foot)
            X = Xa[band]
            dt, ft, nt, side = tr.d[band], tr.foot[band], tr.n[band], tr.side[band]
            outside = dt > 0
            omask = np.zeros(np.prod(shp), bool)
            omask[np.flatnonzero(band)[outside]] = True
            omask = omask.reshape(shp)
            for nm, (cl, mesh) in starts.items():
                if cl is None:
                    res.setdefault("band", {}).setdefault(ph.name, {})[f"{nm} | {scale}x"] = dict(missing=True)
                    print(f"  {ph.name:10s} {scale}x {nm:18s} no surface", flush=True)
                    continue
                P, n = cl(X)
                de = np.sum((X - P) * n, 1)
                e = de - dt
                fe = P - ft
                fet = fe - np.sum(fe * nt, 1, keepdims=True) * nt
                clear = np.abs(dt) > 0.3
                r = dict(n=int(len(X)), sd_bias=float(np.median(e)),
                         sd_iqr=float(np.subtract(*np.percentile(e, [75, 25]))), sd_p95=pct(np.abs(e), 95),
                         foot_med=pct(np.linalg.norm(fe, axis=1), 50), foot_p95=pct(np.linalg.norm(fe, axis=1), 95),
                         foot_tan_med=pct(np.linalg.norm(fet, axis=1), 50),
                         foot_tan_p95=pct(np.linalg.norm(fet, axis=1), 95),
                         sign_err=float(np.mean(np.sign(de[clear]) != np.sign(dt[clear]))))
                J = neighbour_jumps(omask, P[outside], scale)
                r.update(jump_p50=pct(J, 50), jump_p95=pct(J, 95), jump_p999=pct(J, 99.9), jump_gt2=float(np.mean(J > 2)))
                if ph.kind in ("blade", "fin"):
                    two = outside & (np.abs(side) == 1)
                    ps = ph.truth(P[two]).side
                    r["wrong_side"] = float(np.mean(ps != side[two]))
                    inside = ~outside & (dt < -0.3)
                    r["inside_sign_err"] = float(np.mean(de[inside] > 0)) if inside.any() else float("nan")
                if mesh is not None and scale == 1:
                    v, f = mesh
                    ma = M.min_angles(v, f)
                    ar = M.aspect_ratios(v, f)
                    da = M.dual_areas(v, f)
                    okv = away(ph, v)
                    r.update(minangle_p5=pct(ma, 5), minangle_med=pct(ma, 50), aspect_p50=pct(ar, 50),
                             aspect_p95=pct(ar, 95), aspect_max=pct(ar, 100),
                             dual_cv=float(np.std(da[okv]) / np.mean(da[okv])))
                res.setdefault("band", {}).setdefault(ph.name, {})[f"{nm} | {scale}x"] = r
                extra = (f" wrong side {r['wrong_side']:.4f} inside sign {r['inside_sign_err']:.4f}"
                         if "wrong_side" in r else "")
                tq = (f" tri minang p5 {r['minangle_p5']:.1f} aspect p95 {r['aspect_p95']:.2f} dualCV {r['dual_cv']:.2f}"
                      if "aspect_p95" in r else "")
                print(f"  {ph.name:10s} {scale}x {nm:18s} sd {r['sd_bias']:+.3f} ({r['sd_iqr']:.3f}, p95 {r['sd_p95']:.3f})"
                      f" foot {r['foot_med']:.3f}/{r['foot_p95']:.3f} tan {r['foot_tan_med']:.3f}/{r['foot_tan_p95']:.3f}"
                      f" sign {r['sign_err']:.4f} jump p95 {r['jump_p95']:.2f} p99.9 {r['jump_p999']:.2f} >2 {r['jump_gt2']:.4f}"
                      + extra + tq, flush=True)
            print(f"{ph.name} {scale}x: {time.time() - t0:.0f} s", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)


def main():
    res = json.load(open(OUT)) if Path(OUT).exists() else {}
    if WHAT in ("surface", "all"):
        part_surface(res)
        json.dump(res, open(OUT, "w"), indent=1)
    if WHAT in ("band", "all"):
        part_band(res)
        json.dump(res, open(OUT, "w"), indent=1)
    if WHAT in ("fin", "all"):
        part_fin(res)
        json.dump(res, open(OUT, "w"), indent=1)
    if WHAT in ("march", "all"):
        part_march(res)
        json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
