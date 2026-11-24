"""Unequal banks: per-bank fronts across a fused stretch of the central sulcus (M1S1).

User's points (0-based crop, snapped to the WM boundary in plane i = 84, 85 in ITK-SNAP):
bank A (1) from (84, 91, 99) to (84, 52, 103); bank B (2) from (84, 43, 79) to (84, 86, 55).
Everything is computed in 3D; the plane only seeds and displays.

Per bank X: T_X fitted with T = 0 on X's WM wall only (data-steered, signed away from X).
Columns run from X's WM along grad T_X and stop at fluid (open column) or MARGIN mm past
equal arrival T_X >= T_Y (fused column). Calibration per bank, on its own open columns:
thickness t_X (median), and a landmark (the darkest point within 0.2-0.7 of t_X) as a
fraction of the column, by distance and by equivolume. On fused columns, contact rules:
  equal arrival           T_X = T_Y
  calibrated arrival      T_X / t_X = T_Y / t_Y (per-bank speed scaled by thickness)
  landmark, distance      s_landmark / f_X
  landmark, equivolume    v_landmark / f_X^ev
against the observed contact (darkest point beyond the landmark).
"""

import collections
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
from fit_T import fill, solve_field

S = sys.argv[1]
AX, PLANE = 0, 84
BANKS = {"A": [(84, 91, 99), (84, 52, 103)], "B": [(84, 43, 79), (84, 86, 55)]}
LAM, STEP, MARGIN = 0.1, 0.25, 1.6


def main():
    p = load_patch("M1S1")
    rim, vox = p.rim, p.voxel_mm
    vv = np.array(vox)
    miss = missing_tissue(rim, p.flash, vox)
    allp = np.array(BANKS["A"] + BANKS["B"])
    lo = np.array([PLANE - 30, allp[:, 1].min() - 22, allp[:, 2].min() - 22]).clip(0)
    hi = np.minimum(np.array([PLANE + 31, allp[:, 1].max() + 23, allp[:, 2].max() + 23]), rim.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    rb, missb = rim[box], miss[box]
    ctx = (rb == 3) & ~missb
    mb = ctx.astype(float)
    wm = rb == 2
    shell = wm & ~ndi.binary_erosion(wm)
    e = {i: np.log(np.maximum(p.echoes[box + (i,)].astype(float), 1.0)) for i in (1, 2)}
    det = {i: (L - F.mconv(L, mb, F.SHP)) * mb for i, L in e.items()}
    Tc = F.tensor(det[1], mb) + F.tensor(det[2], mb)
    Mt = np.stack([np.stack([Tc[0], Tc[1], Tc[2]], -1), np.stack([Tc[1], Tc[3], Tc[4]], -1),
                   np.stack([Tc[2], Tc[4], Tc[5]], -1)], -2)
    w_, V = np.linalg.eigh(Mt)
    nh = V[..., :, 2]
    coh = np.clip((w_[..., 2] - w_[..., 1]) / np.maximum(w_[..., 2] + w_[..., 1], 1e-12), 0, 1)
    disp = det[1]
    ip = PLANE - lo[0]
    d_out = ndi.distance_transform_edt(~wm, sampling=vox)
    gout = np.stack(np.gradient(ndi.gaussian_filter(d_out, 1.0), *vox), -1)
    gout /= np.linalg.norm(gout, axis=-1, keepdims=True) + 1e-12

    paths, walls, normals = {}, {}, {}
    for X, pts in BANKS.items():
        a = (pts[0][1] - lo[1], pts[0][2] - lo[2]); b = (pts[1][1] - lo[1], pts[1][2] - lo[2])
        p2 = F.boundary_path(shell[ip], a, b)
        path = np.stack([np.full(len(p2), ip), p2[:, 0], p2[:, 1]], 1)
        mn = gout[tuple(path.T)].mean(0); mn /= np.linalg.norm(mn)
        sm = np.zeros(rb.shape, bool); sm[tuple(path.T)] = True
        dpath = ndi.distance_transform_edt(~sm, sampling=vox)
        walls[X] = shell & (dpath < 3.0) & ((gout * mn).sum(-1) > 0.3)
        paths[X], normals[X] = path, mn
    both = walls["A"] & walls["B"]
    for X in walls:
        walls[X] &= ~both
    print(f"walls: A {walls['A'].sum()}, B {walls['B'].sum()} voxels; normals dot {np.dot(normals['A'], normals['B']):+.2f}")

    fields = {}
    for X in ("A", "B"):
        wall = walls[X]
        dom = ctx | wall
        dW = ndi.distance_transform_edt(~wall, sampling=vox)
        g = np.stack(np.gradient(ndi.gaussian_filter(dW, 1.0), *vox), -1)
        gh = g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-12)
        sg = np.sign((nh * gh).sum(-1)); sg[sg == 0] = 1
        dirv = coh[..., None] * nh * sg[..., None] + LAM * gh
        dirv /= np.linalg.norm(dirv, axis=-1, keepdims=True) + 1e-12
        T, info = solve_field(dom, wall, [dirv[..., a] for a in range(3)], vox)
        Tf = fill(T)
        gT = np.stack(np.gradient(Tf, *vox), -1)
        n1 = gT / (np.linalg.norm(gT, axis=-1, keepdims=True) + 1e-12)
        n1s = np.stack([ndi.gaussian_filter(n1[..., a], 1.0) for a in range(3)], -1)
        n1n = n1s / (np.linalg.norm(n1s, axis=-1, keepdims=True) + 1e-12)
        div = sum(np.gradient(n1n[..., a], vox[a], axis=a) for a in range(3))
        div = np.clip(np.where(ctx & (Tf > 0.3), div, 0.0), -2, 2)
        fields[X] = dict(T=T, Tf=Tf, g=[gT[..., a] for a in range(3)], div=div)
        print(f"T_{X}: cg info {info}")

    CONTACTS = [(84, 63, 82)]  # user's joining-line points, 0-based crop
    dA = ndi.distance_transform_edt(~walls["A"], sampling=vox)
    dB = ndi.distance_transform_edt(~walls["B"], sampling=vox)
    for c_ in CONTACTS:
        q = tuple(np.array(c_) - lo)
        print(f"contact point {c_} (crop): rim {rb[q]}, distance to A's wall {dA[q]:.2f} mm, to B's wall {dB[q]:.2f} mm, "
              f"T_A {fields['A']['Tf'][q]:.2f}, T_B {fields['B']['Tf'][q]:.2f}, coherence {coh[q]:.2f}, echo2 det {disp[q]:+.3f}")

    def trace(seed, X, Y):
        fX, fY = fields[X], fields[Y]
        x = np.array(seed, float)
        g0 = np.array([F.interp(c, x) for c in fX["g"]]); x = x + g0 / (np.linalg.norm(g0) + 1e-12)
        pts = [np.array(seed, float), x.copy()]
        s, s_eq = 0.16, None
        for _ in range(320):
            v = np.array([F.interp(c, x) for c in fX["g"]])
            nv = np.linalg.norm(v)
            if nv < 1e-9:
                return np.array(pts), "stall"
            x = x + STEP * v / nv
            s += STEP * 0.16
            pts.append(x.copy())
            q = tuple(np.clip(np.round(x).astype(int), 0, np.array(rb.shape) - 1))
            if rb[q] == 1:
                return np.array(pts), "open"
            if s_eq is None and F.interp(fX["Tf"], x) >= F.interp(fY["Tf"], x):
                s_eq = s
            if s_eq is not None and s >= s_eq + MARGIN:
                return np.array(pts), "fused"
            if wm[q] and s > 1.0:
                return np.array(pts), "WM"
        return np.array(pts), "length"

    def along(pts, X):
        sx = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0) * vv, axis=1))] + 0.08
        ev = ndi.gaussian_filter1d(ndi.map_coordinates(disp, pts.T, order=1), 1.5)
        ds = np.diff(sx, prepend=sx[0])
        u = np.cumsum(ndi.map_coordinates(fields[X]["div"], pts.T, order=1) * ds)
        return sx, ev, np.cumsum(np.exp(u) * ds)

    res = {}
    for X, Y in (("A", "B"), ("B", "A")):
        cols = [trace(sd, X, Y) for sd in paths[X]]
        print(f"bank {X}: {collections.Counter(st for _, st in cols)}")
        if X == "A":
            flab, nfl = ndi.label(rb == 1)
            sizes = np.bincount(flab.ravel())
            for pts, st in cols[::4]:
                q = tuple(np.clip(np.round(pts[-1]).astype(int), 0, np.array(rb.shape) - 1))
                L = np.linalg.norm(np.diff(pts, axis=0) * vv, axis=1).sum()
                comp = flab[q]
                print(f"   A column end {tuple(int(c) for c in np.array(q) + lo)} (crop), status {st}, length {L:.2f} mm, "
                      f"rim {rb[q]}, fluid component size {sizes[comp] if comp else 0}, "
                      f"T_A {F.interp(fields['A']['Tf'], pts[-1]):.2f}, T_B {F.interp(fields['B']['Tf'], pts[-1]):.2f}, "
                      f"coherence {coh[q]:.2f}")
        opn = [along(pts, X) for pts, st in cols if st == "open"]
        if not opn:
            res[X] = dict(cols=cols, cal=None); continue
        t = np.array([o[0][-1] for o in opn]); tv = np.array([o[2][-1] for o in opn])
        tmed = np.median(t)
        fus_, fev_ = [], []
        for (sx, ev, v) in opn:
            m = (sx >= 0.2 * tmed) & (sx <= 0.7 * tmed)
            k_ = np.flatnonzero(m)[np.argmin(ev[m])]
            fus_.append(sx[k_] / sx[-1]); fev_.append(v[k_] / v[-1])
        cal = dict(t=tmed, t_iqr=np.subtract(*np.percentile(t, [75, 25])), n=len(opn),
                   f_us=np.median(fus_), f_us_iqr=np.subtract(*np.percentile(fus_, [75, 25])),
                   f_ev=np.median(fev_), f_ev_iqr=np.subtract(*np.percentile(fev_, [75, 25])))
        print(f"   calibration on {len(opn)} open columns: thickness {tmed:.2f} mm (IQR {cal['t_iqr']:.2f}); "
              f"landmark fraction distance {cal['f_us']:.3f} (IQR {cal['f_us_iqr']:.3f}), "
              f"equivolume {cal['f_ev']:.3f} (IQR {cal['f_ev_iqr']:.3f})")
        res[X] = dict(cols=cols, cal=cal)

    for X, Y in (("A", "B"), ("B", "A")):
        cal, calY = res[X]["cal"], res[Y]["cal"]
        recs = []
        for pts, st in res[X]["cols"]:
            if st != "fused" or cal is None or calY is None:
                recs.append(None); continue
            sx, ev, v = along(pts, X)
            TX = ndi.map_coordinates(fields[X]["Tf"], pts.T, order=1)
            TY = ndi.map_coordinates(fields[Y]["Tf"], pts.T, order=1)
            ieq = np.flatnonzero(TX >= TY)
            ical = np.flatnonzero(TX / cal["t"] >= TY / calY["t"])
            m = (sx >= 0.2 * cal["t"]) & (sx <= 0.7 * cal["t"])
            if not m.any() or not len(ieq):
                recs.append(None); continue
            k_ = np.flatnonzero(m)[np.argmin(ev[m])]
            mc = sx >= sx[k_] + 0.4
            kc = np.flatnonzero(mc)[np.argmin(ev[mc])] if mc.any() else None
            edge = kc is None or sx[kc] > sx[-1] - 0.15
            tgt = v[k_] / cal["f_ev"]
            recs.append(dict(s=sx, e=ev, land=sx[k_], contact=np.nan if edge else sx[kc], end=sx[-1],
                             eq=sx[ieq[0]], cal=sx[ical[0]] if len(ical) else np.nan,
                             lm_us=sx[k_] / cal["f_us"], lm_ev=np.interp(tgt, v, sx) if tgt <= v[-1] else np.nan))
        res[X]["recs"] = recs
        ok = [r for r in recs if r is not None and not np.isnan(r["contact"])]
        print(f"bank {X}: {len(ok)} fused columns with a contact inside the window")
        if ok:
            K = {k: np.array([r[k] for r in ok]) for k in ("contact", "eq", "cal", "lm_us", "lm_ev", "land")}
            res[X]["K"] = K
            print(f"   observed contact {np.median(K['contact']):.2f} mm from {X}'s WM (IQR "
                  f"{np.subtract(*np.percentile(K['contact'], [75, 25])):.2f}); landmark at {np.median(K['land']):.2f}")
            for nm, lab in (("eq", "equal arrival"), ("cal", "thickness-calibrated arrival"),
                            ("lm_us", "landmark -> top, distance"), ("lm_ev", "landmark -> top, equivolume")):
                d = K[nm] - K["contact"]
                print(f"   {lab:30s} - contact: median {np.nanmedian(d):+.2f} mm, IQR "
                      f"{np.subtract(*np.nanpercentile(d, [75, 25])):.2f}, |offset| {np.nanmedian(np.abs(d)):.2f}")

    # ---------- user's joining-line points as the reference contact
    CONTACTS = [(84, 56, 86), (84, 63, 82), (84, 67, 80), (84, 73, 77)]  # 0-based crop
    def back(x, X):
        """From a contact point down -grad T_X to X's WM: X's own column through the point."""
        pts = [np.array(x, float)]
        for _ in range(240):
            vq = -np.array([F.interp(c, pts[-1]) for c in fields[X]["g"]])
            nv = np.linalg.norm(vq)
            if nv < 1e-9:
                return None
            pts.append(pts[-1] + STEP * vq / nv)
            q = tuple(np.clip(np.round(pts[-1]).astype(int), 0, np.array(rb.shape) - 1))
            if wm[q]:
                return np.array(pts[::-1])
        return None
    cres = []
    for c_ in CONTACTS:
        x = np.array(c_, float) - lo
        row = {"pt": c_}
        for X, Y in (("A", "B"), ("B", "A")):
            col = back(x, X)
            cal, calY = res[X]["cal"], res[Y]["cal"]
            if col is None or cal is None or calY is None:
                continue
            q0 = tuple(np.round(col[0]).astype(int))
            sx, ev, v = along(col, X)
            sc = sx[-1]
            TX = ndi.map_coordinates(fields[X]["Tf"], col.T, order=1)
            TY = ndi.map_coordinates(fields[Y]["Tf"], col.T, order=1)
            ieq = np.flatnonzero(TX >= TY)
            ical = np.flatnonzero(TX / cal["t"] >= TY / calY["t"])
            m = (sx >= 0.2 * cal["t"]) & (sx <= 0.7 * cal["t"])
            k_ = np.flatnonzero(m)[np.argmin(ev[m])] if m.any() else None
            # predictions may lie beyond the contact: extend the column straight on for them
            ext = sc + np.arange(1, 40) * 0.04
            row[X] = dict(col=col, on_own_wall=bool(walls[X][q0]), contact=sc,
                          eq=sx[ieq[0]] if len(ieq) else np.inf,
                          cal=sx[ical[0]] if len(ical) else np.inf,
                          lm_us=sx[k_] / cal["f_us"] if k_ is not None else np.nan,
                          lm_ev=(np.interp(v[k_] / cal["f_ev"], v, sx) if k_ is not None and v[k_] / cal["f_ev"] <= v[-1]
                                 else (sc + (v[k_] / cal["f_ev"] - v[-1]) if k_ is not None else np.nan)),
                          land=sx[k_] if k_ is not None else np.nan)
        cres.append(row)
    print("contact rules against the user's joining-line points (offset = predicted - marked, mm along each bank's own column):")
    for row in cres:
        for X in ("A", "B"):
            if X not in row:
                print(f"  {row['pt']} bank {X}: no column"); continue
            r = row[X]
            def f(z):
                return "beyond" if np.isinf(z) else f"{z - r['contact']:+.2f}"
            print(f"  {row['pt']} bank {X}: column from own wall {r['on_own_wall']}, contact at {r['contact']:.2f} mm; "
                  f"equal arrival {f(r['eq'])}, calibrated {f(r['cal'])}, landmark distance {f(r['lm_us'])}, "
                  f"landmark equivolume {f(r['lm_ev'])} (landmark at {r['land']:.2f})")

    # ---------- larger diagnostic view of the plane
    raw = e[1]
    fig2, axs2 = plt.subplots(1, 3, figsize=(27, 9.5))
    sl = (ip, slice(None), slice(None))
    rv = np.nanpercentile(raw[ctx], [1, 99.5])
    stc = {"open": "#2ca02c", "fused": "#ffffff", "stall": "#d62728", "WM": "#9467bd", "length": "#d62728"}
    for k_, ax in enumerate(axs2):
        ax.imshow(raw[sl].T, origin="lower", cmap="gray", vmin=rv[0], vmax=rv[1])
        ax.contour(wm[sl].T.astype(float), levels=[0.5], colors="y", linewidths=0.8)
        ax.contour((rb == 1)[sl].T.astype(float), levels=[0.5], colors="c", linewidths=0.8)
        ax.imshow(np.where(walls["A"][sl], 1.0, np.nan).T, origin="lower", cmap="Blues", vmin=0, vmax=1.1)
        ax.imshow(np.where(walls["B"][sl], 1.0, np.nan).T, origin="lower", cmap="Oranges", vmin=0, vmax=1.1)
        for X in ("A", "B"):
            for q_ in BANKS[X]:
                ax.plot(q_[1] - lo[1], q_[2] - lo[2], "o", mfc="none", mec="m", ms=10, mew=2)
        ax.set_xlabel("j (crop, 0-based) - offset"); ax.set_ylabel("k")
    axs2[0].set_title("raw echo 2; yellow = WM (aseg), cyan = fluid (aseg),\nblue/orange = labelled walls A/B, magenta = your points")
    for ax, X in zip(axs2[1:], ("A", "B")):
        cnt = collections.Counter()
        for pts, st in res[X]["cols"]:
            cnt[st] += 1
            ax.plot(pts[:, 1], pts[:, 2], "-", c=stc.get(st, "r"), lw=0.9)
            ax.plot(pts[-1, 1], pts[-1, 2], "o", c=stc.get(st, "r"), ms=3)
        ax.contour(np.where(ctx, fields["A"]["T"] - fields["B"]["T"], np.nan)[sl].T, levels=[0], colors="lime", linewidths=1.2)
        if res["A"]["cal"] is not None and res["B"]["cal"] is not None:
            ax.contour(np.where(ctx, fields["A"]["T"] / res["A"]["cal"]["t"] - fields["B"]["T"] / res["B"]["cal"]["t"], np.nan)[sl].T,
                       levels=[0], colors="m", linewidths=1.2)
        for c_ in CONTACTS:
            ax.plot(c_[1] - lo[1], c_[2] - lo[2], "x", c="c", ms=10, mew=2.5)
        for row in cres:
            if X in row:
                cc = row[X]["col"]
                ax.plot(cc[:, 1], cc[:, 2], "-", c="c", lw=1.6)
        ax.set_title(f"bank {X}'s columns (3D, projected): " + ", ".join(f"{k} {n}" for k, n in cnt.items()) +
                     "\ngreen = reached fluid, white = fused, red = stalled; lime = equal arrival, magenta = calibrated arrival;\ncyan x = your joining points, cyan line = the bank's column through each")
    fig2.suptitle("M1S1 plane i = 85 (ITK-SNAP numbering): what was selected and where each bank's columns went", fontsize=14)
    fig2.tight_layout()
    fig2.savefig(f"{S}/m1s1_plane_diag.png", dpi=85)
    print("wrote", f"{S}/m1s1_plane_diag.png")

    # ---------- figure
    fig = plt.figure(figsize=(26, 14))
    gs = fig.add_gridspec(2, 3)
    sl = (ip, slice(None), slice(None))
    v = np.nanpercentile(np.abs(disp[ctx]), 98)
    allpath = np.vstack([paths["A"], paths["B"]])
    xr = (allpath[:, 1].min() - 14, allpath[:, 1].max() + 14)
    yr = (allpath[:, 2].min() - 14, allpath[:, 2].max() + 14)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(np.where(ctx, disp, np.nan)[sl].T, origin="lower", cmap="gray", vmin=-v, vmax=v)
    ax.contour(shell[sl].T.astype(float), levels=[0.5], colors="y", linewidths=0.6)
    ax.contour(np.where(ctx, fields["A"]["T"] - fields["B"]["T"], np.nan)[sl].T, levels=[0], colors="lime", linewidths=1.2)
    for X, col in (("A", "#4a90d9"), ("B", "#e8913a")):
        for pts, st in res[X]["cols"]:
            ax.plot(pts[:, 1], pts[:, 2], "-", c=col if st in ("open", "fused") else "r",
                    lw=0.7, ls="-" if st == "fused" else ":")
        ax.plot(paths[X][:, 1], paths[X][:, 2], "-", c=col, lw=2.2)
    ax.set_xlim(*xr); ax.set_ylim(*yr); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("columns from each bank's own WM (A blue, B orange):\nsolid = fused (stopped past equal arrival), dotted = open (reached fluid), red = other;\n"
                 "green = equal arrival", fontsize=10)
    ax = fig.add_subplot(gs[1, 0]); ax.axis("off")
    rows = []
    for X in ("A", "B"):
        c = res[X]["cal"]
        if c is not None:
            rows.append([f"bank {X}", "calibration (open columns)", f"t = {c['t']:.2f} mm", f"f = {c['f_us']:.2f} / {c['f_ev']:.2f}", f"n = {c['n']}"])
        if "K" in res[X]:
            K = res[X]["K"]
            for nm, lab in (("eq", "equal arrival"), ("cal", "calibrated arrival"), ("lm_us", "landmark, distance"), ("lm_ev", "landmark, equivolume")):
                d = K[nm] - K["contact"]
                rows.append([f"bank {X}", lab, f"{np.nanmedian(d):+.2f}", f"{np.subtract(*np.nanpercentile(d, [75, 25])):.2f}", f"{np.nanmedian(np.abs(d)):.2f}"])
    if rows:
        tb = ax.table(cellText=rows, colLabels=["bank", "rule", "median offset (mm)", "IQR (mm)", "median |offset| (mm)"],
                      loc="center", cellLoc="center")
        tb.scale(1, 2.0); tb.set_fontsize(10)
    ax.set_title("calibration and each rule's offset from the observed contact (own columns)")
    for c_, X in enumerate(("A", "B")):
        recs = res[X].get("recs", [])
        path = paths[X]
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0) * vv, axis=1))]
        nb = 70
        img = np.full((nb, len(path)), np.nan)
        for c, (pts, st) in enumerate(res[X]["cols"]):
            if st not in ("open", "fused"):
                continue
            sx, ev, _ = along(pts, X)
            q = np.arange(nb) * 0.08 + 0.04
            okq = q <= sx[-1]
            img[okq, c] = np.interp(q[okq], sx, ev)
        for row, (nm, lab) in enumerate((("lm_us", "distance"), ("lm_ev", "equivolume"))):
            ax = fig.add_subplot(gs[row, 1 + c_])
            ax.imshow(img, origin="lower", aspect="auto", cmap="gray", vmin=-v, vmax=v, extent=[0, arc[-1], 0, nb * 0.08])
            idx = [c for c, r in enumerate(recs) if r is not None and not np.isnan(r["contact"])]
            if idx:
                xs = arc[idx]
                g = lambda k: np.array([recs[c][k] for c in idx])
                ax.plot(xs, g("contact"), "o", mfc="none", mec="c", ms=7, label="observed contact")
                ax.plot(xs, g("eq"), "_", c="lime", ms=12, mew=2, label="equal arrival")
                ax.plot(xs, g("cal"), "_", c="m", ms=12, mew=2, label="thickness-calibrated arrival")
                ax.plot(xs, g(nm), "v", c="#4a90d9" if X == "A" else "#e8913a", ms=6, label=f"landmark → top ({lab})")
                ax.plot(xs, g("land"), ".", c="w", ms=5, label="landmark")
            if res[X]["cal"] is not None:
                ax.axhline(res[X]["cal"]["t"], c="y", lw=0.8, ls="--", label="open-column thickness")
            ax.set_xlabel(f"position along bank {X}'s WM path (mm)")
            ax.set_ylabel(f"distance from bank {X}'s WM along its own column (mm)")
            ax.set_title(f"bank {X}'s own columns, echo 2 (trend removed); landmark → top by {lab}")
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Unequal banks, M1S1 central sulcus, plane i = 85 (ITK-SNAP numbering): per-bank fronts and contact rules; "
                 "all computed in 3D", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{S}/per_bank_m1s1.png", dpi=90)
    print("wrote", f"{S}/per_bank_m1s1.png")


if __name__ == "__main__":
    main()
