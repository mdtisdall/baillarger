"""Feasibility: edits that change the one-pass competition's inputs, with the pial surface
always read off the resulting partition (one owner per cortex voxel), so that it cannot cross
at a contact. Tested on the 23 marked M1S1 contact points (M1S1_test_points.json, "gap_all")
as contact clicks. The editor's code is not used or changed. All 3D.

Setup as click_evidence.py (finding 25's box of 141 voxels on the points; cost ratio 3; root
patches of 6 voxels; the thickness field's weights from column tops: the starting point).

An edit (a contact click at x): the two fronts at x are the end labels of the column line
through it (score_points_sides' sides); each one's arrival at x, S, from the current pass. The
two roots' expected thickness is set to their S (log correction log S - log mu), so both reach
their belief exactly at x and the boundary passes there; sigma unchanged. Variants:
- roots: only those two roots;
- spread 1 mm: the correction also given to the roots around each, weighted
  exp(-d^2 / 2 (0.5 mm)^2) by the distance d along the white surface (seed graph) to the edited
  root, up to 1.5 mm (a weighted mean where several edits reach a root).
Each in 2 rounds (the pass rerun, the end labels and S read again, the corrections applied on
top), since the territories move.

Scored (finding 25's local two-sides score, |offset| of the boundary from each point):
before; all 23 placed (in sample); interleaved (even placed, odd scored, and the reverse).

The surface without a slit: the exposed voxel faces of the tissue (cortex, WM, missing tissue)
against fluid, plus one face for each contact pair (label boundary between roots > 18 voxels
apart, not fold-like, both voxels >= 2 mm from a box face), shared by the two banks: a single
sheet with the two banks on its two sides. Contact faces are moved to the sub-voxel contact
(the pass's frac), then the mesh is relaxed and smoothed as the base (twicing at 2 voxels).
Reported: triangles, edges with more than two triangles (where a contact sheet meets the open
surface), crossing triangles (baillarger.mesh_intersect), and each point's distance to the
contact sheet. Before and after all 23 edits (spread variant), with times per pass and per
extraction.

Writes <work>/owner_edit_M1S1.json, <figures>/owner_edit_M1S1.png.

Usage: owner_edit.py <figures dir> <work dir> <data dir>
"""

import json
import sys
import time

sys.path.insert(0, sys.argv[2])
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import dijkstra

import curv_lib as C
import mesh_smoothing as M
import pial_height as P_
from baillarger.mesh_intersect import intersecting_faces
from click_evidence import FULL, HALF, PATCH, RATIO, FACE_MM, Thickness, end_labels, orientation, score
from onepass_lib import contacts, run, setup

FIG, WORK, DATA = sys.argv[1:4]
SEP, SPREAD_MM, SPREAD_MAX_MM, ROUNDS = 18.0, 0.5, 1.5, 2
T0 = time.time()


def say(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def side_S(s, f, mu, sd, P, u_pts, h, idx):
    """Per point in idx: [(root, S mm)] for its two fronts, from pass f."""
    rows = score(s, f, mu, sd, P, u_pts, h, idx)
    ends = end_labels(s, f, P[idx], u_pts[idx], h)
    out = []
    for (lm, lp), r in zip(ends, rows):
        if lm < 0 or not np.isfinite(r["S_thin"]):
            out.append([]); continue
        thin = lm if mu[lm] < mu[lp] else lp  # as score_points_sides
        thick = lp if thin == lm else lm
        out.append([(int(thin), float(r["S_thin"])), (int(thick), float(r["S_thick"]))])
    return out


def apply(s, mu, sd, P, u_pts, h, idx, spread, timing):
    """mu after the edits at points idx (ROUNDS rounds); the last pass."""
    mu = mu.copy()
    for _ in range(ROUNDS):
        t = time.time(); f = run(s, mu, sd, ratio=RATIO); timing.append(time.time() - t)
        corr = {}
        for sides in side_S(s, f, mu, sd, P, u_pts, h, idx):
            for root, S in sides:
                if S > 0.3:
                    corr.setdefault(root, []).append(np.log(S / (mu[root] * h)))
        roots = np.array(list(corr), int)
        c = np.array([np.mean(corr[r]) for r in roots])
        if not spread:
            mu[roots] *= np.exp(c)
            continue
        d = dijkstra(s.graph, indices=s.rep[roots], limit=SPREAD_MAX_MM / h)[:, s.rep] * h  # (edits, patches) mm
        w = np.exp(-0.5 * (d / SPREAD_MM) ** 2)
        w[~np.isfinite(d)] = 0.0
        wsum = w.sum(0)
        has = wsum > 0
        mean_c = (w * c[:, None]).sum(0)[has] / wsum[has]
        mu[has] *= np.exp(mean_c * np.minimum(1.0, w.max(0)[has]))  # the edited roots themselves get w = 1
    t = time.time(); f = run(s, mu, sd, ratio=RATIO); timing.append(time.time() - t)
    return mu, f


def offsets(rows):
    o = np.array([r["offset"] for r in rows])
    return dict(median_abs=round(float(np.nanmedian(np.abs(o))), 3), median=round(float(np.nanmedian(o)), 3),
                worst=round(float(np.nanmax(np.abs(o))), 3), missing=int(np.isnan(o).sum()))


def shared_surface(B, s, f, mu, sd):
    """The tissue's exposed voxel faces plus one shared face per contact pair, welded, the
    contact faces at the sub-voxel contact, relaxed and smoothed. Returns v, f, is_contact."""
    tissue = (B["rim"] == 2) | (B["rim"] == 3) | B["miss"]
    shape = np.array(tissue.shape)
    quads, shift, kind = [], [], []
    for a in range(3):
        b_, c_ = (a + 1) % 3, (a + 2) % 3
        eb, ec = np.zeros(3), np.zeros(3); eb[b_], ec[c_] = 0.5, 0.5
        for sgn in (-1, 1):
            nb = np.ones_like(tissue)
            src = [slice(None)] * 3; dst = [slice(None)] * 3
            if sgn > 0:
                src[a], dst[a] = slice(1, None), slice(0, -1)
            else:
                src[a], dst[a] = slice(0, -1), slice(1, None)
            nb[tuple(dst)] = tissue[tuple(src)]
            ctr = np.argwhere(tissue & ~nb).astype(float)
            ctr[:, a] += 0.5 * sgn
            q = np.stack([ctr - eb - ec, ctr + eb - ec, ctr + eb + ec, ctr - eb + ec], 1)
            quads.append(q if sgn > 0 else q[:, ::-1]); shift.append(np.zeros((len(q), 3))); kind.append(np.zeros(len(q), bool))
    c = contacts(s, f, mu, sd, sep_min=SEP)
    ra, rb = c.da / mu[c.a], c.db / mu[c.b]
    keep = ~((ra <= 0.8) & (rb <= 0.8))
    keep &= (B["dface"].ravel()[c.p] >= FACE_MM) & (B["dface"].ravel()[c.q] >= FACE_MM)
    keep &= tissue.ravel()[c.p] & tissue.ravel()[c.q]
    pc = np.stack(np.unravel_index(c.p[keep], tissue.shape), -1).astype(float)
    qc = np.stack(np.unravel_index(c.q[keep], tissue.shape), -1).astype(float)
    ax, fr = c.axis[keep], c.frac[keep]
    step = qc - pc  # a unit step along ax
    ctr = 0.5 * (pc + qc)
    e1 = np.zeros((len(ax), 3)); e2 = np.zeros((len(ax), 3))
    e1[np.arange(len(ax)), (ax + 1) % 3] = 0.5; e2[np.arange(len(ax)), (ax + 2) % 3] = 0.5
    quads.append(np.stack([ctr - e1 - e2, ctr + e1 - e2, ctr + e1 + e2, ctr - e1 + e2], 1))
    shift.append((fr - 0.5)[:, None] * step); kind.append(np.ones(len(ax), bool))
    Q = np.concatenate(quads); SH = np.concatenate(shift); K = np.concatenate(kind)
    tri = np.concatenate([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]])
    tsh = np.concatenate([SH, SH]); tk = np.concatenate([K, K])
    key = np.round(tri.reshape(-1, 3) * 2).astype(np.int64)
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    v = uk / 2.0
    fc = inv.reshape(-1, 3)
    # sub-voxel: each vertex on contact faces moves by the mean shift of those faces
    acc = np.zeros_like(v); cnt = np.zeros(len(v))
    for k in range(3):
        np.add.at(acc, fc[tk, k], tsh[tk]); np.add.at(cnt, fc[tk, k], 1)
    v[cnt > 0] += acc[cnt > 0] / cnt[cnt > 0, None]
    v = M.twice(M.relax(v, fc), fc, 2.0)
    return v, fc, tk, int(keep.sum())


def surface_report(v, fc, tk, P, h):
    E = np.sort(np.concatenate([fc[:, [0, 1]], fc[:, [1, 2]], fc[:, [2, 0]]]), 1)
    _, cnt = np.unique(E, axis=0, return_counts=True)
    xf, _ = intersecting_faces(v, fc)
    cl, _, _ = M.closest_on_mesh(v, fc[tk], P)
    d = np.linalg.norm(cl - P, axis=1) * h
    return dict(triangles=int(len(fc)), contact_triangles=int(tk.sum()), edges_over_two_triangles=int(np.sum(cnt > 2)),
                crossing_triangles=int(xf.sum()), crossing_contact_triangles=int((xf & tk).sum()),
                point_to_contact_sheet_mm=dict(median=round(float(np.median(d)), 3), p90=round(float(np.percentile(d, 90)), 3),
                                               worst=round(float(d.max()), 3)), d=d.round(3).tolist())


def main():
    m = json.load(open(f"{DATA}/M1S1_test_points.json"))
    full = np.array(m["gap_all"]) + np.array(m["full_lo"])
    cen = np.round(full.mean(0)).astype(int)
    lo, hi = cen - HALF, cen + HALF + 1
    assert np.all(lo >= 0) and np.all(hi <= FULL)
    P = (full - lo).astype(float)
    n = len(P)
    B = C.load(lo, hi, "click_evidence_M1S1")
    h = float(np.mean(B["vox"]))
    dirv = orientation(B)
    s = setup(B["wm"], B["ctx"], patch=PATCH, direction=dirv)
    u_pts = dirv[tuple(np.round(P).astype(int).T)]
    del dirv
    T = Thickness(B, s, h)
    mu0, sd0, _ = T.weights(T.y_tops)
    t = time.time(); f0 = run(s, mu0, sd0, ratio=RATIO); t_pass = time.time() - t
    say(f"setup; one pass {t_pass:.1f} s ({s.domain.sum():,} voxels)")
    out = dict(pass_seconds=round(t_pass, 2), before=offsets(score(s, f0, mu0, sd0, P, u_pts, h, np.arange(n))))
    say(f"before: {out['before']}")
    keep = {}
    for variant, spread in (("roots", False), ("spread 1 mm", True)):
        timing: list = []
        mu_all, f_all = apply(s, mu0, sd0, P, u_pts, h, np.arange(n), spread, timing)
        res = {"all 23": offsets(score(s, f_all, mu_all, sd0, P, u_pts, h, np.arange(n)))}
        rows = []
        for place, held in ((np.arange(0, n, 2), np.arange(1, n, 2)), (np.arange(1, n, 2), np.arange(0, n, 2))):
            mu_h, f_h = apply(s, mu0, sd0, P, u_pts, h, place, spread, timing)
            rows += score(s, f_h, mu_h, sd0, P, u_pts, h, held)
        res["interleaved (held out)"] = offsets(rows)
        res["roots_changed_all23"] = int(np.sum(np.abs(np.log(mu_all / mu0)) > 1e-6))
        res["pass_seconds_median"] = round(float(np.median(timing)), 2)
        out[variant] = res
        keep[variant] = (mu_all, f_all)
        say(f"{variant}: {res}")

    # the surface without a slit, before and after (spread variant)
    surf = {}
    for name, (mu, f) in (("before", (mu0, f0)), ("after all 23 (spread 1 mm)", keep["spread 1 mm"])):
        t = time.time()
        v, fc, tk, npairs = shared_surface(B, s, f, mu, sd0)
        te = time.time() - t
        rep = surface_report(v, fc, tk, P, h)
        rep.update(contact_pairs=npairs, extraction_seconds=round(te, 1))
        surf[name] = rep
        keep[name] = (v, fc, tk)
        say(f"surface {name}: " + json.dumps({k: x for k, x in rep.items() if k != "d"}))
    out["surface"] = surf
    json.dump(out, open(f"{WORK}/owner_edit_M1S1.json", "w"), indent=1, default=float)

    img = np.where(B["ctx"], B["disp"], np.nan)
    lim = np.nanpercentile(np.abs(img), 98)
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    for row, name in enumerate(("before", "after all 23 (spread 1 mm)")):
        v, fc, tk = keep[name]
        for a in range(3):
            ax = axs[row][a]
            vals, cnt = np.unique(np.round(P[:, a]).astype(int), return_counts=True)
            c = vals[np.argmax(cnt)]
            o = [i for i in range(3) if i != a]
            sl = [slice(None)] * 3; sl[a] = c
            ax.imshow(img[tuple(sl)].T, cmap="gray", origin="lower", vmin=-lim, vmax=lim, interpolation="nearest")
            for ff, col, lw in ((fc[~tk], "#eb6834", 1.0), (fc[tk], "#2a78d6", 1.6)):
                for sg in P_.plane_section(v, ff, a, float(c)):
                    ax.plot(sg[:, o[0]], sg[:, o[1]], color=col, lw=lw)
            on = np.abs(P[:, a] - c) < 1.5
            ax.plot(P[on, o[0]], P[on, o[1]], "x", color="#e34948", ms=7, mew=2)
            lo_, hi_ = P[:, o].min(0) - 25, P[:, o].max(0) + 25
            ax.set_xlim(lo_[0], hi_[0]); ax.set_ylim(lo_[1], hi_[1]); ax.set_xticks([]); ax.set_yticks([])
            r = surf[name]
            ax.set_title(f"{name}, plane {'ijk'[a]} = {c}" + (
                f"\npoints to contact sheet: median {r['point_to_contact_sheet_mm']['median']:.2f} mm; "
                f"crossing triangles {r['crossing_triangles']}" if a == 0 else ""), fontsize=10)
    fig.suptitle("M1S1, the pial surface read off the one pass's owners (no slit): open surface (orange), the contact "
                 "as one shared sheet (blue); red x: the 23 points. Echo 2, trend removed.", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{FIG}/owner_edit_M1S1.png", dpi=110)
    say("done")


if __name__ == "__main__":
    main()
