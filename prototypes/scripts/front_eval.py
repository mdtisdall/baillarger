"""The front method (baillarger.column_front) against crest and one-pass (plan: reports/Crossing
paths and contacts from laminar profiles.md, *Columns as a growing surface*). All 3D.

Phantoms (--phantoms; the fold phantoms of fold_phantoms.py at a site's features and measured
noise): per method (baillarger.column_traces.trace_columns on the phantom), roots >= 2 mm from a
face: the share of the right class (open or crossing), failed, own bank and to the contact; the
path, end and contact errors against the true columns (fold_eval.errors, mm); the front's
per-root dual area against the true column's (the offset surface's Bok area, from the phantom's
true depth and the root's curvatures by the area ratio of neighbouring true columns, see
area_truth) at 0.4, 0.8 and 1.2 mm; and the front's swept volume against the integral of that.

Sites (--sites; the column volumes built by column_volume.py, read from --dir): per method the
classes and failures by outcome (roots >= 2 mm from a face), the marked points that crossing and
to-the-contact columns pass within 0.3 mm of, the columns whose class changed against crest, and
own-bank columns split by their heading where they enter the WM (on, against the root's normal:
a crossing to a near wall, closer than the 5 mm bank rule; back: a return to its own bank), and
the front's path agreement with crest: per root, the largest distance between the two paths over
their common arc length, for roots where both end in the same outcome, and the front's dual area
and swept volume at the open columns' tops.

Usage: front_eval.py <work dir> --phantoms [--site V1V2_banks] [--noise 1.0]
       front_eval.py <work dir> --sites --dir DIR   (DIR holds column_volume_<site>_<method>.npz)
Writes <work>/front_eval_phantoms_<site>.json, <work>/front_eval_sites.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from scipy.spatial import cKDTree

from baillarger import column_traces as CT
from baillarger import streamlines as S

METHODS = ("crest", "one-pass", "front")
SITES = {"V1V2_banks": (198, 58, 65), "M1S1": (378, 224, 448)}
T0 = time.time()


def say(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def phantom_tissue(B):
    """A column_traces.Tissue from a fold phantom (fold_phantoms.make)."""
    return CT.Tissue(patch=None, vox=np.asarray(B["vox"], float), rim=B["rim"], miss=B["miss"],
                     ctx=B["ctx"], wm=B["wm"], det=B["det"], dface=B["dface"], labels=B["labels"])


def pct(a, q=(50, 90)):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return np.percentile(a, q).round(3).tolist() if a.size else None


def phantoms(A):
    import fold_eval as FE
    import fold_phantoms as FP

    res = dict(site_features=A.site, noise=A.noise, phantoms={})
    for pname, fold in FP.standard().items():
        B = FP.make(fold, A.site, noise=A.noise)
        t = phantom_tissue(B)
        row = {}
        paths = {}
        for m in METHODS:
            R = CT.trace_columns(t, "fitted_blend", m)
            x0 = R.v[R.tv]
            keep = ~R.near_face
            truth = fold.column(x0)
            tc = FE.true_class(truth)
            emax, _, eend, econ = FE.errors(fold, truth, R.pos, R.ns, t.vox)
            k = keep
            r = dict(roots=int(k.sum()),
                     right_class=round(float(np.mean(R.cls[k] == tc[k])), 3),
                     failed=round(float(np.mean(R.cls[k] == CT.FAILED)), 3),
                     own_bank=round(float(np.mean(R.cls[k] == CT.OWN)), 3),
                     to_contact=round(float(np.mean(R.cls[k] == CT.TOCONTACT)), 3),
                     path_err_p50_p90=pct(emax[k]), end_err_p50_p90=pct(eend[k]),
                     contact_err_p50_p90=pct(econ[k]) if np.isfinite(econ[k]).any() else None)
            if m == "front":
                r["descending (turn_at >= 0)"] = round(float(np.mean(R.turn_at[k] >= 0)), 3)
                r["area"] = front_area_vs_truth(fold, truth, R, k)
            paths[m] = (R.pos, R.ns, R.outc, k)
            row[m] = r
            say(f"{pname} {m}: right {r['right_class']}, failed {r['failed']}, own {r['own_bank']}, "
                f"path {r['path_err_p50_p90']}, end {r['end_err_p50_p90']}, contact {r['contact_err_p50_p90']}")
        row["front vs crest, path distance (mm) p50/p90/p99"] = agreement(paths["crest"], paths["front"],
                                                                          t.vox)
        res["phantoms"][pname] = row
    json.dump(res, open(f"{A.work}/front_eval_phantoms_{A.site}.json", "w"), indent=1)


def front_area_vs_truth(fold, truth, R, keep):
    """The front's dual area ratio A(s) / A(0) against the true column's at s = 0.4, 0.8, 1.2 mm,
    for kept roots whose front column and true column both reach s. The laminae are offsets of a
    core set, so the true column's Bok area at distance r from the core is (r / r0)^p, r0 the
    root's own distance and r = r0 +- s: p = 2 round a point (the crown's tip), 1 round a line (the
    crown's shaft, the fundus's edge, the kissing pair, across its contact too), 0 off a plane (the
    fundus's flat walls). And the front's swept volume against the integral of the true area."""
    Y = fold.to_local(R.v[R.tv])
    if fold.kind == "crown":
        p = np.where(Y[:, 2] > 0, 2, 1)
    elif fold.kind == "fundus":
        p = np.where(Y[:, 1] < 0, 1, 0)
    else:
        p = np.ones(len(Y), int)
    r0, r1 = truth["r0"], truth["r1"]
    sg = np.sign(r1 - r0)
    out = {}
    for s in (0.4, 0.8, 1.2):
        i = int(round(s / S.STEP_MM))
        m = keep & (R.ns > i) & (np.abs(r1 - r0) > s)
        if m.sum() < 5:
            continue
        r = r0[m] + sg[m] * s
        true = (r / r0[m]) ** p[m]
        got = R.area[m, i] / R.area[m, 0]
        vt = np.where(p[m] == 0, s, r0[m] / (p[m] + 1) * sg[m] * ((r / r0[m]) ** (p[m] + 1) - 1))
        vg = R.vol[m, i] / R.area[m, 0]
        out[f"{s} mm"] = dict(n=int(m.sum()),
                              area_front_over_true_p10_50_90=np.percentile(got / true, [10, 50, 90]).round(3).tolist(),
                              area_total_front_over_true=round(float((R.area[m, i]).sum() / (R.area[m, 0] * true).sum()), 4),
                              vol_total_front_over_true=round(float(R.vol[m, i].sum() / (R.area[m, 0] * vt).sum()), 4))
    return out


def agreement(a, b, vox):
    """Per root where both methods end in the same outcome: the largest distance between the two
    paths over their common arc length (mm); p50/p90/p99."""
    pa, na, oa, keep = a
    pb, nb, ob, _ = b
    same = np.flatnonzero(keep & (oa == ob))
    dev = []
    for i in same:
        n = min(na[i], nb[i])
        dev.append(float(np.max(np.linalg.norm((pa[i, :n] - pb[i, :n]).astype(float) * vox, axis=1))))
    return np.percentile(dev, [50, 90, 99]).round(3).tolist() if dev else None


def sites(A):
    res = {}
    for site, lo in SITES.items():
        Z = {m: np.load(f"{A.dir}/column_volume_{site}_{m}.npz") for m in METHODS
             if Path(f"{A.dir}/column_volume_{site}_{m}.npz").exists()}
        if "front" not in Z:
            continue
        vox = Z["front"]["vox"]
        Pts = Z["front"]["P"]
        r = {}
        for m, z in Z.items():
            keep = ~z["near_face"]
            cls, outc = z["cls"], z["outc"]
            d = dict(classes={CT.CLASS[c]: int(np.sum(keep & (cls == c))) for c in range(5)},
                     failed_by_outcome={S.OUTCOME[o]: int(np.sum(keep & (cls == CT.FAILED) & (outc == o)))
                                        for o in range(10) if np.sum(keep & (cls == CT.FAILED) & (outc == o))})
            if len(Pts):
                tree = cKDTree(Pts * vox)
                hit = {CT.CROSS: set(), CT.TOCONTACT: set()}
                for c in hit:
                    for i in np.flatnonzero(keep & (cls == c)):
                        dd, j = tree.query(z["pos"][i, :z["ns"][i]].astype(float) * vox)
                        hit[c].update(j[dd <= 0.3].tolist())
                d["marked points crossed (crossing columns)"] = len(hit[CT.CROSS])
                d["marked points reached (crossing or to the contact)"] = len(hit[CT.CROSS] | hit[CT.TOCONTACT])
                d["points"] = len(Pts)
            k_ = np.arange(len(cls))
            ns_ = z["ns"]
            d_end = (z["pos"][k_, ns_ - 1] - z["pos"][k_, np.maximum(ns_ - 2, 0)]).astype(float)
            fwd = (d_end * z["nrm"][z["tv"]]).sum(1) > 0
            own = keep & (cls == CT.OWN)
            d["own bank: entering the WM heading on / back"] = [int(np.sum(own & fwd)), int(np.sum(own & ~fwd))]
            ca = z["crest_at"]
            d["meeting a crest, by class"] = {CT.CLASS[c]: int(np.sum(keep & (cls == c) & (ca >= 0)))
                                              for c in range(5)}
            r[m] = d
        zf, zc = Z["front"], Z["crest"]
        keep = ~zf["near_face"]
        assert np.array_equal(zf["tv"], zc["tv"])
        r["front vs crest: path distance (mm) p50/p90/p99, same outcome"] = agreement(
            (zc["pos"], zc["ns"], zc["outc"], keep), (zf["pos"], zf["ns"], zf["outc"], keep), vox)
        ch = {}
        for a in range(5):
            for b in range(5):
                n = int(np.sum(keep & (zc["cls"] == a) & (zf["cls"] == b)))
                if n and a != b:
                    ch[f"{CT.CLASS[a]} -> {CT.CLASS[b]}"] = n
        r["front vs crest: class changes"] = ch
        ta = zf["turn_at"]
        r["front: descending, by class"] = {CT.CLASS[c]: int(np.sum(keep & (zf["cls"] == c) & (ta >= 0)))
                                            for c in range(5)}
        op = keep & (zf["cls"] == CT.OPEN)
        k = np.flatnonzero(op)
        a0, a1 = zf["area"][k, 0], zf["area"][k, zf["ns"][k] - 1]
        L = (zf["ns"][k] - 1) * S.STEP_MM
        r["front, open columns: top area / root area p10/50/90"] = pct(a1 / a0, (10, 50, 90))
        r["front, open columns: swept volume / (root area x length) p10/50/90"] = pct(
            zf["vol"][k, zf["ns"][k] - 1] / (a0 * L), (10, 50, 90))
        res[site] = r
        say(f"{site}: " + json.dumps(r))
    json.dump(res, open(f"{A.work}/front_eval_sites.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--phantoms", action="store_true")
    ap.add_argument("--sites", action="store_true")
    ap.add_argument("--site", default="V1V2_banks")
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--dir")
    A = ap.parse_args()
    if A.phantoms:
        phantoms(A)
    if A.sites:
        sites(A)


if __name__ == "__main__":
    main()
