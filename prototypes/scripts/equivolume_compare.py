"""Equivolume columns against the normal columns (--columns normal) at the marked points of both
sites (plan: reports/Crossing paths and contacts from laminar profiles.md, *Equivolume columns
against the normal columns*): --columns equivolume (the same straight rays, vol measured on the
offset mesh) and --columns front-equivolume (the front grown at equivolume speed, columns bent
normal to its layers). All 3D.

Reads the column volumes that column_volume.py wrote for each method (the same box, handle-free
WM and white mesh, so the same roots): <dir>/column_volume_<site>_<method>.npz and .json, one
directory per method. Roots >= 2 mm from a box face. Per site:
- classes and failures by outcome, per method;
- at the marked points, per method, the columns passing within 0.3 mm of a point (any class) and
  at their nearest sample: the arc length, the equivolume coordinate (normal: vol, from the
  root's curvatures; front-equivolume: tau, the front's time) and, for open columns, the relative
  equivolume depth (coordinate over the coordinate at the column's top); one value per point,
  the median over its columns;
- the path difference between each method and normal, per root passing within 0.3 mm of a point by either:
  the largest distance between its two paths over their common arc length, and the distance at
  the point's arc length;
- the profile split at the points, as column_volume.py scored it (|split - point depth|, mm).

Usage: equivolume_compare.py <out dir> --normal DIR --equivolume DIR --front DIR
Writes <out>/equivolume_compare.json.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from scipy.spatial import cKDTree

from baillarger import column_traces as CT
from baillarger import streamlines as S

SITES = ("V1V2_banks", "M1S1")
NEAR_MM = 0.3
OUTCOME = {
    S.WM: "WM",
    S.FLUID: "fluid",
    S.MISSING: "missing",
    S.FACE: "face",
    S.OTHER: "other",
    S.TURN: "turn",
    S.LENGTH: "length",
    S.CONTACT: "contact",
    S.RIDGE: "ridge",
}


def pct(a, q=(50, 90)):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return np.percentile(a, q).round(3).tolist() if a.size else None


def load(d, site, method):
    Z = np.load(f"{d}/column_volume_{site}_{method}.npz")
    keys = ["tv", "pos", "ns", "cls", "outc", "near_face", "P", "vox", "crest_at"]
    out = {k: Z[k] for k in keys}
    # the equivolume coordinate: front-equivolume's tau (the front's time), else vol (mm)
    out["eqv"] = Z["tau"] if "tau" in Z.files else Z["vol"]
    out["json"] = json.load(open(f"{d}/column_volume_{site}_{method}.json"))
    return out


def passing(M, keep):
    """Per point: the columns (kept) passing within NEAR_MM, and each one's nearest sample."""
    vox = M["vox"]
    tree = cKDTree((M["P"] * vox).astype(float))
    hits = {k: [] for k in range(len(M["P"]))}
    for t in np.flatnonzero(keep):
        p = M["pos"][t, : M["ns"][t]].astype(float) * vox
        d, k = tree.query(p)
        for kk in np.unique(k[d <= NEAR_MM]):
            sel = np.flatnonzero((k == kk) & (d <= NEAR_MM))
            hits[int(kk)].append((int(t), int(sel[np.argmin(d[sel])])))
    return hits


def site(A, s):
    dirs = {"normal": A.normal, "equivolume": A.equivolume, "front-equivolume": A.front}
    M = {m: load(d, s, m) for m, d in dirs.items()}
    N = M["normal"]
    for m in M:
        assert np.array_equal(N["tv"], M[m]["tv"]) and np.allclose(N["pos"][:, 0], M[m]["pos"][:, 0]), \
            f"{m} does not share normal's roots"
    keep = ~N["near_face"]
    R = dict(roots=int(keep.sum()))
    for name, X in M.items():
        R[name] = dict(
            classes={CT.CLASS[c]: int(np.sum(keep & (X["cls"] == c))) for c in range(5)},
            failed_by_outcome={OUTCOME[o]: int(np.sum(keep & (X["cls"] == CT.FAILED) & (X["outc"] == o)))
                               for o in OUTCOME if np.sum(keep & (X["cls"] == CT.FAILED) & (X["outc"] == o))},
            open_meeting_another_front=round(float(np.mean(X["crest_at"][keep & (X["cls"] == CT.OPEN)] >= 0)), 3),
            profile_split_at_points=X["json"].get("at the points (|horizon - point depth|, mm)", {}).get("profile, top"),
        )
        if name != "normal":
            R[name]["class changes from normal"] = {
                f"{CT.CLASS[a]} -> {CT.CLASS[b]}": int(np.sum(keep & (N["cls"] == a) & (X["cls"] == b)))
                for a in range(5) for b in range(5) if a != b and np.sum(keep & (N["cls"] == a) & (X["cls"] == b))}
    # ---- at the points
    H = {m: passing(X, keep) for m, X in M.items()}
    vox = N["vox"]
    rows = []
    for k in range(len(N["P"])):
        row = dict(point=k + 1)
        for name, X in M.items():
            h = H[name][k]
            if not h:
                continue
            t = np.array([a for a, _ in h])
            i = np.array([b for _, b in h])
            eq = X["eqv"][t, i].astype(float)
            top = X["eqv"][t, X["ns"][t] - 1].astype(float)
            op = X["cls"][t] == CT.OPEN
            row[name] = dict(columns=int(t.size), open=int(op.sum()), arc_mm=round(float(np.median(i * S.STEP_MM)), 3),
                             eqv_mm=round(float(np.median(eq)), 3),
                             rel_depth_open=round(float(np.median(eq[op] / top[op])), 3) if op.any() else None)
        # per root through this point, by normal or the other: its two paths' difference
        for name in ("equivolume", "front-equivolume"):
            X = M[name]
            at = {a: b for a, b in H["normal"][k]} | {a: b for a, b in H[name][k]}
            dmax, dpt = [], []
            for t, i in at.items():
                m = int(min(N["ns"][t], X["ns"][t]))
                d = np.linalg.norm((N["pos"][t, :m] - X["pos"][t, :m]).astype(float) * vox, axis=1)
                dmax.append(d.max())
                dpt.append(d[min(i, m - 1)])
            if at:
                row[f"path difference, {name}"] = dict(roots=len(at), max_mm_median=round(float(np.median(dmax)), 3),
                                                       at_point_mm_median=round(float(np.median(dpt)), 3))
        rows.append(row)
    R["points"] = rows

    def per_point(name, key):
        return pct([r[name][key] for r in rows if name in r and r[name][key] is not None])

    summ = dict(points=len(rows))
    for name in M:
        summ[name] = dict(reached=sum(name in r for r in rows), arc_mm_p50_90=per_point(name, "arc_mm"),
                          eqv_mm_p50_90=per_point(name, "eqv_mm"), rel_depth_open_p50_90=per_point(name, "rel_depth_open"))
        if name == "normal":
            continue
        both = [r for r in rows if "normal" in r and name in r]
        summ[name].update(
            vs_normal_eqv_abs_p50_90=pct([abs(r[name]["eqv_mm"] - r["normal"]["eqv_mm"]) for r in both]),
            vs_normal_rel_depth_abs_p50_90=pct([abs(r[name]["rel_depth_open"] - r["normal"]["rel_depth_open"]) for r in both
                                                if r[name]["rel_depth_open"] is not None and r["normal"]["rel_depth_open"] is not None]),
            path_max_mm_p50_90=pct([r[f"path difference, {name}"]["max_mm_median"] for r in rows if f"path difference, {name}" in r]),
            path_at_point_mm_p50_90=pct([r[f"path difference, {name}"]["at_point_mm_median"] for r in rows
                                         if f"path difference, {name}" in r]))
    R["at the points"] = summ
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--normal", required=True)
    ap.add_argument("--equivolume", required=True)
    ap.add_argument("--front", required=True)
    A = ap.parse_args()
    res = {s: site(A, s) for s in SITES}
    json.dump(res, open(f"{A.out}/equivolume_compare.json", "w"), indent=1)
    print(
        json.dumps(
            {s: {k: v for k, v in r.items() if k != "points"} for s, r in res.items()},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
