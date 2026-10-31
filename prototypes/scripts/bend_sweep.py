"""Does the crest method depend on its bend threshold? The column traces of
baillarger.column_traces (method "crest", fitted_blend) with BEND_DEG swept from 10 to 60 deg
(within BEND_MM = 0.3 mm), on a site (the corrected segmentation's WM with --dseg) or on the fold
phantoms at a site's features. Within a bank a column bends about a quarter of a degree per
0.04 mm step, so any threshold well above that and below the bends at contacts should give the
same columns; this measures how far that holds.

Per threshold: the classes of the roots >= 2 mm from a box face, the columns ended at a bend, and
the share of roots whose class or end changes from the 60 deg run; at the sites, the marked
points crossed by crossing columns; on the phantoms, the share of the right class and the path
and end errors (fold_eval.errors).

Writes <work>/bend_sweep_<name>.json (sites) or <work>/bend_sweep_phantoms_<name>.json.
Usage: bend_sweep.py <work dir> --name NAME (--box I J K [--points FILE] [--dseg FILE]
                     | --phantoms)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from baillarger import column_traces as CT
from baillarger import mesh_graph as G
from baillarger import streamlines as S

ANGLES = (10.0, 20.0, 30.0, 45.0, 60.0)
CLS = list(CT.CLASS)
T0 = time.time()


def say(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def sweep(det, ctx, wm, labels, dface, vox):
    """Per angle: pos, ns, outcomes, classes; with the mesh, roots and keep mask."""
    m = ctx.astype(float)
    base = S.structure_tensor(det[1], m) + S.structure_tensor(det[2], m)
    T, _ = S.fitted_field(
        base, ctx, wm, vox, fluid=labels == 1, wm_band_mm=CT.WM_BAND_MM
    )
    del base
    field = S.gradient_tensor(T, vox)
    v, f, nrm = CT.white_mesh(wm)
    tv = CT.root_vertices(v, nrm)
    P = CT.mesh_patches(v, f, vox)
    x0 = v[tv]
    q0 = np.clip(np.round(x0).astype(int), 0, np.array(wm.shape) - 1)
    keep = dface[tuple(q0.T)] >= CT.FACE_MM
    lag = max(1, int(round(CT.BEND_MM / S.STEP_MM)))
    runs = {}
    for a in ANGLES:
        pos, ns, oc = S.trace(
            field,
            labels,
            dface,
            x0,
            nrm[tv],
            vox,
            CT.NMAX,
            CT.BLEND_MM,
            turn_deg=a,
            turn_mm=CT.BEND_MM,
        )
        CT.end_at_bend(pos, ns, oc, lag)
        k = np.arange(len(tv))
        endp = pos[k, ns - 1].astype(float)
        last = endp - pos[k, np.maximum(ns - 2, 0)]
        last = np.where((ns > 1)[:, None], last, -nrm[tv])
        end_patch = P.of_vertex[G.crossing_vertex(v, nrm, endp, last, vox)]
        cls = CT.classify(oc, P.D[P.of_vertex[tv], end_patch])
        runs[a] = dict(pos=pos, ns=ns, out=oc, cls=cls)
        say(
            f"  {a:.0f} deg: "
            + ", ".join(f"{CLS[c]} {np.sum(keep & (cls == c)):,}" for c in range(5))
        )
    return runs, v, tv, keep


def compare(runs, keep, vox):
    ref = runs[ANGLES[-1]]
    out = {}
    for a, r in runs.items():
        k = np.arange(len(r["ns"]))
        e = r["pos"][k, r["ns"] - 1].astype(float)
        e0 = ref["pos"][k, ref["ns"] - 1].astype(float)
        moved = np.linalg.norm((e - e0) * vox, axis=1)
        out[f"{a:.0f}"] = dict(
            classes={CLS[c]: int(np.sum(keep & (r["cls"] == c))) for c in range(5)},
            ended_at_bend=int(np.sum(keep & (r["out"] == S.CONTACT))),
            class_differs_from_60=round(
                float(np.mean(r["cls"][keep] != ref["cls"][keep])), 4
            ),
            end_moved_over_0_1_mm_from_60=round(float(np.mean(moved[keep] > 0.1)), 4),
        )
    return out


def site(A):
    lo = tuple(A.box)
    hi = tuple(a + 141 for a in lo)
    dseg = Path(A.dseg) if A.dseg else None
    t = CT.load_tissue(lo, hi, A.name, dseg=dseg)
    say(f"{A.name}: WM {'handle-free' if dseg else 'aseg'}")
    runs, v, tv, keep = sweep(t.det, t.ctx, t.wm, t.labels, t.dface, t.vox)
    res = dict(
        site=A.name,
        wm="handle-free" if dseg else "aseg",
        roots=int(keep.sum()),
        by_angle=compare(runs, keep, t.vox),
    )
    if A.points:
        Pts = np.array(json.load(open(A.points))["points_full0"], float) - np.array(lo)
        Pts = Pts[np.all((Pts >= 0) & (Pts < 141), 1)]
        tree = cKDTree(Pts * t.vox)
        for a, r in runs.items():
            hit = set()
            for i in np.flatnonzero(keep & (r["cls"] == CT.CROSS)):
                d, j = tree.query(r["pos"][i, : r["ns"][i]].astype(float) * t.vox)
                hit.update(j[d <= 0.3].tolist())
            res["by_angle"][f"{a:.0f}"][f"marked points crossed (of {len(Pts)})"] = len(
                hit
            )
    json.dump(res, open(f"{A.work}/bend_sweep_{A.name}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def phantoms(A):
    sys.path.insert(0, str(Path(__file__).parent))
    import fold_eval as FE
    import fold_phantoms as FP

    res = dict(site_features=A.name, phantoms={})
    for pname, fold in FP.standard().items():
        B = FP.make(fold, A.name, noise=1.0)
        vox = np.asarray(B["vox"], float)
        say(pname)
        runs, v, tv, keep = sweep(
            B["det"], B["ctx"], B["wm"], B["labels"], B["dface"], vox
        )
        truth = fold.column(v[tv])
        tc = FE.true_class(truth)
        row = compare(runs, keep, vox)
        for a, r in runs.items():
            emax, _, eend, _ = FE.errors(fold, truth, r["pos"], r["ns"], vox)
            row[f"{a:.0f}"].update(
                right_class=round(float(np.mean(r["cls"][keep] == tc[keep])), 3),
                path_err_p50_p90=np.nanpercentile(emax[keep], [50, 90])
                .round(3)
                .tolist(),
                end_err_p50_p90=np.nanpercentile(eend[keep], [50, 90])
                .round(3)
                .tolist(),
            )
        res["phantoms"][pname] = row
    json.dump(res, open(f"{A.work}/bend_sweep_phantoms_{A.name}.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--name", required=True)
    ap.add_argument("--box", type=int, nargs=3)
    ap.add_argument("--points")
    ap.add_argument(
        "--dseg", help="the corrected segmentation (python -m baillarger.white dseg)"
    )
    ap.add_argument("--phantoms", action="store_true")
    A = ap.parse_args()
    phantoms(A) if A.phantoms else site(A)


if __name__ == "__main__":
    main()
