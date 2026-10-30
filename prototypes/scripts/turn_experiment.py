"""Do the primary column traces still need a fallback? An experiment in four variants of the
fitted_blend primary (plan: reports/Crossing paths and contacts from laminar profiles.md,
*Column traces at folds*):

- A: the fitted field T as now, the turn rule (60 deg within 0.3 mm), the current tracer;
- B: T extended one voxel into the fluid and 0.5 mm into the WM (streamlines.fitted_field with
  fluid and wm_band_mm), with the turn rule;
- C: T as now, no turn rule;
- D: T extended, no turn rule.
- E: as D, but a trace that meets a crest of T it cannot cross ends at T's maximum along it.
  Without the turn rule such a trace reflects at the crest and goes back down T into WM; it is
  told by its heading: it enters the WM against its root's outward normal (d_end . n_root < 0), so
  it came back instead of going on. It is cut at the sample of T's largest value along it (the
  crest; on T's flat plateau between two equal voxels, its first sample), ends there as CONTACT,
  and its class is to the contact.

Primary traces only, no fallback: a trace that turns or reaches the length cap is a failure.
Classes as baillarger.column_traces (open, crossing, own bank with the 5 mm rule on the mesh
patches, failed). Everything is built fresh on the label given (the corrected segmentation's
WM with --dseg).

Sites (--site NAME --box I J K --points FILE): per variant the primary outcomes and classes of
the roots >= 2 mm from a box face; where A's turned roots go in each variant; the marked points
that crossing traces pass within 0.3 mm of; the paths' agreement with A where A did not turn.
Phantoms (--phantoms, the fold phantoms at the site's features, measured noise): per variant the
share of roots of the right class (open or crossing), and the path, end and contact errors
(fold_eval.errors).

Writes <work>/turn_experiment_<name>.json and, for variant E at the sites,
<work>/turn_experiment_<name>_E_paths.npz (the paths, ragged as turn_diag.py).
Usage: turn_experiment.py <work dir> --name NAME (--box I J K [--points FILE] | --phantoms)
                          [--dseg FILE] [--noise 1.0]
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

VARIANTS = {"A": (False, 60.0), "B": (True, 60.0), "C": (False, None), "D": (True, None), "E": (True, None)}
CLS = list(CT.CLASS)
T0 = time.time()


def say(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def fields(det, ctx, wm, labels, vox):
    m = ctx.astype(float)
    base = S.structure_tensor(det[1], m) + S.structure_tensor(det[2], m)
    T = S.fitted_field(base, ctx, wm, vox)[0]
    Te = S.fitted_field(base, ctx, wm, vox, fluid=labels == 1, wm_band_mm=0.5)[0]
    return {False: S.gradient_tensor(T, vox), True: S.gradient_tensor(Te, vox), "T": Te}


def cut_bounces(pos, ns, oc, T, nrm0, vox):
    """Variant E: traces that entered the WM heading against their root's normal end at T's
    maximum along them (CONTACT). Returns the number cut."""
    k = np.arange(len(ns))
    d_end = (pos[k, ns - 1] - pos[k, np.maximum(ns - 2, 0)]).astype(float) * vox
    back = (oc == S.WM) & ((d_end * nrm0 * vox).sum(1) < 0)
    idx = np.flatnonzero(back)
    if idx.size:
        Ta = S.sample(T, pos[idx])
        j = np.nanargmax(np.where(np.isfinite(Ta), Ta, -np.inf), 1)
        for i, m in zip(idx, j):
            pos[i, m + 1:] = np.nan
            ns[i] = m + 1
        oc[idx] = S.CONTACT
    return idx.size


def run_variants(R, labels, dface, v, f, nrm, tv, vox):
    """Per variant: paths, sample counts, outcomes and classes (primary only)."""
    P = CT.mesh_patches(v, f, vox)
    x0 = v[tv]
    out = {}
    for name, (ext, turn) in VARIANTS.items():
        pos, ns, oc = S.trace(R[ext], labels, dface, x0, nrm[tv], vox, CT.NMAX, CT.BLEND_MM,
                              turn_deg=180.0 if turn is None else turn)
        if name == "E":
            say(f"  variant E: {cut_bounces(pos, ns, oc, R['T'], nrm[tv], vox):,} traces cut at T's maximum")
        k = np.arange(len(tv))
        endp = pos[k, ns - 1].astype(float)
        last = endp - pos[k, np.maximum(ns - 2, 0)]
        last = np.where((ns > 1)[:, None], last, -nrm[tv])
        end_patch = P.of_vertex[G.crossing_vertex(v, nrm, endp, last, vox)]
        cls = CT.classify(oc, P.D[P.of_vertex[tv], end_patch])
        out[name] = dict(pos=pos, ns=ns, out=oc, cls=cls)
        say(f"  variant {name}: " + ", ".join(f"{S.OUTCOME[o]} {np.sum(oc == o):,}" for o in range(10)
                                                if np.sum(oc == o)))
    return out


def counts(a, keep, names):
    return {names[i]: int(np.sum(keep & (a == i))) for i in range(len(names)) if np.sum(keep & (a == i))}


def site(A):
    lo = tuple(A.box)
    hi = tuple(a + 141 for a in lo)
    dseg = Path(A.dseg) if A.dseg else None
    t = CT.load_tissue(lo, hi, A.name, dseg=dseg)
    v, f, nrm = CT.white_mesh(t.wm)
    tv = CT.root_vertices(v, nrm)
    q0 = np.clip(np.round(v[tv]).astype(int), 0, np.array(t.wm.shape) - 1)
    keep = t.dface[tuple(q0.T)] >= CT.FACE_MM
    say(f"{A.name}: {len(tv):,} roots, {keep.sum():,} >= 2 mm from a face")
    R = fields(t.det, t.ctx, t.wm, t.labels, t.vox)
    V = run_variants(R, t.labels, t.dface, v, f, nrm, tv, t.vox)
    res = dict(site=A.name, wm="handle-free" if dseg else "aseg", roots=int(keep.sum()), variants={})
    turnedA = keep & (V["A"]["out"] == S.TURN)
    Pts = None
    if A.points:
        Pts = np.array(json.load(open(A.points))["points_full0"], float) - np.array(lo)
        Pts = Pts[np.all((Pts >= 0) & (Pts < 141), 1)]
    for name, d in V.items():
        r = dict(outcomes=counts(d["out"], keep, list(S.OUTCOME)), classes=counts(d["cls"], keep, CLS),
                 A_turned_now=counts(d["cls"], turnedA, CLS))
        k_ = np.arange(len(tv))
        d_end = (d["pos"][k_, d["ns"] - 1] - d["pos"][k_, np.maximum(d["ns"] - 2, 0)]).astype(float)
        fwd = (d_end * nrm[tv]).sum(1) > 0
        r["own bank, entering the WM heading on (not back)"] = int(np.sum(keep & (d["cls"] == CT.OWN) & fwd))
        if Pts is not None:
            ic = np.flatnonzero(keep & (d["cls"] == CT.CROSS))
            tree = cKDTree(Pts * t.vox)
            hit = set()
            for i in ic:
                dd, j = tree.query(d["pos"][i, :d["ns"][i]].astype(float) * t.vox)
                hit.update(j[dd <= 0.3].tolist())
            r["marked points crossed (of %d)" % len(Pts)] = sorted(int(j) + 1 for j in hit)
        if name != "A":
            same = keep & (V["A"]["out"] != S.TURN) & (d["out"] == V["A"]["out"])
            dev = []
            for i in np.flatnonzero(same)[:20000]:
                n_ = min(V["A"]["ns"][i], d["ns"][i])
                dev.append(np.max(np.linalg.norm((V["A"]["pos"][i, :n_] - d["pos"][i, :n_]) * t.vox, axis=1)))
            r["max distance from A's path where A did not turn (mm) p50/p90/p99"] = (
                np.percentile(dev, [50, 90, 99]).round(3).tolist() if dev else None)
        res["variants"][name] = r
    json.dump(res, open(f"{A.work}/turn_experiment_{A.name}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    d = V["E"]
    starts = np.r_[0, np.cumsum(d["ns"])].astype(np.int64)
    xyz = np.concatenate([d["pos"][i, :d["ns"][i]] for i in range(len(tv))]).astype(np.float32)
    np.savez_compressed(f"{A.work}/turn_experiment_{A.name}_E_paths.npz", lo=np.array(lo), vox=t.vox,
                        step_mm=S.STEP_MM, v=v.astype(np.float32), f=f, nrm=nrm.astype(np.float32), tv=tv,
                        keep=keep, out=d["out"], cls=d["cls"], ns=d["ns"], out_A=V["A"]["out"],
                        cls_A=V["A"]["cls"], path_start=starts, path_xyz=xyz, classes=np.array(CLS),
                        outcomes=np.array(S.OUTCOME))


def phantoms(A):
    sys.path.insert(0, str(Path(__file__).parent))
    import fold_eval as FE
    import fold_phantoms as FP

    res = dict(site_features=A.name, noise=A.noise, phantoms={})
    for pname, fold in FP.standard().items():
        B = FP.make(fold, A.name, noise=A.noise)
        labels, vox = B["labels"], np.asarray(B["vox"], float)
        v, f, nrm = CT.white_mesh(B["wm"])
        tv = CT.root_vertices(v, nrm)
        x0 = v[tv]
        keep = B["dface"][tuple(np.clip(np.round(x0).astype(int), 0, np.array(labels.shape) - 1).T)] >= 2.0
        truth = fold.column(x0)
        tc = FE.true_class(truth)
        say(f"{pname}: {keep.sum()} roots")
        R = fields(B["det"], B["ctx"], B["wm"], labels, vox)
        V = run_variants(R, labels, B["dface"], v, f, nrm, tv, vox)
        row = {}
        for name, d in V.items():
            emax, _, eend, econ = FE.errors(fold, truth, d["pos"], d["ns"], vox)
            k = keep
            row[name] = dict(
                right_class=round(float(np.mean(d["cls"][k] == tc[k])), 3),
                failed=round(float(np.mean(d["cls"][k] == CT.FAILED)), 3),
                to_contact=round(float(np.mean(d["cls"][k] == CT.TOCONTACT)), 3),
                path_err_p50_p90=np.nanpercentile(emax[k], [50, 90]).round(3).tolist(),
                end_err_p50_p90=np.nanpercentile(eend[k], [50, 90]).round(3).tolist(),
                contact_err_p50_p90=(np.nanpercentile(econ[k], [50, 90]).round(3).tolist()
                                     if np.isfinite(econ[k]).any() else None))
        res["phantoms"][pname] = row
        say("  " + "; ".join(f"{n} right {r['right_class']}, failed {r['failed']}, path {r['path_err_p50_p90']}"
                             for n, r in row.items()))
    json.dump(res, open(f"{A.work}/turn_experiment_phantoms_{A.name}.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--name", required=True)
    ap.add_argument("--box", type=int, nargs=3)
    ap.add_argument("--points")
    ap.add_argument("--dseg", help="the corrected segmentation (python -m baillarger.white dseg)")
    ap.add_argument("--phantoms", action="store_true")
    ap.add_argument("--noise", type=float, default=1.0)
    A = ap.parse_args()
    if A.phantoms:
        phantoms(A)
    else:
        site(A)


if __name__ == "__main__":
    main()
