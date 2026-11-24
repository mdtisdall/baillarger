"""Σ of several runs (stop_run.py) at one step: the mesh violations over the
run, roughness (dihedral turns), held and contact vertices, and the distance
of each Σ to the first run's Σ (in h = 0.16 mm, the voxel size of the test
boxes).

usage (from prototypes/scripts): stop_runs_compare.py STEP RUN_A RUN_B ...
"""

import sys, json, numpy as np
from baillarger import mesh_checks as MC, surface_remesh as SR

h = 0.16
k = int(sys.argv[1])
runs = sys.argv[2:]
ref = None
for run in runs:
    L = json.load(open(run + "/log.json"))
    st = L["steps"]
    kk = min(k, len(st))
    z = np.load(f"{run}/snapshots/solve-{kk:03d}.npz", allow_pickle=True)
    V = np.asarray(z["V"], float)
    F = np.asarray(z["F"], np.int64)
    di, _ = MC.edge_dihedrals(V, F)
    di = di[np.isfinite(di)]
    viol = {
        n: int(sum(s[n] for s in st))
        for n in (
            "intersecting_pairs",
            "shared_vertex_fold_pairs",
            "creases",
            "gap_pairs",
        )
    }
    held = np.asarray(z["held"]) > 0
    con = np.asarray(z["contact"], bool)
    area = (
        0.5
        * np.linalg.norm(
            np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1
        ).sum()
    )
    line = (
        f"{run.split('/')[-1]:34s} step {kk}: E_stop {st[kk - 1]['E_stop']:9.0f} | Σ vertices {len(V)}, area {area:6.1f} mm² | violations over the run {viol} | "
        f"turn p50/90/99/max {np.percentile(di, [50, 90, 99, 100]).round(0)}°, > 30°: {100 * (di > 30).mean():.1f} %, > 45°: {100 * (di > 45).mean():.1f} % | "
        f"contact {int(con.sum())}, held without contact {int((held & ~con).sum())} | left-out stacks {st[kk - 1]['eval'].get('dropped_faces')} | run {L.get('s', 0):.0f} s"
    )
    if ref is None:
        ref = (V, F)
    else:
        _, d1, _ = SR.closest_on(ref[0], ref[1], V, 24)
        _, d2, _ = SR.closest_on(V, F, ref[0], 24)
        d = np.concatenate([d1, d2]) / h
        line += f" | distance to the first run's Σ p50/90/99/max {np.percentile(d, [50, 90, 99, 100]).round(2)} h, over 1 h: {100 * (d > 1).mean():.1f} %, over 2 h: {100 * (d > 2).mean():.1f} %"
    print(line)
