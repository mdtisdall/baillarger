"""Measures of a Σ start run (top_surface_run.py), for a person to read.

usage: top_surface_measures.py SITE_NPZ RUN_DIR [CONTACT_FACES_NPZ] [FIG_DIR]
Prints JSON: the vertices and area by state; the distance moved (travel)
by state; the minimum distance between parts of Σ and the intersections;
the time per step; with CONTACT_FACES_NPZ (a global τ result.npz with
'faces', rows x, y, z, axis: the face between voxel (x, y, z) and its +axis
neighbour), the distance from each BLOCKED vertex to the nearest contact
face, and from each contact face to the nearest BLOCKED vertex (mm). With
FIG_DIR, writes three slices of Σ coloured by state, and a histogram of the
travel by state.
"""

import json
import sys
from pathlib import Path

import global_tau_prep as TP
import numpy as np
import top_surface as TS
from scipy.spatial import cKDTree

from baillarger import mesh_intersect as MI


def q(a: np.ndarray) -> dict:
    if len(a) == 0:
        return {"n": 0}
    p = np.percentile(a, [5, 25, 50, 75, 95])
    return {"n": int(len(a)), "p5": p[0], "p25": p[1], "median": p[2], "p75": p[3], "p95": p[4], "max": float(a.max())}


def main() -> None:
    site_p, run = Path(sys.argv[1]), Path(sys.argv[2])
    cf = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "-" else None
    figd = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    z = np.load(site_p)
    A = z["affine"].astype(float)
    vox = np.abs(np.diag(A)[:3])
    h = float(vox.mean())
    r = np.load(run / "result.npz")
    log = json.loads((run / "log.json").read_text())
    V, F, st, tr = r["V"], r["F"], r["state"], r["travel"]
    m = TS.Mesh(V, F, r["origin"], st, tr)
    _, area = TP.vertex_normals_areas(V, F)
    out: dict = {"vertices": len(V), "faces": len(F), "steps": len(log["steps"]), "s": log["s"]}
    out["by_state"] = {
        nm: {"vertices": int((st == i).sum()), "area_mm2": float(area[st == i].sum())}
        for i, nm in enumerate(TS.STATE_NAMES)
    }
    out["travel_mm"] = {nm: q(tr[st == i]) for i, nm in enumerate(TS.STATE_NAMES)}
    gap = log["settings"]["gap_vox"] * h
    _, ng, dmin = TS.gap_check(m, V, 4 * gap, want_dmin=True)
    out["gap_pairs_at_gap"] = int(TS.gap_check(m, V, gap)[1])
    out["min_distance_other_part_mm"] = float(dmin[np.isfinite(dmin)].min()) if np.isfinite(dmin).any() else None
    out["vertices_within_2gap_of_other_part"] = int((dmin < 2 * gap).sum())
    out["intersections"] = int(MI.intersecting_faces(V, F)[1])
    out["step_s"] = q(np.array([s["s"] for s in log["steps"]]))
    out["initial"] = log["initial"]
    if cf is not None:
        fc = np.load(cf)["faces"]
        c_vox = fc[:, :3].astype(float)
        c_vox[np.arange(len(fc)), fc[:, 3]] += 0.5
        c_mm = c_vox * vox
        bl = V[st == TS.BLOCKED]
        if len(bl):
            d1, _ = cKDTree(c_mm).query(bl)
            d2, _ = cKDTree(bl).query(c_mm)
            out["blocked_to_contact_face_mm"] = q(d1)
            out["contact_face_to_blocked_mm"] = q(d2)
            out["contact_faces_within_1vox_of_blocked"] = float((d2 <= h).mean())
            out["blocked_within_1vox_of_contact_face"] = float((d1 <= h).mean())
    print(json.dumps(out, indent=1, default=float))
    if figd is not None:
        figures(z, V, F, st, tr, vox, figd, h)


def figures(z, V, F, st, tr, vox, figd: Path, h: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import top_surface_movie as MV
    from matplotlib.collections import LineCollection

    figd.mkdir(parents=True, exist_ok=True)
    A = z["affine"].astype(float)
    wv = TP.mm_to_vox(A, z["white_v"].astype(float))
    wf = z["white_f"].astype(np.int64)
    cv = np.round((np.linalg.inv(A) @ np.r_[z["site_mm"], 1])[:3]).astype(int)
    e2 = z["echoes"][1].astype(float)
    Vv = V / vox
    cols = {TS.TOP: "red", TS.BLOCKED: "cyan", TS.WM: "orange", TS.EDGE: "yellow", TS.MOVING: "magenta"}
    fstate = np.max(st[F], axis=1)  # a face takes the largest state code of its vertices (BLOCKED wins)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.3))
    for ax in range(3):
        a = axs[ax]
        k = int(cv[ax])
        a.imshow(np.take(e2, k, axis=ax), cmap="gray", origin="lower", interpolation="nearest")
        a.add_collection(LineCollection(MV.cut_segments(wv, wf, ax, k), colors="deepskyblue", linewidths=0.8))
        for s_, c in cols.items():
            sel = F[fstate == s_]
            if len(sel):
                a.add_collection(LineCollection(MV.cut_segments(Vv, sel, ax, k), colors=c, linewidths=1.2))
        a.set_title(f"{'xyz'[ax]} = {k}")
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("Σ after the start: red = stopped at label 0, cyan = blocked (first contact), "
                 "orange = WM limit, yellow = crop edge; blue = the white surface")
    fig.savefig(figd / "start_slices.png", dpi=80)
    plt.close(fig)
    fig, a = plt.subplots(figsize=(7, 4))
    bins = np.arange(0, tr.max() + h / 4, h / 4)
    for i, nm in enumerate(TS.STATE_NAMES):
        if (st == i).any():
            a.hist(tr[st == i], bins=bins, histtype="step", label=f"{nm} ({(st == i).sum()})")
    a.set_xlabel("distance moved from the white surface (mm)")
    a.set_ylabel("vertices")
    a.legend()
    fig.tight_layout()
    fig.savefig(figd / "start_travel.png", dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
