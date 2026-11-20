"""Measures of a Σ start run (top_surface_run.py), for a person to read.

usage: top_surface_measures.py SITE_NPZ RUN_DIR [CONTACT_FACES_NPZ|-] [FIG_DIR]
Prints JSON: the vertices and area by state; the distance moved (travel)
by state; the minimum distance between parts of Σ; all the checks of
baillarger.mesh_checks (intersections, folds at shared vertices, creases,
faces turned from the white surface they grew from, the gap); the time per
step; with CONTACT_FACES_NPZ (a global τ result.npz with
'faces', rows x, y, z, axis: the face between voxel (x, y, z) and its +axis
neighbour), the distance from each contact vertex to the nearest contact
face, and from each contact face to the nearest contact vertex (mm). The
contacts are top_surface.contacts of the final Σ: their number, how far
apart the facing parts are (facing_mm) and how long the push held them
(held, steps in a row refused). With
FIG_DIR, writes three slices of Σ coloured by state, and a histogram of the
travel by state. Always writes RUN_DIR/isolated_top.tsv, the review list of
the TOP vertices, most isolated first (top_surface.isolated_top): vertex,
position (mm), the stopping voxel, lag_mm, comp_voxels, comp_at_crop_edge,
label0_3x3x3.
"""

import json
import sys
from pathlib import Path

import global_tau_prep as TP
import numpy as np
import top_surface as TS
from scipy.spatial import cKDTree

from baillarger import mesh_checks as MC


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
    if "stopvox" in r:
        m.stopvox = r["stopvox"]
    _, area = TP.vertex_normals_areas(V, F)
    out: dict = {"vertices": len(V), "faces": len(F), "steps": len(log["steps"]), "s": log["s"]}
    out["by_state"] = {
        nm: {"vertices": int((st == i).sum()), "area_mm2": float(area[st == i].sum())}
        for i, nm in enumerate(TS.STATE_NAMES)
    }
    out["travel_mm"] = {nm: q(tr[st == i]) for i, nm in enumerate(TS.STATE_NAMES)}
    out["ended"] = log.get("ended")
    ct = TS.contacts(m, vox)
    cv = ct["contact"]
    held = r["held"] if "held" in r else np.zeros(len(V))
    out["contacts"] = {
        "vertices": int(cv.sum()),
        "area_mm2": float(area[cv].sum()),
        "facing_mm": q(ct["facing_mm"][cv]),
        "held_steps": q(held[cv].astype(float)),
        "facing_not_contact": int((np.isfinite(ct["facing_mm"]) & ~cv).sum()),
    }
    gap = log["settings"]["gap_vox"] * h
    _, _, dmin = MC.gap_check(m.top, V, 4 * gap, want_dmin=True)
    out["min_distance_other_part_mm"] = float(dmin[np.isfinite(dmin)].min()) if np.isfinite(dmin).any() else None
    out["vertices_within_2gap_of_other_part"] = int((dmin < 2 * gap).sum())
    out["checks"] = TS.counts(m, gap, log["settings"].get("fold_cos", MC.FOLD_COS))
    out["step_s"] = q(np.array([s["s"] for s in log["steps"]]))
    out["initial"] = log["initial"]
    iso = TS.isolated_top(m, z["labels"])
    with open(run / "isolated_top.tsv", "w") as fh:
        fh.write("vertex\tx_mm\ty_mm\tz_mm\ti\tj\tk\tlag_mm\tcomp_voxels\tcomp_at_crop_edge\tlabel0_3x3x3\n")
        for r_ in range(len(iso["vertex"])):
            v = iso["vertex"][r_]
            i, j, k = iso["stopvox"][r_]
            fh.write(
                f"{v}\t{V[v, 0]:.3f}\t{V[v, 1]:.3f}\t{V[v, 2]:.3f}\t{i}\t{j}\t{k}\t{iso['lag_mm'][r_]:.3f}\t"
                f"{iso['comp_voxels'][r_]}\t{int(iso['comp_at_crop_edge'][r_])}\t{iso['label0_3x3x3'][r_]}\n"
            )
    out["isolated_top"] = {
        "top_vertices": int(len(iso["vertex"])),
        "lag_mm": q(iso["lag_mm"]),
        "comp_voxels": q(iso["comp_voxels"].astype(float)),
        "lag_over_one_step": int((iso["lag_mm"] > log["settings"]["step_vox"] * h).sum()),
    }
    if cf is not None:
        fc = np.load(cf)["faces"]
        c_vox = fc[:, :3].astype(float)
        c_vox[np.arange(len(fc)), fc[:, 3]] += 0.5
        c_mm = c_vox * vox
        bl = V[cv]
        if len(bl):
            d1, _ = cKDTree(c_mm).query(bl)
            d2, _ = cKDTree(bl).query(c_mm)
            out["contact_vertex_to_contact_face_mm"] = q(d1)
            out["contact_face_to_contact_vertex_mm"] = q(d2)
            out["contact_faces_within_1vox_of_contact_vertex"] = float((d2 <= h).mean())
            out["contact_vertices_within_1vox_of_contact_face"] = float((d1 <= h).mean())
    print(json.dumps(out, indent=1, default=float))
    if figd is not None:
        figures(z, V, F, st, cv, tr, vox, figd, h)


def figures(z, V, F, st, cv, tr, vox, figd: Path, h: float) -> None:
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
    CONTACT = len(TS.STATE_NAMES)  # display code for a face with a contact vertex
    cols = {TS.TOP: "red", CONTACT: "cyan", TS.WM: "orange", TS.EDGE: "yellow", TS.MOVING: "magenta"}
    # a face takes the largest state code of its vertices; a contact vertex wins
    fstate = np.max(np.where(cv, CONTACT, st)[F], axis=1)
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
    fig.suptitle("Σ after the start: red = stopped at label 0, cyan = contact, "
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
