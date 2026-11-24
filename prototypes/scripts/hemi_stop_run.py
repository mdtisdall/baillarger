"""Σ over the whole hemisphere moved by the stop energy alone
(stop_field.StopField, stop_solve.SolveRun step=pressure) from a decimated
start (hemi_start.py), with no τ and no grid-sized array but the labels and
the inside bytes.

usage (from prototypes/scripts): hemi_stop_run.py DSEG.nii.gz START.npz OUT_DIR [max_steps=60]
  [edge_mm=0.8241] [start_passes=10] [snap_every=5]
Writes OUT_DIR/log.json (after every step), OUT_DIR/steps/ (Σ at every step in
world mm, for baillarger.viewer --sigma-dir: sigma_steps.write_step),
OUT_DIR/snapshots/solve-NNN.npz (every snap_every steps and the last) and
OUT_DIR/final.npz.
"""
import json, sys, time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import sigma_steps as SS
import sigma_surface as SU
import stop_field as SF
import stop_solve as SV


def main() -> None:
    dseg, start, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    o = dict(max_steps="60", edge_mm="0.8241", start_passes="10", snap_every="5")
    o.update(kv.split("=") for kv in sys.argv[4:])
    prefix = Path(dseg).name.split("_desc-")[0] + "_hemi-R_desc-sigma"
    out.mkdir(parents=True, exist_ok=True)
    (out / "snapshots").mkdir(exist_ok=True)
    t0 = time.time()
    img = nib.load(dseg)
    A = np.asarray(img.affine, float)
    assert np.allclose(A[:3, :3], np.diag(np.diag(A[:3, :3]))) and (np.diag(A[:3, :3]) > 0).all()
    vox = np.diag(A[:3, :3]).copy()
    labels = np.asarray(img.dataobj).astype(np.int16)
    z = np.load(start)
    V = np.asarray(z["V"], float) - A[:3, 3]  # the grid's mm: voxel index × voxel size
    F = np.asarray(z["F"], np.int64)
    e = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]]), 1), axis=0)
    med = float(np.median(np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)))
    print(f"labels {labels.shape}, voxel {vox.round(4).tolist()} mm; start {len(V):,} vertices, {len(F):,} faces, median edge {med:.3f} mm; read in {time.time() - t0:.0f} s", flush=True)
    site = SimpleNamespace(labels=labels, vox=vox)
    s = SV.SolveSettings()
    s.mesh, s.edge_target, s.edge_ratio, s.fold_reference = "remesh", "median", float(o["edge_mm"]) / med, "none"
    s.max_steps, s.step, s.start_passes = int(o["max_steps"]), "pressure", int(o["start_passes"])
    field = SF.StopField(site, progress=lambda t: print(t, flush=True))
    every = int(o["snap_every"])

    def snapshot(name: str, m: SU.Mesh, disp: np.ndarray, meta: dict) -> None:
        k = int(name.split("-")[1])
        # every step: a surface in world mm for the viewer (baillarger.viewer --sigma-dir OUT/steps)
        SS.write_step(out / "steps", prefix, k, m.V + A[:3, 3], m.F, {a_: meta[a_] for a_ in ("step", "energy", "E_stop") if a_ in meta})
        if k % every and k != s.max_steps:
            return
        np.savez(out / "snapshots" / f"{name}.npz", V=m.V.astype(np.float32), F=m.F.astype(np.int32), disp=disp.astype(np.float32),
                 state=m.state.astype(np.int8), held=m.held.astype(np.float32), travel=m.travel.astype(np.float32), meta=json.dumps(meta, default=float))

    def write_log(run: SV.SolveRun, total: float | None = None) -> None:
        log = {"settings": asdict(s), "options": o, "start": start, "median_edge_mm": run.median_edge, "initial": run.initial,
               "steps": run.log, "ended": run.ended if total is not None else "running", "s": time.time() - t0 if total is None else total}
        (out / "log.json").write_text(json.dumps(log, indent=1, default=float))

    m0 = SU.Mesh(V.copy(), F, V.copy(), np.zeros(len(V), np.int64), np.zeros(len(V)))
    run = SV.SolveRun(site, s, progress=lambda t: print(time.strftime("%H:%M:%S"), t[:400], flush=True), snapshot=snapshot, start=m0,
                       evaluator=field, on_step=write_log)
    m = run.run()
    write_log(run, time.time() - t0)
    c = run.contacts or SU.contacts(m, vox)
    np.savez(out / "final.npz", V=m.V, F=m.F, held=m.held, state=m.state, travel=m.travel, contact=c["contact"], world_offset=A[:3, 3])
    print(f"done {time.time() - t0:.0f} s, {len(run.log)} steps, ended: {run.ended}; {len(m.V):,} vertices, contact {int(c['contact'].sum())}", flush=True)


if __name__ == "__main__":
    main()
