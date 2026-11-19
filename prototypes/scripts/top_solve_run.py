"""Run T5 (top_solve.py: Σ and τ with all the costs, from the white surface)
on a site.

usage: top_solve_run.py SITE_NPZ OUT_DIR [snapshots=true] [start=NPZ] [max_steps=N] ...
start=NPZ: the initial Σ (V mm, F; e.g. from sigma_mesh.py), else the white mesh.
Writes OUT_DIR/result.npz (Σ: V, F, travel, held, contact, facing_mm; mm;
tau: the final τ on the grid), OUT_DIR/log.json (settings, the start, one
record per step, why the run ended), and with snapshots=true
OUT_DIR/snapshots/solve-NNN.npz after each accepted step, in the format of
top_surface_run.py (top_surface_movie.py reads both).
"""

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import global_tau_prep as TP
import numpy as np
import top_solve as TSV
import top_surface as TS


def main() -> None:
    site_p, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    s = TSV.SolveSettings()
    snaps = False
    start = None
    for kv in sys.argv[3:]:
        k, _, v = kv.partition("=")
        if k == "snapshots":
            snaps = TP.parse_bool(k, v)
        elif k == "start":
            start = Path(v)
        else:
            TP.set_setting(s, kv)
    site = TP.load_site(site_p)
    sd = out / "snapshots"
    if snaps:
        sd.mkdir(exist_ok=True)

    def snapshot(name: str, m: TS.Mesh, disp: np.ndarray, meta: dict, tau: np.ndarray) -> None:
        c = TS.contacts(m, site.vox)
        meta = {**meta, "contact_vertices": int(c["contact"].sum())}
        np.savez(
            sd / f"{name}.npz",
            V=m.V.astype(np.float32), F=m.F.astype(np.int32), disp=disp.astype(np.float32),
            state=m.state.astype(np.int8), held=m.held.astype(np.float32), contact=c["contact"],
            facing_mm=c["facing_mm"].astype(np.float32), tau=tau.astype(np.float32), meta=json.dumps(meta),
        )

    m0 = None
    if start is not None:
        z = np.load(start)
        V = np.asarray(z["V"], float)
        m0 = TS.Mesh(V.copy(), np.asarray(z["F"], np.int64), V.copy(), np.zeros(len(V), np.int64), np.zeros(len(V)))
    run = TSV.SolveRun(site, s, progress=lambda t: print(t, flush=True), snapshot=snapshot if snaps else None, start=m0)
    t0 = time.time()
    m = run.run()
    total = time.time() - t0
    np.savez(
        out / "result.npz", V=m.V, F=m.F, travel=m.travel, origin=m.origin, held=m.held,
        contact=run.contacts["contact"], facing_mm=run.contacts["facing_mm"], tau=run.final.tau,
    )
    log = {"settings": asdict(s), "start": str(start), "median_edge_mm": run.median_edge, "initial": run.initial,
           "steps": run.log, "ended": run.ended, "contact_vertices": int(run.contacts["contact"].sum()), "s": total}
    (out / "log.json").write_text(json.dumps(log, indent=1, default=float))
    print(f"done {total:.0f} s, {len(run.log)} steps, ended: {run.ended}", flush=True)


if __name__ == "__main__":
    main()
