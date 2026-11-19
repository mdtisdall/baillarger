"""Run T5 (top_solve.py: Σ and τ with all the costs, from the white surface)
on a site.

usage: top_solve_run.py SITE_NPZ OUT_DIR [snapshots=true] [max_steps=N] ...
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
    for kv in sys.argv[3:]:
        k, v = kv.split("=", 1)
        if k == "snapshots":
            snaps = v.lower() == "true"
        else:
            setattr(s, k, type(getattr(s, k))(v))
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

    run = TSV.SolveRun(site, s, progress=lambda t: print(t, flush=True), snapshot=snapshot if snaps else None)
    t0 = time.time()
    m = run.run()
    total = time.time() - t0
    np.savez(
        out / "result.npz", V=m.V, F=m.F, travel=m.travel, origin=m.origin, held=m.held,
        contact=run.contacts["contact"], facing_mm=run.contacts["facing_mm"], tau=run.final.tau,
    )
    log = {"settings": asdict(s), "median_edge_mm": run.median_edge, "initial": run.initial,
           "steps": run.log, "ended": run.ended, "contact_vertices": int(run.contacts["contact"].sum()), "s": total}
    (out / "log.json").write_text(json.dumps(log, indent=1, default=float))
    print(f"done {total:.0f} s, {len(run.log)} steps, ended: {run.ended}", flush=True)


if __name__ == "__main__":
    main()
