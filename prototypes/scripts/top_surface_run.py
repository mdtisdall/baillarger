"""Run the Σ start (top_surface.py, step 1 of docs/plans/top-surface.md) on a
site.

usage: top_surface_run.py SITE_NPZ OUT_DIR [snapshots=true] [max_steps=N]
Writes OUT_DIR/result.npz (Σ: V, F, state, travel, origin; mm),
OUT_DIR/log.json (settings, the start's counts, one record per step), and
with snapshots=true OUT_DIR/snapshots/start-NN.npz after each step (V, F,
the displacement of each vertex in that step, the state, held, the
contacts measured from the geometry (contact, facing_mm), with tau_step=true
the τ behind Σ that drove the step (tau), meta).
result.npz also has held, contact and facing_mm of the final Σ; log.json
says why the start ended ("no step" or "max_steps").
"""

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import global_tau_prep as TP
import numpy as np
import top_surface as TS


def main() -> None:
    site_p, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    s = TS.StartSettings()
    snaps = False
    for kv in sys.argv[3:]:
        k, _, v = kv.partition("=")
        if k == "snapshots":
            snaps = TP.parse_bool(k, v)
        else:
            TP.set_setting(s, kv)
    t0 = time.time()
    site = TP.load_site(site_p)
    print(f"site {len(site.v)} vertices, {len(site.f)} faces, {time.time() - t0:.0f} s", flush=True)
    sd = out / "snapshots"
    if snaps:
        sd.mkdir(exist_ok=True)

    def snapshot(name: str, m: TS.Mesh, disp: np.ndarray, meta: dict) -> None:
        c = TS.contacts(m, site.vox)
        meta = {**meta, "contact_vertices": int(c["contact"].sum())}
        np.savez(
            sd / f"{name}.npz",
            V=m.V.astype(np.float32),
            F=m.F.astype(np.int32),
            disp=disp.astype(np.float32),
            state=m.state.astype(np.int8),
            held=m.held.astype(np.float32),
            contact=c["contact"],
            facing_mm=c["facing_mm"].astype(np.float32),
            meta=json.dumps(meta),
            # with tau_step: the τ behind Σ that drove this step (nan off the domain)
            **({"tau": run.tau_grid.astype(np.float32)} if run.tau_grid is not None else {}),
        )

    run = TS.StartRun(site, s, progress=lambda t: print(t, flush=True), snapshot=snapshot if snaps else None)
    t1 = time.time()
    m = run.run()
    total = time.time() - t1
    np.savez(
        out / "result.npz", V=m.V, F=m.F, state=m.state, travel=m.travel, origin=m.origin, stopvox=m.stopvox,
        held=m.held, contact=run.contacts["contact"], facing_mm=run.contacts["facing_mm"],
    )
    log = {
        "settings": asdict(s),
        "halvings": TS.HALVINGS,
        "median_edge_mm": run.median_edge,
        "initial": run.initial,
        "steps": run.log,
        "ended": run.ended,
        "contact_vertices": int(run.contacts["contact"].sum()),
        "s": total,
    }
    (out / "log.json").write_text(json.dumps(log, indent=1))
    print(f"done {total:.0f} s, {len(run.log)} steps", flush=True)


if __name__ == "__main__":
    main()
