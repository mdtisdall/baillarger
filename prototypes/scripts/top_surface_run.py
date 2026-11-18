"""Run the Σ start (top_surface.py, step 1 of docs/plans/top-surface.md) on a
site.

usage: top_surface_run.py SITE_NPZ OUT_DIR [snapshots=true] [max_steps=N]
Writes OUT_DIR/result.npz (Σ: V, F, state, travel, origin; mm),
OUT_DIR/log.json (settings, the start's counts, one record per step), and
with snapshots=true OUT_DIR/snapshots/start-NN.npz after each step (V, F,
the displacement of each vertex in that step, the state, meta).
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
        k, v = kv.split("=", 1)
        if k == "snapshots":
            snaps = v.lower() == "true"
        else:
            setattr(s, k, type(getattr(s, k))(v))
    t0 = time.time()
    site = TP.load_site(site_p)
    print(f"site {len(site.v)} vertices, {len(site.f)} faces, {time.time() - t0:.0f} s", flush=True)
    sd = out / "snapshots"
    if snaps:
        sd.mkdir(exist_ok=True)

    def snapshot(name: str, m: TS.Mesh, disp: np.ndarray, meta: dict) -> None:
        np.savez(
            sd / f"{name}.npz",
            V=m.V.astype(np.float32),
            F=m.F.astype(np.int32),
            disp=disp.astype(np.float32),
            state=m.state.astype(np.int8),
            meta=json.dumps(meta),
        )

    run = TS.StartRun(site, s, progress=lambda t: print(t, flush=True), snapshot=snapshot if snaps else None)
    t1 = time.time()
    m = run.run()
    total = time.time() - t1
    np.savez(out / "result.npz", V=m.V, F=m.F, state=m.state, travel=m.travel, origin=m.origin)
    log = {
        "settings": asdict(s),
        "halvings": TS.HALVINGS,
        "median_edge_mm": run.median_edge,
        "initial": run.initial,
        "steps": run.log,
        "s": total,
    }
    (out / "log.json").write_text(json.dumps(log, indent=1))
    print(f"done {total:.0f} s, {len(run.log)} steps", flush=True)


if __name__ == "__main__":
    main()
