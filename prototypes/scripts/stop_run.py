"""Run Σ by the stop energy alone (stop_solve.SolveRun with
stop_field.StopField) on a crop set: tau_field_run.py basis=stop of the
prototype branch claude/tau-shell-cc08e1, with the same results.

usage (from prototypes/scripts): stop_run.py CROP OUT_DIR [snapshots=true] [start=NPZ] [key=value ...]

CROP is a crop set of baillarger.crop (its directory or its crop/1
sidecar). start=NPZ: the start mesh (arrays V in the crop's mm, voxel index
× voxel size, and F; e.g. from decimate_start.py); without it, the crop's
white mesh. The other keys go to stop_solve.SolveSettings, for example the
run of the fundus box:

    stop_run.py CROP OUT snapshots=true start=START.npz mesh=remesh edge_target=median \\
        edge_ratio=3.9793 fold_reference=none max_steps=60 step=pressure start_passes=10

Writes OUT_DIR/result.npz (V, F, travel, origin, held, contact, facing_mm),
OUT_DIR/log.json (rewritten after every step, so a stopped run keeps it)
and, with snapshots=true, OUT_DIR/snapshots/solve-NNN.npz (V, F, disp,
state, held, contact, facing_mm, meta).
"""

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import crop_site as CS
import numpy as np
import sigma_surface as SU
import stop_field as SF
import stop_solve as SV


def main() -> None:
    site_p, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    s = SV.SolveSettings()
    snaps = False
    start = None
    for kv in sys.argv[3:]:
        k, _, v = kv.partition("=")
        if k == "snapshots":
            snaps = CS.parse_bool(k, v)
        elif k == "start":
            start = Path(v)
        else:
            CS.set_setting(s, kv)
    site = CS.load_site(site_p)
    prog = lambda t: print(t, flush=True)  # noqa: E731
    field = SF.StopField(site, progress=prog)
    sd = out / "snapshots"
    if snaps:
        sd.mkdir(exist_ok=True)

    def snapshot(name: str, m: SU.Mesh, disp: np.ndarray, meta: dict) -> None:
        c = SU.contacts(m, site.vox)
        meta = {**meta, "contact_vertices": int(c["contact"].sum()),
                "title": f"stop energy: step {meta.get('step')}, E {meta.get('energy', 0):.1f}"}
        np.savez(
            sd / f"{name}.npz",
            V=m.V.astype(np.float32), F=m.F.astype(np.int32), disp=disp.astype(np.float32),
            state=m.state.astype(np.int8), held=m.held.astype(np.float32), contact=c["contact"],
            facing_mm=c["facing_mm"].astype(np.float32), meta=json.dumps(meta),
        )

    m0 = None
    if start is not None:
        z = np.load(start)
        V = np.asarray(z["V"], float)
        m0 = SU.Mesh(V.copy(), np.asarray(z["F"], np.int64), V.copy(), np.zeros(len(V), np.int64), np.zeros(len(V)))
    t0 = time.time()

    def write_log(run: SV.SolveRun, total: float | None = None) -> None:
        log = {"settings": asdict(s), "evaluator": field.setup, "start": str(start), "median_edge_mm": run.median_edge,
               "initial": run.initial, "steps": run.log, "ended": run.ended if total is not None else "running",
               "s": time.time() - t0 if total is None else total}
        if run.contacts:
            log["contact_vertices"] = int(run.contacts["contact"].sum())
        (out / "log.json").write_text(json.dumps(log, indent=1, default=float))

    run = SV.SolveRun(site, s, progress=prog, snapshot=snapshot if snaps else None, start=m0, evaluator=field,
                      on_step=write_log)
    m = run.run()
    total = time.time() - t0
    np.savez(
        out / "result.npz", V=m.V, F=m.F, travel=m.travel, origin=m.origin, held=m.held,
        contact=run.contacts["contact"], facing_mm=run.contacts["facing_mm"],
    )
    write_log(run, total)
    print(f"done {total:.0f} s, {len(run.log)} steps, ended: {run.ended}", flush=True)


if __name__ == "__main__":
    main()
