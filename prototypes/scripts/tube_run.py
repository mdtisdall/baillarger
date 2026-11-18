"""Run the tube front-growth τ solver on one cropped site.

usage: tube_run.py SITE_NPZ OUT_DIR [key=value ...]   (Settings fields)
Writes OUT_DIR/result.npz (τ, owner, patches, thickness) and OUT_DIR/log.json
(settings, steps, events). The site centre is the crop's site_mm.
"""

import json
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import tube_prep as TP
import tube_solver as TS


def main() -> None:
    site_p, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    s = TS.Settings()
    types = {f.name: f.type for f in fields(s)}
    for kv in sys.argv[3:]:
        k, v = kv.split("=")
        cur = getattr(s, k)
        setattr(s, k, type(cur)(v) if not isinstance(cur, float) else float(v))
    t0 = time.time()
    site = TP.load_site(site_p)
    z = np.load(site_p)
    centre_vox = TP.mm_to_vox(site.affine, z["site_mm"][None].astype(float))[0]
    print(f"site loaded {time.time() - t0:.0f} s", flush=True)
    pt = TP.make_patches(site, s.patch_mm)
    sp = np.linalg.norm(site.v[pt.centre[pt.adj[:, 0]]] * site.vox - site.v[pt.centre[pt.adj[:, 1]]] * site.vox, axis=1)
    print(f"patches {len(pt.centre)} (centre spacing median {np.median(sp):.3f} mm), {time.time() - t0:.0f} s", flush=True)
    solver = TS.TubeSolver(site, pt, s, centre_vox)
    print(f"solver ready: colours {solver.ncolour}, band {len(solver.band())}, beta {solver.beta:.3f}, {time.time() - t0:.0f} s", flush=True)
    (out / "snapshots").mkdir(exist_ok=True)
    solver.run(progress=lambda m: print(m, flush=True), snap_dir=out / "snapshots")
    th = solver.thickness()
    np.savez_compressed(
        out / "result.npz",
        tau=solver.grid(solver.tau).astype(np.float32), owner=solver.grid(solver.owner).astype(np.int32),
        inside=site.inside, vpatch=pt.vpatch, centre=pt.centre, area=solver.area, alias=solver.alias,
        thickness=th, logmed=solver.logmed, in_region=solver.in_region, beta=solver.beta,
        coarse=np.stack(pt.coarse), colour=solver.colour, stopped=solver.stopped, centre_vox=centre_vox,
        same_ptr=pt.same_ptr, same_idx=pt.same_idx, adj=pt.adj, H=site.H, vertex_vox=site.v,
    )
    np.savez_compressed(
        out / "sweeps_dtau.npz", step=np.array([x[0] for x in solver.sweep_dtau]),
        full=np.array([x[1] for x in solver.sweep_dtau]),
        dtau=np.stack([x[2] for x in solver.sweep_dtau]) if solver.sweep_dtau else np.zeros((0, solver.ntube)),
    )
    log = {"settings": asdict(s), "w_eq_provisional": True, "beta": solver.beta,
           "patches": len(pt.centre), "patch_spacing_median_mm": float(np.median(sp)),
           "steps": solver.log.steps, "events": solver.log.events, "seconds": time.time() - t0}
    (out / "log.json").write_text(json.dumps(log, default=float))
    print(f"done {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
