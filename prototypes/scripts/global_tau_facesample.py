"""The face test (G9) once, on a sample of face units of a finished run.

usage: global_tau_facesample.py SITE_NPZ RUN_DIR OUT_JSON [N_UNITS] [HALF_MM]
Loads the final τ and owners of RUN_DIR/result.npz, takes the N_UNITS face
units (default 40) whose centroids are nearest the site centre within the
central cube of half-size HALF_MM (default 2 mm), and runs the face test on
them (7 local LM solves each, offsets -3..+3). The lagged direction ĝ₀ of
the pull term is that of the final τ. Writes the unit records (energies per
offset and side, LM iterations, whether the LM limit was hit, the move, the
signed disagreement, the time) to OUT_JSON.
"""

import json
import sys
import time
from pathlib import Path

import global_tau_energy as GE
import global_tau_prep as TP
import global_tau_solver as GS
import jax.numpy as jnp
import numpy as np


def main() -> None:
    site_p, run, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    n_units = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    half = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
    t0 = time.time()
    site = TP.load_site(site_p)
    z = np.load(site_p)
    cv = TP.mm_to_vox(site.affine, z["site_mm"][None].astype(float))[0]
    pt = TP.make_patches(site)
    gt = GS.GlobalTau(site, pt, GS.Settings(), cv)
    gt.progress = lambda m: print(m, flush=True)
    r = np.load(run / "result.npz")
    G = gt.G
    X = tuple(G.coords.T - GS.PAD)
    gt.owner = r["owner"][X].astype(np.int64)
    gt.tau = r["tau"][X].astype(float)
    faces = gt.contact_faces(gt.owner)
    units, _ = gt.face_units(gt.owner, faces)
    cc = (G.coords - GS.PAD) * gt.h
    cmm = cv * gt.h
    cen = np.array([np.concatenate([cc[u["a"]], cc[u["b"]]]).mean(0) for u in units])
    dinf = np.max(np.abs(cen - cmm), 1)
    inside = np.flatnonzero(dinf <= half)
    sel = inside[np.argsort(np.linalg.norm(cen[inside] - cmm, axis=1))][:n_units]
    print(
        f"units {len(units)}, in the central cube {len(inside)}, sampled {len(sel)}",
        flush=True,
    )
    pb = gt.global_problem(gt.owner)
    x = np.zeros(pb.S1)
    x[: pb.S] = gt.tau
    g0 = GE.j_g0(jnp.asarray(x), pb.P)
    _, recs = gt.face_test(gt.owner, g0, "sample", select=set(int(i) for i in sel))
    res = {
        "units_total": len(units),
        "units_in_cube": int(len(inside)),
        "half_mm": half,
        "sampled": [int(i) for i in sel],
        "records": recs,
        "seconds": time.time() - t0,
        "lm_iters_limit": gt.s.lm_iters,
    }
    out.write_text(json.dumps(res, default=float))
    print(f"done {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
