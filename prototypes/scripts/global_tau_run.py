"""Run the global sparse τ solve on one cropped site (tube_crop.py npz).

usage: global_tau_run.py SITE_NPZ OUT_DIR [key=value ...]   (Settings fields)
Writes OUT_DIR/result.npz (τ, owners, faces, patches, per-outer snapshots;
with contacts=ridge also ridge_contact, ridge_below and ridge_beyond),
OUT_DIR/log.json (settings, outer iterations, LM/CG solves, events, face
patches) and prints one progress line for each outer iteration.
"""

import json
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import global_tau_prep as TP
import global_tau_solver as GS
import numpy as np


def main() -> None:
    site_p, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    s = GS.Settings()
    types = {f.name: type(getattr(s, f.name)) for f in fields(s)}
    for kv in sys.argv[3:]:
        k, v = kv.split("=")
        t = types[k]
        setattr(s, k, (v.lower() in ("1", "true")) if t is bool else t(v))
    t0 = time.time()
    site = TP.load_site(site_p)
    z = np.load(site_p)
    centre_vox = TP.mm_to_vox(site.affine, z["site_mm"][None].astype(float))[0]
    pt = TP.make_patches(site, s.patch_mm)
    sp = np.linalg.norm(
        site.v[pt.centre[pt.adj[:, 0]]] * site.vox
        - site.v[pt.centre[pt.adj[:, 1]]] * site.vox,
        axis=1,
    )
    print(
        f"site and patches: {len(pt.centre)} patches (spacing median {np.median(sp):.3f} mm), "
        f"{time.time() - t0:.0f} s",
        flush=True,
    )
    gt = GS.GlobalTau(site, pt, s, centre_vox)
    gt.progress = lambda m: print(m, flush=True)
    if s.snapshots:
        gt.snap_dir = out / "snapshots"
    G = gt.G
    print(
        f"domain {G.S} voxels ({int(gt.inner.sum())} inner), colours {G.ncolour}, "
        f"unreached {int((gt.owner < 0).sum())}, contact faces {len(gt.faces)}, march {gt.timing['march_s']:.1f} s",
        flush=True,
    )
    gt.run()
    grid = G.grid
    snaps = gt.snapshots
    np.savez_compressed(
        out / "result.npz",
        tau=grid(gt.tau).astype(np.float32),
        tau0=grid(gt.tau0).astype(np.float32),
        owner=grid(gt.owner, -2).astype(np.int32),
        owner_march=grid(gt.owner_march, -2).astype(np.int32),
        march_cost=grid(gt.march_cost).astype(np.float32),
        domain=gt.domain,
        inner=grid(gt.inner.astype(float), 0) > 0,
        inside=site.inside,
        faces=np.c_[G.coords[gt.faces[:, 0]] - GS.PAD, gt.faces[:, 2]]
        if len(gt.faces)
        else np.zeros((0, 4), int),
        **{
            f"ridge_{k}": np.c_[G.coords[f[:, 0]] - GS.PAD, f[:, 2]]
            if len(f)
            else np.zeros((0, 4), int)
            for k, f in gt.ridges.items()
        },
        snap_tau=np.stack([grid(q["tau"]).astype(np.float32) for q in snaps]),
        snap_owner=np.stack([grid(q["owner"], -2).astype(np.int32) for q in snaps]),
        vpatch=pt.vpatch,
        centre=pt.centre,
        area=pt.area,
        adj=pt.adj,
        same_ptr=pt.same_ptr,
        same_idx=pt.same_idx,
        logmed=gt.logmed,
        in_region=gt.in_region,
        beta=gt.beta,
        rslot_ok=gt.hasroot,
        vertex_vox=site.v,
        centre_vox=centre_vox,
        coh=site.coh,
        nhat=site.nhat,
        thickness=thickness(gt),
    )
    log = {
        "settings": asdict(s),
        "w_eq_provisional": True,
        "beta": gt.beta,
        "patches": len(pt.centre),
        "domain_voxels": G.S,
        "inner_voxels": int(gt.inner.sum()),
        "colours": G.ncolour,
        "unreached": int((gt.owner_march < 0).sum()),
        "timing": gt.timing,
        "outer": gt.log.outer,
        "solves": gt.log.solves,
        "events": gt.log.events,
        "face_patches": gt.face_records,
        "seconds": time.time() - t0,
    }
    (out / "log.json").write_text(json.dumps(log, default=float))
    print(f"done {time.time() - t0:.0f} s", flush=True)


def thickness(gt: GS.GlobalTau) -> np.ndarray:
    tv = gt.tau[np.maximum(gt.rslot, 0)]
    th = -np.sum(gt.rw * tv, 1)
    return np.where(gt.hasroot, th, np.nan)


if __name__ == "__main__":
    main()
