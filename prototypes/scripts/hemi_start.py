"""The start of Σ for a whole hemisphere: the white surface decimated by
baillarger.decimate with Rules, V in the surface's own (world) mm. Writes
OUT.npz (V, F, vertex_ids) and OUT.json (the measures and the decimation's
log). About 5 minutes for a hemisphere (2026-11-23).

usage: hemi_start.py WHITE.surf.gii OUT.npz [key=value ...]
  eps_out=0.08 eps_in=0.16 (mm)  theta=30  shape=1.4142  size=0.8 (mm; none)  block=10  passes=4
  workers=8  repair=true  census=false (true: why the decimation stopped, D.census)
"""
import sys, json, time
import numpy as np
import nibabel as nib
from baillarger import decimate as D


def main() -> None:
    o = dict(eps_out="0.08", eps_in="0.16", theta="30", shape="1.4142135623730951", size="0.8", block="10", passes="4", workers="8", repair="true", census="false")
    o.update(kv.split("=") for kv in sys.argv[3:])
    t0 = time.time()
    g = nib.load(sys.argv[1]); v = np.asarray(g.darrays[0].data, np.float64); f = np.asarray(g.darrays[1].data, np.int64)
    print(f"read {len(v):,} vertices, {len(f):,} faces in {time.time() - t0:.0f} s", flush=True)
    rules = D.Rules(eps_in=float(o["eps_in"]), normal_cos=float(np.cos(np.radians(float(o["theta"])))) if float(o["theta"]) < 90 else 0.0,
                    size=None if o["size"] == "none" else np.full(len(v), float(o["size"])), shape=float(o["shape"]))
    t1 = time.time()
    d = D.decimate(v, f, float(o["eps_out"]), float(o["block"]), passes=int(o["passes"]), workers=int(o["workers"]),
                   log=lambda s: print(f"[{time.time() - t1:7.1f} s] {s}", flush=True), repair=o["repair"] == "true", rules=rules, with_census=o["census"] == "true")
    t_dec = time.time() - t1
    V, F = v[d.vertex_ids], d.faces
    np.savez(sys.argv[2], V=V, F=F, vertex_ids=d.vertex_ids)
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e = np.stack([np.linalg.norm(B - A, axis=1), np.linalg.norm(C - B, axis=1), np.linalg.norm(A - C, axis=1)], 1)
    nn = np.linalg.norm(np.cross(B - A, C - A), axis=1)
    ratio = e.prod(1) / np.maximum(2 * nn * e.min(1), 1e-300)
    info = {"options": o, "white_vertices": len(v), "white_faces": len(f), "vertices": len(V), "faces": len(F), "decimate_s": round(t_dec, 1),
            "passes": d.passes, "repairs": d.repairs, "pinned": int(len(d.pinned)), "census": d.census, "max_distance_mm": d.max_distance,
            "edge_mm_p5_50_95_max": np.percentile(e, [5, 50, 95, 100]).round(3).tolist(),
            "min_angle_deg_p1_5_50": np.percentile(np.degrees(np.arcsin(np.clip(1 / (2 * ratio), 0, 1))), [1, 5, 50]).round(1).tolist(),
            "faces_radius_edge_over_sqrt2": int((ratio > np.sqrt(2)).sum()), "area_mm2": float(0.5 * nn.sum())}
    open(sys.argv[2].replace(".npz", ".json"), "w").write(json.dumps(info, indent=1, default=float))
    print(json.dumps({k: info[k] for k in info if k not in ("passes", "options")}, default=float))


if __name__ == "__main__":
    main()
