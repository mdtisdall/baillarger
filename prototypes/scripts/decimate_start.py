"""A box's start mesh for Σ: the crop's white surface decimated by
baillarger.decimate with Rules (T3), and its measures (edge lengths, angles,
normals against the white surface, distance to it, the mesh checks) as one
JSON line.

usage (from prototypes/scripts): decimate_start.py CROP OUT.npz [key=value ...]
  eps_out=0.5 eps_in=1.0 (voxels)  theta=30  shape=1.4142  size=5 (voxels; "field": the curvature
  field 1..5; "none")  size_factor=1  block=10 (mm)  passes=4  workers=4  repair=true
OUT.npz holds V (the crop's mm) and F, the start= of stop_run.py.
"""
import sys, json, time, numpy as np
from scipy.spatial import cKDTree
import crop_site as CS, sigma_surface as SU
from baillarger import decimate as D, mesh_checks as MC, surface_remesh as SR


def main() -> None:
    site = CS.load_site(sys.argv[1]); out = sys.argv[2]
    o = dict(eps_out="0.5", eps_in="1.0", theta="30", shape="1.4142135623730951", size="5", size_factor="1", block="10", passes="4", workers="4", repair="true")
    o.update(kv.split("=") for kv in sys.argv[3:])
    h = float(site.vox.mean()); W = site.v * site.vox; Fw = np.asarray(site.f, np.int64); shape = np.array(site.labels.shape)
    t0 = time.time()
    size = SU.sizing(site, 1.0) if o["size"] == "field" else (None if o["size"] == "none" else np.full(len(W), float(o["size"]) * h))
    t_size = time.time() - t0
    rules = D.Rules(eps_in=float(o["eps_in"]) * h, normal_cos=float(np.cos(np.radians(float(o["theta"])))) if float(o["theta"]) < 90 else 0.0,
                    size=size, size_factor=float(o["size_factor"]), shape=float(o["shape"]))
    t0 = time.time()
    d = D.decimate(W, Fw, float(o["eps_out"]) * h, float(o["block"]), passes=int(o["passes"]), workers=int(o["workers"]), log=lambda s: None,
                   repair=o["repair"] == "true", rules=rules)
    t_dec = time.time() - t0
    V, F = W[d.vertex_ids], d.faces
    np.savez(out, V=V, F=F)
    # measures
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e = np.stack([np.linalg.norm(B - A, axis=1), np.linalg.norm(C - B, axis=1), np.linalg.norm(A - C, axis=1)], 1)
    n = np.cross(B - A, C - A); nn = np.linalg.norm(n, axis=1); n /= nn[:, None]
    ratio = e.prod(1) / (2 * nn * e.min(1)); ang_min = np.degrees(np.arcsin(np.clip(1 / (2 * ratio), 0, 1)))
    cen = (A + B + C) / 3; inner = np.minimum(cen / site.vox, shape - 1 - cen / site.vox).min(1) > 4
    nW = site.normal / np.linalg.norm(site.normal, axis=1)[:, None]
    face = np.repeat(np.arange(len(F)), np.diff(d.cptr)); p = W[d.cids]
    Sn = nW[d.vertex_ids[F]].sum(1); np.add.at(Sn, face, nW[d.cids])
    aS = np.degrees(np.arccos(np.clip((n * Sn).sum(1) / np.maximum(np.linalg.norm(Sn, axis=1), 1e-300), -1, 1)))
    dist = D.carried_distances(W, d.vertex_ids[F], d.cptr, d.cids) / h; side = ((p - V[F[face, 0]]) * n[face]).sum(1)
    chk = MC.check_mesh(V, F, gap=0.25 * h)
    rim = MC.boundary_vertices(F, len(V))
    res = {"options": o, "size_s": round(t_size, 1), "decimate_s": round(t_dec, 1), "white_vertices": len(W), "vertices": len(V), "faces": len(F),
           "rim_vertices": int(rim.sum()), "pinned": int(len(d.pinned)), "repair_rounds": len(d.repairs), "passes": [(s["faces_after"], s["collapses"], s["seconds"]) for s in d.passes],
           "inner_faces": int(inner.sum()),
           "edge_h_p5_50_95_max_inner": (np.percentile(e[inner], [5, 50, 95, 100]) / h).round(2).tolist(),
           "min_angle_deg_p1_5_50_inner": np.percentile(ang_min[inner], [1, 5, 50]).round(1).tolist(),
           "faces_radius_edge_over_sqrt2_inner": int((ratio[inner] > np.sqrt(2)).sum()),
           "normal_deg_p50_90_99_max_inner": np.percentile(aS[inner], [50, 90, 99, 100]).round(1).tolist(), "faces_over_30_45_inner": [int((aS[inner] > t).sum()) for t in (30, 45)],
           "mesh_outward_of_white_max_h": round(float(dist[side < 0].max()) if (side < 0).any() else 0.0, 3),
           "mesh_inward_of_white_h_p50_90_99_max": np.percentile(dist[side >= 0], [50, 90, 99, 100]).round(2).tolist(),
           "checks": chk.counts()}
    print(json.dumps(res, default=float))
    


if __name__ == "__main__":
    main()
