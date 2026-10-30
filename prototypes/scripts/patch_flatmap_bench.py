"""Time and distortion of patch_flat_map on a GIfTI white surface (ExternalFileBinary).
usage: patch_flatmap_bench.py <surf.gii> <out.json> [n_seeds]
Run under scripts/memguard (the true hemisphere: peak 15.4 GB)."""

import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix

from baillarger import mesh_graph as MG
from baillarger.mesh_smoothing import edges
from baillarger.viewer import flatmap as FM
from baillarger.viewer import inflation as IN

gii, out_path = Path(sys.argv[1]), Path(sys.argv[2])
n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 6
arrs = []
for da in ET.parse(gii).getroot().iter("DataArray"):
    dt = np.float32 if da.get("DataType") == "NIFTI_TYPE_FLOAT32" else np.int32
    shape = (int(da.get("Dim0")), int(da.get("Dim1")))
    a = np.fromfile(
        gii.parent / da.get("ExternalFileName"),
        dt,
        count=shape[0] * shape[1],
        offset=int(da.get("ExternalFileOffset")),
    ).reshape(shape)
    arrs.append(a)
v, f = arrs[0].astype(np.float64), arrs[1].astype(np.int64)
print(f"{len(v)} vertices, {len(f)} faces", flush=True)
# the graph build, step by step (as MG.edge_graph and IN.vertex_faces do it)
steps = {}
t = time.perf_counter()
E1 = edges(f)
steps["edges"] = time.perf_counter() - t
t = time.perf_counter()
E2 = MG._across(f)
steps["across"] = time.perf_counter() - t
t = time.perf_counter()
E = np.concatenate([E1, E2])
w = np.linalg.norm(v[E[:, 0]] - v[E[:, 1]], axis=1)
steps["weights"] = time.perf_counter() - t
t = time.perf_counter()
G0 = coo_matrix((w, (E[:, 0], E[:, 1])), shape=(len(v),) * 2).tocsr()
steps["coo_to_csr"] = time.perf_counter() - t
t = time.perf_counter()
G = G0.maximum(G0.T).tocsr()
steps["symmetrise"] = time.perf_counter() - t
del E1, E2, E, w, G0
t = time.perf_counter()
vf = IN.vertex_faces(f, len(v))
steps["vertex_faces"] = time.perf_counter() - t
t_graph = sum(steps.values())
print({k: round(x, 2) for k, x in steps.items()}, f"nnz {G.nnz}", flush=True)
seeds = np.random.default_rng(0).choice(len(v), n_seeds, replace=False)
rows = []
for R in (3.0, 5.0, 8.0, 10.0):
    for v0 in seeds.tolist():
        t = time.perf_counter()
        fp, orig, _ = FM.geodesic_patch(G, vf, f, v0, R)
        t_ext = time.perf_counter() - t
        for method in ("lscm", "exp"):
            t = time.perf_counter()
            out = FM.patch_flat_map(v, f, G, vf, v0, R, method)
            t_all = time.perf_counter() - t
            dd = FM.distortion(out)
            rows.append(
                dict(
                    R=R,
                    v0=v0,
                    method=method,
                    V=int(len(out["orig"])),
                    F=int(len(out["f_cut"])),
                    copies=int(len(out["orig"]) - np.unique(out["orig"]).size),
                    loops=out["boundary_loops"],
                    area_mm2=round(float(out["area3"].sum()), 1),
                    t_extract=round(t_ext, 3),
                    t_total=round(t_all, 3),
                    **dd,
                )
            )
            print(json.dumps(rows[-1]), flush=True)
out_path.write_text(json.dumps(dict(t_graph=t_graph, steps=steps, rows=rows), indent=1))
