"""The hemisphere top confidence as viewer overlay maps (.func.gii, one value
per true white-surface vertex: the value of the nearest coarse column in 3D,
NaN beyond 1 mm). args: CONF_NPZ SURF OUT"""
import hashlib
import sys
from pathlib import Path

import numpy as np
from nibabel.gifti.gifti import GiftiImage
from scipy.spatial import cKDTree

from baillarger.gifti import DataArray, write_external

CONF, SURF, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
MAX_MM = 1.0
r = np.load(CONF)
x = r["root"].astype(np.float64)
m = len(x)
has = r["top_k"] >= 0
conf = np.where(has, r["conf"], np.nan)
src = np.where(~has, 0, np.where(r["is_lab"], 1, 2)).astype(np.float32)
queue = 100.0 * r["rank"] / m
img = GiftiImage.from_filename(str(SURF), mmap=True)
pts = np.asarray(img.darrays[0].data, np.float64)
d, near = cKDTree(x).query(pts, k=1, workers=-1)
far = d > MAX_MM
print(f"{len(pts)} true vertices, {m} columns; {int(far.sum())} vertices over {MAX_MM} mm from any column", flush=True)


def pv(a):
    out = np.asarray(a, np.float32)[near]
    out[far] = np.nan
    return out


maps = [
    ("top_conf", conf, "confidence of the column's top: sum of log terms (provisional scales); NaN: no top"),
    ("top_queue_pct", queue, "position in the review queue, % (0 = review first; no-top columns come first)"),
    ("top_mm", r["top_mm"], "the chosen top, mm along the ray from the white surface; NaN: no top"),
    ("top_source", src, "0 no top, 1 label top (outside medium), 2 dark sheet (contrast)"),
]
for j, n in enumerate(r["term_names"]):
    maps.append((f"term_{n}", r["terms"][:, j], f"confidence term {n} (NaN where it does not apply)"))
arrays = [DataArray(pv(a), "NIFTI_INTENT_NONE", {"Name": n, "Description": desc}) for n, a, desc in maps]
sha = hashlib.sha256(SURF.read_bytes()).hexdigest()
OUT.parent.mkdir(parents=True, exist_ok=True)
write_external(OUT, arrays, {"Surface": SURF.name, "SurfaceSHA256": sha,
                             "Producer": "prototypes/scripts/top_overlay.py: top confidence prototype (bent-columns trial)",
                             "Assignment": f"nearest coarse column in 3D, NaN beyond {MAX_MM} mm"})
print("wrote", OUT, flush=True)
