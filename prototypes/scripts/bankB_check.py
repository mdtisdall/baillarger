import sys

sys.path.insert(0, sys.argv[1])
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[1]]
import numpy as np
import cheap_test as C
import flatten_fused2 as F
from baillarger.data import load_patch
from baillarger.frame import missing_tissue
from scipy import ndimage as ndi

p = load_patch("V1V2")
miss = missing_tissue(p.rim, p.flash, p.voxel_mm)
D = C.setup("fused", p, miss)
_, near = ndi.distance_transform_edt(~D["wm"], sampling=p.voxel_mm, return_indices=True)
out = []
for c, (i, j) in enumerate(D["path"]):
    x = np.array([i, j, D["kk"]], float)
    g0 = np.array([F.interp(cc, x) for cc in D["g"]])
    x = x + g0 / (np.linalg.norm(g0) + 1e-12)
    up, st = C.walk(
        D,
        x,
        +1,
        lambda vv, t, pr: (
            "fluid" if D["rb"][vv] == 1 else ("ridge" if t < pr - 1e-3 else None)
        ),
    )
    if st != "ridge":
        continue
    din = up[-1] - up[-2]
    din /= np.linalg.norm(din) + 1e-12
    down, st2 = C.walk_dir(
        D,
        up[-1] + din * 2.5,
        din,
        lambda vv, t, pr: (
            "WM" if D["wm"][vv] else ("fluid" if D["rb"][vv] == 1 else None)
        ),
    )
    top = up[-1]
    end = down[-1]
    # which WM is nearest the ridge point, and how far is it from bank A's seed
    nr = np.array([near[a][tuple(np.round(top).astype(int))] for a in range(3)])
    sA = np.linalg.norm(top - np.array([i, j, D["kk"]])) * 0.16
    sNear = np.linalg.norm(top - nr) * 0.16
    out.append(
        (
            c,
            st2,
            np.round(end).astype(int),
            round(float(np.abs(end[2] - D["kk"]) * 0.16), 2),
            round(float(np.linalg.norm(np.diff(down, axis=0), axis=1).sum() * 0.16), 2),
            round(sA, 2),
            round(sNear, 2),
        )
    )
import collections

print("outcomes:", collections.Counter(o[1] for o in out))
seedpos = {c: np.array([i, j, D["kk"]]) for c, (i, j) in enumerate(D["path"])}
backA = [
    o
    for o in out
    if o[1] == "WM" and min(np.linalg.norm(o[2] - q) for q in seedpos.values()) < 3
]
print("paths landing back on bank A's own WM:", len(backA))
for o in out:
    print(o)
ends = np.array([o[2] for o in out if o[1] == "WM"])
print("distinct B landing voxels:", len({tuple(e) for e in ends}), "of", len(ends))
print("B landing spread (vox) per axis:", ends.min(0), ends.max(0))
