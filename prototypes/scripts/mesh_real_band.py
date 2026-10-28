"""Finding 28, real patches: the start surface as boundary data (all 3D).

1. The march from each start (raw WM-boundary voxels at arrival 0, one label; the narrow
   band of `white_mesh` sigma 1; of the mesh smoothed along its edges, twicing 0.32 mm),
   isotropic and ratio 3 along the data's layer normal (as fronts_eval.py), carrying the
   root position (mesh_smoothing.march_band). Counted: cortex voxels within 1 mm of the
   WM (and >= 2 mm from the faces) with a 6-neighbour whose root is > 1 mm away
   (Euclidean: a lower bound on the distance along the white surface), the same over all
   cortex, and the root map's jump per voxel step near the WM (p95, p99.9).
2. Thin WM (as mesh_real.py): at points around it on the 1x grid and on a 2x grid (cortex
   voxels within 2 voxels of thin WM, and their 8 sub-points at +-0.25 voxel), the
   fraction whose signed distance to each start surface comes out negative (inside), and
   whose foot point is more than 2 voxels away: a foot on the far side of the blade or a
   lost blade.

Usage: mesh_real_band.py <out json> <patch>
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from scipy import ndimage as ndi

import curv_lib as C
import mesh_smoothing as M
import onepass_lib as O
from baillarger.data import PATCHES
from baillarger.fronts import riemann_from_direction
from mesh_phantoms import surfaces

OUT, NAME = sys.argv[1:3]
T0 = time.time()


def ball(r):
    g = np.mgrid[-r : r + 1, -r : r + 1, -r : r + 1]
    return (g**2).sum(0) <= r * r + 1e-9


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def main():
    lo, hi = (np.array(v) for v in PATCHES[NAME])
    B = C.load(lo, hi, f"mesh_real_band_{NAME}")
    vox = B["vox"]
    h = float(np.mean(vox))
    wm, ctx, dface = B["wm"], B["ctx"], B["dface"]
    res = dict(patch=NAME)
    nh, coh = C.tensor_normal(B)
    d_wm = ndi.distance_transform_edt(~wm, sampling=vox)
    g = np.stack(np.gradient(ndi.gaussian_filter(d_wm, 1.0), *vox), -1)
    gh = g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-12)
    sgn = np.sign((nh * gh).sum(-1))
    sgn[sgn == 0] = 1
    dirv = coh[..., None] * nh * sgn[..., None] + 0.1 * gh
    dirv /= np.linalg.norm(dirv, axis=-1, keepdims=True) + 1e-12
    del nh, g, gh, coh
    log("orientation")
    S = surfaces(wm, sigmas=(1.0, 1.5), scales=(2.0,), taubin=())
    X = np.stack(np.meshgrid(*[np.arange(n, dtype=float) for n in wm.shape], indexing="ij"), -1)
    near = ctx & (d_wm <= 1.0) & (dface >= 2.0)
    allc = ctx & (dface >= 2.0)
    for ratio in (1.0, 3.0):
        Mt = riemann_from_direction(dirv, ratio)
        D = None if ratio == 1.0 else np.linalg.inv(Mt)
        for nm in ("raw", "white_mesh s1", "twice 0.32mm"):
            t = time.time()
            if nm == "raw":
                seeds = wm & ndi.binary_dilation(ctx, ndi.generate_binary_structure(3, 1))
                k = np.flatnonzero(seeds.ravel())
                out = M.march_band(Mt, ctx | seeds, seeds, np.zeros(k.size), np.zeros(k.size),
                                   X.reshape(-1, 3)[k], spacing=vox)
            else:
                v, f, _ = S[nm]
                sd, ts, ss, rs = M.surface_seeds(ctx, wm, X, M.mesh_closest(v, f), D, band=2.5)
                out = M.march_band(Mt, ctx, sd, ts * h, ss * h, rs, spacing=vox)
            reached = np.isfinite(out.arrival)
            r = {}
            for tag, m0 in (("near", near), ("all", allc)):
                m = m0 & reached
                hit = np.zeros(m.size, bool)
                js = []
                Rm = out.root.reshape(-1, 3)
                for off in O.N6:
                    pa, pb = O.shifted_pairs(m, m, off)
                    jj = np.linalg.norm((Rm[pa] - Rm[pb]) * vox, axis=1)
                    js.append(jj / h)
                    far = jj > 1.0
                    hit[pa[far]] = True
                    hit[pb[far]] = True
                J = np.concatenate(js)
                r[f"coll_{tag}"] = int(hit.sum())
                r[f"coll_{tag}_frac"] = float(hit.sum() / max(m.sum(), 1))
                r[f"rootjump_{tag}_p95"] = float(np.percentile(J, 95))
                r[f"rootjump_{tag}_p999"] = float(np.percentile(J, 99.9))
            res.setdefault("march", {})[f"{nm} | ratio {ratio:g}"] = r
            log(f"{nm:14s} ratio {ratio:g}: >1 mm root jumps near WM {r['coll_near']:,} ({r['coll_near_frac']:.4f}),"
                f" all cortex {r['coll_all']:,} ({r['coll_all_frac']:.4f}); jump p95 {r['rootjump_near_p95']:.2f}"
                f" p99.9 {r['rootjump_near_p999']:.2f} voxels/step [{time.time() - t:.0f}s]")
            json.dump(res, open(OUT, "w"), indent=1)
        del Mt, D
    # ---- thin WM: sign and foot at 1x and 2x
    thin = wm & ~ndi.binary_opening(wm, structure=ball(2)) & (dface >= 2.0)
    lab, _ = ndi.label(thin, structure=np.ones((3, 3, 3)))
    size = np.bincount(lab.ravel())
    keep = np.isin(lab, np.flatnonzero(size >= 50)[1:] if size.size > 1 else [])
    keep &= lab > 0
    if keep.any():
        around = ctx & (ndi.distance_transform_edt(~keep) <= 2.0)
        P1 = np.argwhere(around).astype(float)
        sub = np.array([[a, b, c] for a in (-0.25, 0.25) for b in (-0.25, 0.25) for c in (-0.25, 0.25)])
        P2 = (P1[:, None, :] + sub[None]).reshape(-1, 3)
        starts = {"voxel faces": M.mesh_closest(*M.voxel_face_mesh(wm))}
        for nm in ("binary contour", "white_mesh s1", "white_mesh s1.5", "twice 0.32mm"):
            v, f, _ = S[nm]
            starts[nm] = M.mesh_closest(v, f)
        starts["implicit s0.5"] = M.implicit_closest(wm, 0.5)
        starts["implicit s1"] = M.implicit_closest(wm, 1.0)
        for nm, cl in starts.items():
            r = {}
            for tag, P in (("1x", P1), ("2x", P2)):
                F, n = cl(P)
                de = np.sum((P - F) * n, 1)
                far = np.linalg.norm(P - F, axis=1) > 2.0
                r[f"inside_{tag}"] = float(np.mean(de < 0))
                r[f"far_{tag}"] = float(np.mean(far))
            res.setdefault("thin", {})[nm] = r
            log(f"thin WM sides, {nm:16s}: inside 1x {r['inside_1x']:.4f} 2x {r['inside_2x']:.4f};"
                f" foot > 2 voxels 1x {r['far_1x']:.4f} 2x {r['far_2x']:.4f}")
        res["thin_points_1x"] = int(len(P1))
    json.dump(res, open(OUT, "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
