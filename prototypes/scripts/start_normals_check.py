"""How well a start mesh's faces keep the white surface's normals (T1): per
face near the white surface, its normal's angle to the white surface it
stands for (the summed normals of the white vertices nearest to it) and to
the smoothed white normals at its corners; against its circumradius over
the size field (sigma_surface.sizing), and its circumradius over its
shortest edge.

usage (from prototypes/scripts): start_normals_check.py CROP NAME=MESH.npz ...
"""
import sys, numpy as np
from scipy.spatial import cKDTree
import crop_site as CS, sigma_surface as SU
from baillarger import surface_remesh as SR
site = CS.load_site(sys.argv[1]); h = float(site.vox.mean()); W = site.v * site.vox; shape = np.array(site.labels.shape)
nW = site.normal / np.linalg.norm(site.normal, axis=1)[:, None]
size = SU.sizing(site, 1.0)                      # the size field, clipped to 1..5 h
wt = cKDTree(W); win = np.minimum(site.v, shape - 1 - site.v).min(1) > 6
for kv in sys.argv[2:]:
    name, path = kv.split("="); z = np.load(path); V = np.asarray(z["V"], float); F = np.asarray(z["F"], np.int64)
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    a, b, c = np.linalg.norm(B - C, axis=1), np.linalg.norm(C - A, axis=1), np.linalg.norm(A - B, axis=1)
    n = np.cross(B - A, C - A); area = 0.5 * np.linalg.norm(n, axis=1); n /= np.maximum(2 * area, 1e-300)[:, None]
    rc = a * b * c / np.maximum(4 * area, 1e-300)
    emin, emax = np.minimum(np.minimum(a, b), c), np.maximum(np.maximum(a, b), c)
    cen = (A + B + C) / 3
    inner = np.minimum(cen / site.vox, shape - 1 - cen / site.vox).min(1) > 8
    # the white surface each face stands for: the inner white vertices whose nearest face it is
    q, d, fc = SR.closest_on(V, F, W[win], 24)
    S = np.zeros((len(F), 3)); np.add.at(S, fc, nW[win] * site.varea[win, None]); cnt = np.bincount(fc, minlength=len(F))
    aS = np.degrees(np.arccos(np.clip((n * S).sum(1) / np.maximum(np.linalg.norm(S, axis=1), 1e-300), -1, 1)))
    # corners: the smoothed white normal at the white vertex nearest each corner
    ic = wt.query(V)[1]; nc = nW[ic]; dc = np.linalg.norm(V - W[ic], axis=1) / h
    aC = np.degrees(np.arccos(np.clip(np.einsum("fd,fkd->fk", n, nc[F]), -1, 1))).max(1)
    sz = size[ic][F].min(1)                       # the smallest size at its corners
    ok = inner & (cnt > 0)
    r1, r2 = rc / sz, rc / np.maximum(emin, 1e-300)
    bad = aS > 30
    print(f"== {name}: faces {len(F)}, judged (inner, standing for white vertices) {int(ok.sum())}; more than 30° off: {int((bad & ok).sum())} ({100 * (bad & ok).sum() / ok.sum():.1f} %); corners within {np.percentile(dc, [50, 99]).round(2)} h of a white vertex (p50/p99)")
    print(f"   corner test against footprint test (30°): both off {int((ok & bad & (aC > 30)).sum())}, footprint only {int((ok & bad & (aC <= 30)).sum())}, corner only {int((ok & ~bad & (aC > 30)).sum())}, neither {int((ok & ~bad & (aC <= 30)).sum())}")
    for nm, x, edges in (("circumradius / size", r1, [0, 0.5, 0.75, 1.0, 1.5, 2.5, 1e9]), ("circumradius / shortest edge", r2, [0, 0.7, 1.0, 1.5, 3, 10, 1e9]), ("longest edge / size", emax / sz, [0, 0.75, 1.0, 1.33, 2, 4, 1e9])):
        row = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = ok & (x >= lo) & (x < hi)
            row.append(f"[{lo:g},{hi:g}): {int(m.sum())} faces, {100 * (bad & m).sum() / max(m.sum(), 1):.0f} % off, p90 angle {np.percentile(aS[m], 90):.0f}°" if m.sum() else f"[{lo:g},{hi:g}): 0")
        print(f"   {nm}: " + " | ".join(row))
    both = ok & (r1 < 1.0) & (r2 < 1.5)
    print(f"   size and shape both held (circumradius < size, circumradius < 1.5 shortest edge): {int(both.sum())} faces, off {int((bad & both).sum())}, angle p50/90/max {np.percentile(aS[both], [50, 90, 100]).round(1) if both.any() else None}")
