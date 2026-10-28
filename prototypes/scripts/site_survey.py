"""Survey a box for kissing faces and make ITK-SNAP files for marking contact points.

All in 3D, in a box loaded from the full volume (box faces are never WM boundary; no gap closure).
1. White-surface curvature and normals (curv_lib.curvature, 0.8 mm scale).
2. One data-steered T from all WM; one upwind pass for roots and arc length S (collide.py, finding 11).
3. Collisions, classified as in collide.py: sheet (two fronts), junction, sheet edge.
4. Sheet pieces: connected sheet voxels (26-neighbour), each with at least MIN_SHEET voxels and centred
   >= 6 mm from the box faces. Per piece: the roots on the two sides of its voxel pairs, grown 1 mm along
   the white surface, split into pieces and sorted into two sides by which pieces face each other
   (collide2.py); each side grown 3 mm is a wall.
5. A rough reading per piece (no per-front fields): its size and centre; per side, the open thickness
   (arc length at the visible tops whose root is on that side's wall), how far the side's contact roots
   lie beyond its last visible top (geodesic, along the white surface), and its mean curvature.
Writes <name>_echo2_detrended.nii.gz, <name>_sites_overlay.nii.gz, <name>_sites_labels.txt and
<name>_sites.json (full-grid corner, and each piece in ITK-SNAP numbering of the crop) to the data folder.

Usage: site_survey.py <datadir> <workdir> <name> <lo_i,lo_j,lo_k> <hi_i,hi_j,hi_k> [<mark_i,mark_j,mark_k> ...]
(marks in full-grid voxels: the review points, drawn in the overlay)
"""

import json
import sys

sys.path.insert(0, sys.argv[2])
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse.csgraph import connected_components, dijkstra

import curv_lib as C
from collide import D_FAR, D_SAME, geodesics, shell_graph
from fit_T import fill
from prob_thickness import upwind_fields

DATA, S, NAME = sys.argv[1], sys.argv[2], sys.argv[3]
LO = np.array([int(v) for v in sys.argv[4].split(",")]); HI = np.array([int(v) for v in sys.argv[5].split(",")])
MARKS = [np.array([int(v) for v in a.split(",")]) for a in sys.argv[6:]]
MIN_SHEET, FACE_MM = 300, 6.0


def collisions(B, root, pts, nrm, ctx):
    """collide.py's collision step: pairs of 6-neighbours whose roots are > D_SAME apart along the white
    surface, and the class of each collision voxel."""
    vox = B["vox"]; shp = np.array(root.shape)
    G = shell_graph(pts, nrm, vox)
    ppos = pts * vox
    cv = np.argwhere(ctx & (root >= 0))
    PL = []
    for d in np.eye(3, dtype=int):
        q = cv + d
        ok = np.all(q < shp, 1)
        a, b = cv[ok], q[ok]
        rb = root[tuple(b.T)]
        k = rb >= 0
        a, b, ra, rb = a[k], b[k], root[tuple(a[k].T)], rb[k]
        far = np.linalg.norm(ppos[ra] - ppos[rb], axis=1) > 0.5
        PL.append(np.c_[a[far], b[far], ra[far], rb[far]])
    PL = np.vstack(PL)
    g = geodesics(G, np.sort(PL[:, 6:8], 1), D_FAR + 0.5)
    PL, gP = PL[g > D_SAME], g[g > D_SAME]
    coll = np.zeros(root.shape, bool)
    coll[tuple(PL[:, 0:3].T)] = True; coll[tuple(PL[:, 3:6].T)] = True
    cvox = np.argwhere(coll)
    offs = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)])
    nbr = np.clip(cvox[:, None, :] + offs[None], 0, shp - 1)
    R = root[tuple(nbr.reshape(-1, 3).T)].reshape(len(cvox), 27)
    Rl, cand = [], set()
    for row in R:
        u = np.unique(row[row >= 0]); Rl.append(u)
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                if np.linalg.norm(ppos[u[i]] - ppos[u[j]]) > 0.5:
                    cand.add((u[i], u[j]))
    cand = np.array(sorted(cand)) if cand else np.zeros((0, 2), int)
    gc = geodesics(G, cand, D_FAR + 0.5) if len(cand) else np.zeros(0)
    gd = {tuple(c): v for c, v in zip(cand, gc)}
    gget = lambda a, b: 0.0 if np.linalg.norm(ppos[a] - ppos[b]) <= 0.5 else gd.get((min(a, b), max(a, b)), np.inf)
    ctype = np.zeros(root.shape, np.int8)
    for v, u in zip(cvox, Rl):
        reps = []
        for r in u:
            if not any(gget(r, s) <= D_FAR for s in reps):
                reps.append(r)
        ctype[tuple(v)] = 1 if len(reps) == 2 else (2 if len(reps) >= 3 else 3)
    return ctype, PL, gP, G


def main():
    B = C.load(LO, HI, NAME)
    vox = B["vox"]; shp = np.array(B["rim"].shape)
    print(f"{NAME}: box {tuple(shp)} from full-grid {tuple(LO)}")
    K = C.curvature(B, 0.8)
    pts, nrm = K["pts"], K["n"]
    H = np.clip(0.5 * (K["k1"] + K["k2"]), -0.8, 0.8)
    Fd = C.fit_all_wm(B)
    print(f"   T fitted (cg {Fd['info']})")
    T, Tf = Fd["T"], Fd["Tf"]
    dom = np.isfinite(T)
    n1 = np.moveaxis(Fd["g"], 0, -1); n1 = n1 / (np.linalg.norm(n1, axis=-1, keepdims=True) + 1e-12)
    rootid = -np.ones(shp, np.int64); rootid[tuple(pts.T)] = np.arange(len(pts))
    order = np.argwhere(dom)[np.argsort(Tf[dom])].astype(np.int64)
    _, Sarc, root = upwind_fields(order, dom, dom & B["wm"], rootid, Tf, n1, np.zeros(shp), vox.astype(float))
    ctx = B["ctx"]
    root = np.where(ctx, root, -1); Sarc = np.where(ctx & dom, Sarc, np.nan)
    ctype, PL, gP, G = collisions(B, root, pts, nrm, ctx)
    for t, nm in ((1, "sheet"), (2, "junction"), (3, "sheet edge")):
        print(f"   {nm}: {np.sum(ctype == t):,} voxels")
    # visible tops and their roots
    tops = ctx & ndi.binary_dilation(B["rim"] == 1) & (B["dface"] >= 2.0) & (root >= 0) & np.isfinite(Sarc)
    tq = np.argwhere(tops); tr = root[tuple(tq.T)]; ts = Sarc[tuple(tq.T)]
    has_top = np.zeros(len(pts), bool); has_top[tr] = True
    # sheet pieces
    lab, nlab = ndi.label(ctype == 1, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lab.ravel())[1:]
    wall_side = np.zeros(len(pts), np.int8)  # 1, 2: the two sides of some piece
    pieces = []
    sheet_pairs = PL[(gP > D_FAR) & (ctype[tuple(PL[:, 0:3].T)] == 1)]
    plab = lab[tuple(sheet_pairs[:, 0:3].T)]
    for li in np.argsort(sizes)[::-1]:
        if sizes[li] < MIN_SHEET:
            break
        vv = np.argwhere(lab == li + 1)
        cen = vv.mean(0)
        if np.min(np.r_[cen, shp - 1 - cen] * np.r_[vox, vox]) < FACE_MM:
            continue
        pp = sheet_pairs[plab == li + 1][:, 6:8]
        if len(pp) < 20:
            continue
        Rr = np.unique(pp)
        D1 = dijkstra(G, indices=Rr, limit=1.0, min_only=True)
        M = np.flatnonzero(np.isfinite(D1))
        nc, labM = connected_components(G[M][:, M], directed=False)
        comp = -np.ones(len(pts), int); comp[M] = labM
        opp = {}
        for a, b in pp:
            ca, cb = comp[a], comp[b]
            if ca != cb:
                key = (min(ca, cb), max(ca, cb)); opp[key] = opp.get(key, 0) + 1
        if not opp:
            continue
        side = -np.ones(nc, int)
        side[max(opp.items(), key=lambda kv: kv[1])[0][0]] = 0
        changed = True
        while changed:
            changed = False
            for (ca, cb), n_ in sorted(opp.items(), key=lambda kv: -kv[1]):
                if side[ca] >= 0 and side[cb] < 0:
                    side[cb] = 1 - side[ca]; changed = True
                elif side[cb] >= 0 and side[ca] < 0:
                    side[ca] = 1 - side[cb]; changed = True
        conflicts = sum(n_ for (ca, cb), n_ in opp.items() if side[ca] >= 0 and side[ca] == side[cb])
        info = dict(id=len(pieces) + 1, voxels=int(sizes[li]),
                    centre_itk=[int(round(c)) + 1 for c in cen], extent_mm=[float(e) for e in (np.ptp(vv, 0) * vox).round(1)],
                    conflicts=int(conflicts), facing_pairs=int(sum(opp.values())), sides=[])
        for sd in (0, 1):
            seeds = M[side[labM] == sd]
            if not len(seeds):
                continue
            wall = np.isfinite(dijkstra(G, indices=seeds, limit=3.0, min_only=True))
            wall_side[wall & (wall_side == 0)] = sd + 1
            fz = np.intersect1d(np.unique(pp), seeds)
            dtop = dijkstra(G, indices=fz, limit=15.0, min_only=True)
            dt = dtop[has_top]
            ontop = wall[tr]
            info["sides"].append(dict(
                wall_voxels=int(wall.sum()),
                open_thickness_mm=float(np.median(ts[ontop])) if ontop.any() else None,
                open_thickness_iqr=float(np.subtract(*np.percentile(ts[ontop], [75, 25]))) if ontop.any() else None,
                beyond_last_top_mm=float(np.min(dt)) if np.isfinite(dt).any() else None,
                fused_roots_to_nearest_top_median_mm=float(np.median(_nearest(G, fz, has_top))),
                mean_curvature=float(np.median(H[fz]))))
        pieces.append(info)
        s_ = info["sides"]
        txt = " | ".join(f"side {k + 1}: open {d['open_thickness_mm'] or float('nan'):.2f} mm, contact roots "
                         f"{d['fused_roots_to_nearest_top_median_mm']:.2f} mm beyond the nearest top, H {d['mean_curvature']:+.2f}"
                         for k, d in enumerate(s_))
        print(f"   piece {info['id']}: {info['voxels']:,} sheet voxels, centre ITK {tuple(info['centre_itk'])}, extent "
              f"{info['extent_mm']} mm, conflicts {conflicts}/{info['facing_pairs']} | {txt}")
    # ITK-SNAP files
    seg = np.zeros(shp, np.uint8)
    seg[B["shell"]] = 1
    wp = np.zeros(shp, np.uint8); wp[tuple(pts.T)] = wall_side
    seg[(wp == 1)] = 5; seg[(wp == 2)] = 6
    seg[ctype == 3] = 4; seg[ctype == 2] = 3; seg[ctype == 1] = 2
    for k_, pc in enumerate(pieces):
        c = np.array(pc["centre_itk"]) - 1
        seg[tuple(np.clip(c, 0, shp - 1))] = 7
    for mk in MARKS:
        q = mk - LO
        if np.all((q >= 0) & (q < shp)):
            for d in [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
                seg[tuple(np.clip(q + d, 0, shp - 1))] = 8
    aff = B["p"].affine
    nib.save(nib.Nifti1Image(np.where(ctx, B["disp"], 0.0).astype(np.float32), aff), f"{DATA}/{NAME}_echo2_detrended.nii.gz")
    nib.save(nib.Nifti1Image(seg, aff), f"{DATA}/{NAME}_sites_overlay.nii.gz")
    open(f"{DATA}/{NAME}_sites_labels.txt", "w").write(
        '    0     0    0    0        0  0  0    "Clear Label"\n'
        '    1   255  220    0      0.6  1  1    "WM boundary"\n'
        '    2   255   40  220        1  1  1    "Sheet (two fronts meet)"\n'
        '    3   255   40   40        1  1  1    "Junction (three or more)"\n'
        '    4    60  200  255      0.4  1  1    "Sheet edge (a front meets itself)"\n'
        '    5    40  120  255        1  1  1    "Wall, side 1"\n'
        '    6   255  140   20        1  1  1    "Wall, side 2"\n'
        '    7   255  255  255        1  1  1    "Sheet piece centre (see the json)"\n'
        '    8    60  255   60        1  1  1    "Review points"\n')
    json.dump(dict(name=NAME, full_lo=[int(v) for v in LO], full_hi=[int(v) for v in HI], shape=[int(v) for v in shp],
                   numbering="ITK-SNAP voxel numbering of this crop (0-based crop index + 1); full grid = crop index + full_lo",
                   pieces=pieces), open(f"{DATA}/{NAME}_sites.json", "w"), indent=1)
    np.savez_compressed(f"{S}/{NAME}_survey.npz", root=root.astype(np.int32), ctype=ctype, S=Sarc.astype(np.float32),
                        lo=LO, wall_side=wall_side)
    print(f"   wrote {NAME}_echo2_detrended.nii.gz, {NAME}_sites_overlay.nii.gz, {NAME}_sites_labels.txt, {NAME}_sites.json")


def _nearest(G, src, has_top):
    """Per source root, the geodesic distance along the white surface to the nearest root with a visible top."""
    if not has_top.any():
        return np.array([np.nan])
    d = dijkstra(G, indices=np.flatnonzero(has_top), limit=15.0, min_only=True)
    return np.where(np.isfinite(d[src]), d[src], 15.0)


if __name__ == "__main__":
    main()
