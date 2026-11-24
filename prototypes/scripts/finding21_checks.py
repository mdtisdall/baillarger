"""The checks behind finding 21 (why PCC piece 1's walls failed; what made piece 3's false sheet).

These first ran as one-off commands; this script reproduces them. Subcommands:

  sides <p1_workdir> <datadir>
      PCC piece 1, at each user point: the all-WM roots just either side of the contact (along the data's
      layer normal), their straight and along-surface distances, which wall each is on, and the collision
      classes within 1 mm. Then the facing root pairs at the sheet near the points, and how many have both
      roots in one piece of the roots grown 1 mm (collide2.py's side-sorting). Writes the ITK-SNAP overlay
      PCC_SPL_p1_run_walls_diag_overlay.nii.gz (walls, the piece, a path joining two sides, sheet, points).
      Needs the work folder of the PCC_SPL_p1_run pipeline (collide_state.npz, collide2_state.npz,
      curv_PCC_SPL_p1_run.npz, PCC_SPL_p1_run_points_box.json) and PCC_SPL_p1_run_echo2_detrended.nii.gz.
  roots <sites_workdir> <datadir>
      PCC piece 3 (crop PCC_SPL_p3): its sheet's roots, the WM half-thickness there against the box, and the
      cortex-labelled voxels within 1 mm of the roots that read WM-dark in echo 3. Writes the ITK-SNAP overlay
      PCC_SPL_p3_roots_overlay.nii.gz. Needs PCC_SPL_p3_survey.npz (site_survey.py) and PCC_SPL_p3_sites.json.
  strip <sites_workdir> <datadir> [i j k]
      The WM-dark, cortex-labelled region connected to a user point (default ITK-SNAP 82, 78, 71), its shape
      and distances, and its echo decay (log echo 1 - log echo 3) against nearby WM, cortex, deep cortex,
      Fomblin and cortex next to fluid. Adds the region (label 8) and the point (9) to the piece 3 overlay.

Usage: finding21_checks.py <subcommand> <workdir> <datadir> [args]; run from the repository root, with the
scripts folder on the path (as for the other prototype scripts).
"""

import json
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra

VOX = np.array([0.16028, 0.16, 0.16028])


def _shell_graph(pts, nrm):
    from collide import shell_graph

    return shell_graph(pts, nrm, VOX)


def sides(work, data):
    import curv_lib as C

    m = json.load(open(f"{work}/PCC_SPL_p1_run_points_box.json"))
    lo, hi = np.array(m["full_lo"]), np.array(m["full_hi"])
    P = np.array(m["gap_all"])
    cs = np.load(f"{work}/collide_state.npz")
    st = np.load(f"{work}/collide2_state.npz")
    cz = np.load(f"{work}/curv_PCC_SPL_p1_run.npz")
    root, ctype, pts, nrm = cs["root"], cs["ctype"], cz["pts"], cz["n"]
    w1, w2 = st["wall1"], st["wall2"]
    B = C.load(lo, hi, "PCC_SPL_p1_run")
    nh, _ = C.tensor_normal(B)
    G = _shell_graph(pts, nrm)
    print(
        "point | collision classes within 1 mm (sheet/junction/edge) | side a, b roots (ITK) | straight / along-surface (mm) | walls"
    )
    cv = np.argwhere(ctype > 0)
    for k, q in enumerate(P):
        n_ = nh[tuple(q)]
        roots = []
        for sgn in (1, -1):
            r = -1
            for d in np.arange(0.3, 1.3, 0.16):
                x = np.round(q + sgn * d * n_ / VOX).astype(int)
                if root[tuple(x)] >= 0:
                    r = root[tuple(x)]
                    break
            roots.append(r)
        ra, rb = roots
        near = cv[np.linalg.norm((cv - q) * VOX, axis=1) < 1.0]
        cnt = [int(np.sum(ctype[tuple(near.T)] == t)) for t in (1, 2, 3)]
        if ra >= 0 and rb >= 0:
            eu = np.linalg.norm((pts[ra] - pts[rb]) * VOX)
            geo = dijkstra(G, indices=ra, limit=40.0)[rb]
        else:
            eu = geo = np.nan
        wl = lambda r: ("1" if w1[r] else ("2" if w2[r] else "-")) if r >= 0 else "?"
        itk = lambda r: tuple(int(c) + 1 for c in pts[r]) if r >= 0 else None
        print(
            f"{k + 1:2d} | {cnt} | {itk(ra)}, {itk(rb)} | {eu:.2f} / {geo:.2f} | {wl(ra)}, {wl(rb)}"
        )
    # facing pairs at the sheet near the points, and the grown root pieces (as collide2.py builds them)
    sheet = np.argwhere(ctype == 1)
    dP = np.min(np.linalg.norm((sheet[:, None, :] - P[None]) * VOX, axis=2), 1)
    sheet = sheet[dP < 4.0]
    shp = np.array(root.shape)
    pairs = []
    for d in np.eye(3, dtype=int):
        for sgn in (1, -1):
            q = np.clip(sheet + sgn * d, 0, shp - 1)
            r1, r2 = root[tuple(sheet.T)], root[tuple(q.T)]
            ok = (
                (r1 >= 0)
                & (r2 >= 0)
                & (
                    np.linalg.norm(
                        (pts[np.maximum(r1, 0)] - pts[np.maximum(r2, 0)]) * VOX, axis=1
                    )
                    > 2.0
                )
            )
            pairs.append(np.c_[r1[ok], r2[ok]])
    pairs = np.vstack(pairs)
    R = np.unique(pairs)
    D1 = dijkstra(G, indices=R, limit=1.0, min_only=True)
    M = np.flatnonzero(np.isfinite(D1))
    _, labM = connected_components(G[M][:, M], directed=False)
    comp = -np.ones(len(pts), int)
    comp[M] = labM
    same = comp[pairs[:, 0]] == comp[pairs[:, 1]]
    print(
        f"facing root pairs near the points: {len(pairs):,}; both roots in one piece: {same.sum():,} ({same.mean():.0%})"
    )
    print("pieces (voxels):", sorted(np.bincount(labM).tolist(), reverse=True))
    a, b = pairs[same][0]
    sub = G[M][:, M]
    ia, ib = np.flatnonzero(M == a)[0], np.flatnonzero(M == b)[0]
    dist, pred = dijkstra(sub, indices=ia, return_predecessors=True)
    path = [ib]
    while path[-1] != ia:
        path.append(pred[path[-1]])
    pp = pts[M[path]]
    seg = np.zeros(shp, np.uint8)
    seg[B["shell"]] = 1
    wp = np.zeros(shp, np.uint8)
    wp[tuple(pts[w1].T)] = 2
    wp[tuple(pts[w2].T)] = 3
    seg[wp > 0] = wp[wp > 0]
    seg[tuple(pts[M].T)] = 4
    for v in pp:
        seg[tuple(v)] = 5
    seg[ctype == 1] = 6
    for q in P:
        for d in [
            (0, 0, 0),
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]:
            seg[tuple(q + d)] = 7
    ref = nib.load(f"{data}/PCC_SPL_p1_run_echo2_detrended.nii.gz")
    nib.save(
        nib.Nifti1Image(seg, ref.affine),
        f"{data}/PCC_SPL_p1_run_walls_diag_overlay.nii.gz",
    )
    open(f"{data}/PCC_SPL_p1_run_walls_diag_labels.txt", "w").write(
        '    0     0    0    0        0  0  0    "Clear Label"\n'
        '    1   255  220    0      0.5  1  1    "WM boundary"\n'
        '    2    40  120  255        1  1  1    "Wall 1"\n'
        '    3   255  140   20        1  1  1    "Wall 2"\n'
        '    4    60  230  230        1  1  1    "Sheet roots grown 1 mm (one piece holds both sides)"\n'
        '    5   255  255  255        1  1  1    "A path joining the two sides through that piece"\n'
        '    6   255   40  220      0.6  1  1    "Sheet"\n'
        '    7   255    0    0        1  1  1    "Your points"\n'
    )
    print("wrote PCC_SPL_p1_run_walls_diag_overlay.nii.gz")


def _p3(work, data):
    import curv_lib as C

    box = json.load(open(f"{data}/PCC_SPL_p3_sites.json"))
    lo, hi = np.array(box["full_lo"]), np.array(box["full_hi"])
    sv = np.load(f"{work}/PCC_SPL_p3_survey.npz")
    B = C.load(lo, hi, "PCC_SPL_p3")
    lab, _ = ndi.label(sv["ctype"] == 1, structure=np.ones((3, 3, 3)))
    c = np.array([71, 72, 71]) - 1
    _, ind = ndi.distance_transform_edt(lab == 0, return_indices=True)
    piece = lab == lab[tuple(ind[:, c[0], c[1], c[2]])]
    return B, sv["root"], sv["ctype"], piece, np.argwhere(B["shell"])


def roots(work, data):
    B, root, ctype, piece, pts = _p3(work, data)
    rr = np.unique(root[piece])
    rr = rr[rr >= 0]
    din = ndi.distance_transform_edt(B["wm"], sampling=VOX)
    mx = ndi.maximum_filter(din, size=7)
    thick = mx[tuple(pts[rr].T)]
    inner = B["dface"][tuple(pts.T)] >= 6
    allthick = mx[tuple(pts.T)]
    print(
        f"piece 3 sheet: {piece.sum()} voxels, {len(rr)} roots; WM half-thickness at the roots median {np.median(thick):.2f} mm "
        f"(p10 {np.percentile(thick, 10):.2f}); all inner WM boundary median {np.median(allthick[inner]):.2f} (p10 {np.percentile(allthick[inner], 10):.2f})"
    )
    print(f"roots spread (mm) {np.round(np.ptp(pts[rr], 0) * VOX, 1)}")
    e3 = np.log(np.maximum(B["p"].echoes[..., 2].astype(float), 1.0))
    rm = np.zeros(root.shape, bool)
    rm[tuple(pts[rr].T)] = True
    zone = B["ctx"] & (ndi.distance_transform_edt(~rm, sampling=VOX) <= 1.0)
    dwm = ndi.distance_transform_edt(~B["wm"], sampling=VOX)
    thr = (
        np.median(e3[B["wm"] & (din >= 0.5)]) + np.median(e3[B["ctx"] & (dwm >= 1.0)])
    ) / 2
    allr = np.zeros(root.shape, bool)
    allr[tuple(pts[inner].T)] = True
    ref = B["ctx"] & (ndi.distance_transform_edt(~allr, sampling=VOX) <= 1.0)
    print(
        f"cortex-labelled voxels within 1 mm of the roots that read WM-dark in echo 3: {np.mean(e3[zone] < thr):.0%} "
        f"(all inner WM boundary: {np.mean(e3[ref] < thr):.0%})"
    )
    seg = np.zeros(root.shape, np.uint8)
    seg[B["wm"]] = 1
    seg[B["shell"]] = 2
    seg[np.isin(root, rr) & B["ctx"]] = 3
    seg[tuple(pts[rr].T)] = 4
    seg[piece] = 5
    seg[(ctype == 3) & (seg == 0)] = 6
    ref_img = nib.load(f"{data}/PCC_SPL_p3_echo2_detrended.nii.gz")
    nib.save(
        nib.Nifti1Image(seg, ref_img.affine), f"{data}/PCC_SPL_p3_roots_overlay.nii.gz"
    )
    open(f"{data}/PCC_SPL_p3_roots_labels.txt", "w").write(
        '    0     0    0    0        0  0  0    "Clear Label"\n'
        '    1   255  240  170      0.35 1  1    "WM (aseg)"\n'
        '    2   255  220    0      0.8  1  1    "WM boundary"\n'
        '    3    60  230  230      0.35 1  1    "Cortex whose columns start at the sheet\'s roots"\n'
        '    4    40  120  255        1  1  1    "The sheet\'s roots on the WM"\n'
        '    5   255   40  220        1  1  1    "Piece 3 sheet"\n'
        '    6    60  200  255      0.4  1  1    "Sheet edge elsewhere"\n'
    )
    print("wrote PCC_SPL_p3_roots_overlay.nii.gz")


def strip(work, data, point=(82, 78, 71)):
    B, root, ctype, piece, pts = _p3(work, data)
    E = [np.log(np.maximum(B["p"].echoes[..., i].astype(float), 1.0)) for i in range(3)]
    q = np.array(point) - 1
    dwm = ndi.distance_transform_edt(~B["wm"], sampling=VOX)
    din = ndi.distance_transform_edt(B["wm"], sampling=VOX)
    e3s = ndi.gaussian_filter(E[2], 0.7)
    thr = (
        np.median(E[2][B["wm"] & (din >= 0.5)])
        + np.median(E[2][B["ctx"] & (dwm >= 1.0)])
    ) / 2
    dark = B["ctx"] & (e3s < thr)
    lab, _ = ndi.label(dark, structure=ndi.generate_binary_structure(3, 1))
    if lab[tuple(q)] == 0:
        _, ind = ndi.distance_transform_edt(lab == 0, return_indices=True)
        cid = lab[tuple(ind[:, q[0], q[1], q[2]])]
    else:
        cid = lab[tuple(q)]
    reg = lab == cid
    vv = np.argwhere(reg) + 1
    dpiece = ndi.distance_transform_edt(~piece, sampling=VOX)
    print(
        f"at ITK {tuple(point)}: label {int(B['rim'][tuple(q)])} (3 = cortex), smoothed log echo 3 {e3s[tuple(q)]:.2f} (midway WM/cortex {thr:.2f}); "
        f"{dwm[tuple(q)]:.2f} mm from the WM label, {dpiece[tuple(q)]:.2f} mm from the sheet"
    )
    print(
        f"connected WM-dark cortex-labelled region: {reg.sum()} voxels, ITK i {vv[:, 0].min()}-{vv[:, 0].max()}, j {vv[:, 1].min()}-{vv[:, 1].max()}, "
        f"k {vv[:, 2].min()}-{vv[:, 2].max()}; touching the WM label: {int(np.sum(reg & ndi.binary_dilation(B['wm'])))}"
    )
    near = ndi.distance_transform_edt(~reg, sampling=VOX) <= 3.0
    dfl = ndi.distance_transform_edt(B["rim"] != 1, sampling=VOX)
    groups = {
        "the dark region": reg,
        "WM nearby (>= 0.5 mm inside)": B["wm"] & (din >= 0.5) & near,
        "cortex nearby (>= 1 mm from WM and fluid)": B["ctx"]
        & (dwm >= 1.0)
        & (dfl >= 0.5)
        & near
        & ~reg,
        "deep cortex nearby (< 0.5 mm from WM)": B["ctx"] & (dwm < 0.5) & near & ~reg,
        "Fomblin nearby (>= 0.5 mm in)": (B["rim"] == 1)
        & (ndi.distance_transform_edt(B["rim"] == 1, sampling=VOX) >= 0.5)
        & (ndi.distance_transform_edt(~reg, sampling=VOX) <= 6.0),
        "cortex next to fluid nearby": B["ctx"]
        & ndi.binary_dilation(B["rim"] == 1)
        & near
        & ~reg,
    }
    print("group: n | log echo 1 / 2 / 3 (medians) | decay, log e1 - log e3")
    for nm, g in groups.items():
        if g.sum():
            v = [np.median(e[g]) for e in E]
            print(
                f"  {nm:44s} {g.sum():6d} | {v[0]:.2f} / {v[1]:.2f} / {v[2]:.2f} | {np.median(E[0][g] - E[2][g]):.2f}"
            )
    path = f"{data}/PCC_SPL_p3_roots_overlay.nii.gz"
    seg = np.asarray(nib.load(path).dataobj).copy()
    seg[seg >= 7] = 0
    seg[reg] = 8
    for d in [
        (0, 0, 0),
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]:
        seg[tuple(q + d)] = 9
    nib.save(nib.Nifti1Image(seg, nib.load(path).affine), path)
    lab_path = f"{data}/PCC_SPL_p3_roots_labels.txt"
    L = [
        x
        for x in open(lab_path).read().splitlines()
        if not x.strip().startswith(("7 ", "8 ", "9 "))
    ]
    L += [
        f'    8   150   60  255        1  1  1    "WM-dark cortex-labelled region connected to {tuple(point)}"',
        f'    9   255    0    0        1  1  1    "The point {tuple(point)}"',
    ]
    open(lab_path, "w").write("\n".join(L) + "\n")
    print("updated PCC_SPL_p3_roots_overlay.nii.gz")


if __name__ == "__main__":
    cmd, work, data = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.path.insert(0, work)
    if cmd == "sides":
        sides(work, data)
    elif cmd == "roots":
        roots(work, data)
    elif cmd == "strip":
        strip(
            work,
            data,
            tuple(int(v) for v in sys.argv[4:7])
            if len(sys.argv) >= 7
            else (82, 78, 71),
        )
    else:
        raise SystemExit(f"unknown subcommand {cmd}")
