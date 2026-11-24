"""Is the kissing stretch unequal? Root walls (finding 11) against finding 8's geometric walls, with
column lengths measured by tracing, not read from T.

All in 3D, in the box of collide.py (the test crop grown by 15 voxels), gap closed as before.
1. Geometric walls, rebuilt with finding 8's rule (prob_thickness.py): WM-boundary voxels within 7 mm of
   the points, facing them, split by the side of the local contact surface (PCA normals of nearby
   points, orientation chained from bank A's end).
2. Per-bank fields on those walls, exactly as the per-front fields of collide2.py (closed cortex within
   9 mm of the points; data normal signed away from the wall).
3. At each point, for both kinds of wall: T at the point; the column traced down -grad T to the wall
   (its length and the WM voxel it reaches); and the straight distance to the nearest voxel of the wall,
   a lower bound that needs no field.
4. Views: the four walls and the traced columns on the user's planes; an ITK-SNAP overlay.

Usage: unequal_check.py <outdir> <workdir> <datadir>. Needs collide2_state.npz, curv_M1S1.npz.
"""

import json
import sys

sys.path.insert(0, sys.argv[2])
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

import curv_lib as C
import flatten_fused2 as F
from collide import CLOSE_MM, GROW, close_gap
from collide2 import REGION_MM, _pts_mask
from fit_T import fill, solve_field

OUT, S, DATA = sys.argv[1], sys.argv[2], sys.argv[3]
WALL_MM = 7.0


def geometric_walls(shell, wm, P, A_end, vox):
    """Finding 8's rule (prob_thickness.py), unchanged."""
    vv = vox
    nrm = np.zeros_like(P)
    for n_, q in enumerate(P):
        d = np.linalg.norm((P - q) * vv, axis=1)
        nb = P[np.argsort(d)[:max(4, int((d < 2.5).sum()))]]
        w_, V_ = np.linalg.eigh(np.cov(((nb - nb.mean(0)) * vv).T))
        nrm[n_] = V_[:, 0]
    start = int(np.argmin(np.linalg.norm((P - A_end.mean(0)) * vv, axis=1)))
    if np.dot(nrm[start], (A_end.mean(0) - P[start]) * vv) < 0:
        nrm[start] *= -1
    done = {start}
    while len(done) < len(P):
        best = None
        for a in done:
            for b in range(len(P)):
                if b in done:
                    continue
                dd = np.linalg.norm((P[a] - P[b]) * vv)
                if best is None or dd < best[0]:
                    best = (dd, a, b)
        _, a, b = best
        if np.dot(nrm[a], nrm[b]) < 0:
            nrm[b] *= -1
        done.add(b)
    sh = np.argwhere(shell)
    dpt = np.linalg.norm((sh[:, None, :] - P[None]) * vv, axis=2)
    near = dpt.min(1); nn = dpt.argmin(1)
    side = ((sh - P[nn]) * vv * nrm[nn]).sum(1)
    d_out = ndi.distance_transform_edt(~wm, sampling=vox)
    gout = np.stack(np.gradient(ndi.gaussian_filter(d_out, 1.0), *vox), -1)
    gout /= np.linalg.norm(gout, axis=-1, keepdims=True) + 1e-12
    to_p = (P[nn] - sh) * vv
    to_p /= np.linalg.norm(to_p, axis=1, keepdims=True) + 1e-12
    facing = (gout[tuple(sh.T)] * to_p).sum(1) > 0.3
    keep = (near < WALL_MM) & facing & (np.abs(side) < 5.0)
    walls = {}
    for X, sgn in (("A", 1), ("B", -1)):
        m = np.zeros(shell.shape, bool)
        m[tuple(sh[keep & (sgn * side > 0.5)].T)] = True
        walls[X] = m
    return walls


def front_field(B, ctx, region, wmask, nh, coh):
    vox = B["vox"]
    dom = (ctx & region) | wmask
    dW = ndi.distance_transform_edt(~wmask, sampling=vox)
    g = np.stack(np.gradient(ndi.gaussian_filter(dW, 1.0), *vox), -1)
    gh = g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-12)
    sg = np.sign((nh * gh).sum(-1)); sg[sg == 0] = 1
    dirv = coh[..., None] * nh * sg[..., None] + C.LAM * gh
    dirv /= np.linalg.norm(dirv, axis=-1, keepdims=True) + 1e-12
    T, info = solve_field(dom, wmask, [dirv[..., a] for a in range(3)], tuple(vox))
    return np.where(dom, T, np.nan)


def descend(Tf, g, x, wall, vox):
    pts_ = [x.copy()]
    shp = np.array(Tf.shape) - 1
    for _ in range(800):
        v = -np.array([F.interp(c, x) for c in g]); nv = np.linalg.norm(v)
        if nv < 1e-9:
            break
        x = x + 0.25 * v / nv; pts_.append(x.copy())
        if wall[tuple(np.clip(np.round(x).astype(int), 0, shp))]:
            break
    p = np.array(pts_)
    return p, float(np.sum(np.linalg.norm(np.diff(p, axis=0) * vox, axis=1)))


def main():
    m = json.load(open(f"{S}/M1S1_test_points.json"))
    lo = np.array(m["full_lo"]) - GROW; hi = np.array(m["full_hi"]) + GROW
    B = C.load(lo, hi, "M1S1")
    vox = B["vox"]; shp = np.array(B["rim"].shape)
    P = np.array(m["gap_all"], float) + GROW
    A_end = np.array(m["banks"]["A"], float) + GROW
    rim0 = B["rim"]; rim = rim0.copy()
    rim[tuple(close_gap(rim0, P, vox, B["dface"]).T)] = 3
    ctx = (rim == 3) & ~B["miss"]
    region = ndi.distance_transform_edt(~_pts_mask(P, rim.shape), sampling=vox) <= REGION_MM
    cz = np.load(f"{S}/curv_M1S1.npz"); pts = cz["pts"]
    st = np.load(f"{S}/collide2_state.npz")
    W = {}
    for X in ("1", "2"):
        w = np.zeros(rim.shape, bool); w[tuple(pts[st[f"wall{X}"]].T)] = True
        W[X] = w
    G8 = geometric_walls(B["shell"], B["wm"], P, A_end, vox)
    W["A"], W["B"] = G8["A"], G8["B"]
    for a in ("1", "2"):
        for b in ("A", "B"):
            print(f"overlap root wall {a} with geometric wall {b}: {np.sum(W[a] & W[b]):,} voxels "
                  f"(of {W[a].sum():,} and {W[b].sum():,})")
    nh, coh = C.tensor_normal(B)
    Tm = {"1": st["T1"].astype(float), "2": st["T2"].astype(float)}
    for X in ("A", "B"):
        Tm[X] = front_field(B, ctx, region, W[X], nh, coh)
        print(f"geometric bank {X}: field fitted")
    Fd = {}
    for X, T in Tm.items():
        Tf = fill(T)
        Fd[X] = (Tf, np.gradient(Tf, *vox))
    wpos = {X: np.argwhere(W[X]) for X in W}
    rows = []
    paths = {X: [] for X in W}
    print("point: per wall, T at the point / traced column length / straight distance to the wall (mm); "
          "traced end ITK")
    itk = lambda v: tuple(int(c) for c in np.round(np.asarray(v) - GROW + 1))
    for k, q in enumerate(P):
        r = {"k": k}
        line = f"{k + 1:2d} ITK {itk(q)}"
        for X in ("1", "2", "A", "B"):
            Tf, g = Fd[X]
            path, plen = descend(Tf, g, q.copy(), W[X], vox)
            paths[X].append(path)
            dmin = np.min(np.linalg.norm((wpos[X] - q) * vox, axis=1))
            r[X] = (F.interp(Tf, q), plen, dmin)
            line += f" | {X}: {r[X][0]:.2f}/{plen:.2f}/{dmin:.2f} end {itk(path[-1])}"
        rows.append(r)
        print(line)
    for X in ("1", "2", "A", "B"):
        a = np.array([r[X] for r in rows])
        print(f"wall {X}: T median {np.median(a[:, 0]):.2f}, traced length median {np.median(a[:, 1]):.2f} "
              f"[{a[:, 1].min():.2f}-{a[:, 1].max():.2f}], straight distance median {np.median(a[:, 2]):.2f} "
              f"[{a[:, 2].min():.2f}-{a[:, 2].max():.2f}]")
    np.save(f"{S}/unequal_rows.npy", rows, allow_pickle=True)

    # ---------- views
    disp = B["disp"]; ctx0 = B["ctx"]
    vmax = np.nanpercentile(np.abs(disp[ctx0]), 98)
    planes = [(0, 45 + GROW, "i = 46"), (0, 27 + GROW, "i = 28"), (0, 19 + GROW, "i = 20"), (2, 89 + GROW, "k = 90")]
    cols = {"1": "#2b7bd9", "2": "#e8913a", "A": "#00c8c8", "B": "#ff4fd8"}
    fig, axs = plt.subplots(2, 4, figsize=(28, 14.5))
    for r_, pair in enumerate((("1", "2"), ("A", "B"))):
        for c_, (ax_i, c0, lab) in enumerate(planes):
            a = axs[r_, c_]
            sl = [slice(None)] * 3; sl[ax_i] = c0; sl = tuple(sl)
            oth = [x for x in range(3) if x != ax_i]
            a.imshow(np.where(ctx[sl], disp[sl], np.nan).T, origin="lower", cmap="gray", vmin=-vmax, vmax=vmax)
            a.contour(B["wm"][sl].T.astype(float), levels=[0.5], colors="y", linewidths=0.5)
            for X in pair:
                vv = np.argwhere(W[X][sl])
                a.scatter(vv[:, 0], vv[:, 1], s=6, c=cols[X], marker="s", linewidths=0,
                          label=("root wall " if X in "12" else "geometric wall ") + X)
                for k, path in enumerate(paths[X]):
                    if abs(P[k, ax_i] - c0) <= 1.5:
                        a.plot(path[:, oth[0]], path[:, oth[1]], "-", c=cols[X], lw=1.2)
            selp = np.abs(P[:, ax_i] - c0) <= 1.5
            a.plot(P[selp, oth[0]], P[selp, oth[1]], "x", c="#ffd23f", ms=9, mew=2)
            win = P[selp][:, oth].mean(0) if selp.any() else P[:, oth].mean(0)
            a.set_xlim(win[0] - 35, win[0] + 35); a.set_ylim(win[1] - 35, win[1] + 35)
            a.set_xticks([]); a.set_yticks([])
            a.set_title(f"{lab} (ITK-SNAP): {'root walls (finding 11)' if r_ == 0 else 'geometric walls (finding 8 rule)'}",
                        fontsize=11)
            if c_ == 0:
                a.legend(loc="lower left", fontsize=8, markerscale=2)
    fig.suptitle("Walls and columns traced down each wall's own field from your points (yellow x), projected; all 3D. "
                 "Top: walls from roots. Bottom: finding 8's geometric walls.", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{OUT}/unequal_walls.png", dpi=75); plt.close(fig)
    fig, a = plt.subplots(figsize=(14, 6))
    ks = np.arange(1, len(rows) + 1)
    for X, mk in (("1", "o"), ("2", "s"), ("A", "^"), ("B", "v")):
        arr = np.array([r[X] for r in rows])
        a.plot(ks, arr[:, 1], mk + "-", c=cols[X], label=f"wall {X}: traced column length")
        a.plot(ks, arr[:, 2], mk + ":", c=cols[X], alpha=0.6, label=f"wall {X}: straight distance (lower bound)")
    a.set_xlabel("your contact point"); a.set_ylabel("mm"); a.set_xticks(ks)
    a.set_title("Each wall's column length up to your points (traced), and the straight distance to the wall")
    a.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(f"{OUT}/unequal_lengths.png", dpi=80); plt.close(fig)
    # ITK-SNAP overlay (test crop space)
    crop = tuple(slice(GROW, -GROW) for _ in range(3))
    seg = np.zeros(rim.shape, np.uint8)
    seg[B["shell"]] = 1
    for X, lab in (("1", 2), ("2", 3), ("A", 4), ("B", 5)):
        seg[W[X] & (seg == 1)] = lab
        seg[W[X] & (seg > 1) & (seg != lab)] = 6  # in a root wall and a geometric wall
    for X, lab in (("1", 7), ("2", 8), ("A", 9), ("B", 10)):
        for path in paths[X]:
            for v in np.round(path).astype(int):
                v = tuple(np.clip(v, 0, shp - 1))
                if seg[v] == 0:
                    seg[v] = lab
    for q in np.round(P).astype(int):
        seg[tuple(q)] = 11
    ref = nib.load(f"{DATA}/M1S1_test_echo2_detrended.nii.gz")
    nib.save(nib.Nifti1Image(seg[crop], ref.affine), f"{DATA}/M1S1_unequal_overlay.nii.gz")
    open(f"{DATA}/M1S1_unequal_labels.txt", "w").write(
        '    0     0    0    0        0  0  0    "Clear Label"\n'
        '    1   255  220    0      0.5  1  1    "WM boundary"\n'
        '    2    40  120  255        1  1  1    "Root wall 1"\n'
        '    3   255  140   20        1  1  1    "Root wall 2"\n'
        '    4     0  200  200        1  1  1    "Geometric wall A (finding 8)"\n'
        '    5   255   80  220        1  1  1    "Geometric wall B (finding 8)"\n'
        '    6   255  255  255        1  1  1    "In a root wall and a geometric wall"\n'
        '    7   100  160  255        1  1  1    "Columns, root wall 1"\n'
        '    8   255  180   90        1  1  1    "Columns, root wall 2"\n'
        '    9    90  230  230        1  1  1    "Columns, geometric wall A"\n'
        '   10   255  140  235        1  1  1    "Columns, geometric wall B"\n'
        '   11   255    0    0        1  1  1    "Your points"\n')
    print("wrote unequal_walls.png, unequal_lengths.png, M1S1_unequal_overlay.nii.gz")


if __name__ == "__main__":
    main()
