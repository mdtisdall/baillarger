"""Rows for zero-thickness tops: two planes that contain the ray at its root
(at right angles to each other), echo 1 with the dseg's cortex and outside
contours, the ray, its inward extension, the root and the top.
Alternates columns with much GM-like signal behind the root (term at the cap)
and columns with none; MODE=fluid: zero tops with >= 1 mm fluid-like signal
behind the root. args: CONF_NPZ FEAT_DIR SURF OUT_DIR N_EACH SEED; inputs: top_paths.py"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np
from nibabel.gifti.gifti import GiftiImage
from scipy.ndimage import map_coordinates

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import top_paths as TP  # noqa: E402

DSEG, ECHOES = TP.dseg(), TP.echoes()

CONF, FEAT, SURF, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
NE, SEED = int(sys.argv[5]), int(sys.argv[6])
OUT.mkdir(parents=True, exist_ok=True)
STEP, INWARD, HALF, RES = 0.04, 1.5, 3.0, 0.02
r = np.load(CONF)
names = list(r["term_names"])
import os
zero = (r["top_k"] >= 0) & (r["top_k"] <= 1)
rng = np.random.default_rng(SEED)
if os.environ.get("MODE") == "fluid":  # zero tops with fluid-like signal behind the root (>= 1 mm)
    pick = list(rng.choice(np.flatnonzero(zero & (r["fl_behind"] >= 1.0)), NE, replace=False))
    group = ["fluid-like behind the root"] * NE
else:
    tg = r["terms"][:, [n.startswith(("gm_behind", "non_wm")) for n in names].index(True)]
    strong = rng.choice(np.flatnonzero(zero & (tg <= -8)), NE, replace=False)
    clear = rng.choice(np.flatnonzero(zero & (tg == 0)), NE, replace=False)
    pick = [x for pair in zip(strong, clear) for x in pair]
    group = ["much GM-like behind the root", "no GM-like behind the root"] * NE
want = {int(r["vert"][i]): j for j, i in enumerate(pick)}
Pk, gb = {}, {}
for f in sorted(glob.glob(str(FEAT / "feat_*.npz"))):
    z = np.load(f)
    if "empty" in z.files:
        continue
    hit = np.flatnonzero(np.isin(z["vert"], list(want)))
    for h in hit:
        j = want[int(z["vert"][h])]
        Pk[j], gb[j] = z["P"][h].astype(np.float64), float(z["gm_behind"][h])
dimg = nib.load(str(DSEG))
A = dimg.affine
inv = np.linalg.inv(A)
eimg = nib.load(str(ECHOES[0]))
spts = np.asarray(GiftiImage.from_filename(str(SURF), mmap=True).darrays[0].data)
rows = []
for j, i in enumerate(pick):
    X = Pk[j]
    root = X[0]
    d0 = X[min(5, len(X) - 1)] - root
    d0 /= np.linalg.norm(d0)
    a = np.eye(3)[np.argmin(np.abs(d0))]
    u = np.cross(d0, a); u /= np.linalg.norm(u)
    v = np.cross(d0, u)
    c = root @ inv[:3, :3].T + inv[:3, 3]
    lo = np.maximum(np.floor(c).astype(int) - 30, 0)
    hi = np.minimum(np.ceil(c).astype(int) + 31, dimg.shape)
    box = tuple(slice(int(p), int(q)) for p, q in zip(lo, hi))
    lab = np.asarray(dimg.dataobj[box], np.uint8)
    e1 = np.asarray(eimg.dataobj[box], np.float32)
    t = np.arange(-HALF, HALF + 1e-9, RES)
    T, S = np.meshgrid(t, t, indexing="ij")  # T along the ray, S across
    near = np.flatnonzero((np.abs(spts - root) < 0.5).all(1))
    tv = int(near[np.argmin(np.linalg.norm(spts[near] - root, axis=1))]) if len(near) else -1
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.6))
    for ax, w, wn in zip(axs, (u, v), ("plane 1", "plane 2 (at right angles)")):
        Xp = root + T[..., None] * d0 + S[..., None] * w
        q = (Xp @ inv[:3, :3].T + inv[:3, 3] - lo).reshape(-1, 3).T
        img = map_coordinates(e1, q, order=1, mode="nearest").reshape(T.shape)
        lb = map_coordinates(lab, q, order=0, mode="nearest").reshape(T.shape)
        vlo, vhi = np.percentile(img, [1, 99])
        ext = (t[0], t[-1], t[0], t[-1])
        ax.imshow(img, cmap="gray", vmin=vlo, vmax=vhi, origin="lower", extent=ext)  # rows: along the ray (up)
        ax.contour(t, t, (lb == 42).astype(float), [0.5], colors="#ffd400", linewidths=1.2)
        ax.contour(t, t, np.isin(lb, (0, 43)).astype(float), [0.5], colors="#00e5ff", linewidths=1.2)
        pt = (X - root) @ d0
        ps = (X - root) @ w
        ax.plot(ps, pt, color="#ff3030", lw=1.6)
        ax.plot([0, 0], [-INWARD, 0], color="#ff8c00", lw=1.6, ls="--")
        ax.plot(0, 0, "o", ms=9, mfc="white", mec="black")
        k = int(r["top_k"][i])
        ax.plot(ps[k], pt[k], "X", ms=13, mfc="#ff3df0", mec="black")
        ax.set_xlabel(f"across the ray, mm ({wn})")
        ax.set_ylabel("along the ray at the root, mm (up = outward)")
        ax.set_xlim(t[0], t[-1]); ax.set_ylim(t[0], t[-1])
    rank_pct = 100 * r["rank"][i] / len(r["rank"])
    info = dict(row=j + 1, true_vertex=tv, root_mm=root.round(2).tolist(), conf=round(float(r["conf"][i]), 2),
                queue_pct=round(float(rank_pct), 1), gm_behind_mm=round(gb[j], 2), top_mm=float(r["top_mm"][i]))
    rows.append(info)
    fig.suptitle(f"Row {j + 1}: true vertex {tv}, root {root.round(1).tolist()} mm, top at {info['top_mm']:.2f} mm\n"
                 "echo 1; yellow: cortex label; cyan: outside medium label; red: ray; orange dashed: 1.5 mm behind the root; "
                 "white dot: root; magenta X: top", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"row_{j + 1}.png", dpi=90)
    plt.close(fig)
    info["group"] = group[j]
    print(json.dumps(info), flush=True)
(OUT / "rows.json").write_text(json.dumps(rows, indent=1))
