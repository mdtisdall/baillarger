"""Diagnostic views of the column sites (baillarger.column_sites): the site
counts and sizes, and a few sites as echo 1 with coloured markers.

args: SITES_TSV CONF_NPZ FEAT_DIR ECHO1 OUT_DIR [RANK ...]

SITES_TSV is a desc-columnsites_sites.tsv (its members table and sidecar are
next to it). The dseg is the third source of its sidecar. With no RANK, the
views show the sites at ranks 1 and 2, the first column-top site, and the
sites at 5 %, 20 % and 50 % of the queue.
"""

import glob
import json
import sys
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

from baillarger import column_sites as C

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SITES, CONF, FEAT, ECHO1, OUT = (Path(a) for a in sys.argv[1:6])
OUT.mkdir(parents=True, exist_ok=True)
rows, side = C.read_column_sites(SITES)
p = C.sites_paths(Path(SITES.name.replace("_desc-columnsites_sites.tsv", "_white.surf.gii")), SITES.parent)
mv, ms = C.read_members(p["members"])
z = np.load(CONF)
vert, conf, top_k, rank = z["vert"], z["conf"], z["top_k"], z["rank"]
s_of = np.empty(len(vert), np.int64)
s_of[np.searchsorted(vert, mv) if np.all(np.diff(vert) > 0) else np.argsort(vert)[np.searchsorted(np.sort(vert), mv)]] = ms
n_sites = len(rows)
cols = np.array([r["columns"] for r in rows])
ext = np.array([r["extent_mm"] for r in rows])
kind = np.array([r["kind"] for r in rows])
no_top = np.array([r["no_top"] for r in rows])
score = np.array([np.nan if r["score"] is None else r["score"] for r in rows])

# ---------------------------------------------------------------- numbers
stats = dict(
    sites=n_sites,
    columns=int(cols.sum()),
    by_kind={k: int((kind == k).sum()) for k in C.KINDS},
    columns_per_site_p10_50_90_99_max=np.percentile(cols, [10, 50, 90, 99, 100]).tolist(),
    single_column_sites=int((cols == 1).sum()),
    columns_in_single_sites=int(cols[cols == 1].sum()),
    extent_mm_p10_50_90=np.round(np.percentile(ext, [10, 50, 90]), 3).tolist(),
    no_top_columns=int(no_top.sum()),
    no_top_columns_in_no_top_sites=int(no_top[kind == "column-no-top"].sum()),
)
for share in (0.01, 0.05, 0.1, 0.2):
    m = int(round(share * len(vert)))  # the front of the column queue
    front = np.argsort(rank)[:m]
    stats[f"sites_holding_front_{int(share * 100)}pct_columns"] = int(len(np.unique(s_of[front])))
first_top = int(np.flatnonzero(kind == "column-top")[0]) + 1
stats["first_column_top_rank"] = first_top
lt = {}
for r in rows[:20000]:
    lt[r["lowest_term"]] = lt.get(r["lowest_term"], 0) + 1
stats["lowest_term_first_20000_sites"] = lt
(OUT / "stats.json").write_text(json.dumps(stats, indent=1))
print(json.dumps(stats, indent=1))

fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axs[0]
b = np.unique(np.round(np.logspace(0, np.log10(cols.max() + 1), 40)))
for k, c in zip(C.KINDS, ("#d62728", "#1f77b4")):
    ax.hist(cols[kind == k], bins=b, histtype="step", lw=1.6, color=c, label=f"{k} ({(kind == k).sum():,})")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("columns per site"); ax.set_ylabel("sites"); ax.legend(fontsize=8)
ax.set_title("Site sizes")
ax = axs[1]
ax.hist(ext, bins=50, color="#555")
ax.set_xlabel("extent: largest root distance to the seed, mm"); ax.set_ylabel("sites")
ax.set_title(f"Site extent (radius {side['RadiusMm']} mm)")
ax = axs[2]
cum = np.cumsum(cols) / cols.sum()
x = np.arange(1, n_sites + 1)
ax.plot(x, 100 * cum, color="#333", label="all columns")
ax.plot(x, 100 * np.cumsum(no_top) / max(no_top.sum(), 1), color="#d62728", label="no-top columns")
ax.axvline(first_top, color="#1f77b4", ls=":", lw=1, label="first column-top site")
ax.set_xscale("log"); ax.set_xlabel("sites reviewed, in queue order"); ax.set_ylabel("% of columns covered")
ax.legend(fontsize=8); ax.set_title("Columns covered by the front of the site queue")
ax2 = ax.twinx()
ax2.plot(x, score, ",", color="#2ca02c", alpha=0.3)
ax2.set_ylabel("seed confidence (green)", color="#2ca02c")
fig.tight_layout()
fig.savefig(OUT / "site_counts.png", dpi=80)
plt.close(fig)

# ---------------------------------------------------------------- site views
if len(sys.argv) > 6:
    pick = [int(a) for a in sys.argv[6:]]
else:
    pick = [1, 2, first_top, int(0.05 * n_sites), int(0.2 * n_sites), int(0.5 * n_sites)]
want = {}
for rk in pick:
    for i in np.flatnonzero(s_of == rk):
        want[int(vert[i])] = int(i)
P = {}
for f in sorted(glob.glob(str(FEAT / "feat_*.npz"))):
    zz = np.load(f)
    hit = np.flatnonzero(np.isin(zz["vert"], list(want)))
    for h in hit:
        P[want[int(zz["vert"][h])]] = zz["P"][h].astype(np.float64)
dseg = nib.load(side["Sources"][2])
A = dseg.affine
inv = np.linalg.inv(A)
eimg = nib.load(str(ECHO1))
HALF, RES = 4.0, 0.04
t = np.arange(-HALF, HALF + 1e-9, RES)
for rk in pick:
    r = rows[rk - 1]
    mem = np.flatnonzero(s_of == rk)
    seed = int(np.flatnonzero(vert == r["vertex"])[0])
    c_mm = P[seed][0]
    cv = c_mm @ inv[:3, :3].T + inv[:3, 3]
    lo = np.maximum(np.floor(cv).astype(int) - 40, 0)
    hi = np.minimum(np.ceil(cv).astype(int) + 41, dseg.shape)
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    lab = np.asarray(dseg.dataobj[box], np.int16)
    e1 = np.asarray(eimg.dataobj[box], np.float32)
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.8))
    for ax, ax_i, name in zip(axs, range(3), ("plane x (sagittal)", "plane y (coronal)", "plane z (axial)")):
        a, b = [d for d in range(3) if d != ax_i]
        U, V = np.meshgrid(t, t, indexing="xy")
        X = np.repeat(c_mm[None, None], len(t), 0).repeat(len(t), 1)
        X = X.copy()
        X[..., a] += U; X[..., b] += V
        q = (X @ inv[:3, :3].T + inv[:3, 3] - lo).reshape(-1, 3).T
        img = map_coordinates(e1, q, order=1, mode="nearest").reshape(U.shape)
        lb = map_coordinates(lab, q, order=0, mode="nearest").reshape(U.shape)
        vlo, vhi = np.percentile(img, [1, 99])
        ext_ = (t[0], t[-1], t[0], t[-1])
        ax.imshow(img, cmap="gray", vmin=vlo, vmax=vhi, origin="lower", extent=ext_)
        ax.contour(t, t, np.isin(lb, (3, 42)).astype(float), [0.5], colors="#ffd400", linewidths=1.0)
        ax.contour(t, t, np.isin(lb, (0, 24, 4, 5, 31, 43, 44, 63)).astype(float), [0.5], colors="#00e5ff", linewidths=1.0)
        # the site's columns: roots (filled, by confidence), rays to the top (thin), tops (X)
        for i in mem:
            X_ = P[int(i)]
            d = X_ - c_mm
            k = int(top_k[i])
            col = "#ff3030" if k < 0 else "#ff8c00"
            near = abs(d[0, ax_i]) <= 1.0
            alpha = 0.9 if near else 0.25
            end = k if k >= 0 else len(X_) - 1
            ax.plot(d[: end + 1, a], d[: end + 1, b], color=col, lw=0.6, alpha=alpha)
            ax.plot(d[0, a], d[0, b], "o", ms=4, mfc=col, mec="black", mew=0.4, alpha=alpha)
            if k >= 0:
                ax.plot(d[k, a], d[k, b], "X", ms=6, mfc="#ff3df0", mec="black", mew=0.4, alpha=alpha)
        ax.plot(0, 0, "o", ms=10, mfc="white", mec="black")
        ax.set_xlim(t[0], t[-1]); ax.set_ylim(t[0], t[-1])
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("mm"); ax.set_ylabel("mm")
    sc = "n/a" if r["score"] is None else f"{r['score']:.2f}"
    fig.suptitle(
        f"Site {rk} of {n_sites:,} ({r['kind']}): {r['columns']} columns, {r['no_top']} with no top, "
        f"seed confidence {sc}, lowest term {r["lowest_term"] or "n/a"}, voxel ({r['i']}, {r['j']}, {r['k']})\n"
        "echo 1; yellow: cortex label; cyan: outside medium label; white dot: seed root; "
        "red: no-top column (root, ray); orange: column with a top (root, ray to the top); magenta X: top; "
        "faint: more than 1 mm from the plane",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(OUT / f"site_{rk}.png", dpi=72)
    plt.close(fig)
    print("site", rk, "done", flush=True)
