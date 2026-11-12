"""Prototype: thick_local with a per-column scale. Tops are unchanged (the propagation is not rerun);
only the final local term is rescored:
    old: -0.5 (r / s0)^2,                    s0 = the stage's sigma_local_mm
    new: -0.5 (r / s)^2, s = sqrt(s0^2 + sn^2), sn = 1.4826 MAD of the neighbours' predictions (mm)
both capped at -8; r and the prediction exactly as top_confidence.py. Then conf = sum of terms, the
queue rank, and comparisons: roughness, the merge tree's regions, the front of the queue.
args: FEAT_DIR CONF_DIR OUT_DIR"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

FEAT, CDIR, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
STEP = 0.04
S0 = float(json.loads((CDIR / "summary.json").read_text())["sigma_local_mm"])
V, PP, EV = [], [], []
for f in sorted(glob.glob(str(FEAT / "feat_*.npz"))):
    z = np.load(f)
    EV.append(z["edges_v"])
    if "empty" in z.files:
        continue
    V.append(z["vert"]); PP.append(z["P"])
vert = np.concatenate(V); P = np.concatenate(PP); del PP
m, ns, _ = P.shape
order = np.argsort(vert); vs = vert[order]


def idx(v):
    j = np.clip(np.searchsorted(vs, v), 0, m - 1)
    return np.where(vs[j] == v, order[j], -1)


E = np.unique(np.sort(np.concatenate(EV), 1), axis=0)
ia, ib = idx(E[:, 0]), idx(E[:, 1]); ok = (ia >= 0) & (ib >= 0); ia, ib = ia[ok], ib[ok]
src, dst = np.r_[ia, ib], np.r_[ib, ia]; o = np.argsort(src, kind="stable")
nb_ptr = np.r_[0, np.cumsum(np.bincount(src, minlength=m))]; nb_idx = dst[o]
Z = dict(np.load(CDIR / "top_confidence.npz"))
assert (Z["vert"] == vert).all(), "column order differs from the features"
top_k = Z["top_k"].astype(np.int64)
names = [str(s) for s in Z["term_names"]]
J = names.index("thick_local")
T_old = Z["terms"].astype(np.float64)
print(f"{m} columns; s0 {S0:.4f} mm", flush=True)


def tangent(c, k):
    a, b = P[c, max(k - 1, 0)], P[c, min(k + 1, ns - 1)]
    d = (b - a).astype(np.float64); n = np.sqrt(d @ d)
    return d / n if n > 0 else None


tl_old = np.full(m, np.nan); tl_new = np.full(m, np.nan); sn = np.full(m, np.nan); res = np.full(m, np.nan)
for b in np.flatnonzero(top_k >= 0):
    k = int(top_k[b]); preds = []
    for c in nb_idx[nb_ptr[b]:nb_ptr[b + 1]]:
        kc = top_k[c]
        if kc < 0:
            continue
        tc = tangent(c, kc)
        if tc is None:
            continue
        h = (P[b] - P[c, kc]) @ tc
        sc = np.flatnonzero(np.signbit(h[1:]) != np.signbit(h[:-1]))
        if len(sc):
            preds.append(int(sc[0]))
        elif h[0] >= 0:
            preds.append(0)
    if not preds:
        tl_old[b] = tl_new[b] = 0.0
        continue
    kp = int(np.median(preds))
    r = (k - kp) * STEP
    pr = np.array(preds) * STEP
    s_n = 1.4826 * float(np.median(np.abs(pr - np.median(pr))))
    s = np.sqrt(S0 ** 2 + s_n ** 2)
    res[b], sn[b] = r, s_n
    tl_old[b] = max(-0.5 * (r / S0) ** 2, -8.0)
    tl_new[b] = max(-0.5 * (r / s) ** 2, -8.0)
has = top_k >= 0
chk = np.isclose(tl_old[has], T_old[has, J], atol=1e-4).mean()
print(f"recomputed old term matches the stored one: {chk:.4f}", flush=True)
T_new = T_old.copy(); T_new[:, J] = np.where(has, tl_new, np.nan)
conf_old = Z["conf"].astype(np.float64)
conf_new = np.where(has, np.nansum(T_new, 1), -np.inf)
rank_new = np.empty(m, np.int64); rank_new[np.argsort(conf_new, kind="stable")] = np.arange(m)
Zn = dict(Z); Zn["terms"] = T_new.astype(np.float32); Zn["conf"] = conf_new; Zn["rank"] = rank_new
np.savez(OUT / "hemi_top_conf.npz", **Zn)
np.save(OUT / "sn.npy", sn); np.save(OUT / "res.npy", res)

# ---- comparisons
el = np.r_[ia][ia < ib], np.r_[ib][ia < ib]


def lag1(f):
    a, b = f[el[0]], f[el[1]]; mm = np.isfinite(a) & np.isfinite(b); return float(np.corrcoef(a[mm], b[mm])[0, 1])


def regions(rank, frac):
    """connected regions of the first frac of the queue on the column graph"""
    sel = rank < frac * m
    a, b = el[0], el[1]; keep = sel[a] & sel[b]
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    idxs = np.flatnonzero(sel); remap = -np.ones(m, np.int64); remap[idxs] = np.arange(len(idxs))
    G = coo_matrix((np.ones(keep.sum()), (remap[a[keep]], remap[b[keep]])), shape=(len(idxs),) * 2)
    _, lab = connected_components(G, directed=False)
    sz = np.bincount(lab)
    return dict(regions=int(len(sz)), singletons=int((sz == 1).sum()), size_p50=float(np.median(sz)),
                size_p90=float(np.percentile(sz, 90)), size_max=int(sz.max()), share_in_singletons=float((sz == 1).sum() / sel.sum()))


rank_old = Z["rank"]
nonwm = names.index("non_wm_behind_root")
out = dict(s0=S0, sn_p50=float(np.nanmedian(sn)), sn_p90=float(np.nanpercentile(sn, 90)))
for nm, tl, cf, rk in (("old", tl_old, conf_old, rank_old), ("new", tl_new, conf_new, rank_new)):
    f = tl[has]
    d = dict(term_lag1=lag1(np.where(has, tl, np.nan)), conf_lag1=lag1(np.where(np.isfinite(cf), cf, np.nan)),
             term_le_m1=float((f <= -1).mean()), term_at_cap=float((f <= -8 + 1e-9).mean()))
    for fr in (0.01, 0.05):
        d[f"regions_first_{fr}"] = regions(rk, fr)
        front = rk < fr * m
        lowest = np.argmin(np.where(has[:, None] & np.isfinite(T_new if nm == "new" else T_old), T_new if nm == "new" else T_old, np.inf), axis=1)
        d[f"front_{fr}_cause"] = dict(no_top=float((~has[front]).mean()),
                                      tl_lowest=float(((lowest == J) & has)[front].mean()),
                                      nonwm_lowest=float(((lowest == nonwm) & has)[front].mean()))
    out[nm] = d
for fr in (0.01, 0.05):
    a, b = rank_old < fr * m, rank_new < fr * m
    out[f"front_{fr}_overlap"] = float((a & b).sum() / a.sum())
    left = a & ~b
    out[f"front_{fr}_left_sn_p50_mm"] = float(np.nanmedian(sn[left])) if left.any() else None
    out[f"front_{fr}_left_res_p50_mm"] = float(np.nanmedian(np.abs(res[left]))) if left.any() else None
(OUT / "compare.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
