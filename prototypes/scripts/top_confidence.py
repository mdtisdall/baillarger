"""Column top confidence over the whole hemisphere, one global pass (no
superblock seams): the candidate tables of top_features.py, best-first
propagation by score on the global column graph, the final score with the
local thickness term and the opposite-bank term, and the review queue.
Terms as top_conf.py (provisional forms; scales from the data where possible).

args: FEAT_DIR OUT_DIR
"""
import glob
import heapq
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from top_stage_lib import gaussian_log_term, two_mode_log_prior  # noqa: E402

FEAT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
STEP = 0.04
SEED_RUN = 8
MIN_TOP_MM = 0.5
WINDOW_MM = 1.0
HBINS = np.r_[0, np.geomspace(1e-5, 10, 400)]
t0 = time.perf_counter()

V, PP, FI, RU, EV, HI, GB, FB, RC = [], [], [], [], [], [], [], [], []
CC = {k: [] for k in ("ci", "k", "lab", "S", "at_label", "far_wm", "mirror")}
base = 0
for f in sorted(glob.glob(str(FEAT / "feat_*.npz"))):
    z = np.load(f)
    EV.append(z["edges_v"])
    if "empty" in z.files:
        continue
    n = len(z["vert"])
    V.append(z["vert"]); PP.append(z["P"]); FI.append(z["first"].astype(np.int32)); RU.append(z["run"].astype(np.int32))
    HI.append(z["hist"]); GB.append(z["gm_behind"]); FB.append(z["fl_behind"]); RC.append(z["root_cls"])
    CC["ci"].append(z["c_ci"].astype(np.int64) + base)
    for k in ("k", "lab", "S", "at_label", "far_wm", "mirror"):
        CC[k].append(z["c_" + k])
    base += n
vert = np.concatenate(V)
P = np.concatenate(PP)
first = np.concatenate(FI)
run = np.concatenate(RU)
gm_behind = np.concatenate(GB)
fl_behind = np.concatenate(FB)
root_cls = np.concatenate(RC)
nonwm = gm_behind + fl_behind  # a root needs WM-like tissue behind it
m, ns, _ = P.shape
C = {k: np.concatenate(v) for k, v in CC.items()}
C["k"] = C["k"].astype(np.int64)
assert len(np.unique(vert)) == m, "a column is owned twice"
print(f"{m} columns, {len(C['k'])} candidates, read in {time.perf_counter() - t0:.0f} s", flush=True)

# the global column graph from the superblocks' triangles (vertex pairs)
order = np.argsort(vert)
vs_sorted = vert[order]
E = np.unique(np.sort(np.concatenate(EV), 1), axis=0)


def idx(v):
    j = np.clip(np.searchsorted(vs_sorted, v), 0, m - 1)
    return np.where(vs_sorted[j] == v, order[j], -1)


ia, ib = idx(E[:, 0]), idx(E[:, 1])
ok = (ia >= 0) & (ib >= 0)
ia, ib = ia[ok], ib[ok]
src = np.r_[ia, ib]
dst = np.r_[ib, ia]
o = np.argsort(src, kind="stable")
nb_ptr = np.r_[0, np.cumsum(np.bincount(src, minlength=m))]
nb_idx = dst[o]
print(f"{len(ia)} edges; columns with no neighbour: {int((np.diff(nb_ptr) == 0).sum())}", flush=True)

# global sheet floor (90th percentile of the cortex samples' score)
hist = np.sum(HI, 0)
cum = np.cumsum(hist) / hist.sum()
S_floor = float(HBINS[1:][np.searchsorted(cum, 0.9)])
kmin = int(round(MIN_TOP_MM / STEP))
clear = (first >= 0) & (run >= SEED_RUN)  # zero thickness included
grid = np.arange(ns) * STEP
bw = 0.2
tc_ = first[clear] * STEP
dens = np.histogram(tc_, np.r_[grid - STEP / 2, grid[-1] + STEP / 2])[0].astype(float)
kern = np.exp(-0.5 * (np.arange(-15, 16) * STEP / bw) ** 2)
dens = np.convolve(dens, kern, "same")
# two modes (near-0 and cortical): split at the trough between the two highest
# peaks; each side relative to its own peak, so neither mode penalises the other
logg, pk, trough = two_mode_log_prior(dens)
modes = (grid[pk] if len(pk) else grid[[np.argmax(dens)]]).round(2).tolist()
print(f"floor {S_floor:.5f}, clear label tops {clear.sum()}, modes {modes} mm, trough {grid[trough] if trough >= 0 else None} mm", flush=True)

# evidence of each candidate (terms independent of the neighbours)
lab = C["lab"]
keep = lab | (C["S"] >= S_floor)
for k in C:
    C[k] = C[k][keep]
lab = C["lab"]
ci, ck = C["ci"], C["k"]
t_glob = logg[ck]
t_lab = np.where(lab, np.where(run[ci] >= SEED_RUN, 1.0, 0.0), np.nan)
t_sheet = np.where(lab, np.nan, np.clip(np.log(np.maximum(C["S"], 1e-12) / S_floor), -1, 2))
atl = C["at_label"]
t_at = np.where(atl, 1.0, np.nan)
mr = C["mirror"]
t_mir = np.where(atl, np.nan, np.where(np.isfinite(mr), np.clip(2 * (mr - 0.3), -2, 1), -1.0))
t_fwm = np.where(atl, np.nan, np.where(C["far_wm"], 0.5, -0.5))
# non-WM signal behind the root (1.5 mm inward; GM-like or fluid-like): cortex the
# column does not account for, or a root not on tissue; a property of the column
excess = nonwm[ci]
ref = clear & (first * STEP > (grid[trough] if trough >= 0 else 0.0)) & np.isfinite(nonwm)
sigma_gm = max(float(np.sqrt(np.mean(nonwm[ref] ** 2))), STEP)
print(f"non-WM behind the root, clear cortical label tops: p50/90/99 {np.percentile(nonwm[ref], [50, 90, 99]).round(3).tolist()} mm, sigma {sigma_gm:.3f} mm", flush=True)
t_gm = gaussian_log_term(excess, sigma_gm)
TERMS = np.stack([t_glob, t_lab, t_at, t_sheet, t_mir, t_fwm, t_gm], 1)
ev = np.nansum(TERMS, 1)
co = np.argsort(ci, kind="stable")
for k in ("ci", "k", "lab"):
    C[k] = C[k][co]
ev, TERMS = ev[co], TERMS[co]
ci, ck, lab = C["ci"], C["k"], C["lab"]
c_ptr = np.r_[0, np.cumsum(np.bincount(ci, minlength=m))]
print(f"{len(ci)} candidates kept; evidence ready at {time.perf_counter() - t0:.0f} s", flush=True)

top_k = np.full(m, -1, np.int64)
top_c = np.full(m, -1, np.int64)  # the candidate row chosen


def tangent(c, k):
    a, b = P[c, max(k - 1, 0)], P[c, min(k + 1, ns - 1)]
    d = b - a
    n = float(np.sqrt(d @ d))
    return d / n if n > 0 else None


def prediction(b):
    preds = []
    for c in nb_idx[nb_ptr[b]:nb_ptr[b + 1]]:
        k = top_k[c]
        if k < 0:
            continue
        tc = tangent(c, k)
        if tc is None:
            continue
        h = (P[b] - P[c, k]) @ tc
        sc = np.flatnonzero(np.signbit(h[1:]) != np.signbit(h[:-1]))
        if len(sc):
            preds.append(sc[0])
        elif h[0] >= 0:  # the neighbour's top plane lies at or behind this root: zero thickness
            preds.append(0)
    return int(np.median(preds)) if preds else None


# local sigma: the same quantity the local term uses (the along-ray residual
# to the plane of the neighbours' tops), measured among clear label tops
top_k[clear] = first[clear]
rng = np.random.default_rng(0)
cs = np.flatnonzero(clear)
cs = rng.choice(cs, min(len(cs), 50000), replace=False)
res = []
for b_ in cs:
    k_ = top_k[b_]
    top_k[b_] = -1
    kp_ = prediction(b_)
    top_k[b_] = k_
    if kp_ is not None:
        res.append((k_ - kp_) * STEP)
res = np.array(res)
sigma_loc = max(1.4826 * float(np.median(np.abs(res - np.median(res)))), STEP)
top_k[:] = -1
print(f"local sigma {sigma_loc:.3f} mm from {len(res)} clear label tops (residual median {np.median(res):+.3f} mm)", flush=True)
heap = []
for r in np.flatnonzero(lab):
    heap.append((-float(ev[r]), int(ci[r]), int(r)))
heapq.heapify(heap)
w = int(round(WINDOW_MM / STEP))


def offer(b):
    kp = prediction(b)
    if kp is None:
        return
    rows = np.arange(c_ptr[b], c_ptr[b + 1])
    if not len(rows):
        return
    kk = ck[rows]
    sel = np.abs(kk - kp) <= w
    for r, k in zip(rows[sel], kk[sel]):
        sc = ev[r] - 0.5 * (((k - kp) * STEP) / sigma_loc) ** 2
        heapq.heappush(heap, (-float(sc), int(b), int(r)))


nf = 0
tl = time.perf_counter()
while heap:
    _, b, r = heapq.heappop(heap)
    if top_k[b] >= 0:
        continue
    top_k[b], top_c[b] = ck[r], r
    nf += 1
    for c in nb_idx[nb_ptr[b]:nb_ptr[b + 1]]:
        if top_k[c] < 0:
            offer(c)
    if nf % 100000 == 0:
        print(f"  fixed {nf} ({time.perf_counter() - tl:.0f} s, heap {len(heap)})", flush=True)
print(f"propagation: {nf} tops in {time.perf_counter() - tl:.0f} s", flush=True)

# final score: the candidate's evidence, the local term against the final
# neighbours, and the opposite-bank term
has = top_k >= 0
names = ["thick_global", "label_clear", "at_label", "sheet", "mirror", "far_wm", "non_wm_behind_root", "thick_local", "bank"]
T = np.full((m, len(names)), np.nan, np.float32)
T[has, :7] = TERMS[top_c[has]]
for b in np.flatnonzero(has):
    k = top_k[b]
    top_k[b] = -1
    kp = prediction(b)
    top_k[b] = k
    T[b, 7] = max(-0.5 * (((k - kp) * STEP) / sigma_loc) ** 2, -8.0) if kp is not None else 0.0
print(f"local term at {time.perf_counter() - t0:.0f} s", flush=True)
hid = np.flatnonzero(has)
Xt = P[hid, top_k[hid]].astype(np.float64)
Tt = np.zeros((len(hid), 3))
for j, b in enumerate(hid):
    t_ = tangent(b, top_k[b])
    Tt[j] = t_ if t_ is not None else 0
bank = np.zeros(len(hid))
tree = cKDTree(Xt)
for s0 in range(0, len(hid), 200000):
    L = tree.query_ball_point(Xt[s0:s0 + 200000], 0.5, workers=-1)
    for j, near in enumerate(L):
        near = np.asarray(near)
        if len(near) and (Tt[near] @ Tt[s0 + j] < -0.3).any():
            bank[s0 + j] = 0.5
# disagreement: an opposite ray passes within 0.2 mm of this top and its own top is > 0.5 mm further on
SUB = 4
Pq = P[:, ::SUB].astype(np.float64)
ksub = np.arange(0, ns, SUB)
rtree = cKDTree(Pq.reshape(-1, 3))
nsub = Pq.shape[1]
for s0 in range(0, len(hid), 200000):
    L = rtree.query_ball_point(Xt[s0:s0 + 200000], 0.2, workers=-1)
    for j, hits in enumerate(L):
        if not hits:
            continue
        hits = np.asarray(hits)
        c_, kq = hits // nsub, ksub[hits % nsub]
        tk = top_k[c_]
        bad = (tk >= 0) & (tk > kq + int(0.5 / STEP))
        if not bad.any():
            continue
        dirs = P[c_[bad], np.minimum(kq[bad] + 1, ns - 1)] - P[c_[bad], np.maximum(kq[bad] - 1, 0)]
        if (dirs @ Tt[s0 + j] < -0.3 * np.linalg.norm(dirs, axis=1)).any():
            bank[s0 + j] = -2.0
T[hid, 8] = bank
conf = np.where(has, np.nansum(T, 1), -np.inf)
rank = np.empty(m, np.int64)
rank[np.argsort(conf, kind="stable")] = np.arange(m)
top_mm = np.where(has, top_k * STEP, np.nan)
summary = {"columns": int(m), "with_top": int(has.sum()), "no_top": int((~has).sum()),
           "from_label": int((has & lab[np.maximum(top_c, 0)]).sum()),
           "conf_p10_25_50_75_90": np.percentile(conf[has], [10, 25, 50, 75, 90]).round(2).tolist(),
           "S_floor": S_floor, "sigma_local_mm": sigma_loc, "modes_mm": modes, "trough_mm": float(grid[trough]) if trough >= 0 else None, "sigma_gm_mm": sigma_gm,
           "zero_tops": int((has & (top_k <= 1)).sum()),
           "tops_over_4.5mm": int((top_mm > 4.5).sum()), "conf_median_tops_le_4.5": float(np.median(conf[has & (top_mm <= 4.5)])),
           "conf_median_tops_gt_4.5": float(np.median(conf[has & (top_mm > 4.5)])) if (top_mm > 4.5).any() else None,
           "bank_disagree": int((bank < 0).sum()), "seconds": time.perf_counter() - t0}
print(json.dumps(summary, indent=1), flush=True)
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
np.savez(OUT / "top_confidence.npz", vert=vert, root=P[:, 0], top_k=top_k, top_mm=top_mm, is_lab=lab[np.maximum(top_c, 0)] & has,
         conf=conf, terms=T, gm_behind=gm_behind, fl_behind=fl_behind, root_cls=root_cls, term_names=np.array(names), rank=rank, first=first)
print("done", time.perf_counter() - t0, flush=True)
