"""Prototype: column tops as one joint problem, solved by parallel soft updates (mean field), against the
current best-first propagation, on the same candidates and the same evidence, in one region.

Evidence per candidate: exactly top_confidence.py's (thick_global, label_clear, at_label, sheet, mirror,
far_wm, non_wm_behind_root), from all columns' statistics. Agreement between column b's candidate i and
neighbour c's candidate j: -0.5 ((k_i - x)*STEP / s_b)^2, capped at -8, where x is where b's ray crosses
c's top plane at j (the plane through c's candidate, normal = c's ray tangent there), as the stage's
prediction; s_b = sqrt(s0^2 + sn_b^2) (sn: the per-column spread from the tl-spread prototype).
Mean field: q_b ~ exp(ev_bi + lam * sum_c sum_j q_cj phi(bi, cj)), all columns at once, damped; the
agreement width starts wide (factor F0) and narrows to 1 (coarse to fine). lam = 1 / (neighbours of b),
so agreement is the mean over neighbours, as the stage's median is one prediction.
Exact test: trusted columns (clear label top chosen by best-first, another candidate >= 0.5 mm away); a
random 5 % get a spurious strongest label candidate there (ev = own max + 2, lab = True). Both solvers
run on the perturbed evidence: restored = top within 1 step of the true; broken = an unperturbed trusted
column moves more than 1 step from its own unperturbed solution.
args: FEAT_DIR CONF_DIR SN_NPY OUT_DIR SUPERBLOCK[,SUPERBLOCK...]"""
import glob
import heapq
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from top_stage_lib import gaussian_log_term, two_mode_log_prior  # noqa: E402

FEAT, CDIR, SN, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
REGION = sys.argv[5].split(",")
OUT.mkdir(parents=True, exist_ok=True)
STEP, SEED_RUN, MIN_TOP_MM, WINDOW_MM = 0.04, 8, 0.5, 1.0
HBINS = np.r_[0, np.geomspace(1e-5, 10, 400)]
t0 = time.perf_counter()

# ---------------------------------------------------------------- inputs, as top_confidence.py
V, PP, FI, RU, EV, HI, GB, FB, SBF = [], [], [], [], [], [], [], [], []
CC = {k: [] for k in ("ci", "k", "lab", "S", "at_label", "far_wm", "mirror")}
base = 0
for f in sorted(glob.glob(str(FEAT / "feat_*.npz"))):
    z = np.load(f)
    EV.append(z["edges_v"])
    if "empty" in z.files:
        continue
    n = len(z["vert"])
    V.append(z["vert"]); PP.append(z["P"]); FI.append(z["first"].astype(np.int32)); RU.append(z["run"].astype(np.int32))
    HI.append(z["hist"]); GB.append(z["gm_behind"]); FB.append(z["fl_behind"])
    SBF.append(np.full(n, Path(f).stem[5:] in REGION))
    CC["ci"].append(z["c_ci"].astype(np.int64) + base)
    for k in ("k", "lab", "S", "at_label", "far_wm", "mirror"):
        CC[k].append(z["c_" + k])
    base += n
vert = np.concatenate(V); P = np.concatenate(PP); first = np.concatenate(FI); run = np.concatenate(RU)
nonwm = np.concatenate(GB) + np.concatenate(FB); inreg = np.concatenate(SBF)
m, ns, _ = P.shape
C = {k: np.concatenate(v) for k, v in CC.items()}; C["k"] = C["k"].astype(np.int64)
hist = np.sum(HI, 0); cum = np.cumsum(hist) / hist.sum(); S_floor = float(HBINS[1:][np.searchsorted(cum, 0.9)])
clear = (first >= 0) & (run >= SEED_RUN)
grid = np.arange(ns) * STEP
dens = np.histogram(first[clear] * STEP, np.r_[grid - STEP / 2, grid[-1] + STEP / 2])[0].astype(float)
dens = np.convolve(dens, np.exp(-0.5 * (np.arange(-15, 16) * STEP / 0.2) ** 2), "same")
logg, pk, trough = two_mode_log_prior(dens)
lab = C["lab"]; keep = lab | (C["S"] >= S_floor)
for k in C:
    C[k] = C[k][keep]
lab, ci, ck = C["lab"], C["ci"], C["k"]
t_glob = logg[ck]
t_lab = np.where(lab, np.where(run[ci] >= SEED_RUN, 1.0, 0.0), np.nan)
t_sheet = np.where(lab, np.nan, np.clip(np.log(np.maximum(C["S"], 1e-12) / S_floor), -1, 2))
atl = C["at_label"]; t_at = np.where(atl, 1.0, np.nan)
mr = C["mirror"]; t_mir = np.where(atl, np.nan, np.where(np.isfinite(mr), np.clip(2 * (mr - 0.3), -2, 1), -1.0))
t_fwm = np.where(atl, np.nan, np.where(C["far_wm"], 0.5, -0.5))
ref = clear & (first * STEP > (grid[trough] if trough >= 0 else 0.0)) & np.isfinite(nonwm)
sigma_gm = max(float(np.sqrt(np.mean(nonwm[ref] ** 2))), STEP)
t_gm = gaussian_log_term(nonwm[ci], sigma_gm)
ev_all = np.nansum(np.stack([t_glob, t_lab, t_at, t_sheet, t_mir, t_fwm, t_gm], 1), 1)
S0 = float(json.loads((CDIR / "summary.json").read_text())["sigma_local_mm"])
sn_all = np.load(SN)
print(f"{m} columns, evidence ready at {time.perf_counter() - t0:.0f} s; s0 {S0:.4f}", flush=True)

# ---------------------------------------------------------------- the region
R = np.flatnonzero(inreg); nR = len(R)
loc = -np.ones(m, np.int64); loc[R] = np.arange(nR)
order = np.argsort(vert); vs = vert[order]


def idx(v):
    j = np.clip(np.searchsorted(vs, v), 0, m - 1)
    return np.where(vs[j] == v, order[j], -1)


E = np.unique(np.sort(np.concatenate(EV), 1), axis=0)
ia, ib = idx(E[:, 0]), idx(E[:, 1]); ok = (ia >= 0) & (ib >= 0) & inreg[ia] & inreg[ib] & (ia != ib)
a, b = loc[ia[ok]], loc[ib[ok]]
src, dst = np.r_[a, b], np.r_[b, a]; o = np.argsort(src, kind="stable")
nb_ptr = np.r_[0, np.cumsum(np.bincount(src, minlength=nR))]; nb_idx = dst[o]
cm = inreg[ci]
cand_col = loc[ci[cm]]; cand_k = ck[cm]; cand_lab = lab[cm].copy(); ev0 = ev_all[cm].copy()
co = np.argsort(cand_col, kind="stable"); cand_col, cand_k, cand_lab, ev0 = cand_col[co], cand_k[co], cand_lab[co], ev0[co]
PR = P[R]
# a ray stops early and is padded with its end point: its end is after the last step of non-zero length
_st = np.linalg.norm(np.diff(PR, axis=1), axis=2) > 1e-6
ray_end = np.where(_st.any(1), _st.shape[1] - np.argmax(_st[:, ::-1], 1), 0).astype(np.int64)
_k = cand_k <= ray_end[cand_col]
print(f"candidates beyond their ray's end (padding copies of the end point), dropped: {(~_k).mean():.1%}", flush=True)
cand_col, cand_k, cand_lab, ev0 = cand_col[_k], cand_k[_k], cand_lab[_k], ev0[_k]
c_ptr = np.r_[0, np.cumsum(np.bincount(cand_col, minlength=nR))]
nc = len(cand_col); s_b = np.sqrt(S0 ** 2 + np.nan_to_num(sn_all[R]) ** 2)
print(f"region {REGION}: {nR} columns, {len(a)} edges, {nc} candidates", flush=True)


def tang(Pc, k):
    lo, hi = np.maximum(k - 1, 0), np.minimum(k + 1, ns - 1)
    d = Pc[np.arange(len(k)), hi] - Pc[np.arange(len(k)), lo]
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return d / np.where(n > 0, n, 1), n[:, 0] > 0


# ---------------------------------------------------------------- crossing table: b, c, candidate j of c
rows_b, rows_j = [], []
for bb in range(nR):
    for cc in nb_idx[nb_ptr[bb]:nb_ptr[bb + 1]]:
        js = np.arange(c_ptr[cc], c_ptr[cc + 1])
        rows_b.append(np.full(len(js), bb)); rows_j.append(js)
rb = np.concatenate(rows_b); rj = np.concatenate(rows_j)
xr = np.full(len(rb), np.nan, np.float32)
for s0 in range(0, len(rb), 100000):
    bsl, jsl = rb[s0:s0 + 100000], rj[s0:s0 + 100000]
    cc = cand_col[jsl]; kc = cand_k[jsl]
    X = PR[cc, kc]; t, okt = tang(PR[cc], kc)
    h = np.einsum("nsd,nd->ns", PR[bsl] - X[:, None, :], t)
    sgn = np.signbit(h)
    ch = sgn[:, 1:] != sgn[:, :-1]
    anyc = ch.any(1)
    eb = ray_end[bsl]
    beyond = (eb.astype(np.float64) + (-h[np.arange(len(bsl)), eb]) / STEP)  # the plane past b's ray end: the shortfall, along the ray
    x = np.where(anyc, ch.argmax(1), np.where(h[:, 0] >= 0, 0, beyond)).astype(np.float32)
    x[(x < 0) | ~okt] = np.nan
    xr[s0:s0 + 100000] = x
print(f"crossing table: {len(rb)} rows, {np.isnan(xr).mean():.2%} without a crossing, at {time.perf_counter() - t0:.0f} s", flush=True)
# expansion: each candidate i of b against each crossing row of b
ei, er = [], []
r_ptr = np.r_[0, np.cumsum(np.bincount(rb, minlength=nR))]
for bb in range(nR):
    ii = np.arange(c_ptr[bb], c_ptr[bb + 1]); rr = np.arange(r_ptr[bb], r_ptr[bb + 1])
    if len(ii) and len(rr):
        ei.append(np.repeat(ii, len(rr))); er.append(np.tile(rr, len(ii)))
ei = np.concatenate(ei); er = np.concatenate(er)
good = np.isfinite(xr[er]); ei, er = ei[good], er[good]
dk = ((cand_k[ei] - xr[er]) * STEP / s_b[cand_col[ei]]).astype(np.float32)
ej = rj[er]
deg = np.maximum(np.diff(nb_ptr), 1).astype(np.float32)
print(f"pair terms: {len(ei)}, at {time.perf_counter() - t0:.0f} s", flush=True)


def softmax_seg(e):
    mx = np.full(nR, -np.inf); np.maximum.at(mx, cand_col, e)
    w = np.exp(e - mx[cand_col]); s = np.bincount(cand_col, w, nR)
    return w / s[cand_col]


def mean_field(ev, iters=60, F0=4.0, narrow=30, damp=0.5):
    q = softmax_seg(ev); hist_ = []
    for it in range(iters):
        F = F0 ** max(0.0, 1 - it / narrow)
        phi = np.maximum(-0.5 * (dk / F) ** 2, -8.0)
        agree = np.bincount(ei, q[ej] * phi, nc) / deg[cand_col]
        qn = softmax_seg(ev + agree)
        dq = float(np.abs(qn - q).max())
        q = damp * q + (1 - damp) * qn
        hist_.append(dict(it=it, F=F, max_dq=dq))
    # final: no damping step, report
    phi = np.maximum(-0.5 * dk ** 2, -8.0)
    agree = np.bincount(ei, q[ej] * phi, nc) / deg[cand_col]
    return q, agree, hist_


def choose(q):
    best = np.full(nR, -1, np.int64); bq = np.full(nR, -1.0)
    for bb in range(nR):
        s, e = c_ptr[bb], c_ptr[bb + 1]
        if e > s:
            i = s + int(np.argmax(q[s:e])); best[bb], bq[bb] = cand_k[i], q[i]
    return best, bq


def best_first(ev, labm):
    top_k = np.full(nR, -1, np.int64)

    def prediction(bb):
        preds = []
        for cc in nb_idx[nb_ptr[bb]:nb_ptr[bb + 1]]:
            k = top_k[cc]
            if k < 0:
                continue
            t, okt = tang(PR[[cc]], np.array([k]))
            if not okt[0]:
                continue
            h = (PR[bb] - PR[cc, k]) @ t[0]
            sc = np.flatnonzero(np.signbit(h[1:]) != np.signbit(h[:-1]))
            if len(sc):
                preds.append(sc[0])
            elif h[0] >= 0:
                preds.append(0)
        return int(np.median(preds)) if preds else None

    heap = [(-float(ev[r]), int(cand_col[r]), int(r)) for r in np.flatnonzero(labm)]
    heapq.heapify(heap)
    w = int(round(WINDOW_MM / STEP))
    while heap:
        _, bb, r = heapq.heappop(heap)
        if top_k[bb] >= 0:
            continue
        top_k[bb] = cand_k[r]
        for cc in nb_idx[nb_ptr[bb]:nb_ptr[bb + 1]]:
            if top_k[cc] >= 0:
                continue
            kp = prediction(cc)
            if kp is None:
                continue
            for r2 in range(c_ptr[cc], c_ptr[cc + 1]):
                if abs(cand_k[r2] - kp) <= w:
                    heapq.heappush(heap, (-float(ev[r2] - 0.5 * (((cand_k[r2] - kp) * STEP) / S0) ** 2), int(cc), int(r2)))
    return top_k



# ---------------------------------------------------------------- diagnosis of the old (one-sided, all-at-once) solver
def old_flips(ev, iters=60, F0=4.0, narrow=30, damp=0.5):
    q = softmax_seg(ev); prev = None; flips = []
    for it in range(iters):
        F = F0 ** max(0.0, 1 - it / narrow)
        phi = np.maximum(-0.5 * (dk / F) ** 2, -8.0)
        agree = np.bincount(ei, q[ej] * phi, nc) / deg[cand_col]
        q = damp * q + (1 - damp) * softmax_seg(ev + agree)
        ch, _ = choose(q)
        if prev is not None:
            flips.append(np.flatnonzero(ch != prev))
        prev = ch
    return flips


# ---------------------------------------------------------------- the symmetric energy
# one term per neighbour pair and candidate pair: the mean of the two one-sided measures that exist
lo, hi = np.minimum(ei, ej), np.maximum(ei, ej)
ukey, uinv, ucnt = np.unique(lo.astype(np.int64) * nc + hi, return_inverse=True, return_counts=True)
u_lo, u_hi = (ukey // nc).astype(np.int64), (ukey % nc).astype(np.int64)
u_w = (2.0 / (deg[cand_col[u_lo]] + deg[cand_col[u_hi]])).astype(np.float64)
print(f"symmetric pair terms: {len(ukey)} (both directions measured for {np.mean(ucnt == 2):.0%})", flush=True)


def phi_sym(F):
    return np.bincount(uinv, np.maximum(-0.5 * (dk / F) ** 2, -8.0), len(ukey)) / ucnt


def agree_of(q, ph):
    a = u_w * ph
    return np.bincount(u_lo, a * q[u_hi], nc) + np.bincount(u_hi, a * q[u_lo], nc)


def objective(q, ev, ph):
    ql = np.where(q > 0, q * np.log(np.maximum(q, 1e-300)), 0.0)
    return float(-(q * ev).sum() - (u_w * ph * q[u_lo] * q[u_hi]).sum() + ql.sum())


# greedy colouring of the column graph: no two neighbours share a colour
colour = np.full(nR, -1, np.int64)
for bb in np.argsort(-np.diff(nb_ptr), kind="stable"):
    used = set(colour[nb_idx[nb_ptr[bb]:nb_ptr[bb + 1]]].tolist())
    cl = 0
    while cl in used:
        cl += 1
    colour[bb] = cl
ncol = int(colour.max()) + 1
cand_colour = colour[cand_col]
print(f"colours: {ncol}", flush=True)


def softmax_cols(e, sel_c):
    """softmax over the candidates of the columns whose candidates are sel_c (a mask)"""
    out = np.zeros(nc)
    ii = np.flatnonzero(sel_c)
    cc = cand_col[ii]
    mx = np.full(nR, -np.inf); np.maximum.at(mx, cc, e[ii])
    w = np.exp(e[ii] - mx[cc]); s = np.bincount(cc, w, nR)
    out[ii] = w / s[cc]
    return out


def coloured_mf(ev, levels=(4.0, 2 ** 1.5, 2.0, 2 ** 0.5, 1.0), max_sweeps=60, rtol=1e-7):
    q = softmax_seg(ev); log = []; prev = None; rises = 0
    for F in levels:
        ph = phi_sym(F)
        f_old = objective(q, ev, ph)
        for sw in range(max_sweeps):
            for cl in range(ncol):
                sel = cand_colour == cl
                a = agree_of(q, ph)
                qn = softmax_cols(ev + a, sel)
                q = np.where(sel, qn, q)
            f_new = objective(q, ev, ph)
            ch, _ = choose(q)
            nfl = int((ch != prev).sum()) if prev is not None else -1
            prev = ch
            if f_new > f_old + 1e-6 * abs(f_old):
                rises += 1
            log.append(dict(F=round(F, 3), sweep=sw, objective=f_new, flips=nfl))
            if abs(f_old - f_new) <= rtol * abs(f_old):
                break
            f_old = f_new
    return q, log, rises

res = {"region": REGION, "columns": int(nR), "candidates": int(nc)}
tb = time.perf_counter(); bf0 = best_first(ev0, cand_lab); res["best_first_s"] = time.perf_counter() - tb
q0 = softmax_seg(ev0); h0 = [dict(max_dq=0.0)] * 5; mf0, mq0 = choose(q0); res['mean_field_s'] = 0.0
res["mf_convergence_last5_max_dq"] = [round(h["max_dq"], 5) for h in h0[-5:]]
print(f"best-first {res['best_first_s']:.0f} s; mean field {res['mean_field_s']:.0f} s; last max dq {h0[-1]['max_dq']:.2e}", flush=True)
both = (bf0 >= 0) & (mf0 >= 0)
res["same_top_within_1_step"] = float(np.mean(np.abs(bf0[both] - mf0[both]) <= 1))
res["no_top_best_first"] = int((bf0 < 0).sum()); res["no_top_mean_field"] = int((mf0 < 0).sum())
res["top_mm_p50_bf_mf"] = [float(np.median(bf0[bf0 >= 0]) * STEP), float(np.median(mf0[mf0 >= 0]) * STEP)]
res["share_top_gt_4.5mm_bf_mf"] = [float(np.mean(bf0[bf0 >= 0] * STEP > 4.5)), float(np.mean(mf0[mf0 >= 0] * STEP > 4.5))]
# own evidence overridden: chosen candidate's ev below the column's best ev
evmax = np.full(nR, -np.inf); np.maximum.at(evmax, cand_col, ev0)
ev_mf = np.full(nR, np.nan)
for bb in np.flatnonzero(mf0 >= 0):
    s, e = c_ptr[bb], c_ptr[bb + 1]; ev_mf[bb] = ev0[s + int(np.argmax(q0[s:e]))]
res["mf_overrides_own_best_by_gt_1"] = float(np.nanmean(evmax - ev_mf > 1))
res["mf_max_q_p10_p50"] = [float(np.percentile(mq0[mq0 >= 0], 10)), float(np.median(mq0[mq0 >= 0]))]
fl = [np.zeros(0, np.int64)] * 10
last = np.unique(np.concatenate(fl[-10:]))
res["old_flipping_columns_last10"] = int(len(last))
res["old_flips_per_round_last10"] = [int(len(f)) for f in fl[-10:]]
tc = time.perf_counter(); qc, logc, rises = coloured_mf(ev0); mc0, mqc0 = choose(qc); res["coloured_s"] = time.perf_counter() - tc
res["coloured_objective_rises"] = rises
res["coloured_sweeps"] = len(logc)
res["coloured_last"] = logc[-1]
res["coloured_per_level_last"] = {str(l["F"]): dict(sweeps=l["sweep"] + 1, flips=l["flips"]) for l in logc}
bothc = (bf0 >= 0) & (mc0 >= 0)
res["coloured_same_as_bf_within_1_step"] = float(np.mean(np.abs(bf0[bothc] - mc0[bothc]) <= 1))
res["coloured_no_top"] = int((mc0 < 0).sum())
res["coloured_max_q_p10_p50"] = [float(np.percentile(mqc0[mqc0 >= 0], 10)), float(np.median(mqc0[mqc0 >= 0]))]
res["coloured_share_top_gt_4.5mm"] = float(np.mean(mc0[mc0 >= 0] * STEP > 4.5))
json.dump(logc, open(OUT / "coloured_log.json", "w"), indent=1)
print(f"old: {len(last)} columns flipped in the last 10 rounds; coloured: {len(logc)} sweeps, objective rises {rises}, at {time.perf_counter() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------- exact test
rng = np.random.default_rng(0)
labk = np.full(nR, -1, np.int64)
for bb in range(nR):
    s, e = c_ptr[bb], c_ptr[bb + 1]
    L = np.flatnonzero(cand_lab[s:e])
    if len(L):
        labk[bb] = cand_k[s + L[np.argmax(ev0[s + L])]]
trusted = (bf0 >= 0) & (bf0 == labk) & clear[R]
alt = np.full(nR, -1, np.int64)
for bb in np.flatnonzero(trusted):
    s, e = c_ptr[bb], c_ptr[bb + 1]
    far = np.flatnonzero(np.abs(cand_k[s:e] - bf0[bb]) * STEP >= 0.5)
    if len(far):
        alt[bb] = s + int(rng.choice(far))
elig = np.flatnonzero(alt >= 0)
pert = rng.choice(elig, max(1, len(elig) // 20), replace=False)
ev1, lab1 = ev0.copy(), cand_lab.copy()
for bb in pert:
    s, e = c_ptr[bb], c_ptr[bb + 1]
    ev1[alt[bb]] = ev0[s:e].max() + 2.0; lab1[alt[bb]] = True
bf1 = bf0.copy()
q1 = q0; h1 = h0; mf1 = mf0
qc1, logc1, rises1 = coloured_mf(ev1); mc1, mqc1 = choose(qc1)
ph1 = phi_sym(1.0); ag1 = agree_of(qc1, ph1)
chosen = np.full(nR, -1, np.int64)
for bb in range(nR):
    s_, e_ = c_ptr[bb], c_ptr[bb + 1]
    if e_ > s_:
        chosen[bb] = s_ + int(np.argmax(qc1[s_:e_]))
has1 = chosen >= 0
dis = np.full(nR, np.nan); dis[has1] = ag1[chosen[has1]]          # neighbour agreement at the chosen top (<= 0)
evmax1 = np.full(nR, -np.inf); np.maximum.at(evmax1, cand_col, ev1)
conf1 = np.full(nR, -np.inf); conf1[has1] = ev1[chosen[has1]] + ag1[chosen[has1]]
def pct(score, idx_):  # share of columns with a top ranked at or before each (lower score = earlier)
    sc = score[has1]; o = np.sort(sc)
    return np.searchsorted(o, score[idx_], side="right") / len(sc)
missed = pert[np.abs(mc1[pert] - bf0[pert]) > 1]; fixed = pert[np.abs(mc1[pert] - bf0[pert]) <= 1]
sc_list = {"max probability": mqc1, "neighbour disagreement": dis, "own evidence + agreement": conf1}
rv = {}
for nm_, sc in sc_list.items():
    pm = pct(sc, missed)
    rv[nm_] = dict(missed_n=int(len(missed)), missed_pct_p50=float(np.median(pm)), missed_in_first_1pct=float(np.mean(pm <= 0.01)),
                   missed_in_first_5pct=float(np.mean(pm <= 0.05)), missed_in_first_10pct=float(np.mean(pm <= 0.10)))
rv["missed_took_the_fake"] = float(np.mean(np.abs(mc1[missed] - cand_k[alt[missed]]) <= 1))
rv["missed_max_q_p50"] = float(np.median(mqc1[missed])); rv["all_max_q_p50"] = float(np.median(mqc1[has1]))
rv["missed_dis_p50"] = float(np.median(dis[missed])); rv["all_dis_p50"] = float(np.nanmedian(dis[has1]))
res["review_of_missed"] = rv
print(json.dumps(rv, indent=1), flush=True)
others = np.setdiff1d(np.flatnonzero(trusted), pert)
res["test"] = dict(trusted=int(trusted.sum()), perturbed=int(len(pert)),
                   restored_bf=float(np.mean(np.abs(bf1[pert] - bf0[pert]) <= 1)),
                   restored_mf=float(np.mean(np.abs(mf1[pert] - bf0[pert]) <= 1)),
                   broken_bf=float(np.mean(np.abs(bf1[others] - bf0[others]) > 1)),
                   broken_mf=float(np.mean(np.abs(mf1[others] - mf0[others]) > 1)),
                   mf_last_max_dq=h1[-1]["max_dq"],
                   restored_coloured=float(np.mean(np.abs(mc1[pert] - bf0[pert]) <= 1)),
                   broken_coloured=float(np.mean(np.abs(mc1[others] - mc0[others]) > 1)),
                   coloured_rises=rises1, coloured_sweeps=len(logc1))
print(json.dumps(res, indent=1), flush=True)
(OUT / "joint_tops.json").write_text(json.dumps(res, indent=1))
np.savez(OUT / "joint_tops.npz", R=R, mc0=mc0, mqc0=mqc0, mc1=mc1, bf0=bf0, mf0=mf0, mq0=mq0, bf1=bf1, mf1=mf1, pert=pert, conv0=np.array([h["max_dq"] for h in h0]))
