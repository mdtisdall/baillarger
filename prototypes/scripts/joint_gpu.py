"""The joint top solver (symmetric pair energy, colour-group mean field, coarse to fine) in PyTorch, for
the whole hemisphere or a region. Same evidence and pair terms as joint_tops.py: padded candidates
(beyond the ray's end) dropped; a neighbour plane past the ray's end scored by the shortfall.
Writes per column: chosen top (k), its probability, agreement at the chosen top, the non-WM-behind-root
term, and whether the column has a top.
args: FEAT_DIR CONF_DIR SN_NPY OUT_DIR DEVICE [SUPERBLOCK,...]"""
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from top_stage_lib import gaussian_log_term, two_mode_log_prior  # noqa: E402

FEAT, CDIR, SN, OUT, DEV = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5]
REGION = sys.argv[6].split(",") if len(sys.argv) > 6 else None
OUT.mkdir(parents=True, exist_ok=True)
dev = torch.device(DEV)
STEP, SEED_RUN = 0.04, 8
HBINS = np.r_[0, np.geomspace(1e-5, 10, 400)]
t0 = time.perf_counter()


def tick(msg):
    print(f"[{time.perf_counter() - t0:6.0f} s] {msg}", flush=True)


# ---------------------------------------------------------------- inputs and evidence (as top_confidence.py)
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
    SBF.append(np.full(n, REGION is None or Path(f).stem[5:] in REGION))
    CC["ci"].append(z["c_ci"].astype(np.int64) + base)
    for k in ("k", "lab", "S", "at_label", "far_wm", "mirror"):
        CC[k].append(z["c_" + k])
    base += n
vert = np.concatenate(V); P = np.concatenate(PP); del PP
first = np.concatenate(FI); run = np.concatenate(RU)
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
col_gm = gaussian_log_term(nonwm, sigma_gm)  # per column: the same for all its candidates
ev_all = np.nansum(np.stack([t_glob, t_lab, t_at, t_sheet, t_mir, t_fwm, col_gm[ci]], 1), 1)
S0 = float(json.loads((CDIR / "summary.json").read_text())["sigma_local_mm"])
sn_all = np.load(SN)
tick(f"{m} columns, {len(ci)} candidates; evidence ready")

# ---------------------------------------------------------------- region, graph, candidates
R = np.flatnonzero(inreg); nR = len(R)
loc = -np.ones(m, np.int64); loc[R] = np.arange(nR)
order = np.argsort(vert); vs = vert[order]
E = np.unique(np.sort(np.concatenate(EV), 1), axis=0)
j_ = np.clip(np.searchsorted(vs, E), 0, m - 1); ok = (vs[j_] == E).all(1)
G = order[j_[ok]]; G = G[(G[:, 0] != G[:, 1]) & inreg[G[:, 0]] & inreg[G[:, 1]]]
a, b = loc[G[:, 0]], loc[G[:, 1]]
src, dst = np.r_[a, b], np.r_[b, a]; o = np.argsort(src, kind="stable"); src, dst = src[o], dst[o]
nb_ptr = np.r_[0, np.cumsum(np.bincount(src, minlength=nR))]
deg = np.maximum(np.diff(nb_ptr), 1).astype(np.float64)
PR = P[R]; del P
st = np.linalg.norm(np.diff(PR, axis=1), axis=2) > 1e-6
ray_end = np.where(st.any(1), st.shape[1] - np.argmax(st[:, ::-1], 1), 0).astype(np.int64); del st
cm = inreg[ci]
cand_col = loc[ci[cm]]; cand_k = ck[cm]; ev0 = ev_all[cm]
kk = cand_k <= ray_end[cand_col]
tick(f"region: {nR} columns, {len(a)} edges; padded candidates dropped: {(~kk).mean():.1%}")
cand_col, cand_k, ev0 = cand_col[kk], cand_k[kk], ev0[kk]
co = np.argsort(cand_col, kind="stable"); cand_col, cand_k, ev0 = cand_col[co], cand_k[co], ev0[co]
ncand = np.bincount(cand_col, minlength=nR); c_ptr = np.r_[0, np.cumsum(ncand)]
nc = len(cand_col)
s_b = np.sqrt(S0 ** 2 + np.nan_to_num(sn_all[R]) ** 2)


def seg_repeat(starts, counts):
    """concatenated aranges: starts[i] + 0..counts[i]-1"""
    tot = counts.sum()
    off = np.repeat(np.cumsum(counts) - counts, counts)
    return np.repeat(starts, counts) + (np.arange(tot) - off)


# crossing rows: directed edge (b, c) x candidate j of c
cnt = ncand[dst]
rb = np.repeat(src, cnt); rj = seg_repeat(c_ptr[dst], cnt)
tick(f"crossing rows: {len(rb)}")
PRt = torch.from_numpy(PR).to(dev)
end_t = torch.from_numpy(ray_end).to(dev)
ck_t = torch.from_numpy(cand_k).to(dev); cc_t = torch.from_numpy(cand_col).to(dev)
xr = torch.empty(len(rb), dtype=torch.float32, device=dev)
B = 200000
ar = torch.arange(ns - 1, device=dev)
for s0 in range(0, len(rb), B):
    bsl = torch.from_numpy(rb[s0:s0 + B]).to(dev); jsl = torch.from_numpy(rj[s0:s0 + B]).to(dev)
    c = cc_t[jsl]; kc = ck_t[jsl]
    X = PRt[c, kc]
    t = PRt[c, torch.clamp(kc + 1, max=ns - 1)] - PRt[c, torch.clamp(kc - 1, min=0)]
    t = t / torch.clamp(t.norm(dim=1, keepdim=True), min=1e-12)
    h = torch.einsum("nsd,nd->ns", PRt[bsl] - X[:, None, :], t)
    sg = h < 0
    ch = sg[:, 1:] != sg[:, :-1]
    anyc = ch.any(1)
    firstc = torch.where(ch, ar[None, :], ns).min(1).values
    eb = end_t[bsl]
    beyond = eb.float() + (-h.gather(1, eb[:, None])[:, 0]) / STEP
    x = torch.where(anyc, firstc.float(), torch.where(h[:, 0] >= 0, torch.zeros_like(beyond), beyond))
    xr[s0:s0 + B] = x
xr = xr.cpu().numpy(); del PRt
tick("crossing table done")
# expansion: candidate i of b x crossing row r of b
r_cnt = np.bincount(rb, minlength=nR); r_ptr = np.r_[0, np.cumsum(r_cnt)]
# rows are grouped by b already (src sorted)
assert (np.diff(rb) >= 0).all()
row_of = np.arange(len(rb))
ei = np.repeat(np.arange(nc), r_cnt[cand_col])
er = seg_repeat(r_ptr[cand_col], r_cnt[cand_col])
dk = ((cand_k[ei] - xr[er]) * STEP / s_b[cand_col[ei]]).astype(np.float32)
ej = rj[er]; del er
tick(f"directed pair terms: {len(ei)}")
lo, hi = np.minimum(ei, ej), np.maximum(ei, ej); del ei, ej
key = lo.astype(np.int64) * nc + hi; del lo, hi
ukey, uinv, ucnt = np.unique(key, return_inverse=True, return_counts=True); del key
u_lo, u_hi = ukey // nc, ukey % nc
u_w = 2.0 / (deg[cand_col[u_lo]] + deg[cand_col[u_hi]])
tick(f"symmetric pair terms: {len(ukey)} (both directions: {np.mean(ucnt == 2):.0%})")

# colouring: greedy, by degree
colour = np.full(nR, -1, np.int64)
for bb in np.argsort(-np.diff(nb_ptr), kind="stable"):
    used = colour[dst[nb_ptr[bb]:nb_ptr[bb + 1]]]
    cl = 0
    while (used == cl).any():
        cl += 1
    colour[bb] = cl
ncol = int(colour.max()) + 1
tick(f"colours: {ncol}")

# ---------------------------------------------------------------- the solver on the device
f32 = torch.float32
ev_t = torch.from_numpy(ev0).to(dev, f32)
dk_t = torch.from_numpy(dk).to(dev); uinv_t = torch.from_numpy(uinv).to(dev)
ucnt_t = torch.from_numpy(ucnt).to(dev, f32)
ulo_t = torch.from_numpy(u_lo).to(dev); uhi_t = torch.from_numpy(u_hi).to(dev)
uw_t = torch.from_numpy(u_w).to(dev, f32)
cc_t = torch.from_numpy(cand_col).to(dev)
ccol_t = torch.from_numpy(colour[cand_col]).to(dev)
nU = len(ukey)


def seg_softmax(e, mask):
    e = torch.where(mask, e, torch.tensor(-float("inf"), device=dev))
    mx = torch.full((nR,), -float("inf"), device=dev).scatter_reduce(0, cc_t, e, "amax")
    w = torch.where(mask, torch.exp(e - mx[cc_t]), torch.zeros_like(e))
    s = torch.zeros(nR, device=dev).index_add_(0, cc_t, w)
    return w / torch.clamp(s[cc_t], min=1e-30)


def phi_sym(F):
    ph = torch.clamp(-0.5 * (dk_t / F) ** 2, min=-8.0)
    return torch.zeros(nU, device=dev).index_add_(0, uinv_t, ph) / ucnt_t


def agree_of(q, ph):
    a_ = uw_t * ph
    out = torch.zeros(nc, device=dev)
    out.index_add_(0, ulo_t, a_ * q[uhi_t]); out.index_add_(0, uhi_t, a_ * q[ulo_t])
    return out


def objective(q, ph):  # float64 on the CPU
    qd = q.cpu().double().numpy(); pd = ph.cpu().double().numpy()
    ent = np.where(qd > 0, qd * np.log(np.maximum(qd, 1e-300)), 0.0).sum()
    return float(-(qd * ev0).sum() - (u_w * pd * qd[u_lo] * qd[u_hi]).sum() + ent)


_starts = c_ptr[:-1][ncand > 0]
_cols = np.flatnonzero(ncand > 0)


def chosen(q):
    """per column: the index of its most probable candidate (nc if none) and that probability (CPU)"""
    qc = q.cpu().numpy()
    best = np.full(nR, nc, np.int64); mx = np.zeros(nR)
    mxs = np.maximum.reduceat(qc, _starts)
    mx[_cols] = mxs
    is_best = qc >= mx[cand_col] - 1e-7
    first_best = np.where(is_best, np.arange(nc), nc)
    best[_cols] = np.minimum.reduceat(first_best, _starts)
    return best, mx


all_mask = torch.ones(nc, dtype=torch.bool, device=dev)
q = seg_softmax(ev_t, all_mask)
log = []; rises = 0; prev = None
for F in (4.0, 2 ** 1.5, 2.0, 2 ** 0.5, 1.0):
    ph = phi_sym(F)
    f_old = objective(q, ph)
    for sw in range(80):
        for cl in range(ncol):
            sel = ccol_t == cl
            qn = seg_softmax(ev_t + agree_of(q, ph), sel)
            q = torch.where(sel, qn, q)
        f_new = objective(q, ph)
        best, _ = chosen(q)
        nfl = int((best != prev).sum()) if prev is not None else -1
        prev = best
        if f_new > f_old + 1e-6 * abs(f_old):
            rises += 1
        log.append(dict(F=round(F, 3), sweep=sw, objective=f_new, flips=nfl))
        if abs(f_old - f_new) <= 1e-7 * abs(f_old):
            break
        f_old = f_new
    tick(f"width x{F:.2f}: {sw + 1} sweeps, objective {f_new:.6g}, flips last sweep {nfl}")
ph = phi_sym(1.0)
ag = agree_of(q, ph)
best, mq = chosen(q)
agc = ag.cpu().numpy()
has = ncand > 0
top_k = np.full(nR, -1, np.int64); top_k[has] = cand_k[best[has]]
agree_at = np.full(nR, np.nan); agree_at[has] = agc[best[has]]
pmax = np.where(has, mq, 0.0)
res = dict(columns=int(nR), with_top=int(has.sum()), objective_rises=rises, sweeps=len(log),
           seconds=time.perf_counter() - t0, device=DEV)
json.dump(dict(res, log=log), open(OUT / "joint_gpu.json", "w"), indent=1)
np.savez(OUT / "joint_gpu.npz", R=R, vert=vert[R], root=PR[:, 0], top_k=top_k, pmax=pmax, agree_at=agree_at,
         col_gm=col_gm[R], nonwm=nonwm[R], ray_end=ray_end)
tick(json.dumps(res))
