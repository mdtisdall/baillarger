"""Top-stage features for every superblock (owned columns only), on the GPU
where it pays: orient's detrended log echoes with the dilated mask, C^-1
weights matched to the outside-medium contrast, second differences, the
sheet score lambda1 - |lambda2| (closed-form eigenvalues), trilinear sampling
along the rays. Writes per superblock: the owned columns' vertices and ray
paths, label tops, the candidate table with its evidence, a histogram of the
sheet score over cortex samples (for one global floor), and the column edges.

args: SCAN_DIR (top_scan.py OUT, with diag/) OUT_DIR   env: DEVICE (mps), WORKERS (1), DILATE_MM (0.48); inputs: top_paths.py
"""
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).parent))
import top_paths as TP  # noqa: E402
from top_stage_lib import behind_root_lengths, nearest_class  # noqa: E402

DSEG, ECHOES, NOISE = TP.dseg(), TP.echoes(), TP.noise()

SCAN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUTSIDE = (0, 43)
WM_FILLED = (41, 49, 50, 51, 52, 192)
CORTEX = 42
STEP = 0.04
SEED_RUN = 8
MIN_TOP_MM = 0.5
MIRROR_MM = (0.12, 1.2)
INWARD_MM = 1.5  # the ray extended inward from the root, for the GM-like run
DIL_MM = float(os.environ.get("DILATE_MM", "0.48"))
HBINS = np.r_[0, np.geomspace(1e-5, 10, 400)]  # sheet-score histogram bins
G = {}


def init():
    from baillarger import orient as O
    from baillarger.backend import device as get_device
    from baillarger.orient_noise import read_noise
    import coarse_kernels as K
    G["O"], G["K"] = O, K
    G["dev"] = get_device(os.environ.get("DEVICE", "mps"))
    G["noise"] = read_noise(NOISE)
    G["dimg"] = nib.load(str(DSEG))
    G["E"] = [nib.load(str(e)).dataobj for e in ECHOES]


def one(f):
    if not G:
        init()
    O, K, dev = G["O"], G["K"], G["dev"]
    t0 = time.perf_counter()
    z = np.load(f)
    own = z["owned"]
    vert_all = z["vert"]
    tri = z["tri"]
    edges = np.unique(np.sort(np.r_[tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]], 1), axis=0)
    edges_v = vert_all[edges]
    oi = np.flatnonzero(own)
    P = z["P"][oi].astype(np.float32)
    m, ns, _ = P.shape
    out = dict(vert=vert_all[oi], P=P, edges_v=edges_v)
    if m == 0:
        np.savez(OUT / f.name.replace("diag_", "feat_"), empty=True, **out)
        return f"{f.name}: empty"
    A = G["dimg"].affine
    inv = np.linalg.inv(A)
    vs = np.sqrt((A[:3, :3] ** 2).sum(0))
    q = P.reshape(-1, 3).astype(np.float64) @ inv[:3, :3].T + inv[:3, 3]
    shp = np.array(G["dimg"].shape)
    lo = np.maximum(np.floor(q.min(0)).astype(int) - 20, 0)
    hi = np.minimum(np.ceil(q.max(0)).astype(int) + 21, shp)
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    lab = np.asarray(G["dimg"].dataobj[box], np.uint8)
    raw = np.stack([np.asarray(e[box], np.float32) for e in G["E"]])
    ctx = lab == CORTEX
    if not ctx.any():
        np.savez(OUT / f.name.replace("diag_", "feat_"), empty=True, **out)
        return f"{f.name}: no cortex"
    dc = distance_transform_edt(~ctx, sampling=vs)
    msk = dc <= DIL_MM + 1e-6
    outside = np.isin(lab, OUTSIDE)
    far_out = outside & (dc >= 2 * vs.min()) & (dc <= 6 * vs.min())
    d_out = distance_transform_edt(~outside, sampling=vs).astype(np.float32) if outside.any() else np.full(lab.shape, 99, np.float32)
    wm_far = np.isin(lab, WM_FILLED) & (dc >= 1.0)
    del dc
    # class means of the log echoes (cortex, far WM, outside medium) for the GM-like run
    lraw = np.log(np.maximum(raw, 1e-6))
    mu = {n_: (np.median(lraw[:, s_], 1) if s_.sum() >= 100 else None)
          for n_, s_ in (("gm", ctx), ("wm", wm_far), ("fl", far_out))}
    del lraw
    td = dev.torch
    te = torch.as_tensor(raw, device=td)
    tc = torch.as_tensor(ctx, device=td)
    tm = torch.as_tensor(msk, device=td)
    le = O.log_echo(te)
    lv = O.masked_smooth(le, tc, O._kernels(O.SIGMA, vs))
    medn = np.nanmedian(lv[:, tc].cpu().numpy().astype(np.float64), 1)  # (E,)
    med = torch.as_tensor(medn, dtype=lv.dtype, device=td)
    lv0 = torch.nan_to_num(torch.where(tm, lv, torch.nan))
    del lv
    det = torch.where(tm, le - lv0, 0.0)
    sm = torch.where(tm, O.masked_smooth(det, tm, [O.taps(O.SD)] * 3), 0.0)
    del det
    C = O._noise_c(medn.reshape(-1, 1, 1, 1), G["noise"])[..., 0, 0, 0]
    tfo = torch.as_tensor(far_out, device=td)
    if far_out.sum() >= 100:
        c = (le[:, tfo] - med[:, None]).cpu().numpy().astype(np.float64); c = np.median(c, 1)
    else:
        c = np.array([-2.78, -2.48, -2.03])  # superblock 2_2_4's, where this box has too little outside medium
    Ci = np.linalg.inv(C)
    w = -(Ci @ c) / (c @ Ci @ c)
    u = torch.einsum("e,e...->...", torch.as_tensor(w, dtype=sm.dtype, device=td), sm)
    del sm, le
    Hc = []
    for i in range(3):
        for j in range(i, 3):
            if i == j:
                d = torch.zeros_like(u)
                a_, b_, c_ = [slice(None)] * 3, [slice(None)] * 3, [slice(None)] * 3
                a_[i], b_[i], c_[i] = slice(2, None), slice(1, -1), slice(None, -2)
                d[tuple(b_)] = u[tuple(a_)] - 2 * u[tuple(b_)] + u[tuple(c_)]
            else:
                d = O.central(O.central(u, i), j)
            Hc.append(d)
    # echoes in cortex units (for the mirror profiles)
    ech = te / torch.as_tensor(np.array([float(np.median(raw[k][ctx])) for k in range(3)], np.float32), device=td)[:, None, None, None]
    vol = torch.stack([*Hc, torch.as_tensor(outside, dtype=torch.float32, device=td),
                       torch.as_tensor(np.isin(lab, WM_FILLED), dtype=torch.float32, device=td),
                       torch.as_tensor(d_out, device=td), *ech])
    del Hc, u, te
    qt = torch.as_tensor((q - lo).astype(np.float32), device=td)
    smp = K.sample_t(vol, qt)  # (12, m*ns)
    del vol
    H6 = smp[:6].T
    M = torch.stack([torch.stack([H6[:, 0], H6[:, 1], H6[:, 2]], 1),
                     torch.stack([H6[:, 1], H6[:, 3], H6[:, 4]], 1),
                     torch.stack([H6[:, 2], H6[:, 4], H6[:, 5]], 1)], 1)
    _, l1, l2 = K.leading_t(M)
    S = torch.clamp(l1 - l2.abs(), min=0).reshape(m, ns).cpu().numpy()
    Ov = smp[6].reshape(m, ns).cpu().numpy()
    Wv = smp[7].reshape(m, ns).cpu().numpy() > 0.5
    dout = smp[8].reshape(m, ns).cpu().numpy()
    I = smp[9:12].reshape(3, m, ns).permute(1, 2, 0).cpu().numpy()
    del smp, M, H6
    # inward extension of each ray from its root, along the first step's direction
    nin = int(round(INWARD_MM / STEP))
    d0 = P[:, 1] - P[:, 0]
    d0 /= np.maximum(np.linalg.norm(d0, axis=1, keepdims=True), 1e-9)
    Pin = P[:, :1] - (np.arange(1, nin + 1)[None, :, None] * STEP) * d0[:, None]  # (m, nin, 3); j=0 next to the root
    qin = Pin.reshape(-1, 3).astype(np.float64) @ inv[:3, :3].T + inv[:3, 3] - lo
    qin = np.clip(qin, 0, np.array(lab.shape) - 1)
    Iin = K.sample_t(ech, torch.as_tensor(qin.astype(np.float32), device=td)).reshape(3, m, nin).permute(1, 2, 0).cpu().numpy()
    del ech
    # GM-like samples: nearest class mean in log echoes, distance weighted by C^-1
    cmed = np.log(np.array([float(np.median(raw[k][ctx])) for k in range(3)]))
    have_cls = all(v is not None for v in mu.values())
    if have_cls:
        X = np.log(np.maximum(np.concatenate([Iin[:, ::-1], I], 1), 1e-6)) + cmed  # (m, nin+ns, 3), deepest inward first
        cls = nearest_class(X, np.array([mu["gm"], mu["wm"], mu["fl"]]), Ci)  # 0 GM-like, 1 WM-like, 2 fluid-like
        del X
        gm_behind, fl_behind = (a.astype(np.float32) for a in behind_root_lengths(cls, nin, STEP))  # behind the root
        root_cls = cls[:, nin].astype(np.int8)
    else:
        gm_behind = np.full(m, np.nan, np.float32)
        fl_behind = np.full(m, np.nan, np.float32)
        root_cls = np.full(m, -1, np.int8)
    kmin = int(round(MIN_TOP_MM / STEP))
    inside_o = Ov > 0.5
    # the first entry into outside medium over the whole ray (a top at sample 0 is zero thickness)
    first = np.where(inside_o.any(1), inside_o.argmax(1), -1)
    # the run of samples inside outside medium from the first one
    run = np.zeros(m, int)
    for i_ in np.flatnonzero(first >= 0):
        r = inside_o[i_, first[i_]:]
        run[i_] = r.argmin() if not r.all() else len(r)
    ctx_s = ~Wv & ~inside_o
    ctx_s[:, :kmin] = False
    hist = np.histogram(S[ctx_s], HBINS)[0]
    floor_local = float(np.percentile(S[ctx_s], 90)) if ctx_s.any() else 0.0
    Smax = np.maximum.reduce([np.roll(S, d_, 1) for d_ in range(-3, 4)])
    cand = (S >= Smax) & (S >= 0.25 * floor_local) & (S > 0)
    cand[:, :kmin] = False
    cand[inside_o] = False
    ci, ck = np.nonzero(cand)
    # label tops as candidates too
    li = np.flatnonzero(first >= 0)  # zero thickness is a valid top
    ci = np.r_[ci, li]
    ck = np.r_[ck, first[li]]
    is_lab = np.r_[np.zeros(len(ci) - len(li), bool), np.ones(len(li), bool)]
    Sc = S[ci, ck]
    at_label = dout[ci, ck] <= float(vs.max())
    Wc = np.cumsum(Wv, 1)
    end = np.minimum(3 * ck, ns - 1)
    far_wm = (Wc[ci, end] - np.where(ck > 0, Wc[ci, np.maximum(ck - 1, 0)], 0)) > 0
    jj = np.arange(int(round(MIRROR_MM[0] / STEP)), int(round(MIRROR_MM[1] / STEP)) + 1)
    kb, ka = ck[:, None] - jj[None], ck[:, None] + jj[None]
    ok = (kb >= 0) & (ka < ns)
    a_ = I[ci[:, None], np.clip(kb, 0, ns - 1)]  # (n, J, 3)
    b_ = I[ci[:, None], np.clip(ka, 0, ns - 1)]
    okk = np.repeat(ok[:, :, None], 3, 2)
    n_ = okk.sum((1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = (a_ * okk).sum((1, 2)) / n_
        mb = (b_ * okk).sum((1, 2)) / n_
        da, db = (a_ - ma[:, None, None]) * okk, (b_ - mb[:, None, None]) * okk
        mirror = (da * db).sum((1, 2)) / np.sqrt((da**2).sum((1, 2)) * (db**2).sum((1, 2)))
    mirror[ok.sum(1) < 10] = np.nan
    out.update(first=first.astype(np.int16), run=run.astype(np.int16), hist=hist, floor_local=floor_local,
               c_ci=ci.astype(np.int32), c_k=ck.astype(np.int16), c_lab=is_lab, c_S=Sc.astype(np.float32),
               c_at_label=at_label, c_far_wm=far_wm, c_mirror=mirror.astype(np.float32), gm_behind=gm_behind, fl_behind=fl_behind, root_cls=root_cls, w=w, c=c,
               mu=np.array([mu[n_] if mu[n_] is not None else np.full(3, np.nan) for n_ in ("gm", "wm", "fl")]))
    np.savez(OUT / f.name.replace("diag_", "feat_"), **out)
    if dev.name == "mps":
        torch.mps.empty_cache()
    return f"{f.name}: {m} columns, {len(ci)} candidates, {time.perf_counter() - t0:.1f} s"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted((SCAN / "diag").glob("diag_*.npz"))
    todo = [f for f in files if not (OUT / f.name.replace("diag_", "feat_")).exists()]
    nw = int(os.environ.get("WORKERS", "1"))
    t0 = time.perf_counter()
    print(f"{len(todo)} of {len(files)} superblocks, {nw} workers", flush=True)
    if nw <= 1:
        for i, f in enumerate(todo):
            print(f"{i + 1}/{len(todo)} {one(f)}", flush=True)
    else:
        with ProcessPoolExecutor(nw, mp_context=mp.get_context("spawn")) as ex:
            futs = [ex.submit(one, f) for f in todo]
            for i, fu in enumerate(as_completed(futs)):
                print(f"{i + 1}/{len(todo)} {fu.result()}", flush=True)
    (OUT / "features.json").write_text(json.dumps({"seconds": time.perf_counter() - t0, "dilate_mm": DIL_MM, "workers": nw}))
    print("done", time.perf_counter() - t0, flush=True)


if __name__ == "__main__":
    main()
