"""Noise correlation by axis: background (fomblin/outside) blocks and deep-WM blocks.

usage: noise_corr.py <data_dir> <out_dir>
"""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

data, out = Path(sys.argv[1]), Path(sys.argv[2])
sub = "sub-INDD121850_ses-7THemix20250516"
echo_f = lambda e: data / f"{sub}_acq-channelCOMBx160um_dir-positive_run-02_echo-{e}_part-mag_FLASH.nii.gz"
aseg = np.asarray(nib.load(data / f"{sub}_aseg.nii.gz").dataobj).astype(np.int16)
print("aseg", aseg.shape, flush=True)

# coarse masks (factor 8) for block selection
f = 8
sh = tuple(s // f for s in aseg.shape)
crop = aseg[: sh[0] * f, : sh[1] * f, : sh[2] * f]
tis_c = (crop != 0).reshape(sh[0], f, sh[1], f, sh[2], f).any(axis=(1, 3, 5))
wm_c = (crop == 41).reshape(sh[0], f, sh[1], f, sh[2], f).all(axis=(1, 3, 5))
far_tis = ~ndimage.binary_dilation(tis_c, iterations=3)  # >= ~3 mm from any label
deep_wm = ndimage.binary_erosion(wm_c, iterations=2)
del crop

def plane_detrend(b):
    g = np.indices(b.shape).reshape(3, -1).T.astype(float)
    g = np.c_[np.ones(len(g)), g, g**2]
    c, *_ = np.linalg.lstsq(g, b.ravel(), rcond=None)
    return b - (g @ c).reshape(b.shape)

def lagcorr(b, axis, lag):
    a = np.moveaxis(b, axis, 0)
    return float((a[:-lag] * a[lag:]).mean() / (a * a).mean())

def blocks(mask_c, B, stride, max_n):
    # block corners (voxels) whose coarse footprint lies entirely in mask_c
    out = []
    for i in range(0, aseg.shape[0] - B, stride):
        for j in range(0, aseg.shape[1] - B, stride):
            for k in range(0, aseg.shape[2] - B, stride):
                ci, cj, ck = i // f, j // f, k // f
                ei, ej, ek = (i + B - 1) // f + 1, (j + B - 1) // f + 1, (k + B - 1) // f + 1
                if ei > sh[0] or ej > sh[1] or ek > sh[2]:
                    continue
                if mask_c[ci:ei, cj:ej, ck:ek].all():
                    out.append((i, j, k))
    rng = np.random.default_rng(0)
    if len(out) > max_n:
        out = [out[t] for t in rng.choice(len(out), max_n, replace=False)]
    return out

bg_blocks = blocks(far_tis, 64, 48, 300)
wm_blocks = blocks(deep_wm, 32, 24, 400)
print("blocks bg", len(bg_blocks), "wm", len(wm_blocks), flush=True)

res = {"shape": aseg.shape, "n_bg_blocks": len(bg_blocks), "n_wm_blocks": len(wm_blocks), "echo": {}}
spectra = {}
bg_noise = {}
for e in (1, 2, 3):
    img = np.asarray(nib.load(echo_f(e)).dataobj)
    print("echo", e, img.shape, img.dtype, flush=True)
    # background: magnitude, plane+quadratic detrend
    stats, lags, P = [], {a: [] for a in range(3)}, np.zeros((64, 64, 64))
    keep = []
    for (i, j, k) in bg_blocks:
        b = img[i:i + 64, j:j + 64, k:k + 64].astype(float)
        m, s = b.mean(), b.std()
        stats.append((m, s))
        keep.append(b)
    stats = np.array(stats)
    # reject blocks with signal (ghosts, container, bubbles): mean above 1.5x median
    med = np.median(stats[:, 0])
    ok = stats[:, 0] < 1.5 * med
    bn = []
    for b, good in zip(keep, ok):
        if not good:
            continue
        d = plane_detrend(b)
        for a in range(3):
            lags[a].append([lagcorr(d, a, L) for L in (1, 2, 3, 4)])
        P += np.abs(np.fft.fftn(d)) ** 2
        bn.append(d)
    bg_noise[e] = bn
    P /= ok.sum()
    spectra[e] = [np.fft.fftshift(P.mean(axis=tuple(x for x in range(3) if x != a))) for a in range(3)]
    res["echo"][e] = {
        "bg_blocks_used": int(ok.sum()),
        "bg_mean_median": float(np.median(stats[ok, 0])),
        "bg_std_median": float(np.median(stats[ok, 1])),
        "bg_lagcorr_axis_L1to4": {a: np.median(lags[a], axis=0).round(3).tolist() for a in range(3)},
    }
    # deep WM: log, quadratic detrend
    wl = {a: [] for a in range(3)}
    PW = np.zeros((32, 32, 32))
    for (i, j, k) in wm_blocks:
        b = np.log(np.maximum(img[i:i + 32, j:j + 32, k:k + 32].astype(float), 1))
        d = plane_detrend(b)
        for a in range(3):
            wl[a].append([lagcorr(d, a, L) for L in (1, 2, 3, 4)])
        PW += np.abs(np.fft.fftn(d)) ** 2
    PW /= max(len(wm_blocks), 1)
    spectra[f"wm{e}"] = [np.fft.fftshift(PW.mean(axis=tuple(x for x in range(3) if x != a))) for a in range(3)]
    res["echo"][e]["wm_lagcorr_axis_L1to4"] = {a: np.median(wl[a], axis=0).round(3).tolist() for a in range(3)}
    del img

# cross-echo correlation of background noise (same blocks)
n = min(len(bg_noise[1]), len(bg_noise[2]), len(bg_noise[3]))
if all(len(bg_noise[e]) == len(bg_noise[1]) for e in (2, 3)):
    for a_, b_ in ((1, 2), (1, 3), (2, 3)):
        cs = [np.corrcoef(x.ravel(), y.ravel())[0, 1] for x, y in zip(bg_noise[a_], bg_noise[b_])]
        res[f"bg_cross_echo_{a_}{b_}"] = float(np.median(cs))

(out / "noise_corr.json").write_text(json.dumps(res, indent=1))
np.savez(out / "spectra.npz", **{f"{k}_ax{a}": v[a] for k, v in spectra.items() for a in range(3)})
print(json.dumps(res, indent=1))
