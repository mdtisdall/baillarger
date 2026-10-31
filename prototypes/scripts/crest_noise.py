"""Checks the predicted noise of the fitted field T (streamlines.step_noise) and the noise-aware
crest test (streamlines.crest_voxels) on the fold phantoms, where the noise-free field is known.

Per phantom (the site's features; measured noise and noise-free): T extended into the fluid and the
WM as the crest method builds it, on both. The actual step noise is the change of (T_noisy -
T_clean) over one voxel along each axis, in the cortex at least 2 mm from a box face; it is
compared, in bins of the predicted noise, with the prediction (their ratio should be about 1). The
crest voxels of the noisy T: their share of the cortex, and their distance to the nearest crest
voxel of the noise-free T (computed with the noisy run's predicted noise, the same test).

Writes <work>/crest_noise_<name>.json. Usage: crest_noise.py <work dir> --name SITE
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
import fold_phantoms as FP

from baillarger import column_traces as CT
from baillarger import streamlines as S


def field(B, vox):
    m = B["ctx"].astype(float)
    base = S.structure_tensor(B["det"][1], m) + S.structure_tensor(B["det"][2], m)
    labels = B["labels"]
    T, _ = S.fitted_field(
        base, B["ctx"], B["wm"], vox, fluid=labels == 1, wm_band_mm=CT.WM_BAND_MM
    )
    noise, sigma2 = S.step_noise(
        base, B["ctx"], S.signed_wm_direction(B["wm"], vox), vox
    )
    return T, noise, sigma2


def gradient_power(sd_noise, sd=S.SD):
    """The gradient noise power per component implied by white noise of each echo's SD, through
    the Gaussian of sd voxels and central differences (summed over the two echoes)."""
    m = int(np.ceil(4 * sd)) + 2
    d = np.zeros((2 * m + 1,) * 3)
    d[m, m, m] = 1.0
    r = np.gradient(ndi.gaussian_filter(d, sd), axis=0)
    return float(sum(s_**2 for s_ in sd_noise) * np.sum(r**2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--name", required=True)
    A = ap.parse_args()
    out = {}
    for pname, fold in FP.standard().items():
        Bn = FP.make(fold, A.name, noise=1.0)
        Bc = FP.make(fold, A.name, noise=0.0)
        vox = np.asarray(Bn["vox"], float)
        Tn, noise, sigma2 = field(Bn, vox)
        Tc, _, sigma2_clean = field(Bc, vox)
        v, f, nrm = CT.white_mesh(Bn["wm"])
        kappa = CT.level_set_concavity(v, f, nrm, Tn, Bn["ctx"], vox)
        sds = Bn["noise_sd"]
        sds = (
            [sds[k] for k in sorted(sds)]
            if isinstance(sds, dict)
            else list(np.atleast_1d(sds))
        )
        dT = Tn - Tc
        inner = Bn["ctx"] & (Bn["dface"] >= 2.0)
        steps, pred = [], []
        for a in range(3):
            sl0 = [slice(None)] * 3
            sl1 = [slice(None)] * 3
            sl0[a], sl1[a] = slice(0, -1), slice(1, None)
            m = inner[tuple(sl0)] & inner[tuple(sl1)]
            steps.append((dT[tuple(sl1)] - dT[tuple(sl0)])[m])
            pred.append(0.5 * (noise[tuple(sl0)] + noise[tuple(sl1)])[m])
        steps, pred = np.concatenate(steps), np.concatenate(pred)
        q = np.quantile(pred, [0, 0.25, 0.5, 0.75, 1.0])
        bins = []
        for lo, hi in zip(q[:-1], q[1:]):
            mm = (pred >= lo) & (pred <= hi)
            bins.append(
                dict(
                    predicted_mm=round(float(np.median(pred[mm])), 5),
                    actual_sd_mm=round(float(np.std(steps[mm])), 5),
                    ratio=round(float(np.std(steps[mm]) / np.median(pred[mm])), 2),
                )
            )
        free = S.fitted_free(
            Bn["ctx"], Bn["wm"], vox, fluid=Bn["labels"] == 1, wm_band_mm=CT.WM_BAND_MM
        )
        cn = S.crest_voxels(Tn, Bn["ctx"], noise, kappa, vox, free=free)
        cc = S.crest_voxels(Tc, Bc["ctx"], noise, kappa, vox, free=free)
        on_contact = None
        if fold.kind == "kiss":
            Xc = np.argwhere(cn & inner).astype(float)
            Yc = fold.to_local(Xc)
            rc = np.linalg.norm(Yc - fold.foot(Yc), axis=1)
            open_ = fold.in_open(Yc)
            on_contact = dict(
                within_1_5_voxel_of_true_contact=round(
                    float(np.mean(np.abs(rc - fold.p["Rc"]) <= 1.5 * FP.H)), 3
                ),
                in_open_sector=round(float(np.mean(open_)), 3),
            )
        d_to_clean = (
            ndi.distance_transform_edt(~cc) if cc.any() else np.full(cn.shape, np.inf)
        )
        dn = d_to_clean[cn & inner]
        out[pname] = dict(
            gradient_noise_power=dict(
                from_smallest_eigenvalue=sigma2,
                from_known_noise=gradient_power(sds),
                smallest_eigenvalue_noise_free=sigma2_clean,
            ),
            step_noise_by_predicted_quartile=bins,
            crest_share_noisy=round(float(cn[inner].mean()), 5),
            kiss_crest_location=on_contact,
            crest_share_clean=round(float(cc[inner].mean()), 5),
            noisy_crest_within_1_5_voxel_of_clean_crest=(
                round(float(np.mean(dn <= 1.5)), 3) if dn.size else None
            ),
        )
        print(pname, json.dumps(out[pname]), flush=True)
    json.dump(out, open(f"{A.work}/crest_noise_{A.name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
