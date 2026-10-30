"""Which sheets are contacts between two stacks? Three tests at every sheet voxel, compared with the user's readings.

For each surveyed crop (site_survey.py overlays; label 2 = sheet) and for the M1S1 stretch (collide.py's
collision classes), at every sheet voxel at least 2 mm from the box faces:
1. Orientation: |cos| between the data's layer normal (summed structure tensor of echoes 2 and 3) and the
   sheet's own normal (smallest principal axis of the sheet voxels within 0.6 mm). A contact between two
   stacks has the layers parallel to the sheet (|cos| near 1); a front meeting itself round a bend has them
   running through it (|cos| low).
2. Where it sits: distance to the WM and to fluid (mm).
3. The profile across it: echo 2, trend removed, along the sheet normal (0.3 mm disc average); the doubled-band
   index is the mean at 0.2-0.4 mm on both sides minus the mean within 0.08 mm (positive: two bright superficial
   bands with a notch between, as at the marked contact, finding 17).
Readings: the user's (contact; not a contact) per piece or region; everything else is unread.
Views: distributions of the three tests by reading, a per-piece table, and a per-voxel overlay for ITK-SNAP.

Usage: sheet_tests.py <outdir> <workdir> <datadir> <M1S1 workdir>
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
from scipy.spatial import cKDTree

import curv_lib as C

OUT, S, DATA, M1 = sys.argv[1:5]
VOX = np.array([0.16028, 0.16, 0.16028])
CROPS = ("V1V2_banks", "V1V2_thin", "PCC_SPL", "PCC_SPL_p2", "PCC_SPL_p3")
# the user's readings: (crop, a point on the piece in ITK-SNAP numbering of the crop, reading, optional region)
READINGS = [
    ("V1V2_banks", (78, 67, 73), "contact", None),
    (
        "PCC_SPL",
        (70, 80, 85),
        "contact",
        lambda v: (
            ~((v[:, 1] >= 39) & (v[:, 1] <= 70) & (v[:, 2] >= 70) & (v[:, 2] <= 90))
        ),
    ),
    (
        "PCC_SPL",
        (70, 80, 85),
        "not a contact",
        lambda v: (v[:, 1] >= 39) & (v[:, 1] <= 70) & (v[:, 2] >= 70) & (v[:, 2] <= 90),
    ),
    ("PCC_SPL_p2", (88, 59, 59), "contact", None),
    ("PCC_SPL_p3", (71, 72, 71), "not a contact", None),
]
COLS = {"contact": "#2b7bd9", "not a contact": "#e8473a", "unread": "#9a9a9a"}


def sheet_measures(B, sheet, nh, coh, rng, nprof=1500):
    vox = VOX
    sv = np.argwhere(sheet)
    tree = cKDTree(sv * vox)
    nbr = tree.query_ball_point(sv * vox, r=0.6)
    snorm = np.zeros((len(sv), 3))
    for i, nb in enumerate(nbr):
        X = sv[nb] * vox
        if len(X) < 5:
            continue
        _, V = np.linalg.eigh(np.cov((X - X.mean(0)).T))
        snorm[i] = V[:, 0]
    dn = nh[tuple(sv.T)]
    cos = np.abs((dn * snorm).sum(1))
    cohv = coh[tuple(sv.T)]
    dwm = ndi.distance_transform_edt(~B["wm"], sampling=vox)[tuple(sv.T)]
    dfl = ndi.distance_transform_edt(B["rim"] != 1, sampling=vox)[tuple(sv.T)]
    # profiles at a random subset
    idx = rng.permutation(len(sv))[:nprof]
    ss = np.arange(-1.0, 1.0001, 0.04)
    band = np.full(len(sv), np.nan)
    disp = np.where(B["ctx"], B["disp"], 0.0)
    for i in idx:
        n_ = snorm[i]
        if not n_.any():
            continue
        a = np.cross(n_, [1.0, 0, 0] if abs(n_[0]) < 0.9 else [0, 1.0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(n_, a)
        prof = []
        for o in [np.zeros(3)] + [
            0.3 * (np.cos(t) * a + np.sin(t) * b)
            for t in np.linspace(0, 2 * np.pi, 6, endpoint=False)
        ]:
            line = sv[i][None] + (ss[:, None] * n_[None] + o[None]) / vox
            prof.append(ndi.map_coordinates(disp, line.T, order=1, mode="nearest"))
        pr = np.mean(prof, 0)
        band[i] = (
            pr[(np.abs(ss) >= 0.2) & (np.abs(ss) <= 0.4)].mean()
            - pr[np.abs(ss) <= 0.08].mean()
        )
    return sv, cos, cohv, dwm, dfl, band


def main():
    rng = np.random.default_rng(0)
    rows = []
    allv = []
    for crop in CROPS + ("M1S1",):
        if crop == "M1S1":
            m = json.load(open(f"{M1}/M1S1_test_points.json"))
            lo = np.array(m["full_lo"]) - 15
            hi = np.array(m["full_hi"]) + 15
            B = C.load(lo, hi, "M1S1")
            sheet = np.load(f"{M1}/collide_state.npz")["ctype"] == 1
            P = np.array(m["gap_all"]) + 15
            dP = ndi.distance_transform_edt(~_mask(P, sheet.shape), sampling=VOX)
        else:
            box = json.load(open(f"{DATA}/{crop}_sites.json"))
            lo, hi = np.array(box["full_lo"]), np.array(box["full_hi"])
            B = C.load(lo, hi, crop)
            sheet = (
                np.asarray(nib.load(f"{DATA}/{crop}_sites_overlay.nii.gz").dataobj) == 2
            )
        sheet &= B["dface"] >= 2.0
        nh, coh = C.tensor_normal(B)
        lab, n = ndi.label(sheet, structure=np.ones((3, 3, 3)))
        sizes = np.bincount(lab.ravel())[1:]
        big = np.flatnonzero(sizes >= 300) + 1
        sheet = np.isin(lab, big)
        sv, cos, cohv, dwm, dfl, band = sheet_measures(B, sheet, nh, coh, rng)
        comp = lab[tuple(sv.T)]
        reading = np.array(["unread"] * len(sv), dtype=object)
        if crop == "M1S1":
            reading[dP[tuple(sv.T)] <= 3.0] = "contact"
        for c_, pt, rd, region in READINGS:
            if c_ != crop:
                continue
            q = np.array(pt) - 1
            d_, ind = ndi.distance_transform_edt(lab == 0, return_indices=True)
            target = lab[tuple(ind[:, q[0], q[1], q[2]])]
            sel = comp == target
            if region is not None:
                sel &= region(sv + 1)
            reading[sel] = rd
        for li in big:
            sel = comp == li
            rds, cnt = np.unique(reading[sel], return_counts=True)
            cen = sv[sel].mean(0) + 1
            rows.append(
                dict(
                    crop=crop,
                    centre=tuple(int(round(c)) for c in cen),
                    voxels=int(sel.sum()),
                    reading=", ".join(f"{r} {c}" for r, c in zip(rds, cnt)),
                    cos=np.median(cos[sel]),
                    coh=np.median(cohv[sel]),
                    dwm10=np.percentile(dwm[sel], 10),
                    dwm=np.median(dwm[sel]),
                    dfl=np.median(dfl[sel]),
                    band=np.nanmedian(band[sel]),
                )
            )
        allv.append(
            dict(
                crop=crop,
                cos=cos,
                coh=cohv,
                dwm=dwm,
                dfl=dfl,
                band=band,
                reading=reading,
            )
        )
        # overlay per crop: sheet voxels coloured by orientation (1: |cos| >= 0.7, 2: 0.4-0.7, 3: < 0.4)
        if crop != "M1S1":
            ov = np.zeros(sheet.shape, np.uint8)
            ov[tuple(sv.T)] = np.where(cos >= 0.7, 1, np.where(cos >= 0.4, 2, 3))
            nib.save(
                nib.Nifti1Image(ov, B["p"].affine),
                f"{DATA}/{crop}_sheet_tests_overlay.nii.gz",
            )
        print(f"{crop}: {len(sv):,} sheet voxels in {len(big)} pieces")
    open(f"{DATA}/sheet_tests_labels.txt", "w").write(
        '    0     0    0    0        0  0  0    "Clear Label"\n'
        '    1    40  120  255        1  1  1    "Sheet: layers parallel (|cos| >= 0.7)"\n'
        '    2   255  200   40        1  1  1    "Sheet: oblique (0.4-0.7)"\n'
        '    3   255   40   40        1  1  1    "Sheet: layers run through (< 0.4)"\n'
    )
    print(
        "\ncrop | piece centre (ITK) | voxels | readings | median |cos| | coherence | distance to WM p10 / median | to fluid | band index"
    )
    for r in rows:
        print(
            f"{r['crop']:11s} {str(r['centre']):16s} {r['voxels']:6d} | {r['reading']:38s} | {r['cos']:.2f} | {r['coh']:.2f} | "
            f"{r['dwm10']:.2f} / {r['dwm']:.2f} | {r['dfl']:.2f} | {r['band']:+.3f}"
        )
    # pooled by reading
    pooled = {
        k: np.concatenate([a[k] for a in allv])
        for k in ("cos", "coh", "dwm", "dfl", "band")
    }
    rd = np.concatenate([a["reading"] for a in allv])
    print(
        "\npooled by reading: n | median |cos| (IQR) | distance to WM median (p10) | to fluid median | band index median (IQR)"
    )
    for r_ in ("contact", "not a contact", "unread"):
        s_ = rd == r_
        print(
            f"  {r_:14s} {s_.sum():6d} | {np.median(pooled['cos'][s_]):.2f} ({np.subtract(*np.percentile(pooled['cos'][s_], [75, 25])):.2f}) | "
            f"{np.median(pooled['dwm'][s_]):.2f} ({np.percentile(pooled['dwm'][s_], 10):.2f}) | {np.median(pooled['dfl'][s_]):.2f} | "
            f"{np.nanmedian(pooled['band'][s_]):+.3f} ({np.subtract(*np.nanpercentile(pooled['band'][s_], [75, 25])):.3f})"
        )
    fig, axs = plt.subplots(1, 4, figsize=(28, 6.5))
    for a, (k, lab_, bins) in zip(
        axs,
        (
            (
                "cos",
                "|cos| between the layer normal and the sheet normal",
                np.linspace(0, 1, 26),
            ),
            (
                "dwm",
                "distance from the sheet voxel to the WM (mm)",
                np.linspace(0, 5, 26),
            ),
            (
                "dfl",
                "distance from the sheet voxel to fluid (mm)",
                np.linspace(0, 5, 26),
            ),
            (
                "band",
                "doubled-band index (echo 2, trend removed)",
                np.linspace(-0.3, 0.3, 31),
            ),
        ),
    ):
        for r_ in ("contact", "not a contact", "unread"):
            s_ = (rd == r_) & np.isfinite(pooled[k])
            if s_.any():
                a.hist(
                    pooled[k][s_],
                    bins=bins,
                    density=True,
                    histtype="step",
                    lw=2,
                    color=COLS[r_],
                    label=f"{r_} ({s_.sum():,})",
                )
        a.set_xlabel(lab_)
        a.legend(fontsize=9)
    axs[0].set_title("orientation: contacts should be near 1")
    axs[1].set_title("depth: contacts sit about a thickness from the WM")
    axs[2].set_title("fluid: contacts meet it only at their edges")
    axs[3].set_title("profile: contacts show two bands and a notch")
    fig.suptitle(
        "Sheet tests at every sheet voxel of the surveyed crops and the M1S1 stretch, by the user's reading",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(f"{OUT}/sheet_tests.png", dpi=75)
    plt.close(fig)
    print("wrote sheet_tests.png")


def _mask(P, shape):
    m = np.zeros(shape, bool)
    m[tuple(np.round(P).astype(int).T)] = True
    return m


if __name__ == "__main__":
    main()
