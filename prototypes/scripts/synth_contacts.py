"""Synthetic contact tests for the one-pass weighted competition (Next step 0).

Shapes with a known answer, all 3D (extruded along y; voxels):
- U-slab: a V-shaped sulcus with a rounded fundus of radius R, lined by a
  ribbon of thickness t = 16 with fluid inside. R = 6 < t: the fronts focus at
  the fundus's centre and meet themselves along the bisector, from the centre
  (arrival R) up to arrival t: a fold, not a contact. R = 22 > t: no meeting.
- P4 (baillarger.phantoms): banks of 12 and 18 voxels, the 16-voxel gap filled
  in a patch, the contact marked by a dark line at x = 25.5 (not at the banks'
  natural thicknesses).
- P8: the same banks touching at their natural thicknesses (contact x = 22.5).
- Omega: a round chamber (radius 34) with a narrow neck to the surface; the
  ribbon is 12 voxels on the left and 18 on the right, and the neck (30
  voxels) is filled, so the two sides touch at their natural thicknesses at
  x = cx - 3.5. The two sides are one front, folded round the chamber.

Weights per root patch (mu, sigma = 0.1 mu):
- equal: one mu for all (equal arrival);
- open: mu from the visible tops of an unweighted pass (patches without a
  top take the nearest one's), as a thickness field would give;
- column tops: the same, counting a top only where the fluid lies along the
  column (within 45 degrees), not beside it;
- oracle: each side's true arrival at the contact (P4, P8, omega).

Metric: unit cost along the EDT gradient (the distance-gradient direction),
cost ratio 3 across it. Arrival accuracy is checked against the EDT for one
label, root patches, and one label per root voxel.

Usage: synth_contacts.py <figures dir> <work dir>
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy import ndimage as ndi

from baillarger import phantoms
from onepass_lib import (
    contact_points,
    contacts,
    open_thickness,
    pure_voronoi,
    run,
    setup,
)

FIG = Path(sys.argv[1])
WORK = Path(sys.argv[2])
RATIO = 3.0
CV = 0.1
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"


def wedge_distance(
    x: np.ndarray, z: np.ndarray, cx: float, zo: float, alpha: float
) -> np.ndarray:
    """Distance in (x, z) from the upward wedge with apex (cx, zo), half-angle alpha."""
    px, pz = x - cx, z - zo
    inside = (pz >= 0) & (np.abs(px) <= pz * np.tan(alpha))
    best = np.full(px.shape, np.inf)
    for sgn in (-1, 1):
        d = np.array([sgn * np.sin(alpha), np.cos(alpha)])
        s = np.maximum(px * d[0] + pz * d[1], 0)
        best = np.minimum(best, np.hypot(px - s * d[0], pz - s * d[1]))
    return np.where(inside, 0.0, best)


def u_slab(R: float, t: float = 16.0) -> dict:
    nx, ny, nz = 100, 24, 112
    cx, alpha, ztop = 50.0, np.radians(25), 94
    zo = 20.0 if R < t else 30.0
    x, z = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
    cav2 = wedge_distance(x, z, cx, zo, alpha) <= R
    nonwm2 = cav2 | (z >= ztop)
    wm = np.broadcast_to((~nonwm2)[:, None, :], (nx, ny, nz)).copy()
    edt = ndi.distance_transform_edt(~wm)
    cortex = ~wm & (edt <= t)
    fluid = ~wm & ~cortex
    hmax = (t - R) / np.sin(alpha) if R < t else 0.0
    return dict(
        name=f"uslab_R{int(R)}",
        wm=wm,
        cortex=cortex,
        fluid=fluid,
        cx=cx,
        zo=zo,
        zmax=zo + hmax,
        R=R,
        t=t,
        y0=ny // 2,
    )


def from_phantom(name: str) -> dict:
    if name == "P4":
        rim, _, e = phantoms.p4_kissing_dark_line()
        known = 25.5
    else:
        rim, _, e, known, _ = phantoms.p8_natural_contact()
    wm, cortex, fluid = rim == 2, rim == 3, rim == 1
    # banks for the oracle weights: the plate a root sits on (by construction)
    nx = rim.shape[0]
    region = e["patch_mask"].any(0)  # (y, z) lines through the contact
    return dict(
        name=name,
        wm=wm,
        cortex=cortex,
        fluid=fluid,
        known=known,
        region=region,
        side_split=nx / 2,
        y0=20,
    )


def omega() -> dict:
    nx, ny, nz = 110, 24, 116
    cx, zc, rc, ztop, tl, tr = 55, 45, 34.0, 92, 12, 18
    x, z = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
    chamber = (x - cx) ** 2 + (z - zc) ** 2 <= rc**2
    neck = (x >= cx - 15) & (x < cx - 15 + tl + tr) & (z >= zc)
    nonwm2 = chamber | neck | (z >= ztop)
    wm = np.broadcast_to((~nonwm2)[:, None, :], (nx, ny, nz)).copy()
    X = np.arange(nx)[:, None, None]
    el = ndi.distance_transform_edt(~(wm & (X < cx)))
    er = ndi.distance_transform_edt(~(wm & (X >= cx)))
    cortex = ~wm & ((el <= tl) | (er <= tr))
    fluid = ~wm & ~cortex
    # the contact: the neck below the WM top (voxel lines along x)
    zz = np.arange(nz)
    zneck = zc + np.sqrt(rc**2 - 15.0**2)
    region = np.broadcast_to(((zz > zneck + 1) & (zz < ztop))[None, :], (ny, nz))
    return dict(
        name="omega",
        wm=wm,
        cortex=cortex,
        fluid=fluid,
        known=cx - 3.5,
        region=region,
        side_split=cx,
        y0=ny // 2,
        tl=tl,
        tr=tr,
    )


def accuracy(s, f, keep) -> dict:
    """Arrival and arc length against the distance to the nearest WM voxel."""
    ex = s.edt[keep]
    ra = np.abs(f.arrival[keep] - ex) / ex
    rl = np.abs(f.length[keep] - ex) / ex
    return dict(
        arrival_median=float(np.median(ra)),
        arrival_p95=float(np.percentile(ra, 95)),
        length_median=float(np.median(rl)),
        length_p95=float(np.percentile(rl, 95)),
    )


def score_lines(pts, ax, region, known) -> dict:
    """Contact pairs stepping along x through the region's (y, z) lines: per
    line, the pair nearest the known contact."""
    k = ax == 0
    yz = np.round(pts[k][:, 1:]).astype(int)
    inr = region[yz[:, 0], yz[:, 1]]
    xs, yz = pts[k][inr, 0], yz[inr]
    nlines = int(region.sum())
    err = {}
    for (y, z), xv in zip(map(tuple, yz), xs, strict=True):
        e = xv - known
        if (y, z) not in err or abs(e) < abs(err[(y, z)]):
            err[(y, z)] = e
    ev = np.array(list(err.values()))
    if ev.size == 0:
        return dict(lines=nlines, found=0)
    q1, q3 = np.percentile(ev, [25, 75])
    return dict(
        lines=nlines,
        found=int(ev.size),
        median=float(np.median(ev)),
        iqr=float(q3 - q1),
        median_abs=float(np.median(np.abs(ev))),
        max_abs=float(np.max(np.abs(ev))),
    )


def pure_lines(plab, region, known) -> dict:
    """The pure partition's label change along x nearest the known contact, per
    (y, z) line of the region (half-voxel resolution)."""
    errs = []
    for y, z in zip(*np.nonzero(region), strict=True):
        line = plab[:, y, z]
        ch = (
            np.flatnonzero((line[:-1] != line[1:]) & (line[:-1] >= 0) & (line[1:] >= 0))
            + 0.5
        )
        if ch.size:
            errs.append(ch[np.argmin(np.abs(ch - known))] - known)
    ev = np.array(errs)
    if ev.size == 0:
        return dict(found=0)
    return dict(
        found=int(ev.size),
        median=float(np.median(ev)),
        median_abs=float(np.median(np.abs(ev))),
    )


def main() -> None:
    cases = [u_slab(6.0), u_slab(22.0), from_phantom("P4"), from_phantom("P8"), omega()]
    results = {}
    for cs in cases:
        s = setup(cs["wm"], cs["cortex"], patch=6)
        keep = cs["cortex"] & (s.edt > 3)
        res: dict = dict(npatch=s.npatch, seeds=int(s.seeds.sum()))
        # arrival accuracy by labelling (unweighted)
        one = np.where(s.seeds, 0, -1).astype(np.int64)
        per = -np.ones(s.seeds.shape, np.int64)
        per[s.seeds] = np.arange(s.seeds.sum())
        res["accuracy"] = {
            name: accuracy(s, run(s, ratio=RATIO, labels=lab), keep)
            for name, lab in (
                ("one label", one),
                ("patches", s.patch),
                ("per voxel", per),
            )
        }
        mu_open, has = open_thickness(s, cs["fluid"], RATIO)
        mu_col, _ = open_thickness(s, cs["fluid"], RATIO, along=True)
        weights = {
            "equal": (np.full(s.npatch, float(np.median(mu_open))),),
            "open": (mu_open,),
            "column tops": (mu_col,),
        }
        if "side_split" in cs:
            left = s.centre[:, 0] < cs["side_split"]
            if cs["name"] == "omega":
                kn = cs["known"]
                # arrivals at the contact from the seed centres on each side
                mu_or = np.where(left, kn - (55 - 16), (55 + 15) - kn)
            else:
                xa = np.nonzero(cs["wm"][:, 20, 40])[0]
                a_seed = xa[xa < cs["side_split"]].max()
                b_seed = xa[xa > cs["side_split"]].min()
                mu_or = np.where(left, cs["known"] - a_seed, b_seed - cs["known"])
            weights["oracle"] = (mu_or.astype(float),)
        res["mu_open_sides"] = None
        if "side_split" in cs:
            left = s.centre[:, 0] < cs["side_split"]
            res["mu_open_sides"] = [
                float(np.median(mu_open[left & has])),
                float(np.median(mu_open[~left & has])),
            ]
        runs = {}
        for wname, (mu,) in weights.items():
            sig = CV * mu
            f = run(s, mu, sig, ratio=RATIO)
            c = contacts(s, f, mu, sig)
            pts = contact_points(c, s.cortex.shape)
            mua, mub = mu[c.a], mu[c.b]
            r: dict = dict(
                pairs=int(c.p.size),
                exact_frac=float(np.mean(c.exact)) if c.p.size else None,
            )
            if c.p.size:
                r["arrival_over_mu_all"] = [
                    float(np.median(c.da / mua)),
                    float(np.median(c.db / mub)),
                ]
                r["sep_median"] = float(
                    np.median(np.where(np.isinf(c.sep), 999, c.sep))
                )
            # the pure weighted Voronoi partition (no crystal growth) as a check
            plab, _ = pure_voronoi(s, mu, sig, RATIO)
            dom = s.domain & (f.label >= 0)
            r["pure_disagree_frac"] = float(np.mean(plab[dom] != f.label[dom]))
            if "known" in cs:
                r["score"] = score_lines(pts, c.axis, cs["region"], cs["known"])
                k = c.axis == 0
                yz = np.round(pts[:, 1:]).astype(int)
                inr = (
                    k
                    & cs["region"][yz[:, 0], yz[:, 1]]
                    & (np.abs(pts[:, 0] - cs["known"]) < 6)
                )
                if inr.any():
                    r["at_contact"] = dict(
                        mu_a=float(np.median(mua[inr])),
                        mu_b=float(np.median(mub[inr])),
                        da=float(np.median(c.da[inr])),
                        db=float(np.median(c.db[inr])),
                        sep=float(
                            np.median(np.where(np.isinf(c.sep[inr]), 999, c.sep[inr]))
                        ),
                    )
                r["pure_score"] = pure_lines(plab, cs["region"], cs["known"])
            else:
                if c.p.size:
                    k = c.axis == 0
                    r["x_err_median"] = (
                        float(np.median(pts[k, 0] - cs["cx"])) if k.any() else None
                    )
                    r["z_range"] = [float(pts[:, 2].min()), float(pts[:, 2].max())]
                r["z_expected"] = [cs["zo"], cs["zmax"]] if cs["R"] < cs["t"] else None
            # display: d / mu in own front, on the slice y0
            d = mu[np.maximum(f.label, 0)] + sig[np.maximum(f.label, 0)] * f.arrival
            runs[wname] = dict(
                d_over_mu=np.where(
                    cs["cortex"], d / mu[np.maximum(f.label, 0)], np.nan
                ),
                pts=pts,
                c=c,
            )
            res[wname] = r
        results[cs["name"]] = res
        figure(cs, s, runs)
        print(cs["name"], json.dumps(res, indent=1))
    (WORK / "synth_contacts.json").write_text(json.dumps(results, indent=1))


def figure(cs, s, runs) -> None:
    y0 = cs["y0"]
    names = list(runs)
    fig, axs = plt.subplots(
        2, len(names), figsize=(4.4 * len(names), 8.6), squeeze=False
    )
    gray = ListedColormap(["#ffffff", "#bdbcb6"])
    # the zoom: round the known contact, or the focal sheet, or the whole slice
    if "known" in cs:
        zz = np.nonzero(cs["region"][y0])[0]
        zc = 0.5 * (zz.min() + zz.max()) if zz.size else cs["wm"].shape[2] / 2
        win = (cs["known"] - 12, cs["known"] + 12, zc - 12, zc + 12)
    elif cs["R"] < cs["t"]:
        win = (cs["cx"] - 14, cs["cx"] + 14, cs["zo"] - 6, cs["zmax"] + 6)
    else:
        win = None
    for row in range(2):
        for ax, wname in zip(axs[row], names):
            rr = runs[wname]
            ax.imshow(cs["wm"][:, y0, :].T, origin="lower", cmap=gray, vmin=0, vmax=1)
            im = ax.imshow(
                rr["d_over_mu"][:, y0, :].T,
                origin="lower",
                cmap="Blues",
                vmin=0,
                vmax=1.4,
            )
            if "known" in cs:
                zz = np.nonzero(cs["region"][y0])[0]
                if zz.size:
                    ax.plot(
                        [cs["known"]] * 2,
                        [zz.min(), zz.max()],
                        "--",
                        color=RED,
                        lw=1.0,
                        zorder=2,
                        label="known contact",
                    )
            elif cs["R"] < cs["t"]:
                ax.plot(
                    [cs["cx"]] * 2,
                    [cs["zo"], cs["zmax"]],
                    "--",
                    color=RED,
                    lw=1.0,
                    zorder=2,
                    label="focal sheet (fold)",
                )
            pts = rr["pts"]
            k = np.abs(pts[:, 1] - y0) <= 0.5
            ax.plot(
                pts[k, 0],
                pts[k, 2],
                "o",
                color=ORANGE,
                ms=3.5 if row else 2.5,
                mec="none",
                zorder=3,
                label="label boundary, roots\n> 3 mm apart on the WM",
            )
            if row == 0:
                ax.set_title(f"{cs['name']}: {wname} weights", fontsize=10)
            elif win is not None:
                ax.set_xlim(win[0], win[1])
                ax.set_ylim(win[2], win[3])
                ax.set_title("zoom", fontsize=9)
            ax.set_xlabel("x (voxels)")
            ax.set_ylabel("z (voxels)")
    axs[0][0].legend(loc="upper left", fontsize=7, framealpha=0.9)
    fig.colorbar(
        im, ax=axs.ravel().tolist(), shrink=0.6, label="arrival in own front / its mu"
    )
    fig.suptitle(
        f"{cs['name']}: one-pass weighted competition (3D; slice y = {y0}; WM grey, fluid white)",
        fontsize=10,
    )
    fig.savefig(FIG / f"onepass_synth_{cs['name']}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
