"""Phase 1.2, Part D: the choice of sigma for orient's local cortex level, by
two views at the test sites.

1. Leakage: along the normal rays (rays.RaySampler, method normal) of the
   roots of each site's superblock, over the first run of cortex outward
   from the root, the level's correlation with depth, and the level's
   standard deviation along the ray over the log echo's.
2. Borders: at the V1V2 site, the stria of Gennari marks V1. Per ray, the
   stria contrast of log echo 3 (mean at depth 0.35-0.55 minus the mean at
   0.10-0.30 and 0.65-0.85; dark is negative). Two classes by k-means on
   the contrast; the border is between them; each root's signed distance to
   the nearest root of the other class (positive in V1). Per sigma, the
   level at mid-depth (0.3-0.7) against that distance, and the log echo's
   for reference.

usage:
    orient_sigma.py ECHO1 ECHO2 ECHO3 --dseg DSEG --surf SURF --out OUT --fig FIG
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from baillarger import orient as O
from baillarger.rays import RaySampler
from baillarger.timing import peak_rss_gb

SITES = {
    "V1V2": (268, 128, 135),
    "M1S1": (455, 250, 496),
    "PCC_SPL": (514, 180, 422),
}
SIGMAS = (0.32, 0.64, 0.96, 1.28)
LEVEL = 1
E_OUT = 5.0
STEP = 0.08
MIN_RUN = 1.5  # mm of cortex along a ray
NBINS = 20  # depth bins of the saved echo-3 profile


def runs(rays, lab_at):  # type: ignore[no-untyped-def]
    """Per column: the sample indices of the first cortex run outward from
    the root (leading non-cortex samples within 0.5 mm skipped)."""
    out = []
    for c in range(len(rays.roots.vertex)):
        a, b = rays.sample_start[c], rays.sample_start[c + 1]
        s = rays.s[a:b]
        lab = lab_at[a:b]
        k0 = np.searchsorted(s, 0.0)
        k = k0
        while k < len(s) and not lab[k] and s[k] < 0.5:
            k += 1
        k1 = k
        while k1 < len(s) and lab[k1]:
            k1 += 1
        if k1 > k and s[k1 - 1] - s[k] >= MIN_RUN:
            out.append((c, a + k, a + k1))
        else:
            out.append((c, -1, -1))
    return out


def child_rays(s, rs, inv, sb, device):  # type: ignore[no-untyped-def]
    """Per owned ray with a cortex run, on the level-0 superblock sb: the
    root, the stria contrast, the raw mid-depth logs (3,), and per sigma
    the corr (3,), the leak ratio (3,) and the mid-depth level (3,)."""
    rays = rs.sample(sb, 0, e_in=0.0, e_out=E_OUT, step=STEP)
    own = rays.roots.owned
    halo = E_OUT + 0.5
    box = s.output_box(sb, 0, halo)
    lo = np.array([b.start for b in box])
    hi = np.array([b.stop for b in box])
    v = rays.position.astype(np.float64) @ inv[:3, :3].T + inv[:3, 3]
    idx = np.floor(v + 0.5).astype(np.int64)
    inb = np.all((idx >= lo) & (idx < hi), 1)
    idx = np.clip(idx, lo, hi - 1) - lo
    ii = tuple(idx.T)
    e, ctx = s.read(box)
    lab_at = ctx[ii] & inb
    logs = O.log_echo(e)[(slice(None), *ii)]  # (3, N) at the samples
    del e, ctx
    rr = [r for r in runs(rays, lab_at) if r[1] >= 0 and own[r[0]]]
    n = len(rr)
    out = {
        "root": rays.roots.root[[r[0] for r in rr]].astype(np.float64).reshape(n, 3),
        "contrast": np.zeros(n),
        "rawmid": np.zeros((n, 3)),
        "profile": np.zeros((n, NBINS)),
        "length": np.zeros(n),
        "corr": np.zeros((len(SIGMAS), n, 3)),
        "leak": np.zeros((len(SIGMAS), n, 3)),
        "mid": np.zeros((len(SIGMAS), n, 3)),
    }
    depths = []
    for q, (_c, k0, k1) in enumerate(rr):
        ss = rays.s[k0:k1]
        depth = (ss - ss[0]) / (ss[-1] - ss[0])
        depths.append(depth)
        Y = logs[2][k0:k1]
        band = (depth >= 0.35) & (depth <= 0.55)
        side = ((depth >= 0.1) & (depth <= 0.3)) | ((depth >= 0.65) & (depth <= 0.85))
        out["contrast"][q] = Y[band].mean() - Y[side].mean()
        m = (depth >= 0.3) & (depth <= 0.7)
        out["length"][q] = ss[-1] - ss[0]
        out["profile"][q] = np.interp((np.arange(NBINS) + 0.5) / NBINS, depth, Y)
        out["rawmid"][q] = [logs[ech][k0:k1][m].mean() for ech in range(3)]
    for j, sig in enumerate(SIGMAS):
        lv = s.level(sb, 0, halo, sig, device).level[(slice(None), *ii)]
        for q, (_c, k0, k1) in enumerate(rr):
            depth = depths[q]
            m = (depth >= 0.3) & (depth <= 0.7)
            for ech in range(3):
                L = lv[ech][k0:k1].astype(np.float64)
                Y = logs[ech][k0:k1]
                if np.std(L) > 0:
                    out["corr"][j, q, ech] = np.corrcoef(L, depth)[0, 1]
                out["leak"][j, q, ech] = np.std(L) / max(np.std(Y), 1e-12)
                out["mid"][j, q, ech] = np.mean(L[m])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("echoes", nargs=3, type=Path)
    ap.add_argument("--dseg", required=True, type=Path)
    ap.add_argument("--surf", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fig", required=True, type=Path)
    ap.add_argument("--device", default="numpy", help="numpy, cpu or mps")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sites", nargs="*", default=list(SITES))
    ap.add_argument("--figures-only", action="store_true")
    a = ap.parse_args()
    import torch

    torch.set_num_threads(a.threads)
    if a.figures_only:
        a.sites = []
    a.out.mkdir(parents=True, exist_ok=True)
    a.fig.mkdir(parents=True, exist_ok=True)
    s = O.OrientSampler(a.echoes, a.dseg, cache_bytes=256 << 20)
    rs = RaySampler(a.surf, a.dseg, cache_bytes=256 << 20)
    inv = np.linalg.inv(s.affine)
    result: dict = {}
    old = a.out / "sigma.json"
    if old.exists():
        result = json.loads(old.read_text())
    for name in a.sites:
        centre = SITES[name]
        t = time.perf_counter()
        sb1 = tuple(int(c) // (64 << LEVEL) for c in centre)
        parts = []
        for dz in range(2):
            for dy in range(2):
                for dx in range(2):
                    sb = (2 * sb1[0] + dx, 2 * sb1[1] + dy, 2 * sb1[2] + dz)
                    parts.append(child_rays(s, rs, inv, sb, a.device))
                    print(
                        name,
                        sb,
                        len(parts[-1]["contrast"]),
                        f"{time.perf_counter() - t:.0f} s",
                        flush=True,
                    )
        r = {
            k: np.concatenate(
                [p[k] for p in parts], axis=1 if k in ("corr", "leak", "mid") else 0
            )
            for k in parts[0]
        }
        site: dict = {"superblock_level1": list(sb1), "rays": len(r["contrast"])}
        for j, sig in enumerate(SIGMAS):
            site[f"sigma_{sig}"] = {
                "median_abs_corr": np.median(np.abs(r["corr"][j]), 0).tolist(),
                "median_leak_ratio": np.median(r["leak"][j], 0).tolist(),
                "p90_leak_ratio": np.percentile(r["leak"][j], 90, 0).tolist(),
            }
            print(name, sig, site[f"sigma_{sig}"], flush=True)
        if name == "V1V2":
            np.savez_compressed(a.out / "v1v2_rays.npz", **r)
        site["seconds"] = time.perf_counter() - t
        site["peak_gb"] = peak_rss_gb()
        result[name] = site
        (a.out / "sigma.json").write_text(json.dumps(result, indent=1))
    if (a.out / "v1v2_rays.npz").exists():
        result["V1V2"].update(border(a.out / "v1v2_rays.npz", a.out))
        (a.out / "sigma.json").write_text(json.dumps(result, indent=1))
    figures(a.out, a.fig)


def stria_score(
    root: np.ndarray, profile: np.ndarray, length: np.ndarray
) -> np.ndarray:
    """The stria score of each ray: the echo-3 profile (minus its mean over
    depth 0-0.85, which leaves out the drop at the outer edge) is averaged
    over the rays of single thickness (length 1.8-3.5 mm) in a 1.5 mm box
    (3 x 3 x 3 bins of 0.5 mm) round the root; the score is its mean at
    depth 0.25-0.45 minus the mean of its means at 0.05-0.2 and 0.5-0.65.
    A dark band at the stria's depth gives a negative score. NaN where the
    box holds fewer than 20 such rays."""
    from scipy import ndimage

    nb = profile.shape[1]
    D = profile - profile[:, : int(0.85 * nb)].mean(1, keepdims=True)
    ok = (length >= 1.8) & (length <= 3.5)
    g = np.floor((root - root.min(0)) / 0.5).astype(np.int64)
    shp = tuple(int(x) for x in g.max(0) + 1)
    lin = np.ravel_multi_index(tuple(g.T), shp)
    S = np.zeros((int(np.prod(shp)), nb))
    N = np.zeros(int(np.prod(shp)))
    np.add.at(S, lin[ok], D[ok])
    np.add.at(N, lin[ok], 1)
    S = ndimage.uniform_filter(S.reshape(*shp, nb), size=(3, 3, 3, 1), mode="constant")
    N = ndimage.uniform_filter(N.reshape(shp), size=3, mode="constant") * 27
    Dm = S.reshape(-1, nb)[lin] * 27 / np.maximum(N.reshape(-1)[lin], 1e-9)[:, None]
    Dm[N.reshape(-1)[lin] < 20] = np.nan
    x = (np.arange(nb) + 0.5) / nb
    band = (x >= 0.25) & (x <= 0.45)
    lo = (x >= 0.05) & (x <= 0.2)
    hi = (x >= 0.5) & (x <= 0.65)
    return np.asarray(
        Dm[:, band].mean(1) - 0.5 * (Dm[:, lo].mean(1) + Dm[:, hi].mean(1))
    )


#: the classes of the stria score: the stria (V1) at or below V1_MAX, no
#: stria at or above OTHER_MIN; the rays between are left out
V1_MAX = -0.17
OTHER_MIN = -0.05


def border(path: Path, out: Path) -> dict:
    """The V1/V2 border at the V1V2 site and the ramps across it."""
    from scipy.spatial import cKDTree

    r = np.load(path)
    score = stria_score(r["root"], r["profile"], r["length"])
    v1 = score <= V1_MAX
    other = score >= OTHER_MIN
    roots = r["root"]
    dist = np.full(len(score), np.nan)
    dist[v1] = cKDTree(roots[other]).query(roots[v1])[0]
    dist[other] = -cKDTree(roots[v1]).query(roots[other])[0]
    bins = np.arange(-3.0, 3.01, 0.25)
    ramp = {}
    series = [("raw", r["rawmid"])] + [
        (str(sig), r["mid"][j]) for j, sig in enumerate(SIGMAS)
    ]
    for key, mid in series:
        rows = []
        for b0, b1 in zip(bins[:-1], bins[1:], strict=True):
            m = (dist >= b0) & (dist < b1)
            rows.append(np.median(mid[m], 0).tolist() if m.sum() >= 20 else [None] * 3)
        ramp[key] = rows
    # the areal step: median at 1.5-3 mm into V1 minus median at 1.5-3 mm
    # out of it, per echo; and its share of the raw step
    far_in = (dist >= 1.5) & (dist <= 3.0)
    far_out = (dist <= -1.5) & (dist >= -3.0)
    step = {
        k: (np.median(m[far_in], 0) - np.median(m[far_out], 0)).tolist()
        for k, m in series
    }
    share = {
        k: (np.asarray(v) / np.asarray(step["raw"])).tolist() for k, v in step.items()
    }
    prof = {}
    for nm, m in (("v1", v1), ("other", other)):
        prof[nm] = np.mean(
            r["profile"][m & (r["length"] >= 1.8) & (r["length"] <= 3.5)], 0
        ).tolist()
    np.savez_compressed(
        out / "v1v2_border.npz", score=score, v1=v1, other=other, dist=dist
    )
    return {
        "stria": {
            "V1Max": V1_MAX,
            "OtherMin": OTHER_MIN,
            "v1_rays": int(v1.sum()),
            "other_rays": int(other.sum()),
            "unclassified_rays": int((~v1 & ~other).sum()),
            "score_percentiles_5_25_50_75_95": np.nanpercentile(
                score, [5, 25, 50, 75, 95]
            ).tolist(),
            "mean_profile_echo3": prof,
        },
        "ramp_bins": bins.tolist(),
        "ramp": ramp,
        "step": step,
        "step_share": share,
    }


def figures(out: Path, fig: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = json.loads((out / "sigma.json").read_text())
    names = list(r)
    colors = ["#1f5f99", "#c25e1f", "#2e8540"]
    f, ax = plt.subplots(1, 2, figsize=(10, 3.8), dpi=90)
    for i, n in enumerate(names):
        for ech, ls in zip(range(3), ("-", "--", ":"), strict=True):
            ax[0].plot(
                SIGMAS,
                [r[n][f"sigma_{s}"]["median_abs_corr"][ech] for s in SIGMAS],
                ls,
                color=colors[i],
                marker="o",
                ms=4,
                label=f"{n} echo {ech + 1}",
            )
            ax[1].plot(
                SIGMAS,
                [r[n][f"sigma_{s}"]["median_leak_ratio"][ech] for s in SIGMAS],
                ls,
                color=colors[i],
                marker="o",
                ms=4,
            )
    ax[0].set_xlabel("sigma, mm")
    ax[0].set_ylabel("median |corr(level, depth)|")
    ax[1].set_xlabel("sigma, mm")
    ax[1].set_ylabel("median sd(level) / sd(log echo) along a ray")
    ax[0].legend(fontsize=7, ncol=1, frameon=False)
    for x in ax:
        x.grid(alpha=0.3)
    f.suptitle("Leakage of the laminae into the local cortex level", fontsize=10)
    f.tight_layout()
    f.savefig(fig / "leakage.png", bbox_inches="tight")
    if "V1V2" in r and "ramp" in r["V1V2"]:
        v = r["V1V2"]
        bins = np.array(v["ramp_bins"])
        x = 0.5 * (bins[:-1] + bins[1:])
        f, ax = plt.subplots(1, 4, figsize=(16, 3.6), dpi=90)
        pr = v["stria"]["mean_profile_echo3"]
        dx = (np.arange(len(pr["v1"])) + 0.5) / len(pr["v1"])
        ax[0].plot(
            dx,
            pr["v1"],
            color="#b2182b",
            lw=2,
            label=f"stria score <= {v['stria']['V1Max']} (V1)",
        )
        ax[0].plot(
            dx,
            pr["other"],
            color="#2166ac",
            lw=2,
            label=f"stria score >= {v['stria']['OtherMin']}",
        )
        ax[0].set_xlabel("depth along the ray (0 at the white surface)")
        ax[0].set_ylabel("log echo 3, mean")
        ax[0].set_title("how the border was found: the two classes", fontsize=9)
        ax[0].legend(fontsize=7, frameon=False)
        ax[0].grid(alpha=0.3)
        ax = ax[1:]
        for ech in range(3):
            for k, sig in enumerate(["raw"] + [str(s) for s in SIGMAS]):
                y = np.array(
                    [np.nan if row[ech] is None else row[ech] for row in v["ramp"][sig]]
                )
                y = y - np.nanmedian(y[x < -2])
                lbl = (
                    "log echo, mid-depth" if sig == "raw" else f"level, sigma {sig} mm"
                )
                ax[ech].plot(
                    x,
                    y,
                    color="#444444" if sig == "raw" else plt.cm.viridis(k / 5),
                    lw=2 if sig == "raw" else 1.5,
                    label=lbl,
                )
            ax[ech].axvline(0, color="#999999", lw=0.8)
            ax[ech].set_title(f"echo {ech + 1}", fontsize=9)
            ax[ech].set_xlabel("distance to the stria border, mm (V1 > 0)")
            ax[ech].grid(alpha=0.3)
        ax[0].set_ylabel("value - its median beyond 2 mm out of V1")
        ax[0].legend(fontsize=7, frameon=False)
        f.suptitle(
            "V1V2: the level across the V1/V2 border (where the stria ends)",
            fontsize=10,
        )
        f.tight_layout()
        f.savefig(fig / "border.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
