"""Figures of a tube-solver run.

usage: tube_figures.py SITE_NPZ RUN_DIR OUT_DIR
Reads RUN_DIR/log.json, RUN_DIR/snapshots/*.npz, and (when present)
RUN_DIR/result.npz and RUN_DIR/diagnostics.json. Writes PNGs to OUT_DIR:
- evolution.png: τ on the three slices through the site at each snapshot;
- levelsets.png: τ level sets over echo 2 on the three slices (last snapshot);
- tubes.png: the tubes (owner, random colours) on the three slices;
- events.png: contact voxels, stopped tubes' fronts and retreats on the slices;
- sweeps.png: sweeps per step, Newton iterations, max change, counts, energy.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TAU_LEVELS = np.arange(-3.0, 0.001, 0.25)


def load_snap(p: Path, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    s = np.load(p)
    tau = np.full(shape, np.nan, np.float32)
    own = np.full(shape, -1, np.int64)
    i = np.unravel_index(s["index"], shape)
    tau[i], own[i] = s["tau"], s["owner"]
    return tau, own


def sl(a: np.ndarray, ax: int, c: int) -> np.ndarray:
    idx = [slice(None)] * 3
    idx[ax] = c
    return a[tuple(idx)].T


def base(axh, img: np.ndarray, labels: np.ndarray) -> None:
    axh.imshow(img, cmap="gray", origin="lower", vmin=np.nanpercentile(img, 1), vmax=np.nanpercentile(img, 99.5))
    axh.contour(labels == 0, [0.5], colors="deepskyblue", linewidths=0.5)
    axh.contour(labels == 41, [0.5], colors="cyan", linewidths=0.4, linestyles=":")
    axh.set_xticks([])
    axh.set_yticks([])


def main() -> None:
    site_p, run, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(site_p)
    labels, e2 = z["labels"], z["echoes"][1].astype(float)
    shape = labels.shape
    cv = np.round((np.linalg.inv(z["affine"]) @ np.r_[z["site_mm"], 1])[:3]).astype(int)
    log = json.loads((run / "log.json").read_text()) if (run / "log.json").exists() else None
    snaps = sorted((run / "snapshots").glob("*.npz"))
    names = [p.stem for p in snaps]
    axn = "xyz"

    # evolution: up to 8 snapshots evenly spaced, three slices each
    pick = np.unique(np.linspace(0, len(snaps) - 1, min(8, len(snaps))).round().astype(int))
    fig, axs = plt.subplots(3, len(pick), figsize=(2.6 * len(pick), 8), squeeze=False)
    for col, k in enumerate(pick):
        tau, _ = load_snap(snaps[k], shape)
        for ax in range(3):
            a = axs[ax, col]
            base(a, sl(e2, ax, cv[ax]), sl(labels, ax, cv[ax]))
            t = sl(tau, ax, cv[ax])
            a.imshow(np.ma.masked_invalid(t), cmap="viridis", vmin=-3, vmax=0.2, origin="lower", alpha=0.6)
            a.contour(np.nan_to_num(t, nan=1.0), [0.0], colors="red", linewidths=0.8)
            if ax == 0:
                a.set_title(names[k], fontsize=9)
            if col == 0:
                a.set_ylabel(f"{axn[ax]} = {cv[ax]}")
    fig.suptitle("τ (colour, mm; inner voxels < 0 inside the WM) and the front τ = 0 (red) per snapshot")
    fig.tight_layout()
    fig.savefig(out / "evolution.png", dpi=90)
    plt.close(fig)

    tau, own = load_snap(snaps[-1], shape)
    # level sets
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.6))
    for ax in range(3):
        a = axs[ax]
        base(a, sl(e2, ax, cv[ax]), sl(labels, ax, cv[ax]))
        t = sl(tau, ax, cv[ax])
        cs = a.contour(np.nan_to_num(t, nan=9.0), TAU_LEVELS[:-1], cmap="plasma", linewidths=0.6)
        a.contour(np.nan_to_num(t, nan=9.0), [0.0], colors="red", linewidths=1.2)
        a.plot(cv[[i for i in range(3) if i != ax][0]], cv[[i for i in range(3) if i != ax][1]], "y+", ms=12)
        a.set_title(f"{axn[ax]} = {cv[ax]}")
    fig.colorbar(cs, ax=axs, shrink=0.7, label="τ (mm below the front)")
    fig.suptitle(f"τ level sets every 0.25 mm over echo 2, front τ = 0 red ({names[-1]}); blue: label 0; cyan dotted: WM label")
    fig.savefig(out / "levelsets.png", dpi=110)
    plt.close(fig)

    # tubes
    rng = np.random.default_rng(0)
    cmap = rng.uniform(0.15, 1.0, (int(own.max()) + 2, 3))
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.6))
    for ax in range(3):
        a = axs[ax]
        base(a, sl(e2, ax, cv[ax]), sl(labels, ax, cv[ax]))
        o = sl(own, ax, cv[ax])
        rgba = np.zeros(o.shape + (4,))
        m = o >= 0
        rgba[m, :3] = cmap[o[m]]
        rgba[m, 3] = 0.55
        a.imshow(rgba, origin="lower")
        a.set_title(f"{axn[ax]} = {cv[ax]}")
    fig.suptitle(f"tubes (one random colour per white-surface patch) over echo 2 ({names[-1]})")
    fig.tight_layout()
    fig.savefig(out / "tubes.png", dpi=110)
    plt.close(fig)

    # events
    if log is not None:
        ev = log["events"]
        P = 4  # tube_kernels.PAD
        sp = np.array(shape) + 2 * P

        def vox(v: list[int]) -> np.ndarray:
            if not v:
                return np.zeros((0, 3), int)
            return np.stack(np.unravel_index(np.array(v), sp), 1) - P

        cvox = vox([e["voxel"] for e in ev["contact"]])
        rvox = vox([e["voxel"] for e in ev["retreat"]])
        stopped = {e["tube"] for e in ev["stop"]}
        fig, axs = plt.subplots(2, 3, figsize=(16, 10.5))
        for ax in range(3):
            a = axs[0, ax]
            base(a, sl(e2, ax, cv[ax]), sl(labels, ax, cv[ax]))
            t = sl(tau, ax, cv[ax])
            a.contour(np.nan_to_num(t, nan=9.0), [0.0], colors="red", linewidths=0.8)
            o = sl(own, ax, cv[ax])
            st = np.isin(o, list(stopped)) & (np.abs(np.nan_to_num(t, nan=9)) < 0.2)
            yy, xx = np.nonzero(st)
            a.plot(xx, yy, ".", color="dodgerblue", ms=2, label="front of stopped tubes")
            oth = [i for i in range(3) if i != ax]
            for arr, colr, lab in ((cvox, "magenta", "contact voxels (all steps)"), (rvox, "yellow", "retreats (all steps)")):
                near = np.abs(arr[:, ax] - cv[ax]) <= 1
                a.plot(arr[near, oth[0]], arr[near, oth[1]], "x", color=colr, ms=3, label=lab)
            a.set_title(f"{axn[ax]} = {cv[ax]} (±1 voxel for events)")
        axs[0, 0].legend(loc="lower left", fontsize=7)
        steps = [s for s in log["steps"] if isinstance(s["step"], int) and s["step"] > 0]
        st_n = np.array([s["step"] for s in steps])
        a = axs[1, 0]
        nret = np.bincount([e["step"] for e in ev["retreat"]], minlength=st_n.max() + 1 if len(st_n) else 1)
        nadv = np.bincount([e["step"] for e in ev["advance"]], minlength=len(nret))
        a.bar(np.arange(len(nret)) - 0.2, nadv, 0.4, label="advances (solve)")
        a.bar(np.arange(len(nret)) + 0.2, nret, 0.4, label="retreats (solve)")
        a.set_xlabel("step")
        a.legend()
        a.set_title("sign changes in the solves")
        a = axs[1, 1]
        a.plot(st_n, [s["stopped"] for s in steps], "o-", label="stopped tubes")
        a.plot(st_n, [s["contacts_new"] for s in steps], "s-", label="new contact pairs")
        a.plot(st_n, [s["predict_blocked"] for s in steps], "^-", label="prediction blocked (topology)")
        a.set_xlabel("step")
        a.legend()
        a.set_title("stops, contacts, topology blocks")
        a = axs[1, 2]
        dpath = run / "diagnostics.json"
        if dpath.exists():
            d = json.loads(dpath.read_text())
            rm = [c["root_mm"] for c in d["contacts"] if np.isfinite(c["root_mm"])]
            cc = [c["signature_corr"] for c in d["contacts"] if np.isfinite(c["signature_corr"])]
            a.hist(rm, bins=30, alpha=0.7, label=f"root distance along white surface (mm), n={len(rm)}")
            a2 = a.twiny()
            a2.hist(cc, bins=30, range=(-1, 1), color="C1", alpha=0.5, label=f"laminar signature corr, n={len(cc)}")
            a.legend(loc="upper left", fontsize=7)
            a2.legend(loc="upper right", fontsize=7)
            a.set_title("contact measures")
        fig.tight_layout()
        fig.savefig(out / "events.png", dpi=100)
        plt.close(fig)

        # sweeps
        fig, axs = plt.subplots(2, 3, figsize=(16, 9))
        a = axs[0, 0]
        a.bar(st_n - 0.2, [len(s["sweeps"]) for s in steps], 0.4, label="window sweeps")
        a.bar(st_n + 0.2, [len(s.get("full_sweeps", [])) for s in steps], 0.4, label="full-depth sweeps")
        a.axhline(log["settings"]["sweep_max"], color="k", ls=":", lw=0.8)
        a.set_xlabel("step")
        a.set_title("sweeps per step (limit dotted)")
        a.legend()
        a = axs[0, 1]
        for s in steps:
            mc = [x["max_change"] for x in s["sweeps"]]
            a.semilogy(np.arange(1, len(mc) + 1), np.maximum(mc, 1e-6), "-o", ms=3, alpha=0.6)
        a.axhline(log["settings"]["sweep_tol_mm"], color="k", ls=":")
        a.set_xlabel("sweep")
        a.set_title("max |Δτ| per sweep (mm), one line per step")
        a = axs[0, 2]
        a.plot(st_n, [np.mean([x["newton_mean"] for x in s["sweeps"]]) for s in steps], "o-", label="mean")
        a.plot(st_n, [max(x["newton_max"] for x in s["sweeps"]) for s in steps], "s-", label="max")
        a.axhline(log["settings"]["newton_iters"], color="k", ls=":")
        a.set_title("Newton iterations per tube solve")
        a.legend()
        a = axs[1, 0]
        a.plot(st_n, [s["tissue"] for s in steps], "o-")
        a.set_title("tissue voxels (τ < 0)")
        a = axs[1, 1]
        a.plot(st_n, [s["seconds"] / 60 for s in steps], "o-")
        a.set_title("minutes per step")
        a = axs[1, 2]
        allst = [s for s in log["steps"] if "energy" in s and isinstance(s["step"], int)]
        for k in ("stop", "align", "smooth", "equi", "pull", "thick", "tsmooth"):
            a.plot([s["step"] for s in allst], [s["energy"][k] for s in allst], "-o", ms=3, label=k)
        a.set_yscale("symlog", linthresh=100)
        a.legend(fontsize=7)
        a.set_title("energy by term (nats)")
        fig.tight_layout()
        fig.savefig(out / "sweeps.png", dpi=100)
        plt.close(fig)
    print("figures in", out)


if __name__ == "__main__":
    main()
