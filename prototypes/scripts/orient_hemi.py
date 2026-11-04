"""Phase 1.2, Part E: orient over all the cortex superblocks of one
hemisphere, and its diagnostics.

Per superblock at LEVEL (Morton order): orient (numpy, with each echo's
tensor), then on its own cortex voxels and its owned white-surface roots:

- the surface agreement (orient_diag.surface_agreement) of each owned root,
  with the root's ray normal (rays.Roots.normal);
- each echo's share of J's trace, by distance from the WM (the depth);
- the angle between each echo's normal and the combined normal, and between
  the echoes 2 and 3, as axes, in interior cortex (cortex voxels whose 6
  face neighbours are cortex);
- per array axis, the agreement, the coherence and the echo 2/3 agreement
  against the angle of the normal to that axis.

All are histograms (counts and sums), for the whole hemisphere and for each
site superblock. The per-root angles are saved, to map on the overview mesh.

usage:
    orient_hemi.py ECHO1 ECHO2 ECHO3 --dseg DSEG --surf SURF --noise NOISE
        --out OUT [--limit N] [--only I,J,K ...]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

from baillarger import orient as O
from baillarger.orient_diag import surface_agreement
from baillarger.orient_noise import superblocks
from baillarger.rays import RaySampler
from baillarger.timing import peak_rss_gb

LEVEL = 1
SITES = {
    "V1V2": (268, 128, 135),
    "M1S1": (455, 250, 496),
    "PCC_SPL": (514, 180, 422),
}
DEPTH_BINS = np.arange(0.0, 5.01, 0.25)  # mm from the WM
ANGLE_BINS = np.arange(0.0, 90.01, 2.0)  # degrees, axis angle between normals
AXIS_BINS = np.arange(0.0, 90.01, 10.0)  # degrees, normal to array axis
AGREE_BINS = np.arange(0.0, 180.01, 2.0)
PAIRS_NAMES = ("e1", "e2", "e3", "e2e3")


def axis_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The angle in degrees between the axes a and b, (..., 3), in [0, 90]."""
    d = np.abs(np.einsum("...a,...a->...", a, b))
    return np.degrees(np.arccos(np.clip(d, 0.0, 1.0)))


class Acc:
    """Histograms for one region."""

    def __init__(self) -> None:
        nd, na, nx = len(DEPTH_BINS) - 1, len(ANGLE_BINS) - 1, len(AXIS_BINS) - 1
        self.trace = np.zeros((nd, 3))  # sum of each echo's trace share
        self.trace_n = np.zeros(nd)
        self.angle = np.zeros((4, nd, na))  # e1, e2, e3 vs combined; e2 vs e3
        self.coh_axis = np.zeros((3, nx))
        self.coh_axis_n = np.zeros((3, nx))
        self.e23_axis = np.zeros((3, nx))  # count within 10 degrees
        self.e23_axis_n = np.zeros((3, nx))
        self.agree = np.zeros(len(AGREE_BINS) - 1)
        self.agree_axis = np.zeros((3, nx, len(AGREE_BINS) - 1))

    def add(self, other: "Acc") -> None:
        for k, v in vars(other).items():
            getattr(self, k).__iadd__(v)

    def json(self) -> dict:
        return {k: v.tolist() for k, v in vars(self).items()}


def measure(s, rs, sb, acc: Acc) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # type: ignore[no-untyped-def]
    vs = float(np.min(s.voxel_mm))
    halo = 2 * vs
    o = s.orient(sb, LEVEL, halo, echo_tensor=True)
    core = s.output_box(sb, LEVEL, 0.0)
    crop = tuple(
        slice(c.start - b.start, c.stop - b.start)
        for c, b in zip(core, o.box, strict=True)
    )
    # agreement of the owned roots
    roots = rs.roots(sb, LEVEL)
    own = roots.owned
    ang, w = surface_agreement(
        roots.root[own],
        roots.normal[own],
        o.normal,
        o.coherence,
        o.sign,
        o.box,
        s.affine,
    )
    vtx = roots.vertex[own]
    nrm = roots.normal[own].astype(np.float64)
    ok = np.isfinite(ang)
    acc.agree += np.histogram(ang[ok], AGREE_BINS)[0]
    for a in range(3):
        ax = np.degrees(np.arccos(np.clip(np.abs(nrm[ok, a]), 0, 1)))
        bi = np.clip(np.digitize(ax, AXIS_BINS) - 1, 0, len(AXIS_BINS) - 2)
        gi = np.clip(np.digitize(ang[ok], AGREE_BINS) - 1, 0, len(AGREE_BINS) - 2)
        np.add.at(acc.agree_axis[a], (bi, gi), 1)
    # voxels of the core box
    n = np.moveaxis(o.normal[(slice(None), *crop)], 0, -1).astype(np.float64)
    coh = o.coherence[crop].astype(np.float64)
    ctx = np.isfinite(coh)
    if not ctx.any():
        return vtx, ang, w
    inner = ndimage.binary_erosion(np.isfinite(o.coherence))[crop] & ctx
    d = s.distance.sample(sb, LEVEL, 0.0, O.SIGN_CAP)
    dc = tuple(
        slice(c.start - b.start, c.stop - b.start)
        for c, b in zip(core, d.box, strict=True)
    )
    dep = d.distance[dc].astype(np.float64)
    good = ctx & np.isfinite(dep) & (dep >= 0)
    di = np.clip(np.digitize(dep, DEPTH_BINS) - 1, 0, len(DEPTH_BINS) - 2)
    et = o.echo_tensor[(slice(None), slice(None), *crop)].astype(np.float64)  # type: ignore[index]
    tr = et[:, 0] + et[:, 3] + et[:, 5]  # PAIRS: xx xy xz yy yz zz
    tot = tr.sum(0)
    share = np.where(tot > 0, tr / np.where(tot > 0, tot, 1), np.nan)
    g = good & np.all(np.isfinite(share), 0)
    for e in range(3):
        np.add.at(acc.trace[:, e], di[g], share[e][g])
    np.add.at(acc.trace_n, di[g], 1)
    en = []
    for e in range(3):
        ne, _ = O.leading_any(et[e])
        en.append(np.asarray(ne, np.float64))
    gi = inner & good
    for k, (a, b) in enumerate(((en[0], n), (en[1], n), (en[2], n), (en[1], en[2]))):
        an = axis_angle(a[gi], b[gi])
        ai = np.clip(np.digitize(an, ANGLE_BINS) - 1, 0, len(ANGLE_BINS) - 2)
        np.add.at(acc.angle[k], (di[gi], ai), 1)
    a23 = axis_angle(en[1][gi], en[2][gi])
    for a in range(3):
        ax = np.degrees(np.arccos(np.clip(np.abs(n[..., a]), 0, 1)))
        bi = np.clip(np.digitize(ax, AXIS_BINS) - 1, 0, len(AXIS_BINS) - 2)
        np.add.at(acc.coh_axis[a], bi[gi], coh[gi])
        np.add.at(acc.coh_axis_n[a], bi[gi], 1)
        np.add.at(acc.e23_axis[a], bi[gi], (a23 <= 10.0).astype(float))
        np.add.at(acc.e23_axis_n[a], bi[gi], 1)
    return vtx, ang, w


def site_slices(s, sb, out: Path, name: str, centre) -> None:  # type: ignore[no-untyped-def]
    """The departure of each echo's normal at the site's centre slice (k)."""
    o = s.orient(sb, LEVEL, 0.0, echo_tensor=True)
    k = int(centre[2]) - o.box[2].start
    n = np.moveaxis(o.normal[:, :, :, k], 0, -1).astype(np.float64)
    dep = []
    for e in range(3):
        ne, _ = O.leading_any(o.echo_tensor[e][:, :, :, k].astype(np.float64))  # type: ignore[index]
        dep.append(axis_angle(np.asarray(ne), n))
    e, _ = s.read(o.box)
    np.savez_compressed(
        out / f"site_{name}.npz",
        coherence=o.coherence[:, :, k],
        departure=np.stack(dep),
        echo3=e[2][:, :, k],
        sign=o.sign[:, :, k],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("echoes", nargs=3, type=Path)
    ap.add_argument("--dseg", required=True, type=Path)
    ap.add_argument("--surf", required=True, type=Path)
    ap.add_argument("--noise", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=[])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    s = O.OrientSampler(a.echoes, a.dseg, cache_bytes=256 << 20, noise=a.noise)
    rs = RaySampler(a.surf, a.dseg, cache_bytes=256 << 20)
    side = s.cells << LEVEL
    site_sb = {n: tuple(int(c) // side for c in v) for n, v in SITES.items()}
    if a.only:
        sbs = [tuple(int(x) for x in t.split(",")) for t in a.only]
    else:
        sbs = list(superblocks(s, LEVEL))
    if a.limit:
        sbs = sbs[: a.limit]
    parts = a.out / "superblocks"
    parts.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    for sb in sbs:
        f = parts / f"{sb[0]}_{sb[1]}_{sb[2]}.npz"
        if f.exists():
            continue
        t = time.perf_counter()
        acc = Acc()
        v, an, w = measure(s, rs, sb, acc)
        for n, ssb in site_sb.items():
            if ssb == sb:
                site_slices(s, sb, a.out, n, SITES[n])
        dt = time.perf_counter() - t
        tmp = parts / f"{sb[0]}_{sb[1]}_{sb[2]}.tmp.npz"
        np.savez_compressed(
            tmp,
            superblock=np.asarray(sb),
            seconds=dt,
            peak_gb=peak_rss_gb(),
            vertex=v,
            angle=an.astype(np.float32),
            weight=w.astype(np.float32),
            **{f"acc_{k}": x for k, x in vars(acc).items()},
        )
        tmp.rename(f)  # the result is there only when it is complete
        print(sb, f"{dt:.1f} s", len(v), f"peak {peak_rss_gb():.2f} GB", flush=True)
    collect(a.out, site_sb, time.perf_counter() - t0)


def collect(out: Path, site_sb: dict, seconds: float) -> None:
    """Sums the superblock files of out/superblocks into out/hemi.json and
    out/agreement.npz."""
    total = Acc()
    sites = {n: Acc() for n in site_sb}
    vtx, ang, wt, log = [], [], [], []
    for f in sorted((out / "superblocks").glob("*_*_*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        z = np.load(f)
        acc = Acc()
        for k in vars(acc):
            getattr(acc, k)[...] = z[f"acc_{k}"]
        sb = tuple(int(x) for x in z["superblock"])
        total.add(acc)
        for n, ssb in site_sb.items():
            if ssb == sb:
                sites[n].add(acc)
        vtx.append(z["vertex"])
        ang.append(z["angle"])
        wt.append(z["weight"])
        log.append(
            {
                "superblock": list(sb),
                "seconds": float(z["seconds"]),
                "peak_gb": float(z["peak_gb"]),
                "roots": len(z["vertex"]),
            }
        )
    np.savez_compressed(
        out / "agreement.npz",
        vertex=np.concatenate(vtx),
        angle=np.concatenate(ang),
        weight=np.concatenate(wt),
    )
    res = {
        "level": LEVEL,
        "superblocks": log,
        "seconds_this_run": seconds,
        "bins": {
            "depth": DEPTH_BINS.tolist(),
            "angle": ANGLE_BINS.tolist(),
            "axis": AXIS_BINS.tolist(),
            "agree": AGREE_BINS.tolist(),
        },
        "total": total.json(),
        "sites": {n: v.json() for n, v in sites.items()},
    }
    (out / "hemi.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
