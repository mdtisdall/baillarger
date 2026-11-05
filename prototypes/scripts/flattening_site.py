"""Flattening at a test site of the hemisphere (phase 3.1, "The hemisphere
run"): flatten_superblock on the superblocks that cover the site, with
their halos, then the views.

    flattening_site.py OUT FIGS --site NAME --surf SURF --dseg DSEG
        --echoes E1 E2 E3 --echo-dir DIR --overview OVERVIEW
        --superblocks I J K [I J K ...] --level M --halo-mm H
        --points POINTS.json [--offset I J K] [--sections N]

The feature scales and the energy reference come from the first superblock
and are given to the others, so that a column's result does not depend on
the superblock. POINTS: the user's marked points in full-grid voxel
coordinates ("points_full0"), or ITK-SNAP crop coordinates (1-based) with
--offset, the crop's full-grid corner (the PCC_SPL files).

Writes OUT/<site>.json (per superblock: columns, pairs, time, the solve's
log; the peak memory) and OUT/<site>.npz (per owned column: edge key,
vertex, misfit, the horizons s of each level), and in FIGS:
- <site>_sections_<k>.png: sections through marked points: echo 2 (log,
  minus its cortex level) against s, and the same flattened (against the
  level), with the points near the section;
- <site>_misfit.png: the misfit per column on the overview mesh.
"""

import argparse
import json
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from baillarger import flattening as FG
from baillarger import rays as R
from baillarger.blocks import ChunkedSource, open_volume
from baillarger.column_volume import ColumnVolumeSampler
from baillarger.white_overview import WhiteOverview

T0 = time.time()
DTAU = FG.DTAU
LEV0, NLEV = -int(round(R.CAP_IN / DTAU)), int(round((R.CAP_IN + R.CAP_OUT) / DTAU)) + 1


def say(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def peak_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1 << 30) if sys.platform == "darwin" else r / (1 << 20)


def load_points(path: Path, offset: list[int] | None) -> np.ndarray:
    d = json.loads(path.read_text())
    if "points_full0" in d:
        return np.asarray(d["points_full0"], np.float64)
    pts = []
    if "pieces" in d:
        for p in d["pieces"].values():
            pts += p.get("contact_points", [])
    else:
        pts = d["contact_points"]
    assert offset is not None, "crop points need --offset"
    return np.asarray(pts, np.float64) - 1 + np.asarray(offset)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("figs", type=Path)
    ap.add_argument("--site", required=True)
    ap.add_argument("--surf", type=Path, required=True)
    ap.add_argument("--dseg", type=Path, required=True)
    ap.add_argument("--echoes", type=Path, nargs=3, required=True)
    ap.add_argument("--echo-dir", type=Path, required=True)
    ap.add_argument("--overview", type=Path, required=True)
    ap.add_argument("--superblocks", type=int, nargs="+", required=True)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--halo-mm", type=float, default=FG.HALO_MM)
    ap.add_argument("--points", type=Path, required=True)
    ap.add_argument("--offset", type=int, nargs=3, default=None)
    ap.add_argument("--sections", type=int, default=2)
    ap.add_argument(
        "--region",
        action="store_true",
        help="also one solve over the owned columns of all the superblocks",
    )
    A = ap.parse_args()
    A.out.mkdir(parents=True, exist_ok=True)
    A.figs.mkdir(parents=True, exist_ok=True)
    sbs = [tuple(A.superblocks[i : i + 3]) for i in range(0, len(A.superblocks), 3)]
    pts = load_points(A.points, A.offset)

    rs = R.RaySampler(A.surf, A.dseg)
    echoes = []
    for f in A.echoes:
        src = open_volume(f, A.echo_dir / f.with_suffix("").name)
        echoes.append(ChunkedSource(src, 64, 1 << 30))
        aff = src.affine
    smp = ColumnVolumeSampler(echoes, rs.affine, [aff] * 3, rs.shape)  # type: ignore[arg-type]
    faces = FG.SurfaceFaces.load(A.surf)
    scales = eref = None
    keys, verts, mis, ech = [], [], [], []
    slim: list[FG.Flattened] = []
    calls = []
    for sb in sbs:
        t = time.time()
        f = FG.flatten_superblock(
            rs,
            smp,
            faces,
            sb,
            A.level,
            A.halo_mm,
            scales=scales,  # type: ignore[arg-type]
            energy_ref=eref,
            log=say,
            keep_echo=True,
        )
        scales, eref = f.scales, f.energy_ref
        calls.append(
            dict(
                superblock=list(sb),
                owned=int(len(f.edge_key)),
                columns=f.n_columns,
                pairs=f.n_pairs,
                seconds=round(time.time() - t, 1),
                peak_gb_so_far=round(peak_gb(), 2),
                fit=f.info,
                s_lo=f.s_lo,
            )
        )
        say(
            f"superblock {sb}: {calls[-1]['owned']:,} owned of {f.n_columns:,}, {calls[-1]['seconds']} s"
        )
        keys.append(f.edge_key)
        verts.append(f.vertex)
        mis.append(f.misfit)
        assert f.echo is not None
        raw = FG.raw_features(f.echo)[0]  # echo 2 (log, minus its cortex level)
        E = np.full(
            (len(f.edge_key), int(round(10.0 / f.step)) + 1), np.nan, np.float32
        )
        i0 = int(round((f.s_lo + R.CAP_IN) / f.step))
        E[:, i0 : i0 + raw.shape[1]] = np.where(f.echo.cortex, raw, np.nan)
        ech.append(E)
        empty = np.zeros((0, 1), np.float32)
        slim.append(replace(f, shift=empty, weight=empty, echo=None))
        del f
    key = np.concatenate(keys)
    vertex = np.concatenate(verts)
    misfit = np.concatenate(mis)
    Es = np.concatenate(ech)
    # the coarse solve: one offset per superblock and level (the gauge)
    lev, c = FG.gauge_offsets(slim)
    say(f"gauge offsets: |c| max {np.abs(c).max():.4f} mm")
    Hs = np.full((len(key), NLEV), np.nan, np.float32)
    j = lev - LEV0
    ok = (j >= 0) & (j < NLEV)
    Hs[:, j[ok]] = np.concatenate(
        [lev[None] * DTAU + FG.in_gauge(f, c[b], lev) for b, f in enumerate(slim)]
    )[:, ok]
    raw_H = np.concatenate(
        [lev[None] * DTAU + FG.in_gauge(f, 0 * c[b], lev) for b, f in enumerate(slim)]
    )[:, ok]
    region = None
    if A.region:
        region = region_solve(A, rs, smp, faces, sbs, scales, eref)
        rk = {int(k): i for i, k in enumerate(region.edge_key)}
        idx = np.array([rk[int(k)] for k in key])
        G = lev[None] * DTAU + FG.common_levels(region, region.r, lev)
        # the region's own edge: compare on the columns whose roots are at
        # least 2 halos inside the region's bounding box of roots
        x = np.asarray(rs.x[vertex], np.float64)
        lo_, hi_ = x.min(0), x.max(0)
        inner = np.all((x >= lo_ + 2 * A.halo_mm) & (x <= hi_ - 2 * A.halo_mm), axis=1)
        m = (lev * DTAU >= 0) & (lev * DTAU <= 3.0)
        dd = (Hs[:, j[ok]] - G[idx][:, ok])[inner][:, m[ok]]
        d0 = (raw_H - G[idx][:, ok])[inner][:, m[ok]]
        say(
            f"region against local + offsets, {int(inner.sum())} inner columns: "
            f"|d| p50/p99/max {np.percentile(np.abs(dd), [50, 99, 100]).round(5).tolist()}; "
            f"without the offsets {np.percentile(np.abs(d0), [50, 99, 100]).round(5).tolist()}"
        )
        region = dict(
            inner_columns=int(inner.sum()),
            with_offsets_abs_p50_p99_max=np.percentile(
                np.abs(dd), [50, 99, 100]
            ).tolist(),
            without_offsets_abs_p50_p99_max=np.percentile(
                np.abs(d0), [50, 99, 100]
            ).tolist(),
            region_fit=region.info,
        )
    stats = dict(
        site=A.site,
        level=A.level,
        halo_mm=A.halo_mm,
        scales=list(scales),
        energy_ref=eref,
        calls=calls,
        peak_gb=peak_gb(),
        total_seconds=round(time.time() - T0, 1),
        misfit_p50_p90_p99=np.nanpercentile(misfit, [50, 90, 99]).tolist(),
        offsets_abs_max=float(np.abs(c).max()),
        region=region,
    )
    (A.out / f"{A.site}.json").write_text(json.dumps(stats, indent=1))
    np.savez_compressed(
        A.out / f"{A.site}.npz",
        edge_key=key,
        vertex=vertex,
        misfit=misfit,
        horizons=Hs,
        lev0=LEV0,
        dtau=DTAU,
    )
    say(f"saved; peak {peak_gb():.2f} GB")
    sections(A, rs, vertex, Hs, Es, pts)
    misfit_map(A, rs, vertex, misfit)
    say("figures done")


def region_solve(A, rs, smp, faces, sbs, scales, eref) -> FG.Flattened:  # type: ignore[no-untyped-def]
    """One solve over the owned columns of all the superblocks (no halo), as
    a Flattened with no halo."""
    from baillarger.column_volume import ColumnVolume  # noqa: F401

    pads, roots = [], []
    for sb in sbs:
        r0 = rs.sample(sb, A.level, e_in=0.0, e_out=0.0)
        e_in, e_out = FG.extents(r0)
        rays = rs.sample(sb, A.level, e_in=e_in, e_out=e_out)
        pads.append(
            FG.pad(
                smp.sample(rays, None),
                FG.FEATURE_ECHOES,
                s_lo=FG.grid_start(R.CAP_IN),
                n=int(round((R.CAP_IN + R.CAP_OUT) / rays.step)) + 1,
            )
        )
        roots.append(rays.roots)
    p = FG.Padded(
        pads[0].s_lo,
        pads[0].step,
        np.concatenate([q.echo for q in pads], 1),
        np.concatenate([q.valid for q in pads]),
        np.concatenate([q.cortex for q in pads]),
    )
    del pads
    F, M, _ = FG.features(p, scales)
    key = np.concatenate([r.edge_key for r in roots])
    vtx = np.concatenate([r.vertex for r in roots])
    xr = np.concatenate([r.root for r in roots]).astype(np.float64)
    vox = (rs.inverse[:3, :3] @ xr.T).T + rs.inverse[:3, 3]
    pairs = FG.orient(FG.mesh_edge_pairs(faces, vtx, vox), key)
    ps = FG.pair_shifts(F, M, pairs, p.step, eref)
    del F, M
    say(f"region: {len(key):,} columns, {len(pairs):,} pairs; peak {peak_gb():.2f} GB")
    w = FG.fit(ps, len(key), p.s_lo, p.step, log=say)
    say(f"region fit; peak {peak_gb():.2f} GB")
    e = np.zeros((0, 1), np.float32)
    return FG.Flattened(
        edge_key=key,
        s_lo=p.s_lo,
        step=p.step,
        tau=w.tau,
        r=w.r,
        anchor_level=w.anchor_level,
        pair_key=np.zeros((0, 2), np.int64),
        shift=e,
        weight=e,
        scales=np.asarray(scales),
        energy_ref=float(eref),
        info=w.info,
        n_columns=len(key),
        n_pairs=len(pairs),
        misfit=np.zeros(len(key)),
        vertex=vtx,
        halo_key=np.zeros(0, np.int64),
        halo_r=np.zeros((0, w.r.shape[1])),
    )


def sections(A, rs, vertex, Hs, Es, pts) -> None:  # type: ignore[no-untyped-def]
    x = np.asarray(rs.x[vertex], np.float64)
    nrm = np.asarray(rs.n[vertex], np.float64)
    v = (rs.inverse[:3, :3] @ x.T).T + rs.inverse[:3, 3]
    step = 0.04
    s_ax = np.arange(Es.shape[1]) * step - R.CAP_IN
    lev_s = (np.arange(NLEV) + LEV0) * DTAU
    order = np.linspace(0, len(pts) - 1, A.sections).round().astype(int)
    for si, pi in enumerate(order):
        p = pts[pi]
        sel = np.flatnonzero(
            (np.abs(v[:, 1] - p[1]) <= 0.5)
            & (np.abs(v[:, 0] - p[0]) <= 48)
            & (np.abs(v[:, 2] - p[2]) <= 48)
        )
        if len(sel) < 10:
            say(f"section {si}: {len(sel)} columns, skipped")
            continue
        xy = v[sel][:, [0, 2]]
        u = np.linalg.svd(xy - xy.mean(0), full_matrices=False)[2][0]
        sel = sel[np.argsort((xy - xy.mean(0)) @ u)]
        raw = Es[sel]
        flat = np.full((len(sel), NLEV), np.nan)
        for j, c in enumerate(sel):
            h = Hs[c]
            okh = np.isfinite(h)
            okr = np.isfinite(raw[j])
            if okh.sum() > 1 and okr.sum() > 1:
                vals = np.interp(
                    h[okh], s_ax[okr], raw[j][okr], left=np.nan, right=np.nan
                )
                # only inside the column's own samples
                inside = (h[okh] >= s_ax[okr].min()) & (h[okh] <= s_ax[okr].max())
                flat[j, np.flatnonzero(okh)] = np.where(inside, vals, np.nan)
        near = pts[np.abs(pts[:, 1] - p[1]) <= 2]
        mm = rs.affine[:3, :3] @ near.T + rs.affine[:3, 3:4]
        px, ps_ = [], []
        for q in mm.T:
            j = int(np.argmin(np.linalg.norm(x[sel] - q, axis=1)))
            px.append(j)
            ps_.append(float((q - x[sel][j]) @ nrm[sel][j]))
        lo, hi = np.nanpercentile(raw, [2, 98])
        fig, ax = plt.subplots(2, 1, figsize=(12, 8.5))
        ax[0].imshow(
            raw.T,
            aspect="auto",
            cmap="gray",
            vmin=lo,
            vmax=hi,
            extent=(-0.5, len(sel) - 0.5, s_ax[-1], s_ax[0]),
            interpolation="nearest",
        )
        ax[0].plot(px, ps_, "x", color="#e34948", ms=6)
        ax[0].set_ylim(4.0, -0.6)
        ax[0].set_ylabel("s along the column (mm)")
        ax[0].set_title(
            f"{A.site}: echo 2 (log, minus its cortex level), section {si} at point "
            f"{tuple(int(a) for a in p)}; {len(sel)} columns; marked points (x)"
        )
        ax[1].imshow(
            flat.T,
            aspect="auto",
            cmap="gray",
            vmin=lo,
            vmax=hi,
            extent=(-0.5, len(sel) - 0.5, lev_s[-1] + DTAU / 2, lev_s[0] - DTAU / 2),
            interpolation="nearest",
        )
        ax[1].set_ylim(4.0, -0.6)
        ax[1].set_ylabel("level (mm at the mean)")
        ax[1].set_xlabel("column, in order along the section")
        ax[1].set_title(
            "flattened: the same columns sampled at the horizons of the warp"
        )
        fig.tight_layout()
        png = A.figs / f"{A.site}_sections_{si}.png"
        fig.savefig(png, dpi=72)
        plt.close(fig)
        say(f"wrote {png}")


def misfit_map(A, rs, vertex, misfit) -> None:  # type: ignore[no-untyped-def]
    ov = WhiteOverview.load(A.overview)
    val = np.full(len(rs.x), np.nan, np.float32)
    val[vertex] = misfit
    v = np.asarray(rs.x[vertex], np.float64)
    lo, hi = v.min(0), v.max(0)
    ovm = (ov.affine[:3, :3] @ ov.v.T).T + ov.affine[:3, 3]
    cen = ovm[ov.f].mean(1)
    fsel = np.flatnonzero(np.all((cen >= lo) & (cen <= hi), axis=1))
    ids, face = ov.carried(fsel)
    vals = val[ids]
    num = np.bincount(face, np.nan_to_num(vals), minlength=len(ov.f))
    den = np.bincount(face, np.isfinite(vals), minlength=len(ov.f))
    fv = np.where(den > 0, num / np.maximum(den, 1), np.nan)[fsel]
    keep = np.isfinite(fv)
    tri = ovm[ov.f[fsel[keep]]]
    vmax = float(np.nanpercentile(fv, 98))
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(14, 6.5))
    for k, (el, az) in enumerate(((20, -60), (20, 120))):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        pc = Poly3DCollection(
            tri, facecolors=cmap(np.clip(fv[keep] / vmax, 0, 1)), edgecolor="none"
        )
        ax.add_collection3d(pc)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.view_init(el, az)
        ax.set_box_aspect(hi - lo)
        ax.set_title(f"view {k + 1}")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    fig.colorbar(
        sm, ax=fig.axes, shrink=0.6, label="weighted RMS misfit per column (mm)"
    )
    fig.suptitle(
        f"{A.site}: misfit of the warp on the overview mesh (mean over the columns each face carries)"
    )
    png = A.figs / f"{A.site}_misfit.png"
    fig.savefig(png, dpi=72)
    plt.close(fig)
    say(f"wrote {png}")


if __name__ == "__main__":
    main()
