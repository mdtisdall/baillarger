"""Why real clicks miss their column (plan: reports/Pial editing and edit priority.md,
step 4, open question): the points of editor_timing.py's M1S1 run, one by one.

Each click was made on a known column r (editor_timing.py: a top at the base top of r moved
along r; a contact at r's ridge point), recovered here by replaying that script's random
draws. For each point:
- the click mapping: the 8 surrounding voxels' roots, arc lengths and labels, which were
  blended, the carried root R and arc length s; R against r's root;
- the column the miss is measured against: the three vertices of R's triangle, each with
  its own column's distance from the click, the arc length there, its ridge, fluid entry
  and why it stopped; the blend of the three;
- column r itself at the click.
Views: prototypes/figures/pial_click_<i>.png, the axis plane that holds most of those
columns, around the click: echo 2 trend removed, the white surface and base pial
sections, the three columns and column r, the click and R.

Usage: diag_click_miss.py <session.npz> <volume.nii.gz> <edits.json> <figures dir> <i> [<i> ...]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

import pial_height as P
import solver as S
from editor_server import load_session

CODE = {0: "out", 1: "cortex", 2: "fluid", 3: "WM"}


def generators(s, z, near_face):
    """The column of every point of editor_timing.py's run, by its random draws."""
    rng = np.random.default_rng(4)
    t0 = s.thickness()
    cand = np.flatnonzero(~near_face & np.isfinite(t0))
    gen = []
    for r in rng.choice(cand, 10, replace=False):
        sgn = rng.choice([-1, 1])
        dt = sgn * rng.uniform(0.1, 0.4) / s.h
        x = s.P.point_at(s.cols, np.array([t0[r] + dt]), np.array([r]))[0]
        if np.isfinite(x).all():
            gen.append(("top", int(r), float(dt)))
    for r in rng.choice(
        np.flatnonzero(~near_face & (z["kind_label"] == P.TOP_RIDGE)), 5, replace=False
    ):
        gen.append(("contact", int(r), 0.0))
    return gen


def main() -> None:
    sess, vol, edits, figdir, *which = sys.argv[1:]
    s, extra = load_session(sess, "label")
    z = np.load(sess)
    img = np.asanyarray(nib.load(vol).dataobj)
    codes = np.zeros(
        img.shape, np.int8
    )  # cortex / fluid / WM from the session's march codes are not saved;
    # rebuild them from the volume's own mask: cortex is where echo 2 is nonzero
    d = json.loads(Path(edits).read_text())["edits"]
    gen = generators(s, z, extra["near_face"])
    base_pos = s.positions()
    for i in map(int, which):
        e = d[i]
        kind, r, dt = gen[i]
        x = s.to_vox(np.asarray(e["mm"]))
        print(
            f"\n=== point {i}: {kind}, made on column {r}"
            + (
                f" at its base top {'+' if dt > 0 else ''}{dt * s.h:.2f} mm"
                if kind == "top"
                else " at its ridge point"
            )
        )
        print(
            f"  click {np.round(x, 2)}; column r: ridge {s.cols.ridge_s[r] * s.h:.2f} mm, fluid {s.cols.fluid_s[r] * s.h:.2f} mm, "
            f"stop {P.STOP_NAMES[int(s.cols.stop_why[r])]} at {s.cols.stop_s[r] * s.h:.2f} mm, extended {bool(s.cols.extended[r])}"
        )
        dr, sr = S.polyline_nearest(s.cols.pts[r].astype(np.float64), x)
        print(
            f"  column r at the click: distance {dr * s.h:.3f} mm, arc length {sr * s.h:.3f} mm"
        )
        # the 8 corners
        i0 = np.floor(x).astype(int)
        print(
            "  corners (voxel, label, arc length mm, root, root's distance from r's root mm):"
        )
        for c in np.stack(np.meshgrid(*[[0, 1]] * 3, indexing="ij"), -1).reshape(-1, 3):
            q = i0 + c
            L = s.g.length[tuple(q)]
            Rq = s.g.root[tuple(q)]
            lab = "fluid" if img[tuple(q)] == 0 else "cortex"
            print(
                f"    {q} {lab:6s} s {L * s.h:6.3f}  root {np.round(Rq, 1)}  {np.linalg.norm(Rq - s.v[r]) * s.h:6.2f}"
            )
        R, sc = P.click_column(s.g, x[None])
        print(
            f"  carried: root {np.round(R[0], 2)} ({np.linalg.norm(R[0] - s.v[r]) * s.h:.2f} mm from r's root), s {sc[0] * s.h:.3f} mm"
        )
        idx, w = s._root_obs(R[0])
        print(
            "  R's triangle: vertex, weight, column's distance from the click mm, arc length there, ridge, fluid, stop:"
        )
        for j, wj in zip(idx, w):
            dj, sj = S.polyline_nearest(s.cols.pts[j].astype(np.float64), x)
            print(
                f"    {j:7d} {wj:5.2f}  {dj * s.h:6.3f}  {sj * s.h:6.3f}  ridge {s.cols.ridge_s[j] * s.h:5.2f}  "
                f"fluid {s.cols.fluid_s[j] * s.h:5.2f}  {P.STOP_NAMES[int(s.cols.stop_why[j])]} at {s.cols.stop_s[j] * s.h:.2f}"
            )
        db, sb = S.polyline_nearest(s._column_of(idx, w), x)
        print(
            f"  blended column: miss {db * s.h:.3f} mm at arc length {sb * s.h:.3f} (carried s {sc[0] * s.h:.3f})"
        )

        # ---- view
        lines = [s.cols.pts[j].astype(np.float64) for j in idx] + [
            s.cols.pts[r].astype(np.float64)
        ]
        allp = np.concatenate([q[np.isfinite(q).all(1)] for q in lines] + [x[None]])
        ax = int(
            np.argmin(np.var(allp - x, 0))
        )  # the plane normal to the axis the columns vary least along
        o = [a for a in range(3) if a != ax]
        c = int(round(x[ax]))
        half = 22
        lo = np.maximum(np.round(x[o]).astype(int) - half, 0)
        hi = np.minimum(np.round(x[o]).astype(int) + half, np.array(img.shape)[o] - 1)
        sl = [slice(None)] * 3
        sl[ax] = c
        sl[o[0]] = slice(lo[0], hi[0] + 1)
        sl[o[1]] = slice(lo[1], hi[1] + 1)
        fig, a = plt.subplots(figsize=(7.5, 7.5))
        a.imshow(
            img[tuple(sl)].T,
            cmap="gray",
            origin="lower",
            vmin=-0.25,
            vmax=0.25,
            interpolation="nearest",
            extent=(lo[0] - 0.5, hi[0] + 0.5, lo[1] - 0.5, hi[1] + 0.5),
        )
        for pts, col, lab in (
            (s.v, "#1baf7a", "white surface"),
            (base_pos, "#eb6834", "base pial (label)"),
        ):
            seg = P.plane_section(pts, s.f, ax, float(x[ax]))
            for k, sg in enumerate(seg):
                a.plot(
                    sg[:, o[0]],
                    sg[:, o[1]],
                    color=col,
                    lw=1,
                    label=lab if k == 0 else None,
                )
        for k, (j, q) in enumerate(zip(idx, lines[:3])):
            q = q[np.isfinite(q).all(1)]
            a.plot(
                q[:, o[0]],
                q[:, o[1]],
                color="#2a78d6",
                lw=1.2,
                alpha=0.9,
                label="columns of R's triangle" if k == 0 else None,
            )
            rs = s.cols.ridge_s[j]
            if np.isfinite(rs):
                pr = P.point_at(s.cols, np.array([rs]), np.array([j]))[0]
                a.plot(
                    pr[o[0]],
                    pr[o[1]],
                    "s",
                    mfc="none",
                    mec="#2a78d6",
                    ms=7,
                    label="their ridge points" if k == 0 else None,
                )
        q = lines[3][np.isfinite(lines[3]).all(1)]
        a.plot(
            q[:, o[0]],
            q[:, o[1]],
            color="#d64bd6",
            lw=1.6,
            label=f"column r ({r}), the click's own",
        )
        a.plot(x[o[0]], x[o[1]], "x", color="#e34948", ms=12, mew=2.5, label="click")
        a.plot(
            R[0, o[0]], R[0, o[1]], "o", color="#f2c230", ms=8, label="carried root R"
        )
        a.set_xlim(lo[0] - 0.5, hi[0] + 0.5)
        a.set_ylim(lo[1] - 0.5, hi[1] + 0.5)
        a.set_xlabel("ijk"[o[0]] + " (voxels)")
        a.set_ylabel("ijk"[o[1]] + " (voxels)")
        a.set_title(
            f"point {i} ({kind}): plane {'ijk'[ax]} = {c} (click at {x[ax]:.1f}); miss {db * s.h:.2f} mm\n"
            "columns drawn projected onto the plane",
            fontsize=10,
        )
        a.legend(fontsize=8, loc="lower left", framealpha=0.8)
        fig.tight_layout()
        fig.savefig(f"{figdir}/pial_click_{i}.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
