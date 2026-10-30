"""The marked M1S1 contact points (M1S1_test_points.json, "gap_all") placed as editor contact
points (key k) on two contoured bases, with the editor's own code (editor_server.Editor,
unchanged): the current base (equal arrival, slit2) and the field-weighted base (click_base.py).
All 3D.

Per base:
- before: no edits;
- all 23: every point as a contact point, remeshed;
- interleaved: the even-numbered points placed, the odd ones held out, and the reverse; each
  point scored by the run that did not place it.
Scored as click_base.py: the distance from each point to the carved contact's midline (halfway
between the nearest sheet and the nearest sheet facing it within 1.5 mm). Also the crossing
triangles after the edits, and how far the surface moved (vertices moved > 0.05 mm, and the
largest move).

Writes <work>/click_edits_M1S1.json, <figures>/click_edits_M1S1.png (per base: before, thin;
all 23 placed, thick; points as red x).

Usage: click_edits.py <figures dir> <work dir> <data dir>
"""

import json
import sys
import tempfile
import time

sys.path.insert(0, sys.argv[2])
sys.path.insert(0, sys.argv[2] + "/pial_edit")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

import pial_height as P_
from click_base import at_points
from editor_server import Editor

FIG, WORK, DATA = sys.argv[1:4]
LO = np.array((380, 175, 421))  # the M1S1 patch corner (baillarger.data.PATCHES)
T0 = time.time()


def say(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def summary(rows):
    mid = np.array(
        [r["midline_mm"] if r["midline_mm"] is not None else np.nan for r in rows]
    )
    return dict(
        carved_at=int(np.isfinite(mid).sum()),
        median=round(float(np.nanmedian(mid)), 3),
        p90=round(float(np.nanpercentile(mid, 90)), 3),
        worst=round(float(np.nanmax(mid)), 3),
    )


def main():
    m = json.load(open(f"{DATA}/M1S1_test_points.json"))
    Pts = (np.array(m["gap_all"]) + np.array(m["full_lo"]) - LO).astype(float)
    n, h = len(Pts), 0.1602
    out, shown = {}, {}
    for base, fname in (("equal (current)", "slit2"), ("field weights", "field")):
        sess = f"{DATA}/pial_M1S1_contour_{fname}_session.npz"
        ed = Editor(sess, "contour", tempfile.mkdtemp() + "/edits.json")
        ed.remesh()
        v0, f, n0 = ed.pos.copy(), ed.s.f, ed.s.n
        before = at_points((v0, f, n0), Pts, h)
        mm = ed.s.to_mm(Pts)
        res = dict(before=summary(before), before_rows=before)

        # all 23
        for p in mm:
            ed.add("contact", p.tolist())
        ed.remesh()
        moved = np.linalg.norm(ed.pos - v0, axis=1) * h
        allrows = at_points((ed.pos, f, n0), Pts, h)
        res["all 23"] = dict(
            **summary(allrows),
            crossing_faces=ed.crossings["faces"],
            crossing_places=ed.crossings["n_clusters"],
            at_contact=ed.crossings["at_contact"],
            moved_vertices=int(np.sum(moved > 0.05)),
            max_move_mm=round(float(moved.max()), 3),
            sides=[p_["sides"] for p_ in ed.points],
        )
        shown[base] = (v0, ed.pos.copy(), f)
        say(f"{base}: before {res['before']}; all 23 {res['all 23']}")

        # interleaved: place half, score the other half
        rows = [None] * n
        for place, held in (
            (np.arange(0, n, 2), np.arange(1, n, 2)),
            (np.arange(1, n, 2), np.arange(0, n, 2)),
        ):
            ed.points = []
            for k in place:
                ed.add("contact", mm[k].tolist())
            ed.remesh()
            r_ = at_points((ed.pos, f, n0), Pts[held], h)
            for k, r in zip(held, r_):
                rows[k] = r
        res["interleaved (held out)"] = summary(rows)
        res["interleaved_rows"] = rows
        say(f"{base}: interleaved, held out {res['interleaved (held out)']}")
        out[base] = res
        del ed
    json.dump(out, open(f"{WORK}/click_edits_M1S1.json", "w"), indent=1, default=float)

    # views: per base, the three planes holding most points; before thin, after all 23 thick
    img_nii = nib.load(f"{DATA}/pial_M1S1_echo2_detrended.nii.gz")
    img = np.asarray(img_nii.dataobj, np.float32)
    lim = np.nanpercentile(np.abs(img[img != 0]), 98)
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    for row, (base, (v0, v1, f)) in enumerate(shown.items()):
        for a in range(3):
            ax = axs[row][a]
            vals, cnt = np.unique(np.round(Pts[:, a]).astype(int), return_counts=True)
            c = vals[np.argmax(cnt)]
            o = [i for i in range(3) if i != a]
            sl = [slice(None)] * 3
            sl[a] = c
            ax.imshow(
                img[tuple(sl)].T,
                cmap="gray",
                origin="lower",
                vmin=-lim,
                vmax=lim,
                interpolation="nearest",
            )
            for v, col, lw in ((v0, "#8a8984", 0.9), (v1, "#eb6834", 1.6)):
                for sg in P_.plane_section(v, f, a, float(c)):
                    ax.plot(sg[:, o[0]], sg[:, o[1]], color=col, lw=lw)
            on = np.abs(Pts[:, a] - c) < 1.5
            ax.plot(Pts[on, o[0]], Pts[on, o[1]], "x", color="#e34948", ms=7, mew=2)
            lo_, hi_ = Pts[:, o].min(0) - 25, Pts[:, o].max(0) + 25
            ax.set_xlim(lo_[0], hi_[0])
            ax.set_ylim(lo_[1], hi_[1])
            ax.set_xticks([])
            ax.set_yticks([])
            r = out[base]
            ax.set_title(
                f"{base}, plane {'ijk'[a]} = {c}"
                + (
                    f"\nmidline to points: before {r['before']['median']:.2f} mm, all 23 placed "
                    f"{r['all 23']['median']:.2f}, held out {r['interleaved (held out)']['median']:.2f}"
                    if a == 0
                    else ""
                ),
                fontsize=10,
            )
    fig.suptitle(
        "M1S1: the 23 marked points placed as editor contact points. Grey: the base before; orange: after "
        "all 23 (remeshed); red x: the points. Echo 2, trend removed.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(f"{FIG}/click_edits_M1S1.png", dpi=110)
    say("done")


if __name__ == "__main__":
    main()
