"""Step 4 check on a real patch (plan: reports/Pial editing and edit priority.md, §6): the
batch editor's server side on a session file, without the page.

Points: 10 tops (at the base top of random vertices >= 2 mm from the box faces, moved
0.1-0.4 mm up or down their columns), 5 contacts (at the ridge point of random columns
whose label top is the ridge), 3 free points (0.2 mm off the pial surface). Then:
- remesh (timed; per point: fallback and miss distance, and whether the click lay above
  or below the base top);
- remesh again: the same positions (determinism);
- the same points shuffled: the same positions (the result depends on the list, not
  its order);
- save, then a fresh server loads the file and remeshes: the same positions;
- a range of 1 mm and back, a switch of base and back (timed).

Usage: editor_timing.py <session.npz> <out json>
"""

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from editor_server import Editor  # noqa: E402


def main() -> None:
    sess, out = sys.argv[1:3]
    rng = np.random.default_rng(4)
    edits = Path(out).with_suffix(".edits.json")
    t = time.perf_counter()
    ed = Editor(sess, "label", str(edits))
    res = dict(load_s=round(time.perf_counter() - t, 2), vertices=len(ed.s.v))
    s, h = ed.s, ed.s.h
    ok = ~ed.extra["near_face"]
    kinds = np.load(sess)["kind_label"]
    t0 = s.thickness()
    side = []
    cand = np.flatnonzero(ok & np.isfinite(t0))
    for r in rng.choice(cand, 10, replace=False):
        sgn = rng.choice([-1, 1])
        x = s.P.point_at(
            s.cols, np.array([t0[r] + sgn * rng.uniform(0.1, 0.4) / h]), np.array([r])
        )[0]
        if np.isfinite(x).all():
            ed.add("top", s.to_mm(x).tolist())
            side.append("above" if sgn > 0 else "below")
    for r in rng.choice(
        np.flatnonzero(ok & (kinds == s.P.TOP_RIDGE)), 5, replace=False
    ):
        x = s.P.point_at(s.cols, np.array([s.cols.ridge_s[r]]), np.array([r]))[0]
        ed.add("contact", s.to_mm(x).tolist())
        side.append("ridge")
    for r in rng.choice(cand, 3, replace=False):
        ed.add("free", s.to_mm(ed.pos[r] + s.n[r] * 0.2 / h).tolist())
        side.append("free")

    def remesh(name):
        t1 = time.perf_counter()
        ed.remesh()
        res[name] = dict(wall_s=round(time.perf_counter() - t1, 2), **ed.timings)
        print(name, res[name], flush=True)
        return ed.pos.copy()

    p1 = remesh("remesh")
    res["points"] = [
        dict(
            kind=p["kind"],
            where=w,
            fallback=p["fallback"],
            miss_mm=p["miss_mm"],
            sides=p["sides"],
        )
        for p, w in zip(ed.points, side)
    ]
    p2 = remesh("remesh again")
    res["again_max_diff_voxels"] = float(np.nanmax(np.abs(p2 - p1)))
    keep = [dict(kind=p["kind"], mm=p["mm"]) for p in ed.points]
    perm = rng.permutation(len(keep))
    ed.points = [dict(keep[i]) for i in perm]
    p3 = remesh("remesh shuffled")
    res["shuffled_max_diff_voxels"] = float(np.nanmax(np.abs(p3 - p1)))
    ed.points = [dict(p) for p in keep]
    ed.remesh()
    ed.save()
    ed2 = Editor(sess, "label", str(edits))
    t1 = time.perf_counter()
    ed2.load()
    res["fresh_load_and_remesh_s"] = round(time.perf_counter() - t1, 2)
    res["fresh_load_max_diff_voxels"] = float(np.nanmax(np.abs(ed2.pos - p1)))
    del ed2
    ed.set_range(1.0)
    remesh("range 1 mm")
    ed.set_range(2.0)
    remesh("range 2 mm")
    ed.set_base("onepass_equal")
    remesh("base onepass_equal")
    ed.set_base("label")
    p4 = remesh("base label")
    res["after_range_and_base_round_trip_max_diff_voxels"] = float(
        np.nanmax(np.abs(p4 - p1))
    )
    res["peak_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "points"}, indent=1))
    for p in res["points"]:
        print(p)


if __name__ == "__main__":
    main()
