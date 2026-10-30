"""Pia editor on an inflated white surface: the local server (plan: reports/Crossing paths and contacts from
laminar profiles.md, *A pia editor on an inflated surface*). Prototype only; binds 127.0.0.1 only.

The page (pia_editor_static/index.html) shows the box's white surface in 3D, true or inflated
(inflate.py; plain WebGL2), and as a small conformal flat map (flat_map.py) for an overview of the
worklist and of what is done; both coloured by a layer of edit_priority_<tag>.npz (the combined
priority, any rank or doubt; a placeholder until that file appears, read again when it changes) or
a few others. A click on the 3D surface sets the crosshair at the nearest subsampled trace, with
two orthogonal sections cut from the true mesh (pia_sections.py) along the screen's horizontal and
vertical carried to the true surface (on the flat map: along the map's axes): echo 2 (trend
removed) along the traces, depth down. A click on a section at a depth adds an edit
{trace: depth} to the pending list; "re-solve" makes the pending edits one step: free =
edit_resolve.neighbourhood(G, roots, the step's traces, R), then edit_resolve.local_solve with every
edit made so far. The steps replay in order from the saved joint result, so the pia depends only
on the list (QPBO-I's rand() reseeded per step, see SURFCUT_SEEDS). Worklist: the top decile of
`combined`, joined into patches by the joint warp's neighbour pairs, ordered by size x mean score.

Requests:
- GET  / (the page); /meta (JSON: map positions of the traces, layers, worklist, steps, marks);
  /layer?name=... (JSON values per trace); /section?i=<trace>&half=<mm> (JSON: the two sections
  at trace i along the flat map's axes and, at a contact, the facing trace's two sections);
  /mesh?level=k[&faces=1] (binary: the white mesh, true or inflated, see mesh_payload);
  /streamlines?i=<trace>&radius=<mm>&half=<mm> (JSON: the streamline panel at trace i's root, see
  Editor.streamlines and streamline_view.py: the traces from and to it, whatever tracer made the
  column volume, the true white mesh and the GM boundary in a box around it, the pia on them).
- POST /section3d {trace, right, up, level, half}: as /section, along the screen's axes on the
  inflated surface carried to the true mesh (inflate.transfer).
- POST /edit {trace, depth}; /unpend {k}; /resolve {R}; /undo (drop the last step);
  /ok {trace} (an OK mark: looked at, no change); /save; /load; /replay (rebuild from the list
  and report the largest difference from the current pia).
Saved to <work>/pia_edits_<tag>.json (the list: per edit trace, depth, box voxel, ITK-SNAP voxel
(1-based, box), full-grid voxel, world mm, time; per step R and its solve; OK marks) and the
current levels to <work>/pia_edits_<tag>_levels.npz.

Usage: pia_editor.py <port> <work> <tag> [--flat <flat_map npz>] [--edits <json>]
  tag as joint_<tag>.npz: V1V2_banks_s0.16, M1S1_s0.16_r3. Run from the repository root.
"""

import argparse
import ctypes
import datetime
import inspect
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

sys.path.insert(0, str(Path(__file__).parent))
import edit_data as ED  # noqa: E402
import edit_resolve as ER  # noqa: E402
import inflate as IN  # noqa: E402
import pia_sections as PS  # noqa: E402
import streamline_view as SV  # noqa: E402
from volume_views import mesh_graph  # noqa: E402

STATIC = Path(__file__).parent / "pia_editor_static"
STEP = ED.STEP
H = 0.04
DECILE = 90
# QPBO-I (baillarger.surfcut.qpbo_pruned -> thinqpbo improve()) visits variables in an order drawn
# from the C library's rand(): the same step could end differently with the process's rand() state
# (measured: one trace by 0.12 mm in a 192-trace step). surfcut from PR #33 on reseeds rand() itself
# (qpbo_pruned(seed=0), which edit_resolve.local_solve gets by default); with an older surfcut each
# step reseeds it here, so either way a step's result depends only on its inputs and a replay of the
# list gives the same levels.
_LIBC = ctypes.CDLL(None)
RAND_SEED = 1
SURFCUT_SEEDS = "seed" in inspect.signature(ER.WC.qpbo_pruned).parameters


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


class Editor:
    def __init__(self, work, tag, flat_path=None, edits_path=None, log=print):
        t0 = time.time()
        self.work, self.tag, self.site = work, tag, ED.site_of(tag)
        self.D, J, Z = ED.load(work, tag, log=log)
        self.sub, self.z0 = J["sub"], J["z"].copy()
        self.cert_saved = J["cert"][-1]
        self.n = self.sub.size
        self.vox, self.lo = Z["vox"], Z["lo"]
        self.vmm = Z["v"].astype(float) * self.vox
        self.G = mesh_graph(Z["v"], Z["f"], self.vox)
        self.rv = Z["tv"][self.sub]
        self.nrm = Z["nrm"][self.rv].astype(float)
        self.roots_of = -np.ones(len(self.vmm), int)
        self.roots_of[self.rv] = np.arange(self.n)
        # every trace of the column volume, for the streamline panel (streamline_view.py)
        self.pos_all, self.ns_all, self.tv_all = Z["pos"], Z["ns"], Z["tv"]
        self.cls_all, self.fallback_all, self.ctx = Z["cls"], Z["tracer"], Z["ctx"]
        self.tracer_name = (
            str(Z["tracer_name"])
            if "tracer_name" in Z.files
            else "not saved (made before tracer_name)"
        )
        self.wm_label = (
            str(Z["wm_label"]) if "wm_label" in Z.files else "not saved (aseg)"
        )
        self.sub_of = -np.ones(len(self.tv_all), np.int64)
        self.sub_of[self.sub] = np.arange(self.sub.size)
        self.pos, self.ns = self.pos_all[self.sub], self.ns_all[self.sub]
        self.echo = Z["fv"][0][self.sub]
        self.vmax = float(np.nanpercentile(np.abs(self.echo), 97))
        self.top = Z["H_top"][self.sub].astype(float)
        self.single = np.load(f"{work}/pia_{self.site}.npz")["c"][self.sub]
        self.flat_path = flat_path or f"{work}/flat_map_{self.site}.npz"
        F = np.load(self.flat_path)
        self.uv, self.T = F["uv_v"][self.rv], F["T_v"][self.rv]
        g = np.einsum("nij,nik->njk", self.T, self.T)
        self.log2_area = -0.5 * np.log2(
            np.linalg.det(g)
        )  # map area / surface area at the root
        self.f = Z["f"].astype(np.int64)
        self.vf = IN.vertex_faces(self.f, len(self.vmm))
        self.vnrm = IN.vertex_normals(self.vmm, self.f)
        Ip = np.load(f"{work}/inflate_{self.site}.npz")
        self.infl = [self.vmm] + [x.astype(float) for x in Ip["x"]]
        self.infl_names = ["true surface"] + [
            f"inflated, lam {lv:g}" for lv in Ip["levels"]
        ]
        self.sulc = IN.sulcal_depth(self.vmm, self.infl[-1], self.f)
        # each vertex shows its nearest subsampled trace along the mesh (none beyond 0.5 mm)
        d, _, src = dijkstra(
            self.G, indices=self.rv, min_only=True, return_predecessors=True, limit=0.5
        )
        self.vtrace = np.where(
            np.isfinite(d), self.roots_of[np.maximum(src, 0)], -1
        ).astype(np.int32)
        self.pairs, self.ties = J["pairs"], J["ties"]
        self.partner = {}
        for a, b in self.ties:
            self.partner.setdefault(int(a), int(b))
            self.partner.setdefault(int(b), int(a))
        P = Z["P"]
        self.P = P
        self.hits = {}
        for p, (tr, dep) in enumerate(ED.point_traces(Z, self.sub, P)):
            for t, d in zip(tr, dep):
                self.hits.setdefault(int(t), []).append((p + 1, float(d)))
        import nibabel as nib

        from baillarger.data import crop_affine, echo_path

        try:
            self.affine = crop_affine(nib.load(str(echo_path(2))).affine, self.lo)
        except Exception as exc:  # no image: world mm left out
            log(f"no echo 2 header ({exc}); world mm left out of the edit records")
            self.affine = None
        self.prio_path = f"{work}/edit_priority_{tag}.npz"
        self.prio_mtime = None
        self.load_priority(log)
        self.edits_path = edits_path or f"{work}/pia_edits_{tag}.json"
        self.steps, self.pending, self.ok = [], [], []
        self.states = [self.z0.copy()]
        log(
            f"editor ready: {self.n:,} traces, {len(self.ties):,} ties, {len(self.hits):,} traces at your points "
            f"({time.time() - t0:.0f} s)"
        )

    # ------------------------------------------------------------ layers
    def load_priority(self, log=print):
        mt = (
            os.path.getmtime(self.prio_path) if os.path.exists(self.prio_path) else None
        )
        if mt is not None and mt == self.prio_mtime:
            return
        self.prio_mtime = mt
        if mt is None:
            # placeholder until edit_priority writes its map: the saved certification and the
            # distance to the single-surface horizon, as percentile ranks, and their mean
            import edit_priority as EP

            unc = (~self.cert_saved).astype(float)
            dss = np.abs(self.z0[-1] - self.single)
            self.prio = {
                "rank_not certified (saved)": EP.ranks(unc),
                "rank_|joint - single-surface|": EP.ranks(dss),
                "doubt_not certified (saved)": unc,
                "doubt_|joint - single-surface|": dss,
            }
            self.prio["combined"] = np.mean(
                [
                    self.prio["rank_not certified (saved)"],
                    self.prio["rank_|joint - single-surface|"],
                ],
                axis=0,
            )
            self.prio_status = "placeholder (edit_priority_<tag>.npz not written yet)"
            self.cert, self.lo_mm, self.hi_mm = self.cert_saved, None, None
        else:
            Q = np.load(self.prio_path)
            assert np.array_equal(Q["sub"], self.sub)
            self.prio = {
                k: Q[k].astype(float)
                for k in Q.files
                if k == "combined" or k.startswith(("rank_", "doubt_"))
            }
            self.cert, self.lo_mm, self.hi_mm = Q["cert"], Q["lo"], Q["hi"]
            self.prio["doubt_not certified"] = (~self.cert).astype(float)
            self.prio_status = f"edit_priority_{self.tag}.npz ({time.ctime(mt)})"
            log(f"priority map loaded: {self.prio_status}")
        self.worklist = self.make_worklist()

    def layer_names(self):
        return list(self.prio) + [
            "pia change (mm)",
            "log2 map area factor",
            "open top (H_top set)",
            "joint - single-surface (mm)",
        ]

    def layer(self, name):
        self.load_priority()
        z = self.states[-1]
        if name in self.prio:
            return self.prio[name]
        return {
            "pia change (mm)": z[-1] - self.z0[-1],
            "log2 map area factor": self.log2_area,
            "open top (H_top set)": np.isfinite(self.top).astype(float),
            "joint - single-surface (mm)": z[-1] - self.single,
        }[name]

    def make_worklist(self):
        c = self.prio["combined"]
        hot = c >= np.percentile(c, DECILE)
        p = self.pairs[hot[self.pairs[:, 0]] & hot[self.pairs[:, 1]]]
        A = coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(self.n, self.n))
        _, lab = connected_components(A, directed=False)
        out = []
        for k in np.unique(lab[hot]):
            tr = np.flatnonzero(hot & (lab == k))
            out.append(
                dict(
                    traces=tr.tolist(),
                    size=int(tr.size),
                    mean=float(c[tr].mean()),
                    score=float(tr.size * c[tr].mean()),
                    peak=int(tr[np.argmax(c[tr])]),
                )
            )
        out.sort(key=lambda d: -d["score"])
        for i, d in enumerate(out):
            d["rank"] = i + 1
        return out

    # ------------------------------------------------------------ sections
    def screen_directions(self, i, right, up, level):
        """The screen's right and up at trace i's root on inflated level `level`, carried to the
        true mesh (inflate.transfer); t_u from right, t_v = n x t_u in the sense of up; and the
        angle between t_v and the carried up (the inflation's angle distortion there)."""
        x, r = self.infl[int(level)], self.rv[i]
        T = np.c_[
            IN.transfer(np.asarray(right, float), r, x, self.vmm, self.f, self.vf),
            IN.transfer(np.asarray(up, float), r, x, self.vmm, self.f, self.vf),
        ]
        return PS.directions(T, self.nrm[i])

    def section_at(self, i, half=8.0, right=None, up=None, level=0):
        i = int(i)
        if right is None:
            tu, tv, ang = PS.directions(self.T[i], self.nrm[i])
            frame = "flat map axes"
        else:
            tu, tv, ang = self.screen_directions(i, right, up, level)
            frame = f"screen axes on the {self.infl_names[int(level)]}"
        out = dict(
            trace=i, angle_error_deg=round(ang, 2), frame=frame, info=self.trace_info(i)
        )
        out["u"] = self.pack(*self.cut(i, tu, half))
        out["v"] = self.pack(*self.cut(i, tv, half))
        j, via = self.facing(i, out)
        if j is not None:
            nj = self.nrm[j]
            out["facing"] = dict(trace=j, via=via, info=self.trace_info(j))
            for name, t in (("u", tu), ("v", tv)):
                tj = t - (t @ nj) * nj
                out["facing"][name] = self.pack(
                    *self.cut(j, tj / np.linalg.norm(tj), half)
                )
        return out

    def cut(self, i, t, half):
        r = self.rv[i]
        return PS.section(
            self.vmm,
            self.G,
            r,
            self.vmm[r],
            self.nrm[i],
            t,
            self.roots_of,
            half=half,
            band=0.75 * float(self.vox.min()),
            f=self.f,
        )

    def facing(self, i, out):
        """The facing trace: i's own tie, else the tie nearest i (within 1 mm) on either section."""
        if i in self.partner:
            return self.partner[i], f"tie of trace {i}"
        best = None
        for name in ("u", "v"):
            for t, d in zip(out[name]["traces"], out[name]["dist"]):
                if (
                    t in self.partner
                    and abs(d) <= 1.0
                    and (best is None or abs(d) < best[0])
                ):
                    best = (abs(d), t)
        if best is None:
            return None, None
        return self.partner[
            best[1]
        ], f"tie of trace {best[1]}, {best[0]:.2f} mm from the crosshair"

    def pack(self, tr, dist):
        z = self.states[-1]
        tr = np.asarray(tr, int)
        img = [
            np.clip(
                np.round(np.nan_to_num(self.echo[t, : self.ns[t]]) / self.vmax * 127),
                -127,
                127,
            )
            .astype(int)
            .tolist()
            for t in tr
        ]
        unc = ~self.cert[tr]
        edits = {}
        for e in self.all_edits():
            edits[e["trace"]] = e
        mine = [
            dict(k=k, depth=e["depth"], pending=e.get("pending", False))
            for k, e in (
                (int(np.flatnonzero(tr == e["trace"])[0]), e)
                for e in edits.values()
                if e["trace"] in set(tr.tolist())
            )
        ]
        r2 = lambda a: [None if not np.isfinite(x) else round(float(x), 3) for x in a]  # noqa: E731
        return dict(
            traces=tr.tolist(),
            dist=r2(dist),
            img=img,
            ns=self.ns[tr].tolist(),
            levels=[r2(z[k, tr]) for k in range(z.shape[0])],
            pia_saved=r2(self.z0[-1, tr]),
            lo=r2(np.where(unc, self.lo_mm[tr], np.nan))
            if self.lo_mm is not None
            else None,
            hi=r2(np.where(unc, self.hi_mm[tr], np.nan))
            if self.hi_mm is not None
            else None,
            uncertified=unc.astype(int).tolist(),
            top=r2(self.top[tr]),
            single=r2(self.single[tr]),
            points=[
                dict(k=k, point=p, depth=round(d, 3))
                for k, t in enumerate(tr.tolist())
                for p, d in self.hits.get(t, [])
            ],
            tied=[int(t in self.partner) for t in tr.tolist()],
            edits=mine,
        )

    def trace_info(self, i):
        return dict(
            trace=int(i),
            map_mm=[round(float(x), 2) for x in self.uv[i]],
            root_itksnap=[int(x) for x in np.round(self.pos[i, 0]) + 1],
            pia_mm=round(float(self.states[-1][-1, i]), 3),
            pia_saved_mm=round(float(self.z0[-1, i]), 3),
            certified=bool(self.cert[i]),
            combined=round(float(self.prio["combined"][i]), 3),
            tie=self.partner.get(int(i)),
        )

    # ------------------------------------------------------------ edits
    def record(self, t, depth):
        t = int(t)
        k = int(np.clip(round(depth / STEP), 0, self.ns[t] - 1))
        p = self.pos[t, k].astype(float)
        e = dict(
            trace=t,
            depth=round(float(depth), 3),
            box_voxel0=[round(float(x), 2) for x in p],
            itksnap=[int(x) for x in np.round(p) + 1],
            full_grid0=[int(x) for x in np.round(p) + self.lo],
            time=now(),
        )
        if self.affine is not None:
            e["world_mm"] = [
                round(float(x), 3) for x in (self.affine @ np.r_[p, 1.0])[:3]
            ]
        return e

    def all_edits(self):
        return [e for s in self.steps for e in s["edits"]] + [
            dict(e, pending=True) for e in self.pending
        ]

    def add_edit(self, t, depth):
        self.pending.append(self.record(t, depth))
        return self.state()

    def unpend(self, k):
        self.pending.pop(int(k))
        return self.state()

    def solve_step(self, z, made, step):
        seeds = np.array([e["trace"] for e in step["edits"]], int)
        for e in step["edits"]:
            made[e["trace"]] = e["depth"]
        free = ER.neighbourhood(self.G, self.rv, seeds, step["R"])
        if not SURFCUT_SEEDS:
            _LIBC.srand(RAND_SEED)
        z2, info = ER.local_solve(self.D, z, free, h=H, edits=made)
        return z2, info

    def made_through(self, k):
        made = {}
        for s in self.steps[:k]:
            for e in s["edits"]:
                made[e["trace"]] = e["depth"]
        return made

    def resolve(self, R):
        if not self.pending:
            return dict(self.state(), message="no pending edits")
        step = dict(R=float(R), edits=[dict(e) for e in self.pending], time=now())
        z, made = self.states[-1], self.made_through(len(self.steps))
        z2, info = self.solve_step(z, made, step)
        d = np.abs(z2[-1] - z[-1])
        step["solve"] = dict(
            free_traces=info["free_traces"],
            seconds=round(info["seconds"], 2),
            pia_changed=int(np.sum(d > 1e-9)),
            pia_max_change_mm=round(float(d.max()), 3),
            certified_last_try=round(info["tries"][-1]["certified"], 4),
        )
        self.steps.append(step)
        self.states.append(z2)
        self.pending = []
        return dict(self.state(), message=f"step {len(self.steps)}: {step['solve']}")

    def undo(self):
        if self.steps:
            s = self.steps.pop()
            self.states.pop()
            self.pending = [
                {k: v for k, v in e.items()} for e in s["edits"]
            ] + self.pending
        return self.state()

    def mark_ok(self, t):
        self.ok.append(
            dict(
                trace=int(t),
                time=now(),
                itksnap=[int(x) for x in np.round(self.pos[int(t), 0]) + 1],
            )
        )
        return self.state()

    def replay(self, steps):
        z, made, states = self.z0.copy(), {}, [self.z0.copy()]
        for s in steps:
            z, _ = self.solve_step(z, made, s)
            states.append(z)
        return states

    def replay_check(self):
        t0 = time.time()
        states = self.replay(self.steps)
        d = float(np.max(np.abs(states[-1] - self.states[-1]))) if self.steps else 0.0
        return dict(
            self.state(),
            message=f"replayed {len(self.steps)} steps in {time.time() - t0:.1f} s: "
            f"largest difference from the current levels {d:.3g} mm",
            replay_max_diff_mm=d,
        )

    def save(self):
        doc = dict(
            tag=self.tag,
            site=self.site,
            flat_map=self.flat_path,
            saved=now(),
            notes="steps replay in order from joint_<tag>.npz z; each step: free = neighbourhood(G, roots, "
            "the step's traces, R), local_solve with every edit so far (later edits of a trace win)",
            steps=self.steps,
            pending=self.pending,
            ok=self.ok,
        )
        json.dump(doc, open(self.edits_path, "w"), indent=1)
        z = self.states[-1]
        np.savez_compressed(
            self.edits_path.replace(".json", "_levels.npz"),
            sub=self.sub,
            z=z,
            z_saved=self.z0,
        )
        return dict(
            self.state(),
            message=f"saved {len(self.steps)} steps, {len(self.pending)} pending, "
            f"{len(self.ok)} OK marks to {self.edits_path}",
        )

    def load(self):
        doc = json.load(open(self.edits_path))
        assert doc["tag"] == self.tag
        self.steps = doc["steps"]
        self.pending, self.ok = doc.get("pending", []), doc.get("ok", [])
        self.states = self.replay(self.steps)
        return dict(
            self.state(),
            message=f"loaded and replayed {len(self.steps)} steps from {self.edits_path}",
        )

    # ------------------------------------------------------------ state
    def state(self):
        touched = {e["trace"] for e in self.all_edits()} | {m["trace"] for m in self.ok}
        done = [bool(touched & set(p["traces"])) for p in self.worklist]
        return dict(
            steps=[
                dict(R=s["R"], n=len(s["edits"]), time=s["time"], solve=s.get("solve"))
                for s in self.steps
            ],
            pending=self.pending,
            edits=self.all_edits(),
            ok=self.ok,
            worklist_done=done,
            priority=self.prio_status,
        )

    def streamlines(self, i, radius=1.0, half=5.0, sigma=1.0):
        """The streamline panel at trace i's root (streamline_view): the traces from and to it, the
        white mesh and the GM boundary in the box of half width `half` mm around it, and the current
        and saved pia on the subsampled traces among them. Positions in mm."""
        root = int(self.rv[i])
        x0 = self.vmm[root]
        fr, to = SV.select(
            self.G, root, x0, self.tv_all, self.pos_all, self.ns_all, self.vox, radius
        )
        r3 = lambda a: np.round(np.asarray(a, float), 3).ravel().tolist()  # noqa: E731
        wv, wf, keep = SV.crop_mesh(self.vmm, self.f, x0 - half, x0 + half)
        wn = self.vnrm[keep]
        lo = np.floor((x0 - half) / self.vox).astype(int)
        hi = np.ceil((x0 + half) / self.vox).astype(int) + 1
        gv, gf = SV.gm_surface(self.ctx, self.vox, lo, hi, sigma)
        z = self.states[-1]
        lines = []
        for kind, ts in (("from", fr), ("to", to)):
            for t in ts:
                n = int(self.ns_all[t])
                k = int(self.sub_of[t])
                lines.append(
                    dict(
                        trace=int(t),
                        kind=kind,
                        cls=int(self.cls_all[t]),
                        fallback=int(self.fallback_all[t]),
                        sub=k,
                        xyz=r3(self.pos_all[t, :n].astype(float) * self.vox),
                    )
                )
        ts = np.r_[fr, to]
        ks = self.sub_of[ts]
        sel = ts[ks >= 0]
        ks = ks[ks >= 0]
        pia = SV.at_depth(self.pos_all[sel], self.ns_all[sel], z[-1, ks], self.vox)
        saved = SV.at_depth(
            self.pos_all[sel], self.ns_all[sel], self.z0[-1, ks], self.vox
        )
        fin = lambda p: p[np.isfinite(p).all(1)]  # noqa: E731
        return dict(
            trace=int(i),
            root=root,
            x0=r3(x0),
            radius=radius,
            half=half,
            tracer=self.tracer_name,
            wm_label=self.wm_label,
            n_from=int(fr.size),
            n_to=int(to.size),
            n_sub=int(sel.size),
            lines=lines,
            pia=r3(fin(pia)),
            pia_saved=r3(fin(saved)),
            wm=dict(v=r3(wv), n=r3(wn), f=wf.ravel().tolist()),
            gm=dict(v=r3(gv), f=gf.ravel().tolist()),
        )

    def mesh_payload(self, level, faces):
        """uint32 header length, JSON header padded to 4 bytes, then float32 positions, normals,
        sulcal depth (inflated level `level`), and with faces: int32 nearest trace per vertex and
        uint32 faces."""
        x = self.infl[int(level)]
        parts = [
            x.astype(np.float32),
            IN.vertex_normals(x, self.f).astype(np.float32),
            self.sulc.astype(np.float32),
        ]
        if faces:
            parts += [self.vtrace, self.f.astype(np.uint32)]
        hdr = dict(
            nv=len(x),
            nf=len(self.f),
            level=int(level),
            faces=bool(faces),
            centre=x.mean(0).tolist(),
            radius=float(np.linalg.norm(x - x.mean(0), axis=1).max()),
        )
        h = json.dumps(hdr).encode()
        h += b" " * (-len(h) % 4)
        return np.uint32(len(h)).tobytes() + h + b"".join(p.tobytes() for p in parts)

    def meta(self):
        self.load_priority()
        wl = [
            dict(
                rank=p["rank"],
                size=p["size"],
                score=round(p["score"], 2),
                mean=round(p["mean"], 3),
                peak=p["peak"],
                traces=p["traces"],
            )
            for p in self.worklist
        ]
        return dict(
            site=self.site,
            tag=self.tag,
            n=self.n,
            uv=np.round(self.uv, 3).tolist(),
            rv=self.rv.tolist(),
            levels=self.infl_names,
            layers=self.layer_names(),
            worklist=wl,
            points=len(self.P),
            vmax=self.vmax,
            spacing=0.16 if "_s0.16" in self.tag else 0.3,
            **self.state(),
        )


class Handler(BaseHTTPRequestHandler):
    editor: Editor
    lock = threading.Lock()

    def log_message(self, fmt, *args):
        pass

    def _send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, fn):
        try:
            with self.lock:
                out = fn()
        except Exception as exc:  # report to the page, keep serving
            import traceback

            traceback.print_exc()
            self._send(
                json.dumps(dict(error=f"{type(exc).__name__}: {exc}")).encode(),
                "application/json",
                500,
            )
            return
        self._send(json.dumps(out, default=float).encode(), "application/json")

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        ed = self.editor
        if u.path in ("/", "/index.html"):
            self._send((STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif u.path == "/meta":
            self._json(ed.meta)
        elif u.path == "/layer":
            self._json(
                lambda: dict(
                    name=q["name"],
                    values=[
                        None if not np.isfinite(x) else round(float(x), 4)
                        for x in ed.layer(q["name"])
                    ],
                )
            )
        elif u.path == "/section":
            self._json(lambda: ed.section_at(int(q["i"]), float(q.get("half", 8.0))))
        elif u.path == "/streamlines":
            self._json(
                lambda: ed.streamlines(
                    int(q["i"]), float(q.get("radius", 1.0)), float(q.get("half", 5.0))
                )
            )
        elif u.path == "/mesh":
            with self.lock:
                body = ed.mesh_payload(int(q.get("level", 0)), q.get("faces") == "1")
            self._send(body, "application/octet-stream")
        else:
            self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n)) if n else {}
        ed = self.editor
        ops = {
            "/edit": lambda: ed.add_edit(req["trace"], float(req["depth"])),
            "/unpend": lambda: ed.unpend(req["k"]),
            "/resolve": lambda: ed.resolve(float(req.get("R", 2.0))),
            "/undo": ed.undo,
            "/ok": lambda: ed.mark_ok(req["trace"]),
            "/save": ed.save,
            "/load": ed.load,
            "/replay": ed.replay_check,
            "/section3d": lambda: ed.section_at(
                int(req["trace"]),
                float(req.get("half", 8.0)),
                req["right"],
                req["up"],
                int(req.get("level", 0)),
            ),
        }
        if self.path not in ops:
            self.send_error(404)
            return
        self._json(ops[self.path])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("work")
    ap.add_argument("tag")
    ap.add_argument("--flat", default=None)
    ap.add_argument("--edits", default=None)
    a = ap.parse_args()
    Handler.editor = Editor(a.work, a.tag, a.flat, a.edits)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    srv.daemon_threads = True
    print(f"serving on http://127.0.0.1:{a.port}/", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
