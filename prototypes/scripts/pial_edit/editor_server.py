"""Pial editor: the local server (plan: reports/Pial editing and edit priority.md, §3 and
§6 step 4). Prototype only; binds 127.0.0.1 only.

A batch design (a feasibility check, not an interactive tool): the user places points,
then asks for a remesh and waits. Loads a session file written by pial_height_run.py
(pial_<patch>_session.npz: white mesh, columns, the march's carried fields, the base
models' tops) and serves static/editor.html.

Requests:
- GET  /volume.nii.gz; /mesh/white, /mesh/pial: uint32 nv, nt, float32 positions (nv x 3,
  world mm), uint32 triangles; /mesh/pial_sign: int8 per pial triangle, +1 where its
  winding gives the outward normal; /state (JSON).
- POST /add {kind: top|contact|free, mm} or {kind: drag, vertex, mm} (surface sessions:
  pial vertex `vertex` moved to mm); /delete {i}; /undo; /redo; /range {mm};
  /base {name}: change the list or the settings only (JSON state back).
- POST /remesh: rebuild every edit from the list (solver.Session.rebuild: tops and
  contacts first, then free edits against the pial surface they give; the result depends
  only on the list) and answer with uint32 header length, a JSON header (state, timings)
  padded to 4 bytes, and float32 world-mm positions of every pial vertex.
- POST /save: the list, settings and last results to the edits file; /load: the list
  from the edits file, then a remesh.
- POST /drag {vertex, mm}: a drag point that refuses to cross (Editor.drag), then the
  same answer as /remesh.
Every answer with positions ends with uint32 count and uint32 indices of the pial
triangles that cross another pial triangle or the white surface, then the same for the
triangles that touch (within touch_mm of a facing pial sheet or of the white surface,
not crossing; POST /touch {mm} sets it); the header's "crossings" has their counts and
clusters.

Usage: editor_server.py <port> <session.npz> <volume.nii.gz> [--base label|onepass_equal]
       [--edits <edits.json>]
"""

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import pial_height as P  # noqa: E402
import solver as S  # noqa: E402

STATIC = Path(__file__).parent / "static"


def load_session(path: str, base: str):
    z = np.load(path)
    if "kind" in z.files and str(z["kind"]) == "surface":
        s = S.SurfaceSession(z["v"], z["f"].astype(np.int64), z["n"], affine=z["affine"])
        s.white_v, s.white_f = z["white_v"], z["white_f"].astype(np.int64)
        return s, dict(bases={"contour": None}, near_face=z["near_face"])
    cols = P.Columns(z["pts"], z["cnt"].astype(np.int64), z["s"].astype(np.float64), z["ridge_s"],
                     z["fluid_s"], z["stop_s"], z["stop_why"].astype(np.int64), z["extended"])
    g = SimpleNamespace(length=z["length"].astype(np.float64), root=z["root"].astype(np.float64))
    bases = {k[2:]: z[k] for k in z.files if k.startswith("t_")}
    s = S.Session(z["v"], z["f"].astype(np.int64), z["n"], g, cols, bases[base], affine=z["affine"])
    return s, dict(bases=bases, near_face=z["near_face"])


class Editor:
    # A vertex has moved (for a drag's local recheck and its sparse answer) above 0.01
    # voxel. The correction field decays but never reaches exactly zero, so with a smaller
    # threshold most of the connected surface counts as moved (M1S1: 58-74k vertices at
    # 1e-6, 22-24k at 1e-3, and 5-15 s per drag). The user's tolerance for a drag against a
    # full remesh is 0.1 voxel; this is ten times inside it. The server's positions are
    # always computed in full; the page's copy and the crossing/touching flags can lag by
    # movements under this until the next full remesh.
    MOVED = 0.01

    def __init__(self, session_path: str, base: str, edits_path: str):
        t = time.perf_counter()
        self.s, self.extra = load_session(session_path, base)
        self.base = self.base_pending = base
        self.range_pending = self.s.prior.range_mm
        self.touch_mm = 0.1  # sheets closer than this (and not crossing) are touching
        self.touch_pending = None  # (vertices, longest edge): a drag's touching still to recheck
        self.edits_path = Path(edits_path)
        self.points: list[dict] = []  # {kind, mm}; results after a remesh
        self.redo_stack: list[dict] = []
        self.dirty = False
        self.timings: dict = {}
        self.pos = self.s.positions()
        self.load_s = time.perf_counter() - t
        print(f"session loaded in {self.load_s:.1f} s: {len(self.s.v):,} vertices", flush=True)

    def mesh_payload(self, which: str) -> bytes:
        if which == "white":
            v, f = getattr(self.s, "white_v", self.s.v), getattr(self.s, "white_f", self.s.f)
        else:
            v, f = self.pos, self.s.f
        head = np.array([len(v), len(f)], np.uint32).tobytes()
        return head + self.s.to_mm(v).astype(np.float32).tobytes() + np.asarray(f).astype(np.uint32).tobytes()

    def face_signs(self) -> bytes:
        """int8 per pial triangle: +1 where its winding (in world mm) gives the outward
        normal, -1 where it is reversed."""
        sign = self._touch_geometry()[1]
        A = self.s.to_mm(np.eye(3)) - self.s.to_mm(np.zeros((1, 3)))
        return (sign * np.sign(np.linalg.det(A))).astype(np.int8).tobytes()

    def state(self) -> dict:
        return dict(base=self.base_pending, bases=list(self.extra["bases"]), range_mm=self.range_pending,
                    touch_mm=self.touch_mm,
                    edits_path=str(self.edits_path), dirty=self.dirty, timings=self.timings,
                    can_undo=bool(self.points), can_redo=bool(self.redo_stack),
                    points=[dict(i=i, **p) for i, p in enumerate(self.points)])

    # -- list and settings
    def add(self, kind: str, mm, vertex=None) -> dict:
        """A point; a drag also names the pial vertex it moves, saved as that vertex's
        position on the unedited surface (from_mm), so a replay finds it again."""
        if kind not in ("top", "contact", "free", "drag"):
            raise ValueError(f"unknown kind {kind}")
        mm = [float(x) for x in mm]
        if not np.all(np.isfinite(mm)):
            raise ValueError(f"position not finite: {mm}")
        p = dict(kind=kind, mm=mm)
        if kind == "drag":
            if not isinstance(self.s, S.SurfaceSession):
                raise ValueError("dragging needs a contoured (surface) session")
            p["from_mm"] = self.s.to_mm(self.s.v[int(vertex)]).tolist()
        self.points.append(p)
        self.redo_stack.clear()
        self.dirty = True
        return self.state()

    def delete(self, i: int) -> dict:
        self.points.pop(int(i))
        self.dirty = True
        return self.state()

    def undo(self) -> dict:
        if self.points:
            self.redo_stack.append(self.points.pop())
            self.dirty = True
        return self.state()

    def redo(self) -> dict:
        if self.redo_stack:
            self.points.append(self.redo_stack.pop())
            self.dirty = True
        return self.state()

    def set_range(self, mm: float) -> dict:
        self.range_pending = float(mm)
        self.dirty = True
        return self.state()

    def set_touch(self, mm: float) -> dict:
        if not (0 < float(mm) < 2):
            raise ValueError(f"touching distance out of range: {mm}")
        self.touch_mm = float(mm)
        self.dirty = True
        return self.state()

    def set_base(self, name: str) -> dict:
        if name not in self.extra["bases"]:
            raise ValueError(f"unknown base {name}")
        self.base_pending = name
        self.dirty = True
        return self.state()

    # -- the remesh
    def _rebuild(self, tm: dict) -> None:
        """Settings, then every edit from the list, then the positions."""
        t0 = time.perf_counter()
        if abs(self.range_pending - self.s.prior.range_mm) > 1e-12:
            self.s.prior.set_range(self.range_pending, self.s.prior.sd_mm)  # rebuild re-adds every observation
            tm["factor_ms"] = 1e3 * (time.perf_counter() - t0)
        if self.base_pending != self.base and self.extra["bases"][self.base_pending] is not None:
            self.s.t_base = np.asarray(self.extra["bases"][self.base_pending], np.float64)
            self.s.thick.set_base(self.s.t_base * self.s.h)
            self.base = self.base_pending
        t1 = time.perf_counter()
        self.made = self.s.rebuild([
            (p["kind"], self.s.to_vox(np.asarray(p["mm"])))
            + ((self.s.to_vox(np.asarray(p["from_mm"])),
                [(self.s.to_vox(np.asarray(u["mm"])), self.s.to_vox(np.asarray(u["from_mm"])))
                 for u in p.get("pushes", [])]) if p["kind"] == "drag" else ())
            for p in self.points])
        tm["edits_ms"] = 1e3 * (time.perf_counter() - t1)
        t2 = time.perf_counter()
        self.pos = self.s.positions()
        tm["positions_ms"] = 1e3 * (time.perf_counter() - t2)
        for p, e in zip(self.points, self.made):
            p.update(fallback=bool(e.fallback), miss_mm=[round(x, 3) for x in e.miss_mm],
                     s_mm=[round(x, 3) for x in e.s_mm], sides=len(e.roots))

    def _white(self):
        return getattr(self.s, "white_v", self.s.v), getattr(self.s, "white_f", self.s.f)

    def _contact_radius(self) -> float:
        """Voxels: the contact radius of a surface session (1 mm otherwise)."""
        return float(getattr(self.s, "contact_radius", 1.0 / self.s.h))

    def _touch_geometry(self):
        """The pial triangles' vertex-to-face table, their outward signs and a locator on
        the white surface, built once."""
        if not hasattr(self, "_tg"):
            if isinstance(self.s, S.SurfaceSession):
                table, sign = self.s.loc.table, self.s.sign
            else:  # a column session: the pial has the white mesh's triangles
                raw = S.face_normals(self.s.v, self.s.f)
                sign = np.where(np.sum(raw * self.s.n[self.s.f].mean(1), 1) < 0, -1.0, 1.0)
                table = self.s.white.table
            wv, wf = self._white()
            self._tg = (table, sign, S.Locator(np.asarray(wv, np.float64), np.asarray(wf, np.int64)))
        return self._tg

    def _crossings(self, tm: dict) -> None:
        """Whole meshes: pial triangles that cross another pial triangle or the white
        surface (self.cross), and touching vertices (self.vtouch; solver's definitions);
        then the summary."""
        from baillarger.mesh_intersect import intersecting_faces

        t0 = time.perf_counter()
        wv, wf = self._white()
        m_self, _ = intersecting_faces(self.pos, self.s.f)
        m_white, _, _ = intersecting_faces(self.pos, self.s.f, wv, wf)
        self.cross = m_self | m_white
        table, sign, white = self._touch_geometry()
        self.vtouch = S.touching_vertices(self.pos, self.s.f, table, sign, self.touch_mm / self.s.h, white)
        self.touch_pending = None
        self._summarise()
        tm["checks_ms"] = tm.get("checks_ms", 0.0) + 1e3 * (time.perf_counter() - t0)

    def _near_region(self, moved: np.ndarray, old: np.ndarray):
        """The vertices whose crossing or touching can have changed after `moved` went
        from `old` to their current positions, the triangles on them, and the longest edge
        used. A vertex can change touching only within delta plus an edge of a moved vertex
        (old or new position), or if a triangle it is on moved (its normal); a triangle can
        change crossing only within an edge of a moved triangle. So: every vertex within
        delta plus two edges of a moved vertex, and every triangle on one of them. (One
        distance query of all vertices against a small tree of the moved points: no
        per-point lists.)"""
        from scipy.spatial import cKDTree

        pos, f = self.pos, self.s.f
        e = S.longest_edge(pos, f)
        R = self.touch_mm / self.s.h + 2 * e
        d, _ = cKDTree(np.vstack([pos[moved], old])).query(pos, distance_upper_bound=R)
        mask = np.isfinite(d)
        mask[moved] = True
        verts = np.flatnonzero(mask)
        sub = np.flatnonzero(mask[f].any(1))
        return verts, sub, e

    def _recheck_cross(self, sub: np.ndarray, tm: dict) -> None:
        from baillarger.mesh_intersect import faces_hitting

        t0 = time.perf_counter()
        wv, wf = self._white()
        self.cross[sub] = faces_hitting(self.pos, self.s.f, sub) | faces_hitting(self.pos, self.s.f, sub, wv, wf)
        tm["cross_ms"] = tm.get("cross_ms", 0.0) + 1e3 * (time.perf_counter() - t0)

    def _recheck_touch(self, verts: np.ndarray, e: float, tm: dict) -> None:
        t0 = time.perf_counter()
        table, sign, white = self._touch_geometry()
        self.vtouch[verts] = S.touching_vertices(self.pos, self.s.f, table, sign, self.touch_mm / self.s.h, white,
                                                 verts, e=e)[verts]
        tm["touch_ms"] = tm.get("touch_ms", 0.0) + 1e3 * (time.perf_counter() - t0)
        tm["checked_vertices"] = int(verts.size)

    def touching(self) -> bytes:
        """The second phase of a drag: touching rechecked in the region the drag left
        pending (with the positions as they are now), then the summary. The answer has no
        positions (an empty sparse list) and the full crossing and touching lists."""
        tm: dict = {}
        if self.touch_pending is not None:
            verts, e = self.touch_pending
            self._recheck_touch(verts, max(e, S.longest_edge(self.pos, self.s.f)), tm)
            self.touch_pending = None
        t0 = time.perf_counter()
        self._summarise()
        tm["summary_ms"] = 1e3 * (time.perf_counter() - t0)
        return self._payload(tm, dict(touching_done=True), rows=np.zeros(0, np.int64))

    def _summarise(self) -> None:
        """Touching triangles from the vertex flags (not crossing), and the places (0.5 mm
        clusters) of crossing and touching, at a contact within the contact radius."""
        from scipy import sparse
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import cKDTree

        self.touch = self.vtouch[self.s.f].any(1) & ~self.cross
        contacts = np.array([self.s.to_vox(np.asarray(p["mm"])) for p in self.points if p["kind"] == "contact"])

        def places(mask):
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                return []
            c = self.pos[self.s.f[idx]].mean(1)
            pr = cKDTree(c).query_pairs(0.5 / self.s.h, output_type="ndarray")
            G = sparse.coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])) if len(pr) else
                                  (np.zeros(0), (np.zeros(0, int), np.zeros(0, int))), shape=(idx.size,) * 2)
            n, lab = connected_components(G, directed=False)
            out = []
            for k in range(n):
                cen = c[lab == k].mean(0)
                near = bool(len(contacts)) and bool(np.min(np.linalg.norm(contacts - cen, axis=1)) < self._contact_radius())
                out.append(dict(mm=[round(float(x), 3) for x in self.s.to_mm(cen)], faces=int(np.sum(lab == k)),
                                at_contact=near))
            out.sort(key=lambda d: (d["at_contact"], -d["faces"]))
            return out

        cl, tl = places(self.cross), places(self.touch)
        self.crossings = dict(faces=int(self.cross.sum()), clusters=cl[:200], n_clusters=len(cl),
                              at_contact=sum(c["at_contact"] for c in cl),
                              touching_faces=int(self.touch.sum()), touching_places=len(tl),
                              touching_at_contact=sum(c["at_contact"] for c in tl), touch_mm=self.touch_mm)

    def _payload(self, tm: dict, extra: dict | None = None, rows: np.ndarray | None = None) -> bytes:
        """Header, then positions: all vertices, or (rows given) only those, as uint32 count,
        uint32 indices, float32 world mm; then the crossing and the touching triangles."""
        tm["total_ms"] = sum(v for k, v in tm.items() if k.endswith("_ms") and k != "total_ms")
        tm["edits"] = len(self.points)
        self.timings = {k: round(v, 1) for k, v in tm.items()}
        self.dirty = False
        head = json.dumps(dict(state=self.state(), crossings=self.crossings, sparse=rows is not None,
                               **(extra or {}))).encode()
        head += b" " * (-len(head) % 4)
        if rows is None:
            body = self.s.to_mm(self.pos).astype(np.float32).tobytes()
        else:
            rows = np.asarray(rows, np.uint32)
            body = (np.array([rows.size], np.uint32).tobytes() + rows.tobytes()
                    + self.s.to_mm(self.pos[rows]).astype(np.float32).tobytes())
        idx = np.flatnonzero(self.cross).astype(np.uint32)
        tdx = np.flatnonzero(self.touch).astype(np.uint32)
        return (np.array([len(head)], np.uint32).tobytes() + head + body
                + np.array([idx.size], np.uint32).tobytes() + idx.tobytes()
                + np.array([tdx.size], np.uint32).tobytes() + tdx.tobytes())

    def remesh(self) -> bytes:
        tm: dict = {}
        self._rebuild(tm)
        self._crossings(tm)
        return self._payload(tm)

    PLACE_MM = 0.5  # a crossing within this of one that was there before is that one, shifted
    PUSH_ROUNDS, PUSH_SPACING_MM, PUSH_MAX = 3, 0.3, 60

    def drag(self, vertex: int, mm, push: bool = False) -> bytes:
        """A drag of pial vertex `vertex` toward mm that refuses to cross, added without a
        rebuild (SurfaceSession.add_drag: the same result as appending it and
        rebuilding). Refused are crossings in a new place: triangles that cross (pial
        against pial or against the white surface), did not cross before, and are more
        than PLACE_MM from every triangle that crossed before (the spacing of the crossing
        places); an existing crossing that only shifts at its edge does not count.

        Without push, a refused drag is pulled back along its path by bisection (8 steps)
        to the furthest point that makes none, and the contact points that hold it are
        named (_holders). With push, the white surface still stops it (the same
        bisection, against the white surface only); then, up to PUSH_ROUNDS times, where
        it crosses another pial sheet that faces it, vertices of that sheet are pushed
        ahead of it (_push). If crossings remain after that (the pushes can make their own,
        where the pushed sheet meets a third), the pushes are dropped and it is a plain
        drag, pulled back as above, so pushing never places a drag less far. The pushes
        are saved with the drag.

        Only observed values change during the searches (pushes add solved columns). Then
        crossing and touching are rechecked near what moved (more than MOVED), and only
        those vertices are sent. It differs from a full remesh only where a movement under
        MOVED (0.01 voxel) changes a crossing or touching flag, or in the page's copy of
        positions, by less than MOVED."""
        from baillarger.mesh_intersect import faces_hitting
        from scipy.spatial import cKDTree

        tm: dict = {}
        if self.dirty or not hasattr(self, "vtouch"):  # bring everything up to date first
            self._rebuild(tm)
            self._crossings(tm)
            full = True
        else:
            full = False
        before = self.cross.copy()
        prev = self.pos.copy()
        r = int(vertex)
        start = prev[r].copy()
        self.add("drag", mm, r)
        p = self.points[-1]
        t1 = time.perf_counter()
        e = self.s.add_drag(self.s.to_vox(np.asarray(p["mm"])), self.s.to_vox(np.asarray(p["from_mm"])))
        for q, ed in zip(self.points, self.s.edits):  # an earlier drag of r is superseded
            if ed.kind == "drag":
                q.update(fallback=False, miss_mm=[round(x, 3) for x in ed.miss_mm], s_mm=[], sides=len(ed.roots))
        tm["edits_ms"] = 1e3 * (time.perf_counter() - t1)
        target = self.s.to_vox(np.asarray(p["mm"]))
        wv, wf = self._white()
        f = self.s.f
        old = np.flatnonzero(before)
        old_tree = cKDTree(prev[f[old]].mean(1)) if old.size else None
        place = self.PLACE_MM / self.s.h
        t0 = time.perf_counter()

        def new_crossings(pos):
            """Triangles crossing in a new place: (against pial, against the white surface)."""
            moved = np.flatnonzero(np.linalg.norm(pos - prev, axis=1) > self.MOVED)
            if moved.size == 0:
                return np.zeros(0, np.int64), np.zeros(0, np.int64)
            sub = np.flatnonzero(np.isin(f, moved).any(1))
            out = []
            for hit in (faces_hitting(pos, f, sub), faces_hitting(pos, f, sub, wv, wf)):
                F = sub[hit & ~before[sub]]
                if F.size and old_tree is not None:
                    F = F[old_tree.query(pos[f[F]].mean(1), distance_upper_bound=place)[0] > place]
                out.append(F)
            return out[0], out[1]

        trials = 0

        def bisect(apply, white_only):
            """The largest s in [0, 1] (8 halvings) at which apply(s) makes no new crossing
            (against the white surface only, or any); applied. Also the last failure."""
            nonlocal trials
            lo, hi, fail = 0.0, 1.0, None
            for _ in range(8):
                mid = 0.5 * (lo + hi)
                apply(mid)
                trials += 1
                pos = self.s.positions()
                c = new_crossings(pos)
                if c[1].size or (not white_only and c[0].size):
                    hi, fail = mid, (pos, c)
                else:
                    lo = mid
            apply(lo)
            return lo, fail

        def drag_to(s):
            self.s.set_drag_target(e, start + s * (target - start))

        self.pos = self.s.positions()
        c0 = new_crossings(self.pos)
        n0 = int(c0[0].size + c0[1].size)
        trials, frac, held_by, fail, push_failed = 1, 1.0, [], None, False
        if n0 and not push:
            frac, fail = bisect(drag_to, white_only=False)
            self.s.set_drag_target(e, target)  # which contact points hold it
            t3 = time.perf_counter()
            held_by = self._holders(start, lambda pos: sum(F.size for F in new_crossings(pos)))
            tm["holders_ms"] = 1e3 * (time.perf_counter() - t3)
            drag_to(frac)
        elif n0:
            if c0[1].size:  # the white surface stops it
                frac, fail = bisect(drag_to, white_only=True)
            pos = self.s.positions()
            t3 = time.perf_counter()
            for _ in range(self.PUSH_ROUNDS):
                pp = new_crossings(pos)[0]
                if pp.size == 0 or not self._push(e, r, pos, pp):
                    break
                pos = self.s.positions()
            tm["push_ms"] = 1e3 * (time.perf_counter() - t3)
            c = new_crossings(pos)
            if c[0].size or c[1].size:  # the pushes did not clear it: a plain drag instead
                for _, obs, _ in e.pushes:
                    for name, k in obs:
                        self.s.free[int(name[-1])].remove(k)
                e.pushes = []
                drag_to(1.0)
                frac, fail = bisect(drag_to, white_only=False)
                push_failed = True
        p["mm"] = [float(x) for x in e.mm]
        if e.pushes:
            p["pushes"] = [dict(from_mm=self.s.to_mm(self.s.v[q]).tolist(), mm=self.s.to_mm(x).tolist())
                           for q, _, x in e.pushes]
        else:
            p.pop("pushes", None)
        self.pos = self.s.positions()
        tm["pullback_ms"] = (1e3 * (time.perf_counter() - t0) - tm.get("holders_ms", 0.0)
                             - tm.get("push_ms", 0.0))
        blocked = None
        if fail is not None:  # where the first refused crossing was, nearest the vertex
            pos_f, (pp, pw) = fail
            F = np.concatenate([pp, pw])
            cen = pos_f[f[F]].mean(1)
            j = int(np.argmin(np.linalg.norm(cen - start, axis=1)))
            blocked = dict(mm=[round(float(x), 2) for x in self.s.to_mm(cen[j])],
                           away_mm=round(float(np.linalg.norm(cen[j] - start)) * self.s.h, 2),
                           against="white surface" if j >= pp.size else "pial surface")
        moved = np.flatnonzero(np.linalg.norm(self.pos - prev, axis=1) > self.MOVED)
        t2 = time.perf_counter()
        if moved.size:
            verts, sub, edge = self._near_region(moved, prev[moved])
            tm["region_ms"] = 1e3 * (time.perf_counter() - t2)
            self._recheck_cross(sub, tm)
            # touching follows in a second request (/touching); a pending region from a
            # drag not yet followed up is merged in
            if self.touch_pending is not None:
                verts = np.union1d(verts, self.touch_pending[0])
                edge = max(edge, self.touch_pending[1])
            self.touch_pending = (verts, edge)
        self._summarise()
        p["pulled_back"] = frac < 1.0
        p["fraction"] = round(frac, 4)
        for k, v in (("held_by", held_by if n0 and not push else None), ("blocked", blocked)):
            if v is None:
                p.pop(k, None)
            else:
                p[k] = v
        info = dict(requested_mm=[float(v) for v in mm], fraction=round(frac, 4), would_cross=n0, trials=trials,
                    moved=int(moved.size), touching_pending=self.touch_pending is not None,
                    held_by=held_by, blocked=blocked, push=bool(push), pushes=len(e.pushes),
                    push_failed=push_failed)
        return self._payload(tm, dict(drag=info), rows=None if full else moved)

    def _push(self, e, r: int, pos, F) -> bool:
        """One round of pushes for drag e of vertex r, at positions pos, where triangles F
        cross another pial sheet. The dragged sheet is the vertices within 1 mm of F whose
        outward normal agrees with r's (dot > 0.5); the other sheet, those within 0.6 mm
        whose normal faces it (dot < -0.5, the touching test's rule). Each vertex q of the
        other sheet less than a gap (twice the touch distance) in front of the dragged
        sheet (measured along the outward normal n of the nearest dragged-sheet vertex p)
        is to go to q + (gap - (q - p).n) n. Of those, a push already made is moved there
        if that is further; new ones are taken deepest first, PUSH_SPACING_MM apart from
        each other and from the pushes there are, up to PUSH_MAX for the drag. False if
        nothing changed."""
        from scipy.spatial import cKDTree

        f, h = self.s.f, self.s.h
        fn = S.face_normals(pos, f, self.s.sign)
        cen = pos[f[F]].mean(1)
        tree = cKDTree(pos)
        near = lambda R: np.unique(np.concatenate([np.asarray(x, np.int64) for x in tree.query_ball_point(cen, R)]))  # noqa: E731
        region = near(1.0 / h)
        sub = np.flatnonzero(np.isin(f, region).any(1))
        vn = S.vertex_normals_from(pos, f[sub], fn[sub])  # right on the region's vertices
        rf = np.flatnonzero((f == r).any(1))
        nr = S.vertex_normals_from(pos, f[rf], fn[rf])[r]
        own = region[vn[region] @ nr > -S.FACING]
        other = np.intersect1d(near(0.6 / h), region)
        other = other[vn[other] @ nr < S.FACING]
        if own.size == 0 or other.size == 0:
            return False
        d, j = cKDTree(pos[own]).query(pos[other])
        pn = vn[own[j]]
        sd = np.sum((pos[other] - pos[own[j]]) * pn, 1)
        gap = 2 * self.touch_mm / h
        need = np.flatnonzero(sd < gap)
        if need.size == 0:
            return False
        tgt = pos[other[need]] + (gap - sd[need])[:, None] * pn[need]
        changed = False
        index = {q: k for k, (q, _, _) in enumerate(e.pushes)}
        kept = [pos[q] for q, _, _ in e.pushes]
        new_q, new_x = [], []
        space = self.PUSH_SPACING_MM / h
        for k in np.argsort(sd[need]):
            q, x = int(other[need[k]]), tgt[k]
            if q in index:
                x0 = e.pushes[index[q]][2]
                if np.dot(x - x0, pn[need[k]]) > self.MOVED:
                    self.s.set_push_target(e, index[q], x)
                    changed = True
                continue
            if len(e.pushes) + len(new_q) >= self.PUSH_MAX:
                break
            if kept and np.min(np.linalg.norm(np.asarray(kept) - pos[q], axis=1)) < space:
                continue
            kept.append(pos[q])
            new_q.append(q)
            new_x.append(x)
        if new_q:
            self.s.add_pushes(e, new_x, new_q)
            changed = True
        return changed

    def _holders(self, start, new_crossings) -> list[dict]:
        """The contact points that hold a refused drag (its target set in full): a set of
        contact points within twice the prior's range of the dragged vertex without which
        the full drag makes no new crossing, none of which can be put back. Taken out
        nearest first until the drag is free, then each put back again if the drag stays
        free without it (farthest first). A contact point is taken out by popping its
        observations, their cached columns kept, so nothing is solved again; all are put
        back at the end. Empty if taking out all of them does not free the drag."""
        R = 2 * self.s.prior.range_mm
        cand = []
        for i, (q, ed) in enumerate(zip(self.points, self.s.edits)):
            if q["kind"] != "contact" or not ed.obs:
                continue
            d = float(np.linalg.norm(self.s.to_vox(np.asarray(q["mm"])) - start)) * self.s.h
            if d <= R:
                cand.append((d, i, ed))
        cand.sort(key=lambda c: c[0])
        out: dict[int, tuple] = {}  # taken out: point index -> (distance, saved observations)

        def take(c) -> None:
            saved = [(self.s.free[int(name[-1])], k) for name, k in c[2].obs]
            out[c[1]] = (c[0], [(fd, k, fd.obs.pop(k), fd.cols.pop(k)) for fd, k in saved])

        def put(i) -> None:
            for fd, k, o, col in out.pop(i)[1]:
                fd.obs[k], fd.cols[k] = o, col

        free = lambda: new_crossings(self.s.positions()) == 0  # noqa: E731
        try:
            for c in cand:
                take(c)
                if free():
                    break
            else:
                return []
            for c in reversed([c for c in cand if c[1] in out]):
                put(c[1])
                if not free():
                    take(c)
            return [dict(i=i, mm_away=round(d, 2)) for i, (d, _) in sorted(out.items(), key=lambda x: x[1][0])]
        finally:
            for i in list(out):
                put(i)

    def save(self) -> dict:
        self.edits_path.write_text(json.dumps(dict(range_mm=self.range_pending, base=self.base_pending,
                                                   edits=self.points), indent=1))
        return self.state()

    def load(self) -> bytes:
        d = json.loads(self.edits_path.read_text())
        self.points = [{k: e[k] for k in ("kind", "mm", "from_mm", "pushes") if k in e} for e in d["edits"]]
        self.range_pending = float(d.get("range_mm", self.range_pending))
        self.base_pending = d.get("base", self.base_pending)
        self.redo_stack.clear()
        return self.remesh()


class Handler(BaseHTTPRequestHandler):
    editor: Editor
    volume: Path
    lock = threading.Lock()  # the session is not thread safe

    def log_message(self, fmt, *args):
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?")[0]
        ed = self.editor
        if p in ("/", "/index.html"):
            self._send((STATIC / "editor.html").read_bytes(), "text/html; charset=utf-8")
        elif p == "/volume.nii.gz":
            self._send(self.volume.read_bytes(), "application/octet-stream")
        elif p in ("/mesh/white", "/mesh/pial"):
            with self.lock:
                self._send(ed.mesh_payload(p.rsplit("/", 1)[1]), "application/octet-stream")
        elif p == "/mesh/pial_sign":
            with self.lock:
                self._send(ed.face_signs(), "application/octet-stream")
        elif p == "/state":
            self._send(json.dumps(ed.state()).encode(), "application/json")
        else:
            self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n)) if n else {}
        ed = self.editor
        ops = {
            "/add": lambda: ed.add(req["kind"], req["mm"], req.get("vertex")), "/delete": lambda: ed.delete(req["i"]),
            "/undo": ed.undo, "/redo": ed.redo, "/range": lambda: ed.set_range(req["mm"]),
            "/base": lambda: ed.set_base(req["name"]), "/save": ed.save,
            "/touch": lambda: ed.set_touch(req["mm"]),
            "/remesh": ed.remesh, "/load": ed.load,
            "/drag": lambda: ed.drag(req["vertex"], req["mm"], bool(req.get("push", False))),
            "/touching": ed.touching,
        }
        if self.path not in ops:
            self.send_error(404)
            return
        try:
            with self.lock:
                out = ops[self.path]()
        except Exception as exc:  # report to the page, keep serving
            self._send(json.dumps(dict(error=f"{type(exc).__name__}: {exc}")).encode(), "application/json", 500)
            return
        if isinstance(out, bytes):
            self._send(out, "application/octet-stream")
        else:
            self._send(json.dumps(out).encode(), "application/json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("session")
    ap.add_argument("volume")
    ap.add_argument("--base", default="label")
    ap.add_argument("--edits", default=None)
    a = ap.parse_args()
    edits = a.edits or str(Path(a.session).with_name(Path(a.session).name.replace("_session.npz", "_edits.json")))
    Handler.editor = Editor(a.session, a.base, edits)
    Handler.volume = Path(a.volume)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    srv.daemon_threads = True
    print(f"serving on http://127.0.0.1:{a.port}/", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
