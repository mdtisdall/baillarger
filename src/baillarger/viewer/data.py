"""A site's files, read for viewing (plan: reports/Crossing paths and contacts from
laminar profiles.md, *A pia editor on an inflated surface*). Nothing here edits or
re-solves anything.

Inputs (only the column volume is needed):
- column volume (column_volume_<name>.npz, prototypes/scripts/column_volume.py): the
  white mesh (box voxels), every trace (positions, sample counts, class, primary or
  fallback), echo 2 along the traces, the cortex label, the horizons, the user's
  points; the tracer and WM label it was made with, if saved.
- joint result (joint_<tag>.npz): the subsampled traces `sub`, their levels `z` (the
  last is the pia), certification, neighbour pairs and ties. Without it every trace is
  shown and there are no levels.
- single-surface horizon (pia_<name>.npz, `c` per trace).
- priority map (edit_priority_<tag>.npz, on the same `sub`): its rank, doubt and
  combined layers, and the QPBO interval where the pia is not certified.
- echo 2 of the full grid (baillarger.data.echo_path(2) by default): cropped to the box
  for the volume views, and its affine for world mm.

The white surface (baillarger.viewer.surface.Surface) holds the mesh's coordinates,
graph, flat map and inflation (cached, keyed by a hash of the mesh, so that a rebuilt
column volume gets new ones); Site adds the traces. World mm: the box's crop of the
echo-2 affine; box mm: voxel index times the voxel size (the sections and the 3D
surface use box mm, the volume views world mm).
"""

import gzip
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from baillarger.data import crop_affine, echo_path
from baillarger.streamlines import STEP_MM
from baillarger.viewer import filters as VF
from baillarger.viewer import inflation as IN
from baillarger.viewer import sections as PS
from baillarger.viewer import streamlines as SV
from baillarger.viewer.surface import Surface, gifti_bytes

type Float = npt.NDArray[np.float64]
type Arr = npt.NDArray[Any]

CLASSES = ("open", "crossing", "own bank", "to the pass's contact", "failed")
NEAREST_MM = 0.5  # a vertex shows the nearest subsampled trace rooted within this
LOCATE_MM = 1.0  # a WM selection takes the nearest shown root within this


def nifti(path: Path) -> nib.nifti1.Nifti1Image:
    img = nib.load(path)
    if not isinstance(img, nib.nifti1.Nifti1Image):
        raise ValueError(f"{path}: not a NIfTI-1 image")
    return img


class Site:
    """One site's data for the viewer; see the module docstring."""

    def __init__(
        self,
        column_volume: Path,
        cache: Path,
        joint: Path | None = None,
        pia: Path | None = None,
        priority: Path | None = None,
        echo: Path | None = None,
        log: Callable[[str], object] = print,
    ) -> None:
        t0 = time.time()
        self.cache = Path(cache)
        self.column_volume = Path(column_volume)
        stem = Path(column_volume).name.removesuffix(".npz")
        self.name = stem.removeprefix("column_volume_")
        self.log = log
        Z = np.load(column_volume)
        self.lo: Arr = Z["lo"]
        # world mm
        self.echo_path = Path(echo) if echo is not None else echo_path(2)
        self.surf = Surface(
            Z["v"].astype(float),
            Z["f"],
            Z["vox"].astype(float),
            crop_affine(nifti(self.echo_path).affine, self.lo),
            self.cache,
            self.name,
            log=log,
        )
        self.vox, self.affine = self.surf.vox, self.surf.affine
        self.v_vox, self.vmm, self.f = self.surf.v_vox, self.surf.vmm, self.surf.f
        self.G = self.surf.G
        self.vnrm: Float = Z["nrm"].astype(float)
        self.tv_all: Arr = Z["tv"]
        self.pos_all: Arr = Z["pos"]
        self.ns_all: Arr = Z["ns"]
        self.cls_all: Arr = Z["cls"]
        self.fallback_all: Arr = Z["tracer"]
        self.ctx: npt.NDArray[np.bool_] = Z["ctx"].astype(bool)
        self.shape = tuple(int(x) for x in self.ctx.shape)
        self.P: Float = Z["P"].astype(float)
        fv0: Arr = Z["fv"][0]
        top_all: Arr = Z["H_top"]
        self.tracer_name = (
            str(Z["tracer_name"]) if "tracer_name" in Z.files else "not saved"
        )
        self.wm_label = str(Z["wm_label"]) if "wm_label" in Z.files else "not saved"
        # the traces shown: the joint result's subsample, or all
        self.joint = None if joint is None else dict(np.load(joint))
        if self.joint is not None:
            self.sub: Arr = self.joint["sub"]
            self.z: Arr | None = self.joint["z"]
            self.pairs: Arr = self.joint["pairs"]
            self.ties: Arr = self.joint["ties"]
            self.cert_saved: Arr = self.joint["cert"][-1].astype(bool)
        else:
            self.sub = np.flatnonzero(self.tv_all >= 0)
            self.z = None
            self.pairs = np.zeros((0, 2), np.int64)
            self.ties = np.zeros((0, 2), np.int64)
            self.cert_saved = np.ones(self.sub.size, bool)
        self.n = int(self.sub.size)
        self.sub_of = -np.ones(len(self.tv_all), np.int64)
        self.sub_of[self.sub] = np.arange(self.n)
        self.rv: Arr = self.tv_all[self.sub]
        self.nrm: Float = self.vnrm[self.rv]
        self.roots_of = -np.ones(len(self.vmm), np.int64)
        self.roots_of[self.rv] = np.arange(self.n)
        self.pos: Arr = self.pos_all[self.sub]
        self.ns: Arr = self.ns_all[self.sub]
        self.echo: Arr = fv0[self.sub]
        self.vmax = float(np.nanpercentile(np.abs(self.echo), 97))
        self.top: Float = top_all[self.sub].astype(float)
        self.single: Float | None = (
            None if pia is None else np.load(pia)["c"][self.sub].astype(float)
        )
        self.partner: dict[int, int] = {}
        for a, b in self.ties:
            self.partner.setdefault(int(a), int(b))
            self.partner.setdefault(int(b), int(a))
        self.priority_path = priority
        self.load_priority()
        self.uv: Float = self.surf.uv_v[self.rv]
        self.T: Float = self.surf.T_v[self.rv]
        g = np.einsum("nij,nik->njk", self.T, self.T)
        self.log2_area: Float = -0.5 * np.log2(np.linalg.det(g))
        self.infl, self.infl_names = self.surf.infl, self.surf.infl_names
        self.vf = self.surf.vf
        d, _, src = dijkstra(
            self.G,
            indices=self.rv,
            min_only=True,
            return_predecessors=True,
            limit=NEAREST_MM,
        )
        self.vtrace = np.where(
            np.isfinite(d), self.roots_of[np.maximum(src, 0)], -1
        ).astype(np.int32)
        self.root_tree = cKDTree(self.world(self.v_vox[self.rv]))
        self.hits = self.point_hits()
        self._bytes: dict[str, bytes] = {}
        # filters (baillarger.viewer.filters): none shown until one is chosen
        self.jobs = VF.FilterJobs(
            self.column_volume, self.cache, self.name, self.surf.key, log=log
        )
        self.filter: VF.Result | None = None
        self.frow = -np.ones(self.n, np.int64)
        self.fvmax = 1.0
        log(
            f"site {self.name}: {self.n:,} traces shown of {len(self.tv_all):,}, "
            f"tracer {self.tracer_name}, WM label {self.wm_label} "
            f"({time.time() - t0:.0f} s)"
        )

    # ------------------------------------------------------------ coordinates
    def world(self, p_vox: npt.ArrayLike) -> Float:
        return self.surf.world(p_vox)

    def locate(self, mm: npt.ArrayLike) -> int | None:
        """The shown trace whose root is nearest world point mm, within LOCATE_MM."""
        d, k = self.root_tree.query(np.asarray(mm, float))
        return int(k) if d <= LOCATE_MM else None

    # ------------------------------------------------------------ the WM selection
    def vertex(self, k: int) -> dict[str, Any]:
        """The WM selection at white-mesh vertex k: its world point, and the shown
        trace that the sections and streamlines use there: the one rooted nearest
        along the mesh within NEAREST_MM (vtrace), else the root nearest in space
        within LOCATE_MM, else none. root_mm: the distance from the point to that
        trace's root (mm)."""
        k = int(k)
        p = self.world(self.v_vox[k])[0]
        t = int(self.vtrace[k])
        if t < 0:
            d, j = self.root_tree.query(p)
            t = int(j) if d <= LOCATE_MM else -1
        root = None
        if t >= 0:
            root = round(
                float(np.linalg.norm(self.world(self.v_vox[self.rv[t]])[0] - p)), 3
            )
        return dict(
            vertex=k,
            world_mm=[round(float(x), 3) for x in p],
            trace=t if t >= 0 else None,
            root_mm=root,
        )

    def snap(self, mm: npt.ArrayLike) -> dict[str, Any]:
        """The WM selection at the white-mesh vertex nearest world point mm."""
        return self.vertex(self.surf.nearest(mm))

    # ------------------------------------------------------------ layers
    def load_priority(self) -> None:
        self.prio: dict[str, Float] = {}
        self.cert = self.cert_saved
        self.lo_mm: Float | None = None
        self.hi_mm: Float | None = None
        if self.priority_path is not None and Path(self.priority_path).exists():
            with np.load(self.priority_path) as Q:
                if not np.array_equal(Q["sub"], self.sub):
                    raise ValueError("the priority map is on other traces than shown")
                self.prio = {
                    k: Q[k].astype(float)
                    for k in Q.files
                    if k == "combined" or k.startswith(("rank_", "doubt_"))
                }
                self.cert = Q["cert"].astype(bool)
                self.lo_mm, self.hi_mm = Q["lo"].astype(float), Q["hi"].astype(float)

    def use_filter(self, key: str | None) -> dict[str, Any]:
        """Show filter result `key` (a finished one), or the raw echo 2 (None)."""
        if key is None:
            self.filter = None
            self.frow = -np.ones(self.n, np.int64)
        else:
            r = self.jobs.get(key)
            row = -np.ones(len(self.tv_all), np.int64)
            row[r.keep] = np.arange(len(r.keep))
            self.filter, self.frow = r, row[self.sub]
            shown = r.prof[self.frow[self.frow >= 0]]
            fin = np.abs(shown[np.isfinite(shown)])
            self.fvmax = float(np.percentile(fin, 97)) if fin.size else 1.0
        return self.filter_meta()

    def filter_meta(self) -> dict[str, Any]:
        r = self.filter
        return dict(
            key=None if r is None else r.key,
            label="raw echo 2" if r is None else r.params.label(),
            frame="depth" if r is None else r.frame,
            shown=int((self.frow >= 0).sum()),
            layers=list(self.layers()),
        )

    def profile(self, t: int) -> Float:
        """Trace t's profile as shown: the filter's row (empty where the filter has
        none), else the raw echo 2."""
        if self.filter is None:
            out: Float = self.echo[t, : self.ns[t]].astype(float)
            return out
        k = int(self.frow[t])
        if k < 0:
            return np.zeros(0)
        row = self.filter.prof[k]
        fin = np.flatnonzero(np.isfinite(row))
        return row[: fin[-1] + 1] if fin.size else np.zeros(0)

    def depths(self, t: int, d: Arr) -> Float:
        """Depths (mm from the root) on trace t in the shown frame."""
        d = np.asarray(d, float)
        if self.filter is None:
            return d
        k = int(self.frow[t])
        if k < 0:
            return np.full(d.shape, np.nan)
        return self.filter.depth_to_frame(k, d)

    def layers(self) -> dict[str, Float]:
        out = dict(self.prio)
        if self.z is not None:
            out["not certified"] = (~self.cert).astype(float)
            if self.single is not None:
                out["joint - single-surface (mm)"] = self.z[-1] - self.single
            out["pia depth (mm)"] = self.z[-1].astype(float)
        out["open top (H_top set)"] = np.isfinite(self.top).astype(float)
        out["class at the root"] = self.cls_all[self.sub].astype(float)
        out["fallback tracer"] = self.fallback_all[self.sub].astype(float)
        out["log2 map area factor"] = self.log2_area
        if self.filter is not None:
            for k, v in self.filter.layers.items():
                ok = self.frow >= 0
                out[VF.PREFIX + k] = np.where(ok, v[np.maximum(self.frow, 0)], np.nan)
        return out

    # ------------------------------------------------------------ points
    def point_hits(self, within: float = 0.3) -> dict[int, list[tuple[int, float]]]:
        """Per shown trace: (point number from 1, depth mm) of each of the user's
        points whose nearest sample along the trace is within `within` mm."""
        out: dict[int, list[tuple[int, float]]] = {}
        if not len(self.P):
            return out
        tree = cKDTree(self.P * self.vox)
        for i in range(self.n):
            d_, k_ = tree.query(self.pos[i, : self.ns[i]].astype(float) * self.vox)
            d, k = np.atleast_1d(d_), np.atleast_1d(k_)
            j = int(np.argmin(d))
            if d[j] <= within:
                out.setdefault(i, []).append((int(k[j]) + 1, j * STEP_MM))
        return out

    # ------------------------------------------------------------ sections
    def screen_directions(
        self, i: int, right: npt.ArrayLike, up: npt.ArrayLike, level: int
    ) -> tuple[Float, Float, float]:
        x, r = self.infl[int(level)], int(self.rv[i])
        T = np.c_[
            IN.transfer(np.asarray(right, float), r, x, self.vmm, self.f, self.vf),
            IN.transfer(np.asarray(up, float), r, x, self.vmm, self.f, self.vf),
        ]
        return PS.directions(T, self.nrm[i])

    def cut(self, i: int, t: Float, half: float) -> tuple[Arr, Float, Float]:
        r = int(self.rv[i])
        return PS.section_curve(
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

    def viewer_planes(self) -> dict[str, Float]:
        """The normals (box mm) of the volume pane's slice planes: NiiVue shows voxel
        planes, each named for the world axis its voxel axis is nearest."""
        A = self.affine[:3, :3] / np.linalg.norm(self.affine[:3, :3], axis=0)
        out: dict[str, Float] = {}
        for name, w in (("sagittal", 0), ("coronal", 1), ("axial", 2)):
            out[name] = np.eye(3)[int(np.argmax(np.abs(A[w])))]
        return out

    def section_at(
        self,
        i: int,
        half: float = 8.0,
        right: npt.ArrayLike | None = None,
        up: npt.ArrayLike | None = None,
        level: int = 0,
        rot: float = 0.0,
    ) -> dict[str, Any]:
        """The two sections at shown trace i: along the flat map's axes, or along the
        screen's right and up on inflated level `level` carried to the true mesh,
        both turned by `rot` degrees about the normal (from u toward v); at a
        contact, the facing trace's sections through the same planes. curves: the
        curves on the white mesh that the u and v sections follow (world mm); snap:
        per volume-pane plane, the rotation nearest `rot` that puts one axis in it
        (sections.snap)."""
        i = int(i)
        if right is None or up is None:
            tu, tv, ang = PS.directions(self.T[i], self.nrm[i])
            frame = "flat map axes"
        else:
            tu, tv, ang = self.screen_directions(i, right, up, level)
            frame = f"screen axes on the {self.infl_names[int(level)]}"
        snaps: dict[str, Any] = {}
        for name, g in self.viewer_planes().items():
            sn = PS.snap(tu, tv, self.nrm[i], g, rot)
            snaps[name] = (
                None
                if sn is None
                else dict(rot=round(sn[0], 2), axis=sn[1], tilt_deg=round(sn[2], 2))
            )
        tu, tv = PS.rotate(tu, tv, rot)
        out: dict[str, Any] = dict(
            trace=i,
            angle_error_deg=round(ang, 2),
            frame=frame,
            rot=float(rot),
            snap=snaps,
            info=self.info(i),
        )
        curves: dict[str, Any] = {}
        for name, t in (("u", tu), ("v", tv)):
            tr, dist, c = self.cut(i, t, half)
            out[name] = self.pack(tr, dist)
            curves[name] = np.round(self.world(c / self.vox), 4).tolist()
        out["curves"] = curves
        j, via = self.facing(i, out)
        if j is not None:
            nj = self.nrm[j]
            fc: dict[str, Any] = dict(trace=j, via=via, info=self.info(j))
            for name, t in (("u", tu), ("v", tv)):
                tj = t - (t @ nj) * nj
                fc[name] = self.pack(*self.cut(j, tj / np.linalg.norm(tj), half)[:2])
            out["facing"] = fc
        return out

    def facing(self, i: int, out: dict[str, Any]) -> tuple[int | None, str | None]:
        if i in self.partner:
            return self.partner[i], f"tie of trace {i}"
        best: tuple[float, int] | None = None
        for name in ("u", "v"):
            for t, d in zip(out[name]["traces"], out[name]["dist"], strict=True):
                if t in self.partner and d is not None and abs(d) <= 1.0:
                    if best is None or abs(d) < best[0]:
                        best = (abs(d), t)
        if best is None:
            return None, None
        return (
            self.partner[best[1]],
            f"tie of trace {best[1]}, {best[0]:.2f} mm from the crosshair",
        )

    def pack(self, tr: Arr, dist: Float) -> dict[str, Any]:
        tr = np.asarray(tr, int)

        def r2(a: Arr) -> list[float | None]:
            return [None if not np.isfinite(x) else round(float(x), 3) for x in a]

        def fr(a: Arr) -> Float:  # a depth per trace of tr, in the shown frame
            a = np.asarray(a, float)
            return np.array(
                [self.depths(int(t), a[k : k + 1])[0] for k, t in enumerate(tr)]
            )

        vmax = self.vmax if self.filter is None else self.fvmax
        prof = [self.profile(int(t)) for t in tr]
        img = [
            np.clip(np.round(np.nan_to_num(p) / vmax * 127), -127, 127)
            .astype(int)
            .tolist()
            for p in prof
        ]
        unc = ~self.cert[tr]
        nan = np.full(tr.size, np.nan)
        return dict(
            traces=tr.tolist(),
            dist=r2(dist),
            img=img,
            ns=[len(p) for p in prof],
            frame="depth" if self.filter is None else self.filter.frame,
            levels=[] if self.z is None else [r2(fr(lv[tr])) for lv in self.z],
            lo=None
            if self.lo_mm is None
            else r2(fr(np.where(unc, self.lo_mm[tr], nan))),
            hi=None
            if self.hi_mm is None
            else r2(fr(np.where(unc, self.hi_mm[tr], nan))),
            uncertified=unc.astype(int).tolist(),
            top=r2(fr(self.top[tr])),
            single=None if self.single is None else r2(fr(self.single[tr])),
            points=[
                dict(k=k, point=p, depth=round(float(dd), 3))
                for k, t in enumerate(tr.tolist())
                for p, d in self.hits.get(t, [])
                for dd in self.depths(t, np.array([d]))
                if np.isfinite(dd)
            ],
            tied=[int(t in self.partner) for t in tr.tolist()],
        )

    def info(self, i: int) -> dict[str, Any]:
        root = self.v_vox[self.rv[i]]
        return dict(
            trace=int(i),
            map_mm=[round(float(x), 2) for x in self.uv[i]],
            root_itksnap=[int(x) for x in np.round(root) + 1],
            root_full_grid0=[int(x) for x in np.round(root) + self.lo],
            world_mm=[round(float(x), 3) for x in self.world(root)[0]],
            pia_mm=None if self.z is None else round(float(self.z[-1, i]), 3),
            certified=bool(self.cert[i]),
            cls=CLASSES[int(self.cls_all[self.sub[i]])],
            fallback=bool(self.fallback_all[self.sub[i]]),
            tie=self.partner.get(int(i)),
        )

    # ------------------------------------------------------------ 3D surface
    def mesh_payload(self, level: int, faces: bool) -> bytes:
        """The 3D surface pane's mesh at inflated level `level`
        (Surface.mesh_payload), with faces: with the nearest shown trace per
        vertex."""
        return self.surf.mesh_payload(level, self.vtrace if faces else None)

    # ------------------------------------------------------------ volume views
    def volume_bytes(self) -> bytes:
        """echo 2 cropped to the box, as .nii.gz bytes (world affine)."""
        if "echo2" not in self._bytes:
            full = nifti(self.echo_path)
            a, b = self.lo, self.lo + np.array(self.shape)
            box = np.asarray(
                full.dataobj[a[0] : b[0], a[1] : b[1], a[2] : b[2]], np.float32
            )
            img = nib.Nifti1Image(box, self.affine)
            self._bytes["echo2"] = gzip.compress(img.to_bytes(), compresslevel=1)
        return self._bytes["echo2"]

    def white_bytes(self) -> bytes:
        return self.surf.white_bytes()

    def gm_bytes(self, sigma: float = 1.0) -> bytes:
        """The cortex label's boundary over the whole box (display only)."""
        if "gm" not in self._bytes:
            v, f = SV.label_surface(self.ctx, (0, 0, 0), self.shape, sigma)
            self._bytes["gm"] = gifti_bytes(self.world(v), f)
        return self._bytes["gm"]

    def overview_tracts(self, cls: int, every: int = 4) -> bytes:
        """Every `every`-th shown trace of class cls, as .tck (world mm)."""
        t = self.sub[::every]
        t = t[self.cls_all[t] == cls]
        return SV.tck([self.world(self.pos_all[k, : self.ns_all[k]]) for k in t])

    # ------------------------------------------------------------ streamlines
    def streamlines(self, i: int, radius: float = 1.0) -> dict[str, Any]:
        """The traces from and to shown trace i's root (streamlines.select) by kind and
        class, with the pia (world mm) on the shown ones among them."""
        root = int(self.rv[i])
        fr, to = SV.select(
            self.G,
            root,
            self.vmm[root],
            self.tv_all,
            self.pos_all,
            self.ns_all,
            self.vox,
            radius,
        )
        groups: dict[str, list[int]] = {}
        for kind, ts in (("from", fr), ("to", to)):
            for c in range(len(CLASSES)):
                groups[f"{kind}_{c}"] = ts[self.cls_all[ts] == c].tolist()
        ts = np.r_[fr, to]
        ks = self.sub_of[ts]
        sel, ks = ts[ks >= 0], ks[ks >= 0]
        pia: list[list[float]] = []
        if self.z is not None and sel.size:
            p = SV.at_depth(self.pos_all[sel], self.ns_all[sel], self.z[-1, ks], 1.0)
            p = p[np.isfinite(p).all(1)]
            pia = np.round(self.world(p), 3).tolist() if len(p) else []
        return dict(
            trace=int(i),
            root_world=[round(float(x), 3) for x in self.world(self.v_vox[root])[0]],
            radius=radius,
            n_from=int(fr.size),
            n_to=int(to.size),
            n_fallback=int(self.fallback_all[ts].sum()) if ts.size else 0,
            groups={k: len(v) for k, v in groups.items()},
            pia_world=pia,
            tracer=self.tracer_name,
            wm_label=self.wm_label,
            _groups=groups,
        )

    def tracts(self, traces: list[int]) -> bytes:
        return SV.tck([self.world(self.pos_all[k, : self.ns_all[k]]) for k in traces])

    # ------------------------------------------------------------ meta
    def meta(self) -> dict[str, Any]:
        return dict(
            site=self.name,
            n=self.n,
            uv=[
                [round(float(a), 3), round(float(b), 3)]
                if np.isfinite(a) and np.isfinite(b)
                else [None, None]
                for a, b in self.uv
            ],
            rv=self.rv.tolist(),
            levels=self.infl_names,
            layers=list(self.layers()),
            classes=list(CLASSES),
            points=len(self.P),
            points_world=np.round(self.world(self.P), 3).tolist()
            if len(self.P)
            else [],
            vmax=self.vmax,
            has_levels=self.z is not None,
            filter=self.filter_meta(),
            tracer=self.tracer_name,
            wm_label=self.wm_label,
        )
