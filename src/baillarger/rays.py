"""The ray sampler: the columns of a superblock (spec:
docs/formats/ray-sampler.md, ray-sampler/1).

A sampler computes its output on demand from published files only
(white-surf/1, wm-dseg/1); nothing per column is stored. A column is the
straight ray of one white-surface vertex (its root) along the root's unit
normal, from s = -e_in to s = e_out (mm; s = 0 is the root), with the label
intervals that the ray crosses over the full caps, -CAP_IN to CAP_OUT. A
column is the same, bit for bit, whichever superblock computes it.

The intervals come from an exact traversal of the voxels the ray crosses
(Amanatides and Woo, 1987), in float64 on the CPU, one column at a time:
every value of a column is computed from that column's root alone, in the
same operations whatever else the call holds, so a column does not depend
on the superblock that asks for it.

The method equivolume adds the column's cross-section A(s): the root's dual
area vector on its 1-ring, with each vertex moved along its own normal,
taken on the root's normal. Each vertex moves linearly in s, so A(s) is a
quadratic in s, a0 + a1 s + a2 s^2, exactly. The sampler computes the three
coefficients per column in float64 (the 1-ring's faces in the order of the
file), and from them A, the volume V (the exact integral of A from the
inward cap, to the first focal point on each side of the root) and the
mark past a focal point.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from nibabel.gifti.gifti import GiftiImage

from baillarger import labels as L
from baillarger._jit import njit, prange
from baillarger.bids import file_sha256, read_sidecar, sidecar_path
from baillarger.blocks import CACHE_BYTES, CHUNK, ChunkedSource, morton, open_volume
from baillarger.white_surf import BLOCK_COLUMNS, SURF_SPEC

VERSION = "ray-sampler/1"

#: the caps, mm: the largest inward and outward extents
CAP_IN = 2.0
CAP_OUT = 8.0
#: the default step between samples, mm
STEP = 0.04
#: the methods of this version
METHODS = ("normal", "equivolume")
#: the margin on the halo box for the faces that equivolume reads, voxels
FACE_MARGIN = 2.0
#: the error of each input of A(s) (positions and normals, float32), as a
#: fraction of its value: one unit in the last place of float32
U_IN = 2.0**-23


#: the classes of a label interval
OUTSIDE = 0
WM = 1
CORTEX = 2
OTHER = 3
OFF_GRID = 4
#: the label of an interval off the grid
OFF_GRID_LABEL = -1

#: the derived classes of a column (diagnostics only)
OPEN = 1
IN_WM = 2
NEITHER = 3

Bool = npt.NDArray[np.bool_]
Float = npt.NDArray[np.float64]
I32 = npt.NDArray[np.int32]
I64 = npt.NDArray[np.int64]
F32 = npt.NDArray[np.float32]
U8 = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class Roots:
    """The columns of a call, without their rays: the owned columns, then
    the halo columns, each in vertex order. Each edge key occurs once."""

    #: (n,) the root's grid-edge key, the key of the column
    edge_key: I64
    #: (n,) the root's row in the vertex array of white-surf/1
    vertex: I64
    #: (n,) true for an owned column, false for a halo column
    owned: Bool
    #: (n, 3) the root's position, scanner mm
    root: F32
    #: (n, 3) the root's unit normal, scanner space, out of the WM
    normal: F32
    #: (n,) |n̄|, the smoothed normal's length before its last division
    normal_length: F32


@dataclass(frozen=True)
class Rays:
    """The columns of a call with their samples and label intervals, as
    flat arrays: column c has the samples sample_start[c] to
    sample_start[c + 1] (k from -k_in[c] to k_out[c]) and the intervals
    interval_start[c] to interval_start[c + 1] (in order of s, tiling
    [-CAP_IN, CAP_OUT])."""

    roots: Roots
    method: str
    #: mm
    step: float
    #: (n,) K_in and K_out of each column
    k_in: I32
    k_out: I32
    #: (n + 1,) offsets into the samples
    sample_start: I64
    #: (N,) s of each sample, mm
    s: F32
    #: (N, 3) the position of each sample, scanner mm
    position: F32
    #: (n + 1,) offsets into the intervals
    interval_start: I64
    #: (Q,) the start and the end s of each interval, mm
    interval_s0: F32
    interval_s1: F32
    #: (Q,) the label of each interval, OFF_GRID_LABEL off the grid
    interval_label: I32
    #: (Q,) the class of each interval (OUTSIDE, WM, CORTEX, OTHER, OFF_GRID)
    interval_class: U8
    #: (n,) the derived class of each column (OPEN, IN_WM, NEITHER)
    derived_class: U8
    #: equivolume: (n, 3) a0, a1, a2 of A(s) = a0 + a1 s + a2 s^2 (mm^2,
    #: mm, 1); (0, 3) for normal
    area_coef: Float
    #: equivolume: (n, 2) r_lo and r_hi, the first focal point inward and
    #: outward, mm (-inf and inf if none); (0, 2) for normal
    focus: Float
    #: equivolume: (N,) A(s) at each sample, mm^2; (0,) for normal
    area: F32
    #: equivolume: (N,) V(s) at each sample, mm^3; (0,) for normal
    volume: F32
    #: equivolume: (N,) true at a sample at or past a focal point; (0,) for
    #: normal
    past_focus: Bool


class RaySampler:
    """The ray sampler on one white surface and its segmentation: surf, the
    _white.surf.gii of white-surf/1; dseg, the _desc-wmtopo_dseg of
    wm-dseg/1 that the surface was made from. The surface's arrays stay
    memory-mapped; a call reads only the blocks and the labels it needs, the
    labels through a chunk cache of cache_bytes, so that calls on superblocks
    in Morton order read each chunk about once."""

    def __init__(self, surf: Path, dseg: Path, cache_bytes: int = CACHE_BYTES) -> None:
        self.surf = Path(surf)
        self.dseg = Path(dseg)
        side = read_sidecar(sidecar_path(self.surf), SURF_SPEC)
        self.sidecar = side
        if file_sha256(self.dseg) not in side["SourcesSHA256"].values():
            raise ValueError(f"{self.dseg.name} is not the surface's segmentation")
        self.affine = np.asarray(side["VoxelToRAS"], np.float64).reshape(4, 4)
        self.shape = tuple(int(n) for n in side["GridShape"])
        self.cells = int(side["BlockCells"])
        vol = open_volume(self.dseg)
        if tuple(vol.shape) != self.shape:
            raise ValueError(f"{self.dseg.name}: grid {vol.shape}, not {self.shape}")
        if not np.array_equal(
            vol.affine.astype(np.float32), self.affine.astype(np.float32)
        ):
            raise ValueError(f"{self.dseg.name}: the affine is not VoxelToRAS")
        self.labels = ChunkedSource(vol, CHUNK, cache_bytes)
        self.inverse = np.linalg.inv(self.affine)
        self.voxel_mm = np.linalg.norm(self.affine[:3, :3], axis=0)
        filled = np.asarray(side["FilledLabels"], np.int64)
        hemi = side["Hemisphere"]
        self.filled = filled
        self.cortex = L.CORTEX_LABEL[hemi]
        g = GiftiImage.from_filename(str(self.surf), mmap=True)
        self.x: npt.NDArray[np.float32] = g.darrays[0].data
        self.f: npt.NDArray[np.int32] = g.darrays[1].data
        ng = GiftiImage.from_filename(
            str(self.surf.with_name(side["NormalsFile"])), mmap=True
        )
        self.n: npt.NDArray[np.float32] = ng.darrays[0].data
        kg = GiftiImage.from_filename(
            str(self.surf.with_name(side["EdgeKeysFile"])), mmap=True
        )
        self.keys: npt.NDArray[np.int32] = kg.darrays[0].data  # (n, 2) words
        lg = GiftiImage.from_filename(
            str(self.surf.with_name(side["NormLenFile"])), mmap=True
        )
        # 1-D; files from before the 1-D writer hold n x 1, which reads the same
        self.normlen: npt.NDArray[np.float32] = lg.darrays[0].data.reshape(-1)
        idx = np.loadtxt(
            self.surf.with_name(side["BlocksFile"]), skiprows=1, ndmin=2
        ).reshape(-1, len(BLOCK_COLUMNS))
        self.block_ijk = idx[:, :3].astype(np.int64)
        self.block_code = morton(self.block_ijk)
        self.block_vertices = idx[:, 3:5].astype(np.int64)
        self.block_faces = idx[:, 5:7].astype(np.int64)
        self.block_box = idx[:, 7:13]

    def _halo_box(
        self, superblock: tuple[int, int, int], level: int, halo_mm: float
    ) -> tuple[Float, Float]:
        """The halo box in voxel coordinates: (lo, hi), each (3,)."""
        side = self.cells << level
        sb = np.asarray(superblock, np.int64)
        h = halo_mm / self.voxel_mm
        return sb * side - h, (sb + 1) * side + h

    def roots(
        self, superblock: tuple[int, int, int], level: int, halo_mm: float = 0.0
    ) -> Roots:
        """The columns of superblock (I, J, K) at level m: the roots its
        blocks own, then the roots in its halo box (halo_mm), without rays."""
        _check_call(superblock, level, halo_mm)
        start = int(morton(np.asarray(superblock, np.int64))) << (3 * level)
        stop = start + (1 << (3 * level))
        mine = (self.block_code >= start) & (self.block_code < stop)
        owned = _ranges(self.block_vertices[mine])
        halo = np.zeros(0, np.int64)
        if halo_mm > 0:
            lo, hi = self._halo_box(superblock, level, halo_mm)
            b = self.block_box
            meet = ~mine & np.all((b[:, :3] <= hi) & (b[:, 3:] >= lo), axis=1)
            cand = _ranges(self.block_vertices[meet])
            if len(cand):
                v = _to_voxel(np.asarray(self.x[cand]), self.inverse)
                halo = cand[np.all((v >= lo) & (v <= hi), axis=1)]
        vertex = np.concatenate([owned, halo])
        words = np.ascontiguousarray(self.keys[vertex], "<i4")
        nl = np.asarray(self.normlen[vertex], np.float32)
        return Roots(
            edge_key=words.view("<i8").ravel().astype(np.int64),
            vertex=vertex,
            owned=np.r_[np.ones(len(owned), bool), np.zeros(len(halo), bool)],
            root=np.asarray(self.x[vertex], np.float32),
            normal=np.asarray(self.n[vertex], np.float32),
            normal_length=nl,
        )

    def sample(
        self,
        superblock: tuple[int, int, int],
        level: int,
        *,
        method: str = "normal",
        e_in: float | npt.ArrayLike = CAP_IN,
        e_out: float | npt.ArrayLike = CAP_OUT,
        step: float = STEP,
        halo_mm: float = 0.0,
    ) -> Rays:
        """The columns of roots(superblock, level, halo_mm), in that order,
        with their samples from -e_in to e_out at step (one value, or one
        per column) and their label intervals over the full caps."""
        if method not in METHODS:
            raise ValueError(f"method {method!r}: not one of {METHODS}")
        if not step > 0:
            raise ValueError(f"step {step}: not > 0")
        roots = self.roots(superblock, level, halo_mm)
        n = len(roots.vertex)
        ein = _extent(e_in, n, CAP_IN, "e_in")
        eout = _extent(e_out, n, CAP_OUT, "e_out")
        # the labels every ray of the call can reach: the halo box widened
        # by the outward cap and one voxel, on each axis
        lo, hi = self._halo_box(superblock, level, halo_mm)
        reach = CAP_OUT / self.voxel_mm + 1
        a = np.maximum(np.floor(lo - reach), 0).astype(np.int64)
        b = np.minimum(np.ceil(hi + reach) + 1, self.shape).astype(np.int64)
        box = tuple(slice(int(p), int(q)) for p, q in zip(a, b, strict=True))
        lab = np.ascontiguousarray(self.labels.read(box), np.int32)
        v0, u = _voxel_rays(
            roots.root.astype(np.float64),
            roots.normal.astype(np.float64),
            self.inverse,
        )
        istart, s0, s1, ilab, bad = _traverse(
            v0, u, lab, a, np.asarray(self.shape, np.int64), -CAP_IN, CAP_OUT
        )
        if bad >= 0:
            raise RuntimeError(
                f"column {bad}: its ray leaves the labels read ({box}); "
                "the affine is not a rotation with voxel sizes"
            )
        cls = label_classes(ilab, self.filled, self.cortex)
        derived = _derived(istart, s1, cls)
        k_in = np.floor(ein / step + 1e-6).astype(np.int32)
        k_out = np.floor(eout / step + 1e-6).astype(np.int32)
        sstart, s, pos = _samples(
            roots.root.astype(np.float64),
            roots.normal.astype(np.float64),
            k_in,
            k_out,
            float(step),
        )
        if method == "equivolume":
            coef, err = self.area_coef(roots, superblock, level, halo_mm)
            area, volume, past, focus = _cross_section(
                coef, err, k_in, k_out, float(step), CAP_IN
            )
        else:
            coef = np.zeros((0, 3), np.float64)
            focus = np.zeros((0, 2), np.float64)
            area = np.zeros(0, np.float32)
            volume = np.zeros(0, np.float32)
            past = np.zeros(0, np.bool_)
        return Rays(
            roots=roots,
            method=method,
            step=float(step),
            k_in=k_in,
            k_out=k_out,
            sample_start=sstart,
            s=s,
            position=pos,
            interval_start=istart,
            interval_s0=s0.astype(np.float32),
            interval_s1=s1.astype(np.float32),
            interval_label=ilab,
            interval_class=cls,
            derived_class=derived,
            area_coef=coef,
            focus=focus,
            area=area,
            volume=volume,
            past_focus=past,
        )

    def faces(
        self, superblock: tuple[int, int, int], level: int, halo_mm: float
    ) -> npt.NDArray[np.int64]:
        """The faces (k, 3) of every block whose box meets the halo box
        widened by FACE_MARGIN voxels, in the order of the file. They hold
        the 1-ring of every root of roots(superblock, level, halo_mm): a
        block's box holds every vertex its faces use, and so the root of
        each of them."""
        lo, hi = self._halo_box(superblock, level, halo_mm)
        lo, hi = lo - FACE_MARGIN, hi + FACE_MARGIN
        b = self.block_box
        meet = np.all((b[:, :3] <= hi) & (b[:, 3:] >= lo), axis=1)
        rows = _ranges(self.block_faces[meet])
        return np.asarray(self.f[rows], np.int64).reshape(-1, 3)

    def area_coef(
        self,
        roots: Roots,
        superblock: tuple[int, int, int],
        level: int,
        halo_mm: float,
    ) -> tuple[Float, Float]:
        """(n, 3) each: a0, a1, a2 of each column's A(s) (spec, "The
        cross-section"), from the faces of the call, and the bound on their
        error (spec, "The focal points"). Raises RuntimeError if a root's
        1-ring is not a closed fan (a mesh boundary, or a face not read)."""
        f = self.faces(superblock, level, halo_mm)
        used, local = np.unique(f, return_inverse=True)
        local = local.reshape(-1, 3).astype(np.int64)
        flat = local.ravel()
        order = np.argsort(flat, kind="stable")
        corner = flat[order]
        face_of = (order // 3).astype(np.int64)
        root = np.searchsorted(used, roots.vertex)
        ok = root < len(used)
        ok[ok] = used[root[ok]] == roots.vertex[ok]
        if not np.all(ok):
            c = int(np.flatnonzero(~ok)[0])
            raise RuntimeError(f"column {c}: no face of its root was read")
        start = np.searchsorted(corner, root, "left").astype(np.int64)
        stop = np.searchsorted(corner, root, "right").astype(np.int64)
        coef, err, bad = _area_coef(
            np.asarray(self.x[used], np.float64),
            np.asarray(self.n[used], np.float64),
            local,
            face_of,
            start,
            stop,
            root.astype(np.int64),
            U_IN,
        )
        if bad >= 0:
            raise RuntimeError(
                f"column {bad}: the 1-ring of its root is not a closed fan"
            )
        return coef, err


def label_classes(
    label: npt.ArrayLike, filled: npt.ArrayLike, cortex: int
) -> npt.NDArray[np.uint8]:
    """The class of each label (docs/formats/ray-sampler.md, "The label
    classes"): WM for the filled labels, CORTEX for cortex, OUTSIDE for the
    outside label (labels.OUTSIDE; the dseg has the ventricles there),
    OFF_GRID for OFF_GRID_LABEL, OTHER for every other label."""
    lab = np.asarray(label)
    out = np.full(lab.shape, OTHER, np.uint8)
    out[lab == L.OUTSIDE] = OUTSIDE
    out[lab == cortex] = CORTEX
    out[np.isin(lab, np.asarray(filled))] = WM
    out[lab == OFF_GRID_LABEL] = OFF_GRID
    return out


def _check_call(superblock: tuple[int, int, int], level: int, halo_mm: float) -> None:
    if len(superblock) != 3 or min(superblock) < 0:
        raise ValueError(f"superblock {superblock}: not three integers >= 0")
    if level < 0:
        raise ValueError(f"level {level}: not >= 0")
    if not halo_mm >= 0:
        raise ValueError(f"halo_mm {halo_mm}: not >= 0")


def _extent(e: float | npt.ArrayLike, n: int, cap: float, name: str) -> Float:
    out = np.broadcast_to(np.asarray(e, np.float64), (n,)).copy()
    if np.any(~(out >= 0) | (out > cap)):
        raise ValueError(f"{name}: a value outside [0, {cap}]")
    return out


def _ranges(r: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """The integers of the ranges [start, stop) in r (k, 2), in order."""
    if not len(r):
        return np.zeros(0, np.int64)
    r = r[np.argsort(r[:, 0], kind="stable")]
    return np.concatenate([np.arange(a, b, dtype=np.int64) for a, b in r])


def _to_voxel(x: npt.ArrayLike, inv: Float) -> Float:
    """Scanner mm (k, 3) to voxel coordinates, in float64, element by element
    (no matrix product, whose rounding can depend on the batch)."""
    p = np.asarray(x, np.float64)
    return np.stack(
        [
            inv[a, 0] * p[:, 0] + inv[a, 1] * p[:, 1] + inv[a, 2] * p[:, 2] + inv[a, 3]
            for a in range(3)
        ],
        axis=1,
    )


def _voxel_rays(x: Float, n: Float, inv: Float) -> tuple[Float, Float]:
    """Each ray in voxel coordinates: v0 = A^-1 x0, and u = M^-1 n per mm."""
    lin = inv.copy()
    lin[:3, 3] = 0
    return _to_voxel(x, inv), _to_voxel(n, lin)


@njit(cache=True)
def _voxel_after(p: float, u: float) -> int:
    """The voxel index on one axis just after a point at coordinate p, for a
    ray moving at u: the cube c - 0.5 <= p < c + 0.5, and on a face the
    voxel the ray moves into."""
    q = p + 0.5
    c = np.floor(q)
    if u < 0 and c == q:
        return int(c) - 1
    return int(c)


@njit(cache=True)
def _next_crossing(v0: float, u: float, c: int) -> float:
    """s where the ray leaves voxel c on one axis (inf if u = 0)."""
    if u > 0:
        return (c + 0.5 - v0) / u
    if u < 0:
        return (c - 0.5 - v0) / u
    return np.inf


@njit(cache=True)
def _walk(
    v0: Float,
    u: Float,
    lab: npt.NDArray[np.int32],
    origin: npt.NDArray[np.int64],
    shape: npt.NDArray[np.int64],
    smin: float,
    smax: float,
    out_s0: Float,
    out_s1: Float,
    out_lab: npt.NDArray[np.int32],
    write: bool,
) -> int:
    """The intervals of one ray from smin to smax: their count, and with
    write their start, end and label. -1 if the ray leaves the labels
    read while on the grid."""
    c = np.empty(3, np.int64)
    for a in range(3):
        c[a] = _voxel_after(v0[a] + smin * u[a], u[a])
    s = smin
    count = 0
    cur = -2
    start = smin
    while True:
        on = True
        for a in range(3):
            if c[a] < 0 or c[a] >= shape[a]:
                on = False
        if on:
            i, j, k = c[0] - origin[0], c[1] - origin[1], c[2] - origin[2]
            if (
                i < 0
                or j < 0
                or k < 0
                or i >= lab.shape[0]
                or j >= lab.shape[1]
                or k >= lab.shape[2]
            ):
                return -1
            here = int(lab[i, j, k])
        else:
            here = -1
        if here != cur:
            if cur != -2:
                if write:
                    out_s0[count] = start
                    out_s1[count] = s
                    out_lab[count] = cur
                count += 1
            cur = here
            start = s
        t = np.inf
        for a in range(3):
            ta = _next_crossing(v0[a], u[a], c[a])
            if ta < t:
                t = ta
        if t >= smax:
            break
        for a in range(3):
            if _next_crossing(v0[a], u[a], c[a]) == t:
                c[a] += 1 if u[a] > 0 else -1
        s = t
    if write:
        out_s0[count] = start
        out_s1[count] = smax
        out_lab[count] = cur
    return count + 1


@njit(cache=True, parallel=True)
def _traverse(
    v0: Float,
    u: Float,
    lab: npt.NDArray[np.int32],
    origin: npt.NDArray[np.int64],
    shape: npt.NDArray[np.int64],
    smin: float,
    smax: float,
) -> tuple[npt.NDArray[np.int64], Float, Float, npt.NDArray[np.int32], int]:
    """The label intervals of every ray (docs/formats/ray-sampler.md, "The
    rule for an interval boundary"), as flat arrays with offsets; the last
    value is a column whose ray left the labels read, or -1."""
    n = len(v0)
    count = np.zeros(n, np.int64)
    d0 = np.zeros(1, np.float64)
    d1 = np.zeros(1, np.int32)
    for c in prange(n):
        count[c] = _walk(v0[c], u[c], lab, origin, shape, smin, smax, d0, d0, d1, False)
    for c in range(n):
        if count[c] < 0:
            return np.zeros(n + 1, np.int64), d0[:0], d0[:0], d1[:0], c
    start = np.zeros(n + 1, np.int64)
    start[1:] = np.cumsum(count)
    q = start[n]
    s0 = np.empty(q, np.float64)
    s1 = np.empty(q, np.float64)
    out = np.empty(q, np.int32)
    for c in prange(n):
        a, b = start[c], start[c + 1]
        _walk(
            v0[c],
            u[c],
            lab,
            origin,
            shape,
            smin,
            smax,
            s0[a:b],
            s1[a:b],
            out[a:b],
            True,
        )
    return start, s0, s1, out, -1


@njit(cache=True)
def _derived(
    start: npt.NDArray[np.int64], s1: Float, cls: npt.NDArray[np.uint8]
) -> npt.NDArray[np.uint8]:
    """The derived class of each column from its intervals' outward part."""
    n = len(start) - 1
    out = np.full(n, NEITHER, np.uint8)
    for c in range(n):
        cortex = False
        for q in range(start[c], start[c + 1]):
            if s1[q] <= 0:
                continue
            if cls[q] == OUTSIDE:
                out[c] = OPEN
                break
            if cls[q] == CORTEX:
                cortex = True
            elif cls[q] == WM and cortex:
                out[c] = IN_WM
                break
    return out


@njit(cache=True, parallel=True)
def _samples(
    x: Float,
    n: Float,
    k_in: npt.NDArray[np.int32],
    k_out: npt.NDArray[np.int32],
    step: float,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """The samples of every ray: s_k = k step, and x0 + s_k n, in float64,
    rounded to float32."""
    m = len(x)
    start = np.zeros(m + 1, np.int64)
    for c in range(m):
        start[c + 1] = start[c] + k_in[c] + k_out[c] + 1
    s = np.empty(start[m], np.float32)
    p = np.empty((start[m], 3), np.float32)
    for c in prange(m):
        for j in range(k_in[c] + k_out[c] + 1):
            sk = (j - k_in[c]) * step
            q = start[c] + j
            s[q] = sk
            for a in range(3):
                p[q, a] = x[c, a] + sk * n[c, a]
    return start, s, p


@njit(cache=True)
def _abs_cross(a: Float, b: Float) -> Float:
    """|a| x |b| without the minus signs: (a_j b_k + a_k b_j) on each axis i,
    with (i, j, k) cyclic; a bound on the change of a cross product."""
    return np.array(
        [
            a[1] * b[2] + a[2] * b[1],
            a[2] * b[0] + a[0] * b[2],
            a[0] * b[1] + a[1] * b[0],
        ]
    )


@njit(cache=True)
def _cross_err(p: Float, ep: Float, r: Float, er: Float) -> Float:
    """A bound, per axis, on |(p + dp) x (r + dr) - p x r| for |dp| <= ep and
    |dr| <= er on each axis."""
    return _abs_cross(ep, np.abs(r)) + _abs_cross(np.abs(p), er) + _abs_cross(ep, er)


@njit(cache=True, parallel=True)
def _area_coef(
    x: Float,
    n: Float,
    f: I64,
    face_of: I64,
    start: I64,
    stop: I64,
    root: I64,
    u_in: float,
) -> tuple[Float, Float, int]:
    """Per column, a0, a1, a2 of A(s) = n0 . (1/6) sum over the faces round
    the root of (x_b(s) - x_a(s)) x (x_c(s) - x_a(s)), x_i(s) = x_i + s n_i,
    in float64, the faces in the order of face_of; and a bound on the change
    of each coefficient when each input changes by u_in of its value (spec,
    "The focal points"). The last value is a column whose 1-ring is not a
    closed fan, or -1."""
    m = len(root)
    out = np.zeros((m, 3), np.float64)
    err = np.zeros((m, 3), np.float64)
    bad = np.full(m, False)
    for c in prange(m):
        k = stop[c] - start[c]
        r = root[c]
        nxt = np.empty(k, np.int64)
        prv = np.empty(k, np.int64)
        acc = np.zeros((3, 3), np.float64)
        eacc = np.zeros((3, 3), np.float64)
        for j in range(k):
            t = face_of[start[c] + j]
            a, b, d = f[t, 0], f[t, 1], f[t, 2]
            if a == r:
                nxt[j], prv[j] = b, d
            elif b == r:
                nxt[j], prv[j] = d, a
            else:
                nxt[j], prv[j] = a, b
            p = x[b] - x[a]
            q = n[b] - n[a]
            v = x[d] - x[a]
            w = n[d] - n[a]
            ep = u_in * (np.abs(x[b]) + np.abs(x[a]))
            eq = u_in * (np.abs(n[b]) + np.abs(n[a]))
            ev = u_in * (np.abs(x[d]) + np.abs(x[a]))
            ew = u_in * (np.abs(n[d]) + np.abs(n[a]))
            acc[0] += np.cross(p, v)
            acc[1] += np.cross(p, w) + np.cross(q, v)
            acc[2] += np.cross(q, w)
            eacc[0] += _cross_err(p, ep, v, ev)
            eacc[1] += _cross_err(p, ep, w, ew) + _cross_err(q, eq, v, ev)
            eacc[2] += _cross_err(q, eq, w, ew)
        # a closed fan: each neighbour starts one edge and ends one edge
        sn = np.sort(nxt)
        sp = np.sort(prv)
        closed = k >= 3
        for j in range(k):
            if sn[j] != sp[j] or (j > 0 and sn[j] == sn[j - 1]):
                closed = False
        bad[c] = not closed
        n0 = n[r]
        a0 = np.abs(n0)
        e0 = u_in * a0
        for i in range(3):
            out[c, i] = (n0[0] * acc[i, 0] + n0[1] * acc[i, 1] + n0[2] * acc[i, 2]) / 6
            sa = np.abs(acc[i])
            err[c, i] = (
                (a0 * eacc[i]).sum() + (e0 * sa).sum() + (e0 * eacc[i]).sum()
            ) / 6
    for c in range(m):
        if bad[c]:
            return out, err, c
    return out, err, -1


@njit(cache=True)
def disc_bound(a0: float, a1: float, a2: float, e: Float) -> float:
    """A bound on the change of a1^2 - 4 a0 a2 when each coefficient a_i
    changes by at most e[i]."""
    e0, e1, e2 = float(e[0]), float(e[1]), float(e[2])
    return 2 * abs(a1) * e1 + e1 * e1 + 4 * (abs(a0) * e2 + abs(a2) * e0 + e0 * e2)


@njit(cache=True)
def focal_bounds(a0: float, a1: float, a2: float, bound: float) -> tuple[float, float]:
    """(r_lo, r_hi): the first focal point of A(s) = a0 + a1 s + a2 s^2 on
    each side of s = 0 (-inf or inf if none), with bound the error bound on
    the discriminant (spec, "The focal points"); (0, 0) if A(0) <= 0."""
    if not a0 > 0:
        return 0.0, 0.0
    cand = np.full(2, np.nan)
    disc = a1 * a1 - 4 * a0 * a2
    if a2 == 0:
        if a1 != 0:
            cand[0] = -a0 / a1
    elif disc > bound or a2 < 0:
        # two real roots (with a0 > 0 and a2 < 0, disc > 0)
        sq = np.sqrt(disc)
        q = -0.5 * (a1 + sq) if a1 >= 0 else -0.5 * (a1 - sq)
        cand[0] = q / a2
        cand[1] = a0 / q
    elif disc >= -bound:
        # a double root, within the error of the inputs
        cand[0] = -a1 / (2 * a2)
    lo = -np.inf
    hi = np.inf
    for r in cand:
        if np.isnan(r):
            continue
        if r < 0:
            lo = max(lo, r)
        elif r > 0:
            hi = min(hi, r)
        else:
            return 0.0, 0.0
    return lo, hi


@njit(cache=True)
def _primitive(t: float, a0: float, a1: float, a2: float) -> float:
    """The integral of A from 0 to t."""
    return t * (a0 + t * (a1 / 2 + t * (a2 / 3)))


@njit(cache=True, parallel=True)
def _cross_section(
    coef: Float,
    err: Float,
    k_in: npt.NDArray[np.int32],
    k_out: npt.NDArray[np.int32],
    step: float,
    cap_in: float,
) -> tuple[F32, F32, Bool, Float]:
    """A(s), V(s) and the mark at the samples of every column, at
    s_k = k step in float64 as _samples gives them, and each column's focal
    bounds (spec, "The cross-section")."""
    m = len(coef)
    start = np.zeros(m + 1, np.int64)
    for c in range(m):
        start[c + 1] = start[c] + k_in[c] + k_out[c] + 1
    area = np.empty(start[m], np.float32)
    vol = np.empty(start[m], np.float32)
    past = np.empty(start[m], np.bool_)
    focus = np.empty((m, 2), np.float64)
    for c in prange(m):
        a0, a1, a2 = coef[c, 0], coef[c, 1], coef[c, 2]
        r_lo, r_hi = focal_bounds(a0, a1, a2, disc_bound(a0, a1, a2, err[c]))
        focus[c, 0] = r_lo
        focus[c, 1] = r_hi
        lo = max(-cap_in, r_lo)
        p_lo = _primitive(lo, a0, a1, a2)
        for j in range(k_in[c] + k_out[c] + 1):
            sk = (j - k_in[c]) * step
            q = start[c] + j
            area[q] = a0 + sk * (a1 + sk * a2)
            t = min(max(sk, lo), r_hi)
            vol[q] = max(_primitive(t, a0, a1, a2) - p_lo, 0.0)
            past[q] = sk <= r_lo or sk >= r_hi
    return area, vol, past, focus
