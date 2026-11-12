"""The surface preview of label edits (docs/plans/label-editing.md, Part 4b): a
working antialiased field in the viewer's memory, its local solves
(baillarger.antialias.local_solve, the radius of docs/preview-radius.md) and the
re-march of the mesh blocks that they change (baillarger.white_surf.remarch). The
preview is approximate; the regeneration is exact.

The state (State):
- field: the working field, the stored field (white_surf.read_field: the field that
  white_surf solved for the dseg, memory-mapped and read box by box) with the local
  solve of each change written in (blocks.PatchedSource). Only the patches are in
  memory.
- overlay: the voxels where the preview's labels differ from the dseg's labels, and
  their labels (a label_edits.Edits): the pending edits and their corrections (4a).
  The preview's WM mask is the dseg's labels with the overlay written in, in
  FilledLabels (WorkingMask). So its sign and the working field's sign agree.
- applied: the label edits that the working field follows. At start, the dseg's
  copy of the edits that it used (the stored field is the dseg's).
- blocks: each mesh block (of the surface's BlockCells) that a solve re-marched,
  its mesh, and the revision at which it last changed. A block with no faces has no
  surface now.
- revision: the number of solves written in.

A change of the label edits from `before` to `after` (SurfacePreview.change) takes
the voxels whose entry differs (label_edits.differ) in groups (Groups: voxels in
neighbouring cells of GROUP voxels a side are one group), one after the other. For
each group:
1. region(group, pending voxels, their labels) (the caller's; viewer.review) gives a
   box round the group and the overlay's new entries in the box: the pending edits
   there, and the labels that the automatic correction gives after them (4a).
2. The overlay's entries in the box are replaced by the new ones.
3. local_solve of the working field and the new mask, at the voxels of the box's old
   and new entries: it keeps the voxels whose side of the mask changed. None: the
   mask did not change, so the field and the mesh do not change.
4. remarch of the blocks whose field values the solve changed, from the field before
   the solve; then the solve is written into the field.
5. applied takes the group's entries of `after`.
A group is done in full or not at all, so a stop between groups (stop) leaves a
state that a later change continues from.

The viewer (viewer.review) keeps one State, the working state: at each surface
preview it first takes the working state to the current edits (a commit), then
takes a copy of it (State.copy: the copy's field is a new PatchedSource over the
same patches, so a write into it does not change the working field) to the edits
with the strokes not yet submitted painted in (the preview), and drops the copy.

Cost of one group (docs/preview-radius.md): the box of the changed voxels plus
R + BAND + 5 (R >= 16 voxels) for the solve, read from the field and the mask;
thin_plate on it (a few hundred iterations); the march of the blocks that meet the
ball of R round the changed voxels. None of these depends on the size of the grid.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi

from baillarger import antialias as AA
from baillarger import label_edits
from baillarger.backend import Device
from baillarger.blocks import PatchedSource, Source
from baillarger.white_surf import remarch

type Arr = npt.NDArray[Any]
type Box = tuple[slice, slice, slice]
type BlockId = tuple[int, int, int]

#: changed voxels in neighbouring cells of GROUP voxels a side are one group
#: (two times the smallest radius, PREVIEW_MIN_RADIUS)
GROUP = 2 * AA.PREVIEW_MIN_RADIUS


class Stopped(Exception):
    """A change stopped between two groups (a newer preview replaced it)."""


@dataclass(frozen=True)
class BlockMesh:
    """The mesh of one block after a solve: v (float32, voxel coordinates of the
    grid), f (int32, into v), and the revision at which it changed."""

    v: npt.NDArray[np.float32]
    f: npt.NDArray[np.int32]
    revision: int


@dataclass(frozen=True)
class Solved:
    """One group's local solve: its box, R, c, delta0, the number of voxels whose
    side changed, the solver's iterations and seconds, and the blocks it
    re-marched."""

    box: Box
    radius: int
    c: float
    delta0: float
    changed: int
    iterations: int
    seconds: float
    blocks: list[BlockId]

    def summary(self) -> dict[str, Any]:
        return dict(
            box=[[s.start, s.stop] for s in self.box],
            radius=self.radius,
            c=round(self.c, 3),
            delta0=self.delta0,
            changed=self.changed,
            iterations=self.iterations,
            seconds=round(self.seconds, 3),
            blocks=[list(b) for b in self.blocks],
        )


@dataclass
class State:
    """The working field, the overlay, the applied edits and the re-marched blocks
    (module docstring)."""

    field: PatchedSource
    overlay: label_edits.Edits
    applied: label_edits.Edits
    blocks: dict[BlockId, BlockMesh] = dc_field(default_factory=dict)
    revision: int = 0

    def copy(self) -> "State":
        """A copy whose changes do not change this state (the arrays are shared:
        no write changes an array in place)."""
        return State(
            PatchedSource(self.field),
            self.overlay,
            self.applied,
            dict(self.blocks),
            self.revision,
        )


class WorkingMask:
    """The preview's WM mask on the whole grid, as a blocks.Source: the dseg's
    labels (the whole grid, in memory) with the overlay written in
    (label_edits.apply on the box read), in filled."""

    def __init__(
        self, labels: Arr, overlay: label_edits.Edits, filled: Sequence[int]
    ) -> None:
        self.labels = labels
        self.overlay = overlay
        self.filled = np.asarray(list(filled))

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(n) for n in self.labels.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(bool)

    def read(self, box: tuple[slice, ...]) -> npt.NDArray[np.bool_]:
        crop = label_edits.apply(
            self.labels[box],
            self.overlay,
            offset=[int(s.start) for s in box],
            shape=self.shape,
        )
        out: npt.NDArray[np.bool_] = np.isin(crop, self.filled)
        return out


def groups(voxels: Arr, cell: int = GROUP) -> list[npt.NDArray[np.int64]]:
    """The rows of voxels (n, 3) in groups: voxels in the same or neighbouring
    (26-adjacent) cells of cell voxels a side are one group. The work is
    O(n log n) and the cells of the voxels' box (the grid / cell at most)."""
    v = np.asarray(voxels, np.int64).reshape(-1, 3)
    if not len(v):
        return []
    c = v // cell
    lo = c.min(0)
    occ = np.zeros(tuple(c.max(0) - lo + 1), bool)
    occ[tuple((c - lo).T)] = True
    lab, _ = ndi.label(occ, structure=np.ones((3, 3, 3), bool))
    g = lab[tuple((c - lo).T)]
    order = np.argsort(g, kind="stable")
    cuts = np.flatnonzero(np.diff(g[order])) + 1
    return [o.astype(np.int64) for o in np.split(order, cuts)]


def in_box(voxels: Arr, box: Box) -> npt.NDArray[np.bool_]:
    """Which voxels (n, 3) are inside box."""
    lo = np.array([s.start for s in box])
    hi = np.array([s.stop for s in box])
    return np.all((voxels >= lo) & (voxels < hi), axis=1)


def _entries(index: npt.ArrayLike, labels: npt.ArrayLike) -> label_edits.Edits:
    """Edits of index and labels, a voxel given twice keeping its later label."""
    idx = np.asarray(index, np.int64)
    lab = np.asarray(labels, np.int64)
    u, last = np.unique(idx[::-1], return_index=True)
    return label_edits.Edits(u, lab[::-1][last].astype(np.int32))


#: region(group voxels (k, 3), pending voxels (n, 3), their labels (n,)) -> the
#: box round the group, and the overlay's new entries in it: voxels (m, 3) and
#: labels (m,), each differing from the dseg's label (module docstring)
type Region = Callable[[Arr, Arr, Arr], tuple[Box, Arr, Arr]]
#: the pending edits of label edits: voxels (n, 3) and their labels (n,)
type Pending = Callable[[label_edits.Edits], tuple[Arr, Arr]]


class SurfacePreview:
    """The working state and its changes (module docstring). field: the stored
    field on the whole grid (white_surf.read_field); cells: the surface's
    BlockCells; filled: its FilledLabels; labels: the dseg's labels (the whole
    grid); start: the edits that the stored field follows (the dseg's copy of the
    edits); device: where the solves and the marches run."""

    def __init__(
        self,
        field: Source,
        cells: int,
        filled: Sequence[int],
        labels: Callable[[], Arr],
        start: label_edits.Edits,
        device: Device,
    ) -> None:
        self.cells = int(cells)
        self.filled = [int(x) for x in filled]
        self.labels = labels
        self.device = device
        self.shape = tuple(int(n) for n in field.shape)
        self.state = State(PatchedSource(field), label_edits.empty(), start)
        #: the cell of groups (groups)
        self.group = GROUP

    def mask(self, overlay: label_edits.Edits) -> WorkingMask:
        return WorkingMask(self.labels(), overlay, self.filled)

    def change(
        self,
        st: State,
        after: label_edits.Edits,
        pending: Pending,
        region: Region,
        stop: Callable[[], bool] = lambda: False,
    ) -> list[Solved]:
        """Take st from its applied edits to after, group by group (module
        docstring); st changes in place. stop() is asked before each group: True
        raises Stopped (the groups done stay in st). Returns the solves."""
        idx, lab = label_edits.differ(after, st.applied)
        if not len(idx):
            return []
        vox = np.stack(np.unravel_index(idx, self.shape), 1).astype(np.int64)
        pv, plab = pending(after)
        pv = np.asarray(pv, np.int64).reshape(-1, 3)
        plab = np.asarray(plab, np.int64)
        out: list[Solved] = []
        for g in groups(vox, self.group):
            if stop():
                raise Stopped("replaced by a newer preview")
            s = self._group(st, vox[g], pv, plab, region)
            if s is not None:
                out.append(s)
            st.applied = label_edits.restore(
                st.applied, label_edits.Edits(idx[g], lab[g])
            )
        return out

    def _group(
        self, st: State, group: Arr, pv: Arr, plab: Arr, region: Region
    ) -> Solved | None:
        box, nv, nlab = region(group, pv, plab)
        ov = st.overlay
        old = np.stack(np.unravel_index(ov.index, self.shape), 1).astype(np.int64)
        old = old[in_box(old, box)]
        new = np.asarray(nv, np.int64).reshape(-1, 3)
        removed = label_edits.restore(
            ov,
            label_edits.Edits(
                np.sort(np.ravel_multi_index(tuple(old.T), self.shape)).astype(
                    np.int64
                ),
                np.full(len(old), label_edits.INPUT, np.int32),
            ),
        )
        added = _entries(np.ravel_multi_index(tuple(new.T), self.shape), nlab)
        overlay = label_edits.restore(removed, added)
        cand = np.concatenate([old, new])
        local = AA.local_solve(st.field, self.mask(overlay), cand, self.device)
        st.overlay = overlay
        if local is None:
            return None
        r = remarch(st.field, local, self.cells, self.device)
        st.field.write(local.box, local.phi)
        st.revision += 1
        ids: list[BlockId] = [(int(a), int(b), int(c)) for a, b, c in r.blocks]
        made = {
            tuple(int(x) for x in o // self.cells): k
            for k, o in enumerate(r.mesh.origin)
        }
        for b in ids:
            k = made.get(b)
            if k is None:  # no surface in this block now
                v = np.zeros((0, 3), np.float32)
                f = np.zeros((0, 3), np.int32)
            else:
                ff = r.f[r.mesh.face_start[k] : r.mesh.face_stop[k]]
                u, inv = np.unique(ff, return_inverse=True)
                v = np.ascontiguousarray(r.v[u], np.float32)
                f = inv.reshape(-1, 3).astype(np.int32)
            st.blocks[b] = BlockMesh(v, f, st.revision)
        return Solved(
            local.box,
            local.radius,
            local.c,
            local.delta0,
            len(local.changed),
            local.solve.iterations,
            local.solve.seconds,
            ids,
        )


def shown(
    working: dict[BlockId, BlockMesh], preview: dict[BlockId, BlockMesh]
) -> dict[BlockId, BlockMesh]:
    """The blocks that the page shows in place of the surface's: the working
    blocks, with the preview's blocks over them."""
    return {**working, **preview}


def block_payload(
    blocks: Sequence[tuple[BlockId, BlockMesh, int]],
) -> tuple[list[list[int]], dict[str, Arr]]:
    """The blocks (id, mesh, kind) as rows [i, j, k, nv, nf, kind] and the arrays
    v (float32 (N, 3), voxel coordinates of the grid) and f (int32 (M, 3), each
    block's faces into its own vertices), block after block."""
    rows = [[*b, len(m.v), len(m.f), kind] for b, m, kind in blocks]
    v = [m.v for _, m, _ in blocks]
    f = [m.f for _, m, _ in blocks]
    return rows, dict(
        v=np.concatenate(v) if v else np.zeros((0, 3), np.float32),
        f=np.concatenate(f) if f else np.zeros((0, 3), np.int32),
    )
