"""The site answers of the WM review, as pure functions of the labels.

Each choice reads the labels of a box round a review site. It returns a
Change: the voxels that it changes and their new labels. It does not change
its input. "WM" means the WM mask: the voxels with a label in
FILLED_LABELS[hemi]. All lengths are in voxels of the grid.

A choice acts only on voxels in the ball of Euclidean radius `reach` round the
site. `support(radius, reach)` gives the half-width of the cube of labels that
a choice reads. The result on a box that holds this cube is the same as the
result on the whole volume.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt
import scipy.ndimage as ndi

from baillarger.handles import kuhn_structure
from baillarger.labels import CORTEX_LABEL, FILLED_LABELS, OUTSIDE, WM_LABEL

Int = npt.NDArray[np.int64]
Bool = npt.NDArray[np.bool_]

#: the six face-neighbour offsets
Vox = tuple[int, ...]

_FACE = np.array(
    [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)],
    dtype=np.int64,
)


@dataclass(frozen=True)
class Change:
    """The voxels that a choice changes.

    voxels: (N, 3) int64, in the coordinates of the box. The rows are sorted
        by C-order linear index. A voxel appears at most once.
    labels: (N,) int64, the new label of each voxel. It is never the current
        label.
    params: the parameters of the choice (radius, reach), to store with the
        answer.
    choice: the name of the choice (a key of CHOICES), set by run.
    """

    voxels: Int
    labels: Int
    params: dict[str, int] = field(default_factory=dict)
    choice: str = ""


def support(radius: int, reach: int) -> int:
    """The half-width of the cube round the site that a choice reads.

    The opening and the closing read 2 * radius beyond the ball. The face
    neighbours read one voxel more. The distance test of `thicken` reads
    radius beyond the ball, which is less.
    """
    _check(radius, reach)
    return reach + 2 * radius + 1


def box_round(
    shape: tuple[int, ...], site: tuple[int, ...] | Int, half: int
) -> tuple[slice, ...]:
    """The slices of the cube of half-width `half` round `site`, cut by the
    edges of the volume."""
    return tuple(
        slice(max(0, int(s) - half), min(n, int(s) + half + 1))
        for s, n in zip(site, shape, strict=True)
    )


def apply(labels: Int, change: Change) -> Int:
    """A copy of `labels` with the change written in."""
    out = np.array(labels, copy=True)
    if len(change.voxels):
        out[tuple(change.voxels.T)] = change.labels
    return out


def leave(
    labels: Int,
    site: tuple[int, int, int],
    hemi: str,
    *,
    radius: int = 1,
    reach: int = 4,
) -> Change:
    """Change nothing."""
    _check(radius, reach)
    return _change(labels, np.zeros(labels.shape, bool), {}, radius, reach)


def remove_strand(
    labels: Int,
    site: tuple[int, int, int],
    hemi: str,
    *,
    radius: int = 1,
    reach: int = 4,
) -> Change:
    """Remove the thin WM piece at the site.

    thin = WM and not opening(WM, ball(radius)). The candidates are the thin
    voxels in the reach ball. Keep the Kuhn piece of the candidates that holds
    the candidate nearest the site. Each removed voxel gets the commonest label
    among its face neighbours that are not WM and not removed (ties: lowest
    label). A voxel with no such neighbour gets its label from the voxels
    already labelled, by the same rule, round after round. A voxel that no
    round reaches gets OUTSIDE.
    """
    _check(radius, reach)
    wm = _wm(labels, hemi)
    thin = wm & ~ndi.binary_opening(wm, _ball(radius))
    piece = _piece(thin & _reach_ball(labels.shape, site, reach), site)
    if piece is None:
        return _empty(labels, radius, reach)
    new = _outer_labels(labels, wm, piece)
    return _change(labels, piece, new, radius, reach)


def fill_gap(
    labels: Int,
    site: tuple[int, int, int],
    hemi: str,
    *,
    radius: int = 1,
    reach: int = 4,
) -> Change:
    """Fill the gap between WM walls at the site.

    gap = closing(WM, ball(radius)) and not WM. The candidates are the gap
    voxels in the reach ball. Keep the Kuhn piece of the candidates that holds
    the candidate nearest the site. Its voxels get the WM label.
    """
    _check(radius, reach)
    piece = _gap_piece(labels, site, hemi, radius, reach)
    if piece is None:
        return _empty(labels, radius, reach)
    new = {tuple(int(c) for c in v): WM_LABEL[hemi] for v in np.argwhere(piece)}
    return _change(labels, piece, new, radius, reach)


def widen_gap(
    labels: Int,
    site: tuple[int, int, int],
    hemi: str,
    *,
    radius: int = 1,
    reach: int = 4,
) -> Change:
    """Open the gap between WM walls at the site.

    The gap piece is the one of `fill_gap`. The changed voxels are the WM
    voxels in the reach ball that share a face with the piece. Each gets the
    commonest non-WM label among its face neighbours (ties: lowest label).
    """
    _check(radius, reach)
    wm = _wm(labels, hemi)
    piece = _gap_piece(labels, site, hemi, radius, reach)
    if piece is None:
        return _empty(labels, radius, reach)
    wall = (
        ndi.binary_dilation(piece, ndi.generate_binary_structure(3, 1))
        & wm
        & _reach_ball(labels.shape, site, reach)
    )
    new = {}
    for v in np.argwhere(wall):
        t = tuple(int(c) for c in v)
        c = _commonest(labels, wm, t, set())
        # a wall voxel has a gap voxel as a face neighbour: c is not None
        new[t] = OUTSIDE if c is None else c
    return _change(labels, wall, new, radius, reach)


def thicken(
    labels: Int,
    site: tuple[int, int, int],
    hemi: str,
    *,
    radius: int = 1,
    reach: int = 4,
) -> Change:
    """Grow the WM into the cortex round the site.

    The cortex voxels in the reach ball with a Euclidean distance of at most
    `radius` to the nearest WM voxel get the WM label.
    """
    _check(radius, reach)
    wm = _wm(labels, hemi)
    if not wm.any():
        return _empty(labels, radius, reach)
    dist2 = ndi.distance_transform_edt(~wm) ** 2
    sel = (
        (labels == CORTEX_LABEL[hemi])
        & (dist2 <= radius * radius + 1e-6)
        & _reach_ball(labels.shape, site, reach)
    )
    new = {tuple(int(c) for c in v): WM_LABEL[hemi] for v in np.argwhere(sel)}
    return _change(labels, sel, new, radius, reach)


def _check(radius: int, reach: int) -> None:
    if radius < 0 or reach < 0:
        raise ValueError("radius and reach must not be negative")


def _wm(labels: Int, hemi: str) -> Bool:
    return np.isin(labels, FILLED_LABELS[hemi])


def _ball(radius: int) -> Bool:
    """The structuring element of the voxels with an offset norm <= radius."""
    g = np.arange(-radius, radius + 1)
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    return np.asarray(x * x + y * y + z * z <= radius * radius)


def _reach_ball(shape: tuple[int, ...], site: tuple[int, int, int], reach: int) -> Bool:
    """The voxels at a Euclidean distance of at most `reach` from the site."""
    out = np.zeros(shape, bool)
    box = box_round(shape, site, reach)
    g = [np.arange(s.start, s.stop) - c for s, c in zip(box, site, strict=True)]
    x, y, z = np.meshgrid(*g, indexing="ij")
    out[box] = x * x + y * y + z * z <= reach * reach
    return out


def _piece(cand: Bool, site: tuple[int, int, int]) -> Bool | None:
    """The Kuhn piece of `cand` with the candidate nearest the site (ties:
    lowest C-order linear index), or None."""
    pts = np.argwhere(cand)
    if len(pts) == 0:
        return None
    d2 = ((pts - np.asarray(site)) ** 2).sum(axis=1)
    near = pts[int(np.argmin(d2))]  # argwhere is in C order: first = lowest
    lab, _ = ndi.label(cand, structure=kuhn_structure())
    return np.asarray(lab == lab[tuple(near)])


def _gap_piece(
    labels: Int, site: tuple[int, int, int], hemi: str, radius: int, reach: int
) -> Bool | None:
    wm = _wm(labels, hemi)
    gap = ndi.binary_closing(wm, _ball(radius)) & ~wm
    return _piece(gap & _reach_ball(labels.shape, site, reach), site)


def _commonest(
    labels: Int,
    wm: Bool,
    v: Vox,
    skip: set[Vox],
    given: dict[Vox, int] | None = None,
) -> int | None:
    """The commonest label among the face neighbours of `v` that are in the
    grid, not WM and not in `skip` (ties: lowest label). Neighbours in `given`
    count with their given label."""
    count: Counter[int] = Counter()
    for o in _FACE:
        n = (v[0] + int(o[0]), v[1] + int(o[1]), v[2] + int(o[2]))
        if any(c < 0 or c >= s for c, s in zip(n, labels.shape, strict=True)):
            continue
        if given is not None and n in given:
            count[given[n]] += 1
        elif not wm[n] and n not in skip:
            count[int(labels[n])] += 1
    if not count:
        return None
    top = max(count.values())
    return min(k for k, c in count.items() if c == top)


def _outer_labels(labels: Int, wm: Bool, piece: Bool) -> dict[Vox, int]:
    """New labels for the removed voxels of `piece`, round after round."""
    todo = [tuple(int(c) for c in v) for v in np.argwhere(piece)]
    skip = set(todo)
    given: dict[Vox, int] = {}
    while todo:
        round_: dict[Vox, int] = {}
        for v in todo:
            c = _commonest(labels, wm, v, skip, given)
            if c is not None:
                round_[v] = c
        if not round_:
            round_ = {v: OUTSIDE for v in todo}
        given.update(round_)
        todo = [v for v in todo if v not in given]
    return given


def _empty(labels: Int, radius: int, reach: int) -> Change:
    return _change(labels, np.zeros(labels.shape, bool), {}, radius, reach)


def _change(
    labels: Int,
    mask: Bool,
    new: dict[Vox, int],
    radius: int,
    reach: int,
) -> Change:
    """Build the Change: rows in C order, no row that keeps its label."""
    pts = np.argwhere(mask).astype(np.int64)  # C order
    if len(pts):
        lab = np.array([new[(int(a), int(b), int(c))] for a, b, c in pts], np.int64)
        keep = lab != labels[tuple(pts.T)]
        pts, lab = pts[keep], lab[keep]
    else:
        pts = np.zeros((0, 3), np.int64)
        lab = np.zeros(0, np.int64)
    return Change(pts, lab, {"radius": radius, "reach": reach})


type ChoiceFunction = Callable[..., Change]

#: the choices by their names in the answers (docs/formats/wm-choices.md)
CHOICES: dict[str, ChoiceFunction] = {
    "remove-strand": remove_strand,
    "thicken": thicken,
    "fill-gap": fill_gap,
    "widen-gap": widen_gap,
    "leave": leave,
}


def run(
    name: str,
    labels: Int,
    site: tuple[int, int, int],
    hemi: str,
    *,
    radius: int = 1,
    reach: int = 4,
) -> Change:
    """The choice with this name (a key of CHOICES) at the site, with its
    name in the Change. Raises KeyError for an unknown name."""
    f = CHOICES[name]
    return replace(f(labels, site, hemi, radius=radius, reach=reach), choice=name)


# ------------------------------------------------------------------ preview


@dataclass(frozen=True)
class Preview:
    """What the correction (baillarger.white_dseg.wm_topo) does after an
    answer, on a box round the site (preview).

    answer: the Change of the answer, in the coordinates of the labels.
    correction: the Change of the correction after the answer: the voxels
        (coordinates of the labels) that it changes and their new labels.
    overrides: whether a fix of the correction changes a voxel of the answer.
    box: the slices of the box in the labels.
    cut_by_box: whether a face of the box that is not an edge of the labels
        has WM on it (after the answer). Then the preview is approximate.
    """

    answer: Change
    correction: Change
    overrides: bool
    box: tuple[slice, ...]
    cut_by_box: bool


def preview(
    labels: Int,
    site: tuple[int, int, int],
    hemi: str,
    answer: str | Change,
    *,
    voxel_mm: npt.ArrayLike,
    radius: int = 1,
    reach: int = 4,
    halo_mm: float = 1.0,
) -> Preview:
    """The answer at the site, then the correction on a box round it, with
    the answer's voxels verified (as dseg does with the answer log).

    `answer` is a key of CHOICES, or a Change (for example a manual paint,
    in the coordinates of the labels). The box is the cube of half-width
    support(radius, reach) round the site, grown to hold the Change's
    voxels, plus a halo of `halo_mm` on each side (rounded up to whole
    voxels per axis), cut by the edges of the labels. Outside the box the
    correction sees no WM.
    """
    # the import is here: white_dseg is heavy, and the choices do not need it
    from baillarger import white_dseg
    from baillarger.wm_edits import VOXEL_DTYPE, Log

    lab = np.asarray(labels)
    shape = lab.shape
    if isinstance(answer, str):
        half = support(radius, reach)
        inner = box_round(shape, site, half)
        sub = np.ascontiguousarray(lab[inner])
        off = np.array([s.start for s in inner], np.int64)
        local = tuple(int(c) - int(o) for c, o in zip(site, off, strict=True))
        ch = run(
            answer,
            sub,
            (local[0], local[1], local[2]),
            hemi,
            radius=radius,
            reach=reach,
        )
        ch = replace(ch, voxels=ch.voxels + off)
        lo = off
        hi = np.array([s.stop for s in inner], np.int64)
    else:
        ch = answer
        lo = np.asarray(site, np.int64)
        hi = lo + 1
    if len(ch.voxels):
        lo = np.minimum(lo, ch.voxels.min(0))
        hi = np.maximum(hi, ch.voxels.max(0) + 1)
    vm = np.asarray(voxel_mm, np.float64)
    halo = np.ceil(halo_mm / vm - 1e-9).astype(np.int64)
    lo = np.maximum(lo - halo, 0)
    hi = np.minimum(hi + halo, np.asarray(shape, np.int64))
    box = tuple(slice(int(p), int(q)) for p, q in zip(lo, hi, strict=True))
    a = apply(lab, ch)[box]
    wm = _wm(a, hemi)
    cut = False
    for ax in range(3):
        if box[ax].start > 0 and wm.take(0, axis=ax).any():
            cut = True
        if box[ax].stop < shape[ax] and wm.take(-1, axis=ax).any():
            cut = True
    none = Change(np.zeros((0, 3), np.int64), np.zeros(0, np.int64))
    if not wm.any():
        return Preview(ch, none, False, box, cut)
    # the answer as a log of one site answer on the box's grid
    bshape = a.shape
    data = np.zeros(len(ch.voxels), VOXEL_DTYPE)
    if len(ch.voxels):
        data["index"] = np.ravel_multi_index(tuple((ch.voxels - lo).T), bshape)
    data["label"], data["stroke"] = ch.labels, -1
    rec = dict(
        Record="answer",
        Kind="site",
        Id=1,
        Voxels=dict(Offset=0, Count=len(data)),
    )
    log = Log(dict(Shape=list(bshape)), (rec,), data)
    before = np.ascontiguousarray(lab[box])
    t = white_dseg.wm_topo(before, vm, log)
    rows = t.diff
    kinds = {f.id: f.kind for f in t.fix_list}
    keep = np.array([kinds[int(f)] != "manual_edit" for f in rows["fix"]], bool)
    rows = rows[keep]
    order = np.argsort(rows["index"], kind="stable")
    rows = rows[order]
    vox = np.stack(np.unravel_index(rows["index"].astype(np.int64), bshape), 1)
    corr = Change(
        (vox + lo).astype(np.int64).reshape(-1, 3),
        rows["new"].astype(np.int64),
    )
    over = any(f.overrides is not None for f in t.fix_list)
    return Preview(ch, corr, over, box, cut)
