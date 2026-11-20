"""The grow brush of the label editing (docs/plans/label-editing.md, Part 6).

The user paints strokes on a few slices. grow turns them into a proposal: the
voxels whose label changes, and their new labels. It does not write the
label edits. The viewer shows the proposal, and an accepted proposal is one
paint (label_edits.paint), so the diff format does not change.

The steps:

1. Seeds: the stroke voxels with their labels, in stroke order (a voxel in
   two strokes keeps the later one).
2. The area: the strokes' region, and the voxels within band voxels of it.
   The region fills the slices between stroked slices (region): a stroke is
   on a slice when all its voxels lie on one slice of exactly one axis (a
   brush disc on one view). Between two stroked slices of one axis that are
   next to each other, the 2D signed distances of their strokes are
   interpolated linearly, and the region is where the result is positive. A
   stroke on no such slice is in the region as it is. The box is the strokes'
   bounding box with a margin of band + zone + 3·sigma + 1 voxels, clipped to
   the grid.
3. Fixed seeds: each voxel of the box outside the area keeps its current
   label.
4. The growth: a random walker (skimage.segmentation.random_walker, BSD) on
   echoes 2 and 3 in the box. Each echo is log(max(e, 1)), scaled by its 2nd
   and 98th percentiles in the area. The classes are the labels of the
   seeds. Each voxel of the area takes its most probable class.
5. The smoothing: the WM mask after the growth, a Gaussian of sigma voxels
   on its signed distance, threshold 0. It changes only voxels of the area
   within zone voxels of a voxel that the growth changed. A voxel that
   leaves the WM takes its most probable non-WM class from the growth; a
   voxel that joins the WM takes the WM label.
6. The proposal: the voxels of the box whose new label is not their current
   label.

The settings are first values from one site (the plan, Part 6). A scan of a
value is a finding, not a setting to tune.
"""

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi

Arr = npt.NDArray[Any]
Box = tuple[tuple[int, int, int], tuple[int, int, int]]


@dataclass(frozen=True)
class Settings:
    """band: voxels round the strokes' region; beta: the random walker's
    beta; sigma: the smoothing's Gaussian (voxels); zone: voxels round the
    growth's change in which the smoothing acts."""

    band: int = 3
    beta: float = 130.0
    sigma: float = 1.0
    zone: int = 4

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


#: the first values (module docstring)
DEFAULT = Settings()


@dataclass(frozen=True)
class Proposal:
    """The voxels whose label changes (flat indices on the grid, sorted,
    unique) and their new labels, the box (lo, hi; hi exclusive) and the
    settings that made them."""

    index: npt.NDArray[np.int64]
    label: npt.NDArray[np.int32]
    box: Box
    settings: Settings = field(default_factory=Settings)

    def voxels(self, shape: Sequence[int]) -> npt.NDArray[np.int64]:
        """The voxels (N, 3) of the proposal."""
        return np.stack(np.unravel_index(self.index, tuple(shape)), 1).astype(np.int64)


def _seeds(
    strokes: Sequence[tuple[int, npt.ArrayLike]],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """The stroke voxels (N, 3) and their labels, a voxel in two strokes
    with the later stroke's label."""
    owner: dict[tuple[int, int, int], int] = {}
    for lab, vox in strokes:
        for v in np.asarray(vox, np.int64).reshape(-1, 3):
            owner[(int(v[0]), int(v[1]), int(v[2]))] = int(lab)
    if not owner:
        raise ValueError("a grow needs at least one stroke voxel")
    v = np.asarray(list(owner), np.int64).reshape(-1, 3)
    return v, np.asarray(list(owner.values()), np.int64)


def _on_slice(v: npt.NDArray[np.int64]) -> int | None:
    """The axis on one slice of which all the voxels lie, if there is exactly
    one such axis; else None."""
    flat = [ax for ax in range(3) if np.all(v[:, ax] == v[0, ax])]
    return flat[0] if len(flat) == 1 else None


def region(
    strokes: Sequence[tuple[int, npt.ArrayLike]], lo: Arr, shape: Sequence[int]
) -> npt.NDArray[np.bool_]:
    """The strokes' region in the box of this shape at lo (module docstring,
    step 2)."""
    out = np.zeros(tuple(shape), bool)
    on: dict[int, dict[int, npt.NDArray[np.bool_]]] = {0: {}, 1: {}, 2: {}}
    for _, vox in strokes:
        v = np.asarray(vox, np.int64).reshape(-1, 3) - lo
        if not len(v):
            continue
        out[tuple(v.T)] = True
        ax = _on_slice(v)
        if ax is None:
            continue
        k = int(v[0, ax])
        if k not in on[ax]:
            on[ax][k] = np.zeros(tuple(n for i, n in enumerate(shape) if i != ax), bool)
        rest = [i for i in range(3) if i != ax]
        on[ax][k][v[:, rest[0]], v[:, rest[1]]] = True
    for ax, sl in on.items():
        ks = sorted(sl)
        d = {k: _sdf(sl[k]) for k in ks}
        for a, b in zip(ks[:-1], ks[1:], strict=True):
            for k in range(a + 1, b):
                w = (k - a) / (b - a)
                idx: list[Any] = [slice(None)] * 3
                idx[ax] = k
                out[tuple(idx)] |= (1 - w) * d[a] + w * d[b] > 0
    return out


def _sdf(m: Arr) -> npt.NDArray[np.float32]:
    """Signed distance in voxels, positive inside."""
    d: npt.NDArray[np.float32] = (
        ndi.distance_transform_edt(m) - ndi.distance_transform_edt(~m)
    ).astype(np.float32)
    return d


def grow(
    strokes: Sequence[tuple[int, npt.ArrayLike]],
    current: Callable[[Box], Arr],
    echoes: Callable[[Box], tuple[Arr, Arr]],
    *,
    shape: Sequence[int],
    wm: int,
    settings: Settings = DEFAULT,
    box_limit: int = 2_000_000,
) -> Proposal:
    """The proposal of the strokes (module docstring). strokes: (label,
    voxels (N, 3)) in stroke order; current(box): the current labels of the
    box (the dseg's labels with the diff's entries); echoes(box): echoes 2
    and 3 of the box; shape: the grid; wm: the WM label. Raises ValueError
    for no stroke voxel, a voxel off the grid, or an area above box_limit
    voxels (nothing is proposed)."""
    from skimage.segmentation import random_walker

    grid = np.asarray(tuple(int(n) for n in shape), np.int64)
    sv, sl = _seeds(strokes)
    if not np.all((sv >= 0) & (sv < grid)):
        raise ValueError("a stroke voxel is outside the grid")
    s = settings
    pad = s.band + s.zone + int(np.ceil(3 * s.sigma)) + 1
    lo = np.maximum(sv.min(0) - pad, 0)
    hi = np.minimum(sv.max(0) + pad + 1, grid)
    if int(np.prod(hi - lo)) > 8 * box_limit:
        raise ValueError(
            f"the grow's box ({int(np.prod(hi - lo))} voxels) is too large: "
            f"paint strokes closer together"
        )
    box: Box = (
        (int(lo[0]), int(lo[1]), int(lo[2])),
        (int(hi[0]), int(hi[1]), int(hi[2])),
    )
    rel = sv - lo
    reg = region(strokes, lo, tuple(hi - lo))
    area = ndi.distance_transform_edt(~reg) <= s.band
    n_area = int(area.sum())
    if n_area > box_limit:
        raise ValueError(
            f"the grow's area ({n_area} voxels) is above the limit {box_limit}: "
            f"paint strokes closer together"
        )

    cur = np.asarray(current(box))
    if cur.shape != area.shape:
        raise ValueError(f"current() gave {cur.shape} for the box {area.shape}")
    seeds = np.where(area, -1, cur.astype(np.int64))
    seeds[tuple(rel.T)] = sl
    classes = np.unique(seeds[seeds >= 0])
    if len(classes) < 2:
        new = np.where(area, classes[0], cur).astype(np.int64)
        prob = None
    else:
        e2, e3 = echoes(box)
        feats = []
        for e in (e2, e3):
            x = np.log(np.maximum(np.asarray(e, np.float64), 1.0))
            a, b = np.percentile(x[area], [2, 98])
            feats.append((x - a) / max(b - a, 1e-6))
        rw_lab = np.zeros(area.shape, np.int32)  # 0: unknown
        known = seeds >= 0
        rw_lab[known] = np.searchsorted(classes, seeds[known]) + 1
        prob = random_walker(
            np.stack(feats, -1),
            rw_lab,
            beta=s.beta,
            mode="cg_j",
            tol=1e-5,
            channel_axis=-1,
            return_full_prob=True,
        )
        new = classes[np.argmax(prob, 0)].astype(np.int64)
        new = np.where(area, new, cur)
        new[tuple(rel.T)] = sl

    # the smoothing of the WM, near the growth's change
    changed = new != cur
    if changed.any() and s.sigma > 0:
        w = new == wm
        smooth = ndi.gaussian_filter(_sdf(w), s.sigma) > 0
        zone = (ndi.distance_transform_edt(~changed) <= s.zone) & area
        zone[tuple(rel.T)] = False  # the strokes keep their labels
        leave = zone & w & ~smooth
        join = zone & ~w & smooth
        if leave.any():
            if prob is not None:
                p = prob.copy()
                if wm in classes:
                    p[int(np.searchsorted(classes, wm))] = -1.0
                other = classes[np.argmax(p, 0)]
            else:
                other = np.full(area.shape, classes[0])
            # with no non-WM class in the growth, the voxel keeps the WM
            ok = other != wm
            new[leave & ok] = other[leave & ok]
        new[join] = wm

    diff = np.nonzero(new != cur)
    gidx = np.ravel_multi_index(tuple(np.asarray(diff) + lo[:, None]), tuple(grid))
    order = np.argsort(gidx)
    return Proposal(
        index=gidx[order].astype(np.int64),
        label=new[diff][order].astype(np.int32),
        box=box,
        settings=s,
    )
