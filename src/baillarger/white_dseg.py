"""The WM topology correction of a hemisphere (dseg): the first of the three
white-surface commands. Run them in this order:

    python -m baillarger.white_dseg ASEG [--edits EDITS] --out DERIVATIVES
    python -m baillarger.white_surf DSEG --out DERIVATIVES [--block-cells N]
        [--device DEVICE]
    python -m baillarger.white_overview SURF [--eps MM] [--workers N]

The WM mask (FILLED_LABELS: the WM and the deep structures it encloses) must
be cavity- and handle-free, so that the white mesh has genus-0 pieces. The
correction is made once, on the whole mask (baillarger.handles,
remove_handles), because a box crop of a handle-free solid has planar surface
pieces, while a box alone cannot tell a handle from a fold that closes outside
it. wm_topo makes the corrected segmentation: the aseg's labels, with the
labels of TO_OUTSIDE (the ventricles, the choroid plexus and CSF, which
are outside in an isolated hemisphere) set to OUTSIDE, the mask's
cavities and handle fills set to the WM label, each handle cut given the
label of its nearest neighbour outside the mask, and each piece of
the mask below CRUMB_MM3 given the labels of its neighbours outside the mask
(small_pieces). Every later stage
reads its labels from the corrected segmentation, never the aseg plus a list
of changes: the whole-hemisphere surface (baillarger.white_surf) and the box
stages (baillarger.column_traces.load_tissue, baillarger.pial_labels, whose
WM is the corrected mask of their box: filled_labels).

The command writes the corrected segmentation (desc-wmtopo_dseg), the
changed voxels (desc-wmtopofixes_dseg), the fix table and the fix diff, with
their .tsv and .json sidecars (docs/formats/wm-dseg.md), and a copy of the
label edits that it used. EDITS is the diff of the label edits (a .npy,
baillarger.label_edits, docs/formats/label-edits.md): it is applied to the
input labels before the automatic correction.
"""

import argparse
import hashlib
import io
import itertools
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse import csgraph

from baillarger import label_edits
from baillarger.bids import (
    COMMON_KEYS,
    Key,
    SidecarSpec,
    dataset_description,
    derivative_path,
    entities,
    file_sha256,
    provenance,
    read_sidecar,
    sidecar_path,
    sources,
    write_sidecar,
)
from baillarger.fix_diff import (
    DIFF_DTYPE,
    Fix,
    apply_diff,
    make_diff,
    table_rows,
    write_table,
)
from baillarger.handles import kuhn_betti, kuhn_structure, remove_handles
from baillarger.label_edits import Edits
from baillarger.label_edits import apply as apply_edits
from baillarger.labels import CRUMB_MM3 as CRUMB_MM3
from baillarger.labels import FILLED_LABELS as FILLED_LABELS
from baillarger.labels import LUT_NAMES as LUT_NAMES
from baillarger.labels import OUTSIDE, TO_OUTSIDE
from baillarger.labels import WM_LABEL as WM_LABEL
from baillarger.timing import Laps, log_to_stderr

type Bool = npt.NDArray[np.bool_]
type Int = npt.NDArray[np.int64]
type Betti = tuple[int, int, int]

# ------------------------------------------------- the corrected segmentation

#: values of the fixes image: a handle cut, a handle fill, a cavity fill, a
#: manual edit (baillarger.label_edits), a voxel of a small piece (small_pieces)
FIXES = {
    1: "handle_cut",
    2: "handle_fill",
    3: "cavity_fill",
    4: "manual_edit",
    5: "small_piece",
}

#: docs/formats/wm-dseg.md
DSEG_SPEC = SidecarSpec(
    "wm-dseg/1",
    COMMON_KEYS
    + (
        Key("Hemisphere", "string", True, "R or L"),
        Key("ToOutside", "object", True, "input label set to outside: count"),
        Key("FilledLabels", "array of integers", True, "labels of the WM mask"),
        Key("WMLabel", "integer", True, "label given to cavity and handle fills"),
        Key("BettiBefore", "array of integers", True, "b0, b1, b2 of the mask"),
        Key("BettiAfter", "array of integers", True, "b0, b1, b2 after"),
        Key("CavityVoxels", "integer", True, "voxels set to WM: cavities"),
        Key("HandleFillVoxels", "integer", True, "voxels set to WM: handle fills"),
        Key("HandleCutVoxels", "integer", True, "voxels cut from the mask"),
        Key("CutLabels", "object", True, "new label of the cut voxels: count"),
        Key("SmallPieces", "integer", True, "pieces of the mask removed: small"),
        Key("SmallPieceVoxels", "integer", True, "voxels of the small pieces"),
        Key("ManualVoxels", "integer", True, "voxels changed by manual edits"),
        Key("Moves", "array of objects", True, "each manual edit and handle move"),
    ),
)
#: docs/formats/wm-dseg.md
FIXES_SPEC = SidecarSpec(
    "wm-dseg-fixes/1",
    COMMON_KEYS
    + (
        Key("FixTable", "string", True, "file name of the fix table"),
        Key("FixDiff", "string", True, "file name of the fix diff"),
        Key("DiffSHA256", "string", True, "SHA-256 of the fix diff file"),
        Key("DiffVoxels", "integer", True, "rows of the fix diff"),
        Key("DiffInputs", "object", True, "SHA-256 of the inputs, and the code"),
        Key("Fixes", "object", True, "number of fixes, by kind"),
    ),
)

#: the directory that keeps every run's fix diff, table and sidecar, by the
#: diff's SHA-256: in the session's directory, next to anat/ (history_dir)
HISTORY = "wmtopofixes-history"


def dseg_path(root: Path, ents: dict[str, str]) -> Path:
    """The path of the dseg command's corrected segmentation
    (desc-wmtopo_dseg.nii.gz) in the BIDS derivatives root, for the subject and
    session of the entities ents."""
    return derivative_path(root, ents, "dseg", ".nii.gz", desc="wmtopo")


def output_paths(dseg: Path) -> dict[str, Path]:
    """The outputs of the dseg command (write_wm_topo) whose corrected
    segmentation is at dseg, by role, in the directory of dseg: dseg, fixes
    (desc-wmtopofixes_dseg), table (the fix table), diff (the fix diff) and
    edits (the copy of the label edits). Raises ValueError if dseg is not
    named as dseg_path names it."""
    d = Path(dseg)
    ents = entities(d)

    def at(suffix: str, ext: str, desc: str) -> Path:
        return d.with_name(derivative_path(d.parent, ents, suffix, ext, desc=desc).name)

    if at("dseg", ".nii.gz", "wmtopo") != d:
        raise ValueError(f"{d.name}: not a corrected segmentation (dseg_path)")
    return dict(
        dseg=d,
        fixes=at("dseg", ".nii.gz", "wmtopofixes"),
        table=at("fixes", ".tsv", "wmtopofixes"),
        diff=at("diff", ".npy", "wmtopofixes"),
        edits=at("edits", ".npy", "wmtopo"),
    )


def history_dir(dseg: Path) -> Path:
    """The directory of the kept runs of the dseg at this path."""
    return dseg.parent.parent / HISTORY


def hemisphere(labels: npt.ArrayLike) -> str:
    """R or L: the hemisphere whose WM label is in the segmentation. Raises
    ValueError for both (a whole brain needs a midline cut first) or
    neither."""
    present = [h for h, w in WM_LABEL.items() if np.any(np.asarray(labels) == w)]
    if len(present) != 1:
        raise ValueError(
            f"WM labels present for hemispheres {present}: need exactly one"
        )
    return present[0]


def to_outside(
    labels: npt.NDArray[np.integer[Any]],
) -> tuple[npt.NDArray[np.integer[Any]], dict[int, int]]:
    """labels with each voxel of a label in TO_OUTSIDE set to OUTSIDE (a new
    array if a voxel changes, else labels itself), and the number of voxels
    set, by input label (only the labels that are present)."""
    m = np.isin(labels, TO_OUTSIDE)
    if not m.any():
        return labels, {}
    old, cnt = np.unique(labels[m], return_counts=True)
    out = labels.copy()
    out[m] = OUTSIDE
    return out, {int(k): int(c) for k, c in zip(old, cnt, strict=True)}


def cavities(obj: Bool) -> Bool:
    """The background voxels that do not reach the array's border (Kuhn
    adjacency, as baillarger.handles.kuhn_betti counts cavities)."""
    lab, _ = ndi.label(np.pad(~obj, 1, constant_values=True), kuhn_structure())
    out: Bool = (lab[1:-1, 1:-1, 1:-1] != lab[0, 0, 0]) & ~obj
    return out


def relabel_cuts(
    labels: npt.NDArray[np.integer[Any]],
    cut: Bool,
    donor: Bool,
    voxel_mm: npt.ArrayLike,
    margin: int = 4,
) -> None:
    """Give each cut voxel the label of the nearest donor voxel (Euclidean, in
    mm), in place. Each connected piece of the cut is done in its bounding
    box plus margin voxels, widened until the nearest donor found is nearer
    than the box's edge."""
    vox = np.asarray(voxel_mm, np.float64)
    lab, _ = ndi.label(cut, np.ones((3, 3, 3), bool))
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        m = margin
        while True:
            box = tuple(
                slice(max(t.start - m, 0), min(t.stop + m, n))
                for t, n in zip(sl, labels.shape, strict=True)
            )
            d = donor[box]
            whole = all(
                b.start == 0 and b.stop == n
                for b, n in zip(box, labels.shape, strict=True)
            )
            if d.any():
                dist, idx = ndi.distance_transform_edt(
                    ~d, sampling=vox, return_indices=True
                )
                sel = lab[box] == k
                if whole or dist[sel].max() <= m * vox.min():
                    break
            elif whole:
                raise ValueError("no donor voxel for a cut")
            m *= 2
        src = tuple(i[sel] for i in idx)
        sub = labels[box]
        sub[sel] = sub[src]


#: the 26 neighbours of a voxel, as offsets
NEIGHBOURS_26 = np.array(
    [o for o in itertools.product((-1, 0, 1), repeat=3) if any(o)], np.intp
)


def small_pieces(
    obj: Bool, voxel_mm: npt.ArrayLike, max_mm3: float = CRUMB_MM3
) -> tuple[tuple[npt.NDArray[np.intp], ...], npt.NDArray[np.intp]]:
    """The voxels of the pieces of obj (Kuhn adjacency, as
    baillarger.handles.kuhn_betti counts pieces) whose volume (voxels x
    prod(voxel_mm)) is less than max_mm3: their indices (as np.nonzero
    gives them), and the number of the piece of each voxel (1 to the number
    of small pieces, in the order of ndimage.label)."""
    lab, n = ndi.label(obj, kuhn_structure())
    sz = np.bincount(lab.ravel(), minlength=n + 1)
    sel = sz * float(np.prod(np.asarray(voxel_mm, np.float64))) < max_mm3
    sel[0] = False
    idx = np.nonzero(sel[lab])
    renum = np.cumsum(sel).astype(np.intp)
    return idx, renum[lab[idx]]


def relabel_small_pieces(
    labels: npt.NDArray[np.integer[Any]],
    idx: tuple[npt.NDArray[np.intp], ...],
    filled: tuple[int, ...],
    origin: npt.ArrayLike = (0, 0, 0),
) -> int:
    """Give each voxel at idx (each has a filled label) a label that is not
    filled, in place, layer by layer from the outside in. Layer 1: each voxel
    with a 26-neighbour (in the array) that does not have a filled label
    takes the most common label of those neighbours. Layer k: each voxel
    that is left and has such a neighbour, outside idx or given its label in
    layers 1 to k-1, takes the most common label of those neighbours. Each
    layer reads only the labels before it, so the order of the voxels does
    not matter. If two labels have the same count, the smaller label value
    wins. Returns the number of layers. Raises RuntimeError if a layer gives
    no label while voxels are left (a piece with no voxel outside the mask
    round it: it fills the array); the error gives a voxel's index plus
    origin (the index in the whole grid, for labels that are a crop)."""
    p = np.stack(idx, axis=1)
    shape = np.array(labels.shape)
    if not np.isin(labels[idx], filled).all():
        raise ValueError("a voxel of a small piece does not have a filled label")
    layers = 0
    while len(p):
        q = p[:, None, :] + NEIGHBOURS_26[None]
        inside = ((q >= 0) & (q < shape)).all(axis=2)
        q = np.clip(q, 0, shape - 1)
        v = labels[q[..., 0], q[..., 1], q[..., 2]]
        ok = inside & ~np.isin(v, filled)
        now = ok.any(axis=1)
        if not now.any():
            at = (p[0] + np.asarray(origin)).tolist()
            raise RuntimeError(
                f"the voxel {at} of a small piece of the WM mask has no "
                "26-neighbour outside the mask, and no voxel of its piece has "
                "one: there is no label to give it"
            )
        v, ok = v[now], ok[now]
        cand = np.unique(v[ok])  # sorted: argmax takes the smallest label of a tie
        counts = ((v[..., None] == cand) & ok[..., None]).sum(axis=1)
        labels[tuple(p[now].T)] = cand[counts.argmax(axis=1)]
        p = p[~now]
        layers += 1
    return layers


#: the 13 offsets of the 26-neighbourhood that come after the voxel in C order
FORWARD_26 = np.array(
    [d for d in itertools.product((-1, 0, 1), repeat=3) if d > (0, 0, 0)], np.int64
)


def manual_pieces(index: Int, shape: tuple[int, ...]) -> list[Int]:
    """The 26-connected pieces of the voxels at index (sorted, unique linear
    indices of a grid of shape), each as sorted linear indices, in the order
    of their smallest index (the order of ndimage.label). Sparse: O(E log E)
    time and O(E) memory for E voxels, whatever their spread."""
    index = np.asarray(index, np.int64)
    n = len(index)
    if not n:
        return []
    ijk = np.stack(np.unravel_index(index, shape), 1)
    rows, cols = [], []
    for d in FORWARD_26:
        q = ijk + d
        ok = np.all((q >= 0) & (q < np.asarray(shape)), axis=1)
        src = np.flatnonzero(ok)
        nb = np.ravel_multi_index(tuple(q[ok].T), shape)
        at = np.minimum(np.searchsorted(index, nb), n - 1)
        hit = index[at] == nb
        rows.append(src[hit])
        cols.append(at[hit])
    r, c = np.concatenate(rows), np.concatenate(cols)
    g = sparse.coo_matrix((np.ones(len(r), np.int8), (r, c)), shape=(n, n))
    _, lab = csgraph.connected_components(g.tocsr(), directed=False)
    # number the pieces by their first voxel: index is sorted
    _, first, inv = np.unique(lab, return_index=True, return_inverse=True)
    piece = np.argsort(np.argsort(first))[inv]
    order = np.argsort(piece, kind="stable")  # stable: each piece stays sorted
    ends = np.cumsum(np.bincount(piece))[:-1]
    return np.split(index[order], ends)


@dataclass(frozen=True)
class WMTopo:
    """The corrected segmentation and what was changed."""

    #: the segmentation with the WM mask made cavity- and handle-free
    dseg: npt.NDArray[np.integer[Any]]
    #: FIXES values at the changed voxels, 0 elsewhere
    fixes: npt.NDArray[np.uint8]
    hemi: str
    betti_before: Betti
    betti_after: Betti
    #: label: count, for the cut voxels
    cut_labels: dict[int, int]
    #: per manual edit, handle move and small piece: Kind, Voxels, Centre
    #: (0-based voxel of the grid)
    moves: list[dict[str, Any]]
    #: voxels changed by the manual edits
    manual_voxels: int = 0
    #: pieces of the mask removed because they were below CRUMB_MM3
    small_pieces: int = 0
    #: the fix diff (baillarger.fix_diff.DIFF_DTYPE): from the aseg with the
    #: label edits applied to dseg
    diff: npt.NDArray[np.void] = field(default_factory=lambda: np.zeros(0, DIFF_DTYPE))
    #: the fixes, by id (1, 2, ...): the voxels that each changed last
    fix_list: list[Fix] = field(default_factory=list)
    #: input label: count, for the voxels set to OUTSIDE (to_outside)
    to_outside: dict[int, int] = field(default_factory=dict)


def groups(ids: npt.ArrayLike, n: int) -> list[Int]:
    """For each label k from 1 to n, the positions i with ids[i] == k, in
    increasing order: one stable sort, O(V log V) for V ids (each id from 0
    to n), not one pass over the ids for each label."""
    ids = np.asarray(ids, np.int64)
    order = np.argsort(ids, kind="stable")
    ends = np.cumsum(np.bincount(ids, minlength=n + 1))
    return np.split(order, ends[:-1])[1:]


def wm_topo(
    aseg: npt.ArrayLike, voxel_mm: npt.ArrayLike, edits: Edits | None = None
) -> WMTopo:
    """The corrected segmentation of an aseg: the label edits (a diff,
    baillarger.label_edits) applied first, then the labels of
    TO_OUTSIDE set to OUTSIDE (to_outside), then the mask of its
    FILLED_LABELS,
    with its cavities filled and its handles removed
    (baillarger.handles.remove_handles). Cavity and handle fills become the
    WM label; each handle cut takes the label of the nearest voxel that is
    outside the corrected mask and not a cut. Then the pieces of the mask
    below CRUMB_MM3 (small_pieces) are removed: each of their voxels takes
    the most common label of its 26-neighbours outside the mask, layer by
    layer from the outside in (relabel_small_pieces). The corrected
    segmentation's voxels with a filled label are exactly the corrected
    mask. Raises RuntimeError if that mask
    still has a handle or a cavity (b1 or b2 not 0), which a fill of one
    piece of the mask touching another can make. Handle removal changes
    the edited voxels (each voxel of the diff) last (they are the verified
    mask of remove_handles), and a fix that changes some records how many
    (Fix.overrides). Each 26-connected piece of the edited voxels whose label
    differs from the input is a fix of the kind manual_edit."""
    lap = Laps()
    a_in = np.asarray(aseg)
    a0 = a_in
    shape = a_in.shape
    # the edited voxels, and those of them whose label differs from the input
    ver_idx = man_idx = np.zeros(0, np.int64)
    if edits is not None:
        a0 = apply_edits(a_in, edits)
        ver_idx = edits.index
        flat_in, flat0 = a_in.reshape(-1), a0.reshape(-1)
        man_idx = ver_idx[flat0[ver_idx] != flat_in[ver_idx]]
        lap("edits")
    a0, outside = to_outside(a0)
    lap("to outside")
    hemi = hemisphere(a0)
    filled, wm_label = FILLED_LABELS[hemi], WM_LABEL[hemi]
    m = np.isin(a0, filled)
    nz = [
        np.flatnonzero(m.any(axis=tuple(j for j in range(3) if j != i)))
        for i in range(3)
    ]
    lo = np.array([max(int(t[0]) - 2, 0) for t in nz])
    hi = np.array([min(int(t[-1]) + 3, s) for t, s in zip(nz, m.shape, strict=True)])
    box = tuple(slice(int(p), int(q)) for p, q in zip(lo, hi, strict=True))
    x = np.ascontiguousarray(m[box])
    del m
    lap("mask")
    before = kuhn_betti(x)
    lap("Betti before")
    cav = cavities(x)
    x |= cav
    lap("cavities")
    verified = np.zeros(x.shape, bool)
    vq = np.stack(np.unravel_index(ver_idx, shape), 1) - lo
    vq = vq[np.all((vq >= 0) & (vq < np.asarray(x.shape)), axis=1)]
    verified[tuple(vq.T)] = True
    r = remove_handles(x, verified)
    del verified, vq
    obj = r.obj
    lap("handles")
    cut, fill = x & ~obj, obj & ~x
    a = a0[box]
    labels = a.copy()
    labels[cav | fill] = wm_label
    relabel_cuts(labels, cut, ~obj & ~cut, voxel_mm)
    lap("relabel cuts")
    # the small pieces, from the labels above, layer by layer
    small, piece = small_pieces(obj, voxel_mm)
    n_small = int(piece.max()) if len(piece) else 0
    relabel_small_pieces(labels, small, filled, lo)
    obj[small] = False
    lap("small pieces")
    if not np.array_equal(np.isin(labels, filled), obj):
        raise AssertionError("the corrected labels do not give the corrected mask")
    after = kuhn_betti(obj)
    lap("Betti after")
    if after[1] or after[2]:
        # remove_handles fixes each piece with the others as background: a fill
        # of one piece can touch another and make a handle or a cavity
        raise RuntimeError(
            f"the corrected WM mask has Betti numbers {after} (before: {before}): "
            "a handle fill of one piece of the mask touched another"
        )
    dseg = a0.copy()
    dseg[box] = labels
    lap("labels")
    # the fixes, with their ids in the fixed order: the manual edits (one fix
    # a 26-connected piece of the edited voxels that differ from the input,
    # in the order of their smallest index), the handle moves (in the order of
    # remove_handles), the cavities (one fix a connected cavity, in the order
    # of ndimage.label), the small pieces
    steps: list[tuple[str, Int, int | None]] = []  # kind, voxels (flat), wall
    if edits is not None:
        for v in manual_pieces(man_idx, shape):
            steps.append(("manual_edit", v, None))
    n_manual = len(steps)

    def flat_of(ijk: npt.ArrayLike) -> Int:
        q = np.asarray(ijk, np.int64).reshape(-1, 3) + lo
        return np.asarray(np.ravel_multi_index(tuple(q.T), shape), np.int64)

    for mv in r.moves:
        kind = "handle_fill" if mv.kind == "fill" else "handle_cut"
        steps.append((kind, flat_of(mv.voxels), len(mv.voxels)))
    cav_lab, n_cav = ndi.label(cav, kuhn_structure())
    cav_vox = np.argwhere(cav_lab > 0)
    cav_id = cav_lab[cav_lab > 0]
    for g in groups(cav_id, n_cav):
        steps.append(("cavity_fill", flat_of(cav_vox[g]), None))
    sp_vox = np.stack(small, axis=1)
    sp_groups = groups(piece, n_small)
    for g in sp_groups:
        steps.append(("small_piece", flat_of(sp_vox[g]), None))
    # the fix that changed each voxel last, in the order that the steps run:
    # the answers, the cavity fill, the handle moves, the small pieces
    n_move = len(r.moves)
    run = (
        list(range(n_manual))
        + list(range(n_manual + n_move, n_manual + n_move + n_cav))
        + list(range(n_manual, n_manual + n_move))
        + list(range(n_manual + n_move + n_cav, len(steps)))
    )
    own_idx = [steps[s][1] for s in run] or [np.zeros(0, np.int64)]
    own_fix = [np.full(len(steps[s][1]), s + 1, np.int64) for s in run]
    oi = np.concatenate(own_idx)
    of = np.concatenate(own_fix) if own_fix else np.zeros(0, np.int64)
    order = np.argsort(oi, kind="stable")
    oi, of = oi[order], of[order]
    last = np.r_[oi[1:] != oi[:-1], True] if len(oi) else np.zeros(0, bool)
    oi, of = oi[last], of[last]
    sub = np.zeros(shape, bool)
    sub[box] = labels != a
    changed = np.flatnonzero(sub)
    del sub
    at = np.searchsorted(oi, changed)
    at = np.minimum(at, max(len(oi) - 1, 0))
    if len(changed) and not np.array_equal(oi[at], changed):
        raise AssertionError("a voxel changed by the correction has no fix")
    diff = make_diff(a0, dseg, changed, of[at] if len(changed) else changed)
    lap("fix diff")
    fixes = np.zeros(shape, np.uint8)
    code = {v: k for k, v in FIXES.items()}
    fixes.reshape(-1)[man_idx] = 4  # an edited voxel that no fix then changes
    kind_code = np.array([0] + [code[st[0]] for st in steps], np.uint8)
    fixes.reshape(-1)[diff["index"]] = kind_code[diff["fix"]]
    # per fix, the voxels that it changed last: the diff's rows, and for an
    # answer its voxels that no fix then changes
    in_diff = np.zeros(len(steps) + 1, np.int64)
    np.add.at(in_diff, diff["fix"].astype(np.int64), 1)
    by_fix = np.argsort(diff["fix"], kind="stable")
    starts = np.r_[0, np.cumsum(in_diff)]
    # per fix, the number of edited voxels that it changes: once for all
    # fixes, O((D + E) log) for D rows of the diff and E edited voxels
    over_n = np.bincount(
        diff["fix"][np.isin(diff["index"], ver_idx)].astype(np.int64),
        minlength=len(steps) + 1,
    )
    # the voxels of the manual pieces that are not in the diff, once for all
    man_len = [len(st[1]) for st in steps[:n_manual]]
    man_all = np.concatenate(
        [np.zeros(0, np.int64)] + [st[1] for st in steps[:n_manual]]
    )
    keep = ~np.isin(man_all, diff["index"])
    man_pid = np.repeat(np.arange(n_manual), man_len)
    man_n = np.bincount(man_pid[keep], minlength=n_manual)
    man_keep = np.split(man_all[keep], np.cumsum(man_n)[:-1])
    fix_list = []
    for s, (kind, _, wall) in enumerate(steps):
        if kind == "manual_edit":
            idx = man_keep[s]
        else:
            idx = np.sort(diff["index"][by_fix[starts[s + 1] : starts[s + 2]]])
        over = None
        if kind != "manual_edit" and over_n[s + 1]:
            over = int(over_n[s + 1])
        fix_list.append(Fix(s + 1, kind, idx.astype(np.int64), wall, over))
    lap("fix table")
    new, cnt = np.unique(labels[cut], return_counts=True)
    moves = []
    for _, flat, _ in steps[:n_manual]:
        v = np.stack(np.unravel_index(flat, shape), 1)
        moves.append(
            dict(
                Kind="manual",
                Voxels=len(v),
                Centre=np.round(v.mean(0)).astype(int).tolist(),
            )
        )
    moves += [
        dict(
            Kind=mv.kind,
            Voxels=len(mv.voxels),
            Centre=np.round(mv.voxels.mean(0) + lo).astype(int).tolist(),
        )
        for mv in r.moves
    ]
    sp = sp_vox + lo
    for g in sp_groups:
        moves.append(
            dict(
                Kind="small_piece",
                Voxels=len(g),
                Centre=np.round(sp[g].mean(0)).astype(int).tolist(),
            )
        )
    return WMTopo(
        dseg,
        fixes,
        hemi,
        before,
        after,
        {int(k): int(c) for k, c in zip(new, cnt, strict=True)},
        moves,
        len(man_idx),
        n_small,
        diff,
        fix_list,
        outside,
    )


def _write_tsv(path: Path, names: dict[int, str]) -> None:
    rows = "".join(f"{k}\t{v}\n" for k, v in sorted(names.items()))
    path.write_text("index\tname\n" + rows)


def write_wm_topo(aseg: Path, out: Path, edits: Path | None = None) -> dict[str, Path]:
    """The dseg command: wm_topo of the aseg at aseg, with the label edits at
    edits (a .npy diff, docs/formats/label-edits.md) if given, written to the
    BIDS derivatives root out (docs/formats/wm-dseg.md). The output keeps a
    copy of the diff that it used (role "edits"; the empty diff if edits is
    None). Raises ValueError for a diff on another grid. Returns the paths
    written, by role."""
    img = nib.load(aseg)
    if not isinstance(img, nib.nifti1.Nifti1Image):
        raise ValueError(f"{aseg.name}: not a NIfTI-1 image")
    ents = entities(aseg)
    vox = tuple(float(z) for z in img.header.get_zooms()[:3])
    grid = tuple(int(n) for n in img.shape[:3])
    e = None
    edits_sha = None
    lap = Laps()
    a = np.asarray(img.dataobj)
    lap("read")
    if edits is not None:
        e, _, edits_sha = label_edits.read_with_sidecar(
            edits, shape=grid, affine=img.affine, hemisphere=hemisphere(a)
        )
    t = wm_topo(a, vox, e)
    lap = Laps()
    dataset_description(out)
    p = output_paths(dseg_path(out, ents))
    if edits is not None and edits.resolve() == p["edits"].resolve():
        raise ValueError(f"{edits.name}: the diff is the output's copy of a diff")
    prov = provenance("baillarger.white_dseg.write_wm_topo")
    inputs = dict(Labels=file_sha256(aseg), Code=f"{prov['Version']}+{prov['Commit']}")
    if edits_sha is not None:
        inputs["Edits"] = edits_sha
    raw = io.BytesIO()
    np.save(raw, t.diff, allow_pickle=False)
    diff_sha = hashlib.sha256(raw.getvalue()).hexdigest()
    hist = history_dir(p["dseg"])
    check_reproduced(hist, inputs, bool(prov["Dirty"]), diff_sha)
    p["dseg"].parent.mkdir(parents=True, exist_ok=True)
    # the copy of the diff that this run used: the sidecars list it
    label_edits.write(
        p["edits"],
        label_edits.empty() if e is None else e,
        hemisphere=t.hemi,
        shape=grid,
        affine=img.affine,
        description="The copy of the label edits that the dseg command used.",
    )
    # the copy's records only: its sidecar holds the time of this run, so its
    # SHA-256 would differ between two runs that are otherwise the same
    src = sources(aseg, p["edits"])
    hdr = img.header.copy()
    hdr.set_data_dtype(t.dseg.dtype)
    nib.save(nib.nifti1.Nifti1Image(t.dseg, img.affine, hdr), p["dseg"])
    nib.save(nib.nifti1.Nifti1Image(t.fixes, img.affine), p["fixes"])
    present = np.unique(t.dseg).tolist()
    _write_tsv(
        p["dseg"].with_name(p["dseg"].name.replace(".nii.gz", ".tsv")),
        {k: LUT_NAMES.get(k, f"label-{k}") for k in present},
    )
    _write_tsv(p["fixes"].with_name(p["fixes"].name.replace(".nii.gz", ".tsv")), FIXES)
    lap("write")
    write_sidecar(
        sidecar_path(p["dseg"]),
        DSEG_SPEC,
        dict(
            SpecVersion=DSEG_SPEC.version,
            Description="The segmentation with the ventricles, the choroid "
            "plexus and CSF set to outside (ToOutside), and its WM mask "
            "(FilledLabels) made cavity- and handle-free, without pieces below "
            f"{CRUMB_MM3:g} mm3.",
            **src,
            Provenance=prov,
            Hemisphere=t.hemi,
            ToOutside={str(k): v for k, v in t.to_outside.items()},
            FilledLabels=list(FILLED_LABELS[t.hemi]),
            WMLabel=WM_LABEL[t.hemi],
            BettiBefore=list(t.betti_before),
            BettiAfter=list(t.betti_after),
            CavityVoxels=int((t.fixes == 3).sum()),
            HandleFillVoxels=int((t.fixes == 2).sum()),
            HandleCutVoxels=int((t.fixes == 1).sum()),
            CutLabels={str(k): v for k, v in t.cut_labels.items()},
            SmallPieces=t.small_pieces,
            SmallPieceVoxels=int((t.fixes == 5).sum()),
            ManualVoxels=t.manual_voxels,
            Moves=t.moves,
        ),
    )
    write_table(p["table"], table_rows(t.fix_list, t.dseg.shape, img.affine))
    p["diff"].write_bytes(raw.getvalue())
    kinds: dict[str, int] = {}
    for f in t.fix_list:
        kinds[f.kind] = kinds.get(f.kind, 0) + 1
    side = dict(
        SpecVersion=FIXES_SPEC.version,
        Description="The changes of the WM topology correction: the image ("
        + ", ".join(f"{k} {v}" for k, v in sorted(FIXES.items()))
        + "), the fix table and the fix diff.",
        **src,
        Provenance=prov,
        FixTable=p["table"].name,
        FixDiff=p["diff"].name,
        DiffSHA256=diff_sha,
        DiffVoxels=len(t.diff),
        DiffInputs=inputs,
        Fixes=kinds,
    )
    for role in ("fixes", "table", "diff"):
        write_sidecar(sidecar_path(p[role]), FIXES_SPEC, side)
    # every run's diff is kept, by its SHA-256
    hist.mkdir(exist_ok=True)
    shutil.copyfile(p["diff"], hist / f"{diff_sha}.npy")
    shutil.copyfile(p["table"], hist / f"{diff_sha}.tsv")
    write_sidecar(hist / f"{diff_sha}.json", FIXES_SPEC, side)
    lap("fix table and diff")
    return p


def check_reproduced(
    history: Path, inputs: dict[str, str], dirty: bool, diff_sha: str
) -> None:
    """Raises RuntimeError if a kept run (a sidecar in the history directory)
    had the same inputs and code (DiffInputs), with no uncommitted change in
    either run, and a fix diff other than diff_sha: the correction did not
    reproduce its diff. A run with uncommitted changes is not checked."""
    if dirty or not history.is_dir():
        return
    for js in sorted(history.glob("*.json")):
        old = read_sidecar(js, FIXES_SPEC)
        if old["Provenance"]["Dirty"] or old["DiffInputs"] != inputs:
            continue
        if old["DiffSHA256"] != diff_sha:
            raise RuntimeError(
                f"the fix diff {diff_sha} does not reproduce the kept diff "
                f"{old['DiffSHA256']} of the same inputs and code ({js.name})"
            )


def replay_dseg(
    aseg: npt.ArrayLike, diff: npt.NDArray[np.void], edits: Edits | None = None
) -> npt.NDArray[Any]:
    """desc-wmtopo rebuilt from the input labels, the label edits
    (baillarger.label_edits.apply), the labels set to outside (to_outside)
    and the fix diff (baillarger.fix_diff.apply_diff), with no correction
    run."""
    a = np.asarray(aseg)
    if edits is not None:
        a = apply_edits(a, edits)
    return apply_diff(to_outside(a)[0], diff)


def dseg_sidecar(dseg: Path) -> dict[str, Any]:
    """The sidecar of a corrected segmentation (write_wm_topo), checked
    against wm-dseg/1."""
    return read_sidecar(sidecar_path(Path(dseg)), DSEG_SPEC)


def filled_labels(dseg: Path) -> tuple[int, ...]:
    """The FilledLabels of a corrected segmentation (write_wm_topo), from its
    sidecar: its voxels with one of these labels are the corrected WM mask."""
    return tuple(int(k) for k in dseg_sidecar(dseg)["FilledLabels"])


def input_labels(dseg: Path) -> Path:
    """The input labels of a corrected segmentation (write_wm_topo): the
    first of its sidecar's Sources (wm-dseg/1)."""
    return Path(dseg_sidecar(dseg)["Sources"][0])


def surface_dseg(sidecar: dict[str, Any]) -> Path:
    """The corrected segmentation of a white surface, from the surface's
    sidecar (white-surf/1, already read): the first of its Sources."""
    return Path(sidecar["Sources"][0])


# ---------------------------------------------------------------- command line


def main(argv: list[str] | None = None) -> None:
    log_to_stderr()
    ap = argparse.ArgumentParser(
        description="the segmentation with a cavity- and handle-free WM mask"
    )
    ap.add_argument("aseg", type=Path, help="a BIDS-named segmentation (NIfTI)")
    ap.add_argument(
        "--edits", type=Path, help="the label edits (a .npy diff, label-edits/1)"
    )
    ap.add_argument("--out", required=True, type=Path, help="BIDS derivatives root")
    a = ap.parse_args(argv)
    for role, path in write_wm_topo(a.aseg, a.out, a.edits).items():
        print(f"{role}: {path}")


if __name__ == "__main__":
    main()
