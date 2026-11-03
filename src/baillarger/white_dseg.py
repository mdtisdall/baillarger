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
mask's cavities and handle fills set to the WM label and each handle cut
given the label of its nearest neighbour outside the mask. Every later stage
reads its labels from the corrected segmentation, never the aseg plus a list
of changes: the whole-hemisphere surface (baillarger.white_surf) and the box
stages (baillarger.column_traces.load_tissue, baillarger.pial_labels, whose
WM is the corrected mask of their box: filled_labels).

The command writes the corrected segmentation (desc-wmtopo_dseg) and the
changed voxels (desc-wmtopofixes_dseg), with their .tsv and .json sidecars
(docs/formats/wm-dseg.md); EDITS, manual edits of the aseg
(desc-wmedits_edits.tsv, baillarger.wm_edits), are made before the automatic
correction.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi

from baillarger.bids import (
    COMMON_KEYS,
    Key,
    SidecarSpec,
    dataset_description,
    derivative_path,
    entities,
    provenance,
    read_sidecar,
    sidecar_path,
    sources,
    write_sidecar,
)
from baillarger.handles import kuhn_betti, kuhn_structure, remove_handles
from baillarger.timing import Laps, log_to_stderr
from baillarger.wm_edits import Edits, apply_edits, read_edits, sources_match

type Bool = npt.NDArray[np.bool_]
type Betti = tuple[int, int, int]

# ------------------------------------------------- the corrected segmentation

#: the labels filled into the WM mask, by hemisphere (FreeSurfer LUT): WM,
#: lateral ventricle, thalamus, caudate, putamen, pallidum, corpus callosum.
#: The hippocampus is not filled: its laminar axis does not run out from the
#: WM below it.
FILLED_LABELS = {"R": (41, 43, 49, 50, 51, 52, 192), "L": (2, 4, 10, 11, 12, 13, 192)}
WM_LABEL = {"R": 41, "L": 2}

#: FreeSurfer LUT names of the labels the corrected segmentation's .tsv lists
LUT_NAMES = {
    0: "Unknown",
    2: "Left-Cerebral-White-Matter",
    3: "Left-Cerebral-Cortex",
    4: "Left-Lateral-Ventricle",
    10: "Left-Thalamus",
    11: "Left-Caudate",
    12: "Left-Putamen",
    13: "Left-Pallidum",
    17: "Left-Hippocampus",
    18: "Left-Amygdala",
    41: "Right-Cerebral-White-Matter",
    42: "Right-Cerebral-Cortex",
    43: "Right-Lateral-Ventricle",
    49: "Right-Thalamus",
    50: "Right-Caudate",
    51: "Right-Putamen",
    52: "Right-Pallidum",
    53: "Right-Hippocampus",
    54: "Right-Amygdala",
    192: "Corpus_Callosum",
}

#: values of the fixes image: a handle cut, a handle fill, a cavity fill, a
#: manual edit (baillarger.wm_edits)
FIXES = {1: "handle_cut", 2: "handle_fill", 3: "cavity_fill", 4: "manual_edit"}

#: docs/formats/wm-dseg.md
DSEG_SPEC = SidecarSpec(
    "wm-dseg/1",
    COMMON_KEYS
    + (
        Key("Hemisphere", "string", True, "R or L"),
        Key("FilledLabels", "array of integers", True, "labels of the WM mask"),
        Key("WMLabel", "integer", True, "label given to cavity and handle fills"),
        Key("BettiBefore", "array of integers", True, "b0, b1, b2 of the mask"),
        Key("BettiAfter", "array of integers", True, "b0, b1, b2 after"),
        Key("CavityVoxels", "integer", True, "voxels set to WM: cavities"),
        Key("HandleFillVoxels", "integer", True, "voxels set to WM: handle fills"),
        Key("HandleCutVoxels", "integer", True, "voxels cut from the mask"),
        Key("CutLabels", "object", True, "new label of the cut voxels: count"),
        Key("ManualVoxels", "integer", True, "voxels changed by manual edits"),
        Key("Moves", "array of objects", True, "each manual edit and handle move"),
    ),
)
#: docs/formats/wm-dseg.md
FIXES_SPEC = SidecarSpec("wm-dseg-fixes/1", COMMON_KEYS)


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
    #: per manual edit (Answer: its Id) and handle move: Kind, Voxels, Centre
    #: (0-based voxel of the grid)
    moves: list[dict[str, Any]]
    #: voxels changed by the manual edits
    manual_voxels: int = 0


def wm_topo(
    aseg: npt.ArrayLike, voxel_mm: npt.ArrayLike, edits: Edits | None = None
) -> WMTopo:
    """The corrected segmentation of an aseg: the manual edits made first
    (baillarger.wm_edits.apply_edits; it raises ValueError for edits that do
    not fit the aseg), then the mask of its FILLED_LABELS,
    with its cavities filled and its handles removed
    (baillarger.handles.remove_handles). Cavity and handle fills become the
    WM label; each handle cut takes the label of the nearest voxel that is
    outside the corrected mask and not a cut. The corrected segmentation's
    voxels with a filled label are exactly the corrected mask. Raises
    RuntimeError if that mask still has a handle or a cavity (b1 or b2 not 0),
    which a fill of one piece of the mask touching another can make."""
    lap = Laps()
    a0 = np.asarray(aseg)
    manual = None
    if edits is not None:
        a0, manual = apply_edits(a0, edits)
        lap("manual edits")
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
    r = remove_handles(x)
    obj = r.obj
    lap("handles")
    cut, fill = x & ~obj, obj & ~x
    a = a0[box]
    labels = a.copy()
    labels[cav | fill] = wm_label
    relabel_cuts(labels, cut, ~obj & ~cut, voxel_mm)
    lap("relabel cuts")
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
    fixes = np.zeros(a0.shape, np.uint8)
    if manual is not None:
        fixes[manual] = 4  # an edited voxel the correction then changes shows that
    fx = fixes[box]
    fx[cut] = 1
    fx[fill] = 2
    fx[cav] = 3
    new, cnt = np.unique(labels[cut], return_counts=True)
    moves = []
    if edits is not None:
        for ans in edits.answers:
            v = edits.voxels[edits.answer == ans.id]
            if len(v):
                moves.append(
                    dict(
                        Kind="manual",
                        Voxels=len(v),
                        Centre=np.round(v.mean(0)).astype(int).tolist(),
                        Answer=ans.id,
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
    return WMTopo(
        dseg,
        fixes,
        hemi,
        before,
        after,
        {int(k): int(c) for k, c in zip(new, cnt, strict=True)},
        moves,
        0 if manual is None else int(manual.sum()),
    )


def _write_tsv(path: Path, names: dict[int, str]) -> None:
    rows = "".join(f"{k}\t{v}\n" for k, v in sorted(names.items()))
    path.write_text("index\tname\n" + rows)


def write_wm_topo(aseg: Path, out: Path, edits: Path | None = None) -> dict[str, Path]:
    """The dseg command: wm_topo of the aseg at aseg, with the manual edits in
    the table at edits (docs/formats/wm-edits.md) if given, written to the BIDS
    derivatives root out (docs/formats/wm-dseg.md). Raises ValueError for edits
    made against another aseg, or that do not fit it. Returns the paths
    written, by role."""
    img = nib.load(aseg)
    if not isinstance(img, nib.nifti1.Nifti1Image):
        raise ValueError(f"{aseg.name}: not a NIfTI-1 image")
    ents = entities(aseg)
    vox = tuple(float(z) for z in img.header.get_zooms()[:3])
    e = None
    if edits is not None:
        e, side = read_edits(edits)
        if not sources_match(side, aseg):
            raise ValueError(f"{edits.name}: the edits are not against {aseg.name}")
    lap = Laps()
    a = np.asarray(img.dataobj)
    lap("read")
    t = wm_topo(a, vox, e)
    lap = Laps()
    dataset_description(out)
    p = dict(
        dseg=derivative_path(out, ents, "dseg", ".nii.gz", desc="wmtopo"),
        fixes=derivative_path(out, ents, "dseg", ".nii.gz", desc="wmtopofixes"),
    )
    p["dseg"].parent.mkdir(parents=True, exist_ok=True)
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
    src = sources(aseg) if edits is None else sources(aseg, edits, sidecar_path(edits))
    prov = provenance("baillarger.white_dseg.write_wm_topo")
    write_sidecar(
        sidecar_path(p["dseg"]),
        DSEG_SPEC,
        dict(
            SpecVersion=DSEG_SPEC.version,
            Description="The segmentation with its WM mask (FilledLabels) made "
            "cavity- and handle-free.",
            **src,
            Provenance=prov,
            Hemisphere=t.hemi,
            FilledLabels=list(FILLED_LABELS[t.hemi]),
            WMLabel=WM_LABEL[t.hemi],
            BettiBefore=list(t.betti_before),
            BettiAfter=list(t.betti_after),
            CavityVoxels=int((t.fixes == 3).sum()),
            HandleFillVoxels=int((t.fixes == 2).sum()),
            HandleCutVoxels=int((t.fixes == 1).sum()),
            CutLabels={str(k): v for k, v in t.cut_labels.items()},
            ManualVoxels=t.manual_voxels,
            Moves=t.moves,
        ),
    )
    write_sidecar(
        sidecar_path(p["fixes"]),
        FIXES_SPEC,
        dict(
            SpecVersion=FIXES_SPEC.version,
            Description="The voxels that the WM topology correction changed: "
            "1 handle cut, 2 handle fill, 3 cavity fill, 4 manual edit.",
            **src,
            Provenance=prov,
        ),
    )
    return p


def filled_labels(dseg: Path) -> tuple[int, ...]:
    """The FilledLabels of a corrected segmentation (write_wm_topo), from its
    sidecar: its voxels with one of these labels are the corrected WM mask."""
    side = read_sidecar(sidecar_path(dseg), DSEG_SPEC)
    return tuple(int(k) for k in side["FilledLabels"])


# ---------------------------------------------------------------- command line


def main(argv: list[str] | None = None) -> None:
    log_to_stderr()
    ap = argparse.ArgumentParser(
        description="the segmentation with a cavity- and handle-free WM mask"
    )
    ap.add_argument("aseg", type=Path, help="a BIDS-named segmentation (NIfTI)")
    ap.add_argument(
        "--edits", type=Path, help="manual edits of the aseg (desc-wmedits_edits.tsv)"
    )
    ap.add_argument("--out", required=True, type=Path, help="BIDS derivatives root")
    a = ap.parse_args(argv)
    for role, path in write_wm_topo(a.aseg, a.out, a.edits).items():
        print(f"{role}: {path}")


if __name__ == "__main__":
    main()
