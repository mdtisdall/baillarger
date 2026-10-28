"""The handle-free WM label: the aseg's WM label (41) with every handle
removed, so that the white mesh of any box has genus-0 pieces.

The correction is made once, on the whole label (baillarger.handles,
remove_handles), because a box crop of a handle-free solid has planar surface
pieces, while a box alone cannot tell a handle from a fold that closes outside
it. wm_changes computes it (for the example hemisphere: a few minutes and
about 20 GB) and returns only the changed voxels, in full-grid coordinates.
Save them once (WMChanges.save); white_matter_box then gives the corrected
label of any box from the aseg and the saved changes.

white_mesh makes the white mesh of a label: marching tetrahedra at 0.5 of
the binary label (so the mesh has the label's Kuhn topology, which
baillarger.handles makes handle-free), then tangential relaxation and
Gaussian smoothing without shrinkage (baillarger.mesh_smoothing), which move
vertices but keep the faces. white_mesh_normals adds the smoothed vertex
normals (out of the WM) that the column traces start from.

A box's white mesh can be made once and saved (WhiteMesh: box_mesh, save),
so that a series of experiments uses one surface. The file holds the box,
its affine, sigma, the SHA-256 of the WM label the mesh was made from and the
provenance of that label; WhiteMesh.check refuses a box or a WM label that
is not the mesh's own (baillarger.column_traces.trace_columns, mesh=).

The reports: changes_summary (the moves on the full grid), box_report (the
white mesh of a box before and after, per piece, and the moves in it) and
overlay (an ITK-SNAP label image of the moves). The command line:

    python -m baillarger.white changes --out OUTDIR
    python -m baillarger.white box --changes OUTDIR/wm_changes.npz --out OUTDIR
        --name NAME --lo X Y Z --hi X Y Z
    python -m baillarger.white mesh [--changes OUTDIR/wm_changes.npz]
        --out OUTDIR --name NAME (--patch P | --lo X Y Z --hi X Y Z)
        [--sigma S]

changes writes wm_changes.npz (the changes), wm_changes.json
(changes_summary) and wm_changes_overlay.nii.gz (full grid: 1 cut, 2 fill,
3 stranded). box writes wm_box_<NAME>.json (box_report),
wm_box_<NAME>_overlay.nii.gz and wm_box_<NAME>_wm_fixed.nii.gz (the box's
handle-free label), on the box grid. mesh writes wm_mesh_<NAME>.npz (the
WhiteMesh, the file that other tools load), wm_mesh_<NAME>.json (the box,
mesh_pieces and the provenance) and wm_mesh_<NAME>.vtk (the mesh in RAS mm,
for viewing); without --changes, the mesh is that of the aseg's WM label.
"""

import argparse
import datetime
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt

from baillarger import __version__
from baillarger import mesh_smoothing as S
from baillarger.data import ASEG_WM, PATCHES, Box, aseg_path, crop_affine
from baillarger.handles import kuhn_betti, remove_handles
from baillarger.mesh import marching_tetrahedra, topology, write_vtk

type Bool = npt.NDArray[np.bool_]
type Betti = tuple[int, int, int]
type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]

#: kinds of change: WM voxels removed (a cut), voxels added (a fill), WM voxels
#: removed that no front could claim (stranded: each would make a handle)
CUT, FILL, STRANDED = 1, 2, 3
KINDS = {"cut": CUT, "fill": FILL, "stranded": STRANDED}


@dataclass(frozen=True)
class WMChanges:
    """The changes that make the WM label handle-free."""

    #: (n, 3) int32, voxels of the full grid
    voxels: npt.NDArray[np.int32]
    #: (n,) int8, CUT, FILL or STRANDED
    kind: npt.NDArray[np.int8]
    #: (n,) int32, the move (one handle, or the stranded voxels) of each voxel
    move: npt.NDArray[np.int32]
    #: (b0, b1, b2) of the WM label before and after (Kuhn adjacency)
    betti_before: Betti
    betti_after: Betti

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            voxels=self.voxels,
            kind=self.kind,
            move=self.move,
            betti_before=np.array(self.betti_before),
            betti_after=np.array(self.betti_after),
        )

    @classmethod
    def load(cls, path: Path) -> "WMChanges":
        z = np.load(path)
        b0, b1 = z["betti_before"].tolist(), z["betti_after"].tolist()
        return cls(
            voxels=z["voxels"].astype(np.int32),
            kind=z["kind"].astype(np.int8),
            move=z["move"].astype(np.int32),
            betti_before=(b0[0], b0[1], b0[2]),
            betti_after=(b1[0], b1[1], b1[2]),
        )


def _aseg(directory: Path | None) -> nib.nifti1.Nifti1Image:
    img = nib.load(aseg_path(directory))
    if not isinstance(img, nib.nifti1.Nifti1Image):
        raise ValueError(f"{aseg_path(directory).name}: not a NIfTI-1 image")
    return img


def wm_changes(directory: Path | None = None) -> WMChanges:
    """Remove the handles of the aseg's WM label, on its bounding box with a
    margin of 2 voxels, and return the changes."""
    img = _aseg(directory)
    wm = np.asarray(img.dataobj) == ASEG_WM
    nz = [
        np.flatnonzero(wm.any(axis=tuple(j for j in range(3) if j != i)))
        for i in range(3)
    ]
    lo = np.array([max(int(t[0]) - 2, 0) for t in nz])
    hi = np.array([min(int(t[-1]) + 3, s) for t, s in zip(nz, wm.shape, strict=True)])
    x = wm[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    del wm
    before = kuhn_betti(x)
    r = remove_handles(x)
    after = kuhn_betti(r.obj)
    n = [len(m.voxels) for m in r.moves]
    if r.moves:
        voxels = np.concatenate([m.voxels for m in r.moves]) + lo
    else:
        voxels = np.zeros((0, 3), np.int64)
    kind = np.repeat([KINDS[m.kind] for m in r.moves], n)
    move = np.repeat(np.arange(len(r.moves)), n)
    return WMChanges(
        voxels.astype(np.int32),
        kind.astype(np.int8),
        move.astype(np.int32),
        before,
        after,
    )


def white_matter_box(
    lo: Box, hi: Box, changes: WMChanges, directory: Path | None = None
) -> Bool:
    """The handle-free WM label of the box [lo, hi) of the full grid (bool).
    Raises ValueError if the box is not inside the grid."""
    img = _aseg(directory)
    if not all(0 <= a < b <= n for a, b, n in zip(lo, hi, img.shape, strict=True)):
        raise ValueError(f"box {lo}-{hi} is not inside the grid {img.shape}")
    sl = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))
    wm: Bool = np.asarray(img.dataobj[sl]) == ASEG_WM
    p = changes.voxels.astype(np.int64) - np.asarray(lo)
    inside = np.all((p >= 0) & (p < np.subtract(hi, lo)), axis=1)
    p, k = p[inside], changes.kind[inside]
    wm[tuple(p[k == FILL].T)] = True
    wm[tuple(p[k != FILL].T)] = False
    return wm


# ---------------------------------------------------------------- the mesh


def white_mesh(wm: Bool, sigma: float = 2.0) -> tuple[Float, Int]:
    """The white mesh of a WM label: vertices (voxel coordinates) and faces
    oriented out of the WM. Marching tetrahedra at 0.5 of the binary label,
    then relax and twice (sigma in voxels)."""
    v0, f = marching_tetrahedra(np.asarray(wm, np.float64), 0.5)
    f = f[:, [0, 2, 1]].copy()  # marching_tetrahedra points into the WM
    if not len(f):
        return v0, f
    return S.twice(S.relax(v0, f), f, sigma), f


def white_mesh_normals(wm: Bool, sigma: float = 2.0) -> tuple[Float, Int, Float]:
    """white_mesh and its smoothed unit vertex normals, out of the WM
    (baillarger.mesh_smoothing.smooth_normals)."""
    v, f = white_mesh(wm, sigma)
    return v, f, S.smooth_normals(v, f)


def label_sha256(wm: Bool) -> str:
    """SHA-256 of a binary label: its shape and its voxels in C order."""
    a = np.ascontiguousarray(np.asarray(wm, bool))
    h = hashlib.sha256(repr(a.shape).encode())
    h.update(np.packbits(a, axis=None).tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class WhiteMesh:
    """The white mesh of one box, made once and loaded by the tools that use
    it."""

    #: (n, 3) vertices, voxel coordinates of the box
    v: Float
    #: (m, 3) faces, oriented out of the WM
    f: Int
    #: (n, 3) smoothed unit vertex normals, out of the WM
    normals: Float
    lo: Box
    hi: Box
    #: voxel to RAS (mm) of the box
    affine: Float
    sigma: float
    #: label_sha256 of the box's WM label that the mesh was made from
    wm_sha256: str
    #: where the WM label came from, and how and when the mesh was made
    provenance: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            v=self.v,
            f=self.f,
            normals=self.normals,
            lo=np.array(self.lo),
            hi=np.array(self.hi),
            affine=self.affine,
            sigma=np.array(self.sigma),
            wm_sha256=np.array(self.wm_sha256),
            provenance=np.array(json.dumps(self.provenance)),
        )

    @classmethod
    def load(cls, path: Path) -> "WhiteMesh":
        z = np.load(path)
        lo, hi = z["lo"].tolist(), z["hi"].tolist()
        return cls(
            v=z["v"].astype(np.float64),
            f=z["f"].astype(np.int64),
            normals=z["normals"].astype(np.float64),
            lo=(lo[0], lo[1], lo[2]),
            hi=(hi[0], hi[1], hi[2]),
            affine=z["affine"].astype(np.float64),
            sigma=float(z["sigma"]),
            wm_sha256=str(z["wm_sha256"]),
            provenance=json.loads(str(z["provenance"])),
        )

    def check(self, lo: Box, hi: Box, wm: Bool) -> None:
        """Raises ValueError unless [lo, hi) is the mesh's box and wm the WM
        label that it was made from."""
        if tuple(lo) != self.lo or tuple(hi) != self.hi:
            raise ValueError(
                f"box {tuple(lo)}-{tuple(hi)} is not the mesh's {self.lo}-{self.hi}"
            )
        if label_sha256(wm) != self.wm_sha256:
            raise ValueError("the WM label is not the one the mesh was made from")


def box_mesh(
    wm: Bool,
    lo: Box,
    hi: Box,
    affine: npt.ArrayLike,
    sigma: float = 2.0,
    provenance: dict[str, Any] | None = None,
) -> WhiteMesh:
    """The WhiteMesh of the WM label wm of the box [lo, hi)."""
    shape = tuple(int(b - a) for a, b in zip(lo, hi, strict=True))
    if wm.shape != shape:
        raise ValueError(f"WM label {wm.shape} is not on the box {shape}")
    v, f, n = white_mesh_normals(wm, sigma)
    return WhiteMesh(
        v,
        f,
        n,
        lo,
        hi,
        np.asarray(affine, np.float64),
        float(sigma),
        label_sha256(wm),
        dict(provenance or {}),
    )


def mesh_pieces(
    v: Float, f: Int, voxel_mm: tuple[float, float, float]
) -> list[dict[str, Any]]:
    """Per connected piece, largest first (by vertices): V, F, boundary loops,
    chi, genus and area in mm^2."""
    _, flab = S.pieces(f, len(v))
    vox = np.asarray(voxel_mm, float)
    out = []
    for p in range(int(flab.max()) + 1 if len(f) else 0):
        fp = f[flab == p]
        t = topology(fp, len(v))
        e1 = (v[fp[:, 1]] - v[fp[:, 0]]) * vox
        e2 = (v[fp[:, 2]] - v[fp[:, 0]]) * vox
        area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1).sum()
        out.append(
            dict(
                V=int(np.unique(fp).size),
                F=int(len(fp)),
                boundary_loops=int(t["boundary_loops"]),
                chi=int(t["chi"]),
                genus=int(t["genus"]),
                area_mm2=round(float(area), 3),
            )
        )
    return sorted(out, key=lambda t: -t["V"])


# ---------------------------------------------------------------- reports

_NAMES = {v: k for k, v in KINDS.items()}


def changes_summary(
    changes: WMChanges, voxel_mm: tuple[float, float, float]
) -> dict[str, Any]:
    """The Betti numbers, the counts by kind, and per move its kind, voxels,
    volume, extent and centre (full grid, 0-based, and ITK-SNAP, 1-based)."""
    vmm = float(np.prod(voxel_mm))
    moves = []
    for i in np.unique(changes.move).tolist():
        sel = changes.move == i
        p = changes.voxels[sel]
        c = np.round(p.mean(0)).astype(int)
        moves.append(
            dict(
                move=int(i),
                kind=_NAMES[int(changes.kind[sel][0])],
                voxels=int(len(p)),
                volume_mm3=round(len(p) * vmm, 4),
                extent_vox=(p.max(0) - p.min(0) + 1).tolist(),
                centre_full0=c.tolist(),
                centre_itksnap=(c + 1).tolist(),
            )
        )
    return dict(
        betti_before=list(changes.betti_before),
        betti_after=list(changes.betti_after),
        voxels_removed=int((changes.kind != FILL).sum()),
        voxels_added=int((changes.kind == FILL).sum()),
        stranded=int((changes.kind == STRANDED).sum()),
        by_kind={
            k: dict(
                moves=sum(m["kind"] == k for m in moves),
                voxels=sum(m["voxels"] for m in moves if m["kind"] == k),
            )
            for k in KINDS
        },
        moves=moves,
    )


def overlay(changes: WMChanges, lo: Box, hi: Box) -> npt.NDArray[np.int16]:
    """The kind (1 cut, 2 fill, 3 stranded) of each changed voxel of the box
    [lo, hi), 0 elsewhere."""
    p = changes.voxels.astype(np.int64) - np.asarray(lo)
    shape = tuple(int(b - a) for a, b in zip(lo, hi, strict=True))
    inside = np.all((p >= 0) & (p < np.array(shape)), axis=1)
    ov = np.zeros(shape, np.int16)
    ov[tuple(p[inside].T)] = changes.kind[inside]
    return ov


def box_report(
    lo: Box,
    hi: Box,
    changes: WMChanges,
    directory: Path | None = None,
    sigma: float = 2.0,
) -> dict[str, Any]:
    """The white mesh of the box [lo, hi) before and after the changes (per
    piece: mesh_pieces), the voxels removed and added, the total areas, and
    the moves in the box: kind, voxels in the box, centre (box, 0-based;
    ITK-SNAP on the box grid, 1-based; full grid, 0-based) and the centre's
    distance to the nearest box face in mm."""
    img = _aseg(directory)
    vox = tuple(float(z) for z in img.header.get_zooms()[:3])
    voxel_mm = (vox[0], vox[1], vox[2])
    sl = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))
    wm0: Bool = np.asarray(img.dataobj[sl]) == ASEG_WM
    wm1 = white_matter_box(lo, hi, changes, directory)
    before = mesh_pieces(*white_mesh(wm0, sigma), voxel_mm)
    after = mesh_pieces(*white_mesh(wm1, sigma), voxel_mm)
    p = changes.voxels.astype(np.int64) - np.asarray(lo)
    shape = np.array(wm0.shape)
    inside = np.all((p >= 0) & (p < shape), axis=1)
    moves = []
    for i in np.unique(changes.move[inside]).tolist():
        sel = inside & (changes.move == i)
        c = np.round(p[sel].mean(0)).astype(int)
        face = np.min(np.r_[c, shape - 1 - c] * np.r_[voxel_mm, voxel_mm])
        moves.append(
            dict(
                move=int(i),
                kind=_NAMES[int(changes.kind[sel][0])],
                voxels_in_box=int(sel.sum()),
                centre_box0=c.tolist(),
                centre_itksnap_box=(c + 1).tolist(),
                centre_full0=(c + np.asarray(lo)).tolist(),
                face_distance_mm=round(float(face), 2),
            )
        )
    return dict(
        lo=list(lo),
        hi=list(hi),
        voxels_removed=int((wm0 & ~wm1).sum()),
        voxels_added=int((wm1 & ~wm0).sum()),
        area_mm2_before=round(sum(t["area_mm2"] for t in before), 3),
        area_mm2_after=round(sum(t["area_mm2"] for t in after), 3),
        pieces_before=before,
        pieces_after=after,
        moves=moves,
    )


# ---------------------------------------------------------------- command line


def write_changes(changes: WMChanges, out: Path, stem: str = "wm_changes") -> None:
    """<stem>.npz, <stem>.json (changes_summary) and <stem>_overlay.nii.gz
    (full grid) in out."""
    out.mkdir(parents=True, exist_ok=True)
    img = _aseg(None)
    changes.save(out / f"{stem}.npz")
    vox = img.header.get_zooms()[:3]
    s = changes_summary(changes, (float(vox[0]), float(vox[1]), float(vox[2])))
    (out / f"{stem}.json").write_text(json.dumps(s, indent=1))
    ov = overlay(changes, (0, 0, 0), (img.shape[0], img.shape[1], img.shape[2]))
    nib.save(
        nib.nifti1.Nifti1Image(ov, img.affine, img.header),
        out / f"{stem}_overlay.nii.gz",
    )


def write_box(
    lo: Box, hi: Box, changes: WMChanges, out: Path, stem: str
) -> dict[str, Any]:
    """<stem>.json (box_report), <stem>_overlay.nii.gz and
    <stem>_wm_fixed.nii.gz on the box grid, in out. Returns the report."""
    out.mkdir(parents=True, exist_ok=True)
    img = _aseg(None)
    rep = box_report(lo, hi, changes)
    (out / f"{stem}.json").write_text(json.dumps(rep, indent=1))
    aff = crop_affine(img.affine, lo)
    nib.save(
        nib.nifti1.Nifti1Image(overlay(changes, lo, hi), aff),
        out / f"{stem}_overlay.nii.gz",
    )
    wm = white_matter_box(lo, hi, changes).astype(np.int16)
    nib.save(nib.nifti1.Nifti1Image(wm, aff), out / f"{stem}_wm_fixed.nii.gz")
    return rep


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_state() -> dict[str, Any]:
    """The commit of the checkout this module is in, and whether tracked files
    differ from it; None for both outside a git checkout."""
    repo = Path(__file__).resolve().parent

    def git(*args: str) -> str | None:
        r = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True
        )
        return r.stdout.strip() if r.returncode == 0 else None

    status = git("status", "--porcelain", "--untracked-files=no")
    return dict(
        commit=git("rev-parse", "HEAD"),
        dirty=None if status is None else bool(status),
    )


def write_mesh(
    lo: Box,
    hi: Box,
    changes_path: Path | None,
    out: Path,
    stem: str,
    sigma: float = 2.0,
) -> WhiteMesh:
    """<stem>.npz (the WhiteMesh), <stem>.json (the box, mesh_pieces and the
    provenance) and <stem>.vtk (RAS mm) in out, for the WM label of the box
    [lo, hi): the handle-free label of the changes at changes_path, or the
    aseg's without. Returns the mesh."""
    img = _aseg(None)
    if changes_path is None:
        if not all(0 <= a < b <= n for a, b, n in zip(lo, hi, img.shape, strict=True)):
            raise ValueError(f"box {lo}-{hi} is not inside the grid {img.shape}")
        sl = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))
        wm: Bool = np.asarray(img.dataobj[sl]) == ASEG_WM
    else:
        wm = white_matter_box(lo, hi, WMChanges.load(changes_path))
    prov: dict[str, Any] = dict(
        created=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        wm_label="aseg" if changes_path is None else "handle-free",
        aseg=dict(path=str(aseg_path().resolve()), sha256=_file_sha256(aseg_path())),
        changes=None
        if changes_path is None
        else dict(path=str(changes_path.resolve()), sha256=_file_sha256(changes_path)),
        process="baillarger.white.white_mesh_normals",
        baillarger_version=__version__,
        **_git_state(),
    )
    m = box_mesh(wm, lo, hi, crop_affine(img.affine, lo), sigma, prov)
    out.mkdir(parents=True, exist_ok=True)
    m.save(out / f"{stem}.npz")
    vox = img.header.get_zooms()[:3]
    rep = dict(
        lo=list(lo),
        hi=list(hi),
        sigma=m.sigma,
        vertices=len(m.v),
        faces=len(m.f),
        wm_sha256=m.wm_sha256,
        npz_sha256=_file_sha256(out / f"{stem}.npz"),
        pieces=mesh_pieces(m.v, m.f, (float(vox[0]), float(vox[1]), float(vox[2]))),
        provenance=prov,
    )
    (out / f"{stem}.json").write_text(json.dumps(rep, indent=1))
    _, flab = S.pieces(m.f, len(m.v))
    piece = np.full(len(m.v), -1.0)
    piece[m.f.ravel()] = np.repeat(flab, 3)
    write_vtk(str(out / f"{stem}.vtk"), m.affine, m.v, m.f, {"piece": piece})
    return m


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("changes", help="remove the handles of the WM label")
    c.add_argument("--out", required=True, type=Path)
    b = sub.add_parser("box", help="report one box before and after")
    b.add_argument("--changes", required=True, type=Path)
    b.add_argument("--out", required=True, type=Path)
    b.add_argument("--name", required=True)
    b.add_argument("--lo", required=True, type=int, nargs=3)
    b.add_argument("--hi", required=True, type=int, nargs=3)
    m = sub.add_parser("mesh", help="save the white mesh of one box")
    m.add_argument("--changes", type=Path, help="default: the aseg's WM label")
    m.add_argument("--out", required=True, type=Path)
    m.add_argument("--name", required=True)
    m.add_argument("--patch", choices=sorted(PATCHES))
    m.add_argument("--lo", type=int, nargs=3)
    m.add_argument("--hi", type=int, nargs=3)
    m.add_argument("--sigma", type=float, default=2.0)
    a = ap.parse_args(argv)
    if a.cmd == "mesh":
        if a.patch is not None and a.lo is None and a.hi is None:
            lo, hi = PATCHES[a.patch]
        elif a.patch is None and a.lo is not None and a.hi is not None:
            lo, hi = (a.lo[0], a.lo[1], a.lo[2]), (a.hi[0], a.hi[1], a.hi[2])
        else:
            ap.error("mesh: give --patch, or --lo and --hi")
        wmesh = write_mesh(lo, hi, a.changes, a.out, f"wm_mesh_{a.name}", a.sigma)
        print(f"{a.name}: {len(wmesh.v)} vertices, {len(wmesh.f)} faces")
    elif a.cmd == "changes":
        ch = wm_changes()
        write_changes(ch, a.out)
        b0, b1 = ch.betti_before, ch.betti_after
        print(f"(b0, b1, b2) {b0} -> {b1}, {len(ch.voxels)} voxels")
    else:
        lo, hi = (a.lo[0], a.lo[1], a.lo[2]), (a.hi[0], a.hi[1], a.hi[2])
        rep = write_box(lo, hi, WMChanges.load(a.changes), a.out, f"wm_box_{a.name}")
        g = [p["genus"] for p in rep["pieces_after"]]
        n0, n1 = rep["voxels_removed"], rep["voxels_added"]
        print(f"{a.name}: removed {n0}, added {n1}, genus {g}")


if __name__ == "__main__":
    main()
