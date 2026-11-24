"""baillarger.crop on a real crop: site 1 of the first hemisphere, 10 mm.

The tests read the whole-hemisphere derivatives in the BIDS derivatives root
that $BAILLARGER_DERIVATIVES names (the dseg, its label edits and the white
surface) and the echoes in baillarger.data.data_dir(). They skip when either
is not there (CI has no data). The real-data checks are in two tests, so that
parallel workers do not each read the crop.
"""

import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from baillarger import crop as C
from baillarger import data, label_edits
from baillarger.bids import read_sidecar, sidecar_path
from baillarger.mesh import marching_tetrahedra_blocks
from baillarger.white_overview import WhiteOverview
from baillarger.white_surf import SURF_SPEC, WhiteSurface, mm_to_voxel

SES = "sub-INDD121850_ses-7THemix20250516"
#: site 1 (the site of the tube and global τ prototypes), scanner mm
SITE_MM = (50.96, -28.391, -30.463)
SIZE_MM = 10.0


def _img(path: Path) -> nib.nifti1.Nifti1Image:
    img = nib.load(path)
    assert isinstance(img, nib.nifti1.Nifti1Image)
    return img


def _root() -> Path | None:
    env = os.environ.get("BAILLARGER_DERIVATIVES")
    return (
        Path(env) / "sub-INDD121850" / "ses-7THemix20250516" / "anat" if env else None
    )


def _paths() -> dict[str, Path] | None:
    anat = _root()
    if anat is None or not data.data_available():
        return None
    p = dict(
        dseg=anat / f"{SES}_desc-wmtopo_dseg.nii.gz",
        edits=anat / f"{SES}_desc-wmtopo_edits.tsv",
        white=anat / f"{SES}_hemi-R_white.surf.gii",
    )
    return p if all(q.is_file() for q in p.values()) else None


PATHS = _paths()
need_data = pytest.mark.skipif(
    PATHS is None, reason="no $BAILLARGER_DERIVATIVES or no example data"
)
ECHOES = [data.echo_path(k) for k in (1, 2, 3)]


def _box() -> C.Box:
    assert PATHS is not None
    img = _img(PATHS["dseg"])
    return C.box_at_mm(SITE_MM, SIZE_MM, img.affine, img.shape[:3])


@need_data
def test_box_at_mm_is_odd_round_the_nearest_voxel() -> None:
    assert PATHS is not None
    img = _img(PATHS["dseg"])
    box = _box()
    vox = C.voxel_size(img.affine)
    half = np.ceil(SIZE_MM / 2 / vox).astype(int)
    assert box.shape == tuple(2 * half + 1)
    centre = np.rint(np.linalg.inv(img.affine) @ np.r_[SITE_MM, 1])[:3]
    assert np.array_equal((np.array(box.lo) + np.array(box.hi) - 1) / 2, centre)
    assert box.request["Method"] == "centre-mm"


@need_data
def test_a_box_past_the_grid_is_an_error() -> None:
    assert PATHS is not None
    img = _img(PATHS["dseg"])
    corner = img.affine @ np.r_[2, 2, 2, 1]
    with pytest.raises(ValueError, match="not inside the grid"):
        C.box_at_mm(corner[:3], SIZE_MM, img.affine, img.shape[:3])
    with pytest.raises(ValueError, match="not inside the grid"):
        C.voxel_box((0, 0, 0), (1, 1, img.shape[2] + 1), img.shape[:3])
    with pytest.raises(ValueError, match="not inside the grid"):
        C.voxel_box((5, 5, 5), (5, 6, 6), img.shape[:3])


@need_data
def test_box_at_vertex_is_box_at_its_point() -> None:
    assert PATHS is not None
    side = read_sidecar(sidecar_path(PATHS["white"]), SURF_SPEC)
    row = np.loadtxt(PATHS["white"].with_name(side["BlocksFile"]), skiprows=1)[0]
    W = WhiteSurface.load(PATHS["white"], block=(int(row[0]), int(row[1]), int(row[2])))
    vertex = int(W.vertex_ids[0])
    box = C.box_at_vertex(PATHS["white"], vertex, SIZE_MM)
    img = _img(PATHS["dseg"])
    mm = img.affine[:3, :3] @ W.v[0] + img.affine[:3, 3]
    want = C.box_at_mm(mm, SIZE_MM, img.affine, img.shape[:3])
    assert (box.lo, box.hi) == (want.lo, want.hi)
    assert box.request["Vertex"] == vertex


@need_data
def test_the_crop_in_memory() -> None:
    assert PATHS is not None
    box = _box()
    c = C.crop(box, PATHS["dseg"], ECHOES, (), PATHS["white"], PATHS["edits"])

    # volumes: the box of each source, in its data type; the affine moves
    # voxel (0, 0, 0) to voxel lo
    dimg = _img(PATHS["dseg"])
    want = np.asarray(dimg.dataobj[box.slices])
    assert c.dseg.dtype == want.dtype and np.array_equal(c.dseg, want)
    for p in ECHOES:
        assert np.array_equal(c.volumes[p], np.asarray(_img(p).dataobj[box.slices]))
    assert c.echo_stack().shape == (3, *box.shape)
    assert np.allclose(c.affine @ np.r_[0, 0, 0, 1], dimg.affine @ np.r_[box.lo, 1])

    # the white mesh is the march of the cropped field: the same vertices
    # (edge keys, in the same order), the same block index, the same faces in
    # each block, and the positions within float32 rounding
    w = c.white
    assert w is not None and c.field is not None
    cells = w.blocks.cells
    v, f, mb = marching_tetrahedra_blocks(c.field, 0.0, cells, np.float32)
    f = f[:, [0, 2, 1]]  # the surface's orientation: out of the WM
    assert len(w.f) > 1000
    assert np.array_equal(w.keys, mb.vertex_key)
    for name in ("origin", "vertex_start", "vertex_stop", "face_start", "face_stop"):
        assert np.array_equal(getattr(w.blocks, name), getattr(mb, name)), name
    for s, e in zip(mb.face_start, mb.face_stop, strict=True):
        got = {tuple(np.roll(t, -int(np.argmin(t)))) for t in w.keys[w.f[s:e]]}
        exp = {tuple(np.roll(t, -int(np.argmin(t)))) for t in mb.vertex_key[f[s:e]]}
        assert got == exp
    vv = mm_to_voxel(w.v_mm, c.affine)
    assert np.abs(vv - v).max() < 1e-3

    # the whole surface's values at the kept vertices, bit for bit
    full = WhiteSurface.load(PATHS["white"])
    assert np.array_equal(w.normals, full.normals[w.source_ids])
    assert np.array_equal(w.normlen, full.normlen[w.source_ids])
    assert np.allclose(full.v[w.source_ids] - np.array(box.lo), vv, atol=1e-3)

    # the label edits: the entries inside the box, on the crop's grid
    e = label_edits.read(PATHS["edits"])
    ijk = np.stack(np.unravel_index(e.index, dimg.shape), 1)
    ins = np.all((ijk >= box.lo) & (ijk < box.hi), axis=1)
    assert c.edits is not None and len(c.edits) == int(ins.sum()) > 0
    back = np.stack(np.unravel_index(c.edits.index, box.shape), 1) + box.lo
    assert np.array_equal(back, ijk[ins])
    assert np.array_equal(c.edits.label, e.label[ins])
    c.edits.check(box.shape)


@need_data
def test_the_written_crop(tmp_path: Path) -> None:
    assert PATHS is not None
    box = _box()
    c = C.crop(box, PATHS["dseg"], ECHOES, (), PATHS["white"], PATHS["edits"])
    out = C.write_crop(c, tmp_path)
    side = read_sidecar(out["crop"], C.CROP_SPEC)
    assert side["BoxLo"] == list(box.lo) and side["GridShape"] == list(box.shape)
    assert set(side["OutputsSHA256"]) == {p.name for r, p in out.items() if r != "crop"}

    # each output has the name of its source
    assert out["dseg"].name == PATHS["dseg"].name
    assert out["echo2"].name == ECHOES[1].name
    assert out["white"].name == PATHS["white"].name
    assert out["edits"].name == PATHS["edits"].name

    # readers of the whole-hemisphere formats read the crop
    W = WhiteSurface.load(out["white"])
    W.check(out["dseg"])
    assert c.white is not None and np.array_equal(W.vertex_key, c.white.keys)
    ov = WhiteOverview.load(out["overview.overview"])
    ov.check(W)
    assert len(ov.vertex_ids) and ov.vertex_ids.max() < len(W.v)
    e = label_edits.read(out["edits"], shape=box.shape, affine=c.affine)
    assert c.edits is not None and np.array_equal(e.index, c.edits.index)
    img = _img(out["echo1"])
    assert np.allclose(img.affine, c.affine)
    assert np.array_equal(np.asarray(img.dataobj), c.volumes[ECHOES[0]])

    # load_crop gives the crop back
    back = C.load_crop(out["crop"])
    assert back.box == box
    assert np.array_equal(back.dseg, c.dseg)
    assert np.array_equal(back.echo_stack(), c.echo_stack())
    assert back.white is not None and c.white is not None
    assert np.array_equal(back.white.f, c.white.f)
    assert np.array_equal(back.white.v_mm, c.white.v_mm)
    assert back.field is not None and c.field is not None
    assert np.array_equal(back.field, c.field)
