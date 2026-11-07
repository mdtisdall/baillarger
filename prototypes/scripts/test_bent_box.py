"""Tests of bent_box, with exact answers, on a synthetic right hemisphere that
white_dseg, white_surf and white_overview write (as in
tests/test_viewer_columns.py): the columns are the ray sampler's own, byte
for byte; the echoes in the box give the column-volume sampler's values; the
faces and edges join chosen roots; the labels are the dseg's.

Run from the repository root:
    nix develop --command uv run pytest -q prototypes/scripts/test_bent_box.py
"""

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).parent))
import bent_box as BB  # noqa: E402
import mesh_smoothing as MS  # noqa: E402

from baillarger import rays as R  # noqa: E402
from baillarger import white_dseg, white_overview, white_surf  # noqa: E402
from baillarger.blocks import open_volume  # noqa: E402
from baillarger.data import echo_path  # noqa: E402
from baillarger.viewer import columns as VC  # noqa: E402
from baillarger.white_surf import WhiteSurface  # noqa: E402

AFF = np.array([[0.2, 0, 0, -3.0], [0, 0.16, 0, 2.0], [0, 0, 0.2, 5.0], [0, 0, 0, 1.0]])
CELLS = 8
RADIUS = 1.0
HALO = 0.5


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A right hemisphere: a WM ball round a ventricle, in cortex, in a grid
    of 8-voxel blocks (1.6 mm), with its overview and three echoes (echo k is
    k times the linear index of each voxel). A copy of the fixture of
    tests/test_viewer_columns.py."""
    d = tmp_path_factory.mktemp("box")
    shape = (44, 40, 36)
    x = np.indices(shape).astype(float)
    r = np.sqrt(((x - np.array([20.3, 19.6, 17.8])[:, None, None, None]) ** 2).sum(0))
    a = np.zeros(shape, np.int16)
    a[r < 15] = 42
    a[r < 12] = 41
    a[r < 4] = 43
    src = d / "in" / "sub-T01_ses-S1_aseg.nii.gz"
    src.parent.mkdir(parents=True)
    nib.save(nib.nifti1.Nifti1Image(a, AFF), src)
    white_dseg.main([str(src), "--out", str(d / "deriv")])
    anat = d / "deriv/sub-T01/ses-S1/anat"
    dseg = anat / "sub-T01_ses-S1_desc-wmtopo_dseg.nii.gz"
    white_surf.main([str(dseg), "--out", str(d / "deriv"), "--block-cells", str(CELLS)])
    surf = anat / "sub-T01_ses-S1_hemi-R_white.surf.gii"
    white_overview.main([str(surf), "--eps", "0.05", "--workers", "1"])
    ed = d / "echoes"
    ed.mkdir()
    for k in (1, 2, 3):
        e = (k * np.arange(int(np.prod(shape)))).reshape(shape).astype(np.float32)
        nib.save(nib.nifti1.Nifti1Image(e, AFF), echo_path(k, ed).with_suffix(""))
    return dict(surf=surf, dseg=dseg, echoes=ed, out=d / "out")


def quiet(m: str) -> None:
    pass


def echoes(data: dict[str, Path]) -> list[Path]:
    return VC.echo_files(data["echoes"])


@pytest.fixture(scope="module")
def cols(data: dict[str, Path]) -> VC.Columns:
    W = WhiteSurface.load(data["surf"])
    return VC.Columns(data["surf"], data["dseg"], W.v, W.f, echoes(data), log=quiet)


def vertex_of(cols: VC.Columns) -> int:
    """A vertex near a corner of its block (in voxels), so that the box
    takes roots from more than one superblock."""
    v = np.asarray(cols.v, float)
    f = v / CELLS - np.floor(v / CELLS)
    return int(np.argmin(np.linalg.norm(np.minimum(f, 1 - f), axis=1)))


@pytest.fixture(scope="module")
def box(data: dict[str, Path], cols: VC.Columns) -> BB.Box:
    return BB.box(
        data["surf"], data["dseg"], echoes(data), vertex_of(cols), RADIUS, HALO
    )


def as_column(r: R.Rays, c: int) -> tuple[bytes, ...]:
    """Every value of column c of r, as bytes (tests/test_rays.py)."""
    a, b = r.sample_start[c], r.sample_start[c + 1]
    q0, q1 = r.interval_start[c], r.interval_start[c + 1]
    ro = r.roots
    return (
        ro.edge_key[c].tobytes(),
        ro.vertex[c].tobytes(),
        ro.owned[c].tobytes(),
        ro.root[c].tobytes(),
        ro.normal[c].tobytes(),
        ro.normal_length[c].tobytes(),
        r.k_in[c].tobytes(),
        r.k_out[c].tobytes(),
        r.s[a:b].tobytes(),
        r.position[a:b].tobytes(),
        r.interval_s0[q0:q1].tobytes(),
        r.interval_s1[q0:q1].tobytes(),
        r.interval_label[q0:q1].tobytes(),
        r.interval_class[q0:q1].tobytes(),
        r.derived_class[c].tobytes(),
    )


def test_the_inner_columns_are_the_viewers_byte_for_byte(
    box: BB.Box, cols: VC.Columns
) -> None:
    ref = cols.compute(VC.Request(vertex=box.vertex, radius=RADIUS), quiet)
    want = ref.rays
    inner = np.flatnonzero(box.inner)
    assert len(inner) == len(want.roots.vertex) > 20
    got = VC.select_columns(box.rays, inner)
    for c in range(len(inner)):
        assert as_column(got, c) == as_column(want, c)
    assert np.array_equal(box.distance[inner].astype(np.float32), ref.distance)
    # every column has the full extents
    assert np.all(box.rays.k_in == round(R.CAP_IN / R.STEP))
    assert np.all(box.rays.k_out == round(R.CAP_OUT / R.STEP))
    # the halo columns lie beyond the radius, within radius + halo
    d = box.distance[~box.inner]
    assert len(d) > 0 and np.all((d > RADIUS) & (d <= RADIUS + HALO))
    assert np.array_equal(box.roots, box.rays.roots.root)
    assert np.array_equal(box.normals, box.rays.roots.normal)


def test_without_halo_the_box_is_the_viewers_request(
    data: dict[str, Path], cols: VC.Columns, box: BB.Box
) -> None:
    b0 = BB.box(data["surf"], data["dseg"], echoes(data), box.vertex, RADIUS, 0.0)
    want = cols.compute(VC.Request(vertex=box.vertex, radius=RADIUS), quiet).rays
    assert np.all(b0.inner)
    assert len(b0.rays.roots.vertex) == len(want.roots.vertex)
    for c in range(len(want.roots.vertex)):
        assert as_column(b0.rays, c) == as_column(want, c)


def test_the_columns_are_in_the_samplers_order(box: BB.Box, cols: VC.Columns) -> None:
    rs = cols.rays
    sampler = rs.sample(
        box.superblock, 0, e_in=R.CAP_IN, e_out=R.CAP_OUT, halo_mm=RADIUS + HALO
    )
    pos = {int(u): k for k, u in enumerate(sampler.roots.vertex)}
    idx = [pos[int(u)] for u in box.rays.roots.vertex]
    assert np.all(np.diff(idx) > 0)
    for c, k in enumerate(idx):
        assert as_column(box.rays, c) == as_column(sampler, k)


def test_the_echoes_in_the_box_give_the_samplers_values(
    box: BB.Box, data: dict[str, Path], cols: VC.Columns
) -> None:
    assert cols.volume is not None
    cv = cols.volume.sample(box.rays, "numpy")
    v = box.to_box(box.rays.position)
    ok = cv.valid
    assert ok.sum() > 1000
    # every valid sample lies inside the box
    top = np.asarray(box.labels.shape) - 1
    assert np.all((v[ok] >= 0) & (v[ok] <= top))
    for e in range(3):
        got = map_coordinates(box.echoes[e], v[ok].T, order=1)
        want = cv.echo[ok, e]
        scale = float(np.abs(want).max())
        assert np.allclose(got, want, rtol=0, atol=4 * scale * 2.0**-24)


def test_the_box_holds_the_dseg_and_the_echoes(
    box: BB.Box, data: dict[str, Path]
) -> None:
    sl = tuple(
        slice(int(o), int(o) + n)
        for o, n in zip(box.offset, box.labels.shape, strict=True)
    )
    dseg = np.asarray(nib.load(data["dseg"]).dataobj)
    assert box.labels.dtype == np.int32
    assert np.array_equal(box.labels, dseg[sl].astype(np.int32))
    assert box.echoes.dtype == np.float32 and box.echoes.shape[0] == 3
    for e, p in enumerate(echoes(data)):
        img = np.asarray(nib.load(p).dataobj, np.float32)
        assert np.array_equal(box.echoes[e], img[sl])
    rs = R.RaySampler(data["surf"], data["dseg"])
    assert np.array_equal(box.affine, rs.affine)
    assert np.array_equal(box.inverse, rs.inverse)
    assert np.array_equal(box.voxel_mm, rs.voxel_mm)
    assert np.array_equal(box.filled, rs.filled)
    assert box.cortex == rs.cortex


def test_the_box_holds_each_straight_ray_with_the_margin(
    box: BB.Box, data: dict[str, Path]
) -> None:
    """The ends of each ray at the full caps, widened by the margin on each
    axis, lie in the box, or the box reaches the grid's edge there."""
    shape = np.asarray(open_volume(data["dseg"]).shape)
    m = BB.MARGIN_MM / box.voxel_mm
    lo = np.zeros(3)
    hi = np.asarray(box.labels.shape) - 1.0
    for s in (-R.CAP_IN, R.CAP_OUT):
        end = box.roots.astype(np.float64) + s * box.normals.astype(np.float64)
        v = box.to_box(end)
        at_lo = box.offset == 0
        at_hi = box.offset + box.labels.shape == shape
        assert np.all((v - m >= lo) | at_lo)
        assert np.all((v + m <= hi) | at_hi)


def test_to_box_is_the_inverse_affine_less_the_offset(box: BB.Box) -> None:
    x = box.rays.position[:50].astype(np.float64)
    h = np.c_[x, np.ones(len(x))]
    want = (box.inverse @ h.T).T[:, :3] - box.offset
    assert np.allclose(box.to_box(x), want, rtol=0, atol=1e-9)
    assert np.allclose((box.mm_to_box @ h.T).T[:, :3], want, rtol=0, atol=1e-9)


def test_faces_and_edges_join_chosen_roots(box: BB.Box, data: dict[str, Path]) -> None:
    n = len(box.rays.roots.vertex)
    assert len(box.faces) > 0
    assert box.faces.min() >= 0 and box.faces.max() < n
    assert np.array_equal(box.edges, MS.edges(box.faces))
    assert box.edges.min() >= 0 and box.edges.max() < n
    assert np.all(box.edges[:, 0] != box.edges[:, 1])
    # the faces are those of the whole mesh with three chosen corners
    W = WhiteSurface.load(data["surf"])
    f = np.asarray(W.f, np.int64)
    vert = box.rays.roots.vertex
    keep = np.all(np.isin(f, vert), axis=1)
    want = {tuple(sorted(t)) for t in f[keep].tolist()}
    got = {tuple(sorted(t)) for t in vert[box.faces].tolist()}
    assert got == want
    assert len(box.faces) == len(want)
    # with the halo, every face round an inner root is in the box
    inner = vert[box.inner]
    ring = f[np.any(np.isin(f, inner), axis=1)]
    assert {tuple(sorted(t)) for t in ring.tolist()} <= got


def test_save_and_load_keep_every_value(box: BB.Box, tmp_path: Path) -> None:
    p = tmp_path / "box.npz"
    BB.save(box, p)
    b = BB.load(p)
    for c in range(len(box.rays.roots.vertex)):
        assert as_column(b.rays, c) == as_column(box.rays, c)
    assert b.rays.method == box.rays.method and b.rays.step == box.rays.step
    for name in (
        "distance",
        "inner",
        "faces",
        "edges",
        "echoes",
        "labels",
        "offset",
        "affine",
        "inverse",
        "voxel_mm",
        "filled",
    ):
        a, c = getattr(box, name), getattr(b, name)
        assert a.dtype == c.dtype and a.shape == c.shape
        assert a.tobytes() == c.tobytes()
    for name in ("vertex", "radius_mm", "halo_mm", "cortex", "superblock"):
        assert getattr(box, name) == getattr(b, name)


def test_the_command_line(data: dict[str, Path], cols: VC.Columns) -> None:
    out = data["out"]
    BB.main(
        [
            "--white",
            str(data["surf"]),
            "--dseg",
            str(data["dseg"]),
            "--echoes",
            *map(str, echoes(data)),
            "--vertex",
            str(vertex_of(cols)),
            "--radius",
            str(RADIUS),
            "--halo",
            str(HALO),
            "--out",
            str(out),
        ]
    )
    b = BB.load(out / BB.NPZ)
    side = json.loads((out / BB.SIDECAR).read_text())
    assert side["vertex"] == vertex_of(cols)
    assert side["radius_mm"] == RADIUS and side["halo_mm"] == HALO
    assert side["columns"] == len(b.rays.roots.vertex)
    assert side["box_shape"] == list(b.labels.shape)


def test_echoes_on_another_grid_are_refused(
    data: dict[str, Path], tmp_path: Path
) -> None:
    bad = tmp_path / "bad.nii"
    nib.save(nib.nifti1.Nifti1Image(np.zeros((4, 4, 4), np.float32), AFF), bad)
    with pytest.raises(ValueError):
        BB.box(data["surf"], data["dseg"], [bad], 0, RADIUS, 0.0)


def test_painted_crops_change_only_the_labels(
    data: dict[str, Path], cols: VC.Columns, box: BB.Box, tmp_path: Path
) -> None:
    """A crop exported by wm_paint, with some labels changed: the box's
    labels change at those voxels only (the ones in the box), the rays and
    the echoes do not."""
    from baillarger import wm_paint

    v = vertex_of(cols)
    site = np.rint(box.to_box(box.roots[0]) + box.offset).astype(int)
    crop = tmp_path / "crop"
    wm_paint.export(data["dseg"], tuple(int(x) for x in site), crop, half=4)
    img = nib.load(crop / wm_paint.LABELS)
    lab = np.asarray(img.dataobj).copy()
    lab[2:5, 3, 4] = 0
    lab[0, 0, 0] = 99
    nib.save(nib.Nifti1Image(lab, img.affine, img.header), crop / wm_paint.LABELS)
    pts, vals = wm_paint.changes(crop)
    b2 = BB.box(data["surf"], data["dseg"], echoes(data), v, 1.0, 0.5, crops=[crop])
    q = pts - b2.offset
    ok = np.all((q >= 0) & (q < np.asarray(b2.labels.shape)), axis=1)
    want = box.labels.copy()
    want[tuple(q[ok].T)] = vals[ok]
    assert np.array_equal(b2.labels, want)
    assert int(np.count_nonzero(b2.labels != box.labels)) >= 1
    assert np.array_equal(b2.echoes, box.echoes)
    assert np.array_equal(b2.rays.position, box.rays.position)
    assert np.array_equal(b2.rays.interval_label, box.rays.interval_label)
