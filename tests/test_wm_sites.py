"""Tests for baillarger.wm_sites: misfacing faces, grouping, the per-site
measures, the queue order, and the files through the sites command (wm_sites.main)."""

from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest

from baillarger import white_dseg, white_surf
from baillarger import wm_sites as WS

type Float = npt.NDArray[np.float64]
type Int = npt.NDArray[np.int64]


def sheet(n: int = 12) -> tuple[Float, Int]:
    """A flat grid of n x n vertices (spacing 1) at z = 5, faces facing +z."""
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    v = np.c_[i.ravel() + 2.0, j.ravel() + 2.0, np.full(n * n, 5.0)]
    F = []
    for a in range(n - 1):
        for b in range(n - 1):
            p, q, r, s = a * n + b, (a + 1) * n + b, (a + 1) * n + b + 1, a * n + b + 1
            F += [(p, q, r), (p, r, s)]
    return v, np.array(F, np.int64)


def fold(v: Float, f: Int, at: tuple[int, int], n: int = 12) -> list[int]:
    """Push the vertex at grid (a, b) past its neighbour at (a + 1, b + 1), so
    the faces between them turn over. Returns the faces that turned over."""
    k, o = at[0] * n + at[1], (at[0] + 1) * n + at[1] + 1
    before = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])[:, 2]
    v[k] = v[o] + [0.6, 0.6, 0.0]
    after = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])[:, 2]
    return np.flatnonzero((before > 0) & (after < 0)).tolist()


def test_misfacing_finds_exactly_the_turned_over_faces() -> None:
    v, f = sheet()
    assert not WS.misfacing(v, f).any()
    turned = fold(v, f, (4, 4))
    assert turned
    got = np.flatnonzero(WS.misfacing(v, f)).tolist()
    assert set(got) <= set(turned) | {
        i for i in range(len(f)) if set(f[i]) & set(np.unique(f[turned]))
    }
    assert set(turned) & set(got)


def test_grouping_links_points_within_the_radius() -> None:
    p = np.array([[0, 0, 0], [2.9, 0, 0], [5.8, 0, 0], [20, 0, 0], [20, 3.0, 0]], float)
    lab = WS.group(p, 3.0)
    assert lab[0] == lab[1] == lab[2] != lab[3]
    assert lab[3] == lab[4]  # exactly at the radius: linked
    assert len(WS.group(np.zeros((0, 3)))) == 0


def test_tissue_names() -> None:
    assert WS.tissue([42, 42, 3]) == "grey matter"
    assert WS.tissue([0, 24]) == "fluid"
    # the ventricles and the choroid plexus are outside the mask: fluid
    for lab in (4, 5, 31, 43, 44, 63):
        assert WS.tissue([lab]) == "fluid"
    assert WS.tissue([42, 43]) == "mixed"
    assert WS.tissue([53]) == "other"
    assert WS.tissue([42, 0]) == "mixed"
    assert WS.tissue([17, 53]) == "other"


def test_measures_on_built_volumes() -> None:
    shape = (31, 31, 31)
    c = np.array([15, 15, 15])
    # a strand one voxel wide in cortex
    seg = np.full(shape, 42, np.int16)
    seg[15, 15, 5:26] = 41
    mask = seg == 41
    m = WS.measure(seg, mask, np.zeros(shape, np.uint8), c)
    assert m["tissue"] == "grey matter" and m["wm_thickness"] == 1.0
    assert (m["wm_face_pieces"], m["wm_any_pieces"]) == (1, 1)
    # two strands touching only along an edge
    seg[16, 16, 5:26] = 41
    seg[15, 15, 5:26] = 42
    seg[15, 15, 5:15] = 41
    seg[16, 16, 16:26] = 42
    mask = seg == 41
    m = WS.measure(seg, mask, np.zeros(shape, np.uint8), c)
    assert m["wm_face_pieces"] > m["wm_any_pieces"]
    # a gap one voxel wide between two WM slabs, in fluid
    seg = np.full(shape, 41, np.int16)
    seg[15, :, :] = 0
    mask = seg == 41
    m = WS.measure(seg, mask, np.zeros(shape, np.uint8), c)
    assert m["tissue"] == "fluid" and m["gap_width"] == 1.0
    # fixes: counted by kind within FIX voxels (a cube), not beyond
    fx = np.zeros(shape, np.uint8)
    fx[15 + WS.FIX, 15, 15] = 1
    fx[15, 15 - WS.FIX, 15] = 2
    fx[15, 15, 15 + WS.FIX + 1] = 3
    m = WS.measure(seg, mask, fx, c)
    assert (m["cut"], m["fill"], m["cavity"]) == (1, 1, 0)


def test_find_sites_groups_folds_and_orders_the_queue() -> None:
    """Two folds far apart on a sheet, one in grey matter and one in fluid:
    two sites, the grey-matter one first, each at its fold."""
    v, f = sheet(16)
    t1 = fold(v, f, (3, 3), 16)
    t2 = fold(v, f, (11, 11), 16)
    shape = (24, 24, 12)
    seg = np.zeros(shape, np.int16)
    seg[:, :, :5] = 41  # WM below the sheet
    seg[:11, :11, 5:] = 42  # cortex above, round the first fold only
    S = WS.find_sites(v, f, v * 0.16, seg, (41,), np.zeros(shape, np.uint8))
    assert S.misfacing_faces >= len(t1) + len(t2) - 2
    assert len(S.rows) == 2
    first, second = S.rows
    assert first["rank"] == 1 and first["tissue"] == "grey matter"
    assert second["tissue"] == "fluid"
    for row, at in ((first, (3, 3)), (second, (11, 11))):
        assert abs(row["i"] - (at[0] + 2.5)) <= 2 and abs(row["j"] - (at[1] + 2.5)) <= 2


@pytest.fixture(scope="module")
def surf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A white surface through the dseg and surface commands."""
    d = tmp_path_factory.mktemp("sites")
    x = np.indices((34, 34, 30)).astype(float)
    r = np.sqrt(((x - np.array([16.6, 16.3, 14.8])[:, None, None, None]) ** 2).sum(0))
    a = np.zeros(x.shape[1:], np.int16)
    a[r < 13] = 42
    a[r < 9] = 41
    src = d / "in" / "sub-T01_ses-S1_aseg.nii.gz"
    src.parent.mkdir(parents=True)
    nib.save(nib.nifti1.Nifti1Image(a, np.diag([0.16, 0.16, 0.16, 1.0])), src)
    white_dseg.main([str(src), "--out", str(d / "deriv")])
    anat = d / "deriv/sub-T01/ses-S1/anat"
    dseg = anat / "sub-T01_ses-S1_desc-wmtopo_dseg.nii.gz"
    args = ["--block-cells", "8"]
    white_surf.main([str(dseg), "--out", str(d / "deriv"), *args])
    return anat / "sub-T01_ses-S1_hemi-R_white.surf.gii"


def test_the_sites_command_writes_and_reads_the_files(surf: Path) -> None:
    WS.main([str(surf)])
    p = WS.sites_paths(surf)
    rows, side = WS.read_sites(p["table"])
    assert side["Sites"] == len(rows) and side["Hemisphere"] == "R"
    assert side["MisfacingFaces"] >= len(rows)
    dseg = surf.with_name("sub-T01_ses-S1_desc-wmtopo_dseg.nii.gz")
    assert any(Path(s).name == dseg.name for s in side["Sources"])
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
    assert {r["kind"] for r in rows} <= {"misfacing"}
    # a table with another header is refused
    bad = p["table"].with_name("bad_desc-wmsites_sites.tsv")
    bad.write_text("rank\ti\n")
    bad.with_suffix(".json").write_text(p["sidecar"].read_text())
    with pytest.raises(ValueError, match="columns"):
        WS.read_sites(bad)


def test_the_sites_command_writes_to_another_directory(
    surf: Path, tmp_path: Path
) -> None:
    out = tmp_path / "elsewhere"
    WS.main([str(surf), "--out", str(out)])
    p = WS.sites_paths(surf, out)
    assert p["table"].parent == out and p["sidecar"].exists()
    rows, _ = WS.read_sites(p["table"])
    assert {r["kind"] for r in rows} <= {"misfacing"}


def test_random_sites_repeat_with_the_seed_and_differ_with_another() -> None:
    v, _ = sheet(16)
    shape = (24, 24, 12)
    seg = np.zeros(shape, np.int16)
    seg[:, :, :5] = 41
    fx = np.zeros(shape, np.uint8)
    a = WS.random_sites(v, 10, 7, seg, (41,), fx)
    assert a == WS.random_sites(v, 10, 7, seg, (41,), fx)
    b = WS.random_sites(v, 10, 8, seg, (41,), fx)
    assert [(r["i"], r["j"]) for r in a] != [(r["i"], r["j"]) for r in b]
    assert WS.random_sites(v, 0, 7, seg, (41,), fx) == []
    with pytest.raises(ValueError, match="vertices"):
        WS.random_sites(v, len(v) + 1, 7, seg, (41,), fx)


def test_random_sites_are_rounded_surface_vertices_and_unranked() -> None:
    v, _ = sheet(16)
    v = v + 0.3  # not on voxel centres, so rounding shows
    shape = (24, 24, 12)
    seg = np.zeros(shape, np.int16)
    seg[:, :, :5] = 41
    rows = WS.random_sites(v, 20, 3, seg, (41,), np.zeros(shape, np.uint8))
    assert len(rows) == 20
    voxels = {tuple(np.rint(p).astype(int)) for p in v}
    got = [(r["i"], r["j"], r["k"]) for r in rows]
    assert set(got) <= voxels
    assert all(
        r["kind"] == "random" and r["rank"] == 0 and r["faces"] == 0 for r in rows
    )
    assert set(rows[0]) == set(WS.COLUMNS)


def test_random_sites_clip_into_the_grid() -> None:
    v = np.array([[-0.4, 3.0, 99.0]])
    seg = np.zeros((5, 5, 5), np.int16)
    (row,) = WS.random_sites(v, 1, 0, seg, (41,), np.zeros((5, 5, 5), np.uint8))
    assert (row["i"], row["j"], row["k"]) == (0, 3, 4)


def test_random_sites_from_a_surface_file(surf: Path) -> None:
    a = WS.read_random_sites(surf, 5, 1)
    assert a == WS.read_random_sites(surf, 5, 1)
    assert len(a) == 5 and all(r["kind"] == "random" and r["rank"] == 0 for r in a)
