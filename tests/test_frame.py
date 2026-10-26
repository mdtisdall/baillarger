"""Tests for baillarger.frame on the phantoms, and on the example data when
data/ has it (skipped otherwise)."""

import numpy as np
import numpy.typing as npt
import pytest
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from baillarger import data
from baillarger import frame as fr
from baillarger import phantoms as ph

VOX = (0.16, 0.16, 0.16)
type Bool3 = npt.NDArray[np.bool_]


@pytest.fixture(scope="module")
def slab() -> tuple[ph.Rim, fr.Frame]:
    """P3's flat slab: WM z 2..9, cortex z 10..24, outside above."""
    rim = ph.p3_stria_band()[0]
    return rim, fr.build_frame(rim, VOX)


def test_flat_slab_depth_is_the_height_above_the_wm(
    slab: tuple[ph.Rim, fr.Frame],
) -> None:
    rim, F = slab
    z = np.broadcast_to(np.arange(rim.shape[2])[None, None, :], rim.shape)
    above = np.zeros(rim.shape, bool)
    above[3:-3, 3:-3, 10:] = True
    m = above & (F.voxel_column >= 0)
    # the white surface is at z = 9.5, between the WM and the cortex
    assert m.sum() > 40000
    assert np.abs(F.depth[m] - (z[m] - 9.5) * 0.16).max() < 1e-4
    assert np.nanmax(np.abs(F.distance[m] - (z[m] - 9.5) * 0.16)) < 1e-4
    # the whole cortex is in a column, and the streamlines stop at D_max above
    assert (F.voxel_column[rim == 3] >= 0).all()
    assert (F.path_end == fr.END_DMAX).sum() > 0.4 * len(F.path_end)
    # every column's thickness: the depth of its last cortex voxel, z = 24
    T = fr.column_thickness(F, rim == 3)
    top = F.vertex_column[F.verts[:, 2] > 9]  # the columns on the slab's top
    assert np.allclose(T[np.unique(top)], 14.5 * 0.16, atol=1e-4)
    r = fr.relative_depth(F, T)
    assert np.allclose(r[m & (rim == 3)], F.depth[m & (rim == 3)] / (14.5 * 0.16))


def test_columns_are_geodesic_voronoi_cells(slab: tuple[ph.Rim, fr.Frame]) -> None:
    for rim, F in (slab, (ph.p5_wm_handle()[0], None)):
        if F is None:
            v, f, _ = fr.white_mesh(rim == 2, VOX)
            col, seed = fr.mesh_columns(v, f, VOX, 0.7)
        else:
            v, f, col, seed = F.verts, F.faces, F.vertex_column, F.column_seed
        E = fr._edges(f)
        w = np.linalg.norm((v[E[:, 0]] - v[E[:, 1]]) * 0.16, axis=1)
        A = sparse.coo_matrix((w, (E[:, 0], E[:, 1])), shape=(len(v), len(v)))
        dall = dijkstra(A.tocsr(), directed=False, indices=seed)  # (seeds, verts)
        d = dall.min(0)
        # every vertex within 0.35 mm of a seed, along the mesh, and in the
        # column of a nearest seed (ties are common on a flat grid)
        assert d.max() <= 0.35
        own = dall[col, np.arange(len(v))]
        assert np.all(own <= d + 1e-9)
        assert (col[seed] == np.arange(len(seed))).all()
        # farthest-point seeds are at least the covering radius apart
        D = dall[:, seed]
        assert np.min(np.where(D > 0, D, np.inf)) >= 0.35


def test_kissing_banks_meet_at_the_ridge() -> None:
    rim, _, e = ph.p1_kissing_banks()
    F = fr.build_frame(rim, VOX)
    patch = e["patch_mask"]
    assert (F.voxel_column[patch] >= 0).all()
    # no column holds voxels of both banks in the touching patch
    a = set(np.unique(F.voxel_column[e["bank_a_mask"] & patch]).tolist())
    b = set(np.unique(F.voxel_column[e["bank_b_mask"] & patch]).tolist())
    assert a and b and not a & b
    # the streamlines of both banks stop at the ridge midway between the WM
    # plates, (14 + 16 + 14) / 2 voxels from each
    assert (F.path_end == fr.END_RIDGE).sum() > 1000
    assert abs(np.nanmax(F.depth[patch]) - 22 * 0.16) < 2 * 0.16


def same_frame(F: fr.Frame, G: fr.Frame) -> None:
    for k in ("verts", "faces", "vertex_column", "path_start", "path_end"):
        assert np.array_equal(getattr(F, k), getattr(G, k)), k
    for k in ("path_points", "path_arc", "distance", "depth"):
        assert np.array_equal(getattr(F, k), getattr(G, k), equal_nan=True), k
    assert np.array_equal(F.voxel_column, G.voxel_column)


def test_frame_does_not_depend_on_the_pial_surface() -> None:
    """Test 8 of the plan: changing the pial surface (cortex against fluid)
    changes neither the streamlines nor the depth."""
    for phantom in (ph.p1_kissing_banks, ph.p2_sulcus_end):
        rim = phantom()[0]
        F = fr.build_frame(rim, VOX)
        rng = np.random.default_rng(0)
        other = rim.copy()
        flip = ((rim == 1) | (rim == 3)) & (rng.random(rim.shape) < 0.3)
        other[flip] = 4 - rim[flip]  # 1 <-> 3
        same_frame(F, fr.build_frame(other, VOX))


def excised_slab() -> tuple[ph.Rim, ph.Flash, Bool3]:
    """P3's slab with a block cut out (WM and cortex), which the labels still
    call tissue but which is black in the FLASH; and a small dark spot that is
    too small to be an excision."""
    rim, flash, _ = ph.p3_stria_band()
    flash = flash.copy()
    block = np.zeros(rim.shape, bool)
    block[16:30, 16:30, 4:22] = True
    flash[block] = 0.02
    flash[5:7, 40:42, 14:16] = 0.02
    return rim, flash, block


def test_missing_tissue_finds_the_excision_only() -> None:
    rim, flash, block = excised_slab()
    miss = fr.missing_tissue(rim, flash, VOX)
    tissue = (rim == 2) | (rim == 3)
    core = block.copy()
    core[16:18] = core[28:30] = False  # smoothing softens the block's edges
    core[:, 16:18] = core[:, 28:30] = False
    core[:, :, 4:6] = core[:, :, 20:22] = False
    assert miss[core & tissue].all()
    assert not miss[~tissue].any()
    # within the margin round the block, and not at the small spot
    far = np.ones(rim.shape, bool)
    far[12:34, 12:34, :] = False
    assert not miss[far].any()


def test_excised_columns_are_cut() -> None:
    rim, flash, block = excised_slab()
    miss = fr.missing_tissue(rim, flash, VOX)
    F = fr.build_frame(rim, VOX, miss)
    # no voxel of the missing tissue, nor any voxel whose nearest streamline
    # ends at it, has a column
    assert not (F.voxel_column[miss] >= 0).any()
    assert F.path_cut.any() and F.column_cut.any()
    assert np.array_equal(F.path_cut, F.path_end == fr.END_MISSING)
    assert not np.isin(
        F.voxel_path[F.voxel_path >= 0], np.flatnonzero(F.path_cut)
    ).any()
    # the streamlines do not enter it (their start points are on the white
    # mesh, which is made from all the WM, the imputed WM too)
    first = np.zeros(len(F.path_points), bool)
    first[F.path_start[:-1][np.diff(F.path_start) > 0]] = True
    s = F.path_points[~first]
    vox = np.clip(np.floor(s + 0.5).astype(int), 0, np.array(rim.shape) - 1)
    assert not miss[vox[:, 0], vox[:, 1], vox[:, 2]].any()
    # away from the cut the slab is as without it
    T = fr.column_thickness(F, rim == 3)
    top = np.unique(F.vertex_column[(F.verts[:, 2] > 9)])
    ok = top[~F.column_cut[top]]
    assert len(ok) > 100
    # beside the cut, D bends round the missing tissue, and so do the
    # streamlines: there the depth is a little longer
    sx, sy = F.verts[F.column_seed[ok], 0], F.verts[F.column_seed[ok], 1]
    away = np.maximum(np.abs(sx - 22.5), np.abs(sy - 22.5)) > 7 + 6
    assert away.sum() > 100
    assert np.allclose(T[ok[away]], 14.5 * 0.16, atol=1e-4)
    assert np.all(np.abs(T[ok] - 14.5 * 0.16) < 0.25)
    # the white mesh on the imputed WM: its streamlines are all cut, so they
    # take no voxels
    vi = np.floor(F.verts + 0.5).astype(int).T
    inside = miss[vi[0], vi[1], vi[2]]
    assert inside.sum() > 50
    assert F.path_cut[inside].all()
    assert not np.isin(F.voxel_path, np.flatnonzero(inside)).any()


# ---------------------------------------------------------------- the example data

need_data = pytest.mark.skipif(not data.data_available(), reason="no example data")


@need_data
def test_v1v2_frame() -> None:
    p = data.load_patch("V1V2")
    miss = fr.missing_tissue(p.rim, p.flash, p.voxel_mm)
    F = fr.build_frame(p.rim, p.voxel_mm, miss)
    assert miss.sum() > 100000  # V1V2 has excised blocks
    cortex = (p.rim == 3) & ~miss
    assert (F.voxel_column[cortex] >= 0).mean() > 0.99
    assert not (F.voxel_column[miss] >= 0).any()
    assert F.column_cut.any() and not F.column_cut.all()
    # the review points of the seams are all in columns, at cortical depths
    for pts in data.V1V2_REVIEW_POINTS.values():
        for q in pts:
            assert F.voxel_column[q] >= 0
            assert 0 < F.depth[q] < 4
    T = fr.column_thickness(F, p.rim == 3)
    assert 1.0 < np.nanmedian(T[~F.column_cut]) < 2.5
