"""Tests for baillarger.column_traces: roots, root patches, classes, and the
traces of a synthetic box with known columns."""

import numpy as np
import pytest

from baillarger import column_traces as CT
from baillarger import streamlines as S
from baillarger.data import Patch

VOX = 0.16


def slab_tissue(shape: tuple[int, int, int] = (36, 30, 30)) -> CT.Tissue:
    """WM for x < 6, cortex for 6 <= x < 26, fluid above; layers parallel to
    the WM surface (a sinusoidal profile in x); no box faces nearby."""
    rim = np.full(shape, 1, np.int16)
    rim[:6] = 2
    rim[6:26] = 3
    ctx = rim == 3
    x = np.indices(shape)[0].astype(float)
    det = {i: np.sin(x * VOX / 0.25) * ctx for i in (1, 2)}
    labels = np.full(shape, 1, np.int16)
    labels[ctx] = 0
    labels[rim == 2] = 2
    patch = Patch(
        name="slab",
        lo=(0, 0, 0),
        hi=shape,
        echoes=np.zeros((*shape, 3), np.float32),
        echo_times_ms=(1.0, 2.0, 3.0),
        aseg=np.zeros(shape, np.int16),
        rim=rim,
        affine=np.eye(4),
        voxel_mm=(VOX, VOX, VOX),
    )
    return CT.Tissue(
        patch=patch,
        vox=np.full(3, VOX),
        rim=rim,
        miss=np.zeros(shape, bool),
        ctx=ctx,
        wm=rim == 2,
        det=det,
        dface=np.full(shape, 99.0),
        labels=labels,
    )


def test_root_vertices_keep_one_vertex_per_cell_and_side() -> None:
    v = np.array([[0.5, 0.5, 0.5], [0.1, 0.9, 0.5], [0.5, 0.5, 0.6], [1.5, 0.5, 0.5]])
    n = np.array([[1.0, 0, 0], [1.0, 0, 0], [-1.0, 0, 0], [1.0, 0, 0]])
    # 0 and 1 share a cell and side (0 is nearer the centre); 2 is the other
    # side; 3 another cell
    assert CT.root_vertices(v, n).tolist() == [0, 2, 3]


def test_classes() -> None:
    outc = np.array([S.FLUID, S.WM, S.WM, S.CONTACT, S.TURN, S.FACE])
    sep = np.array([0.0, CT.SAME_MM, CT.SAME_MM - 0.1, 0.0, 0.0, 9.0])
    cls = CT.classify(outc, sep)
    assert cls.tolist() == [
        CT.OPEN,
        CT.CROSS,
        CT.OWN,
        CT.TOCONTACT,
        CT.FAILED,
        CT.FAILED,
    ]


def test_white_mesh_of_a_ball_lies_on_its_label_with_outward_normals() -> None:
    n = 32
    idx = np.indices((n, n, n)).astype(float)
    c = np.array([15.7, 16.2, 15.9])
    r = np.sqrt(((idx - c[:, None, None, None]) ** 2).sum(0))
    v, f, nrm = CT.white_mesh(r <= 9.0)
    rad = np.linalg.norm(v - c, axis=1)
    assert abs(np.median(rad) - 9.0) < 0.6  # the binary label's own boundary
    assert np.all(np.sum(nrm * (v - c), 1) / rad > 0.9)


def box_tissue(rim: np.ndarray) -> CT.Tissue:
    """A Tissue from rim labels alone: flat echoes (no data normal, so T
    follows the WM distance), no box faces nearby."""
    shape = rim.shape
    patch = Patch(
        name="box",
        lo=(0, 0, 0),
        hi=(shape[0], shape[1], shape[2]),
        echoes=np.ones((*shape, 3), np.float32),
        echo_times_ms=(1.0, 2.0, 3.0),
        aseg=np.zeros(shape, np.int16),
        rim=rim.astype(np.int16),
        affine=np.eye(4),
        voxel_mm=(VOX, VOX, VOX),
    )
    t = CT.tissue(patch)
    t.dface = np.full(shape, 99.0)
    return t


def test_end_at_bend_ends_a_column_at_its_kink() -> None:
    """A path straight along x for 20 samples, then along y: TURN reported a
    few steps after the corner; the column ends at the corner (CONTACT)."""
    nmax = 40
    pos = np.full((1, nmax, 3), np.nan, np.float32)
    p = [(float(i), 0.0, 0.0) for i in range(21)] + [
        (20.0, float(j), 0.0) for j in range(1, 6)
    ]
    pos[0, : len(p)] = p
    ns = np.array([len(p)])
    out = np.array([S.TURN])
    CT.end_at_bend(pos, ns, out, 8)
    assert out[0] == S.CONTACT
    assert pos[0, ns[0] - 1].tolist() == [20.0, 0.0, 0.0]


def test_crest_cut_ends_a_returning_column_at_t_maximum() -> None:
    """cut_at_crest: a column that entered the WM heading against its root's
    normal is cut at T's largest value along it (CONTACT); one that entered
    heading on is left alone."""
    shape = (40, 8, 8)
    T = np.tile((20.0 - np.abs(np.arange(shape[0]) - 20.0))[:, None, None], (1, 8, 8))
    nmax = 60
    pos = np.full((2, nmax, 3), np.nan, np.float32)
    up = np.arange(0.0, 30.0, 1.0)  # out past the crest at x = 20
    back = np.r_[np.arange(0.0, 25.0, 1.0), np.arange(23.0, 0.0, -1.0)]  # out and back
    pos[0, : len(up), 0] = up
    pos[1, : len(back), 0] = back
    pos[:, :, 1:] = 4.0
    ns = np.array([len(up), len(back)])
    out = np.array([S.WM, S.WM])
    n0 = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert CT.cut_at_crest(pos, ns, out, T, n0).tolist() == [1]
    assert out.tolist() == [S.WM, S.CONTACT]
    assert ns[0] == len(up) and pos[1, ns[1] - 1, 0] == 20.0


def test_crest_method_records_where_columns_meet_a_narrow_fused_fundus() -> None:
    """A V of WM with cortex fused in it (x in [6, 30], half-width (x - 6) / 3
    + 1.5 voxels about y = 15), fluid beyond x = 30. Columns from the walls
    are traced through; where they meet the other wall's field they cross
    T's crest on the bisector y = 15, which crest_at records."""
    shape = (34, 30, 20)
    x, y = np.indices(shape)[:2].astype(float)
    rim = np.full(shape, 3)
    rim[np.abs(y - 15) > (x - 6) / 3 + 1.5] = 2
    rim[x < 6] = 2
    rim[x >= 30] = 1
    R = CT.trace_columns(box_tissue(rim), "fitted_blend", "crest")
    met = np.flatnonzero((R.crest_at >= 0) & ~R.near_face)
    assert met.size > 10
    at = R.pos[met, R.crest_at[met]].astype(float)
    assert np.median(np.abs(at[:, 1] - 15.0)) < 1.0
    assert np.max(np.abs(at[:, 1] - 15.0)) < 2.0


def test_crest_method_crosses_parallel_fused_banks() -> None:
    """Two parallel WM slabs with fused cortex between: columns cross to the
    far bank (crossing), and meet the crest at the midline x = 17.5."""
    rim = np.full((36, 30, 24), 3)
    rim[:6] = 2
    rim[30:] = 2
    R = CT.trace_columns(box_tissue(rim), "fitted_blend", "crest")
    x0 = R.v[R.tv]
    mid = (R.nrm[R.tv, 0] > 0.9) & np.all(np.abs(x0[:, 1:] - [15, 12]) < 4, 1)
    assert mid.sum() > 5 and np.all(R.cls[mid] == CT.CROSS)
    i = np.flatnonzero(mid)
    assert np.all(R.crest_at[i] >= 0)
    assert np.all(np.abs(R.pos[i, R.crest_at[i], 0] - 17.5) <= 1.0)


@pytest.mark.parametrize("method", CT.METHODS)
@pytest.mark.parametrize("tracer", CT.TRACERS)
def test_traces_of_a_slab_are_straight_open_columns(tracer: str, method: str) -> None:
    """Layers parallel to a flat white surface: every trace away from the
    side faces goes straight up (along x) to the fluid, OPEN, in one cortex
    thickness, by either method."""
    t = slab_tissue()
    R = CT.trace_columns(t, tracer, method)
    x0 = R.v[R.tv]
    up = (R.nrm[R.tv, 0] > 0.9) & np.all((x0[:, 1:] > 9) & (x0[:, 1:] < 20), 1)
    assert up.sum() > 50
    assert np.all(R.cls[up] == CT.OPEN) and np.all(R.fallback[up] == 0)
    assert np.all(R.crest_at[up] == -1)
    end = R.pos[np.flatnonzero(up), R.ns[up] - 1]
    assert np.all(np.round(end[:, 0]) == 26)
    assert np.max(np.abs(end[:, 1:] - x0[up, 1:])) < 0.05
    L = (R.ns[up] - 1) * S.STEP_MM
    assert np.allclose(L, (26 - 0.5 - x0[up, 0]) * VOX, atol=S.STEP_MM)


def test_a_handle_free_label_replaces_the_aseg_wm_everywhere() -> None:
    """A one-voxel WM bridge cut by the handle-free label becomes cortex, a
    filled voxel becomes WM, and the labels, the cortex and the white mesh
    all follow the same WM."""
    t0 = slab_tissue()
    rim = t0.rim.copy()
    rim[6:26, 15, 15] = 2  # a WM bridge up through the cortex
    p = t0.patch
    patch = Patch(
        name=p.name,
        lo=p.lo,
        hi=p.hi,
        echoes=np.ones_like(p.echoes),
        echo_times_ms=p.echo_times_ms,
        aseg=p.aseg,
        rim=rim,
        affine=p.affine,
        voxel_mm=p.voxel_mm,
    )
    fixed = rim == 2
    fixed[6:26, 15, 15] = False  # the cut
    fixed[6, 3, 3] = True  # a fill
    t = CT.tissue(patch, fixed)
    assert np.array_equal(t.wm, fixed)
    assert np.array_equal(t.labels == 2, fixed)
    assert np.all(t.ctx[7:26, 15, 15]) and t.labels[10, 15, 15] == 0
    assert t.rim[6, 3, 3] == 2 and not t.ctx[6, 3, 3]
    v, _, _ = CT.white_mesh(t.wm)
    # no vertex of the mesh lies on the cut bridge
    assert not np.any(np.all(np.abs(v[:, 1:] - 15) < 0.5, 1) & (v[:, 0] > 8))
    # without a label, the aseg's
    assert np.array_equal(CT.tissue(patch).wm, rim == 2)
