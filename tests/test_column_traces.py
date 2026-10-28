"""Tests for baillarger.column_traces: roots, root patches, classes, and the
traces of a synthetic box with known columns."""

import numpy as np
import pytest
from scipy.sparse.csgraph import dijkstra

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


def test_root_patches_are_a_geodesic_voronoi_partition() -> None:
    t = slab_tissue()
    s = CT.setup(t.wm, t.ctx, patch=4)
    assert s.seeds.sum() == 30 * 30 and np.all(s.seeds[5])  # one WM layer
    d = dijkstra(s.graph, indices=s.rep)
    # centres at least the spacing apart; each seed in its nearest centre's patch
    off = d[:, s.rep][~np.eye(s.npatch, dtype=bool)]
    assert off.min() >= 4.0 - 1e-9
    comp = s.patch.ravel()[s.sflat]
    assert np.allclose(d[comp, np.arange(len(comp))], d.min(0))


@pytest.mark.parametrize("tracer", CT.TRACERS)
def test_traces_of_a_slab_are_straight_open_columns(tracer: str) -> None:
    """Layers parallel to a flat white surface: every trace away from the
    side faces goes straight up (along x) to the fluid, OPEN, in one cortex
    thickness."""
    t = slab_tissue()
    R = CT.trace_columns(t, tracer)
    x0 = R.v[R.tv]
    up = (R.nrm[R.tv, 0] > 0.9) & np.all((x0[:, 1:] > 9) & (x0[:, 1:] < 20), 1)
    assert up.sum() > 50
    assert np.all(R.cls[up] == CT.OPEN) and np.all(R.fallback[up] == 0)
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
