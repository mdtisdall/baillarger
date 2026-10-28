"""Tests for baillarger.column_traces: roots, root patches, classes, and the
traces of a synthetic box with known columns."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from baillarger import column_front as CF
from baillarger import column_traces as CT
from baillarger import streamlines as S
from baillarger import white
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
    thickness, by every method (traverse and front follow the fitted field
    only)."""
    t = slab_tissue()
    if tracer == "current" and method in ("traverse", "front"):
        with pytest.raises(ValueError):
            CT.trace_columns(t, tracer, method)
        return
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


def test_traverse_crosses_parallel_fused_banks_straight() -> None:
    """Two parallel WM slabs with fused cortex between: traverse columns go
    through the crest (x = 17.5) into the far slab's WM, straight opposite
    their root, begin to descend T within the crest's stencil beyond the
    midline, and their samples stay STEP_MM apart."""
    rim = np.full((36, 30, 24), 3)
    rim[:6] = 2
    rim[30:] = 2
    R = CT.trace_columns(box_tissue(rim), "fitted_blend", "traverse")
    assert R.turn_at is not None and R.area is None
    x0 = R.v[R.tv]
    mid = (R.nrm[R.tv, 0] > 0.9) & np.all(np.abs(x0[:, 1:] - [15, 12]) < 4, 1)
    i = np.flatnonzero(mid)
    assert i.size > 5 and np.all(R.cls[i] == CT.CROSS)
    end = R.pos[i, R.ns[i] - 1].astype(float)
    assert np.all(np.round(end[:, 0]) == 30)
    assert np.max(np.abs(end[:, 1:] - x0[i, 1:])) < 0.1
    assert np.all(np.abs(R.pos[i, R.turn_at[i], 0] - 17.5) <= 3.0)
    steps = np.linalg.norm(np.diff(R.pos[i[0], : R.ns[i[0]]], axis=0) * VOX, axis=1)
    assert np.allclose(steps, S.STEP_MM, atol=0.01)


def test_traverse_crosses_oblique_fused_banks() -> None:
    """Two WM slabs at 30 degrees to each other with fused cortex between
    (the left face x = 6, the right face x = 30 + tan 30 (y - 20)), fluid
    beyond y = 34: columns from the left bank meet the crest at an angle,
    where the crest method's traces reflected; traverse columns go on into
    the right bank's WM. At least 90 %: the crest test is statistical, with
    a noise margin, and can miss a column's crest voxel."""
    shape = (44, 40, 20)
    x, y = np.indices(shape)[:2].astype(float)
    rim = np.full(shape, 3)
    rim[x < 6] = 2
    rim[x - 30 > (y - 20) * np.tan(np.radians(30))] = 2
    rim[(x >= 6) & (rim == 3) & (y > 34)] = 1
    R = CT.trace_columns(box_tissue(rim), "fitted_blend", "traverse")
    x0 = R.v[R.tv]
    left = (R.nrm[R.tv, 0] > 0.9) & np.all(np.abs(x0[:, 1:] - [16, 10]) < 4, 1)
    i = np.flatnonzero(left & ~R.near_face)
    assert i.size > 5
    assert np.mean(R.crest_at[i] >= 0) > 0.9
    end = R.pos[i, R.ns[i] - 1].astype(float)
    q = np.round(end).astype(int)
    far = rim[tuple(q.T)] == 2
    assert np.mean(far & (end[:, 0] > 12)) > 0.9


def test_traverse_leaves_columns_open_beside_a_narrow_open_sulcus() -> None:
    """Two WM slabs (x < 6, x >= 30) with a 2-voxel fluid slot at x = 21-22,
    off the WM midline (x = 17.5). T's sign comes from the WM distance through
    tissue, so it rises in each bank up to the fluid: no crest in the left
    bank's cortex, and its columns stay open (with a straight-line distance,
    the crest fell at 17.5, short of the fluid)."""
    rim = np.full((36, 30, 24), 3)
    rim[:6] = 2
    rim[30:] = 2
    rim[21:23] = 1
    R = CT.trace_columns(box_tissue(rim), "fitted_blend", "traverse")
    x0 = R.v[R.tv]
    left = (
        (R.nrm[R.tv, 0] > 0.9)
        & (x0[:, 0] < 7)
        & np.all(np.abs(x0[:, 1:] - [15, 12]) < 4, 1)
    )
    i = np.flatnonzero(left)
    assert i.size > 5
    assert np.mean(R.cls[i] == CT.OPEN) > 0.9
    assert np.mean(R.crest_at[i] < 0) > 0.9
    end = R.pos[i, R.ns[i] - 1].astype(float)
    assert np.all(np.round(end[:, 0]) <= 22)


def test_normal_rays_from_a_ball_are_radial_with_the_equivolume_coordinate() -> None:
    """A ball of WM (R = 9 voxels) in cortex, fluid beyond r = 21: normal
    columns are radial and reach the fluid; their equivolume coordinate
    follows the swept volume of a sphere's offsets, s + s^2/R + s^3/(3 R^2)
    (mm), within the mesh curvature's accuracy."""
    n = 48
    idx = np.indices((n, n, n)).astype(float)
    c = np.array([23.7, 24.2, 23.9])
    r = np.sqrt(((idx - c[:, None, None, None]) ** 2).sum(0))
    rim = np.full(r.shape, 3)
    rim[r <= 9] = 2
    rim[r > 21] = 1
    R = CT.trace_columns(box_tissue(rim), "fitted_blend", "normal")
    assert R.vol is not None
    k = np.flatnonzero(~R.near_face)
    assert np.mean(R.cls[k] == CT.OPEN) > 0.95
    x0 = R.v[R.tv[k]]
    end = R.pos[k, R.ns[k] - 1].astype(float)
    radial = (x0 - c) / np.linalg.norm(x0 - c, axis=1, keepdims=True)
    step = (end - x0) / np.linalg.norm(end - x0, axis=1, keepdims=True)
    assert np.median((radial * step).sum(1)) > 0.995
    s = (R.ns[k] - 1) * S.STEP_MM
    Rm = 9.0 * VOX
    want = s + s**2 / Rm + s**3 / (3 * Rm**2)
    got = R.vol[k, R.ns[k] - 1]
    assert np.median(np.abs(got / want - 1)) < 0.1


def test_normal_rays_cross_parallel_fused_banks_and_meet_at_the_midline() -> None:
    rim = np.full((36, 30, 24), 3)
    rim[:6] = 2
    rim[30:] = 2
    R = CT.trace_columns(box_tissue(rim), "fitted_blend", "normal")
    x0 = R.v[R.tv]
    mid = (R.nrm[R.tv, 0] > 0.9) & np.all(np.abs(x0[:, 1:] - [15, 12]) < 4, 1)
    i = np.flatnonzero(mid)
    assert i.size > 5 and np.all(R.cls[i] == CT.CROSS)
    assert np.all(R.crest_at[i] >= 0)
    assert np.all(np.abs(R.pos[i, R.crest_at[i], 0] - 17.5) <= 1.0)
    end = R.pos[i, R.ns[i] - 1].astype(float)
    assert np.max(np.abs(end[:, 1:] - x0[i, 1:])) < 0.3


def test_normal_rays_meet_on_the_bisector_of_a_narrow_fused_v() -> None:
    shape = (34, 30, 20)
    x, y = np.indices(shape)[:2].astype(float)
    rim = np.full(shape, 3)
    rim[np.abs(y - 15) > (x - 6) / 3 + 1.5] = 2
    rim[x < 6] = 2
    rim[x >= 30] = 1
    R = CT.trace_columns(box_tissue(rim), "fitted_blend", "normal")
    wall = (np.abs(R.nrm[R.tv, 1]) > 0.8) & ~R.near_face
    met = np.flatnonzero(wall & (R.crest_at >= 0))
    assert met.size > 10
    at = R.pos[met, R.crest_at[met]].astype(float)
    assert np.median(np.abs(at[:, 1] - 15.0)) < 1.0


def test_normal_rays_see_no_meeting_beside_an_open_slot() -> None:
    rim = np.full((36, 30, 24), 3)
    rim[:6] = 2
    rim[30:] = 2
    rim[21:23] = 1
    R = CT.trace_columns(box_tissue(rim), "fitted_blend", "normal")
    x0 = R.v[R.tv]
    left = (
        (R.nrm[R.tv, 0] > 0.9)
        & (x0[:, 0] < 7)
        & np.all(np.abs(x0[:, 1:] - [15, 12]) < 4, 1)
    )
    i = np.flatnonzero(left)
    assert i.size > 5
    assert np.all(R.cls[i] == CT.OPEN) and np.all(R.crest_at[i] < 0)


def test_a_saved_white_mesh_gives_the_same_traces(tmp_path: Path) -> None:
    """trace_columns with the box's saved WhiteMesh gives the traces it gives
    when it makes the mesh itself, and refuses a mesh of another WM label."""
    t = slab_tissue()
    m = white.box_mesh(t.wm, t.patch.lo, t.patch.hi, t.patch.affine)
    m.save(tmp_path / "m.npz")
    m = white.WhiteMesh.load(tmp_path / "m.npz")
    R0 = CT.trace_columns(t, "fitted_blend", "crest")
    R1 = CT.trace_columns(t, "fitted_blend", "crest", mesh=m)
    for k in ("v", "f", "nrm", "tv", "pos", "ns", "cls", "crest_at"):
        assert np.array_equal(getattr(R1, k), getattr(R0, k), equal_nan=True), k
    other = t.wm.copy()
    other[6, 3, 3] = True
    m2 = white.box_mesh(other, t.patch.lo, t.patch.hi, t.patch.affine)
    with pytest.raises(ValueError, match="WM label"):
        CT.trace_columns(t, "fitted_blend", "crest", mesh=m2)


# ---------------------------------------------------------------- front


def shell_tissue(
    axes: tuple[int, ...], shape: tuple[int, int, int]
) -> tuple[CT.Tissue, np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """WM within 6 voxels of a centre (a ball) or of an axis (a cylinder: the
    distance over the given axes only), cortex out to 17 voxels, fluid
    beyond. Returns the tissue, the centre and the distance function."""
    c = np.array([21.7, 22.2, 21.9])[: len(shape)]

    def dist(p: np.ndarray) -> np.ndarray:
        out: np.ndarray = np.sqrt(sum((p[..., a] - c[a]) ** 2 for a in axes))
        return out

    r = dist(np.moveaxis(np.indices(shape).astype(float), 0, -1))
    rim = np.full(shape, 1)
    rim[r <= 17] = 3
    rim[r <= 6] = 2
    return box_tissue(rim), c, dist


@pytest.mark.parametrize("axes, power", [((0, 1, 2), 2), ((0, 1), 1)])
def test_front_of_a_ball_and_a_cylinder_is_radial_with_bok_areas(
    axes: tuple[int, ...], power: int
) -> None:
    """Front columns from a WM ball (cylinder) stay radial, and the front's
    dual areas grow as (1 + k s)^2 (as 1 + k s), k = 1 / the root's radius:
    in total within 1 % (a polyhedron's offset), per vertex at the median
    within 3 %; the swept volume is the integral of that area."""
    shape = (44, 44, 44) if power == 2 else (44, 44, 24)
    t, c, dist = shell_tissue(axes, shape)
    R = CT.trace_columns(t, "fitted_blend", "front")
    assert R.area is not None and R.vol is not None
    x0 = R.v[R.tv]
    keep = np.ones(len(R.tv), bool) if power == 2 else np.abs(x0[:, 2] - 12) < 5
    assert np.all(R.outc[keep] == S.FLUID)
    i = 40  # 1.6 mm out, before any reaches the fluid at 11 voxels (1.76 mm)
    ok = keep & (R.ns > i)
    assert ok.mean() > 0.9 * keep.mean()
    s = i * S.STEP_MM
    r0 = dist(x0[ok]) * VOX
    grow = (1 + s / r0) ** power
    a0, a1 = R.area[ok, 0], R.area[ok, i]
    assert abs(a1.sum() / (a0 * grow).sum() - 1) < 0.01
    assert abs(np.median(a1 / a0 / grow) - 1) < 0.03
    vol = a0 * r0 / (power + 1) * ((1 + s / r0) ** (power + 1) - 1)
    assert abs((R.vol[ok, i] * a0).sum() / vol.sum() - 1) < 0.01
    # radial: the angle between the column's chord and the root's radius
    radial = x0[ok] - c
    if power == 1:
        radial[:, 2] = 0
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    chord = R.pos[ok, i] - x0[ok]
    chord /= np.linalg.norm(chord, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((radial * chord).sum(1), -1, 1)))
    assert np.median(ang) < 5 and ang.max() < 10


def test_front_of_a_slab_keeps_its_area() -> None:
    """Layers parallel to a flat white surface: the front's dual areas stay
    as they were, and the swept volume per unit root area is the length."""
    R = CT.trace_columns(slab_tissue(), "fitted_blend", "front")
    assert R.area is not None and R.vol is not None and R.turn_at is not None
    x0 = R.v[R.tv]
    up = (R.nrm[R.tv, 0] > 0.9) & np.all((x0[:, 1:] > 9) & (x0[:, 1:] < 20), 1)
    i = 30
    assert np.allclose(R.area[up, i], R.area[up, 0], rtol=1e-3)
    assert np.allclose(R.vol[up, i], i * S.STEP_MM, rtol=1e-3)
    assert np.all(R.turn_at[up] == -1)


def test_front_crosses_parallel_fused_banks() -> None:
    """Two parallel WM slabs with fused cortex between: front columns cross
    to the far bank (crossing) straight, and begin to descend T beyond the
    midline x = 17.5, within the crest's stencil."""
    rim = np.full((36, 30, 24), 3)
    rim[:6] = 2
    rim[30:] = 2
    R = CT.trace_columns(box_tissue(rim), "fitted_blend", "front")
    assert R.turn_at is not None
    x0 = R.v[R.tv]
    mid = (R.nrm[R.tv, 0] > 0.9) & np.all(np.abs(x0[:, 1:] - [15, 12]) < 4, 1)
    i = np.flatnonzero(mid)
    assert i.size > 5 and np.all(R.cls[mid] == CT.CROSS)
    end = R.pos[i, R.ns[i] - 1]
    assert np.max(np.abs(end[:, 1:] - x0[i, 1:])) < 0.1
    assert np.all(R.turn_at[i] >= 0)
    assert np.all(np.abs(R.pos[i, R.turn_at[i], 0] - 17.5) <= 3.0)


@pytest.mark.parametrize("method", ["crest", "traverse", "front"])
def test_columns_of_a_narrow_fused_fundus(method: str) -> None:
    """A V of WM with cortex fused in it (as above). T's crest on the
    bisector rises toward the opening, and the sampled grad T at the kink
    points along it: the crest method's columns slide along it to the fluid;
    traverse and front columns keep their heading through the kink, then
    descend into the far wall, near the height they started (no slide).
    Wall roots from 9 to 20 voxels up: nearer the opening a column reaches
    the fluid first, and the apex's floor is another wall."""
    shape = (34, 30, 20)
    x, y = np.indices(shape)[:2].astype(float)
    rim = np.full(shape, 3)
    rim[np.abs(y - 15) > (x - 6) / 3 + 1.5] = 2
    rim[x < 6] = 2
    rim[x >= 30] = 1
    R = CT.trace_columns(box_tissue(rim), "fitted_blend", method)
    x0 = R.v[R.tv]
    wall = (x0[:, 0] > 9) & (x0[:, 0] < 20) & (np.abs(x0[:, 1] - 15) > 1)
    wall &= (x0[:, 2] > 4) & (x0[:, 2] < 15)
    i = np.flatnonzero(wall)
    end = R.pos[i, R.ns[i] - 1].astype(float)
    dx = np.abs(end[:, 0] - x0[i, 0])
    if method == "crest":
        assert np.all(R.outc[i] == S.FLUID) and np.median(dx) > 10
    else:
        assert np.all(R.outc[i] == S.WM)
        assert np.all(np.sign(end[:, 1] - 15) != np.sign(x0[i, 1] - 15))
        assert np.median(dx) < 2 and dx.max() < 5


def test_front_crosses_oblique_fused_banks() -> None:
    """The banks at 30 degrees of the traverse test: front columns from the
    left bank meet the crest at an angle, T stops rising along them past the
    kink, and they descend into the right bank's WM. At least 90 %, as for
    traverse."""
    shape = (44, 40, 20)
    x, y = np.indices(shape)[:2].astype(float)
    rim = np.full(shape, 3)
    rim[x < 6] = 2
    rim[x - 30 > (y - 20) * np.tan(np.radians(30))] = 2
    rim[(x >= 6) & (rim == 3) & (y > 34)] = 1
    R = CT.trace_columns(box_tissue(rim), "fitted_blend", "front")
    assert R.turn_at is not None
    x0 = R.v[R.tv]
    left = (R.nrm[R.tv, 0] > 0.9) & np.all(np.abs(x0[:, 1:] - [16, 10]) < 4, 1)
    i = np.flatnonzero(left & ~R.near_face)
    assert i.size > 5
    assert np.mean(R.turn_at[i] >= 0) > 0.9
    end = R.pos[i, R.ns[i] - 1].astype(float)
    far = rim[tuple(np.round(end).astype(int).T)] == 2
    assert np.mean(far & (end[:, 0] > 12)) > 0.9


def test_front_leaves_columns_open_beside_a_narrow_open_sulcus() -> None:
    """The off-centre fluid slot of the traverse test: front columns from the
    left bank stay open and end at the slot. T rises in each bank up to the
    fluid, but it is extended one voxel into the fluid from each bank, so
    one voxel short of a narrow slot the sampled gradient differences the
    two banks' extensions and points back: turn_at records a descent on the
    last step, into the fluid (a property of the field, not a crest)."""
    rim = np.full((36, 30, 24), 3)
    rim[:6] = 2
    rim[30:] = 2
    rim[21:23] = 1
    R = CT.trace_columns(box_tissue(rim), "fitted_blend", "front")
    assert R.turn_at is not None
    x0 = R.v[R.tv]
    left = (
        (R.nrm[R.tv, 0] > 0.9)
        & (x0[:, 0] < 7)
        & np.all(np.abs(x0[:, 1:] - [15, 12]) < 4, 1)
    )
    i = np.flatnonzero(left)
    assert i.size > 5
    assert np.all(R.cls[i] == CT.OPEN)
    end = R.pos[i, R.ns[i] - 1].astype(float)
    assert np.all(np.round(end[:, 0]) == 21)
    assert np.all((R.turn_at[i] == -1) | (R.turn_at[i] >= R.ns[i] - 3))


# ---------------------------------------------------------------- equivolume front

EQ = "front-equivolume"


@pytest.mark.parametrize("axes, power", [((0, 1, 2), 2), ((0, 1), 1)])
def test_equivolume_front_of_a_ball_and_a_cylinder_follows_equal_volume(
    axes: tuple[int, ...], power: int
) -> None:
    """Equivolume front columns from a WM ball (cylinder) of radius r0 stay
    radial, and the front sweeps equal volume per unit root area: at radius
    r its coordinate is tau = (r^3 - r0^3) / (3 r0^2) (tau = (r^2 - r0^2) /
    (2 r0)). r0 is the label surface's radius (the mean over the roots): the
    mesh's sub-voxel bumps (sd about 0.17 voxel in a root's radius) are
    carried out as an offset, so the column through r at a given tau does not
    know its own root's bump. The median column, at 0.4 to 1.6 mm of arc,
    within 2 % (a polyhedron's offset, as the field front's 1-3 %). The
    front is convex, so the speed stays under the cap and the flux agrees,
    per column: vol = tau."""
    shape = (44, 44, 44) if power == 2 else (44, 44, 24)
    t, c, dist = shell_tissue(axes, shape)
    R = CT.trace_columns(t, "fitted_blend", EQ)
    assert R.area is not None and R.vol is not None and R.tau is not None
    x0 = R.v[R.tv]
    keep = np.ones(len(R.tv), bool) if power == 2 else np.abs(x0[:, 2] - 12) < 5
    assert np.all(R.outc[keep] == S.FLUID)
    r0 = dist(x0[keep]).mean() * VOX
    for i in (10, 20, 30, 40):  # the fluid is at 1.76 mm
        ok = keep & (R.ns > i)
        assert ok.mean() > 0.9 * keep.mean()
        r = np.median(dist(R.pos[ok, i].astype(float))) * VOX
        exact = (r ** (power + 1) - r0 ** (power + 1)) / ((power + 1) * r0**power)
        assert abs(np.median(R.tau[ok, i]) / exact - 1) < 0.02
        assert np.allclose(R.vol[ok, i], R.tau[ok, i], rtol=0.02)
    radial = x0[ok] - c
    if power == 1:
        radial[:, 2] = 0
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    chord = R.pos[ok, i] - x0[ok]
    chord /= np.linalg.norm(chord, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((radial * chord).sum(1), -1, 1)))
    assert np.median(ang) < 5 and ang.max() < 10


def test_equivolume_front_of_a_slab_is_straight_at_unit_speed() -> None:
    """A flat white surface: the columns are straight, the area stays as it
    was, and the equivolume coordinate is the arc length."""
    R = CT.trace_columns(slab_tissue(), "fitted_blend", EQ)
    assert R.area is not None and R.tau is not None
    x0 = R.v[R.tv]
    up = (R.nrm[R.tv, 0] > 0.9) & np.all((x0[:, 1:] > 9) & (x0[:, 1:] < 20), 1)
    assert np.all(R.outc[up] == S.FLUID)
    i = 30
    assert np.allclose(R.tau[up, i], i * S.STEP_MM, rtol=1e-3)
    assert np.allclose(R.area[up, i], R.area[up, 0], rtol=1e-3)
    assert np.max(np.abs(R.pos[up, i, 1:] - x0[up, 1:])) < 0.01


def test_equivolume_front_in_a_tube_of_cortex_speeds_up_as_r0_over_r() -> None:
    """The inside of a WM tube (radius r0 = 15 voxels; cortex down to 5
    voxels, fluid within): the concave model of a fundus, where the front
    shrinks and speeds up. Columns stay radial and inward, and at radius r
    the coordinate is tau = (r0^2 - r^2) / (2 r0) (the area falls as r / r0,
    so the speed rises as r0 / r, to 3 at the fluid: under the cap). As the
    ball's: the median column within 2 %, the flux per column."""
    shape = (44, 44, 24)
    c = np.array([21.7, 22.2])
    x, y = np.indices(shape)[:2].astype(float)
    rr = np.hypot(x - c[0], y - c[1])
    rim = np.full(shape, 2)
    rim[rr < 15] = 3
    rim[rr < 5] = 1
    R = CT.trace_columns(box_tissue(rim), "fitted_blend", EQ)
    assert R.tau is not None and R.vol is not None and R.area is not None
    x0 = R.v[R.tv]
    r0v = np.hypot(x0[:, 0] - c[0], x0[:, 1] - c[1])
    keep = (np.abs(x0[:, 2] - 12) < 5) & (r0v < 16)
    assert keep.sum() > 50 and np.all(R.outc[keep] == S.FLUID)
    r0 = r0v[keep].mean() * VOX
    for i in (10, 20, 30, 35):  # the fluid is at 1.6 mm
        ok = keep & (R.ns > i)
        assert ok.mean() > 0.9 * keep.mean()
        p = R.pos[ok, i].astype(float)
        r = np.median(np.hypot(p[:, 0] - c[0], p[:, 1] - c[1])) * VOX
        exact = (r0**2 - r**2) / (2 * r0)
        assert abs(np.median(R.tau[ok, i]) / exact - 1) < 0.02
        assert np.allclose(R.vol[ok, i], R.tau[ok, i], rtol=0.02)
    chord = p - x0[ok]
    radial = np.c_[c - x0[ok, :2], np.zeros(ok.sum())]
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    chord /= np.linalg.norm(chord, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((radial * chord).sum(1), -1, 1)))
    assert np.median(ang) < 5 and ang.max() < 10


def test_equivolume_front_caps_its_speed_at_a_focus() -> None:
    """A tube of cortex down to 3 voxels: past 3.75 voxels the exact speed
    (r0 / r) passes the cap, SPEED_CAP = 4 times the root's. No column
    moves more than one voxel per step_mm of tau (tau rises at least
    step_mm / 4 per step_mm of arc, the stop's partial sample left out);
    the swept volume per root area falls below tau (the median column
    lags; at an edge or a fold a column goes straight and its flux is no
    longer its own column's); and the columns still reach the fluid, at
    least 99 % (the ring shrinks fivefold and tangles near the core: the
    few that miss it go on straight, as through a fused fundus)."""
    shape = (44, 44, 24)
    c = np.array([21.7, 22.2])
    x, y = np.indices(shape)[:2].astype(float)
    rr = np.hypot(x - c[0], y - c[1])
    rim = np.full(shape, 2)
    rim[rr < 15] = 3
    rim[rr < 3] = 1
    R = CT.trace_columns(box_tissue(rim), "fitted_blend", EQ)
    assert R.tau is not None and R.vol is not None and R.area is not None
    x0 = R.v[R.tv]
    r0v = np.hypot(x0[:, 0] - c[0], x0[:, 1] - c[1])
    keep = (np.abs(x0[:, 2] - 12) < 5) & (r0v < 16)
    k = np.flatnonzero(keep)
    assert np.mean(R.outc[k] == S.FLUID) >= 0.99
    tau = R.tau[k].astype(float)
    dtau = np.diff(tau, axis=1)
    whole = np.arange(dtau.shape[1])[None, :] < (R.ns[k] - 2)[:, None]
    assert np.min(dtau[whole]) >= S.STEP_MM / CF.SPEED_CAP * (1 - 1e-3)
    last = R.ns[k] - 1
    lag = R.vol[k, last] / tau[np.arange(len(k)), last]
    assert np.median(lag) < 1.0
