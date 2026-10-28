"""Steps 2-4 of putting the white surface on the image's edge, on phantoms
with known answers: the inward-only label correction (peel), handle removal
on the corrected label (white.wm_changes with removed voxels), and the
inward-only vertex refinement (refine).

The target is the image's edge minus a margin along the outward normal: a
WM voxel stays if its centre lies at least that far inside the edge, and a
vertex with less than the margin moves inward to it, never outward. The
default margin is 0.1 mm (M); margin 0 is tested too. The phantom image is a
blurred step erf(d / (sqrt(2) s)) of the signed distance d (mm) to a known
surface, dark in the WM; s = 0.24 mm unless stated, wide enough that the
spline's error is about 0.002 mm (test_wm_edge.py).
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt
from scipy.special import erf

from baillarger import data, white
from baillarger.handles import kuhn_betti
from baillarger.mesh_intersect import intersecting_faces
from baillarger.mesh_smoothing import smooth_normals
from baillarger.white import white_mesh
from baillarger.wm_edge import (
    MARGIN_MM,
    SCALE_MM,
    EdgeField,
    peel,
    peel_blocks,
    refine,
    tissue_contrast,
)

type F64 = npt.NDArray[np.float64]
type Bool = npt.NDArray[np.bool_]

V = 0.16
VOX = np.array([V, V, V])
M = 0.1  # a margin (mm) for the mechanism tests


def _img(d_mm: F64, s: float = 0.24) -> F64:
    out: F64 = erf(d_mm / (np.sqrt(2) * s))
    return out


def _x(shape: tuple[int, int, int]) -> F64:
    out: F64 = np.indices(shape)[0].astype(float)
    return out


# ---------------------------------------------------------------- step 2: peel


def _plane_peel(xe: float, margin: float = M) -> tuple[Bool, Bool]:
    """WM x <= 20 (boundary at 20.5); the image's edge at x = xe."""
    shape = (40, 12, 12)
    x = _x(shape)
    p = peel(x <= 20, _img((x - xe) * V), VOX, margin_mm=margin)
    return p.removed, p.flagged


def _layers(m: Bool) -> list[int]:
    return sorted(np.unique(np.nonzero(m)[0]).tolist())


def test_peel_plane_edge_inside() -> None:
    """Edge at 18.3: the target is 18.3 - 0.625 = 17.675, so voxels 18-20
    go (their centres lie less than 0.1 mm inside the edge, or outside it)
    and 17 stays (0.208 mm inside). Three rounds, whole layers."""
    removed, flagged = _plane_peel(18.3)
    assert _layers(removed) == [18, 19, 20]
    assert removed[18:21].all()
    assert not flagged.any()


def test_peel_plane_edge_near_or_outside() -> None:
    """Edge at 20.4: voxel 20's centre is 0.064 mm inside it (< 0.1), so
    layer 20 goes and 19 (0.224 mm) stays. Edge at 20.9 or beyond: nothing
    goes; the label never grows."""
    removed, _ = _plane_peel(20.4)
    assert _layers(removed) == [20]
    for xe in (20.9, 21.6):
        removed, flagged = _plane_peel(xe)
        assert not removed.any() and not flagged.any()


def test_peel_edge_beyond_the_inner_end() -> None:
    """Edge at 16.0, 0.64 mm inside voxel 20's centre: no edge in the
    window. Layer 20 is flagged and kept, and nothing deeper is tried."""
    removed, flagged = _plane_peel(16.0)
    assert not removed.any()
    assert _layers(flagged) == [20] and flagged[20].all()


def test_peel_keeps_a_thin_blades_inner_half() -> None:
    """A WM blade 8 voxels thick (x 16-23; distance to the background 1, 2,
    3, 4, 4, 3, 2, 1 voxels) under an image whose WM is x 17.8-21.2. The
    edge would take 16-18 and 21-23; the guard keeps every voxel deeper than
    half the local thickness (4 voxels), so only 16, 17, 22, 23 go."""
    shape = (40, 12, 12)
    x = _x(shape)
    d = np.maximum(17.8 - x, x - 21.2) * V
    p = peel((x >= 16) & (x <= 23), _img(d, 0.1), VOX, margin_mm=M)
    assert _layers(p.removed) == [16, 17, 22, 23]
    assert not p.flagged.any()


def test_peel_margin_zero_takes_only_centres_past_the_edge() -> None:
    """Margin 0: a voxel goes only if its centre lies outside
    the edge, which is where the mesh (half a voxel further out) lies more
    than half a voxel past it. Edge at 20.4 (voxel 20's centre 0.064 mm
    inside): nothing goes. Edge at 19.7: layer 20 goes (0.048 mm outside),
    19 stays."""
    assert MARGIN_MM == M
    removed, _ = _plane_peel(20.4, 0.0)
    assert not removed.any()
    removed, _ = _plane_peel(19.7, 0.0)
    assert _layers(removed) == [20]


def _capsule(shape: tuple[int, int, int]) -> tuple[Bool, F64]:
    """A WM capsule along x (radius 12 voxels) and an echo whose WM edge is a
    capsule of radius 10.9, under a smooth multiplicative bias."""
    X = np.indices(shape).transpose(1, 2, 3, 0).astype(float)
    c = np.array([0.0, (shape[1] - 1) / 2, (shape[2] - 1) / 2])
    a, b = 20.0, shape[0] - 21.0
    q = X - c
    q[..., 0] -= np.clip(X[..., 0], a, b)
    r = np.linalg.norm(q, axis=-1)
    wm = r < 12.0
    step = 0.5 * (1 + _img((r - 10.9) * V))
    bias = np.exp(0.002 * X[..., 0] + 0.004 * X[..., 1])
    return wm, (100 + 40 * step) * bias


def test_peel_in_blocks_equals_the_whole() -> None:
    """The full grid is peeled in blocks with a halo; each block's decisions
    equal those of the whole volume (the trend, the gradient and the spline
    all reach less than the halo)."""
    wm, echo = _capsule((136, 40, 40))
    whole = peel(wm, tissue_contrast(echo, wm), VOX, margin_mm=M, sign=1.0)
    assert whole.removed.sum() > 1000
    blocks = peel_blocks(wm, echo, VOX, sign=1.0, margin_mm=M, block=24)
    assert np.array_equal(blocks.removed, whole.removed)
    assert np.array_equal(blocks.flagged, whole.flagged)


# ---------------------------------------------------------------- step 3


def test_handles_after_the_peel(tmp_path: Path) -> None:
    """A synthetic aseg: a WM block drilled by a 2 x 2 tunnel (one handle).
    The peel removes the block's top layer (kind EDGE) and a 1 x 1 column
    through it, which makes a second handle. Handle removal on the peeled
    label leaves b1 = 0; the column's cheapest fix is one voxel put back, so
    that voxel is in no list (the aseg had it), and every other removed
    voxel is listed as EDGE."""
    a = np.zeros((32, 28, 24), np.int16)
    a[4:20, 4:20, 2:18] = data.ASEG_CORTEX
    a[6:18, 6:18, 4:16] = data.ASEG_WM
    a[11:13, 11:13, 4:16] = data.ASEG_CORTEX
    nib.save(
        nib.nifti1.Nifti1Image(a, np.diag([V, V, V, 1.0])), data.aseg_path(tmp_path)
    )
    top = np.argwhere(a[:, :, 15:16] == data.ASEG_WM) + [0, 0, 15]
    col = np.array([[8, 8, z] for z in range(4, 15)])
    removed = np.concatenate([top, col]).astype(np.int32)
    ch = white.wm_changes(tmp_path, removed=removed)
    assert ch.betti_before == (1, 2, 0) and ch.betti_after == (1, 0, 0)
    edge = {tuple(p) for p in ch.voxels[ch.kind == white.EDGE]}
    assert {tuple(p) for p in top} <= edge
    assert len(edge) == len(removed) - 1
    back = {tuple(p) for p in removed} - edge
    assert len(back) == 1 and next(iter(back))[:2] == (8, 8)
    wm = white.white_matter_box((0, 0, 0), a.shape, ch, tmp_path)
    assert kuhn_betti(wm) == (1, 0, 0)
    # every listed voxel changes the aseg's label, and the rest are as before
    was = a == data.ASEG_WM
    listed = np.zeros(a.shape, bool)
    listed[tuple(ch.voxels.T)] = True
    assert np.array_equal(wm[~listed], was[~listed])
    assert (wm[listed] != was[listed]).all()


# ---------------------------------------------------------------- step 4: refine


def _plane_refine(
    xe: float, s: float = 0.24, margin: float = M
) -> tuple[F64, F64, Bool, Bool]:
    """WM x <= 20, its mesh (the plane x = 20.5), the image's edge at xe.
    Returns the old and new x and the flags, and the interior vertices (away
    from the mesh's border)."""
    shape = (40, 16, 16)
    x = _x(shape)
    wm = x <= 20
    v, f = white_mesh(wm)
    n = smooth_normals(v, f)
    r = refine(
        v, f, n, EdgeField(_img((x - xe) * V, s), VOX), VOX, wm, margin_mm=margin
    )
    inner = np.all(np.abs(v[:, 1:] - 7.5) < 5, axis=1)
    return v[:, 0], r.v[:, 0], r.flagged, inner


def test_refine_plane() -> None:
    """Edge at 20.9: the target is 20.9 - 0.625 = 20.275, 0.225 voxel inside
    the mesh; every vertex moves there. Edge at 21.3: the target is outside
    the mesh, so nothing moves (never outward)."""
    x0, x1, flagged, inner = _plane_refine(20.9)
    assert np.allclose(x0[inner], 20.5)
    assert np.abs(x1[inner] - 20.275).max() < 0.003 / V
    assert not flagged.any()
    x0, x1, flagged, _ = _plane_refine(21.3)
    assert np.array_equal(x0, x1) and not flagged.any()


def test_refine_margin_zero_moves_only_past_the_edge() -> None:
    """Margin 0: edge at 20.6, outside the mesh (20.5): nothing moves. Edge
    at 20.3: every vertex moves onto it."""
    x0, x1, _, _ = _plane_refine(20.6, margin=0.0)
    assert np.array_equal(x0, x1)
    _, x1, _, inner = _plane_refine(20.3, margin=0.0)
    assert np.abs(x1[inner] - 20.3).max() < 0.003 / V


def test_refine_keeps_every_vertex_at_least_the_margin_in() -> None:
    """A plane mesh (x = 20.5) under an edge at 20.9 with a narrow dimple
    (to 20.55 over about a voxel): smoothing alone would spread the dimple's
    deeper target over its neighbours and leave the dimple's own vertices
    with less than the margin. Each vertex's move is at least its own
    target, so every vertex ends at least the margin inside the edge it
    measured; and the smoothing does move the dimple's neighbours (more is
    allowed)."""
    shape = (40, 24, 24)
    X = np.indices(shape).astype(float)
    rr2 = (X[1] - 11.7) ** 2 + (X[2] - 12.2) ** 2
    xe = 20.9 - 0.35 * np.exp(-rr2 / 2.0)
    wm = X[0] <= 20
    v, f = white_mesh(wm)
    n = smooth_normals(v, f)
    r = refine(v, f, n, EdgeField(_img((X[0] - xe) * V), VOX), VOX, wm)
    own = np.minimum(r.offsets.offset_mm - M, 0.0)
    inner = np.all(np.abs(v[:, 1:] - 11.5) < 8, axis=1) & ~r.short
    assert (r.move_mm[inner] <= own[inner] + 1e-12).all()
    assert own[inner].min() < -0.04  # the dimple needs a real move
    assert (r.move_mm[inner] < own[inner] - 0.002).any()  # neighbours go deeper
    assert not r.short[np.all(np.abs(v[:, 1:] - 11.5) < 8, axis=1)].any()


def test_refine_caps_at_half_a_voxel() -> None:
    """Edge at 20.3: the target is 0.825 voxel inside, but a vertex moves at
    most half a voxel, so it never passes a WM voxel centre and the mesh and
    the label still agree about which centres are WM."""
    _, x1, _, inner = _plane_refine(20.3)
    assert np.all(x1[inner] > 20.0)
    assert np.abs(x1[inner] - 20.0).max() < 1e-3


def test_refine_flags_the_inner_end() -> None:
    """Edge at 18.0, 0.4 mm inside the mesh: no edge in the window. The
    vertices are flagged and not moved."""
    x0, x1, flagged, inner = _plane_refine(18.0)
    assert flagged[inner].all()
    assert np.array_equal(x0, x1)


def test_refine_guards_a_thin_blade() -> None:
    """A WM blade two voxels thick (x 20-21, mesh at 19.5 and 21.5) under an
    image whose WM is a thin band at its middle: both sides want to move far
    inward. The guard caps each at half of (the blade's half-thickness, 1
    voxel to the background centre, less half a voxel), 0.04 mm, so the
    sides stay apart."""
    shape = (40, 16, 16)
    x = _x(shape)
    wm = (x >= 20) & (x <= 21)
    v, f = white_mesh(wm)
    n = smooth_normals(v, f)
    d = (np.abs(x - 20.5) - 0.2) * V
    r = refine(v, f, n, EdgeField(_img(d, 0.06), VOX), VOX, wm, margin_mm=M)
    inner = np.all(np.abs(v[:, 1:] - 7.5) < 5, axis=1)
    move = np.einsum("ij,ij->i", r.v - v, n) * V
    np.testing.assert_allclose(move[inner], -0.04, atol=1e-6)
    assert not intersecting_faces(r.v, f)[1]


def test_refine_ball() -> None:
    """A ball label (radius 12.3 voxels) under an image whose edge is a
    sphere of radius 12.7: each vertex outside the sphere of radius 12.7 -
    0.625 lands on it, and each inside it (the binary contour's flats on the
    axes reach 11.93) stays, since nothing moves outward. The measured edge
    is the sphere less the derivative scale's scale² H (0.0049 mm here); what
    is left is the smoothing's residual where the vertices step from the
    flats onto the sphere, under 0.006 mm. The faces are unchanged and do
    not intersect."""
    n_ = 48
    c = np.array([23.7, 24.2, 23.9])
    X = np.indices((n_, n_, n_)).transpose(1, 2, 3, 0)
    r = np.linalg.norm(X - c, axis=-1)
    wm = r < 12.3
    v, f = white_mesh(wm)
    n = smooth_normals(v, f)
    res = refine(v, f, n, EdgeField(_img((r - 12.7) * V), VOX), VOX, wm, margin_mm=M)
    rr = np.linalg.norm(res.v - c, axis=1)
    r0 = np.linalg.norm(v - c, axis=1)
    shift = SCALE_MM**2 / (12.7 * V)
    target = 12.7 - (M + shift) / V
    err = (rr - np.minimum(r0, target)) * V
    assert np.median(np.abs(err)) < 0.001
    assert np.abs(err).max() < 0.006
    assert not res.flagged.any()
    assert not intersecting_faces(res.v, f)[1]


# ---------------------------------------------------------------- all steps


def _write_subject(d: Path) -> None:
    """A synthetic subject: a WM ball (radius 10 voxels) in a cortex shell
    (to 16), and three echoes whose WM edge is a sphere of radius 9.4, so
    the label and its mesh lie past the image's edge, in the cortex."""
    shape = (64, 64, 64)
    X = np.indices(shape).transpose(1, 2, 3, 0)
    r = np.linalg.norm(X - 31.5, axis=-1)
    a = np.zeros(shape, np.int16)
    a[r < 16] = data.ASEG_CORTEX
    a[r < 10] = data.ASEG_WM
    aff = np.diag([V, V, V, 1.0])
    nib.save(nib.nifti1.Nifti1Image(a, aff), data.aseg_path(d))
    step = 0.5 * (1 + _img((r - 9.4) * V))
    for k in (1, 2, 3):
        echo = np.where(r < 16, 1000 + 400 * step, 50).astype(np.int16)
        p = data.echo_path(k, d)
        nib.save(nib.nifti1.Nifti1Image(echo, aff), p)
        side = p.with_name(p.name.removesuffix(".nii.gz") + ".json")
        side.write_text('{"EchoTime": %g}' % (0.005 * k))


def test_all_steps_on_a_synthetic_subject(tmp_path: Path) -> None:
    """Steps 2-4 through the package: the peel removes the WM voxels whose
    centres lie less than the margin inside the image's sphere (r > 9.4 -
    0.625 = 8.775; within 0.05 voxel of that radius either way is allowed,
    the derivative scale's curvature shift being 0.03 voxel); handle removal
    changes nothing more; in the box, every refined vertex lies at least the
    margin inside the edge measured again (to 0.02 mm), none past it, and no
    faces cross; the files are written."""
    from baillarger.wm_edge import box_edges, edge_peel, write_box

    _write_subject(tmp_path)
    p = edge_peel(tmp_path)
    X = np.indices((64, 64, 64)).transpose(1, 2, 3, 0)
    r = np.linalg.norm(X - 31.5, axis=-1)
    assert p.sign == 1.0 and not len(p.flagged)
    got = np.zeros(r.shape, bool)
    got[tuple(p.removed.T)] = True
    cut = 9.4 - M / V
    assert (r[got] > cut - 0.05).all()
    assert got[(r < 10) & (r > cut + 0.05)].all()
    ch = white.wm_changes(tmp_path, removed=p.removed)
    assert ch.betti_after == (1, 0, 0) and (ch.kind == white.EDGE).all()
    be = box_edges((4, 4, 4), (60, 60, 60), ch, tmp_path)
    ref = be.report["refined"]
    assert ref["after_inside_frac"] == 0.0
    assert ref["after_short_of_margin_frac"] == 0.0
    assert ref["intersecting_faces_after"] == 0
    write_box(be, tmp_path / "out", "wm_edge_ball")
    for suffix in (".json", ".npz", ".vtk", "_refined.vtk", "_overlay.nii.gz"):
        assert (tmp_path / "out" / f"wm_edge_ball{suffix}").exists()
