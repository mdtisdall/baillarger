"""Tests of the surface preview of label edits (docs/plans/label-editing.md, 4b):
baillarger.viewer.surface_preview and its routes in baillarger.viewer.review, on a
phantom made by the real commands (white_dseg, then white_surf with blocks of 8
cells) and the real choices (the correction is white_dseg.wm_topo)."""

import json
import shutil
import threading
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest
from scipy import ndimage as ndi

from baillarger import antialias as AA
from baillarger import backend, white_dseg, white_surf
from baillarger import label_edits as LE
from baillarger.blocks import ArraySource
from baillarger.mesh import marching_tetrahedra_blocks
from baillarger.viewer import review as VR
from baillarger.viewer import server as VS
from baillarger.viewer import surface_preview as SP
from baillarger.viewer.columns import decode
from baillarger.viewer.outlines import plane_cut

AFF = np.array([[0.2, 0, 0, -3.0], [0, 0.16, 0, 2.0], [0, 0, 0.2, 5.0], [0, 0, 0, 1.0]])
CELLS = 8
#: a point on the WM ball's surface (radius about 12)
BUMP_AT = (20, 19, 29)


def write_dseg(d: Path) -> Path:
    """A right hemisphere: a WM ball round a ventricle, in cortex, and a separate
    WM block (as tests/test_white_surf.py); through the dseg command."""
    x = np.indices((44, 40, 36)).astype(float)
    r = np.sqrt(((x - np.array([20.3, 19.6, 17.8])[:, None, None, None]) ** 2).sum(0))
    a = np.zeros(x.shape[1:], np.int16)
    a[r < 15] = 42
    a[r < 12] = 41
    a[r < 4] = 43
    a[36:42, 4:12, 4:30] = 41
    src = d / "in" / "sub-T01_ses-S1_aseg.nii.gz"
    src.parent.mkdir(parents=True)
    nib.save(nib.nifti1.Nifti1Image(a, AFF), src)
    white_dseg.main([str(src), "--out", str(d / "deriv")])
    return d / "deriv/sub-T01/ses-S1/anat/sub-T01_ses-S1_desc-wmtopo_dseg.nii.gz"


@pytest.fixture(scope="module")
def made(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    d = tmp_path_factory.mktemp("sp")
    dseg = write_dseg(d)
    white_surf.main([str(dseg), "--out", str(d / "deriv"), "--block-cells", str(CELLS)])
    surf = d / "deriv/sub-T01/ses-S1/anat/sub-T01_ses-S1_hemi-R_white.surf.gii"
    return dseg, surf


@pytest.fixture
def files(made: tuple[Path, Path], tmp_path: Path) -> dict[str, Path]:
    """A copy of the phantom's outputs per test (a test may write edits)."""
    src = made[0].parents[4]
    dst = tmp_path / "copy"
    shutil.copytree(src, dst)
    rel = made[0].relative_to(src)
    dseg = dst / rel
    surf = dst / made[1].relative_to(src)
    # the dseg's sidecar names the input labels by absolute path: the copy's
    side = json.loads(dseg.with_name(dseg.name.replace(".nii.gz", ".json")).read_text())
    assert Path(side["Sources"][0]).exists()
    edits = LE.bids_path(dst / "edits", dict(sub="T01", ses="S1"), "R")
    return dict(dseg=dseg, surf=surf, edits=edits)


def labels_of(dseg: Path) -> npt.NDArray[Any]:
    return np.asarray(nib.nifti1.load(dseg).dataobj)


def make(files: dict[str, Path], surface: Path | None = None) -> VR.Review:
    lab = labels_of(files["dseg"])
    return VR.Review(
        None,
        None,
        files["edits"],
        files["dseg"],
        lambda: lab,
        AFF,
        None,
        surface=files["surf"] if surface is None else surface,
        device="cpu",
    )


@pytest.fixture
def review(files: dict[str, Path]) -> Iterator[VR.Review]:
    rv = make(files)
    try:
        yield rv
    finally:
        rv.close()


def done(rv: VR.Review, out: dict[str, Any], since: int = -1) -> dict[str, Any]:
    """The job's outcome once it is not running."""
    t0 = time.time()
    while out["status"] == "running":
        assert time.time() - t0 < 300
        time.sleep(0.05)
        out = rv.surface_job(out["job"], since)
    assert out["status"] == "done", out
    return out


def bump(r: float = 2.0, at: tuple[int, ...] = BUMP_AT) -> npt.NDArray[np.int64]:
    """The voxels of a ball on the WM ball's surface."""
    x = np.indices((44, 40, 36)).astype(float)
    d = np.sqrt(((x - np.array(at)[:, None, None, None]) ** 2).sum(0))
    return np.argwhere(d <= r).astype(np.int64)


def strokes(voxels: npt.NDArray[np.int64], label: Any = 41) -> list[dict[str, Any]]:
    return [dict(label=label, voxels=voxels.tolist())]


def whole(rv: VR.Review) -> tuple[slice, ...]:
    return tuple(slice(0, n) for n in rv.shape)


def test_groups_join_voxels_in_neighbouring_cells() -> None:
    v = np.array([[0, 0, 0], [40, 0, 0], [5, 5, 5], [200, 0, 0], [70, 0, 0]])
    got = sorted(sorted(g.tolist()) for g in SP.groups(v, 32))
    assert got == [[0, 1, 2, 4], [3]]
    assert SP.groups(np.zeros((0, 3), np.int64)) == []


def test_a_preview_leaves_the_working_field_and_a_paint_changes_it_only_in_its_box(
    review: VR.Review,
) -> None:
    rv = review
    first = done(rv, rv.surface_preview([]))
    assert first["revision"] == 0 and first["blocks"] == [] and first["radius"] is None
    assert rv._sp is not None
    before = np.array(rv._sp.state.field.read(whole(rv)))
    v = bump()
    out = done(rv, rv.surface_preview(strokes(v)))
    assert out["radius"] >= AA.PREVIEW_MIN_RADIUS and out["blocks"]
    assert out["revision"] == 0 and out["working"] == []
    preview_field = np.array(rv.sjobs.get(out["job"])["field"].read(whole(rv)))
    # the preview did not change the working field
    assert np.array_equal(rv._sp.state.field.read(whole(rv)), before)
    assert rv._sp.state.revision == 0 and not rv._sp.state.blocks
    # the paint: the next surface preview takes the working field to it
    rv.paint(strokes(v))
    after = done(rv, rv.surface_preview([]))
    assert after["revision"] == 1 and len(after["kept"]) == 1
    assert after["blocks"] == [] and sorted(after["working"]) == sorted(out["blocks"])
    work = np.array(rv._sp.state.field.read(whole(rv)))
    box = tuple(slice(a, b) for a, b in after["kept"][0]["box"])
    outside = np.ones(rv.shape, bool)
    outside[box] = False
    assert np.array_equal(work[outside], before[outside])
    assert not np.array_equal(work[box], before[box])
    # the preview of the strokes is the working field after their paint
    assert np.array_equal(work, preview_field)


def test_the_preview_equals_the_full_solve_near_the_edit(review: VR.Review) -> None:
    """docs/preview-radius.md, through the viewer: the preview's field is within
    tau = 0.01 voxel of the full solve of the edited mask near the edit."""
    rv = review
    v = bump()
    out = done(rv, rv.surface_preview(strokes(v)))
    r = rv.sjobs.get(out["job"])
    assert rv._sp is not None
    field = np.array(r["field"].read(whole(rv)))
    lab = labels_of(rv.dseg).copy()
    lab[tuple(v.T)] = 41
    mask = np.isin(lab, rv._sp.filled)
    full, solve = AA.thin_plate(mask, backend.device("cpu"))
    assert solve.converged
    stored = np.array(rv._sp.state.field.read(whole(rv)))
    changed = mask != (stored > 0)
    assert changed.any() and np.array_equal(mask, field > 0)
    dist = ndi.distance_transform_edt(~changed)
    near = (dist <= AA.BAND + 3) & (np.abs(AA.signed_distance(mask, 8.0)) <= AA.BAND)
    assert np.abs(full - stored)[near].max() > 0.3  # the edit moves the field
    assert np.abs(field - full)[near].max() < AA.PREVIEW_TAU


def test_an_edit_with_no_change_of_the_wm_mask_gives_no_new_blocks(
    review: VR.Review,
) -> None:
    rv = review
    lab = labels_of(rv.dseg)
    cx = np.argwhere(lab == 42)
    far = cx[np.argmax(np.abs(cx - np.array(BUMP_AT)).sum(1))][None]
    out = done(rv, rv.surface_preview(strokes(far, 0)))
    assert out["blocks"] == [] and out["radius"] is None and out["solves"] == []


def test_an_undo_takes_the_working_field_back_near_the_stored_field(
    review: VR.Review,
) -> None:
    rv = review
    done(rv, rv.surface_preview([]))
    assert rv._sp is not None
    stored = np.array(rv._sp.state.field.read(whole(rv)))
    rv.paint(strokes(bump()))
    done(rv, rv.surface_preview([]))
    rv.undo_last()
    out = done(rv, rv.surface_preview([]))
    assert out["revision"] == 2
    work = np.array(rv._sp.state.field.read(whole(rv)))
    assert np.array_equal(work > 0, stored > 0)
    assert not len(rv._sp.state.overlay)
    band = np.abs(stored) <= AA.BAND
    assert np.abs(work - stored)[band].max() < AA.PREVIEW_TAU


def test_the_working_field_starts_from_the_pending_edits(
    files: dict[str, Path],
) -> None:
    """At start the working field is the stored field with the pending edits
    solved in: the same field as a paint of them in a viewer run."""
    v = bump()
    a = make(files)
    try:
        done(a, a.surface_preview([]))
        a.paint(strokes(v))
        done(a, a.surface_preview([]))
        assert a._sp is not None
        want = np.array(a._sp.state.field.read(whole(a)))
    finally:
        a.close()
    b = make(files)  # the paint is now a pending edit
    try:
        out = done(b, b.surface_preview([]))
        assert len(out["kept"]) == 1 and out["revision"] == 1
        assert b._sp is not None
        assert np.array_equal(b._sp.state.field.read(whole(b)), want)
    finally:
        b.close()


def test_a_stop_between_groups_leaves_a_state_that_a_later_change_continues(
    review: VR.Review,
) -> None:
    rv = review
    done(rv, rv.surface_preview([]))
    sp = rv._sp
    assert sp is not None
    lab = labels_of(rv.dseg)
    two = np.concatenate([bump(1.5), bump(1.5, (20, 19, 6))])
    after, _, _ = rv._painted(LE.empty(), two, np.full(len(two), 41))
    sp.group = 8
    assert len(SP.groups(two, sp.group)) == 2
    st = sp.state.copy()
    with pytest.raises(SP.Stopped):
        sp.change(st, after, rv._surface_pending, rv._surface_region, lambda: True)
    assert st.revision == 0 and not len(st.applied.index)
    calls = iter([False, True])
    with pytest.raises(SP.Stopped):
        sp.change(
            st, after, rv._surface_pending, rv._surface_region, lambda: next(calls)
        )
    assert st.revision == 1 and 0 < len(st.applied.index) < len(after.index)
    sp.change(st, after, rv._surface_pending, rv._surface_region)
    full = sp.state.copy()
    sp.change(full, after, rv._surface_pending, rv._surface_region)
    assert np.array_equal(st.applied.index, after.index)
    assert np.array_equal(st.field.read(whole(rv)), full.field.read(whole(rv)))
    assert np.array_equal(
        SP.WorkingMask(lab, st.overlay, sp.filled).read(whole(rv)),
        np.asarray(full.field.read(whole(rv))) > 0,
    )


def test_the_blocks_are_the_march_of_the_preview_field(review: VR.Review) -> None:
    """The preview's blocks are those of a full march of its field, and the
    mesh route gives them."""
    rv = review
    out = done(rv, rv.surface_preview(strokes(bump())))
    r = rv.sjobs.get(out["job"])
    head, arr = decode(rv.surface_mesh(out["job"]))
    assert [b[:3] for b in head["blocks"]] == out["blocks"]
    assert all(b[5] == 1 for b in head["blocks"])
    v, f, mb = marching_tetrahedra_blocks(r["field"], 0.0, CELLS, np.float32)
    o, kv, kf = 0, 0, 0
    for i, j, k, nv, nf, _ in head["blocks"]:
        hit = np.flatnonzero(np.all(mb.origin // CELLS == [i, j, k], axis=1))
        if not nf:
            assert not len(hit)
            continue
        ff = f[mb.face_start[hit[0]] : mb.face_stop[hit[0]]][:, [0, 2, 1]]
        bv, bf = arr["v"].reshape(-1, 3)[kv : kv + nv], arr["f"].reshape(-1, 3)
        bf = bf[kf : kf + nf]
        assert np.array_equal(bv[bf], v[ff])
        kv, kf, o = kv + nv, kf + nf, o + 1
    assert o


@pytest.fixture
def server(files: dict[str, Path]) -> Iterator[tuple[str, VR.Review]]:
    rv = make(files)
    srv = VS.serve(None, 0, white=object(), wmreview=rv)  # type: ignore[arg-type]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", rv
    finally:
        srv.shutdown()
        rv.close()
        VS.Handler.white = None
        VS.Handler.wmreview = None


def fetch(url: str, body: dict[str, Any] | None = None) -> bytes:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out: bytes = r.read()
        return out


def test_the_routes_give_the_blocks_their_mesh_and_their_cut(
    server: tuple[str, VR.Review],
) -> None:
    url, rv = server
    out = json.loads(
        fetch(f"{url}/review/surface-preview", dict(strokes=strokes(bump())))
    )
    while out["status"] == "running":
        time.sleep(0.05)
        out = json.loads(fetch(f"{url}/review/surface-preview?job={out['job']}"))
    assert out["status"] == "done" and out["blocks"] and out["radius"] >= 16
    head, arr = decode(fetch(f"{url}/review/surface-preview/mesh?job={out['job']}"))
    assert [b[:3] for b in head["blocks"]] == out["blocks"]
    # the cut at the bump's plane: the cut of the blocks' faces in world mm
    mm = float((AFF @ np.r_[np.asarray(BUMP_AT, float), 1.0])[2])
    seg = np.frombuffer(
        fetch(f"{url}/review/surface-preview/cut?job={out['job']}&axis=2&mm={mm}"),
        "<f4",
    ).reshape(-1, 2, 3)
    assert len(seg) and np.allclose(seg[:, :, 2], mm)
    v = arr["v"].reshape(-1, 3).astype(np.float64)
    w = v @ AFF[:3, :3].T + AFF[:3, 3]
    f = arr["f"].reshape(-1, 3).astype(np.int64)
    k0, kf, want = 0, 0, []
    for _, _, _, nv, nf, _ in head["blocks"]:
        want.append(plane_cut(w[k0 : k0 + nv], f[kf : kf + nf], 2, mm))
        k0, kf = k0 + nv, kf + nf
    assert np.allclose(seg, np.concatenate(want).astype(np.float32))
    # after a paint, the working blocks come once: since the page's revision
    rv.paint(strokes(bump()))
    a = json.loads(fetch(f"{url}/review/surface-preview", dict(strokes=[], since=0)))
    while a["status"] == "running":
        time.sleep(0.05)
        a = json.loads(fetch(f"{url}/review/surface-preview?job={a['job']}&since=0"))
    assert a["revision"] == 1 and sorted(a["working"]) == sorted(out["blocks"])
    b = json.loads(fetch(f"{url}/review/surface-preview?job={a['job']}&since=1"))
    assert b["working"] == []


def test_no_field_refuses_the_surface_preview(files: dict[str, Path]) -> None:
    fp = white_surf.field_path(files["surf"])
    fp.unlink()
    rv = make(files)
    try:
        out = rv.surface_preview([])
        out = out if out["status"] != "running" else done_or_failed(rv, out)
        assert out["status"] == "failed" and "no antialiased field" in out["error"]
    finally:
        rv.close()
    lab = labels_of(files["dseg"])
    no_surf = VR.Review(
        None, None, files["edits"], files["dseg"], lambda: lab, AFF, None
    )
    with pytest.raises(ValueError, match="no white surface"):
        no_surf.surface_preview([])


def done_or_failed(rv: VR.Review, out: dict[str, Any]) -> dict[str, Any]:
    while out["status"] == "running":
        time.sleep(0.05)
        out = rv.surface_job(out["job"])
    return out


def test_a_replaced_field_refuses_writes(files: dict[str, Path]) -> None:
    """The field is an output that regenerate replaces: then each write is
    refused, as for the dseg."""
    rv = make(files)
    try:
        fp = white_surf.field_path(files["surf"])
        tmp = fp.with_name("x.nii")
        shutil.copy(fp, tmp)
        tmp.replace(fp)
        with pytest.raises(ValueError, match="restart the viewer"):
            rv.paint(strokes(bump()))
        with pytest.raises(ValueError, match="restart the viewer"):
            rv.surface_preview([])
    finally:
        rv.close()


def test_a_working_mask_reads_the_labels_with_the_overlay() -> None:
    lab = np.zeros((6, 5, 4), np.int16)
    lab[1:4, 1:4, 1:3] = 41
    ov = LE.Edits(
        np.array([np.ravel_multi_index((0, 0, 0), lab.shape)], np.int64),
        np.array([41], np.int32),
    )
    m = SP.WorkingMask(lab, ov, [41])
    box = (slice(0, 2), slice(0, 2), slice(0, 2))
    want = np.isin(lab, [41])
    want[0, 0, 0] = True
    assert np.array_equal(m.read(box), want[box])
    assert np.array_equal(ArraySource(want).read(box), m.read(box))
