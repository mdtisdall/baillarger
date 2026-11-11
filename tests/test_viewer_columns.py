"""Tests of baillarger.viewer.columns (the columns pane's server side), with
exact answers, on a surface written by white_dseg, white_surf and
white_overview from a synthetic segmentation: the columns of a request are the
ray sampler's own, byte for byte; the halo is right at a superblock corner; the
radius is along the surface; the profiles, the TCK and the routes."""

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from baillarger import rays as R
from baillarger import white_dseg, white_overview, white_surf
from baillarger.blocks import open_volume
from baillarger.column_volume import ColumnVolumeSampler
from baillarger.data import echo_path
from baillarger.mesh_graph import edge_graph
from baillarger.viewer import columns as VC
from baillarger.viewer import section as VSec
from baillarger.viewer import server as VS
from baillarger.viewer.tck import read_tck
from baillarger.viewer.white_view import WhiteView
from baillarger.white_surf import WhiteSurface

AFF = np.array([[0.2, 0, 0, -3.0], [0, 0.16, 0, 2.0], [0, 0, 0.2, 5.0], [0, 0, 0, 1.0]])
CELLS = 8


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A right hemisphere: a WM ball round a ventricle, in cortex, in a grid
    of 8-voxel blocks (1.6 mm), with its overview and three echoes (echo k is
    k times the linear index of each voxel)."""
    d = tmp_path_factory.mktemp("cols")
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
    return dict(surf=surf, dseg=dseg, echoes=ed)


@pytest.fixture(scope="module")
def cols(data: dict[str, Path]) -> VC.Columns:
    W = WhiteSurface.load(data["surf"])
    return VC.Columns(
        data["surf"], data["dseg"], W.v, W.f, VC.echo_files(data["echoes"]), log=quiet
    )


def quiet(m: str) -> None:
    pass


def as_column(r: R.Rays, c: int) -> tuple[bytes, ...]:
    """Every value of column c of r, as bytes (tests/test_rays.py)."""
    a, b = r.sample_start[c], r.sample_start[c + 1]
    q0, q1 = r.interval_start[c], r.interval_start[c + 1]
    ro = r.roots
    return (
        ro.edge_key[c].tobytes(),
        ro.vertex[c].tobytes(),
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


def owner(cols: VC.Columns, rays: dict[tuple[int, int, int], R.Rays], v: int) -> Any:
    """Column v as the sampler gives it to the superblock (level 0) that owns
    it, with the extents e_in = 1, e_out = 3."""
    sb = cols.superblock(v)
    if sb not in rays:
        rays[sb] = cols.rays.sample(sb, 0, e_in=1.0, e_out=3.0)
    r = rays[sb]
    c = int(np.flatnonzero(r.roots.vertex == v)[0])
    assert r.roots.owned[c]
    return as_column(r, c)


def whole_graph_near(cols: VC.Columns, v: int, radius: float) -> set[int]:
    g = edge_graph(np.asarray(cols.v, float), cols.f, cols.vox)
    d = dijkstra(g, indices=v, limit=radius)
    return set(np.flatnonzero(np.isfinite(d)).tolist())


def corner_vertex(cols: VC.Columns) -> int:
    """The vertex nearest a corner of its block (in voxels)."""
    v = np.asarray(cols.v, float)
    f = v / CELLS - np.floor(v / CELLS)
    dist = np.linalg.norm(np.minimum(f, 1 - f), axis=1)
    return int(np.argmin(dist))


def test_the_columns_are_the_samplers_own_byte_for_byte(cols: VC.Columns) -> None:
    v = corner_vertex(cols)
    req = VC.Request(vertex=v, radius=1.0, e_in=1.0, e_out=3.0)
    res = cols.compute(req, quiet)
    got = res.rays
    assert len(got.roots.vertex) > 20
    ref: dict[tuple[int, int, int], R.Rays] = {}
    for c, u in enumerate(got.roots.vertex):
        assert as_column(got, c) == owner(cols, ref, int(u))
    # in the sampler's order: owned columns of v's superblock, then halo ones
    sb_roots = cols.rays.roots(res.superblock, 0, 1.0)
    order = {int(u): k for k, u in enumerate(sb_roots.vertex)}
    assert np.all(np.diff([order[int(u)] for u in got.roots.vertex]) > 0)
    # more than one superblock: the corner's radius crosses into its neighbours
    assert len({cols.superblock(int(u)) for u in got.roots.vertex}) > 1


def test_a_request_near_a_corner_gives_the_same_column_as_one_far_from_it(
    cols: VC.Columns,
) -> None:
    """The corner vertex's column, from a request centred on it (its own
    superblock with a halo) and from a request centred on a vertex of another
    superblock whose radius holds it: the same bytes."""
    v = corner_vertex(cols)
    a = cols.compute(VC.Request(vertex=v, radius=0.5), quiet).rays
    near = whole_graph_near(cols, v, 0.8)
    other = next(
        u
        for u in sorted(near)
        if cols.superblock(u) != cols.superblock(v)
        and v in whole_graph_near(cols, u, 1.0)
    )
    b = cols.compute(VC.Request(vertex=other, radius=1.0), quiet).rays
    ca = int(np.flatnonzero(a.roots.vertex == v)[0])
    cb = int(np.flatnonzero(b.roots.vertex == v)[0])
    assert a.roots.owned[ca] and not b.roots.owned[cb]
    assert as_column(a, ca) == as_column(b, cb)


@pytest.mark.parametrize("radius", [0.0, 0.3, 1.0, 2.5])
def test_the_radius_is_along_the_surface(cols: VC.Columns, radius: float) -> None:
    v = corner_vertex(cols)
    got, d = cols.near(v, radius)
    assert set(got.tolist()) == whole_graph_near(cols, v, radius)
    g = edge_graph(np.asarray(cols.v, float), cols.f, cols.vox)
    full = dijkstra(g, indices=v, limit=radius)
    assert np.array_equal(d, full[got])
    res = cols.compute(VC.Request(vertex=v, radius=radius), quiet)
    assert sorted(res.rays.roots.vertex.tolist()) == sorted(got.tolist())
    assert np.array_equal(np.sort(res.distance), np.sort(full[got].astype(np.float32)))


def test_every_and_the_limit(cols: VC.Columns, monkeypatch: pytest.MonkeyPatch) -> None:
    v = corner_vertex(cols)
    all_ = cols.compute(VC.Request(vertex=v, radius=1.5), quiet).rays.roots.vertex
    third = cols.compute(VC.Request(vertex=v, radius=1.5, every=3), quiet)
    assert np.array_equal(third.rays.roots.vertex, all_[::3])
    monkeypatch.setattr(VC, "MAX_COLUMNS", len(all_) - 1)
    with pytest.raises(ValueError, match="more than"):
        cols.compute(VC.Request(vertex=v, radius=1.5), quiet)
    for bad in (dict(radius="9"), dict(method="x"), dict(every="0"), dict(e_out="9")):
        with pytest.raises(ValueError):
            VC.Request.parse(dict(v="0", **bad))


def test_select_columns_keeps_each_column(cols: VC.Columns) -> None:
    sb = cols.superblock(corner_vertex(cols))
    r = cols.rays.sample(sb, 0, e_out=2.0)
    idx = [5, 0, 3]
    s = VC.select_columns(r, idx)
    for k, c in enumerate(idx):
        assert as_column(s, k) == as_column(r, c)
    assert len(VC.select_columns(r, []).s) == 0


def test_the_encoding_round_trips() -> None:
    arrays: dict[str, npt.NDArray[Any]] = dict(
        a=np.arange(5, dtype=np.int64),
        b=np.ones((3, 3), np.float32),
        c=np.array([True, False]),
        d=np.arange(3, dtype=np.uint8),
    )
    head, back = VC.decode(VC.encode(dict(x=1), arrays))
    assert head["x"] == 1
    for k, a in arrays.items():
        assert np.array_equal(back[k], a) and back[k].shape == a.shape
    assert all(t["offset"] % 8 == 0 for t in head["arrays"])


def get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return int(r.status), bytes(r.read())
    except urllib.error.HTTPError as e:
        return int(e.code), bytes(e.read())


def settled(url: str) -> tuple[int, bytes]:
    """GET url as the page does: while the answer is 202 (the columns job runs,
    longer than WAIT_S on a slow runner), wait 0.1 s and ask again; at most
    60 s."""
    t0 = time.monotonic()
    code, body = get(url)
    while code == 202 and time.monotonic() - t0 < 60:
        time.sleep(0.1)
        code, body = get(url)
    return code, body


def test_the_routes(
    data: dict[str, Path],
    cols: VC.Columns,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """/columns gives compute's columns; /columns/profiles the column-volume
    sampler's echoes of them; /columns/<key>.tck their positions; a slow
    request answers 202 and then the columns. The first requests follow that
    contract as the page does (202: wait for the job, ask again), so that a
    slow runner, where the first job takes longer than WAIT_S, gives the same
    answer."""
    w = WhiteView(data["surf"], tmp_path / "cache", log=quiet)
    srv = VS.serve(None, 0, white=w, columns=cols)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        v = corner_vertex(cols)
        q = f"v={v}&radius=1&e_in=0.5&e_out=2"
        code, body = settled(f"{base}/columns?{q}")
        assert code == 200
        head, arr = VC.decode(body)
        ref = cols.compute(VC.Request(vertex=v, radius=1.0, e_in=0.5, e_out=2.0))
        for k, a in VC.ray_arrays(ref.rays).items():
            assert (
                arr[k].tobytes()
                == np.ascontiguousarray(a)
                .astype(np.uint8 if a.dtype == np.bool_ else a.dtype)
                .tobytes()
            ), k
        assert head["n"] == len(ref.rays.roots.vertex) and head["key"] == ref.key
        code, body = settled(f"{base}/columns/profiles?{q}")
        assert code == 200
        _, pa = VC.decode(body)
        srcs = [open_volume(p) for p in VC.echo_files(data["echoes"])]
        smp = ColumnVolumeSampler(srcs, cols.rays.affine, [AFF] * 3, cols.rays.shape)  # type: ignore[arg-type]
        cv = smp.sample(ref.rays, "numpy")
        assert pa["echo"].tobytes() == cv.echo.tobytes()
        assert np.array_equal(pa["valid"].astype(bool), cv.valid)
        lines = read_tck(get(f"{base}/columns/{ref.key}.tck")[1])
        st = ref.rays.sample_start
        assert len(lines) == len(st) - 1
        for c, line in enumerate(lines):
            assert np.array_equal(line, ref.rays.position[st[c] : st[c + 1]])
        meta = json.loads(get(f"{base}/columns/meta")[1])
        assert meta["max_columns"] == VC.MAX_COLUMNS and meta["methods"] == list(
            R.METHODS
        )
        code, body = get(f"{base}/columns?v={v}&radius=99")
        assert code == 400 and "radius" in json.loads(body)["error"]

        gate = threading.Event()
        real = cols.compute

        def slow(req: VC.Request, log: Any = quiet) -> VC.Result:
            gate.wait(10)
            return real(req, log)

        monkeypatch.setattr(cols, "compute", slow)
        monkeypatch.setattr(VC, "WAIT_S", 0.05)
        q2 = f"v={v}&radius=0.7"
        code, body = get(f"{base}/columns?{q2}")
        assert code == 202 and json.loads(body)["status"] == "running"
        code, body = get(f"{base}/columns?v={v}&radius=0.6")
        assert code == 202 and json.loads(body)["status"] == "busy"
        gate.set()
        assert cols.jobs.thread is not None
        cols.jobs.thread.join(10)
        code, body = get(f"{base}/columns?{q2}")
        assert code == 200 and VC.decode(body)[0]["n"] > 0
    finally:
        srv.shutdown()
        srv.server_close()
        VS.Handler.white = None
        VS.Handler.columns = None


def test_no_echoes_no_profiles(data: dict[str, Path], tmp_path: Path) -> None:
    assert VC.echo_files(tmp_path) == []
    W = WhiteSurface.load(data["surf"])
    c = VC.Columns(data["surf"], data["dseg"], W.v, W.f, [], log=quiet)
    key = c.compute(VC.Request(vertex=0, radius=0.2)).key
    c.jobs.remember(key, c.compute(VC.Request(vertex=0, radius=0.2)))
    with pytest.raises(ValueError, match="no echoes"):
        c.profile(key)


def test_the_dseg_next_to_the_surface(data: dict[str, Path]) -> None:
    side = WhiteSurface.load(data["surf"]).sidecar
    assert VC.dseg_for(data["surf"], side) == data["dseg"]


def test_the_white_mode_serves_every_route(
    data: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After the removal of the site mode: main() starts the server in --white
    mode, and the server answers each route of phases 4.0 (the white surface),
    4.0.5 (the bricks, the named volumes, the outlines), 4.3 (the columns) and
    the viewer requests of wave 2.1 (go to a true vertex, the section that
    follows the surface). The routes of NiiVue's box are gone (404)."""
    ed = tmp_path / "data"
    ed.mkdir()
    for k in (1, 2, 3):  # echo k as .nii.gz in data: the defaults of main()
        e = nib.load(echo_path(k, data["echoes"]).with_suffix(""))
        nib.save(e, echo_path(k, ed))
    monkeypatch.setenv("BAILLARGER_DATA", str(ed))
    made: list[Any] = []
    real = VS.serve

    def capture(*a: Any, **k: Any) -> Any:
        made.append(real(*a, **k))
        return made[-1]

    monkeypatch.setattr(VS, "serve", capture)
    argv = ["--white", str(data["surf"]), "--cache", str(tmp_path / "cache")]
    threading.Thread(
        target=VS.main, args=(argv + ["--port", "0"],), daemon=True
    ).start()
    for _ in range(600):
        if made:
            break
        threading.Event().wait(0.1)
    assert made, "the server did not start"
    srv = made[0]
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        meta = json.loads(get(f"{base}/meta")[1])
        assert meta["surface_only"]
        lo, hi = "-9,-9,-9", "9,9,9"
        routes = [
            "/",
            "/?vol=bricks",  # old links
            *(f"/static/{f.name}" for f in sorted(VS.STATIC.iterdir())),
            "/state",
            "/mesh?level=0&faces=1",
            "/vertex?v=0",
            "/snap?mm=0,5,9",
            "/white.gii",
            "/volume",
            "/volume/meta",
            "/volume/bricks?level=0&b=0,0,0",
            "/volume/list",
            f"/outline?axis=2&mm=9&box={lo},{hi}",
            "/columns/meta",
            "/columns?v=0&radius=0.5",
            "/columns/profiles?v=0&radius=0.5",
            "/columns/section?v=0&radius=0.5&u=1,0,0",
            "/columns/section?v=0&radius=0.5&u=1,0,0&mode=mean",
            "/vertex/true?v=0",
            "/vertex/overview?v=0,1,2",
        ]
        for r in routes:
            assert settled(base + r)[0] == 200, r
        for name in json.loads(get(f"{base}/volume/list")[1]):
            assert get(f"{base}/volume/{name['name']}/meta")[0] == 200, name
        key = VC.decode(settled(f"{base}/columns?v=0&radius=0.5")[1])[0]["key"]
        assert get(f"{base}/columns/{key}.tck")[0] == 200
        for r in [
            "/overlay/meta",
            "/overlay/a/b",
            "/overlay/a/b/sites",
            "/overlay/at?v=0",
        ]:
            assert get(base + r)[0] == 404, r  # without --overlay
        for gone in [
            "/filters",
            "/section?i=0",
            "/layer?name=x",
            "/labels/list",
            *(f"/white/echo{k}.nii.gz?c=1,1,1" for k in (1, 2, 3)),
            "/white/surface.gii?c=1,1,1",
            "/white/overlay.nii.gz?c=1,1,1",
        ]:
            assert get(base + gone)[0] == 404, gone
    finally:
        srv.shutdown()
        srv.server_close()
        VS.Handler.white = None
        VS.Handler.volume = None
        VS.Handler.volumes = None
        VS.Handler.outlines = None
        VS.Handler.columns = None
        VS.Handler.overlays = None


def test_the_section_route(cols: VC.Columns) -> None:
    """GET /columns/section on the hemisphere phantom: the steps every half
    voxel, in order of arc length from 0 at the
    selected root; each column an index into the /columns answer, the nearest
    along the surface (checked on the whole graph); the errors."""
    srv = VS.serve(None, 0, white=WhiteView.__new__(WhiteView), columns=cols)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        v, radius = corner_vertex(cols), 1.0
        q = f"v={v}&radius={radius}"
        code, body = settled(f"{base}/columns?{q}")
        assert code == 200
        head, arr = VC.decode(body)
        code, body = get(f"{base}/columns/section?{q}&u=0.3,1,0.2")
        assert code == 200
        sec = json.loads(body)
        assert sec["key"] == head["key"] and sec["vertex"] == v
        assert int(arr["vertex"][sec["column"]]) == v
        arc = np.asarray(sec["arc"])
        assert np.diff(arc) == pytest.approx(sec["step"])
        assert 0.0 in arc.round(9)
        assert -radius - 1e-9 <= arc[0] and arc[-1] <= radius + 1e-9
        # half a voxel (the largest side), whatever the spacing of the roots
        assert sec["step"] == pytest.approx(0.5 * float(max(cols.vox)))
        assert all(len(c) == 1 and 0 <= c[0] < head["n"] for c in sec["columns"])
        # the step at arc 0 is the selected root itself
        k0 = int(np.flatnonzero(arc.round(9) == 0)[0])
        assert sec["columns"][k0] == [sec["column"]] and sec["distance"][k0] == 0
        # the same section on the whole mesh and the whole graph (the route
        # reads only the local faces of the superblock)
        A = cols.rays.affine
        pts = np.asarray(cols.v, np.float64) @ A[:3, :3].T + A[:3, 3]
        g = edge_graph(cols.v, cols.f, cols.vox)
        colv = arr["vertex"].astype(np.int64)
        column_of = -np.ones(len(pts), np.int64)
        column_of[colv] = np.arange(len(colv))
        curve = VSec.plane_cut(
            pts, cols.f, pts[v], arr["normal"][sec["column"]], [0.3, 1, 0.2], radius
        )
        s_arc, s_pts, s_face = VSec.steps(curve, sec["step"])
        assert s_arc == pytest.approx(arc, abs=1e-9)
        assert np.abs(s_pts - np.asarray(sec["point"])).max() < 1e-4
        for mode in VSec.MODES:
            want, wd = VSec.step_columns(
                s_pts,
                s_face,
                pts,
                cols.f,
                g,
                cKDTree(pts),
                column_of,
                mode,
                float(max(cols.vox)),
                radius,
            )
            if mode == "nearest":
                assert sec["columns"] == want
                assert sec["distance"] == pytest.approx(wd.tolist(), abs=1e-9)
            else:
                mean_want = want
        # mean across: the columns within one voxel along the surface
        code, body = get(f"{base}/columns/section?{q}&u=0.3,1,0.2&mode=mean")
        assert code == 200
        mean = json.loads(body)
        assert mean["arc"] == sec["arc"] and mean["voxel"] == float(max(cols.vox))
        assert mean["columns"] == mean_want
        # the roots that the page highlights: the roots of the section's columns
        for a_sec in (sec, mean):
            want_roots = {int(colv[c]) for cs in a_sec["columns"] for c in cs}
            assert a_sec["roots"] == sorted(want_roots)
        # through another column of the request (a click on the section)
        other = int(arr["vertex"][sec["columns"][-1][0]])
        code, body = get(f"{base}/columns/section?{q}&u=0.3,1,0.2&at={other}")
        assert code == 200 and json.loads(body)["vertex"] == other
        # a picked column that is not in the request (a stale one from another
        # selection): the section falls back to the request's vertex, no error
        for stale in (
            -5,
            other + 10**9,
            int(np.setdiff1d(np.arange(len(cols.v)), colv)[0]),
        ):
            code, body = get(f"{base}/columns/section?{q}&u=0.3,1,0.2&at={stale}")
            assert code == 200
            fb = json.loads(body)
            assert fb["vertex"] == v and fb["column"] == sec["column"]
            assert fb["fallback"] == f"vertex {stale}: not a column of the request"
            assert fb["columns"] == sec["columns"] and fb["arc"] == sec["arc"]
        assert sec["fallback"] is None
        assert sec["step_vertices"] == [
            [int(colv[c]) for c in cs] for cs in sec["columns"]
        ]
        # a slice plane (a shift-click in a slice view): the curve is the
        # slice's cut (VSec.section_curve) on the whole mesh, or the fallback
        # where the slice is near the tangent plane (the plane normal to n)
        nrm = np.asarray(arr["normal"][sec["column"]], np.float64)
        for plane in ([0.2, 1.0, 0.6], nrm.tolist()):
            pq = ",".join(repr(float(x)) for x in plane)
            code, body = get(f"{base}/columns/section?{q}&u=0.3,1,0.2&plane={pq}")
            assert code == 200
            sl = json.loads(body)
            curve, why = VSec.section_curve(
                pts, cols.f, pts[v], nrm, [0.3, 1, 0.2], radius, sl["step"], plane
            )
            s_arc, s_pts, _ = VSec.steps(curve, sl["step"])
            assert s_arc == pytest.approx(sl["arc"], abs=1e-9)
            assert np.abs(s_pts - np.asarray(sl["point"])).max() < 1e-4
            pn = np.asarray(plane) / np.linalg.norm(plane)
            assert sl["slice"]["used"] == (why is None)
            assert sl["slice"]["fallback"] == why
            assert sl["slice"]["plane"] == pytest.approx(pn.tolist())
            angle = np.degrees(np.arcsin(abs(pn @ nrm) / np.linalg.norm(nrm)))
            assert sl["slice"]["angle"] == pytest.approx(angle)
            if plane[1] == 1.0:  # the slanted slice cuts through the root
                assert why is None
                assert np.abs((np.asarray(sl["point"]) - pts[v]) @ pn).max() < 1e-3
        assert sl["slice"]["angle"] == pytest.approx(90.0)
        assert sec["slice"] is None
        for bad in (
            "u=1,0",
            "u=0.3,1,0.2&mode=median",
            "u=0.3,1,0.2&plane=0,0,0",
            "u=0.3,1,0.2&plane=1,2",
            "",
        ):
            code, body = get(f"{base}/columns/section?{q}&{bad}")
            assert code == 400 and "error" in json.loads(body), bad
    finally:
        srv.shutdown()
        srv.server_close()
        VS.Handler.white = None
        VS.Handler.columns = None


@pytest.mark.parametrize("plane", [None, (0.0, 1.0, 0.0)])
@pytest.mark.parametrize("mode", VSec.MODES)
def test_every_does_not_change_the_section(
    cols: VC.Columns, mode: str, plane: tuple[float, ...] | None
) -> None:
    """The section chooses from every root within the radius: with every=4 it
    is the section of every=1 (its columns, steps, roots, arc, distances and
    key), also with a slice plane. every=4 alone would leave out some of the
    section's roots, so the test can fail."""
    v, radius, u = corner_vertex(cols), 1.0, (0.3, 1.0, 0.2)
    one = cols.section(VC.Request(vertex=v, radius=radius), u, mode, plane=plane)
    four = cols.section(
        VC.Request(vertex=v, radius=radius, every=4), u, mode, plane=plane
    )
    for k in ("columns", "step_vertices", "roots", "arc", "distance", "key", "n"):
        assert four[k] == one[k], k
    thin = cols.candidates(VC.Request(vertex=v, radius=radius, every=4))
    shown = set(thin[1].vertex[thin[2]].tolist())
    assert not set(one["roots"]) <= shown


def test_the_section_over_the_cap(
    cols: VC.Columns, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request over MAX_COLUMNS: /columns refuses, but /columns/section still
    answers, with the same columns as without the cap; /columns and
    /columns/profiles with cols= (the section's roots) give those columns, each
    the whole request's own column byte for byte, and their echoes."""
    srv = VS.serve(None, 0, white=WhiteView.__new__(WhiteView), columns=cols)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        v, radius = corner_vertex(cols), 1.3  # a radius no other test asks for
        q = f"v={v}&radius={radius}"
        sq = f"{q}&u=0.3,1,0.2"
        free = {}
        for mode in VSec.MODES:
            code, body = get(f"{base}/columns/section?{sq}&mode={mode}")
            assert code == 200
            free[mode] = json.loads(body)
            assert not free[mode]["capped"]
        n = free["nearest"]["n"]
        monkeypatch.setattr(VC, "MAX_COLUMNS", n - 1)
        code, body = settled(f"{base}/columns?{q}")
        assert code == 400 and "more than" in json.loads(body)["error"]
        parts = {}
        for mode in VSec.MODES:
            code, body = get(f"{base}/columns/section?{sq}&mode={mode}")
            assert code == 200
            capped = json.loads(body)
            assert capped["capped"] and capped["n"] == n
            assert capped["max_columns"] == n - 1
            for k in ("columns", "step_vertices", "roots", "arc", "distance"):
                assert capped[k] == free[mode][k], k
            # the section's columns alone, under the cap
            roots = capped["roots"]
            assert 0 < len(roots) < n
            cq = f"{q}&cols={','.join(map(str, roots))}"
            code, body = settled(f"{base}/columns?{cq}")
            assert code == 200
            h2, part = VC.decode(body)
            # in the whole answer's order (owned, then halo): checked below
            assert sorted(part["vertex"].tolist()) == roots
            code, body = settled(f"{base}/columns/profiles?{cq}")
            assert code == 200
            parts[mode] = (h2["key"], part["vertex"].tolist(), VC.decode(body)[1])
        # the whole request, without the cap: the same columns, the same echoes
        monkeypatch.undo()
        code, body = settled(f"{base}/columns?{q}")
        assert code == 200
        head, full = VC.decode(body)
        assert head["n"] == n
        code, body = settled(f"{base}/columns/profiles?{q}")
        assert code == 200
        full_prof = VC.decode(body)[1]
        fr = cols.result(head["key"]).rays
        at = {int(x): i for i, x in enumerate(full["vertex"])}
        for key, pv, prof in parts.values():
            pr = cols.result(key).rays
            assert [at[x] for x in pv] == sorted(at[x] for x in pv)
            for i, x in enumerate(pv):
                assert as_column(pr, i) == as_column(fr, at[x])
                a, b = pr.sample_start[i], pr.sample_start[i + 1]
                fa, fb = fr.sample_start[at[x]], fr.sample_start[at[x] + 1]
                assert np.array_equal(
                    prof["echo"][a:b], full_prof["echo"][fa:fb], equal_nan=True
                )
                assert np.array_equal(prof["valid"][a:b], full_prof["valid"][fa:fb])
        out = np.setdiff1d(np.arange(len(cols.v)), full["vertex"])
        code, body = settled(f"{base}/columns?{q}&cols={int(out[0])}")
        assert code == 400 and "not a column" in json.loads(body)["error"]
    finally:
        srv.shutdown()
        srv.server_close()
        VS.Handler.white = None
        VS.Handler.columns = None


def test_the_page_has_no_reference_to_niivue() -> None:
    """The page and its static scripts name no NiiVue and load no script from
    another host (the NiiVue CDN was the only one)."""
    files = sorted(VS.STATIC.iterdir())
    assert any(f.name == "index.html" for f in files)
    for f in files:
        text = f.read_text().lower()
        assert "niivue" not in text, f.name
        assert "cdn.jsdelivr" not in text and "https://" not in text, f.name


def test_a_failed_job_reaches_the_page(
    cols: VC.Columns, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With WAIT_S = 0 every job outlasts its request: the first request answers
    202, and the next one gives the job's error (400), not 202 for ever and
    without a second run. The request after it runs the job again."""
    calls: list[int] = []

    def bad(req: VC.Request, log: Any = quiet) -> VC.Result:
        calls.append(1)
        time.sleep(0.05)
        raise ValueError("more than the cap")

    monkeypatch.setattr(cols, "compute", bad)
    monkeypatch.setattr(VC, "WAIT_S", 0.0)
    srv = VS.serve(None, 0, white=WhiteView.__new__(WhiteView), columns=cols)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        q = f"v={corner_vertex(cols)}&radius=0.9"  # a radius no other test asks for
        for path in ("/columns", "/columns/profiles"):
            calls.clear()
            code, body = get(f"{base}{path}?{q}")
            assert code == 202 and json.loads(body)["status"] == "running"
            code, body = settled(f"{base}{path}?{q}")
            assert code == 400
            assert (
                json.loads(body)["error"]
                == "RuntimeError: ValueError: more than the cap"
            )
            assert calls == [1]
    finally:
        srv.shutdown()
        srv.server_close()
        VS.Handler.white = None
        VS.Handler.columns = None
