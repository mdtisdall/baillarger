"""Tests of baillarger.viewer.overlay (per-vertex scalar maps on the overview),
with exact answers, on a phantom surface written by white_dseg, white_surf and
white_overview: the aggregate against a plain loop over the carry file, the
ranked sites against a brute-force greedy, the sites' fields against the
surface's, the meta, the bad files and the routes."""

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest

from baillarger import white_dseg, white_overview, white_surf
from baillarger.data import echo_path
from baillarger.gifti import DataArray, write_external
from baillarger.viewer import overlay as VOL
from baillarger.viewer import server as VS
from baillarger.viewer.white_view import WhiteView

AFF = np.array([[0.2, 0, 0, -3.0], [0, 0.16, 0, 2.0], [0, 0, 0.2, 5.0], [0, 0, 0, 1.0]])


def quiet(m: str) -> None:
    pass


def get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return int(r.status), bytes(r.read())
    except urllib.error.HTTPError as e:
        return int(e.code), bytes(e.read())


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A right hemisphere (a WM ball round a ventricle) with its overview, three
    echoes, and the map files."""
    d = tmp_path_factory.mktemp("overlay")
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
    white_surf.main([str(dseg), "--out", str(d / "deriv"), "--block-cells", "8"])
    surf = anat / "sub-T01_ses-S1_hemi-R_white.surf.gii"
    white_overview.main([str(surf), "--eps", "0.05", "--workers", "1"])
    ed = d / "echoes"
    ed.mkdir()
    for k in (1, 2, 3):
        e = (k * np.arange(int(np.prod(shape)))).reshape(shape).astype(np.float32)
        nib.save(nib.nifti1.Nifti1Image(e, AFF), echo_path(k, ed).with_suffix(""))
    return dict(surf=surf, echoes=ed, dir=d)


@pytest.fixture(scope="module")
def white(data: dict[str, Path]) -> WhiteView:
    return WhiteView(data["surf"], data["dir"] / "cache", log=quiet)


def write_maps(path: Path, maps: dict[str, npt.NDArray[np.float32]]) -> Path:
    write_external(
        path,
        [DataArray(m, "NIFTI_INTENT_NONE", dict(Name=n)) for n, m in maps.items()],
    )
    return path


@pytest.fixture(scope="module")
def files(data: dict[str, Path], white: WhiteView) -> dict[str, Path]:
    n = len(white.W.v)
    rng = np.random.default_rng(1)
    rnd = rng.normal(size=n).astype(np.float32)
    rnd[rng.random(n) < 0.3] = np.nan
    xs = np.asarray(white.pts[:, 0], np.float32)
    nan = np.full(n, np.nan, np.float32)
    d = data["dir"] / "maps"
    return dict(
        one=write_maps(d / "one.func.gii", dict(rnd=rnd, x=xs, nan=nan)),
        two=write_maps(
            d / "two.shape.gii", {"a b": xs[::-1].copy(), "c/d": -xs, "plain": rnd}
        ),
    )


@pytest.fixture(scope="module")
def ovl(white: WhiteView, files: dict[str, Path]) -> VOL.Overlays:
    return VOL.Overlays(
        [files["one"], files["two"]], white.ov, white.pts, len(white.W.v)
    )


def reference(ov: Any, a: npt.NDArray[np.float32], fn: Any) -> npt.NDArray[np.float32]:
    """The aggregate by a plain loop: per overview vertex, over its incident
    faces, the carried values and its own."""
    inc: list[list[int]] = [[] for _ in ov.vertex_ids]
    for fc, tri in enumerate(ov.f):
        for k in tri:
            inc[int(k)].append(fc)
    out = np.full(len(ov.vertex_ids), np.nan, np.float32)
    for k, t in enumerate(ov.vertex_ids):
        vals = [a[t]]
        for fc in inc[k]:
            vals.extend(a[ov.carried_ids[ov.ptr[fc] : ov.ptr[fc + 1]]])
        arr = np.asarray(vals, np.float32)
        if np.isfinite(arr).any():
            out[k] = fn(arr)
    return out


def test_aggregate_equals_a_loop_over_the_carry_file(ovl: VOL.Overlays) -> None:
    ov = ovl.ov
    assert len(ov.carried_ids) > len(ov.vertex_ids)
    for file, name in [("one", "rnd"), ("one", "x"), ("one", "nan"), ("two", "plain")]:
        a = np.asarray(ovl.values(file, name))
        for agg, fn in (("max", np.nanmax), ("min", np.nanmin)):
            got = ovl.aggregate(file, name, agg)
            assert got.dtype == np.float32 and got.shape == (len(ov.vertex_ids),)
            np.testing.assert_array_equal(got, reference(ov, a, fn))
    assert np.isnan(ovl.aggregate("one", "nan", "max")).all()


def brute(
    val: npt.NDArray[np.float64],
    pts: npt.NDArray[np.float64],
    n: int,
    spacing: float,
    order: str,
) -> list[int]:
    cand = [i for i in range(len(val)) if np.isfinite(val[i])]
    cand.sort(key=lambda i: (-val[i] if order == "high" else val[i], i))
    kept: list[int] = []
    for i in cand:
        if len(kept) >= n:
            break
        if all(np.linalg.norm(pts[i] - pts[j]) > spacing for j in kept):
            kept.append(i)
    return kept


@pytest.mark.parametrize("order", ["high", "low"])
@pytest.mark.parametrize("spacing", [0.0, 1.0, 2.0, 3.5])
def test_ranked_sites_equals_a_brute_force_greedy(order: str, spacing: float) -> None:
    rng = np.random.default_rng(3)
    pts = rng.integers(0, 12, size=(3000, 3)).astype(np.float64)  # exact distances
    val = rng.integers(0, 40, size=3000).astype(np.float64)  # many ties
    val[rng.random(3000) < 0.2] = np.nan
    got = VOL.ranked_sites(val, pts, 50, spacing, order)
    assert got.tolist() == brute(val, pts, 50, spacing, order)
    assert not np.isnan(val[got]).any()
    d = np.linalg.norm(pts[got][:, None] - pts[got][None], axis=2)
    assert (d[~np.eye(len(got), dtype=bool)] > spacing).all()
    best = np.nanmax(val) if order == "high" else np.nanmin(val)
    assert val[got[0]] == best
    assert got[0] == np.flatnonzero(val == best)[0]  # ties: the lower id


def test_ranked_sites_stops_when_candidates_run_out() -> None:
    pts = np.arange(30, dtype=np.float64)[:, None] * [1.0, 0, 0]
    val = np.arange(30, dtype=np.float64)
    val[::2] = np.nan  # 15 finite
    assert len(VOL.ranked_sites(val, pts, 100, 0.0)) == 15
    got = VOL.ranked_sites(val, pts, 100, 2.5)  # spacing drops neighbours
    assert got.tolist() == brute(val, pts, 100, 2.5, "high")
    assert len(VOL.ranked_sites(np.full(30, np.nan), pts, 5, 1.0)) == 0
    assert len(VOL.ranked_sites(val, pts, 3, 0.0)) == 3


def test_ranked_sites_across_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(VOL, "SITE_CHUNK", 7)
    rng = np.random.default_rng(5)
    pts = rng.integers(0, 10, size=(500, 3)).astype(np.float64)
    val = rng.integers(0, 20, size=500).astype(np.float64)
    got = VOL.ranked_sites(val, pts, 30, 2.0)
    assert got.tolist() == brute(val, pts, 30, 2.0, "high")


def test_sites_fields(ovl: VOL.Overlays, white: WhiteView) -> None:
    ov = white.ov
    t0 = time.time()
    s = ovl.sites("one", "rnd", n=60, spacing=0.3)
    print(f"sites: {time.time() - t0:.3f} s")
    assert len(s) > 10
    pts = np.asarray(white.pts, np.float64)
    inv = {int(t): k for k, t in enumerate(ov.vertex_ids)}
    n_own = 0
    for i, d in enumerate(s):
        t = d["vertex"]
        assert d["rank"] == i + 1
        assert d["voxel"] == white.voxel(t)
        assert d["world_mm"] == [round(float(x), 3) for x in pts[t]]
        assert d["value"] == float(ovl.values("one", "rnd")[t])
        k = d["overview_vertex"]
        assert k is not None
        if t in inv:
            assert k == inv[t]
            n_own += 1
        else:  # a corner of a face that carries t, the nearest of its corners
            faces = np.flatnonzero(
                [
                    t in ov.carried_ids[ov.ptr[f] : ov.ptr[f + 1]]
                    for f in range(len(ov.f))
                ]
            )
            assert len(faces) > 0
            corners = {int(c) for f in faces for c in ov.f[f]}
            assert k in corners
    assert n_own >= 0


def test_the_overview_vertex_of_a_site_is_itself(ovl: VOL.Overlays) -> None:
    ov = ovl.ov
    k = np.arange(0, len(ov.vertex_ids), 7)
    got = ovl.markers(ov.vertex_ids[k])
    assert got == k.tolist()


def test_meta(ovl: VOL.Overlays) -> None:
    m = {(d["file"], d["map"]): d for d in ovl.meta()}
    assert set(m) == {
        ("one", "rnd"),
        ("one", "x"),
        ("one", "nan"),
        ("two", "a b"),
        ("two", "c/d"),
        ("two", "plain"),
    }
    v = np.asarray(ovl.values("one", "rnd"))
    fin = v[np.isfinite(v)]
    d = m[("one", "rnd")]
    assert [d["p1"], d["p50"], d["p99"]] == [
        float(x) for x in np.percentile(fin, [1, 50, 99])
    ]
    assert d["finite"] == len(fin) and d["n"] == len(v)
    e = m[("one", "nan")]
    assert (e["p1"], e["p50"], e["p99"], e["finite"]) == (None, None, None, 0)


def test_bad_files(
    white: WhiteView,
    files: dict[str, Path],
    data: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n = len(white.W.v)
    bad = write_maps(tmp_path / "bad.func.gii", dict(m=np.zeros(n - 1, np.float32)))
    with pytest.raises(ValueError, match=r"bad\.func\.gii.*'m'.*" + str(n)):
        VOL.Overlays([bad], white.ov, white.pts, n)
    two_d = write_maps(tmp_path / "wide.func.gii", dict(m=np.zeros((n, 2), np.float32)))
    with pytest.raises(ValueError, match="wide"):
        VOL.Overlays([two_d], white.ov, white.pts, n)
    col = write_maps(tmp_path / "col.func.gii", dict(m=np.zeros((n, 1), np.float32)))
    VOL.Overlays([col], white.ov, white.pts, n)  # (n, 1) is fine
    sub = tmp_path / "sub"
    twin = write_maps(sub / "one.func.gii", dict(m=np.zeros(n, np.float32)))
    with pytest.raises(ValueError, match="one"):
        VOL.Overlays([files["one"], twin], white.ov, white.pts, n)
    z = np.zeros(n, np.float32)
    dup = tmp_path / "dup.func.gii"
    write_external(
        dup,
        [DataArray(z, "NIFTI_INTENT_NONE", dict(Name="m")) for _ in range(2)],
    )
    with pytest.raises(ValueError, match="'m'"):
        VOL.Overlays([dup], white.ov, white.pts, n)
    for r in ("at", "meta"):
        res = write_maps(tmp_path / f"{r}.func.gii", dict(m=np.zeros(n, np.float32)))
        with pytest.raises(ValueError, match="reserved"):
            VOL.Overlays([res], white.ov, white.pts, n)

    # main() stops before it serves
    ed = tmp_path / "data"
    ed.mkdir()
    for k in (1, 2, 3):
        e = nib.load(echo_path(k, data["echoes"]).with_suffix(""))
        nib.save(e, echo_path(k, ed))
    monkeypatch.setenv("BAILLARGER_DATA", str(ed))

    def no_serve(*a: Any, **k: Any) -> Any:
        raise AssertionError("served")

    monkeypatch.setattr(VS, "serve", no_serve)
    argv = ["--white", str(data["surf"]), "--cache", str(tmp_path / "cache")]
    with pytest.raises(SystemExit):
        VS.main(argv + ["--overlay", str(bad)])


def test_the_routes(white: WhiteView, files: dict[str, Path]) -> None:
    ovl = VOL.Overlays(
        [files["one"], files["two"]], white.ov, white.pts, len(white.W.v)
    )
    srv = VS.serve(None, 0, white=white, overlays=ovl)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        code, body = get(f"{base}/overlay/meta")
        assert code == 200
        meta = json.loads(body)
        assert [f["file"] for f in meta["files"]] == ["one", "two"]
        assert [m["map"] for m in meta["files"][1]["maps"]] == ["a b", "c/d", "plain"]
        assert meta["files"][0]["maps"][2]["p50"] is None
        for agg in ("max", "min"):
            t0 = time.time()
            code, body = get(f"{base}/overlay/one/rnd?agg={agg}")
            print(f"aggregate {agg}: {time.time() - t0:.3f} s")
            assert code == 200
            assert body == ovl.aggregate("one", "rnd", agg).astype("<f4").tobytes()
        assert (
            get(f"{base}/overlay/one/rnd")[1]
            == ovl.aggregate("one", "rnd", "max").tobytes()
        )
        sp = "n=5&spacing=0.4&order=low"
        code, body = get(f"{base}/overlay/one/x/sites?{sp}")
        assert code == 200
        assert json.loads(body) == dict(sites=ovl.sites("one", "x", 5, 0.4, "low"))
        vs = [0, 5, 17]
        code, body = get(f"{base}/overlay/at?v=0,5,17")
        assert code == 200
        at = json.loads(body)
        assert at["v"] == vs
        exp = np.asarray(ovl.values("one", "rnd"))[vs]
        assert at["values"]["one"]["rnd"] == [
            None if np.isnan(a) else float(a) for a in exp
        ]
        assert at["values"]["one"]["nan"] == [None, None, None]
        # names with a space or a slash, URL-encoded
        a_b = urllib.parse.quote("a b", safe="")
        c_d = urllib.parse.quote("c/d", safe="")
        assert get(f"{base}/overlay/two/{a_b}")[0] == 200
        code, body = get(f"{base}/overlay/two/{c_d}?agg=min")
        assert code == 200 and body == ovl.aggregate("two", "c/d", "min").tobytes()
        assert get(f"{base}/overlay/two/{c_d}/sites?n=3")[0] == 200
        # errors
        assert get(f"{base}/overlay/one/nope")[0] == 404
        assert get(f"{base}/overlay/nope/rnd")[0] == 404
        assert get(f"{base}/overlay/one/rnd/nope")[0] == 404
        for bad in (
            "/overlay/one/rnd?agg=mean",
            "/overlay/one/rnd/sites?order=up",
            "/overlay/one/rnd/sites?n=0",
            "/overlay/one/rnd/sites?n=x",
            "/overlay/one/rnd/sites?spacing=-1",
            "/overlay/at?v=a",
            "/overlay/at?v=-1",
        ):
            code, body = get(base + bad)
            assert code == 400 and "error" in json.loads(body), bad
    finally:
        srv.shutdown()
        srv.server_close()
        VS.Handler.white = None
        VS.Handler.overlays = None
