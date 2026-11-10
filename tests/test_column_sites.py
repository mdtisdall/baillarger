"""Tests for baillarger.column_sites: the queue order, the column graph, the
grouping of columns into sites, the per-site rows, and the files through the
command (column_sites.main), with answers at a site in a wm-edits/1 log."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from nibabel.gifti.gifti import GiftiImage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from test_wm_sites import surf  # noqa: F401  (the fixture)

from baillarger import bids
from baillarger import column_sites as C
from baillarger import wm_edits as E

NAMES = np.array(["a", "non_wm_behind_root", "c"])


def chain(n: int) -> tuple[np.ndarray, np.ndarray]:
    """n columns on the x axis, 1 mm apart, each linked to the next."""
    pos = np.c_[np.arange(n, dtype=float), np.zeros(n), np.zeros(n)]
    return pos, np.c_[np.arange(n - 1), np.arange(1, n)].astype(np.int64)


def test_queue_order_is_the_inverse_of_rank() -> None:
    assert C.queue_order([2, 0, 1]).tolist() == [1, 2, 0]
    with pytest.raises(ValueError, match="permutation"):
        C.queue_order([0, 0, 1])


def test_column_edges_drop_other_vertices_loops_and_repeats() -> None:
    vert = np.array([30, 10, 20])
    ev = np.array([[10, 20], [20, 10], [10, 99], [30, 30], [30, 10], [5, 6]])
    assert C.column_edges(vert, ev).tolist() == [[0, 1], [1, 2]]
    assert C.column_edges(vert, np.zeros((0, 2))).shape == (0, 2)


def test_grouping_on_a_chain_as_computed_by_hand() -> None:
    pos, edges = chain(8)
    order = np.array([3, 0, 1, 2, 4, 5, 6, 7])
    g = C.group_columns(order, edges, pos, 2.5)
    # 3 seeds site 0; 0 seeds site 1; 1 joins 0's site; 2 touches both sites
    # and joins the earlier seed (3); 4 and 5 are within 2.5 mm of 3; 6 is
    # 3 mm from 3 and seeds site 2; 7 joins it.
    assert g.seeds.tolist() == [3, 0, 6]
    assert g.site.tolist() == [1, 1, 0, 0, 0, 0, 2, 2]


def test_a_chain_longer_than_the_radius_gives_more_sites() -> None:
    pos, edges = chain(10)
    g = C.group_columns(np.arange(10), edges, pos, 2.0)
    assert g.seeds.tolist() == [0, 3, 6, 9]
    assert g.site.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3]
    # with a radius larger than the chain: one site
    assert C.group_columns(np.arange(10), edges, pos, 9.0).seeds.tolist() == [0]


def test_columns_with_no_edge_are_sites_of_their_own() -> None:
    pos, _ = chain(3)
    g = C.group_columns(np.array([2, 1, 0]), np.zeros((0, 2), np.int64), pos, 5.0)
    assert g.seeds.tolist() == [2, 1, 0] and g.site.tolist() == [2, 1, 0]


def test_grouping_properties_on_a_random_graph() -> None:
    rng = np.random.default_rng(3)
    n = 400
    pos = rng.uniform(0, 10, (n, 3))
    d = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    a, b = np.nonzero(np.triu(d < 1.5, 1))
    edges = np.c_[a, b].astype(np.int64)
    order = rng.permutation(n)
    R = 2.0
    g = C.group_columns(order, edges, pos, R)
    # every column is in exactly one site; each site has its seed
    assert (g.site >= 0).all() and g.site.max() == len(g.seeds) - 1
    assert (g.site[g.seeds] == np.arange(len(g.seeds))).all()
    # each column is within R of its seed
    assert (np.linalg.norm(pos - pos[g.seeds[g.site]], axis=1) <= R + 1e-12).all()
    # the sites are in the order of their seeds in the queue
    at = np.empty(n, np.int64)
    at[order] = np.arange(n)
    assert (np.diff(at[g.seeds]) > 0).all()
    # each site is connected on the column graph
    same = g.site[a] == g.site[b]
    A = coo_matrix((np.ones(same.sum()), (a[same], b[same])), shape=(n, n))
    comp = connected_components(A, directed=False)[1]
    for s in range(len(g.seeds)):
        assert len(np.unique(comp[g.site == s])) == 1


def conf_arrays(n: int) -> dict[str, np.ndarray]:
    conf = np.linspace(-3, 3, n).astype(np.float32)
    conf[1] = -np.inf  # no top
    top_k = np.full(n, 10)
    top_k[1] = -1
    terms = np.zeros((n, 3), np.float32)
    terms[:, 1] = -1.0
    terms[1] = np.nan
    return dict(
        vert=np.arange(100, 100 + n),
        conf=conf,
        top_k=top_k,
        top_mm=np.where(top_k < 0, np.nan, 2.0),
        terms=terms,
        term_names=NAMES,
        gm_behind=np.full(n, 0.5, np.float32),
        fl_behind=np.zeros(n, np.float32),
    )


def test_column_causes() -> None:
    nan = np.nan
    terms = np.array(
        [
            [-1.0, -2.0, 0.0],  # non-WM term lowest
            [-2.0, -1.0, 0.0],  # another term lowest
            [-2.0, -2.0, 0.0],  # a tie: the first lowest term counts
            [nan, -0.5, nan],  # NaN terms do not count
            [nan, nan, nan],  # no finite term
            [-9.0, -9.5, 0.0],  # no top, whatever the terms
        ]
    )
    top_k = np.array([3, 3, 3, 3, 3, -1])
    got = C.column_causes(top_k, terms, NAMES)
    assert got.tolist() == [1, 2, 2, 1, 2, 0]
    # without the non-WM term, no column has that cause
    assert C.column_causes(top_k, terms, ["a", "b", "c"]).tolist() == [2] * 5 + [0]


def test_a_tie_of_causes_goes_to_the_first_cause() -> None:
    pos, edges = chain(2)
    c = conf_arrays(2)
    c["top_k"][0], c["conf"][0] = -1, -np.inf  # column 0: no top
    c["top_k"][1] = 10  # column 1: non-WM behind the root
    c["terms"][1] = [0.0, -1.0, 0.0]
    g = C.group_columns(np.array([0, 1]), edges, pos, 5.0)
    vox = np.zeros((2, 3), np.int64)
    (row,) = C.site_rows(g, c, pos, vox)
    assert (row["no_top"], row["non_wm_behind_root"], row["other"]) == (1, 1, 0)
    assert row["cause"] == "no-top"


def test_site_rows_on_a_chain() -> None:
    pos, edges = chain(6)
    c = conf_arrays(6)
    order = np.argsort(c["conf"], kind="stable")  # 1, 0, 2, 3, 4, 5
    g = C.group_columns(order, edges, pos, 1.0)
    # 1 seeds site 0 and takes 0 and 2; 3 seeds site 1 and takes 4; 5 alone
    assert g.seeds.tolist() == [1, 3, 5]
    vox = np.c_[np.arange(6), np.ones(6), np.ones(6)].astype(np.int64)
    rows = C.site_rows(g, c, pos, vox)
    assert [r["rank"] for r in rows] == [1, 2, 3]
    assert [r["kind"] for r in rows] == ["column-no-top", "column-top", "column-top"]
    first, second = rows[0], rows[1]
    assert (first["i"], first["j"], first["k"]) == (1, 1, 1)
    assert first["score"] is None and first["top_mm"] is None
    assert first["lowest_term"] == "n/a" and first["vertex"] == 101
    assert first["columns"] == 3 and first["no_top"] == 1
    # causes: 1 has no top; the others' lowest term is non_wm_behind_root
    assert (first["non_wm_behind_root"], first["other"]) == (2, 0)
    assert first["cause"] == "non-wm-behind-root"
    assert first["median_score"] == pytest.approx(round(float(c["conf"][0]), 4))
    assert first["extent_mm"] == 1.0
    assert second["score"] == pytest.approx(round(float(c["conf"][3]), 4))
    assert second["lowest_term"] == "non_wm_behind_root" and second["top_mm"] == 2.0
    assert second["gm_behind_mm"] == 0.5 and second["fluid_behind_mm"] == 0.0
    assert rows[2]["columns"] == 1 and rows[2]["extent_mm"] == 0.0


def test_the_command_writes_sites_that_the_answer_log_takes(
    surf: Path,  # noqa: F811
    tmp_path: Path,
) -> None:
    v_mm = np.asarray(
        GiftiImage.from_filename(str(surf)).agg_data("pointset"), np.float64
    )
    n = 40
    vert = np.arange(0, 2 * n, 2)  # every other vertex is a column
    c = conf_arrays(n)
    c["vert"] = vert
    rng = np.random.default_rng(5)
    c["rank"] = np.argsort(np.argsort(c["conf"], kind="stable"))
    c["root"] = v_mm[vert].astype(np.float32)
    conf = tmp_path / "top_confidence.npz"
    np.savez(conf, **c)  # type: ignore[arg-type]
    feats = []
    for k in range(2):  # the graph in two files, with a vertex that is no column
        ev = np.c_[vert[:-1], vert[1:]][k::2]
        ev = np.r_[ev, [[1, vert[0]]]]
        f = tmp_path / f"feat_{k}.npz"
        np.savez(f, edges_v=ev, P=rng.normal(size=(2, 2)))
        feats.append(f)
    out = tmp_path / "sites"
    C.main([str(surf), str(conf), *map(str, feats), "--out", str(out)])
    p = C.sites_paths(surf, out)
    assert p["table"].name == "sub-T01_ses-S1_hemi-R_desc-columnsites_sites.tsv"
    rows, side = C.read_column_sites(p["table"])
    assert side["Sites"] == len(rows) and side["Columns"] == n
    assert side["Edges"] == n - 1 and side["Hemisphere"] == "R"
    assert side["RadiusMm"] == C.RADIUS and side["TermNames"] == NAMES.tolist()
    assert sum(side["SitesByKind"].values()) == len(rows)
    assert sum(side["SitesByCause"].values()) == len(rows)
    assert sum(side["ColumnsByCause"].values()) == n
    assert side["ColumnsByCause"]["no-top"] == 1
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
    assert rows[0]["kind"] == "column-no-top" and rows[0]["vertex"] == vert[1]
    # the members: every column once, in a site of the table
    mv, ms = C.read_members(p["members"])
    assert sorted(mv.tolist()) == vert.tolist()
    assert set(ms.tolist()) == {r["rank"] for r in rows}
    assert [int((ms == r["rank"]).sum()) for r in rows] == [r["columns"] for r in rows]
    # the site voxel is the seed's root on the dseg's grid
    dseg = Path(side["Sources"][2])
    img = nib.load(dseg)
    assert isinstance(img, nib.nifti1.Nifti1Image)
    for r in rows:
        ijk = nib.affines.apply_affine(np.linalg.inv(img.affine), v_mm[r["vertex"]])
        assert [r["i"], r["j"], r["k"]] == np.rint(ijk).astype(int).tolist()
    # a session on this queue, a site shown, an answer: the log takes them,
    # and the replay applies the answer
    log = E.EditLog.create(
        tmp_path / "sub-T01_ses-S1_desc-wmedits_edits.jsonl", dseg, "R"
    )
    s = log.start_session(
        0.0, None, bids.file_sha256(dseg), bids.file_sha256(p["table"])
    )
    site = (rows[0]["i"], rows[0]["j"], rows[0]["k"])
    log.shown(s, site, rows[0]["kind"], rows[0]["rank"], rows[0]["score"])
    log.answer(s, site, "manual-paint", [site], [0], None, 3.0)
    L = E.read_log(log.path)
    assert L.answers()[0]["SiteKind"] == "column-no-top"
    assert L.answers()[0]["Rank"] == 1 and L.answers()[0]["Score"] is None
    labels = np.asarray(img.dataobj).copy()
    labels[site] = 41
    assert E.replay(labels, L).labels[site] == 0


def test_a_table_with_other_columns_is_refused(tmp_path: Path) -> None:
    t = tmp_path / "x_desc-columnsites_sites.tsv"
    t.write_text("rank\ti\n")
    side = dict(
        SpecVersion="column-sites/1",
        Description="d",
        Sources=[],
        SourcesSHA256={},
        Provenance={},
        Hemisphere="R",
        RadiusMm=2.0,
        Columns=0,
        Edges=0,
        Sites=0,
        SitesByKind={},
        SitesByCause={},
        ColumnsByCause={},
        TermNames=[],
    )
    bids.write_sidecar(bids.sidecar_path(t), C.COLUMN_SITES_SPEC, side)
    with pytest.raises(ValueError, match="columns"):
        C.read_column_sites(t)
    m = tmp_path / "x_desc-columnsites_members.tsv"
    m.write_text("site\tvertex\n")
    with pytest.raises(ValueError, match="columns"):
        C.read_members(m)
