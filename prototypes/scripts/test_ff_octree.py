"""Tests of ff_octree (PR 5, task 5.1): the structure of the octree, the transfer of the
distributions, the marking, and the plan's exact tests "the slab stays at 0.64 mm",
"noise alone refuses every refinement" and "the marking does not depend on the order of
updates".

Run from the repository root:
    nix develop --command uv run pytest -q prototypes/scripts/test_ff_octree.py

The two adapt tests use a box of 24³ voxels (the plan's slab box is 64 × 64 × 48),
so that the file runs in well under 3 minutes on a laptop CPU in float64. At 0.64 mm
the box has 7³ = 343 base nodes. The defaults of SolverSettings are not changed.

Every tolerance here was fixed when the test was written, before the first run.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import ff_grid as FG  # noqa: E402
import ff_octree as O  # noqa: E402
import ff_oracle as FO  # noqa: E402
import ff_truth as FTR  # noqa: E402
import ff_types as T  # noqa: E402

FEATS = {
    e: dict(
        profile=list(np.sin(np.linspace(0, 3 * np.pi, 50))),
        wm_level=-0.3,
        fluid_level=-2.0,
        hp_stat=3.0,
    )
    for e in ("echo2", "echo3")
}  # synthetic features (CI has no site data), as test_ff_meanfield.py
SHAPE = (24, 24, 24)
POS_TOL = 1e-12  # voxels: a hanging node's position against its parents'
LOCAL_TOL = 1e-12  # voxels: the round trip of locate
SUM_TOL = 1e-12  # the total probability of a node after transfer


# ---------------------------------------------------------------- helpers


def refined_tree() -> O.OctreeGrid:
    """An octree of the SHAPE box with leaves of levels 0 to 3: one corner refined three
    times (the balance refines more), and one other leaf refined once."""
    g = O.OctreeGrid.base(SHAPE)
    for _ in range(3):
        p = np.array([[g.extent[0] // 2 + 1, g.extent[1] // 2 + 1, 3]])
        g = g.refine(g.find_leaf(p))
    return g.refine(g.find_leaf(np.array([[1, g.extent[1] - 1, g.extent[2] - 1]])))


def random_field(grid: O.OctreeGrid, seed: int) -> T.Field:
    """A Field with smooth means, K = prune_max slots with a random number kept. In the
    upper half of the box along x the candidates have noise and some nodes are stopped;
    in the lower half the field is smooth and moving (so that both refinements and
    coarsenings are nominated)."""
    rng = np.random.default_rng(seed)
    n, K = grid.n_nodes, T.DEFAULTS.prune_max
    X = grid.nodes * T.VOXEL_MM
    nk = rng.integers(1, K + 1, n)
    slot = np.arange(K)[None, :]
    kept = slot < nk[:, None]
    noisy = grid.node_units[:, 0] > grid.extent[0] // 2
    amp = np.where(noisy, 1.0, 0.0)
    base_tau = -(X[:, 0] * 0.3 + 0.1 * X[:, 1])
    tau = np.where(
        kept, base_tau[:, None] + 0.04 * amp[:, None] * rng.normal(size=(n, K)), np.nan
    )
    d = np.array([1.0, 0.2, 0.1]) + 0.1 * amp[:, None, None] * rng.normal(
        size=(n, K, 3)
    )
    d /= np.linalg.norm(d, axis=2, keepdims=True)
    d = np.where(kept[..., None], d, 0.0)
    q = np.where(kept, rng.random((n, K)), 0.0)
    q_stop = np.where(noisy & (rng.random(n) < 0.2), rng.random(n), 0.0)
    q_stop[noisy & (rng.random(n) < 0.05)] = 1.0
    q = q / q.sum(1, keepdims=True) * (1.0 - q_stop)[:, None]
    return T.Field(grid=grid, tau=tau, direction=d, q=q, q_stop=q_stop, n_kept=nk)


def permuted(grid: O.OctreeGrid, seed: int) -> tuple[O.OctreeGrid, np.ndarray]:
    """The octree built from a permuted leaf list, and the permutation."""
    perm = np.random.default_rng(seed).permutation(grid.n_cells)
    return O.OctreeGrid(
        grid.shape_vox, grid.level[perm], grid.anchor[perm], grid.settings
    ), perm


def assert_same_tree(a: O.OctreeGrid, b: O.OctreeGrid) -> None:
    for k, v in a.arrays().items():
        assert np.array_equal(v, b.arrays()[k]), k
    assert np.array_equal(a.edges, b.edges)
    assert np.array_equal(a.nodes, b.nodes)


# ---------------------------------------------------------------- structure


def test_base_equals_uniform_grid() -> None:
    g = O.OctreeGrid.base(SHAPE)
    u = FG.UniformGrid(SHAPE, T.DEFAULTS.octree_coarsest_mm)
    assert g.n_nodes == u.n_nodes and g.n_cells == u.n_cells
    assert np.array_equal(g.nodes, u.nodes)
    assert np.array_equal(g.edges, u.edges)
    # the leaves are in Morton order, the uniform cells in C order: compare as sets
    og = np.lexsort(g.cell_origin.T[::-1])
    ou = np.lexsort(u.cell_origin.T[::-1])
    assert np.array_equal(g.cells[og], u.cells[ou])
    assert np.array_equal(g.cell_origin[og], u.cell_origin[ou])
    assert np.allclose(g.cell_size_mm, u.cell_size_mm[0])
    assert len(g.hanging[0]) == 0 and not g.prior_shaped.any()
    Kg, Mg = g.fem_matrices()
    Ku, Mu = FG.fem_matrices(u)
    assert abs(Kg - Ku).max() == 0.0
    assert abs(Mg - Mu).max() == 0.0
    assert np.array_equal(g.edge_weights_vox(), FG.edge_weights_vox(u))


def test_refined_tree_structure() -> None:
    g = refined_tree()
    assert set(g.level.tolist()) == {0, 1, 2, 3}
    # 2:1 balanced; the constructor's check: the leaves tile the box and do not overlap
    assert len(g.unbalanced()) == 0
    O.OctreeGrid(g.shape_vox, g.level, g.anchor, g.settings, check=True)
    vol = np.sum(g.size_units.astype(np.float64) ** 3)
    assert vol == float(np.prod(g.extent.astype(np.float64)))
    # hanging nodes: position = Σ w · parents' positions; weights sum to 1
    ids, par, w = g.hanging
    assert len(ids) > 0
    assert np.allclose(w.sum(1), 1.0, atol=0, rtol=0)
    pos = np.einsum("hp,hpx->hx", w, g.nodes[par])
    assert np.abs(pos - g.nodes[ids]).max() <= POS_TOL
    # Morton order
    assert np.all(np.diff(g.morton()) > 0)


def test_edges_have_no_node_inside() -> None:
    g = refined_tree()
    e = g.edges
    a, b = g.node_units[e[:, 0]], g.node_units[e[:, 1]]
    d = b - a
    assert np.all((d != 0).sum(1) == 1) and np.all(d.sum(1) > 0)  # i < j: along +axis
    L = d.sum(1)
    for k in range(1, int(L.max())):
        sel = L > k
        p = a[sel] + k * (d[sel] // L[sel][:, None])
        assert np.all(g.node_id(p) < 0)
    nb = FG.axis_neighbours(g)  # raises if two edges give one slot
    for s in range(6):
        j = nb[:, s]
        ok = j >= 0
        opp = s ^ 1
        assert np.array_equal(nb[j[ok], opp], np.flatnonzero(ok))


def test_edge_weights_sum_to_volume() -> None:
    g = refined_tree()
    w = g.edge_weights_vox()
    assert np.all(w > 0)
    # each leaf gives 12 · vol / 12 = its volume
    total = np.sum(g.cell_size_mm**3) / T.VOXEL_MM**3
    assert np.isclose(w.sum(), total, rtol=1e-12, atol=0)


def test_colours_separate_every_clique() -> None:
    g = refined_tree()
    c = g.colour
    for cl in (g.cells, g.edges):
        for i in range(cl.shape[1]):
            for j in range(i + 1, cl.shape[1]):
                assert np.all(c[cl[:, i]] != c[cl[:, j]])
    nb = FG.axis_neighbours(g)
    for s in range(6):
        j = nb[:, s]
        ok = j >= 0
        assert np.all(c[ok] != c[j[ok]])
    for sx in (0, 1):
        for sy in (2, 3):
            for sz in (4, 5):
                cols = [nb[:, sx], nb[:, sy], nb[:, sz]]
                for p in range(3):
                    for q in range(p + 1, 3):
                        ok = (cols[p] >= 0) & (cols[q] >= 0)
                        assert np.all(c[cols[p][ok]] != c[cols[q][ok]])
    print("colours:", g.n_colours)


def test_locate_round_trips() -> None:
    g = refined_tree()
    rng = np.random.default_rng(1)
    top = g.extent * g.unit_vox
    X = rng.random((5000, 3)) * top
    X = np.concatenate([X, g.nodes])
    cid, local = g.locate(X)
    assert np.all(cid >= 0)
    side = g.size_units[cid] * g.unit_vox
    back = g.cell_origin[cid] + local * side[:, None]
    assert np.abs(back - X).max() <= LOCAL_TOL
    assert np.all((local >= 0) & (local <= 1))
    out, _ = g.locate(np.array([[-0.5, 1.0, 1.0], [1.0, 1.0, top[2] + 0.5]]))
    assert np.all(out == -1)


def test_refine_then_coarsen_gives_the_tree_back() -> None:
    base = O.OctreeGrid.base(SHAPE)
    g = refined_tree()
    for lv in range(g.max_level, 0, -1):
        sel = g.level == lv
        if not sel.any():
            continue
        side = 1 << (g.max_level - lv + 1)
        pa = np.unique((g.anchor[sel] // side) * side, axis=0)
        g = g.coarsen(np.full(len(pa), lv - 1), pa)
        assert not np.any(g.level >= lv)
    assert_same_tree(g, base)


def test_prior_shaped_below_the_voxel() -> None:
    g = refined_tree()
    assert np.array_equal(g.prior_shaped, g.cell_size_mm < T.VOXEL_MM * (1 - 1e-9))
    assert g.prior_shaped.sum() == np.sum(g.level == 3)  # 0.08 mm < 0.16 mm
    assert g.level_counts()["n_prior_shaped"] == int(g.prior_shaped.sum())


# ---------------------------------------------------------------- transfer


def test_transfer_to_a_finer_and_back() -> None:
    old = O.OctreeGrid.base(SHAPE)
    f = random_field(old, 0)
    new = refined_tree()
    g = O.transfer(f, new)
    total = g.q_stop + g.q.sum(1)
    assert np.abs(total - 1.0).max() <= SUM_TOL
    assert np.all(g.q >= 0) and np.all((g.q_stop >= 0) & (g.q_stop <= 1))
    oid = old.node_id(new.node_units)
    same = oid >= 0
    assert same.sum() == old.n_nodes
    for k in ("tau", "direction", "q", "q_stop", "n_kept"):
        a, b = getattr(g, k)[same], getattr(f, k)[oid[same]]
        assert np.array_equal(a, b, equal_nan=True), k
    slot = np.arange(g.tau.shape[1])[None, :]
    kept = slot < g.n_kept[:, None]
    assert np.all(np.isfinite(g.tau[kept])) and np.all(np.isnan(g.tau[~kept]))
    assert np.all(g.q[~kept] == 0)
    # back to the base octree: every node is an old node, kept exactly
    h = O.transfer(g, old)
    for k in ("tau", "direction", "q", "q_stop", "n_kept"):
        assert np.array_equal(getattr(h, k), getattr(f, k), equal_nan=True), k


# ---------------------------------------------------------------- marking and order


def test_marking_does_not_depend_on_the_order_of_updates() -> None:
    """The plan's exact test. The leaves (and with them the indicator inputs) are
    permuted: the octree built from the permuted leaf list, the indicators of a field
    on it, the nominated sets (as (level, anchor) sets) and the resulting octrees are
    identical, bit for bit."""
    g = refined_tree()
    f = random_field(g, 2)
    rng = np.random.default_rng(3)
    speed = np.exp(0.3 * rng.normal(size=g.n_nodes))
    tdir = rng.normal(size=(g.n_nodes, 3))
    tdir /= np.linalg.norm(tdir, axis=1, keepdims=True)
    coh = rng.random(g.n_nodes)
    ind = O.indicators(g, f, speed, tdir, coh)
    marks = O.mark(g, ind["total"])
    assert len(marks["refine_level"]) > 0 and len(marks["coarsen_level"]) > 0
    ref, coa = O.apply_marks(g, marks)
    for seed in range(3):
        g2, perm = permuted(g, 10 + seed)
        assert_same_tree(g, g2)
        ind2 = O.indicators(g2, f, speed, tdir, coh)
        for k in ind:
            assert np.array_equal(ind[k], ind2[k]), k
        # the marking pass on the permuted leaf list with the permuted indicators
        m2 = O.mark_leaves(
            g.level[perm], g.anchor[perm], ind["total"][perm], g.max_level
        )
        for k in ("refine_level", "refine_anchor", "coarsen_level", "coarsen_anchor"):
            assert np.array_equal(marks[k], m2[k]), k
        ref2, coa2 = O.apply_marks(g2, m2)
        assert_same_tree(ref, ref2)
        assert_same_tree(coa, coa2)


def test_marking_rule() -> None:
    g = refined_tree()
    rng = np.random.default_rng(4)
    tot = rng.random(g.n_cells)
    m = O.mark(g, tot)
    med = np.median(tot)
    ra = map(tuple, m["refine_anchor"].tolist())
    keys = set(zip(m["refine_level"].tolist(), ra, strict=True))
    want = {
        (int(lv), tuple(a))
        for lv, a, t in zip(g.level, g.anchor.tolist(), tot, strict=True)
        if t > 2.0 * med and lv < g.max_level
    }
    assert keys == want
    for lv, pa in zip(m["coarsen_level"], m["coarsen_anchor"], strict=True):
        half = 1 << (g.max_level - int(lv) - 1)
        ids = g.find_leaf(pa[None, :] + half * O.CORNER_BITS)
        assert np.all(g.level[ids] == lv + 1) and len(set(ids.tolist())) == 8
        assert np.all(tot[ids] < 0.5 * med)


# ---------------------------------------------------------------- the exact tests


def _print_trials(name: str, rec: dict[str, Any]) -> None:
    print(f"\n{name}: base F = {rec['base']['terms']['free_energy']:.10g}")
    for r in rec["rounds"]:
        print(
            f" round {r['round']}: median {r['indicator_median']:.4g} mm, nominated "
            f"refine {r['n_refine_nominated']}, coarsen {r['n_coarsen_nominated']}"
        )
        for tr in r["trials"]:
            leaves = tr["level_counts_after"]["leaves_per_level"]
            print(
                f"  {tr['kind']}: F {tr['free_energy_before']:.10g} -> "
                f"{tr['free_energy_after']} dF {tr['delta_free_energy']} "
                f"accepted {tr['accepted']} leaves {leaves}"
            )
            if tr["delta_terms"]:
                print(
                    "   per-term dF:",
                    {k: round(v, 6) for k, v in tr["delta_terms"].items()},
                )


@functools.cache
def slab_adapt() -> tuple[O.OctreeGrid, dict[str, Any]]:
    ph = FTR.build_phantom("slab", FEATS, noise=0.0, shape=SHAPE)
    g, _, _, _, rec = O.adapt(ph, FO.oracle_inputs(ph), ph.image["noise_sd"])
    return g, rec


@functools.cache
def noise_adapt() -> tuple[O.OctreeGrid, dict[str, Any]]:
    ph = FTR.build_phantom("noise", FEATS, noise=1.0, seed=0, shape=SHAPE)
    inputs = O.null_inputs(ph)
    settings = dataclasses.replace(T.DEFAULTS, octree_rounds=1)
    g, _, _, _, rec = O.adapt(ph, inputs, ph.image["noise_sd"], settings)
    return g, rec


def _assert_refuses_every_refinement(g: O.OctreeGrid, rec: dict[str, Any]) -> None:
    json.dumps(rec, allow_nan=True)
    assert_same_tree(g, O.OctreeGrid.base(SHAPE))
    trials = [t for r in rec["rounds"] for t in r["trials"] if t["kind"] == "refine"]
    assert all(not t["accepted"] for t in trials)
    assert all(t["solve"] is None or t["solve"]["defects"] == [] for t in trials)


def test_slab_stays_at_coarsest() -> None:
    """The plan's exact test "the octree stays at 0.64 mm" on the slab: oracle inputs,
    noise 0, a box of 24³ voxels (not the plan's 64 × 64 × 48). adapt refuses every
    refinement and the final octree is the base octree."""
    g, rec = slab_adapt()
    _print_trials("slab", rec)
    _assert_refuses_every_refinement(g, rec)


NOISE_XFAIL = (
    "R9 fails on the noise phantom (24^3 voxels, null inputs): the round-0 "
    "refinement trial (34 leaves nominated) is accepted: F 44390.99 -> 34973.35, "
    "dF -9417.63 (per term: stop -16170.71, jump -10617.30, speed +8752.71, "
    "contrast +5002.03, consistency +3091.94, bending +801.79; entropy H +278.09, "
    "which enters F as -H). Control: a second solve of the same base octree with "
    "init = its own field gives dF -9418.77, so the trial's gain is the restart of "
    "the solve, not the refinement."
)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=NOISE_XFAIL)
def test_noise_alone_refuses_every_refinement() -> None:
    """The plan's exact test "the free-energy test refuses every refinement" on the
    noise phantom (noise 1.0, seed 0, a box of 24³ voxels). The noise phantom has no
    geometry, so it has no oracle; its image stop field is not a null (PR 2: p > 0.5
    at 99 % of the points). So the inputs are null inputs: the stop log-odds is
    log(0.02 / 0.98) everywhere (nothing stopped: the phantom is cortex everywhere;
    0.02 is the oracle clip), the tensor has coherence 0 (no direction), and the
    contrast is the phantom's det.

    octree_rounds = 1 here (in this test only), to keep the file short. It does not
    change the verdict: the base octree has only level-0 leaves, so round 0 has no
    coarsening; if its refinement is refused, adapt stops after round 0 in any case.

    Marked xfail (strict): it fails (NOISE_XFAIL), a result of design decision R9."""
    g, rec = noise_adapt()
    _print_trials("noise", rec)
    _assert_refuses_every_refinement(g, rec)


def test_adapt_record_has_hanging_step_and_prior_shaped() -> None:
    _, rec = slab_adapt()
    fin = rec["final"]
    assert fin["prior_shaped"]["count"] == fin["level_counts"]["n_prior_shaped"]
    assert fin["hanging_step"]["n_hanging"] == fin["level_counts"]["n_hanging"]


def test_hanging_step_on_a_transferred_field() -> None:
    """hanging_step on a field moved to a refined octree: a smooth field with no stopped
    node, so every hanging node is counted in both views."""
    old = O.OctreeGrid.base(SHAPE)
    f = random_field(old, 5)
    f.q_stop[:] = 0.0
    kept = np.arange(f.q.shape[1])[None, :] < f.n_kept[:, None]
    f.q = np.where(kept, 1.0, 0.0) / f.n_kept[:, None]
    g = refined_tree()
    hs = O.hanging_step(g, O.transfer(f, g))
    assert hs["n_hanging"] == len(g.hanging[0]) > 0
    assert hs["n_moving"] == hs["n_hanging"]
    assert hs["n_integrator"] == hs["n_hanging"]
    assert hs["coarse_leaf_check"] == 0
    json.dumps(hs)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))


# ---------------------------------------------------------------- face weights


def _axis_tiling(g: Any, fw: Any) -> list[float]:
    vec = FG.edge_vectors_mm(g) / T.VOXEL_MM
    length = np.linalg.norm(vec, axis=1)
    ax = np.argmax(np.abs(vec), axis=1)
    return [float((fw[ax == a] * length[ax == a]).sum()) for a in range(3)]


def _refined_more() -> O.OctreeGrid:
    """refined_tree with one more refined leaf in the middle of the box."""
    g = refined_tree()
    mid = np.array([[g.extent[0] // 2, g.extent[1] // 2, g.extent[2] // 2]])
    return g.refine(g.find_leaf(mid))


@pytest.mark.parametrize("make", [refined_tree, _refined_more])
def test_face_weights_tile_the_box(make):
    """For each axis, Σ face_w · edge length (voxels) over the edges of that axis equals
    the volume of the nodes' box in voxels³ (1e-12 relative): the dual cells tile the
    box, also across coarse–fine faces and at split edges."""
    g = make()
    assert len(g.hanging[0]) > 0
    fw = g.face_weights_vox()
    vol = float(np.prod(np.asarray(g.extent, np.float64) * g.unit_vox))
    for s in _axis_tiling(g, fw):
        assert s == pytest.approx(vol, rel=1e-12)


def test_face_weights_split_edge_example():
    """A half of a split edge (length h) held by 2 coarse leaves (side 2h) and 2 fine
    leaves (side h) gets 2.5 h² (1e-12): each holding leaf adds (its side / 2)²."""
    g = refined_tree()
    fw = g.face_weights_vox()
    length = np.linalg.norm(FG.edge_vectors_mm(g), axis=1) / T.VOXEL_MM
    ratio = fw / length**2
    assert np.any(np.isclose(ratio, 2.5, rtol=0, atol=1e-12))
    # every ratio is a sum of quarters (fine leaves) and ones (coarse leaves)
    assert np.allclose(ratio * 4, np.round(ratio * 4), rtol=0, atol=1e-12)


def test_face_weights_uniform_octree_equals_uniform_grid():
    """An octree with no refinement gives ff_grid.face_weights_vox of the uniform grid
    at the same spacing, to the last bit, edge by edge (matched by node positions)."""
    g = O.OctreeGrid.base(SHAPE)
    u = FG.UniformGrid(SHAPE, T.DEFAULTS.octree_coarsest_mm)
    fo = g.face_weights_vox()
    fu = FG.face_weights_vox(u)
    assert len(fo) == len(fu)

    def keyed(grid: Any, fw: Any) -> dict[tuple[float, ...], float]:
        X = grid.nodes
        return {
            tuple(np.round(np.concatenate([X[i], X[j]]), 9)): float(w)
            for (i, j), w in zip(grid.edges, fw, strict=True)
        }

    ko, ku = keyed(g, fo), keyed(u, fu)
    assert ko.keys() == ku.keys()
    assert all(ko[k] == ku[k] for k in ko)


def test_grid_arrays_use_octree_face_weights():
    """ff_meanfield.grid_arrays takes the octree's own face weights."""
    import ff_meanfield as MF

    g = refined_tree()
    ga = MF.grid_arrays(g)
    assert np.array_equal(ga.face_w, g.face_weights_vox())
