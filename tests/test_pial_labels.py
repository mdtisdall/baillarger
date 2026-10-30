"""Tests for baillarger.pial_labels: the noise level, the darkness test, the
components' shape, cells, position and topology, the change file, and the
proposals on synthetic boxes with known channels, vessels and a dark stria."""

from pathlib import Path

import numpy as np
import pytest
from test_white import read_labels, write_data

from baillarger import column_traces as CT
from baillarger import pial_labels as PL
from baillarger import white
from baillarger.data import RIM_CORTEX, RIM_OUTSIDE, RIM_WM, Patch

VOX = 0.16
CORTEX, WM, FLUID = (900.0, 500.0, 300.0), (800.0, 400.0, 230.0), (45.0, 45.0, 45.0)


def synthetic(
    rim: np.ndarray, dark: dict[float, np.ndarray], noise: float = 0.05, seed: int = 0
) -> CT.Tissue:
    """A Tissue from rim labels (1 fluid, 2 WM, 3 cortex) and echoes: each
    label at its level (CORTEX, WM, FLUID), with the voxels of dark[f] a
    mixture of fluid fraction f and cortex, times log-normal noise."""
    shape = rim.shape
    e = np.zeros((*shape, 3), np.float32)
    for lab, lev in ((3, CORTEX), (2, WM), (1, FLUID)):
        e[rim == lab] = lev
    for f, m in dark.items():
        e[m] = (1 - f) * np.array(CORTEX) + f * np.array(FLUID)
    rng = np.random.default_rng(seed)
    e *= np.exp(noise * rng.standard_normal(e.shape)).astype(np.float32)
    patch = Patch(
        name="synthetic",
        lo=(0, 0, 0),
        hi=(shape[0], shape[1], shape[2]),
        echoes=e,
        echo_times_ms=(1.0, 2.0, 3.0),
        aseg=np.zeros(shape, np.int16),
        rim=rim.astype(np.int16),
        affine=np.eye(4),
        voxel_mm=(VOX, VOX, VOX),
    )
    t = CT.tissue(patch)
    t.dface = np.full(shape, 99.0)
    return t


def proposed_mask(p: PL.Proposal) -> np.ndarray:
    """Box voxels of review cells that would apply: accepted or proposed."""
    ch = p.changes
    out = np.zeros(p.comp.shape, bool)
    sel = ch.status[ch.component] != PL.REJECTED
    out[tuple((ch.voxels[sel] - np.array(ch.lo)).T)] = True
    return out


# ---------------------------------------------------------------- noise


def test_gradient_gain_of_central_differences_is_one_half() -> None:
    """Without smoothing the derivative taps are +-1/2: gain 1/4 + 1/4."""
    assert PL.gradient_gain(0.0) == pytest.approx(0.5, abs=1e-12)


def test_wishart_factor_is_one_for_many_degrees_of_freedom() -> None:
    """W / n tends to the identity: its smallest eigenvalue to 1, with a
    spread of order sqrt(2 / n)."""
    assert PL.wishart_min_factor(1e6, draws=20_000) == pytest.approx(1.0, abs=5e-3)
    assert PL.wishart_min_factor(39.0) < 0.75


def test_echo_noise_recovers_white_noise() -> None:
    """White noise of 0.2 in a cube: the structure tensor's median smallest
    eigenvalue, corrected by the gain and the Wishart factor, gives 0.2
    within the model's few per cent (high: echo_noise)."""
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 0.2, (64, 64, 64))
    m = np.zeros(x.shape, bool)
    m[8:-8, 8:-8, 8:-8] = True
    assert 0.2 <= PL.echo_noise(x * m, m) <= 0.2 * 1.06


def test_echo_noise_ignores_layers() -> None:
    """Layers along one axis (amplitude 0.5, period 1 mm) add power to the
    largest eigenvalue only, to first order: the noise of a layered cube is
    the noise alone, within 8 %."""
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 0.2, (64, 64, 64))
    x += 0.5 * np.sin(np.arange(64) * VOX / 0.25)[:, None, None]
    m = np.zeros(x.shape, bool)
    m[8:-8, 8:-8, 8:-8] = True
    assert 0.2 <= PL.echo_noise(x * m, m) <= 0.2 * 1.08


def test_min_component_is_the_first_size_expected_less_than_once() -> None:
    """N p (e 25 p)^(n - 1) for N = 1e6, p = 1e-3: 1000, 68, 4.6, 0.31."""
    assert PL.min_component(1_000_000, 1e-3) == 4
    assert PL.min_component(100, 1e-3) == 1


# ---------------------------------------------------------------- darkness


def slab_rim(shape: tuple[int, int, int] = (32, 24, 24)) -> np.ndarray:
    """WM for x < 6, cortex for 6 <= x < 26, fluid above."""
    rim = np.full(shape, 1)
    rim[:6] = 2
    rim[6:26] = 3
    return rim


def test_fluid_fraction_of_a_mixture() -> None:
    """Voxels mixing 80 % fluid with cortex: a fluid fraction of 0.8 (to the
    noise), and dark; 20 %: not dark."""
    rim = slab_rim()
    m8 = np.zeros(rim.shape, bool)
    m8[15, 10:13, 10:13] = True
    m2 = np.zeros(rim.shape, bool)
    m2[15, 16:19, 16:19] = True
    t = synthetic(rim, {0.8: m8, 0.2: m2}, noise=0.01)
    dk = PL.darkness(t)
    assert np.allclose(dk.frac[m8], 0.8, atol=0.03)
    assert np.allclose(dk.frac[m2], 0.2, atol=0.03)
    assert dk.dark[m8].all() and not dk.dark[m2].any()
    assert dk.fluid[0] == pytest.approx(45.0, rel=0.02)


def test_darkness_reads_echo_1() -> None:
    """Voxels at the fluid level in echo 1 only (cortex level in echoes 2 and
    3): their contrast is negative only through echo 1, by its noise weight,
    and their fluid fraction is echo 1's share of the weights."""
    rim = slab_rim()
    m = np.zeros(rim.shape, bool)
    m[15, 10:13, 10:13] = True
    t = synthetic(rim, {}, noise=0.01)
    t.patch.echoes[m, 0] = FLUID[0]
    dk = PL.darkness(t)
    assert PL.ECHOES == (0, 1, 2) and dk.d.shape[0] == 3
    assert np.all(dk.d[0][m] < -2.5) and np.all(np.abs(dk.d[1:][:, m]) < 0.1)
    assert np.all(dk.z[m] < -dk.k)
    w = [((1 - FLUID[i] / CORTEX[i]) / dk.sigma[i]) ** 2 for i in range(3)]
    assert np.allclose(dk.frac[m], w[0] / sum(w), atol=0.05)


# ------------------------------------------------- shape, cells, topology


def test_shape_kinds() -> None:
    """A line is a tube, a square a sheet, a cube a blob."""
    line = np.array([[i, 0, 0] for i in range(20)])
    sq = np.array([[i, j, 0] for i in range(10) for j in range(10)])
    cube = np.array([[i, j, k] for i in range(5) for j in range(5) for k in range(5)])
    assert PL.shape_kind(line, [VOX] * 3)[0] == "tube"
    assert PL.shape_kind(sq, [VOX] * 3)[0] == "sheet"
    assert PL.shape_kind(cube, [VOX] * 3)[0] == "blob"


def test_review_cells_of_a_line() -> None:
    """A line of 30 voxels, cells 6 voxels (0.96 mm) apart: centres at 0, 6,
    ..., 24, so 5 cells, each a run of consecutive voxels."""
    line = np.array([[i, 0, 0] for i in range(30)])
    cell = PL.review_cells(line, [VOX] * 3, 0.96)
    assert cell.max() == 4 and (cell >= 0).all()
    assert np.all(np.diff(cell) >= 0)


def test_topology_change_of_a_slot_a_cavity_and_a_tunnel() -> None:
    """Tissue cube: a slot from its face changes nothing; an enclosed
    pocket is a cavity (b2 + 1); a tube through it a tunnel (b1 + 1)."""
    tissue = np.zeros((12, 12, 12), bool)
    tissue[1:-1, 1:-1, 1:-1] = True
    slot = np.array([[i, j, 5] for i in range(1, 6) for j in range(1, 11)])
    pocket = np.array([[6, 6, 6]])
    tube = np.array([[i, 6, 6] for i in range(1, 11)])
    assert PL.topology_change(tissue, slot, margin=6) == ((1, 0, 0), (1, 0, 0))
    assert PL.topology_change(tissue, pocket, margin=6) == ((1, 0, 0), (1, 0, 1))
    assert PL.topology_change(tissue, tube, margin=6) == ((1, 0, 0), (1, 1, 0))


def test_traces_through_counts_each_trace_once() -> None:
    """Two traces: one crosses component 1 twice and component 2, one
    passes nothing; samples past ns do not count."""
    lab = np.zeros((10, 10, 10), np.int64)
    lab[2, 2, 2] = lab[2, 2, 4] = 1
    lab[5, 5, 5] = 2
    lab[8, 8, 8] = 2
    pos = np.full((2, 6, 3), np.nan, np.float32)
    pos[0, :4] = [[2, 2, 2], [2, 2, 3], [2, 2, 4], [5, 5, 5]]
    pos[1, :3] = [[0, 0, 0], [1, 1, 1], [8, 8, 8]]
    out = PL.traces_through(lab, pos, np.array([4, 2]))
    assert out.tolist() == [0, 1, 1]


# ---------------------------------------------------------------- change file


def changes() -> PL.LabelChanges:
    return PL.LabelChanges(
        lo=(10, 10, 10),
        hi=(14, 14, 14),
        voxels=np.array([[11, 11, 11], [12, 12, 12], [13, 13, 13]], np.int32),
        label=np.full(3, RIM_OUTSIDE, np.int8),
        component=np.array([0, 1, 2], np.int32),
        status=np.array([PL.ACCEPTED, PL.PROPOSED, PL.REJECTED], np.int8),
        components=[dict(id=i, rank=i, voxels=1, kind="blob") for i in range(3)],
    )


def test_apply_takes_the_best_guess_or_the_reviewed_cells(
    tmp_path: Path,
) -> None:
    """Accepted and proposed apply (proposed not with reviewed_only), rejected
    never; a later
    brush edit of a voxel wins; WM is never changed; the file round-trips."""
    rim = np.full((4, 4, 4), RIM_CORTEX, np.int16)
    rim[0, 0, 0] = 2
    ch = changes().with_brush([[13, 13, 13], [10, 10, 10]], RIM_OUTSIDE)
    ch = ch.with_brush([[11, 11, 11]], RIM_CORTEX)
    ch.save(tmp_path / "c.npz")
    ch = PL.LabelChanges.load(tmp_path / "c.npz")
    a = ch.apply(rim, (10, 10, 10))
    assert a[1, 1, 1] == RIM_CORTEX  # accepted, then brushed back
    assert a[2, 2, 2] == RIM_OUTSIDE  # proposed: the best guess
    assert a[3, 3, 3] == RIM_OUTSIDE  # rejected, then brushed
    assert a[0, 0, 0] == 2  # WM
    b = ch.apply(rim, (10, 10, 10), reviewed_only=True)
    assert b[2, 2, 2] == RIM_CORTEX and b[1, 1, 1] == RIM_CORTEX
    assert ch.components[1]["kind"] == "blob"
    ov = PL.overlay(ch)
    assert ov[2, 2, 2] == 1 and ov[1, 1, 1] == PL.OV_BRUSH_CORTEX
    assert ov[3, 3, 3] == PL.OV_BRUSH_FLUID


def test_tissue_applies_label_changes() -> None:
    """column_traces.tissue with label changes: the accepted voxel is fluid."""
    rim = slab_rim()
    t0 = synthetic(rim, {})
    ch = PL.LabelChanges(
        lo=(0, 0, 0),
        hi=rim.shape,
        voxels=np.array([[20, 5, 5]], np.int32),
        label=np.array([RIM_OUTSIDE], np.int8),
        component=np.array([0], np.int32),
        status=np.array([PL.ACCEPTED], np.int8),
    )
    t = CT.tissue(t0.patch, pial_changes=ch)
    assert t.labels[20, 5, 5] == 1 and t0.labels[20, 5, 5] == 0
    assert ch.finalize(t0.rim).betti_after == (1, 0, 1)  # a fluid pocket


# ---------------------------------------------------------------- proposals


def fused_banks() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two WM slabs (x < 6, x >= 30) with cortex between them below y = 28
    and fluid above; a hidden fluid channel at the WM midline (x = 17, 18;
    y < 22, so it does not reach the fluid) and a dark stria band in each
    bank (x = 11, 12 and 23, 24; y < 22), both labelled cortex."""
    shape = (36, 32, 20)
    rim = np.full(shape, 3)
    rim[:6] = 2
    rim[30:] = 2
    y = np.indices(shape)[1]
    rim[(rim == 3) & (y >= 28)] = 1
    chan = np.zeros(shape, bool)
    chan[17:19, :22] = True
    stria = np.zeros(shape, bool)
    stria[11:13, :22] = True
    stria[23:25, :22] = True
    return rim, chan, stria, y


def test_stria_trap_on_fused_banks() -> None:
    """The hidden channel lies where the two banks' fields meet: proposed.
    The stria bands are as dark as a 70 % fluid mixture, so darkness alone
    would take them for fluid; T passes through them within each bank:
    none of their voxels is proposed."""
    rim, chan, stria, _ = fused_banks()
    t = synthetic(rim, {1.0: chan, 0.7: stria})
    p = PL.propose(t)
    prop = proposed_mask(p)
    assert p.dark.dark[stria].mean() > 0.9  # darkness alone would take them
    assert not prop[stria].any()
    inner = chan.copy()
    inner[:, :2] = False  # the box face
    assert prop[inner].mean() > 0.95
    rows = [r for r in p.changes.components if r["auto"] == "inside"]
    assert rows and all(r["group_pass_through"] > 0.5 for r in rows)
    # the channel reaches no fluid: it waits for review
    st = p.changes.status[p.cell[inner] - 1]
    assert (st == PL.PROPOSED).mean() > 0.95


def test_stria_trap_on_a_slab_with_a_surface_vessel() -> None:
    """One bank (slab): a dark stria band at mid depth (x = 15, 16) is not
    proposed; a vessel (a tube along z, x = 24, 25, y = 11, 12) on the pial
    surface, labelled cortex, is proposed, and its shape is a tube."""
    rim = slab_rim((32, 24, 40))
    stria = np.zeros(rim.shape, bool)
    stria[15:17] = True
    vessel = np.zeros(rim.shape, bool)
    vessel[24:26, 11:13] = True
    t = synthetic(rim, {0.7: stria, 1.0: vessel})
    p = PL.propose(t)
    prop = proposed_mask(p)
    assert p.dark.dark[stria].mean() > 0.9
    assert not prop[stria].any()
    assert prop[vessel].mean() > 0.95
    # it touches the fluid and keeps the topology: accepted without review
    assert (p.changes.status[p.cell[vessel] - 1] == PL.ACCEPTED).mean() > 0.95
    g = {r["group"] for r in p.changes.components if prop[tuple(r["at_box0"])]}
    assert len(g) == 1
    pts = np.argwhere(p.comp == g.pop() + 1)
    assert PL.shape_kind(pts, t.vox)[0] == "tube"


def test_relabelled_channel_removes_the_crest() -> None:
    """With the proposals applied, the fused banks' hidden channel is fluid:
    T's crest no longer lies at the midline's cortex."""
    rim, chan, stria, _ = fused_banks()
    t = synthetic(rim, {1.0: chan, 0.7: stria})
    p = PL.propose(t)
    before = p.crest[16:20, 4:20].sum()
    t2 = CT.tissue(t.patch, pial_changes=p.changes)
    inner = chan.copy()
    inner[:, :2] = False
    assert (t2.labels[inner] == 1).mean() > 0.95
    v, f, nrm = CT.white_mesh(t2.wm)
    C = CT.fitted_crest(t2, CT.base_tensor(t2), v, f, nrm)
    assert before > 50
    assert C.crest[16:20, 4:20].sum() < 0.1 * before


def test_the_command_reads_its_labels_from_a_dseg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """propose --dseg: the box's rim labels are the corrected segmentation's,
    with its WM mask as WM (the tunnel's plug and the ventricle are WM)."""
    _, p = write_data(tmp_path)
    monkeypatch.setenv("BAILLARGER_DATA", str(tmp_path))
    lo, hi = (5, 3, 2), (25, 25, 22)
    out = tmp_path / "out"
    argv = ["propose", "--out", str(out), "--name", "blk", "--dseg", str(p)]
    PL.main(argv + ["--lo", *map(str, lo), "--hi", *map(str, hi)])
    sl = tuple(slice(q, r) for q, r in zip(lo, hi, strict=True))
    mask = np.isin(read_labels(p), white.filled_labels(p))[sl]
    rim = read_labels(out / "pial_labels_blk_labels.nii.gz")
    assert np.array_equal(rim == RIM_WM, mask)
