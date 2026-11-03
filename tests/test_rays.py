"""The ray sampler (docs/formats/ray-sampler.md): the spec against the code."""

import dataclasses
import inspect
import itertools
import typing
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest

from baillarger import rays, white_dseg, white_surf

DOC = Path(__file__).resolve().parents[1] / "docs" / "formats" / "ray-sampler.md"


def doc_rows(heading: str) -> list[list[str]]:
    """The cells of the table rows (first cell in backticks) under heading."""
    lines = DOC.read_text().splitlines()
    i = lines.index(heading)
    out: list[list[str]] = []
    for line in lines[i + 1 :]:
        if line.startswith("#") and out:
            break
        if line.startswith("| `"):
            out.append([c.strip().strip("`") for c in line.strip("|").split("|")])
    return out


def field_type(t: object) -> str:
    """int64 for npt.NDArray[np.int64], the class name otherwise."""
    args = typing.get_args(t)
    if args:
        return str(args[-1].__name__)
    if t is str:
        return "string"
    return str(getattr(t, "__name__", t))


@pytest.mark.parametrize("cls", [rays.Roots, rays.Rays])
def test_the_spec_lists_the_output_fields(cls: type) -> None:
    hints = typing.get_type_hints(cls)
    want = [[f.name, field_type(hints[f.name])] for f in dataclasses.fields(cls)]
    got = [row[:2] for row in doc_rows(f"### `{cls.__name__}`")]
    assert got == want


@pytest.mark.parametrize(
    ("heading", "names"),
    [
        ("## The label classes", ("OUTSIDE", "WM", "CORTEX", "OTHER", "OFF_GRID")),
        ("## The derived class of a column", ("OPEN", "IN_WM", "NEITHER")),
    ],
)
def test_the_spec_lists_the_class_codes(heading: str, names: tuple[str, ...]) -> None:
    want = [[n, str(getattr(rays, n))] for n in names]
    assert [row[:2] for row in doc_rows(heading)] == want


def test_the_spec_gives_the_constants() -> None:
    text = DOC.read_text()
    assert f"SpecVersion `{rays.VERSION}`" in text
    assert f"`CAP_IN` = {rays.CAP_IN}" in text
    assert f"`CAP_OUT` = {rays.CAP_OUT}" in text
    assert f"`STEP` ({rays.STEP})" in text
    assert rays.METHODS == ("normal",)
    label = str(rays.OFF_GRID_LABEL).replace("-", "\N{MINUS SIGN}")
    assert f"the part of the ray outside the grid. Its label is {label}" in text


def test_the_spec_lists_the_parameters() -> None:
    sig = inspect.signature(rays.RaySampler.sample)
    names = [p for p in sig.parameters if p != "self"]
    assert [row[0] for row in doc_rows("### Parameters")] == names


# ---------------------------------------------------------------- the traversal


def brute_intervals(
    v0: npt.NDArray[np.float64],
    u: npt.NDArray[np.float64],
    lab: npt.NDArray[np.int32],
    smin: float,
    smax: float,
) -> list[tuple[float, float, int]]:
    """The intervals by the rule, the slow way: every face crossing on every
    axis, sorted; the label at the middle of each piece; equal labels
    merged."""
    cuts = [smin, smax]
    for a in range(3):
        if u[a] == 0:
            continue
        lo, hi = sorted((v0[a] + smin * u[a], v0[a] + smax * u[a]))
        for c in range(int(np.floor(lo - 0.5)) - 1, int(np.ceil(hi + 0.5)) + 2):
            s = (c + 0.5 - v0[a]) / u[a]
            if smin < s < smax:
                cuts.append(s)
    cuts = sorted(set(cuts))
    out: list[tuple[float, float, int]] = []
    for s0, s1 in itertools.pairwise(cuts):
        m = v0 + 0.5 * (s0 + s1) * u
        ijk = np.floor(m + 0.5).astype(np.int64)
        inside = bool(np.all((ijk >= 0) & (ijk < np.asarray(lab.shape))))
        here = int(lab[ijk[0], ijk[1], ijk[2]]) if inside else -1
        if out and out[-1][2] == here:
            out[-1] = (out[-1][0], s1, here)
        else:
            out.append((s0, s1, here))
    return out


def traverse(
    v0: npt.NDArray[np.float64], u: npt.NDArray[np.float64], lab: npt.NDArray[np.int32]
) -> list[list[tuple[float, float, int]]]:
    shape = np.asarray(lab.shape, np.int64)
    start, s0, s1, ilab, bad = rays._traverse(
        v0, u, lab, np.zeros(3, np.int64), shape, -rays.CAP_IN, rays.CAP_OUT
    )
    assert bad == -1
    return [
        [(s0[q], s1[q], int(ilab[q])) for q in range(start[c], start[c + 1])]
        for c in range(len(v0))
    ]


def test_the_traversal_follows_the_rule() -> None:
    """Random rays through random labels: the intervals equal the slow
    reference exactly, and tile the caps."""
    rng = np.random.default_rng(1)
    lab = rng.integers(0, 3, (12, 11, 13)).astype(np.int32)
    lab[4:8, 4:8, 4:8] = 7  # runs longer than a voxel
    v0 = rng.uniform(2, 9, (300, 3))
    d = rng.normal(size=(300, 3))
    u = d / np.linalg.norm(d, axis=1, keepdims=True) * rng.uniform(0.3, 2.0, (300, 1))
    u[:20, 1:] = 0  # along an axis
    v0[20:40] = np.round(v0[20:40]) + 0.5  # from a corner of a voxel
    u[20:40] = [0.7, 0.7, 0.7]  # through corners and edges
    v0[40:50, 2] = 5.5  # in a face plane
    u[40:50, 2] = 0
    got = traverse(v0, u, lab)
    for c in range(len(v0)):
        want = brute_intervals(v0[c], u[c], lab, -rays.CAP_IN, rays.CAP_OUT)
        assert got[c] == want, c
        assert got[c][0][0] == -rays.CAP_IN and got[c][-1][1] == rays.CAP_OUT
        assert all(a[1] == b[0] for a, b in itertools.pairwise(got[c]))


def test_a_ray_on_an_axis() -> None:
    """A ray along i from i = 3.25 at 2 voxels per mm through labels set by
    i: the boundaries are at the faces, i = c + 0.5."""
    lab = np.zeros((40, 3, 3), np.int32)
    lab[:5] = 41
    lab[5:9] = 42
    got = traverse(np.array([[3.25, 1.0, 1.0]]), np.array([[2.0, 0.0, 0.0]]), lab)[0]
    # i(s) = 3.25 + 2 s: off the grid below i = -0.5 (s < -1.875), 41 to
    # i = 4.5 (s = 0.625), 42 to i = 8.5 (s = 2.625), then 0 to the cap
    assert got == [
        (-2.0, -1.875, -1),
        (-1.875, 0.625, 41),
        (0.625, 2.625, 42),
        (2.625, 8.0, 0),
    ]


def test_a_ray_that_leaves_the_labels_read_is_refused() -> None:
    lab = np.zeros((4, 4, 4), np.int32)
    shape = np.array([100, 100, 100], np.int64)
    *_, bad = rays._traverse(
        np.array([[2.0, 2.0, 2.0]]),
        np.array([[1.0, 0.0, 0.0]]),
        lab,
        np.zeros(3, np.int64),
        shape,
        -2.0,
        8.0,
    )
    assert bad == 0


def test_the_classes_of_the_labels() -> None:
    lab = np.array([0, 41, 49, 42, 53, 54, 44, -1, 3])
    got = rays.label_classes(lab, [41, 43, 49], 42)
    o, w, c, x, g = rays.OUTSIDE, rays.WM, rays.CORTEX, rays.OTHER, rays.OFF_GRID
    assert got.tolist() == [o, w, w, c, x, x, x, g, x]


@pytest.mark.parametrize(
    ("classes", "want"),
    [
        ([("WM", -2, 0.1), ("CORTEX", 0.1, 3), ("OUTSIDE", 3, 8)], "OPEN"),
        ([("WM", -2, 0.1), ("CORTEX", 0.1, 3), ("WM", 3, 8)], "IN_WM"),
        ([("WM", -2, 0.1), ("CORTEX", 0.1, 8)], "NEITHER"),
        ([("WM", -2, 1), ("OUTSIDE", 1, 8)], "OPEN"),  # a cut face
        ([("CORTEX", -2, 2), ("OTHER", 2, 3), ("WM", 3, 8)], "IN_WM"),
        ([("CORTEX", -2, 2), ("OFF_GRID", 2, 8)], "NEITHER"),
        ([("OUTSIDE", -2, 0), ("WM", 0, 1), ("CORTEX", 1, 8)], "NEITHER"),
        ([("WM", -2, 4), ("CORTEX", 4, 5), ("WM", 5, 6), ("OUTSIDE", 6, 8)], "IN_WM"),
    ],
)
def test_the_derived_class(classes: list[tuple[str, float, float]], want: str) -> None:
    cls = np.array([getattr(rays, k) for k, _, _ in classes], np.uint8)
    s1 = np.array([b for _, _, b in classes], np.float64)
    got = rays._derived(np.array([0, len(cls)], np.int64), s1, cls)
    assert got[0] == getattr(rays, want)


# ---------------------------------------------------------------- phantoms

VS = 0.16
PHANTOM_AFF = np.diag([VS, VS, VS, 1.0])
PHANTOM_AFF[:3, 3] = [-3.0, 2.0, 5.0]


def write_phantom(d: Path, a: npt.NDArray[np.int16]) -> tuple[Path, Path]:
    """Labels a (a right hemisphere) through the dseg and the mesh commands
    (wm-dseg/1, white-surf/1, 8-cell blocks): the surface and the dseg."""
    src = d / "in" / "sub-T01_ses-S1_aseg.nii.gz"
    src.parent.mkdir(parents=True)
    nib.save(nib.nifti1.Nifti1Image(a, PHANTOM_AFF), src)
    out = d / "deriv"
    white_dseg.main([str(src), "--out", str(out)])
    anat = out / "sub-T01/ses-S1/anat"
    dseg = anat / "sub-T01_ses-S1_desc-wmtopo_dseg.nii.gz"
    white_surf.main([str(dseg), "--out", str(out), "--block-cells", "8"])
    return anat / "sub-T01_ses-S1_hemi-R_white.surf.gii", dseg


def slabs(*runs: tuple[int, int, int]) -> npt.NDArray[np.int16]:
    """Labels on a 40 x 40 x 40 grid: a box of cortex (i, j in [3, 37), k in
    [3, 37)) in outside, with layers (k0, k1, label) across it."""
    a = np.zeros((40, 40, 40), np.int16)
    a[3:37, 3:37, 3:37] = 42
    for k0, k1, lab in runs:
        a[6:34, 6:34, k0:k1] = lab
    return a


def ball() -> npt.NDArray[np.int16]:
    x = np.indices((40, 40, 40)).astype(float)
    r = np.sqrt(((x - np.array([19.6, 20.2, 19.9])[:, None, None, None]) ** 2).sum(0))
    a = np.zeros((40, 40, 40), np.int16)
    a[r < 14] = 42
    a[r < 8] = 41
    return a


PHANTOMS = {
    "slab": slabs((6, 16, 41)),
    "ball": ball(),
    "fused": slabs((6, 14, 41), (22, 30, 41)),
    "slot": slabs((6, 14, 41), (17, 19, 0), (22, 30, 41)),
    "blade": slabs((18, 20, 41)),
}


@pytest.fixture(scope="module")
def phantom(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, tuple[rays.RaySampler, npt.NDArray[np.int16]]]:
    out = {}
    for name, a in PHANTOMS.items():
        surf, dseg = write_phantom(tmp_path_factory.mktemp(name), a)
        sampler = rays.RaySampler(surf, dseg)
        out[name] = (sampler, np.asarray(nib.load(dseg).dataobj))  # type: ignore[attr-defined]
    return out


def every_column(sampler: rays.RaySampler) -> rays.Rays:
    """The columns of the whole surface: one superblock that holds every
    block (40 voxels, 8-cell blocks: level 3)."""
    return sampler.sample((0, 0, 0), 3)


def column(r: rays.Rays, c: int) -> list[tuple[float, float, int]]:
    q = range(r.interval_start[c], r.interval_start[c + 1])
    return [
        (float(r.interval_s0[i]), float(r.interval_s1[i]), int(r.interval_label[i]))
        for i in q
    ]


def top_face(r: rays.Rays, k: float) -> npt.NDArray[np.int64]:
    """The columns whose root is on the face k (voxels) of the slabs, away
    from its edges, with a normal along +k."""
    v = (r.roots.root.astype(np.float64) - PHANTOM_AFF[:3, 3]) / VS
    sel = (
        (np.abs(v[:, 2] - k) < 1)
        & np.all((v[:, :2] > 12) & (v[:, :2] < 28), axis=1)
        & (r.roots.normal[:, 2] > 0.999)
    )
    assert sel.sum() > 50
    return np.flatnonzero(sel)


@pytest.mark.parametrize("name", list(PHANTOMS))
def test_the_intervals_follow_the_rule_on_the_files(
    phantom: dict[str, tuple[rays.RaySampler, npt.NDArray[np.int16]]], name: str
) -> None:
    """Every column of a phantom: its intervals are the slow reference's on
    the whole label array, and tile the caps; its classes follow its labels."""
    sampler, lab = phantom[name]
    r = every_column(sampler)
    assert len(r.roots.vertex) == sampler.sidecar["Vertices"]
    v0, u = rays._voxel_rays(
        r.roots.root.astype(np.float64),
        r.roots.normal.astype(np.float64),
        sampler.inverse,
    )
    for c in range(len(v0)):
        want = brute_intervals(v0[c], u[c], lab.astype(np.int32), -2.0, 8.0)
        got = column(r, c)
        assert [g[2] for g in got] == [w[2] for w in want]
        assert np.array_equal(
            np.array([g[:2] for g in got], np.float32),
            np.array([w[:2] for w in want], np.float32),
        )
    want_cls = rays.label_classes(r.interval_label, [41], 42)
    assert np.array_equal(r.interval_class, want_cls)


def boundary(r: rays.Rays, c: int, before: int, after: int) -> float:
    """s > 0 where the column first goes from class before to class after."""
    for q in range(r.interval_start[c], r.interval_start[c + 1] - 1):
        if r.interval_s1[q] <= 0:
            continue
        if r.interval_class[q] == before and r.interval_class[q + 1] == after:
            return float(r.interval_s1[q])
    raise AssertionError(f"column {c}: no {before} -> {after}")


def test_a_flat_slab(phantom: dict[str, tuple[rays.RaySampler, Any]]) -> None:
    """At the depths the geometry gives, from a root at k0 on the slab's top
    face (WM in k [6, 16), cortex to k = 37): inward, the WM (1.6 mm) and
    the cortex below it; outward, the cortex to the face k = 36.5, outside,
    and off the grid past k = 39.5."""
    r = every_column(phantom["slab"][0])
    for c in top_face(r, 15.5):
        k0 = (float(r.roots.root[c, 2]) - 5.0) / VS
        q = range(*r.interval_start[c : c + 2])
        assert [r.interval_class[i] for i in q] == [
            rays.CORTEX,
            rays.WM,
            rays.CORTEX,
            rays.OUTSIDE,
            rays.OFF_GRID,
        ]
        want = np.array([5.5, 15.5, 36.5, 39.5]) - k0
        assert np.allclose(r.interval_s1[q][:-1], want * VS, atol=1e-5)
        assert r.derived_class[c] == rays.OPEN


def test_a_ball(phantom: dict[str, tuple[rays.RaySampler, Any]]) -> None:
    """Near-radial rays: the cortex ends at the outer radius, give or take
    the staircase of the voxels (half a voxel diagonal)."""
    r = every_column(phantom["ball"][0])
    centre = np.array([19.6, 20.2, 19.9]) * VS + PHANTOM_AFF[:3, 3]
    x = r.roots.root.astype(np.float64)
    n = r.roots.normal.astype(np.float64)
    radial = np.sum(n * (x - centre), 1) / np.linalg.norm(x - centre, axis=1)
    assert radial.min() > 0.95
    for c in range(len(x)):
        s = boundary(r, c, rays.CORTEX, rays.OUTSIDE)
        rad = np.linalg.norm(x[c] + s * n[c] - centre) / VS
        assert 14 - 0.9 < rad < 14 + 0.9
    assert np.all(r.derived_class == rays.OPEN)


def test_fused_banks_run_into_the_far_wm(
    phantom: dict[str, tuple[rays.RaySampler, Any]],
) -> None:
    """Cortex to cortex with no fluid: from the lower bank's WM through the
    cortex into the far bank's WM, at the face k = 21.5."""
    r = every_column(phantom["fused"][0])
    for c in top_face(r, 13.5):
        k0 = (float(r.roots.root[c, 2]) - 5.0) / VS
        assert abs(boundary(r, c, rays.CORTEX, rays.WM) - (21.5 - k0) * VS) < 1e-3
        assert r.derived_class[c] == rays.IN_WM


def test_an_open_slot(phantom: dict[str, tuple[rays.RaySampler, Any]]) -> None:
    r = every_column(phantom["slot"][0])
    for c in top_face(r, 13.5):
        k0 = (float(r.roots.root[c, 2]) - 5.0) / VS
        assert abs(boundary(r, c, rays.CORTEX, rays.OUTSIDE) - (16.5 - k0) * VS) < 1e-3
        assert r.derived_class[c] == rays.OPEN


def test_the_inward_ray_crosses_a_thin_blade(
    phantom: dict[str, tuple[rays.RaySampler, Any]],
) -> None:
    """A WM blade two voxels thick (k in [18, 20)): from its upper face the
    inward ray crosses the blade and enters the far cortex at k = 17.5."""
    r = every_column(phantom["blade"][0])
    for c in top_face(r, 19.5):
        k0 = (float(r.roots.root[c, 2]) - 5.0) / VS
        q = range(r.interval_start[c], r.interval_start[c + 1])
        inward = [i for i in q if r.interval_s0[i] < 0]
        cls = [r.interval_class[i] for i in inward]
        assert cls[-2:] == [rays.CORTEX, rays.WM] or cls[-3:-1] == [
            rays.CORTEX,
            rays.WM,
        ]
        far = max(
            float(r.interval_s1[i])
            for i in inward
            if r.interval_class[i] == rays.CORTEX and r.interval_s1[i] < 0
        )
        assert abs(far - (17.5 - k0) * VS) < 1e-3


def test_the_extents_only_add_samples(
    phantom: dict[str, tuple[rays.RaySampler, Any]],
) -> None:
    """A shorter ray has the same samples at each s as a longer one, and the
    same intervals; per-column extents too."""
    sampler = phantom["ball"][0]
    full = sampler.sample((0, 0, 0), 3)
    n = len(full.roots.vertex)
    e_out = np.linspace(0.0, 8.0, n)
    short = sampler.sample((0, 0, 0), 3, e_in=0.5, e_out=e_out)
    for name in ("interval_start", "interval_s0", "interval_s1", "interval_label"):
        assert np.array_equal(getattr(short, name), getattr(full, name))
    assert np.array_equal(full.k_in, np.full(n, 50)) and np.all(full.k_out == 200)
    assert np.all(short.k_in == 12)
    for c in range(0, n, 7):
        a, b = short.sample_start[c], short.sample_start[c + 1]
        off = full.sample_start[c] + 50 - 12
        assert b - a == 12 + short.k_out[c] + 1
        assert np.array_equal(short.s[a:b], full.s[off : off + b - a])
        assert np.array_equal(short.position[a:b], full.position[off : off + b - a])
    # the step changes the samples, not the intervals
    coarse = sampler.sample((0, 0, 0), 3, step=0.1)
    assert np.array_equal(coarse.interval_s0, full.interval_s0)


def test_a_column_is_the_same_in_every_superblock(
    phantom: dict[str, tuple[rays.RaySampler, Any]],
) -> None:
    """Superblocks at level 1 (16 voxels, 2.56 mm) with a 3 mm halo: each
    column owned by one and in the halo of another is bit for bit the same."""
    sampler = phantom["ball"][0]
    calls = {
        (i, j, k): sampler.sample((i, j, k), 1, halo_mm=3.0, e_out=5.0)
        for i, j, k in itertools.product(range(3), repeat=3)
    }
    owned: dict[int, tuple[tuple[int, int, int], int]] = {}
    for sb, r in calls.items():
        for c in np.flatnonzero(r.roots.owned):
            assert int(r.roots.edge_key[c]) not in owned
            owned[int(r.roots.edge_key[c])] = (sb, int(c))
    assert len(owned) == sampler.sidecar["Vertices"]
    pairs = 0
    for r in calls.values():
        for c in np.flatnonzero(~r.roots.owned):
            sb, c2 = owned[int(r.roots.edge_key[c])]
            assert as_column(r, int(c)) == as_column(calls[sb], c2)
            pairs += 1
    assert pairs > 1000


def as_column(r: rays.Rays, c: int) -> tuple[bytes, ...]:
    """Every value of column c but its owned flag, as bytes."""
    a, b = r.sample_start[c], r.sample_start[c + 1]
    p, q = r.interval_start[c], r.interval_start[c + 1]
    fields = [f.name for f in dataclasses.fields(r.roots) if f.name != "owned"]
    parts = [getattr(r.roots, name)[c] for name in fields]
    parts += [r.k_in[c], r.k_out[c], r.derived_class[c]]
    parts += [r.s[a:b], r.position[a:b]]
    parts += [
        x[p:q]
        for x in (r.interval_s0, r.interval_s1, r.interval_label, r.interval_class)
    ]
    return tuple(np.asarray(x).tobytes() for x in parts)


def test_the_halo_columns_follow_the_definition(
    phantom: dict[str, tuple[rays.RaySampler, Any]],
) -> None:
    sampler = phantom["ball"][0]
    every = sampler.roots((0, 0, 0), 3)
    v = rays._to_voxel(every.root, sampler.inverse)
    for sb, h in [((1, 1, 1), 0.5), ((0, 1, 2), 2.0), ((2, 2, 2), 0.0)]:
        r = sampler.roots(sb, 1, halo_mm=h)
        lo = np.array(sb) * 16 - h / VS
        hi = (np.array(sb) + 1) * 16 + h / VS
        inbox = set(every.vertex[np.all((v >= lo) & (v <= hi), axis=1)].tolist())
        assert set(r.vertex[~r.owned].tolist()) == inbox - set(
            r.vertex[r.owned].tolist()
        )
        assert np.all(np.diff(r.vertex[r.owned]) > 0)
        assert np.all(np.diff(r.vertex[~r.owned]) > 0)


def test_the_call_refuses_bad_parameters(
    phantom: dict[str, tuple[rays.RaySampler, Any]],
) -> None:
    sampler = phantom["ball"][0]
    for kw in (
        {"e_in": 2.5},
        {"e_out": -1.0},
        {"step": 0.0},
        {"method": "equivolume"},
        {"halo_mm": -1.0},
    ):
        with pytest.raises(ValueError):
            sampler.sample((0, 0, 0), 1, **kw)


def test_the_normal_length_is_the_roots(
    phantom: dict[str, tuple[rays.RaySampler, Any]],
) -> None:
    """Each column carries its root's |n̄|: 1 on the flat face of the blade,
    below 1 at its edges (|n̄| measures the last smoothing step only; its
    exact values are phase 1.0's tests)."""
    sampler = phantom["blade"][0]
    r = every_column(sampler)
    assert r.roots.normal_length.shape == r.roots.vertex.shape
    assert np.array_equal(r.roots.normal_length, sampler.normlen[r.roots.vertex])
    assert np.all((r.roots.normal_length >= 0) & (r.roots.normal_length <= 1.0001))
    assert np.all(r.roots.normal_length[top_face(r, 19.5)] > 0.999)
    assert r.roots.normal_length.min() < 0.99


def test_the_labels_come_through_the_chunk_cache(
    phantom: dict[str, tuple[rays.RaySampler, Any]],
) -> None:
    """A call reads its label box through the cache: a second call on the
    same superblock reads no chunk again."""
    sampler = phantom["ball"][0]
    sampler.sample((1, 1, 1), 1, halo_mm=1.0)
    before = sampler.labels.reads
    sampler.sample((1, 1, 1), 1, halo_mm=1.0)
    assert before > 0 and sampler.labels.reads == before
