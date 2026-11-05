"""The ray sampler (docs/formats/ray-sampler.md): the spec against the code."""

import dataclasses
import inspect
import itertools
import json
import typing
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest
from nibabel.gifti.gifti import GiftiImage

from baillarger import rays, white_dseg, white_surf
from baillarger.bids import file_sha256, read_sidecar, sidecar_path

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
    assert rays.METHODS == ("normal", "equivolume")
    assert f"`FACE_MARGIN` ({rays.FACE_MARGIN})" in text
    assert rays.U_IN == 2.0**-23 and "`U_IN` = 2⁻²³" in text
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
    column owned by one and in the halo of another is bit for bit the same,
    its cross-section too (equivolume)."""
    sampler = phantom["ball"][0]
    calls = {
        (i, j, k): sampler.sample(
            (i, j, k), 1, halo_mm=3.0, e_out=5.0, method="equivolume"
        )
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
    if len(r.area_coef):
        parts += [r.area_coef[c], r.focus[c]]
        parts += [r.area[a:b], r.volume[a:b], r.past_focus[a:b]]
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
        {"method": "bogus"},
        {"halo_mm": -1.0},
    ):
        with pytest.raises(ValueError):
            sampler.sample((0, 0, 0), 1, **kw)


def test_the_normal_length_is_the_roots(
    phantom: dict[str, tuple[rays.RaySampler, Any]],
) -> None:
    """Each column carries its root's |n̄| from the surface's file. The
    values of |n̄| are phase 1.0's tests; here they only differ between the
    roots, so that a wrong vertex order shows."""
    sampler = phantom["blade"][0]
    r = every_column(sampler)
    assert r.roots.normal_length.shape == r.roots.vertex.shape
    assert np.array_equal(r.roots.normal_length, sampler.normlen[r.roots.vertex])
    assert np.all((r.roots.normal_length >= 0) & (r.roots.normal_length <= 1.0001))
    assert len(np.unique(r.roots.normal_length)) > 1


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


# ---------------------------------------------------------------- equivolume


def reshape_phantom(
    surf: Path,
    fn: typing.Callable[
        [npt.NDArray[np.float64]],
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    ],
) -> None:
    """Moves the vertices of a white-surf/1 phantom in place: fn maps the
    positions (scanner mm, float64) to new positions and unit normals
    (float64), which are rounded to float32 and written into the .dat files.
    The faces, edge keys and blocks stay; the block boxes and the data
    hashes follow the new positions."""
    side = read_sidecar(sidecar_path(surf), white_surf.SURF_SPEC)
    nv = int(side["Vertices"])
    dat = surf.with_name(surf.name.split(".", 1)[0] + ".dat")
    ndat = surf.with_name(side["NormalsFile"].split(".", 1)[0] + ".dat")
    x = np.memmap(dat, "<f4", "r+", shape=(nv, 3))
    n = np.memmap(ndat, "<f4", "r+", shape=(nv, 3))
    nx, nn = fn(np.asarray(x, np.float64))
    x[:] = nx.astype(np.float32)
    n[:] = nn.astype(np.float32)
    x.flush()
    n.flush()
    aff = np.asarray(side["VoxelToRAS"], np.float64).reshape(4, 4)
    v = white_surf.mm_to_voxel(np.asarray(x), aff).astype(np.float64)
    g = GiftiImage.from_filename(str(surf))
    f = np.asarray(g.darrays[1].data, np.int64)
    bpath = surf.with_name(side["BlocksFile"])
    rows = np.loadtxt(bpath, skiprows=1, ndmin=2)
    with bpath.open("w") as fh:
        fh.write("\t".join(white_surf.BLOCK_COLUMNS) + "\n")
        for r in rows:
            p = v[f[int(r[5]) : int(r[6])]].reshape(-1, 3)
            lo = np.floor(p.min(0) * 1000) / 1000
            hi = np.ceil(p.max(0) * 1000) / 1000
            ints = "\t".join(str(int(z)) for z in r[:7])
            fh.write(ints + "\t" + "\t".join(f"{z:.3f}" for z in (*lo, *hi)) + "\n")
    names = [dat, ndat]
    side["DataSHA256"].update({d.name: file_sha256(d) for d in names})
    sidecar_path(surf).write_text(json.dumps(side, indent=2))


CENTRE = np.array([19.6, 20.2, 19.9]) * VS + PHANTOM_AFF[:3, 3]
R_BALL = 8 * VS  # the WM radius of ball(), mm


def unit(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    out: npt.NDArray[np.float64] = v / np.linalg.norm(v, axis=1, keepdims=True)
    return out


def on_sphere(
    x: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Each vertex moved radially onto the sphere of radius R_BALL round
    CENTRE, with the exact radial unit normal."""
    d = unit(x - CENTRE)
    return CENTRE + R_BALL * d, d


def rod() -> npt.NDArray[np.int16]:
    """A WM rod along k (radius 6 voxels, k in [6, 34)) in cortex."""
    ij = np.indices((40, 40)).astype(float)
    r = np.sqrt((ij[0] - 19.7) ** 2 + (ij[1] - 20.1) ** 2)
    a = np.zeros((40, 40, 40), np.int16)
    a[(r < 12)[:, :, None] & (np.arange(40) >= 2)[None, None, :]] = 42
    a[..., 38:] = 0
    a[(r < 6)[:, :, None] & ((np.arange(40) >= 6) & (np.arange(40) < 34))] = 41
    return a


AXIS = np.array([19.7, 20.1]) * VS + PHANTOM_AFF[:2, 3]
R_ROD = 6 * VS


def on_cylinder(
    x: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """The vertices of the rod's side, k in (12, 28) voxels, moved onto the
    cylinder of radius R_ROD with the exact radial normal; the others stay,
    with their normals from the mesh."""
    out, n = x.copy(), np.zeros_like(x)
    k = (x[:, 2] - PHANTOM_AFF[2, 3]) / VS
    side = (k > 12) & (k < 28)
    d = x[side, :2] - AXIS
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    out[side, :2] = AXIS + R_ROD * d
    n[side, :2] = d
    n[~side] = np.nan
    return out, n


SEMI = np.array([10.0, 7.5, 5.0]) * VS  # mm


def ellipsoid() -> npt.NDArray[np.int16]:
    x = (
        np.indices((40, 40, 40)).astype(float) * VS
        + PHANTOM_AFF[:3, 3, None, None, None]
    )
    q = (((x - CENTRE[:, None, None, None]) / SEMI[:, None, None, None]) ** 2).sum(0)
    a = np.zeros((40, 40, 40), np.int16)
    a[q < 1.9] = 42
    a[q < 1] = 41
    return a


def on_ellipsoid(
    x: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    d = x - CENTRE
    y = d / np.sqrt(((d / SEMI) ** 2).sum(1, keepdims=True))
    return CENTRE + y, unit(y / SEMI**2)


def ellipsoid_curvatures(y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """(m, 2): the principal curvatures, largest first, at points y on the
    ellipsoid (relative to its centre), from its Gaussian and mean
    curvature."""
    a2 = SEMI**2
    h = (y**2 / a2**2).sum(1)
    big_k = 1 / (np.prod(a2) * h**2)
    big_h = np.abs((y**2).sum(1) - a2.sum()) / (2 * np.prod(SEMI) ** 2 * h**1.5)
    sq = np.sqrt(np.maximum(big_h**2 - big_k, 0))
    return np.stack([big_h + sq, big_h - sq], 1)


@pytest.fixture(scope="module")
def exact(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, rays.RaySampler]:
    """Phantoms with exact geometry: the marching-tetrahedra mesh of the
    labels with its vertices moved onto the exact surface and its normals
    set to the exact unit normals (the plane: the slab's top face)."""
    out = {}

    def plane(
        x: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        v = (x - PHANTOM_AFF[:3, 3]) / VS
        top = (np.abs(v[:, 2] - 15.5) < 1) & np.all((v[:, :2] > 8) & (v[:, :2] < 32), 1)
        y = x.copy()
        y[top, 2] = 15.5 * VS + PHANTOM_AFF[2, 3]
        n = np.full_like(x, np.nan)
        n[top] = [0.0, 0.0, 1.0]
        return y, n

    def rng_sphere(
        x: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The sphere, each value moved by up to a quarter of a float32 ulp
        before it is rounded: an error of up to 0.75 ulp in the files."""
        y, n = on_sphere(x)
        g = np.random.default_rng(7)
        y = y + g.uniform(-0.25, 0.25, y.shape) * np.spacing(y.astype(np.float32))
        n = n + g.uniform(-0.25, 0.25, n.shape) * np.spacing(n.astype(np.float32))
        return y, n

    cases = {
        "plane": (slabs((6, 16, 41)), plane),
        "sphere": (ball(), on_sphere),
        "noisy sphere": (ball(), rng_sphere),
        "cylinder": (rod(), on_cylinder),
        "ellipsoid": (ellipsoid(), on_ellipsoid),
    }
    for name, (a, fn) in cases.items():
        surf, dseg = write_phantom(tmp_path_factory.mktemp(name.replace(" ", "-")), a)
        old = rays.RaySampler(surf, dseg).n.astype(np.float64)

        def with_mesh_normals(
            x: npt.NDArray[np.float64],
            fn: Any = fn,
            old: npt.NDArray[np.float64] = old,
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            y, n = fn(x)
            keep = np.isnan(n[:, 0])
            n[keep] = old[keep]
            return y, n

        reshape_phantom(surf, with_mesh_normals)
        out[name] = rays.RaySampler(surf, dseg)
    return out


def equivolume(sampler: rays.RaySampler, **kw: Any) -> rays.Rays:
    return sampler.sample((0, 0, 0), 3, method="equivolume", **kw)


def per_column(r: rays.Rays, x: npt.NDArray[Any], c: Any) -> npt.NDArray[Any]:
    return x[r.sample_start[c] : r.sample_start[c + 1]]


def ring_inside(
    sampler: rays.RaySampler, select: npt.NDArray[np.bool_]
) -> npt.NDArray[np.bool_]:
    """The vertices whose 1-ring lies wholly in select."""
    f = np.asarray(sampler.f, np.int64)
    bad = np.zeros(len(select), bool)
    out_face = ~select[f].all(1)
    bad[f[out_face].ravel()] = True
    out: npt.NDArray[np.bool_] = select & ~bad
    return out


def test_a_plane_has_a_constant_cross_section(exact: dict[str, Any]) -> None:
    """On the plane, every vertex of the 1-ring has the same normal: A is
    a0, a1 = a2 = 0, nothing is marked, and V = a0 (s + CAP_IN)."""
    sampler = exact["plane"]
    r = equivolume(sampler)
    v = (np.asarray(sampler.x, np.float64) - PHANTOM_AFF[:3, 3]) / VS
    top = (np.abs(v[:, 2] - 15.5) < 1e-3) & np.all((v[:, :2] > 8) & (v[:, :2] < 32), 1)
    cols = np.flatnonzero(ring_inside(sampler, top)[r.roots.vertex])
    assert len(cols) > 100
    for c in cols:
        a0, a1, a2 = r.area_coef[c]
        assert a0 > 0 and a1 == 0 and a2 == 0
        # s_k = k step in float64, as the sampler computes it
        s = np.arange(-int(r.k_in[c]), int(r.k_out[c]) + 1) * r.step
        assert np.all(per_column(r, r.area, c) == np.float32(a0))
        assert not per_column(r, r.past_focus, c).any()
        want = a0 * (s + rays.CAP_IN)
        assert np.allclose(per_column(r, r.volume, c), want, rtol=1e-6, atol=0)
        assert tuple(r.focus[c]) == (-np.inf, np.inf)


def test_a_sphere_with_radial_normals(exact: dict[str, Any]) -> None:
    """A(s) = A0 (1 + s/R)^2 to float rounding, with a double root at
    s = -R: the mark is set from -R inward, and V, the integral of A from
    -CAP_IN (R < CAP_IN), is 0 there and the exact cubic outward. Past -R,
    A is positive again; the mark stays set."""
    sampler = exact["sphere"]
    r = equivolume(sampler)
    a0 = r.area_coef[:, 0]
    assert np.all(a0 > 0)
    # "to float rounding": within the bound e on the coefficients that the
    # sampler derives from one float32 ulp in each input
    _, e = sampler.area_coef(r.roots, (0, 0, 0), 3, 0.0)
    s = np.concatenate(
        [
            np.arange(-int(a), int(b) + 1) * r.step
            for a, b in zip(r.k_in, r.k_out, strict=True)
        ]
    )
    col = np.repeat(np.arange(len(a0)), np.diff(r.sample_start))
    g = (1 + s / R_BALL) ** 2
    want_a = a0[col] * g
    tol = e[col, 0] * (1 + g) + e[col, 1] * np.abs(s) + e[col, 2] * s**2
    assert np.all(np.abs(r.area - want_a) <= tol + 1e-7 * np.abs(want_a))
    assert np.allclose(r.focus[:, 0], -R_BALL, rtol=1e-3)
    assert np.all(r.focus[:, 1] == np.inf)
    # the step 0.04 does not hit -R = -1.28 exactly: compare away from it
    far = np.abs(s + R_BALL) > 1e-3
    assert np.array_equal(r.past_focus[far], (s <= -R_BALL)[far])
    t = np.maximum(s, -R_BALL) + R_BALL
    want_v = a0[col] * R_BALL / 3 * (t / R_BALL) ** 3
    assert np.allclose(r.volume, want_v, rtol=1e-4, atol=1e-6 * a0.max())
    assert np.all(r.volume[s <= -R_BALL] < 1e-6 * a0.max())
    assert np.all(r.area[s < -R_BALL - 0.1] > 0)


def test_a_sphere_with_float32_noise(exact: dict[str, Any]) -> None:
    """Each position and normal off by up to 0.75 ulp: still within the
    bound of the discriminant (one ulp), so the focus is the double root
    at -R, to 1e-4 R."""
    r = equivolume(exact["noisy sphere"])
    assert np.all(np.isfinite(r.focus[:, 0]))
    assert np.max(np.abs(r.focus[:, 0] + R_BALL)) < 1e-4 * R_BALL


def test_a_cylinder_focuses_at_its_axis(exact: dict[str, Any]) -> None:
    """On a cylinder of radius r with radial normals, A = A0 (1 + s/r): it
    reaches 0 at s = -r, the mark is set there and inward, and V stops
    rising."""
    sampler = exact["cylinder"]
    r = equivolume(sampler)
    x = np.asarray(sampler.x, np.float64)
    k = (x[:, 2] - PHANTOM_AFF[2, 3]) / VS
    rad = np.linalg.norm(x[:, :2] - AXIS, axis=1)
    side = (k > 12) & (k < 28) & (np.abs(rad - R_ROD) < 1e-5)
    cols = np.flatnonzero(ring_inside(sampler, side)[r.roots.vertex])
    assert len(cols) > 100
    a = r.area_coef[cols]
    assert np.allclose(a[:, 1] / a[:, 0], 1 / R_ROD, rtol=1e-4)
    assert np.all(np.abs(a[:, 2] / a[:, 0]) < 1e-4)
    assert np.allclose(r.focus[cols, 0], -R_ROD, rtol=1e-4)
    for c in cols:
        s = per_column(r, r.s, c).astype(np.float64)
        past = per_column(r, r.past_focus, c)
        vol = per_column(r, r.volume, c)
        far = np.abs(s + R_ROD) > 1e-3
        assert np.array_equal(past[far], (s <= -R_ROD)[far])
        assert np.all(per_column(r, r.area, c)[s < -R_ROD - 0.05] < 0)
        assert np.all(vol[s <= -R_ROD] == vol[0])
        assert vol[0] == 0 and np.all(np.diff(vol[s > -R_ROD]) > 0)


def test_an_ellipsoid_has_two_focal_points(exact: dict[str, Any]) -> None:
    """On an ellipsoid with exact normals, A/A0 is near
    (1 + k1 s)(1 + k2 s): two real roots inward. A is negative between them
    and positive again past both; the mark is set from the first root
    inward, and V is constant there. The roots match -1/k1 and -1/k2 to the
    mesh's discretization (a ring of about one voxel against radii of
    curvature of 2.5 voxels and more), so they have a tolerance."""
    sampler = exact["ellipsoid"]
    r = equivolume(sampler)
    y = r.roots.root.astype(np.float64) - CENTRE
    kap = ellipsoid_curvatures(y)
    a0, a1, a2 = r.area_coef.T
    disc = a1**2 - 4 * a0 * a2
    two = (kap[:, 0] > 1.3 * kap[:, 1]) & (1 / kap[:, 1] < rays.CAP_IN - 0.2)
    assert two.sum() > 50
    assert np.all(disc[two] > 0)
    sq = np.sqrt(disc[two])
    near = (-a1[two] + sq) / (2 * a2[two])
    far = (-a1[two] - sq) / (2 * a2[two])
    e_near = np.abs(near * kap[two, 0] + 1)
    e_far = np.abs(far * kap[two, 1] + 1)
    print(
        f"ellipsoid: median |r k + 1| {np.median(e_near):.3f}, {np.median(e_far):.3f}"
    )
    assert np.median(e_near) < 0.05 and np.median(e_far) < 0.05
    assert np.allclose(r.focus[two, 0], near, rtol=1e-12)
    for c in np.flatnonzero(two):
        s = per_column(r, r.s, c).astype(np.float64)
        area = per_column(r, r.area, c)
        vol = per_column(r, r.volume, c)
        past = per_column(r, r.past_focus, c)
        lo, hi = sorted(np.roots([a2[c], a1[c], a0[c]]).real)
        between = (s > lo + 0.01) & (s < hi - 0.01)
        beyond = s < lo - 0.01
        assert np.all(area[between] < 0)
        assert np.all(area[beyond] > 0)
        assert np.array_equal(past, s <= r.focus[c, 0])
        assert np.all(vol[past] == vol[0]) and vol[0] == 0


def test_the_marching_ball_against_the_sphere(
    phantom: dict[str, tuple[rays.RaySampler, Any]],
) -> None:
    """The marching-tetrahedra ball (WM radius R = 8 voxels): A(s)/A(0)
    against (1 + s/R)^2, from s = -R/2 to 2 mm. Its normals are smoothed,
    not radial (to 0.95 in cosine), and its vertices lie on the voxels'
    staircase within half a voxel, so each column's curvature differs from
    1/R: the median and the spread have a tolerance, not each column."""
    r = equivolume(phantom["ball"][0])
    s = r.s.astype(np.float64)
    col = np.repeat(np.arange(len(r.area_coef)), np.diff(r.sample_start))
    sel = (s >= -R_BALL / 2) & (s <= 2.0)
    ratio = r.area[sel] / r.area_coef[col[sel], 0]
    rel = ratio / (1 + s[sel] / R_BALL) ** 2 - 1
    med = float(np.median(rel))
    q10, q90 = np.quantile(rel, [0.1, 0.9])
    print(
        f"A/A0 against (1 + s/R)^2: median {med:+.4f}, 10-90 % [{q10:+.4f}, {q90:+.4f}]"
    )
    # measured: median +0.0003, 10-90 % [-0.14, +0.31]. The spread is the
    # columns' own curvature on this mesh (a 1-ring of about one voxel with
    # smoothed normals), not rounding
    assert abs(med) < 0.02
    assert q90 - q10 < 0.6


def test_the_volume_does_not_depend_on_the_extent_or_the_step(
    exact: dict[str, Any],
) -> None:
    """V at each s is the same for every extent (bit for bit) and every
    step (to float32 rounding: s_k = k step differs in its last bit)."""
    sampler = exact["ellipsoid"]
    full = equivolume(sampler)
    n = len(full.roots.vertex)
    e_out = np.linspace(0.0, 8.0, n)
    short = equivolume(sampler, e_in=0.5, e_out=e_out)
    assert np.array_equal(short.area_coef, full.area_coef)
    for c in range(0, n, 5):
        a, b = short.sample_start[c], short.sample_start[c + 1]
        off = full.sample_start[c] + 50 - 12
        for name in ("area", "volume", "past_focus"):
            got = getattr(short, name)[a:b]
            assert np.array_equal(got, getattr(full, name)[off : off + b - a])
    coarse = equivolume(sampler, step=0.2)
    for c in range(0, n, 5):
        fine = per_column(full, full.volume, c)[::5]
        got = per_column(coarse, coarse.volume, c)
        assert np.allclose(got, fine, rtol=1e-6, atol=1e-9)


def test_normal_gives_the_same_samples_and_no_cross_section(
    phantom: dict[str, tuple[rays.RaySampler, Any]],
) -> None:
    sampler = phantom["ball"][0]
    nr = sampler.sample((0, 0, 0), 3)
    ev = equivolume(sampler)
    for f in dataclasses.fields(rays.Rays):
        if f.name in ("method", "area_coef", "focus", "area", "volume", "past_focus"):
            continue
        a, b = getattr(nr, f.name), getattr(ev, f.name)
        if f.name == "roots":
            for g in dataclasses.fields(rays.Roots):
                assert getattr(a, g.name).tobytes() == getattr(b, g.name).tobytes()
        else:
            assert np.asarray(a).tobytes() == np.asarray(b).tobytes(), f.name
    assert nr.area_coef.shape == (0, 3) and nr.focus.shape == (0, 2)
    for name in ("area", "volume", "past_focus"):
        assert getattr(nr, name).shape == (0,)
    n, big_n = len(ev.roots.vertex), len(ev.s)
    assert ev.area_coef.shape == (n, 3) and ev.focus.shape == (n, 2)
    assert ev.area.shape == ev.volume.shape == ev.past_focus.shape == (big_n,)


def test_the_faces_of_a_call_hold_every_1_ring(
    phantom: dict[str, tuple[rays.RaySampler, Any]],
) -> None:
    """For superblocks and halos: every face that uses a root of the call is
    among the faces the call reads."""
    sampler = phantom["ball"][0]
    f = np.asarray(sampler.f, np.int64)
    ijk = [sampler.block_ijk[i] for i in (0, len(sampler.block_ijk) // 2)]
    b = [(int(z[0]), int(z[1]), int(z[2])) for z in ijk]
    calls: list[tuple[tuple[int, int, int], int, float]] = [
        ((1, 1, 1), 1, 0.0),
        ((0, 1, 1), 1, 2.0),
        (b[0], 0, 0.5),
        (b[1], 0, 0.0),
    ]
    for sb, lev, h in calls:
        roots = sampler.roots(sb, lev, h)
        read = {tuple(t) for t in sampler.faces(sb, lev, h).tolist()}
        mine = np.isin(f, roots.vertex).any(1)
        assert mine.sum() > 0
        assert {tuple(t) for t in f[mine].tolist()} <= read


def test_an_open_1_ring_is_refused(
    phantom: dict[str, tuple[rays.RaySampler, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A face missing from a root's 1-ring (a mesh boundary, which
    white-surf/1 does not have, or a face not read) raises."""
    sampler = phantom["ball"][0]
    every = sampler.faces((0, 0, 0), 3, 0.0)
    monkeypatch.setattr(sampler, "faces", lambda *a: every[1:])
    with pytest.raises(RuntimeError, match="closed fan"):
        equivolume(sampler)


def fan(
    normals: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """The coefficients and their error bounds of the root (vertex 0) of a
    closed fan round it: vertices x (k + 1, 3), normals (k + 1, 3)."""
    k = len(x) - 1
    f = np.array([[0, 1 + j, 1 + (j + 1) % k] for j in range(k)], np.int64)
    flat = f.ravel()
    order = np.argsort(flat, kind="stable")
    coef, err, bad = rays._area_coef(
        x,
        normals,
        f,
        (order // 3).astype(np.int64),
        np.array([0], np.int64),
        np.array([k], np.int64),
        np.array([0], np.int64),
        rays.U_IN,
    )
    assert bad == -1
    return coef[0], err[0]


def test_a_plane_with_float32_noise_marks_nothing() -> None:
    """A flat hexagonal fan at a position like the hemisphere's (about
    50 mm from the origin), its normals (0, 0, 1) moved by a few float32
    ulps, over many seeds: no focal point within the caps."""
    g = np.random.default_rng(3)
    ang = np.arange(6) * np.pi / 3
    ring = np.c_[np.cos(ang), np.sin(ang), np.zeros(6)] * 0.16
    for _ in range(2000):
        base = g.uniform(-60, 60, 3)
        x = (base + np.r_[[[0.0, 0.0, 0.0]], ring]).astype(np.float32)
        n = np.tile([0.0, 0.0, 1.0], (7, 1)) + g.integers(-3, 4, (7, 3)) * 2.0**-24
        n = (n / np.linalg.norm(n, axis=1, keepdims=True)).astype(np.float32)
        (a0, a1, a2), err = fan(n.astype(np.float64), x.astype(np.float64))
        bound = rays.disc_bound(a0, a1, a2, err)
        lo, hi = rays.focal_bounds(a0, a1, a2, bound)
        assert lo < -10 * rays.CAP_IN and hi > 10 * rays.CAP_OUT, (a0, a1, a2, lo, hi)


@pytest.mark.parametrize(
    ("coef", "want"),
    [
        ((1.0, 0.0, 0.0), (-np.inf, np.inf)),  # a plane
        ((1.0, 2.0, 1.0), (-1.0, np.inf)),  # a sphere, R = 1: a double root
        ((1.0, -2.0, 1.0), (-np.inf, 1.0)),  # a cup outward
        ((1.0, 1.0, 0.0), (-1.0, np.inf)),  # a cylinder
        ((1.0, 3.0, 2.0), (-0.5, np.inf)),  # roots -1/2 and -1
        ((1.0, 0.0, -1.0), (-1.0, 1.0)),  # a saddle: roots on both sides
        ((1.0, 2.0, 2.0), (-np.inf, np.inf)),  # disc = -4, far past the bound
        ((0.0, 1.0, 1.0), (0.0, 0.0)),  # A(0) <= 0
        ((-1.0, 1.0, 1.0), (0.0, 0.0)),
    ],
)
def test_the_focal_bounds(
    coef: tuple[float, float, float], want: tuple[float, float]
) -> None:
    assert rays.focal_bounds(*coef, 1e-12) == pytest.approx(want)


def test_a_double_root_within_the_bound() -> None:
    """disc slightly below 0: a double root within the bound, none
    beyond it."""
    a0, a1, a2 = 1.0, 2.0, 1.0 + 1e-9  # disc = -4e-9
    assert rays.focal_bounds(a0, a1, a2, 1e-8)[0] == pytest.approx(-1.0, rel=1e-8)
    assert rays.focal_bounds(a0, a1, a2, 1e-10) == (-np.inf, np.inf)
