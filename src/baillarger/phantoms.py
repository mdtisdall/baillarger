"""Synthetic label volumes with a known answer for the seam between touching
cortical banks.

Labels as in the rim labels: 0 unknown, 1 outside (CSF / background), 2 white
matter (WM), 3 cortex. Voxel size 0.16 mm. P1-P5 return (rim, flash,
expected): rim is int16, flash is float32 (cortex 1.0, WM 0.8, outside 0.1,
Gaussian noise SD 0.05, fixed seed per phantom), expected is a dict. Axes: x
across the U, y along its long axis, z depth (up = increasing z). Every
phantom is below 100^3 voxels.

Keys of every expected dict:

- seam_required: True if the growth must cut the cortex somewhere.
- seam_target_mask: where the cut must lie (all False when no seam is
  required).
- seam_forbidden_mask: cortex voxels where a seam is wrong.

P1 kissing banks. The WM is one U: two plates (6 voxels thick) joined by a
   bottom slab. Cortex (14 voxels) covers the inner face of each plate and the
   top face of the bottom slab. The CSF gap (16 voxels wide) is open at the top
   and at both ends along y. In the patch y 14..25, z 36..45 the gap is filled
   with cortex, so the banks touch without CSF. The plane x = 32 splits the
   banks (A below, B from 32). Seam required: target = the 2 voxel sheet
   x = 31, 32 in the patch; forbidden = cortex more than 3 voxels from the
   target. Also in expected: bank_a_mask, bank_b_mask (they partition the
   cortex) and patch_mask (the filled gap voxels). The banks are joined
   through the patch and also through the cortex on the U bottom (z < 22),
   which is not part of the seam question.
P2 sulcus end. WM slab, cortex slab 16 voxels thick, and a cylindrical CSF
   pocket (radius 5) from the outer face down to 5 voxels above the WM. No
   seam: forbidden = all cortex.
P3 stria band. WM slab, cortex slab 15 voxels thick, and in the flash a dark
   band (0.5), 2 voxels thick, at 60 % of the depth. The labels stay 3. No
   seam: forbidden = all cortex.
P4 kissing banks with a dark line. As P1 but bank A is 12 voxels and bank B 18
   (ratio 1.5), and in the patch a 1 voxel dark sheet (flash 0.3, labels stay
   3) at x = 25 marks the contact, which is not midway. Seam required: target =
   the dark sheet; forbidden = cortex more than 3 voxels from it. Bank A is
   x <= 25, bank B x >= 26.
P5 WM handle. The WM is a solid torus (tube radius 5, ring radius 24), cortex
   a shell of 13 voxels around it, outside beyond. No seam: forbidden = all
   cortex. expected also has wm_euler (0) and genus (1).

P6-P9 are variants of these, each with its own return value:

P6 stria in a fused bank (p6_fused_stria): P4 without its dark line, with a
   stria-like band in the thick bank.
P7 visible slit (p7_visible_slit): a dark sheet in the cortex with no CSF
   label.
P8 natural contact (p8_natural_contact): banks of different thickness that
   touch at their natural thicknesses, with no dark line.
P9 dark line with holes (p9_dark_line_with_holes): P4 with gaps in the dark
   line.
"""

from typing import NotRequired, TypedDict

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi

type Bool3 = npt.NDArray[np.bool_]
type Rim = npt.NDArray[np.int16]
type Flash = npt.NDArray[np.float32]


class Expected(TypedDict):
    """The known answer of a phantom."""

    seam_required: bool
    seam_target_mask: Bool3
    seam_forbidden_mask: Bool3
    bank_a_mask: NotRequired[Bool3]
    bank_b_mask: NotRequired[Bool3]
    patch_mask: NotRequired[Bool3]
    wm_euler: NotRequired[int]
    genus: NotRequired[int]


type Phantom = tuple[Rim, Flash, Expected]

VOXEL_MM = 0.16
_LEVEL = {0: 0.1, 1: 0.1, 2: 0.8, 3: 1.0}
_NOISE_SD = 0.05
_GAP = 16  # CSF gap between the cortex of the two plates
_WM_T = 6  # thickness of the WM plates
_MARGIN = 4  # outside at the x faces
# WM starts, plates' cortex starts, plates end, bottom cortex ends
_Z0, _Z_BOT, _Z_TOP, _Z_CBOT = 2, 8, 52, 22
_PATCH_Y, _PATCH_Z = (14, 26), (36, 46)


def _flash(
    rim: Rim, seed: int, dark: list[tuple[Bool3, float]] | tuple[()] = ()
) -> Flash:
    """Mean intensity per label (dark = [(mask, value)] overrides it) plus
    Gaussian noise."""
    base = np.zeros(rim.shape)
    for lab, v in _LEVEL.items():
        base[rim == lab] = v
    for mask, v in dark:
        base[mask] = v
    rng = np.random.default_rng(seed)
    return (base + _NOISE_SD * rng.standard_normal(rim.shape)).astype(np.float32)


def _far_from(target: Bool3, cortex: Bool3, dist: float = 3.0) -> Bool3:
    return cortex & (ndi.distance_transform_edt(~target) > dist)


def _u_banks(
    ta: int,
    tb: int,
    split: int,
    seed: int,
    dark_x: int | None = None,
    gap: int = _GAP,
) -> Phantom:
    """P1 / P4 / P8 geometry. ta, tb: cortex thickness of the banks; split:
    first x of bank B; dark_x: x of the 1 voxel dark sheet (P4), else None;
    gap: width of the CSF gap between the banks."""
    gx0 = _MARGIN + _WM_T + ta
    gx1 = gx0 + gap
    nx = gx1 + tb + _WM_T + _MARGIN
    shape = (nx, 40, 58)
    rim = np.ones(shape, np.int16)
    rim[_MARGIN : nx - _MARGIN, :, _Z0:_Z_BOT] = 2  # bottom slab
    rim[_MARGIN : _MARGIN + _WM_T, :, _Z0:_Z_TOP] = 2  # plate A
    rim[nx - _MARGIN - _WM_T : nx - _MARGIN, :, _Z0:_Z_TOP] = 2  # plate B
    rim[_MARGIN + _WM_T : gx0, :, _Z_BOT:_Z_TOP] = 3  # cortex of A
    rim[gx1 : gx1 + tb, :, _Z_BOT:_Z_TOP] = 3  # cortex of B
    rim[gx0:gx1, :, _Z_BOT:_Z_CBOT] = 3  # cortex on the U bottom
    patch = np.zeros(shape, bool)
    patch[gx0:gx1, _PATCH_Y[0] : _PATCH_Y[1], _PATCH_Z[0] : _PATCH_Z[1]] = True
    rim[patch] = 3
    cortex = rim == 3
    x = np.arange(nx)[:, None, None]
    bank_b = cortex & (x >= split)
    bank_a = cortex & ~bank_b
    dark: list[tuple[Bool3, float]]
    if dark_x is None:
        target = patch & ((x == split - 1) | (x == split))
        dark = []
    else:
        target = patch & (x == dark_x)
        dark = [(target, 0.3)]
    expected = Expected(
        seam_required=True,
        seam_target_mask=target,
        seam_forbidden_mask=_far_from(target, cortex),
        bank_a_mask=bank_a,
        bank_b_mask=bank_b,
        patch_mask=patch,
    )
    return rim, _flash(rim, seed, dark), expected


def p1_kissing_banks() -> Phantom:
    """P1: kissing banks of equal thickness, no dark line."""
    return _u_banks(14, 14, split=32, seed=1)


def p4_kissing_dark_line() -> Phantom:
    """P4: kissing banks of 12 and 18 voxels, a dark line at the contact."""
    return _u_banks(12, 18, split=26, seed=4, dark_x=25)


def _slab(
    thick: int,
    seed: int,
    dark_z: tuple[int, int] | None = None,
    pocket: bool = False,
) -> Phantom:
    shape = (48, 48, 38)
    rim = np.ones(shape, np.int16)
    z0, z1 = 10, 10 + thick
    rim[:, :, 2:z0] = 2
    rim[:, :, z0:z1] = 3
    if pocket:
        X, Y = np.meshgrid(np.arange(48), np.arange(48), indexing="ij")
        cyl = (X - 24) ** 2 + (Y - 24) ** 2 <= 5**2
        z = np.arange(38)[None, None, :]
        rim[cyl[:, :, None] & (z >= z1 - 11) & (z < z1)] = 1
    dark: list[tuple[Bool3, float]] = []
    if dark_z is not None:
        band = np.zeros(shape, bool)
        band[:, :, dark_z[0] : dark_z[1]] = True
        dark = [(band & (rim == 3), 0.5)]
    cortex = rim == 3
    expected = Expected(
        seam_required=False,
        seam_target_mask=np.zeros(shape, bool),
        seam_forbidden_mask=cortex,
    )
    return rim, _flash(rim, seed, dark), expected


def p2_sulcus_end() -> Phantom:
    """P2: a CSF pocket that ends inside the cortex."""
    return _slab(16, seed=2, pocket=True)


def p3_stria_band() -> Phantom:
    """P3: a dark band at 60 % of the cortical depth."""
    # cortex z 10..24 (15 voxels): 60 % depth is z = 19, band z 18, 19
    return _slab(15, seed=3, dark_z=(18, 20))


def p5_wm_handle() -> Phantom:
    """P5: a WM torus with a cortex shell."""
    shape = (90, 90, 44)
    X, Y, Z = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    d = np.hypot(np.hypot(X - 45, Y - 45) - 24, Z - 22)
    rim = np.ones(shape, np.int16)
    rim[d <= 5 + 13] = 3
    rim[d <= 5] = 2
    expected = Expected(
        seam_required=False,
        seam_target_mask=np.zeros(shape, bool),
        seam_forbidden_mask=rim == 3,
        wm_euler=0,
        genus=1,
    )
    return rim, _flash(rim, 5), expected


def p6_fused_stria() -> tuple[Rim, Flash, Expected, Bool3, float, tuple[Bool3, Bool3]]:
    """P6: P4 without its dark line, and with a stria-like band (3 voxels,
    0.75) at 60 % depth of the thick bank, across the touching patch.

    The seam must stay near the midpoint between the WM plates and must not
    move onto the band. The band of P3 (2 voxels at 0.5) is much more
    line-like than the stria of the V1V2 crop; this one is closer to it (its
    sheetness is below 0.12 for 99 % of its voxels, as in V1V2).

    Returns (rim, flash, expected of P4, band, mid, banks): mid is the x of
    the midpoint between the WM plates, banks the cortex on each side of it.
    """
    rim, flash, e = p4_kissing_dark_line()
    rng = np.random.default_rng(6)
    flash = flash.copy()
    line = e["seam_target_mask"]
    flash[line] = 1.0 + 0.05 * rng.standard_normal(line.sum())
    ta, tb = 12, 18  # P4's bank thicknesses
    a_face = _MARGIN + _WM_T  # first cortex x of bank A
    b_face = _MARGIN + _WM_T + ta + _GAP + tb  # first WM x of plate B
    xb = int(round(b_face - 0.6 * tb))
    band = np.zeros(rim.shape, bool)
    band[xb - 1 : xb + 2] = True
    band &= (rim == 3) & e["bank_b_mask"]
    flash[band] = 0.75 + 0.05 * rng.standard_normal(band.sum())
    mid = (a_face + b_face - 1) / 2
    x = np.arange(rim.shape[0])[:, None, None]
    # the banks as split by the expected seam
    banks = ((rim == 3) & (x < mid), (rim == 3) & (x > mid))
    return rim, flash, e, band, mid, banks


def p7_visible_slit() -> tuple[Rim, Flash, Bool3]:
    """P7: P2's slab without the pocket, and a dark sheet (1 voxel, 0.3) from
    the outer surface 10 voxels into the cortex, labelled as cortex: a
    collapsed dead-end sulcus that only the image shows.

    Returns (rim, flash, slit).
    """
    rim, flash, e = p2_sulcus_end()
    z = np.arange(rim.shape[2])[None, None, :]
    rim = np.where(rim == 1, np.where(z >= 26, 1, rim), rim)
    rim[(rim == 1) & (z < 26) & (z > 2)] = 3
    rng = np.random.default_rng(7)
    flash = flash.copy()
    flash[rim == 3] = 1.0 + 0.05 * rng.standard_normal((rim == 3).sum())
    slit = np.zeros(rim.shape, bool)
    slit[24, 10:38, 16:26] = True
    slit &= rim == 3
    flash[slit] = 0.3 + 0.05 * rng.standard_normal(slit.sum())
    return rim, flash, slit


def p8_natural_contact(
    ta: int = 12, tb: int = 18, gap: int = 2
) -> tuple[Rim, Flash, Expected, float, float]:
    """P8: banks of different thickness (12 and 18 voxels) that touch at their
    natural thicknesses.

    P1's geometry with a gap of only 2 voxels, filled in the touching patch.
    No dark line. The contact is at the gap, which is (ta + tb) / 2 - ta = 3
    voxels from the midpoint between the WM plates.

    Returns (rim, flash, expected, contact, mid): the x of the contact and of
    the midpoint between the WM plates.
    """
    gx0 = _MARGIN + _WM_T + ta
    rim, flash, e = _u_banks(ta, tb, split=gx0 + 1, seed=8, gap=gap)
    a_face = _MARGIN + _WM_T
    b_face = gx0 + gap + tb
    return rim, flash, e, gx0 + (gap - 1) / 2, (a_face + b_face - 1) / 2


def p9_dark_line_with_holes(
    frac: float = 0.25, seed: int = 9
) -> tuple[Rim, Flash, Expected, Bool3]:
    """P9: P4 with holes in the dark line.

    Blocks of 2 x 2 voxels of the line (each block with probability frac;
    about 40 % of the line voxels with the defaults) go back to cortex
    intensity, so the fluid's sheet has gaps for the banks to touch through.

    Returns (rim, flash, expected of P4, holes).
    """
    rim, flash, e = p4_kissing_dark_line()
    rng = np.random.default_rng(seed)
    flash = flash.copy()
    line = e["seam_target_mask"]
    holes = np.zeros(rim.shape, bool)
    ys, zs = np.nonzero(line.any(0))
    for y, z in zip(ys[::2], zs[::2], strict=True):
        if rng.random() < frac:
            holes[:, y : y + 2, z : z + 2] = True
    holes &= line
    flash[holes] = 1.0 + 0.05 * rng.standard_normal(holes.sum())
    return rim, flash, e, holes
