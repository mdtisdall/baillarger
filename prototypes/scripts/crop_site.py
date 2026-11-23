"""A cropped site for the Σ stop stage: the parts of global_tau_prep.py
(prototype branch claude/tau-shell-cc08e1) that the stage uses, unchanged.

A site is a crop set written by baillarger.crop (python -m baillarger.crop
...): echoes (3, X, Y, Z), labels, affine (voxel -> mm) and the crop's white
surface. load_site derives:

- the white surface in voxel coordinates, its unit normals (out of the WM),
  its vertex areas and its mean curvature H̄ (smoothed over CURV_MM along the
  surface, baillarger.frame.mesh_curvatures);
- the signed distance d (mm, > 0 outside the WM) of the voxels near the white
  surface, and the inside mask (d < 0 near the surface, label 41 elsewhere);
- the laminar normal n̂ and its coherence from the structure tensor of echoes 2
  and 3 (as in baillarger.column_traces: log, trend removed, cortex label).

Also the parsing of "key=value" run settings (parse_bool, set_setting).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from baillarger import streamlines as S
from baillarger.frame import mesh_curvatures

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

WM, CORTEX, OUTSIDE = 41, 42, 0
CURV_MM = 5.0
TREND = 4.0  # column_traces.TREND (voxels)
NEAR_VOX = 3  # signed distance computed within this many voxels of the WM edge


@dataclass
class Site:
    echoes: npt.NDArray[np.float32]
    labels: npt.NDArray[np.int16]
    affine: F64
    vox: F64  # voxel size (mm) per axis
    v: F64  # white vertices, voxel coordinates
    f: I64
    normal: F64  # unit, out of the WM, voxel-coordinate frame (mm-scaled)
    varea: F64  # vertex area (mm²)
    H: F64  # mean curvature (k1 + k2) / 2 per mm, smoothed
    dist: F64  # signed distance (mm), nan far from the surface
    inside: npt.NDArray[np.bool_]
    nhat: npt.NDArray[np.float32]  # (X, Y, Z, 3)
    coh: npt.NDArray[np.float32]


def parse_bool(key: str, v: str) -> bool:
    """A boolean command-line value: true/false/1/0 in any case. Any other
    value stops the run with an error (bool("False") would be True)."""
    b = {"true": True, "1": True, "false": False, "0": False}.get(v.lower())
    if b is None:
        raise SystemExit(f"{key}: expected true/false/1/0, got {v!r}")
    return b


def set_setting(s: object, kv: str) -> None:
    """Set one field of a settings dataclass from "key=value", with the type
    of its default; booleans by parse_bool. An unknown key or a "kv" without
    "=" stops the run with an error."""
    k, sep, v = kv.partition("=")
    if not sep:
        raise SystemExit(f"expected key=value, got {kv!r}")
    if not hasattr(s, k):
        raise SystemExit(f"unknown setting {k!r}")
    t = type(getattr(s, k))
    setattr(s, k, parse_bool(k, v) if t is bool else t(v))


def vertex_normals_areas(v_mm: F64, f: I64) -> tuple[F64, F64]:
    """Area-weighted unit vertex normals (face orientation) and vertex areas."""
    cr = np.cross(v_mm[f[:, 1]] - v_mm[f[:, 0]], v_mm[f[:, 2]] - v_mm[f[:, 0]])
    n = np.zeros_like(v_mm)
    a = np.zeros(len(v_mm))
    fa = 0.5 * np.linalg.norm(cr, axis=1)
    for k in range(3):
        np.add.at(n, f[:, k], cr)
        np.add.at(a, f[:, k], fa / 3)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return n, a


def mm_to_vox(affine: F64, p: F64) -> F64:
    out: F64 = (np.linalg.inv(affine) @ np.c_[p, np.ones(len(p))].T).T[:, :3]
    return out


def sample_labels(labels: npt.NDArray[np.integer], pv: F64) -> npt.NDArray[np.int64]:
    """Labels at the nearest voxel of points pv (voxel coordinates); -1 outside."""
    i = np.round(pv).astype(np.int64)
    ok = np.all((i >= 0) & (i < np.array(labels.shape)), axis=1)
    out = np.full(len(pv), -1, np.int64)
    out[ok] = labels[i[ok, 0], i[ok, 1], i[ok, 2]]
    return out


def signed_distance(
    labels: npt.NDArray[np.integer], v: F64, n: F64, vox: F64
) -> tuple[F64, npt.NDArray[np.bool_]]:
    """Signed distance (mm, > 0 outside the WM) by the nearest vertex and its
    normal, for voxels within NEAR_VOX of the WM label's edge; inside mask."""
    wm = labels == WM
    edge = wm ^ ndi.binary_erosion(wm)
    near = ndi.binary_dilation(edge, iterations=NEAR_VOX + 1)
    idx = np.argwhere(near)
    tree = cKDTree(v * vox)
    _, j = tree.query(idx * vox)
    d = np.einsum("ij,ij->i", idx * vox - v[j] * vox, n[j])
    dist = np.full(labels.shape, np.nan)
    dist[tuple(idx.T)] = d
    inside = wm.copy()
    inside[tuple(idx.T)] = d < 0
    return dist, inside


def laminar_normal(
    echoes: npt.NDArray[np.float32], labels: npt.NDArray[np.integer]
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """n̂ and coherence from echoes 2 and 3 over the cortex label
    (column_traces.tissue + base_tensor + data_normal, with S.eigen)."""
    ctx = (labels == CORTEX).astype(float)
    tensor = None
    for i in (1, 2):
        L = np.log(np.maximum(echoes[i].astype(float), 1.0))
        det = (L - S.masked_smooth(L, ctx, TREND)) * ctx
        t = S.structure_tensor(det, ctx)
        tensor = t if tensor is None else tensor + t
    assert tensor is not None
    nh, coh = S.eigen(tensor)
    coh = np.where(ctx > 0, coh, 0).astype(np.float32)
    return nh, coh


def crop_sidecar(path: Path) -> Path:
    """The crop/1 sidecar (desc-crop_box.json) of a crop set: path itself,
    or the one sidecar under the directory path."""
    path = Path(path)
    if path.is_dir():
        found = sorted(path.rglob("*_desc-crop_box.json"))
        if len(found) != 1:
            raise ValueError(f"{path}: {len(found)} crop sidecars, need one")
        return found[0]
    return path


def site_arrays(path: Path) -> dict[str, npt.NDArray]:
    """A site from a crop set written by baillarger.crop (python -m
    baillarger.crop ...; path: its directory or its crop/1 sidecar): echoes
    (3, X, Y, Z) float32, labels (the dseg, int16; the crop's label edits
    are not applied), affine (the crop's), white_v (mm) and white_f (the
    crop's white surface), and centre_vox: the middle of the box (the site
    centre of figures and movies)."""
    from baillarger import crop as C

    c = C.load_crop(crop_sidecar(path))
    if c.white is None or len(c.echoes) != 3:
        raise ValueError(f"{path}: a site needs the white surface and three echoes")
    return {
        "echoes": c.echo_stack(),
        "labels": np.asarray(c.dseg).astype(np.int16),
        "affine": np.asarray(c.affine, float),
        "white_v": np.asarray(c.white.v_mm, float),
        "white_f": np.asarray(c.white.f, np.int64),
        "centre_vox": 0.5 * (np.asarray(c.box.shape, float) - 1.0),
    }


def load_site(path: Path) -> Site:
    """The Site of a crop set (site_arrays)."""
    z = site_arrays(path)
    A = z["affine"].astype(float)
    vox = np.abs(np.diag(A)[:3]).astype(float)
    labels = z["labels"].astype(np.int16)
    v = mm_to_vox(A, z["white_v"].astype(float))
    f = z["white_f"].astype(np.int64)
    n, area = vertex_normals_areas(v * vox, f)
    # orient out of the WM: the label 0.5 voxel along the normal
    out = sample_labels(labels, v + 0.5 * n / vox) != WM
    inn = sample_labels(labels, v - 0.5 * n / vox) == WM
    if np.mean(out & inn) < np.mean(~out & ~inn):
        n = -n
        f = f[:, ::-1].copy()
    k1, k2 = mesh_curvatures(v * vox, f, n, CURV_MM)
    dist, inside = signed_distance(labels, v, n, vox)
    nh, coh = laminar_normal(z["echoes"], labels)
    return Site(
        z["echoes"],
        labels,
        A,
        vox,
        v,
        f,
        n,
        area,
        0.5 * (k1 + k2),
        dist,
        inside,
        nh,
        coh,
    )
