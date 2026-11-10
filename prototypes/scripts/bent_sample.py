"""Trilinear sampling of a 3D volume in torch, with gradients to the points.

Points are in voxel coordinates: voxel centres are at integers, and the first
coordinate indexes the first axis of the volume (as in Box.to_box).

Method: an explicit gather of the eight corner voxels, with weights from the
fractional part. It is used instead of grid_sample because it needs no
axis-order flip, no normalised grid, and only plain indexing and arithmetic
(no 3D grid_sample backward on the device).

Outside the volume: the coordinates are clamped to [0, n - 1] on each axis.
The value is that of the nearest edge voxel (scipy mode="nearest", as in
streamlines.sample). The gradient with respect to a clamped coordinate is
zero. Use `inside` to find the points that are not clamped.
"""

import torch


def inside(shape: tuple[int, ...], points: torch.Tensor) -> torch.Tensor:
    """Bool mask (...): True where points (..., 3) lie in [0, n - 1] on every
    axis of shape (the range of voxel centres)."""
    hi = torch.tensor(
        [s - 1 for s in shape[-3:]], dtype=points.dtype, device=points.device
    )
    return ((points >= 0) & (points <= hi)).all(dim=-1)


def trilinear(volume: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Sample volume at points, trilinearly.

    volume: (X, Y, Z) or (C, X, Y, Z). points: (..., 3), voxel coordinates.
    Returns (...) or (C, ...), in the dtype of volume. Outside the volume,
    the value of the nearest edge voxel (see the module docstring). Gradients
    flow to points and to volume.
    """
    single = volume.ndim == 3
    vol = volume[None] if single else volume
    c = vol.shape[0]
    shape = vol.shape[1:]
    lead = points.shape[:-1]
    p = points.reshape(-1, 3).to(vol.dtype)
    flat = vol.reshape(c, -1)
    lo: list[torch.Tensor] = []
    hi: list[torch.Tensor] = []
    frac: list[torch.Tensor] = []
    for k in range(3):
        n = shape[k]
        pk = p[:, k].clamp(0.0, float(n - 1))
        i0 = pk.detach().floor().clamp(0, max(n - 2, 0))
        lo.append(i0.long())
        hi.append((i0 + 1).clamp(max=n - 1).long())
        frac.append(pk - i0)
    sy, sz = shape[1] * shape[2], shape[2]

    def corner(a: int, b: int, d: int) -> torch.Tensor:
        ix = (hi if a else lo)[0] * sy
        iy = (hi if b else lo)[1] * sz
        iz = (hi if d else lo)[2]
        return flat[:, ix + iy + iz]

    fx, fy, fz = frac
    # Interpolate along z, then y, then x. At f = 0 the result is exact.
    c00 = torch.lerp(corner(0, 0, 0), corner(0, 0, 1), fz)
    c01 = torch.lerp(corner(0, 1, 0), corner(0, 1, 1), fz)
    c10 = torch.lerp(corner(1, 0, 0), corner(1, 0, 1), fz)
    c11 = torch.lerp(corner(1, 1, 0), corner(1, 1, 1), fz)
    out = torch.lerp(
        torch.lerp(c00, c01, fy),
        torch.lerp(c10, c11, fy),
        fx,
    )
    out = out.reshape(c, *lead)
    return out[0] if single else out
