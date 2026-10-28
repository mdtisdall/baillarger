"""The streamlines at a point of the white surface: now baillarger.viewer.streamlines. gm_surface
keeps the prototype editor's signature (vertices in mm)."""

from baillarger.viewer.streamlines import at_depth, crop_mesh, geodesic, label_surface, select  # noqa: F401


def gm_surface(mask, vox, lo, hi, sigma=1.0):
    """label_surface with the vertices in mm (voxel index times vox)."""
    v, f = label_surface(mask, lo, hi, sigma)
    return v * vox, f
