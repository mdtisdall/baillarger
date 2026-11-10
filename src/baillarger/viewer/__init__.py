"""A viewer of a white surface and the published outputs on it, in the browser: the
volume (our brick pyramid of the whole echo, with label overlays and surface
outlines), the white surface true or inflated (its overview), and the columns
from the published samplers; any number of windows, kept in step by the server.

- bricks: the brick pyramid of a volume, for the volume pane.
- volumes, outlines: the named volumes (echoes, labels, WM mask, fixes) and the
  surface outlines on the brick pane.
- columns: the columns near a vertex, from the ray sampler and the column-volume
  sampler (the columns pane).
- jobs: the job pattern (one long computation at a time, in a thread).
- tck: the TCK track format (the columns pane's download).
- filters: filters of the profiles along columns (smoothing, flattening, stacking).
- flatmap: topology of the white mesh, handle cuts, least-squares conformal flat map.
- inflation: the inflated surface, sulcal depth, tangent transfer to the true mesh.
- surface: the white surface for viewing (Surface): coordinates, graph, flat map,
  inflation, picking.
- overview: the overview of a white surface (baillarger.white_overview), and a
  Surface on it.
- white_view: a white surface (baillarger.white_surf): its overview and the WM
  selection.
- server: the local web server, its shared state and the page (static/index.html).

Run: python -m baillarger.viewer --white <..._white.surf.gii> --cache DIR
     [--echo-dir DIR] [--dseg ...]
"""
