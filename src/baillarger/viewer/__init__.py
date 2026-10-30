"""A viewer of a box's column data in the browser: echo 2 in orthogonal views and 3D
with the white and GM surfaces and the streamlines (NiiVue), the white surface true or
inflated, a flat overview, and sections along the traces; any number of windows, kept
in step by the server.

- flatmap: topology of the white mesh, handle cuts, least-squares conformal flat map.
- inflation: the inflated surface, sulcal depth, tangent transfer to the true mesh.
- sections: plane-and-mesh sections through a trace.
- streamlines: the traces from and to a point of the white surface, the mesh and a
  label's boundary in a box around it, and the TCK track format.
- surface: the white surface for viewing (Surface): coordinates, graph, flat map,
  inflation, picking.
- overview: the overview that the mesh command writes with a white surface, and a
  Surface on it.
- data: a site's files, read for viewing (Site, which holds a Surface).
- labels: the review of pial-side label proposals (baillarger.pial_labels).
- server: the local web server, its shared state and the page (static/index.html).

Run: python -m baillarger.viewer <column_volume_<name>.npz> --cache DIR [--joint ...]
     [--labels pial_labels_<name>.npz]
     python -m baillarger.viewer --labels pial_labels_<name>.npz  (labels only)
"""
