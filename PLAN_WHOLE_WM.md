# Plan: the white mesh of a whole hemisphere

Planning file for the first stage of the move from boxes to whole hemispheres:
the white (WM) mesh. Later stages stay on boxes until their turn; each is
scaled when we get to it, and only the computational problems that we meet
are fixed. When this stage is done, the lasting content goes into the module
docstrings and the format specs (`docs/formats/`), and this file is deleted.

## The two commands

```
python -m baillarger.white dseg ASEG --out DERIV
python -m baillarger.white mesh DSEG --out DERIV [--sigma S]
```

1. `dseg` reads the aseg and writes a corrected segmentation: the aseg with
   the filled WM mask made handle-free and cavity-free. The aseg file itself
   is never changed.
2. `mesh` reads that segmentation and writes the white mesh of the filled WM
   mask, as GIfTI.

Every later stage reads the corrected segmentation, not the aseg plus a list
of changes, so the mesh and the labels cannot disagree. Neither command takes
a box: they work on the whole grid.

## BIDS conventions

- Inputs are paths to BIDS-named files. The `sub` and `ses` entities are read
  from the input's name and carried to the outputs. The aseg's name
  (`sub-INDD121850_ses-7THemix20250516_aseg.nii.gz`) has a nonstandard
  suffix; any NIfTI with `sub-` and `ses-` entities is accepted.
- `--out` is a BIDS derivatives root. Files go to
  `DERIV/sub-<sub>/ses-<ses>/anat/`. The command writes
  `DERIV/dataset_description.json` (`DatasetType: derivative`, `GeneratedBy`:
  baillarger, its version and commit) if it is not there.
- Every output has a JSON sidecar with the BIDS keys `Description` and
  `Sources` (input paths), and our own keys, all listed in the spec:
  `SourcesSHA256`, `Provenance` (created, function, version, commit, dirty)
  and `SpecVersion`.
- BIDS has no settled rule for surface derivatives (the structural
  derivatives extension is a draft). Where it says nothing, the spec records
  our choice as BIDS-like.

## Format specs

Each output has a written, versioned spec in `docs/formats/`
(`wm-dseg.md`, `white-surf.md`): every file, array and sidecar key, with
type, units, required or not, and meaning. One writer and one reader per
format, in `baillarger.white`, enforce it: the writer cannot leave out a
required key, and the reader refuses a missing key or an unknown
`SpecVersion`. A JSON Schema for each sidecar lets a test check it. A field
changes only through a PR that edits the spec and bumps the version.

## Step 1: `dseg`

Labels (FreeSurfer LUT). The filled WM mask of the right hemisphere is
41 (WM), 43 (lateral ventricle), 49–52 (thalamus, caudate, putamen,
pallidum) and 192 (corpus callosum); of the left, 2, 4, 10–13 and 192. The
hemisphere is the one whose WM label is in the aseg. The hippocampus (53 /
17) is not filled: it is archicortex, whose laminar axis does not run out
from the WM below it.

1. Read the aseg. Mask = its voxels with a filled label, on the mask's
   bounding box plus 2 voxels.
2. Fill the cavities: background components (Kuhn adjacency, as
   `handles.kuhn_betti` counts them) that do not reach the box's border.
3. Remove the handles (`handles.remove_handles`), which cuts or fills whole
   voxels with the Kuhn adjacency that marching tetrahedra matches, so the
   mesh of the result is genus 0.
4. The corrected segmentation: the aseg, with
   - cavity voxels and handle fills set to the WM label;
   - each handle cut given the label of the nearest voxel that is outside the
     mask and not a cut (Euclidean distance transform with indices, on the
     cuts' bounding box plus a margin).

   Check: its voxels with a filled label are exactly the mask of step 3.
5. Write:
   - `sub-<sub>_ses-<ses>_desc-wmtopo_dseg.nii.gz`: the corrected
     segmentation (int16, the aseg's grid and header), with `_dseg.tsv` (the
     label names present, from the FreeSurfer LUT) and `_dseg.json`:
     the filled labels, hemisphere, the Betti numbers (b0, b1, b2) before and
     after, voxels set to WM by cavity fill and by handle fill, voxels cut and
     their new labels, and per move its kind, voxels and centre.
   - `sub-<sub>_ses-<ses>_desc-wmtopofixes_dseg.nii.gz` (diagnostic, uint8):
     1 cut, 2 handle fill, 3 cavity fill; with `.tsv` and `.json`.

## Step 2: `mesh`

1. Read the corrected segmentation; mask = its voxels with a filled label
   (the label set from its sidecar). Check `kuhn_betti(mask)` has b1 = 0 and
   b2 = 0, and refuse otherwise.
2. Marching tetrahedra at 0.5 of the mask, in blocks (below), faces oriented
   out of the WM.
3. Tangential relaxation, twicing Gaussian smoothing (sigma in voxels,
   default 2) and smoothed normals (`mesh_smoothing`), as now.
4. Write:
   - `sub-<sub>_ses-<ses>_hemi-<R|L>_white.surf.gii`: vertices (float32,
     scanner RAS mm, `NIFTI_INTENT_POINTSET`, coordinate system
     `NIFTI_XFORM_SCANNER_ANAT`) and triangles (int32,
     `NIFTI_INTENT_TRIANGLE`); GZipBase64Binary; metadata
     `AnatomicalStructurePrimary` (`CortexRight` / `CortexLeft`) and
     `GeometricType` (`Anatomical`).
   - The smoothed unit normals (float32, out of the WM), as a vector array in
     a GIfTI of their own, so that the `.surf.gii` holds only what every
     surface reader expects. Its name is settled in the spec.
   - `..._white.json`: `Sources` (the corrected segmentation) and its
     SHA-256, the filled labels, sigma, the voxel-to-RAS affine of the grid
     (so a reader gets voxel coordinates without the segmentation), and the
     summary: vertices, faces and per piece its genus, boundary loops and
     area.

The reader (`WhiteSurface.load`) returns vertices in voxel coordinates of the
grid, faces, normals and the affine; `check(dseg_path)` compares the file's
SHA-256 with the sidecar's.

No per-vertex cortex label: every vertex is a column root. Columns run out
from the WM until they reach background (label 0) or a maximum length, and
what they pass through (cortex, hippocampus, water mislabelled as cortex) is
left to later stages.

## Blocked marching tetrahedra

Needed: in one piece, `marching_tetrahedra` keeps about 6.5 kB of arrays per
surface cell, about 40 GB for the hemisphere's 5.85 M surface cells.

- Cells are taken in blocks (for example 64³ cells). A block whose voxels are
  all inside or all outside is skipped: no whole-grid float copy and no
  whole-grid passes.
- Each block's vertices are keyed by their grid edge in the full grid, so the
  blocks are joined by one `np.unique` over the keys at the end. Each
  tetrahedron belongs to one cell, so the result is exactly the mesh of the
  whole grid at once.
- Peak: a few GB.

## What goes

- From `baillarger.white`: `WMChanges`, `wm_changes` and the `changes`
  subcommand (replaced by `dseg`); `white_matter_box`; the `box` subcommand,
  `box_report`, `write_box` and `overlay`; `box_mesh`, the box fields of
  `WhiteMesh` and its `.npz`; the mesh command's `--patch`, `--lo`, `--hi`;
  the `.vtk` output (`mesh.write_vtk` stays for other tools).
- The box stages that took the WM changes (`column_traces.load_tissue`,
  `changes=`; `pial_labels`, `--wm-changes`) read their box's labels from the
  corrected segmentation instead, as they read the aseg now. Their `mesh=` /
  `--mesh` option goes: they make their own box's mesh until their stage is
  scaled.
- `prototypes/scripts/wm_handles*.py` stay as records and are not updated.

## Tests

- Blocked marching tetrahedra equal the one-shot result on phantoms
  (vertices, faces, orientation), with blocks smaller than the features, and
  skipped blocks change nothing.
- `dseg` on a synthetic aseg with a torus-shaped WM, an enclosed ventricle
  and a cavity: Betti numbers after (b0, 0, 0), the filled-label voxels equal
  the corrected mask, cavity voxels set to WM, cuts relabelled by the nearest
  rule, other voxels unchanged.
- Names: entities read from the input and carried; derivatives layout;
  `dataset_description.json` written once.
- GIfTI round trip: coordinates (voxel → mm → voxel), faces, orientation,
  normals; the reader refuses a missing key or a wrong `SpecVersion`; each
  sidecar validates against its JSON Schema.
- `mesh` refuses a segmentation whose mask has a handle or a cavity.

## The full run

Both commands on the example hemisphere under `scripts/memguard`, logging the
time and peak memory of each step; results in `scratch/whole-wm/`.

Measured so far (right hemisphere, grid 700 × 384 × 1148, mask box
632 × 323 × 1006):

| Step | Time | Peak | Result |
|---|---|---|---|
| Filled mask, cavities filled | 8 s | 3.6 GB | Betti (1, 77, 0) |
| Handle removal | 3 min | 21.4 GB | 79 moves: 580 voxels cut, 556 filled; (1, 0, 0) |
| Marching tetrahedra, one shot (V1V2 box) | 2.8 s | 2 GB | 0.91 M vertices, 1.8 M faces |
| Relax, smoothing, normals (V1V2 box) | 25 s | < 0.2 GB more | |

For comparison, label 41 alone has Betti (1, 243, 31), and 14 % of the
surface of its outer voxels faces something other than cortex (8 % with the
filled labels, mostly the medial cut face against the Fomblin).

Expected for the hemisphere: about 17 M vertices and 35 M faces; about
15–20 min in all; peak about 21 GB (handle removal); the `.surf.gii` about
0.5 GB.

Left until a measurement asks for it: faster smoothing (one shared
adjacency, 1-D edge keys instead of `np.unique(axis=0)`, `bincount` instead
of `np.add.at`); handle removal in blocks round each handle (whole-brain
scale would be about 43 GB); decimating the mesh (a question for the column
stage).

## Open

- The normals' file name and intent, settled when writing `white-surf.md`.
- Whole brains: the corpus callosum joins the two hemispheres' filled masks,
  so a whole brain needs a midline cut (FreeSurfer's cutting planes) before
  step 1. Not needed for the example hemisphere.
