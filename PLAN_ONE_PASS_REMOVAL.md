# Plan: column traces without the one pass

Branch `feature/traces-without-one-pass`. This file stays on the branch and goes before the merge;
lasting content moves into the report.

## Goal

`baillarger.column_traces` builds the column volume's traces from one surface (the white mesh) and
one field (the fitted field T), with no fast-marching pass and no fallback path, by default. The
method before stays available (`--columns one-pass`) for head-to-head comparison.

## What changed and why

- **Columns turned where T's gradient broke down, not where they went wrong.** The turn
  diagnostic (`turn_diag.py`, `turn_cases.py`) showed:
  - the primaries turned about 2 mm out, not at the root;
  - near the pia it was an artefact (T filled flat outside the cortex);
  - elsewhere the turns were at T's crest, where two banks' fields meet.

  The paths were right; a one-step wiggle tripped the 60° rule, and 14–16% of good paths were
  replaced by the fallback.
- **T extended into the fluid and the WM** (`fitted_field(fluid=, wm_band_mm=)`): the gradient is
  T's own up to the pia. On the fold phantoms, crowns and fundi go from 84–88% to 95–96% right
  class (`turn_experiment.py`).
- **No turn rule and no fallback.** A bend angle cannot tell a column's ending from a crossing
  that kinks at a contact: at the sites, the classes change by 5% (60° to 45°) and 34% (at 10°;
  `bend_sweep.py`).
- **The contact is recorded, not decided.** Where a column meets another bank's field it crosses
  T's crest, found by `streamlines.crest_voxels`:
  - the test is a kink test: T rises into the voxel, does not rise beyond it, and its slope changes
    by more than a smooth level set's curvature allows (L² κ, with κ from the white mesh carried to
    depth) plus a noise margin;
  - the noise comes from `step_noise`, derived from the structure tensor's own eigenvalues and the
    filters (`n_eff`), with no calibration;
  - it is checked on the phantoms (`crest_noise.py`): actual over predicted noise is 1.0–1.2 (1.5
    in the lowest quartile); one-bank phantoms have 0.01–1.1% crest voxels; the crests are 98–99.7%
    unchanged by noise.

  The crest is the midline between the two WM surfaces, not the anatomical contact. It is 0.3 mm
  off for banks of 2.0 and 2.6 mm, and it does not see a narrow fluid gap. So columns are traced
  through, and `crest_at` records where each one meets it: there the boundary has to come from the
  reading of the columns (step 2), and from a human where that reading is uncertain.
- **Reversals.** A column that reflects off a crest it cannot cross and goes back into its own WM
  is cut at T's maximum along it (`cut_at_crest`; "to the contact").
- **Mesh patches.** The bank rule (5 mm along the white surface) uses geodesic patches on the
  white mesh (`baillarger.mesh_graph`):
  - the graph includes the edges across neighbouring triangles: at most 8.2% over the straight
    line on a plane, where the triangles' own edges give up to 41%;
  - measured at M1S1: 1.3–1.7% over the straight line at p10, against 6–7% for the voxel seed
    graph.
- **Viewers.** The one-pass horizons ("one pass, equal", "one pass, field", "field prior") are
  gone from the npz, `pia_horizon.py`, `flatten_volume.py`, `chart.py` and `edit_priority.py`
  (agreed with the edit-priority session).

## Still to do on this branch

1. The site builds by both methods (`scratch/columns-methods/`): classes, crest hits, marked
   points, against the one-pass method.
2. The report's subsection: results per role and estimator, as current state.
3. Checks; delete this file; PR.

## Not in this PR

- The GM/WM boundary refinement of the white surface (sent to the wm-topology session).
- Step 2's use of `crest_at` (an edit-priority layer; to discuss with that session when we get
  there).
- A top boundary for T at the pia, so that its crest would land on the anatomical contact.
