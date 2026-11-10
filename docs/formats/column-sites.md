# column-sites: the review queue of the column top stage

SpecVersion `column-sites/1`. Written by
`python -m baillarger.column_sites SURF CONF FEAT... [--out DIR] [--radius MM]`
(`baillarger.column_sites.write_column_sites`); read by
`baillarger.column_sites.read_column_sites` and
`baillarger.column_sites.read_members`. The sidecar's keys are enforced by
`baillarger.column_sites.COLUMN_SITES_SPEC`, and `tests/test_bids.py` checks
that the table below lists the same keys, types and requirements.

The column top stage (`reports/Column top stage.md`) gives each coarse
column a top and a confidence. Its review queue puts the columns with no
top first, then the lowest confidence first. The sites group the columns of
that queue, so that a person answers one problem (for example one root in
fluid) one time. Every column is in exactly one site. Confidence orders the
sites. It does not gate a column: no column is left out, and the reviewer
stops where they choose.

The answers at these sites go to the answer log of the WM review
(`wm-edits/1`): the site's voxel is the `Site`, its `kind` is the
`SiteKind`, its `rank` is the `Rank` and its `score` is the `Score` (null
for `n/a`). The session records the SHA-256 of this table as its
`QueueSHA256`. `dseg` replays the answers as it replays any site answer.

## Inputs

- `SURF`: the white surface (`white-surf/1`) that the stage cast its rays
  from. Its sidecar's first source is the corrected segmentation
  (`wm-dseg/1`). The sites use that segmentation's grid.
- `CONF`: the stage's top confidence (`.npz`, from
  `prototypes/scripts/top_confidence.py`). The sites read these keys, one
  row per column: `vert` (the column's vertex of the white surface), `root`
  (the root, world mm), `rank` (place in the stage's queue, from 0), `conf`
  (the confidence, `-inf` for no top), `top_k` (negative for no top),
  `top_mm` (the top's distance from the root, NaN for no top), `terms` and
  `term_names` (the confidence terms, NaN for a term that does not apply),
  `gm_behind` and `fl_behind` (mm of GM-like and fluid-like signal behind the
  root).
- `FEAT`: the stage's feature files (`.npz`). The sites read `edges_v` from
  each: edges of the column graph, as pairs of vertices. An edge with a
  vertex that is not a column is dropped.

## The sites

1. Take the columns in the order of `rank`.
2. A column joins the site of a neighbour (on the column graph) that came
   before it, if the column is within `RadiusMm` (world mm, between roots)
   of that site's first column, its seed. With more than one such site, it
   joins the site with the earliest seed.
3. Otherwise, the column starts a new site and is its seed.

So each site is connected on the column graph, and all its columns are
within `RadiusMm` of its seed. `RadiusMm` is the size of one review view,
not a measure of correctness. The default is 2 mm.

## Files

Next to the white surface, in `DERIV/sub-<sub>/ses-<ses>/anat/`, or in `DIR`
if given, hemisphere `<h>`:

| File | Content |
|---|---|
| `sub-<sub>_ses-<ses>_hemi-<h>_desc-columnsites_sites.tsv` | One row per site, in queue order |
| `sub-<sub>_ses-<ses>_hemi-<h>_desc-columnsites_sites.json` | Sidecar, `column-sites/1` |
| `sub-<sub>_ses-<ses>_hemi-<h>_desc-columnsites_members.tsv` | One row per column: `vertex` and `site` (the site's `rank`), sorted by site, then vertex |

The `sites` suffix is ours (BIDS has none for this), in BIDS's entity
order. The tables are tab-separated, with a header row. `n/a` is no value.

## The table

The sites in the order of their seeds in the stage's queue. The first six
columns are as in `wm-sites/1` (`rank`, `kind`, `i`, `j`, `k`, then
`score`).

| Column | Type | Meaning |
|---|---|---|
| `rank` | integer | Place in the queue, from 1 |
| `kind` | string | `column-no-top` (the seed has no top) or `column-top` |
| `i`, `j`, `k` | integer | The seed's root, rounded to the nearest voxel (0-based) of the segmentation's grid, clipped into the grid |
| `score` | number or `n/a` | The seed's confidence; `n/a` for no top |
| `vertex` | integer | The seed's vertex of the white surface |
| `columns` | integer | The site's columns |
| `no_top` | integer | Its columns with no top |
| `median_score` | number or `n/a` | The median confidence of its columns; `n/a` if not finite |
| `extent_mm` | number | The largest distance (mm) from a column's root to the seed's root |
| `top_mm` | number or `n/a` | The seed's top, mm from its root; `n/a` for no top |
| `gm_behind_mm`, `fluid_behind_mm` | number | The seed's GM-like and fluid-like length behind its root (mm) |
| `lowest_term` | string | The name of the seed's lowest confidence term; `n/a` if it has none |

The measures after `score` order nothing. They help the reviewer and the
diagnostics.

## Sidecar keys: column-sites/1

| Key | Type | Required | Meaning |
|---|---|---|---|
| `SpecVersion` | string | yes | `column-sites/1` |
| `Description` | string | yes | What the files are |
| `Sources` | array of strings | yes | The white surface's `.surf.gii` and `.dat`, the corrected segmentation, the top confidence, then the feature files (absolute paths) |
| `SourcesSHA256` | object | yes | SHA-256 of each source, by path |
| `Provenance` | object | yes | `Created` (ISO 8601), `Function`, `Version` (baillarger), `Commit`, `Dirty` |
| `Hemisphere` | string | yes | `R` or `L` |
| `RadiusMm` | number | yes | The largest distance (mm) of a column to its site's seed |
| `Columns` | integer | yes | The columns in the sites (rows of the members table) |
| `Edges` | integer | yes | The edges of the column graph that the grouping used |
| `Sites` | integer | yes | Rows of the table |
| `SitesByKind` | object | yes | Rows of the table, by kind |
| `TermNames` | array of strings | yes | The stage's confidence terms, in its order |
