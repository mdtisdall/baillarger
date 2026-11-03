# Phase 1.5: the label set

Stream 1, phase 5 of `docs/version-2-plan.md` (§3d). This phase needs no other
phase. It is in round 1 (`docs/plans/round-1.md`).

The plan also asks for a list of the FLASH hemispheres that exist and are
expected. The user removed that item: there are many hemispheres, and the
model improves as it trains on them. Do not write an inventory.

The plan also asks for test regions: boxes labelled from scratch and held
out of training. The user removed them too. For now, success means that a
person stops needing manual edits to get the hemisphere they want. A
quantitative measure, when it is needed, uses a new hemisphere. No data is
held out. Do not write a test-region spec.

## Goal

1. A label set with a written spec, and one module in the code that holds
   it.
2. A decision for each label that FreeSurfer's fill puts in a different
   place from our `FILLED_LABELS`, with the reason.

## Done when

1. `docs/formats/label-set.md` exists.
2. `src/baillarger/labels.py` holds the label set.
3. A stacked PR on phase 1.0's split makes `white_dseg` use `labels.py`.
4. `nix develop --command scripts/check` passes on each PR.

## Facts from the current code

- `src/baillarger/white.py:93–98`:
  - `FILLED_LABELS = {"R": (41, 43, 49, 50, 51, 52, 192), "L": (2, 4, 10,
    11, 12, 13, 192)}`: WM, lateral ventricle, thalamus, caudate, putamen,
    pallidum, CC. The hippocampus is not filled.
  - `WM_LABEL = {"R": 41, "L": 2}`.
- `white.py:101–122`: `LUT_NAMES` for 0, 2, 3, 4, 10–13, 17, 18, 41–43,
  49–54, 192. It does not have 5/44, 26/58, 28/60, 30/62 or 31/63. The
  writer names a label that is not in the table `label-{k}`.
- `src/baillarger/data.py:33`: `ASEG_OUTSIDE, ASEG_WM, ASEG_CORTEX = 0, 41,
  42`.
- `docs/formats/wm-dseg.md:32–39` has the filled-label table and the reason
  for the hippocampus.
- No file records which labels occur in the hemisphere aseg.
- The data: `data/sub-INDD121850_ses-7THemix20250516_aseg.nii.gz` (11 MB),
  right hemisphere.
- Specs: each doc in `docs/formats/` has a "Files" table, content sections,
  and a "Sidecar keys: <version>" table. `test_bids.py:102` checks the key
  table against the code's `SidecarSpec`.

## Phase 1.0 runs at the same time

Phase 1.0 splits `white.py` into `white_dseg.py`, `white_surf.py` and
`white_overview.py`. Its first PR is `refactor/white-split`.

- Do not change `white.py` or the new white modules on your first branch.
- After the coordinator tells you that the split PR is open, make your second
  branch from it. On that branch, `white_dseg` and `data.py` take their label
  constants from `labels.py`.

## Design

### Step 1: the labels in the aseg

Load the aseg. List each label value, its FreeSurferColorLUT name and its
voxel count. Put the table in the PR body. This is a light job (less than
1 GB). It needs no message to the coordinator.

### Step 2: compare with FreeSurfer's fill

FreeSurfer makes its filled WM with `mri_edit_wm_with_aseg` and `mri_fill`.

1. Find what FreeSurfer does with each of these labels: accumbens (26/58),
   ventral DC (28/60), inferior lateral ventricle (5/44), choroid plexus
   (31/63), vessel (30/62). Also check the labels in our `FILLED_LABELS`.
2. Use the FreeSurfer documentation and source to learn the behaviour. Do not
   copy FreeSurfer code. Give a link and a line reference for each claim.
3. For each label, write a proposal: in the filled mask, or out of it. Give
   the reason. Give the voxel count from step 1. The plan says: follow
   FreeSurfer, and deviate only with cause and data.
4. These are proposals. The user decides in the PR review. Mark each one
   "Proposed" in the spec.
5. The hippocampus (17/53) stays open. Phase 1.8 decides it. Write that in
   the spec.

### Step 3: the label-set spec

Write `docs/formats/label-set.md`. It has:

1. The classes that the segmentation must give: outside, WM, cortex, the
   deep structures of the filled mask, the hippocampus, and other labels only
   where a later step needs them. Give the FreeSurferColorLUT values for each
   hemisphere.
2. The filled mask, with the decisions from step 2.
3. "Excisions are outside". The label set has no class for missing tissue
   (plan §3c, step 1).
4. The majority rule for partial volume (plan §3c, step 1, item 5). A voxel
   takes the class that fills most of it. Write it as the rule that a person
   follows when they paint labels. Give the two cases from the plan: a fluid
   channel narrower than half a voxel, and two banks pressed pia to pia.
5. A version: `label-set/1`.

### Step 4: `labels.py`

Make `src/baillarger/labels.py`. It holds:

- the label values for each hemisphere and their names (the content of
  `LUT_NAMES`, with the new labels);
- `FILLED_LABELS` and `WM_LABEL`, with the step 2 decisions;
- the version string `label-set/1`.

Write a test that checks that `docs/formats/label-set.md` lists the same
filled labels as `labels.FILLED_LABELS`.

On the stacked branch only: `white_dseg` and `data.py` import from
`labels.py`. If a step 2 decision changes `FILLED_LABELS`, tell the
coordinator before you push. The change alters the hemisphere outputs of
phase 1.0.

## Work breakdown

| Step | Work | Who | Can run with |
|---|---|---|---|
| 1 | The table of labels in the aseg | Haiku sub-agent | 2 |
| 2 | FreeSurfer's fill: the research and the proposals | Phase session (it needs judgment and links) | 1 |
| 3 | The label-set spec | Phase session | after 1 and 2 |
| 4 | `labels.py` and its test | Sonnet sub-agent | after 3 |
| 5 | The stacked PR: `white_dseg` and `data.py` use `labels.py` | Phase session | after 4 and the 1.0 split PR |

## PRs

1. `docs/label-set`: steps 1–4.
2. `refactor/dseg-label-set`: step 5. Stacked on phase 1.0's
   `refactor/white-split`.

## Coordination

- Send the step 2 proposals to the coordinator when they are ready.
- Do not decide the hippocampus.
