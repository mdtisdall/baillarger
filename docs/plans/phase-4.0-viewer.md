# Phase 4.0: the viewer on stream 1's outputs

Stream 4, phase 0 of `docs/version-2-plan.md` (§6d). The user added this
phase to round 1 (`docs/plans/round-1.md`). The rules in `round-1.md`
apply.

## Why it can start now

The plan says that phase 4.0 needs phase 1.0. Only the check on the
hemisphere needs 1.0's outputs. The other work needs only:

- the published formats `wm-dseg/1`, `white-surf/1` and `white-overview/1`,
  which exist on `main`;
- the new module names from phase 1.0's first PR, `refactor/white-split`;
- the `mesh_graph` fix, for the flat map of #69.

Do the work in the order of "Work breakdown". Use phantoms for the tests
until 1.0's hemisphere outputs exist.

## Goal

The viewer on `main` reads only published outputs for the white surface, and
it includes the work of three open PRs:

- #84: the brick volume viewer (`viewer/bricks.py`, `BrickSource`);
- #82: the volume pane in `--white` mode (a box round the WM selection);
- #69: the flat map of a geodesic patch round a selected vertex.

## Done when

1. The viewer imports the white surface, the overview and the dseg from
   1.0's new modules, not from `white.py`.
2. The code of #84, #82 and #69 is on this phase's branches, with their
   tests, and the conflicts between them are resolved.
3. The flat map of #69 uses the fixed `mesh_graph.edge_graph`.
4. The viewer runs on 1.0's hemisphere outputs in `scratch/v2/derivatives/`.
   The PR body shows screenshots of each pane in `--white` mode.
5. `nix develop --command scripts/check` passes on each PR.

## Facts from the current code

### The viewer modules on `main`

| Module | Lines | Reads |
|---|---|---|
| `viewer/server.py` | 468 | `filters`, `streamlines`, `data.Site`, `labels.LabelReview`, `white_view.WhiteView` |
| `viewer/overview.py` | 65 | `white.WhiteOverview`, `white.WhiteSurface`, `white.overview_paths` |
| `viewer/white_view.py` | 86 | `viewer.overview` |
| `viewer/surface.py` | 184 | `mesh_smoothing.edges`, `flatmap`, `inflation`. It has its own function `mesh_graph`. It is not the module. |
| `viewer/flatmap.py`, `inflation.py`, `sections.py` | 618, 229, 215 | "port as is" in the plan |
| `viewer/data.py`, `filters.py`, `labels.py`, `streamlines.py` | | prototype `.npz` files. Phase 4.3 replaces them. |

### The three PRs

| PR | Branch | Changes |
|---|---|---|
| #84 | `feature/volume-viewer` (1 commit) | new `viewer/bricks.py`, `tests/test_viewer_bricks.py`; `server.py` (+72), `static/index.html` (+287), `docs/scaling.md` |
| #82 | `feature/white-volume-pane` (1 commit) | `server.py` (+26), `static/index.html` (+45), `white_view.py` (+117), `tests/test_viewer_overview.py` (+72) |
| #69 | `feature/geodesic-patch-flatmap` (2 commits) | `viewer/flatmap.py` (+177), `tests/test_viewer_flatmap.py` (+117) |

- #84 and #82 both change `server.py` and `index.html`. #84's body says
  that a one-line conflict in the `surface_only` block is expected, and
  that #82's logic plus the `?vol=bricks` line resolves it.
- #69's tests call `mesh_graph.edge_graph`. The fix is in a different session
  (`fix/mesh-graph-duplicate-edges`).

## Old code

The user's decision: a module that the plan drops stays until the phase
that replaces its callers deletes it. In this phase:

- Keep `data.py`, `filters.py`, `labels.py`, `streamlines.py` and the
  server routes that use them. Phase 4.3 replaces them.
- Do not close #84, #82 or #69. Write "Superseded by #N" in a comment on
  each PR when its code is in one of this phase's PRs. The user closes them.

## Design

### Step 1: bring in #84 and #82

1. Make the branch `feature/viewer-volume-panes` from `origin/main`.
2. Merge `origin/feature/volume-viewer`. Then merge
   `origin/feature/white-volume-pane`. Resolve the conflicts as #84's body
   says. Keep both PRs' commits (use merges, not a squash), so that the
   history shows where each change came from.
3. Run all the viewer tests.
4. Start the viewer with a phantom or a small box, in the browser pane. Check
   that both panes work: the NiiVue box in `--white` mode, and
   `?vol=bricks`. Use the `dev-workflow:browser-check-localhost` skill if the
   page needs a local server.

### Step 2: the new white modules

Do this step after phase 1.0's `refactor/white-split` PR is open. The
coordinator tells you when. Make the branch `refactor/viewer-white-modules`
from `feature/viewer-volume-panes`, then merge `refactor/white-split` into
it.

1. Change `viewer/overview.py` to import from `white_surf` and
   `white_overview`.
2. Change every other import of `white` in `viewer/` and in the viewer tests.
3. No route or page changes in this step.

### Step 3: the geodesic-patch flat map

Do this step after the `mesh_graph` fix PR is open. The coordinator tells
you when.

1. Make the branch `feature/viewer-patch-flatmap` from
   `refactor/viewer-white-modules`. Merge the fix branch and
   `origin/feature/geodesic-patch-flatmap` into it.
2. Run #69's tests on the fixed `edge_graph`. If an expected value changes,
   find the cause. Change the value only when the new one is correct. Do not
   change a tolerance.
3. Check whether the viewer calls `geodesic_patch` yet. If no route calls it,
   write that in the PR body. Do not add a route in this phase unless the
   coordinator agrees.

### Step 4: the hemisphere check

Do this step after phase 1.0 writes its hemisphere outputs. The coordinator
tells you when.

1. Start the viewer in `--white` mode on `scratch/v2/derivatives/` and the
   echoes in `data/`.
2. The viewer builds the brick pyramid of echo 2 (about 1.4 GB peak in #84).
   This is not a heavy run. It needs no message to the coordinator.
3. Take screenshots: the overview, a WM selection with the NiiVue box and the
   fixes overlay, the bricks pane at two zoom levels, and the patch flat map
   if a route calls it. Attach them to the PR of the last step.
4. If a pane fails on the new outputs, diagnose it and write a test with an
   exact answer before you fix it.

### Step 5, optional: the normal length

When phase 1.0's `feature/white-normlen` PR is open, ask the coordinator
whether to show |n̄| as a surface overlay. Do not start it without an answer.

## Remote use

The user wants to keep this use possible: the server on a big machine, and
the browser on the laptop through an SSH tunnel. Do not add a change that
needs the browser and the server on the same machine. Examples: file paths
in the page, or `file://` URLs.

## Work breakdown

| Step | Work | Who | Can run with |
|---|---|---|---|
| 1 | Merge #84 and #82, resolve the conflicts, check in the browser | Phase session | now |
| 2 | Imports from the new white modules | Sonnet sub-agent; the phase session reviews | after the 1.0 split PR |
| 3 | The patch flat map on the fixed `edge_graph` | Phase session | after the `mesh_graph` fix PR and step 2 |
| 4 | The hemisphere check and screenshots | Phase session | after 1.0's run |

## PRs

Stack them in this order:

1. `feature/viewer-volume-panes`: step 1.
2. `refactor/viewer-white-modules`: step 2. Stacked on PR 1 and on 1.0's
   `refactor/white-split`.
3. `feature/viewer-patch-flatmap`: step 3. Stacked on PR 2 and on the
   `mesh_graph` fix.

Write the screenshots of step 4 in the last open PR.

## Coordination

- Ask the coordinator before a change to a published format or to a module
  that another phase owns (`white_*`, `mesh_graph`, `excisions`, `labels`).
- While you wait for a PR of another phase, continue the steps that do not
  need it.
