# Wave 2.5 of the version-2 plan

This file gives the rules for wave 2.5 of `docs/version-2-plan.md` (§9,
*Waves*). Wave 2.5 adds only the tools that a disciplined review of the wave
2 outputs needs. Then the user reviews. It adds no new algorithm, no new
estimator and no stitching work.

The rules of `round-2.md` apply, with the changes in this file. Where this
file and `round-2.md` differ, this file is correct.

## The steps of wave 2.5

| Step | What | Plan | Who |
|---|---|---|---|
| 1 | The input set: `excisions`, `dseg`, the white surface and the overview, made again from `main` | this file, "The input set" | the coordinator |
| 2 | Signals per column, as batch runs | `wave-2.5-signals.md` | session S |
| 3 | Viewer tools | `wave-2.5-viewer.md` | session V |
| 4 | The review | the plan's §4b and §9 | the user |
| 5 | The plan revision, which writes wave 3 | — | the coordinator, with the user |

Sessions S and V start at the same time. Session V does not wait for S's
outputs: it tests on phantom files in the format that
`wave-2.5-signals.md` gives. Step 4 starts when the hemisphere signals exist
and V's pane shows them.

Flattened profiles at a test site are not in wave 2.5. They come after the
user finds the columns good enough (plan §1, working rules).

## The input set

1. The reference set is `scratch/v2/derivatives/`. It does not change. It is
   kept for comparison.
2. The new set is `scratch/v2/derivatives-main/`. It is made from `main` at
   1f67376, with all four steps (`excisions`, `white_dseg`, `white_surf`,
   `white_overview`). The logs and the commit are in
   `scratch/v2/logs/main/`.
3. The coordinator compares the new set with the reference set before any
   session reads it. The coordinator sends the result to both sessions: the
   files that differ in format, the counts (vertices, faces, pieces,
   misfacing faces), and the sidecar changes.
4. All hemisphere runs and checks of wave 2.5 read the new set. Each output,
   PR body, figure and note names its input set.
5. Vertex ids change from one set to the next. Do not compare
   vertex-indexed results across sets. Compare only through edge keys
   (`desc-edgekeys_white.func.gii`) or through 3D positions.

## The sessions

| Session | Plan | Branches |
|---|---|---|
| S | `wave-2.5-signals.md` | `feature/column-signals`, then `feature/neighbour-shifts` (stacked) |
| V | `wave-2.5-viewer.md` | `fix/viewer-wave2-bugs`, `feature/viewer-signals`, then `feature/viewer-notes` (stacked) |

The two sessions change different files. Session S changes no file in
`src/baillarger/viewer/`. Session V changes no file outside
`src/baillarger/viewer/`, `tests/test_viewer_*.py` and
`docs/formats/review-notes.md`. It can also change the local, untracked
`.claude/launch.json`.

The format of the signals is in `wave-2.5-signals.md`: the published GIFTI
files (`column-signals/1`) and the provisional `edges.npz`. Session S
writes the spec file `docs/formats/column-signals.md` in its first PR. If
session S must change the format, it asks the coordinator first. The
coordinator tells session V.

## Heavy runs

1. A heavy job is one with a peak above 8 GB, or one that uses more than 4
   cores for more than 10 minutes (as in `round-2.md`).
2. The regeneration of the input set runs first. No other heavy job starts
   until the coordinator says that the new set is ready.
3. Session S's hemisphere runs are heavy. Each one asks the coordinator for
   a slot, as `round-2.md` says.
4. The viewer server on the hemisphere is not heavy if its peak stays below
   8 GB. If it goes above, ask for a slot.

## Writing

As in `round-2.md`: ASD-STE100 style, with the `asd-ste100` skill, and one
word for one thing.

These words have one meaning in wave 2.5:

- **signal**: one value for each root (or each mesh edge, or each mesh
  face), from a batch run. A signal shows where to look. It is not a pass or
  fail rule.
- **site**: one row of a ranked list: a vertex and its value.
- **note**: one record that the user writes in the review.
- **input set**: one derivatives directory, named by its path and the
  SHA-256 of its white-surface sidecar.
- **provisional limit**: a known limit of a working version, stated with
  each signal that comes from it (plan §1, working rules).

## Questions and decisions

As in `round-2.md`, with one change: the user is not away. Send questions
to the coordinator. The coordinator asks the user where a decision is the
user's.

Diagnostics, not criteria: no signal gets a threshold that decides "good"
or "bad". A ranked list orders the sites by value. The user decides what is
an error.

## When wave 2.5 is done

1. Every PR of sessions S and V is open, and its CI is green.
2. The hemisphere signals exist on the new set, and the viewer shows them,
   with the ranked lists and the notes log.
3. Each session sends the coordinator a report: the PRs, the findings that
   change the plan, the decisions for review, and the open questions.
4. The user's review (step 4) and the plan revision (step 5) follow.
