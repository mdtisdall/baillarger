# Phase 1.3: the review's formats and code

Stream 1, phase 3 of `docs/version-2-plan.md` (§3d). Read the plan's §1
(*Data for training, and for judging success*, items 3 and 4), §2 (*Manual
inputs*), §3c step 2 (*Answers before the correction*, *The topology
guarantee wins over an answer*, *Fixes are kept as a diff*) and §3c step 6.
The rules in `docs/plans/round-2.md` apply.

## Goal

Everything that the review pane (phase 4.2) calls, as stream 1 code with
written specs. Phase 1.3 needs no viewer.

1. The site finder (`wm-sites/1`), with random sites on request.
2. `dseg` publishes its fixes as a table and a sparse diff
   (`wm-dseg-fixes/1`).
3. `wm-edits/1` revised: answers stand on their own, in an append-only log,
   with scribbles, reviewer time, how each site was chosen, and session
   records. The log is the review queue's own record, which `dseg` does not
   read. `white_dseg --edits` applies one current diff of (index, label)
   records (`label-edits/1`, `docs/plans/label-editing.md`), with no edit
   history. The topology guarantee wins over an edit, and `dseg` marks and
   queues each such fix.
4. The site-answer choices as published code with a written contract.
5. The ITK-SNAP bootstrap, as a command (`wm_paint`), kept until the viewer's paint worked. The paint mode of `docs/plans/label-editing.md` (Part 2) removed it.

## Done when

1. `docs/formats/wm-sites.md`, `docs/formats/wm-edits.md` and
   `docs/formats/wm-dseg.md` agree with the code. `tests/test_bids.py`
   checks each sidecar spec against its document.
2. Each part passes the exact tests in its section below.
3. The site finder, `dseg` with the fix table, and a test edits diff run on
   the hemisphere. The PR bodies give the diagnostics in
   "The hemisphere runs".
4. Phase 4.2 can write code against a stable API. Send the API (Part F) to
   the coordinator before PR 3 opens.
5. `nix develop --command scripts/check` passes on each PR.

## The user's decision on #80 and #81

Do not stack on #80 or #81, and do not push to their branches. Copy what you
need into new branches from `origin/main`. The user closes #80 and #81
later.

## Facts from the current code

### PR #81 (`origin/feature/wm-sites`)

- New files: `src/baillarger/wm_sites.py` (289 lines),
  `docs/formats/wm-sites.md` (73 lines), `tests/test_wm_sites.py` (154
  lines). They do not collide with `main`.
- It also changes `src/baillarger/white.py` (a `sites` subcommand) and
  `tests/test_bids.py` (one parametrize row). Both conflict with `main`:
  `main` split `white.py` into `white_dseg.py`, `white_surf.py` and
  `white_overview.py`.
- `wm_sites.py` imports `SURF_SPEC` and `WhiteSurface` from
  `baillarger.white`. On `main` they are in `baillarger.white_surf`.
- Outputs: `..._hemi-<h>_desc-wmsites_sites.tsv` and `.json`. Columns:
  rank, i, j, k, faces, tissue, cut, fill, cavity, wm_face_pieces,
  wm_any_pieces, gap_face_pieces, gap_any_pieces, wm_thickness, gap_width.
- The queue order: grey matter, mixed, fluid, other. In each, faces
  descending, then fixes, then the centre voxel.

### PR #80 (`origin/feature/wm-review:PLAN_WM_REVIEW.md`)

- Lines 54–57 define the choices in prose:
  - `remove-strand`: WM voxels of the thin part, at most 1 voxel from
    outside by distance transform, face-connected.
  - `thicken`: cortex voxels within 1 (or 2) voxels of WM, set to the
    hemisphere's WM label.
  - `fill-gap`: non-WM voxels at most 1 voxel from WM on two sides (a gap 1
    voxel wide), set to the WM label.
  - `widen-gap`: WM voxels face-adjacent to such a gap, set to the most
    common non-WM label of their face neighbours.
- Line 136 leaves the parameters open (the radius R, thicken by 1 or 2).

### `wm-edits/1` on `main`

- `docs/formats/wm-edits.md` (66 lines): one row per changed voxel
  (`i,j,k,old,new,answer`), as a diff against one input aseg. `dseg`
  refuses the edits if `old` differs from the aseg's label. Answers carry
  `Id`, `Site`, `Choice`, `DecidedAgainst`, `Time`.
- `src/baillarger/wm_edits.py` (268 lines): `Answer` (56–102), `Edits`
  (109–185), `apply_edits` (188), `write_edits` (208), `read_edits` (248),
  `sources_match` (265).
- Callers: `white_dseg.py` lines 57, 189 (`wm_topo`), 290
  (`write_wm_topo`) and 370 (`--edits`). No viewer code uses it.
- Tests: `tests/test_wm_edits.py`, seven tests.

### `dseg` on `main`

- `white_dseg.wm_topo` (line 173) applies the edits first (187–190), then
  fills cavities (205–207), runs `remove_handles` (208–210), relabels cuts,
  and checks Betti (222–229).
- The fixes image (233–238): `fixes[manual] = 4`, then cut 1, fill 2,
  cavity 3. The codes are the `FIXES` dict, line 65. The image has no fix
  id.
- `handles.Move(kind, voxels)` (`handles.py` 335–343): kind is cut, fill or
  stranded; voxels in input coordinates. Cavity fills are not moves. They
  come from the `cavities()` array.
- `wm_topo` builds the sidecar's `Moves` (247–267): kind, voxel count,
  centre. No id and no bounding box.
- `DSEG_SPEC` (71–87). `FIXES_SPEC` (line 90) has `COMMON_KEYS` only.

### Specs

- `bids.SidecarSpec` (`bids.py` 173), `write_sidecar` checks before it
  writes (212). `tests/test_bids.py` 92–102 lists `(doc, spec)` pairs, and
  `doc_keys` (78) reads the table under `## Sidecar keys: <version>`.
- Each new or changed spec: define the `SidecarSpec`, write the table, add
  the pair to the test.
- Spec versions stay at 1 until a release. Edit them in place.

## Shared files with other phases

- **Phase 1.0.5 changes `white_dseg.py` and `wm-dseg.md`.** It adds the fix
  kind 5, `small_piece`. Part B stacks on 1.0.5's PR, so that the fix table
  includes that kind. Do not start Part B code in `white_dseg.py` until the
  coordinator tells you that 1.0.5's PR is open.
- Phase 4.2 builds on Parts C, D and F. It does not change your files.
- Ask the coordinator before you change `handles.py`.

## Design

### Part A: the site finder (`wm-sites/1`)

1. Copy `wm_sites.py`, `wm-sites.md` and `test_wm_sites.py` from
   `origin/feature/wm-sites`. Fix the imports to `baillarger.white_surf`.
2. Give the finder its own command, `python -m baillarger.wm_sites`, in the
   pattern of `python -m baillarger.white_dseg`. Do not bring back
   `white.py`'s subcommand.
3. Add the parametrize row to `tests/test_bids.py`.
4. **Random sites.** Add a function that draws n random, unranked sites for
   a session, from a seed that the session records. A random site is a
   voxel on the white surface, chosen with equal weight for each vertex.
   Mark each one as random in the output. The ranked sites do not change.
5. Keep the current ranking. Part C adds the fix sites.

Exact tests: the tests of #81 pass on `main`; the same seed gives the same
random sites; a different seed gives different sites; random sites are on
the surface.

### Part B: the fix table and the sparse diff (`wm-dseg-fixes/1`)

Follow the plan, §3c step 2, *Fixes are kept as a diff*.

1. Give each fix an integer id, in a fixed order: manual answers, then
   handle moves in the order that `remove_handles` makes them, then each
   cavity (one fix for each connected cavity), then the small pieces of
   1.0.5. The same input gives the same ids.
2. **The table:** `_desc-wmtopofixes_fixes.tsv`. One row per fix: id,
   kind, voxel count, centroid in mm, bounding box in mm, wall size,
   whether it overrides an answer (Part C). The margin in P is phase 1.6's.
   Do not add its column now.
3. **The diff:** per changed voxel, its linear index, fix id, old label and
   new label, in Morton order of the 64³ block grid. "Old" is the label
   after the edits are applied. Store it as a binary file with a sidecar.
   Choose the file format and give the reason in the spec (for example a
   NumPy `.npy` record array, or a `.tsv.gz`). It must be 12–16 bytes per
   voxel or less.
4. **Rebuild and revert.** A function rebuilds `desc-wmtopo` from the input
   labels, the edits diff and the fix diff. A function reverts one fix for
   inspection.
5. **Each run's diff is kept.** The sidecar records the SHA-256 of the input
   labels, P (none until phase 8), the edits diff and the code version. A
   rerun checks that it reproduces the stored diff when the inputs match.
6. Edit `docs/formats/wm-dseg.md` in place. Add the keys to `FIXES_SPEC`.

Exact tests on phantoms: a rebuild reproduces `desc-wmtopo` byte for byte;
revert of one fix changes only that fix's voxels; the diff is in Morton
order; ids are the same on two runs; the table's counts equal the fixes
image's counts.

### Part C: `wm-edits/1` revised, and the edits diff

Follow the plan, §3c steps 2 and 6, and §1 item 4. Edit `wm-edits.md` in
place. Keep the name `wm-edits/1`.

1. **Answers stand on their own.** An answer is the verified label of each
   voxel it covers: voxels and a new label, with no `old` and no check
   against an input. It is valid for any input labels.
2. **An append-only log.** Answers are in the order that they were made. A
   correction is a new answer. A later answer writes over an earlier one
   where they overlap. An answer can name the answers that it supersedes.
   Nothing in the log is rewritten or removed. `Edits.without` goes; give
   revert for inspection only, as a function that returns a new log
   without changing the file.
3. **Each answer records:** id, site (or none), choice, the class of each
   stroke for a scribble, the SHA-256 of the `dseg` the reviewer looked at
   (`DecidedAgainst`), the time it was made, the reviewer time from opening
   the site to submitting the answer, and how the site was chosen: random,
   or ranked with its rank and score.
4. **Two kinds of answer.** A site answer has voxels that `dseg` applies. A
   scribble has strokes, each with one class from the label set. Scribbles
   are training labels only. `dseg` does not apply them (§2, *Manual
   inputs*).
5. **An answer with no voxels** (`leave`) is recorded, so the site counts as
   answered.
6. **Session records.** For each session: its id, start time, the random
   share chosen at its start, the random seed, and every site shown, with
   the sites skipped. Raw facts only. Do not compute a measure in the log.
7. **`white_dseg --edits` applies the edits diff** over the input labels,
   before the correction. Each site answer writes its voxels into the diff.
   The log is not read by `dseg`. The SHA is provenance, not a gate.
8. **The topology guarantee wins.** Handle removal and the cavity fill rank
   verified voxels last. Where a defect can be fixed only through a
   verified voxel, `dseg` fixes it and marks the fix in the fix table as
   overriding an answer, with that answer's id. Ask the coordinator before
   you change the priority order in `handles.py`.
9. **The way out of the queue** (§3c step 2): the choice `accept-fix` at an override
   site accepts the fix and writes its labels into the edits diff; `leave` writes nothing. On a rerun, an answered override
   site goes back to the queue only if its fix is new or different. Match
   fixes across runs by voxel overlap with the kept diffs. Write the
   overlap rule in the spec.
10. **The queue.** The site finder adds fix sites to its output: large
    handle fixes by wall size, and override fixes. Use one queue file per
    hemisphere, with a column that gives the site kind.

The file layout is your choice. Give the reasons in the spec. One
suggestion: a JSON Lines log, one record per answer or session event, with
voxels in a binary side file keyed by answer id. Ask the coordinator before
you fix it, because phase 4.2 writes this file.

Exact tests on phantoms:

- Two answers on the same voxel: the diff keeps the later label.
- Supersession: a correction is a new record, and the old record stays.
- The diff applies to a different input label unchanged.
- A scribble does not change `desc-wmtopo`.
- An answer that leaves a handle: `dseg` fixes it, the fix table marks the
  override with the answer's id, and the site is in the queue.
- The choice `accept-fix` at that site accepts the fix and writes its labels into the diff. On a rerun with the same input, the
  site is not in the queue. On a rerun where the fix moves, it is.
- Update the seven tests in `tests/test_wm_edits.py` to the new model.

### Part D: the site-answer choices

1. One module, for example `src/baillarger/wm_choices.py`, with one function
   for each choice: `remove-strand`, `thicken`, `fill-gap`, `widen-gap`,
   `leave`. Each takes the current labels (a box round the site), the site,
   and its parameters. Each returns the voxels it changes and their new
   labels. It changes nothing.
2. Start from #80's definitions (above). Use Kuhn adjacency where the
   choice counts pieces (plan §2).
3. **Parameters.** The radius of each choice is a parameter with a default:
   1 voxel. The review pane offers 1 and 2 for `thicken`. Record the
   parameters in the answer.
4. **A preview function** runs the choice, then the correction (`wm_topo`)
   on a box round the site, and returns: the voxels the answer changes, the
   voxels that the correction would then change, and whether the
   correction would override the answer. The box has a halo in mm; give its
   size and the reason. Where the box cuts a handle, say what the preview
   cannot show.
5. `manual-paint` is the paint of Part E. It is not a function here.
6. Write the contract in `docs/formats/wm-choices.md`: inputs, parameters,
   outputs, units. It is code with a contract, like a sampler (plan §1).

Exact tests on phantoms: a strand one voxel thick is removed and nothing
else changes; a one-voxel gap is filled; a widened gap opens to two voxels;
`thicken` grows a two-voxel blade to four; `leave` returns no voxels; each
choice on a box gives the same voxels as on the whole phantom; the preview
of an answer that leaves a handle shows the override.

### Part E: the ITK-SNAP bootstrap

A command with two subcommands:

1. `export`: write a crop of the echoes and the current `desc-wmtopo` round
   a site, as `.nii.gz`, with the crop's offset in the sidecar.
2. `import`: read the painted crop, compare it with the exported labels,
   and write the changed voxels into the edits diff. The command was a
   bootstrap only. The viewer's paint mode (`docs/plans/label-editing.md`,
   Part 2) removed it with `wm_paint`.

Exact tests: export then import with no change gives an answer with no
voxels; one painted voxel gives one voxel at the right world position.

### Part F: the API for phase 4.2

Write a short API note in the PR body of PR 3, and send it to the
coordinator. Phase 4.2 calls these functions and writes nothing else:

- read the queue for a hemisphere, ranked and random;
- start a session (random share, seed) and record each site shown or
  skipped;
- run a choice and its preview at a site;
- write a site answer into the edits diff, and record it in the log;
- read the log.

## The hemisphere runs

Read the outputs in `scratch/v2/derivatives/`. Write to
`scratch/phase-1.3/`. `dseg` on the hemisphere peaks at about 22 GB. It is a
heavy job: ask the coordinator for a slot.

1. **Part A:** the site finder on the hemisphere. Counts by site kind, and
   the time. Compare with #81's 91 sites.
2. **Part B:** `dseg` with the fix table and the diff. Counts by kind, the
   size of the diff file, a replay that reproduces `desc-wmtopo`, the time
   and the peak.
3. **Part C:** a test edits diff of three edits at real sites (one of each
   choice that changes voxels), applied by `dseg`. Report what changed. Do
   not publish these edits as real edits. Name the file as a test.

## Work breakdown

| Step | Work | Who | Can run with |
|---|---|---|---|
| 1 | Part A: port #81, random sites | Sonnet sub-agent | 2, 3 |
| 2 | Part C: the spec draft for the log (send to the coordinator) | Phase session | 1, 3 |
| 3 | Part D: the choice functions, without the preview | Sonnet sub-agent | 1, 2 |
| 4 | Part B: fix ids, the table and the diff | Phase session | after 1.0.5's PR opens |
| 5 | Part C: the log code and the edits diff, the override rule | Phase session | after 2 and 4 |
| 6 | Part D: the preview | Phase session | after 3 and 5 |
| 7 | Part E: the ITK-SNAP bootstrap command | Haiku sub-agent, the phase session reviews | after 5 |
| 8 | The hemisphere runs | Phase session | after each part |

## PRs

1. `feature/wm-sites-finder`: Part A. Branch from `origin/main`.
2. `feature/dseg-fix-diff`: Part B. Stacked on 1.0.5's PR
   (`feature/dseg-small-pieces`).
3. `feature/wm-edits-log`: Part C and the queue's fix sites. Stacked on PR 2;
   merge PR 1's branch into it.
4. `feature/wm-choices`: Parts D and E. Stacked on PR 3.

Tell the coordinator when each PR opens. Phase 4.2 stacks on PR 4.

## Decisions for review (made by the coordinator)

- Copy from #80 and #81, not stack on them (the user's decision).
- The fix kind 5 is `small_piece` (shared with phase 1.0.5).
- The choices' default radius is 1 voxel. `thicken` offers 1 and 2.
- Scribbles are training labels only. `dseg` does not apply them. This is
  the plan's rule (§2). The plan leaves open whether a scribble also
  corrects the current hemisphere (§3c, Open). Phase 1.3 does not decide
  it.
