# Phase 4.2: the stream 1 review pane

Stream 4, phase 2 of `docs/version-2-plan.md` (§6d). Read the plan's §1
(*Data for training, and for judging success*, items 3 and 4), §3c step 6,
and §6. The rules in `docs/plans/round-2.md` apply. The session that did
phase 4.0.5 does this phase.

## Goal

A pane in the viewer where the user answers stream 1's review queue on the
hemisphere: site answers with previews, and scribbles, written to
`wm-edits/1`. The pane holds no stream 1 logic. It calls phase 1.3's
published code. Wave 2 ends when the user can run the first review session
with it.

## Done when

1. The pane runs on the hemisphere in `--white` mode with the brick pane
   and its overlays (4.0.5), and a test session writes a valid edit log.
2. Every value in §1 item 4 that the pane logs is in the log: each answer's
   reviewer time and how its site was chosen; each session's random share
   and every site shown, skipped ones included.
3. The exact tests below pass.
4. The PR body has screenshots of one site of each kind, a preview, and a
   scribble, and the test log's content. Use the built-in browser pane.
5. `nix develop --command scripts/check` passes.

## Inputs from other phases

- **Phase 1.3** (session C): the queue (`wm-sites/1`, ranked and random),
  the session records, the choices and their preview, and the append-only
  log. Its PR 4 (`feature/wm-choices`) is the last one. Its PR 3 body has
  the API note (Part F). Do not start the server routes until the
  coordinator sends you that API.
- **Phase 4.0.5** (your own): the named volumes, the label overlay and the
  outlines.

## Facts from the current code

See `phase-4.0.5-overlays.md`, "Facts from the current code". Also:

- `viewer/labels.py` (170 lines, `LabelReview`) is the old pial-label
  review: a list (`/labels/list`), `POST /labels/decide`, `POST
  /labels/brush`, and a `labels_seq` bump through the shared state. It is
  the nearest pattern for a review pane. Its storage (an `.npz` changed in
  place) is not the pattern: `wm-edits/1` is append-only. Do not change
  `labels.py`. Phase 4.3 owns it.
- `Shared.update` (`server.py` 84–113) takes any keys. The shared state is
  in memory only and is not saved.
- `Handler.lock` serialises the site computations. A writer of the edit
  log needs its own lock.

## Shared files with other phases

`server.py` and `index.html` are shared with phase 4.3. Put the pane in
`viewer/review.py` and `static/review.js`. In `server.py` and `index.html`,
add only the hooks. Ask the coordinator before you change more.

## Design

### Part A: the server side (`viewer/review.py`)

1. New arguments for `--white` mode: `--sites` (the queue), `--log` (the
   log) and `--edits` (the label edits, `label-edits/1`, which the answers
   write into). The paths are given. The pane never chooses a default path
   for the real log or the edits.
2. Routes, under `/review/`:
   - `GET /review/queue`: the ranked sites, with kind, rank and score.
   - `POST /review/session`: start a session with a random share (0 to 1,
     no default: the page asks for it) and a seed. Phase 1.3 draws the
     random sites.
   - `GET /review/next`: the next site of the session. The server records
     it as shown.
   - `POST /review/skip`: record a skip.
   - `POST /review/preview`: a choice and its parameters at the site.
     Returns the voxels the answer changes, the voxels the correction then
     changes, and whether the correction overrides the answer.
   - `POST /review/answer`: a site answer, with the reviewer time.
   - `POST /review/scribble`: strokes, each with one class from the label
     set, with the reviewer time.
3. Each write goes through phase 1.3's API, under a lock, and is flushed
   before the response.
4. A preview runs on a box round the site. If it takes more than about 2 s
   on the hemisphere, run it as a job, with the filter-job pattern of
   `viewer/filters.py` (one job, status polled). Do not import
   `filters.py`. Copy the pattern.
5. No response scales with the whole mesh or volume.

### Part B: the page (`static/review.js`)

1. A queue list: site kind, rank, score, and the state (not shown, shown,
   answered, skipped). A start-session dialog asks for the random share.
2. **Opening a site** moves the crosshair to it (through the shared state)
   and starts the reviewer time.
3. **The choices:** `remove-strand`, `thicken` (1 or 2), `fill-gap`,
   `widen-gap`, `leave`, `manual-paint`. Each choice shows its preview on
   the brick tiles: the answer's voxels in one colour, the correction's
   voxels in another, and a warning if the correction overrides the answer.
   `manual-paint` opens the paint mode (`docs/plans/label-editing.md`,
   Part 2); its strokes are the answer.
4. **Submit** writes the answer with the reviewer time: from opening the
   site to submitting, in seconds. The next site opens.
5. **Scribbles:** a key picks the class (WM or cortex by default; the other
   label-set classes from a list). A stroke paints the voxels under the
   cursor on the current tile, with a brush radius in voxels. A stroke has
   one class. Submit writes the scribble.
6. A random site shows that it is random only after the answer, so that the
   kind does not change how the reviewer looks. **Coordinator decision for
   review.** If the coordinator or the user decides otherwise, show it at
   once.
7. Keys for each choice, for submit and for skip. List them in the pane.

### Part C: what the pane draws

Use 4.0.5's label overlay and outlines. The preview voxels are a sparse
list. Draw them on `#bvo` as squares per tile, not as a volume. Give the
largest preview size that the canvas draws at 60 fps.

## Exact tests

- The server routes, on a phantom with a small queue and a temporary log
  (follow `tests/test_viewer_bricks.py`, `test_the_server_sends_the_bricks`,
  which starts a real server):
  - a session records its random share, its seed and each site shown;
  - a skip is recorded and the site stays unanswered;
  - an answer appends one record with its reviewer time and how its site
    was chosen;
  - a scribble appends one record with its strokes and classes;
  - two answers in order: the log keeps both, in order;
  - the log that the pane writes passes phase 1.3's reader.
- A preview through the route equals phase 1.3's preview function on the
  same input.

## The hemisphere check

1. Run the site finder of phase 1.3 on `scratch/v2/derivatives/` (or use
   its output in `scratch/phase-1.3/`). Start the viewer with `--sites`,
   `--log scratch/phase-4.2/test-edits/...` and `--edits` a test diff in the
   same directory. Never write the test log or the test edits next to the
   real derivatives.
2. In the built-in browser pane: start a session with a random share of
   0.2; open three sites of different kinds; preview each choice; answer
   two, skip one; make one scribble.
3. Read the test log back with phase 1.3's reader. Give its records in the
   PR body.
4. Replay the test log with `dseg` (phase 1.3) on the hemisphere, as a
   check that the answers apply. This is a heavy job (about 22 GB). Ask the
   coordinator for a slot.
5. Give the steps the user follows to start the first real review session:
   the command, the paths, and where the real log goes. Ask the coordinator
   for the path of the real log.

## Work breakdown

| Step | Work | Who | Can run with |
|---|---|---|---|
| 1 | The branch: from 1.3's PR 4 branch, merge 4.0.5's branch | Phase session | — |
| 2 | Part A: routes and the lock | Sonnet sub-agent, after the API arrives | 3 |
| 3 | Part B: the queue list and the session dialog | Sonnet sub-agent | 2 |
| 4 | Part B and C: choices, previews, scribbles, timing | Phase session | after 2 and 3 |
| 5 | The tests | Sonnet sub-agent | after 2 |
| 6 | The hemisphere check | Phase session | after 4 and 5 |

## PRs

1. `feature/review-pane`. Branch from 1.3's `feature/wm-choices`, then
   merge `feature/brick-overlays` (4.0.5) into it. The base is
   `feature/wm-choices`. The first line of the body: "Stacked on #N;
   contains #M".
2. When either branch below changes, merge it into yours. Do not rebase.

Tell the coordinator when the PR opens.

## Decisions for review (made by the coordinator)

- The pane shows that a site is random only after the answer.
- The log path is always an argument. The pane has no default path for the
  real log.
