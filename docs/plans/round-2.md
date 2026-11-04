# Round 2 of the version-2 plan

This file gives the rules for wave 2 of `docs/version-2-plan.md` (§9,
*Waves*). Each phase in wave 2 has its own plan in this directory. The rules
of `round-1.md` do not apply. This file replaces them.

## The phases in wave 2

| Phase | Plan | Needs | Session |
|---|---|---|---|
| 1.0.5: small pieces | `phase-1.0.5-small-pieces.md` | 1.0 (done) | A |
| 1.1: the WM-distance sampler | `phase-1.1-wm-distance.md` | 1.0 (done) | B |
| 1.2: `orient` | `phase-1.2-orient.md` | 1.1 (stacks on its PR) | B |
| 1.3: the review's formats and code | `phase-1.3-review.md` | 1.0 (done) | C |
| 2.1: the equivolume cross-section | `phase-2.1-equivolume.md` | 2.0 (done) | D |
| 3.1: flattening | `phase-3.1-flattening.md` | 3.0 (done) | E |
| 4.0.5: overlays on the brick pane | `phase-4.0.5-overlays.md` | 4.0 (done) | F |
| 4.2: the stream 1 review pane | `phase-4.2-review-pane.md` | 4.0.5 and 1.3 (stacks on both) | F |
| 4.3: columns in the viewer | `phase-4.3-columns.md` | 2.0 and 4.0 (done) | G |

Each letter is one session with one worktree for each branch. Session B does
1.1, then 1.2. Session F does 4.0.5, then 4.2. The other sessions do one
phase each.

Wave 2 ends with the first review session on the hemisphere. The user does
that review. The sessions stop when 4.2's pane works on the hemisphere.

## Rules for every phase session

### Writing

1. Write plans, PR bodies, specs, docstrings and messages to the coordinator
   in the style of ASD-STE100. Use the `asd-ste100` skill.
2. Use one word for one thing in a document.

### Git and PRs

1. Make each worktree with the `start-task` skill. Use the branch names that
   your plan gives.
2. Run `nix develop --command uv sync` one time in each new worktree, before
   any sub-agent starts.
3. Commit, push and open a PR with the `ship` skill when a piece of work is
   ready. Do not wait for approval. Show the commit message and the PR body
   in your report.
4. Do not merge a PR. The user reviews and merges.
5. You can stack PRs. If a piece of work needs a PR that is not merged, make
   a new branch from that PR's branch. Set the base of the new PR to that
   branch. Write "Stacked on #N" in the first line of the PR body.
6. If a piece of work needs two PRs that are not merged, make the branch
   from the first PR's branch, then merge the second PR's branch into it.
   Set the base to the first PR's branch. Write "Stacked on #N; contains
   #M" in the first line of the PR body.
7. When a PR below yours changes, merge its branch into yours. Do not rebase
   a branch that is pushed.
8. Use the Actions API for CI status (`ci-status.sh`, `gh run list`), not
   `gh pr checks`.

### Sub-agents

1. Use the `parallel-agents` skill.
2. Give each sub-agent different files. Sub-agents do not run git commands
   that write.
3. Review every diff from a sub-agent before you commit it.
4. Your plan gives a model for each step. Use a stronger model if the step
   is harder than the plan expects. Do not use a weaker model than the plan
   gives.

### The plan document

1. Do not edit `docs/version-2-plan.md` or the files in `docs/plans/`.
2. If you find that the plan is wrong or out of date, send the finding to the
   coordinator. The coordinator keeps one list of changes and applies them
   in one PR at the end of wave 2.

### Heavy runs

1. Run each heavy job under `scripts/memguard`. Set the cap to about 1.5 × the
   peak memory of the job.
2. This machine has 64 GB and 10 cores. Only one heavy job can run on it at a
   time, across all sessions. A heavy job is a job with a peak above 8 GB, or
   a job that uses more than 4 cores for more than 10 minutes.
3. Before a heavy job starts, send a message to the coordinator. Give the
   expected peak and time. Wait for the coordinator to reply "go". Send a
   second message when the job ends, with the measured peak and time.
4. If you do not know the peak, measure it on a small part first (one
   superblock, or a box).

### Published outputs

1. The hemisphere outputs of phase 1.0 are in `scratch/v2/derivatives/` in
   the main checkout. Each worktree has a link to `scratch/`. All phases in
   wave 2 read these outputs.
2. Phase 1.0.5 writes its new outputs to `scratch/v2/derivatives-1.0.5/`. It
   does not change `scratch/v2/derivatives/`. Other phases do not read the
   new outputs in wave 2. The user decides when they replace the old ones.
3. Write your own outputs, views and figures to `scratch/<phase>/`, for
   example `scratch/phase-2.1/`.
4. No code or test names `scratch/`. Give each path as an argument.
5. Figures that go into a report or a PR body go to
   `reports/figures/<slug>/` as PNG files. Keep each file well under 1 MB.

### Questions and decisions

1. Send questions to the coordinator session with `SendMessage`. Do not send
   them to the user. The user is away until 07:00.
2. Ask before you change a spec in `docs/formats/`, a public API that
   another phase uses, or a design decision in the plan.
3. If two phases must change the same file, ask the coordinator first.
4. Do not stop work to wait for an answer if other work in your plan can
   continue.
5. The coordinator makes a provisional decision where the data cannot
   decide. Each PR body that depends on such a decision has a section
   "Decisions for review". That section gives each decision, its reason and
   what changes if the user reverses it.
6. Where the plan asks for a choice by diagnostics (for example a σ), make
   the views, make a provisional choice, and give the views and the choice
   in the PR body. Do not tune a constant to hide a symptom.

### When a phase is done

1. Every PR of the phase is open, and its CI is green.
2. Send the coordinator a report. Give the PR numbers, the findings that
   change the plan, the decisions for review, and the open questions for the
   user.
