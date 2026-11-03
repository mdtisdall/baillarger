# Round 1 of the version-2 plan

This file gives the rules for the first round of work on
`docs/version-2-plan.md`. Each phase in round 1 has its own plan in this
directory.

## The phases in round 1

| Phase | Plan | Needs |
|---|---|---|
| 1.0: the pipeline on the purple-mri aseg | `phase-1.0-pipeline.md` | nothing |
| 3.2: a QPBO that is not GPL | `phase-3.2-qpbo.md` | nothing |
| Shared: the `mesh_graph` duplicate-edge fix | `mesh-graph-fix.md` | nothing |
| 1.5: the label set | `phase-1.5-label-set.md` | nothing (one stacked PR on 1.0) |
| 4.0: the viewer on stream 1's outputs | `phase-4.0-viewer.md` | 1.0's split PR, the `mesh_graph` fix, 1.0's hemisphere outputs (each for one step only) |
| 2.0: the ray sampler and the chunked reader | `phase-2.0-rays.md` | 1.0's outputs for the hemisphere run only |
| 3.0: the column-volume sampler | `phase-3.0-column-volume.md` | holds until 2.0's spec PR is open; stacks on it |

The first four start at the same time. The user added 4.0, 2.0 and 3.0 later. Each runs in its own session and its own
worktree.

## Rules for every phase session

### Writing

1. Write plans, PR bodies, specs, docstrings and messages to the coordinator
   in the style of ASD-STE100. Use the `asd-ste100` skill.
2. Use one word for one thing in a document.

### Git and PRs

1. Make your worktree with the `start-task` skill. Use the branch name that
   your plan gives.
2. Run `nix develop --command uv sync` one time in the worktree, before any
   sub-agent starts.
3. Commit, push and open a PR with the `ship` skill when a piece of work is
   ready. Do not wait for approval.
4. Do not merge a PR. The user reviews and merges.
5. You can stack PRs. If a piece of work needs a PR that is not merged, make
   a new branch from that PR's branch. Set the base of the new PR to that
   branch. Write "Stacked on #N" in the first line of the PR body.
6. When a PR below yours changes, merge its branch into yours. Do not rebase
   a branch that is pushed.

### Sub-agents

1. Use the `parallel-agents` skill.
2. Give each sub-agent different files. Sub-agents do not run git commands
   that write.
3. Review every diff from a sub-agent before you commit it.
4. Your plan gives a model for each step. Use a stronger model if the step
   is harder than the plan expects.

### The plan document

1. Do not edit `docs/version-2-plan.md`.
2. If you find that the plan is wrong or out of date, send the finding to the
   coordinator. The coordinator keeps one list of changes and applies them
   in one PR at the end of round 1.

### Heavy runs

1. Run each heavy job under `scripts/memguard`. Set the cap to about 1.5 × the
   peak memory of the job.
2. Only one heavy job can run on this machine at a time, across all
   sessions. A heavy job is a job with a peak above 8 GB.
3. Before a heavy job starts, send a message to the coordinator. Wait for
   the coordinator to reply "go". Send a second message when the job ends.

### Published outputs

1. Phase 1.0 writes the hemisphere outputs to
   `scratch/v2/derivatives/` in the main checkout. Each worktree has a link
   to `scratch/`.
2. Later phases read the outputs from that directory. No code or test names
   `scratch/`. Give the path as an argument.

### Questions and coordination

1. Send questions to the coordinator session with `SendMessage`. Do not send
   them to the user.
2. Ask before you change a spec in `docs/formats/`, a public API that
   another phase uses, or a design decision in the plan.
3. If two phases must change the same file, ask the coordinator first.
4. Do not stop work to wait for an answer if other work in your plan can
   continue.

### When a phase is done

1. Every PR of the phase is open, and its CI is green.
2. Send the coordinator a report. Give the PR numbers, the findings that
   change the plan, and the open questions for the user.
