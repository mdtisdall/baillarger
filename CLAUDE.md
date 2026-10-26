# baillarger

Python 3.12 project. Toolchain from the Nix devShell (`flake.nix`, loaded by direnv): uv for dependencies, ruff for lint and format, mypy (strict) for types, pytest for tests. Source is in `src/baillarger/`, tests in `tests/`.

## Branch and PR workflow

Every change to a tracked file gets to `main` through its own
branch and a reviewed pull request. There are no exceptions for documentation
or process changes: `CLAUDE.md`, `README.md`, `.envrc`, `flake.nix`,
`.gitignore`, CI configuration, and hooks also go through a branch and a PR.

One branch, one concern. Do not add an unrelated fix, chore, or doc edit to a
branch that already has work in progress. Start a new branch from
`main` for it.

The dev-workflow Claude Code plugin has the procedures: `start-task`, `ship`,
`finish-task`, `parallel-agents`, `github-token`, `project-setup`. Specific to
this project:

- Default branch: `main`. Merge method: squash.
- Checks, before each PR (CI runs the same checks):
  `nix develop --command scripts/check`. If `scripts/check` accepts
  `--changed-since`, PRs run
  `nix develop --command scripts/check --changed-since origin/main`
  (only the tests that the change can affect), and `main` runs all
  the tests after each merge.
- Worktrees: one for each task, in `.worktrees/<short-name>` of the main
  checkout (git-ignored), made with the `start-task` skill. Do not create a
  worktree outside the project directory, inside another worktree, or with a
  different tool. The hook blocks `git worktree add` and `git worktree move`
  to any other path.
- Dependency sync, one time in each new worktree, before sub-agents start:
  `nix develop --command uv sync`
- GitHub CLI: `gh ...`. In Claude Code's Bash commands, the
  `.claude/hooks/devshell-gh.sh` hook makes `gh` the devShell's, with this
  repository's own fine-grained token from the git-ignored `.envrc.local`.
  Elsewhere, and if the hook reports a problem: `direnv exec . gh ...`.
  The permissions that the token needs are in `.claude/gh-token-permissions`. `git` uses SSH.
- CI status: the Actions API (`ci-status.sh`, `gh run list`), not
  `gh pr checks`.
- `git commit` and `git push` to `main` are blocked for Claude
  Code by `.claude/hooks/block-main-writes.sh`. To run the git command
  yourself in a terminal is a break-glass action for a stuck state, not a
  shortcut for routine edits.
- Commit, push and open the PR without waiting for approval. Show the
  commit message and the PR body in the reply. To merge a PR to `main`
  needs an explicit instruction from the user that names the PR.
