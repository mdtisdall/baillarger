#!/usr/bin/env bash
# PreToolUse(Bash) hook: stop Claude Code from writing to the default branch,
# and from making worktrees outside the project's .worktrees/ directory.
#
# Blocks:
#   - `git commit` when the target checkout is on the default branch;
#   - `git push` that would update the default branch: a push from the default
#     branch with no explicit branch, or a refspec that names it (`main`,
#     `HEAD:main`, `--delete main`, `--all`, `--mirror`);
#   - `git worktree add` or `git worktree move` to a path that is not directly
#     in `<main checkout>/.worktrees/`: a path outside the project directory,
#     or a path inside another worktree.
#
# The target checkout is the payload's "cwd", not $CLAUDE_PROJECT_DIR (which
# is always the main checkout, also in a worktree session). A leading
# `cd <dir> &&` or `git -C <dir>` in the command overrides it.
#
# Every change gets to the default branch through a reviewed PR, and each task
# has its worktree in .worktrees/<short-name> (see CLAUDE.md). This hook is a
# guard rail, not a lock: for a genuine exception, run the git command yourself
# in a terminal. Installed by the dev-workflow plugin. Keep it bash 3.2
# compatible (macOS /bin/bash).
set -euo pipefail

input="$(cat)"

# Fast exit for commands that cannot be a git commit, push, or worktree command.
case "$input" in
  *git*commit* | *git*push* | *git*worktree*) ;;
  *) exit 0 ;;
esac

# Without jq: a JSON string field on one line. The value stops at its first
# escaped quote, which is enough to find the git subcommand.
json_field_fallback() {
  printf '%s' "$input" \
    | grep -o "\"$1\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" \
    | head -n 1 \
    | sed -E "s/^\"$1\"[[:space:]]*:[[:space:]]*\"(.*)\"\$/\1/" \
    || true
}

if command -v jq >/dev/null 2>&1 && [ -z "${DEV_WORKFLOW_HOOK_NO_JQ:-}" ]; then
  command_field="$(printf '%s' "$input" | jq -r '.tool_input.command // ""' 2>/dev/null || true)"
  cwd_field="$(printf '%s' "$input" | jq -r '.cwd // ""' 2>/dev/null || true)"
else
  command_field="$(json_field_fallback command)"
  cwd_field="$(json_field_fallback cwd)"
fi

dir="${cwd_field:-${CLAUDE_PROJECT_DIR:-$PWD}}"

strip_quotes() {
  local v="$1"
  v="${v%\"}"
  v="${v#\"}"
  v="${v%\'}"
  v="${v#\'}"
  printf '%s' "$v"
}

# Resolve a path relative to $dir.
resolve_dir() {
  # shellcheck disable=SC2088 # the patterns match a literal ~ in the command text
  case "$1" in
    /*) printf '%s' "$1" ;;
    "~") printf '%s' "$HOME" ;;
    "~/"*) printf '%s/%s' "$HOME" "${1#\~/}" ;;
    *) printf '%s/%s' "$dir" "$1" ;;
  esac
}

# Remove `.`, `..`, and repeated `/` from an absolute path, without the file
# system.
normalize_path() {
  local out="" part rest="$1/"
  while [ -n "$rest" ]; do
    part="${rest%%/*}"
    rest="${rest#*/}"
    case "$part" in
      "" | .) ;;
      ..) out="${out%/*}" ;;
      *) out="$out/$part" ;;
    esac
  done
  printf '%s' "${out:-/}"
}

# Physical path (symbolic links resolved) of an absolute path whose last
# components do not have to exist yet.
physical_path() {
  local p="$1" tail=""
  while [ ! -d "$p" ]; do
    tail="/${p##*/}$tail"
    p="${p%/*}"
    [ -n "$p" ] || p=/
  done
  normalize_path "$(cd "$p" && pwd -P)$tail"
}

path_re='("[^"]*"|'\''[^'\'']*'\''|[^[:space:];&|]+)'

cd_re='^[[:space:]]*cd[[:space:]]+'"$path_re"'[[:space:]]*(&&|;)'
if [[ "$command_field" =~ $cd_re ]]; then
  dir="$(resolve_dir "$(strip_quotes "${BASH_REMATCH[1]}")")"
fi

gitc_re='git[[:space:]]+-C[[:space:]]+'"$path_re"
if [[ "$command_field" =~ $gitc_re ]]; then
  dir="$(resolve_dir "$(strip_quotes "${BASH_REMATCH[1]}")")"
fi

# git [global options] commit|push|worktree add|worktree move
opts_re='([[:space:]]+(-C[[:space:]]+'"$path_re"'|-c[[:space:]]+[^[:space:]]+|--[a-z-]+(=[^[:space:]]*)?))*'
commit_re='(^|[;&|({[:space:]])git'"$opts_re"'[[:space:]]+commit([[:space:]]|$)'
push_re='(^|[;&|({[:space:]])git'"$opts_re"'[[:space:]]+push([[:space:]].*)?$'
worktree_re='(^|[;&|({[:space:]])git'"$opts_re"'[[:space:]]+worktree[[:space:]]+(add|move)([[:space:]].*)?$'

is_commit=0
is_push=0
is_worktree=0
push_args=""
wt_sub=""
wt_args=""
if [[ "$command_field" =~ $commit_re ]]; then
  is_commit=1
fi
if [[ "$command_field" =~ $push_re ]]; then
  is_push=1
  push_args="${BASH_REMATCH[${#BASH_REMATCH[@]} - 1]}"
fi
if [[ "$command_field" =~ $worktree_re ]]; then
  is_worktree=1
  wt_sub="${BASH_REMATCH[${#BASH_REMATCH[@]} - 2]}"
  wt_args="${BASH_REMATCH[${#BASH_REMATCH[@]} - 1]}"
fi
[ "$is_commit" -eq 1 ] || [ "$is_push" -eq 1 ] || [ "$is_worktree" -eq 1 ] || exit 0

git -C "$dir" rev-parse --git-dir >/dev/null 2>&1 || exit 0
dir="$(cd "$dir" && pwd -P)"
branch="$(git -C "$dir" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
default="$(git -C "$dir" symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null || true)"
default="${default#origin/}"
default="${default:-main}"
main="$(git -C "$dir" worktree list --porcelain 2>/dev/null | sed -n '1s/^worktree //p')"
main="$(physical_path "${main:-$dir}")"

block() {
  cat >&2 <<EOF
Blocked: $1

Every change gets to \`$default\` through a reviewed pull request, so Claude
Code does not commit to \`$default\` or push to it.

  Start a branch:  git -C $main worktree add -b <type>/<name> .worktrees/<name> origin/$default
                   (or use the dev-workflow start-task skill)
  Commit there, push the branch, and open a PR against $default.

See CLAUDE.md ("Branch and PR workflow"). For a deliberate exception, run the
git command yourself in a terminal.
EOF
  exit 2
}

block_worktree() {
  cat >&2 <<EOF
Blocked: \`git worktree $wt_sub\` to $1.

Each worktree of this project is directly in $main/.worktrees/:
not outside the project directory, and not inside another worktree.

  Start a branch:  git -C $main worktree add -b <type>/<name> .worktrees/<name> origin/$default
                   (or use the dev-workflow start-task skill)

See CLAUDE.md ("Branch and PR workflow"). For a deliberate exception, run the
git command yourself in a terminal.
EOF
  exit 2
}

if [ "$is_worktree" -eq 1 ]; then
  # Only the words of this worktree command: stop at the next command separator.
  wt_args="${wt_args%%[;&|]*}"
  wt_args="${wt_args%%$'\n'*}"
  # `add [options] <path> [<commit>]`, `move <worktree> <new-path>`
  want=1
  [ "$wt_sub" != move ] || want=2
  wt_path=""
  positional=0
  skip=0
  set -f
  for word in $wt_args; do
    if [ "$skip" -eq 1 ]; then
      skip=0
      continue
    fi
    word="$(strip_quotes "$word")"
    case "$word" in
      -b | -B | --reason) skip=1 ;;
      -*) ;;
      *)
        positional=$((positional + 1))
        if [ "$positional" -eq "$want" ]; then wt_path="$word"; fi
        ;;
    esac
  done
  set +f
  if [ -n "$wt_path" ]; then
    target="$(physical_path "$(normalize_path "$(resolve_dir "$wt_path")")")"
    case "$target" in
      "$main/.worktrees/"*)
        case "${target#"$main/.worktrees/"}" in
          "" | */*) block_worktree "$target" ;;
        esac
        ;;
      *) block_worktree "$target" ;;
    esac
  fi
fi

if [ "$is_commit" -eq 1 ] && [ "$branch" = "$default" ]; then
  block "\`git commit\` on \`$default\` in $dir."
fi

if [ "$is_push" -eq 1 ]; then
  # Only the words of this push command: stop at the next command separator.
  push_args="${push_args%%[;&|]*}"
  push_args="${push_args%%$'\n'*}"
  targets_default=0
  positional=0
  set -f
  for word in $push_args; do
    word="$(strip_quotes "$word")"
    case "$word" in
      --all | --mirror) targets_default=1 ;;
      -*) ;;
      *)
        positional=$((positional + 1))
        if [ "$positional" -ge 2 ]; then
          case "$word" in
            "$default" | "+$default" | *":$default" | "refs/heads/$default" | *":refs/heads/$default")
              targets_default=1
              ;;
            HEAD | +HEAD)
              if [ "$branch" = "$default" ]; then targets_default=1; fi
              ;;
          esac
        fi
        ;;
    esac
  done
  set +f
  # `git push` or `git push <remote>` pushes the current branch.
  if [ "$positional" -le 1 ] && [ "$branch" = "$default" ]; then
    targets_default=1
  fi
  if [ "$targets_default" -eq 1 ]; then
    block "\`git push\` that would update \`$default\`."
  fi
fi

exit 0
