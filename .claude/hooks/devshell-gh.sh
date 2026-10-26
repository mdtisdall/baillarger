#!/usr/bin/env bash
# SessionStart and CwdChanged hook: make `gh` in Claude Code's Bash commands
# the devShell's gh, with this project's token.
#
# Without it, a plain `gh` finds the host's gh (or none) and its global login.
# The hook asks direnv for the devShell's gh, then writes two lines to
# $CLAUDE_ENV_FILE, which Claude Code runs before each Bash command:
#
#   - the directory of that gh, first on PATH. Only gh's directory: the other
#     commands keep the host tools (BSD tools on macOS);
#   - `. <project>/.envrc.local`, which sets GH_TOKEN. The token stays only in
#     that file (mode 600), not in Claude Code's environment file.
#
# Claude Code clears the file at each CwdChanged, so the hook runs again for
# the new directory. Outside a project with an .envrc, it writes nothing.
# `direnv exec . gh ...` keeps working, with or without this hook.
#
# It never blocks: a problem gives a notice, and gh stays the host's.
# Installed by the dev-workflow plugin. Keep it bash 3.2 compatible.
set -uo pipefail

input="$(cat)"

notice() {
  # A JSON string: the message has no quotes or backslashes.
  printf '{"systemMessage": "devshell-gh: %s"}\n' "$1"
  exit 0
}

[ -n "${CLAUDE_ENV_FILE:-}" ] || exit 0

field() {
  if command -v jq >/dev/null 2>&1 && [ -z "${DEV_WORKFLOW_HOOK_NO_JQ:-}" ]; then
    printf '%s' "$input" | jq -r --arg f "$1" '.[$f] // ""' 2>/dev/null || true
  else
    printf '%s' "$input" \
      | grep -o "\"$1\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" \
      | head -n 1 \
      | sed -E "s/^\"$1\"[[:space:]]*:[[:space:]]*\"(.*)\"\$/\1/" \
      || true
  fi
}

dir="$(field new_cwd)"
[ -n "$dir" ] || dir="$(field cwd)"
[ -n "$dir" ] || dir="${CLAUDE_PROJECT_DIR:-$PWD}"
[ -d "$dir" ] || exit 0

# The nearest directory with an .envrc: the main checkout or a worktree.
root="$(cd "$dir" && pwd -P)"
while [ ! -f "$root/.envrc" ]; do
  [ "$root" != / ] || exit 0
  root="$(dirname "$root")"
done

command -v direnv >/dev/null 2>&1 || notice "direnv is not on PATH; gh is the host's"
# In a worktree, .envrc.local is a link to the main checkout's file.
# `|| true`: a devShell without gh is not a direnv failure.
gh_path="$(cd "$root" && direnv exec "$root" sh -c 'command -v gh || true' 2>/dev/null)" ||
  notice "direnv exec failed in $root (run direnv allow there); gh is the host's"
case "$gh_path" in
  /*/gh) ;;
  *) notice "the devShell of $root has no gh; gh is the host's" ;;
esac
gh_dir="$(dirname "$gh_path")"

# The values go in single quotes: refuse a path that contains one.
case "$gh_dir$root" in
  *\'*) notice "a path contains a single quote; gh is the host's" ;;
esac

{
  printf "export PATH='%s':\"\$PATH\"\n" "$gh_dir"
  if [ -f "$root/.envrc.local" ]; then
    printf "if [ -r '%s/.envrc.local' ]; then . '%s/.envrc.local'; fi\n" "$root" "$root"
  fi
} >>"$CLAUDE_ENV_FILE"
exit 0
