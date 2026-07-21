#!/usr/bin/env bash
set -euo pipefail

ROBOREV_BIN="${ML4T_ROBOREV_BIN:-roborev}"
GIT_BIN="${ML4T_GIT_BIN:-git}"
BASE_BRANCH="${ML4T_ROBOREV_BASE_BRANCH:-main}"

if ! command -v "$ROBOREV_BIN" >/dev/null 2>&1; then
    printf 'RoboRev executable not found: %s\n' "$ROBOREV_BIN" >&2
    exit 1
fi

repo_root=$("$GIT_BIN" rev-parse --show-toplevel)
cd "$repo_root"
branch=$("$GIT_BIN" branch --show-current)

if [[ -z "$branch" ]]; then
    printf 'RoboRev PR gate requires a named branch.\n' >&2
    exit 1
fi
if [[ "$branch" == "$BASE_BRANCH" ]]; then
    printf 'Direct pushes from %s are not allowed; use a PR branch.\n' "$BASE_BRANCH" >&2
    exit 1
fi

printf 'Running RoboRev branch review for %s against %s...\n' "$branch" "$BASE_BRANCH"
"$ROBOREV_BIN" review \
    --branch \
    --base "$BASE_BRANCH" \
    --agent codex \
    --panel none \
    --min-severity low \
    --wait

open_reviews=$("$ROBOREV_BIN" fix --open --list --branch "$branch")
if grep -q '^Job #[0-9]' <<< "$open_reviews"; then
    printf '%s\n' "$open_reviews" >&2
    printf 'Push blocked: resolve every open RoboRev finding first.\n' >&2
    exit 1
fi

printf 'RoboRev PR gate passed: no open findings on %s.\n' "$branch"
