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

declare -a updates=()
while read -r local_ref local_sha remote_ref remote_sha; do
    [[ -z "${local_ref:-}" ]] && continue
    updates+=("$local_ref $local_sha $remote_ref $remote_sha")
done

if (( ${#updates[@]} == 0 )); then
    branch=$("$GIT_BIN" branch --show-current)
    if [[ -z "$branch" ]]; then
        printf 'RoboRev PR gate requires a named branch.\n' >&2
        exit 1
    fi
    updates+=("refs/heads/$branch $("$GIT_BIN" rev-parse "refs/heads/$branch") refs/heads/$branch 0")
fi

declare -a branches=()
declare -A seen=()
for update in "${updates[@]}"; do
    read -r local_ref local_sha remote_ref remote_sha <<< "$update"
    if [[ "$remote_ref" == "refs/heads/$BASE_BRANCH" ]]; then
        printf 'Direct pushes to %s are not allowed; use a PR branch.\n' "$BASE_BRANCH" >&2
        exit 1
    fi
    if [[ "$local_sha" =~ ^0+$ || "$remote_ref" == refs/tags/* ]]; then
        continue
    fi
    if [[ "$remote_ref" != refs/heads/* ]]; then
        printf 'Unsupported pre-push ref update: %s -> %s\n' "$local_ref" "$remote_ref" >&2
        exit 1
    fi

    if [[ "$local_ref" == refs/heads/* ]]; then
        branch=${local_ref#refs/heads/}
    elif [[ "$local_ref" == "HEAD" ]]; then
        branch=$("$GIT_BIN" symbolic-ref --quiet --short HEAD) || {
            printf 'Cannot review HEAD from a detached checkout.\n' >&2
            exit 1
        }
    else
        printf 'Unsupported pre-push source ref: %s\n' "$local_ref" >&2
        exit 1
    fi

    current_sha=$("$GIT_BIN" rev-parse "$local_ref")
    if [[ "$current_sha" != "$local_sha" ]]; then
        printf 'Local ref changed before review: %s\n' "$local_ref" >&2
        exit 1
    fi
    if [[ -z "${seen[$branch]:-}" ]]; then
        branches+=("$branch")
        seen[$branch]=1
    fi
done

if (( ${#branches[@]} == 0 )); then
    printf 'RoboRev PR gate: no branch updates require review.\n'
    exit 0
fi

for branch in "${branches[@]}"; do
    printf 'Running RoboRev branch review for %s against %s...\n' "$branch" "$BASE_BRANCH"
    "$ROBOREV_BIN" review \
        --branch "$branch" \
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
done
