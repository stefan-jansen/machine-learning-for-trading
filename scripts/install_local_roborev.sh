#!/usr/bin/env bash
set -euo pipefail

ROBOREV_BIN="${ML4T_ROBOREV_BIN:-roborev}"
GIT_BIN="${ML4T_GIT_BIN:-git}"

if ! command -v "$ROBOREV_BIN" >/dev/null 2>&1; then
    printf 'RoboRev executable not found: %s\n' "$ROBOREV_BIN" >&2
    exit 1
fi

repo_root=$("$GIT_BIN" rev-parse --show-toplevel)
gate="$repo_root/scripts/roborev_pr_gate.sh"
hook_path=$("$GIT_BIN" rev-parse --git-path hooks/pre-push)

if [[ ! -x "$gate" ]]; then
    printf 'RoboRev gate is missing or not executable: %s\n' "$gate" >&2
    exit 1
fi

if [[ -e "$hook_path" ]] && ! grep -q '^# ml4t roborev pre-push hook v1$' "$hook_path"; then
    printf 'Refusing to replace an existing pre-push hook: %s\n' "$hook_path" >&2
    exit 1
fi

"$ROBOREV_BIN" init --agent codex --no-daemon

hook_dir=$(dirname "$hook_path")
mkdir -p "$hook_dir"
tmp_hook=$(mktemp "$hook_dir/pre-push.XXXXXX")
trap 'rm -f "$tmp_hook"' EXIT

printf '%s\n' \
    '#!/usr/bin/env bash' \
    '# ml4t roborev pre-push hook v1' \
    'set -euo pipefail' \
    'repo_root=$(git rev-parse --show-toplevel)' \
    'exec "$repo_root/scripts/roborev_pr_gate.sh" "$@"' > "$tmp_hook"
chmod +x "$tmp_hook"
mv "$tmp_hook" "$hook_path"
trap - EXIT

printf 'Installed local RoboRev hooks for %s\n' "$repo_root"
