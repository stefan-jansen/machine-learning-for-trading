#!/usr/bin/env bash
# Regenerate the CI fixture set in ml4t/third-edition-test-data.
#
# Run from the root of this repo (~/ml4t/public). The three steps are separable
# and each is individually re-runnable; the whole chain is rarely what you want.
# The usual trigger is a single dataset going red on a schema gate, in which case
# step 1 with --dataset plus step 2 for the affected case studies is enough.
#
#   Step 1  create_test_data.py          subsample production data -> data/
#   Step 2  generate_intermediates.py    run pipeline stages       -> intermediates/
#   Step 3  sample_registry_for_tests.py copy production registry rows into
#                                        intermediates/<cs>/run_log/registry.db
#
# Step 3 is not optional, and it was missing from both this script and the
# test-data README. Several late-stage notebooks (us_firm_characteristics/08a_ipca
# and 08_latent_factors, and the model_analysis and strategy_analysis stages
# across case studies) are replay-only: they refuse to fit and read registered
# results instead. Without step 3 they fail on a cache miss that reads like a
# code bug rather than a missing fixture.
#
# The previous version of this script referenced ~/ml4t/third-edition and
# ~/ml4t/technical_review (neither exists) and called scripts/deploy_to_review.py
# (which does not exist either). The generator now lives in the public repo and
# there is no separate review repo to deploy to.
#
# Usage:
#   bash tests/generate_test_data.sh [TEST_DATA_DIR]
#
# Env:
#   ML4T_SOURCE_DATA   production data root to subsample (default ~/ml4t/code/data)
#   STEPS              comma-separated subset of 1,2,3 (default all)
#   CASE_STUDIES       space-separated list for step 2 (default: all nine)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DATA_DIR="${1:-$HOME/ml4t/test-data}"
SOURCE_DATA="${ML4T_SOURCE_DATA:-$HOME/ml4t/code/data}"
STEPS="${STEPS:-1,2,3}"

# Only step 1 reads production data; steps 2 and 3 run off the fixture set and
# the case-study registries, so requiring it would block the common single-step
# re-run.
if [[ ",$STEPS," == *",1,"* && ! -d "$SOURCE_DATA" ]]; then
  echo "ERROR: production data root not found: $SOURCE_DATA" >&2
  echo "Set ML4T_SOURCE_DATA to the directory holding the full datasets." >&2
  exit 1
fi

echo "=== ML4T test-data regeneration ==="
echo "Repo:        $REPO_ROOT"
echo "Source data: $SOURCE_DATA"
echo "Output:      $TEST_DATA_DIR"
echo "Steps:       $STEPS"
echo ""

cd "$REPO_ROOT"

# MPLBACKEND=Agg is needed for headless execution.
#
# PLOTLY_RENDERER is not exported here, but note that it is not thereby unset:
# step 2 runs notebooks through run_notebook(), which sets PLOTLY_RENDERER=json
# itself. That is safe in this script because it writes only parquet
# intermediates -- the executed notebook goes to a gitignored _executed_*.ipynb
# and is discarded. The rule the JSON renderer breaks applies to re-executing a
# *committed* notebook: it emits a JSON blob instead of image/png, so the figures
# render as raw JSON on GitHub. Never use this script's step 2 to refresh a
# committed notebook.
export MPLBACKEND=Agg

if [[ ",$STEPS," == *",1,"* ]]; then
  echo "=== Step 1: subsample raw data ==="
  # --clean so a rebuild leaves no file the current spec does not produce. The
  # manifest is rewritten from the specs either way, so without it a discontinued
  # or renamed artifact loses its manifest entry and stays on disk, where the
  # "git add -A" below would commit it back into the fixture set.
  uv run python tests/create_test_data.py \
      --source "$SOURCE_DATA" \
      --output "$TEST_DATA_DIR/data" \
      --clean
  echo ""
fi

if [[ ",$STEPS," == *",2,"* ]]; then
  echo "=== Step 2: generate pipeline intermediates ==="
  cs_args=()
  if [[ -n "${CASE_STUDIES:-}" ]]; then
    # shellcheck disable=SC2206
    cs_args=(--case-studies ${CASE_STUDIES})
  fi
  ML4T_DATA_PATH="$TEST_DATA_DIR/data" \
  uv run python tests/generate_intermediates.py \
      --output "$TEST_DATA_DIR/intermediates" \
      "${cs_args[@]}"
  echo ""
fi

if [[ ",$STEPS," == *",3,"* ]]; then
  echo "=== Step 3: sample production registries ==="
  # Reads each case study's production run_log/registry.db and copies a subset in.
  # Read-only with respect to the production registries; it never writes them.
  uv run python tests/sample_registry_for_tests.py \
      --output "$TEST_DATA_DIR/intermediates"
  echo ""
fi

echo "=== Done ==="
du -sh "$TEST_DATA_DIR/data" "$TEST_DATA_DIR/intermediates" 2>/dev/null || true
echo ""
echo "Review and commit in the test-data repo:"
echo "  cd $TEST_DATA_DIR && git status && git add -A && git commit"
