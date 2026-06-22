#!/bin/bash
# Regenerate all test data (raw + intermediates) for CI.
#
# Run from the review repo root. Requires:
# - ML4T_DATA_PATH to point to full production data (default: ~/Dropbox/ml4t/data)
# - The working repo at ~/ml4t/third-edition
# - The review repo at ~/ml4t/technical_review
#
# Usage:
#   cd ~/ml4t/technical_review
#   bash tests/generate_test_data.sh [TEST_DATA_DIR]

set -euo pipefail

TEST_DATA_DIR="${1:-$HOME/ml4t/test-data}"
WORKING_REPO="$HOME/ml4t/third-edition"
REVIEW_REPO="$HOME/ml4t/technical_review"

echo "=== ML4T Test Data Generator ==="
echo "Output:       $TEST_DATA_DIR"
echo "Working repo: $WORKING_REPO"
echo "Review repo:  $REVIEW_REPO"
echo ""

# Step 1: Generate subsampled raw data
echo "=== Step 1: Generating subsampled data ==="
cd "$WORKING_REPO"
uv run python tests/create_test_data.py \
    --source "${ML4T_DATA_PATH:-$HOME/Dropbox/ml4t/data}" \
    --output "$TEST_DATA_DIR/data" \
    --clean
echo ""

# Step 2: Deploy latest notebooks from third-edition -> review repo
echo "=== Step 2: Deploying latest notebooks ==="
cd "$WORKING_REPO"
uv run python scripts/deploy_to_review.py --all
echo ""

# Step 3: Generate intermediates (runs pipeline notebooks via Papermill)
echo "=== Step 3: Generating intermediates ==="
cd "$REVIEW_REPO"
ML4T_DATA_PATH="$TEST_DATA_DIR/data" \
MPLBACKEND=Agg \
PLOTLY_RENDERER=json \
uv run python tests/generate_intermediates.py \
    --output "$TEST_DATA_DIR/intermediates"
echo ""

# Summary
echo "=== Done ==="
echo "Test data directory: $TEST_DATA_DIR"
du -sh "$TEST_DATA_DIR/data" "$TEST_DATA_DIR/intermediates" 2>/dev/null || true
echo ""
echo "Next steps:"
echo "  cd $TEST_DATA_DIR"
echo "  git add -A"
echo "  git commit -m 'update: regenerate test data'"
echo "  git push"
