## Task 2: Previous RTH Volume Profile

### Status
Implemented and committed.

### Commit
- `102af95` - Add previous RTH volume profile

### Changes
- Added deterministic 0.25-point MNQ volume binning with 40% value-area expansion, POC, VAH, VAL, and fixed 20th-percentile LVN zones.
- Added previous fully closed RTH profile attachment using the Task 1 normalized data contract, with same-day/future leakage prevention.
- Added validation and focused profile tests.

### Verification
- `uv run --with pytest pytest tests/research/test_mnq_data_contract.py tests/research/test_volume_profile.py -q` - 22 passed.
- `git diff --check` - passed.

### Notes
- Existing unrelated untracked `docs/superpowers/` was not modified.

## Review Fixes: 2026-09-11

### Status
Review fixes implemented; focused verification passed.

### Commit
- `430b36f` - Fix Task 2 volume profile review findings

### Changes
- Prior profiles are built only from RTH sessions containing a closed 5-minute bar starting at `15:55` ET, the bar that ends at the expected `16:00` ET close. Incomplete prior sessions are excluded.
- `timestamp_ny` is required session metadata; missing contract metadata is rejected before profile attachment.
- Value-area expansion adds both available adjacent bins together while the profile is surrounded, with deterministic one-sided expansion at a price-range edge.
- `LVN_PERCENTILE = 0.20` is documented as the fixed 20th-percentile threshold, and LVN bins are selected strictly below that threshold; contiguous qualifying bins are merged.
- Added partial-session leakage, non-RTH, missing metadata, symmetric expansion, and edge-behavior tests.

### Verification
- `uv run --with pytest python -m pytest tests/research/test_volume_profile.py -q` - PASS, 12 tests passed.
- `git diff --check` - PASS.

### Concerns
- The Task 1 contract represents 5-minute bar timestamps as bar starts, so the explicit 16:00 ET close invariant is implemented as a required 15:55 ET final-bar start.
- Unrelated untracked `docs/superpowers/` remains untouched.
