## Task 3: Objective Signal Definitions

### Status
Implemented, verified, and committed.

### Commit
- Implementation committed on `feat/mnq-objective-strategy`; see the branch head for the final commit.

### Changes
- Added deterministic Midnight Open rejection detection using the exact `00:00` New York bar, closed-bar gating, a `2.0` wick/body threshold, outer-30% close location, level crossing, and first-rejection-per-session gating.
- Added the momentum helper with body/range `>= 0.60`, outer-20% close location, six prior closed bars, and `1.5x` prior-body threshold. Momentum is not emitted by `combine_allowed_signals`.
- Added LVN break/retest/confirmation detection from previous-session zones. The profile threshold is the fixed 25th percentile. The state machine requires a later closed retest and a later closed confirmation, applies direction-specific invalidation, and emits at most the first confirmation per session.
- Added the 10:00 setup, which evaluates only the closed `10:00-10:05` bar, derives direction from momentum, and emits at `10:05` New York time.
- Added deterministic combine precedence: rejection, then LVN, then 10:00, with one output signal per bar.
- Added optional same-bar ES coherence filtering when `es_direction` or ES close columns are present; MNQ behavior remains testable when ES data is absent.
- Added focused signal tests and a regression test preventing a break bar from also serving as its own retest.

### Verification
- `uv run --with pytest pytest tests/research/test_signals.py -q` — **16 passed**.
- `uv run --with pytest pytest tests/research -q` — **44 passed**.
- `uv run --with ruff ruff check research/mnq_strategy/signals.py research/mnq_strategy/volume_profile.py tests/research/test_signals.py tests/research/test_volume_profile.py` — **passed**.
- `uv run --with ruff ruff format --check research/mnq_strategy/signals.py research/mnq_strategy/volume_profile.py tests/research/test_signals.py tests/research/test_volume_profile.py` — **4 files already formatted**.
- `uv run python -m py_compile research/mnq_strategy/signals.py research/mnq_strategy/volume_profile.py tests/research/test_signals.py tests/research/test_volume_profile.py` — **passed**.
- `git diff --check` — **passed**.
- Manual fixture inspection confirmed Midnight entry at the rejection close, LVN entry at confirmation, and 10:00 entry at `10:05`.
- Repository-wide `uv run --with pytest pytest -q` was started but timed out after 120 seconds while running unrelated existing tests; it had already exposed unrelated failures outside `tests/research`. The scoped MNQ suite is green.

### Caveats
- The canonical MNQ input contract does not include ES data. The optional ES coherence path is implemented and tested for present ES columns, but no ES feed integration or production ES fixture is included.
- The base project environment did not expose `pytest` or `ruff` as direct executables; verification used `uv run --with pytest` and `uv run --with ruff` without changing project dependencies.
