### Task 2: Build previous-session 40% Volume Profile and LVN extraction

**Files:**
- Create: `research/mnq_strategy/volume_profile.py`
- Create: `tests/research/test_volume_profile.py`

**Interfaces:**
- Produces `build_rth_profile(rth_bars: pl.DataFrame, value_area_fraction: float = 0.40) -> dict`.
- Returns `poc`, `vah`, `val`, `value_area_fraction`, `volume_by_price`, and `lvn_zones`.
- Produces `attach_previous_rth_profile(bars: pl.DataFrame) -> pl.DataFrame`.

- [ ] **Step 1: Write failing tests**

```python
import polars as pl
from research.mnq_strategy.volume_profile import build_rth_profile


def test_value_area_is_40_percent_and_contains_poc():
    bars = pl.DataFrame({
        "price": [100.0, 100.25, 100.5, 100.75, 101.0],
        "volume": [10, 40, 100, 30, 10],
    })
    profile = build_rth_profile(bars, value_area_fraction=0.40)
    assert profile["value_area_fraction"] == 0.40
    assert profile["poc"] == 100.5
    assert profile["val"] <= profile["poc"] <= profile["vah"]
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/research/test_volume_profile.py -q`
Expected: FAIL because the profile module does not exist.

- [ ] **Step 3: Implement the profile**

Aggregate volume at the configured MNQ tick/price bins. Select the POC as the highest-volume bin. Expand symmetrically by adjacent volume contribution until the cumulative included volume reaches 40% of total volume. Define LVN zones from contiguous bins below a documented percentile threshold; make the threshold a fixed configuration value, not an optimizer parameter in version 1. When attaching profiles, only use the previous fully closed RTH date.

- [ ] **Step 4: Add leakage regression coverage and run tests**

Assert that a current-day RTH bar never receives a profile built from its own later bars. Run: `uv run pytest tests/research/test_volume_profile.py -q`
Expected: PASS.

---

