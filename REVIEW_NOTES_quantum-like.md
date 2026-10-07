# Review notes for Stefan (from the quantum-like fork)

Reviewed the 3rd-edition tip `827444d` (`fix(ch24): make forecasting notebooks evidence-backed and reproducible`) on branch `review/3e-fixes`. The fork's `main` is still the 2nd edition (`f652d79`, last synced Aug 2024) and was not modified, reset, or pushed.

This is a static review plus two small unit tests. It is not a full re-run of the notebooks or the case-study pipelines.

## Commits

| SHA | File | Category | Why |
|-----|------|----------|-----|
| `5e6b08a` | `utils/modeling.py` | broken code | `get_cv_config("nasdaq100_microstructure")` sliced `labels.buffer` of `16min` to `int("16mi")` and raised before setting an embargo. |
| `b134a5b` | `.github/workflows/weekly-external.yml` | broken code | The weekly JUnit parser matched `name="` inside `classname="`, so every failure was filed as `weekly-flake: tests.test_chapter_notebooks` (#1046) instead of the notebook that failed. |
| `c2ca50b` | `09_model_based_features/13_regime_as_feature.py` | look-ahead (explanation) | The target is `rolling_sum(5).shift(-5)`, which is returns through \(t+5\). The markdown said \(t+4\). The `TimeSeriesSplit` gap was already 5; only the description of how far the label reaches was short. |

## Questions for Stefan

1. **Chapter 8 vol-scaled momentum.** In `08_financial_features/01_price_volume_features.py` the displayed formula is \(r_{21d}/\sigma_{21d}\), and the column is `vol_21d`, but the code is `pct_change(21) / (daily_return.rolling_std(21) * sqrt(252))`. That divides a 21-day return by an annualized volatility. Is the `sqrt(252)` intentional, or should the denominator be the 21-day realized vol (daily std, or daily std times `sqrt(21)`)?

2. **Mean-variance cap vs. equal-weight fallback.** `case_studies/utils/allocation.py` `compute_mvo_ledoit_wolf_weights` uses long-only bounds `(0, max_weight)` with default `max_weight=0.15` and a constraint that weights sum to 1. For `top_k < 7` the cap sum is below 1, so SLSQP cannot succeed, and the fallback is equal weight `1/n`, which itself breaks the cap. Is that the intended fallback when the quadratic program is infeasible?

3. **Weekly Hugging Face failure, separate from the parser bug.** Run [37320316180](https://github.com/stefan-jansen/machine-learning-for-trading/actions/runs/37320316180) (2026-10-05) failed `22_rag_financial_research/02_domain_embeddings_comparison.py` in cell 27 with `OSError: We couldn't connect to 'https://huggingface.co'`. The workflow comment says `HF_TOKEN` was added after the 2026-09-14 failure. The title on #1046 hid which notebook it was; the connection error itself looks environmental. Is that model expected to be cached in `ml4t/ml4t:latest` so the weekly job does not have to reach Hugging Face?

## What I could not check

- Notebook execution and case-study training. This environment is Python 3.12; the project requires 3.14, and the `ml4t-*` libraries and the datasets were not installed. I did not download data or fit models.
- The private `ml4t/third-edition-test-data` checkout the weekly job uses.
- Whether a fresh `uv sync` against the unpinned `pandas>=2.2` bound still matches the Docker image (no image run).
- Chapters 12–27 and the nine case studies beyond the leakage, label, CV, allocation, and cost paths I read. The Gate-0 detectors in `tests/test_leakage_detectors.py` already ratchet shuffled splits, backward fills, and global HMM decoding for chapters 1–11; I did not re-audit every allowlisted occurrence.
- The numerical Almgren–Chriss trajectory in `18_transaction_costs/05_almgren_chriss_optimal_execution.py` and the HRP bisection in `case_studies/utils/allocation.py` looked consistent with the papers, but I did not re-derive them against a reference implementation.
