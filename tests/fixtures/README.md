# Test Fixtures

Minimal results JSON files seeded into test output directories so downstream
notebooks (latent factors, DL, backtest, etc.) can find baseline results
without depending on upstream notebook execution order.

These are NOT real results — they contain plausible placeholder values that
allow the comparison/selection code paths to execute without crashing.

Seeded by `conftest.py::seeded_output_dir` fixture.
