"""`chain_worker_pool` must configure sampler workers without changing this process.

The helper exists because `threadpoolctl` cannot reach a `pm.sample(cores=...)` chain
worker: those are fresh interpreters started through multiprocessing's forkserver, and
they size their pools from the environment when they import numpy. Measured on an
sp500_options SV fit, peak threads across the process tree were 113 with nothing set,
115 with `threadpool_limits` wrapped around the sampler, and 41 with this.

What the cases below hold is the part that fails silently. If the variables leak past the
block, every later fit in the notebook runs single-threaded too - the GARCH walk that
`04_model_based_features` spends most of its time in - and nothing reports it.
"""

import os

import pytest

from case_studies.utils.temporal import chain_worker_pool

POOL_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def test_every_pool_variable_is_set_inside_the_block() -> None:
    with chain_worker_pool(1):
        assert [os.environ[name] for name in POOL_VARS] == ["1", "1", "1"]


def test_a_variable_absent_before_is_absent_again_after(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in POOL_VARS:
        monkeypatch.delenv(name, raising=False)
    with chain_worker_pool(2):
        assert os.environ["OMP_NUM_THREADS"] == "2"
    assert [name for name in POOL_VARS if name in os.environ] == []


def test_a_value_set_before_is_restored_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "6")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "6")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    with chain_worker_pool(1):
        pass
    assert os.environ["OMP_NUM_THREADS"] == "6"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "6"
    assert "MKL_NUM_THREADS" not in os.environ


def test_the_block_restores_even_when_the_sampler_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "6")
    with pytest.raises(RuntimeError, match="sampler failed"), chain_worker_pool(1):
        raise RuntimeError("sampler failed")
    assert os.environ["OMP_NUM_THREADS"] == "6"


@pytest.mark.parametrize("n_threads", [0, -1])
def test_a_pool_smaller_than_one_thread_is_refused(n_threads: int) -> None:
    with pytest.raises(ValueError, match="at least one thread"), chain_worker_pool(n_threads):
        pass
