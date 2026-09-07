"""Every iterative latent-factor fit records a convergence determination, and the runner
refuses a cohort that did not converge.

`_require_ipca_convergence` reads a ``converged`` flag that only the IPCA branch of
``library_bridge`` writes. The SDF, CAE and SAE branches wrote no such flag, so a fit that
never identified registered, its predictions entered the population, and `require_complete`
and the notebook's IC table both passed - the failure was invisible at the point where it
happened.

The determination is not the same statement for every model and these tests say which is
which. IPCA reports whether its alternating least squares settled within ``tol``. The three
gradient-descent models have no such tolerance - SAE's training loss routinely rises on the
last step of a short fit - so what is checkable there is that the fit produced a finite
terminal objective, and for SDF a finite terminal Sharpe as well. A delta is recorded as a
diagnostic and is deliberately not gated on.
"""

from __future__ import annotations

import numpy as np
import pytest

from case_studies.utils.latent_factors import library_bridge
from case_studies.utils.latent_factors.cv import _require_fit_convergence


def _panel(seed: int = 0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    chars_train = rng.normal(size=(24, 6, 3)).astype(np.float32)
    returns_train = (rng.normal(size=(24, 6)) * 0.01).astype(np.float32)
    chars_val = rng.normal(size=(8, 6, 3)).astype(np.float32)
    returns_val = (rng.normal(size=(8, 6)) * 0.01).astype(np.float32)
    return chars_train, returns_train, chars_val, returns_val


def test_a_diverged_tail_is_not_hidden_by_an_earlier_finite_objective() -> None:
    """The terminal objective is the last one recorded, not the last finite one.

    `_latest_sdf_sharpe` scans backwards for a finite value, so a fit that ran to NaN reports
    the Sharpe from before it diverged. A convergence determination read the same way would
    call that fit converged, which is the opposite of what happened.
    """
    history = [
        {"epoch": 1.0, "train_loss": 0.5},
        {"epoch": 2.0, "train_loss": 0.4},
        {"epoch": 3.0, "train_loss": float("nan")},
    ]
    determination = library_bridge.fit_convergence(history)
    assert determination["converged"] is False
    assert np.isnan(determination["terminal_objective"])
    assert determination["iterations"] == 3


def test_the_cae_validation_summary_entry_is_not_the_terminal_objective() -> None:
    """CAE appends a ``validation_best`` entry after its per-checkpoint ones, and that entry
    carries ``val_loss`` and no ``train_loss``. Reading the objective off the last entry
    would report every CAE fit as having produced none."""
    history = [
        {"epoch": 5.0, "train_loss": 0.3},
        {"epoch": 10.0, "train_loss": 0.2},
        {"epoch": 0.0, "checkpoint": "validation_best", "val_loss": 0.1},
    ]
    determination = library_bridge.fit_convergence(history)
    assert determination["converged"] is True
    assert determination["terminal_objective"] == pytest.approx(0.2)
    assert determination["objective_delta"] == pytest.approx(-0.1)
    assert determination["iterations"] == 2


def test_a_rising_objective_still_converges() -> None:
    """A short SAE fit ends on a higher loss than the step before it. Gating on the delta
    would refuse a fit that is fine, so the delta is recorded and not gated on."""
    history = [
        {"epoch": 1.0, "train_loss": 1.61},
        {"epoch": 2.0, "train_loss": 1.44},
        {"epoch": 3.0, "train_loss": 1.64},
    ]
    determination = library_bridge.fit_convergence(history)
    assert determination["converged"] is True
    assert determination["objective_delta"] == pytest.approx(0.2)


def test_an_empty_history_never_converges() -> None:
    determination = library_bridge.fit_convergence([])
    assert determination["converged"] is False
    assert determination["terminal_objective"] is None
    assert determination["objective_delta"] is None
    assert determination["iterations"] == 0


def test_the_sdf_criterion_also_requires_a_finite_terminal_sharpe() -> None:
    """A finite loss with a non-finite Sharpe is a fit whose reported statistic does not
    exist. The same history passes the criterion that does not ask for one."""
    history = [
        {"epoch": 1.0, "phase": "conditional", "train_loss": 0.4, "train_sharpe": 0.2},
        {"epoch": 2.0, "phase": "conditional", "train_loss": 0.3, "train_sharpe": float("nan")},
    ]
    assert library_bridge.fit_convergence(history, require_finite_sharpe=True)["converged"] is False
    assert library_bridge.fit_convergence(history)["converged"] is True


def test_a_real_sdf_fit_records_its_convergence_determination() -> None:
    chars_train, returns_train, chars_val, returns_val = _panel()
    _, extras = library_bridge.run_sdf_fold_with_library(
        chars_train,
        returns_train,
        chars_val,
        returns_val,
        n_epochs_unc=3,
        n_epochs_moment=2,
        n_epochs_cond=3,
        checkpoint_epochs=[3],
        beta_n_epochs=3,
        beta_checkpoint_epochs=[3],
        beta_default_checkpoint=3,
        output_mode="weights",
        seed=1,
    )
    assert extras["converged"] is True
    assert extras["convergence_criterion"] == "finite_terminal_objective_and_sharpe"
    assert np.isfinite(extras["terminal_objective"])
    assert np.isfinite(extras["terminal_sharpe"])
    assert extras["iterations"] == 6


def test_a_real_cae_fit_records_its_convergence_determination() -> None:
    chars_train, returns_train, chars_val, returns_val = _panel()
    _, extras = library_bridge.run_cae_fold_with_library(
        chars_train,
        returns_train,
        chars_val,
        returns_val,
        n_factors=2,
        n_epochs=3,
        checkpoint_interval=None,
        checkpoint_epochs=[3],
        seed=1,
    )
    assert extras["converged"] is True
    assert extras["convergence_criterion"] == "finite_terminal_objective"
    assert np.isfinite(extras["terminal_objective"])


def test_a_real_sae_fit_records_its_convergence_determination() -> None:
    chars_train, returns_train, chars_val, returns_val = _panel()
    _, extras = library_bridge.run_sae_fold_with_library(
        chars_train,
        returns_train,
        chars_val,
        returns_val,
        n_epochs=3,
        checkpoint_interval=None,
        checkpoint_epochs=[3],
        seed=1,
    )
    assert extras["converged"] is True
    assert extras["convergence_criterion"] == "finite_terminal_objective"
    assert np.isfinite(extras["terminal_objective"])


@pytest.mark.parametrize("model_name", ["sdf", "cae", "sae", "ipca"])
def test_the_guard_refuses_a_cohort_a_fold_did_not_converge_in(model_name: str) -> None:
    with pytest.raises(RuntimeError, match=r"folds \[2\].*refusing to register"):
        _require_fit_convergence(
            model_name,
            [{"fold_id": 1, "converged": True}, {"fold_id": 2, "converged": False}],
        )


@pytest.mark.parametrize("model_name", ["sdf", "cae", "sae"])
def test_the_guard_refuses_a_cohort_that_recorded_no_determination(model_name: str) -> None:
    """A runner that stops writing the flag must fail loudly rather than silently lose the
    guard. IPCA is excluded: its branch treats an absent flag as a fit that did not settle,
    and that reading is what the historical extras were written under."""
    with pytest.raises(RuntimeError, match="no convergence determination"):
        _require_fit_convergence(model_name, [{"fold_id": 0}])


def test_the_guard_leaves_pca_alone() -> None:
    """PCA is a deterministic decomposition with nothing to converge, so it records no
    determination and must not be asked for one."""
    _require_fit_convergence("pca", [{"fold_id": 0}])


def test_the_determination_never_reaches_the_training_identity() -> None:
    """A convergence determination says how a fit went, not what was fitted.

    `computation` is hashed whole, so a key that reaches it reprices every registered row in
    every registry. `_apply_latent_factor_runtime_spec` copies a fixed allowlist out of the
    first fold's extras, and none of the determination's keys is on it - this pins that,
    because adding one there is a refit of four case studies rather than a diagnostic.
    """
    from case_studies.utils.latent_factors.cv import _apply_latent_factor_runtime_spec

    determination = library_bridge.fit_convergence(
        [{"epoch": 1.0, "train_loss": 0.4, "train_sharpe": 0.2}],
        require_finite_sharpe=True,
    )

    def _spec(fold_extras: list[dict]) -> dict:
        return _apply_latent_factor_runtime_spec(
            spec={
                "family": "latent_factors",
                "config_name": "sdf",
                "label": "fwd_ret_1m",
                "params": {"n_factors": 3},
                "seed": 42,
            },
            model_name="sdf",
            n_factors=3,
            n_epochs=2,
            model_kwargs={},
            fold_extras=fold_extras,
            feature_names=["a", "b"],
            splits=[{"fold": 0}],
            task_type="regression",
            class_values=None,
            eval_label_col=None,
            input_digest="deadbeef",
            macro_digest=None,
            runtime_spec={"device": "cpu"},
        )

    base = {"fold_id": 0, "checkpoint_epochs": [0, 5], "output_mode": "weights"}
    assert _spec([base]) == _spec([{**base, **determination}])
