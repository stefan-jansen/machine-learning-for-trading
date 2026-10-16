"""TabM has to be an ensemble, and the class that shipped was algebraically one MLP.

Both copies of TabM in this repository put their rank-one adapters *after* the last
nonlinearity and trained a single loss on the already-averaged output. Member ``i``
computed ``W_i @ (a_i * h)`` from the shared trunk output ``h``, so the ensemble mean was
``mean_i(W_i @ diag(a_i)) @ h + mean_i(b_i)``: one linear readout of one trunk. Chapter 12
describes members that "overfit in different directions" and an average that cancels
their errors, and neither property is available to a model with one effective member.

The existing identity tests pin parameter names, shapes and counts, all of which the
collapsed architecture satisfied. These pin what those pins could not see, and they are
written as properties of the model's behaviour rather than of its attribute names, so
they still decide the question for an implementation that reaches the defect another way.
"""

from __future__ import annotations

import numpy as np
import torch

from case_studies.utils.tabular_dl import TabMModel, _train_tabm_fold

# n_members well above hidden_dim is what makes the rank test below decidable: a
# collapsed ensemble's member outputs span at most hidden_dim dimensions however many
# members it declares.
N_FEATURES = 6
HIDDEN_DIM = 4
N_MEMBERS = 12
N_INPUTS = 200


class _CollapsedTabM(torch.nn.Module):
    """The architecture that shipped, kept so the rank test can be shown to fail on it.

    Without it, the test below would assert that some integer is large and prove nothing
    about whether a wrong implementation would make it small.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(N_FEATURES, HIDDEN_DIM),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            torch.nn.ReLU(),
        )
        self.adapters = torch.nn.Parameter(torch.randn(N_MEMBERS, HIDDEN_DIM) * 0.1)
        self.heads = torch.nn.ModuleList([torch.nn.Linear(HIDDEN_DIM, 1) for _ in range(N_MEMBERS)])

    def forward_members(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return torch.stack(
            [self.heads[i](h * self.adapters[i].unsqueeze(0)) for i in range(N_MEMBERS)], dim=0
        )


def _member_output_rank(model: torch.nn.Module) -> tuple[int, np.ndarray]:
    """The dimension the k member outputs actually span, over a fixed input sample.

    If every member is an affine readout of one shared vector ``h`` of width
    ``HIDDEN_DIM``, every column of this matrix lies in the column space of that trunk's
    activations and the rank is at most ``HIDDEN_DIM``, whatever k is. If the members are
    different functions, the rank is k.
    """
    torch.manual_seed(1)
    inputs = torch.randn(N_INPUTS, N_FEATURES)
    with torch.no_grad():
        members = model.forward_members(inputs).squeeze(-1).T.numpy()
    centered = members - members.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    return int(np.linalg.matrix_rank(centered, tol=1e-6 * singular[0])), singular


class TestTheEnsembleDoesNotCollapse:
    def test_the_members_are_not_affine_readouts_of_one_shared_vector(self) -> None:
        """The defect this file exists to prevent, as a rank.

        Measured 2026-10-15: rank 12 of 12 members here against rank 4 for the
        architecture that shipped, whose fourth singular value is already 7.9e-07.
        """
        torch.manual_seed(0)
        model = TabMModel(
            n_features=N_FEATURES, hidden_dim=HIDDEN_DIM, n_members=N_MEMBERS, dropout=0.0
        ).eval()

        rank, singular = _member_output_rank(model)

        assert rank > HIDDEN_DIM, (
            f"the {N_MEMBERS} member outputs span only {rank} dimensions against a hidden "
            f"width of {HIDDEN_DIM}, so every member is an affine readout of one shared "
            f"vector and the ensemble has one effective member "
            f"(singular values {np.array2string(singular[:6], precision=3)})"
        )
        assert rank == N_MEMBERS

    def test_the_architecture_that_shipped_fails_that_rank_test(self) -> None:
        """The negative selftest: the assertion above is decidable, not decorative."""
        torch.manual_seed(0)

        rank, _ = _member_output_rank(_CollapsedTabM().eval())

        assert rank <= HIDDEN_DIM, (
            f"the collapsed reference model spans {rank} dimensions, more than its hidden "
            f"width of {HIDDEN_DIM}; it no longer reproduces the defect and the test above "
            "is no longer shown to catch anything"
        )


class TestEveryMemberIsTrained:
    def test_each_member_is_a_working_model_rather_than_a_term_in_an_average(self) -> None:
        """One loss on the averaged output leaves the individual members unfit.

        Each member must predict the target about as well as the ensemble does, which is
        what "k models that share their weights" means. Measured 2026-10-15 on this fold:
        the worst member is 1.5x the ensemble's error under the mean of the per-member
        losses, and 36x under one loss on the averaged output. The ensemble's own error is
        nearly identical either way, 0.0357 against 0.0331, which is why the collapse was
        invisible in every headline number the notebooks report.
        """
        rng = np.random.default_rng(0)
        x_train = rng.standard_normal((256, N_FEATURES))
        weights = rng.standard_normal(N_FEATURES)
        y_train = x_train @ weights + 0.1 * rng.standard_normal(256)
        x_val = rng.standard_normal((64, N_FEATURES))
        y_val = x_val @ weights + 0.1 * rng.standard_normal(64)

        torch.manual_seed(0)
        model = TabMModel(n_features=N_FEATURES, hidden_dim=32, n_members=3, dropout=0.0)
        _train_tabm_fold(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            y_val,
            np.repeat(np.array([0, 1]), 32),
            None,
            n_epochs=60,
            batch_size=64,
            checkpoint_interval=60,
            device=torch.device("cpu"),
        )

        model.eval()
        targets = torch.tensor(y_val, dtype=torch.float32)
        with torch.no_grad():
            inputs = torch.tensor(x_val, dtype=torch.float32)
            member_errors = ((model.forward_members(inputs).squeeze(-1) - targets) ** 2).mean(dim=1)
            ensemble_error = float(((model(inputs) - targets) ** 2).mean())

        worst = float(member_errors.max())
        assert worst < 4.0 * ensemble_error, (
            f"the worst member's error is {worst:.4f} against the ensemble's "
            f"{ensemble_error:.4f}, so the loss is fitting the average rather than the "
            "members and no member is a model in its own right"
        )
