"""Declared behaviour versions for the latent-factor family.

These literals are what a latent training identity records instead of a digest of the
source files, so a fix that changes no result must not move them and a change that
changes one must. They live in a module that imports nothing so that both readers -
``adapter.py``, which builds the resolved identity, and ``cv.py``, which builds the
legacy one - can read the same values without either importing the other. Two copies
would decide separately what a model's version is, and a disagreement between them
would register two identities for one fitted result.

``LATENT_ADAPTER_VERSION`` covers the shared machinery every model runs through: this
package's panel and fold preparation, the CV loop, the library bridge, and the adapter
itself. A change confined to one model bumps only that model's entry, so editing
``sae.py`` does not refit IPCA - which the whole-file digest this replaced could not
express.

Each entry describes the runner in the file of the same name. ``sae`` is at 2 because
``run_sae_fold_with_library`` omitted ``batch_size``, and the library read the absent
value as one full-panel batch; results fitted before that fix are a different
computation and must not be reused.
"""

from __future__ import annotations

LATENT_ADAPTER_VERSION = 1

LATENT_MODEL_VERSIONS = {
    "cae": 1,
    "ipca": 1,
    "pca": 1,
    "sae": 2,
    "sdf": 1,
}

_LATENT_MODELS = frozenset(LATENT_MODEL_VERSIONS)


def latent_model_version(model_name: str) -> int:
    """The declared behaviour version of one latent model."""
    try:
        return LATENT_MODEL_VERSIONS[model_name]
    except KeyError as exc:
        raise KeyError(f"no declared runner version for latent model {model_name!r}") from exc
