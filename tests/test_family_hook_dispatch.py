"""A model-family hook the package does not export is not a hook, and `getattr` says so silently.

`case_studies.research.models._family_module` returns the family's PACKAGE, so a function
defined in a submodule and left out of `__init__.py` resolves to `None`. Every caller then
degrades quietly: `_holdout_training_floor` returns no floor, `_rekey_holdout_spec` raises
`NotImplementedError`. Both read as "this family does not implement it" rather than as the
export bug they are, and that is exactly how the missing `holdout_training_floor` export
shipped once already.

Resolving the way production resolves is the only check that can tell those two apart, which
is why this imports the real packages instead of asserting against `__all__`. It therefore
needs the modelling environment: `case_studies.utils.latent_factors` imports torch at module
scope on purpose, so this file runs in `test-unit-image` and is quarantined out of `test-unit`.
"""

from __future__ import annotations

import pytest

# Every family that `_family_module` can be asked for, and the hooks the dispatch calls on it.
FAMILY_HOOKS = {
    "latent_factors": ("rekey_holdout_spec", "holdout_training_floor"),
    "linear": ("rekey_holdout_spec",),
}


@pytest.mark.parametrize("family", sorted(FAMILY_HOOKS))
def test_every_family_hook_is_reachable_through_the_dispatch_that_calls_it(family: str) -> None:
    from case_studies.research.models import _family_module

    module = _family_module(family)
    for hook in FAMILY_HOOKS[family]:
        assert callable(getattr(module, hook, None)), (
            f"{family}.{hook} is defined in a submodule but not exported from the package "
            f"`_family_module` returns, so the dispatch cannot see it"
        )
