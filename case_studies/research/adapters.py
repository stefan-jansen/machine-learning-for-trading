from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType


@dataclass(frozen=True)
class AdapterBinding:
    kind: str
    name: str
    module: str

    def load(self) -> ModuleType:
        return importlib.import_module(self.module)


_BINDINGS: dict[tuple[str, str], AdapterBinding] = {}
_KINDS = {"causal", "model"}


def register_adapter(kind: str, name: str, module: str, *, replace: bool = False) -> None:
    if kind not in _KINDS:
        raise ValueError(f"unsupported adapter kind {kind!r}; expected {sorted(_KINDS)}")
    if not name or not module:
        raise ValueError("adapter name and module are required")
    key = (kind, name)
    binding = AdapterBinding(kind, name, module)
    existing = _BINDINGS.get(key)
    if existing is not None and existing != binding and not replace:
        raise ValueError(f"adapter {kind}/{name} is already registered by {existing.module}")
    _BINDINGS[key] = binding


def get_adapter(kind: str, name: str) -> ModuleType:
    try:
        binding = _BINDINGS[(kind, name)]
    except KeyError as exc:
        available = sorted(key_name for key_kind, key_name in _BINDINGS if key_kind == kind)
        raise ValueError(
            f"unsupported {kind} adapter {name!r}; available adapters: {available}"
        ) from exc
    return binding.load()


def registered_adapters(kind: str) -> tuple[AdapterBinding, ...]:
    if kind not in _KINDS:
        raise ValueError(f"unsupported adapter kind {kind!r}; expected {sorted(_KINDS)}")
    return tuple(
        binding for (binding_kind, _), binding in sorted(_BINDINGS.items()) if binding_kind == kind
    )


for _name, _module in {
    "deep_learning": "case_studies.utils.deep_learning",
    "gbm": "case_studies.utils.gbm",
    "latent_factors": "case_studies.utils.latent_factors",
    "linear": "case_studies.utils.linear",
    "tabular_dl": "case_studies.utils.tabular_dl",
}.items():
    register_adapter("model", _name, _module)

register_adapter("causal", "dml", "case_studies.utils.causal")
