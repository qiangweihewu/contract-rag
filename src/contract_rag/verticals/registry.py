from __future__ import annotations

from contract_rag.config import Settings
from contract_rag.verticals.base import Vertical

_REGISTRY: dict[str, Vertical] = {}


def register_vertical(vertical: Vertical) -> None:
    """Public extension point: register a vertical so the engines resolve it by name.
    Adding a vertical requires only this call — no edits to any generic module."""
    _REGISTRY[vertical.name] = vertical


def get_vertical(name: str) -> Vertical:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise NotImplementedError(
            f"vertical {name!r} not registered; available: {sorted(_REGISTRY)}"
        ) from None


def get_vertical_for(settings: Settings) -> Vertical:
    return get_vertical(settings.vertical)


def default_vertical() -> Vertical:
    """The active vertical, resolved from settings (VERTICAL env; defaults to contract).
    The shared fallback for engines that don't take an explicit vertical, so extractor,
    enrich, verify, metrics, and the agent all agree on which vertical is active."""
    from contract_rag.config import get_settings
    return get_vertical_for(get_settings())


def _register_builtins() -> None:
    from contract_rag.verticals.contract.vertical import ContractVertical
    from contract_rag.verticals.nda.vertical import NDAVertical
    register_vertical(ContractVertical())
    register_vertical(NDAVertical())


_register_builtins()
