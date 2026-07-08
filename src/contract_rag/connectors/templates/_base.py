from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from contract_rag.connectors.base import SourceRef


class TemplateConnector:
    """Base for inert connector templates. Structurally a `Connector`, but every
    operation raises until implemented. Copy a template module, fill in the SDK calls
    in list_documents/fetch, then inject the instance (connectors are passed in, not
    auto-registered — keeping the credential-free default path intact)."""

    name = "template"

    def __init__(self, **config) -> None:
        self.config = config

    def list_documents(self, *, prefix: str | None = None) -> Iterable[SourceRef]:
        raise NotImplementedError(
            f"{type(self).__name__}.list_documents is a template — implement it "
            "(see the module docstring for the auth model + API surface)."
        )

    def fetch(self, ref: SourceRef) -> Path:
        raise NotImplementedError(
            f"{type(self).__name__}.fetch is a template — implement it "
            "(see the module docstring for the auth model + API surface)."
        )
