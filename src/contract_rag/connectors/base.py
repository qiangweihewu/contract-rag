from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    id: str                                # connector-native id / path
    name: str
    uri: str | None = None
    metadata: dict = Field(default_factory=dict)


@runtime_checkable
class Connector(Protocol):
    """One interface for every source system. A document flows
    list_documents() -> fetch(ref) -> a local Path -> ingest_document(...)."""

    name: str

    def list_documents(self, *, prefix: str | None = None) -> Iterable[SourceRef]: ...
    def fetch(self, ref: SourceRef) -> Path: ...
