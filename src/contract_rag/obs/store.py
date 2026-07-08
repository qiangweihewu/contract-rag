from __future__ import annotations

from pathlib import Path
from typing import Protocol

from contract_rag.obs.models import Trace


class TraceStore(Protocol):
    def add(self, trace: Trace) -> None: ...
    def all(self) -> list[Trace]: ...


class InMemoryTraceStore:
    def __init__(self) -> None:
        self.traces: list[Trace] = []

    def add(self, trace: Trace) -> None:
        self.traces.append(trace)

    def all(self) -> list[Trace]:
        return list(self.traces)


class JsonlTraceStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def add(self, trace: Trace) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(trace.model_dump_json() + "\n")

    def all(self) -> list[Trace]:
        if not self.path.exists():
            return []
        out: list[Trace] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(Trace.model_validate_json(line))
        return out
