"""Monotonic event counters — the countable side of observability (e.g. the
'0 permission-leaks' G1 metric). Mirrors obs/store.py: a Protocol + in-memory default."""
from __future__ import annotations

from collections import Counter
from typing import Protocol


class CounterStore(Protocol):
    def incr(self, name: str, by: int = 1) -> None: ...
    def value(self, name: str) -> int: ...
    def snapshot(self) -> dict[str, int]: ...


class InMemoryCounterStore:
    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()

    def incr(self, name: str, by: int = 1) -> None:
        self._counts[name] += by

    def value(self, name: str) -> int:
        return self._counts[name]

    def snapshot(self) -> dict[str, int]:
        return dict(self._counts)
