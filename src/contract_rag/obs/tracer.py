from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Protocol, runtime_checkable

from contract_rag.obs.models import Span, SpanStatus, Trace


@runtime_checkable
class TracerProtocol(Protocol):
    """Structural interface both Tracer and NoopTracer satisfy. Annotate cross-lane
    `tracer:` seams (agent S3, API S4) with this, not the concrete class."""

    def start(self, doc_id: str, trace_id: str | None = None) -> Trace: ...
    def span(self, trace: Trace, name: str) -> AbstractContextManager[Span]: ...
    def finish(self, trace: Trace) -> None: ...


class Tracer:
    def __init__(
        self,
        store=None,
        clock: Callable[[], float] = time.perf_counter,
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self._store = store
        self._clock = clock
        self._id_factory = id_factory

    def start(self, doc_id: str, trace_id: str | None = None) -> Trace:
        return Trace(trace_id=trace_id or self._id_factory(), doc_id=doc_id)

    @contextmanager
    def span(self, trace: Trace, name: str) -> Iterator[Span]:
        span = Span(name=name)
        started = self._clock()
        try:
            yield span
        except TimeoutError:
            span.status = SpanStatus.TIMEOUT
            span.error_type = "TimeoutError"
            raise
        except Exception as exc:
            span.status = SpanStatus.ERROR
            span.error_type = type(exc).__name__
            raise
        finally:
            span.duration_ms = (self._clock() - started) * 1000.0
            trace.spans.append(span)

    def finish(self, trace: Trace) -> None:
        if self._store is not None:
            self._store.add(trace)


class NoopTracer:
    def start(self, doc_id: str, trace_id: str | None = None) -> Trace:
        return Trace(trace_id="noop", doc_id=doc_id)

    @contextmanager
    def span(self, trace: Trace, name: str) -> Iterator[Span]:
        yield Span(name=name)

    def finish(self, trace: Trace) -> None:
        return None
