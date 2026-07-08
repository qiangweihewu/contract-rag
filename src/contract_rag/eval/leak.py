"""Permission-leak eval: the over-privilege / 越权 dimension (spec §3.2, gate G1
'0 permission-leaks'). Runs the S2 guard over (identity, retrieved-chunks) cases and
tallies leaks into an obs counter so the metric is COUNTABLE, not just a raised error
(S2-review carry-forward a). Credential-free: pure ABAC set logic, no network."""
from __future__ import annotations

from pydantic import BaseModel

from contract_rag.chunk.models import Chunk
from contract_rag.obs.counters import CounterStore
from contract_rag.security.abac import Principal, allowed_tags_for
from contract_rag.security.guard import Violation, audit_results


class LeakCase(BaseModel):
    name: str
    principal: Principal
    retrieved: list[Chunk]


def evaluate_leaks(cases: list[LeakCase], counter: CounterStore | None = None) -> dict:
    violations: list[Violation] = []
    n_chunks = 0
    for case in cases:
        allowed = allowed_tags_for(case.principal)
        n_chunks += len(case.retrieved)
        violations.extend(audit_results(case.retrieved, allowed))
    if counter is not None:
        counter.incr("permission_leaks", by=len(violations))
        counter.incr("permission_checks", by=n_chunks)
    n_leaks = len(violations)
    return {
        "n_cases": len(cases),
        "n_chunks_checked": n_chunks,
        "n_leaks": n_leaks,
        "leak_rate": (n_leaks / n_chunks if n_chunks else 0.0),
        "clean": n_leaks == 0,
        "violations": violations,
    }


def format_leaks(res: dict) -> str:
    status = "CLEAN" if res["clean"] else "LEAK DETECTED"
    return "\n".join([
        f"=== permission-leak eval: {status} ===",
        f"cases:          {res['n_cases']}",
        f"chunks checked: {res['n_chunks_checked']}",
        f"leaks:          {res['n_leaks']}  (rate {res['leak_rate']:.3f})",
    ])
