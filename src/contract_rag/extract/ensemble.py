"""Field-level ensemble extraction backend (`EXTRACT_BACKEND=ensemble`).

Measured motivation (see CLAUDE.md's per-field tables): `rule` beats the LLM
backends on `effective_date` (0.94 vs constrained's 0.438) and is close on
`governing_law`; the `constrained` LLM backend beats `rule` on `counterparty`
(0.744 vs 0.49) and `auto_renewal` (0.647 vs 0.53). No single backend wins every
field, so `EnsembleExtractor` composes two (or more) child extractors and routes
per-field to whichever one measurably wins that field, falling back to the other
child when the routed one comes back empty.

Routing:
  - `DEFAULT_ROUTING` reflects the measured table above: `effective_date`,
    `governing_law`, `termination_notice_days` -> `rule`; `counterparty`,
    `auto_renewal`, `total_value` -> `constrained`.
  - Fields absent from `DEFAULT_ROUTING` (e.g. an NDA field, or a future contract
    field) default to `rule` — the credential-free, deterministic floor.
  - Overridable per-field via the `ENSEMBLE_ROUTING` env var, e.g.
    `ENSEMBLE_ROUTING=counterparty=constrained,effective_date=rule`
    (`parse_routing_env` — comma-separated `field=backend` pairs; malformed
    entries are ignored) or the `routing=` constructor kwarg directly (tests).
  - Children default to real backends built via `extract.extractor.get_extractor`
    (so `assert_backend_allowed` gates any child that needs it — e.g. routing a
    field to `openai` still requires `ALLOW_EXTERNAL_LLM=true`), but are fully
    injectable via `children=` for dep-free unit tests.

Fallback: if the routed child returns an empty clause for a field and another
configured child has a populated clause for that field, the populated one is used
instead (`last_fallbacks` counts how many fields fell back, keyed by field name).

Vertical-generic: iterates `vertical.field_names`, so a new vertical (e.g. NDA)
works with zero edits here — its fields simply all default to `rule` unless
`ENSEMBLE_ROUTING`/`routing` says otherwise.

Re-attribution: `EnsembleExtractor.extract()` runs the Feature-1 `wrong_span`
post-pass (`extract.reattribute.reattribute_facts`) on the merged result by
default (`reattribute=True`). This is safe to default-on here specifically
because `ensemble` is a brand-new backend with no existing byte-identical-output
contract to preserve (unlike `rule`/`constrained`/etc., whose output must stay
unchanged) — set `reattribute=False` to disable.

Measured end-to-end (2026-07-12, 40-doc CUAD, `openai` gpt-5.5 as the LLM child;
counterparty/governing_law/termination_notice_days/auto_renewal -> openai,
effective_date/total_value -> rule): field-F1 0.692 (CI95 [0.640, 0.743]) vs
rule 0.676 — +0.016, paired permutation p=0.615, statistically indistinguishable
on the blended metric — but per-field-on-labeled every routed field lands at or
above its best parent: counterparty 0.949 (rule 0.49), governing_law 0.939,
auto_renewal 0.706, termination_notice_days 0.545, effective_date 0.938
(preserved via rule + fallback). Fallback fired on 14 fields (effective_date 9,
termination 5); re-attribution fired 0 times because gpt-5.5 cited perfectly
(wrong_span 0, source-accuracy 0.994) — its value awaits a `constrained` child,
whose standalone run had 16 wrong_span cases. The blended-F1 flatness is gold
sparsity + LLM invention on unlabeled docs (13 vs rule's 8), not routing
failure. The `constrained`-child pairing still needs the GPU/Ollama rig.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from pydantic import BaseModel

from contract_rag.config import Settings
from contract_rag.ir import DocumentIR

if TYPE_CHECKING:
    from contract_rag.verticals.base import Vertical

# Measured per-field winner (CLAUDE.md rule-vs-gpt-5.5/constrained tables).
DEFAULT_ROUTING: dict[str, str] = {
    "effective_date": "rule",
    "governing_law": "rule",
    "termination_notice_days": "rule",
    "counterparty": "constrained",
    "auto_renewal": "constrained",
    "total_value": "constrained",
}


def parse_routing_env(raw: str | None) -> dict[str, str]:
    """Parse `ENSEMBLE_ROUTING`-style `field=backend,field2=backend2` into a dict.
    `None`/blank -> `{}`. Entries missing `=` or with an empty field/backend name
    are silently skipped rather than raising, so a stray trailing comma or typo
    degrades to "no override for that entry" instead of crashing extraction."""
    if not raw:
        return {}
    out: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        field, _, backend = pair.partition("=")
        field, backend = field.strip(), backend.strip()
        if field and backend:
            out[field] = backend
    return out


def resolve_routing(vertical: Vertical, override: Mapping[str, str] | None = None) -> dict[str, str]:
    """Per-field backend name for every field of `vertical`. Fields not covered by
    `DEFAULT_ROUTING` (any non-contract vertical, or a future contract field)
    default to `"rule"`. `override` (env or explicit) wins per-field over the
    default; unrecognized field names in `override` are ignored (they can't route
    a field the vertical doesn't have)."""
    routing = {name: DEFAULT_ROUTING.get(name, "rule") for name in vertical.field_names}
    for name, backend in (override or {}).items():
        if name in routing:
            routing[name] = backend
    return routing


def _build_child(backend: str, settings: Settings, vertical: Vertical) -> object:
    """Build one named backend the same way `get_extractor` would, so any gating
    (`assert_backend_allowed`) that backend requires still applies — e.g. routing
    a field to `openai` still needs `ALLOW_EXTERNAL_LLM=true`."""
    from contract_rag.config import assert_backend_allowed
    from contract_rag.extract.extractor import get_extractor

    child_settings = settings.model_copy(update={"extract_backend": backend})
    assert_backend_allowed(child_settings)
    return get_extractor(child_settings, vertical)


class EnsembleExtractor:
    """`EXTRACT_BACKEND=ensemble` — see module docstring."""

    def __init__(
        self,
        settings: Settings,
        vertical: Vertical | None = None,
        children: Mapping[str, object] | None = None,
        routing: Mapping[str, str] | None = None,
        reattribute: bool = True,
    ) -> None:
        from contract_rag.verticals.registry import get_vertical_for

        v = vertical or get_vertical_for(settings)
        self._vertical = v
        override = dict(routing) if routing is not None else parse_routing_env(settings.ensemble_routing)
        self._routing = resolve_routing(v, override)
        if children is not None:
            self._children: dict[str, object] = dict(children)
        else:
            needed = sorted(set(self._routing.values()))
            self._children = {b: _build_child(b, settings, v) for b in needed}
        self._reattribute = reattribute
        self.last_fallbacks: dict[str, int] = {}
        self.last_reattributions: dict[str, int] = {}

    def extract(self, ir: DocumentIR) -> BaseModel:
        v = self._vertical
        cache: dict[str, BaseModel] = {}

        def facts_for(backend: str) -> BaseModel:
            if backend not in cache:
                cache[backend] = self._children[backend].extract(ir)
            return cache[backend]

        fallbacks: dict[str, int] = {}
        field_values: dict[str, object] = {}
        for name in v.field_names:
            backend = self._routing.get(name, "rule")
            clause = getattr(facts_for(backend), name)
            if not clause.value:
                for other in sorted(self._children):
                    if other == backend:
                        continue
                    other_clause = getattr(facts_for(other), name)
                    if other_clause.value:
                        clause = other_clause
                        fallbacks[name] = 1
                        break
            field_values[name] = clause

        facts = v.facts_model(**field_values)
        self.last_fallbacks = fallbacks
        if self._reattribute:
            from contract_rag.extract.reattribute import reattribute_facts

            facts, repairs = reattribute_facts(facts, ir, v)
            self.last_reattributions = repairs
        else:
            self.last_reattributions = {}
        return facts
