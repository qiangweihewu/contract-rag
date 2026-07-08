"""Server-side schema-constrained structured decoding backend (`constrained`).

The instructor TOOLS-mode path failed schema validation on 12/40 docs (30%) when
driven over Ollama — malformed nested JSON (an extra `ExtractedClause` wrapper)
and cited `block_id`s missing their `#` prefix. This backend replaces client-side
function calling with the server's own grammar-constrained decoder: the
OpenAI-compatible `response_format={"type": "json_schema", ...}` supported by
vLLM, SGLang, and Ollama (>= 0.5, one request shape for all three). The grammar
is built from the vertical's facts model, so eval/verify stay backend-agnostic.

Even a constrained server can't guarantee *semantics*, so a bounded post-parse
validates against the facts model and applies only the two documented repairs
(restore a dropped `#` prefix when the repaired id exists in the IR; unwrap one
accidental extra nesting level when the inner payload validates). A field that
still fails validation degrades to an empty clause — a miss, never an invention.
Repairs are counted on `last_repairs` (and an optional injected CounterStore),
mirroring `last_tokens`/`last_cost_usd` on the instructor backends.

Like `local`/`mlx`, this backend is UNGATED by ALLOW_EXTERNAL_LLM: the endpoint
is a local server and the document never leaves it.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from contract_rag.config import Settings
from contract_rag.ir import DocumentIR
from contract_rag.verticals.base import ExtractedClause

if TYPE_CHECKING:
    from contract_rag.obs.counters import CounterStore
    from contract_rag.verticals.base import Vertical

_CLAUSE_KEYS = frozenset(ExtractedClause.model_fields)
# JSON Schema keywords that grammar compilers (xgrammar/outlines/llguidance) commonly
# reject; dropping them is safe because Pydantic re-enforces them at validation time.
_UNSUPPORTED_KEYWORDS = ("default", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """`model_json_schema()` tightened for strict server-side decoding: every object
    gets `additionalProperties: false` and *all* properties required (strict decoders
    demand every key present; Pydantic defaults still apply on our side), and
    keywords grammar compilers reject (`default`, numeric bounds) are dropped."""
    schema = model.model_json_schema()
    _tighten(schema)
    return schema


def _tighten(node: Any) -> None:
    if isinstance(node, dict):
        # Only prune keywords on dicts that are themselves schemas, so a hypothetical
        # *field* named e.g. "default" inside a `properties` map is never clobbered.
        if "type" in node or "anyOf" in node or "$ref" in node or "allOf" in node:
            for kw in _UNSUPPORTED_KEYWORDS:
                node.pop(kw, None)
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"])
        for value in node.values():
            _tighten(value)
    elif isinstance(node, list):
        for value in node:
            _tighten(value)


def _openai_client(base_url: str, api_key: str = "local"):
    """Plain OpenAI SDK client (no instructor — the *server* constrains the output).
    Same trust_env=False posture as `_instructor_client`: a local endpoint must be
    reached directly, never through a system proxy."""
    import httpx
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key, http_client=httpx.Client(trust_env=False))


def _parse_payload(content: str, repairs: Counter[str]) -> dict[str, Any]:
    """Parse the model reply into a dict. Tolerates a markdown code fence (counted);
    anything else unparseable counts `payload_invalid` and yields {} → all-empty facts."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        stripped = (content or "").strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[-1]
            stripped = stripped.rsplit("```", 1)[0]
            try:
                data = json.loads(stripped)
                repairs["fence_strip"] += 1
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                pass
        repairs["payload_invalid"] += 1
        return {}
    if not isinstance(data, dict):
        repairs["payload_invalid"] += 1
        return {}
    return data


def _fix_block_id(clause: ExtractedClause, block_ids: frozenset[str],
                  repairs: Counter[str]) -> ExtractedClause:
    """Repair (a): the documented mlx source-accuracy bug — the model drops the `#`
    prefix on a cited block_id. Only applied when the repaired id actually exists in
    the IR, so it can never *create* an attribution."""
    sbid = clause.source_block_id
    if sbid and sbid not in block_ids and f"#{sbid}" in block_ids:
        repairs["hash_prefix"] += 1
        return clause.model_copy(update={"source_block_id": f"#{sbid}"})
    return clause


def _coerce_clause(raw: Any, block_ids: frozenset[str],
                   repairs: Counter[str]) -> ExtractedClause:
    """Validate one field payload into an ExtractedClause with bounded repairs.
    On failure after repairs, return an empty clause (miss, not invention)."""
    if raw is None:
        return ExtractedClause()
    if not isinstance(raw, dict):
        repairs["field_dropped"] += 1
        return ExtractedClause()
    # Repair (b), checked *before* plain validation: the documented extra nesting
    # level, e.g. {"ExtractedClause": {...}}. Pydantic ignores extra keys, so plain
    # validation would silently accept the wrapper as an all-default (empty) clause
    # and lose a real extraction — the wrapper shape must win when the inner
    # payload looks like, and validates as, a clause.
    if len(raw) == 1:
        ((_key, inner),) = raw.items()
        if isinstance(inner, dict) and set(inner) & _CLAUSE_KEYS:
            try:
                clause = ExtractedClause.model_validate(inner)
            except ValidationError:
                pass
            else:
                repairs["unwrap_field"] += 1
                return _fix_block_id(clause, block_ids, repairs)
    try:
        clause = ExtractedClause.model_validate(raw)
    except ValidationError:
        repairs["field_dropped"] += 1
        return ExtractedClause()
    return _fix_block_id(clause, block_ids, repairs)


def validate_with_repairs(
    data: dict[str, Any], facts_model: type[BaseModel], block_ids: frozenset[str]
) -> tuple[BaseModel, dict[str, int]]:
    """Build a facts_model instance from raw decoded JSON, applying only the bounded,
    principled repairs documented above and counting each one. Field-by-field, so one
    malformed field degrades to a single empty clause instead of sinking the doc —
    the failure mode that cost 12/40 docs under instructor TOOLS mode."""
    repairs: Counter[str] = Counter()
    field_names = set(facts_model.model_fields)
    # Root-level unwrap: the whole facts object nested under one stray key,
    # e.g. {"ContractFacts": {...}}. Same principle as the per-field unwrap.
    if len(data) == 1:
        ((key, inner),) = data.items()
        if key not in field_names and isinstance(inner, dict) and set(inner) & field_names:
            repairs["unwrap_root"] += 1
            data = inner
    fields = {
        name: _coerce_clause(data.get(name), block_ids, repairs) for name in field_names
    }
    return facts_model(**fields), dict(repairs)


class ConstrainedExtractor:
    """`EXTRACT_BACKEND=constrained` — see module docstring. `client` and `counters`
    are injectable seams so unit tests stay dep-free (no openai import, no network)."""

    def __init__(self, settings: Settings, vertical: Vertical | None = None,
                 client=None, counters: CounterStore | None = None):
        from contract_rag.verticals.registry import get_vertical_for

        v = vertical or get_vertical_for(settings)
        self._facts_model = v.facts_model
        self._prompt = v.extraction_prompt  # same prompt contract as the instructor backends
        self._model = settings.constrained_model or settings.local_model
        self._client = client or _openai_client(
            settings.constrained_endpoint or settings.local_endpoint
        )
        self._counters = counters
        self.last_tokens: int = 0
        self.last_cost_usd: float = 0.0
        self.last_repairs: dict[str, int] = {}

    def extract(self, ir: DocumentIR) -> BaseModel:
        from contract_rag.extract.extractor import build_context
        from contract_rag.obs.pricing import cost_for, estimate_tokens

        prompt = self._prompt + build_context(ir)
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": self._facts_model.__name__,
                    "strict": True,
                    "schema": strict_schema(self._facts_model),
                },
            },
        )
        content = completion.choices[0].message.content or ""
        self.last_tokens = int(
            getattr(getattr(completion, "usage", None), "total_tokens", 0) or 0
        ) or estimate_tokens(prompt)
        self.last_cost_usd = cost_for(self._model, self.last_tokens)

        repairs: Counter[str] = Counter()
        data = _parse_payload(content, repairs)
        block_ids = frozenset(b.block_id for b in ir.blocks)
        facts, validate_repairs = validate_with_repairs(data, self._facts_model, block_ids)
        repairs.update(validate_repairs)
        self.last_repairs = dict(repairs)
        if self._counters is not None:
            for name, count in self.last_repairs.items():
                self._counters.incr(f"extract.constrained.repair.{name}", count)
        return facts
