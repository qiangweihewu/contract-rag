from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

from contract_rag.config import Settings, assert_backend_allowed
from contract_rag.ir import DocumentIR

if TYPE_CHECKING:
    from contract_rag.verticals.base import Vertical


def build_context(ir: DocumentIR) -> str:
    return "\n\n".join(f"[{b.block_id}] {b.text}" for b in ir.blocks)


class Extractor(Protocol):
    def extract(self, ir: DocumentIR) -> BaseModel: ...


class FakeExtractor:
    def __init__(self, canned: BaseModel) -> None:
        self._canned = canned

    def extract(self, ir: DocumentIR) -> BaseModel:
        return self._canned


def _instructor_client(base_url: str, api_key: str = "local"):
    """instructor over an OpenAI-compatible HTTP server (vLLM/SGLang/Ollama).
    `api_key` is a placeholder local servers ignore but the OpenAI SDK requires.

    `trust_env=False` makes httpx ignore any HTTP(S)_PROXY / system proxy: a local
    endpoint must be reached directly. (A common footgun — with a VPN/system proxy
    set, the default client routes 127.0.0.1 through the proxy and the connection
    fails, even though curl works.)"""
    import httpx
    import instructor
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, http_client=httpx.Client(trust_env=False))
    return instructor.from_openai(client)


class _InstructorExtractor:
    """Schema-constrained extraction over any OpenAI-compatible instructor client.
    `instructor` forces the model's reply into `facts_model`, so schema validity
    holds regardless of which model/server (OpenAI, vLLM, Ollama) sits behind it.

    Captures per-call token usage on `last_tokens`/`last_cost_usd` so the obs cost
    dashboard goes live (S1-review carry-forward). Real usage when the client exposes
    `create_with_completion`; otherwise a prompt-length estimate.

    When `facts_model` or `prompt` are not given they default to the contract vertical
    so existing direct constructions stay valid."""

    def __init__(self, client, model: str, *, facts_model=None, prompt=None):
        if facts_model is None or prompt is None:
            from contract_rag.verticals.registry import get_vertical
            v = get_vertical("contract")
            facts_model = facts_model or v.facts_model
            prompt = prompt or v.extraction_prompt
        self._client = client
        self._model = model
        self._facts_model = facts_model
        self._prompt = prompt
        self.last_tokens: int = 0
        self.last_cost_usd: float = 0.0

    def extract(self, ir: DocumentIR) -> BaseModel:
        from contract_rag.obs.pricing import cost_for, estimate_tokens

        prompt = self._prompt + build_context(ir)
        messages = [{"role": "user", "content": prompt}]
        comps = self._client.chat.completions
        if hasattr(comps, "create_with_completion"):
            facts, completion = comps.create_with_completion(
                model=self._model, response_model=self._facts_model, messages=messages
            )
            self.last_tokens = int(
                getattr(getattr(completion, "usage", None), "total_tokens", 0) or 0
            )
        else:
            facts = comps.create(
                model=self._model, response_model=self._facts_model, messages=messages
            )
            self.last_tokens = estimate_tokens(prompt)
        self.last_cost_usd = cost_for(self._model, self.last_tokens)
        return facts


class OpenAIExtractor(_InstructorExtractor):
    def __init__(self, settings: Settings, vertical: Vertical | None = None):
        assert_backend_allowed(settings)
        import instructor
        from openai import OpenAI

        from contract_rag.verticals.registry import get_vertical_for

        v = vertical or get_vertical_for(settings)
        super().__init__(instructor.from_openai(OpenAI()), settings.openai_model,
                         facts_model=v.facts_model, prompt=v.extraction_prompt)


class LocalExtractor(_InstructorExtractor):
    """On-device extraction via an OpenAI-compatible local server (vLLM / SGLang).
    No ALLOW_EXTERNAL_LLM gate: the document never leaves the local endpoint, which
    is the whole point of this backend (privacy/production). `client` is an injectable
    seam so unit tests stay dep-free."""

    def __init__(self, settings: Settings, vertical: Vertical | None = None, client=None):
        from contract_rag.verticals.registry import get_vertical_for

        v = vertical or get_vertical_for(settings)
        super().__init__(
            client or _instructor_client(settings.local_endpoint),
            settings.local_model,
            facts_model=v.facts_model,
            prompt=v.extraction_prompt,
        )


class MLXExtractor(_InstructorExtractor):
    """Fully-offline on-Mac extraction via Ollama's OpenAI-compatible endpoint.
    Same no-gate privacy posture as LocalExtractor; differs only in endpoint/model."""

    def __init__(self, settings: Settings, vertical: Vertical | None = None, client=None):
        from contract_rag.verticals.registry import get_vertical_for

        v = vertical or get_vertical_for(settings)
        super().__init__(
            client or _instructor_client(settings.mlx_endpoint),
            settings.mlx_model,
            facts_model=v.facts_model,
            prompt=v.extraction_prompt,
        )


def get_extractor(settings: Settings, vertical: Vertical | None = None) -> Extractor:
    from contract_rag.verticals.registry import get_vertical_for

    vertical = vertical or get_vertical_for(settings)
    if settings.extract_backend == "openai":
        return OpenAIExtractor(settings, vertical)
    if settings.extract_backend == "rule":
        return vertical.rule_extractor
    if settings.extract_backend == "local":
        return LocalExtractor(settings, vertical)
    if settings.extract_backend == "mlx":
        return MLXExtractor(settings, vertical)
    if settings.extract_backend == "constrained":
        # Server-side json_schema structured decoding; ungated like local/mlx
        # (local endpoint, document never leaves). Lazy import keeps this module light.
        from contract_rag.extract.constrained import ConstrainedExtractor
        return ConstrainedExtractor(settings, vertical)
    if settings.extract_backend == "ensemble":
        # Field-level ensemble over two (default rule + constrained) child backends;
        # each child is built + gated the same way get_extractor would build it directly.
        from contract_rag.extract.ensemble import EnsembleExtractor
        return EnsembleExtractor(settings, vertical)
    if settings.extract_backend == "fake":
        return FakeExtractor(vertical.empty_facts())
    raise NotImplementedError(f"extract_backend={settings.extract_backend!r} not available")
