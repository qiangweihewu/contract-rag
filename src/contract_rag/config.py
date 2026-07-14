from __future__ import annotations

import ipaddress
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field

Backend = Literal["fake", "rule", "openai", "local", "mlx", "constrained", "ensemble"]


class Settings(BaseModel):
    extract_backend: Backend = "fake"
    vertical: str = "contract"
    allow_external_llm: bool = False
    openai_model: str = "gpt-4o"
    local_endpoint: str = "http://localhost:8000/v1"   # vLLM / SGLang
    local_model: str = "Qwen3-14B"
    mlx_endpoint: str = "http://localhost:11434/v1"     # Ollama OpenAI-compatible
    mlx_model: str = "qwen3"
    # constrained backend: any OpenAI-compatible server with json_schema structured
    # output (vLLM / SGLang / Ollama >= 0.5). None → reuse local_endpoint/local_model,
    # so an existing LOCAL_ENDPOINT setup needs no extra config.
    constrained_endpoint: str | None = None
    constrained_model: str | None = None
    # ensemble backend: per-field routing override, "field=backend,field2=backend2"
    # (see extract/ensemble.py:parse_routing_env). None -> DEFAULT_ROUTING only.
    ensemble_routing: str | None = None
    vlm_endpoint: str | None = None
    vlm_model: str = "Unlimited-OCR"
    vlm_prompt: str = "Multi page parsing."
    vlm_raw_dir: Path | None = None
    vlm_timeout: int = 1200
    # cap per-page generation; None omits the field (server default). Guards against
    # OCR-VLM repetition loops on degraded/noisy pages (measured: 100k+ junk tokens,
    # 20-30 min/page uncapped); real page markdown is 1-3k tokens so a generous cap
    # cannot alter a non-looping parse.
    vlm_max_tokens: int | None = None
    franken_bin: str | None = None  # path/name of the focr CPU-only OCR binary
    text_coverage_threshold: float = 0.8
    redact_pii: bool = True
    max_upload_mb: int = 25
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    golden_set_dir: Path = Field(default_factory=lambda: Path("golden_set"))
    cuad_dir: Path = Field(default_factory=lambda: Path("cuad"))
    kleister_dir: Path = Field(default_factory=lambda: Path("kleister-nda"))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_dotenv(path: Path | str = ".env") -> None:
    """Minimal, dependency-free `.env` loader: `KEY=VALUE` lines into `os.environ`.
    Never overrides a variable already set in the real environment, so inline/exported
    vars still win. Supports `#` comments, blank lines, an `export ` prefix, and
    surrounding single/double quotes."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def get_settings() -> Settings:
    _load_dotenv()
    return Settings(
        extract_backend=os.environ.get("EXTRACT_BACKEND", "fake"),  # type: ignore[arg-type]
        vertical=os.environ.get("VERTICAL", "contract"),
        allow_external_llm=_env_bool("ALLOW_EXTERNAL_LLM", False),
        openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        local_endpoint=os.environ.get("LOCAL_ENDPOINT", "http://localhost:8000/v1"),
        local_model=os.environ.get("LOCAL_MODEL", "Qwen3-14B"),
        mlx_endpoint=os.environ.get("MLX_ENDPOINT", "http://localhost:11434/v1"),
        mlx_model=os.environ.get("MLX_MODEL", "qwen3"),
        constrained_endpoint=os.environ.get("CONSTRAINED_ENDPOINT"),
        constrained_model=os.environ.get("CONSTRAINED_MODEL"),
        ensemble_routing=os.environ.get("ENSEMBLE_ROUTING"),
        data_dir=Path(os.environ.get("DATA_DIR", "data")),
        golden_set_dir=Path(os.environ.get("GOLDEN_SET_DIR", "golden_set")),
        cuad_dir=Path(os.environ.get("CUAD_DIR", "cuad")),
        kleister_dir=Path(os.environ.get("KLEISTER_DIR", "kleister-nda")),
        vlm_endpoint=os.environ.get("VLM_ENDPOINT"),
        vlm_model=os.environ.get("VLM_MODEL", "Unlimited-OCR"),
        vlm_prompt=os.environ.get("VLM_PROMPT", "Multi page parsing."),
        vlm_raw_dir=(
            Path(os.environ["VLM_RAW_DIR"]) if os.environ.get("VLM_RAW_DIR") else None
        ),
        vlm_timeout=int(os.environ.get("VLM_TIMEOUT", "1200")),
        vlm_max_tokens=(
            int(os.environ["VLM_MAX_TOKENS"]) if os.environ.get("VLM_MAX_TOKENS") else None
        ),
        franken_bin=os.environ.get("FRANKEN_BIN"),
        text_coverage_threshold=float(os.environ.get("TEXT_COVERAGE_THRESHOLD", "0.8")),
        redact_pii=_env_bool("REDACT_PII", True),
        max_upload_mb=int(os.environ.get("MAX_UPLOAD_MB", "25")),
    )


def _is_local_endpoint(url: str | None) -> bool:
    """True when `url`'s host is loopback or private-LAN/on-prem — lenient: an
    internal network is treated as inside the data-isolation boundary, not just
    literal localhost, since on-device backends (local/mlx/constrained) are often
    reached over a LAN GPU box. Unparseable/empty host fails closed (False)."""
    if not url:
        return False
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    if host == "localhost" or host.endswith((".local", ".internal", ".lan")):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local


def _resolved_local_endpoint(settings: Settings) -> str | None:
    """The endpoint a local/mlx/constrained backend actually talks to (None for
    other backends). `constrained` reuses `local_endpoint` when unset, mirroring
    `extract/constrained.py`'s own fallback."""
    if settings.extract_backend == "local":
        return settings.local_endpoint
    if settings.extract_backend == "mlx":
        return settings.mlx_endpoint
    if settings.extract_backend == "constrained":
        return settings.constrained_endpoint or settings.local_endpoint
    return None


def assert_backend_allowed(settings: Settings) -> None:
    if settings.extract_backend == "openai" and not settings.allow_external_llm:
        raise PermissionError(
            "openai extraction backend requires ALLOW_EXTERNAL_LLM=true; "
            "refusing to send documents to a third party by default."
        )
    endpoint = _resolved_local_endpoint(settings)
    if endpoint is not None and not settings.allow_external_llm and not _is_local_endpoint(endpoint):
        raise PermissionError(
            f"{settings.extract_backend} extraction backend is configured with a "
            f"non-local endpoint ({endpoint!r}); refusing to send documents off-network "
            "without ALLOW_EXTERNAL_LLM=true."
        )
