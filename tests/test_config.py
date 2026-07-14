import os

import pytest

from contract_rag.config import (
    Settings,
    _is_local_endpoint,
    _load_dotenv,
    assert_backend_allowed,
    get_settings,
)


def test_defaults_are_safe_and_offline():
    s = Settings()
    assert s.extract_backend == "fake"
    assert s.allow_external_llm is False


def test_openai_backend_blocked_without_flag():
    s = Settings(extract_backend="openai", allow_external_llm=False)
    with pytest.raises(PermissionError):
        assert_backend_allowed(s)


def test_openai_backend_allowed_with_flag():
    s = Settings(extract_backend="openai", allow_external_llm=True)
    assert_backend_allowed(s)  # must not raise


def test_get_settings_reads_environment(monkeypatch):
    monkeypatch.setenv("EXTRACT_BACKEND", "openai")
    monkeypatch.setenv("ALLOW_EXTERNAL_LLM", "true")
    s = get_settings()
    assert s.extract_backend == "openai"
    assert s.allow_external_llm is True


def test_vlm_endpoint_defaults_none_and_threshold_default():
    s = Settings()
    assert s.vlm_endpoint is None
    assert s.text_coverage_threshold == 0.8


def test_get_settings_reads_vlm_endpoint(monkeypatch):
    monkeypatch.setenv("VLM_ENDPOINT", "http://gpu:10000/v1")
    monkeypatch.setenv("TEXT_COVERAGE_THRESHOLD", "0.7")
    s = get_settings()
    assert s.vlm_endpoint == "http://gpu:10000/v1"
    assert s.text_coverage_threshold == 0.7


def test_local_and_mlx_endpoint_model_defaults():
    s = Settings()
    assert s.local_endpoint == "http://localhost:8000/v1"   # vLLM/SGLang
    assert s.local_model == "Qwen3-14B"
    assert s.mlx_endpoint == "http://localhost:11434/v1"     # Ollama OpenAI-compatible
    assert s.mlx_model == "qwen3"


def test_get_settings_reads_local_and_mlx_env(monkeypatch):
    monkeypatch.setenv("LOCAL_ENDPOINT", "http://gpu:8000/v1")
    monkeypatch.setenv("LOCAL_MODEL", "Qwen3-32B")
    monkeypatch.setenv("MLX_ENDPOINT", "http://mac:11434/v1")
    monkeypatch.setenv("MLX_MODEL", "qwen3:8b")
    s = get_settings()
    assert s.local_endpoint == "http://gpu:8000/v1"
    assert s.local_model == "Qwen3-32B"
    assert s.mlx_endpoint == "http://mac:11434/v1"
    assert s.mlx_model == "qwen3:8b"


def test_local_and_mlx_backends_need_no_external_llm_gate():
    # Privacy path: data never leaves the local endpoint, so no ALLOW_EXTERNAL_LLM gate.
    assert_backend_allowed(Settings(extract_backend="local"))   # must not raise
    assert_backend_allowed(Settings(extract_backend="mlx"))     # must not raise


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://localhost:8000/v1", True),
        ("http://127.0.0.1:8000/v1", True),
        ("http://[::1]:8000/v1", True),
        ("http://10.0.0.5:8000/v1", True),          # private LAN — lenient gate
        ("http://192.168.1.20:11434/v1", True),
        ("http://172.20.0.4:8000/v1", True),
        ("http://gpu-box.internal:8000/v1", True),
        ("http://169.254.1.1:8000/v1", True),        # link-local
        ("https://api.openai.com/v1", False),
        ("http://8.8.8.8:8000/v1", False),           # public IP
        ("", False),
        (None, False),
    ],
)
def test_is_local_endpoint_classifies_host(url, expected):
    assert _is_local_endpoint(url) is expected


def test_local_backend_with_remote_endpoint_blocked_without_flag():
    s = Settings(extract_backend="local", local_endpoint="https://gpu.example.com/v1")
    with pytest.raises(PermissionError):
        assert_backend_allowed(s)


def test_local_backend_with_remote_endpoint_allowed_with_flag():
    s = Settings(
        extract_backend="local",
        local_endpoint="https://gpu.example.com/v1",
        allow_external_llm=True,
    )
    assert_backend_allowed(s)  # must not raise


def test_mlx_backend_with_remote_endpoint_blocked_without_flag():
    s = Settings(extract_backend="mlx", mlx_endpoint="https://remote-ollama.example.com/v1")
    with pytest.raises(PermissionError):
        assert_backend_allowed(s)


def test_local_backend_with_private_lan_endpoint_allowed_without_flag():
    # Lenient gate: on-prem/private-LAN GPU boxes are within the isolation boundary.
    s = Settings(extract_backend="local", local_endpoint="http://10.1.2.3:8000/v1")
    assert_backend_allowed(s)  # must not raise


def test_constrained_backend_falls_back_to_local_endpoint_for_gate():
    # constrained_endpoint unset -> reuses local_endpoint, mirroring extract/constrained.py
    s = Settings(extract_backend="constrained", local_endpoint="https://gpu.example.com/v1")
    with pytest.raises(PermissionError):
        assert_backend_allowed(s)

    s_ok = Settings(extract_backend="constrained", local_endpoint="http://localhost:8000/v1")
    assert_backend_allowed(s_ok)  # must not raise


def test_constrained_backend_own_endpoint_takes_priority_over_local():
    s = Settings(
        extract_backend="constrained",
        constrained_endpoint="https://remote.example.com/v1",
        local_endpoint="http://localhost:8000/v1",
    )
    with pytest.raises(PermissionError):
        assert_backend_allowed(s)


def test_vlm_timeout_default_and_env(monkeypatch):
    assert Settings().vlm_timeout == 1200
    monkeypatch.setenv("VLM_TIMEOUT", "60")
    assert get_settings().vlm_timeout == 60


def test_max_upload_mb_default_and_env(monkeypatch):
    assert Settings().max_upload_mb == 25
    monkeypatch.setenv("MAX_UPLOAD_MB", "5")
    assert get_settings().max_upload_mb == 5


def test_load_dotenv_sets_missing_but_never_overrides_real_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "\n"
        "export OPENAI_API_KEY='sk-from-dotenv'\n"   # export prefix + quotes
        'EXTRACT_BACKEND="openai"\n'
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("EXTRACT_BACKEND", "rule")     # already set: real env must win

    _load_dotenv(env)

    assert os.environ["OPENAI_API_KEY"] == "sk-from-dotenv"   # loaded, quotes/export stripped
    assert os.environ["EXTRACT_BACKEND"] == "rule"            # NOT overridden


def test_redact_pii_defaults_true_and_reads_env(monkeypatch):
    monkeypatch.delenv("REDACT_PII", raising=False)
    assert get_settings().redact_pii is True
    monkeypatch.setenv("REDACT_PII", "false")
    assert get_settings().redact_pii is False


def test_load_dotenv_missing_file_is_noop(tmp_path):
    _load_dotenv(tmp_path / "does-not-exist.env")    # must not raise


def test_get_settings_loads_dotenv_from_cwd(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("EXTRACT_BACKEND=openai\nALLOW_EXTERNAL_LLM=true\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("EXTRACT_BACKEND", raising=False)
    monkeypatch.delenv("ALLOW_EXTERNAL_LLM", raising=False)

    s = get_settings()

    assert s.extract_backend == "openai"
    assert s.allow_external_llm is True


def test_vlm_model_prompt_rawdir_defaults(monkeypatch):
    for var in ("VLM_MODEL", "VLM_PROMPT", "VLM_RAW_DIR"):
        monkeypatch.delenv(var, raising=False)
    s = get_settings()
    assert s.vlm_model == "Unlimited-OCR"
    assert s.vlm_prompt == "Multi page parsing."
    assert s.vlm_raw_dir is None


def test_vlm_model_prompt_rawdir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("VLM_MODEL", "dots.ocr")
    monkeypatch.setenv("VLM_PROMPT", "Extract this page as markdown.")
    monkeypatch.setenv("VLM_RAW_DIR", str(tmp_path / "raw"))
    s = get_settings()
    assert s.vlm_model == "dots.ocr"
    assert s.vlm_prompt == "Extract this page as markdown."
    assert s.vlm_raw_dir == tmp_path / "raw"
