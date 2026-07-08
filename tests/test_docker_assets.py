from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_dockerfile_serves_the_api():
    df = (ROOT / "Dockerfile").read_text()
    assert "uvicorn" in df
    assert "contract_rag.api.app:app" in df
    assert "--extra api" in df  # the API extra is installed in the image


def test_dockerignore_excludes_data_and_caches():
    di = (ROOT / ".dockerignore").read_text()
    for pat in (".git", "data", ".ir_cache", "__pycache__"):
        assert pat in di
