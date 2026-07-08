from pathlib import Path

CI = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"


def test_ci_runs_pytest_with_extras():
    text = CI.read_text()
    assert "uv run" in text and "pytest" in text
    # gated app/api tests must actually run in CI
    assert "--extra api" in text


def test_ci_builds_docker_image():
    assert "docker build" in CI.read_text()
