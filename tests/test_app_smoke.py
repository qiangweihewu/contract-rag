"""Fast smoke test of the Streamlit demo app via the official AppTest harness:
the script executes and renders its initial state without raising. Guards against
import errors and session_state-flow regressions. Gated on the `app` extra
(streamlit); the heavier full-run/ask flow lives in a manual scratch script."""
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("streamlit") is None,
    reason="needs the 'app' extra (streamlit)",
)

_APP = str(Path(__file__).resolve().parent.parent / "src" / "contract_rag" / "demo" / "app.py")


def test_app_initial_render_has_no_exception():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_APP, default_timeout=30).run()

    assert not at.exception, f"app raised on initial render: {at.exception}"
    assert any("Run pipeline" in m.value for m in at.info)   # the idle prompt
