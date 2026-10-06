"""Headless Streamlit tests using streamlit.testing.v1.AppTest.

The app honours two environment variables meant for tests and demos without a key:
RAAPID_UI_EMBEDDER=hashing (fast embedder) and RAAPID_UI_USE_LLM=0 (retrieval-only).
"""

import pytest
from streamlit.testing.v1 import AppTest

from raapid_rag.config import PROJECT_ROOT

APP = PROJECT_ROOT / "src" / "raapid_rag" / "ui" / "app.py"


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("RAAPID_UI_EMBEDDER", "hashing")
    monkeypatch.setenv("RAAPID_UI_USE_LLM", "0")
    monkeypatch.setenv("RAAPID_UI_CACHE_DIR", str(tmp_path))
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    return at


def test_ui_renders_without_exceptions_and_shows_index_state(app):
    assert not app.exception
    sidebar_text = " ".join(m.value for m in app.sidebar.markdown)
    assert "chunks" in sidebar_text and "retrieval only" in sidebar_text.lower()
    assert app.text_input(key="question") is not None
    assert app.button(key="ask") is not None


def test_ui_answers_a_question_in_retrieval_only_mode(app):
    app.text_input(key="question").input("Are audio-only telehealth encounters acceptable?").run()
    app.button(key="ask").click().run()
    assert not app.exception
    page_text = " ".join(m.value for m in app.markdown)
    assert "retrieval_only" in page_text
    assert "needs human review" in page_text.lower()
    assert "policy_002" in page_text


def test_ui_reports_a_gate_refusal(app):
    app.text_input(key="question").input("zxqv plorth wibble").run()
    app.button(key="ask").click().run()
    assert not app.exception
    page_text = " ".join(m.value for m in app.markdown)
    assert "insufficient_context" in page_text
