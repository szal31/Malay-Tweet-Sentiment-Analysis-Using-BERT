import pytest
import streamlit as st
import transformers
from streamlit.testing.v1 import AppTest


@pytest.fixture
def app(monkeypatch):
    st.cache_resource.clear()

    def fake_pipeline(*args, **kwargs):  # no 423 MB download in tests
        return lambda text: [[{"label": "Negative", "score": 0.1}, {"label": "Neutral", "score": 0.2},
                              {"label": "Positive", "score": 0.7}]]
    monkeypatch.setattr(transformers, "pipeline", fake_pipeline)
    monkeypatch.setattr("transformers.pipelines.pipeline", fake_pipeline)  # where the lazy import resolves it
    return AppTest.from_file("../demo/streamlit_app.py", default_timeout=30)


def test_empty_page_invites_input(app):
    app.run()
    assert not app.exception
    assert app.info and "example" in app.info[0].value.lower()


def test_typed_text_shows_label_and_confidence(app):
    app.run()
    app.text_area[0].input("Best gila makanan kat sini!").run()
    text = " ".join(m.value for m in app.markdown)
    assert "Positive" in text and "70%" in text


def test_example_fills_the_box(app):
    app.run()
    app.button_group[0].set_value("Esok mesyuarat pukul 10 pagi kat bilik 3.").run()
    assert app.text_area[0].value == "Esok mesyuarat pukul 10 pagi kat bilik 3."
