from unittest.mock import patch
import httpx
import pytest
from agent.providers import list_atp_models, create_chat_model


def test_discovery_and_redacted_errors():
    with patch("agent.providers.httpx.get", return_value=httpx.Response(200, json={"data": [{"id": "b"}, {"id": "a"}, {"id": "a"}]})):
        assert list_atp_models("secret") == ["a", "b"]
    with patch("agent.providers.httpx.get", return_value=httpx.Response(401, text="secret")):
        with pytest.raises(ValueError, match="HTTP 401") as error:
            list_atp_models("secret")
        assert "secret" not in str(error.value)


def test_atp_uses_chat_completions():
    model = create_chat_model("ATP", "test-only", "claude-haiku-4-5")
    assert model.openai_api_base == "https://api.atptoken.ai/v1"
    assert model.use_responses_api is False
    assert model.model_name == "claude-haiku-4-5"


def test_atp_ui_discovers_models_without_network():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    with patch("ui.agent_page.get_api_key", return_value="test-only"), patch(
        "ui.agent_page.list_atp_models", return_value=["claude-haiku-4-5", "gpt-5.4"]
    ):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
        assert next(b for b in app.button if b.label == "送出").disabled
        next(b for b in app.button if b.label == "查詢／更新 ATP 模型").click().run()
        assert not app.exception
        assert next(s for s in app.selectbox if s.label == "ATP 模型").value == "claude-haiku-4-5"
        assert not next(b for b in app.button if b.label == "送出").disabled
