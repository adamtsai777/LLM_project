from unittest.mock import patch
from pathlib import Path
import pandas as pd
from streamlit.testing.v1 import AppTest
from state import Workspace

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_missing_secrets_falls_back_to_environment():
    import os
    from config import get_api_key
    from streamlit.errors import StreamlitSecretNotFoundError
    with patch("streamlit.secrets") as secrets, patch.dict(os.environ, {"MISTRAL_API_KEY": "test-only"}):
        secrets.get.side_effect = StreamlitSecretNotFoundError("missing")
        assert get_api_key() == "test-only"


def test_no_api_key_does_not_block_ml():
    with patch("ui.agent_page.get_api_key", return_value=None):
        app = AppTest.from_file(APP_PATH).run(timeout=30)
        assert not app.exception
        assert len(app.tabs) == 2
        assert any("機器學習功能可獨立使用" in info.value for info in app.info)


def test_ui_training_survives_rerun_and_parameter_change():
    with patch("ui.agent_page.get_api_key", return_value=None):
        app = AppTest.from_file(APP_PATH)
        app.session_state["workspace"] = Workspace(
            dataset=pd.DataFrame({"x": list(range(80)), "target": [0, 1] * 40}),
            dataset_id="fixture", dataset_name="fixture.csv",
        )
        app.run(timeout=30)
        next(x for x in app.selectbox if x.label == "目標欄位").select("target").run()
        next(x for x in app.selectbox if x.label == "演算法").select("LightGBM").run()
        app.radio[0].set_value("手動調參").run()
        next(x for x in app.number_input if x.label == "n_estimators").set_value(10).run()
        next(x for x in app.button if x.label == "開始模型訓練").click().run(timeout=30)
        assert not app.exception
        result = app.session_state["workspace"].result
        assert result.model_name == "LightGBM"
        assert result.params["n_estimators"] == 10
        next(x for x in app.selectbox if x.label == "演算法").select("XGBoost").run()
        assert app.session_state["workspace"].result.run_id == result.run_id
        assert len(app.metric) == 5
        next(x for x in app.button if x.label == "計算 SHAP").click().run(timeout=30)
        assert not app.exception
        assert app.session_state["workspace"].result.explanation is not None
        assert not app.error
