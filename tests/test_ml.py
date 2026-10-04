import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from ml.models import MODEL_PARAMS, build_model, resolve_params
from ml.preprocessing import prepare_data
from ml.training import train_model
from ml.explainability import explain_model, waterfall_png
from state import init_state, set_dataset


@pytest.fixture
def dataset():
    X, y = make_classification(n_samples=120, n_features=5, n_informative=3, random_state=17)
    df = pd.DataFrame(X, columns=["a", "b", "c", "d", "e"])
    df["category"] = np.where(df.a > 0, "high", "low")
    df.loc[0:5, "a"] = np.nan
    df.loc[6:10, "category"] = None
    df["empty"] = np.nan
    df["target"] = np.where(y, "yes", "no")
    return df


@pytest.mark.parametrize("name", MODEL_PARAMS)
def test_training_and_shap_for_all_models(dataset, name):
    original = dataset.copy(deep=True)
    result = train_model(dataset, "target", name, {"n_estimators": 20})
    assert result.pipeline.named_steps["model"].n_estimators == 20
    assert 0 <= result.evaluation["metrics"]["ROC-AUC"] <= 1
    assert sum(map(sum, result.evaluation["confusion_matrix"])) == len(result.X_test)
    explanation = explain_model(result)
    assert explanation.values.shape == explanation.data.shape
    assert explanation is explain_model(result)
    reconstructed = explanation.base_value + explanation.values.sum(axis=1)
    if name == "Random Forest":
        expected = result.pipeline.predict_proba(result.X_test)[:, 1]
    else:
        probabilities = result.pipeline.predict_proba(result.X_test)[:, 1]
        expected = np.log(probabilities / (1 - probabilities))
    np.testing.assert_allclose(reconstructed, expected, rtol=1e-4, atol=1e-4)
    assert waterfall_png(explanation, 0).startswith(b"\x89PNG")
    pd.testing.assert_frame_equal(dataset, original)


@pytest.mark.parametrize("target", [[0.5, 1] * 60, [0, 1, 2] * 40, [0] * 120, [None] * 120])
def test_reject_invalid_targets(dataset, target):
    dataset["target"] = target
    with pytest.raises(ValueError, match="二元分類"):
        train_model(dataset, "target")


def test_conflicting_columns_and_bad_params(dataset):
    with pytest.raises(ValueError):
        train_model(dataset, "target", drop_cols=["target"])
    with pytest.raises(ValueError):
        train_model(dataset, "target", drop_cols=["a"], categorical_cols=["a"])
    with pytest.raises(ValueError):
        resolve_params("LightGBM", {"n_estimators": -1})
    with pytest.raises(ValueError):
        resolve_params("XGBoost", {"learning_rate": float("nan")})
    with pytest.raises(ValueError):
        resolve_params("Random Forest", {"unknown": 3})
    assert build_model("LightGBM", {"subsample": 0.5}).subsample_freq == 1


def test_preprocessing_fits_training_only_and_handles_unseen_category():
    df = pd.DataFrame({"number": [1., 3., 999., np.nan], "cat": ["a", "a", "b", None], "y": [0, 1, 0, 1]})
    X, _, preprocessor = prepare_data(df, "y")
    preprocessor.fit(X.iloc[:2])
    assert preprocessor.named_transformers_["num"].statistics_[0] == 2
    encoded = preprocessor.transform(X.iloc[2:])
    assert encoded.shape[0] == 2


def test_session_dataset_change_clears_stale_results(dataset):
    state = {}
    init_state(state)
    set_dataset(state, dataset, "first", "one.csv")
    state["workspace"].result = "trained"
    state["agent_messages"] = ["old"]
    set_dataset(state, dataset, "first", "one.csv")
    assert state["workspace"].result == "trained"
    set_dataset(state, dataset, "second", "two.csv")
    assert state["workspace"].result is None
    assert state["agent_messages"] == []
