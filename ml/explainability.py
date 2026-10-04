from dataclasses import dataclass
from io import BytesIO
from threading import RLock
import numpy as np
import pandas as pd
from config import MAX_SHAP_CELLS, MAX_SHAP_ROWS, RANDOM_STATE

PLOT_LOCK = RLock()


@dataclass
class ExplanationResult:
    values: np.ndarray
    base_value: float
    data: pd.DataFrame
    importance: pd.DataFrame
    scale: str

    def sample_table(self, sample_idx):
        if not 0 <= sample_idx < len(self.data):
            raise ValueError(f"樣本位置須介於 0 與 {len(self.data) - 1}。")
        values = self.values[sample_idx]
        frame = pd.DataFrame({
            "feature": self.data.columns,
            "feature_value": self.data.iloc[sample_idx].values,
            "shap_value": values,
            "impact": np.where(values > 0, "positive", np.where(values < 0, "negative", "neutral")),
            "abs_shap": np.abs(values),
        })
        return frame.sort_values("abs_shap", ascending=False)


def explain_model(result):
    import shap
    if result.explanation is not None:
        return result.explanation
    X = result.X_test
    if len(X) > MAX_SHAP_ROWS:
        X = X.sample(MAX_SHAP_ROWS, random_state=RANDOM_STATE)
    preprocessor = result.pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(X)
    if transformed.shape[0] * transformed.shape[1] > MAX_SHAP_CELLS:
        raise ValueError("編碼後特徵過多，請移除高基數欄位後重新訓練再計算 SHAP。")
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    model = result.pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(transformed)
    positive_idx = list(model.classes_).index(1)
    if isinstance(values, list):
        values = values[positive_idx]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, positive_idx]
    base = np.asarray(explainer.expected_value).reshape(-1)
    base_value = float(base[positive_idx] if len(base) > 1 else base[0])
    data = pd.DataFrame(transformed, columns=preprocessor.get_feature_names_out(), index=X.index)
    if values.shape != data.shape:
        raise ValueError("SHAP 回傳維度與特徵不符，請檢查套件版本。")
    importance = pd.DataFrame({
        "feature": data.columns, "importance": np.abs(values).mean(axis=0),
    }).sort_values("importance", ascending=False)
    result.explanation = ExplanationResult(
        values, base_value, data, importance,
        "probability" if result.model_name == "Random Forest" else "log-odds",
    )
    return result.explanation


def waterfall_png(explanation, sample_idx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    explanation.sample_table(sample_idx)
    with PLOT_LOCK:
        try:
            item = shap.Explanation(
                values=explanation.values[sample_idx], base_values=explanation.base_value,
                data=explanation.data.iloc[sample_idx].values,
                feature_names=explanation.data.columns.tolist(),
            )
            shap.plots.waterfall(item, max_display=15, show=False)
            output = BytesIO()
            plt.gcf().savefig(output, format="png", dpi=160, bbox_inches="tight")
            return output.getvalue()
        finally:
            plt.close("all")
