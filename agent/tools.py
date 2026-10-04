from typing import Literal
from langchain_core.tools import tool
from ml.models import MODEL_PARAMS
from ml.training import train_model
from ml.explainability import explain_model


def build_tools(workspace):
    """Bind tools to one session's workspace, never to process-wide user data."""

    @tool
    def inspect_dataset() -> dict:
        """Inspect uploaded dataset column names, types, missing counts and current ML selections."""
        with workspace.lock:
            df = workspace.dataset
            if df is None:
                return {"error": "請先在機器學習分頁上傳 CSV。"}
            return {
                "name": workspace.dataset_name, "rows": len(df),
                "columns": [{"name": str(c), "dtype": str(df[c].dtype),
                             "missing": int(df[c].isna().sum())} for c in df],
                "selection": workspace.selection,
            }

    @tool
    def available_models() -> dict:
        """List model hyperparameters: each entry is type, default, min/max or choices. Depth 0 means unlimited for RF/LightGBM."""
        return MODEL_PARAMS

    @tool
    def train_classifier(
        model_name: Literal["Random Forest", "LightGBM", "XGBoost"],
        target: str,
        params: dict | None = None,
        drop_cols: list[str] | None = None,
        categorical_cols: list[str] | None = None,
        test_size: float | None = None,
    ) -> dict:
        """Train only when requested. Target must be 0/1 or yes/no. Unspecified params use model defaults; omitted column/split options use current UI selections; [] clears a selection. Saves the new model only on success."""
        with workspace.lock:
            if workspace.dataset is None:
                return {"error": "請先在機器學習分頁上傳 CSV。"}
            selection = workspace.selection
            try:
                result = train_model(
                    workspace.dataset, target, model_name, params,
                    drop_cols if drop_cols is not None else selection.get("drop_cols", []),
                    categorical_cols if categorical_cols is not None else selection.get("categorical_cols", []),
                    test_size if test_size is not None else selection.get("test_size", 0.2),
                )
                workspace.result = result
                return result.summary()
            except Exception as error:
                return {"error": str(error), "previous_result_retained": workspace.result is not None}

    @tool
    def get_model_results() -> dict:
        """Get the latest trained model, actual hyperparameters, metrics and confusion matrix."""
        with workspace.lock:
            if workspace.result is None:
                return {"error": "目前沒有已訓練的模型。"}
            return workspace.result.summary()

    @tool
    def explain_predictions(sample_idx: int = 0) -> dict:
        """Compute SHAP for up to 200 held-out rows and explain a zero-based position in that subset. Returns positive-class scale, importance and sample contributions."""
        with workspace.lock:
            if workspace.result is None:
                return {"error": "請先訓練模型。"}
            try:
                explanation = explain_model(workspace.result)
                table = explanation.sample_table(sample_idx)
                return {
                    "run_id": workspace.result.run_id,
                    "model": workspace.result.model_name, "positive_class": 1,
                    "scale": explanation.scale, "base_value": explanation.base_value,
                    "explained_rows": len(explanation.data), "sample_idx": sample_idx,
                    "source_index": str(explanation.data.index[sample_idx]),
                    "output_value": float(explanation.base_value + explanation.values[sample_idx].sum()),
                    "top_features": explanation.importance.head(10).to_dict("records"),
                    "sample_contributions": table.head(10).to_dict("records"),
                }
            except Exception as error:
                return {"error": str(error)}

    return [inspect_dataset, available_models, train_classifier, get_model_results, explain_predictions]
