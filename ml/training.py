from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from config import RANDOM_STATE
from ml.models import build_model, resolve_params
from ml.preprocessing import prepare_data
from ml.evaluation import evaluate


@dataclass
class TrainingResult:
    model_name: str
    params: dict
    target: str
    drop_cols: list
    categorical_cols: list
    test_size: float
    pipeline: Any
    X_test: Any
    y_test: Any
    evaluation: dict
    train_rows: int
    run_id: str = field(default_factory=lambda: uuid4().hex)
    explanation: Any = None

    def summary(self):
        return {
            "run_id": self.run_id, "model": self.model_name,
            "params": self.params, "target": self.target,
            "drop_cols": self.drop_cols, "categorical_cols": self.categorical_cols,
            "test_size": self.test_size, "random_state": RANDOM_STATE,
            "train_rows": self.train_rows, "test_rows": len(self.X_test),
            "positive_class": 1, **self.evaluation,
        }


def train_model(df, target, model_name="Random Forest", params=None,
                drop_cols=None, categorical_cols=None, test_size=0.2):
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("測試集比例必須介於 0.1 與 0.4。")
    params = resolve_params(model_name, params)
    X, y, preprocessor = prepare_data(df, target, drop_cols, categorical_cols)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )
    except ValueError as error:
        raise ValueError("資料不足以進行分層切分，請增加樣本或調整測試集比例。") from error
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError("訓練集與測試集都需要包含兩個類別，請增加少數類別樣本。")
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", build_model(model_name, params))])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    positive_idx = list(pipeline.classes_).index(1)
    probabilities = pipeline.predict_proba(X_test)[:, positive_idx]
    return TrainingResult(
        model_name, params, target, list(drop_cols or []), list(categorical_cols or []),
        test_size, pipeline, X_test, y_test,
        evaluate(y_test, predictions, probabilities), len(X_train),
    )
