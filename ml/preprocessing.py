import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def prepare_data(df, target, drop_cols=None, categorical_cols=None):
    drop_cols = list(drop_cols or [])
    categorical_cols = list(categorical_cols or [])
    if df.empty or not df.columns.is_unique:
        raise ValueError("資料不可為空，且欄位名稱不得重複。")
    if target not in df.columns:
        raise ValueError("請指定資料中存在的目標欄位。")
    if target in drop_cols or target in categorical_cols:
        raise ValueError("目標欄位不能同時列為刪除欄位或類別特徵。")
    if set(drop_cols) & set(categorical_cols):
        raise ValueError("刪除欄位與類別特徵不可重疊。")
    if (set(drop_cols) | set(categorical_cols)) - set(df.columns):
        raise ValueError("指定的特徵欄位不存在。")
    raw_y = df[target]
    if pd.api.types.is_numeric_dtype(raw_y) or pd.api.types.is_bool_dtype(raw_y):
        y = pd.to_numeric(raw_y, errors="coerce")
    else:
        y = raw_y.astype("string").str.strip().str.lower().map(
            {"yes": 1, "no": 0, "1": 1, "0": 0}
        )
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("目前僅支援二元分類：目標須包含 0/1 或 yes/no，且不可缺值。")
    y = y.astype(int)
    if y.value_counts().min() < 2:
        raise ValueError("每個類別至少需要兩筆資料才能分層切分。")
    X = df.drop(columns=[target, *drop_cols]).copy()
    if X.shape[1] == 0:
        raise ValueError("至少需要一個特徵欄位。")
    numeric, categorical = [], []
    for col in X:
        if col not in categorical_cols and pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype(float).replace([np.inf, -np.inf], np.nan)
            numeric.append(col)
        else:
            X[col] = X[col].map(lambda value: str(value) if pd.notna(value) else np.nan)
            categorical.append(col)
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]), categorical),
    ])
    return X, y, preprocessor
