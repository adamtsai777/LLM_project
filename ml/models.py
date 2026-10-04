from numbers import Real

# type, default, minimum, maximum (or choices for select)
MODEL_PARAMS = {
    "Random Forest": {
        "n_estimators": ("int", 100, 10, 1000),
        "max_depth": ("int", 0, 0, 50),
        "min_samples_split": ("int", 2, 2, 100),
        "min_samples_leaf": ("int", 1, 1, 100),
        "max_features": ("select", "sqrt", ["sqrt", "log2", None]),
    },
    "LightGBM": {
        "n_estimators": ("int", 200, 10, 1000),
        "max_depth": ("int", 0, 0, 50),
        "learning_rate": ("float", 0.05, 0.001, 1.0),
        "num_leaves": ("int", 31, 2, 256),
        "min_child_samples": ("int", 20, 1, 100),
        "subsample": ("float", 0.8, 0.1, 1.0),
        "colsample_bytree": ("float", 0.8, 0.1, 1.0),
        "reg_alpha": ("float", 0.0, 0.0, 100.0),
        "reg_lambda": ("float", 0.0, 0.0, 100.0),
    },
    "XGBoost": {
        "n_estimators": ("int", 200, 10, 1000),
        "max_depth": ("int", 4, 1, 50),
        "learning_rate": ("float", 0.05, 0.001, 1.0),
        "min_child_weight": ("float", 1.0, 0.0, 100.0),
        "subsample": ("float", 0.8, 0.1, 1.0),
        "colsample_bytree": ("float", 0.8, 0.1, 1.0),
        "reg_alpha": ("float", 0.0, 0.0, 100.0),
        "reg_lambda": ("float", 1.0, 0.0, 100.0),
    },
}


def resolve_params(name, overrides=None):
    if name not in MODEL_PARAMS:
        raise ValueError("模型必須為 Random Forest、LightGBM 或 XGBoost。")
    specs = MODEL_PARAMS[name]
    params = {key: spec[1] for key, spec in specs.items()}
    overrides = overrides or {}
    unknown = set(overrides) - set(specs)
    if unknown:
        raise ValueError(f"不支援的超參數：{sorted(unknown)}")
    params.update(overrides)
    for key, value in params.items():
        spec = specs[key]
        if spec[0] == "select":
            valid = value in spec[2]
        else:
            valid = (isinstance(value, Real) and not isinstance(value, bool)
                     and spec[2] <= value <= spec[3])
            if spec[0] == "int":
                valid = valid and int(value) == value
        if not valid:
            raise ValueError(f"{key} 超出允許範圍：{spec[2:]}")
        if spec[0] == "int":
            params[key] = int(value)
    return params


def build_model(name, params=None):
    from config import RANDOM_STATE
    values = resolve_params(name, params)
    common = {"random_state": RANDOM_STATE, "n_jobs": 2}
    if name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        values["max_depth"] = values["max_depth"] or None
        return RandomForestClassifier(**values, **common)
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        values["max_depth"] = values["max_depth"] or -1
        return LGBMClassifier(**values, **common, subsample_freq=1, verbosity=-1)
    from xgboost import XGBClassifier
    return XGBClassifier(**values, **common, eval_metric="logloss", tree_method="hist")
