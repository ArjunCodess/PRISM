from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from constants import RANDOM_STATE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

EXCLUDE = {"event_id", "y", "story", "c_object_type"}

MAE_XGB_PARAMS = {
    "objective": "reg:absoluteerror",
    "n_estimators": 400,
    "learning_rate": 0.04,
    "max_depth": 5,
    "min_child_weight": 8,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "eval_metric": "mae",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "missing": np.nan,
}


@dataclass
class TrainedRegressor:
    model: object
    feature_names: list[str]
    kind: str


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    cols = []
    for column in frame.columns:
        if column in EXCLUDE:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            cols.append(column)
    return cols


def model_matrix(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    aligned = frame.reindex(columns=feature_names)
    return aligned.apply(pd.to_numeric, errors="coerce")


def persistence_predict(frame: pd.DataFrame) -> np.ndarray:
    return frame["risk"].to_numpy(dtype=float)


def median_predict(train: pd.DataFrame, n: int) -> np.ndarray:
    return np.full(n, float(np.median(train["y"].to_numpy(dtype=float))))


def fit_ridge(train: pd.DataFrame) -> TrainedRegressor:
    numeric = numeric_columns(train)
    categorical = ["c_object_type"] if "c_object_type" in train.columns else []
    transformer = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            ),
        ]
    )
    model = Pipeline(
        [
            ("prep", transformer),
            ("ridge", Ridge(alpha=2.0)),
        ]
    )
    x = train[numeric + categorical]
    y = train["y"].to_numpy(dtype=float)
    model.fit(x, y)
    return TrainedRegressor(model=model, feature_names=numeric + categorical, kind="ridge")


def _xgb_regressor(**overrides: object) -> XGBRegressor:
    params = dict(MAE_XGB_PARAMS)
    params.update(overrides)
    return XGBRegressor(**params)


def fit_xgboost(
    train: pd.DataFrame,
    eval_frame: pd.DataFrame | None = None,
    *,
    residual: bool = False,
    params: dict[str, object] | None = None,
) -> TrainedRegressor:
    numeric = numeric_columns(train)
    x = model_matrix(train, numeric)
    y = train["y"].to_numpy(dtype=float)
    if residual:
        y = y - train["risk"].to_numpy(dtype=float)
    overrides: dict[str, object] = dict(params or {})
    if eval_frame is not None and len(eval_frame) >= 20:
        overrides["early_stopping_rounds"] = 40
    model = _xgb_regressor(**overrides)
    fit_kwargs: dict[str, object] = {}
    if eval_frame is not None and len(eval_frame) >= 20:
        x_eval = model_matrix(eval_frame, numeric)
        y_eval = eval_frame["y"].to_numpy(dtype=float)
        if residual:
            y_eval = y_eval - eval_frame["risk"].to_numpy(dtype=float)
        fit_kwargs["eval_set"] = [(x_eval, y_eval)]
        fit_kwargs["verbose"] = False
    model.fit(x, y, **fit_kwargs)
    return TrainedRegressor(model=model, feature_names=numeric, kind="xgboost")


def predict_model(trained: TrainedRegressor, frame: pd.DataFrame) -> np.ndarray:
    if trained.kind == "ridge":
        x = frame[trained.feature_names]
        return np.asarray(trained.model.predict(x), dtype=float)
    x = model_matrix(frame, trained.feature_names)
    return np.asarray(trained.model.predict(x), dtype=float)
