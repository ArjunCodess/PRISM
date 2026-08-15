from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from constants import RANDOM_STATE


EXCLUDE = {"event_id", "y", "story", "c_object_type"}


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


def fit_xgboost(train: pd.DataFrame) -> TrainedRegressor:
    numeric = numeric_columns(train)
    x = train[numeric]
    y = train["y"].to_numpy(dtype=float)
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=6,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        missing=np.nan,
    )
    model.fit(x, y)
    return TrainedRegressor(model=model, feature_names=numeric, kind="xgboost")


def predict_model(trained: TrainedRegressor, frame: pd.DataFrame) -> np.ndarray:
    if trained.kind == "ridge":
        x = frame[trained.feature_names]
        return np.asarray(trained.model.predict(x), dtype=float)
    x = frame[trained.feature_names]
    return np.asarray(trained.model.predict(x), dtype=float)
