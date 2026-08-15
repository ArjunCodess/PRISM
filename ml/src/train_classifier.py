from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from constants import HIGH_RISK_THRESHOLD, RANDOM_STATE
from train_regressor import numeric_columns


@dataclass
class TrainedClassifier:
    model: XGBClassifier
    feature_names: list[str]


def fit_warning_classifier(train: pd.DataFrame) -> TrainedClassifier:
    numeric = numeric_columns(train)
    x = train[numeric].apply(pd.to_numeric, errors="coerce")
    y = (train["y"].to_numpy(dtype=float) >= HIGH_RISK_THRESHOLD).astype(int)
    pos = max(int(y.sum()), 1)
    neg = max(int((1 - y).sum()), 1)
    model = XGBClassifier(
        n_estimators=180,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=neg / pos,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        missing=np.nan,
    )
    model.fit(x, y)
    return TrainedClassifier(model=model, feature_names=numeric)
