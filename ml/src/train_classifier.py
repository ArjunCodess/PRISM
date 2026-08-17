from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from constants import FLOOR_MARGIN, HIGH_RISK_THRESHOLD, NEGLIGIBLE_RISK, RANDOM_STATE
from evaluate import esa_loss, clip_for_esa
from sklearn.metrics import fbeta_score
from train_regressor import model_matrix, numeric_columns
from xgboost import XGBClassifier


@dataclass
class TrainedClassifier:
    model: XGBClassifier
    feature_names: list[str]
    kind: str = "warning"


def _fit_binary(train: pd.DataFrame, labels: np.ndarray) -> TrainedClassifier:
    numeric = numeric_columns(train)
    x = model_matrix(train, numeric)
    pos = max(int(labels.sum()), 1)
    neg = max(int((1 - labels).sum()), 1)
    model = XGBClassifier(
        n_estimators=220,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=4,
        subsample=0.85,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=neg / pos,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        missing=np.nan,
    )
    model.fit(x, labels)
    return TrainedClassifier(model=model, feature_names=numeric)


def fit_warning_classifier(train: pd.DataFrame) -> TrainedClassifier:
    labels = (train["y"].to_numpy(dtype=float) >= HIGH_RISK_THRESHOLD).astype(int)
    trained = _fit_binary(train, labels)
    trained.kind = "warning"
    return trained


def fit_collapse_classifier(train: pd.DataFrame) -> TrainedClassifier:
    labels = (train["y"].to_numpy(dtype=float) <= NEGLIGIBLE_RISK + FLOOR_MARGIN).astype(int)
    trained = _fit_binary(train, labels)
    trained.kind = "collapse"
    return trained


def classifier_proba(trained: TrainedClassifier, frame: pd.DataFrame) -> np.ndarray:
    x = model_matrix(frame, trained.feature_names)
    return np.asarray(trained.model.predict_proba(x)[:, 1], dtype=float)


def tune_f2_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    labels = (np.asarray(y_true, dtype=float) >= HIGH_RISK_THRESHOLD).astype(int)
    scores = np.asarray(proba, dtype=float)
    best_threshold = 0.5
    best = -1.0
    for threshold in np.linspace(0.05, 0.8, 16):
        preds = (scores >= threshold).astype(int)
        score = float(fbeta_score(labels, preds, beta=2, zero_division=0))
        if score > best:
            best = score
            best_threshold = float(threshold)
    return best_threshold


def should_promote_high_risk(
    y_true: np.ndarray,
    point: np.ndarray,
    proba: np.ndarray,
    threshold: float,
) -> bool:
    y_true = np.asarray(y_true, dtype=float)
    point = np.asarray(point, dtype=float)
    promoted = point.copy()
    promote = (np.asarray(proba) >= threshold) & (promoted < HIGH_RISK_THRESHOLD)
    promoted[promote] = HIGH_RISK_THRESHOLD
    base = esa_loss(y_true, clip_for_esa(point))["esa_loss"]
    alt = esa_loss(y_true, clip_for_esa(promoted))["esa_loss"]
    mae_base = float(np.mean(np.abs(y_true - point)))
    mae_alt = float(np.mean(np.abs(y_true - promoted)))
    return bool(alt < base and (mae_alt - mae_base) <= 0.05)
