from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from constants import HIGH_RISK_THRESHOLD, NEGLIGIBLE_RISK, RANDOM_STATE
from evaluate import floor_mask, level_scoreboard_row
from train_classifier import TrainedClassifier
from train_regressor import numeric_columns
from xgboost import XGBClassifier

THRESHOLD_GRID = np.linspace(0.05, 0.95, 19)


def floor_labels(y: np.ndarray | pd.Series) -> np.ndarray:
    return floor_mask(np.asarray(y, dtype=float)).astype(int)


def non_floor_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[~floor_mask(frame["y"].to_numpy(dtype=float))].copy()


def fit_floor_classifier(train: pd.DataFrame) -> TrainedClassifier:
    numeric = numeric_columns(train)
    x = train[numeric].apply(pd.to_numeric, errors="coerce")
    y = floor_labels(train["y"])
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


def predict_floor_proba(trained: TrainedClassifier, frame: pd.DataFrame) -> np.ndarray:
    x = frame[trained.feature_names].apply(pd.to_numeric, errors="coerce")
    return np.asarray(trained.model.predict_proba(x)[:, 1], dtype=float)


def combine_floor_hurdle(
    proba: np.ndarray,
    reconstructed: np.ndarray,
    threshold: float,
    risk: np.ndarray | None = None,
    use_persist_guard: bool = False,
) -> np.ndarray:
    pred = np.asarray(reconstructed, dtype=float).copy()
    pred[np.asarray(proba, dtype=float) >= threshold] = NEGLIGIBLE_RISK
    if use_persist_guard:
        if risk is None:
            raise ValueError("persist guard requires snapshot risk")
        risk_arr = np.asarray(risk, dtype=float)
        guard = risk_arr >= HIGH_RISK_THRESHOLD
        pred = np.where(guard, risk_arr, pred)
    return pred


def floor_confusion(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float | int]:
    actual = floor_mask(np.asarray(y_true, dtype=float))
    predicted = np.asarray(proba, dtype=float) >= threshold
    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    tn = int(np.sum(~actual & ~predicted))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "threshold": float(threshold),
        "nFloor": int(np.sum(actual)),
        "nNonFloor": int(np.sum(~actual)),
        "predictedFloor": int(np.sum(predicted)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


@dataclass
class HurdleChoice:
    threshold: float
    use_persist_guard: bool
    validation: dict[str, float]


def choose_hurdle_policy(
    y: np.ndarray,
    proba: np.ndarray,
    reconstructed: np.ndarray,
    risk: np.ndarray,
) -> HurdleChoice:
    best: HurdleChoice | None = None
    best_key: tuple[float, float, float] | None = None
    for use_guard in (False, True):
        for threshold in THRESHOLD_GRID:
            pred = combine_floor_hurdle(
                proba,
                reconstructed,
                float(threshold),
                risk=risk,
                use_persist_guard=use_guard,
            )
            row = level_scoreboard_row(y, pred)
            key = (row["mae"], row["floorExcludedMae"], row["esaLoss"])
            if best_key is None or key < best_key:
                best_key = key
                best = HurdleChoice(
                    threshold=float(threshold),
                    use_persist_guard=use_guard,
                    validation=row,
                )
    if best is None:
        raise RuntimeError("hurdle threshold grid was empty")
    return best
