from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    fbeta_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    roc_auc_score,
)

from constants import HIGH_RISK_THRESHOLD, LOW_RISK_CLIP


def esa_loss(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    high = y_true >= HIGH_RISK_THRESHOLD
    pred_high = y_pred >= HIGH_RISK_THRESHOLD
    f2 = float(fbeta_score(high.astype(int), pred_high.astype(int), beta=2, zero_division=0))
    if high.any():
        mse_hr = float(mean_squared_error(y_true[high], y_pred[high]))
    else:
        mse_hr = 0.0
    loss = mse_hr / max(f2, 1e-6)
    return {"esa_loss": loss, "mse_hr": mse_hr, "f2": f2}


def clip_for_esa(y_pred: np.ndarray) -> np.ndarray:
    clipped = y_pred.copy()
    clipped[clipped < HIGH_RISK_THRESHOLD] = LOW_RISK_CLIP
    return clipped


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    abs_err = np.abs(y_true - y_pred)
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "median_ae": float(np.median(abs_err)),
        "within_0_5": float(np.mean(abs_err <= 0.5)),
        "within_1_0": float(np.mean(abs_err <= 1.0)),
    }
    high = y_true >= HIGH_RISK_THRESHOLD
    if high.any():
        metrics["mae_high_risk"] = float(mean_absolute_error(y_true[high], y_pred[high]))
    else:
        metrics["mae_high_risk"] = float("nan")
    metrics.update(esa_loss(y_true, clip_for_esa(y_pred)))
    return metrics


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    labels = (y_true >= HIGH_RISK_THRESHOLD).astype(int)
    preds = (proba >= threshold).astype(int)
    precision, recall, _ = precision_recall_curve(labels, proba)
    pr_auc = float(np.trapezoid(precision[::-1], recall[::-1])) if len(precision) > 1 else 0.0
    try:
        roc = float(roc_auc_score(labels, proba))
    except ValueError:
        roc = float("nan")
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc,
        "brier": float(brier_score_loss(labels, np.clip(proba, 0, 1))),
        "precision": float(np.mean(labels[preds == 1] == 1) if preds.any() else 0.0),
        "recall": float(np.mean(preds[labels == 1] == 1) if labels.any() else 0.0),
    }


def persistence_improvement(y_true: np.ndarray, y_model: np.ndarray, y_persist: np.ndarray) -> dict[str, float]:
    mae_model = float(mean_absolute_error(y_true, y_model))
    mae_persist = float(mean_absolute_error(y_true, y_persist))
    return {
        "mae_model": mae_model,
        "mae_persist": mae_persist,
        "mae_improvement": mae_persist - mae_model,
        "beats_persistence": float(mae_model < mae_persist),
    }
