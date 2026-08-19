from __future__ import annotations

import numpy as np
from constants import FLOOR_EPS, HIGH_RISK_THRESHOLD, LOW_RISK_CLIP, NEGLIGIBLE_RISK, RANDOM_STATE
from scipy.stats import wilcoxon
from sklearn.metrics import (
    brier_score_loss,
    fbeta_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    roc_auc_score,
)

N_BOOTSTRAP = 1000


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


def floor_mask(y_true: np.ndarray, eps: float = FLOOR_EPS) -> np.ndarray:
    return np.asarray(y_true, dtype=float) <= (NEGLIGIBLE_RISK + eps)


def _error_slice(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    if y_true.size == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "median_ae": float("nan")}
    abs_err = np.abs(y_true - y_pred)
    return {
        "n": int(y_true.size),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "median_ae": float(np.median(abs_err)),
    }


def floor_sliced_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, eps: float = FLOOR_EPS
) -> dict[str, object]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    floor = floor_mask(y_true, eps)
    return {
        "all": _error_slice(y_true, y_pred),
        "nonFloor": _error_slice(y_true[~floor], y_pred[~floor]),
        "floor": _error_slice(y_true[floor], y_pred[floor]),
        "nFloor": int(np.sum(floor)),
        "nNonFloor": int(np.sum(~floor)),
    }


def residual_movement_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, risk: np.ndarray
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    risk = np.asarray(risk, dtype=float)
    actual_move = y_true - risk
    predicted_move = y_pred - risk
    return {
        "residualMaeActual": float(np.mean(np.abs(actual_move))),
        "residualMaePredicted": float(np.mean(np.abs(predicted_move))),
        "residualMae": float(np.mean(np.abs(actual_move - predicted_move))),
    }


def bootstrap_mae_advantage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_persist: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_STATE,
    alpha: float = 0.05,
) -> dict[str, float | int | bool]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_persist = np.asarray(y_persist, dtype=float)
    n = int(y_true.size)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        deltas[i] = mean_absolute_error(y_true[idx], y_persist[idx]) - mean_absolute_error(
            y_true[idx], y_pred[idx]
        )
    point = float(mean_absolute_error(y_true, y_persist) - mean_absolute_error(y_true, y_pred))
    low, high = np.quantile(deltas, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "deltaMae": point,
        "ci95Low": float(low),
        "ci95High": float(high),
        "nBootstrap": int(n_bootstrap),
        "coversZero": bool(low <= 0.0 <= high),
    }


def paired_wilcoxon_abs_error(
    y_true: np.ndarray, y_pred: np.ndarray, y_persist: np.ndarray
) -> dict[str, float | int | str]:
    err_model = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    err_persist = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_persist, dtype=float))
    if not np.any(np.abs(err_model - err_persist) > 0):
        return {
            "statistic": 0.0,
            "pvalue": 1.0,
            "n": int(err_model.size),
            "note": "all paired abs errors identical",
        }
    try:
        result = wilcoxon(err_model, err_persist, alternative="two-sided", zero_method="wilcox")
    except ValueError as exc:
        return {
            "statistic": float("nan"),
            "pvalue": float("nan"),
            "n": int(err_model.size),
            "note": str(exc),
        }
    return {
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "n": int(err_model.size),
    }


def honest_system_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    risk: np.ndarray,
    y_persist: np.ndarray | None = None,
    compare_to_persistence: bool = False,
) -> dict[str, object]:
    report: dict[str, object] = {}
    report.update(floor_sliced_metrics(y_true, y_pred))
    report.update(residual_movement_metrics(y_true, y_pred, risk))
    if compare_to_persistence and y_persist is not None:
        report["maeAdvantageVsPersistence"] = bootstrap_mae_advantage(y_true, y_pred, y_persist)
        report["wilcoxonAbsError"] = paired_wilcoxon_abs_error(y_true, y_pred, y_persist)
    return report


def honest_metrics_bundle(
    y_true: np.ndarray,
    risk: np.ndarray,
    persist: np.ndarray,
    xgb_pred: np.ndarray,
    ens_pred: np.ndarray,
) -> dict[str, object]:
    floor = floor_mask(y_true)
    return {
        "floor": NEGLIGIBLE_RISK,
        "floorEps": FLOOR_EPS,
        "nTest": int(np.asarray(y_true).size),
        "nFloor": int(np.sum(floor)),
        "nNonFloor": int(np.sum(~floor)),
        "systems": {
            "persistence": honest_system_report(y_true, persist, risk),
            "xgboost": honest_system_report(
                y_true, xgb_pred, risk, persist, compare_to_persistence=True
            ),
            "ensemble": honest_system_report(
                y_true, ens_pred, risk, persist, compare_to_persistence=True
            ),
        },
    }


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


def classification_metrics(
    y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
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


def persistence_improvement(
    y_true: np.ndarray, y_model: np.ndarray, y_persist: np.ndarray
) -> dict[str, float]:
    mae_model = float(mean_absolute_error(y_true, y_model))
    mae_persist = float(mean_absolute_error(y_true, y_persist))
    esa_model = esa_loss(y_true, clip_for_esa(y_model))["esa_loss"]
    esa_persist = esa_loss(y_true, clip_for_esa(y_persist))["esa_loss"]
    return {
        "mae_model": mae_model,
        "mae_persist": mae_persist,
        "mae_improvement": mae_persist - mae_model,
        "esa_loss_model": esa_model,
        "esa_loss_persist": esa_persist,
        "esa_loss_improvement": esa_persist - esa_model,
        "beats_persistence": float(mae_model < mae_persist and esa_model < esa_persist),
    }


def reliability_bins(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int = 8
) -> list[dict[str, float]]:
    labels = (y_true >= HIGH_RISK_THRESHOLD).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float]] = []
    for i in range(n_bins):
        left, right = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (proba >= left) & (proba <= right)
        else:
            mask = (proba >= left) & (proba < right)
        if not np.any(mask):
            continue
        rows.append(
            {
                "mid": float((left + right) / 2),
                "predicted": float(np.mean(proba[mask])),
                "observed": float(np.mean(labels[mask])),
                "n": float(np.sum(mask)),
            }
        )
    return rows


def error_gallery(
    event_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    persist: np.ndarray,
) -> dict[str, list[dict[str, float | int]]]:
    residual = y_pred - y_true
    high = y_true >= HIGH_RISK_THRESHOLD
    pred_high = y_pred >= HIGH_RISK_THRESHOLD

    def pack(indices: np.ndarray) -> list[dict[str, float | int]]:
        rows = []
        for idx in indices:
            rows.append(
                {
                    "eventId": int(event_ids[idx]),
                    "actual": float(y_true[idx]),
                    "predicted": float(y_pred[idx]),
                    "persistence": float(persist[idx]),
                    "error": float(residual[idx]),
                }
            )
        return rows

    under = np.argsort(residual)[:5]
    over = np.argsort(residual)[::-1][:5]
    missed = np.where(high & ~pred_high)[0][:5]
    false_esc = np.where(~high & pred_high)[0][:5]
    return {
        "worstUnderpredictions": pack(under),
        "worstOverpredictions": pack(over),
        "missedHighRisk": pack(missed),
        "falseEscalations": pack(false_esc),
    }
