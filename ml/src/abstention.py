from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from constants import (
    ABSTENTION_DISAGREEMENT,
    FALSE_REASSURANCE_DEFINITION,
    FLOOR_MARGIN,
    HIGH_RISK_THRESHOLD,
    NEGLIGIBLE_RISK,
)
from evaluate import regression_metrics

REASON_CROSSES = "spread_crosses_threshold"
REASON_MISSING = "missing_critical_fields"
REASON_DISAGREEMENT = "ensemble_disagreement"
REASON_SUSPICIOUS_FLOOR = "floor_forecast_while_current_risk_elevated"
REASON_WARNING_CONFLICT = "warning_head_flags_a_safe_point_forecast"

REASON_TEXT = {
    REASON_CROSSES: (
        "the 90% conformal band crosses the ESA challenge class log10(Pc) ≥ −6"
    ),
    REASON_MISSING: "current reported risk or miss distance is missing",
    REASON_DISAGREEMENT: (
        f"bootstrap models disagree by more than {ABSTENTION_DISAGREEMENT} log-risk units"
    ),
    REASON_SUSPICIOUS_FLOOR: (
        "the model forecasts the dataset floor while today's report is still far from negligible"
    ),
    REASON_WARNING_CONFLICT: (
        "the high-risk warning head is elevated while the point forecast stays below −6"
    ),
}


@dataclass(frozen=True)
class AbstentionDecision:
    abstained: bool
    reasons: list[str]
    disagreement: float
    crosses_threshold: bool
    missing_critical: bool


def decide_abstention(
    ensemble_preds: np.ndarray,
    current_risk: float,
    miss_distance: float,
    disagreement_threshold: float = ABSTENTION_DISAGREEMENT,
    interval90: tuple[float, float] | None = None,
    point: float | None = None,
    warning_proba: float | None = None,
    warning_threshold: float = 0.15,
) -> AbstentionDecision:
    preds = np.asarray(ensemble_preds, dtype=float)
    if interval90 is None:
        lo, hi = np.quantile(preds, [0.05, 0.95])
    else:
        lo, hi = float(interval90[0]), float(interval90[1])
    disagreement = float(np.std(preds))
    crosses = bool((lo < HIGH_RISK_THRESHOLD <= hi) or (hi < HIGH_RISK_THRESHOLD <= lo))
    missing = bool(not np.isfinite(current_risk) or not np.isfinite(miss_distance))
    reasons: list[str] = []
    if crosses:
        reasons.append(REASON_CROSSES)
    if missing:
        reasons.append(REASON_MISSING)
    if disagreement > disagreement_threshold:
        reasons.append(REASON_DISAGREEMENT)
    if (
        point is not None
        and np.isfinite(point)
        and np.isfinite(current_risk)
        and point <= NEGLIGIBLE_RISK + FLOOR_MARGIN
        and current_risk > -10.0
    ):
        reasons.append(REASON_SUSPICIOUS_FLOOR)
    if (
        warning_proba is not None
        and point is not None
        and np.isfinite(warning_proba)
        and np.isfinite(point)
        and warning_proba >= warning_threshold
        and point < HIGH_RISK_THRESHOLD
    ):
        reasons.append(REASON_WARNING_CONFLICT)
    return AbstentionDecision(
        abstained=bool(reasons),
        reasons=reasons,
        disagreement=disagreement,
        crosses_threshold=crosses,
        missing_critical=missing,
    )


def abstain_mask(
    ensemble_matrix: np.ndarray,
    current_risk: np.ndarray,
    miss_distance: np.ndarray,
    disagreement_threshold: float = ABSTENTION_DISAGREEMENT,
    interval90: np.ndarray | None = None,
    point: np.ndarray | None = None,
    warning_proba: np.ndarray | None = None,
    warning_threshold: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if interval90 is None:
        lo = np.quantile(ensemble_matrix, 0.05, axis=1)
        hi = np.quantile(ensemble_matrix, 0.95, axis=1)
    else:
        lo = np.asarray(interval90[0], dtype=float)
        hi = np.asarray(interval90[1], dtype=float)
    disagreement = np.std(ensemble_matrix, axis=1)
    crosses = (lo < HIGH_RISK_THRESHOLD) & (hi >= HIGH_RISK_THRESHOLD)
    missing = ~np.isfinite(current_risk) | ~np.isfinite(miss_distance)
    abstained = crosses | missing | (disagreement > disagreement_threshold)
    if point is not None:
        values = np.asarray(point, dtype=float)
        suspicious = (
            (values <= NEGLIGIBLE_RISK + FLOOR_MARGIN)
            & np.isfinite(current_risk)
            & (current_risk > -10.0)
        )
        abstained = abstained | suspicious
    if warning_proba is not None and point is not None:
        conflict = (
            (np.asarray(warning_proba, dtype=float) >= warning_threshold)
            & (np.asarray(point, dtype=float) < HIGH_RISK_THRESHOLD)
        )
        abstained = abstained | conflict
    return abstained.astype(bool), disagreement, crosses


def _safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def selective_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    persist: np.ndarray,
    abstained: np.ndarray,
    proba: np.ndarray | None = None,
) -> dict[str, float | int | dict[str, float]]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    persist = np.asarray(persist, dtype=float)
    abstained = np.asarray(abstained, dtype=bool)
    accepted = ~abstained
    high = y_true >= HIGH_RISK_THRESHOLD
    pred_high = y_pred >= HIGH_RISK_THRESHOLD
    n = int(len(y_true))
    n_accepted = int(accepted.sum())
    n_abstained = int(abstained.sum())
    n_high = int(high.sum())
    false_reassurance = int((accepted & high & ~pred_high).sum())
    high_risk_captured = int((high & (abstained | pred_high)).sum())
    result: dict[str, float | int | dict[str, float]] = {
        "n": n,
        "nAccepted": n_accepted,
        "nAbstained": n_abstained,
        "coverage": float(n_accepted / n) if n else 0.0,
        "maeAll": _safe_mae(y_true, y_pred),
        "maeAccepted": _safe_mae(y_true[accepted], y_pred[accepted]),
        "maeAbstained": _safe_mae(y_true[abstained], y_pred[abstained]),
        "persistenceMaeAccepted": _safe_mae(y_true[accepted], persist[accepted]),
        "nHighRisk": n_high,
        "nHighRiskAbstained": int((high & abstained).sum()),
        "nHighRiskAccepted": int((high & accepted).sum()),
        "highRiskRecallAccepted": (
            float(((high & accepted) & pred_high).sum() / (high & accepted).sum())
            if (high & accepted).any()
            else float("nan")
        ),
        "highRiskCapture": float(high_risk_captured / n_high) if n_high else float("nan"),
        "falseReassurance": false_reassurance,
        "falseReassuranceDefinition": FALSE_REASSURANCE_DEFINITION,
        "falseReassuranceRate": float(false_reassurance / n_high) if n_high else 0.0,
        "accepted": (
            regression_metrics(y_true[accepted], y_pred[accepted]) if n_accepted else {}
        ),
        "abstained": (
            regression_metrics(y_true[abstained], y_pred[abstained]) if n_abstained else {}
        ),
    }
    if proba is not None:
        result["meanHighRiskProbabilityAccepted"] = (
            float(np.mean(proba[accepted])) if n_accepted else float("nan")
        )
    return result


def coverage_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    persist: np.ndarray,
    ensemble_matrix: np.ndarray,
    current_risk: np.ndarray,
    miss_distance: np.ndarray,
    thresholds: np.ndarray | None = None,
    interval90: np.ndarray | None = None,
) -> list[dict[str, float | int]]:
    if thresholds is None:
        thresholds = np.concatenate(
            ([0.05], np.linspace(0.25, 3.0, 12), [ABSTENTION_DISAGREEMENT])
        )
        thresholds = np.unique(np.round(thresholds, 4))
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        abstained, _, _ = abstain_mask(
            ensemble_matrix,
            current_risk,
            miss_distance,
            disagreement_threshold=float(threshold),
            interval90=interval90,
            point=y_pred,
        )
        metrics = selective_metrics(y_true, y_pred, persist, abstained)
        rows.append(
            {
                "disagreementThreshold": float(threshold),
                "coverage": float(metrics["coverage"]),
                "maeAccepted": float(metrics["maeAccepted"]),
                "maeAbstained": float(metrics["maeAbstained"]),
                "nAccepted": int(metrics["nAccepted"]),
                "nAbstained": int(metrics["nAbstained"]),
                "highRiskCapture": float(metrics["highRiskCapture"]),
                "falseReassurance": int(metrics["falseReassurance"]),
                "operatingPoint": float(np.isclose(threshold, ABSTENTION_DISAGREEMENT)),
            }
        )
    rows.sort(key=lambda item: float(item["coverage"]))
    return rows
