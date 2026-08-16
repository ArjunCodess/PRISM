from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from constants import (
    ABSTENTION_DISAGREEMENT,
    FALSE_REASSURANCE_DEFINITION,
    HIGH_RISK_THRESHOLD,
)
from evaluate import regression_metrics

REASON_CROSSES = "spread_crosses_threshold"
REASON_MISSING = "missing_critical_fields"
REASON_DISAGREEMENT = "ensemble_disagreement"

REASON_TEXT = {
    REASON_CROSSES: (
        "the 90% bootstrap band crosses the ESA challenge class log10(Pc) ≥ −6"
    ),
    REASON_MISSING: "current reported risk or miss distance is missing",
    REASON_DISAGREEMENT: (
        f"bootstrap models disagree by more than {ABSTENTION_DISAGREEMENT} log-risk units"
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
) -> AbstentionDecision:
    preds = np.asarray(ensemble_preds, dtype=float)
    lo, hi = np.quantile(preds, [0.05, 0.95])
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = np.quantile(ensemble_matrix, 0.05, axis=1)
    hi = np.quantile(ensemble_matrix, 0.95, axis=1)
    disagreement = np.std(ensemble_matrix, axis=1)
    crosses = (lo < HIGH_RISK_THRESHOLD) & (hi >= HIGH_RISK_THRESHOLD)
    missing = ~np.isfinite(current_risk) | ~np.isfinite(miss_distance)
    abstained = crosses | missing | (disagreement > disagreement_threshold)
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
) -> list[dict[str, float | int]]:
    if thresholds is None:
        thresholds = np.concatenate(
            ([0.05], np.linspace(0.25, 3.0, 12), [ABSTENTION_DISAGREEMENT])
        )
        thresholds = np.unique(np.round(thresholds, 4))
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        abstained, _, _ = abstain_mask(
            ensemble_matrix, current_risk, miss_distance, disagreement_threshold=float(threshold)
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
