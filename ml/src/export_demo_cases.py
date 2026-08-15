from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from constants import CUTOFF_DAYS, DISCLAIMER, HIGH_RISK_THRESHOLD, MODEL_VERSION
from explain import explanation_text, local_factors, shap_explainer
from train_regressor import TrainedRegressor, predict_model


def risk_band(prob: float, abstained: bool) -> str:
    if abstained:
        return "review"
    if prob >= 0.7:
        return "high"
    if prob >= 0.4:
        return "review"
    return "low"


def predict_event(
    *,
    trained: TrainedRegressor,
    ensemble_preds: np.ndarray,
    calibrator,
    explainer,
    row: pd.Series,
    messages: list[dict[str, Any]],
    event_id: str,
) -> dict[str, Any]:
    point = float(np.median(ensemble_preds))
    lo, hi = np.quantile(ensemble_preds, [0.05, 0.95])
    proba = float(calibrator.predict_proba(np.array([point]))[0])
    crosses = (lo < HIGH_RISK_THRESHOLD <= hi) or (hi < HIGH_RISK_THRESHOLD <= lo)
    missing_critical = bool(pd.isna(row.get("risk")) or pd.isna(row.get("miss_distance")))
    disagreement = float(np.std(ensemble_preds))
    abstained = bool(crosses or missing_critical or disagreement > 1.25)
    _, factors = local_factors(trained, explainer, row)
    payload = {
        "eventId": str(event_id),
        "predictedFinalRiskLog10": point,
        "predictedFinalPc": float(10**point),
        "interval90Log10": [float(lo), float(hi)],
        "configuredHighRiskProbability": proba,
        "highRiskThresholdLog10": HIGH_RISK_THRESHOLD,
        "riskBand": risk_band(proba, abstained),
        "abstained": abstained,
        "topFactors": [
            {
                "feature": item.feature,
                "direction": item.direction,
                "contribution": item.contribution,
                "label": item.label,
            }
            for item in factors[:6]
        ],
        "explanation": explanation_text(factors),
        "modelVersion": MODEL_VERSION,
        "disclaimer": DISCLAIMER,
        "cutoffHours": int(CUTOFF_DAYS * 24),
        "nMessagesUsed": len(messages),
    }
    return payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
