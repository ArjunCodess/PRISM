from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from constants import CUTOFF_DAYS, DISCLAIMER, HIGH_RISK_THRESHOLD, MODEL_VERSION
from explain import explanation_text, local_factors
from train_regressor import TrainedRegressor


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
    lo, inner_lo, inner_hi, hi = np.quantile(ensemble_preds, [0.05, 0.25, 0.75, 0.95])
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
        "interval50Log10": [float(inner_lo), float(inner_hi)],
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


def frequency_phrase(log_risk: float) -> str:
    if not np.isfinite(log_risk):
        return "unknown"
    if log_risk <= -9:
        return "vanishingly small"
    count = max(1, int(round(10 ** -float(log_risk))))
    if count >= 1_000_000_000:
        return "less than 1 in a billion"
    return f"1 in {count:,}"


def _spoken_chance(log_risk: float) -> str:
    phrase = frequency_phrase(log_risk)
    if phrase.startswith("1 in"):
        return f"about {phrase}"
    return phrase


def case_briefing(story: str, persist: float, pred: float, actual: float, abstained: bool) -> str:
    today = _spoken_chance(persist)
    guess = _spoken_chance(pred)
    later = _spoken_chance(actual)
    if abstained:
        return (
            f"Today {today}. Guesses cross the 1-in-a-million line, so a person should review this."
        )
    if story == "low":
        return f"Today {today}. Forecast stays quiet ({guess})."
    if story == "escalate":
        return f"Today {today}. Forecast rises to {guess}."
    if story == "deescalate":
        return f"Today {today}. Forecast calms to {guess}."
    if story == "failure":
        return f"Forecast {guess}; later update {later}."
    return f"Today {today}. Forecast {guess}."


def story_fit(story: str, pred: float, persist: float, actual: float, abstained: bool) -> float:
    pred_err = abs(pred - actual)
    persist_err = abs(persist - actual)
    if story == "low":
        if abstained or actual >= -7 or pred >= -7 or persist >= -7:
            return -1.0
        return 2.0 - 0.05 * pred_err
    if story == "escalate":
        if abstained or actual < -6 or pred <= persist:
            return -1.0
        return 1.5 + (persist_err - pred_err)
    if story == "deescalate":
        if abstained or actual >= -6 or persist <= actual or pred >= persist:
            return -1.0
        return 1.5 + (persist_err - pred_err)
    if story == "uncertain":
        return 3.0 if abstained else -1.0
    if story == "failure":
        if abstained:
            return -1.0
        late_jump = actual >= HIGH_RISK_THRESHOLD and persist < HIGH_RISK_THRESHOLD - 0.25
        missed = actual >= HIGH_RISK_THRESHOLD and pred < HIGH_RISK_THRESHOLD
        under = pred < actual - 0.45
        if missed and late_jump:
            return 6.0 + (actual - pred)
        if missed:
            return 4.0 + (actual - pred)
        if late_jump and under:
            return 3.0 + (actual - pred)
        return -1.0
    return -1.0


def write_json(path: Path, payload: object) -> None:
    def json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            return None
        if isinstance(value, np.integer):
            return int(value)
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
