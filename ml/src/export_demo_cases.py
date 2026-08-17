from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from abstention import REASON_TEXT, decide_abstention
from constants import CUTOFF_DAYS, DEMO_SLOTS, DISCLAIMER, HIGH_RISK_THRESHOLD, MODEL_VERSION
from explain import explanation_text, local_factors
from train_regressor import TrainedRegressor


def risk_band(prob: float, abstained: bool, point: float | None = None) -> str:
    if abstained:
        return "review"
    if (point is not None and point >= HIGH_RISK_THRESHOLD) or prob >= 0.7:
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
    point: float | None = None,
    interval90: tuple[float, float] | None = None,
    interval50: tuple[float, float] | None = None,
    warning_score: float | None = None,
) -> dict[str, Any]:
    if point is None:
        point = float(np.median(ensemble_preds))
    if interval90 is None:
        lo, hi = np.quantile(ensemble_preds, [0.05, 0.95])
    else:
        lo, hi = float(interval90[0]), float(interval90[1])
    if interval50 is None:
        inner_lo, inner_hi = np.quantile(ensemble_preds, [0.25, 0.75])
    else:
        inner_lo, inner_hi = float(interval50[0]), float(interval50[1])
    score = point if warning_score is None else float(warning_score)
    proba = float(calibrator.predict_proba(np.array([score]))[0])
    current_risk = float(row["risk"]) if pd.notna(row.get("risk")) else float("nan")
    miss_distance = (
        float(row["miss_distance"]) if pd.notna(row.get("miss_distance")) else float("nan")
    )
    decision = decide_abstention(
        ensemble_preds,
        current_risk,
        miss_distance,
        interval90=(lo, hi),
        point=point,
        warning_proba=None if warning_score is None else float(warning_score),
    )
    abstained = decision.abstained
    _, factors = local_factors(trained, explainer, row)
    payload = {
        "eventId": str(event_id),
        "predictedFinalRiskLog10": point,
        "predictedFinalPc": float(10**point),
        "interval90Log10": [float(lo), float(hi)],
        "interval50Log10": [float(inner_lo), float(inner_hi)],
        "configuredHighRiskProbability": proba,
        "highRiskThresholdLog10": HIGH_RISK_THRESHOLD,
        "riskBand": risk_band(proba, abstained, point),
        "abstained": abstained,
        "abstentionReasons": [REASON_TEXT[reason] for reason in decision.reasons],
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
    if abstained or story == "uncertain":
        return (
            f"Today {today}. Guesses cross the 1-in-a-million line, so a person should review this."
        )
    if story == "low":
        return f"Today {today}. Forecast stays quiet ({guess})."
    if story == "high_now":
        return f"Today {today}. Already at the ESA class, so the forecast copies today's report."
    if story == "high_stays":
        return f"Today {today}. Forecast copies today's report ({guess})."
    if story == "high_drop":
        return f"Today {today}. Forecast copies today's report ({guess})."
    if story == "high":
        return f"Today {today}. Forecast {guess}."
    return f"Today {today}. Forecast {guess}."


def story_fit(story: str, pred: float, persist: float, actual: float, abstained: bool) -> float:
    pred_err = abs(pred - actual)
    persist_err = abs(persist - actual)
    if story == "low":
        if abstained or actual >= -7 or pred >= -7 or persist >= -7 or pred_err > 0.5:
            return -1.0
        away_from_floor = 0.6 if persist > -29 else 0.0
        return 3.0 - pred_err - 0.1 * persist_err + away_from_floor
    if story == "uncertain":
        return 3.0 if abstained else -1.0
    if story == "high_now":
        if abstained or persist < HIGH_RISK_THRESHOLD:
            return -1.0
        collapse = 1.5 if actual <= -20 else 0.0
        return 2.0 + collapse + max(persist - HIGH_RISK_THRESHOLD, 0.0)
    if story == "high_stays":
        if abstained or persist < HIGH_RISK_THRESHOLD or actual < HIGH_RISK_THRESHOLD:
            return -1.0
        return 3.0 + max(actual - HIGH_RISK_THRESHOLD, 0.0)
    if story == "high_drop":
        if abstained or persist < HIGH_RISK_THRESHOLD or actual >= HIGH_RISK_THRESHOLD:
            return -1.0
        not_floor = 1.2 if actual > -29 else 0.0
        return 2.0 + not_floor
    if story == "high":
        if abstained:
            return -1.0
        if (
            persist < HIGH_RISK_THRESHOLD
            and pred < HIGH_RISK_THRESHOLD
            and actual < HIGH_RISK_THRESHOLD
        ):
            return -1.0
        return 1.0 + max(persist, pred, actual)
    return -1.0


def messages_from_history(history: pd.DataFrame) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for _, row in history.sort_values("time_to_tca", ascending=False).iterrows():
        rows.append(
            {
                "timeToTcaDays": float(row["time_to_tca"]),
                "riskLog10": float(row["risk"]),
                "missDistanceM": float(row["miss_distance"]),
                "relativeSpeedMps": float(row["relative_speed"]),
                "maxRiskEstimate": float(row["max_risk_estimate"]),
                "relativePositionR": float(row["relative_position_r"]),
                "relativePositionT": float(row["relative_position_t"]),
                "relativePositionN": float(row["relative_position_n"]),
                "relativeVelocityR": float(row["relative_velocity_r"]),
                "relativeVelocityT": float(row["relative_velocity_t"]),
                "relativeVelocityN": float(row["relative_velocity_n"]),
                "tSigmaR": float(row["t_sigma_r"]),
                "tSigmaT": float(row["t_sigma_t"]),
                "tSigmaN": float(row["t_sigma_n"]),
                "cSigmaR": float(row["c_sigma_r"]),
                "cSigmaT": float(row["c_sigma_t"]),
                "cSigmaN": float(row["c_sigma_n"]),
                "tObsUsed": float(row["t_obs_used"]),
                "cObsUsed": float(row["c_obs_used"]),
                "tObsAvailable": float(row["t_obs_available"]),
                "cObsAvailable": float(row["c_obs_available"]),
                "cObjectType": str(row["c_object_type"]),
            }
        )
    return rows


def _as_int_id(value: object) -> int:
    return int(value)


def select_demo_event_ids(
    slots: list[dict[str, str]],
    predictions: dict[int, dict[str, Any]],
    features: pd.DataFrame,
    test_ids: set[int],
) -> list[tuple[dict[str, str], int]]:
    indexed = features.copy()
    indexed["event_id"] = indexed["event_id"].map(_as_int_id)
    features_by_id = indexed.set_index("event_id", drop=False)
    used: set[int] = set()
    chosen: list[tuple[dict[str, str], int]] = []
    for slot in slots:
        ranked: list[tuple[float, int]] = []
        for raw_id, prediction in predictions.items():
            event_id = _as_int_id(raw_id)
            if event_id in used:
                continue
            feature_row = features_by_id.loc[event_id]
            persist = float(feature_row["risk"])
            actual = float(feature_row["y"])
            pred = float(prediction["predictedFinalRiskLog10"])
            abstained = bool(prediction["abstained"])
            score = story_fit(slot["key"], pred, persist, actual, abstained)
            prefer = 1.5 if event_id in test_ids else 0.0
            if score >= 0:
                ranked.append((score + prefer, event_id))
        pool = ranked
        if not pool and slot["story"] == "high":
            for raw_id, prediction in predictions.items():
                event_id = _as_int_id(raw_id)
                if event_id in used:
                    continue
                feature_row = features_by_id.loc[event_id]
                persist = float(feature_row["risk"])
                abstained = bool(prediction["abstained"])
                if abstained or persist < HIGH_RISK_THRESHOLD:
                    continue
                prefer = 1.5 if event_id in test_ids else 0.0
                pool.append((persist + prefer, event_id))
        if not pool:
            pool = [
                (0.0, _as_int_id(event_id))
                for event_id in predictions
                if _as_int_id(event_id) not in used
            ]
        pool.sort(key=lambda item: item[0], reverse=True)
        event_id = pool[0][1]
        used.add(event_id)
        chosen.append((slot, event_id))
    return chosen


def assemble_demo_cases(
    *,
    slots: list[dict[str, str]],
    predictions: dict[int, dict[str, Any]],
    features: pd.DataFrame,
    event_by_id: dict[int, dict[str, Any]],
    aligned: pd.DataFrame,
    ensemble_matrix: np.ndarray,
    trained: TrainedRegressor,
    calibrator: Any,
    explainer: Any,
    test_ids: set[int],
    points: np.ndarray | None = None,
    interval90: np.ndarray | None = None,
    interval50: np.ndarray | None = None,
    warning_scores: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    event_ids = features["event_id"].map(_as_int_id).to_numpy()
    id_to_position = {int(event_id): position for position, event_id in enumerate(event_ids)}
    selected = select_demo_event_ids(slots, predictions, features, test_ids)
    indexed = features.copy()
    indexed["event_id"] = indexed["event_id"].map(_as_int_id)
    features_by_id = indexed.set_index("event_id", drop=False)
    events = {_as_int_id(event_id): event for event_id, event in event_by_id.items()}
    demo_cases: list[dict[str, Any]] = []
    for slot, event_id in selected:
        event = events[event_id]
        feature_row = features_by_id.loc[event_id]
        position = id_to_position[event_id]
        history = messages_from_history(event["history"])
        later = event["full_history"]
        future = later[later["time_to_tca"] < CUTOFF_DAYS]
        prediction = predict_event(
            trained=trained,
            ensemble_preds=ensemble_matrix[position],
            calibrator=calibrator,
            explainer=explainer,
            row=aligned.iloc[position],
            messages=history,
            event_id=f"demo-{event_id}",
            point=None if points is None else float(points[position]),
            interval90=(
                None
                if interval90 is None
                else (float(interval90[0, position]), float(interval90[1, position]))
            ),
            interval50=(
                None
                if interval50 is None
                else (float(interval50[0, position]), float(interval50[1, position]))
            ),
            warning_score=None if warning_scores is None else float(warning_scores[position]),
        )
        persist = float(feature_row["risk"])
        actual = float(feature_row["y"])
        demo_cases.append(
            {
                "id": f"demo-{event_id}",
                "story": slot["story"],
                "missionAlias": f"MISSION-{int(event['mission_id']):02d}",
                "title": slot["title"],
                "blurb": slot["blurb"],
                "briefing": case_briefing(
                    slot["key"],
                    persist,
                    float(prediction["predictedFinalRiskLog10"]),
                    actual,
                    bool(prediction["abstained"]),
                ),
                "prediction": prediction,
                "baselineRiskLog10": persist,
                "actualFinalRiskLog10": actual,
                "messages": history,
                "futureMessages": messages_from_history(future),
            }
        )
    return demo_cases


def refresh_from_frozen() -> list[dict[str, Any]]:
    from abstention import abstain_mask
    from build_events import build_event_histories
    from inference import PrismModel
    from ingest import load_esa_training
    from validate import validate_cdm_frame

    root = Path(__file__).resolve().parents[2]
    features = pd.read_csv(root / "data" / "processed" / "events.csv")
    features["event_id"] = features["event_id"].map(_as_int_id)
    split_path = root / "ml" / "artifacts" / "split_manifest.json"
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    test_ids = {int(event_id) for event_id in splits["test"]}
    model = PrismModel()
    out = model.predict_frame(features)
    persist = features["risk"].to_numpy(dtype=float)
    abstained, _, _ = abstain_mask(
        out.ensemble,
        persist,
        features["miss_distance"].to_numpy(dtype=float),
        interval90=out.interval90,
        point=out.point,
        warning_proba=out.warning_proba,
        warning_threshold=model.policy.f2_threshold,
    )
    predictions = {
        int(event_id): {
            "predictedFinalRiskLog10": float(out.point[index]),
            "abstained": bool(abstained[index]),
        }
        for index, event_id in enumerate(features["event_id"].to_numpy())
    }
    chosen = select_demo_event_ids(DEMO_SLOTS, predictions, features, test_ids)
    selected_ids = {event_id for _, event_id in chosen}
    raw = load_esa_training(root / "data" / "raw")
    raw = raw[raw["event_id"].map(_as_int_id).isin(selected_ids)].copy()
    events = build_event_histories(validate_cdm_frame(raw))
    event_by_id = {_as_int_id(event["event_id"]): event for event in events}
    aligned = features.reindex(columns=model.feature_names).apply(pd.to_numeric, errors="coerce")
    cases = assemble_demo_cases(
        slots=DEMO_SLOTS,
        predictions=predictions,
        features=features,
        event_by_id=event_by_id,
        aligned=aligned,
        ensemble_matrix=out.ensemble,
        trained=model.trained,
        calibrator=model.calibrator,
        explainer=model.explainer,
        test_ids=test_ids,
        points=out.point,
        interval90=out.interval90,
        interval50=out.interval50,
        warning_scores=out.warning_proba,
    )
    write_json(root / "ml" / "artifacts" / "demo_cases.json", cases)
    write_json(root / "apps" / "web" / "public" / "demo_cases.json", cases)
    return cases


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


if __name__ == "__main__":
    cases = refresh_from_frozen()
    summary = ", ".join(f"{item['story']}:{item['id']}" for item in cases)
    print(f"wrote {len(cases)} demo cases ({summary})")
