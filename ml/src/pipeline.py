from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from build_events import build_event_histories  # noqa: E402
from calibrate import fit_isotonic  # noqa: E402
from constants import HIGH_RISK_THRESHOLD, MODEL_VERSION, RANDOM_STATE, SNAPSHOT_COLUMNS, STORY_COPY  # noqa: E402
from evaluate import (  # noqa: E402
    classification_metrics,
    error_gallery,
    persistence_improvement,
    regression_metrics,
    reliability_bins,
)
from explain import grouped_importance, shap_explainer  # noqa: E402
from export_demo_cases import case_briefing, predict_event, story_fit, write_json  # noqa: E402
from features import build_feature_table  # noqa: E402
from generate_synthetic import generate_synthetic_cdms  # noqa: E402
from split import grouped_splits, subset  # noqa: E402
from train_classifier import fit_warning_classifier  # noqa: E402
from train_regressor import (  # noqa: E402
    fit_ridge,
    fit_xgboost,
    median_predict,
    persistence_predict,
    predict_model,
)
from validate import validate_cdm_frame  # noqa: E402


def _messages(history: pd.DataFrame) -> list[dict[str, float | str]]:
    rows = []
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


def _bootstrap_models(train: pd.DataFrame, n_models: int = 8) -> list[XGBRegressor]:
    rng = np.random.default_rng(RANDOM_STATE)
    models: list[XGBRegressor] = []
    for _ in range(n_models):
        idx = rng.integers(0, len(train), size=len(train))
        sample = train.iloc[idx]
        models.append(fit_xgboost(sample).model)
    return models


def run_pipeline(n_events: int = 420) -> dict[str, object]:
    artifacts = ROOT / "ml" / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    raw = generate_synthetic_cdms(n_events=n_events)
    frame = validate_cdm_frame(raw)
    frame.to_csv(ROOT / "data" / "interim" / "synthetic_cdms.csv", index=False)
    events = build_event_histories(frame)
    features = build_feature_table(events)
    features["story"] = [event.get("story") for event in events]
    features.to_csv(ROOT / "data" / "processed" / "events.csv", index=False)

    splits = grouped_splits(features)
    train = subset(features, splits.train_ids)
    validation = subset(features, splits.validation_ids)
    calibration = subset(features, splits.calibration_ids)
    test = subset(features, splits.test_ids)

    persist_test = persistence_predict(test)
    median_test = median_predict(train, len(test))
    ridge = fit_ridge(train)
    ridge_pred = predict_model(ridge, test)
    booster = fit_xgboost(train)
    model_pred = predict_model(booster, test)
    x_test = test[booster.feature_names].apply(pd.to_numeric, errors="coerce")

    ensemble = _bootstrap_models(pd.concat([train, validation], ignore_index=True))
    ens_matrix = np.column_stack([model.predict(x_test) for model in ensemble])
    ens_pred = np.median(ens_matrix, axis=1)

    snapshot_cols = [col for col in SNAPSHOT_COLUMNS if col in train.columns and col != "c_object_type"]
    snap_model = fit_xgboost(train[snapshot_cols + ["y", "event_id", "c_object_type"]])
    snap_pred = predict_model(snap_model, test)

    classifier = fit_warning_classifier(train)
    joblib.dump(classifier, artifacts / "warning_classifier.joblib")
    cal_scores = predict_model(booster, calibration)
    cal_labels = (calibration["y"].to_numpy() >= HIGH_RISK_THRESHOLD).astype(int)
    calibrator = fit_isotonic(cal_scores, cal_labels)
    test_proba = calibrator.predict_proba(ens_pred)

    metrics = {
        "modelVersion": MODEL_VERSION,
        "nEvents": len(features),
        "splits": {
            "train": len(splits.train_ids),
            "validation": len(splits.validation_ids),
            "calibration": len(splits.calibration_ids),
            "test": len(splits.test_ids),
        },
        "persistence": regression_metrics(test["y"].to_numpy(), persist_test),
        "median": regression_metrics(test["y"].to_numpy(), median_test),
        "ridge": regression_metrics(test["y"].to_numpy(), ridge_pred),
        "xgboost": regression_metrics(test["y"].to_numpy(), model_pred),
        "ensemble": regression_metrics(test["y"].to_numpy(), ens_pred),
        "improvement": persistence_improvement(test["y"].to_numpy(), ens_pred, persist_test),
        "warning": classification_metrics(test["y"].to_numpy(), test_proba),
        "calibration": reliability_bins(test["y"].to_numpy(), test_proba),
        "ablation": {
            "snapshot_mae": float(regression_metrics(test["y"].to_numpy(), snap_pred)["mae"]),
            "full_mae": float(regression_metrics(test["y"].to_numpy(), model_pred)["mae"]),
            "ensemble_mae": float(regression_metrics(test["y"].to_numpy(), ens_pred)["mae"]),
        },
        "featureGroups": grouped_importance(
            booster.feature_names,
            booster.model.get_booster().get_score(importance_type="gain"),
        ),
        "failures": error_gallery(
            test["event_id"].to_numpy(),
            test["y"].to_numpy(),
            ens_pred,
            persist_test,
        ),
    }

    explainer = shap_explainer(booster, train)
    event_by_id = {event["event_id"]: event for event in events}
    predictions: dict[int, dict[str, object]] = {}
    for _, feature_row in features.iterrows():
        event_id = int(feature_row["event_id"])
        event = event_by_id[event_id]
        aligned = pd.DataFrame([{name: float(feature_row[name]) for name in booster.feature_names}])
        boot = np.array([model.predict(aligned)[0] for model in ensemble])
        predictions[event_id] = predict_event(
            trained=booster,
            ensemble_preds=boot,
            calibrator=calibrator,
            explainer=explainer,
            row=aligned.iloc[0],
            messages=_messages(event["history"]),
            event_id=f"demo-{event_id}",
        )

    stories_needed = ["low", "escalate", "deescalate", "uncertain", "failure"]
    used: set[int] = set()
    test_ids = set(splits.test_ids)
    demo_cases: list[dict[str, object]] = []
    for story in stories_needed:
        ranked: list[tuple[float, int]] = []
        fallback: list[tuple[float, int]] = []
        for event_id, prediction in predictions.items():
            if event_id in used:
                continue
            feature_row = features[features["event_id"] == event_id].iloc[0]
            persist = float(feature_row["risk"])
            actual = float(feature_row["y"])
            pred = float(prediction["predictedFinalRiskLog10"])
            abstained = bool(prediction["abstained"])
            score = story_fit(story, pred, persist, actual, abstained)
            prefer = 1.5 if event_id in test_ids else 0.0
            same_story = str(feature_row.get("story")) == story
            if score >= 0 and same_story:
                ranked.append((score + prefer, event_id))
            elif score >= 0:
                fallback.append((score + prefer - 0.4, event_id))
            elif story == "failure" and not abstained:
                late = actual >= -6 and persist < actual - 0.4
                under = pred < actual - 0.35
                if late and under:
                    fallback.append((actual - pred + prefer, event_id))
        pool = ranked or fallback
        if not pool and story == "failure":
            for event_id, prediction in predictions.items():
                if event_id in used:
                    continue
                feature_row = features[features["event_id"] == event_id].iloc[0]
                persist = float(feature_row["risk"])
                actual = float(feature_row["y"])
                pred = float(prediction["predictedFinalRiskLog10"])
                if actual >= -6 and pred < actual - 0.35:
                    pool.append((actual - pred, event_id))
        if not pool:
            pool = [(0.0, event_id) for event_id in predictions if event_id not in used]
        pool.sort(key=lambda item: item[0], reverse=True)
        event_id = pool[0][1]
        used.add(event_id)
        event = event_by_id[event_id]
        feature_row = features[features["event_id"] == event_id].iloc[0]
        prediction = predictions[event_id]
        persist = float(feature_row["risk"])
        actual = float(feature_row["y"])
        copy = STORY_COPY[story]
        demo_cases.append(
            {
                "id": f"demo-{event_id}",
                "story": story,
                "missionAlias": f"MISSION-{int(event['mission_id']):02d}",
                "title": copy["title"],
                "blurb": copy["blurb"],
                "briefing": case_briefing(
                    story,
                    persist,
                    float(prediction["predictedFinalRiskLog10"]),
                    actual,
                    bool(prediction["abstained"]),
                ),
                "prediction": prediction,
                "baselineRiskLog10": persist,
                "actualFinalRiskLog10": actual,
                "messages": _messages(event["history"]),
                "futureMessages": _messages(event["full_history"][event["full_history"]["time_to_tca"] < 2.0]),
            }
        )

    booster.model.save_model(artifacts / "risk_regressor.json")
    joblib.dump(
        {
            "calibrator": calibrator,
            "feature_names": booster.feature_names,
            "ensemble": ensemble,
        },
        artifacts / "warning_calibrator.joblib",
    )
    write_json(artifacts / "feature_schema.json", {"features": booster.feature_names})
    write_json(artifacts / "metrics.json", metrics)
    write_json(
        artifacts / "split_manifest.json",
        {
            "train": splits.train_ids,
            "validation": splits.validation_ids,
            "calibration": splits.calibration_ids,
            "test": splits.test_ids,
        },
    )
    write_json(artifacts / "demo_cases.json", demo_cases)
    write_json(
        artifacts / "model_card.json",
        {
            "modelVersion": MODEL_VERSION,
            "intendedUse": "Education and interpretability demonstration only.",
            "outOfScope": [
                "spacecraft operations",
                "autonomous manoeuvres",
                "claims about specific real satellites",
            ],
            "highRiskThresholdLog10": HIGH_RISK_THRESHOLD,
            "metrics": metrics["ensemble"],
            "beatsPersistence": bool(metrics["improvement"]["beats_persistence"]),
        },
    )
    return metrics


if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps(result["improvement"], indent=2))
    print(json.dumps(result["ensemble"], indent=2))
