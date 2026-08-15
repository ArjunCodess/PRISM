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
from constants import HIGH_RISK_THRESHOLD, MODEL_VERSION, RANDOM_STATE  # noqa: E402
from evaluate import classification_metrics, persistence_improvement, regression_metrics  # noqa: E402
from explain import shap_explainer  # noqa: E402
from export_demo_cases import predict_event, write_json  # noqa: E402
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
                "tSigmaR": float(row["t_sigma_r"]),
                "cSigmaR": float(row["c_sigma_r"]),
                "tObsUsed": float(row["t_obs_used"]),
                "cObsUsed": float(row["c_obs_used"]),
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

    ensemble = _bootstrap_models(pd.concat([train, validation], ignore_index=True))
    ens_matrix = np.column_stack(
        [model.predict(test[booster.feature_names]) for model in ensemble]
    )
    ens_pred = np.median(ens_matrix, axis=1)

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
        "ablation": {
            "snapshot_mae": float("nan"),
        },
    }

    explainer = shap_explainer(booster, train)
    event_by_id = {event["event_id"]: event for event in events}
    stories_needed = ["low", "escalate", "deescalate", "uncertain", "failure"]
    demo_cases: list[dict[str, object]] = []
    used: set[int] = set()
    for story in stories_needed:
        match = features[features["story"] == story]
        if match.empty:
            match = features
        row = match.iloc[0]
        event_id = int(row["event_id"])
        used.add(event_id)
        event = event_by_id[event_id]
        history = event["history"]
        full = event["full_history"]
        feature_row = features[features["event_id"] == event_id].iloc[0]
        boot = np.array(
            [model.predict(feature_row[booster.feature_names].to_frame().T)[0] for model in ensemble]
        )
        prediction = predict_event(
            trained=booster,
            ensemble_preds=boot,
            calibrator=calibrator,
            explainer=explainer,
            row=feature_row,
            messages=_messages(history),
            event_id=f"demo-{event_id}",
        )
        demo_cases.append(
            {
                "id": f"demo-{event_id}",
                "story": story,
                "missionAlias": f"MISSION-{int(event['mission_id']):02d}",
                "title": {
                    "low": "Stable low-risk flyby",
                    "escalate": "Risk climbing toward TCA",
                    "deescalate": "Early alarm, later geometry safer",
                    "uncertain": "Interval crosses the warning line",
                    "failure": "Late jump the model can miss",
                }[story],
                "prediction": prediction,
                "baselineRiskLog10": float(feature_row["risk"]),
                "actualFinalRiskLog10": float(feature_row["y"]),
                "messages": _messages(history),
                "futureMessages": _messages(full[full["time_to_tca"] < 2.0]),
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
