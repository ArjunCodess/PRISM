from __future__ import annotations

import argparse
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

from abstention import abstain_mask  # noqa: E402
from build_events import build_event_histories  # noqa: E402
from calibrate import fit_isotonic  # noqa: E402
from constants import (  # noqa: E402
    ABSTENTION_RULE,
    HIGH_RISK_THRESHOLD,
    MODEL_VERSION,
    RANDOM_STATE,
    RESEARCH_QUESTION,
    STORY_COPY,
)
from evaluate import (  # noqa: E402
    classification_metrics,
    error_gallery,
    persistence_improvement,
    regression_metrics,
    reliability_bins,
)
from experiments import (  # noqa: E402
    abstention_study,
    cluster_test_failures,
    forecast_horizon_table,
    historical_ablation,
    shap_outcome_contrast,
)
from explain import grouped_importance, shap_explainer  # noqa: E402
from export_demo_cases import case_briefing, predict_event, story_fit, write_json  # noqa: E402
from features import build_feature_table  # noqa: E402
from generate_synthetic import generate_synthetic_cdms  # noqa: E402
from ingest import (  # noqa: E402
    load_esa_training,
    realistic_training_events,
    validate_official_test_compatibility,
)
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


def _bootstrap_models(train: pd.DataFrame, n_models: int = 10) -> list[XGBRegressor]:
    rng = np.random.default_rng(RANDOM_STATE)
    models: list[XGBRegressor] = []
    for _ in range(n_models):
        idx = rng.integers(0, len(train), size=len(train))
        sample = train.iloc[idx]
        models.append(fit_xgboost(sample).model)
    return models


def _slice_metrics(
    frame: pd.DataFrame, predictions: np.ndarray, mask: pd.Series | np.ndarray
) -> dict[str, float | int]:
    selected = np.asarray(mask, dtype=bool)
    if not selected.any():
        return {"n": 0, "highRiskEvents": 0}
    selected_frame = frame.iloc[np.flatnonzero(selected)]
    result: dict[str, float | int] = {
        "n": int(selected.sum()),
        "highRiskEvents": int((selected_frame["y"] >= HIGH_RISK_THRESHOLD).sum()),
    }
    result.update(regression_metrics(selected_frame["y"].to_numpy(), predictions[selected]))
    return result


def _robustness_slices(test: pd.DataFrame, predictions: np.ndarray) -> dict[str, object]:
    combined_sigma = test["t_sigma_r"].fillna(0).abs() + test["c_sigma_r"].fillna(0).abs()
    return {
        "byObjectType": {
            str(name): _slice_metrics(test, predictions, test["c_object_type"] == name)
            for name in sorted(test["c_object_type"].dropna().unique())
        },
        "byMessageCount": {
            "one": _slice_metrics(test, predictions, test["n_messages"] <= 1),
            "twoToFive": _slice_metrics(
                test, predictions, (test["n_messages"] >= 2) & (test["n_messages"] <= 5)
            ),
            "sixOrMore": _slice_metrics(test, predictions, test["n_messages"] >= 6),
        },
        "byMissDistance": {
            "under500m": _slice_metrics(test, predictions, test["miss_distance"] < 500),
            "500mTo2km": _slice_metrics(
                test,
                predictions,
                (test["miss_distance"] >= 500) & (test["miss_distance"] < 2_000),
            ),
            "2kmOrMore": _slice_metrics(test, predictions, test["miss_distance"] >= 2_000),
        },
        "byRadialUncertainty": {
            "under500m": _slice_metrics(test, predictions, combined_sigma < 500),
            "500mTo5km": _slice_metrics(
                test, predictions, (combined_sigma >= 500) & (combined_sigma < 5_000)
            ),
            "5kmOrMore": _slice_metrics(test, predictions, combined_sigma >= 5_000),
        },
        "bySnapshotAge": {
            "under6h": _slice_metrics(test, predictions, test["hours_before_cutoff"] < 6),
            "6hTo24h": _slice_metrics(
                test,
                predictions,
                (test["hours_before_cutoff"] >= 6) & (test["hours_before_cutoff"] < 24),
            ),
            "24hOrMore": _slice_metrics(test, predictions, test["hours_before_cutoff"] >= 24),
        },
    }


def run_pipeline(source: str = "real", n_events: int = 420) -> dict[str, object]:
    artifacts = ROOT / "ml" / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    processed = ROOT / "data" / "processed"
    interim = ROOT / "data" / "interim"
    processed.mkdir(parents=True, exist_ok=True)
    interim.mkdir(parents=True, exist_ok=True)

    if source == "real":
        raw = load_esa_training(ROOT / "data" / "raw")
        source_rows = len(raw)
        raw = realistic_training_events(raw)
        data_source = "ESA Collision Avoidance Challenge training archive"
        compatibility = validate_official_test_compatibility(
            ROOT / "data" / "raw", set(raw.columns) - {"t_ecc", "c_ecc"}
        )
    elif source == "synthetic":
        raw = generate_synthetic_cdms(n_events=n_events)
        source_rows = len(raw)
        data_source = "Synthetic ESA-schema data"
        compatibility = None
        raw.to_csv(interim / "synthetic_cdms.csv", index=False)
    else:
        raise ValueError(f"unsupported source {source!r}; choose 'real' or 'synthetic'")

    frame = validate_cdm_frame(raw)
    events = build_event_histories(frame)
    features = build_feature_table(events)
    features["story"] = [event.get("story") for event in events]
    mission_features = build_feature_table(events, include_mission=True)
    mission_features["story"] = features["story"].to_numpy()
    features.to_csv(processed / "events.csv", index=False)
    write_json(
        interim / "training_manifest.json",
        {
            "dataSource": data_source,
            "sourceRows": source_rows,
            "eligibleRows": len(frame),
            "eligibleEvents": len(events),
            "officialTestCompatibility": compatibility,
        },
    )

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
    raw_ens_matrix = np.column_stack([model.predict(x_test) for model in ensemble])
    persist_guard = test["risk"].to_numpy(dtype=float) >= HIGH_RISK_THRESHOLD
    ens_matrix = np.where(
        persist_guard[:, None], test["risk"].to_numpy(dtype=float)[:, None], raw_ens_matrix
    )
    ens_pred = np.median(ens_matrix, axis=1)
    interval_50 = np.quantile(ens_matrix, [0.25, 0.75], axis=1)
    interval_90 = np.quantile(ens_matrix, [0.05, 0.95], axis=1)

    mission_train = subset(mission_features, splits.train_ids)
    mission_test = subset(mission_features, splits.test_ids)
    mission_model = fit_xgboost(mission_train)
    mission_pred = predict_model(mission_model, mission_test)

    rng = np.random.default_rng(RANDOM_STATE)
    mission_ids = np.asarray(sorted(mission_features["mission_id"].dropna().unique()))
    rng.shuffle(mission_ids)
    n_held_out = max(1, int(np.ceil(len(mission_ids) * 0.2)))
    held_out_missions = mission_ids[:n_held_out]
    held_out_mask = mission_features["mission_id"].isin(held_out_missions).to_numpy()
    holdout_train = features.iloc[np.flatnonzero(~held_out_mask)]
    holdout_test = features.iloc[np.flatnonzero(held_out_mask)]
    holdout_model = fit_xgboost(holdout_train)
    holdout_pred = predict_model(holdout_model, holdout_test)
    holdout_persist = persistence_predict(holdout_test)

    ablation = historical_ablation(train, test)
    classifier = fit_warning_classifier(train)
    joblib.dump(classifier, artifacts / "warning_classifier.joblib")
    raw_cal_scores = predict_model(booster, calibration)
    cal_scores = np.where(
        calibration["risk"].to_numpy(dtype=float) >= HIGH_RISK_THRESHOLD,
        calibration["risk"].to_numpy(dtype=float),
        raw_cal_scores,
    )
    cal_labels = (calibration["y"].to_numpy() >= HIGH_RISK_THRESHOLD).astype(int)
    calibrator = fit_isotonic(cal_scores, cal_labels)
    test_proba = calibrator.predict_proba(ens_pred)
    test_abstained, _, _ = abstain_mask(
        ens_matrix,
        test["risk"].to_numpy(dtype=float),
        test["miss_distance"].to_numpy(dtype=float),
    )
    n_high_eligible = int((features["y"] >= HIGH_RISK_THRESHOLD).sum())
    n_high_test = int((test["y"] >= HIGH_RISK_THRESHOLD).sum())
    warning_metrics = classification_metrics(test["y"].to_numpy(), test_proba)
    warning_metrics.update(
        {
            "nHighRiskEligible": n_high_eligible,
            "nHighRiskTest": n_high_test,
            "thresholdNote": (
                "ESA challenge class log10(Pc) ≥ −6, not an operational threshold"
            ),
        }
    )
    horizon_primary = {
        "cutoffHours": 48,
        "eligibleEvents": len(features),
        "trainEvents": len(train),
        "testEvents": len(test),
        "overlapTrain": len(train),
        "overlapTest": len(test),
        "model": regression_metrics(test["y"].to_numpy(), model_pred),
        "persistence": regression_metrics(test["y"].to_numpy(), persist_test),
        "maeImprovement": float(
            regression_metrics(test["y"].to_numpy(), persist_test)["mae"]
            - regression_metrics(test["y"].to_numpy(), model_pred)["mae"]
        ),
    }

    metrics = {
        "modelVersion": MODEL_VERSION,
        "researchQuestion": RESEARCH_QUESTION,
        "dataSource": data_source,
        "dataSourceKind": source,
        "selectedPolicy": (
            "bootstrap xgboost median with a persistence guard at the ESA class; "
            "the guard and abstention thresholds were locked before test evaluation"
        ),
        "sourceRows": source_rows,
        "eligibleRows": len(frame),
        "nEvents": len(features),
        "nHighRiskEligible": n_high_eligible,
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
        "warning": warning_metrics,
        "calibration": reliability_bins(test["y"].to_numpy(), test_proba),
        "uncertainty": {
            "method": "spread across 10 bootstrap xgboost models",
            "interpretation": (
                "Bootstrap disagreement is ensemble spread, not calibrated "
                "predictive uncertainty."
            ),
            "interval50Coverage": float(
                np.mean(
                    (test["y"].to_numpy() >= interval_50[0])
                    & (test["y"].to_numpy() <= interval_50[1])
                )
            ),
            "interval90Coverage": float(
                np.mean(
                    (test["y"].to_numpy() >= interval_90[0])
                    & (test["y"].to_numpy() <= interval_90[1])
                )
            ),
            "meanInterval50Width": float(np.mean(interval_50[1] - interval_50[0])),
            "meanInterval90Width": float(np.mean(interval_90[1] - interval_90[0])),
            "nModels": len(ensemble),
        },
        "robustness": _robustness_slices(test, ens_pred),
        "missionIdComparison": {
            "why": (
                "Adding mission_id provides negligible improvement and does not "
                "materially change performance, so it is excluded from production."
            ),
            "withoutMissionId": regression_metrics(test["y"].to_numpy(), model_pred),
            "withMissionId": regression_metrics(test["y"].to_numpy(), mission_pred),
        },
        "missionHoldout": {
            "why": (
                "Random event splits can hide distribution shift across mission "
                "families, especially on the rare high-risk tail."
            ),
            "heldOutMissions": [int(value) for value in sorted(held_out_missions)],
            "trainEvents": len(holdout_train),
            "testEvents": len(holdout_test),
            "nHighRiskTest": int((holdout_test["y"] >= HIGH_RISK_THRESHOLD).sum()),
            "model": regression_metrics(holdout_test["y"].to_numpy(), holdout_pred),
            "persistence": regression_metrics(holdout_test["y"].to_numpy(), holdout_persist),
        },
        "ablation": ablation,
        "horizons": forecast_horizon_table(
            frame,
            splits.train_ids,
            splits.test_ids,
            primary=horizon_primary,
        ),
        "abstention": abstention_study(
            test["y"].to_numpy(),
            ens_pred,
            persist_test,
            ens_matrix,
            test["risk"].to_numpy(dtype=float),
            test["miss_distance"].to_numpy(dtype=float),
            test_abstained,
            test_proba,
        ),
        "shapContrast": shap_outcome_contrast(booster, test, model_pred),
        "failureClusters": cluster_test_failures(test, ens_pred, persist_test),
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
        "abstentionRule": ABSTENTION_RULE,
    }

    event_by_id = {event["event_id"]: event for event in events}
    aligned_all = features[booster.feature_names].apply(pd.to_numeric, errors="coerce")
    raw_all_ensemble = np.column_stack([model.predict(aligned_all) for model in ensemble])
    all_persist_guard = features["risk"].to_numpy(dtype=float) >= HIGH_RISK_THRESHOLD
    all_ensemble = np.where(
        all_persist_guard[:, None],
        features["risk"].to_numpy(dtype=float)[:, None],
        raw_all_ensemble,
    )
    all_point = np.median(all_ensemble, axis=1)
    all_probability = calibrator.predict_proba(all_point)
    all_abstained, _, _ = abstain_mask(
        all_ensemble,
        features["risk"].to_numpy(dtype=float),
        features["miss_distance"].to_numpy(dtype=float),
    )
    event_ids = features["event_id"].astype(int).to_numpy()
    id_to_position = {event_id: position for position, event_id in enumerate(event_ids)}
    predictions = {
        event_id: {
            "predictedFinalRiskLog10": float(all_point[position]),
            "configuredHighRiskProbability": float(all_probability[position]),
            "abstained": bool(all_abstained[position]),
        }
        for position, event_id in enumerate(event_ids)
    }
    features_by_id = features.set_index("event_id", drop=False)

    stories_needed = ["low", "escalate", "deescalate", "uncertain", "failure"]
    used: set[int] = set()
    test_ids = set(splits.test_ids)
    demo_cases: list[dict[str, object]] = []
    explainer = shap_explainer(booster, train)
    for story in stories_needed:
        ranked: list[tuple[float, int]] = []
        fallback: list[tuple[float, int]] = []
        for event_id, prediction in predictions.items():
            if event_id in used:
                continue
            feature_row = features_by_id.loc[event_id]
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
                feature_row = features_by_id.loc[event_id]
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
        feature_row = features_by_id.loc[event_id]
        position = id_to_position[event_id]
        prediction = predict_event(
            trained=booster,
            ensemble_preds=all_ensemble[position],
            calibrator=calibrator,
            explainer=explainer,
            row=aligned_all.iloc[position],
            messages=_messages(event["history"]),
            event_id=f"demo-{event_id}",
        )
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
                "futureMessages": _messages(
                    event["full_history"][event["full_history"]["time_to_tca"] < 2.0]
                ),
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
            "dataSource": data_source,
            "intendedUse": (
                "Research prototype for offline, explainable conjunction-risk forecasting."
            ),
            "outOfScope": [
                "flight software",
                "operational decision systems",
                "spacecraft operations",
                "autonomous manoeuvres",
                "claims about specific real satellites",
            ],
            "researchQuestion": RESEARCH_QUESTION,
            "highRiskThresholdLog10": HIGH_RISK_THRESHOLD,
            "highRiskThresholdNote": (
                "ESA challenge class log10(Pc) ≥ −6, not an operational threshold"
            ),
            "abstentionRule": ABSTENTION_RULE,
            "nHighRiskEligible": n_high_eligible,
            "nHighRiskTest": n_high_test,
            "metrics": metrics["ensemble"],
            "uncertainty": metrics["uncertainty"],
            "missionHoldout": metrics["missionHoldout"],
            "ablation": metrics["ablation"],
            "horizons": metrics["horizons"],
            "abstention": metrics["abstention"]["operatingPoint"],
            "beatsPersistence": bool(metrics["improvement"]["beats_persistence"]),
        },
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and export PRISM artifacts")
    parser.add_argument("--source", choices=("real", "synthetic"), default="real")
    parser.add_argument("--synthetic-events", type=int, default=420)
    args = parser.parse_args()
    result = run_pipeline(source=args.source, n_events=args.synthetic_events)
    print(json.dumps(result["improvement"], indent=2))
    print(json.dumps(result["ensemble"], indent=2))
