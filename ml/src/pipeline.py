from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from abstention import abstain_mask  # noqa: E402
from build_events import build_event_histories  # noqa: E402
from calibrate import fit_isotonic_proba  # noqa: E402
from constants import (  # noqa: E402
    ABSTENTION_RULE,
    DEMO_SLOTS,
    ESA_LOSS_DEFINITION,
    FALSE_REASSURANCE_DEFINITION,
    HIGH_RISK_THRESHOLD,
    MODEL_VERSION,
    RANDOM_STATE,
    RESEARCH_QUESTION,
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
from export_demo_cases import assemble_demo_cases, write_json  # noqa: E402
from features import build_feature_table  # noqa: E402
from generate_synthetic import generate_synthetic_cdms  # noqa: E402
from hurdle import fit_hurdle_policy, predict_hurdle  # noqa: E402
from ingest import (  # noqa: E402
    load_esa_training,
    realistic_training_events,
    validate_official_test_compatibility,
)
from split import grouped_splits, subset  # noqa: E402
from train_regressor import (  # noqa: E402
    fit_ridge,
    fit_xgboost,
    median_predict,
    persistence_predict,
    predict_model,
)
from validate import validate_cdm_frame  # noqa: E402


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
    booster = fit_xgboost(train, eval_frame=validation)
    model_pred = predict_model(booster, test)
    policy = fit_hurdle_policy(train, validation, calibration)
    selected = predict_hurdle(policy, test)
    ens_pred = selected.point
    ens_matrix = selected.ensemble
    interval_50 = selected.interval50
    interval_90 = selected.interval90

    mission_train = subset(mission_features, splits.train_ids)
    mission_test = subset(mission_features, splits.test_ids)
    mission_model = fit_xgboost(mission_train, eval_frame=subset(mission_features, splits.validation_ids))
    mission_pred = predict_model(mission_model, mission_test)

    rng = np.random.default_rng(RANDOM_STATE)
    mission_ids = np.asarray(sorted(mission_features["mission_id"].dropna().unique()))
    rng.shuffle(mission_ids)
    n_held_out = max(1, int(np.ceil(len(mission_ids) * 0.2)))
    held_out_missions = mission_ids[:n_held_out]
    held_out_mask = mission_features["mission_id"].isin(held_out_missions).to_numpy()
    holdout_train = features.iloc[np.flatnonzero(~held_out_mask)]
    holdout_test = features.iloc[np.flatnonzero(held_out_mask)]
    holdout_split = grouped_splits(holdout_train)
    holdout_fit = subset(holdout_train, holdout_split.train_ids)
    holdout_val = subset(holdout_train, holdout_split.validation_ids)
    if len(holdout_fit) < 80:
        holdout_fit = holdout_train
        holdout_val = holdout_train
    holdout_policy = fit_hurdle_policy(holdout_fit, holdout_val, search=False, n_bootstrap=4)
    holdout_pred = predict_hurdle(holdout_policy, holdout_test).point
    holdout_persist = persistence_predict(holdout_test)

    ablation = historical_ablation(train, test)
    classifier = policy.warning
    joblib.dump(classifier, artifacts / "warning_classifier.joblib")
    cal_out = predict_hurdle(policy, calibration)
    cal_labels = (calibration["y"].to_numpy() >= HIGH_RISK_THRESHOLD).astype(int)
    calibrator = fit_isotonic_proba(cal_out.warning_proba, cal_labels)
    test_proba = calibrator.predict_proba(selected.warning_proba)
    test_abstained, _, _ = abstain_mask(
        ens_matrix,
        test["risk"].to_numpy(dtype=float),
        test["miss_distance"].to_numpy(dtype=float),
        interval90=interval_90,
        point=ens_pred,
        warning_proba=selected.warning_proba,
        warning_threshold=policy.f2_threshold,
    )
    n_high_eligible = int((features["y"] >= HIGH_RISK_THRESHOLD).sum())
    n_high_test = int((test["y"] >= HIGH_RISK_THRESHOLD).sum())
    warning_metrics = classification_metrics(
        test["y"].to_numpy(), test_proba, threshold=policy.f2_threshold
    )
    warning_metrics.update(
        {
            "nHighRiskEligible": n_high_eligible,
            "nHighRiskTest": n_high_test,
            "operatingThreshold": policy.f2_threshold,
            "mixKind": policy.mix_kind,
            "floorThreshold": policy.floor_threshold,
            "promoteHighRisk": policy.promote_high_risk,
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
        "model": regression_metrics(test["y"].to_numpy(), ens_pred),
        "persistence": regression_metrics(test["y"].to_numpy(), persist_test),
        "maeImprovement": float(
            regression_metrics(test["y"].to_numpy(), persist_test)["mae"]
            - regression_metrics(test["y"].to_numpy(), ens_pred)["mae"]
        ),
    }

    metrics = {
        "modelVersion": MODEL_VERSION,
        "researchQuestion": RESEARCH_QUESTION,
        "definitions": {
            "mae": "Mean absolute error in log10(Pc) units on the final reported risk.",
            "esaLoss": ESA_LOSS_DEFINITION,
            "falseReassurance": FALSE_REASSURANCE_DEFINITION,
        },
        "dataSource": data_source,
        "dataSourceKind": source,
        "selectedPolicy": (
            "hurdle residual xgboost (MAE loss) mixed with a collapse-to-floor "
            "classifier, persistence guard at the ESA −6 class, and conformal "
            "intervals; object type and encounter-plane geometry are in the "
            "snapshot features. The −6 class follows the ESA challenge "
            "definition. The guard and 1.25 disagreement threshold were fixed "
            "before test evaluation."
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
            "method": "split conformal bands localized by predicted floor vs moving events",
            "interpretation": (
                "Intervals are split-conformal residuals on the calibration "
                "set, localized by whether the hurdle predicts a floor collapse. "
                "Bootstrap disagreement remains an abstention trigger."
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
            "nModels": len(policy.ensemble),
            "mixKind": policy.mix_kind,
            "floorThreshold": policy.floor_threshold,
        },
        "robustness": _robustness_slices(test, ens_pred),
        "missionIdComparison": {
            "why": (
                "Adding mission_id provides negligible improvement and does not "
                "materially change performance, so it is excluded from the "
                "deployed exhibit."
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
            interval90=interval_90,
        ),
        "shapContrast": shap_outcome_contrast(policy.residual, test, ens_pred),
        "failureClusters": cluster_test_failures(test, ens_pred, persist_test),
        "featureGroups": grouped_importance(
            policy.residual.feature_names,
            policy.residual.model.get_booster().get_score(importance_type="gain"),
        ),
        "failures": error_gallery(
            test["event_id"].to_numpy(),
            test["y"].to_numpy(),
            ens_pred,
            persist_test,
        ),
        "abstentionRule": ABSTENTION_RULE,
    }

    event_by_id = {int(event["event_id"]): event for event in events}
    all_out = predict_hurdle(policy, features)
    aligned_all = features.reindex(columns=policy.feature_names).apply(
        pd.to_numeric, errors="coerce"
    )
    all_probability = calibrator.predict_proba(all_out.warning_proba)
    all_abstained, _, _ = abstain_mask(
        all_out.ensemble,
        features["risk"].to_numpy(dtype=float),
        features["miss_distance"].to_numpy(dtype=float),
        interval90=all_out.interval90,
        point=all_out.point,
        warning_proba=all_out.warning_proba,
        warning_threshold=policy.f2_threshold,
    )
    event_ids = features["event_id"].astype(int).to_numpy()
    predictions = {
        int(event_id): {
            "predictedFinalRiskLog10": float(all_out.point[position]),
            "configuredHighRiskProbability": float(all_probability[position]),
            "abstained": bool(all_abstained[position]),
        }
        for position, event_id in enumerate(event_ids)
    }
    explainer = shap_explainer(policy.residual, train)
    demo_cases = assemble_demo_cases(
        slots=DEMO_SLOTS,
        predictions=predictions,
        features=features,
        event_by_id=event_by_id,
        aligned=aligned_all,
        ensemble_matrix=all_out.ensemble,
        trained=policy.residual,
        calibrator=calibrator,
        explainer=explainer,
        test_ids=set(int(event_id) for event_id in splits.test_ids),
        points=all_out.point,
        interval90=all_out.interval90,
        interval50=all_out.interval50,
        warning_scores=all_out.warning_proba,
    )

    policy.residual.model.save_model(artifacts / "risk_regressor.json")
    joblib.dump(
        {
            "calibrator": calibrator,
            "feature_names": policy.feature_names,
            "ensemble": policy.ensemble,
            "collapse": policy.collapse,
            "warning": policy.warning,
            "mix_kind": policy.mix_kind,
            "floor_threshold": policy.floor_threshold,
            "f2_threshold": policy.f2_threshold,
            "promote_high_risk": policy.promote_high_risk,
            "conformal": policy.conformal,
        },
        artifacts / "warning_calibrator.joblib",
    )
    write_json(artifacts / "feature_schema.json", {"features": policy.feature_names})
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
            "falseReassuranceDefinition": FALSE_REASSURANCE_DEFINITION,
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
