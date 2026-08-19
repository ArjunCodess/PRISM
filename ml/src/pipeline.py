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
    DEMO_SLOTS,
    ESA_LOSS_DEFINITION,
    FALSE_REASSURANCE_DEFINITION,
    HIGH_RISK_THRESHOLD,
    RANDOM_STATE,
    RESEARCH_QUESTION,
)
from evaluate import (  # noqa: E402
    classification_metrics,
    error_gallery,
    honest_metrics_bundle,
    level_scoreboard_row,
    persistence_improvement,
    pick_validation_winner,
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
from ingest import (  # noqa: E402
    load_esa_training,
    realistic_training_events,
    validate_official_test_compatibility,
)
from split import grouped_splits, subset  # noqa: E402
from train_classifier import fit_warning_classifier  # noqa: E402
from train_regressor import (  # noqa: E402
    fit_residual_xgboost,
    fit_ridge,
    fit_xgboost,
    median_predict,
    persistence_predict,
    predict_model,
    predict_reconstructed,
)
from validate import validate_cdm_frame  # noqa: E402

HONEST_DEFINITIONS = {
    "floorExcludedMae": (
        "MAE on events whose final reported log10(Pc) is above the dataset floor of -30."
    ),
    "residualMaeActual": (
        "Mean |y - risk|: how far the later report moved from the T-48 snapshot."
    ),
    "residualMaePredicted": "Mean |pred - risk|: how far the model moves the snapshot.",
    "residualMae": "Mean |(y - risk) - (pred - risk)|, equal to MAE(y, pred).",
    "maeAdvantageCi": (
        "95% bootstrap interval on MAE(persistence) - MAE(model) from 1000 event resamples."
    ),
}


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


def _write_honest_metrics(
    metrics: dict[str, object],
    y_true: np.ndarray,
    risk: np.ndarray,
    persist: np.ndarray,
    xgb_pred: np.ndarray,
    ens_pred: np.ndarray,
    residual_pred: np.ndarray | None = None,
) -> dict[str, object]:
    definitions = dict(metrics.get("definitions") or {})  # type: ignore[arg-type]
    definitions.update(HONEST_DEFINITIONS)
    metrics["definitions"] = definitions
    honest = honest_metrics_bundle(
        y_true, risk, persist, xgb_pred, ens_pred, residual_pred=residual_pred
    )
    metrics["honestMetrics"] = honest
    return honest


def score_frozen_honest_metrics() -> dict[str, object]:
    artifacts = ROOT / "ml" / "artifacts"
    manifest_path = artifacts / "split_manifest.json"
    metrics_path = artifacts / "metrics.json"
    if not manifest_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("frozen split_manifest.json and metrics.json are required")

    raw = realistic_training_events(load_esa_training(ROOT / "data" / "raw"))
    features = build_feature_table(build_event_histories(validate_cdm_frame(raw)))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test = subset(features, manifest["test"])
    if test.empty:
        raise RuntimeError("frozen test ids did not match the rebuilt feature table")

    train = subset(features, manifest["train"])
    validation = subset(features, manifest["validation"])
    if train.empty or validation.empty:
        raise RuntimeError("frozen train or validation ids did not match the rebuilt feature table")

    persist_test = persistence_predict(test)
    persist_val = persistence_predict(validation)
    schema = json.loads((artifacts / "feature_schema.json").read_text(encoding="utf-8"))["features"]
    booster_model = XGBRegressor()
    booster_model.load_model(str(artifacts / "risk_regressor.json"))
    x_test = test[schema].apply(pd.to_numeric, errors="coerce")
    model_pred = np.asarray(booster_model.predict(x_test), dtype=float)
    xgb_val = np.asarray(
        booster_model.predict(validation[schema].apply(pd.to_numeric, errors="coerce")),
        dtype=float,
    )

    bundle = joblib.load(artifacts / "warning_calibrator.joblib")
    ensemble = bundle["ensemble"]
    feature_names = list(bundle.get("feature_names") or schema)
    x_ens = test[feature_names].apply(pd.to_numeric, errors="coerce")
    raw_ens_matrix = np.column_stack([model.predict(x_ens) for model in ensemble])
    persist_guard = test["risk"].to_numpy(dtype=float) >= HIGH_RISK_THRESHOLD
    ens_matrix = np.where(
        persist_guard[:, None], test["risk"].to_numpy(dtype=float)[:, None], raw_ens_matrix
    )
    ens_pred = np.median(ens_matrix, axis=1)

    residual = fit_residual_xgboost(train)
    residual.model.save_model(str(artifacts / "residual_regressor.json"))
    residual_test = predict_reconstructed(residual, test)
    residual_val = predict_reconstructed(residual, validation)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    honest = _write_honest_metrics(
        metrics,
        test["y"].to_numpy(),
        test["risk"].to_numpy(dtype=float),
        persist_test,
        model_pred,
        ens_pred,
        residual_pred=residual_test,
    )
    test_y = test["y"].to_numpy()
    val_y = validation["y"].to_numpy()
    test_board = {
        "persistence": level_scoreboard_row(test_y, persist_test),
        "xgboost": level_scoreboard_row(test_y, model_pred),
        "residual": level_scoreboard_row(test_y, residual_test),
    }
    val_board = {
        "persistence": level_scoreboard_row(val_y, persist_val),
        "xgboost": level_scoreboard_row(val_y, xgb_val),
        "residual": level_scoreboard_row(val_y, residual_val),
    }
    metrics["residualModel"] = {
        "target": "y - risk on cutoff-safe rows; reconstruct pred = risk + residual_hat",
        "fitOn": "frozen train event ids only",
        "artifact": "ml/artifacts/residual_regressor.json",
        "replacesExhibit": False,
        "test": test_board,
        "validation": val_board,
        "winnerSoFar": {
            "split": "validation",
            "criterion": "mae, then floor-excluded mae, then esa-style loss",
            "name": pick_validation_winner(val_board),
        },
    }
    write_json(metrics_path, metrics)
    card_path = artifacts / "model_card.json"
    if card_path.exists():
        card = json.loads(card_path.read_text(encoding="utf-8"))
        card["honestMetrics"] = honest
        card["residualModel"] = metrics["residualModel"]
        write_json(card_path, card)
    return honest


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
        "researchQuestion": RESEARCH_QUESTION,
        "definitions": {
            "mae": "Mean absolute error in log10(Pc) units on the final reported risk.",
            "esaLoss": ESA_LOSS_DEFINITION,
            "falseReassurance": FALSE_REASSURANCE_DEFINITION,
        },
        "dataSource": data_source,
        "dataSourceKind": source,
        "selectedPolicy": (
            "T−48 bootstrap xgboost median of the final reported log10(Pc), "
            "with a persistence guard at the ESA −6 class; features use only "
            "messages with time_to_tca ≥ 2 days. The −6 class follows the ESA "
            "challenge definition. The guard and 1.25 disagreement threshold "
            "were fixed before test evaluation."
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
    _write_honest_metrics(
        metrics,
        test["y"].to_numpy(),
        test["risk"].to_numpy(dtype=float),
        persist_test,
        model_pred,
        ens_pred,
    )

    event_by_id = {int(event["event_id"]): event for event in events}
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
    predictions = {
        int(event_id): {
            "predictedFinalRiskLog10": float(all_point[position]),
            "configuredHighRiskProbability": float(all_probability[position]),
            "abstained": bool(all_abstained[position]),
        }
        for position, event_id in enumerate(event_ids)
    }
    explainer = shap_explainer(booster, train)
    demo_cases = assemble_demo_cases(
        slots=DEMO_SLOTS,
        predictions=predictions,
        features=features,
        event_by_id=event_by_id,
        aligned=aligned_all,
        ensemble_matrix=all_ensemble,
        trained=booster,
        calibrator=calibrator,
        explainer=explainer,
        test_ids=set(int(event_id) for event_id in splits.test_ids),
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
            "honestMetrics": metrics["honestMetrics"],
        },
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and export PRISM artifacts")
    parser.add_argument("--source", choices=("real", "synthetic"), default="real")
    parser.add_argument("--synthetic-events", type=int, default=420)
    parser.add_argument(
        "--frozen",
        action="store_true",
        help="score frozen exhibit models and the residual candidate; do not replace the exhibit",
    )
    args = parser.parse_args()
    if args.frozen:
        honest = score_frozen_honest_metrics()
        print(json.dumps(honest, indent=2, default=str))
    else:
        result = run_pipeline(source=args.source, n_events=args.synthetic_events)
        print(json.dumps(result["improvement"], indent=2))
        print(json.dumps(result["ensemble"], indent=2))
