from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from abstention import (
    REASON_CONFORMAL,
    REASON_MISSING,
    AbstentionDecision,
    conformal_abstain_mask,
    selective_metrics,
)
from calibrate import (
    conformal_bounds,
    fit_absolute_conformal,
    fit_isotonic,
    interval_report,
)
from constants import HIGH_RISK_THRESHOLD
from floor_model import combine_floor_hurdle, predict_floor_proba
from train_classifier import TrainedClassifier
from train_regressor import TrainedRegressor, predict_reconstructed
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[2]


def booster_feature_names(model: XGBRegressor | XGBClassifier, fallback: list[str]) -> list[str]:
    stored = model.get_booster().feature_names
    if stored:
        return list(stored)
    return list(fallback)


def subset(frame: pd.DataFrame, ids: list[object]) -> pd.DataFrame:
    wanted = {int(event_id) for event_id in ids}
    return frame.loc[frame["event_id"].map(int).isin(wanted)].copy()


def load_floor_hurdle(artifacts: Path, schema: list[str]) -> dict[str, Any]:
    floor_clf_model = XGBClassifier()
    floor_clf_model.load_model(str(artifacts / "floor_classifier.json"))
    floor_clf = TrainedClassifier(
        model=floor_clf_model,
        feature_names=booster_feature_names(floor_clf_model, schema),
    )
    floor_residual_model = XGBRegressor()
    floor_residual_model.load_model(str(artifacts / "floor_residual_regressor.json"))
    floor_residual = TrainedRegressor(
        model=floor_residual_model,
        feature_names=booster_feature_names(floor_residual_model, schema),
        kind="residual_xgboost",
    )
    policy = json.loads((artifacts / "floor_hurdle.json").read_text(encoding="utf-8"))
    return {
        "classifier": floor_clf,
        "residual": floor_residual,
        "threshold": float(policy["threshold"]),
        "usePersistGuard": bool(policy["usePersistGuard"]),
        "policy": policy,
    }


def predict_floor_hurdle(bundle: dict[str, Any], frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    proba = predict_floor_proba(bundle["classifier"], frame)
    reconstructed = predict_reconstructed(bundle["residual"], frame)
    pred = combine_floor_hurdle(
        proba,
        reconstructed,
        bundle["threshold"],
        risk=frame["risk"].to_numpy(dtype=float),
        use_persist_guard=bundle["usePersistGuard"],
    )
    return pred, proba


def decide_policy_abstention(
    *,
    rule: str,
    current_risk: float,
    miss_distance: float,
    lo90: float,
    hi90: float,
) -> AbstentionDecision:
    missing = bool(not np.isfinite(current_risk) or not np.isfinite(miss_distance))
    crosses = bool(lo90 < HIGH_RISK_THRESHOLD <= hi90)
    reasons: list[str] = []
    if missing:
        reasons.append(REASON_MISSING)
    if rule == "conformal" and crosses:
        reasons.append(REASON_CONFORMAL)
    return AbstentionDecision(
        abstained=bool(reasons),
        reasons=reasons,
        disagreement=0.0,
        crosses_threshold=crosses,
        missing_critical=missing,
    )


def _missing_only_mask(current_risk: np.ndarray, miss_distance: np.ndarray) -> np.ndarray:
    return (~np.isfinite(current_risk) | ~np.isfinite(miss_distance)).astype(bool)


def _abstention_rule_text(rule: str) -> str:
    if rule == "conformal":
        return (
            "PRISM abstains when the 90% conformal band around the floor-hurdle "
            "forecast crosses the ESA challenge class log10(Pc) ≥ −6, or when "
            "current risk or miss distance is missing. The −6 class follows the "
            "ESA challenge definition. The floor threshold and the decision not "
            "to copy today's report were frozen on validation before the test split."
        )
    return (
        "PRISM abstains when current risk or miss distance is missing. The −6 "
        "class follows the ESA challenge definition. The floor threshold and the "
        "decision not to copy today's report were frozen on validation before "
        "the test split."
    )


def ship_selected_policy(
    *,
    features: pd.DataFrame,
    events: list[dict[str, Any]],
    manifest: dict[str, Any],
    artifacts: Path,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    winner = str(metrics["floorModel"]["winnerSoFar"]["name"])
    if winner != "floorHurdle":
        raise RuntimeError(f"validation winner is {winner}, not floorHurdle")

    schema = json.loads((artifacts / "feature_schema.json").read_text(encoding="utf-8"))["features"]
    bundle = load_floor_hurdle(artifacts, schema)
    validation = subset(features, manifest["validation"])
    calibration = subset(features, manifest["calibration"])
    test = subset(features, manifest["test"])
    if validation.empty or calibration.empty or test.empty:
        raise RuntimeError("frozen split ids did not match the rebuilt feature table")

    floor_val, _ = predict_floor_hurdle(bundle, validation)
    floor_cal, _ = predict_floor_hurdle(bundle, calibration)
    floor_test, _ = predict_floor_hurdle(bundle, test)
    floor_all, proba_all = predict_floor_hurdle(bundle, features)

    quantiles = fit_absolute_conformal(
        calibration["y"].to_numpy(), floor_cal, alphas=(0.5, 0.1)
    )
    q50 = float(quantiles[0.5])
    q90 = float(quantiles[0.1])
    conf50_test = conformal_bounds(floor_test, q50)
    conf90_test = conformal_bounds(floor_test, q90)
    conf90_val = conformal_bounds(floor_val, q90)
    conf90_all = conformal_bounds(floor_all, q90)

    persist_val = validation["risk"].to_numpy(dtype=float)
    miss_val = validation["miss_distance"].to_numpy(dtype=float)
    val_y = validation["y"].to_numpy()
    missing_val = _missing_only_mask(persist_val, miss_val)
    conf_val, _ = conformal_abstain_mask(conf90_val[0], conf90_val[1], persist_val, miss_val)
    missing_sel = selective_metrics(val_y, floor_val, persist_val, missing_val)
    conf_sel = selective_metrics(val_y, floor_val, persist_val, conf_val)
    missing_key = (int(missing_sel["falseReassurance"]), -float(missing_sel["coverage"]))
    conf_key = (int(conf_sel["falseReassurance"]), -float(conf_sel["coverage"]))
    chosen_rule = "conformal" if conf_key < missing_key else "missingOnly"

    persist_test = test["risk"].to_numpy(dtype=float)
    miss_test = test["miss_distance"].to_numpy(dtype=float)
    test_y = test["y"].to_numpy()
    missing_test = _missing_only_mask(persist_test, miss_test)
    conf_test, _ = conformal_abstain_mask(conf90_test[0], conf90_test[1], persist_test, miss_test)
    live_abs_test = conf_test if chosen_rule == "conformal" else missing_test
    live_abs_val = conf_val if chosen_rule == "conformal" else missing_val

    calibrator = fit_isotonic(floor_cal, (calibration["y"].to_numpy() >= HIGH_RISK_THRESHOLD))
    joblib.dump(
        {"calibrator": calibrator, "feature_names": schema, "pointPredictor": "floorHurdle"},
        artifacts / "exhibit_calibrator.joblib",
    )
    exhibit_conformal = {
        "q50": q50,
        "q90": q90,
        "replacesExhibit": True,
        "pointPredictor": "floorHurdle",
        "fitOn": "frozen calibration event ids only",
    }
    (artifacts / "exhibit_conformal.json").write_text(
        json.dumps(exhibit_conformal, indent=2) + "\n", encoding="utf-8"
    )

    policy_payload = {
        "name": "floorHurdle",
        "chosenOn": "validation",
        "criterion": "mae, then floor-excluded mae, then esa-style loss",
        "usePersistGuard": bundle["usePersistGuard"],
        "threshold": bundle["threshold"],
        "interval": "split conformal around the floor-hurdle point",
        "abstention": chosen_rule,
        "abstentionRule": _abstention_rule_text(chosen_rule),
        "artifacts": {
            "classifier": "ml/artifacts/floor_classifier.json",
            "residualRegressor": "ml/artifacts/floor_residual_regressor.json",
            "policy": "ml/artifacts/floor_hurdle.json",
            "calibrator": "ml/artifacts/exhibit_calibrator.joblib",
            "conformal": "ml/artifacts/exhibit_conformal.json",
        },
    }
    (artifacts / "selected_policy.json").write_text(
        json.dumps(policy_payload, indent=2) + "\n", encoding="utf-8"
    )

    hurdle = dict(bundle["policy"])
    hurdle["replacesExhibit"] = True
    (artifacts / "floor_hurdle.json").write_text(
        json.dumps(hurdle, indent=2) + "\n", encoding="utf-8"
    )

    live_sel_test = selective_metrics(test_y, floor_test, persist_test, live_abs_test)
    live_sel_val = selective_metrics(val_y, floor_val, persist_val, live_abs_val)
    honest_floor = metrics["honestMetrics"]["systems"]["floorHurdle"]
    metrics["augustExhibitSnapshot"] = {
        "name": "bootstrapXgboostMedianPersistGuard",
        "note": "18 August exhibit policy; research baseline only, not a live mode",
        "ensemble": metrics["ensemble"],
        "abstention": metrics.get("abstention"),
        "uncertainty": metrics.get("uncertainty"),
    }
    metrics["selectedPolicy"] = {
        **policy_payload,
        "replacesExhibit": True,
        "validation": {"missingOnly": missing_sel, "conformal": conf_sel, "chosen": live_sel_val},
        "test": {"missingOnly": selective_metrics(test_y, floor_test, persist_test, missing_test),
                 "conformal": selective_metrics(test_y, floor_test, persist_test, conf_test),
                 "chosen": live_sel_test},
        "exhibitConformal": {
            "50": interval_report(test_y, conf50_test[0], conf50_test[1]),
            "90": interval_report(test_y, conf90_test[0], conf90_test[1]),
            "q50": q50,
            "q90": q90,
        },
        "liveMae": honest_floor["all"]["mae"],
        "liveMedianAe": honest_floor["all"]["median_ae"],
        "liveFloorExcludedMae": honest_floor["nonFloor"]["mae"],
        "liveEsaLoss": metrics["floorModel"]["test"]["floorHurdle"]["esaLoss"],
        "liveF2": metrics["floorModel"]["test"]["floorHurdle"]["f2"],
    }
    metrics["floorModel"]["replacesExhibit"] = True
    metrics["abstentionRule"] = policy_payload["abstentionRule"]
    metrics["definitions"] = dict(metrics.get("definitions") or {})
    metrics["definitions"]["selectedPolicy"] = (
        "Validation winner on MAE, then floor-excluded MAE, then ESA-style loss. "
        "Test and official-test are reports only."
    )

    from export_demo_cases import assemble_demo_cases_from_model, write_json
    from inference import PrismModel

    model = PrismModel()
    persist_all = features["risk"].to_numpy(dtype=float)
    miss_all = features["miss_distance"].to_numpy(dtype=float)
    if chosen_rule == "conformal":
        abstained_all, _ = conformal_abstain_mask(
            conf90_all[0], conf90_all[1], persist_all, miss_all
        )
    else:
        abstained_all = _missing_only_mask(persist_all, miss_all)
    predictions = {
        int(event_id): {
            "predictedFinalRiskLog10": float(floor_all[index]),
            "abstained": bool(abstained_all[index]),
        }
        for index, event_id in enumerate(features["event_id"].map(int).to_numpy())
    }
    event_by_id = {int(event["event_id"]): event for event in events}
    demo_cases = assemble_demo_cases_from_model(
        predictions=predictions,
        features=features,
        event_by_id=event_by_id,
        model=model,
        test_ids={int(event_id) for event_id in manifest["test"]},
    )
    write_json(artifacts / "demo_cases.json", demo_cases)
    write_json(ROOT / "apps" / "web" / "public" / "demo_cases.json", demo_cases)
    write_json(artifacts / "metrics.json", metrics)
    write_json(ROOT / "apps" / "web" / "public" / "metrics.json", metrics)

    card_path = artifacts / "model_card.json"
    if card_path.exists():
        card = json.loads(card_path.read_text(encoding="utf-8"))
        card["selectedPolicy"] = metrics["selectedPolicy"]
        card["augustExhibitSnapshot"] = {
            "name": metrics["augustExhibitSnapshot"]["name"],
            "note": metrics["augustExhibitSnapshot"]["note"],
            "mae": metrics["ensemble"]["mae"],
            "esa_loss": metrics["ensemble"]["esa_loss"],
        }
        card["floorModel"] = metrics["floorModel"]
        card["abstentionRule"] = metrics["abstentionRule"]
        write_json(card_path, card)
        write_json(ROOT / "apps" / "web" / "public" / "model_card.json", card)

    return metrics
