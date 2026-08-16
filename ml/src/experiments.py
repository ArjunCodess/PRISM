from __future__ import annotations

import numpy as np
import pandas as pd
from abstention import coverage_curve, selective_metrics
from build_events import build_event_histories
from constants import HIGH_RISK_THRESHOLD, HORIZON_HOURS, NEGLIGIBLE_RISK
from evaluate import regression_metrics
from explain import feature_group, shap_explainer
from feature_sets import FAMILIES, columns_for_family
from features import build_feature_table
from split import subset
from train_regressor import fit_xgboost, persistence_predict, predict_model


def _keep_columns(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    feature_cols = columns_for_family(list(frame.columns), family)
    identity = [column for column in ("event_id", "y", "c_object_type") if column in frame.columns]
    return frame[identity + feature_cols].copy()


def historical_ablation(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, object]:
    y = test["y"].to_numpy(dtype=float)
    persist = persistence_predict(test)
    persist_mae = float(regression_metrics(y, persist)["mae"])
    families: dict[str, object] = {}
    previous_mae: float | None = None
    for family in FAMILIES:
        trained = fit_xgboost(_keep_columns(train, family))
        pred = predict_model(trained, _keep_columns(test, family))
        metrics = regression_metrics(y, pred)
        families[family] = {
            "nFeatures": len(trained.feature_names),
            "mae": metrics["mae"],
            "esa_loss": metrics["esa_loss"],
            "mae_high_risk": metrics["mae_high_risk"],
            "deltaFromPreviousMae": (
                None if previous_mae is None else float(previous_mae - metrics["mae"])
            ),
            "beatsPersistenceMae": bool(metrics["mae"] < persist_mae),
        }
        previous_mae = float(metrics["mae"])
    persist_metrics = regression_metrics(y, persist)
    snapshot_mae = float(families["snapshot"]["mae"])  # type: ignore[index]
    history_mae = float(families["snapshot_history"]["mae"])  # type: ignore[index]
    return {
        "question": (
            "Does historical evolution add predictive information beyond the latest snapshot?"
        ),
        "families": families,
        "persistenceMae": persist_metrics["mae"],
        "historyDeltaMae": float(snapshot_mae - history_mae),
        "historyHelps": bool(history_mae < snapshot_mae),
    }


def forecast_horizon_table(
    frame: pd.DataFrame,
    train_ids: list[int],
    test_ids: list[int],
    primary: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    train_id_set = set(train_ids)
    test_id_set = set(test_ids)
    for hours in HORIZON_HOURS:
        if hours == 48 and primary is not None:
            rows.append(primary)
            continue
        cutoff_days = hours / 24.0
        events = build_event_histories(frame, cutoff_days=cutoff_days)
        features = build_feature_table(events)
        eligible_ids = set(int(value) for value in features["event_id"].to_numpy())
        train = subset(features, [event_id for event_id in train_ids if event_id in eligible_ids])
        test = subset(features, [event_id for event_id in test_ids if event_id in eligible_ids])
        if len(train) < 40 or len(test) < 15:
            rows.append(
                {
                    "cutoffHours": hours,
                    "eligibleEvents": len(features),
                    "trainEvents": len(train),
                    "testEvents": len(test),
                    "skipped": True,
                }
            )
            continue
        trained = fit_xgboost(train)
        pred = predict_model(trained, test)
        persist = persistence_predict(test)
        y = test["y"].to_numpy(dtype=float)
        model_metrics = regression_metrics(y, pred)
        persist_metrics = regression_metrics(y, persist)
        rows.append(
            {
                "cutoffHours": hours,
                "eligibleEvents": len(features),
                "trainEvents": len(train),
                "testEvents": len(test),
                "overlapTrain": int(
                    sum(1 for event_id in train_id_set if event_id in eligible_ids)
                ),
                "overlapTest": int(sum(1 for event_id in test_id_set if event_id in eligible_ids)),
                "model": model_metrics,
                "persistence": persist_metrics,
                "maeImprovement": float(persist_metrics["mae"] - model_metrics["mae"]),
            }
        )
    return rows


def shap_outcome_contrast(
    trained,
    test: pd.DataFrame,
    predictions: np.ndarray,
    max_rows: int = 400,
) -> dict[str, object]:
    explainer = shap_explainer(trained, test[trained.feature_names].head(20))
    y = test["y"].to_numpy(dtype=float)
    abs_err = np.abs(predictions - y)
    correct = abs_err <= 0.5
    incorrect = abs_err >= 2.0
    x = test[trained.feature_names].apply(pd.to_numeric, errors="coerce")

    def summarize(mask: np.ndarray) -> dict[str, object]:
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            return {"n": 0, "groups": []}
        if len(indices) > max_rows:
            rng = np.random.default_rng(0)
            indices = np.sort(rng.choice(indices, size=max_rows, replace=False))
        values = np.asarray(explainer.shap_values(x.iloc[indices]), dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        mean_abs = np.mean(np.abs(values), axis=0)
        totals: dict[str, float] = {}
        for name, score in zip(trained.feature_names, mean_abs, strict=True):
            group = feature_group(name)
            totals[group] = totals.get(group, 0.0) + float(score)
        ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
        return {
            "n": int(mask.sum()),
            "nExplained": int(len(indices)),
            "meanAbsError": float(np.mean(abs_err[mask])),
            "groups": [
                {"group": name, "meanAbsShap": value}
                for name, value in ranked
                if value > 0
            ],
        }

    return {
        "correct": summarize(correct),
        "incorrect": summarize(incorrect),
        "rule": "correct: |error| ≤ 0.5; incorrect: |error| ≥ 2.0; single XGBoost model",
    }


def cluster_test_failures(
    test: pd.DataFrame,
    predictions: np.ndarray,
    persist: np.ndarray,
) -> dict[str, object]:
    y = test["y"].to_numpy(dtype=float)
    pred = np.asarray(predictions, dtype=float)
    persist = np.asarray(persist, dtype=float)
    err = pred - y
    abs_err = np.abs(err)
    n_messages = test["n_messages"].to_numpy(dtype=float)
    miss = test["miss_distance"].to_numpy(dtype=float)
    hours = test["hours_before_cutoff"].to_numpy(dtype=float)
    sigma = test["t_sigma_r"].fillna(0).to_numpy(dtype=float) + test[
        "c_sigma_r"
    ].fillna(0).to_numpy(dtype=float)
    high = y >= HIGH_RISK_THRESHOLD
    pred_high = pred >= HIGH_RISK_THRESHOLD
    modes = []
    for i in range(len(test)):
        if abs_err[i] <= 0.5:
            modes.append("accurate")
        elif y[i] <= NEGLIGIBLE_RISK + 0.5 and pred[i] > -15:
            modes.append("final_collapses_to_negligible")
        elif high[i] and persist[i] < HIGH_RISK_THRESHOLD - 0.25 and not pred_high[i]:
            modes.append("late_high_risk_jump")
        elif high[i] and not pred_high[i]:
            modes.append("missed_high_risk")
        elif (not high[i]) and pred_high[i]:
            modes.append("false_high_risk")
        elif n_messages[i] <= 1:
            modes.append("sparse_history_error")
        elif miss[i] < 500:
            modes.append("close_approach_error")
        elif err[i] < -1:
            modes.append("underprediction")
        elif err[i] > 1:
            modes.append("overprediction")
        else:
            modes.append("moderate_error")
    mode_array = np.asarray(modes)
    summary: dict[str, object] = {}
    for mode in (
        "accurate",
        "final_collapses_to_negligible",
        "late_high_risk_jump",
        "missed_high_risk",
        "false_high_risk",
        "sparse_history_error",
        "close_approach_error",
        "underprediction",
        "overprediction",
        "moderate_error",
    ):
        mask = mode_array == mode
        if not mask.any():
            continue
        summary[mode] = {
            "n": int(mask.sum()),
            "share": float(mask.mean()),
            "mae": float(np.mean(abs_err[mask])),
            "highRiskShare": float(high[mask].mean()),
            "meanMessages": float(np.mean(n_messages[mask])),
            "meanMissDistanceM": float(np.nanmean(miss[mask])),
            "meanHoursBeforeCutoff": float(np.mean(hours[mask])),
            "meanRadialSigmaM": float(np.mean(sigma[mask])),
        }
    dominant_failures = [
        name
        for name, payload in sorted(
            summary.items(), key=lambda item: int(item[1]["n"]), reverse=True
        )
        if name != "accurate"
    ][:3]
    return {
        "modes": summary,
        "dominantFailures": dominant_failures,
        "nTest": int(len(test)),
        "nInaccurate": int((mode_array != "accurate").sum()),
    }


def abstention_study(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    persist: np.ndarray,
    ensemble_matrix: np.ndarray,
    current_risk: np.ndarray,
    miss_distance: np.ndarray,
    abstained: np.ndarray,
    proba: np.ndarray,
) -> dict[str, object]:
    operating = selective_metrics(y_true, y_pred, persist, abstained, proba=proba)
    curve = coverage_curve(
        y_true, y_pred, persist, ensemble_matrix, current_risk, miss_distance
    )
    return {
        "rule": (
            "Abstain if the 90% bootstrap band crosses log10(Pc) ≥ −6, "
            "if current risk or miss distance is missing, or if bootstrap "
            "disagreement exceeds 1.25 log-risk units. These thresholds were "
            "locked before test evaluation."
        ),
        "operatingPoint": operating,
        "coverageCurve": curve,
    }
