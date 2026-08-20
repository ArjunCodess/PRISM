from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from abstention import conformal_abstain_mask, selective_metrics
from build_events import build_event_histories
from calibrate import conformal_bounds, interval_report, split_conformal_quantile
from constants import (
    CLASS_THRESHOLDS,
    FEATURE_DICTIONARY,
    HIGH_RISK_THRESHOLD,
    HORIZON_HOURS,
    NEGLIGIBLE_RISK,
    RANDOM_STATE,
)
from evaluate import (
    binary_classification_metrics,
    bootstrap_mae_advantage,
    floor_mask,
    floor_mask_at,
    floor_sliced_metrics,
    reliability_bins,
)
from feature_sets import (
    IDENTITY,
    SNAPSHOT_DERIVED,
    covariance_trend_columns,
    history_columns,
    snapshot_columns,
)
from features import build_feature_table
from floor_model import floor_confusion
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from split import subset
from train_regressor import (
    _xgboost_regressor,
    fit_xgboost,
    persistence_predict,
    predict_model,
)

CENSOR_THRESHOLDS = (-20.0, -25.0, -30.0)
FLOOR_POLICY_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
CONFORMAL_ALPHAS = (0.50, 0.40, 0.30, 0.20, 0.10, 0.05)
GITHUB_URL = "https://github.com/ArjunCodess/PRISM"


def _json_float(value: float) -> float:
    if value is None or not np.isfinite(value):
        return float("nan")
    return float(value)


def _mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y, dtype=float) - np.asarray(pred, dtype=float))))


def _system_slice(y: np.ndarray, pred: np.ndarray, floor: np.ndarray) -> dict[str, float | int]:
    sliced = {
        "mae": _mae(y, pred),
        "floorExcludedMae": _mae(y[~floor], pred[~floor]) if np.any(~floor) else float("nan"),
        "floorMae": _mae(y[floor], pred[floor]) if np.any(floor) else float("nan"),
        "nFloor": int(np.sum(floor)),
        "nNonFloor": int(np.sum(~floor)),
        "floorRate": float(np.mean(floor)) if y.size else 0.0,
    }
    return sliced


def reproducibility_spec() -> dict[str, object]:
    booster = _xgboost_regressor()
    return {
        "code": GITHUB_URL,
        "dataset": "ESA Collision Avoidance Challenge / Zenodo 4463683",
        "splitSeed": RANDOM_STATE,
        "earlyStopping": False,
        "missingValueHandling": "xgboost native nan",
        "regressor": {
            "objective": booster.objective,
            "nEstimators": int(booster.n_estimators),
            "learningRate": float(booster.learning_rate),
            "maxDepth": int(booster.max_depth),
            "minChildWeight": float(booster.min_child_weight),
            "subsample": float(booster.subsample),
            "colsampleBytree": float(booster.colsample_bytree),
            "regAlpha": float(booster.reg_alpha),
            "regLambda": float(booster.reg_lambda),
            "randomState": int(booster.random_state),
        },
        "floorClassifier": {
            "nEstimators": 180,
            "learningRate": 0.05,
            "maxDepth": 3,
            "minChildWeight": 5,
            "subsample": 0.85,
            "colsampleBytree": 0.8,
            "evalMetric": "logloss",
            "scalePosWeight": "neg/pos on train",
            "randomState": RANDOM_STATE,
        },
        "note": (
            "Hyperparameters are the frozen exhibit specification. "
            "They were not retuned for this review-armor pass."
        ),
    }


def feature_catalog(columns: list[str]) -> dict[str, object]:
    present = [name for name in columns if name not in IDENTITY]
    snapshot = [name for name in present if name in set(snapshot_columns())]
    history = [name for name in present if name in set(history_columns())]
    covariance = [name for name in present if name in set(covariance_trend_columns())]
    other = [
        name
        for name in present
        if name not in set(snapshot) | set(history) | set(covariance)
    ]
    groups = {
        "snapshot": snapshot,
        "history": history,
        "covarianceTrends": covariance,
        "other": other,
    }
    labeled = []
    for name in present:
        labeled.append(
            {
                "name": name,
                "plain": FEATURE_DICTIONARY.get(name, name.replace("_", " ")),
                "family": next(
                    (family for family, names in groups.items() if name in names),
                    "other",
                ),
                "leakageSafe": True,
            }
        )
    return {
        "nFeatures": len(present),
        "nSnapshot": len(snapshot),
        "nHistory": len(history),
        "nCovarianceTrends": len(covariance),
        "snapshotDerived": list(SNAPSHOT_DERIVED),
        "groups": {key: names for key, names in groups.items()},
        "features": labeled,
        "note": (
            "All features use only cutoff-safe CDM rows. dilution_gap is a research "
            "probe and is excluded from the shipped regressor."
        ),
    }


def censoring_sensitivity(
    y: np.ndarray,
    persist: np.ndarray,
    xgb_pred: np.ndarray,
    floor_pred: np.ndarray,
    thresholds: tuple[float, ...] = CENSOR_THRESHOLDS,
) -> dict[str, object]:
    rows = []
    for threshold in thresholds:
        floor = floor_mask_at(y, threshold)
        rows.append(
            {
                "threshold": float(threshold),
                "floorRate": float(np.mean(floor)),
                "nFloor": int(np.sum(floor)),
                "persistence": _system_slice(y, persist, floor),
                "xgboost": _system_slice(y, xgb_pred, floor),
                "floorHurdle": _system_slice(y, floor_pred, floor),
            }
        )
    return {
        "framing": (
            "−30 is a reporting floor / censored target, not a precise physical Pc. "
            "A two-part hurdle models floor vs non-floor; Tobit-style censored "
            "regression is an alternative not fit here."
        ),
        "thresholds": rows,
    }


def matched_cohort_horizons(
    raw: pd.DataFrame,
    train_ids: list[int],
    test_ids: list[int],
    hours: tuple[int, ...] = HORIZON_HOURS,
) -> dict[str, object]:
    by_hours: dict[int, pd.DataFrame] = {}
    eligible: dict[int, set[int]] = {}
    for hour in hours:
        events = build_event_histories(raw, cutoff_days=hour / 24.0)
        features = build_feature_table(events)
        by_hours[hour] = features
        eligible[hour] = {int(value) for value in features["event_id"].to_numpy()}
    common = set.intersection(*eligible.values()) if eligible else set()
    train_common = [event_id for event_id in train_ids if event_id in common]
    test_common = [event_id for event_id in test_ids if event_id in common]
    rows = []
    for hour in hours:
        train = subset(by_hours[hour], train_common)
        test = subset(by_hours[hour], test_common)
        y = test["y"].to_numpy(dtype=float)
        persist = persistence_predict(test)
        pred = predict_model(fit_xgboost(train), test) if len(train) >= 20 and len(test) >= 8 else persist
        floor = floor_mask(y)
        persist_mae = _mae(y, persist)
        model_mae = _mae(y, pred)
        persist_non = _mae(y[~floor], persist[~floor]) if np.any(~floor) else float("nan")
        model_non = _mae(y[~floor], pred[~floor]) if np.any(~floor) else float("nan")
        advantage = persist_mae - model_mae
        non_floor_delta = (
            persist_non - model_non
            if np.isfinite(persist_non) and np.isfinite(model_non)
            else float("nan")
        )
        ci = bootstrap_mae_advantage(y, pred, persist, n_bootstrap=400)
        rows.append(
            {
                "cutoffHours": int(hour),
                "nTrain": int(len(train)),
                "nTest": int(len(test)),
                "floorRate": float(np.mean(floor)) if y.size else 0.0,
                "nFloor": int(np.sum(floor)),
                "persistenceMae": persist_mae,
                "xgboostMae": model_mae,
                "deltaMae": float(advantage),
                "deltaMaeCi95Low": float(ci["ci95Low"]),
                "deltaMaeCi95High": float(ci["ci95High"]),
                "nonFloorDeltaMae": _json_float(non_floor_delta),
                "persistenceFloorExcludedMae": _json_float(persist_non),
                "xgboostFloorExcludedMae": _json_float(model_non),
            }
        )
    return {
        "nCommonEvents": int(len(common)),
        "nTrain": int(len(train_common)),
        "nTest": int(len(test_common)),
        "note": (
            "Same events observed at 72/48/24/12 h. Train/test IDs are the seed-42 "
            "split intersected with that cohort, so horizon comparisons are not mixed "
            "with a changing eligible set."
        ),
        "rows": rows,
    }


def mission_grouped_split(
    features: pd.DataFrame, seed: int = RANDOM_STATE
) -> dict[str, object]:
    if "mission_id" not in features.columns:
        raise ValueError("mission_id is required for a mission-grouped split")
    unique = features.drop_duplicates("event_id")
    groups = unique["mission_id"].to_numpy()
    ids = unique["event_id"].map(int).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    dummy = np.zeros((len(ids), 1))
    train_idx, test_idx = next(splitter.split(dummy, dummy[:, 0], groups))
    train = subset(features, [int(value) for value in ids[train_idx]])
    test = subset(features, [int(value) for value in ids[test_idx]])
    y = test["y"].to_numpy(dtype=float)
    persist = persistence_predict(test)
    pred = predict_model(fit_xgboost(train), test)
    floor = floor_mask(y)
    high = y >= HIGH_RISK_THRESHOLD
    n_high = int(np.sum(high))
    high_risk_mae = _mae(y[high], pred[high]) if n_high else float("nan")
    persist_hr = _mae(y[high], persist[high]) if n_high else float("nan")
    interpretable = n_high >= 5
    return {
        "grouping": "mission_id",
        "objectPairIdsAvailable": False,
        "calendarDatesAvailable": False,
        "nTrain": int(len(train)),
        "nTest": int(len(test)),
        "nMissionsTrain": int(train["mission_id"].nunique()),
        "nMissionsTest": int(test["mission_id"].nunique()),
        "nHighRiskTest": n_high,
        "highRiskMetricInterpretable": bool(interpretable),
        "highRiskCaveat": (
            None
            if interpretable
            else (
                "High-risk n is too small for statistical interpretation "
                f"(n={n_high})."
            )
        ),
        "xgboost": _system_slice(y, pred, floor),
        "persistence": _system_slice(y, persist, floor),
        "xgboostHighRiskMae": _json_float(high_risk_mae),
        "persistenceHighRiskMae": _json_float(persist_hr),
        "note": (
            "The public archive has no NORAD or object-pair identifiers and no "
            "calendar dates. Mission grouping is the available entity split. "
            "Official-test scoring is the frozen distribution-shift check."
        ),
    }


def floor_classifier_evaluation(
    y: np.ndarray,
    proba: np.ndarray,
    risk: np.ndarray,
    gap: np.ndarray,
    threshold: float = 0.15,
) -> dict[str, object]:
    labels = floor_mask(y).astype(int)
    scores = binary_classification_metrics(labels, proba, threshold=threshold)
    grid = [floor_confusion(y, proba, float(cut)) for cut in FLOOR_POLICY_THRESHOLDS]
    called = np.asarray(proba, dtype=float) >= threshold
    actual = labels.astype(bool)
    fp = called & ~actual
    fn = ~called & actual
    fp_y = y[fp]
    return {
        "threshold": float(threshold),
        "scores": scores,
        "reliability": reliability_bins(y, proba, labels=labels.astype(float)),
        "confusionGrid": grid,
        "falsePositives": {
            "n": int(np.sum(fp)),
            "meanY": _json_float(float(np.mean(fp_y)) if fp_y.size else float("nan")),
            "medianY": _json_float(float(np.median(fp_y)) if fp_y.size else float("nan")),
            "shareYBelowNeg20": _json_float(
                float(np.mean(fp_y <= -20.0)) if fp_y.size else float("nan")
            ),
            "meanDistanceToFloor": _json_float(
                float(np.mean(fp_y - NEGLIGIBLE_RISK)) if fp_y.size else float("nan")
            ),
            "meanRisk": _json_float(float(np.mean(risk[fp])) if np.any(fp) else float("nan")),
            "meanDilutionGap": _json_float(
                float(np.nanmean(gap[fp])) if np.any(fp) else float("nan")
            ),
            "note": (
                "False floor calls are later reports that did not sit at −30. "
                "Many sit near the floor rather than in the high-risk tail."
            ),
        },
        "falseNegatives": {
            "n": int(np.sum(fn)),
            "meanRisk": _json_float(float(np.mean(risk[fn])) if np.any(fn) else float("nan")),
        },
    }


def trend_predict(frame: pd.DataFrame) -> np.ndarray:
    risk = frame["risk"].to_numpy(dtype=float)
    if "risk_delta_last2" in frame.columns:
        delta = pd.to_numeric(frame["risk_delta_last2"], errors="coerce").to_numpy(dtype=float)
        delta = np.where(np.isfinite(delta), delta, 0.0)
    else:
        delta = np.zeros(len(frame))
    return np.clip(risk + delta, NEGLIGIBLE_RISK, 0.0)


def _keep(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    identity = [column for column in ("event_id", "y", "c_object_type") if column in frame.columns]
    present = [name for name in names if name in frame.columns]
    return frame[identity + present].copy()


def scientific_baselines(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, object]:
    y = test["y"].to_numpy(dtype=float)
    persist = persistence_predict(test)
    trend = trend_predict(test)
    floor = floor_mask(y)
    history_only = _keep(train, history_columns())
    cov_names = [
        name
        for name in snapshot_columns() + covariance_trend_columns()
        if name.startswith(("t_sigma", "c_sigma", "log_", "mahalanobis", "miss_over_sigma"))
        or name in covariance_trend_columns()
    ]
    families = {
        "persistence": persist,
        "trend": trend,
        "snapshotOnly": predict_model(fit_xgboost(_keep(train, snapshot_columns())), _keep(test, snapshot_columns())),
        "historyOnly": predict_model(fit_xgboost(history_only), _keep(test, history_columns()))
        if len(history_columns())
        else persist,
        "covarianceOnly": predict_model(
            fit_xgboost(_keep(train, cov_names)),
            _keep(test, cov_names),
        ),
    }
    rows = {name: _system_slice(y, pred, floor) for name, pred in families.items()}
    return {
        "note": (
            "Trend is clipped one-step extrapolation from the last two cutoff-safe "
            "risk reports. Family models are extra measurements, not shipped policies."
        ),
        "systems": rows,
    }


def simple_floor_baselines(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, object]:
    y_train = floor_mask(train["y"].to_numpy(dtype=float)).astype(int)
    y_test = floor_mask(test["y"].to_numpy(dtype=float)).astype(int)
    train_gap = attach_gap(train)
    test_gap = attach_gap(test)

    def _threshold_rule(series_train: np.ndarray, series_test: np.ndarray, greater: bool) -> dict[str, object]:
        grid = np.nanquantile(series_train, np.linspace(0.05, 0.95, 19))
        best_t = float(grid[0])
        best_f1 = -1.0
        for cut in grid:
            pred = series_train >= cut if greater else series_train <= cut
            actual = y_train.astype(bool)
            tp = int(np.sum(actual & pred))
            fp = int(np.sum(~actual & pred))
            fn = int(np.sum(actual & ~pred))
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(cut)
        test_pred = series_test >= best_t if greater else series_test <= best_t
        scores = binary_classification_metrics(y_test, test_pred.astype(float), threshold=0.5)
        scores["threshold"] = best_t
        scores["direction"] = "greater" if greater else "less"
        return scores

    specs = {
        "riskThreshold": _threshold_rule(
            train["risk"].to_numpy(dtype=float),
            test["risk"].to_numpy(dtype=float),
            greater=False,
        ),
        "gapThreshold": _threshold_rule(
            train_gap["dilution_gap"].to_numpy(dtype=float),
            test_gap["dilution_gap"].to_numpy(dtype=float),
            greater=True,
        ),
    }
    logit_sets = {
        "risk": ["risk"],
        "riskAndMaxRisk": ["risk", "max_risk_estimate"],
        "dilutionGap": ["dilution_gap"],
    }
    for name, cols in logit_sets.items():
        pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]
        )
        pipe.fit(train_gap[cols], y_train)
        proba = pipe.predict_proba(test_gap[cols])[:, 1]
        specs[f"logistic_{name}"] = binary_classification_metrics(y_test, proba, threshold=0.5)
        specs[f"logistic_{name}"]["features"] = cols
    return {
        "note": (
            "Simple floor detectors trained on the frozen train split only. "
            "If logistic on two fields approaches the XGBoost floor classifier, "
            "the finding is the data structure, not the booster."
        ),
        "baselines": specs,
    }


def attach_gap(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "dilution_gap" not in out.columns:
        out["dilution_gap"] = pd.to_numeric(out["max_risk_estimate"], errors="coerce") - pd.to_numeric(
            out["risk"], errors="coerce"
        )
    return out


def conformal_depth(
    y: np.ndarray,
    pred: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    floor: np.ndarray,
    risk: np.ndarray,
) -> dict[str, object]:
    overall = interval_report(y, lo, hi)
    overall["coverageCi95"] = _coverage_ci(y, lo, hi)
    high = risk >= HIGH_RISK_THRESHOLD
    mid = (~high) & (~floor)
    return {
        "overall": overall,
        "floor": interval_report(y[floor], lo[floor], hi[floor]) if np.any(floor) else {},
        "nonFloor": interval_report(y[~floor], lo[~floor], hi[~floor]) if np.any(~floor) else {},
        "snapshotHighRisk": interval_report(y[high], lo[high], hi[high]) if np.any(high) else {},
        "snapshotMid": interval_report(y[mid], lo[mid], hi[mid]) if np.any(mid) else {},
        "interpretation": (
            "Split conformal is statistically honest on average but operationally "
            "wide: most of the 90% width is paid on heterogeneous non-floor events."
        ),
    }


def _coverage_ci(
    y: np.ndarray, lo: np.ndarray, hi: np.ndarray, n_bootstrap: int = 400, seed: int = RANDOM_STATE
) -> dict[str, float]:
    covered = ((y >= lo) & (y <= hi)).astype(float)
    rng = np.random.default_rng(seed)
    n = int(covered.size)
    stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        stats[i] = float(np.mean(covered[idx]))
    low, high = np.quantile(stats, [0.025, 0.975])
    return {"low": float(low), "high": float(high), "nBootstrap": float(n_bootstrap)}


def selective_prediction_curve(
    y: np.ndarray,
    pred: np.ndarray,
    persist: np.ndarray,
    cal_y: np.ndarray,
    cal_pred: np.ndarray,
    current_risk: np.ndarray,
    miss_distance: np.ndarray,
    alphas: tuple[float, ...] = CONFORMAL_ALPHAS,
) -> dict[str, object]:
    scores = np.abs(np.asarray(cal_y, dtype=float) - np.asarray(cal_pred, dtype=float))
    rows = []
    for alpha in alphas:
        q_hat = split_conformal_quantile(scores, float(alpha))
        lo, hi = conformal_bounds(pred, q_hat)
        abstained, _ = conformal_abstain_mask(lo, hi, current_risk, miss_distance)
        selected = selective_metrics(y, pred, persist, abstained)
        widths = hi - lo
        rows.append(
            {
                "nominalCoverage": float(1.0 - alpha),
                "qHat": float(q_hat),
                "nAbstained": int(selected["nAbstained"]),
                "coverage": float(selected["coverage"]),
                "maeAccepted": float(selected["maeAccepted"]),
                "highRiskCapture": _json_float(float(selected["highRiskCapture"])),
                "falseReassurance": int(selected["falseReassurance"]),
                "meanWidth": float(np.mean(widths)),
            }
        )
    return {
        "rule": (
            "Abstain when the split-conformal band around the frozen floor-hurdle "
            "point crosses log10(Pc) ≥ −6, or when risk or miss distance is missing."
        ),
        "curve": rows,
    }


def h4_partial_effects(test: pd.DataFrame) -> dict[str, object]:
    frame = attach_gap(test)
    y_move = np.abs(frame["y"].to_numpy(dtype=float) - frame["risk"].to_numpy(dtype=float))
    names = ["log_combined_sigma_det", "risk", "n_messages", "miss_distance"]
    x = frame[names].apply(pd.to_numeric, errors="coerce")
    mask = np.isfinite(y_move) & np.isfinite(x.to_numpy()).all(axis=1)
    x = x.loc[mask]
    y = y_move[np.asarray(mask)]
    x_std = (x - x.mean()) / x.std(ddof=0).replace(0.0, 1.0)
    model = LinearRegression()
    model.fit(x_std, y)
    coef = {
        name: float(value) for name, value in zip(names, model.coef_, strict=True)
    }
    spearman_ci = {}
    for name in names + ["dilution_gap"]:
        spearman_ci[name] = _spearman_ci(
            pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float),
            y_move if name != "dilution_gap" else y_move,
        )
    return {
        "outcome": "|y - risk|",
        "standardizedCoefficients": coef,
        "intercept": float(model.intercept_),
        "n": int(mask.sum()),
        "spearmanCi": spearman_ci,
        "language": "association, not causation",
    }


def _spearman_ci(
    x: np.ndarray, y: np.ndarray, n_bootstrap: int = 400, seed: int = RANDOM_STATE
) -> dict[str, float]:
    from scipy.stats import spearmanr

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    rho, pvalue = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    n = int(x.size)
    stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        stats[i], _ = spearmanr(x[idx], y[idx])
    low, high = np.nanquantile(stats, [0.025, 0.975])
    return {
        "rho": float(rho) if np.isfinite(rho) else float("nan"),
        "pvalue": float(pvalue) if np.isfinite(pvalue) else float("nan"),
        "ci95Low": float(low),
        "ci95High": float(high),
    }


def base_rates(
    test: pd.DataFrame,
    horizons: list[dict[str, object]] | None = None,
    matched: dict[str, object] | None = None,
) -> dict[str, object]:
    y = test["y"].to_numpy(dtype=float)
    local = {
        "split": "seed42-event",
        "n": int(len(test)),
        "floorRate": float(np.mean(floor_mask(y))),
        "nHighRisk": int(np.sum(y >= HIGH_RISK_THRESHOLD)),
        "byClass": [
            {
                "t": float(cut),
                "nPlus": int(np.sum(y >= cut)),
            }
            for cut in CLASS_THRESHOLDS
        ],
    }
    horizon_rates = []
    for row in horizons or []:
        if "persistence" not in row:
            continue
        horizon_rates.append(
            {
                "cutoffHours": row.get("cutoffHours"),
                "nTest": row.get("testEvents"),
                "note": "eligible set is horizon-specific, not matched",
            }
        )
    return {
        "localTest": local,
        "unmatchedHorizons": horizon_rates,
        "matchedCohort": matched,
    }


def residual_shap_summary(residual_model, test: pd.DataFrame, max_rows: int = 80) -> dict[str, object]:
    try:
        import shap
    except ImportError:
        return {"skipped": True, "reason": "shap not installed"}
    names = list(residual_model.feature_names)
    x = test[names].apply(pd.to_numeric, errors="coerce")
    if len(x) > max_rows:
        x = x.head(max_rows)
    explainer = shap.TreeExplainer(residual_model.model)
    values = np.asarray(explainer.shap_values(x), dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    mean_abs = np.mean(np.abs(values), axis=0)
    order = np.argsort(mean_abs)[::-1][:10]
    top = [
        {
            "feature": names[int(i)],
            "meanAbsShap": float(mean_abs[int(i)]),
            "physicalHypothesis": _physical_tag(names[int(i)]),
        }
        for i in order
    ]
    return {
        "nRows": int(len(x)),
        "top": top,
        "note": (
            "SHAP explains the residual regressor, not a physical collision cause. "
            "Compare top features with H4 (covariance volume and max-risk gap)."
        ),
    }


def _physical_tag(name: str) -> str:
    if "cov" in name or "sigma" in name or "mahalanobis" in name:
        return "covariance / uncertainty volume"
    if "risk" in name:
        return "reported Pc trajectory"
    if "miss" in name or "relative" in name:
        return "geometry"
    if "obs" in name:
        return "tracking completeness"
    return "other"


def dataset_flow_counts(raw_n_rows: int, raw_n_events: int, n_eligible: int, manifest: dict) -> dict[str, int]:
    return {
        "cdmRows": int(raw_n_rows),
        "events": int(raw_n_events),
        "eligibleT48": int(n_eligible),
        "train": int(len(manifest["train"])),
        "validation": int(len(manifest["validation"])),
        "calibration": int(len(manifest["calibration"])),
        "test": int(len(manifest["test"])),
        "officialTest": 2167,
    }


def what_does_not_work_table(metrics: dict[str, object]) -> list[dict[str, object]]:
    systems = (metrics.get("honestMetrics") or {}).get("systems") or {}
    official = ((metrics.get("officialTest") or {}).get("esa") or {})
    rows = []
    mapping = {
        "persistence": "Persistence",
        "xgboost": "Unguarded XGB",
        "residual": "Residual XGB",
        "floorHurdle": "Floor hurdle",
        "ensemble": "Guarded ensemble",
    }
    for key, label in mapping.items():
        local = systems.get(key) or {}
        all_row = local.get("all") or {}
        non = local.get("nonFloor") or {}
        esa = official.get(key) or {}
        mae = all_row.get("mae")
        persist_mae = (systems.get("persistence") or {}).get("all", {}).get("mae")
        persist_non = (systems.get("persistence") or {}).get("nonFloor", {}).get("mae")
        rows.append(
            {
                "system": label,
                "improvesMeanMae": (
                    None if mae is None or persist_mae is None else bool(mae < persist_mae)
                ),
                "improvesNonFloorMae": (
                    None
                    if non.get("mae") is None or persist_non is None
                    else bool(non["mae"] < persist_non)
                ),
                "improvesOfficialL": (
                    None
                    if not esa
                    else bool(float(esa.get("esaLoss", 1e9)) < 0.694 - 1e-9)
                ),
            }
        )
    return rows


def score_review_armor(
    *,
    raw: pd.DataFrame,
    features: pd.DataFrame,
    mission_features: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    calibration: pd.DataFrame,
    manifest: dict,
    persist: np.ndarray,
    xgb_pred: np.ndarray,
    floor_pred: np.ndarray,
    floor_proba: np.ndarray,
    residual_model,
    cal_pred: np.ndarray,
    q90: float,
    metrics: dict[str, object],
) -> dict[str, object]:
    y = test["y"].to_numpy(dtype=float)
    risk = test["risk"].to_numpy(dtype=float)
    miss = test["miss_distance"].to_numpy(dtype=float)
    gap = attach_gap(test)["dilution_gap"].to_numpy(dtype=float)
    lo90, hi90 = conformal_bounds(floor_pred, q90)
    floor = floor_mask(y)
    matched = matched_cohort_horizons(raw, manifest["train"], manifest["test"])
    cal_y = calibration["y"].to_numpy(dtype=float)
    armor = {
        "reproducibility": reproducibility_spec(),
        "featureCatalog": feature_catalog(list(train.columns)),
        "censoring": censoring_sensitivity(y, persist, xgb_pred, floor_pred),
        "matchedCohortHorizons": matched,
        "missionGrouped": mission_grouped_split(mission_features),
        "floorClassifier": floor_classifier_evaluation(y, floor_proba, risk, gap),
        "scientificBaselines": scientific_baselines(train, test),
        "simpleFloorBaselines": simple_floor_baselines(train, test),
        "conformalDepth": conformal_depth(y, floor_pred, lo90, hi90, floor, risk),
        "selectivePrediction": selective_prediction_curve(
            y, floor_pred, persist, cal_y, cal_pred, risk, miss
        ),
        "h4Partial": h4_partial_effects(test),
        "baseRates": base_rates(test, metrics.get("horizons"), matched),
        "datasetFlow": dataset_flow_counts(
            int(len(raw)),
            int(raw["event_id"].nunique()),
            int(len(features)),
            manifest,
        ),
        "whatDoesNotWork": what_does_not_work_table(metrics),
        "residualShap": residual_shap_summary(residual_model, test),
        "predVsActual": {
            "y": [float(value) for value in y[:400]],
            "pred": [float(value) for value in floor_pred[:400]],
            "floor": [bool(value) for value in floor[:400]],
            "truncated": bool(len(y) > 400),
        },
    }
    scatter = attach_gap(test)
    armor["dilutionScatter"] = {
        "gap": [float(value) for value in scatter["dilution_gap"].fillna(0).to_numpy()[:800]],
        "absMove": [
            float(value)
            for value in np.abs(
                scatter["y"].to_numpy(dtype=float) - scatter["risk"].to_numpy(dtype=float)
            )[:800]
        ],
        "floor": [bool(value) for value in floor_mask(scatter["y"].to_numpy(dtype=float))[:800]],
    }
    return armor


def write_review_armor(metrics_path: Path, payload: dict[str, object]) -> None:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["reviewArmor"] = payload
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
