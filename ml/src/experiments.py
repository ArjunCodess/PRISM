from __future__ import annotations

import numpy as np
import pandas as pd
from abstention import coverage_curve, selective_metrics
from build_events import build_event_histories
from constants import CLASS_THRESHOLDS, HIGH_RISK_THRESHOLD, HORIZON_HOURS, NEGLIGIBLE_RISK, RANDOM_STATE
from evaluate import (
    clip_for_esa,
    esa_loss,
    false_reassurance_analogue,
    floor_mask,
    level_scoreboard_row,
    regression_metrics,
)
from explain import feature_group, shap_explainer
from feature_sets import FAMILIES, columns_for_family
from features import build_feature_table
from split import REDRAW_SEEDS, grouped_splits, subset
from train_regressor import (
    fit_residual_xgboost,
    fit_xgboost,
    persistence_predict,
    predict_model,
    predict_reconstructed,
)


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
        "historyBlock": (
            "The history block consists of temporal transforms of variables already "
            "available in the latest snapshot, plus message count and recency."
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
            "disagreement exceeds 1.25 log-risk units. The −6 class follows "
            "the ESA challenge definition. The persistence guard and 1.25 "
            "disagreement threshold were fixed design choices before "
            "evaluating the test split."
        ),
        "falseReassuranceDefinition": (
            "An accepted forecast (no abstention) with predicted log10(Pc) < −6 "
            "while the final reported value is ≥ −6."
        ),
        "operatingPoint": operating,
        "coverageCurve": curve,
    }


def attach_dilution_gap(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "dilution_gap" not in out.columns:
        out["dilution_gap"] = pd.to_numeric(
            out["max_risk_estimate"], errors="coerce"
        ) - pd.to_numeric(out["risk"], errors="coerce")
    return out


def _spearman(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    from scipy.stats import spearmanr

    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    mask = np.isfinite(xx) & np.isfinite(yy)
    n = int(mask.sum())
    if n < 10:
        return {"n": n, "rho": float("nan"), "pvalue": float("nan")}
    rho, pvalue = spearmanr(xx[mask], yy[mask])
    return {"n": n, "rho": float(rho), "pvalue": float(pvalue)}


def _quartile_slices(
    gap: np.ndarray,
    abs_move: np.ndarray,
    floor: np.ndarray,
    persist_err: np.ndarray,
    model_err: np.ndarray | None,
    edges: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i in range(len(edges) - 1):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i == len(edges) - 2:
            mask = (gap >= lo) & (gap <= hi)
        else:
            mask = (gap >= lo) & (gap < hi)
        mask = mask & np.isfinite(gap)
        payload: dict[str, object] = {
            "quartile": i + 1,
            "gapLow": lo,
            "gapHigh": hi,
            "n": int(mask.sum()),
            "floorRate": float(np.mean(floor[mask])) if mask.any() else float("nan"),
            "meanAbsMove": float(np.mean(abs_move[mask])) if mask.any() else float("nan"),
            "persistMae": float(np.mean(persist_err[mask])) if mask.any() else float("nan"),
        }
        if model_err is not None:
            payload["xgboostMae"] = (
                float(np.mean(model_err[mask])) if mask.any() else float("nan")
            )
        rows.append(payload)
    return rows


def dilution_probe(
    train: pd.DataFrame,
    test: pd.DataFrame,
    persist_test: np.ndarray | None = None,
    xgb_test: np.ndarray | None = None,
) -> dict[str, object]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline

    train = attach_dilution_gap(train)
    test = attach_dilution_gap(test)
    y_test = test["y"].to_numpy(dtype=float)
    risk_test = test["risk"].to_numpy(dtype=float)
    abs_move = np.abs(y_test - risk_test)
    floor = floor_mask(y_test).astype(float)
    persist_err = (
        np.abs(y_test - np.asarray(persist_test, dtype=float))
        if persist_test is not None
        else abs_move
    )
    model_err = (
        np.abs(y_test - np.asarray(xgb_test, dtype=float)) if xgb_test is not None else None
    )

    candidates = {
        "dilution_gap": test["dilution_gap"],
        "log_t_cov_det": test.get("log_t_cov_det"),
        "log_c_cov_det": test.get("log_c_cov_det"),
        "log_combined_sigma_det": test.get("log_combined_sigma_det"),
        "t_sigma_r": test.get("t_sigma_r"),
        "t_obs_used_slope": test.get("t_obs_used_slope"),
        "n_messages": test.get("n_messages"),
        "F10": test.get("F10"),
    }
    spearman_abs: dict[str, object] = {}
    spearman_floor: dict[str, object] = {}
    for name, series in candidates.items():
        if series is None:
            continue
        values = np.asarray(series, dtype=float)
        spearman_abs[name] = _spearman(values, abs_move)
        spearman_floor[name] = _spearman(values, floor)

    logit_cols = ["dilution_gap", "miss_distance", "n_messages"]
    y_floor_train = floor_mask(train["y"].to_numpy(dtype=float)).astype(int)
    y_floor_test = floor_mask(y_test).astype(int)
    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            ),
        ]
    )
    pipe.fit(train[logit_cols], y_floor_train)
    proba = pipe.predict_proba(test[logit_cols])[:, 1]
    auc = (
        float(roc_auc_score(y_floor_test, proba))
        if len(np.unique(y_floor_test)) > 1
        else float("nan")
    )
    coef = pipe.named_steps["model"].coef_[0]
    intercept = float(pipe.named_steps["model"].intercept_[0])

    train_gap = train["dilution_gap"].to_numpy(dtype=float)
    edges = np.unique(np.nanquantile(train_gap, [0.0, 0.25, 0.5, 0.75, 1.0]))
    if len(edges) < 3:
        edges = np.array([float(np.nanmin(train_gap)), float(np.nanmax(train_gap))])
    quartiles = _quartile_slices(
        test["dilution_gap"].to_numpy(dtype=float),
        abs_move,
        floor,
        persist_err,
        model_err,
        edges,
    )
    gap_abs = spearman_abs.get("dilution_gap") or {}
    cov_abs = spearman_abs.get("log_combined_sigma_det") or {}
    if not cov_abs:
        cov_abs = spearman_abs.get("log_t_cov_det") or {}
    f10_abs = spearman_abs.get("F10") or {}

    def _sig_positive(row: dict[str, object]) -> bool:
        rho = row.get("rho")
        pvalue = row.get("pvalue")
        if not isinstance(rho, (int, float)) or not isinstance(pvalue, (int, float)):
            return False
        return bool(np.isfinite(rho) and np.isfinite(pvalue) and rho > 0.05 and pvalue < 0.05)

    def _sig(row: dict[str, object]) -> bool:
        rho = row.get("rho")
        pvalue = row.get("pvalue")
        if not isinstance(rho, (int, float)) or not isinstance(pvalue, (int, float)):
            return False
        return bool(
            np.isfinite(rho) and np.isfinite(pvalue) and abs(float(rho)) >= 0.05 and pvalue < 0.05
        )

    gap_rho = gap_abs.get("rho")
    return {
        "hypothesis": (
            "Events whose pessimistic max-risk sits far above the current report, "
            "or whose covariance is still large, move more before TCA. "
            "F10 is a negative control. This is not a covariance-tasking model."
        ),
        "dilutionGap": "max_risk_estimate - risk on the cutoff-safe snapshot",
        "fitOn": (
            "logistic coefficients from frozen train event ids only; "
            "Spearman and quartiles reported on test"
        ),
        "replacesExhibit": False,
        "spearmanAbsMove": spearman_abs,
        "spearmanFloor": spearman_floor,
        "logisticFloor": {
            "features": logit_cols,
            "coefficients": {
                name: float(value) for name, value in zip(logit_cols, coef, strict=True)
            },
            "intercept": intercept,
            "testAuc": auc,
            "nTrain": int(len(train)),
            "nTest": int(len(test)),
            "trainFloorRate": float(y_floor_train.mean()),
            "testFloorRate": float(y_floor_test.mean()),
        },
        "quartiles": quartiles,
        "h4Supported": _sig_positive(cov_abs),
        "h4GapPredictsMoreMovement": _sig_positive(gap_abs),
        "h4GapAssociationExists": _sig(gap_abs),
        "h4GapSign": (
            "negative"
            if isinstance(gap_rho, (int, float)) and float(gap_rho) < 0
            else "positive"
        ),
        "negativeControl": {
            "name": "F10",
            "spearmanAbsMove": f10_abs,
        },
        "covarianceSpearmanAbsMove": cov_abs,
    }


def _mean_sd(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(arr)), "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0}


def _split_scoreboard(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, object]:
    y = test["y"].to_numpy(dtype=float)
    persist = persistence_predict(test)
    xgb_pred = predict_model(fit_xgboost(train), test)
    residual_pred = predict_reconstructed(fit_residual_xgboost(train), test)
    persist_row = level_scoreboard_row(y, persist)
    xgb_row = level_scoreboard_row(y, xgb_pred)
    residual_row = level_scoreboard_row(y, residual_pred)
    persist_mae = float(persist_row["mae"])
    return {
        "nTest": int(len(test)),
        "nFloor": int(floor_mask(y).sum()),
        "nHighRisk": int((y >= HIGH_RISK_THRESHOLD).sum()),
        "persistence": persist_row,
        "xgboost": {
            **xgb_row,
            "maeAdvantage": persist_mae - float(xgb_row["mae"]),
            "residualMae": float(xgb_row["mae"]),
        },
        "residual": {
            **residual_row,
            "maeAdvantage": persist_mae - float(residual_row["mae"]),
            "residualMae": float(residual_row["mae"]),
        },
    }


def repeated_grouped_splits(
    features: pd.DataFrame, seeds: tuple[int, ...] = REDRAW_SEEDS
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        splits = grouped_splits(features, seed=seed)
        train = subset(features, splits.train_ids)
        test = subset(features, splits.test_ids)
        payload = _split_scoreboard(train, test)
        payload["seed"] = int(seed)
        payload["reportedSplit"] = bool(seed == RANDOM_STATE)
        rows.append(payload)
    summary = {}
    for system in ("xgboost", "residual"):
        summary[system] = {
            "maeAdvantage": _mean_sd([float(row[system]["maeAdvantage"]) for row in rows]),
            "floorExcludedMae": _mean_sd(
                [float(row[system]["floorExcludedMae"]) for row in rows]
            ),
            "residualMae": _mean_sd([float(row[system]["residualMae"]) for row in rows]),
            "mae": _mean_sd([float(row[system]["mae"]) for row in rows]),
        }
    summary["persistence"] = {
        "mae": _mean_sd([float(row["persistence"]["mae"]) for row in rows]),
        "floorExcludedMae": _mean_sd(
            [float(row["persistence"]["floorExcludedMae"]) for row in rows]
        ),
    }
    return {
        "seeds": list(seeds),
        "reportedSeed": RANDOM_STATE,
        "replacesExhibit": False,
        "note": (
            "Each redraw retrains unguarded XGBoost and residual XGBoost on that "
            "seed's train ids only. Seed 42 remains the reported local split."
        ),
        "splits": rows,
        "summary": summary,
    }


def leave_one_high_risk_out(features: pd.DataFrame) -> dict[str, object]:
    high_ids = [
        int(event_id)
        for event_id in features.loc[
            features["y"].to_numpy(dtype=float) >= HIGH_RISK_THRESHOLD, "event_id"
        ]
    ]
    rows: list[dict[str, object]] = []
    persist_closer = 0
    residual_closer = 0
    ties = 0
    for event_id in high_ids:
        train = features.loc[features["event_id"] != event_id]
        test = features.loc[features["event_id"] == event_id]
        if train.empty or test.empty:
            continue
        persist = float(persistence_predict(test)[0])
        pred = float(predict_reconstructed(fit_residual_xgboost(train), test)[0])
        y = float(test["y"].iloc[0])
        persist_err = abs(y - persist)
        residual_err = abs(y - pred)
        if residual_err < persist_err - 1e-12:
            winner = "residual"
            residual_closer += 1
        elif persist_err < residual_err - 1e-12:
            winner = "persistence"
            persist_closer += 1
        else:
            winner = "tie"
            ties += 1
        rows.append(
            {
                "eventId": event_id,
                "y": y,
                "risk": persist,
                "residualPred": pred,
                "persistAbsError": persist_err,
                "residualAbsError": residual_err,
                "closer": winner,
            }
        )
    n = len(rows)
    return {
        "nHighRisk": n,
        "replacesExhibit": False,
        "fitOn": "all eligible events except the held-out high-risk event",
        "persistCloser": persist_closer,
        "residualCloser": residual_closer,
        "ties": ties,
        "residualCloserShare": float(residual_closer / n) if n else float("nan"),
        "meanPersistAbsError": float(np.mean([row["persistAbsError"] for row in rows]))
        if rows
        else float("nan"),
        "meanResidualAbsError": float(np.mean([row["residualAbsError"] for row in rows]))
        if rows
        else float("nan"),
        "events": rows,
    }


def threshold_sweep(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    abstained: np.ndarray | None = None,
    thresholds: tuple[float, ...] = CLASS_THRESHOLDS,
) -> dict[str, object]:
    y_true = np.asarray(y_true, dtype=float)
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        systems: dict[str, object] = {}
        n_positives = int((y_true >= threshold).sum())
        for name, pred in predictions.items():
            pred_arr = np.asarray(pred, dtype=float)
            esa = esa_loss(y_true, clip_for_esa(pred_arr, threshold), threshold)
            analogue = false_reassurance_analogue(
                y_true, pred_arr, threshold, abstained=abstained
            )
            systems[name] = {
                "esaLoss": esa["esa_loss"],
                "mseHr": esa["mse_hr"],
                "f2": esa["f2"],
                "falseReassuranceAnalogue": analogue["falseReassuranceAnalogue"],
                "missedClass": analogue["missedClass"],
            }
        rows.append(
            {
                "threshold": threshold,
                "nPositives": n_positives,
                "systems": systems,
            }
        )
    return {
        "split": "frozen local test",
        "replacesExhibit": False,
        "retuned": False,
        "note": (
            "Same frozen predictions; only the class definition changes. "
            "Operational LEO reaction is nearer log10(Pc) −4 to −5. "
            "ESA scored −6 to have enough positives. "
            "False-reassurance analogue is an accepted forecast (existing −6 "
            "abstention mask, if supplied) with pred < t while y ≥ t."
        ),
        "thresholds": list(thresholds),
        "rows": rows,
    }
