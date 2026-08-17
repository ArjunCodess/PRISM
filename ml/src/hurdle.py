from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from constants import (
    FLOOR_MARGIN,
    HIGH_RISK_THRESHOLD,
    LOW_RISK_CLIP,
    NEGLIGIBLE_RISK,
    RANDOM_STATE,
)
from evaluate import regression_metrics
from train_classifier import (
    TrainedClassifier,
    classifier_proba,
    fit_collapse_classifier,
    fit_warning_classifier,
    tune_f2_threshold,
)
from train_regressor import (
    TrainedRegressor,
    fit_xgboost,
    model_matrix,
    numeric_columns,
)
from xgboost import XGBRegressor

RESIDUAL_GRID: tuple[dict[str, object], ...] = (
    {
        "max_depth": 4,
        "min_child_weight": 6,
        "learning_rate": 0.05,
        "n_estimators": 300,
    },
    {
        "max_depth": 5,
        "min_child_weight": 8,
        "learning_rate": 0.04,
        "n_estimators": 400,
    },
    {
        "max_depth": 6,
        "min_child_weight": 10,
        "learning_rate": 0.03,
        "n_estimators": 500,
    },
)


@dataclass
class ConformalBands:
    floor_lo: float
    floor_hi: float
    move_lo: float
    move_hi: float
    floor_lo50: float
    floor_hi50: float
    move_lo50: float
    move_hi50: float


@dataclass
class HurdleOutput:
    point: np.ndarray
    residual: np.ndarray
    p_floor: np.ndarray
    warning_proba: np.ndarray
    ensemble: np.ndarray
    interval50: np.ndarray
    interval90: np.ndarray
    is_floor: np.ndarray


@dataclass
class HurdlePolicy:
    residual: TrainedRegressor
    collapse: TrainedClassifier
    warning: TrainedClassifier
    feature_names: list[str]
    mix_kind: str
    floor_threshold: float
    f2_threshold: float
    promote_high_risk: bool
    conformal: ConformalBands
    ensemble: list[XGBRegressor]


def is_floor_label(y: np.ndarray) -> np.ndarray:
    return np.asarray(y, dtype=float) <= NEGLIGIBLE_RISK + FLOOR_MARGIN


def apply_persistence_guard(pred: np.ndarray, current_risk: np.ndarray) -> np.ndarray:
    current = np.asarray(current_risk, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if pred.ndim == 2 and current.ndim == 1:
        current = current[:, None]
    guarded = np.where(current >= HIGH_RISK_THRESHOLD, current, pred)
    invented = (current < HIGH_RISK_THRESHOLD) & (guarded >= HIGH_RISK_THRESHOLD)
    return np.where(invented, LOW_RISK_CLIP, guarded)


def mix_residual(
    current_risk: np.ndarray,
    delta: np.ndarray,
    p_floor: np.ndarray,
    *,
    mix_kind: str,
    floor_threshold: float,
    warning_proba: np.ndarray | None = None,
) -> np.ndarray:
    current_risk = np.asarray(current_risk, dtype=float)
    delta = np.asarray(delta, dtype=float)
    p_floor = np.asarray(p_floor, dtype=float)
    moving = current_risk + delta
    _ = warning_proba
    if mix_kind == "hard":
        mixed = np.where(p_floor >= floor_threshold, NEGLIGIBLE_RISK, moving)
    else:
        mixed = p_floor * NEGLIGIBLE_RISK + (1.0 - p_floor) * moving
    return np.clip(mixed, NEGLIGIBLE_RISK, 0.0)


def _non_floor_frame(frame: pd.DataFrame) -> pd.DataFrame:
    mask = ~is_floor_label(frame["y"].to_numpy(dtype=float))
    selected = frame.loc[mask].copy()
    if len(selected) < 80:
        return frame.copy()
    return selected


def _evaluate_mix(
    frame: pd.DataFrame,
    residual: TrainedRegressor,
    collapse: TrainedClassifier,
    mix_kind: str,
    floor_threshold: float,
) -> float:
    delta = np.asarray(residual.model.predict(model_matrix(frame, residual.feature_names)))
    p_floor = classifier_proba(collapse, frame)
    mixed = mix_residual(
        frame["risk"].to_numpy(dtype=float),
        delta,
        p_floor,
        mix_kind=mix_kind,
        floor_threshold=floor_threshold,
    )
    mixed = apply_persistence_guard(mixed, frame["risk"].to_numpy(dtype=float))
    return float(np.mean(np.abs(frame["y"].to_numpy(dtype=float) - mixed)))


def _choose_residual(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    collapse: TrainedClassifier,
) -> tuple[TrainedRegressor, str, float, dict[str, object]]:
    train_move = _non_floor_frame(train)
    val_move = _non_floor_frame(validation)
    mix_grid: list[tuple[str, float]] = [("soft", 1.0)]
    mix_grid.extend(("hard", float(threshold)) for threshold in (0.35, 0.45, 0.55, 0.65, 0.75))
    best: tuple[float, TrainedRegressor, str, float, dict[str, object]] | None = None
    for params in RESIDUAL_GRID:
        residual = fit_xgboost(
            train_move,
            eval_frame=val_move if len(val_move) >= 20 else None,
            residual=True,
            params=params,
        )
        for mix_kind, threshold in mix_grid:
            mae = _evaluate_mix(validation, residual, collapse, mix_kind, threshold)
            candidate = (mae, residual, mix_kind, threshold, params)
            if best is None or mae < best[0]:
                best = candidate
    assert best is not None
    return best[1], best[2], best[3], best[4]


def _bootstrap_residuals(
    train: pd.DataFrame,
    feature_names: list[str],
    params: dict[str, object],
    n_models: int = 10,
) -> list[XGBRegressor]:
    rng = np.random.default_rng(RANDOM_STATE)
    models: list[XGBRegressor] = []
    move = _non_floor_frame(train)
    for _ in range(n_models):
        idx = rng.integers(0, len(move), size=len(move))
        sample = move.iloc[idx]
        models.append(
            fit_xgboost(sample, residual=True, params=params).model
        )
        _ = feature_names
    return models


def _signed_quantiles(residuals: np.ndarray, low: float, high: float) -> tuple[float, float]:
    if len(residuals) == 0:
        return -1.0, 1.0
    n = len(residuals)
    lo_q = min(max((low * (n + 1)) / n, 0.0), 1.0)
    hi_q = min(max((high * (n + 1)) / n, 0.0), 1.0)
    return float(np.quantile(residuals, lo_q)), float(np.quantile(residuals, hi_q))


def fit_conformal(
    y_true: np.ndarray,
    point: np.ndarray,
    is_floor: np.ndarray,
) -> ConformalBands:
    y_true = np.asarray(y_true, dtype=float)
    point = np.asarray(point, dtype=float)
    is_floor = np.asarray(is_floor, dtype=bool)
    residual = y_true - point
    floor_lo, floor_hi = _signed_quantiles(residual[is_floor], 0.05, 0.95)
    move_lo, move_hi = _signed_quantiles(residual[~is_floor], 0.05, 0.95)
    floor_lo50, floor_hi50 = _signed_quantiles(residual[is_floor], 0.25, 0.75)
    move_lo50, move_hi50 = _signed_quantiles(residual[~is_floor], 0.25, 0.75)
    return ConformalBands(
        floor_lo=floor_lo,
        floor_hi=floor_hi,
        move_lo=move_lo,
        move_hi=move_hi,
        floor_lo50=floor_lo50,
        floor_hi50=floor_hi50,
        move_lo50=move_lo50,
        move_hi50=move_hi50,
    )


def _interval(point: np.ndarray, is_floor: np.ndarray, lo_f: float, hi_f: float, lo_m: float, hi_m: float) -> np.ndarray:
    lo = np.where(is_floor, point + lo_f, point + lo_m)
    hi = np.where(is_floor, point + hi_f, point + hi_m)
    return np.vstack([np.minimum(lo, hi), np.maximum(lo, hi)])


def fit_hurdle_policy(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    calibration: pd.DataFrame | None = None,
    n_bootstrap: int = 10,
    search: bool = True,
) -> HurdlePolicy:
    collapse = fit_collapse_classifier(train)
    warning = fit_warning_classifier(train)
    if search:
        residual, mix_kind, floor_threshold, params = _choose_residual(
            train, validation, collapse
        )
    else:
        same_ids = set(validation["event_id"]) == set(train["event_id"])
        eval_frame = None if same_ids else _non_floor_frame(validation)
        residual = fit_xgboost(
            _non_floor_frame(train),
            eval_frame=eval_frame if eval_frame is not None and len(eval_frame) >= 20 else None,
            residual=True,
        )
        mix_kind = "soft"
        floor_threshold = 1.0
        params = {"n_estimators": 400, "learning_rate": 0.04, "max_depth": 5}
    feature_names = residual.feature_names
    bootstrap_train = pd.concat([train, validation], ignore_index=True)
    ensemble = _bootstrap_residuals(bootstrap_train, feature_names, params, n_models=n_bootstrap)
    val_out = _predict_ungarded(residual, collapse, warning, validation, mix_kind, floor_threshold, ensemble)
    f2_threshold = tune_f2_threshold(validation["y"].to_numpy(dtype=float), val_out.warning_proba)
    promote = False
    cal_frame = calibration if calibration is not None else validation
    cal_out = _predict_ungarded(residual, collapse, warning, cal_frame, mix_kind, floor_threshold, ensemble)
    cal_point = apply_persistence_guard(cal_out.point, cal_frame["risk"].to_numpy(dtype=float))
    if promote:
        cal_point = promote_if_needed(cal_point, cal_out.warning_proba, f2_threshold)
    conformal = fit_conformal(cal_frame["y"].to_numpy(dtype=float), cal_point, cal_out.is_floor)
    return HurdlePolicy(
        residual=residual,
        collapse=collapse,
        warning=warning,
        feature_names=feature_names,
        mix_kind=mix_kind,
        floor_threshold=floor_threshold,
        f2_threshold=f2_threshold,
        promote_high_risk=promote,
        conformal=conformal,
        ensemble=ensemble,
    )


def promote_if_needed(point: np.ndarray, warning_proba: np.ndarray, threshold: float) -> np.ndarray:
    promoted = np.asarray(point, dtype=float).copy()
    mask = (np.asarray(warning_proba) >= threshold) & (promoted < HIGH_RISK_THRESHOLD)
    promoted[mask] = HIGH_RISK_THRESHOLD
    return promoted


def _predict_ungarded(
    residual: TrainedRegressor,
    collapse: TrainedClassifier,
    warning: TrainedClassifier,
    frame: pd.DataFrame,
    mix_kind: str,
    floor_threshold: float,
    ensemble: list[XGBRegressor],
) -> HurdleOutput:
    x = model_matrix(frame, residual.feature_names)
    delta = np.asarray(residual.model.predict(x), dtype=float)
    p_floor = classifier_proba(collapse, frame)
    warning_proba = classifier_proba(warning, frame)
    current = frame["risk"].to_numpy(dtype=float)
    point = mix_residual(
        current,
        delta,
        p_floor,
        mix_kind=mix_kind,
        floor_threshold=floor_threshold,
        warning_proba=warning_proba,
    )
    members = []
    for model in ensemble:
        member_delta = np.asarray(model.predict(x), dtype=float)
        members.append(
            mix_residual(
                current,
                member_delta,
                p_floor,
                mix_kind=mix_kind,
                floor_threshold=floor_threshold,
                warning_proba=warning_proba,
            )
        )
    ens = np.column_stack(members) if members else point.reshape(-1, 1)
    dummy = ConformalBands(-0.5, 0.5, -0.5, 0.5, -0.2, 0.2, -0.2, 0.2)
    is_floor = p_floor >= (floor_threshold if mix_kind == "hard" else 0.5)
    interval90 = _interval(point, is_floor, dummy.floor_lo, dummy.floor_hi, dummy.move_lo, dummy.move_hi)
    interval50 = _interval(point, is_floor, dummy.floor_lo50, dummy.floor_hi50, dummy.move_lo50, dummy.move_hi50)
    return HurdleOutput(
        point=point,
        residual=delta,
        p_floor=p_floor,
        warning_proba=warning_proba,
        ensemble=ens,
        interval50=interval50,
        interval90=interval90,
        is_floor=is_floor,
    )


def predict_hurdle(policy: HurdlePolicy, frame: pd.DataFrame) -> HurdleOutput:
    raw = _predict_ungarded(
        policy.residual,
        policy.collapse,
        policy.warning,
        frame,
        policy.mix_kind,
        policy.floor_threshold,
        policy.ensemble,
    )
    current = frame["risk"].to_numpy(dtype=float)
    point = apply_persistence_guard(raw.point, current)
    if policy.promote_high_risk:
        point = promote_if_needed(point, raw.warning_proba, policy.f2_threshold)
    ens = apply_persistence_guard(raw.ensemble, current)
    is_floor = raw.is_floor
    interval90 = _interval(
        point,
        is_floor,
        policy.conformal.floor_lo,
        policy.conformal.floor_hi,
        policy.conformal.move_lo,
        policy.conformal.move_hi,
    )
    interval50 = _interval(
        point,
        is_floor,
        policy.conformal.floor_lo50,
        policy.conformal.floor_hi50,
        policy.conformal.move_lo50,
        policy.conformal.move_hi50,
    )
    return HurdleOutput(
        point=point,
        residual=raw.residual,
        p_floor=raw.p_floor,
        warning_proba=raw.warning_proba,
        ensemble=ens,
        interval50=interval50,
        interval90=interval90,
        is_floor=is_floor,
    )


def hurdle_metrics(policy: HurdlePolicy, frame: pd.DataFrame) -> dict[str, float]:
    out = predict_hurdle(policy, frame)
    return regression_metrics(frame["y"].to_numpy(dtype=float), out.point)


def residual_feature_names(train: pd.DataFrame) -> list[str]:
    return numeric_columns(train)
