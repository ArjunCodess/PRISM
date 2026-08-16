from __future__ import annotations

import numpy as np
import pandas as pd
from constants import CUTOFF_DAYS, SNAPSHOT_COLUMNS, TREND_COLUMNS


def _slope(values: np.ndarray, times: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    t = times.astype(float)
    z = values.astype(float)
    if np.allclose(t, t[0]):
        return 0.0
    design = np.column_stack([np.ones(len(t)), t])
    beta, *_ = np.linalg.lstsq(design, z, rcond=None)
    # time_to_tca decreases toward TCA; negate so positive means rising toward TCA
    return float(-beta[1])


def _safe_div(num: float, den: float) -> float:
    if abs(den) < 1e-12:
        return 0.0
    return num / den


def event_features(
    event: dict[str, object], include_mission: bool = False
) -> dict[str, float | str]:
    history: pd.DataFrame = event["history"].copy()  # type: ignore[union-attr]
    snapshot: pd.Series = event["snapshot"]  # type: ignore[assignment]
    history["log_t_position_covariance_det"] = np.log(
        history["t_position_covariance_det"].clip(lower=1e-12)
    )
    history["log_c_position_covariance_det"] = np.log(
        history["c_position_covariance_det"].clip(lower=1e-12)
    )
    cutoff_days = float(event.get("cutoff_days", CUTOFF_DAYS))
    features: dict[str, float | str] = {
        "event_id": snapshot["event_id"],
        "y": float(event["y"]),
        "n_messages": float(len(history)),
        "hours_before_cutoff": float((snapshot["time_to_tca"] - cutoff_days) * 24.0),
    }
    if include_mission:
        features["mission_id"] = float(snapshot["mission_id"])

    for column in SNAPSHOT_COLUMNS:
        if column in {"event_id", "c_object_type", "mission_id"}:
            continue
        if column in snapshot:
            value = snapshot[column]
            features[column] = float(value) if pd.notna(value) else np.nan

    features["c_object_type"] = str(snapshot.get("c_object_type", "UNKNOWN"))

    pos = np.array(
        [
            snapshot["relative_position_r"],
            snapshot["relative_position_t"],
            snapshot["relative_position_n"],
        ],
        dtype=float,
    )
    vel = np.array(
        [
            snapshot["relative_velocity_r"],
            snapshot["relative_velocity_t"],
            snapshot["relative_velocity_n"],
        ],
        dtype=float,
    )
    derived_miss = float(np.linalg.norm(pos))
    derived_speed = float(np.linalg.norm(vel))
    features["derived_miss_distance"] = derived_miss
    features["derived_relative_speed"] = derived_speed
    features["miss_distance_residual"] = float(snapshot["miss_distance"] - derived_miss)
    combined_sigma = float(
        np.sqrt(snapshot["t_sigma_r"] ** 2 + snapshot["c_sigma_r"] ** 2)
        + np.sqrt(snapshot["t_sigma_t"] ** 2 + snapshot["c_sigma_t"] ** 2)
        + np.sqrt(snapshot["t_sigma_n"] ** 2 + snapshot["c_sigma_n"] ** 2)
    )
    features["normalized_separation"] = _safe_div(
        float(snapshot["miss_distance"]), combined_sigma + 1e-6
    )
    features["t_obs_usage_ratio"] = _safe_div(
        float(snapshot["t_obs_used"]), float(snapshot["t_obs_available"])
    )
    features["c_obs_usage_ratio"] = _safe_div(
        float(snapshot["c_obs_used"]), float(snapshot["c_obs_available"])
    )
    features["log_t_cov_det"] = float(np.log(snapshot["t_position_covariance_det"] + 1e-12))
    features["log_c_cov_det"] = float(np.log(snapshot["c_position_covariance_det"] + 1e-12))

    for column in TREND_COLUMNS:
        series = history[column].astype(float).to_numpy()
        times = history["time_to_tca"].astype(float).to_numpy()
        features[f"{column}_first"] = float(series[0])
        features[f"{column}_last"] = float(series[-1])
        features[f"{column}_change"] = float(series[-1] - series[0])
        features[f"{column}_mean"] = float(np.mean(series))
        features[f"{column}_std"] = float(np.std(series))
        features[f"{column}_min"] = float(np.min(series))
        features[f"{column}_max"] = float(np.max(series))
        features[f"{column}_slope"] = _slope(series, times)
        if len(series) >= 2:
            features[f"{column}_delta_last2"] = float(series[-1] - series[-2])
        else:
            features[f"{column}_delta_last2"] = 0.0
        if len(series) >= 3:
            features[f"{column}_delta_last3"] = float(series[-1] - series[-3])
        else:
            features[f"{column}_delta_last3"] = features[f"{column}_delta_last2"]
        features[f"{column}_rising"] = float(series[-1] > series[0])

    features["hours_since_prev"] = (
        float((history["time_to_tca"].iloc[-2] - history["time_to_tca"].iloc[-1]) * 24.0)
        if len(history) >= 2
        else 0.0
    )
    return features


def build_feature_table(
    events: list[dict[str, object]], include_mission: bool = False
) -> pd.DataFrame:
    rows = [event_features(event, include_mission=include_mission) for event in events]
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
