from __future__ import annotations

from constants import SNAPSHOT_COLUMNS, TREND_COLUMNS

HISTORY_STEMS = (
    "risk",
    "miss_distance",
    "relative_speed",
    "max_risk_estimate",
    "t_obs_used",
    "c_obs_used",
)

COVARIANCE_STEMS = (
    "t_sigma_r",
    "t_sigma_t",
    "t_sigma_n",
    "c_sigma_r",
    "c_sigma_t",
    "c_sigma_n",
    "log_t_position_covariance_det",
    "log_c_position_covariance_det",
)

TREND_SUFFIXES = (
    "_first",
    "_last",
    "_change",
    "_mean",
    "_std",
    "_min",
    "_max",
    "_slope",
    "_delta_last2",
    "_delta_last3",
    "_rising",
)

SNAPSHOT_DERIVED = (
    "derived_miss_distance",
    "derived_relative_speed",
    "miss_distance_residual",
    "normalized_separation",
    "t_obs_usage_ratio",
    "c_obs_usage_ratio",
    "log_t_cov_det",
    "log_c_cov_det",
    "hours_before_cutoff",
)

HISTORY_META = ("n_messages", "hours_since_prev")
IDENTITY = {"event_id", "y", "story", "c_object_type", "mission_id"}

FAMILIES = (
    "snapshot",
    "snapshot_history",
    "snapshot_history_covariance",
    "full",
)


def _stems_to_columns(stems: tuple[str, ...]) -> list[str]:
    return [f"{stem}{suffix}" for stem in stems for suffix in TREND_SUFFIXES]


def snapshot_columns() -> list[str]:
    base = [
        column
        for column in SNAPSHOT_COLUMNS
        if column not in {"event_id", "c_object_type", "mission_id"}
    ]
    return base + list(SNAPSHOT_DERIVED)


def history_columns() -> list[str]:
    return list(HISTORY_META) + _stems_to_columns(HISTORY_STEMS)


def covariance_trend_columns() -> list[str]:
    return _stems_to_columns(COVARIANCE_STEMS)


def columns_for_family(available: list[str], family: str) -> list[str]:
    present = [column for column in available if column not in IDENTITY]
    if family == "snapshot":
        wanted = set(snapshot_columns())
    elif family == "snapshot_history":
        wanted = set(snapshot_columns()) | set(history_columns())
    elif family == "snapshot_history_covariance":
        wanted = set(snapshot_columns()) | set(history_columns()) | set(covariance_trend_columns())
    elif family == "full":
        return present
    else:
        raise ValueError(f"unknown feature family {family!r}")
    return [column for column in present if column in wanted]


def assert_trend_stems_are_known() -> None:
    unknown = [column for column in TREND_COLUMNS if column not in HISTORY_STEMS + COVARIANCE_STEMS]
    if unknown:
        raise ValueError(f"trend columns missing from ablation stems: {unknown}")
