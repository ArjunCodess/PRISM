from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = [
    "event_id",
    "mission_id",
    "time_to_tca",
    "risk",
    "miss_distance",
    "relative_speed",
    "relative_position_r",
    "relative_position_t",
    "relative_position_n",
    "relative_velocity_r",
    "relative_velocity_t",
    "relative_velocity_n",
]


def validate_cdm_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if frame.empty:
        raise ValueError("cdm frame is empty")
    if frame["event_id"].isna().any():
        raise ValueError("event_id contains nulls")
    # ESA includes a small number of final CDMs issued shortly after nominal TCA.
    # They are valid targets, but can never enter the pre-T−48 feature history.
    if (frame["time_to_tca"] < -1.0).any():
        raise ValueError("time_to_tca cannot be more than one day after TCA")
    duplicates = frame.duplicated(subset=["event_id", "time_to_tca"]).sum()
    if duplicates:
        raise ValueError(f"duplicate event/time rows: {duplicates}")
    return frame.sort_values(["event_id", "time_to_tca"], ascending=[True, False]).reset_index(
        drop=True
    )
