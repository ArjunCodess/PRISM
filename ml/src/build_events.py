from __future__ import annotations

import pandas as pd
from constants import CUTOFF_DAYS


def build_event_histories(
    frame: pd.DataFrame, require_later_target: bool = True
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for event_id, group in frame.groupby("event_id", sort=True):
        ordered = group.sort_values("time_to_tca", ascending=False).reset_index(drop=True)
        eligible = ordered[ordered["time_to_tca"] >= CUTOFF_DAYS]
        if eligible.empty:
            continue
        target_row = ordered.iloc[-1]
        snapshot = eligible.iloc[-1]
        if require_later_target:
            later = ordered[ordered["time_to_tca"] < snapshot["time_to_tca"]]
            if later.empty:
                continue
            target_row = later.iloc[-1]
        events.append(
            {
                "event_id": event_id,
                "mission_id": snapshot["mission_id"],
                "snapshot": snapshot,
                "history": eligible,
                "full_history": ordered,
                "y": float(target_row["risk"]),
                "target_time_to_tca": float(target_row["time_to_tca"]),
                "story": snapshot["story"] if "story" in snapshot else None,
            }
        )
    return events
