from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from constants import RANDOM_STATE
from sklearn.model_selection import GroupShuffleSplit

REDRAW_SEEDS = (42, 43, 44, 45, 46)

@dataclass(frozen=True)
class SplitManifest:
    train_ids: list[int]
    validation_ids: list[int]
    calibration_ids: list[int]
    test_ids: list[int]


def grouped_splits(frame: pd.DataFrame, seed: int = RANDOM_STATE) -> SplitManifest:
    event_ids = frame["event_id"].to_numpy()
    unique = np.unique(event_ids)
    groups = unique
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    dummy_x = unique.reshape(-1, 1)
    dummy_y = np.zeros(len(unique))
    train_val_idx, test_idx = next(splitter.split(dummy_x, dummy_y, groups))
    train_val_ids = unique[train_val_idx]
    test_ids = unique[test_idx]

    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
    dummy_inner = train_val_ids.reshape(-1, 1)
    train_cal_idx, val_idx = next(
        inner.split(dummy_inner, np.zeros(len(train_val_ids)), train_val_ids)
    )
    train_cal_ids = train_val_ids[train_cal_idx]
    validation_ids = train_val_ids[val_idx]

    cal_split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 2)
    dummy_cal = train_cal_ids.reshape(-1, 1)
    train_idx, cal_idx = next(
        cal_split.split(dummy_cal, np.zeros(len(train_cal_ids)), train_cal_ids)
    )
    train_ids = train_cal_ids[train_idx]
    calibration_ids = train_cal_ids[cal_idx]

    return SplitManifest(
        train_ids=[int(x) for x in train_ids],
        validation_ids=[int(x) for x in validation_ids],
        calibration_ids=[int(x) for x in calibration_ids],
        test_ids=[int(x) for x in test_ids],
    )


def subset(frame: pd.DataFrame, ids: list[int]) -> pd.DataFrame:
    return frame[frame["event_id"].isin(ids)].copy()
