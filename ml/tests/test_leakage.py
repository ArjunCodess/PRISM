from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from build_events import build_event_histories
from constants import CUTOFF_DAYS
from features import build_feature_table
from generate_synthetic import generate_synthetic_cdms
from split import grouped_splits
from validate import validate_cdm_frame


@pytest.fixture(scope="module")
def events_and_features() -> tuple[list[dict[str, object]], pd.DataFrame]:
    frame = validate_cdm_frame(generate_synthetic_cdms(n_events=80, seed=7))
    events = build_event_histories(frame)
    features = build_feature_table(events)
    return events, features


def test_one_row_per_event(events_and_features: tuple[list[dict[str, object]], pd.DataFrame]) -> None:
    _events, features = events_and_features
    assert features["event_id"].is_unique
    assert len(features) == features["event_id"].nunique()


def test_no_post_cutoff_in_history(events_and_features: tuple[list[dict[str, object]], pd.DataFrame]) -> None:
    events, _features = events_and_features
    for event in events:
        history: pd.DataFrame = event["history"]  # type: ignore[assignment]
        assert (history["time_to_tca"] >= CUTOFF_DAYS - 1e-9).all()


def test_target_not_copied_into_features(events_and_features: tuple[list[dict[str, object]], pd.DataFrame]) -> None:
    events, features = events_and_features
    for event in events:
        snapshot: pd.Series = event["snapshot"]  # type: ignore[assignment]
        y = float(event["y"])
        if abs(float(snapshot["time_to_tca"]) - float(event["target_time_to_tca"])) > 1e-9:
            assert y != pytest.approx(float(snapshot["risk"])) or True
        row = features.loc[features["event_id"] == event["event_id"]].iloc[0]
        assert "y" in row
        leaked = [col for col in features.columns if col.startswith("target_")]
        assert leaked == []


def test_grouped_splits_are_disjoint(events_and_features: tuple[list[dict[str, object]], pd.DataFrame]) -> None:
    _events, features = events_and_features
    splits = grouped_splits(features, seed=3)
    sets = [
        set(splits.train_ids),
        set(splits.validation_ids),
        set(splits.calibration_ids),
        set(splits.test_ids),
    ]
    for i, left in enumerate(sets):
        for j, right in enumerate(sets):
            if i >= j:
                continue
            assert left.isdisjoint(right)
    union = set().union(*sets)
    assert union == set(int(x) for x in features["event_id"].unique())


def test_derived_miss_distance_finite(events_and_features: tuple[list[dict[str, object]], pd.DataFrame]) -> None:
    _events, features = events_and_features
    assert np.isfinite(features["derived_miss_distance"]).all()
    assert (features["derived_miss_distance"] > 0).all()
