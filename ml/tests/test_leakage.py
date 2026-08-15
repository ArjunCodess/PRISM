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


def test_target_uses_later_cdm(events_and_features: tuple[list[dict[str, object]], pd.DataFrame]) -> None:
    events, features = events_and_features
    leaked = [col for col in features.columns if col.startswith("target_")]
    assert leaked == []
    for event in events:
        snapshot: pd.Series = event["snapshot"]  # type: ignore[assignment]
        assert float(event["target_time_to_tca"]) < float(snapshot["time_to_tca"]) - 1e-9
        row = features.loc[features["event_id"] == event["event_id"]].iloc[0]
        assert float(row["risk"]) == pytest.approx(float(snapshot["risk"]))
        assert float(row["y"]) == pytest.approx(float(event["y"]))


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


def test_derived_geometry_is_checked(events_and_features: tuple[list[dict[str, object]], pd.DataFrame]) -> None:
    _events, features = events_and_features
    assert np.isfinite(features["derived_miss_distance"]).all()
    assert np.isfinite(features["derived_relative_speed"]).all()
    assert np.isfinite(features["miss_distance_residual"]).all()
    # Synthetic miss_distance is an independent draw; residual is a quality flag, never a silent replacement.
    assert "miss_distance" in features.columns
    assert not np.allclose(features["derived_miss_distance"], features["miss_distance"])
