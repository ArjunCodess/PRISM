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


def test_one_row_per_event(
    events_and_features: tuple[list[dict[str, object]], pd.DataFrame],
) -> None:
    _events, features = events_and_features
    assert features["event_id"].is_unique
    assert len(features) == features["event_id"].nunique()


def test_no_post_cutoff_in_history(
    events_and_features: tuple[list[dict[str, object]], pd.DataFrame],
) -> None:
    events, _features = events_and_features
    for event in events:
        history: pd.DataFrame = event["history"]  # type: ignore[assignment]
        cutoff = float(event.get("cutoff_days", CUTOFF_DAYS))
        assert (history["time_to_tca"] >= cutoff - 1e-9).all()


def test_target_uses_later_cdm(
    events_and_features: tuple[list[dict[str, object]], pd.DataFrame],
) -> None:
    events, features = events_and_features
    leaked = [col for col in features.columns if col.startswith("target_")]
    assert leaked == []
    for event in events:
        snapshot: pd.Series = event["snapshot"]  # type: ignore[assignment]
        assert float(event["target_time_to_tca"]) < float(snapshot["time_to_tca"]) - 1e-9
        row = features.loc[features["event_id"] == event["event_id"]].iloc[0]
        assert float(row["risk"]) == pytest.approx(float(snapshot["risk"]))
        assert float(row["y"]) == pytest.approx(float(event["y"]))


def test_grouped_splits_are_disjoint(
    events_and_features: tuple[list[dict[str, object]], pd.DataFrame],
) -> None:
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


def test_derived_geometry_is_checked(
    events_and_features: tuple[list[dict[str, object]], pd.DataFrame],
) -> None:
    _events, features = events_and_features
    assert np.isfinite(features["derived_miss_distance"]).all()
    assert np.isfinite(features["derived_relative_speed"]).all()
    assert np.isfinite(features["miss_distance_residual"]).all()
    # Synthetic miss_distance is an independent draw; the residual is a quality flag,
    # never a silent replacement.
    assert "miss_distance" in features.columns
    assert not np.allclose(features["derived_miss_distance"], features["miss_distance"])
    numeric = features.select_dtypes(include=[np.number])
    assert not np.isinf(numeric.to_numpy()).any()


def test_validator_allows_final_messages_shortly_after_tca() -> None:
    frame = generate_synthetic_cdms(n_events=4, seed=11)
    final_index = frame.groupby("event_id")["time_to_tca"].idxmin().iloc[0]
    frame.loc[final_index, "time_to_tca"] = -0.1
    validated = validate_cdm_frame(frame)
    assert validated["time_to_tca"].min() == pytest.approx(-0.1)


def test_validator_rejects_implausibly_late_messages() -> None:
    frame = generate_synthetic_cdms(n_events=4, seed=12)
    frame.loc[frame.index[0], "time_to_tca"] = -1.1
    with pytest.raises(ValueError, match="more than one day"):
        validate_cdm_frame(frame)
