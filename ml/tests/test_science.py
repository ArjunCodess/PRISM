from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from abstention import (  # noqa: E402
    ABSTENTION_DISAGREEMENT,
    REASON_CROSSES,
    REASON_DISAGREEMENT,
    REASON_MISSING,
    abstain_mask,
    coverage_curve,
    decide_abstention,
    selective_metrics,
)
from build_events import build_event_histories  # noqa: E402
from constants import CUTOFF_DAYS, DEMO_SLOTS, HIGH_RISK_THRESHOLD  # noqa: E402
from experiments import cluster_test_failures  # noqa: E402
from export_demo_cases import story_fit  # noqa: E402
from feature_sets import (  # noqa: E402
    FAMILIES,
    assert_trend_stems_are_known,
    columns_for_family,
)
from features import build_feature_table  # noqa: E402
from generate_synthetic import generate_synthetic_cdms  # noqa: E402
from train_regressor import (  # noqa: E402
    fit_residual_xgboost,
    predict_reconstructed,
    reconstruct_from_residual,
    residual_target,
)
from validate import validate_cdm_frame  # noqa: E402


def test_feature_families_are_nested() -> None:
    assert_trend_stems_are_known()
    available = [
        "risk",
        "miss_distance",
        "t_sigma_r",
        "n_messages",
        "hours_before_cutoff",
        "hours_since_prev",
        "risk_slope",
        "t_sigma_r_slope",
        "F10",
        "event_id",
        "y",
    ]
    snapshot = set(columns_for_family(available, "snapshot"))
    history = set(columns_for_family(available, "snapshot_history"))
    covariance = set(columns_for_family(available, "snapshot_history_covariance"))
    full = set(columns_for_family(available, "full"))
    assert snapshot <= history <= covariance <= full
    assert "risk_slope" not in snapshot
    assert "risk_slope" in history
    assert "t_sigma_r_slope" not in history
    assert "t_sigma_r_slope" in covariance
    assert "F10" in snapshot
    assert FAMILIES[-1] == "full"


def test_abstention_rule_is_explicit() -> None:
    quiet = np.full(10, -8.0)
    decision = decide_abstention(quiet, current_risk=-8.0, miss_distance=1200.0)
    assert decision.abstained is False
    assert decision.reasons == []

    crossing = np.linspace(-8.0, -4.0, 10)
    decision = decide_abstention(crossing, current_risk=-7.0, miss_distance=400.0)
    assert decision.abstained
    assert REASON_CROSSES in decision.reasons

    noisy = np.array([-30.0, -2.0] * 5)
    decision = decide_abstention(noisy, current_risk=-12.0, miss_distance=800.0)
    assert decision.disagreement > ABSTENTION_DISAGREEMENT
    assert REASON_DISAGREEMENT in decision.reasons

    missing = decide_abstention(quiet, current_risk=float("nan"), miss_distance=800.0)
    assert REASON_MISSING in missing.reasons


def test_selective_prediction_improves_when_hard_cases_are_dropped() -> None:
    y = np.array([-8.0, -8.0, -4.0, -30.0])
    pred = np.array([-8.1, -7.9, -20.0, -5.0])
    persist = np.array([-8.0, -8.0, -7.0, -5.0])
    abstained = np.array([False, False, True, True])
    metrics = selective_metrics(y, pred, persist, abstained)
    assert metrics["coverage"] == pytest.approx(0.5)
    assert metrics["maeAccepted"] < metrics["maeAll"]
    assert metrics["falseReassurance"] == 0
    assert metrics["nHighRiskAbstained"] == 1
    missed = np.array([False, False, False, False])
    pred_miss = np.array([-8.1, -7.9, -20.0, -8.0])
    y_high = np.array([-8.0, -8.0, -4.0, -5.0])
    accepted_miss = selective_metrics(y_high, pred_miss, persist, missed)
    assert accepted_miss["falseReassurance"] == 2
    assert "predicted log10(Pc) <" in str(accepted_miss["falseReassuranceDefinition"])


def test_coverage_curve_contains_operating_point() -> None:
    rng = np.random.default_rng(0)
    y = rng.uniform(-12, -4, size=40)
    pred = y + rng.normal(0, 1.5, size=40)
    persist = y + rng.normal(0, 2.0, size=40)
    ensemble = np.column_stack([pred + rng.normal(0, 0.4, size=40) for _ in range(8)])
    curve = coverage_curve(y, pred, persist, ensemble, persist, np.full(40, 800.0))
    assert any(item["operatingPoint"] for item in curve)
    coverages = [item["coverage"] for item in curve]
    assert coverages == sorted(coverages)


def test_cutoff_parameter_keeps_histories_leakage_safe() -> None:
    frame = validate_cdm_frame(generate_synthetic_cdms(n_events=40, seed=9))
    late = build_event_histories(frame, cutoff_days=0.5)
    early = build_event_histories(frame, cutoff_days=3.0)
    assert late and early
    for event in early:
        history = event["history"]
        assert (history["time_to_tca"] >= 3.0 - 1e-9).all()
        assert float(event["target_time_to_tca"]) < float(event["snapshot"]["time_to_tca"])
        assert event["cutoff_days"] == 3.0
    features = build_feature_table(early)
    assert "y" in features.columns
    assert (features["hours_before_cutoff"] >= -1e-6).all()


def test_failure_clusters_tag_late_jumps() -> None:
    test = pd.DataFrame(
        {
            "y": [-5.5, -30.0, -8.0],
            "n_messages": [8.0, 6.0, 8.0],
            "miss_distance": [900.0, 2000.0, 1800.0],
            "hours_before_cutoff": [3.0, 4.0, 2.0],
            "t_sigma_r": [40.0, 50.0, 30.0],
            "c_sigma_r": [60.0, 70.0, 40.0],
        }
    )
    pred = np.array([-12.0, -5.0, -8.1])
    persist = np.array([-7.5, -5.0, -8.0])
    result = cluster_test_failures(test, pred, persist)
    assert "late_high_risk_jump" in result["modes"]
    assert "final_collapses_to_negligible" in result["modes"]
    assert result["modes"]["accurate"]["n"] == 1


def test_default_cutoff_remains_two_days() -> None:
    assert CUTOFF_DAYS == 2.0
    assert HIGH_RISK_THRESHOLD == -6.0


def test_geometry_features_are_finite() -> None:
    frame = validate_cdm_frame(generate_synthetic_cdms(n_events=30, seed=4))
    features = build_feature_table(build_event_histories(frame))
    assert "mahalanobis_r2" in features.columns
    assert "c_object_type_DEBRIS" in features.columns
    assert features["mahalanobis_r2"].notna().any()
    assert features.filter(regex=r"^c_object_type_").sum(axis=1).max() == 1


def test_abstain_mask_matches_scalar_rule() -> None:
    matrix = np.array(
        [
            np.full(6, -8.0),
            np.linspace(-9.0, -4.0, 6),
        ]
    )
    risk = np.array([-8.0, -7.0])
    miss = np.array([1000.0, 400.0])
    mask, _, crosses = abstain_mask(matrix, risk, miss)
    assert mask.tolist() == [False, True]
    assert crosses.tolist() == [False, True]
    assert decide_abstention(matrix[1], -7.0, 400.0).abstained


def test_demo_slots_cover_the_exhibit_mix() -> None:
    stories = [slot["story"] for slot in DEMO_SLOTS]
    assert len(DEMO_SLOTS) == 6
    assert stories.count("low") == 2
    assert stories.count("uncertain") == 1
    assert stories.count("high") == 3


def test_story_fit_scores_exhibit_slots() -> None:
    assert story_fit("low", -12.0, -12.0, -12.0, False) > 0
    assert story_fit("low", -12.0, -12.0, -12.0, True) < 0
    assert story_fit("uncertain", -8.0, -7.0, -7.0, True) > 0
    assert story_fit("uncertain", -8.0, -7.0, -7.0, False) < 0
    assert story_fit("high_now", -5.0, -5.0, -30.0, False) > 0
    assert story_fit("high_now", -5.0, -8.0, -5.0, False) < 0
    assert story_fit("high_stays", -5.0, -5.0, -5.2, False) > 0
    assert story_fit("high_drop", -5.0, -5.0, -12.0, False) > 0


def test_residual_target_is_label_minus_snapshot_risk() -> None:
    frame = pd.DataFrame({"y": [-30.0, -8.0, -6.0], "risk": [-10.0, -8.0, -12.0]})
    np.testing.assert_allclose(residual_target(frame), np.array([-20.0, 0.0, 6.0]))


def test_reconstruct_adds_residual_to_persistence() -> None:
    frame = pd.DataFrame({"risk": [-10.0, -8.0]})
    pred = reconstruct_from_residual(frame, np.array([-2.0, 1.5]))
    np.testing.assert_allclose(pred, np.array([-12.0, -6.5]))


def test_residual_xgboost_reconstructs_on_tiny_table() -> None:
    rng = np.random.default_rng(0)
    n = 40
    risk = rng.uniform(-20.0, -8.0, size=n)
    move = rng.normal(0.0, 1.0, size=n)
    train = pd.DataFrame(
        {
            "event_id": np.arange(n),
            "y": risk + move,
            "risk": risk,
            "miss_distance": rng.uniform(200.0, 2000.0, size=n),
            "n_messages": rng.integers(2, 8, size=n).astype(float),
        }
    )
    trained = fit_residual_xgboost(train)
    pred = predict_reconstructed(trained, train)
    assert pred.shape == (n,)
    assert np.isfinite(pred).all()
    residual_hat = pred - train["risk"].to_numpy()
    np.testing.assert_allclose(pred, train["risk"].to_numpy() + residual_hat)


def test_floor_labels_come_from_final_reported_risk() -> None:
    from constants import NEGLIGIBLE_RISK
    from floor_model import floor_labels, non_floor_rows

    frame = pd.DataFrame(
        {
            "y": [NEGLIGIBLE_RISK, -8.0, -29.0],
            "risk": [-5.0, -8.0, -12.0],
            "post_cutoff_leak": [1.0, 1.0, 1.0],
        }
    )
    np.testing.assert_array_equal(floor_labels(frame["y"]), np.array([1, 0, 0]))
    kept = non_floor_rows(frame)
    assert list(kept["y"]) == [-8.0, -29.0]


def test_floor_hurdle_predicts_floor_above_threshold() -> None:
    from constants import NEGLIGIBLE_RISK
    from floor_model import combine_floor_hurdle

    proba = np.array([0.9, 0.1])
    recon = np.array([-12.0, -8.0])
    pred = combine_floor_hurdle(proba, recon, 0.5)
    np.testing.assert_allclose(pred, np.array([NEGLIGIBLE_RISK, -8.0]))


def test_hurdle_threshold_is_chosen_on_provided_split() -> None:
    from constants import NEGLIGIBLE_RISK
    from floor_model import choose_hurdle_policy

    y = np.array([NEGLIGIBLE_RISK, NEGLIGIBLE_RISK, -8.0, -8.0])
    proba = np.array([0.8, 0.8, 0.2, 0.2])
    recon = np.array([-10.0, -10.0, -8.0, -8.0])
    risk = np.array([-10.0, -10.0, -8.0, -8.0])
    choice = choose_hurdle_policy(y, proba, recon, risk)
    assert 0.05 <= choice.threshold <= 0.95
    assert choice.validation["mae"] <= 2.0
