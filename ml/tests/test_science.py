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
from experiments import cluster_test_failures, dilution_probe  # noqa: E402
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
    numeric_columns,
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


def test_split_conformal_covers_gaussian_near_nominal() -> None:
    from calibrate import conformal_bounds, interval_report, split_conformal_quantile

    rng = np.random.default_rng(0)
    y_cal = rng.normal(0.0, 1.0, size=800)
    y_test = rng.normal(0.0, 1.0, size=800)
    pred = np.zeros(800)
    q90 = split_conformal_quantile(np.abs(y_cal - pred), 0.1)
    lo, hi = conformal_bounds(pred, q90)
    report = interval_report(y_test, lo, hi)
    assert 0.86 <= report["coverage"] <= 0.94
    q50 = split_conformal_quantile(np.abs(y_cal - pred), 0.5)
    lo50, hi50 = conformal_bounds(pred, q50)
    mid = interval_report(y_test, lo50, hi50)
    assert 0.42 <= mid["coverage"] <= 0.60


def test_dilution_gap_is_max_risk_minus_risk() -> None:
    frame = validate_cdm_frame(generate_synthetic_cdms(n_events=40, seed=3))
    features = build_feature_table(build_event_histories(frame))
    expected = features["max_risk_estimate"] - features["risk"]
    np.testing.assert_allclose(features["dilution_gap"], expected, equal_nan=True)
    assert "dilution_gap" not in numeric_columns(features)


def test_dilution_probe_logistic_recovers_floor_signal() -> None:
    rng = np.random.default_rng(0)
    n = 400
    gap = rng.normal(2.0, 1.0, size=n)
    miss = rng.uniform(50.0, 2000.0, size=n)
    messages = rng.integers(2, 12, size=n).astype(float)
    floor = (gap + 0.001 * miss > 2.2).astype(float)
    y = np.where(floor > 0.5, -30.0, -8.0)
    risk = np.full(n, -8.0)
    frame = pd.DataFrame(
        {
            "event_id": np.arange(n),
            "y": y,
            "risk": risk,
            "max_risk_estimate": risk + gap,
            "dilution_gap": gap,
            "miss_distance": miss,
            "n_messages": messages,
            "F10": rng.normal(70.0, 5.0, size=n),
            "log_t_cov_det": rng.normal(0.0, 1.0, size=n),
        }
    )
    train = frame.iloc[:280].copy()
    test = frame.iloc[280:].copy()
    report = dilution_probe(train, test)
    rho = float(report["spearmanAbsMove"]["dilution_gap"]["rho"])
    assert -1.0 <= rho <= 1.0
    assert report["logisticFloor"]["testAuc"] > 0.75
    assert len(report["quartiles"]) >= 2
    assert report["replacesExhibit"] is False


def test_repeated_splits_and_loo_run_on_synthetic() -> None:
    from constants import HIGH_RISK_THRESHOLD
    from experiments import leave_one_high_risk_out, repeated_grouped_splits

    frame = validate_cdm_frame(generate_synthetic_cdms(n_events=60, seed=5))
    features = build_feature_table(build_event_histories(frame))
    repeats = repeated_grouped_splits(features, seeds=(5, 6))
    assert repeats["replacesExhibit"] is False
    assert len(repeats["splits"]) == 2
    assert np.isfinite(repeats["summary"]["xgboost"]["maeAdvantage"]["mean"])
    loo = leave_one_high_risk_out(features)
    n_high = int((features["y"].to_numpy() >= HIGH_RISK_THRESHOLD).sum())
    assert loo["nHighRisk"] == n_high
    assert loo["persistCloser"] + loo["residualCloser"] + loo["ties"] == n_high


def test_censoring_sensitivity_changes_floor_rate() -> None:
    from review_armor import censoring_sensitivity, trend_predict

    y = np.array([-30.0, -22.0, -8.0, -6.0])
    persist = np.array([-10.0, -22.0, -8.0, -6.0])
    pred = np.array([-30.0, -22.0, -7.5, -12.0])
    payload = censoring_sensitivity(y, persist, pred, pred)
    rates = [row["floorRate"] for row in payload["thresholds"]]
    assert rates[0] > rates[-1]
    assert payload["thresholds"][-1]["threshold"] == -30.0
    frame = pd.DataFrame({"risk": [-10.0, -8.0], "risk_delta_last2": [-2.0, 0.0]})
    np.testing.assert_allclose(trend_predict(frame), np.array([-12.0, -8.0]))


def test_floor_classifier_eval_counts_false_positives() -> None:
    from review_armor import floor_classifier_evaluation

    y = np.array([-30.0, -30.0, -8.0, -7.0])
    proba = np.array([0.9, 0.2, 0.8, 0.1])
    risk = np.array([-12.0, -11.0, -9.0, -7.0])
    gap = np.array([8.0, 7.0, 1.0, 0.2])
    payload = floor_classifier_evaluation(y, proba, risk, gap, threshold=0.5)
    assert payload["falsePositives"]["n"] == 1
    assert payload["scores"]["roc_auc"] >= 0.0
    assert any(row["threshold"] == 0.15 for row in payload["confusionGrid"])


def test_matched_cohort_uses_same_events() -> None:
    from review_armor import matched_cohort_horizons
    from split import grouped_splits

    frame = validate_cdm_frame(generate_synthetic_cdms(n_events=80, seed=11))
    features = build_feature_table(build_event_histories(frame))
    splits = grouped_splits(features, seed=11)
    payload = matched_cohort_horizons(frame, splits.train_ids, splits.test_ids)
    ns = {row["nTest"] for row in payload["rows"]}
    assert len(ns) == 1
    assert payload["nTest"] == next(iter(ns))


def test_selective_curve_sweeps_several_nominal_coverages() -> None:
    from review_armor import CONFORMAL_ALPHAS, selective_prediction_curve

    assert len(CONFORMAL_ALPHAS) >= 4
    y = np.array([-30.0, -30.0, -8.0, -5.0, -12.0, -30.0])
    pred = np.array([-30.0, -29.0, -9.0, -20.0, -11.0, -30.0])
    persist = np.array([-12.0, -11.0, -8.0, -6.0, -12.0, -10.0])
    cal_y = np.array([-30.0, -30.0, -8.0, -7.0])
    cal_pred = np.array([-30.0, -30.0, -9.0, -8.0])
    risk = np.array([-12.0, -11.0, -8.0, -6.5, -12.0, -10.0])
    miss = np.full(6, 800.0)
    curve = selective_prediction_curve(y, pred, persist, cal_y, cal_pred, risk, miss)
    nominal = [row["nominalCoverage"] for row in curve["curve"]]
    assert nominal == sorted(nominal)
    assert len(set(nominal)) >= 4
    assert max(row["nAbstained"] for row in curve["curve"]) >= 0


def test_h4_partial_effects_runs_on_tiny_table() -> None:
    from review_armor import h4_partial_effects

    n = 40
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "y": rng.uniform(-30, -6, n),
            "risk": rng.uniform(-20, -6, n),
            "log_combined_sigma_det": rng.normal(0, 1, n),
            "n_messages": rng.integers(2, 10, n).astype(float),
            "miss_distance": rng.uniform(100, 2000, n),
            "max_risk_estimate": rng.uniform(-12, -4, n),
        }
    )
    payload = h4_partial_effects(frame)
    assert "risk" in payload["standardizedCoefficients"]
    assert payload["language"].startswith("association")
    from calibrate import interval_report

    y = np.array([-8.0, -30.0, -6.0])
    lo = np.array([-10.0, -32.0, -20.0])
    hi = np.array([-6.0, -28.0, -10.0])
    report = interval_report(y, lo, hi)
    assert report["nCovered"] == 2
    assert report["medianWidth"] == pytest.approx(4.0)
    assert report["q25Width"] <= report["q75Width"]
