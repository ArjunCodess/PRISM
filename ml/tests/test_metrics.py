from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from constants import NEGLIGIBLE_RISK
from evaluate import (
    clip_for_esa,
    esa_loss,
    false_reassurance_analogue,
    floor_mask,
    paired_wilcoxon_abs_error,
    bootstrap_mae_advantage,
)
from experiments import threshold_sweep, error_anatomy
from explain import Factor, explanation_text, grouped_importance


def test_esa_loss_penalizes_false_negatives() -> None:
    y = np.array([-5.0, -7.0, -4.5])
    good = np.array([-5.1, -6.2, -4.6])
    bad = np.array([-7.0, -7.0, -7.0])
    assert esa_loss(y, clip_for_esa(good))["esa_loss"] < esa_loss(y, clip_for_esa(bad))["esa_loss"]


def test_explanation_is_deterministic() -> None:
    factors = [
        Factor("risk_slope", "higher", 0.4, "whether chance is climbing"),
        Factor("miss_distance", "lower", -0.2, "predicted miss distance"),
    ]
    text = explanation_text(factors)
    assert "More worrying" in text
    assert "Less worrying" in text
    assert explanation_text(factors) == text


def test_floor_mask_tags_dataset_floor() -> None:
    y = np.array([NEGLIGIBLE_RISK, NEGLIGIBLE_RISK + 1e-9, -29.0, -6.0])
    mask = floor_mask(y)
    assert mask.tolist() == [True, True, False, False]


def test_bootstrap_mae_ci_is_a_finite_interval() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(-10.0, 2.0, size=40)
    persist = y + rng.normal(0.0, 1.5, size=40)
    pred = y + rng.normal(0.0, 0.4, size=40)
    ci = bootstrap_mae_advantage(y, pred, persist, n_bootstrap=200, seed=0)
    assert np.isfinite(ci["ci95Low"])
    assert np.isfinite(ci["ci95High"])
    assert np.isfinite(ci["deltaMae"])
    assert ci["ci95Low"] < ci["ci95High"]


def test_wilcoxon_runs_on_a_tiny_synthetic_pair() -> None:
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    closer = y + 0.1
    farther = y + 2.0
    result = paired_wilcoxon_abs_error(y, closer, farther)
    assert np.isfinite(result["statistic"])
    assert 0.0 <= float(result["pvalue"]) <= 1.0
    assert result["n"] == 6


def test_trend_features_are_not_lumped_into_current_risk() -> None:
    groups = grouped_importance(
        ["risk", "risk_change"],
        {"risk": 1.0, "risk_change": 2.0},
    )
    names = {item["group"]: item["gain"] for item in groups}
    assert names["today's reported risk"] == 1.0
    assert names["whether risk is climbing or falling"] == 2.0


def test_esa_loss_at_minus_six_still_clips_to_low_risk() -> None:
    clipped = clip_for_esa(np.array([-7.0, -5.0]), threshold=-6.0)
    assert clipped[0] == -6.001
    assert clipped[1] == -5.0


def test_threshold_sweep_uses_same_predictions() -> None:
    y = np.array([-3.5, -4.5, -5.5, -6.5, -7.5])
    persist = np.array([-3.6, -7.0, -5.6, -6.4, -7.4])
    other = persist.copy()
    sweep = threshold_sweep(y, {"persistence": persist, "xgboost": other})
    n_pos = [row["nPositives"] for row in sweep["rows"]]
    assert n_pos == [5, 4, 3, 2, 1]
    assert sweep["retuned"] is False
    minus_six = next(row for row in sweep["rows"] if row["threshold"] == -6.0)
    analogue = false_reassurance_analogue(y, persist, threshold=-6.0)
    assert analogue["nPositives"] == 3
    assert analogue["missedClass"] == 1
    assert minus_six["systems"]["persistence"]["missedClass"] == 1
    assert minus_six["nPositives"] == 3


def test_error_anatomy_counts_floor_collapses() -> None:
    y = np.array([-30.0, -30.0, -6.0])
    risk = np.array([-8.0, -30.0, -6.1])
    pred = np.array([-10.0, -30.0, -6.0])
    payload = error_anatomy(y, risk, pred)
    assert payload["nFloorCollapse"] == 1
    assert payload["nUnmoved"] == 1
    assert payload["exactPersistenceShare"] == pytest.approx(1.0 / 3.0)
    assert "p90AbsError" in payload
    assert payload["floor"]["n"] == 2
    assert payload["nonFloor"]["n"] == 1
    assert sum(payload["actualMoveCounts"]) == 3
    assert sum(payload["residualErrorCounts"]) == 3
