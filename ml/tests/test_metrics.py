from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from constants import NEGLIGIBLE_RISK
from evaluate import (
    bootstrap_mae_advantage,
    clip_for_esa,
    esa_loss,
    floor_mask,
    paired_wilcoxon_abs_error,
)
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
