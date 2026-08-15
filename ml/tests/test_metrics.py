from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from evaluate import clip_for_esa, esa_loss
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


def test_trend_features_are_not_lumped_into_current_risk() -> None:
    groups = grouped_importance(
        ["risk", "risk_change"],
        {"risk": 1.0, "risk_change": 2.0},
    )
    names = {item["group"]: item["gain"] for item in groups}
    assert names["today's reported risk"] == 1.0
    assert names["whether risk is climbing or falling"] == 2.0
