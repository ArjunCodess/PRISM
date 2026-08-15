from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from evaluate import clip_for_esa, esa_loss
from explain import Factor, explanation_text


def test_esa_loss_penalizes_false_negatives() -> None:
    y = np.array([-5.0, -7.0, -4.5])
    good = np.array([-5.1, -6.2, -4.6])
    bad = np.array([-7.0, -7.0, -7.0])
    assert esa_loss(y, clip_for_esa(good))["esa_loss"] < esa_loss(y, clip_for_esa(bad))["esa_loss"]


def test_explanation_is_deterministic() -> None:
    factors = [
        Factor("risk_slope", "higher", 0.4, "log-risk trend toward closest approach"),
        Factor("miss_distance", "lower", -0.2, "predicted miss distance at closest approach"),
    ]
    text = explanation_text(factors)
    assert text.startswith("The forecast")
    assert "miss distance" in text
    assert explanation_text(factors) == text
