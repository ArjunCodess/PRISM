from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "ml" / "artifacts"
SRC = ROOT / "ml" / "src"
sys.path.insert(0, str(SRC))

from explain import local_factors, shap_explainer
from export_demo_cases import risk_band
from train_regressor import TrainedRegressor


def test_existing_high_risk_forecast_is_never_labelled_low() -> None:
    assert risk_band(0.01, False, -5.5) == "high"
    assert risk_band(0.01, True, -5.5) == "review"


def test_frozen_artifacts_exist() -> None:
    for name in [
        "risk_regressor.json",
        "warning_calibrator.joblib",
        "feature_schema.json",
        "metrics.json",
        "demo_cases.json",
        "model_card.json",
    ]:
        assert (ART / name).exists(), name


def test_demo_cases_match_prd() -> None:
    cases = json.loads((ART / "demo_cases.json").read_text(encoding="utf-8"))
    stories = {item["story"] for item in cases}
    assert stories == {"low", "escalate", "deescalate", "uncertain", "failure"}
    assert any(item["prediction"]["abstained"] for item in cases)
    assert all(item["prediction"]["disclaimer"] for item in cases)
    failure = next(item for item in cases if item["story"] == "failure")
    pred = float(failure["prediction"]["predictedFinalRiskLog10"])
    actual = float(failure["actualFinalRiskLog10"])
    persist = float(failure["baselineRiskLog10"])
    missed = actual >= -6 and pred < -6
    late = actual > persist + 0.4
    under = pred < actual - 0.35
    confident = not failure["prediction"]["abstained"]
    assert confident
    assert (missed or (late and under) or abs(pred - actual) >= 2.0) and abs(pred - actual) >= 0.35
    uncertain = next(item for item in cases if item["story"] == "uncertain")
    assert uncertain["prediction"]["abstained"]
    assert uncertain["prediction"]["abstentionReasons"]
    for item in cases:
        assert all(msg["timeToTcaDays"] >= 2.0 for msg in item["messages"])
        if item["futureMessages"]:
            assert all(msg["timeToTcaDays"] < 2.0 for msg in item["futureMessages"])


def test_persistence_claim_matches_frozen_metrics() -> None:
    metrics = json.loads((ART / "metrics.json").read_text(encoding="utf-8"))
    actually_beats = (
        metrics["ensemble"]["mae"] < metrics["persistence"]["mae"]
        and metrics["ensemble"]["esa_loss"] < metrics["persistence"]["esa_loss"]
    )
    assert bool(metrics["improvement"]["beats_persistence"]) is actually_beats


def test_reloaded_booster_matches_saved_schema() -> None:
    schema = json.loads((ART / "feature_schema.json").read_text(encoding="utf-8"))["features"]
    model = XGBRegressor()
    model.load_model(ART / "risk_regressor.json")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(8, len(schema))), columns=schema)
    pred = np.asarray(model.predict(frame), dtype=float)
    assert pred.shape == (8,)
    assert np.isfinite(pred).all()


def test_shap_adds_to_prediction() -> None:
    schema = json.loads((ART / "feature_schema.json").read_text(encoding="utf-8"))["features"]
    model = XGBRegressor()
    model.load_model(ART / "risk_regressor.json")
    trained = TrainedRegressor(model=model, feature_names=schema, kind="xgboost")
    explainer = shap_explainer(trained, pd.DataFrame(np.zeros((4, len(schema))), columns=schema))
    row = pd.Series({name: 0.0 for name in schema})
    base, factors = local_factors(trained, explainer, row, top_k=len(schema))
    x = row[schema].to_frame().T
    pred = float(model.predict(x)[0])
    shap_sum = base + sum(item.contribution for item in factors)
    assert shap_sum == pytest.approx(pred, abs=1e-3)
