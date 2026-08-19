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


def test_demo_cases_match_exhibit() -> None:
    cases = json.loads((ART / "demo_cases.json").read_text(encoding="utf-8"))
    stories = [item["story"] for item in cases]
    assert len(cases) == 6
    assert stories.count("low") == 2
    assert stories.count("uncertain") == 1
    assert stories.count("high") == 3
    assert all(item["prediction"]["disclaimer"] for item in cases)
    lows = [item for item in cases if item["story"] == "low"]
    assert all(not item["prediction"]["abstained"] for item in lows)
    assert all(item["prediction"]["riskBand"] == "low" for item in lows)
    uncertain = next(item for item in cases if item["story"] == "uncertain")
    assert uncertain["prediction"]["abstained"]
    assert uncertain["prediction"]["abstentionReasons"]
    highs = [item for item in cases if item["story"] == "high"]
    assert all(not item["prediction"]["abstained"] for item in highs)
    assert all(item["prediction"]["riskBand"] == "high" for item in highs)
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


def test_honest_metrics_exist_on_frozen_split() -> None:
    metrics = json.loads((ART / "metrics.json").read_text(encoding="utf-8"))
    honest = metrics["honestMetrics"]
    assert honest["nTest"] == metrics["splits"]["test"]
    for name in ("persistence", "xgboost", "ensemble"):
        system = honest["systems"][name]
        assert "median_ae" in system["all"]
        assert "mae" in system["nonFloor"]
        assert "residualMae" in system
    xgb = honest["systems"]["xgboost"]["maeAdvantageVsPersistence"]
    ens = honest["systems"]["ensemble"]["maeAdvantageVsPersistence"]
    assert xgb["ci95Low"] < xgb["ci95High"]
    assert ens["ci95Low"] < ens["ci95High"]
    assert np.isfinite(xgb["deltaMae"])
    assert np.isfinite(ens["deltaMae"])


def test_residual_candidate_exists_on_frozen_split() -> None:
    metrics = json.loads((ART / "metrics.json").read_text(encoding="utf-8"))
    residual = metrics["residualModel"]
    assert residual["replacesExhibit"] is False
    assert residual["winnerSoFar"]["split"] == "validation"
    for split_name in ("test", "validation"):
        for system in ("persistence", "xgboost", "residual"):
            row = residual[split_name][system]
            assert "mae" in row
            assert "medianAe" in row
            assert "floorExcludedMae" in row
            assert "esaLoss" in row
            assert "f2" in row
    assert "residual" in metrics["honestMetrics"]["systems"]
    assert (ART / "residual_regressor.json").exists()


def test_floor_candidate_exists_on_frozen_split() -> None:
    metrics = json.loads((ART / "metrics.json").read_text(encoding="utf-8"))
    floor = metrics["floorModel"]
    assert floor["replacesExhibit"] is False
    assert floor["winnerSoFar"]["split"] == "validation"
    assert "threshold" in floor
    confusion = floor["confusion"]["test"]
    assert confusion["nFloor"] == metrics["honestMetrics"]["nFloor"]
    assert confusion["tp"] + confusion["fn"] == confusion["nFloor"]
    assert confusion["fp"] + confusion["tn"] == confusion["nNonFloor"]
    for split_name in ("test", "validation"):
        row = floor[split_name]["floorHurdle"]
        assert "mae" in row
        assert "floorExcludedMae" in row
        assert "esaLoss" in row
    assert "floorHurdle" in metrics["honestMetrics"]["systems"]
    assert (ART / "floor_classifier.json").exists()
    assert (ART / "floor_residual_regressor.json").exists()
    assert (ART / "floor_hurdle.json").exists()


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
