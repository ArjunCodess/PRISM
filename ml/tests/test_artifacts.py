from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "ml" / "artifacts"


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
