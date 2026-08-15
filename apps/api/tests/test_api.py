from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from main import app

ROOT = Path(__file__).resolve().parents[3]
CASES = ROOT / "ml" / "artifacts" / "demo_cases.json"


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predicts_curated_case() -> None:
    assert CASES.exists()
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    client = TestClient(app)
    response = client.post(
        "/v1/risk/predict",
        json={"eventId": cases[0]["id"], "cutoffHours": 48, "messages": cases[0]["messages"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert "predictedFinalRiskLog10" in body
    assert body["disclaimer"]


def test_rejects_post_cutoff_messages() -> None:
    assert CASES.exists()
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    messages = list(cases[0]["messages"])
    messages.append(
        {
            "timeToTcaDays": 0.4,
            "riskLog10": -4.0,
            "missDistanceM": 100.0,
            "relativeSpeedMps": 8000.0,
        }
    )
    client = TestClient(app)
    response = client.post(
        "/v1/risk/predict",
        json={"eventId": "leak", "cutoffHours": 48, "messages": messages},
    )
    assert response.status_code == 400
