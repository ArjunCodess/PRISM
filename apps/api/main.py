from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[2]
ML_SRC = ROOT / "ml" / "src"
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))

from constants import DISCLAIMER, HIGH_RISK_THRESHOLD  # noqa: E402
from inference import PrismModel  # noqa: E402

app = FastAPI(title="PRISM API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    timeToTcaDays: float = Field(ge=0, le=365)
    riskLog10: float = Field(ge=-30, le=0)
    missDistanceM: float = Field(gt=0, le=10_000_000)
    relativeSpeedMps: float = Field(gt=0, le=100_000)
    tSigmaR: float | None = Field(default=None, ge=0)
    cSigmaR: float | None = Field(default=None, ge=0)
    tObsUsed: float | None = Field(default=None, ge=0)
    cObsUsed: float | None = Field(default=None, ge=0)
    cObjectType: str | None = None


class PredictRequest(BaseModel):
    eventId: str
    cutoffHours: int = 48
    messages: list[Message] = Field(min_length=1)


class FactorModel(BaseModel):
    feature: str
    direction: str
    contribution: float
    label: str | None = None


class PredictResponse(BaseModel):
    predictedFinalRiskLog10: float
    predictedFinalPc: float
    interval90Log10: list[float]
    interval50Log10: list[float]
    configuredHighRiskProbability: float
    highRiskThresholdLog10: float
    riskBand: str
    abstained: bool
    abstentionReasons: list[str] | None = None
    topFactors: list[FactorModel]
    explanation: str | None = None
    disclaimer: str
    cutoffHours: int | None = None
    nMessagesUsed: int | None = None
    eventId: str | None = None


@lru_cache(maxsize=1)
def get_model() -> PrismModel:
    return PrismModel()


def demo_cases() -> list[dict[str, Any]]:
    path = ROOT / "ml" / "artifacts" / "demo_cases.json"
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/model-card")
def model_card() -> dict[str, Any]:
    path = ROOT / "ml" / "artifacts" / "model_card.json"
    metrics = json.loads((ROOT / "ml" / "artifacts" / "metrics.json").read_text(encoding="utf-8"))
    card = json.loads(path.read_text(encoding="utf-8"))
    card["metricsFull"] = metrics
    card["disclaimer"] = DISCLAIMER
    card["highRiskThresholdLog10"] = HIGH_RISK_THRESHOLD
    return card


@app.get("/v1/cases")
def list_cases() -> list[dict[str, Any]]:
    return demo_cases()


@app.get("/v1/cases/{case_id}")
def get_case(case_id: str) -> dict[str, Any]:
    for item in demo_cases():
        if item["id"] == case_id:
            return item
    raise HTTPException(status_code=404, detail="case not found")


@app.post("/v1/risk/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if payload.cutoffHours != 48:
        raise HTTPException(status_code=400, detail="only a 48-hour cutoff is supported")
    if any(item.timeToTcaDays < 2.0 for item in payload.messages):
        raise HTTPException(status_code=400, detail="post-cutoff messages are not allowed")
    try:
        result = get_model().predict_messages(
            payload.eventId,
            [item.model_dump(exclude_none=True) for item in payload.messages],
            cutoff_hours=payload.cutoffHours,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictResponse.model_validate(result)
