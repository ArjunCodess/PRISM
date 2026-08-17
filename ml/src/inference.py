from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from build_events import build_event_histories
from constants import HIGH_RISK_THRESHOLD, MODEL_NAME
from explain import shap_explainer
from export_demo_cases import predict_event
from features import build_feature_table
from train_regressor import TrainedRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "ml" / "artifacts"


class PrismModel:
    def __init__(self) -> None:
        payload = json.loads((ARTIFACTS / "feature_schema.json").read_text(encoding="utf-8"))
        self.feature_names: list[str] = list(payload["features"])
        model = XGBRegressor()
        model.load_model(ARTIFACTS / "risk_regressor.json")
        bundle = joblib.load(ARTIFACTS / "warning_calibrator.joblib")
        self.calibrator = bundle["calibrator"]
        self.ensemble: list[XGBRegressor] = bundle["ensemble"]
        self.trained = TrainedRegressor(
            model=model, feature_names=self.feature_names, kind="xgboost"
        )
        background = pd.DataFrame(
            np.zeros((20, len(self.feature_names))), columns=self.feature_names
        )
        self.explainer = shap_explainer(self.trained, background)
        self.model_name = MODEL_NAME

    def predict_messages(
        self, event_id: str, messages: list[dict[str, Any]], cutoff_hours: int = 48
    ) -> dict[str, Any]:
        cutoff_days = cutoff_hours / 24.0
        if any(float(item["timeToTcaDays"]) < cutoff_days - 1e-9 for item in messages):
            raise ValueError("post-cutoff messages are not allowed")
        frame = _messages_to_frame(messages)
        events = build_event_histories(
            frame, require_later_target=False, cutoff_days=cutoff_days
        )
        if not events:
            raise ValueError("no eligible pre-cutoff messages")
        features = build_feature_table(events)
        row = features.iloc[0]
        aligned = pd.DataFrame(
            [
                {
                    name: float(row[name]) if name in row and pd.notna(row[name]) else np.nan
                    for name in self.feature_names
                }
            ]
        )
        series = aligned.iloc[0]
        ens = np.array([model.predict(aligned)[0] for model in self.ensemble])
        if float(series["risk"]) >= HIGH_RISK_THRESHOLD:
            ens.fill(float(series["risk"]))
        return predict_event(
            trained=self.trained,
            ensemble_preds=ens,
            calibrator=self.calibrator,
            explainer=self.explainer,
            row=series,
            messages=messages,
            event_id=event_id,
        )


def _num(item: dict[str, Any], key: str, default: float) -> float:
    value = item.get(key, default)
    if value is None:
        return default
    return float(value)


def _messages_to_frame(messages: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in messages:
        miss = float(item["missDistanceM"])
        speed = float(item["relativeSpeedMps"])
        rows.append(
            {
                "event_id": 0,
                "mission_id": 0,
                "time_to_tca": float(item["timeToTcaDays"]),
                "risk": float(item["riskLog10"]),
                "max_risk_estimate": _num(item, "maxRiskEstimate", float(item["riskLog10"]) + 0.4),
                "max_risk_scaling": _num(item, "maxRiskScaling", 1.0),
                "miss_distance": miss,
                "relative_speed": speed,
                "relative_position_r": _num(item, "relativePositionR", miss * 0.2),
                "relative_position_t": _num(item, "relativePositionT", miss * 0.8),
                "relative_position_n": _num(item, "relativePositionN", 10.0),
                "relative_velocity_r": _num(item, "relativeVelocityR", 0.0),
                "relative_velocity_t": _num(item, "relativeVelocityT", speed),
                "relative_velocity_n": _num(item, "relativeVelocityN", 0.0),
                "azimuth": _num(item, "azimuth", 0.0),
                "elevation": _num(item, "elevation", 0.0),
                "geocentric_latitude": _num(item, "geocentricLatitude", 0.0),
                "c_object_type": str(item.get("cObjectType") or "UNKNOWN"),
                "F10": _num(item, "f10", 100.0),
                "F3M": _num(item, "f3m", 100.0),
                "AP": _num(item, "ap", 10.0),
                "SSN": _num(item, "ssn", 50.0),
                "t_sigma_r": _num(item, "tSigmaR", 100.0),
                "t_sigma_t": _num(item, "tSigmaT", 140.0),
                "t_sigma_n": _num(item, "tSigmaN", 80.0),
                "c_sigma_r": _num(item, "cSigmaR", 150.0),
                "c_sigma_t": _num(item, "cSigmaT", 200.0),
                "c_sigma_n": _num(item, "cSigmaN", 90.0),
                "t_sigma_rdot": 2.0,
                "t_sigma_tdot": 2.0,
                "t_sigma_ndot": 2.0,
                "c_sigma_rdot": 3.0,
                "c_sigma_tdot": 3.0,
                "c_sigma_ndot": 3.0,
                "t_position_covariance_det": _num(item, "tSigmaR", 100.0) ** 6,
                "c_position_covariance_det": _num(item, "cSigmaR", 150.0) ** 6,
                "t_span": 4.0,
                "c_span": 2.0,
                "t_rcs_estimate": 2.0,
                "c_rcs_estimate": 0.5,
                "t_ecc": 0.001,
                "c_ecc": 0.01,
                "t_j2k_inc": 97.0,
                "c_j2k_inc": 86.0,
                "t_j2k_sma": 7000.0,
                "c_j2k_sma": 7100.0,
                "t_h_apo": 600.0,
                "c_h_apo": 700.0,
                "t_h_per": 550.0,
                "c_h_per": 500.0,
                "t_obs_available": _num(item, "tObsUsed", 20.0) + 4,
                "c_obs_available": _num(item, "cObsUsed", 10.0) + 3,
                "t_obs_used": _num(item, "tObsUsed", 20.0),
                "c_obs_used": _num(item, "cObsUsed", 10.0),
                "t_actual_od_span": 2.0,
                "c_actual_od_span": 2.0,
                "t_recommended_od_span": 3.0,
                "c_recommended_od_span": 3.0,
                "t_weighted_rms": 1.0,
                "c_weighted_rms": 1.2,
                "t_cd_area_over_mass": 0.02,
                "c_cd_area_over_mass": 0.04,
            }
        )
    return pd.DataFrame(rows)
