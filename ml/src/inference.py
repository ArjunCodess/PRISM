from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from build_events import build_event_histories
from constants import MODEL_VERSION
from explain import shap_explainer
from export_demo_cases import predict_event
from features import build_feature_table
from train_regressor import TrainedRegressor

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
        self.trained = TrainedRegressor(model=model, feature_names=self.feature_names, kind="xgboost")
        background = pd.DataFrame(
            np.zeros((20, len(self.feature_names))), columns=self.feature_names
        )
        self.explainer = shap_explainer(self.trained, background)
        self.model_version = MODEL_VERSION

    def predict_messages(self, event_id: str, messages: list[dict[str, Any]], cutoff_hours: int = 48) -> dict[str, Any]:
        cutoff_days = cutoff_hours / 24.0
        if any(float(item["timeToTcaDays"]) < cutoff_days - 1e-9 for item in messages):
            raise ValueError("post-cutoff messages are not allowed")
        frame = _messages_to_frame(event_id, messages)
        events = build_event_histories(frame)
        if not events:
            raise ValueError("no eligible pre-cutoff messages")
        features = build_feature_table(events)
        row = features.iloc[0]
        aligned = pd.DataFrame([{name: row[name] if name in row else np.nan for name in self.feature_names}])
        series = aligned.iloc[0]
        ens = np.array([model.predict(aligned)[0] for model in self.ensemble])
        return predict_event(
            trained=self.trained,
            ensemble_preds=ens,
            calibrator=self.calibrator,
            explainer=self.explainer,
            row=series,
            messages=messages,
            event_id=event_id,
        )


def _messages_to_frame(event_id: str, messages: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in messages:
        rows.append(
            {
                "event_id": 0,
                "mission_id": 0,
                "time_to_tca": float(item["timeToTcaDays"]),
                "risk": float(item["riskLog10"]),
                "max_risk_estimate": float(item.get("maxRiskEstimate", item["riskLog10"] + 0.4)),
                "max_risk_scaling": float(item.get("maxRiskScaling", 1.0)),
                "miss_distance": float(item["missDistanceM"]),
                "relative_speed": float(item["relativeSpeedMps"]),
                "relative_position_r": float(item.get("relativePositionR", item["missDistanceM"] * 0.2)),
                "relative_position_t": float(item.get("relativePositionT", item["missDistanceM"] * 0.8)),
                "relative_position_n": float(item.get("relativePositionN", 10.0)),
                "relative_velocity_r": float(item.get("relativeVelocityR", 0.0)),
                "relative_velocity_t": float(item.get("relativeVelocityT", item["relativeSpeedMps"])),
                "relative_velocity_n": float(item.get("relativeVelocityN", 0.0)),
                "azimuth": float(item.get("azimuth", 0.0)),
                "elevation": float(item.get("elevation", 0.0)),
                "geocentric_latitude": float(item.get("geocentricLatitude", 0.0)),
                "c_object_type": str(item.get("cObjectType", "UNKNOWN")),
                "F10": float(item.get("f10", 100.0)),
                "F3M": float(item.get("f3m", 100.0)),
                "AP": float(item.get("ap", 10.0)),
                "SSN": float(item.get("ssn", 50.0)),
                "t_sigma_r": float(item.get("tSigmaR", 100.0)),
                "t_sigma_t": float(item.get("tSigmaT", 140.0)),
                "t_sigma_n": float(item.get("tSigmaN", 80.0)),
                "c_sigma_r": float(item.get("cSigmaR", 150.0)),
                "c_sigma_t": float(item.get("cSigmaT", 200.0)),
                "c_sigma_n": float(item.get("cSigmaN", 90.0)),
                "t_sigma_rdot": 2.0,
                "t_sigma_tdot": 2.0,
                "t_sigma_ndot": 2.0,
                "c_sigma_rdot": 3.0,
                "c_sigma_tdot": 3.0,
                "c_sigma_ndot": 3.0,
                "t_position_covariance_det": float(item.get("tSigmaR", 100.0) ** 6),
                "c_position_covariance_det": float(item.get("cSigmaR", 150.0) ** 6),
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
                "t_obs_available": float(item.get("tObsUsed", 20) + 4),
                "c_obs_available": float(item.get("cObsUsed", 10) + 3),
                "t_obs_used": float(item.get("tObsUsed", 20)),
                "c_obs_used": float(item.get("cObsUsed", 10)),
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
