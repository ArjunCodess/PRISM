from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from build_events import build_event_histories
from calibrate import conformal_bounds
from explain import shap_explainer
from export_demo_cases import predict_event
from selected_policy import decide_policy_abstention, load_floor_hurdle, predict_floor_hurdle
from train_regressor import TrainedRegressor

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "ml" / "artifacts"


class PrismModel:
    def __init__(self) -> None:
        policy = json.loads((ARTIFACTS / "selected_policy.json").read_text(encoding="utf-8"))
        if policy["name"] != "floorHurdle":
            raise RuntimeError(f"unsupported selected policy {policy['name']}")
        schema = json.loads((ARTIFACTS / "feature_schema.json").read_text(encoding="utf-8"))
        self.feature_names: list[str] = list(schema["features"])
        self.policy = policy
        self.bundle = load_floor_hurdle(ARTIFACTS, self.feature_names)
        self.trained: TrainedRegressor = self.bundle["residual"]
        self.calibrator = joblib.load(ARTIFACTS / "exhibit_calibrator.joblib")["calibrator"]
        conformal = json.loads((ARTIFACTS / "exhibit_conformal.json").read_text(encoding="utf-8"))
        self.q50 = float(conformal["q50"])
        self.q90 = float(conformal["q90"])
        self.abstention_rule = str(policy["abstention"])
        background = pd.DataFrame(
            np.zeros((20, len(self.trained.feature_names))),
            columns=self.trained.feature_names,
        )
        self.explainer = shap_explainer(self.trained, background)

    def predict_frame(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        point, floor_proba = predict_floor_hurdle(self.bundle, frame)
        lo90, hi90 = conformal_bounds(point, self.q90)
        lo50, hi50 = conformal_bounds(point, self.q50)
        return {
            "point": point,
            "floorProba": floor_proba,
            "lo90": lo90,
            "hi90": hi90,
            "lo50": lo50,
            "hi50": hi50,
        }

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
        row_frame = features.iloc[[0]]
        series = row_frame.iloc[0]
        scored = self.predict_frame(row_frame)
        point = float(scored["point"][0])
        floor_called = float(scored["floorProba"][0]) >= self.bundle["threshold"]
        current_risk = float(series["risk"]) if pd.notna(series.get("risk")) else float("nan")
        miss_distance = (
            float(series["miss_distance"]) if pd.notna(series.get("miss_distance")) else float("nan")
        )
        decision = decide_policy_abstention(
            rule=self.abstention_rule,
            current_risk=current_risk,
            miss_distance=miss_distance,
            lo90=float(scored["lo90"][0]),
            hi90=float(scored["hi90"][0]),
        )
        return predict_event(
            trained=self.trained,
            calibrator=self.calibrator,
            explainer=self.explainer,
            row=series,
            messages=messages,
            event_id=event_id,
            point=point,
            interval90=(float(scored["lo90"][0]), float(scored["hi90"][0])),
            interval50=(float(scored["lo50"][0]), float(scored["hi50"][0])),
            decision=decision,
            floor_called=floor_called,
            interval_kind="conformal",
        )


def build_feature_table(events: list) -> pd.DataFrame:
    from features import build_feature_table as _build

    return _build(events)


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
