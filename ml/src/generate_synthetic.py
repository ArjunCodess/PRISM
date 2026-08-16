from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from constants import NEGLIGIBLE_RISK, RANDOM_STATE

OBJECT_TYPES = np.array(["DEBRIS", "ROCKET BODY", "PAYLOAD", "UNKNOWN"])


def _event_rows(rng: np.random.Generator, event_id: int, story: str) -> list[dict[str, object]]:
    n_messages = int(rng.integers(6, 14))
    mission_id = int(rng.integers(1, 21))
    object_type = str(rng.choice(OBJECT_TYPES, p=[0.62, 0.18, 0.12, 0.08]))
    times = np.sort(rng.uniform(0.15, 7.2, size=n_messages))[::-1]
    if times.min() >= 2.0:
        times[-1] = rng.uniform(0.2, 0.9)
    if times.max() < 2.2:
        times[0] = rng.uniform(2.4, 6.5)

    start_risk = float(rng.uniform(-8.5, -4.8))
    miss0 = float(rng.uniform(80, 25000))
    speed = float(rng.uniform(2000, 15000))
    t_sigma = float(rng.uniform(20, 900))
    c_sigma = float(rng.uniform(40, 1800))

    if story == "low":
        start_risk = float(rng.uniform(-12.0, -7.5))
        final_risk = float(rng.uniform(-18.0, -8.0))
        miss0 = float(rng.uniform(1500, 40000))
    elif story == "escalate":
        start_risk = float(rng.uniform(-7.8, -6.2))
        final_risk = float(rng.uniform(-5.8, -3.9))
        miss0 = float(rng.uniform(80, 700))
    elif story == "deescalate":
        start_risk = float(rng.uniform(-5.4, -4.2))
        final_risk = float(rng.uniform(-9.5, -6.4))
        miss0 = float(rng.uniform(200, 1200))
    elif story == "uncertain":
        start_risk = float(rng.uniform(-6.8, -5.6))
        final_risk = float(rng.uniform(-6.4, -5.5))
        t_sigma = float(rng.uniform(400, 2200))
        c_sigma = float(rng.uniform(600, 2800))
    else:  # failure-like jump
        start_risk = float(rng.uniform(-9.5, -7.0))
        final_risk = float(rng.uniform(-5.2, -3.6))
        t_sigma = float(rng.uniform(700, 2500))

    eligible_idx = np.where(times >= 2.0)[0]
    last_eligible = int(eligible_idx[-1]) if len(eligible_idx) else 0
    n_pre = last_eligible + 1
    n_post = n_messages - n_pre
    midpoint = (start_risk + final_risk) / 2
    risks = np.empty(n_messages, dtype=float)
    risks[:n_pre] = np.linspace(start_risk, midpoint, n_pre)
    if n_post:
        risks[n_pre:] = np.linspace(midpoint, final_risk, n_post)
    else:
        risks[-1] = final_risk
    if story == "low":
        risks = np.clip(risks, NEGLIGIBLE_RISK, -7.0)
        risks[-1] = min(final_risk, -8.0)
    elif story == "failure":
        # Keep the early messages quiet so a late jump is actually hidden past T-48.
        quiet = np.linspace(start_risk, min(start_risk, -7.2), n_pre)
        risks[:n_pre] = quiet
        if n_post:
            risks[n_pre:] = np.linspace(quiet[-1], final_risk, n_post)
        risks[-1] = final_risk

    pos_r = float(rng.normal(0, miss0 * 0.3))
    pos_t = float(rng.normal(0, miss0 * 0.7))
    pos_n = float(np.sqrt(max(miss0**2 - pos_r**2 - pos_t**2, 1.0)) * rng.choice([-1, 1]))
    vel = rng.normal(0, speed / np.sqrt(3), size=3)
    vel = vel / np.linalg.norm(vel) * speed

    rows: list[dict[str, object]] = []
    for i, tca in enumerate(times):
        shrink = 0.55 + 0.45 * (tca / max(times[0], 1e-6))
        miss = max(20.0, miss0 * (0.85 + 0.3 * rng.normal()))
        t_sig = t_sigma * shrink
        c_sig = c_sigma * shrink
        obs_avail = int(rng.integers(8, 80))
        obs_used = int(max(3, obs_avail - rng.integers(0, 12)))
        rows.append(
            {
                "event_id": event_id,
                "mission_id": mission_id,
                "time_to_tca": float(tca),
                "risk": float(risks[i]),
                "max_risk_estimate": float(min(-2.5, risks[i] + rng.uniform(0.3, 1.4))),
                "max_risk_scaling": float(rng.uniform(0.4, 3.5)),
                "miss_distance": float(miss),
                "relative_speed": float(speed),
                "relative_position_r": pos_r,
                "relative_position_t": pos_t,
                "relative_position_n": pos_n,
                "relative_velocity_r": float(vel[0]),
                "relative_velocity_t": float(vel[1]),
                "relative_velocity_n": float(vel[2]),
                "azimuth": float(rng.uniform(-180, 180)),
                "elevation": float(rng.uniform(-90, 90)),
                "geocentric_latitude": float(rng.uniform(-80, 80)),
                "c_object_type": object_type,
                "F10": float(rng.uniform(70, 180)),
                "F3M": float(rng.uniform(70, 160)),
                "AP": float(rng.uniform(2, 40)),
                "SSN": float(rng.uniform(0, 180)),
                "t_sigma_r": t_sig,
                "t_sigma_t": t_sig * 1.4,
                "t_sigma_n": t_sig * 0.8,
                "c_sigma_r": c_sig,
                "c_sigma_t": c_sig * 1.6,
                "c_sigma_n": c_sig * 0.9,
                "t_sigma_rdot": t_sig / 40,
                "t_sigma_tdot": t_sig / 35,
                "t_sigma_ndot": t_sig / 45,
                "c_sigma_rdot": c_sig / 40,
                "c_sigma_tdot": c_sig / 35,
                "c_sigma_ndot": c_sig / 45,
                "t_position_covariance_det": float((t_sig**2) ** 3),
                "c_position_covariance_det": float((c_sig**2) ** 3),
                "t_span": float(rng.uniform(1.5, 12.0)),
                "c_span": float(rng.uniform(0.4, 8.0)),
                "t_rcs_estimate": float(rng.uniform(0.5, 12.0)),
                "c_rcs_estimate": float(rng.uniform(0.05, 6.0)),
                "t_ecc": float(rng.uniform(0.0001, 0.02)),
                "c_ecc": float(rng.uniform(0.0001, 0.05)),
                "t_j2k_inc": float(rng.uniform(50, 99)),
                "c_j2k_inc": float(rng.uniform(40, 100)),
                "t_j2k_sma": float(rng.uniform(6800, 7500)),
                "c_j2k_sma": float(rng.uniform(6700, 7800)),
                "t_h_apo": float(rng.uniform(400, 900)),
                "c_h_apo": float(rng.uniform(350, 1200)),
                "t_h_per": float(rng.uniform(380, 850)),
                "c_h_per": float(rng.uniform(300, 900)),
                "t_obs_available": obs_avail,
                "c_obs_available": int(rng.integers(4, 40)),
                "t_obs_used": obs_used,
                "c_obs_used": int(rng.integers(3, 30)),
                "t_actual_od_span": float(rng.uniform(0.5, 5.0)),
                "c_actual_od_span": float(rng.uniform(0.4, 6.0)),
                "t_recommended_od_span": float(rng.uniform(1.0, 6.0)),
                "c_recommended_od_span": float(rng.uniform(1.0, 7.0)),
                "t_weighted_rms": float(rng.uniform(0.6, 1.8)),
                "c_weighted_rms": float(rng.uniform(0.6, 2.4)),
                "t_cd_area_over_mass": float(rng.uniform(0.005, 0.04)),
                "c_cd_area_over_mass": float(rng.uniform(0.01, 0.12)),
                "story": story,
            }
        )
    return rows


def generate_synthetic_cdms(n_events: int = 400, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stories = np.array(["low", "escalate", "deescalate", "uncertain", "failure"])
    weights = np.array([0.62, 0.12, 0.12, 0.08, 0.06])
    rows: list[dict[str, object]] = []
    for event_id in range(n_events):
        story = str(rng.choice(stories, p=weights))
        rows.extend(_event_rows(rng, event_id, story))
    return pd.DataFrame(rows)


def write_synthetic(path: Path, n_events: int = 400) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_synthetic_cdms(n_events=n_events)
    frame.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    write_synthetic(root / "data" / "interim" / "synthetic_cdms.csv")
