from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from constants import FEATURE_DICTIONARY
from train_regressor import TrainedRegressor


@dataclass
class Factor:
    feature: str
    direction: str
    contribution: float
    label: str


_BASE_PHRASES = {
    "risk": "the reported collision chance",
    "max_risk_estimate": "the most pessimistic chance in the message",
    "max_risk_scaling": "how stretched the pessimistic chance is",
    "miss_distance": "the predicted miss distance",
    "relative_speed": "the closing speed",
    "normalized_separation": "how wide the miss is compared with the uncertainty",
    "t_obs_used": "how many satellite observations were used",
    "c_obs_used": "how many observations of the other object were used",
    "t_position_covariance_det": "how uncertain the satellite's position is",
    "c_position_covariance_det": "how uncertain the other object's position is",
    "t_sigma_r": "the satellite's radial position uncertainty",
    "t_sigma_t": "the satellite's along-track position uncertainty",
    "t_sigma_n": "the satellite's cross-track position uncertainty",
    "c_sigma_r": "the other object's radial position uncertainty",
    "c_sigma_t": "the other object's along-track position uncertainty",
    "c_sigma_n": "the other object's cross-track position uncertainty",
    "hours_before_cutoff": "hours left before the 48-hour line",
    "n_messages": "the number of early messages",
    "derived_miss_distance": "the miss distance implied by the geometry",
    "derived_relative_speed": "the closing speed implied by the velocities",
    "miss_distance_residual": "the gap between the stated miss distance and the geometry",
}


def human_label(name: str) -> str:
    if name in FEATURE_DICTIONARY:
        return FEATURE_DICTIONARY[name]
    suffixes = (
        ("_delta_last3", "last-three change in {}"),
        ("_delta_last2", "last-two change in {}"),
        ("_slope", "trend in {}"),
        ("_change", "change in {}"),
        ("_rising", "whether {} rose"),
        ("_std", "how jumpy {} is"),
        ("_mean", "average {}"),
        ("_first", "earliest {}"),
        ("_last", "latest {}"),
        ("_min", "smallest {}"),
        ("_max", "largest {}"),
    )
    for suffix, template in suffixes:
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            phrase = _BASE_PHRASES.get(base, base.replace("_", " "))
            return template.format(phrase)
    return _BASE_PHRASES.get(name, name.replace("_", " "))


def shap_explainer(trained: TrainedRegressor, background: pd.DataFrame) -> shap.Explainer:
    _ = background
    return shap.TreeExplainer(trained.model)


def local_factors(
    trained: TrainedRegressor,
    explainer: shap.Explainer,
    row: pd.Series,
    top_k: int = 6,
) -> tuple[float, list[Factor]]:
    x = row[trained.feature_names].to_frame().T.apply(pd.to_numeric, errors="coerce")
    values = np.asarray(explainer.shap_values(x), dtype=float).reshape(-1)
    base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    order = np.argsort(np.abs(values))[::-1][:top_k]
    factors: list[Factor] = []
    for idx in order:
        name = trained.feature_names[int(idx)]
        contribution = float(values[int(idx)])
        factors.append(
            Factor(
                feature=name,
                direction="higher" if contribution > 0 else "lower",
                contribution=contribution,
                label=human_label(name),
            )
        )
    return base, factors


def _join_reasons(items: list[Factor]) -> str:
    labels = [item.label for item in items]
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def explanation_text(factors: list[Factor]) -> str:
    higher = [item for item in factors if item.direction == "higher"][:2]
    lower = [item for item in factors if item.direction == "lower"][:2]
    parts: list[str] = []
    if higher:
        parts.append(f"More worrying: {_join_reasons(higher)}.")
    if lower:
        parts.append(f"Less worrying: {_join_reasons(lower)}.")
    if not parts:
        return "No dominant reason for this guess."
    return " ".join(parts)


def feature_group(name: str) -> str:
    lowered = name.lower()
    if lowered in {"f10", "f3m", "ap", "ssn"}:
        return "space weather"
    if any(token in lowered for token in ("obs_", "od_span", "weighted_rms", "hours_since", "hours_before")):
        return "how complete the tracking is"
    if any(token in lowered for token in ("sigma", "cov_det", "log_t_cov", "log_c_cov")):
        return "how uncertain the positions still are"
    if any(token in lowered for token in ("relative_speed", "relative_velocity")):
        return "closing speed"
    if any(
        token in lowered
        for token in ("miss_distance", "normalized_separation", "relative_position", "derived_miss")
    ):
        return "how close they pass"
    if name in {"risk", "max_risk_estimate", "max_risk_scaling"}:
        return "today's reported risk"
    if name.startswith("risk_") or name.startswith("max_risk"):
        return "whether risk is climbing or falling"
    if any(token in lowered for token in ("ecc", "j2k", "h_apo", "h_per", "rcs", "area_over", "t_span", "c_span")):
        return "orbit shape and size"
    return "other"


def grouped_importance(feature_names: list[str], scores: dict[str, float]) -> list[dict[str, float | str]]:
    totals: dict[str, float] = {}
    for key, score in scores.items():
        if key in feature_names:
            name = key
        elif key.startswith("f") and key[1:].isdigit():
            idx = int(key[1:])
            name = feature_names[idx] if idx < len(feature_names) else key
        else:
            name = key
        group = feature_group(name)
        totals[group] = totals.get(group, 0.0) + float(score)
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    return [{"group": name, "gain": value} for name, value in ranked if value > 0]
