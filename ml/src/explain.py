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
                label=FEATURE_DICTIONARY.get(name, name.replace("_", " ")),
            )
        )
    return base, factors


def explanation_text(factors: list[Factor]) -> str:
    higher = [item for item in factors if item.direction == "higher"][:3]
    lower = [item for item in factors if item.direction == "lower"][:3]
    parts: list[str] = []
    if higher:
        reasons = ", ".join(item.label for item in higher)
        parts.append(f"the forecast moved toward higher risk because {reasons}")
    if lower:
        reasons = ", ".join(item.label for item in lower)
        parts.append(f"it moved toward lower risk because {reasons}")
    if not parts:
        return "the model did not find a dominant local driver for this forecast."
    text = "; ".join(parts)
    return text[0].upper() + text[1:] + "."


FEATURE_GROUPS = {
    "current estimated risk": ("risk", "max_risk"),
    "risk trend": ("risk_change", "risk_slope", "risk_delta", "risk_rising", "risk_std"),
    "miss-distance geometry": ("miss_distance", "normalized_separation", "relative_position"),
    "relative speed": ("relative_speed", "relative_velocity"),
    "position uncertainty": ("sigma", "cov_det", "log_t_cov", "log_c_cov"),
    "observation quality": ("obs_", "od_span", "weighted_rms"),
    "orbit properties": ("ecc", "j2k", "h_apo", "h_per", "span", "rcs", "area_over"),
    "space weather": ("F10", "F3M", "AP", "SSN"),
}


def grouped_importance(feature_names: list[str], scores: dict[str, float]) -> list[dict[str, float | str]]:
    totals = {group: 0.0 for group in FEATURE_GROUPS}
    totals["other"] = 0.0
    for key, score in scores.items():
        if key in feature_names:
            name = key
        elif key.startswith("f") and key[1:].isdigit():
            idx = int(key[1:])
            name = feature_names[idx] if idx < len(feature_names) else key
        else:
            name = key
        placed = False
        for group, needles in FEATURE_GROUPS.items():
            if any(needle in name for needle in needles):
                totals[group] += float(score)
                placed = True
                break
        if not placed:
            totals["other"] += float(score)
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    return [{"group": name, "gain": value} for name, value in ranked if value > 0]
