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
    x = row[trained.feature_names].to_frame().T
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
