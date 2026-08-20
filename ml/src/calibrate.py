from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from constants import HIGH_RISK_THRESHOLD
from sklearn.isotonic import IsotonicRegression


def split_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    values = np.sort(np.abs(np.asarray(scores, dtype=float)))
    n = int(values.size)
    if n == 0:
        return float("inf")
    rank = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
    if rank >= n:
        return float("inf")
    return float(values[max(rank, 0)])


def conformal_bounds(pred: np.ndarray, q_hat: float) -> tuple[np.ndarray, np.ndarray]:
    point = np.asarray(pred, dtype=float)
    return point - q_hat, point + q_hat


def interval_report(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=float)
    low = np.asarray(lo, dtype=float)
    high = np.asarray(hi, dtype=float)
    covered = (y >= low) & (y <= high)
    widths = high - low
    if y.size == 0:
        return {
            "n": 0,
            "nCovered": 0,
            "coverage": 0.0,
            "meanWidth": 0.0,
            "medianWidth": 0.0,
            "q25Width": 0.0,
            "q75Width": 0.0,
        }
    q25, median, q75 = np.quantile(widths, [0.25, 0.5, 0.75])
    return {
        "n": int(y.size),
        "nCovered": int(np.sum(covered)),
        "coverage": float(np.mean(covered)),
        "meanWidth": float(np.mean(widths)),
        "medianWidth": float(median),
        "q25Width": float(q25),
        "q75Width": float(q75),
    }


def fit_absolute_conformal(
    y_cal: np.ndarray, pred_cal: np.ndarray, alphas: tuple[float, ...] = (0.5, 0.1)
) -> dict[float, float]:
    scores = np.abs(np.asarray(y_cal, dtype=float) - np.asarray(pred_cal, dtype=float))
    return {float(alpha): split_conformal_quantile(scores, float(alpha)) for alpha in alphas}


@dataclass
class ProbabilityCalibrator:
    isotonic: IsotonicRegression
    threshold: float = HIGH_RISK_THRESHOLD

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        clipped = np.clip(scores, -20.0, 0.0)
        return np.asarray(self.isotonic.predict(clipped), dtype=float)


def fit_isotonic(raw_scores: np.ndarray, labels: np.ndarray) -> ProbabilityCalibrator:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.clip(raw_scores, -20.0, 0.0), labels.astype(float))
    return ProbabilityCalibrator(isotonic=iso)
