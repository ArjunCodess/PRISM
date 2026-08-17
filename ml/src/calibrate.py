from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class ProbabilityCalibrator:
    isotonic: IsotonicRegression
    kind: str = "score"

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=float)
        if self.kind == "proba":
            clipped = np.clip(values, 0.0, 1.0)
        else:
            clipped = np.clip(values, -20.0, 0.0)
        return np.asarray(self.isotonic.predict(clipped), dtype=float)


def fit_isotonic(raw_scores: np.ndarray, labels: np.ndarray) -> ProbabilityCalibrator:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.clip(raw_scores, -20.0, 0.0), labels.astype(float))
    return ProbabilityCalibrator(isotonic=iso, kind="score")


def fit_isotonic_proba(raw_scores: np.ndarray, labels: np.ndarray) -> ProbabilityCalibrator:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.clip(raw_scores, 0.0, 1.0), labels.astype(float))
    return ProbabilityCalibrator(isotonic=iso, kind="proba")
