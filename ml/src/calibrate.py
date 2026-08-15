from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression

from constants import HIGH_RISK_THRESHOLD


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
