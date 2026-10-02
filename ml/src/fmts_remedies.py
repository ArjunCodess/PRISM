"""Established remedies for an observed archive atom, without latent-semantics claims."""

from __future__ import annotations

import numpy as np
import pandas as pd
from calibrate import split_conformal_quantile
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def atom_mask(y, floor=-30.0, atol=1e-6):
    values = np.asarray(y, dtype=float)
    if not np.isfinite(values).all() or np.any(values < floor - atol):
        raise ValueError("target outside documented observed support")
    return np.isclose(values, floor, atol=atol, rtol=0)


def categories(y, floor=-30.0, high=-6.0):
    values = np.asarray(y, dtype=float)
    return np.where(atom_mask(values, floor), 0, np.where(values >= high, 2, 1))


def exact_esa(y, pred, high=-6.0):
    # Uriot: cap forecasts at -6.001 when the current forecast is below -6.
    # The convention is shared with evaluate.clip_for_esa and inspected by tests.
    from evaluate import clip_for_esa

    y = np.asarray(y)
    pred = clip_for_esa(np.asarray(pred))
    actual, called = y >= high, pred >= high
    tp = int(np.sum(actual & called))
    fp = int(np.sum(~actual & called))
    fn = int(np.sum(actual & ~called))
    f2 = 5 * tp / (5 * tp + 4 * fn + fp) if actual.any() else None
    mse = float(np.mean((y[actual] - pred[actual]) ** 2)) if actual.any() else None
    return {
        "esa_loss": mse / f2 if f2 else None,
        "esa_undefined": "no positives"
        if not actual.any()
        else "F2=0 (infinite loss)"
        if not f2
        else None,
        "f2": f2,
        "high_mse": mse,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": tp / (tp + fn) if tp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
    }


def decision_cost(y, pred, cfg, review=None):
    y, pred = np.asarray(y), np.asarray(pred)
    review = np.zeros(len(y), dtype=bool) if review is None else np.asarray(review)
    positive, called = y >= cfg["high_threshold"], pred >= cfg["high_threshold"]
    costs = np.where(
        positive & ~called,
        cfg["false_negative_cost"],
        np.where(~positive & called, cfg["false_positive_cost"], 0.0),
    )
    return np.where(review, cfg["review_cost"], costs)


def point_report(y, pred, cfg):
    y, pred = np.asarray(y), np.asarray(pred)
    floor = atom_mask(y, cfg["floor"], cfg["floor_atol"])
    error = np.abs(y - pred)
    contributions = {
        name: float(np.sum(error[mask]) / len(y))
        for name, mask in (("floor", floor), ("nonfloor", ~floor))
    }
    return {
        **exact_esa(y, pred, cfg["high_threshold"]),
        "n": len(y),
        "n_floor": int(floor.sum()),
        "mae": float(error.mean()),
        "floor_mae": float(error[floor].mean()) if floor.any() else None,
        "nonfloor_mae": float(error[~floor].mean()) if (~floor).any() else None,
        "mae_contributions": contributions,
        "decision_cost": float(decision_cost(y, pred, cfg).mean()),
    }


def selection_key(y, pred, cfg):
    row = point_report(y, pred, cfg)
    return (
        float("inf") if row["esa_loss"] is None else row["esa_loss"],
        row["decision_cost"],
        row["nonfloor_mae"] or 0.0,
    )


def mondrian_quantiles(y, pred, cfg):
    group = categories(y, cfg["floor"], cfg["high_threshold"])
    scores = np.abs(np.asarray(y) - pred)
    return {
        str(g): {
            "n": int(np.sum(group == g)),
            "q": split_conformal_quantile(scores[group == g], cfg["alpha"]),
        }
        for g in range(3)
    }


def prediction_set(pred, quantiles, cfg):
    """Candidate-category inversion, evaluated without seeing the future label."""
    pred = np.asarray(pred)
    floor, high = cfg["floor"], cfg["high_threshold"]
    atom = np.abs(floor - pred) <= quantiles["0"]["q"]
    # nextafter represents the open endpoints of the non-floor low-risk domain.
    low_lo = np.maximum(np.nextafter(floor, np.inf), pred - quantiles["1"]["q"])
    low_hi = np.minimum(np.nextafter(high, -np.inf), pred + quantiles["1"]["q"])
    high_lo = np.maximum(high, pred - quantiles["2"]["q"])
    high_hi = np.minimum(0.0, pred + quantiles["2"]["q"])
    low_valid, high_valid = low_lo <= low_hi, high_lo <= high_hi
    width = np.where(low_valid, low_hi - low_lo, 0) + np.where(high_valid, high_hi - high_lo, 0)
    low_possible = atom | low_valid
    empty = ~low_possible & ~high_valid
    review = empty | (low_possible & high_valid)
    return {
        "atom": atom,
        "low_lo": low_lo,
        "low_hi": low_hi,
        "high_lo": high_lo,
        "high_hi": high_hi,
        "width": width,
        "review": review,
        "empty": empty,
    }


def set_covered(y, sets, cfg):
    group = categories(y, cfg["floor"], cfg["high_threshold"])
    y = np.asarray(y)
    return np.where(
        group == 0,
        sets["atom"],
        np.where(
            group == 1,
            (y >= sets["low_lo"]) & (y <= sets["low_hi"]),
            (y >= sets["high_lo"]) & (y <= sets["high_hi"]),
        ),
    )


def paired_intervals(y, predictions, cfg):
    rng = np.random.default_rng(cfg["seed"])
    n = len(y)
    baseline = predictions["persistence"]
    result = {}
    # Reuse precisely the same event resamples for all policies and metrics.
    draws = [rng.integers(0, n, n) for _ in range(cfg["bootstrap"])]
    for name, pred in predictions.items():
        differences = {"mae": [], "decision_cost": [], "esa_loss": []}
        for ix in draws:
            yy, pp, bb = y[ix], pred[ix], baseline[ix]
            differences["mae"].append(float(np.mean(np.abs(yy - pp) - np.abs(yy - bb))))
            differences["decision_cost"].append(
                float(np.mean(decision_cost(yy, pp, cfg) - decision_cost(yy, bb, cfg)))
            )
            a, b = exact_esa(yy, pp), exact_esa(yy, bb)
            if a["esa_loss"] is not None and b["esa_loss"] is not None:
                differences["esa_loss"].append(a["esa_loss"] - b["esa_loss"])
        result[name] = {
            metric: {
                "ci95": np.quantile(values, [0.025, 0.975]).tolist() if values else None,
                "valid_draws": len(values),
                "undefined_draws": cfg["bootstrap"] - len(values),
            }
            for metric, values in differences.items()
        }
    return result


class GaussianTobit:
    """Known left clipping only; no ESA latent censoring inference."""

    def fit(self, x, y, floor=-30.0):
        self.floor = floor
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = StandardScaler()
        z = self.scaler.fit_transform(self.imputer.fit_transform(x))
        z = np.column_stack([np.ones(len(z)), z])
        censored = atom_mask(y, floor)

        def objective(theta):
            mu, sigma = z @ theta[:-1], np.exp(theta[-1])
            likelihood = np.where(
                censored,
                log_ndtr((floor - mu) / sigma),
                -0.5 * ((y - mu) / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2 * np.pi),
            )
            return -float(likelihood.mean()) + 1e-5 * float(np.sum(theta[1:-1] ** 2))

        initial = np.r_[np.linalg.lstsq(z, y, rcond=None)[0], np.log(max(np.std(y), 1))]
        result = minimize(
            objective, initial, method="L-BFGS-B", bounds=[(None, None)] * (z.shape[1]) + [(-5, 5)]
        )
        self.converged = bool(result.success)
        self.message = str(result.message)
        self.theta = result.x
        return self

    def predict(self, x):
        z = self.scaler.transform(self.imputer.transform(x))
        mu = np.column_stack([np.ones(len(z)), z]) @ self.theta[:-1]
        sigma = np.exp(self.theta[-1])
        a = (self.floor - mu) / sigma
        return (
            self.floor * ndtr(a) + mu * ndtr(-a) + sigma * np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
        )


def ordered_event_split(frame: pd.DataFrame):
    if frame.groupby("event_id")["event_time"].nunique().max() != 1:
        raise ValueError("an event has inconsistent timestamps")
    events = (
        frame[["event_id", "event_time"]].drop_duplicates().sort_values(["event_time", "event_id"])
    )
    # Keep tied timestamps together rather than leaking simultaneous groups.
    times = np.sort(events.event_time.unique())
    cuts = [times[int(len(times) * fraction)] for fraction in (0.45, 0.65, 0.8)]
    masks = [
        events.event_time < cuts[0],
        (events.event_time >= cuts[0]) & (events.event_time < cuts[1]),
        (events.event_time >= cuts[1]) & (events.event_time < cuts[2]),
        events.event_time >= cuts[2],
    ]
    return {
        name: events.loc[mask, "event_id"].astype(int).tolist()
        for name, mask in zip(("train", "validation", "calibration", "test"), masks)
    }
