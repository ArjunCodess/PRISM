"""Run the review protocol without changing the deployed artifact bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from build_events import build_event_histories
from calibrate import split_conformal_quantile
from features import build_feature_table
from fmts_remedies import (
    GaussianTobit,
    atom_mask,
    categories,
    decision_cost,
    mondrian_quantiles,
    ordered_event_split,
    paired_intervals,
    point_report,
    prediction_set,
    selection_key,
    set_covered,
)
from ingest import (
    attach_official_test_labels,
    load_esa_training,
    load_official_test,
    load_official_test_labels,
    realistic_training_events,
)
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from split import grouped_splits
from train_regressor import numeric_columns
from validate import validate_cdm_frame
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ml/artifacts/fmts_review"


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def save(path, payload):
    path.write_text(json.dumps(clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_split(frame, manifest):
    all_ids = [event for ids in manifest.values() for event in ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("event leakage between splits")
    if set(all_ids) != set(frame.event_id):
        raise ValueError("manifest must partition every eligible event")


def interval_summary(y, pred, cal_y, cal_pred, cfg):
    q = split_conformal_quantile(cal_y - cal_pred, cfg["alpha"])
    quantiles = mondrian_quantiles(cal_y, cal_pred, cfg)
    sets = prediction_set(pred, quantiles, cfg)
    marginal = np.abs(y - pred) <= q
    conditional = set_covered(y, sets, cfg)
    group = categories(y, cfg["floor"], cfg["high_threshold"])
    summary = {
        "marginal_radius": q,
        "category_quantiles": quantiles,
        "marginal_width": 2 * q,
        "set_mean_width": float(sets["width"].mean()),
        "empty_sets": int(sets["empty"].sum()),
        "review_fraction": float(sets["review"].mean()),
        "accepted_false_reassurance": int(
            np.sum(~sets["review"] & (pred < cfg["high_threshold"]) & (y >= cfg["high_threshold"]))
        ),
        "review_decision_cost": float(decision_cost(y, pred, cfg, sets["review"]).mean()),
        "coverage": {},
    }
    from scipy.stats import beta

    for label, mask in [("all", np.ones(len(y), bool))] + [
        (name, group == g) for g, name in enumerate(("floor", "nonfloor_low", "high"))
    ]:
        summary["coverage"][label] = {"n": int(mask.sum())}
        for name, covered in (("marginal", marginal), ("mondrian", conditional)):
            n, k = int(mask.sum()), int(covered[mask].sum())
            interval = (
                [
                    float(beta.ppf(0.025, k, n - k + 1)) if k else 0.0,
                    float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0,
                ]
                if n
                else None
            )
            summary["coverage"][label][name] = {
                "covered": k,
                "coverage": k / n if n else None,
                "ci95": interval,
            }
    return summary, sets, q


def fit_systems(parts, cfg, known_clipping=False):
    train, val = parts["train"], parts["validation"]
    columns = numeric_columns(train)
    columns = [c for c in columns if c not in ("event_time", "mission_id", "target_time_to_tca")]
    x = {name: frame.reindex(columns=columns) for name, frame in parts.items()}
    reg = {**cfg["regressor"], "random_state": cfg["seed"], "n_jobs": 4}
    direct = XGBRegressor(**reg).fit(x["train"], train.y)
    residual = XGBRegressor(**reg).fit(x["train"], train.y - train.risk)
    nonfloor = ~atom_mask(train.y.to_numpy())
    conditional = XGBRegressor(**reg).fit(
        x["train"].loc[nonfloor], (train.y - train.risk).loc[nonfloor]
    )
    label = (~nonfloor).astype(int)
    # Reproduce the old class-weighted hurdle before correcting probability calibration.
    classifier = XGBClassifier(
        **cfg["classifier"],
        random_state=cfg["seed"],
        n_jobs=4,
        eval_metric="logloss",
        scale_pos_weight=nonfloor.sum() / label.sum(),
    )
    classifier.fit(x["train"], label)
    raw = {name: classifier.predict_proba(xx)[:, 1] for name, xx in x.items()}
    sigmoid = LogisticRegression(C=1e6, random_state=cfg["seed"])
    sigmoid.fit(
        logit(np.clip(raw["validation"], 1e-6, 1 - 1e-6)).reshape(-1, 1),
        atom_mask(val.y.to_numpy()).astype(int),
    )
    probability = {
        name: sigmoid.predict_proba(logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1))[:, 1]
        for name, p in raw.items()
    }
    nonfloor_pred = {
        name: np.clip(frame.risk.to_numpy() + conditional.predict(x[name]), -30, 0)
        for name, frame in parts.items()
    }
    predictions = {}
    for name, frame in parts.items():
        risk = frame.risk.to_numpy()
        mean = probability[name] * cfg["floor"] + (1 - probability[name]) * nonfloor_pred[name]
        predictions[name] = {
            "persistence": risk,
            "direct": direct.predict(x[name]),
            "residual": risk + residual.predict(x[name]),
            "mixture_mean": mean,
        }
    old_threshold = min(
        np.linspace(0.05, 0.95, 19),
        key=lambda t: float(
            np.mean(
                np.abs(
                    val.y.to_numpy()
                    - np.where(raw["validation"] >= t, -30, nonfloor_pred["validation"])
                )
            )
        ),
    )
    candidates = {}
    for threshold in cfg["hurdle_thresholds"]:
        for weight in cfg["blend_weights"]:
            for guard in (False, True):
                point = np.where(
                    probability["validation"] >= threshold, -30, nonfloor_pred["validation"]
                )
                point = weight * point + (1 - weight) * val.risk.to_numpy()
                if guard:
                    point = np.where(val.risk >= -6, val.risk, point)
                candidates[(threshold, weight, guard)] = point
    chosen = min(candidates, key=lambda k: selection_key(val.y.to_numpy(), candidates[k], cfg))
    for name, frame in parts.items():
        predictions[name]["old_hurdle"] = np.where(
            raw[name] >= old_threshold, -30, nonfloor_pred[name]
        )
        threshold, weight, guard = chosen
        point = np.where(probability[name] >= threshold, -30, nonfloor_pred[name])
        point = weight * point + (1 - weight) * frame.risk.to_numpy()
        if guard:
            point = np.where(frame.risk >= -6, frame.risk, point)
        predictions[name]["decision_hurdle"] = point
    selection = {
        "old_mae_threshold": old_threshold,
        "decision_parameters": chosen,
        "candidates": [
            {"parameters": key, "validation": point_report(val.y.to_numpy(), pred, cfg)}
            for key, pred in candidates.items()
        ],
        "features": columns,
        "parameters": reg,
        "sigmoid_coefficient": sigmoid.coef_.tolist(),
        "sigmoid_intercept": sigmoid.intercept_.tolist(),
    }
    if known_clipping:
        tobit = GaussianTobit().fit(x["train"], train.y.to_numpy())
        selection["tobit"] = {"converged": tobit.converged, "message": tobit.message}
        for name in parts:
            predictions[name]["known_clipping_tobit"] = tobit.predict(x[name])
    return predictions, raw, probability, selection


def run_dataset(label, frame, manifest, cfg, transfer=None, known_clipping=False):
    validate_split(frame, manifest)
    parts = {name: frame[frame.event_id.isin(ids)].copy() for name, ids in manifest.items()}
    if transfer is not None:
        parts["official"] = transfer
    predictions, raw, probability, selection = fit_systems(parts, cfg, known_clipping)
    report = {
        "split": manifest,
        "split_integrity": True,
        "selection": selection,
        "split_counts": {name: len(value) for name, value in parts.items()},
        "evaluations": {},
    }
    cal_y = parts["calibration"].y.to_numpy()
    for name in ("test", "official"):
        if name not in parts:
            continue
        y = parts[name].y.to_numpy()
        board = {model: point_report(y, pred, cfg) for model, pred in predictions[name].items()}
        intervals, arrays = (
            {},
            {
                "event_id": parts[name].event_id.to_numpy(),
                "y": y,
                "raw_floor_probability": raw[name],
                "calibrated_floor_probability": probability[name],
            },
        )
        for model, pred in predictions[name].items():
            intervals[model], sets, q = interval_summary(
                y, pred, cal_y, predictions["calibration"][model], cfg
            )
            arrays[model] = pred
            arrays[model + "_marginal_covered"] = np.abs(y - pred) <= q
            for key, value in sets.items():
                arrays[model + "_set_" + key] = value
        np.savez_compressed(OUT / f"{label}_{name}_predictions.npz", **arrays)
        report["evaluations"][name] = {
            "models": board,
            "intervals": intervals,
            "paired_difference_model_minus_persistence": paired_intervals(
                y, predictions[name], cfg
            ),
            "floor_brier_raw": brier_score_loss(atom_mask(y).astype(int), raw[name]),
            "floor_brier_calibrated": brier_score_loss(atom_mask(y).astype(int), probability[name]),
        }
    save(OUT / f"{label}.json", report)
    print(label, "selected", selection["decision_parameters"], flush=True)
    return report


def simulation(seed, rate, mechanism, drift, cfg):
    rng = np.random.default_rng(seed)
    n = cfg["simulation"]["n_events"]
    time = np.arange(n)
    a, b = rng.normal(size=(2, n))
    shift = np.where((time >= int(n * 0.65)) & drift, 3.0, 0.0)
    latent = -12 + 5 * a + 2 * b + rng.normal(0, 2, n) + shift
    if mechanism == "clipping":
        from scipy.stats import norm

        latent = latent + (-30 - norm.ppf(rate) * np.std(latent) - np.mean(latent))
        y = np.maximum(-30, np.minimum(0, latent))
    else:
        from scipy.optimize import brentq

        intercept = brentq(lambda v: expit(v - 1.5 * a).mean() - rate, -20, 20)
        at_atom = rng.random(n) < expit(intercept - 1.5 * a - shift / 2)
        y = np.where(at_atom, -30, np.clip(latent, -29.99, 0))
    # No target or atom indicator is available to the predictor.
    history = np.clip(-12 + 5 * a[:, None] + 2 * b[:, None] + rng.normal(0, 3, (n, 4)), -30, 0)
    frame = pd.DataFrame(
        {
            "event_id": time,
            "event_time": time,
            "y": y,
            "risk": history[:, -1],
            "risk_first": history[:, 0],
            "risk_mean": history.mean(axis=1),
            "x": a,
            "z": b,
        }
    )
    # Split integrity is also checked on the full four-observation representation.
    observations = pd.DataFrame({"event_id": np.repeat(time, 4), "event_time": np.repeat(time, 4)})
    manifest = ordered_event_split(observations)
    return frame, manifest


def generate_outputs():
    esa = json.loads((OUT / "esa_seed42.json").read_text())
    rows = esa["evaluations"]["test"]["models"]
    names = list(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    floor = [rows[n]["mae_contributions"]["floor"] for n in names]
    nonfloor = [rows[n]["mae_contributions"]["nonfloor"] for n in names]
    axes[0].barh(names, floor, label="Recorded -30 contribution")
    axes[0].barh(names, nonfloor, left=floor, label="Non-floor contribution")
    axes[0].set_xlabel("Contribution to total MAE, log10(Pc)")
    axes[0].legend(fontsize=8)
    axes[1].barh(names, [rows[n]["high_mse"] for n in names])
    axes[1].set_xlabel("ESA high-risk MSE after challenge clipping")
    axes[1].set_title("Decision tail shown separately")
    fig.savefig(ROOT / "docs/figures/fmts-remedy-decomposition.png", dpi=180)
    plt.close(fig)
    text = [
        "% Generated by ml/src/fmts_review.py; do not hand-edit.",
        r"\begin{tabular}{lrrrrrr}\hline",
        r"Model & $L$ & Recall & Precision & Floor & Non-floor & MAE\\\hline",
    ]
    for name, row in rows.items():
        loss = r"$\infty$" if row["esa_loss"] is None else f"{row['esa_loss']:.3f}"
        precision = "--" if row["precision"] is None else f"{row['precision']:.3f}"
        text.append(
            f"{name.replace('_', ' ')} & {loss} & {row['recall']:.3f} & "
            f"{precision} & {row['floor_mae']:.3f} & "
            f"{row['nonfloor_mae']:.3f} & {row['mae']:.3f}" + r"\\"
        )
    text += [r"\hline\end{tabular}"]
    (ROOT / "paper/fmts_review_table.tex").write_text("\n".join(text), encoding="utf-8")
    simulations = []
    for path in sorted(OUT.glob("sim_*.json")):
        report = json.loads(path.read_text())
        evaluation = report["evaluations"]["test"]
        simulations.append(
            {"run": path.stem, "models": evaluation["models"], "intervals": evaluation["intervals"]}
        )
    save(OUT / "simulation_summary.json", simulations)
    save(
        OUT / "table_manifest.json",
        {
            "source": "esa_seed42.json#/evaluations/test/models",
            "source_sha256": sha(OUT / "esa_seed42.json"),
            "table_sha256": sha(ROOT / "paper/fmts_review_table.tex"),
            "figure_sha256": sha(ROOT / "docs/figures/fmts-remedy-decomposition.png"),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/fmts_review.json")
    parser.add_argument("--dataset", choices=("esa", "simulation", "outputs"), required=True)
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / "configuration.json", cfg)
    save(
        OUT / "runtime.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xgboost": __import__("xgboost").__version__,
            "scipy": __import__("scipy").__version__,
            "sklearn": __import__("sklearn").__version__,
            "config_sha256": sha(ROOT / args.config),
            "source_sha256": sha(Path(__file__)),
            "remedies_sha256": sha(ROOT / "ml/src/fmts_remedies.py"),
        },
    )
    if args.dataset == "esa":
        raw = validate_cdm_frame(realistic_training_events(load_esa_training(ROOT / "data/raw")))
        # Strictly before cutoff features; later target rows never enter the table.
        frame = build_feature_table(build_event_histories(raw))
        official = attach_official_test_labels(
            build_feature_table(
                build_event_histories(
                    validate_cdm_frame(load_official_test(ROOT / "data/raw")),
                    require_later_target=False,
                )
            ),
            load_official_test_labels(ROOT / "data/raw"),
        )
        save(
            OUT / "input_provenance.json",
            {
                name: sha(ROOT / "data/raw" / name)
                for name in ("train_data.zip", "test_data.csv", "zenodo_4463683.zip")
            },
        )
        for seed in cfg["redraw_seeds"]:
            manifest = grouped_splits(frame, seed)
            ids = {
                "train": manifest.train_ids,
                "validation": manifest.validation_ids,
                "calibration": manifest.calibration_ids,
                "test": manifest.test_ids,
            }
            if seed == 42:
                frozen = json.loads((ROOT / "ml/artifacts/split_manifest.json").read_text())
                assert all(set(ids[k]) == set(frozen[k]) for k in ids), "frozen split mismatch"
            run_dataset(
                f"esa_seed{seed}",
                frame,
                ids,
                {**cfg, "seed": seed},
                transfer=official if seed == 42 else None,
            )
    elif args.dataset == "simulation":
        for mechanism in cfg["simulation"]["mechanisms"]:
            for rate in cfg["simulation"]["floor_rates"]:
                for drift in cfg["simulation"]["drift"]:
                    for seed in cfg["redraw_seeds"]:
                        frame, manifest = simulation(seed, rate, mechanism, drift, cfg)
                        label = f"sim_{mechanism}_mass{rate}_drift{int(drift)}_seed{seed}"
                        run_dataset(
                            label,
                            frame,
                            manifest,
                            {**cfg, "seed": seed},
                            known_clipping=mechanism == "clipping",
                        )
    else:
        generate_outputs()


if __name__ == "__main__":
    main()
