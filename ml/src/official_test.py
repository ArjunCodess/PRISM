from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from build_events import build_event_histories
from constants import HIGH_RISK_THRESHOLD
from download import download_zenodo_labels
from evaluate import clip_for_esa, esa_loss, honest_system_report, level_scoreboard_row
from features import build_feature_table
from floor_model import combine_floor_hurdle
from ingest import (
    attach_official_test_labels,
    load_esa_training,
    load_official_test,
    load_official_test_labels,
    official_test_identity_report,
)
from train_classifier import TrainedClassifier
from train_regressor import (
    TrainedRegressor,
    persistence_predict,
    predict_reconstructed,
)
from validate import validate_cdm_frame
from xgboost import XGBClassifier, XGBRegressor


def _feature_matrix(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    aligned = frame.copy()
    for name in names:
        if name not in aligned.columns:
            aligned[name] = np.nan
    return aligned[names].apply(pd.to_numeric, errors="coerce")


def _booster_names(model: XGBRegressor | XGBClassifier, fallback: list[str]) -> list[str]:
    stored = model.get_booster().feature_names
    if stored:
        return list(stored)
    return list(fallback)


def _guarded_ensemble(
    frame: pd.DataFrame, ensemble: list[object], feature_names: list[str]
) -> np.ndarray:
    x = _feature_matrix(frame, feature_names)
    raw = np.column_stack([model.predict(x) for model in ensemble])
    risk = frame["risk"].to_numpy(dtype=float)
    guard = risk >= HIGH_RISK_THRESHOLD
    matrix = np.where(guard[:, None], risk[:, None], raw)
    return np.median(matrix, axis=1)


def score_official_test(root: Path, artifacts: Path) -> dict[str, object]:
    raw_dir = root / "data" / "raw"
    download_zenodo_labels(raw_dir, root / "data" / "PROVENANCE.md")
    train_raw = load_esa_training(raw_dir)
    official_inputs = load_official_test(raw_dir)
    identity = official_test_identity_report(train_raw, official_inputs)
    labels = load_official_test_labels(raw_dir)
    events = build_event_histories(
        validate_cdm_frame(official_inputs), require_later_target=False
    )
    features = attach_official_test_labels(build_feature_table(events), labels)
    if features.empty:
        raise RuntimeError("official-test labels did not match cutoff-safe inputs")

    schema = list(
        json.loads((artifacts / "feature_schema.json").read_text(encoding="utf-8"))["features"]
    )
    persist = persistence_predict(features)
    y = features["y"].to_numpy(dtype=float)
    risk = features["risk"].to_numpy(dtype=float)

    xgb_model = XGBRegressor()
    xgb_model.load_model(str(artifacts / "risk_regressor.json"))
    xgb_names = _booster_names(xgb_model, schema)
    xgb_pred = np.asarray(xgb_model.predict(_feature_matrix(features, xgb_names)), dtype=float)

    bundle = joblib.load(artifacts / "warning_calibrator.joblib")
    ensemble = bundle["ensemble"]
    ens_names = list(bundle.get("feature_names") or schema)
    ens_pred = _guarded_ensemble(features, ensemble, ens_names)

    residual_model = XGBRegressor()
    residual_model.load_model(str(artifacts / "residual_regressor.json"))
    residual_names = _booster_names(residual_model, schema)
    residual = TrainedRegressor(
        model=residual_model, feature_names=residual_names, kind="residual_xgboost"
    )
    residual_pred = predict_reconstructed(residual, features)

    floor_clf_model = XGBClassifier()
    floor_clf_model.load_model(str(artifacts / "floor_classifier.json"))
    floor_clf = TrainedClassifier(
        model=floor_clf_model,
        feature_names=_booster_names(floor_clf_model, schema),
    )
    x_floor = _feature_matrix(features, floor_clf.feature_names)
    floor_proba = np.asarray(floor_clf.model.predict_proba(x_floor)[:, 1], dtype=float)
    floor_residual_model = XGBRegressor()
    floor_residual_model.load_model(str(artifacts / "floor_residual_regressor.json"))
    floor_residual = TrainedRegressor(
        model=floor_residual_model,
        feature_names=_booster_names(floor_residual_model, schema),
        kind="residual_xgboost",
    )
    policy = json.loads((artifacts / "floor_hurdle.json").read_text(encoding="utf-8"))
    floor_pred = combine_floor_hurdle(
        floor_proba,
        predict_reconstructed(floor_residual, features),
        float(policy["threshold"]),
        risk=risk,
        use_persist_guard=bool(policy["usePersistGuard"]),
    )

    systems = {
        "persistence": persist,
        "xgboost": xgb_pred,
        "ensemble": ens_pred,
        "residual": residual_pred,
        "floorHurdle": floor_pred,
    }
    board = {name: level_scoreboard_row(y, pred) for name, pred in systems.items()}
    honest = {
        name: honest_system_report(
            y, pred, risk, persist, compare_to_persistence=name != "persistence"
        )
        for name, pred in systems.items()
    }
    esa = {name: esa_loss(y, clip_for_esa(pred)) for name, pred in systems.items()}
    return {
        "frozenBeforeLook": True,
        "source": {
            "inputs": "data/raw/test_data.csv",
            "labels": "zenodo_4463683.zip test_data_private.csv true_risk",
            "doi": "10.5281/zenodo.4463683",
            "released": "2021-01-25",
        },
        "nEvents": int(len(features)),
        "nHighRisk": int((y >= HIGH_RISK_THRESHOLD).sum()),
        "nFloor": int((y <= -30.0 + 1e-6).sum()),
        "identity": identity,
        "replacesExhibit": False,
        "board": board,
        "esa": {
            name: {"esaLoss": row["esa_loss"], "mseHr": row["mse_hr"], "f2": row["f2"]}
            for name, row in esa.items()
        },
        "honest": honest,
        "uriotReference": {
            "lrp": 0.694,
            "sesc": 0.556,
            "note": (
                "Uriot et al. 2021 ESA-style loss on this official-test distribution, "
                "after clipping predictions below -6 to -6.001."
            ),
        },
    }
