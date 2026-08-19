from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

TRAIN_ARCHIVE_NAME = "train_data.zip"
TRAIN_MEMBER_NAME = "train_data.csv"
OFFICIAL_TEST_NAME = "test_data.csv"
ZENODO_ARCHIVE_NAME = "zenodo_4463683.zip"
ZENODO_LABEL_MEMBER = (
    "Collision Avoidance Challenge - Dataset/kelvins_competition_data/test_data_private.csv"
)


def normalize_esa_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the stable internal aliases used by the feature pipeline."""
    aliases = {
        "t_j2k_ecc": "t_ecc",
        "c_j2k_ecc": "c_ecc",
    }
    normalized = frame.copy()
    for source, target in aliases.items():
        if source in normalized.columns and target not in normalized.columns:
            normalized[target] = normalized[source]
    return normalized


def load_esa_training(raw_dir: Path) -> pd.DataFrame:
    archive = raw_dir / TRAIN_ARCHIVE_NAME
    if not archive.exists():
        raise FileNotFoundError(
            f"missing {archive}; run `python ml/src/download.py` or `python main.py`"
        )
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if TRAIN_MEMBER_NAME not in names:
            raise ValueError(f"{archive.name} does not contain {TRAIN_MEMBER_NAME}; found {names}")
        with bundle.open(TRAIN_MEMBER_NAME) as source:
            frame = pd.read_csv(source, low_memory=False)
    return normalize_esa_columns(frame)


def load_official_test(raw_dir: Path, nrows: int | None = None) -> pd.DataFrame:
    path = raw_dir / OFFICIAL_TEST_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run `python ml/src/download.py` or `python main.py`"
        )
    return normalize_esa_columns(pd.read_csv(path, nrows=nrows, low_memory=False))


def realistic_training_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep events shaped like the original challenge evaluation population."""
    grouped = frame.groupby("event_id")["time_to_tca"]
    summary = grouped.agg(["min", "max", "size"])
    eligible = summary[(summary["min"] < 1.0) & (summary["max"] >= 2.0) & (summary["size"] >= 2)]
    return frame[frame["event_id"].isin(eligible.index)].copy()


def validate_official_test_compatibility(
    raw_dir: Path, expected_columns: set[str]
) -> dict[str, int]:
    sample = load_official_test(raw_dir, nrows=100)
    missing = sorted(expected_columns - set(sample.columns))
    if missing:
        raise ValueError(f"official test file is missing expected columns: {missing}")
    return {"sampleRows": len(sample), "columns": len(sample.columns)}


def load_official_test_labels(raw_dir: Path) -> pd.DataFrame:
    archive = raw_dir / ZENODO_ARCHIVE_NAME
    if not archive.exists():
        raise FileNotFoundError(
            f"missing {archive}; run download_zenodo_labels() or `python ml/src/download.py`"
        )
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if ZENODO_LABEL_MEMBER not in names:
            raise ValueError(
                f"{archive.name} does not contain {ZENODO_LABEL_MEMBER}; found {names}"
            )
        with bundle.open(ZENODO_LABEL_MEMBER) as source:
            private = pd.read_csv(source, low_memory=False)
    if "event_id" not in private.columns or "true_risk" not in private.columns:
        raise ValueError("official-test labels need event_id and true_risk")
    keep = {"event_id", "true_risk", "time_to_tca"}
    leaked = [column for column in private.columns if column not in keep]
    labels = private[["event_id", "true_risk"]].copy()
    labels["y"] = labels["true_risk"].to_numpy(dtype=float)
    labels = labels.drop(columns=["true_risk"])
    if "time_to_tca" in private.columns:
        labels["target_time_to_tca"] = private["time_to_tca"].to_numpy(dtype=float)
    if labels["event_id"].duplicated().any():
        raise ValueError("official-test labels are not one row per event")
    labels.attrs["ignoredPrivateColumns"] = leaked
    return labels


def attach_official_test_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    if "true_risk" in features.columns:
        raise ValueError("official-test features must not contain true_risk")
    keyed = features.drop(columns=["y"], errors="ignore")
    merged = keyed.merge(labels[["event_id", "y"]], on="event_id", how="inner")
    if "true_risk" in merged.columns:
        raise ValueError("true_risk leaked into the feature table")
    return merged


def official_test_identity_report(
    train: pd.DataFrame, official_inputs: pd.DataFrame
) -> dict[str, object]:
    train_ids = set(train["event_id"].astype(int))
    test_ids = set(official_inputs["event_id"].astype(int))
    overlap = train_ids & test_ids

    def _snapshot(frame: pd.DataFrame) -> pd.DataFrame:
        cols = ["event_id", "time_to_tca", "risk", "mission_id", "miss_distance"]
        pre = frame.loc[frame["time_to_tca"] >= 2.0, cols]
        return pre.sort_values(["event_id", "time_to_tca"]).groupby("event_id").last()

    identical = 0
    if overlap:
        train_snap = _snapshot(train)
        test_snap = _snapshot(official_inputs)
        shared = train_snap.index.intersection(test_snap.index)
        if len(shared):
            identical = int(
                (
                    (train_snap.loc[shared, "risk"] == test_snap.loc[shared, "risk"])
                    & (train_snap.loc[shared, "mission_id"] == test_snap.loc[shared, "mission_id"])
                    & (
                        train_snap.loc[shared, "miss_distance"]
                        == test_snap.loc[shared, "miss_distance"]
                    )
                ).sum()
            )
    return {
        "trainEvents": len(train_ids),
        "officialTestEvents": len(test_ids),
        "numericIdOverlap": len(overlap),
        "identicalPreCutoffSnapshots": identical,
        "interpretation": (
            "event_id is independently numbered in the Kelvins train and test files. "
            "Official-test features come only from test_data.csv; labels come from "
            "Zenodo test_data_private.csv true_risk. Post-cutoff private fields are not features."
        ),
    }
