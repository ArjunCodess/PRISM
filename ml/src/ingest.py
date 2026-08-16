from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

TRAIN_ARCHIVE_NAME = "train_data.zip"
TRAIN_MEMBER_NAME = "train_data.csv"
OFFICIAL_TEST_NAME = "test_data.csv"


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
