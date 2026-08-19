from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlretrieve

TRAIN_URL = (
    "https://kelvins.esa.int/media/public/competitions/collision-avoidance-challenge/train_data.zip"
)
TEST_URL = (
    "https://kelvins.esa.int/media/public/competitions/collision-avoidance-challenge/test_data.csv"
)
ZENODO_RECORD = "4463683"
ZENODO_DOI = "10.5281/zenodo.4463683"
ZENODO_RELEASED = "2021-01-25"
ZENODO_URL = (
    "https://zenodo.org/api/records/4463683/files/"
    "Collision%20Avoidance%20Challenge%20-%20Dataset.zip/content"
)
ZENODO_ARCHIVE_NAME = "zenodo_4463683.zip"
ZENODO_MD5 = "d19dc8875229f2f6893253c38adddc87"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_provenance(path: Path) -> dict[str, dict[str, str | int]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def _write_provenance(path: Path, records: dict[str, dict[str, str | int]]) -> None:
    path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def _record(
    url: str, dest: Path, extra: dict[str, str | int] | None = None
) -> dict[str, str | int]:
    payload: dict[str, str | int] = {
        "url": url,
        "downloaded_at": datetime.now(UTC).isoformat(),
        "bytes": dest.stat().st_size,
        "sha256": _sha256(dest),
    }
    if extra:
        payload.update(extra)
    return payload


def download_esa(raw_dir: Path, provenance_path: Path) -> dict[str, dict[str, str | int]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    records = _read_provenance(provenance_path)
    for url, name in ((TRAIN_URL, "train_data.zip"), (TEST_URL, "test_data.csv")):
        dest = raw_dir / name
        urlretrieve(url, dest)
        records[name] = _record(url, dest)
    _write_provenance(provenance_path, records)
    return records


def download_zenodo_labels(
    raw_dir: Path, provenance_path: Path, force: bool = False
) -> dict[str, dict[str, str | int]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / ZENODO_ARCHIVE_NAME
    records = _read_provenance(provenance_path)
    if dest.exists() and not force and _md5(dest) == ZENODO_MD5:
        records[ZENODO_ARCHIVE_NAME] = records.get(ZENODO_ARCHIVE_NAME) or _record(
            ZENODO_URL,
            dest,
            extra={
                "doi": ZENODO_DOI,
                "record": ZENODO_RECORD,
                "released": ZENODO_RELEASED,
                "md5": ZENODO_MD5,
            },
        )
        _write_provenance(provenance_path, records)
        return records
    urlretrieve(ZENODO_URL, dest)
    digest = _md5(dest)
    if digest != ZENODO_MD5:
        raise ValueError(f"{dest.name} md5 {digest} does not match Zenodo {ZENODO_MD5}")
    records[ZENODO_ARCHIVE_NAME] = _record(
        ZENODO_URL,
        dest,
        extra={
            "doi": ZENODO_DOI,
            "record": ZENODO_RECORD,
            "released": ZENODO_RELEASED,
            "md5": digest,
        },
    )
    _write_provenance(provenance_path, records)
    return records


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    raw = root / "data" / "raw"
    provenance = root / "data" / "PROVENANCE.md"
    if not (raw / "train_data.zip").exists() or not (raw / "test_data.csv").exists():
        download_esa(raw, provenance)
    download_zenodo_labels(raw, provenance)
