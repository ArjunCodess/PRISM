from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlretrieve

TRAIN_URL = (
    "https://kelvins.esa.int/media/public/competitions/"
    "collision-avoidance-challenge/train_data.zip"
)
TEST_URL = (
    "https://kelvins.esa.int/media/public/competitions/"
    "collision-avoidance-challenge/test_data.csv"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_esa(raw_dir: Path, provenance_path: Path) -> dict[str, dict[str, str | int]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, str | int]] = {}
    for url, name in ((TRAIN_URL, "train_data.zip"), (TEST_URL, "test_data.csv")):
        dest = raw_dir / name
        urlretrieve(url, dest)
        records[name] = {
            "url": url,
            "downloaded_at": datetime.now(UTC).isoformat(),
            "bytes": dest.stat().st_size,
            "sha256": _sha256(dest),
        }
    provenance_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    download_esa(root / "data" / "raw", root / "data" / "PROVENANCE.md")
