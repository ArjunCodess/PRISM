"""Download primary metadata and inspect the released archive without extracting it."""

import gzip
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ml/artifacts/fmts_review"
DOIS = [
    "10.2307/1907382",
    "10.2307/1909582",
    "10.1080/00401706.1992.10485228",
    "10.1021/ac60259a007",
    "10.1080/01621459.2017.1307116",
    "10.1093/imaiai/iaaa017",
    "10.1287/mnsc.2020.3922",
    "10.1007/978-3-642-04747-3_26",
    "10.1007/s42064-021-0101-5",
]


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "PRISM-research-audit/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    record = fetch("https://zenodo.org/api/records/4463683")
    (OUT / "zenodo_metadata.json").write_text(json.dumps(record["metadata"], indent=2))
    refs = {}
    for doi in DOIS:
        try:
            message = fetch("https://api.crossref.org/works/" + doi)["message"]
            refs[doi] = {
                key: message.get(key)
                for key in (
                    "DOI",
                    "title",
                    "author",
                    "publisher",
                    "container-title",
                    "published",
                    "URL",
                )
            }
        except Exception as exc:
            refs[doi] = {"verification_error": str(exc)}
    (OUT / "verified_references.json").write_text(json.dumps(refs, indent=2))
    path = ROOT / "data/raw/zenodo_4463683.zip"
    if not path.exists():
        urllib.request.urlretrieve(record["files"][0]["links"]["self"], path)
    prefix = "Collision Avoidance Challenge - Dataset/"
    with zipfile.ZipFile(path) as bundle:
        documentation = bundle.read(prefix + "raw_data/raw_data_2015-2019.txt").decode()
        with gzip.GzipFile(
            fileobj=bundle.open(prefix + "raw_data/raw_data_2015-2019.gz")
        ) as handle:
            raw = pd.read_csv(handle, low_memory=False)
        with zipfile.ZipFile(ROOT / "data/raw/train_data.zip") as train_zip:
            train = pd.read_csv(train_zip.open("train_data.csv"), low_memory=False)
        result = {
            "zip_sha256": sha(path),
            "members": bundle.namelist(),
            "documentation_mentions_minus30": "-30" in documentation,
            "documentation_mentions_clipping": "clip" in documentation.lower(),
            "documentation_mentions_floor": "floor" in documentation.lower(),
            "released_processing_code": [
                n for n in bundle.namelist() if n.endswith((".py", ".ipynb", ".m", ".R"))
            ],
            "calendar_columns": [
                c
                for c in raw.columns
                if any(k in c.lower() for k in ("date", "epoch", "timestamp"))
            ],
        }
        for name, frame in (("raw", raw), ("competition_train", train)):
            risk = frame.risk.to_numpy()
            result[name] = {
                "rows": len(frame),
                "events": int(frame.event_id.nunique()),
                "minimum_risk": float(np.nanmin(risk)),
                "minus30_rows": int(np.isclose(risk, -30, atol=1e-6, rtol=0).sum()),
                "below_minus30_rows": int(np.sum(risk < -30 - 1e-6)),
                "missing_risk_rows": int(np.isnan(risk).sum()),
            }
        # Event identifiers were renumbered. Match on unchanged message features, not IDs.
        keys = ["time_to_tca", "mission_id", "miss_distance", "relative_speed"]
        unique_raw = raw.drop_duplicates(keys, keep=False)
        joined = train.merge(unique_raw[keys + ["risk"]], on=keys, suffixes=("", "_raw"))
        floor = np.isclose(joined.risk, -30, atol=1e-6, rtol=0)
        result["unique_feature_match"] = {
            "keys": keys,
            "matched_rows": len(joined),
            "matched_floor_rows": int(floor.sum()),
            "floor_raw_same": int(
                np.sum(floor & np.isclose(joined.risk_raw, -30, atol=1e-6, rtol=0))
            ),
            "floor_raw_below": int(np.sum(floor & (joined.risk_raw < -30 - 1e-6))),
            "risk_mismatch_rows": int(
                np.sum(~np.isclose(joined.risk, joined.risk_raw, atol=1e-6, rtol=0))
            ),
        }
    (OUT / "archive_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
