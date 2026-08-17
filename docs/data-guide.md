# ESA data and complete pipeline guide

## What is already downloaded

`data/raw/train_data.zip` is the labeled ESA Collision Avoidance Challenge training archive. It contains `train_data.csv` with 162,634 CDM rows, 13,154 event IDs, and 103 columns. This is the file PRISM can use for local training and held-out evaluation.

`data/raw/test_data.csv` is the official 2019 challenge test input. It contains cutoff-safe histories but no final outcome labels, so PRISM uses it only to verify schema compatibility. It must not be used to claim local model quality.

The download URLs are:

- Training archive: https://kelvins.esa.int/media/public/competitions/collision-avoidance-challenge/train_data.zip
- Official test input: https://kelvins.esa.int/media/public/competitions/collision-avoidance-challenge/test_data.csv
- ESA data page: https://kelvins.esa.int/collision-avoidance-challenge/data/

`data/PROVENANCE.md` records the download timestamp, byte size, and SHA-256 for both local files. Although its extension is `.md`, the content is intentionally machine-readable JSON.

## One-command workflow

From the repository root, run:

```powershell
python main.py
```

The command performs this sequence:

1. Keeps the existing non-empty ESA downloads, or downloads missing files and records their checksums.
2. Loads `train_data.csv` directly from the ZIP without extracting a second 233 MB copy.
3. Normalizes ESA column names, validates the CDMs, and keeps realistic events with a pre-T−48 message and a final update inside T−24.
4. Builds one leakage-safe feature row per event (snapshot, history, covariance trends, encounter-plane geometry, object-type dummies) and creates disjoint train, validation, calibration, and test event sets.
5. Trains persistence, median, Ridge, XGBoost on the T−48 target, high-risk calibration, a 10-model bootstrap ensemble, then runs snapshot-versus-history ablation, multi-horizon evaluation, abstention coverage, failure clustering, and SHAP contrast.
6. Evaluates the untouched local test set, exports SHAP explanations and six curated cases, and saves all frozen artifacts.
7. Generates PNG evaluation graphs in `docs/figures/` and copies JSON into `apps/web/public/` for optional static copies. The Next app does not use those files as a fallback; FastAPI reads `ml/artifacts/`.
8. Runs Ruff, pytest, Vitest, ESLint, TypeScript, and the Next.js production build.
9. Starts FastAPI at `http://127.0.0.1:8000` and the web exhibit at `http://127.0.0.1:3000`. `NEXT_PUBLIC_API_URL` must be set.

The first real-data training run can take several minutes. Later exhibit starts can reuse the artifacts:

```powershell
python main.py --skip-download --skip-train --skip-graphs --skip-checks
```

Useful alternatives:

```powershell
# Prepare and verify everything without starting servers
python main.py --build-only

# Redownload both ESA files even when they already exist
python main.py --force-download --build-only

# Fast pipeline smoke test using generated data
python main.py --source synthetic --synthetic-events 420 --build-only

# Use different local ports and do not open a browser
python main.py --skip-train --api-port 8100 --web-port 3100 --no-browser
```

Do not leave a synthetic retrain in `ml/artifacts/` if you are about to present. The frozen exhibit numbers are from `--source real`.

## Manual stages

Each stage remains independently runnable for debugging:

```powershell
python ml/src/download.py
python ml/src/pipeline.py --source real
python ml/src/plots.py
python -m pytest ml/tests apps/api/tests -q
cd apps/web
npm ci
npm run build
```

Do not extract or edit the original ZIP in `data/raw/`; keeping it unchanged preserves provenance. Generated feature tables belong in `data/processed/`, transient manifests belong in `data/interim/`, trained files belong in `ml/artifacts/`, and evaluation images belong in `docs/figures/`.

Geometry features live in `ml/src/features.py`. The T−48 ensemble lives in `ml/src/pipeline.py` and `ml/src/inference.py`. Abstention rules live in `ml/src/abstention.py`.
