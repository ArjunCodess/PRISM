# PRISM

**Predictive Risk Intelligence for Space Monitoring**

**TL;DR:** PRISM is an offline, explainable T−48-hour conjunction-risk forecasting exhibit trained on the real ESA Collision Avoidance Challenge archive. It converts pre-cutoff CDM histories into a final log-risk forecast, calibrated warning probability, bootstrap model spread, abstention decision, and SHAP explanation. It is an educational research prototype, not flight software.

## Key Achievements

- **Real ESA pipeline:** Reads the original 87.7 MB training ZIP directly, validates 162,634 CDM rows, builds 8,293 cutoff-safe events, and verifies the official unlabeled test schema.
- **Leakage-safe evaluation:** Keeps event IDs disjoint across training, validation, calibration, and test partitions, then adds a separate four-mission hold-out and mission-ID ablation.
- **Honest baseline result:** The selected guarded ensemble lowers held-out MAE from `5.080` for persistence to `3.053`, a `39.9%` reduction, but ties persistence on ESA-style loss at `0.167`; PRISM therefore does not claim full baseline superiority.
- **Glass-box exhibit:** Ships five real-data stories, SHAP-based explanations, a calibrated warning view, model-spread bands, review-required abstention, and a laboratory with failures and robustness slices.
- **One-command operation:** Root [`main.py`](main.py) downloads or verifies data, trains, evaluates, draws figures, runs every check, builds both applications, and launches FastAPI and Next.js together.

## Overview

### What it does

PRISM forecasts the last reported collision-risk value for a conjunction event using only messages available at least 48 hours before time of closest approach. The primary target is `log10(Pc)`. A secondary calibrated view estimates whether the final event belongs to the ESA challenge high-risk class, `log10(Pc) ≥ −6`.

Each case combines the current reported risk, encounter geometry, covariance, observation history, forecast, model spread, and the factors that moved the prediction. Future messages stay hidden until the user explicitly reveals the final outcome.

### Why it matters

Conjunction estimates can move sharply as later observations refine uncertain orbits. Waiting improves the estimate but leaves less time to plan, so the practical research question is whether a model can identify important changes from the information available two days early. PRISM makes that question inspectable and keeps a human in control; it never recommends a manoeuvre or claims operational certification.

### What is distinctive here

The repository treats scientific restraint as part of the product. It compares every learned model with persistence, separates event and mission evaluation, measures the actual coverage of its ensemble spread, exports failure cases, and changes the UI claim when the acceptance rule is not met. The result is a demonstrable forecasting system whose limits remain visible.

### How it works

1. **Acquire and verify.** Download the ESA training archive and official test input, then record the URL, byte count, and SHA-256 in [`data/PROVENANCE.md`](data/PROVENANCE.md).
2. **Freeze time.** Keep messages at or before T−48 hours for features while using the last later update only as the training target.
3. **Build event features.** Summarize the latest safe snapshot, multi-message trends, geometry, covariance, observation counts, recency, and missingness into one row per event.
4. **Split and train.** Create disjoint event partitions and fit persistence, median, Ridge, XGBoost, a calibrated warning model, and ten bootstrap XGBoost regressors.
5. **Evaluate honestly.** Report MAE, ESA loss, warning metrics, interval coverage, ablations, robustness slices, mission tests, and the worst errors.
6. **Explain and serve.** Export frozen real-data cases with SHAP factors, generate figures, and serve them through FastAPI and a fully offline-capable Next.js interface.

### What we found

The selected ensemble is materially better on average error but does not beat persistence under the PRD's complete acceptance rule. On the 1,659-event untouched test split:

- Ensemble MAE is `3.053`, compared with `5.080` for persistence.
- Ensemble and persistence ESA-style loss are both `0.167`, with `F2 = 0.361`.
- Median absolute error is `0.476`; `50.8%` of predictions are within 0.5 log-risk units.
- Adding `mission_id` slightly worsens ordinary XGBoost MAE from `2.843` to `2.845`, so production excludes it.
- The nominal 50% and 90% bootstrap bands cover only `27.8%` and `49.2%` of outcomes, so they are model-spread indicators rather than calibrated predictive intervals.
- A four-mission hold-out exposes weak generalization on the rare high-risk tail. This is why the interface retains persistence, abstention, and failure cases instead of presenting a single confident score.

## Running

Install Python 3.11+ and Node.js 20+, then run from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

The first run verifies or downloads the real data, trains the full model bundle, generates graphs, executes Ruff, pytest, Vitest, ESLint and the production build, then opens:

- Web exhibit: `http://127.0.0.1:3000`
- API documentation: `http://127.0.0.1:8000/docs`

Useful modes:

```powershell
# Prepare, train, test, and build without starting servers
python main.py --build-only

# Start quickly from already verified data and frozen artifacts
python main.py --skip-download --skip-train --skip-graphs --skip-checks

# Redownload and verify both ESA inputs
python main.py --force-download --build-only

# Fast synthetic smoke test; never use its metrics as the project result
python main.py --source synthetic --synthetic-events 420 --build-only
```

The complete stage-by-stage guide is in [`docs/data-guide.md`](docs/data-guide.md).

## Data

PRISM uses the [ESA Collision Avoidance Challenge data](https://kelvins.esa.int/collision-avoidance-challenge/data/):

- `data/raw/train_data.zip`: 162,634 labeled CDM rows across 13,154 events; the pipeline retains 8,293 events that satisfy the local T−48 evaluation contract.
- `data/raw/test_data.csv`: 24,484 cutoff-safe rows across 2,167 events; final outcomes are withheld, so this file verifies input compatibility and is not used for local performance claims.

If either file is absent or fails its recorded checksum, `python main.py` obtains it from ESA through [`ml/src/download.py`](ml/src/download.py). Raw archives are preserved unchanged.

## Interpretation Caveats

- The `−6` high-risk threshold is the ESA challenge scoring class, not an ISRO operational threshold.
- The dataset contains historical anonymized ESA-supported events from 2015–2019, not live Indian catalogue data and not identifiable spacecraft.
- Bootstrap disagreement measures model stability. Its measured under-coverage means it must not be read as a calibrated 90% probability statement.
- High-risk examples are rare: only 66 eligible events in the labeled archive meet the configured class, including nine in the frozen test split.
- The learned model lowers aggregate MAE but does not improve the frozen ESA-style loss over persistence. Flight or manoeuvre decisions are explicitly out of scope.

## Outputs

- [`ml/artifacts/metrics.json`](ml/artifacts/metrics.json): frozen metrics, calibration, coverage, robustness, ablations, mission tests, and failure galleries.
- [`ml/artifacts/demo_cases.json`](ml/artifacts/demo_cases.json): five curated real-data cases bundled for offline use.
- [`ml/artifacts/model_card.json`](ml/artifacts/model_card.json) and [`docs/model-card.md`](docs/model-card.md): intended use, limits, and current result.
- [`docs/figures`](docs/figures): model comparison, calibration, feature importance, and feature ablation charts.
- [`data/processed/events.csv`](data/processed/events.csv): generated event-level feature table.

## Latest Full Run

The frozen bundle was produced on 16 August 2026 with:

```powershell
python main.py --build-only
```

It used 3,731 training, 1,659 validation, 1,244 calibration, and 1,659 test events. All 19 Python/API/ML tests, the web unit test, lint checks, TypeScript compilation, and the Next.js production build passed. See [`docs/prd-audit.md`](docs/prd-audit.md) for the exact PRD status; physical projector, Wi-Fi-off, timing, and backup-video checks still require the presentation laptop.

## Repository Layout

- [`main.py`](main.py): complete cross-platform project orchestrator.
- [`ml/src`](ml/src): acquisition, validation, feature engineering, training, calibration, evaluation, explanations, plotting, and inference.
- [`ml/artifacts`](ml/artifacts): frozen trained models and exported evidence.
- [`apps/api`](apps/api): FastAPI validation and inference service.
- [`apps/web`](apps/web): minimal Next.js exhibit with an offline artifact fallback.
- [`docs`](docs): model card, data guide, PRD audit, presentation material, and figures.
- [`prd.md`](prd.md): locked product and acceptance specification.

## Acknowledgments and Sources

PRISM was created by Arjun Vijay Prakash for City Montessori School, Kanpur Road Campus. The project relies on the ESA Space Debris Office's public Collision Avoidance Challenge dataset and challenge design described by Uriot et al. Context comes from [ISRO's NETRA control centre](https://www.isro.gov.in/ISRO%20SSAControl%20Centre.html), [NASA CARA](https://www.nasa.gov/cara/), and the CCSDS Conjunction Data Message standard. These references do not imply endorsement by ESA, ISRO, NASA, ISTRAC, or CCSDS.
