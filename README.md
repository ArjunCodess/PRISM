# PRISM

**Predicting final conjunction risk from information available 48 hours before closest approach.**

PRISM tests whether pre-T−48 conjunction histories contain enough predictive signal to improve forecasts of later reported `log10(Pc)` over persistence.

It is a research prototype for offline, explainable conjunction-risk forecasting. **Not flight software. Not an operational decision system.**

## Result

Held-out performance on 1,659 untouched test events. MAE is in `log10(Pc)` units.

| | Persistence | PRISM |
|---|---:|---:|
| MAE | 5.080 | **3.052** |
| ESA-style loss | **0.167** | **0.167** |

ESA-style loss is the challenge objective: high-risk MSE divided by F2 (F-beta with β=2, so recall of `log10(Pc) ≥ −6` is weighted more than precision). F2 is 0.361 for both. That exact tie is expected: the persistence guard copies the current report whenever it is already at or above `−6`, which is the region ESA-style loss scores, so PRISM matches persistence on the high-risk tail by design. The 39.9% MAE reduction is therefore continuous-risk accuracy, not a better risk-weighted decision score.

PRISM’s nominal 90% bootstrap band covers only **48.6%** of outcomes. It is therefore shown as model spread, not predictive probability.

On four missions held out of training, high-risk MAE is **18.1** (one held-out high-risk event). Random-event accuracy does not establish mission-level generalization.

The model contains useful signal beyond persistence, especially at longer forecast horizons, but that signal does not automatically translate into better risk-sensitive decisions or calibrated uncertainty.

## Research question

Conjunction estimates move as later observations refine orbital geometry, covariance, and observational uncertainty. Waiting usually improves the estimate and leaves less time to plan.

PRISM freezes the clock 48 hours before closest approach and asks:

**How much useful predictive information about the eventual reported collision probability exists in the CDM history available at T−48, beyond simply carrying the latest reported value forward?**

PRISM therefore studies forecasting under a fixed information constraint. It does not predict manoeuvres or operational collision outcomes. The target is the later reported `log10(Pc)`, not a physical collision probability.

T−48 is the ESA challenge information cutoff (`time_to_tca ≥ 2` days). The pipeline also reports T−72, T−24, and T−12.

The contribution is a controlled evaluation of whether historical CDM evolution provides useful early-warning signal at T−48, including horizon analysis, selective prediction, failure analysis, and mission-held-out testing.

## Findings

### What the model can do

1. **Average accuracy and decision quality are not the same thing.** A large MAE gain can coexist with an unchanged ESA-style score because that loss cares about the `log10(Pc) ≥ −6` tail, not average log-risk error. The exact F2 and ESA-style loss tie is the persistence guard working, not a scoring bug.

2. **History provides a measurable gain, while covariance trends add little marginal information once snapshot and history features are present.** The history block consists of temporal transforms of variables already available in the latest snapshot, allowing the ablation to isolate information from their evolution rather than simply adding unrelated measurements. On the single XGBoost model, snapshot-only MAE is 2.960. Adding historical summaries of risk, miss distance, speed, and observation counts lowers it to 2.842. Covariance trends after that add almost nothing (2.838).

3. **The value of learned forecasting is highest when information is sparse.** Waiting helps persistence more than it helps PRISM: the learned advantage is largest at T−72 and nearly gone at T−12.

   | Horizon | XGBoost | Persistence |
   |---|---:|---:|
   | T−72 | 3.241 | 7.748 |
   | T−48 | 2.838 | 5.080 |
   | T−24 | 2.097 | 2.634 |
   | T−12 | 1.390 | 1.444 |

4. **Abstention is selective prediction.** The `−6` class follows the ESA challenge definition. The persistence guard and 1.25 disagreement threshold were fixed design choices before evaluating the test split. PRISM abstains if the 90% bootstrap band crosses `log10(Pc) ≥ −6`, if current risk or miss distance is missing, or if bootstrap disagreement exceeds 1.25 log-risk units. That keeps 78.2% coverage (21.8% abstention) and drops accepted MAE from 3.052 to 1.920. All nine test high-risk events are either flagged by the prediction or sent to review. False reassurance means an accepted forecast with predicted `log10(Pc) < −6` while the final reported value is `≥ −6`. There are none.

### Where it is weak

5. **Random-event performance is stronger than mission-held-out performance.** A four-mission hold-out remains weak on the rare high-risk tail (one held-out high-risk event; high-risk MAE 18.1). Adding `mission_id` provides negligible improvement (2.838 → 2.835) and does not materially change performance, so it is excluded from the deployed exhibit.

6. **Ensemble disagreement is not equivalent to calibrated uncertainty.** Nominal 50% and 90% bootstrap bands cover 26.4% and 48.6% of outcomes. The interface labels them as model spread.

7. **The high-risk estimate is based on a very small positive class.** Only 66 eligible events meet the ESA class `log10(Pc) ≥ −6`, including nine in the test split. Because only nine test events are positive, this probability estimate should be treated as a scarce-label fit, not an operational warning system.

8. **Failures are not one bucket.** Inaccurate forecasts cluster into over-prediction (362), under-prediction (210), sparse-history errors (62), and 40 events where the final reported risk collapses to the dataset floor of −30. Only two test events are late high-risk jumps. Tracking-completeness features have higher mean |SHAP| among errors of two or more log units than among accurate cases.

## Experimental setup

**Target.** Forecast of the final reported `log10(Pc)` after the cutoff.

**Information constraint.** Features use only messages with `time_to_tca ≥` the cutoff. The later update is the label, never an input.

**Data.** 162,634 CDM rows → cutoff-safe event histories → 8,293 eligible events. The frozen bundle from 16 August 2026 uses 3,731 training, 1,659 validation, 1,244 calibration, and 1,659 test events. Train, validation, calibration, and test are event-disjoint; all model and policy choices are frozen before the untouched test evaluation. Validation is not reused for test selection.

**Model.** Event-level XGBoost on inspectable summaries. A sequence model is not used so temporal signals can be inspected directly and the exhibit stays reproducible offline. Inference is a CPU-only 10-model XGBoost ensemble on a few hundred tabular features; no GPU is required.

**Baselines.** Persistence, training-set median, and Ridge. The exhibit’s selected policy is a ten-model bootstrap XGBoost median with a persistence guard when the current report is already at or above `−6`. The `−6` class follows the ESA challenge definition. The persistence guard and 1.25 disagreement threshold were fixed design choices before evaluating the test split.

**License.** The PRISM code in this repository is MIT-licensed. The ESA Collision Avoidance Challenge dataset remains under ESA’s terms.

## Limitations

- Persistence remains competitive under the loss that motivated the original challenge.
- Bootstrap intervals are miscalibrated; they are used for abstention, not as 90% probability statements.
- Mission-level generalization, especially on high-risk events, is not established.
- The dataset is historical anonymized ESA-supported events from 2015–2019, not live catalogue data.
- Manoeuvre decisions are out of scope.

## Exhibit

Five frozen real-data cases: easy correct, hard correct, de-escalation, abstention, and confident failure. Each shows the current report, the forecast of the final reported `log10(Pc)`, model-spread bands, a calibrated estimate of high-risk-event probability based on a very small positive class, SHAP factors, and a reveal-only later outcome.

The figures that carry the argument are [`docs/figures/forecast-horizon.png`](docs/figures/forecast-horizon.png), [`docs/figures/abstention-coverage.png`](docs/figures/abstention-coverage.png), and [`docs/figures/shap-contrast.png`](docs/figures/shap-contrast.png).

## Running

Install Python 3.11+ and Node.js 20+, then run from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

The first run verifies or downloads the real data, trains, evaluates, draws figures, runs checks, builds both applications, and opens:

- Web exhibit: `http://127.0.0.1:3000`
- API documentation: `http://127.0.0.1:8000/docs`

A public deploy must set `NEXT_PUBLIC_API_URL` so the site loads cases from FastAPI and each case page runs live `POST /v1/risk/predict`. Frozen JSON is only a local fallback in development. Copy [`apps/web/.env.example`](apps/web/.env.example). See [`docs/deploy.md`](docs/deploy.md).

```powershell
python main.py --build-only
python main.py --skip-download --skip-train --skip-graphs --skip-checks
python main.py --force-download --build-only
python main.py --source synthetic --synthetic-events 420 --build-only
```

Stage-by-stage notes are in [`docs/data-guide.md`](docs/data-guide.md).

## Outputs

- [`ml/artifacts/metrics.json`](ml/artifacts/metrics.json): metrics, ablations, horizons, abstention, calibration, coverage, robustness, mission tests, failure clusters, SHAP contrast.
- [`ml/artifacts/demo_cases.json`](ml/artifacts/demo_cases.json): six curated real-data cases (two low, one review, three high).
- [`ml/artifacts/model_card.json`](ml/artifacts/model_card.json) and [`docs/model-card.md`](docs/model-card.md).
- [`docs/figures`](docs/figures): comparison, ablation, horizon, abstention, failure, and SHAP charts.
- [`data/processed/events.csv`](data/processed/events.csv): event-level feature table.

## Repository layout

- [`main.py`](main.py): project orchestrator.
- [`ml/src`](ml/src): acquisition, features, training, evaluation, experiments, explanations, inference.
- [`ml/artifacts`](ml/artifacts): frozen models and evidence.
- [`apps/api`](apps/api): FastAPI service.
- [`apps/web`](apps/web): Next.js exhibit with an offline artifact fallback.
- [`docs`](docs): model card, data guide, deploy guide, PRD audit, presentation scripts, figures.
- [`prd.md`](prd.md): locked product specification.

## Acknowledgments and sources

PRISM was created by Arjun Vijay Prakash for City Montessori School, Kanpur Road Campus. The project uses the ESA Space Debris Office’s public Collision Avoidance Challenge dataset and challenge design described by Uriot et al. Context comes from [ISRO’s NETRA control centre](https://www.isro.gov.in/ISRO%20SSAControl%20Centre.html), [NASA CARA](https://www.nasa.gov/cara/), and the CCSDS Conjunction Data Message standard. These references do not imply endorsement by ESA, ISRO, NASA, ISTRAC, or CCSDS.
