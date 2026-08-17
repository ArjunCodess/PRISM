# PRISM

**Predicting final conjunction risk from information available 48 hours before closest approach.**

PRISM tests whether pre-T−48 conjunction histories contain enough predictive signal to improve forecasts of later reported `log10(Pc)` over persistence.

It is a research prototype for explainable conjunction-risk forecasting. **Not flight software. Not an operational decision system.**

Model version `prism-0.3.0`. The selected policy is a **hurdle residual**: XGBoost trained on MAE of the residual `Δ = y − current risk`, mixed with a collapse-to-floor classifier, then a persistence guard at the ESA `−6` class.

## Result

Held-out performance on 1,659 untouched test events. MAE is in `log10(Pc)` units.

| | Persistence | PRISM |
|---|---:|---:|
| MAE | 5.080 | **2.800** |
| ESA-style loss | **0.167** | **0.167** |

That is a 44.9% MAE cut. ESA-style loss is the challenge objective: high-risk MSE divided by F2 (F-beta with β=2, so recall of `log10(Pc) ≥ −6` is weighted more than precision). F2 is 0.361 for both. The exact tie is expected: the persistence guard copies the current report whenever it is already at or above `−6`, and invented high-risk forecasts are clamped just below `−6`, so PRISM matches persistence on the high-risk tail by design. The MAE reduction is continuous-risk accuracy, not a better risk-weighted decision score.

A single unguarded MAE XGBoost scores 2.550 MAE but F2 = 0. The constant training-set median scores 3.002 MAE.

Nominal 50% and 90% **split-conformal** bands cover **82.0%** and **91.4%** of outcomes (localized by whether the hurdle predicts a floor collapse). Bootstrap disagreement is still an abstention trigger, not the interval itself.

On four missions held out of training, overall MAE is 3.588 versus persistence 4.843. There is one held-out high-risk event; high-risk MAE ties persistence at 0.114 because of the guard. Random-event accuracy does not establish mission-level generalization.

The model contains useful signal beyond persistence, especially at longer forecast horizons, but that signal does not automatically translate into better risk-sensitive decisions. Two of nine test high-risk events remain false reassurance: accepted forecasts below `−6` while the later report is `≥ −6`.

## Research question

Conjunction estimates move as later observations refine orbital geometry, covariance, and observational uncertainty. Waiting usually improves the estimate and leaves less time to plan.

PRISM freezes the clock 48 hours before closest approach and asks:

**How much useful predictive information about the eventual reported collision probability exists in the CDM history available at T−48, beyond simply carrying the latest reported value forward?**

PRISM therefore studies forecasting under a fixed information constraint. It does not predict manoeuvres or operational collision outcomes. The target is the later reported `log10(Pc)`, not a physical collision probability.

T−48 is the ESA challenge information cutoff (`time_to_tca ≥ 2` days). The pipeline also reports T−72, T−24, and T−12.

The contribution is a controlled evaluation of whether historical CDM evolution provides useful early-warning signal at T−48, including horizon analysis, selective prediction, failure analysis, and mission-held-out testing.

## Findings

### What the model can do

1. **Average accuracy and decision quality are not the same thing.** A large MAE gain can coexist with an unchanged ESA-style score because that loss cares about the `log10(Pc) ≥ −6` tail, not average log-risk error. The exact F2 and ESA-style loss tie is the persistence guard (and high-risk clamp) working, not a scoring bug.

2. **Geometry and object type are in the snapshot.** Encounter-plane Mahalanobis distance, miss/σ, a combined-size proxy, and dummy flags for debris / payload / rocket body / unknown are tree features. The raw `c_object_type` string is not passed as a numeric column.

3. **On the MAE residual model, extra history does not help average error.** Snapshot-only MAE is 2.509. Adding historical summaries of risk, miss distance, speed, and observation counts moves it to 2.535. Covariance trends after that are 2.526. The history block is still temporal transforms of snapshot fields; under MAE loss it does not buy a further average-error cut the way it did under the older squared-error exhibit.

4. **The value of learned forecasting is highest when information is sparse.** Waiting helps persistence more than it helps PRISM. Horizon numbers below are the selected hurdle policy, not the unguarded booster.

   | Horizon | Hurdle | Persistence |
   |---|---:|---:|
   | T−72 | 4.578 | 7.748 |
   | T−48 | 2.800 | 5.080 |
   | T−24 | 2.199 | 2.634 |
   | T−12 | 1.385 | 1.444 |

5. **Abstention is selective prediction.** The `−6` class follows the ESA challenge definition. The persistence guard and 1.25 disagreement threshold were fixed design choices before evaluating the test split. PRISM abstains if the 90% conformal band crosses `log10(Pc) ≥ −6`, if current risk or miss distance is missing, if bootstrap disagreement exceeds 1.25 log-risk units, if it forecasts the dataset floor while today's report is still far from negligible, or if the warning head is elevated while the point forecast stays below `−6`. That keeps **88.97%** coverage (183 of 1,659 sent to review) and drops accepted MAE from 2.800 to **2.085**. Seven of nine test high-risk events are sent to review.

### Where it is weak

6. **False reassurance is not zero.** Two accepted forecasts stay below `−6` while the final report is `≥ −6` (2/9 high-risk test events). Those are late jumps the conformal band did not cross. Do not claim the exhibit never misses a high-risk event on accepted cases.

7. **Random-event performance is stronger than mission-held-out performance.** A four-mission hold-out still has only one high-risk event. Adding `mission_id` slightly worsens unguarded XGBoost MAE (2.550 → 2.570) and is excluded from the deployed exhibit.

8. **The high-risk estimate is based on a very small positive class.** Only 66 eligible events meet the ESA class `log10(Pc) ≥ −6`, including nine in the test split. Treat that probability as a scarce-label fit, not an operational warning system. Warning-head PR-AUC is 0.046; ROC-AUC is 0.941.

9. **Failures are not one bucket.** Of 1,659 test events, 1,275 are accurate to 0.5 log units. Dominant errors: 135 floor collapses to −30, 107 under-predictions, 41 over-predictions, 38 moderate errors, 28 sparse-history errors, 20 false high-risk calls, 13 close-approach errors, and 2 late high-risk jumps. SHAP on large errors puts more weight on risk trend and tracking completeness than on today's reported risk.

## Experimental setup

**Target.** Forecast of the final reported `log10(Pc)` after the cutoff.

**Information constraint.** Features use only messages with `time_to_tca ≥` the cutoff. The later update is the label, never an input.

**Data.** 162,634 CDM rows → cutoff-safe event histories → 8,293 eligible events. The frozen bundle uses 3,731 training, 1,659 validation, 1,244 calibration, and 1,659 test events. Train, validation, calibration, and test are event-disjoint; all model and policy choices are frozen before the untouched test evaluation. Validation is not reused for test selection.

**Model.** Event-level XGBoost on inspectable summaries (`reg:absoluteerror` on the residual from current risk), a collapse-to-floor classifier mixed with a hard threshold of 0.35, a warning classifier with isotonic calibration, split-conformal intervals, and a 10-model bootstrap used for disagreement. A sequence model is not used so temporal signals can be inspected directly. Inference is CPU-only; no GPU is required. SHAP explains the residual booster, not the mixed point forecast.

**Baselines.** Persistence, training-set median, and Ridge. The exhibit’s selected policy is the hurdle mix with a persistence guard when the current report is already at or above `−6`.

**License.** The PRISM code in this repository is MIT-licensed. The ESA Collision Avoidance Challenge dataset remains under ESA’s terms.

## Limitations

- Persistence remains competitive under the loss that motivated the original challenge.
- Two accepted high-risk misses remain on this split.
- Mission-level generalization, especially on high-risk events, is not established.
- The dataset is historical anonymized ESA-supported events from 2015–2019, not live catalogue data.
- Manoeuvre decisions are out of scope.
- The website requires a running FastAPI process. There is no silent JSON fallback.

## Exhibit

Six frozen real-data cases (two low, one review, three high). Each shows the current report, the forecast of the final reported `log10(Pc)`, conformal bands, a calibrated estimate of high-risk-event probability based on a very small positive class, SHAP factors, and a reveal-only later outcome.

The figures that carry the argument are [`docs/figures/forecast-horizon.png`](docs/figures/forecast-horizon.png), [`docs/figures/abstention-coverage.png`](docs/figures/abstention-coverage.png), and [`docs/figures/shap-contrast.png`](docs/figures/shap-contrast.png).

Run the exhibit on the presentation laptop with Next and FastAPI both up (`NEXT_PUBLIC_API_URL=http://127.0.0.1:8000`). Wi-Fi can be off. If the API is down, the site shows an error.

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

The website **requires** `NEXT_PUBLIC_API_URL`. Copy [`apps/web/.env.example`](apps/web/.env.example). There is no frozen-JSON fallback. See [`docs/deploy.md`](docs/deploy.md).

```powershell
python main.py --build-only
python main.py --skip-download --skip-train --skip-graphs --skip-checks
python main.py --force-download --build-only
python main.py --source synthetic --synthetic-events 420 --build-only
```

Stage-by-stage notes are in [`docs/data-guide.md`](docs/data-guide.md).

## Outputs

- [`ml/artifacts/metrics.json`](ml/artifacts/metrics.json): metrics, ablations, horizons, abstention, calibration, coverage, robustness, mission tests, failure clusters, SHAP contrast.
- [`ml/artifacts/demo_cases.json`](ml/artifacts/demo_cases.json): six curated real-data cases (two low, one review, three high). The API serves these; the website does not read them as a fallback.
- [`ml/artifacts/model_card.json`](ml/artifacts/model_card.json) and [`docs/model-card.md`](docs/model-card.md).
- [`docs/figures`](docs/figures): comparison, ablation, horizon, abstention, failure, and SHAP charts.
- [`data/processed/events.csv`](data/processed/events.csv): event-level feature table.

## Repository layout

- [`main.py`](main.py): project orchestrator.
- [`ml/src`](ml/src): acquisition, features, hurdle training, evaluation, experiments, explanations, inference.
- [`ml/artifacts`](ml/artifacts): frozen models and evidence.
- [`apps/api`](apps/api): FastAPI service (required for the exhibit).
- [`apps/web`](apps/web): Next.js exhibit (API-only).
- [`docs`](docs): model card, data guide, deploy guide, PRD audit, presentation scripts, figures.
- [`prd.md`](prd.md): locked product specification, with an 18 August 2026 addendum for the hurdle exhibit.

## Acknowledgments and sources

PRISM was created by Arjun Vijay Prakash for City Montessori School, Kanpur Road Campus. The project uses the ESA Space Debris Office’s public Collision Avoidance Challenge dataset and challenge design described by Uriot et al. Context comes from [ISRO’s NETRA control centre](https://www.isro.gov.in/ISRO%20SSAControl%20Centre.html), [NASA CARA](https://www.nasa.gov/cara/), and the CCSDS Conjunction Data Message standard. These references do not imply endorsement by ESA, ISRO, NASA, ISTRAC, or CCSDS.
