# PRISM

**Predicting final conjunction risk from information available 48 hours before closest approach.**

PRISM tests whether pre-T−48 conjunction histories contain enough predictive signal to improve forecasts of later reported `log10(Pc)` over persistence.

It is a research prototype for explainable conjunction-risk forecasting. **Not flight software. Not an operational decision system.**

The selected policy is a **T−48 bootstrap XGBoost median**: it forecasts the later reported `log10(Pc)` from cutoff-safe CDM summaries, then copies today's report when that report is already at or above the ESA `−6` class.

The living manuscript is IEEE conference format: [`paper/main.tex`](paper/main.tex) ([`paper/main.pdf`](paper/main.pdf)). Rebuild it after every paper change with `scripts/compile-paper.ps1`.

Honest metrics on the frozen 18 August models (`python main.py --skip-train --build-only`) live in `ml/artifacts/metrics.json`. Floor-excluded MAE, residual MAE, bootstrap CIs, Wilcoxon tests, a residual-to-persistence candidate, a two-part floor candidate, split-conformal coverage, a one-shot official-test score, a dilution / max-risk probe, five grouped-split redraws, leave-one-high-risk-out, and a class-threshold sweep are measured.

## Result

Held-out performance on 1,659 untouched test events. 1,352 of them later report the dataset floor `−30`. Errors are in `log10(Pc)` units. ΔMAE is MAE(persistence) − MAE(model) with a 95% bootstrap interval from 1,000 event resamples. Wilcoxon *p* is two-sided on paired `|error|`.

| | Persistence | Unguarded XGBoost | Residual XGBoost | Floor hurdle | Selected ensemble |
|---|---:|---:|---:|---:|---:|
| MAE | 5.080 | 2.809 | 2.760 | **2.109** | 3.059 |
| Median AE | **0.000** | 0.554 | 0.453 | **0.000** | 0.473 |
| Floor-excluded MAE | **4.073** | 7.562 | 7.375 | 9.311 | 7.332 |
| Floor-only MAE | 5.308 | 1.730 | 1.712 | **0.474** | 2.088 |
| Residual MAE (`|y − risk|` vs `|pred − risk|`) | 5.080 / 0.000 | 5.080 / 5.222 | 5.080 / 5.134 | 5.080 / 5.945 | 5.080 / 4.582 |
| ΔMAE vs persistence [95% CI] | — | 2.270 [1.930, 2.638] | 2.319 [1.977, 2.685] | 2.971 [2.524, 3.415] | 2.021 [1.716, 2.380] |
| Wilcoxon *p* | — | 0.91 | 0.92 | **&lt;10<sup>−33</sup>** | 0.31 |
| ESA-style loss | **0.167** | ∞ (F2 = 0) | ∞ (F2 = 0) | ∞ (F2 = 0) | **0.167** |

The overall MAE drop is real as a mean (the CI excludes 0) and is not a typical-event win for unguarded or residual XGBoost: floor-excluded MAE is worse than persistence, median AE for persistence is already 0, and Wilcoxon does not reject equal `|error|`. Across **five grouped redraws** (seeds 42–46; seed 42 is still the reported split), unguarded XGBoost MAE advantage is **2.34 ± 0.04** and floor-excluded MAE is **7.78 ± 0.21** (persistence floor-excluded **4.32 ± 0.23**). Residual reconstruction is **2.35 ± 0.04** MAE advantage. Predicting `y − risk` and adding the hat back to persistence is almost the same as unguarded level-valued XGBoost (test MAE 2.760 vs 2.809). A two-part floor model — `P(y = −30)` plus a residual regressor fit only on non-floor training events, with threshold `0.15` chosen on validation and no persistence guard — reaches test MAE **2.109** and median AE 0. Floor-class confusion on test is TP 1319 / FP 174 / FN 33 / TN 133 (recall 0.976, precision 0.883). Wilcoxon versus persistence is `p < 10^−33`, but floor-excluded MAE is 9.311: the win is still a floor-call statistic (H2). On **validation**, the floor hurdle ranks first by MAE (1.916 vs unguarded XGBoost 2.669). It is a stored candidate (`floor_classifier.json`, `replacesExhibit: false`), not the live exhibit. ESA-style loss is the challenge objective: high-risk MSE divided by F2 (F-beta with β=2). F2 is 0.361 for persistence and the selected ensemble, and 0 for the unguarded, residual, and floor candidates. The exact tie is the persistence guard copying the current report whenever it is already at or above `−6`.

A constant training-set median scores 3.002 MAE.

PRISM’s nominal 90% bootstrap band covers **47.7%** of outcomes (50% band **26.0%**). It is therefore shown as **model spread**, not predictive probability. Split-conformal 90% intervals around the same exhibit point, fit on 1,244 calibration events, cover **89.7%** of the untouched test (50% conformal **49.6%**). The calibrated 90% band is wide (mean width 18.33 log units) because floor jumps live in the calibration scores.

On four missions held out of training, overall MAE is 2.688 versus persistence 4.843. High-risk MAE is **19.2** on one held-out high-risk event. Random-event accuracy does not establish mission-level generalization.

The model contains useful signal beyond persistence on mean error, especially at longer forecast horizons, but that signal is concentrated in floor collapses and does not automatically translate into better risk-sensitive decisions or calibrated uncertainty.

Official-test labels (Zenodo 4463683, 25 Jan 2021) were scored once after freeze on 2,167 events (150 high-risk, 1,673 floor). Features come only from `test_data.csv`. Shared `event_id` integers with training are independent numbering (0 identical pre-cutoff snapshots). Clipped persistence ESA-style loss is **0.694**, matching Uriot last-risk-prediction on this distribution. The selected ensemble **ties** that loss via the `−6` persist guard. Residual and floor candidates lower MAE (3.476 / 3.177 vs 5.209) but raise *L* to **104** and **70.5** because *F*<sub>2</sub> collapses. None of the frozen models beat published sesc *L* = 0.556. These numbers are research results, not a website leaderboard, and were not used to retune.

| Official test | Persistence | Unguarded XGBoost | Residual XGBoost | Floor hurdle | Selected ensemble |
|---|---:|---:|---:|---:|---:|
| MAE | 5.209 | 3.504 | 3.476 | **3.177** | 3.107 |
| Median AE | **0.000** | 0.690 | 0.594 | **0.000** | 0.439 |
| Floor-excluded MAE | **4.287** | 9.333 | 9.289 | 11.832 | 6.750 |
| ESA-style loss | **0.694** | ~1.75×10<sup>6</sup> (*F*<sub>2</sub>=0) | 104 | 70.5 | **0.694** |
| *F*<sub>2</sub> | **0.739** | 0.000 | 0.017 | 0.025 | **0.739** |

H4 (dilution / max-risk probe, logistic and Spearman only; not an extra exhibit model). On the frozen local test, `dilution_gap = max_risk_estimate − risk` has Spearman **ρ = −0.768** with `|y − risk|` and **ρ = +0.407** with the floor label. Combined covariance volume (`log_combined_sigma_det`) has **ρ = +0.399** with `|y − risk|`. A train-fit logistic of floor ~ gap + miss distance + `n_messages` has test AUC **0.819**. Large gaps are already-floor snapshots (Q1 floor rate 0.56 and mean `|Δrisk|` 13.03; Q3–Q4 floor rate > 0.95 and mean movement < 0.75). F10 is a weak control (ρ = 0.108 with `|Δrisk|`). The naive “large max-risk gap means the report will still move” story is rejected. Large covariance predicting more movement is supported. This is a paper figure (`docs/figures/dilution-probe.png`), not a homepage widget.

Class-threshold sweep on the **same frozen predictions** (no retune). Operational LEO reaction is nearer `−4` to `−5`; ESA scored `−6` to have enough positives. False-reassurance analogue: accepted forecast under the existing `−6` abstention mask with `pred < t` while `y ≥ t`.

| `t` | *n*+ | Persist *F*<sub>2</sub> | Persist FR | Persist *L* | Ensemble *F*<sub>2</sub> | Ensemble FR | Ensemble *L* |
|---|---:|---:|---:|---:|---:|---:|---:|
| −8 | 68 | 0.660 | 2 | 1.66 | 0.210 | 5 | 9.49 |
| −7 | 37 | 0.502 | 3 | 0.777 | 0.215 | 4 | 2.51 |
| −6 | 9 | 0.361 | 1 | **0.167** | 0.361 | 1 | **0.167** |
| −5 | 3 | 0.349 | 0 | **0.032** | 0.349 | 0 | **0.032** |
| −4 | 1 | 0 | 1 | — | 0 | 1 | — |

Unguarded XGBoost, residual reconstruction, and the floor hurdle have *F*<sub>2</sub> = 0 from `−7` through `−4`. The exhibit has no threshold picker; this table lives in the paper and `metrics.json`.

## Research question

Conjunction estimates move as later observations refine orbital geometry, covariance, and observational uncertainty. Waiting usually improves the estimate and leaves less time to plan.

PRISM freezes the clock 48 hours before closest approach and asks:

**How much useful predictive information about the eventual reported collision probability exists in the CDM history available at T−48, beyond simply carrying the latest reported value forward?**

PRISM therefore studies forecasting under a fixed information constraint. It does not predict manoeuvres or operational collision outcomes. The target is the later reported `log10(Pc)`, not a physical collision probability.

T−48 is the ESA challenge information cutoff (`time_to_tca ≥ 2` days). The pipeline also reports T−72, T−24, and T−12.

The contribution is a controlled evaluation of whether historical CDM evolution provides useful early-warning signal at T−48, including horizon analysis, selective prediction, failure analysis, and mission-held-out testing.

## Findings

### What the model can do

1. **Average accuracy and decision quality are not the same thing, and mean error is not typical error.** A large MAE gain can coexist with an unchanged ESA-style score because that loss cares about the `log10(Pc) ≥ −6` tail. Floor-excluded MAE is worse than persistence, and Wilcoxon tests on paired `|error|` do not reject equality. The exact F2 and ESA-style loss tie is the persistence guard working, not a scoring bug.

2. **History provides a measurable gain, while covariance trends add little after that.** The history block consists of temporal transforms of variables already available in the latest snapshot, allowing the ablation to isolate information from their evolution. Snapshot also includes encounter-plane geometry and object-type dummies. On the single XGBoost model, snapshot-only MAE is 2.904. Adding historical summaries of risk, miss distance, speed, and observation counts lowers it to 2.851. Covariance trends after that reach 2.808.

3. **The value of learned forecasting is highest when information is sparse.** Waiting helps persistence more than it helps PRISM: the learned advantage is largest at T−72 and nearly gone at T−12. Horizon numbers below are single XGBoost (the T−48 row of the selected ensemble is 3.059).

   | Horizon | XGBoost | Persistence |
   |---|---:|---:|
   | T−72 | 3.214 | 7.748 |
   | T−48 | 2.808 | 5.080 |
   | T−24 | 2.110 | 2.634 |
   | T−12 | 1.384 | 1.444 |

4. **Abstention is selective prediction.** The `−6` class follows the ESA challenge definition. The persistence guard and 1.25 disagreement threshold were fixed design choices before evaluating the test split. PRISM abstains if the 90% bootstrap band crosses `log10(Pc) ≥ −6`, if current risk or miss distance is missing, or if bootstrap disagreement exceeds 1.25 log-risk units. That keeps **77.7%** coverage (370 of 1,659 sent to review) and drops accepted MAE from 3.059 to **1.902**.

### Where it is weak

5. **False reassurance is not zero.** One accepted forecast stays below `−6` while the final report is `≥ −6` (1/9 high-risk test events). Eight of nine high-risk events are flagged or sent to review.

6. **Random-event performance is stronger than mission-held-out performance.** A four-mission hold-out remains weak on the rare high-risk tail (one held-out high-risk event; high-risk MAE 19.2). Adding `mission_id` slightly worsens unguarded XGBoost MAE (2.808 → 2.840) and is excluded from the deployed exhibit.

7. **Ensemble disagreement is not equivalent to calibrated uncertainty.** Nominal 50% and 90% bootstrap bands cover 26.0% and 47.7% of outcomes. Split conformal covers 49.6% and 89.7%. The case UI still shows bootstrap ranges as model spread. Conformal numbers live on the research surface (paper, laboratory, `metrics.json`).

8. **The high-risk estimate is based on a very small positive class.** Only 66 eligible events meet the ESA class `log10(Pc) ≥ −6`, including nine in the test split. Leave-one-high-risk-out: train residual XGBoost without that event, score it. Persistence is closer on **66 / 66** (mean `|error|` 1.21 vs 12.59). Treat the high-risk probability as a scarce-label fit, not an operational warning system. At operational cuts the local test has **3** events at `−5` and **1** at `−4`; every frozen system misses that one `−4` event.

9. **Failures are not one bucket.** Of 1,659 test events, 837 are accurate to 0.5 log units. Dominant errors: 366 over-predictions, 209 under-predictions, 97 moderate errors, 63 sparse-history errors, 39 floor collapses to −30, 26 close-approach errors, 20 false high-risk calls, and 2 late high-risk jumps.

## Experimental setup

**Target.** Forecast of the final reported `log10(Pc)` after the T−48 cutoff.

**Information constraint.** Features use only messages with `time_to_tca ≥` the cutoff. The later update is the label, never an input.

**Data.** 162,634 CDM rows → cutoff-safe event histories → 8,293 eligible events. The frozen bundle uses 3,731 training, 1,659 validation, 1,244 calibration, and 1,659 test events. Train, validation, calibration, and test are event-disjoint; all model and policy choices are frozen before the untouched test evaluation. Validation is not reused for test selection. Official-test `true_risk` is joined only after that freeze. Seeds 43–46 are extra grouped redraws in `metrics.json`, not extra apps.

**Model.** Event-level XGBoost on inspectable summaries. A sequence model is not used so temporal signals can be inspected directly. Inference is a CPU-only 10-model XGBoost ensemble on a few hundred tabular features; no GPU is required.

**Baselines.** Persistence, training-set median, and Ridge. The exhibit’s selected policy is a ten-model bootstrap XGBoost median with a persistence guard when the current report is already at or above `−6`.

**License.** The PRISM code in this repository is MIT-licensed. The ESA Collision Avoidance Challenge dataset remains under ESA’s terms.

## Limitations

- Persistence remains competitive under the loss that motivated the original challenge. On official test it matches Uriot LRP (*L* = 0.694); sesc (*L* = 0.556) is not beaten.
- Bootstrap bands are model spread (47.7% at a 90% label). Split-conformal 90% intervals cover 89.7% of test outcomes and are the calibrated uncertainty claim. The exhibit still uses bootstrap spread for the on-screen band.
- One accepted high-risk miss remains on this split.
- Mission-level generalization, especially on high-risk events, is not established.
- The dataset is historical anonymized ESA-supported events from 2015–2019, not live catalogue data.
- Manoeuvre decisions are out of scope.
- The website requires a running FastAPI process. There is no silent JSON fallback.

## Exhibit

Six frozen real-data cases (two low, one review, three high). Each shows the current report, the T−48 forecast of the final reported `log10(Pc)`, model-spread bands, a calibrated estimate of high-risk-event probability based on a very small positive class, SHAP factors, and a reveal-only later outcome.

The figures that carry the argument are [`docs/figures/forecast-horizon.png`](docs/figures/forecast-horizon.png), [`docs/figures/coverage-calibration.png`](docs/figures/coverage-calibration.png), [`docs/figures/official-test-esa.png`](docs/figures/official-test-esa.png), [`docs/figures/dilution-probe.png`](docs/figures/dilution-probe.png), [`docs/figures/abstention-coverage.png`](docs/figures/abstention-coverage.png), and [`docs/figures/shap-contrast.png`](docs/figures/shap-contrast.png).

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
- [`ml/artifacts/demo_cases.json`](ml/artifacts/demo_cases.json): six curated real-data cases. The API serves these; the website does not read them as a fallback.
- [`ml/artifacts/model_card.json`](ml/artifacts/model_card.json) and [`docs/model-card.md`](docs/model-card.md).
- [`docs/figures`](docs/figures): comparison, ablation, horizon, abstention, failure, and SHAP charts.
- [`paper/main.pdf`](paper/main.pdf): living paper
- [`data/processed/events.csv`](data/processed/events.csv): event-level feature table.

## Repository layout

- [`main.py`](main.py): project orchestrator.
- [`ml/src`](ml/src): acquisition, features, training, evaluation, experiments, explanations, inference.
- [`ml/artifacts`](ml/artifacts): frozen models and evidence.
- [`apps/api`](apps/api): FastAPI service (required for the exhibit).
- [`apps/web`](apps/web): Next.js exhibit (API-only).
- [`docs`](docs): model card, data guide, deploy guide, PRD audit, presentation scripts, figures, v2 plan.
- [`paper`](paper): living IEEE conference manuscript (`main.tex`, `references.bib`, `main.pdf`).
- [`prd.md`](prd.md): locked product specification, with an 18 August 2026 addendum.

## Acknowledgments and sources

PRISM was created by Arjun Vijay Prakash for City Montessori School, Kanpur Road Campus. The project uses the ESA Space Debris Office’s public Collision Avoidance Challenge dataset and challenge design described by Uriot et al. Context comes from [ISRO’s NETRA control centre](https://www.isro.gov.in/ISRO%20SSAControl%20Centre.html), [NASA CARA](https://www.nasa.gov/cara/), and the CCSDS Conjunction Data Message standard. These references do not imply endorsement by ESA, ISRO, NASA, ISTRAC, or CCSDS.
