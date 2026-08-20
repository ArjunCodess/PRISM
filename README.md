# PRISM

**Predicting final conjunction risk from information available 48 hours before closest approach.**

PRISM tests whether pre-T−48 conjunction histories contain enough predictive signal to improve forecasts of later reported `log10(Pc)` over persistence.

It is a research prototype for explainable conjunction-risk forecasting. **Not flight software. Not an operational decision system.**

The selected policy is a **T−48 floor hurdle**: it forecasts later reported `log10(Pc)` from cutoff-safe CDM summaries. A classifier may call a later collapse to the dataset floor `−30`; otherwise a residual XGBoost adjusts today's report. It does **not** copy today's report when that report is already at the ESA `−6` class. The exhibit shows that forecast, a conformal interval or REVIEW REQUIRED, and a SHAP explanation.

The living manuscript is IEEE conference format: [`paper/main.tex`](paper/main.tex) ([`paper/main.pdf`](paper/main.pdf)), with overflow in [`paper/supplement.pdf`](paper/supplement.pdf). Rebuild with `scripts/compile-paper.ps1`. Code: https://github.com/ArjunCodess/PRISM.

Honest metrics live in `ml/artifacts/metrics.json`, including a `reviewArmor` block (matched-cohort horizons, censoring sensitivity, mission-grouped split, floor-classifier calibration, simple floor baselines, conformal width, selective prediction, H4 partial effects). SI-style paper figures regenerate with `python ml/src/plots.py`.

## Result

Held-out performance on 1,659 untouched test events. 1,352 of them later report the dataset floor `−30`. Errors are in `log10(Pc)` units. ΔMAE is MAE(persistence) − MAE(model) with a 95% bootstrap interval from 1,000 event resamples. Wilcoxon *p* is two-sided on paired `|error|`.

| | Persistence | Unguarded XGBoost | Residual XGBoost | Floor hurdle (live) | August snapshot |
|---|---:|---:|---:|---:|---:|
| MAE | 5.080 | 2.809 | 2.760 | **2.109** | 3.059 |
| Median AE | **0.000** | 0.554 | 0.453 | **0.000** | 0.473 |
| Floor-excluded MAE | **4.073** | 7.562 | 7.375 | 9.311 | 7.332 |
| Floor-only MAE | 5.308 | 1.730 | 1.712 | **0.474** | 2.088 |
| Residual MAE (`|y − risk|` vs `|pred − risk|`) | 5.080 / 0.000 | 5.080 / 5.222 | 5.080 / 5.134 | 5.080 / 5.945 | 5.080 / 4.582 |
| ΔMAE vs persistence [95% CI] | — | 2.270 [1.930, 2.638] | 2.319 [1.977, 2.685] | 2.971 [2.524, 3.415] | 2.021 [1.716, 2.380] |
| Wilcoxon *p* | — | 0.91 | 0.92 | **&lt;10<sup>−33</sup>** | 0.31 |
| ESA-style loss | **0.167** | ∞ (F2 = 0) | ∞ (F2 = 0) | ∞ (F2 = 0) | **0.167** |

The mean-error improvement is dominated by later collapses to the reporting floor; it is not a typical-event win for unguarded or residual XGBoost. Floor-excluded MAE is worse than persistence, median AE for persistence is already 0, and Wilcoxon does not reject equal `|error|`. Across **five grouped redraws** (seeds 42–46; seed 42 is still the reported split), unguarded XGBoost MAE advantage is **2.34 ± 0.04** and floor-excluded MAE is **7.78 ± 0.21** (persistence floor-excluded **4.32 ± 0.23**). Residual reconstruction is **2.35 ± 0.04** MAE advantage. Predicting `y − risk` and adding the hat back to persistence is almost the same as unguarded level-valued XGBoost (test MAE 2.760 vs 2.809). A two-part floor model — `P(y = −30)` plus a residual regressor fit only on non-floor training events, with threshold `0.15` chosen on validation and no persistence guard — reaches test MAE **2.109** and median AE 0. Floor-class confusion on test is TP 1319 / FP 174 / FN 33 / TN 133 (recall 0.976, precision 0.883). Wilcoxon versus persistence is `p < 10^−33`, but floor-excluded MAE is 9.311: the win is still a floor-call statistic (H2). On **validation**, the floor hurdle ranks first by MAE (1.916 vs unguarded XGBoost 2.669). It is the live CDM policy (`selected_policy.json`). The 18 August bootstrap ensemble is an exhibit snapshot in the tables, not a second live mode. ESA-style loss is the challenge objective: high-risk MSE divided by F2 (F-beta with β=2). F2 is 0.361 for persistence and the August snapshot, and 0 for the unguarded, residual, and floor policies. The August tie is the persistence guard copying the current report whenever it is already at or above `−6`.

A constant training-set median scores 3.002 MAE.

PRISM’s live 90% interval is split conformal around the floor-hurdle point: test coverage **90.1%**, mean width **21.03**. The 50% conformal radius is 0 (most calibration scores are exact floor hits). Research tables also report the August snapshot: nominal 90% bootstrap coverage **47.7%** (50% band **26.0%**). Split-conformal 90% intervals around that snapshot cover **89.7%** (mean width 18.33).

On a **mission-grouped** split (`GroupShuffleSplit` by `mission_id`, seed 42, 20% of missions held out; 15 missions / 3,957 events train vs 4 missions / 4,336 events test; same unguarded XGBoost spec, floor hurdle not retuned), high-risk MAE is **11.9 versus 1.50** on 37 held-out high-risk events. An earlier four-mission draw had high-risk MAE 19.2 on **one** event and is not interpretable for the tail. The public archive has no calendar dates and no object-pair IDs.

The model contains useful signal beyond persistence on mean error, especially at longer forecast horizons, but that signal is concentrated in floor collapses and does not automatically translate into better risk-sensitive decisions or calibrated uncertainty.

Official-test labels (Zenodo 4463683, 25 Jan 2021) were scored once after freeze on 2,167 events (150 high-risk, 1,673 floor). Features come only from `test_data.csv`. Shared `event_id` integers with training are independent numbering (0 identical pre-cutoff snapshots). Clipped persistence ESA-style loss is **0.694**, matching Uriot last-risk-prediction on this distribution. The August ensemble snapshot **ties** that loss via the `−6` persist guard. Residual and floor policies lower MAE (3.476 / 3.177 vs 5.209) but raise *L* to **104** and **70.5** because *F*<sub>2</sub> collapses. None of the frozen models beat the selected challenge entry *sesc* (*L* = 0.556). These numbers are research results, not a website leaderboard, and were not used to retune.

| Official test | Persistence | Unguarded XGBoost | Residual XGBoost | Floor hurdle (live) | August snapshot |
|---|---:|---:|---:|---:|---:|
| MAE | 5.209 | 3.504 | 3.476 | **3.177** | 3.107 |
| Median AE | **0.000** | 0.690 | 0.594 | **0.000** | 0.439 |
| Floor-excluded MAE | **4.287** | 9.333 | 9.289 | 11.832 | 6.750 |
| ESA-style loss | **0.694** | ~1.75×10<sup>6</sup> (*F*<sub>2</sub>=0) | 104 | 70.5 | **0.694** |
| *F*<sub>2</sub> | **0.739** | 0.000 | 0.017 | 0.025 | **0.739** |

H4 (dilution / max-risk probe, logistic and Spearman only; not an extra exhibit model). On the frozen local test, `dilution_gap = max_risk_estimate − risk` has Spearman **ρ = −0.768** with `|y − risk|` and **ρ = +0.407** with the floor label. Combined covariance volume (`log_combined_sigma_det`) has **ρ = +0.399** with `|y − risk|`. A train-fit logistic of floor ~ gap + miss distance + `n_messages` has test AUC **0.819**. Large gaps are already-floor snapshots (Q1 floor rate 0.56 and mean `|Δrisk|` 13.03; Q3–Q4 floor rate > 0.95 and mean movement < 0.75). F10 is a weak control (ρ = 0.108 with `|Δrisk|`). The naive “large max-risk gap means the report will still move” story is rejected. Large covariance predicting more movement is supported. This is a paper figure (`docs/figures/dilution-probe.png`), not a homepage widget. Horizon decay with T−48 redraw whiskers is `docs/figures/horizon-decay.png`. Error anatomy (`y − risk` versus `y − pred`, with the −30 floor tail) is `docs/figures/error-anatomy.png`.

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

1. **Average accuracy and decision quality are not the same thing, and mean error is not typical error.** A large MAE gain can coexist with a worse ESA-style score because that loss cares about the `log10(Pc) ≥ −6` tail. Floor-excluded MAE is worse than persistence. The August F2 and ESA-style loss tie is the persistence guard working, not a scoring bug. The live floor hurdle has F2 = 0 on the local test.

2. **History provides a measurable gain, while covariance trends add little after that.** The history block consists of temporal transforms of variables already available in the latest snapshot, allowing the ablation to isolate information from their evolution. Snapshot also includes encounter-plane geometry and object-type dummies. On the single XGBoost model, snapshot-only MAE is 2.904. Adding historical summaries of risk, miss distance, speed, and observation counts lowers it to 2.851. Covariance trends after that reach 2.808.

3. **The value of learned forecasting is highest when information is sparse.** On a **matched cohort** of 1,459 test events observed at all four horizons, extra MAE beyond persistence is **4.49 / 2.20 / 0.43 / 0.03** at T−72 / T−48 / T−24 / T−12 (the 12-hour interval covers zero). Non-floor ΔMAE is **negative at every horizon**: off the floor the model never beats persistence in that cohort. The older overlapping-set table (4.53 / 2.27 / 0.52 / 0.06) mixed a changing eligible set with time-to-approach.

4. **Abstention is selective prediction.** The `−6` class follows the ESA challenge definition. The live floor hurdle abstains when the conformal band crosses `−6` or a field is missing. At 50–80% nominal coverage the radius is 0 (no abstention). At 88% / 90% / 95%, review counts are **80 / 159 / 166**, accepted MAE **1.812 / 1.706 / 1.689**, and false reassurance **5 / 2 / 2** of 9. The 90% operating point is the live rule. The August snapshot used bootstrap spread (77.7% coverage, accepted MAE 1.902, one false reassurance).

### Where it is weak

5. **False reassurance is not zero.** Two accepted live forecasts stay below `−6` while the final report is `≥ −6` (2/9 high-risk test events). Seven of nine high-risk events are sent to review.

6. **Random-event performance is stronger than mission-held-out performance on the tail.** A mission-grouped split with 37 held-out high-risk events still prefers persistence on high-risk MAE (1.50 vs 11.9). Adding `mission_id` as a feature slightly worsens unguarded XGBoost MAE (2.808 → 2.840) and is excluded from the exhibit.

7. **A conformal interval is not a tight 90% band.** Live intervals are split conformal around the floor-hurdle forecast. August snapshot bootstrap 50% and 90% bands cover 26.0% and 47.7% of outcomes. Snapshot conformal covers 49.6% and 89.7%. The case UI labels the live band as a conformal interval.

8. **The high-risk estimate is based on a very small positive class.** Only 66 eligible events meet the ESA class `log10(Pc) ≥ −6`, including nine in the test split. Leave-one-high-risk-out: train residual XGBoost without that event, score it. Persistence is closer on **66 / 66** (mean `|error|` 1.21 vs 12.59). Treat the high-risk probability as a scarce-label fit, not an operational warning system. At operational cuts the local test has **3** events at `−5` and **1** at `−4`; every frozen system misses that one `−4` event.

9. **Failures are not one bucket.** Of 1,659 test events, 837 are accurate to 0.5 log units. Dominant errors: 366 over-predictions, 209 under-predictions, 97 moderate errors, 63 sparse-history errors, 39 floor collapses to −30, 26 close-approach errors, 20 false high-risk calls, and 2 late high-risk jumps.

## Experimental setup

**Target.** Forecast of the final reported `log10(Pc)` after the T−48 cutoff.

**Information constraint.** Features use only messages with `time_to_tca ≥` the cutoff. The later update is the label, never an input.

**Data.** 162,634 CDM rows → cutoff-safe event histories → 8,293 eligible events. The frozen bundle uses 3,731 training, 1,659 validation, 1,244 calibration, and 1,659 test events. Train, validation, calibration, and test are event-disjoint; all model and policy choices are frozen before the untouched test evaluation. Validation is not reused for test selection. Official-test `true_risk` is joined only after that freeze. Seeds 43–46 are extra grouped redraws in `metrics.json`, not extra apps.

**Model.** Event-level XGBoost on inspectable summaries. Inference is the validation-selected floor hurdle (classifier + residual, threshold 0.15, no persist guard), CPU-only, with SHAP on the residual regressor.

**Baselines.** Persistence, training-set median, Ridge, unguarded XGBoost, and the 18 August bootstrap ensemble with a `−6` persistence guard (exhibit snapshot).

**License.** The PRISM code in this repository is MIT-licensed. The ESA Collision Avoidance Challenge dataset remains under ESA’s terms.

## Limitations

- Persistence remains competitive under the loss that motivated the original challenge. On official test it matches Uriot LRP (*L* = 0.694); sesc (*L* = 0.556) is not beaten.
- Bootstrap bands are model spread (47.7% at a 90% label). Split-conformal 90% intervals cover 89.7% of test outcomes and are the calibrated uncertainty claim. The exhibit still uses bootstrap spread for the on-screen band.
- One accepted high-risk miss remains on this split.
- Mission-level generalization, especially on high-risk events, is not established.
- The dataset is historical anonymized ESA-supported events from 2015–2019, not live catalogue data. There are no calendar dates or object-pair IDs in the public file. $-30$ is a reporting floor, not a precise physical $P_c$. The target is later reported $\log_{10} P_c$, not a collision occurrence.
- Manoeuvre decisions are out of scope.
- The website requires a running FastAPI process. There is no silent JSON fallback.

## Exhibit

Six frozen real-data cases (two low, one review, three high). Each shows the current report, the T−48 forecast of the final reported `log10(Pc)`, model-spread bands, a calibrated estimate of high-risk-event probability based on a very small positive class, SHAP factors, and a reveal-only later outcome.

The figures that carry the argument are [`docs/figures/horizon-decay.png`](docs/figures/horizon-decay.png), [`docs/figures/error-anatomy.png`](docs/figures/error-anatomy.png), [`docs/figures/coverage-calibration.png`](docs/figures/coverage-calibration.png), and [`docs/figures/dilution-probe.png`](docs/figures/dilution-probe.png). Optional official-test loss is [`docs/figures/official-test-esa.png`](docs/figures/official-test-esa.png). Regenerate them with `python ml/src/plots.py`. They are paper figures, not homepage widgets.

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
- [`docs/figures`](docs/figures): SI-style paper figures (`horizon-decay`, `error-anatomy`, `coverage-calibration`, `dilution-probe`) plus lab charts.
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
