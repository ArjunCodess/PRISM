# PRISM

**Predicting final conjunction risk from information available 48 hours before closest approach.**

PRISM tests whether pre-T−48 conjunction histories contain enough predictive signal to improve forecasts of later reported `log10(Pc)` over persistence.

It is a research prototype for offline, explainable conjunction-risk forecasting. **Not flight software. Not an operational decision system.**

## Result

Held-out performance on 1,659 events:

| | Persistence | PRISM |
|---|---:|---:|
| MAE | 5.080 | **3.052** |
| ESA-style loss | **0.167** | **0.167** |

PRISM’s nominal 90% bootstrap band covers only **48.6%** of outcomes.

**Yes for MAE. No for ESA-style loss.** History adds signal, and the learned advantage is largest when information is sparse. The model is not universally trustworthy: abstention helps, while mission shift and rare high-risk cases remain weak.

The result is a demonstrable forecasting system whose limits remain visible.

## Research question

Conjunction estimates move as later observations refine orbital geometry, covariance, and observational uncertainty. Waiting usually improves the estimate and leaves less time to plan.

PRISM freezes the clock 48 hours before closest approach and asks:

**How much useful predictive information about the eventual reported collision probability exists in the CDM history available at T−48, beyond simply carrying the latest reported value forward?**

PRISM therefore studies forecasting under a fixed information constraint. It does not predict manoeuvres or operational collision outcomes. The target is the later reported `log10(Pc)`, not a physical collision probability.

T−48 is the ESA challenge information cutoff (`time_to_tca ≥ 2` days). The pipeline also reports T−72, T−24, and T−12. History features encode recent change, slope, range, variability, and observation-count evolution across the pre-cutoff CDM sequence.

## Findings

1. **Average accuracy and decision quality are not the same thing.** The guarded ensemble reduces MAE from 5.080 to 3.052 (39.9%) and still ties persistence at ESA-style loss 0.167, F2 0.361.

2. **History provides a measurable gain, while covariance trends add little marginal information once snapshot and history features are present.** On the single XGBoost model, snapshot-only MAE is 2.960. Adding historical summaries of risk, miss distance, speed, and observation counts lowers it to 2.842. Covariance trends after that add almost nothing (2.838).

3. **Waiting helps persistence more than it helps PRISM.** The value of learned forecasting is highest when information is sparse:

   | Horizon | XGBoost | Persistence |
   |---|---:|---:|
   | T−72 | 3.241 | 7.748 |
   | T−48 | 2.838 | 5.080 |
   | T−24 | 2.097 | 2.634 |
   | T−12 | 1.390 | 1.444 |

4. **Abstention is selective prediction.** The rule was locked from the ESA class and a priori disagreement threshold, not from test outcomes: abstain if the 90% bootstrap band crosses `log10(Pc) ≥ −6`, if current risk or miss distance is missing, or if bootstrap disagreement exceeds 1.25 log-risk units. That keeps 78.2% coverage (21.8% abstention) and drops accepted MAE from 3.052 to 1.920. All nine test high-risk events are either flagged by the prediction or sent to review. There is no accepted false reassurance.

5. **Random-event performance is stronger than mission-held-out performance.** A four-mission hold-out remains weak on the rare high-risk tail (one held-out high-risk event; high-risk MAE 18.1). Adding `mission_id` provides negligible improvement (2.838 → 2.835) and does not materially change performance, so it is excluded from production.

6. **Ensemble disagreement is not equivalent to calibrated uncertainty.** Nominal 50% and 90% bootstrap bands cover 26.4% and 48.6% of outcomes. The interface labels them as model spread.

7. **The high-risk estimate is based on a very small positive class.** Only 66 eligible events meet the ESA class `log10(Pc) ≥ −6`, including nine in the test split. Treat the calibrated estimate of high-risk-event probability as a scarce-label fit, not an operational warning system.

8. **Failures are not one bucket.** Inaccurate forecasts cluster into over-prediction (362), under-prediction (210), sparse-history errors (62), and 40 events where the final reported risk collapses to the dataset floor of −30. Only two test events are late high-risk jumps. When the single XGBoost model is wrong by two or more log units, tracking-completeness features rise in mean |SHAP| relative to accurate cases.

## Experimental setup

**Target.** Forecast of the final reported `log10(Pc)` after the cutoff.

**Information constraint.** Features use only messages with `time_to_tca ≥` the cutoff. The later update is the label, never an input.

**Data.** 162,634 CDM rows → cutoff-safe event histories → 8,293 eligible events → 1,659 test events. Event IDs are disjoint across train, validation, calibration, and test.

**Model.** Event-level XGBoost on inspectable summaries. A sequence model is not used so temporal signals can be inspected directly and the exhibit stays reproducible offline.

**Baselines.** Persistence, training-set median, and Ridge. The exhibit’s selected policy is a ten-model bootstrap XGBoost median with a persistence guard when the current report is already at or above `−6`. The guard, the `−6` class, and the 1.25 disagreement threshold were locked from the ESA challenge definition and design choices. They were not tuned on test outcomes.

## Limitations

- Persistence remains competitive under the loss that motivated the original challenge.
- Bootstrap intervals are miscalibrated; they are used for abstention, not as 90% probability statements.
- Mission-level generalization, especially on high-risk events, is not established.
- The dataset is historical anonymized ESA-supported events from 2015–2019, not live catalogue data.
- Manoeuvre decisions are out of scope.

## Exhibit

Five frozen real-data cases: easy correct, hard correct, de-escalation, abstention, and confident failure. Each shows the current report, the forecast of the final reported `log10(Pc)`, model-spread bands, a calibrated estimate of high-risk-event probability based on a very small positive class, SHAP factors, and a reveal-only later outcome.

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

```powershell
python main.py --build-only
python main.py --skip-download --skip-train --skip-graphs --skip-checks
python main.py --force-download --build-only
python main.py --source synthetic --synthetic-events 420 --build-only
```

Stage-by-stage notes are in [`docs/data-guide.md`](docs/data-guide.md).

## Outputs

- [`ml/artifacts/metrics.json`](ml/artifacts/metrics.json): metrics, ablations, horizons, abstention, calibration, coverage, robustness, mission tests, failure clusters, SHAP contrast.
- [`ml/artifacts/demo_cases.json`](ml/artifacts/demo_cases.json): five curated real-data cases.
- [`ml/artifacts/model_card.json`](ml/artifacts/model_card.json) and [`docs/model-card.md`](docs/model-card.md).
- [`docs/figures`](docs/figures): comparison, ablation, horizon, abstention, failure, and SHAP charts.
- [`data/processed/events.csv`](data/processed/events.csv): event-level feature table.

## Latest full run

The frozen bundle was produced on 16 August 2026 with `python main.py --build-only`. It used 3,731 training, 1,659 validation, 1,244 calibration, and 1,659 test events.

## Repository layout

- [`main.py`](main.py): project orchestrator.
- [`ml/src`](ml/src): acquisition, features, training, evaluation, experiments, explanations, inference.
- [`ml/artifacts`](ml/artifacts): frozen models and evidence.
- [`apps/api`](apps/api): FastAPI service.
- [`apps/web`](apps/web): Next.js exhibit with an offline artifact fallback.
- [`docs`](docs): model card, data guide, PRD audit, presentation scripts, figures.
- [`prd.md`](prd.md): locked product specification.

## Acknowledgments and sources

PRISM was created by Arjun Vijay Prakash for City Montessori School, Kanpur Road Campus. The project uses the ESA Space Debris Office’s public Collision Avoidance Challenge dataset and challenge design described by Uriot et al. Context comes from [ISRO’s NETRA control centre](https://www.isro.gov.in/ISRO%20SSAControl%20Centre.html), [NASA CARA](https://www.nasa.gov/cara/), and the CCSDS Conjunction Data Message standard. These references do not imply endorsement by ESA, ISRO, NASA, ISTRAC, or CCSDS.
