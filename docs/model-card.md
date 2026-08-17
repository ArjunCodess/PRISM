# PRISM model card

**Version.** `prism-0.2.1`

**Research question.** Do pre-T−48 conjunction histories contain enough predictive signal to improve forecasts of later reported `log10(Pc)` over persistence?

**Intended use.** Research prototype for explainable conjunction-risk forecasting. The exhibit laptop runs Next.js plus FastAPI; the website has no JSON fallback.

**Out of scope.** Flight software, operational decision systems, spacecraft operations, autonomous manoeuvres, claims about specific real satellites, and live catalogue data without retraining.

**Threshold.** High-risk class is `log10(Pc) ≥ −6`. This is the ESA challenge scoring class, not an ISRO operational rule. Only 66 eligible events in the labeled archive meet it, including nine in the frozen test split. Because only nine test events are positive, this probability estimate should be treated as a scarce-label fit, not an operational warning system.

**Split.** Train, validation, calibration, and test are event-disjoint; all model and policy choices are frozen before the untouched test evaluation. Validation is not reused for test selection. A separate four-mission hold-out tests generalization beyond random event splits.

**Data.** 162,634 CDM rows → cutoff-safe event histories → 8,293 eligible events → 3,731 / 1,659 / 1,244 / 1,659 train / validation / calibration / test events. The official challenge test file verifies input compatibility because final labels are unavailable.

**Why T−48.** The ESA challenge test set contains only messages with `time_to_tca ≥ 2` days. PRISM inherits that information cutoff as the primary experiment and also reports T−72 / T−24 / T−12.

**Why XGBoost.** The claim is about inspectable event-level summaries of the T−48 history, not sequence modeling. XGBoost fits medium-sized, missing-valued tabular features, stays reproducible offline, and keeps SHAP attached to named quantities. Inference is a CPU-only 10-model ensemble that predicts final `log10(Pc)` directly. Snapshot features include encounter-plane geometry and object-type dummies. No GPU is required.

**Current result.** MAE is in `log10(Pc)` units. The selected T−48 ensemble reduces held-out MAE from 5.080 for persistence to 3.059, but both score 0.167 on ESA-style loss (high-risk MSE / F2, β=2) with F2 0.361. That exact tie is expected: the persistence guard copies the current report whenever it is already at or above `−6`. Unguarded single XGBoost is 2.808 MAE with F2 = 0. The MAE gain is continuous-risk accuracy, not a better risk-weighted decision score.

**History.** The history block consists of temporal transforms of variables already available in the latest snapshot, plus message count and recency. Snapshot-only MAE is 2.904. Adding those summaries lowers it to 2.851. Covariance trends add a little more (2.808).

**Horizons.** The value of learned forecasting is highest when information is sparse. T−72 / T−48 / T−24 / T−12 single-XGBoost MAE: 3.214 / 2.808 / 2.110 / 1.384, versus persistence 7.748 / 5.080 / 2.634 / 1.444.

**Abstention.** Coverage is 77.7% (370 of 1,659 abstained) and accepted MAE is 1.902, versus 3.059 on all test events. Eight of nine test high-risk events are flagged or sent to review.

**False reassurance.** An accepted forecast (no abstention) with predicted `log10(Pc) < −6` while the final reported value is `≥ −6`. There is **one** such case on this split.

**Uncertainty.** Ensemble disagreement is not equivalent to calibrated uncertainty. Nominal 50% and 90% bootstrap bands cover 25.8% and 47.7% of outcomes. They are shown as model spread, not predictive probability.

**Policy.** PRISM abstains when the 90% bootstrap band crosses `log10(Pc) ≥ −6`, when current risk or miss distance is missing, or when bootstrap disagreement exceeds 1.25 log-risk units. The persistence guard and 1.25 disagreement threshold were fixed design choices before evaluating the test split.

**Mission identity.** Adding `mission_id` slightly worsens unguarded XGBoost MAE (2.808 → 2.840) and is excluded from the deployed exhibit. Mission-held-out overall MAE is 2.688 versus persistence 4.843. High-risk MAE is 19.2 on one held-out high-risk event.

**Failures.** Tracking-completeness features often have higher mean |SHAP| among large errors than among accurate cases. This is an association in model attribution, not a physical cause.

**Human control.** Forecasts are advisory. Review required means a person must look. The model never commands a manoeuvre.

**License.** MIT for the PRISM code in this repository. The ESA dataset remains under ESA’s terms.

See `ml/artifacts/metrics.json` for the frozen numbers, including historical ablation, forecast horizons, abstention coverage, failure clusters, and SHAP contrast.
