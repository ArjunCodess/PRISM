# PRISM model card

**Research question.** Do pre-T−48 conjunction histories contain enough predictive signal to improve forecasts of later reported `log10(Pc)` over persistence?

**Intended use.** Research prototype for offline, explainable conjunction-risk forecasting.

**Out of scope.** Flight software, operational decision systems, spacecraft operations, autonomous manoeuvres, claims about specific real satellites, and live catalogue data without retraining.

**Threshold.** High-risk class is `log10(Pc) ≥ −6`. This is the ESA challenge scoring class, not an ISRO operational rule. Only 66 eligible events in the labeled archive meet it, including nine in the frozen test split. Because only nine test events are positive, this probability estimate should be treated as a scarce-label fit, not an operational warning system.

**Split.** Train, validation, calibration, and test are event-disjoint; all model and policy choices are frozen before the untouched test evaluation. Validation is not reused for test selection. A separate four-mission hold-out tests generalization beyond random event splits.

**Data.** 162,634 CDM rows → cutoff-safe event histories → 8,293 eligible events → 3,731 / 1,659 / 1,244 / 1,659 train / validation / calibration / test events. The official challenge test file verifies input compatibility because final labels are unavailable.

**Why T−48.** The ESA challenge test set contains only messages with `time_to_tca ≥ 2` days. PRISM inherits that information cutoff as the primary experiment and also reports T−72 / T−24 / T−12.

**Why XGBoost.** The claim is about inspectable event-level summaries, not sequence modeling. XGBoost fits medium-sized, missing-valued tabular features, stays reproducible offline, and keeps SHAP attached to named quantities. Inference is a CPU-only 10-model ensemble; no GPU is required.

**Current result.** MAE is in `log10(Pc)` units. The selected conservative ensemble reduces held-out MAE from 5.080 for persistence to 3.052, but both score 0.167 on ESA-style loss (high-risk MSE / F2, β=2) with F2 0.361. That exact tie is expected: the persistence guard copies the current report whenever it is already at or above `−6`, so PRISM matches persistence on the high-risk tail by design. The MAE gain is therefore continuous-risk accuracy, not a better risk-weighted decision score.

**History.** The history block consists of temporal transforms of variables already available in the latest snapshot, plus message count and recency. Snapshot-only MAE is 2.960. Adding those summaries lowers it to 2.842. Covariance trends add little once snapshot and history features are present (2.838).

**Horizons.** The value of learned forecasting is highest when information is sparse. T−72 / T−48 / T−24 / T−12 single-XGBoost MAE: 3.241 / 2.838 / 2.097 / 1.390, versus persistence 7.748 / 5.080 / 2.634 / 1.444.

**Abstention.** The `−6` class follows the ESA challenge definition. The persistence guard and 1.25 disagreement threshold were fixed design choices before evaluating the test split. Coverage is 78.2% (21.8% abstention) and accepted MAE is 1.920, versus 3.052 on all test events. All nine test high-risk events are either flagged by the prediction or sent to review.

**False reassurance.** An accepted forecast (no abstention) with predicted `log10(Pc) < −6` while the final reported value is `≥ −6`. There are zero such cases on this split.

**Uncertainty.** Ensemble disagreement is not equivalent to calibrated uncertainty. Nominal 50% and 90% bootstrap bands cover 26.4% and 48.6% of outcomes. They are shown as model spread, not predictive probability.

**Policy.** PRISM abstains when the 90% bootstrap band crosses `log10(Pc) ≥ −6`, when current risk or miss distance is missing, or when bootstrap disagreement exceeds 1.25 log-risk units.

**Mission identity.** Adding `mission_id` provides negligible improvement on the single XGBoost model (2.838 → 2.835) and does not materially change performance, so it is excluded from the deployed exhibit. Mission-held-out high-risk MAE is 18.1 on one held-out high-risk event.

**Failures.** Tracking-completeness features have higher mean |SHAP| among errors of two or more log units than among accurate cases. This is an association in model attribution, not a physical cause.

**Human control.** Forecasts are advisory. Review required means a person must look. The model never commands a manoeuvre.

**License.** MIT for the PRISM code in this repository. The ESA dataset remains under ESA’s terms.

See `ml/artifacts/metrics.json` for the frozen numbers, including historical ablation, forecast horizons, abstention coverage, failure clusters, and SHAP contrast.
