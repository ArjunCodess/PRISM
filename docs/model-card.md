# PRISM model card

**Research question.** Do pre-T−48 conjunction histories contain enough predictive signal to improve forecasts of later reported `log10(Pc)` over persistence?

**Intended use.** Research prototype for offline, explainable conjunction-risk forecasting.

**Out of scope.** Flight software, operational decision systems, spacecraft operations, autonomous manoeuvres, claims about specific real satellites, and live catalogue data without retraining.

**Threshold.** High-risk class is `log10(Pc) ≥ −6`. This is the ESA challenge scoring class, not an ISRO operational rule. Only 66 eligible events in the labeled archive meet it, including nine in the frozen test split. Treat the calibrated estimate of high-risk-event probability as a fit based on a very small positive class.

**Split.** Grouped by `event_id` into train / validation / calibration / test. Calibration never sees test labels. A separate four-mission hold-out tests generalization beyond random event splits.

**Data.** 162,634 CDM rows → cutoff-safe event histories → 8,293 eligible events → 1,659 test events. The official challenge test file verifies input compatibility because final labels are unavailable.

**Why T−48.** The ESA challenge test set contains only messages with `time_to_tca ≥ 2` days. PRISM inherits that information cutoff as the primary experiment and also reports T−72 / T−24 / T−12.

**Why XGBoost.** The claim is about inspectable event-level summaries, not sequence modeling. XGBoost fits medium-sized, missing-valued tabular features, stays reproducible offline, and keeps SHAP attached to named quantities.

**Current result.** The selected conservative ensemble reduces held-out MAE from 5.080 for persistence to 3.052, but both score 0.167 on ESA-style loss. PRISM does not claim to beat persistence under the full acceptance rule.

**History.** History features encode recent change, slope, range, variability, and observation-count evolution across the pre-cutoff CDM sequence. Snapshot-only MAE is 2.960. Adding those summaries lowers it to 2.842. Covariance trends add little once snapshot and history features are present (2.838).

**Horizons.** Waiting helps persistence more than it helps PRISM. T−72 / T−48 / T−24 / T−12 single-XGBoost MAE: 3.241 / 2.838 / 2.097 / 1.390, versus persistence 7.748 / 5.080 / 2.634 / 1.444.

**Abstention.** The rule was locked from the ESA class and an a priori disagreement threshold, not from test outcomes. Coverage is 78.2% (21.8% abstention) and accepted MAE is 1.920, versus 3.052 on all test events. All nine test high-risk events are either flagged by the prediction or sent to review. Zero accepted false reassurance.

**Uncertainty.** Ensemble disagreement is not equivalent to calibrated uncertainty. Nominal 50% and 90% bootstrap bands cover 26.4% and 48.6% of outcomes.

**Policy.** PRISM abstains when the 90% bootstrap band crosses `log10(Pc) ≥ −6`, when current risk or miss distance is missing, or when bootstrap disagreement exceeds 1.25 log-risk units.

**Mission identity.** Adding `mission_id` provides negligible improvement on the single XGBoost model (2.838 → 2.835) and does not materially change performance, so it is excluded from production.

**Failures.** When the single XGBoost model is wrong by two or more log units, tracking-completeness features rise in mean |SHAP| relative to accurate cases.

**Human control.** Forecasts are advisory. Review required means a person must look. The model never commands a manoeuvre.

See `ml/artifacts/metrics.json` for the frozen numbers, including historical ablation, forecast horizons, abstention coverage, failure clusters, and SHAP contrast.
