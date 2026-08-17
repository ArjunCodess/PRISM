# PRISM model card

**Version.** `prism-0.3.0`

**Research question.** Do pre-T−48 conjunction histories contain enough predictive signal to improve forecasts of later reported `log10(Pc)` over persistence?

**Intended use.** Research prototype for explainable conjunction-risk forecasting. The exhibit laptop runs Next.js plus FastAPI; the website has no JSON fallback.

**Out of scope.** Flight software, operational decision systems, spacecraft operations, autonomous manoeuvres, claims about specific real satellites, and live catalogue data without retraining.

**Threshold.** High-risk class is `log10(Pc) ≥ −6`. This is the ESA challenge scoring class, not an ISRO operational rule. Only 66 eligible events in the labeled archive meet it, including nine in the frozen test split. Because only nine test events are positive, this probability estimate should be treated as a scarce-label fit, not an operational warning system.

**Split.** Train, validation, calibration, and test are event-disjoint; all model and policy choices are frozen before the untouched test evaluation. Validation is not reused for test selection. A separate four-mission hold-out tests generalization beyond random event splits.

**Data.** 162,634 CDM rows → cutoff-safe event histories → 8,293 eligible events → 3,731 / 1,659 / 1,244 / 1,659 train / validation / calibration / test events. The official challenge test file verifies input compatibility because final labels are unavailable.

**Why T−48.** The ESA challenge test set contains only messages with `time_to_tca ≥ 2` days. PRISM inherits that information cutoff as the primary experiment and also reports T−72 / T−24 / T−12.

**Why a hurdle residual.** Most labels sit at the dataset floor of −30 or near today's report. Predicting `y` directly wastes capacity on that floor. The selected policy trains XGBoost with `reg:absoluteerror` on `Δ = y − current risk`, mixes with a collapse-to-floor classifier (hard mix, threshold 0.35), copies today's report when it is already at or above `−6`, and clamps invented highs just below `−6` so ESA-style F2 stays tied with persistence. Snapshot features include encounter-plane geometry and object-type dummies. A sequence model is not used so named SHAP stays attached to inspectable summaries. Inference is CPU-only. SHAP explains the residual booster, not the mixed ŷ.

**Current result.** MAE is in `log10(Pc)` units. The selected hurdle policy reduces held-out MAE from 5.080 for persistence to 2.800 (44.9%), but both score 0.167 on ESA-style loss (high-risk MSE / F2, β=2) with F2 0.361. Unguarded MAE XGBoost is 2.550 with F2 = 0. The MAE gain is continuous-risk accuracy, not a better risk-weighted decision score. 76.9% of test events are within 0.5 log units.

**History.** The history block consists of temporal transforms of variables already available in the latest snapshot, plus message count and recency. On the MAE residual ablation, snapshot-only MAE is 2.509. Adding those summaries moves it to 2.535. Covariance trends are 2.526. History does not help average MAE under this loss.

**Horizons.** The value of learned forecasting is highest when information is sparse. T−72 / T−48 / T−24 / T−12 hurdle MAE: 4.578 / 2.800 / 2.199 / 1.385, versus persistence 7.748 / 5.080 / 2.634 / 1.444.

**Abstention.** Coverage is 88.97% (183 of 1,659 abstained) and accepted MAE is 2.085, versus 2.800 on all test events. Seven of nine test high-risk events are sent to review.

**False reassurance.** An accepted forecast (no abstention) with predicted `log10(Pc) < −6` while the final reported value is `≥ −6`. There are **two** such cases on this split (2/9 high-risk test events). They are late jumps.

**Uncertainty.** Intervals are split-conformal residuals on the calibration set, localized by whether the hurdle predicts a floor collapse. Nominal 50% and 90% bands cover 82.0% and 91.4% of outcomes. Bootstrap disagreement remains an abstention trigger.

**Policy.** PRISM abstains when the 90% conformal band crosses `log10(Pc) ≥ −6`, when current risk or miss distance is missing, when bootstrap disagreement exceeds 1.25 log-risk units, when the model forecasts the dataset floor while today's report is still far from negligible, or when the warning head is elevated while the point forecast stays below `−6`. The persistence guard and 1.25 disagreement threshold were fixed design choices before evaluating the test split.

**Mission identity.** Adding `mission_id` slightly worsens unguarded XGBoost MAE (2.550 → 2.570) and is excluded from the deployed exhibit. Mission-held-out overall MAE is 3.588 versus persistence 4.843. High-risk MAE is 0.114 on one held-out high-risk event (tied with persistence).

**Failures.** Dominant inaccurate modes: 135 floor collapses, 107 under-predictions, 41 over-predictions. Tracking-completeness and risk-trend features have higher mean |SHAP| among large errors than today's reported risk does. This is an association in model attribution, not a physical cause.

**Human control.** Forecasts are advisory. Review required means a person must look. The model never commands a manoeuvre.

**License.** MIT for the PRISM code in this repository. The ESA dataset remains under ESA’s terms.

See `ml/artifacts/metrics.json` for the frozen numbers, including historical ablation, forecast horizons, abstention coverage, failure clusters, and SHAP contrast.
