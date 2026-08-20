# PRISM model card

**Research question.** Do pre-T−48 conjunction histories contain enough predictive signal to improve forecasts of later reported `log10(Pc)` over persistence?

**Intended use.** Research prototype for explainable conjunction-risk forecasting. The exhibit laptop runs Next.js plus FastAPI; the website has no JSON fallback.

**Out of scope.** Flight software, operational decision systems, spacecraft operations, autonomous manoeuvres, claims about specific real satellites, and live catalogue data without retraining.

**Threshold.** High-risk class is `log10(Pc) ≥ −6`. This is the ESA challenge scoring class, not an ISRO operational rule. Only 66 eligible events in the labeled archive meet it, including nine in the frozen test split. Because only nine test events are positive, this probability estimate should be treated as a scarce-label fit, not an operational warning system.

**Split.** Train, validation, calibration, and test are event-disjoint; all model and policy choices are frozen before the untouched test evaluation. Validation is not reused for test selection. A separate four-mission hold-out tests generalization beyond random event splits.

**Data.** 162,634 CDM rows → cutoff-safe event histories → 8,293 eligible events → 3,731 / 1,659 / 1,244 / 1,659 train / validation / calibration / test events. The public file has no calendar dates and no object-pair IDs. Official-test labels (Zenodo 4463683) are scored once after freeze.

**Why T−48.** The ESA challenge test set contains only messages with `time_to_tca ≥ 2` days. PRISM inherits that information cutoff as the primary experiment and also reports T−72 / T−24 / T−12.

**Why this model.** The claim is about inspectable event-level summaries of the T−48 history. The live policy is a floor hurdle chosen on validation: a classifier for later floor reports plus residual XGBoost on non-floor training events (threshold 0.15, no persist guard). SHAP is attached to the residual regressor. The 18 August 10-model bootstrap ensemble is a baseline snapshot, not a live mode.

**Current result.** (1) Here is the model: T−48 floor hurdle, conformal interval, SHAP. Test MAE 2.109 versus persistence 5.080; median AE 0; floor-excluded MAE 9.311; F2 = 0. (2) Here is what the study measured: horizon decay, floor anatomy, official-test *L*, and the August exhibit snapshot (ensemble MAE 3.059, ESA-style loss 0.167 / F2 0.361). The August numbers are a baseline row, not a live mode.

**History.** The history block consists of temporal transforms of variables already available in the latest snapshot, plus message count and recency. Snapshot-only MAE is 2.904. Adding those summaries lowers it to 2.851. Covariance trends add a little more (2.808).

**Horizons.** On a matched cohort of 1,459 test events, extra MAE beyond persistence is 4.49 / 2.20 / 0.43 / 0.03 at T−72 / T−48 / T−24 / T−12. Non-floor ΔMAE is negative at every horizon. The overlapping-set table is not the identification result.

**Abstention.** Live conformal coverage is 90.4% (159 of 1,659 abstained) and accepted MAE is 1.706. False reassurance is 2 of 9 high-risk test events.

**False reassurance.** An accepted forecast (no abstention) with predicted `log10(Pc) < −6` while the final reported value is `≥ −6`. Live count is **two** on this split.

**Uncertainty.** Live intervals are split conformal around the floor-hurdle point. August snapshot bootstrap 50% and 90% bands cover 25.8% and 47.7% of outcomes (model spread).

**Policy.** The live floor hurdle abstains when the 90% conformal band crosses `log10(Pc) ≥ −6` or a critical field is missing. Threshold 0.15 and no persist guard were frozen on validation. The August snapshot used bootstrap spread and a 1.25 disagreement cap.

**Mission identity.** Adding `mission_id` slightly worsens unguarded XGBoost MAE (2.808 → 2.840) and is excluded from the exhibit. A mission-grouped split with 37 held-out high-risk events prefers persistence on the tail (high-risk MAE 1.50 vs 11.9).

**Failures.** Tracking-completeness features often have higher mean |SHAP| among large errors than among accurate cases. This is an association in model attribution, not a physical cause.

**Human control.** Forecasts are advisory. Review required means a person must look. The model never commands a manoeuvre.

**License.** MIT for the PRISM code in this repository (https://github.com/ArjunCodess/PRISM). The ESA dataset remains under ESA’s terms. The target is later reported `log10(Pc)`, not physical collision probability. $-30$ is a reporting floor.

See `ml/artifacts/metrics.json` for the frozen numbers, including historical ablation, forecast horizons, abstention coverage, failure clusters, and SHAP contrast.
