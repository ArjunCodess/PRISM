# PRISM presentation scripts

Use the abstention case for the 90-second path and the laboratory for longer questions. Keep the language exactly as written around thresholds and results: `−6` is the ESA challenge class, the ranges are model spread, and the model lowers MAE but ties persistence on ESA loss.

## 90-second script

“Collision-risk estimates can change as new tracking observations arrive, but an operator still needs time to prepare. PRISM asks a narrow question: do the conjunction messages available 48 hours before closest approach contain enough information to forecast the final reported `log10(Pc)` better than carrying the latest value forward?

These five cases come from the real ESA Collision Avoidance Challenge archive. I’ll open the abstention case. The page is frozen before T−48. It shows the current log collision risk, the forecast of the final reported value, the calibrated estimate of high-risk-event probability based on a very small positive class, and the spread across ten bootstrap models. PRISM abstains when that spread crosses the ESA class `log10(Pc) ≥ −6`, when a critical field is missing, or when the models disagree by more than 1.25 log units—so it asks for human review.

The explanation shows which cutoff-safe features moved this model’s result. These are SHAP contributions, so they explain model behavior rather than physical causation. The future is still hidden. When I reveal it, later messages and the final reported value appear.

In the laboratory, the ensemble lowers held-out mean absolute error from 5.080 to 3.052, but it ties persistence on ESA-style high-risk loss. Nominal 90% bootstrap bands cover only 48.6% of outcomes, so ensemble spread is not calibrated uncertainty. I therefore do not claim that AI wins every required metric. PRISM is a research prototype for offline, explainable forecasting; it is not flight software, and it never commands a manoeuvre.”

## 3-minute script

“Spacecraft conjunction alerts describe a possible close approach between two objects. The reported collision probability can rise or fall as new observations change the projected geometry and covariance. Waiting gives better information, but less time to plan. PRISM forecasts how serious an event’s last reported risk will look while stopping the clock two days early.

The data are the public ESA Collision Avoidance Challenge archive from 2015–2019. The original training ZIP contains 162,634 CDM rows across 13,154 events. PRISM keeps 8,293 events that have both a usable pre-T−48 history and a later labeled outcome: 162,634 rows → event histories → 8,293 eligible events → 1,659 test events. The official challenge test CSV has no final labels, so I use it to verify input compatibility, not to claim model quality.

The leakage rule is simple and tested: every feature message has `time_to_tca ≥ 2 days`; the target comes from a later final update; and train, validation, calibration, and test contain different event IDs. The feature table summarizes the latest safe snapshot plus changes in reported risk, miss distance, covariance, observations, and encounter geometry.

The learned model is XGBoost on event-level summaries rather than a sequence model, so temporal signals can be inspected directly and the exhibit stays reproducible offline. I compare it with a constant median, Ridge, and the strongest simple baseline: persistence, which copies the latest pre-T−48 risk forward. Ten bootstrap XGBoost models provide a point forecast and model spread. A separate calibration split maps the forecast to the probability of the ESA challenge high-risk class, `log10(Pc) ≥ −6`. That line is a competition scoring class, not an ISRO operational rule. Only 66 eligible events meet it, including nine in the test split.

[Open the abstention case.]

This page contains only cutoff-safe messages. The forecast is shown next to persistence, the 50% and 90% model-spread bands, geometry, observations, and a SHAP explanation. Because the wide band crosses the configured class, the system abstains and says ‘Review required.’ I can compare the learned forecast with persistence in one click, then reveal the final update. Reveal is the only route that loads post-T−48 truth.

[Open Model lab.]

On 1,659 untouched test events, the selected guarded ensemble reduces MAE from 5.080 to 3.052, or 39.9%. Both it and persistence score 0.167 on ESA-style loss, so the full PRD requirement to beat persistence is not met. Waiting helps persistence more than it helps PRISM: the learned advantage is largest at T−72 and nearly gone at T−12. The model-spread bands also under-cover: the nominal 90% band contains only 48.6% of outcomes. The interface calls it model spread rather than a calibrated interval.

The failure gallery keeps the worst misses visible, and a four-mission holdout shows that the rare high-risk tail remains difficult. The honest conclusion is that learned trends improve average error, while persistence remains essential for high-risk safety. PRISM is a research exhibit, not flight software, and every recommendation remains a request for human review.”

## 7-minute script

“Space surveillance systems repeatedly update close-approach estimates as radar and optical observations improve the two objects’ orbits. A small miss distance is not enough to infer collision probability because the uncertainty region can be large, shifted, or shrinking. Operators need to read projected separation, covariance, and probability together. They also need time: late estimates are more accurate, but leave less time for analysis and contingency planning.

PRISM addresses the forecasting problem posed by ESA’s Collision Avoidance Challenge. For each event, it predicts the final reported `log10(Pc)` using only messages available at least 48 hours before closest approach. It also returns a configured high-risk warning probability, model spread, an abstention state, and a SHAP explanation. It forecasts a later reported risk; it does not recompute collision probability from orbital mechanics and does not identify live objects.

The source is the real ESA training archive: 162,634 rows, 13,154 anonymized events, and 103 columns. After applying the local evaluation contract, 8,293 events remain. The official test input is cutoff-safe but unlabeled, so it verifies schema compatibility only. Every local download is tied to its ESA URL, byte count, timestamp, and SHA-256.

Leakage prevention drives the pipeline. First, feature histories include only rows where `time_to_tca ≥ 2`. Second, the final target is never an engineered input. Third, event IDs are disjoint across 3,731 training, 1,659 validation, 1,244 calibration, and 1,659 test events. Automated tests enforce those rules. A stricter robustness experiment also holds out all events from four missions.

One event row contains the last safe risk report, miss distance, relative speed, radial/tangential/normal state and covariance terms, observation counts, missingness, update recency, and multi-message slopes and deltas. Raw covariance determinants are converted to log-domain trends to avoid numerical overflow. Mission identity is excluded from production because adding it provides negligible improvement (2.838 → 2.835) and does not materially change performance.

The baselines are a training-set median, Ridge regression, and persistence. Persistence matters because the latest report already carries strong physical processing; a learned model must add value beyond copying it. The primary regressor is seed-fixed XGBoost. The exhibit’s selected policy takes the median of ten bootstrapped XGBoost models, but preserves the current reported risk once that report is already at or above the `−6` challenge class. That conservative guard prevents the learned correction from erasing an existing high-risk signal.

A separate calibration partition fits the high-risk-event probability map without seeing test labels. The bootstrap distribution supplies 50% and 90% spread bands. Their empirical coverage is only 26.4% and 48.6%, so they are explicitly labeled model spread, not statistical confidence intervals. That is a headline finding: ensemble disagreement is not equivalent to calibrated uncertainty. PRISM abstains when the outer band crosses the class threshold, when critical fields are missing, or when model disagreement exceeds 1.25 log-risk units.

[Open the uncertain case.]

The initial payload contains no future messages or final outcome. The status strip puts current risk, forecast of the final reported `log10(Pc)`, both spread bands, and the calibrated estimate of high-risk-event probability together. That estimate is based on a very small positive class. The chart marks T−48 and hides the future region. The explanation ranks signed SHAP contributions and then renders deterministic plain language from those verified directions. SHAP tells us why this fitted model moved; it does not prove that a feature physically caused the encounter risk.

The geometry panel keeps miss distance next to uncertainty and observation count. The forecast/persistence control supports the scientific comparison. Reveal calls a dedicated route and releases the frozen later messages and final reported outcome. If the Python API is stopped, the complete curated flow still runs from versioned local JSON, which makes the exhibit independent of venue internet.

[Open the laboratory.]

On the untouched 1,659-event test split, persistence MAE is 5.080 and the selected ensemble MAE is 3.052, a 39.9% reduction. Median absolute error is 0.474 and 50.8% of events fall within half a log unit. The difficult result is the high-risk tail: both selected ensemble and persistence score 0.167 on the ESA-style loss with F2 of 0.361. The acceptance contract requires lower MAE and lower ESA loss, so I keep the AI-superiority claim disabled.

History provides a measurable gain, while covariance trends add little once snapshot and history features are present: snapshot-only MAE 2.960, snapshot plus history 2.842, then 2.838. Waiting helps persistence more than it helps PRISM: the learned advantage is largest at T−72 and nearly gone at T−12. Abstention is selective prediction: 78.2% coverage, 21.8% sent to review, accepted MAE 1.920. All nine test high-risk events are either flagged by the prediction or sent to review. The guard, the −6 class, and the 1.25 disagreement threshold were locked before test evaluation.

The raw XGBoost model has lower aggregate MAE than the selected ensemble, but it misses the configured high-risk class and produces a very large ESA loss. The persistence guard intentionally trades some average accuracy for the same high-risk behavior as the baseline. That tradeoff is visible rather than hidden.

Robustness slices report performance by object type, message count, miss distance, radial uncertainty, and snapshot age. Events with six or more messages have MAE 2.742, while sparse histories are worse. The four-mission holdout confirms that the model still generalizes poorly on rare high-risk outcomes. Failure tables show the largest under-predictions, over-predictions, missed high-risk events, and false escalations.

The defensible finding is narrow: cutoff-safe learned features reduce average forecast error on this historical archive, while they do not displace persistence for the high-risk objective. Bootstrap spread is not calibrated uncertainty, and mission-held-out high-risk generalization is weak. PRISM remains a research prototype for offline, explainable conjunction-risk forecasting. Not flight software. It never recommends firing thrusters, and any operational response requires qualified human judgment.”

## Likely questions

**Why XGBoost instead of a transformer?** The research claim is about whether inspectable historical summaries contain signal beyond the latest snapshot. The dataset is structured tabular history, contains extensive missingness, and has only 8,293 eligible examples. XGBoost keeps those features named and the exhibit reproducible offline. A sequence model should replace it only after winning on the same frozen splits.

**Did PRISM beat the ESA competition winners?** No. The official test labels remain hidden, so PRISM reports only its frozen local event split and does not compare itself with the 2019 leaderboard.

**Why is the MAE numerically large?** Many events jump between the dataset floor of `−30` and a finite final risk. A few such errors add tens of log units, which strongly affects the mean; median absolute error is therefore shown beside MAE.

**Is 48.6% coverage acceptable for a 90% interval?** No. That result proves the raw bootstrap spread is under-calibrated. The UI labels it model spread and uses it for abstention, while formal conformal calibration remains future work.

**Why keep persistence if the ensemble lowers MAE?** Persistence preserves existing high-risk reports better and ties the selected ensemble on ESA loss. Removing it would make the exhibit look better on average while weakening the safety-relevant tail.

**What still requires the presentation laptop?** A Wi-Fi-off rehearsal, projector check, measured start/load/inference timing on that exact machine, and recording the 90-second backup video.
