# Remedy outcomes and unresolved questions

All ESA values here refer to the refitted seed-42 cohort in `ml/artifacts/fmts_review/esa_seed42.json`. Current-runtime refits are not the submitted frozen predictions.

The calibrated mixture mean improves atom Brier score from 0.1154 to 0.0876 and overall MAE to 2.735, but local recall is zero, non-floor MAE is 8.315, and ESA loss is infinite. Probability calibration does not make a conditional-mean forecast a good warning policy. Official high-risk recall is also zero.

The prior hurdle architecture, refitted and selected on validation MAE, chooses threshold 0.15. Local MAE is 2.188 and non-floor MAE 9.793, with zero high-risk recall. Outcome-category sets improve non-floor coverage, but send every local and official event to review. This is a calibration improvement with unusable selectivity, not an effective operational remedy.

The revised calibrated hurdle selected on validation ESA loss chooses threshold 0.95, full weight, and the current-report guard. Local non-floor MAE improves to 3.309, and official non-floor MAE to 3.192, with paired intervals excluding zero for model-minus-persistence differences. ESA loss ties persistence on both populations. Raw high-risk MAE does not establish improvement and is worse on the local point estimate. The guard explains the identical high-risk class calls; clipping explains why altered low forecasts do not change ESA's high-risk MSE.

Candidate-category conformal around direct regression raises official high-risk coverage from 24.0% to 93.3%, with 140/150 covered and exact 95% interval [88.1%,96.8%]. This application of known Mondrian calibration addresses measured category undercoverage, but does not create a new statistical guarantee. Sets around the selected policy review 30.0% of official events and still accept nine falsely reassuring forecasts. No claim of safe abstention is supported. Review-cost improvements assume perfect downstream review, and should be read as sensitivity proxies.

Gaussian Tobit in stationary known-clipping simulation at nominal atom mass 0.8 has mean MAE 0.338 versus direct regression 0.374. It is not evaluated on ESA because a latent censoring likelihood is unverified. That simulation has no high-risk positives, so it cannot support a decision-tail claim. Structural-atom simulation gives the decision hurdle worse mean MAE than direct regression at mass 0.8, 10.177 versus 5.207, but lower analyst-defined cost, 0.385 versus 1.401. No remedy uniformly dominates across mechanisms and objectives. Every simulation run, including undefined metrics, remains in the aggregate.

Unresolved questions are the original −30 computation/export rule, a calendar or object-pair validation dataset, a justified second empirical domain, author contribution details, prior AI-use history, and independent author verification. Source-code audits and modeling cannot manufacture these facts.
