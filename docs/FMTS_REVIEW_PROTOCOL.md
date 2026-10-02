# Review revision protocol

The clean submission commit is `091d7c49fa74d28405c4cb1c9cfff4b9ab793712`, preserved by annotated tag `fmts-2026-submitted`. Work is on `fmts-review-improvements`. The files under `submissions/fmts_2026` remain the submitted record.

This protocol and `experiments/fmts_review.json` were written before running the new remedy experiments. This is a retrospective revision of a previously inspected dataset, not a preregistered study or an untouched new test set. Published local and official test outcomes have already been seen. New choices use validation only; calibration fits uncertainty only. Test outcomes do not choose policies. Repeat seeds are sensitivity analyses, not independent replications.

## Target and model choice

Treat exact recorded −30 values as an archive atom. Do not infer a latent detection limit from the atom alone. Use an observed-target two-part model with sigmoid calibration fit on validation, an unconditional mixture mean, a hard hurdle, and a guarded hurdle selected on validation ESA loss. Include persistence, direct and residual XGBoost, and the prior MAE-selected hurdle. This implements established mixture and decision-aware selection methods, not a new estimator. Censored Gaussian regression is restricted to simulation with known clipping, because its likelihood assumes semantics not established for ESA.

Primary point metric is ESA high-risk MSE divided by F2, with predictions clipped by the published challenge rule and infinity when F2=0. Undefined denominators are stored as null with a reason, not silently stabilized. Report recall, precision, high-risk MSE, floor and non-floor MAE contributions, and paired percentile bootstrap intervals over events. A bootstrap draw lacking positives has no ESA loss; report its count. Diagnostic MAE does not select the revised policy.

Decision cost is a sensitivity proxy with FN=20, FP=1, and review=0.25 per event. These are explicit analyst assumptions, not ESA manoeuvre costs. Review has a cost but no modeled downstream error: this is optimistic about human review. Compare all models at the same costs. Report accepted fraction and false reassurance counts; coverage after selective acceptance has no conformal guarantee.

## Uncertainty

Use marginal split conformal and outcome-category Mondrian inversion on floor, non-floor low risk, and high risk. For each candidate value, use the quantile for that candidate's category. The returned prediction set is a union of the floor atom and up to two intervals; it can be disconnected or empty. Never use the unseen actual category to choose a radius. Finite-sample ranks include the n+1 correction; small or absent categories receive infinite radii. Guarantees require exchangeability within category, not arbitrary distribution shift or calendar ordering. Outcome categories are disjoint: exact −30, (−30,−6), and [−6,0]. Retrospective coverage uses the actual category only for scoring.

## Splits and generality

ESA uses event-disjoint splits and cutoff-safe temporal features. Random event IDs are not timestamps. A true calendar split cannot be manufactured; inspect the released original archive for recoverable timestamps. If unavailable, record the failure and leave calendar validation unresolved. The released official population is enriched for high risk, so transfer coverage is measured without claiming exchangeability. Mission grouping is complementary, not chronological validation.

A bounded second-data audit considers UCI Air Quality and EPA water-quality data. UCI −200 denotes missingness, not a detection floor, and is incompatible with the proposed censoring test. EPA documents legitimate censoring but variable laboratory limits, qualifiers, licenses, and repeated-station forecast construction need validation before modeling. No justified second empirical forecasting dataset has been established in this revision. Use controlled simulation with known clipping, a structural atom, and a missing-update sentinel, at two mass rates, with stationary and drifting mechanisms and five seeds. This supports mechanism testing only, not cross-domain empirical validation or an ML-destination claim.

Simulation includes four observations per event, an explicit event time, and ordered train/validation/calibration/test blocks of 45/20/15/20%. All observations of an event remain in its split. Covariates are generated before the target. The drifting version changes the latent target and atom propensity after training. Its conformal coverage is descriptive only. Gaussian Tobit is tested only in the known-clipping setting; its observed prediction is E[max(floor, latent)], not a latent mean passed off as the recorded target.

## Output integrity

Rebuild ESA features from the raw training zip, align to the frozen split, and write hash provenance. Keep new artifacts in `ml/artifacts/fmts_review`; do not replace deployed models or their frozen scores. Emit per-event predictions, split IDs, model parameters, probability and set diagnostics, bootstrap intervals, JSON metrics, and generated manuscript tables. Assertions check event disjointness and additive error decomposition. Save negative remedy results. Remove all model-scaling and foundation-model efficacy claims: no compatible pretrained TSFM is evaluated.
