# Completion audit

This records what was completed and what requires unavailable evidence. A documented limitation is not a claim that the missing result was obtained. The intended destination is aerospace; the revision does not meet the two-real-domain requirement for a general machine-learning method paper.

| Requested work | Status and evidence |
|---|---|
| Clean initial tree, exact submission, preserved reference, revision branch | Complete. Submission `091d7c49fa74d28405c4cb1c9cfff4b9ab793712`, annotated tag `fmts-2026-submitted`, branch `fmts-review-improvements`. Submitted package unchanged. |
| Every supplied chair/reviewer concern | Complete for the supplied rejection summary. `FMTS_REVIEW_RESPONSE_MATRIX.md`; original individual reviews were not supplied. |
| Literature audit and related-work matrix | Complete. Detection/reporting limits, Tobit, two-part and zero-inflated models, tail evaluation, decision-focused prediction, and conformal coverage are attributed in `FMTS_RELATED_WORK.md`. DOI/publisher checks are saved. |
| Establish what −30 means | Investigation complete; generation semantics unresolved. Full raw release and documentation audited. `FMTS_ARCHIVE_SEMANTICS.md` records hypotheses separately from evidence. Producer questions below identify what would resolve this. |
| Unsupported novelty, TSFM and scaling claims | Removed from the active paper. `FMTS_CLAIM_AUDIT.md` gives old and revised claims. No TSFM or scaling evidence is claimed. |
| Credible remedies and all requested baseline families | Complete. Persistence, direct, residual, refitted prior hurdle, calibrated mixture, guarded decision-selected hurdle, and conformal comparisons. Standard Tobit is evaluated only under known simulation clipping. |
| Primary operational metrics | Complete. Official ESA loss leads point evaluation. Strata, tail precision/recall, calibration, explicit decision-cost proxy, review rate and uncertainty coverage accompany it. This proxy does not validate manoeuvre utility. |
| Paired uncertainty | Complete. Paired event bootstrap intervals and exact binomial coverage intervals, including undefined draws and absent tail groups, are saved. |
| Keep all messages of an event in one split | Complete. Tests and saved manifests enforce event integrity and original seed-42 IDs. |
| Event-grouped chronological validation | Complete in controlled simulation. ESA calendar validation remains unavailable because timestamps and object-pair IDs are absent in the public release. Random IDs are not used as time. |
| Second dataset or justified simulation fallback | Simulation fallback complete: 60 controlled runs cover known clipping, structural atoms, missing-update conventions and drift. No second empirical domain is claimed. UCI/EPA candidates were assessed without inventing target semantics. |
| Error decomposition and separate tail figure | Complete. Generated figure decomposes total error additively and presents challenge-tail MSE separately. |
| Abstract, positioning and limitations | Complete. Aerospace benchmark positioning, known methods, unresolved archive issue, measured results and failure boundaries appear in the active paper. |
| Data availability, ethics, funding and conflicts | Complete. ESA CC BY 4.0, code/simulation MIT, sole author Arjun Vijay Prakash, no funding or conflicts as confirmed by the author. |
| Detailed human contribution roles and earlier AI use | Awaiting author confirmation. These cannot be inferred from sole authorship. Current Codex assistance is explicitly disclosed. |
| Independent author verification and approval | Awaiting the author. Automated numerical checks do not constitute human verification. |
| Fixed configurations, seeds and machine-readable outputs | Complete. Five ESA runs and 60 simulations were rerun before commits; numeric predictions, splits, runtime versions and outputs are retained. |
| Negative results | Complete. `FMTS_NEGATIVE_REMEDIES.md` and all reports retain failed remedies, false reassurance, universal review and undefined losses. |
| Requested integrity, floor, decomposition, coverage and table tests | Complete. Full-suite release tests additionally recompute every model score and stratum coverage across all 65 saved runs, verify prediction event IDs, all report hashes, and final manuscript/PDF hashes. |
| Build and new PDFs | Complete. Web production build, frontend tests/lint, Python tests/lint and both PDFs pass. `FMTS_VALIDATION.md` records commands and failures. Installed MiKTeX builds the PDFs; native editor initialization remains unavailable. |
| Small commits with short lowercase messages | Complete on the revision branch. Source audit, methods, results by mechanism, validation, manuscripts, and documentation are separate commits. |
| No push or pull request | Respected. `FMTS_PR_DESCRIPTION.md` contains the prepared title and body. |

## Evidence needed from the data producers

These questions are prepared for a later authorized enquiry; no message has been sent.

1. Which original collision-probability algorithm and version generated the released `risk` field, and which numerical value is returned for an exact zero, underflow or failed computation?
2. Is −30 introduced before or after the logarithm? Please provide the exact clipping, replacement or export rule, including whether values below 10^-30 are retained anywhere.
3. Can −30 encode a missing or stale update, invalid covariance, algorithm failure, or another quality flag? If so, how can those cases be identified in the public columns?
4. Does “negligible” identify an operational archive category or a computed probability? Which decisions, if any, use that category?
5. Can privacy-preserving absolute event dates and stable object-pair groups be released for temporal and repeated-pair validation, with a documented license?

The additional official-source check revisited [Kelvins data documentation](https://kelvins.esa.int/collision-avoidance-challenge/data/) and the [ESA challenge workshop](https://indico.esa.int/event/385/). They explain the delayed target, selected test population and mission context, but do not supply the −30 generation rule. An operational slide search result was not adopted as evidence: the linked UNOOSA PDF returned 404 during retrieval. Neither a dashboard screenshot nor another model could establish the challenge archive's transformation rule.
