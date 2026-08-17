# PRISM PRD conformance audit

Audited 18 August 2026 against `prd.md` version 1.0 plus the 18 August addendum. Automated checks prove repository behavior, while operational validation still requires the exhibit laptop and domain review.

## Ready in the repository

- The event builder enforces the T−48 cutoff, creates one row per event, and keeps event IDs disjoint across train, validation, calibration, and test splits.
- Persistence, median, Ridge, MAE residual XGBoost, collapse and warning classifiers, hurdle mix, calibrated estimate of high-risk-event probability, split-conformal bands, 10-model bootstrap disagreement, deterministic SHAP explanations, explicit abstention, serialized artifacts, and six curated cases covering the required scenario types are implemented.
- Snapshot features include encounter-plane geometry and object-type dummies. Model version is `prism-0.3.0`.
- The laboratory includes held-out baseline metrics, ESA-style loss, snapshot-versus-history ablation, forecast-horizon evaluation, abstention coverage, failure clusters, SHAP contrast, calibration, grouped importance, both under- and over-prediction galleries, missed high-risk events, false escalations, provenance, and limitations.
- FastAPI rejects post-cutoff messages, empty histories, invalid log-risk values, negative uncertainty, and unreasonable geometry or speed values.
- The Next.js app provides the event queue, case workspace, and model laboratory, loads cases and metrics from FastAPI only, and makes reveal-outcome the only route for post-T−48 truth. If the API is down, `app/error.tsx` surfaces the failure. There is no silent JSON fallback.
- Risk is encoded with words and symbols as well as colour. The app includes keyboard focus states, a skip link, a useful 404, responsive layouts, print styling, and a visible safety statement on prediction screens.
- The labeled ESA archive is downloaded and checksummed. The real-data pipeline reads all 162,634 rows, retains 8,293 realistic labeled events, verifies the official test schema, trains frozen artifacts, and generates evaluation figures.
- The evaluation bundle exports empirical 50% and 90% conformal coverage (82.0% / 91.4%), mission-ID ablation, a four-mission hold-out, and robustness slices by object type, history length, miss distance, radial uncertainty, and snapshot age.
- Root `main.py` performs verified download reuse, training, graph generation, artifact syncing, automated checks, production build, and coordinated API/web startup without requiring PowerShell scripts.
- `docs/presentation.md` contains complete 90-second, 3-minute, and 7-minute scripts with the frozen hurdle results and likely judge questions.
- The user explicitly superseded the PRD's dark mission-control visual language with a minimal editorial direction. The redesigned app keeps the required screens, risk semantics, monospaced telemetry, and accessibility behavior while removing the decorative console treatment.

## Blocking full PRD conformance

- **M3 / AI-improvement claim:** on the frozen real-data hold-out, the selected hurdle policy reduces MAE from 5.080 to 2.800 log-risk units but ties persistence on ESA-style loss at 0.167. The product correctly disables the claim that it beats persistence until both required metrics improve.
- **G6 / A8 / N7 (API-stopped offline):** the PRD asked the demo to work with the Python API stopped via JSON fallback. That fallback was removed on purpose. The laptop exhibit still works with Wi-Fi disabled **if FastAPI and Next are both local**. It does not work if the API process is stopped.
- **False reassurance:** the older exhibit claimed zero accepted high-risk misses. The hurdle split has two. That is documented, not hidden.
- **Exhibit operations:** a Wi-Fi-off rehearsal, sub-60-second cold-start measurement, sub-200-ms cached-load measurement, sub-2-second live-inference measurement, projector test at 1920×1080, and the 90-second backup recording require the presentation laptop and are not repository-verifiable.

## Release decision

The app runs on historical ESA challenge data with the hurdle residual policy. Software checks in the pipeline are expected to pass. It remains short of full original-PRD acceptance because M3 is not met, JSON-offline-with-API-stopped is intentionally unmet, and the physical-laptop exhibit checks and recording remain outstanding.
