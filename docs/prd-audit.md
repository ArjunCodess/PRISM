# PRISM PRD conformance audit

Audited 16 August 2026 against `prd.md` version 1.0. Automated checks prove repository behavior, while operational validation still requires the exhibit laptop and domain review.

## Ready in the repository

- The event builder enforces the T−48 cutoff, creates one row per event, and keeps event IDs disjoint across train, validation, calibration, and test splits.
- Persistence, median, Ridge, XGBoost, calibrated warning probability, 10-model bootstrap spread, deterministic SHAP explanations, abstention, serialized artifacts, and five required scenario types are implemented.
- FastAPI rejects post-cutoff messages, empty histories, invalid log-risk values, negative uncertainty, and unreasonable geometry or speed values.
- The Next.js app provides the event queue, case workspace, and model laboratory, uses local frozen JSON when the Python API is unavailable, and makes reveal-outcome the only request for post-T−48 truth.
- Risk is encoded with words and symbols as well as colour. The app includes keyboard focus states, a skip link, a useful 404, responsive layouts, print styling, and a visible safety statement on prediction screens.
- The laboratory includes held-out baseline metrics, ESA-style loss, calibration, grouped importance, ablation, both under- and over-prediction galleries, missed high-risk events, false escalations, provenance, and limitations.
- The labeled ESA archive is downloaded and checksummed. The real-data pipeline reads all 162,634 rows, retains 8,293 realistic labeled events, verifies the official test schema, trains frozen artifacts, and generates evaluation figures.
- The evaluation bundle exports empirical 50% and 90% spread coverage, mission-ID ablation, a four-mission hold-out, and robustness slices by object type, history length, miss distance, radial uncertainty, and snapshot age.
- Root `main.py` performs verified download reuse, training, graph generation, artifact syncing, automated checks, production build, and coordinated API/web startup without requiring PowerShell scripts.

## Blocking full PRD conformance

- **M3 / AI-improvement claim:** on the frozen real-data hold-out, the selected conservative ensemble reduces MAE from 5.080 to 3.053 log-risk units but ties persistence on ESA-style loss at 0.167. The product correctly disables the claim that it beats persistence until both required metrics improve.
- **Exhibit operations:** a Wi-Fi-off rehearsal, sub-60-second cold-start measurement, sub-200-ms cached-load measurement, sub-2-second live-inference measurement, projector test at 1920×1080, and the 90-second backup recording require the presentation laptop and are not repository-verifiable.
- **Presentation assets:** `docs/presentation.md` contains the talk structure, but the 3-minute and 7-minute sections are prompts rather than fully rehearsable scripts.

## Release decision

The app now runs on historical ESA challenge data and all software checks pass. It remains short of full PRD acceptance because M3 is not met and the physical-laptop exhibit checks and recording remain outstanding.
