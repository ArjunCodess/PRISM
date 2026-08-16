# PRISM model card

**Intended use.** Education, interpretability demonstration, and research into early conjunction-risk forecasting. Educational prototype only.

**Out of scope.** Spacecraft operations, autonomous manoeuvres, claims about specific real satellites, safety-critical decisions, and live Indian catalogue data without retraining.

**Threshold.** High-risk class is log10(Pc) ≥ −6. This matches the ESA challenge scoring class. It is not an ISRO operational rule. Typical LEO reaction bands are 10⁻⁵ to 10⁻⁴.

**Split.** Grouped by event_id into train / validation / calibration / test. Calibration never sees test labels.

**Data.** The frozen artifacts use the labeled ESA Collision Avoidance Challenge training archive: 162,634 CDM rows, filtered to 8,293 events with a pre-T−48 message and a final update inside T−24. The official challenge test file is used only for schema compatibility because final labels are unavailable.

**Current result.** The selected conservative ensemble reduces held-out MAE from 5.080 for persistence to 3.053, but both score 0.167 on the ESA-style loss. PRISM therefore does not currently claim to beat persistence under the full acceptance rule.

**Uncertainty.** The displayed bands are bootstrap-model spread, not calibrated predictive intervals. On the held-out event split the nominal 50% band covers 27.8% of outcomes and the nominal 90% band covers 49.2%, so the interface labels these bands as model spread and abstains when the 90% band crosses the configured threshold.

**Robustness.** `metrics.json` exports results by object type, history length, miss distance, radial uncertainty, and snapshot age. Adding the anonymized `mission_id` does not improve ordinary hold-out MAE (2.845 with it versus 2.843 without it), so the production model excludes it. A stricter four-mission hold-out confirms that the learned regressor still misses the rare high-risk tail, while persistence remains stronger on ESA loss; this limits the defensible claim to lower aggregate MAE.

**Human control.** Forecasts are advisory. REVIEW REQUIRED means a person must look. The model never commands a manoeuvre.

See `ml/artifacts/metrics.json` for the frozen numbers.
